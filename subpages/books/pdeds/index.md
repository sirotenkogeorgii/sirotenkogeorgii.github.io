---
layout: default
title: Partial Differential Equations in Data Science
date: 2026-04-20
excerpt: Lecture notes on PDEs in data science: gradient flows, energy landscapes, existence, uniqueness, and stability, with a view toward applications in physics and machine learning.
tags:
  - pde
  - gradient-flows
  - analysis
  - machine-learning
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

# Partial Differential Equations in Data Science

Notes based on the Summer 2026 lecture course by Prof. Dr. Tim Laux.

## Problems

[Selected Problems](/subpages/books/pdeds/problems/)

## Chapter 1: Gradient Flows

### 1.1 Motivation

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(Importance of Gradient Flows)</span></p>

A large variety of dynamical problems are **gradient flows**, meaning they can be viewed as the *steepest descent* in an energy landscape. Such problems are ubiquitous in the physical world, and also in human-made systems: gradient flows are the workhorse of today's machine learning algorithms.

After an introduction in the finite-dimensional setting — which gives rise to systems of ordinary differential equations — this course builds up the general theory for gradient flows. We then address a selection of problems from physics and data science that can (almost) be put into this abstract framework. Along the way, we familiarize ourselves with basic themes of modern analysis.

</div>

### 1.2 Outline

Instead of directly defining gradient flows in a general setup, we first start with the simple **Euclidean case**. In this setting, a gradient flow is a system of ordinary differential equations: given an "energy" (or "entropy") $E: \mathbb{R}^N \to [0, \infty)$ and initial data $x_0 \in \mathbb{R}^N$, solve

$$
\begin{cases}
\dot{x}(t) = -\nabla E(x(t)) & \text{for } t > 0, \\
x(0) = x_0,
\end{cases} \tag{1.1}
$$

where $\dot{x} = \tfrac{dx}{dt}$. By classical ODE theory (Picard–Lindelöf / Cauchy–Lipschitz), there exists a unique solution whenever $\nabla E$ is Lipschitz, i.e., $E \in C^{1,1}(\mathbb{R}^N)$.

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/gradient_flow_trajectory.png' | relative_url }}" alt="Gradient-flow trajectories on an anisotropic quadratic energy landscape, all converging to the unique minimizer at the origin" loading="lazy">
  <figcaption>Gradient-flow trajectories on $E(x_1,x_2)=\frac{1}{2}(x_1^2+3x_2^2)$. Each curve solves $\dot{x}=-\nabla E$ from a different initial condition; all converge to the minimizer $x_*=0$. Trajectories cross level sets at right angles.</figcaption>
</figure>

**Energy dissipation.** Differentiating the energy along the trajectory,

$$
\frac{d}{dt} E(x(t)) = dE(x(t)).\dot{x}(t) = \langle \nabla E(x(t)), \dot{x}(t) \rangle = -|\dot{x}(t)|^2 \le 0. \tag{1.2}
$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On the dot in $dE(x(t)).\dot{x}(t)$)</span></p>

The dot "$.$" here is **not division** — it denotes *application of a linear map to a vector*. Reading it as $dE(x(t))/\dot{x}(t)$ or $\frac{dE(x(t))}{dx(t)} \cdot \dot{x}(t)$ would be a category error, because $dE(x(t))$ is itself a linear map, not a quantity one divides by.

* The object $dE(x(t))$ is the **differential** of $E$ at the point $x(t)$, i.e., the linear map

  $$dE(x(t)): \mathbb{R}^N \to \mathbb{R}, \qquad v \mapsto dE(x(t)).v.$$

* The expression $dE(x(t)).\dot{x}(t)$ is then this linear map *evaluated* on the velocity vector $\dot{x}(t) \in \mathbb{R}^N$. Some authors write $dE(x(t))[\dot{x}(t)]$ or $dE(x(t))(\dot{x}(t))$ for the same thing.

* This is exactly the chain rule: if $f(t) := E(x(t))$, then $f'(t) = dE(x(t)).\dot{x}(t)$.

In finite dimensions, one *could* equivalently write $\frac{\partial E}{\partial x}(x(t)) \cdot \dot{x}(t)$ (Jacobian row times velocity), and arithmetically the answer is the same. The reason the pedantic notation is preferred here is that the course will soon generalize to non-Euclidean settings where "$dE/dx$" no longer makes sense, while the differential $dE$ as a linear map continues to.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Differential vs. Gradient)</span></p>

The notation above is pedantic on purpose, distinguishing between the **differential** $dE$ and the **gradient** $\nabla E$. This distinction will be crucial in the non-Euclidean case later.

* The differential $dE(x)$ is the *linear map* that best approximates $E$ in a neighborhood of $x$.
* The gradient $\nabla E(x)$ is its *Riesz representative*:

  $$\langle \nabla E(x), v \rangle = dE(x).v \quad \text{for all } v \in \mathbb{R}^N.$$

</div>

In particular, the energy is **non-increasing** in time. Integrating (1.2) yields, for any $T > 0$,

$$
E(x(T)) + \int_0^T |\dot{x}(t)|^2 \, dt \le E(x(0)). \tag{1.3}
$$

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof that (1.2) $\Rightarrow$ (1.3)</summary>

Fix $T > 0$. Equation (1.2) states that, along the trajectory,

$$
\frac{d}{dt} E(x(t)) = -|\dot{x}(t)|^2 \quad \text{for all } t \in (0, T).
$$

Both sides are continuous in $t$ (assuming $x \in C^1$, which follows from $\nabla E$ being Lipschitz), so we may integrate from $0$ to $T$:

$$
\int_0^T \frac{d}{dt} E(x(t)) \, dt = -\int_0^T |\dot{x}(t)|^2 \, dt.
$$

By the **fundamental theorem of calculus**, the left-hand side equals $E(x(T)) - E(x(0))$, so

$$
E(x(T)) - E(x(0)) = -\int_0^T |\dot{x}(t)|^2 \, dt,
$$

or, rearranging,

$$
E(x(T)) + \int_0^T |\dot{x}(t)|^2 \, dt = E(x(0)). \tag{$\ast$}
$$

This is in fact an **equality**, which is strictly stronger than (1.3). The inequality form in (1.3) is stated because:

* it is the form that survives in non-smooth / weak settings, where $\dot{x}$ may only exist almost everywhere and $E \circ x$ may only be absolutely continuous (so that the FTC gives "$\le$" rather than "$=$"); and
* it is exactly the form we will recognize later as the **Energy–Dissipation Inequality (EDI)**, a cornerstone of the abstract theory of gradient flows in metric spaces.

In the smooth Euclidean setting treated here, ($\ast$) and (1.3) are equivalent, and either reading is fine. $\square$

</details>
</div>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/energy_dissipation.png' | relative_url }}" alt="Energy E(x(t)) decreases along the trajectory while the integral of squared velocity grows; their sum stays constant and equal to E(x_0)" loading="lazy">
  <figcaption>The identity $E(x(t))+\int_0^t |\dot{x}|^2\,ds = E(x_0)$ visualised. Energy (blue) drains away along the trajectory and is exactly recovered as accumulated dissipation (orange); the dashed line shows their sum, which is conserved.</figcaption>
</figure>

At the risk of stating the obvious: $-\nabla E(x(t))$ is the *direction of steepest descent* of the energy (or entropy) $E$. For all $v \in \mathbb{R}^N$ with $\|v\| = \|\nabla E(x(t))\|$,

$$
-\langle \nabla E(x(t)), \nabla E(x(t)) \rangle \le \langle v, \nabla E(x(t)) \rangle.
$$

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/steepest_descent.png' | relative_url }}" alt="Unit directions at a point colored by their inner product with the gradient; the minimum is attained at minus the normalized gradient" loading="lazy">
  <figcaption>At a fixed point $x_*$, the directional derivative $\langle v,\nabla E(x_*)\rangle$ varies as $\cos$ over the unit sphere (right). Among unit vectors, it is minimized exactly when $v=-\nabla E/|\nabla E|$ (left, green) and maximized at $v=+\nabla E/|\nabla E|$ (red).</figcaption>
</figure>

This is exactly what characterizes a gradient flow: **it is the steepest descent in an energy landscape**. Moreover, we expect that in the long-time limit $t \to \infty$, the trajectory $x(t)$ converges to a critical point, or a local (or even global!) minimizer of $E$.

**Typical questions** we will learn to appreciate and to answer (partially):

1. **Existence, uniqueness, and stability** beyond the Picard–Lindelöf / Cauchy–Lipschitz framework. The standard regularity requirement is too restrictive for most interesting gradient flows. The additional structure of the right-hand side of (1.1) allows us to develop tools tailored to these equations and therefore very robust.
2. **Long-term asymptotics** towards (local or global?) minimizers under suitable conditions.
3. **Convergence of gradient flows**: given a sequence of energy functionals $E_k$ (possibly defined on different spaces $X_k$), under which conditions do the corresponding gradient flows converge?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why study gradient flows?)</span></p>

Many ODEs and PDEs have a gradient flow structure, and the gradient flow framework provides powerful tools to study them. Furthermore, gradient flows arise in optimization — in particular in high dimensions $N \gg 1$ where it is too expensive to evaluate second derivatives of $E$, which is typically the case in modern machine learning problems.

</div>

In the first two sections, we will (in the Euclidean case with $E$ convex)

1. prove existence of solutions;
2. prove uniqueness and stability properties;
3. study the long-term behavior of solutions.

Later in the course, we will see examples of partial differential equations and free boundary problems that can be interpreted as gradient flows, touching on recent (and possibly current) research. Although the questions above are quite simple in the convex Euclidean setting (as we will see in the first few lectures), they become very subtle in other applications.

**Critical points and the Euler–Lagrange equation.** The long-time limit of a gradient flow, when it exists, is a *critical point* of $E$ — a point $x_*$ with $\nabla E(x_*) = 0$ (or $0 \in \partial E(x_*)$ in the non-smooth setting). For functionals defined on infinite-dimensional spaces of functions, the analogue of "$\nabla E = 0$" is a differential equation in its own right, called the **Euler–Lagrange equation**.

Concretely, consider a functional

$$
J[y] = \int_a^b L\big(x,\,y(x),\,y'(x)\big)\,dx
$$

acting on smooth curves $y:[a,b]\to\mathbb{R}$ with prescribed boundary values $y(a)=\alpha$, $y(b)=\beta$. The smooth function $L:\mathbb{R}^3\to\mathbb{R}$ is called the **Lagrangian**. Plugging in a variation $y_\varepsilon = y + \varepsilon\,\eta$ with $\eta(a)=\eta(b)=0$, expanding to first order, and integrating by parts on the $\eta'$-term gives

$$
\frac{d}{d\varepsilon}\bigg|_{\varepsilon=0} J[y_\varepsilon]
= \int_a^b \!\left[\frac{\partial L}{\partial y} - \frac{d}{dx}\frac{\partial L}{\partial y'}\right]\eta(x)\,dx.
$$

Demanding this vanish for every admissible $\eta$ and applying the fundamental lemma of the calculus of variations yields the **Euler–Lagrange equation**:

$$
\frac{\partial L}{\partial y} - \frac{d}{dx}\!\left(\frac{\partial L}{\partial y'}\right) = 0.
$$

This is a (generally nonlinear, generally second-order) ODE in $y$. Three canonical examples illustrate its scope:

* **Geodesics in the plane.** With $L(x,y,y') = \sqrt{1+(y')^2}$, the EL equation reduces to $y'' = 0$, recovering straight lines as the curves of shortest length.
* **Classical mechanics.** With $L(q,\dot q) = \frac{1}{2} m\,\lvert\dot q\rvert^2 - E(q)$ (kinetic minus potential energy), the EL equation reads $m\ddot q = -\nabla E(q)$ — exactly Newton's law without friction (cf. (1.5) below with $\lambda=0$). This is **Hamilton's principle of stationary action**: physical trajectories are precisely the stationary points of $\int L\,dt$.
* **Dirichlet energy.** For $J[u] = \frac{1}{2}\int_\Omega \lvert\nabla u\rvert^2\,dx$ on functions $u:\Omega\to\mathbb{R}$, the multivariable analogue of (1.4) is the **Laplace equation** $\Delta u = 0$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(EL equation vs. gradient flow)</span></p>

The Euler–Lagrange equation is a *static* condition — it picks out the critical points of a functional $J$. A **gradient flow of $J$** is the corresponding *dynamic* equation

$$
\partial_t u = -\nabla_{\!\mathcal H}\, J[u],
$$

where $\nabla_{\!\mathcal H}$ denotes the gradient with respect to a chosen Hilbert structure $\mathcal H$ on the space of admissible $u$'s. Its steady states, when they exist, are exactly the EL solutions of $J$.

Two viewpoints on the same picture:

* **Static (variational):** look for critical points of $J$ directly by solving the EL equation.
* **Dynamic (evolutionary):** start from any $u_0$, run the gradient flow, and let $t\to\infty$.

For instance, the $L^2$-gradient flow of the Dirichlet energy is the **heat equation** $\partial_t u = \Delta u$, and its stationary solutions are harmonic. We will return to this duality repeatedly when we lift the finite-dimensional theory of (1.1) into PDE settings.

</div>

### 1.3 Gradient Flow as Overdamped Limit

**Newtonian motivation.** Newton's law dictates that the trajectory $x: [0, T) \to \mathbb{R}^N$ (think of $N = 2$ or $N = 3$) of a particle with mass $m$ satisfies

$$
m \ddot{x} = \sum \text{Forces}. \tag{1.4}
$$

Suppose there are two forces: one coming from a potential energy $E$ and one from friction (with friction parameter $\lambda \ge 0$). We obtain the system

$$
m \ddot{x} = -\nabla E(x) - \lambda v, \tag{1.5}
$$

$$
v = \dot{x}. \tag{1.6}
$$

The **total energy** (potential + kinetic) satisfies

$$
\begin{aligned}
\frac{d}{dt}\!\left( E(x(t)) + \tfrac{1}{2} m \lvert v(t)\rvert^2 \right) &= \langle \nabla E(x(t)), \dot{x} \rangle + \langle m v(t), \dot{v}(t) \rangle \\
&= \langle \nabla E(x(t)), v(t) \rangle - \langle v(t), \nabla E(x(t)) \rangle - \lambda \lvert v(t)\rvert^2 \\
&= -\lambda \lvert v(t)\rvert^2.
\end{aligned}
$$

* For $\lambda = 0$: **energy is conserved**.
* For $\lambda > 0$: **energy is dissipated**.

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/total_energy_friction.png' | relative_url }}" alt="Total mechanical energy under Newton dynamics with friction: flat for lambda zero, monotonically decreasing for positive friction" loading="lazy">
  <figcaption>Total energy $E(x)+\frac{1}{2}m|v|^2$ along the Newton-with-friction dynamics for the harmonic potential $E(x)=\frac{1}{2}x^2$. For $\lambda=0$ it is conserved (oscillation between potential and kinetic); for $\lambda>0$ it strictly decreases at rate $-\lambda|v|^2$.</figcaption>
</figure>

**The overdamped limit $\lambda \to \infty$.** One can think of a heavy ball sinking into a glass of honey under the influence of gravity. Since we expect slow motion, we pass to a slow time scale:

$$
t' = \frac{1}{\lambda} t, \qquad x' = x, \qquad v' = \frac{dx'}{dt'} = \lambda v.
$$

Then the equation

$$
m \frac{dv}{dt} = -\nabla E(x(t)) - \lambda v(t)
$$

transforms into

$$
\frac{m}{\lambda} \frac{d v'}{dt'} = -\nabla E(x'(t')) - v'(t').
$$

Taking $\lambda \to \infty$ yields $0 = -\nabla E(x'(t')) - v'(t')$, i.e.,

$$
\frac{d x'}{d t'} = -\nabla E(x'(t')),
$$

which is precisely our gradient flow equation (1.1). In words: **the gradient flow is the overdamped limit of Newtonian mechanics with friction**.

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/overdamped_limit.png' | relative_url }}" alt="Newton trajectories: oscillating for small friction, smoothly descending for large friction; in the rescaled time they collapse onto the gradient flow as lambda grows" loading="lazy">
  <figcaption>Left: Newton-with-friction trajectories $\ddot{x}+\lambda\dot{x}+x=0$ in original time, transitioning from underdamped oscillation to overdamped descent as $\lambda$ grows. Right: the same trajectories plotted against the rescaled time $t'=t/\lambda$. As $\lambda\to\infty$ they collapse onto the gradient flow $x(t')=e^{-t'}$ (dashed).</figcaption>
</figure>

### 1.4 Existence

To build intuition for gradient flows, we first focus on the simple case of gradient flows in Euclidean space. Then a gradient flow simply describes a system of ordinary differential equations. We will see that, thanks to the special gradient-flow structure, we can go beyond the standard ODE theory and prove existence and uniqueness **with less regularity**. Under suitable assumptions, the results of this chapter can be generalized to arbitrary Hilbert spaces.

A natural assumption on $E$ is **convexity**. A convex function need not be differentiable everywhere, but we can always define its subdifferential.

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/convex_vs_nonconvex.png' | relative_url }}" alt="Convex versus non-convex 1D potentials with their gradient-flow trajectories: convex always converges to the unique minimizer, non-convex converges to whichever minimizer lies in the basin of the initial condition" loading="lazy">
  <figcaption>Why convexity matters. Left column: $E(x)=\frac{1}{2}x^2$ is convex, has a unique minimizer, and every gradient-flow trajectory converges to it. Right column: $E(x)=(x^2-1)^2$ has two minimizers; the limit of $x(t)$ depends on the basin of attraction of $x_0$.</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Subdifferential)</span></p>

For a convex function $E: \mathbb{R}^N \to [0, +\infty]$, the **subdifferential** at a point $x$ is

$$
\partial E(x) := \left\lbrace p \in \mathbb{R}^N \,:\, E(y) \ge E(x) + \langle p, y - x \rangle \ \text{ for all } y \in \mathbb{R}^N \right\rbrace. \tag{1.7}
$$

Elements of $\partial E(x)$ are called **subgradients** of $E$ at $x$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/subdifferential_kink.png' | relative_url }}" alt="The convex function E(x) = absolute value of x, shown together with its fan of supporting lines at the kink x = 0; subgradients have slope between minus one and one" loading="lazy">
  <figcaption>The subdifferential at a kink. For $E(x)=|x|$ the graph at $x=0$ admits a whole interval of supporting affine minorants (slopes in $[-1,1]$, blue/green); a slope outside this interval (red, $1.4$) fails the inequality on one side. Hence $\partial E(0)=[-1,+1]$.</figcaption>
</figure>

With the subdifferential in hand, we formulate the gradient flow as a **differential inclusion**:

$$
\begin{cases}
\dot{x}(t) \in -\partial E(x(t)) & \text{for } t > 0, \\
x(0) = x_0.
\end{cases} \tag{1.8}
$$

A few facts about the subdifferential are collected in the following exercise.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 1</span><span class="math-callout__name">(Convex analysis)</span></p>

Let $E: \mathbb{R}^N \to [0, +\infty]$ be convex, i.e.,

$$
E(\lambda x + (1-\lambda) y) \le \lambda E(x) + (1-\lambda) E(y) \quad \text{for all } x, y \in \mathbb{R}^N \text{ and } \lambda \in [0, 1]. \tag{1.9}
$$

Show the following:

1. If $E \in C^1$, then $\partial E(x) = \lbrace \nabla E(x) \rbrace$.
2. $E$ is differentiable at $x$ if and only if $\partial E(x)$ is a singleton.
3. The set $\partial E(x)$ is convex; it is nonempty whenever $E(x) < +\infty$.

</div>

**Minimizing movements.** We will prove existence of solutions via the so-called **minimizing movements** (also known as JKO / variational) scheme. Let $E: \mathbb{R}^N \to [0, \infty)$ be convex, let $x_0 \in \mathbb{R}^N$ be given, and let $h > 0$ denote a time-step size. For $\ell = 1, 2, 3, \dots$, define iteratively

$$
\chi_h^{(\ell)} := \arg \min_{x \in \mathbb{R}^N} \left\lbrace E(x) + \frac{1}{2h} \left| x - \chi_h^{(\ell-1)} \right|^2 \right\rbrace, \tag{1.10}
$$

and let

$$
x_h(t) := \chi_h^{(\ell)} \quad \text{for } t \in [(\ell-1)h, \ell h) \tag{1.11}
$$

be its piecewise-constant interpolation in time.

The **Euler–Lagrange equation** (TODO: what is it?) (necessary optimality condition) for (1.10) is

$$
\frac{\chi_h^{(\ell)} - \chi_h^{(\ell-1)}}{h} \in -\partial E(\chi_h^{(\ell)}). \tag{1.12}
$$

When $E$ is differentiable, this reduces to

$$
\frac{\chi_h^{(\ell)} - \chi_h^{(\ell-1)}}{h} = -\nabla E(\chi_h^{(\ell)}), \tag{1.13}
$$

which is precisely the **implicit Euler scheme** (TODO: what is it?) for the ODE (1.1).

With this construction in hand, we state the existence result. Its proof is more technical than the results in the next section and will be deferred.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Existence via minimizing movements)</span></p>

As $h \downarrow 0$, the discrete interpolants satisfy $x_h \to x$ locally uniformly, and the limit $x$ solves (1.8). In particular, there exists a solution to the differential inclusion (1.8).

</div>

### 1.5 Uniqueness and Stability

Thanks to the convexity of $E$, we also obtain a **uniqueness and stability** result: two solutions starting from possibly different initial conditions cannot drift apart — the distance between them is non-increasing in time.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2</span><span class="math-callout__name">(Uniqueness and Stability)</span></p>

Let $E$ be convex, and let $x_1, x_2$ be two solutions of $\dot{x} \in -\partial E(x)$. Then

$$
t \mapsto |x_1(t) - x_2(t)| \quad \text{is non-increasing.} \tag{1.14}
$$

In particular, the solution to (1.8) is **unique** (given an initial condition).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The proof is short: we only need to check that $\tfrac{d}{dt} \|x_1(t) - x_2(t)\| \le 0$, or — equivalently and more conveniently —

$$
\frac{d}{dt} \, \tfrac{1}{2} |x_1(t) - x_2(t)|^2 \le 0.
$$

*Proof of Theorem 2.* Set $f(t) := \tfrac{1}{2} \|x_1(t) - x_2(t)\|^2$. Then

$$
\frac{d}{dt} f = \langle x_1(t) - x_2(t), \, \dot{x}_1(t) - \dot{x}_2(t) \rangle. \tag{1.15}
$$

A standard fact from convex analysis (the $N$-dimensional version of the fact that the derivative of a convex function is non-decreasing) states that whenever $p_i \in \partial E(x_i(t))$ for $i = 1, 2$,

$$
\langle x_1(t) - x_2(t), \, p_1 - p_2 \rangle \ge 0.
$$

By assumption, $-\dot{x}\_i(t) \in \partial E(x_i(t))$, so we may pick $p_i$ with $\dot{x}\_i(t) = -p_i$. Substituting,

$$
\frac{d}{dt} f = \langle x_1(t) - x_2(t), \, -(p_1 - p_2) \rangle = -\langle x_1(t) - x_2(t), \, p_1 - p_2 \rangle \le 0,
$$

which proves the monotonicity. **Uniqueness** follows by taking equal initial conditions $x_1(0) = x_2(0)$, so that $f(0) = 0$, and hence $f \equiv 0$. $\square$

</details>
</div>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/contraction.png' | relative_url }}" alt="Two gradient-flow trajectories from nearby initial points, with their pairwise distance shown as a non-increasing function of time" loading="lazy">
  <figcaption>The contraction property. Two trajectories of the same gradient flow (left) are joined by grey segments at corresponding times; the segment lengths shrink monotonically. The right panel plots $|x_1(t)-x_2(t)|$ as a function of $t$, confirming the non-increasing behavior that yields uniqueness.</figcaption>
</figure>

