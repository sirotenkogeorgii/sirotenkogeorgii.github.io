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

# Partial Differential Equations in Data Science

Notes based on the Summer 2026 lecture course by Prof. Dr. Tim Laux.

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

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gradient Flow in Euclidean space)</span></p>

Instead of directly defining gradient flows in a general setup, we first start with the simple **Euclidean case**. In this setting, a **gradient flow** is a system of ordinary differential equations: given an "*energy*" (or "*entropy*") $E: \mathbb{R}^N \to [0, \infty)$ and initial data $x_0 \in \mathbb{R}^N$, solve

$$
\begin{cases}
\dot{x}(t) = -\nabla E(x(t)) & \text{for } t > 0, \\
x(0) = x_0,
\end{cases} \tag{1.1}
$$

where $\dot{x} = \tfrac{dx}{dt}$. 

By classical ODE theory (Picard–Lindelöf / Cauchy–Lipschitz), there exists a unique solution whenever $\nabla E$ is Lipschitz, i.e., $E \in C^{1,1}(\mathbb{R}^N)$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/gradient_flow_trajectory.png' | relative_url }}" alt="Gradient-flow trajectories on an anisotropic quadratic energy landscape, all converging to the unique minimizer at the origin" loading="lazy">
  <figcaption>Gradient-flow trajectories on $E(x_1,x_2)=\frac{1}{2}(x_1^2+3x_2^2)$. Each curve solves $\dot{x}=-\nabla E$ from a different initial condition; all converge to the minimizer $x_*=0$. Trajectories cross level sets at right angles.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Energy dissipation)</span></p>

Differentiating the energy along the trajectory,

$$
\frac{d}{dt} E(x(t)) = dE(x(t)).\dot{x}(t) = \langle \nabla E(x(t)), \dot{x}(t) \rangle = -|\dot{x}(t)|^2 \le 0. \tag{1.2}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On the dot in $dE(x(t)).\dot{x}(t)$)</span></p>

The dot "$.$" here is **not multiplication** — it denotes *application of a linear map to a vector*. Reading it as $dE(x(t))/\dot{x}(t)$ or $\frac{dE(x(t))}{dx(t)} \cdot \dot{x}(t)$ would be a category error, because $dE(x(t))$ is itself a linear map, not a quantity one multiplies by.

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

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Energy is non-decreasing in time in gradient flow)</span></p>

The energy is **non-increasing** in time. Integrating (1.2) yields, for any $T > 0$,

$$
E(x(T)) + \int_0^T |\dot{x}(t)|^2 \, dt \le E(x(0)). \tag{1.3}
$$

</div>

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

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Property</span><span class="math-callout__name">(Direction of steepest descent of the energy)</span></p>

$-\nabla E(x(t))$ is the **direction of steepest descent** of the energy (or entropy) $E$. For all $v \in \mathbb{R}^N$ with $\|v\| = \|\nabla E(x(t))\|$,

$$
-\langle \nabla E(x(t)), \nabla E(x(t)) \rangle \le \langle v, \nabla E(x(t)) \rangle.
$$

This is exactly what characterizes a gradient flow: **it is the steepest descent in an energy landscape**. Moreover, we expect that in the long-time limit $t \to \infty$, the trajectory $x(t)$ converges to a critical point, or a local (or even global!) minimizer of $E$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/steepest_descent.png' | relative_url }}" alt="Unit directions at a point colored by their inner product with the gradient; the minimum is attained at minus the normalized gradient" loading="lazy">
  <figcaption>At a fixed point $x_*$, the directional derivative $\langle v,\nabla E(x_*)\rangle$ varies as $\cos$ over the unit sphere (right). Among unit vectors, it is minimized exactly when $v=-\nabla E/|\nabla E|$ (left, green) and maximized at $v=+\nabla E/|\nabla E|$ (red).</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Typical Questions</span><span class="math-callout__name">(Scope of our interest in Gradient Flows)</span></p>

**Typical questions** we will learn to appreciate and to answer (partially):

1. **Existence, uniqueness, and stability** beyond the Picard–Lindelöf / Cauchy–Lipschitz framework. The standard regularity requirement is too restrictive for most interesting gradient flows. The additional structure of the right-hand side of (1.1) allows us to develop tools tailored to these equations and therefore very robust.
2. **Long-term asymptotics** towards (local or global?) minimizers under suitable conditions.
3. **Convergence of gradient flows**: given a sequence of energy functionals $E_k$ (possibly defined on different spaces $X_k$), under which conditions do the corresponding gradient flows converge?

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why study gradient flows?)</span></p>

Many ODEs and PDEs have a gradient flow structure, and the gradient flow framework provides powerful tools to study them. Furthermore, gradient flows arise in optimization — in particular in high dimensions $N \gg 1$ where it is too expensive to evaluate second derivatives of $E$, which is typically the case in modern machine learning problems.

</div>

**Critical points and the Euler–Lagrange equation.** The long-time limit of a gradient flow, when it exists, is a *critical point* of $E$ — a point $x_*$ with $\nabla E(x_\ast) = 0$ (or $0 \in \partial E(x_\ast)$ in the non-smooth setting). For functionals defined on infinite-dimensional spaces of functions, the analogue of "$\nabla E = 0$" is a differential equation in its own right, called the **Euler–Lagrange equation**.

**Why study it.** The Euler–Lagrange equation is the central object of the **calculus of variations**, one of the oldest branches of analysis (its founding problem, the *brachistochrone* of Johann Bernoulli, 1696, asks for the curve along which a bead slides between two given points in the shortest time). The same equation appears wherever a quantity is determined by minimisation:

* **Mechanics.** Hamilton's *principle of stationary action*: physical trajectories are stationary points of an action functional $\int L\,dt$.
* **Geometry.** Geodesics on a Riemannian manifold are stationary points of an arc-length functional; minimal surfaces are stationary points of an area functional.
* **Optics.** Fermat's *principle of least time*: light rays follow stationary points of an optical-path-length functional.
* **PDE / mathematical physics.** Many fundamental PDEs — Laplace, Poisson, the harmonic-map equation, the minimal-surface equation, the equations of nonlinear elasticity — are EL equations of natural energy functionals.
* **Optimisation.** Modern variational problems in image processing, optimal transport, machine learning are formulated as energy minimisation; their stationarity conditions are EL equations.

In each setting the EL equation is the *necessary first-order condition* for a function to be a stationary point of a functional. Below we formalize this in the simplest one-dimensional setting; the same idea extends to vector-valued curves, multivariate domains, and abstract Banach/Hilbert/manifold settings.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Functional, Lagrangian, admissible class)</span></p>

A **functional** is a real-valued map $J:\mathcal A\to\mathbb R$ defined on a set $\mathcal A$ of functions. The classical first-order one-dimensional case is

$$
J[y] = \int_a^b L\bigl(x,\,y(x),\,y'(x)\bigr)\,dx,
$$

where:

* $L:U\subseteq\mathbb R^3\to\mathbb R$ is a $C^2$ function called the **Lagrangian** (or *Lagrangian density*);
* the **admissible class** $\mathcal A$ consists of $C^1$ (or $C^2$) curves $y:[a,b]\to\mathbb R$ with prescribed boundary values $y(a)=\alpha,\ y(b)=\beta$;
* $J[y]$ is finite for all $y\in\mathcal A$.

The space of **admissible variations** is

$$
\mathcal V := \bigl\lbrace \eta\in C^1([a,b]):\eta(a)=\eta(b)=0\bigr\rbrace,
$$

so that $y+\varepsilon\eta\in\mathcal A$ whenever $y\in\mathcal A$ and $\eta\in\mathcal V$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stationary point of a functional)</span></p>

A curve $y\in\mathcal A$ is a **stationary point** (or **critical point**) of $J$ if

$$
\delta J[y;\eta]\;:=\;\left.\frac{d}{d\varepsilon}\right|_{\varepsilon=0}\!J[y+\varepsilon\eta]\;=\;0\qquad\text{for every }\eta\in\mathcal V.
$$

The quantity $\delta J[y;\eta]$ is called the **first variation** of $J$ at $y$ in the direction $\eta$.

In particular every (local) minimizer or maximizer of $J$ in $\mathcal A$ is a stationary point — but not every stationary point is an extremum (saddles also satisfy $\delta J=0$).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Euler–Lagrange equation)</span></p>

Let $L\in C^2(U)$ and let $y\in C^2([a,b])$ be a stationary point of

$$
J[y]=\int_a^b L\bigl(x,y(x),y'(x)\bigr)\,dx
$$

in the admissible class $\mathcal A$ with fixed endpoints. Then $y$ satisfies the **Euler–Lagrange equation**

$$
\frac{\partial L}{\partial y}\bigl(x,y(x),y'(x)\bigr) \;-\; \frac{d}{dx}\!\left(\frac{\partial L}{\partial y'}\bigl(x,y(x),y'(x)\bigr)\right) = 0\qquad\text{for all }x\in(a,b). \tag{EL}
$$

Together with the boundary conditions $y(a)=\alpha,\ y(b)=\beta$, this is a (generally nonlinear, generally second-order) two-point boundary-value problem for $y$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Derivation of (EL) — the variational argument</summary>

Fix $\eta\in\mathcal V$ (so $\eta(a)=\eta(b)=0$) and define

$$
\Phi(\varepsilon)\;:=\;J[y+\varepsilon\eta]\;=\;\int_a^b L\bigl(x,\,y(x)+\varepsilon\eta(x),\,y'(x)+\varepsilon\eta'(x)\bigr)\,dx.
$$

Since $L\in C^2$ and the integrand is smooth in $\varepsilon$, we may differentiate under the integral sign:

$$
\Phi'(0) \;=\; \int_a^b \left[\frac{\partial L}{\partial y}(x,y,y')\,\eta(x)\;+\;\frac{\partial L}{\partial y'}(x,y,y')\,\eta'(x)\right]dx.
$$

Apply integration by parts to the $\eta'$-term:

$$
\int_a^b \frac{\partial L}{\partial y'}\,\eta'(x)\,dx
\;=\;\left[\frac{\partial L}{\partial y'}\,\eta\right]_a^b\;-\;\int_a^b \frac{d}{dx}\!\left(\frac{\partial L}{\partial y'}\right)\eta(x)\,dx.
$$

The boundary term vanishes because $\eta(a)=\eta(b)=0$. Substituting back,

$$
\Phi'(0)\;=\;\int_a^b\!\left[\frac{\partial L}{\partial y}\;-\;\frac{d}{dx}\!\left(\frac{\partial L}{\partial y'}\right)\right]\eta(x)\,dx.
$$

By the definition of stationarity, this must vanish for every $\eta\in\mathcal V$. Apply the **fundamental lemma of the calculus of variations**:

> If $g\in C([a,b])$ and $\int_a^b g(x)\,\eta(x)\,dx=0$ for every $\eta\in C_c^\infty((a,b))$, then $g\equiv 0$ on $[a,b]$.

(Proof sketch: if $g(x_0)>0$ at some interior $x_0$, choose a non-negative bump $\eta\in C_c^\infty((a,b))$ supported in a small neighbourhood of $x_0$ where $g>0$; the integral is then strictly positive, a contradiction.)

Apply this lemma with

$$
g(x)\;:=\;\frac{\partial L}{\partial y}\bigl(x,y(x),y'(x)\bigr)-\frac{d}{dx}\!\left(\frac{\partial L}{\partial y'}\bigl(x,y(x),y'(x)\bigr)\right),
$$

which is continuous on $[a,b]$ since $y\in C^2$ and $L\in C^2$. The conclusion is $g\equiv 0$ on $(a,b)$, which is exactly (EL). $\square$

</details>
</div>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/el_first_variation.png' | relative_url }}" alt="Two-by-two grid: top row shows perturbations of a stationary curve and a non-stationary curve; bottom row shows the value of the arc-length functional along each one-parameter family. The stationary case has minimum at zero with horizontal tangent; the non-stationary case has slanted tangent at zero." loading="lazy">
  <figcaption>Stationarity, visualized for the arc-length functional $J[y]=\int_0^1\sqrt{1+y'(x)^2}\,dx$. <em>Left column</em>: $y_0(x)=x$ is the stationary point (the EL solution); the family $y_0+\varepsilon\eta$ with $\eta(x)=\sin(\pi x)$ has functional value $\Phi(\varepsilon)$ minimised at $\varepsilon=0$ with horizontal tangent — i.e. $\Phi'(0)=0$. <em>Right column</em>: $y_0(x)=x+\frac{1}{2}\sin(\pi x)$ is a non-stationary curve; the same family has $\Phi'(0)\neq 0$, signalling that perturbing in the right direction strictly decreases $J$. The Euler–Lagrange equation is the equation that distinguishes the left case from the right.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Generalizations)</span></p>

The same variational argument extends without conceptual change:

* **Vector-valued curves $y:[a,b]\to\mathbb R^N$.** Then (EL) becomes a *system* of $N$ equations $\partial_{y_i}L - \frac{d}{dx}\partial_{y'_i}L = 0$ for $i=1,\dots,N$.
* **Multivariate domains.** For functionals $J[u]=\int_\Omega L(x,u,\nabla u)\,dx$ on functions $u:\Omega\subseteq\mathbb R^d\to\mathbb R$, the integration-by-parts step produces a divergence and the EL equation becomes a PDE: $\partial_u L - \mathrm{div}\bigl(\nabla_p L\bigr)=0$, where $p:=\nabla u$.
* **Higher-order Lagrangians** ($L$ depending on $y''$, etc.) yield higher-order EL equations after multiple integrations by parts.
* **Free boundaries** (boundary values not prescribed) yield additional **natural boundary conditions** from the boundary terms that no longer vanish automatically.

In all of these the underlying logic — first variation $=0$ for all admissible perturbations, integrate by parts, apply the fundamental lemma — is the same.

</div>

**Canonical examples.**

* **Geodesics in the plane.** With $L(x,y,y') = \sqrt{1+(y')^2}$ (arc length), the EL equation reduces to $y'' = 0$, recovering straight lines as the curves of shortest length.

* **Brachistochrone.** A bead slides under gravity along a curve from $A=(0,0)$ to $B$ in the lower half-plane, starting at rest. By energy conservation its speed at height $y$ is $v=\sqrt{-2gy}$, so the transit time is

  $$T[y]=\int_0^{x_B}\frac{\sqrt{1+y'(x)^2}}{\sqrt{-2g\,y(x)}}\,dx.$$

  This is a Lagrangian of the form $L(x,y,y')$ with no explicit $x$-dependence, so the EL equation has a first integral (Beltrami's identity). Solving it gives a **cycloid** — the curve traced by a point on a rolling circle — as the unique brachistochrone. The figure below shows the cycloid winning a head-to-head against three other reasonable candidate curves between the same endpoints.

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/el_brachistochrone.png' | relative_url }}" alt="Four candidate curves between two points A and B with gravity: a straight line, a parabola, a cubic dip, and a cycloid; each labeled with the bead's transit time. The cycloid has the smallest transit time." loading="lazy">
  <figcaption>Brachistochrone: bead from $A=(0,0)$ to $B=(1,-1)$ under gravity ($g=1$). For each candidate curve the transit time $T[y]=\int\sqrt{1+y'^2}/\sqrt{-2gy}\,dx$ is shown. The cycloid (the EL solution) wins ($T\approx 1.83$) — narrowly over the parabola and cubic dip, decisively over the straight line. This is the founding problem of the calculus of variations (Bernoulli, 1696).</figcaption>
</figure>

* **Classical mechanics.** With $L(q,\dot q) = \frac{1}{2} m\,\lvert\dot q\rvert^2 - E(q)$ (kinetic minus potential energy), the EL equation reads $m\ddot q = -\nabla E(q)$ — exactly Newton's law without friction (cf. (1.5) below with $\lambda=0$). This is **Hamilton's principle of stationary action**: physical trajectories are precisely the stationary points of $\int L\,dt$. The figure below illustrates this for a 1D free particle thrown up under gravity with prescribed endpoints.

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/el_hamilton_action.png' | relative_url }}" alt="Two panels: left shows the physical parabolic trajectory of a particle under gravity together with several perturbed trajectories sharing the same endpoints; right shows the action as a function of the perturbation amplitude, with horizontal tangent at zero indicating the physical trajectory is a stationary point of the action." loading="lazy">
  <figcaption>Hamilton's principle for a particle in 1D under gravity, $L=\frac{1}{2}\dot q^2-q$, fixed endpoints $q(0)=q(1)=0$. <em>Left</em>: the physical trajectory is the parabola $q_*(t)=\frac{1}{2}t(1-t)$ (red); other admissible trajectories sharing the same endpoints differ from $q_*$ by $\alpha\sin(\pi t)$. <em>Right</em>: the action $S[q_\alpha]=\int_0^1(\frac{1}{2}\dot q^2-q)\,dt$ as a function of $\alpha$. The graph has horizontal tangent at $\alpha=0$, i.e. $\frac{dS}{d\alpha}|_{0}=0$, which is exactly the EL equation $\ddot q_*=-1$.</figcaption>
</figure>

* **Dirichlet energy.** For $J[u] = \frac{1}{2}\int_\Omega \lvert\nabla u\rvert^2\,dx$ on functions $u:\Omega\to\mathbb{R}$, the multivariate EL equation is the **Laplace equation** $\Delta u = 0$.

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why $-\nabla E$ instead of "force"? Energy and force, related)</span></p>

We are not replacing forces with energies — we are using the **energy form of a conservative force**. Equation (1.5) is just Newton's law $m\ddot x = F_{\text{total}}$ with the force split into two pieces: the conservative force $F_{\text{pot}} = -\nabla E$ from the potential, and the dissipative force $F_{\text{fric}} = -\lambda v$ from friction.

**Conservative forces and the relation $F = -\nabla V$.** A force field $F:\mathbb R^N\to\mathbb R^N$ is **conservative** if any of the following equivalent conditions holds:

1. **Path-independent work.** The work $\int_\gamma F\cdot d\ell$ along a path $\gamma$ from $a$ to $b$ depends only on $a$ and $b$, not on the path.
2. **Zero work around any closed loop.** $\oint F\cdot d\ell = 0$ for every closed loop.
3. **Existence of a potential.** There exists a scalar function $V$ (called the **potential energy**) with

   $$F(x) \;=\; -\nabla V(x).$$

These are the same statement. Given (1), define $V(x) := -\int_{x_0}^{x} F\cdot d\ell$ along *any* path from a fixed reference point — well-defined by path-independence; then $F=-\nabla V$ by the FTC.

The minus sign is conventional and makes the **force point in the direction of decreasing potential energy**: a ball rolls *down* the energy landscape; equilibria are critical points of $V$; stable equilibria are minima.

**Three canonical examples:**

| Force | Potential | Verification |
|---|---|---|
| Uniform gravity $F = -mg\,\hat z$ | $V(z) = mgz$ | $-\partial_z V = -mg$ ✓ |
| Linear spring $F = -kx$ | $V(x) = \tfrac{1}{2}kx^2$ | $-V'(x) = -kx$ ✓ |
| Coulomb / Newtonian gravity $F(r) = -\dfrac{C}{r^2}\hat r$ | $V(r) = -\dfrac{C}{r}$ | $-\partial_r V = -C/r^2$ ✓ |

**Why friction is a separate term.** Not every force admits a potential. Friction *always* opposes motion, so the work $\oint F_{\text{fric}}\cdot d\ell$ around a closed loop is *negative*, not zero — condition (2) fails. There is no scalar function $V$ with $F_{\text{fric}}=-\nabla V$. That is why $-\lambda v$ appears in (1.5) as a separate term rather than being folded into the gradient of some energy.

A useful classification:

| Force type | Form | Energy behavior |
|---|---|---|
| Conservative | $F=-\nabla V(x)$ | $T+V$ conserved |
| Linear friction (dissipative) | $F=-\lambda v$ | $T+V$ decreases at rate $-\lambda \lvert v\rvert^2$ |
| Magnetic Lorentz | $F=qv\times B$ | $\perp v$; does no work; $T$ alone conserved |
| External driving | $F=F(t)$ | $T+V$ changes by external work $\int F\cdot v\,dt$ |

**Why the energy form is what we want.** Three reasons the notes prefer $-\nabla E$ to a generic "$F$":

1. **It surfaces the energy structure.** The total-energy dissipation calculation (just below) reads $\frac{d}{dt}(E+\frac{1}{2}m\lvert v\rvert^2)=-\lambda\lvert v\rvert^2$, which depends precisely on the force being a gradient. Writing $F=-\nabla E$ from the start makes the calculation transparent.
2. **It makes the overdamped limit produce the gradient flow.** As $\lambda\to\infty$, the inertia $m\ddot x$ becomes negligible and the equation reduces to $0 = -\nabla E - \lambda v$, i.e. $\dot x = -\frac{1}{\lambda}\nabla E$ — after rescaling time, this is exactly the gradient flow (1.1). The derivation only works when the conservative force is the gradient of an energy.
3. **It is the right form for generalisation.** In Hilbert spaces and metric spaces "force" has no canonical meaning, but **energy** does — energies are scalar functionals one can define on any space. The whole abstract gradient-flow framework (Otto / Wasserstein, JKO) is built on the energy side of this equivalence, not the force side.

</div>

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

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Gradient Flow via Newtonain mechanics with friction)</span></p>

The gradient flow is the overdamped limit of Newtonian mechanics with friction, where the friction parameter $\lambda \geq 0$ goes to infinity.

</div>

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Is the subdifferential defined for convex functions only?)</span></p>

The definition (1.7) is meaningful only when $E$ is convex. The defining inequality $E(y)\ge E(x)+\langle p,y-x\rangle$ for **all** $y$ asks for a *global* supporting affine minorant tangent at $x$, and asking that to hold globally implicitly demands convexity. For a non-convex $E$ the set (1.7) is typically empty everywhere — there is no global affine minorant tangent at $x$.

For non-convex functions, several *generalizations* called "subdifferential" exist. They all collapse to the convex subdifferential when $E$ is convex, and to $\lbrace\nabla E(x)\rbrace$ when $E$ is smooth at $x$.

* **Fréchet (regular) subdifferential.**

  $$
  \partial^F E(x)=\left\lbrace p:\liminf_{y\to x}\frac{E(y)-E(x)-\langle p,y-x\rangle}{\|y-x\|}\ge 0\right\rbrace.
  $$

  A *local* one-sided first-order condition; does not require convexity.

* **Proximal subdifferential.**

  $$
  \partial^P E(x)=\left\lbrace p:\exists\,\sigma,\delta>0\ \text{ s.t. } E(y)\ge E(x)+\langle p,y-x\rangle-\sigma\|y-x\|^2\ \forall \|y-x\|<\delta\right\rbrace.
  $$

  Like Fréchet, with a quadratic correction.

* **Limiting (Mordukhovich) subdifferential.** $\partial^L E(x)$ is the closure of $\partial^F E$ under sequential limits $x_k\to x,\ E(x_k)\to E(x),\ p_k\in\partial^F E(x_k)$. Has the best calculus rules in non-convex variational analysis.

* **Clarke subdifferential.** For $E$ locally Lipschitz, $\partial^C E(x)$ is the convex hull of all limits $\lim\nabla E(x_k)$ at points $x_k\to x$ where $E$ is differentiable (Rademacher).

The hierarchy of inclusions is

$$
\partial^P E(x)\;\subseteq\;\partial^F E(x)\;\subseteq\;\partial^L E(x)\;\subseteq\;\partial^C E(x).
$$

**Why these notes use only the convex one.** The §1.5 contraction proof is driven by the *monotonicity* property of the convex subdifferential,

$$
p_i\in\partial E(x_i)\;\Longrightarrow\;\langle p_1-p_2,\,x_1-x_2\rangle\ge 0,
$$

and the long-time asymptotics rely on the fact that $0\in\partial E(x)$ is *equivalent* (not just necessary) to $x$ being a *global* minimizer. Both properties hold **only** for convex $E$. The non-convex generalizations above lose them: $0\in\partial^F E(x)$ is necessary but not sufficient for a local minimizer, and the operator $-\partial^F E$ is not monotone, so the contraction argument fails. This is why the course chooses convexity as a structural hypothesis — it is the regularity replacement that keeps the gradient-flow theory clean.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gradient Flow as a Differential Inclusion)</span></p>

With the subdifferential in hand, we formulate the gradient flow as a **differential inclusion**:

$$
\begin{cases}
\dot{x}(t) \in -\partial E(x(t)) & \text{for } t > 0, \\
x(0) = x_0.
\end{cases} \tag{1.8}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Subdifferential)</span></p>

Let $E: \mathbb{R}^N \to [0, +\infty]$ be convex, i.e.,

$$
E(\lambda x + (1-\lambda) y) \le \lambda E(x) + (1-\lambda) E(y) \quad \text{for all } x, y \in \mathbb{R}^N \text{ and } \lambda \in [0, 1]. \tag{1.9}
$$

1. If $E \in C^1$, then $\partial E(x) = \lbrace \nabla E(x) \rbrace$.
2. $E$ is differentiable at $x$ $\iff$ $\partial E(x)$ is a singleton.
3. The set $\partial E(x)$ is convex; it is nonempty whenever $E(x) < +\infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Minimizing movements)</span></p>

We will prove existence of solutions via the so-called **minimizing movements** (also known as JKO / variational) scheme. Minimizing movements, introduced by Ennio De Giorgi, refers to a variational approach for defining **gradient flows of non-smooth energies**, often approximating solutions by discretizing time and finding local minima at each step. It links to Euler schemes to describe steep descent for nonlinear diffusion, geometric evolution equations like mean curvature flow, and is used to study viscosity solutions. 

Let $E: \mathbb{R}^N \to [0, \infty)$ be convex, let $x_0 \in \mathbb{R}^N$ be given, and let $h > 0$ denote a time-step size. For $\ell = 1, 2, 3, \dots$, define iteratively

$$
\chi_h^{(\ell)} := \arg \min_{x \in \mathbb{R}^N} \left\lbrace E(x) + \frac{1}{2h} \left| x - \chi_h^{(\ell-1)} \right|^2 \right\rbrace, \tag{1.10}
$$

and let

$$
x_h(t) := \chi_h^{(\ell)} \quad \text{for } t \in [(\ell-1)h, \ell h) \tag{1.11}
$$

be its piecewise-constant interpolation in time.

</div>

The **Euler–Lagrange equation** for (1.10) — i.e. the *first-order optimality condition* for $\chi_h^{(\ell)}$ to be a minimizer of the bracketed functional, in the same variational sense as in §1.2 — is

$$
\frac{\chi_h^{(\ell)} - \chi_h^{(\ell-1)}}{h} \in -\partial E(\chi_h^{(\ell)}). \tag{1.12}
$$

When $E$ is differentiable, this reduces to

$$
\frac{\chi_h^{(\ell)} - \chi_h^{(\ell-1)}}{h} = -\nabla E(\chi_h^{(\ell)}), \tag{1.13}
$$

which is precisely the **implicit Euler scheme** for the ODE (1.1):

$$
\chi_h^{(\ell)}= \chi_h^{(\ell-1)} -h\nabla E(\chi_h^{(\ell)}).
$$

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Why is (1.12) called an Euler–Lagrange equation, and how is it the first-order optimality condition?</summary>

**What is being minimized.** Equation (1.10) defines $\chi_h^{(\ell)}$ as the minimizer of

$$
F(x) \;:=\; E(x) \;+\; \tfrac{1}{2h}\bigl|x-\chi_h^{(\ell-1)}\bigr|^2.
$$

So the question "what equation does $\chi_h^{(\ell)}$ satisfy?" is the same as "what does it mean for $\chi_h^{(\ell)}$ to be a minimizer of $F$?"

**First-order optimality, in plain calculus.** If $F$ is differentiable, the very first thing one learns about minimizers is the **first-order optimality condition**:

$$
x^\ast \text{ is a (local) minimizer of } F \;\Longrightarrow\; \nabla F(x^\ast)=0.
$$

It is called *first-order* because it involves only the first derivative (no Hessian), and it is *necessary, not sufficient* — every maximizer and every saddle satisfies it too. For convex but possibly non-smooth $E$, the analogue is

$$
0 \in \partial F(x^\ast),
$$

which is *necessary and sufficient* once $F$ is convex.

**Apply it to (1.10).** Decompose $F = E + Q$ where $Q(x):=\tfrac{1}{2h}\|x-\chi_h^{(\ell-1)}\|^2$. Then

* $Q$ is smooth with $\nabla Q(x) = \tfrac{1}{h}(x-\chi_h^{(\ell-1)})$,
* $E$ is convex with subdifferential $\partial E(x)$,
* by the sum rule for subdifferentials, $\partial F(x) = \partial E(x) + \nabla Q(x)$.

The first-order optimality condition $0 \in \partial F(\chi_h^{(\ell)})$ becomes

$$
0 \;\in\; \partial E\bigl(\chi_h^{(\ell)}\bigr) \;+\; \tfrac{1}{h}\bigl(\chi_h^{(\ell)}-\chi_h^{(\ell-1)}\bigr),
$$

and rearranging gives (1.12) exactly. If $E\in C^1$, $\partial E=\lbrace\nabla E\rbrace$ and the inclusion becomes the equality (1.13).

**Why call this an Euler–Lagrange equation?** In §1.2 the EL equation came from a recipe: take the functional $J$, compute the first variation $\delta J[y;\eta] = \tfrac{d}{d\varepsilon}\|\_{\varepsilon=0} J[y+\varepsilon\eta]$, set it to zero for every admissible $\eta$. That is literally the first-order optimality condition, written in function-space language: differentiate the objective along every admissible perturbation, demand stationarity. In §1.2 the integration-by-parts step gave the classical PDE/ODE form (EL); in (1.10) one does not need integration by parts because the variable is just $x\in\mathbb R^N$, but **the underlying logic is identical**:

| §1.2 (function space) | (1.10) (finite-dim, non-smooth) |
|---|---|
| Variable: $y\in\mathcal A$ | Variable: $x\in\mathbb R^N$ |
| Admissible perturbations: $\eta\in\mathcal V$ | Admissible perturbations: any $v\in\mathbb R^N$ |
| Stationarity: $\delta J[y;\eta]=0\ \forall\eta$ | Stationarity: $\nabla F(x)=0$ (or $0\in\partial F(x)$) |
| ⇓ integration by parts | ⇓ algebra (sum rule for $\partial$) |
| Classical EL equation (PDE/ODE) | Equation (1.12) |

**Step-for-step derivation in the smooth case.** One can derive (1.12) by mimicking §1.2 verbatim. Set $\Phi(\varepsilon):=F(\chi_h^{(\ell)}+\varepsilon v)$ for arbitrary $v\in\mathbb R^N$. Stationarity demands $\Phi'(0)=0$:

$$
\Phi'(0) \;=\; \bigl\langle \nabla E(\chi_h^{(\ell)}),\,v\bigr\rangle + \tfrac{1}{h}\bigl\langle \chi_h^{(\ell)}-\chi_h^{(\ell-1)},\,v\bigr\rangle \;=\; 0 \quad\forall v\in\mathbb R^N,
$$

which forces $\nabla E(\chi_h^{(\ell)})+\tfrac{1}{h}(\chi_h^{(\ell)}-\chi_h^{(\ell-1)})=0$ — exactly (1.13). Same recipe as §1.2: vary in every admissible direction, set the first variation to zero. The non-smooth (1.12) is the same statement read through the subdifferential.

**Summary.** "Euler–Lagrange equation" and "first-order optimality condition" are two names for the same idea: the equation that says the first variation of the objective vanishes in every admissible direction. In §1.2 the objective is a functional and the equation is a PDE/ODE; in (1.10) the objective is a function on $\mathbb R^N$ and the equation is (1.12), which in the smooth case collapses to "gradient = zero."

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Why is $\nabla E$ evaluated at the new point $\chi_h^{(\ell)}$ and not at the starting point $\chi_h^{(\ell-1)}$?</summary>

A natural intuition says: "for the optimal step from $\chi_h^{(\ell-1)}$, the next point should move in the direction of $-\nabla E(\chi_h^{(\ell-1)})$ — the steepest descent direction *at the current point*." That intuition is correct — but it is the **explicit Euler scheme**, not (1.13). (1.10) defines the **implicit Euler scheme**. The two are *different algorithms*, and the reason $\nabla E$ ends up at $\chi_h^{(\ell)}$ rather than at $\chi_h^{(\ell-1)}$ is that (1.10) asks a different question.

**Two competing recipes for "next point."**

* **Explicit Euler (the natural intuition).** "I am at $\chi_h^{(\ell-1)}$. The steepest descent direction *here* is $-\nabla E(\chi_h^{(\ell-1)})$. Step a distance $h$ in that direction":

  $$
  \chi_h^{(\ell)} \;=\; \chi_h^{(\ell-1)} - h\,\nabla E(\chi_h^{(\ell-1)}). \tag{$\star$}
  $$

  This is gradient descent. It is **not** what (1.10) defines.

* **Implicit Euler / JKO ((1.10)).** Among all candidate points $x$, pick the one that minimizes the auxiliary function

  $$
  F(x) \;=\; E(x) \;+\; \tfrac{1}{2h}\bigl|x-\chi_h^{(\ell-1)}\bigr|^2.
  $$

  This is not "take a step from where I am" — it is "find the bottom of $F$." And the gradient of $F$ that we set to zero is evaluated **at the candidate point being tested**, not at the anchor $\chi_h^{(\ell-1)}$.

The difference is the same as between "walk one step in this direction" and "find the equilibrium of a force field." The first names a recipe; the second names a fixed-point problem.

**Spring picture.** Think of $\chi_h^{(\ell-1)}$ as a fixed peg in the ground. Attach one end of a spring (stiffness $1/h$) to the peg; attach the other end to a ball that lives in the potential energy landscape $E$. Two forces act on the ball at any candidate position $x$:

* spring force, pulling the ball toward the peg: $-\tfrac{1}{h}(x - \chi_h^{(\ell-1)})$,
* energy force, pushing the ball downhill: $-\nabla E(x)$.

The ball comes to rest where these forces **cancel**. Crucially, both are evaluated **at the ball's resting position**, because that is where the ball actually sits:

$$
-\nabla E(\chi_h^{(\ell)}) \;-\; \tfrac{1}{h}(\chi_h^{(\ell)} - \chi_h^{(\ell-1)}) \;=\; 0,
$$

which rearranges to (1.13). The peg location $\chi_h^{(\ell-1)}$ enters only through the spring's anchor point; it never enters as a place where $\nabla E$ is evaluated. If you were to evaluate $\nabla E$ at the peg instead, you would be solving a different problem — namely, "where does the ball sit if the energy force is constant, equal to its value at the peg?" — and that answer is exactly $(\star)$.

**Mathematical statement: why the first-order condition lives at $\chi_h^{(\ell)}$.** (1.10) asks for the minimizer of $F(x)$. The first-order condition is $\nabla F(x^\ast) = 0$, i.e. *the gradient of $F$ vanishes at the minimizer*. Compute:

$$
\nabla F(x) \;=\; \nabla E(x) \;+\; \tfrac{1}{h}(x - \chi_h^{(\ell-1)}).
$$

Setting this to zero at $x = \chi_h^{(\ell)}$:

$$
\nabla E(\chi_h^{(\ell)}) \;+\; \tfrac{1}{h}(\chi_h^{(\ell)} - \chi_h^{(\ell-1)}) \;=\; 0.
$$

The point $\chi_h^{(\ell-1)}$ is a *frozen parameter* of $F$, not a variable. So when we differentiate $F$ in $x$ and evaluate at the minimizer, the only place $\chi_h^{(\ell-1)}$ shows up is inside the spring term. The $\nabla E$ slot is unconditionally evaluated at the variable $x$, which at the minimum equals $\chi_h^{(\ell)}$.

**A 1D example by hand.** Let $E(x)=\tfrac12 x^2$ (so $\nabla E(x)=x$) and $\chi_h^{(\ell-1)}=1$.

* *Explicit:* $\chi_h^{(\ell)} = 1 - h\cdot 1 = 1-h$.
* *Implicit:* solve $x + (x-1)/h = 0 \;\Rightarrow\; \chi_h^{(\ell)} = \tfrac{1}{1+h}$.

Compare with the exact gradient flow $\dot x=-x$, $x(0)=1$, whose value at time $h$ is $e^{-h}$:

| $h$ | explicit $1-h$ | implicit $1/(1+h)$ | exact $e^{-h}$ |
|---|---|---|---|
| $0.1$ | $0.900$ | $0.909$ | $0.905$ |
| $1$ | $0$ | $0.5$ | $0.368$ |
| $10$ | $-9$ | $0.091$ | $4.5\!\times\!10^{-5}$ |
| $100$ | $-99$ | $0.0099$ | $\approx 0$ |

For small $h$ both schemes agree, and both approximate the true flow. For large $h$ the explicit scheme **overshoots wildly** — it shoots to $-9$ when the true trajectory is monotonically decreasing toward $0$ — while the implicit scheme stays well-behaved. This is the famous *unconditional stability* of implicit Euler for gradient flows of convex energies.

**Reconciliation.** Both intuitions are correct *for their own algorithm*. As $h \to 0$, both ($\star$) and (1.13) converge to the same continuous flow $\dot x = -\nabla E(x)$, so in the limit they agree. For finite $h$, however, they are genuinely different schemes:

* The *explicit* scheme is what you get if you literally "take a step in the steepest descent direction at the current point."
* The *implicit / JKO* scheme is what you get if you instead **solve a small minimization problem at each step**, where the proximity to the previous point is penalized but the new point is otherwise allowed to settle wherever $F$ has its minimum.

(1.10) defines the second one. That is why $\nabla E$ is evaluated at $\chi_h^{(\ell)}$ — the new point is found by *solving an equation*, not by *stepping from the old point*. The reason the chapter prefers the implicit (JKO) version, even though it is computationally more expensive, is that the variational structure "$\chi_h^{(\ell)}$ is the *minimizer* of $E + \tfrac{1}{2h}\|\cdot - \chi_h^{(\ell-1)}\|^2$" is what generalizes to non-smooth $E$, to infinite-dimensional spaces, and (later in the chapter) to metric spaces where there is no "$\nabla E$" at all. The price paid for that generality is that the gradient lives at the new point.

</details>
</div>

With this construction in hand, we state the existence result. Its proof is more technical than the results in the next section and will be deferred.

<figure>
  <div class="mm-viz">
    <style>
      .mm-viz {
        --color-text-primary: #1a1a1a;
        --color-text-secondary: #4a4a4a;
        --color-text-tertiary: #777777;
        --color-background-primary: #ffffff;
        --color-background-secondary: #f3f4f8;
        --color-border-primary: #aaaaaa;
        --color-border-secondary: #c8c8c8;
        --color-border-tertiary: #dddddd;
        --border-radius-md: 6px;
        --font-sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      }
      .mm-viz .mm-controls { display: grid; gap: 14px; margin-top: 1rem; }
      .mm-viz .row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
      .mm-viz .row > label.lbl { font-size: 13px; color: var(--color-text-secondary); min-width: 92px; }
      .mm-viz .row > input[type="range"] { flex: 1; min-width: 160px; max-width: 320px; }
      .mm-viz .val { font-size: 13px; min-width: 36px; font-variant-numeric: tabular-nums; color: var(--color-text-primary); text-align: right; }
      .mm-viz .pill-group { display: inline-flex; gap: 4px; flex-wrap: wrap; }
      .mm-viz .pill { font-size: 12px; padding: 5px 10px; border: 0.5px solid var(--color-border-tertiary); border-radius: var(--border-radius-md); background: transparent; cursor: pointer; color: var(--color-text-primary); font-family: var(--font-sans); }
      .mm-viz .pill.active { background: var(--color-background-secondary); border-color: var(--color-border-primary); }
      .mm-viz .toggles { display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px; color: var(--color-text-secondary); }
      .mm-viz .toggles label { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }
      .mm-viz .info-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 0.25rem; }
      .mm-viz .info-cell { background: var(--color-background-secondary); border-radius: var(--border-radius-md); padding: 10px 14px; }
      .mm-viz .info-label { font-size: 12px; color: var(--color-text-secondary); }
      .mm-viz .info-val { font-size: 16px; font-weight: 500; font-variant-numeric: tabular-nums; margin-top: 2px; color: var(--color-text-primary); }
      .mm-viz .hint { font-size: 12px; color: var(--color-text-tertiary); }
      .mm-viz .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); border: 0; }
      .mm-viz #mm-plot { user-select: none; touch-action: none; }
    </style>

    <svg id="mm-plot" width="100%" viewBox="0 0 680 360" style="display: block; cursor: ew-resize;" role="img">
      <title>One-step minimizing-movements construction in 1D</title>
      <desc>Plot of energy E(x), parabola pull, and their sum F(x); the next iterate is the minimum of F.</desc>
      <defs>
        <clipPath id="mm-plot-clip"><rect x="30" y="20" width="480" height="300"/></clipPath>
        <marker id="mm-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </marker>
      </defs>
      <g id="mm-grid"></g>
      <g id="mm-curve-e" clip-path="url(#mm-plot-clip)"></g>
      <g id="mm-curve-p" clip-path="url(#mm-plot-clip)"></g>
      <g id="mm-curve-f" clip-path="url(#mm-plot-clip)"></g>
      <g id="mm-markers" clip-path="url(#mm-plot-clip)"></g>
      <g id="mm-legend"></g>
    </svg>

    <div class="mm-controls">
      <div class="row">
        <label class="lbl" for="mm-h-slider">Time step h</label>
        <input id="mm-h-slider" type="range" min="0.05" max="1.5" step="0.05" value="0.30">
        <span class="val" id="mm-h-val">0.30</span>
      </div>
      <div class="row">
        <label class="lbl">Energy E(x)</label>
        <div class="pill-group" id="mm-energy-group">
          <button class="pill active" data-energy="quad">½ x²</button>
          <button class="pill" data-energy="quartic">x⁴ / 8</button>
          <button class="pill" data-energy="abs">|x|</button>
        </div>
      </div>
      <div class="toggles">
        <label><input type="checkbox" id="mm-t-e" checked> Energy E(x)</label>
        <label><input type="checkbox" id="mm-t-p" checked> Parabola pull</label>
        <label><input type="checkbox" id="mm-t-f" checked> Combined F(x)</label>
      </div>
      <div class="hint">Drag along the x-axis to move χ<sub>ℓ-1</sub>; the next iterate χ<sub>ℓ</sub> always sits at the minimum of F</div>
      <div class="info-row">
        <div class="info-cell">
          <div class="info-label">Previous χ<sub>ℓ-1</sub></div>
          <div class="info-val" id="mm-i-prev">1.50</div>
        </div>
        <div class="info-cell">
          <div class="info-label">Next χ<sub>ℓ</sub></div>
          <div class="info-val" id="mm-i-next">1.15</div>
        </div>
        <div class="info-cell">
          <div class="info-label">Step |χ<sub>ℓ</sub> − χ<sub>ℓ-1</sub>|</div>
          <div class="info-val" id="mm-i-step">0.35</div>
        </div>
      </div>
    </div>

    <script>
    (function() {
      const PLOT_X = 30, PLOT_Y = 20, PLOT_W = 480, PLOT_H = 300;
      const X_MIN = -3, X_MAX = 3, Y_MIN = -1, Y_MAX = 5;
      const SX = PLOT_W / (X_MAX - X_MIN);
      const SY = PLOT_H / (Y_MAX - Y_MIN);
      const NS = 'http://www.w3.org/2000/svg';
      const m2sX = x => PLOT_X + (x - X_MIN) * SX;
      const m2sY = y => PLOT_Y + (Y_MAX - y) * SY;
      const s2mX = sx => (sx - PLOT_X) / SX + X_MIN;

      const COLORS = {
        e: '#888780', parab: '#BA7517', f: '#7F77DD',
        prev: '#185FA5', next: '#D85A30', arrow: '#444441'
      };

      const energies = {
        quad: { E: x => 0.5*x*x, prox: (y,h) => y/(1+h) },
        quartic: {
          E: x => Math.pow(x,4)/8,
          prox: (y,h) => {
            let x = y;
            for (let i = 0; i < 25; i++) {
              const f = (h/2)*x*x*x + x - y;
              const fp = (3*h/2)*x*x + 1;
              if (Math.abs(fp) < 1e-12) break;
              x -= f/fp;
            }
            return x;
          }
        },
        abs: { E: x => Math.abs(x), prox: (y,h) => Math.sign(y)*Math.max(Math.abs(y)-h,0) }
      };

      const state = {
        eKey: 'quad', h: 0.30, prev: 1.5,
        showE: true, showP: true, showF: true, dragging: false
      };

      function svgEl(name, attrs = {}) {
        const el = document.createElementNS(NS, name);
        for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
        return el;
      }
      const fmt = (x, d=2) => x.toFixed(d);
      function E(x) { return energies[state.eKey].E(x); }
      function nextIter() { return energies[state.eKey].prox(state.prev, state.h); }
      function P(x) { return (1/(2*state.h))*(x-state.prev)*(x-state.prev); }
      function F(x) { return E(x) + P(x); }

      function renderGrid() {
        const g = document.getElementById('mm-grid');
        g.innerHTML = '';
        g.appendChild(svgEl('rect', { x: PLOT_X, y: PLOT_Y, width: PLOT_W, height: PLOT_H, fill: 'var(--color-background-secondary)', 'fill-opacity': 0.45, stroke: 'var(--color-border-tertiary)', 'stroke-width': 0.5, rx: 4 }));
        g.appendChild(svgEl('line', { x1: m2sX(X_MIN), y1: m2sY(0), x2: m2sX(X_MAX), y2: m2sY(0), stroke: 'var(--color-border-secondary)', 'stroke-width': 0.5 }));
        g.appendChild(svgEl('line', { x1: m2sX(0), y1: m2sY(Y_MIN), x2: m2sX(0), y2: m2sY(Y_MAX), stroke: 'var(--color-border-secondary)', 'stroke-width': 0.5 }));
        for (let i = -3; i <= 3; i++) {
          if (i === 0) continue;
          const sx = m2sX(i), sy = m2sY(0);
          g.appendChild(svgEl('line', { x1: sx, y1: sy-3, x2: sx, y2: sy+3, stroke: 'var(--color-border-secondary)', 'stroke-width': 0.5 }));
          const lbl = svgEl('text', { x: sx, y: sy+13, 'font-size': 10, 'text-anchor': 'middle', 'font-family': 'var(--font-sans)', fill: 'var(--color-text-tertiary)' });
          lbl.textContent = i;
          g.appendChild(lbl);
        }
        for (let j = 1; j <= 5; j++) {
          const sx = m2sX(0), sy = m2sY(j);
          g.appendChild(svgEl('line', { x1: sx-3, y1: sy, x2: sx+3, y2: sy, stroke: 'var(--color-border-secondary)', 'stroke-width': 0.5 }));
          const lbl = svgEl('text', { x: sx-6, y: sy+3, 'font-size': 10, 'text-anchor': 'end', 'font-family': 'var(--font-sans)', fill: 'var(--color-text-tertiary)' });
          lbl.textContent = j;
          g.appendChild(lbl);
        }
        const xL = svgEl('text', { x: m2sX(X_MAX)-5, y: m2sY(0)-6, 'font-size': 11, 'text-anchor': 'end', 'font-family': 'var(--font-sans)', fill: 'var(--color-text-tertiary)' });
        xL.textContent = 'x';
        g.appendChild(xL);
      }

      function drawCurve(gid, fn, color, dashed, weight, opacity) {
        const g = document.getElementById(gid);
        g.innerHTML = '';
        if (!fn) return;
        const N = 320;
        let d = '', inPath = false;
        for (let i = 0; i <= N; i++) {
          const x = X_MIN + (X_MAX-X_MIN)*i/N;
          let y;
          try { y = fn(x); } catch (err) { y = NaN; }
          if (!isFinite(y) || y > Y_MAX+2 || y < Y_MIN-2) { inPath = false; continue; }
          const sx = m2sX(x), sy = m2sY(y);
          d += (inPath ? ' L ' : ' M ') + sx.toFixed(1) + ',' + sy.toFixed(1);
          inPath = true;
        }
        if (d) {
          const attrs = { d: d, fill: 'none', stroke: color, 'stroke-width': weight, opacity: opacity, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' };
          if (dashed) attrs['stroke-dasharray'] = '4 3';
          g.appendChild(svgEl('path', attrs));
        }
      }

      function renderCurves() {
        drawCurve('mm-curve-e', state.showE ? E : null, COLORS.e, false, 1.4, 0.75);
        drawCurve('mm-curve-p', state.showP ? P : null, COLORS.parab, true, 1.2, 0.75);
        drawCurve('mm-curve-f', state.showF ? F : null, COLORS.f, false, 1.9, 0.92);
      }

      function renderMarkers() {
        const g = document.getElementById('mm-markers');
        g.innerHTML = '';
        const xp = state.prev, xn = nextIter();
        const ep = E(xp), fmin = F(xn);
        const topPrev = Math.min(Y_MAX, Math.max(ep, fmin));
        g.appendChild(svgEl('line', { x1: m2sX(xp), y1: m2sY(0), x2: m2sX(xp), y2: m2sY(topPrev), stroke: COLORS.prev, 'stroke-width': 0.75, 'stroke-dasharray': '3 3', opacity: 0.55 }));
        g.appendChild(svgEl('line', { x1: m2sX(xn), y1: m2sY(0), x2: m2sX(xn), y2: m2sY(Math.min(Y_MAX, fmin)), stroke: COLORS.next, 'stroke-width': 0.75, 'stroke-dasharray': '3 3', opacity: 0.55 }));
        if (Math.abs(xp-xn) > 0.05) {
          const sy = m2sY(0)-12, x1 = m2sX(xp), x2 = m2sX(xn);
          const dir = x2 > x1 ? 1 : -1;
          g.appendChild(svgEl('line', { x1: x1+8*dir, y1: sy, x2: x2-9*dir, y2: sy, stroke: COLORS.arrow, 'stroke-width': 1.25, opacity: 0.7, 'marker-end': 'url(#mm-arrow)' }));
        }
        if (state.showE && ep >= Y_MIN && ep <= Y_MAX) {
          g.appendChild(svgEl('circle', { cx: m2sX(xp), cy: m2sY(ep), r: 4, fill: COLORS.prev, opacity: 0.6 }));
        }
        if (state.showF && fmin >= Y_MIN && fmin <= Y_MAX) {
          g.appendChild(svgEl('circle', { cx: m2sX(xn), cy: m2sY(fmin), r: 5, fill: COLORS.next, stroke: 'var(--color-background-primary)', 'stroke-width': 2 }));
        }
        g.appendChild(svgEl('circle', { cx: m2sX(xp), cy: m2sY(0), r: 7.5, fill: COLORS.prev, stroke: 'var(--color-background-primary)', 'stroke-width': 2, style: 'cursor: grab' }));
        g.appendChild(svgEl('circle', { cx: m2sX(xn), cy: m2sY(0), r: 5, fill: COLORS.next, stroke: 'var(--color-background-primary)', 'stroke-width': 1.5 }));
        const lblP = svgEl('text', { x: m2sX(xp), y: m2sY(0)+30, 'font-size': 12, 'text-anchor': 'middle', 'font-family': 'var(--font-sans)', fill: COLORS.prev, 'font-weight': 500 });
        const t1 = svgEl('tspan'); t1.textContent = 'χ'; lblP.appendChild(t1);
        const sub1 = svgEl('tspan', { 'font-size': 9, dy: 3 }); sub1.textContent = 'ℓ−1'; lblP.appendChild(sub1);
        g.appendChild(lblP);
        const lblN = svgEl('text', { x: m2sX(xn), y: m2sY(0)+30, 'font-size': 12, 'text-anchor': 'middle', 'font-family': 'var(--font-sans)', fill: COLORS.next, 'font-weight': 500 });
        const t2 = svgEl('tspan'); t2.textContent = 'χ'; lblN.appendChild(t2);
        const sub2 = svgEl('tspan', { 'font-size': 9, dy: 3 }); sub2.textContent = 'ℓ'; lblN.appendChild(sub2);
        g.appendChild(lblN);
      }

      function renderLegend() {
        const g = document.getElementById('mm-legend');
        g.innerHTML = '';
        const items = [
          { color: COLORS.e, label: 'Energy E(x)', visible: state.showE, opacity: 0.75 },
          { color: COLORS.parab, label: 'Parabola pull', dashed: true, visible: state.showP, opacity: 0.8 },
          { color: COLORS.f, label: 'Combined F(x)', visible: state.showF, opacity: 0.95 },
          { color: COLORS.prev, label: 'Previous iterate', visible: true, isDot: true },
          { color: COLORS.next, label: 'Next iterate', visible: true, isDot: true }
        ].filter(it => it.visible);
        const lx = 524, ly = 24, lh = 18*items.length+12, lw = 138;
        g.appendChild(svgEl('rect', { x: lx, y: ly, width: lw, height: lh, rx: 6, fill: 'var(--color-background-secondary)', stroke: 'var(--color-border-tertiary)', 'stroke-width': 0.5 }));
        let y = ly+16;
        for (const it of items) {
          if (it.isDot) {
            g.appendChild(svgEl('circle', { cx: lx+19, cy: y, r: 4.5, fill: it.color, stroke: 'var(--color-background-primary)', 'stroke-width': 1.5 }));
          } else {
            const la = { x1: lx+10, y1: y, x2: lx+28, y2: y, stroke: it.color, 'stroke-width': 1.75 };
            if (it.dashed) la['stroke-dasharray'] = '3 2';
            if (it.opacity !== undefined) la.opacity = it.opacity;
            g.appendChild(svgEl('line', la));
          }
          const t = svgEl('text', { x: lx+34, y: y+4, 'font-size': 11, fill: 'var(--color-text-secondary)', 'font-family': 'var(--font-sans)' });
          t.textContent = it.label;
          g.appendChild(t);
          y += 18;
        }
      }

      function updateInfo() {
        const xn = nextIter();
        document.getElementById('mm-i-prev').textContent = fmt(state.prev);
        document.getElementById('mm-i-next').textContent = fmt(xn);
        document.getElementById('mm-i-step').textContent = fmt(Math.abs(state.prev-xn));
      }

      function render() {
        renderGrid(); renderCurves(); renderMarkers(); renderLegend(); updateInfo();
      }

      document.getElementById('mm-h-slider').addEventListener('input', (e) => {
        state.h = parseFloat(e.target.value);
        document.getElementById('mm-h-val').textContent = state.h.toFixed(2);
        render();
      });
      document.querySelectorAll('#mm-energy-group .pill').forEach(btn => {
        btn.addEventListener('click', () => {
          document.querySelectorAll('#mm-energy-group .pill').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          state.eKey = btn.dataset.energy;
          render();
        });
      });
      document.getElementById('mm-t-e').addEventListener('change', (e) => { state.showE = e.target.checked; render(); });
      document.getElementById('mm-t-p').addEventListener('change', (e) => { state.showP = e.target.checked; render(); });
      document.getElementById('mm-t-f').addEventListener('change', (e) => { state.showF = e.target.checked; render(); });

      const svg = document.getElementById('mm-plot');
      function getCoords(cx, cy) {
        const pt = svg.createSVGPoint();
        pt.x = cx; pt.y = cy;
        const c = pt.matrixTransform(svg.getScreenCTM().inverse());
        return [c.x, c.y];
      }
      function setPrevFromX(sx) {
        const mx = Math.max(X_MIN+0.05, Math.min(X_MAX-0.05, s2mX(sx)));
        state.prev = mx;
        render();
      }
      svg.addEventListener('mousedown', (e) => {
        const [sx, sy] = getCoords(e.clientX, e.clientY);
        if (sx < PLOT_X || sx > PLOT_X+PLOT_W || sy < PLOT_Y || sy > PLOT_Y+PLOT_H) return;
        state.dragging = true;
        setPrevFromX(sx);
        e.preventDefault();
      });
      svg.addEventListener('mousemove', (e) => {
        if (!state.dragging) return;
        const [sx] = getCoords(e.clientX, e.clientY);
        setPrevFromX(sx);
      });
      window.addEventListener('mouseup', () => { state.dragging = false; });
      svg.addEventListener('touchstart', (e) => {
        if (!e.touches[0]) return;
        const [sx, sy] = getCoords(e.touches[0].clientX, e.touches[0].clientY);
        if (sx < PLOT_X || sx > PLOT_X+PLOT_W || sy < PLOT_Y || sy > PLOT_Y+PLOT_H) return;
        state.dragging = true;
        setPrevFromX(sx);
        e.preventDefault();
      }, { passive: false });
      svg.addEventListener('touchmove', (e) => {
        if (!state.dragging || !e.touches[0]) return;
        const [sx] = getCoords(e.touches[0].clientX, e.touches[0].clientY);
        setPrevFromX(sx);
        e.preventDefault();
      }, { passive: false });
      window.addEventListener('touchend', () => { state.dragging = false; });

      render();
    })();
    </script>
  </div>
  <figcaption>Interactive: one step of the minimizing-movements scheme (1.10). Drag the previous iterate $\chi_{\ell-1}$ along the $x$-axis; the next iterate $\chi_\ell$ always sits at the minimum of the combined function $F(x)=E(x)+\frac{1}{2h}(x-\chi_{\ell-1})^2$ (the parabola pull penalises moving far from $\chi_{\ell-1}$). Use the slider to vary the time step $h$, and the buttons to switch the energy ($\frac{1}{2}x^2$, $x^4/8$, $|x|$).</figcaption>
</figure>

<figure>
  <div class="mm2d-viz">
    <style>
      .mm2d-viz {
        --color-text-primary: #1a1a1a;
        --color-text-secondary: #4a4a4a;
        --color-text-tertiary: #777777;
        --color-text-danger: #c0392b;
        --color-background-primary: #ffffff;
        --color-background-secondary: #f3f4f8;
        --color-border-primary: #aaaaaa;
        --color-border-secondary: #c8c8c8;
        --color-border-tertiary: #dddddd;
        --border-radius-md: 6px;
        --font-sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      }
      .mm2d-viz .mm-controls { display: grid; gap: 14px; margin-top: 1rem; }
      .mm2d-viz .row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
      .mm2d-viz .row > label.lbl { font-size: 13px; color: var(--color-text-secondary); min-width: 92px; }
      .mm2d-viz .row > input[type="range"] { flex: 1; min-width: 160px; max-width: 320px; }
      .mm2d-viz .val { font-size: 13px; min-width: 36px; font-variant-numeric: tabular-nums; color: var(--color-text-primary); text-align: right; }
      .mm2d-viz .pill-group { display: inline-flex; gap: 4px; flex-wrap: wrap; }
      .mm2d-viz .pill { font-size: 12px; padding: 5px 10px; border: 0.5px solid var(--color-border-tertiary); border-radius: var(--border-radius-md); background: transparent; cursor: pointer; color: var(--color-text-primary); font-family: var(--font-sans); font-variant-numeric: tabular-nums; }
      .mm2d-viz .pill.active { background: var(--color-background-secondary); border-color: var(--color-border-primary); }
      .mm2d-viz .toggles { display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px; color: var(--color-text-secondary); }
      .mm2d-viz .toggles label { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }
      .mm2d-viz .info-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 0.25rem; }
      .mm2d-viz .info-cell { background: var(--color-background-secondary); border-radius: var(--border-radius-md); padding: 10px 14px; }
      .mm2d-viz .info-label { font-size: 12px; color: var(--color-text-secondary); }
      .mm2d-viz .info-val { font-size: 16px; font-weight: 500; font-variant-numeric: tabular-nums; margin-top: 2px; color: var(--color-text-primary); }
      .mm2d-viz .stability { font-size: 12px; color: var(--color-text-secondary); min-height: 16px; }
      .mm2d-viz .stability.warn { color: var(--color-text-danger); }
      .mm2d-viz .hint { font-size: 12px; color: var(--color-text-tertiary); }
      .mm2d-viz .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); border: 0; }
      .mm2d-viz button { font-family: var(--font-sans); font-size: 13px; padding: 6px 12px; border: 0.5px solid var(--color-border-tertiary); border-radius: var(--border-radius-md); background: var(--color-background-primary); color: var(--color-text-primary); cursor: pointer; }
      .mm2d-viz button:hover { background: var(--color-background-secondary); }
    </style>

    <svg id="mm2d-plot" width="100%" viewBox="0 0 680 380" style="display: block; cursor: crosshair;" role="img">
      <title>Minimizing movements visualization</title>
      <desc>Two-dimensional plot showing iterates of the minimizing-movements scheme on a quadratic energy, alongside continuous gradient flow and explicit Euler.</desc>
      <defs>
        <clipPath id="mm2d-plot-clip"><rect x="30" y="10" width="480" height="360"/></clipPath>
      </defs>
      <g id="mm2d-grid"></g>
      <g id="mm2d-contours" clip-path="url(#mm2d-plot-clip)"></g>
      <g id="mm2d-flow" clip-path="url(#mm2d-plot-clip)"></g>
      <g id="mm2d-ee-traj" clip-path="url(#mm2d-plot-clip)"></g>
      <g id="mm2d-prox" clip-path="url(#mm2d-plot-clip)"></g>
      <g id="mm2d-mm-traj" clip-path="url(#mm2d-plot-clip)"></g>
      <g id="mm2d-current" clip-path="url(#mm2d-plot-clip)"></g>
      <g id="mm2d-legend"></g>
    </svg>

    <div class="mm-controls">
      <div class="row">
        <label class="lbl" for="mm2d-h-slider">Time step h</label>
        <input id="mm2d-h-slider" type="range" min="0.05" max="1.5" step="0.05" value="0.30">
        <span class="val" id="mm2d-h-val">0.30</span>
      </div>
      <div class="row">
        <label class="lbl">Energy E(x, y)</label>
        <div class="pill-group" id="mm2d-aniso-group">
          <button class="pill" data-b="1">½(x² + y²)</button>
          <button class="pill active" data-b="2">½(x² + 2y²)</button>
          <button class="pill" data-b="4">½(x² + 4y²)</button>
          <button class="pill" data-b="8">½(x² + 8y²)</button>
        </div>
      </div>
      <div class="row">
        <button id="mm2d-btn-step">Step</button>
        <button id="mm2d-btn-run">Auto</button>
        <button id="mm2d-btn-reset">Reset</button>
        <span class="hint">Click anywhere on the plot to set a new initial point χ₀</span>
      </div>
      <div class="toggles">
        <label><input type="checkbox" id="mm2d-t-flow" checked> Gradient flow</label>
        <label><input type="checkbox" id="mm2d-t-ee" checked> Explicit Euler</label>
        <label><input type="checkbox" id="mm2d-t-prox" checked> Proximal pull</label>
      </div>
      <div class="stability" id="mm2d-stab-line"></div>
      <div class="info-row">
        <div class="info-cell">
          <div class="info-label">Iteration ℓ</div>
          <div class="info-val" id="mm2d-i-l">0</div>
        </div>
        <div class="info-cell">
          <div class="info-label">Position χ<sub>ℓ</sub></div>
          <div class="info-val" id="mm2d-i-x">(2.50, 2.00)</div>
        </div>
        <div class="info-cell">
          <div class="info-label">Energy E(χ<sub>ℓ</sub>)</div>
          <div class="info-val" id="mm2d-i-e">3.13</div>
        </div>
      </div>
    </div>

    <script>
    (function() {
      const PLOT_X = 30, PLOT_Y = 10, PLOT_W = 480, PLOT_H = 360, SCALE = 60;
      const X_MIN = -4, X_MAX = 4, Y_MIN = -3, Y_MAX = 3;
      const NS = 'http://www.w3.org/2000/svg';
      const m2sX = x => PLOT_X + (x - X_MIN) * SCALE;
      const m2sY = y => PLOT_Y + (Y_MAX - y) * SCALE;
      const s2mX = sx => (sx - PLOT_X) / SCALE + X_MIN;
      const s2mY = sy => Y_MAX - (sy - PLOT_Y) / SCALE;

      const COLORS = {
        contour: '#888780',
        mm: '#185FA5',
        ee: '#D85A30',
        flow: '#1D9E75',
        prox: '#BA7517'
      };

      const state = {
        a: 1, b: 2, h: 0.30,
        x0: [2.5, 2.0],
        trajMM: [[2.5, 2.0]],
        trajEE: [[2.5, 2.0]],
        showFlow: true, showEE: true, showProx: true,
        running: false, timer: null
      };

      const fmt = (x, d=2) => {
        if (!isFinite(x)) return '∞';
        if (Math.abs(x) > 999) return x.toExponential(1).replace('+', '');
        return x.toFixed(d);
      };

      function svgEl(name, attrs = {}) {
        const el = document.createElementNS(NS, name);
        for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
        return el;
      }

      function E(x, y) { return 0.5 * (state.a*x*x + state.b*y*y); }

      function renderGrid() {
        const g = document.getElementById('mm2d-grid');
        g.innerHTML = '';
        g.appendChild(svgEl('rect', {
          x: PLOT_X, y: PLOT_Y, width: PLOT_W, height: PLOT_H,
          fill: 'var(--color-background-secondary)', 'fill-opacity': 0.45,
          stroke: 'var(--color-border-tertiary)', 'stroke-width': 0.5,
          rx: 4
        }));
        g.appendChild(svgEl('line', {
          x1: m2sX(X_MIN), y1: m2sY(0), x2: m2sX(X_MAX), y2: m2sY(0),
          stroke: 'var(--color-border-secondary)', 'stroke-width': 0.5
        }));
        g.appendChild(svgEl('line', {
          x1: m2sX(0), y1: m2sY(Y_MIN), x2: m2sX(0), y2: m2sY(Y_MAX),
          stroke: 'var(--color-border-secondary)', 'stroke-width': 0.5
        }));
        for (let i = -3; i <= 3; i++) {
          if (i === 0) continue;
          g.appendChild(svgEl('line', {
            x1: m2sX(i), y1: m2sY(0)-3, x2: m2sX(i), y2: m2sY(0)+3,
            stroke: 'var(--color-border-secondary)', 'stroke-width': 0.5
          }));
        }
        for (let j = -2; j <= 2; j++) {
          if (j === 0) continue;
          g.appendChild(svgEl('line', {
            x1: m2sX(0)-3, y1: m2sY(j), x2: m2sX(0)+3, y2: m2sY(j),
            stroke: 'var(--color-border-secondary)', 'stroke-width': 0.5
          }));
        }
        const xLab = svgEl('text', { x: m2sX(X_MAX) - 10, y: m2sY(0) - 6, 'font-size': 11, 'font-family': 'var(--font-sans)', fill: 'var(--color-text-tertiary)', 'text-anchor': 'end' });
        xLab.textContent = 'x';
        g.appendChild(xLab);
        const yLab = svgEl('text', { x: m2sX(0) + 6, y: m2sY(Y_MAX) + 12, 'font-size': 11, 'font-family': 'var(--font-sans)', fill: 'var(--color-text-tertiary)' });
        yLab.textContent = 'y';
        g.appendChild(yLab);
      }

      function renderContours() {
        const g = document.getElementById('mm2d-contours');
        g.innerHTML = '';
        const levels = [0.25, 0.5, 1, 2, 4, 8];
        const cx = m2sX(0), cy = m2sY(0);
        for (const c of levels) {
          const rx = Math.sqrt(2*c/state.a) * SCALE;
          const ry = Math.sqrt(2*c/state.b) * SCALE;
          g.appendChild(svgEl('ellipse', {
            cx: cx, cy: cy, rx: rx, ry: ry,
            fill: 'none', stroke: COLORS.contour,
            'stroke-width': 0.5, opacity: 0.4
          }));
        }
        g.appendChild(svgEl('circle', { cx: cx, cy: cy, r: 2.5, fill: '#5F5E5A' }));
        const lbl = svgEl('text', { x: cx + 5, y: cy + 12, 'font-size': 10, 'font-family': 'var(--font-sans)', fill: 'var(--color-text-tertiary)' });
        lbl.textContent = 'argmin E';
        g.appendChild(lbl);
      }

      function renderFlow() {
        const g = document.getElementById('mm2d-flow');
        g.innerHTML = '';
        if (!state.showFlow) return;
        const T = 6 * Math.max(1/state.a, 1/state.b);
        const N = 140;
        const pts = [];
        for (let i = 0; i <= N; i++) {
          const t = T * i / N;
          const x = state.x0[0] * Math.exp(-state.a * t);
          const y = state.x0[1] * Math.exp(-state.b * t);
          pts.push([m2sX(x), m2sY(y)]);
        }
        const d = 'M ' + pts.map(p => p.map(v => v.toFixed(1)).join(',')).join(' L ');
        g.appendChild(svgEl('path', {
          d: d, fill: 'none', stroke: COLORS.flow,
          'stroke-width': 1.5, 'stroke-dasharray': '4 3', opacity: 0.85
        }));
      }

      function renderTrajectory(id, traj, color) {
        const g = document.getElementById(id);
        g.innerHTML = '';
        if (traj.length === 0) return;
        if (traj.length > 1) {
          const pts = traj.map(p => [m2sX(p[0]), m2sY(p[1])]);
          const d = 'M ' + pts.map(p => p.map(v => isFinite(v) ? v.toFixed(1) : '0').join(',')).join(' L ');
          g.appendChild(svgEl('path', {
            d: d, fill: 'none', stroke: color, 'stroke-width': 1.75, opacity: 0.9
          }));
        }
        for (let i = 1; i < traj.length - 1; i++) {
          const mx = traj[i][0], my = traj[i][1];
          if (!isFinite(mx) || !isFinite(my)) continue;
          g.appendChild(svgEl('circle', {
            cx: m2sX(mx), cy: m2sY(my), r: 2.5, fill: color
          }));
        }
      }

      function renderProx() {
        const g = document.getElementById('mm2d-prox');
        g.innerHTML = '';
        if (!state.showProx) return;
        const tip = state.trajMM[state.trajMM.length - 1];
        const cx = m2sX(tip[0]), cy = m2sY(tip[1]);
        for (const c of [0.5, 1, 2]) {
          const r = Math.sqrt(2 * state.h * c) * SCALE;
          g.appendChild(svgEl('circle', {
            cx: cx, cy: cy, r: r, fill: 'none', stroke: COLORS.prox,
            'stroke-width': 0.75, 'stroke-dasharray': '3 3', opacity: 0.6
          }));
        }
      }

      function renderCurrent() {
        const g = document.getElementById('mm2d-current');
        g.innerHTML = '';
        g.appendChild(svgEl('circle', {
          cx: m2sX(state.x0[0]), cy: m2sY(state.x0[1]), r: 6,
          fill: 'var(--color-background-primary)',
          stroke: COLORS.mm, 'stroke-width': 2
        }));
        if (state.trajMM.length > 1) {
          const tip = state.trajMM[state.trajMM.length - 1];
          g.appendChild(svgEl('circle', {
            cx: m2sX(tip[0]), cy: m2sY(tip[1]), r: 5,
            fill: COLORS.mm, stroke: 'var(--color-background-primary)', 'stroke-width': 1.5
          }));
        }
      }

      function renderLegend() {
        const g = document.getElementById('mm2d-legend');
        g.innerHTML = '';
        const items = [
          { color: COLORS.contour, label: 'Energy contours', visible: true, opacity: 0.6 },
          { color: COLORS.mm, label: 'Min. movements', visible: true },
          { color: COLORS.ee, label: 'Explicit Euler', visible: state.showEE },
          { color: COLORS.flow, label: 'Gradient flow', dashed: true, visible: state.showFlow },
          { color: COLORS.prox, label: 'Proximal pull', dashed: true, visible: state.showProx, opacity: 0.7 }
        ].filter(it => it.visible);

        const lx = 524, ly = 16, lh = 18 * items.length + 12, lw = 138;
        g.appendChild(svgEl('rect', {
          x: lx, y: ly, width: lw, height: lh, rx: 6,
          fill: 'var(--color-background-secondary)',
          stroke: 'var(--color-border-tertiary)', 'stroke-width': 0.5
        }));
        let y = ly + 16;
        for (const it of items) {
          const lineAttrs = {
            x1: lx + 10, y1: y, x2: lx + 28, y2: y,
            stroke: it.color, 'stroke-width': 1.75
          };
          if (it.dashed) lineAttrs['stroke-dasharray'] = '3 2';
          if (it.opacity !== undefined) lineAttrs.opacity = it.opacity;
          g.appendChild(svgEl('line', lineAttrs));
          const t = svgEl('text', {
            x: lx + 34, y: y + 4, 'font-size': 11,
            fill: 'var(--color-text-secondary)', 'font-family': 'var(--font-sans)'
          });
          t.textContent = it.label;
          g.appendChild(t);
          y += 18;
        }
      }

      function updateInfo() {
        document.getElementById('mm2d-i-l').textContent = state.trajMM.length - 1;
        const tip = state.trajMM[state.trajMM.length - 1];
        document.getElementById('mm2d-i-x').textContent = '(' + fmt(tip[0]) + ', ' + fmt(tip[1]) + ')';
        document.getElementById('mm2d-i-e').textContent = fmt(E(tip[0], tip[1]), 3);

        const stabLine = document.getElementById('mm2d-stab-line');
        if (!state.showEE) {
          stabLine.textContent = '';
          stabLine.classList.remove('warn');
        } else {
          const factor = state.h * Math.max(state.a, state.b);
          if (factor >= 2) {
            stabLine.textContent = 'Explicit Euler unstable: h · max(a,b) = ' + fmt(factor) + ' ≥ 2. Implicit Euler still converges.';
            stabLine.classList.add('warn');
          } else {
            stabLine.textContent = 'Both methods stable: h · max(a,b) = ' + fmt(factor) + ' < 2.';
            stabLine.classList.remove('warn');
          }
        }
      }

      function render() {
        renderGrid();
        renderContours();
        renderFlow();
        renderTrajectory('mm2d-ee-traj', state.showEE ? state.trajEE : [], COLORS.ee);
        renderProx();
        renderTrajectory('mm2d-mm-traj', state.trajMM, COLORS.mm);
        renderCurrent();
        renderLegend();
        updateInfo();
      }

      function step() {
        const tip = state.trajMM[state.trajMM.length - 1];
        state.trajMM.push([
          tip[0] / (1 + state.h * state.a),
          tip[1] / (1 + state.h * state.b)
        ]);
        if (state.trajEE.length === state.trajMM.length - 1) {
          const eeT = state.trajEE[state.trajEE.length - 1];
          let nx = (1 - state.h * state.a) * eeT[0];
          let ny = (1 - state.h * state.b) * eeT[1];
          if (Math.abs(nx) > 50) nx = Math.sign(nx) * 50;
          if (Math.abs(ny) > 50) ny = Math.sign(ny) * 50;
          state.trajEE.push([nx, ny]);
        }
        render();
      }

      function reset() {
        state.trajMM = [state.x0.slice()];
        state.trajEE = [state.x0.slice()];
        if (state.running) toggleAuto();
        render();
      }

      function toggleAuto() {
        if (state.running) {
          clearInterval(state.timer);
          state.timer = null;
          state.running = false;
          document.getElementById('mm2d-btn-run').textContent = 'Auto';
        } else {
          state.timer = setInterval(() => {
            step();
            if (state.trajMM.length > 60) {
              clearInterval(state.timer);
              state.timer = null;
              state.running = false;
              document.getElementById('mm2d-btn-run').textContent = 'Auto';
            }
          }, 320);
          state.running = true;
          document.getElementById('mm2d-btn-run').textContent = 'Pause';
        }
      }

      document.getElementById('mm2d-h-slider').addEventListener('input', (e) => {
        state.h = parseFloat(e.target.value);
        document.getElementById('mm2d-h-val').textContent = state.h.toFixed(2);
        reset();
      });
      document.querySelectorAll('#mm2d-aniso-group .pill').forEach(btn => {
        btn.addEventListener('click', () => {
          document.querySelectorAll('#mm2d-aniso-group .pill').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          state.b = parseFloat(btn.dataset.b);
          reset();
        });
      });
      document.getElementById('mm2d-btn-step').addEventListener('click', step);
      document.getElementById('mm2d-btn-run').addEventListener('click', toggleAuto);
      document.getElementById('mm2d-btn-reset').addEventListener('click', reset);
      document.getElementById('mm2d-t-flow').addEventListener('change', (e) => { state.showFlow = e.target.checked; render(); });
      document.getElementById('mm2d-t-ee').addEventListener('change', (e) => { state.showEE = e.target.checked; render(); });
      document.getElementById('mm2d-t-prox').addEventListener('change', (e) => { state.showProx = e.target.checked; render(); });

      const svg = document.getElementById('mm2d-plot');
      svg.addEventListener('click', (e) => {
        const pt = svg.createSVGPoint();
        pt.x = e.clientX; pt.y = e.clientY;
        const cursor = pt.matrixTransform(svg.getScreenCTM().inverse());
        if (cursor.x < PLOT_X || cursor.x > PLOT_X + PLOT_W) return;
        if (cursor.y < PLOT_Y || cursor.y > PLOT_Y + PLOT_H) return;
        const mx = s2mX(cursor.x);
        const my = s2mY(cursor.y);
        state.x0 = [Math.round(mx*100)/100, Math.round(my*100)/100];
        reset();
      });

      render();
    })();
    </script>
  </div>
  <figcaption>Interactive: full iteration of (1.10) on the 2D anisotropic energy $E(x,y)=\frac{1}{2}(a x^2+b y^2)$. The blue trail is the minimizing-movements scheme, the red trail is explicit Euler, the green dashed curve is the continuous gradient flow, and the dashed orange rings show the parabolic "pull" $\frac{1}{2h}|x-\chi_{\ell-1}|^2$ around the current iterate. Use <em>Step</em> for a single iteration, <em>Auto</em> to play, <em>Reset</em> to clear; click anywhere to set a new initial point $\chi_0$. Try the most anisotropic energy ($\frac{1}{2}(x^2+8y^2)$) at moderate $h$: explicit Euler oscillates and blows up while minimizing movements stays put — the very statement of $A$-stability.</figcaption>
</figure>

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

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Why don't kinks of $E$ break uniqueness? A worked piecewise-linear example</summary>

A natural worry: at a kink of $E$, the subdifferential $\partial E$ contains many distinct supporting linear functionals — many "tangents that minimize the linear-approximation error in different ways". If two solutions starting at the same point pick two different tangents, will they not drift apart, contradicting (1.14)?

The picture *of the function* is correct: a kink really has a whole interval of subgradients. What rescues uniqueness is that being a *solution* of the differential inclusion is a constraint over an entire forward time interval, not at a single instant. Almost every supporting tangent at a kink is exposed as a fake the moment the trajectory tries to step on it, because the trajectory leaves the kink and lands in a smooth segment whose unique gradient disagrees with the chosen tangent. Out of the entire interval $\partial E(x)$ at a kink, exactly one element is consistent with the inclusion for $t>0$: the **minimum-norm element** $\partial^0 E(x)$, the projection of $0$ onto the closed convex set $\partial E(x)$.

We illustrate this on a piecewise-linear bowl with several kinks.

**Setup: a five-segment convex bowl.** Define

$$
E(x) \;=\;
\begin{cases}
-2x-3 & x\le -2 \\
-x-1 & -2\le x\le -1 \\
0 & -1\le x\le 1 \\
x-1 & 1\le x\le 2 \\
2x-3 & x\ge 2
\end{cases}
$$

Slopes (left-to-right) $-2,\,-1,\,0,\,+1,\,+2$ — strictly increasing, so $E$ is convex. There are four kinks at $x=-2,\,-1,\,1,\,2$, plus a flat plateau on $[-1,1]$ where $E\equiv 0$ (the global minimum set).

**Subdifferential at every point.** The subdifferential is the closed interval between the left-slope and right-slope:

| $x$ | $\partial E(x)$ | $\partial^0\!E(x)$ |
|---|---|---|
| $x<-2$ | $\{-2\}$ | $-2$ |
| $x=-2$ | $[-2,-1]$ | $-1$ |
| $-2<x<-1$ | $\{-1\}$ | $-1$ |
| $x=-1$ | $[-1,\;0]$ | $\;\;0$ |
| $-1<x<1$ | $\{0\}$ | $\;\;0$ |
| $x=1$ | $[\;0,\;+1]$ | $\;\;0$ |
| $1<x<2$ | $\{+1\}$ | $+1$ |
| $x=2$ | $[+1,+2]$ | $+1$ |
| $x>2$ | $\{+2\}$ | $+2$ |

At every kink $\partial E$ is a non-trivial interval — the multivaluedness is real. The min-norm column is the *unique* element each kink contributes to the actual gradient flow; everything else in the interval is a tangent that fails to be a velocity.

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/uniqueness_multi_segment.png' | relative_url }}" alt="Four-panel figure: (a) piecewise-linear bowl with fan of supporting tangents at x=2; (b) subdifferential as a multivalued staircase with min-norm selection overlaid; (c) candidate trajectories from x_0=2 showing only the minimum-norm choice satisfies the inclusion; (d) two genuine solutions from ±2 with their non-increasing pairwise distance" loading="lazy">
  <figcaption>(a) The piecewise-linear bowl with kinks at $\pm 1,\pm 2$ and the fan of supporting linear functionals at the kink $x=2$ — slope $1$ (the min-norm tangent, green) and the rest of $\partial E(2)=[1,2]$ (dashed red). (b) The subdifferential as a multivalued map: blue traces $\partial E$ (singletons on smooth segments, vertical intervals at kinks); the green dashed line is the min-norm selection $\partial^0\!E$. (c) Three candidate trajectories starting at $x_0=2$: only $p=1$ satisfies the inclusion for $t>0$ (the other choices land in $(1,2)$ where $\partial E=\{1\}$ forces $\dot x=-1$, contradicting $\dot x=-p$). (d) Two genuine solutions from $x_0=\pm 2$, with their pairwise distance (red dashed) decreasing from $4$ to $2$ until each parks at the boundary of the flat plateau — illustrating the contraction (1.14).</figcaption>
</figure>

**The kink at $x_0=2$: enumerate the candidates and watch them fail.** Start two solutions at $x_1(0)=x_2(0)=2$. The kinematic options at $t=0$ are: pick any $p\in\partial E(2)=[1,2]$ and set $\dot x=-p$. So $\dot x\in[-2,-1]$ at $t=0$.

For any small $t>0$ the trajectory is at $x(t)=2-pt\in(1,2)$, *strictly inside the linear segment* where $E(x)=x-1$. On that segment $E$ is smooth with $\nabla E\equiv 1$, so the inclusion at time $t>0$ has only one option:

$$
\dot x(t) \;\in\; -\partial E(x(t)) \;=\; \{-1\}.
$$

We check each candidate against this hard constraint:

| Tangent $p\in\partial E(2)$ | Velocity $\dot x = -p$ | Required $\dot x$ on $(1,2)$ | Verdict |
|---|---|---|---|
| $p=1$ (min-norm) | $-1$ | $-1$ | ✓ consistent |
| $p=1.25$ | $-1.25$ | $-1$ | ✗ contradiction |
| $p=1.5$ | $-1.5$ | $-1$ | ✗ contradiction |
| $p=2$ (max) | $-2$ | $-1$ | ✗ contradiction |

Out of the entire interval $\partial E(2)=[1,2]$, exactly one element gives a valid solution: the closest one to $0$. Every other "tangent at the kink" is exposed as a fake by the very next moment of the flow. Two solutions trying to "pick different tangents" produce one solution and one **non-solution** — not two solutions drifting apart.

**The full trajectory from $x_0=2$.** Tracking $\partial^0 E$ at every instant:

* For $t\in[0,1]$: $x(t)=2-t$, on the segment $(1,2)$ where $\nabla E=1$. Velocity $-1$.
* At $t=1$: $x=1$, the kink between rising segment and plateau. $\partial E(1)=[0,1]$, min-norm $0$.
* For $t>1$: $\dot x = 0$, so $x(t)\equiv 1$ forever.

Why doesn't the trajectory keep going into the flat region? If $x(t)$ ever ventured into $(-1,1)$, it would land in $\partial E=\{0\}$, forcing $\dot x=0$, so it can't move there in the first place. The trajectory parks exactly at the boundary of the flat region. The "lazy" min-norm selection is precisely what produces this behavior — the trajectory stops as soon as it can.

**Sanity-check Theorem 2 with two trajectories.** Take $x_1(0)=2$ and $x_2(0)=-2$. By symmetry, the unique solutions are

$$
x_1(t) = \begin{cases} 2-t & 0\le t\le 1 \\ 1 & t\ge 1 \end{cases},
\qquad
x_2(t) = \begin{cases} -2+t & 0\le t\le 1 \\ -1 & t\ge 1 \end{cases}.
$$

Their pairwise distance is

$$
|x_1(t)-x_2(t)| =
\begin{cases}
4-2t & 0\le t\le 1 \\
2 & t\ge 1
\end{cases}
$$

— monotonically non-increasing, exactly as (1.14) predicts. The two trajectories converge toward the flat plateau but, once they reach its boundary, freeze at distance $2$: they never reach a single common minimizer because the plateau gives them an entire continuum of minimizers to choose from. *Distance is non-increasing, not strictly decreasing* — the non-strictness of (1.14) comes precisely from non-uniqueness of the minimizer of $E$.

**The "drift" scenario, made fully explicit.** Insist that $x_1(0)=x_2(0)=2$ and that they pick different tangents $p_1=1,\,p_2=2$. Then:

* $x_1$ with $p_1=1$: $x_1(t)=2-t$. Inclusion holds for $t>0$. ✓
* $x_2$ with $p_2=2$: $x_2(t)=2-2t$. Inclusion fails on every $t\in(0,1/2)$. ✗

The would-be drift $\|x_1(t)-x_2(t)\|=t$ is between *one* solution and *one phantom*. The phantom satisfies the inclusion only at $t=0$, not on any open interval. If we instead force $x_2$ to be a real solution from $x_0=2$, it has to pick $p_2=1$ as well, and then $x_1\equiv x_2$. No drift. Uniqueness.

**Summary.**

* Each kink has a whole interval of supporting tangents — the multivaluedness of $\partial E$ at kinks is real.
* But only one tangent per kink is consistent with the differential inclusion remaining satisfied for $t>0$: the min-norm element $\partial^0 E$.
* Every other tangent in $\partial E$ becomes a "bad subgradient" the instant the trajectory tries to step on it, because the trajectory leaves the kink and lands in a smooth segment whose unique gradient disagrees with the chosen tangent.
* The "two drifting solutions" are then revealed as one solution and one phantom: the phantom is a curve that satisfies the inclusion at $t=0$ only, not on any open interval.
* Theorem 2 then holds across genuine solutions: $\|x_1-x_2\|$ never grows.

</details>
</div>

