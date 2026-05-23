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
  <figcaption>The identity $E(x(t))+\int_0^t \|\dot{x}\|^2\,ds = E(x_0)$ visualised. Energy (blue) drains away along the trajectory and is exactly recovered as accumulated dissipation (orange); the dashed line shows their sum, which is conserved.</figcaption>
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
  <figcaption>At a fixed point $x_*$, the directional derivative $\langle v,\nabla E(x_*)\rangle$ varies as $\cos$ over the unit sphere (right). Among unit vectors, it is minimized exactly when $v=-\nabla E/\|\nabla E\|$ (left, green) and maximized at $v=+\nabla E/\|\nabla E\|$ (red).</figcaption>
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
  <figcaption>Hamilton's principle for a particle in 1D under gravity, $L=\frac{1}{2}\dot q^2-q$, fixed endpoints $q(0)=q(1)=0$. <em>Left</em>: the physical trajectory is the parabola $q_*(t)=\frac{1}{2}t(1-t)$ (red); other admissible trajectories sharing the same endpoints differ from $q_*$ by $\alpha\sin(\pi t)$. <em>Right</em>: the action $S[q_\alpha]=\int_0^1(\frac{1}{2}\dot q^2-q)\,dt$ as a function of $\alpha$. The graph has horizontal tangent at $\alpha=0$, i.e. $\frac{dS}{d\alpha}\|_{0}=0$, which is exactly the EL equation $\ddot q_*=-1$.</figcaption>
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
  <figcaption>Total energy $E(x)+\frac{1}{2}m\|v\|^2$ along the Newton-with-friction dynamics for the harmonic potential $E(x)=\frac{1}{2}x^2$. For $\lambda=0$ it is conserved (oscillation between potential and kinetic); for $\lambda>0$ it strictly decreases at rate $-\lambda\|v\|^2$.</figcaption>
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
  <figcaption>The subdifferential at a kink. For $E(x)=\|x\|$ the graph at $x=0$ admits a whole interval of supporting affine minorants (slopes in $[-1,1]$, blue/green); a slope outside this interval (red, $1.4$) fails the inequality on one side. Hence $\partial E(0)=[-1,+1]$.</figcaption>
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
  <figcaption>Interactive: one step of the minimizing-movements scheme (1.10). Drag the previous iterate $\chi_{\ell-1}$ along the $x$-axis; the next iterate $\chi_\ell$ always sits at the minimum of the combined function $F(x)=E(x)+\frac{1}{2h}(x-\chi_{\ell-1})^2$ (the parabola pull penalises moving far from $\chi_{\ell-1}$). Use the slider to vary the time step $h$, and the buttons to switch the energy ($\frac{1}{2}x^2$, $x^4/8$, $\|x\|$).</figcaption>
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
  <figcaption>Interactive: full iteration of (1.10) on the 2D anisotropic energy $E(x,y)=\frac{1}{2}(a x^2+b y^2)$. The blue trail is the minimizing-movements scheme, the red trail is explicit Euler, the green dashed curve is the continuous gradient flow, and the dashed orange rings show the parabolic "pull" $\frac{1}{2h}\|x-\chi_{\ell-1}\|^2$ around the current iterate. Use <em>Step</em> for a single iteration, <em>Auto</em> to play, <em>Reset</em> to clear; click anywhere to set a new initial point $\chi_0$. Try the most anisotropic energy ($\frac{1}{2}(x^2+8y^2)$) at moderate $h$: explicit Euler oscillates and blows up while minimizing movements stays put — the very statement of $A$-stability.</figcaption>
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
  <figcaption>The contraction property. Two trajectories of the same gradient flow (left) are joined by grey segments at corresponding times; the segment lengths shrink monotonically. The right panel plots $\|x_1(t)-x_2(t)\|$ as a function of $t$, confirming the non-increasing behavior that yields uniqueness.</figcaption>
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

### 1.6 Long-term Asymptotics

So far we have established that solutions to the gradient-flow equation (1.1) (or the differential inclusion (1.8) in the convex non-smooth setting) **exist** and are **unique**. The next natural question is: what does the trajectory $x(t)$ do as $t\to\infty$? We expect convergence to a critical point — and, under suitable convexity, to the unique global minimizer of $E$. The next two theorems make this expectation quantitative. They differ only in how strong the convexity assumption is:

* **uniform convexity** $\Rightarrow$ *exponential* convergence;
* mere **convexity** $\Rightarrow$ *algebraic* (rate $1/t$) convergence.

These are the simplest cases. For general non-convex energies, even existence of a long-time limit is delicate — the trajectory may oscillate, get stuck at a saddle, or escape to infinity.

#### Auxiliary inequalities

Two analytical workhorses appear repeatedly in the proofs that follow: **Young's inequality**, which converts a product into a weighted sum of squares, and **Gronwall's inequality**, which converts a differential inequality into an exponential bound. We record them here for reference.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Young's inequality)</span></p>

For all $a,b\ge 0$ and any conjugate exponents $p,q\in(1,\infty)$ with $\tfrac1p+\tfrac1q=1$,

$$
ab \;\le\; \frac{a^p}{p} + \frac{b^q}{q}, \tag{Y}
$$

with equality iff $a^p=b^q$. The most important special case is $p=q=2$:

$$
ab \;\le\; \tfrac12 a^2 + \tfrac12 b^2, \tag{Y$_2$}
$$

with equality iff $a=b$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Young's inequality</summary>

The case $p=q=2$ follows immediately from the trivial $(a-b)^2\ge 0$:

$$
0\;\le\;(a-b)^2 = a^2 - 2ab + b^2 \quad\Longleftrightarrow\quad 2ab\;\le\; a^2+b^2.
$$

The general case follows from the **convexity of $\exp$**. For $a,b>0$, write $a=\exp(\tfrac1p\log a^p)$ and $b=\exp(\tfrac1q\log b^q)$, so that, using $\tfrac1p+\tfrac1q=1$ and Jensen's inequality applied to $\exp$,

$$
ab \;=\; \exp\!\Bigl(\tfrac1p\log a^p+\tfrac1q\log b^q\Bigr) \;\le\; \tfrac1p\exp(\log a^p) + \tfrac1q\exp(\log b^q) \;=\; \frac{a^p}{p}+\frac{b^q}{q}.
$$

Equality in Jensen's holds iff the two values $\log a^p$ and $\log b^q$ agree — i.e. iff $a^p=b^q$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Weighted Young's inequality)</span></p>

For all $a,b\ge 0$ and any $\varepsilon>0$,

$$
ab \;\le\; \frac{1}{2\varepsilon}\,a^2 + \frac{\varepsilon}{2}\,b^2, \tag{Y$_\varepsilon$}
$$

with equality iff $a=\varepsilon b$. More generally, for conjugate exponents $p,q$ as above and any $\varepsilon>0$,

$$
ab \;\le\; \frac{a^p}{p\,\varepsilon^{p-1}} + \frac{\varepsilon\, b^q}{q}.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why the weight $\varepsilon$ matters)</span></p>

The weighted form is just (Y) applied to the rescaled pair $(a/\sqrt\varepsilon,\,b\sqrt\varepsilon)$ — same content, same proof. The point is the **free parameter** $\varepsilon$:

* Choosing $\varepsilon$ small concentrates the weight on $a^2$ (with cost $1/\varepsilon$ amplifying it).
* Choosing $\varepsilon$ large concentrates the weight on $b^2$ (with cost $\varepsilon$ amplifying it).

In practice, $\varepsilon$ is chosen so that one of the two squared terms — typically the $b^2$ term — *cancels exactly* a quantity already present in the inequality with the opposite sign. This was the maneuver in the proof of Theorem 3, where $\varepsilon=\lambda$ was chosen so that $\tfrac\varepsilon2 b^2 = \tfrac\lambda2\|x-x^\ast\|^2$ cancelled the $-\tfrac\lambda2\|x-x^\ast\|^2$ supplied by uniform convexity. The same trick reappears in the proof of Theorem 6.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Gronwall's inequality, differential form)</span></p>

Let $u:[0,T]\to\mathbb R$ be absolutely continuous and let $\alpha:[0,T]\to\mathbb R$ be locally integrable. If

$$
\dot u(t) \;\le\; \alpha(t)\,u(t) \qquad \text{for a.e. } t\in[0,T], \tag{G}
$$

then

$$
u(t) \;\le\; u(0)\,\exp\!\Bigl(\int_0^t \alpha(s)\,ds\Bigr) \qquad \text{for all } t\in[0,T].
$$

In particular, if $\alpha\equiv -c$ is a negative constant,

$$
\dot u(t) \le -c\,u(t) \quad\Longrightarrow\quad u(t) \le u(0)\,e^{-c t}.
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Gronwall — the integrating-factor trick</summary>

Set $F(t):=\exp\bigl(-\int_0^t\alpha(s)\,ds\bigr)>0$. Then $F$ is absolutely continuous with $\dot F(t) = -\alpha(t)\,F(t)$. Compute the derivative of the product $u(t)F(t)$ using the product rule:

$$
\frac{d}{dt}\bigl(u(t)\,F(t)\bigr) \;=\; \dot u(t)\,F(t) + u(t)\,\dot F(t) \;=\; \bigl(\dot u(t) - \alpha(t)\,u(t)\bigr)\,F(t) \;\le\; 0,
$$

where the last step uses (G) and $F>0$. So $t\mapsto u(t)F(t)$ is non-increasing, giving $u(t)F(t)\le u(0)F(0)=u(0)$, i.e.

$$
u(t)\;\le\; \frac{u(0)}{F(t)} \;=\; u(0)\,\exp\!\Bigl(\int_0^t\alpha(s)\,ds\Bigr). \qquad\square
$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Variants of Gronwall)</span></p>

Gronwall's inequality has several common variants, all built on the same integrating-factor trick:

* **Integral form.** If $u(t)\le \alpha + \int_0^t \beta(s)\,u(s)\,ds$ for $\alpha\ge 0$ and $\beta\ge 0$ locally integrable, then $u(t)\le \alpha\exp\bigl(\int_0^t\beta(s)\,ds\bigr)$. This is the version most often invoked in ODE/PDE existence theory.
* **Nonlinear (Bihari–LaSalle).** If $\dot u\le f(u)$ for some non-decreasing $f>0$, then $u$ is bounded by the solution of $\dot v=f(v)$ with the same initial data. Yields polynomial/algebraic decay when $f$ is nonlinear — this is implicitly the engine behind Theorem 5, where $\dot{\mathcal E}\le -\mathcal E^2/\mathcal H(0)$ leads to the $1/t$ rate (the linear Gronwall would have given exponential decay, which is unavailable here).
* **Discrete form.** If $a_{n+1}\le (1+\alpha_n)\,a_n$ with $\alpha_n\ge 0$, then $a_n\le a_0\,\prod_{k=0}^{n-1}(1+\alpha_k)\le a_0\,\exp\bigl(\sum_{k=0}^{n-1}\alpha_k\bigr)$. This is the version one typically meets when proving stability of numerical schemes.

In every variant, the moral is the same: a self-referential bound of the shape "$u$ is controlled by something involving $u$ itself" can be unwound into a closed-form bound by an integrating factor.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Uniformly convex function)</span></p>

A differentiable $E:\mathbb R^N\to\mathbb R$ is **uniformly convex** with parameter $\lambda>0$ if

$$
E(y) \;\ge\; E(x) + \langle \nabla E(x),\,y-x\rangle + \tfrac12\lambda\,|y-x|^2
\qquad\text{for all } x,y\in\mathbb R^N. \tag{1.16}
$$

Equivalently (in the smooth case), $\nabla^2 E(x)\succeq \lambda\,\mathrm{Id}$ for all $x$, i.e. all eigenvalues of the Hessian are bounded below by $\lambda$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric meaning of uniform convexity)</span></p>

Plain convexity says the graph of $E$ lies above every tangent plane:

$$E(y) \ge E(x) + \langle \nabla E(x),\,y-x\rangle.$$

**Uniform convexity** strengthens this by requiring the graph to lie above the tangent plane *plus a quadratic margin* $\tfrac12\lambda\|y-x\|^2$:

* The margin grows quadratically with the displacement, which means $E$ is "lifted" off its tangents by a fixed parabola from below — the function is no flatter than a quadratic of curvature $\lambda$.
* In particular, $E$ has a **unique** global minimizer $x^\ast$, and $E$ grows at least quadratically away from it: $E(y)\ge E(x^\ast)+\tfrac12\lambda\|y-x^\ast\|^2$ (apply (1.16) with $x=x^\ast$, where $\nabla E(x^\ast)=0$).
* The prototype is $E(x)=\tfrac12\lambda\|x\|^2$, for which (1.16) holds with equality. Adding any convex function preserves uniform convexity with the same $\lambda$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3</span><span class="math-callout__name">(Long-term asymptotics from uniform convexity)</span></p>

Assume $E$ is differentiable and uniformly convex in the sense of (1.16) with parameter $\lambda>0$. Let $x^\ast := \arg\min_{x\in\mathbb R^N} E(x)$, and let $x:[0,\infty)\to\mathbb R^N$ be the unique solution to (1.1). Then

$$
|x(t)-x^\ast|^2 \;\le\; \tfrac{2}{\lambda}\,e^{-2\lambda t}\bigl(E(x_0)-E(x^\ast)\bigr). \tag{1.17}
$$

In other words, the trajectory of the gradient flow converges to the global minimizer **exponentially fast**, with rate $\lambda$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 3 — Gronwall on the excess energy</summary>

The strategy is to introduce two scalar functionals along the trajectory, derive a differential inequality between them, and apply Gronwall.

**Step 1 — excess energy and dissipation.** Define

$$
\mathcal E(t) := E(x(t)) - E(x^\ast)\ge 0,
\qquad
\mathcal D(t) := |\nabla E(x(t))|^2 \ge 0.
$$

Apply uniform convexity (1.16) with $x=x(t)$ and $y=x^\ast$:

$$
E(x^\ast) \;\ge\; E(x(t)) + \langle \nabla E(x(t)),\,x^\ast - x(t)\rangle + \tfrac12\lambda\,|x^\ast - x(t)|^2.
$$

Rearranging and using Cauchy–Schwarz,

$$
\mathcal E(t)
\;\le\; \langle \nabla E(x(t)),\,x(t)-x^\ast\rangle - \tfrac12\lambda\,|x^\ast - x(t)|^2
\;\le\; |\nabla E(x(t))|\,|x(t)-x^\ast| - \tfrac12\lambda\,|x^\ast - x(t)|^2.
$$

Now apply **Young's inequality** $ab\le \tfrac{1}{2\lambda}a^2+\tfrac{\lambda}{2}b^2$ to the first term with $a=\|\nabla E(x(t))\|$ and $b=\|x(t)-x^\ast\|$. The two $\tfrac{\lambda}{2}\|x-x^\ast\|^2$ terms cancel and we are left with

$$
\mathcal E(t)\;\le\; \tfrac{1}{2\lambda}|\nabla E(x(t))|^2 \;=\; \tfrac{1}{2\lambda}\,\mathcal D(t).
$$

**Step 2 — differential inequality.** By the chain rule and the gradient-flow equation (1.1),

$$
\frac{d}{dt}\mathcal E(t)
\;=\; \langle \nabla E(x(t)),\,\dot x(t)\rangle
\;=\; -|\nabla E(x(t))|^2
\;=\; -\mathcal D(t)
\;\le\; -2\lambda\,\mathcal E(t).
$$

**Step 3 — Gronwall.** The differential inequality $\dot{\mathcal E} \le -2\lambda \mathcal E$ implies

$$
\mathcal E(t) \;\le\; e^{-2\lambda t}\,\mathcal E(0).
$$

**Step 4 — translate energy into distance.** Apply uniform convexity (1.16) once more, with the roles reversed: $x=x^\ast$ and $y=x(t)$. Since $\nabla E(x^\ast)=0$, (1.16) reduces to

$$
E(x(t)) \;\ge\; E(x^\ast) + \tfrac12\lambda\,|x(t)-x^\ast|^2,
$$

i.e. $\tfrac12\lambda\,\|x(t)-x^\ast\|^2 \le \mathcal E(t)$. Combined with Step 3,

$$
|x(t)-x^\ast|^2 \;\le\; \tfrac{2}{\lambda}\,\mathcal E(t) \;\le\; \tfrac{2}{\lambda}\,e^{-2\lambda t}\bigl(E(x_0)-E(x^\ast)\bigr).
$$

This is (1.17). $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why uniform convexity is crucial)</span></p>

The proof of Theorem 3 used uniform convexity twice — once to bound $\mathcal E$ in terms of $\mathcal D$, and once to translate energy into distance. Both steps used the **quadratic margin** $\tfrac12\lambda\|y-x\|^2$. Without it, the differential inequality $\dot{\mathcal E}\le -2\lambda\mathcal E$ collapses to merely $\dot{\mathcal E}\le -\mathcal D$ — true but useless on its own, since $\mathcal D$ may decay arbitrarily fast (or slow).

For a merely convex $E$, convergence to equilibrium is therefore **slower** — typically $1/t$, as we will see in Theorem 5 — and one cannot in general single out a unique limit point: the minimizer set $\arg\min E$ may be a non-trivial convex set rather than a single point.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Convex but not uniformly convex)</span></p>

Give an example of a convex (but not uniformly convex) energy $E:\mathbb R\to[0,\infty)$ such that different initial conditions lead to different long-term limits.

*Hint.* Either let $E$ be flat on a non-trivial interval (e.g. $E(x)=\max\lbrace 0,\|x\|-1\rbrace$), or let $E$ be strictly convex but not uniformly so (e.g. $E(x)=x^4/4$, where the Hessian degenerates at $x=0$).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5</span><span class="math-callout__name">(Long-term asymptotics from convexity)</span></p>

Let $E\in C^2(\mathbb R^N)$ be convex, let $x^\ast\in\arg\min_{x\in\mathbb R^N}E(x)$, and let $x:[0,\infty)\to\mathbb R^N$ be the unique solution to (1.1). Define

$$
\mathcal E(t):=E(x(t))-E(x^\ast),\qquad
\mathcal D(t):=-\frac{d}{dt}\mathcal E(t)=|\nabla E(x(t))|^2,\qquad
\mathcal H(t):=|x(t)-x^\ast|^2.
$$

Then

$$
\mathcal E(t) \;\le\; \min\Bigl\lbrace\mathcal E(0),\; \tfrac{\mathcal H(0)}{t}\Bigr\rbrace, \qquad
\mathcal D(t) \;\le\; \tfrac{4\,\mathcal H(0)}{t^2}, \qquad
\mathcal H(t) \;\le\; \mathcal H(0). \tag{1.22}
$$

In particular, $\mathcal E(t)=O(1/t)$ and $\mathcal D(t)=O(1/t^2)$ as $t\to\infty$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 5 — three differential and one algebraic relation</summary>

The proof has three steps: (i) check that $\mathcal E,\mathcal D,\mathcal H$ are all non-increasing along the flow, (ii) derive an algebraic comparison $\mathcal E^2\le\mathcal H\,\mathcal D$, and (iii) combine to extract the rate.

**Step 1 — differential relations.** We claim

$$
\frac{d\mathcal E}{dt}\le -\mathcal D, \tag{1.18}
$$

$$
\frac{d\mathcal D}{dt}\le 0, \tag{1.19}
$$

$$
\frac{d\mathcal H}{dt}\le 0. \tag{1.20}
$$

The first holds with **equality** by (1.1) and the chain rule, $\dot{\mathcal E}=\langle\nabla E,\dot x\rangle=-\|\nabla E\|^2$. For the second, differentiating $\mathcal D=\|\nabla E(x)\|^2$ once more,

$$
\frac{d\mathcal D}{dt} \;=\; 2\bigl\langle \nabla E(x(t)),\,\nabla^2 E(x(t))\,\dot x(t)\bigr\rangle \;=\; -2\,\nabla^2 E(x(t))\bigl[\nabla E(x(t)),\nabla E(x(t))\bigr] \;\le\; 0,
$$

since $\nabla^2 E\succeq 0$ by convexity. The third specialises Theorem 2: $\|x_1-x_2\|$ is non-increasing for any pair of solutions, so applying it to $x_2\equiv x^\ast$ (a stationary solution) gives $\mathcal H(t)\le \mathcal H(0)$ and indeed $\dot{\mathcal H}\le 0$.

**Step 2 — algebraic relation.** We claim

$$
\mathcal E \;\le\; (\mathcal H\,\mathcal D)^{1/2}. \tag{1.21}
$$

Indeed, by **convexity** $E(y)\ge E(x)+\langle\nabla E(x),y-x\rangle$ applied with $x=x(t),\,y=x^\ast$,

$$
\mathcal E(t) \;=\; E(x(t))-E(x^\ast) \;\le\; \langle \nabla E(x(t)),\,x(t)-x^\ast\rangle \;\le\; |\nabla E(x(t))|\,|x(t)-x^\ast| \;=\; \bigl(\mathcal D\,\mathcal H\bigr)^{1/2}.
$$

**Step 3 — decay estimates.** Combine (1.18), (1.21), and (1.20):

$$
\frac{d\mathcal E}{dt} \;\stackrel{(1.18)}{\le}\; -\mathcal D \;\stackrel{(1.21)}{\le}\; -\frac{\mathcal E^2}{\mathcal H}\;\stackrel{(1.20)}{\le}\;-\frac{\mathcal E^2}{\mathcal H(0)}.
$$

Set $u:=1/\mathcal E$ (assuming $\mathcal E>0$; else there is nothing to show). Then $\dot u = -\dot{\mathcal E}/\mathcal E^2 \ge 1/\mathcal H(0)$, so $u(t)\ge u(0)+t/\mathcal H(0)\ge t/\mathcal H(0)$, i.e.

$$
\mathcal E(t) \;\le\; \frac{\mathcal H(0)}{t}.\tag{1.22a}
$$

The complementary bound $\mathcal E(t)\le\mathcal E(0)$ follows from (1.18) and $\mathcal D\ge 0$.

For $\mathcal D$, integrate (1.19) — which says $\mathcal D$ is non-increasing — and note that for any $0<t<T$,

$$
(T-t)\,\mathcal D(T) \;\stackrel{(1.19)}{\le}\; \int_t^T \mathcal D(\tau)\,d\tau \;=\; \mathcal E(t)-\mathcal E(T) \;\le\; \mathcal E(t) \;\stackrel{(1.22a)}{\le}\; \frac{\mathcal H(0)}{t}.
$$

Hence $\mathcal D(T)\le\mathcal H(0)/(t(T-t))$ for any $T>t$. Choose $T=2t$ to get $\mathcal D(2t)\le\mathcal H(0)/t^2$; rescaling $t\mapsto t/2$ yields $\mathcal D(t)\le 4\mathcal H(0)/t^2$.

Finally, $\mathcal H(t)\le\mathcal H(0)$ is just (1.20). $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why the rates worsen from $\lambda$ to $1/t$)</span></p>

Compare the two regimes:

* **Uniform convexity** gave $\dot{\mathcal E}\le -2\lambda\mathcal E$ — a *linear* differential inequality, whose solution decays exponentially in $t$.
* **Plain convexity** gives only $\dot{\mathcal E}\le -\mathcal E^2/\mathcal H(0)$ — a *nonlinear* differential inequality that integrates to $1/t$ decay.

The shift is the same as the one between the gradient flow of $E(x)=\tfrac12\lambda x^2$ (linear ODE, exponential decay) and that of $E(x)=\tfrac14 x^4$ (nonlinear ODE, $\dot x = -x^3$, algebraic decay). Without the quadratic margin of uniform convexity, the energy can become "flat" near the minimizer, the gradient becomes too small to produce a fixed exponential rate, and we lose the cleanest version of Gronwall.

</div>

#### Long-term asymptotics via the Łojasiewicz inequality

So far, both Theorem 3 and Theorem 5 used **convexity** to prove convergence — Theorem 3 to a unique global minimizer, Theorem 5 to a (possibly non-unique) minimizer with a quantitative rate. There is a strikingly different approach — pioneered by **Stanisław Łojasiewicz** in 1963 in the analytic setting, and extended to many non-analytic settings since — that proves convergence to a *single* limit point under a purely **local** inequality between the energy gap and the gradient. No convexity is required.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Łojasiewicz inequality at a point)</span></p>

A continuously differentiable $E:\mathbb R^N\to\mathbb R$ is said to satisfy the **Łojasiewicz inequality** at $x_\ast\in\mathbb R^N$ if there exist a neighborhood $U$ of $x_\ast$, a constant $C<\infty$, and an exponent $\theta\in(0,1)$ such that

$$
|E(y)-E(x_\ast)|^\theta \;\le\; C\,|\nabla E(y)| \qquad \text{for all } y\in U. \tag{Ł}
$$

The exponent $\theta$ is called the **Łojasiewicz exponent** at $x_\ast$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(When is (Ł) a non-trivial constraint?)</span></p>

(Ł) is only a *genuine* constraint at **critical points** $x_\ast$ (where $\nabla E(x_\ast)=0$):

* **Non-critical $x_\ast$.** If $\nabla E(x_\ast)\ne 0$, then $\|\nabla E(y)\|$ is bounded away from $0$ on a small enough neighborhood, while $\|E(y)-E(x_\ast)\|$ is small. So the inequality holds trivially with any $\theta$ and a suitable $C$.
* **Critical $x_\ast$.** If $\nabla E(x_\ast)=0$, the right-hand side vanishes at $y=x_\ast$, and (Ł) becomes a quantitative statement: the energy gap $\|E(y)-E(x_\ast)\|$ is no flatter than $\|\nabla E(y)\|^{1/\theta}$ near $x_\ast$.

The exponent $\theta$ encodes how flat the critical point is allowed to be:

* $\theta=\tfrac12$ corresponds to a **non-degenerate** critical point — the energy looks locally quadratic, $E(y)-E(x_\ast)\sim\|y-x_\ast\|^2$, and (Ł) reduces to $\|y-x_\ast\|\lesssim\|\nabla E(y)\|$. This is essentially the local form of uniform convexity (Theorem 3) and yields *exponential* convergence.
* $\theta\to 1^-$ corresponds to **very flat** critical points (e.g. $E(y)=\|y\|^k$ for large $k$). The trajectory still converges, but arbitrarily slowly.

A landmark theorem of **Łojasiewicz (1963)** asserts that *every real-analytic function* $E$ satisfies (Ł) at every critical point with some $\theta\in(0,1)$. This is what makes the inequality applicable in essentially all "natural" optimization landscapes — including, with various caveats, the loss surfaces that arise in deep learning.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Long-term asymptotics via the Łojasiewicz inequality)</span></p>

Let $E\in C^1(\mathbb R^N)$ and let $x:[0,\infty)\to\mathbb R^N$ solve the gradient flow (1.1). Suppose there exist a constant $E_\infty\in\mathbb R$, a constant $C<\infty$, and an exponent $\theta\in(0,1)$ such that

$$
|E(x(t))-E_\infty|^\theta \;\le\; C\,|\nabla E(x(t))| \qquad \text{for all sufficiently large } t. \tag{Ł'}
$$

Then the curve $t\mapsto x(t)$ has **finite length**,

$$
\int_0^\infty |\dot x(t)|\,dt \;<\; \infty,
$$

and consequently $x(t)$ converges to some $x^\ast\in\mathbb R^N$ as $t\to\infty$. Moreover, $x^\ast$ is a **critical point** of $E$, i.e. $\nabla E(x^\ast)=0$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof — differentiating $\mathcal E^{1-\theta}$ along the flow</summary>

The trick is to track *not* the excess energy $\mathcal E(t):=E(x(t))-E_\infty$ itself, but its concave power $\mathcal E^{1-\theta}$. The exponent $1-\theta\in(0,1)$ is the bridge that turns Łojasiewicz into a finite-length statement. This is the canonical instance of a **desingularizing function**; the general framework, and why the exponent is forced rather than chosen, is unpacked in [Appendix A](#appendix-a).

**Setup.** By energy decrease (1.2), $E(x(t))$ is non-increasing in $t$; since $E\ge 0$, the limit $E_\infty := \lim_{t\to\infty}E(x(t))$ exists. Set $\mathcal E(t):=E(x(t))-E_\infty\ge 0$. Note $\mathcal E$ is non-increasing in $t$, so in particular $\mathcal E^{1-\theta}$ is well-defined and non-increasing (we set $\mathcal E^{1-\theta}=0$ if $\mathcal E=0$).

**Compute the derivative of $\mathcal E^{1-\theta}$.** On the set where $\mathcal E>0$,

$$
\frac{d}{dt}\mathcal E^{1-\theta}(t)
\;=\; (1-\theta)\,\mathcal E^{-\theta}(t)\,\frac{d\mathcal E}{dt}
\;=\; (1-\theta)\,\mathcal E^{-\theta}(t)\,\langle\nabla E(x(t)),\dot x(t)\rangle.
$$

On the gradient flow $\dot x=-\nabla E$, so $\langle\nabla E,\dot x\rangle = -\|\nabla E\|^2 = -\|\nabla E\|\,\|\dot x\|$ (since $\|\dot x\|=\|\nabla E\|$ on (1.1)). Hence

$$
\frac{d}{dt}\mathcal E^{1-\theta}(t) \;=\; -(1-\theta)\,\mathcal E^{-\theta}(t)\,|\nabla E(x(t))|\,|\dot x(t)|.
$$

**Apply Łojasiewicz.** Hypothesis (Ł') gives $\mathcal E(t)^\theta\le C\,\|\nabla E(x(t))\|$, i.e. $\mathcal E^{-\theta}(t)\,\|\nabla E(x(t))\|\ge 1/C$. Substituting,

$$
\frac{d}{dt}\mathcal E^{1-\theta}(t) \;\le\; -\frac{1-\theta}{C}\,|\dot x(t)|.
$$

**Integrate.** Since $\mathcal E^{1-\theta}$ is non-increasing and bounded below by $0$, $\lim_{t\to\infty}\mathcal E^{1-\theta}(t)=:L\ge 0$ exists. Integrating from $0$ to $\infty$,

$$
\frac{1-\theta}{C}\int_0^\infty |\dot x(t)|\,dt \;\le\; \mathcal E^{1-\theta}(0)-L \;\le\; \mathcal E^{1-\theta}(0) \;<\; \infty.
$$

So $\int_0^\infty\|\dot x\|\,dt<\infty$ — the trajectory has *finite length*.

**Limit point and criticality.** Finite length means $x(t)$ is Cauchy: for any $\varepsilon>0$, choose $T$ so that $\int_T^\infty\|\dot x\|<\varepsilon$; then for $t,s>T$,

$$
|x(t)-x(s)|\;\le\; \int_s^t|\dot x(\tau)|\,d\tau \;<\;\varepsilon.
$$

Hence $x^\ast := \lim_{t\to\infty}x(t)\in\mathbb R^N$ exists. Moreover, on the gradient flow $\|\nabla E(x(t))\|=\|\dot x(t)\|$, so

$$
\int_0^\infty |\nabla E(x(t))|\,dt \;=\; \int_0^\infty |\dot x(t)|\,dt \;<\; \infty.
$$

A continuous, non-negative function on $[0,\infty)$ whose integral is finite must have $\liminf_{t\to\infty}=0$. Combined with continuity of $\nabla E$ and $x(t)\to x^\ast$, this forces $\|\nabla E(x^\ast)\|=0$. So $x^\ast$ is a critical point. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What Łojasiewicz buys us, beyond convexity)</span></p>

* **No convexity needed.** The proof uses only differentiability and the local inequality (Ł') — there is no convexity assumption on $E$ anywhere. This is decisive in non-convex settings (deep-network losses, phase-field energies, semilinear PDEs) where convexity simply fails.
* **Convergence to a single point.** Theorem 5 only gave $\mathcal E(t),\mathcal D(t)\to 0$, and even when $E$ is strictly convex, the minimizer set may have many points (Exercise 2). Łojasiewicz pins down a *single* limit point $x^\ast$ — at the cost of finite-length, not rate.
* **No rate, in general.** The proof gives finite length but **no quantitative rate** of convergence. To extract a rate, one needs to know the Łojasiewicz exponent $\theta$ at the limit point: $\theta=\tfrac12$ recovers exponential convergence (the "tame" case), while $\theta\in(\tfrac12,1)$ gives algebraic decay $\|x(t)-x^\ast\|=O\bigl(t^{-(1-\theta)/(2\theta-1)}\bigr)$. Computing $\theta$ for a given energy is in general hard.
* **The role of $\mathcal E^{1-\theta}$.** Why does the proof differentiate $\mathcal E^{1-\theta}$ rather than $\mathcal E$ itself? Because the Łojasiewicz inequality controls $\mathcal E^\theta$ by $\|\nabla E\|$, so the natural scalar quantity whose decay rate is dimensionally homogeneous to $\|\dot x\|$ — and therefore integrates to the *length* of the trajectory rather than to its squared $L^2$ norm — is $\mathcal E^{1-\theta}$. This concave reparametrization is the heart of the trick, and a special case of a recurring template called the **Kurdyka–Łojasiewicz desingularization**. See [Appendix A](#appendix-a) for the general principle and a list of other places where the same idea reappears.

</div>

### 1.7 Convergence rates for minimizing movements

We now compare the **discrete** minimizing-movements scheme (1.10) to the **continuous** gradient flow (1.1), under the same convexity assumption that made the long-term asymptotics work. The result quantifies what we already saw in §1.4 informally: as $h\downarrow 0$, the discrete iterates $\chi_h^{(\ell)}$ converge to the continuous trajectory $x(\ell h)$, and the rate is $O(h)$.

A subtle point on logical structure: the result below assumes a-priori that a continuous solution $x$ to (1.1) exists. We will actually prove existence rigorously in §1.8 (Theorem 8) — but the convergence-rate proof here uses only the gradient-flow equation and convexity, not existence per se, so the two arguments are logically independent.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6</span><span class="math-callout__name">(Quantitative convergence of minimizing movements; Rulla '96, Nochetto–Savaré–Verdi '00)</span></p>

Let $E$ be convex and differentiable, let $x:[0,\infty)\to\mathbb R^N$ solve the gradient-flow equation (1.1), and let $\chi_\ell:=\chi_h^{(\ell)}$ be the iterates of the minimizing-movements scheme (1.10) starting from the same initial condition $\chi_h^{(0)}=x_0=x(0)$. Then

$$
\sup_{\ell\in\mathbb N}\,|\chi_\ell - x(\ell h)| \;\le\; \frac{h}{\sqrt 2}\,|\nabla E(x_0)|. \tag{1.23}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On the constant $\tfrac1{\sqrt 2}$ and on regularity)</span></p>

The bound says the discrete trajectory is within $O(h)$ of the continuous one *uniformly in $\ell$* — not only on a finite time window. Of course, under stricter regularity assumptions (e.g. $E\in C^2$ with bounded Hessian), classical numerical-analysis arguments give the same $O(h)$ rate. The point of this proof is that **convexity alone** suffices: no Hessian bound, no Lipschitz constant for $\nabla E$, no time-cutoff $T<\infty$. Convexity does the work of regularity.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 6 — telescoping a convex-combination energy</summary>

We pursue a one-step bound and iterate.

**Step 1 — reduction to one step.** It suffices to prove

$$
|\chi_\ell - x(\ell h)|^2 + \tfrac{h^2}{2}\,|\nabla E(\chi_\ell)|^2 \;\le\; |\chi_{\ell-1}-x((\ell-1)h)|^2 + \tfrac{h^2}{2}\,|\nabla E(\chi_{\ell-1})|^2. \tag{1.24}
$$

Iterating (1.24) from $\ell=L$ down to $\ell=1$ and using $\chi_0=x(0)$,

$$
|\chi_L-x(Lh)|^2 + \tfrac{h^2}{2}|\nabla E(\chi_L)|^2 \;\le\; \tfrac{h^2}{2}|\nabla E(x_0)|^2,
$$

which gives $\|\chi_L-x(Lh)\|^2\le \tfrac{h^2}{2}\|\nabla E(x_0)\|^2$, i.e. (1.23).

**Step 2 — one-step argument.** Without loss of generality $\ell=1$. Define the **convex-combination error**

$$
e(t) \;:=\; \frac{t}{h}\cdot\tfrac12|\chi_1-x(t)|^2 + \frac{h-t}{h}\cdot\tfrac12|x_0-x(t)|^2, \qquad t\in[0,h]. \tag{1.25}
$$

So $e(0)=\tfrac12\|x_0-x(0)\|^2=0$ (under matched initial conditions) and $e(h)=\tfrac12\|\chi_1-x(h)\|^2$.

Differentiate $e$ in $t$:

$$
\dot e(t) = \tfrac{1}{2h}|\chi_1-x(t)|^2 - \tfrac{1}{2h}|x_0-x(t)|^2 + \tfrac{t}{h}\langle\chi_1-x(t),-\dot x(t)\rangle + \tfrac{h-t}{h}\langle x_0-x(t),-\dot x(t)\rangle.
$$

*First two terms.* Using the polarization identity $\|a\|^2-\|b\|^2=\langle a-b,a+b\rangle$ with $a=\chi_1-x(t),\ b=x_0-x(t)$:

$$
\tfrac{1}{2h}\bigl(|\chi_1-x(t)|^2-|x_0-x(t)|^2\bigr) \;=\; \tfrac12\bigl\langle \tfrac{\chi_1-x_0}{h},\,2(\chi_1-x(t))-(\chi_1-x_0)\bigr\rangle.
$$

Apply the **Euler–Lagrange equation** (1.13), $\nabla E(\chi_1)+\tfrac{\chi_1-x_0}{h}=0$, i.e. $\tfrac{\chi_1-x_0}{h}=-\nabla E(\chi_1)$ and $\chi_1-x_0=-h\nabla E(\chi_1)$. Substituting,

$$
=\;-\tfrac12\bigl\langle\nabla E(\chi_1),\,2(\chi_1-x(t))+h\nabla E(\chi_1)\bigr\rangle \;=\; -\langle\nabla E(\chi_1),\chi_1-x(t)\rangle - \tfrac{h}{2}|\nabla E(\chi_1)|^2.
$$

By convexity $E(x(t))\ge E(\chi_1)+\langle\nabla E(\chi_1),x(t)-\chi_1\rangle$, so $-\langle\nabla E(\chi_1),\chi_1-x(t)\rangle\le E(x(t))-E(\chi_1)$. Hence

$$
\tfrac{1}{2h}\bigl(|\chi_1-x(t)|^2-|x_0-x(t)|^2\bigr) \;\le\; -\tfrac{h}{2}|\nabla E(\chi_1)|^2 + E(x(t))-E(\chi_1).
$$

*Last two terms.* Use $\dot x(t)=-\nabla E(x(t))$ and convexity $\langle\nabla E(x(t)),y-x(t)\rangle\le E(y)-E(x(t))$ once for $y=\chi_1$ and once for $y=x_0$:

$$
\tfrac{t}{h}\langle\chi_1-x(t),-\dot x(t)\rangle + \tfrac{h-t}{h}\langle x_0-x(t),-\dot x(t)\rangle \;\le\; \tfrac{t}{h}E(\chi_1) + \tfrac{h-t}{h}E(x_0) - E(x(t)).
$$

*Combining.* The two estimates add to

$$
\dot e(t) \;\le\; -\tfrac{h}{2}|\nabla E(\chi_1)|^2 + \tfrac{h-t}{h}\bigl(E(x_0)-E(\chi_1)\bigr).
$$

*Bound the energy gap.* Apply convexity in the form $E(\chi_1)\ge E(x_0)+\langle\nabla E(x_0),\chi_1-x_0\rangle$ together with EL ($\chi_1-x_0=-h\nabla E(\chi_1)$):

$$
E(x_0)-E(\chi_1) \;\le\; -\langle\nabla E(x_0),\chi_1-x_0\rangle \;=\; h\langle\nabla E(x_0),\nabla E(\chi_1)\rangle \;\le\; \tfrac{h}{2}|\nabla E(x_0)|^2 + \tfrac{h}{2}|\nabla E(\chi_1)|^2,
$$

where the last step is Cauchy–Schwarz followed by Young's inequality. Hence

$$
\dot e(t) \;\le\; -\tfrac{h}{2}|\nabla E(\chi_1)|^2 + (h-t)\,\Bigl(\tfrac12|\nabla E(x_0)|^2+\tfrac12|\nabla E(\chi_1)|^2\Bigr).
$$

**Step 3 — integrate from $0$ to $h$.** Since $\int_0^h(h-t)\,dt=h^2/2$,

$$
e(h)-e(0) \;\le\; -\tfrac{h^2}{2}|\nabla E(\chi_1)|^2 + \tfrac{h^2}{4}|\nabla E(x_0)|^2 + \tfrac{h^2}{4}|\nabla E(\chi_1)|^2 \;=\; \tfrac{h^2}{4}|\nabla E(x_0)|^2 - \tfrac{h^2}{4}|\nabla E(\chi_1)|^2.
$$

Substituting $e(0)=0,\ e(h)=\tfrac12\|\chi_1-x(h)\|^2$, multiplying by $2$ and reordering yields (1.24) for $\ell=1$. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Mismatched initial conditions)</span></p>

Inspect the proof above and state the corresponding result when the initial conditions of the discrete and continuous schemes differ: $\chi_h^{(0)}\ne x(0)$. How does the resulting estimate relate to Theorem 2 (uniqueness/stability)?

</div>

### 1.8 Existence via minimizing movements

We now make rigorous the existence claim previewed informally as Theorem 1 in §1.4. The strategy is the standard **compactness + passage to the limit** argument:

1. **A priori estimate.** Use the energy-dissipation identity along the discrete iterates to bound — uniformly in $h$ — the discrete derivative of an interpolation $\tilde x_h$.
2. **Compactness.** The bound, combined with Arzelà–Ascoli and reflexivity of $L^2$, yields a subsequence and a limit $x$.
3. **Pass to the limit.** Take $h\downarrow 0$ in the Euler–Lagrange inequality (1.12) (or equation (1.13) in the smooth case), using the lower semi-continuity of $E$.

We work with two interpolations of the discrete data $\chi_h^{(0)},\chi_h^{(1)},\dots$:

* the **piecewise-constant** interpolation $x_h$ defined in §1.4 by (1.11);
* the **piecewise-linear** interpolation

$$
\tilde x_h(t) \;:=\; \frac{t-(\ell-1)h}{h}\,\chi_h^{(\ell)} + \frac{\ell h - t}{h}\,\chi_h^{(\ell-1)} \qquad\text{for } t\in[(\ell-1)h,\ell h]. \tag{1.26}
$$

(This is well-defined because $\mathbb R^N$ is a convex space: the convex combination is itself a point of the ambient space. In a metric or non-Euclidean setting, such a linear interpolation generally fails to make sense.) The piecewise-linear interpolation is differentiable a.e. — its derivative is the discrete velocity $(\chi_h^{(\ell)}-\chi_h^{(\ell-1)})/h$ on each subinterval — which is what we will pass to the limit on. The piecewise-constant interpolation, on the other hand, is the natural "snapshot" of the discrete iterate at time $t$, and the EL equation directly constrains $x_h$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8</span><span class="math-callout__name">(Existence via minimizing movements — formal restatement)</span></p>

Let $E:\mathbb R^N\to[0,+\infty]$ and $x_0\in\mathbb R^N$ with $E(x_0)<+\infty$. Let $x_h$ and $\tilde x_h$ be the piecewise-constant (1.11) and piecewise-linear (1.26) interpolations of the minimizing-movements iterates (1.10), starting from $\chi_h^{(0)}=x_0$. Then:

**(i) Compactness.** The a priori bound

$$
\int_0^\infty \Bigl|\tfrac{d\tilde x_h}{dt}\Bigr|^2\,dt \;\le\; 2\,E(x_0) \tag{1.27}
$$

holds, and for any $0\le s\le t<\infty$,

$$
|\tilde x_h(t)-\tilde x_h(s)| \;\le\; \sqrt{2E(x_0)}\,\sqrt{t-s}. \tag{1.28}
$$

The families $(\tilde x_h)\_{h\in(0,1]}$ and $(x_h)\_{h\in(0,1]}$ are precompact: for any sequence $h\downarrow 0$, there exists a (non-relabeled) subsequence and a limit $x\in C^{1/2}([0,\infty);\mathbb R^N)\cap H^1((0,\infty);\mathbb R^N)$ with

$$
\tilde x_h\to x\quad\text{locally uniformly on }[0,\infty), \tag{1.29}
$$

$$
\frac{d\tilde x_h}{dt}\rightharpoonup \frac{dx}{dt}\quad\text{weakly in }L^2((0,\infty);\mathbb R^N), \tag{1.30}
$$

and analogously

$$
x_h\to x\quad\text{locally uniformly on }[0,\infty), \tag{1.31}
$$

$$
\frac{x_h(\cdot+h)-x_h(\cdot)}{h}\rightharpoonup \frac{dx}{dt}\quad\text{weakly in }L^2((0,\infty);\mathbb R^N). \tag{1.32}
$$

**(ii) Convergence in the equation.**

*(a)* If $E\in C^1(\mathbb R^N)$, then $x$ solves $\dot x(t)=-\nabla E(x(t))$ for all $t>0$.

*(b)* If $E$ is convex (possibly non-smooth), then $x$ satisfies the differential inclusion

$$
\frac{dx}{dt}\in -\partial E(x(t)) \qquad\text{for almost every }t>0. \tag{1.33}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why two interpolations)</span></p>

The seemingly redundant pair $(x_h,\tilde x_h)$ plays distinct roles:

* **$\tilde x_h$ (piecewise-linear)** is *Lipschitz continuous in $t$* with derivative $\frac{d\tilde x_h}{dt}=\frac{\chi_h^{(\ell)}-\chi_h^{(\ell-1)}}{h}$ on $((\ell-1)h,\ell h)$. This is the object whose derivative converges to $\dot x$ in $L^2$.
* **$x_h$ (piecewise-constant)** equals $\chi_h^{(\ell)}$ on $((\ell-1)h,\ell h]$. It is the natural "snapshot" interpolation, and the discrete EL equation (1.12) is a literal statement about $x_h$ (not $\tilde x_h$).

The two interpolations differ only on a set of measure $h$ per subinterval, so in the limit $h\downarrow 0$ they converge to the same continuous $x(t)$. But on the way, we need both: $\tilde x_h$ for the regularity (1.27)–(1.28), and $x_h$ for the EL passage to the limit.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 8</summary>

**Argument for (i) — compactness via discrete energy dissipation.**

From the very definition of $\chi_h^{(\ell)}$ as a minimizer of $E(x)+\tfrac{1}{2h}\|x-\chi_h^{(\ell-1)}\|^2$, comparing the value at $\chi_h^{(\ell)}$ against the value at $\chi_h^{(\ell-1)}$ (which is a candidate point) gives

$$
E(\chi_h^{(\ell)}) + \tfrac{1}{2h}|\chi_h^{(\ell)}-\chi_h^{(\ell-1)}|^2 \;\le\; E(\chi_h^{(\ell-1)}).
$$

Iterating from $\ell=1$ to $\ell=L$ and telescoping,

$$
E(\chi_h^{(L)}) + \tfrac{h}{2}\sum_{\ell=1}^L \Bigl|\tfrac{\chi_h^{(\ell)}-\chi_h^{(\ell-1)}}{h}\Bigr|^2 \;\le\; E(x_0). \tag{1.34}
$$

Note that $\tilde x_h$ is Lipschitz with $\tfrac{d\tilde x_h}{dt}=\tfrac{\chi_h^{(\ell)}-\chi_h^{(\ell-1)}}{h}$ on $((\ell-1)h,\ell h)$, so

$$
h\,\Bigl|\tfrac{\chi_h^{(\ell)}-\chi_h^{(\ell-1)}}{h}\Bigr|^2 = \int_{(\ell-1)h}^{\ell h} \Bigl|\tfrac{d\tilde x_h}{dt}\Bigr|^2\,dt,
$$

and summing,

$$
E(\tilde x_h(Lh)) + \tfrac12\int_0^{Lh}\Bigl|\tfrac{d\tilde x_h}{dt}\Bigr|^2\,dt \;\le\; E(x_0). \tag{1.35}
$$

Since $E\ge 0$, dropping the energy term and letting $L\to\infty$ gives (1.27). The Hölder estimate (1.28) follows from Jensen's (or Cauchy–Schwarz against the constant function $1$):

$$
|\tilde x_h(t)-\tilde x_h(s)| \;\le\; \int_s^t \Bigl|\tfrac{d\tilde x_h}{d\tau}\Bigr|\,d\tau \;\le\; \sqrt{t-s}\,\Bigl(\int_s^t \Bigl|\tfrac{d\tilde x_h}{d\tau}\Bigr|^2 d\tau\Bigr)^{1/2} \;\le\; \sqrt{t-s}\cdot\sqrt{2E(x_0)}.
$$

Now extract a convergent subsequence: fix any $T<\infty$. By (1.28), $(\tilde x_h)$ is uniformly Hölder $C^{1/2}$ on $[0,T]$, hence equicontinuous, hence (by **Arzelà–Ascoli**) precompact in $C([0,T];\mathbb R^N)$. Passing to a subsequence and a diagonal extraction over $T\uparrow\infty$, $\tilde x_h\to x$ locally uniformly on $[0,\infty)$ for some limit $x\in C^{1/2}\cap H^1$. This is (1.29).

For the weak convergence (1.30), the bound (1.27) controls $\tfrac{d\tilde x_h}{dt}$ in $L^2$, so along a further subsequence $\tfrac{d\tilde x_h}{dt}\rightharpoonup f$ weakly in $L^2$ for some $f$. Identify $f=\dot x$ in distributions: for any $\varphi\in C_c^\infty((0,\infty))$,

$$
\int_0^\infty \varphi(t)\,\tfrac{d\tilde x_h}{dt}\,dt = -\int_0^\infty \tilde x_h(t)\,\dot\varphi(t)\,dt \;\xrightarrow{h\downarrow 0}\; -\int_0^\infty x(t)\,\dot\varphi(t)\,dt,
$$

which is precisely the definition of $f$ being the weak derivative of $x$. The corresponding statements (1.31)–(1.32) for $x_h$ and its discrete-difference quotient follow by an analogous argument (left as Exercise 4 below).

**Argument for (ii)(b) — passing to the limit in the EL inequality.**

The Euler–Lagrange inclusion (1.12) of the minimizing-movements scheme reads, for each $\ell$,

$$
-\tfrac{\chi_h^{(\ell)}-\chi_h^{(\ell-1)}}{h} \in \partial E(\chi_h^{(\ell)}).
$$

By the very definition of the subdifferential of a convex function,

$$
0 \;\ge\; E(x_h(t)) - E(y) + \Bigl\langle -\tfrac{x_h(t)-x_h(t-h)}{h},\,y-x_h(t)\Bigr\rangle \quad \text{for all } y\in\mathbb R^N.
$$

Test against $\varphi\in C_c((0,\infty))$ with $\varphi\ge 0$ and integrate:

$$
0 \;\ge\; \int_0^\infty \varphi(t)\Bigl(E(x_h(t)) - E(y) + \Bigl\langle -\tfrac{x_h(t)-x_h(t-h)}{h},\,y-x_h(t)\Bigr\rangle\Bigr)\,dt.
$$

Pass $h\downarrow 0$:

* For the $E(x_h)$ term, use **lower semi-continuity** of $E$ (which holds since $E$ is convex) together with the strong convergence (1.31) and Fatou.
* The discrete-difference quotient converges weakly in $L^2$ to $\dot x$ by (1.32), and $y-x_h(t)\to y-x(t)$ strongly in $L^2_{\mathrm{loc}}$ by (1.31), so the inner product passes to the limit (weak $\times$ strong $\to$ weak).

The result is

$$
0 \;\ge\; \int_0^\infty \varphi(t)\Bigl(E(x(t)) - E(y) + \Bigl\langle -\dot x(t),\,y-x(t)\Bigr\rangle\Bigr)\,dt,
$$

valid for all $\varphi\ge 0$. Hence the bracketed integrand is $\le 0$ for almost every $t\in(0,\infty)$:

$$
0 \;\ge\; E(x(t)) - E(y) + \Bigl\langle -\dot x(t),\,y-x(t)\Bigr\rangle \quad \text{for a.e. } t.
$$

Use this for all $y$ in a countable dense subset $Y\subset\mathbb R^N$. Countable unions of null sets are null, so for almost every $t\in(0,\infty)$ the inequality holds simultaneously for **every** $y\in Y$, and density plus lower semi-continuity extend it to every $y\in\mathbb R^N$. By the definition of the subdifferential, this is exactly $\dot x(t)\in -\partial E(x(t))$ for a.e. $t$, i.e. (1.33).

The smooth case (a) is similar but easier — pass to the limit directly in the equation $\nabla E(x_h)+\tfrac{x_h-x_h(\cdot-h)}{h}=0$ — and is left as Exercise 5. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Convergence of the piecewise-constant interpolation)</span></p>

Building on (1.29) and (1.34), show (1.31) and (1.32). *(Hint: $\|x_h(t)-\tilde x_h(t)\|\le \|\chi_h^{(\ell)}-\chi_h^{(\ell-1)}\|$ on each subinterval, and (1.34) controls the right-hand side in $L^2$.)*

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Smooth case)</span></p>

Show part (ii)(a): if $E\in C^1$, then the limit $x$ solves $\dot x=-\nabla E(x)$ pointwise. *(Hint: pass to the limit in the Euler–Lagrange equation (1.13) using the continuity of $\nabla E$.)*

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sharpness: the missing factor $\tfrac12$)</span></p>

The a priori estimate (1.34) is **not sharp** by a factor of $\tfrac12$. To see this, recall that for a smooth solution of the gradient flow,

$$
\frac{d}{dt}E(x(t)) = \langle \nabla E(x(t)),\dot x(t)\rangle = -|\dot x(t)|^2.
$$

Integrating yields the **energy dissipation identity**

$$
E(x(T)) + \int_0^T |\dot x(t)|^2\,dt = E(x_0). \tag{1.36}
$$

But (1.34) is

$$
E(\chi_h^{(L)}) + \tfrac12\cdot h\sum_{\ell=1}^L \Bigl|\tfrac{\chi_h^{(\ell)}-\chi_h^{(\ell-1)}}{h}\Bigr|^2 \;\le\; E(x_0),
$$

with the factor $\tfrac12$ in front of the dissipation, whereas (1.36) has full unity. So the discrete dissipation is *under-counted* by a factor of $2$ relative to the continuous identity. For a convex energy, a sharper computation recovers the full factor: applying the EL equation $-\tfrac{\chi_h^{(\ell)}-\chi_h^{(\ell-1)}}{h}\in \partial E(\chi_h^{(\ell)})$ in the *definition* of the subdifferential yields

$$
E(\chi_h^{(\ell)}) \;\le\; E(\chi_h^{(\ell-1)}) + \Bigl\langle -\tfrac{\chi_h^{(\ell)}-\chi_h^{(\ell-1)}}{h},\,\chi_h^{(\ell)}-\chi_h^{(\ell-1)}\Bigr\rangle \;=\; E(\chi_h^{(\ell-1)}) - h\,\Bigl|\tfrac{\chi_h^{(\ell)}-\chi_h^{(\ell-1)}}{h}\Bigr|^2.
$$

Summing over $\ell=1,\dots,L$,

$$
E(\chi_h^{(L)}) + h\sum_{\ell=1}^L \Bigl|\tfrac{\chi_h^{(\ell)}-\chi_h^{(\ell-1)}}{h}\Bigr|^2 \;\le\; E(x_0), \tag{1.37}
$$

which exactly resembles (1.36).

This $1/2$ vs.\ $1$ discrepancy turns out to be the same phenomenon as the gap between the "naive" identity (1.2) and the symmetric form (1.39) below — and is the entry point for the **energy-dissipation inequality** of §1.9.

</div>

### 1.9 The energy-dissipation inequality

The identity (1.36) is the cleanest expression of "the gradient flow dissipates energy." We now derive a *stronger* inequality which has the surprising property of being **equivalent** to the gradient-flow equation itself.

Let $T>0$ and $x:[0,T]\to\mathbb R^N$ solve the gradient flow

$$
\begin{cases}\dot x(t) = -\nabla E(x(t)) & \text{for } t\in(0,T),\\ x(0)=x_0,\end{cases} \tag{1.38}
$$

with $E\in C^1(\mathbb R^N)$. On the gradient flow, $\|\dot x\|=\|\nabla E\|$, so $-\|\dot x\|^2 = -\tfrac12\|\dot x\|^2-\tfrac12\|\nabla E\|^2$. Equation (1.2) thus may be rewritten symmetrically as

$$
\frac{d}{dt}E(x(t)) \;=\; \langle \nabla E(x(t)),\dot x(t)\rangle \;=\; -\tfrac12|\dot x(t)|^2 - \tfrac12|\nabla E(x(t))|^2. \tag{1.39}
$$

This is *the same number* as (1.2), just expressed as a symmetric average of two non-negative quantities. Integrating yields the **energy-dissipation inequality (EDI)**

$$
E(x(T)) + \tfrac12\int_0^T|\dot x(t)|^2\,dt + \tfrac12\int_0^T|\nabla E(x(t))|^2\,dt \;\le\; E(x_0). \tag{1.40}
$$

We state (1.40) as an *inequality* because the next lemma will go in the *converse* direction: any curve satisfying (1.40) is automatically a solution of the gradient-flow equation. So the equality "$=$" version is equivalent to the inequality "$\le$" version, and both are equivalent to (1.38).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 9</span><span class="math-callout__name">(EDI implies the gradient-flow equation)</span></p>

Let $E\in C^1(\mathbb R^N)$ and let $x:[0,T]\to\mathbb R^N$ be continuous on $[0,T]$ and differentiable on $(0,T)$ with $x(0)=x_0$. If $x$ satisfies the energy-dissipation inequality (1.40), then $x$ solves the gradient-flow equation (1.38).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 9 — completing the square</summary>

By the fundamental theorem of calculus and the chain rule,

$$
E(x(T))-E(x(0)) \;=\; \int_0^T \frac{d}{dt}E(x(t))\,dt \;=\; \int_0^T\langle\nabla E(x(t)),\dot x(t)\rangle\,dt.
$$

Substituting in (1.40) and rearranging, we get

$$
\int_0^T \Bigl(\langle\nabla E(x(t)),\dot x(t)\rangle + \tfrac12|\dot x(t)|^2 + \tfrac12|\nabla E(x(t))|^2\Bigr)\,dt \;\le\; 0.
$$

The integrand is exactly $\tfrac12\|\dot x(t)+\nabla E(x(t))\|^2 \ge 0$. A non-negative integrand whose integral is $\le 0$ must vanish a.e., so

$$
\dot x(t) = -\nabla E(x(t)) \qquad\text{for a.e. }t\in(0,T).
$$

By continuity of both sides, the equation holds for every $t$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(EDI as a robust definition of "gradient flow")</span></p>

Lemma 9 is the cornerstone of the abstract theory of gradient flows in metric and Banach spaces. The key insight: although (1.38) refers to the *velocity* $\dot x$ (which requires a vector-space structure on the target) and the *gradient* $\nabla E$ (which requires a Riemannian structure), the EDI (1.40) refers only to **scalar quantities**:

* $E(x(t))$ — the energy at a point, defined on any space where $E$ makes sense;
* $\|\dot x(t)\|$ — a *speed*, definable in any metric space via the metric derivative

  $$|\dot x|(t):=\liminf_{s\to t}\frac{d(x(s),x(t))}{|s-t|};$$

* $\|\nabla E(x(t))\|$ — a *slope*, definable in any metric space via the local Lipschitz constant

  $$|\nabla E|(x):=\limsup_{y\to x}\frac{\bigl(E(x)-E(y)\bigr)^+}{d(x,y)}.$$

This is the entry point for **De Giorgi's theory** of *minimizing movements in metric spaces* and the **Ambrosio–Gigli–Savaré** abstract framework for gradient flows in Wasserstein space. We do not pursue this generalization in this course, but the punchline is: *the EDI is the "right" definition of a gradient flow in the sense that it survives outside the smooth Euclidean setting*.

We have already seen the discrete analogue: equation (1.37), the sharp form of the minimizing-movements energy estimate, is the *discrete* EDI. The fact that the naive estimate (1.34) is off by a factor of $2$ but the convex-EL improvement (1.37) is sharp foreshadows the same phenomenon in the continuous case: the gradient-flow equation gives the sharp factor automatically.

</div>

### 1.10 Gradient flows in Riemannian manifolds

Throughout the Euclidean discussion we systematically distinguished between the **differential** $dE(x)$ (a linear functional on $\mathbb R^N$) and the **gradient** $\nabla E(x)$ (a vector in $\mathbb R^N$), tied together by Riesz representation, $dE(x).v=\langle\nabla E(x),v\rangle$. In Euclidean space this distinction was pedantic. On a manifold it becomes essential.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gradient flow on a Riemannian manifold)</span></p>

Let $(M,g)$ be a compact Riemannian manifold, $E\in C^1(M)$, and $x_0\in M$. The **gradient-flow equation** for $E$ on $(M,g)$ is

$$
\dot x(t) = -\nabla E(x(t)) \in T_{x(t)}M, \qquad x(0)=x_0,
$$

where $\nabla E(x)\in T_xM$ is the **gradient** of $E$ at $x$ — i.e. the **unique tangent vector** characterized by

$$
dE(x).v \;=\; g\bigl(\nabla E(x),\,v\bigr) \qquad\text{for all } v\in T_xM. \tag{1.41}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Differential vs. gradient on a manifold)</span></p>

Why is the distinction now unavoidable? Three points:

* **The differential lives in the cotangent space.** $dE(x)$ is a linear functional on $T_xM$, i.e. an element of $T_x^\ast M = (T_xM)^\ast$. It exists as soon as $E$ is differentiable, and depends only on the smooth structure of $M$ — not on any metric.
* **The gradient lives in the tangent space, and depends on the metric.** $\nabla E(x)\in T_xM$ is the **Riesz representative** of $dE(x)$ with respect to the metric $g$. Different metrics on the same $M$ yield different gradients of the same $E$. In coordinates, if $g$ is given by the matrix $(g_{ij})$ with inverse $(g^{ij})$, then $\nabla E$ has components $(\nabla E)^i = g^{ij}\partial_j E$ — the metric "raises the index" on $dE$.
* **The gradient-flow equation lives in the tangent bundle.** Both sides of $\dot x=-\nabla E(x)$ are in $T_{x(t)}M$ — the equation makes sense fibrewise. The differential alone is *not enough*: $dE(x)\in T_x^\ast M$ would not even be of the right type to equate to $\dot x\in T_xM$.

In Euclidean space we silently used the canonical inner product $g_{\text{Eucl}}=\delta_{ij}$, which trivially identifies $T_x\mathbb R^N\cong \mathbb R^N\cong (\mathbb R^N)^\ast\cong T_x^\ast\mathbb R^N$, and the distinction collapses. As soon as we change the metric — even on $\mathbb R^N$, say to a position-dependent quadratic form $g_x(v,w)=\langle v,A(x)w\rangle$ — the gradient $A^{-1}(x)\nabla_{\text{Eucl}} E(x)$ differs from the Euclidean gradient $\nabla_{\text{Eucl}} E(x)$.

This is the entry point for **gradient flows in non-Euclidean geometries** — most notably in **Wasserstein space** $(\mathcal P_2(\mathbb R^N),W_2)$, where the gradient flow of an energy $\mathcal F:\mathcal P_2\to\mathbb R$ recovers a host of evolutionary PDEs (heat equation, Fokker–Planck, porous medium, Keller–Segel, …). We will return to these examples later in the course.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Manifold versions of the previous results)</span></p>

Inspect the proofs of Theorems 3, 5, 6, 8 and Lemma 9 and check which statements generalise to the Riemannian setting. Replace $\|\cdot\|$ by $g(\cdot,\cdot)^{1/2}$, $\nabla E$ by the Riesz representative defined via (1.41), and $\langle\cdot,\cdot\rangle$ by $g(\cdot,\cdot)$. Which steps require *flatness* of the ambient space (e.g. the convex-combination interpolation (1.26)), and which survive the curvature?

</div>

## Chapter 2: Optimal Transport

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(Why optimal transport, here and now)</span></p>

The theory of optimal transportation — nowadays simply **optimal transport** — goes back to **Monge** (1780s), who asked how to move a pile of rubble to an excavation while doing the least amount of work. We will see that, as Monge formulated it, this is a highly **non-linear and hard** problem. In the 1940s, **Kantorovich** found a striking way to *relax* the problem to a **linear program**, and we will see today that this linear problem has a natural **dual**. Both the relaxed primal and its dual are far more tractable than Monge's original formulation. Later in the course, we will see yet another interpretation due to **Benamou and Brenier**, who reveal a **dynamic** (i.e., time-dependent) interpretation of optimal transport in the spirit of continuum mechanics, in which one minimises an action functional. There is still plenty of active research on the subject; this chapter is a *soft* introduction.

This chapter also picks up the thread left dangling at the end of §1.10. Once we are willing to change the geometry on the space we run gradient flows in — e.g. from $\mathbb R^N$ with the Euclidean inner product to the **Wasserstein space** $(\mathcal P_2(\mathbb R^d),W_2)$ — the cost of moving mass around becomes the very metric in which the gradient flow lives. Optimal transport is the bridge.

</div>

### 2.1 Monge's problem

Given a pile of rubble, we want to move it to a prescribed location and shape, minimising the total work. In rigorous terms, we are given two non-negative measures $\mu$ and $\nu$ on Euclidean space $\mathbb R^d$ together with a **cost** function $c=c(x,y)$, and we look for a map $T:\mathbb R^d\to\mathbb R^d$ that transports the mass distributed according to $\mu$ to the mass distributed according to $\nu$, while making the total cost $\int c(x,T(x))\,d\mu(x)$ as small as possible.

The transportation constraint on $T$ is encoded by the **push-forward** of $\mu$ under $T$:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Push-forward measure)</span></p>

Let $\mu$ be a non-negative measure on $\mathbb R^d$ and $T:\mathbb R^d\to\mathbb R^d$ a (Borel-)measurable map. The **push-forward** $T_{\sharp}\mu$ is the measure on $\mathbb R^d$ defined by

$$
T_{\sharp}\mu(A) := \mu\bigl(T^{-1}(A)\bigr) \qquad\text{for every Borel set } A\subset\mathbb R^d,
$$

or, equivalently, by the change-of-variables identity

$$
\int_{\mathbb R^d} \zeta(y)\,d(T_{\sharp}\mu)(y) \;=\; \int_{\mathbb R^d} \zeta(T(x))\,d\mu(x) \qquad\text{for every } \zeta\in C_b(\mathbb R^d).
$$

The constraint **$T_{\sharp}\mu = \nu$** therefore means: for every continuous test function $\zeta$,

$$
\int_{\mathbb R^d}\zeta(T(x))\,d\mu(x) \;=\; \int_{\mathbb R^d}\zeta(y)\,d\nu(y). \tag{*}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(How to read $T_{\sharp}\mu = \nu$ — three equivalent pictures)</span></p>

The push-forward identity is one of those notations whose meaning crystallises only after seeing it three different ways. It is worth pinning down all three, because each will be the right one in different proofs.

* **Set-theoretic.** "The mass that $\nu$ assigns to a target region $A$ equals the mass that $\mu$ assigned to the *pre-image* $T^{-1}(A)$." This is the literal definition $\nu(A)=\mu(T^{-1}(A))$. It says the map $T$ moves *all* of $\mu$ — no mass is lost or created — and the way mass piles up on the target side is governed entirely by where $T$ sends things.
* **Test-function / weak.** Equation (*) is the dual statement: integrating any continuous $\zeta$ against $T_{\sharp}\mu$ is the same as integrating $\zeta\circ T$ against $\mu$. This is the form one uses in proofs because it works even when $T$ is far from a diffeomorphism (so set-theoretic intuition fails).
* **Probabilistic.** If $X\sim\mu$ is a random variable, then $T(X)\sim T_{\sharp}\mu$. The push-forward is just the **law of $T(X)$**. Asking $T_{\sharp}\mu=\nu$ is asking: "*find a map $T$ that takes a $\mu$-distributed sample to a $\nu$-distributed sample, deterministically*."

In all three readings, mass is conserved exactly: $T_{\sharp}\mu(\mathbb R^d) = \mu(\mathbb R^d)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Monge's problem)</span></p>

Given measures $\mu,\nu$ on $\mathbb R^d$ with the same total mass and a cost function $c:\mathbb R^d\times\mathbb R^d\to[0,+\infty]$, **Monge's problem** is

$$
\min_{T:\mathbb R^d\to\mathbb R^d} \int_{\mathbb R^d} c(x,T(x))\,d\mu(x) \qquad\text{subject to } T_{\sharp}\mu=\nu. \tag{2.1}
$$

The infimum is taken over all (measurable) maps $T$ whose push-forward equals $\nu$.

</div>

This problem is incredibly **hard** because the constraint $T_\sharp\mu=\nu$ is highly **non-linear** in $T$. The next two examples make this concrete.

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/ot_monge_pushforward.png' | relative_url }}" alt="Two-panel figure: left shows source density mu (blue) and target density nu (red, bimodal) with grey arrows from sample points x to T(x) sketching the deterministic transport map; right shows the change-of-variables identity by transporting three coloured mass-blocks from a Gaussian source under a smooth monotone T, with the destination blocks visibly stretched/compressed according to det DT" loading="lazy">
  <figcaption>Monge's problem in pictures. <strong>Left.</strong> A deterministic map $T$ sends each source point $x$ to a single destination $T(x)$, deforming the source density $\mu$ (blue) into the target density $\nu$ (red); the greying arrows show several individual source–target pairs. <strong>Right.</strong> The Jacobian rule $f(x)=g(T(x))\,\det DT(x)$ visualised: three sample mass-blocks of equal $\mu$-mass are transported to destination blocks whose width is dilated or contracted by $1/\det DT$, so densities scale inversely to local volume change. Mass is conserved exactly; only the geometry of "where it lives" is rearranged.</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Absolutely continuous case — the Monge–Ampère equation)</span></p>

If $\mu$ and $\nu$ are both absolutely continuous with respect to Lebesgue measure with densities $f$ and $g$, i.e. $d\mu = f\,dx$ and $d\nu = g\,dy$, and $T$ is a $C^1$-diffeomorphism, then the change-of-variables formula turns $T_\sharp\mu=\nu$ into the **pointwise** identity

$$
f(x) \;=\; g(T(x))\,\det DT(x). \tag{2.2 — Jacobian eq.}
$$

A priori, it is not even clear that *any* map $T$ exists satisfying this constraint. When the cost is quadratic (so that $T=\nabla\varphi$ for a convex $\varphi$, by Brenier's theorem ahead), (2.2) becomes the celebrated **Monge–Ampère equation**

$$
\det D^2\varphi(x) \;=\; \frac{f(x)}{g(\nabla\varphi(x))},
$$

a fully non-linear second-order PDE — one of the reasons optimal transport is so connected to PDE theory.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(No transport map exists in general)</span></p>

Let $d=1$ and consider the discrete measures

$$
\mu = \delta_0, \qquad \nu = \tfrac12\delta_{-1} + \tfrac12\delta_{+1}.
$$

There is **no** map $T:\mathbb R\to\mathbb R$ with $T_\sharp\mu=\nu$.

Indeed, since $\mu$ is concentrated at the single point $0$, the push-forward $T_\sharp\mu$ must be $\delta_{T(0)}$ — a Dirac mass at one point. But $\nu$ puts mass at *two* points, so no single-valued map can match it.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The structural failure mode of Monge — and the cure that follows)</span></p>

Example 2 is not a pathological corner case; it is the *typical* obstruction to Monge's problem and the precise reason Kantorovich's relaxation will be needed in the next section.

* **What goes wrong.** A map $T$ assigns *one* destination to *each* source point $x$. So if $\mu$ has an atom at $x_0$ — concentrating a finite chunk of mass at a single point — that whole chunk is forced to land at the single destination $T(x_0)$. There is no room to **split** mass.
* **What $\nu$ may demand.** A target measure with mass spread across multiple points (as in Example 2) requires the chunk at $x_0$ to be split. Maps simply cannot do this.
* **General principle.** Every time Monge's constraint $T_\sharp\mu=\nu$ is feasible, $\nu$ is in some sense "no more diffuse than $\mu$" — the source must already be at least as spread out as the target. When $\mu$ is more atomic, the problem is empty and the infimum in (2.1) is $+\infty$ by convention.
* **The cure.** Kantorovich's idea (next section) is precisely to *allow splitting*: instead of a deterministic destination $T(x)$ for each source point, one keeps a **joint distribution** $\pi(x,y)$ describing how much mass moves from $x$ to $y$. This restores feasibility (since the product measure $\mu\otimes\nu$ is always feasible), and turns the non-linear problem (2.1) into a linear one.

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/ot_no_map_vs_split.png' | relative_url }}" alt="Two-panel figure contrasting Monge's failure with Kantorovich's success on the same atomic data. Left panel shows mu = delta_0 as a single blue spike at 0 of height 1 and nu = (1/2)(delta_-1 + delta_+1) as two red spikes of height 0.5 at -1 and +1; two grey curved arrows from 0 to -1 and 0 to +1 are crossed out by a red 'no map can split delta_0' callout. Right panel shows the same source and target with two thick green arrows of weight 1/2 each, splitting the Dirac mass at 0 between the two targets, with a green box stating pi = (1/2) delta_(0,-1) + (1/2) delta_(0,+1) is in Pi(mu,nu)" loading="lazy">
  <figcaption>The structural failure of Monge and Kantorovich's cure. <strong>Left.</strong> For $\mu=\delta_0$ and $\nu=\tfrac12\delta_{-1}+\tfrac12\delta_{+1}$, no map $T$ can satisfy $T_{\sharp}\mu=\nu$: the push-forward of an atom is itself an atom, so $T_{\sharp}\delta_0=\delta_{T(0)}$ — a single Dirac, not a sum of two. <strong>Right.</strong> A Kantorovich coupling can split the source atom: the plan $\pi=\tfrac12\delta_{(0,-1)}+\tfrac12\delta_{(0,+1)}$ has the right marginals $(\mu,\nu)$ even though it is not concentrated on the graph of any function. Replacing "graphs of maps" by "couplings on $\mathbb R^d\times\mathbb R^d$" is precisely what restores feasibility.</figcaption>
</figure>

### 2.2 Kantorovich's formulation

In the 1940s, Kantorovich found a **relaxation** of Monge's problem that can be viewed as a **linear program**, with a tractable natural **dual**. Both are far easier to study than Monge's original formulation.

The idea is simple: instead of insisting that all mass at a single point $x$ travel to a single destination, allow it to be **split** across several destinations. The transport behaviour is now recorded by a **plan** $\pi(x,y)$ specifying, for each pair $(x,y)$, how much mass moves from $x$ to $y$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Transference plan / coupling)</span></p>

Given two probability measures $\mu,\nu$ on $\mathbb R^d$, a **transference plan** (or **coupling**) of $\mu$ and $\nu$ is a non-negative Borel measure $\pi$ on the product space $\mathbb R^d\times\mathbb R^d$ whose **marginals** are $\mu$ and $\nu$:

$$
\int_{\mathbb R^d\times\mathbb R^d}\varphi(x)\,d\pi(x,y) = \int_{\mathbb R^d}\varphi(x)\,d\mu(x) \;\;\text{and}\;\; \int_{\mathbb R^d\times\mathbb R^d}\psi(y)\,d\pi(x,y) = \int_{\mathbb R^d}\psi(y)\,d\nu(y)
$$

for all bounded continuous test functions $\varphi,\psi$. We write $\pi\in\Pi(\mu,\nu)$ for the set of all such plans.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Plans generalise maps)</span></p>

Every Monge-style transport map $T$ with $T_\sharp\mu=\nu$ induces a plan

$$
\pi_T := (\mathrm{id},T)_\sharp\mu, \qquad\text{i.e.}\quad \int F(x,y)\,d\pi_T(x,y) = \int F(x,T(x))\,d\mu(x),
$$

which is concentrated on the **graph** $\{(x,T(x))\}\subset\mathbb R^d\times\mathbb R^d$. The marginals of $\pi_T$ are exactly $\mu$ and $\nu$, so $\pi_T\in\Pi(\mu,\nu)$. In this sense the Kantorovich problem **contains** Monge's problem: the feasible set is enlarged from "graphs of maps" to "all couplings".

The enlargement is strict whenever splitting is needed (e.g. $\mu=\delta_0$, $\nu=\tfrac12\delta_{-1}+\tfrac12\delta_{+1}$ from Example 2 above): the plan $\pi=\tfrac12(\delta_0\otimes\delta_{-1}+\delta_0\otimes\delta_{+1})$ is feasible, even though no map is.

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/ot_kantorovich_plan_marginals.png' | relative_url }}" alt="Two-panel joint-space figure with shared marginals on the axes. Left panel shows a Monge plan supported on the graph y = T(x): a thin diagonal-like curve in the (x,y)-plane, coloured by the mu-density along it, with the source marginal mu (blue) on the top strip and the target marginal nu (red) on the right strip. Right panel shows a more diffuse Kantorovich coupling: a 2D pink-purple cloud spread across two diagonal branches, with the same blue mu marginal on top and red nu marginal on the right, illustrating that many distinct couplings share the same prescribed marginals" loading="lazy">
  <figcaption>Couplings live on the product space, with prescribed marginals. <strong>Left.</strong> The Monge plan $\pi=(\mathrm{id},T)_{\sharp}\mu$ is concentrated on the *graph* $\{(x,T(x))\}\subset\mathbb R\times\mathbb R$ (a singular measure on a 1-dimensional curve); the colour intensity along the graph is the source density $\mu$. <strong>Right.</strong> A non-deterministic coupling spreads mass over a 2D region while keeping the same marginal projections $\mu$ (top, blue) and $\nu$ (right, red). Both are valid elements of $\Pi(\mu,\nu)$; the Kantorovich relaxation enlarges the feasible set from "1D curves" to "all positive measures with the right marginals", which is why it is automatically convex and well-posed.</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Kantorovich's primal problem)</span></p>

For $\mu,\nu$ probability measures on $\mathbb R^d$ and $c:\mathbb R^d\times\mathbb R^d\to[0,+\infty]$ a cost function, **Kantorovich's primal problem** is

$$
\min_{\pi\in\Pi(\mu,\nu)} I(\pi), \qquad I(\pi) := \int_{\mathbb R^d\times\mathbb R^d} c(x,y)\,d\pi(x,y). \tag{2.2}
$$

</div>

This is a **linear** problem: both the objective $\pi\mapsto I(\pi)$ and the marginal constraints are linear in $\pi$. Like every linear program, it has a natural **dual**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Kantorovich's dual problem)</span></p>

The **dual problem** of (2.2) is

$$
\sup_{\varphi+\psi\le c}\;J(\varphi,\psi), \qquad J(\varphi,\psi) := \int_{\mathbb R^d}\varphi(x)\,d\mu(x) + \int_{\mathbb R^d}\psi(y)\,d\nu(y), \tag{2.3}
$$

where the supremum is over all pairs of bounded continuous functions $(\varphi,\psi)$ with $\varphi(x)+\psi(y)\le c(x,y)$ for all $x,y\in\mathbb R^d$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10</span><span class="math-callout__name">(Kantorovich duality)</span></p>

Let $\mu,\nu$ be two probability measures on $\mathbb R^d$ and $c:\mathbb R^d\times\mathbb R^d\to[0,+\infty]$ a lower semi-continuous cost function. Then

$$
\min_{\pi\in\Pi(\mu,\nu)} I(\pi) \;=\; \sup_{\varphi+\psi\le c} J(\varphi,\psi),
$$

where the supremum on the right is over all bounded continuous $(\varphi,\psi)$ with $\varphi(x)+\psi(y)\le c(x,y)$ for all $x,y\in\mathbb R^d$. Moreover, the minimum on the left is **attained** by some $\pi_\ast\in\Pi(\mu,\nu)$.

</div>

For now, we only prove the easy direction — the inequality "$\ge$". The other direction follows from the **minimax theorem**, a general result in convex analysis that we postpone.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11</span><span class="math-callout__name">(Kantorovich duality, easy direction)</span></p>

Under the assumptions of Theorem 10,

$$
\min_{\pi\in\Pi(\mu,\nu)} I(\pi) \;\ge\; \sup_{\varphi+\psi\le c} J(\varphi,\psi).
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 11 — turning constraints into a penalty</summary>

The trick is to rewrite the marginal constraint on $\pi$ as a $\sup$ over test pairs $(\varphi,\psi)$, so that swapping that $\sup$ with the $\inf$ over $\pi$ produces the dual. Concretely:

**Step 1: a constraint-as-sup identity.** For any $\pi\ge 0$,

$$
\sup_{\varphi,\psi}\Bigl\{ \int\varphi\,d\mu + \int\psi\,d\nu - \int(\varphi(x)+\psi(y))\,d\pi(x,y)\Bigr\} \;=\; \begin{cases} 0, & \pi\in\Pi(\mu,\nu),\\ +\infty, & \text{otherwise}.\end{cases}
$$

The reason: if $\pi\in\Pi(\mu,\nu)$, every term in braces is $0$, so the supremum is $0$ (attained at $\varphi=\psi=0$). If $\pi\notin\Pi(\mu,\nu)$, the marginals of $\pi$ disagree with $\mu$ or $\nu$ on some test function, and we can scale that test function to make the bracketed quantity arbitrarily large.

**Step 2: lift the constraint into the objective.** Because the bracketed expression is $0$ on the feasible set and $+\infty$ off it,

$$
\inf_{\pi\in\Pi(\mu,\nu)} I(\pi) \;=\; \inf_{\pi\ge 0}\Bigl\{ I(\pi) + \sup_{\varphi,\psi}\Bigl[\int\varphi\,d\mu+\int\psi\,d\nu - \int(\varphi+\psi)\,d\pi\Bigr]\Bigr\}.
$$

**Step 3: swap $\inf$ and $\sup$ (weakly).** In general $\inf\sup\ge\sup\inf$, so

$$
\inf_{\pi\ge 0}\Bigl\{\cdots\Bigr\} \;\ge\; \sup_{\varphi,\psi}\Bigl\{ \int\varphi\,d\mu+\int\psi\,d\nu + \inf_{\pi\ge 0}\int\bigl(c(x,y)-(\varphi(x)+\psi(y))\bigr)\,d\pi(x,y)\Bigr\}.
$$

**Step 4: evaluate the inner $\inf$ over $\pi\ge 0$.** Because $\pi$ is a free non-negative measure,

$$
\inf_{\pi\ge 0}\int\bigl(c(x,y)-(\varphi(x)+\psi(y))\bigr)\,d\pi(x,y) \;=\; \begin{cases} 0, & \varphi(x)+\psi(y)\le c(x,y) \text{ for all }x,y,\\ -\infty, & \text{otherwise}.\end{cases}
$$

(If the integrand is non-negative everywhere, the optimal $\pi$ is the zero measure. If it is negative somewhere, putting more and more mass at that point drives the integral to $-\infty$.)

**Step 5: combine.** Plugging Step 4 into Step 3, only $(\varphi,\psi)$ with $\varphi+\psi\le c$ contribute (the others give $-\infty$, which the sup discards), and on that set the inner $\inf$ is $0$. Therefore

$$
\inf_{\pi\in\Pi(\mu,\nu)} I(\pi) \;\ge\; \sup_{\varphi+\psi\le c}\Bigl\{ \int\varphi\,d\mu+\int\psi\,d\nu\Bigr\} \;=\; \sup_{\varphi+\psi\le c} J(\varphi,\psi). \;\;\square
$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The pricing interpretation of duality)</span></p>

Kantorovich's dual admits a beautiful **economic** reading that makes weak duality intuitive.

* **The primal — you do it yourself.** You own a company that needs to transport goods from supplier locations distributed as $\mu$ to customer locations distributed as $\nu$, and you pay $c(x,y)$ per unit of mass to move it from $x$ to $y$. Your minimal cost is exactly the Kantorovich primal $\min_\pi I(\pi)$.
* **The dual — outsource the job.** A logistics contractor offers to do the transport on your behalf. They quote you two prices: $\varphi(x)$ to **load** a unit of mass at location $x$, and $\psi(y)$ to **unload** a unit at location $y$. Your total bill is $\int\varphi\,d\mu+\int\psi\,d\nu = J(\varphi,\psi)$.
* **The arbitrage constraint.** You will only accept the contract if $\varphi(x)+\psi(y)\le c(x,y)$ for every pair $(x,y)$ — otherwise you could "go around" the contractor by transporting that particular unit yourself for less. This is exactly the dual feasibility constraint.
* **The contractor's optimization.** The contractor wants to charge as much as possible, so they maximise $J(\varphi,\psi)$ subject to the arbitrage constraint. Their best price is the dual sup.

Weak duality $\min_\pi I(\pi)\ge\sup_{\varphi+\psi\le c}J(\varphi,\psi)$ is then the trivial economic fact that **outsourcing under no-arbitrage cannot be cheaper than doing the job yourself**. Strong duality (Theorem 10) is the deeper statement that *the contractor can drive their price right up to the minimum self-cost*: at the optimum, you are indifferent between doing the job and contracting it out.

(In real life, of course, the contractor charges a small surcharge, which you might still pay for the convenience.)

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/ot_duality_pricing.png' | relative_url }}" alt="Two-panel figure illustrating Kantorovich duality for the quadratic cost. Left panel: 1D slice along s = x - y, showing the cost c(s) = s^2/2 as a blue parabola and the additive envelope phi(x)+psi(y) = s - 1/2 as a green line; the green line lies entirely below the blue parabola, with a single tangency at s = 1 marked by a red dot, and the grey region between them is labelled as the duality gap (s-1)^2/2. Right panel: 2D heatmap of the gap c(x,y) - phi(x) - psi(y) on the (x,y)-plane, coloured grey-to-black (zero on a thin red line y = x - 1, growing outward); thin blue diagonal level curves of the cost are overlaid for reference" loading="lazy">
  <figcaption>The geometry of duality for $c(x,y)=\tfrac12(x-y)^2$. <strong>Left.</strong> Take the genuine feasible pair $\varphi(x)=x-1$, $\psi(y)=-y+\tfrac12$ (a Fenchel–Young dual to $u\mapsto\tfrac12(u-1)^2+\tfrac12$). Along the slice $s:=x-y$, the cost $c=s^2/2$ (blue) and the envelope $\varphi+\psi=s-\tfrac12$ (green) satisfy $\varphi+\psi\le c$ everywhere, with the duality gap $\tfrac12(s-1)^2$ shaded grey. <strong>Right.</strong> In the full $(x,y)$-plane, the gap vanishes precisely on the line $y=x-1$ (red) — the single 1-dimensional curve along which $\varphi+\psi=c$. By complementary slackness, *the optimal coupling for any compatible $\mu,\nu$ must be supported on this very curve*: the contact set of a feasible $(\varphi,\psi)$ is the geometric locus of optimal transport pairs.</figcaption>
</figure>

We saw in Example 2 that Monge's problem can be **infeasible**. Kantorovich's problem, by contrast, is **always solvable** — this is part of Theorem 10, but worth restating with a self-contained proof.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 12</span><span class="math-callout__name">(Existence of optimal transference plans)</span></p>

Let $\mu,\nu$ be probability measures on $\mathbb R^d$ and $c:\mathbb R^d\times\mathbb R^d\to[0,+\infty]$ a lower semi-continuous cost function. Then there exists an optimal transport plan $\pi_\ast\in\Pi(\mu,\nu)$ with

$$
I(\pi_\ast) \;=\; \min_{\pi\in\Pi(\mu,\nu)} I(\pi).
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Existence proof, three-step structure)</span></p>

The proof is the **direct method of the calculus of variations** — the same template that drove the existence proof for minimizing movements (§§1.4, 1.8). It always boils down to three ingredients:

1. **Non-emptiness.** The feasible set $\Pi(\mu,\nu)$ is non-empty, because the **product measure** $\mu\otimes\nu$ is always a coupling (if $X\sim\mu$ and $Y\sim\nu$ are *independent*, then $(X,Y)\sim\mu\otimes\nu\in\Pi(\mu,\nu)$).
2. **Compactness.** $\Pi(\mu,\nu)$ is **weakly compact** — a sequence of plans, all having the same prescribed marginals, is automatically *tight*, and Prokhorov's theorem extracts a weak limit. This is the "calculus of variations" half of the argument.
3. **Lower semi-continuity.** The cost functional $I(\pi)=\int c\,d\pi$ is lower semi-continuous along weak limits — this is where the lower semi-continuity hypothesis on $c$ enters, via approximation from below by bounded continuous functions and monotone convergence.

These three ingredients combine in the standard way: take any minimizing sequence; extract a weakly convergent subsequence by (2); use (3) to pass the cost $I$ through the limit; the limit is the minimizer.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 12 — direct method</summary>

**Step 1: $\Pi(\mu,\nu)$ is non-empty.** Take $\pi:=\mu\otimes\nu$, the product measure. Its marginals are obviously $\mu$ and $\nu$, so $\pi\in\Pi(\mu,\nu)$.

**Step 2: $\Pi(\mu,\nu)$ is weakly compact.** Let $(\pi_n)\subset\Pi(\mu,\nu)$. We show *tightness*: for every $\varepsilon>0$, there is a compact set $K\subset\mathbb R^d\times\mathbb R^d$ with $\pi_n(K)\ge 1-\varepsilon$ uniformly in $n$.

Since $\mu,\nu$ are individually tight (any single Borel probability measure on $\mathbb R^d$ is), pick compacts $K_1,K_2\subset\mathbb R^d$ with $\mu(K_1)\ge 1-\tfrac\varepsilon2$ and $\nu(K_2)\ge 1-\tfrac\varepsilon2$. Then $K:=K_1\times K_2$ is compact and

$$
\pi_n(K_1\times K_2) \;\ge\; 1 - \pi_n((\mathbb R^d\setminus K_1)\times\mathbb R^d) - \pi_n(\mathbb R^d\times(\mathbb R^d\setminus K_2)) \;=\; 1 - \mu(K_1^c) - \nu(K_2^c) \;\ge\; 1-\varepsilon,
$$

using the marginal identity $\pi_n(A\times\mathbb R^d)=\mu(A)$ and analogously for the second factor. So $(\pi_n)$ is tight.

By **Prokhorov's theorem**, there is a subsequence (not relabelled) and a probability measure $\pi$ on $\mathbb R^d\times\mathbb R^d$ with $\pi_n\rightharpoonup\pi$ weakly — i.e. $\int\zeta\,d\pi_n\to\int\zeta\,d\pi$ for every $\zeta\in C_b(\mathbb R^d\times\mathbb R^d)$.

It remains to check that $\pi$ has the right marginals. For any $\varphi\in C_b(\mathbb R^d)$, the function $(x,y)\mapsto\varphi(x)$ is in $C_b(\mathbb R^d\times\mathbb R^d)$, so

$$
\int\varphi(x)\,d\pi(x,y) \;=\; \lim_{n\to\infty}\int\varphi(x)\,d\pi_n(x,y) \;=\; \lim_{n\to\infty}\int\varphi\,d\mu \;=\; \int\varphi\,d\mu,
$$

and analogously for $\psi(y)$ and $\nu$. Hence $\pi\in\Pi(\mu,\nu)$, proving weak compactness.

**Step 3: $I$ is lower semi-continuous.** Let $\pi_n\rightharpoonup\pi$ in $\Pi(\mu,\nu)$. Since $c\ge 0$ is l.s.c. on $\mathbb R^d\times\mathbb R^d$, write $c$ as the pointwise increasing limit of bounded continuous $c_k:\mathbb R^d\times\mathbb R^d\to[0,\infty)$ with $c_k\nearrow c$. (Such an approximation exists by the standard "$c_k(z)=\inf_{w}\{c(w)+k\|z-w\|\}\wedge k$" construction.) Then

$$
\liminf_{n\to\infty}I(\pi_n) \;=\; \liminf_{n\to\infty}\int\Bigl(\lim_{k\to\infty}c_k\Bigr)\,d\pi_n \;\ge\; \limsup_{k\to\infty}\liminf_{n\to\infty}\int c_k\,d\pi_n \;=\; \lim_{k\to\infty}\int c_k\,d\pi \;=\; \int c\,d\pi,
$$

where the second step uses $c\ge c_k$ and the limsup-liminf inequality, the third step uses weak convergence of $\pi_n$ together with $c_k\in C_b$, and the last uses **monotone convergence** (since $c_k\nearrow c$ pointwise and $c\ge 0$). So $I(\pi)\le\liminf_n I(\pi_n)$.

**Step 4: combine.** Take a minimizing sequence $(\pi_n)\subset\Pi(\mu,\nu)$ with $I(\pi_n)\to\min_{\pi\in\Pi(\mu,\nu)}I(\pi)$ — well-defined because $\Pi(\mu,\nu)\neq\emptyset$. By Step 2, extract a weak limit $\pi_\ast\in\Pi(\mu,\nu)$. By Step 3,

$$
I(\pi_\ast) \;\le\; \liminf_{n\to\infty}I(\pi_n) \;=\; \min_{\pi\in\Pi(\mu,\nu)}I(\pi),
$$

so the inequality is in fact an equality and $\pi_\ast$ is optimal. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why tightness is "free" for couplings)</span></p>

The key compactness argument in Step 2 is striking precisely because it is **so easy**: tightness of *individual* probability measures on $\mathbb R^d$ — a non-trivial property in general — is *automatic* for couplings, because the marginal constraints lock in the support of $\pi_n$ from both sides. No quantitative bound on the cost is needed; the bound on the *marginals* alone is enough to keep mass from escaping to infinity.

This is the structural reason Kantorovich's relaxation is so much more tractable than Monge's: the feasible set $\Pi(\mu,\nu)$ is **automatically** a weakly compact convex set, so the existence theory is essentially trivial. All the analytic difficulty is shifted into *characterising* the optimum (which map, when does the plan come from a map, etc.) — that is the content of Brenier's theorem ahead.

</div>

### 2.3 Brenier's theorem

We have seen that Monge's problem is very hard — it can even be infeasible — while Kantorovich's relaxation is much more tractable, with a clean dual and an automatic existence theorem. **Brenier's theorem** is the bridge back: in the very relevant case of the **quadratic cost**

$$
c(x,y) \;=\; \tfrac12\|x-y\|^2 \qquad\text{(the prefactor }\tfrac12\text{ is purely cosmetic)},
$$

the optimal Kantorovich plan is in fact concentrated on the graph of a **transport map** — and that map is the **gradient of a convex function**.

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/ot_brenier_preview.png' | relative_url }}" alt="Three-panel preview of Brenier's theorem in 1D. Panel (a) shows a smooth convex potential phi(x) (purple) with a dashed grey tangent line at x=-0.8 illustrating the supporting hyperplane characterisation of convexity. Panel (b) shows its derivative T(x) = phi'(x) (teal), a non-decreasing curve that is steep where phi is curved and flat where phi is nearly linear; orange step-arrows on two pairs (x_1 < x_2) demonstrate that T(x_1) <= T(x_2). Panel (c) shows source density mu (blue, single Gaussian centred at 0) and target density nu (red, bimodal mixture); curved grey arrows from sample x to T(x) show how the unimodal source is rearranged into a bimodal target by following the monotone gradient" loading="lazy">
  <figcaption>Brenier's theorem in pictures (1D preview). <strong>(a)</strong> A convex potential $\varphi$. Convexity is exactly the existence at every $x$ of a supporting tangent line lying entirely below the graph (dashed grey). <strong>(b)</strong> Its derivative $T=\varphi'$ is automatically *monotone non-decreasing*: $x_1\le x_2$ implies $T(x_1)\le T(x_2)$ (orange step-arrows). <strong>(c)</strong> The map $T=\varphi'$ pushes the source density $f$ (blue) onto the target density $g$ (red): each source point $x$ is sent to the unique $T(x)$ such that the cumulative source mass to the left of $x$ equals the cumulative target mass to the left of $T(x)$ (the **monotone rearrangement** / quantile transform). Brenier's theorem says this is the unique optimal map for the quadratic cost $c(x,y)=\tfrac12\|x-y\|^2$ — and the analogous statement holds in any dimension, with $T=\nabla\varphi$ for a convex $\varphi:\mathbb R^d\to\mathbb R$.</figcaption>
</figure>

#### Preparation: exploiting the quadratic structure

To prepare the ground for Brenier's theorem, we first rewrite the dual admissibility condition $\varphi+\psi\le c$ in a way that makes the role of convex duality manifest. Expanding $c(x,y)=\tfrac12\\|x-y\\|^2=\tfrac12\\|x\\|^2-x\cdot y+\tfrac12\\|y\\|^2$, the condition

$$
\varphi(x)+\psi(y) \;\le\; \tfrac12\|x-y\|^2 \quad\text{for $\mu$-a.e. $x$ and $\nu$-a.e. $y$}
$$

is equivalent — after rearrangement — to

$$
x\cdot y \;\le\; \bigl(\tfrac12\|x\|^2-\varphi(x)\bigr) + \bigl(\tfrac12\|y\|^2-\psi(y)\bigr). \tag{2.4}
$$

The right-hand side begs to be read as the Fenchel inequality for the conjugate of *something*. So define

$$
\tilde\varphi(x) \;:=\; \tfrac12\|x\|^2-\varphi(x), \qquad \tilde\psi(y) \;:=\; \tfrac12\|y\|^2-\psi(y),
$$

and, abusing notation, *drop the tildes* — from here on $\varphi,\psi$ refer to these shifted potentials, and (2.4) reads simply $x\cdot y\le\varphi(x)+\psi(y)$.

Recall the **convex conjugate** (or Legendre–Fenchel transform) of a function $\varphi:\mathbb R^d\to\mathbb R\cup\\{+\infty\\}$:

$$
\varphi^\ast(y) \;:=\; \sup_{x\in\mathbb R^d}\bigl\{\,x\cdot y-\varphi(x)\,\bigr\}.
$$

Assume now that $\mu,\nu$ have **finite second moments**, i.e.

$$
M_2 \;:=\; \int_{\mathbb R^d}\|x\|^2\,d\mu(x) + \int_{\mathbb R^d}\|y\|^2\,d\nu(y) \;<\; +\infty.
$$

Then both Kantorovich's primal and dual problems become equivalent to **inner-product problems**:

$$
\inf_{\pi\in\Pi(\mu,\nu)} I(\pi) \;=\; \tfrac12 M_2 \;-\; \sup_{\pi\in\Pi(\mu,\nu)}\int_{\mathbb R^d\times\mathbb R^d} x\cdot y\,d\pi(x,y),
$$

$$
\sup_{\varphi+\psi\le c} J(\varphi,\psi) \;=\; \tfrac12 M_2 \;-\; \inf_{x\cdot y\le\varphi+\psi} J(\varphi,\psi),
$$

so Kantorovich duality reads, in this rewritten form,

$$
\boxed{\;\sup_{\pi\in\Pi(\mu,\nu)}\int_{\mathbb R^d\times\mathbb R^d}x\cdot y\,d\pi(x,y) \;=\; \inf_{x\cdot y\le\varphi+\psi} J(\varphi,\psi).\;}
$$

The whole problem is now phrased in terms of the **bilinear pairing** $x\cdot y$ and the **Fenchel-type constraint** $x\cdot y\le\varphi(x)+\psi(y)$. This is the natural arena for convex duality.

#### The double convexification trick

In the rewritten dual, any admissible pair $(\varphi,\psi)$ can be **systematically improved**, in two steps, until it consists of a *convex function and its conjugate*. The trick is built directly into the constraint $x\cdot y\le\varphi(x)+\psi(y)$.

*Step 1.* Fixing $\varphi$, the constraint forces $\psi(y)\ge x\cdot y-\varphi(x)$ for $\mu$-a.e. $x$. Taking the supremum over $x$,

$$
\psi(y) \;\ge\; \sup_{x\in\mathbb R^d}\bigl\{\,x\cdot y-\varphi(x)\,\bigr\} \;=\; \varphi^\ast(y) \qquad\text{for }\nu\text{-a.e. }y.
$$

So replacing $\psi$ by the *smaller* function $\varphi^\ast$ preserves admissibility and **lowers** $J$: $J(\varphi,\varphi^\ast)\le J(\varphi,\psi)$.

*Step 2.* Now run the same trick on $\varphi$: by definition of $\varphi^\ast$, we always have $\varphi(x)\ge\sup_y\\{x\cdot y-\varphi^\ast(y)\\}=\varphi^{\ast\ast}(x)$, so $(\varphi^{\ast\ast},\varphi^\ast)$ is admissible and

$$
J(\varphi^{\ast\ast},\varphi^\ast) \;\le\; J(\varphi,\varphi^\ast) \;\le\; J(\varphi,\psi).
$$

The function $\varphi^{\ast\ast}$ is automatically a **lower semi-continuous proper convex** function (the biconjugate is always the largest l.s.c. convex minorant). The upshot:

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Reduction</span><span class="math-callout__name">(It suffices to optimise over conjugate pairs)</span></p>

In the rewritten dual, one may restrict attention to admissible pairs of the form $(\varphi,\varphi^\ast)$ with $\varphi$ a lower semi-continuous proper convex function. Every other admissible pair can be improved to one of this form without raising $J$.

</div>

#### A reminder from convex analysis

For the proofs ahead, we collect — without proofs — the facts about convex functions that we will need.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definitions</span><span class="math-callout__name">(Proper / convex / subdifferential / conjugate)</span></p>

Let $\varphi:\mathbb R^d\to\mathbb R\cup\\{+\infty\\}$.

1. $\varphi$ is **proper** if it is not identically $+\infty$.
2. $\varphi$ is **convex** if $\varphi(tx+(1-t)y)\le t\varphi(x)+(1-t)\varphi(y)$ for all $x,y\in\mathbb R^d$ and all $t\in[0,1]$.
3. The **subdifferential** of a convex $\varphi$ at $x$ is

   $$\partial\varphi(x) \;=\; \bigl\{\,p\in\mathbb R^d \,:\, \varphi(y)\ge\varphi(x)+p\cdot(y-x)\;\text{for all }y\in\mathbb R^d\,\bigr\}$$

   It is the set of *supporting affine functions* to $\varphi$ at $x$. If $\varphi$ is differentiable at $x$, then $\partial\varphi(x)=\\{\nabla\varphi(x)\\}$ (cf. Chapter 1).
4. The **convex conjugate** $\varphi^\ast(y)=\sup_x\\{x\cdot y-\varphi(x)\\}$ is automatically a proper l.s.c. convex function. **Biconjugation theorem:** $\varphi^{\ast\ast}=\varphi$ iff $\varphi$ is a proper l.s.c. convex function. The **Fenchel inequality** $x\cdot y\le\varphi(x)+\varphi^\ast(y)$ holds for all $x,y\in\mathbb R^d$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Regularity of convex functions — Rademacher)</span></p>

A convex function $\varphi:\mathbb R^d\to\mathbb R\cup\\{+\infty\\}$ is **locally Lipschitz** on the interior of its domain $\\{\varphi<+\infty\\}$. By **Rademacher's theorem**, every locally Lipschitz function is differentiable almost everywhere (with respect to Lebesgue measure). Hence on the interior of its domain, $\varphi$ is differentiable a.e., and the *exceptional* set on which it is not differentiable is small — it has Hausdorff dimension at most $d-1$.

This is exactly the property that will let us upgrade "the optimal plan lives in $\partial\varphi$" to "the optimal plan lives on the graph of $\nabla\varphi$", under a mild assumption on $\mu$.

</div>

The single most useful fact about conjugates — and the engine behind Brenier — is that the Fenchel inequality has an **equality case** characterising the subdifferential.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Characterisation of the subdifferential)</span></p>

Let $\varphi:\mathbb R^d\to\mathbb R\cup\\{+\infty\\}$ be a proper lower semi-continuous convex function. Then for any $x,y\in\mathbb R^d$,

$$
x\cdot y \;=\; \varphi(x)+\varphi^\ast(y) \;\;\Longleftrightarrow\;\; y\in\partial\varphi(x) \;\;\Longleftrightarrow\;\; x\in\partial\varphi^\ast(y). \tag{2.5}
$$

<details class="proof" markdown="1">
<summary>Proof of (2.5) — Fenchel equality unpacked</summary>

Fix $x,y\in\mathbb R^d$. By the Fenchel inequality the "$\le$" direction is automatic, so equality in $x\cdot y\le\varphi(x)+\varphi^\ast(y)$ is equivalent to "$\ge$":

$$
x\cdot y \;\ge\; \varphi(x)+\varphi^\ast(y).
$$

Now unfold $\varphi^\ast(y)=\sup_z\\{y\cdot z-\varphi(z)\\}$: the inequality above holds iff for **every** $z\in\mathbb R^d$,

$$
x\cdot y \;\ge\; \varphi(x)+y\cdot z-\varphi(z),
$$

which rearranges to

$$
\varphi(z) \;\ge\; \varphi(x)+y\cdot(z-x) \qquad\text{for all }z\in\mathbb R^d.
$$

This is precisely the definition of $y\in\partial\varphi(x)$. The second equivalence $y\in\partial\varphi(x)\Leftrightarrow x\in\partial\varphi^\ast(y)$ follows by symmetry: since $\varphi^{\ast\ast}=\varphi$ in our setting, swapping $\varphi$ and $\varphi^\ast$ runs the same argument. $\square$

</details>

</div>

This *bookkeeping identity* (2.5) is what converts the dual constraint $x\cdot y\le\varphi+\varphi^\ast$, holding with equality on the support of the optimal plan, into the geometric statement "the optimal plan is supported on the graph of $\partial\varphi$".

#### Existence of dual minimisers

Before stating Brenier's theorem we record that the rewritten dual — now restricted to conjugate pairs — actually attains its minimum.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 13</span><span class="math-callout__name">(Existence of dual minimisers among convex conjugate pairs)</span></p>

Let $\mu,\nu$ be probability measures on $\mathbb R^d$ with finite second moments and set

$$
\Phi(\mu,\nu) \;:=\; \bigl\{\,(\varphi,\psi)\in L^1(\mu)\times L^1(\nu) \,:\, x\cdot y\le\varphi(x)+\psi(y)\;\text{for }\mu\text{-a.e. }x\text{ and }\nu\text{-a.e. }y\,\bigr\}.
$$

Then there exists a minimiser of $J$ over $\Phi(\mu,\nu)$, and one may choose it to be a pair of **convex conjugates**: more precisely, there is a pair $(\varphi,\varphi^\ast)\in\Phi(\mu,\nu)$ of lower semi-continuous proper convex functions such that

$$
J(\varphi,\varphi^\ast) \;=\; \inf_{\Phi(\mu,\nu)} J.
$$

</div>

We omit the proof — it combines the double-convexification reduction with a standard direct-method/lower-semicontinuity argument à la Proposition 12.

#### The main theorem

Brenier's theorem really comes in two layers: a *general optimality criterion* (due to **Knott–Smith**, valid for arbitrary $\mu$ with finite second moment), and a *refined structure theorem* (due to **Brenier**) under the additional assumption that $\mu$ does not give mass to small sets. Both layers, plus the symmetry, are usually packaged into a single theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13</span><span class="math-callout__name">(Brenier / Knott–Smith)</span></p>

Let $\mu,\nu$ be probability measures on $\mathbb R^d$ with finite second moments, and let $c(x,y)=\tfrac12\\|x-y\\|^2$.

**(i) Knott–Smith optimality criterion.** A transference plan $\pi\in\Pi(\mu,\nu)$ is optimal **if and only if** there exists a lower semi-continuous convex function $\varphi$ such that

$$
\operatorname{supp}\pi \;\subset\; \operatorname{graph}\partial\varphi, \tag{2.6}
$$

i.e.

$$
y\in\partial\varphi(x) \qquad\text{for }\pi\text{-a.e. }(x,y)\in\mathbb R^d\times\mathbb R^d. \tag{2.7}
$$

In that case, the pair $(\varphi,\varphi^\ast)$ is a minimiser of the rewritten dual problem $\inf\\{J(\tilde\varphi,\tilde\psi):x\cdot y\le\tilde\varphi+\tilde\psi\\}$, and the pair $\bigl(\tfrac12\\|x\\|^2-\varphi(x),\,\tfrac12\\|y\\|^2-\varphi^\ast(y)\bigr)$ solves the original dual Kantorovich problem $\inf_{\tilde\varphi+\tilde\psi\le c}J(\tilde\varphi,\tilde\psi)$.

**(ii) Brenier's theorem (uniqueness and gradient structure).** If $\mu$ **does not give mass to small sets** (cf. Remark 14), then the optimal transference plan $\pi$ is **unique** and is of the form

$$
d\pi(x,y) \;=\; d\mu(x)\otimes\delta_{\nabla\varphi(x)}, \qquad\text{or equivalently}\qquad \pi \;=\; (\mathrm{id},\nabla\varphi)_\sharp\mu, \tag{2.8}
$$

where $\nabla\varphi$ is the **unique** (up to $\mu$-null sets) gradient of a convex function such that $(\nabla\varphi)_\\#\mu=\nu$.

**(iii) Solution to Monge's problem.** Under the assumption of (ii), $\nabla\varphi$ is the unique solution to Monge's problem for the quadratic cost:

$$
\int_{\mathbb R^d}\|x-\nabla\varphi(x)\|^2\,d\mu(x) \;=\; \inf_{T_\sharp\mu=\nu}\int_{\mathbb R^d}\|x-T(x)\|^2\,d\mu(x).
$$

**(iv) Symmetry.** If in addition $\nu$ does not give mass to small sets, then $\nabla\varphi^\ast$ is the optimal map from $\nu$ to $\mu$, and the two are *inverses of each other* in the a.e. sense:

$$
\nabla\varphi^\ast\circ\nabla\varphi(x) \;=\; x \;\;\text{for }\mu\text{-a.e. }x, \qquad \nabla\varphi\circ\nabla\varphi^\ast(y) \;=\; y \;\;\text{for }\nu\text{-a.e. }y.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 14</span><span class="math-callout__name">("Does not give mass to small sets")</span></p>

A measure $\mu$ on $\mathbb R^d$ **does not give mass to small sets** if $\mu(N)=0$ for every Borel set $N\subset\mathbb R^d$ of Hausdorff dimension at most $d-1$. In particular, any measure that is *absolutely continuous with respect to Lebesgue* satisfies this — Lebesgue itself gives zero mass to lower-dimensional sets, and absolute continuity inherits the property. The condition is *exactly* what we need to pair with Rademacher's theorem: it forces the set where $\varphi$ fails to be differentiable to be $\mu$-null, so that $\partial\varphi(x)=\\{\nabla\varphi(x)\\}$ is a single vector for $\mu$-a.e. $x$.

</div>

#### Geometric interpretation

Let us pause on what $T=\nabla\varphi$ *means*, since the formulation is geometrically curious. The statement is: every point $x\in\mathbb R^d$ is transported to $T(x)=\nabla\varphi(x)\in\mathbb R^d$. To the geometrically minded this is odd — $\nabla\varphi(x)$ is by nature a *tangent vector at $x$*, not a point of the manifold. In flat $\mathbb R^d$ we identify tangent vectors and points without comment, but on a Riemannian manifold $M$ one would have to write

$$
x \;\longmapsto\; T(x) \;=\; \exp_x\!\Bigl(\nabla\bigl(\varphi-\tfrac12\|\cdot\|^2\bigr)\Big|_x\Bigr),
$$

where $\exp_x$ is the exponential map at $x$, which converts a tangent vector at $x$ into a point of $M$. Then $T:M\to M$ really is a map between points.

Two geometric features of $T=\nabla\varphi$ are worth highlighting:

* **Curl-free transport.** A gradient field has zero curl: the transport plan moves mass *radially* with respect to the potential $\varphi$, never in circular swirls. Circular transport is geometrically wasteful — you have moved mass without making net progress — and Brenier says the optimal plan rules it out.
* **Monotone transport.** The convexity of $\varphi$ translates into **monotonicity** of its gradient: $\langle\nabla\varphi(x_1)-\nabla\varphi(x_2),x_1-x_2\rangle\ge 0$ for all $x_1,x_2$. Mass closer in source space stays closer in target space, in the inner-product sense. In 1D this is just the statement that $\varphi'$ is non-decreasing — the *quantile rearrangement* of Figure (c) above.

These are the geometric and PDE-side reasons Brenier's theorem is the entry point to Wasserstein geometry: the optimal map is *the canonical irrotational, monotone rearrangement* between two measures, and combining $T=\nabla\varphi$ with the Jacobian/pushforward equation (2.2) yields the **Monge–Ampère equation** $\det D^2\varphi=f/(g\circ\nabla\varphi)$ — the analytic foundation for regularity theory and Wasserstein gradient flows.

#### Proof of Theorem 13

<details class="proof" markdown="1">
<summary>Proof of Theorem 13 — three steps</summary>

**Step 1: Argument for (i) (Knott–Smith).** By Proposition 12 there exists an optimal transference plan $\pi\in\Pi(\mu,\nu)$, and by Proposition 13 there exists a pair of l.s.c. proper convex functions $(\varphi,\varphi^\ast)\in\Phi(\mu,\nu)$ minimising $J$. By the (rewritten) duality identity at optimum,

$$
\int_{\mathbb R^d\times\mathbb R^d} x\cdot y\,d\pi(x,y) \;=\; \int_{\mathbb R^d}\varphi(x)\,d\mu(x)+\int_{\mathbb R^d}\varphi^\ast(y)\,d\nu(y) \;=\; \int_{\mathbb R^d\times\mathbb R^d}\bigl(\varphi(x)+\varphi^\ast(y)\bigr)d\pi(x,y),
$$

where the last step uses that $\pi$ has marginals $\mu,\nu$. Rearranging,

$$
\int_{\mathbb R^d\times\mathbb R^d}\underbrace{\bigl(\varphi(x)+\varphi^\ast(y)-x\cdot y\bigr)}_{\ge\,0\text{ by Fenchel}}d\pi(x,y) \;=\; 0.
$$

A non-negative integrand whose integral vanishes is zero $\pi$-a.e. So $\varphi(x)+\varphi^\ast(y)=x\cdot y$ for $\pi$-a.e. $(x,y)$, and by (2.5) this is exactly $y\in\partial\varphi(x)$ for $\pi$-a.e. $(x,y)$. That is the forward direction in (i).

Conversely, suppose $\pi\in\Pi(\mu,\nu)$ satisfies (2.7) for some l.s.c. convex $\varphi$. Then (2.5) gives $\varphi(x)+\varphi^\ast(y)=x\cdot y$ on $\operatorname{supp}\pi$, and integrating against $\pi$,

$$
\int_{\mathbb R^d\times\mathbb R^d} x\cdot y\,d\pi(x,y) \;=\; \int_{\mathbb R^d}\varphi\,d\mu + \int_{\mathbb R^d}\varphi^\ast\,d\nu. \tag{2.9}
$$

Weak duality bounds the left side from below by the dual minimum and the right side from above by it — so both inequalities are equalities, and both $\pi$ and $(\varphi,\varphi^\ast)$ are optimal.

**Step 2: Argument for (2.8) in (ii) (the map structure).** Assume now that $\mu$ does not give mass to small sets, and take $\varphi$ from Step 1. Since $\varphi\in L^1(\mu)$, it is finite $\mu$-a.e., so $\mu(\\{\varphi=+\infty\\})=0$. The boundary $\partial\\{\varphi<+\infty\\}$ of a convex set has Hausdorff dimension $\le d-1$, so by hypothesis $\mu(\partial\\{\varphi<+\infty\\})=0$; combined,

$$
\mu\bigl(\operatorname{Int}\{\varphi<+\infty\}\bigr) \;=\; 1.
$$

On the interior of its domain, the convex function $\varphi$ is locally Lipschitz, hence by Rademacher differentiable a.e. with respect to Lebesgue — and the non-differentiability set has Hausdorff dimension $\le d-1$, hence (again by hypothesis) is $\mu$-null. So $\partial\varphi(x)=\lbrace\nabla\varphi(x)\rbrace$ for $\mu$-a.e. $x$, and the Knott–Smith condition (2.7) collapses to $y=\nabla\varphi(x)$ for $\pi$-a.e. $(x,y)$. This is exactly $\pi=(\mathrm{id},\nabla\varphi)\_\\#\mu$, proving (2.8). The pushforward identity $(\nabla\varphi)\_\\#\mu=\nu$ follows because the second marginal of $\pi$ is $\nu$.

**Step 3: Uniqueness of the gradient field.** Suppose $\tilde\varphi$ is another l.s.c. proper convex function with $(\nabla\tilde\varphi)\_\\#\mu=\nu$. By (i) applied to the plan $(\mathrm{id},\nabla\tilde\varphi)\_\\#\mu$ (which is admissible and supported on $\operatorname{graph}\partial\tilde\varphi$), this plan is also optimal, so $(\tilde\varphi,\tilde\varphi^\ast)$ minimises the dual. In particular,

$$
J(\tilde\varphi,\tilde\varphi^\ast) \;=\; J(\varphi,\varphi^\ast) \;=\; \inf_{\Phi(\mu,\nu)}J \;=\; \sup_{\Pi(\mu,\nu)}\int x\cdot y\,d\pi \;=\; \int x\cdot y\,d\pi.
$$

Spelling out the leftmost and rightmost expressions in terms of *the* optimal $\pi=(\mathrm{id},\nabla\varphi)_\\#\mu$ from Step 2,

$$
\int_{\mathbb R^d\times\mathbb R^d}\bigl(\tilde\varphi(x)+\tilde\varphi^\ast(y)\bigr)d\pi(x,y) \;=\; \int_{\mathbb R^d\times\mathbb R^d} x\cdot y\,d\pi(x,y),
$$

and using the explicit form of $\pi$ to push everything onto $\mu$,

$$
\int_{\mathbb R^d}\underbrace{\bigl(\tilde\varphi(x)+\tilde\varphi^\ast(\nabla\varphi(x))-x\cdot\nabla\varphi(x)\bigr)}_{\ge\,0\text{ by Fenchel}}d\mu(x) \;=\; 0.
$$

Once more, a non-negative integrand integrating to zero vanishes — this time $\mu$-a.e. By (2.5),

$$
\nabla\varphi(x) \;\in\; \partial\tilde\varphi(x) \qquad\text{for }\mu\text{-a.e. }x.
$$

But $\tilde\varphi$ is also differentiable $\mu$-a.e. (same Rademacher + small-sets argument), so $\partial\tilde\varphi(x)=\lbrace\nabla\tilde\varphi(x)\rbrace$ at $\mu$-a.e. $x$, and we conclude $\nabla\varphi(x)=\nabla\tilde\varphi(x)$ for $\mu$-a.e. $x$. This proves uniqueness in (ii).

The remaining items (iii) and (iv) follow easily: (iii) because any $T$ with $T_\\#\mu=\nu$ yields an admissible plan $(\mathrm{id},T)_\\#\mu\in\Pi(\mu,\nu)$ with cost $\int\\|x-T(x)\\|^2d\mu$, and the optimal one is $T=\nabla\varphi$ by (ii); (iv) by applying (ii) symmetrically to the inverse problem from $\nu$ to $\mu$, and using $\varphi^{\ast\ast}=\varphi$. $\square$

</details>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 15</span><span class="math-callout__name">(Step 3 proves more than uniqueness of the plan)</span></p>

The uniqueness argument in Step 3 of the proof yields a strictly stronger statement than "the optimal plan is unique": it proves that **any gradient field $\nabla\tilde\varphi$ (of an l.s.c. proper convex function) transporting $\mu$ to $\nu$ must coincide $\mu$-a.e. with $\nabla\varphi$**, regardless of whether one knew a priori that $(\mathrm{id},\nabla\tilde\varphi)_\\#\mu$ was optimal.

In other words: under the "no mass on small sets" assumption, **there is essentially a single gradient of a convex function pushing $\mu$ forward to $\nu$**. The whole optimal-transport question, in the quadratic case, reduces to finding that one convex potential.

</div>

#### Where this leaves us

Brenier's theorem hands us, for the quadratic cost, **three coupled objects**:

| Primal | Dual | Map |
|---|---|---|
| optimal coupling $\pi_\ast\in\Pi(\mu,\nu)$ | optimal potential pair $(\varphi,\varphi^\ast)$ | optimal map $T=\nabla\varphi$ |
| supported on $\operatorname{graph}\partial\varphi$ | conjugate pair, $\mu$-a.e. equality in Fenchel | curl-free, monotone |

The Kantorovich relaxation, which a priori is *strictly* more general than Monge, turns out to be **non-strict** for absolutely continuous $\mu$ — every quadratic-cost optimal plan is concentrated on a single-valued map. Strictly speaking, this is true whenever $\mu$ does not charge $(d-1)$-dimensional sets; absolute continuity with respect to Lebesgue is the standard sufficient condition.

This closes the loop of the chapter: Monge's hard problem $\to$ Kantorovich's tractable relaxation $\to$ Brenier's identification of the optimum as $\nabla\varphi$. The Monge–Ampère equation $\det D^2\varphi=f/(g\circ\nabla\varphi)$ is the corresponding **PDE characterisation** of $\varphi$, and is the starting point for the regularity theory of optimal transport, the Otto calculus on Wasserstein space, and Wasserstein gradient flows — all topics that fit naturally on top of what we have built.

### 2.4 Brenier's polar factorization

Brenier's theorem in §2.3 takes *two probability measures* $(\mu,\nu)$ and singles out the optimal map from one to the other as the gradient of a convex function. **Polar factorization** is the parallel statement at the level of *maps*: any sufficiently non-degenerate vector-valued map $h\colon\Omega\to\mathbb R^d$ admits a unique decomposition

$$
h \;=\; \nabla\psi\circ s, \qquad s\in S(\Omega), \quad \psi\text{ convex},
$$

where $s$ is **measure-preserving** and $\nabla\psi$ is the gradient of a convex function. The name is borrowed from linear algebra:

| Matrices: $A=UP$ | Maps: $h=\nabla\psi\circ s$ |
|---|---|
| $U$ orthogonal (rotation) | $s$ measure-preserving (volume-rotation) |
| $P=\sqrt{A^\top A}$ symmetric positive (stretch) | $\nabla\psi$ gradient of convex $\psi$ (irrotational stretch) |

A real matrix decomposes uniquely (modulo non-invertibility) as an isometry composed with a symmetric positive-semidefinite stretch; a non-degenerate $L^2$ map decomposes uniquely as a measure-preserving rearrangement composed with a convex-gradient stretch. The "rotation" part now lives on the infinite-dimensional sphere $S(\Omega)\subset L^2(\Omega;\mathbb R^d)$, and the "stretch" is the optimal transport map from §2.3.

Before stating the theorem, we set up the two building blocks: rearrangements and measure-preserving maps.

#### Rearrangements and measure-preserving maps

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 17</span><span class="math-callout__name">(Rearrangement)</span></p>

Let $(W,\lambda)$ and $(X,\mu)$ be measure spaces and let $m\colon W\to X$ be a measurable map. We call $\tilde m\colon W\to X$ a **rearrangement** of $m$ if

$$
\int_W F\circ m\,d\lambda \;=\; \int_W F\circ\tilde m\,d\lambda
$$

for every measurable function $F\colon X\to\mathbb R$ such that $F\circ m,\;F\circ\tilde m\in L^1(\lambda)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reading the definition)</span></p>

The condition is best understood through its consequence: testing against every $F$ pins down the **push-forward**. Indeed, $\tilde m$ is a rearrangement of $m$ if and only if $\tilde m_\sharp\lambda = m_\sharp\lambda$. The two maps may be wildly different point-by-point, but they distribute mass on $X$ identically.

* **Why call it a "rearrangement"?** Picture $\lambda$ as a pile of unit mass on $W$ and $m$ as the instructions for where each speck of mass ends up in $X$. A rearrangement gives different instructions that pile the mass the same way in $X$. The mass distribution at the destination is preserved; the labelling of which speck went where is not.
* **Special case.** In 1D, the "monotone rearrangement" of a function is the unique increasing rearrangement — the quantile function of the push-forward measure. This is exactly the same construction.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 18</span><span class="math-callout__name">(Measure-preserving maps)</span></p>

Let $(W,\lambda)$ be a measure space and $s\colon W\to W$ a measurable map. We call $s$ **measure-preserving** if $s_\sharp\lambda=\lambda$, i.e.

$$
\int_W F\circ s\,d\lambda \;=\; \int_W F\,d\lambda
$$

for every measurable $F\colon W\to\mathbb R$ with $F\circ s,\,F\in L^1(\lambda)$. The space of all measure-preserving maps on $(W,\lambda)$ is denoted $S(W)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The unit-Jacobian characterisation)</span></p>

In the cases relevant to us, $W=\Omega\subset\mathbb R^d$ is a bounded domain and $\lambda$ is Lebesgue measure. For $C^1$-diffeomorphisms there is a clean pointwise criterion:

$$
s\colon\Omega\to\Omega \text{ is a }C^1\text{-diffeo and measure-preserving} \;\iff\; \bigl|\det\nabla s(x)\bigr|=1 \;\text{ for all }x\in\Omega.
$$

This is the change-of-variables formula in disguise. For any smooth $F$,

$$
\int_\Omega F(s(x))\,dx \;=\; \int_\Omega F(y)\,\frac{1}{|\det\nabla s(s^{-1}(y))|}\,dy,
$$

so identity with $\int F\,dy$ forces $\lvert\det\nabla s\rvert=1$ a.e. The **absolute value** is essential: measure preservation only cares about *volume*, not *orientation*. Orientation-reversing diffeomorphisms with $\det\nabla s=-1$ are still measure-preserving.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 19</span><span class="math-callout__name">(Diffeomorphism groups)</span></p>

The group of diffeomorphisms on $\Omega$ is denoted by $\operatorname{Diff}(\Omega)$. The subgroup of **measure-preserving** diffeomorphisms (those with $\lvert\det\nabla s\rvert=1$) is denoted by $\operatorname{SDiff}(\Omega)$, and the subgroup of diffeomorphisms with $\det\nabla s=1$ (orientation- *and* volume-preserving) is denoted by $G(\Omega)$.

The chain of inclusions is

$$
G(\Omega) \;\subset\; \operatorname{SDiff}(\Omega) \;\subset\; \operatorname{Diff}(\Omega).
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why three groups, not two)</span></p>

The distinction $G(\Omega)\subsetneq\operatorname{SDiff}(\Omega)$ matters in fluid mechanics. The flow of an incompressible Euler equation is a curve in $G(\Omega)$ — the determinant is $+1$ all the way from the identity, not just $\pm 1$ — because it is a *continuous deformation from* $\mathrm{id}$, and identifying $\det\nabla\phi(0,\cdot)=1$ rules out a jump to $-1$ along the way. So while $\operatorname{SDiff}(\Omega)$ is the right algebraic object (closed under composition, contains all volume-preserving diffeomorphisms), $G(\Omega)$ is the right *topological component* for trajectories of incompressible flows. The Arnold geodesic interpretation later in this section will live in $G(\Omega)$, not in $\operatorname{SDiff}(\Omega)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 20</span><span class="math-callout__name">(Measure-preserving maps and rearrangements)</span></p>

Let $(W,\lambda)$ and $(X,\mu)$ be measure spaces and $m\colon W\to X$ a measurable map. Then:

**(i)** If $s\in S(W)$ and $m\colon W\to X$ is measurable, then $\tilde m:=m\circ s$ is a rearrangement of $m$.

**(ii)** If $\tilde m$ is a rearrangement of $m$ and $\tilde m\colon W\to X$ is *invertible*, then $\tilde m^{-1}\circ m\in S(W)$.

</div>

<details class="proof" markdown="1">
<summary>Proof of Proposition 20 — direct calculation</summary>

**(i)** For any test function $F\colon X\to\mathbb R$,

$$
\int_W F\circ\tilde m\,d\lambda \;=\; \int_W F\circ m\circ s\,d\lambda \;=\; \int_W F\circ m\,d\lambda,
$$

where the second equality uses $s_\sharp\lambda=\lambda$ applied to $F\circ m$ in place of $F$.

**(ii)** Set $\sigma:=\tilde m^{-1}\circ m$. For any test function $G\colon W\to\mathbb R$ such that $G,\;G\circ\sigma\in L^1(\lambda)$, apply the rearrangement identity to $F:=G\circ\tilde m^{-1}$ (defined on $X$ wherever $\tilde m^{-1}$ exists):

$$
\int_W F\circ\tilde m\,d\lambda \;=\; \int_W F\circ m\,d\lambda
\;\Longleftrightarrow\; \int_W G\,d\lambda \;=\; \int_W G\circ\sigma\,d\lambda.
$$

That is the defining identity $\sigma_\sharp\lambda=\lambda$, so $\sigma\in S(W)$. $\square$

</details>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why this is an "almost" equivalence)</span></p>

Proposition 20 says rearrangements and measure-preserving maps are *two sides of the same coin* — but only almost. Direction (i) is unconditional: composing with an MP map on the source always gives a rearrangement. Direction (ii) requires **invertibility** of $\tilde m$.

* Without invertibility (ii) fails. Two rearrangements of the same $m$ can differ in genuinely non-MP ways: e.g., on $W=[0,1]$ with $m(x)=0$ identically, *every* map $\tilde m\colon W\to X$ with image $\\{0\\}$ is trivially a rearrangement of $m$, and these can certainly fail to be related by an MP map of $W$.
* With invertibility (ii) is sharp. Polar factorization, which we are heading toward, will produce a rearrangement of the form $\tilde m=\nabla\psi$. Whether $\nabla\psi$ is invertible is exactly the question of whether the convex potential $\psi$ is strictly convex — a question of regularity, not algebra.

The takeaway: the rearrangement concept is the *primary* one in optimal transport. Measure-preserving maps are how rearrangements are *generated* when one has enough invertibility — but this is not automatic.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 21</span><span class="math-callout__name">($S(W)$ is curved, not linear)</span></p>

Assume the second moment $\int_X\\|x\\|^2\,d\lambda(x)<+\infty$. Then every measure-preserving map $s\in S(W)$ automatically sits in $L^2(\lambda)$, and they **all have the same $L^2$-norm**:

$$
\|s\|_{L^2(\lambda)}^2 \;=\; \int_W |s(w)|^2\,d\lambda(w) \;=\; \int_W |w|^2\,d\lambda(w) \;=\; \|\mathrm{id}\|_{L^2(\lambda)}^2,
$$

the second equality by the definition $s_\sharp\lambda=\lambda$ applied to $F(w)=\\|w\\|^2$. So $S(W)\subset L^2(\lambda)$ lies on a single **sphere** centred at the origin.

In particular, $S(W)$ is a **curved subset** of $L^2(\lambda)$, not a linear subspace. The convex combination $\tfrac12(s_1+s_2)$ of two MP maps is generally *not* MP — it lies strictly inside the sphere. This is why Brenier's theorem produces an $L^2$-projection onto $S(W)$ rather than a linear projection: the geometry is that of a sphere in Hilbert space.

</div>

#### The polar factorization theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 22</span><span class="math-callout__name">(Brenier's polar factorization)</span></p>

Let $\Omega\subset\mathbb R^d$ be a bounded domain and $\lambda$ Lebesgue measure on $\Omega$. Let $h\in L^2(\Omega;\mathbb R^d)$ be **non-degenerate** in the sense that

$$
\lambda\bigl(h^{-1}(N)\bigr) \;=\; 0 \qquad\text{for every small set }N\subset\mathbb R^d, \tag{2.16}
$$

i.e. the push-forward $\mu:=h_\sharp\lambda$ does not give mass to small sets (cf. Remark 14).

Then there exists a unique pair $(s,\nabla\psi)$ such that

* $\nabla\psi$ is a rearrangement of $h$ in the class of $L^2$ gradients of convex functions,
* $s\in S(\Omega)$ is a measure-preserving map,
* $h \;=\; \nabla\psi\circ s$.

Moreover, **$s$ is the unique $L^2$-projection of $h$ onto $S(\Omega)$**:

$$
\|h-s\|_{L^2}^2 \;=\; \inf_{s'\in S(\Omega)}\|h-s'\|_{L^2}^2. \tag{2.17}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Three facets of the same statement)</span></p>

Theorem 22 packages three statements that deserve to be separated:

1. **Existence of a polar decomposition.** Any non-degenerate $h$ factors as $h=\nabla\psi\circ s$ with $\nabla\psi$ a convex-gradient and $s$ measure-preserving.
2. **Uniqueness.** Both factors $\nabla\psi$ and $s$ are uniquely determined ($\lambda$-a.e.) by $h$.
3. **Variational characterisation of $s$.** Among all measure-preserving maps, $s$ is the one closest to $h$ in $L^2$.

Statement (3) is the geometric content — it says the polar decomposition computes the **orthogonal projection** of the map $h$ onto the (curved) set $S(\Omega)$. The convex potential $\psi$ is then the residual "stretch" needed to reach $h$ from its closest MP approximation.

Compare to matrices: for an invertible matrix $A$, the polar factor $U$ from $A=UP$ is the closest orthogonal matrix to $A$ in Frobenius norm. The analogy is exact: the rotation part is the closest isometry, in both finite and infinite dimensions.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why "non-degenerate")</span></p>

The hypothesis (2.16) is exactly what makes Brenier's theorem applicable to $\mu=h_\sharp\lambda$: it says $\mu$ does not give mass to small sets, which is the assumption under which the optimal plan from $\mu$ to $\nu$ is single-valued (cf. Remark 14, Theorem 13).

Concretely: a $C^1$-diffeomorphism $h$ with $\det\nabla h\neq 0$ a.e. has Jacobian-bounded push-forward, hence $\mu=h_\sharp\lambda$ is absolutely continuous, and (2.16) holds automatically. Conversely, a map $h$ that collapses positive-volume regions to lower-dimensional sets — e.g. $h(x)=(x_1,0)$ on $\Omega\subset\mathbb R^2$ — fails (2.16) and cannot be polar-factored in the strong sense above.

</div>

#### Motivation: incompressible Euler and projection onto $G(\Omega)$

A key motivation for Brenier was **fluid mechanics**, where projection onto measure-preserving diffeomorphisms is the natural way to enforce the incompressibility constraint. The following formal discussion shows why.

The simplest model of an incompressible fluid is the **incompressible Euler equation** for the velocity field $v\colon [0,T)\times\Omega\to\mathbb R^d$ in a container $\Omega\subset\mathbb R^d$ (a bounded open set with smooth boundary). Starting from initial data $v_0\colon\Omega\to\mathbb R^d$ with $\nabla\cdot v_0=0$ and $v_0\cdot n=0$ on $\partial\Omega$, the equations read

$$
\partial_t v + (v\cdot\nabla)v \;=\; -\nabla p, \tag{2.10}
$$

$$
\nabla\cdot v \;=\; 0, \tag{2.11}
$$

with **free-slip** boundary condition $v\cdot n=0$ on $\partial\Omega$. The pressure $p\colon\Omega\times[0,T)\to\mathbb R$ is an *unknown* of the system, acting as a **Lagrange multiplier** for the incompressibility constraint (2.11).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 23</span><span class="math-callout__name">(Energy conservation for smooth solutions to Euler)</span></p>

Let $v$ be a smooth solution to the incompressible Euler equations (2.10)–(2.11) on $[0,T)\times\Omega$. Then the kinetic energy is conserved in time:

$$
\int_\Omega \tfrac12 |v(x,t)|^2\,dx \;=\; \int_\Omega \tfrac12 |v_0(x)|^2\,dx \qquad\text{for all }t\in[0,T).
$$

<details class="proof" markdown="1">
<summary>Proof — momentum equation, IBP, boundary &amp; incompressibility</summary>

Differentiate the kinetic energy and use (2.10):

$$
\begin{aligned}
\frac{d}{dt}\int_\Omega\tfrac12|v|^2\,dx
&= \int_\Omega v\cdot\partial_t v\,dx \\
&= -\int_\Omega v\cdot\bigl((v\cdot\nabla)v+\nabla p\bigr)\,dx.
\end{aligned}
$$

The convective term is a perfect gradient: using $v\cdot(v\cdot\nabla)v = v\_j\partial\_j v\_i\cdot v\_i = \tfrac12 v\_j\partial\_j\\|v\\|^2 = v\cdot\nabla(\tfrac12\\|v\\|^2)$,

$$
\frac{d}{dt}\int_\Omega\tfrac12|v|^2\,dx \;=\; -\int_\Omega v\cdot\nabla\bigl(\tfrac12|v|^2+p\bigr)\,dx.
$$

Integrate by parts:

$$
\frac{d}{dt}\int_\Omega\tfrac12|v|^2\,dx \;=\; -\int_{\partial\Omega}(v\cdot n)\bigl(\tfrac12|v|^2+p\bigr)\,dS \;+\; \int_\Omega (\nabla\cdot v)\bigl(\tfrac12|v|^2+p\bigr)\,dx.
$$

The boundary integral vanishes by the free-slip condition $v\cdot n=0$; the bulk integral vanishes by incompressibility $\nabla\cdot v=0$. Hence kinetic energy is conserved. $\square$

</details>

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Failure for weak solutions — Onsager's conjecture)</span></p>

Energy conservation in Proposition 23 is **not** true for *weak* solutions of Euler. The celebrated work of **De Lellis–Székelyhidi** and **Isett** constructs weak solutions that dissipate energy — and this dissipation can even be prescribed. The threshold is the **Onsager conjecture**: $C^{0,\alpha}$ regularity with $\alpha>1/3$ implies energy conservation (proved); $\alpha<1/3$ allows dissipation (proved via convex integration). The case $\alpha=1/3$ is the borderline. The relevance here is that *smoothness is essential* to the proof above — the integration-by-parts manipulations break down without it.

</div>

##### Eulerian vs Lagrangian, and the flow map

Fluids (and physical systems generally) admit two complementary descriptions:

* **Eulerian:** record the velocity $v(x,t)$ at each fixed point $x\in\Omega$. The unknown is a *field*.
* **Lagrangian:** track each individual particle along its trajectory $x(t)$ starting from $x(0)=x_0$. The unknown is a *flow*.

The link between the two pictures is the **flow map**

$$
\phi\colon[0,T)\times\Omega\to\Omega, \qquad \phi(t,x_0):=x(t),
$$

where $x(t)$ solves $\dot x(t)=v(t,x(t))$ with $x(0)=x_0$.

The incompressibility constraint (2.11) translates exactly into a constraint on $\phi$:

$$
\nabla\cdot v \;=\; 0 \;\iff\; \phi(t,\cdot)\colon\Omega\to\Omega \text{ is measure-preserving for all }t\in[0,T). \tag{2.12}
$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why (2.12) is the right translation)</span></p>

The mechanical content of "incompressible" is that *no volume gets created or destroyed* as the fluid moves. Eulerian-side that says the velocity field has zero divergence; Lagrangian-side it says the flow map preserves volumes. These are the same statement viewed from two coordinate systems.

Sketch of (2.12): for sufficiently smooth $v$, the Jacobian $J(t,x_0):=\det\nabla_{x_0}\phi(t,x_0)$ satisfies $\partial_t J = (\nabla\cdot v)(\phi(t,x_0),t)\cdot J$ (Liouville's formula). With $J(0,\cdot)=1$, the equation $\partial_t J=0$ is equivalent to $\nabla\cdot v=0$, giving $J\equiv 1$, i.e. $\phi(t,\cdot)\in G(\Omega)$.

</div>

##### The Lagrangian form of Euler

The first Euler equation (2.10) is **Newton's second law** for the fluid: the only force on a parcel is $-\nabla p$, so

$$
-(\nabla p)(x(t),t) \;=\; \rho\,\frac{d^2}{dt^2}x(t).
$$

Differentiating the trajectory identity $\frac{d}{dt}\phi(t,x_0)=v(\phi(t,x_0),t)$ once more in time gives, by the chain rule,

$$
\frac{d^2}{dt^2}\phi(t,x_0) \;=\; \partial_t v + (v\cdot\nabla)v,
$$

so the Lagrangian form of (2.10) is simply

$$
\boxed{\;\frac{d^2}{dt^2}\phi \;=\; -\nabla p\circ\phi,\qquad \phi\colon[0,T)\to G(\Omega).\;} \tag{2.10'}
$$

The two equations have a clean correspondence:

| Eulerian | Lagrangian |
|---|---|
| (2.10) momentum balance | (2.10') $\ddot\phi=-\nabla p\circ\phi$ |
| (2.11) $\nabla\cdot v=0$ | $\phi(t,\cdot)\in G(\Omega)$ |
| (free-slip) $v\cdot n=0$ on $\partial\Omega$ | $\phi(t,\Omega)=\Omega$ |

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Arnold's interpretation</span><span class="math-callout__name">(Euler = geodesic on $G(\Omega)$)</span></p>

Equation (2.10') can be read as the **geodesic equation** on the (infinite-dimensional) manifold $G(\Omega)$, with respect to the $L^2$-metric on velocities. Heuristically:

* The "manifold" is $G(\Omega)\subset\operatorname{Diff}(\Omega)$ — orientation- and volume-preserving diffeomorphisms.
* The "tangent space" at $\phi$ consists of vector fields $w$ on $\Omega$ with $\nabla\cdot w=0$ (the linearisation of the incompressibility constraint).
* The "metric" is the $L^2$ inner product $\langle w_1,w_2\rangle=\int_\Omega w_1\cdot w_2\,dx$ — that is, kinetic energy.
* The "Christoffel symbol" — the curvature-correction needed to keep $\phi$ on $G(\Omega)$ — comes from the **pressure gradient** $-\nabla p\circ\phi$. The pressure is precisely the Lagrange multiplier projecting the unconstrained acceleration $\partial_t v+(v\cdot\nabla)v$ back onto the tangent space of $G(\Omega)$.

This is **Arnold's observation** (1966). It reframes Euler from "PDE for a vector field" to "geodesic flow on an infinite-dimensional Lie group", which:

* explains energy conservation as constant-speed geodesic motion;
* connects fluid stability to sectional curvature (Arnold's stability theorem);
* sets the stage for Brenier: *projecting* a non-MP map onto $G(\Omega)$ is exactly the polar factorization problem, and it is the discretized analogue of the constraint projection that produces $-\nabla p\circ\phi$ in the continuous flow.

</div>

#### A more general statement

Theorem 22 is the convenient form where source and target both live on $\Omega$ with Lebesgue measure. The general statement allows different domains and reference measures.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 24</span><span class="math-callout__name">(Brenier's polar factorization, general version)</span></p>

Let $W,X,Y\subset\mathbb R^d$ be measurable, $\lambda\in\mathcal P(W)$, $\nu\in\mathcal P(Y)$ with $\int_Y\\|y\\|^2\,d\nu(y)<+\infty$. Assume both $\mu:=h_\sharp\lambda$ and $\nu$ do not give mass to small sets, and let $h\colon W\to X$ with $h\in L^2(\lambda)$.

Then there exists a unique pair $(s,\nabla\psi)$ such that

**(i)** $s\colon W\to Y$ pushes $\lambda$ forward to $\nu$: $s_\sharp\lambda=\nu$,

**(ii)** $\psi\colon Y\to X$ is the restriction of a convex function on $\mathbb R^d$, and

**(iii)** $h \;=\; \nabla\psi\circ s$ $\lambda$-a.e.

Moreover, $s$ is the unique $L^2(\lambda)$-orthogonal projection of $h$ onto

$$
S(W,Y) \;:=\; \{\sigma\colon W\to Y : \sigma_\sharp\lambda=\nu\}.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What's the role of $\nu$?)</span></p>

Two extra knobs distinguish Theorem 24 from Theorem 22:

* The target domain $Y$ may differ from the source $W$.
* The reference measure $\nu$ on $Y$ is **not** taken to be the push-forward $h_\sharp\lambda$; it is supplied separately.

Concretely, $S(W,Y)=\\{\sigma\colon\sigma_\sharp\lambda=\nu\\}$ is the set of "$\lambda$-to-$\nu$ rearrangements" — and the projection of $h$ onto this set is the part of $h$ that respects the target's mass distribution. The convex-gradient $\nabla\psi$ then transports $\nu$ to $\mu=h_\sharp\lambda$, which closes the diagram

$$
\lambda \xrightarrow{\;s\;} \nu \xrightarrow{\;\nabla\psi\;} \mu \quad=\quad \lambda \xrightarrow{\;h\;} \mu.
$$

In Theorem 22, by contrast, $W=Y=\Omega$ and $\nu=\lambda$, so $S(W,Y)=S(\Omega)$ — the "rotation part" is a self-map of $\Omega$.

Also, $\sigma\in S(W,Y)$ all share the same $L^2$-norm (since $\\|\sigma\\|^2\_{L^2(\lambda)}=\int_Y\\|y\\|^2\,d\nu<\infty$ depends only on $\nu$), so $S(W,Y)$ lies on a sphere in $L^2(\lambda)$, generalising Remark 21.

</div>

#### Proof of Theorem 22

The proof has a beautiful structural idea: **reduce the $L^2$-projection problem onto the curved set $S(W,Y)$ to the standard linear OT problem from §2.3, then read off the polar factorization from Brenier's theorem.**

<details class="proof" markdown="1">
<summary>Proof of Theorem 22 — four steps</summary>

**Step 1: Reformulation as an OT problem.** We want the $s\in S(W,Y)$ that minimises the $L^2$-distance to $h$:

$$
\min_{\sigma\in S(W,Y)} \int |h(w)-\sigma(w)|^2\,d\lambda(w). \tag{2.13}
$$

For any candidate $\sigma$, define the transference plan $\pi:=(h,\sigma)_\sharp\lambda$. By construction $\pi$ has marginals $\mu=h_\sharp\lambda$ (first coordinate) and $\nu=\sigma_\sharp\lambda$ (second coordinate), so $\pi\in\Pi(\mu,\nu)$. Pushing forward the integrand,

$$
\int_W |h(w)-\sigma(w)|^2\,d\lambda(w) \;=\; \int_{X\times Y}|x-y|^2\,d\pi(x,y),
$$

so (2.13) becomes

$$
\min\Bigl\{\int_{X\times Y}|x-y|^2\,d\pi(x,y)\colon \pi=(h,\sigma)_\sharp\lambda,\ \sigma_\sharp\lambda=\nu\Bigr\}. \tag{2.14}
$$

This is *almost* a standard OT problem — except the plan is constrained to come from a single $\sigma$. We relax this constraint:

$$
\min\Bigl\{\int_{X\times Y}|x-y|^2\,d\pi(x,y)\colon \pi\in\Pi(\mu,\nu)\Bigr\}. \tag{2.15}
$$

The relaxed problem (2.15) is the standard quadratic-cost Kantorovich problem from §2.3. Any minimiser of (2.14) is admissible in (2.15), so

$$
\min(2.15) \;\le\; \min(2.14).
$$

The plan of attack — explained in the bracketed comment of the manuscript — is: solve (2.15) using Brenier, recover an *admissible plan for (2.14)* from the Brenier map, and observe that this plan attains the lower bound, hence is optimal for *both* problems.

**Step 2: Existence and uniqueness of the OT map.** Both hypotheses of Theorem 13 hold:

* Finite second moments: $\int_Y \\|y\\|^2\,d\nu<+\infty$ is assumed, and $\int_X\\|x\\|^2\,d\mu=\int_X\\|x\\|^2\,d(h_\sharp\lambda)=\int_W\\|h(w)\\|^2\,d\lambda(w)<+\infty$ because $h\in L^2(\lambda)$.
* Neither $\mu$ nor $\nu$ gives mass to small sets (by assumption).

So Theorem 13 (in its symmetric form) yields a pair $(\varphi,\varphi^\ast)$ of convex conjugates, uniquely determined $\mu$- and $\nu$-a.e., with **mutually inverse gradients** ($\nabla\varphi^\ast\circ\nabla\varphi=\mathrm{id}$ $\mu$-a.e. and $\nabla\varphi\circ\nabla\varphi^\ast=\mathrm{id}$ $\nu$-a.e.), solving the dual Kantorovich problem and satisfying

$$
\nu \;=\; (\nabla\varphi)_\sharp\mu \;=\; (\nabla\varphi)_\sharp(h_\sharp\lambda) \;=\; (\nabla\varphi\circ h)_\sharp\lambda.
$$

**Step 3: Existence of the polar decomposition.** Define

$$
s \;:=\; \nabla\varphi\circ h.
$$

The pushforward identity in Step 2 gives $s_\sharp\lambda=\nu$, so $s\in S(W,Y)$. Set $\pi:=(h,s)_\sharp\lambda$. By construction $\pi=(\mathrm{id},\nabla\varphi)_\sharp\mu$ is concentrated on the graph of $\nabla\varphi$, so by Brenier (Theorem 13(ii)) $\pi$ is the **unique** solution of (2.15). It is admissible for (2.14), so it solves (2.14), so $s$ solves the projection problem (2.13).

Now define $\psi:=\varphi^\ast$ — a convex function — and check that $\psi$ rebuilds $h$ from $s$. Since $\nabla\psi\circ\nabla\varphi=\nabla\varphi^\ast\circ\nabla\varphi=\mathrm{id}$ $\mu$-a.e., and $h_\sharp\lambda=\mu$,

$$
\nabla\psi\circ s \;=\; \nabla\psi\circ\nabla\varphi\circ h \;=\; h \qquad \lambda\text{-a.e.}
$$

Hence $(s,\psi)$ is a polar factorization of $h$.

**Step 4: Uniqueness of the polar decomposition.** Uniqueness of $\nabla\psi$ (as a gradient of a convex function rearrangement of $h$) is direct from Brenier's uniqueness in Step 2. It remains to show **uniqueness of the projection $s$**.

Let $s'$ be another $L^2$-projection of $h$ onto $S(W,Y)$. Both $s$ and $s'$ realise the minimum in (2.13), so the plans $(h,s)_\sharp\lambda$ and $(h,s')_\sharp\lambda$ both minimise (2.14), hence both are admissible for (2.15) at the same cost — and by uniqueness in Step 2,

$$
(h,s)_\sharp\lambda \;=\; (h,s')_\sharp\lambda. \tag{$\dagger$}
$$

*(Equality of joint pushforwards does **not** imply $s=s'$ $\lambda$-a.e. directly — two different $W\to Y$ maps can produce identical $X\times Y$ pushforwards if $h$ collapses regions where $s$ and $s'$ differ.)*

To upgrade ($\dagger$) to $s=s'$ $\lambda$-a.e., test against the bilinear form $F(x,y):=\nabla\varphi(x)\cdot y$:

$$
\int_W F(h(w),s(w))\,d\lambda(w) \;=\; \int_W F(h(w),s'(w))\,d\lambda(w).
$$

The left integrand evaluates to $(\nabla\varphi\circ h)\cdot s = s\cdot s = \\|s\\|^2$ $\lambda$-a.e. (using the definition $s=\nabla\varphi\circ h$). The right integrand evaluates to $(\nabla\varphi\circ h)\cdot s' = s\cdot s'$ $\lambda$-a.e. So

$$
\int_W |s|^2\,d\lambda \;=\; \int_W s\cdot s'\,d\lambda. \tag{$\ddagger$}
$$

But by Remark 21 (or its analogue in $S(W,Y)$), $\int_W \\|s\\|^2\,d\lambda = \int_W \\|s'\\|^2\,d\lambda$, so ($\ddagger$) gives

$$
0 \;=\; \int_W |s|^2 - s\cdot s'\,d\lambda \;=\; \tfrac12\int_W \bigl(|s|^2 + |s'|^2 - 2s\cdot s'\bigr)\,d\lambda \;=\; \tfrac12\int_W |s-s'|^2\,d\lambda,
$$

forcing $s=s'$ $\lambda$-a.e. $\square$

</details>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The "Curiously…" of Step 4)</span></p>

The parenthetical aside in Step 4 — "this does not in general imply that $s=s'$ $\lambda$-a.e." — is more than a technicality. It points to the fact that *equal joint distributions on $X\times Y$ do not pin down a $W\to Y$ map* if $h$ has degeneracies.

Concrete obstruction: suppose $h$ is constant on a set $A\subset W$ of positive measure. Then on $A$, any reshuffling of $s$ leaves $(h,s)\_\sharp\lambda$ unchanged (the first coordinate is constant, so only the marginal distribution of $s\rvert\_A$ matters, not the pointwise values). So degeneracy of $h$ creates a *gauge symmetry* of plans-to-maps that ($\dagger$) cannot detect.

The non-degeneracy hypothesis (2.16) suppresses exactly this gauge: by Brenier's $\mu$-a.e. uniqueness, $\nabla\varphi$ has a well-defined inverse $\nabla\varphi^\ast$, and the test against $F=\nabla\varphi\cdot y$ in Step 4 *uses* this inverse to pull the $X$-side information back down to $W$, eliminating the gauge ambiguity. Geometrically, the test function $F$ is the **unique** bilinear form for which the Step 4 identity collapses cleanly — it is built precisely to convert the (degenerate-looking) equality of plans into the (non-degenerate) equality of $L^2$ norms.

</div>

## Appendix A: Desingularizing Functions and the Kurdyka–Łojasiewicz Framework {#appendix-a}

In §1.6, the proof of long-term asymptotics via Łojasiewicz uses an unusual move: rather than tracking the excess energy $\mathcal E(t):=E(x(t))-E_\infty$ along the gradient flow, it tracks the **concave power** $\mathcal E^{1-\theta}(t)$. Why this exact exponent, and not $\mathcal E$ itself, or $\sqrt{\mathcal E}$, or anything else? The answer is *not* a technicality — it points to a deep recurring template in analysis, often called **desingularization** or the **Kurdyka–Łojasiewicz (KL) framework**.

This appendix unpacks the principle at three levels: the narrow technical answer, why the exponent is forced, and the general framework that subsumes it.

### A.1 The narrow answer: matching units

#### A.1.1 The arc-length goal and the Lyapunov template

We want **finite arc-length** — the integral

$$\int_0^\infty \|\dot x(t)\|\,dt < \infty.$$

The reason is one short implication: finite length means the trajectory is *Cauchy*, and Cauchy in $\mathbb R^N$ means convergent. So once we control the integral above, the limit $x^\ast = \lim_{t\to\infty} x(t)$ exists automatically.

The whole machinery rests on a single recurring **template trick** in analysis:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Lyapunov template for finite arc-length)</span></p>

If we can find a scalar function $L(t)\ge 0$ that is non-increasing along the flow and satisfies

$$
\dot L(t) \;\le\; -\|\dot x(t)\| \qquad \text{for a.e. } t, \tag{A.0}
$$

then the trajectory has finite arc-length, with the explicit bound

$$
\int_0^\infty \|\dot x(t)\|\,dt \;\le\; L(0).
$$

</div>

The proof is one line. Integrate (A.0) from $0$ to $\infty$:

$$
\int_0^\infty \|\dot x(t)\|\,dt \;\le\; -\int_0^\infty \dot L(t)\,dt \;=\; L(0)-L(\infty) \;\le\; L(0).
$$

Note carefully: $L$ is a **scalar** quantity, just a real number depending on $t$. There is no vector field associated with $L$; the only thing that matters is its *time derivative* $\dot L(t)$ along the trajectory. The whole game of A.1–A.3 is to **construct such an $L$** out of the energy.

#### A.1.2 Why the energy $\mathcal E$ itself is the wrong $L$

The most obvious candidate is $L(t) := \mathcal E(t) = E(x(t))-E_\infty$. By the chain rule and the gradient-flow equation $\dot x = -\nabla E$,

$$
\dot{\mathcal E}(t) \;=\; \langle \nabla E(x(t)),\dot x(t)\rangle \;=\; -\|\nabla E(x(t))\|^2 \;=\; -\|\dot x(t)\|^2.
$$

So $\mathcal E$ **is** a perfectly good Lyapunov function — non-negative, non-increasing, and dissipating energy. The issue is only the **rate** at which it decreases:

| Template (A.0) wants | What $\mathcal E$ gives us |
|---|---|
| $\dot L \;\le\; -\|\dot x\|$ (first power of speed) | $\dot{\mathcal E} \;=\; -\|\dot x\|^2$ (squared speed) |

A single power of $\|\dot x\|$ short — but that one power is decisive.

Naively integrating $\dot{\mathcal E} = -\|\dot x\|^2$ gives

$$
\int_0^\infty \|\dot x(t)\|^2\,dt \;=\; \mathcal E(0)-\mathcal E_\infty \;<\; \infty,
$$

i.e., the **squared** speed integrates. This is the standard energy-dissipation identity (1.36). But $\int\|\dot x\|^2\,dt < \infty$ does **not** imply $\int\|\dot x\|\,dt < \infty$ — a slowly oscillating tail can have $\|\dot x\|^2$ integrable while $\|\dot x\|$ is not. So $\mathcal E$ controls the wrong norm of the speed, and energy decrease alone cannot conclude finite length.

#### A.1.3 What "matching units" actually means

We need to **modify** $\mathcal E$ so its derivative loses one factor of $\|\dot x\|$. The general ansatz: try $L = G(\mathcal E)$ for some smooth increasing $G$. Chain rule:

$$
\dot L(t) \;=\; G'(\mathcal E(t))\cdot \dot{\mathcal E}(t) \;=\; -G'(\mathcal E(t))\cdot \|\dot x(t)\|^2.
$$

For this to satisfy the template $\dot L \le -\|\dot x\|$ we need

$$
G'(\mathcal E)\cdot \|\dot x\|^2 \;\ge\; \|\dot x\| \quad\Longleftrightarrow\quad \boxed{\;G'(\mathcal E)\cdot \|\dot x\| \;\ge\; 1\;} \tag{A.0'}
$$

uniformly along the trajectory — even as $\mathcal E\to 0$ and $\|\dot x\|\to 0$.

This is what "matching units" means. It is a **product** condition between two factors that are both shrinking:

* $\|\dot x\|$ is shrinking because the trajectory is slowing down.
* $G'(\mathcal E)$ must therefore **grow** at the same rate, so that the product (A.0') stays bounded below by $1$.

The Łojasiewicz inequality is precisely the input that makes this possible: it tells us *how fast* $\|\dot x\| = \|\nabla E\|$ can shrink as a function of the energy gap,

$$\|\dot x\| \;=\; \|\nabla E\| \;\ge\; \mathcal E^\theta / C.$$

Plugging into (A.0'), the requirement becomes $G'(\mathcal E)\cdot \mathcal E^\theta/C \ge 1$, i.e.

$$G'(\mathcal E) \;\ge\; \frac{C}{\mathcal E^\theta}.$$

The minimal $G$ — the one that just barely satisfies the bound — is the **antiderivative**:

$$
G(\mathcal E) \;=\; \int_0^{\mathcal E} \frac{C}{\sigma^\theta}\,d\sigma \;=\; \frac{C}{1-\theta}\,\mathcal E^{1-\theta}.
$$

That is the **desingularizing function** $\varphi$, and the appearance of the exponent $1-\theta$ is now demystified: it is the unique power that (a) integrates the singular slope $\sigma^{-\theta}$ near $0$, and (b) makes $G'(\mathcal E)\cdot\|\dot x\|$ bounded below by a constant along the flow — i.e., the unique power that turns the wrong-rate Lyapunov function $\mathcal E$ into a right-rate one. Sections A.2 and A.3 below pick up from here and verify the forcing in two different ways.

### A.2 Why the exponent $1-\theta$ is forced

Try $G(\mathcal E) := \mathcal E^\alpha$ for some $\alpha\in(0,1)$. Differentiating along the flow,

$$\frac{d}{dt}\mathcal E^\alpha = \alpha\,\mathcal E^{\alpha-1}\dot{\mathcal E} = -\alpha\,\mathcal E^{\alpha-1}|\dot x|^2 = -\alpha\,\mathcal E^{\alpha-1}|\nabla E|\,|\dot x|.$$

Now apply **Łojasiewicz** $\mathcal E^\theta \le C\|\nabla E\|$, i.e. $\|\nabla E\|\ge \mathcal E^\theta/C$:

$$\frac{d}{dt}\mathcal E^\alpha \;\le\; -\frac{\alpha}{C}\,\mathcal E^{\alpha-1+\theta}\,|\dot x|.$$

For this to control $\|\dot x\|$ uniformly — independently of how small $\mathcal E$ has become — we need

$$\boxed{\;\alpha-1+\theta = 0 \;\Longleftrightarrow\; \alpha = 1-\theta.\;}$$

Any other exponent fails:

* If $\alpha > 1-\theta$, the exponent of $\mathcal E$ on the right is positive, so the bound on $\|\dot x\|$ degrades to $0$ as $\mathcal E\to 0$ — useless near the limit.
* If $\alpha < 1-\theta$, the exponent is negative, and the bound *blows up* as $\mathcal E\to 0$ — the inequality formally survives, but you no longer get a clean integrable bound.

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/appendix_a_exponent_choice.png' | relative_url }}" alt="Two-panel figure: left panel shows the Łojasiewicz factor E^(α-1+θ) along a 1D Łojasiewicz gradient flow for three values of α — only α=1-θ yields a constant-in-t factor, α<1-θ blows up, α>1-θ decays to zero; right panel compares cumulative arc length to E^(1-θ) and E itself, with E^(1-θ) tracking the remaining length while E decays too fast" loading="lazy">
  <figcaption>Why $\alpha=1-\theta$ is forced. <strong>Left.</strong> Along the Łojasiewicz gradient flow on $E(x)=x^4/4$ (so $\theta=1/2$), the Łojasiewicz factor $\mathcal E^{\alpha-1+\theta}$ is constant in $t$ only at $\alpha=1-\theta=0.5$ (green); smaller $\alpha$ overshoots and the factor blows up (red), larger $\alpha$ degrades and the factor decays to $0$ (orange). <strong>Right.</strong> The cumulative arc length $\int_0^t\|\dot x\|\,ds$ (blue) converges to $1$. The right reparametrization $\mathcal E^{1-\theta}(t)$ (green dashed) tracks the <em>remaining</em> length; the bare energy $\mathcal E(t)$ (red dotted) decays too fast and undercounts.</figcaption>
</figure>

So $1-\theta$ is the **unique** exponent at which the units balance — the only place where the ratio $\mathcal E^{\alpha-1}\|\nabla E\|$ stays dimensionally constant along the flow.

### A.3 The general principle: desingularizing functions

Step back. The structure of the problem is:

* a **dissipation rate** $\dot{\mathcal E} = -\|\dot x\|^2$ (from the gradient-flow structure);
* a **slope inequality** $\|\nabla E\| \ge f(\mathcal E)$ for some non-decreasing $f:[0,\infty)\to[0,\infty)$ with $f(0)=0$ (for Łojasiewicz, $f(s)=s^\theta/C$);
* a desired **arc-length bound** $\int\|\dot x\|\,dt < \infty$.

The general question is: when can we extract finite length, and how?

**Construction.** Define the **desingularizing function**

$$\varphi(s) := \int_0^s \frac{1}{f(\sigma)}\,d\sigma. \tag{A.1}$$

This $\varphi$ is the unique (up to constants) primitive of $1/f$, and it is well-defined on a neighborhood of $0$ whenever $1/f$ is integrable near $0$. For Łojasiewicz, $f(s)=s^\theta/C$, so

$$\varphi(s) = C\int_0^s \sigma^{-\theta}\,d\sigma = \frac{C}{1-\theta}\,s^{1-\theta}.$$

**There it is** — the power $\mathcal E^{1-\theta}$ from §1.6 is *exactly the antiderivative of $1/f$*, up to the constant $C/(1-\theta)$.

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/appendix_a_desingularizing_function.png' | relative_url }}" alt="Two-panel figure: left shows desingularizing function φ(s)=s^(1-θ) plotted for θ=0.1,0.3,0.5,0.7,0.9, all concave and equal at s=1, with the dashed identity line for reference; right shows derivative φ'(s)=(1-θ)s^(-θ) on a log scale, blowing up as s→0 for every θ>0" loading="lazy">
  <figcaption>The desingularizing function $\varphi(s)=s^{1-\theta}$ for several Łojasiewicz exponents $\theta\in(0,1)$. <strong>Left.</strong> $\varphi$ is concave with $\varphi(0)=0$; the larger $\theta$, the more pronounced the concavity. <strong>Right.</strong> The slope $\varphi'(s)=(1-\theta)s^{-\theta}$ blows up at $s\to 0^+$ — that is precisely what "desingularizes" the flat critical point. The borderline $\theta\to 1^-$ is where $\varphi'$ becomes non-integrable, $\varphi$ ceases to be finite, and the finite-length argument fails.</figcaption>
</figure>

**General theorem.** Compute the time derivative of $\varphi(\mathcal E(t))$ along the gradient flow:

$$\frac{d}{dt}\,\varphi(\mathcal E(t)) \;=\; \varphi'(\mathcal E)\,\dot{\mathcal E} \;=\; -\frac{|\dot x|^2}{f(\mathcal E)} \;\le\; -|\dot x|,$$

where the last step uses $\|\dot x\|=\|\nabla E\|\ge f(\mathcal E)$. Integrating from $0$ to $\infty$,

$$\boxed{\;\int_0^\infty|\dot x|\,dt \;\le\; \varphi(\mathcal E(0)) - \varphi(\mathcal E_\infty) \;\le\; \varphi(\mathcal E(0)).\;}$$

So the deep statement is:

> **A gradient flow has finite length whenever the slope-vs-energy relation $\|\nabla E\|\ge f(\mathcal E)$ has an integrable reciprocal $1/f$ near $0$.**

Łojasiewicz with exponent $\theta\in(0,1)$ corresponds to $f(s)=s^\theta$, whose reciprocal $s^{-\theta}$ is integrable near $0$ iff $\theta<1$ — exactly the hypothesis in the theorem of §1.6. The borderline $\theta\to 1^-$ marks the regime where $\varphi$ blows up: $\int 1/f$ diverges, length is infinite, and the trajectory may wander indefinitely without converging.

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/appendix_a_finite_length.png' | relative_url }}" alt="Two-panel figure: left shows contour plot of the degenerate energy E=(x1²+x2²)²/4 with four gradient-flow trajectories spiraling into the flat critical point at the origin; right shows for one of those trajectories the cumulative arc length growing to about 1.63 and saturating, the remaining length decaying to zero, and the rescaled φ(E)=E^(1-θ) tracking the remaining length" loading="lazy">
  <figcaption>Finite arc-length under Łojasiewicz, illustrated. <strong>Left.</strong> Gradient flow on the degenerate energy $E(x)=\tfrac14(x_1^2+x_2^2)^2$ (Łojasiewicz exponent $\theta=3/4$): four trajectories converge to the flat minimizer at the origin. <strong>Right.</strong> For one trajectory, the cumulative arc length $\int_0^t\|\dot x\|\,ds$ (solid blue) saturates at finite total length $\approx 1.63$; the desingularizing function $\varphi(\mathcal E(t))=\mathcal E^{1-\theta}$ (green dashed, rescaled) decreases from this asymptotic length to $0$ in lockstep with the <em>remaining</em> length (dotted blue). The two are equal up to the constant $C/(1-\theta)$ — the bound $\int_0^\infty\|\dot x\|\le \varphi(\mathcal E(0))$ is tight here.</figcaption>
</figure>

### A.4 The Kurdyka–Łojasiewicz framework

The construction of A.3 admits a much cleaner abstract formulation, due to **Kurdyka (1998)** building on Łojasiewicz:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Kurdyka–Łojasiewicz inequality)</span></p>

$E\in C^1(\mathbb R^N)$ satisfies the **Kurdyka–Łojasiewicz inequality** at $x_\ast$ if there exist a neighborhood $U$ of $x_\ast$, a constant $r>0$, and a **desingularizing function** $\varphi:[0,r)\to[0,\infty)$ — i.e. $\varphi$ is continuous on $[0,r)$, $C^1$ on $(0,r)$, concave, with $\varphi(0)=0$ and $\varphi'(s)>0$ — such that

$$
\varphi'\bigl(E(y)-E(x_\ast)\bigr)\,|\nabla E(y)| \;\ge\; 1 \qquad\text{for all } y\in U \text{ with } 0<E(y)-E(x_\ast)<r. \tag{KL}
$$

</div>

The condition (KL) is exactly the assertion "$\varphi'(\mathcal E)\,\|\nabla E\|\ge 1$" that drove the proof in A.3.

**Łojasiewicz as a special case.** The classical Łojasiewicz inequality $\mathcal E^\theta\le C\|\nabla E\|$ corresponds to $\varphi(s) = \frac{C}{1-\theta}\,s^{1-\theta}$, since then

$$\varphi'(s) = \frac{C}{s^\theta}, \qquad \varphi'(\mathcal E)\,|\nabla E| = \frac{C\,|\nabla E|}{\mathcal E^\theta} \ge 1$$

— precisely (KL).

**Why "desingularizing"?** Geometrically, $\varphi$ takes the *graph* of $E$ near a critical point and "unfolds" the singularity at $E(x_\ast)$ into a smooth curve. After the change of coordinates $u\leftrightarrow\varphi(\mathcal E)$, the apparently degenerate gradient flow becomes uniformly Lipschitz in the new variable, in the sense that $\|du/dt\|\le -\|\dot x\|$ — bounded purely by the speed. The pathological flatness of $E$ near its critical point has been "regularized" by the change of variable.

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/appendix_a_unfolding.png' | relative_url }}" alt="Two-panel figure: left shows energies E(x)=|x|^k for k=2,4,6,10, all with a critical point at the origin but progressively flatter wells as k increases; right shows the same after applying the desingularizing map φ(s)=s^(2/k), which collapses every curve to the same parabola x²" loading="lazy">
  <figcaption>Geometric desingularization. <strong>Left.</strong> Energies $E(x)=\|x\|^k$ have a critical point at the origin that becomes increasingly degenerate as $k$ grows ($k=2$ is non-degenerate, $k=4,6,10$ are progressively flatter). <strong>Right.</strong> After reparametrization by the matching desingularizing function $\varphi(s)=s^{2/k}$ (corresponding to Łojasiewicz exponent $\theta=1-2/k$), every flat well collapses to the same canonical parabola $x^2$. The "singularity" at the critical point — the unbounded slope $\varphi'(s)\to\infty$ as $s\to 0$ — is exactly what is needed to compensate the unbounded flatness of $E$.</figcaption>
</figure>

**Modern significance.** The KL framework, developed in the 2000s by Bolte, Daniilidis, Lewis and others, has become the workhorse of convergence theory for non-convex optimization algorithms (proximal gradient, ADMM, block-coordinate descent) on **tame functions** — i.e., functions definable in an o-minimal structure, which include all real-analytic, semi-algebraic, and globally subanalytic functions encountered in practice. The function $\varphi$ from (KL) directly controls the convergence *rate* of these algorithms: the Łojasiewicz exponent $\theta=\tfrac12$ gives linear (geometric) convergence, $\theta\in(\tfrac12,1)$ gives sublinear $O(t^{-(1-\theta)/(2\theta-1)})$ convergence, and $\theta=0$ (the *finite-time-convergence* regime) gives exact arrival in finite time.

### A.5 Where this idea recurs

The pattern "**find the right monotone reparametrization $\varphi$ of the Lyapunov function so its derivative cleanly bounds what you want to integrate**" appears in many places under different names:

| Setting | Lyapunov | Reparametrization $\varphi$ |
|---|---|---|
| **KL gradient flow** (§1.6) | $\mathcal E$ | $\mathcal E^{1-\theta}$, antiderivative of $\mathcal E^{-\theta}$ |
| **Bihari–LaSalle nonlinear Gronwall** ($\dot u\le -f(u)$) | $u$ | antiderivative of $1/f$, gives implicit decay rate |
| **Carleman / log-Sobolev → exponential entropy decay** | entropy $H$ | $H$ itself; log-Sobolev gives $\dot H\le -\lambda H$ |
| **Polynomial decay in degenerate parabolic PDE** ($\dot u\le -u^{1+\alpha}$) | $u$ | $u^{-\alpha}$, antiderivative of $u^{-1-\alpha}$ — *Aronson–Bénilan estimate* |
| **Forward–backward / proximal methods on tame objectives** | objective $f$ | KL function $\varphi(f-f_\ast)$ |
| **Geometric measure theory: rectifiability via density bounds** | density $\Theta$ | a power $\Theta^\beta$ chosen so its derivative integrates |
| **Kondratiev / weighted Sobolev for elliptic regularity at corners** | distance to corner $r$ | $r^\beta$ with $\beta$ tuned to integrate the singularity |

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The unifying moral)</span></p>

When you have a self-referential bound — "the rate of change of $X$ is controlled by some function of $X$" — the right move is rarely to work with $X$ itself. It is to find the **monotone change of variable $\varphi(X)$** under which the bound becomes affine (or even constant). That change of variable is *forced* by the form of the bound: it is the **antiderivative of the reciprocal** of the bounding function.

In ODEs this is "separation of variables"; in dynamical systems it's "the right Lyapunov function"; in PDE it's the "right test function" or "weighted norm"; in non-smooth optimization it's a "desingularizing function." All are the same idea wearing different hats.

So: the choice of $\mathcal E^{1-\theta}$ in §1.6 is *not* a technicality. It is the canonical example of a recurring template — the **Kurdyka–Łojasiewicz desingularization** — that is worth learning once and recognizing forever.

</div>

