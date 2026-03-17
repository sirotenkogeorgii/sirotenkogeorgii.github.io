---
title: Nonlinear Dynamics and Chaos (Strogatz)
layout: default
noindex: true
---

# Nonlinear Dynamics and Chaos

## Chapter 1: Overview

### Chaos, Fractals, and Dynamics

Chaos and fractals are part of a grander subject known as **dynamics** — the subject that deals with change, with systems that evolve in time. Whether the system settles down to equilibrium, keeps repeating in cycles, or does something more complicated, it is dynamics that we use to analyze the behavior.

The book presents two overviews of the subject: one historical and one logical, and then concludes with a "dynamical view of the world" — a framework that guides the rest of the studies.

### Capsule History of Dynamics

Dynamics is an interdisciplinary subject today, but it was originally a branch of physics. The subject began in the mid-1600s, when Newton invented differential equations, discovered his laws of motion and universal gravitation, and combined them to explain Kepler's laws of planetary motion. Newton solved the **two-body problem** — the problem of calculating the motion of the earth around the sun, given the inverse-square law of gravitational attraction. The **three-body problem** (e.g., sun, earth, and moon) turned out to be essentially *impossible* to solve in the sense of obtaining explicit formulas.

The breakthrough came with **Poincare** in the late 1800s. He introduced a new point of view that emphasized *qualitative* rather than quantitative questions (e.g., "Is the solar system stable forever?"). Poincare developed a powerful **geometric** approach and was also the first person to glimpse the possibility of **chaos** — a deterministic system exhibiting aperiodic behavior that depends sensitively on initial conditions, rendering long-term prediction impossible.

Key milestones:

| Period | Figure | Contribution |
| --- | --- | --- |
| 1666 | Newton | Invention of calculus, planetary motion |
| 1890s | Poincare | Geometric approach, nightmares of chaos |
| 1920–1960 | Birkhoff, Kolmogorov, Arnol'd, Moser | Complex behavior in Hamiltonian mechanics |
| 1963 | Lorenz | Strange attractor in simple model of convection |
| 1970s | Ruelle & Takens, May, Feigenbaum | Turbulence and chaos, logistic map, universality |
| 1970s | Winfree, Mandelbrot | Nonlinear oscillators in biology, fractals |
| 1980s | — | Widespread interest in chaos, fractals, and applications |

**Lorenz's discovery (1963):** He studied a simplified model of convection rolls in the atmosphere. The solutions never settled to equilibrium or a periodic state — instead they oscillated in an irregular, aperiodic fashion. Two slightly different initial conditions would lead to totally different behaviors. The system was *inherently* unpredictable. When plotted in three dimensions, the solutions fell onto a butterfly-shaped set of points — a **strange attractor** — which he argued was "an infinite complex of surfaces" (today recognized as an example of a fractal).

### The Importance of Being Nonlinear

There are two main types of dynamical systems: **differential equations** and **iterated maps** (also known as difference equations). Differential equations describe evolution in continuous time, whereas iterated maps arise in problems where time is discrete.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(General System of ODEs)</span></p>

A very general framework for ordinary differential equations is provided by the system

$$
\dot{x}_1 = f_1(x_1, \dots, x_n), \quad \dots, \quad \dot{x}_n = f_n(x_1, \dots, x_n),
$$

where the overdots denote differentiation with respect to $t$, i.e. $\dot{x}_i \equiv dx_i/dt$. The variables $x_1, \dots, x_n$ might represent concentrations of chemicals, populations of species, or positions and velocities of planets. The functions $f_1, \dots, f_n$ are determined by the problem at hand.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Damped Harmonic Oscillator)</span></p>

The equation $m\ddot{x} + b\dot{x} + kx = 0$ can be rewritten in the general form by introducing $x_1 = x$ and $x_2 = \dot{x}$:

$$
\dot{x}_1 = x_2, \qquad \dot{x}_2 = -\frac{b}{m} x_2 - \frac{k}{m} x_1.
$$

This system is **linear** because all the $x_i$ on the right-hand side appear to the first power only.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Pendulum — Nonlinear System)</span></p>

The pendulum equation $\ddot{x} + \frac{g}{L}\sin x = 0$ becomes the nonlinear system

$$
\dot{x}_1 = x_2, \qquad \dot{x}_2 = -\frac{g}{L}\sin x_1.
$$

The $\sin x_1$ term makes this nonlinear. The usual small-angle approximation $\sin x \approx x$ linearizes the problem, but discards physics like the pendulum whirling over the top.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Trajectory and Phase Space)</span></p>

Given a solution $(x_1(t), \dots, x_n(t))$ to the system, the corresponding curve in the abstract space with coordinates $(x_1, \dots, x_n)$ is called a **trajectory**. The space itself is called the **phase space** for the system. The phase space is completely filled with trajectories, since each point can serve as an initial condition.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric Approach)</span></p>

The key idea is to draw the trajectories in phase space *without actually solving the system*. In many cases, geometric reasoning provides qualitative information about the solutions directly.

</div>

**Nonautonomous Systems.** The system above does not include explicit *time dependence*. Time-dependent or **nonautonomous** equations like $m\ddot{x} + b\dot{x} + kx = F\cos t$ can be handled by introducing $x_3 = t$, $\dot{x}_3 = 1$, converting the system into an equivalent autonomous system of one higher dimension. By this trick, we can always remove explicit time dependence by adding an extra dimension to the system.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Are Nonlinear Problems So Hard?)</span></p>

The essential difference is that *linear systems can be broken down into parts*. Each part can be solved separately and finally recombined to get the answer (superposition principle). This idea underlies normal modes, Laplace transforms, superposition arguments, and Fourier analysis. A linear system is precisely equal to the sum of its parts.

Nonlinear systems do not decompose in this way. Whenever parts of a system interfere, cooperate, or compete, there are nonlinear interactions going on, and the principle of superposition fails.

</div>

### A Dynamical View of the World

The framework for dynamics and its applications has two axes:

1. **Number of variables** (equivalently, the dimension of the phase space): $n = 1$, $n = 2$, $n \geq 3$, or continuum.
2. **Linearity vs. nonlinearity** of the system.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Classifying Systems)</span></p>

* **Exponential growth** $\dot{x} = rx$: first-order ($n = 1$), linear.
* **Pendulum** $\ddot{x} + \frac{g}{L}\sin x = 0$: second-order ($n = 2$), nonlinear.
* **Lorenz system**: third-order ($n = 3$), nonlinear — exhibits chaos.

</div>

**The linear, small-$n$ corner** (upper left of the framework) contains the simplest systems studied in early courses: growth/decay/equilibrium when $n = 1$, oscillations when $n = 2$. The **upper right corner** contains classical applied mathematics and mathematical physics — linear PDEs like Maxwell's equations, the heat equation, and Schrodinger's equation, which involve an infinite "continuum" of variables.

**The nonlinear half** (lower part of the framework) is the focus of the book. Starting at $n = 1$ and heading right, we encounter new phenomena at every step: **fixed points** and **bifurcations** when $n = 1$, **limit cycles** and **nonlinear oscillations** when $n = 2$, and finally **chaos** and **fractals** when $n \geq 3$. In all cases, a geometric approach proves to be very powerful, providing most of the information we want even though we usually cannot solve the equations in closed form.

The framework also contains a region marked "The frontier" — problems that are both large and nonlinear (spatiotemporal complexity, turbulence, fibrillation, etc.). These lie at the limits of current understanding.

## Chapter 2: Flows on the Line

### Introduction

We now begin with the simplest case $n = 1$. A single equation of the form

$$\dot{x} = f(x)$$

is called a **one-dimensional** or **first-order system**. Here $x(t)$ is a real-valued function of time $t$, and $f(x)$ is a smooth real-valued function of $x$. We do not allow $f$ to depend explicitly on time — nonautonomous equations $\dot{x} = f(x, t)$ require *two* pieces of information ($x$ and $t$) and are really two-dimensional systems.

### A Geometric Way of Thinking

Pictures are often more helpful than formulas for analyzing nonlinear systems. The key technique is to *interpret a differential equation as a vector field*.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\dot{x} = \sin x$ — Geometric Analysis)</span></p>

Consider $\dot{x} = \sin x$. Although this can be solved in closed form (yielding $t = \ln\left\lvert\frac{\csc x_0 + \cot x_0}{\csc x + \cot x}\right\rvert$), the formula is hard to interpret.

Instead, we think of $t$ as time, $x$ as the position of an imaginary particle on the real line, and $\dot{x}$ as its velocity. The equation $\dot{x} = \sin x$ represents a **vector field** on the line: it dictates the velocity $\dot{x}$ at each $x$. We plot $\dot{x}$ versus $x$ and draw arrows on the $x$-axis — to the right when $\dot{x} > 0$, to the left when $\dot{x} < 0$.

The **flow** is to the right when $\dot{x} > 0$ and to the left when $\dot{x} < 0$. At points where $\dot{x} = 0$, there is no flow; such points are **fixed points**. There are two kinds:
* Solid black dots represent **stable** fixed points (attractors or sinks) — the flow is toward them.
* Open circles represent **unstable** fixed points (repellers or sources) — the flow is away from them.

For $\dot{x} = \sin x$: the fixed points are at $x^* = k\pi$. The even multiples of $\pi$ are unstable, and the odd multiples are stable. A particle starting at $x_0 = \pi/4$ moves right, accelerates until $x = \pi/2$ (where $\sin x$ is maximum), then decelerates and asymptotically approaches $x = \pi$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Qualitative vs. Quantitative)</span></p>

The graphical approach cannot tell us certain *quantitative* things — for instance, the precise time at which the speed $\lvert\dot{x}\rvert$ is greatest. But in many cases *qualitative* information is what we care about, and pictures suffice.

</div>

### Fixed Points and Stability

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fixed Point, Phase Portrait)</span></p>

For the system $\dot{x} = f(x)$, we imagine a fluid flowing along the real line with local velocity $f(x)$. This imaginary fluid is the **phase fluid**, and the real line is the **phase space**. A point placed at $x_0$ and carried along by the flow is called a **phase point**; the resulting function $x(t)$ is the **trajectory** based at $x_0$.

A picture showing all the qualitatively different trajectories of the system is called a **phase portrait**.

The phase portrait is controlled by the **fixed points** $x^*$, defined by $f(x^*) = 0$. Fixed points represent **equilibrium** solutions (also called steady, constant, or rest solutions), since if $x = x^*$ initially, then $x(t) = x^*$ for all time.

An equilibrium is **stable** if all sufficiently small disturbances away from it damp out in time. Otherwise it is **unstable**. Stable equilibria are represented geometrically by stable fixed points (where the local flow is toward the point), and unstable equilibria by unstable fixed points (where the local flow is away).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\dot{x} = x^2 - 1$)</span></p>

Find all fixed points and classify their stability.

Here $f(x) = x^2 - 1$. Setting $f(x^*) = 0$ gives $x^* = \pm 1$. To determine stability, we plot $x^2 - 1$ and sketch the vector field. The flow is to the right where $x^2 - 1 > 0$ and to the left where $x^2 - 1 < 0$. Thus $x^* = -1$ is **stable** and $x^* = 1$ is **unstable**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Local vs. Global Stability)</span></p>

The definition of stable equilibrium is based on *small* disturbances; certain large disturbances may fail to decay. For instance, in the example $\dot{x} = x^2 - 1$, all small disturbances to $x^* = -1$ will decay, but a large disturbance sending $x$ to the right of $x = 1$ will cause the phase point to be repelled out to $+\infty$. We say $x^* = -1$ is **locally stable** but not globally stable. A fixed point that is approached from *all* initial conditions is **globally stable**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(RC Circuit)</span></p>

A resistor $R$ and capacitor $C$ in series with a battery of constant voltage $V_0$. Let $Q(t)$ be the charge on the capacitor for $t \geq 0$ with $Q(0) = 0$. By Kirchhoff's voltage law:

$$\dot{Q} = f(Q) = \frac{V_0}{R} - \frac{Q}{RC}.$$

The graph of $f(Q)$ is a straight line with negative slope. The unique fixed point $Q^* = CV_0$ is **globally stable** — the flow is always toward $Q^*$. The solution $Q(t)$ increases monotonically and is concave down as it approaches $Q^*$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\dot{x} = x - \cos x$)</span></p>

To find fixed points, we need to solve $x = \cos x$. Rather than plotting $f(x) = x - \cos x$, we can graph $y = x$ and $y = \cos x$ separately and find where they intersect. They intersect at exactly one point $x^*$. Since the line lies above the cosine curve for $x > x^*$, we have $\dot{x} > 0$ there (flow to the right), and similarly $\dot{x} < 0$ for $x < x^*$ (flow to the left). Hence $x^*$ is the only fixed point, and it is **unstable** — even though we don't have a formula for $x^*$ itself.

</div>

### Population Growth

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Logistic Equation)</span></p>

The simplest model for population growth is $\dot{N} = rN$, which predicts exponential growth $N(t) = N_0 e^{rt}$. To model the effects of overcrowding and limited resources, we assume the per capita growth rate $\dot{N}/N$ decreases linearly with $N$. This leads to the **logistic equation**

$$\dot{N} = rN\left(1 - \frac{N}{K}\right),$$

where $r > 0$ is the intrinsic growth rate and $K > 0$ is the **carrying capacity**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Analysis of the Logistic Equation)</span></p>

Fixed points occur at $N^* = 0$ and $N^* = K$. Plotting $\dot{N}$ versus $N$ (for $N \geq 0$) gives a downward-opening parabola with maximum at $N = K/2$.

* $N^* = 0$ is **unstable**: a small positive population grows exponentially fast and runs away from $N = 0$.
* $N^* = K$ is **stable**: if $N$ is disturbed slightly from $K$, the disturbance decays monotonically and $N(t) \to K$ as $t \to \infty$.

For any $N_0 > 0$, the population always approaches the carrying capacity $K$. The solution $N(t)$ is S-shaped (**sigmoid**) for $N_0 < K/2$: the population initially accelerates (concave up) until $N = K/2$, then decelerates (concave down) as it asymptotes to $K$.

</div>

### Linear Stability Analysis

So far we have relied on graphical methods. Now we develop a more quantitative approach by **linearizing** about a fixed point.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Linear Stability Analysis)</span></p>

Let $x^*$ be a fixed point of $\dot{x} = f(x)$, and let $\eta(t) = x(t) - x^*$ be a small perturbation. Then by Taylor expansion:

$$\dot{\eta} = f(x^* + \eta) = \underbrace{f(x^*)}_{= 0} + \eta f'(x^*) + O(\eta^2).$$

If $f'(x^*) \neq 0$, the $O(\eta^2)$ terms are negligible for small $\eta$, yielding the **linearization about $x^*$**:

$$\dot{\eta} \approx \eta f'(x^*).$$

Hence the perturbation grows exponentially if $f'(x^*) > 0$ (unstable) and decays exponentially if $f'(x^*) < 0$ (stable). The quantity $\lvert f'(x^*)\rvert$ is the rate of exponential growth or decay, and its reciprocal $\tau = 1/\lvert f'(x^*)\rvert$ is the **characteristic time scale** — the time required for $x(t)$ to vary significantly near $x^*$.

If $f'(x^*) = 0$, the linearization is inconclusive and a nonlinear analysis (graphical methods) is needed.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Linear Stability of $\dot{x} = \sin x$)</span></p>

The fixed points are $x^* = k\pi$. Then $f'(x^*) = \cos(k\pi) = \begin{cases} 1, & k \text{ even} \\ -1, & k \text{ odd}.\end{cases}$

Hence $x^*$ is **unstable** if $k$ is even and **stable** if $k$ is odd, agreeing with the graphical analysis.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Linear Stability of the Logistic Equation)</span></p>

Here $f(N) = rN(1 - N/K)$, so $f'(N) = r - 2rN/K$. At $N^* = 0$: $f'(0) = r > 0$ (unstable). At $N^* = K$: $f'(K) = -r < 0$ (stable). The characteristic time scale in either case is $\tau = 1/r$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(When $f'(x^*) = 0$ — Marginal Cases)</span></p>

When $f'(x^*) = 0$, the linearization fails and the stability must be determined case by case using graphical methods. Consider:

* (a) $\dot{x} = -x^3$: $x^* = 0$ is **stable** (flow is toward origin from both sides).
* (b) $\dot{x} = x^3$: $x^* = 0$ is **unstable** (flow is away from origin).
* (c) $\dot{x} = x^2$: $x^* = 0$ is **half-stable** — attracting from the left, repelling from the right.
* (d) $\dot{x} = 0$: every point is a fixed point; perturbations neither grow nor decay.

These cases arise naturally in the context of **bifurcations**.

</div>

### Existence and Uniqueness

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Existence and Uniqueness)</span></p>

Consider the initial value problem $\dot{x} = f(x)$, $x(0) = x_0$. Suppose that $f(x)$ and $f'(x)$ are continuous on an open interval $R$ of the $x$-axis, and $x_0 \in R$. Then the initial value problem has a solution $x(t)$ on some time interval $(-\tau, \tau)$ about $t = 0$, and the solution is **unique**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Smoothness is Essential)</span></p>

If $f(x)$ is not smooth enough, uniqueness can fail. For example, $\dot{x} = x^{1/3}$ with $x(0) = 0$ has both the obvious solution $x(t) = 0$ and the non-obvious solution $x(t) = \left(\frac{2}{3}t\right)^{3/2}$. In fact, there are *infinitely* many solutions from this initial condition. The problem is that $f'(0)$ is infinite at $x^* = 0$, violating the smoothness hypothesis of the theorem.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Blow-up in Finite Time)</span></p>

The theorem guarantees that solutions exist and are unique, but it does *not* say that solutions exist for all time. Consider $\dot{x} = 1 + x^2$, $x(0) = 0$. By separation of variables, $x(t) = \tan t$, which exists only for $-\pi/2 < t < \pi/2$. The solution reaches infinity in finite time — a phenomenon called **blow-up**. This is of physical relevance in models of combustion and other runaway processes.

</div>

### Impossibility of Oscillations

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(No Periodic Solutions on the Line)</span></p>

There are **no periodic solutions** to $\dot{x} = f(x)$. Trajectories on the real line are forced to increase or decrease monotonically, or remain constant. The phase point never reverses direction.

This is fundamentally topological: if you flow monotonically on a *line*, you'll never come back to your starting place. (On a *circle*, however, you could — this is why periodic solutions are possible for vector fields on the circle, discussed in Chapter 4.)

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Mechanical Analog: Overdamped Systems)</span></p>

The impossibility of oscillations makes physical sense if we view $\dot{x} = f(x)$ as a limiting case of Newton's law $m\ddot{x} + b\dot{x} = F(x)$ in the **overdamped** limit ($b\dot{x} \gg m\ddot{x}$). Then $b\dot{x} \approx F(x)$, i.e. $\dot{x} = f(x)$ with $f(x) = b^{-1}F(x)$. The mass is dragged slowly to a stable equilibrium by the restoring force, with no overshoot because the damping is enormous.

</div>

### Potentials

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Potential)</span></p>

For the first-order system $\dot{x} = f(x)$, the **potential** $V(x)$ is defined by

$$f(x) = -\frac{dV}{dx}.$$

We picture a particle sliding down the walls of a potential well. The particle is heavily damped — its inertia is negligible compared to the damping force and the force due to the potential.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Potential)</span></p>

* $V(t)$ **decreases along trajectories**: since $\frac{dV}{dt} = \frac{dV}{dx}\frac{dx}{dt} = -\left(\frac{dV}{dx}\right)^2 \leq 0$, the particle always moves toward lower potential.
* **Equilibria** occur at the fixed points of the vector field, i.e. where $dV/dx = 0$ (since $\dot{x} = 0$ iff $dV/dx = 0$).
* Local **minima** of $V(x)$ correspond to **stable** fixed points.
* Local **maxima** of $V(x)$ correspond to **unstable** fixed points.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Potential for $\dot{x} = -x$)</span></p>

Solving $-dV/dx = -x$ gives $V(x) = \frac{1}{2}x^2 + C$. Taking $C = 0$, the potential is a parabola with a single minimum at $x = 0$. Hence $x = 0$ is the only equilibrium point, and it is stable.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Double-Well Potential: $\dot{x} = x - x^3$)</span></p>

Solving $-dV/dx = x - x^3$ yields $V(x) = -\frac{1}{2}x^2 + \frac{1}{4}x^4$. The local minima at $x = \pm 1$ correspond to stable equilibria, and the local maximum at $x = 0$ corresponds to an unstable equilibrium. This system is called **bistable** because it has two stable equilibria. The potential is a **double-well potential**.

</div>

### Solving Equations on the Computer

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Euler's Method)</span></p>

Given $\dot{x} = f(x)$, $x(t_0) = x_0$, and a step size $\Delta t$, the **Euler method** computes successive approximations:

$$x_{n+1} = x_n + f(x_n)\,\Delta t.$$

This is a first-order method: the error $E \propto \Delta t$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Improved Euler Method)</span></p>

The **improved Euler method** (also called Heun's method) averages the derivative at both ends of the interval:

$$\tilde{x}_{n+1} = x_n + f(x_n)\,\Delta t \quad \text{(trial step)}$$

$$x_{n+1} = x_n + \tfrac{1}{2}\bigl[f(x_n) + f(\tilde{x}_{n+1})\bigr]\,\Delta t \quad \text{(real step)}.$$

This is a second-order method: $E \propto (\Delta t)^2$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Fourth-Order Runge–Kutta Method)</span></p>

The **fourth-order Runge–Kutta method** computes four intermediate quantities:

$$k_1 = f(x_n)\,\Delta t, \quad k_2 = f(x_n + \tfrac{1}{2}k_1)\,\Delta t, \quad k_3 = f(x_n + \tfrac{1}{2}k_2)\,\Delta t, \quad k_4 = f(x_n + k_3)\,\Delta t.$$

Then:

$$x_{n+1} = x_n + \tfrac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4).$$

This method gives accurate results without requiring an excessively small step size. It is the standard workhorse of numerical integration.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Round-off Error)</span></p>

One cannot simply pick a tiny $\Delta t$ to get arbitrary accuracy. Computers have finite precision — they cannot distinguish numbers that differ by less than $\delta \approx 10^{-7}$ (single precision) or $\delta \approx 10^{-16}$ (double precision). **Round-off error** accumulates at every calculation step, and becomes a serious problem if $\Delta t$ is too small. In practice, adaptive step-size control (e.g., Runge–Kutta with automatic step-size adjustment) is recommended.

</div>

## Chapter 3: Bifurcations

### Introduction

As we saw in Chapter 2, the dynamics of vector fields on the line is very limited: all solutions either settle down to equilibrium or head out to $\pm\infty$. What makes one-dimensional systems interesting is their *dependence on parameters*. The qualitative structure of the flow can change as parameters are varied. In particular, fixed points can be created or destroyed, or their stability can change. These qualitative changes in the dynamics are called **bifurcations**, and the parameter values at which they occur are called **bifurcation points**.

Bifurcations provide models of transitions and instabilities as some *control parameter* is varied (e.g., the buckling of a beam as the load increases).

### Saddle-Node Bifurcation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Saddle-Node Bifurcation)</span></p>

The **saddle-node bifurcation** is the basic mechanism by which fixed points are *created and destroyed*. As a parameter is varied, two fixed points move toward each other, collide, and mutually annihilate.

The **normal form** (prototypical example) is

$$\dot{x} = r + x^2.$$

* For $r < 0$: two fixed points, $x^* = \pm\sqrt{-r}$, one stable and one unstable.
* For $r = 0$: the two fixed points coalesce into a single half-stable fixed point at $x^* = 0$.
* For $r > 0$: no fixed points at all.

The bifurcation occurs at $r = 0$. The **bifurcation diagram** plots $x^*$ vs. $r$, with solid lines for stable fixed points and dashed lines for unstable ones.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Alternative Normal Form and Terminology)</span></p>

The alternative form $\dot{x} = r - x^2$ gives the "reverse" picture: no fixed points for $r < 0$, and a pair appearing "out of the clear blue sky" for $r > 0$ — this is called a **blue sky bifurcation**. The word "bifurcation" itself means "splitting into two branches."

Other names for the saddle-node bifurcation include **fold bifurcation** and **turning-point bifurcation**. The name "saddle-node" comes from the higher-dimensional analogue where saddle points and nodes collide and annihilate (see Section 8.1).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Linear Stability at a Saddle-Node)</span></p>

For $\dot{x} = r - x^2$, the fixed points for $r > 0$ are $x^* = \pm\sqrt{r}$. Then $f'(x) = -2x$, so:
* $f'(+\sqrt{r}) = -2\sqrt{r} < 0$ — **stable**.
* $f'(-\sqrt{r}) = +2\sqrt{r} > 0$ — **unstable**.

At the bifurcation point $r = 0$, the fixed points coalesce and $f'(x^*) = 0$; the linearization vanishes.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\dot{x} = r - x - e^{-x}$)</span></p>

We cannot find fixed points explicitly, so we use a geometric approach: plot $r - x$ and $e^{-x}$ on the same axes and look for intersections.

* For large $r$, the line $r - x$ lies above the curve $e^{-x}$, giving two intersection points (one stable on the right, one unstable on the left).
* As $r$ decreases, the line slides down. At a critical value $r = r_c$, the line becomes *tangent* to the curve — a saddle-node bifurcation.
* For $r < r_c$, the line lies entirely below the curve — no fixed points.

To find $r_c$: we require both $r - x = e^{-x}$ and $-1 = -e^{-x}$ (tangency of derivatives). The second equation gives $x = 0$, and the first gives $r_c = 1$. The bifurcation occurs at $r = 1$, $x = 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Saddle-Nodes Are Generic)</span></p>

Near a saddle-node bifurcation at $(x^*, r_c)$, Taylor expansion gives

$$\dot{x} = f(x, r) \approx a(r - r_c) + b(x - x^*)^2 + \cdots$$

where $a = \partial f/\partial r\rvert_{(x^*, r_c)}$ and $b = \frac{1}{2}\partial^2 f/\partial x^2\rvert_{(x^*, r_c)}$, since $f(x^*, r_c) = 0$ and $\partial f/\partial x\rvert_{(x^*, r_c)} = 0$ (tangency condition). When $a, b \neq 0$, this has the same algebraic form as the normal form $\dot{x} = r + x^2$ after rescaling. This is why the saddle-node is the *generic* one-parameter bifurcation of fixed points.

</div>

### Transcritical Bifurcation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Transcritical Bifurcation)</span></p>

In some problems, a fixed point must exist for all values of a parameter and can never be destroyed. For example, in population models there is always a fixed point at zero population. However, such a fixed point may *change its stability* as the parameter is varied. The **transcritical bifurcation** is the standard mechanism for this.

The **normal form** is

$$\dot{x} = rx - x^2.$$

* For $r < 0$: there is an unstable fixed point at $x^* = r$ and a stable fixed point at $x^* = 0$.
* As $r$ increases, the unstable fixed point approaches the origin and coalesces with it at $r = 0$.
* For $r > 0$: the origin has become unstable, and $x^* = r$ is now stable.

An **exchange of stabilities** has taken place between the two fixed points. Crucially, unlike the saddle-node, the two fixed points don't disappear after the bifurcation — they just switch stability.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\dot{x} = r\ln x + x - 1$ near $x = 1$)</span></p>

We analyze the dynamics near $x = 1$. Note that $x = 1$ is a fixed point for all $r$. Let $u = x - 1$ (small). Taylor expanding:

$$\dot{u} = r\bigl[u - \tfrac{1}{2}u^2 + O(u^3)\bigr] + u \approx (r+1)u - \tfrac{1}{2}ru^2 + O(u^3).$$

A transcritical bifurcation occurs at $r_c = -1$ (where the linear coefficient vanishes). To put into normal form, let $u = av$, choose $a = 2/r$ to eliminate the coefficient of $v^2$, let $R = r + 1$ and $X = v$. Then $\dot{X} \approx RX - X^2$ near the bifurcation.

</div>

### Laser Threshold

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Laser Model — Transcritical Bifurcation)</span></p>

A simplified model for a solid-state laser (Haken 1983): let $n(t)$ be the number of photons in the laser field. Then

$$\dot{n} = \text{gain} - \text{loss} = GnN - kn,$$

where $G > 0$ is the gain coefficient, $N(t)$ is the number of excited atoms, and $k > 0$ is the photon loss rate. The excited atom population is depleted by the laser process: $N(t) = N_0 - \alpha n$, where $N_0$ is the pump strength and $\alpha > 0$. Substituting:

$$\dot{n} = (GN_0 - k)\,n - (\alpha G)\,n^2.$$

This is a first-order system of the transcritical form $\dot{x} = rx - x^2$.

* For $N_0 < k/G$: the fixed point $n^* = 0$ is stable (the laser acts as a **lamp**).
* For $N_0 > k/G$: the origin loses stability and a new stable fixed point $n^* = (GN_0 - k)/\alpha G > 0$ appears (spontaneous **laser** action).

The **laser threshold** is $N_0 = k/G$, where a transcritical bifurcation occurs.

</div>

### Pitchfork Bifurcation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Pitchfork Bifurcation)</span></p>

The **pitchfork bifurcation** is common in problems that have a **symmetry** between left and right (i.e., the equation is invariant under $x \to -x$). Fixed points tend to appear and disappear in symmetric pairs. There are two types: **supercritical** and **subcritical**.

</div>

#### Supercritical Pitchfork Bifurcation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Supercritical Pitchfork)</span></p>

The normal form is

$$\dot{x} = rx - x^3.$$

Note the invariance under $x \to -x$ (the equation is unchanged). The cubic term is *stabilizing* — it acts as a restoring force pulling $x(t)$ back toward $x = 0$.

* For $r < 0$: the origin is the only fixed point and it is stable.
* For $r = 0$: the origin is still stable, but much more weakly so — decay is no longer exponential but algebraic (**critical slowing down**).
* For $r > 0$: the origin has become unstable. Two new stable fixed points appear symmetrically at $x^* = \pm\sqrt{r}$.

The bifurcation diagram has the shape of a pitchfork.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\dot{x} = -x + \beta\tanh x$ — Magnets and Neural Networks)</span></p>

This equation arises in statistical mechanics and neural networks. Fixed points satisfy $x = \beta\tanh x$. We graph $y = x$ and $y = \beta\tanh x$:
* For $\beta < 1$: the origin is the only intersection (only fixed point), and it is stable.
* At $\beta = 1$: the $\tanh$ curve develops a slope of 1 at the origin — pitchfork bifurcation.
* For $\beta > 1$: two new stable fixed points appear symmetrically, and the origin becomes unstable.

To get a numerically accurate bifurcation diagram, we compute $\beta = x^*/\tanh x^*$ for each $x^*$ (exploiting the fact that $f$ depends more simply on $\beta$ than on $x$).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Potential for Supercritical Pitchfork)</span></p>

For $\dot{x} = rx - x^3$, the potential is $V(x) = -\frac{1}{2}rx^2 + \frac{1}{4}x^4$.

* $r < 0$: a single quadratic minimum at the origin.
* $r = 0$: a much flatter quartic minimum at the origin.
* $r > 0$: a local *maximum* appears at the origin, and a symmetric pair of minima form at $x = \pm\sqrt{r}$.

</div>

#### Subcritical Pitchfork Bifurcation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Subcritical Pitchfork)</span></p>

If the cubic term is *destabilizing* instead of stabilizing, we get a **subcritical** pitchfork bifurcation. The normal form is

$$\dot{x} = rx + x^3.$$

* For $r < 0$: the origin is stable. Two *unstable* fixed points exist at $x^* = \pm\sqrt{-r}$.
* For $r > 0$: the origin is unstable, and there are no other fixed points. Trajectories are driven out to $\pm\infty$ — the cubic term lends a "helping hand" to the instability, leading to **blow-up** in finite time.

The bifurcation diagram is an *inverted* pitchfork compared to the supercritical case. The subcritical bifurcation is sometimes called an **inverted** or **backward** bifurcation, and is related to discontinuous or first-order phase transitions in physics.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stabilization by Higher-Order Terms)</span></p>

In real physical systems, the explosive instability of $\dot{x} = rx + x^3$ is usually opposed by stabilizing higher-order terms. The canonical example with fifth-order stabilization is

$$\dot{x} = rx + x^3 - x^5.$$

The bifurcation diagram now has a richer structure:
* For $r < r_s < 0$: the origin is the only stable state.
* For $r_s < r < 0$: two qualitatively different stable states coexist — the origin and two large-amplitude branches. The initial condition $x_0$ determines which is approached. The origin is **locally stable** but not globally stable.
* At $r = 0$: the origin loses stability. The slightest nudge causes the state to **jump** to one of the large-amplitude branches.
* Decreasing $r$ back past $0$ does *not* return the state to the origin — one must go all the way past $r_s$ for that. This lack of reversibility is called **hysteresis**.
* The bifurcation at $r = r_s$ is a saddle-node bifurcation where the large-amplitude stable and unstable branches are born.

</div>

### Overdamped Bead on a Rotating Hoop

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Bead on a Rotating Hoop)</span></p>

A bead of mass $m$ slides along a wire hoop of radius $r$, rotating at constant angular velocity $\omega$ about its vertical axis. The bead is subject to gravity, centrifugal force, and viscous friction ($b\dot{\phi}$). Let $\phi$ be the angle from the downward vertical. Newton's law gives:

$$mr\ddot{\phi} = -b\dot{\phi} - mg\sin\phi + mr\omega^2\sin\phi\cos\phi.$$

In the **overdamped limit** ($b\dot{\phi} \gg mr\ddot{\phi}$), the inertial term $mr\ddot{\phi}$ is negligible and the equation reduces to a first-order system. Nondimensionalizing with $\tau = t/T$ where $T = b/(mg)$:

$$\frac{d\phi}{d\tau} = -\sin\phi + \gamma\sin\phi\cos\phi = f(\phi),$$

where $\gamma = r\omega^2/g$ is a dimensionless parameter measuring the ratio of centrifugal to gravitational forces.

**Analysis:**
* $\phi^* = 0$ (bottom of the hoop) is always a fixed point. Its stability: $f'(0) = -1 + \gamma$.
  * Stable for $\gamma < 1$, unstable for $\gamma > 1$.
* For $\gamma > 1$, two new fixed points appear at $\phi^* = \pm\cos^{-1}(1/\gamma)$ (symmetrically placed above the equator). These are stable.

This is a **supercritical pitchfork bifurcation** at $\gamma = 1$ (i.e., $\omega^2 = g/r$). The physical interpretation: for slow rotation the bead sits at the bottom; for fast rotation the bottom becomes unstable and the bead moves to one side.

The two new fixed points are **symmetry-broken** solutions: even though the governing equation has perfect left-right symmetry ($\phi \to -\phi$), the bead must choose one side.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dimensional Analysis and the Overdamped Limit)</span></p>

The original second-order equation $mr\ddot{\phi} = -b\dot{\phi} - mg\sin\phi + mr\omega^2\sin\phi\cos\phi$ has five parameters ($m, g, r, \omega, b$). Introducing the dimensionless time $\tau = t/T$ and dividing by $mg$ yields

$$\varepsilon\frac{d^2\phi}{d\tau^2} = -\frac{d\phi}{d\tau} - \sin\phi + \gamma\sin\phi\cos\phi,$$

where $\gamma = r\omega^2/g$ and $\varepsilon = m^2 gr/b^2$. The five parameters have been reduced to two dimensionless groups. The **overdamped limit** corresponds to $\varepsilon \to 0$, valid when $b^2 \gg m^2 gr$ (strong damping or small mass).

Setting $\varepsilon = 0$ gives the first-order approximation $d\phi/d\tau = f(\phi)$ analyzed above.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Resolution of the Paradox — Singular Limit and Phase Plane)</span></p>

Replacing a second-order equation (which requires *two* initial conditions: $\phi_0$ and $\dot{\phi}_0$) by a first-order equation (which requires only one) seems paradoxical. The resolution involves **phase plane** analysis.

Rewrite the second-order system as a vector field on the $(\phi, \Omega)$ plane, where $\Omega = d\phi/d\tau$:

$$\phi' = \Omega, \qquad \Omega' = \frac{1}{\varepsilon}\bigl(f(\phi) - \Omega\bigr).$$

In the limit $\varepsilon \to 0$, $\Omega'$ becomes enormous whenever $\Omega \neq f(\phi)$. Hence all trajectories slam vertically onto the curve $C$ defined by $\Omega = f(\phi)$ in a rapid initial **transient**, and then slowly ooze along $C$ until reaching a fixed point. After this transient, the motion is well approximated by the first-order equation $\phi' = f(\phi)$.

This is an example of a **singular limit** — the highest-order derivative drops out of the equation. The rapid transient plays the role of a thin "boundary layer" in time near $t = 0$. The branch of mathematics that deals with such limits is called **singular perturbation theory**.

</div>

### Imperfect Bifurcations and Catastrophes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Imperfect Bifurcation)</span></p>

Pitchfork bifurcations require a symmetry ($x \to -x$). In real-world problems, this symmetry is only approximate — small **imperfections** break it. To study the effect, we add an **imperfection parameter** $h$ to the normal form of the supercritical pitchfork:

$$\dot{x} = h + rx - x^3.$$

When $h = 0$, we recover the standard pitchfork. When $h \neq 0$, the symmetry is broken.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Analysis of the Imperfect Bifurcation)</span></p>

Fixed points satisfy $h + rx - x^3 = 0$. We analyze graphically by plotting $y = rx - x^3$ and $y = -h$ and looking for intersections.

* For $r \leq 0$: the cubic is monotonically decreasing, so there is exactly one intersection for any $h$ — **one fixed point** always.
* For $r > 0$: the cubic has a local max and min. There can be one, two, or three intersections depending on $\lvert h\rvert$.

The critical values of $h$ at which saddle-node bifurcations occur are found by requiring tangency. The local extrema of the cubic are at $x_{\max} = \sqrt{r/3}$ with value $\frac{2r}{3}\sqrt{r/3}$. Hence the **bifurcation curves** are

$$h_c(r) = \pm\frac{2r}{3}\sqrt{\frac{r}{3}}, \qquad r > 0.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stability Diagram and Cusp Point)</span></p>

Plotting $h = \pm h_c(r)$ in the $(r, h)$ parameter plane gives a **stability diagram**. The two bifurcation curves meet tangentially at $(r, h) = (0, 0)$, called the **cusp point**. Inside the cusp-shaped region (for $r > 0$, $\lvert h\rvert < h_c(r)$) there are three fixed points; outside there is one. Saddle-node bifurcations occur all along the boundary of these regions. The cusp point itself is a **codimension-2 bifurcation** — it requires tuning *two* parameters ($r$ and $h$) simultaneously.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Effect of Imperfection on the Bifurcation Diagram)</span></p>

* For $h = 0$: the standard pitchfork diagram.
* For $h \neq 0$: the pitchfork **disconnects** into two separate pieces. The upper piece consists entirely of stable fixed points; the lower piece has both stable and unstable branches. As $r$ increases from negative values, there is no longer a sharp transition — the fixed point simply glides smoothly along the upper branch. The lower branch of stable fixed points is not accessible unless a fairly large disturbance is made.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cusp Catastrophe)</span></p>

If we plot the fixed points $x^*$ above the $(r, h)$ parameter plane, we obtain the **cusp catastrophe** surface. This surface folds over itself in certain places; the projection of the folds onto the $(r, h)$ plane yields the bifurcation curves $h = \pm h_c(r)$.

The term **catastrophe** is motivated by the fact that as parameters change, the state of the system can be carried over the edge of the upper surface, after which it drops discontinuously to the lower surface — a potentially catastrophic jump (e.g., the sudden buckling of a bridge or collapse of an ecosystem).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Bead on a Tilted Wire)</span></p>

A bead of mass $m$ slides along a straight wire inclined at angle $\theta$ to the horizontal. The bead is attached to a spring of stiffness $k$ and relaxed length $L_0$, and is also acted on by gravity. When $\theta = 0$ (horizontal wire), the system has perfect left-right symmetry, and the stability of $x = 0$ depends on whether $L_0 < a$ (spring in tension, $x = 0$ stable) or $L_0 > a$ (spring in compression, pitchfork bifurcation to two symmetric equilibria). Tilting the wire ($\theta \neq 0$) breaks the symmetry — an imperfect bifurcation. For steep enough tilt, the uphill equilibrium may suddenly disappear, causing the bead to jump catastrophically to the downhill equilibrium.

</div>

### Insect Outbreak

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Spruce Budworm Model — Ludwig et al. 1978)</span></p>

The spruce budworm is a serious pest in eastern Canada. Ludwig et al. (1978) proposed a model exploiting the separation of time scales: budworm population evolves *fast* (months), while the forest grows *slowly* (decades). Hence forest variables ($K$, $R$) are treated as constants for the budworm dynamics:

$$\dot{N} = RN\left(1 - \frac{N}{K}\right) - p(N),$$

where $p(N) = \frac{BN^2}{A^2 + N^2}$ models predation by birds. The predation is negligible for small $N$, turns on sharply near $N = A$, and saturates at $B$ for $N \gg A$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dimensionless Formulation)</span></p>

Setting $x = N/A$, $\tau = Bt/A$, $r = RA/B$, $k = K/A$, the model becomes

$$\frac{dx}{d\tau} = rx\left(1 - \frac{x}{k}\right) - \frac{x^2}{1 + x^2},$$

where $r$ and $k$ are the dimensionless growth rate and carrying capacity. The nondimensionalization was chosen so that all dimensionless parameters appear in the *logistic* part, while the predation term $x^2/(1+x^2)$ is parameter-free.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Analysis of Fixed Points)</span></p>

The fixed point $x^* = 0$ is always unstable. The other fixed points satisfy

$$r\left(1 - \frac{x}{k}\right) = \frac{x}{1 + x^2}.$$

The left side is a straight line (slope $-r/k$, intercepts at $k$ and $r$). The right side is a fixed curve $x/(1+x^2)$, independent of parameters — this is why the nondimensionalization was so convenient.

* For small $k$: the line intersects the curve at exactly **one** point — a single stable equilibrium (the **refuge** level $a$).
* For large $k$: there can be **one, two, or three** intersections depending on $r$. When three exist, the stability alternates: $a$ is stable (refuge), $b$ is unstable (threshold), $c$ is stable (**outbreak** level).

An **outbreak** occurs when the parameters $r$ and $k$ drift so that the refuge point $a$ and threshold $b$ coalesce in a saddle-node bifurcation. The population then jumps catastrophically to the outbreak level $c$. Worse, even if the parameters are restored, hysteresis prevents the population from dropping back — the outbreak persists.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bifurcation Curves and Stability Diagram)</span></p>

The bifurcation curves in the $(k, r)$ plane are computed in parametric form. Requiring the line to be tangent to the curve $x/(1+x^2)$ gives:

$$r(x) = \frac{2x^3}{(1 + x^2)^2}, \qquad k(x) = \frac{2x^3}{x^2 - 1}, \qquad x > 1.$$

Plotting $(k(x), r(x))$ yields two curves that divide the $(k, r)$ plane into three regions:
* **Refuge** only (low $r$): one stable fixed point at a low level.
* **Outbreak** only (high $r$): one stable fixed point at a high level.
* **Bistable** (intermediate): both refuge and outbreak levels are stable; the initial condition determines the outcome.

The stability diagram closely resembles the cusp catastrophe of Section 3.6. Biologically plausible parameter values place the system in the bistable region — the forest slowly grows ($k$ increases), until a saddle-node bifurcation triggers a sudden, irreversible outbreak.

</div>

## Chapter 4: Flows on the Circle

### Introduction

So far we have concentrated on the equation $\dot{x} = f(x)$, which we visualized as a vector field on the line. Now we consider a new kind of differential equation and its corresponding phase space. The equation

$$\dot{\theta} = f(\theta)$$

corresponds to a **vector field on the circle**. Here $\theta$ is a point on the circle and $\dot{\theta}$ is the velocity vector at that point, determined by the rule $\dot{\theta} = f(\theta)$. Like the line, the circle is one-dimensional, but it has an important new property: by flowing in one direction, a particle can eventually return to its starting place. Thus periodic solutions become possible for the first time. In other words, *vector fields on the circle provide the most basic model of systems that can oscillate*.

### Examples and Definitions

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Vector Field $\dot{\theta} = \sin\theta$ on the Circle)</span></p>

We assign coordinates on the circle in the usual way, with $\theta = 0$ in the direction of "east" and $\theta$ increasing counterclockwise. The fixed points are defined by $\dot{\theta} = 0$, which occur at $\theta^* = 0$ and $\theta^* = \pi$. Since $\sin\theta > 0$ on the upper semicircle, the flow is counterclockwise there; since $\sin\theta < 0$ on the lower semicircle, the flow is clockwise. Hence $\theta^* = \pi$ is stable and $\theta^* = 0$ is unstable.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Why $\dot{\theta} = \theta$ Is Not a Vector Field on the Circle)</span></p>

The equation $\dot{\theta} = \theta$ cannot be regarded as a vector field on the circle, for $\theta$ in the range $-\infty < \theta < \infty$. The velocity is not uniquely defined: for example, $\theta = 0$ and $\theta = 2\pi$ are two labels for the same point on the circle, but the first implies a velocity of $0$ while the second implies a velocity of $2\pi$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vector Field on the Circle)</span></p>

A **vector field on the circle** is a rule that assigns a unique velocity vector to each point on the circle. In practice, such vector fields arise when we have a first-order system $\dot{\theta} = f(\theta)$, where $f(\theta)$ is a real-valued, $2\pi$-periodic function. That is, $f(\theta + 2\pi) = f(\theta)$ for all real $\theta$. We assume $f(\theta)$ is smooth enough to guarantee existence and uniqueness of solutions.

The periodicity of $f(\theta)$ ensures that the velocity $\dot{\theta}$ is uniquely defined at each point $\theta$ on the circle, whether we call that point $\theta$ or $\theta + 2\pi k$ for any integer $k$.

</div>

### Uniform Oscillator

A point on a circle is often called an **angle** or a **phase**. The simplest oscillator of all is one in which the phase $\theta$ changes uniformly:

$$\dot{\theta} = \omega$$

where $\omega$ is a constant. The solution is

$$\theta(t) = \omega t + \theta_0,$$

which corresponds to uniform motion around the circle at an angular frequency $\omega$. This solution is **periodic**, in the sense that $\theta(t)$ changes by $2\pi$, and therefore returns to the same point on the circle, after a time $T = 2\pi / \omega$. We call $T$ the **period** of the oscillation.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(No Amplitude in This Model)</span></p>

There is no amplitude variable in our system. There is really no amplitude variable in our system — the oscillation occurs at some *fixed* amplitude, corresponding to the radius of our circular phase space. Amplitude plays no role in the dynamics.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Two Joggers — Beat Phenomenon)</span></p>

Two joggers, Speedy and Pokey, are running at a steady pace around a circular track. It takes Speedy $T_1$ seconds to run once around the track, whereas it takes Pokey $T_2 > T_1$ seconds. How long does it take for Speedy to lap Pokey?

*Solution:* Let $\dot{\theta}_1 = \omega_1 = 2\pi / T_1$ and $\dot{\theta}_2 = \omega_2 = 2\pi / T_2$. Define the **phase difference** $\phi = \theta_1 - \theta_2$. Then $\dot{\phi} = \omega_1 - \omega_2$. The lapping time is when $\phi$ increases by $2\pi$:

$$T_{\text{lap}} = \frac{2\pi}{\omega_1 - \omega_2} = \left(\frac{1}{T_1} - \frac{1}{T_2}\right)^{-1}.$$

This illustrates the **beat phenomenon**: two noninteracting oscillators with different frequencies will periodically go in and out of phase with each other.

</div>

### Nonuniform Oscillator

The equation

$$\dot{\theta} = \omega - a\sin\theta$$

arises in many different branches of science and engineering:

* *Electronics* (phase-locked loops)
* *Biology* (oscillating neurons, firefly flashing rhythm, human sleep-wake cycle)
* *Condensed-matter physics* (Josephson junction, charge-density waves)
* *Mechanics* (overdamped pendulum driven by a constant torque)

We assume $\omega > 0$ and $a \geq 0$ for convenience.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Vector Fields of the Nonuniform Oscillator)</span></p>

A typical graph of $f(\theta) = \omega - a\sin\theta$ is a sine wave shifted up by $\omega$. The parameter $\omega$ is the mean and $a$ is the amplitude.

* If $a = 0$: reduces to the uniform oscillator.
* If $a < \omega$: the parameter $a$ introduces a **nonuniformity** in the flow — the flow is fastest at $\theta = -\pi/2$ and slowest at $\theta = \pi/2$. When $a$ is slightly less than $\omega$, the oscillation is very jerky: the phase point $\theta(t)$ takes a long time to pass through a **bottleneck** near $\theta = \pi/2$, after which it zips around the rest of the circle on a much faster time scale.
* If $a = \omega$: the system stops oscillating altogether; a half-stable fixed point has been born in a **saddle-node bifurcation** at $\theta = \pi/2$.
* If $a > \omega$: the half-stable fixed point splits into a stable and unstable fixed point. All trajectories are attracted to the stable fixed point as $t \to \infty$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Linear Stability Analysis for $a > \omega$)</span></p>

The fixed points $\theta^*$ satisfy $\sin\theta^* = \omega / a$, so $\cos\theta^* = \pm\sqrt{1 - (\omega/a)^2}$. Their linear stability is determined by

$$f'(\theta^*) = -a\cos\theta^* = \mp a\sqrt{1 - (\omega/a)^2}.$$

The fixed point with $\cos\theta^* > 0$ is the stable one, since $f'(\theta^*) < 0$.

</div>

#### Oscillation Period

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Period of the Nonuniform Oscillator)</span></p>

For $a < \omega$, the period of oscillation can be found analytically. The time required for $\theta$ to change by $2\pi$ is

$$T = \int dt = \int_0^{2\pi} \frac{d\theta}{\omega - a\sin\theta} = \frac{2\pi}{\sqrt{\omega^2 - a^2}}.$$

The period increases with $a$ and diverges as $a \to \omega^-$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Square-Root Scaling Law)</span></p>

As $a \to \omega^-$, we can estimate:

$$\sqrt{\omega^2 - a^2} = \sqrt{\omega + a}\sqrt{\omega - a} \approx \sqrt{2\omega}\sqrt{\omega - a}.$$

Hence

$$T \approx \frac{\pi\sqrt{2}}{\sqrt{\omega}} \cdot \frac{1}{\sqrt{\omega - a}},$$

which shows that $T$ blows up like $(\omega - a)^{-1/2}$. This is the **square-root scaling law**.

</div>

#### Ghosts and Bottlenecks

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Ghosts and Bottlenecks)</span></p>

The square-root scaling law is a *very general feature of systems that are close to a saddle-node bifurcation*. Just after the fixed points collide, there is a saddle-node remnant or **ghost** that leads to slow passage through a bottleneck.

For example, consider $\dot{\theta} = \omega - a\sin\theta$ for decreasing values of $a$, starting with $a > \omega$. As $a$ decreases, the two fixed points approach each other, collide, and disappear. For $a$ slightly less than $\omega$, the fixed points near $\pi/2$ no longer exist, but they still make themselves felt through a saddle-node ghost: the trajectory slows dramatically near $\theta = \pi/2$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Bottleneck Time via Normal Form)</span></p>

Generically, $\dot{\theta}$ looks *parabolic* near the bottleneck minimum. By a local rescaling, the dynamics can be reduced to the normal form for a saddle-node bifurcation:

$$\dot{x} = r + x^2,$$

where $r$ is proportional to the distance from the bifurcation and $0 < r \ll 1$. The time spent in the bottleneck is

$$T_{\text{bottleneck}} \approx \int_{-\infty}^{\infty} \frac{dx}{r + x^2} = \frac{\pi}{\sqrt{r}},$$

which confirms the generality of the square-root scaling law.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Bottleneck Estimate for the Nonuniform Oscillator)</span></p>

Estimate the period of $\dot{\theta} = \omega - a\sin\theta$ in the limit $a \to \omega^-$, using the normal form method.

*Solution:* The period is essentially the time required to get through the bottleneck. We Taylor-expand about $\theta = \pi/2$, where the bottleneck occurs. Let $\phi = \theta - \pi/2$, where $\phi$ is small. Then

$$\dot{\phi} = \omega - a\sin(\phi + \tfrac{\pi}{2}) = \omega - a\cos\phi = \omega - a + \tfrac{1}{2}a\phi^2 + \cdots$$

Letting $x = (a/2)^{1/2}\phi$ and $r = \omega - a$, we get $(2/a)^{1/2}\dot{x} \approx r + x^2$ to leading order. Separating variables:

$$T \approx (2/a)^{1/2} \int_{-\infty}^{\infty} \frac{dx}{r + x^2} = (2/a)^{1/2} \cdot \frac{\pi}{\sqrt{r}}.$$

Since $a \to \omega^-$, we may replace $2/a$ by $2/\omega$. Hence

$$T \approx \frac{\pi\sqrt{2}}{\sqrt{\omega}} \cdot \frac{1}{\sqrt{\omega - a}},$$

which agrees with the exact result.

</div>

### Overdamped Pendulum

We now consider a simple mechanical example of a nonuniform oscillator: an overdamped pendulum driven by a constant torque. Let $\theta$ denote the angle between the pendulum and the downward vertical, and suppose that $\theta$ increases counterclockwise.

Newton's law yields the second-order equation

$$mL^2 \ddot{\theta} + b\dot{\theta} + mgL\sin\theta = \Gamma,$$

where $m$ is the mass, $L$ is the length, $b$ is a viscous damping constant, $g$ is the acceleration due to gravity, and $\Gamma$ is a constant applied torque. In the **overdamped limit** of extremely large $b$, the inertia term $mL^2\ddot{\theta}$ is negligible and the equation becomes

$$b\dot{\theta} + mgL\sin\theta = \Gamma.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dimensionless Form of the Overdamped Pendulum)</span></p>

Nondimensionalizing by letting

$$\tau = \frac{mgL}{b}\,t, \qquad \gamma = \frac{\Gamma}{mgL},$$

the equation becomes

$$\theta' = \gamma - \sin\theta,$$

where $\theta' = d\theta/d\tau$. The dimensionless group $\gamma$ is the ratio of the applied torque to the maximum gravitational torque.

* If $\gamma > 1$: the applied torque can never be balanced by the gravitational torque and *the pendulum will overturn continually*. The rotation rate is nonuniform — gravity helps the applied torque on one side and opposes it on the other.
* As $\gamma \to 1^+$: the pendulum takes longer and longer to climb past $\theta = \pi/2$ on the slow side.
* When $\gamma = 1$: a fixed point appears at $\theta^* = \pi/2$, and then splits into two for $\gamma < 1$.
* When $\gamma = 0$: the applied torque vanishes, yielding an unstable equilibrium at the top (inverted pendulum) and a stable equilibrium at the bottom.

</div>

### Fireflies

Fireflies provide one of the most spectacular examples of synchronization in nature. In some parts of southeast Asia, thousands of male fireflies gather in trees at night and flash on and off in unison.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Phase Walkthrough and Entrainment)</span></p>

When one firefly sees the flash of another, it adjusts its rhythm — slowing down or speeding up so as to flash more nearly in phase on the next cycle. When a firefly is able to match its frequency to the periodic stimulus, it has been **entrained**. If the stimulus is too fast or too slow, the firefly cannot keep up and **entrainment is lost** — a kind of beat phenomenon called **phase walkthrough** or **phase drift**.

</div>

#### Model

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Firefly Entrainment Model — Ermentrout and Rinzel 1984)</span></p>

Suppose $\theta(t)$ is the phase of the firefly's flashing rhythm ($\theta = 0$ corresponds to a flash being emitted). In the absence of stimuli, $\dot{\theta} = \omega$. A periodic stimulus with phase $\Theta$ satisfies $\dot{\Theta} = \Omega$. The firefly's response to this stimulus is modeled by

$$\dot{\theta} = \omega + A\sin(\Theta - \theta),$$

where $A > 0$ is the **resetting strength** — the firefly's ability to modify its instantaneous frequency. If the stimulus is ahead ($0 < \Theta - \theta < \pi$), the firefly speeds up; if it's flashing too early, it slows down.

</div>

#### Analysis

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reduction to the Nonuniform Oscillator)</span></p>

Define the phase difference $\phi = \Theta - \theta$. Subtracting yields

$$\dot{\phi} = \dot{\Theta} - \dot{\theta} = \Omega - \omega - A\sin\phi,$$

which is a nonuniform oscillator equation. Nondimensionalizing with $\tau = At$ and $\mu = (\Omega - \omega)/A$ gives

$$\phi' = \mu - \sin\phi.$$

The dimensionless group $\mu$ measures the frequency difference relative to the resetting strength.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Phase-Locking and Phase Drift)</span></p>

* When $\mu = 0$: all trajectories flow toward a stable fixed point at $\phi^* = 0$. The firefly entrains with **zero phase difference** — it flashes *simultaneously* with the stimulus.
* When $0 < \mu < 1$: the stable fixed point shifts to $\phi^* > 0$. The firefly is still entrained but now **phase-locked**: it flashes at the same frequency as the stimulus but with a constant phase lag $\phi^*$. This makes sense because $\mu > 0$ means $\Omega > \omega$ — the stimulus is inherently faster, so the firefly always lags behind.
* When $\mu = 1$: the stable and unstable fixed points coalesce in a saddle-node bifurcation.
* When $\mu > 1$: both fixed points have disappeared. The phase difference $\phi$ increases indefinitely, corresponding to **phase drift**. The phases don't separate at a uniform rate — $\phi$ increases most slowly under the minimum of the sine wave at $\phi = \pi/2$, and most rapidly under the maximum at $\phi = -\pi/2$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Range of Entrainment and Phase Drift Period)</span></p>

Entrainment is predicted within a symmetric interval of driving frequencies:

$$\omega - A \leq \Omega \leq \omega + A.$$

This is called the **range of entrainment**. During entrainment, the phase difference satisfies

$$\sin\phi^* = \frac{\Omega - \omega}{A},$$

where $-\pi/2 \leq \phi^* \leq \pi/2$ corresponds to the *stable* fixed point.

For $\mu > 1$, the period of phase drift is

$$T_{\text{drift}} = \frac{2\pi}{\sqrt{(\Omega - \omega)^2 - A^2}}.$$

</div>

### Superconducting Josephson Junctions

Josephson junctions are superconducting devices capable of generating voltage oscillations of extraordinarily high frequency, typically $10^{10} - 10^{11}$ cycles per second. They have great technological promise as amplifiers, voltage standards, detectors, mixers, and fast switching devices for digital circuits.

#### Physical Background

A Josephson junction consists of two closely spaced superconductors separated by a weak connection (an insulator, a normal metal, a semiconductor, or a weakened superconductor). The two superconducting regions may be characterized by quantum mechanical wave functions $\psi_1 e^{i\phi_1}$ and $\psi_2 e^{i\phi_2}$. In the superconducting ground state, the electrons form "Cooper pairs" that all adopt the same phase — a macroscopic quantum coherence.

#### The Josephson Relations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Josephson Relations)</span></p>

When a Josephson junction is connected to a dc current source with a constant current $I > 0$, the phase difference $\phi = \phi_2 - \phi_1$ satisfies two fundamental relations:

1. **Josephson current-phase relation:**

   $$I = I_c \sin\phi,$$

   where $I_c$ is the **critical current**. For $I < I_c$, a constant phase difference is maintained and no voltage develops — the junction acts as if it had zero resistance.

2. **Josephson voltage-phase relation:**

   $$V = \frac{\hbar}{2e}\dot{\phi},$$

   where $V(t)$ is the instantaneous voltage across the junction, $\hbar$ is Planck's constant divided by $2\pi$, and $e$ is the electron charge.

When $I$ exceeds $I_c$, a constant phase difference can no longer be maintained and a voltage develops across the junction.

</div>

#### Equivalent Circuit and Pendulum Analog

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(RCSJ Model)</span></p>

The total current through the junction includes contributions from the supercurrent, a displacement current (capacitor $C$), and an ordinary current (resistor $R$). Applying Kirchhoff's laws to the equivalent parallel circuit:

$$C\dot{V} + \frac{V}{R} + I_c\sin\phi = I.$$

Using the voltage-phase relation $V = (\hbar/2e)\dot{\phi}$, this becomes

$$\frac{\hbar C}{2e}\ddot{\phi} + \frac{\hbar}{2eR}\dot{\phi} + I_c\sin\phi = I,$$

which is precisely analogous to a damped pendulum driven by a constant torque $mL^2\ddot{\theta} + b\dot{\theta} + mgL\sin\theta = \Gamma$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Pendulum–Junction Analogy)</span></p>

| Pendulum | Josephson junction |
| --- | --- |
| Angle $\theta$ | Phase difference $\phi$ |
| Angular velocity $\dot{\theta}$ | Voltage $\frac{\hbar}{2e}\dot{\phi}$ |
| Mass $m$ | Capacitance $C$ |
| Applied torque $\Gamma$ | Bias current $I$ |
| Damping constant $b$ | Conductance $1/R$ |
| Maximum gravitational torque $mgL$ | Critical current $I_c$ |

</div>

#### Dimensionless Formulation

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dimensionless Josephson Equation)</span></p>

Defining dimensionless time $\tau = (2eI_cR / \hbar)\,t$, the equation becomes

$$\beta\phi'' + \phi' + \sin\phi = \frac{I}{I_c},$$

where $\phi' = d\phi/d\tau$ and the **McCumber parameter**

$$\beta = \frac{2eI_cR^2C}{\hbar}$$

is a dimensionless capacitance. Depending on the junction, $\beta$ can range from $\beta \approx 10^{-6}$ to $\beta \approx 10^6$.

In the **overdamped limit** $\beta \ll 1$, the term $\beta\phi''$ may be neglected, reducing the equation to a nonuniform oscillator:

$$\phi' = \frac{I}{I_c} - \sin\phi.$$

From Section 4.3, the solutions tend to a stable fixed point when $I < I_c$, and vary periodically when $I > I_c$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(I–V Curve in the Overdamped Limit)</span></p>

Find the **current–voltage curve** analytically in the overdamped limit: find $\langle V \rangle$ as a function of constant applied current $I$.

*Solution:* Since $\langle V \rangle = (\hbar/2e)\langle\dot{\phi}\rangle$ from the voltage-phase relation, and $\langle\dot{\phi}\rangle = (d\phi/dt) = (2eI_cR/\hbar)\langle\phi'\rangle$, we have

$$\langle V \rangle = I_c R \langle\phi'\rangle.$$

* When $I \leq I_c$: all solutions approach a fixed point $\phi^* = \sin^{-1}(I/I_c)$. Thus $\phi' \to 0$ in steady state, so $\langle V \rangle = 0$.
* When $I > I_c$: all solutions are periodic with period

$$T = \frac{2\pi}{\sqrt{(I/I_c)^2 - 1}},$$

obtained from the formula of Section 4.3. We compute $\langle\phi'\rangle = 2\pi / T$.

Combining:

$$\langle V \rangle = \begin{cases} 0 & \text{for } I \leq I_c, \\ I_c R\sqrt{(I/I_c)^2 - 1} & \text{for } I > I_c. \end{cases}$$

As $I$ increases past $I_c$, $\langle V \rangle$ rises sharply and eventually asymptotes to the Ohmic behavior $\langle V \rangle \approx IR$ for $I \gg I_c$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hysteresis When $\beta \neq 0$)</span></p>

The analysis above applies only to the overdamped limit $\beta \ll 1$. When $\beta$ is not negligible, the I–V curve can be **hysteretic**: as the bias current is increased slowly from $I = 0$, the voltage remains at $V = 0$ until $I > I_c$, at which point the voltage jumps up to a nonzero value. However, if we then slowly *decrease* $I$, the voltage doesn't drop back to zero at $I_c$ — we have to go *below* $I_c$ before the voltage returns to zero.

The hysteresis arises because the system has **inertia** when $\beta \neq 0$. In the pendulum analog, $I_c$ is analogous to the critical torque needed to get the pendulum overturning. Once the pendulum has started whirling, its inertia keeps it going even if the torque is reduced somewhat below the critical value.

</div>

## Chapter 5: Linear Systems

### Introduction

In one-dimensional phase spaces the flow is extremely confined — all trajectories are forced to move monotonically or remain constant. In higher-dimensional phase spaces, trajectories have much more room to maneuver, and so a wider range of dynamical behavior becomes possible. Rather than attack all this complexity at once, we begin with the simplest class of higher-dimensional systems, namely *linear systems in two dimensions*. These systems are interesting in their own right, and they also play an important role in the classification of fixed points of *nonlinear* systems.

### Definitions and Examples

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Two-Dimensional Linear System)</span></p>

A **two-dimensional linear system** is a system of the form

$$\dot{x} = ax + by, \qquad \dot{y} = cx + dy,$$

where $a, b, c, d$ are parameters. In matrix form this is written compactly as

$$\dot{\mathbf{x}} = A\mathbf{x},$$

where $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ and $\mathbf{x} = \begin{pmatrix} x \\ y \end{pmatrix}$.

Such a system is **linear** in the sense that if $\mathbf{x}_1$ and $\mathbf{x}_2$ are solutions, then so is any linear combination $c_1\mathbf{x}_1 + c_2\mathbf{x}_2$. Note that $\dot{\mathbf{x}} = \mathbf{0}$ when $\mathbf{x} = \mathbf{0}$, so $\mathbf{x}^* = \mathbf{0}$ is always a fixed point for any choice of $A$.

The solutions of $\dot{\mathbf{x}} = A\mathbf{x}$ can be visualized as trajectories moving on the $(x, y)$ plane, in this context called the **phase plane**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Simple Harmonic Oscillator)</span></p>

The vibrations of a mass on a linear spring are governed by $m\ddot{x} + kx = 0$. The **state** of the system is characterized by position $x$ and velocity $v$. Letting $\omega^2 = k/m$, the system becomes

$$\dot{x} = v, \qquad \dot{v} = -\omega^2 x.$$

The vector field $(\dot{x}, \dot{v}) = (v, -\omega^2 x)$ swirls about the origin. The origin is a **fixed point** (a phase point placed there would remain motionless). A phase point starting anywhere else would circulate around the origin and return to its starting point, forming **closed orbits** — these correspond to periodic oscillations of the mass. The closed orbits are actually *ellipses* given by $\omega^2 x^2 + v^2 = C$, which is equivalent to conservation of energy.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Uncoupled System with Parameter $a$)</span></p>

Consider $\dot{\mathbf{x}} = A\mathbf{x}$ where $A = \begin{pmatrix} a & 0 \\ 0 & -1 \end{pmatrix}$. The two equations are **uncoupled**: $\dot{x} = ax$ and $\dot{y} = -y$, with solutions

$$x(t) = x_0 e^{at}, \qquad y(t) = y_0 e^{-t}.$$

In each case, $y(t)$ decays exponentially. The phase portrait depends on $a$:

* **$a < -1$:** Both components decay; trajectories approach the origin tangent to the *slower* direction ($y$-axis). The fixed point is a **stable node**.
* **$a = -1$:** A special case — decay rates are equal and all trajectories are straight lines through the origin. The fixed point is a **star node** (symmetrical node).
* **$-1 < a < 0$:** Still a stable node, but now trajectories approach tangent to the $x$-direction (the slower-decaying one).
* **$a = 0$:** $x(t) \equiv x_0$, so there is an entire **line of fixed points** along the $x$-axis. All trajectories approach these fixed points along vertical lines.
* **$a > 0$:** $x(t)$ grows exponentially. Most trajectories veer away from $\mathbf{x}^*$ and head to infinity. The fixed point is a **saddle point**. The $y$-axis is the **stable manifold** (initial conditions $\mathbf{x}_0$ such that $\mathbf{x}(t) \to \mathbf{x}^*$ as $t \to \infty$); the $x$-axis is the **unstable manifold** ($\mathbf{x}(t) \to \mathbf{x}^*$ as $t \to -\infty$).

</div>

#### Stability Language

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Types of Stability)</span></p>

* **Attracting:** A fixed point $\mathbf{x}^* = \mathbf{0}$ is **attracting** if all trajectories that start near $\mathbf{x}^*$ approach it as $t \to \infty$. If it attracts *all* trajectories in the phase plane, it is called **globally attracting**.
* **Liapunov stable:** A fixed point $\mathbf{x}^*$ is **Liapunov stable** if all trajectories that start sufficiently close to $\mathbf{x}^*$ remain close to it for all time.
* **Asymptotically stable (stable):** If a fixed point is *both* Liapunov stable and attracting, it is called **asymptotically stable** or simply **stable**.
* **Neutrally stable:** A fixed point that is Liapunov stable but *not* attracting (nearby trajectories are neither attracted nor repelled). Example: the center of the simple harmonic oscillator.
* **Unstable:** A fixed point that is neither attracting nor Liapunov stable.

Note: it is possible for a fixed point to be attracting but not Liapunov stable (e.g., $\dot{\theta} = 1 - \cos\theta$ on the circle — $\theta^* = 0$ attracts all trajectories as $t \to \infty$, but trajectories starting close to $\theta^*$ must travel all the way around the circle before returning).

Graphical convention: open dots denote unstable fixed points; solid black dots denote Liapunov stable fixed points.

</div>

### Classification of Linear Systems

The examples in Section 5.1 had special matrices with zero entries. Now we study the general case of an arbitrary $2 \times 2$ matrix $A$, aiming to classify all possible phase portraits.

#### Eigenvalues and Eigenvectors

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Straight-Line Trajectories and Eigensolutions)</span></p>

In the uncoupled examples, the coordinate axes played a crucial geometric role — they contained special **straight-line trajectories** along which the motion is purely exponential growth or decay. For the general case, we seek trajectories of the form

$$\mathbf{x}(t) = e^{\lambda t}\mathbf{v},$$

where $\mathbf{v} \neq \mathbf{0}$ is a fixed vector and $\lambda$ is a growth rate. Substituting into $\dot{\mathbf{x}} = A\mathbf{x}$ yields

$$A\mathbf{v} = \lambda\mathbf{v},$$

so $\mathbf{v}$ must be an **eigenvector** of $A$ with corresponding **eigenvalue** $\lambda$. We call $\mathbf{x}(t) = e^{\lambda t}\mathbf{v}$ an **eigensolution**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Eigenvalues of a $2 \times 2$ Matrix)</span></p>

For $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$, the eigenvalues are given by the **characteristic equation** $\det(A - \lambda I) = 0$, which expands to

$$\lambda^2 - \tau\lambda + \Delta = 0,$$

where

$$\tau = \operatorname{trace}(A) = a + d, \qquad \Delta = \det(A) = ad - bc.$$

The eigenvalues are

$$\lambda_{1,2} = \frac{\tau \pm \sqrt{\tau^2 - 4\Delta}}{2}.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(General Solution for Distinct Eigenvalues)</span></p>

When $\lambda_1 \neq \lambda_2$, the corresponding eigenvectors $\mathbf{v}_1$ and $\mathbf{v}_2$ are linearly independent and span the entire plane. Any initial condition $\mathbf{x}_0 = c_1\mathbf{v}_1 + c_2\mathbf{v}_2$ yields the general solution

$$\mathbf{x}(t) = c_1 e^{\lambda_1 t}\mathbf{v}_1 + c_2 e^{\lambda_2 t}\mathbf{v}_2.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Solving a Linear System — Saddle Point)</span></p>

Solve $\dot{x} = x + y$, $\dot{y} = 4x - 2y$ with initial condition $(x_0, y_0) = (2, -3)$.

*Solution:* The matrix $A = \begin{pmatrix} 1 & 1 \\ 4 & -2 \end{pmatrix}$ has $\tau = -1$ and $\Delta = -6$, so $\lambda^2 + \lambda - 6 = 0$ gives $\lambda_1 = 2$ and $\lambda_2 = -3$.

For $\lambda_1 = 2$: the eigenvector equation gives $\mathbf{v}_1 = (1, 1)$.
For $\lambda_2 = -3$: the eigenvector equation gives $\mathbf{v}_2 = (1, -4)$.

The general solution is $\mathbf{x}(t) = c_1 \binom{1}{1} e^{2t} + c_2 \binom{1}{-4} e^{-3t}$. Applying the initial condition: $c_1 = 1$, $c_2 = 1$, so

$$x(t) = e^{2t} + e^{-3t}, \qquad y(t) = e^{2t} - 4e^{-3t}.$$

The phase portrait is a **saddle point**: the stable manifold is the line spanned by $\mathbf{v}_2 = (1, -4)$ (the decaying eigensolution), and the unstable manifold is the line spanned by $\mathbf{v}_1 = (1, 1)$ (the growing eigensolution).

</div>

#### Real Distinct Eigenvalues: Nodes

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stable and Unstable Nodes)</span></p>

When $\lambda_2 < \lambda_1 < 0$, both eigensolutions decay exponentially — the fixed point is a **stable node**. Trajectories typically approach the origin tangent to the **slow eigendirection**, the direction spanned by the eigenvector with the *smaller* $\lvert\lambda\rvert$. In backwards time ($t \to -\infty$), trajectories become parallel to the **fast eigendirection**.

If we reverse all arrows (equivalently, if $0 < \lambda_1 < \lambda_2$), we obtain an **unstable node**.

</div>

#### Complex Eigenvalues: Spirals and Centers

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Complex Eigenvalues — Spirals and Centers)</span></p>

Complex eigenvalues occur when $\tau^2 - 4\Delta < 0$. Writing $\lambda_{1,2} = \alpha \pm i\omega$ where

$$\alpha = \frac{\tau}{2}, \qquad \omega = \frac{1}{2}\sqrt{4\Delta - \tau^2},$$

the general solution involves combinations of $e^{\alpha t}\cos\omega t$ and $e^{\alpha t}\sin\omega t$.

* If $\alpha = \operatorname{Re}(\lambda) < 0$: exponentially **decaying oscillations** — a **stable spiral**.
* If $\alpha = \operatorname{Re}(\lambda) > 0$: exponentially **growing oscillations** — an **unstable spiral**.
* If $\alpha = 0$ (purely imaginary eigenvalues): all solutions are periodic with period $T = 2\pi/\omega$ — a **center**. The oscillations have fixed amplitude and the fixed point is neutrally stable.

For both centers and spirals, the sense of rotation (clockwise or counterclockwise) is determined by computing a few vectors of the vector field.

</div>

#### Repeated Eigenvalues: Stars and Degenerate Nodes

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Repeated Eigenvalues)</span></p>

When $\lambda_1 = \lambda_2 = \lambda$, there are two possibilities:

1. **Two independent eigenvectors** (the eigenspace is two-dimensional). Then every vector is an eigenvector with eigenvalue $\lambda$, so $A = \begin{pmatrix} \lambda & 0 \\ 0 & \lambda \end{pmatrix}$. All trajectories are straight lines through the origin ($\mathbf{x}(t) = e^{\lambda t}\mathbf{x}_0$). This is a **star node**. If $\lambda \neq 0$, trajectories move radially inward ($\lambda < 0$) or outward ($\lambda > 0$). If $\lambda = 0$, the whole plane is filled with fixed points.

2. **Only one independent eigenvector** (the eigenspace is one-dimensional). This happens for matrices of the form $A = \begin{pmatrix} \lambda & b \\ 0 & \lambda \end{pmatrix}$ with $b \neq 0$. The fixed point is a **degenerate node**. As $t \to +\infty$ and as $t \to -\infty$, all trajectories become parallel to the single available eigendirection. The degenerate node lies on the borderline between a spiral and a node — the trajectories are trying to wind around in a spiral, but they don't quite make it.

</div>

#### Classification of Fixed Points

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Classification Diagram in the $(\Delta, \tau)$ Plane)</span></p>

The type and stability of all fixed points of a $2 \times 2$ linear system $\dot{\mathbf{x}} = A\mathbf{x}$ can be determined from the **trace** $\tau = \operatorname{trace}(A)$ and the **determinant** $\Delta = \det(A)$:

* **$\Delta < 0$:** Eigenvalues are real with opposite signs — **saddle point**.
* **$\Delta > 0$, $\tau^2 - 4\Delta > 0$:** Eigenvalues are real with the same sign — **node**.
  * $\tau < 0$: stable node. $\tau > 0$: unstable node.
* **$\Delta > 0$, $\tau^2 - 4\Delta < 0$:** Eigenvalues are complex conjugates — **spiral**.
  * $\tau < 0$: stable spiral. $\tau > 0$: unstable spiral.
* **$\Delta > 0$, $\tau = 0$:** Eigenvalues are purely imaginary — **center** (neutrally stable).
* **$\Delta > 0$, $\tau^2 - 4\Delta = 0$:** Repeated eigenvalues — **star node** or **degenerate node** (borderline between nodes and spirals).
* **$\Delta = 0$:** At least one eigenvalue is zero — **non-isolated fixed point** (a line or plane of fixed points).

Saddle points, nodes, and spirals are the major types — they occur in large open regions of the $(\Delta, \tau)$ plane. Centers, stars, degenerate nodes, and non-isolated fixed points are **borderline cases** that occur along curves. Of these, centers are by far the most important, arising commonly in frictionless mechanical systems where energy is conserved.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Classifying Fixed Points from $\tau$ and $\Delta$)</span></p>

**Example 1:** $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$. Then $\Delta = 4 - 6 = -2 < 0$, so the fixed point is a **saddle point**.

**Example 2:** $A = \begin{pmatrix} 2 & 1 \\ 3 & 4 \end{pmatrix}$. Then $\Delta = 8 - 3 = 5 > 0$, $\tau = 6 > 0$, and $\tau^2 - 4\Delta = 36 - 20 = 16 > 0$, so the fixed point is an **unstable node**.

</div>

### Love Affairs

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Romeo and Juliet — A Linear Model of Love)</span></p>

Let $R(t)$ = Romeo's love/hate for Juliet and $J(t)$ = Juliet's love/hate for Romeo (positive values signify love, negative values signify hate). Romeo is in love with Juliet, but Juliet is a fickle lover: the more Romeo loves her, the more she wants to run away. But when Romeo backs off, Juliet finds him strangely attractive. Romeo echoes Juliet — he warms up when she loves him, and grows cold when she hates him. A model for this is

$$\dot{R} = aJ, \qquad \dot{J} = -bR,$$

where $a, b > 0$. The governing system has a center at $(R, J) = (0, 0)$ — a neverending cycle of love and hate. At least they manage to achieve simultaneous love one-quarter of the time.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Two Cautious Lovers)</span></p>

What happens when two identically cautious lovers get together? The system is

$$\dot{R} = aR + bJ, \qquad \dot{J} = bR + aJ,$$

with $a < 0$ (cautiousness) and $b > 0$ (responsiveness). The matrix $A = \begin{pmatrix} a & b \\ b & a \end{pmatrix}$ has $\tau = 2a < 0$, $\Delta = a^2 - b^2$, and $\tau^2 - 4\Delta = 4b^2 > 0$. The eigenvalues are $\lambda_1 = a + b$ and $\lambda_2 = a - b$, with eigenvectors $\mathbf{v}_1 = (1, 1)$ and $\mathbf{v}_2 = (1, -1)$.

* If $a^2 > b^2$ (caution dominates): $\Delta > 0$ and both eigenvalues are negative — a **stable node**. The relationship fizzles out to mutual indifference. Excessive caution leads to apathy.
* If $a^2 < b^2$ (responsiveness dominates): $\Delta < 0$ — a **saddle point**. The relationship is explosive. Depending on initial feelings, they either end up in a love fest or a war. In either case, all trajectories approach the line $R = J$, so their feelings are eventually mutual.

</div>

## Chapter 6: Phase Plane

### Introduction

This chapter begins our study of two-dimensional *nonlinear* systems. First we consider some of their general properties. Then we classify the kinds of fixed points that can arise, building on our knowledge of linear systems (Chapter 5). The theory is further developed through a series of examples from biology (competition between two species) and physics (conservative systems, reversible systems, and the pendulum). The chapter concludes with a discussion of index theory, a topological method that provides global information about the phase portrait.

This chapter is mainly about fixed points. The next two chapters will discuss closed orbits and bifurcations in two-dimensional systems.

### Phase Portraits

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Two-Dimensional Nonlinear System)</span></p>

The general form of a vector field on the phase plane is

$$\dot{x}_1 = f_1(x_1, x_2), \qquad \dot{x}_2 = f_2(x_1, x_2),$$

or in vector notation $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$, where $\mathbf{x} = (x_1, x_2)$ and $\mathbf{f}(\mathbf{x}) = (f_1(\mathbf{x}), f_2(\mathbf{x}))$. Here $\mathbf{x}$ represents a point in the phase plane, and $\dot{\mathbf{x}}$ is the velocity vector at that point. By flowing along the vector field, a phase point traces out a solution $\mathbf{x}(t)$, corresponding to a trajectory winding through the phase plane. The entire phase plane is filled with trajectories, since each point can play the role of an initial condition.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Salient Features of Phase Portraits)</span></p>

The most important features of any phase portrait are:

1. The **fixed points** ($\mathbf{f}(\mathbf{x}^*) = \mathbf{0}$), corresponding to steady states or equilibria.
2. The **closed orbits** — periodic solutions for which $\mathbf{x}(t + T) = \mathbf{x}(t)$ for some $T > 0$.
3. The arrangement of trajectories near the fixed points and closed orbits.
4. The stability or instability of the fixed points and closed orbits.

For nonlinear systems, there's typically no hope of finding trajectories analytically. Our goal is to determine the *qualitative* behavior of the solutions — the system's phase portrait — directly from the properties of $\mathbf{f}(\mathbf{x})$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Nullclines)</span></p>

To sketch the phase portrait, it is helpful to plot the **nullclines**, defined as the curves where either $\dot{x} = 0$ or $\dot{y} = 0$. The nullclines indicate where the flow is purely horizontal or vertical:

* The $\dot{y} = 0$ nullcline: the flow is purely horizontal along this curve.
* The $\dot{x} = 0$ nullcline: the flow is purely vertical along this curve.

The nullclines also partition the plane into regions where $\dot{x}$ and $\dot{y}$ have various signs, making it possible to sketch representative vectors and get a good sense of the overall flow pattern.

</div>

### Existence, Uniqueness, and Topological Consequences

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Existence and Uniqueness)</span></p>

Consider the initial value problem $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$, $\mathbf{x}(0) = \mathbf{x}_0$. Suppose that $\mathbf{f}$ is continuous and that all its partial derivatives $\partial f_i / \partial x_j$, $i, j = 1, \ldots, n$, are continuous for $\mathbf{x}$ in some open connected set $D \subset \mathbb{R}^n$. Then for $\mathbf{x}_0 \in D$, the initial value problem has a solution $\mathbf{x}(t)$ on some time interval $(-\tau, \tau)$ about $t = 0$, and the solution is unique.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Topological Consequences)</span></p>

The existence and uniqueness theorem has an important corollary: **different trajectories never intersect**. If two trajectories *did* intersect, then there would be two solutions starting from the same point (the crossing point), violating uniqueness. In more intuitive language, a trajectory can't move in two directions at once.

In two-dimensional phase spaces (as opposed to higher-dimensional ones), these results have especially strong topological consequences. For example, suppose there is a closed orbit $C$ in the phase plane. Then any trajectory starting inside $C$ is trapped in there forever. What is its fate? If there are no fixed points inside $C$, then the **Poincare-Bendixson theorem** states that the trajectory must eventually approach a closed orbit.

</div>

### Fixed Points and Linearization

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Jacobian Matrix and Linearized System)</span></p>

Consider the system $\dot{x} = f(x, y)$, $\dot{y} = g(x, y)$ with a fixed point at $(x^*, y^*)$. Let $u = x - x^*$ and $v = y - y^*$ denote small disturbances from the fixed point. Taylor-expanding about $(x^*, y^*)$:

$$\dot{u} = u\frac{\partial f}{\partial x} + v\frac{\partial f}{\partial y} + O(u^2, v^2, uv),$$

$$\dot{v} = u\frac{\partial g}{\partial x} + v\frac{\partial g}{\partial y} + O(u^2, v^2, uv),$$

where all partial derivatives are evaluated at $(x^*, y^*)$. The matrix

$$A = \begin{pmatrix} \frac{\partial f}{\partial x} & \frac{\partial f}{\partial y} \\ \frac{\partial g}{\partial x} & \frac{\partial g}{\partial y} \end{pmatrix}\bigg\rvert_{(x^*, y^*)}$$

is called the **Jacobian matrix** at the fixed point $(x^*, y^*)$. It is the multivariable analog of the derivative $f'(x^*)$ in Section 2.4.

Neglecting the quadratic terms, we obtain the **linearized system**:

$$\begin{pmatrix} \dot{u} \\ \dot{v} \end{pmatrix} = A \begin{pmatrix} u \\ v \end{pmatrix},$$

whose dynamics can be analyzed by the methods of Chapter 5.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Effect of Small Nonlinear Terms)</span></p>

Is it safe to neglect the quadratic terms? The answer is *yes, as long as the fixed point for the linearized system is not one of the borderline cases* discussed in Section 5.2. If the linearized system predicts a saddle, node, or spiral, then the fixed point *really is* a saddle, node, or spiral for the original nonlinear system.

The borderline cases (centers, degenerate nodes, stars, non-isolated fixed points) are much more delicate — they can be altered by small nonlinear terms.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Linearization Predicting Correctly)</span></p>

Find all fixed points of $\dot{x} = -x + x^3$, $\dot{y} = -2y$ and use linearization to classify them.

*Solution:* Fixed points occur where $\dot{x} = 0$ and $\dot{y} = 0$ simultaneously, giving $(0, 0)$, $(1, 0)$, and $(-1, 0)$. The Jacobian is

$$A = \begin{pmatrix} -1 + 3x^2 & 0 \\ 0 & -2 \end{pmatrix}.$$

* At $(0, 0)$: $A = \begin{pmatrix} -1 & 0 \\ 0 & -2 \end{pmatrix}$, so the origin is a **stable node**.
* At $(\pm 1, 0)$: $A = \begin{pmatrix} 2 & 0 \\ 0 & -2 \end{pmatrix}$, so both are **saddle points**.

Since stable nodes and saddle points are not borderline cases, the linearization predicts correctly.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Linearization Failing for a Center)</span></p>

Consider $\dot{x} = -y + ax(x^2 + y^2)$, $\dot{y} = x + ay(x^2 + y^2)$, where $a$ is a parameter. The linearization about the origin gives $A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$, which has $\tau = 0$, $\Delta = 1 > 0$, so it *incorrectly* predicts a center for all values of $a$.

Switching to polar coordinates: $\dot{r} = ar^3$, $\dot{\theta} = 1$. All trajectories rotate about the origin with constant angular velocity $\dot{\theta} = 1$. But the radial motion depends on $a$:

* $a < 0$: $r(t) \to 0$ — a **stable spiral**.
* $a = 0$: $r(t) = r_0$ — a **center** (only this case agrees with linearization).
* $a > 0$: $r(t) \to \infty$ — an **unstable spiral**.

This shows why centers are so delicate: all trajectories are required to close *perfectly* after one cycle. The slightest perturbation converts the center into a spiral.

</div>

#### Hyperbolic Fixed Points and Structural Stability

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hyperbolic Fixed Point, Structural Stability)</span></p>

A fixed point is called **hyperbolic** if $\operatorname{Re}(\lambda_i) \neq 0$ for all eigenvalues of the Jacobian. Hyperbolic fixed points are sturdy — their stability type is unaffected by small nonlinear terms. Nonhyperbolic fixed points are the fragile ones.

The important **Hartman-Grobman theorem** states that the local phase portrait near a hyperbolic fixed point is "topologically equivalent" to the phase portrait of the linearization. Here **topologically equivalent** means there is a **homeomorphism** (a continuous deformation with a continuous inverse) that maps one local phase portrait onto the other, such that trajectories map onto trajectories and the sense of time is preserved.

A phase portrait is **structurally stable** if its topology cannot be changed by an arbitrarily small perturbation to the vector field. For instance, a saddle point is structurally stable, but a center is not — an arbitrarily small amount of damping converts the center to a spiral.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Coarse Classification of Fixed Points)</span></p>

If we are only interested in *stability* and not the detailed geometry of trajectories, we can classify fixed points coarsely:

**Robust cases:**
* **Repellers** (sources): both eigenvalues have positive real part.
* **Attractors** (sinks): both eigenvalues have negative real part.
* **Saddles**: one eigenvalue is positive and one is negative.

**Marginal cases:**
* **Centers**: both eigenvalues are pure imaginary.
* **Higher-order and non-isolated fixed points**: at least one eigenvalue is zero.

The marginal cases are those where at least one eigenvalue satisfies $\operatorname{Re}(\lambda) = 0$.

</div>

### Rabbits versus Sheep

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Lotka-Volterra Competition Model)</span></p>

Consider the classic **Lotka-Volterra model of competition** between rabbits ($x$) and sheep ($y$), where both compete for the same limited food supply (grass):

$$\dot{x} = x(3 - x - 2y), \qquad \dot{y} = y(2 - x - y),$$

with $x, y \geq 0$. Each species grows logistically in the absence of the other (carrying capacities 3 and 2 respectively). The interspecific competition terms $-2xy$ and $-xy$ reduce the growth rate; the effect is more severe for the rabbits.

**Fixed points:** $(0, 0)$, $(0, 2)$, $(3, 0)$, and $(1, 1)$.

The Jacobian is $A = \begin{pmatrix} 3 - 2x - 2y & -2x \\ -y & 2 - x - 2y \end{pmatrix}$.

* $(0, 0)$: $A = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}$, $\lambda = 3, 2$ — **unstable node**. Trajectories leave tangent to the slow eigendirection ($y$-axis).
* $(0, 2)$: $A = \begin{pmatrix} -1 & 0 \\ -2 & -2 \end{pmatrix}$, $\lambda = -1, -2$ — **stable node**. Trajectories approach along the slow eigendirection spanned by $(1, -2)$.
* $(3, 0)$: $A = \begin{pmatrix} -3 & -6 \\ 0 & -1 \end{pmatrix}$, $\lambda = -3, -1$ — **stable node**. Trajectories approach along the slow eigendirection spanned by $(3, -1)$.
* $(1, 1)$: $A = \begin{pmatrix} -1 & -2 \\ -1 & -1 \end{pmatrix}$, $\tau = -2$, $\Delta = -1$, $\lambda = -1 \pm \sqrt{2}$ — **saddle point**.

The $x$ and $y$ axes contain straight-line trajectories (since $\dot{x} = 0$ when $x = 0$, and $\dot{y} = 0$ when $y = 0$). The stable manifold of the saddle at $(1, 1)$ acts as a **separatrix**, dividing the first quadrant into two basins of attraction.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Competitive Exclusion and Basins of Attraction)</span></p>

The phase portrait shows that one species generally drives the other to extinction. Trajectories starting below the stable manifold lead to eventual extinction of the sheep ($y \to 0$, rabbits win); those starting above lead to extinction of the rabbits ($x \to 0$, sheep win). This dichotomy illustrates the **principle of competitive exclusion**: two species competing for the same limited resource typically cannot coexist.

Given an attracting fixed point $\mathbf{x}^*$, its **basin of attraction** is the set of initial conditions $\mathbf{x}_0$ such that $\mathbf{x}(t) \to \mathbf{x}^*$ as $t \to \infty$. The stable manifold of the saddle serves as the **basin boundary**. The two trajectories that comprise the stable manifold are traditionally called **separatrices**, and they are important because they partition the phase space into regions of different long-term behavior.

</div>

### Conservative Systems

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Conservative System and Conserved Quantity)</span></p>

Newton's law $F = ma$ generates second-order systems of the form $m\ddot{x} = F(x)$, where $F$ is independent of $\dot{x}$ and $t$ (no damping, no friction, no time-dependent driving). Let $V(x)$ denote the **potential energy**, defined by $F(x) = -dV/dx$. Then

$$m\ddot{x} + \frac{dV}{dx} = 0.$$

Multiplying by $\dot{x}$ and noting this is an exact time-derivative, the total **energy**

$$E = \tfrac{1}{2}m\dot{x}^2 + V(x)$$

is constant on trajectories.

More generally, given a system $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$, a **conserved quantity** is a real-valued continuous function $E(\mathbf{x})$ that is constant on trajectories (i.e., $dE/dt = 0$) and is nonconstant on every open set. Systems for which a conserved quantity exists are called **conservative systems**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(No Attracting Fixed Points in Conservative Systems)</span></p>

A conservative system cannot have any attracting fixed points. *Proof:* Suppose $\mathbf{x}^*$ were attracting. Then all points in its basin of attraction would have the same energy $E(\mathbf{x}^*)$, making $E$ constant on an open set — contradicting the definition of a conserved quantity.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Double-Well Potential)</span></p>

Consider a particle of mass $m = 1$ moving in a double-well potential $V(x) = -\tfrac{1}{2}x^2 + \tfrac{1}{4}x^4$. The force is $-dV/dx = x - x^3$, so the equation of motion becomes

$$\dot{x} = y, \qquad \dot{y} = x - x^3.$$

The fixed points are $(0, 0)$ and $(\pm 1, 0)$. The Jacobian is $A = \begin{pmatrix} 0 & 1 \\ 1 - 3x^2 & 0 \end{pmatrix}$.

* At $(0, 0)$: $\Delta = -1 < 0$ — a **saddle point**.
* At $(\pm 1, 0)$: $\tau = 0$, $\Delta = 2 > 0$ — predicted to be **centers**.

Normally we'd worry about the linearization incorrectly predicting centers, but here the system is conservative with $E = \tfrac{1}{2}y^2 - \tfrac{1}{2}x^2 + \tfrac{1}{4}x^4$. The trajectories are closed curves of constant energy, so the centers are genuine.

Each center is surrounded by a family of small closed orbits (small oscillations in one well). There are also large orbits encircling all three fixed points (energetic oscillations over the hump). The two special trajectories that start and end at the saddle point are called **homoclinic orbits**. A homoclinic orbit does *not* correspond to a periodic solution, because the trajectory takes forever to reach the fixed point.

</div>

#### Nonlinear Centers

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Nonlinear Centers for Conservative Systems)</span></p>

Consider $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$, where $\mathbf{x} = (x, y) \in \mathbb{R}^2$ and $\mathbf{f}$ is continuously differentiable. Suppose there exists a conserved quantity $E(\mathbf{x})$ and that $\mathbf{x}^*$ is an isolated fixed point. If $\mathbf{x}^*$ is a local minimum of $E$, then all trajectories sufficiently close to $\mathbf{x}^*$ are closed.

*Idea of proof:* Since $E$ is constant on trajectories, each trajectory is contained in some contour of $E$. Near a local minimum (or maximum), the contours are closed curves. Because $\mathbf{x}^*$ is an *isolated* fixed point, there cannot be any fixed points on contours sufficiently close to $\mathbf{x}^*$. Hence all nearby trajectories are closed orbits, and $\mathbf{x}^*$ is a center.

*Remarks:* The theorem is also valid for local maxima of $E$ (just replace $E$ by $-E$). We also need $\mathbf{x}^*$ to be isolated — otherwise there could be fixed points on the energy contour.

</div>

### Reversible Systems

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reversible System)</span></p>

Many mechanical systems have **time-reversal symmetry**: their dynamics look the same whether time runs forward or backward. Any mechanical system of the form $m\ddot{x} = F(x)$ is symmetric under time reversal. The equivalent system is

$$\dot{x} = y, \qquad \dot{y} = \tfrac{1}{m}F(x).$$

Under the change of variables $t \to -t$ and $y \to -y$, both equations stay the same. Hence if $(x(t), y(t))$ is a solution, then so is $(x(-t), -y(-t))$. Every trajectory has a twin: they differ only by time-reversal and a reflection in the $x$-axis.

More generally, a **reversible system** is *any* second-order system that is invariant under $t \to -t$ and $y \to -y$. For example, any system of the form

$$\dot{x} = f(x, y), \qquad \dot{y} = g(x, y),$$

where $f$ is *odd* in $y$ and $g$ is *even* in $y$ (i.e., $f(x, -y) = -f(x, y)$ and $g(x, -y) = g(x, y)$) is reversible.

</div>

Reversible systems are different from conservative systems, but they share many of the same properties. In particular, centers are robust in reversible systems as well.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Nonlinear Centers for Reversible Systems)</span></p>

Suppose the origin $\mathbf{x}^* = \mathbf{0}$ is a linear center for the continuously differentiable system

$$\dot{x} = f(x, y), \qquad \dot{y} = g(x, y),$$

and suppose that the system is reversible. Then sufficiently close to the origin, all trajectories are closed curves.

*Idea of proof:* Consider a trajectory starting on the positive $x$-axis near the origin. Sufficiently near the origin, the flow swirls around the origin thanks to the dominant influence of the linear center, and the trajectory eventually intersects the negative $x$-axis. By reversibility, reflecting the trajectory across the $x$-axis and changing the sign of $t$ gives a twin trajectory with the same endpoints but with its arrow reversed. Together the two trajectories form a closed orbit. Hence all trajectories sufficiently close to the origin are closed.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Nonlinear Center via Reversibility)</span></p>

The system $\dot{x} = y - y^3$, $\dot{y} = -x - y^2$ has a nonlinear center at the origin. The Jacobian at the origin is

$$A = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix},$$

which has $\tau = 0$, $\Delta > 0$, so the origin is a linear center. Furthermore, the system is invariant under $t \to -t$, $y \to -y$, so it is reversible. By Theorem 6.6.1, the origin is a nonlinear center.

The other fixed points $(-1, 1)$ and $(-1, -1)$ are saddle points. The trajectories above the $x$-axis have twins below with arrows reversed. The twin saddle points are joined by **heteroclinic trajectories** (also called **saddle connections**) — trajectories connecting two different saddle points. Like homoclinic orbits, heteroclinic trajectories are much more common in reversible or conservative systems than in other types of systems.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Homoclinic Orbit via Reversibility)</span></p>

The system $\dot{x} = y$, $\dot{y} = x - x^2$ has a homoclinic orbit in the half-plane $x \ge 0$. The unstable manifold of the saddle at the origin leaves along the vector $(1, 1)$, enters the first quadrant, and eventually reaches $y = 0$ at $x = 1$. By reversibility, the reflected trajectory completes the homoclinic orbit.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(General Definition of Reversibility)</span></p>

There is a more general definition of reversibility that extends to higher-order systems. Consider any mapping $R(\mathbf{x})$ of the phase space to itself that satisfies $R^2(\mathbf{x}) = \mathbf{x}$ (applying the mapping twice returns all points to where they started). Then the system $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$ is **reversible** if it is invariant under the change of variables $t \to -t$, $\mathbf{x} \to R(\mathbf{x})$. In our two-dimensional examples, a reflection about the $x$-axis (or any axis through the origin) has this property.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Reversible but Not Conservative)</span></p>

The system

$$\dot{x} = -2\cos x - \cos y, \qquad \dot{y} = -2\cos y - \cos x$$

is reversible but *not* conservative. It is invariant under $t \to -t$, $x \to -x$, $y \to -y$, so $R(x, y) = (-x, -y)$. To show it is not conservative, note that it has an attracting fixed point (a conservative system can never have an attracting fixed point). The fixed points satisfy $\cos x^* = \cos y^*$, yielding $(x^*, y^*) = (\pm \pi/2, \pm \pi/2)$. At $(-\pi/2, -\pi/2)$ the Jacobian is

$$A = \begin{pmatrix} -2 & -1 \\ -1 & -2 \end{pmatrix},$$

which has $\tau = -4$, $\Delta = 3$, $\tau^2 - 4\Delta = 4$. This is a stable node, so the system is not conservative.

The stable node at $(-\pi/2, -\pi/2)$ is the twin of the unstable node at $(\pi/2, \pi/2)$ under the reversibility symmetry.

</div>

### The Pendulum

In the absence of damping and external driving, the motion of a pendulum is governed by

$$\frac{d^2\theta}{dt^2} + \frac{g}{L}\sin\theta = 0,$$

where $\theta$ is the angle from the downward vertical, $g$ is the acceleration due to gravity, and $L$ is the length of the pendulum. Nondimensionalizing by introducing $\omega = \sqrt{g/L}$ and $\tau = \omega t$, the equation becomes

$$\ddot{\theta} + \sin\theta = 0,$$

where the overdot now denotes differentiation with respect to $\tau$. The corresponding system in the phase plane is

$$\dot{\theta} = v, \qquad \dot{v} = -\sin\theta.$$

The fixed points are $(\theta^*, v^*) = (k\pi, 0)$ for any integer $k$. Since angles differing by $2\pi$ are physically identical, we focus on $(0, 0)$ and $(\pi, 0)$.

**At $(0, 0)$:** The Jacobian is $A = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$, so the origin is a linear center. In fact, the origin is a *nonlinear* center for two reasons:

1. The system is **reversible** (invariant under $\tau \to -\tau$, $v \to -v$), so Theorem 6.6.1 applies.
2. The system is **conservative** with energy $E(\theta, v) = \tfrac{1}{2}v^2 - \cos\theta$, which has a local minimum at the origin, so Theorem 6.5.1 applies.

**At $(\pi, 0)$:** The Jacobian is $A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, giving $\lambda^2 - 1 = 0$, so $\lambda_1 = -1$, $\lambda_2 = 1$. This is a **saddle** with eigenvectors $\mathbf{v}_1 = (1, -1)$ and $\mathbf{v}_2 = (1, 1)$.

Including the energy contours $E = \tfrac{1}{2}v^2 - \cos\theta$ for different values of $E$, the phase portrait is periodic in the $\theta$-direction.

**Physical interpretation:** The center corresponds to the pendulum at rest, hanging straight down — the lowest energy state ($E = -1$). Small closed orbits around the center represent **librations** (small oscillations). As $E$ increases, the orbits grow. The critical case $E = 1$ corresponds to the heteroclinic trajectories joining the saddles — these represent an inverted pendulum at rest. For $E > 1$, the pendulum whirls repeatedly over the top; these **rotations** are periodic solutions since $\theta = -\pi$ and $\theta = +\pi$ are the same physical position.

#### Cylindrical Phase Space

The phase portrait is more naturally viewed on the surface of a cylinder, since $\theta$ is an angle while $v$ is a real number. On the cylinder:

- The periodic whirling motions are closed orbits encircling the cylinder (for $E > 1$).
- All the saddle points in the planar portrait become the same physical state (inverted pendulum at rest).
- The heteroclinic trajectories of the planar picture become **homoclinic orbits** on the cylinder.

There is an obvious symmetry between the top and bottom halves of the cylinder: both homoclinic orbits have the same energy and shape. Plotting energy $E$ vertically instead of $v$ bends the cylinder into a **U-tube**, where the two arms correspond to clockwise and counterclockwise whirling. At low energies (librations), this distinction vanishes. The homoclinic orbits lie at $E = 1$, the borderline between rotations and librations.

#### Damping

Adding linear damping to the pendulum gives

$$\ddot{\theta} + b\dot{\theta} + \sin\theta = 0,$$

where $b > 0$ is the damping strength. Then centers become **stable spirals** while saddles remain saddles. On the U-tube, *all trajectories continually lose altitude* except at the fixed points. This can be verified by computing the change in energy along a trajectory:

$$\frac{dE}{d\tau} = \frac{d}{d\tau}\left(\tfrac{1}{2}\dot{\theta}^2 - \cos\theta\right) = \dot{\theta}(\ddot{\theta} + \sin\theta) = -b\dot{\theta}^2 \le 0.$$

Hence $E$ decreases monotonically along trajectories, except at fixed points where $\dot{\theta} \equiv 0$. Physically, a pendulum initially whirling clockwise loses energy, spirals down the arm of the U-tube until $E < 1$, then settles into small oscillations (librations) about the stable equilibrium.

### Index Theory

Linearization is a *local* method: it gives a detailed microscopic view near a fixed point, but says nothing about global behavior. **Index theory** provides *global* information about the phase portrait, addressing questions such as: Must a closed trajectory always encircle a fixed point? What types of fixed points are permitted? What types can coalesce in bifurcations?

#### The Index of a Closed Curve

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Index of a Closed Curve)</span></p>

Suppose $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$ is a smooth vector field on the phase plane. Let $C$ be a simple closed curve (no self-intersections) that does not pass through any fixed points. At each point $\mathbf{x}$ on $C$, the vector field $\dot{\mathbf{x}} = (\dot{x}, \dot{y})$ makes a well-defined angle $\phi = \tan^{-1}(\dot{y}/\dot{x})$ with the positive $x$-axis.

As $\mathbf{x}$ moves counterclockwise around $C$, the angle $\phi$ changes continuously. Over one full circuit, $\phi$ changes by an integer multiple of $2\pi$. The **index of the closed curve $C$** with respect to the vector field $\mathbf{f}$ is

$$I_C = \frac{1}{2\pi}[\phi]_C,$$

where $[\phi]_C$ denotes the net change in $\phi$ over one circuit. Thus $I_C$ is the net number of counterclockwise revolutions made by the vector field as $\mathbf{x}$ moves once counterclockwise around $C$.

</div>

To compute the index, we do not need to know the vector field everywhere — only along $C$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Index Computation from Pictures)</span></p>

If the vectors along $C$ rotate once counterclockwise as $\mathbf{x}$ traverses $C$ counterclockwise, then $I_C = +1$. If they rotate once clockwise, then $I_C = -1$.

A useful method: number the vectors in counterclockwise order along $C$, then translate them (without rotation) so all tails lie at a common origin. The index equals the net number of counterclockwise revolutions made by the translated vectors.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Index of the Unit Circle)</span></p>

For the vector field $\dot{x} = x^2 y$, $\dot{y} = x^2 - y^2$, with $C$ the unit circle $x^2 + y^2 = 1$: computing the vector field at several points around $C$ (e.g., at $(1, 0)$, $(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}})$, $(0, 1)$, etc.) and translating the vectors shows that they rotate $180°$ clockwise, then $360°$ counterclockwise, then $180°$ clockwise again. The net change is $[\phi]_C = -\pi + 2\pi - \pi = 0$, so $I_C = 0$.

</div>

#### Properties of the Index

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Properties of the Index)</span></p>

1. **Deformation invariance:** If $C$ can be continuously deformed into $C'$ without passing through a fixed point, then $I_C = I_{C'}$. *(Proof: As $C$ deforms, $I_C$ varies continuously. But $I_C$ is an integer, so it must be constant.)*

2. **Empty curves:** If $C$ does not enclose any fixed points, then $I_C = 0$. *(Proof: By property (1), shrink $C$ to a tiny circle. Then $\phi$ is essentially constant, so $[\phi]_C = 0$.)*

3. **Time-reversal invariance:** Reversing all arrows in the vector field ($t \to -t$) does not change the index. *(Proof: All angles change from $\phi$ to $\phi + \pi$, so $[\phi]_C$ stays the same.)*

4. **Closed orbits:** If $C$ is a trajectory (i.e., a closed orbit), then $I_C = +1$. *(The vector field is everywhere tangent to $C$, so the tangent vector rotates once as $\mathbf{x}$ winds around $C$.)*

</div>

#### Index of a Point

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Index of a Fixed Point)</span></p>

Suppose $\mathbf{x}^*$ is an isolated fixed point. The **index** $I$ of $\mathbf{x}^*$ is defined as $I_C$, where $C$ is *any* closed curve that encloses $\mathbf{x}^*$ and no other fixed points. By property (1) above, $I_C$ is independent of $C$ and is therefore a property of $\mathbf{x}^*$ alone.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Indices of Standard Fixed Points)</span></p>

* **Stable node:** $I = +1$ (the vector field near a stable node looks like Example 6.8.1).
* **Unstable node:** $I = +1$ (all arrows are reversed compared to a stable node, but by property (3) this doesn't change the index. The index is *not related to stability*.)
* **Saddle point:** $I = -1$. A saddle point is truly a different animal from all the other familiar types of isolated fixed points (spirals, centers, degenerate nodes, and stars all have $I = +1$).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Index Sum for Closed Curves)</span></p>

If a closed curve $C$ surrounds $n$ isolated fixed points $\mathbf{x}_1^*, \dots, \mathbf{x}_n^*$, then

$$I_C = I_1 + I_2 + \cdots + I_n,$$

where $I_k$ is the index of $\mathbf{x}_k^*$ for $k = 1, \dots, n$.

*Idea of proof:* Deform $C$ (without crossing fixed points) into a new closed curve $\Gamma$ consisting of $n$ small circles $\gamma_1, \dots, \gamma_n$ around the fixed points, connected by two-way bridges. Since $I_\Gamma = I_C$ by property (1), and the bridge contributions cancel (each bridge is traversed once in each direction), we get $I_\Gamma = \sum_{k=1}^n I_k$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Index Requirement for Closed Orbits)</span></p>

Any closed orbit in the phase plane must enclose fixed points whose indices sum to $+1$.

*Proof:* Let $C$ denote the closed orbit. From property (4), $I_C = +1$. Then Theorem 6.8.1 implies $\sum_{k=1}^n I_k = +1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Consequences of the Index Requirement)</span></p>

Theorem 6.8.2 has several practical consequences:

- There is always at least one fixed point inside any closed orbit.
- If there is *only one* fixed point inside, it cannot be a saddle (since saddles have $I = -1 \neq +1$).
- The theorem can sometimes be used to **rule out** the existence of closed orbits entirely.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(No Closed Orbits in Rabbits vs. Sheep)</span></p>

Consider the "rabbit vs. sheep" system $\dot{x} = x(3 - x - 2y)$, $\dot{y} = y(2 - x - y)$ for $x, y \ge 0$. It has four fixed points: $(0, 0)$ is an unstable node ($I = +1$); $(0, 2)$ and $(3, 0)$ are stable nodes ($I = +1$ each); and $(1, 1)$ is a saddle ($I = -1$).

Now suppose the system had a closed orbit. There are three qualitatively different types of curves $C_1, C_2, C_3$ to consider:

- Orbits like $C_1$ that don't enclose any fixed points: impossible (property 2 requires $I_C = +1$, but $I_C = 0$).
- Orbits like $C_2$ enclosing fixed points with indices not summing to $+1$: impossible.
- Orbits like $C_3$ that would satisfy the index requirement but must cross the $x$-axis or $y$-axis, which contain straight-line trajectories. This violates the rule that trajectories can't cross.

Hence no closed orbits exist for this system.

</div>

## Chapter 7: Limit Cycles

### Introduction

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Limit Cycle)</span></p>

A **limit cycle** is an isolated closed trajectory. *Isolated* means that neighboring trajectories are not closed; they spiral either toward or away from the limit cycle.

- If all neighboring trajectories approach the limit cycle, it is **stable** (or *attracting*).
- If all neighboring trajectories flee from it, it is **unstable**.
- In exceptional cases, it may be **half-stable** (attracting on one side, repelling on the other).

</div>

Limit cycles are inherently nonlinear phenomena — they cannot occur in linear systems. A linear system $\dot{\mathbf{x}} = A\mathbf{x}$ can have closed orbits, but they won't be *isolated*: if $\mathbf{x}(t)$ is periodic, then so is $c\mathbf{x}(t)$ for any constant $c \neq 0$, giving a one-parameter family of closed orbits. In contrast, limit cycle oscillations are determined by the structure of the system itself, not by initial conditions.

Stable limit cycles are very important scientifically — they model self-sustained oscillations in systems that oscillate even in the absence of external periodic forcing. Examples include: the beating of a heart, the periodic firing of a pacemaker neuron, daily rhythms in body temperature, chemical reactions that oscillate spontaneously, and self-excited vibrations in bridges and airplane wings.

### Examples

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(A Simple Limit Cycle)</span></p>

Consider the system in polar coordinates:

$$\dot{r} = r(1 - r^2), \qquad \dot{\theta} = 1,$$

where $r \ge 0$. The radial and angular dynamics are uncoupled. Treating $\dot{r} = r(1 - r^2)$ as a vector field on the line, $r^* = 0$ is an unstable fixed point and $r^* = 1$ is stable. Hence all trajectories (except $r^* = 0$) approach the unit circle $r = 1$ monotonically. The motion in the $\theta$-direction is simply uniform rotation. So all trajectories spiral asymptotically toward a limit cycle at $r = 1$.

The limit cycle solution is $x(t) = \cos(t + \theta_0)$ — a sinusoidal oscillation of constant amplitude, corresponding to the standard circle in the phase plane.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Van der Pol Oscillator)</span></p>

The **van der Pol equation** is

$$\ddot{x} + \mu(x^2 - 1)\dot{x} + x = 0,$$

where $\mu \ge 0$ is a parameter. It looks like a simple harmonic oscillator with a **nonlinear damping** term $\mu(x^2 - 1)\dot{x}$. This term acts like ordinary positive damping for $\lvert x \rvert > 1$, but like *negative* damping for $\lvert x \rvert < 1$. Large-amplitude oscillations decay, but small ones are pumped back up.

The van der Pol equation has a unique, stable limit cycle for each $\mu > 0$. Unlike Example 7.1.1, the limit cycle is not a circle and the stable waveform is not a sine wave. This result follows from Lienard's theorem (Section 7.4).

</div>

### Ruling Out Closed Orbits

#### Gradient Systems

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gradient System)</span></p>

A system that can be written in the form $\dot{\mathbf{x}} = -\nabla V$ for some continuously differentiable, single-valued scalar function $V(\mathbf{x})$ is called a **gradient system** with **potential function** $V$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(No Closed Orbits in Gradient Systems)</span></p>

Closed orbits are impossible in gradient systems.

*Proof:* Suppose there were a closed orbit of period $T$. Then $\Delta V = 0$ after one circuit since $V$ is single-valued. But on the other hand,

$$\Delta V = \int_0^T \frac{dV}{dt}\,dt = \int_0^T (\nabla V \cdot \dot{\mathbf{x}})\,dt = -\int_0^T \lVert \dot{\mathbf{x}} \rVert^2\,dt < 0$$

(unless $\dot{\mathbf{x}} \equiv \mathbf{0}$, i.e., the trajectory is a fixed point). This contradiction shows that closed orbits cannot exist.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gradient System Check)</span></p>

The system $\dot{x} = \sin y$, $\dot{y} = x\cos y$ is a gradient system with $V(x,y) = -x\sin y$, since $\dot{x} = -\partial V/\partial x$ and $\dot{y} = -\partial V/\partial y$. Hence there are no closed orbits.

</div>

#### Liapunov Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Liapunov Function)</span></p>

For a system $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$ with a fixed point at $\mathbf{x}^*$, a **Liapunov function** is a continuously differentiable, real-valued function $V(\mathbf{x})$ with the properties:

1. $V(\mathbf{x}) > 0$ for all $\mathbf{x} \neq \mathbf{x}^*$, and $V(\mathbf{x}^*) = 0$ (i.e., $V$ is **positive definite**).
2. $\dot{V} < 0$ for all $\mathbf{x} \neq \mathbf{x}^*$ (all trajectories flow "downhill" toward $\mathbf{x}^*$).

If such a function exists, then $\mathbf{x}^*$ is globally asymptotically stable: $\mathbf{x}(t) \to \mathbf{x}^*$ as $t \to \infty$ for all initial conditions. In particular, the system has no closed orbits.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Nonlinearly Damped Oscillator)</span></p>

The nonlinearly damped oscillator $\ddot{x} + (\dot{x})^3 + x = 0$ has no periodic solutions. Consider the energy function $E(x, \dot{x}) = \tfrac{1}{2}(x^2 + \dot{x}^2)$. Then $\dot{E} = \dot{x}(x + \ddot{x}) = \dot{x}(-\dot{x}^3) = -\dot{x}^4 \le 0$, with equality only when $\dot{x} \equiv 0$ (which would make the trajectory a fixed point). Hence $\Delta E < 0$ around any putative closed orbit, contradicting $\Delta E = 0$. So there are no periodic solutions.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Liapunov Function by Construction)</span></p>

The system $\dot{x} = -x + 4y$, $\dot{y} = -x - y^3$ has no closed orbits. Consider $V(x, y) = x^2 + ay^2$ where $a$ is a parameter. Then $\dot{V} = 2x\dot{x} + 2ay\dot{y} = 2x(-x + 4y) + 2ay(-x - y^3) = -2x^2 + (8 - 2a)xy - 2ay^4$. Choosing $a = 4$ eliminates the $xy$ term: $\dot{V} = -2x^2 - 8y^4$. Since $V > 0$ and $\dot{V} < 0$ for all $(x, y) \neq (0, 0)$, $V = x^2 + 4y^2$ is a Liapunov function. There are no closed orbits, and all trajectories approach the origin.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

There is no systematic way to construct Liapunov functions. Divine inspiration is usually required, although sometimes one can work backwards. Sums of squares occasionally work.

</div>

#### Dulac's Criterion

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Dulac's Criterion)</span></p>

Let $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$ be a continuously differentiable vector field defined on a simply connected subset $R$ of the plane. If there exists a continuously differentiable, real-valued function $g(\mathbf{x})$ such that $\nabla \cdot (g\dot{\mathbf{x}})$ has one sign throughout $R$, then there are no closed orbits lying entirely in $R$.

*Proof:* Suppose there were a closed orbit $C$ in $R$. Let $A$ be the region inside $C$. By Green's theorem,

$$\iint_A \nabla \cdot (g\dot{\mathbf{x}})\,dA = \oint_C g\dot{\mathbf{x}} \cdot \mathbf{n}\,d\ell.$$

The left side is nonzero since $\nabla \cdot (g\dot{\mathbf{x}})$ has one sign in $R$. The right side equals zero because $C$ is a trajectory, so $\dot{\mathbf{x}}$ is tangent to $C$ and hence $\dot{\mathbf{x}} \cdot \mathbf{n} = 0$ everywhere. This contradiction implies no such $C$ can exist.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Dulac's Criterion Applied)</span></p>

The system $\dot{x} = x(2 - x - y)$, $\dot{y} = y(4x - x^2 - 3)$ has no closed orbits in the positive quadrant $x, y > 0$. Choosing $g = 1/xy$ gives

$$\nabla \cdot (g\dot{\mathbf{x}}) = \frac{\partial}{\partial x}\left(\frac{2 - x - y}{y}\right) + \frac{\partial}{\partial y}\left(\frac{4x - x^2 - 3}{x}\right) = -\frac{1}{y} < 0.$$

Since $g$ and $\mathbf{f}$ satisfy the smoothness conditions and $\nabla \cdot (g\dot{\mathbf{x}}) < 0$ throughout the simply connected region $x, y > 0$, Dulac's criterion rules out closed orbits.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Like Liapunov's method, Dulac's criterion has no algorithm for finding $g(\mathbf{x})$. Candidates that occasionally work are $g = 1$, $1/x^a y^b$, $e^{ax}$, and $e^{ay}$.

</div>

### Poincare–Bendixson Theorem

Having discussed how to rule out closed orbits, we now turn to proving their existence.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Poincare–Bendixson Theorem)</span></p>

Suppose that:

1. $R$ is a closed, bounded subset of the plane;
2. $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$ is a continuously differentiable vector field on an open set containing $R$;
3. $R$ does not contain any fixed points; and
4. There exists a trajectory $C$ that is "confined" in $R$ (it starts in $R$ and stays in $R$ for all future time).

Then either $C$ is a closed orbit, or $C$ spirals toward a closed orbit as $t \to \infty$. In either case, $R$ contains a closed orbit.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Trapping Regions)</span></p>

The standard trick to satisfy condition (4) is to construct a **trapping region** $R$: a closed connected set such that the vector field points "inward" everywhere on the boundary. Then *all* trajectories in $R$ are confined, and if $R$ is free of fixed points, the Poincare–Bendixson theorem guarantees a closed orbit inside $R$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Trapping Region in Polar Coordinates)</span></p>

Consider the system $\dot{r} = r(1 - r^2) + \mu r\cos\theta$, $\dot{\theta} = 1$. When $\mu = 0$, there is a stable limit cycle at $r = 1$. For $\mu > 0$ small, a closed orbit still exists. We seek concentric circles $r_{\min}$ and $r_{\max}$ such that $\dot{r} < 0$ on the outer circle and $\dot{r} > 0$ on the inner circle.

For $r_{\min}$: we require $r(1 - r^2) + \mu r\cos\theta > 0$ for all $\theta$. Since $\cos\theta \ge -1$, it suffices to have $1 - r^2 - \mu > 0$, giving $r_{\min} < \sqrt{1 - \mu}$ (valid for $\mu < 1$).

For $r_{\max}$: we require $r(1 - r^2) + \mu r\cos\theta < 0$ for all $\theta$. Since $\cos\theta \le 1$, it suffices to have $1 - r^2 + \mu < 0$, giving $r_{\max} > \sqrt{1 + \mu}$.

The annulus $r_{\min} \le r \le r_{\max}$ is a trapping region with no fixed points (since $\dot{\theta} = 1 > 0$), so by the Poincare–Bendixson theorem a closed orbit exists for all $\mu < 1$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Glycolytic Oscillator)</span></p>

A model of glycolysis (Sel'kov 1968) is given by

$$\dot{x} = -x + ay + x^2 y, \qquad \dot{y} = b - ay - x^2 y,$$

where $x, y$ are concentrations of ADP and F6P, and $a, b > 0$ are kinetic parameters. The nullclines are $y = x/(a + x^2)$ (for $\dot{x} = 0$) and $y = b/(a + x^2)$ (for $\dot{y} = 0$).

A trapping region can be constructed from the nullclines and a diagonal line of slope $-1$ extending from the point $(b, b/a)$. For large $x$, the vector field along the diagonal satisfies $\dot{x} - (-\dot{y}) = b - x < 0$ when $x > b$, so the flow points inward.

There is a single fixed point at $x^* = b$, $y^* = b/(a + b^2)$. The Jacobian is

$$A = \begin{pmatrix} -1 + 2xy & a + x^2 \\ -2xy & -(a + x^2) \end{pmatrix}.$$

At the fixed point, $\Delta = a + b^2 > 0$ and

$$\tau = -\frac{b^4 + (2a - 1)b^2 + (a + a^2)}{a + b^2}.$$

The fixed point is unstable when $\tau > 0$ and stable when $\tau < 0$. The boundary $\tau = 0$ defines a curve $b^2 = \tfrac{1}{2}(1 - 2a \pm \sqrt{1 - 8a})$ in $(a, b)$ parameter space. When the fixed point is unstable ($\tau > 0$), the repeller drives neighboring trajectories into the trapping region, which is free of fixed points (after puncturing out the repeller), and the Poincare–Bendixson theorem guarantees a stable limit cycle.

</div>

#### No Chaos in the Phase Plane

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(No Chaos in Two Dimensions)</span></p>

The Poincare–Bendixson theorem is one of the central results of nonlinear dynamics. It says that in the phase plane, the dynamical possibilities are very limited: if a trajectory is confined to a closed, bounded region containing no fixed points, it must eventually approach a closed orbit. Nothing more complicated is possible.

This depends crucially on the two-dimensionality of the plane. In higher-dimensional systems ($n \ge 3$), the theorem no longer applies, and trajectories may wander around forever in a bounded region without settling to a fixed point or a closed orbit. Such trajectories are attracted to a **strange attractor** — a fractal set on which the motion is aperiodic and sensitive to initial conditions. This is **chaos**, and it can never occur in the phase plane.

</div>

### Lienard Systems

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lienard Equation)</span></p>

**Lienard's equation** is

$$\ddot{x} + f(x)\dot{x} + g(x) = 0,$$

a generalization of the van der Pol oscillator $\ddot{x} + \mu(x^2 - 1)\dot{x} + x = 0$. It can be interpreted mechanically as a unit mass subject to a nonlinear damping force $-f(x)\dot{x}$ and a nonlinear restoring force $-g(x)$. The equivalent system is

$$\dot{x} = y, \qquad \dot{y} = -g(x) - f(x)y.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Lienard's Theorem)</span></p>

Suppose that $f(x)$ and $g(x)$ satisfy the following conditions:

1. $f(x)$ and $g(x)$ are continuously differentiable for all $x$;
2. $g(-x) = -g(x)$ for all $x$ (i.e., $g$ is an **odd** function);
3. $g(x) > 0$ for $x > 0$;
4. $f(-x) = f(x)$ for all $x$ (i.e., $f$ is an **even** function);
5. The odd function $F(x) = \int_0^x f(u)\,du$ has exactly one positive zero at $x = a$, is negative for $0 < x < a$, is positive and nondecreasing for $x > a$, and $F(x) \to \infty$ as $x \to \infty$.

Then the system has a unique, stable limit cycle surrounding the origin in the phase plane.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The assumptions on $g(x)$ mean that the restoring force acts like an ordinary spring (tending to reduce displacement). The assumptions on $f(x)$ imply that damping is negative at small $\lvert x \rvert$ and positive at large $\lvert x \rvert$. Small oscillations are pumped up and large oscillations are damped down, so the system settles into a self-sustained oscillation of intermediate amplitude.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Van der Pol Has a Unique Stable Limit Cycle)</span></p>

For the van der Pol equation $\ddot{x} + \mu(x^2 - 1)\dot{x} + x = 0$, we have $f(x) = \mu(x^2 - 1)$ and $g(x) = x$. Conditions (1)–(4) are clearly satisfied. For condition (5):

$$F(x) = \mu\left(\tfrac{1}{3}x^3 - x\right) = \tfrac{1}{3}\mu x(x^2 - 3).$$

This has its unique positive zero at $x = a = \sqrt{3}$. It is negative for $0 < x < \sqrt{3}$, positive and nondecreasing for $x > \sqrt{3}$, and $F(x) \to \infty$ as $x \to \infty$. Hence the van der Pol equation has a unique, stable limit cycle.

</div>

### Relaxation Oscillations

For the van der Pol equation with $\mu \gg 1$ (the *strongly nonlinear* limit), the limit cycle consists of an extremely slow buildup followed by a sudden discharge, repeated periodically. These are called **relaxation oscillations** because the "stress" accumulated during the slow buildup is "relaxed" during the sudden discharge.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Phase Plane Analysis of van der Pol for $\mu \gg 1$)</span></p>

Introducing the Lienard transformation $w = \dot{x} + \mu(\tfrac{1}{3}x^3 - x)$ and $F(x) = \tfrac{1}{3}x^3 - x$, the van der Pol equation becomes

$$\dot{x} = \mu[y - F(x)], \qquad \dot{y} = -\frac{1}{\mu}x,$$

where $y = w/\mu$. The nullclines are key: the **cubic nullcline** $y = F(x) = \tfrac{1}{3}x^3 - x$ (where $\dot{x} = 0$) and the $y$-axis (where $\dot{y} = 0$).

Away from the cubic nullcline, $y - F(x) \sim O(1)$, so $\lvert \dot{x} \rvert \sim O(\mu) \gg 1$ while $\lvert \dot{y} \rvert \sim O(\mu^{-1}) \ll 1$. Hence trajectories move nearly horizontally ("fast") until they hit the cubic nullcline. On the nullcline, $y \approx F(x)$ and the trajectory "crawls" slowly along it. At the knee of the cubic (a local extremum), the trajectory jumps horizontally to the other branch. The limit cycle thus has two **slow** branches (crawling along the cubic) and two **fast** jumps.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Period of Relaxation Oscillation)</span></p>

The period $T$ of the van der Pol limit cycle for $\mu \gg 1$ is essentially the time spent on the two slow branches (the fast jumps contribute negligibly). By symmetry, both branches contribute equally. On a slow branch, $y \approx F(x)$, so $\dot{y} \approx F'(x)\dot{x} = (x^2 - 1)\dot{x}$. Since $\dot{y} = -x/\mu$, we get

$$dt \approx -\frac{\mu(x^2 - 1)}{x}\,dx.$$

The positive slow branch goes from $x_A = 2$ to $x_B = 1$, giving

$$T \approx 2\int_2^1 \frac{-\mu(x^2 - 1)}{x}\,dx = 2\mu\left[\frac{x^2}{2} - \ln x\right]_1^2 = \mu(3 - 2\ln 2).$$

With more work, the period can be refined to $T \approx \mu[3 - 2\ln 2] + 2\alpha\mu^{-1/3} + \cdots$, where $\alpha \approx 2.338$ is the smallest root of $\text{Ai}(-\alpha) = 0$ (Airy function correction from the time to turn the corners).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Two Time Scales)</span></p>

The relaxation oscillation has two **widely separated time scales**: the crawls take $\Delta t \sim O(\mu)$ and the jumps take $\Delta t \sim O(\mu^{-1})$. The waveform $x(t)$ shows long plateaus interrupted by rapid transitions — a signature of relaxation oscillations. This structure arises in many scientific contexts, from stick-slip oscillations of a bowed violin string to the periodic firing of nerve cells.

</div>

### Weakly Nonlinear Oscillators

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Weakly Nonlinear Oscillator)</span></p>

Equations of the form

$$\ddot{x} + x + \varepsilon h(x, \dot{x}) = 0,$$

where $0 \le \varepsilon \ll 1$ and $h(x, \dot{x})$ is an arbitrary smooth function, represent small perturbations of the linear oscillator $\ddot{x} + x = 0$ and are called **weakly nonlinear oscillators**. Two fundamental examples are:

- The **van der Pol equation** (in the weak limit): $\ddot{x} + x + \varepsilon(x^2 - 1)\dot{x} = 0$.
- The **Duffing equation**: $\ddot{x} + x + \varepsilon x^3 = 0$.

For $\varepsilon = 0$, all solutions are $x(t) = r\cos(t + \phi)$ — circles in the $(x, \dot{x})$ phase plane. For small $\varepsilon > 0$, the amplitude $r$ and phase $\phi$ evolve slowly over many cycles. The trajectory is a slowly winding spiral that asymptotes to an approximately circular limit cycle.

</div>

#### Regular Perturbation Theory and Its Failure

As a first approach, we seek solutions as a power series in $\varepsilon$:

$$x(t, \varepsilon) = x_0(t) + \varepsilon x_1(t) + \varepsilon^2 x_2(t) + \cdots$$

This is called **regular perturbation theory**. Substituting into the equation and collecting powers of $\varepsilon$ yields a hierarchy of equations that can be solved sequentially.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Failure of Regular Perturbation Theory)</span></p>

Consider the weakly damped linear oscillator $\ddot{x} + 2\varepsilon\dot{x} + x = 0$ with $x(0) = 0$, $\dot{x}(0) = 1$. The exact solution is

$$x(t, \varepsilon) = (1 - \varepsilon^2)^{-1/2} e^{-\varepsilon t}\sin\left[(1 - \varepsilon^2)^{1/2}t\right].$$

Regular perturbation theory gives $x_0 = \sin t$ at $O(1)$. At $O(\varepsilon)$, the equation for $x_1$ is $\ddot{x}_1 + x_1 = -2\cos t$ — a **resonant** forcing term that produces a **secular term** $x_1(t) = -t\sin t$, which grows without bound. The perturbation series becomes

$$x(t, \varepsilon) = \sin t - \varepsilon t\sin t + O(\varepsilon^2),$$

which is valid only for $t \ll 1/\varepsilon$. For fixed $\varepsilon$, it breaks down as $t$ grows — precisely the regime we care about.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Regular Perturbation Theory Fails)</span></p>

The true solution has two time scales:

1. A **fast time** $t \sim O(1)$ for the sinusoidal oscillations.
2. A **slow time** $t \sim 1/\varepsilon$ over which the amplitude decays.

The secular term $-\varepsilon t \sin t$ is perturbation theory's misguided attempt to capture the slow exponential decay $e^{-\varepsilon t} = 1 - \varepsilon t + O(\varepsilon^2 t^2)$. Additionally, the true frequency $(1 - \varepsilon^2)^{1/2} \approx 1 - \tfrac{1}{2}\varepsilon^2$ is slightly shifted from $\omega = 1$, which after a very long time $t \sim O(1/\varepsilon^2)$ produces a significant cumulative phase error — a third, *super-slow* time scale.

</div>

#### Two-Timing

The method of **two-timing** (also called the method of multiple scales) builds the existence of two time scales into the approximation from the start.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Two-Timing Method)</span></p>

Let $\tau = t$ denote the fast $O(1)$ time and $T = \varepsilon t$ denote the slow time. Treat $\tau$ and $T$ as *independent* variables. Expand the solution as

$$x(t, \varepsilon) = x_0(\tau, T) + \varepsilon x_1(\tau, T) + O(\varepsilon^2).$$

By the chain rule, the time derivatives become

$$\dot{x} = \partial_\tau x + \varepsilon\,\partial_T x, \qquad \ddot{x} = \partial_{\tau\tau} x + \varepsilon(2\,\partial_{\tau T} x) + O(\varepsilon^2).$$

Substituting into the governing equation and collecting powers of $\varepsilon$ yields a hierarchy of equations. The key step is to **set the coefficients of resonant terms to zero** in the equation for $x_1$, which determines how the slowly-varying amplitude and phase evolve on the slow time scale $T$. This elimination of secular terms is the hallmark of all two-timing calculations.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Two-Timing the Damped Oscillator)</span></p>

For $\ddot{x} + 2\varepsilon\dot{x} + x = 0$ with $x(0) = 0$, $\dot{x}(0) = 1$:

**$O(1)$:** $\partial_{\tau\tau}x_0 + x_0 = 0$, so $x_0 = A(T)\sin\tau + B(T)\cos\tau$.

**$O(\varepsilon)$:** $\partial_{\tau\tau}x_1 + x_1 = -2(\partial_T x_0 + \partial_\tau x_0)$. Substituting $x_0$ and requiring no resonant forcing (no $\sin\tau$ or $\cos\tau$ terms on the right) gives

$$A' + A = 0, \qquad B' + B = 0,$$

with solutions $A(T) = A(0)e^{-T}$, $B(T) = B(0)e^{-T}$.

**Initial conditions:** $x(0) = 0$ gives $B(0) = 0$; $\dot{x}(0) = 1$ gives $A(0) = 1$. Hence

$$x \approx e^{-T}\sin\tau = e^{-\varepsilon t}\sin t + O(\varepsilon),$$

which matches the exact solution beautifully for all $t$, not just $t \ll 1/\varepsilon$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Two-Timing the Van der Pol Oscillator)</span></p>

For $\ddot{x} + x + \varepsilon(x^2 - 1)\dot{x} = 0$:

**$O(1)$:** $\partial_{\tau\tau}x_0 + x_0 = 0$, with general solution $x_0 = r(T)\cos(\tau + \phi(T))$, where $r(T)$ and $\phi(T)$ are the **slowly-varying amplitude and phase**.

**$O(\varepsilon)$:** Substituting $x_0$ into the $O(\varepsilon)$ equation and using the trigonometric identity $\sin(\theta)\cos^2(\theta) = \tfrac{1}{4}[\sin(\theta) + \sin(3\theta)]$, the resonant terms yield

$$r' = \tfrac{1}{2}r - \tfrac{1}{8}r^3 = \tfrac{1}{8}r(4 - r^2), \qquad \phi' = 0.$$

The amplitude equation $r' = \tfrac{1}{8}r(4 - r^2)$ is a one-dimensional flow on $r \ge 0$: $r^* = 0$ is unstable and $r^* = 2$ is stable. Hence $r(T) \to 2$ as $T \to \infty$, and the phase is constant: $\phi(T) = \phi_0$.

Therefore $x(t) \to 2\cos(t + \phi_0) + O(\varepsilon)$ as $t \to \infty$ — a **stable limit cycle** of radius $= 2 + O(\varepsilon)$ and frequency $\omega = 1 + O(\varepsilon^2)$.

For initial conditions $x(0) = 1$, $\dot{x}(0) = 0$: $r(0) = 1$ and $\phi(0) = 0$. Solving the amplitude equation by separation of variables gives

$$r(T) = \frac{2}{\sqrt{1 + 3e^{-T}}},$$

so that $x(t, \varepsilon) \approx \frac{2}{\sqrt{1 + 3e^{-\varepsilon t}}}\cos t + O(\varepsilon)$, which agrees strikingly well with numerical solutions even for $\varepsilon$ not especially small (e.g., $\varepsilon = 0.1$).

</div>

#### Averaged Equations

The same steps recur in every two-timing calculation. General formulas can be derived once and for all.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Averaged Equations for Weakly Nonlinear Oscillators)</span></p>

For the general weakly nonlinear oscillator $\ddot{x} + x + \varepsilon h(x, \dot{x}) = 0$, let $x_0 = r(T)\cos(\tau + \phi(T))$ be the $O(1)$ solution. The **averaged equations** (or **slow-time equations**) governing the slowly-varying amplitude $r$ and phase $\phi$ are

$$r' = \langle h\sin\theta \rangle, \qquad r\phi' = \langle h\cos\theta \rangle,$$

where $\theta = \tau + \phi$, $h = h(r\cos\theta, -r\sin\theta)$, and the angled brackets $\langle \cdot \rangle$ denote an average over one cycle of $\theta$:

$$\langle f \rangle = \frac{1}{2\pi}\int_0^{2\pi} f(\theta)\,d\theta.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Useful Averages)</span></p>

The following averages appear frequently:

$$\langle \cos\theta \rangle = \langle \sin\theta \rangle = 0, \quad \langle \cos^3\theta \rangle = \langle \sin^3\theta \rangle = 0, \quad \langle \cos^{2n+1}\theta \rangle = \langle \sin^{2n+1}\theta \rangle = 0,$$

$$\langle \cos^2\theta \rangle = \langle \sin^2\theta \rangle = \tfrac{1}{2}, \quad \langle \cos^4\theta \rangle = \langle \sin^4\theta \rangle = \tfrac{3}{8}, \quad \langle \cos^2\theta\sin^2\theta \rangle = \tfrac{1}{8}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Duffing Equation — Amplitude-Dependent Frequency)</span></p>

For the Duffing equation $\ddot{x} + x + \varepsilon x^3 = 0$, we have $h = x^3 = r^3\cos^3\theta$. The averaged equations give

$$r' = \langle h\sin\theta \rangle = r^3\langle \cos^3\theta\sin\theta \rangle = 0,$$

$$r\phi' = \langle h\cos\theta \rangle = r^3\langle \cos^4\theta \rangle = \tfrac{3}{8}r^3.$$

Hence $r(T) \equiv a$ (constant amplitude — consistent with the Duffing equation being conservative) and $\phi' = \tfrac{3}{8}a^2$. The angular frequency is

$$\omega = 1 + \varepsilon\phi' = 1 + \tfrac{3}{8}\varepsilon a^2 + O(\varepsilon^2).$$

**Physical interpretation:** The Duffing equation describes a unit mass on a nonlinear spring with restoring force $F(x) = -x - \varepsilon x^3$ and effective stiffness $k(x) = 1 + \varepsilon x^2$.

- For $\varepsilon > 0$: the spring gets *stiffer* with displacement (**hardening spring**), so the frequency *increases* with amplitude.
- For $\varepsilon < 0$: the spring gets *softer* (**softening spring**), so the frequency *decreases* with amplitude (as exemplified by the pendulum).

This amplitude-dependent frequency is intrinsically nonlinear — it cannot occur in linear oscillators.

</div>

#### Validity of Two-Timing

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Validity of Two-Timing)</span></p>

The one-term approximation $x_0$ is within $O(\varepsilon)$ of the true solution $x$ for all times up to and including $t \sim O(1/\varepsilon)$. If $x$ is a periodic solution, the approximation is even better: $x_0$ remains within $O(\varepsilon)$ of $x$ for *all* $t$.

The method of two-timing is closely related to the **method of averaging**, which provides the same slow-time equations. For rigorous statements about validity and asymptotic approximation, see Guckenheimer and Holmes (1983) or Grimshaw (1990).

</div>

## Chapter 8: Bifurcations Revisited

### Introduction

This chapter extends the earlier work on bifurcations (Chapter 3) from one-dimensional to two-dimensional systems. As we move up, we still find that fixed points can be created or destroyed or destabilized as parameters are varied — but now the same is true of closed orbits as well. Thus we can begin to *describe the ways in which oscillations can be turned on or off*.

In this broader context, a **bifurcation** means a change in the topological structure of the phase portrait as a parameter is varied (Section 6.3). Examples include changes in the number or stability of fixed points, closed orbits, or saddle connections.

The chapter is organized as follows: for each bifurcation, we start with a simple prototypical example, then graduate to more challenging examples. Models of genetic switches, chemical oscillators, driven pendula, and Josephson junctions illustrate the theory.

### 8.1 Saddle-Node, Transcritical, and Pitchfork Bifurcations

The bifurcations of fixed points discussed in Chapter 3 have analogs in two dimensions (and indeed, in *all* dimensions). Nothing really new happens when more dimensions are added — all the action is confined to a one-dimensional subspace along which the bifurcations occur, while in the extra dimensions the flow is either simple attraction or repulsion from that subspace.

#### Saddle-Node Bifurcation

The saddle-node bifurcation is the basic mechanism for the creation and destruction of fixed points. The prototypical example in two dimensions is:

$$\dot{x} = \mu - x^2, \qquad \dot{y} = -y.$$

In the $x$-direction we see the bifurcation behavior discussed in Section 3.1, while in the $y$-direction the motion is exponentially damped.

Consider the phase portrait as $\mu$ varies. For $\mu > 0$, there are two fixed points, a stable node at $(\sqrt{\mu}, 0)$ and a saddle at $(-\sqrt{\mu}, 0)$. As $\mu$ decreases, the saddle and node approach each other, then collide when $\mu = 0$, and finally disappear when $\mu < 0$.

Even after the fixed points have annihilated each other, they continue to influence the flow — as in Section 4.3, they leave a *ghost*, a bottleneck region that sucks trajectories in and delays them before allowing passage out the other side. The time spent in the bottleneck generically increases as $(\mu - \mu_c)^{-1/2}$, where $\mu_c$ is the value at which the saddle-node bifurcation occurs.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(General Saddle-Node Picture)</span></p>

Consider a two-dimensional system $\dot{x} = f(x, y)$, $\dot{y} = g(x, y)$ that depends on a parameter $\mu$. For some value of $\mu$ the nullclines intersect, and each intersection corresponds to a fixed point. As $\mu$ varies, the nullclines pull away from each other, becoming *tangent* at $\mu = \mu_c$. The fixed points approach each other and collide when $\mu = \mu_c$; after the nullclines pull apart, there are no intersections and the fixed points disappear with a bang. *All* saddle-node bifurcations have this character locally.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.1.1</span><span class="math-callout__name">(Genetic Control System)</span></p>

The following system has been discussed by Griffith (1971) as a model for a genetic control system. The activity of a certain gene is assumed to be directly induced by two copies of the protein for which it codes — leading to an autocatalytic feedback process. In dimensionless form, the equations are

$$\dot{x} = -ax + y, \qquad \dot{y} = \frac{x^2}{1 + x^2} - by,$$

where $x$ and $y$ are proportional to the concentrations of the protein and the messenger RNA, respectively, and $a, b > 0$ are parameters that govern the rate of degradation of $x$ and $y$.

**Task:** Show that the system has three fixed points when $a < a_c$, where $a_c$ is to be determined. Show that two of these coalesce in a saddle-node bifurcation when $a = a_c$. Then sketch the phase portrait for $a < a_c$ and give a biological interpretation.

**Solution:** The nullclines intersect when $ax = \frac{x^2}{b(1+x^2)}$. One solution is $x^* = 0$, giving $y^* = 0$. The other intersections satisfy the quadratic equation

$$ab(1 + x^2) = x, \qquad \text{i.e.,} \quad x^* = \frac{1 \pm \sqrt{1 - 4a^2 b^2}}{2ab}.$$

This has two solutions if $1 - 4a^2 b^2 > 0$, i.e., $2ab < 1$. These solutions coalesce when $2ab = 1$, hence $a_c = 1/2b$. Note that $x^* = 1$ at the bifurcation.

The Jacobian matrix at $(x, y)$ is

$$A = \begin{pmatrix} -a & 1 \\ \frac{2x}{(1+x^2)^2} & -b \end{pmatrix}.$$

$A$ has trace $\tau = -(a+b) < 0$, so all fixed points are either sinks or saddles, depending on $\Delta$. At $(0, 0)$, $\Delta = ab > 0$, so the origin is always a stable node (since $\tau^2 - 4\Delta = (a-b)^2 > 0$, except in the degenerate case $a = b$). At the other two fixed points, using $y^* = 1 + (x^*)^2$, one finds

$$\Delta = ab\left[\frac{(x^*)^2 - 1}{1 + (x^*)^2}\right].$$

So $\Delta < 0$ for the "middle" fixed point which has $0 < x^* < 1$; this is a *saddle point*. The fixed point with $x^* > 1$ is always a *stable node*, since $\Delta < ab$ and therefore $\tau^2 - 4\Delta > (a-b)^2 > 0$.

**Biological interpretation:** The system can act like a *biochemical switch*, but only if the mRNA and protein degrade slowly enough — specifically, their decay rates must satisfy $ab < 1/2$. In this case, there are two stable steady states: one at the origin (the gene is silent, no protein around to turn it on) and one where $x$ and $y$ are large (the gene is active, sustained by the high level of protein). The stable manifold of the saddle acts like a threshold; it determines whether the gene turns on or off, depending on the initial values of $x$ and $y$.

All trajectories relax rapidly onto the unstable manifold of the saddle, which plays a completely analogous role to the $x$-axis in the idealized saddle-node example. Thus, in many respects, the bifurcation is a fundamentally one-dimensional event, with the fixed points sliding toward each other along the unstable manifold like beads on a string. *This is why we spent so much time looking at bifurcations in one-dimensional systems* — they're the building blocks of analogous bifurcations in higher dimensions.

</div>

#### Transcritical and Pitchfork Bifurcations

Using the same idea as above, we can construct prototypical examples of transcritical and pitchfork bifurcations at a stable fixed point. In the $x$-direction the dynamics are given by the normal forms discussed in Chapter 3, and in the $y$-direction the motion is exponentially damped:

$$\dot{x} = \mu x - x^2, \quad \dot{y} = -y \qquad \text{(transcritical)}$$

$$\dot{x} = \mu x - x^3, \quad \dot{y} = -y \qquad \text{(supercritical pitchfork)}$$

$$\dot{x} = \mu x + x^3, \quad \dot{y} = -y \qquad \text{(subcritical pitchfork)}$$

The analysis in each case follows the same pattern, so we'll discuss only the supercritical pitchfork and leave the other two cases as exercises.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.1.2</span><span class="math-callout__name">(Supercritical Pitchfork)</span></p>

Plot the phase portraits for the supercritical pitchfork system $\dot{x} = \mu x - x^3$, $\dot{y} = -y$, for $\mu < 0$, $\mu = 0$, and $\mu > 0$.

**Solution:** For $\mu < 0$, the only fixed point is a stable node at the origin. For $\mu = 0$, the origin is still stable, but now we have very slow (algebraic) decay along the $x$-direction instead of exponential decay; this is the phenomenon of "critical slowing down" discussed in Section 3.4 and Exercise 2.4.9. For $\mu > 0$, the origin loses stability and gives birth to two new stable fixed points symmetrically located at $(x^*, y^*) = (\pm\sqrt{\mu}, 0)$. By computing the Jacobian at each point, one can check that the origin is a saddle and the other two fixed points are stable nodes.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.1.3</span><span class="math-callout__name">(Pitchfork with Symmetry)</span></p>

Show that a supercritical pitchfork bifurcation occurs at the origin in the system

$$\dot{x} = \mu x + y + \sin x, \qquad \dot{y} = x - y,$$

and determine the bifurcation value $\mu_c$. Plot the phase portrait near the origin for $\mu$ slightly greater than $\mu_c$.

**Solution:** The system is invariant under the change of variables $x \to -x$, $y \to -y$, so the phase portrait must be symmetric under reflection through the origin. The origin is a fixed point for all $\mu$, and its Jacobian is

$$A = \begin{pmatrix} \mu + 1 & 1 \\ 1 & -1 \end{pmatrix},$$

which has $\tau = \mu$ and $\Delta = -(\mu + 2)$. Hence the origin is a stable fixed point if $\mu < -2$ and a saddle if $\mu > -2$. This suggests that a pitchfork bifurcation occurs at $\mu_c = -2$.

To confirm this, we seek a symmetric pair of fixed points close to the origin for $\mu$ close to $\mu_c$. The fixed points satisfy $y = x$ and hence $(\mu + 1)x + \sin x = 0$. One solution is $x = 0$. Now suppose $x$ is small and nonzero, and expand the sine as a power series. Then

$$(\mu + 1)x + x - \frac{x^3}{3!} + O(x^5) = 0.$$

After dividing through by $x$: $\mu + 2 - x^2/6 \approx 0$. Hence there is a pair of fixed points with $x^* \approx \pm\sqrt{6(\mu + 2)}$ for $\mu$ slightly greater than $-2$. Thus a *supercritical* pitchfork bifurcation occurs at $\mu_c = -2$. Because the bifurcation is supercritical, we know the new fixed points are stable *without even checking*.

At the bifurcation, the Jacobian has eigenvectors $(1, 1)$ and $(1, -1)$, with eigenvalues $\lambda = 0$ and $\lambda = -2$. For $\mu$ slightly greater than $-2$, the origin becomes a saddle and so the zero eigenvalue becomes slightly positive. This information implies the phase portrait.

Note that because of the approximations we've made, this picture is only valid *locally* in both parameter and phase space — if we're not near the origin and if $\mu$ is not close to $\mu_c$, all bets are off.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Zero-Eigenvalue Bifurcations)</span></p>

In all of the examples above, the bifurcation occurs when $\Delta = 0$, or equivalently, when one of the eigenvalues equals zero. More generally, the saddle-node, transcritical, and pitchfork bifurcations are all examples of **zero-eigenvalue bifurcations**. (There are other examples, but these are the most common.) Such bifurcations always involve the collision of two or more fixed points.

In the next section we consider a fundamentally new kind of bifurcation, one that has no counterpart in one-dimensional systems. It provides a way for a fixed point to lose stability without colliding with any other fixed points.

</div>

### 8.2 Hopf Bifurcations

Suppose a two-dimensional system has a stable fixed point. What are all the possible ways it could lose stability as a parameter $\mu$ varies? The eigenvalues of the Jacobian are the key. If the fixed point is stable, the eigenvalues $\lambda_1, \lambda_2$ must both lie in the left half-plane $\operatorname{Re} \lambda < 0$. Since the $\lambda$'s satisfy a quadratic equation with real coefficients, there are two possible pictures: either the eigenvalues are both real and negative, or they are complex conjugates. To destabilize the fixed point, we need one or both of the eigenvalues to cross into the right half-plane as $\mu$ varies.

In Section 8.1 we explored the cases in which a real eigenvalue passes through $\lambda = 0$. These were just our old friends from Chapter 3, namely the saddle-node, transcritical, and pitchfork bifurcations. Now we consider the other possible scenario, in which two complex conjugate eigenvalues simultaneously cross the imaginary axis into the right half-plane.

#### Supercritical Hopf Bifurcation

Suppose we have a physical system that settles down to equilibrium through exponentially damped oscillations. Small disturbances decay after "ringing" for a while. Now suppose that the decay rate depends on a control parameter $\mu$. If the decay becomes slower and slower and finally changes to *growth* at a critical value $\mu_c$, the equilibrium state will lose stability. In many cases the resulting motion is a small-amplitude, sinusoidal, limit cycle oscillation about the former steady state. Then we say that the system has undergone a **supercritical Hopf bifurcation**.

In terms of the flow in phase space, a supercritical Hopf bifurcation occurs when a stable spiral changes into an unstable spiral surrounded by a small, nearly elliptical limit cycle.

A simple example of a supercritical Hopf bifurcation is given by the following system:

$$\dot{r} = \mu r - r^3, \qquad \dot{\theta} = \omega + br^2.$$

There are three parameters: $\mu$ controls the stability of the fixed point at the origin, $\omega$ gives the frequency of infinitesimal oscillations, and $b$ determines the dependence of frequency on amplitude for larger amplitude oscillations.

For $\mu < 0$ the origin $r = 0$ is a stable spiral whose sense of rotation depends on the sign of $\omega$. For $\mu = 0$ the origin is still a stable spiral, though a very weak one: the decay is only algebraically fast (this case was shown in Figure 6.3.2). Recall that the linearization wrongly predicts a center. Finally, for $\mu > 0$ there is an unstable spiral at the origin and a stable circular limit cycle at $r = \sqrt{\mu}$.

To see how the eigenvalues behave during the bifurcation, we rewrite the system in Cartesian coordinates. Writing $x = r\cos\theta$, $y = r\sin\theta$:

$$\dot{x} = \mu x - \omega y + \text{cubic terms}, \qquad \dot{y} = \omega x + \mu y + \text{cubic terms}.$$

So the Jacobian at the origin is

$$A = \begin{pmatrix} \mu & -\omega \\ \omega & \mu \end{pmatrix},$$

which has eigenvalues $\lambda = \mu \pm i\omega$. As expected, the eigenvalues cross the imaginary axis from left to right as $\mu$ increases from negative to positive values.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Rules of Thumb for Supercritical Hopf Bifurcations)</span></p>

Our idealized case illustrates two rules that hold *generically* for supercritical Hopf bifurcations:

1. The size of the limit cycle grows continuously from zero, and increases proportional to $\sqrt{\mu - \mu_c}$, for $\mu$ close to $\mu_c$.
2. The frequency of the limit cycle is given approximately by $\omega = \operatorname{Im}\lambda$, evaluated at $\mu = \mu_c$. This formula is exact at the birth of the limit cycle, and correct within $O(\mu - \mu_c)$ for $\mu$ close to $\mu_c$. The period is therefore $T = (2\pi / \operatorname{Im}\lambda) + O(\mu - \mu_c)$.

Our idealized example also has some artifactual properties. First, in Hopf bifurcations encountered in practice, the limit cycle is elliptical, not circular, and its shape becomes distorted as $\mu$ moves away from the bifurcation point. Our example is only typical topologically, not geometrically. Second, in our idealized case the eigenvalues move on horizontal lines as $\mu$ varies, i.e., $\operatorname{Im}\lambda$ is strictly independent of $\mu$. Normally, the eigenvalues would follow a curvy path and cross the imaginary axis with nonzero slope.

</div>

#### Subcritical Hopf Bifurcation

Like pitchfork bifurcations, Hopf bifurcations come in both super- and subcritical varieties. The subcritical case is always much more dramatic, and potentially dangerous in engineering applications. After the bifurcation, the trajectories must *jump* to a distant attractor, which may be a fixed point, another limit cycle, infinity, or — in three and higher dimensions — a chaotic attractor.

Consider the two-dimensional example

$$\dot{r} = \mu r + r^3 - r^5, \qquad \dot{\theta} = \omega + br^2.$$

The important difference from the earlier supercritical case is that the cubic term $r^3$ is now *destabilizing*; it helps to drive trajectories away from the origin.

The phase portraits are as follows. For $\mu < 0$ there are two attractors, a stable limit cycle and a stable fixed point at the origin. Between them lies an unstable cycle, shown as a dashed curve; it's the player to watch in this scenario. As $\mu$ increases, the unstable cycle tightens like a noose around the fixed point. A **subcritical Hopf bifurcation** occurs at $\mu = 0$, where the unstable cycle shrinks to zero amplitude and engulfs the origin, rendering it unstable. For $\mu > 0$, the large-amplitude limit cycle is suddenly the only attractor in town. Solutions that used to remain near the origin are now forced to grow into large-amplitude oscillations.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hysteresis in Subcritical Hopf Bifurcation)</span></p>

Note that the system exhibits *hysteresis*: once large-amplitude oscillations have begun, they cannot be turned off by bringing $\mu$ back to zero. In fact, the large oscillations will persist until $\mu = -1/4$ where the stable and unstable cycles collide and annihilate. This destruction of the large-amplitude cycle occurs via another type of bifurcation, to be discussed in Section 8.4.

</div>

#### Subcritical, Supercritical, or Degenerate Bifurcation?

Given that a Hopf bifurcation occurs, how can we tell if it's sub- or supercritical? The linearization doesn't provide a distinction: in both cases, a pair of eigenvalues moves from the left to the right half-plane.

An analytical criterion exists, but it can be difficult to use (see Exercises 8.2.12–15 for some tractable cases). A quick and dirty approach is to use the computer. If a small, attracting limit cycle appears immediately after the fixed point goes unstable, and if its amplitude shrinks back to zero as the parameter is reversed, the bifurcation is supercritical; otherwise, it's probably subcritical.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Degenerate Hopf Bifurcation)</span></p>

You should also be aware of a **degenerate Hopf bifurcation**. An example is given by the damped pendulum $\ddot{x} + \mu\dot{x} + \sin x = 0$. As we change the damping $\mu$ from positive to negative, the fixed point at the origin changes from a stable to an unstable spiral. However, at $\mu = 0$ we do *not* have a true Hopf bifurcation because there are no limit cycles on either side of the bifurcation. Instead, at $\mu = 0$ we have a continuous band of closed orbits surrounding the origin. These are *not* limit cycles! (Recall that a limit cycle is an *isolated* closed orbit.)

This degenerate case typically arises when a nonconservative system suddenly becomes conservative at the bifurcation point. Then the fixed point becomes a nonlinear center, rather than the weak spiral required by a Hopf bifurcation.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.2.1</span><span class="math-callout__name">(Hopf Bifurcation Classification)</span></p>

Consider the system $\dot{x} = \mu x - y + xy^2$, $\dot{y} = x + \mu y + y^3$. Show that a Hopf bifurcation occurs at the origin as $\mu$ varies. Is the bifurcation subcritical, supercritical, or degenerate?

**Solution:** The Jacobian at the origin is $A = \begin{pmatrix} \mu & -1 \\ 1 & \mu \end{pmatrix}$, which has $\tau = 2\mu$, $\Delta = \mu^2 + 1 > 0$, and $\lambda = \mu \pm i$. Hence, as $\mu$ increases through zero, the origin changes from a stable spiral to an unstable spiral. This suggests that some kind of Hopf bifurcation takes place at $\mu = 0$.

To decide whether the bifurcation is subcritical, supercritical, or degenerate, we use simple reasoning and numerical integration. If we transform to polar coordinates, we find that

$$\dot{r} = \mu r + ry^2,$$

as you should check. Hence $\dot{r} \geq \mu r$. This implies that for $\mu > 0$, $r(t)$ grows at *least* as fast as $r_0 e^{\mu t}$. In other words, all trajectories are repelled out to infinity! So there are certainly no closed orbits for $\mu > 0$. In particular, the unstable spiral is *not* surrounded by a stable limit cycle; hence the bifurcation cannot be supercritical.

Could the bifurcation be degenerate? That would require that the origin be a nonlinear center when $\mu = 0$. But $\dot{r}$ is strictly positive away from the $x$-axis, so closed orbits are still impossible.

By process of elimination, we expect that the bifurcation is *subcritical*. This is confirmed by the computer-generated phase portrait for $\mu = -0.2$: an *unstable* limit cycle surrounds the stable fixed point, just as we expect in a subcritical bifurcation. Furthermore, the cycle is nearly elliptical and surrounds a gently winding spiral — these are typical features of *either* kind of Hopf bifurcation.

</div>

### 8.3 Oscillating Chemical Reactions

For an application of Hopf bifurcations, we now consider a class of experimental systems known as **chemical oscillators**. These systems are remarkable, both for their spectacular behavior and for the story behind their discovery.

#### Belousov's "Supposedly Discovered Discovery"

In the early 1950s the Russian biochemist Boris Belousov was trying to create a test tube caricature of the Krebs cycle, a metabolic process that occurs in living cells. When he mixed citric acid and bromate ions in a solution of sulfuric acid, and in the presence of a cerium catalyst, he observed to his astonishment that the mixture became yellow, then faded to colorless after about a minute, then returned to yellow a minute later, then became colorless again, and continued to oscillate dozens of times before finally reaching equilibrium after about an hour.

Today it comes as no surprise that chemical reactions can oscillate spontaneously — such reactions have become a standard demonstration in chemistry classes. But in Belousov's day, his discovery was so radical that he couldn't get his work published. It was thought that all solutions of chemical reagents must go *monotonically* to equilibrium, because of the laws of thermodynamics. Belousov's paper was rejected by one journal after another.

Belousov finally managed to publish a brief abstract in the obscure proceedings of a Russian medical meeting (Belousov 1959). In 1961 a graduate student named Zhabotinsky was assigned to look into it. Zhabotinsky confirmed that Belousov was right all along, and brought this work to light at an international conference in Prague in 1968. The **BZ reaction**, as it came to be called, was seen as a manageable model of more complex biological and biochemical oscillations.

The analogy to biology turned out to be surprisingly close: Zaikin and Zhabotinsky (1970) and Winfree (1972) observed beautiful propagating *waves* of oxidation in thin unstirred layers of BZ reagent, analogous to waves of excitation in neural or cardiac tissue.

#### Chlorine Dioxide–Iodine–Malonic Acid Reaction

Lengyel et al. (1990) proposed and analyzed a particularly elegant model of the chlorine dioxide-iodine-malonic acid ($\text{ClO}_2$-$\text{I}_2$-MA) reaction. After suitable nondimensionalization, the model becomes

$$\dot{x} = a - x - \frac{4xy}{1 + x^2}, \qquad \dot{y} = bx\left(1 - \frac{y}{1 + x^2}\right),$$

where $x$ and $y$ are the dimensionless concentrations of $\text{I}^-$ and $\text{ClO}_2^-$. The parameters $a, b > 0$ depend on the empirical rate constants and on the concentrations assumed for the slow reactants.

We begin the analysis by constructing a trapping region and applying the Poincaré–Bendixson theorem. Then we'll show that the chemical oscillations arise from a supercritical Hopf bifurcation.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.3.1</span><span class="math-callout__name">(Existence of Closed Orbit)</span></p>

Prove that the system has a closed orbit in the positive quadrant $x, y > 0$ if $a$ and $b$ satisfy certain constraints, to be determined.

**Solution:** As in Example 7.3.2, the nullclines help us to construct a trapping region. Equation $\dot{x} = 0$ gives the curve

$$y = \frac{(a - x)(1 + x^2)}{4x},$$

and $\dot{y} = 0$ on the $y$-axis and on the parabola $y = 1 + x^2$.

Consider a dashed box that encloses the intersection of the nullclines. It's a trapping region because all the vectors on the boundary point into the box.

We can't apply the Poincaré–Bendixson theorem yet, because there's a fixed point

$$x^* = a/5, \qquad y^* = 1 + (x^*)^2 = 1 + (a/5)^2$$

inside the box at the intersection of the nullclines. But now we argue as in Example 7.3.3: if the fixed point turns out to be a repeller, we *can* apply the Poincaré-Bendixson theorem to the "punctured" box obtained by removing the fixed point.

The Jacobian at $(x^*, y^*)$ is

$$\frac{1}{1 + (x^*)^2}\begin{pmatrix} 3(x^*)^2 - 5 & -4x^* \\ 2b(x^*)^2 & -bx^* \end{pmatrix}.$$

The determinant and trace are

$$\Delta = \frac{5bx^*}{1 + (x^*)^2} > 0, \qquad \tau = \frac{3(x^*)^2 - 5 - bx^*}{1 + (x^*)^2}.$$

Since $\Delta > 0$, the fixed point is never a saddle. Hence $(x^*, y^*)$ is a repeller if $\tau > 0$, i.e., if

$$b < b_c \equiv 3a/5 - 25/a.$$

When this condition holds, the Poincaré-Bendixson theorem implies the existence of a closed orbit somewhere in the punctured box.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.3.2</span><span class="math-callout__name">(Hopf Bifurcation in Chemical Oscillator)</span></p>

Using numerical integration, show that a Hopf bifurcation occurs at $b = b_c$ and decide whether the bifurcation is sub- or supercritical.

**Solution:** The analytical results above show that as $b$ decreases through $b_c$, the fixed point changes from a stable spiral to an unstable spiral; this is the signature of a Hopf bifurcation. (Here we have chosen $a = 10$; then $b_c = 3.5$.) When $b > b_c$, all trajectories spiral into the stable fixed point. For $b < b_c$ they are attracted to a stable limit cycle.

Hence the bifurcation is *supercritical* — after the fixed point loses stability, it is surrounded by a stable limit cycle. Moreover, by plotting phase portraits as $b \to b_c$ from below, we could confirm that the limit cycle shrinks continuously to a point, as required.

Our results are summarized in a stability diagram in the $(a, b)$-parameter plane. The boundary between the two regions is given by the Hopf bifurcation locus $b = 3a/5 - 25/a$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.3.3</span><span class="math-callout__name">(Period of the Limit Cycle)</span></p>

Approximate the period of the limit cycle for $b$ slightly less than $b_c$.

**Solution:** The frequency is approximated by the imaginary part of the eigenvalues at the bifurcation. As usual, the eigenvalues satisfy $\lambda^2 - \tau\lambda + \Delta = 0$. Since $\tau = 0$ and $\Delta > 0$ at $b = b_c$, we find

$$\lambda = \pm i\sqrt{\Delta}.$$

But at $b_c$,

$$\Delta = \frac{5b_c x^*}{1 + (x^*)^2} = \frac{5\left(\frac{3a}{5} - \frac{25}{a}\right)\left(\frac{a}{5}\right)}{1 + (a/5)^2} = \frac{15a^2 - 625}{a^2 + 25}.$$

Hence $\omega \approx \Delta^{1/2} = \left[\frac{15a^2 - 625}{a^2 + 25}\right]^{1/2}$ and therefore

$$T = 2\pi/\omega = 2\pi\left[\frac{a^2 + 25}{15a^2 - 625}\right]^{1/2}.$$

As $a \to \infty$, $T \to 2\pi/\sqrt{15} \approx 1.63$.

</div>

### 8.4 Global Bifurcations of Cycles

In two-dimensional systems, there are four common ways in which limit cycles are created or destroyed. The Hopf bifurcation is the most famous, but the other three deserve their day in the sun. They are harder to detect because they involve large regions of the phase plane rather than just the neighborhood of a single fixed point. Hence they are called **global bifurcations**.

#### Saddle-Node Bifurcation of Cycles

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fold / Saddle-Node Bifurcation of Cycles)</span></p>

A bifurcation in which two limit cycles coalesce and annihilate is called a **fold** or **saddle-node bifurcation of cycles**, by analogy with the related bifurcation of fixed points.

</div>

An example occurs in the system

$$\dot{r} = \mu r + r^3 - r^5, \qquad \dot{\theta} = \omega + br^2,$$

studied in Section 8.2 in connection with the subcritical Hopf bifurcation at $\mu = 0$. Now we concentrate on the dynamics for $\mu < 0$.

It is helpful to regard the radial equation $\dot{r} = \mu r + r^3 - r^5$ as a one-dimensional system. This system undergoes a saddle-node bifurcation of fixed points at $\mu_c = -1/4$. Returning to the two-dimensional system, these fixed points correspond to circular *limit cycles*.

At $\mu_c$ a half-stable cycle is born out of the clear blue sky. As $\mu$ increases it splits into a pair of limit cycles, one stable, one unstable. Viewed in the other direction, a stable and unstable cycle collide and disappear as $\mu$ decreases through $\mu_c$. Notice that the origin remains stable throughout; it does not participate in this bifurcation.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Amplitude at Birth)</span></p>

For future reference, note that at birth the cycle has $O(1)$ amplitude, in contrast to the Hopf bifurcation, where the limit cycle has small amplitude proportional to $(\mu - \mu_c)^{1/2}$.

</div>

#### Infinite-Period Bifurcation

Consider the system

$$\dot{r} = r(1 - r^2), \qquad \dot{\theta} = \mu - \sin\theta,$$

where $\mu \geq 0$. This system combines two one-dimensional systems that we have studied previously in Chapters 3 and 4. In the radial direction, all trajectories (except $r^* = 0$) approach the unit circle monotonically as $t \to \infty$. In the angular direction, the motion is everywhere counterclockwise if $\mu > 1$, whereas there are two invariant rays defined by $\sin\theta = \mu$ if $\mu < 1$.

Hence as $\mu$ decreases through $\mu_c = 1$, the phase portraits change dramatically. As $\mu$ decreases, the limit cycle $r = 1$ develops a bottleneck at $\theta = \pi/2$ that becomes increasingly severe as $\mu \to 1^+$. The oscillation period lengthens and finally becomes infinite at $\mu_c = 1$, when a fixed point appears on the circle; hence the term **infinite-period bifurcation**. For $\mu < 1$, the fixed point splits into a saddle and a node.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Scaling Near Infinite-Period Bifurcation)</span></p>

As the bifurcation is approached, the amplitude of the oscillation stays $O(1)$ but the period increases like $(\mu - \mu_c)^{-1/2}$, for the same reasons discussed in Section 4.3.

</div>

#### Homoclinic Bifurcation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Homoclinic / Saddle-Loop Bifurcation)</span></p>

In this scenario, part of a limit cycle moves closer and closer to a saddle point. At the bifurcation the cycle touches the saddle point and becomes a **homoclinic orbit**. This is another kind of infinite-period bifurcation; to avoid confusion, we call it a **saddle-loop** or **homoclinic bifurcation**.

</div>

It is hard to find an analytically transparent example, so we resort to the computer. Consider the system

$$\dot{x} = y, \qquad \dot{y} = \mu y + x - x^2 + xy.$$

Numerically, the bifurcation is found to occur at $\mu_c \approx -0.8645$. For $\mu < \mu_c$, say $\mu = -0.92$, a stable limit cycle passes close to a saddle point at the origin. As $\mu$ increases to $\mu_c$, the limit cycle swells and bangs into the saddle, creating a homoclinic orbit. Once $\mu > \mu_c$, the saddle connection breaks and the loop is destroyed.

The key to this bifurcation is the behavior of the unstable manifold of the saddle. Look at the branch of the unstable manifold that leaves the origin to the northeast: after it loops around, it either hits the origin (creating a homoclinic orbit) or veers off to one side or the other.

#### Scaling Laws

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Scaling Laws for Bifurcations of Cycles)</span></p>

For each of the bifurcations given here, there are characteristic *scaling laws* that govern the amplitude and period of the limit cycle as the bifurcation is approached. Let $\mu$ denote a dimensionless measure of the distance from the bifurcation, and assume $\mu \ll 1$. The generic scaling laws for bifurcations of cycles in two-dimensional systems are:

| Bifurcation | Amplitude of stable limit cycle | Period of cycle |
| --- | --- | --- |
| Supercritical Hopf | $O(\mu^{1/2})$ | $O(1)$ |
| Saddle-node bifurcation of cycles | $O(1)$ | $O(1)$ |
| Infinite-period | $O(1)$ | $O(\mu^{-1/2})$ |
| Homoclinic | $O(1)$ | $O(\ln\mu)$ |

All of these laws have been explained previously, except those for the homoclinic bifurcation. The scaling of the period in that case is obtained by estimating the time required for a trajectory to pass by a saddle point.

Exceptions to these rules can occur, but only if there is some symmetry or other special feature that renders the problem nongeneric.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.4.1</span><span class="math-callout__name">(Van der Pol Oscillator — Degenerate Hopf)</span></p>

The van der Pol oscillator $\ddot{x} + \varepsilon\dot{x}(x^2 - 1) + x = 0$ does not seem to fit anywhere in the table above. At $\varepsilon = 0$, the eigenvalues at the origin are pure imaginary ($\lambda = \pm i$), suggesting that a Hopf bifurcation occurs at $\varepsilon = 0$. But we know from Section 7.6 that for $0 < \varepsilon \ll 1$, the system has a limit cycle of amplitude $r \approx 2$. Thus the cycle is born "full grown," not with size $O(\varepsilon^{1/2})$ as predicted by the scaling law. What's the explanation?

**Solution:** The bifurcation at $\varepsilon = 0$ is degenerate. The nonlinear term $\varepsilon\dot{x}x^2$ vanishes at precisely the same parameter value as the eigenvalues cross the imaginary axis. That's a nongeneric coincidence if there ever was one!

We can rescale $x$ to remove this degeneracy. Write the equation as $\ddot{x} + x + \varepsilon x^2\dot{x} - \varepsilon\dot{x} = 0$. Let $u^2 = \varepsilon x^2$ to remove the $\varepsilon$-dependence of the nonlinear term. Then $u = \varepsilon^{1/2}x$ and the equation becomes

$$\ddot{u} + u + u^2\dot{u} - \varepsilon\dot{u} = 0.$$

Now the nonlinear term is not destroyed when the eigenvalues become pure imaginary. From Section 7.6 the limit cycle solution is $x(t, \varepsilon) \approx 2\cos t$ for $0 < \varepsilon \ll 1$. In terms of $u$ this becomes

$$u(t, \varepsilon) \approx (2\sqrt{\varepsilon})\cos t.$$

Hence the amplitude grows like $\varepsilon^{1/2}$, just as expected for a Hopf bifurcation.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Importance of Scaling Laws)</span></p>

Why should you care about these scaling laws? Suppose you're an experimental scientist and the system you're studying exhibits a stable limit cycle oscillation. Now suppose you change a control parameter and the oscillation stops. By examining the scaling of the period and amplitude near this bifurcation, you can learn something about the system's dynamics (which are usually not known precisely, if at all). In this way, possible models can be eliminated or supported.

</div>

### 8.5 Hysteresis in the Driven Pendulum and Josephson Junction

This section deals with a physical problem in which both homoclinic and infinite-period bifurcations arise. The problem was introduced back in Sections 4.4 and 4.6: the dynamics of a damped pendulum driven by a constant torque, or equivalently, a superconducting Josephson junction driven by a constant current. Because we weren't ready for two-dimensional systems at that time, we reduced both problems to vector fields on the circle by looking at the heavily *overdamped limit* of negligible mass (for the pendulum) or negligible capacitance (for the Josephson junction).

Now we're ready to tackle the full two-dimensional problem. As we claimed at the end of Section 4.6, for sufficiently weak damping the pendulum and the Josephson junction can exhibit intriguing hysteresis effects, thanks to the coexistence of a stable limit cycle and a stable fixed point.

#### Governing Equations

As explained in Section 4.6, the governing equation for the Josephson junction is

$$\frac{\hbar C}{2e}\ddot{\phi} + \frac{\hbar}{2eR}\dot{\phi} + I_c\sin\phi = I_B,$$

where $\hbar$ is Planck's constant divided by $2\pi$, $e$ is the charge on the electron, $I_B$ is the constant bias current, $C$, $R$, and $I_c$ are the junction's capacitance, resistance, and critical current, and $\phi(t)$ is the phase difference across the junction.

To highlight the role of damping, we nondimensionalize differently from Section 4.6. Let

$$\tilde{t} = \left(\frac{2eI_c}{\hbar C}\right)^{1/2}t, \quad I = \frac{I_B}{I_c}, \quad \alpha = \left(\frac{\hbar}{2eI_cR^2C}\right)^{1/2}.$$

Then the equation becomes

$$\phi'' + \alpha\phi' + \sin\phi = I,$$

where $\alpha$ and $I$ are the dimensionless damping and applied current, and the prime denotes differentiation with respect to $\tilde{t}$. Here $\alpha > 0$ on physical grounds, and we may choose $I \geq 0$ without loss of generality. Letting $y = \phi'$, the system becomes

$$\phi' = y, \qquad y' = I - \sin\phi - \alpha y.$$

As in Section 6.7 the phase space is a *cylinder*, since $\phi$ is an angular variable and $y$ is a real number (best thought of as an angular velocity).

#### Fixed Points

The fixed points satisfy $y^* = 0$ and $\sin\phi^* = I$. Hence there are two fixed points on the cylinder if $I < 1$, and none if $I > 1$. When the fixed points exist, one is a saddle and other is a sink, since the Jacobian

$$A = \begin{pmatrix} 0 & 1 \\ -\cos\phi^* & -\alpha \end{pmatrix}$$

has $\tau = -\alpha < 0$ and $\Delta = \cos\phi^* = \pm\sqrt{1 - I^2}$. When $\Delta > 0$, we have a stable node if $\tau^2 - 4\Delta = \alpha^2 - 4\sqrt{1-I^2} > 0$ (i.e., if the damping is strong enough or if $I$ is close to 1); otherwise the sink is a stable spiral. At $I = 1$ the stable node and the saddle coalesce in a **saddle-node bifurcation of fixed points**.

#### Existence of a Closed Orbit

What happens when $I > 1$? There are no more fixed points available; something new has to happen. We claim that *all trajectories are attracted to a unique, stable limit cycle*.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Existence and Uniqueness of Limit Cycle for $I > 1$)</span></p>

For $I > 1$, the system $\phi' = y$, $y' = I - \sin\phi - \alpha y$ has a unique, stable limit cycle (a rotation on the cylinder). All trajectories are attracted to it.

</div>

The proof proceeds by constructing a Poincaré map on a suitable cross-section of the cylinder.

**Step 1 — Existence (Poincaré map argument):** Consider the nullcline $y = \alpha^{-1}(I - \sin\phi)$ where $y' = 0$. The flow is downward above the nullcline and upward below it. In particular, all trajectories eventually enter the strip $y_1 \leq y \leq y_2$ (Figure 8.5.1), and stay in there forever. Here $0 < y_1 < (I-1)/\alpha$ and $y_2 > (I+1)/\alpha$ are arbitrary fixed numbers.

Also, since $\phi = 0$ and $\phi = 2\pi$ are equivalent on the cylinder, we may confine our attention to the rectangular box $0 \leq \phi \leq 2\pi$, $y_1 \leq y \leq y_2$. Inside the strip, the flow is always to the right, because $y > 0$ implies $\phi' > 0$.

Now consider a trajectory that starts at a height $y$ on the left side of the box, and follow it until it intersects the right side of the box at some new height $P(y)$. The mapping from $y$ to $P(y)$ is called the **Poincaré map** (also called the **first-return map**).

The key point: we can't compute $P(y)$ explicitly, but *if we can show that there's a point $y^*$ such that $P(y^*) = y^*$, then the corresponding trajectory will be a closed orbit* (because it returns to the same location on the cylinder after one lap).

To show that such a $y^*$ must exist, we need to know what the graph of $P(y)$ looks like, at least roughly. One shows that:
- $P(y_1) > y_1$ (because the flow is strictly upward at first and the trajectory can never return below $y_1$),
- $P(y_2) < y_2$ (by the same kind of argument),
- $P(y)$ is a *continuous* function (because solutions of differential equations depend continuously on initial conditions),
- $P(y)$ is a *monotonic* function (because if $P(y)$ were not monotonic, two trajectories would cross — and that's forbidden).

By the intermediate value theorem, the graph of $P(y)$ must cross the $45°$ diagonal *somewhere*; that intersection is our desired $y^*$.

**Step 2 — Uniqueness:** The argument above proves the *existence* of a closed orbit, and almost proves its uniqueness. But we haven't excluded the possibility that $P(y) \equiv y$ on some interval, in which case there would be a band of infinitely many closed orbits.

To nail down uniqueness, we recall from Section 6.7 that there are two topologically different kinds of periodic orbits on a cylinder: **librations** and **rotations**. For $I > 1$, librations are impossible because any libration must encircle a fixed point, by index theory — but there are no fixed points when $I > 1$. Hence we only need to consider rotations.

Suppose there were two different rotations. The phase portrait on the cylinder would have one rotation lying strictly above the other. Let $y_U(\phi)$ and $y_L(\phi)$ denote the "upper" and "lower" rotations, where $y_U(\phi) > y_L(\phi)$ for all $\phi$.

The existence of two such rotations leads to a contradiction, as shown by the following energy argument. Let

$$E = \tfrac{1}{2}y^2 - \cos\phi.$$

After one circuit around any rotation $y(\phi)$, the change in energy $\Delta E$ must vanish. Hence

$$0 = \Delta E = \int_0^{2\pi} \frac{dE}{d\phi}\,d\phi.$$

Using the governing equations, one finds $dE/d\phi = I - \alpha y$. Thus

$$0 = \int_0^{2\pi}(I - \alpha y)\,d\phi,$$

which implies that any rotation must satisfy

$$\int_0^{2\pi} y(\phi)\,d\phi = \frac{2\pi I}{\alpha}.$$

But since $y_U(\phi) > y_L(\phi)$, we would have $\int_0^{2\pi} y_U(\phi)\,d\phi > \int_0^{2\pi} y_L(\phi)\,d\phi$, and so the constraint cannot hold for *both* rotations. This contradiction proves that the rotation for $I > 1$ is unique, as claimed.

#### Homoclinic Bifurcation

Suppose we slowly decrease $I$, starting from some value $I > 1$. What happens to the rotating solution? Think about the pendulum: as the driving torque is reduced, the pendulum struggles more and more to make it over the top. At some critical value $I < 1$, the torque is insufficient to overcome gravity and damping, and the pendulum can no longer whirl. Then the rotation disappears and all solutions damp out to the rest state.

The corresponding bifurcation in phase space depends on the damping parameter $\alpha$. If $\alpha$ is sufficiently small, the stable limit cycle is destroyed in a **homoclinic bifurcation**.

For $I_c < I < 1$, the system is bistable: a sink coexists with a stable limit cycle. Keep your eye on the trajectory labeled $U$ that is a branch of the unstable manifold of the saddle. As $t \to \infty$, $U$ asymptotically approaches the stable limit cycle. As $I$ decreases, the stable limit cycle moves down and squeezes $U$ closer to the stable manifold of the saddle. When $I = I_c$, *the limit cycle merges with $U$* in a homoclinic bifurcation. Now $U$ is a homoclinic orbit — it joins the saddle to itself. Finally, when $I < I_c$ the saddle connection breaks and $U$ spirals into the sink.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stability Diagram)</span></p>

The scenario described above is valid only if the dimensionless damping $\alpha$ is sufficiently small. For large $\alpha$ (the overdamped limit studied in Section 4.6), the periodic solution is destroyed by an **infinite-period bifurcation** (a saddle and a node are born on the former limit cycle). So it's plausible that an infinite-period bifurcation should also occur if $\alpha$ is large but finite.

Putting it all together, we arrive at a stability diagram in the $(\alpha, I)$-parameter plane. Three types of bifurcations occur: homoclinic and infinite-period bifurcations of periodic orbits, and a saddle-node bifurcation of fixed points. For small $\alpha$ the limit cycle is destroyed by a homoclinic bifurcation; for large $\alpha$ it is destroyed by an infinite-period bifurcation. The homoclinic bifurcation curve is tangent to the line $I = 4\alpha/\pi$ as $\alpha \to 0$.

</div>

#### Hysteretic Current-Voltage Curve

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hysteresis in the Josephson Junction)</span></p>

The stability diagram explains why lightly damped Josephson junctions have hysteretic $I$–$V$ curves. Suppose $\alpha$ is small and $I$ is initially below the homoclinic bifurcation curve. Then the junction will be operating at the stable fixed point, corresponding to the zero-voltage state. As $I$ is increased, nothing changes until $I$ exceeds $1$. Then the stable fixed point disappears in a saddle-node bifurcation, and the junction jumps into a nonzero voltage state (the limit cycle).

If $I$ is brought back down, the limit cycle persists below $I = 1$ but its frequency tends to zero continuously as $I_c$ is approached. Specifically, the frequency tends to zero like $[\ln(I - I_c)]^{-1}$, just as expected from the scaling law for homoclinic bifurcations. Now recall from Section 4.6 that the junction's dc-voltage is proportional to its oscillation frequency. Hence the voltage also returns to zero continuously as $I \to I_c^+$.

In practice, the voltage appears to jump discontinuously back to zero, but that is to be expected because $[\ln(I - I_c)]^{-1}$ has *infinite derivatives of all orders* at $I_c$! The steepness of the curve makes it impossible to resolve the continuous return to zero experimentally.

</div>

### 8.6 Coupled Oscillators and Quasiperiodicity

Besides the plane and the cylinder, another important two-dimensional phase space is the **torus**. It is the natural phase space for systems of the form

$$\dot{\theta}_1 = f_1(\theta_1, \theta_2), \qquad \dot{\theta}_2 = f_2(\theta_1, \theta_2),$$

where $f_1$ and $f_2$ are periodic in both arguments. For instance, a simple model of **coupled oscillators** is given by

$$\dot{\theta}_1 = \omega_1 + K_1\sin(\theta_2 - \theta_1), \qquad \dot{\theta}_2 = \omega_2 + K_2\sin(\theta_1 - \theta_2),$$

where $\theta_1, \theta_2$ are the *phases* of the oscillators, $\omega_1, \omega_2 > 0$ are their *natural frequencies*, and $K_1, K_2 \geq 0$ are *coupling constants*. This equation has been used to model the interaction between human circadian rhythms and the sleep-wake cycle (Strogatz 1986, 1987).

An intuitive way to think about this system is to imagine two friends jogging on a circular track. Here $\theta_1(t)$, $\theta_2(t)$ represent their positions on the track, and $\omega_1$, $\omega_2$ are proportional to their preferred running speeds. If they were uncoupled, each would run at his or her preferred speed and the faster one would periodically overtake the slower one (as in Example 4.2.1). But these are *friends* — they want to run around *together*! So they need to compromise, with each adjusting his or her speed as necessary. If their preferred speeds are too different, phase-locking will be impossible and they may want to find new running partners.

Since the curved surface of a torus makes it hard to draw phase portraits, we prefer to use an equivalent representation: a *square with periodic boundary conditions*. Then if a trajectory runs off an edge, it magically reappears on the opposite edge, as in some video games.

#### Uncoupled System

Even the seemingly trivial case of uncoupled oscillators ($K_1 = K_2 = 0$) holds some surprises. Then the system reduces to $\dot{\theta}_1 = \omega_1$, $\dot{\theta}_2 = \omega_2$. The corresponding trajectories on the square are straight lines with constant slope $d\theta_2/d\theta_1 = \omega_2/\omega_1$. There are two qualitatively different cases, depending on whether the slope is a rational or an irrational number.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Rational vs. Irrational Frequency Ratio)</span></p>

If the slope is **rational**, then $\omega_1/\omega_2 = p/q$ for some integers $p$, $q$ with no common factors. In this case *all trajectories are closed orbits* on the torus, because $\theta_1$ completes $p$ revolutions in the same time that $\theta_2$ completes $q$ revolutions. The resulting curves are called $p{:}q$ **torus knots**. For example, when $p = 3$, $q = 2$, the trajectory on the square closes after 3 horizontal and 2 vertical traversals, and when plotted on the torus it gives a **trefoil knot**.

In fact the trajectories are always knotted if $p$, $q \geq 2$ have no common factors.

If the slope is **irrational**, the flow is said to be **quasiperiodic**. Every trajectory winds around endlessly on the torus, never intersecting itself and yet never quite closing. How can we be sure the trajectories never close? Any closed trajectory necessarily makes an integer number of revolutions in both $\theta_1$ and $\theta_2$; hence the slope would have to be rational, contrary to assumption.

Furthermore, when the slope is irrational, each trajectory is **dense** on the torus: in other words, each trajectory comes arbitrarily close to any given point on the torus. This is *not* to say that the trajectory passes *through* each point; it just comes arbitrarily close (Exercise 8.6.3).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Significance of Quasiperiodicity)</span></p>

Quasiperiodicity is significant because it is a new type of long-term behavior. Unlike the earlier entries (fixed point, closed orbit, homoclinic orbit and heteroclinic cycles), quasiperiodicity occurs only on the torus.

</div>

#### Coupled System

Now consider the coupled case where $K_1, K_2 > 0$. The dynamics can be deciphered by looking at the **phase difference** $\phi = \theta_1 - \theta_2$. Then

$$\dot{\phi} = \omega_1 - \omega_2 - (K_1 + K_2)\sin\phi,$$

which is just the nonuniform oscillator studied in Section 4.3. By drawing the standard picture, we see that there are two fixed points for $\phi$ if $\lvert\omega_1 - \omega_2\rvert < K_1 + K_2$ and none if $\lvert\omega_1 - \omega_2\rvert > K_1 + K_2$. A saddle-node bifurcation occurs when $\lvert\omega_1 - \omega_2\rvert = K_1 + K_2$.

Suppose for now that there are two fixed points, defined implicitly by

$$\sin\phi^* = \frac{\omega_1 - \omega_2}{K_1 + K_2}.$$

All trajectories of the $\phi$-equation asymptotically approach the stable fixed point. Therefore, back on the torus, the trajectories approach a stable **phase-locked** solution in which the oscillators are separated by a constant phase difference $\phi^*$. The phase-locked solution is *periodic*; in fact, both oscillators run at a constant frequency given by $\omega^* = \dot{\theta}_1 = \dot{\theta}_2 = \omega_2 + K_2\sin\phi^*$. Substituting for $\sin\phi^*$ yields

$$\omega^* = \frac{K_1\omega_2 + K_2\omega_1}{K_1 + K_2}.$$

This is called the **compromise frequency** because it lies between the natural frequencies of the two oscillators.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Frequency Shifts and Coupling Strength)</span></p>

The compromise is not generally halfway; instead the frequencies are shifted by an amount proportional to the coupling strengths, as shown by the identity

$$\frac{\lvert\Delta\omega_1\rvert}{\lvert\Delta\omega_2\rvert} \equiv \frac{\lvert\omega_1 - \omega^*\rvert}{\lvert\omega_2 - \omega^*\rvert} = \frac{K_1}{K_2}.$$

The stable and unstable locked solutions appear as diagonal lines of slope 1 on the square, since $\dot{\theta}_1 = \dot{\theta}_2 = \omega^*$.

</div>

If we pull the natural frequencies apart, say by detuning one of the oscillators, then the locked solutions approach each other and coalesce when $\lvert\omega_1 - \omega_2\rvert = K_1 + K_2$. Thus the locked solution is destroyed in a **saddle-node bifurcation of cycles** (Section 8.4). After the bifurcation, the flow is like that in the uncoupled case studied earlier: we have either quasiperiodic or rational flow, depending on the parameters. The only difference is that now the trajectories on the square are curvy, not straight.

### 8.7 Poincaré Maps

In Section 8.5 we used a Poincaré map to prove the existence of a periodic orbit for the driven pendulum and Josephson junction. Now we discuss Poincaré maps more generally.

Poincaré maps are useful for studying swirling flows, such as the flow near a periodic orbit (or as we'll see later, the flow in some chaotic systems). Consider an $n$-dimensional system $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$. Let $S$ be an $n-1$ dimensional **surface of section** (or **Poincaré section**). $S$ is required to be transverse to the flow, i.e., all trajectories starting on $S$ flow through it, not parallel to it.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Poincaré Map)</span></p>

The **Poincaré map** $P$ is a mapping from $S$ to itself, obtained by following trajectories from one intersection with $S$ to the next. If $\mathbf{x}_k \in S$ denotes the $k$th intersection, then the Poincaré map is defined by

$$\mathbf{x}_{k+1} = P(\mathbf{x}_k).$$

Suppose that $\mathbf{x}^*$ is a **fixed point** of $P$, i.e., $P(\mathbf{x}^*) = \mathbf{x}^*$. Then a trajectory starting at $\mathbf{x}^*$ returns to $\mathbf{x}^*$ after some time $T$, and is therefore a **closed orbit** for the original system $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$. Moreover, by looking at the behavior of $P$ near this fixed point, we can determine the stability of the closed orbit. Thus the Poincaré map converts problems about closed orbits (which are difficult) into problems about fixed points of a mapping (which are easier in principle, though not always in practice). The snag is that it's typically impossible to find a formula for $P$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.7.1</span><span class="math-callout__name">(Poincaré Map in Polar Coordinates)</span></p>

Consider the vector field given in polar coordinates by $\dot{r} = r(1 - r^2)$, $\dot{\theta} = 1$. Let $S$ be the positive $x$-axis, and compute the Poincaré map. Show that the system has a unique periodic orbit and classify its stability.

**Solution:** Let $r_0$ be an initial condition on $S$. Since $\dot{\theta} = 1$, the first return to $S$ occurs after a time of flight $t = 2\pi$. Then $r_1 = P(r_0)$, where $r_1$ satisfies

$$\int_{r_0}^{r_1} \frac{dr}{r(1 - r^2)} = \int_0^{2\pi} dt = 2\pi.$$

Evaluation of the integral (Exercise 8.7.1) yields $r_1 = \left[1 + e^{-4\pi}(r_0^{-2} - 1)\right]^{-1/2}$. Hence

$$P(r) = \left[1 + e^{-4\pi}(r^{-2} - 1)\right]^{-1/2}.$$

A fixed point occurs at $r^* = 1$ where the graph intersects the $45°$ line. The **cobweb** construction enables us to iterate the map graphically. Given an input $r_k$, draw a vertical line until it intersects the graph of $P$; that height is the output $r_{k+1}$. To iterate, we make $r_{k+1}$ the new input by drawing a horizontal line until it intersects the $45°$ diagonal line. Then repeat the process.

The cobweb shows that the fixed point $r^* = 1$ is stable and unique. No surprise, since we knew from Example 7.1.1 that this system has a stable limit cycle at $r = 1$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.7.2</span><span class="math-callout__name">(Sinusoidally Forced RC Circuit)</span></p>

A sinusoidally forced $RC$-circuit can be written in dimensionless form as $\ddot{x} + x = A\sin\omega t$, where $\omega > 0$. Using a Poincaré map, show that this system has a unique, globally stable limit cycle.

**Solution:** This is one of the few time-dependent systems discussed in this book. Such systems can always be made time-independent by adding a new variable. Here we introduce $\theta = \omega t$ and regard the system as a vector field on a cylinder: $\dot{\theta} = \omega$, $\dot{x} + x = A\sin\theta$. Any vertical line on the cylinder is an appropriate section $S$; we choose $S = \lbrace(\theta, x) : \theta = 0 \bmod 2\pi\rbrace$. Consider an initial condition on $S$ given by $\theta(0) = 0$, $x(0) = x_0$. Then the time of flight between successive intersections is $t = 2\pi/\omega$. In physical terms, we strobe the system once per drive cycle and look at the consecutive values of $x$.

To compute $P$, we need to solve the differential equation. Its general solution is a sum of homogeneous and particular solutions: $x(t) = c_1 e^{-t} + c_2\sin\omega t + c_3\cos\omega t$. The constants $c_2$ and $c_3$ can be found explicitly, but the important point is that they depend on $A$ and $\omega$ but *not* on the initial condition $x_0$; only $c_1$ depends on $x_0$. To make the dependence on $x_0$ explicit, observe that at $t = 0$, $x = x_0 = c_1 + c_3$. Thus

$$x(t) = (x_0 - c_3)e^{-t} + c_2\sin\omega t + c_3\cos\omega t.$$

Then $P$ is defined by $x_1 = P(x_0) = x(2\pi/\omega)$. Substitution yields

$$P(x_0) = x_0 e^{-2\pi/\omega} + c_4,$$

where $c_4 = c_3(1 - e^{-2\pi/\omega})$.

The graph of $P$ is a straight line with slope $e^{-2\pi/\omega} < 1$. Since $P$ has slope less than 1, it intersects the diagonal at a unique point. Furthermore, the cobweb shows that the deviation of $x_k$ from the fixed point is reduced by a constant factor with each iteration. Hence the fixed point is unique and globally stable.

In physical terms, the circuit always settles into the same forced oscillation, regardless of the initial conditions. This is a familiar result from elementary physics, looked at in a new way.

</div>

#### Linear Stability of Periodic Orbits

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Linearized Poincaré Map and Floquet Multipliers)</span></p>

Now consider the general case: given a system $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$ with a closed orbit, how can we tell whether the orbit is stable or not? Equivalently, we ask whether the corresponding fixed point $\mathbf{x}^*$ of the Poincaré map is stable.

Let $\mathbf{v}_0$ be an infinitesimal perturbation such that $\mathbf{x}^* + \mathbf{v}_0$ is in $S$. Then after the first return to $S$,

$$\mathbf{x}^* + \mathbf{v}_1 = P(\mathbf{x}^* + \mathbf{v}_0) = P(\mathbf{x}^*) + [DP(\mathbf{x}^*)]\mathbf{v}_0 + O\!\left(\lVert\mathbf{v}_0\rVert^2\right),$$

where $DP(\mathbf{x}^*)$ is an $(n-1) \times (n-1)$ matrix called the **linearized Poincaré map** at $\mathbf{x}^*$. Since $\mathbf{x}^* = P(\mathbf{x}^*)$, we get $\mathbf{v}_1 = [DP(\mathbf{x}^*)]\mathbf{v}_0$, assuming we can neglect the small $O(\lVert\mathbf{v}_0\rVert^2)$ terms. Iterating the linearized map $k$ times gives

$$\mathbf{v}_k = \sum_{j=1}^{n-1} v_j (\lambda_j)^k \mathbf{e}_j,$$

where $\lbrace\mathbf{e}_j\rbrace$ are the eigenvectors and $\lambda_j$ the eigenvalues of $DP(\mathbf{x}^*)$ (assuming no repeated eigenvalues).

The stability criterion is: **the closed orbit is linearly stable if and only if $\lvert\lambda_j\rvert < 1$ for all $j = 1, \dots, n-1$.**

The $\lambda_j$ are called the **characteristic multipliers** or **Floquet multipliers** of the periodic orbit. (Strictly speaking, there is always an additional trivial multiplier $\lambda \equiv 1$ corresponding to perturbations *along* the periodic orbit. We have ignored such perturbations since they just amount to time-translation.)

A borderline case occurs when the largest eigenvalue has magnitude $\lvert\lambda_m\rvert = 1$; this occurs at bifurcations of periodic orbits, and then a nonlinear stability analysis is required.

In general, the characteristic multipliers can only be found by numerical integration (see Exercise 8.7.10). The following examples are two of the rare exceptions.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.7.3</span><span class="math-callout__name">(Characteristic Multiplier for a Simple Limit Cycle)</span></p>

Find the characteristic multiplier for the limit cycle of Example 8.7.1.

**Solution:** We linearize about the fixed point $r^* = 1$ of the Poincaré map. Let $r = 1 + \eta$, where $\eta$ is infinitesimal. Then $\dot{r} = \dot{\eta} = (1 + \eta)(1 - (1 + \eta)^2)$. After neglecting $O(\eta^2)$ terms, we get $\dot{\eta} = -2\eta$. Thus $\eta(t) = \eta_0 e^{-2t}$. After a time of flight $t = 2\pi$, the new perturbation is $\eta_1 = e^{-4\pi}\eta_0$. Hence $e^{-4\pi}$ is the characteristic multiplier. Since $\lvert e^{-4\pi}\rvert < 1$, the limit cycle is linearly stable.

For this simple two-dimensional system, the linearized Poincaré map degenerates to a $1 \times 1$ matrix, i.e., a number. Exercise 8.7.1 asks you to show explicitly that $P'(r^*) = e^{-4\pi}$, as expected from the general theory above.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.7.4</span><span class="math-callout__name">(Coupled Josephson Junctions — In-Phase Solution)</span></p>

The $N$-dimensional system

$$\dot{\phi}_i = \Omega + a\sin\phi_i + \frac{1}{N}\sum_{j=1}^{N}\sin\phi_j,$$

for $i = 1, \dots, N$, describes the dynamics of a series array of overdamped Josephson junctions in parallel with a resistive load (Tsang et al. 1991). For technological reasons, there is great interest in the solution where all the junctions oscillate in phase. This **in-phase solution** is given by $\phi_1(t) = \phi_2(t) = \dots = \phi_N(t) = \phi^*(t)$, where $\phi^*(t)$ denotes the common waveform. Find conditions under which the in-phase solution is periodic, and calculate the characteristic multipliers of this solution.

**Solution:** For the in-phase solution, all $N$ equations reduce to

$$\frac{d\phi^*}{dt} = \Omega + (a + 1)\sin\phi^*.$$

This has a periodic solution (on the circle) if and only if $\lvert\Omega\rvert > \lvert a + 1\rvert$. To determine the stability of the in-phase solution, let $\phi_i(t) = \phi^*(t) + \eta_i(t)$, where the $\eta_i(t)$ are infinitesimal perturbations. Then substituting $\phi_i$ into the original system and dropping quadratic terms in $\eta$ yields

$$\dot{\eta}_i = [a\cos\phi^*(t)]\eta_i + [\cos\phi^*(t)]\frac{1}{N}\sum_{j=1}^{N}\eta_j.$$

We don't have $\phi^*(t)$ explicitly, but that doesn't matter, thanks to two tricks. First, the linear system decouples if we change variables to

$$\mu = \frac{1}{N}\sum_{j=1}^{N}\eta_j, \qquad \xi_i = \eta_{i+1} - \eta_i, \quad i = 1, \dots, N-1.$$

Then $\dot{\xi}_i = [a\cos\phi^*(t)]\xi_i$. Separation of variables yields

$$\frac{d\xi_i}{\xi_i} = [a\cos\phi^*]\,dt = \frac{[a\cos\phi^*]\,d\phi^*}{\Omega + (a+1)\sin\phi^*},$$

where we've used the governing equation for $\phi^*$ to eliminate $dt$ (that was the second trick).

Now we compute the change in the perturbations after one circuit around the closed orbit $\phi^*$:

$$\oint \frac{d\xi_i}{\xi_i} = \int_0^{2\pi}\frac{[a\cos\phi^*]\,d\phi^*}{\Omega + (a+1)\sin\phi^*} \implies \ln\frac{\xi_i(T)}{\xi_i(0)} = \frac{a}{a+1}\ln\left[\Omega + (a+1)\sin\phi^*\right]_0^{2\pi} = 0.$$

Hence $\xi_i(T) = \xi_i(0)$. Similarly, one can show that $\mu(T) = \mu(0)$. Thus $\eta_i(T) = \eta_i(0)$ for all $i$; all perturbations are unchanged after one cycle! Therefore all the characteristic multipliers $\lambda_j = 1$.

This calculation shows that the in-phase state is (linearly) **neutrally stable**. That's discouraging technologically — one would like the array to lock into coherent oscillation, thereby greatly increasing the output power over that available from a single junction.

Since the calculation above is based on linearization, you might wonder whether the neglected nonlinear terms could stabilize the in-phase state. In fact they don't: a reversibility argument shows that the in-phase state is not attracting, even if the nonlinear terms are kept (Exercise 8.7.11).

</div>

## Chapter 9: Lorenz Equations

### 9.0 Introduction

We begin our study of chaos with the **Lorenz equations**

$$\dot{x} = \sigma(y - x), \qquad \dot{y} = rx - y - xz, \qquad \dot{z} = xy - bz.$$

Here $\sigma, r, b > 0$ are parameters. Ed Lorenz (1963) derived this three-dimensional system from a drastically simplified model of convection rolls in the atmosphere. The same equations also arise in models of lasers and dynamos.

Lorenz discovered that this simple-looking deterministic system could have extremely erratic dynamics: over a wide range of parameters, the solutions oscillate irregularly, never exactly repeating but always remaining in a bounded region of phase space. When he plotted the trajectories in three dimensions, he discovered that they settled onto a complicated set, now called a **strange attractor**. Unlike stable fixed points and limit cycles, the strange attractor is not a point or a curve or even a surface — it's a fractal, with a fractional dimension between 2 and 3.

### 9.1 A Chaotic Waterwheel

A neat mechanical model of the Lorenz equations was invented by Willem Malkus and Lou Howard at MIT in the 1970s. The simplest version is a toy waterwheel with leaky paper cups suspended from its rim.

Water is poured in steadily from the top. Three qualitatively different regimes arise depending on the flow rate:

1. **Low flow rate:** The top cups never fill up enough to overcome friction, so the wheel remains motionless.
2. **Moderate flow rate:** The top cup gets heavy enough to start the wheel turning. Eventually the wheel settles into a steady rotation in one direction or the other. By symmetry, rotation in either direction is equally possible; the outcome depends on initial conditions.
3. **High flow rate:** The motion becomes chaotic — the wheel rotates one way for a few turns, then some cups get too full and the wheel doesn't have enough inertia to carry them over the top, so the wheel slows down and may even reverse its direction. The wheel keeps changing direction erratically.

#### Notation

The key variables and parameters describing the wheel's motion:

| Symbol | Meaning |
| --- | --- |
| $\theta$ | angle in the lab frame ($\theta = 0 \leftrightarrow$ 12:00) |
| $\omega(t)$ | angular velocity of the wheel |
| $m(\theta, t)$ | mass distribution of water around the rim |
| $Q(\theta)$ | inflow rate at position $\theta$ |
| $r$ | radius of the wheel |
| $K$ | leakage rate |
| $\nu$ | rotational damping rate |
| $I$ | moment of inertia of the wheel |

The unknowns are $m(\theta, t)$ and $\omega(t)$.

#### Conservation of Mass

Consider any sector $[\theta_1, \theta_2]$ fixed in space. The mass in that sector is $M(t) = \int_{\theta_1}^{\theta_2} m(\theta, t)\,d\theta$. The change $\Delta M$ has four contributions:

1. Mass pumped in by nozzles: $\left[\int_{\theta_1}^{\theta_2} Q\,d\theta\right]\Delta t$.
2. Mass that leaks out: $\left[-\int_{\theta_1}^{\theta_2} Km\,d\theta\right]\Delta t$ (leakage is proportional to $m$).
3. Mass carried in by rotation at $\theta_1$: $m(\theta_1)\omega\Delta t$.
4. Mass carried out at $\theta_2$: $-m(\theta_2)\omega\Delta t$.

Since this holds for all $\theta_1$ and $\theta_2$, we obtain the **continuity equation**

$$\frac{\partial m}{\partial t} = Q - Km - \omega\frac{\partial m}{\partial \theta}.$$

This is a partial differential equation, unlike all the others considered so far in the book.

#### Torque Balance

The rotation of the wheel is governed by Newton's law $F = ma$, expressed as a balance between the applied torques and the rate of change of angular momentum. After the transients decay, the equation of motion is

$$I\dot{\omega} = \text{damping torque} + \text{gravitational torque}.$$

The damping torque is $-\nu\omega$ (where $\nu > 0$), and the gravitational torque is like that of an inverted pendulum (water is pumped in at the top). Letting $g$ denote the effective gravitational constant ($g = g_0 \sin\alpha$, where $\alpha$ is the tilt of the wheel from horizontal), the torque balance equation becomes

$$I\dot{\omega} = -\nu\omega + gr\int_0^{2\pi} m(\theta, t)\sin\theta\,d\theta.$$

This is an **integro-differential equation** because it involves both derivatives and integrals.

#### Amplitude Equations

Since $m(\theta, t)$ is periodic in $\theta$, we can write it as a Fourier series

$$m(\theta, t) = \sum_{n=0}^{\infty}\left[a_n(t)\sin n\theta + b_n(t)\cos n\theta\right].$$

The inflow is also expanded as $Q(\theta) = \sum_{n=0}^{\infty} q_n \cos n\theta$ (no $\sin n\theta$ terms because water is added symmetrically at the top).

Substituting into the continuity equation and the torque balance equation, and using orthogonality, we obtain:

$$\dot{a}_n = n\omega b_n - Ka_n, \qquad \dot{b}_n = -n\omega a_n - Kb_n + q_n.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Miracle of Fourier Analysis)</span></p>

When we substitute the Fourier series into the torque balance equation, only one term survives in the integral by orthogonality:

$$I\dot{\omega} = -\nu\omega + gr\int_0^{2\pi} a_1\sin^2\theta\,d\theta = -\nu\omega + \pi gra_1.$$

Hence only $a_1$ enters the equation for $\dot{\omega}$. But then the equations for $a_1$, $b_1$, and $\omega$ form a **closed system** — these three variables are decoupled from all the other $a_n$, $b_n$ for $n \neq 1$. The resulting equations are

$$\dot{a}_1 = \omega b_1 - Ka_1, \qquad \dot{b}_1 = -\omega a_1 - Kb_1 + q_1, \qquad \dot{\omega} = (-\nu\omega + \pi gra_1)/I.$$

The original pair of integro-partial differential equations has boiled down to a three-dimensional system. It turns out that this system is equivalent to the Lorenz equations (via a change of variables).

</div>

#### Fixed Points

Setting all the derivatives equal to zero yields two kinds of fixed points:

1. **No rotation** ($\omega = 0$): Then $a_1 = 0$ and $b_1 = q_1/K$. The fixed point $(a_1^*, b_1^*, \omega^*) = (0,\; q_1/K,\; 0)$ corresponds to a state of no rotation, with inflow balanced by leakage.

2. **Steady rotation** ($\omega \neq 0$): Solving the system yields $(\omega^*)^2 = \frac{\pi grq_1}{\nu} - K^2$. These solutions exist if and only if

$$\frac{\pi grq_1}{K^2\nu} > 1.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Rayleigh Number)</span></p>

The dimensionless group $\frac{\pi grq_1}{K^2\nu}$ is called the **Rayleigh number**. It measures how hard the system is being driven (by gravity $g$ and inflow $q_1$), relative to the dissipation (leakage $K$ and damping $\nu$). Steady rotation is possible only if the Rayleigh number is large enough (greater than 1).

The Rayleigh number also appears in fluid mechanics, notably convection, where it is proportional to the temperature difference across a heated layer of fluid. For small Rayleigh numbers, heat is conducted vertically and the fluid remains motionless. Past a critical value, an instability occurs and convection rolls form — completely analogous to the steady rotation of the waterwheel.

</div>

### 9.2 Simple Properties of the Lorenz Equations

Lorenz followed a systematic approach: he took the analysis as far as possible using standard techniques, and one by one eliminated all the known possibilities for the long-term behavior of his system.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lorenz Equations)</span></p>

The **Lorenz equations** are

$$\dot{x} = \sigma(y - x), \qquad \dot{y} = rx - y - xz, \qquad \dot{z} = xy - bz.$$

Here $\sigma, r, b > 0$ are parameters: $\sigma$ is the **Prandtl number**, $r$ is the **Rayleigh number**, and $b$ is related to the aspect ratio of the convection rolls.

</div>

**Nonlinearity.** The system has only two nonlinearities: the quadratic terms $xy$ and $xz$. This should remind us of the waterwheel equations, which had the two nonlinearities $\omega a_1$ and $\omega b_1$.

#### Symmetry

There is an important **symmetry** in the Lorenz equations. If we replace $(x, y) \to (-x, -y)$, the equations stay the same. Hence, if $(x(t), y(t), z(t))$ is a solution, so is $(-x(t), -y(t), z(t))$. In other words, all solutions are either symmetric themselves, or have a symmetric partner.

#### Volume Contraction

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Dissipative Nature of the Lorenz System)</span></p>

The Lorenz system is **dissipative**: volumes in phase space contract under the flow. For a general three-dimensional system $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$, the rate of change of a volume $V(t)$ enclosed by a surface $S(t)$ evolving with the flow is

$$\dot{V} = \int_V \nabla \cdot \mathbf{f}\,dV.$$

For the Lorenz system,

$$\nabla \cdot \mathbf{f} = \frac{\partial}{\partial x}[\sigma(y - x)] + \frac{\partial}{\partial y}[rx - y - xz] + \frac{\partial}{\partial z}[xy - bz] = -\sigma - 1 - b < 0.$$

Since the divergence is constant, $\dot{V} = -(\sigma + 1 + b)V$, which has solution $V(t) = V(0)e^{-(\sigma + 1 + b)t}$. Thus **volumes in phase space shrink exponentially fast**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Consequences of Volume Contraction)</span></p>

If we start with an enormous solid blob of initial conditions, it eventually shrinks to a limiting set of zero volume. All trajectories starting in the blob end up somewhere in this limiting set; it consists of fixed points, limit cycles, or for some parameter values, a strange attractor.

Volume contraction imposes strong constraints on the possible solutions of the Lorenz equations.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.2.1</span><span class="math-callout__name">(No Quasiperiodic Solutions)</span></p>

Show that there are no quasiperiodic solutions of the Lorenz equations.

**Solution:** By contradiction. If there were a quasiperiodic solution, it would have to lie on the surface of a torus, and this torus would be invariant under the flow. Hence the volume inside the torus would be constant in time. But this contradicts the fact that all volumes shrink exponentially fast. $\blacksquare$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.2.2</span><span class="math-callout__name">(No Repelling Fixed Points or Closed Orbits)</span></p>

Show that the Lorenz system cannot have repelling fixed points or repelling closed orbits. (By *repelling*, we mean that *all* nearby trajectories are driven away.)

**Solution:** Repellers are incompatible with volume contraction because they are sources of volume. Encase a repeller with a closed surface of initial conditions (a small sphere around a fixed point, or a thin tube around a closed orbit). A short time later, the surface will have expanded as the corresponding trajectories are driven away. Thus the volume inside the surface would increase — contradicting the fact that all volumes contract. $\blacksquare$

</div>

By elimination, all fixed points must be sinks or saddles, and closed orbits (if they exist) must be stable or saddle-like.

#### Fixed Points

Like the waterwheel, the Lorenz system has two types of fixed points. The origin $(x^*, y^*, z^*) = (0, 0, 0)$ is a fixed point for all values of the parameters — it corresponds to the motionless state of the waterwheel. For $r > 1$, there is also a symmetric pair of fixed points

$$C^+\text{ and }C^-: \quad x^* = y^* = \pm\sqrt{b(r-1)}, \quad z^* = r - 1.$$

These represent left- or right-turning convection rolls (analogous to the steady rotations of the waterwheel). As $r \to 1^+$, $C^+$ and $C^-$ coalesce with the origin in a **pitchfork bifurcation**.

#### Linear Stability of the Origin

The linearization at the origin is $\dot{x} = \sigma(y - x)$, $\dot{y} = rx - y$, $\dot{z} = -bz$, obtained by omitting the $xy$ and $xz$ nonlinearities. The equation for $z$ is decoupled and shows that $z(t) \to 0$ exponentially fast. The other two directions are governed by

$$\begin{pmatrix}\dot{x}\\\dot{y}\end{pmatrix} = \begin{pmatrix}-\sigma & \sigma \\ r & -1\end{pmatrix}\begin{pmatrix}x\\y\end{pmatrix},$$

with trace $\tau = -\sigma - 1 < 0$ and determinant $\Delta = \sigma(1 - r)$. If $r > 1$, the origin is a **saddle point** (since $\Delta < 0$). Including the decaying $z$-direction, the saddle has one outgoing and two incoming directions. If $r < 1$, all directions are incoming and the origin is a **stable node** (since $\tau^2 - 4\Delta = (\sigma - 1)^2 + 4\sigma r > 0$).

#### Global Stability of the Origin

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Global Stability for $r < 1$)</span></p>

For $r < 1$, **every** trajectory approaches the origin as $t \to \infty$; the origin is **globally stable**. Hence there can be no limit cycles or chaos for $r < 1$.

**Proof:** Construct the Liapunov function $V(x, y, z) = \frac{1}{\sigma}x^2 + y^2 + z^2$. The surfaces of constant $V$ are concentric ellipsoids about the origin. Calculate:

$$\tfrac{1}{2}\dot{V} = \tfrac{1}{\sigma}x\dot{x} + y\dot{y} + z\dot{z} = (yx - x^2) + (ryx - y^2 - xyz) + (zxy - bz^2).$$

The $xyz$ terms cancel, and after completing the square in the first two terms:

$$\tfrac{1}{2}\dot{V} = -\left[x - \tfrac{r+1}{2}y\right]^2 - \left[1 - \left(\tfrac{r+1}{2}\right)^2\right]y^2 - bz^2.$$

For $r < 1$, each term on the right-hand side is strictly negative unless $(x, y, z) = (0, 0, 0)$. Hence $\dot{V} < 0$ everywhere except at the origin, so $V$ decreases along all non-trivial trajectories, and the origin is globally stable.

</div>

#### Stability of $C^+$ and $C^-$

For $r > 1$, the fixed points $C^+$ and $C^-$ exist. Their stability calculation (Exercise 9.2.1) shows that they are linearly stable for

$$1 < r < r_H = \frac{\sigma(\sigma + b + 3)}{\sigma - b - 1}$$

(assuming $\sigma - b - 1 > 0$). At $r = r_H$, the fixed points $C^+$ and $C^-$ lose stability in a **Hopf bifurcation**.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Subcritical Hopf Bifurcation)</span></p>

What happens immediately after the bifurcation, for $r$ slightly greater than $r_H$? One might suppose that $C^+$ and $C^-$ would each be surrounded by a small stable limit cycle. That would occur if the Hopf bifurcation were supercritical. But actually it's **subcritical** — the limit cycles are **unstable** and exist only for $r < r_H$.

For $r < r_H$ the phase portrait near $C^+$ shows the fixed point encircled by a **saddle cycle** — an unstable limit cycle that is possible only in phase spaces of three or more dimensions. As $r \to r_H$ from below, the saddle cycle shrinks down around the fixed point. At the Hopf bifurcation, the fixed point absorbs the saddle cycle and becomes a saddle point.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Paradox at $r > r_H$)</span></p>

For $r > r_H$, trajectories must fly away to a distant attractor. But the partial bifurcation diagram shows no hint of any stable objects for $r > r_H$. All trajectories are confined to a bounded region and attracted to a set of zero volume, yet there are no stable fixed points, no stable limit cycles, and (by volume contraction) no quasiperiodic solutions. Furthermore, Lorenz gave a persuasive argument that for $r$ slightly greater than $r_H$, any limit cycles would have to be unstable. So the trajectories must have a bizarre kind of long-term behavior — like balls in a pinball machine, repelled from one unstable object after another, yet confined to a bounded set of zero volume.

</div>

### 9.3 Chaos on a Strange Attractor

Lorenz used numerical integration to see what the trajectories would do in the long run. He studied the particular case $\sigma = 10$, $b = 8/3$, $r = 28$. This value of $r$ is just past the Hopf bifurcation value $r_H = \sigma(\sigma + b + 3)/(\sigma - b - 1) \approx 24.74$, so he knew that something strange had to occur.

Starting from the initial condition $(0, 1, 0)$, close to the saddle point at the origin, the solution $y(t)$ settles into an irregular oscillation that persists as $t \to \infty$ but never repeats exactly. The motion is **aperiodic**.

When $x(t)$ is plotted against $z(t)$, a butterfly pattern appears. The trajectory starts near the origin, swings to the right, dives into the center of a spiral on the left, spirals outward, shoots back over to the right, spirals around, and so on indefinitely. The number of circuits made on either side varies unpredictably from one cycle to the next, with the characteristics of a random sequence.

#### The Strange Attractor

When the trajectory is viewed in all three dimensions, it appears to settle onto an exquisitely thin set that looks like a pair of butterfly wings. This limiting set is the **strange attractor** (a term coined by Ruelle and Takens (1971)). It is the attracting set of zero volume whose existence was deduced in Section 9.2.

The geometrical structure of the strange attractor is remarkable. It appears to be a pair of surfaces that merge into one. But since trajectories can't cross or merge (by uniqueness), the two surfaces only *appear* to merge. As Lorenz (1963) explained: each surface is really a pair of surfaces, so where they appear to merge there are really four surfaces. Continuing, there are really eight surfaces, etc., leading to an infinite complex of surfaces — each extremely close to one or the other of two merging surfaces.

Today this "infinite complex of surfaces" would be called a **fractal**. It is a set of points with zero volume but infinite surface area, with a dimension of about 2.05.

#### Exponential Divergence of Nearby Trajectories

The motion on the attractor exhibits **sensitive dependence on initial conditions**. Two trajectories starting very close together will rapidly diverge from each other, and thereafter have totally different futures. This is why long-term prediction becomes impossible in a system like this — small uncertainties are amplified enormously fast.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sensitive Dependence on Initial Conditions)</span></p>

Suppose $\mathbf{x}(t)$ is a point on the attractor and $\mathbf{x}(t) + \boldsymbol{\delta}(t)$ is a nearby point, where $\boldsymbol{\delta}$ is a tiny separation vector with $\lVert\boldsymbol{\delta}_0\rVert = 10^{-15}$, say. In numerical studies of the Lorenz attractor, one finds that

$$\lVert\boldsymbol{\delta}(t)\rVert \sim \lVert\boldsymbol{\delta}_0\rVert e^{\lambda t}$$

where $\lambda \approx 0.9$. Hence **neighboring trajectories separate exponentially fast**. The constant $\lambda$ is called the **Lyapunov exponent**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Implications for Predictability)</span></p>

Sensitive dependence on initial conditions is the hallmark of chaos. Since all real measurements have finite precision, there is always some initial uncertainty $\lVert\boldsymbol{\delta}_0\rVert > 0$, no matter how small. Because the uncertainty grows exponentially, after a time of order $t \sim \frac{1}{\lambda}\ln\frac{a}{\lVert\boldsymbol{\delta}_0\rVert}$ (where $a$ is a characteristic size of the attractor), our predictions become useless. Even though the system is deterministic, long-term prediction is effectively impossible.

</div>

After a time $t$, the discrepancy grows to $\lVert\boldsymbol{\delta}(t)\rVert \sim \lVert\boldsymbol{\delta}_0\rVert e^{\lambda t}$. Let $a$ be a measure of our tolerance — if a prediction is within $a$ of the true state, we consider it acceptable. Then our prediction becomes intolerable when $\lVert\boldsymbol{\delta}(t)\rVert \geq a$; this occurs after a time

$$t_{\text{horizon}} \sim O\!\left(\frac{1}{\lambda}\ln\frac{a}{\lVert\boldsymbol{\delta}_0\rVert}\right).$$

The logarithmic dependence on $\lVert\boldsymbol{\delta}_0\rVert$ is what hurts us. No matter how hard we work to reduce the initial measurement error, we can't predict longer than a few multiples of $1/\lambda$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.3.1</span><span class="math-callout__name">(Prediction Horizon)</span></p>

Suppose we're trying to predict the future state of a chaotic system to within a tolerance of $a = 10^{-3}$. Given that our estimate of the initial state is uncertain to within $\lVert\boldsymbol{\delta}_0\rVert = 10^{-7}$, for about how long can we predict? Now suppose we improve our initial error to $\lVert\boldsymbol{\delta}_0\rVert = 10^{-13}$ (a millionfold improvement). How much longer can we predict?

**Solution:** The original prediction has $t_{\text{horizon}} \approx \frac{1}{\lambda}\ln\frac{10^{-3}}{10^{-7}} = \frac{1}{\lambda}\ln(10^4) = \frac{4\ln 10}{\lambda}$. The improved prediction has $t_{\text{horizon}} \approx \frac{1}{\lambda}\ln\frac{10^{-3}}{10^{-13}} = \frac{1}{\lambda}\ln(10^{10}) = \frac{10\ln 10}{\lambda}$.

Thus, after a millionfold improvement in our initial uncertainty, we can predict only $10/4 = 2.5$ times longer! $\blacksquare$

</div>

#### Defining Chaos

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Chaos)</span></p>

No definition of the term *chaos* is universally accepted, but almost everyone would agree on the following working definition:

**Chaos** is *aperiodic long-term behavior in a deterministic system that exhibits sensitive dependence on initial conditions*.

1. "Aperiodic long-term behavior" means that there are trajectories which do not settle down to fixed points, periodic orbits, or quasiperiodic orbits as $t \to \infty$. For practical reasons, such trajectories should not be too rare — for instance, they should occur with nonzero probability, given a random initial condition.
2. "Deterministic" means that the system has no random or noisy inputs or parameters. The irregular behavior arises from the system's nonlinearity, rather than from noisy driving forces.
3. "Sensitive dependence on initial conditions" means that nearby trajectories separate exponentially fast, i.e., the system has a positive Lyapunov exponent.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.3.2</span><span class="math-callout__name">(Instability Is Not Chaos)</span></p>

Some people think that chaos is just a fancy word for instability. For instance, the system $\dot{x} = x$ is deterministic and shows exponential separation of nearby trajectories. Should we call this system chaotic?

**Solution:** No. Trajectories are repelled to infinity, and never return. So infinity acts like an attracting fixed point. Chaotic behavior should be aperiodic, and that excludes fixed points as well as periodic behavior. $\blacksquare$

</div>

#### Defining Attractor and Strange Attractor

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Attractor)</span></p>

An **attractor** is a closed set $A$ with the following properties:

1. $A$ is an **invariant set**: any trajectory $\mathbf{x}(t)$ that starts in $A$ stays in $A$ for all time.
2. $A$ **attracts an open set of initial conditions**: there is an open set $U$ containing $A$ such that if $\mathbf{x}(0) \in U$, then the distance from $\mathbf{x}(t)$ to $A$ tends to zero as $t \to \infty$. The largest such $U$ is called the **basin of attraction** of $A$.
3. $A$ is **minimal**: there is no proper subset of $A$ that satisfies conditions 1 and 2.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.3.3</span><span class="math-callout__name">(Invariant Set vs. Attractor)</span></p>

Consider the system $\dot{x} = x - x^3$, $\dot{y} = -y$. Let $I$ denote the interval $-1 \leq x \leq 1$, $y = 0$. Is $I$ an invariant set? Does it attract an open set of initial conditions? Is it an attractor?

**Solution:** The phase portrait has stable fixed points at $(\pm 1, 0)$ and a saddle point at the origin. Any trajectory that starts in $I$ stays in $I$ forever (since $y(0) = 0$ implies $y(t) = 0$ for all $t$), so condition 1 is satisfied. Moreover, $I$ attracts all trajectories in the $xy$ plane, so condition 2 is satisfied.

But $I$ is **not** an attractor because it is not minimal. The stable fixed points $(\pm 1, 0)$ are proper subsets of $I$ that also satisfy conditions 1 and 2. These points are the only attractors for the system. $\blacksquare$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Strange Attractor)</span></p>

A **strange attractor** is an attractor that exhibits sensitive dependence on initial conditions. Strange attractors were originally called strange because they are often fractal sets. Nowadays the geometric property (fractal structure) is regarded as less important than the dynamical property (sensitive dependence on initial conditions). The terms **chaotic attractor** and **fractal attractor** are used when one wishes to emphasize one or the other of those aspects.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Minimality and the Lorenz Attractor)</span></p>

Even if a certain set attracts all trajectories, it may fail to be an attractor because it may not be minimal — it may contain one or more smaller attractors. The same could be true for the Lorenz equations. Although all trajectories are attracted to a bounded set of zero volume, that set is not necessarily an attractor since it might not be minimal. Doubts about this delicate issue lingered for many years, but were eventually laid to rest in 1999, when Warwick Tucker proved that the Lorenz equations do, in fact, have a strange attractor (Tucker 1999, 2002).

</div>

### 9.4 Lorenz Map

Lorenz (1963) found a beautiful way to analyze the dynamics on his strange attractor. He directs our attention to a particular view of the attractor and writes:

> the trajectory apparently leaves one spiral only after exceeding some critical distance from the center. Moreover, the extent to which this distance is exceeded appears to determine the point at which the next spiral is entered; this in turn seems to determine the number of circuits to be executed before changing spirals again.

The "single feature" that Lorenz focuses on is $z_n$, the $n$th local maximum of $z(t)$. His idea is that $z_n$ should predict $z_{n+1}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lorenz Map)</span></p>

The function $z_{n+1} = f(z_n)$ is called the **Lorenz map**. Lorenz numerically integrated the equations for a long time, measured the local maxima of $z(t)$, and plotted $z_{n+1}$ vs. $z_n$. The remarkable result is that *the data from the chaotic time series appear to fall neatly on a curve* — there is almost no "thickness" to the graph.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Nature of the Lorenz Map)</span></p>

Several important clarifications about the Lorenz map:

1. The graph does have some thickness — strictly speaking, $f(z)$ is not a well-defined function, because there can be more than one output $z_{n+1}$ for a given input $z_n$. But the thickness is so small, and there is so much to be gained by treating the graph as a curve, that we proceed with this approximation (keeping in mind that the subsequent analysis is plausible but not rigorous).

2. The Lorenz map is reminiscent of a Poincaré map (Section 8.7), in that both reduce the analysis of a differential equation to an iterated map. But the Lorenz map characterizes the trajectory by only *one* number, not two. This simpler approach works only if the attractor is very "flat," i.e., close to two-dimensional, as the Lorenz attractor is.

</div>

#### Ruling Out Stable Limit Cycles

The key observation is that the graph of the Lorenz map satisfies

$$\lvert f'(z)\rvert > 1$$

everywhere. This property ultimately implies that if any limit cycles exist, they are necessarily **unstable**.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.4.1</span><span class="math-callout__name">(All Closed Orbits Are Unstable)</span></p>

Given the Lorenz map approximation $z_{n+1} = f(z_n)$, with $\lvert f'(z)\rvert > 1$ for all $z$, show that *all* closed orbits are unstable.

**Solution:** Consider the sequence $\lbrace z_n\rbrace$ corresponding to an arbitrary closed orbit. Since the orbit eventually closes, the sequence must eventually repeat: $z_{n+p} = z_n$ for some integer $p \geq 1$ (here $p$ is the **period** of the sequence, and $z_n$ is a **period-$p$ point**).

To show instability, consider a small deviation $\eta_n$ from the closed orbit, where $z_n = z^* + \eta_n$. After one iteration, $\eta_{n+1} \approx f'(z_n)\eta_n$ by linearization. After $p$ iterations:

$$\eta_{n+p} \approx \left[\prod_{k=0}^{p-1} f'(z_{n+k})\right]\eta_n.$$

Each factor in the product has absolute value greater than 1 (since $\lvert f'(z)\rvert > 1$ for all $z$). Hence $\lvert\eta_{n+p}\rvert > \lvert\eta_n\rvert$, which proves that the closed orbit is unstable. $\blacksquare$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Tucker's Theorem)</span></p>

Since the Lorenz map is not truly a well-defined function (its graph has some thickness), the argument above wouldn't convince a hypothetical skeptic who insists the attractor might just be a stable limit cycle of incredibly long period. The matter was finally laid to rest in 1999, when graduate student Warwick Tucker proved that the Lorenz equations do, in fact, have a strange attractor (Tucker 1999, 2002). Tucker's theorem dispels any lingering concerns that numerical errors might be deceiving us — the strange attractor and the chaotic motion are genuine properties of the Lorenz equations themselves.

</div>

### 9.5 Exploring Parameter Space

So far we have concentrated on the particular parameter values $\sigma = 10$, $b = 8/3$, $r = 28$. There is a vast three-dimensional parameter space to explore. To simplify matters, many investigators have kept $\sigma = 10$ and $b = 8/3$ while varying $r$.

#### Bifurcation Diagram

The behavior for small values of $r$ can be summarized as follows:

| Range of $r$ | Behavior |
| --- | --- |
| $r < 1$ | Origin is globally stable |
| $r = 1$ | Supercritical pitchfork bifurcation: $C^+$, $C^-$ are born |
| $1 < r < 13.926$ | $C^+$, $C^-$ are stable; no limit cycles exist |
| $r \approx 13.926$ | Homoclinic bifurcation: unstable limit cycles are born |
| $13.926 < r < 24.06$ | $C^+$, $C^-$ are stable; transient chaos possible |
| $r \approx 24.06$ | Strange attractor appears (from the complicated invariant set) |
| $24.06 < r < 24.74$ | Coexistence: strange attractor **and** stable fixed points $C^+$, $C^-$ |
| $r_H \approx 24.74$ | Subcritical Hopf bifurcation: $C^+$, $C^-$ lose stability |
| $r > r_H$ | Strange attractor (for most values of $r$) |

As we decrease $r$ from $r_H$, the unstable limit cycles born at the Hopf bifurcation expand and pass precariously close to the saddle point at the origin. At $r \approx 13.926$ the cycles touch the saddle point and become homoclinic orbits; hence we have a **homoclinic bifurcation**. Below $r = 13.926$ there are no limit cycles.

At the homoclinic bifurcation, an amazingly complicated invariant set is born — a thicket of infinitely many saddle-cycles and aperiodic orbits. It is not an attractor and is not observable directly, but it generates sensitive dependence on initial conditions in its neighborhood. Trajectories can get hung up near this set, rattling around chaotically for a while, but eventually escape and settle down to $C^+$ or $C^-$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.5.1</span><span class="math-callout__name">(Transient Chaos)</span></p>

Show numerically that the Lorenz equations can exhibit **transient chaos** when $r = 21$ (with $\sigma = 10$ and $b = 8/3$).

**Solution:** The trajectory initially appears to trace out a strange attractor, but eventually it stays on one side and spirals down toward the stable fixed point $C^+$ (or $C^-$, depending on initial conditions). The time series of $y$ vs. $t$ shows the same result: an initially erratic solution that ultimately damps down to equilibrium.

By our definition, this dynamics is not "chaotic" because the long-term behavior is not aperiodic. On the other hand, the dynamics do exhibit sensitive dependence on initial conditions — a slightly different initial condition could cause the trajectory to end up at $C^-$ instead of $C^+$. This shows that a deterministic system can be unpredictable even if its final states are very simple. $\blacksquare$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Coexistence and Hysteresis)</span></p>

For $24.06 < r < 24.74$, there are **two** types of attractors: stable fixed points and a strange attractor. This coexistence means that we can have hysteresis between chaos and equilibrium by varying $r$ slowly back and forth past these two endpoints. It also means that a large enough perturbation can knock a steadily rotating waterwheel into permanent chaos — reminiscent (in spirit, though not detail) of fluid flows that mysteriously become turbulent even though the basic laminar flow is still linearly stable.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.5.2</span><span class="math-callout__name">(Large $r$: Globally Attracting Limit Cycle)</span></p>

Describe the long-term dynamics for large values of $r$, for $\sigma = 10$, $b = 8/3$.

**Solution:** Numerical simulations indicate that the system has a globally attracting limit cycle for all $r > 313$ (Sparrow 1982). For $r = 350$, the trajectory in the $xz$-plane approaches a limit cycle, and the time series of $y$ vs. $t$ shows a periodic oscillation.

This solution predicts that the waterwheel should ultimately rock back and forth like a pendulum, turning once to the right, then back to the left, and so on. This is observed experimentally. $\blacksquare$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Windows of Periodicity)</span></p>

For $r$ between 28 and 313, the story is much more complicated. For most values of $r$ one finds chaos, but there are also small windows of periodic behavior interspersed. The three largest windows are $99.524\ldots < r < 100.795\ldots$; $145 < r < 166$; and $r > 214.4$. The alternating pattern of chaotic and periodic regimes resembles that seen in the logistic map (Chapter 10).

</div>

### 9.6 Using Chaos to Send Secret Messages

One of the most exciting developments in nonlinear dynamics is the realization that chaos can be *useful*. One application involves **private communications**: you mask a secret message with much louder chaos. An outside listener only hears the chaos, which sounds like meaningless noise. But the intended recipient has a magic receiver that perfectly reproduces the chaos — then he can subtract off the chaotic mask and listen to the message.

#### Synchronized Chaos

The key idea, due to Pecora and Carroll (1990), is that two identical chaotic systems can be made to **synchronize** perfectly, even though chaos is ordinarily associated with sensitive dependence on initial conditions.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.6.1</span><span class="math-callout__name">(Proof of Synchronization via Liapunov Function)</span></p>

Consider a transmitter governed by the Lorenz equations (written in scaled variables $u = \frac{1}{10}x$, $v = \frac{1}{10}y$, $w = \frac{1}{20}z$):

$$\dot{u} = \sigma(v - u), \qquad \dot{v} = ru - v - 20uw, \qquad \dot{w} = 5uv - bw.$$

The receiver is an identical Lorenz circuit, except that the drive signal $u(t)$ from the transmitter replaces the receiver's own signal $u_r(t)$ at a crucial place:

$$\dot{u}_r = \sigma(v_r - u_r), \qquad \dot{v}_r = ru(t) - v_r - 20u(t)w_r, \qquad \dot{w}_r = 5u(t)v_r - bw_r.$$

Show that $\mathbf{e}(t) \to \mathbf{0}$ as $t \to \infty$, where $\mathbf{e} = \mathbf{d} - \mathbf{r}$ is the error between driver and receiver.

**Solution:** Subtracting the receiver equations from the transmitter equations yields the error dynamics:

$$\dot{e}_1 = \sigma(e_2 - e_1), \qquad \dot{e}_2 = -e_2 - 20u(t)e_3, \qquad \dot{e}_3 = 5u(t)e_2 - be_3.$$

This is a linear system for $\mathbf{e}(t)$, but it has a chaotic time-dependent coefficient $u(t)$ in two terms. The idea is to construct a Liapunov function in such a way that *the chaos cancels out*. Multiply the second equation by $e_2$ and the third by $4e_3$ and add:

$$e_2\dot{e}_2 + 4e_3\dot{e}_3 = -e_2^2 - 20u(t)e_2e_3 + 20u(t)e_2e_3 - 4be_3^2 = -e_2^2 - 4be_3^2.$$

The chaotic term disappears! The left-hand side is $\frac{1}{2}\frac{d}{dt}(e_2^2 + 4e_3^2)$. This suggests the Liapunov function

$$E(\mathbf{e}, t) = \tfrac{1}{2}\!\left(\tfrac{1}{\sigma}e_1^2 + e_2^2 + 4e_3^2\right).$$

$E$ is positive definite. Computing $\dot{E}$:

$$\dot{E} = \tfrac{1}{\sigma}e_1\dot{e}_1 + e_2\dot{e}_2 + 4e_3\dot{e}_3 = -\left[e_1 - \tfrac{1}{2}e_2\right]^2 - \tfrac{3}{4}e_2^2 - 4be_3^2.$$

Hence $\dot{E} \leq 0$, with equality only if $\mathbf{e} = \mathbf{0}$. Therefore $E$ is a Liapunov function, and $\mathbf{e} = \mathbf{0}$ is globally asymptotically stable — the **receiver asymptotically approaches perfect synchrony with the transmitter, starting from any initial conditions**. $\blacksquare$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Chaotic Communication)</span></p>

The practical implementation uses an electronic circuit built from resistors, capacitors, operational amplifiers, and analog multiplier chips. The voltages $u$, $v$, $w$ at three different points in the circuit are proportional to Lorenz's $x$, $y$, $z$, so the circuit acts as an analog computer for the Lorenz equations. The transmitter adds a message (e.g., a speech signal) to the chaotic signal $u(t)$. When this masked signal is sent to the receiver, its output synchronizes almost perfectly to the original chaos, and after instant electronic subtraction, the original message is recovered with only a tiny amount of distortion.

A stronger result is possible: the error $\mathbf{e}(t)$ decays *exponentially fast* (Cuomo, Oppenheim, and Strogatz 1993), which is important because rapid synchronization is necessary for the desired application.

</div>

## Chapter 10: One-Dimensional Maps

### 10.0 Introduction

This chapter deals with a new class of dynamical systems in which time is *discrete*, rather than continuous. These systems are known variously as difference equations, recursion relations, iterated maps, or simply **maps**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(One-Dimensional Map)</span></p>

A **one-dimensional map** is a rule of the form $x_{n+1} = f(x_n)$, where $f$ is a function from $\mathbb{R}$ to $\mathbb{R}$. The points $x_n$ belong to the one-dimensional space of real numbers. The sequence $x_0, x_1, x_2, \ldots$ is called the **orbit** starting from $x_0$.

</div>

Maps arise in various ways:

1. **As tools for analyzing differential equations.** Poincaré maps prove the existence and stability of periodic solutions (Section 8.7). The Lorenz map (Section 9.4) provided strong evidence that the Lorenz attractor is truly strange.
2. **As models of natural phenomena.** In digital electronics, finance theory, impulsively driven mechanical systems, and certain animal populations where successive generations do not overlap, time is naturally discrete.
3. **As simple examples of chaos.** Maps are capable of much wilder behavior than differential equations because the points $x_n$ *hop* along their orbits rather than flow continuously.

### 10.1 Fixed Points and Cobwebs

We develop tools for analyzing one-dimensional maps of the form $x_{n+1} = f(x_n)$, where $f$ is a smooth function from the real line to itself.

#### Fixed Points and Linear Stability

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fixed Point and Multiplier of a Map)</span></p>

A point $x^*$ satisfying $f(x^*) = x^*$ is a **fixed point** of the map $x_{n+1} = f(x_n)$, since the orbit remains at $x^*$ for all future iterations. To determine its stability, consider a nearby orbit $x_n = x^* + \eta_n$. Substitution and Taylor expansion yield

$$\eta_{n+1} = f'(x^*)\eta_n + O(\eta_n^2).$$

The quantity $\lambda = f'(x^*)$ is called the **eigenvalue** or **multiplier** of the fixed point. The linearized map is $\eta_{n+1} = \lambda\eta_n$, with solution $\eta_n = \lambda^n\eta_0$. Hence:

* If $\lvert\lambda\rvert = \lvert f'(x^*)\rvert < 1$, then $\eta_n \to 0$ and the fixed point is **linearly stable**.
* If $\lvert\lambda\rvert = \lvert f'(x^*)\rvert > 1$, the fixed point is **unstable**.
* If $\lvert\lambda\rvert = 1$, the linearization is inconclusive (**marginal case**).

Fixed points with multiplier $\lambda = 0$ are called **superstable** because perturbations decay like $\eta_n \sim \eta_0^{(2^n)}$, which is much faster than the usual $\eta_n \sim \lambda^n\eta_0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.1.1</span><span class="math-callout__name">(Fixed Points of $x_{n+1} = x_n^2$)</span></p>

Find the fixed points of the map $x_{n+1} = x_n^2$ and determine their stability.

**Solution:** The fixed points satisfy $x^* = (x^*)^2$, so $x^* = 0$ or $x^* = 1$. The multiplier is $\lambda = f'(x^*) = 2x^*$. At $x^* = 0$: $\lvert\lambda\rvert = 0 < 1$, so it is stable (in fact, superstable). At $x^* = 1$: $\lvert\lambda\rvert = 2 > 1$, so it is unstable. $\blacksquare$

</div>

#### Cobwebs

The **cobweb** construction provides a graphical method for iterating a map. Given $x_{n+1} = f(x_n)$ and an initial condition $x_0$:

1. Draw a vertical line from $x_0$ on the horizontal axis up to the graph of $f$; the height is $x_1 = f(x_0)$.
2. Draw a horizontal line from that point to the diagonal $x_{n+1} = x_n$; this transfers the output $x_1$ to the horizontal axis.
3. Repeat: go vertically to the curve, then horizontally to the diagonal.

Cobwebs reveal global behavior at a glance, supplementing the local information from linearization.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Cobweb Geometry and the Sign of $\lambda$)</span></p>

The convergence pattern depends on the sign of the multiplier $\lambda = f'(x^*)$:
* If $\lambda < 0$: the cobweb spirals into (or away from) the fixed point — convergence occurs through **damped oscillations**.
* If $\lambda > 0$: the cobweb staircase monotonically approaches (or recedes from) the fixed point.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.1.2</span><span class="math-callout__name">(Marginal Case: $x_{n+1} = \sin x_n$)</span></p>

Consider the map $x_{n+1} = \sin x_n$. Show that the fixed point $x^* = 0$ is globally stable, even though linear analysis is inconclusive.

**Solution:** The multiplier at $x^* = 0$ is $f'(0) = \cos(0) = 1$, which is the marginal case. However, a cobweb diagram shows that the orbit slowly rattles down a narrow channel and heads monotonically toward the fixed point. To see that the stability is global, note that for any $x_0$, the first iterate $x_1 = \sin x_0$ satisfies $-1 \leq x_1 \leq 1$ (since $\lvert\sin x\rvert \leq 1$). Within the interval $[-1, 1]$, the cobweb converges to 0, so all orbits satisfy $x_n \to 0$. $\blacksquare$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.1.3</span><span class="math-callout__name">(The Cosine Map: $x_{n+1} = \cos x_n$)</span></p>

Given $x_{n+1} = \cos x_n$, how does $x_n$ behave as $n \to \infty$?

**Solution:** Pressing the cosine button repeatedly on a calculator, you find that $x_n \to 0.739\ldots$, regardless of starting point. This is the unique solution of $x = \cos x$, and it corresponds to a fixed point of the map. A cobweb diagram shows that a typical orbit spirals into the fixed point $x^* = 0.739\ldots$ as $n \to \infty$ — the spiraling indicates damped oscillations, characteristic of fixed points with $\lambda < 0$. $\blacksquare$

</div>

### 10.2 Logistic Map: Numerics

In a fascinating review article, Robert May (1976) emphasized that even simple nonlinear maps could have very complicated dynamics.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Logistic Map)</span></p>

The **logistic map** is

$$x_{n+1} = rx_n(1 - x_n),$$

a discrete-time analog of the logistic equation for population growth (Section 2.3). Here $x_n \geq 0$ is a dimensionless measure of the population in the $n$th generation and $r \geq 0$ is the intrinsic growth rate. The graph is a parabola with maximum value $r/4$ at $x = 1/2$. We restrict $r$ to the range $0 \leq r \leq 4$ so that the map sends the interval $[0, 1]$ into itself.

</div>

#### Period-Doubling

The qualitative behavior of the logistic map changes dramatically as $r$ increases:

| Range of $r$ | Long-term behavior |
| --- | --- |
| $r < 1$ | Population goes extinct: $x_n \to 0$ |
| $1 < r < 3$ | Population reaches a nonzero steady state |
| $r = 3$ | Onset of oscillation |
| $3 < r < 3.449\ldots$ | **Period-2 cycle**: population alternates between two values |
| $3.449\ldots < r < 3.54409\ldots$ | **Period-4 cycle** |
| $3.54409\ldots < r < 3.5644\ldots$ | **Period-8 cycle** |
| $\vdots$ | Further **period-doublings** to cycles of period $16, 32, \ldots$ |
| $r_\infty = 3.569946\ldots$ | Onset of chaos (period $\to \infty$) |

The successive bifurcations come faster and faster. Let $r_n$ denote the value of $r$ where a $2^n$-cycle first appears. Then the $r_n$ converge to the limiting value $r_\infty$ in a geometric fashion: the distance between successive transitions shrinks by a constant factor

$$\delta = \lim_{n\to\infty}\frac{r_n - r_{n-1}}{r_{n+1} - r_n} = 4.669\ldots$$

This remarkable number $\delta$ is called **Feigenbaum's constant** and will be discussed further in Section 10.6.

#### Chaos and Periodic Windows

For many values of $r > r_\infty$, the sequence $\lbrace x_n\rbrace$ never settles down to a fixed point or a periodic orbit — instead the long-term behavior is aperiodic. This is a discrete-time version of the chaos encountered in the Lorenz equations (Chapter 9).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Orbit Diagram)</span></p>

The **orbit diagram** plots the system's attractor as a function of $r$. To generate it: for each value of $r$, iterate the logistic map from a random initial condition $x_0$ for about 300 cycles (to let transients decay), then plot the next several hundred iterates $x_n$ above that value of $r$. Sweeping across all $r$ produces the iconic diagram of nonlinear dynamics.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Structure of the Orbit Diagram)</span></p>

The orbit diagram for $3.4 \leq r \leq 4$ reveals:

* At $r = 3.4$, the attractor is a period-2 cycle (two branches).
* As $r$ increases, both branches split simultaneously, yielding period-4, then period-8, etc. — the **period-doubling cascade**.
* At $r = r_\infty \approx 3.57$, the map becomes chaotic and the attractor changes from a finite to an infinite set of points.
* For $r > r_\infty$, the orbit diagram reveals an unexpected mixture of order and chaos, with **periodic windows** interspersed between chaotic clouds of dots.
* The large window beginning near $r \approx 3.83$ contains a stable period-3 cycle.
* A blow-up of the period-3 window reveals a copy of the orbit diagram reappearing in miniature! This **self-similarity** is a hallmark of fractals.

</div>

### 10.3 Logistic Map: Analysis

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.3.1</span><span class="math-callout__name">(Fixed Points of the Logistic Map)</span></p>

Find all the fixed points of the logistic map $x_{n+1} = rx_n(1 - x_n)$ for $0 \leq x_n \leq 1$ and $0 \leq r \leq 4$, and determine their stability.

**Solution:** The fixed points satisfy $x^* = rx^*(1 - x^*)$, giving $x^* = 0$ or $x^* = 1 - 1/r$. The origin is a fixed point for all $r$; the second fixed point $x^* = 1 - 1/r$ lies in $[0, 1]$ only if $r \geq 1$.

Stability depends on the multiplier $f'(x^*) = r - 2rx^*$:

* At $x^* = 0$: $f'(0) = r$. Stable for $r < 1$, unstable for $r > 1$.
* At $x^* = 1 - 1/r$: $f'(x^*) = r - 2r(1 - 1/r) = 2 - r$. Stable for $-1 < 2 - r < 1$, i.e., $1 < r < 3$. Unstable for $r > 3$. $\blacksquare$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bifurcations of the Logistic Map)</span></p>

At $r = 1$, the fixed point $x^*$ bifurcates from the origin in a **transcritical bifurcation**: for $r < 1$ the parabola lies below the diagonal and the origin is the only fixed point; for $r > 1$ the parabola intersects the diagonal in a second fixed point $x^* = 1 - 1/r$, while the origin loses stability.

At $r = 3$, the multiplier at $x^*$ reaches $f'(x^*) = -1$. This is called a **flip bifurcation** — it is often associated with period-doubling. Indeed, the flip bifurcation at $r = 3$ spawns a 2-cycle.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.3.2</span><span class="math-callout__name">(Existence of the 2-Cycle)</span></p>

Show that the logistic map has a 2-cycle for all $r > 3$.

**Solution:** A 2-cycle consists of two points $p$ and $q$ such that $f(p) = q$ and $f(q) = p$. Equivalently, $p$ is a fixed point of the **second-iterate map** $f^2(x) \equiv f(f(x))$. Since $f$ is a quadratic polynomial, $f^2(x)$ is a quartic polynomial. The fixed points $x^* = 0$ and $x^* = 1 - 1/r$ are trivially solutions of $f^2(x) = x$ (since $f(x^*) = x^*$ implies $f^2(x^*) = x^*$). Factoring these out, the problem reduces to a quadratic equation. The roots are:

$$p, q = \frac{r + 1 \pm \sqrt{(r-3)(r+1)}}{2r}.$$

These are real for $r > 3$, confirming that a 2-cycle exists for all $r > 3$. At $r = 3$, the roots coincide and equal $x^* = 1 - 1/r = 2/3$, showing that the 2-cycle bifurcates continuously from $x^*$. For $r < 3$ the roots are complex, so no 2-cycle exists. $\blacksquare$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.3.3</span><span class="math-callout__name">(Stability of the 2-Cycle)</span></p>

Show that the 2-cycle of Example 10.3.2 is stable for $3 < r < 1 + \sqrt{6} = 3.449\ldots$.

**Solution:** To analyze the stability of a cycle, reduce it to a question about the stability of a fixed point: both $p$ and $q$ are fixed points of the second-iterate map $f^2(x)$. The original 2-cycle is stable precisely if $p$ and $q$ are stable fixed points for $f^2$.

The multiplier of $f^2$ at $p$ is

$$\lambda = \frac{d}{dx}f^2(x)\Big\rvert_{x=p} = f'(f(p))\cdot f'(p) = f'(q)\cdot f'(p).$$

(Note: the same $\lambda$ is obtained at $x = q$, by symmetry. Hence the $p$ and $q$ branches bifurcate simultaneously.) Substituting $f'(x) = r(1 - 2x)$ and the expressions for $p$ and $q$:

$$\lambda = r(1 - 2q)\cdot r(1 - 2p) = r^2[1 - 2(p+q) + 4pq] = 4 + 2r - r^2.$$

Therefore the 2-cycle is linearly stable for $\lvert 4 + 2r - r^2\rvert < 1$, which gives $3 < r < 1 + \sqrt{6}$. $\blacksquare$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Flip Bifurcations: Supercritical and Subcritical)</span></p>

A cobweb diagram reveals how flip bifurcations give rise to period-doubling. Near a fixed point where $f'(x^*) \approx -1$, if the graph of $f$ is concave down, the cobweb tends to produce a small, stable 2-cycle close to the fixed point (supercritical flip bifurcation). But like pitchfork bifurcations, flip bifurcations can also be subcritical, in which case the 2-cycle exists *below* the bifurcation and is *unstable*.

</div>

### 10.4 Periodic Windows

One of the most intriguing features of the orbit diagram is the occurrence of periodic windows for $r > r_\infty$. The period-3 window that occurs near $3.8284\ldots \leq r \leq 3.8415\ldots$ is the most conspicuous. Suddenly, against a backdrop of chaos, a stable 3-cycle appears out of the blue.

#### Birth of the Period-3 Window

The key to understanding the 3-cycle is the **third-iterate map** $f^3(x)$. Any point $p$ in a period-3 cycle satisfies $p = f^3(p)$ and is therefore a fixed point of $f^3$. Since $f^3(x)$ is an eighth-degree polynomial, it has up to eight intersections with the diagonal. Six of these are genuine period-3 points (marked with dots); the other two are impostors — the fixed points $x^* = 0$ and $x^* = 1 - 1/r$, which trivially satisfy $f^3(x^*) = x^*$.

At $r = 3.835$, the graph of $f^3(x)$ crosses the diagonal at six points — three corresponding to a stable 3-cycle (where the slope of $f^3$ is shallow) and three corresponding to an unstable 3-cycle (where the slope exceeds 1).

As $r$ decreases toward the chaotic regime, the hills of $f^3(x)$ move down and the valleys rise up, pulling the curve away from the diagonal. At the critical value $r = 1 + \sqrt{8} = 3.8284\ldots$, the graph of $f^3(x)$ becomes **tangent** to the diagonal. At this point, the stable and unstable period-3 cycles coalesce and annihilate in a **tangent bifurcation**. This transition defines the beginning of the periodic window.

#### Intermittency

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Intermittency)</span></p>

For $r$ just below the period-3 window, the system exhibits a distinctive kind of chaos called **intermittency** (Pomeau and Manneville 1980). The orbit alternates between long stretches of nearly periodic behavior (the "ghost" of the 3-cycle that no longer exists) and intermittent bouts of chaos.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Mechanism of Intermittency)</span></p>

The geometry underlying intermittency involves the three narrow channels between the diagonal and the graph of $f^3(x)$. These channels were formed in the aftermath of the tangent bifurcation, as the hills and valleys of $f^3(x)$ pulled away from the diagonal. The orbit takes many iterations to squeeze through each channel; during the passage, $f^3(x_n) \approx x_n$ and the orbit looks like a 3-cycle (we are seeing the "ghost" of the vanished cycle).

Eventually the orbit escapes from the channel and bounces around chaotically until fate sends it back into a channel at some unpredictable later time and place. As $r$ is moved farther from the periodic window, the channels widen and the chaotic bursts become more frequent until the system is fully chaotic. This progression is known as the **intermittency route to chaos**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Period-Doubling Within the Window)</span></p>

Inside the period-3 window, a miniature copy of the orbit diagram reappears. Just after the stable 3-cycle is created in the tangent bifurcation, the slope of $f^3(x)$ at the black dots is close to $+1$. As $r$ increases, the hills rise and the valleys sink. The slope of $f^3(x)$ at the stable cycle points decreases steadily. When the slope reaches $-1$, a flip bifurcation occurs, spawning a period-6 cycle. Further period-doublings to period 12, 24, etc. follow in a cascade that mirrors the original period-doubling route seen in the full orbit diagram. This self-similar structure is characteristic of one-dimensional maps.

</div>

### 10.5 Liapunov Exponent

The logistic map can exhibit aperiodic orbits for certain parameter values, but how do we know that this is really chaos? To be called "chaotic," a system should also show **sensitive dependence on initial conditions**, in the sense that neighboring orbits separate exponentially fast, on average. In Section 9.3 we quantified sensitive dependence by defining the Liapunov exponent for a chaotic differential equation. Now we extend the definition to one-dimensional maps.

Given an initial condition $x_0$, consider a nearby point $x_0 + \delta_0$, where the initial separation $\delta_0$ is extremely small. Let $\delta_n$ be the separation after $n$ iterates. If $|\delta_n| \approx |\delta_0| e^{n\lambda}$, then $\lambda$ is called the **Liapunov exponent**. A positive Liapunov exponent is a signature of chaos.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Liapunov Exponent for One-Dimensional Maps)</span></p>

A more precise formula for $\lambda$ can be derived. By taking logarithms and noting that $\delta_n = f^n(x_0 + \delta_0) - f^n(x_0)$, we obtain

$$\lambda \approx \frac{1}{n}\ln\left|\frac{\delta_n}{\delta_0}\right| = \frac{1}{n}\ln\left|\frac{f^n(x_0 + \delta_0) - f^n(x_0)}{\delta_0}\right| = \frac{1}{n}\ln|(f^n)'(x_0)|,$$

where we have taken the limit $\delta_0 \to 0$ in the last step. By the chain rule, $(f^n)'(x_0) = \prod_{i=0}^{n-1} f'(x_i)$. If the limit as $n \to \infty$ exists, we define the **Liapunov exponent** for the orbit starting at $x_0$:

$$\lambda = \lim_{n \to \infty}\left\lbrace \frac{1}{n}\sum_{i=0}^{n-1}\ln|f'(x_i)|\right\rbrace.$$

Note that $\lambda$ depends on $x_0$. However, it is the same for all $x_0$ in the basin of attraction of a given attractor. For stable fixed points and cycles, $\lambda$ is negative; for chaotic attractors, $\lambda$ is positive.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.5.1</span><span class="math-callout__name">(Liapunov Exponent for Stable Cycles)</span></p>

Suppose that $f$ has a stable $p$-cycle containing the point $x_0$. Show that the Liapunov exponent $\lambda < 0$. If the cycle is superstable, show that $\lambda = -\infty$.

**Solution:** As usual, we convert questions about $p$-cycles of $f$ into questions about fixed points of $f^p$. Since $x_0$ is an element of a $p$-cycle, $x_0$ is a fixed point of $f^p$. By assumption, the cycle is stable; hence the multiplier $|(f^p)'(x_0)| < 1$. Therefore $\ln|(f^p)'(x_0)| < \ln(1) = 0$, a result that we'll use in a moment.

Next observe that for a $p$-cycle,

$$\lambda = \lim_{n\to\infty}\left\lbrace\frac{1}{n}\sum_{i=0}^{n-1}\ln|f'(x_i)|\right\rbrace = \frac{1}{p}\sum_{i=0}^{p-1}\ln|f'(x_i)|$$

since the same $p$ terms keep appearing in the infinite sum. Finally, using the chain rule in reverse, we obtain

$$\frac{1}{p}\sum_{i=0}^{p-1}\ln|f'(x_i)| = \frac{1}{p}\ln|(f^p)'(x_0)| < 0,$$

as desired. If the cycle is superstable, then $|(f^p)'(x_0)| = 0$ by definition, and thus $\lambda = \frac{1}{p}\ln(0) = -\infty$. $\blacksquare$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.5.2</span><span class="math-callout__name">(Liapunov Exponent of the Tent Map)</span></p>

The **tent map** is defined by

$$f(x) = \begin{cases} rx, & 0 \le x \le \tfrac{1}{2},\\ r - rx, & \tfrac{1}{2} \le x \le 1, \end{cases}$$

for $0 \le r \le 2$ and $0 \le x \le 1$. Because it is piecewise linear, it is far easier to analyze than the logistic map. Show that $\lambda = \ln r$ for the tent map, independent of the initial condition $x_0$.

**Solution:** Since $f'(x) = \pm r$ for all $x$, we find $\lambda = \lim_{n\to\infty}\left\lbrace\frac{1}{n}\sum_{i=0}^{n-1}\ln|f'(x_i)|\right\rbrace = \ln r.$ $\blacksquare$

</div>

Example 10.5.2 suggests that the tent map has chaotic solutions for all $r > 1$, since $\lambda = \ln r > 0$. In fact, the dynamics of the tent map can be understood in complete detail, even in the chaotic regime; see Devaney (1989).

In general, one needs to use a computer to calculate Liapunov exponents.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.5.3</span><span class="math-callout__name">(Numerical Computation of $\lambda$ for the Logistic Map)</span></p>

Describe a numerical scheme to compute $\lambda$ for the logistic map $f(x) = rx(1 - x)$. Graph the results as a function of the control parameter $r$, for $3 \le r \le 4$.

**Solution:** Fix some value of $r$. Then, starting from a random initial condition, iterate the map long enough to allow transients to decay, say 300 iterates or so. Next compute a large number of additional iterates, say 10,000. You only need to store the current value of $x_n$, not all the previous iterates. Compute $\ln|f'(x_n)| = \ln|r - 2rx_n|$ and add it to the sum of the previous logarithms. The Liapunov exponent is then obtained by dividing the grand total by 10,000. Repeat this procedure for the next $r$, and so on.

Comparing the graph of $\lambda(r)$ to the orbit diagram, we notice that $\lambda$ remains negative for $r < r_\infty \approx 3.57$, and approaches zero at the period-doubling bifurcations. The negative spikes correspond to the $2^n$-cycles. The onset of chaos is visible near $r \approx 3.57$, where $\lambda$ first becomes positive. For $r > 3.57$ the Liapunov exponent generally increases, except for the dips caused by the windows of periodic behavior. Note the large dip due to the period-3 window near $r = 3.83$.

All the dips actually drop down to $\lambda = -\infty$, because a superstable cycle is guaranteed to occur somewhere near the middle of each dip, and such cycles have $\lambda = -\infty$ by Example 10.5.1. $\blacksquare$

</div>

### 10.6 Universality and Experiments

This section deals with some of the most astonishing results in all of nonlinear dynamics.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.6.1</span><span class="math-callout__name">(The Sine Map)</span></p>

Plot the graph of the **sine map** $x_{n+1} = r\sin\pi x_n$ for $0 \le r \le 1$ and $0 \le x \le 1$, and compare it to the logistic map. Then plot the orbit diagrams for both maps, and list some similarities and differences.

**Solution:** The graph of the sine map has the same shape as the graph of the logistic map. Both curves are smooth, concave down, and have a single maximum. Such maps are called **unimodal**.

The orbit diagrams for the sine map and the logistic map show an incredible resemblance. Both diagrams have the same vertical scale, but the horizontal axis of the sine map diagram is scaled by a factor of 4. This normalization is appropriate because the maximum of $r\sin\pi x$ is $r$, whereas that of $rx(1-x)$ is $\frac{1}{4}r$.

The *qualitative* dynamics of the two maps are identical. They both undergo period-doubling routes to chaos, followed by periodic windows interwoven with chaotic bands. Even more remarkably, the periodic windows occur in the same order, and with the same relative sizes. For instance, the period-3 window is the largest in both cases, and the next largest windows preceding it are period-5 and period-6.

But there are *quantitative* differences. For instance, the period-doubling bifurcations occur later in the logistic map, and the periodic windows are thinner. $\blacksquare$

</div>

#### Qualitative Universality: The U-Sequence

Example 10.6.1 illustrates a powerful theorem due to Metropolis et al. (1973). They considered all unimodal maps of the form $x_{n+1} = rf(x_n)$, where $f(x)$ also satisfies $f(0) = f(1) = 0$. Metropolis et al. proved that as $r$ is varied, the order in which stable periodic solutions appear is *independent* of the unimodal map being iterated. That is, *the periodic attractors always occur in the same sequence*, now called the universal or **U-sequence**. This amazing result implies that the algebraic form of $f(x)$ is irrelevant; only its overall shape matters.

Up to period 6, the U-sequence is

$$1,\ 2,\ 2\times 2,\ 6,\ 5,\ 3,\ 2\times 3,\ 5,\ 6,\ 4,\ 6,\ 5,\ 6.$$

The beginning of this sequence is familiar: periods 1, 2, and $2 \times 2$ are the first stages in the period-doubling scenario. Next, periods 6, 5, 3 correspond to the large windows mentioned above. Period $2 \times 3$ is the first period-doubling of the period-3 cycle.

The U-sequence has been found in experiments on the Belousov-Zhabotinsky chemical reaction. Simoyi et al. (1982) studied the reaction in a continuously stirred flow reactor and found a regime in which periodic and chaotic states alternate as the flow rate is increased. Within the experimental resolution, the periodic states occurred in the exact order predicted by the U-sequence.

#### Quantitative Universality

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Feigenbaum's $\delta$)</span></p>

Around 1975, Mitchell Feigenbaum began to study period-doubling in the logistic map. He noticed that the bifurcation values $r_n$ converge geometrically, with the distance between successive transitions shrinking by a constant factor of about 4.669. In fact, the same convergence rate appears *no matter what unimodal map is iterated*. In this sense, the number

$$\delta = \lim_{n\to\infty}\frac{r_n - r_{n-1}}{r_{n+1} - r_n} = 4.669\ldots$$

is **universal**. It is a new mathematical constant, as basic to period-doubling as $\pi$ is to circles.

</div>

Let $\Delta_n = r_n - r_{n+1}$ denote the distance between consecutive bifurcation values. Then $\Delta_n / \Delta_{n+1} \to \delta$ as $n \to \infty$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Feigenbaum's $\alpha$)</span></p>

There is also universal scaling in the $x$-direction. Let $x_m$ denote the maximum of $f$, and let $d_n$ denote the distance from $x_m$ to the *nearest* point in a $2^n$-cycle. The nearest point in the $2^n$-cycle is alternately above and below $x_m$, so $d_n$ are alternately positive and negative. Then the ratio $d_n / d_{n+1}$ tends to a universal limit as $n \to \infty$:

$$\frac{d_n}{d_{n+1}} \to \alpha = -2.5029\ldots,$$

independent of the precise form of $f$.

</div>

Feigenbaum went on to develop a beautiful theory that explained why $\alpha$ and $\delta$ are universal. He borrowed the idea of renormalization from statistical physics, and thereby found an analogy between $\alpha$, $\delta$ and the universal exponents observed in experiments on second-order phase transitions in magnets, fluids, and other physical systems. In Section 10.7, we give a brief look at this renormalization theory.

#### Experimental Tests

Since Feigenbaum's work, sequences of period-doubling bifurcations have been measured in a variety of experimental systems.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Libchaber's Convection Experiment)</span></p>

In the convection experiment of Libchaber et al. (1982), a box containing liquid mercury is heated from below. The control parameter is the Rayleigh number $R$, a dimensionless measure of the externally imposed temperature gradient from bottom to top. For $R$ less than a critical value $R_c$, heat is conducted upward while the fluid remains motionless. But for $R > R_c$, the motionless state becomes unstable and **convection** occurs — hot fluid rises on one side, loses its heat at the top, and descends on the other side, setting up a pattern of counterrotating cylindrical **rolls**.

For $R$ just slightly above $R_c$, the rolls are straight and the motion is steady. With more heating, another instability sets in. A wave propagates back and forth along each roll, causing the temperature to oscillate at each point. Further increases in $R$ generate additional period-doublings. By carefully measuring the values of $R$ at the period-doubling bifurcations, Libchaber et al. (1982) arrived at a value of $\delta = 4.4 \pm 0.1$, in reasonable agreement with the theoretical result $\delta \approx 4.669$.

</div>

| Experiment | Number of period doublings | $\delta$ | Authors |
| --- | --- | --- | --- |
| **Hydrodynamic** | | | |
| water | 4 | 4.3(8) | Giglio et al. (1981) |
| mercury | 4 | 4.4(1) | Libchaber et al. (1982) |
| **Electronic** | | | |
| diode | 4 | 4.5(6) | Linsay (1981) |
| diode | 5 | 4.3(1) | Testa et al. (1982) |
| transistor | 4 | 4.7(3) | Arecchi and Lisi (1982) |
| Josephson simul. | 3 | 4.5(3) | Yeh and Kao (1982) |

It is important to understand that these measurements are difficult. Since $\delta \approx 5$, each successive bifurcation requires about a fivefold improvement in the experimenter's ability to measure the external control parameter. Also, experimental noise tends to blur the structure of high-period orbits. Given these difficulties, the agreement between theory and experiment is impressive.

#### What Do 1-D Maps Have to Do with Science?

The predictive power of Feigenbaum's theory may strike you as mysterious. How can it apply to real physical systems like convecting fluids or electronic circuits, given that it deals with one-dimensional maps, not the physics of real systems?

The key idea is to use Lorenz's trick for obtaining a map from a flow (Section 9.4). For a given value of the control parameter, record the successive local maxima of $x(t)$ for a trajectory on the strange attractor. Then plot $x_{n+1}$ vs. $x_n$, where $x_n$ denotes the $n$th local maximum. This **Lorenz map** is an approximate one-dimensional map derived from the continuous-time system.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Rössler System)</span></p>

The **Rössler system** is

$$\dot{x} = -y - z, \qquad \dot{y} = x + ay, \qquad \dot{z} = b + z(x - c),$$

where $a$, $b$, and $c$ are parameters. This system contains only one nonlinear term, $zx$, and is even simpler than the Lorenz system (Chapter 9), which has two nonlinearities.

Two-dimensional projections of the system's attractor for different values of $c$ (with $a = b = 0.2$ held fixed) show that at $c = 2.5$ the attractor is a simple limit cycle. As $c$ is increased to 3.5, the limit cycle goes around twice before closing, and its period is approximately twice that of the original cycle. This is what period-doubling looks like in a continuous-time system! A **period-doubling bifurcation of cycles** must have occurred somewhere between $c = 2.5$ and $c = 3.5$. Another period-doubling bifurcation creates the four-loop cycle shown at $c = 4$. After an infinite cascade of further period-doublings, one obtains the strange attractor shown at $c = 5$.

The Lorenz map for $c = 5$ falls very nearly on a one-dimensional curve. Note the uncanny resemblance to the logistic map! One can even compute an orbit diagram for the Rössler system, which reveals the period-doubling route to chaos and the large period-3 window — all our old friends are here.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Applicability of the Theory)</span></p>

Now we can see why certain physical systems are governed by Feigenbaum's universality theory — if the system's Lorenz map is nearly one-dimensional and unimodal, then the theory applies. This is certainly the case for the Rössler system, and probably for Libchaber's convecting mercury. But not all systems have one-dimensional Lorenz maps. For the Lorenz map to be almost one-dimensional, the strange attractor has to be very flat, i.e., only slightly more than two-dimensional. This requires the system to be highly dissipative; only two or three degrees of freedom are truly active, and the rest follow along slavishly.

So while the theory works for some mildly chaotic systems, it does not apply to fully turbulent fluids or fibrillating hearts, where there are many active degrees of freedom corresponding to complicated behavior in space as well as time.

</div>

### 10.7 Renormalization

This section gives an intuitive introduction to Feigenbaum's (1979) renormalization theory for period-doubling. For nice expositions at a higher mathematical level, see Feigenbaum (1980), Collet and Eckmann (1980), Schuster (1989), Drazin (1992), and Cvitanovic (1989b).

First we introduce some notation. Let $f(x, r)$ denote a unimodal map that undergoes a period-doubling route to chaos as $r$ increases, and suppose that $x_m$ is the maximum of $f$. Let $r_n$ denote the value of $r$ at which a $2^n$-cycle is born, and let $R_n$ denote the value of $r$ at which the $2^n$-cycle is superstable.

Feigenbaum phrased his analysis in terms of the superstable cycles, so let's get some practice with them.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.7.1</span><span class="math-callout__name">(Superstable Fixed Point and 2-Cycle)</span></p>

Find $R_0$ and $R_1$ for the map $f(x, r) = r - x^2$.

**Solution:** At $R_0$ the map has a superstable fixed point, by definition. The fixed point condition is $x^* = R_0 - (x^*)^2$ and the superstability condition is $\lambda = (\partial f/\partial x)_{x = x^*} = -2x$, we must have $x^* = 0$, i.e., the fixed point is the maximum of $f$. Substituting $x^* = 0$ into the fixed point condition yields $R_0 = 0$.

At $R_1$ the map has a superstable 2-cycle. Let $p$ and $q$ denote the points of the cycle. Superstability requires that the multiplier $\lambda = (-2p)(-2q) = 0$, so the point $x = 0$ must be one of the points in the 2-cycle. Then the period-2 condition $f^2(0, R_1) = 0$ implies $f^2(0, R_1) = R_1 - (R_1)^2 = 0$. Hence $R_1 = 1$ (since the other root gives a fixed point, not a 2-cycle). $\blacksquare$

</div>

Example 10.7.1 illustrates a general rule: a superstable cycle of a unimodal map always contains $x_m$ as one of its points. Consequently, there is a simple graphical way to locate $R_n$: draw a horizontal line at height $x_m$ in the orbit diagram; then $R_n$ occurs where this line intersects the **figtree** portion of the orbit diagram (Feigenbaum = *figtree* in German). Note that $R_n$ lies between $r_n$ and $r_{n+1}$. Numerical experiments show that the spacing between successive $R_n$ also shrinks by the universal factor $\delta \approx 4.669$.

#### The Renormalization Idea

The renormalization theory is based on the **self-similarity** of the figtree — the twigs look like the earlier branches, except they are scaled down in both the $x$ and $r$ directions. This structure reflects the endless repetition of the same dynamical processes: a $2^n$-cycle is born, then becomes superstable, and then loses stability in a period-doubling bifurcation.

To express the self-similarity mathematically, we compare $f$ with its second iterate $f^2$ at corresponding values of $r$, and then "renormalize" one map into the other. Specifically, look at the graphs of $f(x, R_0)$ and $f^2(x, R_1)$. This is a fair comparison because the maps have the same stability properties: $x_m$ is a superstable fixed point for both of them. To obtain $f^2(x, R_1)$ we took the second iterate of $f$ *and* increased $r$ from $R_0$ to $R_1$. This $r$-shifting is a basic part of the renormalization procedure.

The key point is that a small box around $x_m$ in the graph of $f^2(x, R_1)$ looks practically identical to the graph of $f(x, R_0)$, except for a change of scale and a reversal of both axes.

After translating the origin to $x_m$, rescaling $x \to x/\alpha$, and shifting $r$ to the next superstable value, the resemblance between $f(x, R_0)$ and $f^2(x/\alpha, R_1)$ shows that

$$f(x, R_0) \approx \alpha\, f^2\!\left(\frac{x}{\alpha},\, R_1\right).$$

In summary, $f$ has been **renormalized** by taking its second iterate, rescaling $x \to x/\alpha$, and shifting $r$ to the next superstable value.

There is no reason to stop at $f^2$. We can renormalize $f^2$ to generate $f^4$; it too has a superstable fixed point if we shift $r$ to $R_2$. The same reasoning yields

$$f^2\!\left(\frac{x}{\alpha},\, R_1\right) \approx \alpha\, f^4\!\left(\frac{x}{\alpha^2},\, R_2\right).$$

When expressed in terms of the original map $f(x, R_0)$, this equation becomes

$$f(x, R_0) \approx \alpha^2 f^4\!\left(\frac{x}{\alpha^2},\, R_2\right).$$

After renormalizing $n$ times we get

$$f(x, R_0) \approx \alpha^n f^{(2^n)}\!\left(\frac{x}{\alpha^n},\, R_n\right).$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Universal Function $g_0$)</span></p>

Feigenbaum found numerically that

$$\lim_{n \to \infty}\alpha^n f^{(2^n)}\!\left(\frac{x}{\alpha^n},\, R_n\right) = g_0(x),$$

where $g_0(x)$ is a **universal function** with a superstable fixed point. The limiting function exists only if $\alpha$ is chosen correctly, specifically, $\alpha = -2.5029\ldots$

Here "universal" means that the limiting function $g_0(x)$ is independent of the original $f$ (almost). This seems incredible at first, but the form of the limit suggests an explanation: $g_0(x)$ depends on $f$ only through its behavior near $x = 0$, since that's all that survives in the argument $x/\alpha^n$ as $n \to \infty$. With each renormalization, we're blowing up a smaller and smaller neighborhood of the maximum of $f$, so practically all information about the global shape of $f$ is lost.

One caveat: the *order* of the maximum is never forgotten. Hence a more precise statement is that $g_0(x)$ is universal for all $f$ *with a quadratic maximum* (the generic case). A different $g_0(x)$ is found for $f$'s with a fourth-degree maximum, etc.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Functional Equation)</span></p>

To obtain other universal functions $g_i(x)$, start with $f(x, R_i)$ instead of $f(x, R_0)$:

$$g_i(x) = \lim_{n\to\infty}\alpha^n f^{(2^n)}\!\left(\frac{x}{\alpha^n},\, R_{n+i}\right).$$

Here $g_i(x)$ is a universal function with a superstable $2^i$-cycle. The case where we start with $R_i = R_\infty$ (at the onset of chaos) is the most interesting and important, since then

$$f(x, R_\infty) \approx \alpha\, f^2\!\left(\frac{x}{\alpha},\, R_\infty\right).$$

For once, we don't have to shift $r$ when we renormalize! The limiting function $g_\infty(x)$, usually called $g(x)$, satisfies

$$g(x) = \alpha\, g^2\!\left(\frac{x}{\alpha}\right).$$

This is a **functional equation** for $g(x)$ and the universal scale factor $\alpha$. It is self-referential: $g(x)$ is defined in terms of itself.

The functional equation is not complete until we specify boundary conditions on $g(x)$. After the shift of origin, all our unimodal $f$'s have a maximum at $x = 0$, so we require $g'(0) = 0$. Also, we can set $g(0) = 1$ without loss of generality. (This just defines the scale for $x$; if $g(x)$ is a solution, so is $\mu g(x/\mu)$, with the same $\alpha$.)

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Solving the Functional Equation)</span></p>

Now we solve for $g(x)$ and $\alpha$. At $x = 0$ the functional equation gives $g(0) = \alpha\, g(g(0))$. But $g(0) = 1$, so $1 = \alpha\, g(1)$. Hence

$$\alpha = 1/g(1),$$

which shows that $\alpha$ is determined by $g(x)$. No one has ever found a closed-form solution for $g(x)$, so we resort to a power series solution

$$g(x) = 1 + c_2 x^2 + c_4 x^4 + \ldots$$

(which assumes that the maximum is quadratic). The coefficients are determined by substituting the power series into the functional equation and matching like powers of $x$. Feigenbaum (1979) used a seven-term expansion, and found $c_2 \approx -1.5276$, $c_4 \approx 0.1048$, along with $\alpha \approx -2.5029$. Thus the renormalization theory has succeeded in explaining the value of $\alpha$ observed numerically.

The theory also explains the value of $\delta$. Unfortunately, that part of the story requires more sophisticated apparatus than we are prepared to discuss (operators in function space, Frechet derivatives, etc.).

</div>

#### Renormalization for Pedestrians

The following pedagogical calculation, modified from May and Oster (1980) and Helleman (1980), is intended to clarify the renormalization process. As a bonus, it gives closed-form approximations for $\alpha$ and $\delta$.

Let $f(x, \mu)$ be any unimodal map that undergoes a period-doubling route to chaos. Suppose that the variables are defined such that the period-2 cycle is born at $x = 0$ when $\mu = 0$. Then for both $x$ and $\mu$ close to 0, the map is approximated by

$$x_{n+1} = -(1 + \mu)x_n + x_n^2 + \ldots,$$

since the eigenvalue is $-1$ at the bifurcation. Without loss of generality we can set $a = 1$ by rescaling $x \to x/a$. So locally our map has the normal form

$$x_{n+1} = -(1 + \mu)x_n + x_n^2 + \ldots.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.7.2</span><span class="math-callout__name">(Period-4 Bifurcation Value)</span></p>

Using the renormalization transformation, calculate the value of $\mu$ at which the original map gives birth to a period-4 cycle. Compare your result to the value $r_2 = 1 + \sqrt{6}$ found for the logistic map in Example 10.3.3.

**Solution:** The period-4 solution is born when $\tilde{\mu} = \mu^2 + 4\mu - 2 = 0$. Solving this quadratic equation yields $\mu = -2 + \sqrt{6}$. (The other solution is negative and is not relevant.) Now recall that the origin of $\mu$ was defined such that $\mu = 0$ at the birth of period-2, which occurs at $r = 3$ for the logistic map. Hence $r_2 = 3 + (-2 + \sqrt{6}) = 1 + \sqrt{6}$, which recovers the result obtained in Example 10.3.3. $\blacksquare$

</div>

Because the renormalized map has the same form as the original map, we can do the same analysis all over again, now regarding the renormalized map as the fundamental map. In other words, we can renormalize *ad infinitum*! This allows us to bootstrap our way to the onset of chaos, using only the **renormalization transformation** $\tilde{\mu} = \mu^2 + 4\mu - 2$.

Let $\mu_k$ denote the parameter value at which the original map gives birth to a $2^k$-cycle. By definition of $\mu$, we have $\mu_1 = 0$; by Example 10.7.2, $\mu_2 = -2 + \sqrt{6} \approx 0.449$. In general, the $\mu_k$ satisfy

$$\mu_{k-1} = \mu_k^2 + 4\mu_k - 2.$$

To convert this into a forward iteration, solve for $\mu_k$ in terms of $\mu_{k-1}$:

$$\mu_k = -2\sqrt{6 + \mu_{k-1}}.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.7.3</span><span class="math-callout__name">(Finding $\mu^*$ and Predicting $r_\infty$)</span></p>

Find $\mu^*$, the stable fixed point of the renormalization transformation.

**Solution:** It is slightly easier to work with the backward form. The fixed point satisfies $\mu^* = (\mu^*)^2 + 4\mu^* - 2$, and is given by

$$\mu^* = \tfrac{1}{2}\left(-3 + \sqrt{17}\right) \approx 0.56.$$

Incidentally, this gives a remarkably accurate prediction of $r_\infty$ for the logistic map. Recall that $\mu = 0$ corresponds to the birth of period-2, which occurs at $r = 3$ for the logistic map. Thus $\mu^*$ corresponds to $r_\infty \approx 3.56$ whereas the actual numerical result is $r_\infty \approx 3.57$! $\blacksquare$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Approximate Values of $\delta$ and $\alpha$)</span></p>

Finally we get to see how $\delta$ and $\alpha$ make their entry. For $k \gg 1$, the $\mu_k$ should converge geometrically to $\mu^*$ at a rate given by the universal constant $\delta$. Hence $\delta \approx (\mu_{k-1} - \mu^*)/(\mu_k - \mu^*)$. As $k \to \infty$, this ratio tends to $0/0$ and therefore may be evaluated by L'Hôpital's rule. The result is

$$\delta \approx \frac{d\mu_{k-1}}{d\mu_k}\bigg\rvert_{\mu = \mu^*} = 2\mu^* + 4.$$

Finally, we substitute for $\mu^*$ and obtain

$$\delta \approx 1 + \sqrt{17} \approx 5.12.$$

This estimate is about 10 percent larger than the true $\delta \approx 4.67$, which is not bad considering our approximations.

To find the approximate $\alpha$, note that we used $C$ as a rescaling parameter when we defined $\tilde{x}_n = C\eta_n$. Hence $C$ plays the role of $\alpha$. Substitution of $\mu^*$ yields

$$C = \frac{1 + \sqrt{17}}{2} - 3\left[\frac{1 + \sqrt{17}}{2}\right]^{1/2} \approx -2.24,$$

which is also within 10 percent of the actual value $\alpha \approx -2.50$.

</div>

## Chapter 11: Fractals

### 11.0 Introduction

Back in Chapter 9, we found that the solutions of the Lorenz equations settle down to a complicated set in phase space — the strange attractor. As Lorenz (1963) realized, the geometry of this set must be very peculiar, something like an "infinite complex of surfaces." In this chapter we develop the ideas needed to describe such strange sets more precisely. The tools come from **fractal geometry**.

Roughly speaking, **fractals** are complex geometric shapes with fine structure at arbitrarily small scales. Usually they have some degree of self-similarity. In other words, if we magnify a tiny part of a fractal, we will see features reminiscent of the whole. Sometimes the similarity is exact; more often it is only approximate or statistical.

Fractals are of great interest because of their exquisite combination of beauty, complexity, and endless structure. They are reminiscent of natural objects like mountains, clouds, coastlines, blood vessel networks, and even broccoli, in a way that classical shapes like cones and squares can't match. They have also turned out to be useful in scientific applications ranging from computer graphics and image compression to the structural mechanics of cracks and the fluid mechanics of viscous fingering.

### 11.1 Countable and Uncountable Sets

This section reviews the parts of set theory that we'll need in later discussions of fractals.

Are some infinities larger than others? Surprisingly, the answer is yes. In the late 1800s, Georg Cantor invented a clever way to compare different infinite sets. Two sets $X$ and $Y$ are said to have the same **cardinality** (or number of elements) if there is an invertible mapping that pairs each element $x \in X$ with precisely one $y \in Y$. Such a mapping is called a **one-to-one correspondence**.

A familiar infinite set is the set of natural numbers $\mathbb{N} = \lbrace 1, 2, 3, 4, \ldots\rbrace$. This set provides a basis for comparison — if another set $X$ can be put into one-to-one correspondence with the natural numbers, then $X$ is said to be **countable**. Otherwise $X$ is **uncountable**.

There is an equivalent characterization: a set $X$ is countable if it can be written as a list $\lbrace x_1, x_2, x_3, \ldots\rbrace$, with every $x \in X$ appearing somewhere in the list.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.1.1</span><span class="math-callout__name">(Even Numbers Are Countable)</span></p>

Show that the set of even natural numbers $E = \lbrace 2, 4, 6, \ldots\rbrace$ is countable.

**Solution:** We need to find a one-to-one correspondence between $E$ and $\mathbb{N}$. Such a correspondence is given by the invertible mapping that pairs each natural number $n$ with the even number $2n$; thus $1 \leftrightarrow 2$, $2 \leftrightarrow 4$, $3 \leftrightarrow 6$, and so on.

Hence there are exactly as many even numbers as natural numbers. You might have thought that there would be only *half* as many, since all the odd numbers are missing! $\blacksquare$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.1.2</span><span class="math-callout__name">(Integers Are Countable)</span></p>

Show that the integers are countable.

**Solution:** Here's an algorithm for listing all the integers: we start with 0 and then work in order of increasing absolute value. Thus the list is $\lbrace 0, 1, -1, 2, -2, 3, -3, \ldots\rbrace$. Any particular integer appears eventually, so the integers are countable. $\blacksquare$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.1.3</span><span class="math-callout__name">(Positive Rationals Are Countable)</span></p>

Show that the positive rational numbers are countable.

**Solution:** Here's a *wrong* way: we start listing the numbers $\frac{1}{1}, \frac{1}{2}, \frac{1}{3}, \frac{1}{4}, \ldots$ in order. Unfortunately we never finish the $\frac{1}{n}$'s and so numbers like $\frac{2}{3}$ are never counted!

The right way is to make a table where the $pq$-th entry is $p/q$. Then the rationals can be counted by the weaving procedure shown in Figure 11.1.1. Any given $p/q$ is reached after a finite number of steps, so the rationals are countable. $\blacksquare$

</div>

Now we consider our first example of an uncountable set.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.1.4</span><span class="math-callout__name">(The Reals Are Uncountable)</span></p>

Let $X$ denote the set of all real numbers between 0 and 1. Show that $X$ is uncountable.

**Solution:** The proof is by contradiction. If $X$ were countable, we could list all the real numbers between 0 and 1 as a set $\lbrace x_1, x_2, x_3, \ldots\rbrace$. Rewrite these numbers in decimal form:

$$x_1 = 0.x_{11}x_{12}x_{13}x_{14}\cdots$$

$$x_2 = 0.x_{21}x_{22}x_{23}x_{24}\cdots$$

$$x_3 = 0.x_{31}x_{32}x_{33}x_{34}\cdots$$

where $x_{ij}$ denotes the $j$th digit of the real number $x_i$.

To obtain a contradiction, we'll show that there's a number $r$ between 0 and 1 that is *not* on the list. We construct $r$ as follows: its first digit is *anything other than* $x_{11}$, the first digit of $x_1$. Similarly, its second digit is anything other than the second digit of $x_2$. In general, the $n$th digit of $r$ is $\overline{x}_{nn}$, defined as any digit other than $x_{nn}$. Then the number $r = 0.\overline{x}_{11}\overline{x}_{22}\overline{x}_{33}\cdots$ is not on the list. Why not? It can't be equal to $x_1$, because it differs from $x_1$ in the first decimal place. Similarly, $r$ differs from $x_2$ in the second decimal place, from $x_3$ in the third decimal place, and so on. Hence $r$ is not on the list, and thus $X$ is uncountable. $\blacksquare$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Cantor's Diagonal Argument)</span></p>

This argument (devised by Cantor) is called the **diagonal argument**, because $r$ is constructed by changing the diagonal entries $x_{nn}$ in the matrix of digits $[x_{ij}]$.

</div>

### 11.2 Cantor Set

Now we turn to another of Cantor's creations, a fractal known as the Cantor set. It is simple and therefore pedagogically useful, but it is also much more than that — as we'll see in Chapter 12, the Cantor set is intimately related to the geometry of strange attractors.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Construction of the Cantor Set)</span></p>

We start with the closed interval $S_0 = [0, 1]$ and remove its open middle third $(\frac{1}{3}, \frac{2}{3})$, leaving the endpoints behind. This produces the pair of closed intervals $S_1 = [0, \frac{1}{3}] \cup [\frac{2}{3}, 1]$. Then we remove the open middle thirds of *those* two intervals to produce $S_2$, and so on. The limiting set $C = S_\infty$ is the **Cantor set**. It consists of an infinite number of infinitesimal pieces, separated by gaps of various sizes.

</div>

#### Fractal Properties of the Cantor Set

The Cantor set $C$ has several properties that are typical of fractals more generally:

1. **$C$ has structure at arbitrarily small scales.** If we enlarge part of $C$ repeatedly, we continue to see a complex pattern of points separated by gaps of various sizes. This structure is neverending, like worlds within worlds. In contrast, when we look at a smooth curve or surface under repeated magnification, the picture becomes more and more featureless.

2. **$C$ is self-similar.** It contains smaller copies of itself at all scales. For instance, if we take the left part of $C$ (the part contained in the interval $[0, \frac{1}{3}]$) and enlarge it by a factor of three, we get $C$ back again. Similarly, the parts of $C$ in each of the four intervals of $S_2$ are geometrically similar to $C$, except scaled down by a factor of nine. Warning: the strict self-similarity of the Cantor set is found only in the simplest fractals. More general fractals are only approximately self-similar.

3. **The dimension of $C$ is not an integer.** As we'll show in Section 11.3, its dimension is actually $\ln 2 / \ln 3 \approx 0.63$! The idea of a noninteger dimension is bewildering at first, but it turns out to be a natural generalization of our intuitive ideas about dimension, and provides a very useful tool for quantifying the structure of fractals.

Two other important properties of the Cantor set are: *$C$ has measure zero* and *it consists of uncountably many points*.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.2.1</span><span class="math-callout__name">(The Cantor Set Has Measure Zero)</span></p>

Show that the **measure** of the Cantor set is zero, in the sense that it can be covered by intervals whose total length is arbitrarily small.

**Solution:** The construction shows that each set $S_n$ completely covers all the sets that come after it in the construction. Hence the Cantor set $C = S_\infty$ is covered by *each* of the sets $S_n$. So the total length of the Cantor set must be less than the total length of $S_n$, for any $n$. Let $L_n$ denote the length of $S_n$. Then from the construction we see that $L_0 = 1$, $L_1 = \frac{2}{3}$, $L_2 = \left(\frac{2}{3}\right)^2$, and in general, $L_n = \left(\frac{2}{3}\right)^n$. Since $L_n \to 0$ as $n \to \infty$, the Cantor set has a total length of zero. $\blacksquare$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.2.2</span><span class="math-callout__name">(Base-3 Characterization)</span></p>

Show that the Cantor set $C$ consists of all points $c \in [0, 1]$ that have no 1's in their base-3 expansion.

**Solution:** First let's remember how to write an arbitrary number $x \in [0, 1]$ in base-3. We expand in powers of $1/3$: thus if $x = \frac{a_1}{3} + \frac{a_2}{3^2} + \frac{a_3}{3^3} + \ldots$, then $x = .a_1 a_2 a_3 \ldots$ in base-3, where the digits $a_n$ are 0, 1, or 2.

If we imagine that $[0,1]$ is divided into three equal pieces, then the first digit $a_1$ tells us whether $x$ is in the left, middle, or right piece. All numbers with $a_1 = 0$ are in the left piece, etc.

Now think about the base-3 expansion of points in the Cantor set $C$. We deleted the middle third of $[0, 1]$ at the first stage of constructing $C$; this removed all points whose first digit is 1. So the points left over must have 0 or 2 as their first digit. Similarly, points whose *second* digit is 1 were deleted at the next stage in the construction. By repeating this argument, we see that $C$ consists of all points whose base-3 expansion contains no 1's, as claimed. $\blacksquare$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.2.3</span><span class="math-callout__name">(The Cantor Set Is Uncountable)</span></p>

Show that the Cantor set is uncountable.

**Solution:** This is just a rewrite of the Cantor diagonal argument of Example 11.1.4. Suppose there were a list $\lbrace c_1, c_2, c_3, \ldots\rbrace$ of all points in $C$. To show that $C$ is uncountable, we produce a point $\overline{c}$ that is in $C$ but not on the list. Let $c_{ij}$ denote the $j$th digit in the base-3 expansion of $c_i$. Define $\overline{c} = .\overline{c}_{11}\overline{c}_{22}\ldots$, where the overbar means we swap 0's and 2's: thus $\overline{c}_{nn} = 0$ if $c_{nn} = 2$ and $\overline{c}_{nn} = 2$ if $c_{nn} = 0$. Then $\overline{c}$ is in $C$, since it's written solely with 0's and 2's, but $\overline{c}$ is not on the list, since it differs from $c_n$ in the $n$th digit. This contradicts the original assumption that the list is complete. Hence $C$ is uncountable. $\blacksquare$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Topological Cantor Set)</span></p>

There are so many different Cantor-like sets that mathematicians have abstracted their essence in the following definition. A closed set $S$ is called a **topological Cantor set** if it satisfies the following properties:

1. $S$ is "totally disconnected." This means that $S$ contains no connected subsets (other than single points). In this sense, all points in $S$ are separated from each other. For the middle-thirds Cantor set and other subsets of the real line, this condition simply says that $S$ contains no intervals.

2. On the other hand, $S$ contains no "isolated points." This means that every point in $S$ has a neighbor arbitrarily close by — given any point $p \in S$ and any small distance $\varepsilon > 0$, there is some other point $q \in S$ within a distance $\varepsilon$ of $p$.

The paradoxical aspects of Cantor sets arise because the first property says that points in $S$ are spread apart, whereas the second property says they're packed together! Notice that the definition says nothing about self-similarity or dimension. These notions are geometric rather than topological.

</div>

### 11.3 Dimension of Self-Similar Fractals

What is the "dimension" of a set of points? For familiar geometric objects, the answer is clear — lines and smooth curves are one-dimensional, planes and smooth surfaces are two-dimensional, solids are three-dimensional. If forced to give a definition, we could say that *the dimension is the minimum number of coordinates needed to describe every point in the set*. For instance, a smooth curve is one-dimensional because every point on it is determined by one number, the arc length from some fixed reference point on the curve.

But when we try to apply this definition to fractals, we quickly run into paradoxes. Consider the **von Koch curve**, defined recursively: we start with a line segment $S_0$. To generate $S_1$, we delete the middle third of $S_0$ and replace it with the other two sides of an equilateral triangle. Subsequent stages are generated recursively by the same rule: $S_n$ is obtained by replacing the middle third of each line segment in $S_{n-1}$ by the other two sides of an equilateral triangle. The limiting set $K = S_\infty$ is the von Koch curve.

#### A Paradox

What is the dimension of the von Koch curve? Since it's a curve, you might be tempted to say it's one-dimensional. But the trouble is that $K$ has *infinite arc length*! If the length of $S_0$ is $L_0$, then the length of $S_1$ is $L_1 = \frac{4}{3}L_0$, because $S_1$ contains four segments, each of length $\frac{1}{3}L_0$. The length increases by a factor of $\frac{4}{3}$ at each stage of the construction, so $L_n = (\frac{4}{3})^n L_0 \to \infty$ as $n \to \infty$.

Moreover, the arc length between *any* two points on $K$ is infinite, by similar reasoning. Hence points on $K$ aren't determined by their arc length from a particular point, because every point is infinitely far from every other!

This suggests that $K$ is more than one-dimensional. But would we really want to say that $K$ is two-dimensional? It certainly doesn't seem to have any "area." So the dimension should be *between* 1 and 2, whatever that means.

#### Similarity Dimension

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Similarity Dimension)</span></p>

The simplest fractals are self-similar, i.e., they are made of scaled-down copies of themselves, all the way down to arbitrarily small scales. The dimension of such fractals can be defined by extending an elementary observation about *classical* self-similar sets like line segments, squares, or cubes.

For instance, consider a square region. If we shrink the square by a factor of 2 in each direction, it takes four of the small squares to equal the whole. If we scale the original square down by a factor of 3, then nine small squares are required. In general, if we reduce the linear dimensions of the square region by a factor of $r$, it takes $r^2$ of the smaller squares to equal the original.

Similarly, a solid cube scaled down by a factor of $r$ requires $r^3$ of the smaller cubes to make up the larger one. The exponents 2 and 3 reflect the two-dimensionality of the square and the three-dimensionality of the cube.

This connection between dimensions and exponents suggests the following definition. Suppose that a self-similar set is composed of $m$ copies of itself scaled down by a factor of $r$. Then the **similarity dimension** $d$ is the exponent defined by $m = r^d$, or equivalently,

$$d = \frac{\ln m}{\ln r}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.3.1</span><span class="math-callout__name">(Dimension of the Cantor Set)</span></p>

Find the similarity dimension of the Cantor set $C$.

**Solution:** $C$ is composed of two copies of itself, each scaled down by a factor of 3. So $m = 2$ when $r = 3$. Therefore $d = \ln 2 / \ln 3 \approx 0.63$. $\blacksquare$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.3.2</span><span class="math-callout__name">(Dimension of the Von Koch Curve)</span></p>

Show that the von Koch curve has a similarity dimension of $\ln 4 / \ln 3 \approx 1.26$.

**Solution:** The curve is made up of four equal pieces, each of which is similar to the original curve but is scaled down by a factor of 3 in both directions. Hence $m = 4$ when $r = 3$, and therefore $d = \ln 4 / \ln 3$. $\blacksquare$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.3.3</span><span class="math-callout__name">(Even-Fifths Cantor Set)</span></p>

Other self-similar fractals can be generated by changing the recursive procedure. For instance, divide an interval into five equal pieces, delete the second and fourth subintervals, and then repeat the process indefinitely. We call the limiting set the **even-fifths Cantor set**. Find its similarity dimension.

**Solution:** Let the original interval be denoted $S_0$, and let $S_n$ denote the $n$th stage of the construction. If we scale $S_n$ down by a factor of five, we get one third of the set $S_{n+1}$. Now setting $n = \infty$, we see that the even-fifths Cantor set is made of three copies of itself, shrunk by a factor of 5. Hence $m = 3$ when $r = 5$, and so $d = \ln 3 / \ln 5$. $\blacksquare$

</div>

### 11.4 Box Dimension

To deal with fractals that are not self-similar, we need to generalize our notion of dimension still further.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Box Dimension)</span></p>

Let $S$ be a subset of $D$-dimensional Euclidean space, and let $N(\varepsilon)$ be the minimum number of $D$-dimensional cubes of side $\varepsilon$ needed to cover $S$. For classical sets, $N(\varepsilon)$ depends on $\varepsilon$ via a power law: for a smooth curve of length $L$, $N(\varepsilon) \propto L/\varepsilon$; for a planar region of area $A$ bounded by a smooth curve, $N(\varepsilon) \propto A/\varepsilon^2$. The key observation is that the dimension of the set equals the exponent $d$ in the power law $N(\varepsilon) \propto 1/\varepsilon^d$.

This power law also holds for most fractal sets $S$, except that $d$ is no longer an integer. By analogy with the classical case, we interpret $d$ as a dimension, usually called the **capacity** or **box dimension** of $S$. An equivalent definition is

$$d = \lim_{\varepsilon \to 0}\frac{\ln N(\varepsilon)}{\ln(1/\varepsilon)},\quad\text{if the limit exists.}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.4.1</span><span class="math-callout__name">(Box Dimension of the Cantor Set)</span></p>

Find the box dimension of the Cantor set.

**Solution:** Recall that the Cantor set is covered by each of the sets $S_n$ used in its construction. Each $S_n$ consists of $2^n$ intervals of length $(1/3)^n$, so if we pick $\varepsilon = (1/3)^n$, we need all $2^n$ of these intervals to cover the Cantor set. Hence $N = 2^n$ when $\varepsilon = (1/3)^n$. Since $\varepsilon \to 0$ as $n \to \infty$, we find

$$d = \lim_{\varepsilon \to 0}\frac{\ln N(\varepsilon)}{\ln(1/\varepsilon)} = \frac{\ln(2^n)}{\ln(3^n)} = \frac{n\ln 2}{n\ln 3} = \frac{\ln 2}{\ln 3},$$

in agreement with the similarity dimension found in Example 11.3.1. $\blacksquare$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Discrete $\varepsilon$-Sequences)</span></p>

The solution of Example 11.4.1 illustrates a helpful trick. We used a discrete sequence $\varepsilon = (1/3)^n$ that tends to zero as $n \to \infty$, even though the definition of box dimension says that we should let $\varepsilon \to 0$ continuously. If $\varepsilon \ne (1/3)^n$, the covering will be slightly wasteful — some boxes hang over the edge of the set — but the limiting value of $d$ is the same.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.4.2</span><span class="math-callout__name">(A Non-Self-Similar Fractal)</span></p>

A fractal that is *not* self-similar is constructed as follows. A square region is divided into nine equal squares, and then one of the small squares is selected at random and discarded. Then the process is repeated on each of the eight remaining small squares, and so on. What is the box dimension of the limiting set?

**Solution:** Pick the unit of length to equal the side of the original square. Then $S_1$ is covered (with no wastage) by $N = 8$ squares of side $\varepsilon = \frac{1}{3}$. Similarly, $S_2$ is covered by $N = 8^2$ squares of side $\varepsilon = (\frac{1}{3})^2$. In general, $N = 8^n$ when $\varepsilon = (\frac{1}{3})^n$. Hence

$$d = \lim_{\varepsilon \to 0}\frac{\ln N(\varepsilon)}{\ln(1/\varepsilon)} = \frac{\ln(8^n)}{\ln(3^n)} = \frac{n\ln 8}{n\ln 3} = \frac{\ln 8}{\ln 3}. \quad\blacksquare$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Critique of Box Dimension)</span></p>

When computing the box dimension, it is not always easy to find a minimal cover. There's an equivalent way to compute the box dimension that avoids this problem: cover the set with a square mesh of boxes of side $\varepsilon$, count the number of occupied boxes $N(\varepsilon)$, and then compute $d$ as before.

Even with this improvement, the box dimension is rarely used in practice. Its computation requires too much storage space and computer time, compared to other types of fractal dimension. The box dimension also suffers from some mathematical drawbacks — for example, the set of rational numbers between 0 and 1 can be proven to have a box dimension of 1, even though the set has only countably many points.

Falconer (1990) discusses other fractal dimensions, the most important of which is the **Hausdorff dimension**. It is more subtle than the box dimension. The main conceptual difference is that the Hausdorff dimension uses coverings by small sets of *varying* sizes, not just boxes of fixed size $\varepsilon$. It has nicer mathematical properties than the box dimension, but unfortunately it is even harder to compute numerically.

</div>

### 11.5 Pointwise and Correlation Dimensions

Now it's time to return to dynamics. Suppose that we're studying a chaotic system that settles down to a strange attractor in phase space. Given that strange attractors typically have fractal microstructure (as we'll see in Chapter 12), how could we estimate the fractal dimension?

First we generate a set of very many points $\lbrace\mathbf{x}_i, i = 1, \ldots, n\rbrace$ on the attractor by letting the system evolve for a long time (after taking care to discard the initial transient, as usual). Almost all trajectories on a strange attractor have the same long-term statistics so it's sufficient to run one trajectory for an extremely long time.

Grassberger and Procaccia (1983) proposed a more efficient approach that has become standard.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Pointwise and Correlation Dimensions)</span></p>

Fix a point $\mathbf{x}$ on the attractor $A$. Let $N_\mathbf{x}(\varepsilon)$ denote the number of points on $A$ inside a ball of radius $\varepsilon$ about $\mathbf{x}$. Most of the points in the ball are unrelated to the immediate portion of the trajectory through $\mathbf{x}$; instead they come from later parts that just happen to pass close to $\mathbf{x}$. Thus $N_\mathbf{x}(\varepsilon)$ measures how frequently a typical trajectory visits an $\varepsilon$-neighborhood of $\mathbf{x}$.

As $\varepsilon$ increases, the number of points in the ball typically grows as a power law:

$$N_\mathbf{x}(\varepsilon) \propto \varepsilon^d,$$

where $d$ is called the **pointwise dimension** at $\mathbf{x}$. The pointwise dimension can depend significantly on $\mathbf{x}$; it will be smaller in rarefied regions of the attractor.

To get an overall dimension of $A$, one averages $N_\mathbf{x}(\varepsilon)$ over many $\mathbf{x}$. The resulting quantity $C(\varepsilon)$ is found empirically to scale as

$$C(\varepsilon) \propto \varepsilon^d,$$

where $d$ is called the **correlation dimension**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Correlation vs. Box Dimension)</span></p>

The correlation dimension takes account of the density of points on the attractor, and thus differs from the box dimension, which weights all occupied boxes equally, no matter how many points they contain. (Mathematically speaking, the correlation dimension involves an invariant measure supported on a fractal, not just the fractal itself.) In general, $d_\text{correlation} \le d_\text{box}$, although they are usually very close (Grassberger and Procaccia 1983).

To estimate $d$, one plots $\log C(\varepsilon)$ vs. $\log\varepsilon$. If the relation $C(\varepsilon) \propto \varepsilon^d$ were valid for all $\varepsilon$, we'd find a straight line of slope $d$. In practice, the power law holds only over an intermediate range of $\varepsilon$ — the **scaling region** where

$$\text{(minimum separation of points on } A) \ll \varepsilon \ll \text{(diameter of } A).$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.5.1</span><span class="math-callout__name">(Correlation Dimension of the Lorenz Attractor)</span></p>

Estimate the correlation dimension of the Lorenz attractor, for the standard parameter values $r = 28$, $\sigma = 10$, $b = \frac{8}{3}$.

**Solution:** Grassberger and Procaccia (1983) integrated the system numerically with a Runge-Kutta method. A line of slope $d_\text{corr} = 2.05 \pm 0.01$ gives an excellent fit to the data, except for large $\varepsilon$, where the expected saturation occurs.

These results were obtained with 15,000 points; convergence was rapid and the correlation dimension could be estimated to within $\pm 5$ percent using only a few thousand points. $\blacksquare$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.5.2</span><span class="math-callout__name">(Correlation Dimension at the Onset of Chaos)</span></p>

Consider the logistic map $x_{n+1} = rx_n(1 - x_n)$ at the parameter value $r = r_\infty = 3.5699456\ldots$, corresponding to the onset of chaos. Show that the attractor is a Cantor-like set, although it is not strictly self-similar. Then compute its correlation dimension numerically.

**Solution:** We visualize the attractor by building it up recursively. Roughly speaking, the attractor looks like a $2^n$-cycle, for $n \gg 1$. The superstable $2^n$-cycles approach a topological Cantor set as $n \to \infty$, with points separated by gaps of various sizes. But the set is not strictly self-similar — the gaps scale by different factors depending on their location. In other words, some of the "wishbones" in the orbit diagram are wider than others at the same $r$.

The correlation dimension of the limiting set has been estimated by Grassberger and Procaccia (1983). They generated a single trajectory of 30,000 points, starting from $x_0 = \frac{1}{2}$. Their plot of $\log C(\varepsilon)$ vs. $\log\varepsilon$ is well fit by a straight line of slope $d_\text{corr} = 0.500 \pm 0.005$.

This is smaller than the box dimension $d_\text{box} \approx 0.538$ (Grassberger 1981), as expected. $\blacksquare$

</div>

#### Multifractals

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Multifractals)</span></p>

In the logistic attractor of Example 11.5.2, the scaling varies from place to place, unlike in the middle-thirds Cantor set, where there is a uniform scaling by $\frac{1}{3}$ everywhere. Thus we cannot completely characterize the logistic attractor by its dimension, or any other single number — we need some kind of distribution function that tells us how the dimension varies across the attractor. Sets of this type are called **multifractals**.

The notion of pointwise dimension allows us to quantify the local variations in scaling. Given a multifractal $A$, let $S_\alpha$ be the subset of $A$ consisting of all points with pointwise dimension $\alpha$. If $\alpha$ is a typical scaling factor on $A$, then $S_\alpha$ will be a relatively large set; if $\alpha$ is unusual, then $S_\alpha$ will be a small set. Each $S_\alpha$ is itself a fractal, so it makes sense to measure its "size" by its fractal dimension. Let $f(\alpha)$ denote the dimension of $S_\alpha$. Then $f(\alpha)$ is called the **multifractal spectrum** of $A$ or the **spectrum of scaling indices** (Halsey et al. 1986).

Roughly speaking, you can think of the multifractal as an interwoven set of fractals of different dimensions $\alpha$, where $f(\alpha)$ measures their relative weights. The maximum value of $f(\alpha)$ turns out to be the box dimension (Halsey et al. 1986).

For systems at the onset of chaos, multifractals lead to a more powerful version of the universality theory mentioned in Section 10.6. The universal quantity is now a *function* $f(\alpha)$, rather than a single number; it therefore offers much more information, and the possibility of more stringent tests.

</div>

## Chapter 12: Strange Attractors

### 12.0 Introduction

Our work in the previous three chapters has revealed quite a bit about chaotic systems, but something important is missing: intuition. We know *what* happens but not *why* it happens. For instance, we don't know what causes sensitive dependence on initial conditions, nor how a differential equation can generate a fractal attractor. Our first goal is to understand such things in a simple, geometric way.

In the mid-1970s, the only known examples of strange attractors were the Lorenz attractor (1963) and some mathematical constructions of Smale (1967). Thus there was a need for other concrete examples, preferably as transparent as possible. These were supplied by Hénon (1976) and Rössler (1976), using the intuitive concepts of **stretching and folding**. The chapter concludes with experimental examples of strange attractors from chemistry and mechanics, illustrating the techniques of attractor reconstruction and Poincaré sections.

### 12.1 The Simplest Examples

Strange attractors have two properties that seem hard to reconcile. Trajectories on the attractor remain confined to a bounded region of phase space, yet they separate from their neighbors exponentially fast (at least initially). How can trajectories diverge endlessly and yet stay bounded?

The basic mechanism involves repeated **stretching and folding**. Consider a small blob of initial conditions in phase space. A strange attractor typically arises when the flow contracts the blob in some directions (reflecting the dissipation in the system) and stretches it in others (leading to sensitive dependence on initial conditions). The stretching cannot go on forever — the distorted blob must be folded back on itself to remain in the bounded region.

#### Making Pastry

To illustrate the effects of stretching and folding, we consider a domestic example. The process used to make filo pastry or croissant works as follows: the dough is rolled out and flattened, then folded over, then rolled out again, and so on. After many repetitions, the end product is a flaky, layered structure — the culinary analog of a fractal attractor.

Furthermore, this process automatically generates sensitive dependence on initial conditions. Suppose that a small drop of food coloring is put in the dough, representing nearby initial conditions. After many iterations of stretching, folding, and re-injection, the coloring will be spread throughout the dough.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Pastry Map and Cantor Sets)</span></p>

The **pastry map** is modeled as a continuous mapping of a rectangle into itself. The rectangle $abcd$ is flattened, stretched, and folded into a horseshoe shape $a'b'c'd'$. Repeating this transformation, the layers become thinner and there are twice as many of them at each stage ($S_1, S_2, S_3, \ldots$).

The limiting set $S_\infty$ consists of infinitely many smooth layers, separated by gaps of various sizes. In fact, a vertical cross section through the middle of $S_\infty$ would resemble a *Cantor set*! Thus $S_\infty$ is (locally) the product of a smooth curve with a Cantor set. The fractal structure of the attractor is a consequence of the stretching and folding that created $S_\infty$ in the first place.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Terminology: Pastry Map vs. Smale Horseshoe)</span></p>

The pastry map transformation is normally called a horseshoe map, but we avoid that name because it encourages confusion with the **Smale horseshoe**, which has very different properties. In particular, Smale's horseshoe map does *not* have a strange attractor; its invariant set is more like a strange saddle. The Smale horseshoe is fundamental to rigorous discussions of chaos, but its analysis is best deferred to a more advanced course.

</div>

#### The Baker's Map

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 12.1.1</span><span class="math-callout__name">(The Baker's Map)</span></p>

The **baker's map** $B$ of the square $0 \le x \le 1$, $0 \le y \le 1$ to itself is given by

$$(x_{n+1}, y_{n+1}) = \begin{cases} (2x_n,\, a y_n) & \text{for } 0 \le x_n \le \tfrac{1}{2},\\[4pt] (2x_n - 1,\, a y_n + \tfrac{1}{2}) & \text{for } \tfrac{1}{2} \le x_n \le 1, \end{cases}$$

where $a$ is a parameter in the range $0 < a \le \frac{1}{2}$. Illustrate the geometric action of $B$ by showing its effect on a face drawn in the unit square.

**Solution:** The transformation may be regarded as a product of two simpler transformations. First the square is stretched and flattened into a $2 \times a$ rectangle. Then the rectangle is cut in half, yielding two $1 \times a$ rectangles, and the right half is stacked on top of the left half such that its base is at the level $y = \frac{1}{2}$.

The baker's map exhibits sensitive dependence on initial conditions, thanks to the stretching in the $x$-direction. It has many chaotic orbits — uncountably many, in fact. $\blacksquare$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 12.1.2</span><span class="math-callout__name">(Fractal Attractor of the Baker's Map)</span></p>

Show that for $a < \frac{1}{2}$, the baker's map has a fractal attractor $A$ that attracts all orbits. More precisely, show that there is a set $A$ such that for any initial condition $(x_0, y_0)$, the distance from $B^n(x_0, y_0)$ to $A$ converges to zero as $n \to \infty$.

**Solution:** First we construct the attractor. Let $S$ denote the square $0 \le x \le 1$, $0 \le y \le 1$; this includes all possible initial conditions. The first three images of $S$ under the map $B$ are shown as shaded regions.

The first image $B(S)$ consists of two strips of height $a$, as we know from Example 12.1.1. Then $B(S)$ is itself flattened, stretched, cut, and stacked to yield $B^2(S)$. Now we have four strips of height $a^2$. Continuing in this way, we see that $B^n(S)$ consists of $2^n$ horizontal strips of height $a^n$. The limiting set $A = B^\infty(S)$ is a fractal. Topologically, it is a Cantor set of line segments.

A technical point: we can be sure that the limiting set exists by invoking a standard theorem from point-set topology. The successive images of the square are **nested** inside each other like Chinese boxes: $B^{n+1}(S) \subset B^n(S)$ for all $n$. Moreover each $B^n(S)$ is a compact set. The theorem (Munkres 1975) assures us that the countable intersection of a nested family of compact sets is a *non-empty* compact set — this set is our $A$. Furthermore, $A \subset B^n(S)$ for all $n$.

The nesting property also helps us show that $A$ attracts all orbits. The point $B^n(x_0, y_0)$ lies somewhere in one of the strips of $B^n(S)$, and all points in these strips are within a distance $a^n$ of $A$, because $A$ is contained in $B^n(S)$. Since $a^n \to 0$ as $n \to \infty$, the distance from $B^n(x_0, y_0)$ to $A$ tends to zero as $n \to \infty$, as required. $\blacksquare$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 12.1.3</span><span class="math-callout__name">(Box Dimension of the Baker's Map Attractor)</span></p>

Find the box dimension of the attractor for the baker's map with $a < \frac{1}{2}$.

**Solution:** The attractor $A$ is approximated by $B^n(S)$, which consists of $2^n$ strips of height $a^n$ and length 1. Now cover $A$ with square boxes of side $\varepsilon = a^n$.

Since the strips have length 1, it takes about $a^{-n}$ boxes to cover each of them. There are $2^n$ strips altogether, so $N \approx a^{-n} \times 2^n = (a/2)^{-n}$. Thus

$$d = \lim_{\varepsilon \to 0}\frac{\ln N}{\ln(1/\varepsilon)} = \lim_{n \to 0}\frac{\ln[(a/2)^{-n}]}{\ln(a^{-n})} = 1 + \frac{\ln\frac{1}{2}}{\ln a}.$$

As a check, note that $d \to 2$ as $a \to \frac{1}{2}$; this makes sense because the attractor fills an increasingly large portion of square $S$ as $a \to \frac{1}{2}$. $\blacksquare$

</div>

#### The Importance of Dissipation

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dissipation and Strange Attractors)</span></p>

For $a < \frac{1}{2}$, the baker's map shrinks areas in phase space. Given any region $R$ in the square,

$$\text{area}(B(R)) < \text{area}(R).$$

This follows from elementary geometry: the baker's map elongates $R$ by a factor of 2 and flattens it by a factor of $a$, so $\text{area}(B(R)) = 2a \times \text{area}(R)$. Since $a < \frac{1}{2}$ by assumption, $\text{area}(B(R)) < \text{area}(R)$ as required.

Area contraction is the analog of the volume contraction that we found for the Lorenz equations in Section 9.2. As in that case, it yields several conclusions: the attractor $A$ for the baker's map must have zero area, and it cannot have any repelling fixed points, since such points would expand area elements in their neighborhood.

In contrast, when $a = \frac{1}{2}$ the baker's map is **area-preserving**: $\text{area}(B(R)) = \text{area}(R)$. Now the square $S$ is mapped *onto* itself, with no gaps between the strips. The map has qualitatively different dynamics in this case. Transients never decay — the orbits shuffle around endlessly in the square but never settle to a lower-dimensional attractor. This is a kind of chaos we have not seen before!

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dissipative vs. Conservative Systems)</span></p>

The distinction between $a < \frac{1}{2}$ and $a = \frac{1}{2}$ exemplifies a broader theme in nonlinear dynamics. In general, if a map or flow contracts volumes in phase space, it is called **dissipative**. Dissipative systems commonly arise as models of physical situations involving friction, viscosity, or some other process that dissipates energy. In contrast, area-preserving maps are associated with conservative systems, particularly with the Hamiltonian systems of classical mechanics.

The distinction is crucial because **area-preserving maps cannot have attractors** (strange or otherwise). As defined in Section 9.3, an "attractor" should attract all orbits starting in a sufficiently small open set containing it; that requirement is incompatible with area-preservation.

</div>

### 12.2 Hénon Map

In this section we discuss another two-dimensional map with a strange attractor. It was devised by the theoretical astronomer Michel Hénon (1976) to illuminate the microstructure of strange attractors.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hénon Map)</span></p>

The **Hénon map** is given by

$$x_{n+1} = y_n + 1 - ax_n^2, \qquad y_{n+1} = bx_n,$$

where $a$ and $b$ are adjustable parameters. Hénon (1976) arrived at this map by an elegant line of reasoning. To simulate the stretching and folding that occurs in the Lorenz system, he considered the following chain of transformations:

1. **Fold:** $T': x' = x,\quad y' = 1 + y - ax^2$. The bottom and top of a rectangle get mapped to parabolas. The parameter $a$ controls the folding.

2. **Contract:** $T'': x'' = bx',\quad y'' = y'$, where $-1 < b < 1$. This contracts the region along the $x$-axis.

3. **Reflect:** $T''': x''' = y'',\quad y''' = x''$. This reflects across the line $y = x$.

Then the composite transformation $T = T'''T''T'$ yields the Hénon mapping.

</div>

#### Elementary Properties of the Hénon Map

The Hénon map captures several essential properties of the Lorenz system:

1. **The Hénon map is invertible.** This is the counterpart of the fact that in the Lorenz system, there is a unique trajectory through each point in phase space. In particular, each point has a unique past. In this respect the Hénon map is superior to the logistic map, its one-dimensional analog, which is not invertible since all points (except the maximum) come from *two* pre-images.

2. **The Hénon map is dissipative.** It contracts areas, and does so at the same rate everywhere in phase space. This property is the analog of constant negative divergence in the Lorenz system.

3. **For certain parameter values, the Hénon map has a trapping region.** There is a region $R$ that gets mapped inside itself. As in the Lorenz system, the strange attractor is enclosed in the trapping region.

4. **Some trajectories of the Hénon map escape to infinity.** In contrast, all trajectories of the Lorenz system are bounded.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 12.2.1</span><span class="math-callout__name">(Invertibility of the Hénon Map)</span></p>

Show that the Hénon map $T$ is invertible if $b \ne 0$, and find the inverse $T^{-1}$.

**Solution:** We solve for $x_n$ and $y_n$, given $x_{n+1}$ and $y_{n+1}$. Algebra yields $x_n = b^{-1}y_{n+1}$, $y_n = x_{n+1} - 1 + ab^{-2}(y_{n+1})^2$. Thus $T^{-1}$ exists for all $b \ne 0$. $\blacksquare$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 12.2.2</span><span class="math-callout__name">(Area Contraction)</span></p>

Show that the Hénon map contracts areas if $-1 < b < 1$.

**Solution:** To decide whether an arbitrary two-dimensional map $x_{n+1} = f(x_n, y_n)$, $y_{n+1} = g(x_n, y_n)$ is area-contracting, we compute the determinant of its Jacobian matrix

$$\mathbf{J} = \begin{pmatrix} \partial f/\partial x & \partial f/\partial y \\ \partial g/\partial x & \partial g/\partial y \end{pmatrix}.$$

If $|\det \mathbf{J}(x, y)| < 1$ for all $(x, y)$, the map is area-contracting. This rule follows from a fact of multivariable calculus: if $\mathbf{J}$ is the Jacobian of a two-dimensional map $T$, then $T$ maps an infinitesimal rectangle at $(x, y)$ with area $dx\,dy$ into an infinitesimal parallelogram with area $|\det \mathbf{J}(x, y)|\,dx\,dy$.

For the Hénon map, we have $f(x, y) = 1 - ax^2 + y$ and $g(x, y) = bx$. Therefore

$$\mathbf{J} = \begin{pmatrix} -2ax & 1 \\ b & 0 \end{pmatrix}$$

and $\det \mathbf{J}(x, y) = -b$ for all $(x, y)$. Hence the map is area-contracting for $-1 < b < 1$, as claimed. In particular, the area of any region is reduced by a *constant* factor of $|b|$ with each iteration. $\blacksquare$

</div>

#### Choosing Parameters

As Hénon (1976) explains, $b$ should not be too close to zero, or else the area contraction will be excessive and the fine structure of the attractor will be invisible. But if $b$ is too large, the folding won't be strong enough. A good choice is $b = 0.3$.

To find a good value of $a$, Hénon had to do some exploring. If $a$ is too small or too large, all trajectories escape to infinity; there is no attractor in these cases. For intermediate values of $a$, the trajectories either escape to infinity or approach an attractor, depending on the initial conditions. As $a$ increases through this range, the attractor changes from a stable fixed point to a stable 2-cycle. The system then undergoes a period-doubling route to chaos, followed by chaos intermingled with periodic windows. Hénon picked $a = 1.4$, well into the chaotic region.

#### Zooming In on a Strange Attractor

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Self-Similar Structure of the Hénon Attractor)</span></p>

In a striking series of plots, Hénon provided the first direct visualization of the fractal structure of a strange attractor. He set $a = 1.4$, $b = 0.3$ and generated ten thousand successive iterates of the map, starting from the origin. The attractor is bent like a boomerang and is made of many parallel curves.

Zooming into a small square of the attractor, the characteristic fine structure begins to emerge. There seem to be six parallel curves: a lone curve near the middle of the frame, then two closely spaced curves above it, and then three more. If we zoom in on those three curves, it becomes clear that they are actually six curves, grouped one, two, three, exactly as before! And those curves are themselves made of thinner curves in the same pattern, and so on. The self-similarity continues to arbitrarily small scales.

</div>

#### The Unstable Manifold of the Saddle Point

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Structure of the Hénon Attractor)</span></p>

The zooming-in plots suggest that the Hénon attractor is Cantor-like in the transverse direction, but smooth in the longitudinal direction. There's a reason for this. The attractor is closely related to a locally smooth object — the unstable manifold of a saddle point that sits on the edge of the attractor. To be more precise, Benedicks and Carleson (1991) have proven that the attractor is the closure of a branch of the unstable manifold; see also Simó (1979).

Hobson (1993) developed a method for computing this unstable manifold to very high accuracy. As expected, it is indistinguishable from the strange attractor.

</div>

### 12.3 Rössler System

So far we have used two-dimensional maps to help us understand how stretching and folding can generate strange attractors. Now we return to differential equations.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Rössler System)</span></p>

In the culinary spirit of the pastry map and the baker's map, Otto Rössler (1976) found inspiration in a taffy-pulling machine. By pondering its action, he was led to a system of three differential equations with a simpler strange attractor than Lorenz's. The **Rössler system** has only one quadratic nonlinearity $xz$:

$$\dot{x} = -y - z, \qquad \dot{y} = x + ay, \qquad \dot{z} = b + z(x - c).$$

We first met this system in Section 10.6, where we saw that it undergoes a period-doubling route to chaos as $c$ is increased.

</div>

Numerical integration shows that this system has a strange attractor for $a = b = 0.2$, $c = 5.7$. A schematic version of the attractor illustrates the key mechanism: neighboring trajectories separate by spiraling out ("stretching"), then cross without intersecting *by going into the third dimension* ("folding"), and then circulate back near their starting places ("re-injection"). We can now see why three dimensions are needed for a flow to be chaotic.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric Model of the Rössler Attractor)</span></p>

Following the visual approach of Abraham and Shaw (1983), our goal is to construct a geometric model of the Rössler attractor, guided by the stretching, folding, and re-injection seen in numerical integrations of the system.

Near a typical trajectory, in one direction there's *compression toward* the attractor, and in the other direction there's *divergence along* the attractor. The flow folds the wide part of the sheet in two and then bends it around so that it nearly joins the narrow part. Overall, the flow has taken the single sheet and produced *two* sheets after one circuit. Repeating the process, those two sheets produce four, and then those produce eight, and so on.

In effect, the flow is acting like the pastry transformation, and the phase space is acting like the dough! Ultimately the flow generates an infinite complex of tightly packed surfaces: the strange attractor.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poincaré Section and Cantor Set Structure)</span></p>

A **Poincaré section** of the attractor is obtained by slicing the attractor with a plane, thereby exposing its cross section. (In the same way, biologists examine complex three-dimensional structures by slicing them and preparing slides.)

If we take a further one-dimensional slice or **Lorenz section** through the Poincaré section, we find an infinite set of points separated by gaps of various sizes. This pattern of dots and gaps is a topological Cantor set. Since each dot corresponds to one layer of the complex, our model of the Rössler attractor is a **Cantor set of surfaces**. More precisely, the attractor is locally topologically equivalent to the Cartesian product of a ribbon and a Cantor set. This is precisely the structure we would expect, based on our earlier work with the pastry map.

</div>

### 12.4 Chemical Chaos and Attractor Reconstruction

Strange attractors are not merely mathematical curiosities — they appear in real physical and chemical systems. One of the most compelling demonstrations comes from the Belousov-Zhabotinsky (BZ) reaction, a chemical oscillator that we encountered earlier in the context of limit cycles (Section 8.3). By the 1970s, it became natural to ask whether the BZ reaction could also exhibit chaos under the right conditions.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Experimental Evidence for Chemical Chaos)</span></p>

Early reports of chemical chaos by Schmitz, Graziani, and Hudson (1977) were met with skepticism. Critics argued that the observed complex dynamics might simply result from uncontrolled fluctuations in experimental parameters. What was needed was a convincing demonstration that the dynamics genuinely obeyed the laws of chaos, rather than being artifacts of imperfect control.

The decisive experiments were performed by Roux, Simoyi, Wolf, and Swinney using a continuous-flow stirred tank reactor. Fresh chemicals are pumped in at a constant rate to keep the system far from equilibrium, and the mixture is continuously stirred to enforce spatial homogeneity. This reduces the effective number of degrees of freedom. The dynamics are monitored by recording the concentration of bromide ions $B(t)$ over time.

The resulting time series appears roughly periodic at first glance, but closer inspection reveals erratic fluctuations in the amplitude — a hallmark of deterministic chaos on a strange attractor.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Attractor Reconstruction)</span></p>

**Attractor reconstruction** is a technique for recovering the geometry of a strange attractor from measurements of a single time series. The key idea is that a single observed variable, measured over time, can carry enough information to reconstruct the full phase-space dynamics.

Given a scalar time series $B(t)$, one constructs a **delay vector**

$$\mathbf{x}(t) = (B(t),\, B(t+\tau),\, \dots,\, B(t + (d-1)\tau))$$

for some chosen delay $\tau > 0$ and embedding dimension $d$. Plotting these vectors traces out a trajectory in a $d$-dimensional reconstructed phase space. Under suitable conditions, this reconstructed attractor is topologically equivalent to the true attractor in the full (possibly very high-dimensional) phase space. This remarkable fact was established by Packard et al. (1980) and placed on rigorous footing by Takens (1981).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Application to the BZ Reaction)</span></p>

When Roux et al. (1983) applied attractor reconstruction to their bromide time series with delay $\tau = 8.8$ seconds, the two-dimensional reconstruction $(B(t), B(t+\tau))$ traced out a shape remarkably similar to the Rössler attractor. They also performed a three-dimensional reconstruction $\mathbf{x}(t) = (B(t), B(t+\tau), B(t+2\tau))$ and computed a Poincaré section. The data fell on an approximately one-dimensional curve, confirming that the chaotic trajectories are confined to a nearly two-dimensional sheet — just as in the Rössler system.

By constructing an approximate one-dimensional return map from the Poincaré section (plotting successive intersections $X_{n+1}$ vs. $X_n$), the data fell on a smooth unimodal curve resembling the logistic map. This strongly suggests that the chaotic state is reached via a period-doubling cascade. Indeed, Roux et al. observed many distinct periodic windows as the flow rate was varied, occurring in exactly the order predicted by the $U$-sequence of universality theory (Section 10.6).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Issues in Attractor Reconstruction)</span></p>

Two important choices must be made when implementing attractor reconstruction:

1. **Embedding dimension $d$:** One needs enough delays so that the attractor can "disentangle itself" in the reconstructed phase space. The standard approach is to increase $d$ and compute the correlation dimension of the resulting attractors. As $d$ grows, the estimated dimension initially increases but eventually levels off at the true value — provided $d$ is large enough. However, for very large embedding dimensions, the sparsity of data in high-dimensional space causes statistical sampling problems, limiting our ability to reliably estimate the dimension of high-dimensional attractors.

2. **Delay $\tau$:** For noisy real-world data, the optimal delay is typically around one-tenth to one-quarter of the mean orbital period of the attractor. If $\tau$ is too small, the delay coordinates are highly correlated and the reconstructed attractor is squeezed near the diagonal; if $\tau$ is too large, the coordinates become essentially independent and the reconstruction degenerates.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Reconstruction of a Limit Cycle)</span></p>

Consider a system with a limit-cycle attractor for which $x(t) = \sin t$. Plotting the time-delayed trajectory $\mathbf{x}(t) = (x(t),\, x(t+\tau))$ for different values of $\tau$ yields ellipses. For small $\tau$ (e.g. $\tau = \pi/6$), the ellipse is elongated along the diagonal $y = x$, so the attractor is poorly resolved. At $\tau = \pi/2$ (one-quarter of the period $T = 2\pi$), the trajectory traces a perfect circle, since $x(t) = \sin t$ and $x(t+\pi/2) = \cos t$. This is optimal because the reconstructed attractor is as "open" as possible. For larger $\tau$ (e.g. $\tau = 5\pi/6$), the ellipse is stretched along $y = -x$, again becoming cigar-shaped. In general, narrow reconstructions are more vulnerable to noise corruption.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Significance and Limitations)</span></p>

Attractor reconstruction can in principle distinguish low-dimensional chaos from noise: as the embedding dimension increases, the correlation dimension levels off for chaos but keeps growing for noise. This has inspired attempts to detect deterministic chaos in stock prices, heart rhythms, brain waves, and sunspots. However, most of this research is dubious — for a sensible discussion and a state-of-the-art method for distinguishing chaos from noise, see Kaplan and Glass (1993).

</div>

### 12.5 Forced Double-Well Oscillator

Up to this point, all examples of strange attractors have arisen from autonomous systems — systems with no explicit time dependence. Once we allow external forcing (making the system nonautonomous), strange attractors become far more common.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Magneto-Elastic Mechanical System)</span></p>

Moon and Holmes (1979) studied a magneto-elastic beam: a slender steel beam clamped in a rigid frame with two permanent magnets at the base pulling it in opposite directions. The beam has two stable buckled states, one leaning toward each magnet, separated by an energy barrier corresponding to the unstable straight configuration.

To drive the system, the apparatus is shaken periodically by an electromagnetic vibration generator. For weak forcing, the beam vibrates gently near one of its buckled equilibria. As the forcing amplitude increases, the beam eventually begins whipping back and forth erratically between the two magnets — and this irregular motion is sustained for tens of thousands of drive cycles. This is chaos in a simple mechanical system.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Double-Well Duffing Oscillator)</span></p>

The magneto-elastic beam system is modeled by the dimensionless equation

$$\ddot{x} + \delta \dot{x} - x + x^3 = F\cos\omega t,$$

where $\delta > 0$ is the damping constant, $F$ is the forcing amplitude, and $\omega$ is the forcing frequency. This can also be interpreted as Newton's law for a particle in a **double-well potential** $V(x) = \tfrac{1}{4}x^4 - \tfrac{1}{2}x^2$, subject to damping and a periodic driving force $F\cos\omega t$. Note that $x$ is the displacement relative to the moving frame, not the lab frame.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Physical Intuition for the Double-Well)</span></p>

If the well is shaken periodically, several regimes emerge depending on the forcing strength:

- **Weak forcing:** The particle jiggles slightly near the bottom of one well — a small-amplitude, low-energy oscillation.
- **Moderate forcing:** At least two types of stable oscillations coexist: a small oscillation confined to one well, and a large-amplitude oscillation that samples both wells by crossing the hump.
- **Very strong forcing:** The particle is always flung back and forth across the hump regardless of initial conditions.
- **Intermediate forcing (potentially chaotic):** When the particle has barely enough energy to reach the top of the hump and the balance between forcing and damping keeps the system in this precarious state, the particle may fall one way or the other depending sensitively on the precise timing. This is the regime where chaos is expected.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Coexisting Limit Cycles, $F = 0.18$)</span></p>

Fix $\delta = 0.25$ and $\omega = 1$ throughout. For $F = 0.18$, numerical integration reveals several coexisting stable limit cycles. Different initial conditions converge to different periodic solutions, all of which correspond physically to oscillations confined to a single well.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Chaotic Regime, $F = 0.40$)</span></p>

For $F = 0.40$ and initial conditions $(x_0, y_0) = (0, 0)$ (where $y = \dot{x}$), the time series of $x(t)$ and $y(t)$ appear aperiodic — the displacement repeatedly changes sign, indicating that the particle crosses the central hump over and over. The phase portrait $(x, y)$ looks like a tangled mess, but this is because the system is nonautonomous: the full state is $(x, y, t)$, and we are seeing a two-dimensional projection of a three-dimensional trajectory.

A **Poincaré section** — obtained by plotting $(x(t), y(t))$ only at times that are integer multiples of the drive period $2\pi/\omega$ — resolves the tangle. The successive strobed points fall on a fractal set, which we interpret as a cross section of a strange attractor. The points hop erratically over this set, exhibiting sensitive dependence on initial conditions.

The numerical simulations reproduce the sustained chaos seen in the beam experiments, with good qualitative agreement between experimental and simulated time series.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Transient Chaos)</span></p>

**Transient chaos** refers to a regime in which the system exhibits chaotic behavior for a long but finite time before eventually settling onto a periodic attractor. This occurs when no strange attractor exists but the system still has complicated transient dynamics — for instance, when two or more stable limit cycles coexist and the ghost of a destroyed strange attractor lingers as a chaotic saddle.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Transient Chaos with $F = 0.25$)</span></p>

For $F = 0.25$, two nearby initial conditions can both exhibit wild, chaotic-looking transients before finally converging to *different* periodic attractors. For instance, starting at $(x_0, y_0) = (0.2, 0.1)$, the trajectory wanders chaotically before settling into a periodic state with $x > 0$ (oscillations in the right well). A tiny change to $x_0 = 0.195$ leads to a similarly chaotic transient, but the trajectory ultimately converges to the *left* well instead. The choice of final attractor depends sensitively on the initial conditions.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Fractal Basin Boundaries)</span></p>

When multiple attractors coexist, each one has a **basin of attraction** — the set of initial conditions that eventually converge to it. For the forced double-well oscillator, the boundary between basins can be a **fractal**. Coloring each initial condition on a fine grid according to which attractor it reaches reveals large patches of uniform color (clearly in one basin or the other), but near the boundary the colors intermingle at arbitrarily fine scales. This fractal basin boundary means that near the boundary, the slightest change in initial conditions can switch the system from one attractor to the other, making long-term prediction essentially impossible.

</div>
