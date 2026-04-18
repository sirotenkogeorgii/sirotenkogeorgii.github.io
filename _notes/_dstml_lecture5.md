## Lecture 5

### Cycles in Nonlinear Maps

#### Introduction to Discrete-Time Systems

This section transitions our focus from continuous flows to discrete maps, exploring their unique behaviors, such as fixed points and cycles. We will begin with a foundational example that is famous for its complexity and its role in the history of chaos theory: the logistic map.

#### The Logistic Map: A Canonical Example

The logistic map is a simple, scalar (one-dimensional) map defined by a quadratic equation. It was famously analyzed by Robert May in a 1976 Nature paper, which highlighted how such a simple deterministic equation could produce extraordinarily complex, chaotic dynamics.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Logistic Map)</span></p>

The **logistic map** is a recursive function that maps a value $x_t$ to a new value $x_{t+1}$. It is defined by the equation:

$$x_{t+1} = \alpha x_t (1 - x_t)$$

Where:

* $x_t$ represents the state of the system at time step $t$.
* $\alpha$ is a single parameter that controls the behavior of the system.

For our analysis, we impose the following constraints:

* The initial condition $x_0$ is in the interval $[0, 1]$.
* The parameter $\alpha$ is in the interval $[0, 4]$.

Under these conditions, it can be shown that the state $x_t$ will remain bounded within the interval $[0, 1]$ for all subsequent time steps $t$.

</div>

#### Fixed Points and Their Stability

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Fixed Points of Logistic Map)</span></p>

Logistic map

$$x_t = \alpha x_{t-1}(1-x_{t-1})$$

has two fixed points:

1. $x_1^\ast = 0$
2. $\alpha x^\ast + 1 - \alpha = 0 \implies x_2^\ast = \frac{\alpha - 1}{\alpha}$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

To find the fixed points of the logistic map, we set $x_{t+1} = x_t = x^\ast$ and solve the resulting equation:

$$x^* = \alpha x^* (1 - x^*)$$

Rearranging this gives us a quadratic equation:

$$\alpha (x^*)^2 + (1 - \alpha) x^* = 0$$

We can factor out $x^*$:

$$x^* (\alpha x^* + 1 - \alpha) = 0$$

This equation yields two solutions for the fixed points:

1. $x_1^\ast = 0$
2. $\alpha x^\ast + 1 - \alpha = 0 \implies x_2^\ast = \frac{\alpha - 1}{\alpha}$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Finding Fixed Points on Coweb)</span></p>

The intersections of the map’s curve with the bisector represent points where the input equals the output — these are the fixed points of the system.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Existence of Fixed Points)</span></p>

From these solutions, we can immediately see two things:

* The fixed point at $x^* = 0$ exists for all values of $\alpha$.
* The second fixed point, $x_2^* = \frac{\alpha - 1}{\alpha}$, only exists within our interval of interest $[0, 1]$ if the parameter $\alpha$ is greater than or equal to $1$.

The return plot illustrates two scenarios. For a small $\alpha$, only one fixed point exists at $x^* = 0$. For a larger $\alpha$, a second fixed point appears where the parabola intersects the bisector.

A cobweb plot starting near the origin converges to the fixed point at $x^* = 0$ when the slope of the map at this point has an absolute value less than $1$, suggesting the point is stable. For a larger $\alpha$, a cobweb plot starting near the origin is repelled from it, while a plot starting near the second fixed point converges towards it. The slope at $x^* = 0$ is now steep (absolute value greater than $1$), indicating it is unstable, while the slope at the second point is shallower, indicating it is stable.

This graphical analysis suggests that the stability of a fixed point is determined by the slope of the map at that point.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/coweb_plot_logistic_map.png' | relative_url }}" alt="a" loading="lazy">
  <!-- <figcaption>First- and second-order return maps of logistic function for $\alpha=3.3$, illustrating $2$-cycle in first-order map as fixed points of second-order map.</figcaption> -->
</figure>

#### Formal Stability Analysis via Linearization

To formalize our intuition, we analyze the behavior of a small perturbation around a fixed point, a technique analogous to the one we used for differential equations.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Derivation of the Stability Condition)</span></p>

Let $x^\ast$ be a fixed point of the map $x_{t+1} = f(x_t)$. Consider a small perturbation $\epsilon_t$ from this fixed point at time $t$:

$$x_t = x^* + \epsilon_t$$

The state at the next time step, $x_{t+1}$, will be:

$$x_{t+1} = x^* + \epsilon_{t+1} = f(x^* + \epsilon_t)$$

Assuming $\epsilon_t$ is small, we can perform a Taylor expansion of $f(x^\ast + \epsilon_t)$ around $x^\ast$:

$$x^* + \epsilon_{t+1} \approx f(x^*) + f'(x^*) \epsilon_t + O(\epsilon_t^2)$$

By definition, $f(x^\ast) = x^\ast$. We can therefore cancel the $x^\ast$ terms on both sides. Ignoring higher-order terms for our linear approximation, we get a recursive map for the perturbation:

$$\epsilon_{t+1} \approx f'(x^*) \epsilon_t$$

This is a linear map describing the evolution of the perturbation. The perturbation $\epsilon_t$ will decay to zero (i.e., the fixed point is stable) if the magnitude of the multiplier is less than one. Conversely, it will grow if the magnitude is greater than one.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Stability of Fixed Points for 1D Maps)</span></p>

Let $x^\ast$ be a fixed point of a nonlinear map $f(x)$. The stability of $x^\ast$ is determined by the derivative of the map evaluated at the fixed point, $f'(x^\ast)$:

* If $\|f'(x^\ast)\| < 1$, the fixed point is locally stable.
* If $\|f'(x^\ast)\| > 1$, the fixed point is locally unstable.
* If $\|f'(x^\ast)\| = 1$, the stability cannot be determined by this linear analysis. This is a non-hyperbolic case, and higher-order terms of the Taylor expansion must be considered.

</div>

#### Generalization to Higher-Dimensional Maps

This stability criterion extends naturally to multivariate (higher-dimensional) maps.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($n$-Dimensional Maps and the Jacobian)</span></p>

Consider a system in $m$ dimensions described by a map $\mathbf{x}_{t+1} = \mathbf{F}(\mathbf{x}_t)$, where $\mathbf{x}_t$ is a vector in $\mathbb{R}^m$. The stability analysis is analogous, but the scalar derivative is replaced by the Jacobian matrix, $J$.

The Jacobian matrix is the matrix of all first-order partial derivatives of the vector-valued function $\mathbf{F}$:

$$J = \begin{pmatrix} \frac{\partial F_1}{\partial x_1} & \cdots & \frac{\partial F_1}{\partial x_m} \\ \vdots & \ddots & \vdots \\ \frac{\partial F_m}{\partial x_1} & \cdots & \frac{\partial F_m}{\partial x_m} \end{pmatrix}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Stability for $nD$ Maps)</span></p>

Let $\mathbf{x}^\ast$ be a fixed point of the map $\mathbf{F}(\mathbf{x})$. The stability of $\mathbf{x}^\ast$ is determined by the eigenvalues of the Jacobian matrix evaluated at the fixed point, $J(\mathbf{x}^\ast)$.

* The fixed point $\mathbf{x}^\ast$ is **stable** if the maximum absolute value (or modulus, for complex eigenvalues) of all eigenvalues of $J(\mathbf{x}^\ast)$ is less than $1$.

$$\max_i |\lambda_i| < 1$$

* The fixed point $\mathbf{x}^\ast$ is **unstable** if the maximum absolute value of any eigenvalue of $J(\mathbf{x}^\ast)$ is greater than $1$.

$$\max_i |\lambda_i| > 1$$

* If the maximum absolute value of the eigenvalues is exactly equal to $1$ (i.e., the largest eigenvalue lies on the unit circle in the complex plane), the system is non-hyperbolic, and a **linear stability analysis is inconclusive**.

</div>

#### The Emergence of $k$-Cycles

What happens when all fixed points in a bounded system become unstable? The trajectory cannot settle into a fixed point, but it also cannot escape to infinity. The system must find another form of stable, persistent behavior. In maps, this often leads to the emergence of cycles.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Path to a 2-Cycle)</span></p>

Consider the logistic map for $\alpha > 3$. At these parameter values, the slopes at both fixed points ($x^* = 0$ and $x^* = (\alpha - 1)/\alpha$) are greater than $1$ in absolute value. This means both fixed points are unstable.

Since we know the system is confined to the interval $[0, 1]$, the trajectory must go somewhere else. This "somewhere else" is often a cycle, where the system visits a finite sequence of points repeatedly.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($k$-Cycle)</span></p>

A **$k$-cycle** is a periodic trajectory where the system iterates through $K$ distinct points. A 2-cycle, for example, is a pair of points $\lbrace x_a, x_b \rbrace$ such that:

$$f(x_a) = x_b \quad \text{and} \quad f(x_b) = x_a$$

The system perpetually jumps back and forth between these two points.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(A 2-Cycle in the Logistic Map)</span></p>

For a parameter value of $\alpha = 3.3$, the logistic map exhibits a stable 2-cycle.

* If we trace the evolution of the system with a cobweb plot, we see that the trajectory, instead of spiraling into a single fixed point, converges to a rectangular box that bounces between two distinct values.
* A time-series plot of $x_t$ versus $t$ would show the system initially behaving transiently before settling into a steady oscillation between two values.

This behavior, where an increase in a parameter causes a stable fixed point to lose stability and give rise to a stable 2-cycle, is a common route to more complex dynamics in nonlinear systems.

</div>

### Cycles in Iterated Maps

#### Two-Cycles as Fixed Points of the Iterated Map

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(From Cycles to Fixed Points)</span></p>

When analyzing discrete maps, we often encounter cycles, where the system iterates between a set of distinct points. A two-cycle, for instance, involves iterating between two different points. If we start at one point, a single application of the map takes us to the second point, and the next application takes us back to the first.

This observation leads to a powerful insight: a point on a two-cycle returns to its original position after exactly two applications of the map. Therefore, any point belonging to a two-cycle of a map $f$ must be a fixed point of the twice-iterated map, denoted as $f^2(x) = f(f(x))$. This reframes the problem of finding cycles into the more familiar problem of finding fixed points, albeit for a more complex function.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Logistic Map Iterated Twice)</span></p>

Let's consider the logistic map, which is a second-order polynomial (a "square map"). If we construct the twice-iterated map, $f^2(x) = f(f(x))$, by substituting the logistic equation into itself, the resulting function is a fourth-order polynomial.

The explicit form of this map is given by:

$$f^2(x^*) = - \alpha^3 (x^*)^4 + 2 \alpha^3 (x^*)^3 - (\alpha^2 + \alpha^3) (x^*)^2$$

The key takeaway is that an iterated map yields another function, and the fixed points of this new function correspond to the cycles of the original map.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/OneAndSecondOrderMap.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>First- and second-order return maps of logistic function for $\alpha=3.3$, illustrating $2$-cycle in first-order map as fixed points of second-order map.</figcaption>
</figure>

#### Stability of Cycles

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stability via the Iterated Map)</span></p>

Since we can treat the points of a cycle as fixed points of an iterated map, we can determine the stability of the cycle by analyzing the stability of these corresponding fixed points. The method is the same as for simple fixed points: we check the slope of the function at the fixed point.

For a two-cycle of the map $f$, its stability is determined by the derivative of the twice-iterated map, $f^2(x)$, at the points of the cycle.

The plot of $f^2(x)$ for the logistic map at $\alpha = 3.3$ reveals four fixed points. Two of these are the original, now unstable, fixed points of $f(x)$. The other two are the points that constitute the stable two-cycle.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Stability of a Cycle)</span></p>

The stability of a $k$-cycle can be determined by checking the slope of the $k$-times iterated function, $f^k(x)$, at any point $x_i^\ast$ on the cycle. 

* The cycle is stable if the absolute value of this slope is less than one.

  $$\left| \frac{d}{dx} f^k(x) \bigg|_{x=x_i^*} \right| < 1$$

* The cycle is **unstable** if this value is greater than one.

</div>

#### Generalization to $k$-Cycles

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($k$-Cycle)</span></p>

For a continuous map $f$, a $k$-cycle is a set of $k$ distinct points, $\lbrace x_1^\ast, x_2^\ast, \ldots, x_k^\ast \rbrace$, which are visited sequentially by iteration of $f$. This implies two critical conditions:

1. **Fixed Point of the Iterated Map:** Each point $x_i^\ast$ in the set (for $i = 1, \ldots, k$) is a fixed point of the $k$-times iterated map.

   $$x_i^* = f^k(x_i^*)$$

2. **Minimality and Distinctness:** To be a true $k$-cycle, two additional constraints must be met:
   * $k$ must be the smallest integer for which the fixed-point condition holds. This ensures that a two-cycle is not misidentified as a four-cycle, for example.
   * All points in the set must be distinct: $x_i^\ast \neq x_j^\ast$ for all $i \neq j$. This prevents a lower-order cycle (like a fixed point where $x_1^\ast = x_2^\ast$) from being classified as a higher-order cycle.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Analytical Advantage of Iterated Maps)</span></p>

This framework provides a significant advantage. While the original map $f$ might be complex and its iterated version $f^k$ even more so, we at least have a closed-form expression. This closed form gives us direct, analytical access to the stability of cycles. This is a powerful tool that is not generally available for analyzing the stability of limit cycles in continuous-time systems described by differential equations.

</div>

### Poincaré Map

#### Motivation: Reducing Complexty of Analyzing Periodic Orbits

In the study of dynamical systems, one often wants to understand the long-term behavior of trajectories generated by a differential equation. For many nonlinear systems, directly analyzing continuous motion in time can be difficult, especially when one is interested in periodic behavior or the geometry of trajectories near a periodic orbit. A powerful idea for reducing this complexity is the **Poincaré map**.

The Poincaré map transforms a continuous-time dynamical system into a discrete-time system by recording only selected moments of the motion: the successive times at which a trajectory intersects a chosen surface. In this way, problems about continuous trajectories can often be reduced to problems about iterating a map. This makes the Poincaré map one of the central tools in the qualitative theory of differential equations.

Earlier, this idea can be used to prove the existence of periodic orbits in systems such as the driven pendulum or the Josephson junction. Here we discuss the Poincaré map in a more general setting.

#### Continuous-Time Dynamics and the Surface of Section

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poincaré Map Idea)</span></p>

Consider an $n$-dimensional dynamical system

$$\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}),$$

where $\mathbf{x} \in \mathbb{R}^n$ and $\mathbf{f}$ is a vector field. The trajectory of a point $\mathbf{x}_0$ is the solution of this differential equation starting from $\mathbf{x}_0$.

Suppose the flow of the system has a swirling structure, for example near a periodic orbit. Instead of following the entire continuous trajectory, we choose a hypersurface $S$ of dimension $n-1$, called a **surface of section**, and observe where the trajectory crosses this surface.

The surface $S$ must be **transverse** to the flow. This means that the vector field is not tangent to the surface at the points of interest. Geometrically, trajectories must pass through $S$, not slide along it. This condition ensures that intersections are well-defined and occur as isolated events.

The surface of section acts like a checkpoint. Each time a trajectory returns to $S$, we record its location. From these successive intersection points, we build a discrete dynamical system.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/poincare_map.png' | relative_url }}" alt="a" loading="lazy">
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Poincaré Map)</span></p>

Let $S$ be a transverse surface of section. If a trajectory starting on $S$ intersects $S$ again, we define a mapping from one intersection point to the next.

If $\mathbf{x}_k \in S$ denotes the $k$-th intersection of a trajectory with $S$, then the **Poincaré map** $P$ is defined by

$$\mathbf{x}_{k+1} = P(\mathbf{x}_k)$$

Thus,

$$P : S \to S$$

This map takes a point on the section and returns the next point at which the trajectory intersects the same section.

The crucial idea is that the original system is continuous in time, whereas the Poincaré map is discrete. Instead of studying the full motion $\mathbf{x}(t)$, we study the sequence

$$\mathbf{x}_0,\mathbf{x}_1,\mathbf{x}_2,\dots$$

generated by repeated application of $P$.

This reduction is especially useful because $S$ has dimension $n-1$, so the Poincaré map typically lives in one dimension lower than the original phase space.

</div>

#### Fixed Points, Stabilities and Periodic Orbits with Poincaré Map

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Fixed Points and Periodic Orbits)</span></p>

The most important connection is the one between **fixed points** of the Poincaré map and **closed orbits** of the original system.

Suppose $\mathbf{x}^\ast\in S$ satisfies

$$P(\mathbf{x}^\ast) = \mathbf{x}^\ast.$$

Then $\mathbf{x}^\ast$ is a fixed point of the Poincaré map. By definition, the trajectory starting at $\mathbf{x}^\ast$ leaves the section and returns to the same point after some time $T$. Therefore, the corresponding trajectory in the original system is a **closed orbit** or **periodic orbit**.

This is one of the main reasons the Poincaré map is so valuable: it turns the difficult problem of finding periodic orbits of a differential equation into the often simpler problem of finding fixed points of a map.

In summary:

* a fixed point of $P$ corresponds to a periodic orbit of the flow,
* the return time to the section is the period $T$ of that orbit.

This discrete viewpoint is often much easier to handle, both conceptually and analytically.

</div>

The Poincaré map is useful not only for proving that a periodic orbit exists, but also for determining whether it is stable.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Stability of a Periodic Orbit)</span></p>

Suppose $\mathbf{x}^\ast$ is a fixed point of $P$. Consider an initial point $\mathbf{x}_0$ near $\mathbf{x}^\ast$. If repeated application of the map gives points

$$\mathbf{x}_1=P(\mathbf{x}_0), \quad \mathbf{x}_2=P(\mathbf{x}_1), \quad \dots$$

that approach $\mathbf{x}^\ast$, then the fixed point is stable. In the original flow, this means nearby trajectories spiral toward the periodic orbit. On the other hand, if the iterates move away from $\mathbf{x}^\ast$, then the fixed point is unstable, and the corresponding periodic orbit is unstable as well.

Thus, the local behavior of $P$ near a fixed point tells us the local behavior of the flow near the associated closed orbit.

In practice, one often studies the linearization of the Poincaré map near $\mathbf{x}^\ast$. The eigenvalues of this linearized map determine whether perturbations decay or grow from one return to the next. Even without carrying out the full calculation, the conceptual message is clear: **stability of a periodic orbit is encoded in the stability of the fixed point of the Poincaré map**.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/Limit_cycle_Poincare_map.svg.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Stable limit cycle (shown in bold) and two other trajectories spiraling into it</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why the Poincaré Map Is Powerful)</span></p>

The Poincaré map offers several important advantages.

First, it reduces dimension. Instead of analyzing motion in an $n$-dimensional phase space, one studies a map on an $(n-1)$-dimensional section.

Second, it replaces a continuous-time problem with a discrete-time one. Iterating a map is often simpler than solving a nonlinear differential equation explicitly.

Third, it captures the essential recurrent behavior of the flow. If the system repeatedly revisits the same region of phase space, the Poincaré map isolates these returns and makes their structure visible.

For these reasons, Poincaré maps are useful in the study of:

* periodic orbits,
* stability of oscillations,
* bifurcations of periodic motion,
* and, more broadly, complicated recurrent behavior in nonlinear systems.

They are particularly valuable in systems with rotational or oscillatory motion, where trajectories repeatedly pass near the same region.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Limitations and Practical Difficulties)</span></p>

Although the idea of the Poincaré map is elegant, one important difficulty remains: in most systems it is not possible to write down an explicit formula for the map $P$.

To compute $P(\mathbf{x})$, one generally has to start at $\mathbf{x}\in S$, follow the trajectory determined by the differential equation, and wait until the trajectory hits $S$ again. This usually requires solving the system numerically. As a result, the Poincaré map is conceptually simple but not always easy to compute in practice.

This is the main limitation of the method. The theory tells us that periodic orbits correspond to fixed points of $P$, but the map itself is often only implicitly defined through the flow.

For that reason, examples in which $P$ can be computed explicitly are especially valuable. They illustrate the method in its clearest form and show how the general theory works in concrete cases.

</div>

### The Phase Description of Oscillators

In the study of dynamical systems, oscillators represent a fundamental class of behaviors characterized by periodic motion, often visualized as a closed orbit or limit cycle in the state space. To analyze these systems, particularly when they interact, it is incredibly useful to reduce their complex dynamics to a single, essential variable: the phase.

#### Flows on the Circle

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vector Field On Circle)</span></p>

The equation

$$\dot \theta = f(\theta)$$

corresonds to the **vector field on the circle**.

* $\theta$ is a **point on the circle** and is the
* $\dot \theta$ is a **velocity vector at that point**

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Vector Field on Circle via Phase Variable)</span></p>

For the system

$$\dot{\theta}=\sin\theta,$$

consider the motion on the unit circle, where $\theta=0$ points to the right and $\theta$ increases counterclockwise.

To draw the vector field, first identify the equilibria by solving

$$\dot{\theta}=0 \quad \Longrightarrow \quad \sin\theta=0$$

This gives two fixed points:

$$\theta^*=0 \quad \text{and} \quad \theta^*=\pi$$

To determine the direction of motion, look at the sign of $\sin\theta$:

* On the upper semicircle, $\sin\theta>0$, so $\dot{\theta}>0$. Thus the flow moves counterclockwise.
* On the lower semicircle, $\sin\theta<0$, so $\dot{\theta}<0$. Thus the flow moves clockwise.

Therefore, trajectories move away from $\theta=0$, so $\theta=0$ is **unstable**, while trajectories move toward $\theta=\pi$, so $\theta=\pi$ is **stable**.

This example is the circular version of the one-dimensional system $\dot{x}=\sin x$, but viewing it on the circle makes the geometry of the flow much easier to see.

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/flow_sinx_line.png' | relative_url }}" alt="a" loading="lazy">
    <figcaption>Vector field on the line</figcaption>
  </figure>
  
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/flow_sinx_circle.png' | relative_url }}" alt="a" loading="lazy">
    <figcaption>Vector field on the circle</figcaption>
  </figure>
</div>

</div>

#### The Phase Variable

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Phase Variable)</span></p>

* The phase of an oscillator describes its position along its limit cycle. It is represented by a **phase variable** or **angle**, typically denoted as $\theta$. 
* A single full iteration of the oscillator corresponds to the **phase variable** completing a full cycle. 
* By convention, the phase is often defined to evolve in the interval $[0, 2\pi]$, though other intervals such as $[0, 1]$ are also used.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Phase Dynamics)</span></p>

The central idea of this approach is to shift our focus from the full state-space variables of the oscillator to just its phase. By doing this, we can formulate a new, often simpler, differential equation that describes the evolution of the phase itself:

$$\dot{\theta} = f(\theta)$$

This simplification allows us to capture the essential timing and rhythm of the oscillator, which is paramount when studying phenomena like synchronization.

</div>

#### Uniform Oscillator: Constant Angular Velocity

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Uniform Oscillator)</span></p>

Consider the simplest case of an oscillator: one that traverses its limit cycle at a constant speed. This means its phase variable increases at a constant rate.

* **Dynamics:** The differential equation for the phase is linear:

  $$\dot{\theta} = \omega$$

  where $\omega$ is the constant angular velocity.

* **Solution:** The explicit equation for the phase at time $t$ is:

  $$\theta(t) = (\omega t + \theta_0) \pmod{2\pi}$$

  where $\theta_0$ is the initial phase at $t = 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Non-Uniform Speed in Nonlinear Oscillators)</span></p>

While this constant-speed model is a useful starting point, it is crucial to remember that it is not generally true for non-linear oscillators. In most realistic systems, an oscillator will speed up and slow down as it moves through different parts of its cycle. For instance, it might move quickly through one region of its state space and very slowly through another.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Nonuniform Oscillator)</span></p>

A **nonuniform oscillator** is a system on the circle whose phase $\theta$ evolves according to

$$\dot{\theta}=f(\theta),$$

where $f(\theta)$ is **$2\pi$-periodic** and has the **same sign for all $\theta$**, usually

$$f(\theta)>0 \quad \text{for all } \theta$$

This means the state keeps going around the circle forever in one direction, but its angular speed is **not constant**; it depends on where it is on the circle.

So:

* **uniform oscillator:** $\dot{\theta}=\omega$ with constant speed,
* **nonuniform oscillator:** $\dot{\theta}=f(\theta)$ with position-dependent speed.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Important Nonuniform Oscillator)</span></p>

The equation

$$\dot \theta = \omega - a\sin \theta$$

arises in many different branches of science and engineering.

</div>

#### Calculating the Oscillation Period

If we have the differential equation for the phase, $\dot{\theta} = f(\theta)$, we can derive a formula to calculate the temporal period of one full oscillation, $T_{\text{osc}}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Oscillation Period for 1D Oscillator)</span></p>

Oscillation period of *any* 1D osciallator defined by the angular speed $\dot \theta = f(\theta)$ is

$$T_{\text{osc}} = \int_0^{2\pi} \frac{1}{f(\theta)} \, d\theta$$

</div>

<details class="accordion" markdown="1">
<summary>Proof</summary>

The period $T_{\text{osc}}$ is, by definition, the time it takes to complete one cycle. We can express this with a simple integral:

$$T_{\text{osc}} = \int_0^{T_{\text{osc}}} dt$$

To connect this to the phase variable, we perform a change of variables from time $t$ to phase $\theta$. As time progresses from $0$ to $T_{\text{osc}}$, the phase progresses from $0$ to $2\pi$. We can introduce $d\theta$ into the integral:

$$T_{\text{osc}} = \int_0^{2\pi} \frac{dt}{d\theta} \, d\theta$$

We know the differential equation for the phase is $\frac{d\theta}{dt} = f(\theta)$. Therefore, its inverse is $\frac{dt}{d\theta} = \frac{1}{f(\theta)}$. Substituting this into the integral gives the final formula:

$$T_{\text{osc}} = \int_0^{2\pi} \frac{1}{f(\theta)} \, d\theta$$

</details>

#### Study Case: Saddle-Node Bifurcation in Nonuniform Oscillators

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Saddle-Node Bifurcation in Nonuniform Oscillators (I))</span></p>

Consider the differential equation

$$\dot{\theta} = \omega - a \sin \theta$$

To study equation this differential equation, we assume for simplicity that $\omega > 0$ and $a \geq 0$; the case of negative $\omega$ and $a$ is similar.

On the figure, $\omega$ represents the average value, while $a$ gives the amplitude.

**Vector fields**

* **$a=0$, equation becomes the equation of a uniform oscillator.** Introducing the parameter $a$ makes the motion around the circle uneven: the flow is fastest at $\theta=-\pi/2$ and slowest at $\theta=\pi/2$ (see Figure). This nonuniformity grows stronger as $a$ increases.

* **$a$ is slightly smaller than $\omega$, the oscillation becomes highly uneven:** the phase point $\theta(t)$ spends a long time moving through a **bottleneck** near $\theta=\pi/2$, then quickly travels through the rest of the circle on a much shorter timescale.

* **$a=\omega$, the oscillation disappears completely.** At this point, a half-stable fixed point is created at $\theta=\pi/2$ through a **saddle-node bifurcation**.

* **$a>\omega$, this half-stable fixed point splits into two fixed points, one stable and one unstable**. As time goes to infinity, every trajectory approaches the stable fixed point.

The same behavior can also be illustrated by drawing the vector fields directly on the circle, as in Figure.

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/saddle_point_bifurcation_oscillator_line.png' | relative_url }}" alt="a" loading="lazy">
    <figcaption>Vector field on the line</figcaption>
  </figure>
  
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/saddle_point_bifurcation_oscillator_circle.png' | relative_url }}" alt="a" loading="lazy">
    <figcaption>Vector field on the circle</figcaption>
  </figure>
</div>

</div>

<div id="no-container" style="margin:2em auto;max-width:1060px;">
  <h4 style="text-align:center;margin:0 0 .2em;">Interactive: Nonuniform Oscillator \(\dot{\theta}=\omega - a\sin\theta\)</h4>
  <p style="text-align:center;color:#888;font-size:.82em;margin:0 0 .5em;">
    Increase \(a\) past \(\omega\) to see the saddle-node bifurcation that kills the oscillation. The bottleneck and period divergence are visible in \(\theta(t)\).
  </p>
  <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:12px;">
    <div style="text-align:center;">
      <div id="no-ltitle" style="font-size:.85em;font-weight:600;margin-bottom:3px;">Flow: oscillating (a &lt; &omega;)</div>
      <canvas id="no-lc" style="border:1px solid #ddd;border-radius:3px;background:#fff;max-width:100%;"></canvas>
    </div>
    <div style="text-align:center;">
      <div id="no-rtitle" style="font-size:.85em;font-weight:600;margin-bottom:3px;">Phase evolution &theta;(t)</div>
      <canvas id="no-rc" style="border:1px solid #ddd;border-radius:3px;background:#fff;max-width:100%;"></canvas>
    </div>
  </div>
  <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin-top:8px;flex-wrap:wrap;">
    <span style="font-size:.85em;font-family:serif;">a =</span>
    <input type="range" id="no-a" min="0" max="3" step="0.01" value="0.5" style="width:180px;">
    <span id="no-a-val" style="font-size:.85em;font-family:serif;min-width:35px;">0.50</span>
    <span style="font-size:.85em;font-family:serif;margin-left:12px;">&omega; =</span>
    <input type="range" id="no-omega" min="0.1" max="3" step="0.1" value="1" style="width:140px;">
    <span id="no-omega-val" style="font-size:.85em;font-family:serif;min-width:30px;">1.0</span>
  </div>
  <div id="no-info" style="text-align:center;font-size:.82em;margin-top:.4em;font-family:serif;color:#555;"></div>
</div>

<script>
(function(){
  var S=500,PI=Math.PI,PI2=2*PI;
  var omega=1,a=0.5;
  var lc=document.getElementById('no-lc'),rc=document.getElementById('no-rc');
  var aS=document.getElementById('no-a'),aV=document.getElementById('no-a-val');
  var oS=document.getElementById('no-omega'),oV=document.getElementById('no-omega-val');
  var dpr=window.devicePixelRatio||1;
  function initC(c){c.width=S*dpr;c.height=S*dpr;c.style.width=S+'px';c.style.height=S+'px';var x=c.getContext('2d');x.scale(dpr,dpr);return x;}
  var L=initC(lc),R=initC(rc);

  function f(th){return omega-a*Math.sin(th);}
  function eqs(){
    if(a<omega-.005)return[];
    if(Math.abs(a-omega)<.005)return[{th:PI/2,half:true}];
    var s=Math.asin(omega/a);
    return[{th:s,s:true},{th:PI-s,s:false}];
  }
  function per(){return a<omega-1e-6?PI2/Math.sqrt(omega*omega-a*a):Infinity;}

  function integrate(th0,tMax,dt){
    var pts=[{t:0,th:th0}],th=th0;
    for(var t=dt;t<=tMax+dt/2;t+=dt){
      var k1=f(th),k2=f(th+.5*dt*k1),k3=f(th+.5*dt*k2),k4=f(th+dt*k3);
      th+=dt/6*(k1+2*k2+2*k3+k4);pts.push({t:t,th:th});
    }
    return pts;
  }

  function ln2(c,a2,b,d,e){c.beginPath();c.moveTo(a2,b);c.lineTo(d,e);c.stroke();}
  function circ(c,x,y,r,fl){c.beginPath();c.arc(x,y,r,0,PI2);if(fl)c.fill();else c.stroke();}
  function halfDot(c,cx,cy,rd){
    c.fillStyle='#4CAF50';c.beginPath();c.arc(cx,cy,rd,PI*.5,PI*1.5);c.closePath();c.fill();
    c.fillStyle='#F44336';c.beginPath();c.arc(cx,cy,rd,-PI*.5,PI*.5);c.closePath();c.fill();
    c.strokeStyle='#555';c.lineWidth=1.5;circ(c,cx,cy,rd,false);
  }

  function drawLeft(){
    L.clearRect(0,0,S,S);
    var yLo=Math.min(omega-a,0)-.4,yHi=Math.max(omega+a,1)+.4;
    var PL=40,PR2=12,PT=18,PB=55,W=S-PL-PR2,H=S-PT-PB;
    function lx(th){return PL+th/PI2*W;}
    function ly(v){return PT+(yHi-v)/(yHi-yLo)*H;}

    // Grid
    L.strokeStyle='#f0f0f0';L.lineWidth=.5;
    [PI/2,PI,3*PI/2].forEach(function(th){var x=lx(th);ln2(L,x,PT,x,PT+H);});
    var step=Math.max(.5,Math.round((yHi-yLo)/6*2)/2);
    for(var v=Math.ceil(yLo/step)*step;v<=yHi;v+=step){if(Math.abs(v)<.01)continue;var y=ly(v);ln2(L,PL,y,PL+W,y);}

    // Zero line (phase line)
    var y0=ly(0);
    if(y0>PT&&y0<PT+H){L.strokeStyle='#333';L.lineWidth=2;ln2(L,PL,y0,PL+W,y0);}
    L.strokeStyle='#81D4FA';L.lineWidth=1;ln2(L,PL,PT,PL,PT+H);

    // Shading
    var eq=eqs(),roots=eq.map(function(e){return e.th;}).sort(function(a2,b2){return a2-b2;});
    var segs=[0].concat(roots).concat([PI2]);
    for(var i=0;i<segs.length-1;i++){
      var thA=segs[i],thB=segs[i+1],mid=(thA+thB)/2;
      var col=f(mid)>0?'rgba(255,152,0,0.1)':'rgba(33,150,243,0.15)';
      L.fillStyle=col;L.beginPath();L.moveTo(lx(thA),y0);
      for(var th=thA;th<=thB+.02;th+=.03)L.lineTo(lx(th),ly(Math.max(yLo,Math.min(yHi,f(th)))));
      L.lineTo(lx(thB),y0);L.closePath();L.fill();
    }

    // Curve
    L.strokeStyle='#1976D2';L.lineWidth=2.5;L.beginPath();
    for(var th=0;th<=PI2;th+=.02){if(th<.01)L.moveTo(lx(th),ly(f(th)));else L.lineTo(lx(th),ly(f(th)));}
    L.stroke();

    // ω reference line
    var yw=ly(omega);
    if(yw>PT&&yw<PT+H){L.save();L.setLineDash([4,3]);L.strokeStyle='#aaa';L.lineWidth=1;ln2(L,PL,yw,PL+W,yw);L.restore();}

    // Flow arrows
    var arrowY=Math.min(y0+22,PT+H+15);
    for(var th=.2;th<PI2;th+=.28){
      var skip=false;eq.forEach(function(e){if(Math.abs(th-e.th)<.15)skip=true;});
      if(skip)continue;
      var fv=f(th);if(Math.abs(fv)<.01)continue;
      var dir=fv>0?1:-1,sz=Math.min(10,Math.max(4,Math.abs(fv)*2.5)),cx=lx(th);
      L.fillStyle=fv>0?'rgba(230,81,0,0.55)':'rgba(21,101,192,0.55)';
      L.beginPath();L.moveTo(cx+dir*sz,arrowY);L.lineTo(cx-dir*sz*.55,arrowY-sz*.5);L.lineTo(cx-dir*sz*.55,arrowY+sz*.5);L.closePath();L.fill();
    }

    // Eq dots
    eq.forEach(function(e){
      var cx=lx(e.th),cy=y0;
      if(e.half)halfDot(L,cx,cy,9);
      else if(e.s){L.fillStyle='#4CAF50';circ(L,cx,cy,9,true);L.strokeStyle='#2E7D32';L.lineWidth=2;circ(L,cx,cy,9,false);}
      else{L.fillStyle='#F44336';circ(L,cx,cy,9,true);L.strokeStyle='#C62828';L.lineWidth=2;circ(L,cx,cy,9,false);}
    });

    // Labels
    L.font='12px "Times New Roman",serif';L.fillStyle='#888';
    L.fillText('\u03B8',PL+W+3,y0+4);L.fillText('\u03B8\u0307',PL+5,PT-3);
    L.font='9px sans-serif';L.fillStyle='#bbb';
    [{v:0,l:'0'},{v:PI/2,l:'\u03C0/2'},{v:PI,l:'\u03C0'},{v:3*PI/2,l:'3\u03C0/2'},{v:PI2,l:'2\u03C0'}].forEach(function(t){
      L.fillText(t.l,lx(t.v)-6,y0>PT+H-10?y0-6:y0+14);
    });
    for(var v=Math.ceil(yLo/step)*step;v<=yHi;v+=step){if(Math.abs(v)<.01)continue;L.fillText(v.toFixed(1),2,ly(v)+3);}
    L.font='11px "Times New Roman",serif';L.fillStyle='#1976D2';
    L.fillText('\u03B8\u0307 = '+omega.toFixed(1)+' \u2212 '+a.toFixed(2)+'sin\u03B8',PL+5,PT+14);
    if(yw>PT&&yw<PT+H){L.fillStyle='#aaa';L.fillText('\u03C9',PL+W-12,yw-4);}
  }

  function drawRight(){
    R.clearRect(0,0,S,S);
    var T=per();
    var totalT=T<Infinity?Math.min(Math.max(3*T,8),50):25;
    var dt=.02,pts=integrate(0,totalT,dt);
    var thMax=pts[pts.length-1].th;
    var tLo=0,tHi=totalT,thLo=-.3,thHi=Math.max(thMax+.5,PI+.5);
    var PL=45,PR2=12,PT=18,PB=35,W=S-PL-PR2,H=S-PT-PB;
    function rx(t){return PL+(t-tLo)/(tHi-tLo)*W;}
    function ry(th){return PT+(thHi-th)/(thHi-thLo)*H;}

    // θ=nπ grid
    R.strokeStyle='#f0f0f0';R.lineWidth=.5;
    for(var n=1;n*PI<=thHi;n++){var y=ry(n*PI);if(y>PT&&y<PT+H)ln2(R,PL,y,PL+W,y);}
    var tStep=T<Infinity&&T<15?T:Math.max(1,Math.round(totalT/8));
    for(var t=tStep;t<tHi;t+=tStep){var x=rx(t);ln2(R,x,PT,x,PT+H);}
    var y0=ry(0);if(y0>PT&&y0<PT+H){R.strokeStyle='#ddd';R.lineWidth=1;ln2(R,PL,y0,PL+W,y0);}
    R.strokeStyle='#81D4FA';R.lineWidth=1;ln2(R,PL,PT,PL,PT+H);ln2(R,PL,PT+H,PL+W,PT+H);

    // Period markers
    if(T<Infinity&&T<totalT){
      R.save();R.setLineDash([4,3]);R.strokeStyle='#FF9800';R.lineWidth=1;
      for(var k=1;k*T<=tHi;k++)ln2(R,rx(k*T),PT,rx(k*T),PT+H);
      R.restore();
    }

    // θ=nπ labels
    R.font='9px sans-serif';R.fillStyle='#bbb';
    for(var n=0;n*PI<=thHi;n++){var y=ry(n*PI);if(y>PT+5&&y<PT+H-5)R.fillText(n?n+'\u03C0':'0',2,y+3);}

    // Fixed point line
    if(a>=omega-.005){
      var eq2=eqs();eq2.forEach(function(e){
        if(!e.s&&!e.half)return;
        var y=ry(e.th);if(y>PT&&y<PT+H){
          R.save();R.setLineDash([5,3]);R.strokeStyle=e.half?'#FF9800':'#4CAF50';R.lineWidth=1.5;
          ln2(R,PL,y,PL+W,y);R.restore();
          R.font='10px "Times New Roman",serif';R.fillStyle=e.half?'#FF9800':'#4CAF50';
          R.fillText('\u03B8*='+e.th.toFixed(2),PL+W-55,y-4);
        }
      });
    }

    // Trajectory
    R.strokeStyle='#1565C0';R.lineWidth=2;R.beginPath();
    R.moveTo(rx(pts[0].t),ry(pts[0].th));
    for(var i=1;i<pts.length;i++)R.lineTo(rx(pts[i].t),ry(pts[i].th));
    R.stroke();
    R.fillStyle='#1565C0';circ(R,rx(0),ry(0),4,true);

    // Axis labels
    R.font='12px "Times New Roman",serif';R.fillStyle='#888';
    R.fillText('t',PL+W+3,PT+H+4);R.fillText('\u03B8(t)',PL+3,PT-3);
    R.font='9px sans-serif';R.fillStyle='#bbb';
    if(T<Infinity&&T<15){for(var k=1;k*T<=tHi;k++){R.fillStyle='#FF9800';R.fillText(k+'T',rx(k*T)-4,PT+H+13);}}
    else{for(var t=tStep;t<tHi;t+=tStep){R.fillText(t.toFixed(0),rx(t)-4,PT+H+13);}}
    R.font='11px "Times New Roman",serif';R.fillStyle='#333';
    R.fillText(T<Infinity?'T = '+T.toFixed(2):'T = \u221E',PL+5,PT+14);
  }

  function updInfo(){
    var el=document.getElementById('no-info'),T=per(),eq=eqs();
    var t='a = '+a.toFixed(2)+' &nbsp;|&nbsp; \u03C9 = '+omega.toFixed(1)+' &nbsp;|&nbsp; ';
    if(a<omega-.005)t+='<span style="color:#1565C0">Oscillating</span> &nbsp;|&nbsp; T = '+T.toFixed(3)+' &nbsp;|&nbsp; a/\u03C9 = '+(a/omega).toFixed(2);
    else if(Math.abs(a-omega)<.005)t+='<span style="color:#FF9800">Saddle-node bifurcation at a = \u03C9</span> &nbsp;|&nbsp; T \u2192 \u221E';
    else t+='<span style="color:#F44336">No oscillation</span> &nbsp;|&nbsp; stable \u03B8* = '+eq[0].th.toFixed(3)+', unstable \u03B8* = '+eq[1].th.toFixed(3);
    el.innerHTML=t;
  }
  function updTitles(){
    document.getElementById('no-ltitle').textContent=a<omega-.005?'Flow: oscillating (a < \u03C9)':Math.abs(a-omega)<.005?'Flow: saddle-node (a = \u03C9)':'Flow: fixed points (a > \u03C9)';
    var T=per();
    document.getElementById('no-rtitle').textContent=a<omega-.005?'Phase \u03B8(t): period T = '+(T<100?T.toFixed(1):'\u221E'):'Phase \u03B8(t): convergence to \u03B8*';
  }
  function redraw(){drawLeft();drawRight();updInfo();updTitles();}

  aS.addEventListener('input',function(){a=parseFloat(this.value);aV.textContent=a.toFixed(2);redraw();});
  oS.addEventListener('input',function(){omega=parseFloat(this.value);oV.textContent=omega.toFixed(1);redraw();});
  redraw();
})();
</script>

#### Period Divergence at Saddle-Node Bifurcation Point

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Period Divergence at Saddle-Node Bifurcation Point (II))</span></p>

Apply linear stability analysis to determine the nature of the fixed points when $a>\omega$.

The fixed points $\theta^\ast$ satisfy

$$\sin\theta^*=\frac{\omega}{a}, \qquad \cos\theta^*=\pm\sqrt{1-\left(\frac{\omega}{a}\right)^2}$$

Their stability is governed by the derivative

$$f'(\theta^*)=-a\cos\theta^* =\mp a\sqrt{1-\left(\frac{\omega}{a}\right)^2}$$

Therefore, the fixed point for which $\cos\theta^\ast>0$ is stable, because in that case $f'(\theta^\ast)<0$.

**Oscillation period**

For the case $a<\omega$, the oscillation period can be computed explicitly. The time needed for $\theta$ to increase by $2\pi$ is

$$
T=\int dt
=\int_0^{2\pi}\frac{dt}{d\theta}d\theta
=\int_0^{2\pi}\frac{d\theta}{\omega-a\sin\theta},
$$

$$T=\frac{2\pi}{\sqrt{\omega^2-a^2}}$$

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/period_divergence.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Graph of $T$ as a function of $a$. Period divergence at the bifurcation point $a=w$.</figcaption>
</figure>

When $a=0$, formula for the period $T$ simplifies to

$$T=\frac{2\pi}{\omega},$$

which is the standard period of a uniform oscillator. As $a$ increases, the period also increases, and it tends to infinity as $a\to\omega^{-}$.

To estimate how this divergence occurs, observe that

$$
\sqrt{\omega^2-a^2}
=\sqrt{\omega+a}\sqrt{\omega-a}
\approx \sqrt{2\omega}\sqrt{\omega-a}
\quad \text{as } a\to\omega^{-}.
$$

Hence,

$$
T\approx
\left(\frac{\pi\sqrt{2}}{\sqrt{\omega}}\right)
\frac{1}{\sqrt{\omega-a}}.
$$

This shows that $T$ diverges like

$$
(a_c-a)^{-1/2},
\qquad \text{with } a_c=\omega.
$$

So the blow-up follows a **square-root scaling law**.

</div>

#### Ghosts and Bottlenecks near Saddle-Node Bifurcation Point

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Ghosts and Bottlenecks near Saddle-Node Bifurcation Point (III))</span></p>

The **square-root scaling law derived above is a very common feature of systems near a saddle-node bifurcation**. Right after two fixed points merge and disappear, their influence does not vanish immediately. Instead, a leftover effect remains — often called a **saddle-node ghost** — which causes trajectories to move very slowly through a **bottleneck** region.

Consider

$$\dot{\theta}=\omega-a\sin\theta$$

while gradually decreasing $a$ from values with $a>\omega$. As $a$ becomes smaller, the two fixed points move toward one another, collide, and then disappear. This is the same sequence shown earlier in Figure, but now it should be interpreted from right to left. When $a$ is just below $\omega$, the fixed points near $\pi/2$ are gone, but their former presence is still noticeable through the saddle-node ghost (Figure).

A plot of $\theta(t)$: the trajectory spends almost all of its time slowly crossing the bottleneck region.

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/bottleneck_due_to_ghost.png' | relative_url }}" alt="a" loading="lazy">
    <!-- <figcaption>Vector field on the line</figcaption> -->
  </figure>
  
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/bottleneck_in_phase_variable.png' | relative_url }}" alt="a" loading="lazy">
    <!-- <figcaption>Vector field on the circle</figcaption> -->
  </figure>
</div>

We now want to find a general scaling law for the time needed to pass through such a bottleneck. The essential part of the dynamics is the behavior of $\dot{\theta}$ very close to its minimum, because the time spent there dominates all other timescales in the system. In general, near that minimum, $\dot{\theta}$ has an approximately parabolic shape. This greatly simplifies the analysis, since the dynamics can then be reduced to the normal form of a saddle-node bifurcation. After a suitable local rescaling of space, the vector field can be written as

$$\dot{x}=r+x^2$$

where $r$ measures the distance from the bifurcation and satisfies $0<r\ll 1$. The graph of $\dot{x}$ is shown in Figure.

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/saddle-node-normal-form.png' | relative_url }}" alt="a" loading="lazy">
</figure>

To estimate how long the system remains in the bottleneck, we compute the time required for $x$ to move from $-\infty$ to $+\infty$, meaning from one side of the bottleneck to the other. This gives

$$
T_{\text{bottleneck}}
\approx
\int_{-\infty}^{\infty}\frac{dx}{r+x^2} = \frac{\pi}{\sqrt{r}}.
$$

This confirms that the square-root scaling law is a general phenomenon.

</div>

### Systems of Uncoupled Oscillators

We now extend our analysis from a single oscillator to a system of multiple oscillators. To build our understanding, we first consider the case where two oscillators evolve independently, without any coupling between them.

#### The State-Space Torus

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Two Independent Runners on a Circular Track)</span></p>

Two runners move around the same circular track with constant angular speeds. Let Speedy complete one lap in $T_1$ seconds and Pokey complete one lap in $T_2$ seconds, where $T_2>T_1$. Since Speedy is faster, he will eventually catch up and overtake Pokey. We want to compute the time for one such overtake.

Let $\theta_1(t)$ and $\theta_2(t)$ denote the angular positions of Speedy and Pokey. Because each runner moves at constant speed,

$$
\dot{\theta}_1=\omega_1=\frac{2\pi}{T_1},
\qquad
\dot{\theta}_2=\omega_2=\frac{2\pi}{T_2}
$$

To describe when Speedy laps Pokey, define the phase difference

$$\phi=\theta_1-\theta_2$$

Speedy has lapped Pokey once exactly when this phase difference has increased by $2\pi$. Differentiating gives

$$\dot{\phi}=\dot{\theta}_1-\dot{\theta}_2=\omega_1-\omega_2$$

So the relative angle grows at the constant rate $\omega_1-\omega_2$. Therefore, the time needed for $\phi$ to increase by $2\pi$ is

$$T_{\text{lap}}=\frac{2\pi}{\omega_1-\omega_2}$$

Substituting $\omega_1=2\pi/T_1$ and $\omega_2=2\pi/T_2$ gives

$$T_{\text{lap}} = \left(\frac{1}{T_1}-\frac{1}{T_2}\right)^{-1}$$

---

Another more general solution is to trivially derive 

$$\theta_1(t) = \frac{2\pi}{T_1}t + \theta_1(0) \quad \text{and} \quad \theta_2(t) = \frac{2\pi}{T_2}t + \theta_2(0)$$

Then

$$\phi(t) = 2\pi(\frac{1}{T_1}-\frac{1}{T_2})t$$

We want 

$$\phi(t) (\text{mod } 1\pi) = 0 \implies (\frac{1}{T_1}-\frac{1}{T_2})t \in \mathbb{N}_0$$

$\implies t = (\frac{1}{T_1}-\frac{1}{T_2})^{-1}k \quad k \in \mathbb{N}_0$.

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/two_runners_example.png' | relative_url }}" alt="a" loading="lazy">
</figure>

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Torus as State Space)</span></p>

Since each oscillator is described by a variable on a circle (from $0$ to $2\pi$), the combined state space of two such oscillators, $(\theta_1, \theta_2)$, can be visualized as a torus (a donut shape). One phase variable, say $\theta_1$, represents motion around the main radius of the torus, while the other, $\theta_2$, represents motion around the circular cross-section.

A trajectory in this two-dimensional state space represents the simultaneous evolution of both phases. As both oscillators cycle, the combined state $(\theta_1(t), \theta_2(t))$ traces a path that coils around the surface of the torus. A central question arises: Under what conditions will this trajectory eventually return to its starting point, forming a closed orbit on the torus?

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/torus_state_space.png' | relative_url }}" alt="a" loading="lazy">
</figure>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/torus.gif' | relative_url }}" alt="a" loading="lazy">
</figure>

</div>

#### Closed Orbits and Rational Frequency Ratios

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Commensurate Frequencies)</span></p>

Two frequencies, $\omega_1$ and $\omega_2$, are **commensurate** if their ratio is a rational number. That is, there exist two integers, $p$ and $q$, such that:

$$\frac{\omega_1}{\omega_2} = \frac{p}{q}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(State-Space Torus as a Closed Orbit)</span></p>

A trajectory on the **state-space torus will be a closed orbit** if and only if the frequencies of the two oscillators are commensurate.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Physical Meaning of Commensurate Frequencies)</span></p>

This condition has a very clear physical meaning. If the ratio of frequencies is $\frac{p}{q}$, it means that the first oscillator completes exactly $p$ cycles in the same amount of time that the second oscillator completes exactly $q$ cycles. After this time has elapsed, both oscillators will have returned to their initial phases simultaneously, thus closing the trajectory on the torus.

For instance, if $\omega_1$ is three times faster than $\omega_2$ ($\frac{\omega_1}{\omega_2} = \frac{3}{1}$), the first runner will complete three laps around the track in the exact time it takes the second runner to complete one. At that moment, they are both back at their starting positions.

</div>

#### Quasi-periodicity and Irrational Frequency Ratios

If the condition for a closed orbit is not met, a fascinating and more complex behavior emerges.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Quasi-periodicity)</span></p>

If the ratio of the oscillator frequencies $\frac{\omega_1}{\omega_2}$ is an irrational number, the trajectory on the torus will never close. Instead, it will wind around indefinitely, eventually passing arbitrarily close to every point on the surface of the torus. This motion is called **quasi-periodic**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Quasi-periodicity occurs only on the torus)</span></p>

**Quasiperiodicity** is significant because it is a new type of long-term behavior. Unlike the earlier entries (fixed point, closed orbit, homoclinic and heteroclinic orbits and cycles), **quasiperiodicity occurs only on the torus**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A Middle Ground Between Periodicity and Chaos)</span></p>

**Quasi-periodic motion is an interesting middle ground between simple periodic behavior (like a limit cycle) and the more complex behavior of chaos.** The system is "almost" periodic, but because the frequencies never align perfectly, the trajectory never repeats. Over time, the path of the system state will densely fill the entire surface of the torus. This is a unique property that arises in systems of two or more oscillators.

Furthermore, when the ration is irrational, each trajectory is **dense** on the torus: in other words, each trajectory comes arbitrarily close to any given point
on the torus. This is not to say that the trajectory passes through each point; it just comes arbitrarily close.

</div>

<div id="uo-container" style="margin:2em auto;max-width:1060px;">
  <h4 style="text-align:center;margin:0 0 .2em;">Interactive: Uncoupled Oscillators on the Torus</h4>
  <p style="text-align:center;color:#888;font-size:.82em;margin:0 0 .5em;">
    Set \(\omega_1,\omega_2\). Rational ratio \(\to\) closed orbit. Irrational \(\to\) quasi-periodic (dense on torus). Increase wraps for irrational presets to watch the torus fill.
  </p>
  <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:12px;">
    <div style="text-align:center;">
      <div style="font-size:.85em;font-weight:600;margin-bottom:3px;">Flat torus (&theta;&sub1;, &theta;&sub2;)</div>
      <canvas id="uo-fc" style="border:1px solid #ddd;border-radius:3px;background:#fff;max-width:100%;"></canvas>
    </div>
    <div style="text-align:center;">
      <div style="font-size:.85em;font-weight:600;margin-bottom:3px;">3D Torus</div>
      <canvas id="uo-tc" style="border:1px solid #ddd;border-radius:3px;background:#fff;max-width:100%;"></canvas>
    </div>
  </div>
  <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin-top:8px;flex-wrap:wrap;">
    <span style="font-size:.85em;font-family:serif;">&omega;&sub1; =</span>
    <input type="range" id="uo-o1" min="0.1" max="5" step="0.01" value="2" style="width:120px;">
    <span id="uo-o1v" style="font-size:.85em;font-family:serif;min-width:35px;">2.00</span>
    <span style="font-size:.85em;font-family:serif;margin-left:6px;">&omega;&sub2; =</span>
    <input type="range" id="uo-o2" min="0.1" max="5" step="0.01" value="1" style="width:120px;">
    <span id="uo-o2v" style="font-size:.85em;font-family:serif;min-width:35px;">1.00</span>
    <span style="font-size:.85em;font-family:serif;margin-left:6px;">wraps =</span>
    <input type="range" id="uo-wr" min="1" max="150" step="1" value="5" style="width:220px;">
    <span id="uo-wrv" style="font-size:.85em;font-family:serif;min-width:25px;">5</span>
  </div>
  <div id="uo-presets" style="display:flex;gap:5px;justify-content:center;margin-top:6px;flex-wrap:wrap;"></div>
  <div id="uo-info" style="text-align:center;font-size:.82em;margin-top:.4em;font-family:serif;color:#555;"></div>
</div>

<script>
(function(){
  var S=500,PI=Math.PI,PI2=2*PI;
  var omega1=2,omega2=1,wraps=5;
  var Rm=1.5,rm=0.55; // torus radii
  var vEl=0.45,vAz=-0.35; // view angles

  var fc=document.getElementById('uo-fc'),tc=document.getElementById('uo-tc');
  var o1S=document.getElementById('uo-o1'),o1V=document.getElementById('uo-o1v');
  var o2S=document.getElementById('uo-o2'),o2V=document.getElementById('uo-o2v');
  var wrS=document.getElementById('uo-wr'),wrV=document.getElementById('uo-wrv');
  var dpr=window.devicePixelRatio||1;
  function initC(c){c.width=S*dpr;c.height=S*dpr;c.style.width=S+'px';c.style.height=S+'px';var x=c.getContext('2d');x.scale(dpr,dpr);return x;}
  var F=initC(fc),T=initC(tc);

  // Preset buttons
  var presets=[
    {l:'1:1',w1:1,w2:1},{l:'2:1',w1:2,w2:1},{l:'3:2',w1:3,w2:2},{l:'5:3',w1:5,w2:3},{l:'5:4',w1:5,w2:4},
    {l:'\u221A2 : 1',w1:Math.sqrt(2),w2:1},{l:'\u03C0 : 1',w1:PI,w2:1},{l:'e : 1',w1:Math.E,w2:1}
  ];
  var pDiv=document.getElementById('uo-presets');
  presets.forEach(function(pr){
    var b=document.createElement('button');
    b.textContent=pr.l;
    b.style.cssText='font-size:.78em;padding:2px 8px;border:1px solid #ccc;border-radius:3px;background:#f8f8f8;cursor:pointer;font-family:serif;';
    b.addEventListener('click',function(){
      omega1=pr.w1;omega2=pr.w2;
      o1S.value=Math.min(5,omega1);o1V.textContent=omega1.toFixed(2);
      o2S.value=Math.min(5,omega2);o2V.textContent=omega2.toFixed(2);
      redraw();
    });
    pDiv.appendChild(b);
  });

  function tPt(th1,th2){return{x:(Rm+rm*Math.cos(th2))*Math.cos(th1),y:(Rm+rm*Math.cos(th2))*Math.sin(th1),z:rm*Math.sin(th2)};}
  function proj(p){
    var ca=Math.cos(vAz),sa=Math.sin(vAz),ce=Math.cos(vEl),se=Math.sin(vEl);
    var x1=p.x*ca-p.y*sa,y1=p.x*sa+p.y*ca;
    return{sx:x1,sy:-(y1*se+p.z*ce),d:y1*ce-p.z*se};
  }
  var SC=S/5;
  function tx(sx){return S/2+sx*SC;}
  function ty(sy){return S/2+sy*SC;}

  function ln2(c,a,b,d,e){c.beginPath();c.moveTo(a,b);c.lineTo(d,e);c.stroke();}

  function approxFrac(x,mq){
    var bp=Math.round(x),bq=1,be=Math.abs(x-bp);
    for(var q=2;q<=mq;q++){var p=Math.round(x*q),e=Math.abs(x-p/q);if(e<be){be=e;bp=p;bq=q;}}
    return{p:bp,q:bq,e:be};
  }

  // === FLAT TORUS ===
  function drawFlat(){
    F.clearRect(0,0,S,S);
    var PL=35,PR2=10,PT=15,PB=35,W=S-PL-PR2,H=S-PT-PB;
    function fx(th){return PL+((th%PI2+PI2)%PI2)/PI2*W;}
    function fy(th){return PT+H-((th%PI2+PI2)%PI2)/PI2*H;}

    // Grid
    F.strokeStyle='#f0f0f0';F.lineWidth=.5;
    [PI/2,PI,3*PI/2].forEach(function(v){
      var x=fx(v);ln2(F,x,PT,x,PT+H);
      var y=fy(v);ln2(F,PL,y,PL+W,y);
    });
    // Box
    F.strokeStyle='#aaa';F.lineWidth=1;
    F.strokeRect(PL,PT,W,H);

    // Trajectory
    var maxOm=Math.max(omega1,omega2);
    var totalT=wraps*PI2/maxOm;
    var nPts=Math.min(12000,Math.max(300,Math.floor(wraps*80)));
    var dt=totalT/nPts;
    var prevX=null,prevY=null;

    F.lineWidth=1.2;
    for(var i=0;i<=nPts;i++){
      var t=i*dt;
      var th1=(omega1*t)%PI2,th2=(omega2*t)%PI2;
      var cx=fx(th1),cy=fy(th2);
      if(prevX!==null){
        var dx=Math.abs(cx-prevX),dy=Math.abs(cy-prevY);
        if(dx<W*.4&&dy<H*.4){
          var blend=i/nPts;
          var r2=Math.floor(21+blend*209),g=Math.floor(101-blend*20),b=Math.floor(192-blend*192);
          F.strokeStyle='rgba('+r2+','+g+','+b+',0.5)';
          ln2(F,prevX,prevY,cx,cy);
        }
      }
      prevX=cx;prevY=cy;
    }

    // Labels
    F.font='12px "Times New Roman",serif';F.fillStyle='#888';
    F.fillText('\u03B8\u2081',PL+W+3,PT+H/2+4);F.fillText('\u03B8\u2082',PL+W/2-4,PT-3);
    F.font='9px sans-serif';F.fillStyle='#bbb';
    F.fillText('0',PL-10,PT+H+4);F.fillText('2\u03C0',PL+W-8,PT+H+13);F.fillText('2\u03C0',PL-14,PT+4);
  }

  // === 3D TORUS ===
  function draw3D(){
    T.clearRect(0,0,S,S);

    // Wireframe
    T.lineWidth=.5;
    // Longitudinal circles (constant θ₁)
    for(var i=0;i<16;i++){
      var th1=i*PI2/16;
      T.strokeStyle='rgba(180,180,180,0.25)';
      T.beginPath();
      for(var j=0;j<=60;j++){
        var th2=j*PI2/60,p=proj(tPt(th1,th2));
        if(!j)T.moveTo(tx(p.sx),ty(p.sy));else T.lineTo(tx(p.sx),ty(p.sy));
      }
      T.stroke();
    }
    // Meridional circles (constant θ₂)
    for(var i=0;i<12;i++){
      var th2=i*PI2/12;
      T.strokeStyle='rgba(180,180,180,0.2)';
      T.beginPath();
      for(var j=0;j<=80;j++){
        var th1=j*PI2/80,p=proj(tPt(th1,th2));
        if(!j)T.moveTo(tx(p.sx),ty(p.sy));else T.lineTo(tx(p.sx),ty(p.sy));
      }
      T.stroke();
    }

    // Trajectory — compute all points
    var maxOm=Math.max(omega1,omega2);
    var totalT=wraps*PI2/maxOm;
    var nPts=Math.min(10000,Math.max(300,Math.floor(wraps*80)));
    var dt=totalT/nPts;
    var pts=[],minD=1e9,maxD=-1e9;
    for(var i=0;i<=nPts;i++){
      var t=i*dt,p=proj(tPt(omega1*t,omega2*t));
      pts.push(p);
      if(p.d<minD)minD=p.d;if(p.d>maxD)maxD=p.d;
    }
    var dR=maxD-minD||1,medD=(minD+maxD)/2;

    // Back layer
    T.strokeStyle='rgba(21,101,192,0.12)';T.lineWidth=.6;T.beginPath();
    for(var i=1;i<pts.length;i++){
      if((pts[i-1].d+pts[i].d)/2>medD){T.moveTo(tx(pts[i-1].sx),ty(pts[i-1].sy));T.lineTo(tx(pts[i].sx),ty(pts[i].sy));}
    }
    T.stroke();
    // Front layer
    T.strokeStyle='rgba(21,101,192,0.55)';T.lineWidth=1.2;T.beginPath();
    for(var i=1;i<pts.length;i++){
      if((pts[i-1].d+pts[i].d)/2<=medD){T.moveTo(tx(pts[i-1].sx),ty(pts[i-1].sy));T.lineTo(tx(pts[i].sx),ty(pts[i].sy));}
    }
    T.stroke();

    // Start dot
    var sp=pts[0];
    T.fillStyle='#F44336';T.beginPath();T.arc(tx(sp.sx),ty(sp.sy),4,0,PI2);T.fill();
  }

  function updInfo(){
    var el=document.getElementById('uo-info');
    var ratio=omega1/omega2,fr=approxFrac(ratio,60);
    var t='\u03C9\u2081 = '+omega1.toFixed(3)+' &nbsp;|&nbsp; \u03C9\u2082 = '+omega2.toFixed(3)+' &nbsp;|&nbsp; \u03C9\u2081/\u03C9\u2082 = '+ratio.toFixed(4)+' &nbsp;|&nbsp; ';
    if(fr.e<.002)t+='<span style="color:#4CAF50">\u2248 '+fr.p+'/'+fr.q+' (closed orbit \u2014 '+fr.p+':'+fr.q+')</span>';
    else t+='<span style="color:#1565C0">quasi-periodic (irrational ratio)</span>';
    el.innerHTML=t;
  }
  function redraw(){drawFlat();draw3D();updInfo();}

  o1S.addEventListener('input',function(){omega1=parseFloat(this.value);o1V.textContent=omega1.toFixed(2);redraw();});
  o2S.addEventListener('input',function(){omega2=parseFloat(this.value);o2V.textContent=omega2.toFixed(2);redraw();});
  wrS.addEventListener('input',function(){wraps=parseInt(this.value);wrV.textContent=wraps;redraw();});
  redraw();
})();
</script>

### Coupled Oscillators and Synchronization

Having explored uncoupled systems, we now introduce coupling, allowing the oscillators to influence one another. This is where the most interesting phenomena, such as synchronization, arise.

#### Phase-Dependent Coupling

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Phase-Dependent Effects)</span></p>

In most non-linear oscillators, the effect of an external perturbation depends critically on the phase at which the perturbation is applied.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(A Biological Neuron)</span></p>

Consider a biological neuron that fires action potentials periodically. If we inject a small pulse of electrical current into the neuron, the effect will depend on when we inject it.

* If the neuron has just fired and is in its refractory period, the current may have very little effect.
* If the neuron is close to its firing threshold, the same pulse of current could be enough to trigger an action potential immediately, thereby advancing the phase of the oscillator.
* At other times, the pulse could delay the next action potential.

When two such neurons are coupled, they perturb each other back and forth. The first neuron fires, sending a signal that advances or delays the second neuron. The second neuron then fires, sending a signal back that advances or delays the first. It is through this dynamic interplay that the oscillators may eventually fall into a synchronized rhythm.

</div>

#### A Model for Two Coupled Oscillators

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Coupled Oscillators, Phase Space for Coupled Oscillators)</span></p>

Besides the plane and the cylinder, another important two-dimensional phase is the **torus**. It is the natural phase space for the systems of the form

$$\dot{\theta}_1 = f_1(\theta_1 - \theta_2)$$

$$\dot{\theta}_2 = f_2(\theta_2 - \theta_1)$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Model of Interaction Between Human Circadian Rhythms and the sleep-wake cycle)</span></p>

A common simple model for two **coupled oscillators** is given by the following system of differential equations:

$$\dot{\theta}_1 = \omega_1 + A \sin(\theta_1 - \theta_2)$$

$$\dot{\theta}_2 = \omega_2 + A \sin(\theta_2 - \theta_1)$$

* $\theta_1$, $\theta_2$ are **phases** of the oscillators.
* $\omega_1$, $\omega_2$ are their **natural frequencies**.
* $K_1$, $K_2$ are **coupling constants**.

Here, $K_1=A$, $K_2=A$ is the coupling strength, which determines how strongly the oscillators influence each other. The function $\sin(\cdot)$ is chosen as a simple periodic function to model the phase-dependent interaction, but other functions could be used.

</div>

#### The Phase Difference Equation

To analyze whether these oscillators will synchronize, the most effective technique is to study the evolution of their phase difference.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Phase Difference of Coupled Oscillators)</span></p>

Consider two coupled oscillators defined as

$$\dot{\theta}_1 = \omega_1 + A \sin(\theta_1 - \theta_2)$$

$$\dot{\theta}_2 = \omega_2 + A \sin(\theta_2 - \theta_1)$$

The **phase difference**, $\phi$, between the two oscillators is defined as:

$$\phi = \theta_1 - \theta_2$$

If the oscillators synchronize perfectly (zero phase locking), this difference will be constant at $\phi = 0$. If they lock at a different, but still constant, phase relationship, $\phi$ will be a non-zero constant. Therefore, synchronization corresponds to the phase difference $\phi$ reaching a stable fixed point.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callPut__name">(Differentail Equation of Phase Difference of Coupled Oscillators)</span></p>

The differential equation for the phase difference:

$$\dot{\phi} = (\omega_1 - \omega_2) + 2A \sin(\phi)$$

</div>

#### Phase Locking and Fixed Point Analysis

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Synchronization as a Fixed Point Problem)</span></p>

The concept of synchronization or phase locking is now transformed into a familiar problem: finding the fixed points of the scalar differential equation for $\phi$. A fixed point occurs where $\dot{\phi} = 0$, which means the phase difference stops changing and the oscillators are locked.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Identical Intrinsic Frequencies)</span></p>

Let's analyze the simplest case, where the two oscillators have the same intrinsic frequency, $\omega_1 = \omega_2$. The phase difference equation simplifies to:

$$\dot{\phi} = 2A \sin(\phi)$$

We can analyze the stability of the fixed points by examining the graph of $\dot{\phi}$ versus $\phi$.

* **Fixed Points:** The fixed points are where $\dot{\phi} = 0$, which occurs when $\sin(\phi) = 0$. This gives fixed points at $\phi = 0, \pi, 2\pi, \ldots$.
* **Stability Analysis:**
  * For a value of $\phi$ slightly greater than $0$, $\dot{\phi}$ is positive, causing $\phi$ to increase and move away from $0$.
  * For a value of $\phi$ slightly less than $\pi$ (but greater than $0$), $\dot{\phi}$ is positive, causing $\phi$ to increase towards $\pi$. For a value of $\phi$ slightly greater than $\pi$, $\dot{\phi}$ is negative, causing $\phi$ to decrease towards $\pi$.
  * Based on the lecture's analysis, the system exhibits one stable fixed point and one unstable fixed point. The stable fixed point represents a state of stable phase locking, where if the oscillators are perturbed slightly from this state, they will return to it. The unstable fixed point represents a state where any small perturbation will cause the oscillators to drift away from that phase relationship.
  * The state at the center is a stable fixed point. This is the point of phase locking.
  * The system dynamics cause the phase difference $\phi$ to be repelled from the unstable fixed point and attracted to the stable one.

The analysis can be extended by considering what happens as the difference in intrinsic frequencies, $\omega_1 - \omega_2$, is increased. This corresponds to vertically shifting the sine curve of the $\dot{\phi}$ vs. $\phi$ graph. The question then becomes: how large can this frequency difference be before synchronization is lost?

</div>

### Synchronization and Phase Locking in Coupled Oscillators

This section explores the fascinating phenomenon of synchronization, where coupled oscillating systems adjust their rhythms to lock into a common pattern. We will investigate the conditions under which this occurs and introduce a powerful graphical tool for mapping these behaviors.

#### The Concept of Phase Locking

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Phase Locking)</span></p>

* Two or more oscillators are said to be synchronized or **phase-locked** when the difference between their phases, $\phi$, becomes **constant over time**. 
* Mathematically, this corresponds to a stable fixed point of the phase difference dynamics. If $\dot{\phi} = 0$ for some phase difference $\phi^*$, the oscillators have achieved **phase locking**.

</div>

#### Conditions for Synchronization: Frequency Difference and Coupling Strength

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Interplay of Intrinsic Frequency and Coupling)</span></p>

The ability of two oscillators to synchronize is a result of the competition between their intrinsic properties and the strength of the interaction connecting them.

Consider two oscillators with intrinsic frequencies $\omega_1$ and $\omega_2$. The dynamics of their phase difference, $\phi$, can often be described by an equation of the form:

$$\dot{\phi} = (\omega_1 - \omega_2) + a \cdot g(\phi)$$

where $g(\phi)$ is a periodic coupling function (e.g., a sigmoid or sine function) and $a$ is the amplitude of the coupling.

* **The Role of Frequency Difference** ($\omega_1 - \omega_2$): This term acts as a constant vertical shift to the coupling function $g(\phi)$. If the difference is zero, synchronization can occur even with zero coupling ($a = 0$). However, as the difference $\|\omega_1 - \omega_2\|$ grows, this vertical shift becomes larger.
* **The Role of Coupling Amplitude** ($a$): This term scales the magnitude of the coupling function. Increasing $a$ makes the peaks and troughs of $a \cdot g(\phi)$ more pronounced.

For phase locking to occur, the graph of $\dot{\phi}$ must intersect the horizontal axis ($\dot{\phi} = 0$).

* If the frequency difference $\|\omega_1 - \omega_2\|$ becomes too large for a given coupling strength $a$, the entire curve of $\dot{\phi}$ may be shifted above or below the zero-axis. In this case, no fixed point exists, and the oscillators desynchronize; their phase difference will continuously drift.
* However, we can often compensate for a large difference in intrinsic frequencies by ramping up the amplitude of coupling, $a$. A larger $a$ increases the magnitude of the coupling term, allowing it to overcome the frequency difference and create intersections with the zero-axis, re-establishing a stable fixed point and thus, synchronization.

In summary, for any given coupling strength, there is a limited range of frequency differences within which oscillators can phase lock. Increasing the coupling strength broadens this range.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/phase_difference_plots.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Phase plots (left) and time graphs (right) for the coupled phase oscillators for different levels of frequency detuning as indicated. Red portions of curve on left indicate trajectory corresponding to time graphs on the right. Bottom graph: Note the slowly shifting phase difference while the trajectory passes the “attractor ghost” (or “ruin”) interrupted by fast "phase slips" (for the time graph, phase was not reset to 0 after each revolution, to better illustrate the constant drift).</figcaption>
</figure>

#### The Phenomenon of Phase Slips

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Border of Synchronization)</span></p>

An interesting special case occurs when the curve of $\dot{\phi}$ is shifted just enough to become tangent to the zero-axis, touching it at a single point. This is the critical boundary between synchronization and desynchronization.

In this state, the system exhibits phase slips.

* When the phase difference, $\phi$, approaches the region of this tangency, its derivative $\dot{\phi}$ becomes very close to zero. The system spends a long time in this "tunnel," as if it is almost synchronized.
* Eventually, it passes through this narrow channel and its derivative increases again, causing the phases to rapidly drift apart.
* This cycle repeats, leading to a behavior where the oscillators remain nearly in-phase for extended periods, punctuated by sudden "slips" where they fall out of phase before re-entering the next quasi-synchronized period.

</div>

#### Generalized P:Q Phase Locking

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($P$:$Q$ Phase Locking)</span></p>

We can generalize the concept of synchronization beyond a simple 1:1 relationship. We speak of **$P$:$Q$ phase locking** if the difference between the unbounded phase variables, $\theta_1(t)$ and $\theta_2(t)$, remains bounded when weighted by integers $p$ and $q$: 

$$\lvert p\theta_1(t) - q\theta_2(t)\rvert < \epsilon$$

Specifically, $P$:$Q$ phase locking occurs if the quantity:

$$p \cdot \theta_1(t) - q \cdot \theta_2(t)$$

remains bounded over time. This describes a state where one oscillator completes $p$ cycles in the same amount of time that the other oscillator completes $q$ cycles. The case of simple synchronization discussed previously is 1:1 phase locking.

</div>

#### Arnold Tongues: Mapping Synchronization Regions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Arnold Tongues)</span></p>

**Arnold tongues** are regions in a parameter space that depict where phase locking of a specific $P$:$Q$ ratio occurs. Typically, this space is plotted with the difference in intrinsic frequencies ($\omega_1 - \omega_2$) on one axis and the coupling amplitude ($a$) on the other. Each "tongue" represents a combination of parameters for which the system will synchronize in a particular mode (e.g., 1:1, 1:2, 2:1).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Structure of Arnold Tongues)</span></p>

The plot of Arnold tongues provides a powerful map of a coupled oscillator system's behavior.

* **The 1:1 Tongue:** The most prominent region is typically the 1:1 coupling (or exact synchrony) tongue. It is centered at a frequency difference of zero ($\omega_1 = \omega_2$). At this central point, synchronization can occur with zero coupling ($a = 0$). As the coupling strength $a$ increases, the 1:1 tongue grows broader, signifying that a stronger coupling can enforce synchronization across a larger range of intrinsic frequency differences.
* **Higher-Order Tongues:** For many systems, smaller, higher-order tongues corresponding to $P$:$Q$ couplings like 1:2, 2:1, 1:3, 3:1, etc., appear as side-branches. These tongues are typically narrower than the main 1:1 tongue, indicating that these more complex locking modes occur over a smaller range of parameters. The existence and prominence of these higher-order tongues depend on the specific properties of the system and its interaction function.
* **Intersection and Complex Dynamics:** As the coupling amplitude $a$ is increased, the tongues broaden. At a certain point, these tongues may begin to intersect. The regions where Arnold tongues overlap correspond to parameter values where complex and interesting dynamics can emerge, which are a subject of more advanced study.

</div>

### Numerical Integration of Ordinary Differential Equations

#### The Need for Numerical Methods

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Numerical Methods?)</span></p>

While the theoretical analysis of dynamical systems provides deep insights into qualitative behavior, many systems of practical interest cannot be solved analytically. For these systems, we must rely on numerical methods to approximate the solution of ordinary differential equations (ODEs). This section introduces the foundational ideas behind numerical ODE solvers.

</div>

#### The Explicit Euler Method

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Explicit Euler Method)</span></p>

The **explicit (or forward) Euler method** is the simplest numerical method for solving an initial value problem of the form $\dot{x} = f(x, t)$ with $x(0) = x_0$. The method discretizes time into steps of size $\delta_t$ and approximates the solution at each step using the tangent line:

$$x(t + \delta_t) = x(t) + \int_{t}^{t + \delta_t} f(x(\tau))d\tau$$

$$x(t + \delta_t) \approx x(t) + \delta_t \cdot f(x_n, t_n)$$

Or in the index notation

$$x_{n+1} \approx x_n + \delta_t \cdot f(x_n, t_n)$$

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/exact_euler_method.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Explicit Euler Method</figcaption>
</figure>

#### The Runge-Kutta Family of Methods

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Higher-Order Methods)</span></p>

The Euler method, while conceptually simple, has limited accuracy for a given step size. The Runge-Kutta family of methods achieves higher accuracy by evaluating the derivative $f(x, t)$ at multiple intermediate points within each time step, effectively using a weighted average of these evaluations to better approximate the true trajectory. The most commonly used variant is the fourth-order Runge-Kutta method (RK4).

</div>

#### Practical Considerations: Explicit vs. Implicit Solvers for Stiff Systems

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stiff Systems)</span></p>

A key practical consideration in numerical integration is the concept of stiffness. A stiff system is one that contains dynamics operating on vastly different time scales. For example, a system might have one variable that changes very rapidly and another that changes very slowly.

* **Explicit solvers** (like the Euler method or standard RK4) require very small time steps to maintain numerical stability when applied to stiff systems. The step size is dictated by the fastest time scale in the system, making the computation extremely expensive if one also needs to simulate the slow dynamics over long times.
* **Implicit solvers** are designed to handle stiff systems more efficiently. Instead of extrapolating forward from the current state, they define the next state implicitly, often requiring the solution of an algebraic equation at each step. While each step is more computationally expensive, implicit solvers can take much larger time steps without becoming unstable, making them far more efficient overall for stiff problems.

The choice between explicit and implicit solvers is a critical practical decision in computational dynamical systems and depends heavily on the nature of the system being simulated.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/rosenbrock_fw_euler_solvers.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Neutrally stable harmonic oscillations in linear ODE system. The graph illustrates that the choice of numerical ODE solver is important when integrating ODEs numerically: While the exact analytical solution (blue) and that of an implicit second-order numerical solver (Rosenbrock-2, green) tightly agree (in fact, a blue curve is not visible since the green curve falls on top of it), a simple forward Euler scheme (red) diverges from the true solution. </figcaption>
</figure>

