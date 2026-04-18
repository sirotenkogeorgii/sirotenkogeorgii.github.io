## Lecture 3

### Introduction to Nonlinear Systems

#### Flow on a Line: A First Look at Nonlinear Dynamics

To introduce the core properties of nonlinear systems, we will begin with the simplest case: a one-dimensional system, also known as a flow on a line ($\mathbb{R}$).


<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(A Cubic Vector Field)</span></p>

Let's consider a one-dimensional nonlinear dynamical system whose vector field is defined by a third-order polynomial:

$$\dot{x} = f(x) = x - x^3$$

Here, $x$ is a state on the real number line, and its rate of change, $\dot{x}$, is given by the nonlinear function $f(x)$. The specific form of the function is less important than the general properties it illustrates.

</div>

#### Graphical Analysis of Equilibria

A powerful technique for understanding 1D systems is to create a state space portrait by plotting the vector field $f(x)$ (i.e., the derivative $\dot{x}$) as a function of the state $x$.

For our example, $f(x) = x - x^3$, the plot is a cubic function that looks something like this:

(Conceptual sketch: A curve starting from the top-left, crossing the $x$-axis at $-1$, peaking, crossing the $x$-axis at $0$, reaching a trough, and crossing the $x$-axis at $+1$ before heading to the bottom-right.)

From this simple graph, we can extract a wealth of information about the system's dynamics.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Finding and Classifying Equilibria Graphically)</span></p>

On the plot of $f(x)$ versus $x$, the equilibria are simply the points where the graph crosses the horizontal axis (i.e., where $f(x)=0$).

* **Direction of Flow:** The sign of $f(x)$ tells us the direction of movement.
  * If $f(x) > 0$, then $\dot{x} > 0$, and the state x moves to the right (increases).
  * If $f(x) < 0$, then $\dot{x} < 0$, and the state x moves to the left (decreases).
* **Stability:** By observing the flow around each fixed point, we can determine its stability.
  * A fixed point is stable if nearby trajectories move toward it. In the graphical analysis, this corresponds to a point where the function $f(x)$ crosses the axis with a negative slope.
  * A fixed point is unstable if nearby trajectories move away from it. This corresponds to a point where $f(x)$ crosses the axis with a positive slope.

Applying this to our example $f(x) = x - x^3$, we find three equilibria. The two outer points are stable, as the flow converges toward them from both sides. The central point is unstable, as the flow moves away from it.

</div>

#### Bistability and Basins of Attraction

The presence of multiple isolated, stable fixed points is a hallmark of nonlinear systems that is impossible in linear ones.

* A linear system can have either a single isolated fixed point or a continuous line or manifold of neutrally stable points. It cannot have two or more isolated stable equilibria.
* This property of having two stable states is called bistability. With more than two, it is called multistability. This behavior is fundamentally important in fields like recurrent neural networks and machine learning.

Each stable equilibrium, or attractor, governs a region of the state space.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name">(Basin of Attraction)</span></p>

The **basin of attraction** for an attractor (such as a stable fixed point) is the set of all initial conditions in the state space from which trajectories converge to that attractor as $t \to \infty$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Attractor, Basin of Attraction)</span></p>

For a dynamical $(\Phi_t)_{t\in I}$ on the metric space $(E,\text{dist})$ an **attractor** $A\subset B \subset E$ is a closed subset such that
* $\Phi_t(A) \subset A, \forall t\in I$,
* there exists a **basin of attraction** $B$ of $A,$ which is defined as the open set

$$B = \lbrace x\in E \mid\lim_{t\to\infty} d(\Phi_t(x),A) = 0 \rbrace$$

where

$$d(x,A) = \min_{a\in A} \text{dist}(x,a)$$

$A$ is minimal, i.e. there exists no subset of $A$ satisfying the first two properties.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Basins in the 1D Example)</span></p>

In our example, the unstable fixed point at the center acts as a boundary separating the domains of the two stable fixed points. Let's say the unstable fixed point is located at a value we call $-\alpha$.

* The basin of attraction for the stable fixed point on the right is the interval $(-\alpha, +\infty)$. Any initial condition within this interval will eventually converge to this rightmost fixed point.
* Similarly, the stable fixed point on the left has its own basin of attraction, bounded on the right by $-\alpha$.

The stable fixed points themselves are referred to as point attractors.

</div>

#### Formal Stability Analysis: The Power of Linearization

While graphical analysis is insightful for 1D systems, we need a more general, analytical method to determine the stability of fixed points, especially in higher dimensions. The key technique is linearization, which involves analyzing the system's behavior in the immediate vicinity of a fixed point.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Linearization near a fixed point)</span></p>

Consider the autonomous system $\dot{x}=f(x)$, and let $x^\ast$ be a fixed point, so that $f(x^\ast)=0$. If $f$ is differentiable near $x^\ast$, then for a small perturbation $\epsilon$ defined by

$$x=x^*+\epsilon$$

the perturbation evolves according to

$$\dot{\epsilon} \approx J(x^*)\epsilon$$

where $J(x^\ast)$ is the Jacobian of $f$ evaluated at $x^\ast$. Thus, the local stability of the nonlinear system near $x^\ast$ is determined by the stability of this linear system.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let

$$x=x^*+\epsilon,$$

where $\epsilon$ is a small perturbation from the fixed point $x^\ast$. Differentiating with respect to time gives

$$\dot{x}=\frac{d}{dt}(x^*+\epsilon)$$

Since $x^\ast$ is a fixed point, it is constant in time, so $\dot{x}^\ast=0$. Hence,

$$\dot{x}=\dot{\epsilon}$$

On the other hand, from the original system,

$$\dot{x}=f(x)=f(x^*+\epsilon).$$

Therefore,

$$\dot{\epsilon}=f(x^*+\epsilon).$$

Now expand $f$ in a Taylor series about $x^\ast$. In one dimension,

$$f(x^\ast+\epsilon)\approx f(x^\ast)+\left.\frac{df}{dx}\right|_{x=x^\ast}\epsilon+O(\epsilon^2).$$

In higher dimensions, this becomes

$$f(x^*+\epsilon)\approx f(x^*)+J(x^*)\epsilon+O(\|\epsilon\|^2).$$

Because $x^\ast$ is a fixed point, we have $f(x^\ast)=0$. Substituting this into the expansion yields

$$\dot{\epsilon}\approx J(x^*)\epsilon.$$

Thus, near the fixed point, the perturbation is governed by the linear system

$$\dot{\epsilon}=J(x^*)\epsilon.$$

This shows that the local behavior, and therefore the local stability, of the nonlinear system near $x^\ast$ is determined by the linearization at $x^\ast$. $\square$

</details>
</div>

This is a linear system of differential equations describing the evolution of the perturbation. We have thus linearized the dynamics around the fixed point. The stability of the original nonlinear system, in the local vicinity of $x^{\ast}$, is determined by the stability of this linear system.

#### Classification of Equilibria in Nonlinear Systems

By analyzing the eigenvalues of the Jacobian matrix $J$ evaluated at the equilibrium point $\mathbf{x}_0$, we can classify its stability, just as we did for fully linear systems.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stability of an Equilibrium Point)</span></p>

Let $\mathbf{x}_0$ be an equilibrium point of the system $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$, and let $J = J(\mathbf{x}_0)$ be the Jacobian evaluated at that point. Let $\lbrace\lambda_i\rbrace$ be the set of eigenvalues of $J$.

* The equilibrium $\mathbf{x}_0$ is **stable** if all eigenvalues of $J$ have a negative real part.
  * $Re(\lambda_i) < 0$ for all $i$.
* The equilibrium $\mathbf{x}_0$ is **unstable** if there is at least one eigenvalue of $J$ with a positive real part.
  * There exists at least one $i$ such that $Re(\lambda_i) > 0$.
* The equilibrium $\mathbf{x}_0$ is a **saddle point** if there is at least one eigenvalue with a negative real part and at least one eigenvalue with a positive real part.
  * There exist $i, j$ such that $Re(\lambda_i) < 0$ and $Re(\lambda_j) > 0$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/Fixed_Points.gif' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
  <figcaption>Schematic visualization of 4 of the most common kinds of fixed points.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Clarifications and Edge Cases)</span></p>

* **Saddle vs. Unstable:** It is a very good point that the definition of an unstable point includes saddle points. Some authors may use "unstable" more strictly to refer to a point where all trajectories move away (i.e., all eigenvalues have positive real parts), but the broader definition provided above is common. A saddle is unstable because trajectories will diverge along at least one direction.
* **The Non-Hyperbolic Case:** What happens if one or more eigenvalues have a real part that is exactly zero? In this case, the linearization fails. The terms we ignored in the Taylor expansion ($O(\epsilon^2)$ and higher) become critical in determining the stability. Our linear approximation is not good enough to make a conclusion, and a more advanced analysis involving the higher-order terms is required.

</div>

#### The Nature of Nonlinear Systems

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

A fundamental challenge and a source of rich complexity in the study of nonlinear systems is that we can no longer expect to find explicit, closed-form solutions for the state of the system as a function of time, i.e., an equation for $x(t)$. Unlike linear systems, where such solutions are often readily available, nonlinear systems require a different, more qualitative approach. Our goal will be to understand the overall behavior and geometry of the system's evolution in its state space without necessarily solving the equations of motion explicitly.

It is remarkable how frequently similar mathematical structures appear across disparate scientific fields. Simple polynomial systems with product terms, which we will study here, are not just academic curiosities. They are foundational in describing:

* **Atmospheric Convection:** Simple climate models often reduce to this form.
* **Chemical Reaction Systems:** The rate of reaction between two species, $X$ and $Y$, is often proportional to the likelihood of them encountering each other, leading to product terms like $XY$.
* **Epidemiology:** Models for infectious diseases, such as the Susceptible-Infected-Recovered (SIR) model for epidemics like COVID, use these very types of equations to describe the dynamics of a population.

Recognizing these patterns allows us to transfer insights from one field to another, which is a powerful aspect of dynamical systems theory.

</div>

#### System Archetype: The Predator-Prey Model

To ground our discussion, we will use a classic model from population dynamics that describes the interaction between a population of predators (e.g., foxes) and a population of prey (e.g., rabbits).

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Lotka-Volterra Equations)</span></p>

Let $x$ represent the population of prey (rabbits) and $y$ represent the population of predators (foxes). The dynamics of their interaction can be modeled by the following system of differential equations:

$$
\begin{align*}
\frac{dx}{dt} &= \alpha x - \beta xy \\
\frac{dy}{dt} &= \gamma xy - \lambda y
\end{align*}
$$

Intuitive Breakdown of the Terms:

* $\boldsymbol{\alpha x}$: In the absence of predators, the prey population grows exponentially at a rate $\alpha$.
* $\boldsymbol{-\beta xy}$: The prey population decreases due to encounters with predators. This decrease is proportional to the size of both populations, as more of either species increases the frequency of encounters.
* $\boldsymbol{\gamma xy}$: The predator population grows when there is prey to eat. This growth is also proportional to the rate of encounters.
* $\boldsymbol{-\lambda y}$: In the absence of prey, the predator population decays exponentially at a rate $\lambda$ due to starvation.

</div>

<div class="pmf-grid">
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/LotkaVolterraPhasePortrait1.png' | relative_url }}" alt="Normal PDF" loading="lazy">
    <figcaption>$a=0.8, c=0.8$$</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/LotkaVolterraPhasePortrait2.png' | relative_url }}" alt="Normal PDF" loading="lazy">
    <figcaption>$a=0.8, c=0.2$</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/LotkaVolterraPhasePortrait3.png' | relative_url }}" alt="Normal PDF" loading="lazy">
    <figcaption>$a=0.2, c=0.8$</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/LotkaVolterraPhasePortrait4.png' | relative_url }}" alt="Normal PDF" loading="lazy">
    <figcaption>$a=0.2, c=0.2$</figcaption>
  </figure>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Systems as Feedback Loops)</span></p>

It is often conceptually useful to visualize such systems as a network of interacting feedback loops. This perspective was central to the development of Cybernetics in the 1940s and 50s.

For our predator-prey model:

* The prey population $x$ promotes its own growth (a positive feedback, via $\alpha$).
* The prey population $x$ promotes the growth of the predator population $y$ (via $\gamma$).
* The predator population $y$ inhibits the growth of the prey population $x$ (a negative feedback, via $\beta$).
* The predator population $y$ promotes its own decline in the absence of food (a negative feedback, via $\lambda$).

While Cybernetics as a distinct field is no longer prominent, the mathematical tools and the conceptual framework of analyzing systems through their feedback structures remain invaluable.

</div>

#### Finding Equilibria: The System's Stationary States

The first step in analyzing any dynamical system is to find its equilibrium points (also known as fixed points). These are the points in the state space where the system is stationary—that is, where all rates of change are zero.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Equilibrium Point vs. Fixed Point)</span></p>

In dynamical systems, they are often the same idea, but used in slightly different settings.

A **fixed point** usually means a point that is unchanged by the system’s update rule.

For a discrete-time system

$$x_{n+1} = f(x_n),$$

a fixed point $x^\ast$ satisfies

$$f(x^*) = x^*$$

An **equilibrium** usually means a state where nothing changes in time.

For a continuous-time system

$$\dot{x} = g(x),$$

an equilibrium $x^\ast$ satisfies

$$g(x^*) = 0$$

So the difference is mostly about language and context:

* **fixed point** is the standard term for maps and discrete-time systems
* **equilibrium** is the standard term for ODEs and continuous-time systems

They both describe a state that stays where it is once the system reaches it.

There is also a nice connection between them. If $x^\ast$ is an equilibrium of a flow, then starting at $x^\ast$ gives a constant trajectory, so it is also a fixed point of the time-$t$ flow map for every $t$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Equilibrium Point)</span></p>

An **equilibrium point** $x_0$ of a dynamical system $\dot{x} = f(x)$ is a point where the vector field is zero.  

$$f(x_0) = 0$$  

For our 2D system, this corresponds to the condition:  

$$\frac{dx}{dt} = 0 \quad \text{and} \quad \frac{dy}{dt} = 0$$ 

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Calculating Equilibria for the Predator-Prey Model)</span></p>

We set the equations of motion to zero:

$$\begin{align*} x(\alpha - \beta y) &= 0 \\ y(\gamma x - \lambda) &= 0 \end{align*}$$  

From this system, we can identify two distinct solutions.

1. **The Trivial Equilibrium:** If we set $x=0$ and $y=0$, both equations are satisfied.
   
   $$(x^\ast, y^\ast) = (0, 0)$$
   
   This corresponds to the extinction of both species.
2. **The Coexistence Equilibrium:** Assuming $x \neq 0$ and $y \neq 0$, we can divide by them to solve the parenthetical terms:
   
  $$\begin{align*} \alpha - \beta y &= 0 \implies y = \frac{\alpha}{\beta} \\ \gamma x - \lambda &= 0 \implies x = \frac{\lambda}{\gamma} \end{align*}$$  
   
  This gives us the second equilibrium point:
  
  $$(x^\ast, y^\ast) = \left(\frac{\lambda}{\gamma}, \frac{\alpha}{\beta}\right)$$  
  
  This corresponds to a state where both predator and prey populations coexist in a stable balance.

</div>

#### Stability Analysis via Linearization

Once we have found the equilibrium points, the next crucial question is: what happens to the system if it starts near one of these points? Will it return to the equilibrium, or will it be repelled? This is the question of stability. The primary tool for answering this is linearization.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The core idea is to approximate the complex nonlinear system with a simpler linear system in the immediate vicinity of an equilibrium point. The behavior of this local linear system is determined by the Jacobian matrix, and its properties (specifically, its eigenvalues) tell us almost everything we need to know about the stability of the equilibrium point for the original nonlinear system.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Jacobian for the Predator-Prey Model)</span></p>

Our system is:

$$\begin{align*} f_1(x, y) &= \alpha x - \beta xy \\ f_2(x, y) &= \gamma xy - \lambda y \end{align*} $$ 

Let's compute the partial derivatives:

* $\frac{\partial f_1}{\partial x} = \alpha - \beta y$
* $\frac{\partial f_1}{\partial y} = -\beta x$
* $\frac{\partial f_2}{\partial x} = \gamma y$
* $\frac{\partial f_2}{\partial y} = \gamma x - \lambda$

Assembling these into the Jacobian matrix gives:

$$J(x, y) = \begin{pmatrix} \alpha - \beta y & -\beta x \\ \gamma y & \gamma x - \lambda \end{pmatrix}$$

To analyze the stability of an equilibrium point $(x^{\ast}, y^{\ast})$, we evaluate this matrix at that specific point and then find its eigenvalues.

</div>

### Case Study: Characterizing the Equilibria of a Predator-Prey System

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Analysis of the Equilibrium at $(0, 0)$)</span></p>


Let's make our model concrete with a specific set of parameters and analyze its behavior.

**Parameters:**
* $\alpha = 3$
* $\beta = 1$
* $\gamma = -1$
* $\lambda = -2$

**Equilibrium Points:** Using the formulas derived earlier, the two equilibrium points are:
1. $(x^{\ast}_1, y^{\ast}_1) = (0, 0)$
2. $(x^{\ast}_2, y^{\ast}_2) = (\frac{\lambda}{\gamma}, \frac{\alpha}{\beta}) = (\frac{-2}{-1}, \frac{3}{1}) = (2, 3)$

Now, we linearize the system at each of these points.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Analysis of the Equilibrium at $(0, 0)$)</span></p>

At the point $(0,0)$ the Lotka-Volterra has eigenvalues $\lambda_1 = 3$ and $\lambda_2 = 2$.

Since both eigenvalues are real and positive, trajectories starting near the origin will be repelled from it along all directions. This type of equilibrium is called an unstable node.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

First, we evaluate the general Jacobian matrix at the point $(x, y) = (0, 0)$:  

$$J(0, 0) = \begin{pmatrix} \alpha - \beta(0) & -\beta(0) \\ \gamma(0) & \gamma(0) - \lambda \end{pmatrix} = \begin{pmatrix} \alpha & 0 \\ 0 & -\lambda \end{pmatrix}$$  

Plugging in our specific parameter values $(\alpha=3, \lambda=-2)$:  

$$J(0, 0) = \begin{pmatrix} 3 & 0 \\ 0 & -(-2) \end{pmatrix} = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}$$  

The eigenvalues of a diagonal matrix are simply its diagonal entries. Therefore, the eigenvalues are $\lambda_1 = 3$ and $\lambda_2 = 2$.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Analysis of the Equilibrium at $(2, 3)$)</span></p>

At the point $(2,3)$ the Lotka-Volterra has eigenvalues $\lambda_1 = +\sqrt{6} \approx 2.45$ and $\lambda_2 = -\sqrt{6} \approx -2.45$.

Since we have one positive real eigenvalue and one negative real eigenvalue, trajectories are attracted towards the equilibrium along one direction (the stable direction) and repelled from it along another direction (the unstable direction). This quintessential feature defines a saddle point.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Next, we evaluate the Jacobian at the coexistence equilibrium $(x, y) = (2, 3)$ using our parameters $(\alpha=3, \beta=1, \gamma=-1, \lambda=-2)$:  

$$J(2, 3) = \begin{pmatrix} \alpha - \beta(3) & -\beta(2) \\ \gamma(3) & \gamma(2) - \lambda \end{pmatrix} = \begin{pmatrix} 3 - 1(3) & -1(2) \\ -1(3) & -1(2) - (-2) \end{pmatrix}$$   

$$J(2, 3) = \begin{pmatrix} 0 & -2 \\ -3 & 0 \end{pmatrix}$$  

To find the eigenvalues, we solve the characteristic equation $\text{det}(J - \lambda I) = 0$:  

$$\text{det} \begin{pmatrix} -\lambda & -2 \\ -3 & -\lambda \end{pmatrix} = (-\lambda)(-\lambda) - (-2)(-3) = \lambda^2 - 6 = 0$$  

This gives $\lambda^2 = 6$, so the eigenvalues are $\lambda_1 = +\sqrt{6} \approx 2.45$ and $\lambda_2 = -\sqrt{6} \approx -2.45$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Manifolds and Heteroclinic Orbits)</span></p>

The qualitative behavior of this system is quite structured.

* The set of points that converge to the saddle point at $(2,3)$ is called its stable manifold. In this case, it appears as a curve passing through the saddle.
* The set of points that are repelled from the saddle point is its unstable manifold, another curve passing through the saddle.
* An especially interesting feature in some systems is when an orbit starts at one equilibrium point and ends at another. In our example, the unstable manifold of the unstable node at $(0,0)$ connects to the stable manifold of the saddle at $(2,3)$. An orbit that connects two different fixed points is called a heteroclinic orbit. Such structures are not mere curiosities; they play important roles in phenomena like computation in neural systems.

(We use the term manifold loosely for now; it has a precise mathematical definition that we will explore later.)

</div>

#### Formal Definitions of Stability

Our analysis using eigenvalues is powerful, but it relies on linearization. To build a more robust foundation, we need formal definitions that do not depend on this approximation. These definitions, often associated with mathematicians like Perko, are based on the behavior of trajectories in neighborhoods around the equilibrium point.

Let $x_0$ be an equilibrium point and let $\phi_t(x)$ be the flow operator, which maps a starting point $x$ to its position at time $t$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stable Equilibrium (in the sense of Lyapunov))</span></p>

An **equilibrium point $x_0$ is stable** if for every neighborhood $\mathcal{U}$ of $x_0$ (e.g., an open ball of radius $\epsilon > 0$), there exists a smaller neighborhood $\mathcal{V}$ of $x_0$ (e.g., a ball of radius $\delta > 0$) such that any trajectory starting in $\mathcal{V}$ remains within $\mathcal{U}$ for all future time.

Formally: For every $\epsilon > 0$, there exists a $\delta > 0$ such that for any point $x$ with $\lvert x - x_0\rvert < \delta$, we have $\lvert\phi_t(x) - x_0\rvert < \epsilon$ for all $t \ge 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name">("If you start close enough, you stay close enough.")</span></p>

Note that this does not require the trajectory to approach $x_0$. A center in a frictionless pendulum system is stable, as trajectories that start nearby will orbit it, staying close but never converging.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Asymptotically Stable Equilibrium)</span></p>


An **equilibrium point $x_0$ is asymptotically stable** if:

1. It is stable.
2. There exists a neighborhood $\mathcal{W}$ of $x_0$ such that every trajectory starting in $\mathcal{W}$ converges to $x_0$ as time goes to infinity.

Formally: It is stable, and there exists $\eta > 0$ such that if $\lvert x - x_0\rvert < \eta$, then

$$\lim_{t \to \infty} \phi_t(x) = x_0$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name">("If you start close enough, you eventually arrive at the equilibrium.")</span></p>

Sinks and stable spirals are asymptotically stable. This more general framework is crucial as it correctly classifies cases where linearization is inconclusive, such as when the eigenvalues of the Jacobian have zero real parts.

</div>

### Foundational Concepts in Topological Dynamics

#### Asymptotic Stability: A General Definition

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Previously, definitions of stability may have been restricted to hyperbolic dynamical systems—systems where the real parts of the eigenvalues of the linearized system at an equilibrium point are all non-zero. That definition, while useful, is not universally applicable. We introduce a more general definition of asymptotic stability that also covers non-hyperbolic systems, providing a more robust tool for analysis.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Asymptotic Stability)</span></p>

An equilibrium point $x_0$ is **asymptotically stable** if there exists a neighborhood around it such that any trajectory starting within that neighborhood converges to $x_0$.

Formally, there exists a $ \delta > 0$ such that for any initial condition $x$ within the open neighborhood $N_\delta(x_0)$, the trajectory starting at $x$ converges to $x_0$:  

$$\forall x \in N_\delta(x_0), \quad \lim_{t \to \infty} \phi_t(x) = x_0$$  

where $\phi_t(x)$ is the flow of the dynamical system.

</div>

#### Homeomorphisms and Topological Equivalence

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

A central task in the study of dynamical systems is the ability to compare them. We need a "recipe" to determine if two different systems are, in some essential way, the same. This is not just an academic exercise; it is crucial for modern applications. For instance, when machine learning algorithms attempt to infer the governing equations of a system from data, we want the reconstructed system to preserve the fundamental properties of the original. The concept of topological equivalence, which is built upon the idea of a homeomorphism, provides the mathematical language for this.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Homeomorphism)</span></p>

Let $X$ be a metric space (a space endowed with a distance function), and let $A$ and $B$ be subsets of $X$.

A **homeomorphism** is a function $h: A \to B$ that satisfies the following three properties:

1. **Continuity:** The function $h$ is continuous.
2. **One-to-One and Onto (Bijective):** The function $h$ is a one-to-one map (injective) and maps onto the entire set $B$.
3. **Continuous Inverse:** The inverse function $h^{-1}: B \to A$ exists and is also continuous.

In essence, a homeomorphism is a continuous stretching and bending of space that defines a unique, reversible mapping between two sets, $A$ and $B$.

</div>

#### An Introduction to Manifolds

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The concept of a manifold formalizes the idea of a space that, on a local level, resembles standard Euclidean space. For any point within a manifold, we can find a small neighborhood around it that can be smoothly mapped to an open ball in $\mathbb{R}^n$. This ensures that every point is "surrounded" by other points in the set, a property that is crucial for defining smooth dynamics on these structures.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Manifold)</span></p>

An **$n$-dimensional topological manifold** is a topological space $X$ such that for each point $x_0 \in X$, there exists an open neighborhood of $x_0$ that is homeomorphic to an $n$-dimensional open Euclidean ball.

More formally, for each $x_0 \in X$, there is:

1. An open neighborhood $N_\epsilon(x_0)$ of $x_0$.
2. A homeomorphism $h: N_\epsilon(x_0) \to B^n$, where $B^n$ is the $n$-dimensional open unit ball.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Open Unit Ball)</span></p>

The **$n$-dimensional open unit ball** is defined as the set of all points in an $n$-dimensional space whose distance from the origin is strictly less than one:  

$$B^n = \lbrace (x_1, \dots, x_n) \in \mathbb{R}^n \mid \sum_{i=1}^n x_i^2 < 1 \rbrace$$  

The "strictly less than" condition is what makes the set open, meaning for any point in the ball, we can find a small neighborhood around it that is still entirely contained within the ball.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name"></span></p>

* **A manifold:** A closed orbit (like a circle) is a simple example of a 1-dimensional manifold. For any point on the circle, you can find a small arc around it that is homeomorphic to an open interval in $\mathbb{R}^1$.
* **Not a manifold:** A line segment is not a manifold. Consider a point at one of the endpoints. No matter how small a neighborhood (an open ball) you draw around this endpoint, it will never be completely contained within the line segment. Therefore, the endpoint does not have a neighborhood homeomorphic to an open Euclidean ball.

</div>

### Stable and Unstable Manifolds

Stable and unstable manifolds are fundamental geometric structures in dynamical systems. They are the sets of all points that approach (or move away from) an equilibrium point, such as a fixed point or a periodic orbit. They play a huge role in characterizing the long-term behavior of a system.

#### The Local Perspective: Neighborhoods of an Equilibrium

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The construction of stable and unstable manifolds is a two-step process. First, we define them locally in a small neighborhood around an equilibrium point $x_0$. In this local region, the manifold's structure is determined by the linear dynamics—specifically, the stable and unstable directions given by the eigenvectors of the system's Jacobian at $x_0$. After defining this local structure, we can extend it to find the full, global manifold.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Local Stable Manifold)</span></p>

The **local stable manifold** of an equilibrium point $x_0$, denoted $W^s_{loc}(x_0)$, is the set of all points $x$ in a small neighborhood of $x_0$ that not only converge to $x_0$ as time goes to infinity but also remain within that neighborhood for all future time.

Formally, for a given neighborhood $N_\epsilon(x_0)$:  

$$W^s_{loc}(x_0) = \lbrace x \in N_\epsilon(x_0) \mid \lim_{t \to \infty} \phi_t(x) = x_0 \quad \text{and} \quad \phi_t(x) \in N_\epsilon(x_0) \text{ for all } t \ge 0 \rbrace$$ 

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Local Unstable Manifold)</span></p>

The **local unstable manifold** of an equilibrium point $x_0$, denoted $W^u_{loc}(x_0)$, is defined analogously but using reverse time. It is the set of all points $x$ in a small neighborhood of $x_0$ that converge to $x_0$ as time goes to negative infinity and remain within that neighborhood for all past time.

Formally, for a given neighborhood $N_\epsilon(x_0)$:  

$$W^u_{loc}(x_0) = \lbrace x \in N_\epsilon(x_0) \mid \lim_{t \to -\infty} \phi_t(x) = x_0 \quad \text{and} \quad \phi_t(x) \in N_\epsilon(x_0) \text{ for all } t \le 0 \rbrace$$ 

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The condition that trajectories must stay within the neighborhood is critical. Consider a saddle point. There are many points in an initial neighborhood that will eventually be repelled and leave the neighborhood. These points are not part of the local stable manifold. The definition precisely isolates only those initial conditions that are perfectly aligned with the stable direction and will thus converge to the equilibrium point without first being pushed away.

</div>

#### The Global Perspective: Extending Local Manifolds

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Once the local stable and unstable manifolds are defined, the global manifolds are constructed by integrating the points on these local sets forward or backward in time. This process "collects" all the points in the entire phase space that are connected to the equilibrium point via its stable or unstable dynamics.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Global Stable Manifold)</span></p>

The **global stable manifold** of an equilibrium point $x_0$, denoted $W^s(x_0)$, is the union of all backward-time images of the local stable manifold. Intuitively, we take the set of points that are already known to be approaching $x_0$ (the local manifold) and trace their paths backward in time to find every point in the phase space that will eventually enter this local neighborhood and converge to $x_0$.

Formally, it is the union of the flow operator applied to the local manifold for all non-positive time:

$$W^s(x_0) = \bigcup_{t \le 0} \phi_t(W^s_{loc}(x_0))$$ 

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Global Unstable Manifold)</span></p>

The **global unstable manifold** of an equilibrium point $x_0$, denoted $W^u(x_0)$, is the union of all forward-time images of the local unstable manifold. Intuitively, we take the points that are locally diverging from $x_0$ (which converge to $x_0$ in reverse time) and follow their trajectories forward to trace out the entire structure of repulsion from the equilibrium.

Formally, it is the union of the flow operator applied to the local manifold for all non-negative time:

$$W^u(x_0) = \bigcup_{t \ge 0} \phi_t(W^u_{loc}(x_0))$$ 

</div>

### Foundational Concepts in Phase Space

This chapter introduces three fundamental concepts that form the bedrock for analyzing the qualitative behavior of dynamical systems. We will explore special types of orbits that structure the phase space, the notion of sets that are preserved by the system's evolution, and a rigorous way to determine when two different systems can be considered "the same" from a topological standpoint.

#### Special Orbits: Homoclinic and Heteroclinic

Certain orbits, or solution curves, play a crucial role in defining the global structure of a dynamical system's phase space. A solution curve is the path traced by a point $x_0$ under the action of the flow operator $\phi_t$ for all time $t$, both forward and backward. Among the most important of these are homoclinic and heteroclinic orbits, which connect equilibrium points.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Homoclinic Orbit)</span></p>

A **homoclinic orbit** $\Gamma$ is a solution curve that connects an equilibrium point to itself. It originates from the equilibrium point (as $t \to -\infty$) and returns to the same equilibrium point (as $t \to +\infty$).

Formally, for an equilibrium point $x_0$, a homoclinic orbit is a trajectory $\Gamma$ such that:

$$\Gamma \subset W^s(x_0) \cap W^u(x_0)$$  

This means the orbit is simultaneously part of the stable manifold and the unstable manifold of the same point. The trajectory diverges from $x_0$ along its unstable manifold and converges back to $x_0$ along its stable manifold.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Heteroclinic Orbit)</span></p>

A **heteroclinic orbit** $\Gamma$ is a solution curve that connects two different equilibrium points. It originates from one equilibrium point (as $t \to -\infty$) and terminates at another (as $t \to +\infty$).

Formally, for two distinct equilibrium points $x_0 \neq x_1$, a heteroclinic orbit is a trajectory $\Gamma$ connecting $x_0$ to $x_1$ such that:

$$\Gamma \subset W^u(x_0) \cap W^s(x_1)$$  

This means the orbit is part of the unstable manifold of $x_0$ and the stable manifold of $x_1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Homoclinic and heteroclinic orbits are of fundamental importance in the study of dynamical systems. Their presence can have profound implications for the system's behavior.

* **Structural Skeletons:** These orbits act as organizing centers or "skeletons" for the dynamics in phase space.
* **Indicators of Chaos:** The existence of a homoclinic orbit, in particular, is often a strong indicator—almost a guarantee—that the system exhibits chaotic behavior. We will explore this connection in greater detail later.
* **Separatrixs:** Structures built from sequences of heteroclinic (and sometimes homoclinic) orbits are called separatrix cycles. These cycles can create boundaries in the phase space that separate regions of qualitatively different behavior.

</div>


#### Invariant Sets

The concept of invariance is central to identifying regions in the phase space that are self-contained under the system's dynamics.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Invariant Set)</span></p>

Let $E \subset \mathbb{R}^n$ be an open set representing the state space, and let $\phi_t: \mathbb{R} \times E \to E$ be a flow operator defined for all times $t \in \mathbb{R}$.

A set $S \subset E$ is called an **invariant set** with respect to the flow $\phi_t$ if any trajectory starting in $S$ remains in $S$ for all time, both forward and backward.

More formally, $S$ is invariant if for any point $x \in S$, its entire orbit remains within $S$. This is expressed as:

$$\phi_t(S) = S \quad \text{for all } t \in \mathbb{R}$$  

where $\phi_t(S) = \lbrace \phi_t(x) \mid x \in S \rbrace$.

</div>

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/invaraint_set_2D.png' | relative_url }}" alt="a" loading="lazy">
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/invaraint_set_1D.svg' | relative_url }}" alt="a" loading="lazy">
  </figure>
</div>


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

* **Forward and Backward Invariance:** One can also define one-sided invariance. A set $S$ is forward invariant if $\phi_t(S) \subset S$ for all $t \ge 0$. This means that once you are in the set, you can never leave it as time moves forward. A set is backward invariant if $\phi_t(S) \subset S$ for all $t \le 0$.
* **Discrete Time Systems (Maps):** The concept applies equally to discrete maps. For a map $f: E \to E$, a set $S$ is invariant if applying the map (or its inverse) any number of times to a point in $S$ yields a point that is still in $S$. That is, $f(S) = S$.

The simplest examples of invariant sets are equilibrium points and limit cycles. Understanding which sets are invariant helps decompose the phase space into dynamically independent regions.

</div>

#### Topological Equivalence of Dynamical Systems

A powerful idea in the study of dynamical systems is to determine when two systems, which may have different equations, are qualitatively "the same." The concept of topological equivalence provides a rigorous framework for this comparison.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Topological Equivalence)</span></p>

Consider two dynamical systems defined by continuously differentiable vector fields:

1. $\dot{x} = f(x)$, where $x \in A$ and $A \subset \mathbb{R}^n$ is an open set.
2. $\dot{y} = g(y)$, where $y \in B$ and $B \subset \mathbb{R}^m$ is an open set.

Let $\phi^A_t$ be the flow of the first system and $\phi^B_t$ be the flow of the second.

These two systems are **topologically equivalent** if there exists a homeomorphism $h: A \to B$ what preserves the direction of flow (any time I guess).

* A homeomorphism is a continuous function with a continuous inverse, meaning it provides a smooth, one-to-one mapping between the state spaces $A$ and $B$.

The condition of "preserving the direction of time" means that forward-time evolution in one system corresponds to forward-time evolution in the other. Formally, for every $x_0 \in A$ and any time $t$, the following relationship must hold:

$$h(\phi^A_t(x_0)) = \phi^B_{\tau(x_0, t)}(h(x_0))$$ 

Here, $\tau(x_0, t)$ is a reparameterization of time. To preserve the direction of flow, for any fixed $x_0$, $\tau$ must be a strictly increasing function of $t$. This is guaranteed if its derivative with respect to $t$ is always positive: 

$$\frac{\partial \tau}{\partial t} > 0$$ 

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Topological equivalence means we can stretch, bend, or compress one state space to make it look like the other, such that the orbit structures align perfectly. The time it takes to travel along corresponding segments of orbits might differ (hence the reparameterization $\tau$), but the direction of travel is the same.

If two systems are topologically equivalent, then properties like the number and type of fixed points, the existence of closed orbits, and the stability of these objects are preserved. For instance, if one system has an orbit that converges to a fixed point, the corresponding orbit in the equivalent system will also converge to the corresponding fixed point. The diagram below illustrates the commutative relationship:

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name"></span></p>

Let's demonstrate topological equivalence with a simple example.

Consider the following two one-dimensional systems:

1. System $A$: $\dot{x} = -x$
2. System $B$: $\dot{y} = y$

The flow for System $A$ is $\phi^A_t(x) = e^{-t}x$. Trajectories in this system decay exponentially toward the equilibrium at $x=0$. The flow for System $B$ is $\phi^B_t(y) = e^t y$. Trajectories in this system grow exponentially away from the equilibrium at $y=0$.

Let's propose a candidate homeomorphism $h(x) = 1/x$. This map is defined for $x \neq 0$ and its inverse is $h^{-1}(y) = 1/y$, which is also continuous.

We now check if the condition for topological equivalence, $h(\phi^A_t(x_0)) = \phi^B_{\tau}(h(x_0))$, can be satisfied for a valid time reparameterization $\tau$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution</span><span class="math-callout__name"></span></p>

1. **Calculate the left-hand side (LHS):** First, we apply the flow of System $A$ to a point $x_0$, and then map the result using $h$.
  
   $$h(\phi^A_t(x_0)) = h(e^{-t}x_0) = \frac{1}{e^{-t}x_0} = \frac{e^t}{x_0}$$ 

2. **Calculate the right-hand side (RHS):** First, we map the point $x_0$ using $h$, and then apply the flow of System $B$ for a reparameterized time $\tau$.
  
   $$\phi^B_{\tau}(h(x_0)) = \phi^B_{\tau}\left(\frac{1}{x_0}\right) = e^{\tau} \cdot \frac{1}{x_0} = \frac{e^{\tau}}{x_0}$$ 

3. **Equate LHS and RHS to find $\tau$:** For the systems to be equivalent, we must have LHS = RHS.
   
   $$\frac{e^t}{x_0} = \frac{e^{\tau}}{x_0} \implies e^t = e^{\tau} \implies \tau = t$$

4. **Verify the time-preservation condition:** The reparameterization is $\tau(x_0, t) = t$. We check its derivative:
  
   $$\frac{\partial \tau}{\partial t} = \frac{d}{dt}(t) = 1$$
   
   Since $\frac{\partial \tau}{\partial t} = 1 > 0$, then condition is satisfied?

---

What is wrong in the example above is the function $h$ is not a homeomorphism. The key issue is the **domain**.

The proposed map $h(x)=\frac1x$ is **not defined at $x=0$**. So it is not a homeomorphism $\mathbb{R}\to\mathbb{R}$.

That matters a lot here, because $x=0$ and $y=0$ are the equilibria, and on the full line these two systems are qualitatively different:

* $\dot x=-x$: $0$ is a **sink** (stable)
* $\dot y=y$: $0$ is a **source** (unstable)

So on $\mathbb{R}$, they are **not** topologically equivalent.

The computation $h(\phi_t^A(x))=\phi_t^B(h(x))$ is actually correct, but only on the **punctured line** $\mathbb{R}\setminus{0}$

On that space, $h(x)=1/x$ is a homeomorphism, and the systems are not just topologically equivalent but actually **topologically conjugate** with no time reparameterization needed, since $\tau=t$.

So the right verdict is:

* **On $\mathbb{R}$: incorrect**
* **On $\mathbb{R}\setminus{0}$: correct**

**A quick sanity check with equilibria immediately shows a contradiction**

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Quick Sanity Check For Homeomorphism)</span></p>

For 1D systems, a very fast check is:

* are equilibria matched?
* are stable equilibria matched with stable ones?
* are unstable equilibria matched with unstable ones?

</div>

### Topological Equivalence and Conjugacy

In the study of dynamical systems, we are often less concerned with the precise numerical solution of a trajectory and more interested in its qualitative behavior. Does it spiral into a fixed point? Does it diverge to infinity? Does it oscillate forever? To formalize these qualitative similarities, we introduce the concepts of topological equivalence and the more restrictive topological conjugacy. These powerful ideas allow us to state precisely when two different systems share the same fundamental geometric structure.

#### Defining Topological Equivalence

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Topological Equivalence)</span></p>

Two dynamical systems, defined by flows $\phi_t^A$ and $\phi_t^B$ on state spaces $A$ and $B$ respectively, are said to be **topologically equivalent** if there exists a homeomorphism $h: A \to B$ that maps trajectories of $\phi^A $ onto trajectories of $\phi^B$ while preserving the direction of time.

* A homeomorphism is a continuous function between topological spaces that has a continuous inverse function. It is a map that preserves all the topological properties of a given space.

This means that for every point $x \in A$, the curve traced by $\phi_t^A(x)$ is mapped by $h$ to the curve traced by $\phi_t^B(h(x))$. While the direction of flow along the curve is preserved, the speed is not. The time parameter may be re-scaled, meaning a point that takes 1 second to travel a path in system $A$ might take 2 seconds (or 0.5 seconds) to travel the corresponding path in system $B$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/topological_equivalence_visualization_clean.png' | relative_url }}" alt="a" loading="lazy">
</figure>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/not_topologically_equivalent.png' | relative_url }}" alt="a" loading="lazy">
</figure>

#### A Critical Counterexample: The Importance of the Homeomorphism

Let's consider two simple one-dimensional systems:

1. System $A$: $\dot{x} = x$ (an unstable fixed point at $x=0$)
2. System $B$: $\dot{y} = -y$ (a stable fixed point at $y=0$)

At first glance, one might wonder if these systems could be considered equivalent. After all, they both feature straight-line trajectories moving away from or towards the origin. Let's propose a map $h(x) = 1/x$ and investigate if it establishes an equivalence.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What Went Wrong?)</span></p>

Intuitively, these two systems have fundamentally different behaviors. One diverges from the origin, while the other converges to it. A continuous transformation should not be able to reverse the fundamental stability of a system. The direction of flow is a core topological property, and reversing it is a major violation. Let's see how this intuition manifests mathematically.

The proposed map $h(x) = 1/x$ is not a homeomorphism for this problem. A homeomorphism must be continuous everywhere on the domain of interest. Here, the domain includes the equilibrium point at $x=0$. The function $1/x$ has a discontinuity precisely at this equilibrium point.

Because the proposed mapping function $h$ is not continuous at the equilibrium point, it is not a valid homeomorphism. Therefore, it cannot be used to establish topological equivalence.

This example reveals a crucial insight:

Stable nodes, unstable nodes, and saddles are all topologically distinct. You cannot find a homeomorphism that continuously deforms the phase portrait of a stable node into that of an unstable node.

</div>

#### Conditions for Equivalence in Linearizable Systems

While a stable and an unstable node are not equivalent, other pairings are. For instance, a stable node and a stable spiral are topologically equivalent. Both systems draw all nearby trajectories into the fixed point, and one can be continuously deformed into the other. This leads to a more formal condition for equivalence in a large class of systems.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Topological Equivalence for Hyperbolic Systems)</span></p>

For linear or linearizable systems, two systems are topologically equivalent in a neighborhood of a hyperbolic fixed point if the Jacobian matrices evaluated at that fixed point have the same number of eigenvalues with positive real parts and the same number of eigenvalues with negative real parts.

* Recall that a hyperbolic fixed point is one where the Jacobian has no eigenvalues with a zero real part.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

This theorem provides a powerful and practical criterion for determining equivalence. It essentially states that the local "picture" of the dynamics is determined entirely by the number of stable (negative real part) and unstable (positive real part) directions. The exact values of the eigenvalues don't matter for equivalence, nor does the presence of imaginary parts (which create spirals), as long as the counts of stable and unstable dimensions match.

</div>

#### Topological Conjugacy: Preserving Time

Topological conjugacy is a stricter condition than equivalence. It requires not only that the trajectories map onto one another but also that the parameterization by time is preserved.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Topological Conjugacy)</span></p>

Two systems with flows $\phi_t^A$ and $\phi_t^B$ are **topologically conjugate** if they are topologically equivalent via a homeomorphism $h$, and this mapping preserves the time parameter. Formally, for any initial point $x_0$ and any time $t$:

$$h(\phi_t^A(x_0)) = \phi_t^B(h(x_0))$$

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/topological_conjugacy_visualization.png' | relative_url }}" alt="a" loading="lazy">
</figure>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/topological_equivalent_not_conjugate.png' | relative_url }}" alt="a" loading="lazy">
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Notice the crucial difference: the time variable $t$ is exactly the same on both sides of the equation. This means that if it takes the first system $T$ seconds to get from point $x_0$ to $x_1$, it must also take the second system exactly $T$ seconds to get from the mapped point $h(x_0)$ to the mapped point $h(x_1)$. The systems are not just qualitatively similar; their flows are perfectly synchronized through the lens of the homeomorphism $h$.

</div>

#### Example of Topologically Conjugate Systems

Let's demonstrate this stronger condition with an example.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name"></span></p>

Consider the following two systems:

1. System $A$: $\dot{x} = -x$
2. System $B$: $\dot{y} = -2y$

The solutions, or flow operators, for these systems are:

* $\phi_t^A(x) = x e^{-t}$
* $\phi_t^B(y) = y e^{-2t}$

We claim these two systems are topologically conjugate. To prove this, we must find a suitable homeomorphism $h$. Let's define one as follows:

$$
h(x) =
\begin{cases}
x^2 & \text{if } x \ge 0 \\
-x^2 & \text{if } x < 0
\end{cases}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

This function $h(x)$ is a valid homeomorphism. It is continuous everywhere, including at $x=0$ where both pieces of the function converge to zero. It is also continuously differentiable at the origin.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Conjugacy)</span></p>

We must now verify that our chosen $h$ satisfies the condition $h(\phi_t^A(x)) = \phi_t^B(h(x))$. Let's analyze the left and right sides of the equation separately, assuming $x \ge 0$ for simplicity.

1. **Left-Hand Side (LHS):** First, we apply the flow of system $A$, and then we map the result with $h$.
  
  $$h(\phi_t^A(x)) = h(x e^{-t})$$  
   
  Since $xe^{-t}$ will have the same sign as $x$, we use the $x^2$ part of our definition for $h$:  
  
  $$h(x e^{-t}) = (x e^{-t})^2 = x^2 e^{-2t}$$ 

2. **Right-Hand Side (RHS):** First, we map the initial point $x$ with $h$, and then we apply the flow of system B to the result.
   
  $$\phi_t^B(h(x)) = \phi_t^B(x^2)$$  
   
  Let $ y = h(x) = x^2 $. Applying the flow for system B gives:
   
  $$\phi_t^B(y) = y e^{-2t} = (x^2) e^{-2t} = x^2 e^{-2t}$$ 

Since LHS = RHS, we have shown that $h(\phi_t^A(x)) = \phi_t^B(h(x))$. The systems are indeed topologically conjugate.

</div>

<div id="tc2-container" style="margin:2em auto;max-width:920px;">
  <h4 style="text-align:center;margin:0 0 .2em;">Interactive: Topological Conjugacy of Planar Nodes</h4>
  <p style="text-align:center;color:#888;font-size:.82em;margin:0 0 .5em;">
    Two systems \(\dot x=-x,\;\dot y=-\lambda_i y\) conjugated by \(h(x,y)=(x,\,\mathrm{sgn}(y)\,|y|^{\lambda_2/\lambda_1})\). The homeomorphism \(h\) is continuous but <em>not</em> differentiable at \(y=0\) when \(\lambda_1\neq\lambda_2\).
  </p>
  <div style="display:flex;align-items:center;justify-content:center;gap:12px;margin-bottom:10px;flex-wrap:wrap;">
    <span style="font-size:.85em;font-family:serif;">&lambda;&sub1; =</span>
    <input type="range" id="tc2-s1" min="10" max="50" step="1" value="20" style="width:160px;">
    <span id="tc2-o1" style="font-size:.85em;font-family:serif;min-width:28px;">2.0</span>
    <span style="font-size:.85em;font-family:serif;margin-left:10px;">&lambda;&sub2; =</span>
    <input type="range" id="tc2-s2" min="10" max="50" step="1" value="40" style="width:160px;">
    <span id="tc2-o2" style="font-size:.85em;font-family:serif;min-width:28px;">4.0</span>
  </div>
  <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:10px;">
    <div style="text-align:center;">
      <div style="font-size:.82em;font-weight:600;margin-bottom:3px;">Phase portrait — system 1 (&lambda;&sub1;)</div>
      <canvas id="tc2-c1" width="420" height="420" style="border:1px solid #ddd;border-radius:3px;background:#fff;max-width:100%;"></canvas>
    </div>
    <div style="text-align:center;">
      <div style="font-size:.82em;font-weight:600;margin-bottom:3px;">Phase portrait — system 2 (&lambda;&sub2;)</div>
      <canvas id="tc2-c2" width="420" height="420" style="border:1px solid #ddd;border-radius:3px;background:#fff;max-width:100%;"></canvas>
    </div>
  </div>
  <div style="text-align:center;margin-top:10px;">
    <div style="font-size:.82em;font-weight:600;margin-bottom:3px;">Conjugacy map h &mdash; uniform grid (left) vs image (right)</div>
    <canvas id="tc2-c3" width="860" height="330" style="border:1px solid #ddd;border-radius:3px;background:#fff;max-width:100%;"></canvas>
  </div>
  <div style="text-align:center;margin-top:10px;">
    <div style="font-size:.82em;font-weight:600;margin-bottom:3px;">|&part;h&sub2;/&part;y| along y &mdash; derivative singularity at y = 0</div>
    <canvas id="tc2-c4" width="860" height="260" style="border:1px solid #ddd;border-radius:3px;background:#fff;max-width:100%;"></canvas>
  </div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin-top:10px;">
    <div style="background:rgba(0,0,0,.03);border-radius:6px;padding:8px 14px;min-width:100px;text-align:center;">
      <div style="font-size:11px;color:#888;">&alpha; = &lambda;&sub2; / &lambda;&sub1;</div>
      <div style="font-size:18px;font-weight:500;" id="tc2-mA">2.00</div>
    </div>
    <div style="background:rgba(0,0,0,.03);border-radius:6px;padding:8px 14px;min-width:100px;text-align:center;">
      <div style="font-size:11px;color:#888;">&part;h&sub2;/&part;y as y&rarr;0</div>
      <div style="font-size:18px;font-weight:500;" id="tc2-mS">&rarr; 0</div>
    </div>
    <div style="background:rgba(0,0,0,.03);border-radius:6px;padding:8px 14px;min-width:100px;text-align:center;">
      <div style="font-size:11px;color:#888;">H&ouml;lder exponent</div>
      <div style="font-size:18px;font-weight:500;" id="tc2-mH">0.50</div>
    </div>
    <div style="background:rgba(0,0,0,.03);border-radius:6px;padding:8px 14px;min-width:100px;text-align:center;">
      <div style="font-size:11px;color:#888;">Smoothness class</div>
      <div style="font-size:18px;font-weight:500;" id="tc2-mC">C&sup0; \ C&sup1;</div>
    </div>
  </div>
</div>

<script>
(function(){
  var sl1=document.getElementById('tc2-s1'),sl2=document.getElementById('tc2-s2');
  var p1c='#534AB7',p2c='#0F6E56',acc='#A32D2D',mu2='#999',grc='rgba(0,0,0,.05)';
  var f1c='rgba(83,74,183,.12)',f2c='rgba(15,110,86,.12)';

  function ct(id){return document.getElementById(id).getContext('2d');}
  function cv2(id){return document.getElementById(id);}

  function drawPhase(id,lam,col,fc){
    var c=cv2(id),x=ct(id),W=c.width,H=c.height;
    x.fillStyle='#fff';x.fillRect(0,0,W,H);
    var cx=W/2,cy=H/2,sc=W*.38;
    x.strokeStyle=grc;x.lineWidth=.5;
    for(var i=-1;i<=1;i+=.5){
      x.beginPath();x.moveTo(cx+i*sc,0);x.lineTo(cx+i*sc,H);x.stroke();
      x.beginPath();x.moveTo(0,cy+i*sc);x.lineTo(W,cy+i*sc);x.stroke();
    }
    x.strokeStyle=mu2;x.lineWidth=.7;
    x.beginPath();x.moveTo(0,cy);x.lineTo(W,cy);x.stroke();
    x.beginPath();x.moveTo(cx,0);x.lineTo(cx,H);x.stroke();
    for(var s=-1;s<=1;s+=2){
      for(var c2=-1.3;c2<=1.3;c2+=.1){
        if(Math.abs(c2)<.02)continue;
        x.strokeStyle=fc;x.lineWidth=.8;x.beginPath();var st=true;
        for(var xi=.002;xi<=1.5;xi+=.004){
          var px_=cx+s*xi*sc,py_=cy-c2*Math.pow(xi,lam)*sc;
          if(px_<-10||px_>W+10||py_<-10||py_>H+10){st=true;continue;}
          st?(x.moveTo(px_,py_),st=false):x.lineTo(px_,py_);
        }x.stroke();
      }
    }
    var hl=[.35,-.35,.7,-.7,1.1,-1.1];
    hl.forEach(function(c2){
      for(var s=-1;s<=1;s+=2){
        x.strokeStyle=col;x.lineWidth=2;x.beginPath();var st=true;
        for(var xi=.002;xi<=1.5;xi+=.003){
          var px_=cx+s*xi*sc,py_=cy-c2*Math.pow(xi,lam)*sc;
          if(px_<-10||px_>W+10||py_<-10||py_>H+10){st=true;continue;}
          st?(x.moveTo(px_,py_),st=false):x.lineTo(px_,py_);
        }x.stroke();
        var ax=s*.3,ay=c2*Math.pow(.3,lam);
        var ddx=-s*.01,ddy=-c2*lam*Math.pow(.3,lam-1)*.01*s;
        var ln2=Math.sqrt(ddx*ddx+ddy*ddy);if(ln2<1e-8)return;
        var ux=ddx/ln2,uy=ddy/ln2;
        var apx=cx+ax*sc,apy=cy-ay*sc;
        x.fillStyle=col;x.beginPath();
        x.moveTo(apx+ux*8,apy-uy*8);
        x.lineTo(apx-ux*2+uy*4,apy+uy*2+ux*4);
        x.lineTo(apx-ux*2-uy*4,apy+uy*2-ux*4);
        x.closePath();x.fill();
      }
    });
    x.fillStyle=acc;x.beginPath();x.arc(cx,cy,5,0,7);x.fill();
    x.fillStyle=mu2;x.font='12px "Times New Roman",serif';x.textAlign='center';
    x.fillText('x',W-14,cy-8);x.fillText('y',cx+10,16);
  }

  function drawConj(l1,l2){
    var c=cv2('tc2-c3'),x=ct('tc2-c3'),W=c.width,H=c.height;
    x.fillStyle='#fff';x.fillRect(0,0,W,H);
    var al=l2/l1,lC=W*.24,rC=W*.76,cy=H/2,sc=H*.36;
    [lC,rC].forEach(function(ccx){
      x.strokeStyle=grc;x.lineWidth=.5;
      for(var i=-1;i<=1;i+=.5){
        x.beginPath();x.moveTo(ccx+i*sc,cy-sc);x.lineTo(ccx+i*sc,cy+sc);x.stroke();
        x.beginPath();x.moveTo(ccx-sc,cy+i*sc);x.lineTo(ccx+sc,cy+i*sc);x.stroke();
      }
      x.strokeStyle=mu2;x.lineWidth=.5;
      x.beginPath();x.moveTo(ccx-sc,cy);x.lineTo(ccx+sc,cy);x.stroke();
      x.beginPath();x.moveTo(ccx,cy-sc);x.lineTo(ccx,cy+sc);x.stroke();
    });
    var N=10;
    for(var gi=0;gi<=N;gi++){
      var gv=-1+2*gi/N;
      x.strokeStyle=f1c;x.lineWidth=.7;
      x.beginPath();x.moveTo(lC-sc,cy-gv*sc);x.lineTo(lC+sc,cy-gv*sc);x.stroke();
      x.beginPath();x.moveTo(lC+gv*sc,cy-sc);x.lineTo(lC+gv*sc,cy+sc);x.stroke();
      x.strokeStyle=f2c;x.lineWidth=.7;
      x.beginPath();
      for(var j=0;j<=60;j++){var yy=-1+2*j/60;var hy=(yy>=0?1:-1)*Math.pow(Math.abs(yy),al);x.lineTo(rC+gv*sc,cy-hy*sc);}
      x.stroke();
      x.beginPath();
      for(var j=0;j<=60;j++){var xx=-1+2*j/60;var hy2=(gv>=0?1:-1)*Math.pow(Math.abs(gv),al);x.lineTo(rC+xx*sc,cy-hy2*sc);}
      x.stroke();
    }
    x.strokeStyle=acc+'50';x.lineWidth=.8;
    var Nm=7;
    for(var i=0;i<=Nm;i++){for(var j=0;j<=Nm;j++){
      var gx=-1+2*i/Nm,gy=-1+2*j/Nm;
      var hy=(gy>=0?1:-1)*Math.pow(Math.abs(gy),al);
      x.beginPath();x.moveTo(lC+gx*sc,cy-gy*sc);x.lineTo(rC+gx*sc,cy-hy*sc);x.stroke();
    }}
    x.fillStyle='#333';x.font='14px "Times New Roman",serif';x.textAlign='center';
    x.fillText('h',W/2,22);
    x.strokeStyle='#33380';x.lineWidth=1.2;
    x.beginPath();x.moveTo(W/2-35,28);x.lineTo(W/2+35,28);x.stroke();
    x.beginPath();x.moveTo(W/2+30,24);x.lineTo(W/2+35,28);x.lineTo(W/2+30,32);x.stroke();
    x.fillStyle=mu2;x.font='11px "Times New Roman",serif';
    x.fillText('Uniform grid (system 1)',lC,H-6);x.fillText('Image under h (system 2)',rC,H-6);
  }

  function drawDeriv(l1,l2){
    var c=cv2('tc2-c4'),x=ct('tc2-c4'),W=c.width,H=c.height;
    x.fillStyle='#fff';x.fillRect(0,0,W,H);
    var al=l2/l1,pad=55,gw=W-2*pad,gh=H-2*pad;
    var data=[],mx=0,NS=500;
    for(var i=0;i<=NS;i++){var y=-1+2*i/NS;if(Math.abs(y)<.003){data.push(null);continue;}var d=al*Math.pow(Math.abs(y),al-1);data.push(d);if(d<100&&d>mx)mx=d;}
    var yM=Math.min(Math.max(mx*1.15,3),100);
    x.strokeStyle=grc;x.lineWidth=.5;
    var nt=Math.min(5,Math.ceil(yM)),step=Math.max(1,Math.floor(yM/nt));
    for(var v=0;v<=yM;v+=step){var yy=pad+gh-(v/yM)*gh;x.beginPath();x.moveTo(pad,yy);x.lineTo(pad+gw,yy);x.stroke();x.fillStyle=mu2;x.font='11px sans-serif';x.textAlign='right';x.fillText(v.toFixed(v>10?0:1),pad-8,yy+4);}
    ['-1.0','-0.5','0','0.5','1.0'].forEach(function(l,i){var xx=pad+(i/4)*gw;x.beginPath();x.moveTo(xx,pad);x.lineTo(xx,pad+gh);x.stroke();x.fillStyle=mu2;x.font='11px sans-serif';x.textAlign='center';x.fillText(l,xx,pad+gh+16);});
    x.strokeStyle=acc;x.lineWidth=1.5;x.setLineDash([4,4]);
    var zx=pad+.5*gw;x.beginPath();x.moveTo(zx,pad);x.lineTo(zx,pad+gh);x.stroke();x.setLineDash([]);
    x.strokeStyle=p1c;x.lineWidth=2.5;x.beginPath();var st=false;
    for(var i=0;i<=NS;i++){if(data[i]===null||data[i]>yM*1.5){st=false;continue;}var xx=pad+(i/NS)*gw,yy=pad+gh-(Math.min(data[i],yM)/yM)*gh;st?x.lineTo(xx,yy):(x.moveTo(xx,yy),st=true);}x.stroke();
    if(al>1){x.fillStyle=acc;x.font='12px "Times New Roman",serif';x.textAlign='center';x.fillText('\u2192 0',zx,pad+gh+30);}
    else if(al<1){x.fillStyle=acc;x.font='13px "Times New Roman",serif';x.textAlign='center';x.fillText('\u2192 \u221E',zx,pad-8);x.beginPath();x.fillStyle=acc;x.moveTo(zx,pad+2);x.lineTo(zx-5,pad+10);x.lineTo(zx+5,pad+10);x.closePath();x.fill();}
    x.fillStyle=mu2;x.font='11px "Times New Roman",serif';x.textAlign='center';x.fillText('y',pad+gw/2,H-4);
    x.textAlign='left';x.fillText('|\u2202h\u2082/\u2202y|',pad+4,pad-10);
    x.strokeStyle=mu2+'60';x.lineWidth=.8;x.setLineDash([2,4]);
    var refY=pad+gh-(1/yM)*gh;
    if(refY>pad&&refY<pad+gh){x.beginPath();x.moveTo(pad,refY);x.lineTo(pad+gw,refY);x.stroke();x.setLineDash([]);x.fillStyle=mu2;x.textAlign='right';x.fillText('1',pad-8,refY+4);}
    x.setLineDash([]);
  }

  function update(){
    var l1=parseInt(sl1.value)/10,l2=parseInt(sl2.value)/10;
    document.getElementById('tc2-o1').textContent=l1.toFixed(1);
    document.getElementById('tc2-o2').textContent=l2.toFixed(1);
    var al=l2/l1;
    document.getElementById('tc2-mA').textContent=al.toFixed(2);
    if(Math.abs(al-1)<.01){
      document.getElementById('tc2-mS').textContent='= 1';
      document.getElementById('tc2-mC').innerHTML='C<sup>\u221E</sup> (identity)';
      document.getElementById('tc2-mH').textContent='1.00';
    }else if(al>1){
      document.getElementById('tc2-mS').textContent='\u2192 0';
      document.getElementById('tc2-mC').innerHTML='C\u2070 \\ C\u00B9';
      document.getElementById('tc2-mH').textContent=Math.min(1,1/al).toFixed(2);
    }else{
      document.getElementById('tc2-mS').textContent='\u2192 \u221E';
      document.getElementById('tc2-mC').innerHTML='C\u2070 \\ C\u00B9';
      document.getElementById('tc2-mH').textContent=al.toFixed(2);
    }
    drawPhase('tc2-c1',l1,p1c,f1c);
    drawPhase('tc2-c2',l2,p2c,f2c);
    drawConj(l1,l2);
    drawDeriv(l1,l2);
  }

  sl1.addEventListener('input',update);
  sl2.addEventListener('input',update);
  update();
})();
</script>

<div id="nh-container" style="margin:2em auto;max-width:920px;">
  <h4 style="text-align:center;margin:0 0 .2em;">Interactive: Nondifferentiable Homeomorphism</h4>
  <p style="text-align:center;color:#888;font-size:.82em;margin:0 0 .5em;">
    A circle deformed into a polygon via a homeomorphism \(h(\theta,t)\). The map is continuous (C&sup0;) but <em>not</em> differentiable at the corners — the tangent magnitude \(\|dh/d\theta\|\) has jump discontinuities there.
  </p>
  <div style="display:flex;align-items:center;justify-content:center;gap:12px;margin-bottom:8px;flex-wrap:wrap;">
    <span style="font-size:.85em;font-family:serif;">t =</span>
    <input type="range" id="nh-t" min="0" max="100" step="1" value="0" style="width:220px;">
    <span id="nh-tv" style="font-size:.85em;font-family:serif;min-width:32px;">0.00</span>
    <button id="nh-sq" style="font-size:.78em;padding:3px 10px;border:1px solid #ccc;border-radius:3px;background:#f8f8f8;cursor:pointer;font-family:serif;">Circle &rarr; Square</button>
    <button id="nh-star" style="font-size:.78em;padding:3px 10px;border:1px solid #ccc;border-radius:3px;background:#f8f8f8;cursor:pointer;font-family:serif;">Circle &rarr; Star</button>
    <button id="nh-tri" style="font-size:.78em;padding:3px 10px;border:1px solid #ccc;border-radius:3px;background:#f8f8f8;cursor:pointer;font-family:serif;">Circle &rarr; Triangle</button>
  </div>
  <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:10px;">
    <div style="text-align:center;">
      <div style="font-size:.82em;font-weight:600;margin-bottom:3px;">Homeomorphic deformation h(&theta;, t)</div>
      <canvas id="nh-sc" width="420" height="420" style="border:1px solid #ddd;border-radius:3px;background:#fff;max-width:100%;"></canvas>
    </div>
    <div style="text-align:center;">
      <div style="font-size:.82em;font-weight:600;margin-bottom:3px;">&Vert;dh/d&theta;&Vert; &mdash; tangent magnitude vs &theta;</div>
      <canvas id="nh-dc" width="420" height="420" style="border:1px solid #ddd;border-radius:3px;background:#fff;max-width:100%;"></canvas>
    </div>
  </div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin-top:10px;">
    <div style="background:rgba(0,0,0,.03);border-radius:6px;padding:8px 14px;min-width:110px;text-align:center;">
      <div style="font-size:11px;color:#888;">Max &Vert;dh/d&theta;&Vert;</div>
      <div style="font-size:18px;font-weight:500;" id="nh-md">1.00</div>
    </div>
    <div style="background:rgba(0,0,0,.03);border-radius:6px;padding:8px 14px;min-width:110px;text-align:center;">
      <div style="font-size:11px;color:#888;">Discontinuities in dh/d&theta;</div>
      <div style="font-size:18px;font-weight:500;" id="nh-nd">0</div>
    </div>
    <div style="background:rgba(0,0,0,.03);border-radius:6px;padding:8px 14px;min-width:110px;text-align:center;">
      <div style="font-size:11px;color:#888;">Homeomorphism class</div>
      <div style="font-size:18px;font-weight:500;" id="nh-cl">C&infin;</div>
    </div>
  </div>
</div>

<script>
(function(){
  var tS=document.getElementById('nh-t'),tV=document.getElementById('nh-tv');
  var sc=document.getElementById('nh-sc'),dc=document.getElementById('nh-dc');
  var S=sc.getContext('2d'),D=dc.getContext('2d');
  var mdEl=document.getElementById('nh-md'),ndEl=document.getElementById('nh-nd'),clEl=document.getElementById('nh-cl');
  var PI=Math.PI,PI2=2*PI;
  var mode='square';
  var lineCol='#534AB7',accentCol='#A32D2D',mutedCol='#999',gridCol='rgba(0,0,0,.06)';

  function targetR(th,m){
    if(m==='square'){return 1/Math.max(Math.abs(Math.cos(th)),Math.abs(Math.sin(th)));}
    if(m==='star'){return 1-.45*Math.abs(Math.sin(2.5*th));}
    var n=3,ang=((th%PI2)+PI2)%PI2,sec=Math.floor(ang/(PI2/n)),loc=ang-sec*(PI2/n)-PI/n;
    return Math.cos(PI/n)/Math.cos(loc);
  }
  function hPt(th,t,m){var r=(1-t)+t*targetR(th,m);return[r*Math.cos(th),r*Math.sin(th)];}
  function numDeriv(th,t,m){
    var e=1e-4,a=hPt(th-e,t,m),b=hPt(th+e,t,m);
    var dx=(b[0]-a[0])/(2*e),dy=(b[1]-a[1])/(2*e);
    return Math.sqrt(dx*dx+dy*dy);
  }
  function corners(m){
    if(m==='square')return[PI/4,3*PI/4,5*PI/4,7*PI/4];
    if(m==='star'){var p=[];for(var k=0;k<10;k++)p.push(k*PI/5);return p;}
    return[PI/6,PI/6+PI2/3,PI/6+4*PI/3];
  }

  function draw(){
    var t=parseFloat(tS.value)/100;
    tV.textContent=t.toFixed(2);
    var W=sc.width,H=sc.height,cx=W/2,cy=H/2,scale=W*.35;

    S.fillStyle='#fff';S.fillRect(0,0,W,H);
    S.strokeStyle=gridCol;S.lineWidth=.5;
    for(var i=-1;i<=1;i+=.5){S.beginPath();S.moveTo(cx+i*scale*1.5,0);S.lineTo(cx+i*scale*1.5,H);S.stroke();S.beginPath();S.moveTo(0,cy+i*scale*1.5);S.lineTo(W,cy+i*scale*1.5);S.stroke();}
    // Reference circle
    S.strokeStyle=mutedCol;S.lineWidth=.5;S.setLineDash([4,4]);S.beginPath();
    for(var i=0;i<=200;i++){var th=i/200*PI2;var x=cx+Math.cos(th)*scale,y=cy-Math.sin(th)*scale;i?S.lineTo(x,y):S.moveTo(x,y);}
    S.closePath();S.stroke();S.setLineDash([]);
    // Deformed shape
    S.strokeStyle=lineCol;S.lineWidth=2.5;S.beginPath();
    var N=500;
    for(var i=0;i<=N;i++){var th=i/N*PI2,p=hPt(th,t,mode);var x=cx+p[0]*scale,y=cy-p[1]*scale;i?S.lineTo(x,y):S.moveTo(x,y);}
    S.closePath();S.stroke();
    // Corner dots
    if(t>.01){corners(mode).forEach(function(th){var p=hPt(th,t,mode);S.fillStyle=accentCol;S.beginPath();S.arc(cx+p[0]*scale,cy-p[1]*scale,4,0,PI2);S.fill();});}

    // Derivative plot
    var DW=dc.width,DH=dc.height;
    D.fillStyle='#fff';D.fillRect(0,0,DW,DH);
    var pad=45,gw=DW-2*pad,gh=DH-2*pad;
    var samp=400,derivs=[],maxD=0;
    for(var i=0;i<=samp;i++){var th=i/samp*PI2;var d=numDeriv(th,t,mode);derivs.push(d);if(d>maxD)maxD=d;}
    var yMax=Math.max(2,Math.ceil(maxD*1.15));
    D.strokeStyle=gridCol;D.lineWidth=.5;
    var step2=Math.max(1,Math.floor(yMax/4));
    for(var v=0;v<=yMax;v+=step2){var yy=pad+gh-(v/yMax)*gh;D.beginPath();D.moveTo(pad,yy);D.lineTo(pad+gw,yy);D.stroke();D.fillStyle=mutedCol;D.font='11px sans-serif';D.textAlign='right';D.fillText(v.toFixed(0),pad-6,yy+4);}
    ['0','\u03C0/2','\u03C0','3\u03C0/2','2\u03C0'].forEach(function(l,i){var xx=pad+(i/4)*gw;D.beginPath();D.moveTo(xx,pad);D.lineTo(xx,pad+gh);D.stroke();D.fillStyle=mutedCol;D.font='11px sans-serif';D.textAlign='center';D.fillText(l,xx,pad+gh+16);});
    // Derivative curve
    D.strokeStyle=lineCol;D.lineWidth=2;D.beginPath();
    for(var i=0;i<=samp;i++){var xx=pad+(i/samp)*gw,yy=pad+gh-(Math.min(derivs[i],yMax)/yMax)*gh;i?D.lineTo(xx,yy):D.moveTo(xx,yy);}
    D.stroke();
    // Corner markers
    if(t>.01){corners(mode).forEach(function(th){var norm=((th%PI2)+PI2)%PI2;var xx=pad+(norm/PI2)*gw;D.strokeStyle=accentCol;D.lineWidth=1.5;D.setLineDash([3,3]);D.beginPath();D.moveTo(xx,pad);D.lineTo(xx,pad+gh);D.stroke();D.setLineDash([]);});}
    D.fillStyle=mutedCol;D.font='11px sans-serif';D.textAlign='center';D.fillText('\u03B8',pad+gw/2,DH-4);

    // Metrics
    mdEl.textContent=maxD.toFixed(2);
    ndEl.textContent=t>.01?corners(mode).length:'0';
    clEl.innerHTML=t<.01?'C<sup>\u221E</sup>':'C\u2070 (not C\u00B9)';
  }

  tS.addEventListener('input',draw);
  document.getElementById('nh-sq').addEventListener('click',function(){mode='square';draw();});
  document.getElementById('nh-star').addEventListener('click',function(){mode='star';draw();});
  document.getElementById('nh-tri').addEventListener('click',function(){mode='triangle';draw();});
  draw();
})();
</script>

### The Hartman-Grobman Theorem

We now arrive at one of the most fundamental and powerful results in the study of dynamical systems: the Hartman-Grobman Theorem. This theorem provides the rigorous justification for one of our most common analytical techniques: linearization. It formally establishes that, under certain conditions, the complex behavior of a nonlinear system in the close vicinity of an equilibrium point is qualitatively identical to the much simpler behavior of its linear approximation.

#### Intuition: Connecting Nonlinear and Linear Worlds

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The core message of the Hartman-Grobman theorem is profound:

In a small neighborhood of a hyperbolic equilibrium point, a nonlinear system is topologically conjugate to its linearization.

This is an extremely strong result. It doesn't just say the nonlinear system is "similar" to its linear approximation; it says there is a continuous, invertible map that transforms the nonlinear trajectories precisely onto the linear ones, while preserving the flow of time. This justifies our entire approach of analyzing the Jacobian matrix to understand the stability and local geometry of equilibria like nodes, saddles, and spirals. It guarantees that the picture we see in the linear system is not an illusion but a topologically faithful representation of the nonlinear system's behavior near the fixed point.

</div>

#### Formal Statement of the Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hartman-Grobman)</span></p>
  
Let $\mathbf{x_0}$ be an equilibrium point for the continuously differentiable system $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$ where $\mathbf{x} \in E$, an open subset of $\mathbb{R}^m$. Let $J(\mathbf{x_0})$ be the Jacobian matrix of $\mathbf{f}$ evaluated at $\mathbf{x_0}$.

If the equilibrium point $\mathbf{x_0}$ is hyperbolic (i.e., the matrix $J(\mathbf{x_0})$ has no eigenvalues with a real part equal to zero), then there exist neighborhoods $U$ and $V$ of $\mathbf{x_0}$ in $\mathbb{R}^m$ and a homeomorphism $h: U \to V$ such that for every initial point $\mathbf{x} \in U$, the flow of the nonlinear system, $\phi_t(\mathbf{x})$, is related to the flow of its linearization, $e^{J(\mathbf{x_0})t}\mathbf{z}$, by the conjugacy:

$$h(\phi_t(\mathbf{x})) = e^{J(\mathbf{x_0})t} h(\mathbf{x})$$

This must hold for all $t$ within some open time interval $I_0 \subset \mathbb{R}$ containing $0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Hartman-Grobman Visualization)</span></p>

It shows a hyperbolic equilibrium at the origin for:

$$
\text{linearized system: } \begin{cases} x' = x \\ y' = -y \end{cases}
\qquad
\text{nonlinear system: } \begin{cases} x' = x \\ y' = -y + x^2 \end{cases}
$$

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/hartman_grobmann_theorem.png' | relative_url }}" alt="a" loading="lazy">
</figure>

The point of the graphic is that near the origin, the nonlinear flow has the same qualitative structure as its linearization: both are saddles, with one stable direction and one unstable direction. The nonlinear system bends the unstable manifold into a curve, but the local phase portrait is still topologically equivalent to the linear one, which is exactly what the Hartman–Grobman theorem says.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The Hartman-Grobman Theorem is profoundly important because it provides the rigorous justification for linearization. It tells us that in the local neighborhood of a hyperbolic equilibrium point, the behavior of a nonlinear system is topologically equivalent to the behavior of its linearization around that point. This is why we can confidently use the tools of linear systems analysis (e.g., Jacobian eigenvalues) to determine the local stability and properties of equilibria even in highly nonlinear systems.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Power and Implications of Hartman-Grobman)</span></p>

This theorem gives us confidence in our analytical methods. When we encounter a complex, high-dimensional nonlinear system, we can:

1. Find its equilibrium points.
2. Linearize the system at each of these points by computing the Jacobian.
3. Calculate the eigenvalues of the Jacobian.

If the point is hyperbolic, Hartman-Grobman assures us that the local dynamics are completely characterized by the behavior of the linear system $\dot{\mathbf{z}} = J(\mathbf{x_0})\mathbf{z}$. The stability, the presence of saddle dynamics, and the spiral or nodal nature of the trajectories are all preserved. This allows us to use the well-understood tools of linear systems theory to draw robust conclusions about the behavior of highly nonlinear systems.

</div>

