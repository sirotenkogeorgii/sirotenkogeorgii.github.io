---
title: Dynamical Systems Theory in Machine Learning
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

# Dynamical Systems Theory in Machine Learning 

## Recommended Reading

These notes are designed to be a self-contained introduction. However, the field is vast, and further reading is highly encouraged. The following texts offer different perspectives on the material.

* **For an Intuitive Introduction:** These books provide an excellent, accessible entry point into the world of dynamical systems without an overwhelming focus on mathematical rigor. They are exceptionally well-written and focus on building conceptual understanding.
  * *Strogatz, S. H. Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering.*
  * *Alligood, K. T., Sauer, T. D., & Yorke, J. A. Chaos: An Introduction to Dynamical Systems.*
* **For a Rigorous Mathematical Treatment:** For those seeking a deeper, proof-oriented understanding, these texts are standard references in the field. Much of the material in the first part of this course is based on these foundational books.
  * *Perko, L. Differential Equations and Dynamical Systems.*
  * *Guckenheimer, J., & Holmes, P. Nonlinear Oscillations, Dynamical Systems, and Bifurcations of Vector Fields.*
  * *Kuznetsov, Y. A. Elements of Applied Bifurcation Theory.*
* **For Data-Driven and Machine Learning Approaches:** The second part of the course moves into modern, data-driven techniques. The field evolves rapidly, so primary literature is essential. However, these books provide a solid foundation.
  * *Kantz, H., & Schreiber, T. Nonlinear Time Series Analysis. (Covers older, yet still relevant, data-driven techniques).*
  * *Goodfellow, I., Bengio, Y., & Courville, A. Deep Learning. (A general, foundational text for deep learning).*
* **Primary Research Venues:** To stay at the cutting edge of dynamical systems in machine learning, it is crucial to follow the proceedings of the major machine learning conferences:
  * *NeurIPS*: Neural Information Processing Systems
  * *ICML*: International Conference on Machine Learning
  * *ICLR*: International Conference on Learning Representations

## Part I: An Introduction to Dynamical Systems

## Chapter 1: Fundamentals and Linear Systems

This field of dynamical systems provides the mathematical language for describing any system that evolves over time or another dimension. From the planets orbiting the sun to the firing of neurons in the brain, dynamical systems theory gives us the tools to model, understand, and predict change.

### Defining Dynamical Systems: Continuous and Discrete Time

A **dynamical system** is a mathematical framework for describing a system whose state evolves over time. The rule governing this evolution is fixed, meaning that the future state of the system is uniquely determined by its current state.

The formulation of a continuous dynamical system is the following:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dynamical system)</span></p>

Let $I \times R \subseteq \mathbb{R} \times \mathbb{R}^n$.
A map $\Phi : I \times R \to R$ is called a dynamical system or flow if:
1. $\Phi(0, x) = x, \quad \forall x \in M,$
2. $\Phi(t + s, x) = \Phi(t, \Phi(s, x)), \quad \forall s, t \in \mathbb{R}, x \in M,$
3. $\Phi$ is continuous in $(t, x)$.

</div>

At its core, the theory of dynamical systems is the study of systems that change. We can describe these changes using different mathematical objects, depending on whether the evolution is continuous or discrete. Dynamical systems are broadly classified into two categories based on how time is treated:

#### Continuous-Time Systems (Flows)

These systems evolve continuously. These are typically described by differential equations. They are most often described by **ordinary differential equations (ODEs)** or **partial differential equations (PDEs)**.

* **Ordinary Differential Equations (ODEs):** These describe the rate of change of a system's variables with respect to a single dimension, typically time. The notation often uses a dot to represent the time derivative.
  * A system can be multi-dimensional, with a state vector $x \in \mathbb{R}^p$.
* Here, $\dot{x}$ is a vector of temporal derivatives $[\dot{x}_1, \dot{x}_2, \dots, \dot{x}_p]^T$, and $f(x)$ is a function, often called the vector field, that maps the current state x to its rate of change. This single vector equation represents a set of $p$ coupled differential equations: 

$$\begin{align*} \dot{x}_1 &= f_1(x_1, x_2, \dots, x_p) \\ \dot{x}_2 &= f_2(x_1, x_2, \dots, x_p) \\ &\vdots \\ \dot{x}_p &= f_p(x_1, x_2, \dots, x_p) \end{align*} $$

* **Partial Differential Equations (PDEs):** These are used for systems that evolve along multiple dimensions simultaneously, such as time and space. For example, describing the temperature $u$ across a physical object would involve derivatives with respect to time $\dot{u}$ and spatial coordinates $u_x, u_y, \dots$.

#### Discrete-Time Systems (Maps)

These systems evolve in discrete steps. They are described by iterative functions or "maps."

* The state at the next time step, $x_t$, is a function of the state at the current time step, $x_{t-1}$.

$$x_t = f(x_{t-1})$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Discrete-Time Systems)</span></p>

* **Population Biology:** Models describing population growth, like the logistic map, often use discrete-time equations to represent generational changes.
* **Recurrent Neural Networks (RNNs):** All RNNs are fundamentally discrete-time dynamical systems, where the hidden state at each step is a function of the previous hidden state and the current input.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The distinction between continuous and discrete time can be fluid)</span></p>

The distinction between continuous and discrete time can be fluid. When we measure a real-world system (like climate or brain activity), we always do so at discrete time intervals due to the sampling frequency of our measurement devices. Furthermore, when we solve a differential equation on a computer, we must numerically approximate it by converting it into a discrete-time system. This tight connection is a recurring theme in the field.

</div>

### The State Space, Trajectories, and Vector Fields

A central concept in dynamical systems theory is to move away from looking at time series plots of individual variables and instead visualize the system's evolution geometrically.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(State Space, Trajectory)</span></p>

The **state space** is the set of all possible states a system can occupy. For a system with $p$ variables, the state space can be visualized as a $p$-dimensional space (typically $\mathbb{R}^p$) where each axis corresponds to one of the system's variables $(x_1, x_2, \dots, x_p)$.

* A specific state of the system at a given time $t$, defined by the vector $x(t) = [x_1(t), \dots, x_p(t)]$, corresponds to a single point in this space.
* As the system evolves over time, this point moves, tracing out a curve called a **trajectory**.
* A fundamental requirement for a well-defined dynamical system is that trajectories are **unique**. From any given point in state space, the path forward in time is uniquely determined. A point where trajectories could split indicates an incomplete description of the system; some variable is missing.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vector Field)</span></p>

The function $f(x)$ in the equation $\dot{x} = f(x)$ is called the **vector field**. It can be visualized by imagining that at every point $x$ in the state space, there is a vector attached. This vector, $f(x)$, points in the direction the system will move next, and its length indicates the speed of that movement. The trajectories of the system are simply curves that are everywhere tangent to this vector field.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometry and topology of state space, Attractors)</span></p>

Much of dynamical systems theory is concerned with the geometry and topology of the state space. By studying the structure of the vector field, we can understand the long-term behavior of the system without needing to solve the equations explicitly. We can ask questions like: Where do trajectories eventually end up? Do they converge to a stable point, get trapped in a periodic orbit, or exhibit more complex behavior? These long-term destinations for trajectories are known as **attractors**.

</div>

### Autonomous vs. Non-Autonomous Systems

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Autonomous and Non-Autonomous Systems)</span></p>

* An **autonomous system** is one where the governing equations do not explicitly depend on time. The vector field $f$ is a function of the state $x$ only.  
  
$$\dot{x} = f(x) $$

* A **non-autonomous system** is one where the rules of evolution explicitly change over time. This can be due to a time-varying parameter or an external input, often called a **forcing function**. 
  
$$\dot{x} = f(x, t)$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Forced Oscillator)</span></p>

Consider a simple oscillator. In its autonomous form, its behavior is self-contained. If we add a rhythmic external push (e.g., an "air puff" driving a pendulum), we introduce a forcing function $F(t)$.

$$\ddot{x} + a\dot{x} + bx = F(t)$$

The function $F(t)$, such as $k \cos(\omega t)$, makes the system non-autonomous because the dynamics now explicitly depend on $t$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Converting Non-Autonomous to Autonomous)</span></p>

There is a mathematical trick to convert any non-autonomous system into an autonomous one by augmenting the state space. While this can be convenient for theoretical analysis, it can sometimes obscure the underlying physics of what is forcing the system.

**The Trick:** For a non-autonomous system $\dot{x}_1 = f(x_1, t)$, we introduce a new variable $x_2 = t$. This creates a new, higher-dimensional autonomous system:

$$
\begin{align*}
\dot{x}_1 &= f(x_1, x_2) \\
\dot{x}_2 &= 1
\end{align*}
$$

The new system is autonomous because the right-hand side no longer explicitly contains $t$.

</div>

### Rewriting Higher-Order Systems

A key insight is that any higher-order ODE can be rewritten as a system of first-order ODEs. This simplifies our analysis, as we only need to develop tools for first-order systems.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Conversion of Higher-Order ODEs)</span></p>

An $n^{th}$-order one-dimensional ODE of the form $F(\frac{d^n x}{dt^n}, \dots, \frac{dx}{dt}, x, t) = 0$ can always be rewritten as a system of $m$ coupled first-order ODEs in an $n$-dimensional state space.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Harmonic Oscillator)</span></p>

Let's start with a second-order linear ODE for a harmonic oscillator: 

$$\ddot{x} + a\dot{x} + bx = 0$$

To convert this to a first-order system, we introduce two new state variables:
* $x_1 = x$ (position)
* $x_2 = \dot{x}$ (velocity)

Now, we take their time derivatives:
* $\dot{x}_1 = \dot{x} = x_2$
* $\dot{x}_2 = \ddot{x}$. From the original equation, we know $\ddot{x} = -a\dot{x} - bx$. Substituting our new variables, we get $\dot{x}_2 = -ax_2 - bx_1$.

This gives us a two-dimensional, first-order linear system:  

$$\begin{align*} \dot{x}_1 &= x_2 \\ \dot{x}_2 &= -bx_1 - ax_2 \end{align*} $$

</div>

### Analysis of Linear Systems

Linear systems are a cornerstone of dynamical systems theory. While they don't exhibit the complex behaviors of nonlinear systems (like chaos), they are fundamental for two reasons:

1. They are one of the few classes of systems that can be solved completely analytically.
> The behavior of a nonlinear system in the close vicinity of an equilibrium point can often be accurately approximated by a linear system.

### The One-Dimensional Case: $\dot{x} = ax$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Solution for 1D Linear System)</span></p>

For the 1D linear system

$$\dot{x} = ax$$

the solution is

$$x(t) = x_0 e^{at}$$

for the initial condition $x(0) = x_0$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let's analyze the simplest linear system, a single variable whose rate of change is proportional to its value, with an initial condition $x(0) = x_0$.

*Proof:* Solution by Separation of Variables

1. **Rearrange the equation** to separate variables $x$ and $t$: 
  
  $$\frac{dx}{x} = a dt$$

2. **Integrate both sides**:
  
  $$\int \frac{1}{x} dx = \int a dt$$

3. **Perform the integration**, which yields a logarithm and an integration constant $C$:
   
  $$\ln \lvert x\rvert = at + C$$

4. **Solve for** $x$ by taking the exponent of both sides:
   
  $$\lvert x\rvert = e^{at+C} = e^C e^{at}$$

5. **Define a new constant** $\tilde{C} = \pm e^C$ to absorb the absolute value and the constant term. This gives the general solution, which as you can see depends on the inition condition:
  
  $$x(t) = \tilde{C} e^{at}$$

6. **Apply the initial condition** $x(0) = x_0$. At $t=0$, we have $x(0) = \tilde{C}e^0 = \tilde{C}$. Therefore, $\tilde{C} = x_0$.

The final solution is: 

$$x(t) = x_0 e^{at}$$

</details>
</div>


<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fixed point or Equilibrium point)</span></p>

A point $x^{\ast}\in\mathbb{R}^n$ is called equilibrium point of a system ODEs, if $f(t,x^\ast)=0$ for all $t\in I$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Analysis</span><span class="math-callout__name">(Stability of the Equilibrium (1D case))</span></p>

The point $x=0$ is an **equilibrium point** (or **fixed point**) because if the system starts there ($x_0=0$), its derivative $\dot{x}$ is zero, and it remains there for all time. The stability of this equilibrium depends entirely on the sign of the coefficient $a$.

* **Case 1: $a > 0$ (Unstable Equilibrium)**
  * The solution $x(t) = x_0e^{at}$ grows exponentially.
  * If the system is perturbed even slightly from the origin, it will move away from it at an accelerating rate.
  * The vector field on the 1D line points away from the origin on both sides. This is also called a **repeller** or **source**.
* **Case 2: $a < 0$ (Stable Equilibrium)**
  * The solution $x(t) = x_0e^{at}$ decays exponentially to zero.
  * No matter where the system starts (besides the origin itself), it will always return to the equilibrium at $x=0$.
  * The vector field points towards the origin from both sides. This is also called an **attractor** or **sink**.
* **Case 3: $a = 0$ (Neutrally or Marginally Stable)**
  * The equation becomes $\dot{x} = 0$, meaning the velocity is always zero.
  * Wherever the system starts, it stays there forever. The entire $x$-axis is a continuum of fixed points.
  * It is called "marginally" stable because a small perturbation does not return to the original point, but it also doesn't grow unboundedly; it simply moves to a new fixed point.

</div>

### Higher-Dimensional Linear Systems: $\dot{x} = Ax$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Linear vector field, Non-linear trajectories)</span></p>

Let a dynamical system be linear and $x\in\mathbb{R}^2$, then

$$\dot x = Ax \implies x(t) = x(0)\exp^{At} \qquad \text{(for constant $A$)},$$

there the transformation $A \in \mathbb{R}^{2\times 2}$ is a lightly **damped rotation** $\implies$ **spiral toward the origin**:

$$A \sim \begin{pmatrix} \lambda_1 & -1 \\ 1 & -0.2 \end{pmatrix}$$

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/linear_vf_nonlinear_trajectory.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>The dynamical system is linear, but the trajectory is not necessarily linear</figcaption>
</figure>

</div>


<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(General Solution of Linear Systems)</span></p>

Consider a system of $n$ coupled linear ODEs, where $x \in \mathbb{R}^n$ and $A$ is an $n \times n$ matrix. 

$$\dot{x} = Ax, \quad x(0) = x_0$$

Assuming the matrix $A$ has $n$ distinct eigenvalues $\lambda_1, \dots, \lambda_n$ with corresponding eigenvectors $v_1, \dots, v_n$, the eigenvectors form a basis for the state space. Since the system is linear, any linear combination of individual solutions is also a solution. The general solution can therefore be written as a sum:

$$x(t) = \sum_{i=1}^{n} c_i v_i e^{\lambda_i t}$$

The coefficients $c_i$ are determined by the initial condition $x(0) = x_0$:

$$x_0 = \sum_{i=1}^{n} c_i v_i$$

The behavior of the system is a superposition of simple exponential behaviors along each of the eigendirections.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Derivation of the General Solution</summary>

*Proof:*

To find the solution, we can use an ansatz inspired by the 1D case, proposing a solution of a similar exponential form.

1. **Propose a solution form**, where $v$ is a constant vector (initial position, $x(0)$) and $\lambda$ is a scalar: 
    
  $$x(t) = v e^{\lambda t}$$
  
  We assume $v \neq 0$ to find a non-trivial solution.
2. **Substitute the ansatz into the ODE**. First, find the derivative $\dot{x}$: 
   
  $$\dot{x} = \frac{d}{dt}(v e^{\lambda t}) = \lambda v e^{\lambda t}$$

3. **Set the two sides of the ODE equal**:
 
  $$\dot{x} = Ax \implies \lambda v e^{\lambda t} = A(v e^{\lambda t})$$

4. **Simplify the equation**. Since $e^{\lambda t}$ is a non-zero scalar, we can cancel it from both sides: 
  
  $$\lambda v = Av$$

This is the fundamental **eigenvalue problem**. To find the solutions to our differential equation, we need to find the **eigenvalues** $\lambda$ and corresponding **eigenvectors $v$** of the matrix $A$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What this calculation actually proves)</span></p>

This calculation actually proves the following statement:

> If $v\neq 0$ and $(\lambda,v)$ is an eigenpair of $A$ (i.e. $Av=\lambda v$), then
> 
> $$x(t)=v e^{\lambda t}$$
> 
> is a solution of $\dot{x}=Ax$.

So it's a **verification**: you propose a form and check it satisfies the ODE under a condition, which turns into the eigenvalue problem.

It does **not** prove that *every* solution has that form. It only produces a family of solutions (one per eigenpair, when they exist).

</div>

**The General Solution**

We showed that each eigenpair $(\lambda_i, v_i)$ of $A$ generates a solution of the system: if the initial condition is an eigenvector, $x(0)=v_i$, then

$$x(t)=v_i e^{\lambda_i t}$$

satisfies $\dot x = Ax$.

If $A$ has $n$ linearly independent eigenvectors, then these vectors form a basis of the state space. Consequently, any initial state $x_0=x(0)$ can be written as a linear combination of eigenvectors:

$$x_0=\sum_{i=1}^n c_i v_i.$$

Because the system is linear, it preserves superposition: the derivative of a linear combination of solutions is the same linear combination of their derivatives. Indeed, using $Av_i=\lambda_i v_i$,

$$A\Big(\sum_{i=1}^n c_i v_i\Big)=\sum_{i=1}^n c_i Av_i=\sum_{i=1}^n c_i \lambda_i v_i.$$

Therefore, the solution starting from $x_0$ is obtained by summing the individual eigenvector solutions with the same coefficients:

$$x(t)=\sum_{i=1}^n c_i v_i e^{\lambda_i t}.$$

This gives the general solution, with the constants $c_i$ chosen to match the initial condition.


</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(General Solution of Linear Systems with Complex Eigenvalues)</span></p>

Assuming the matrix $A$ has $n$ distinct **complex** eigenvalues $\lambda_1, \dots, \lambda_n$ with corresponding eigenvectors $v_1, \dots, v_n$, the eigenvectors form a basis for the state space. Since the system is linear, any linear combination of individual solutions is also a solution. The general solution can therefore be written as a sum:

$$x(t) = \sum_{i=1}^{n} c_i v_i e^{\alpha_i t} (\cos(\omega_i t) + i\sin(\omega_i t))$$

The solution form reveals that the system's behavior has two components:

* An **exponential growth or decay** component, governed by the real part of the eigenvalue, $e^{\alpha_i t}.$
* An **oscillatory** component, governed by the imaginary part of the eigenvalue, $\cos(\omega_i t) + i\sin(\omega_i t)$.

The overall behavior is a spiral: the system oscillates while its amplitude grows or shrinks exponentially.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Derivation for Complex Eigenvalues</summary>

**Solution with Complex Eigenvalues**

The eigenvalues of a real matrix $A$ can be complex. Since $A$ is real, its complex eigenvalues must come in conjugate pairs: $\lambda = \alpha \pm i\omega$.

Recalling **Euler's formula**: 

$$e^{i\theta} = \cos(\theta) + i\sin(\theta)$$

We can rewrite the exponential term for a complex eigenvalue $\lambda_i = \alpha_i + i\omega_i$:

$$e^{\lambda_i t} = e^{(\alpha_i + i\omega_i)t} = e^{\alpha_i t} e^{i\omega_i t} = e^{\alpha_i t}(\cos(\omega_i t) + i\sin(\omega_i t))$$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Having linearly independent eigenvectors is enough)</span></p>

The real requirement is **$n$ linearly independent eigenvectors**, not **$n$ distinct eigenvalues**.

What your theorem currently says is a **sufficient condition**:

* If $A$ has $n$ **distinct eigenvalues**, then the corresponding eigenvectors are automatically linearly independent.
* So $A$ is diagonalizable, and the formula
  
  $$x(t)=\sum_{i=1}^n c_i v_i e^{\lambda_i t}$$
  
  is valid.

But "distinct eigenvalues" is **stronger than necessary**. The theorem still works whenever $A$ has a basis of eigenvectors, even if some eigenvalues are repeated.

</div>

### A Geometric Classification of 2D Linear Equilibria

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Nullclines)</span></p>

**Nullclines** are curves in the state space where the rate of change of one of the variables is zero.
* The $x_1$-nullcline is the set of points where $\dot{x}_1 = 0$.
* The $x_2$-nullcline is the set of points where $\dot{x}_2 = 0$.

Equilibrium points must lie at the intersection of all nullclines, as this is where all derivatives are zero simultaneously. For linear systems, nullclines are straight lines passing through the origin. They divide the state space into regions with different flow directions.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Node)</span></p>

The *equilibrium point* of a linear system of ODEs is called **node** if $A$ has *two real eigenvalues*.

</div>

The origin $x=0$ is always an equilibrium point for the system $\dot{x} = Ax$. We can classify the geometry of the flow around this equilibrium based on the eigenvalues of the matrix $A$. Let's consider a 2D system with eigenvalues $\lambda_1, \lambda_2$.

**Case 1: Node: Real Eigenvalues ($\omega_1 = \omega_2 = 0$)**

* **Stable Node**: Both eigenvalues are real and negative ($\lambda_1 < \lambda_2 < 0$).
  * *Geometry*: All trajectories move directly toward the origin. The system decays exponentially along all directions.
  * *Time Series*: Both $x_1(t)$ and $x_2(t)$ decay exponentially to zero.
  * *Stability*: Stable. Also called a **sink**.
* **Unstable Node**: Both eigenvalues are real and positive ($0 < \lambda_1 < \lambda_2$).
  * *Geometry*: All trajectories move directly away from the origin. The system grows exponentially along all directions.
  * *Time Series*: Both variables diverge exponentially.
  * *Stability*: Unstable. Also called a **source** or **repeller**.
* **Saddle Node**: Eigenvalues are real and have opposite signs ($\lambda_1 < 0 < \lambda_2$).
  * *Geometry*: This is a critical configuration. There is one special direction (the eigenvector $v_1$ corresponding to $\lambda_1 < 0$) along which trajectories converge toward the origin. There is another special direction (the eigenvector $v_2$ corresponding to $\lambda_2 > 0$) along which trajectories diverge. All other trajectories approach the origin for a time before being swept away along the unstable direction.
  * *Stability*: Unstable.
  * *Manifolds*:
    * The line spanned by $v_1$ is the **stable manifold** ($E^s$): The set of all points that flow to the equilibrium.
    * The line spanned by $v_2$ is the **unstable manifold** ($E^u$): The set of all points that flow away from the equilibrium.
    * Manifolds are **invariant**: any trajectory starting on a manifold stays on that manifold forever.

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/real_eigen_negative.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Stable Node: For two real eigenvalues with both being smaller than zero we get a convergence to zero in each dimension. All state space trajectories converge to the origin, the stable node.</figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/real_eigen_positive.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Unstable Node: For two real eigenvalues with both being lager than zero we get a divergence in each dimension. All state space trajectories diverge, except the one starting in origin, the unstable node.</figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/real_eigen_positive_negative.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Saddle Node: For two real eigenvalues with one being lager and the other being smaller than zero we get a convergence in the dimension which holds the eigenvalue smaller than zero and divergence in the other. The state space trajectories behave accordingly. The dynamical system has a saddle node in the origin.</figcaption>
</figure>


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Attractor or Manifold)</span></p>

If **some eigenvalues equal zero, attractors (manifolds) exist**. These will be introduced in subsequent chapters.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Spiral Point)</span></p>

The *equilibrium point* of a linear system of ODEs is called **spiral point** if $A$ *has complex eigenvalues*.

</div>

**Case 2: Spiral: Complex Conjugate Eigenvalues ($\lambda = \alpha \pm i\omega$, with $\omega \neq 0$)**

* **Stable Spiral (or Focus)**: The real parts of both eigenvalues are negative ($\alpha < 0$).
  * *Geometry*: Trajectories spiral inward toward the origin.
  * *Time Series*: Variables exhibit damped oscillations, converging to zero.
  * *Stability*: Stable.
* **Unstable Spiral (or Focus)**: The real parts of both eigenvalues are positive ($\alpha > 0$).
  * *Geometry*: Trajectories spiral outward, away from the origin.
  * *Time Series*: Variables exhibit oscillations with growing amplitude.
  * *Stability*: Unstable.
* **Center**: The real parts of both eigenvalues are exactly zero ($\alpha = 0, \lambda = \pm i\omega$).
  * *Geometry*: This is a very special case. Trajectories are perfect, closed orbits (ellipses or circles) around the equilibrium. The state space is filled with a continuous family of these orbits.
  * *Time Series*: Variables exhibit perfect, sustained oscillations with constant amplitude.
  * *Stability*: Neutrally stable. A small perturbation moves the system to a new, nearby orbit; it neither returns nor diverges.

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/complex_negative.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Stable Spiral: We get a damped oscillation in both dimensions. The state space trajectories are a inwards turning spirals converging to the stable spiral point at the origin.</figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/complex_positive.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Unstable Spiral / Focus: We get a increasing oscillation in both dimensions. The state space trajectories are a outwards turning spirals diverging from the unstable spiral point at the origin.</figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/complex_zero.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Center: We get an undamped oscillation in both dimensions. The state space trajectory are a circles and the origin is the center of the dynamical system.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Attractor or Manifold)</span></p>

**Case $\lambda_1$, $\lambda_2$ are real and $\lambda_1 = 0 > \lambda_2$:**

For two real eigenvalues with one being smaller and the other being equal to zero we get a convergence in the dimension which holds the eigenvalue smaller than zero and neither a convergence or a divergence in the other. This will be introduces as **attractor (manifold)**.

</div>

**Case 3: Some Eigenvalues with Zero Real Part**
* **Line or Plane of Equilibria:** One eigenvalue is zero, and the others are negative (e.g., $\lambda_1 < 0$, $\lambda_2 = 0$).
  * *Geometry*: There is an entire line (or plane in higher dimensions) of fixed points. This line is the eigenspace corresponding to $\lambda_2 = 0$. Trajectories from off this line will converge toward it along the stable eigendirections.
  * *Stability*: Marginally stable.
  * *Remark*: This configuration, sometimes called a **line attractor**, is crucial in neuroscience and machine learning for modeling memory. The system can be placed at any point along the line and will stay there, effectively "remembering" that state.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Line Attractor: Hypothalamus of Mice)</span></p>

The new Cell paper, used machine learning to model brain activity, which revealed that the **neural signal causing the state of aggression in mice was a line attractor**.

A line attractor is a specific pattern of activity, created by the interconnections between brain cells, that follows the shape of a valley. In a graph showing the flow of energy among neurons over time, the energy in a line attractor system tends to flow down the valley, like a ball rolling down into a trough. Once neural energy has reached the bottom, it tends to stay there and flow along a line, like a river moving along the bottom of a valley.

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/line_attractor1.png' | relative_url }}" alt="Kalman Smoother Schema" loading="lazy">
</figure>

In the line attractor signal encoding aggression, the farther that neural energy flows along the line, the more the animal's aggressive state escalates. Then after a fight, it takes time for the neural energy to flow back out of the valley. The researchers speculate that this gradual decay may correspond to the time it takes someone to calm down if they are very upset or angry.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Line Attractor: GRU)</span></p>

Two GRUs exhibit a pseudo-line attractor. Nullclines intersect at one point, but are close enough on a finite region to mimic an analytic line attractor in practice. (A,B) depict the same phase portrait on [−1.5, 1.5] and [−0.2, 0.2], respectively.

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/line_attractor2.png' | relative_url }}" alt="Kalman Smoother Schema" loading="lazy">
</figure>

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Why do we consider only two eigenvalues?)</span></p>

The text above does not state it explicitly, but we were considering a **special case $n=2$**: **2-dimensional linear system**. 

That is why it talks about exactly **two** eigenvalues and uses words like *node*, *saddle*, and *spiral* in a plane-phase-portrait sense. **We do not consider only two eigenvalues in general.**

That classification is for the **planar case**.

For an $n$-dimensional linear system

$$\dot x = Ax, \qquad A\in \mathbb R^{n\times n}$$

the matrix $A$ has $n$ eigenvalues counting algebraic multiplicity. **In general, all $n$ eigenvalues matter**.

What changes in higher dimension is mostly the **language** and **geometry**:

* In $2D$, you can draw the phase portrait, so you get names like *node*, *saddle*, *center*, *spiral*.
* In $nD$, the more useful description is in terms of:

  * **stable subspace** $E^s$: directions where solutions decay,
  * **unstable subspace** $E^u$: directions where solutions grow,
  * **center subspace** $E^c$: directions where growth/decay is not decided by the linear part alone.

So in higher dimensions, instead of saying "this is a saddle node in the plane," one usually says things like:

* "$A$ has 3 eigenvalues with negative real part and 1 with positive real part, so the equilibrium has a 3D stable subspace and a 1D unstable subspace."
* "All eigenvalues have negative real part, so the origin is asymptotically stable."
* "There are eigenvalues with both positive and negative real parts, so the origin is a saddle-type equilibrium."

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Is not only one eigenvalue enough to cause instable behavior?)</span></p>

> If we have $n$-dimensional system and the matrix $A$ has $n$ linearly independent eigenvectors forming a basis of the original vector space, then we can describe the solution $x(t)$ as a linear combination of evolutions of those eigenvectors. Because those evolutions are described via exponential functions, each element of the solution vector $x(t)$ we can write as a sum of exponential functions. If at least one eigenvalue if greater than 0, then the component diverges unless the corresponding coefficient is zero. Is it correct?

### 1. Is your reasoning about sums of exponentials correct?**

**The diagonalizable real-eigenvalue case**

If $A$ has $n$ linearly independent eigenvectors $v_1,\dots,v_n$, then yes:

$$x(t)=\sum_{i=1}^n c_i e^{\lambda_i t} v_i$$

And the $j$-th coordinate is

$$x_j(t)=\sum_{i=1}^n c_i (v_i)_j e^{\lambda_i t}$$

So each coordinate is indeed a linear combination of exponential terms.

That part is correct.

### 2. What needs correction?

#### 2.1. "If one eigenvalue is $>0$, then the component diverges" is not always true coordinate-by-coordinate**

What is true is this:

* If there is an eigenvalue with **positive real part**, then the equilibrium is **unstable**.
* For a **generic initial condition**, the solution grows in norm, because it usually has some nonzero component in the unstable eigendirection.

But an individual coordinate $x_j(t)$ need not diverge.

Why not? Because the unstable mode might not appear in that coordinate.

Example:

$$
A=\begin{pmatrix}1&0\\0&-1\end{pmatrix},
\qquad
x(t)=c_1 e^t \binom10 + c_2 e^{-t}\binom01.
$$

Then

$$x_1(t)=c_1 e^t,\qquad x_2(t)=c_2 e^{-t}$$

Here there is a positive eigenvalue $1$, so the system is unstable. But $x_2(t)\to 0$, not $\infty$.

So the correct statement is:

* If some eigenvalue has positive real part and the initial condition has a nonzero component in the corresponding unstable subspace, then **the solution vector** typically grows without bound in norm.
* Not every coordinate must diverge.

#### 2.2. If the unstable coefficient is zero, growth may disappear

If all coefficients corresponding to eigenvalues with positive real part are zero, then the initial condition lies entirely in the stable/center directions.

For example, if

$$x(t)=\sum_{\Re \lambda_i <0} c_i e^{\lambda_i t} v_i,$$

then the solution decays, even if $A$ also has other unstable eigenvalues, because this particular initial condition does not excite them.

That is exactly the idea behind the **stable manifold / stable subspace** for linear systems.

#### 2.3. Complex eigenvalues change the form

In real systems, you may also have complex eigenvalues

$$\lambda=\alpha \pm i\beta$$

Then the real solution is not just a sum of plain exponentials, but terms like

$$e^{\alpha t}(\cos \beta t, \sin \beta t)$$

So the real behavior is:

* $e^{\alpha t}$ controls growth/decay,
* $\cos(\beta t)$, $\sin(\beta t)$ cause oscillation.

That is where spirals and oscillations come from.

So in full generality, it is better to say:

* solutions are built from modes of the form $e^{\lambda t}$,
* and for real-valued solutions, complex eigenvalues produce oscillatory terms.

### 2.4. If $A$ is not diagonalizable, polynomial factors appear

If $A$ does not have a full eigenbasis, then the solution is not only sums of $e^{\lambda t}$, but may include terms like

$$t e^{\lambda t},\quad t^2 e^{\lambda t},\dots$$

coming from Jordan blocks.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(The clean corrected version)</span></p>

For the linear system

$$\dot x = Ax,$$

if $A$ is diagonalizable and has eigenpairs $(\lambda_i,v_i)$, then

$$x(t)=\sum_i c_i e^{\lambda_i t} v_i.$$

From this:

* If **all** eigenvalues satisfy $\Re(\lambda_i)<0$, then $x(t)\to 0$ for every initial condition.
* If **some** eigenvalue satisfies $\Re(\lambda_i)>0$, then the equilibrium is unstable.
* A particular solution still may converge to $0$ if its initial condition has no component in the unstable eigenspaces.
* What matters for stability is the behavior of the **whole vector** $x(t)$, usually measured by $\|x(t)\|$, not whether every coordinate separately diverges.

**In the language of stable and unstable manifolds**

For a linear system:

* the **stable subspace** is spanned by eigenvectors with $\Re(\lambda)<0$,
* the **unstable subspace** is spanned by eigenvectors with $\Re(\lambda)>0$,
* the **center subspace** is spanned by eigenvectors with $\Re(\lambda)=0$.

If your initial condition lies in the stable subspace, the solution decays.
If it has any nonzero component in the unstable subspace, the solution typically grows.

That is the higher-dimensional generalization of the $2D$ saddle picture.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why only real parts of eigenvalues matter)</span></p>

**Why “real part” specifically?**

Because for any complex $\lambda=\alpha+i\omega$,

$$e^{\lambda t}=e^{(\alpha+i\omega)t}=e^{\alpha t}(\cos\omega t+i\sin\omega t),$$

and the magnitude is

$$\lvert e^{\lambda t}\rvert=e^{\alpha t}.$$

So $\alpha=\operatorname{Re}(\lambda)$ is exactly the **growth/decay rate**.

Because the solution is built from **matrix exponentials**, and exponentials with **negative real exponent** decay.

**The key fact: $x(t)=e^{At}x_0$**

For the linear system $\dot x=Ax$, the unique solution is

$$x(t)=e^{At}x_0.$$

So the long-time behavior is entirely controlled by how $e^{At}$ behaves as $t\to\infty$.

**If $A$ has eigenvalues with negative real part, the exponential factors decay**

1. *Diagonalizable case (clean intuition)*
   
   If $A$ is diagonalizable, $A=V\Lambda V^{-1}$ with $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_n)$. Then

   $$e^{At}=V e^{\Lambda t} V^{-1},\qquad e^{\Lambda t}=\mathrm{diag}(e^{\lambda_1 t},\dots,e^{\lambda_n t}).$$

   If $\operatorname{Re}(\lambda_i)<0$, then

$$\lvert e^{\lambda_i t}\rvert = e^{\operatorname{Re}(\lambda_i)t}\to 0.$$

So every eigen-direction is multiplied by a decaying factor, which forces $e^{At}x_0\to 0$ for any initial state $x_0$. (The change of basis $V,V^{-1}$ only distorts by constant factors; it doesn’t change “decays to zero” into “doesn’t decay.”)

1. *Complex eigenvalues: spirals are just “oscillation × decay”*
   
   If $\lambda=\alpha\pm i\omega$, the real solutions look like

   $$e^{\alpha t}\big(\cos(\omega t)u + \sin(\omega t)w\big),$$

   for some real vectors $u,w$. The ($\cos$/$\sin$) part just rotates/oscillates, while the amplitude is scaled by $e^{\alpha t}$. If $\alpha<0$, the amplitude shrinks to $0$, so trajectories spiral inward.

**Even if $A$ is not diagonalizable, negative real parts still win**

In general, $A$ can be put into Jordan form $A=VJV^{-1}$. Each Jordan block with eigenvalue $\lambda$ contributes terms like

$$t^k e^{\lambda t}$$

(for some nonnegative integer $k$). Taking magnitudes gives roughly

$$t^k e^{\operatorname{Re}(\lambda)t}.$$

If $\operatorname{Re}(\lambda)<0$, the exponential decay dominates any polynomial factor $t^k$, so these terms still go to $0$. Hence $e^{At}\to 0$ and therefore $x(t)\to 0$.

A common way to summarize this is:

> If all eigenvalues satisfy $\max_i \operatorname{Re}(\lambda_i) < 0$, then there exist constants $M,\gamma>0$ such that
> 
> $$\lvert\lvert x(t)\rvert\rvert \le M e^{-\gamma t}\lvert\lvert x_0\rvert\rvert \quad \text{for all } t\ge 0,$$
> 
> so every trajectory converges to the origin exponentially fast.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Initial Conditions Manifolds)</span></p>

A stable or unstable manifold is **not just the eigenvectors themselves**.
It is the **set of all initial conditions** whose trajectories have the corresponding behavior.

For a **linear system**

$$\dot x = Ax,$$

this becomes very simple:

* the **stable manifold** is the subspace spanned by all eigenvectors whose eigenvalues have **negative real part**,
* the **unstable manifold** is the subspace spanned by all eigenvectors whose eigenvalues have **positive real part**,
* the **center manifold / center subspace** is spanned by eigenvectors whose eigenvalues have **zero real part**.

So in the linear case, yes: these manifolds are exactly the corresponding **eigenspaces / invariant subspaces**.

More precisely, for a diagonalizable linear system:

$$E^s = \operatorname{span}\lbrace v_i : \Re(\lambda_i)<0\rbrace$$

$$E^u = \operatorname{span}\lbrace v_i : \Re(\lambda_i)>0\rbrace$$

$$E^c = \operatorname{span}\lbrace v_i : \Re(\lambda_i)=0\rbrace$$

These are usually called **stable subspace**, **unstable subspace**, and **center subspace**.

The reason people say “manifold” is that in the **nonlinear case** the corresponding objects are usually **curved**, not linear subspaces. Then:

* the stable manifold is tangent to $E^s$ at the equilibrium,
* the unstable manifold is tangent to $E^u$,
* the center manifold is tangent to $E^c$.

So the clean distinction is:

* **Linear system:** manifold = subspace spanned by the relevant eigenvectors.
* **Nonlinear system:** manifold is generally curved, but near the equilibrium it points in those eigendirections.

Also, one more subtlety: if there are complex eigenvalues, you do not speak about a single real eigenvector in the same way. Instead, the stable/unstable/center subspaces are built from the corresponding **real invariant subspaces** associated with those eigenvalues.

> Stable and unstable manifolds are the sets of initial conditions with stable/unstable behavior; for linear systems they are exactly the invariant subspaces spanned by eigenvectors corresponding to eigenvalues with negative/positive real part.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hyperbolic equilibrium point, Hyperbolic Systems)</span></p>

An equilibrium point is called **hyperbolic** if none of its eigenvalues have a real part equal to zero. This means the system has no **centers** and no directions of marginal stability. Stable nodes, unstable nodes, saddle nodes, and spirals are all hyperbolic. This is an important property because the local behavior of hyperbolic equilibria is robust to small changes in the system.

</div>

### General Solutions for Linear Systems

We previously considered linear dynamical systems defined by systems of ordinary differential equations (ODEs) of the form:

$$\dot{\mathbf{x}} = A \mathbf{x}$$

where $\mathbf{x} \in \mathbb{R}^m$ is the state vector and $A$ is a square $m \times m$ matrix.

Under the strong assumption that the matrix $A$ has distinct eigenvalues ($\lambda_i$) and that its corresponding eigenvectors ($\mathbf{v}_i$) form a basis for the space, we derived a general solution. This solution expresses the evolution of the system, $\mathbf{x}(t)$, from an initial condition $\mathbf{x}_0$ as a linear combination of exponential and oscillatory terms.

This formulation allowed us to classify various types of equilibria (fixed points), such as stable/unstable nodes, saddles, stable/unstable spirals, and centers. However, the initial assumptions are restrictive. They do not cover all possible linear systems, specifically those where eigenvalues are repeated. To address this, we must develop a more general framework.

### A More General Approach: The Fundamental Theorem

To formulate a solution that covers all cases, we first introduce the concept of similar matrices, which helps classify systems based on their underlying dynamics.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Similar Matrices)</span></p>

Two square matrices, $A_1$ and $A_2$, are called **similar** if there exists an *invertible* matrix $S$ such that the following relationship holds:

$$A_1 = S A_2 S^{-1}$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Dynamics of Similar Matrices)</span></p>

All similar matrices have the same dynamics

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition behind Similar Matrices)</span></p>

The matrix $S$ represents an invertible transformation, or a change of variables (a change of basis). If two matrices are similar, it means that the dynamical systems they define are topologically equivalent; they possess the same fundamental dynamics, merely viewed from a different coordinate system. The eigendecomposition of a matrix, for instance, is a transformation that reveals its similarity to a diagonal matrix of its eigenvalues.

</div>

#### Canonical Forms for 2x2 Systems

For any $2 \times 2$ matrix, it can be shown that it is similar to one of three distinct canonical forms. These forms represent the fundamental classes of dynamics possible in two-dimensional linear systems.

1. **Distinct Real Eigenvalues:** The matrix is similar to a diagonal form.

   $$A \sim \begin{pmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{pmatrix}$$

   - **Eigenvalues:** The matrix has two real eigenvalues, $\lambda_1 = a$ and $\lambda_2 = b$.
   - **Dynamics:** This form corresponds to dynamics without an oscillatory component, such as saddles and stable or unstable nodes.

2. **Complex Conjugate Eigenvalues:** The matrix is similar to a form representing rotation and scaling.

   $$A \sim \begin{pmatrix} a & -b \\ b & a \end{pmatrix}$$

   - **Eigenvalues:** The matrix has two complex eigenvalues, $\lambda_{1,2} = a \pm ib$.
   - **Dynamics:** Eigenvalues for such a matrix come in complex conjugate pairs. This form can be decomposed into a scaling component (related to $a$) and a rotational component (related to $b$). This gives rise to spirals (stable if $a<0$, unstable if $a>0$) and centers (if $a=0$).

3. **Repeated Eigenvalues (Degenerate Case):** This is the case our previous solution did not cover. The matrix is similar to the form

   $$A \sim \begin{pmatrix} a & 1 \\ 0 & a \end{pmatrix}$$

   - **Eigenvalues:** The matrix has only one eigenvalue, $\lambda_1 = a$.
   - **Dynamics:** This matrix has one eigenvalue, $a$, with algebraic multiplicity two. However, it has only one corresponding eigenvector direction. This case is called degenerate because the eigenvectors do not form a basis for the space. The dynamics align with the single eigenvector direction, and the specific behavior depends on the initial conditions and the value of $a$.

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/DefectiveRepeatedEigenvalueSystem.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Phase portrait of the degenerate (defective) repeated-eigenvalue system</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Defective Repeated-Eigenvalue Case vs. Repeated-Eigenvalue Case)</span></p>

The case above is describing the **defective repeated-eigenvalue** case (a **Jordan block**), not just “repeated eigenvalue” in general.

**What’s special about**

$$A \sim \begin{pmatrix} a & 1 \\ 0 & a \end{pmatrix}$$

* The eigenvalue is $a$ with **algebraic multiplicity 2** (it appears twice in the characteristic polynomial).
* But the eigenspace is only **1-dimensional** (**geometric multiplicity 1**): there is only **one independent eigenvector direction**. You *don’t* get a basis of eigenvectors in $\mathbb R^2$. That’s what they mean by “degenerate.”

**Geometry/dynamics: it’s “scaling + shear”**

For the linear system $\dot x = Jx$, the matrix exponential is

$$e^{Jt}=e^{at}\begin{pmatrix}1&t\\0&1\end{pmatrix}.$$

So solutions are

$$x_2(t)=C_2 e^{at}, \qquad x_1(t)=e^{at}(C_1 + C_2 t).$$

**Geometric interpretation:**

* The factor $e^{at}$ is uniform expansion/decay (depending on sign of $a$).
* The $\begin{pmatrix} 1 & t \\ 0 & 1 \end{pmatrix}$ part is a **shear**: it pushes points sideways in the $x_1$ direction at a rate proportional to $t$ and to their $x_2$-component.

**Why “trajectories align with the single eigenvector direction”**

The eigenvector direction is the $x_1$-axis (the line $x_2=0$).

If $C_2\neq 0$, then

$$\frac{x_2(t)}{x_1(t)}=\frac{C_2}{C_1+C_2 t}\to 0 \quad \text{as } t\to\infty,$$

so the trajectory becomes **asymptotically tangent** to the eigenvector line $x_2=0$.

That’s what in the lecture was loosely called “collapse into a one-dimensional space”: not that the system literally becomes 1D, but that **the long-time direction of motion is dominated by the single eigenvector direction**.

**Dependence on $a$**

* $a<0$: everything goes to the origin (stable), but trajectories typically curve and approach the origin tangent to the eigenline (**stable improper node**).
* $a>0$: everything blows up (unstable), again typically aligning with the eigenline for large $t$.
* $a=0$: no exponential growth/decay; it’s pure shear: $x_2(t)=C_2$ constant and $x_1(t)=C_1+C_2 $t. Here there’s **no** “collapse to the origin”; trajectories are horizontal lines drifting.

Repeated eigenvalues **can** still have two independent eigenvectors (e.g. $A=aI$), but the Jordan form with the “1” above the diagonal is exactly the case where they **don’t** — that’s the degeneracy.

</div>

This theorem provides a single, universal solution for any linear system of ODEs, regardless of its eigenvalue structure.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Fundamental Theorem of Linear Dynamical Systems)</span></p>

Let $A$ be an $m \times m$ matrix and let $\mathbf{x}_0 \in \mathbb{R}^m$ be an initial condition. The initial value problem defined by:

$$\dot{\mathbf{x}} = A \mathbf{x}$$

$$\mathbf{x}(0) = \mathbf{x}_0$$

has a **unique solution** $x:\mathbb{R}\to \mathbb{R}^n$ of the form:

$$\mathbf{x}(t) = e^{At} \mathbf{x}_0 = \sum_{k=0}^{\infty} \frac{(At)^k}{k!} x_0,$$

where $e^{At}$ is the matrix exponential.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Difference between two general solutions)</span></p>

So they are equivalent under the condition that all values are distinct. If they are not distinct the first solution doesn't apply because it rest on the assumptions. And the degenerate case is only covered by the more general solution.

</div>


<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Matrix Exponential)</span></p>

The matrix exponential $e^{At}$ is defined in a manner analogous to the Taylor series expansion of the scalar exponential function:

$$e^{At} = \sum_{k=0}^{\infty} \frac{(At)^k}{k!} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \dots$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

It is straightforward to see why **this form constitutes a solution to the ODE**. If we take the temporal derivative of the solution $\mathbf{x}(t) = e^{At} \mathbf{x}_0$, we differentiate the series term-by-term:

$$\frac{d}{dt} \mathbf{x}(t) = \frac{d}{dt} \left( \sum_{k=0}^{\infty} \frac{A^k t^k}{k!} \right) \mathbf{x}_0$$

$$= \left( \sum_{k=1}^{\infty} \frac{A^k k t^{k-1}}{k!} \right) \mathbf{x}_0 = A \left( \sum_{k=1}^{\infty} \frac{A^{k-1} t^{k-1}}{(k-1)!} \right) \mathbf{x}_0$$

By re-indexing the sum (let $j=k-1$), we recover the original series:

$$= A \left( \sum_{j=0}^{\infty} \frac{(At)^j}{j!} \right) \mathbf{x}_0 = A e^{At} \mathbf{x}_0 = A \mathbf{x}(t)$$

This confirms that $\dot{\mathbf{x}} = A\mathbf{x}(t)$, satisfying the differential equation. The full proof of the theorem also requires showing this solution is unique, which can be done by assuming two distinct solutions and demonstrating they must be identical.

</div>

#### Equivalence of Solutions for Diagonalizable Systems

While the matrix exponential provides a powerful general solution, it is important to verify that it is consistent with the eigenvector-based solution we derived earlier for the case where $A$ is diagonalizable (i.e., has distinct eigenvalues).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name"></span></p>

Let's demonstrate that for a diagonalizable matrix $A$, the two solution forms are equivalent.

1. Recall the eigenvector-based solution in matrix form. The initial condition $\mathbf{x}_0$ is a linear combination of eigenvectors: $\mathbf{x}_0 = \sum c_i \mathbf{v}_i$. In matrix form, this is $\mathbf{x}_0 = V\mathbf{c}$, where $V$ is the matrix whose columns are the eigenvectors $\mathbf{v}_i$ and $\mathbf{c}$ is the vector of coefficients $c_i$. Since the eigenvectors form a basis, $V$ is invertible, so $\mathbf{c} = V^{-1}\mathbf{x}_0$.
2. The solution at time $t$ is $\mathbf{x}(t) = \sum c_i e^{\lambda_i t} \mathbf{v}_i$. This can be written in matrix form as:
   
  $$\mathbf{x}(t) = V \cdot \text{diag}(e^{\lambda_i t}) \cdot \mathbf{c} = V \begin{pmatrix} e^{\lambda_1 t} & & 0 \\ & \ddots & \\ 0 & & e^{\lambda_m t} \end{pmatrix} V^{-1} \mathbf{x}_0 $$

3. Now, analyze the matrix exponential solution. Since $A$ is diagonalizable, we can write its eigendecomposition as $A = V \Lambda V^{-1}$, where $\Lambda$ is the diagonal matrix of eigenvalues. Let's substitute this into the series definition of $e^{At}$:

  $$e^{At} = \sum_{k=0}^{\infty} \frac{(V \Lambda V^{-1}t)^k}{k!}$$
  
  Consider the term $(V \Lambda V^{-1})^k$:  
  
  $$(V \Lambda V^{-1})^k = (V \Lambda V^{-1})(V \Lambda V^{-1})\dots(V \Lambda V^{-1})$$
  
  The inner $V^{-1}V$ terms cancel out, leaving:  
  
  $$= V \Lambda^k V^{-1}$$  
  
  Substituting this back into the series:  
  
  $$e^{At} = \sum_{k=0}^{\infty} \frac{V \Lambda^k V^{-1}t^k}{k!} = V \left( \sum_{k=0}^{\infty} \frac{(\Lambda t)^k}{k!} \right) V^{-1}$$

4. Recognize the series. The sum in the middle is simply the definition of the matrix exponential for the diagonal matrix $\Lambda$. For a diagonal matrix, this is equivalent to taking the exponential of each diagonal element:
  
  $$\sum_{k=0}^{\infty} \frac{(\Lambda t)^k}{k!} = \text{diag}(e^{\lambda_i t}) = \begin{pmatrix} e^{\lambda_1 t} & & 0 \\ & \ddots & \\ 0 & & e^{\lambda_m t} \end{pmatrix}$$
  
  Therefore, the matrix exponential solution is:
  
  $$\mathbf{x}(t) = e^{At}\mathbf{x}_0 = V \begin{pmatrix} e^{\lambda_1 t} & & 0 \\ & \ddots & \\ 0 & & e^{\lambda_m t} \end{pmatrix} V^{-1} \mathbf{x}_0$$

This is identical to the matrix form of the eigenvector-based solution. The two forms are fully consistent when the matrix $A$ is diagonalizable.

</div>

#### The Degenerate Case: Repeated Eigenvalues

The true power of the Fundamental Theorem is that it also provides the solution for the degenerate case, where eigenvalues are repeated and the matrix is not diagonalizable.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


In the degenerate case, the solution involves not just exponential terms, but also polynomials of time ($t$). These polynomial terms arise from the off-diagonal elements in the canonical form of the matrix (e.g., the '1' in the third canonical form).


</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name"></span></p>


Consider the $2 \times 2$ matrix from the third canonical form, which has a repeated eigenvalue a: 

$$A = \begin{pmatrix} a & 1 \\ 0 & a \end{pmatrix}$$

The solution for a system governed by this matrix, $\mathbf{x}(t) = e^{At}\mathbf{x}_0$, has the form: 

$$\mathbf{x}(t) = e^{At}\mathbf{x}_0 = e^{at} \begin{pmatrix} 1 & t \\ 0 & 1 \end{pmatrix} \mathbf{x}_0$$

Notice the appearance of the linear term $t$ in the matrix. For higher-dimensional degenerate systems, higher-order polynomials of $t$ can appear in the solution. This is a direct consequence of the structure of the matrix exponential for non-diagonalizable matrices.


</div>


### Analysis of Extended Linear Systems

In our initial exploration, we focused on homogeneous linear systems of the form $\dot{x} = Ax$. We now extend this analysis to two important cases: systems with a constant offset (affine systems) and systems that explicitly depend on time (non-autonomous systems).

#### Inhomogeneous (Affine) Systems of ODEs

An affine system introduces a constant vector term, shifting the dynamics in state space.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Affine System of ODEs)</span></p>

An affine or inhomogeneous linear system of ordinary differential equations is defined by:  

$$\dot{x} = Ax + b$$

where $x, b \in \mathbb{R}^m$ and $A$ is an $m \times m$ matrix.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Shifting the Equilibrium)</span></p>


The addition of the constant vector b does not alter the fundamental dynamics of the system, which are dictated by the matrix $A$. Instead, its effect is to move the system's equilibrium point. The vector field remains unchanged relative to this new equilibrium.

To understand this, we first locate the new equilibrium, or fixed point, by finding the point $x^{\ast}$ where the flow is zero ($\dot{x}=0$).

Assuming the matrix $A$ is invertible, we solve for the fixed point: 

$$0 = Ax^{\ast} + b \implies Ax^{\ast} = -b \implies x^{\ast} = -A^{-1}b$$ 

This point $x^{\ast}$ is our new equilibrium, shifted from the origin.


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Equivalence of Dynamics via Change of Variables)</span></p>

We can formally prove that the dynamics remain the same by defining a new variable y that represents the state relative to the fixed point $x^{\ast}$.

1. **Define a new variable:** Let $y = x - x^{\ast}$. This is equivalent to $x = y + x^{\ast}$.
2. **Consider the dynamics of the new variable:** The temporal derivative of $y$ is $\dot{y} = \dot{x}$, since $x^{\ast}$ is a constant and its derivative is zero.
3. **Substitute into the original equation:** We can now express $\dot{y}$ in terms of $y$
   
   $\dot{y} = \dot{x} = Ax + b$  
   
   Substitute $$x = y + x^{\ast}: \dot{y} = A(y + x^{\ast}) + b = Ay + Ax^{\ast} + b$$  
   
   Now, substitute the expression for the fixed point, $x^{\ast} = -A^{-1}b$:
   
   $$\dot{y} = Ay + A(-A^{-1}b) + b = Ay - b + b$$

4. **Result:** The dynamics for the new variable are:
   
   $$\dot{y} = Ay$$
   
   This is precisely the homogeneous linear system we have already analyzed. The dynamics (stability, rotation, etc.) around the fixed point $x^{\ast}$ are identical to the dynamics of the homogeneous system around the origin. To find the full solution for $x(t)$, one solves for $y(t)$ and then recovers $x(t) = y(t) + x^{\ast}$.


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Non-Invertible Case)</span></p>

If the matrix $A$ is not invertible, it possesses at least one zero eigenvalue. In this scenario, a unique fixed point does not exist. This corresponds to the case of a center or, more generally, a line attractor (or plane/hyperplane attractor in higher dimensions). The system has a continuous manifold of equilibrium points along the direction of the eigenvector(s) associated with the zero eigenvalue(s).

</div>


#### Non-autonomous Systems with a Forcing Function

We now consider systems where the dynamics are explicitly influenced by time, driven by an external "forcing function."

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Non-autonomous System with Forcing Function)</span></p>

A non-autonomous linear system with a forcing function $f(t)$ is defined as a system that explicitly depends on time. For simplicity, we will analyze the scalar case:

$$\dot{x} = ax + f(t)$$ 

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Variation of Parameters)</span></p>

To solve this type of equation, we employ a powerful technique known as variation of parameters. The logic is as follows: we know the solution to the homogeneous part of the equation ($\dot{x} = ax$) is $x(t) = C e^{at}$, where $C$ is a constant. We now "promote" this constant to a time-dependent function, $k(t)$, and propose an ansatz (an educated guess) for the full solution that has a similar form. This allows the solution to adapt to the time-varying influence of $f(t)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Derivation of the Solution)</span></p>

1. **Formulate the ansatz:** Let the solution be of the form

   $$x(t) = (h(t) + C)e^{at}$$

   where $h(t)$ is an unknown function we need to determine and $C$ is a constant of integration.

2. **Take the temporal derivative:** Using the product rule, the derivative of our ansatz is:

   $$\dot{x}(t) = \frac{d}{dt}[(h(t) + C)e^{at}] = \dot{h}(t)e^{at} + a(h(t) + C)e^{at}$$

3. **Equate with the original ODE:** The definition of the system states that $\dot{x} = ax + f(t)$. We can substitute our ansatz for $x(t)$ into this definition:

   $$\dot{x}(t) = a[(h(t) + C)e^{at}] + f(t)$$

4. **Compare the two expressions for $\dot{x}(t)$:**

   $$\dot{h}(t)e^{at} + (h(t) + C)ae^{at} = a(h(t) + C)e^{at} + f(t)$$

   The term $(h(t) + C)ae^{at}$ appears on both sides and cancels out.

5. **Isolate the derivative of $h(t)$:** We are left with a simple expression:

   $$\dot{h}(t)e^{at} = f(t)$$

   Multiplying through by $e^{-at}$ gives:

   $$\dot{h}(t) = f(t)e^{-at}$$

6. **Integrate to find $h(t)$:** To find the function $h(t)$, we integrate both sides with respect to time:

   $$h(t) = \int f(t)e^{-at}\,dt$$

The full solution to the non-autonomous equation is therefore found by computing this integral for $h(t)$ and substituting it back into our original ansatz. This method provides a general recipe for solving first-order linear ODEs with a forcing function.

</div>


### Linear Maps (Discrete-Time)

Many dynamical systems, including various types of recurrent neural networks, are defined as maps rather than differential equations. These are discrete-time systems.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name"></span></p>

A discrete-time autonomous dynamical system is defined by a recursive prescription:  

$$\mathbf{x}_t = f(\mathbf{x}_{t-1})$$

We will focus on the affine linear map, which is the discrete-time analogue of the inhomogeneous systems discussed earlier:  

$$\mathbf{x}_t = A \mathbf{x}_{t-1} + \mathbf{b}$$  

Such a map generates a sequence of vector-valued numbers $\lbrace\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T\rbrace$ starting from an initial condition $\mathbf{x}_1$. A primary goal is to understand the limiting behavior of this sequence as $t \to \infty$.

</div>

#### Iterative Solution and Limiting Behavior (Scalar Case)

To build intuition, let's analyze the scalar case:

$$x_t = ax_{t-1} + b$$

**Recursive Expansion:** We can expand the expression recursively to understand its structure over time:

* At $t=2: x_2 = ax_1 + b$
* At $t=3: x_3 = a(x_2) + b = a(ax_1 + b) + b = a^2x_1 + ab + b$ 
* At $t=4: x_4 = a(x_3) + b = a(a^2x_1 + ab + b) + b = a^3x_1 + a^2b + ab + b$

**General Form:** Observing the pattern, the state at a general time step $T$ is:
  
  $$x_T = a^{T-1}x_1 + b(a^{T-2} + a^{T-3} + \dots + a^1 + a^0)$$
  
  The second term is a finite geometric series. We can write this more compactly as:
  
  $$x_T = a^{T-1}x_1 + b \sum_{i=0}^{T-2} a^i$$

**Limiting Behavior:** We are interested in what happens as $t \to \infty$. The convergence of this sequence depends entirely on the value of $a$.

* Condition for Convergence: The sequence converges only if the absolute value of a is less than one, i.e., $\lvert a\rvert < 1$.
* Analysis of Terms:
  * Initial Condition Term: For $\lvert a\rvert<1$, the term $a^{T-1}x_1$ decays to zero as $T \to \infty$. This means the system forgets about the initial condition exponentially fast.
  * Geometric Series Term: For $\lvert a\rvert <1$, the infinite geometric series converges to a fixed value:
* The Limit: Combining these results, the limit of $x_t$ as $t \to \infty$ is: 
  
  $$\lim_{t \to \infty} x_t = b \left( \frac{1}{1-a} \right)$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A Different View)</span></p>

Another powerful way to illustrate this solution is to plot $x_{t+1}$ as a function of $x_t$. The fixed points of the map are found where this function intersects the bisectrix line, defined by $x_{t+1} = x_t$.

</div>

### One-Dimensional Discrete-Time Linear Systems

We begin our exploration with the simplest case: a one-dimensional, discrete-time linear system. These systems, while seemingly basic, exhibit a rich set of behaviors that provide a foundational understanding for more complex, higher-dimensional systems.

#### The Recursive Linear Map

A one-dimensional discrete-time linear system is described by a recursive relationship that maps the state of the system at time $t$, denoted by $x_t$, to its state at the next time step, $t+1$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(1D Linear Map)</span></p>

The state $x_{t+1}$ of the system at time $t+1$ is given by an affine transformation of its state $x_t$ at time $t$:

$$x_{t+1} = f(x_t) = ax_t + b$$

where a and b are scalar constants. The parameter $a$ represents the slope, and $b$ is the intercept or offset.

</div>

#### Geometric Interpretation: The Cobweb Plot

To gain a deeper intuition for the system's evolution over time, we can visualize this recursive process graphically. We plot the function $x_{t+1} = ax_t + b$ against the bisectrix, which is the line $x_{t+1} = x_t$. The intersection of these two lines holds special significance, as we will see shortly.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The Cobweb Plot The Cobweb Plot is a powerful geometric technique for visualizing the trajectory of a discrete-time system. It provides an immediate feel for whether the system converges to a specific value, diverges to infinity, or exhibits other behaviors. The procedure is as follows:

1. **Initialization:** Start with an initial condition, $x_0$, on the horizontal axis.
2. **Evaluation:** Move vertically from $x_0$ to the function line $x_{t+1} = ax_t + b$. The height of this point gives the next state, $x_1$.
3. **Iteration:** To use $x_1$ as the next input, move horizontally from the point on the function line to the bisectrix ($x_{t+1} = x_t$). This transfers the output value $x_1$ to the horizontal axis, preparing it for the next iteration.
4. **Repeat:** From this new point on the bisectrix, move vertically again to the function line to find $x_2$, then horizontally to the bisectrix, and so on.

The path traced by these movements often resembles a spider's web, spiraling inwards or outwards, which gives the method its name.

<div class="pmf-grid">
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/coweb_line_attractor.jpeg' | relative_url }}" alt="a" loading="lazy">
    
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/coweb_minu_one_a.jpeg' | relative_url }}" alt="a" loading="lazy">
    
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/coweb_minus_one_a.jpeg' | relative_url }}" alt="a" loading="lazy">
    
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/coweb_negative_a_divergence.jpeg' | relative_url }}" alt="a" loading="lazy">
    
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/coweb_positive_a_convergence.jpeg' | relative_url }}" alt="a" loading="lazy">
    
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/coweb_positive_a_divergence.jpeg' | relative_url }}" alt="a" loading="lazy">
    
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/coweb_positive_one_a_negative_b.jpeg' | relative_url }}" alt="a" loading="lazy">
    
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/coweb_positive_one_a_positive_b.jpeg' | relative_url }}" alt="a" loading="lazy">
    
  </figure>
</div>

</div>

#### Fixed Points in One Dimension

##### Definition and Geometric Intuition

A central concept in dynamical systems is the notion of a fixed point, which is analogous to an equilibrium in continuous-time systems described by differential equations.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fixed Point)</span></p>

A point $x^{\ast}$ is a fixed point of a discrete-time system $x_{t+1} = f(x_t)$ if it remains unchanged by the map. That is, it satisfies the condition:


$$x^{\ast} = f(x^{\ast})$$

If the system is initialized at a fixed point, it will remain there for all future time steps. There is no movement at this point.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric)</span></p>

View of Fixed Points Geometrically, a fixed point is simply the intersection of the function graph $y = f(x)$ and the bisectrix $y=x$. At this specific point, the input to the function is exactly equal to its output, satisfying the definition $x^{\ast} = f(x^{\ast})$.

</div>

##### Algebraic Solution

We can find the fixed point not only graphically but also by solving the defining equation algebraically.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Derivation of the 1D Fixed Point)</span></p>

To find the fixed point $x^{\ast}$, we set the output equal to the input according to the definition:

$$x^{\ast} = ax^{\ast} + b$$

We then solve for 

$$x^{\ast}:x^{\ast} - ax^{\ast} = b$$

$$(1-a)x^{\ast} = b$$

Assuming $a \neq 1$, we can divide by $(1-a)$ to find the unique fixed point:

$$x^{\ast} = \frac{b}{1-a}$$

This algebraic solution precisely matches the limiting solution for convergent systems and identifies the point of intersection on the cobweb plot.

</div>

#### Stability Analysis of Fixed Points

A fixed point can be stable, unstable, or neutrally stable, depending on the behavior of nearby trajectories. This stability is determined entirely by the slope parameter, $a$.

##### Stable Fixed Points: $\lvert a\rvert < 1$

If the absolute value of the slope is less than one, any initial condition will lead to a trajectory that converges to the fixed point.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stable Fixed Point)</span></p>

A fixed point $x^{\ast}$ is stable if trajectories starting near $x^{\ast}$ converge towards it as $t \to \infty$. In the linear 1D case, this occurs when $\lvert a\rvert < 1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

On the Cobweb Plot, a slope with $\lvert a\rvert < 1$ is less steep than the bisectrix. This geometric configuration ensures that each step of the cobweb construction brings the state closer to the intersection point, causing the "web" to spiral inwards towards the fixed point.

</div>

##### Unstable Fixed Points: $\lvert a\rvert > 1$

If the absolute value of the slope is greater than one, the system will diverge from the fixed point, unless it starts exactly on it.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Unstable Fixed Point)</span></p>

A fixed point $x^{\ast}$ is unstable if trajectories starting near $x^{\ast}$ move away from it as $t \to \infty$. In the linear 1D case, this occurs when $\lvert a\rvert > 1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

When $\lvert a\rvert > 1$, the function line is steeper than the bisectrix. The Cobweb Plot immediately reveals that each iteration throws the state further away from the intersection point, causing the "web" to spiral outwards. The system still possesses a fixed point, but any infinitesimal perturbation from it will lead to divergence.


</div>

##### Neutrally Stable Points: $\lvert a\rvert = 1$

The case where the slope has an absolute value of exactly one represents a boundary between stability and instability.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Neutrally Stable Point)</span></p>

A point or system is neutrally stable if nearby trajectories neither converge towards nor diverge away from it, but instead remain in a bounded orbit. This occurs when $\lvert a\rvert = 1$.

We must consider two sub-cases:

* **Case 1:** $a = 1$ 
  * If $b \neq 0$, the system becomes $x_{t+1} = x_t + b$. This represents linear divergence, as a constant amount $b$ is added at each time step. There is no fixed point.
  * If $b = 0$, the system is $x_{t+1} = x_t$. In this scenario, every point is a fixed point. This is sometimes referred to as a line attractor, as there is a continuous set of fixed points.
* **Case 2:** $a = -1$
  * The system takes the form $x_{t+1} = -x_t + b$. This leads to oscillatory behavior.
  * If $b=0$, the system $x_{t+1} = -x_t$ simply flips the sign at each step (e.g., $x_0, -x_0, x_0, \dots$).
  * If $b \neq 0$, the system oscillates between two distinct values. This is known as a flip oscillation.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Flip Oscillation)</span></p>

Consider the system $x_{t+1} = -x_t + 1$. Let the initial state be $x_1 = 2$.

* $x_2 = -x_1 + 1 = -(2) + 1 = -1$
* $x_3 = -x_2 + 1 = -(-1) + 1 = 2$
* $x_4 = -x_3 + 1 = -(2) + 1 = -1$ 
  
The system enters a stable 2-cycle, oscillating between the values $2$ and $-1$. The amplitude of this oscillation depends on the initial value, but the oscillatory nature is preserved. This behavior is analogous to the center case in systems of linear differential equations, where solutions form a continuous set of stable orbits.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Analogy to Continuous Systems)</span></p>

The spectrum of solutions observed in discrete-time linear systems—stable and unstable fixed points, and mutually stable oscillations—is precisely the same class of solutions found in continuous-time linear systems of ordinary differential equations. This parallel provides a powerful conceptual bridge between the two domains.

</div>

### Higher-Dimensional Discrete-Time Linear Systems

We now generalize our analysis to systems with $m$ dimensions, where the state is represented by a vector and the dynamics are governed by a matrix transformation.

#### The General Affine Map

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($m$-Dimensional Linear Map)</span></p>

The state of the system is a vector $\vec{x}_t \in \mathbb{R}^m$. The evolution is given by the affine map:


$$\vec{x}_{t+1} = A\vec{x}_t + \vec{b}$$

where $A$ is an $m \times m$ square matrix and $\vec{b} \in \mathbb{R}^m$ is a constant offset vector.


</div>

#### Solving for Fixed Points in $m$ Dimensions

The definition of a fixed point remains the same: it is a point that is mapped onto itself.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Derivation of the $m$-Dimensional Fixed Point)</span></p>

Let $\vec{x}^{\ast}$ be a fixed point. It must satisfy the condition $\vec{x}^{\ast} = A\vec{x}^{\ast} + \vec{b}$. We solve for $\vec{x}^{\ast}$:

$$\vec{x}^{\ast} - A\vec{x}^{\ast} = \vec{b}$$

$$(I - A)\vec{x}^{\ast} = \vec{b}$$

where $I$ is the $m \times m$ identity matrix. If the matrix $(I-A)$ is invertible, we can find the unique fixed point by multiplying by its inverse:  

$$\vec{x}^{\ast} = (I-A)^{-1}\vec{b}$$

If $(I-A)$ is not invertible (i.e., it is singular), a unique fixed point does not exist. In this case, the system may have no fixed points or a continuous set of fixed points, such as a line attractor or a higher-dimensional manifold attractor.

</div>

#### System Dynamics and Diagonalization

To understand the system's trajectory, we analyze the behavior of the map when iterated over time. For simplicity, we first consider the homogeneous case where $\vec{b} = \vec{0}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Iterated Map Dynamics)</span></p>

For the system $\vec{x}_{t+1} = A\vec{x}_t$, the state at time $T$ is related to the initial state $\vec{x}_1$ by:

$$\vec{x}_T = A^{T-1}\vec{x}_1$$ 

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The Role of Diagonalization Calculating matrix powers $A^{T-1}$ can be complex. However, if the matrix $A$ is diagonalizable, the calculation simplifies significantly. A diagonalizable matrix can be written as $A = V \Lambda V^{-1}$, where $V$ is the matrix of eigenvectors and $\Lambda$ is a diagonal matrix of the corresponding eigenvalues $\lambda_i$.


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(System Solution via Diagonalization)</span></p>

If $A = V \Lambda V^{-1}$, then the power $A^{T-1}$ becomes:


$$A^{T-1} = (V \Lambda V^{-1})^{T-1} = V \Lambda^{T-1} V^{-1}$$

Raising the diagonal matrix $\Lambda$ to a power is trivial; we simply raise each diagonal element (the eigenvalues) to that power. The solution for the system's state at time $T$ is therefore:  

$$\vec{x}_T = V \Lambda^{T-1} V^{-1} \vec{x}_1$$  

This expression reveals that the long-term behavior of the system is governed by the powers of the eigenvalues of $A$.


</div>

#### Stability Analysis via Eigenvalues

The stability of the fixed point (in this case, the origin, since $\vec{b}=\vec{0}$) is determined by the magnitudes of the eigenvalues of the matrix $A$.

* Convergence: The system converges to the fixed point if all eigenvalues have an absolute value less than 1.  
  
  $$\text{If } \max_i \lvert\lambda_i\rvert < 1 \implies \text{Convergence}$$
  
  As $T \to \infty$, $\Lambda^{T-1} \to 0$, causing $\vec{x}_T \to \vec{0}$.
* Divergence: The system diverges if at least one eigenvalue has an absolute value greater than 1.  
  
  $$\text{If } \max_i \lvert\lambda_i\rvert > 1 \implies \text{Divergence}$$
  
  The component of the trajectory along the eigenvector corresponding to this eigenvalue will grow without bound.
* Neutral Stability / Manifold Attractors: If at least one eigenvalue has an absolute value of exactly 1 (and no eigenvalues have absolute values greater than 1), the system has neutral directions.  
  
  $$\text{If } \max_i \lvert\lambda_i\rvert = 1 \implies \text{Line or Manifold Attractor}$$
  
  The system will neither converge to the origin nor diverge to infinity, but will instead move along a stable manifold defined by the eigenvectors associated with the eigenvalues of magnitude one.
* Saddle-like Behavior: If the matrix $A$ has a mix of eigenvalues with magnitudes greater than and less than one, the system exhibits behavior analogous to a saddle point. Trajectories will converge towards the fixed point along directions spanned by eigenvectors with $\lvert\lambda_i\rvert < 1$ but will diverge along directions spanned by eigenvectors with $\lvert\lambda_i\rvert > 1$.


### The Flow of Dynamical Systems

#### The Flow Operator in Linear Systems

We begin our formal study by introducing a concept central to understanding how systems evolve over time: the flow operator. Consider a system of linear ordinary differential equations with an initial value.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Flow Map (or Flow Operator))</span></p>

For a linear system of differential equations of the form $\dot{x} = Ax$, with an initial condition $x(0) = x_0$, the solution is given by:

$$x(t) = e^{At} x_0$$

The operator $e^{At}$ that propagates the initial state $x_0$ forward in time is known as the flow operator.

More generally, we can define a flow map, denoted by $\phi$, which is a function of time $t$ and an initial condition $x_0$:

$$\phi(t, x_0) = e^{At} x_0$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name"></span></p>

You can visualize the flow map as a mechanism that takes a set of initial conditions and "transports" it forward in time by an amount $t$ to a new location in the state space. If you vary the time $t$, the path traced out by a single initial point $x_0$ is called its orbit or trajectory.

</div>

#### From Continuous to Discrete Time: The Sampling Equivalence

In many scientific and engineering contexts, particularly in physics, systems are naturally modeled using continuous-time differential equations. However, our observation and measurement of these systems are almost always discrete, taken at specific moments in time. This raises a crucial question: can we find a discrete-time system that is equivalent to a continuous-time one?

##### Equivalence for Linear Systems

Let us assume we have a continuous-time system that we sample at fixed time steps of duration $\Delta t$. The measurements are taken at times $0, \Delta t, 2\Delta t, \dots, n\Delta t$.

  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/flow_discretization.jpeg' | relative_url }}" alt="a" loading="lazy">
    
  </figure>

The flow map for this system transports a state from one sample point to the next:

$$x((n+1)\Delta t) = \phi(\Delta t, x(n\Delta t))$$

We can define a new matrix that encapsulates this discrete-time evolution.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Discrete-Time Equivalent Matrix)</span></p>

Let a continuous-time linear system be defined by $\dot{x} = Ax$. Its equivalent discrete-time evolution matrix, $\tilde{A}$, for a sampling time step $\Delta t$ is defined as:

$$\tilde{A} = e^{A \Delta t}$$

With this definition, we can construct a discrete-time linear map:

$$x_{n+1} = \tilde{A} x_n$$

where $x_n$ represents the state at time $t = n \Delta t$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

This linear map is equivalent to the continuous-time linear ODE system in a specific sense: for the same initial condition $x_0$, the solutions of the discrete and continuous systems agree exactly at the sampling points $t = n \Delta t$. The construction of $\tilde{A}$ ensures this correspondence, as it is precisely the flow operator for a duration of $\Delta t$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Equivalent Discrete-Time System)</span></p>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/EquivalentDiscreteTimeSystem.png' | relative_url }}" alt="a" loading="lazy">
</figure>

* **Left:** a continuous-time trajectory for $\dot x = Ax$, with states sampled every $\Delta t$.
* **Arrows between sample points:** the flow map $x_{n+1}=\phi(\Delta t,x_n)$.
* **Right:** the corresponding discrete-time evolution of the sampled states, governed by
  
  $$x_{n+1}=\tilde A x_n, \qquad \tilde A=e^{A\Delta t}$$
  
So the picture makes the key point explicit: the discrete system is not an approximation here, but the **exact sampled version** of the continuous linear system at times $t=n\Delta t$.

</div>

##### Equivalence for Affine Systems

This concept of equivalence can be extended to affine systems of differential equations, which include a constant offset term.

Consider the continuous-time affine system:

$$\dot{x} = Ax + c$$ 

We seek an equivalent discrete-time affine system of the form:

$$x_{n+1} = \tilde{A} x_n + b$$

where $\tilde{A}$ is defined as before: $\tilde{A} = e^{A \Delta t}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Determining the Discrete Offset)</span></p>

To find the corresponding offset vector $b$, we can enforce the condition that the fixed points of both the continuous and discrete systems must be identical.

1. **Find the fixed point of the continuous system:** 
   1. Set $\dot{x} = 0$. 
   2. $0 = Ax^{\ast} + c \implies x^{\ast} = -A^{-1}c$
2. **Find the fixed point of the discrete system:** 
   1. Set $x_{n+1} = x_n = x^{\ast}$. 
   2. $x^{\ast} = \tilde{A}x^{\ast} + b \implies (I - \tilde{A})x^{\ast} = b$
3. **Equate and Solve for $b$:** By substituting the expression for $x^{\ast}$ from the continuous system into the discrete system's fixed point equation, we can solve for $b$.

</div>

#### Applications and Advanced Concepts

The principles of establishing equivalence between continuous and discrete systems are not merely theoretical exercises. They have profound implications in modern machine learning and computational neuroscience.

##### Piecewise Linear Recurrent Neural Networks

An important class of models in machine learning is the Piecewise Linear Recurrent Neural Network (PL-RNN). These networks are often defined using the Rectified Linear Unit (ReLU) activation function, which is a piecewise linear function.

A typical PL-RNN update rule has the form:

$$x_{n+1} = Ax_{n} + Wg(x_n) + b$$

where $g$ is the ReLU nonlinearity, defined as $g(z) = \max(0, z)$.

The ideas of state-space dynamics and the equivalence between continuous and discrete forms can be extended to analyze these powerful computational models. For those interested in the details of this connection, the following resources are recommended:
* A paper by Monfared and Durstewitz presented at ICML 2020.
* The book Time Series Analysis (2013) by Ozaki, which contains a chapter on defining equivalent formulations for some nonlinear systems.

##### Line Attractors, Time Constants, and Memory

Let's revisit the concept of a line attractor, a continuous set of neutrally stable equilibria. In a 2D system with variables $z_1$ and $z_2$, a line attractor can arise when the nullclines (lines where $\dot{z}_1 = 0$ or $\dot{z}_2 = 0$) precisely overlap.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name">(Detuning for Arbitrary Time Constants)</span></p>

What happens if we slightly "detune" the system, so the nullclines no longer perfectly overlap but are very close? The vector field, which was exactly zero on the line attractor, will now be non-zero but very small in the "channel" between the slightly separated nullclines.

This has a profound consequence: by making subtle changes to the system's parameters (e.g., the slopes of the nullclines), we can create dynamics that evolve on arbitrarily long time scales. The system can be made to move extremely slowly without introducing any large physical time constants. This ability to generate a wide range of temporal scales is fundamental for complex information processing.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Addition Problem in Machine Learning)</span></p>

A classic benchmark for recurrent neural networks is the addition problem. The network receives two input streams:
1. A sequence of real numbers between 0 and 1.
2. A binary indicator bit (0 or 1).

The task is for the network to sum the real numbers only when the corresponding indicator bit is 1. The challenge lies in the potentially long gaps between periods where the indicator is active. The network must store the intermediate sum in its memory.

A line attractor provides a simple and elegant solution. A two-unit PL-RNN can solve this task:
* Integration and Storage: One unit integrates the input values (when the indicator bit is active) and stores the running total as a state on a line attractor. The system's state remains stable on this line, effectively acting as a memory device.
* Final Output: Once the sequence is complete, the final state on the line attractor represents the total sum.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Attractors in Natural Intelligence)</span></p>

This is not merely a machine learning construct. There is evidence for the existence of line attractors, plane attractors, and even torus attractors (shaped like a donut) in biological brains, for example, in the hippocampus, an area critical for memory and navigation.

</div>

### The Flow Map and Trajectories

Having built some intuition, we now proceed to a formal mathematical definition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dynamical System)</span></p>

A **dynamical system** is a commutative group or semigroup action, $\phi$, defined on a domain $T \times R$. It is composed of the following elements:
1. **A Time Domain ($T$)**: This is the set from which time values are drawn.
  * For continuous-time systems defined for all time, $T = \mathbb{R}$ (a group).
  * For systems defined only in forward time, $T = \mathbb{R}_{\ge 0}$ (a semigroup).
  * For discrete-time systems, $T = \mathbb{Z}$ (the integers).
2. **A State Space ($R$)**: This is an open set, $R \subseteq \mathbb{R}^d$, which contains all possible states the system can occupy. It is the space spanned by the dynamical variables.
3. **A Flow Map ($\phi$)**: An operator that maps a time and a state to a new state.
   
  $$\phi: T \times R \to R$$  
   
  We write this as $\phi(t, x)$ or sometimes abbreviate it as $\phi_t(x)$.

</div>

#### The Flow Operator

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Flow Map)</span></p>

Let $x$ be a point in the state space and let $s$, $t$ be elements of the time domain $T$ (e.g., $\mathbb{R}$ for continuous time).

The **flow map** $\phi$ must satisfy the following properties:
* **Neutral Element:** For any state $x$ in the state space $R$, evolving for zero time leaves the state unchanged.

$$\forall x \in R, \quad \phi(0, x)=\phi_0(x) = x$$ 

* **Semigroup (or Group) Property:** Evolving a point for a time $s+t$ is equivalent to first evolving it for time $t$ and then evolving the result for time $s$ (or vice versa). This property is described as commutative, meaning the order of time evolution operations can be exchanged.
  
$$\phi_{s+t}(x) = \phi_s(\phi_t(x)) = \phi_t(\phi_s(x))$$ 

* **Inverse Operation (for Groups):** As a consequence of the group property, if the system is time-reversible (i.e., time can be negative), we have an inverse operation. Evolving forward by time $t$ and then backward by time $t$ returns the system to its original state.
  
$$\phi_t(\phi_{-t}(x)) = x$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition)</span></p>

Imagine a particle tracing a path in the state space. It should not matter whether you calculate its position after 5 seconds by moving it forward 2 seconds and then 3 seconds, or by moving it 3 seconds and then 2 seconds. The final position must be the same as moving it forward for 5 seconds directly. This consistency is fundamental.

</div>

#### Trajectories and Orbits

With the flow operator established, we can now precisely define the path that a point carves out in the state space over time.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Trajectory or Orbit)</span></p>

The **trajectory (or orbit)** of a dynamical system starting from an initial point $x_0$ is the solution curve, denoted $\gamma(x_0)$. It is the set of all points in the state space that lie on this solution curve for all time $t \in T$.

$$\gamma(x_0) = \lbrace \phi_t(x_0) \mid t \in T \rbrace$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

A critical feature of a well-defined dynamical system is the uniqueness of its trajectories. For any given initial point $x_0$, there can be only one trajectory passing through it. If two different curves could originate from the same starting point, it would imply that the state space is missing crucial information needed to predict the future state, and we would not have a deterministic dynamical system. We will explore the conditions that guarantee this uniqueness in the next chapter.

</div>

### Existence and Uniqueness of Solutions

Having defined the concepts of flows and trajectories, a fundamental question arises: given a system of differential equations, can we always expect it to have a unique solution for a given starting condition? This is the central question of existence and uniqueness.

#### The Core Problem: Do Unique Solutions Always Exist?

The unfortunate answer is no, unique solutions are not guaranteed for all systems. However, the fortunate reality is that for the vast majority of well-behaved systems, they almost always do. The conditions where uniqueness fails are quite specific.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name"></span></p>

Consider the following initial value problem:

$$\dot{x} = 3x^{2/3}, \quad x(0) = 0$$

This system has two distinct solutions that satisfy the initial condition:

1. **The Trivial Solution:** $u(t) = 0$
2. **A Non-trivial Solution:** $v(t) = t^3$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name"></span></p>

We must verify that both functions satisfy the differential equation and the initial condition.

* For $u(t) = 0$:
  * Initial Condition: $u(0) = 0$, which is satisfied.
  * Differential Equation: The time derivative is $\dot{u}(t) = 0$. Plugging into the equation gives $0 = (0)^{2/3} = 0$. The equation holds.
* For $v(t) = t^3$:
  * Initial Condition: $v(0) = 0^3 = 0$, which is satisfied.
  * Differential Equation: The time derivative is $\dot{v}(t) = 3t^2 = 3v^{2/3}$. 

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What causes this failure of uniqueness?)</span></p>

The problem lies at the point $x=0$. The vector field $f(x) = 3x^{2/3}$ is continuous at $x=0$, but it is not continuously differentiable there. Let's examine its derivative with respect to the dynamical variable $x$:

 $$\frac{df}{dx} = \frac{d}{dx}(3x^{2/3}) = 2x^{-1/3}$$
 
This derivative is undefined at $x=0$. This lack of smoothness in the vector field is precisely what allows for multiple solution paths to emerge from the same point.
  
<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/SeveralSolutionCurvesPassThroughTheSameInitialPoint.png' | relative_url }}" alt="a" loading="lazy">
</figure>

The key picture is already clear: the slope field is continuous along $x=0$, but the derivative with respect to $x$ blows up there, so uniqueness can fail.

Here is the plot of the slope field for the ODE

together with several solution curves that all pass through the same initial point $(0,0)$:

A few key facts:

$$f(x)=3x^{2/3}$$

is **continuous at $x=0$** because

$$\lim_{x\to 0} 3x^{2/3}=0=f(0)$$

But it is **not continuously differentiable there**. For $x\neq 0$,

$$f'(x)=2x^{-1/3},$$

which blows up as $x\to 0$, so $f'$ does not extend continuously to $0$. In fact, $f$ is not even differentiable at $0$.

That is exactly why uniqueness breaks down. The usual uniqueness theorem needs at least local Lipschitz regularity in $x$, and this vector field fails that at $0$. As a result, multiple solution paths can emerge from the same point $(0,0)$, including

$$x(t)\equiv $$

and also the delayed solutions

$$
x_a(t)=
\begin{cases}
0, & t\le a,\\
(t-a)^3, & t\ge a,
\end{cases}
\qquad a\ge 0.
$$

So the lack of smoothness at $0$ is precisely what allows several distinct trajectories to start from the same initial condition.

<div class="pmf-grid">
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/continuous_function_at_0.png' | relative_url }}" alt="Continuous function at 0" loading="lazy">
  </figure>

  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/discontinuous_derivative_at_0.png' | relative_url }}" alt="Discontinuous derivative at 0" loading="lazy">
  </figure>
</div>

</div>

#### The Fundamental Existence and Uniqueness Theorem

The issue identified in the counterexample is the exact problem that the following powerful theorem resolves. If we can guarantee that our vector field is smooth enough, we can guarantee a unique solution.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Fundamental Existence and Uniqueness Theorem)</span></p>

Let $E \subseteq \mathbb{R}^m$ be an open set (our state space) and let the vector field $f: E \to \mathbb{R}^m$ be a continuously differentiable function (i.e., $f \in C^1(E)$).

Then, for any initial condition $x_0 \in E$, there exists a constant $a > 0$ such that the initial value problem  

$$\dot{x} = f(x), \quad x(0) = x_0$$

has a **unique solution**, $x(t)$, within the so-called maximum interval of existence $(-a, a)$, which is a subset of $\mathbb{R}$.

Furthermore, this unique solution has the general form:

$$x(t) = x_0 + \int_0^t f(x(s))ds$$     

This integral expression is sometimes referred to as the solution operator.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

This theorem is the bedrock for much of dynamical systems theory. It tells us that as long as our system's rules of evolution (the vector field $f$) are smooth, we don't have to worry about non-uniqueness. The solution might not exist for all time (it could "blow up" in finite time), but in some local time interval around our starting point, the path is uniquely determined. While the integral form of the solution is general, it may not be solvable analytically and often requires a numerical solver.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bedrock for much of dynamical systems theory)</span></p>

This theorem is the bedrock for much of dynamical systems theory. It tells us that as long as our system's rules of evolution (the vector field $f$) are smooth, we don't have to worry about non-uniqueness. The solution might not exist for all time (it could "blow up" in finite time), but in some local time interval around our starting point, the path is uniquely determined. While the integral form of the solution is general, it may not be solvable analytically and often requires a numerical solver.

</div>

The requirement of being continuously differentiable ($C^1$) is sufficient, but it is actually stronger than necessary. A weaker, more general condition also guarantees uniqueness.

The theorem can be proven under the weaker assumption that the function $f$ is locally Lipschitz continuous. For any two points $x$ and $y$ in some local interval, a function is Lipschitz continuous if the absolute difference in its values is bounded by a constant multiple of the distance between the points.  

$$\lvert f(x) - f(y)\rvert \le L \lvert x - y\rvert$$

Here, $L$ is a positive real number known as the Lipschitz constant. Intuitively, this condition means that the slope of the function is bounded. Every continuously differentiable function is locally Lipschitz, but not every Lipschitz continuous function is differentiable, making this a more general condition.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Picard-Lindelof)</span></p>
  
Let an IVP be given. Let $f$ be globally Lipschitz-continuous with respect to $x$. Then there exists a unique solution $x: I \to R$ of the IVP for each $x_0 \in R$, where $R$ is a some subset of $\mathbb{R}^n$.

</div>

#### On Solving Non-Linear Systems

While the theorem guarantees the existence of a unique solution for a broad class of systems, it does not provide a general method for finding it.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

In the most general case, non-linear systems of differential equations cannot be solved analytically. However, for certain scalar cases or systems with special structures, analytical techniques exist. These include:

* **Separation of Variables:** Rearranging the equation so that all terms involving one variable are on one side and all terms involving the other variable are on the other side, allowing for direct integration.
* **Variational Calculus:** A more advanced method for solving certain classes of problems.

For most complex systems encountered in practice, numerical methods are the primary tool for approximating the unique solution trajectories that the theorem guarantees.

</div>

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

For a dynamical $(\Phi_t)_{t\in I}$ on the metric space $(E,\text{dist})$ an attractor $A\subset B \subset E$ is a closed subset such that
* $\Phi_t(A) \subset A, \forall t\in I$,
* there exists a basin of attraction $B$ of $A,$ which is defined as the open set

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

Let's assume $x^{\ast}$ is a fixed point of the system $\dot{x} = f(x)$. To examine its stability, we introduce a small perturbation, $\epsilon$, and observe whether it grows or decays.

Let $x = x^{\ast} + \epsilon$. The dynamics of this perturbed state are given by:

$$\dot{x} = \frac{d}{dt}(x^{\ast} + \epsilon) = f(x^{\ast} + \epsilon)$$

Now, we perform a Taylor series expansion of $f(x)$ around the fixed point $x^{\ast}$. We are interested in the behavior for very small $\epsilon$, so we only keep the linear terms.

$$f(x^{\ast} + \epsilon) \approx f(x^{\ast}) + \frac{df}{dx}\bigg|_{x=x^{\ast}} \cdot \epsilon + O(\epsilon^2)$$

In higher dimensions, this becomes:

$$\mathbf{f}(\mathbf{x}^{\ast} + \boldsymbol{\epsilon}) \approx \mathbf{f}(\mathbf{x}^{\ast}) + J(\mathbf{x}^{\ast}) \boldsymbol{\epsilon} + O(\|\boldsymbol{\epsilon}\|^2)$$

Now we can formulate a differential equation for the perturbation $\boldsymbol{\epsilon}$.

$$\dot{\boldsymbol{\epsilon}} = \dot{\mathbf{x}} - \dot{\mathbf{x}}^{\ast}$$

By definition of a fixed point, $\mathbf{x}^{\ast}$ is constant, so $\dot{\mathbf{x}}^{\ast} = 0$. Furthermore, $\mathbf{f}(\mathbf{x}^{\ast}) = 0$. Substituting our Taylor expansion for $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}^{\ast} + \boldsymbol{\epsilon})$, we get:

$$\dot{\boldsymbol{\epsilon}} \approx (\mathbf{f}(\mathbf{x}^{\ast}) + J(\mathbf{x}^{\ast}) \boldsymbol{\epsilon}) - 0$$

$$\dot{\boldsymbol{\epsilon}} \approx J(\mathbf{x}^{\ast}) \boldsymbol{\epsilon}$$

This is a linear system of differential equations describing the evolution of the perturbation. We have thus linearized the dynamics around the fixed point. The stability of the original nonlinear system, in the local vicinity of $x^{\ast}$, is determined by the stability of this linear system.

#### Classification of Equilibria in Nonlinear Systems

By analyzing the eigenvalues of the Jacobian matrix $J$ evaluated at the equilibrium point $\mathbf{x}_0$, we can classify its stability, just as we did for fully linear systems.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stability of an Equilibrium Point)</span></p>

Let $\mathbf{x}_0$ be an equilibrium point of the system $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$, and let $J = J(\mathbf{x}_0)$ be the Jacobian evaluated at that point. Let $\lbrace\lambda_i\rbrace$ be the set of eigenvalues of $J$.

* The equilibrium $\mathbf{x}_0$ is stable if all eigenvalues of $J$ have a negative real part.
  * $Re(\lambda_i) < 0$ for all $i$.
* The equilibrium $\mathbf{x}_0$ is unstable if there is at least one eigenvalue of $J$ with a positive real part.
  * There exists at least one $i$ such that $Re(\lambda_i) > 0$.
* The equilibrium $\mathbf{x}_0$ is a saddle point if there is at least one eigenvalue with a negative real part and at least one eigenvalue with a positive real part.
  * There exist $i, j$ such that $Re(\lambda_i) < 0$ and $Re(\lambda_j) > 0$.

</div>

<figure>
  <img src="{{ '/assets/Fixed_Points.gif assets/images/notes/dynamical-systems/Fixed_Points.gif' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
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

* Atmospheric Convection: Simple climate models often reduce to this form.
* Chemical Reaction Systems: The rate of reaction between two species, $X$ and $Y$, is often proportional to the likelihood of them encountering each other, leading to product terms like $XY$.
* Epidemiology: Models for infectious diseases, such as the Susceptible-Infected-Recovered (SIR) model for epidemics like COVID, use these very types of equations to describe the dynamics of a population.

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
[
x_{n+1} = f(x_n),
]
a fixed point (x^*) satisfies
[
f(x^*) = x^*.
]

An **equilibrium** usually means a state where nothing changes in time.

For a continuous-time system
[
\dot{x} = g(x),
]
an equilibrium (x^*) satisfies
[
g(x^*) = 0.
]

So the difference is mostly about language and context:

* **fixed point** is the standard term for maps and discrete-time systems
* **equilibrium** is the standard term for ODEs and continuous-time systems

They both describe a state that stays where it is once the system reaches it.

There is also a nice connection between them. If (x^*) is an equilibrium of a flow, then starting at (x^*) gives a constant trajectory, so it is also a fixed point of the time-(t) flow map for every (t).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Equilibrium Point)</span></p>

An equilibrium point $x_0$ of a dynamical system $\dot{x} = f(x)$ is a point where the vector field is zero.  

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

#### Case Study: Characterizing the Equilibria of a Predator-Prey System

Let's make our model concrete with a specific set of parameters and analyze its behavior.

Parameters:

* $\alpha = 3$
* $\beta = 1$
* $\gamma = -1$
* $\lambda = -2$

Equilibrium Points: Using the formulas derived earlier, the two equilibrium points are:

1. $(x^{\ast}_1, y^{\ast}_1) = (0, 0)$
2. $(x^{\ast}_2, y^{\ast}_2) = (\frac{\lambda}{\gamma}, \frac{\alpha}{\beta}) = (\frac{-2}{-1}, \frac{3}{1}) = (2, 3)$

Now, we linearize the system at each of these points.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Analysis of the Equilibrium at $(0, 0)$)</span></p>


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name"></span></p>


First, we evaluate the general Jacobian matrix at the point $(x, y) = (0, 0)$:  

$$J(0, 0) = \begin{pmatrix} \alpha - \beta(0) & -\beta(0) \\ \gamma(0) & \gamma(0) - \lambda \end{pmatrix} = \begin{pmatrix} \alpha & 0 \\ 0 & -\lambda \end{pmatrix}$$  

Plugging in our specific parameter values $(\alpha=3, \lambda=-2)$:  

$$J(0, 0) = \begin{pmatrix} 3 & 0 \\ 0 & -(-2) \end{pmatrix} = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}$$  

The eigenvalues of a diagonal matrix are simply its diagonal entries. Therefore, the eigenvalues are $\lambda_1 = 3$ and $\lambda_2 = 2$.

Since both eigenvalues are real and positive, trajectories starting near the origin will be repelled from it along all directions. This type of equilibrium is called an unstable node.


</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Analysis of the Equilibrium at $(2, 3)$)</span></p>


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name"></span></p>

Next, we evaluate the Jacobian at the coexistence equilibrium $(x, y) = (2, 3)$ using our parameters $(\alpha=3, \beta=1, \gamma=-1, \lambda=-2)$:  

$$J(2, 3) = \begin{pmatrix} \alpha - \beta(3) & -\beta(2) \\ \gamma(3) & \gamma(2) - \lambda \end{pmatrix} = \begin{pmatrix} 3 - 1(3) & -1(2) \\ -1(3) & -1(2) - (-2) \end{pmatrix}$$   

$$J(2, 3) = \begin{pmatrix} 0 & -2 \\ -3 & 0 \end{pmatrix}$$  

To find the eigenvalues, we solve the characteristic equation $\text{det}(J - \lambda I) = 0$:  

$$\text{det} \begin{pmatrix} -\lambda & -2 \\ -3 & -\lambda \end{pmatrix} = (-\lambda)(-\lambda) - (-2)(-3) = \lambda^2 - 6 = 0$$  

This gives $\lambda^2 = 6$, so the eigenvalues are $\lambda_1 = +\sqrt{6} \approx 2.45$ and $\lambda_2 = -\sqrt{6} \approx -2.45$.

Since we have one positive real eigenvalue and one negative real eigenvalue, trajectories are attracted towards the equilibrium along one direction (the stable direction) and repelled from it along another direction (the unstable direction). This quintessential feature defines a saddle point.

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


An equilibrium point $x_0$ is called stable if for every neighborhood $\mathcal{U}$ of $x_0$ (e.g., an open ball of radius $\epsilon > 0$), there exists a smaller neighborhood $\mathcal{V}$ of $x_0$ (e.g., a ball of radius $\delta > 0$) such that any trajectory starting in $\mathcal{V}$ remains within $\mathcal{U}$ for all future time.

Formally: For every $\epsilon > 0$, there exists a $\delta > 0$ such that for any point $x$ with $\lvert x - x_0\rvert < \delta$, we have $\lvert\phi_t(x) - x_0\rvert < \epsilon$ for all $t \ge 0$.


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name">("If you start close enough, you stay close enough.")</span></p>

Note that this does not require the trajectory to approach $x_0$. A center in a frictionless pendulum system is stable, as trajectories that start nearby will orbit it, staying close but never converging.


</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Asymptotically Stable Equilibrium)</span></p>


An equilibrium point $x_0$ is asymptotically stable if:

1. It is stable.
2. There exists a neighborhood $\mathcal{W}$ of $x_0$ such that every trajectory starting in $\mathcal{W}$ converges to $x_0$ as time goes to infinity.

Formally: It is stable, and there exists $a \eta > 0$ such that if $\lvert x - x_0\rvert < \eta$, then

$$\lim_{t \to \infty} \phi_t(x) = x_0$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name">("If you start close enough, you eventually arrive at the equilibrium.")</span></p>

Sinks and stable spirals are asymptotically stable. This more general framework is crucial as it correctly classifies cases where linearization is inconclusive, such as when the eigenvalues of the Jacobian have zero real parts.

</div>

### Foundational Concepts in Topological Dynamics

This chapter introduces the fundamental topological concepts that are essential for a deeper understanding of dynamical systems. We will move beyond simple classifications to establish a rigorous framework for comparing different systems and analyzing the geometric structures they produce.

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

1. Continuity: The function h is continuous.
2. One-to-One and Onto (Bijective): The function h is a one-to-one map (injective) and maps onto the entire set $B$.
3. Continuous Inverse: The inverse function $h^{-1}: B \to A$ exists and is also continuous.

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
* **Separatrix Cycles:** Structures built from sequences of heteroclinic (and sometimes homoclinic) orbits are called separatrix cycles. These cycles can create boundaries in the phase space that separate regions of qualitatively different behavior.

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

* Forward and Backward Invariance: One can also define one-sided invariance. A set $S$ is forward invariant if $\phi_t(S) \subset S$ for all $t \ge 0$. This means that once you are in the set, you can never leave it as time moves forward. A set is backward invariant if $\phi_t(S) \subset S$ for all $t \le 0$.
* Discrete Time Systems (Maps): The concept applies equally to discrete maps. For a map $f: E \to E$, a set $S$ is invariant if applying the map (or its inverse) any number of times to a point in $S$ yields a point that is still in $S$. That is, $f(S) = S$.

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name"></span></p>


1. **Calculate the left-hand side (LHS):** First, we apply the flow of System $A$ to a point $x_0$, and then map the result using $h$.
  
   $$h(\phi^A_t(x_0)) = h(e^{-t}x_0) = \frac{1}{e^{-t}x_0} = \frac{e^t}{x_0}$$ 

2. **Calculate the right-hand side (RHS):** First, we map the point $x_0$ using $h$, and then apply the flow of System $B$ for a reparameterized time $\tau$.
  
   $$\phi^B_{\tau}(h(x_0)) = \phi^B_{\tau}\left(\frac{1}{x_0}\right) = e^{\tau} \cdot \frac{1}{x_0} = \frac{e^{\tau}}{x_0}$$ 

3. **Equate LHS and RHS to find $\tau$:** For the systems to be equivalent, we must have LHS = RHS.
   
   $$\frac{e^t}{x_0} = \frac{e^{\tau}}{x_0} \implies e^t = e^{\tau} \implies \tau = t$$

4. **Verify the time-preservation condition:** The reparameterization is $\tau(x_0, t) = t$. We check its derivative:
  
   $$\frac{\partial \tau}{\partial t} = \frac{d}{dt}(t) = 1$$
   
   Since $\frac{\partial \tau}{\partial t} = 1 > 0$, the condition is satisfied.


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

#### The Power and Implications of Hartman-Grobman

This theorem gives us confidence in our analytical methods. When we encounter a complex, high-dimensional nonlinear system, we can:

1. Find its equilibrium points.
2. Linearize the system at each of these points by computing the Jacobian.
3. Calculate the eigenvalues of the Jacobian.

If the point is hyperbolic, Hartman-Grobman assures us that the local dynamics are completely characterized by the behavior of the linear system $\dot{\mathbf{z}} = J(\mathbf{x_0})\mathbf{z}$. The stability, the presence of saddle dynamics, and the spiral or nodal nature of the trajectories are all preserved. This allows us to use the well-understood tools of linear systems theory to draw robust conclusions about the behavior of highly nonlinear systems.


### Foundational Concepts and Clarifications

This chapter revisits several central concepts from our previous discussions to solidify your understanding and clarify important nuances. A firm grasp of these ideas is essential before proceeding to more complex systems.

#### Recap: Core Concepts of System Equivalence

Previously, we introduced a set of powerful tools for comparing different dynamical systems. These include:

* Homeomorphism: A continuous function between topological spaces that has a continuous inverse function. It provides a formal way to say two spaces are "topologically the same."
* Topological Equivalence and Conjugacy: These concepts use homeomorphisms to establish when the phase portraits of two different dynamical systems are qualitatively identical. They allow us to transfer knowledge about a simpler system (like a linear one) to a more complex, nonlinear one.
* Manifolds and Invariant Sets: We discussed key geometric structures within the state space, such as stable and unstable manifolds, which are invariant sets—meaning any trajectory that starts on the set stays on the set for all time.

These ideas culminated in the Hartman-Grobman Theorem, a cornerstone of dynamical systems theory.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The Hartman-Grobman Theorem is profoundly important because it provides the rigorous justification for linearization. It tells us that in the local neighborhood of a hyperbolic equilibrium point, the behavior of a nonlinear system is topologically equivalent to the behavior of its linearization around that point. This is why we can confidently use the tools of linear systems analysis (e.g., Jacobian eigenvalues) to determine the local stability and properties of equilibria even in highly nonlinear systems.

</div>

#### Clarification: The Nature of Manifolds

There was a subtle but critical point regarding the definition of manifolds, specifically whether they are open or closed sets. The answer depends on the context in which you view the manifold.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Let us clarify the nature of a manifold with respect to its own dimensionality versus its embedding in a higher-dimensional ambient space.

* A Manifold is Locally an Open Set: When we say a manifold is an "open set," we are speaking locally and with respect to its own dimension. At any point $p$ on a $k$-dimensional manifold, one can always find a $k$-dimensional neighborhood around $p$ that contains only other points from the manifold. This neighborhood is homeomorphic to a $k$-dimensional open Euclidean ball.
  * Example: Consider a 1D limit cycle (a closed orbit) living in a 2D state space. If you pick any point on this circular manifold, you can always find a small 1D line segment (an open interval) around it that is entirely contained within the manifold.
* A Manifold is a Closed Set in its Ambient Space: When viewed from the perspective of the higher-dimensional space it resides in (the "ambient space"), the manifold is a closed set.
  * Example: For the same 1D limit cycle in 2D space, you cannot draw a 2D open ball around any point on the cycle that is completely contained within the manifold. Any such 2D ball will inevitably contain points from the surrounding 2D space that are not on the 1D limit cycle. Therefore, with respect to the 2D topology, the limit cycle is a closed set.

This distinction is crucial for a precise understanding of the geometric structures that govern system dynamics.

</div>

#### Clarification: Equilibria as Temporal Anchors

We also discussed an example of a mapping that failed to be a homeomorphism because of its behavior at an equilibrium point. This highlights a fundamental principle: the nature of equilibria profoundly impacts the global topology of a phase portrait.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


A system with a stable equilibrium and a system with an unstable equilibrium are not topologically equivalent. The intuition behind this is that equilibria serve as temporal anchors for the dynamics.

* An equilibrium point fixes where trajectories go as $t \to \infty$ (for a stable equilibrium) or where they came from as $t \to -\infty$ (for an unstable one). It imposes a specific and preferred temporal orientation on the flow in its vicinity.
* If you were to remove the equilibrium point from the state space, the defining "anchor" of the flow would be gone. In this modified space (without the equilibrium), it might be possible to find a homeomorphism between two different flows.
* However, once the equilibrium is included, its role as a temporal anchor breaks the equivalence. A continuous mapping (a homeomorphism) cannot reconcile the fundamentally different long-term behaviors of convergence versus divergence.

Think of equilibria as providing the ultimate destinations or origins for trajectories, and this function cannot be smoothly mapped away.


</div>


### Multistability and the Wilson-Cowan Model

We now move to a fascinating and ubiquitous phenomenon in nonlinear systems: multistability. This is the capacity for a system to possess more than one stable state (e.g., multiple stable equilibria) for a single set of parameters.

#### Introduction to Multistability

When a system is multistable, its long-term behavior depends entirely on its initial conditions. The state space is partitioned into distinct regions, known as basins of attraction. If a trajectory starts within a particular basin, it will inevitably converge to the stable state (or attractor) associated with that basin. The boundaries between these basins are called separatrices.

To explore these concepts, we will analyze a classic and influential model from computational neuroscience.

#### The Wilson-Cowan Model: A Neurodynamic System

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


The Wilson-Cowan model, introduced in the 1970s, was designed to explain the origin of oscillatory activity in the brain, such as the brain waves measured by an electroencephalogram (EEG). It is a foundational model that continues to inspire modern work in both neuroscience and machine learning (e.g., Neural ODEs).

The model simplifies the complexity of the brain by considering only two large, interacting populations of neurons:

1. A population of excitatory neurons, which tend to increase the activity of other neurons.
2. A population of inhibitory neurons, which tend to decrease the activity of other neurons.

The core mechanism for generating complex dynamics, including oscillations, arises from their interaction via feedback loops:
* Positive Feedback: The excitatory population excites itself.
* Interacting Feedback: The excitatory population excites the inhibitory one, which in turn sends back inhibition to the excitatory population, creating a negative feedback loop.

This model is an example of a mean-field approach, where the collective activity of a large, assumedly homogeneous group of cells is described by a single continuous variable.


</div>

#### Mathematical Formulation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name"></span></p>


The state of the Wilson-Cowan system is described by the average firing rates of the excitatory and inhibitory populations, denoted by $\nu_e(t)$ and $\nu_i(t)$, respectively. The dynamics are governed by the following system of nonlinear ordinary differential equations:

$$\tau_e \frac{d\nu_e}{dt} = -\nu_e + f_e(w_{ee}\nu_e - w_{ie}\nu_i - \theta_e)$$

$$\tau_i \frac{d\nu_i}{dt} = -\nu_i + f_i(w_{ei}\nu_e - \theta_i)$$

Where:

* $\tau_e, \tau_i$ are the time constants for each population.
* $w_{ab}$ represents the synaptic weight from population b to population a.
* $\theta_e, \theta_i$ are activation thresholds.
* $f(\cdot)$ is a nonlinear sigmoid activation function, given by: $f(x) = \frac{1}{1 + e^{-\beta x}} = (1 + e^{-\beta x})^{-1}$


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


The sigmoid function is a crucial component, modeling the input-output relationship of the neuron populations. It has two key properties:

1. It is bounded between $0$ and $1$, representing a firing rate that cannot be negative or infinitely high.
2. Its steepness is controlled by the slope parameter $\beta$. A larger $\beta$ results in a sharper, more switch-like transition from "off" $0$ to "on" $1$.

The term inside the sigmoid function, such as $w_{ee}\nu_e - w_{ie}\nu_i - \theta_e$, represents the total input current to the population.


</div>

#### State Space Analysis: Nullclines and Equilibria

To understand the model's behavior, we analyze its phase portrait in the $\nu_e, \nu_i$ state space. The first step is to find the nullclines.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name"></span></p>


A nullcline for a given state variable is the set of points in the state space where the rate of change of that variable is zero.

* The $\nu_e$-nullcline is the curve where $\frac{d\nu_e}{dt} = 0$. From the equations, this is defined by: 
  
  $$\nu_e = \left(1 + \exp\left[-\beta_e(w_{ee}\nu_e - w_{ie}\nu_i - \theta_e)\right]\right)^{-1}$$
  
  Graphically, this equation traces out an $N$-shaped curve in the ($\nu_e, \nu_i$) plane.

* The $\nu_i$-nullcline is the curve where $\frac{d\nu_i}{dt} = 0$. This is defined by: 
  
  $$\nu_i = \left(1 + \exp\left[-\beta_i(w_{ei}\nu_e - \theta_i)\right]\right)^{-1}$$
  
  This equation is simply a sigmoid function of $\nu_e$, resulting in a monotonically increasing curve.


</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name"></span></p>


The equilibria, or fixed points, of the system are the points where the dynamics cease, meaning all derivatives are simultaneously zero. Geometrically, these are the points where the nullclines intersect.

For certain parameter settings, the $N$-shaped $\nu_e$-nullcline can intersect the sigmoid-shaped $\nu_i$-nullcline at three distinct points, giving rise to three equilibria.


</div>

#### Qualitative Analysis of the Vector Field

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


The nullclines are powerful analytical tools because they segregate the state space into distinct regions of flow. The sign of a variable's derivative must flip every time a trajectory crosses that variable's nullcline.

Let's consider the flow directions in the ($\nu_e, \nu_i$) plane:

* Across the $\nu_i$-nullcline (the sigmoid curve):
  * For points above this curve, the linear term $-\nu_i$ dominates, so $\frac{d\nu_i}{dt} < 0$. The flow is downward.
  * For points below this curve, $\frac{d\nu_i}{dt} > 0$. The flow is upward.
* Across the $\nu_e$-nullcline (the $N$-shaped curve):
  * For points to the right of this curve, $\frac{d\nu_e}{dt} < 0$. The flow is to the left.
  * For points to the left of this curve, $\frac{d\nu_e}{dt} > 0$. The flow is to the right.

By sketching these general directions in each region bounded by the nullclines, we can build a qualitative picture of the system's phase portrait.


</div>

#### Stability and Saddle Manifolds

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


Combining the qualitative flow analysis with the location of the equilibria, we can infer their stability properties. In the three-equilibrium case:

* The two outer equilibria are stable fixed points. Trajectories starting nearby will spiral or move directly into them.
* The middle equilibrium is a saddle point. It is unstable, attracting trajectories along one direction (its stable manifold) and repelling them along another (its unstable manifold).

These stability assignments can be formally proven by calculating the Jacobian matrix of the system at each fixed point and analyzing its eigenvalues.

The saddle point and its manifolds play a crucial structural role. The stable manifold of the saddle is a particularly important curve. Any trajectory initiated exactly on this manifold will flow directly into the saddle point. More importantly, this manifold acts as the separatrix dividing the basins of attraction of the two stable equilibria.

This leads to a critical question: what happens if we start a trajectory just slightly to one side of the saddle's stable manifold?


</div>

### Attractors, Basins, and Limit Sets

In the study of dynamical systems, a central goal is to understand the long-term behavior of trajectories. Where do they originate, and where do they ultimately lead? This chapter introduces the foundational concepts of attractors, the regions from which they draw trajectories (their basins), and the mathematical tools used to formalize these ideas, namely alpha and omega limit sets.

#### Bistability, Basins of Attraction, and the Separatrix

Consider a simple one-dimensional system with two stable fixed points (equilibria) and one unstable fixed point between them.

* Point Attractors: The two stable equilibria are called point attractors. Trajectories that start near them will converge to them as time progresses.
* Basin of Attraction: Each point attractor has a surrounding neighborhood from which all trajectories converge to it. This neighborhood is called the basin of attraction.
* Separatrix: In this system, the unstable fixed point acts as a boundary. If an initial condition is to the left of this point, its trajectory will converge to the left attractor. If it starts to the right, it will converge to the right attractor. This boundary, which separates the basins of attraction, is known as a separatrix.

A system exhibiting this property of having two distinct attractors, each with its own basin of attraction, is said to possess bistability. This is a common and important feature in many natural and engineered systems.

#### Alpha and Omega Limit Sets

To more rigorously describe the long-term behavior of trajectories, we introduce the concepts of $\omega$-limit sets (for forward time) and $\alpha$-limit sets (for backward time).

##### Intuitive Understanding

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

An omega $\omega$ limit set is the set of points that a trajectory approaches as time goes to positive infinity $t \to \infty$. It describes where the system "ends up." For example, in a system with a single point attractor, the $\omega$-limit set for any point in its basin of attraction is the attractor itself.

An alpha $\alpha$ limit set is the set of points that a trajectory approaches as time goes to negative infinity $t \to -\infty$. It describes where the system "came from."

Consider a heteroclinic orbit, which is a trajectory that connects two different equilibrium points. For instance, an orbit that starts at an unstable saddle point and flows into a stable node.

* The $\omega$-limit set of this trajectory is the stable node.
* The $\alpha$-limit set of this trajectory is the saddle point.

For a trajectory in the basin of attraction of a point attractor that does not lie on a specific manifold, its $\alpha$-limit set might be at infinity, depending on whether the system's state space is bounded.

</div>

##### Formal Definition

Let a dynamical system be defined by a state space $E \subseteq \mathbb{R}^m$ and a flow operator $\phi_t(x_0)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Omega Limit Set)</span></p>

The omega limit set of a point $x_0 \in E$, denoted $\Omega(x_0)$, is the set of points that the trajectory through $x_0$ approaches as $t \to \infty$. It is formally defined as the intersection of the closures of the trajectory's future paths:

$$\Omega(x_0) = \bigcap_{s \in \mathbb{R}} \overline{\bigcup_{t > s} \phi_t(x_0)} = \bigcap_{s \in \mathbb{R}} \overline{ \lbrace \phi_t(x_0)\mid t > s\rbrace }$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

This definition works by considering the entire future path of the trajectory starting from some time $s$. As we let $s$ increase towards infinity, we take the intersection of all these future paths. This process "trims away" the transient parts of the trajectory, leaving only the set of points that the system visits infinitely often as $t \to \infty$. The closure (denoted by the overline) ensures that the limit points themselves are included in the set.


</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Alpha Limit Set)</span></p>

The alpha limit set of a point $x_0 \in E$, denoted $A(x_0)$, is defined analogously for reverse time $t \to -\infty$:

$$A(x_0) = \bigcap_{s \in \mathbb{R}} \overline{\bigcup_{t < s} \phi_t(x_0)} =  = \bigcap_{s \in \mathbb{R}} \overline{ \lbrace \phi_t(x_0)\mid t < s\rbrace }$$

</div>

#### Formal Definition of an Attractor

Using the concepts above, we can now provide a rigorous definition of an attractor and its associated basin of attraction.

##### Defining Properties

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Attractor)</span></p>

Given a dynamical system $(\mathbb{R} \times E, \phi)$ where $E \subseteq \mathbb{R}^m$ is the state space, a set $A \subseteq E$ is an **attractor** if it satisfies the following three properties:

1. **Invariance:** The set $A$ is invariant under the flow. This means that if a trajectory starts in $A$, it remains in $A$ for all time.
  
   $$\phi_t(A) = A \quad \forall t \in \mathbb{R}$$

2. **Attraction:** There exists a neighborhood of $A$, called the basin of attraction $B$ (where $A \subset B \subseteq E$), such that all trajectories starting in $B$ converge to $A$ as $t \to \infty$.
   
   $$\forall x_0 \in B, \quad \lim_{t \to \infty} d(\phi_t(x_0), A) = 0$$
  
   Here, $d(p, S)$ can be defined as the minimum distance from a point $p$ to any point in the set 
  
   $$S: d(p, S) = \inf_{y \in S} \|p-y\|$$

3. **Minimality:** $A$ is a minimal set with the first two properties. There is no smaller, proper subset of $A$ that is also an attractor with the same basin of attraction. If such a smaller set existed, the other points in $A$ would simply be part of the basin for that smaller attractor.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

An attractor is not necessarily a single point. It can be a more complex set, such as a closed orbit or even a fractal structure (a strange attractor). The key properties are that it is a self-contained "destination" for trajectories and that it draws in all trajectories from a surrounding region.

</div>

##### The Basin of Attraction

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Basin of Attraction)</span></p>

The **basin of attraction** $B$ for an attractor $A$ is defined as the largest set of initial conditions whose trajectories converge to $A$. It is the maximal set that satisfies the attraction property defined above.

</div>

### Periodic Behavior: Closed Orbits and Limit Cycles

While fixed points describe states of equilibrium, many systems exhibit sustained, periodic behavior. This corresponds to trajectories that form closed loops in the state space. This chapter explores these closed orbits, distinguishes between different types, and introduces the important concept of a limit cycle.

#### Closed Orbits in State Space

A closed orbit is a trajectory that returns to its starting point after a finite time, tracing a closed loop in the state space. Not all closed orbits are attractors. To understand a crucial distinction, we first analyze a specific nonlinear system.

#### Case Study: The Lotka-Volterra System as a Nonlinear Center

##### System Definition and Equilibria

Let us revisit the Lotka-Volterra system, a model of predator-prey dynamics:

$$\begin{aligned} \dot{x} &= \alpha x - \beta xy \\ \dot{y} &= \gamma xy - \lambda y \end{aligned}$$

We will consider the specific parameter set: $\alpha = 3, \beta = 1, \gamma = 0.5$, and $\lambda = 1$.

This system has two equilibria:

1. A trivial equilibrium at $(0, 0)$. For these parameters, this point is a saddle.
2. A co-existence equilibrium at $(\lambda/\gamma, \alpha/\beta)$, which for our parameters is at $(2, 3)$.

##### Linearization and Non-Hyperbolic Systems

To analyze the stability of the equilibrium at $(2, 3)$, we compute the Jacobian matrix at this point:  

$$J(2, 3) = \begin{pmatrix} 0 & -2 \\ 3/2 & 0 \end{pmatrix}$$  

The eigenvalues of this matrix are purely imaginary:  

$$\lambda_{1,2} = \pm i\sqrt{3} \approx \pm 1.73i$$ 

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

In linear systems, purely imaginary eigenvalues correspond to a center, around which trajectories form closed, neutrally stable orbits. However, we must be cautious when applying this intuition to a nonlinear system.


The equilibrium at $(2, 3)$ is non-hyperbolic because its eigenvalues have a real part equal to zero. For such systems, the Hartman-Grobman theorem does not necessarily hold. This theorem guarantees that the behavior of a nonlinear system near a hyperbolic equilibrium is qualitatively the same as its linearization. Since the theorem does not apply here, we cannot be certain of the system's behavior based on the first-order Taylor expansion (the linearization) alone; higher-order terms could fundamentally change the dynamics.


</div>

##### Conservative Systems and Neutrally Stable Orbits

In this particular case, the linearization does correctly predict the qualitative behavior. The system exhibits a dense set of closed, neutrally stable orbits around the equilibrium at $(2,3)$. This structure is called a nonlinear center.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The term "neutrally stable" means that if the system is on one closed orbit and is perturbed slightly, it does not return to the original orbit nor does it spiral away. Instead, it simply settles onto a new, nearby closed orbit.


This behavior is not typical for a randomly chosen nonlinear system. It arises here because the Lotka-Volterra system is a conservative system. This means it possesses a quantity that is conserved (remains constant) along any given trajectory. Such systems belong to a special class known as Hamiltonian systems. The existence of this conserved quantity is what enforces the structure of a continuous family of closed orbits.


</div>

#### Introduction to Limit Cycles

The nonlinear center seen in the Lotka-Volterra system is a special case. A more common and structurally robust form of periodic behavior is the limit cycle.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Limit Cycle)</span></p>

A **limit cycle** is a closed orbit that is isolated.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The key distinction is "isolated." Unlike the dense family of orbits in a nonlinear center, a limit cycle has no other closed orbits in its immediate vicinity. Trajectories near a stable limit cycle will spiral towards it, making it an attractor. Trajectories near an unstable limit cycle will spiral away from it. This property of being isolated makes limit cycles far more robust to perturbations than the neutrally stable orbits of a center.


</div>

### Limit Cycles and Nonlinear Oscillations

In our previous analysis, we focused on systems whose long-term behavior converges to a single point in state space—an equilibrium or fixed point. However, many systems in nature, from the firing of neurons to the orbits of planets, exhibit sustained, stable oscillations. These phenomena cannot be explained by fixed points alone. This chapter introduces a new type of attractor: the limit cycle, which provides the mathematical framework for understanding stable, nonlinear oscillations.

#### From Stable Points to Stable Orbits: An Example

Let us revisit the Wilson-Cowan model of excitatory and inhibitory neuron populations. The dynamics are described by a set of differential equations, and their equilibria are found at the intersections of the nullclines. By adjusting the system's parameters, we can fundamentally change its behavior.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


Consider a scenario where we adjust the parameters of the Wilson-Cowan model, specifically the $\beta$ parameter that governs the slope of the inhibitory nullcline's sigmoid function. If we make this slope sufficiently steep, a significant change occurs. The system's single fixed point can transform from a stable spiral, which draws all trajectories inward, into an unstable spiral.

When the fixed point is an unstable spiral, trajectories starting near it are pushed outwards. However, due to the bounded nature of the sigmoid functions in the equations, trajectories starting very far from the origin are pushed back inwards. This creates a "push-pull" dynamic: trajectories are repelled from the center and corralled from the periphery. The result is that the system's state does not fly to infinity or settle at a point; instead, it converges to a closed orbit that encircles the unstable fixed point. This isolated, closed orbit is what we call a limit cycle.

This behavior represents a nonlinear oscillation. If we were to plot one of the system's variables (e.g., the firing rate of the excitatory population) against time, we would observe a stable, repeating wave. Unlike a simple sine wave from a linear system, the shape of this oscillation can be quite complex, reflecting the underlying nonlinear dynamics.


</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name"></span></p>

Limit cycles are not merely a mathematical curiosity; they appear in numerous models across science and engineering.

* **Van der Pol Oscillator:** A classic 2D textbook example that describes oscillations in a vacuum tube circuit.
* **Duffing Oscillator:** Another famous 2D example used to illustrate nonlinear oscillations and chaotic behavior.

A compelling biological application is modeling short-term or working memory.

* **The Model:** Consider a task where a subject is shown a stimulus, which is then removed. After a delay, the subject must make a choice based on the remembered stimulus.
* **Neural Correlate:** During the delay period, populations of neurons in frontal brain regions are observed to jump to a high-firing state, often called an "upstate," and maintain this activity until it is no longer needed. They then "hop down" to a baseline state.
* **Dynamical Systems Interpretation:** This phenomenon can be modeled as a bistable system with two point attractors (a "downstate" and an "upstate"). The stimulus effectively "kicks" the system into the basin of attraction of the upstate, where it remains, thus "holding" the information online. This concept can be extended where different attractors correspond to different memories. The central idea is that information is maintained through the stable states of a dynamical system.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Generality of Dynamical Systems Theory)</span></p>

One might question if these simple models are "expressive enough" to capture the complexity of natural phenomena. A core strength of dynamical systems theory is its generality. It describes the evolution of a system over time purely from the perspective of its dynamics. Whether the underlying substrate is a set of neurons, a physical circuit, or an ecological population, the system is described by differential equations. The principles of attractors, repellors, and orbits provide a universal language for understanding the emergent behavior, regardless of the system's specific implementation.

</div>


#### Formal Definition of a Limit Cycle

We can now state the formal mathematical definition of a limit cycle.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Limit Cycle)</span></p>

A limit cycle, $\Gamma$, is an isolated closed orbit in the state space of a dynamical system.

A trajectory $\mathbf{x}(t) = \phi(t, \mathbf{x}_0)$ is a closed orbit if there exists a period $T > 0$ such that $\mathbf{x}(t+T) = \mathbf{x}(t)$ for all $t$. For any point $\mathbf{x}_0$ on the limit cycle $\Gamma$, its trajectory must satisfy:

$$\phi(t+T, \mathbf{x}_0) = \phi(t, \mathbf{x}_0)$$

where $T$ is the smallest positive number for which this relation holds. The term "isolated" means that there are no other closed orbits in the immediate neighborhood of $\Gamma$.

</div>

#### Stability of Limit Cycles

Just like fixed points, limit cycles can be classified by their stability. The stability determines whether trajectories near the cycle converge to it, diverge from it, or exhibit a combination of behaviors.

* Stable Limit Cycle: Trajectories starting in a neighborhood of the cycle, both inside and outside, converge to the limit cycle as $t \to \infty$. This limit cycle is an attractor. The Wilson-Cowan example described above features a stable limit cycle.
* Unstable Limit Cycle: Trajectories starting in a neighborhood of the cycle are repelled from it as $t \to \infty$. An unstable cycle acts as a repellor or a boundary between basins of attraction.
* Half-Stable (or Saddle) Cycle: Trajectories approach the cycle from one direction (e.g., from the inside) but are repelled from it in another direction (e.g., from the outside).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


The formal definition of stability for a limit cycle is analogous to the "Lyapunov-like" definition used for equilibria. For a cycle to be stable, any trajectory starting within a small neighborhood ($\epsilon$) of the cycle must remain within that neighborhood for all future time. For it to be asymptotically stable, the trajectory must also converge to the cycle as $t \to \infty$.

</div>

#### Methods for Detecting Limit Cycles

Proving the existence of a limit cycle is generally more complex than finding a fixed point (which only requires solving $\dot{\mathbf{x}} = 0$). We will introduce several powerful concepts for analyzing systems in the 2D plane.

##### The Poincaré-Bendixson Theorem and Trapping Regions

The Poincaré-Bendixson Theorem provides a powerful method for proving the existence of a limit cycle in a 2D system without explicitly solving the equations. The theorem's central idea relies on identifying a trapping region.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Trapping Region)</span></p>


A trapping region is a closed set in the phase space such that any trajectory that starts inside the region remains inside for all future time. Critically, the vector field on the boundary of this region must point inwards everywhere.


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


The theorem states that if a trapping region in a 2D system contains no fixed points, then it must contain at least one closed orbit. A more common application is when a trapping region contains a single fixed point that is unstable (like an unstable spiral or node).

Let's apply this to our Wilson-Cowan example:

1. We have a single fixed point which we have made into an unstable spiral. Any trajectory starting close to this point will spiral outwards. We can therefore draw a small boundary around the fixed point where the flow is always directed outwards.
2. The sigmoid functions in the Wilson-Cowan equations are bounded. This means that if you go far enough out in the state space, the linear $-x$ terms will dominate, and the flow will be directed back inwards towards the origin. We can therefore draw a large boundary far from the origin where the flow is always directed inwards.
3. The annular region between these two boundaries is a trapping region. Trajectories cannot escape inwards because they are repelled by the unstable fixed point, and they cannot escape outwards because they are pushed back by the system's dynamics at large distances.
4. Since this region contains no other fixed points, the Poincaré-Bendixson theorem guarantees that there must be a closed orbit—our limit cycle—within this annulus.

##### A Note on Unstable Cycles: Time Reversal

Unstable limit cycles are difficult to observe in simulations because trajectories are driven away from them. A simple but effective trick to locate them is to invert time. By making the substitution $t \to -t$, the system's differential equation $\dot{\mathbf{x}} = f(\mathbf{x})$ becomes $\dot{\mathbf{x}} = -f(\mathbf{x})$. This transformation reverses the flow of the vector field.

* An unstable limit cycle, from which trajectories diverged, becomes a stable limit cycle to which trajectories converge.
* This makes the formerly unstable cycle visible and easy to locate through numerical simulation.

##### The Poincaré Map

For a more systematic analysis of limit cycles, especially their stability, one can use the Poincaré map. The core idea is to convert the continuous flow around the cycle into a discrete map. By analyzing the properties of this map, one can deduce properties of the limit cycle itself. This topic will be revisited in greater detail later.


</div>


### Topological Tools: Index Theory

Another powerful tool for analyzing 2D vector fields is index theory. It uses topological properties to constrain the types and numbers of fixed points that can exist within a closed curve.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Winding Number)</span></p>


The core concept is the winding number. Imagine walking along a closed curve $C$. As you walk, consider a vector pointing from your position to a fixed point inside the curve. The winding number is the total number of full counter-clockwise rotations this vector makes during your complete circuit. Because the curve is closed, this must be an integer.


</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Index of a Fixed Point)</span></p>


The index of a fixed point is a property of the vector field surrounding it. To calculate it, we draw a small closed curve $C$ around the fixed point and traverse it once counter-clockwise. We observe the direction of the vectors of the vector field $\dot{\mathbf{x}}$ at each point on $C$. The index is the total number of counter-clockwise revolutions that the vector field itself makes.

Let's calculate the index for different types of fixed points:

* Stable Node: The vector field points inwards from all directions. As we move counter-clockwise around the curve $C$, the vector field also rotates counter-clockwise by one full turn.
  * Index = $+1$
* Unstable Node: The vector field points outwards in all directions. As we move counter-clockwise around $C$, the vector field again rotates counter-clockwise by one full turn.
  * Index = $+1$
* Saddle Point: The flow moves inwards along the stable manifold and outwards along the unstable manifold. Let's trace the vector's rotation as we move around the curve $C$:
  * As we start on the right and move up (counter-clockwise), the vector field points mostly down and left. As we cross the stable manifold, the vector points... (The analysis from the source context is incomplete here).


</div>


### Topological Properties of Orbits in the Plane

In the study of two-dimensional dynamical systems, we often encounter closed orbits known as limit cycles. The behavior of the vector field around these orbits and the equilibria they enclose are not arbitrary; they are governed by strict topological rules. This chapter introduces the concept of the index of a closed curve and a fundamental theorem that constrains the types of equilibria a limit cycle can contain.

#### The Index of a Closed Curve

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Index of a Closed Curve)</span></p>

The **index of a closed curve** (also known as the Poincaré index) quantifies the total rotation of the vector field as one traverses the curve. By convention, we traverse the closed curve in a counter-clockwise direction. The index is an integer value representing the number of net rotations the vectors make.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Imagine walking along a closed path on a field of arrows (the vector field). As you walk, you keep track of the direction the nearest arrow is pointing. The index tells you how many full circles the arrow's direction has turned by the time you return to your starting point. A positive index means the vector field rotates in the same direction as your path (counter-clockwise), while a negative index means it rotates in the opposite direction (clockwise).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name"></span></p>

* Index of $+1$: If we draw a closed curve around a stable or unstable equilibrium point (like a node or a focus), the vectors of the field will make one full counter-clockwise rotation as we move counter-clockwise along the curve. This configuration has an index of $+1$.
* Index of $-1$: Consider a saddle point. If we trace a closed curve around it in a counter-clockwise direction, the vector field will appear to rotate one full turn in the clockwise direction. By convention, this gives the curve an index of $-1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Additivity of Indices)</span></p>

For a closed curve, often called a Jordan curve, that encloses multiple regions, its total index is the sum of the indices of its subcurves.

</div>

#### A Theorem on Limit Cycles and Equilibria

The concept of the index leads to a powerful and restrictive theorem concerning limit cycles in the plane.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Poincaré–Bendixson Index Theorem (variant))</span></p>

For any limit cycle in a two-dimensional (2D) continuous dynamical system, the sum of the indices of all equilibria (fixed points) contained within the limit cycle must be exactly $+1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

This theorem provides a profound insight into the "topological relationships between different objects in state space." It acts as a strict rule for what configurations are possible within a planar system.

* A limit cycle can enclose a single stable or unstable equilibrium (node or focus), as these have an index of $+1$.
* A limit cycle cannot enclose only a single saddle point, because a saddle has an index of $-1$, which violates the theorem. This explains why we do not see stable orbits circulating around a solitary saddle point.
* A limit cycle can enclose more complex configurations, as long as the indices sum to $+1$. For example, a limit cycle can contain two stable points (each with index $+1$) and one saddle point (with index $-1$), because the total index is $(+1) + (+1) + (-1) = +1$.

</div>

#### Topological Constraints in System Reconstruction

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Implications)</span></p>

These topological relationships are not merely mathematical curiosities; they have significant practical importance, especially when attempting to reconstruct dynamical systems from observational data. In real-world scenarios, we may not have observed every component or state of a system. Knowledge of these underlying topological rules provides powerful constraints on what the complete system can look like.

If we observe an oscillation that appears to be a limit cycle, this theorem immediately tells us what kind of fixed-point structures must lie inside it. We know, for instance, that there cannot be just a saddle point inside. This a priori knowledge helps guide the modeling process and allows us to infer the existence of unobserved features. While this specific theorem is for 2D systems, similar topological principles and constraints exist in higher-dimensional systems as well, making them a crucial tool for understanding complex dynamics.

</div>

### Timescale Separation and Bifurcation Analysis

This chapter explores a powerful technique for analyzing complex dynamical systems that evolve on different timescales. By separating the system's variables into slow and fast categories, we can gain profound insights into its behavior, including phenomena like bursting oscillations. The primary tool for this analysis is the bifurcation graph.

#### The Method of Timescale Separation

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


Many complex systems in nature, from neural networks to climate models, feature variables that change at vastly different rates. The core idea behind timescale separation is to simplify the analysis by first understanding the behavior of the fast-moving variables while treating the slow-moving variables as temporarily constant.

The general technique involves the following steps:

1. Segregate the system variables into distinct slow and fast groups.
2. Analyze the fast subsystem: Consider the dynamics of the fast variables alone, treating the slow variables as fixed parameters.
3. Construct a bifurcation graph: Plot the stable and unstable solutions (or "objects") of the fast subsystem as a function of the slow "parameter."
4. Synthesize: Understand the behavior of the full system by envisioning it as a point that moves along this bifurcation graph as the slow variable evolves according to its own dynamics.

</div>

#### Introduction to Bifurcation Graphs

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bifurcation Graph)</span></p>


A bifurcation graph is a diagram that plots the state of a system's stable and unstable objects (such as fixed points and limit cycles) as a function of a chosen system parameter.


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


In the context of timescale separation, the "parameter" for our graph is one of the system's own slow variables. Let's consider a system with fast variables like voltage ($V$) and a slow variable, $h$. The bifurcation graph would be plotted with the slow variable h on the horizontal axis and a state variable of the fast system (e.g., $V$) on the vertical axis.

Conventionally, in these graphs:

* Stable objects (like stable fixed points or stable limit cycles) are drawn with solid lines.
* Unstable objects (like saddle points or unstable limit cycles) are drawn with dashed lines.

This visualization allows us to see, at a glance, how the fundamental nature of the fast subsystem changes as the slow variable $h$ evolves.


</div>

#### Case Study: Analysis of a System with a Slow Variable

Let us analyze a system where a slow variable, $h$, controls the dynamics of a faster subsystem (whose state can be described by variables such as voltage, $V$, and another variable, $n$).

##### Fixed Point Dynamics

By treating $h$ as a parameter, we can find the fixed points of the fast subsystem for each value of $h$. The analysis reveals three distinct regimes:

* For low values of $h$: The system has only one stable fixed point, located at a relatively high voltage.
* For high values of $h$: The system again has only one stable fixed point, but it is located at a much lower voltage.
* For an intermediate range of $h$: The system exhibits three fixed points.
  * The central fixed point is an unstable saddle point. In the bifurcation graph, this is represented by a dashed curve.
  * The two "outer" fixed points have different stability properties. One is a stable fixed point, while the other is an unstable spiral.

This appearance and disappearance of fixed points is associated with a type of bifurcation referred to in the lecture as a "set" bifurcation (likely a saddle-node bifurcation).

##### Limit Cycle Dynamics and Homoclinic Orbits

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The existence of an unstable spiral fixed point is significant. Often, such a point is surrounded by a stable limit cycle, representing a sustained oscillation in the fast subsystem. This is precisely what occurs in the intermediate range of h.

To represent this limit cycle on the two-dimensioal bifurcation graph, we plot its extremities.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Visualizing the Limit Cycle)</span></p>


The limit cycle (represented by a yellow curve in the lecture's diagram) is added to the bifurcation graph by plotting its maximum and minimum voltage values for each corresponding value of $h$. This creates a U-shaped curve enclosing the unstable spiral.

As the parameter h is increased within this intermediate range, the limit cycle grows in size. This expansion continues until a critical event occurs: the limit cycle becomes so large that it collides with the saddle point.


</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Homoclinic Orbit)</span></p>

A **homoclinic orbit** (or **homoclinic connection**) is a trajectory in a dynamical system that joins a saddle equilibrium point to itself. The event where the expanding limit cycle collides with the saddle point creates such an orbit. The bifurcation that occurs at this point is known as a homoclinic bifurcation.

</div>

#### The Complete Picture: Dynamics of the Full System

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The true power of this analysis comes when we remember that $h$ is not a static parameter but a slowly-evolving dynamical variable. The full system's state, therefore, travels back and forth along the structures depicted in the bifurcation graph.

This dynamic interplay leads to complex, emergent behavior:

1. The system might start at a stable fixed point (a quiescent state).
2. As the slow variable $h$ changes, it can drive the system into the region where the limit cycle exists.
3. The system state will "hop onto the cycle," initiating a burst of rapid oscillations.
4. As h continues to evolve, it may then drag the system into a region where the limit cycle no longer exists, causing the oscillations to cease and the system to fall back to a stable fixed point.

This continuous process of being driven back and forth between a stable fixed point and an oscillatory cycle explains the "bursting" behavior observed in many such systems.

</div>


## Lecture 5

### Cycles in Nonlinear Maps

#### Introduction to Discrete-Time Systems

In our study of dynamical systems, we have explored continuous-time systems described by differential equations. These are ideal for modeling phenomena where change is constant. However, many systems, particularly in fields like biology or epidemiology, are more naturally described in discrete time steps. For instance, we might count an infected population once per day or track a species' population generation by generation. For these scenarios, we use maps, which are recursive descriptions that define the state of a system at time $t+1$ based on its state at time $t$.

This chapter transitions our focus from continuous flows to discrete maps, exploring their unique behaviors, such as fixed points and cycles. We will begin with a foundational example that is famous for its complexity and its role in the history of chaos theory: the logistic map.

#### The Logistic Map: A Canonical Example

The logistic map is a simple, scalar (one-dimensional) map defined by a quadratic equation. It was famously analyzed by Robert May in a 1976 Nature paper, which highlighted how such a simple deterministic equation could produce extraordinarily complex, chaotic dynamics.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Logistic Map)</span></p>

The logistic map is a recursive function that maps a value $x_t$ to a new value $x_{t+1}$. It is defined by the equation:

$$x_{t+1} = \alpha x_t (1 - x_t)$$

Where:

* $x_t$ represents the state of the system at time step $t$.
* $\alpha$ is a single parameter that controls the behavior of the system.

For our analysis, we impose the following constraints:

* The initial condition $x_0$ is in the interval $[0, 1]$.
* The parameter $\alpha$ is in the interval $[0, 4]$.

Under these conditions, it can be shown that the state $x_t$ will remain bounded within the interval $[0, 1]$ for all subsequent time steps $t$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Visualizing the Map with a Return Plot)</span></p>

To understand the behavior of a map, we use a return plot, which graphs $x_{t+1}$ as a function of $x_t$. For the logistic map, this function is a parabola opening downwards. We also plot the line $x_{t+1} = x_t$, known as the bisector or identity line.

The intersections of the map's curve with the bisector are significant, as they represent points where the input equals the output—these are the fixed points of the system. We can trace the evolution of the system graphically using a "cobweb plot":

1. Start at an initial value $x_0$ on the horizontal axis.
2. Move vertically to the parabola to find the value of $x_1$.
3. Move horizontally from the parabola to the bisector. The corresponding point on the horizontal axis is now $x_1$.
4. Repeat the process: move vertically to the parabola to find $x_2$, horizontally to the bisector, and so on.

This graphical method provides a powerful intuition for whether the system converges to a fixed point, diverges, or enters a cycle.

</div>

#### Fixed Points and Their Stability

A fixed point is a state of the system that does not change over time. It is a point where, if the system starts there, it stays there.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fixed Point)</span></p>

A point $x^*$ is a fixed point of a map $f$ if it satisfies the condition: $x^* = f(x^*)$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Finding the Fixed Points of the Logistic Map)</span></p>

To find the fixed points of the logistic map, we set $x_{t+1} = x_t = x^*$ and solve the resulting equation:

$$x^* = \alpha x^* (1 - x^*)$$

Rearranging this gives us a quadratic equation: $\alpha (x^*)^2 + (1 - \alpha) x^* = 0$.

We can factor out $x^*$: $x^* (\alpha x^* + 1 - \alpha) = 0$.

This equation yields two solutions for the fixed points:

1. $x_1^* = 0$
2. $\alpha x^* + 1 - \alpha = 0 \implies \mathbf{x_2^* = \frac{\alpha - 1}{\alpha}}$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

From these solutions, we can immediately see two things:

* The fixed point at $x^*=0$ exists for all values of $\alpha$.
* The second fixed point, $x_2^* = \frac{\alpha-1}{\alpha}$, only exists within our interval of interest $[0, 1]$ if the parameter $\alpha$ is greater than or equal to $1$.

The return plot illustrates two scenarios. For a small $\alpha$, only one fixed point exists at $x^*=0$. For a larger $\alpha$, a second fixed point appears where the parabola intersects the bisector. This graphical analysis suggests that the stability of a fixed point is determined by the slope of the map at that point.

</div>

#### Formal Stability Analysis via Linearization

To formalize our intuition, we analyze the behavior of a small perturbation around a fixed point, a technique analogous to the one we used for differential equations.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Derivation of the Stability Condition)</span></p>

Let $x^*$ be a fixed point of the map $x_{t+1} = f(x_t)$. Consider a small perturbation $\epsilon_t$ from this fixed point at time $t$: $x_t = x^* + \epsilon_t$. The state at the next time step, $x_{t+1}$, will be: $x_{t+1} = x^* + \epsilon_{t+1} = f(x^* + \epsilon_t)$.

Assuming $\epsilon_t$ is small, we can perform a Taylor expansion of $f(x^* + \epsilon_t)$ around $x^*$:

$$x^* + \epsilon_{t+1} \approx f(x^*) + f'(x^*) \epsilon_t + O(\epsilon_t^2)$$

By definition, $f(x^*) = x^*$. We can therefore cancel the $x^*$ terms on both sides. Ignoring higher-order terms for our linear approximation, we get a recursive map for the perturbation:

$$\epsilon_{t+1} \approx f'(x^*) \epsilon_t$$

This is a linear map describing the evolution of the perturbation. The perturbation $\epsilon_t$ will decay to zero (i.e., the fixed point is stable) if the magnitude of the multiplier is less than one. Conversely, it will grow if the magnitude is greater than one.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Stability of Fixed Points for 1D Maps)</span></p>

Let $x^*$ be a fixed point of a nonlinear map $f(x)$. The stability of $x^*$ is determined by the derivative of the map evaluated at the fixed point, $f'(x^*)$:

* If $\|f'(x^*)\| < 1$, the fixed point is locally stable.
* If $\|f'(x^*)\| > 1$, the fixed point is locally unstable.
* If $\|f'(x^*)\| = 1$, the stability cannot be determined by this linear analysis. This is a non-hyperbolic case, and higher-order terms of the Taylor expansion must be considered.

</div>

#### Generalization to Higher-Dimensional Maps

This stability criterion extends naturally to multivariate (higher-dimensional) maps.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(N-Dimensional Maps and the Jacobian)</span></p>

Consider a system in $m$ dimensions described by a map $\mathbf{x}_{t+1} = \mathbf{F}(\mathbf{x}_t)$, where $\mathbf{x}_t$ is a vector in $\mathbb{R}^m$. The stability analysis is analogous, but the scalar derivative is replaced by the Jacobian matrix, $J$.

The Jacobian matrix is the matrix of all first-order partial derivatives of the vector-valued function $\mathbf{F}$:

$$J = \begin{pmatrix} \frac{\partial F_1}{\partial x_1} & \cdots & \frac{\partial F_1}{\partial x_m} \\ \vdots & \ddots & \vdots \\ \frac{\partial F_m}{\partial x_1} & \cdots & \frac{\partial F_m}{\partial x_m} \end{pmatrix}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Stability for N-D Maps)</span></p>

Let $\mathbf{x}^*$ be a fixed point of the map $\mathbf{F}(\mathbf{x})$. The stability of $\mathbf{x}^*$ is determined by the eigenvalues of the Jacobian matrix evaluated at the fixed point, $J(\mathbf{x}^*)$.

* The fixed point $\mathbf{x}^*$ is stable if the maximum absolute value (or modulus, for complex eigenvalues) of all eigenvalues of $J(\mathbf{x}^*)$ is less than $1$: $\max_i |\lambda_i| < 1$.
* The fixed point $\mathbf{x}^*$ is unstable if the maximum absolute value of any eigenvalue of $J(\mathbf{x}^*)$ is greater than $1$: $\max_i |\lambda_i| > 1$.
* If the maximum absolute value of the eigenvalues is exactly equal to $1$ (i.e., the largest eigenvalue lies on the unit circle in the complex plane), the system is non-hyperbolic, and a linear stability analysis is inconclusive.

</div>

#### The Emergence of K-Cycles

What happens when all fixed points in a bounded system become unstable? The trajectory cannot settle into a fixed point, but it also cannot escape to infinity. The system must find another form of stable, persistent behavior. In maps, this often leads to the emergence of cycles.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Path to a 2-Cycle)</span></p>

Consider the logistic map for $\alpha > 3$. At these parameter values, the slopes at both fixed points ($x^*=0$ and $x^*=(\alpha-1)/\alpha$) are greater than $1$ in absolute value. This means both fixed points are unstable.

Since we know the system is confined to the interval $[0, 1]$, the trajectory must go somewhere else. This "somewhere else" is often a cycle, where the system visits a finite sequence of points repeatedly.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(K-Cycle)</span></p>

A K-cycle is a periodic trajectory where the system iterates through $K$ distinct points. A 2-cycle, for example, is a pair of points $\lbrace x_a, x_b \rbrace$ such that: $f(x_a) = x_b$ and $f(x_b) = x_a$. The system perpetually jumps back and forth between these two points.

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
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

When analyzing discrete maps, we often encounter cycles, where the system iterates between a set of distinct points. A two-cycle, for instance, involves iterating between two different points. If we start at one point, a single application of the map takes us to the second point, and the next application takes us back to the first.

This observation leads to a powerful insight: a point on a two-cycle returns to its original position after exactly two applications of the map. Therefore, any point belonging to a two-cycle of a map $f$ must be a fixed point of the twice-iterated map, denoted as $f^2(x) = f(f(x))$. This reframes the problem of finding cycles into the more familiar problem of finding fixed points, albeit for a more complex function.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Logistic Map)</span></p>

Let's consider the logistic map, which is a second-order polynomial (a "square map"). If we construct the twice-iterated map, $f^2(x) = f(f(x))$, by substituting the logistic equation into itself, the resulting function is a fourth-order polynomial.

The explicit form of this map is given by: $f^2(x^*) = - \alpha^3 (x^*)^4 + 2 \alpha^3 (x^*)^3 - (\alpha^2 + \alpha^3) (x^*)^2$. The key takeaway is that an iterated map yields another function, and the fixed points of this new function correspond to the cycles of the original map.

</div>

#### Stability of Cycles

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Since we can treat the points of a cycle as fixed points of an iterated map, we can determine the stability of the cycle by analyzing the stability of these corresponding fixed points. The method is the same as for simple fixed points: we check the slope of the function at the fixed point.

For a two-cycle of the map $f$, its stability is determined by the derivative of the twice-iterated map, $f^2(x)$, at the points of the cycle.

The plot of $f^2(x)$ for the logistic map at $\alpha = 3.3$ reveals four fixed points. Two of these are the original, now unstable, fixed points of $f(x)$. The other two are the points that constitute the stable two-cycle.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Stability of a Cycle)</span></p>

The stability of a k-cycle can be determined by checking the slope of the k-times iterated function, $f^k(x)$, at any point $x_i^*$ on the cycle. The cycle is stable if the absolute value of this slope is less than one:

$$\left| \frac{d}{dx} f^k(x) \right|_{x=x_i^*} < 1$$

The cycle is unstable if this value is greater than one.

</div>

#### Generalization to k-Cycles

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(k-Cycle)</span></p>

For a continuous map $f$, a k-cycle is a set of $k$ distinct points, $\lbrace x_1^*, x_2^*, ..., x_k^* \rbrace$, which are visited sequentially by iteration of $f$. This implies two critical conditions:

1. **Fixed Point of the Iterated Map:** Each point $x_i^*$ in the set (for $i = 1, ..., k$) is a fixed point of the k-times iterated map: $x_i^* = f^k(x_i^*)$.
2. **Minimality and Distinctness:** To be a true k-cycle, two additional constraints must be met:
   * $k$ must be the smallest integer for which the fixed-point condition holds. This ensures that a two-cycle is not misidentified as a four-cycle, for example.
   * All points in the set must be distinct: $x_i^* \neq x_j^*$ for all $i \neq j$. This prevents a lower-order cycle (like a fixed point where $x_1^* = x_2^*$) from being classified as a higher-order cycle.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

This framework provides a significant advantage. While the original map $f$ might be complex and its iterated version $f^k$ even more so, we at least have a closed-form expression. This closed form gives us direct, analytical access to the stability of cycles. This is a powerful tool that is not generally available for analyzing the stability of limit cycles in continuous-time systems described by differential equations.

</div>

### The Phase Description of Oscillators

In the study of dynamical systems, oscillators represent a fundamental class of behaviors characterized by periodic motion, often visualized as a closed orbit or limit cycle in the state space. To analyze these systems, particularly when they interact, it is incredibly useful to reduce their complex dynamics to a single, essential variable: the phase.

#### The Phase Variable

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Phase Variable)</span></p>

The phase of an oscillator describes its position along its limit cycle. It is represented by a phase variable, typically denoted as $\theta$. A single full iteration of the oscillator corresponds to the phase variable completing a full cycle. By convention, the phase is often defined to evolve in the interval $[0, 2\pi]$, though other intervals such as $[0, 1]$ are also used.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The central idea of this approach is to shift our focus from the full state-space variables of the oscillator to just its phase. By doing this, we can formulate a new, often simpler, differential equation that describes the evolution of the phase itself: $\dot{\theta} = f(\theta)$. This simplification allows us to capture the essential timing and rhythm of the oscillator, which is paramount when studying phenomena like synchronization.

</div>

#### A Simple Oscillator: Constant Angular Velocity

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Constant Angular Velocity)</span></p>

Consider the simplest case of an oscillator: one that traverses its limit cycle at a constant speed. This means its phase variable increases at a constant rate.

* **Dynamics:** The differential equation for the phase is linear: $\dot{\theta} = \Omega$, where $\Omega$ is the constant angular velocity.
* **Solution:** The explicit equation for the phase at time $t$ is: $\theta(t) = (\Omega t + \theta_0) \pmod{2\pi}$, where $\theta_0$ is the initial phase at $t=0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

While this constant-speed model is a useful starting point, it is crucial to remember that it is not generally true for non-linear oscillators. In most realistic systems, an oscillator will speed up and slow down as it moves through different parts of its cycle.

</div>

#### Calculating the Oscillation Period

If we have the differential equation for the phase, $\dot{\theta} = f(\theta)$, we can derive a formula to calculate the temporal period of one full oscillation, $T_{osc}$.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Derivation of the Oscillation Period)</span></p>

The period $T_{osc}$ is, by definition, the time it takes to complete one cycle. We can express this with a simple integral: $T_{osc} = \int_0^{T_{osc}} dt$. To connect this to the phase variable, we perform a change of variables from time $t$ to phase $\theta$. As time progresses from $0$ to $T_{osc}$, the phase progresses from $0$ to $2\pi$. We can introduce $d\theta$ into the integral:

$$T_{osc} = \int_0^{2\pi} \frac{dt}{d\theta} d\theta$$

We know the differential equation for the phase is $\frac{d\theta}{dt} = f(\theta)$. Therefore, its inverse is $\frac{dt}{d\theta} = \frac{1}{f(\theta)}$. Substituting this into the integral gives the final formula:

$$T_{osc} = \int_0^{2\pi} \frac{1}{f(\theta)} d\theta$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

This formula provides a direct recipe for calculating the oscillation period for any one-dimensional phase oscillator, provided we know the function $f(\theta)$ that governs its dynamics.

</div>

### Systems of Uncoupled Oscillators

We now extend our analysis from a single oscillator to a system of multiple oscillators. To build our understanding, we first consider the case where two oscillators evolve independently, without any coupling between them.

#### The State-Space Torus

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Two Runners on a Circular Track)</span></p>

Imagine two runners on a circular track, each running at their own constant, but different, velocity. Let the first runner have velocity $\omega_1$ and the second have velocity $\omega_2$.

* Runner 1 phase dynamics: $\dot{\theta}_1 = \omega_1$
* Runner 2 phase dynamics: $\dot{\theta}_2 = \omega_2$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Since each oscillator is described by a variable on a circle (from $0$ to $2\pi$), the combined state space of two such oscillators, $(\theta_1, \theta_2)$, can be visualized as a torus (a donut shape). One phase variable, say $\theta_1$, represents motion around the main radius of the torus, while the other, $\theta_2$, represents motion around the circular cross-section.

A trajectory in this two-dimensional state space represents the simultaneous evolution of both phases. A central question arises: Under what conditions will this trajectory eventually return to its starting point, forming a closed orbit on the torus?

</div>

#### Closed Orbits and Rational Frequency Ratios

A trajectory on the state-space torus will be a closed orbit if and only if the frequencies of the two oscillators are commensurate.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Commensurate Frequencies)</span></p>

Two frequencies, $\omega_1$ and $\omega_2$, are commensurate if their ratio is a rational number. That is, there exist two integers, $p$ and $q$, such that: $\frac{\omega_1}{\omega_2} = \frac{p}{q}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

This condition has a very clear physical meaning. If the ratio of frequencies is $\frac{p}{q}$, it means that the first oscillator completes exactly $p$ cycles in the same amount of time that the second oscillator completes exactly $q$ cycles. After this time has elapsed, both oscillators will have returned to their initial phases simultaneously, thus closing the trajectory on the torus.

For instance, if $\omega_1$ is three times faster than $\omega_2$ ($\frac{\omega_1}{\omega_2} = \frac{3}{1}$), the first runner will complete three laps around the track in the exact time it takes the second runner to complete one. At that moment, they are both back at their starting positions.

</div>

#### Quasi-periodicity and Irrational Frequency Ratios

If the condition for a closed orbit is not met, a fascinating and more complex behavior emerges.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Quasi-periodicity)</span></p>

If the ratio of the oscillator frequencies $\frac{\omega_1}{\omega_2}$ is an irrational number, the trajectory on the torus will never close. Instead, it will wind around indefinitely, eventually passing arbitrarily close to every point on the surface of the torus. This motion is called quasi-periodic.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Quasi-periodic motion is an interesting middle ground between simple periodic behavior (like a limit cycle) and the more complex behavior of chaos. The system is "almost" periodic, but because the frequencies never align perfectly, the trajectory never repeats. Over time, the path of the system state will densely fill the entire surface of the torus. This is a unique property that arises in systems of two or more oscillators.

</div>

### Coupled Oscillators and Synchronization

Having explored uncoupled systems, we now introduce coupling, allowing the oscillators to influence one another. This is where the most interesting phenomena, such as synchronization, arise.

#### Phase-Dependent Coupling

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

In most non-linear oscillators, the effect of an external perturbation depends critically on the phase at which the perturbation is applied.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Biological Neuron)</span></p>

Consider a biological neuron that fires action potentials periodically. If we inject a small pulse of electrical current into the neuron, the effect will depend on when we inject it.

* If the neuron has just fired and is in its refractory period, the current may have very little effect.
* If the neuron is close to its firing threshold, the same pulse of current could be enough to trigger an action potential immediately, thereby advancing the phase of the oscillator.
* At other times, the pulse could delay the next action potential.

When two such neurons are coupled, they perturb each other back and forth. The first neuron fires, sending a signal that advances or delays the second neuron. The second neuron then fires, sending a signal back that advances or delays the first. It is through this dynamic interplay that the oscillators may eventually fall into a synchronized rhythm.

</div>

#### A Model for Two Coupled Oscillators

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Coupled Phase Oscillator Model)</span></p>

A common model for two coupled oscillators is given by the following system of differential equations:

$$\dot{\theta}_1 = \omega_1 + A \sin(\theta_1 - \theta_2)$$

$$\dot{\theta}_2 = \omega_2 + A \sin(\theta_2 - \theta_1)$$

Here, $A$ is the coupling strength, which determines how strongly the oscillators influence each other. The function $\sin(\cdot)$ is chosen as a simple periodic function to model the phase-dependent interaction, but other functions could be used.

</div>

#### The Phase Difference Equation

To analyze whether these oscillators will synchronize, the most effective technique is to study the evolution of their phase difference.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Phase Difference)</span></p>

The phase difference, $\phi$, between the two oscillators is defined as: $\phi = \theta_1 - \theta_2$. If the oscillators synchronize perfectly (zero phase locking), this difference will be constant at $\phi=0$. If they lock at a different, but still constant, phase relationship, $\phi$ will be a non-zero constant. Therefore, synchronization corresponds to the phase difference $\phi$ reaching a stable fixed point.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Derivation of the Phase Difference Equation)</span></p>

We can derive a differential equation for $\phi$ by differentiating its definition with respect to time: $\dot{\phi} = \dot{\theta}_1 - \dot{\theta}_2$. Now, substitute the model equations:

$$\dot{\phi} = \left( \omega_1 + A \sin(\theta_1 - \theta_2) \right) - \left( \omega_2 + A \sin(\theta_2 - \theta_1) \right)$$

Group the terms:

$$\dot{\phi} = (\omega_1 - \omega_2) + A \sin(\theta_1 - \theta_2) - A \sin(\theta_2 - \theta_1)$$

Using the trigonometric identity that sine is an odd function, $\sin(-x) = -\sin(x)$, we have $\sin(\theta_2 - \theta_1) = -\sin(\theta_1 - \theta_2)$. Substituting and simplifying:

$$\dot{\phi} = (\omega_1 - \omega_2) + 2A \sin(\phi)$$

</div>

#### Phase Locking and Fixed Point Analysis

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The concept of synchronization or phase locking is now transformed into a familiar problem: finding the fixed points of the scalar differential equation for $\phi$. A fixed point occurs where $\dot{\phi} = 0$, which means the phase difference stops changing and the oscillators are locked.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Identical Intrinsic Frequencies)</span></p>

Let's analyze the simplest case, where the two oscillators have the same intrinsic frequency, $\omega_1 = \omega_2$. The phase difference equation simplifies to: $\dot{\phi} = 2A \sin(\phi)$.

* **Fixed Points:** The fixed points are where $\dot{\phi} = 0$, which occurs when $\sin(\phi) = 0$. This gives fixed points at $\phi=0, \pi, 2\pi, \dots$.
* **Stability Analysis:**
  * For a value of $\phi$ slightly greater than $0$, $\dot{\phi}$ is positive, causing $\phi$ to increase and move away from $0$.
  * For a value of $\phi$ slightly less than $\pi$ (but greater than $0$), $\dot{\phi}$ is positive, causing $\phi$ to increase towards $\pi$. For a value of $\phi$ slightly greater than $\pi$, $\dot{\phi}$ is negative, causing $\phi$ to decrease towards $\pi$.
  * The state at the center is a stable fixed point. This is the point of phase locking.
  * The system dynamics cause the phase difference $\phi$ to be repelled from the unstable fixed point and attracted to the stable one.

The analysis can be extended by considering what happens as the difference in intrinsic frequencies, $\omega_1 - \omega_2$, is increased. This corresponds to vertically shifting the sine curve of the $\dot{\phi}$ vs. $\phi$ graph.

</div>

### Synchronization and Phase Locking in Coupled Oscillators

This section explores the phenomenon of synchronization, where coupled oscillating systems adjust their rhythms to lock into a common pattern. We investigate the conditions under which this occurs and introduce a powerful graphical tool for mapping these behaviors.

#### The Concept of Phase Locking

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Phase Locking)</span></p>

Two or more oscillators are said to be synchronized or phase-locked when the difference between their phases, $\phi$, becomes constant over time. Mathematically, this corresponds to a stable fixed point of the phase difference dynamics. If $\dot{\phi} = 0$ for some phase difference $\phi^*$, the oscillators have achieved phase locking.

</div>

#### Conditions for Synchronization: Frequency Difference and Coupling Strength

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Interplay of Intrinsic Frequency and Coupling)</span></p>

The ability of two oscillators to synchronize is a result of the competition between their intrinsic properties and the strength of the interaction connecting them.

Consider two oscillators with intrinsic frequencies $\omega_1$ and $\omega_2$. The dynamics of their phase difference, $\phi$, can often be described by an equation of the form:

$$\dot{\phi} = (\omega_1 - \omega_2) + a \cdot g(\phi)$$

where $g(\phi)$ is a periodic coupling function (e.g., a sigmoid or sine function) and $a$ is the amplitude of the coupling.

* **The Role of Frequency Difference ($\omega_1 - \omega_2$):** This term acts as a constant vertical shift to the coupling function $g(\phi)$. If the difference is zero, synchronization can occur even with zero coupling ($a=0$). However, as the difference $|\omega_1 - \omega_2|$ grows, this vertical shift becomes larger.
* **The Role of Coupling Amplitude ($a$):** This term scales the magnitude of the coupling function. Increasing $a$ makes the peaks and troughs of $a \cdot g(\phi)$ more pronounced.

For phase locking to occur, the graph of $\dot{\phi}$ must intersect the horizontal axis ($\dot{\phi}=0$). If the frequency difference $|\omega_1 - \omega_2|$ becomes too large for a given coupling strength $a$, the entire curve of $\dot{\phi}$ may be shifted above or below the zero-axis. However, we can often compensate for a large difference in intrinsic frequencies by ramping up the amplitude of coupling, $a$.

In summary, for any given coupling strength, there is a limited range of frequency differences within which oscillators can phase lock. Increasing the coupling strength broadens this range.

</div>

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
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(P:Q Phase Locking)</span></p>

We can generalize the concept of synchronization beyond a simple 1:1 relationship. We speak of P:Q phase locking if the difference between the unbounded phase variables, $\theta_1(t)$ and $\theta_2(t)$, remains bounded when weighted by integers $p$ and $q$.

Specifically, P:Q phase locking occurs if the quantity:

$$p \cdot \theta_1(t) - q \cdot \theta_2(t)$$

remains bounded over time. This describes a state where one oscillator completes $p$ cycles in the same amount of time that the other oscillator completes $q$ cycles. The case of simple synchronization discussed previously is 1:1 phase locking.

</div>

#### Arnold Tongues: Mapping Synchronization Regions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Arnold Tongues)</span></p>

Arnold tongues are regions in a parameter space that depict where phase locking of a specific P:Q ratio occurs. Typically, this space is plotted with the difference in intrinsic frequencies ($\omega_1 - \omega_2$) on one axis and the coupling amplitude ($a$) on the other. Each "tongue" represents a combination of parameters for which the system will synchronize in a particular mode (e.g., 1:1, 1:2, 2:1).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Structure of Arnold Tongues)</span></p>

The plot of Arnold tongues provides a powerful map of a coupled oscillator system's behavior.

* **The 1:1 Tongue:** The most prominent region is typically the 1:1 coupling (or exact synchrony) tongue. It is centered at a frequency difference of zero ($\omega_1 = \omega_2$). At this central point, synchronization can occur with zero coupling ($a=0$). As the coupling strength $a$ increases, the 1:1 tongue grows broader, signifying that a stronger coupling can enforce synchronization across a larger range of intrinsic frequency differences.
* **Higher-Order Tongues:** For many systems, smaller, higher-order tongues corresponding to P:Q couplings like 1:2, 2:1, 1:3, 3:1, etc., appear as side-branches. These tongues are typically narrower than the main 1:1 tongue, indicating that these more complex locking modes occur over a smaller range of parameters. The existence and prominence of these higher-order tongues depend on the specific properties of the system and its interaction function.
* **Intersection and Complex Dynamics:** As the coupling amplitude $a$ is increased, the tongues broaden. At a certain point, these tongues may begin to intersect. The regions where Arnold tongues overlap correspond to parameter values where complex and interesting dynamics can emerge, which are a subject of more advanced study.

</div>

### Numerical Integration of Ordinary Differential Equations

#### The Need for Numerical Methods

Most ordinary differential equations arising in practical applications cannot be solved analytically. We therefore rely on numerical methods to approximate solutions.

#### The Explicit Euler Method

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Explicit Euler Method)</span></p>

The Explicit Euler Method is the simplest numerical scheme for solving an initial value problem $\dot{x} = f(x, t)$. Given a step size $\Delta t$, the approximation at the next time step is:

$$x_{n+1} = x_n + \Delta t \cdot f(x_n, t_n)$$

</div>

#### The Runge-Kutta Family of Methods

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The Euler method, while intuitive, can be inaccurate for large step sizes. The Runge-Kutta family of methods provides higher-order approximations by evaluating the vector field at multiple intermediate points within each time step, achieving better accuracy without requiring excessively small step sizes.

</div>

#### Practical Considerations: Explicit vs. Implicit Solvers for Stiff Systems

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

For stiff systems—those with dynamics occurring on vastly different time scales—explicit methods like Euler or standard Runge-Kutta can become unstable unless the step size is made extremely small. In such cases, implicit solvers, which evaluate the vector field at the future time step, provide better stability properties at the cost of requiring the solution of an algebraic equation at each step.

</div>
## Lecture 6

### Dynamical Systems with Special Functionals

In the study of dynamical systems, certain classes of systems exhibit unique behaviors due to the existence of special functions, or functionals, defined on their state space. These functionals, such as potentials, energy functions, or Hamiltonians, impose strong constraints on the system's dynamics. Understanding these functions allows us to predict the qualitative behavior of trajectories and the nature of equilibrium points without solving the differential equations explicitly. This section explores two fundamental types of such systems: Hamiltonian systems, often associated with conservation laws in physics, and Gradient systems, which describe processes moving towards local minima.

#### Hamiltonian Systems

Hamiltonian systems are a cornerstone of classical mechanics and dynamical systems theory. They are characterized by a conserved quantity, the Hamiltonian, which often corresponds to the total energy of the system. This conservation property leads to highly structured and constrained dynamics.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hamiltonian Function)</span></p>

Let the state space $E$ be an open set in $\mathbb{R}^{2m}$. A function $H: E \to \mathbb{R}$ is called a Hamiltonian function if it is twice differentiable ($C^2$) on $E$. The state variables are typically written as $(x, y)$ where $x, y \in \mathbb{R}^m$.

A dynamical system described by the differential equations:

$$\begin{aligned}
\dot{x} &= \frac{\partial H}{\partial y} \\
\dot{y} &= -\frac{\partial H}{\partial x}
\end{aligned}$$

is defined as a **Hamiltonian system**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Equilibria of Hamiltonian Systems)</span></p>

Any non-degenerate equilibrium point of a Hamiltonian system is either a saddle or a center. Specifically, if the equilibrium point corresponds to:

* A saddle point of the Hamiltonian function $H(x,y)$, then the equilibrium is a saddle.
* A local maximum or minimum of the Hamiltonian function $H(x,y)$, then the equilibrium is a center (a non-linear center).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Significance of the Theorem)</span></p>

This theorem is a powerful tool for classifying equilibrium points in non-linear systems. Typically, determining if an equilibrium is a true non-linear center requires analyzing higher-order terms of the system's Taylor expansion. However, if we can demonstrate that a system possesses a Hamiltonian function, we can classify its equilibria simply by analyzing the local extrema of $H$. This provides a direct and elegant method for proving the existence of non-linear centers, which are characterized by a dense set of closed orbits in their vicinity.

Systems that possess a Hamiltonian are often called conservative systems. The value of the Hamiltonian, $H(x,y)$, remains constant along any trajectory of the system, acting as a constant of motion.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Lotka-Volterra System)</span></p>

Let us revisit the Lotka-Volterra model for predator-prey interaction, which we can demonstrate is a Hamiltonian system.

The system is defined by the equations:

$$\begin{aligned} \dot{x} &= \alpha x - \beta xy \\ \dot{y} &= \gamma xy - \lambda y \end{aligned}$$

where $x$ represents the prey population and $y$ represents the predator population. All parameters $\alpha, \beta, \gamma, \lambda$ are positive constants.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Deriving the Hamiltonian for Lotka-Volterra)</span></p>

To show this system is Hamiltonian, we must construct a function $H(x,y)$ that is constant along the system's trajectories and satisfies the necessary conditions.

1. **Combine the Differential Equations:** We can eliminate the time variable $dt$ by dividing the two equations:

$$\frac{dx}{dy} = \frac{\dot{x}}{\dot{y}} = \frac{\alpha x - \beta xy}{\gamma xy - \lambda y} = \frac{x(\alpha - \beta y)}{y(\gamma x - \lambda)}$$

2. **Separate Variables:** We rearrange the equation to group terms involving $x$ and $y$ on opposite sides.

$$\frac{\gamma x - \lambda}{x} dx = \frac{\alpha - \beta y}{y} dy$$

3. **Integrate Both Sides:**

$$\int \left(\gamma - \frac{\lambda}{x}\right) dx = \int \left(\frac{\alpha}{y} - \beta\right) dy$$

This yields: $\gamma x - \lambda \ln(x) = \alpha \ln(y) - \beta y + C$, where $C$ is a constant of integration.

4. **Define the Hamiltonian:** By rearranging the terms, we define:

$$H(x,y) = \alpha \ln(y) - \beta y - \gamma x + \lambda \ln(x)$$

Since the logarithm is only defined for positive arguments, this Hamiltonian is valid in the positive quadrant ($x > 0, y > 0$).

5. **Verification with Auxiliary Variables:** We introduce auxiliary variables: let $p = \ln(x)$ and $q = \ln(y)$, so $x = e^p$ and $y = e^q$. Rewriting: $H(p, q) = \alpha q - \beta e^q + \lambda p - \gamma e^p$.

6. **First condition for $\dot{p}$:** Using the chain rule, $\dot{p} = \frac{1}{x}\dot{x} = \alpha - \beta y = \alpha - \beta e^q$. And $\frac{\partial H}{\partial q} = \alpha - \beta e^q$. The condition $\dot{p} = \frac{\partial H}{\partial q}$ is satisfied.

7. **Second condition for $\dot{q}$:** Similarly, $\dot{q} = \frac{1}{y}\dot{y} = \gamma x - \lambda = \gamma e^p - \lambda$. And $-\frac{\partial H}{\partial p} = \gamma e^p - \lambda$. The condition $\dot{q} = -\frac{\partial H}{\partial p}$ is also satisfied.

We have successfully shown that the Lotka-Volterra system is a Hamiltonian system in the positive quadrant. Consequently, its equilibrium points must be either saddles or centers, which aligns with numerical simulations that show a dense set of closed orbits.

</div>

#### Gradient Systems

Another important class of systems are gradient systems, where the dynamics are governed by the gradient of a potential function. These systems model phenomena where a state moves to minimize a certain quantity, such as energy.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Potential Function and Gradient System)</span></p>

Let the state space $E$ be an open set in $\mathbb{R}^m$. Let $V: E \to \mathbb{R}$ be a twice differentiable ($C^2$) function on $E$.

A dynamical system is a gradient system if its vector field is given by the negative gradient of a potential function $V(x)$:

$$\dot{x} = -\frac{\partial V}{\partial x}$$

This can also be written using the gradient operator as $\dot{x} = -\nabla V(x)$. The function $V(x)$ is also referred to as a gradient field.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Impossibility of Closed Orbits in Gradient Systems)</span></p>

For any system with a potential function $V(x)$, closed orbits are impossible unless the orbit is a single equilibrium point.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A Powerful Restrictive Result)</span></p>

This theorem provides a very strong constraint on the possible behaviors of a gradient system. It implies that such systems cannot have centers or limit cycles. If one can construct a potential function for a given system, it immediately rules out any periodic or oscillatory behavior. All trajectories in a gradient system must eventually approach an equilibrium point or diverge to infinity.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Impossibility of Closed Orbits)</span></p>

Let us consider the change in the potential function, $\Delta V$, along an orbit of the system over a time interval. This change is given by the integral of the time derivative of $V$:

$$\Delta V = \int \frac{dV}{dt} dt$$

Using the chain rule: $\frac{dV}{dt} = \frac{\partial V}{\partial x} \frac{dx}{dt} = (\nabla V) \cdot \dot{x}$.

By the definition of a potential function, $\nabla V = -\dot{x}$. Substituting:

$$\frac{dV}{dt} = (-\dot{x}) \cdot \dot{x} = - \|\dot{x}\|^2$$

Therefore: $\Delta V = -\int \|\dot{x}\|^2 dt$. The term $\|\dot{x}\|^2$ is always non-negative, so $\Delta V \le 0$. Equality holds only if $\dot{x} = 0$ for the entire trajectory, i.e., at an equilibrium point.

For a closed orbit, the system must return to its starting point, meaning $\Delta V = 0$. But $\Delta V = 0$ requires $\dot{x} = 0$ everywhere on the orbit, which contradicts the definition of a closed orbit (which involves movement). Therefore, no closed orbits can exist in a gradient system.

</div>

#### Equilibrium Points in Gradient Systems

The behavior of a gradient system can be intuitively understood by visualizing the potential function $V(x)$ as a landscape of hills and valleys. The dynamics, $\dot{x} = -\nabla V(x)$, describe a state "rolling downhill" towards lower potential values.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

* A stable equilibrium (a stable node) corresponds to a local minimum of the potential function $V$. Any small perturbation from the bottom of a "valley" will result in the system returning to that minimum.
* An unstable equilibrium corresponds to a local maximum of the potential function $V$. Any small perturbation from the top of a "hill" will cause the system to roll away, typically towards a nearby minimum.
* A saddle point of the potential function $V$ corresponds to a saddle equilibrium of the dynamical system.

This intuitive picture is central to understanding how gradient systems store information. The minima of the potential function act as point attractors, representing stable memory states.

</div>

#### Applications to Continuous-Time Neural Networks

The properties of gradient systems are essential to certain classes of neural networks, most famously the Hopfield network. John Hopfield defined a set of differential equations for a neural network that possess a potential function, which he termed an "energy function." This construction guarantees that the network's dynamics will always converge to one of several stable equilibrium points, which represent stored memory states.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(General Class of Continuous-Time Networks)</span></p>

A general class of continuous-time neural networks, which includes Hopfield networks and the Wilson-Cowan equations, can be written in the form:

$$\tau \dot{x} = -x + W \sigma(x) + I$$

Where:

* $x$ is the vector of neural activities.
* $\tau$ is a time constant.
* $W$ is the weight matrix describing the connectivity between units.
* $\sigma(\cdot)$ is a non-linear transfer function (often a sigmoid function like $\sigma(z) = \frac{1}{1 + e^{-z}}$).
* $I$ is a bias term.

For a system in this general class to have a potential function, three conditions must be met:

1. **Symmetric Connectivity:** The weight matrix must be symmetric, i.e., $W = W^T$.
2. **Positive Time Constant:** The time constant $\tau$ must be greater than zero ($\tau > 0$).
3. **Monotonically Increasing Transfer Function:** The activation function $\sigma$ must be monotonically increasing.

If these conditions are met, the system is guaranteed to be a gradient system, and therefore its only attractors are stable fixed points. This is the case for original Hopfield networks. Conversely, if the weights are not symmetric ($W \neq W^T$), a potential function does not exist, and more complex dynamics like limit cycles can emerge.

</div>

### Bifurcation Theory

Bifurcation theory is a broad and critical area within the study of dynamical systems. It analyzes how the qualitative behavior of a system changes as one or more of its underlying parameters are varied.

#### Defining Bifurcations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bifurcation)</span></p>

A bifurcation is a qualitative, topological change in the state space of a system that occurs as a parameter is changed. This means the vector fields before and after the bifurcation point are not topologically equivalent. Such changes can include:

* The creation or destruction of new attractors (e.g., equilibrium points).
* A change in the stability of an existing attractor.

</div>

#### The Saddle-Node Bifurcation

One of the most important and common types of bifurcations is the saddle-node bifurcation. In this event, a stable equilibrium (a node) and an unstable equilibrium (a saddle) collide and annihilate each other.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Bifurcations in Wilson-Cowan Equations)</span></p>

Consider the Wilson-Cowan model for interacting populations of excitatory ($E$) and inhibitory ($I$) neurons. The equilibria of this system are found at the intersections of the nullclines for each population.

The system's behavior is governed by parameters such as the connection weights ($w_{EE}, w_{EI}$, etc.). Let's consider the self-excitation weight of the excitatory population, $w_{EE}$, as our control parameter.

* If we increase $w_{EE}$, the S-shaped nullcline for the excitatory population ($\dot{\nu}_E = 0$) is lifted upwards.
* At a critical value of $w_{EE}$, the S-shaped nullcline becomes tangent to the inhibitory nullcline ($\dot{\nu}_I = 0$). This point of tangency is the saddle-node bifurcation point. At this exact moment, a stable node and a saddle point merge.
* If $w_{EE}$ is increased further, the two equilibria disappear, leaving only one other equilibrium point.

The region of parameter space between these two bifurcation points is a bistable regime, where the system has two stable equilibrium points separated by an unstable saddle.

</div>

#### Bifurcation Diagrams

To visualize these changes, we use a bifurcation diagram. This diagram plots the location of the system's equilibria (often denoted as $\bar{x}$) against the value of the control parameter.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

For the saddle-node bifurcation in the Wilson-Cowan example, the bifurcation diagram would look like an "S" shape turned on its side:

* The x-axis represents the control parameter (e.g., $w_{EE}$).
* The y-axis represents the position of the equilibria ($\bar{x}$).
* The upper and lower branches of the curve represent the stable equilibria (stable nodes).
* The middle branch (often drawn with a dashed line) represents the unstable equilibrium (the saddle).
* The points where the upper/lower branches meet the middle branch are the saddle-node bifurcation points.

This diagram powerfully illustrates phenomena like hysteresis and sudden transitions. A system state might follow the lower stable branch as the parameter increases, and then, upon reaching the bifurcation point, make a sudden jump to the upper stable branch. Such rapid transitions are seen in many real-world systems, including epileptic seizures in the brain.

</div>

#### The Normal Form for a Saddle-Node Bifurcation

Near a bifurcation point, many complex nonlinear systems can be simplified and shown to behave like a much simpler mathematical equation known as a normal form. The normal form captures the essential dynamics of the bifurcation.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Normal Form of a Saddle-Node Bifurcation)</span></p>

The normal form for a saddle-node bifurcation is given by the one-dimensional differential equation:

$$\dot{x} = r + x^2$$

Here, $r$ is the bifurcation parameter (or control parameter).

* When $r < 0$: The parabola $y = r + x^2$ is shifted down and intersects the x-axis at two points. These are the equilibria: one stable and one unstable.
* When $r = 0$: The parabola $y = x^2$ is tangent to the x-axis at $x=0$. The stable and unstable equilibria have merged into a single, half-stable equilibrium. This is the exact moment of the saddle-node bifurcation.
* When $r > 0$: The parabola $y = r + x^2$ is shifted up and does not intersect the x-axis. There are no equilibria.

</div>

#### The Concept of a Normal Form

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Normal Form)</span></p>

A normal form of a bifurcation is the simplest, minimal differential equation that exhibits the essential dynamics of that bifurcation. By analyzing the normal form, we can understand the universal properties of a whole class of systems near the bifurcation point, regardless of their specific physical or biological details.

</div>

#### Transcritical Bifurcation

A transcritical bifurcation is a fundamental type of bifurcation where two fixed points collide and exchange their stability properties.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The core idea behind a transcritical bifurcation is an exchange of stability. As we tune a control parameter, two equilibria move towards each other, merge at the bifurcation point, and then re-emerge with their stability profiles swapped.

The normal form for the transcritical bifurcation is: $\dot{x} = rx - x^2$.

The fixed points are $x_1^* = 0$ and $x_2^* = r$.

* **$r < 0$:** The fixed point at $x=r$ is unstable; the fixed point at $x=0$ is stable.
* **$r = 0$:** The two fixed points merge at $x=0$ (non-hyperbolic).
* **$r > 0$:** The fixed point at $x=0$ is now unstable; the fixed point at $x=r$ is now stable.

At the bifurcation point $(0,0)$, the two branches cross and exchange stability.

</div>

#### Pitchfork Bifurcation

The pitchfork bifurcation is characteristic of systems possessing a certain symmetry.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

In a pitchfork bifurcation, a single stable fixed point loses its stability as a parameter is varied. As it becomes unstable, two new stable fixed points are simultaneously created, branching off symmetrically from the original state.

The normal form for the supercritical pitchfork bifurcation is: $\dot{x} = rx - x^3$. The symmetry is evident: if $x(t)$ is a solution, then so is $-x(t)$.

* **$r \le 0$:** The only real solution is $x=0$, which is stable.
* **$r > 0$:** There are three fixed points: $x_1^* = 0$ (now unstable), $x_2^* = +\sqrt{r}$ (stable), and $x_3^* = -\sqrt{r}$ (stable).

The **supercritical** form gives rise to two new stable fixed points, while the **subcritical** form involves two unstable fixed points merging with a stable one, annihilating it and leaving no stable equilibria nearby.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Pitchfork Bifurcations in Neural Systems)</span></p>

1. **Symmetric Wilson-Cowan System:** If a parameter is changed causing one nullcline to flatten, a single stable equilibrium can give way to three equilibria (one unstable, two stable), characteristic of a pitchfork bifurcation.
2. **Single Neuron with Sigmoidal Self-Coupling:** A simple one-dimensional model of a neuron with a sigmoidal input-output function can exhibit this behavior. Changing the sigmoid's slope can cause the system to transition from one stable fixed point to three through a pitchfork bifurcation.

</div>

#### Critical Slowing Down: A Signature of Approaching Bifurcations

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Critical Slowing Down)</span></p>

As a system approaches a bifurcation point, the basin of attraction of a stable equilibrium point "flattens out." The derivative $\dot{x}$ becomes very close to zero in a wide region around the fixed point, causing the system's dynamics to become extremely slow.

**Observable Consequences:**

1. **Slow Recovery from Perturbations:** A system close to a bifurcation will take much longer to return to its stable state after being perturbed.
2. **Increased Variance:** In the presence of noise, the system's state will fluctuate with a much larger amplitude. An increase in the variance of a time series can therefore be a signature that the underlying system is approaching a bifurcation.

This phenomenon provides a powerful, model-independent warning sign that a system is approaching a critical transition or "tipping point." It is actively studied in fields like climate science.

</div>

#### Hopf Bifurcation and the Birth of Limit Cycles

While the bifurcations discussed so far concern fixed points (equilibria), the Hopf bifurcation is the most important bifurcation for the creation of a limit cycle from an equilibrium point.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hopf Bifurcation)</span></p>

A Hopf bifurcation is a bifurcation where a fixed point of a dynamical system loses stability as a pair of complex conjugate eigenvalues of the linearized system cross the imaginary axis of the complex plane. This typically results in the birth of a small-amplitude limit cycle around the fixed point.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Mechanism)</span></p>

Consider a stable spiral fixed point. Trajectories spiral inwards towards it. The stability is governed by the real part of the eigenvalues; a negative real part ensures stability. At the Hopf bifurcation point, the real part becomes exactly zero. As the parameter is changed further, the real part becomes positive, the fixed point becomes an unstable spiral, and trajectories spiral outwards, captured by a newly born, stable limit cycle.

**Core Properties:**

1. **Equilibrium Type:** The eigenvalues must be a complex conjugate pair, $\lambda = \alpha \pm i\beta$.
2. **Stability Change:** The bifurcation occurs when the real part crosses zero ($\alpha = 0$).
3. **Oscillation Frequency:** The imaginary part must be non-zero ($\beta \neq 0$) at the bifurcation point. The frequency of the resulting oscillation is approximately $\beta$.

</div>

#### The Supercritical Hopf Bifurcation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Supercritical Hopf Bifurcation)</span></p>

In a supercritical Hopf bifurcation, a stable spiral equilibrium loses its stability and becomes an unstable spiral. At the exact moment of stability change, a stable limit cycle is born with an infinitesimally small amplitude, which then grows smoothly as the parameter continues to change.

The normal form for the radius (amplitude) is: $\dot{r} = \mu r - r^3$.

* For $\mu < 0$: The only stable equilibrium is at $r=0$.
* For $\mu > 0$: The equilibrium at $r=0$ becomes unstable. A new stable limit cycle appears at $r = \sqrt{\mu}$.

</div>

#### The Subcritical Hopf Bifurcation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Subcritical Hopf Bifurcation)</span></p>

In a subcritical Hopf bifurcation, a stable spiral equilibrium loses stability when the equilibrium point coalesces with a pre-existing unstable limit cycle. This leads to abrupt, dramatic changes in system behavior.

The normal form for the radius is: $\dot{r} = \mu r + r^3$.

* For $\mu < 0$: The equilibrium at $r=0$ is stable. An unstable limit cycle exists at $r=\sqrt{-\mu}$.
* For $\mu > 0$: The equilibrium at $r=0$ becomes unstable. The unstable limit cycle has vanished, and trajectories are pushed away from the origin.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The subcritical Hopf bifurcation can be dramatic in real-world systems because it leads to sudden, large-scale changes. As the control parameter slowly approaches the bifurcation point, the system appears stable. The moment it crosses the threshold, the equilibrium vanishes, and the system "hops" to a completely different state—often a large-amplitude oscillation. A famous analogy is soldiers marching in step across a bridge: at a critical point, the bridge can jump from small vibration to large-amplitude, destructive oscillation.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Morris-Lecar Model)</span></p>

The Morris-Lecar model is a simplified 2D biophysical model describing the membrane potential of a single neuron. It consists of two coupled ODEs for the membrane potential ($V$) and a gating variable ($n$):

$$\begin{align*}
C \dot{V} &= I - g_L(V - E_L) - g_{Ca} m_{\infty}(V)(V - E_{Na}) - g_K n (V - E_K) \\
\dot{n} &= \frac{n_{\infty}(V) - n}{\tau_n(V)}
\end{align*}$$

For certain parameter configurations, the phase space has: a stable equilibrium, an unstable limit cycle surrounding the equilibrium, and a large-amplitude stable limit cycle corresponding to continuous spiking. As the injected current $I$ changes, a subcritical Hopf bifurcation occurs, forcing the system to "hop" to the spiking state. This system exhibits **bistability**: the coexistence of a stable resting state and a stable spiking state.

</div>

#### Co-dimension Two Bifurcations

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

In co-dimension two bifurcations, the system's behavior changes qualitatively when two parameters are varied simultaneously. Notable examples include the **Cusp bifurcation** and the **Bogdanov-Takens bifurcation**. These bifurcations organize the overall bifurcation structure in parameter space and are important in understanding phenomena like neuronal bursting as a bifurcation phenomenon.

</div>
## Lecture 7

### An Introduction to Chaos and the Logistic Map

This lecture introduces the fundamental concepts of chaos theory using the logistic map as a primary example. We explore the key signatures of chaotic systems and the process by which a simple, deterministic system can exhibit complex, aperiodic behavior.

#### The Logistic Map Revisited

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Logistic Map)</span></p>

The logistic map is a discrete-time dynamical system defined by the equation:

$$x_{t} = \alpha x_{t-1}(1 - x_{t-1})$$

Where $x_t$ represents the state at time $t$ and $\alpha$ is a control parameter. For the system to remain bounded within $[0, 1]$, we consider $x_0 \in [0, 1]$ and $\alpha \in [0, 4]$.

</div>

#### Core Characteristics of Chaos

As we increase the parameter $\alpha$ beyond the values that produce simple fixed points and low-period cycles, the system's behavior changes dramatically. The period of the cycles doubles at an accelerating rate until the trajectory becomes irregular and appears to fill a dense region of the state space. This phenomenon is known as chaos.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Aperiodicity in Deterministic Systems)</span></p>

The first signature of chaos is aperiodic or irregular behavior in the complete absence of noise. For a chaotic trajectory, there is no integer $n$ for which the system exactly repeats its state: $x_{t+n} \neq x_t$ for all $n > 0$. The trajectory never closes on itself.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sensitive Dependence on Initial Conditions)</span></p>

Two trajectories starting from infinitesimally different initial states will diverge from each other at an exponential rate. Even a minuscule difference in starting points leads to completely different outcomes after a short period. This exponential divergence is a hallmark of chaotic dynamics.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Boundedness)</span></p>

Despite the exponential divergence of nearby trajectories, the system remains bounded. For the logistic map, it can be proven that for $\alpha \in [0, 4]$ and $x_0 \in [0, 1]$, the trajectory will never leave the unit interval. The system is simultaneously divergent on a local scale and constrained on a global scale.

</div>

#### The Period Doubling Route to Chaos

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The transition to chaos in the logistic map occurs via the **period-doubling cascade**:

1. **Stable Fixed Point:** For small $\alpha$, the system has a single stable fixed point.
2. **First Bifurcation:** As $\alpha$ increases, the fixed point becomes unstable and gives rise to a stable 2-cycle.
3. **Cascade:** The 2-cycle bifurcates into a 4-cycle, then 8-cycle, 16-cycle, etc. The $\alpha$-interval over which each subsequent cycle is stable becomes progressively shorter.
4. **Onset of Chaos:** This cascade culminates at a finite value of $\alpha$, beyond which the behavior becomes chaotic and aperiodic.

</div>

#### The Strange Attractor

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Even in the chaotic regime, the object to which trajectories converge is still an attractor. Initial conditions from within a certain basin of attraction converge toward this complex, bounded object. There is simultaneous convergence from the outside and divergence within the attractor itself. Such an object is called a **strange attractor**.

Periodic windows embedded within the chaotic regime are also visible in the bifurcation diagram. As $\alpha$ increases through the chaotic region, the system can suddenly revert to stable periodic behavior for narrow parameter ranges before returning to chaos.

</div>

### Deeper Properties of One-Dimensional Maps

#### Unstable Periodic Orbits (UPOs)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

For certain parameter values, the chaotic attractor is built upon a "skeleton" of infinitely many unstable periodic orbits (UPOs). For the logistic map at $\alpha = 4$, UPOs exist for every possible integer period $k \geq 1$. These are complete, periodic cycles that the system could follow, but they are all unstable—any small perturbation will cause the trajectory to move away.

</div>

#### Topological Equivalence: The Tent Map

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Tent Map)</span></p>

The tent map is a piecewise linear map defined by:

$$f(x) = \alpha \cdot \min(x, 1-x)$$

The map is typically studied for $x \in [0, 1]$ and $\alpha \in [0, 2]$. Its graph has a characteristic "tent" shape.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The logistic map is topologically equivalent to the tent map. The study of piecewise linear systems is important because: (1) many mathematical analyses become much easier while still exhibiting complex behaviors like chaos; and (2) many recurrent neural networks (RNNs) used in deep learning are fundamentally piecewise linear systems and can exhibit the full spectrum of dynamical behaviors.

</div>

#### The Li-Yorke Theorem: Period Three Implies Chaos

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Li-Yorke Theorem, 1975)</span></p>

For a one-dimensional map, if a period-3 cycle is observed, then the system must also exhibit chaotic behavior.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

This is a powerful and surprising result. The mere existence of a cycle with period three is a sufficient condition to guarantee that the system's dynamics are complex enough to be classified as chaotic.

</div>

#### The Sharkovskii Ordering of Periodic Orbits

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Sharkovskii's Theorem)</span></p>

There exists a specific ordering of natural numbers, known as the Sharkovskii ordering, which dictates a hierarchy of implications for the existence of periodic orbits in one-dimensional maps. If a map possesses a periodic orbit of period $k$, it must also possess periodic orbits for all periods that appear to the right of $k$ in the ordering:

$$\begin{align*}
3 &\implies 5 \implies 7 \implies 9 \implies \dots \implies (2n+1) \implies \dots \\
\implies 3 \cdot 2 &\implies 5 \cdot 2 \implies 7 \cdot 2 \implies \dots \implies (2n+1) \cdot 2 \implies \dots \\
\implies 3 \cdot 2^2 &\implies 5 \cdot 2^2 \implies \dots \implies (2n+1) \cdot 2^2 \implies \dots \\
\vdots \\
\implies 2^n &\implies 2^{n-1} \implies \dots \implies 2^3 \implies 2^2 \implies 2 \implies 1
\end{align*}$$

The number $3$ is first in the ordering. Therefore, the existence of a 3-cycle implies the existence of cycles of all other integer periods, connecting back to the Li-Yorke theorem.

</div>

#### The Feigenbaum Constants

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

For maps exhibiting the period-doubling route to chaos, the rate at which bifurcations occur is governed by a universal constant. Let $\alpha_n$ be the parameter value at which the period-doubling bifurcation from a $2^{n-1}$-cycle to a $2^n$-cycle occurs. The ratio of successive bifurcation intervals converges to the first Feigenbaum constant:

$$\delta = \lim_{n \to \infty} \frac{\alpha_n - \alpha_{n-1}}{\alpha_{n+1} - \alpha_n} \approx 4.669...$$

This universality indicates that a wide class of systems transitions to chaos in a quantitatively identical manner.

</div>

### Examples of Chaotic Behavior

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Population Dynamics in Beetles)</span></p>

An empirical study of a beetle population provided evidence for the period-doubling cascade in a biological system. By manipulating a control parameter, researchers observed the population's long-term behavior transition from a stable equilibrium to 2-cycles, then 4-cycles, and eventually to chaotic fluctuations, providing a real-world parallel to the dynamics of the logistic map.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Bursting Neuron Model)</span></p>

Chaos can also be found in complex, higher-dimensional continuous systems. A three-dimensional model of a bursting neuron can exhibit chaotic dynamics through bifurcations such as a homoclinic bifurcation where a limit cycle crashes into a saddle point. Within the parameter space of such biophysical models, chaotic dynamics can emerge, leading to irregular, non-repeating patterns of neural activity.

</div>

### Chaos in Continuous-Time Dynamical Systems

#### What is Chaos? An Initial View

At its core, chaos describes complex, unpredictable behavior in deterministic dynamical systems. A key indicator of potential chaos is the presence of homoclinic orbits, which are trajectories that connect a saddle-type equilibrium point to itself.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The presence of homoclinic structures is a strong hint that a system might exhibit chaotic behavior. For certain classes of systems, particularly discrete maps, the existence of homoclinic orbits is not just a hint—it is a guarantee. Formal theorems prove this connection.

</div>

#### Examples of Chaotic Systems

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Bifurcation in a Neuron Model)</span></p>

A full three-dimensional model of a biological neuron exhibits the following sequence as a key conductance parameter is increased:

1. **Bursting Behavior:** Rapid bursts followed by quiescence.
2. **Chaotic Regime:** Highly irregular, aperiodic spiking. The trajectory densely fills a bounded region—the hallmark of a chaotic attractor.
3. **Regular Spiking:** Stable, periodic firing pattern.

This progression is observed empirically in biological experiments. The chaotic dynamics are produced by highly nonlinear NMDA conductances. If the model were linearized, these complex phenomena would vanish entirely.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Lorenz System)</span></p>

The Lorenz system is a set of three coupled, nonlinear, first-order ordinary differential equations:

$$\begin{aligned} \dot{x} &= s(y - x) \\ \dot{y} &= x(r - z) - y \\ \dot{z} &= xy - bz \end{aligned}$$

The system has three dynamical variables ($x, y, z$) and three positive constant parameters ($s, r, b$). The nonlinearity stems from just two terms: the product $xz$ in the second equation and the product $xy$ in the third. By varying $r$ while keeping $s$ and $b$ fixed, the system exhibits a range of behaviors, culminating in the renowned Lorenz attractor.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Attractor)</span></p>

An attractor is a set of states in the state space towards which a system's trajectory evolves over time. For a chaotic attractor, trajectories that start nearby converge onto this object, then move around chaotically, densely filling a region of the state space.

</div>

#### Core Properties of Chaotic Systems

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Aperiodic Trajectory)</span></p>

An aperiodic trajectory is one that never closes up or repeats itself exactly. No matter how long the system is observed, the state vector will never return to a previous value.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Minimum Dimension for Chaos)</span></p>

For continuous-time dynamical systems (described by ordinary differential equations), chaos is only possible in three or more dimensions.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(by Intuition)</span></p>

1. **One Dimension:** A trajectory can only move along a line. It can approach a fixed point or go to infinity, but cannot exhibit complex dynamics.
2. **Two Dimensions:** Trajectories can form limit cycles. However, a fundamental property of deterministic ODEs is that trajectories cannot cross. A non-intersecting, bounded curve in a 2D plane must eventually close on itself, forming a periodic orbit. It cannot densely fill a 2D area.
3. **Three Dimensions:** A trajectory has enough freedom to move without intersecting itself while remaining bounded, weaving through space in a complex, aperiodic pattern.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sensitive Dependence on Initial Conditions)</span></p>

Two trajectories starting from arbitrarily close initial points will, on average, diverge from each other exponentially over time. This is the **butterfly effect**: a tiny change in the system's initial state can lead to macroscopically different outcomes. This makes long-term prediction fundamentally impossible.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Structured Nature of Chaos)</span></p>

While individual trajectories are unpredictable in the long term, the collective behavior is highly structured:

* The chaotic attractor is a bounded object with specific geometry. A trajectory on the Lorenz attractor will always remain within the "butterfly wings."
* The system possesses **ergodic statistics**: averages over a long time for a single trajectory equal averages over the entire attractor at a single moment.

Therefore, while we cannot predict the exact state far into the future, we can predict the probability of finding the system in a particular region of its attractor.

</div>

### Characterizing Chaos: Lyapunov Exponents

The Lyapunov exponent provides a quantitative measure of the average rate of divergence or convergence of nearby trajectories. A positive Lyapunov exponent is the hallmark of chaotic dynamics.

#### Defining the Maximum Lyapunov Exponent

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Maximum Lyapunov Exponent, Discrete Time)</span></p>

For a discrete-time system $x_{n+1} = f(x_n)$, the maximum Lyapunov exponent $\lambda_{\text{max}}$ is:

$$\lambda_{\text{max}} = \lim_{n \to \infty} \frac{1}{n} \log \left\| \prod_{i=0}^{n-1} J(x_i) \right\|$$

where $J(x_i)$ is the Jacobian of $f$ at $x_i$ and $\| \cdot \|$ is the spectral norm.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Maximum Lyapunov Exponent, Continuous Time)</span></p>

For a continuous-time system with flow operator $\phi_t(x_0)$:

$$\lambda_{\text{max}} = \lim_{t \to \infty} \lim_{\| \Delta x_0 \| \to 0} \frac{1}{t} \log \left( \frac{\| \phi_t(x_0 + \Delta x_0) - \phi_t(x_0) \|}{\| \Delta x_0 \|} \right)$$

This can also be expressed as: $\lambda_{\text{max}} = \lim_{t \to \infty} \frac{1}{t} \log \left\| \frac{\partial \phi_t(x_0)}{\partial x_0} \right\|$.

</div>

#### The Lyapunov Spectrum and Geometric Intuition

In a $d$-dimensional system, the full Lyapunov spectrum consists of $d$ exponents: $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stretching and Contracting Space)</span></p>

Imagine a small spherical ball of initial conditions. As the system evolves:

* Directions with positive $\lambda_i$: the ball is **stretched**.
* Directions with negative $\lambda_i$: the ball is **compressed**.
* Directions with zero $\lambda_i$: distance is preserved on average.

For a chaotic attractor, this continuous stretching and folding generates the complex, fractal structures characteristic of strange attractors. The connection to singular value decomposition (SVD) is direct: the maximum Lyapunov exponent corresponds to the maximum singular value of the Jacobian product matrix.

</div>

#### Lyapunov Exponents as a Diagnostic Tool

| Value of $\lambda_{\text{max}}$ | Implied Long-Term Behavior |
| --- | --- |
| $\lambda_{\text{max}} > 0$ | Chaotic evolution. Trajectories diverge exponentially. |
| $\lambda_{\text{max}} = 0$ | Neutral stability. Often occurs on a limit cycle. |
| $\lambda_{\text{max}} < 0$ | Stable fixed point or stable periodic orbit. |

#### A Rigorous Definition of Chaos

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Chaos)</span></p>

A dynamical system is considered chaotic if it satisfies:

1. **Sensitive Dependence on Initial Conditions:** At least one positive Lyapunov exponent ($\lambda_{\text{max}} > 0$).
2. **Boundedness:** Trajectories are confined to a bounded region of state space.

An additional condition sometimes included: all Lyapunov exponents are unequal to zero (aperiodicity).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Condition for a Chaotic Attractor)</span></p>

A chaotic system possesses an attractor only if the sum of all its Lyapunov exponents is less than zero:

$$\sum_{i=1}^{d} \lambda_i < 0$$

This ensures that while the system stretches along at least one direction, the contraction along other directions is strong enough to ensure overall volume shrinkage towards a bounded, fractal strange attractor.

</div>

### Fractal Structures in Chaotic Systems

#### Introduction to Fractals in Chaos

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

A hallmark of chaotic systems is a profound sensitivity to initial conditions, which becomes even more pronounced when basins of attraction possess fractal boundaries. The emergence of fractal geometries is due to three key actions: (1) stretching, (2) contraction, and (3) folding or reinjection. Repeated iteration gives rise to **self-similarity**, where zooming into a part of the structure reveals smaller copies of the whole.

</div>

#### The Cantor Set: A Prototypical Fractal

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Cantor Set)</span></p>

The Cantor set is constructed through an iterative procedure starting with the closed interval $[0, 1]$.

1. Let $K_0 = [0, 1]$.
2. Remove the open middle third, $(\frac{1}{3}, \frac{2}{3})$, to get $K_1 = [0, \frac{1}{3}] \cup [\frac{2}{3}, 1]$.
3. Remove the open middle third from each remaining segment to get $K_2$.
4. Continue iteratively. The Cantor set $K = \lim_{n \to \infty} K_n$.

At the $n$-th iteration, $K_n$ consists of $2^n$ closed intervals. The Cantor set consists of all numbers in $[0, 1]$ whose ternary representation uses only the digits $0$ and $2$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Uncountability of the Cantor Set)</span></p>

The Cantor set is an uncountably infinite set.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name"></span></p>

1. **Assume for Contradiction:** Assume the Cantor set is countable. Create an enumeration $x_1, x_2, x_3, \dots$.
2. **Represent in Ternary:** Each $x_i$ has ternary representation using only digits $\lbrace 0, 2 \rbrace$: $x_i = 0.a_{i1} a_{i2} a_{i3} \dots$.
3. **Construct a New Number:** Define $R = 0.r_1 r_2 r_3 \dots$ by examining the diagonal: if $a_{ii} = 0$, set $r_i = 2$; if $a_{ii} = 2$, set $r_i = 0$.
4. **Show Contradiction:** $R$ uses only digits $\lbrace 0, 2 \rbrace$ (so belongs to the Cantor set), but differs from every $x_i$ at position $i$. Therefore $R$ is not in our list—a contradiction.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The total length of intervals removed from $[0, 1]$ sums to $1$. The Cantor set has zero Lebesgue measure, yet contains uncountably many points. This combination of zero measure and uncountable cardinality is a defining feature of many fractal sets.

</div>

#### The Smale Horseshoe Map

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Smale Horseshoe Map)</span></p>

The Smale horseshoe map $F$ is a mapping of the unit square $S = [0, 1] \times [0, 1]$ onto $\mathbb{R}^2$. The process involves:

1. **Stretching and Contraction:** A linear transformation stretches the square by a factor of $3$ horizontally and compresses it by a factor of $3$ vertically.
2. **Folding:** The long rectangle is bent into a "horseshoe" shape and placed back over the original square.

Only the parts of the horseshoe within the original square $S$ are considered for the next iteration. After the first application, these consist of two vertical rectangular regions $H_1$ and $H_2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Lyapunov Exponents)</span></p>

The Smale Horseshoe map has two constant Lyapunov exponents:

* $\lambda_1 = \log(3) > 0$ (stretching direction)
* $\lambda_2 = \log(1/3) = -\log(3) < 0$ (contracting direction)

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Invariant Set $\Lambda$)</span></p>

The invariant set $\Lambda$ is the set of all points in $S$ that remain in $S$ for all time, both forward and backward:

$$\Lambda = \bigcap_{k=-\infty}^{\infty} f^k(S)$$

$\Lambda$ is the intersection of horizontal stripes (forward iteration) and vertical stripes (backward iteration). It is a fractal set—the product of two Cantor sets.

</div>

#### Symbolic Dynamics

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Each point in $\Lambda$ can be assigned a unique bi-infinite sequence of symbols $...x_{-2}x_{-1}.x_0x_1x_2...$ by tracking which stripe it belongs to at each iteration. This symbolic coding transforms complex geometric problems into algebraic ones.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Structure of Orbits in $\Lambda$)</span></p>

The invariant set $\Lambda$ of the Smale Horseshoe map contains:

1. A countable set of unstable periodic orbits of every possible length (period).
2. An uncountable set of aperiodic but bounded orbits.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Skeleton of Chaos)</span></p>

Steven Strogatz provides a beautiful metaphor for the structure of a chaotic attractor: an "infinite skeleton of unstable periodic orbits." The system state wanders between these orbits, repelled by one and attracted to another, much like a "ball in a pinball wizard machine." The chaotic attractor is organized around this dense, unstable backbone of periodic behaviors.

</div>
## Lecture 8

### Characterizing the Geometry of Attractors

#### The Need for New Geometric Measures

In our previous discussions, we explored chaos and fractal sets. We established that chaotic attractors often produce intricate fractal structures. A natural question arises: how can we describe their geometry? Classical Euclidean geometry, which deals with integer dimensions, seems inadequate. The Cantor set has zero Lebesgue measure yet contains uncountably many points. This signals the need for new types of measurement.

#### The Box-Counting Dimension

The fundamental idea is remarkably simple: how many boxes of a given size are needed to completely cover the set?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition and Scaling)</span></p>

We tile our space with boxes of side length $\epsilon$ and count $N(\epsilon)$, the number of boxes intersected by our set $S$. The scaling law is:

$$N(\epsilon) \propto \left(\frac{1}{\epsilon}\right)^D$$

* **A Line** ($D=1$): Halving the box size doubles the count: $N(\epsilon') = 2^1 N(\epsilon)$.
* **A Plane** ($D=2$): Halving the box size quadruples the count: $N(\epsilon') = 2^2 N(\epsilon)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Box-Counting Dimension)</span></p>

For a set $S$ in $\mathbb{R}^m$, the box-counting dimension $D_{box}$ is:

$$D_{box} = \lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log(1/\epsilon)}$$

where $N(\epsilon)$ is the minimum number of $m$-dimensional boxes of side length $\epsilon$ required to cover $S$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Box-Counting Dimension of the Cantor Set)</span></p>

After $n$ iterations of the Cantor set construction, we have $2^n$ intervals of length $(1/3)^n$. Setting $\epsilon_n = (1/3)^n$ and $N(\epsilon_n) = 2^n$:

$$D_{box} = \lim_{n \to \infty} \frac{\log(2^n)}{\log(3^n)} = \lim_{n \to \infty} \frac{n \log 2}{n \log 3} = \frac{\log 2}{\log 3} \approx 0.6309$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The result is a fractal dimension—a non-integer value. The Cantor set is more complex than a collection of points (dimension $0$) but less space-filling than a line (dimension $1$). This fractional value captures the set's self-similar structure and is a general property of chaotic attractors.

Empirically, the box-counting dimension can be estimated by plotting $\log N(\epsilon)$ against $\log(1/\epsilon)$ and extracting the slope of the linear region.

</div>

#### The Correlation Dimension

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Motivation)</span></p>

While the box-counting dimension is powerful, it can be computationally prohibitive for high-dimensional systems due to the curse of dimensionality. The correlation dimension provides a more practical, trajectory-based alternative. Its value is always $D_{corr} \le D_{box}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Correlation Dimension)</span></p>

The correlation dimension $D_{corr}$ is:

$$D_{corr} = \lim_{\epsilon \to 0} \frac{\log C(\epsilon)}{\log \epsilon}$$

where $C(\epsilon)$ is the correlation integral, measuring the average number of neighbors within an $\epsilon$-ball around points on the trajectory.

The method: (1) take points along a trajectory, (2) for each point, place a ball of radius $\epsilon$, (3) count the average number of other points within this ball, $C(\epsilon)$, (4) examine how $C(\epsilon)$ scales as $\epsilon \to 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Maps vs. Continuous Systems)</span></p>

* **Maps:** Chaotic behavior can occur in one-dimensional maps (e.g., the logistic map).
* **Continuous-Time ODEs:** Chaos requires at least three dimensions (Poincaré-Bendixson theorem). In four or more dimensions, hyperchaos can emerge.

</div>

### From Theory to Practice: Analyzing Empirical Data

#### The Fundamental Challenge of Real-World Systems

We now pivot from idealized models to empirical data. We often have data from a real-world system without knowing the underlying equations. The central questions are: (1) How can we apply dynamical systems theory to characterize real data? (2) Can we derive data-driven models of the underlying process?

#### The Measurement Problem

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

We assume an unknown data-generating system $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$ where $\mathbf{x} \in \mathbb{R}^m$. We never observe the full state $\mathbf{x}$ directly. Instead, a measurement function $h$ gives us a time series of scalar observations: $s(t_k) = h(\mathbf{x}(t_k))$. Our goal is to reconstruct the dynamics from this limited data.

</div>

### State Space Reconstruction from Time Series Data

#### The Intuition of Time Delay Embedding

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Simple Harmonic Oscillator)</span></p>

If we observe a single point on a sine wave, we cannot tell if it's about to increase or decrease. But if we also consider the value at a slightly earlier time, $t-\tau$, we gain crucial context. A two-dimensional space $(y_t, y_{t-\tau})$ resolves the ambiguity—the trajectory forms a clean, non-intersecting closed loop.

For more complex oscillations (e.g., ECG-like signals), even two dimensions may not suffice, and a three-dimensional space $(y_t, y_{t-\tau}, y_{t-2\tau})$ may be needed.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

A fundamental property of a valid state space is that trajectories cannot intersect. By augmenting the current measurement with its own past values, we create new dimensions, "unfolding" the trajectory into a higher-dimensional space where intersections are eliminated.

</div>

#### The Delay Coordinate Map

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Delay Coordinate Map)</span></p>

The delay coordinate map transforms a scalar time series $y_t \in \mathbb{R}$ into a vector in an $m$-dimensional space:

$$\mathbf{Y}_t = (y_t, y_{t-\tau}, y_{t-2\tau}, \dots, y_{t-(m-1)\tau})$$

Two crucial parameters:

* $m$: The embedding dimension.
* $\tau$: The time lag or time delay.

</div>

#### Choosing the Embedding Parameters

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Time Lag $\tau$)</span></p>

* **Too small:** Components become highly correlated; the reconstruction collapses onto a diagonal.
* **Too large:** Components become uncorrelated; deterministic structure is lost.

$\tau$ must be in an intermediate range. A common heuristic: choose $\tau$ near the first minimum of the autocorrelation function.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Empirical Autocorrelation Function)</span></p>

$$R_y(\tau) = \frac{\sum_{t} (y_t - \bar{y})(y_{t-\tau} - \bar{y})}{S_y^2}$$

The numerator is the covariance between the time series and its time-lagged version; the denominator normalizes this value.

</div>

#### Properties of a "Good" Embedding

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Diffeomorphism)</span></p>

A homeomorphism is a continuous, one-to-one mapping with a continuous inverse. A **diffeomorphism** additionally requires both the map and its inverse to be continuously differentiable. The derivative $DF$ (the "push-forward") transports the vector field from one domain to another and must also be one-to-one.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Desired Properties of the Embedding Map)</span></p>

1. **Topological Preservation (One-to-One):** Distinct points on the original attractor map to distinct points in reconstruction space.
2. **Dynamical Preservation (Preserving the Vector Field):** The derivative $dF$ must also be one-to-one, ensuring local dynamics are preserved. This requires $dF$ to have full rank.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Avoiding Intersections in Higher Dimensions)</span></p>

Given two manifolds of dimensions $d_1$ and $d_2$, the ambient space must have dimension at least $d_1 + d_2 + 1$ so that randomly placed manifolds will almost certainly not intersect. In 2D, random lines likely intersect; in 3D, they almost certainly pass as skew lines.

</div>

### Embedding Theorems

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Whitney's Embedding Theorem)</span></p>

Let $A \subset \mathbb{R}^m$ be a compact, smooth manifold of dimension $D$. Then almost any smooth map $f: \mathbb{R}^m \to \mathbb{R}^k$ where $k \geq 2D + 1$ is an embedding (diffeomorphism onto its image).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Takens' Delay Embedding Theorem)</span></p>

Assume $A \subset \mathbb{R}^m$ is a $D$-dimensional smooth manifold invariant under the flow $\Phi$. Let $h$ be a generic measurement function, and $F_{\tau} = G_{\tau} \circ h$ a delay coordinate map with generic delay $\tau$.

If $k \geq 2D + 1$, then $F_{\tau}$ is an embedding (a diffeomorphism).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

A "bad" choice of $\tau$ (e.g., an integer multiple of the period for a periodic system) can ruin the embedding. The "generic" condition means almost any choice works; if one fails, a slight change will succeed.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Fractal Delay Embedding Prevalence Theorem)</span></p>

Assume $A \subset \mathbb{R}^m$ is a compact subset invariant under the flow $\Phi$, with box-counting dimension $D_{box}$. If $k > 2 D_{box}$, then a delay coordinate map with generic measurement and delay is an embedding.

</div>

#### Practical Estimation: The False Neighbors Method

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The False Neighbors Method)</span></p>

In an embedding with insufficient dimension, points far apart on the true attractor may appear close ("false neighbors"). As dimension increases, these false neighbors separate.

**Methodology:**

1. Start with low embedding dimension $d=1, 2, 3, \dots$.
2. For each point, find its nearest neighbor in $d$ dimensions.
3. Check if distance increases dramatically in $d+1$ dimensions.
4. If so, classify as a "false neighbor."
5. Plot false neighbor percentage vs. $d$. The dimension where it drops to near zero is the sufficient embedding dimension.

</div>

#### Applications: Computing Invariants in Embedded Space

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Computable Quantities)</span></p>

Once a proper embedding is constructed:

* **Attractor Dimension:** Box-counting or correlation dimension computed from the reconstructed trajectories.
* **Lyapunov Exponents:** Estimated by tracking the evolution of distances between nearby points. For initially close trajectories: $\delta(t) \approx \Delta_0 e^{\lambda t}$, so $\ln(\delta(t)) \approx \ln(\Delta_0) + \lambda t$. The slope of the initial linear region gives $\lambda$.

</div>

### Inferring Models from Data

#### Two Fundamental Approaches

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Model Inference Approaches)</span></p>

1. **Vector Field Approximation:** For $\dot{x} = f(x)$, find $\hat{f}_{\theta}(x)$ that approximates $f(x)$.
2. **Flow Operator Approximation:** For $x_{k+1} = \Phi(x_k)$, find $\hat{F}_{\theta}(x_k)$ that approximates $\Phi$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Machine Learning Pipeline)</span></p>

1. **Specify a Model:** Choose a flexible function class (e.g., neural network) as $\hat{f}_{\theta}$.
2. **Specify a Loss Function:** Quantify discrepancy between predictions and observations.
3. **Training (Optimization):** Use gradient descent to minimize the loss function.

</div>

### SINDy: Sparse Identification of Nonlinear Dynamics

SINDy, introduced by Brunton, Proctor, and Kutz (2016), is a comparatively simple yet elegant approach that focuses on discovering the vector field.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Core Assumptions)</span></p>

SINDy operates on the assumption that the vector field can be expressed as a **sparse** linear combination of functions from a predefined library. Key assumptions:

* It directly approximates the vector field $f(x)$ in $\dot{x} = f(x)$.
* It assumes access to state variable measurements (or uses delay embedding first).
* It requires numerical derivatives, which amplify noise.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(SINDy Model Formulation)</span></p>

**Library of Basis Functions:** A library $\Theta(x)$ of candidate functions $\lbrace \phi_b(x) \rbrace$ is defined a priori (e.g., polynomials, trigonometric functions).

**Linear Model Structure:** Each component of the vector field is:

$$\dot{x}_i = f_i(x) \approx c_{0i} + \sum_{b=1}^{P} c_{bi} \phi_b(x_1, ..., x_n)$$

The approximation in matrix form: $\hat{\mathbf{f}}(\mathbf{x}(t)) = \mathbf{C} \cdot \mathbf{\Theta}(\mathbf{x}(t))$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Stone-Weierstrass Theorem)</span></p>

On a compact set, any continuous function can be approximated arbitrarily well by a polynomial function of sufficiently high order.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Mean Squared Error Loss)</span></p>

$$L(\mathbf{C}) = \sum_{t=1}^{T} \| \hat{\dot{\mathbf{x}}}(t) - \mathbf{C} \cdot \mathbf{\Theta}(\mathbf{x}(t)) \|_2^2$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lasso Regularization)</span></p>

To achieve sparsity, a penalty is added:

$$L(\mathbf{C}) = \sum_{t=1}^{T} \| \hat{\dot{\mathbf{x}}}(t) - \mathbf{C} \cdot \mathbf{\Theta}(\mathbf{x}(t)) \|_2^2 + \lambda \sum_{i,j} |c_{ij}|$$

The parameter $\lambda$ controls the sparsity-accuracy trade-off. The $L_1$-norm penalty forces some coefficients to exactly zero.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Closed-Form Solution for $\lambda=0$)</span></p>

Without regularization, setting the derivative to zero gives:

$$\hat{\mathbf{C}} = \left( \sum_{t=1}^{T} \hat{\dot{\mathbf{x}}}(t) \mathbf{\Theta}(\mathbf{x}(t))^T \right) \left( \sum_{t=1}^{T} \mathbf{\Theta}(\mathbf{x}(t)) \mathbf{\Theta}(\mathbf{x}(t))^T \right)^{-1}$$

This closed-form solution is computationally fast and does not require numerical iteration. When $\lambda > 0$, the sign function in the derivative makes a closed-form solution difficult, and iterative numerical methods are required.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Recovering the Lorenz Equations)</span></p>

SINDy was successfully applied to the Lorenz system: a polynomial library was chosen, Lasso regression forced many coefficients to zero, and the final sparse model recovered equations very close to the true Lorenz equations.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Limitations)</span></p>

SINDy is effective when the library is geared towards the system being studied. For real empirical data where the functional form is unknown, the method can fail if essential functions are not in the library. The choice of library often requires physical domain knowledge.

</div>
## Lecture 9

### Introduction to Universal Approximators for Dynamical Systems

#### Beyond Pre-defined Libraries

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Methods like SINDy rely on a pre-defined library of functions specified a priori. Deep learning methods do not have this limitation: they are **universal approximators**, capable of learning complex functions directly from data without manually defining a basis of candidate functions.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Universal Approximation of Dynamical Systems)</span></p>

A Recurrent Neural Network (RNN) can be formally shown to be a universal approximator of dynamical systems. These models were, and in many domains remain, state-of-the-art for time series prediction and modeling of dynamical systems.

</div>

### The Architecture and Motivation of Recurrent Neural Networks

#### Neuroscientific Origins and Core Concepts

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Like many foundational architectures, RNNs have roots in neuroscience. The key components are:

* **Units (Neurons):** Nodes with activation $x_i^t$ at time $t$.
* **Synaptic Connections (Weights):** Weight $w_{ij}$ from unit $j$ to unit $i$.
* **Recurrent Connections:** Feedback connections forming cycles, enabling the network to maintain an internal state or "memory."
* **External Inputs:** Input time series $S_t$.

</div>

#### Mathematical Formulation of an RNN

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(RNN Activation Dynamics)</span></p>

The activation of unit $i$ at time $t$ is:

$$x_i^t = \phi \left( \sum_j w_{ij} x_j^{t-1} + h_i + \sum_k c_{ik} S_k^t \right)$$

Where:

* $\phi$ is a non-linear activation function.
* $w_{ij}$ is the connection weight from unit $j$ to unit $i$.
* $h_i$ is a learnable bias term.
* $c_{ik}$ is the weight for the $k$-th external input.

The learnable parameters include connection weights ($w_{ij}$), input weights ($c_{ik}$), and bias terms ($h_i$).

</div>

#### The RNN in Vector Notation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The RNN in Vector Notation)</span></p>

The RNN state update equation:

$$z_t = \phi(W z_{t-1} + C s_t + h)$$

Or more generally: $z_t = f(z_{t-1}, s_t; \theta)$, where:

* $z_t \in \mathbb{R}^M$ is the state vector (latent states).
* $W \in \mathbb{R}^{M \times M}$ is the weight matrix.
* $h \in \mathbb{R}^M$ is the bias vector.
* $s_t \in \mathbb{R}^K$ is the external input.
* $C \in \mathbb{R}^{M \times K}$ is the input weight matrix.
* $\phi(\cdot)$ is the element-wise activation function.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The RNN as a Discrete-Time Dynamical System)</span></p>

The recursive formulation $z_t = f(z_{t-1}, s_t; \theta)$ reveals that an RNN is a **discrete-time, multi-dimensional recursive map**, directly paralleling maps like the logistic map. Depending on parameters and initial conditions, an RNN can:

* Converge to different fixed points.
* Exhibit periodic behavior (cycles).
* Undergo bifurcations.
* Display chaotic dynamics.

</div>

### Training Recurrent Neural Networks

#### The Gradient Descent Paradigm

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Gradient descent dominates not because it is the most powerful technique, but because it is effective and scalable. Since RNNs are highly nonlinear, no analytical closed-form solution exists for optimal parameters—we must rely on iterative numerical optimization.

</div>

#### Defining the Core Components

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Dataset)</span></p>

The training data consists of $P$ patterns. For each pattern $p \in \lbrace 1, \dots, P \rbrace$:

* **Inputs:** A sequence $\lbrace s_t^{(p)} \rbrace_{t=1}^{T_p}$, $s_t^{(p)} \in \mathbb{R}^K$.
* **Targets:** A sequence $\lbrace x_t^{(p)} \rbrace_{t=1}^{T_p}$, $x_t^{(p)} \in \mathbb{R}^N$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Model Architecture)</span></p>

The complete model consists of two parts:

1. **Recursive Core (State Equation):** $z_t = f(z_{t-1}, s_t; \theta)$
2. **Decoder (Observation Model):** $\hat{x}_t = g(z_t; \lambda)$

A common decoder is a linear mapping: $\hat{x}_t = B z_t$, where $B \in \mathbb{R}^{N \times M}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Loss Function)</span></p>

The Sum of Squared Errors (SSE) loss:

$$L(\theta, \lambda) = \sum_{p=1}^{P} \sum_{t=1}^{T_p} \| x_t^{(p)} - \hat{x}_t^{(p)} \|^2$$

</div>

#### The Gradient Descent Algorithm

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gradient Descent)</span></p>

Parameters are updated iteratively:

$$\theta_n = \theta_{n-1} - \gamma \nabla L(\theta_{n-1})$$

where $\gamma$ is the learning rate. The gradient $\nabla L(\theta)$ points in the direction of steepest ascent; we move opposite to it.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Training as a Dynamical System)</span></p>

The iterative update rule $\theta_n = \theta_{n-1} - \gamma \nabla L(\theta_{n-1})$ is itself a **discrete-time dynamical system**. The training process can converge to a fixed point (desired), become oscillatory, or exhibit chaos. The entire toolset of dynamical systems theory applies to the training process itself.

</div>

### Challenges in Gradient-Based Optimization

#### Local Minima and Saddle Points

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

At any point where $\nabla L(\theta) = 0$, the update step becomes zero. These can be **local minima** (suboptimal valleys) or **saddle points** (minima in one dimension, maxima in another). If the RNN operates in a chaotic regime, the loss landscape can be incredibly complex and even fractal.

</div>

#### Learning Rate Selection

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

* **$\gamma$ too small:** Minuscule updates in flat regions; extremely slow convergence.
* **$\gamma$ too large:** Overshooting in steep regions; oscillations or divergence.

The ideal learning rate would be adaptive: large in flat regions and small in steep regions.

</div>

### Classical Remedies and Modern Approaches

#### Multiple Initial Conditions

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Run optimization multiple times from different random initial parameters $\theta_0$, then choose the model with the lowest final loss. Simple but computationally expensive.

</div>

#### Overparameterization: Double Descent and the Lottery Ticket Hypothesis

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

**Double descent:** Classical theory predicts a U-shaped test loss curve. However, if you continue increasing parameters far beyond overfitting, test loss can decrease again.

The **Lottery Ticket Hypothesis:** A very large network is like a lottery containing many tickets. Within it exists a smaller, optimal sub-network (the "winning ticket"). Training effectively carves out this sub-network by pruning unnecessary connections.

</div>

#### Stochastic Gradient Descent (SGD)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stochastic Gradient Descent)</span></p>

In SGD, the gradient update is calculated on a randomly drawn mini-batch of the training data, rather than the entire dataset. The inherent noise helps the optimization "jump out" of shallow local minima.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Time Series Data)</span></p>

When applying SGD to time series, randomly sampling individual points destroys temporal structure. Instead, sample **consecutive blocks or segments** to preserve dynamic relationships.

</div>

#### Adaptive Learning Rates

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Modern optimizers use adaptive learning rates:

* **Adagrad:** Adapts based on historical sum of squared gradients.
* **Momentum:** Adds fraction of previous update, building "velocity."
* **Adam:** Combines momentum and adaptive scaling. Widely used as default.
* **RAdam:** Corrects for high variance in early training stages.

</div>

#### Second-Order Methods

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Hessian Matrix)</span></p>

The Hessian is the matrix of second-order partial derivatives of the loss. A second-order update:

$$\theta_{n+1} = \theta_n - \gamma [H(\theta_n)]^{-1} \nabla_{\theta} L(\theta_n)$$

While theoretically superior, computing and inverting the $N \times N$ Hessian is prohibitive for large networks. **Quasi-Newton methods** build efficient numerical approximations of the inverse Hessian.

</div>

### Backpropagation Through Time (BPTT)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Backpropagation Through Time)</span></p>

BPTT is the standard algorithm for training RNNs. It conceptually transforms the RNN's temporal recursion into a spatial deep structure by "unwrapping" the network through its time steps. Each time step becomes a distinct layer, with **shared weights** across all layers.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Procedure)</span></p>

1. **Forward Pass:** Propagate activity from $t=1$ to $t=T$.
2. **Calculate Errors:** Compute deviation between prediction and target.
3. **Backward Pass:** Propagate error signals from $T$ to $1$, updating shared weights at each step.

BPTT is storage-efficient and linear in time complexity. Input/output configurations include sequence-to-sequence (time series modeling) and sequence-to-value (classification).

</div>

#### The Gradient Calculation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Total Loss)</span></p>

$L = \sum_t l_t$, where $l_t$ is the loss at time step $t$. The gradient decomposes as: $\frac{\partial L}{\partial \theta_i} = \sum_t \frac{\partial l_t}{\partial \theta_i}$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Decomposing the Gradient with the Chain Rule)</span></p>

Since parameters are reused at every time step, a parameter $\theta_i$ at early time $\tau$ influences the state at later time $t$. The full gradient:

$$\frac{\partial l_t}{\partial \theta_i} = \sum_{\tau=1}^{t} \frac{\partial l_t}{\partial x_t} \frac{\partial x_t}{\partial x_\tau} \frac{\partial x_\tau}{\partial \theta_i}$$

Where:

* $\frac{\partial l_t}{\partial x_t}$: Local gradient of loss w.r.t. output at time $t$.
* $\frac{\partial x_t}{\partial x_\tau}$: **Temporal Jacobian** — how state at $t$ depends on state at $\tau$. This is the crux of BPTT.
* $\frac{\partial x_\tau}{\partial \theta_i}$: Direct influence of the parameter on the state at $\tau$.

The temporal Jacobian decomposes as a product of single-step Jacobians:

$$\frac{\partial x_t}{\partial x_\tau} = \prod_{u=\tau+1}^{t} \frac{\partial x_u}{\partial x_{u-1}}$$

</div>

### The Exploding and Vanishing Gradient Problem

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

For a standard RNN with $x_t = \phi(W x_{t-1} + \dots)$, the single-step Jacobian is:

$$\frac{\partial x_t}{\partial x_{t-1}} = \text{diag}(\phi'(W x_{t-1} + \dots)) \cdot W$$

The temporal Jacobian becomes a product of these matrices, effectively raising $W$ to a power:

$$\frac{\partial x_t}{\partial x_\tau} = \prod_{u=\tau+1}^{t} \left( \text{diag}(\phi'(\dots)) \cdot W \right)$$

* **Exploding Gradients:** If eigenvalue magnitudes $> 1$ on average, the product grows exponentially. Training becomes unstable.
* **Vanishing Gradients:** If eigenvalue magnitudes $< 1$ on average, the product shrinks to zero. The network cannot learn long-range dependencies.

This is a fundamental obstacle deeply connected to Lyapunov exponents and stability analysis.

</div>

### Long Short-Term Memory (LSTM) Networks

LSTMs are designed to handle long-term dependencies by introducing a memory cell and gating mechanisms.

#### The Memory Cell Update

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Memory Cell Update)</span></p>

$$c_t = (f_t \odot c_{t-1}) + (i_t \odot \tanh(z_{t-1} + h_c))$$

Where $f_t$ is the forget gate, $i_t$ is the input gate, $\odot$ is the Hadamard product.

* **Forgetting:** $f_t \odot c_{t-1}$ filters what to keep from old memory.
* **Inputting:** $i_t \odot \tanh(z_{t-1} + h_c)$ determines what new information to add.

</div>

#### The Gating Mechanisms

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gate Equations)</span></p>

* **Forget Gate:** $f_t = \sigma(W_f z_{t-1} + h_f)$
* **Input Gate:** $i_t = \sigma(W_i z_{t-1} + h_i)$
* **Output Gate:** $o_t = \sigma(W_o c_{t-1} + h_o)$

where $\sigma(y) = \frac{1}{1 + e^{-y}}$ is the sigmoid function.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Final Output)</span></p>

$$z_t = o_t \odot \tanh(c_t)$$

The output gate controls what part of the internal memory is exposed to the next layer or time step.

</div>

#### Fundamental Design Principles

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Power of Linearity)</span></p>

The linear nature of the memory update through the forget gate ($f_t \odot c_{t-1}$) is critically important. If $f_t = 1$, the previous memory is passed through unmodified, allowing the network to preserve information across long time horizons. Linearity introduces a form of control that is difficult to achieve in purely nonlinear systems.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gating)</span></p>

The principle of using multiplicative interactions to control information flow is powerful and influential. It appears not only in LSTMs but also in modern architectures like Mamba.

</div>

#### Variants and Simplifications: GRUs

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The **Gated Recurrent Unit** (GRU), introduced by Cho et al. (2014), simplifies the LSTM by combining the forget and input gates into a single "update gate." GRUs aim to capture the essence of gated RNNs with a simpler architecture.

</div>
## Lecuture 10

### Revisiting Recurrent Networks and Their Challenges

This chapter provides a brief review of a fundamental challenge in training Recurrent Neural Networks (RNNs) and introduces the problem setting of Dynamical Systems Reconstruction, which motivates the advanced architectures discussed later.

#### The Exploding and Vanishing Gradient Problem

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


When training Recurrent Neural Networks using gradient descent, the primary objective is to minimize a loss function that quantifies the difference between the network's predictions and the true observed data. A common choice for this is the mean squared error over a time series.

The process of backpropagation through time, which is necessary to compute the gradients for an RNN, involves the chain rule. This results in a long product of Jacobian matrices, each representing the derivative of the latent state at one time step with respect to the latent state at the preceding time step.

The core of the exploding and vanishing gradient problem lies in this product. If the magnitudes (norms) of these Jacobian matrices are consistently greater than one, their product will grow exponentially, leading to an "exploding" gradient. Conversely, if their magnitudes are consistently less than one, the product will shrink exponentially towards zero, causing the gradient to "vanish." Both scenarios severely impede the network's ability to learn long-term dependencies in the data, as the influence of past states on the current loss becomes either overwhelmingly large or negligible.


</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Standard Loss Function)</span></p>

The optimization is typically performed on a loss function, $\mathcal{L}$, calculated as the sum of squared differences between the observed time series values, $X_t$, and the estimated values, $\hat{X}_t$, produced by the network's decoder or observation function.

$$\mathcal{L} = \sum_{t=1}^{T} \|X_t - \hat{X}_t\|^2$$

When minimizing this loss via gradient descent, we encounter terms derived from the chain rule involving products of Jacobians of the latent state transition function, $f$:

$$\frac{\partial z_t}{\partial z_{t-k}} = \frac{\partial z_t}{\partial z_{t-1}} \frac{\partial z_{t-1}}{\partial z_{t-2}} \cdots \frac{\partial z_{t-k+1}}{\partial z_{t-k}}$$

These products are the source of the exploding and vanishing gradient problem.

</div>

#### Dynamical Systems Reconstruction (DSR)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


Dynamical Systems Reconstruction (DSR) is the process of inferring the underlying mathematical model that generated an observed time series. The core assumption is that the data we collect is the product of some unknown, underlying dynamical system. We do not observe the true state of this system directly but rather through a measurement or observation function.

The goal of DSR is to use the observed time series data, $X_t$, to find a parameterized function, $f_\lambda$, that serves as a good approximation of the system's true, unknown flow operator. Simultaneously, we often need to estimate the observation function, $g_\lambda$, that maps the system's internal states to the measurements we can see. In the context of machine learning, an RNN is a natural candidate for this task: the network's recursive update rule acts as the approximate flow operator, and its output layer (or decoder) acts as the approximate observation function.


</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The DSR Setup)</span></p>


The problem of Dynamical Systems Reconstruction is formally defined by the following components:

1. Data Generating System: An unknown system whose state evolves over time according to a deterministic or stochastic rule. This evolution is governed by an unknown flow operator, $F$.
2. Observation Function: We do not have direct access to the true state of the system. Instead, we observe it through an unknown measurement function, $G$.
3. Observed Time Series: The sequence of measurements collected over time forms the time series data, denoted as $X_t$, for a duration of length $T$.
4. **Modeling Goal:** The objective is to estimate a parameterized function, $f_\lambda$, that approximates the true flow operator F, and a parameterized function, $g_\lambda$, that approximates the true observation function $G$. The parameters, denoted collectively by $\lambda$, are learned from the observed data $X_t$.

In the RNN framework, this translates to:

* The latent state update is an approximation of the flow operator: $z_t = f_\lambda(z_{t-1}, s_t)$, where $s_t$ are potential external inputs.
* The network output is generated by the observation function (decoder): $\hat{X}\_t = g_\lambda(z_t)$.

The overall aim is to estimate the functions $f_\lambda$ and $g_\lambda$ from the data.

</div>


### Reconstructing Dynamical Systems from Time Series

#### The Goal of Systems Reconstruction

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


In the study of dynamical systems, a central challenge is to move beyond general machine learning benchmarks and develop models that can accurately recreate the specific system that generated an observed set of time series data. When a model, such as a Recurrent Neural Network (RNN), is trained on data from a system (e.g., a bursting neuron model or human ECG data), it must learn the underlying rules governing its evolution. This task is distinct from simple prediction; the goal is to build a generative model that embodies the system's dynamics.

This endeavor often requires specialized training techniques and amendments to standard optimization procedures like gradient descent. The following sections will explore what it means to successfully "reconstruct" a dynamical system and how we can formally define and measure this success.


</div>

#### Qualitative Hallmarks of a Successful Reconstruction

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


Before diving into formal definitions, it is useful to establish an intuitive understanding of what a good reconstruction looks like. When we only have access to a finite, and often short, trajectory from a real system, a powerful model should be able to infer the global properties of the dynamics.

Key properties of a high-quality reconstruction include:

* Reconstruction of the Full Attractor: The model should be able to generate the complete extent of the system's attractor, even if it was only trained on a small portion of it. For instance, a network trained on a short segment of a trajectory from the chaotic Rössler system should learn to reproduce the entire Rössler attractor.
* Generalization to Nearby Initial Conditions: A successful model must not only replicate the trajectory it was trained on but also accurately predict the evolution of the system from initial conditions it has not seen before, provided they are within the same basin of attraction.

The ability to achieve this is non-trivial. It demonstrates that the model has learned the underlying governing equations, or a system topologically equivalent to them, rather than simply memorizing a single time series.

When working with real-world empirical data, a crucial prerequisite for these comparisons is to perform an optimal delay embedding. This ensures that the observed time series is represented in a state space that properly unfolds the system's dynamics, making a meaningful comparison to the model's generated state space possible. For developing and validating new methods, it is imperative to first test them on simulated data where the ground-truth governing equations are precisely known.


</div>

#### A Formal Definition of Reconstruction via Topological Conjugacy

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dynamical Systems Reconstruction)</span></p>


Let us formalize the concept of reconstruction.

Consider two dynamical systems, $D$ and $D^{\ast}$.

1. Let $D = (\mathbb{R}, \mathcal{R}, \phi)$ be the original, underlying system, where:
  * $\mathbb{R}$ is the set of time.
  * $\mathcal{R} \subseteq \mathbb{R}^m$ is an open set representing the state space.
  * $\phi$ is the flow operator that governs the system's evolution.
2. Let $D^{\ast} = (\mathbb{R}, \mathcal{R}^{\ast}, \phi^{\ast})$ be the candidate reconstruction (e.g., a trained RNN), where:
  * $\mathbb{R}$ is the set of time.
  * $\mathcal{R}^{\ast} \subseteq \mathbb{R}^m$ is an open set representing the model's state space.
  * $\phi^{\ast}$ is the model's transition rule or flow operator.

Further, let $A$ be an attractor of the original system $D$, with a corresponding basin of attraction $B \subseteq \mathcal{R}$.

We call the dynamical system $D^{\ast}$ a dynamical systems reconstruction of $D$ on the domain $B$ if the flow $\phi^{\ast}$ is topologically conjugate to the flow $\phi$ on $B$.


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


Recall that topological conjugacy implies a deep structural equivalence between two systems. It means that there exists a homeomorphism $g: B \to \mathcal{R}^{\ast}$ (a continuous, invertible function with a continuous inverse) that maps the original state space to the reconstructed one.

This conjugacy ensures that for any initial condition $x_0 \in B$, the trajectory generated by the original system, $x(t) = \phi(t, x_0)$, is topologically equivalent to the trajectory generated by the reconstructed system from the mapped initial condition, $x^{\ast}(t) = \phi^{\ast}(t, g(x_0))$. This equivalence must also preserve the parameterization by time.

In simple terms, if a model is topologically conjugate to the real system, it has perfectly captured the qualitative structure of the dynamics—it's like a stretched or compressed, but unbroken, version of the original.


</div>

#### Beyond Topology: Quantifying Geometric Similarity

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


While topological conjugacy provides a powerful definition of equivalence, it does not capture every aspect we might be interested in. Specifically, it is a topological definition and is insensitive to the geometric properties of the attractor. For a reconstruction to be truly useful, we often want the geometry of the generated attractor to be similar to that of the original.

To compare geometric properties, especially when the two systems may live in different state spaces ($\mathcal{R}$ and $\mathcal{R}^{\ast}$), we require a measure that can assess the similarity between their structures.


</div>

##### The Kullback-Leibler Divergence for State-Space Geometry

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


Real-world data is invariably noisy. Even for deterministic chaotic systems, it is often fruitful to adopt a statistical perspective and describe the system's behavior using an invariant measure, which describes the long-term probability of finding the system in a particular region of its state space.

We can therefore quantify the geometric similarity between the true system and our model by comparing the probability distributions they induce over their respective state spaces. A powerful tool for this is the Kullback-Leibler (KL) divergence.


</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(State-Space KL Divergence)</span></p>


The KL divergence measures the difference between two probability distributions. Let $P_{true}(x)$ be the probability distribution of states from the true underlying system, and let $P_{gen}(x)$ be the distribution of states generated by our model. The KL divergence is defined as:

$$D_{KL}(P_{true} \,\|\, P_{gen}) = \int_{\mathcal{D}} P_{true}(x) \log \left( \frac{P_{true}(x)}{P_{gen}(x)} \right) dx$$

where the integral is taken over the entire domain $\mathcal{D}$ of interest.

* If the distributions are identical ($P_{true}(x) = P_{gen}(x)$ everywhere), the fraction inside the logarithm is 1, making $\log(1) = 0$, and thus $D_{KL} = 0$.
* As the distributions diverge, the KL divergence becomes a positive value greater than zero.


</div>

##### Practical Estimation of KL Divergence

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


Calculating the integral in the KL divergence definition is often intractable. We must therefore rely on numerical estimation methods.

##### Method 1: Grid-Based Estimation (Binning)

A straightforward approach, analogous to the box-counting method for fractal dimension, is to discretize the state space.

1. Place a grid of $K$ bins (or boxes) over the state space.
2. Approximate the continuous probabilities $P(x)$ by estimating the relative frequencies $\hat{p}$ of observed data points that fall into each bin.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Binned KL Divergence Estimator)</span></p>


The estimator for the KL divergence becomes a sum over the $K$ bins:

$$\hat{D}_{KL} \approx \sum_{k=1}^{K} \hat{p}_{true}(k) \log \left( \frac{\hat{p}_{true}(k)}{\hat{p}_{gen}(k)} \right)$$

where $\hat{p}\_{true}(k)$ is the estimated probability of the true data being in bin $k$, and $\hat{p}\_{gen}(k)$ is the same for the generated data.


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


This method has a significant caveat: it performs poorly in high-dimensional state spaces due to the "curse of dimensionality," where the number of bins required to cover the space grows exponentially. Furthermore, one must be careful to ensure that no bin used in the calculation has a zero probability in the denominator ($\hat{p}_{gen}(k) = 0$), as this would cause the expression to diverge to infinity. This typically requires choosing a sufficiently large bin size or other regularization techniques.

##### Method 2: Gaussian Mixture Models (GMM)

An alternative and more sophisticated approach, particularly for higher-dimensional spaces, avoids a rigid global grid.

1. Instead of binning the entire space, define local $\epsilon$-neighborhoods along the observed trajectory.
2. Model the probability distribution within each of these neighborhoods using a Gaussian distribution.
3. The overall probability distribution is then represented as a Gaussian Mixture Model (GMM), which is a weighted sum of these individual Gaussians.

This GMM representation can then be used to estimate the probabilities $P_{true}(x)$ and $P_{gen}(x)$ needed for the KL divergence calculation. While the technical details are extensive, the core idea is to use a more flexible, data-driven method to model the probability distributions, which is more robust in high dimensions.

</div>


### Evaluating Dynamical Systems Reconstructions

When we train a model, such as a recurrent neural network, to replicate a dynamical system from data, a fundamental question arises: How do we measure success? When can we confidently say that our model is a good "dynamic systems reconstruction"? This chapter explores the nuances of this evaluation, highlighting the inadequacy of standard statistical measures for chaotic systems and introducing a more robust framework rooted in the principles of dynamical systems theory.

#### The Challenge of Assessing Performance in Chaotic Systems

The core challenge stems from a defining characteristic of many real-world systems: chaos. In chaotic systems, trajectories that begin from infinitesimally different initial conditions will diverge exponentially over time. This sensitivity to initial conditions has profound implications for evaluation.

* For a non-chaotic system, such as one that settles into a complex limit cycle, a well-trained model initialized on a true trajectory should be able to replicate that trajectory almost perfectly. A point-by-point comparison is meaningful.
* For a chaotic system, this is not the case. Even a perfect model of the Lorenz '63 system will produce a trajectory that quickly diverges from the original data it was trained on. Therefore, expecting a precise, long-term overlap in the time domain is not a sensible goal. Our evaluation must instead focus on whether the model has captured the underlying rules and structure of the system, not a specific path through its state space.

#### Why Traditional Metrics Fail: The Case of Mean Squared Error

In standard time series analysis and machine learning, the Mean Squared Error (MSE) is a ubiquitous metric for performance. However, for the reasons outlined above, it is fundamentally unsuited for evaluating reconstructions of chaotic systems.


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Deception of Mean Squared Error)</span></p>


The MSE is misleading when applied to chaotic systems. Consider two trajectories generated from the exact same Lorenz '63 system with the same parameters. Due to sensitivity to initial conditions, they will always diverge. Once this divergence occurs, the point-wise MSE between them will grow large, even though they both perfectly represent the same underlying dynamics.

An untrained or poorly trained recurrent neural network might produce a simple, non-chaotic oscillatory output. This simple output might, by chance, stay close to one major oscillation period of the true chaotic system for a short time, resulting in a deceptively low MSE. Conversely, a perfectly trained network that correctly captures the chaotic nature of the Lorenz attractor will produce a valid trajectory that, by definition, diverges from the training data, leading to a higher MSE.

This leads to a paradoxical and dangerous conclusion if one relies on MSE: the poorer model appears to perform better. This is an extremely important message to internalize. You cannot use out-of-the-box classical measures from statistics and machine learning to assess the quality of a dynamical systems reconstruction.


</div>


<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Visualizing the MSE Paradox)</span></p>


Consider reconstructions of the Lorenz '63 system produced by a recurrent neural network, where blue represents the original simulated trajectories and red represents the network's output.

| Reconstruction Quality | State Space Plot | Time Series | Mean Squared Error | Analysis |
|------------------------|-----------------|-------------|-------------------|----------|
| Poor                   | The model fails to capture the classic "butterfly" geometry of the Lorenz attractor. | The model only captures a single major oscillation pattern. | Low | The MSE is deceptively low because the model's simple, non-chaotic output happens to align with the training data for a short period before the true system diverges elsewhere. |
| Excellent              | The model's output traces the geometry of the Lorenz attractor almost perfectly. | The model's output is qualitatively similar but quickly diverges from the specific path of the blue trajectory. | High | The MSE is high precisely because the model has successfully learned the chaotic dynamics. Both the model and the true system are chaotic and thus their trajectories rapidly separate. |

This illustrates that to truly evaluate our models, we need measures that assess the similarity of the underlying temporal and geometric structures, not the point-wise agreement of specific trajectories.


</div>

#### A Dynamical Systems Approach to Evaluation

A robust evaluation framework assesses agreement on multiple fronts: the geometry of the attractor in state space and the general temporal structure of the signals.

##### Assessing Geometric Overlap

We want to confirm that the attractor generated by our model has the same shape and density as the true system's attractor.


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(From Topology to Geometry)</span></p>


We often desire more than just topological agreement (e.g., both attractors have one hole). We want geometrical agreement, meaning the shapes are quantitatively similar. A probabilistic formulation is particularly well-suited for this task. In dynamical systems theory, chaotic attractors are often described by invariant measures, which capture the probability of finding the system in a particular region of its state space. This approach is also robust to the noise that is always present in real-world data.


</div>


Metrics for assessing geometric overlap include:

* Vaserstein Distance: A measure of the distance between probability distributions.
* Kullback-Leibler (KL) Divergence: Another measure to quantify the difference between two probability distributions, often used to assess the overlap of the invariant measures on the attractors.

##### Assessing Temporal Structure

When we seek "temporal overlap," we do not mean the precise matching of time-domain patterns. Instead, we want to measure the similarity in the general temporal structure, such as characteristic frequencies and correlations.

Key tools for this include:

* Power Spectra: Derived from the Fourier Transform, the power spectrum reveals the dominant frequencies present in a signal.
* Autocorrelation Functions: These functions measure the correlation of a signal with a delayed copy of itself, revealing characteristic time scales and periodicities.

To quantify the similarity between the power spectra of a true signal and a reconstructed one, the Hellinger distance is a common and effective choice.


<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hellinger Distance for Power Spectra)</span></p>

Let $P_{true}(\omega)$ and $P_{recon}(\omega)$ be the power spectra of the true and reconstructed signals, respectively, as a function of frequency $\omega$. We first normalize both spectra such that their total area is one:

$$\int_{-\infty}^{\infty} P_{true}(\omega) \, d\omega = 1 \quad \text{and} \quad \int_{-\infty}^{\infty} P_{recon}(\omega) \, d\omega = 1$$

The Hellinger distance $H$ is then defined as:

$$H(P_{true}, P_{recon}) = \sqrt{1 - \int_{-\infty}^{\infty} \sqrt{P_{true}(\omega) P_{recon}(\omega)} \, d\omega}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Understanding the Hellinger Distance)</span></p>

* The integral term acts like a correlation measure in Fourier space. It measures the product of the "power" (square root of the power spectrum) at each frequency.
* The entire distance is a normalized measure bounded between 0 and 1.
* If the power spectra are identical, the integral becomes 1, and the distance $H$ is 0 (perfect agreement).
* If the spectra have no overlap, the integral is 0, and the distance $H$ is 1 (maximum disagreement).
* In practice, the integral is approximated numerically.

</div>


##### Other Dynamical Invariants

Beyond direct comparisons of geometry and spectra, we can compare fundamental quantitative properties—or invariants—of the dynamical systems.

* Lyapunov Exponents:
  * Maximum Lyapunov Exponent ($\lambda_{max}$): This measures the average exponential rate of divergence of nearby trajectories. A positive $\lambda_{max}$ is a hallmark of chaos. It can be estimated from real data (e.g., using a delay embedding) and computed directly from a trained model. Comparing these values is a powerful test of the model's learned dynamics.
  * Full Lyapunov Spectrum: This is the complete set of Lyapunov exponents. While very tricky to estimate from real data, it can be computed for a known model (like a trained RNN) from its Jacobians, providing a detailed fingerprint of the system's dynamics.
* Fractal Dimensionality: Chaotic attractors are often fractals. We can assess if our model's attractor has the same dimensionality as the true one.
  * Box-Counting Dimension: A straightforward method suitable for lower-dimensional data.
  * Correlation Dimension: A more practical estimator for higher-dimensional data, which describes the scaling behavior of data points within an $\epsilon$-ball placed along trajectories.
  * Kaplan-Yorke Dimension: An estimator for the fractal dimensionality that can be calculated from the Lyapunov spectrum.


<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Successful Reconstructions)</span></p>


Well-trained recurrent neural networks have been shown to successfully reconstruct a variety of dynamical systems:

* Chaotic Lorenz System: Models can capture the chaotic attractor, exhibiting similar power spectra and Lyapunov exponents, even though the time-domain trajectories diverge. In some cases, these models even correctly recover system properties like equilibria that were not explicitly present in the training data (which consisted only of trajectories on the attractor).
* Bursting Neuron Model (Limit Cycle): For this complex but non-chaotic system, a trained model can achieve near-perfect overlap in the time domain when initialized correctly.
* Higher-Dimensional Neural Population Models (Chaotic): Similar to the Lorenz system, successful reconstructions capture the general temporal and geometric structure without precise trajectory matching.

As a general rule, a visual representation of these various measures (KL divergence, power spectra agreement, Kaplan-Yorke dimension, $\lambda_{max}$) shows a clear trend: as the reconstruction quality improves from poor to excellent, all these dynamical measures show progressively better agreement.


</div>


### The Deep Connection Between Gradients and Chaos

In training recurrent neural networks, we often encounter the exploding and vanishing gradient problem. This issue, which can destabilize or stall the learning process, is not just a numerical quirk. For RNNs trained on dynamical systems, there is a profound and direct connection between the behavior of these gradients and the intrinsic stability properties of the system being modeled, as characterized by its Lyapunov spectrum. This connection was explored in detail in a 2020 NeurIPS paper by Yasin Abbasi-Asl, Zeinab Sadegh-Zadeh, and others.

#### Revisiting the Exploding & Vanishing Gradient Problem

The core of training an RNN involves backpropagation through time (BPTT), where the gradient of a loss function is calculated with respect to the network's parameters. This calculation involves a long chain of matrix multiplications, one for each time step. If the matrices in this chain are consistently larger than one, the gradients explode; if they are smaller than one, the gradients vanish. We will now show that the matrices involved are, in fact, the Jacobians of the system's dynamics.

#### Formalizing System Dynamics: The Lyapunov Spectrum

To understand a system's stability, we analyze how small perturbations evolve. The Lyapunov exponents quantify the average exponential rate of separation (or convergence) of nearby trajectories.


<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Maximum Lyapunov Exponent)</span></p>


Consider a generic discrete-time dynamical system described by a recursive map $f$:

$$z_t = f(z_{t-1})$$

where $z_t \in \mathbb{R}^m$ is the state of the system at time $t$. The evolution of an infinitesimal perturbation is governed by the product of the system's Jacobians along a trajectory. The maximum Lyapunov exponent, $\lambda_{max}$, is defined as:

$$\lambda_{max} = \lim_{T \to \infty} \frac{1}{T} \log \left\| \prod_{r=0}^{T-2} J(z_r) \right\|$$

where:

* $J(z_r) = \frac{\partial f(z_r)}{\partial z_r} = \frac{\partial z_{r+1}}{\partial z_r}$ is the Jacobian matrix of the map $f$ evaluated at state $z_r$.
* The product $\prod_{r=0}^{T-2} J(z_r)$ represents the accumulated linearization of the dynamics over $T-1$ steps.
* $\|\| \cdot \|\|$ denotes a matrix norm, such as the spectral norm, which yields the maximum singular value. Taking the maximum singular value gives us the exponent corresponding to the fastest rate of expansion.


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Product of Jacobians)</span></p>


The key takeaway here is the product series of Jacobians. This mathematical object is the heart of the Lyapunov exponent calculation. It tells us how the system expands or contracts volumes in its state space over long periods. As we are about to see, this exact same structure appears in the gradient calculation for RNNs.


</div>


#### Deconstructing the Learning Process: The Loss Gradient

Now, let's analyze the training process. We have a generic RNN, which can be an LSTM, a PLRNN, or any other type, described by a map $F$ with parameters $\theta$:

$$z_t = F(z_{t-1}, \theta)$$

(We omit external inputs for simplicity). We define a total loss $L$ which is the sum of losses $L_t$ at each time step, $L = \sum_t L_t$. Our goal is to compute the gradient of this loss with respect to a parameter $\theta$.


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Deriving the Loss Gradient)</span></p>


Let's derive the gradient $\frac{\partial L}{\partial \theta}$. By linearity, this is the sum of the gradients of the per-time-step losses:

$$\frac{\partial L}{\partial \theta} = \sum_t \frac{\partial L_t}{\partial \theta}$$

Now, we focus on a single term $\frac{\partial L_t}{\partial \theta}$. The loss at time $t$, $L_t$, depends on the network's parameters $\theta$ through the entire history of states $\lbrace z_1, z_2, \dots, z_t\rbrace$. Using the chain rule, we can express this dependency as a sum over all prior time steps $r \le t$:

$$\frac{\partial L_t}{\partial \theta} = \sum_{r=1}^{t} \frac{\partial L_t}{\partial z_t} \frac{\partial z_t}{\partial z_r} \frac{\partial z_r}{\partial \theta}$$

The crucial term here is $\frac{\partial z_t}{\partial z_r}$, which describes how a change in an earlier state $z_r$ affects a later state $z_t$. We can "unwrap" this term by applying the chain rule recursively:

$$\frac{\partial z_t}{\partial z_r} = \frac{\partial z_t}{\partial z_{t-1}} \frac{\partial z_{t-1}}{\partial z_{t-2}} \cdots \frac{\partial z_{r+1}}{\partial z_r}$$

Recognizing that each term $\frac{\partial z_{k+1}}{\partial z_k}$ is simply the Jacobian of the network's transition function, $J(z_k)$, we can write this as a product series:

$$\frac{\partial z_t}{\partial z_r} = \prod_{k=r}^{t-1} J(z_k)$$

Substituting this back into our expression for the full gradient, we see the complete structure. The gradient of the loss with respect to the parameters is a sum over time, and each term in that sum contains a product series of Jacobians.


</div>


#### The Fundamental Link: Jacobians in Dynamics and Learning

By comparing the definitions, the connection becomes strikingly clear:

| Concept                | Defining Formula                                                                 | Key Component           |
|------------------------|-----------------------------------------------------------------------------------|-------------------------|
| Max. Lyapunov Exponent | $\lambda_{max} = \lim_{T \to \infty} \frac{1}{T} \log \lvert \prod_{r=0}^{T-2} J(z_r) \rvert$ | Product of Jacobians   |
| Loss Gradient Term     | $\frac{\partial L_t}{\partial z_r} = \frac{\partial L_t}{\partial z_t} \left( \prod_{k=r}^{t-1} J(z_k) \right)$ | Product of Jacobians   |

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Deep Connection)</span></p>

The very same mathematical structure—the product of Jacobians along a trajectory—governs two seemingly different phenomena:

1. System Dynamics: The long-term product of Jacobians determines the Lyapunov spectrum, which tells us if the system is stable, periodic, or chaotic. A system with exponents whose magnitudes are greater than zero will cause perturbations to grow exponentially.
2. **Learning Dynamics:** The product of Jacobians over finite time horizons determines how gradients propagate backward in time during training. If the Jacobians correspond to an expanding (chaotic) system, their product will grow exponentially, leading to exploding gradients. If they correspond to a highly contracting system, their product will shrink exponentially, leading to vanishing gradients.

Therefore, the exploding and vanishing gradient problem is not merely a numerical issue to be solved with clever tricks; it is a direct reflection of the intrinsic stability properties of the dynamical system the network is trying to learn. This insight is critical for designing architectures and training methods that are truly suited for the complex task of dynamical systems reconstruction.

</div>


  * 2.1 Sparse Teacher Forcing: A Control-Theoretic Approach
    * 2.1.1 Core Idea and Formalism
    * 2.1.2 Optimal Forcing Interval and the Predictability Time
    * 2.1.3 Empirical Validation and Generative Properties
  * 2.2 Multiple Shooting: A Segmentation-Based Approach
    * 2.2.1 Core Idea and Formalism


### The Challenge of Modeling Chaotic Systems with Recurrent Networks

When attempting to use recurrent neural networks (RNNs) to reconstruct the dynamics of a system, a fundamental challenge arises if the underlying system is chaotic. The very nature of chaos, characterized by sensitive dependence on initial conditions, directly translates into a significant numerical problem during model training: the explosion of loss gradients. This section will detail the mathematical origins of this problem and explain why it is an unavoidable consequence of accurately modeling chaotic behavior.

#### The Gradient Recursion and the Product of Jacobians

To understand the problem, we must first recall how gradients are calculated in an RNN. The loss function, $L$, depends on the sequence of latent states, $\lbrace z_1, z_2, \dots, z_T\rbrace$, generated by the network. The gradient of the loss with respect to a state $z_t$ at a past time step involves a product of Jacobian matrices from that point forward in time. This relationship, derived from the chain rule of differentiation, can be expressed as:

$$\frac{\partial L}{\partial z_t} \propto \prod_{k=t}^{T-1} J_k$$


where $J_k$ is the Jacobian of the recurrent transition function $f$ at time step $k$. This product series is central to the issue of gradient stability.

#### The Inevitable Link Between Chaos and Exploding Gradients

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


A defining characteristic of a chaotic system is that its maximum Lyapunov exponent is greater than zero ($\lambda_{max} > 0$). This positive exponent signifies that nearby trajectories in the state space diverge exponentially over time. For an RNN to successfully capture this chaotic behavior, its internal dynamics, represented by the function $f$, must also exhibit this exponential divergence.

When the model successfully learns the chaotic dynamics, the Jacobian matrices $J_k$ in the product series will have singular values with absolute values greater than one. Consequently, as the product series extends over time, its norm will grow exponentially. This leads directly to an explosion in the magnitude of the loss gradients.

This is not a flaw in the model or a bug in the training process; it is a fundamental and inevitable consequence. If we demand that our RNN genuinely reconstructs the underlying chaotic system, it must produce chaos. If it produces chaos, its loss gradients will explode. We cannot avoid this problem if our goal is a true dynamical systems reconstruction, such as modeling the Lorenz system, which would be incomplete without its characteristic chaotic behavior.


</div>


### Techniques for Gradient Stabilization

Given that exploding gradients are an intrinsic property of training recurrent models on chaotic data, we must employ specialized techniques to manage them. These methods aim to regularize the training process, allowing the network to learn the system's long-term structure without being overwhelmed by numerical instability.

#### Sparse Teacher Forcing: A Control-Theoretic Approach

One classical and effective technique for managing gradient explosion is teacher forcing. In our context, we use a specialized version known as sparse teacher forcing, which can be understood as a technique from control theory applied to the training of a dynamical model.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


The core idea behind sparse teacher forcing is to allow the model-generated trajectory to evolve freely for a period, letting it explore the dynamics of the state space, but then periodically "pulling it back" towards the true trajectory observed in the data. This prevents the model's trajectory from diverging too far, which in turn prevents the associated gradients from exploding.

This method strikes a crucial balance. If we correct the trajectory at every single time step (classical teacher forcing), the model only learns to make one-step-ahead predictions and fails to capture the essential long-term properties and geometric structure of the system's attractor. If we never correct it, we encounter the exploding gradient problem. Sparse teacher forcing, by applying corrections at well-chosen intervals, allows the model to learn about long-term behavior while keeping the training process stable.

This idea has historical roots in papers by Williams and Zipser (1989) and Pearlmutter (1990).


</div>

##### Core Idea and Formalism

Let us formalize the sparse teacher forcing mechanism.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Model and Control Series Setup)</span></p>


1. Observed Data: We assume an observed time series of length $T$, denoted as $\lbrace x_1, x_2, \dots, x_T\rbrace$, where each $x_t \in \mathbb{R}^n$.
2. Recurrent Model: Our RNN generates a sequence of latent states $\lbrace z_1, z_2, \dots, z_T\rbrace$, where $z_t \in \mathbb{R}^m$, according to the transition function $z_{t+1} = f(z_t)$.
3. **Observation Model:** The observed data $x_t$ is related to the latent state $z_t$ through a decoder or observation function, $g$. For simplicity, let's consider a linear mapping:
  
  $$x_t = C z_t$$
  
  where $C$ is a matrix mapping from the $m$-dimensional latent space to the $n$-dimensional observation space.
4. **Control Series:** To "force" the model back to the real trajectory, we need an estimate of the "true" latent states directly from the data. We construct a control series, $\lbrace\tilde{z}_1, \tilde{z}_2, \dots, \tilde{z}_T\rbrace$, by inverting the decoder model. For our linear example, this can be achieved using the pseudo-inverse $C^+$:
   
  $$\tilde{z}_t = C^+ x_t  $$
  
  where $C^+$ is the pseudo-inverse of $C$.

More generally, this involves solving a regression problem to find an estimate of the latent state that corresponds to a given observation.


</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Sparse Forcing Update Rule)</span></p>


We define a set of forcing times, $\mathcal{T}$, at regular intervals determined by a forcing interval $\tau \in \mathbb{N}^+$.

$$\mathcal{T} = \lbrace n \cdot \tau \mid n \in \mathbb{N}\rbrace$$  

During training, the latent state $z_{t+1}$ is computed using a conditional rule:  

$$z_{t+1} = \begin{cases} f(\tilde{z}_t) & \text{if } t \in \mathcal{T} \ f(z_t) & \text{if } t \notin \mathcal{T} \end{cases}$$     


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(How Forcing Cuts Off Gradients)</span></p>


The key to stabilizing training lies in how this update rule affects backpropagation. At a forcing time step $t \in \mathcal{T}$, the input to the function $f$ is $\tilde{z}\_t$. Since $\tilde{z}\_t$ is derived directly from the data $x_t$, it is treated as a constant with respect to the previous latent state $z_{t-1}$. Therefore, the gradient chain is broken:  

$$\frac{\partial z_{t+1}}{\partial z_t} = 0 \quad \text{for } t \in \mathcal{T}$$  

This effectively "resets" the product of Jacobians every $\tau$ steps, preventing it from growing uncontrollably over long time horizons.


</div>

##### The Predictability Time: Choosing an Optimal Forcing Interval

The choice of the forcing interval $\tau$ is critical. An improperly chosen $\tau$ can either fail to solve the gradient problem or prevent the model from learning essential dynamics.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Optimal Forcing Interval)</span></p>


The optimal choice for the forcing interval $\tau$ is approximately given by the system's predictability time, which is inversely related to the maximum Lyapunov exponent $\lambda_{max}$. 

$$\tau_{optimal} \approx \frac{\ln(2)}{\lambda_{max}}$$ 


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>


This relationship is highly intuitive.

* A larger $\lambda_{max}$ implies a more chaotic system where trajectories diverge more quickly. This requires a smaller $\tau$ to provide more frequent corrections and keep the model's trajectory from straying too far.
* A smaller $\lambda_{max}$ implies a less chaotic system with slower divergence, allowing for a larger $\tau$ and giving the model more freedom to evolve on its own.

Empirical evidence shows that choosing $\tau$ is a delicate balance.

* If $\tau$ is too small, the system is over-regularized. It only learns one-step-ahead predictions and fails to capture the long-term, global structure of the dynamics.
* If $\tau$ is too large, the forcing is too infrequent to prevent the product of Jacobians from diverging, and the exploding gradient problem persists.


</div>

##### Empirical Results and Model Validation

The effectiveness of choosing $\tau$ based on the predictability time has been demonstrated on real-world data.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name"></span></p>


* Brain Recordings: This technique has been successfully applied to the reconstruction of dynamics from complex, real-world time series such as human fMRI and EEG data.
* Performance Metrics: By plotting metrics of dynamical similarity (e.g., state-space divergence or Heller distance) against different values of $\tau$, studies show that the error indeed reaches a minimum when $\tau$ is chosen to be near the predictability time of the empirical time series.


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Generative Nature of the Model)</span></p>

It is crucial to emphasize that a successfully trained RNN for dynamical systems reconstruction is a generative model. The patterns and trajectories it produces after training (e.g., simulated brain activity patterns) are not simply literal fits to the training data. Instead, they are novel generations synthesized by the model itself, demonstrating that it has learned the underlying rules and structure of the system's dynamics.

</div>

#### Multiple Shooting: A Segmentation-Based Approach

A related idea, originating from the dynamical systems literature, is a technique known as multiple shooting.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The core concept of multiple shooting is to divide the full time series into smaller, more manageable segments. Instead of trying to propagate a single trajectory from one initial condition across the entire time series, we estimate a new initial condition for each segment. The optimization process then involves two simultaneous goals:

1. Fit the trajectory within each individual segment.
2. Ensure that the trajectories are continuous across the boundaries of the segments (i.e., the end state of one segment matches the initial state of the next).

</div>

This approach inherently breaks the long-term dependencies that lead to exploding gradients by limiting the backpropagation horizon to the length of a single segment. A key reference for this method is a 2004 paper by Fattal et al. in Physical Review.

##### Formalism: The Optimization Problem

Let's formalize the objective for multiple shooting.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multiple Shooting Loss Function)</span></p>

1. We partition the time series into N segments.
2. For each segment $n \in \lbrace 1, \dots, N\rbrace$, we estimate a unique initial condition, $z_0^{(n)}$, which becomes a trainable parameter.
3. The overall objective is to minimize a loss function that sums the errors over all segments. A typical squared-error loss would be:

$$\min_{\theta, \lbrace z_0^{(n)}\rbrace_{n=1}^N} \sum_{n=1}^{N} \sum_{t=1}^{T_{seg}} \left\| x_t^{(n)} - g(z_t^{(n)}) \right\|^2$$

where $\theta$ represents the parameters of the recurrent model $f$, $T_{seg}$ is the length of each segment, and the evolution of $z_t^{(n)}$ depends on the initial condition $z_0^{(n)}$. The constraint of temporal continuity between segments must also be enforced, often through additional terms in the loss function or as part of the optimization procedure.

</div>

#### Multiple Shooting for Gradient Stability

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Challenge of Long Time Series)</span></p>

When training models of dynamical systems on long time series, propagating gradients back through many time steps can lead to the well-known problems of exploding or vanishing gradients. This makes it difficult for gradient descent-based optimizers to find good parameters. One effective strategy to mitigate this is Multiple Shooting. The core idea is to break the long time series into smaller, more manageable segments or "batches." We then train the model on these individual segments, but with an additional constraint that ensures the trajectory remains continuous across the segment boundaries.


</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multiple Shooting with a Continuity Constraint)</span></p>


In the multiple shooting framework, we divide the time series into segments. For each segment $n$, we optimize an initial condition, which we can denote as $z_0^{(n)}$. The standard loss is computed over each segment.

To ensure continuity, we introduce a constraint: the initial condition of a segment, $z_0^{(n+1)}$, must match the state that results from forward-propagating the dynamics from the end of the previous segment. This is enforced through a regularization term added to the primary loss function.

The total loss, $\mathcal{L}_{\text{total}}$, is formulated as:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{usual}} + \lambda \mathcal{L}_{\text{reg}}$$

where:
* $\mathcal{L}_{\text{usual}}$ is the standard reconstruction or prediction loss.
* $\lambda$ is a regularization parameter that controls the strength of the continuity constraint.
* $\mathcal{L}_{\text{reg}}$ is the regularization loss that penalizes discontinuities between segments.

The regularization loss enforces the condition that the initial condition of segment $n+1$ must be close to the forward-propagated state from the end of segment $n$. If a segment has a length of $T$ time steps, this can be written as:

$$\mathcal{L}_{\text{reg}} = \sum_{n} \left\| z_0^{(n+1)} - f_{\theta}^T(z_0^{(n)}) \right\|^2$$

Here, $z_0^{(n+1)}$ is the estimated initial condition for segment $n+1$, and $f_{\theta}^T(z_0^{(n)})$ represents the state obtained by applying the learned dynamics function, $f_{\theta}$, for $T$ time steps, starting from the initial condition of the previous segment, $z_0^{(n)}$.


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Knitting Trajectories Together)</span></p>


This technique can be visualized as learning short trajectory pieces independently and then "knitting them together." By enforcing continuity across the boundaries of these shorter intervals, we can reconstruct a globally consistent long-term trajectory while keeping the backpropagation paths short and the gradients stable.


</div>


#### A Control-Theoretic Perspective on Training

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(From Observation to Control)</span></p>

Control Theory is a branch of mathematics focused on influencing the behavior of dynamical systems. Instead of merely observing a system's evolution, the goal is to find optimal control signals that can steer the system towards a desired state or behavior. For example, a control signal might be used to stabilize a chaotic system onto a stable limit cycle.

This concept is directly relevant to training neural models of dynamical systems. Techniques like teacher forcing can be viewed through a control-theoretic lens. In this context, the loss gradients act as the control signal, actively pushing the model's trajectory towards the true data trajectory.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(A Biophysical Neuron Model)</span></p>


Consider a biophysical model of a spiking neuron with a voltage signal and two gating variables. We can introduce a control term directly into the system's differential equations. This term is proportional to the difference between the observed voltage from a real cell and the voltage produced by the model:

$$\frac{dV}{dt} = \dots + \kappa (V_{\text{observed}} - V_{\text{model}})$$

Here, $\kappa$ is a parameter that determines the strength of the control. When attempting to estimate the system's parameters from real voltage recordings, this control term serves two purposes:

1. Guidance: Much like sparse teacher forcing, it guides the model's trajectory, preventing it from diverging wildly from the real data.
2. **Loss Landscape Smoothing:** For a sufficiently large $\kappa$, this control term can smooth out the loss function, making it more convex. A more convex loss landscape is significantly easier for gradient-based optimizers to navigate, reducing the risk of getting stuck in poor local minima.

During training, the control parameter $\kappa$ would need to be regulated, similar to how teacher forcing schedules are used. This approach provides a foundation for more advanced techniques.


</div>


#### Generalized Teacher Forcing

Introduced in a 2023 ICML paper by Florian Hess et al., Generalized Teacher Forcing is a powerful technique that formalizes the control-theoretic approach in a graded, adaptive manner.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Weighted Average State)</span></p>


Unlike sparse teacher forcing, which replaces the model's state with the true state at certain time steps, generalized teacher forcing creates a new state, $\tilde{z}_t$, which is a weighted average of the model-propagated state and an estimate from the true data.


</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Teacher Forcing)</span></p>

Generalized teacher forcing creates a new state, $\tilde{z}_t$, which is a weighted average of the model-propagated state and an estimate from the true data.

The new state at time $t$ is defined as:

$$\tilde{z}_t = (1 - \alpha) f_{\theta}(\tilde{z}_{t-1}) + \alpha \hat{z}_t$$

where:
* $\tilde{z}_t$ is the new, "controlled" state at time $t$.
* $f_{\theta}(\tilde{z}\_{t-1})$ is the one-step forward propagation of the previous state using the learned dynamics model $f_{\theta}$.
* $\hat{z}_t$ is the latent state estimated from the real data observation $x_t$ at time $t$, typically obtained by inverting a decoder model.


* $\alpha$ is a weighting parameter between 0 and 1 that balances the influence of the model's internal dynamics and the guidance from the real data.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(An Old Idea, Revisited)</span></p>


This idea of blending model dynamics with data was first introduced by Kenji Doya in 1992. However, it remained dormant for a long time, partly because a principled method for choosing the mixing parameter $\alpha$ was not established. The key contribution of the modern approach is to provide a smart, adaptive choice for $\alpha$.


</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(The Product Series of Jacobians)</span></p>


To understand how to choose $\alpha$, we must analyze the product series of Jacobians, which governs gradient flow through time. The Jacobian of the map from $\tilde{z}_{t-1}$ to $\tilde{z}_t$ is crucial.

Let's compute the derivative of $\tilde{z}\_t$ with respect to $\tilde{z}\_{t-1}$:

$$J_t = \frac{\partial \tilde{z}_t}{\partial \tilde{z}_{t-1}}$$

Applying the chain rule to the definition of $\tilde{z}_t$:

$$\frac{\partial \tilde{z}_t}{\partial \tilde{z}_{t-1}} = \frac{\partial}{\partial \tilde{z}_{t-1}} \left[ (1 - \alpha) f_{\theta}(\tilde{z}_{t-1}) + \alpha \hat{z}_t \right]$$

Since $\hat{z}_t$ is derived directly from the data $x_t$ and does not depend on the previous model state $\tilde{z}\_{t-1}$, its derivative is zero. This leaves:

$$J_t = (1 - \alpha) \frac{\partial f_{\theta}(\tilde{z}_{t-1})}{\partial \tilde{z}_{t-1}}$$

Let's call the Jacobian of the underlying dynamics model $G_t = \frac{\partial f_{\theta}(\tilde{z}\_{t-1})}{\partial \tilde{z}\_{t-1}}$. Then, the Jacobian of the controlled system is simply:

$$J_t = (1 - \alpha) G_t$$    

The product of Jacobians over $T$ time steps, which determines the magnitude of backpropagated gradients, is therefore:

$$\prod_{k=0}^{T-1} J_{T-k} = \prod_{k=0}^{T-1} (1 - \alpha_{T-k}) G_{T-k}$$  


</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(An Optimal Choice for $\alpha$)</span></p>


The derivation above reveals a clear strategy for controlling gradient flow. The magnitude of the gradients is determined by the singular values of the Jacobian product. If the largest singular value is much greater than 1, gradients will explode. If it is much less than 1, they will vanish.

This suggests that $\alpha$ should be chosen at each time step to regulate the singular values of the individual Jacobians, $G_t$. A smart choice for $\alpha$ at time $t$ is one that forces the norm of the controlled Jacobian, $J_t$, to be close to 1. We can achieve this by setting:

$$1 - \alpha_t = \frac{1}{\sigma_{\max}(G_t)}$$

which implies:

$$\alpha_t = 1 - \frac{1}{\sigma_{\max}(G_t)}$$

where $\sigma_{\max}(G_t)$ is the maximum singular value of the dynamics Jacobian $G_t$. By choosing $\alpha$ in this way, we can adaptively keep the gradients in check, preventing both exploding and vanishing gradient problems.


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Implementation)</span></p>


A significant caveat is that computing a full Singular Value Decomposition (SVD) to find $\sigma_{\max}$ at every single time step is computationally prohibitive. However, this theoretical result provides the foundation for practical approximations. In practice, one can:

* Use computationally efficient proxies for the SVD.
* Update the value of $\alpha$ less frequently, for instance, every 10 time steps, rather than at every step.

This makes the automatic, adaptive regulation of gradients feasible for training on real-world systems. Furthermore, an optimal choice of $\alpha$ not only stabilizes gradients but also tends to make the loss landscape nearly convex, greatly aiding the optimization process.


</div>


#### Summary of Techniques

To effectively reconstruct dynamical systems, especially from real-world data, it is crucial to employ techniques that manage gradient flow during training and allow the model to explore the state space. The key methods discussed are:

* Multiple Shooting: This technique addresses the exploding/vanishing gradient problem in long sequences by breaking the trajectory into shorter segments and enforcing continuity between them with a regularization term.
* Sparse Teacher Forcing: (Referenced as a point of comparison) This involves occasionally replacing the model's predicted state with the ground truth state, providing a strong corrective signal.
* Generalized Teacher Forcing: A more sophisticated, control-theoretic method that creates a new state at each time step as a weighted average of the model's prediction and a data-derived estimate. The weighting parameter, $\alpha$, can be chosen adaptively to regulate the singular values of the system's Jacobians, ensuring stable gradients and leading to a smoother, more convex loss landscape.


[SINDy](/subpages/dynamical-systems/dynamical-systems-in-machine-learning/sindy/)


