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

**Table of Contents**
- TOC
{:toc}

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

## Lecture 1

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

### 1. Is your reasoning about sums of exponentials correct?

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

## Lecture 2

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

$$x_{n+1} = f(x_n),$$

a fixed point (x^*) satisfies

$$f(x^*) = x^*$$

An **equilibrium** usually means a state where nothing changes in time.

For a continuous-time system

$$\dot{x} = g(x),$$

an equilibrium (x^*) satisfies

$$g(x^*) = 0$$

So the difference is mostly about language and context:

* **fixed point** is the standard term for maps and discrete-time systems
* **equilibrium** is the standard term for ODEs and continuous-time systems

They both describe a state that stays where it is once the system reaches it.

There is also a nice connection between them. If $x^\ast$ is an equilibrium of a flow, then starting at $x^\ast$ gives a constant trajectory, so it is also a fixed point of the time-$t$ flow map for every $t$.

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


## Lecture 4

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

This section transitions our focus from continuous flows to discrete maps, exploring their unique behaviors, such as fixed points and cycles. We will begin with a foundational example that is famous for its complexity and its role in the history of chaos theory: the logistic map.

#### The Logistic Map: A Canonical Example

The logistic map is a simple, scalar (one-dimensional) map defined by a quadratic equation. It was famously analyzed by Robert May in a 1976 Nature paper, which highlighted how such a simple deterministic equation could produce extraordinarily complex, chaotic dynamics.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">The Logistic Map</span></p>

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
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Visualizing the Map with a Return Plot</span></p>

To understand the behavior of a map, we use a return plot, which graphs $x_{t+1}$ as a function of $x_t$. For the logistic map, this function is a parabola opening downwards. We also plot the line $x_{t+1} = x_t$, known as the bisector or identity line.

The intersections of the map's curve with the bisector are significant, as they represent points where the input equals the output — these are the fixed points of the system. We can trace the evolution of the system graphically using a "cobweb plot":

1. Start at an initial value $x_0$ on the horizontal axis.
2. Move vertically to the parabola to find the value of $x_1$.
3. Move horizontally from the parabola to the bisector. The corresponding point on the horizontal axis is now $x_1$.
4. Repeat the process: move vertically to the parabola to find $x_2$, horizontally to the bisector, and so on.

This graphical method provides a powerful intuition for whether the system converges to a fixed point, diverges, or enters a cycle.

</div>

#### Fixed Points and Their Stability

A fixed point is a state of the system that does not change over time. It is a point where, if the system starts there, it stays there.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">Fixed Point</span></p>

A point $x^\ast$ is a fixed point of a map $f$ if it satisfies the condition:

$$x^* = f(x^*)$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">Finding the Fixed Points of the Logistic Map</span></p>

To find the fixed points of the logistic map, we set $x_{t+1} = x_t = x^\ast$ and solve the resulting equation:

$$x^* = \alpha x^* (1 - x^*)$$

Rearranging this gives us a quadratic equation:

$$\alpha (x^*)^2 + (1 - \alpha) x^* = 0$$

We can factor out $x^*$:

$$x^* (\alpha x^* + 1 - \alpha) = 0$$

This equation yields two solutions for the fixed points:

1. $x_1^\ast = 0$
2. $\alpha x^\ast + 1 - \alpha = 0 \implies x_2^\ast = \frac{\alpha - 1}{\alpha}$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Existence of Fixed Points</span></p>

From these solutions, we can immediately see two things:

* The fixed point at $x^* = 0$ exists for all values of $\alpha$.
* The second fixed point, $x_2^* = \frac{\alpha - 1}{\alpha}$, only exists within our interval of interest $[0, 1]$ if the parameter $\alpha$ is greater than or equal to $1$.

The return plot illustrates two scenarios. For a small $\alpha$, only one fixed point exists at $x^* = 0$. For a larger $\alpha$, a second fixed point appears where the parabola intersects the bisector.

A cobweb plot starting near the origin converges to the fixed point at $x^* = 0$ when the slope of the map at this point has an absolute value less than $1$, suggesting the point is stable. For a larger $\alpha$, a cobweb plot starting near the origin is repelled from it, while a plot starting near the second fixed point converges towards it. The slope at $x^* = 0$ is now steep (absolute value greater than $1$), indicating it is unstable, while the slope at the second point is shallower, indicating it is stable.

This graphical analysis suggests that the stability of a fixed point is determined by the slope of the map at that point.

</div>

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
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">N-Dimensional Maps and the Jacobian</span></p>

Consider a system in $m$ dimensions described by a map $\mathbf{x}_{t+1} = \mathbf{F}(\mathbf{x}_t)$, where $\mathbf{x}_t$ is a vector in $\mathbb{R}^m$. The stability analysis is analogous, but the scalar derivative is replaced by the Jacobian matrix, $J$.

The Jacobian matrix is the matrix of all first-order partial derivatives of the vector-valued function $\mathbf{F}$:

$$J = \begin{pmatrix} \frac{\partial F_1}{\partial x_1} & \cdots & \frac{\partial F_1}{\partial x_m} \\ \vdots & \ddots & \vdots \\ \frac{\partial F_m}{\partial x_1} & \cdots & \frac{\partial F_m}{\partial x_m} \end{pmatrix}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">Stability for N-D Maps</span></p>

Let $\mathbf{x}^\ast$ be a fixed point of the map $\mathbf{F}(\mathbf{x})$. The stability of $\mathbf{x}^\ast$ is determined by the eigenvalues of the Jacobian matrix evaluated at the fixed point, $J(\mathbf{x}^\ast)$.

* The fixed point $\mathbf{x}^\ast$ is stable if the maximum absolute value (or modulus, for complex eigenvalues) of all eigenvalues of $J(\mathbf{x}^\ast)$ is less than $1$.

$$\max_i |\lambda_i| < 1$$

* The fixed point $\mathbf{x}^\ast$ is unstable if the maximum absolute value of any eigenvalue of $J(\mathbf{x}^\ast)$ is greater than $1$.

$$\max_i |\lambda_i| > 1$$

* If the maximum absolute value of the eigenvalues is exactly equal to $1$ (i.e., the largest eigenvalue lies on the unit circle in the complex plane), the system is non-hyperbolic, and a linear stability analysis is inconclusive.

</div>

#### The Emergence of K-Cycles

What happens when all fixed points in a bounded system become unstable? The trajectory cannot settle into a fixed point, but it also cannot escape to infinity. The system must find another form of stable, persistent behavior. In maps, this often leads to the emergence of cycles.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">The Path to a 2-Cycle</span></p>

Consider the logistic map for $\alpha > 3$. At these parameter values, the slopes at both fixed points ($x^* = 0$ and $x^* = (\alpha - 1)/\alpha$) are greater than $1$ in absolute value. This means both fixed points are unstable.

Since we know the system is confined to the interval $[0, 1]$, the trajectory must go somewhere else. This "somewhere else" is often a cycle, where the system visits a finite sequence of points repeatedly.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">K-Cycle</span></p>

A $K$-cycle is a periodic trajectory where the system iterates through $K$ distinct points. A 2-cycle, for example, is a pair of points $\lbrace x_a, x_b \rbrace$ such that:

$$f(x_a) = x_b \quad \text{and} \quad f(x_b) = x_a$$

The system perpetually jumps back and forth between these two points.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">A 2-Cycle in the Logistic Map</span></p>

For a parameter value of $\alpha = 3.3$, the logistic map exhibits a stable 2-cycle.

* If we trace the evolution of the system with a cobweb plot, we see that the trajectory, instead of spiraling into a single fixed point, converges to a rectangular box that bounces between two distinct values.
* A time-series plot of $x_t$ versus $t$ would show the system initially behaving transiently before settling into a steady oscillation between two values.

This behavior, where an increase in a parameter causes a stable fixed point to lose stability and give rise to a stable 2-cycle, is a common route to more complex dynamics in nonlinear systems.

</div>

### Cycles in Iterated Maps

#### Two-Cycles as Fixed Points of the Iterated Map

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">From Cycles to Fixed Points</span></p>

When analyzing discrete maps, we often encounter cycles, where the system iterates between a set of distinct points. A two-cycle, for instance, involves iterating between two different points. If we start at one point, a single application of the map takes us to the second point, and the next application takes us back to the first.

This observation leads to a powerful insight: a point on a two-cycle returns to its original position after exactly two applications of the map. Therefore, any point belonging to a two-cycle of a map $f$ must be a fixed point of the twice-iterated map, denoted as $f^2(x) = f(f(x))$. This reframes the problem of finding cycles into the more familiar problem of finding fixed points, albeit for a more complex function.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">The Logistic Map Iterated Twice</span></p>

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
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Stability via the Iterated Map</span></p>

Since we can treat the points of a cycle as fixed points of an iterated map, we can determine the stability of the cycle by analyzing the stability of these corresponding fixed points. The method is the same as for simple fixed points: we check the slope of the function at the fixed point.

For a two-cycle of the map $f$, its stability is determined by the derivative of the twice-iterated map, $f^2(x)$, at the points of the cycle.

The plot of $f^2(x)$ for the logistic map at $\alpha = 3.3$ reveals four fixed points. Two of these are the original, now unstable, fixed points of $f(x)$. The other two are the points that constitute the stable two-cycle.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">Stability of a Cycle</span></p>

The stability of a $k$-cycle can be determined by checking the slope of the $k$-times iterated function, $f^k(x)$, at any point $x_i^*$ on the cycle. The cycle is stable if the absolute value of this slope is less than one.

$$\left| \frac{d}{dx} f^k(x) \bigg|_{x=x_i^*} \right| < 1$$

The cycle is unstable if this value is greater than one.

</div>

#### Generalization to k-Cycles

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">k-Cycle</span></p>

For a continuous map $f$, a $k$-cycle is a set of $k$ distinct points, $\lbrace x_1^\ast, x_2^\ast, \ldots, x_k^\ast \rbrace$, which are visited sequentially by iteration of $f$. This implies two critical conditions:

1. **Fixed Point of the Iterated Map:** Each point $x_i^\ast$ in the set (for $i = 1, \ldots, k$) is a fixed point of the $k$-times iterated map.

$$x_i^* = f^k(x_i^*)$$

2. **Minimality and Distinctness:** To be a true $k$-cycle, two additional constraints must be met:
   * $k$ must be the smallest integer for which the fixed-point condition holds. This ensures that a two-cycle is not misidentified as a four-cycle, for example.
   * All points in the set must be distinct: $x_i^\ast \neq x_j^\ast$ for all $i \neq j$. This prevents a lower-order cycle (like a fixed point where $x_1^\ast = x_2^\ast$) from being classified as a higher-order cycle.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Analytical Advantage of Iterated Maps</span></p>

This framework provides a significant advantage. While the original map $f$ might be complex and its iterated version $f^k$ even more so, we at least have a closed-form expression. This closed form gives us direct, analytical access to the stability of cycles. This is a powerful tool that is not generally available for analyzing the stability of limit cycles in continuous-time systems described by differential equations.

</div>

#TODO: MISSING PART: POINCARE MAP AND CONNECTION BETWEEN DISCRETE MAPS AND FLOWS!!!

### The Phase Description of Oscillators

In the study of dynamical systems, oscillators represent a fundamental class of behaviors characterized by periodic motion, often visualized as a closed orbit or limit cycle in the state space. To analyze these systems, particularly when they interact, it is incredibly useful to reduce their complex dynamics to a single, essential variable: the phase.

#### The Phase Variable

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Phase Variable)</span></p>

The phase of an oscillator describes its position along its limit cycle. It is represented by a **phase variable**, typically denoted as $\theta$. A single full iteration of the oscillator corresponds to the phase variable completing a full cycle. By convention, the phase is often defined to evolve in the interval $[0, 2\pi]$, though other intervals such as $[0, 1]$ are also used.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Phase Dynamics)</span></p>

The central idea of this approach is to shift our focus from the full state-space variables of the oscillator to just its phase. By doing this, we can formulate a new, often simpler, differential equation that describes the evolution of the phase itself:

$$\dot{\theta} = f(\theta)$$

This simplification allows us to capture the essential timing and rhythm of the oscillator, which is paramount when studying phenomena like synchronization.

</div>

#### A Simple Oscillator: Constant Angular Velocity

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Constant Angular Velocity)</span></p>

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

#### Calculating the Oscillation Period

If we have the differential equation for the phase, $\dot{\theta} = f(\theta)$, we can derive a formula to calculate the temporal period of one full oscillation, $T_{\text{osc}}$.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Derivation of the Oscillation Period)</span></p>

The period $T_{\text{osc}}$ is, by definition, the time it takes to complete one cycle. We can express this with a simple integral:

$$T_{\text{osc}} = \int_0^{T_{\text{osc}}} dt$$

To connect this to the phase variable, we perform a change of variables from time $t$ to phase $\theta$. As time progresses from $0$ to $T_{\text{osc}}$, the phase progresses from $0$ to $2\pi$. We can introduce $d\theta$ into the integral:

$$T_{\text{osc}} = \int_0^{2\pi} \frac{dt}{d\theta} \, d\theta$$

We know the differential equation for the phase is $\frac{d\theta}{dt} = f(\theta)$. Therefore, its inverse is $\frac{dt}{d\theta} = \frac{1}{f(\theta)}$. Substituting this into the integral gives the final formula:

$$T_{\text{osc}} = \int_0^{2\pi} \frac{1}{f(\theta)} \, d\theta$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A Direct Recipe)</span></p>

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
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Torus as State Space)</span></p>

Since each oscillator is described by a variable on a circle (from $0$ to $2\pi$), the combined state space of two such oscillators, $(\theta_1, \theta_2)$, can be visualized as a torus (a donut shape). One phase variable, say $\theta_1$, represents motion around the main radius of the torus, while the other, $\theta_2$, represents motion around the circular cross-section.

A trajectory in this two-dimensional state space represents the simultaneous evolution of both phases. As both oscillators cycle, the combined state $(\theta_1(t), \theta_2(t))$ traces a path that coils around the surface of the torus. A central question arises: Under what conditions will this trajectory eventually return to its starting point, forming a closed orbit on the torus?

</div>

#### Closed Orbits and Rational Frequency Ratios

A trajectory on the state-space torus will be a closed orbit if and only if the frequencies of the two oscillators are commensurate.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Commensurate Frequencies)</span></p>

Two frequencies, $\omega_1$ and $\omega_2$, are commensurate if their ratio is a rational number. That is, there exist two integers, $p$ and $q$, such that:

$$\frac{\omega_1}{\omega_2} = \frac{p}{q}$$

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

If the ratio of the oscillator frequencies $\frac{\omega_1}{\omega_2}$ is an irrational number, the trajectory on the torus will never close. Instead, it will wind around indefinitely, eventually passing arbitrarily close to every point on the surface of the torus. This motion is called quasi-periodic.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A Middle Ground Between Periodicity and Chaos)</span></p>

Quasi-periodic motion is an interesting middle ground between simple periodic behavior (like a limit cycle) and the more complex behavior of chaos. The system is "almost" periodic, but because the frequencies never align perfectly, the trajectory never repeats. Over time, the path of the system state will densely fill the entire surface of the torus. This is a unique property that arises in systems of two or more oscillators.

</div>

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

We can formalize this concept with a mathematical model. Let's assume we have two oscillators with their own intrinsic (natural) frequencies, $\omega_1$ and $\omega_2$. We then add a coupling term that depends on the difference between their phases.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Coupled Phase Oscillator Model)</span></p>

A common model for two **coupled oscillators** is given by the following system of differential equations:

$$\dot{\theta}_1 = \omega_1 + A \sin(\theta_1 - \theta_2)$$

$$\dot{\theta}_2 = \omega_2 + A \sin(\theta_2 - \theta_1)$$

Here, $A$ is the coupling strength, which determines how strongly the oscillators influence each other. The function $\sin(\cdot)$ is chosen as a simple periodic function to model the phase-dependent interaction, but other functions could be used.

</div>

#### The Phase Difference Equation

To analyze whether these oscillators will synchronize, the most effective technique is to study the evolution of their phase difference.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Phase Difference)</span></p>

The **phase difference**, $\phi$, between the two oscillators is defined as:

$$\phi = \theta_1 - \theta_2$$

If the oscillators synchronize perfectly (zero phase locking), this difference will be constant at $\phi = 0$. If they lock at a different, but still constant, phase relationship, $\phi$ will be a non-zero constant. Therefore, synchronization corresponds to the phase difference $\phi$ reaching a stable fixed point.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Derivation of the Phase Difference Equation)</span></p>

We can derive a differential equation for $\phi$ by differentiating its definition with respect to time:

$$\dot{\phi} = \dot{\theta}_1 - \dot{\theta}_2$$

Now, substitute the model equations for $\dot{\theta}_1$ and $\dot{\theta}_2$:

$$\dot{\phi} = \left( \omega_1 + A \sin(\theta_1 - \theta_2) \right) - \left( \omega_2 + A \sin(\theta_2 - \theta_1) \right)$$

Group the terms:

$$\dot{\phi} = (\omega_1 - \omega_2) + A \sin(\theta_1 - \theta_2) - A \sin(\theta_2 - \theta_1)$$

Using the trigonometric identity that sine is an odd function, $\sin(-x) = -\sin(x)$, we have $\sin(\theta_2 - \theta_1) = -\sin(\theta_1 - \theta_2)$. Substituting this into the equation:

$$\dot{\phi} = (\omega_1 - \omega_2) + A \sin(\theta_1 - \theta_2) - A \left( -\sin(\theta_1 - \theta_2) \right)$$

$$\dot{\phi} = (\omega_1 - \omega_2) + A \sin(\theta_1 - \theta_2) + A \sin(\theta_1 - \theta_2)$$

Finally, by substituting $\phi = \theta_1 - \theta_2$, we arrive at the differential equation for the phase difference:

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

Two or more oscillators are said to be synchronized or phase-locked when the difference between their phases, $\phi$, becomes constant over time. Mathematically, this corresponds to a stable fixed point of the phase difference dynamics. If $\dot{\phi} = 0$ for some phase difference $\phi^*$, the oscillators have achieved **phase locking**.

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

Arnold tongues are regions in a parameter space that depict where phase locking of a specific $P$:$Q$ ratio occurs. Typically, this space is plotted with the difference in intrinsic frequencies ($\omega_1 - \omega_2$) on one axis and the coupling amplitude ($a$) on the other. Each "tongue" represents a combination of parameters for which the system will synchronize in a particular mode (e.g., 1:1, 1:2, 2:1).

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
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Explicit Euler Method)</span></p>

The **explicit (or forward) Euler method** is the simplest numerical method for solving an initial value problem of the form $\dot{x} = f(x, t)$ with $x(0) = x_0$. The method discretizes time into steps of size $\delta_t$ and approximates the solution at each step using the tangent line:

$$x(t + \delta_t) = x(t) + \int_{t}^{t + \delta_t} f(x(\tau))d\tau$$

$$x(t + \delta_t) \approx x(t) + \delta_t \cdot f(x_n, t_n)$$

Or in the index notation

$$x_{n+1} \approx x_n + \delta_t \cdot f(x_n, t_n)$$

</div>

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

## Lecture 6

### Dynamical Systems with Special Functionals

In the study of dynamical systems, certain classes of systems exhibit unique behaviors due to the existence of special functions, or functionals, defined on their state space. These functionals, such as potentials, energy functions, or Hamiltonians, impose strong constraints on the system's dynamics. Understanding these functions allows us to predict the qualitative behavior of trajectories and the nature of equilibrium points without solving the differential equations explicitly. This chapter explores two fundamental types of such systems: Hamiltonian systems, often associated with conservation laws in physics, and Gradient systems, which describe processes moving towards local minima.

#### Hamiltonian Systems

Hamiltonian systems are a cornerstone of classical mechanics and dynamical systems theory. They are characterized by a conserved quantity, the Hamiltonian, which often corresponds to the total energy of the system. This conservation property leads to highly structured and constrained dynamics.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hamiltonian Function)</span></p>

Let the state space $E$ be an open set in $\mathbb{R}^{2m}$. A function $H: E \to \mathbb{R}$ is called a **Hamiltonian function** if it is twice differentiable ($C^2$) on $E$. The state variables are typically written as $(x, y)$ where $x, y \in \mathbb{R}^m$.

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

* A **saddle point** of the Hamiltonian function $H(x,y)$, then the equilibrium is a saddle.
* A **local maximum or minimum** of the Hamiltonian function $H(x,y)$, then the equilibrium is a center (a non-linear center).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Significance of the Theorem)</span></p>

This theorem is a powerful tool for classifying equilibrium points in non-linear systems. Typically, determining if an equilibrium is a true non-linear center requires analyzing higher-order terms of the system's Taylor expansion. However, if we can demonstrate that a system possesses a Hamiltonian function, we can classify its equilibria simply by analyzing the local extrema of $H$. This provides a direct and elegant method for proving the existence of non-linear centers, which are characterized by a dense set of closed orbits in their vicinity.

Systems that possess a Hamiltonian are often called **conservative systems**. The value of the Hamiltonian, $H(x,y)$, remains constant along any trajectory of the system, acting as a constant of motion.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Lotka-Volterra System)</span></p>

Let us revisit the Lotka-Volterra model for predator-prey interaction, which we can demonstrate is a Hamiltonian system.

The system is defined by the equations:

$$\begin{aligned}
\dot{x} &= \alpha x - \beta xy \\
\dot{y} &= \gamma xy - \lambda y
\end{aligned}$$

where $x$ represents the prey population and $y$ represents the predator population. All parameters $\alpha, \beta, \gamma, \lambda$ are positive constants.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Deriving the Hamiltonian for Lotka-Volterra)</span></p>

To show this system is Hamiltonian, we must construct a function $H(x,y)$ that is constant along the system's trajectories and satisfies the necessary conditions.

**Step 1: Combine the Differential Equations.** We can eliminate the time variable $dt$ by dividing the two equations:

$$\frac{dx}{dy} = \frac{\dot{x}}{\dot{y}} = \frac{\alpha x - \beta xy}{\gamma xy - \lambda y} = \frac{x(\alpha - \beta y)}{y(\gamma x - \lambda)}$$

**Step 2: Separate Variables.** We rearrange the equation to group terms involving $x$ and $y$ on opposite sides.

$$\frac{\gamma x - \lambda}{x} dx = \frac{\alpha - \beta y}{y} dy$$

**Step 3: Integrate Both Sides.** We integrate both sides of the equation to find the conserved quantity.

$$\int \left(\frac{\gamma x - \lambda}{x}\right) dx = \int \left(\frac{\alpha - \beta y}{y}\right) dy$$

$$\int \left(\gamma - \frac{\lambda}{x}\right) dx = \int \left(\frac{\alpha}{y} - \beta\right) dy$$

This yields:

$$\gamma x - \lambda \ln(x) = \alpha \ln(y) - \beta y + C$$

where $C$ is a constant of integration.

**Step 4: Define the Hamiltonian.** By rearranging the terms, we can define a function $H(x,y)$ that is constant along any trajectory.

$$H(x,y) = \alpha \ln(y) - \beta y - \gamma x + \lambda \ln(x)$$

Since the logarithm is only defined for positive arguments, this Hamiltonian is valid in the positive quadrant ($x > 0$, $y > 0$), which is the only biologically meaningful region for population models.

**Step 5: Verification with Auxiliary Variables.** To formally verify that this is a Hamiltonian system according to the definition, we introduce auxiliary variables:

* Let $p = \ln(x)$ and $q = \ln(y)$.
* This implies $x = e^p$ and $y = e^q$.

**Step 6: Rewriting the Hamiltonian** in terms of $p$ and $q$:

$$H(p, q) = \alpha q - \beta e^q + \lambda p - \gamma e^p$$

**Step 7: Verify the Hamiltonian conditions.**

We must verify that $\dot{p} = \frac{\partial H}{\partial q}$ and $\dot{q} = -\frac{\partial H}{\partial p}$.

*First condition for $\dot{p}$:* Using the chain rule, $\dot{p} = \frac{d}{dt}(\ln(x)) = \frac{1}{x}\dot{x}$.

$$\dot{p} = \frac{1}{x}(\alpha x - \beta xy) = \alpha - \beta y = \alpha - \beta e^q$$

Now, we compute the partial derivative of $H$ with respect to $q$:

$$\frac{\partial H}{\partial q} = \frac{\partial}{\partial q} (\alpha q - \beta e^q + \lambda p - \gamma e^p) = \alpha - \beta e^q$$

The condition $\dot{p} = \frac{\partial H}{\partial q}$ is satisfied.

*Second condition for $\dot{q}$:* Similarly, $\dot{q} = \frac{1}{y}\dot{y}$.

$$\dot{q} = \frac{1}{y}(\gamma xy - \lambda y) = \gamma x - \lambda = \gamma e^p - \lambda$$

Next, we compute the negative partial derivative of $H$ with respect to $p$:

$$-\frac{\partial H}{\partial p} = -\frac{\partial}{\partial p} (\alpha q - \beta e^q + \lambda p - \gamma e^p) = -(\lambda - \gamma e^p) = \gamma e^p - \lambda$$

The condition $\dot{q} = -\frac{\partial H}{\partial p}$ is also satisfied.

**Step 8: Conclusion.** We have successfully shown that the Lotka-Volterra system is a Hamiltonian system in the positive quadrant. Consequently, its equilibrium points must be either saddles or centers, which aligns with numerical simulations that show a dense set of closed orbits.

</div>

#### Gradient Systems

Another important class of systems are gradient systems, where the dynamics are governed by the gradient of a potential function. These systems model phenomena where a state moves to minimize a certain quantity, such as energy.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Potential Function and Gradient System)</span></p>

Let the state space $E$ be an open set in $\mathbb{R}^m$. Let $V: E \to \mathbb{R}$ be a twice differentiable ($C^2$) function on $E$.

A dynamical system is a **gradient system** if its vector field is given by the negative gradient of a potential function $V(x)$:

$$\dot{x} = -\frac{\partial V}{\partial x}$$

This can also be written using the gradient operator as $\dot{x} = -\nabla V(x)$. The function $V(x)$ is also referred to as a **gradient field**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Impossibility of Closed Orbits in Gradient Systems)</span></p>

Closed orbits are impossible in a gradient system.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A Powerful Restrictive Result)</span></p>

This theorem provides a very strong constraint on the possible behaviors of a gradient system. It implies that such systems cannot have centers or limit cycles. If one can construct a potential function for a given system, it immediately rules out any periodic or oscillatory behavior. All trajectories in a gradient system must eventually approach an equilibrium point or diverge to infinity.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Impossibility of Closed Orbits: Intuitive Argument)</span></p>

Let us provide the intuition behind this proof.

**Step 1: Assumption for Contradiction.** Suppose a closed orbit exists in the system. Let this orbit be parameterized by time $t$, and let $T$ be its period.

**Step 2: Property of the Potential Function.** A trajectory starting at a point $x_0$ on this orbit returns to the exact same point after time $T$. Since the potential function $V(x)$ is continuous (and in fact, twice differentiable), its value must be the same at the start and end of the orbit. Therefore, the total change in $V$ along one full orbit, $\Delta V$, must be zero.

$$\Delta V = V(x(T)) - V(x(0)) = 0$$

**Step 3: Calculating the Change in $V$.** The total change in $V$ along the orbit can also be calculated by integrating its rate of change, $\frac{dV}{dt}$, over the period $T$:

$$\Delta V = \int_0^T \frac{dV}{dt} \, dt$$

(The proof in the source material is paused at this point, but would continue by showing that $\frac{dV}{dt}$ is strictly non-positive and only zero at equilibrium points, leading to a contradiction.)

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Impossibility of Closed Orbits: Complete Argument)</span></p>

A defining characteristic of gradient systems is that they do not permit closed orbits (i.e., limit cycles). This can be proven by examining how the potential function $V$ changes over time along any given orbit.

Let us consider the change in the potential function, $\Delta V$, along an orbit of the system over a time interval. This change is given by the integral of the time derivative of $V$:

$$\Delta V = \int \frac{dV}{dt} \, dt$$

Using the chain rule, we can express $\frac{dV}{dt}$ as:

$$\frac{dV}{dt} = \frac{\partial V}{\partial x} \frac{dx}{dt} = (\nabla V) \cdot \dot{x}$$

By the definition of a potential function, we know that $\nabla V = -\dot{x}$. Substituting this into the equation gives:

$$\frac{dV}{dt} = (-\dot{x})^\top \cdot \dot{x} = - \lVert \dot{x}\rVert^2$$

Substituting this back into the integral for $\Delta V$, we get:

$$\Delta V = -\int \lVert \dot{x}\rVert^2 \, dt \leq 0$$

The term $\lVert \dot{x}\rVert^2$ is always non-negative. Therefore, the integral is also non-negative, and the entire expression for $\Delta V$ is always less than or equal to zero:

$$\Delta V \le 0$$

Equality, $\Delta V = 0$, holds only if $\dot{x} = 0$ for the entire duration of the trajectory. A state where $\dot{x} = 0$ is, by definition, an equilibrium point.

For a closed orbit, the system must return to its starting point, meaning the net change in the potential, $\Delta V$, must be zero. However, as we have shown, $\Delta V$ can only be zero if the system is at an equilibrium point. A true closed orbit involves movement ($\dot{x} \neq 0$), for which $\Delta V$ must be strictly negative. This is a contradiction. Therefore, no closed orbits can exist in a gradient system.

</div>

#### Equilibrium Points in Gradient Systems

The behavior of a gradient system can be intuitively understood by visualizing the potential function $V(x)$ as a landscape of hills and valleys. The dynamics, $\dot{x} = -\nabla V(x)$, describe a state "rolling downhill" towards lower potential values.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gradient System Equilibria as Landscape Features)</span></p>

* A **stable equilibrium** (a stable node) corresponds to a **local minimum** of the potential function $V$. Any small perturbation from the bottom of a "valley" will result in the system returning to that minimum.
* An **unstable equilibrium** corresponds to a **local maximum** of the potential function $V$. Any small perturbation from the top of a "hill" will cause the system to roll away, typically towards a nearby minimum.
* A **saddle point** of the potential function $V$ corresponds to a **saddle equilibrium** of the dynamical system.

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

1. **Symmetric Connectivity:** The weight matrix must be symmetric, i.e., $W = W^\top$. Intuitively, this symmetry gives rise to squared terms in the potential function's expression, which is crucial for its existence.
2. **Positive Time Constant:** The time constant $\tau$ must be greater than zero ($\tau > 0$).
3. **Monotonically Increasing Transfer Function:** The activation function $\sigma$ must be monotonically increasing.

If these conditions are met, the system is guaranteed to be a gradient system, and therefore its only attractors are stable fixed points. This is the case for original Hopfield networks. Conversely, if the weights are not symmetric ($W \neq W^\top$), as is more general for the Wilson-Cowan model, a potential function does not exist, and more complex dynamics like limit cycles can emerge.

</div>

### Bifurcation Theory

Bifurcation theory is a broad and critical area within the study of dynamical systems. It analyzes how the qualitative behavior of a system changes as one or more of its underlying parameters are varied.

#### Defining Bifurcations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bifurcation)</span></p>

A **bifurcation** is a qualitative, topological change in the state space of a system that occurs as a parameter is changed. This means the vector fields before and after the bifurcation point are not topologically equivalent. Such changes can include:

* The creation or destruction of new attractors (e.g., equilibrium points).
* A change in the stability of an existing attractor.

</div>

#### The Saddle-Node Bifurcation

One of the most important and common types of bifurcations is the saddle-node bifurcation. In this event, a stable equilibrium (a node) and an unstable equilibrium (a saddle) collide and annihilate each other.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Bifurcations in Wilson-Cowan Equations)</span></p>

Consider the Wilson-Cowan model for interacting populations of excitatory ($E$) and inhibitory ($I$) neurons, described by differential equations for their firing rates, $\nu_E$ and $\nu_I$. The equilibria of this system are found at the intersections of the nullclines for each population.

The system's behavior is governed by parameters such as the connection weights ($w_{EE}$, $w_{EI}$, etc.). Let's consider the self-excitation weight of the excitatory population, $w_{EE}$, as our control parameter.

* If we increase $w_{EE}$, the S-shaped nullcline for the excitatory population ($\dot{\nu}_E = 0$) is lifted upwards.
* At a critical value of $w_{EE}$, the S-shaped nullcline becomes tangent to the inhibitory nullcline ($\dot{\nu}_I = 0$). This point of tangency is the saddle-node bifurcation point. At this exact moment, a stable node and a saddle point merge.
* If $w_{EE}$ is increased further, the two equilibria disappear, leaving only one other equilibrium point.
* Conversely, decreasing $w_{EE}$ can lead to another saddle-node bifurcation at the lower bend of the S-curve.

The region of parameter space between these two bifurcation points is a **bistable regime**, where the system has two stable equilibrium points separated by an unstable saddle.

</div>

#### Bifurcation Diagrams

To visualize these changes, we use a bifurcation diagram. This diagram plots the location of the system's equilibria (often denoted as $\bar{x}$) against the value of the control parameter.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bifurcation Diagram for Saddle-Node)</span></p>

For the saddle-node bifurcation in the Wilson-Cowan example, the bifurcation diagram would look like an "S" shape turned on its side:

* The x-axis represents the control parameter (e.g., $w_{EE}$).
* The y-axis represents the position of the equilibria ($\bar{x}$).
* The upper and lower branches of the curve represent the stable equilibria (stable nodes).
* The middle branch (often drawn with a dashed line) represents the unstable equilibrium (the saddle).
* The points where the upper/lower branches meet the middle branch are the saddle-node bifurcation points.

This diagram powerfully illustrates phenomena like **hysteresis** and **sudden transitions**. A system state might follow the lower stable branch as the parameter increases, and then, upon reaching the bifurcation point, make a sudden jump to the upper stable branch. Such rapid transitions are seen in many real-world systems, including epileptic seizures in the brain.

</div>

#### The Normal Form for a Saddle-Node Bifurcation

Near a bifurcation point, many complex nonlinear systems can be simplified and shown to behave like a much simpler mathematical equation known as a normal form. The normal form captures the essential dynamics of the bifurcation.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Normal Form of a Saddle-Node Bifurcation)</span></p>

The **normal form for a saddle-node bifurcation** is given by the one-dimensional differential equation:

$$\dot{x} = r + x^2$$

Here, $r$ is the bifurcation parameter (or control parameter).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Analysis of the Normal Form)</span></p>

Let's analyze the behavior of this system by plotting $\dot{x}$ versus $x$ for different values of $r$:

* **When $r < 0$:** The parabola $y = r + x^2$ is shifted down and intersects the $x$-axis at two points. These are the equilibria:
  * One stable equilibrium (where the slope is negative).
  * One unstable equilibrium (where the slope is positive).
* **When $r = 0$:** The parabola $y = x^2$ is tangent to the $x$-axis at $x=0$. At this point, the stable and unstable equilibria have merged into a single, half-stable equilibrium. This is the exact moment of the saddle-node bifurcation.
* **When $r > 0$:** The parabola $y = r + x^2$ is shifted up and does not intersect the $x$-axis. There are no equilibria. The system state will always move towards $+\infty$ or $-\infty$.

This minimal equation perfectly captures the qualitative behavior of a saddle-node bifurcation: the collision and annihilation of a stable and an unstable fixed point as a parameter is varied.

</div>

#### The Concept of a Normal Form

When studying a specific type of bifurcation, it is incredibly useful to distill the system down to its essential mathematical structure. This leads to the concept of a normal form.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Normal Form)</span></p>

A **normal form** of a bifurcation is the simplest, minimal differential equation that exhibits the essential dynamics of that bifurcation. By analyzing the normal form, we can understand the universal properties of a whole class of systems near the bifurcation point, regardless of their specific physical or biological details. For instance, the normal form for a saddle-node bifurcation is $\dot{x} = r + x^2$.

</div>

#### Transcritical Bifurcation

A transcritical bifurcation is a fundamental type of bifurcation where two fixed points collide and exchange their stability properties.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Exchange of Stability)</span></p>

The core idea behind a transcritical bifurcation is an exchange of stability. Imagine two equilibrium states. As we tune a control parameter, these two equilibria move towards each other, merge at the bifurcation point, and then re-emerge, but with their stability profiles swapped -- the one that was stable is now unstable, and vice versa.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Normal Form of a Transcritical Bifurcation)</span></p>

The **normal form for the transcritical bifurcation** is given by the one-dimensional differential equation:

$$\dot{x} = rx - x^2$$

Here, $x$ is the state variable and $r$ is the control parameter.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Analysis of the Transcritical Bifurcation</span></p>

To understand the behavior, we analyze the system for different values of the parameter $r$. We can find the fixed points by setting $\dot{x} = 0$, which gives $x(r - x) = 0$. This yields two fixed points: $x^{\ast}_1 = 0$ and $x^{\ast}_2 = r$. Their existence and stability depend on the value of $r$.

**Case 1: $r < 0$**

* **Fixed Points:** We have two distinct fixed points, one at $x=0$ and another at $x=r$ (which is negative).
* **Stability:** The plot of $\dot{x}$ versus $x$ is an inverted parabola crossing the $x$-axis at $r$ and $0$.
  * The fixed point at $x=r$ (the "left one") is unstable.
  * The fixed point at $x=0$ (the "right one") is stable.

| Fixed Point ($x^*$) | Value | Stability |
|---|---|---|
| $x^*_1$ | $r$ | Unstable |
| $x^*_2$ | $0$ | Stable |

**Case 2: $r = 0$**

* **Fixed Points:** The two fixed points merge into one at $x=0$. The equation becomes $\dot{x} = -x^2$. The vertex of the parabola now touches the $x$-axis at the origin.
* **Stability:** This is a non-hyperbolic fixed point. The flow moves towards the fixed point from the positive side ($x>0$) and away from it on the negative side ($x<0$).

**Case 3: $r > 0$**

* **Fixed Points:** We again have two distinct fixed points, one at $x=0$ and another at $x=r$ (which is positive).
* **Stability:** The stability has now been exchanged compared to the $r<0$ case.
  * The fixed point at $x=0$ (the "left one") is now unstable.
  * The fixed point at $x=r$ (the "right one") is now stable.

| Fixed Point ($x^*$) | Value | Stability |
|---|---|---|
| $x^*_1$ | $0$ | Unstable |
| $x^*_2$ | $r$ | Stable |

**Bifurcation Diagram**

The bifurcation diagram visually summarizes these dynamics by plotting the location of the fixed points ($x^*$) as a function of the control parameter ($r$).

* A solid line represents a branch of stable fixed points.
* A dashed line represents a branch of unstable fixed points.

The diagram for the transcritical bifurcation shows two lines intersecting at the origin $(r, x) = (0, 0)$:

1. A horizontal line at $x=0$. This branch is stable for $r<0$ and becomes unstable for $r>0$.
2. A diagonal line representing $x=r$. This branch is unstable for $r<0$ and becomes stable for $r>0$.

At the bifurcation point $(0,0)$, the two branches cross and exchange stability.

</div>

#### Pitchfork Bifurcation

The pitchfork bifurcation is characteristic of systems possessing a certain symmetry. As its name suggests, the bifurcation diagram resembles a pitchfork.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Pitchfork Bifurcation Intuition</span></p>

In a pitchfork bifurcation, a single stable fixed point loses its stability as a parameter is varied. As it becomes unstable, two new stable fixed points are simultaneously created, branching off symmetrically from the original state. This often happens in systems where states can be equivalent but opposite (e.g., left/right, up/down).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">Normal Form of the Supercritical Pitchfork Bifurcation</span></p>

From the geometry of the bifurcation, one can guess that it involves a third-order polynomial. The normal form for the supercritical pitchfork bifurcation is:

$$\dot{x} = rx - x^3$$

Once again, $r$ is the control parameter. The symmetry is evident in the equation: if $x(t)$ is a solution, then so is $-x(t)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Analysis of the Pitchfork Bifurcation</span></p>

We find the fixed points by setting $\dot{x} = 0$, which gives $x(r - x^2) = 0$. The solutions depend critically on the sign of $r$.

**Case 1: $r \le 0$**

* **Fixed Points:** The only real solution is $x=0$.
* **Stability:** The plot of $\dot{x}$ versus $x$ shows a single crossing at the origin. This single fixed point is stable. For $r=0$, the fixed point is non-hyperbolic but still attracts nearby trajectories.

**Case 2: $r > 0$**

* **Fixed Points:** There are now three distinct fixed points: $x^{\ast}_1 = 0$, $x^{\ast}_2 = +\sqrt{r}$, and $x^{\ast}_3 = -\sqrt{r}$.
* **Stability:**
  * The original fixed point at $x=0$ has become unstable.
  * The two new, symmetrically located fixed points at $x = \pm\sqrt{r}$ are both stable.

**Bifurcation Diagram**

The bifurcation diagram for the supercritical pitchfork bifurcation has the following structure:

* For $r \le 0$, there is a single branch of stable fixed points at $x=0$.
* At $r=0$, this branch becomes unstable (represented by a dashed line for $r>0$).
* Simultaneously at $r=0$, two new branches of stable fixed points emerge, following the curves $x = \pm\sqrt{r}$. This creates the characteristic three-pronged "pitchfork" shape.

</div>

##### Supercritical and Subcritical Forms

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Supercritical vs. Subcritical</span></p>

The bifurcation described above is known as a **supercritical pitchfork bifurcation**. There is also a corresponding subcritical version.

* **Supercritical Pitchfork Bifurcation:** A stable fixed point becomes unstable and gives rise to two new stable fixed points. This is the case we analyzed, with the normal form $\dot{x} = rx - x^3$. The "fork" opens in the direction of increasing $r$.
* **Subcritical Pitchfork Bifurcation:** This is the reverse scenario. Two unstable fixed points merge with a stable fixed point, annihilating it and leaving no stable equilibria nearby. The bifurcation graph looks like a pitchfork opening in the direction of decreasing $r$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">Pitchfork Bifurcations in Neural Systems</span></p>

The pitchfork bifurcation often appears in models with inherent symmetries.

1. **Symmetric Wilson-Cowan System:** Consider a Wilson-Cowan model where the nullclines of the excitatory and inhibitory populations are perfectly symmetric. If a parameter (like the time constants) is changed, causing one nullcline to flatten, a single stable equilibrium can give way to three equilibria (one unstable, two stable), characteristic of a pitchfork bifurcation.
2. **Single Neuron with Sigmoidal Self-Coupling:** A simple one-dimensional model of a neuron with a sigmoidal input-output function can also exhibit this behavior. The fixed points are the intersections of the sigmoid function with the line $y=x$. If the sigmoid is symmetric around the origin, changing its slope can cause the system to transition from one stable fixed point to three through a pitchfork bifurcation.

</div>

#### Critical Slowing Down: A Signature of Approaching Bifurcations

A fascinating and important phenomenon occurs in the vicinity of many bifurcations, including the saddle-node and pitchfork bifurcations: critical slowing down.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Critical Slowing Down</span></p>

As a system approaches a bifurcation point, the basin of attraction of a stable equilibrium point "flattens out." In the phase portrait of $\dot{x}$ vs. $x$, the curve of the vector field becomes nearly tangent to the $x$-axis near the fixed point.

* **Mechanism:** Because the derivative $\dot{x}$ (the "velocity" of the system) becomes very close to zero in a wide region around the fixed point, the system's dynamics become extremely slow. Trajectories take an arbitrarily long time to converge to the equilibrium. This is critical slowing down.
* **Saddle-Node Example:** For a saddle-node bifurcation, if the parabola $\dot{x} = r + x^2$ is just slightly above the $x$-axis (for $r > 0$, before any fixed points are created), the flow through the narrow "ghost" channel where the fixed points are about to appear becomes very, very slow.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Observable Consequences of Critical Slowing Down</span></p>

This phenomenon provides a powerful, model-independent warning sign that a system is approaching a critical transition or "tipping point." This is actively studied in fields like climate science.

1. **Slow Recovery from Perturbations:** A system close to a bifurcation will take much longer to return to its stable state after being perturbed.
2. **Increased Variance:** In the presence of noise or random perturbations, the system's state will fluctuate with a much larger amplitude. Because the restoring force (the "friction") is so weak in the flattened region, noise can push the system far back and forth. An increase in the variance of a time series can therefore be a signature that the underlying system is approaching a bifurcation.

</div>

#### Hopf Bifurcation and the Birth of Limit Cycles

While the bifurcations discussed so far concern fixed points (equilibria), dynamical systems can also feature more complex behaviors like oscillations. The bifurcations of these oscillatory states are crucial for understanding rhythmic phenomena in nature.

##### Bifurcations of Equilibria vs. Limit Cycles

The concepts of bifurcations can be extended from fixed points to limit cycles (isolated, closed orbits). For example:

* A saddle-node bifurcation of cycles can occur, where a stable and an unstable limit cycle coalesce and annihilate each other.

However, the most important bifurcation for the creation of a limit cycle from an equilibrium point is the Hopf bifurcation.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">Hopf Bifurcation</span></p>

A **Hopf bifurcation** is a bifurcation where a fixed point of a dynamical system loses stability as a pair of complex conjugate eigenvalues of the linearized system cross the imaginary axis of the complex plane. This typically results in the birth of a small-amplitude limit cycle around the fixed point.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Mechanism of the Hopf Bifurcation</span></p>

* **The Mechanism:** Consider a stable spiral fixed point. Trajectories spiral inwards towards it. The stability is governed by the real part of the eigenvalues of the Jacobian matrix at that point; a negative real part ensures stability. As a system parameter is changed, the real part of the eigenvalues can change. At the Hopf bifurcation point, the real part becomes exactly zero. The point is no longer a stable spiral but a center, with trajectories orbiting it. As the parameter is changed further, the real part becomes positive, the fixed point becomes an unstable spiral, and trajectories spiral outwards. This outward flow is often captured by a newly born, stable limit cycle.
* **In summary:** A Hopf bifurcation marks the transition of a fixed point from a stable spiral to an unstable spiral, giving birth to a stable oscillation (limit cycle) in the process.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">Hopf Bifurcation in Neural Models</span></p>

The Hopf bifurcation is fundamental to the generation of rhythmic activity in many neural models.

1. **Wilson-Cowan System:** In the Wilson-Cowan model, changing parameters such as the excitatory-to-inhibitory drive can cause a stable fixed point to lose stability via a Hopf bifurcation. The result is the emergence of a stable limit cycle, which corresponds to a sustained oscillation in the activity of the neural populations.
2. **Bursting Neuron Models (e.g., Hodgkin-Huxley type):** In biophysical models of spiking neurons, a technique called slow-fast separation is often used. A slow variable (like an ion channel gating variable $h$) is treated as a control parameter for the fast subsystem (membrane voltage $V$ and another gating variable $n$).
   * In this fast subsystem, a fixed point can become an unstable spiral via a Hopf bifurcation.
   * This gives rise to a limit cycle that corresponds to the repetitive, fast spiking activity (action potentials) of the neuron. The system spirals out from the unstable fixed point onto this limit cycle, generating a train of spikes.

</div>

### The Hopf Bifurcation: Detailed Analysis

The Hopf bifurcation is a fundamental mechanism through which oscillations are born in a dynamical system. It describes the local birth or death of a periodic solution (a limit cycle) from an equilibrium point as a control parameter is varied. This section will explore the essential properties of this bifurcation, examine its two primary forms -- supercritical and subcritical -- and provide a concrete biophysical example.

#### General Properties of the Hopf Bifurcation

A Hopf bifurcation is characterized by a specific change in the stability of an equilibrium point. For this bifurcation to occur, the system must involve at least two dimensions (or variables), as oscillations require a "plane" to unfold.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Core Intuition</span></p>

At its core, the Hopf bifurcation marks the point where a system's tendency to return to a stable equilibrium is perfectly balanced by a force pushing it away, leading to sustained, stable oscillations. Before the bifurcation, disturbances might cause damped oscillations that spiral back to the equilibrium. After the bifurcation, the equilibrium becomes unstable, and disturbances spiral outwards, eventually settling onto a newly formed limit cycle.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Core Properties</span></p>

The key features of a system at a Hopf bifurcation point are rooted in the eigenvalues of the Jacobian matrix evaluated at the equilibrium.

1. **Equilibrium Type:** The equilibrium point must be a spiral. This means the eigenvalues of the system linearized around the equilibrium are a complex conjugate pair, $\lambda = \alpha \pm i\beta$.
2. **Stability Change:** The bifurcation occurs precisely when the real part of the complex eigenvalues crosses zero ($\alpha = 0$). The stability of the spiral point flips at this moment.
   * If $\alpha < 0$, the spiral is stable.
   * If $\alpha > 0$, the spiral is unstable.
3. **Oscillation Frequency:** The imaginary part of the eigenvalues must be non-zero ($\beta \neq 0$) at the bifurcation point. In the neighborhood of the bifurcation, the frequency of the resulting oscillation is approximately equal to this imaginary part, $\beta$.

The trajectory of the eigenvalues in the complex plane as a control parameter is varied looks as follows: they move across the imaginary axis.

| Eigenvalue Property | System Behavior |
|---|---|
| $\text{Re}(\lambda) < 0$ | Stable spiral (damped oscillations) |
| $\text{Re}(\lambda) = 0$ | Hopf Bifurcation (center-like) |
| $\text{Re}(\lambda) > 0$ | Unstable spiral (growing oscillations) |

</div>

#### The Supercritical Hopf Bifurcation

The supercritical Hopf bifurcation is characterized by a smooth, gradual onset of oscillation.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">Supercritical Hopf Bifurcation</span></p>

In a supercritical Hopf bifurcation, as a control parameter is varied, a stable spiral equilibrium loses its stability and becomes an unstable spiral. At the exact moment of stability change, a stable limit cycle is born with an infinitesimally small amplitude, which then grows smoothly as the parameter continues to change.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Bifurcation Diagram for Supercritical Hopf</span></p>

The state of the system can be visualized in a bifurcation diagram, where the vertical axis represents an oscillation amplitude (e.g., the maximum value of a variable on the limit cycle) and the horizontal axis represents the control parameter.

* **Before the bifurcation point:** A single line represents the stable spiral equilibrium.
* **At the bifurcation point:** The stable spiral becomes unstable (often denoted by a dashed line). A parabolic curve emerges from this point, representing the maximum and minimum amplitudes of the newly created stable limit cycle.

| Parameter Regime | System State |
|---|---|
| Before Bifurcation | One stable spiral (equilibrium point). |
| At Bifurcation Point | The spiral loses stability. |
| After Bifurcation | One unstable spiral and one stable limit cycle surrounding it. |

Imagine a system perturbed from its equilibrium.

* **Before the bifurcation:** The system exhibits damped oscillations, spiraling back into the stable equilibrium.
* **After the bifurcation:** The system spirals out from the now-unstable equilibrium, and the oscillations grow until they settle onto the stable limit cycle, resulting in sustained periodic behavior.

A known example of this phenomenon occurs in the Wilson-Cowan equations.

</div>

#### The Subcritical Hopf Bifurcation

In contrast to the smooth transition of the supercritical case, the subcritical Hopf bifurcation is associated with abrupt and dramatic changes in system behavior.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">Subcritical Hopf Bifurcation</span></p>

In a subcritical Hopf bifurcation, a stable spiral equilibrium loses stability and becomes an unstable spiral. However, this occurs when the equilibrium point coalesces with a pre-existing unstable limit cycle. This unstable cycle acts as a separatrix between the basin of attraction of the stable spiral and another, potentially distant, stable attractor.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bifurcation Diagram for Subcritical Hopf)</span></p>

The bifurcation diagram for a subcritical Hopf looks like a "flipped" version of the supercritical one.

* **Before the bifurcation point:** There is a stable equilibrium. Separately, there may exist an unstable limit cycle (often represented by a dashed parabola) and a large-amplitude stable limit cycle. This region can exhibit bistability.
* **At the bifurcation point:** The stable equilibrium becomes unstable as it merges with and annihilates the unstable limit cycle.
* **After the bifurcation point:** The only remaining attractor is the large-amplitude stable limit cycle.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Danger of Subcritical Hopf Bifurcations)</span></p>

The subcritical Hopf bifurcation can be frightening in real-world systems because it leads to sudden, large-scale changes. As the control parameter slowly approaches the bifurcation point, the system appears stable. The moment it crosses the threshold, the equilibrium vanishes, and the system "hops" or jumps to a completely different state -- often a large-amplitude oscillation.

* **Example:** A famous analogy is soldiers marching in step across a bridge. The frequency of their marching acts as a forcing parameter. At a critical point, the bridge can undergo a subcritical Hopf bifurcation, suddenly jumping from a state of small vibration to a large-amplitude, destructive oscillation, causing it to break.

This type of bifurcation highlights that changes in natural systems are not always gradual; they can be abrupt and catastrophic.

</div>

#### Normal Forms for Hopf Bifurcations

To study the local dynamics near a Hopf bifurcation, it is useful to transform the system into a simpler "normal form." These are typically written in polar coordinates $(r, \theta)$, where $r$ represents the amplitude of the oscillation and $\theta$ represents its phase. The control parameter $\mu$ is defined such that the bifurcation occurs at $\mu = 0$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Normal Form: Supercritical Hopf)</span></p>

The normal form for the radius (amplitude) of the supercritical Hopf bifurcation is:

$$\dot{r} = \mu r - r^3$$

The equation for the phase is often given as $\dot{\theta} = \omega$, where $\omega$ is a constant frequency.

* For $\mu < 0$: The only stable equilibrium is at $r=0$. The $-r^3$ term is negligible for small $r$, and the $\mu r$ term drives the system to the origin.
* For $\mu > 0$: The equilibrium at $r=0$ becomes unstable (since $\dot{r} > 0$ for small $r$). A new stable limit cycle appears at $r = \sqrt{\mu}$, where $\dot{r} = 0$. The amplitude of this cycle grows smoothly from zero as $\sqrt{\mu}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Normal Form: Subcritical Hopf)</span></p>

Analogous to the subcritical pitchfork bifurcation, the normal form for the radius is:

$$\dot{r} = \mu r + r^3$$

* For $\mu < 0$: The equilibrium at $r=0$ is stable. An unstable limit cycle exists at $r=\sqrt{-\mu}$.
* For $\mu > 0$: The equilibrium at $r=0$ becomes unstable. The unstable limit cycle has vanished, and trajectories are now pushed away from the origin by both terms.

</div>

#### The Morris-Lecar Model: A Subcritical Hopf Example

The Morris-Lecar model is a simplified 2D biophysical model describing the membrane potential of a single neuron. It serves as an excellent example of a system exhibiting a subcritical Hopf bifurcation.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Morris-Lecar Equations)</span></p>

The model consists of two coupled ordinary differential equations for the membrane potential ($V$) and a gating variable ($n$):

$$
\begin{align*}
C \dot{V} &= I - g_L(V - E_L) - g_{Ca} m_{\infty}(V)(V - E_{Na}) - g_K n (V - E_K) \\
\dot{n} &= \frac{n_{\infty}(V) - n}{\tau_n(V)}
\end{align*}$$

Here, $m_{\infty}(V)$ and $n_{\infty}(V)$ are sigmoidal functions of voltage, representing the activation of ion channels. This model can be seen as a simplification of the 3D biophysical neuron models discussed previously.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Phase Space Analysis of the Morris-Lecar Model)</span></p>

By plotting the nullclines of the system ($\dot{V} = 0$ and $\dot{n} = 0$), we can analyze its dynamics. For certain parameter configurations, the phase space has the following structure:

1. A stable equilibrium (a node or spiral).
2. An unstable limit cycle surrounding the equilibrium.
3. A large-amplitude stable limit cycle, corresponding to continuous spiking activity.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Bifurcation and Bistability)</span></p>

As a parameter (such as the injected current, $I$) is changed, the unstable limit cycle can shrink and eventually coalesce with the stable equilibrium.

* At this point, a subcritical Hopf bifurcation occurs.
* The equilibrium and the unstable cycle annihilate each other.
* The system state is forced to "hop" from the former equilibrium to the only remaining attractor: the large, stable limit cycle representing neuronal spiking.

A crucial feature of this system is the existence of a parameter regime before the bifurcation point where two stable attractors coexist: the stable equilibrium (a resting state) and the stable limit cycle (a spiking state). This phenomenon is known as **bistability**. The system's final state depends on the initial conditions -- it will either settle to rest or engage in continuous spiking.

</div>

### Co-dimension Two Bifurcations and Complex Dynamics

#### A Landscape of Bifurcations

#### Introduction to Co-dimension Two Bifurcations

#### The Cusp Bifurcation

##### Normal Form and Properties

##### Parameter-Dependent Dynamics

##### The Geometry of the Cusp

#### Applications in Biophysical Systems

##### The Bogdanov-Takens Bifurcation

##### Neuronal Bursting as a Bifurcation Phenomenon

## Lecture 7

### An Introduction to Chaos and the Logistic Map

This chapter introduces the fundamental concepts of chaos theory using the logistic map as a primary example. We will explore the key signatures of chaotic systems and the process by which a simple, deterministic system can exhibit complex, aperiodic behavior.

#### The Logistic Map Revisited

We begin by recalling the logistic map, a simple equation originally developed to model population dynamics.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Logistic Map)</span></p>

The logistic map is a discrete-time dynamical system defined by the equation:

$$x_{t} = \alpha x_{t-1}(1 - x_{t-1})$$

Where:

* $x_t$ represents the state of the system at time $t$.
* $\alpha$ is a control parameter.

For the system to remain bounded within the unit interval $[0, 1]$, we consider initial conditions $x_0 \in [0, 1]$ and the parameter range $\alpha \in [0, 4]$. The function's graph is a parabola, and its dynamics can be visualized using bifurcation diagrams, which show the long-term behavior of the system as $\alpha$ is varied. We have previously identified its fixed points and periodic cycles.

</div>

#### Core Characteristics of Chaos

As we increase the parameter $\alpha$ beyond the values that produce simple fixed points and low-period cycles, the system's behavior changes dramatically. The period of the cycles doubles at an accelerating rate until, at a critical value of $\alpha$, the trajectory becomes irregular and appears to fill a dense region of the state space. This phenomenon is known as chaos. Chaotic systems, though entirely deterministic, exhibit several defining characteristics.

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/chaos_logistic_map.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Higher-order cycles ($\alpha = 3.5$, top row) and chaos ($\alpha = 3.9$, center) in the logistic map. Bottom graph illustrates quick divergence of time series in chaotic regime for just slightly different (by an $\epsilon$ of $10^{-4}$) initial conditions</figcaption>
</figure>

##### Aperiodicity in Deterministic Systems

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Aperiodicity</span></p>

The first signature of chaos is aperiodic or irregular behavior that arises in the complete absence of noise or external randomness. For a chaotic trajectory, there is no integer $n$ for which the system exactly repeats its state.

$$x_{t+n} \neq x_t \quad \forall n > 0$$

This means the system never settles into a periodic cycle. One might consider this a "cycle of infinite period," but the crucial point is that the trajectory never closes on itself. The emergence of such non-repeating behavior from a simple, deterministic equation was a surprising discovery in the mid-20th century.

</div>

##### Sensitive Dependence on Initial Conditions

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Sensitive Dependence</span></p>

The second key feature of chaos is a sensitive dependence on initial conditions. This means that two trajectories starting from infinitesimally different initial states will diverge from each other at an exponential rate. Even a minuscule, imperceptible difference in starting points will lead to completely different outcomes after a short period of time. This exponential divergence is a hallmark of chaotic dynamics.

</div>

##### Boundedness

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Boundedness</span></p>

A third, and somewhat puzzling, characteristic is that the system remains bounded. Despite the exponential divergence of nearby trajectories, the system's state does not grow infinitely. For the logistic map, it can be proven that for $\alpha \in [0, 4]$ and $x_0 \in [0, 1]$, the trajectory will never leave the unit interval. The system is simultaneously divergent on a local scale (nearby trajectories separate) and constrained on a global scale (the overall dynamics are confined to a specific region).

</div>

#### The Period Doubling Route to Chaos

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Period-Doubling Cascade</span></p>

Systems can transition to chaos through various mechanisms as a control parameter is changed. For the logistic map, this transition occurs via the famous period-doubling route to chaos, also known as the period-doubling cascade.

By examining the bifurcation diagram of the logistic map, we can observe this process:

1. **Stable Fixed Point:** For small values of $\alpha$, the system has a single stable fixed point.
2. **First Bifurcation:** As $\alpha$ increases, the fixed point becomes unstable and gives rise to a stable 2-cycle. The system now oscillates between two values.
3. **Cascade:** With further increases in $\alpha$, this 2-cycle becomes unstable and bifurcates into a stable 4-cycle. This process repeats, creating an 8-cycle, a 16-cycle, and so on. The interval of $\alpha$ over which each subsequent cycle is stable becomes progressively shorter.
4. **Onset of Chaos:** This cascade of period-doubling bifurcations culminates at a finite value of $\alpha$, beyond which the system's behavior becomes chaotic and aperiodic.

</div>

#### The Strange Attractor

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Strange Attractor</span></p>

Even in the chaotic regime, the object to which trajectories converge is still an attractor. This means that initial conditions from within a certain basin of attraction will converge toward this complex, bounded object. This is another puzzling aspect of chaos: there is simultaneous convergence to the attractor from the outside and divergence within the attractor itself. Such an object is often called a strange attractor.

</div>

#### Periodic Windows in the Chaotic Regime

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Periodic Windows</span></p>

A final interesting feature visible in the bifurcation diagram is the presence of periodic windows embedded within the chaotic regime. As $\alpha$ is increased through the chaotic region, the system can suddenly revert to stable, periodic behavior (for example, a stable 3-cycle or 5-cycle) for a narrow range of $\alpha$ values, before returning to chaos.

</div>

<div class="pmf-grid">
<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/magnification_logistic_bifurcation_diagram.png' | relative_url }}" alt="a" loading="lazy">
  <!-- <figcaption>Neutrally stable harmonic oscillations in linear ODE system. The graph illustrates that the choice of numerical ODE solver is important when integrating ODEs numerically: While the exact analytical solution (blue) and that of an implicit second-order numerical solver (Rosenbrock-2, green) tightly agree (in fact, a blue curve is not visible since the green curve falls on top of it), a simple forward Euler scheme (red) diverges from the true solution. </figcaption> -->
</figure>
<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/periodic_windows1.png' | relative_url }}" alt="a" loading="lazy">
  <!-- <figcaption>Neutrally stable harmonic oscillations in linear ODE system. The graph illustrates that the choice of numerical ODE solver is important when integrating ODEs numerically: While the exact analytical solution (blue) and that of an implicit second-order numerical solver (Rosenbrock-2, green) tightly agree (in fact, a blue curve is not visible since the green curve falls on top of it), a simple forward Euler scheme (red) diverges from the true solution. </figcaption> -->
</figure>
<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/periodic_windows_logistic_map.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>The famous orbit diagram for the logistic map. Three periodic windows in the chaotic regime are marked by vertical solid lines.</figcaption>
</figure>
</div>

### Deeper Properties of One-Dimensional Maps

The study of one-dimensional maps like the logistic map has revealed a rich mathematical structure underlying their chaotic behavior. Here, we summarize several key theorems and concepts.

#### Unstable Periodic Orbits (UPOs)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Unstable Periodic Orbits</span></p>

For certain parameter values, the chaotic attractor is built upon a "skeleton" of infinitely many unstable periodic orbits (UPOs). For the logistic map at $\alpha = 4$, it can be shown that there exists an infinite number of these UPOs.

* These are complete, periodic cycles that the system could follow.
* However, they are all unstable, meaning any small perturbation will cause the trajectory to move away from the cycle.
* For the logistic map at $\alpha=4$, UPOs exist for every possible integer period $k \geq 1$.

</div>

#### Topological Equivalence: The Tent Map

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Topological Equivalence</span></p>

The logistic map is topologically equivalent to another map called the tent map. This means there is a continuous, one-to-one mapping (a homeomorphism) that transforms one system's dynamics into the other while preserving the direction of time. Consequently, they exhibit the same fundamental dynamical phenomena, including the period-doubling route to chaos.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">The Tent Map</span></p>

The tent map is a piecewise linear map defined by:

$$f(x) = \alpha \cdot \min(x, 1-x)$$

The map is typically studied for $x \in [0, 1]$ and $\alpha \in [0, 2]$. Its graph has a characteristic "tent" shape.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Importance of Piecewise Linear Systems</span></p>

The study of piecewise linear systems is important for two main reasons:

1. **Analytical Tractability:** Many mathematical analyses that are difficult for general nonlinear systems become much easier for piecewise linear systems. They can exhibit complex behaviors like chaos while remaining mathematically manageable.
2. **Relevance to Modern Applications:** Many recurrent neural networks (RNNs) used in deep learning are fundamentally piecewise linear systems and can exhibit the full spectrum of dynamical behaviors, including chaos.

</div>

#### The Li-Yorke Theorem: Period Three Implies Chaos

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">Li-Yorke Theorem (1975)</span></p>

For a one-dimensional map, if a period-3 cycle is observed, then the system must also exhibit chaotic behavior.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

This is a powerful and surprising result. The mere existence of a cycle with period three is a sufficient condition to guarantee that the system's dynamics are complex enough to be classified as chaotic.

</div>

#### The Sharkovskii Ordering of Periodic Orbits

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">Sharkovskii's Theorem</span></p>

There exists a specific ordering of natural numbers, known as the Sharkovskii ordering, which dictates a hierarchy of implications for the existence of periodic orbits in one-dimensional maps.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Sharkovskii Ordering</span></p>

If a map possesses a periodic orbit of period $k$, it must also possess periodic orbits for all periods that appear to the right of $k$ in the Sharkovskii ordering.

The ordering begins as follows:

$$\begin{align*}
3 &\implies 5 \implies 7 \implies 9 \implies \dots \implies (2n+1) \implies \dots \\
\implies 3 \cdot 2 &\implies 5 \cdot 2 \implies 7 \cdot 2 \implies \dots \implies (2n+1) \cdot 2 \implies \dots \\
\implies 3 \cdot 2^2 &\implies 5 \cdot 2^2 \implies \dots \implies (2n+1) \cdot 2^2 \implies \dots \\
\vdots \\
\implies 2^n &\implies 2^{n-1} \implies \dots \implies 2^3 \implies 2^2 \implies 2 \implies 1
\end{align*}$$

The most significant implication of this theorem is that the number 3 is first in the ordering. Therefore, the existence of a 3-cycle implies the existence of cycles of all other integer periods, which connects back to the Li-Yorke theorem.

</div>

#### The Feigenbaum Constants

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Feigenbaum Constants</span></p>

For maps that exhibit the period-doubling route to chaos (like the logistic and tent maps), the rate at which the bifurcations occur is governed by a universal constant.

Let $\alpha_n$ be the parameter value at which the period-doubling bifurcation from a $2^{n-1}$-cycle to a $2^n$-cycle occurs. The ratio of the intervals between successive bifurcation points converges to a universal constant, known as the first Feigenbaum constant, $\delta$.

$$\delta = \lim_{n \to \infty} \frac{\alpha_n - \alpha_{n-1}}{\alpha_{n+1} - \alpha_n} \approx 4.669...$$

This universality is a profound property, indicating that a wide class of systems transitions to chaos in a quantitatively identical manner.

</div>

### Examples of Chaotic Behavior

The principles of chaos are not just mathematical curiosities; they appear in a wide range of natural and engineered systems.

#### Empirical Evidence: Population Dynamics in Beetles

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">Population Dynamics in Beetles</span></p>

An empirical study of a beetle population provided evidence for the period-doubling cascade in a biological system. By manipulating a control parameter (related to the beetles' environment or resources), researchers observed the population's long-term behavior transition from a stable equilibrium to 2-cycles, then 4-cycles, and eventually to chaotic fluctuations in population numbers. This provides a real-world parallel to the dynamics of the logistic map.

</div>

#### Biophysical Systems: The Bursting Neuron Model

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">The Bursting Neuron Model</span></p>

Chaos can also be found in more complex, higher-dimensional continuous systems. We previously discussed a three-dimensional model of a bursting neuron, described by a set of differential equations for the membrane potential ($V$) and two gating variables ($n$ and $h$).

By performing a fast-slow separation of variables, we analyzed the bifurcations of the fast subsystem (governed by $V$ and $n$). We identified various bifurcations, such as a homoclinic bifurcation where a limit cycle (representing spiking) crashes into a saddle point, terminating the spiking behavior. It turns out that within the parameter space of such biophysical models, chaotic dynamics can also emerge, leading to irregular, non-repeating patterns of neural activity.

</div>

### An Introduction to Chaos in Dynamical Systems

Welcome to the study of chaotic systems. For decades, chaos was considered a mathematical curiosity, but we now understand it to be an almost inevitable phenomenon in many natural systems. As soon as a system incorporates a sufficient number of elements interacting in a nonlinear fashion, the emergence of chaos becomes highly probable. This chapter will introduce the fundamental concepts of chaos, explore its appearance in both biological and physical models, and define its core mathematical and conceptual properties.

#### What is Chaos? An Initial View

At its core, chaos describes a specific type of complex, unpredictable behavior in deterministic dynamical systems. A key indicator of potential chaos is the presence of homoclinic orbits (also referred to as homoclinic intersections), which are trajectories that connect a saddle-type equilibrium point to itself.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Homoclinic Orbits and Chaos</span></p>

The presence of these homoclinic structures is a strong hint that a system might exhibit chaotic behavior. For certain classes of systems, particularly discrete maps, the existence of homoclinic orbits is not just a hint -- it is a guarantee. There are formal theorems that prove this connection, solidifying the link between a system's geometric structure in state space and its dynamic behavior over time.

</div>

#### Examples of Chaotic Systems

To build a concrete understanding of chaos, we will examine two canonical examples: a modern model from computational neuroscience and the classic Lorenz system from atmospheric science.

##### A Continuous-Time Biological Neuron Model

Let us first consider a full three-dimensional model of a biological neuron. The dynamics of this system are governed by several parameters, one of which is a conductance parameter. By systematically increasing this single parameter, we can observe the system's trajectory through distinct behavioral regimes.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">Bifurcation in a Neuron Model</span></p>

The model exhibits the following sequence of behaviors as a key conductance parameter is increased:

1. **Bursting Behavior:** The neuron fires in rapid bursts followed by periods of quiescence.
2. **Chaotic Regime:** The system transitions into a state of highly irregular, aperiodic spiking activity. In state space, the trajectory begins to densely fill a specific, bounded region. This is the hallmark of a chaotic attractor.
3. **Regular Spiking Behavior:** As the parameter increases further, the system settles into a stable, periodic firing pattern.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Empirical Validation</span></p>

This progression is not merely a theoretical construct; it is observed empirically. In biological experiments using "slice preparations," real, individual neurons exhibit the same transitions from bursting to chaotic to regular spiking when subjected to similar changes in their electrochemical environment. Statistical tests can be applied to these empirical time series to support the conclusion that the intermediate, irregular regime is indeed chaos.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">The Role of Nonlinearity</span></p>

The emergence of this complex behavior is deeply tied to nonlinearity. In this neuron model, the chaotic and bursting dynamics are produced by highly nonlinear NMDA conductances. These elements drive the system back and forth between a region of state space where a stable limit cycle exists and another region where only a stable fixed point exists. If one were to linearize the model equations, these complex phenomena would vanish entirely. The bifurcation diagram for this system reveals a limit cycle "crashing" into the homoclinic orbit of a saddle point, triggering the complex dynamics.

</div>

##### The Lorenz Equations of Atmospheric Convection

One of the most famous and foundational examples of chaos arises from the Lorenz equations. Conceived by Edward Lorenz, this model is a significant simplification of a much larger set of equations describing atmospheric convection.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">The Lorenz System</span></p>

The Lorenz system is a set of three coupled, nonlinear, first-order ordinary differential equations:

$$\begin{aligned}
\dot{x} &= s(y - x) \\
\dot{y} &= x(r - z) - y \\
\dot{z} &= xy - bz
\end{aligned}$$

* The system has three dynamical variables: $x$, $y$, and $z$.
* It is controlled by three positive constants (parameters): $s$, $r$, and $b$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Simplicity of the Lorenz System</span></p>

The power of the Lorenz system lies in its simplicity. The first equation is entirely linear. The nonlinearity of the entire system stems from just two terms: the product $x \cdot z$ in the second equation and the product $x \cdot y$ in the third. Such simple polynomial or multinomial forms of nonlinearity are not unique to this system; they appear frequently in models across various fields, including Lotka-Volterra predator-prey models, epidemic models, and chemical reaction models.

In their original physical context, the variables and parameters have specific interpretations:

* The variables relate to horizontal and vertical temperature gradients and convection velocity.
* The parameter $s$ is related to the Prandtl number, and $r$ is related to the Rayleigh number.

By varying the parameter $r$ while keeping $s$ and $b$ fixed, the system exhibits a range of behaviors, culminating in the formation of the renowned Lorenz attractor. This "butterfly wing" structure is one of the most iconic images in the study of chaos.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">Attractor</span></p>

An attractor is a set of states in the state space towards which a system's trajectory evolves over time. For a chaotic attractor, trajectories that start nearby will converge onto this object. Once on the attractor, the trajectory moves around chaotically, densely filling a region of the state space. It is important to note that chaos can also be unstable or transient; not all chaotic behavior is confined to an attractor.

</div>

#### Core Properties of Chaotic Systems

Across different examples, chaotic systems share a set of defining characteristics. Understanding these properties is crucial for distinguishing chaos from other types of complex or random behavior.

##### Aperiodicity and Boundedness

A key feature of a chaotic trajectory is that it is aperiodic.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">Aperiodic</span></p>

An aperiodic trajectory is one that never closes up or repeats itself exactly. No matter how long the system is observed, the state vector will never return to a previous value.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

In practice, we observe this aperiodicity numerically by running a simulation for a very long time and seeing that the trajectory never settles into a repeating loop (a limit cycle). Furthermore, while a limit cycle is a one-dimensional curve, a chaotic trajectory will start to densely fill a higher-dimensional region of the state space.

At the same time, a chaotic attractor is a bounded object. The trajectory is confined to a specific region of state space and will not diverge to infinity.

</div>

##### The Dimensionality Requirement for Continuous Systems

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">Minimum Dimension for Chaos</span></p>

For continuous-time dynamical systems (described by ordinary differential equations), chaos is only possible in three or more dimensions.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">by Intuition</span></p>

1. **One Dimension:** In one dimension ($\dot{x} = f(x)$), a trajectory can only move along a line. It can approach a fixed point or go to infinity, but it cannot oscillate or exhibit complex dynamics, as it cannot turn around without stopping at a fixed point.
2. **Two Dimensions:** In two dimensions, trajectories can form cycles (limit cycles). However, a fundamental property of solutions to deterministic ODEs is that trajectories in state space cannot cross. If a trajectory were to cross itself, it would imply two different future paths from the same point, violating determinism. For a trajectory to be chaotic, it must explore a region densely without repeating. In a 2D plane, a non-intersecting, bounded curve must eventually close on itself, forming a periodic orbit. It cannot densely fill a 2D area.
3. **Three Dimensions:** In three dimensions, a trajectory has enough freedom to move without intersecting itself while remaining in a bounded region. This allows it to weave through space in a complex, aperiodic pattern, forming the intricate structure of a chaotic attractor.

</div>

##### Sensitive Dependence on Initial Conditions (The Butterfly Effect)

Perhaps the most famous characteristic of chaos is its sensitive dependence on initial conditions (SDIC).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">Sensitive Dependence on Initial Conditions</span></p>

This property means that two trajectories starting from arbitrarily close initial points will, on average, diverge from each other exponentially over time.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">The Butterfly Effect</span></p>

This is famously known as the butterfly effect, a metaphor suggesting that the flap of a butterfly's wings in one part of the world might ultimately cause a hurricane in another. A tiny, imperceptible change in the system's initial state can lead to macroscopically different outcomes. This property is what makes long-term prediction for chaotic systems fundamentally impossible. While the system is deterministic (the rules are fixed), our inability to measure the initial state with infinite precision means any small error will grow exponentially, rendering long-term forecasts useless.

</div>

##### The Structured Nature of Chaos

The butterfly effect might suggest that chaotic systems are completely random and unpredictable. This is a common misconception. While individual trajectories are unpredictable in the long term, the collective behavior is highly structured.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Structure Within Chaos</span></p>

The key is the existence of the chaotic attractor.

* It is a bounded object with a specific geometry. A trajectory on the Lorenz attractor will always remain within the "butterfly wings"; it cannot go just anywhere in state space. Trajectories that start off the attractor are pulled into it.
* It has characteristic statistics. Although the point-by-point future is unknown, the long-term statistical properties of the system's behavior are often stable and predictable. The system possesses what are known as ergodic statistics, meaning that averages over a long time for a single trajectory are the same as averages over the entire attractor at a single moment.

Therefore, while we cannot predict the exact state of the system far into the future, we can predict the probability of finding the system in a particular region of its attractor. Statistically speaking, a chaotic system is not completely random or unpredictable; it possesses a deep and beautiful structure.

</div>

### Characterizing Chaos: Lyapunov Exponents

In our exploration of dynamical systems, we often encounter behavior that is complex, aperiodic, and seemingly random, yet arises from deterministic rules. This phenomenon is known as chaos. To move beyond a purely qualitative description, we need a quantitative measure to characterize this complexity. The Lyapunov exponent provides just such a tool, measuring the average rate of divergence or convergence of nearby trajectories in the state space. A positive Lyapunov exponent is a hallmark of chaotic dynamics, signifying the sensitive dependence on initial conditions that is central to the concept of chaos.

#### Lyapunov Spectrum

Consider a 1D map x_t = F_{\alpha}(x_{t-1}), where \alpha is a control parameter. Map's orbit is \lbrace x_1, \dots, x_t, \dots \rbrace, looking in a finite limit of this orbit. Lyapunov number if l = \lim_{n\to\infty} \prod_{t=1}^n \lvert F'_{\alpha}(x_t) \rvert^{\frac{1}{n}}. In fact, we are taking a geometric mean of these series. Lyapunov exponent is defined a logarithm of this product series: \lambda = \lim_{n\to\infty} \frac{1}{n} \sum{t=1}^n \log \lvert F'_{\alpha}(x_t) \rvert. Let's extend this to a multivariate case.

#### Defining the Maximum Lyapunov Exponent

The maximum Lyapunov exponent, denoted by $\lambda_{\text{max}}$, quantifies the most rapid rate of separation of trajectories. Its formulation differs slightly between discrete-time maps and continuous-time flows, but the underlying principle remains the same.

##### The Discrete-Time Case: Product of Jacobians

For a multi-dimensional discrete-time system, the evolution of a small perturbation is governed by the Jacobian matrix of the map at each step. To understand the long-term behavior, we must consider the product of these Jacobians along a trajectory.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Maximum Lyapunov Exponent: Discrete Time)</span></p>

For a discrete-time dynamical system defined by the map $x_{n+1} = f(x_n)$, the maximum Lyapunov exponent $\lambda_{\text{max}}$ is defined as:

$$\lambda_{\text{max}} = \lim_{n \to \infty} \frac{1}{n} \log \left\| \prod_{i=0}^{n-1} J(x_i) \right\|$$

where $J(x_i)$ is the Jacobian matrix of the map $f$ evaluated at point $x_i$ on the trajectory, and $\| \cdot \|$ denotes a matrix norm, typically the spectral norm.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The term $\prod_{i=0}^{n-1} J(x_i)$ represents the total linearization of the dynamics along a trajectory of length $n$. It tells us how an infinitesimal ball of initial conditions is stretched and rotated after $n$ iterations. By taking the logarithm, we are converting this multiplicative stretching factor into an additive rate. Finally, dividing by $n$ and taking the limit as $n \to \infty$ gives us the average exponential rate of separation per iteration over the entire trajectory.

</div>

##### The Continuous-Time Case: Trajectory Separation

For continuous-time systems, we can define the exponent by directly observing the evolution of the distance between two initially close trajectories.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Exponential Divergence Ansatz)</span></p>

Consider two trajectories starting at infinitesimally close initial conditions, $x_0$ and $x_0 + \Delta x_0$. We can make an ansatz that the distance between them, $\Delta x(t)$, grows exponentially over time. This can be expressed as:

$$\| \Delta x(t) \| \approx \| \Delta x_0 \| e^{\lambda t}$$

Here, $\lambda$ is the rate of separation, which we identify as the maximum Lyapunov exponent. This simple relationship forms the basis for the formal definition.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Maximum Lyapunov Exponent: Continuous Time)</span></p>

For a continuous-time dynamical system with flow operator $\phi_t(x_0)$, the maximum Lyapunov exponent $\lambda_{\text{max}}$ is defined by a two-step limit process:

$$\lambda_{\text{max}} = = \lim_{t \to \infty} \lim_{\| \Delta x_0 \| \to 0} \frac{1}{t} \log \left( \frac{\| \Delta x(t) \|}{\| \Delta x_0 \|} \right) \lim_{t \to \infty} \lim_{\| \Delta x_0 \| \to 0} \frac{1}{t} \log \left( \frac{\| \phi_t(x_0 + \Delta x_0) - \phi_t(x_0) \|}{\| \Delta x_0 \|} \right)$$

In the limit where the initial separation $\| \Delta x_0 \|$ approaches zero, the numerator can be expressed in terms of the partial derivatives of the flow operator. This leads to an analogous form:

$$\lambda_{\text{max}} = \lim_{t \to \infty} \frac{1}{t} \log \left\| \frac{\partial \phi_t(x_0)}{\partial x_0} \right\|$$

This formulation highlights the parallel with the discrete case, where the product of Jacobians is replaced by the Jacobian of the flow operator over time $t$.

</div>

#### The Lyapunov Spectrum and Geometric Intuition

In a $d$-dimensional system, separation and contraction can occur at different rates in different directions. The full set of these rates is known as the Lyapunov spectrum, consisting of $d$ Lyapunov exponents: $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d$. The maximum exponent, $\lambda_1 = \lambda_{\text{max}}$, governs the overall stability, but the entire spectrum provides a richer description of the dynamics.

##### Stretching and Contracting Space

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric Interpretation)</span></p>

Imagine a small, spherical ball (an $\epsilon$-ball) of initial conditions in the state space. As the system evolves, this sphere is deformed along the trajectory.

* In directions associated with positive Lyapunov exponents, the ball is stretched into an ellipsoid.
* In directions associated with negative Lyapunov exponents, the ball is compressed.
* In directions associated with zero Lyapunov exponents, the distance is, on average, preserved.

For a chaotic attractor, this process of stretching and folding is continuous. The ellipsoid is stretched along unstable directions ($\lambda_i > 0$) and simultaneously compressed along stable directions ($\lambda_j < 0$). This continuous deformation, constrained within a bounded region of state space, is what generates the complex, fractal structures characteristic of strange attractors.

</div>

##### Connection to Singular Value Decomposition

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(SVD Connection)</span></p>

The stretching and reorientation of this evolving ellipsoid of initial conditions can be formally described by the Singular Value Decomposition (SVD) of the Jacobian product matrix, $G = \prod J(x_i)$. Recall that the singular values of a matrix describe its scaling action along principal axes.

* The maximum Lyapunov exponent corresponds to the maximum singular value of this product matrix.
* The singular values, $\sigma_i$, of $G$ are given by the square root of the eigenvalues of the matrix $G^\top G$.

This provides a powerful connection between the abstract concept of Lyapunov exponents and the concrete geometric transformations occurring in the state space. It is analogous to the eigenvalue analysis we perform for linear systems to classify fixed points, but extended to the non-linear, trajectory-dependent dynamics of chaotic systems. The singular values quantify the stretching and contraction along different directions.

</div>

#### Lyapunov Exponents as a Diagnostic Tool

The value of the maximum Lyapunov exponent serves as a powerful indicator of the long-term behavior of a dynamical system.

| Value of $\lambda_{\text{max}}$ | Implied Long-Term Behavior |
|---|---|
| $\lambda_{\text{max}} > 0$ | Chaotic evolution. Trajectories diverge exponentially. |
| $\lambda_{\text{max}} = 0$ | Neutral stability. Often occurs on a limit cycle. |
| $\lambda_{\text{max}} < 0$ | Stable fixed point or stable periodic orbit. |

An unstable fixed point, which features local divergence, will also be associated with a positive Lyapunov exponent for trajectories originating near it.

#### A Rigorous Definition of Chaos

While chaos is often described qualitatively by properties like aperiodic or irregular evolution, the Lyapunov exponents allow for a precise mathematical definition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Chaos)</span></p>

A dynamical system is considered chaotic if it satisfies the following core conditions:

1. **Sensitive Dependence on Initial Conditions:** The system must have at least one positive Lyapunov exponent. That is, $\lambda_{\text{max}} > 0$. This ensures that nearby trajectories diverge exponentially over time.
2. **Boundedness:** The system's trajectories must be confined to a bounded region of the state space. This prevents trajectories from simply exploding to infinity.

An additional condition is sometimes included for mathematical rigor in certain systems:

* **Aperiodicity:** All Lyapunov exponents are unequal to zero. This condition is sometimes omitted but helps to exclude simple periodic orbits.

The two primary conditions -- divergence within a bounded set -- are the most crucial elements to remember.

</div>

#### The Nature of Chaotic Attractors

For a chaotic system to be an attractor, it must not only keep trajectories bounded but must, on average, contract volumes in the state space. If the system expanded volume overall, it would not be an attractor, as volumes would eventually grow to fill all of space.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Volume Contraction)</span></p>

The rate of volume change in the state space is related to the sum of all the Lyapunov exponents in the spectrum.

* A system is volume-contracting if the sum of its Lyapunov exponents is negative.

Therefore, for a system to possess a chaotic attractor, a crucial condition must be met.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Condition for a Chaotic Attractor)</span></p>

A chaotic system possesses an attractor only if the sum of all its Lyapunov exponents is less than zero:

$$\sum_{i=1}^{d} \lambda_i < 0$$

This ensures that while the system stretches along at least one direction (due to $\lambda_{\text{max}} > 0$), the contraction along other directions is strong enough to ensure that overall, volumes shrink and trajectories are drawn towards a bounded, fractal object -- the strange attractor.

</div>

### Fractal Structures in Chaotic Systems

#### Introduction to Fractals in Chaos

In the study of chaotic systems, we often encounter geometric structures of immense complexity. These are not the simple lines, planes, or spheres of classical geometry, but intricate, jagged, and infinitely detailed objects known as fractals.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Fractal Basins of Attraction)</span></p>

A hallmark of chaotic systems is a profound sensitivity to initial conditions. This sensitivity becomes even more pronounced when the basins of attraction for different attractors possess fractal boundaries. In such systems, the distinct basins are interwoven, meaning that within any infinitesimally small neighborhood of one basin, there exists a point belonging to another. Consequently, a minute perturbation to an initial condition might not merely shift the final state along the same attractor but could propel it into a completely different dynamical regime.

The emergence of these fractal geometries in chaotic attractors is often due to a fundamental dynamical process involving three key actions:

1. Stretching along certain directions.
2. Contraction along other directions.
3. Folding or reinjection, where the stretched and contracted structure is mapped back into itself.

This sequence of stretching, contracting, and folding, when repeated iteratively, gives rise to a recursive structure. A key property arising from this process is self-similarity, where zooming into a part of the structure reveals smaller copies of the whole, repeating at ever-finer scales.

</div>

#### The Cantor Set: A Prototypical Fractal

To understand the properties of fractal sets, we begin with a foundational example: the Cantor set. It is a set of points on a line segment with a number of counter-intuitive and remarkable properties.

##### Iterative Construction

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Cantor Set)</span></p>

The Cantor set is constructed through an iterative procedure starting with the closed interval $[0, 1]$.

1. **Start with $K_0$:** Let $K_0$ be the interval $[0, 1]$.
2. **Iteration 1:** Remove the open middle third, $(\frac{1}{3}, \frac{2}{3})$, from $K_0$. The remaining set is 
   
   $$K_1 = [0, \frac{1}{3}] \cup [\frac{2}{3}, 1]$$

3. **Iteration 2:** Remove the open middle third from each of the remaining segments in $K_1$. This yields 
   
   $$K_2 = [0, \frac{1}{9}] \cup [\frac{2}{9}, \frac{3}{9}] \cup [\frac{6}{9}, \frac{7}{9}] \cup [\frac{8}{9}, 1]$$

4. **Continue Iteratively:** Repeat this process for all subsequent sets $K_n$. The Cantor set, $K$, is the set of points that remain after this process is carried out an infinite number of times, i.e., 
   
   $$K = \lim_{n \to \infty} K_n$$

At the $n$-th iteration, the set $K_n$ consists of $2^n$ closed intervals.

#TODO: Proposition: Cantor set has zero measure

</div>

##### Ternary Representation and Properties

To analyze the Cantor set more deeply, it is useful to represent the numbers in the interval $[0, 1]$ using a base-3, or ternary, number system.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">Ternary Representation</span></p>

A number $x \in [0, 1]$ can be expressed in a ternary representation as:

$$x = \sum_{i=1}^{\infty} a_i 3^{-i} = a_1 \cdot 3^{-1} + a_2 \cdot 3^{-2} + a_3 \cdot 3^{-3} + \dots$$

This can be written as $x = (0.a_1 a_2 a_3 \dots)_3$, where each digit $a_i$ is an element of the set $\lbrace 0, 1, 2 \rbrace$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Ternary Interpretation of the Cantor Construction)</span></p>

The iterative construction of the Cantor set has a direct and elegant interpretation in the ternary system.

* The interval $[0, 1]$ is divided into three parts: $[0, \frac{1}{3}]$, $[\frac{1}{3}, \frac{2}{3}]$, and $[\frac{2}{3}, 1]$.
* Numbers in the first third, $[0, \frac{1}{3}]$, have a ternary representation where the first digit, $a_1$, is $0$. (e.g., $0.0\dots_3$).
* Numbers in the second third, $[\frac{1}{3}, \frac{2}{3}]$, have a ternary representation where the first digit, $a_1$, is $1$. (e.g., $0.1\dots_3$).
* Numbers in the third third, $[\frac{2}{3}, 1]$, have a ternary representation where the first digit, $a_1$, is $2$. (e.g., $0.2\dots_3$).

In the first step of the Cantor construction ($K_1$), we remove the middle third. This is equivalent to removing all numbers whose ternary representation begins with the digit $1$.

In the second step ($K_2$), we remove the middle third of the remaining segments. This is equivalent to removing all numbers whose second ternary digit, $a_2$, is $1$.

This reveals the fundamental property of the Cantor set: it consists of all numbers in the interval $[0, 1]$ whose ternary representation can be written using only the digits $0$ and $2$.

A minor mathematical subtlety exists at the boundaries between intervals. For example, the number $\frac{1}{3}$ can be written as $(0.1)_3$ or as $(0.0222\dots)_3$. Since a representation without the digit '$1$' exists, such boundary points remain in the set.

</div>

##### The Uncountability of the Cantor Set

Despite its construction by removing intervals, the Cantor set contains more than just the endpoints of those intervals. In fact, it contains an uncountably infinite number of points, a fact we can prove using a technique similar to Cantor's famous diagonal argument.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Uncountability of the Cantor Set)</span></p>

The Cantor set is an uncountably infinite set.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name"></span></p>

1. **Assume for Contradiction:** Assume the Cantor set is countable. This means we can create an exhaustive list, or an enumeration, of all numbers in the set. Let this list be $x_1, x_2, x_3, \dots$.
2. **Represent in Ternary:** Each number $x_i$ in our list belongs to the Cantor set, so its ternary representation consists only of the digits $0$ and $2$. We can write our list as follows:
   * $x_1 = 0.a_{11} a_{12} a_{13} a_{14} \dots$
   * $x_2 = 0.a_{21} a_{22} a_{23} a_{24} \dots$
   * $x_3 = 0.a_{31} a_{32} a_{33} a_{34} \dots$
   * ...
3. where every digit $a_{ij} \in \lbrace 0, 2 \rbrace$.
4. **Construct a New Number:** We will now construct a new number, let's call it $R$, which is also composed only of digits $0$ and $2$, but which cannot be on our list. We define the digits of $R = 0.r_1 r_2 r_3 \dots$ by examining the diagonal elements of our list ($a_{11}, a_{22}, a_{33}, \dots$).
5. For each digit $r_i$, we set its value as follows:
   * If $a_{ii} = 0$, then set $r_i = 2$.
   * If $a_{ii} = 2$, then set $r_i = 0$.
6. **Show Contradiction:** By this construction, the number $R$ is composed exclusively of digits $0$ and $2$, so it must belong to the Cantor set. However, $R$ cannot be on our list.
   * $R$ cannot be $x_1$ because its first digit $r_1$ is different from $a_{11}$.
   * $R$ cannot be $x_2$ because its second digit $r_2$ is different from $a_{22}$.
   * In general, $R$ cannot be $x_i$ for any $i$, because its $i$-th digit $r_i$ is different from $a_{ii}$.
7. **Conclusion:** Our newly constructed number $R$ belongs to the Cantor set but is not in our supposedly complete list. This is a contradiction. Therefore, our initial assumption that the Cantor set is countable must be false. The set is uncountably infinite.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Zero Measure, Uncountable Cardinality)</span></p>

This result presents a fascinating puzzle. The total length of the intervals removed from $[0, 1]$ is 

$$\frac{1}{3} + 2(\frac{1}{9}) + 4(\frac{1}{27}) + \dots = \sum_{n=0}^{\infty} \frac{2^n}{3^{n+1}} = 1$$

The Cantor set has a zero measure (or zero length). Yet, as we have just proven, it contains an uncountably infinite number of points. This combination of zero measure and uncountable cardinality is a defining feature of many fractal sets.

</div>

#### The Smale Horseshoe Map

We now move from a static set to a dynamic map that generates a fractal structure. The Smale horseshoe map is a famous construction that captures the essential topological mechanisms -- stretching, contraction, and folding -- that produce chaos in systems like the Lorenz attractor.

##### Motivation: Distilling the Essence of Chaos

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Motivation)</span></p>

The Smale horseshoe map is an elegant abstraction of chaotic dynamics. It can be viewed as a 2D Poincare map of a 3D system, designed to distill the core properties of chaos into their essence. The map operates on a simple geometric space (a square) but, through iteration, generates a structure of incredible complexity, analogous to a chaotic attractor. The entire process is driven by two simple components: a linear transformation (the Jacobian) and a geometric fold.

</div>

##### The Mapping Process: Stretch, Contract, and Fold

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Smale Horseshoe Map)</span></p>

The Smale horseshoe map $F$ is a mapping of the unit square $S = [0, 1] \times [0, 1]$ onto the real plane $\mathbb{R}^2$. The process involves two distinct operations.

1. **Stretching and Contraction:** The map first applies a linear transformation to the square. This transformation is characterized by a local Jacobian matrix (3 0 \\ 0 1/3) that stretches the square along one dimension and contracts it along another.
2. This Jacobian stretches the square by a factor of $3$ in the horizontal direction and compresses it by a factor of $3$ in the vertical direction, transforming the unit square into a long, thin rectangle.
3. **Folding:** The long rectangle is then bent into a "horseshoe" shape and placed back over the original unit square $S$.

Only the parts of the horseshoe that lie within the original square $S$ are considered for the next iteration. After the first application of the map $F$, these parts consist of two vertical rectangular regions, which we can label $H_1$ and $H_2$.

</div>

##### Iterations and Fractal Generation

The complexity of the system arises when we iterate the map. Let's consider what happens when we apply the map a second time, $F^2(S)$, to the set that remained in the square after the first iteration (i.e., to $H_1 \cup H_2$).

1. The two vertical strips ($H_1$ and $H_2$) are themselves stretched horizontally and contracted vertically.
2. This new, even longer and thinner shape is then folded back onto the square.

The result is that within each of the new vertical strips, there are now smaller segments corresponding to both of the original strips, $H_1$ and $H_2$. This repetitive stretching and folding creates an increasingly fine and complex structure. The set of points that remain within the square $S$ after an infinite number of iterations forms a fractal set that is, in fact, a product of two Cantor sets.

### The Smale Horseshoe and Symbolic Dynamics

#### The Horseshoe Map: Construction and Iteration

The Smale Horseshoe is a foundational example in the study of dynamical systems that elegantly demonstrates how simple operations of stretching and folding can generate profoundly complex, chaotic behavior. The map operates on a unit square, which we will denote as $S$.

The process is iterative and consists of three fundamental steps applied to the square $S$:

1. **Stretch:** The square is stretched in one direction (e.g., horizontally) to three times its original length.
2. **Contract:** Simultaneously, the square is contracted in the perpendicular direction (vertically) to one-third of its original height.
3. **Fold:** The resulting long, thin rectangle is then folded into a 'U' shape, or horseshoe, and placed back over the area of the original square.

Let us trace the evolution of subsets through this process. We begin with the square $S$. After one iteration, the mapping $f(S)$ results in two horizontal stripes, which we can label $H_1$ and $H_2$.

Now, consider a second iteration of the map, $f(f(S)) = f^2(S)$. The process is applied again to the stripes $H_1$ and $H_2$. Each stripe is stretched, contracted, and folded. The resulting image consists of four even thinner horizontal stripes.

To understand the dynamics, we must establish a consistent labeling system.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(An Indexing System)</span></p>

We can create a systematic index for the stripes generated at each iteration. Let the original stripes from the first iteration be $H_1$ and $H_2$. In the second iteration, the image of $H_1$ will consist of two new stripes, which we can label $H_{11}$ and $H_{21}$. Similarly, the image of $H_2$ will produce stripes $H_{12}$ and $H_{22}$.

Let's clarify the geometric arrangement after the fold. The folding operation places the transformed regions back onto the original square. The region corresponding to $H_2$ is folded on top of the region corresponding to $H_1$. This means the resulting stripes will appear as follows, from top to bottom:

* $H_{12}$
* $H_{22}$
* $H_{21}$
* $H_{11}$

At each iteration $k$, this process generates $2^k$ horizontal stripes. This indexing system, where we append a digit with each iteration, forms the basis for a more powerful analytical tool we will explore later, known as symbolic dynamics.

</div>

#### Properties of the Map: Lyapunov Exponents

To quantify the stretching and contracting nature of the map, we can examine its Jacobian.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Jacobian of the Horseshoe Map</span></p>

The map is constructed from two distinct parts: the linear transformation (stretching and contracting) and the non-linear folding. The Jacobian reflects the linear part of this operation. For the Smale Horseshoe, this Jacobian remains constant across the regions of interest.

From the Jacobian of the map, we can directly determine the Lyapunov exponents, which measure the average rate of separation or convergence of nearby trajectories.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">Lyapunov Exponents of the Horseshoe Map</span></p>

For the Smale Horseshoe map, the Jacobian reveals two constant Lyapunov exponents:

* $\lambda_1 = \log(3) > 0$
* $\lambda_2 = \log(1/3) = -\log(3) < 0$

The positive exponent, $\lambda_1$, corresponds to the stretching direction, indicating that nearby points are exponentially separated. The negative exponent, $\lambda_2$, corresponds to the contracting direction, indicating that nearby points are exponentially brought closer together. This combination of expansion and contraction is a hallmark of chaotic systems.

</div>

#### The Invariant Set $\Lambda$

The truly interesting dynamics of the horseshoe map occur not on the entire square, but on a specific subset of points that remain within the square under all forward and backward iterations of the map. This is known as the invariant set.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Forward and Backward Iteration</span></p>

As we iterate the map forward in time ($k \to \infty$), the horizontal stripes become progressively thinner. At iteration $k$, we have $2^k$ stripes, each with a height of $(1/3)^k$. As $k$ approaches infinity, this collection of stripes converges to a geometric structure known as a Cantor set in the vertical direction.

We can also consider the inverse of the map, $f^{-1}$, which corresponds to iterating the system backward in time. Applying the inverse map involves reversing the steps: unfolding, contracting horizontally, and stretching vertically. This process, when iterated, creates a set of vertical stripes.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">The Invariant Set $\Lambda$</span></p>

The invariant set, denoted by $\Lambda$, is defined as the set of all points in the square $S$ that remain in $S$ for all time, both forward and backward. Mathematically, it is the intersection of all forward and backward images of the square:

$$\Lambda = \bigcap_{k=-\infty}^{\infty} f^k(S)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">The Geometry of $\Lambda$</span></p>

The set $\Lambda$ is the geometric intersection of the horizontal stripes generated by forward iteration and the vertical stripes generated by backward iteration. After a finite number of forward and backward iterations, the set appears as a grid of small squares. As we approach an infinite number of iterations, this structure becomes infinitely fine. $\Lambda$ is a fractal set -- it is the product of two Cantor sets, one aligned vertically and one horizontally.

</div>

#### Introduction to Symbolic Dynamics

To analyze the complex structure of $\Lambda$ and the dynamics upon it, we introduce a powerful technique called symbolic dynamics. The core idea is to "code" the history of each point by tracking which subset it belongs to at each iteration.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">A Bi-Infinite Coding System</span></p>

We can extend the indexing system developed earlier to account for both forward and backward time. Let us use a dot ('.') as a separator between past and future.

* **Forward Time:** For forward iterations ($k > 0$), we assign an index based on the horizontal stripe a point occupies. We can write this sequence to the right of the dot, e.g., $.x_0x_1x_2...$.
* **Backward Time:** For backward iterations ($k < 0$), we assign an index based on the vertical stripe a point occupies. We write this sequence to the left of the dot, e.g., $...x_{-2}x_{-1}$.

Combining these gives a unique, bi-infinite sequence of symbols for each point in the invariant set $\Lambda$:

$$...x_{-2}x_{-1}.x_0x_1x_2...$$

For example, consider the small squares formed by the intersection of the stripes. The square in the bottom-left corner after the first forward and first backward iteration would have an index like $...1.1...$. The square in the top-right would have an index like $...2.2...$. As we iterate infinitely, we find points within $\Lambda$ corresponding to unique bi-infinite sequences.

* One point will correspond to the sequence $...111.111...$.
* Another point will correspond to the sequence $...222.222...$.

This method of assigning a sequence of symbols to orbits is known as symbolic coding.

</div>

#### The Power of Symbolic Coding: Unveiling Chaos

The true power of symbolic dynamics is that it transforms a complex geometric problem into a simpler, algebraic one. By analyzing the space of all possible symbol sequences, we can prove profound theorems about the dynamics on the invariant set $\Lambda$. It can be formally shown that any bi-infinite sequence of symbols corresponds to a unique point in $\Lambda$. This leads to remarkable conclusions about the nature of the orbits within this set.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">The Structure of Orbits in $\Lambda$</span></p>

The invariant set $\Lambda$ of the Smale Horseshoe map contains:

1. A countable set of unstable periodic orbits of every possible length (period).
2. An uncountable set of aperiodic but bounded orbits.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">The Skeleton of Chaos</span></p>

This theorem reveals the intricate structure of chaos. The horseshoe map generates an infinite number of repeating, periodic behaviors, but all of them are unstable. A tiny perturbation will knock a trajectory off a periodic orbit. Furthermore, there is an even larger infinity of orbits that never repeat at all yet remain bounded within the square.

This method of symbolic coding has been instrumental in proving the existence of chaos in many other systems. It distills the essential topology of the dynamics into a symbolic form, upon which rigorous proofs can be built.

Steven Strogatz provides a beautiful metaphor for the structure of a chaotic attractor, which is highly relevant here. He describes it as an "infinite skeleton of unstable periodic orbits." The system state wanders between these orbits, repelled by one and attracted to another, much like a "ball in a pinball wizard machine." The chaotic attractor is organized around this dense, unstable backbone of periodic behaviors.

</div>

## Lecture 8

[SINDy](/subpages/dynamical-systems/dynamical-systems-in-machine-learning/sindy/)

### Characterizing the Geometry of Attractors

#### Introduction: The Need for New Geometric Measures

In our previous discussions, we explored the concepts of chaos and fractal sets, exemplified by the Cantor set. We established that chaotic attractors, which arise from processes like the Smale horseshoe mapping that continuously reinject trajectories into a bounded region of space, often produce these intricate fractal structures.

A natural and pressing question arises when studying these objects: how can we describe their geometry? Classical Euclidean geometry, which deals with integer dimensions (a line is 1D, a plane is 2D), seems inadequate. The Cantor set, for instance, has a Lebesgue measure of zero, yet it contains an uncountably infinite number of points. This paradox signals the need for a new type of measurement to characterize such peculiar sets. This chapter introduces powerful concepts for quantifying the dimensionality of complex objects, which will be essential not only for understanding chaos but also for the machine learning applications we will explore later.

### The Box-Counting Dimension

#### Intuition and Scaling

The fundamental idea behind the box-counting dimension is remarkably simple. Imagine we have some dynamical object—it could be a trajectory, a limit cycle, a fixed point, or a complex chaotic set—residing in a state space. To measure its dimension, we ask: How many boxes of a given size are needed to completely cover the set?

Let's formalize this. We tile our space with a grid of boxes, each with a side length of $\epsilon$. We then count the number of boxes, $N(\epsilon)$, that are intersected by our set $S$.

The crucial question is: How does $N(\epsilon)$ scale as the box size $\epsilon$ approaches zero?

Intuitively, the smaller the boxes, the more we will need. We can propose a scaling law that relates the number of boxes to the size $\epsilon$ and the object's intrinsic dimension, $D$:

$$N(\epsilon) \propto \left(\frac{1}{\epsilon}\right)^D$$

or, introducing a proportionality constant $C$,

$$N(\epsilon) \approx C \left(\frac{1}{\epsilon}\right)^D$$

Let's test this intuition with familiar objects:

* A Line (1-dimensional object): Suppose we cover a line segment with boxes of size $\epsilon$. Now, let's reduce the box size by half, to $\epsilon' = \epsilon/2$. It is clear we will now need twice as many boxes to cover the same line. The number of boxes scales linearly: 
  
  $$N(\epsilon') = 2 N(\epsilon) = 2^1 N(\epsilon)$$
  
  This matches our proposed law, where $D=1$.

* A Plane (2-dimensional object): If we cover a planar area with squares of side length $\epsilon$ and then reduce the side length to $\epsilon' = \epsilon/2$, we will need four times as many squares to cover the same area. The number of boxes scales quadratically:
  
  $$N(\epsilon') = 4 N(\epsilon) = 2^2 N(\epsilon)$$
  
  This again matches our law, with $D=2$.

This relationship provides a robust method for defining a dimension $D$ that extends beyond integer values.

#### Formal Definition

To isolate the dimension $D$ from our scaling law, we can use logarithms. Taking the logarithm of both sides of 

$$N(\epsilon) = C (1/\epsilon)^D$$

gives:

$$\log(N(\epsilon)) = \log(C) + D \log\left(\frac{1}{\epsilon}\right)$$

Rearranging for $D$, we get:

$$D = \frac{\log(N(\epsilon)) - \log(C)}{\log(1/\epsilon)}$$

To get a precise definition, we examine the behavior in the limit as $\epsilon \to 0$. In this limit, the $\log(N(\epsilon))$ and $\log(1/\epsilon)$ terms will grow, while the constant term $\log(C)$ becomes negligible. This allows us to drop the constant and arrive at a formal definition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Box-Counting Dimension)</span></p>

For a set $S$ in an $m$-dimensional Euclidean space $\mathbb{R}^m$, the box-counting dimension, $D_{box}$, is defined as:

$$D_{box} = \lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log(1/\epsilon)}$$

where $N(\epsilon)$ is the minimum number of $m$-dimensional boxes of side length $\epsilon$ required to cover the set $S$.

</div>

#### Example: The Cantor Set

Let's apply this definition to the middle-third Cantor set, which we construct iteratively.

* Recall the Construction: We begin with the interval $[0, 1]$. At each iteration $n$, we remove the middle third of every existing interval.
* Counting the Boxes: After $n$ iterations, we are left with $2^n$ disjoint intervals.
* Determining the Box Size: The length of each of these small intervals is $(1/3)^n$.

We can directly use these properties to calculate the box-counting dimension. For the $n$-th iteration of the construction, we can choose our box size $\epsilon_n$ to be the exact length of the intervals, $\epsilon_n = (1/3)^n$. The number of boxes required to cover the set is then precisely $N(\epsilon_n) = 2^n$.

The limit $\epsilon \to 0$ is equivalent to the number of iterations $n \to \infty$. Plugging these into the definition:

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Calculation of $D_{box}$ for the Cantor Set)</span></p>

1. Start with the definition of the box-counting dimension:
   
   $$D_{box} = \lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log(1/\epsilon)}$$

2. Substitute $N = 2^n$ and $\epsilon = (1/3)^n$. The limit becomes $n \to \infty$: 
   
   $$D_{box} = \lim_{n \to \infty} \frac{\log(2^n)}{\log(1/(1/3)^n)}$$

3. Simplify the denominator: 
   
   $$D_{box} = \lim_{n \to \infty} \frac{\log(2^n)}{\log(3^n)}$$

4. Use the logarithm property $\log(a^b) = b \log(a)$ to bring the exponents down:
   
   $$D_{box} = \lim_{n \to \infty} \frac{n \log 2}{n \log 3}$$

5. The $n$ terms cancel out, leaving a constant value. The limit is therefore this constant: 
   
   $$D_{box} = \frac{\log 2}{\log 3} \approx 0.6309$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The result is a fractal dimension—a non-integer value. This mathematically confirms our intuition that the Cantor set is more complex than a collection of points (which would have dimension $0$) but less "space-filling" than a continuous line segment (which has dimension $1$). This fractional value elegantly captures the set's intricate, self-similar structure. This is a general property of chaotic attractors that possess a fractal structure.

</div>

#### Empirical Estimation

The box-counting dimension is not just a theoretical construct; it can be estimated from data. For an observed dataset representing a chaotic attractor, one can:

1. Embed the data in its state space.
2. Overlay grids of decreasing box size $\epsilon$.
3. For each $\epsilon$, count the number of boxes $N(\epsilon)$ that contain at least one data point.
4. Plot $\log N(\epsilon)$ against $\log(1/\epsilon)$.
5. For a range of $\epsilon$ where the data exhibits scaling, these points should fall on a straight line.
6. The slope of this line provides an empirical estimate of the box-counting dimension, $D_{box}$.

This is illustrated by the log-log plot below, where the slope directly yields the dimension $D$.

(Note: As per instructions, no images can be rendered. The source describes a log plot where the slope of $\log N(\epsilon)$ vs $\log(1/\epsilon)$ is the dimension D.)

### The Correlation Dimension

#### Motivation: A Practical Alternative

While the box-counting dimension is powerful, it can be computationally prohibitive to implement in practice. For systems with a state space of dimension four or higher, creating a grid and counting boxes becomes exceptionally tedious as the number of required boxes explodes exponentially (the "curse of dimensionality").

To overcome this, a more practical, trajectory-based measure was developed: the correlation dimension.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

A key property and practical advantage of the correlation dimension is that its value is always less than or equal to the box-counting dimension: 

$$D_{corr} \le D_{box}$$

</div>

#### Definition and Intuition

Instead of covering the entire set with a grid, the correlation dimension is defined along the orbits or trajectories of the system. This makes it especially well-suited for analyzing time-series data from experiments.

The core idea is to:

1. Take points along a trajectory of the system.
2. For each point, place a ball of radius $\epsilon$ around it.
3. Count the average number of other points on the trajectory that fall within this ball. Let's call this quantity $C(\epsilon)$.
4. Examine how $C(\epsilon)$ scales as $\epsilon \to 0$.

The logic remains the same:

* For a 1D object (a line), the number of neighbors within radius $\epsilon$ should scale proportionally to $\epsilon^1$.
* For a 2D object (a plane), the number of neighbors should scale proportionally to $\epsilon^2$.

This leads to the scaling law:

$$C(\epsilon) \approx k \cdot \epsilon^D$$

where $k$ is a constant and $D$ is the correlation dimension. By again taking the logarithm and considering the limit as $\epsilon \to 0$, we arrive at the formal definition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Correlation Dimension)</span></p>

The correlation dimension, $D_{corr}$, is defined as:

$$D_{corr} = \lim_{\epsilon \to 0} \frac{\log C(\epsilon)}{\log \epsilon}$$

where $C(\epsilon)$ is the correlation integral, which measures the average number of neighbors within an $\epsilon$-ball around points on the trajectory.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Note the subtle but important difference in the denominator: $\log \epsilon$ instead of $\log(1/\epsilon)$. This is because as $\epsilon \to 0$, the number of boxes $N(\epsilon)$ needed to cover a set increases (so $\log(1/\epsilon)$ is positive and grows), whereas the number of neighbors $C(\epsilon)$ found within a small ball decreases (so $\log \epsilon$ is negative and approaches $-\infty$). The ratio remains a well-defined, positive dimension.

</div>

### A Note on Dimensionality and System Type

It is important to distinguish between discrete-time systems (maps) and continuous-time systems (ODEs) when discussing the dimensionality required for chaos.

* Maps: Chaotic behavior can occur in very low-dimensional maps. The logistic map, for instance, is a one-dimensional system ($x_{n+1} = rx_n(1-x_n)$) that exhibits chaos.
* Continuous-Time ODEs: For systems of ordinary differential equations, chaos is not possible in one or two dimensions (due to the Poincaré-Bendixson theorem). A minimum of three dimensions is required to generate the complex stretching and folding of trajectories characteristic of chaos. In four or more dimensions, even more complex phenomena like hyperchaos can emerge.

### From Theory to Practice: Analyzing Empirical Data

#### The Fundamental Challenge of Real-World Systems

We now pivot from idealized model systems to the domain of empirical data. As experimentalists or data scientists, we are often confronted with data recorded from a real-world system—be it temperature measurements from a climate system or neural recordings from the brain. We do not know the underlying equations of motion.

The central questions we will now address are:

1. How can we apply the powerful concepts of dynamical systems theory to characterize and understand real data?
2. Can we go even further and actually derive data-driven models of the underlying data-generating process?

#### The Measurement Problem

Let's formalize the typical experimental setup. We assume there is some underlying, unknown data-generating system that can be described as a dynamical system:

$$\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}) \quad \text{where} \quad \mathbf{x} \in \mathbb{R}^m$$

Here, $\mathbf{x}$ represents the complete state of the system in an $m$-dimensional state space. For now, we will assume the system is autonomous (the function $\mathbf{f}$ does not explicitly depend on time).

The crucial challenge is that we never observe the full state vector $\mathbf{x}$ directly. Instead, we use a recording device (e.g., a thermometer, an electrode, a microphone) which acts as a measurement function, $h$. This function takes the true system state $\mathbf{x}$ and transforms it into a measurable quantity. Furthermore, we can only record this quantity at discrete points in time.

This means the data we actually have is a time series of scalar observations $s(t_k)$, given by:

$$s(t_k) = h(\mathbf{x}(t_k))$$

Our goal is to reconstruct the dynamics of the full $m$-dimensional system from this limited, one-dimensional time series. This seemingly impossible task is the subject of our next topic: temporal delay embeddings.

### State Space Reconstruction from Time Series Data

In the study of complex dynamical systems, such as the climate or the human brain, we are often faced with a significant challenge: the systems are defined by a vast number of interacting variables, yet we can only measure a small fraction of them. This chapter introduces the foundational concepts that allow us to reconstruct the system's underlying dynamics from limited, often scalar, time series data.

#### The Fundamental Problem: Incomplete Observations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Observational Setup)</span></p>

Let a true dynamical system evolve in an $M$-dimensional state space. We typically do not have access to all $M$ variables. Instead, we observe a time series of measurements, denoted as $y_t$, which are captured at discrete time intervals.

* The measurement times are given by $t = k \cdot \Delta t$, where $\Delta t$ is the inverse of the sampling frequency and $k$ is a natural number.
* The observations $y_t$ form a time series of finite length, where $t$ ranges from $1$ to $T$.
* These observations reside in a $k$-dimensional space, where it is common for the measurement dimension $k$ to be much smaller than the true system dimension $M$ (i.e., $k \ll M$).

</div>

For the purposes of our initial discussion, we will consider the most extreme case where our measurements are scalar, meaning $k=1$. The central question is: can we recover the dynamics of the true, high-dimensional system from a single scalar time series? The answer, remarkably, is yes. The mathematical theorems that support this form the basis of state space reconstruction.

#### The Intuition of Time Delay Embedding

The core idea behind reconstruction is that if the variables in a system are all coupled, then the dynamics of the entire system will leave a "trace" on any single variable we measure. While the instantaneous value of our measurement, $y_t$, may be ambiguous, its history contains the information needed to resolve the state of the system.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(A Simple Harmonic Oscillator)</span></p>

Imagine we have a scalar measurement from a simple harmonic oscillator, which looks like a sine wave.

Conceptual sketch of a simple oscillator's time series.

If we observe a single point on this curve, say at a specific time $t$, we face an ambiguity: is the system's value about to increase or decrease? We cannot predict the future from this single point.

However, if we also consider the value at a slightly earlier time, $t-\tau$, we gain crucial context. By knowing the preceding value, we can determine the direction of the trajectory. In other words, a two-dimensional state space spanned by $(y_t, y_{t-\tau})$ is sufficient to resolve the ambiguity. In this new space, the trajectory of the simple oscillator would form a clean, non-intersecting closed loop.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Recall that a fundamental property of a valid state space is that trajectories cannot intersect. A single dimension is often insufficient to satisfy this property for even simple oscillatory systems. By augmenting our current measurement with its own past values, we effectively create new dimensions, allowing us to "unfold" the trajectory into a higher-dimensional space where intersections are eliminated.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(A More Complex Oscillator)</span></p>

Consider a more complex, non-sinusoidal oscillation, similar to an electrocardiogram (ECG) signal, which features sharp spikes and other intricate patterns.

Conceptual sketch of a complex oscillator's time series.

In this case, many points on the time series have the same amplitude. If we attempt to project this into a two-dimensional delay space $(y_t, y_{t-\tau})$, we might still find that the trajectory intersects with itself.

Conceptual sketch of a 2D projection with self-intersection.

To resolve these intersections, we may need to include more of the system's history. By moving to a three-dimensional space, using the coordinates $(y_t, y_{t-\tau}, y_{t-2\tau})$, we can often fully resolve the trajectory, creating a smooth, non-intersecting object.

</div>

#### Formalizing the Delay Coordinate Map

The intuitive process of using past values to construct a state space vector can be formalized mathematically.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Delay Coordinate Map)</span></p>

The **delay coordinate map** is an operation that transforms a scalar time series, $y_t \in \mathbb{R}$, into a vector in an $m$-dimensional space. This vector, known as the delay embedding vector, is constructed as follows:

$$\mathbf{Y}_t = (y_t, y_{t-\tau}, y_{t-2\tau}, \dots, y_{t-(m-1)\tau})$$

This map is defined by two crucial parameters:

* $m$: The embedding dimension, which is the dimensionality of the new state space vector.
* $\tau$: The time lag or time delay, which determines the time separation between the components of the vector.

</div>

#### Choosing the Embedding Parameters

The success of the reconstruction critically depends on the appropriate choice of the time lag $\tau$ and the embedding dimension $m$.

**The Time Lag $\tau$**

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Trade-off in Choosing $\tau$)</span></p>

The choice of $\tau$ presents a delicate balance, especially in the presence of measurement noise.

* If $\tau$ is too small ($\tau \to 0$): The components of the delay vector, $y_t$ and $y_{t-\tau}$, become highly correlated. In the noise-free limit, this might be acceptable. However, with even a small amount of noise, the reconstructed trajectory will collapse onto a lower-dimensional object (e.g., the diagonal line where $y_t \approx y_{t-\tau}$), and the underlying structure will be obscured by this "jitter."
* If $\tau$ is too large: The components of the vector, $y_t$ and $y_{t-\tau}$, may become completely uncorrelated due to the chaotic nature of the system or the influence of noise. The reconstructed points will appear to jump erratically in the embedding space, and any deterministic structure will be lost.

Therefore, $\tau$ must be chosen in an intermediate range: large enough to avoid redundancy, but small enough to preserve the correlation that defines the system's dynamics.

</div>

A common empirical method for selecting $\tau$ is to examine the autocorrelation function of the time series. This function measures the linear correlation between measurements as a function of their time separation.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Empirical Autocorrelation Function)</span></p>

The **empirical autocorrelation function**, $R_y(\tau)$, for a time series $y_t$ with sample mean $\bar{y}$ and variance $S_y^2$ is calculated as:

$$R_y(\tau) = \frac{\sum_{t} (y_t - \bar{y})(y_{t-\tau} - \bar{y})}{S_y^2}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The numerator represents the covariance between the time series and a time-lagged version of itself, while the denominator normalizes this value. A typical heuristic is to choose $\tau$ to be near the first minimum of the autocorrelation function. This ensures that the information in $y_t$ and $y_{t-\tau}$ is sufficiently different, while still being dynamically related.

Other, more advanced methods exist, such as using the mutual information, which is a nonlinear measure of dependency between variables. It is crucial to remember that the problem of choosing an optimal $\tau$ is primarily an empirical one that arises from the presence of noise. Theoretical derivations, which operate in the noise-free limit, do not provide a unique prescription for $\tau$.

</div>

**The Embedding Dimension $m$**

The choice of the embedding dimension $m$ is a more fundamental question that can be addressed systematically. The goal is to choose an $m$ that is large enough to fully "unfold" the attractor from its projection in the measurement space.

#### Properties of a "Good" Embedding

To formalize what we require from our reconstruction, we define a complete map from the original, true state space to our new, reconstructed space.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Delay Embedding Map)</span></p>

Let the original system's attractor live on a manifold $A$ in the true state space. The full **delay embedding map**, which we will call $F$, is a composition of two functions:

$$F := G \circ h(x)$$

1. The measurement function, $h$, which maps a point $x$ on the true attractor to a scalar observation $y$.
2. The delay coordinate map, $G$, which takes the time series of observations and constructs the delay vector.

The map $F$ takes the manifold $A$ to its image, $F(A)$, in the $m$-dimensional delay embedding space (also called the reconstruction space).

$$A \xrightarrow{F} F(A)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Desired Properties of the Map $F$)</span></p>

For the reconstruction $F(A)$ to be scientifically useful, the map $F$ must preserve essential properties of the original system on $A$.

1. Topological Preservation (One-to-One Mapping): At a minimum, we require the map $F$ to be one-to-one (injective). This ensures that distinct points on the original attractor are mapped to distinct points in the reconstruction space. This property guarantees that trajectories do not intersect, thereby creating a topologically faithful representation (a homeomorphism).
2. Dynamical Preservation (Preserving the Vector Field): For a truly useful embedding, preserving the topology is not enough. We also want to preserve the local dynamics—the vector field on the attractor. This is a much stronger condition. It requires that the derivative of the map, denoted $dF$, also be one-to-one. The map $dF$ is a linear map between the tangent spaces (local vector fields, or Jacobians) at any point on $A$ and its corresponding image on $F(A)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Since $dF$ is a linear map between tangent spaces, the condition that it be one-to-one is equivalent to requiring that $dF$ has full rank. This ensures that the local geometric structure of the dynamics is preserved in the reconstruction. The famous Takens' Embedding Theorem provides the conditions under which such a map is guaranteed to exist.

</div>

### Embedding Theorems: From Geometry to Dynamics

#### The Concept of Diffeomorphic Equivalence

In the study of dynamical systems, we are often concerned with transformations between state spaces. A key goal is to find a mapping that preserves the essential geometric and dynamic properties of the system. This leads us to the concept of a diffeomorphism, a particularly strong form of equivalence between two systems.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Diffeomorphism)</span></p>

A homeomorphism is a continuous, one-to-one mapping between two spaces that has a continuous inverse. It preserves topological properties.

A **diffeomorphism** is a map that is a homeomorphism, but with the additional requirement that both the map and its inverse are continuously differentiable. This means that a diffeomorphism is not only one-to-one in its points but also in its derivatives.

The derivative of the mapping, often denoted as $DF$, is also called a push-forward. It effectively transports the vector field from one domain to another. For a mapping to be a diffeomorphism, this transportation of the vector field must also be a one-to-one relationship.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Diffeomorphism Matters)</span></p>

Consider a trajectory in a three-dimensional space that we wish to represent, or "embed," in a lower-dimensional space. A simple one-to-one projection (a homeomorphism) is not sufficient. While it would prevent the trajectory from intersecting itself, it could introduce artificial and misleading features into the system's dynamics.

For instance, a smooth, flowing trajectory in the original space might be projected in such a way that its corresponding vector field in the embedded space exhibits sudden, sharp jumps. A diffeomorphism prevents this. It guarantees that any smooth changes in the original vector field correspond to smooth changes in the embedded vector field.

The central challenge addressed by embedding theorems is finding the right embedding—specifically, the right embedding dimensionality—that guarantees the existence of such a diffeomorphism.

</div>

#### Intuition: Avoiding Intersections in Higher Dimensions

The core principle behind embedding theorems can be understood through a simple geometric question: Given two manifolds, one of dimension $d_1$ and the other of dimension $d_2$, how large must the dimension of the ambient space be so that the manifolds will most likely not intersect?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The required dimensionality of the ambient space is at least $d_1 + d_2 + 1$. In a space of this dimension or higher, the probability of two randomly placed manifolds intersecting is zero (or, more formally, the measure of the set of intersecting configurations is zero).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

* Lines in 2D vs. 3D: Imagine randomly "throwing" straight lines (1-dimensional manifolds, so $d_1 = d_2 = 1$) into a space.
  * In a 2-dimensional plane, it is very likely that two randomly placed lines will intersect (unless they are perfectly parallel).
  * In a 3-dimensional space, however, two randomly placed lines will almost certainly not intersect. They will pass by each other as skew lines. With probability one, they will miss.

This powerful intuitive idea—that increasing the dimension of the ambient space provides "more room" to avoid intersections—is the foundation upon which the formal embedding theorems are built.

</div>

#### The Whitney Embedding Theorem

The first formalization of this intuition comes from the field of differential geometry. The Whitney Embedding Theorem provides the conditions under which a smooth manifold can be embedded into a Euclidean space of a certain dimension.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Whitney's Embedding Theorem, c. 1930s)</span></p>

Let $A \subset \mathbb{R}^m$ be a compact, smooth, and differentiable manifold of dimension $D$.

Then, almost any smooth map $f: \mathbb{R}^m \to \mathbb{R}^k$ where $k \geq 2D + 1$ is an embedding of $A$.

An embedding in this context is a diffeomorphism onto its image.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On "Almost Any" and "Generic" Maps)</span></p>

The phrase "almost any" is a probabilistic statement, meaning "with probability one." It acknowledges that while maps that fail to be embeddings might exist, they are exceptionally rare.

This is often expressed using the term "generic." A property is generic if the set of maps possessing that property is dense in the overall space of possible measurement functions. This has a powerful practical implication: if you choose a map $f$ that does not work, any slight perturbation of $f$ will produce a map that does the job.

</div>

#### Takens' Delay Embedding Theorem for Dynamical Systems

Floris Takens, in a landmark 1981 paper, adapted the principles of Whitney's theorem specifically for the analysis of dynamical systems observed through time series data. This theorem provides the theoretical justification for the method of delay-coordinate embedding.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Takens' Delay Embedding Theorem)</span></p>

Assume $A \subset \mathbb{R}^m$ is a $D$-dimensional smooth manifold that is invariant under the flow $\Phi$ of a dynamical system.

Let $h$ be a generic measurement function, and let $F_{\tau} = G_{\tau} \circ h$ be a delay coordinate map from $\mathbb{R}^m$ to $\mathbb{R}^k$, constructed with a generic delay $\tau$.

If the embedding dimension $k$ satisfies $k \geq 2D + 1$, then the map $F_{\tau}$ is an embedding (a diffeomorphism).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Invariance and Generic Delay)</span></p>

* Invariance: A set $A$ is invariant under a flow $\Phi$ if any trajectory starting in $A$ remains in $A$ for all time. That is, $\Phi_t(x) \in A$ for all $x \in A$ and for all time $t$. The theorem applies to the attractors of dynamical systems, which are by definition invariant sets.
* Generic Delay ($\tau$): The choice of the time delay $\tau$ is crucial. A "bad" choice can ruin the embedding. For example, if the system is periodic, choosing $\tau$ to be an integer multiple of the period would be a poor choice, as all delay coordinates would be identical, and the structure would collapse. The "generic" condition means that almost any choice of $\tau$ will work, as long as it avoids such special, resonant values. If a choice of $\tau$ fails, a slightly different value will succeed.

</div>

#### The Fractal Delay Embedding Prevalence Theorem

Many dynamical systems, particularly chaotic ones, have attractors that are not smooth manifolds but are instead fractal sets. The theorem by Sauer, Yorke, and Casdagli (1991) extends Takens' results to cover these more complex geometric objects by using the box-counting dimension.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Fractal Delay Embedding Prevalence Theorem)</span></p>

Assume $A \subset \mathbb{R}^m$ is a compact subset that is invariant under the flow $\Phi$, and let its box-counting dimension be $D_{box}$.

Let $h$ be a generic measurement function, and let $F_{\tau} = G_{\tau} \circ h$ be a delay coordinate map from $\mathbb{R}^m$ to $\mathbb{R}^k$, constructed with a generic delay $\tau$.

If the embedding dimension $k$ satisfies $k > 2 D_{box}$, then the map $F_{\tau}$ is a delay embedding (a diffeomorphism).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Power of Embedding)</span></p>

Together, these theorems provide a rigorous foundation for a powerful technique. They demonstrate that from a single, observed time series, one can construct a state space that is topologically equivalent to the original, unobserved system. This reconstructed space preserves not only the geometry of the attractor but also the dynamics (the vector field) upon it.

</div>

#### Practical Estimation of Embedding Dimension: The False Neighbors Method

The embedding theorems provide a lower bound for the embedding dimension ($k \geq 2D + 1$ or $k > 2 D_{box}$), but in practice, the dimension $D$ or $D_{box}$ of the underlying attractor is often unknown. The False Neighbors method (Kennel et al., 1992) is a practical, empirical technique to estimate a sufficient embedding dimension directly from the data.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Principle of False Neighbors)</span></p>

The fundamental principle is that the orbits of a deterministic dynamical system cannot intersect. In an embedding space with insufficient dimension, the projection of the attractor can cause points that are far apart on the true attractor to appear as close neighbors in the projected space. These are "false neighbors."

As the embedding dimension is increased, the attractor "unfolds." When the dimension is sufficient, these false neighbors will move apart, revealing their true, larger distance from each other. The method works by tracking this unfolding process.

</div>

**Methodology**

1. Construct Embeddings: Start with a low embedding dimension, $d=1, 2, 3, \dots$.
2. Identify Neighbors: For each point in the $d$-dimensional embedded trajectory, find its nearest neighbor.
3. Check in Next Dimension: Observe the distance between this same pair of points in the $d+1$-dimensional embedding.
4. Count False Neighbors: If the distance between the points increases dramatically when moving from dimension $d$ to $d+1$, the pair is classified as a "false neighbor." This jump in distance indicates that the proximity in dimension $d$ was merely an artifact of the projection.
5. Analyze the Trend: Plot the percentage of false neighbors as a function of the embedding dimension $d$. Typically, this curve will show a high percentage of false neighbors for low $d$, which then drops sharply and plateaus at or near zero. The embedding dimension at which this "kink" occurs and the percentage of false neighbors becomes negligible is chosen as the minimum sufficient embedding dimension.

#### Applications: Computing Invariants in Embedded Space

Once a proper embedding has been constructed, the reconstructed state space can be used to compute key dynamical invariants of the original system, even if that system was never directly observed.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Computable Quantities)</span></p>

* Attractor Dimension: If an estimate of the attractor's dimension (e.g., box-counting dimension, correlation dimension) was not available beforehand, it can be computed from the trajectories in the newly constructed delay-embedding space.
* Lyapunov Exponents: The delay-embedding space allows for the empirical estimation of Lyapunov exponents, which quantify the rate of separation of infinitesimally close trajectories and are a hallmark of chaotic systems.

</div>

**Method for Estimating Lyapunov Exponents**

1. In the delay-embedding space, for each point on a trajectory, identify a neighborhood of nearby points.
2. Track the evolution of the distance between the reference point and each of its neighbors over time. For initially close trajectories with separation $\delta(0) = \Delta_0$, the separation typically grows exponentially for a chaotic system: 
   
   $$\delta(t) \approx \Delta_0 e^{\lambda t}$$

3. By taking the logarithm, we get 
   
   $$\ln(\delta(t)) \approx \ln(\Delta_0) + \lambda t$$

4. By averaging the evolution of $\ln(\delta(t))$ over many initial points and their respective neighbors, one can plot this quantity against time $t$ (or discrete time steps $k \cdot \Delta t$). The slope of the initial linear region of this plot provides an empirical estimate of the largest Lyapunov exponent, $\lambda$.

### From Data to Dynamics: Reconstruction and Inference

This chapter bridges the gap between the theoretical analysis of dynamical systems and the practical challenge of working with real-world, observational data. We begin by revisiting the powerful technique of delay embedding, which allows us to reconstruct a system's state space from a single time series. We then explore how to compute key dynamical invariants, such as Lyapunov exponents, within this reconstructed space. Finally, we pose a more fundamental question: Can we move beyond merely characterizing a system to inferring its underlying mathematical model directly from data? This leads us to the modern intersection of dynamical systems and machine learning, setting the stage for advanced reconstruction techniques.

#### A Recap on Delay Embedding

The delay embedding theorems provide a remarkable guarantee: under certain conditions, we can reconstruct a state space that is topologically equivalent (diffeomorphic) to the original, unseen state space of a dynamical system using only a sequence of measurements from a single observable.

**The Delay Coordinate Map**

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Delay Coordinate Map)</span></p>

Given a time series of scalar measurements $\{s_1, s_2, ..., s_k\}$, the delay coordinate map constructs a series of state vectors in a higher-dimensional space. A vector $x_t$ in the reconstructed space is formed as:

$$x_t = (s_t, s_{t-\tau}, s_{t-2\tau}, ..., s_{t-(m-1)\tau})$$

Where:

* $m$ is the embedding dimension.
* $\tau$ is the time lag or delay time.

The collection of all such vectors $\{x_t\}$ forms a trajectory in an $m$-dimensional space that, with proper parameter choices, preserves the geometric and dynamic properties of the original system's attractor.

</div>

**The Role of Parameters: $m$ and $\tau$**

The success of delay embedding hinges on the careful selection of its two key parameters, $m$ and $\tau$.

* The Time Lag ($\tau$):
  * Theoretical Role: While theoretically less critical than $m$, $\tau$ must be chosen so that it does not align with a natural periodicity of the underlying signal, which would cause the coordinates to become correlated and the embedding to collapse.
  * Empirical Selection: A common heuristic is to choose $\tau$ based on the autocorrelation function of the time series. A value is often selected where the function first drops significantly, indicating that $s_t$ and $s_{t-\tau}$ are sufficiently independent to serve as distinct coordinates.
* The Embedding Dimension ($m$):
  * Theoretical Role: This parameter is theoretically crucial. Takens' theorem guarantees a successful embedding if the dimension $m$ is at least twice the box-counting dimension of the underlying attractor ($D_{box}$): $$m > 2 D_{box}$$
  * Empirical Selection: The false neighbor technique is a practical method to determine an appropriate $m$. It works by checking how many "neighboring" points in an $m$-dimensional embedding remain neighbors when the dimension is increased to $m+1$. If they are no longer neighbors, they were "false neighbors" resulting from a projection into a space of insufficient dimension. One increases $m$ until the percentage of false neighbors drops to zero.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The goal of delay embedding is to "unfold" the trajectory. If you imagine a complex trajectory like a tangled ball of yarn (e.g., the Lorenz attractor), a 2D projection might show intersections that do not exist in the true 3D space. By choosing a sufficiently high embedding dimension $m$, we provide enough "room" for the trajectory to resolve these apparent self-intersections, creating a faithful representation of the system's dynamics. An improperly chosen $\tau$ can also distort the reconstruction; a $\tau$ that is too small will cause the embedded points to cluster along a diagonal line, while a $\tau$ that is too large can over-fold the attractor.

</div>

#### Estimating Lyapunov Exponents from Time Series

Once a state space has been successfully reconstructed via delay embedding, we can use the embedded trajectory to compute dynamical invariants. A key invariant is the maximum Lyapunov exponent ($\lambda_{max}$), which measures the average rate of exponential divergence of nearby trajectories.

**The Divergence Method**

The standard algorithm for estimating $\lambda_{max}$ from data operates on the reconstructed trajectory. It involves tracking the evolution of distances between initially close pairs of points. The slope of the average logarithmic divergence of these pairs over time provides an estimate of the exponent.

**Interpreting the Divergence Plot**

When plotting the average logarithmic distance between trajectory pairs versus time, a characteristic curve often emerges. Correctly interpreting this curve is essential for an accurate estimation of $\lambda_{max}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Anatomy of the Divergence Plot)</span></p>

A typical empirical divergence plot can be divided into three regions:

1. Initial Sharp Rise: The curve often begins with a steep, rapid increase. This is typically an artifact of noise in the measurement data. Noise is uncorrelated with the system dynamics and causes an initial, non-dynamical separation of points across all available dimensions.
2. Linear Slope: Following the initial noise-driven rise, the plot should exhibit a linear region. This slope reflects the true exponential divergence governed by the system's underlying dynamics. The slope of this linear portion is the estimate of the maximum Lyapunov exponent.
3. Plateau: Eventually, the curve will flatten out and plateau. This saturation occurs because the attractor is bounded in space. As trajectories diverge, their separation cannot grow indefinitely; it is limited by the maximum possible distance across the attractor. At this point, the average distance will fluctuate around this maximum value.

Therefore, the empirical task is to identify the linear region of this plot, fit a line to it, and extract its slope as the estimate for $\lambda_{max}$.

</div>

#### The Core Problem: Inferring Models from Data

While delay embedding and the calculation of invariants are powerful tools for characterizing a system, they do not provide a formal, predictive model of the system itself. As scientists, we often seek to uncover the underlying equations of motion. This leads to a central question for the remainder of our study:

Can we infer a formal model of an underlying dynamical system using only its observational time series data?

This approach, often called Dynamical Systems Reconstruction, aims to automate the discovery of models, moving beyond the classical scientific cycle of proposing a model, making predictions, and performing experiments to refine it.

**Two Fundamental Approaches: Vector Fields vs. Flow Maps**

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Model Inference Approaches)</span></p>

1. Vector Field Approximation: For a continuous-time system described by a set of differential equations $\dot{x} = f(x)$, this approach seeks to find a function $\hat{f}_{\theta}(x)$ that approximates the true underlying vector field $f(x)$. The function $\hat{f}$ belongs to a flexible function class parameterized by a set of parameters $\theta$.
2. Flow Operator Approximation: For a discrete-time system (or a discretized continuous system) where the state at the next time step is given by $x_{k+1} = \Phi(x_k)$, this approach seeks to find a function $\hat{F}_{\theta}(x_k)$ that directly approximates the flow operator (or map) $\Phi$.

</div>

**The Machine Learning Framework for Reconstruction**

Modern approaches to this problem leverage the framework of machine learning. This involves a three-step process to find a good model approximator from a general, powerful function class (e.g., a deep neural network) "blindly," without presupposing its specific mathematical form.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Machine Learning Pipeline)</span></p>

1. **Specify a Model:** We choose a highly flexible class of functions, such as a neural network, to serve as our model candidate, $\hat{f}\_{\theta}$ or $\hat{F}\_{\theta}$. The parameters $\theta$ represent the weights and biases of the network. We also must consider that our observations $y_t$ may be related to the true states $x_t$ via a measurement function $h_{\psi}$, which may also need to be learned: $y_t = h_{\psi}(x_t)$.
2. **Specify a Loss Function:** We define a loss function, $R(\theta \mid \text{data})$, which quantifies the discrepancy between our model's predictions and the observed data. This function could be a mean squared error, a likelihood function, or another metric of model quality. The loss function creates a "surface" over the parameter space.
3. **Training (Optimization):** We employ an iterative numerical optimization algorithm (often a variant of gradient descent) to search the parameter space for a set of parameters $\theta$ that minimizes the loss function. This "training" procedure refines the model until it provides the best possible fit to the data, hopefully corresponding to a global (or at least a good local) minimum of the loss function.

</div>

### SINDy: Sparse Identification of Nonlinear Dynamics

One of the first modern, interpretable methods for dynamical systems reconstruction is SINDy (Sparse Identification of Nonlinear Dynamics), introduced by Brunton, Proctor, and Kutz in 2016. It is a comparatively simple yet elegant approach that focuses on discovering the vector field of the system.

#### The Core Idea and Assumptions

SINDy operates on the assumption that the vector field of many physical systems can be expressed as a sparse linear combination of functions from a predefined library. "Sparse" means that only a few terms in the library are active, making the resulting model interpretable.

The method makes several key assumptions and has important caveats:

* **Vector Field Approach:** It directly approximates the vector field $f(x)$ in $\dot{x} = f(x)$.
* **State Variable Access:** It assumes measurements of the relevant state variables are available. If not, a delay embedding must first be performed to reconstruct the state space. The method, in its basic form, does not automatically discover the measurement function $h$.
* **Numerical Derivatives:** To connect the model to the data, it requires an estimate of the time derivatives from the time series. This is often done via finite differences:    
  
  $$\nabla x_t \approx \frac{x_{t+1} - x_t}{\Delta t}$$ 
  
  * **Caveat:** This step is highly sensitive to measurement noise. The process of taking differences amplifies high-frequency noise, which can lead to unreliable derivative estimates.

#### The Core Idea: Function Approximation

The central goal of the SINDy algorithm is to discover the governing equations of a dynamical system directly from time-series data. We begin with the assumption that the system can be described by an ordinary differential equation of the form:

$$\frac{d}{dt}\mathbf{x}(t) = \mathbf{f}(\mathbf{x}(t))$$

where $\mathbf{x}(t)$ is the state of the system at time $t$. The core challenge is that the function $\mathbf{f}$, which represents the underlying vector field, is unknown. The SINDy method addresses this by approximating $\mathbf{f}$ with a linear combination of candidate functions from a pre-defined library. The key innovation is to find a representation of $\mathbf{f}$ that is sparse, meaning most of the candidate functions have coefficients of zero, thus revealing the simplest mathematical model that describes the dynamics.

#### Constructing the Candidate Library

**Basis Functions**

To approximate the unknown function $\mathbf{f}$, we construct a library, or a set of basis functions, denoted by $\mathbf{\Theta}(\mathbf{x}(t))$. These are candidate functions that we believe might constitute the true underlying vector field. The choice of these functions is flexible and is determined by the person applying the algorithm.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Basis Function Library)</span></p>

A **basis function library** $\mathbf{\Theta}(\mathbf{x}(t))$ is a collection of candidate functions that depend on the state variables $\mathbf{x} = [x_1, x_2, ..., x_n]$. The approximation of the true vector field, $\hat{\mathbf{f}}(\mathbf{x}(t))$, is constructed as a linear combination of these basis functions.

$$\hat{\mathbf{f}}(\mathbf{x}(t)) = \mathbf{C} \cdot \mathbf{\Theta}(\mathbf{x}(t))$$

where $\mathbf{C}$ is a matrix of coefficients that determines the weight of each basis function in the approximation.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Candidate Basis Functions)</span></p>

The library can be constructed from a wide variety of functions. Common choices include:

* **Monomials:** Simple first-order terms like $x_1$, $x_2$, etc.
* **Polynomials:** Higher-order terms, such as squares ($x_1^2$), or multinomials ($x_1 x_2$, $x_2 x_3$).
* **Trigonometric Functions:** Terms like $\sin(x_i)$ or $\cos(x_j)$.
* **Other Functions:** Any other function could be included, such as radial basis functions.

A typical library for a system with state vector $\mathbf{x}(t)$ might include a constant term, linear terms, and polynomial terms up to a certain order. For instance:

$$\mathbf{\Theta}(\mathbf{x}(t)) = \begin{bmatrix} 1 \\ \mathbf{x}(t) \\ \mathbf{x}(t)^{\otimes 2} \\ \vdots \end{bmatrix}$$

where $\mathbf{x}(t)^{\otimes 2}$ represents all second-order polynomial terms.

</div>

**Theoretical Underpinning: The Stone-Weierstrass Theorem**

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Stone-Weierstrass Theorem)</span></p>

On a compact set, any continuous function can be approximated arbitrarily well by a polynomial function of a sufficiently high order.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Polynomials are a Good Starting Point)</span></p>

The Stone-Weierstrass theorem provides a strong theoretical justification for including polynomial terms in the basis function library. It suggests that, in theory, we can construct a polynomial function that accurately approximates the true underlying vector field $\mathbf{f}$. Many physical systems are already described by polynomials, lending further virtue to this choice.

However, there is a significant practical caveat. The theorem guarantees approximation but does not specify the required order of the polynomial. For complex systems, an "infinitely high order" might be necessary, which is computationally infeasible. This highlights a central challenge: the effectiveness of the algorithm depends heavily on the choice of the library.

</div>

#### The SINDy Algorithm: A Mathematical Formulation

**System Representation in Matrix Form**

To find the coefficients that define our model, we first express the problem in matrix notation. Our goal is to find a set of coefficients such that our approximation $\hat{\mathbf{f}}(\mathbf{x}(t))$ is as close as possible to the time derivative $\dot{\mathbf{x}}(t)$, which is derived numerically from the empirical data.

We define our approximation at a specific time $t$ as: 

$$\hat{\mathbf{f}}(\mathbf{x}(t)) = \mathbf{C} \cdot \mathbf{\Theta}(\mathbf{x}(t))$$

| Variable | Description | Dimensions |
|---|---|---|
| $\hat{\mathbf{f}}(\mathbf{x}(t))$ | The estimated time derivative of the state vector. | $n \times 1$ |
| $\mathbf{C}$ | The matrix of unknown coefficients we want to find. | $n \times (B+1)$ |
| $\mathbf{\Theta}(\mathbf{x}(t))$ | The library of basis functions evaluated at $\mathbf{x}(t)$. | $(B+1) \times 1$ |

Here, $n$ is the number of state variables and $B$ is the number of basis functions in our library.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The $B+1$ Dimension)</span></p>

The "$B+1$" dimension for the library and coefficient matrix is a standard statistical trick to handle a constant offset term (e.g., $c_0$). We augment the library vector with a leading '1':

$$\mathbf{\Theta}(\mathbf{x}(t)) = \begin{bmatrix} 1 \\ \psi_1(\mathbf{x}) \\ \psi_2(\mathbf{x}) \\ \vdots \\ \psi_B(\mathbf{x}) \end{bmatrix}$$

Correspondingly, the coefficient matrix $\mathbf{C}$ includes a first column, $[\mathbf{c}\_{01}, \dots, \mathbf{c}\_{0n}]^\top$, which represents the constant terms in the differential equations for each state variable. This simplifies the mathematical formulation by incorporating the offset directly into the matrix product.

</div>

**The Optimization Problem: Finding the Coefficients**

To find the optimal coefficient matrix $\mathbf{C}$, we define a loss function, $L(\mathbf{C})$, which measures the discrepancy between our model's prediction and the data-derived derivative. We aim to minimize this loss across all measured time steps. The standard choice for this is the Mean Squared Error.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Mean Squared Error Loss Function)</span></p>

The loss function $L(\mathbf{C})$ is defined as the sum of squared Euclidean distances between the numerically estimated derivative, $\hat{\dot{\mathbf{x}}}(t)$, and the model's approximation, $\hat{\mathbf{f}}(\mathbf{x}(t))$, summed over all time steps $T$:

$$L(\mathbf{C}) = \sum_{t=1}^{T} \| \hat{\dot{\mathbf{x}}}(t) - \hat{\mathbf{f}}(\mathbf{x}(t)) \|_2^2$$

Substituting the matrix form of our approximation, we get:

$$L(\mathbf{C}) = \sum_{t=1}^{T} \| \hat{\dot{\mathbf{x}}}(t) - \mathbf{C} \cdot \mathbf{\Theta}(\mathbf{x}(t)) \|_2^2$$

This can also be written in its quadratic form: 

$$L(\mathbf{C}) = \sum_{t=1}^{T} (\hat{\dot{\mathbf{x}}}(t) - \mathbf{C} \cdot \mathbf{\Theta}(\mathbf{x}(t)))^\top (\hat{\dot{\mathbf{x}}}(t) - \mathbf{C} \cdot \mathbf{\Theta}(\mathbf{x}(t)))$$

</div>

**The Role of Sparsity: Lasso Regularization**

Minimizing the mean squared error alone would typically result in a dense $\mathbf{C}$ matrix, where every basis function contributes to the model. This model would be complex and difficult to interpret. The core idea of SINDy is to find a sparse model.

To achieve this, we add a penalty term to the loss function that penalizes the number and magnitude of non-zero coefficients. This technique is known as Lasso Regression (Least Absolute Shrinkage and Selection Operator).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lasso Regularization Term)</span></p>

A regularization term, weighted by a coefficient $\lambda$, is added to the loss function. This term is the sum of the absolute values (the $L_1$-norm) of all coefficients in the matrix $\mathbf{C}$.

$$\text{Regularization Term} = \lambda \sum_{i,j} |c_{ij}|$$

The complete loss function with the regularization term is:

$$L(\mathbf{C}) = \sum_{t=1}^{T} \| \hat{\dot{\mathbf{x}}}(t) - \mathbf{C} \cdot \mathbf{\Theta}(\mathbf{x}(t)) \|_2^2 + \lambda \sum_{i,j} |c_{ij}|$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Sparsity Trade-off)</span></p>

The regularization term is the key to achieving sparsity. It forces some of the parameters to drop out (become exactly zero).

* The first term (mean squared error) pushes the model to fit the data as closely as possible.
* The second term (Lasso penalty) pushes the coefficients $c_{ij}$ towards zero.
* The parameter $\lambda$ controls the trade-off. A larger $\lambda$ results in a sparser model (more zero coefficients) at the potential cost of a poorer fit to the data. The optimization will try to strike a balance between these two competing objectives.

</div>

#### Solving for the System Dynamics

To minimize the loss function $L(\mathbf{C})$, we take its derivative with respect to $\mathbf{C}$ and set it to zero.

The derivative of the loss function is:

$$\frac{\partial L}{\partial \mathbf{C}} = \frac{\partial}{\partial \mathbf{C}} \left( \sum_{t=1}^{T} \| \hat{\dot{\mathbf{x}}}(t) - \mathbf{C}\mathbf{\Theta}(t) \|_2^2 \right) + \frac{\partial}{\partial \mathbf{C}} \left( \lambda \sum_{i,j} |c_{ij}| \right) = 0$$

This yields:

$$\frac{\partial L}{\partial \mathbf{C}} = -2 \sum_{t=1}^{T} [\hat{\dot{\mathbf{x}}}(t) - \mathbf{C}\mathbf{\Theta}(\mathbf{x}(t))] \mathbf{\Theta}(\mathbf{x}(t))^\top + \lambda \mathbf{D} = 0$$

where $\mathbf{D}$ is a matrix with elements $d_{ij} = \text{sign}(c_{ij})$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Linearity Advantage)</span></p>

A crucial feature of this formulation is that the model approximation, $\hat{\mathbf{f}}(\mathbf{x}(t)) = \mathbf{C} \cdot \mathbf{\Theta}(\mathbf{x}(t))$, is linear in the parameters $\mathbf{C}$. This makes the loss function quadratic (a squared function) in the parameters, which ensures that it has a unique minimum. This allows us to find an optimal solution by setting the derivatives to zero.

</div>

**Case 1: Standard Linear Regression (No Regularization)**

Let's first consider the simpler case where $\lambda=0$. This reduces the problem to a standard linear regression.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Deriving the Closed-Form Solution for $\mathbf{C}$ when $\lambda=0$)</span></p>

1. Start with the derivative equation without regularization: 
   
   $$-2 \sum_{t=1}^{T} [\hat{\dot{\mathbf{x}}}(t) - \mathbf{C}\mathbf{\Theta}(\mathbf{x}(t))] \mathbf{\Theta}(\mathbf{x}(t))^\top = 0$$

2. Divide by $-2$ and distribute the terms: 
   
   $$\sum_{t=1}^{T} \hat{\dot{\mathbf{x}}}(t) \mathbf{\Theta}(\mathbf{x}(t))^T - \sum_{t=1}^{T} \mathbf{C}\mathbf{\Theta}(\mathbf{x}(t)) \mathbf{\Theta}(\mathbf{x}(t))^\top = 0$$

3. Isolate the term containing $\mathbf{C}$. Since $\mathbf{C}$ does not depend on time $t$, we can pull it out of the summation: 
   
   $$\sum_{t=1}^{T} \hat{\dot{\mathbf{x}}}(t) \mathbf{\Theta}(\mathbf{x}(t))^\top = \mathbf{C} \left( \sum_{t=1}^{T} \mathbf{\Theta}(\mathbf{x}(t)) \mathbf{\Theta}(\mathbf{x}(t))^\top \right)$$
   
4. Solve for $\mathbf{C}$ by post-multiplying by the inverse of the term in the parenthesis:
   
   $$\hat{\mathbf{C}} = \left( \sum_{t=1}^{T} \hat{\dot{\mathbf{x}}}(t) \mathbf{\Theta}(\mathbf{x}(t))^\top \right) \left( \sum_{t=1}^{T} \mathbf{\Theta}(\mathbf{x}(t)) \mathbf{\Theta}(\mathbf{x}(t))^\top \right)^{-1}$$ 
   
   This provides a closed-form, explicit solution for the coefficient matrix $\mathbf{C}$. This is highly desirable as it is computationally fast and does not require a numerical iteration procedure.

</div>

**Case 2: Lasso Regression (With Sparsity Constraint)**

When $\lambda > 0$, the presence of the sign function in the derivative makes finding a closed-form solution more difficult. The optimization must find a set of coefficients $\mathbf{C}$ whose signs are consistent with the derivatives. If the constraints imposed by the two terms in the loss function cannot be satisfied for a given coefficient, the optimization will force that coefficient to become exactly zero. This is the mathematical mechanism that induces sparsity. Solving this system typically requires iterative numerical methods.

#### Practical Considerations and Examples

**Application to the Lorenz System**

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Recovering the Lorenz Equations)</span></p>

The SINDy algorithm was successfully applied to data generated from the Lorenz system.

1. A polynomial basis expansion was chosen as the candidate library.
2. The coefficient matrix $\mathbf{C}$ was estimated using the Lasso regression procedure.
3. The regularization forced many of the estimated coefficients to become zero.
4. The final, sparse model recovered a system of equations very close to the true Lorenz equations.

</div>

**A Note on Choosing the Library**

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The "Beauty" and the "Breakdown")</span></p>

The SINDy method is beautiful in its simplicity. It transforms a difficult nonlinear system identification problem into a linear regression problem. However, its success is highly dependent on the initial choice of the basis function library.

* When it works well: The Lorenz system example works nicely because the true system is already in a polynomial form, which was included in the library. SINDy is effective when the library is geared towards the system on which the discovery is being performed.
* When it can fail: For real empirical data, where the true functional form of the dynamics is unknown, the method can easily break down. If the essential functions are not included in the library, or if the system requires a very high-dimensional library, the problem can become computationally infeasible. Therefore, the choice of the library often requires physical domain knowledge.

</div>

## Lecture 9

### Introduction to Universal Approximators for Dynamical Systems

#### Beyond Pre-defined Libraries: The Need for Universal Approximators

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Motivation for Universal Approximators)</span></p>

In previous discussions, we explored methods for inferring dynamical systems from data, such as SINDy. A notable characteristic of such approaches is their reliance on a pre-defined library of functions that must be specified a priori. This requirement, while powerful in certain contexts, can be a limitation.

The focus of our study now shifts to a class of methods that do not have this caveat: deep learning methods. These models are known as universal approximators, capable of learning complex functions directly from data without the need to manually define a basis or library of candidate functions. This chapter will introduce a foundational deep learning architecture for modeling time-dependent systems.

</div>

#### Deep Learning and Recurrent Neural Networks (RNNs)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Universal Approximation of Dynamical Systems)</span></p>

All the methods discussed henceforth are universal approximators of functions in general, and of dynamical systems in particular. A key architecture in this domain is the Recurrent Neural Network (RNN).

An RNN can be formally shown to be a universal approximator of dynamical systems. While we will not delve into the formal proofs of the theorems that establish this property, we will build a comprehensive understanding of their structure, function, and application. These models were, and in many domains remain, state-of-the-art for time series prediction and the modeling of dynamical systems.

</div>

### The Architecture and Motivation of Recurrent Neural Networks

#### Neuroscientific Origins and Core Concepts

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Neuroscientific Origins)</span></p>

Like many foundational neural network architectures, RNNs have their roots in neuroscience and psychology, where they were initially introduced as abstract models of the brain. The core idea is to model a system of interconnected processing units, or neurons, that influence each other's activity over time.

The key components of this model are:

* **Units (Neurons):** These are the nodes of the network, each possessing an activation value at a given point in time. We can denote the activation of unit $i$ at time $t$ as $x_i^t$.
* **Synaptic Connections (Weights):** The units are coupled through connections, each having an associated weight, denoted $w_{ij}$, which represents the strength of the connection from unit $j$ to unit $i$. These weights are adjustable parameters learned from data.
* **Recurrent Connections:** The defining feature of RNNs is the presence of feedback connections. Unlike feed-forward architectures (like many Convolutional Neural Networks) where information flows in a single direction, RNNs can have connections that form cycles. This allows for both forward and backward connections between units, enabling the network to maintain an internal state or "memory" of past events. This is what makes the network recurrent.
* **External Inputs:** Some or all units may receive input from the external world. This input is represented as a time series, $S_t$.
* **Outputs:** Similarly, some or all units can produce outputs that are sent back to the external world.

</div>

#### Mathematical Formulation of an RNN

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(RNN Activation Dynamics)</span></p>

The activation of a specific unit $i$ at time $t$, denoted $x_i^t$, is determined by an activation function, $\phi$. This function processes a weighted sum of inputs from other units at the previous time step ($t-1$), any external inputs at the current time step ($t$), and a unit-specific bias term.

The general formulation for the activation of unit $i$ is:

$$x_i^t = \phi \left( \sum_j w_{ij} x_j^{t-1} + h_i + \sum_k c_{ik} S_k^t \right)$$

Where:

* $x_i^t$ is the activation of unit $i$ at time $t$.
* $\phi$ is a non-linear activation function.
* $w_{ij}$ is the connection weight from unit $j$ to unit $i$.
* $x_j^{t-1}$ is the activation of unit $j$ at the previous time step, $t-1$.
* $h_i$ is a unit-specific, learnable bias term.
* $c_{ik}$ is the weight for the $k$-th external input to unit $i$.
* $S_k^t$ is the value of the $k$-th external input at time $t$.

The learnable parameters of the model, which are adjusted during training, include the connection weights ($w_{ij}$), the input weights ($c_{ik}$), and the bias terms ($h_i$).

</div>

#### Historical Context

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Context)</span></p>

The foundational concepts and training algorithms for RNNs were first developed in the late 1980s and early 1990s. Key figures associated with their invention include Jeff Elman and Paul Werbos (referred to in the source as "Bar Palmer or Zipa").

For a significant period, RNNs were not widely popular in the broader machine learning community due to challenges in training them effectively. However, with advancements in algorithms and computational power, they have become indispensable tools, particularly for sequence and time-series data. We will return to the topic of training challenges and modern solutions later in the course.

</div>

### Formalizing Recurrent Neural Networks

#### From Scalar Operations to Vector Notation

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(From Scalars to Vectors)</span></p>

Recurrent Neural Networks (RNNs) are an inherently natural architectural choice for modeling time series and dynamical systems. Their structure, which processes information sequentially and maintains an internal state that evolves over time, mirrors the fundamental nature of such systems. This stands in contrast to other architectures, such as transformers, which may be adapted for these tasks but lack the intrinsic recursive formulation of an RNN.

To analyze these systems rigorously, we move from a component-wise description of individual network units to a more compact and powerful matrix notation. This allows us to treat the entire network's state as a single vector and its evolution as a unified vector-valued map.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The RNN in Vector Notation)</span></p>

A recurrent neural network's state at a discrete time step $t$ can be described by a vector of activation values, $z_t$. The evolution of this state from one time step to the next is governed by a recursive map.

The components of this map are:

* **State Vector:** $z_t \in \mathbb{R}^M$. This is a vector containing the activation values of the $M$ units in the network at time $t$. These are also referred to as latent states, as they represent an internal, unobserved configuration of the system.
* **Weight Matrix:** $W \in \mathbb{R}^{M \times M}$. A square matrix containing the weights of the connections between the network's units. This matrix does not change with time.
* **Bias Vector:** $h \in \mathbb{R}^M$. Also known as a bias term, this vector applies a constant offset to the pre-activation of each unit, biasing it towards a particular activity regime.
* **External Input Vector:** $s_t \in \mathbb{R}^K$. An optional vector representing $K$ external inputs to the system at time $t$. The dimensionality $K$ does not need to equal the internal state dimensionality $M$.
* **Input Weight Matrix:** $C \in \mathbb{R}^{M \times K}$. This matrix maps the $K$-dimensional external input space to the $M$-dimensional latent state space.
* **Nonlinear Activation Function:** $\phi(\cdot)$. A scalar function (e.g., a sigmoid) that is applied element-wise to the pre-activation vector.

The state update equation, which describes the evolution of the network, can be written as:

$$z_t = \phi(W z_{t-1} + C s_t + h)$$

More generally, we can express the dynamics of an RNN as a function $f$ parameterized by a set of parameters $\theta$, which includes $W$, $C$, and $h$.

$$z_t = f(z_{t-1}, s_t; \theta)$$

This formulation makes it explicit that the state at time $t$ is a function of the state at the previous time step, $t-1$, and any external inputs at time $t$.

</div>

#### The RNN as a Discrete-Time Dynamical System

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(RNNs as Dynamical Systems)</span></p>

The recursive formulation $z_t = f(z_{t-1}, s_t; \theta)$ is of profound importance. It reveals that a recurrent neural network is, in essence, a discrete-time, multi-dimensional recursive map. This directly parallels the discrete-time maps, such as the logistic map, that are central to the study of dynamical systems.

This connection is not merely an analogy; it has direct and critical consequences. Because an RNN is a discrete dynamical system, it is subject to the full range of complex behaviors that these systems can exhibit. Specifically, depending on the parameters ($\theta$) and initial conditions ($z_0$), an RNN can:

* Converge to different fixed points.
* Exhibit periodic behavior (cycles).
* Undergo bifurcations as its parameters are changed.
* Display chaotic dynamics.

Understanding these potential behaviors is crucial for both analyzing and effectively training these networks.

</div>

### Training Recurrent Neural Networks

#### The Gradient Descent Paradigm

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Gradient Descent)</span></p>

While numerous techniques exist for training machine learning models, the field is overwhelmingly dominated by methods based on gradient descent. This is not because gradient descent is the most powerful optimization technique available -- other methods may yield more accurate parameter estimates. Rather, its dominance stems from its effectiveness and scalability. Gradient descent-based techniques are generally well-understood, straightforward to implement, and scale favorably with the size of the dataset. For these reasons, it is the primary method for training RNNs.

The objective of training is to find a set of model parameters that minimizes a given loss function. Since RNNs are highly nonlinear devices, it is impossible to find an analytical, closed-form solution for the optimal parameters. We must therefore rely on iterative numerical optimization algorithms like gradient descent.

</div>

#### Defining the Core Components for Training

To train an RNN, we require three fundamental components: a dataset, a model architecture that links latent states to observable outputs, and a loss function to quantify performance.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Dataset)</span></p>

The training data consists of a set of $P$ patterns or sequences. For each pattern $p \in \lbrace 1, \dots, P \rbrace$, the dataset provides:

* **Inputs:** A sequence of input vectors $\lbrace s_t^{(p)} \rbrace_{t=1}^{T_p}$, where $s_t^{(p)} \in \mathbb{R}^K$. These are optional, depending on the task.
* **Desired Outputs (Targets):** A sequence of target vectors $\lbrace x_t^{(p)} \rbrace_{t=1}^{T_p}$, where $x_t^{(p)} \in \mathbb{R}^N$. These are the ground-truth values the model should aim to produce.

Here, $T_p$ is the length of the $p$-th sequence.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Model Architecture: State Dynamics and Observation)</span></p>

The complete model consists of two parts:

1. **The Recursive Core (State Equation):** This is the RNN itself, which describes the evolution of the latent states $z_t$.

$$z_t = f(z_{t-1}, s_t; \theta)$$

2. **The Decoder (Observation Model):** This is a function, $g$, that maps the latent state $z_t$ to a predicted output $\hat{x}_t$. This is necessary because the latent states are not directly observed; the decoder must learn to translate them into the space of the target outputs.

$$\hat{x}_t = g(z_t; \lambda)$$

The decoder has its own set of parameters, denoted by $\lambda$.

This two-part structure is analogous to concepts in dynamical systems where an unobservable internal state generates observable measurements. Models of this form are sometimes referred to as State-Space Models, though this term often implies additional probabilistic assumptions.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Linear Decoder)</span></p>

A simple and common choice for the decoder $g$ is a linear mapping, also known as a linear layer in neural network terminology:

$$\hat{x}_t = B z_t$$

Here, the parameter set $\lambda$ is simply the matrix $B \in \mathbb{R}^{N \times M}$, which maps the $M$-dimensional latent space to the $N$-dimensional output space. The dimensionality of the latent space, $M$, is a design choice and does not need to be equal to the input or output dimensions.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Loss Function)</span></p>

The loss function, $L$, quantifies the discrepancy between the model's predicted outputs and the true target outputs. It is a function of the model's parameters ($\theta$ and $\lambda$). The goal of training is to minimize this function. A common and straightforward choice is the Sum of Squared Errors (SSE) loss, which is calculated by summing the squared deviations over all time steps and all patterns in the dataset.

Given the observed output $x_t^{(p)}$ and the predicted output $\hat{x}_t^{(p)}$, the SSE loss is:

$$L(\theta, \lambda) = \sum_{p=1}^{P} \sum_{t=1}^{T_p} \| x_t^{(p)} - \hat{x}_t^{(p)} \|^2$$

While SSE is used here for concreteness, any differentiable loss function (e.g., likelihood functions) can be used within the gradient descent framework. The fundamental goal remains the same: to adjust the network's parameters to make the predicted output as close as possible to the observed output.

</div>

#### The Gradient Descent Algorithm

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Gradient Descent Overview</span></p>

Gradient descent is an iterative algorithm that seeks to find a minimum of the loss function. The process begins with an initial guess for the parameters and repeatedly adjusts them in the direction that most steeply decreases the loss.

**Algorithm Outline:**

1. **Initialization:** Start with an initial guess for the parameters, $\theta_0$ and $\lambda_0$. A common practice is to draw these initial values from a probability distribution, such as a Gaussian distribution with zero mean.

$$\theta_0, \lambda_0 \sim \mathcal{N}(0, \sigma^2 I)$$

2. **Iteration:** Initialize an iteration counter, e.g., $k=1$. Begin a loop that continues until a stopping criterion is met. In each step of the loop, the parameters are updated based on the gradient of the loss function. (The process of calculating the gradient and performing the update will be detailed subsequently.)

</div>

### Gradient Descent-Based Training

#### The Core Idea: Minimizing the Loss Function

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Intuition for Gradient Descent</span></p>

The central goal of training a model is to find a set of parameters, which we'll denote by $\theta$, that minimizes a loss function, $L(\theta)$. This function quantifies how poorly our model is performing on a given dataset; a lower loss value corresponds to a better model.

The idea behind gradient descent is intuitive: we start with an initial guess for our parameters $\theta$ and iteratively update them by taking small steps in the direction that most steeply decreases the loss. The gradient of the loss function, $\nabla L(\theta)$, points in the direction of the steepest ascent. Therefore, to minimize the loss, we must move in the opposite direction of the gradient.

Imagine a hilly landscape where the altitude represents the loss value for any given parameter set $\theta$. Our goal is to find the lowest valley.

* If we are on a slope where the gradient is positive, we need to move in the negative direction (downhill).
* If we are on a slope where the gradient is negative, we need to move in the positive direction (also downhill).

In both cases, we "go against the gradient." This iterative process continues until we reach a point where the loss is sufficiently low, ideally a minimum.

It is important to note that in classical machine learning, the aim was often to find the global optimum -- the single best parameter set that corresponds to the absolute lowest point in the loss landscape. However, in modern practice, finding the global optimum is often not feasible, nor is it always desirable. As we will discuss later, forcing a model to the global minimum on the training data can lead to a phenomenon known as overfitting.

</div>

#### The Gradient Descent Algorithm

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">Gradient Descent</span></p>

Gradient Descent is an iterative optimization algorithm used to find a local minimum of a differentiable function. The parameters are updated at each step $n$ according to the following rule:

$$\theta_n = \theta_{n-1} - \gamma \nabla L(\theta_{n-1})$$

Where:

* $\theta_n$ is the vector of parameters at iteration $n$.
* $\theta_{n-1}$ is the vector of parameters from the previous iteration.
* $\gamma$ is a positive scalar known as the learning rate, which controls the size of the step taken at each iteration.
* $\nabla L(\theta_{n-1})$ is the gradient of the loss function $L$ evaluated at the parameters $\theta_{n-1}$.

The gradient $\nabla L(\theta)$ is a vector of partial derivatives:

$$\nabla L(\theta) = \left[ \frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, \dots, \frac{\partial L}{\partial \theta_L} \right]$$

where $\theta_1, \dots, \theta_L$ are the individual parameters of the model.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">A Simple Gradient Descent Loop</span></p>

A simple implementation of this algorithm can be formulated as a while loop, which continues as long as the improvement in loss is significant and a maximum number of iterations has not been reached.

**Algorithm:**

1. Initialize parameters $\theta_0$ and counter $n = 0$.
2. Set a learning rate $\gamma > 0$, a minimum loss change threshold $\epsilon$, and a maximum number of iterations $N_{\max}$.
3. **while** $\Delta L(\theta) > \epsilon$ **and** $n < N_{\max}$:
   * Calculate the gradient: $g = \nabla L(\theta_n)$
   * Update the parameters: $\theta_{n+1} = \theta_n - \gamma g$
   * Increment the counter: $n = n + 1$

This core idea forms the basis for the most common optimization procedures in machine learning. While more sophisticated versions are implemented in standard toolboxes, they are fundamentally built upon this principle. The same concept can be used for maximization (e.g., in Maximum Likelihood Estimation) by simply moving with the gradient (i.e., using a $+$ sign instead of a $-$), which is equivalent to minimizing the negative of the function.

</div>

#### The Training Algorithm as a Dynamical System

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Training as a Dynamical System</span></p>

This is a fundamentally important point. The iterative update rule of gradient descent:

$$\theta_n = \theta_{n-1} - \gamma \nabla L(\theta_{n-1})$$

is a recursive procedure. It defines the state of the parameters at step $n$ based on their state at step $n-1$. This structure is precisely what defines a discrete-time dynamical system.

The implications of this are profound:

* The entire toolset of dynamical systems theory can be applied to analyze the training process itself.
* Just like the dynamical systems we have studied, the training process can exhibit complex behaviors. The parameter updates can:
  * **Converge to a fixed point:** This is often the desired outcome, as a fixed point where the gradient is zero corresponds to a local minimum (or a saddle point). Hopfield networks are an example where fixed points are defined to be local minima.
  * **Become oscillatory:** The parameters might not settle down, but instead cycle through a set of values.
  * **Exhibit chaos:** The parameter updates could be chaotic, never converging or repeating in a predictable pattern.

This perspective reveals that training a neural network is not just a simple optimization problem but a dynamical process with its own stability properties and potential complexities. We will see the consequences of this in the next section.

</div>

### Challenges in Gradient-Based Optimization

While powerful, the gradient descent algorithm is not without its challenges. The nature of the loss landscape -- the high-dimensional surface defined by $L(\theta)$ -- can introduce significant difficulties for the optimization process.

#### Local Minima and Saddle Points

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Local Minima and Saddle Points</span></p>

A primary issue in gradient-based optimization is that the algorithm can get "stuck." The update rule relies on the gradient to determine the direction of movement. At any point where the gradient is zero ($\nabla L(\theta) = 0$), the update step becomes zero, and the algorithm halts.

These points can be:

* **Local Minima:** These are valley bottoms in the loss landscape that are not the single lowest point (the global minimum). If the algorithm converges to a local minimum, the resulting model may be suboptimal.
* **Saddle Points:** These are points that are a minimum along one dimension but a maximum along another. The gradient is also zero here, causing the algorithm to stall.

In either case, the optimizer may fail to find a better solution, even if one exists elsewhere in the parameter space.

</div>

#### Widely Differing Loss Slopes and Learning Rate Selection

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">The Learning Rate Dilemma</span></p>

The geometry of the loss landscape presents another major challenge related to selecting an appropriate learning rate ($\gamma$). Loss functions for complex models are rarely smooth, uniform bowls. They often contain regions of vastly different curvature: some areas might be extremely steep "valleys," while others are wide, flat "plateaus."

This creates a dilemma for choosing $\gamma$:

* **If $\gamma$ is too small:** In flat regions of the loss landscape, the gradients will be very small. A small learning rate will result in minuscule updates, and the algorithm will take an extremely long time to converge, if it converges at all.
* **If $\gamma$ is too large:** In very steep regions, a large learning rate can cause the algorithm to overshoot the minimum. The update step may be so large that it jumps completely across the valley to a point where the loss is even higher. This can lead to oscillations where the parameters bounce back and forth, failing to converge, and may even cause the algorithm to diverge entirely.

The ideal learning rate would be adaptive: large in flat regions to speed up progress and small in steep regions to ensure careful convergence. This challenge has motivated the development of more advanced optimization algorithms beyond simple gradient descent.

</div>

#### The Impact of System Dynamics on the Loss Landscape

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Dynamics Shape the Loss Landscape</span></p>

The dynamics of the model being trained have direct implications for the structure of its loss function. While the loss function is defined over the parameter space, not the state space of the model, the behavior of the model's dynamics shapes the landscape.

Consider training a Recurrent Neural Network.

* If the RNN is operating in a chaotic regime, its output can be extremely sensitive to small changes in its parameters.
* This sensitivity translates to the loss function. The resulting loss landscape can be incredibly complex and may even be fractal.
* Trying to perform gradient descent on such a landscape is exceptionally difficult, as the gradient can change dramatically and unpredictably with tiny steps.

In practice, the loss landscape for large systems (with potentially hundreds, thousands, or even billions of parameters) is extremely high-dimensional. While we cannot visualize it directly, we can analyze its properties by plotting cross-sections in subspaces or by observing the behavior of the training process itself.

</div>

### Classical Remedies and Modern Approaches

Over the years, researchers and practitioners have developed numerous techniques to mitigate the challenges of gradient-based optimization. Here, we survey a few key ideas.

#### Addressing Local Minima

##### Multiple Initial Conditions

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Multiple Initial Conditions)</span></p>

A straightforward, classical approach to increase the chances of finding a good minimum is to run the entire optimization process multiple times from different, randomly chosen initial parameter values ($\theta_0$).

If the loss landscape contains many local minima, starting from different points explores different regions of the space. After all the runs are complete, one simply chooses the model that achieved the lowest final loss value. While simple, this can be computationally expensive. This technique is not as commonly used today in its basic form for large-scale deep learning, but the principle of exploration remains important.

</div>

##### Overparameterization: Double Descent and the Lottery Ticket Hypothesis

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Double Descent and the Lottery Ticket Hypothesis</span></p>

A more modern and perhaps counter-intuitive approach involves using strongly overparameterized models -- that is, using many more parameters than one might think are necessary to represent the data.

This is related to a phenomenon known as **double descent**. Classical statistical theory suggests that as you increase model complexity (number of parameters), the test loss (error on unseen data) will first decrease (good) and then increase as the model begins to overfit the training data (bad). This creates a U-shaped curve.

However, a surprising observation, highlighted in a notable 2019 paper by Belkin, Hsu, Ma, and Mandal (related to the Franklin and Carbone paper mentioned in the lecture), is that if you continue to increase the number of parameters far beyond the point of overfitting, the test loss can decrease again. This second drop is the "double descent."

This leads to the **Lottery Ticket Hypothesis**. The idea is that a very large, overparameterized network is like a lottery containing many tickets. Within this massive network, there exists a smaller, optimal sub-network (the "winning ticket") that is perfectly suited for the given task. The gradient descent training process, in this view, doesn't just tune all the parameters, but effectively carves out this winning sub-network from the larger structure by pruning unnecessary connections.

Note: This is an active area of research, and while powerful, this approach does not always work.

</div>

##### Stochasticity: Adding Noise to Gradients

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Adding Noise to Gradients</span></p>

Another strategy is to intentionally introduce randomness into the optimization process. By adding a small amount of noise, $\epsilon$, to the gradient calculation at each step, the parameter updates become probabilistic:

$$\theta_{n+1} = \theta_n - \gamma (\nabla L(\theta_n) + \epsilon)$$

where $\epsilon$ is drawn from some probability distribution.

The purpose of this noise is to provide a chance for the parameters to "jump out" of a local minimum. If the algorithm is stuck in a shallow valley, a random nudge from the noise term might be enough to push it over the hill and into a deeper, better region of the loss landscape.

This principle is exploited by entire classes of models, such as Boltzmann Machines, which use thermal noise in a principled way to probabilistically find the global optimum of the system. This topic, however, goes beyond the scope of our current discussion.

</div>

### Advanced Optimization for Neural Network Training

#### Mitigating Local Minima: Stochastic Gradient Descent (SGD)

A primary challenge in gradient-based optimization is the risk of the algorithm converging to a local minimum in the loss function rather than the desired global minimum. One of the most common and effective procedures to address this is not to inject artificial noise into the gradient updates, but rather to leverage the noise inherent in the data itself.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stochastic Gradient Descent: SGD)</span></p>

Stochastic Gradient Descent (SGD) is an optimization algorithm where, at each gradient step, the update is calculated based on a randomly drawn subsample (a "mini-batch") of the full training dataset, rather than the entire dataset.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition for SGD)</span></p>

The core idea is that data is inherently noisy. By randomly drawing a different subset of data for each step, we introduce noise into the gradient calculation. This stochasticity can help the optimization process "jump out" of shallow local minima and continue its search for a better solution in the broader parameter space. The effect is conceptually similar to injecting noise directly into the gradient updates.

</div>

##### A Note on Time Series Data

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Handling Autocorrelations in Temporal Data)</span></p>

When applying SGD or related subsampling techniques to time series and dynamical systems, special care must be taken. These data types are characterized by significant autocorrelations, where the value of a point depends on previous points.

* **Problem:** Randomly sampling individual data points from a time series will destroy its temporal structure and, consequently, the very dynamics the model is intended to learn.
* **Solution:** To preserve the temporal integrity, one must sample consecutive blocks or segments of the time series for each gradient update. This ensures that the essential dynamic relationships within the data are maintained during training.

</div>

#### Addressing Varying Slopes: Adaptive Learning Rates

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Adaptive Learning Rates)</span></p>

Another significant challenge in training deep networks is the presence of "ravines" or "valleys" in the loss landscape, where the slope is very steep in one direction and very shallow in another. A fixed learning rate can cause oscillations across the steep direction while making painfully slow progress along the shallow one.

The naive but effective approach to this problem is to make the learning rate adaptive. Instead of a fixed scalar $\gamma$, we can use a learning rate $\gamma_n$ that changes at each step $n$. This adaptation can be based on the history of the gradients, their variance, or other principles designed to accelerate progress in flat directions and dampen updates in steep directions.

</div>

##### Common Adaptive Rate Algorithms

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A Toolbox of Optimizers)</span></p>

Modern machine learning frameworks provide a host of optimizers that implement adaptive learning rate schemes. While a deep dive into each is beyond our current scope, it is essential to be aware of the most prominent examples:

* **Adagrad:** Adapts the learning rate based on the historical sum of squared gradients for each parameter.
* **Momentum:** Aims to accelerate descent by adding a fraction of the previous update vector to the current one, helping to build "velocity" in a consistent direction. It provides an implicit estimate of the slope.
* **Adam (Adaptive Moment Estimation):** A highly popular algorithm that combines the ideas of momentum and adaptive scaling of gradients (similar to RMSprop).
* **RAdam (Rectified Adam):** An enhancement to Adam that seeks to correct for the high variance of adaptive learning rates in the early stages of training.

In contemporary practice, a significant portion of the field has settled on using Adam or RAdam as default, robust optimizers for a wide range of problems. All these techniques function by adjusting learning rates in an intelligent manner during the gradient descent procedure.

</div>

#### Incorporating Curvature: Second-Order Algorithms

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Second-Order Methods Intuition)</span></p>

While first-order methods like gradient descent only use the gradient (first derivative) of the loss function, second-order algorithms incorporate additional information about the curvature of the loss surface.

The virtue of second-order methods is that they are often superior to standard gradient descent because they possess a more detailed "map" of the loss landscape. By considering how the gradient itself is changing (i.e., the second derivative), they can make more informed steps. The idea is that if the first derivative is small, the change in the derivative is also likely to be small. These methods weigh the gradient update by the magnitude of the second derivatives.

</div>

##### The Hessian and the Newton-Raphson Procedure

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Hessian Matrix)</span></p>

The Hessian is the matrix of second-order partial derivatives of the loss function. It describes the local curvature of the function at a given point.

A naive second-order update rule modifies the parameters $\theta$ not just with the gradient of the loss $\nabla_{\theta} L$, but by pre-multiplying it with the inverse of the Hessian, $H^{-1}$:

$$\theta_{n+1} = \theta_n - \gamma [H(\theta_n)]^{-1} \nabla_{\theta} L(\theta_n)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Relation to Newton-Raphson)</span></p>

In its strict formulation, this update rule gives rise to the Newton-Raphson procedure, a well-known root-finding algorithm from statistics and numerical analysis.

</div>

##### Challenges and Adjustments

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Caveats of Second-Order Methods)</span></p>

Despite their theoretical advantages, pure second-order methods are rarely used in modern large-scale machine learning for two primary reasons:

1. **Computational Demand:** Calculating, storing, and inverting the Hessian matrix is computationally prohibitive. For a network with $N$ parameters, the Hessian is an $N \times N$ matrix, which quickly becomes intractable.
2. **Inclination and Saddle Points:** The naive update rule can get stuck in inclination points or saddle points where both the first and second derivatives vanish. Furthermore, it does not distinguish between local minima and local maxima, which is problematic as we only wish to find minima.

To make these methods viable, adjustments are necessary. A notable proposal by Pascanu and Bengio (c. 2014) involves modifying the Hessian to ensure updates always point towards a minimum.

* A Singular Value Decomposition (SVD) of the Hessian is performed.
* All singular values are set to be positive (conceptually, taking an absolute value, denoted here as $\|H\|$).
* This procedure ensures that the second derivatives cannot change sign at the same time as the first derivative vanishes, preventing convergence to maxima.

The adjusted update rule can be conceptualized as:

$$\theta_{n+1} = \theta_n - \gamma [|H(\theta_n)|]^{-1} \nabla_{\theta} L(\theta_n)$$

</div>

##### Quasi-Newton Methods and Their Relevance

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Quasi-Newton Methods)</span></p>

Quasi-Newton methods are a class of algorithms that seek to capture the benefits of second-order information without the prohibitive cost of computing the full Hessian. They do so by building an efficient numerical approximation of the inverse Hessian at each step.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Recursive Least Squares)</span></p>

Recursive Least Squares (RLS) is an algorithm formerly used for updating recurrent networks that falls into the family of quasi-Newton methods.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Enduring Value of Second-Order Thinking)</span></p>

While most large-scale applications have moved away from these methods, they should not be forgotten. For scientific applications with smaller datasets, the precision offered by incorporating curvature information can be extremely valuable. Furthermore, concepts in machine learning have a tendency to resurface, and a solid understanding of these powerful techniques remains a significant asset.

</div>

#### A Specialized Algorithm for RNNs: Backpropagation Through Time (BPTT)

We now turn to a very specific, time-efficient gradient descent algorithm tailored for Recurrent Neural Networks (RNNs): Backpropagation Through Time (BPTT).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Backpropagation Through Time: BPTT)</span></p>

BPTT is the standard algorithm for training RNNs. It is an adaptation of the general backpropagation algorithm that applies gradient descent to an RNN by first "unwrapping" or "unrolling" the network through its time steps.

</div>

##### Historical Context

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Context of BPTT)</span></p>

BPTT was introduced and refined by several researchers over the years, with key contributions from:

* Paul Werbos (1988)
* Ronald Williams and David Zipser
* David Rumelhart and others, with a famous paper in 1995.

</div>

##### The Core Idea: Unwrapping in Time

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(From Recurrence to Depth)</span></p>

The foundational insight of BPTT is to train a recurrent network in the exact same way as a standard feed-forward network. This is achieved by conceptually transforming the RNN's temporal recursion into a spatial deep structure.

Consider a simple RNN with two units, whose activations are $x_1$ and $x_2$, and a weight matrix $W$ that includes recurrent couplings like $W_{11}$, $W_{12}$, $W_{21}$, and $W_{22}$.

To train this network on a time series of length $T$, we perform the following "unwrapping" procedure:

1. **Create a Layer for Each Time Step:** The RNN is converted into a deep feed-forward network where each time step, from $t=1$ to $t=T$, becomes a distinct layer.
2. **Propagate Activations:** The state of the network at time $t$ becomes the input to the layer representing time $t+1$. An activation $x_1(t-1)$ propagates to influence $x_1(t)$, $x_2(t)$, and so on.
3. **Share Weights Across Layers:** This is the crucial feature that distinguishes an unwrapped RNN from a standard deep network. The same set of weights ($W_{11}$, $W_{12}$, $\dots$) is used at every layer (i.e., at every time step). The weights are effectively copied and pasted across the entire time-unrolled structure.

The following illustrates this transformation from a recurrent graph to a deep, feed-forward graph:

* **At Time $t=1$:** The network has units with activations $x_1(1)$ and $x_2(1)$.
* **At Time $t=2$:** This forms the next layer. The connection from $x_1(1)$ to $x_1(2)$ is governed by weight $W_{11}$. The connection from $x_2(1)$ to $x_1(2)$ is governed by $W_{12}$, and so on.
* ...and so on, until **Time $t=T$:** The final layer corresponds to the final time step, with activations $x_1(T)$ and $x_2(T)$.

This unwrapped structure is simply another way to write down the recursive update procedure of the RNN. Instead of updating a single network state recursively, we can think of it as propagating activity through a deep network where each layer corresponds to a moment in time.

</div>

##### The Backpropagation Procedure

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Backpropagation Procedure)</span></p>

BPTT is a specific, algorithmically efficient implementation of gradient descent.

1. **Forward Pass:** Propagate activity forward through the unrolled network, from $t=1$ to $t=T$.
2. **Calculate Errors:** At the output layer(s), calculate the error, which is the deviation between the network's prediction and the target value.
3. **Backward Pass:** Propagate these error signals backward through the network, from layer $T$ down to layer $1$. At each layer (time step), update the shared weight parameters based on the propagated error.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Algorithmic Efficiency)</span></p>

BPTT is a highly storage-efficient procedure. At each step of the backward pass, it only needs to account for the values present at that particular time step, as it leverages the already-computed values from the subsequent step. The complexity is linear in time, as it proceeds layer by layer.

</div>

##### Input and Output Configurations

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Input and Output Configurations)</span></p>

The specific structure of the unwrapped network depends on the task at hand.

* **External Inputs:** The network can receive external inputs at any or all time steps. For example, in sentence processing, each word could be an input at a sequential time step.
* **Target Outputs:** The network can be trained to produce a target output at any or all time steps.
  * **Sequence-to-Sequence (e.g., Time Series Modeling):** If we want an RNN to reproduce a temperature time series, we would have a target output (the desired temperature) at each time step.
  * **Sequence-to-Value (e.g., Classification):** If we want to perform sentiment classification on a sentence, we might provide word inputs at each time step but only have a single target output at the final time step ($t=T$), representing the overall sentiment.

</div>

##### Formalism for Training

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Simplifications for Derivation)</span></p>

For clarity in the following derivation, we will:

1. Neglect External Inputs: These do not fundamentally change the derivation of the gradient updates.
2. Consider a Single Data Pattern: The logic extends trivially to multiple patterns by summing or averaging the loss.

Let our RNN be given by the recursive form, and let our set of trainable parameters (weights and biases) be denoted by $\theta$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Loss Function for BPTT)</span></p>

A typical loss function $L(\theta)$ for an RNN trained on a sequence of length $T$ is the mean squared error, averaged over time:

$$L(\theta) = \frac{1}{T} \sum_{t=1}^{T} \sum_{k=1}^{N} (x_k(t) - x_k^*(t))^2$$

where $N$ is the number of units, $x_k(t)$ is the activation of unit $k$ at time $t$, and $x_k^*(t)$ is the desired or target output for that unit at that time.

</div>

### Training Recurrent Networks: Backpropagation Through Time (Detailed Derivation)

This section delves into the fundamental mechanics of training Recurrent Neural Networks (RNNs). The core challenge in training these models lies in how to properly assign credit -- or blame -- to parameters that are reused at every step of a temporal sequence. The algorithm for this is a special case of backpropagation known as Backpropagation Through Time (BPTT). We will derive the gradient calculations step-by-step and, in doing so, uncover a critical instability that plagued early research in this area: the exploding and vanishing gradient problem.

#### The Optimization Problem: Minimizing a Loss Function

To train any neural network, we must first define an objective. This objective is typically formulated as the minimization of a loss function, $L$, which measures the discrepancy between the network's predictions and the observed data. For a single observed time series, we can define a total loss over the entire sequence.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Total Loss)</span></p>

The total loss, $L$, for a given time series is the sum of the losses incurred at each individual time step. If we denote the loss at a specific time step $t$ as $l_t$, the total loss is given by:

$$L = \sum_t l_t$$

This decomposition is possible due to the linearity of gradients, which allows us to consider the contribution of each time step to the total parameter gradient independently.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Squared Error Loss)</span></p>

For concreteness, a common choice for the loss function is the squared error loss. Let $x_{\text{obs}}(t)$ be the observed value at time $t$ and $x(t)$ be the value predicted by our model. The loss at that time step, $l_t$, would be:

$$l_t = (x_{\text{obs}}(t) - x(t))^2$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Generality of the Loss)</span></p>

While we use the squared error for this example, the derivations that follow are general. You can substitute any differentiable loss function $l_t$ without changing the core mechanics of the backpropagation algorithm. The loss $l_t$ is a function of both the system parameters, which we'll call $\theta$, and the network's state or activation at that time, $x(t)$.

</div>

#### Gradient Calculation for Recurrent Architectures

Our goal is to adjust the model's parameters, $\theta_i$, to minimize the total loss $L$. We achieve this using gradient descent, which requires computing the derivative of the loss with respect to each parameter, $\frac{\partial L}{\partial \theta_i}$.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(The Total Loss Gradient)</span></p>

By the linearity of the gradient operator, the derivative of the total loss is the sum of the derivatives of the per-time-step losses:

$$\frac{\partial L}{\partial \theta_i} = \frac{\partial}{\partial \theta_i} \sum_t l_t = \sum_t \frac{\partial l_t}{\partial \theta_i}$$

Now, we must analyze the term $\frac{\partial l_t}{\partial \theta_i}$. In a standard feedforward network, a parameter only affects the loss at the output layer. In an RNN, however, the situation is more complex. The parameters (e.g., the weight matrix $W$) are reused at every time step. This means a parameter $\theta_i$ at an early time step $\tau$ influences the state $x_t$ at a later time step $t$.

Consequently, to calculate the gradient of the loss at time $t$, we must sum over the influence of the parameter $\theta_i$ as it appears at all preceding time steps $\tau$ from $1$ to $t$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Decomposing the Gradient with the Chain Rule)</span></p>

Applying the chain rule to the term $\frac{\partial l_t}{\partial \theta_i}$ reveals this dependency. The loss $l_t$ is an explicit function of the state $x_t$. The state $x_t$, in turn, is a function of all previous states, including $x_\tau$, where the parameter $\theta_i$ has an effect. This creates a recursive dependency that we must unroll.

The full expression for the gradient of the loss at time $t$ with respect to a parameter $\theta_i$ is a sum over all previous time steps $\tau \le t$ where that parameter appears:

$$\frac{\partial l_t}{\partial \theta_i} = \sum_{\tau=1}^{t} \frac{\partial l_t}{\partial x_t} \frac{\partial x_t}{\partial x_\tau} \frac{\partial x_\tau}{\partial \theta_i}$$

Let's break down the components of this expression:

1. $\frac{\partial l_t}{\partial x_t}$: This is the local gradient of the loss at time $t$ with respect to the network's output/state at that same time. It measures how the final error at step $t$ changes with respect to the output at step $t$.
2. $\frac{\partial x_t}{\partial x_\tau}$: This is the **temporal Jacobian matrix**. It measures how the state at a later time $t$ is influenced by the state at an earlier time $\tau$. This term is the crux of BPTT, as it carries the gradient information backward through the unrolled network.
3. $\frac{\partial x_\tau}{\partial \theta_i}$: This term measures the direct influence of the parameter $\theta_i$ on the state $x_\tau$ at the time step $\tau$ where the parameter is applied.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Jacobian Matrix Dimensions)</span></p>

To better understand the mathematical objects we are manipulating, consider an RNN with an $m$-dimensional state vector $x \in \mathbb{R}^m$. The dimensions of the terms in the chain rule are as follows:

* $\frac{\partial l_t}{\partial x_t}$: A $1 \times m$ row vector (the gradient of the scalar loss w.r.t. the state vector).
* $\frac{\partial x_t}{\partial x_\tau}$: An $m \times m$ matrix, representing the Jacobian of the state at time $t$ with respect to the state at time $\tau$. Each element $(j, k)$ of this matrix is $\frac{\partial x_j(t)}{\partial x_k(\tau)}$.
* $\frac{\partial x_\tau}{\partial \theta_i}$: If $\theta_i$ is a scalar parameter, this is an $m \times 1$ column vector.

</div>

#### The Recursive Chain Rule and Temporal Dependencies

The most important and complex term in our gradient expression is the temporal Jacobian, $\frac{\partial x_t}{\partial x_\tau}$. This term quantifies the long-range dependencies in the sequence. We can decompose it further by another application of the chain rule.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Unrolling the Temporal Jacobian)</span></p>

The state $x_t$ is a direct function of $x_{t-1}$, which is a function of $x_{t-2}$, and so on. We can express the derivative of $x_t$ with respect to a distant past state $x_\tau$ as a product of intermediate, single-step Jacobians:

$$\frac{\partial x_t}{\partial x_\tau} = \frac{\partial x_t}{\partial x_{t-1}} \frac{\partial x_{t-1}}{\partial x_{t-2}} \cdots \frac{\partial x_{\tau+1}}{\partial x_\tau}$$

This can be written more compactly using product notation:

$$\frac{\partial x_t}{\partial x_\tau} = \prod_{u=\tau+1}^{t} \frac{\partial x_u}{\partial x_{u-1}}$$

Each term in this product, $\frac{\partial x_u}{\partial x_{u-1}}$, is the Jacobian of the state transition function at a single time step.

</div>

#### The Exploding and Vanishing Gradient Problem

Let's now investigate the structure of the single-step Jacobian, $\frac{\partial x_u}{\partial x_{u-1}}$, to understand the long-term behavior of this product.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(A Simple RNN Update Rule)</span></p>

Consider a standard RNN where the state $x_t$ is updated according to the following rule:

$$x_t = \phi(W x_{t-1} + \dots)$$

Here, $W$ is the recurrent weight matrix and $\phi$ is a non-linear, element-wise activation function. To find the Jacobian $\frac{\partial x_t}{\partial x_{t-1}}$, we apply the chain rule (outer derivative times inner derivative):

* The derivative of the inner part, $W x_{t-1}$, with respect to $x_{t-1}$ is simply the matrix $W$.
* The derivative of the outer element-wise function $\phi$ results in a diagonal matrix containing the derivatives of $\phi$ evaluated at each input component. Let's denote this $\text{diag}(\phi'(\dots))$.

Therefore, the single-step Jacobian is:

$$\frac{\partial x_t}{\partial x_{t-1}} = \text{diag}(\phi'(W x_{t-1} + \dots)) \cdot W$$

Substituting this back into our product expression for the temporal Jacobian, we get:

$$\frac{\partial x_t}{\partial x_\tau} = \prod_{u=\tau+1}^{t} \left( \text{diag}(\phi'(W x_{u-1} + \dots)) \cdot W \right)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Exploding and Vanishing Gradient Problem)</span></p>

This product of matrices is the source of a fundamental instability in training RNNs. The expression involves repeatedly multiplying the weight matrix $W$, effectively raising it to the power of the time difference, $t-\tau$. The behavior of this matrix power is governed by the eigenvalues of the matrices in the product.

* **Exploding Gradients:** If the magnitudes of the leading eigenvalues of the Jacobian matrices are, on average, greater than $1$, their product will grow exponentially as the time gap $t-\tau$ increases. The gradients will "explode" to enormous values, leading to unstable training and divergent weight updates.
* **Vanishing Gradients:** If the magnitudes of the leading eigenvalues are, on average, less than $1$, their product will shrink exponentially towards zero as $t-\tau$ increases. The gradients will "vanish." This is also highly problematic, as it means the influence of early time steps on the loss at later time steps is effectively erased. The network becomes incapable of learning long-range dependencies, as the information required to update the parameters is lost during backpropagation.

This exploding and vanishing gradient problem is not merely a numerical inconvenience; it is a fundamental obstacle to learning long-term structure in sequential data with simple RNNs. The dynamics of this process are deeply connected to concepts from dynamical systems theory, such as the calculation of Lyapunov exponents and the stability analysis of linear systems converging to fixed points. The repeated matrix multiplication is precisely the process used to determine the stability of a linear dynamical system. This insight highlights why these "vanilla" training approaches are no longer standard practice and motivated the development of more sophisticated architectures.

</div>

### Long Short-Term Memory (LSTM) Networks

Long Short-Term Memory (LSTM) networks are a specialized type of recurrent neural network (RNN) architecture designed to handle long-term dependencies in sequential data. This section details the precise equations governing the LSTM cell and explores the core principles that make it effective.

#### The Complete LSTM Architecture

The power of an LSTM lies in its internal structure, which is composed of a memory cell and several gates that regulate the flow of information. These components work in concert to decide what information to store, what to discard, and what to output at each time step.

##### The Memory Cell Update

The most important component of the LSTM is the memory cell, which carries information through time. Its state, denoted by $c_t$, is updated at each time step $t$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Memory Cell Update)</span></p>

The state of the memory cell $c_t$ at time step $t$ is updated according to the following equation:

$$c_t = (f_t \odot c_{t-1}) + (i_t \odot \tanh(z_{t-1} + h_c))$$

Where:

* $c_{t-1}$ is the state of the memory cell from the previous time step.
* $f_t$ is the forget gate's activation vector.
* $i_t$ is the input gate's activation vector.
* $z_{t-1}$ is the total output from the previous time step.
* $h_c$ is a bias term for the candidate memory content.
* $\odot$ denotes the pointwise multiplication (Hadamard product) of vectors.
* $\tanh$ is the hyperbolic tangent activation function.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition for the Memory Cell Update)</span></p>

This update equation has two primary parts:

1. **Forgetting:** The term $f_t \odot c_{t-1}$ determines which parts of the old memory cell state $c_{t-1}$ should be preserved or discarded. The forget gate $f_t$ acts as a filter; if an element of $f_t$ is close to $0$, the corresponding information in $c_{t-1}$ is forgotten. If it is close to $1$, the information is kept.
2. **Inputting:** The term $i_t \odot \tanh(z_{t-1} + h_c)$ determines what new information should be added to the cell state. A candidate memory content is first computed (the $\tanh$ part), and the input gate $i_t$ decides which parts of this new information are relevant enough to be stored in $c_t$.

The combination of these two operations allows the LSTM to selectively update its memory, preserving crucial long-term information while incorporating new, relevant inputs.

</div>

##### The Gating Mechanisms

The flow of information into and out of the memory cell is controlled by three gates: the input gate ($i_t$), the forget gate ($f_t$), and the output gate ($o_t$). These gates are implemented using a sigmoid activation function, which outputs values between $0$ and $1$, representing the degree to which information is allowed to pass.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">Gate Equations</span></p>

The activation of each gate at time step $t$ is calculated as follows:

* **Forget Gate** ($f_t$):

$$f_t = \sigma(W_f z_{t-1} + h_f)$$

* **Input Gate** ($i_t$):

$$i_t = \sigma(W_i z_{t-1} + h_i)$$

* **Output Gate** ($o_t$):

$$o_t = \sigma(W_o c_{t-1} + h_o)$$

Where:

* $\sigma$ is the sigmoid function, defined as:

$$\sigma(y) = \frac{1}{1 + e^{-y}}$$

* $W_f$, $W_i$, and $W_o$ are weight matrices for the respective gates.
* $h_f$, $h_i$, and $h_o$ are the bias vectors for the respective gates.
* $z_{t-1}$ is the output from the previous time step.
* $c_{t-1}$ is the memory cell state from the previous time step.

</div>

##### The Final Output

The final output of the LSTM cell at time step $t$, denoted as $z_t$, is a filtered version of the memory cell state $c_t$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">Final Output</span></p>

The total output $z_t$ is computed by passing the memory cell state through a $\tanh$ function and then multiplying it pointwise by the output gate's activation:

$$z_t = o_t \odot \tanh(c_t)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Output Gate Intuition</span></p>

This mechanism allows the network to control what part of its internal memory is exposed to the next layer or the next time step. The $\tanh$ function squashes the values of the memory cell to be between $-1$ and $1$, and the output gate $o_t$ then decides which of these values are relevant to pass on as the final output.

</div>

#### Fundamental Design Principles of LSTMs

The specific architectural choices in the LSTM are not arbitrary; they embody two crucial principles for processing sequential data: linearity and gating.

##### The Power of Linearity

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">The Power of Linearity</span></p>

A key feature of the LSTM is the linear nature of the memory update through the forget gate. The operation $f_t \odot c_{t-1}$ is a linear interaction. This is critically important for preserving information over long time horizons.

* **Information Preservation:** If the forget gate $f_t$ is set to $1$, the previous memory content $c_{t-1}$ is passed through to the next step unmodified. This allows the network to "literally copy and paste the previous content" and "rescue it across long periods of time."
* **Control:** This introduces a form of control that is difficult to achieve in purely nonlinear systems. By managing the forget gate, the network can learn to maintain a stable memory state when needed. As stated in the lecture, "Linearity is important. Linearity allows a certain type of control that you don't that easily have in nonlinear systems."

</div>

##### The Concept of Gating

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">The Concept of Gating</span></p>

The second foundational concept is gating, where information flow is modulated by multiplicative units (the gates). This idea of using multiplicative interactions to control pathways in a neural network is powerful and has proven influential. This principle is not unique to LSTMs and can be found in other modern architectures, such as Mamba.

</div>

#### Variants and Simplifications: GRUs

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">Gated Recurrent Units (GRUs)</span></p>

The LSTM architecture, while powerful, is also complex. This has led to the development of several simplified variants. One of the most prominent is the Gated Recurrent Unit (GRU).

* **Origin:** The GRU was introduced in a 2014 formulation by Cho, et al., in collaboration with Yoshua Bengio.
* **Purpose:** GRUs aim to capture the essence of gated RNNs with a simpler architecture, often combining the forget and input gates into a single "update gate."
* **Availability:** You will find GRUs, along with many other LSTM variants, implemented in virtually any standard machine learning toolbox.

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

* If the distributions are identical ($P_{true}(x) = P_{gen}(x)$ everywhere), the fraction inside the logarithm is $1$, making $\log(1) = 0$, and thus $D_{KL} = 0$.
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

## Lecture 11

This lecture provides a comprehensive study of how Recurrent Neural Networks (RNNs) can be used as surrogate models for dynamical systems. We cover the goals and evaluation of dynamical systems reconstruction, specialized training methodologies for chaotic systems, the analytically tractable Piecewise Linear RNN (PLRNN), bifurcation phenomena during training, flow operator properties, Reservoir Computing, Autoencoders, and the integration of SINDy for latent dynamics discovery.

### Dynamical Systems Reconstruction

The primary objective in reconstructing dynamical systems using RNNs is to identify an approximate flow map, $\phi^*$, that effectively models the underlying dynamics of a system.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dynamical System Reconstruction)</span></p>

A **dynamical system reconstruction** aims to find an approximate flow map $\phi^*$, typically modeled through an RNN, that is **topologically conjugate** to the underlying dynamical system described by a flow operator $\phi$ on a domain $D$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Beyond Prediction)</span></p>

In a scientific context, we do not view these RNNs merely as black-box prediction models. Instead, we treat them as **surrogate models**. The goal is to capture the dynamical properties of the system to gain insight into the underlying mechanisms governing the observed data.

</div>

#### Assessing Reconstruction Quality

When evaluating how well an RNN has reconstructed a dynamical system, we look beyond simple error metrics and focus on geometrical, temporal, and dynamical properties.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Key Performance Measures)</span></p>

* **Geometrical Properties:** We assess the overlap in state space. One common measure is the Kullback-Leibler (KL) divergence applied to the distribution of states in the state space.
* **Temporal Properties:** To quantify how well the long-term temporal properties are matched, we use the Hellinger distance defined on the power spectra of the true and generated signals.
* **Dynamical Properties:** We calculate and compare the Lyapunov exponents (specifically the maximal Lyapunov exponent, $\lambda_{\max}$) of both the original system and the reconstructed model.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Attractor Localization)</span></p>

In an ideal reconstruction, the RNN can detect features of the underlying system that were not explicitly present in the training trajectories. For instance, a model trained only on trajectories residing on an attractor might still accurately localize the system’s equilibria (fixed points).

</div>

### Training Methodologies for Chaotic Systems

Training RNNs on chaotic systems presents significant challenges, most notably the exploding and vanishing gradient problem, which is often inevitable when dealing with underlying chaotic dynamics. To mitigate these issues, specialized training techniques are employed. These techniques build on the teacher forcing and generalized teacher forcing methods discussed in Lecture 10.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Teacher Forcing)</span></p>

**Teacher Forcing** is a training technique where the latent state $z_t$ of the RNN is replaced by an estimate $\hat{z}_t$ derived from the actual data.

* The estimate $\hat{z}_t$ is obtained by inverting (or pseudo-inverting) the decoder/observation function $G$.
* This replacement occurs every $\tau$ time steps.
* The interval $\tau$ is chosen based on the Lyapunov spectrum or the maximal Lyapunov exponent of the system.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Teacher Forcing)</span></p>

**Generalized Teacher Forcing** is a refinement where, instead of a total replacement, a weighted average is used to update the state:

$$z_t^{\text{updated}} = \alpha \, z_t^{\text{forward}} + (1 - \alpha)\,\hat{z}_t^{\text{data}}$$

where:
* $z_t^{\text{forward}}$ is the state predicted by the RNN.
* $\hat{z}_t^{\text{data}}$ is the estimate inferred from the data.
* $\alpha$ is a weighting factor.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimization of $\alpha$)</span></p>

The parameter $\alpha$ can be adjusted optimally by considering the Singular Value Decomposition (SVD) of the underlying Jacobian matrix of the system. Recall from Lecture 10 that the optimal choice is $\alpha_t = 1 - \frac{1}{\sigma_{\max}(G_t)}$, which keeps gradient magnitudes controlled.

</div>

### Piecewise Linear Recurrent Neural Networks (PLRNN)

To ensure mathematical tractability and interpretability, we often utilize Piecewise Linear Recurrent Neural Networks (PLRNN).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(PLRNN State Equations)</span></p>

The latent variables $z_t$ in a PLRNN evolve according to the following multivariate map:

$$z_t = A z_{t-1} + W \phi(z_{t-1}) + h$$

where:
* $A$ is a weight matrix (often diagonal).
* $W$ is the weight matrix for the non-linear term.
* $h$ is a bias term.
* $\phi$ is the Rectified Linear Unit (ReLU) activation function, defined as $\phi(z) = \max(0, z)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Observation Function)</span></p>

The latent states are linked to the actual observations $x_t$ through an observation function $G$:

$$x_t = G(z_t;\, \lambda)$$

where $\lambda$ represents trainable parameters.

</div>

#### Mathematical Analysis of Trained Models

The piecewise linear nature of the PLRNN allows us to reformulate the system into a more analytically accessible form.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Matrix Representation of ReLU)</span></p>

The $\max$ operator in the PLRNN can be rewritten as a time-dependent diagonal matrix $D_{t-1}$, allowing the system to be expressed as an affine mapping:

$$z_t = (A + W D_{t-1})\, z_{t-1} + h$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Construction of the Indicator Matrix)</span></p>

1. Let $\phi(z) = \max(0, z)$ be the ReLU activation function.
2. Define a diagonal matrix $D_t$ such that the $i$-th element on the diagonal corresponds to the $i$-th component of the state vector $z_t$.
3. Set the diagonal entries as:

$$D_{ii} = \begin{cases} 1 & \text{if } z_i > 0 \\ 0 & \text{if } z_i \leq 0 \end{cases}$$

4. Substituting this into the state equation: $W\phi(z_{t-1}) = W D_{t-1} z_{t-1}$.
5. The full state equation becomes:

$$z_t = A z_{t-1} + W D_{t-1} z_{t-1} + h = (A + W D_{t-1})\, z_{t-1} + h$$

This confirms that for any given state $z$, the system behaves as a linear mapping specific to the "quadrant" or region of state space defined by the signs of the components of $z$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fixed Points of the Map)</span></p>

A **fixed point** $z^*$ of the map is a state that remains constant under the iteration of the map, such that $z_t = z_{t-1} = z^*$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Analytical Extraction of Fixed Points)</span></p>

To find the fixed point $z^*$, we assume the system has settled into a specific linear region defined by $D^*$:

1. Start with the steady-state equation: $z^* = A z^* + W D^* z^* + h$.
2. Group the terms involving $z^*$: $z^* - A z^* - W D^* z^* = h$.
3. Factor out $z^*$: $(I - A - W D^*)\, z^* = h$.
4. Solve for $z^*$:

$$z^* = (I - A - W D^*)^{-1} h$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Consistency Constraint)</span></p>

While the above formula provides a candidate for a fixed point, it is not purely analytical. One must verify that the resulting $z^*$ is **consistent** with the matrix $D^*$. That is, the signs of the components of the calculated $z^*$ must actually produce the diagonal entries of $D^*$ used in the calculation.

</div>

### Fixed Points and Periodic Orbits in RNNs

In the study of RNNs as dynamical systems, identifying the long-term behavior of the system—specifically its fixed points and periodic orbits (cycles)—is essential for understanding the model’s computational properties and its validity as a surrogate for real-world systems.

#### The Consistency Problem in Fixed Point Localization

To find an exact solution for a fixed point $Z^*$, we must ensure that the state of the system is consistent with the activation of its units. In many RNN architectures, the transition is governed by a diagonal matrix $D$ that represents the "on/off" state of the neurons (often associated with rectified linear units or similar activations).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Consistency Condition)</span></p>

A candidate fixed point $Z^*$ and its associated configuration matrix $D^*$ are considered **consistent** if and only if:

$$D_{ii}^* = 1 \iff Z_i^* > 0$$

and $D_{ii}^* = 0$ otherwise.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Combinatorial Complexity)</span></p>

Finding a fixed point is fundamentally a combinatorial problem. Because each unit in a hidden layer of dimension $m$ can be either active or inactive, there are $2^m$ possible configurations for the matrix $D^*$. In low-dimensional spaces, one could exhaustively check every configuration, but this becomes computationally intractable as $m$ increases.

</div>

#### Mathematical Formulation of $k$-Cycles

Beyond individual fixed points, we are interested in cycles—sets of points that the system visits in a repeating sequence.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($k$-Cycle)</span></p>

A **$k$-cycle** is a set of $k$ distinct points $\lbrace Z_1^*, Z_2^*, \dots, Z_k^* \rbrace$ such that each point is a fixed point of the $k$-times iterated map $f^{(k)}$. That is:

$$Z_{m}^* = f^{(k)}(Z_{m}^*) \quad \text{for } m \in \lbrace 1, \dots, k \rbrace$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Iterated Map of an RNN)</span></p>

For a system defined by an affine transition of the form $Z_t = (A + W D_{t-1})\, Z_{t-1} + h$, the two-time iterated map is expressed as:

$$Z_t = (A + W D_{t-1}) \left[ (A + W D_{t-2})\, Z_{t-2} + h \right] + h$$

Expanding this, we obtain:

$$Z_t = (A + W D_{t-1})(A + W D_{t-2})\, Z_{t-2} + (A + W D_{t-1})\, h + h$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(General $k$-Iteration)</span></p>

To derive the general form for a $k$-times iterated map, we apply the recursive rule repeatedly:

1. Let the Jacobian of the map be defined as $J_t = (A + W D_t)$.
2. For a $k$-step iteration from $Z_{t-k}$ to $Z_t$, the state is:

$$Z_t = \left( \prod_{j=1}^{k} J_{t-j} \right) Z_{t-k} + \sum_{i=1}^{k-1} \left( \prod_{j=1}^{i} J_{t-j} \right) h + h$$

3. This resulting expression remains an affine map of the initial state $Z_{t-k}$. Thus, finding a $k$-cycle reduces to solving a linear system, provided the sequence of matrices $\lbrace D_{t-1}, \dots, D_{t-k} \rbrace$ is known and consistent with the resulting states.

</div>

#### Search Algorithms for Fixed Points and Cycles

Given the $2^m$ complexity of the combinatorial search, efficient numerical procedures are required to locate these points exactly rather than relying on approximations.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(The "Virtual Point" Heuristic)</span></p>

A highly efficient heuristic for finding fixed points and cycles involves the following steps:

1. **Initialization:** Start with an initial configuration matrix $D$.
2. **Candidate Computation:** Solve the affine equation to find a candidate solution $Z^*$ (a "virtual" fixed point).
3. **Consistency Check:** Verify if the signs of $Z^*$ match the configuration $D$.
4. **Update:** If inconsistent, use the configuration $D$ derived from the current $Z^*$ to initialize the next round.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Efficiency of the Virtual Point Heuristic)</span></p>

While the theoretical worst-case remains combinatorial, this heuristic often behaves linearly in time relative to the dimension. For certain matrix conditions, it can even be proven to converge in at most linear time. This efficiency is crucial for making RNNs "tractable" or "interpretable," allowing researchers to use them as surrogate systems to analyze underlying real-world data.

</div>

### Training Dynamics and Bifurcation Analysis

The process of training a neural network is itself a dynamical system. When we update parameters using gradient descent, we are moving through a parameter space that can fundamentally change the qualitative behavior of the network’s internal dynamics.

#### Optimization as a Dynamical System

Consider a standard gradient descent update rule:

$$\theta_{\text{next}} = \theta_{\text{prev}} - \eta \nabla L(\theta_{\text{prev}})$$

where $\theta$ represents the parameters (weights $W$, biases $h$) and $L$ is the loss function.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dynamical Phenomena in Training)</span></p>

Because the training process is a recursive update, it is subject to all dynamical phenomena:

* **Oscillations:** The parameters may bounce around an optimum.
* **Chaos:** The training path may become unpredictable.
* **Attractors:** The system may converge to stable fixed points.
* **Information Loss:** If the system converges too strongly to a fixed point, it may lose the gradient information required to learn the underlying system.

</div>

#### Case Study: The Single-Unit RNN

To understand how parameter changes affect system behavior, we examine a one-unit RNN with a sigmoid activation function $\phi$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(1-Unit Scalar System)</span></p>

Let $W$ and $Z$ be scalars. The system is defined by:

$$Z_t = \phi(W Z_{t-1} + h)$$

As we vary the parameters $W$ (weight) and $h$ (bias), the system undergoes various bifurcations:

* **Varying $h$:** Changing the bias shifts the sigmoid function along the $Z_{t-1}$ axis.
* **Varying $W$:** Changing the weight alters the slope (steepness) of the function.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Pitchfork Bifurcation in RNNs)</span></p>

If the sigmoid function $\phi$ is perfectly symmetric around its inflection point, increasing the weight $W$ can lead to a **pitchfork bifurcation**.

* Initially, the system has a single stable fixed point.
* As $W$ increases and the slope becomes steeper, two new stable fixed points simultaneously appear while the original fixed point becomes unstable.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bifurcation Visualization)</span></p>

This is visualized in a bifurcation graph where the stable states are plotted as a function of $W$. Understanding these transitions is vital because moving through a bifurcation during training can radically change the loss landscape and the network’s ability to represent the target system.

</div>

### Bifurcations in Neural Dynamics

In high-dimensional recurrent networks, the system’s behavior is governed by its fixed points and their stability. As we adjust parameters during training, the system may undergo qualitative changes in its topological structure, known as **bifurcations**. This connects to the general bifurcation theory introduced in Part I.

#### The Saddle-Node Bifurcation in Sigmoidal Units

Consider a single sigmoidal unit within a network where we adjust a parameter $H$. As $H$ varies, the intersection between the state update function and the bisectrix changes.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Saddle-Node Bifurcation Mechanism)</span></p>

As we shift the curve defined by $H$, it will eventually touch the bisectrix. This contact point gives rise to a **saddle-node bifurcation**. Before this point, we might have two stable fixed points (one at a lower value, one at an upper value). As we move through the bifurcation point, these fixed points can merge and disappear, or a single stable point may remain.

</div>

#### Bifurcation Graphs and Parameter Sensitivity

We can visualize these transitions by plotting the fixed point $z^*$ against the parameter $H$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bifurcation Graph)</span></p>

A **bifurcation graph** represents the location and stability of fixed points $z^*$ as a function of a system parameter $H$. For a sigmoidal system undergoing a saddle-node bifurcation, the graph typically displays a characteristic "arc" or "fold" where stable and unstable branches meet.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Learning Barrier — Kenji Doya, 1998)</span></p>

In a 1998 paper, Kenji Doya illustrated why certain configurations are unlearnable. If a desired state for a network is located at an unstable node surrounded by a cycle, gradient descent will fail to stabilize the system at that point. Because the target is unstable, the system will naturally drift away, preventing the network from ever reaching the desired output.

</div>

#### Impact on Training: The Loss Landscape

Training an RNN via gradient descent involves iteratively readjusting parameters like $H$ to minimize a loss function. However, bifurcations create significant obstacles for this process.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The "Jump" Phenomenon)</span></p>

Imagine the system is in a regime with two stable fixed points (lower and upper arcs). If the target $z$ requires increasing $H$, the state will move along the current arc. Upon reaching the bifurcation point, the current stable equilibrium disappears, forcing the state to "jump" abruptly to the other arc. This transition causes a steep, discontinuous jump in the loss function.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Gradient Behavior at Bifurcations)</span></p>

For certain types of bifurcations and dynamical systems, it can be formally proven that at the bifurcation point, the gradients will either:

1. **Diverge/Explode:** Tend toward infinity.
2. **Abruptly vanish:** Instantly go to zero.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Empirical Loss Landscapes)</span></p>

Research (e.g., Eisman et al.) using algorithms like PyDSTool to locate bifurcation curves has shown that huge jumps in the loss landscape coincide exactly with these curves. When plotting parameter trajectories during training, the loss spikes precisely as the trajectory crosses a bifurcation boundary.

</div>

#### Avoiding Bifurcations with Generalized Teacher Forcing

Standard backpropagation through time (BPTT) is highly susceptible to the instabilities caused by bifurcations. However, specific algorithms can mitigate this.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Smooth Loss via Alignment)</span></p>

Generalized Teacher Forcing (GTF), as defined earlier, is an algorithm that can formally be shown to avoid certain bifurcations. By aligning the system with the observed data at each time point, GTF "smooths out" the loss function. It effectively pushes the system into the correct dynamical regime without requiring it to cross the discontinuous "cliffs" in the loss landscape found in straightforward BPTT.

</div>

### Flow Operators and Continuous-Time RNNs

While RNNs are often defined in discrete time, they are frequently used to approximate systems that exist in continuous time. To do this accurately, the network must behave like a true flow operator.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Flow Operator)</span></p>

A **flow operator** maps an initial state $x_0$ to a future state $x_t$ after a duration $t$. For a true flow operator, we expect:

1. The ability to continuously vary $\Delta t$ and obtain valid outputs.
2. Adherence to the semi-group property.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Semi-Group Property)</span></p>

If we advance a system by time steps $s$ and $t$ in succession, the resulting state must be the same as advancing the system by a single step of $(s + t)$:

$$\Phi_{s+t} = \Phi_s \circ \Phi_t = \Phi_t \circ \Phi_s$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Path Independence)</span></p>

In a training context, if we have data $x_0$ and we want to reach a state at a future time, the result must be identical regardless of the path taken:

* **Path A:** Move from $x_0$ to $x_1$ (time $\tau_1$) then to $x_2$ (time $\tau_2$).
* **Path B:** Move directly from $x_0$ to $x_2$ (time $\tau_1 + \tau_2$).
* **Path C:** Move by $\tau_2$ first, then $\tau_1$.

All these paths must yield the same result for the system to be a mathematically consistent flow.

</div>

#### Recursive Descriptions and Neutral Elements

To enforce these properties, we can define the RNN using a recursive structure, as suggested by Chen and Wu (2003).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Recursive Flow Approximation)</span></p>

A system can be defined by the following recursive description:

$$z_t = z_{t - \Delta\tau} + \Delta\tau \cdot \sigma(z_{t - \Delta\tau},\, \Delta\tau)$$

where $\sigma$ is an activation function (which could itself be a deep feed-forward neural network) and $\Delta\tau$ is the time step.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Convergence to the Neutral Element)</span></p>

We demand that if the time step $\Delta t$ is zero, the state remains unchanged (the neutral element). Using the recursive definition:

$$z_t = \lim_{\Delta\tau \to 0} \left[ z_{t - \Delta\tau} + \Delta\tau \cdot \sigma(\dots) \right]$$

As $\Delta\tau \to 0$:

$$z_t = z_t + 0 \cdot \sigma(\dots) = z_t$$

This demonstrates that the recursive formulation automatically satisfies the neutral element property of a flow operator.

</div>

#### Enforcing Flow Properties through Loss Function Design

Standard RNN training does not guarantee the semi-group properties. To ensure the learned model behaves like a physical flow, we must use regularization terms in the loss function.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Flow Operator Composition Law)</span></p>

A true flow operator $f$ acting on a state $z_t$ with time steps $\tau_1$ and $\tau_2$ must satisfy the following composition law:

$$f(z_t,\, \tau_1 + \tau_2) = f(f(z_t,\, \tau_1),\, \tau_2)$$

This implies that advancing the system by $\tau_1$ and then by $\tau_2$ must be equivalent to advancing it by the combined step $\tau_1 + \tau_2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Regularization as a Constraint Mechanism)</span></p>

While we can attempt to design architectures that inherently respect these properties, a more flexible approach is to enforce them through the loss function. By adding a regularization term to the standard objective, we penalize the model when it deviates from the requirements of a flow operator. This principle can be extended to other physical constraints, such as Hamiltonian conservation laws.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(The Regularized Loss Function)</span></p>

The complete training loss $\mathcal{L}$ is constructed as the sum of a standard Mean Squared Error (MSE) and a regularization term $\lambda$ that enforces the flow properties:

$$\mathcal{L} = \text{MSE} + \lambda \sum \text{Deviations}$$

where the deviations are defined as:

1. **Composition Error:** The difference between a single step of $(\tau_1 + \tau_2)$ and the sequential application of $\tau_1$ and $\tau_2$:

$$\lVert f(z_t,\, \tau_1 + \tau_2) - f(f(z_t,\, \tau_1),\, \tau_2) \rVert^2$$

2. **Commutativity Error:** The difference resulting from swapping the order of $\tau_1$ and $\tau_2$:

$$\lVert f(f(z_t,\, \tau_1),\, \tau_2) - f(f(z_t,\, \tau_2),\, \tau_1) \rVert^2$$

</div>

### Reservoir Computing (Echo State Machines)

Training RNNs is notoriously difficult and computationally expensive. Techniques like generalized teacher forcing offer improvements, but **Reservoir Computing** (also known as Echo State Machines) approaches the problem from an entirely different perspective.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reservoir Computing)</span></p>

First introduced by Jaeger and Haas (2004) in *Science*, **Reservoir Computing** is a type of RNN where the internal connectivity is fixed and only the output layer is trained. It aims to maintain the simplicity of linear regression while retaining the ability to approximate complex, nonlinear dynamical systems.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Core Idea)</span></p>

The core concept is to project an input into a high-dimensional "pool" or reservoir of complex dynamics. Instead of meticulously training every connection in the network, we use a large, fixed reservoir that expresses a wide variety of dynamical behaviors. We then "shape" or "read out" these dynamics through a simple linear layer to match our observations.

</div>

#### Architecture and Dynamics of Reservoir Systems

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reservoir System Variables)</span></p>

* $x_t \in \mathbb{R}^N$: The true observed state (e.g., a time series from a Lorenz system or temperature data).
* $\hat{x}_t \in \mathbb{R}^N$: The network’s predicted state.
* $z_t \in \mathbb{R}^M$: The reservoir state (or latent state).
* **Dimensionality Constraint:** Crucially, $M \gg N$. The reservoir must be high-dimensional to provide a sufficiently rich "pool" of possibilities.

</div>

The reservoir state $z_t$ evolves according to a nonlinear function, typically utilizing a sigmoid activation:

$$z_t = \sigma(W z_{t-1} + h + W_{\text{in}}\, s_t)$$

where:
* $W$: The internal reservoir connectivity matrix. This is **fixed** and not changed during training.
* $W_{\text{in}}$: Input weights, also fixed.
* $s_t$: External inputs or forced true states $x_t$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reservoir Connectivity Properties)</span></p>

To prevent the reservoir from exhibiting "boring" dynamics (such as immediately collapsing to a fixed point), the matrix $W$ is carefully initialized:

* **Sparse Connectivity:** Connections are often sparse.
* **Spectral Norm:** The eigenvalue spectrum is typically scaled so the spectral norm is close to 1. This ensures the reservoir is at the "edge of chaos"—neither exploding nor decaying too rapidly.

</div>

The prediction $\hat{x}_t$ is generated via a linear mapping from the reservoir state:

$$\hat{x}_t = B z_t$$

In some cases, a basis expansion of $z_t$ is performed to improve performance. For example, concatenating $z_t$ with its squared terms: $\hat{x}_t = B [z_t;\, z_t^2]$. Importantly, the system remains linear in the parameters $B$.

#### Training Reservoir Computers via Linear Regression

Because the internal weights $W$ are fixed, the training process does not require backpropagation through time. Instead, it boils down to a simple regression problem.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Training vs. Inference)</span></p>

* **Training (Entrainment):** The reservoir is forced with the true states $x_t$. We record the resulting reservoir states $z_t$.
* **Test Time (Inference):** The true input $x_t$ is replaced by the network’s own previous prediction $\hat{x}_t$, letting the system run recursively.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Closed-Form Solution for Readout Weights)</span></p>

The optimal readout matrix $B$ can be solved analytically by minimizing the Mean Squared Error:

$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^T \lVert x_t - B z_t \rVert^2$$

Setting the derivative with respect to $B$ to zero:

$$\frac{\partial \mathcal{L}}{\partial B} = \sum 2(x_t - B z_t)\, z_t^T = 0$$

This yields the closed-form solution:

$$B = \left( \sum_{t=1}^T x_t\, z_t^T \right) \left( \sum_{t=1}^T z_t\, z_t^T \right)^{-1}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Derivation of the Readout Matrix)</span></p>

1. Represent the loss in terms of the $L_2$ norm: $\mathcal{L} \propto (x_t - B z_t)^T (x_t - B z_t)$.
2. Expand the product: $x_t^T x_t - z_t^T B^T x_t - x_t^T B z_t + z_t^T B^T B z_t$.
3. Differentiate with respect to $B$:
   * The derivative of $-2\, x_t^T B z_t$ with respect to $B$ is $-2\, x_t\, z_t^T$.
   * The derivative of $z_t^T B^T B z_t$ with respect to $B$ is $2\, B z_t\, z_t^T$.
4. Equate to zero: $\sum x_t\, z_t^T = B \sum z_t\, z_t^T$.
5. Isolate $B$ by multiplying by the inverse of the covariance-like matrix of reservoir states:

$$B = \left(\sum x_t\, z_t^T\right) \left(\sum z_t\, z_t^T\right)^{-1}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Practical Application)</span></p>

In practice, training a reservoir computer is "one line of Python code." Once the reservoir is entrained with the training data and the matrix $B$ is calculated, the model can predict complex sequences (like the Lorenz system) with surprising accuracy, provided the reservoir properties (spectral norm, sparsity) are correctly tuned.

</div>

#### Refining Reservoir Computing for Topology Preservation

While standard Reservoir Computing provides a computationally efficient framework for time-series prediction, it often fails to capture the true limiting dynamics of a system—the behavior as $t \to \infty$. To reconstruct a dynamical system properly, the model must do more than minimize immediate error; it must replicate the system’s invariant properties.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reservoir Weight Constraints)</span></p>

To ensure the reservoir remains stable and possesses the "Echo State Property," the internal weight matrix $W$ is typically constrained. A common condition is that the maximum singular value $\sigma_{\max}$ of $W$ is constrained:

$$\sigma_{\max}(W) \leq 1$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multi-Step Loss Function)</span></p>

Instead of minimizing the error for just $t+1$, we minimize the squared deviations over a window $U$:

$$\mathcal{L}_{\text{multi}} = \sum_{u=0}^{U} \sum_{t} \lVert b \cdot f^u(z_{t-u}) - x_t \rVert^2$$

where:
* $b$ is the readout matrix.
* $f^u$ represents the $u$-th composition of the reservoir’s recurrent transition function.
* $z_{t-u}$ is the latent state at time $t-u$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Multi-Step Prediction)</span></p>

By forcing the network to predict multiple steps into the future using its own previous predictions (recursive mode), we encourage the system to obey longer-term dynamics. This prevents the "drift" often seen in models trained only on single-step transitions.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Statistical Regularization)</span></p>

The total loss $\mathcal{L}$ can be augmented by a penalty term that measures the deviation of invariant statistics:

$$\mathcal{L} = \text{MSE} + \lambda \lVert C_{\text{data}} - C_{\text{model}} \rVert$$

where $C$ represents a dynamical invariant such as:

* **Maximum Lyapunov Exponent:** The rate of exponential separation of nearby trajectories.
* **Lyapunov Spectrum:** The full set of exponents characterizing the system’s stability.
* **Fractal Dimensionality:** Measures like the Correlation Dimension or the Kaplan-Yorke Dimension.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Physics-Based Training)</span></p>

This approach explicitly builds the "physics" or the "limiting dynamics" into the training process. If the true system is chaotic with a specific fractal dimension, we penalize the neural network if its autonomous behavior produces a different dimensionality.

</div>

### Autoencoders: Nonlinear Dimensionality Reduction

A central challenge in dynamical systems reconstruction is dimensionality. While a system might be observed in a high-dimensional space $X \in \mathbb{R}^D$, its true degrees of freedom often live on a much lower-dimensional manifold. An **Autoencoder** (AE) is a feed-forward neural network designed to learn a compressed representation of the input data. Unlike PCA, which is limited to linear projections, Autoencoders can capture nonlinear manifolds.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Autoencoder Structure)</span></p>

An **Autoencoder** consists of two primary components:

1. **Encoder** ($\phi$): Maps the high-dimensional input $x_t$ to a low-dimensional latent state $z_t$.
2. **Decoder** ($\phi^{-1}$): Maps the latent state $z_t$ back to the reconstructed input $\hat{x}_t$.

The architecture follows an "hourglass" shape, where the inner layer (the bottleneck) has significantly fewer units than the input/output layers.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(The Reconstruction Objective)</span></p>

The objective of the Autoencoder is to approximate the identity function through a bottleneck. We minimize the Mean Squared Error (MSE):

$$\mathcal{L}_{\text{AE}} = \sum_{t} \lVert x_t - \hat{x}_t \rVert^2 = \sum_{t} \lVert x_t - \phi^{-1}(\phi(x_t)) \rVert^2$$

Successful training implies that:

$$x_t \approx \phi^{-1}(\phi(x_t))$$

This suggests that $\phi^{-1}$ acts as an approximate inverse of $\phi$, and $z_t = \phi(x_t)$ captures the essential information required to reconstruct $x_t$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Deep Autoencoder: Layer-wise Formulation)</span></p>

Modern Autoencoders are implemented as Deep Neural Networks, alternating affine mappings with nonlinear activation functions. The latent representation $z_t$ for a three-layer encoder can be written as:

$$z_t = \sigma(W_3\, \sigma(W_2\, \sigma(W_1\, x_t + h_1) + h_2) + h_3)$$

where $\sigma$ is a nonlinearity like the ReLU ($\max(0, x)$), $W_i$ are weight matrices, and $h_i$ are bias vectors. The decoder follows a symmetric structure to expand $z_t$ back to the original dimensionality.

</div>

### Joint Manifold Discovery and SINDy Reconstruction

The ultimate goal in modern reconstruction (e.g., Champion and Brunton, 2019) is to combine the dimensionality reduction of Autoencoders with the interpretability of structural models like SINDy (Sparse Identification of Non-linear Dynamics). This builds upon the SINDy framework introduced in earlier lectures.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Latent Dynamics Assumption)</span></p>

We assume that while our measurements $x_t$ are high-dimensional and correlated, they are governed by a low-dimensional latent dynamic $\dot{z} = f(z)$. By training an Autoencoder and a dynamical model simultaneously, we identify the coordinate system ($z$) in which the dynamics are most "sparse" or "simple."

</div>

#### SINDy in Latent Space

Once the Autoencoder projects the data into the latent space $z_t$, we apply the SINDy framework to identify the governing differential equations.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(SINDy in Latent Space)</span></p>

We represent the dynamics of the latent state $z$ using a library of candidate basis functions $\Theta(z)$:

$$\dot{z} = \Theta(z)\,\Xi$$

where:
* $\Theta(z)$ contains functions like polynomials ($z_1, z_2, z_1^2, z_1 z_2, \dots$) or trigonometric functions.
* $\Xi$ is a sparse matrix of coefficients that determines which terms in the library actually contribute to the dynamics.

</div>

#### The Autoencoder-SINDy Architecture

To move from high-dimensional observations $x$ to a low-dimensional latent space $z$, we utilize an Autoencoder structure combined with a dynamics model:

* **Encoder** ($\phi$): A function that projects the input observations into a latent space: $z = \phi(x)$.
* **Decoder/Approximate Inverse** ($s$): A function that maps the latent state back to the observation space: $\hat{x} = s(z)$. Note that $s$ is an approximation of $\phi^{-1}$.
* **Latent Dynamics:** Within the latent space, the dynamics are governed by a function $\Theta(z)\,\xi$, representing the identified vector field.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Parsimonious Coordinates)</span></p>

This configuration allows the model to learn a coordinate transformation where the dynamics become parsimonious (sparse). Instead of modeling complex, high-dimensional noise, the system identifies the "true" underlying degrees of freedom.

</div>

#### Mathematical Formulation of the Loss Function

The training of this integrated system relies on a multi-term loss function designed to satisfy reconstruction accuracy, dynamical consistency, and sparsity.

**I. Reconstruction Loss**

The first term ensures the autoencoder can accurately reconstruct the input signal:

$$L_{\text{rec}} = \lVert x_t - s(\phi(x_t)) \rVert^2$$

**II. Latent Space Derivative Loss**

To ensure the dynamics in the latent space match the observed temporal evolution of the data, we must relate the latent derivatives $\dot{z}$ to the empirical observation derivatives $\dot{x}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Latent Derivatives via the Chain Rule)</span></p>

Given the relationship $z = \phi(x)$, we find the temporal derivative $\dot{z}$ by applying the chain rule:

$$\dot{z} = \frac{d}{dt} \phi(x) = \nabla_x \phi(x) \cdot \frac{dx}{dt}$$

In the context of our learned dynamics $\Theta(z)\,\xi$, we require:

$$\nabla_x \phi(x) \cdot \dot{x} \approx \Theta(z)\,\xi$$

Thus, the latent derivative loss is defined as:

$$L_{\dot{z}} = \lVert \nabla_x \phi(x) \cdot \dot{x} - \Theta(z)\,\xi \rVert^2$$

where $\dot{x}$ is an empirical estimate (e.g., first-order temporal differences).

</div>

**III. Observation Space Derivative Loss**

We also enforce that the derivatives reconstructed from the latent space match the empirical derivatives in the original observation space.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Observation Derivatives)</span></p>

Starting from $x \approx s(z)$, the temporal derivative is:

$$\dot{x} \approx \frac{d}{dt} s(z) = \nabla_z s(z) \cdot \dot{z}$$

Substituting the latent dynamics $\dot{z} = \Theta(z)\,\xi$:

$$\dot{x} \approx \nabla_z s(z) \cdot (\Theta(z)\,\xi)$$

The resulting loss term is:

$$L_{\dot{x}} = \lVert \dot{x} - \nabla_z s(z) \cdot (\Theta(z)\,\xi) \rVert^2$$

</div>

**IV. Regularization Loss**

To enforce the sparsity of the identified dynamics (the SINDy principle), we apply an $L_1$ penalty to the parameters $\xi$:

$$L_{\text{reg}} = \lVert \xi \rVert_1$$

**V. Total Integrated Loss**

The complete objective function is a weighted sum of these terms:

$$L_{\text{total}} = L_{\text{rec}} + \lambda_1 L_{\dot{z}} + \lambda_2 L_{\dot{x}} + \lambda_3 L_{\text{reg}}$$

where $\lambda_1, \lambda_2, \lambda_3$ are hyper-parameters that weigh the importance of reconstruction, dynamical fidelity, and sparsity.

#### Empirical Validation: The Lorenz Attractor

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(High-Dimensional Lorenz Projection)</span></p>

To evaluate the performance of this approach, it is tested on the Lorenz Attractor:

1. **System Dynamics:** The Lorenz system is defined in 3D latent space $(z_1, z_2, z_3)$:
   * $\dot{z}_1 = \sigma(z_2 - z_1)$
   * $\dot{z}_2 = z_1(R - z_3) - z_2$
   * $\dot{z}_3 = z_1 z_2 - V z_3$
2. **Projection:** The 3D system is projected into a high-dimensional space (e.g., 128 dimensions) using non-linear basis functions $U_1, \dots, U_6$.
3. **Task:** The model must take the 128D empirical data, project it back to a 3D latent manifold, and correctly identify the Lorenz equations.
4. **Result:** The framework successfully retrieves the underlying 3D structure and the sparse coefficients of the vector field from the high-dimensional embedding.

</div>

#### Supplemental Insights: Reservoir Computing and Predictability

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Lyapunov Exponents and Predictability)</span></p>

Including Lyapunov exponents in the training criterion significantly improves the "mean valid prediction time." Models trained with these dynamical constraints can predict the future state of a chaotic system for a much longer duration compared to those trained solely on standard error metrics.

</div>

## Lecture 12

This lecture introduces Koopman Operator Theory as a principled framework for linearizing nonlinear dynamics, and then transitions from deterministic to stochastic formulations. We cover Koopman Autoencoders, the move to probabilistic latent variable models via Variational Autoencoders (VAEs), maximum likelihood estimation with latent variables, the Evidence Lower Bound (ELBO), and the reparameterization trick that makes end-to-end training possible.

### Koopman Operator Theory

In Lecture 11, we explored Autoencoders as a mechanism for extracting lower-dimensional embeddings from high-dimensional data. Our objective now shifts toward a more ambitious goal: identifying a specific space of observations where a nonlinear system can be approximated by a **linear** dynamical system. For linear systems, the mathematical landscape is well-defined — we possess explicit expressions for equilibria and can rigorously analyze global behavior. The formal mathematical background for this coordinate transformation is known as **Koopman Operator Theory**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Koopman Operator)</span></p>

The **Koopman operator** $K$ is a linear, infinite-dimensional operator that acts on functions of the state space (observation functions) rather than the state space itself.

Given a discrete-time nonlinear dynamical system $x_{t} = f(x_{t-1})$, we define a set of observation functions or mappings $g$ that translate our observed state space into an infinite-dimensional Hilbert space $\mathcal{H}$. In this space, there exists a linear operator $K$ such that:

$$K\, g(x_t) = g(f(x_t)) = g(x_{t+1})$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Infinite-Dimensional Caveat)</span></p>

The fundamental theorem of Koopman theory guarantees that a linear representation exists if we move to an infinite-dimensional space. This is the primary challenge in practical applications: we must find a finite-dimensional approximation (a set of "basis functions") that captures the essential dynamics without requiring an infinite number of dimensions.

</div>

#### Deep Learning for Koopman Representations

To move from theory to application, we utilize the framework proposed by Lusch, Kutz, and Brunton (2018). Their approach uses an Autoencoder to learn the optimal coordinate transformation that projects data into a space where dynamics are linear.

The goal is to find a representation $\phi$ such that the forward iteration in the latent space is governed by a linear operator $\tilde{K}$:

$$\phi(x_{t+1}) = \tilde{K}\, \phi(x_t)$$

If the operator $\tilde{K}$ is diagonalizable, we can further simplify the representation. By expressing the dynamics in terms of the eigenvectors of $K$, we can replace $\tilde{K}$ with a diagonal matrix of eigenvalues $\Lambda$:

$$\phi(x_{t+1}) = \Lambda\, \phi(x_t)$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Applicability and Limitations)</span></p>

* **Simple Systems:** For systems like limit cycles, which exhibit peaks at finite frequencies in their power spectrum, this method works exceptionally well.
* **Chaotic Systems:** In chaotic or more complex systems, the power spectrum is often "smeared out," making an exact linear approximation significantly more difficult.

</div>

#### Mathematical Parameterization of the Linear Operator

To characterize the dynamics within the linear latent space, we parameterize the operator $\tilde{K}$ using its eigenvalues. Recall from the linear systems analysis in Part I that the real and imaginary parts of the eigenvalues $\lambda$ provide a complete characterization of the system’s behavior.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Eigenvalue Decomposition of $\tilde{K}$)</span></p>

We define the eigenvalues $\lambda$ in terms of their real parts $\mu$ and imaginary parts $\omega$:

$$\lambda = \mu \pm i\omega$$

The general solution for a diagonalizable linear system can be expressed as a combination of eigenvectors $v_i$ and coefficients $c_i$:

$$x(t) = \sum_{i=1}^{m} c_i\, v_i\, e^{\lambda_i t}$$

Using Euler’s formula, this can be expanded to show the decay/growth and oscillatory components:

$$e^{\lambda_i t} = e^{\mu_i t} (\cos(\omega_i t) + i \sin(\omega_i t))$$

</div>

To implement this in a real-valued neural network, we construct $\tilde{K}$ using a **block diagonal form**. Since imaginary eigenvalues always appear in complex conjugate pairs, we use $2 \times 2$ blocks to represent each pair.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Block Diagonal Construction)</span></p>

Each block $B$ is defined by a decay term and a rotation matrix:

$$B = e^{\mu \Delta t} \begin{pmatrix} \cos(\omega \Delta t) & -\sin(\omega \Delta t) \\ \sin(\omega \Delta t) & \cos(\omega \Delta t) \end{pmatrix}$$

By learning the values of $\mu$ and $\omega$ for each block, the autoencoder approximates the dynamics in what is sometimes called the **measurement space**.

</div>

#### Autoencoder Architecture for Koopman Embedding

The architecture consists of three components:

* **Encoder:** Projects the observed state $x_t$ into the latent/measurement space $y_t$.
* **Linear Propagator:** Applies the linear operator $K(\lambda)$ to progress the state forward in time: $y_{k+1} = K\, y_k$.
* **Decoder:** Maps the latent state back to the original observation space, effectively implementing the inverse of the encoder.

### Koopman Autoencoders (KAE)

Koopman Autoencoders bridge the gap between deep learning and classical dynamical systems theory by seeking a latent space where non-linear dynamics can be represented as a linear system.

#### The Koopman Embedding Objective

The primary goal of a Koopman Autoencoder is to find a mapping $\phi$ (the encoder) and its approximate inverse $\phi^{-1}$ (the decoder) such that the dynamics in the embedding space follow a linear operator $\tilde{K}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reconstruction Loss)</span></p>

The **Reconstruction Loss** ($\mathcal{L}_{\text{recon}}$) ensures that the embedding space preserves the essential information of the observation space. It penalizes the difference between the original input $x_t$ and its reconstruction through the autoencoder:

$$\mathcal{L}_{\text{recon}} = \sum_{t} \lVert x_t - \phi^{-1}(\phi(x_t)) \rVert$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Approximate Inverses)</span></p>

It is important to note that $\phi^{-1}$ is an approximate expression. In most neural network architectures, it is not a guaranteed mathematical inverse of $\phi$, but rather a parameterized function (the decoder) trained to minimize the reconstruction error.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Koopman Loss)</span></p>

The **Koopman Loss** ($\mathcal{L}_{\text{Koopman}}$) enforces the linear evolution property within the latent space. It ensures that the projected state at the next time step $\phi(x_{t+1})$ matches the result of applying the linear operator $\tilde{K}$ to the current projected state $\phi(x_t)$:

$$\mathcal{L}_{\text{Koopman}} = \sum_{t} \lVert \phi(x_{t+1}) - \tilde{K}\, \phi(x_t) \rVert$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Parameterization of $\tilde{K}$)</span></p>

The matrix $\tilde{K}$ is not typically learned as a dense weight matrix. Instead, it is parameterized via its eigenvalues. Specifically, the real and imaginary parts of the eigenvalues are the free parameters $\theta$ adjusted during training. These form $2 \times 2$ blocks that constitute the block-diagonal structure of $\tilde{K}$.

</div>

#### Multi-Step Prediction and Robustness

To increase the stability and robustness of the learned dynamics, we often extend the one-step loss to an $M$-step iterated loss. This technique is philosophically similar to the sparse and generalized teacher forcing methods discussed in Lectures 10–11.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($M$-Step Latent Loss)</span></p>

The $M$-step Latent Loss ensures that the linear operator remains valid over longer trajectories in the embedding space:

$$\mathcal{L}_{m\text{-step}} = \sum_{t} \lVert \phi(x_{t+m}) - \tilde{K}^m\, \phi(x_t) \rVert$$

where $\tilde{K}^m$ represents the $m$-th power of the linear operator, signifying the $m$-step forward iteration in the latent space.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Observed $M$-Step Loss)</span></p>

To further constrain the system and prevent the latent space from drifting into unphysical solutions, we define a loss in the actual observation space for $m$ steps ahead:

$$\mathcal{L}_{\text{obs},\, m} = \sum_{t} \lVert x_{t+m} - \phi^{-1}(\tilde{K}^m\, \phi(x_t)) \rVert$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Redundancy for Stability)</span></p>

While $\mathcal{L}_{\text{obs},\, m}$ might appear redundant if $\mathcal{L}_{m\text{-step}}$ and $\mathcal{L}_{\text{recon}}$ are minimized, explicitly including it provides the algorithm with more robustness. It constrains the degrees of freedom by forcing the long-term latent predictions to remain decodable into accurate observations.

</div>

#### Total Loss Formulation and Regularization

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Total KAE Loss)</span></p>

The total loss $\mathcal{L}_{\text{total}}$ is defined as:

$$\mathcal{L}_{\text{total}} = \alpha_1\, \mathcal{L}_{\text{recon}} + \alpha_2\, \mathcal{L}_{\text{obs},\, m} + \alpha_3\, \mathcal{L}_{\text{Koopman}} + \alpha_4\, \mathcal{L}_{\text{reg}}$$

where:
* $\alpha_i$ are weighting hyperparameters.
* $\mathcal{L}_{\text{reg}}$ is a penalty on the parameters $\theta$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Regularization and Sparsity)</span></p>

The parameter $\theta$ includes the real and imaginary parts of the eigenvalues of $\tilde{K}$ as well as the weights of the autoencoder. Penalizing the size of these parameters acts as a regularization strategy to prevent overfitting, similar to the SINDy approach from Lecture 11, effectively encouraging the model to find the most salient features of the dynamics.

</div>

#### Global vs. Local Operator Parameterization

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Global vs. Local Koopman Operator)</span></p>

When designing the Koopman matrix $\tilde{K}$, there are two primary approaches:

1. **Global Koopman Operator:** A single operator $\tilde{K}$ is learned for the entire system. This implies a truly linear system in the latent space.
2. **Local Koopman Operator:** The eigenvalues (and thus $\tilde{K}$) can be parameterized to depend on the current time step. While the operation at any single step is linear, the progression across steps becomes non-linear. This offers significantly more flexibility for complex systems while maintaining the interpretability of a linear step-by-step transition.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Linearization of Fast-Slow Systems)</span></p>

Consider a system defined by two differential equations where one variable fluctuates much faster than the other: $\dot{x}_2 \propto \mu(x_2 - x_1^2)$, with $\mu \gg \lambda$ (where $\lambda$ represents the slower timescale).

Observations from KAE Reconstruction:

* The KAE identifies a latent space where the dynamics quickly converge to a lower-dimensional manifold.
* The system learns to linearize the underlying non-linear dynamics.
* By analyzing the learned eigenvalues of $\tilde{K}$, researchers can directly understand the composition and stability of the underlying system, such as how quickly the "fast" variable converges toward the "slow" manifold.

</div>

### Transition to Stochastic Formulations

The dynamical systems theory discussed thus far has been largely deterministic. However, real-world data is inherently stochastic, containing noise and probabilistic uncertainties. To account for this, we must move beyond standard autoencoders.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(From Deterministic to Probabilistic)</span></p>

In a standard autoencoder (as used in Lecture 11), we project a point $x_t$ to a single point in the latent space. In a **Variational Autoencoder** (VAE), we aim to accommodate probabilistic properties in both the latent space and the observation space. This allows the model to handle noise and represent the underlying distribution of the dynamics rather than just a single trajectory.

</div>

#### Motivations for Stochasticity

There are two primary motivations for introducing stochasticity into our dynamical models:

1. **Stochastic Latent Evolution:** We often want the underlying latent system to be stochastic. In continuous time, this leads us to Stochastic Differential Equations (SDEs). In discrete time, this is typically handled by adding a noise term (e.g., Gaussian noise) to the state transition. This accounts for processes within the dynamical system that we cannot control or explicitly model, but which impact its temporal evolution.
2. **Diverse Observation Modalities:** Many real-world observations do not follow a continuous Gaussian form. We require a framework capable of handling "non-continuous" and "non-Gaussian" data (e.g., point processes, count data).

#### Embedding Theorems for Point Processes

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Delay Embedding for Point Processes)</span></p>

As established in literature (e.g., Sauer, 1994), delay embedding theorems can be extended to point processes. This suggests that even from spike processes, it is mathematically possible to reconstruct the dynamical system (such as the Lorenz attractor) in a topological sense.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Lorenz Attractor from Point Process Data)</span></p>

Recall the Lorenz attractor’s "butterfly wing" structure, where the system switches between two loops. If we only record the maxima of the trajectories (a form of point process), we can still achieve a reconstruction that is topologically equivalent to the original attractor.

</div>

### Stochastic Latent Dynamical Systems

We now define a discrete-time latent dynamical system that incorporates stochasticity.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Latent Variable)</span></p>

We define $z_t \in \mathbb{R}^m$ as the **latent state** of the system at time $t$. In the context of an RNN, these would be the latent states.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Latent Model)</span></p>

The evolution of the latent state is governed by a transition function $f_\theta$ and a noise term:

$$z_t = f_\theta(z_{t-1},\, s_t) + \epsilon_t$$

where:
* $z_{t-1}$ is the previous state.
* $s_t$ represents external inputs.
* $\epsilon_t$ is a random variable representing system noise.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Observation Model)</span></p>

The observations $x_t$ are coupled to the latent process through a decoder. For simplicity, we can assume a linear mapping with additive noise:

$$x_t = B\, z_t + \eta_t$$

where:
* $B$ is a transition matrix.
* $\eta_t$ is the observation noise.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Zero Mean Noise?)</span></p>

In these noise processes ($\epsilon_t$, $\eta_t$), we generally assume a mean of zero. This is because any non-zero mean (offset) can be mathematically absorbed into the constant terms of the function $f_\theta$ or the observation matrix $B$.

</div>

#### Probability Densities in State-Space Models

In a probabilistic setting, every variable is treated as being drawn from a distribution.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multivariate Gaussian Distribution)</span></p>

We assume the noise $\epsilon_t$ and the initial state $z_1$ are Gaussian distributed. A multivariate Gaussian distribution $\mathcal{N}(\mu, \Sigma)$ is characterized by its mean $\mu$ and covariance matrix $\Sigma$.

The Probability Density Function (PDF) for an $m$-dimensional Gaussian variable is:

$$p(\epsilon) = \frac{1}{(2\pi)^{m/2} \lvert\Sigma\rvert^{1/2}} \exp \left( -\frac{1}{2} (\epsilon - \mu)^T \Sigma^{-1} (\epsilon - \mu) \right)$$

The initial state is also a random variable: $z_1 \sim \mathcal{N}(\mu_0, \Sigma_0)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(General Formulation)</span></p>

While we often add a noise term (e.g., $z_t = f(z_{t-1}) + \epsilon_t$), a more general and powerful way to express this is to state that the next state is sampled directly from a conditional distribution: $z_t \sim p(z_t \mid z_{t-1})$. This allows us to move beyond Gaussian assumptions to any probability distribution.

</div>

#### Probabilistic Observation Models

In dynamical systems, we treat our observations $X_t$ as being generated by an underlying probability distribution $P(X_t \mid Z_t)$ conditioned on the latent state $Z_t$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Linear Observation Model)</span></p>

The observation $X_t$ is defined as:

$$X_t = B\, Z_t + \eta_t$$

where $B$ is a transition matrix, $Z_t$ is the latent state, and $\eta_t$ is the observation noise.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Conditional Distribution of $X_t$)</span></p>

To find the probability distribution $P(X_t \mid Z_t)$, we must determine its mean and variance given that $Z_t$ is fixed.

1. **Assumption:** Let $\eta_t$ be Gaussian noise with mean zero and covariance $\Gamma$, such that $\eta_t \sim \mathcal{N}(0, \Gamma)$.
2. **Expectation:** Using the linearity of the expectation operator:

$$\mathbb{E}[X_t \mid Z_t] = \mathbb{E}[B\, Z_t + \eta_t] = B\, Z_t + 0 = B\, Z_t$$

3. **Conclusion:** Because the only source of randomness is the Gaussian variable $\eta_t$, $X_t$ also follows a normal distribution:

$$P(X_t \mid Z_t) = \mathcal{N}(B\, Z_t,\, \Gamma)$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Latent Transition Model)</span></p>

The probability distribution of the current latent state $Z_t$ given the preceding state $Z_{t-1}$ is:

$$Z_t = f(Z_{t-1},\, s_t) + \epsilon_t$$

where $\epsilon_t$ has covariance $\Sigma$. Once $Z_{t-1}$ is given, $\epsilon_t$ is the sole source of randomness, so:

$$P(Z_t \mid Z_{t-1}) = \mathcal{N}(f(Z_{t-1},\, s_t),\, \Sigma)$$

</div>

#### Modeling Non-Gaussian Data

The probabilistic framework is flexible enough to handle data types that do not follow a Gaussian distribution.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Poisson Observation Model)</span></p>

For count data $C_t$ at time $t$, the probability of observing a specific count is given by:

$$P(C_t \mid Z_t) = \frac{\lambda_t^{C_t}\, e^{-\lambda_t}}{C_t!}$$

where $\lambda_t$ is the mean parameter of the distribution.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Exponential Link Function)</span></p>

A common way to parameterize $\lambda_t$ is to use an affine transformation of $Z_t$ passed through an exponential function:

$$\lambda_t = \exp(B\, Z_t + b_0)$$

The exponential function ensures that $\lambda_t$ is strictly positive, as required for count data. Other functions like the ReLU can also be employed.

</div>

### Variational Autoencoders (VAEs)

Variational Autoencoders represent a significant advancement in machine learning (notably introduced by Kingma & Welling and Rezende et al. in 2014). They extend the autoencoder concept to operate on complete probability distributions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Encoder — Inference Model)</span></p>

The encoder is a probability distribution $q_\phi(Z_t \mid X_t)$ parameterized by $\phi$. It serves as an approximation of the true (but unknown) posterior distribution of the latent states given the observations.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Decoder — Generative Model)</span></p>

The decoder is a probability distribution $p_\theta(X_t \mid Z_t)$ parameterized by $\theta$. It describes how to generate observations back from the latent state.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Approximating the True Posterior)</span></p>

The goal of the VAE is to make the learned distribution $q_\phi(Z_t \mid X_t)$ as close as possible to the true probability distribution of the latent states $Z_t$ given the observations $X_t$. Since the true posterior is inaccessible, we use this variational approximation.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Prior Distribution)</span></p>

The **prior** $p(Z_t)$ represents our beliefs about the latent states before observing any data. In the context of dynamical systems, this prior often emerges from the conditional latent model (the transition $Z_t$ given $Z_{t-1}$).

</div>

### Maximum Likelihood Estimation with Latent Variables

To estimate model parameters, we adopt the **Maximum Likelihood Estimation** (MLE) framework. The core objective is to identify the parameters that make the observed data most plausible under the assumed model.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Likelihood Function)</span></p>

Let $X$ represent a sequence of observed variables across all time steps. Let $\theta$ represent the parameters of the model. The **likelihood function** is defined as the probability (or probability density) of observing the exact data $X$ given the parameters $\theta$:

$$L(\theta) = P(X \mid \theta)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition behind MLE)</span></p>

We seek the parameters $\theta$ that maximize the probability of our actual observations. We want to avoid models where our observed data has a very low chance of occurring; instead, we favor the model that assigns the highest possible chance to the data we actually collected.

</div>

#### The Log-Likelihood

In practice, it is often more convenient to maximize the logarithm of the likelihood function.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Log-Likelihood)</span></p>

The **log-likelihood** is the natural logarithm of the likelihood function:

$$\ell(\theta) = \ln P(X \mid \theta)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Mathematical Convenience)</span></p>

There are two primary reasons for using the log-likelihood:

1. **Exponent Reduction:** In many common distributions (Poisson, Gaussian), the probability density involves an exponential term. Taking the logarithm "pulls down" the exponent, simplifying differentiation.
2. **Product-to-Sum Conversion:** If observations are independent, the joint probability is a product of individual probabilities. The logarithm converts these products into sums, which are significantly easier to manipulate.

</div>

#### Marginalization and the Path Integral

In systems involving latent states $Z$, the observations $X$ depend on these hidden variables. To obtain the likelihood of $X$, we must "integrate out" all possible latent states.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Path Integral)</span></p>

In the context of a dynamical system over $T$ time steps with an $m$-dimensional latent vector $Z$ at each step, we must account for every possible trajectory the system could have taken in the latent space:

$$P(X \mid \theta) = \int P(X, Z \mid \theta)\, dZ$$

where $Z$ represents the collection of all latent variables $z_1, z_2, \dots, z_T$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dimensionality)</span></p>

If $Z$ is an $m$-dimensional vector at each time step and we have $T$ time steps, the integral is over $T \times m$ random variables. This "path integral" sums the joint probability density $P(X, Z)$ across the entire latent space to find the marginal probability density.

</div>

### Variational Inference and the ELBO

In most realistic scenarios involving non-linear functions (such as those found in RNNs), the integral required to find $P(X \mid \theta)$ is **intractable**. To resolve this, we introduce an auxiliary distribution $Q_\phi(Z \mid X)$ (the encoder) as a "helping construct."

We can rewrite the log-likelihood by inserting this distribution:

$$\ln P(X \mid \theta) = \ln \int Q_\phi(Z \mid X) \frac{P_\theta(X, Z)}{Q_\phi(Z \mid X)}\, dZ$$

At this stage, the expression remains exact because $Q_\phi$ cancels out. However, the logarithm remains outside the integral. To bring it inside, we use Jensen’s Inequality.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Jensen’s Inequality)</span></p>

For a concave function $f$ and a random variable $x$:

$$f(\mathbb{E}[x]) \geq \mathbb{E}[f(x)]$$

Given that the logarithm ($\ln$) is a concave function, the following holds for a probability distribution $Q$:

$$\ln \int Q(Z)\, f(Z)\, dZ \geq \int Q(Z)\, \ln f(Z)\, dZ$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Geometric Intuition for Jensen’s Inequality)</span></p>

1. Consider a concave curve (like the logarithm).
2. Pick two points on the curve, $x_0$ and $x_1$, and draw a line segment (a chord) connecting them.
3. For any concave function, the line segment connecting two points lies **below or on** the curve.
4. Mathematically, for $\alpha \in [0, 1]$:

$$\alpha\, f(x_0) + (1-\alpha)\, f(x_1) \leq f(\alpha\, x_0 + (1-\alpha)\, x_1)$$

5. Generalizing from a discrete weighted sum to an integral, and noting that $Q(Z)$ acts as the weights (where $Q(Z) \geq 0$ and $\int Q(Z)\, dZ = 1$), the inequality allows us to "pull" the logarithm inside the integral.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Evidence Lower Bound — ELBO)</span></p>

The **ELBO** (also related to "free energy" in physics) is defined as:

$$\text{ELBO} = \int Q_\phi(Z \mid X)\, \ln \frac{P_\theta(X, Z)}{Q_\phi(Z \mid X)}\, dZ$$

By Jensen’s Inequality:

$$\ln P(X \mid \theta) \geq \text{ELBO}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimization Strategy)</span></p>

Since the true log-likelihood is intractable, we instead maximize the ELBO. Because the ELBO is a lower bound on the log-likelihood, maximizing the ELBO pushes the likelihood upward. Under certain conditions, this bound can become tight (reaching equality), effectively allowing us to optimize the parameters $\theta$ and $\phi$ of our dynamical system efficiently.

</div>

#### ELBO Decomposition

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(ELBO Decomposition)</span></p>

The objective can be expressed as:

$$\mathbb{E}_{q_\phi} \left[ \log \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \right] = \mathbb{E}_{q_\phi} [\log p_\theta(x, z)] - \mathbb{E}_{q_\phi} [\log q_\phi(z \mid x)]$$

By applying the chain rule of probability to the joint density $p_\theta(x, z) = p_\theta(x \mid z)\, p(z)$, we obtain three terms:

1. $\log p_\theta(x \mid z)$: The decoder model (likelihood of the data given the latent).
2. $\log p(z)$: The latent model (the prior distribution).
3. $-\log q_\phi(z \mid x)$: The term associated with the approximate posterior.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Entropy Interpretation)</span></p>

The term $-\int q_\phi \log q_\phi$ is known as the **entropy** of the distribution. Thus, the objective can be viewed as the expectation of the latent model and decoder likelihood plus the entropy of the approximate posterior $q_\phi(z \mid x)$.

</div>

#### Sampling-Based Approximations

While we have an expression for the expectation, it remains a high-dimensional integral. We approximate it using a sum.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sampling Estimator)</span></p>

We draw $L$ samples $z_l$ from the probability distribution $q_\phi(z \mid x)$ and approximate the expectation by calculating the average:

$$\mathbb{E}_{q_\phi}[f(z)] \approx \frac{1}{L} \sum_{l=1}^{L} f(z_l)$$

Applying this to our ELBO formulation:

$$\mathcal{L} \approx \frac{1}{L} \sum_{l=1}^{L} \left[ \log p_\theta(x \mid z_l) + \log p_\theta(z_l) - \log q_\phi(z_l \mid x) \right]$$

where $z_l \sim q_\phi(z \mid x)$.

</div>

### The Reparameterization Trick

Our goal is to perform gradient descent with respect to the parameters $\theta$ (decoder/latent) and $\phi$ (encoder). Taking the derivative with respect to $\theta$ is straightforward. However, a significant issue arises with $\phi$: because the samples $z_l$ are drawn from a distribution $q_\phi$ that depends on $\phi$, the parameters are only "indirectly" involved in the expression. If we take the derivative directly, the contribution of $\phi$ through the sampling process evaluates to zero, which is mathematically incorrect.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Reparameterization Trick)</span></p>

The **reparameterization trick** involves expressing a random variable $z$ from a distribution $q_\phi(z \mid x)$ as a deterministic transformation of a parameterless noise variable $\epsilon$:

$$z = g_\phi(\epsilon,\, x)$$

For a Gaussian distribution, this becomes:

$$z = \mu + \sigma \odot \epsilon$$

where:
* $\mu$ and $\sigma$ are the mean and standard deviation (derived from $\phi$ via a neural network).
* $\epsilon$ is drawn from a standard normal distribution $\mathcal{N}(0, I)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Reparameterization is Necessary)</span></p>

If we simply sample a value $z$ from a distribution, $z$ becomes a "number" in the computational graph. The sampling operation itself is non-differentiable. By using the reparameterization trick, the stochasticity is isolated in the variable $\epsilon$, which has no parameters. The parameters we care about ($\phi$) are now part of the deterministic function $g_\phi$. This makes it possible to calculate the gradient of the expectation with respect to $\phi$ using standard backpropagation.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Gradient Estimation via Sampling)</span></p>

To optimize the ELBO, we approximate the integral using samples:

1. **Objective:** We want to compute $\nabla_\phi \mathbb{E}_{q_\phi(z)} [f(z)]$.
2. **Transformation:** Express $z = \mu + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$.
3. **Substitution:** The expectation becomes $\mathbb{E}_{p(\epsilon)} [f(g_\phi(\epsilon))]$.
4. **Key step:** Since the distribution of $\epsilon$ no longer depends on $\phi$, we can move the gradient inside the expectation:

$$\nabla_\phi \mathbb{E}_{p(\epsilon)} [f(g_\phi(\epsilon))] = \mathbb{E}_{p(\epsilon)} [\nabla_\phi f(g_\phi(\epsilon))]$$

5. **Monte Carlo Estimation:** We approximate by taking $L$ samples:

$$\nabla_\phi \mathbb{E}_{q_\phi(z)} [f(z)] \approx \frac{1}{L} \sum_{i=1}^{L} \nabla_\phi f(g_\phi(\epsilon_i))$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Distribution Compatibility)</span></p>

While common for Gaussian variables, the reparameterization trick can be applied to various distributions within the exponential family, such as the Poisson distribution. The core requirement is the ability to rephrase the distribution in terms of a parameterless distribution.

</div>

### Stochastic Gradient Variational Bayes (SGVB)

By integrating the reparameterization trick into the sampling estimator, we can compute the gradients of the ELBO with respect to both $\theta$ and $\phi$ simultaneously.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stochastic Gradient Variational Bayes)</span></p>

**SGVB** is the procedure of optimizing the ELBO by:

1. Reparameterizing the latent variable $z$.
2. Approximating the expectation via sampling.
3. Computing the gradients with respect to parameters $\theta$ and $\phi$ using standard backpropagation.

This formulation allows for efficient, end-to-end training of latent variable models using stochastic gradient descent.

</div>

#### Neural Network Integration in Probabilistic Models

In modern dynamical systems reconstruction, we utilize neural networks to map input data onto the parameters of the probability distributions that describe latent states or observations.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Parameter Mapping)</span></p>

The parameters of our distributions (such as the mean $\mu$ and variance $\sigma^2$) are not learned directly as static values. Instead, they are the **outputs of functional mappings** (neural networks) parameterized by $\theta$ (decoder parameters) and $\phi$ (encoder parameters).

* **Mean Mapping:** A deep neural network maps the observation $x_t$ onto a mean vector: $\mu(z) = \text{NN}_\phi(x_t)$.
* **Variance Mapping:** To ensure variances remain positive, a network might output the log-variance, or use $\sigma^2 = \exp(\text{NN}_\phi(x_t))$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Decoder Flexibility)</span></p>

The decoder model is not limited to a single distribution type. Depending on the nature of the empirical data, one might use:

* **Gaussian Distributions:** Common for continuous state spaces.
* **Count Processes:** Useful for discrete, event-based data.
* **Recurrent Neural Networks (RNNs):** These describe the temporal evolution in the latent space, which is then coupled to the observation model to estimate the likelihood of the trajectories.

</div>

#### Model Evaluation: Divergence Measures and Attractor Reconstruction

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(KL Divergence as a Quality Measure)</span></p>

The **Kullback-Leibler Divergence** is used to quantify the "geometrical disagreement" between two distributions in the state space. In the context of evaluation (rather than training), it compares the distribution of the true trajectories with the distribution of the trajectories generated by the model.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Lorenz Attractor Reconstruction)</span></p>

In experiments involving the Lorenz attractor, researchers aimed to reconstruct the attractor object from data.

* **Method:** Trajectories were encoded according to the octant in which they resided, providing this categorical information to the estimation algorithm.
* **Observation:** Providing this additional structural information significantly improved reconstruction.
* **Result:** The cumulative density of the KL divergence shifted toward smaller values, indicating a higher quality of reconstruction and better agreement between the true attractor and the model-generated one.

</div>

#### Challenges and Future Directions

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Exploding Gradient Problem)</span></p>

A recurring issue in dynamical systems reconstruction is exploding gradients. When we allow a model the freedom to explore the future by simulating trajectories, the sensitivity of the long-term output to the initial parameters can grow exponentially. This makes the gradient descent process unstable.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Continuous-Time Models — Neural ODEs)</span></p>

To address gradient instability and better represent physical systems, the field is moving toward **Neural Ordinary Differential Equations** (Neural ODEs). Instead of discrete-time updates, the system is modeled using continuous-time dynamics. This provides a more natural framework for describing the temporal evolution of latent variables and offers specialized techniques to mitigate the exploding gradient problem. The integration of random variables with continuous-time models represents a robust combined approach, improving both the exploration of the state space and the stability of the reconstruction process.

</div>

## Lecture 13

This lecture brings together the probabilistic framework for dynamical systems with continuous-time neural network architectures. We begin by formalizing latent variable models and variational inference, then extend these ideas to multimodal data integration. The second half covers the transition from discrete residual networks to Neural ODEs, the adjoint sensitivity method for training, and Physics-Informed Neural Networks (PINNs).

### Foundations of Probabilistic Latent Variable Models

In the study of dynamical systems through the lens of machine learning, we often deal with systems where the underlying governing laws are not directly observable. Instead, we possess a dataset of vector-valued observations, $\mathbf{x}$, from which we aim to reconstruct the latent dynamics.

#### The Generative Framework: Latent States and Observations

We assume the existence of an underlying latent process, $\mathbf{z}_t$, governed by a recursive map. In modern applications, this map is typically represented by a Recurrent Neural Network (RNN) with parameters $\theta$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Latent Dynamical Model)</span></p>

The latent state evolution and the corresponding observation model are defined by the following system of equations:

1. **Latent Transition:** $\mathbf{z}_t = f_{\theta}^{\text{latent}}(\mathbf{z}_{t-1}) + \boldsymbol{\epsilon}$
2. **Observation (Decoder) Model:** $\mathbf{x}_t = g(\mathbf{z}_t) + \boldsymbol{\eta}$

Where:

* $f_{\theta}^{\text{latent}}$ is a recursive function (e.g., an RNN).
* $g(\cdot)$ is the decoder mapping latent states to observations. In the simplest case, this is a linear mapping $\mathbf{B}\mathbf{z}_t$, though it can be an arbitrarily complex deep neural network.
* $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \Sigma)$ represents process noise.
* $\boldsymbol{\eta} \sim \mathcal{N}(0, \Gamma)$ represents observation noise.

</div>

#### Probabilistic Reformulation of Dynamical Systems

To account for stochasticity at both the process and observation levels, we reformulate the model using probability distributions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Probabilistic State-Space Model)</span></p>

The system is defined by the transition distribution and the emission distribution:

1. **Transition Probability:** $p(\mathbf{z}_t \mid \mathbf{z}_{t-1}) = \mathcal{N}(f_{\theta}^{\text{latent}}(\mathbf{z}_{t-1}),\; \Sigma)$
2. **Emission Probability:** $p(\mathbf{x}_t \mid \mathbf{z}_t) = \mathcal{N}(g(\mathbf{z}_t),\; \Gamma)$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Flexibility of the Probabilistic Framework)</span></p>

By adopting a probabilistic framework, we are no longer restricted to Gaussian noise. We can "sneak in" any distribution that suits the data type. For instance, if the observations consist of discrete counts, a Gaussian model is inappropriate.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Poisson Observations)</span></p>

For $n$-dimensional count data $\mathbf{x}_t \in \mathbb{N}^n$, we can employ a Poisson decoder:

$$p(\mathbf{x}_t \mid \mathbf{z}_t) = \text{Poisson}(\lambda_t)$$

To ensure the rate parameter $\lambda_t$ remains positive ($\lambda_t \geq 0$), we use a mapping such as the exponential function:

$$\lambda_t = \exp(\mathbf{B}\mathbf{z}_t)$$

</div>

#### Variational Inference and the Evidence Lower Bound (ELBO)

The objective of learning is to find parameters $\theta$ that maximize the log-likelihood of the observed data $\mathbf{x}$:

$$\max_{\theta} \log p_{\theta}(\mathbf{x})$$

However, calculating $p_{\theta}(\mathbf{x})$ requires integrating out the latent states $\mathbf{z}$ across all possible paths:

$$p_{\theta}(\mathbf{x}) = \int p_{\theta}(\mathbf{x}, \mathbf{z})\, d\mathbf{z}$$

This integral is generally intractable for complex high-dimensional systems. To solve this, we introduce a helper distribution, $q_{\phi}(\mathbf{z} \mid \mathbf{x})$, known as the **encoder**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Evidence Lower Bound (ELBO))</span></p>

The log-likelihood can be lower-bounded by the ELBO, which is defined as:

$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})} \left[ \log p_{\theta}(\mathbf{x}|\mathbf{z}) + \log p_{\theta}(\mathbf{z}) - \log q_{\phi}(\mathbf{z}|\mathbf{x}) \right]$$

</div>

<details class="accordion" markdown="1">
<summary>Proof: Derivation of the ELBO</summary>

Using Bayes' Law and Jensen's Inequality, we decompose the log-likelihood:

1. Start with the log-marginal likelihood: $\log p_{\theta}(\mathbf{x}) = \log \int p_{\theta}(\mathbf{x}, \mathbf{z})\, d\mathbf{z}$.
2. Introduce the encoder distribution $q_{\phi}(\mathbf{z}|\mathbf{x})$:

$$\log \int q_{\phi}(\mathbf{z}|\mathbf{x}) \frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{q_{\phi}(\mathbf{z}|\mathbf{x})}\, d\mathbf{z}$$

3. Apply the definition of expectation:

$$\log \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})} \left[ \frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{q_{\phi}(\mathbf{z}|\mathbf{x})} \right]$$

4. By Jensen's Inequality, move the logarithm inside the expectation:

$$\geq \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})} \left[ \log \frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{q_{\phi}(\mathbf{z}|\mathbf{x})} \right]$$

5. Expand the joint distribution $p_{\theta}(\mathbf{x}, \mathbf{z}) = p_{\theta}(\mathbf{x}|\mathbf{z})p_{\theta}(\mathbf{z})$:

$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})} \left[ \log p_{\theta}(\mathbf{x}|\mathbf{z}) + \log p_{\theta}(\mathbf{z}) \right] + H(q_{\phi})$$

where $H(q_{\phi})$ is the entropy of the encoder distribution.

</details>

#### Stochastic Gradient Variational Bayes (SGVB)

To optimize the ELBO, we use a sampling estimate. However, we cannot sample $\mathbf{z}$ directly from $q_{\phi}$ because the gradient with respect to the encoder parameters $\phi$ would not propagate through the random sampling process.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Reparameterization Trick)</span></p>

We express the latent variable $\mathbf{z}$ as a deterministic function of a parameter-free random variable $\boldsymbol{\epsilon}$:

$$\mathbf{z} = g(\phi, \boldsymbol{\epsilon}), \quad \boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$$

For a Gaussian encoder $\mathcal{N}(\mu_{\phi}, \sigma_{\phi}^2)$, this becomes:

$$\mathbf{z} = \mu_{\phi} + \sigma_{\phi} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stochastic Gradient Variational Bayes)</span></p>

This trick allows us to use standard gradient descent. We compute the gradient of the ELBO with respect to both $\phi$ (encoder parameters) and $\theta$ (latent and decoder model parameters) and maximize it. This procedure is known as **Stochastic Gradient Variational Bayes**.

</div>

### Multimodal Teacher Forcing

While the probabilistic framework is robust, it faces challenges common in dynamical systems reconstruction, such as exploding gradients or the inability to capture long-term statistical properties. Methods like sparse teacher forcing address these in deterministic settings; **Multimodal Teacher Forcing** aims to bring these advantages to the probabilistic regime.

#### Integrating Multiple Data Modalities

In complex real-world systems, observations often come in different forms. Multimodal Teacher Forcing is designed to handle systems where data of varying types (modalities) are observed simultaneously.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Climate Research Data)</span></p>

In climate science, a single dynamical system might produce:

* Continuous variables (e.g., temperature).
* Discrete events (e.g., occurrence of extreme weather events).
* Count data.

The model generalizes by utilizing multiple types of decoders (Gaussian, Poisson, etc.) tied to the same latent process $\mathbf{z}_t$, allowing the latent model to learn from diverse data streams simultaneously.

</div>

#### Multimodal Data Integration and Latent Space Parameterization

In real-world applications, such as medical monitoring or climate science, systems are often observed through multiple, simultaneous measurement modalities. We now explore how to unify these disparate data streams into a single latent model using probabilistic approaches.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multimodal Observation Vector)</span></p>

The multimodal observation vector $\mathbf{y}$ represents the concatenation of different data modalities observed at the same time step. If $\mathbf{x}$ represents Gaussian observations and $\mathbf{c}$ represents count observations, the combined vector is defined as:

$$\mathbf{y} = [\mathbf{x}, \mathbf{c}, \dots]^T$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Multimodal Data in Practice)</span></p>

* **Climate Science:** Simultaneous recordings of temperatures (modeled as continuous/Gaussian) and counts of extreme weather events like tornadoes (modeled as discrete/count data).
* **Medicine:** Combining continuous blood pressure readings with categorical lab results or binary (Bernoulli) indicators of symptom presence.

</div>

##### Parameterizing Distributions via Latent States

The core principle is that the parameters of various probability distributions are parameterized by latent states. This allows a single underlying dynamical model to govern multiple observation types.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Latent State Parameterization)</span></p>

In a probabilistic framework, the latent state $z_t$ determines the parameters of the observation distribution. For an observation $x_t$, the mapping is:

$$p(x_t \mid z_t) = \text{Dist}(\text{parameters} = f(z_t))$$

where $f$ can be a linear model, an exponential linear model, or a deep neural network.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Common Distribution Modalities)</span></p>

Depending on the data type, the following distributions are typically employed:

1. **Gaussian Observations:** Used for continuous numerical data. $p(x_t) = \mathcal{N}(x_t \mid \mu(z_t), \sigma^2(z_t))$
2. **Count Observations:** Used for discrete event frequencies, modeled via a Poisson distribution with parameter $\lambda$. $p(c_t) = \text{Poisson}(c_t \mid \lambda(z_t))$
3. **Categorical/Bernoulli Observations:** Used for class labels or binary flags where numerical information is absent.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Flexibility of the Mapping Function)</span></p>

The choice of the mapping function $f(z_t)$ (e.g., a neural network) is flexible. The critical requirement is that the latent state $z_t$ must capture enough information to reconstruct all modalities simultaneously, even if they come from different statistical families.

</div>

#### Architecture of the Multimodal Variational Autoencoder (MVAE)

To map high-dimensional multimodal observations into a lower-dimensional latent space, we employ an encoder model, typically structured as a neural network.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Variational Encoder)</span></p>

The encoder $q_\phi(z \mid \mathbf{y})$ is defined as a Gaussian distribution where the mean $\mu$ and covariance $\Sigma$ are functions of the input data, parameterized by a neural network with parameters $\phi$:

$$q_\phi(z \mid \mathbf{y}) = \mathcal{N}(z \mid \mu_\phi(\mathbf{y}), \Sigma_\phi(\mathbf{y}))$$

When parameterizing the covariance matrix $\Sigma$, the model must ensure that the diagonal entries are non-negative ($\geq 0$). This is often achieved using square operations or outer product forms.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(CNNs in the Time Domain)</span></p>

While deep neural networks (DNNs) can be used, **Temporal Convolutional Neural Networks** are often preferred for the encoder. In this context, the CNN treats the time series as a matrix where rows represent time steps and columns represent concatenated data modalities (Gaussian, counts, categorical). The CNN uses:

* **Receptive Fields:** Units that pay attention to localized segments of the spatial or temporal scene.
* **Kernels/Weights ($W$):** Shared matrices that perform convolutions to identify features (e.g., shapes in images or specific patterns in time series).
* **Pooling Operations:** Operations like "Max Pooling" that reduce dimensionality by selecting the maximum activation within a window, making the representation more compact.

</div>

#### Integration with Temporal Dynamics and Teacher Forcing

Training a variational autoencoder for dynamical systems requires balancing the encoder's output with the long-term statistics of the reconstruction model.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Encoder vs. Reconstruction Latent States)</span></p>

A key distinction must be made between two types of latent states:

* $\tilde{z}_t$: The latent state produced by the multimodal VAE encoder $q_\phi$ (derived directly from observations).
* $z_t$: The latent state of the reconstruction model (e.g., a recurrent neural network or a forward-propagating dynamical system).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Goal of Coupling)</span></p>

While $\tilde{z}$ and $z_t$ arise from different components of the architecture, they describe the same underlying system. We must force these states to agree so that $\tilde{z}$ can serve as a proxy for observations during teacher forcing, where the model's own predicted state is occasionally replaced by the encoded state from the data.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sparse Teacher Forcing)</span></p>

In this regime, the reconstruction model generally propagates forward from an initial condition. However, at specific intervals, its internal latent state $z_t$ is replaced by the encoder's latent state $\tilde{z}_t$ (the "pseudo-observation").

The MVAE framework proceeds as follows:

1. **Input:** Concatenated multimodal data $\mathbf{y}$.
2. **Encoding:** The MVAE (implemented via CNN/DNN) maps $\mathbf{y}$ to the distribution of $\tilde{z}$.
3. **Initialization:** The reconstruction model is initialized by $\tilde{z}_0$.
4. **Propagation:** The reconstruction model propagates forward in time to generate $z_t$.
5. **Alignment:** To ensure $z_t$ and $\tilde{z}_t$ represent the same underlying system, both are fed into the same set of decoder models.
6. **Decoding:** The decoders map both $z_t$ and $\tilde{z}_t$ back to the original data modalities (e.g., $\hat{\mathbf{y}}$), enforcing consistency between the encoder and the dynamical model.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Data Flow in the MVAE)</span></p>

The reconstruction model does not "see" the raw data $\mathbf{y}$ directly; it only interacts with the data through the teacher forcing signal provided by the encoder. This setup allows for training a probabilistic reconstruction model on multimodal data in a way that is more stable than standard VAE formulations.

</div>

#### The MVAE Encoder and Decoder Models

The architecture transitions from a standard VAE to a Multimodal VAE by handling different probability distributions at the output simultaneously.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(MVAE Encoder and Decoder Models)</span></p>

* **Encoder ($q$):** Maps multimodal data $Y$ to a latent model described by a probability distribution: $q_{\phi}(\tilde{z} \mid Y)$. In practice, this is often parameterized by a Convolutional Neural Network (CNN).
* **Decoders ($p_{\theta}$):** Map the latent states back to observation space. In a multimodal setting, we may use different distributions for different data types:
  1. **Gaussian Decoder** (for continuous data $x_t$): $p(x_t \mid \tilde{z}) = \mathcal{N}(B \tilde{z}, \Gamma)$, where $B$ is a linear mapping and $\Gamma$ is the covariance matrix.
  2. **Poisson Decoder** (for count data $c_t$): $p(c_t \mid \tilde{z}) = \text{Poisson}(\lambda)$, $\lambda = s(\tilde{z})$, where $s(\cdot)$ is a function (e.g., a neural network) mapping the latent state to the Poisson rate parameter.

</div>

#### The Variational Loss Function

To optimize the Multimodal VAE, we utilize the Evidence Lower Bound (ELBO). To frame this as an optimization problem for gradient descent, we minimize the negative ELBO.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multimodal VAE Loss)</span></p>

The loss function for the multimodal variational autoencoder is defined as:

$$\mathcal{L}_{\text{MVAE}} = - \mathbb{E}_{q} \left[ \ln p_{\theta}(Y \mid \tilde{z}) \right] - \ln p_{\theta}^{\text{latent}}(\tilde{z}) + H(q(\tilde{z} \mid Y))$$

Where:

* $\mathbb{E}_{q} [ \ln p_{\theta}(Y \mid \tilde{z}) ]$ represents the reconstruction log-likelihood.
* $p_{\theta}^{\text{latent}}(\tilde{z})$ is the prior over the latent states.
* $H(q)$ is the entropy of the encoder distribution.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Implementation Details)</span></p>

To propagate gradients through the stochastic latent variables, the reparameterization trick (defined earlier) is employed, allowing the expectation to be approximated via sampling.

</div>

#### Coupling with Reconstruction Models (RNNs)

The "Grand Architecture" involves coupling a reconstruction model (RNN) to the same decoders used by the VAE.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Shared Decoder Parameters)</span></p>

To ensure that the latent states $z_t$ (from the RNN) and $\tilde{z}$ (from the VAE) reside in the same manifold, the system enforces strict parameter sharing:

1. The RNN latent state $z_t$ is passed through the exact same Gaussian decoder: $p(x_t \mid z_t) = \mathcal{N}(B z_t, \Gamma)$
2. The RNN latent state $z_t$ is passed through the exact same Poisson decoder: $p(c_t \mid z_t) = \text{Poisson}(s(z_t))$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Use Shared Decoders?)</span></p>

By channeling both the VAE and the RNN through the same decoder models with identical parameters $B$, $\Gamma$, and the function $s(\cdot)$, we mathematically force $z_t$ to be close to $\tilde{z}$. This consistency is required for teacher forcing, where the latent state $z_t$ is replaced by $\tilde{z}$ at specific time steps to maintain stability during training.

</div>

#### Consistency and the Integrated Loss Framework

A critical innovation is treating the RNN's output as the prior for the VAE.

<details class="accordion" markdown="1">
<summary>Proof: Deriving the Latent Consistency Term</summary>

We assume the prior $p_{\theta}^{\text{latent}}(\tilde{z})$ is a Gaussian distribution whose mean is provided by the RNN state $z_t$. We approximate the expectation over the latent states using $L$ samples across $T$ time steps.

1. Start with the Log-Gaussian Prior: $- \ln p_{\theta}^{\text{latent}}(\tilde{z}) = - \ln \mathcal{N}(\tilde{z};\; z_t, \Sigma)$
2. Expand the Log-Probability: Ignoring constant terms ($2\pi$, etc.), the negative log-likelihood for $L$ samples and $T$ time steps becomes:

$$\sum_{l=1}^{L} \sum_{t=1}^{T} \left( \frac{1}{2} \ln |\Sigma| + \frac{1}{2} (\tilde{z}_t^{(l)} - z_t)^T \Sigma^{-1} (\tilde{z}_t^{(l)} - z_t) \right)$$

3. **Result:** This term penalizes the squared distance between the encoder's output $\tilde{z}$ and the RNN's prediction $z_t$, effectively enforcing direct consistency between the two components.

</details>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Total Multimodal Teacher Forcing Loss)</span></p>

The final objective function for end-to-end training combines the VAE loss and the reconstruction model loss:

$$\mathcal{L}_{\text{total}} = \underbrace{- \mathbb{E}_q [\ln p_{\theta}(Y \mid \tilde{z})] + H(q)}_{\text{Reconstruction/Entropy}} + \underbrace{- \mathbb{E}_q [\ln p_{\theta}^{\text{latent}}(\tilde{z} \mid z_t)]}_{\text{Consistency Prior}} + \underbrace{- \ln p(Y \mid z_t)}_{\text{RNN Decoder Loss}}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Complexity of Modern AI)</span></p>

Modern architectures are rarely single deep neural networks. Instead, they are "Grand Architectures"—compositions of standard components (CNN encoders, RNN transition models, Gaussian/Poisson decoders) integrated into a single differentiable framework optimized end-to-end.

</div>

### From Discrete-Time Models to Continuous-Time Neural Networks

As we transition from discrete-time models to continuous-time frameworks, we explore how we can reconstruct complex chaotic attractors and how the architecture of deep learning has evolved into the elegant domain of Neural Ordinary Differential Equations (Neural ODEs).

#### Reconstruction of Chaotic Attractors

Before diving into the architecture of neural networks, we must understand the power of dynamical reconstruction. Even when a system is highly obscured or simplified, its underlying topology and dynamics can often be recovered.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dealing with Noise in Observations)</span></p>

In practical applications, we rarely observe a pristine dynamical system. Often, the original signal—such as the trajectories of a Lorenz attractor—is obscured by significant noise, appearing as "squiggly lines" that bear little resemblance to the underlying butterfly shape. Even when the signal-to-noise ratio is poor, a system can be reconstructed if we incorporate additional observations. For example, in the Lorenz system, integrating "counts" or other secondary observations alongside the noisy data allows the model to identify the true underlying manifold.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Symbolic Lorenz Reconstruction)</span></p>

A more radical approach to reconstruction involves symbolic encoding. Instead of using continuous coordinates, we partition the state space into discrete "boxes":

1. **Discretization:** The entire state space of the Lorenz attractor is divided into regions (boxes).
2. **Binary Assignment:** Each box is assigned a unique binary code.
3. **Training:** A model is trained exclusively on these binary sequences, with no access to the original continuous coordinates.
4. **Result:** Post-training, the model can generate a reconstructed attractor that maintains the same topology and a similar Lyapunov exponent as the original system.

</div>

#### From Residual Networks to Neural ODEs

The transition to Neural ODEs is motivated by the desire to make deep neural networks continuous in depth. To understand this, we first look at the structure of a Residual Network (ResNet).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Residual Network (ResNet))</span></p>

A ResNet is a feed-forward architecture where activations at one layer are passed directly to subsequent layers, bypassing certain transformations. This "residual connection" helps combat the vanishing/exploding gradient problem.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(ResNet Discrete-Time Dynamics)</span></p>

If we denote $n \in \lbrace 1, \dots, N \rbrace$ as the layer index and $h_n$ as the activation at layer $n$, the activation at the next layer $h_{n+1}$ is defined by:

$$h_{n+1} = h_n + f(h_n, \theta_n, I_n)$$

Where:

* $h_n$ is the residual connection (the previous state added directly).
* $f(h_n, \theta_n, I_n)$ represents the nonlinear transformation (e.g., an affine mapping followed by a ReLU activation).
* $\theta_n$ represents the weights/parameters at layer $n$.
* $I_n$ represents potential inputs at that layer.

</div>

##### The Continuous Limit

By considering the step between layers as an infinitesimal increment, we can transform this discrete mapping into a continuous one.

<details class="accordion" markdown="1">
<summary>Proof: Transition to Neural ODEs</summary>

1. Let the change in activation between layers be $\Delta h = h_{n+1} - h_n$.
2. From our ResNet definition: $\Delta h = f(h_n, \theta_n, I_n)$.
3. Let $\Delta n$ represent the "step size" between layers. If we divide by $\Delta n$ and let $\Delta n \to 0$:

$$\lim_{\Delta n \to 0} \frac{h_{n+\Delta n} - h_n}{\Delta n} = \frac{dh}{dt}$$

4. This yields a continuous-time formulation where the processing across "layers" is represented by a differential equation:

$$\frac{dh}{dt} = f(h(t), t, \theta)$$

In this view, the depth of the network is analogous to the time dimension in an ODE.

</details>

#### Motivation for Continuous-Time Models

Why move from discrete layers to continuous differential equations? The motivations vary across disciplines.

* **Computational Neuroscience:** In neuroscience, biophysical quantities (like membrane potentials) naturally evolve in continuous time. Continuous-time recurrent networks are essential for modeling these systems accurately.
* **Machine Learning Efficiency:** Neural ODEs offer lower memory costs (no need to store all intermediate activations for backpropagation) and often allow for faster training compared to very deep discrete networks.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Problem of Irregular Sampling)</span></p>

Traditional discrete-time RNNs struggle with data that is not sampled at regular intervals. In a discrete RNN, you must "bin" your time steps, which leads to a loss of precision regarding the exact moment an observation occurred ($t_i$).

Neural ODEs are uniquely suited for **point processes**, where events occur at irregular intervals. Examples include:

* Spikes in the nervous system.
* Particles hitting a surface in a collider.
* Customers entering a shop.
* Extreme weather events like tornadoes.

Because Neural ODEs run in continuous time, they can simply integrate the state from one observation time point $t_i$ to the next $t_{i+1}$, regardless of the spacing between them.

</div>

#### Training Neural ODEs

To train these systems, we treat the state transitions as an integration problem.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Neural ODE State Integration)</span></p>

We define the state of the system as $z(t)$. To find the state at time $t_1$ given an initial state at $t_0$, we integrate the vector field $f$:

$$z(t_1) = z(t_0) + \int_{t_0}^{t_1} f(z(\tau), \tau, \theta)\, d\tau$$

where $f$ can be any differentiable function, such as a deep neural network or a convolutional neural network.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Loss Function)</span></p>

The loss function $L$ is calculated based on observations $\mathbf{x}$ at specific time points $t_i$. It is the sum of the differences between the observations and the decoded latent states $h(t_i)$:

$$L = \sum_{i} \text{Loss}(x(t_i), \text{decoder}(h(t_i)))$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Training via Adjoint Sensitivity)</span></p>

Standard gradient descent would require backpropagating through every operation of the numerical ODE solver. This is problematic because it is computationally expensive and restricts the choice of solver (especially for implicit solvers). To solve this, we treat the ODE solver as a black box and use the **Adjoint Sensitivity Method** (a technique dating back to the 1960s), which allows for a "reverse mode automatic differentiation" that is much more flexible and efficient.

</div>

### The Adjoint Sensitivity Method

In the study of dynamical systems, particularly when integrated with machine learning (such as Neural ODEs), we often need to compute the gradient of a loss function with respect to the parameters of a vector field. This process is essentially a continuous-time version of Backpropagation Through Time (BPTT).

Instead of unwrapping a discrete neural network into layers, we consider a state $z(t)$ that evolves according to a differential equation defined by a vector field $f_\theta(z, t)$. To perform gradient descent on the parameters $\theta$, we require the derivative of the loss $L$ with respect to $\theta$. This is achieved through the Adjoint Sensitivity Method.

#### The Adjoint State

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Adjoint State)</span></p>

The adjoint state $a(t)$ is defined as the partial derivative of the loss function $L$ with respect to the state $z$ at time $t$:

$$a(t) = \frac{\partial L}{\partial z(t)}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Physical Intuition)</span></p>

The adjoint can be thought of as an "error signal" that we propagate backward in time. Just as standard backpropagation uses the chain rule to send gradients from the output layer back to the input, the adjoint method uses a differential equation to move the gradient from the final time $t_1$ back to the initial time $t_0$.

</div>

#### Derivation of the Adjoint Differential Equation

To use the adjoint in a computational framework, we must define how it evolves over time. It turns out that $a(t)$ follows its own differential equation, which depends on the Jacobian of the system's vector field.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Adjoint ODE)</span></p>

The dynamics of the adjoint state $a(t)$ are governed by the following linear differential equation:

$$\frac{da(t)}{dt} = -a(t)^\top \frac{\partial f_\theta(z, t)}{\partial z}$$

where $\frac{\partial f_\theta}{\partial z}$ is the Jacobian of the vector field with respect to the state $z$.

</div>

<details class="accordion" markdown="1">
<summary>Proof: Derivation via Limit Definition</summary>

We derive this by considering the recursion of the state in a small time step $\epsilon$ and applying the chain rule, then taking the limit as $\epsilon \to 0$.

1. **State Evolution:** Assume we have a state at time $t + \epsilon$. By integrating the ODE $z' = f_\theta(z, t)$, we can approximate the state as:

$$z(t + \epsilon) = z(t) + \int_{t}^{t+\epsilon} f_\theta(z, t)\, dt \approx z(t) + \epsilon f_\theta(z(t), t)$$

We define this mapping as $T_\epsilon(z(t))$.

2. **Chain Rule Recursion:** In the spirit of backpropagation, the derivative of the loss with respect to the state at time $t$ depends on the state at $t + \epsilon$:

$$\frac{\partial L}{\partial z(t)} = \frac{\partial L}{\partial z(t + \epsilon)} \frac{\partial z(t + \epsilon)}{\partial z(t)}$$

Substituting the definition of the adjoint $a(t)$:

$$a(t) = a(t + \epsilon) \frac{\partial T_\epsilon(z(t))}{\partial z(t)}$$

3. **Applying the Derivative:** Using our Taylor approximation $T_\epsilon(z(t)) \approx z(t) + \epsilon f_\theta(z, t)$, we take the derivative with respect to $z(t)$:

$$\frac{\partial T_\epsilon(z(t))}{\partial z(t)} = I + \epsilon \frac{\partial f_\theta(z, t)}{\partial z} + O(\epsilon^2)$$

where $I$ is the identity matrix.

4. **Forming the Difference Quotient:**

$$a(t) = a(t + \epsilon) \left( I + \epsilon \frac{\partial f_\theta}{\partial z} + O(\epsilon^2) \right)$$

$$a(t) = a(t + \epsilon) + \epsilon\, a(t + \epsilon) \frac{\partial f_\theta}{\partial z} + O(\epsilon^2)$$

Rearranging to find the change in $a$ over time:

$$\frac{a(t + \epsilon) - a(t)}{\epsilon} = -a(t + \epsilon) \frac{\partial f_\theta}{\partial z} + O(\epsilon)$$

5. **Taking the Limit:** As $\epsilon \to 0$, the left side becomes the derivative $\frac{da}{dt}$, and $a(t + \epsilon)$ becomes $a(t)$:

$$\frac{da(t)}{dt} = -a(t) \frac{\partial f_\theta(z, t)}{\partial z}$$

The negative sign arises because we are looking at the change in the reverse direction relative to the standard derivative definition.

</details>

#### Gradient Computation and Augmented States

The primary goal is to find $\frac{dL}{d\theta}$ to update our parameters. This is calculated by integrating the relationship between the adjoint and the vector field's sensitivity to parameters over time.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Parameter Gradient)</span></p>

The derivative of the loss with respect to the parameters $\theta$ is given by the integral:

$$\frac{dL}{d\theta} = -\int_{t_1}^{t_0} a(t)^\top \frac{\partial f_\theta(z, t)}{\partial \theta}\, dt$$

This integration is performed backward in time from the final state $t_1$ to the initial state $t_0$, which is referred to as reverse mode differentiation.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Augmented State)</span></p>

To solve for the state $z$, the adjoint $a$, and the parameter gradients $\frac{dL}{d\theta}$ simultaneously, we construct an augmented state. This is a mathematical trick to treat parameters as part of the dynamical system:

$$z_{\text{aug}} = \begin{bmatrix} z \\ \theta \end{bmatrix}$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Augmented Vector Field)</span></p>

The augmented vector field $f_{\text{aug}}$ describes the dynamics of the augmented state:

$$f_{\text{aug}}(z, \theta, t) = \begin{bmatrix} f_\theta(z, t) \\ 0 \end{bmatrix}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dynamics of Parameters)</span></p>

In the augmented vector field, the second component is $0$ because the parameters $\theta$ do not depend on time ($\frac{d\theta}{dt} = 0$). By concatenating these, we can use a single ODE solver to integrate the entire system—states and adjoints—jointly across the time interval.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Augmented Adjoint)</span></p>

Just as we augmented the state, we can augment the adjoint to include $a_\theta$, which corresponds to $\frac{\partial L}{\partial \theta}$. The augmented adjoint $a_{\text{aug}}$ is:

$$a_{\text{aug}} = \begin{bmatrix} a \\ a_\theta \end{bmatrix}$$

By applying the adjoint ODE formula to the augmented system, we can derive the full set of differential equations required to solve the gradient descent step in one backward pass.

</div>

#### Formal Derivation: Augmented Vector Field Jacobian

The Adjoint Sensitivity Method allows us to compute gradients of a loss function with respect to the parameters of a differential equation without backpropagating through the internal steps of an ODE solver.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Augmented Vector Field Jacobian)</span></p>

To compute gradients in a continuous-time system, we consider an augmented vector field that incorporates both the state $z$ and the parameters $\theta$. The Jacobian of this augmented vector field with respect to the augmented states is defined as:

$$J = \begin{pmatrix} \frac{\partial f_\theta}{\partial z} & \frac{\partial f_\theta}{\partial \theta} \\ 0 & 0 \end{pmatrix}$$

where $f_\theta$ represents the dynamics of the system. The bottom row consists of zeros because the parameters $\theta$ are constant over time.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Total Loss Derivative via the Adjoint)</span></p>

The total derivative of the loss $L$ with respect to the parameters $\theta$ can be obtained by integrating the adjoint state $a(t)$ and the partial derivative of the vector field over the time interval $[t_0, t_1]$:

$$\frac{dL}{d\theta} = -\int_{t_1}^{t_0} a(t)^\top \frac{\partial f_\theta}{\partial \theta}\, dt$$

</div>

<details class="accordion" markdown="1">
<summary>Proof: Deriving the Adjoint Gradient</summary>

1. **Augmented State Dynamics:** Let our augmented state be $\tilde{z} = [z, \theta]$. The dynamics are given by $\frac{d\tilde{z}}{dt} = [f_\theta(z, t), 0]$.
2. **Applying the Jacobian:** We multiply the augmented adjoint vector by the Jacobian derived above:

$$\begin{pmatrix} a(t) & a_\theta(t) \end{pmatrix} \begin{pmatrix} \frac{\partial f_\theta}{\partial z} & \frac{\partial f_\theta}{\partial \theta} \\ 0 & 0 \end{pmatrix}$$

3. **Component Result:** This multiplication yields two components:
   * $a(t) \frac{\partial f_\theta}{\partial z}$: The original adjoint derivative for the state.
   * $a(t) \frac{\partial f_\theta}{\partial \theta}$: The term required to construct the gradient with respect to the parameters.
4. **Integration:** To find the total gradient at $t_0$, we propagate the adjoint back from $t_1$ to $t_0$:

$$\frac{dL}{d\theta} = - \int_{t_1}^{t_0} a(t) \frac{\partial f_\theta}{\partial \theta}\, dt$$

This result allows us to propagate gradients backwards through time in a mathematically rigorous and computationally efficient manner.

</details>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition on Continuous Time)</span></p>

When transitioning from discrete networks to continuous models, we must treat time as a continuous variable. This necessitates the use of integrals rather than discrete summations. By letting the time step $\epsilon \to 0$, we leverage the properties of differential equations to derive these gradients, which steered significant research in both machine learning and dynamical systems reconstruction.

</div>

### Neural Stochastic Differential Equations (Neural SDEs)

While Neural ODEs describe deterministic processes, many real-world systems are subject to noise. **Neural Stochastic Differential Equations (SDEs)** extend the Neural ODE framework to include stochasticity, effectively serving as a continuous-time analog to Recurrent Neural Networks (RNNs) with noise terms.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stochastic Differential Equation (SDE))</span></p>

A **Stochastic Differential Equation** with Gaussian input is defined by two primary components: a deterministic drift term and a stochastic diffusion term.

$$dh_t = f(h_t, t, \theta)\, dt + g(h_t, t, \phi)\, dW_t$$

* $f(h_t, t, \theta)$: The **Drift Term**, representing the deterministic part of the dynamics.
* $g(h_t, t, \phi)$: The **Diffusion Term**, representing the scaling of the noise process.
* $dW_t$: The **Wiener Process** (or Brownian motion).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Wiener Process)</span></p>

The **Wiener Process** $W_t$ is characterized by the following properties:

1. **Independent Increments:** The differences between terms at different times, $W_{t + \Delta t} - W_t$, are independent.
2. **Normal Distribution:** The increments follow a normal distribution with zero mean and variance equal to the time lag: $W_{t + \Delta t} - W_t \sim \mathcal{N}(0, \Delta t)$.
3. **Continuous Limit:** It is viewed as the limit of a Gaussian process as the time step $\Delta t \to 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Neural Parameterization)</span></p>

In the context of "Neural" SDEs, both the drift term $f$ and the diffusion term $g$ are parameterized using Deep Neural Networks. This allows the model to learn complex, non-linear deterministic trajectories while simultaneously learning the structure of the noise or uncertainty in the system.

</div>

### Physics-Informed Neural Networks (PINNs) and Hybrid Modeling

The core objective of Physics-Informed Neural Networks (PINNs) and hybrid models is to balance the flexibility of "black-box" deep learning with the rigorous constraints of known physical laws.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hybrid Drift Formulation)</span></p>

In a hybrid model, the drift term of a differential equation is augmented to include both a learned neural component and a term representing prior physical knowledge:

$$\frac{dh}{dt} = f_\theta(h, t) + \gamma K_\psi(h, t)$$

* $f_\theta$: A black-box deep neural network.
* $K_\psi$: A function embedding prior physical knowledge (e.g., known differential equations).
* $\gamma$: A weighting constant or matrix (often diagonal) that determines the trade-off between the neural approximation and the physical prior.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Lorenz 63 System as a Prior)</span></p>

If we possess prior knowledge that a system behaves similarly to the chaotic Lorenz 63 attractor, we can define $K_\psi$ using the Lorenz equations:

$$K = \begin{pmatrix} \sigma(h_2 - h_1) \\ h_1(\rho - h_3) - h_2 \\ h_1 h_2 - \beta h_3 \end{pmatrix}$$

By injecting these equations into the learning process, the model is constrained to respect the known dynamics of the Lorenz system while the neural network $f_\theta$ accounts for residuals or deviations from this idealized model.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Versatility of PINNs)</span></p>

The PINN framework is highly flexible and can be extended to:

* Partial Differential Equations (PDEs).
* Stochastic Differential Equations (SDEs).
* Decoding Techniques in fields like computational neuroscience, where Neural ODEs are used to describe underlying processes giving rise to observed data, such as single spike recordings.

</div>

#### Spatiotemporal Dynamical Systems

In many physical contexts, such as astrophysics or atmospheric science, we define a dynamical variable $u$ that depends on both time $t$ and space $x$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Spatiotemporal PDE)</span></p>

A general partial differential equation for a variable $u(t, x)$ can be expressed as:

$$\frac{\partial u}{\partial t} = g\left(u(t, x),\; t,\; x,\; \frac{\partial u}{\partial x},\; \frac{\partial^2 u}{\partial x^2},\; \dots \right)$$

Where:

* $u(t, x)$ is the state variable (e.g., temperature, velocity).
* $t$ denotes time.
* $x$ denotes spatial coordinates.
* $g(\cdot)$ is a function representing the physics of the system, potentially including higher-order spatial derivatives.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Combining Physics and Neural Networks)</span></p>

The goal is to combine our prior "physics knowledge" (expressed via the function $g$) with the flexible approximation capabilities of deep neural networks. This allows us to reconstruct the system even when our observations are sparse or noisy.

</div>

#### Parameterization via Deep Neural Networks

In the PINN framework, we do not use a traditional numerical solver. Instead, we parameterize the dynamical variable $u$ directly using a deep neural network.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(DNN Parameterization)</span></p>

We approximate the unknown solution $u(t, x)$ with a deep neural network $u_{\theta}(t, x)$, where $\theta$ represents the trainable weights of the network:

$$u(t, x) \approx u_{\theta}(t, x)$$

This neural network takes time $t$ and spatial coordinates $x$ as inputs and outputs the estimated state at those coordinates.

</div>

#### The Physics-Informed Loss Function

To train the network $u_{\theta}$, we construct a loss function that penalizes both deviations from empirical data and deviations from the known physical laws (the PDE).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The PINN Loss Function)</span></p>

Given $N$ observations $\tilde{u}$ at various points in time $t_i$ and space $x_i$, the total loss function $\mathcal{L}$ is defined as the sum of a data-driven loss and a physics-informed penalty:

$$\mathcal{L} = \sum_{i=1}^{N} \left\| \tilde{u}(t_i, x_i) - u_{\theta}(t_i, x_i) \right\|^2 + \lambda \mathcal{L}_{\text{physics}}$$

Where:

* The first term is the Mean Squared Error (MSE) between true observations and the network's predictions.
* Note that observations $(t_i, x_i)$ do not need to be regularly spaced.
* $\mathcal{L}_{\text{physics}}$ is the penalty for violating the underlying differential equation.

</div>

<details class="accordion" markdown="1">
<summary>Proof: Construction of the Physics Penalty</summary>

To ensure the neural network adheres to the physics model $G$ (for example, the Lorenz equations or a specific fluid dynamics model), we define the residual of the PDE.

1. Compute the temporal derivative of the neural network output $\frac{\partial u_{\theta}}{\partial t}$ using automatic differentiation (similar to the method used in Neural ODEs).
2. The physics-informed term $\mathcal{L}_{\text{physics}}$ minimizes the difference between this derivative and the prescribed vector field:

$$\mathcal{L}_{\text{physics}} = \left\| \frac{\partial u_{\theta}}{\partial t} - G(u_{\theta}, t, x, \dots) \right\|^2$$

3. By minimizing $\mathcal{L}$, the network simultaneously learns to fit the data and satisfy the governing physical equations.

</details>

#### Numerical Implementation and Training Challenges

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Training Stability)</span></p>

PINNs are notoriously tedious to train. Unlike standard supervised learning, the optimization landscape for PINNs is complex, and the models often diverge during the training process. In many practical scenarios, Recurrent Neural Networks (RNNs) may actually be easier to train and more stable, despite lacking the explicit physics prior of a PINN.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Applications of PINNs)</span></p>

* **Turbulence:** Modeling fluid flow in gas or water tanks.
* **Quantum Physics:** Describing wave functions and particle dynamics.
* **Astrophysics:** Modeling large-scale cosmic structures.

</div>

## Lecture 14

### Out-of-Domain Generalization in Dynamical Systems Reconstruction

In the context of artificial intelligence and machine learning, generalization is the "holy grail." However, the definition of what it means to "generalize" differs significantly between classical statistical learning and the study of dynamical systems.

#### Characterizing Generalization

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Distribution Shift)</span></p>

In classical machine learning, generalization is typically defined in the context of a **distribution shift**. We assume a training data distribution $P_{\text{train}}$ and a test or inference distribution $P_{\text{test}}$. Generalization is the ability of a model to perform accurately when:

$$P_{\text{train}} \neq P_{\text{test}}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Clinical Diagnostics)</span></p>

Consider a classifier trained to differentiate healthy subjects from patients using physiological measures (e.g., ECG, blood tests) collected across a specific set of hospitals. If this model is applied to a new set of hospitals in a different country, the underlying data distribution may shift due to demographic or environmental factors. If the original classifier fails to adapt to this new location in the feature space, it has failed to generalize across the distribution shift.

</div>

#### Generalization within a Single Basin

When reconstructing dynamical systems—using models such as Recurrent Neural Networks (RNNs), Neural Ordinary Differential Equations (NODEs), or Reservoir Computers—we often seek to reconstruct a chaotic attractor from limited data.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Rössler System)</span></p>

The Rössler system is a 3D chaotic attractor, often viewed as a simplification of the Lorenz system. In a typical reconstruction task, we may only have access to a single, short trajectory starting from a specific initial condition $x_0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Ergodic Distribution and Generalization)</span></p>

It is often asked whether training a model on one trajectory and testing it on another (from a different initial condition) constitutes "out-of-distribution" generalization.

In dynamical systems, if the trajectories converge to the same limit set (the chaotic attractor), they share the same **ergodic distribution**. Therefore, generalizing to nearby initial conditions within the same basin of attraction is generally not considered out-of-distribution generalization in the classical machine learning sense, as the long-term statistics of the trajectories are identical.

</div>

#### The Frontiers: Out-of-Domain Generalization

The "Holy Grail" of dynamical systems reconstruction is **Out-of-Domain (OOD) Generalization**. This refers to the ability of a model to predict the behavior of a system in regimes or regions of the state space that were never observed during training.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Out-of-Domain Generalization)</span></p>

In the context of dynamical systems, OOD generalization is the capacity to:

1. Generalize to different dynamical regimes (e.g., moving from a fixed point to a limit cycle).
2. Predict behavior in unobserved basins of attraction in a multi-stable system.
3. Generalize across bifurcation points where the qualitative behavior of the system changes fundamentally due to a parameter shift.

</div>

#### Multi-stability and Bifurcations

Most complex real-world systems—such as the climate, the human brain, or ecosystems—are inherently multi-stable. They possess multiple stable regimes that the system may transition between due to noise or external perturbations.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Subcritical Hopf Bifurcation)</span></p>

Consider a system undergoing a subcritical Hopf bifurcation. In such a system, we may observe:

* A stable equilibrium (point attractor).
* As a control parameter changes, the equilibrium becomes an unstable spiral.
* A stable limit cycle may coexist with a stable fixed point, separated by an unstable limit cycle.

If we only observe data from the stable equilibrium regime, the challenge is to determine if a model can predict the existence of the stable limit cycle that appears after the bifurcation.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Spiking Neuron Model)</span></p>

A 2D spiking neuron model can exhibit bi-stability, where a stable point attractor (resting state) and a stable limit cycle (spiking state) coexist in the vector field.

* If a model (like an RNN) is trained only on trajectories converging to the limit cycle (the spiking regime), it often fails to reconstruct the vector field in the region of the point attractor.
* This failure highlights the difficulty of reconstructing the global topology of the state space from local observations.

</div>

#### Theoretical Limits of Reconstruction

While OOD generalization is a primary goal, there are formal mathematical constraints on what can be achieved through data-driven reconstruction.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Impossibility of General Out-of-Domain Generalization)</span></p>

Formulated in research by Nicholas Goring (ICML), it can be shown that in the most general scenario, OOD generalization is mathematically impossible without additional structural assumptions.

</div>

<details class="accordion" markdown="1">
<summary>Proof: Intuition via Bump Functions</summary>

The proof relies on the fact that one can mathematically construct a vector field using bump functions (e.g., Gaussian-like functions that taper to zero at the boundaries of a basin).

1. Define a system where each basin of attraction is governed by its own independent set of differential equations.
2. Use a bump function to ensure that the dynamics of System A are only active within Basin A and decay to zero at the boundary.
3. Define System B similarly for Basin B.
4. Because the dynamics in Basin A contain zero information about the functional form of the dynamics in Basin B, no model trained exclusively on data from Basin A can possibly predict the behavior in Basin B.

</details>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Implications for Scientific Theory)</span></p>

Any good scientific theory is expected to make predictions about unobserved regimes. Therefore, while pure data-driven reconstruction faces these "impossibility" hurdles, the integration of physical constraints or prior knowledge is essential to achieve the level of generalization required for robust scientific modeling.

</div>

### Generalization and the Problem of Multiple Basins

In classical machine learning, generalization refers to the ability of a model to perform well on unseen data from the same distribution. In dynamical systems, however, we encounter the problem of out-of-domain generalization, where a model trained on one region of the state space must predict dynamics in a completely different region—specifically, in a different basin of attraction.

#### The Limits of Out-of-Domain Generalization

When a dynamical system possesses multiple stable states, the vector field may behave differently in each basin. Standard architectures like Reservoir Computers or Neural Ordinary Differential Equations (Neural ODEs) often fail to generalize across these boundaries.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Bistable Duffing Oscillator)</span></p>

The bistable Duffing oscillator is a 2D system of differential equations characterized by having two attracting spiral points. Each point has its own basin of attraction.

* If a model is trained exclusively on trajectories from one basin, it typically fails to capture the dynamics of the second basin.
* Even if a model has successfully learned both basins, retraining it on data from only one basin often leads to "unlearning" the dynamics of the other.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Different Equations per Basin)</span></p>

Intuitively, one might think of a system with multiple basins as being governed by different sets of equations in each region. While mathematically we can construct a vector field where this is true, real-world systems are typically governed by a single, global set of equations (e.g., a specific physical law). The challenge is whether a data-driven model can recover that global law from local observations.

</div>

#### Learning Global Dynamics via SINDy

Sparse Identification of Nonlinear Dynamics (SINDy) offers a potential solution to the generalization problem if certain conditions are met.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Global Reconstruction from Local Trajectories)</span></p>

Assume a vector field is described by a global set of equations $\dot{x} = f(x)$, where $f(x)$ can be represented as a linear combination of basis functions:

$$f(x) \approx \sum_{i} \alpha_i \phi_i(x)$$

If the library of basis functions $\lbrace \phi_i \rbrace$ contains the true functional forms of the system (a "physical prior"), then the coefficients $\alpha_i$ learned from trajectories in a single basin of attraction can correctly describe the dynamics in all other basins.

**Conditions for Failure:**

1. **Incomplete Library:** If the basis expansion does not contain the necessary terms, the model cannot reconstruct the global field.
2. **Singular Matrices:** Certain trajectories (e.g., specific limit cycles) may result in data matrices that are singular, preventing the identification of a unique solution for the coefficients.

</div>

#### Loss Landscapes and Unlearning

The empirical behavior of models during training reveals significant differences between dynamical systems and standard machine learning tasks.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Broad vs. Narrow Minima)</span></p>

In traditional machine learning, "broad" minima in the loss landscape are generally associated with better generalization. In the context of learning dynamical systems across basins, the opposite is often observed: the most generalizing solutions correspond to **narrower minima**.

**The Phenomenon of Unlearning:** If a model that has already learned the global dynamics (both basins) is retrained on data from only one basin, the out-of-domain performance degrades significantly.

* The generalizing solution (the global minimum) becomes a spurious minimum in the presence of restricted data.
* As training continues on the single-basin data, the model's loss landscape shifts, causing it to lose the parameters that defined the dynamics in the unobserved basin.

</div>

### Tipping Points and Topological Shifts

Predicting sudden, drastic changes in system behavior is a critical objective in dynamical systems theory, particularly when those changes involve "tipping points."

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Tipping Point)</span></p>

A **tipping point** is a sudden change in a system's behavior characterized by a topological shift. In this context, the vector field before the shift is not topologically equivalent to the vector field after the shift.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Tipping as a Bifurcation)</span></p>

A tipping point often corresponds to a qualitative shift in the system's stability or the creation/destruction of attractors. This is mathematically defined as a **bifurcation**, where a change in a control parameter crosses a critical threshold, altering the system's phase portrait.

</div>

Tipping points are highly relevant in several fields:

* **Climate Change:** Identifying the threshold at which a subsystem (like an ice sheet) shifts to a new, potentially irreversible state.
* **Medicine (Sepsis):** In clinical settings, a patient may transition from a healthy state to a state of exponential bacterial growth and fever. This shift acts like a transition to a different attractor state once a "bacterial load" threshold is crossed.

#### B-Tipping: Bifurcation-Induced Transitions

The major class of tipping points is known as B-tipping, which is directly related to bifurcations caused by slowly changing parameters. To understand B-tipping, we look at the normal form of a bifurcation. Consider a supercritical Hopf bifurcation, which describes the transition from a stable spiral point to a stable limit cycle.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Normal Form — Polar Coordinates)</span></p>

The dynamics of a supercritical Hopf bifurcation can be expressed as:

$$\dot{r} = \mu r - r^3, \qquad \dot{\theta} = \omega$$

where:

* $r$ is the radius from the equilibrium point.
* $\mu$ is the control parameter.
* $\omega$ is the constant angular velocity.

</div>

In B-tipping, we assume that the control parameter $\mu$ is not constant but changes slowly over time: $\mu = \mu(t)$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Vibration Example)</span></p>

Consider an aircraft wing. Initially, the system sits at a stable equilibrium (the wing is steady). As a parameter $\mu$ (such as airspeed) slowly increases, the system may undergo a bifurcation.

1. At $\mu < 0$, the equilibrium is a stable spiral point.
2. Small, sub-threshold vibrations may occur but die out.
3. As $\mu$ crosses $0$, the equilibrium becomes unstable, and the system "tips" into a limit cycle, resulting in large-scale, sustained oscillations (flutter).

This transition represents B-tipping: the system is pushed into a new attractor object (the limit cycle) due to the slow drift of the control parameter.

</div>

### Classification of Tipping Points

In the study of complex dynamical systems, a tipping point refers to a critical threshold at which a small change in an input or parameter results in a disproportionately large change in the state of the system. While the previous discussion focused on bifurcation-induced tipping, there are at least three distinct mechanisms through which tipping can occur.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(B-Tipping — Bifurcation-Induced)</span></p>

**B-Tipping** occurs when a control parameter drifts slowly across a critical value, causing a qualitative change in the system's stability (a bifurcation), such as the disappearance of a stable fixed point or the emergence of a new attractor.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(N-Tipping — Noise-Induced)</span></p>

**N-Tipping** occurs in multi-stable systems when stochastic fluctuations (noise) provide enough energy to push the system state across the boundary of a basin of attraction and into the regime of a different attractor. This occurs even when the underlying parameters of the system remain constant.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(R-Tipping — Rate-Induced)</span></p>

**R-Tipping** occurs when a control parameter changes at a rate that is too fast for the system to track its internal state. Even if the system would remain stable under any fixed value of the parameter, the speed of the transition causes the system to "slip" out of its current basin of attraction.

</div>

#### Mathematical Analysis of Rate-Induced Tipping

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Intuition of Potential Landscapes)</span></p>

To visualize R-tipping, imagine a system state residing in a minimum of a potential landscape. As a control parameter changes, it pulls the entire landscape. If the landscape moves slowly, the system state "rolls" along the moving minimum, effectively tracking the attractor. However, if the rate of movement exceeds the system's ability to relax toward the minimum, the state may be left behind or pulled over a ridge into a different basin of attraction.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Quadratic Rate-Induced Model)</span></p>

Consider a system described by a quadratic differential equation where the control parameter $\lambda$ varies at a constant rate $r$:

$$\dot{x} = -(x + \lambda)^2 + \mu, \qquad \dot{\lambda} = r$$

To analyze whether the system tracks the attractor or undergoes R-tipping, we introduce a new coordinate $y$ that moves with the parameter: $y = x + \lambda$.

</div>

<details class="accordion" markdown="1">
<summary>Proof: Stability Conditions for Tracking</summary>

1. **Differentiate the new variable:** Compute the derivative of $y$ with respect to time:

   $$\dot{y} = \dot{x} + \dot{\lambda}$$

2. **Substitute the system equations:**

   $$\dot{y} = [-(x + \lambda)^2 + \mu] + r$$

   Since $y = x + \lambda$, we substitute:

   $$\dot{y} = -y^2 + \mu + r$$

3. **Identify fixed points:** To find the equilibrium where the system successfully tracks the moving parameter, set $\dot{y} = 0$:

   $$0 = -y^2 + \mu + r \implies y^2 = \mu + r \implies y^* = \pm \sqrt{\mu + r}$$

4. **Analyze existence and tipping:** The existence of these fixed points depends on the relationship between the system's internal parameters ($\mu$) and the rate of change ($r$).
   * In the original context of the quadratic form $\dot{x} = (x + \lambda)^2 - \mu$ (as used in the derivation), the fixed point locations are $y = \pm \sqrt{\mu - r}$.
   * If $r > \mu$, the term under the square root becomes negative, and the fixed points disappear.
   * **Conclusion:** If the rate of change $r$ exceeds the critical threshold $\mu$, the system can no longer maintain a stable tracking state and will be pulled out of the basin of attraction.

</details>

#### Non-Autonomous Dynamical Systems

The scenarios described above—where parameters are time-dependent or external forcing is present—fall under the category of non-autonomous systems.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Non-Autonomous System)</span></p>

A system is **non-autonomous** if the vector field depends explicitly on time:

$$\dot{x} = f(x, t)$$

In these systems, the evolution of the state depends not only on the initial state $x_0$ and the duration of time elapsed but also on the specific initial time $t_0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Autonomous Transformation Trick)</span></p>

A non-autonomous system in $n$ dimensions can formally be rewritten as an autonomous system in $n+1$ dimensions by introducing a new state variable $x_{n+1}$ to represent time: $\dot{x}_{n+1} = 1$. While this trick makes the system "formally" autonomous, it often complicates analysis because the system never reaches a standard attractor; the state is constantly drifting along the time dimension.

</div>

#### The Theory of Pullback Attractors

Standard attractor theory (looking at $t \to \infty$) is often insufficient for non-autonomous systems because the "landscape" is constantly shifting. Instead, we use the concept of a pullback attractor.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Pullback Attractor)</span></p>

A **pullback attractor** is a set of states $A(t)$ defined by looking into the infinite past. We ask: "What is the set of states that converges to the current time $t$ if we started the system infinitely far back in time?" Formally, we evaluate the limit as the starting time $t_0$ approaches negative infinity:

$$\lim_{t_0 \to -\infty} \phi(t, t_0, x_0)$$

where $\phi$ is the flow of the system. This provides a snapshot of the attractor's structure at a specific time $t$.

</div>

### Modeling Approaches for Post-Tipping Dynamics

To predict how a system behaves after a tipping point (particularly B-tipping), researchers utilize specific architectural priors in their models.

#### Skew Product Systems

Skew product systems are mathematical constructs that assume a clear separation of time scales between the "driving" force and the "dependent" system.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Skew Product System)</span></p>

The system is partitioned into two coupled sets of equations:

1. **Driving System:** A slow-moving system that governs the parameters: $\dot{y} = \epsilon\, g(y)$
2. **Dependent System:** The system of interest, driven by the state of $y$: $\dot{x} = f(x, y)$

where $\epsilon$ is a small magnitude parameter ensuring $y$ changes slowly relative to $x$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Application in Machine Learning)</span></p>

This structure can be used as a structural prior in reconstruction algorithms like Recurrent Neural Networks (RNNs). For example, in Piecewise Linear Recurrent Neural Networks (PLRNNs), one can encourage the model to learn the underlying "driving variables" by regularizing the system to follow a slow manifold. This allows the model to capture the non-autonomous nature of the data and potentially predict post-tipping dynamics by identifying the hidden parameters that drive the system toward a bifurcation.

</div>

#### Subspace Regularization and Manifold Attractors

One approach to capturing drifting dynamics is to incorporate a prior into our loss function that encourages the emergence of a manifold attractor—a continuous sheet of stable fixed points.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Regularized Loss Function)</span></p>

The objective function is modified by adding a regularization term to the standard Mean Squared Error (MSE):

$$\mathcal{L} = \text{MSE} + \gamma \cdot \mathcal{L}_{\text{reg}}$$

where $\gamma$ is a regularization parameter and $\mathcal{L}_{\text{reg}}$ penalizes deviations from a desired subspace structure.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Goal of Subspace Regularization)</span></p>

The goal is to drive a specific subsystem to act as the "driving" system. By regularizing a subspace of the model (for example, in an $m$-dimensional state space), we can force the system to learn its own "slow" control parameters or time constants without making rigid assumptions about their precise values.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Piecewise Linear Model Regularization)</span></p>

In a system defined by matrices $\mathbf{A}$, $\mathbf{W}$, and $\mathbf{H}$, we can split the matrices to isolate a non-regularized subspace.

* The diagonal elements of the $\mathbf{A}$ matrix for the regularized states are driven towards $1$.
* The corresponding terms in $\mathbf{W}$ and $\mathbf{H}$ are pushed towards $0$.
* This creates a subsystem that does not receive input from the rest of the states but provides input to them, effectively acting as an internal controller.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Biophysical Bursting Neuron)</span></p>

A system trained on a trajectory of a biophysical bursting neuron using this regularization was able to track the system as a control parameter changed slowly. The model successfully followed the system through a bifurcation, moving from a complex bursting regime into a simple regular spiking limit cycle.

</div>

#### Explicit Time-Dependent Parameterization

An alternative to implicit regularization is to explicitly define the parameters of the system as functions of time.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Time-Dependent Parameters)</span></p>

In this framework, parameters such as the bias term $\mathbf{h}_t$ or weight matrices $\mathbf{W}$ are not constant but are defined by a function $f(t)$:

$$\mathbf{h}_t = f(t)$$

where $f$ can be a simple linear mapping, an affine form ($\mathbf{a} + \mathbf{b}t$), or a complex Multi-Layer Perceptron (MLP).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Expressiveness of Time-Dependent Models)</span></p>

By making the parameters explicit functions of time, we allow the model to adapt its internal dynamics to non-stationary data. If $f(t)$ is represented by a deep neural network, the model becomes highly expressive, capable of mapping diverse temporal trajectories to parameter shifts. The entire system, including the mapping from time to parameters and the recurrent dynamics, is trained end-to-end.

</div>

### Early Warning Signs of Tipping Points

While full dynamical reconstruction is a primary goal, a more modest but critical task is the detection of **Early Warning Signs (EWS)** for tipping points (bifurcations). The detection of tipping points often relies on the principle of Critical Slowing Down.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Critical Slowing Down)</span></p>

As a system approaches a local bifurcation (such as a saddle-node bifurcation), the recovery rate from perturbations decreases. Mathematically, as the system nears the bifurcation point, the magnitude of the dominant eigenvalue of the Jacobian approaches zero, causing the flow of the vector field to become increasingly slow.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The "Tunnel" Effect)</span></p>

Imagine a 2D system where the nullclines of two variables are nearly touching. This creates a "tunnel" in the vector field. As the system moves closer to a saddle-node bifurcation, the flow through this tunnel slows down.

If we introduce noise into such a system, the slow dynamics fail to push the state back to the attractor quickly. This results in the state "wiggling" or drifting more freely along the direction of the bifurcation.

</div>

<details class="accordion" markdown="1">
<summary>Proof: Increase in Variance</summary>

1. Let a system be governed by $\dot{x} = f(x, \mu) + \sigma \eta(t)$, where $\eta(t)$ is noise.
2. Near a bifurcation, the "restoring force" (the gradient of the potential, if one exists) becomes nearly flat.
3. Because the dynamics are "friction-free" or nearly flat in the direction of the bifurcation, any stochastic perturbation $\sigma \eta(t)$ is uncountered by the deterministic flow.
4. Consequently, the fluctuations of the system state $x$ around the equilibrium increase.
5. Therefore, a dramatic increase in variance serves as a primary signature (EWS) that a tipping point is ahead.

</details>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Earthquake Detection)</span></p>

Earthquake detection systems utilize these signatures. By monitoring the increase in variance in seismic signals, researchers attempt to predict the proximity of a critical transition (the earthquake) based on these characteristic features of bifurcations.

</div>

### Foundation Models for Dynamical Systems

A recent development in the field (circa 2024–2025) is the application of **Foundation Models** to dynamical systems.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Foundation Model)</span></p>

A **Foundation Model** is a large-scale model pre-trained on a vast corpus of data, designed to be adaptable to a wide range of downstream tasks. In the context of dynamical systems, these models are trained on diverse forms of dynamics to learn generalizable representations.

</div>

#### Coupled Training Framework

In the study of complex systems, we often encounter multiple, distinct subsystems that share underlying structural similarities. Rather than training isolated models for each, we can utilize a coupled training framework to leverage shared information.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(System-Specific Parameters and Observations)</span></p>

Consider a set of $K$ different dynamical subsystems. For each subsystem $k \in \lbrace 1, \dots, K \rbrace$, we define:

* **Latent State Equation:** The evolution of the latent state $z_t^{(k)}$ is governed by parameters specific to that system, such as $A_k$ (system matrix), $W_k$ (weight matrix), and $H_k$.
* **Observation Function:** A decoder or observation function $G$ with parameters $\theta_{\text{obs}}^{(k)}$ that maps the latent state onto the observed state $x_t^{(k)}$:

$$x_t^{(k)} = G(z_t^{(k)};\; \theta_{\text{obs}}^{(k)})$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Coupling Strategy)</span></p>

While each subsystem could theoretically be trained independently, coupled training imposes restrictions on the parameters. This forces the models to share knowledge across the different systems, treating the collection not as $K$ independent problems, but as a unified learning task.

</div>

#### Transfer Learning and Universal Properties

The motivation for coupling the training of multiple dynamical systems lies in the concept of transfer learning. Dynamical systems across disparate fields—from quantum physics and astrophysics to biology and sociology—exhibit **universal properties**. There are only a finite set of possible attractor states and specific classes of bifurcations. By sharing parameters, a model can learn these universal "building blocks" of dynamics from one system and apply that knowledge to another.

#### Parameterization via Hypernetworks and Feature Vectors

To implement coupled training, we decouple the high-dimensional system parameters from the unique identity of the subsystem. This is achieved using a hypernetwork and low-dimensional feature vectors.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hypernetwork Parameterization)</span></p>

We define a function $f$, parameterized by group-level parameters $\theta_{\text{group}}$, that computes the model-specific parameters $\theta_k$ from a low-dimensional feature vector $L_k$:

$$\theta_k = f(L_k;\; \theta_{\text{group}})$$

* $L_k$: A low-dimensional feature vector representing system-specific characteristics. The dimension of $L_k$ is typically much smaller than the number of parameters in $\theta_k$.
* $\theta_{\text{group}}$: Global parameters shared across all subsystems.
* **Hypernetwork** ($f$): This can be a simple affine/linear mapping or a complex Multi-Layer Perceptron (MLP).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Parameter Unwrapping)</span></p>

If we define $\theta_k$ by unwrapping the system matrices into a single vector (concatenating the diagonal of $A$, then $W_1, W_2, \dots$), the hypernetwork $f$ transforms the low-dimensional feature $L_k$ into this expanded parameter space. In a simple case, a scalar feature $L_k$ could scale an entire parameter vector, though more complex mappings are usually required.

</div>

#### Loss Functions and Optimization Strategies

The training of these coupled systems typically employs techniques like sparse or generalized teacher forcing.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Coupled Loss Function)</span></p>

The objective is to minimize a loss function across all $K$ observed systems and all time points $T_k$. This is often formulated as an augmented Mean Square Error (MSE) or a Gaussian log-likelihood:

$$\mathcal{L} = \sum_{k=1}^{K} \sum_{t=1}^{T_k} \left[ \frac{1}{2} \ln |\Sigma| + \frac{1}{2} (x_t^{(k)} - \hat{x}_t^{(k)})^\top \Sigma^{-1} (x_t^{(k)} - \hat{x}_t^{(k)}) \right]$$

where $x_t^{(k)}$ is the observed state, $\hat{x}_t^{(k)}$ is the predicted state from the model, and $\Sigma$ is a co-trained covariance matrix.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Role of $\Sigma$)</span></p>

The covariance matrix $\Sigma$ is crucial when training on systems with very different geometries or scales. By co-training $\Sigma$, the system automatically learns to provide similar scaling for all subsystems, allowing the optimizer to handle diverse dynamical regimes within a single framework.

</div>

#### Empirical Results: Control Parameters and System Behavior

Applying hypernetworks to benchmarks like the Lorenz, Rössler, and Lorenz 96 systems reveals a profound emergence of structure in the feature space.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Discovery of Control Parameters)</span></p>

A significant result of this architecture is that the low-dimensional features $L_k$ often learn to correspond directly to the physical control parameters of the underlying system. For example, in a Lorenz system where the parameter $\rho$ (Rayleigh number) is varied, the network's extracted features often show a linear relationship with the ground truth $\rho$. This suggests the model is identifying the fundamental "knobs" that govern the dynamics.

**Extrapolation and Fine-Tuning:** Coupled training allows for high efficiency in new regimes:

* **Data Efficiency:** While a standard model might require thousands of time points to learn a new regime, a pre-trained coupled model may only require 10 to 200 observations.
* **Fine-Tuning:** One only needs to optimize the low-dimensional feature value $L$ for the new observations to achieve accurate predictions.

</div>

#### Applications in Time Series Classification and Neuroimaging

The feature space created by the hypernetwork acts as a metric space where proximity reflects dynamical similarity.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(EEG and Epilepsy Diagnosis)</span></p>

In studies using human electroencephalographic (EEG) recordings, feature vectors can distinguish between different physiological states:

* **Clustering:** Features from healthy controls and epileptic patients tend to cluster in separate regions of the feature space.
* **Performance:** This dynamical feature-based classification often outperforms traditional time-series classification tools.
* **Distance Measures:** The feature space provides a way to define "distance" between different dynamical systems based on their underlying temporal logic rather than raw signal similarity.

</div>

#### In-Context and Zero-Shot Learning

Recent advancements have introduced models capable of In-Context Learning or Zero-Shot Learning within the domain of dynamical systems.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Zero-Shot Learning in Dynamics)</span></p>

**Zero-shot learning** refers to the ability of a model to be presented with a novel multivariate signal and, without any further training or parameter updates, generate forward predictions of the system's long-term properties and its attractor.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Future Outlook)</span></p>

This approach, exemplified by recent work presented at NeurIPS, mimics the capabilities of large language models. The model "observes" the context of a signal and immediately internalizes the underlying dynamical rules to predict future behavior, representing a significant shift from traditional training-heavy paradigms.

</div>

### The Mixture of Experts (MoE) Architecture

The architecture of Dynamical Foundation Models draws inspiration from large language models (LLMs), specifically utilizing a Mixture of Experts (MoE) framework combined with attention mechanisms. Traditional modeling of dynamical systems often requires fine-tuning or training specifically on the system of interest. In contrast, these models aim to provide a "zero-shot" inference capability.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Mixture of Experts)</span></p>

The MoE architecture consists of $J$ individual "experts"—in this context, piecewise linear models—that specialize in different types of dynamical behaviors. Instead of a single model making a prediction, each expert generates an individual prediction, which is then aggregated into a final output.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Next State Prediction)</span></p>

The prediction for the next state $\hat{x}_t$ is defined as the weighted sum of the predictions made by $J$ experts:

$$\hat{x}_t = \sum_{j=1}^{J} w^{\text{ex}}_{j, t-1} \cdot \hat{x}_{j, t}$$

where $x \in \mathbb{R}^M$ represents the $M$-dimensional state space, $w^{\text{ex}}_{j, t-1}$ are the expert weights at time $t-1$, and $\hat{x}_{j, t}$ is the prediction of the $j$-th expert for time $t$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Constraints on Weights)</span></p>

To ensure a valid weighted average, the expert weights must satisfy the condition $\sum_{j=1}^{J} w^{\text{ex}}_j = 1$. This is typically achieved using a Softmax function within the network.

</div>

#### Feature Extraction via Convolutional Neural Networks

The expert weights are not static; they are determined by a Multi-Layer Perceptron (MLP) that processes a modified version of the context signal.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Modified Context Signal)</span></p>

The raw context signal $C$ is processed by a Convolutional Neural Network (CNN) to extract temporal features:

$$\tilde{C} = \text{CNN}(C)$$

These temporal features $\tilde{C}$ characterize the time series across various scales, providing a more robust input for the weighting mechanism than raw signal data alone.

</div>

#### The Attention Mechanism

The model determines the importance of different segments of the context signal through an attention mechanism, which is functionally similar to those found in transformer-based language models.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Attention Weights)</span></p>

The attention weights $a_t$ at time step $t$ are computed using a softmax function that measures the relevance between the current state and the context $C$:

$$a_t = \text{Softmax} \left( \frac{|C| \cdot D \cdot x_t}{\tau} \right)$$

where $C$ is the context matrix of dimensions $n \times T_c$, $D$ is a trainable weight matrix, $x_t$ is the current state, and $\tau$ is a temperature parameter.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Temperature Effects)</span></p>

The temperature parameters ($\tau$) control the "sharpness" of the model's focus:

* **Large $\tau$:** The impact of the context is diffused. All experts and context segments receive relatively equal weighting, leading to a broader, less specific average.
* **Small $\tau$:** The model places a high focus on specific, highly relevant bits of the context and specific experts, allowing for precise specialization.

</div>

#### Training Methodology: Sparse Teacher Forcing

The training of these foundation models presents a departure from standard deep learning norms regarding data volume.

* **Dataset Size:** Surprisingly, the model can be trained on a "tiny" dataset consisting of only 34 different dynamical systems. These systems are primarily chaotic or limit cycle attractors.
* **Training Technique:** The model utilizes Sparse Teacher Forcing.
* **Data Partitioning:** During training, segments of a trajectory are defined as "context" and others as "to be predicted." An overlap is often introduced between these segments to ease the training process.

#### Zero-Shot Inference and Generalization

A key strength of this architecture is its ability to perform zero-shot inference, meaning it can simulate systems not present in its training set without any additional fine-tuning.

**Generalization to Novel Topologies:** The model demonstrates the ability to reproduce the dynamics of systems that are topologically different from the training data. This includes correctly matching the state space trajectory and accurately reproducing the power spectrum of the unseen system.

**Generalization to New Initial Conditions:** The foundation model does not merely "memorize" the context trajectory. If the system is moved to a new initial condition—one not present in the provided context—the model can still correctly reproduce the expected trajectory based on the underlying dynamics it has inferred.

#### Comparative Analysis: Dynamics vs. Time Series Models

The lecture distinguishes between Dynamical Systems Foundation Models and Time Series Foundation Models (e.g., Amazon's Kronos).

| Feature | Time Series Foundation Models | Dynamics Foundation Models |
| --- | --- | --- |
| Training Data | Thousands to millions of datasets. | Tiny dataset (e.g., 34 systems). |
| Long-term Stability | Often fail; converge to fixed points. | Preserves long-term chaotic/limit cycle behavior. |
| Principles | Statistical/General Time Series. | Dynamical Systems Principles. |

Despite being trained on a small set of mathematical attractors, these models generalize effectively to real-world data, including:

* **Energy:** Electrical transformer data.
* **Infrastructure:** Cloud requests and traffic data.
* **Environment:** Weather data.
* **Biology:** Human fMRI signals and spiking neuron trajectories.

### Synthesis of AI and Dynamical Systems

The integration of Dynamical Systems Theory with modern Machine Learning (ML) architectures represents a convergence of classical reconstruction principles and contemporary foundation models. This synthesis allows for the modeling of complex, multi-scale phenomena that were previously difficult to capture using traditional linear methods.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dynamical Systems Reconstruction)</span></p>

The process of identifying the underlying rules or state-space representations governing a system based on observed data. In the context of modern AI, this involves combining historical reconstruction techniques with high-capacity neural architectures.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection to General Artificial Intelligence)</span></p>

The relationship between current architectural trends and Artificial General Intelligence (AGI) is defined by the integration of specific principles:

* **Combination of Principles:** Current models are not monolithic; they combine disparate mechanisms like attention and expert mixtures.
* **Foundation Models:** These serve as the basis for broader intelligence by being trained on expansive, large-scale datasets.
* **Methodological Convergence:** The intersection of Dynamical Systems Reconstruction and AI architecture is a primary driver in the development of models that exhibit generalizable behavior.

</div>

#### Chaotic Signal Reconstruction and Generalization

A critical benchmark for any dynamical system model is its ability to handle chaotic signals and generalize to data it has not encountered during the training phase.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Chaotic Signal)</span></p>

In the context of reconstruction, a signal is considered chaotic when it exhibits:

* **Multiple Time Scales:** The presence of various temporal resolutions and frequencies within a single signal.
* **Complex Structure:** A non-trivial internal organization that is difficult for simple models to approximate.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Generalization to Unseen Data)</span></p>

One of the most significant capabilities of modern foundation models is their performance on data points the model "hasn't seen in training at all." Despite the lack of prior exposure to specific chaotic instances, a well-constructed model can:

* Produce an output that is fairly similar in structure to the original context signal.
* Maintain the integrity of the underlying dynamical scales.
* Successfully reconstruct the signal's complex structure through learned foundational patterns.

The goal of the model is not mere memorization but the internalizing of structural rules. When a model produces a "similar structure" to a chaotic signal it hasn't seen, it demonstrates that it has captured the essential dynamics of the system rather than just the specific data points of the training set.

</div>
