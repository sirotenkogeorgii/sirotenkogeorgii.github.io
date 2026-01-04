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

## Table of Contents

* **Part I: An Introduction to Dynamical Systems**
  * **Chapter 1: Fundamentals and Linear Systems**
    * Defining Dynamical Systems: Continuous and Discrete Time
    * The Concept of State Space, Trajectories, and Vector Fields
    * Autonomous vs. Non-Autonomous Systems
    * Rewriting Higher-Order Systems as First-Order Systems
    * Analysis of 1D Linear Systems: Equilibria and Stability
    * Analysis of N-Dimensional Linear Systems: The Eigenvalue Problem
    * A Geometric Classification of 2D Linear Equilibria
  * **Chapter 2: Nonlinear Dynamics**
    * Introduction to Nonlinear Oscillations
  * **Chapter 3: Advanced Topics in Dynamical Systems Theory**
    * Systems with Potentials, Energy Functions, and Hamiltonians
    * The Relationship Between Recursive Maps and Continuous-Time Systems
    * Chaos Theory and Fractal Geometry
    * Bifurcation Theory and Tipping Points
* **Part II: Dynamical Systems in Machine Learning and AI**
  * **Chapter 4: Inferring Dynamical Systems from Data**
    * Traditional Techniques for Time Series Analysis
    * Dynamical Systems Reconstruction with Machine Learning
    * Specialized Architectures: Physics-Informed Neural Networks (PINNs)
    * Specialized Architectures: Neural Ordinary Differential Equations (Neural ODEs)
    * The Role of Recurrent Neural Networks (RNNs)
  * **Chapter 5: Current Frontiers in Scientific Machine Learning**
    * Advanced Architectures: Encoders, Decoders, and Complex Structures
    * The Challenge of Out-of-Domain Generalization
    * Foundation Models for Scientific Discovery

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
2. $\Phi(t + s, x) = \Phi(t, \Phi(s, x)), \quad \forall s, t \in \mathbb{R}, , x \in M,$
3. $\Phi$ is continuous in $(t, x)$.

</div>

At its core, the theory of dynamical systems is the study of systems that change. We can describe these changes using different mathematical objects, depending on whether the evolution is continuous or discrete. Dynamical systems are broadly classified into two categories based on how time is treated:

#### Continuous-Time Systems (Flows)

These systems evolve continuously. These are typically described by differential equations. They are most often described by **ordinary differential equations (ODEs)** or **partial differential equations (PDEs)**.

* **Ordinary Differential Equations (ODEs):** These describe the rate of change of a system's variables with respect to a single dimension, typically time. The notation often uses a dot to represent the time derivative.
  * A system can be multi-dimensional, with a state vector $x \in \mathbb{R}^p$.
* Here, $\dot{x}$ is a vector of temporal derivatives $[\dot{x}_1, \dot{x}_2, \dots, \dot{x}_p]^T$, and $f(x)$ is a function, often called the vector field, that maps the current state x to its rate of change. This single vector equation represents a set of $p$ coupled differential equations: 

$$\begin{align*} \dot{x}_1 &= f_1(x_1, x_2, \dots, x_p) \\ \dot{x}_2 &= f_2(x_1, x_2, \dots, x_p) \\ &\vdots \\ \dot{x}_p &= f_p(x_1, x_2, \dots, x_p) \end{align*} $$

* **Partial Differential Equations (PDEs):** These are used for systems that evolve along multiple dimensions simultaneously, such as time and space. For example, describing the temperature u across a physical object would involve derivatives with respect to time $\dot{u}$ and spatial coordinates $u_x, u_y, \dots$.

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
2. The behavior of a nonlinear system in the close vicinity of an equilibrium point can often be accurately approximated by a linear system.

### The One-Dimensional Case: $\dot{x} = ax$

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

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fixed point or Equilibrium point)</span></p>

A point $x^{\ast}\in\mathbb{R}^n$ is called equilibrium point of a system ODEs, if $f(t,x)=0$ for all $t\in I$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Analysis</span>(Stability of the Equilibrium (1D case))<span class="math-callout__name"></span></p>

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

Now we consider a system of $n$ coupled linear ODEs, where $x \in \mathbb{R}^n$ and $A$ is an $n \times n$ matrix. 

$$\dot{x} = Ax, \quad x(0) = x_0$$

*Proof:* Derivation of the General Solution

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

$$x(t)=v_i,e^{\lambda_i t}$$

satisfies $\dot x = Ax$.

If $A$ has $n$ linearly independent eigenvectors, then these vectors form a basis of the state space. Consequently, any initial state $x_0=x(0)$ can be written as a linear combination of eigenvectors:

$$x_0=\sum_{i=1}^n c_i v_i.$$

Because the system is linear, it preserves superposition: the derivative of a linear combination of solutions is the same linear combination of their derivatives. Indeed, using $Av_i=\lambda_i v_i$,

$$A\Big(\sum_{i=1}^n c_i v_i\Big)=\sum_{i=1}^n c_i Av_i=\sum_{i=1}^n c_i \lambda_i v_i.$$

Therefore, the solution starting from $x_0$ is obtained by summing the individual eigenvector solutions with the same coefficients:

$$x(t)=\sum_{i=1}^n c_i v_i e^{\lambda_i t}.$$

This gives the general solution, with the constants $c_i$ chosen to match the initial condition.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(General Solution of Linear Systems)</span></p>

Assuming the matrix $A$ has $n$ distinct eigenvalues $\lambda_1, \dots, \lambda_n$ with corresponding eigenvectors $v_1, \dots, v_n$, the eigenvectors form a basis for the state space. Since the system is linear, any linear combination of individual solutions is also a solution. The general solution can therefore be written as a sum:

$$x(t) = \sum_{i=1}^{n} c_i v_i e^{\lambda_i t}$$

The coefficients $c_i$ are determined by the initial condition $x(0) = x_0$:

$$x_0 = \sum_{i=1}^{n} c_i v_i$$

The behavior of the system is a superposition of simple exponential behaviors along each of the eigendirections.

</div>

**Solution with Complex Eigenvalues**

The eigenvalues of a real matrix $A$ can be complex. Since $A$ is real, its complex eigenvalues must come in conjugate pairs: $\lambda = \alpha \pm i\omega$.

Recalling **Euler's formula**: 

$$e^{i\theta} = \cos(\theta) + i\sin(\theta)$$

We can rewrite the exponential term for a complex eigenvalue $\lambda_i = \alpha_i + i\omega_i$:

$$e^{\lambda_i t} = e^{(\alpha_i + i\omega_i)t} = e^{\alpha_i t} e^{i\omega_i t} = e^{\alpha_i t}(\cos(\omega_i t) + i\sin(\omega_i t))$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(General Solution of Linear Systems with Complex Eigenvalues)</span></p>

Assuming the matrix $A$ has $n$ distinct **complex** eigenvalues $\lambda_1, \dots, \lambda_n$ with corresponding eigenvectors $v_1, \dots, v_n$, the eigenvectors form a basis for the state space. Since the system is linear, any linear combination of individual solutions is also a solution. The general solution can therefore be written as a sum:

$$x(t) = \sum_{i=1}^{n} c_i v_i e^{\alpha_i t} (\cos(\omega_i t) + i\sin(\omega_i t))$$

The solution form reveals that the system's behavior has two components:

* An **exponential growth or decay** component, governed by the real part of the eigenvalue, $e^{\alpha_i t}.$
* An **oscillatory** component, governed by the imaginary part of the eigenvalue, $\cos(\omega_i t) + i\sin(\omega_i t)$.

The overall behavior is a spiral: the system oscillates while its amplitude grows or shrinks exponentially.

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

**Case: Some Eigenvalues with Zero Real Part**
* **Line or Plane of Equilibria:** One eigenvalue is zero, and the others are negative (e.g., $\lambda_1 < 0$, $\lambda_2 = 0$).
  * *Geometry*: There is an entire line (or plane in higher dimensions) of fixed points. This line is the eigenspace corresponding to $\lambda_2 = 0$. Trajectories from off this line will converge toward it along the stable eigendirections.
  * *Stability*: Marginally stable.
  * *Remark*: This configuration, sometimes called a **line attractor**, is crucial in neuroscience and machine learning for modeling memory. The system can be placed at any point along the line and will stay there, effectively "remembering" that state.

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

2. *Complex eigenvalues: spirals are just “oscillation × decay”*

If $\lambda=\alpha\pm i\omega$, the real solutions look like

$$e^{\alpha t}\big(\cos(\omega t),u + \sin(\omega t),w\big),$$

for some real vectors $u,w$. The ($\cos$/$\sin$) part just rotates/oscillates, while the amplitude is scaled by $e^{\alpha t}$. If $\alpha<0$, the amplitude shrinks to 0, so trajectories spiral inward.

**Even if $A$ is not diagonalizable, negative real parts still win**

In general, $A$ can be put into Jordan form $A=VJV^{-1}$. Each Jordan block with eigenvalue $\lambda$ contributes terms like

$$t^k e^{\lambda t}$$

(for some nonnegative integer $k$). Taking magnitudes gives roughly

$$t^k e^{\operatorname{Re}(\lambda)t}.$$

If $\operatorname{Re}(\lambda)<0$, the exponential decay dominates any polynomial factor $t^k$, so these terms still go to 0. Hence $e^{At}\to 0$ and therefore $x(t)\to 0$.

A common way to summarize this is:

> If all eigenvalues satisfy $\max_i \operatorname{Re}(\lambda_i) < 0$, then there exist constants $M,\gamma>0$ such that
> 
> $$\lvert\lvert x(t)\rvert\rvert \le M e^{-\gamma t}\lvert\lvert x_0\rvert\rvert \quad \text{for all } t\ge 0,$$
> 
> so every trajectory converges to the origin exponentially fast.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hyperbolic equilibrium point, Hyperbolic Systems)</span></p>

An equilibrium point is called **hyperbolic** if none of its eigenvalues have a real part equal to zero. This means the system has no **centers** and no directions of marginal stability. Stable nodes, unstable nodes, saddle nodes, and spirals are all hyperbolic. This is an important property because the local behavior of hyperbolic equilibria is robust to small changes in the system.

</div>

---

# Lecture 2

Classification of dynamics in terms of classes of similar matrices
Fundamental theorem of linear dynamical systems: general solution in terms of matrix exponential
  non-degenerate case (all eigenvalues distinct): demonstration of equivalence to form presented in first lecture
  degenerate case
Inhomogeneous systems
  Location of equilibrium
  Change of variables centering equilibrium
Non-autonomous systems
  Variation of parameters approach
Linear maps (discrete-time DS)
  Analysis of limiting behaviour
  1d maps: cobweb plots
  Higher-dimensional maps: classification in terms of eigenvalues
Relation between linear maps and linear ODEs
  Discretization
  Ex. Piecewise-Linear Recurrent Neural Networks
  Ex. line attractors: implement arbitrary timescales by continuously changing parameters
Interlude: Machine learning benchmark tasks for RNNs
  Ex. addition problem
  Role of line/plane/torus attractors
Def. flows of dynamical systems (flow operator/map)
  group actions
  trajectories/orbits
Fundamental uniqueness/existence theorem for solutions of ODEs (Picard–Lindelöf/Cauchy–Lipschitz)

Chapter 1: General Solutions for Linear Systems

This chapter generalizes our understanding of linear dynamical systems. We will move beyond the specific case of systems with distinct eigenvalues to introduce a universal solution applicable to any linear system, articulated by the Fundamental Theorem of Linear Dynamical Systems.

1.1 Recap: The Eigenvector-Based Solution

We previously considered linear dynamical systems defined by systems of ordinary differential equations (ODEs) of the form:

$$\dot{\mathbf{x}} = A \mathbf{x}$$

where $\mathbf{x} \in \mathbb{R}^m$ is the state vector and $A$ is a square $m \times m$ matrix.

Under the strong assumption that the matrix $A$ has distinct eigenvalues ($\lambda_i$) and that its corresponding eigenvectors ($\mathbf{v}_i$) form a basis for the space, we derived a general solution. This solution expresses the evolution of the system, $\mathbf{x}(t)$, from an initial condition $\mathbf{x}_0$ as a linear combination of exponential and oscillatory terms:

$$\mathbf{x}(t) = \sum_{i=1}^{m} c_i e^{\lambda_i t} \mathbf{v}_i = \sum_{i=1}^{m} c_i e^{\alpha_i t} (\cos(\omega_i t) + i \sin(\omega_i t)) \mathbf{v}_i$$

Here, the eigenvalues $\lambda_i = \alpha_i + i\omega_i$ are split into their real parts ($\alpha_i$), which govern exponential growth or decay, and their imaginary parts ($\omega_i$), which govern oscillations. The coefficients $c_i$ are determined by the initial conditions.

This formulation allowed us to classify various types of equilibria (fixed points), such as stable/unstable nodes, saddles, stable/unstable spirals, and centers. However, the initial assumptions are restrictive. They do not cover all possible linear systems, specifically those where eigenvalues are repeated. To address this, we must develop a more general framework.

1.2 A More General Approach: The Fundamental Theorem

To formulate a solution that covers all cases, we first introduce the concept of similar matrices, which helps classify systems based on their underlying dynamics.

Similar Matrices and Topological Equivalence

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Similar Matrices)</span></p>

Two square matrices, $A_1$ and $A_2$, are called **similar** if there exists an *invertible* matrix $S$ such that the following relationship holds:

$$A_1 = S A_2 S^{-1}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition behind Similar Matrices)</span></p>

The matrix $S$ represents an invertible transformation, or a change of variables (a change of basis). If two matrices are similar, it means that the dynamical systems they define are topologically equivalent; they possess the same fundamental dynamics, merely viewed from a different coordinate system. The eigendecomposition of a matrix, for instance, is a transformation that reveals its similarity to a diagonal matrix of its eigenvalues.

</div>

Canonical Forms for 2x2 Systems

For any $2 \times 2$ matrix, it can be shown that it is similar to one of three distinct canonical forms. These forms represent the fundamental classes of dynamics possible in two-dimensional linear systems.

1. **Distinct Real Eigenvalues:** The matrix is similar to a diagonal form. 
   
   $$A \sim \begin{pmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{pmatrix} $$

  * **Eigenvalues:** has the two real eigenvalues $λ_1 = a$ and $λ_2 = b$.
  * **Dynamics:** This form corresponds to dynamics without an oscillatory component, such as saddles and stable/unstable nodes.
  
2. **Complex Conjugate Eigenvalues:** The matrix is similar to a form representing rotation and scaling. 
   
   $$A \sim \begin{pmatrix} a & -b \\ b & a \end{pmatrix}$$

  * **Eigenvalues:** has the two complex eigenvalues $λ_{1,2} = a\pm ib$.
  * **Dynamics:** Eigenvalues for such a matrix come in complex conjugate pairs. This form can be decomposed into a scaling component (related to $a$) and a rotational component (related to $b$). This gives rise to spirals (stable if $a<0$, unstable if $a>0$) and centers (if $a=0$).
  
3. **Repeated Eigenvalues (Degenerate Case):** This is the case our previous solution did not cover. The matrix is similar to the form:
   
  $$A \sim \begin{pmatrix} a & 1 \\ 0 & a \end{pmatrix}$$

  * **Eigenvalues:** has only the one eigenvalue $\lambda_1 = a$.
  * **Dynamics:** This matrix has one eigenvalue, $a$, with a multiplicity of two. However, it only has one corresponding eigenvector direction. This case is called degenerate because the eigenvectors do not form a basis for the space; instead, two eigenvector directions align. The dynamics ultimately collapse into a one-dimensional space, with all trajectories aligning with the single eigenvector direction. The specific path of convergence or divergence depends on the initial conditions and the value of $a$.

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
* The $\begin{pmatrix}1&t\\0&1\end{pmatrix}$ part is a **shear**: it pushes points sideways in the $x_1$ direction at a rate proportional to $t$ and to their $x_2$-component.

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

has a unique solution $x:\mathbb{R}\to \mathbb{R}^n$ of the form:

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

Remark/Intuition

It is straightforward to see why this form constitutes a solution to the ODE. If we take the temporal derivative of the solution $\mathbf{x}(t) = e^{At} \mathbf{x}_0$, we differentiate the series term-by-term:

$$\frac{d}{dt} \mathbf{x}(t) = \frac{d}{dt} \left( \sum_{k=0}^{\infty} \frac{A^k t^k}{k!} \right) \mathbf{x}_0$$

$$= \left( \sum_{k=1}^{\infty} \frac{A^k k t^{k-1}}{k!} \right) \mathbf{x}_0 = A \left( \sum_{k=1}^{\infty} \frac{A^{k-1} t^{k-1}}{(k-1)!} \right) \mathbf{x}_0$$

By re-indexing the sum (let $j=k-1$), we recover the original series:

$$= A \left( \sum_{j=0}^{\infty} \frac{(At)^j}{j!} \right) \mathbf{x}_0 = A e^{At} \mathbf{x}_0 = A \mathbf{x}(t)$$


This confirms that $\dot{\mathbf{x}} = A\mathbf{x}(t)$, satisfying the differential equation. The full proof of the theorem also requires showing this solution is unique, which can be done by assuming two distinct solutions and demonstrating they must be identical.


--------------------------------------------------------------------------------


Equivalence of Solutions for Diagonalizable Systems

While the matrix exponential provides a powerful general solution, it is important to verify that it is consistent with the eigenvector-based solution we derived earlier for the case where $A$ is diagonalizable (i.e., has distinct eigenvalues).

Proof

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

--------------------------------------------------------------------------------


The Degenerate Case: Repeated Eigenvalues

The true power of the Fundamental Theorem is that it also provides the solution for the degenerate case, where eigenvalues are repeated and the matrix is not diagonalizable.

Remark/Intuition

In the degenerate case, the solution involves not just exponential terms, but also polynomials of time ($t$). These polynomial terms arise from the off-diagonal elements in the canonical form of the matrix (e.g., the '1' in the third canonical form).

Example

Consider the $2 \times 2$ matrix from the third canonical form, which has a repeated eigenvalue a: 

$$A = \begin{pmatrix} a & 1 \\ 0 & a \end{pmatrix}$$

The solution for a system governed by this matrix, $\mathbf{x}(t) = e^{At}\mathbf{x}_0$, has the form: 

$$\mathbf{x}(t) = e^{At}\mathbf{x}_0 = e^{at} \begin{pmatrix} 1 & t \\ 0 & 1 \end{pmatrix} \mathbf{x}_0$$

Notice the appearance of the linear term $t$ in the matrix. For higher-dimensional degenerate systems, higher-order polynomials of $t$ can appear in the solution. This is a direct consequence of the structure of the matrix exponential for non-diagonalizable matrices.


1. Analysis of Extended Linear Systems

In our initial exploration, we focused on homogeneous linear systems of the form $\dot{x} = Ax$. We now extend this analysis to two important cases: systems with a constant offset (affine systems) and systems that explicitly depend on time (non-autonomous systems).

1.1 Inhomogeneous (Affine) Systems of ODEs

An affine system introduces a constant vector term, shifting the dynamics in state space.

Definition: Affine System of ODEs

An affine or inhomogeneous linear system of ordinary differential equations is defined by:  $\dot{x} = Ax + b$  where $x, b \in \mathbb{R}^m$ and $A$ is an $m \times m$ matrix.

Remark/Intuition: Shifting the Equilibrium

The addition of the constant vector b does not alter the fundamental dynamics of the system, which are dictated by the matrix A. Instead, its effect is to move the system's equilibrium point. The vector field remains unchanged relative to this new equilibrium.

To understand this, we first locate the new equilibrium, or fixed point, by finding the point $x^{\ast}$ where the flow is zero ($\dot{x}=0$).

Assuming the matrix $A$ is invertible, we solve for the fixed point: 

$$0 = Ax^{\ast} + b \implies Ax^{\ast} = -b \implies x^{\ast} = -A^{-1}b$$ 

This point $x^{\ast}$ is our new equilibrium, shifted from the origin.

Proof: Equivalence of Dynamics via Change of Variables

We can formally prove that the dynamics remain the same by defining a new variable y that represents the state relative to the fixed point $x^{\ast}$.

1. Define a new variable: Let $y = x - x^{\ast}$. This is equivalent to $x = y + x^{\ast}$.
2. Consider the dynamics of the new variable: The temporal derivative of $y$ is $\dot{y} = \dot{x}$, since $x^{\ast}$ is a constant and its derivative is zero.
3. Substitute into the original equation: We can now express $\dot{y}$ in terms of $y$
   
   $\dot{y} = \dot{x} = Ax + b$  
   
   Substitute $$x = y + x^{\ast}: \dot{y} = A(y + x^{\ast}) + b = Ay + Ax^{\ast} + b$$  
   
  Now, substitute the expression for the fixed point, $x^{\ast} = -A^{-1}b$:
   
  $$\dot{y} = Ay + A(-A^{-1}b) + b = Ay - b + b$$

1. Result: The dynamics for the new variable are:
   
  $$\dot{y} = Ay$$
   
  This is precisely the homogeneous linear system we have already analyzed. The dynamics (stability, rotation, etc.) around the fixed point $x^{\ast}$ are identical to the dynamics of the homogeneous system around the origin. To find the full solution for $x(t)$, one solves for $y(t)$ and then recovers $x(t) = y(t) + x^{\ast}$.

Remark/Intuition: The Non-Invertible Case

If the matrix $A$ is not invertible, it possesses at least one zero eigenvalue. In this scenario, a unique fixed point does not exist. This corresponds to the case of a center or, more generally, a line attractor (or plane/hyperplane attractor in higher dimensions). The system has a continuous manifold of equilibrium points along the direction of the eigenvector(s) associated with the zero eigenvalue(s).


--------------------------------------------------------------------------------


1.2 Non-autonomous Systems with a Forcing Function

We now consider systems where the dynamics are explicitly influenced by time, driven by an external "forcing function."

Definition: Non-autonomous System with Forcing Function

A non-autonomous linear system with a forcing function $f(t)$ is defined as a system that explicitly depends on time. For simplicity, we will analyze the scalar case:

$$\dot{x} = ax + f(t)$$ 

Remark/Intuition: Variation of Parameters

To solve this type of equation, we employ a powerful technique known as variation of parameters. The logic is as follows: we know the solution to the homogeneous part of the equation ($\dot{x} = ax$) is $x(t) = C e^{at}$, where $C$ is a constant. We now "promote" this constant to a time-dependent function, $k(t)$, and propose an ansatz (an educated guess) for the full solution that has a similar form. This allows the solution to adapt to the time-varying influence of $f(t)$.

Proof: Derivation of the Solution

1. Formulate the ansatz: Let the solution be of the form $x(t) = (h(t) + C)e^{at}$, where $h(t)$ is an unknown function we need to determine and $C$ is a constant of integration.
2. Take the temporal derivative: Using the product rule, the derivative of our ansatz is:
  
  $$\dot{x}(t) = \frac{d}{dt}[(h(t) + C)e^{at}] = \dot{h}(t)e^{at} + a(h(t) + C)e^{at}$$ 

3. Equate with the original ODE: The definition of the system states that $\dot{x} = ax + f(t)$. We can substitute our ansatz for $x(t)$ into this definition: 
   
  $$\dot{x}(t) = a[(h(t) + C)e^{at}] + f(t)$$

4. Compare the two expressions for $\dot{x}(t)$:
   
  $$\dot{h}(t)e^{at} + (h(t) + C)ae^{at} = a(h(t) + C)e^{at} + f(t)$$
  
  The term $(h(t) + C)ae^{at}$ appears on both sides and cancels out.
5. Isolate the derivative of $h(t)$: We are left with a simple expression:
  
  $$\dot{h}(t)e^{at} = f(t)$$  
  
  Multiplying through by $e^{-at}$ gives:  
  
  $$\dot{h}(t) = f(t)e^{-at}$$ 

6. Integrate to find $h(t)$: To find the function $h(t)$, we integrate both sides with respect to time:
   
  $$h(t) = \int f(t)e^{-at} dt$$ 

The full solution to the non-autonomous equation is therefore found by computing this integral for $h(t)$ and substituting it back into our original ansatz. This method provides a general recipe for solving first-order linear ODEs with a forcing function.

---

3. Linear Maps (Discrete-Time)

Many dynamical systems, including various types of recurrent neural networks, are defined as maps rather than differential equations. These are discrete-time systems.

System Definition

A discrete-time autonomous dynamical system is defined by a recursive prescription:  

$$\mathbf{x}_t = f(\mathbf{x}_{t-1})$$

We will focus on the affine linear map, which is the discrete-time analogue of the inhomogeneous systems discussed earlier:  

$$\mathbf{x}_t = A \mathbf{x}_{t-1} + \mathbf{b}$$  

Such a map generates a sequence of vector-valued numbers $\lbrace\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T\rbrace$ starting from an initial condition $\mathbf{x}_1$. A primary goal is to understand the limiting behavior of this sequence as $t \to \infty$.

Iterative Solution and Limiting Behavior (Scalar Case)

To build intuition, let's analyze the scalar case:

$$x_t = ax_{t-1} + b$$

1. Recursive Expansion: We can expand the expression recursively to understand its structure over time:

* At $t=2: x_2 = ax_1 + b$
* At $t=3: x_3 = a(x_2) + b = a(ax_1 + b) + b = a^2x_1 + ab + b$ 
* At $t=4: x_4 = a(x_3) + b = a(a^2x_1 + ab + b) + b = a^3x_1 + a^2b + ab + b$

2. General Form: Observing the pattern, the state at a general time step $T$ is:
  
  $$x_T = a^{T-1}x_1 + b(a^{T-2} + a^{T-3} + \dots + a^1 + a^0)$$
  
  The second term is a finite geometric series. We can write this more compactly as:
  
  $$x_T = a^{T-1}x_1 + b \sum_{i=0}^{T-2} a^i$$

3. Limiting Behavior: We are interested in what happens as $t \to \infty$. The convergence of this sequence depends entirely on the value of $a$.

* Condition for Convergence: The sequence converges only if the absolute value of a is less than one, i.e., $\lvert a\rvert < 1$.
* Analysis of Terms:
  * Initial Condition Term: For $\lvert a\rvert<1$, the term $a^{T-1}x_1$ decays to zero as $T \to \infty$. This means the system forgets about the initial condition exponentially fast.
  * Geometric Series Term: For $\lvert a\rvert <1$, the infinite geometric series converges to a fixed value:
* The Limit: Combining these results, the limit of $x_t$ as $t \to \infty$ is: 
  
  $$\lim_{t \to \infty} x_t = b \left( \frac{1}{1-a} \right)$$

Remark/Intuition: A Different View

Another powerful way to illustrate this solution is to plot $x_{t+1}$ as a function of $x_t$. The fixed points of the map are found where this function intersects the bisectrix line, defined by $x_{t+1} = x_t$.

A Study of Discrete-Time Linear Systems

<!-- Table of Contents

1. One-Dimensional Discrete-Time Linear Systems

* 1.1 The Recursive Linear Map
* 1.2 Geometric Interpretation: The Cobweb Plot
* 1.3 Fixed Points in One Dimension
  * 1.3.1 Definition and Geometric Intuition
  * 1.3.2 Algebraic Solution
* 1.4 Stability Analysis of Fixed Points
  * 1.4.1 Stable Fixed Points: |a| < 1
  * 1.4.2 Unstable Fixed Points: |a| > 1
  * 1.4.3 Neutrally Stable Points: |a| = 1

2. Higher-Dimensional Discrete-Time Linear Systems

* 2.1 The General Affine Map
* 2.2 Solving for Fixed Points in m Dimensions
* 2.3 System Dynamics and Diagonalization
* 2.4 Stability Analysis via Eigenvalues -->


--------------------------------------------------------------------------------


1. One-Dimensional Discrete-Time Linear Systems

We begin our exploration with the simplest case: a one-dimensional, discrete-time linear system. These systems, while seemingly basic, exhibit a rich set of behaviors that provide a foundational understanding for more complex, higher-dimensional systems.

1.1 The Recursive Linear Map

A one-dimensional discrete-time linear system is described by a recursive relationship that maps the state of the system at time $t$, denoted by $x_t$, to its state at the next time step, $t+1$.

Definition: 1D Linear Map The state $x_{t+1}$ of the system at time $t+1$ is given by an affine transformation of its state $x_t$ at time $t$:

$$x_{t+1} = f(x_t) = ax_t + b$$

where a and b are scalar constants. The parameter $a$ represents the slope, and $b$ is the intercept or offset.

1.2 Geometric Interpretation: The Cobweb Plot

To gain a deeper intuition for the system's evolution over time, we can visualize this recursive process graphically. We plot the function $x_{t+1} = ax_t + b$ against the bisectrix, which is the line $x_{t+1} = x_t$. The intersection of these two lines holds special significance, as we will see shortly.

Remark/Intuition: The Cobweb Plot The Cobweb Plot is a powerful geometric technique for visualizing the trajectory of a discrete-time system. It provides an immediate feel for whether the system converges to a specific value, diverges to infinity, or exhibits other behaviors. The procedure is as follows:

1. Initialization: Start with an initial condition, $x_0$, on the horizontal axis.
2. Evaluation: Move vertically from $x_0$ to the function line $x_{t+1} = ax_t + b$. The height of this point gives the next state, $x_1$.
3. Iteration: To use $x_1$ as the next input, move horizontally from the point on the function line to the bisectrix ($x_{t+1} = x_t$). This transfers the output value $x_1$ to the horizontal axis, preparing it for the next iteration.
4. Repeat: From this new point on the bisectrix, move vertically again to the function line to find $x_2$, then horizontally to the bisectrix, and so on.

The path traced by these movements often resembles a spider's web, spiraling inwards or outwards, which gives the method its name.

<div class="pmf-grid">
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/coweb_line_attractor.jpeg' | relative_url }}" alt="a" loading="lazy">
    <!-- <figcaption>Sketch of the Rice's theorem proof</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/coweb_minu_one_a.jpeg' | relative_url }}" alt="a" loading="lazy">
    <!-- <figcaption>Sketch of the Rice's theorem proof</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/coweb_minus_one_a.jpeg' | relative_url }}" alt="a" loading="lazy">
    <!-- <figcaption>Sketch of the Rice's theorem proof</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/coweb_negative_a_divergence.jpeg' | relative_url }}" alt="a" loading="lazy">
    <!-- <figcaption>Sketch of the Rice's theorem proof</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/coweb_positive_a_convergence.jpeg' | relative_url }}" alt="a" loading="lazy">
    <!-- <figcaption>Sketch of the Rice's theorem proof</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/coweb_positive_a_divergence.jpeg' | relative_url }}" alt="a" loading="lazy">
    <!-- <figcaption>Sketch of the Rice's theorem proof</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/coweb_positive_one_a_negative_b.jpeg' | relative_url }}" alt="a" loading="lazy">
    <!-- <figcaption>Sketch of the Rice's theorem proof</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/coweb_positive_one_a_positive_b.jpeg' | relative_url }}" alt="a" loading="lazy">
    <!-- <figcaption>Sketch of the Rice's theorem proof</figcaption> -->
  </figure>
</div>

1.3 Fixed Points in One Dimension

1.3.1 Definition and Geometric Intuition

A central concept in dynamical systems is the notion of a fixed point, which is analogous to an equilibrium in continuous-time systems described by differential equations.

Definition: Fixed Point A point $x^{\ast}$ is a fixed point of a discrete-time system $x_{t+1} = f(x_t)$ if it remains unchanged by the map. That is, it satisfies the condition:

$$x^{\ast} = f(x^{\ast})$$

If the system is initialized at a fixed point, it will remain there for all future time steps. There is no movement at this point.

Remark/Intuition: Geometric View of Fixed Points Geometrically, a fixed point is simply the intersection of the function graph $y = f(x)$ and the bisectrix $y=x$. At this specific point, the input to the function is exactly equal to its output, satisfying the definition $x^{\ast} = f(x^{\ast})$.

1.3.2 Algebraic Solution

We can find the fixed point not only graphically but also by solving the defining equation algebraically.

Proof: Derivation of the 1D Fixed Point To find the fixed point $x^{\ast}$, we set the output equal to the input according to the definition:

$$x^{\ast} = ax^{\ast} + b$$

We then solve for 

$$x^{\ast}:x^{\ast} - ax^{\ast} = b$$

$$(1-a)x^{\ast} = b$$

Assuming $a \neq 1$, we can divide by $(1-a)$ to find the unique fixed point:

$$x^{\ast} = \frac{b}{1-a}$$

This algebraic solution precisely matches the limiting solution for convergent systems and identifies the point of intersection on the cobweb plot.

1.4 Stability Analysis of Fixed Points

A fixed point can be stable, unstable, or neutrally stable, depending on the behavior of nearby trajectories. This stability is determined entirely by the slope parameter, $a$.

1.4.1 Stable Fixed Points: $\lvert a\rvert < 1$

If the absolute value of the slope is less than one, any initial condition will lead to a trajectory that converges to the fixed point.

Definition: Stable Fixed Point A fixed point $x^{\ast}$ is stable if trajectories starting near $x^{\ast}$ converge towards it as $t \to \infty$. In the linear 1D case, this occurs when $\lvert a\rvert < 1$.

Remark/Intuition On the Cobweb Plot, a slope with $\lvert a\rvert < 1$ is less steep than the bisectrix. This geometric configuration ensures that each step of the cobweb construction brings the state closer to the intersection point, causing the "web" to spiral inwards towards the fixed point.

1.4.2 Unstable Fixed Points: $\lvert a\rvert > 1$

If the absolute value of the slope is greater than one, the system will diverge from the fixed point, unless it starts exactly on it.

Definition: Unstable Fixed Point A fixed point $x^{\ast}$ is unstable if trajectories starting near $x^{\ast}$ move away from it as $t \to \infty$. In the linear 1D case, this occurs when $\lvert a\rvert > 1$.

Remark/Intuition When $\lvert a\rvert > 1$, the function line is steeper than the bisectrix. The Cobweb Plot immediately reveals that each iteration throws the state further away from the intersection point, causing the "web" to spiral outwards. The system still possesses a fixed point, but any infinitesimal perturbation from it will lead to divergence.

1.4.3 Neutrally Stable Points: $\lvert a\rvert = 1$

The case where the slope has an absolute value of exactly one represents a boundary between stability and instability.

Definition: Neutrally Stable Point A point or system is neutrally stable if nearby trajectories neither converge towards nor diverge away from it, but instead remain in a bounded orbit. This occurs when $\lvert a\rvert = 1$.

We must consider two sub-cases:

* Case 1: $a = 1$ 
  * If $b \neq 0$, the system becomes $x_{t+1} = x_t + b$. This represents linear divergence, as a constant amount $b$ is added at each time step. There is no fixed point.
  * If $b = 0$, the system is $x_{t+1} = x_t$. In this scenario, every point is a fixed point. This is sometimes referred to as a line attractor, as there is a continuous set of fixed points.
* Case 2: $a = -1$
  * The system takes the form $x_{t+1} = -x_t + b$. This leads to oscillatory behavior.
  * If $b=0$, the system $x_{t+1} = -x_t$ simply flips the sign at each step (e.g., $x_0, -x_0, x_0, \dots$).
  * If $b \neq 0$, the system oscillates between two distinct values. This is known as a flip oscillation.

Example: Flip Oscillation Consider the system $x_{t+1} = -x_t + 1$. Let the initial state be $x_1 = 2$.

* $x_2 = -x_1 + 1 = -(2) + 1 = -1$
* $x_3 = -x_2 + 1 = -(-1) + 1 = 2$
* $x_4 = -x_3 + 1 = -(2) + 1 = -1$ The system enters a stable 2-cycle, oscillating between the values 2 and -1. The amplitude of this oscillation depends on the initial value, but the oscillatory nature is preserved. This behavior is analogous to the center case in systems of linear differential equations, where solutions form a continuous set of stable orbits.

Remark/Intuition: Analogy to Continuous Systems The spectrum of solutions observed in discrete-time linear systems—stable and unstable fixed points, and mutually stable oscillations—is precisely the same class of solutions found in continuous-time linear systems of ordinary differential equations. This parallel provides a powerful conceptual bridge between the two domains.


--------------------------------------------------------------------------------


1. Higher-Dimensional Discrete-Time Linear Systems

We now generalize our analysis to systems with $m$ dimensions, where the state is represented by a vector and the dynamics are governed by a matrix transformation.

2.1 The General Affine Map

Definition: $m$-Dimensional Linear Map The state of the system is a vector $\vec{x}_t \in \mathbb{R}^m$. The evolution is given by the affine map:

$$\vec{x}_{t+1} = A\vec{x}_t + \vec{b}$$

where $A$ is an $m \times m$ square matrix and $\vec{b} \in \mathbb{R}^m$ is a constant offset vector.

2.2 Solving for Fixed Points in $m$ Dimensions

The definition of a fixed point remains the same: it is a point that is mapped onto itself.

Proof: Derivation of the $m$-Dimensional Fixed Point Let $\vec{x}^{\ast}$ be a fixed point. It must satisfy the condition $\vec{x}^{\ast} = A\vec{x}^{\ast} + \vec{b}$. We solve for $\vec{x}^{\ast}$:  

$$\vec{x}^{\ast} - A\vec{x}^{\ast} = \vec{b}$$

$$(I - A)\vec{x}^{\ast} = \vec{b}$$

where $I$ is the $m \times m$ identity matrix. If the matrix $(I-A)$ is invertible, we can find the unique fixed point by multiplying by its inverse:  

$$\vec{x}^{\ast} = (I-A)^{-1}\vec{b}$$

If $(I-A)$ is not invertible (i.e., it is singular), a unique fixed point does not exist. In this case, the system may have no fixed points or a continuous set of fixed points, such as a line attractor or a higher-dimensional manifold attractor.

2.3 System Dynamics and Diagonalization

To understand the system's trajectory, we analyze the behavior of the map when iterated over time. For simplicity, we first consider the homogeneous case where $\vec{b} = \vec{0}$.

Theorem: Iterated Map Dynamics For the system $\vec{x}_{t+1} = A\vec{x}_t$, the state at time $T$ is related to the initial state $\vec{x}_1$ by:  

$$\vec{x}_T = A^{T-1}\vec{x}_1$$ 

Remark/Intuition: The Role of Diagonalization Calculating matrix powers $A^{T-1}$ can be complex. However, if the matrix $A$ is diagonalizable, the calculation simplifies significantly. A diagonalizable matrix can be written as $A = V \Lambda V^{-1}$, where $V$ is the matrix of eigenvectors and $\Lambda$ is a diagonal matrix of the corresponding eigenvalues $\lambda_i$.

Proof: System Solution via Diagonalization If $A = V \Lambda V^{-1}$, then the power $A^{T-1}$ becomes:  

$$A^{T-1} = (V \Lambda V^{-1})^{T-1} = V \Lambda^{T-1} V^{-1}$$

Raising the diagonal matrix $\Lambda$ to a power is trivial; we simply raise each diagonal element (the eigenvalues) to that power. The solution for the system's state at time $T$ is therefore:  

$$\vec{x}_T = V \Lambda^{T-1} V^{-1} \vec{x}_1$$  

This expression reveals that the long-term behavior of the system is governed by the powers of the eigenvalues of $A$.

2.4 Stability Analysis via Eigenvalues

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

--------------------------------------------------------------------------------


Chapter 1: The Flow of Dynamical Systems

1.1 The Flow Operator in Linear Systems

We begin our formal study by introducing a concept central to understanding how systems evolve over time: the flow operator. Consider a system of linear ordinary differential equations with an initial value.

Definition: The Flow Map (or Flow Operator)

For a linear system of differential equations of the form $\dot{x} = Ax$, with an initial condition $x(0) = x_0$, the solution is given by:

$$x(t) = e^{At} x_0$$

The operator $e^{At}$ that propagates the initial state $x_0$ forward in time is known as the flow operator.

More generally, we can define a flow map, denoted by $\phi$, which is a function of time $t$ and an initial condition $x_0$:

$$\phi(t, x_0) = e^{At} x_0$$

Intuition

You can visualize the flow map as a mechanism that takes a set of initial conditions and "transports" it forward in time by an amount $t$ to a new location in the state space. If you vary the time $t$, the path traced out by a single initial point $x_0$ is called its orbit or trajectory.


--------------------------------------------------------------------------------


1.2 From Continuous to Discrete Time: The Sampling Equivalence

In many scientific and engineering contexts, particularly in physics, systems are naturally modeled using continuous-time differential equations. However, our observation and measurement of these systems are almost always discrete, taken at specific moments in time. This raises a crucial question: can we find a discrete-time system that is equivalent to a continuous-time one?

1.2.1 Equivalence for Linear Systems

Let us assume we have a continuous-time system that we sample at fixed time steps of duration $\Delta t$. The measurements are taken at times $0, \Delta t, 2\Delta t, \dots, n\Delta t$.

  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/flow_discretization.jpeg' | relative_url }}" alt="a" loading="lazy">
    <!-- <figcaption>Sketch of the Rice's theorem proof</figcaption> -->
  </figure>

The flow map for this system transports a state from one sample point to the next:

$$x((n+1)\Delta t) = \phi(\Delta t, x(n\Delta t))$$

We can define a new matrix that encapsulates this discrete-time evolution.

Definition: Discrete-Time Equivalent Matrix

Let a continuous-time linear system be defined by $\dot{x} = Ax$. Its equivalent discrete-time evolution matrix, $\tilde{A}$, for a sampling time step $\Delta t$ is defined as:

$$\tilde{A} = e^{A \Delta t}$$

With this definition, we can construct a discrete-time linear map:

$$x_{n+1} = \tilde{A} x_n$$

where $x_n$ represents the state at time $t = n \Delta t$.

Remark

This linear map is equivalent to the continuous-time linear ODE system in a specific sense: for the same initial condition $x_0$, the solutions of the discrete and continuous systems agree exactly at the sampling points $t = n \Delta t$. The construction of $\tilde{A}$ ensures this correspondence, as it is precisely the flow operator for a duration of $\Delta t$.

1.2.2 Equivalence for Affine Systems

This concept of equivalence can be extended to affine systems of differential equations, which include a constant offset term.

Consider the continuous-time affine system:

$$\dot{x} = Ax + c$$ 

We seek an equivalent discrete-time affine system of the form:

$$x_{n+1} = \tilde{A} x_n + b$$

where $\tilde{A}$ is defined as before: $\tilde{A} = e^{A \Delta t}$.

Remark: Determining the Discrete Offset

To find the corresponding offset vector $b$, we can enforce the condition that the fixed points of both the continuous and discrete systems must be identical.

1. Find the fixed point of the continuous system: Set $\dot{x} = 0. 0 = Ax^{\ast} + c \implies x^{\ast} = -A^{-1}c$
2. Find the fixed point of the discrete system: Set $x_{n+1} = x_n = x^{\ast}$. $x^{\ast} = \tilde{A}x^{\ast} + b \implies (I - \tilde{A})x^{\ast} = b$
3. Equate and Solve for $b$: By substituting the expression for $x^{\ast}$ from the continuous system into the discrete system's fixed point equation, we can solve for $b$.


--------------------------------------------------------------------------------


1.3 Applications and Advanced Concepts

The principles of establishing equivalence between continuous and discrete systems are not merely theoretical exercises. They have profound implications in modern machine learning and computational neuroscience.

1.3.1 Piecewise Linear Recurrent Neural Networks

An important class of models in machine learning is the Piecewise Linear Recurrent Neural Network (PL-RNN). These networks are often defined using the Rectified Linear Unit (ReLU) activation function, which is a piecewise linear function.

A typical PL-RNN update rule has the form:

$$x_{n+1} = Ax_{n} + Wg(x_n) + b$$

where $g$ is the ReLU nonlinearity, defined as $g(z) = \max(0, z)$.

The ideas of state-space dynamics and the equivalence between continuous and discrete forms can be extended to analyze these powerful computational models. For those interested in the details of this connection, the following resources are recommended:
* A paper by Monfared and Durstewitz presented at ICML 2020.
* The book Time Series Analysis (2013) by Ozaki, which contains a chapter on defining equivalent formulations for some nonlinear systems.

1.3.2 Line Attractors, Time Constants, and Memory

Let's revisit the concept of a line attractor, a continuous set of neutrally stable equilibria. In a 2D system with variables $z_1$ and $z_2$, a line attractor can arise when the nullclines (lines where $\dot{z}_1 = 0$ or $\dot{z}_2 = 0$) precisely overlap.

Intuition: Detuning for Arbitrary Time Constants

What happens if we slightly "detune" the system, so the nullclines no longer perfectly overlap but are very close? The vector field, which was exactly zero on the line attractor, will now be non-zero but very small in the "channel" between the slightly separated nullclines.

This has a profound consequence: by making subtle changes to the system's parameters (e.g., the slopes of the nullclines), we can create dynamics that evolve on arbitrarily long time scales. The system can be made to move extremely slowly without introducing any large physical time constants. This ability to generate a wide range of temporal scales is fundamental for complex information processing.

Example: The Addition Problem in Machine Learning

A classic benchmark for recurrent neural networks is the addition problem. The network receives two input streams:
1. A sequence of real numbers between 0 and 1.
2. A binary indicator bit (0 or 1).

The task is for the network to sum the real numbers only when the corresponding indicator bit is 1. The challenge lies in the potentially long gaps between periods where the indicator is active. The network must store the intermediate sum in its memory.

A line attractor provides a simple and elegant solution. A two-unit PL-RNN can solve this task:
* Integration and Storage: One unit integrates the input values (when the indicator bit is active) and stores the running total as a state on a line attractor. The system's state remains stable on this line, effectively acting as a memory device.
* Final Output: Once the sequence is complete, the final state on the line attractor represents the total sum.

Example: Attractors in Natural Intelligence This is not merely a machine learning construct. There is evidence for the existence of line attractors, plane attractors, and even torus attractors (shaped like a donut) in biological brains, for example, in the hippocampus, an area critical for memory and navigation.


--------------------------------------------------------------------------------


1.4 Formal Definition of a Dynamical System
Chapter 1: The Flow Map and Trajectories

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

1.1 The Flow Operator

<div class="math-callout math-callout--theorem" markdown="1">
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

* Remark/Intuition: Imagine a particle tracing a path in the state space. It should not matter whether you calculate its position after 5 seconds by moving it forward 2 seconds and then 3 seconds, or by moving it 3 seconds and then 2 seconds. The final position must be the same as moving it forward for 5 seconds directly. This consistency is fundamental.

1.3 Trajectories and Orbits

With the flow operator established, we can now precisely define the path that a point carves out in the state space over time.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Trajectory or Orbit)</span></p>

The **trajectory (or orbit)** of a dynamical system starting from an initial point $x_0$ is the solution curve, denoted $\gamma(x_0)$. It is the set of all points in the state space that lie on this solution curve for all time $t \in T$.

$$\gamma(x_0) = \lbrace \phi_t(x_0) \mid t \in T \rbrace$$

</div>

Remark/Intuition: A critical feature of a well-defined dynamical system is the uniqueness of its trajectories. For any given initial point $x_0$, there can be only one trajectory passing through it. If two different curves could originate from the same starting point, it would imply that the state space is missing crucial information needed to predict the future state, and we would not have a deterministic dynamical system. We will explore the conditions that guarantee this uniqueness in the next chapter.


Chapter 2: Existence and Uniqueness of Solutions

Having defined the concepts of flows and trajectories, a fundamental question arises: given a system of differential equations, can we always expect it to have a unique solution for a given starting condition? This is the central question of existence and uniqueness.

2.1 The Core Problem: Do Unique Solutions Always Exist?

The unfortunate answer is no, unique solutions are not guaranteed for all systems. However, the fortunate reality is that for the vast majority of well-behaved systems, they almost always do. The conditions where uniqueness fails are quite specific.

Examples: Consider the following initial value problem:

$$\dot{x} = 3x^{2/3}, \quad x(0) = 0$$

This system has two distinct solutions that satisfy the initial condition:

1. The Trivial Solution: $u(t) = 0$
2. A Non-trivial Solution: $v(t) = t^3$

Proof: We must verify that both functions satisfy the differential equation and the initial condition.

* For $u(t) = 0$:
  * Initial Condition: $u(0) = 0$, which is satisfied.
  * Differential Equation: The time derivative is $\dot{u}(t) = 0$. Plugging into the equation gives $0 = (0)^{2/3} = 0$. The equation holds.
* For $v(t) = t^3$:
  * Initial Condition: $v(0) = 0^3 = 0$, which is satisfied.
  * Differential Equation: The time derivative is $\dot{v}(t) = 3t^2 = 3v^{2/3}$. 
  <!-- We check if this equals $v(t)^{2/3}$. -->
<!-- * Since the derivative $3t^2$ is not equal to $t^2$, there appears to be a transcription error in the original lecture notes. Let's re-examine the example with a slight correction to match the lecture's conclusion. Let's assume the example was meant to be $x' = 3x^{2/3}$ or the solution was $v(t) = (t/3)^3$. Assuming the intended solution was $v(t) = (t/3)^3$ for $\dot{x}=x^{2/3}$, the derivative would be $\dot{v}(t) = 3(t/3)^2 \cdot (1/3) = t^2/9$, while $v^{2/3} = ((t/3)^3)^{2/3} = (t/3)^2 = t^2/9$. This seems more plausible. Let's proceed with the lecture's reasoning about the function's smoothness, which is the core lesson. -->

Remark/Intuition: What causes this failure of uniqueness? The problem lies at the point $x=0$. The vector field $f(x) = 3x^{2/3}$ is continuous at $x=0$, but it is not continuously differentiable there. Let's examine its derivative with respect to the dynamical variable $x$:
 
 $$\frac{df}{dx} = \frac{d}{dx}(3x^{2/3}) = 2x^{-1/3}$$
 
This derivative is undefined at $x=0$. This lack of smoothness in the vector field is precisely what allows for multiple solution paths to emerge from the same point.

2.2 The Fundamental Existence and Uniqueness Theorem

The issue identified in the counterexample is the exact problem that the following powerful theorem resolves. If we can guarantee that our vector field is smooth enough, we can guarantee a unique solution.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Fundamental Existence and Uniqueness Theorem)</span></p>

Let $E \subseteq \mathbb{R}^m$ be an open set (our state space) and let the vector field $f: E \to \mathbb{R}^m$ be a continuously differentiable function (i.e., $f \in C^1(E)$).

Then, for any initial condition $x_0 \in E$, there exists a constant $a > 0$ such that the initial value problem  

$$\dot{x} = f(x), \quad x(0) = x_0$$

has a unique solution, $x(t)$, within the so-called maximum interval of existence $(-a, a)$, which is a subset of $\mathbb{R}$.

Furthermore, this unique solution has the general form:

$$x(t) = x_0 + \int_0^t f(x(s))ds$$     

This integral expression is sometimes referred to as the solution operator.

</div>

Remark/Intuition: This theorem is the bedrock for much of dynamical systems theory. It tells us that as long as our system's rules of evolution (the vector field $f$) are smooth, we don't have to worry about non-uniqueness. The solution might not exist for all time (it could "blow up" in finite time), but in some local time interval around our starting point, the path is uniquely determined. While the integral form of the solution is general, it may not be solvable analytically and often requires a numerical solver.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A Weaker Condition: Lipschitz Continuity)</span></p>

The requirement of being continuously differentiable ($C^1$) is sufficient, but it is actually stronger than necessary. A weaker, more general condition also guarantees uniqueness.

The theorem can be proven under the weaker assumption that the function $f$ is locally Lipschitz continuous. For any two points $x$ and $y$ in some local interval, a function is Lipschitz continuous if the absolute difference in its values is bounded by a constant multiple of the distance between the points.  

$$\lvert f(x) - f(y)\rvert \le L \lvert x - y\rvert$$

Here, $L$ is a positive real number known as the Lipschitz constant. Intuitively, this condition means that the slope of the function is bounded. Every continuously differentiable function is locally Lipschitz, but not every Lipschitz continuous function is differentiable, making this a more general condition.

</div>


2.4 On Solving Non-Linear Systems

While the theorem guarantees the existence of a unique solution for a broad class of systems, it does not provide a general method for finding it.

Remark/Intuition: In the most general case, non-linear systems of differential equations cannot be solved analytically. However, for certain scalar cases or systems with special structures, analytical techniques exist. These include:

* Separation of Variables: Rearranging the equation so that all terms involving one variable are on one side and all terms involving the other variable are on the other side, allowing for direct integration.
* Variational Calculus: A more advanced method for solving certain classes of problems.

For most complex systems encountered in practice, numerical methods are the primary tool for approximating the unique solution trajectories that the theorem guarantees.

# Lecture 3

Nonlinear ODE systems, homeomorphisms & topological equivalence, Hartman–Grobman theorem. (29.10.25)

With example nonlinear flow on a line, 1d-vector field, introduce:
  Bistability/multistability
  Basins of attraction
  Point attractors
  Stability of equilibria for nonlinear systems (Taylor expansion of perturbation, Jacobian)
  Def. "à la Strogatz" for stability in terms of eigenvalues of Jacobian
2d case: phase plane
  Ex. Lotka–Volterra system
  Interpretation as feedback loops
  stable/unstable manifolds
  heteroclinic orbits
More formal and general (including non-hyperbolic DS) definitions ("à la Perko"):
  (asymptotic) stability
  homeomorphisms
  (topological) manifolds
  stable/unstable manifold (local, global)
  homoclinic, heteroclinic orbits (+ mention of separatrix cycles)
  invariant sets
  topological equivalence
  almost-example
  Theorem on eigenvalues of Jacobians of topologically equivalent objects
  topological conjugacy
  true example
Hartman–Grobman theorem

A Study Book on Dynamical Systems

Table of Contents

I. Introduction to Nonlinear Systems

* Recap: Linear Systems and Core Concepts
* 1.1. Flow on a Line: A First Look at Nonlinear Dynamics
* 1.2. Graphical Analysis of Equilibria
* 1.3. Bistability and Basins of Attraction
* 1.4. Formal Stability Analysis: The Power of Linearization
* 1.5. Classification of Equilibria in Nonlinear Systems

II. Two-Dimensional Systems: The Phase Plane (Preview)

* 2.1. The Lotka-Volterra System


--------------------------------------------------------------------------------


I. Introduction to Nonlinear Systems

This chapter transitions from the world of linear systems to the more complex and richer domain of nonlinear systems. While linear systems provide a foundational understanding of dynamics, many real-world phenomena exhibit behaviors that can only be captured by nonlinear models. Today's focus will be heavy on definitions, as these form the essential vocabulary needed to explore one of the most fundamental theorems in dynamical systems theory.

Recap: Linear Systems and Core Concepts

Before delving into nonlinear systems, let us briefly recall the key concepts covered previously:

* Linear Systems: The general solution to linear systems of ordinary differential equations of the form $\dot{\mathbf{x}} = A\mathbf{x}$ has been thoroughly discussed.
* Dynamical System & Flow: A dynamical system is characterized by its flow operator, denoted as $\phi_t(x_0)$. This function describes the evolution of a state $x_0$ over a time $t$. The flow operator possesses crucial group or semigroup properties. For systems of differential equations defined by a vector field $\mathbf{f}$, the flow can be seen as the solution operator.
* Existence and Uniqueness: The Picard-Lindelöf theorem guarantees that for a given initial condition, a unique solution exists, provided the vector field $\mathbf{f}$ is continuously differentiable or at least locally Lipschitz.

With these foundations, we can now begin our exploration of systems where the governing equations are nonlinear.

1.1. Flow on a Line: A First Look at Nonlinear Dynamics

To introduce the core properties of nonlinear systems, we will begin with the simplest case: a one-dimensional system, also known as a flow on a line ($\mathbb{R}$).


--------------------------------------------------------------------------------


Example: A Cubic Vector Field

Let's consider a one-dimensional nonlinear dynamical system whose vector field is defined by a third-order polynomial:

$$\dot{x} = f(x) = x - x^3$$

Here, $x$ is a state on the real number line, and its rate of change, $\dot{x}$, is given by the nonlinear function $f(x)$. The specific form of the function is less important than the general properties it illustrates.


--------------------------------------------------------------------------------


1.2. Graphical Analysis of Equilibria

A powerful technique for understanding 1D systems is to create a state space portrait by plotting the vector field $f(x)$ (i.e., the derivative $\dot{x}$) as a function of the state $x$.

For our example, $f(x) = x - x^3$, the plot is a cubic function that looks something like this:

(Conceptual sketch: A curve starting from the top-left, crossing the $x$-axis at -1, peaking, crossing the $x$-axis at 0, reaching a trough, and crossing the $x$-axis at +1 before heading to the bottom-right.)

From this simple graph, we can extract a wealth of information about the system's dynamics.


--------------------------------------------------------------------------------


Definition: Equilibrium (Fixed Point)

An equilibrium, also known as a fixed point or critical point, of a dynamical system $\dot{x} = f(x)$ is a state $x^{\ast}$ where the vector field vanishes.

$$f(x^{\ast}) = 0$$

At an equilibrium, the derivative is zero ($\dot{x}^{\ast} = 0$), meaning the state does not change over time.


--------------------------------------------------------------------------------


Remark/Intuition: Finding and Classifying Equilibria Graphically

On the plot of $f(x)$ versus $x$, the equilibria are simply the points where the graph crosses the horizontal axis (i.e., where $f(x)=0$).

* Direction of Flow: The sign of $f(x)$ tells us the direction of movement.
  * If $f(x) > 0$, then $\dot{x} > 0$, and the state x moves to the right (increases).
  * If $f(x) < 0$, then $\dot{x} < 0$, and the state x moves to the left (decreases).
* Stability: By observing the flow around each fixed point, we can determine its stability.
  * A fixed point is stable if nearby trajectories move toward it. In the graphical analysis, this corresponds to a point where the function $f(x)$ crosses the axis with a negative slope.
  * A fixed point is unstable if nearby trajectories move away from it. This corresponds to a point where $f(x)$ crosses the axis with a positive slope.

Applying this to our example $f(x) = x - x^3$, we find three equilibria. The two outer points are stable, as the flow converges toward them from both sides. The central point is unstable, as the flow moves away from it.

1.3. Bistability and Basins of Attraction

The presence of multiple isolated, stable fixed points is a hallmark of nonlinear systems that is impossible in linear ones.

* A linear system can have either a single isolated fixed point or a continuous line or manifold of neutrally stable points. It cannot have two or more isolated stable equilibria.
* This property of having two stable states is called bistability. With more than two, it is called multistability. This behavior is fundamentally important in fields like recurrent neural networks and machine learning.

Each stable equilibrium, or attractor, governs a region of the state space.


--------------------------------------------------------------------------------


Definition: Basin of Attraction

The basin of attraction for an attractor (such as a stable fixed point) is the set of all initial conditions in the state space from which trajectories converge to that attractor as $t \to \infty$.


--------------------------------------------------------------------------------


Remark/Intuition: Basins in the 1D Example

In our example, the unstable fixed point at the center acts as a boundary separating the domains of the two stable fixed points. Let's say the unstable fixed point is located at a value we call $-\alpha$.

* The basin of attraction for the stable fixed point on the right is the interval $(-\alpha, +\infty)$. Any initial condition within this interval will eventually converge to this rightmost fixed point.
* Similarly, the stable fixed point on the left has its own basin of attraction, bounded on the right by $-\alpha$.

The stable fixed points themselves are referred to as point attractors.

1.4. Formal Stability Analysis: The Power of Linearization

While graphical analysis is insightful for 1D systems, we need a more general, analytical method to determine the stability of fixed points, especially in higher dimensions. The key technique is linearization, which involves analyzing the system's behavior in the immediate vicinity of a fixed point.

Let's assume $x^{\ast}$ is a fixed point of the system $\dot{x} = f(x)$. To examine its stability, we introduce a small perturbation, $\epsilon$, and observe whether it grows or decays.

Let $x = x^{\ast} + \epsilon$. The dynamics of this perturbed state are given by:

$$\dot{x} = \frac{d}{dt}(x^{\ast} + \epsilon) = f(x^{\ast} + \epsilon)$$

Now, we perform a Taylor series expansion of $f(x)$ around the fixed point $x^{\ast}$. We are interested in the behavior for very small $\epsilon$, so we only keep the linear terms.

$$f(x^{\ast} + \epsilon) \approx f(x^{\ast}) + \frac{df}{dx}\bigg|_{x=x^{\ast}} \cdot \epsilon + O(\epsilon^2)$$

In higher dimensions, this becomes:

$$\mathbf{f}(\mathbf{x}^{\ast} + \boldsymbol{\epsilon}) \approx \mathbf{f}(\mathbf{x}^{\ast}) + J(\mathbf{x}^{\ast}) \boldsymbol{\epsilon} + O(\|\boldsymbol{\epsilon}\|^2)$$


Here, $J$ is the Jacobian matrix.


--------------------------------------------------------------------------------


Definition: The Jacobian Matrix

For a vector-valued function $\mathbf{f}(\mathbf{x})$ with components $f_i$ and variables $x_j$, the Jacobian matrix $J$ evaluated at a point $\mathbf{x}_0$ is the matrix of all first-order partial derivatives:

$$
J(\mathbf{x}_0) = D\mathbf{f}(\mathbf{x}_0) = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots \\
\vdots & \vdots & \ddots
\end{pmatrix}_{\mathbf{x}=\mathbf{x}_0}
$$

In the 1D case, the Jacobian is simply the scalar derivative $f'(x_0)$.


--------------------------------------------------------------------------------


Now we can formulate a differential equation for the perturbation $\boldsymbol{\epsilon}$.

$$\dot{\boldsymbol{\epsilon}} = \dot{\mathbf{x}} - \dot{\mathbf{x}}^{\ast}$$

By definition of a fixed point, $\mathbf{x}^{\ast}$ is constant, so $\dot{\mathbf{x}}^{\ast} = 0$. Furthermore, $\mathbf{f}(\mathbf{x}^{\ast}) = 0$. Substituting our Taylor expansion for $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}^{\ast} + \boldsymbol{\epsilon})$, we get:

$$\dot{\boldsymbol{\epsilon}} \approx (\mathbf{f}(\mathbf{x}^{\ast}) + J(\mathbf{x}^{\ast}) \boldsymbol{\epsilon}) - 0$$

$$\dot{\boldsymbol{\epsilon}} \approx J(\mathbf{x}^{\ast}) \boldsymbol{\epsilon}$$

This is a linear system of differential equations describing the evolution of the perturbation. We have thus linearized the dynamics around the fixed point. The stability of the original nonlinear system, in the local vicinity of $x^{\ast}$, is determined by the stability of this linear system.

1.5. Classification of Equilibria in Nonlinear Systems

By analyzing the eigenvalues of the Jacobian matrix $J$ evaluated at the equilibrium point $\mathbf{x}_0$, we can classify its stability, just as we did for fully linear systems.


--------------------------------------------------------------------------------


Definition: Stability of an Equilibrium Point

Let $\mathbf{x}_0$ be an equilibrium point of the system $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$, and let $J = J(\mathbf{x}_0)$ be the Jacobian evaluated at that point. Let $\lbrace\lambda_i\rbrace$ be the set of eigenvalues of $J$.

* The equilibrium \mathbf{x}_0 is stable if all eigenvalues of $J$ have a negative real part.
  * $Re(\lambda_i) < 0$ for all $i$.
* The equilibrium $\mathbf{x}_0$ is unstable if there is at least one eigenvalue of $J$ with a positive real part.
  * There exists at least one $i$ such that $Re(\lambda_i) > 0$.
* The equilibrium $\mathbf{x}_0$ is a saddle point if there is at least one eigenvalue with a negative real part and at least one eigenvalue with a positive real part.
  * There exist $i, j$ such that $Re(\lambda_i) < 0$ and $Re(\lambda_j) > 0$.


--------------------------------------------------------------------------------


Remark/Intuition: Clarifications and Edge Cases

* Saddle vs. Unstable: It is a very good point that the definition of an unstable point includes saddle points. Some authors may use "unstable" more strictly to refer to a point where all trajectories move away (i.e., all eigenvalues have positive real parts), but the broader definition provided above is common. A saddle is unstable because trajectories will diverge along at least one direction.
* The Non-Hyperbolic Case: What happens if one or more eigenvalues have a real part that is exactly zero? In this case, the linearization fails. The terms we ignored in the Taylor expansion (O($\epsilon^2$) and higher) become critical in determining the stability. Our linear approximation is not good enough to make a conclusion, and a more advanced analysis involving the higher-order terms is required.

A Student's Guide to Dynamical Systems

Table of Contents

* Chapter 1: An Introduction to Nonlinear Systems and Stability
  * 1.1 The Nature of Nonlinear Systems
  * 1.2 System Archetype: The Predator-Prey Model
  * 1.3 Finding Equilibria: The System's Stationary States
  * 1.4 Stability Analysis via Linearization
  * 1.5 Case Study: Characterizing the Equilibria of a Predator-Prey System
  * 1.6 Formal Definitions of Stability


--------------------------------------------------------------------------------


Chapter 1: An Introduction to Nonlinear Systems and Stability

Welcome to the fascinating world of dynamical systems. In this chapter, we will move beyond the predictable realm of linear systems to explore the rich and often surprising behavior of nonlinear systems. We will begin by introducing a classic biological model to build our intuition, then develop the core mathematical tools—finding equilibrium points and analyzing their stability—that are essential for understanding any dynamical system.

1.1 The Nature of Nonlinear Systems

Remark/Intuition

A fundamental challenge and a source of rich complexity in the study of nonlinear systems is that we can no longer expect to find explicit, closed-form solutions for the state of the system as a function of time, i.e., an equation for $x(t)$. Unlike linear systems, where such solutions are often readily available, nonlinear systems require a different, more qualitative approach. Our goal will be to understand the overall behavior and geometry of the system's evolution in its state space without necessarily solving the equations of motion explicitly.

It is remarkable how frequently similar mathematical structures appear across disparate scientific fields. Simple polynomial systems with product terms, which we will study here, are not just academic curiosities. They are foundational in describing:

* Atmospheric Convection: Simple climate models often reduce to this form.
* Chemical Reaction Systems: The rate of reaction between two species, $X$ and $Y$, is often proportional to the likelihood of them encountering each other, leading to product terms like $XY$.
* Epidemiology: Models for infectious diseases, such as the Susceptible-Infected-Recovered (SIR) model for epidemics like COVID, use these very types of equations to describe the dynamics of a population.

Recognizing these patterns allows us to transfer insights from one field to another, which is a powerful aspect of dynamical systems theory.

1.2 System Archetype: The Predator-Prey Model

To ground our discussion, we will use a classic model from population dynamics that describes the interaction between a population of predators (e.g., foxes) and a population of prey (e.g., rabbits).

Example: The Lotka-Volterra Equations

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

Remark/Intuition: Systems as Feedback Loops

It is often conceptually useful to visualize such systems as a network of interacting feedback loops. This perspective was central to the development of Cybernetics in the 1940s and 50s.

For our predator-prey model:

* The prey population $x$ promotes its own growth (a positive feedback, via $\alpha$).
* The prey population $x$ promotes the growth of the predator population $y$ (via $\gamma$).
* The predator population $y$ inhibits the growth of the prey population $x$ (a negative feedback, via $\beta$).
* The predator population $y$ promotes its own decline in the absence of food (a negative feedback, via $\lambda$).

While Cybernetics as a distinct field is no longer prominent, the mathematical tools and the conceptual framework of analyzing systems through their feedback structures remain invaluable.

1.3 Finding Equilibria: The System's Stationary States

The first step in analyzing any dynamical system is to find its equilibrium points (also known as fixed points). These are the points in the state space where the system is stationary—that is, where all rates of change are zero.

Definition: Equilibrium Point

An equilibrium point $x_0$ of a dynamical system $\dot{x} = f(x)$ is a point where the vector field is zero.  

$$f(x_0) = 0$$  

For our 2D system, this corresponds to the condition:  

$$\frac{dx}{dt} = 0 \quad \text{and} \quad \frac{dy}{dt} = 0$$ 

Example: Calculating Equilibria for the Predator-Prey Model

We set the equations of motion to zero:

$$\begin{align*} x(\alpha - \beta y) &= 0 \\ y(\gamma x - \lambda) &= 0 \end{align*}$$  

From this system, we can identify two distinct solutions.

1. The Trivial Equilibrium: If we set $x=0$ and $y=0$, both equations are satisfied.
   
   $$(x^, y^) = (0, 0)$$
   
   This corresponds to the extinction of both species.
2. The Coexistence Equilibrium: Assuming $x \neq 0 and y \neq 0$, we can divide by them to solve the parenthetical terms:
   
  $$\begin{align*} \alpha - \beta y &= 0 \implies y = \frac{\alpha}{\beta} \\ \gamma x - \lambda &= 0 \implies x = \frac{\lambda}{\gamma} \end{align*}$$  
   
  This gives us the second equilibrium point:
  
  $$(x^, y^) = \left(\frac{\lambda}{\gamma}, \frac{\alpha}{\beta}\right)$$  
  
  This corresponds to a state where both predator and prey populations coexist in a stable balance.

1.4 Stability Analysis via Linearization

Once we have found the equilibrium points, the next crucial question is: what happens to the system if it starts near one of these points? Will it return to the equilibrium, or will it be repelled? This is the question of stability. The primary tool for answering this is linearization.

Remark/Intuition

The core idea is to approximate the complex nonlinear system with a simpler linear system in the immediate vicinity of an equilibrium point. The behavior of this local linear system is determined by the Jacobian matrix, and its properties (specifically, its eigenvalues) tell us almost everything we need to know about the stability of the equilibrium point for the original nonlinear system.

Definition: The Jacobian Matrix

For a 2D system given by $\dot{x} = f_1(x, y)$ and $\dot{y} = f_2(x, y)$, the Jacobian matrix $J$ is the matrix of all first-order partial derivatives:

$$J(x, y) = \begin{pmatrix} \frac{\partial f_1}{\partial x} & \frac{\partial f_1}{\partial y} \\ \frac{\partial f_2}{\partial x} & \frac{\partial f_2}{\partial y} \end{pmatrix}$$

Example: The Jacobian for the Predator-Prey Model

Our system is:

$$\begin{align*} f_1(x, y) &= \alpha x - \beta xy \\ f_2(x, y) &= \gamma xy - \lambda y \end{align*} $$ 

Let's compute the partial derivatives:

* $\frac{\partial f_1}{\partial x} = \alpha - \beta y$
* $\frac{\partial f_1}{\partial y} = -\beta x$
* $\frac{\partial f_2}{\partial x} = \gamma y$
* $\frac{\partial f_2}{\partial y} = \gamma x - \lambda$

Assembling these into the Jacobian matrix gives:

$$J(x, y) = \begin{pmatrix} \alpha - \beta y & -\beta x \ \gamma y & \gamma x - \lambda \end{pmatrix}$$

To analyze the stability of an equilibrium point $(x^{\ast}, y^{\ast})$, we evaluate this matrix at that specific point and then find its eigenvalues.

1.5 Case Study: Characterizing the Equilibria of a Predator-Prey System

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

Analysis of the Equilibrium at (0, 0)

Proof

First, we evaluate the general Jacobian matrix at the point $(x, y) = (0, 0)$:  

$$J(0, 0) = \begin{pmatrix} \alpha - \beta(0) & -\beta(0) \\ \gamma(0) & \gamma(0) - \lambda \end{pmatrix} = \begin{pmatrix} \alpha & 0 \\ 0 & -\lambda \end{pmatrix}$$  

Plugging in our specific parameter values $(\alpha=3, \lambda=-2)$:  

$$J(0, 0) = \begin{pmatrix} 3 & 0 \\ 0 & -(-2) \end{pmatrix} = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}$$  

The eigenvalues of a diagonal matrix are simply its diagonal entries. Therefore, the eigenvalues are $\lambda_1 = 3$ and $\lambda_2 = 2$.

Since both eigenvalues are real and positive, trajectories starting near the origin will be repelled from it along all directions. This type of equilibrium is called an unstable node.

Analysis of the Equilibrium at (2, 3)

Proof

Next, we evaluate the Jacobian at the coexistence equilibrium $(x, y) = (2, 3)$ using our parameters $(\alpha=3, \beta=1, \gamma=-1, \lambda=-2)$:  

$$J(2, 3) = \begin{pmatrix} \alpha - \beta(3) & -\beta(2) \\ \gamma(3) & \gamma(2) - \lambda \end{pmatrix} = \begin{pmatrix} 3 - 1(3) & -1(2) \\ -1(3) & -1(2) - (-2) \end{pmatrix}$$   

$$J(2, 3) = \begin{pmatrix} 0 & -2 \\ -3 & 0 \end{pmatrix}$$  

To find the eigenvalues, we solve the characteristic equation $\text{det}(J - \lambda I) = 0$:  

$$\text{det} \begin{pmatrix} -\lambda & -2 \\ -3 & -\lambda \end{pmatrix} = (-\lambda)(-\lambda) - (-2)(-3) = \lambda^2 - 6 = 0$$  

This gives $\lambda^2 = 6$, so the eigenvalues are $\lambda_1 = +\sqrt{6} \approx 2.45$ and $\lambda_2 = -\sqrt{6} \approx -2.45$.

Since we have one positive real eigenvalue and one negative real eigenvalue, trajectories are attracted towards the equilibrium along one direction (the stable direction) and repelled from it along another direction (the unstable direction). This quintessential feature defines a saddle point.

Remark/Intuition: Manifolds and Heteroclinic Orbits

The qualitative behavior of this system is quite structured.

* The set of points that converge to the saddle point at (2,3) is called its stable manifold. In this case, it appears as a curve passing through the saddle.
* The set of points that are repelled from the saddle point is its unstable manifold, another curve passing through the saddle.
* An especially interesting feature in some systems is when an orbit starts at one equilibrium point and ends at another. In our example, the unstable manifold of the unstable node at (0,0) connects to the stable manifold of the saddle at (2,3). An orbit that connects two different fixed points is called a heteroclinic orbit. Such structures are not mere curiosities; they play important roles in phenomena like computation in neural systems.

(We use the term manifold loosely for now; it has a precise mathematical definition that we will explore later.)

1.6 Formal Definitions of Stability

Our analysis using eigenvalues is powerful, but it relies on linearization. To build a more robust foundation, we need formal definitions that do not depend on this approximation. These definitions, often associated with mathematicians like Perko, are based on the behavior of trajectories in neighborhoods around the equilibrium point.

Let $x_0$ be an equilibrium point and let $\phi_t(x)$ be the flow operator, which maps a starting point $x$ to its position at time $t$.

Definition: Stable Equilibrium (in the sense of Lyapunov)

An equilibrium point $x_0$ is called stable if for every neighborhood $\mathcal{U}$ of $x_0$ (e.g., an open ball of radius $\epsilon > 0$), there exists a smaller neighborhood $\mathcal{V}$ of $x_0$ (e.g., a ball of radius $\delta > 0$) such that any trajectory starting in $\mathcal{V}$ remains within $\mathcal{U}$ for all future time.

Formally: For every $\epsilon > 0$, there exists a $\delta > 0$ such that for any point $x$ with $\lvert x - x_0\rvert < \delta$, we have $\lvert\phi_t(x) - x_0\rvert < \epsilon$ for all $t \ge 0$.

Intuition: "If you start close enough, you stay close enough." Note that this does not require the trajectory to approach $x_0$. A center in a frictionless pendulum system is stable, as trajectories that start nearby will orbit it, staying close but never converging.

Definition: Asymptotically Stable Equilibrium

An equilibrium point $x_0$ is asymptotically stable if:

1. It is stable.
2. There exists a neighborhood $\mathcal{W}$ of $x_0$ such that every trajectory starting in $\mathcal{W}$ converges to $x_0$ as time goes to infinity.

Formally: It is stable, and there exists $a \eta > 0$ such that if $\lvert x - x_0\rvert < \eta$, then

$$\lim_{t \to \infty} \phi_t(x) = x_0$$

Intuition: "If you start close enough, you eventually arrive at the equilibrium." Sinks and stable spirals are asymptotically stable.

This more general framework is crucial as it correctly classifies cases where linearization is inconclusive, such as when the eigenvalues of the Jacobian have zero real parts.


A Student's Guide to Dynamical Systems: Topology and Manifolds

Table of Contents

1. Foundational Concepts in Topological Dynamics
  * 1.1. Asymptotic Stability: A General Definition
  * 1.2. Homeomorphisms and Topological Equivalence
  * 1.3. An Introduction to Manifolds
2. Stable and Unstable Manifolds
  * 2.1. The Local Perspective: Neighborhoods of an Equilibrium
  * 2.2. The Global Perspective: Extending Local Manifolds
3. Advanced Orbit Structures (Preview)
  * 3.1. Homoclinic and Heteroclinic Orbits


--------------------------------------------------------------------------------


1. Foundational Concepts in Topological Dynamics

This chapter introduces the fundamental topological concepts that are essential for a deeper understanding of dynamical systems. We will move beyond simple classifications to establish a rigorous framework for comparing different systems and analyzing the geometric structures they produce.

1.1. Asymptotic Stability: A General Definition

Remark/Intuition

Previously, definitions of stability may have been restricted to hyperbolic dynamical systems—systems where the real parts of the eigenvalues of the linearized system at an equilibrium point are all non-zero. That definition, while useful, is not universally applicable. We introduce a more general definition of asymptotic stability that also covers non-hyperbolic systems, providing a more robust tool for analysis.

Definition: Asymptotic Stability

An equilibrium point $x_0$ is asymptotically stable if there exists a neighborhood around it such that any trajectory starting within that neighborhood converges to $x_0$.

Formally, there exists a $ \delta > 0$ such that for any initial condition $x$ within the open neighborhood $N_\delta(x_0)$, the trajectory starting at $x$ converges to $x_0$:  

$$\forall x \in N_\delta(x_0), \quad \lim_{t \to \infty} \phi_t(x) = x_0$$  

where $\phi_t(x)$ is the flow of the dynamical system.

1.2. Homeomorphisms and Topological Equivalence

Remark/Intuition

A central task in the study of dynamical systems is the ability to compare them. We need a "recipe" to determine if two different systems are, in some essential way, the same. This is not just an academic exercise; it is crucial for modern applications. For instance, when machine learning algorithms attempt to infer the governing equations of a system from data, we want the reconstructed system to preserve the fundamental properties of the original. The concept of topological equivalence, which is built upon the idea of a homeomorphism, provides the mathematical language for this.

Definition: Homeomorphism

Let $X$ be a metric space (a space endowed with a distance function), and let $A$ and $B$ be subsets of $X$.

A homeomorphism is a function $h: A \to B$ that satisfies the following three properties:

1. Continuity: The function h is continuous.
2. One-to-One and Onto (Bijective): The function h is a one-to-one map (injective) and maps onto the entire set $B$.
3. Continuous Inverse: The inverse function $h^{-1}: B \to A$ exists and is also continuous.

In essence, a homeomorphism is a continuous stretching and bending of space that defines a unique, reversible mapping between two sets, $A$ and $B$.

1.3. An Introduction to Manifolds

Remark/Intuition

The concept of a manifold formalizes the idea of a space that, on a local level, resembles standard Euclidean space. For any point within a manifold, we can find a small neighborhood around it that can be smoothly mapped to an open ball in $\mathbb{R}^n$. This ensures that every point is "surrounded" by other points in the set, a property that is crucial for defining smooth dynamics on these structures.

Definition: Manifold

An $n$-dimensional topological manifold is a topological space $X$ such that for each point $x_0 \in X$, there exists an open neighborhood of $x_0$ that is homeomorphic to an $n$-dimensional open Euclidean ball.

More formally, for each $x_0 \in X$, there is:

1. An open neighborhood $N_\epsilon(x_0)$ of $x_0$.
2. A homeomorphism $h: N_\epsilon(x_0) \to B^n$, where $B^n$ is the $n$-dimensional open unit ball.

The $n$-dimensional open unit ball is defined as the set of all points in an $n$-dimensional space whose distance from the origin is strictly less than one:  

$$B^n = \lbrace (x_1, \dots, x_n) \in \mathbb{R}^n \mid \sum_{i=1}^n x_i^2 < 1 \rbrace$$  

The "strictly less than" condition is what makes the set open, meaning for any point in the ball, we can find a small neighborhood around it that is still entirely contained within the ball.

Examples

* A manifold: A closed orbit (like a circle) is a simple example of a 1-dimensional manifold. For any point on the circle, you can find a small arc around it that is homeomorphic to an open interval in $\mathbb{R}^1$.
* Not a manifold: A line segment is not a manifold. Consider a point at one of the endpoints. No matter how small a neighborhood (an open ball) you draw around this endpoint, it will never be completely contained within the line segment. Therefore, the endpoint does not have a neighborhood homeomorphic to an open Euclidean ball.


--------------------------------------------------------------------------------


2. Stable and Unstable Manifolds

Stable and unstable manifolds are fundamental geometric structures in dynamical systems. They are the sets of all points that approach (or move away from) an equilibrium point, such as a fixed point or a periodic orbit. They play a huge role in characterizing the long-term behavior of a system.

2.1. The Local Perspective: Neighborhoods of an Equilibrium

Remark/Intuition

The construction of stable and unstable manifolds is a two-step process. First, we define them locally in a small neighborhood around an equilibrium point $x_0$. In this local region, the manifold's structure is determined by the linear dynamics—specifically, the stable and unstable directions given by the eigenvectors of the system's Jacobian at $x_0$. After defining this local structure, we can extend it to find the full, global manifold.

Definition: Local Stable Manifold

The local stable manifold of an equilibrium point $x_0$, denoted $W^s_{loc}(x_0)$, is the set of all points $x$ in a small neighborhood of $x_0$ that not only converge to $x_0$ as time goes to infinity but also remain within that neighborhood for all future time.

Formally, for a given neighborhood $N_\epsilon(x_0)$:  

$$W^s_{loc}(x_0) = \lbrace x \in N_\epsilon(x_0) \mid \lim_{t \to \infty} \phi_t(x) = x_0 \quad \text{and} \quad \phi_t(x) \in N_\epsilon(x_0) \text{ for all } t \ge 0 \rbrace$$ 

Definition: Local Unstable Manifold

The local unstable manifold of an equilibrium point $x_0$, denoted $W^u_{loc}(x_0)$, is defined analogously but using reverse time. It is the set of all points $x$ in a small neighborhood of $x_0$ that converge to $x_0$ as time goes to negative infinity and remain within that neighborhood for all past time.

Formally, for a given neighborhood $N_\epsilon(x_0)$:  

$$W^u_{loc}(x_0) = \lbrace x \in N_\epsilon(x_0) \mid \lim_{t \to -\infty} \phi_t(x) = x_0 \quad \text{and} \quad \phi_t(x) \in N_\epsilon(x_0) \text{ for all } t \le 0 \rbrace$$ 

Remark/Intuition

The condition that trajectories must stay within the neighborhood is critical. Consider a saddle point. There are many points in an initial neighborhood that will eventually be repelled and leave the neighborhood. These points are not part of the local stable manifold. The definition precisely isolates only those initial conditions that are perfectly aligned with the stable direction and will thus converge to the equilibrium point without first being pushed away.

2.2. The Global Perspective: Extending Local Manifolds

Remark/Intuition

Once the local stable and unstable manifolds are defined, the global manifolds are constructed by integrating the points on these local sets forward or backward in time. This process "collects" all the points in the entire phase space that are connected to the equilibrium point via its stable or unstable dynamics.

Definition: Global Stable Manifold

The global stable manifold of an equilibrium point $x_0$, denoted $W^s(x_0)$, is the union of all backward-time images of the local stable manifold. Intuitively, we take the set of points that are already known to be approaching $x_0$ (the local manifold) and trace their paths backward in time to find every point in the phase space that will eventually enter this local neighborhood and converge to $x_0$.

Formally, it is the union of the flow operator applied to the local manifold for all non-positive time:

$$W^s(x_0) = \bigcup_{t \le 0} \phi_t(W^s_{loc}(x_0))$$ 

Definition: Global Unstable Manifold

The global unstable manifold of an equilibrium point $x_0$, denoted $W^u(x_0)$, is the union of all forward-time images of the local unstable manifold. Intuitively, we take the points that are locally diverging from $x_0$ (which converge to $x_0$ in reverse time) and follow their trajectories forward to trace out the entire structure of repulsion from the equilibrium.

Formally, it is the union of the flow operator applied to the local manifold for all non-negative time:

$$W^u(x_0) = \bigcup_{t \ge 0} \phi_t(W^u_{loc}(x_0))$$ 


--------------------------------------------------------------------------------


3. Advanced Orbit Structures (Preview)

3.1. Homoclinic and Heteroclinic Orbits

With the formal definitions of stable and unstable manifolds, we can now begin to describe more complex and important dynamical phenomena. The interactions between these manifolds give rise to rich structures, including homoclinic and heteroclinic orbits, which will be formally defined next.

An Introduction to Dynamical Systems: A Study Guide

Table of Contents

Chapter 1: Foundational Concepts in Phase Space

* 1.1 Special Orbits: Homoclinic and Heteroclinic
* 1.2 Invariant Sets
* 1.3 Topological Equivalence of Dynamical Systems


--------------------------------------------------------------------------------


Chapter 1: Foundational Concepts in Phase Space

This chapter introduces three fundamental concepts that form the bedrock for analyzing the qualitative behavior of dynamical systems. We will explore special types of orbits that structure the phase space, the notion of sets that are preserved by the system's evolution, and a rigorous way to determine when two different systems can be considered "the same" from a topological standpoint.

1.1 Special Orbits: Homoclinic and Heteroclinic

Certain orbits, or solution curves, play a crucial role in defining the global structure of a dynamical system's phase space. A solution curve is the path traced by a point $x_0$ under the action of the flow operator $\phi_t$ for all time $t$, both forward and backward. Among the most important of these are homoclinic and heteroclinic orbits, which connect equilibrium points.

Definition: Homoclinic Orbit

A homoclinic orbit $\Gamma$ is a solution curve that connects an equilibrium point to itself. It originates from the equilibrium point (as $t \to -\infty$) and returns to the same equilibrium point (as $t \to +\infty$).

Formally, for an equilibrium point $x_0$, a homoclinic orbit is a trajectory $\Gamma$ such that:

$$\Gamma \subset W^s(x_0) \cap W^u(x_0)$$  

This means the orbit is simultaneously part of the stable manifold and the unstable manifold of the same point. The trajectory diverges from $x_0$ along its unstable manifold and converges back to $x_0$ along its stable manifold.

Definition: Heteroclinic Orbit

A heteroclinic orbit $\Gamma$ is a solution curve that connects two different equilibrium points. It originates from one equilibrium point (as $t \to -\infty$) and terminates at another (as $t \to +\infty$).

Formally, for two distinct equilibrium points $x_0 \neq x_1$, a heteroclinic orbit is a trajectory $\Gamma$ connecting $x_0$ to $x_1$ such that:

$$\Gamma \subset W^u(x_0) \cap W^s(x_1)$$  

This means the orbit is part of the unstable manifold of $x_0$ and the stable manifold of $x_1$.

Remark/Intuition

Homoclinic and heteroclinic orbits are of fundamental importance in the study of dynamical systems. Their presence can have profound implications for the system's behavior.

* Structural Skeletons: These orbits act as organizing centers or "skeletons" for the dynamics in phase space.
* Indicators of Chaos: The existence of a homoclinic orbit, in particular, is often a strong indicator—almost a guarantee—that the system exhibits chaotic behavior. We will explore this connection in greater detail later.
* Separatrix Cycles: Structures built from sequences of heteroclinic (and sometimes homoclinic) orbits are called separatrix cycles. These cycles can create boundaries in the phase space that separate regions of qualitatively different behavior.


--------------------------------------------------------------------------------


1.2 Invariant Sets

The concept of invariance is central to identifying regions in the phase space that are self-contained under the system's dynamics.

Definition: Invariant Set

Let $E \subset \mathbb{R}^n$ be an open set representing the state space, and let $\phi_t: \mathbb{R} \times E \to E$ be a flow operator defined for all times $t \in \mathbb{R}$.

A set $S \subset E$ is called an invariant set with respect to the flow $\phi_t$ if any trajectory starting in $S$ remains in $S$ for all time, both forward and backward.

More formally, $S$ is invariant if for any point $x \in S$, its entire orbit remains within $S$. This is expressed as:

$$\phi_t(S) = S \quad \text{for all } t \in \mathbb{R}$$  

where $\phi_t(S) = \lbrace \phi_t(x) \mid x \in S \rbrace$.

Remark/Intuition

* Forward and Backward Invariance: One can also define one-sided invariance. A set $S$ is forward invariant if $\phi_t(S) \subset S$ for all $t \ge 0$. This means that once you are in the set, you can never leave it as time moves forward. A set is backward invariant if $\phi_t(S) \subset S$ for all $t \le 0$.
* Discrete Time Systems (Maps): The concept applies equally to discrete maps. For a map $f: E \to E$, a set $S$ is invariant if applying the map (or its inverse) any number of times to a point in $S$ yields a point that is still in $S$. That is, $f(S) = S$.

The simplest examples of invariant sets are equilibrium points and limit cycles. Understanding which sets are invariant helps decompose the phase space into dynamically independent regions.


--------------------------------------------------------------------------------


1.3 Topological Equivalence of Dynamical Systems

A powerful idea in the study of dynamical systems is to determine when two systems, which may have different equations, are qualitatively "the same." The concept of topological equivalence provides a rigorous framework for this comparison.

Definition: Topological Equivalence

Consider two dynamical systems defined by continuously differentiable vector fields:

1. $\dot{x} = f(x)$, where $x \in A$ and $A \subset \mathbb{R}^n$ is an open set.
2. $\dot{y} = g(y)$, where $y \in B$ and $B \subset \mathbb{R}^m$ is an open set.

Let $\phi^A_t$ be the flow of the first system and $\phi^B_t$ be the flow of the second.

These two systems are topologically equivalent if there exists a homeomorphism $h: A \to B$ that maps the orbits of the first system onto the orbits of the second system while preserving the direction of time.

* A homeomorphism is a continuous function with a continuous inverse, meaning it provides a smooth, one-to-one mapping between the state spaces $A$ and $B$.

The condition of "preserving the direction of time" means that forward-time evolution in one system corresponds to forward-time evolution in the other. Formally, for every $x_0 \in A$ and time $t$, the following relationship must hold:

$$h(\phi^A_t(x_0)) = \phi^B_{\tau(x_0, t)}(h(x_0))$$ 

Here, $\tau(x_0, t)$ is a reparameterization of time. To preserve the direction of flow, for any fixed $x_0$, $\tau$ must be a strictly increasing function of $t$. This is guaranteed if its derivative with respect to $t$ is always positive: 

$$\frac{\partial \tau}{\partial t} > 0$$ 

Remark/Intuition

Topological equivalence means we can stretch, bend, or compress one state space to make it look like the other, such that the orbit structures align perfectly. The time it takes to travel along corresponding segments of orbits might differ (hence the reparameterization $\tau$), but the direction of travel is the same.

If two systems are topologically equivalent, then properties like the number and type of fixed points, the existence of closed orbits, and the stability of these objects are preserved. For instance, if one system has an orbit that converges to a fixed point, the corresponding orbit in the equivalent system will also converge to the corresponding fixed point. The diagram below illustrates the commutative relationship:

        φ^A_t
x₀   -------->   φ^A_t(x₀)
 |                 |
h|                 |h
 ↓                 ↓
h(x₀) -------->  h(φ^A_t(x₀)) = φ^B_τ(h(x₀))
        φ^B_τ


Example

Let's demonstrate topological equivalence with a simple example.

Consider the following two one-dimensional systems:

1. System A: $\dot{x} = -x$
2. System B: $\dot{y} = y$

The flow for System A is $\phi^A_t(x) = e^{-t}x$. Trajectories in this system decay exponentially toward the equilibrium at $x=0$. The flow for System B is $\phi^B_t(y) = e^t y$. Trajectories in this system grow exponentially away from the equilibrium at $y=0$.

Let's propose a candidate homeomorphism $h(x) = 1/x$. This map is defined for $x \neq 0$ and its inverse is $h^{-1}(y) = 1/y$, which is also continuous.

We now check if the condition for topological equivalence, $h(\phi^A_t(x_0)) = \phi^B_{\tau}(h(x_0))$, can be satisfied for a valid time reparameterization $\tau$.

Proof:

1. Calculate the left-hand side (LHS): First, we apply the flow of System A to a point $x_0$, and then map the result using $h$.
  
  $$h(\phi^A_t(x_0)) = h(e^{-t}x_0) = \frac{1}{e^{-t}x_0} = \frac{e^t}{x_0}$$ 

2. Calculate the right-hand side (RHS): First, we map the point $x_0$ using $h$, and then apply the flow of System B for a reparameterized time $\tau$.
  
  $$\phi^B_{\tau}(h(x_0)) = \phi^B_{\tau}\left(\frac{1}{x_0}\right) = e^{\tau} \cdot \frac{1}{x_0} = \frac{e^{\tau}}{x_0}$$ 

3. Equate LHS and RHS to find $\tau$: For the systems to be equivalent, we must have LHS = RHS.
   
  $\frac{e^t}{x_0} = \frac{e^{\tau}}{x_0} \implies e^t = e^{\tau} \implies \tau = t$ 

4. Verify the time-preservation condition: The reparameterization is $\tau(x_0, t) = t$. We check its derivative:
  
  $$\frac{\partial \tau}{\partial t} = \frac{d}{dt}(t) = 1$$
   
  Since $\frac{\partial \tau}{\partial t} = 1 > 0$, the condition is satisfied.

Conclusion: The systems $\dot{x} = -x and \dot{y} = y$ are topologically equivalent. The homeomorphism $h(x) = 1/x$ maps the converging dynamics of the first system onto the diverging dynamics of the second while preserving the direction of flow along the orbits. An orbit moving from $x=1$ towards $x=0$ in System A is mapped to an orbit moving from $y=1$ towards $y=\infty$ in System B. In both cases, the trajectory moves in the same "direction" on its respective path.

A Student's Guide to Dynamical Systems Theory

Table of Contents

* 1. Topological Equivalence and Conjugacy
  * 1.1 Defining Topological Equivalence
  * 1.2 A Critical Counterexample: The Importance of the Homeomorphism
  * 1.3 Conditions for Equivalence in Linearizable Systems
  * 1.4 Topological Conjugacy: Preserving Time
  * 1.5 Example of Topologically Conjugate Systems
* 2. The Hartman-Grobman Theorem
  * 2.1 Intuition: Connecting Nonlinear and Linear Worlds
  * 2.2 Formal Statement of the Theorem
  * 2.3 The Power and Implications of Hartman-Grobman


--------------------------------------------------------------------------------


1. Topological Equivalence and Conjugacy

In the study of dynamical systems, we are often less concerned with the precise numerical solution of a trajectory and more interested in its qualitative behavior. Does it spiral into a fixed point? Does it diverge to infinity? Does it oscillate forever? To formalize these qualitative similarities, we introduce the concepts of topological equivalence and the more restrictive topological conjugacy. These powerful ideas allow us to state precisely when two different systems share the same fundamental geometric structure.

1.1 Defining Topological Equivalence

Definition: Topological Equivalence

Two dynamical systems, defined by flows $\phi_t^A$ and $ phi_t^B$ on state spaces $A$ and $B$ respectively, are said to be topologically equivalent if there exists a homeomorphism $h: A \to B$ that maps trajectories of $ \phi^A $ onto trajectories of $\phi^B$ while preserving the direction of time.

* A homeomorphism is a continuous function between topological spaces that has a continuous inverse function. It is a map that preserves all the topological properties of a given space.

This means that for every point $x \in A$, the curve traced by $\phi_t^A(x)$ is mapped by $h$ to the curve traced by $\phi_t^B(h(x))$. While the direction of flow along the curve is preserved, the speed is not. The time parameter may be re-scaled, meaning a point that takes 1 second to travel a path in system A might take 2 seconds (or 0.5 seconds) to travel the corresponding path in system B.

1.2 A Critical Counterexample: The Importance of the Homeomorphism

Let's consider two simple one-dimensional systems:

1. System A: $\dot{x} = x$ (an unstable fixed point at $x=0$)
2. System B: $\dot{y} = -y$ (a stable fixed point at $y=0$)

At first glance, one might wonder if these systems could be considered equivalent. After all, they both feature straight-line trajectories moving away from or towards the origin. Let's propose a map $h(x) = 1/x$ and investigate if it establishes an equivalence.

Remark/Intuition: What Went Wrong?

Intuitively, these two systems have fundamentally different behaviors. One diverges from the origin, while the other converges to it. A continuous transformation should not be able to reverse the fundamental stability of a system. The direction of flow is a core topological property, and reversing it is a major violation. Let's see how this intuition manifests mathematically.

The proposed map $h(x) = 1/x$ is not a homeomorphism for this problem. A homeomorphism must be continuous everywhere on the domain of interest. Here, the domain includes the equilibrium point at $x=0$. The function $1/x$ has a discontinuity precisely at this equilibrium point.

Because the proposed mapping function $h$ is not continuous at the equilibrium point, it is not a valid homeomorphism. Therefore, it cannot be used to establish topological equivalence.

This example reveals a crucial insight:

Stable nodes, unstable nodes, and saddles are all topologically distinct. You cannot find a homeomorphism that continuously deforms the phase portrait of a stable node into that of an unstable node.

1.3 Conditions for Equivalence in Linearizable Systems

While a stable and an unstable node are not equivalent, other pairings are. For instance, a stable node and a stable spiral are topologically equivalent. Both systems draw all nearby trajectories into the fixed point, and one can be continuously deformed into the other. This leads to a more formal condition for equivalence in a large class of systems.

Theorem: Topological Equivalence for Hyperbolic Systems

For linear or linearizable systems, two systems are topologically equivalent in a neighborhood of a hyperbolic fixed point if the Jacobian matrices evaluated at that fixed point have the same number of eigenvalues with positive real parts and the same number of eigenvalues with negative real parts.

* Recall that a hyperbolic fixed point is one where the Jacobian has no eigenvalues with a zero real part.

Remark/Intuition

This theorem provides a powerful and practical criterion for determining equivalence. It essentially states that the local "picture" of the dynamics is determined entirely by the number of stable (negative real part) and unstable (positive real part) directions. The exact values of the eigenvalues don't matter for equivalence, nor does the presence of imaginary parts (which create spirals), as long as the counts of stable and unstable dimensions match.

1.4 Topological Conjugacy: Preserving Time

Topological conjugacy is a stricter condition than equivalence. It requires not only that the trajectories map onto one another but also that the parameterization by time is preserved.

Definition: Topological Conjugacy

Two systems with flows $\phi_t^A$ and $\phi_t^B$ are topologically conjugate if they are topologically equivalent via a homeomorphism $h$, and this mapping preserves the time parameter. Formally, for any initial point $x_0$ and any time $t$:

$$h(\phi_t^A(x_0)) = \phi_t^B(h(x_0))$$

Remark/Intuition

Notice the crucial difference: the time variable $t$ is exactly the same on both sides of the equation. This means that if it takes the first system $T$ seconds to get from point $x_0$ to $x_1$, it must also take the second system exactly $T$ seconds to get from the mapped point $h(x_0)$ to the mapped point $h(x_1)$. The systems are not just qualitatively similar; their flows are perfectly synchronized through the lens of the homeomorphism $h$.

1.5 Example of Topologically Conjugate Systems

Let's demonstrate this stronger condition with an example.

Example

Consider the following two systems:

1. System A: $\dot{x} = -x$
2. System B: $\dot{y} = -2y$

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

Remark/Intuition

This function $h(x)$ is a valid homeomorphism. It is continuous everywhere, including at $x=0$ where both pieces of the function converge to zero. It is also continuously differentiable at the origin.

Proof of Conjugacy

We must now verify that our chosen $h$ satisfies the condition $h(\phi_t^A(x)) = \phi_t^B(h(x))$. Let's analyze the left and right sides of the equation separately, assuming $x \ge 0$ for simplicity.

1. Left-Hand Side (LHS): First, we apply the flow of system A, and then we map the result with $h$.
  
  $$h(\phi_t^A(x)) = h(x e^{-t})$$  
   
  Since $xe^{-t}$ will have the same sign as $x$, we use the $x^2$ part of our definition for $h$:  
  
  $$h(x e^{-t}) = (x e^{-t})^2 = x^2 e^{-2t}$$ 

1. Right-Hand Side (RHS): First, we map the initial point $x$ with $h$, and then we apply the flow of system B to the result.  
   
  $$\phi_t^B(h(x)) = \phi_t^B(x^2)$$  
   
  Let $ y = h(x) = x^2 $. Applying the flow for system B gives:
   
  $$\phi_t^B(y) = y e^{-2t} = (x^2) e^{-2t} = x^2 e^{-2t}$$ 

Since LHS = RHS, we have shown that $h(\phi_t^A(x)) = \phi_t^B(h(x))$. The systems are indeed topologically conjugate.


--------------------------------------------------------------------------------


1. The Hartman-Grobman Theorem

We now arrive at one of the most fundamental and powerful results in the study of dynamical systems: the Hartman-Grobman Theorem. This theorem provides the rigorous justification for one of our most common analytical techniques: linearization. It formally establishes that, under certain conditions, the complex behavior of a nonlinear system in the close vicinity of an equilibrium point is qualitatively identical to the much simpler behavior of its linear approximation.

2.1 Intuition: Connecting Nonlinear and Linear Worlds

Remark/Intuition

The core message of the Hartman-Grobman theorem is profound:

In a small neighborhood of a hyperbolic equilibrium point, a nonlinear system is topologically conjugate to its linearization.

This is an extremely strong result. It doesn't just say the nonlinear system is "similar" to its linear approximation; it says there is a continuous, invertible map that transforms the nonlinear trajectories precisely onto the linear ones, while preserving the flow of time. This justifies our entire approach of analyzing the Jacobian matrix to understand the stability and local geometry of equilibria like nodes, saddles, and spirals. It guarantees that the picture we see in the linear system is not an illusion but a topologically faithful representation of the nonlinear system's behavior near the fixed point.

2.2 Formal Statement of the Theorem

Theorem: The Hartman-Grobman Theorem

Let $\mathbf{x_0}$ be an equilibrium point for the continuously differentiable system $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$ where $\mathbf{x} \in E$, an open subset of $\mathbb{R}^m$. Let $J(\mathbf{x_0})$ be the Jacobian matrix of $\mathbf{f}$ evaluated at $\mathbf{x_0}$.

If the equilibrium point $\mathbf{x_0}$ is hyperbolic (i.e., the matrix $J(\mathbf{x_0})$ has no eigenvalues with a real part equal to zero), then there exist neighborhoods $U$ and $V$ of $\mathbf{x_0}$ in $\mathbb{R}^m$ and a homeomorphism $h: U \to V$ such that for every initial point $\mathbf{x} \in U$, the flow of the nonlinear system, $\phi_t(\mathbf{x})$, is related to the flow of its linearization, $e^{J(\mathbf{x_0})t}\mathbf{z}$, by the conjugacy:

$$h(\phi_t(\mathbf{x})) = e^{J(\mathbf{x_0})t} h(\mathbf{x})$$

This must hold for all $t$ within some open time interval $I_0$ containing $0$.

2.3 The Power and Implications of Hartman-Grobman

This theorem gives us confidence in our analytical methods. When we encounter a complex, high-dimensional nonlinear system, we can:

1. Find its equilibrium points.
2. Linearize the system at each of these points by computing the Jacobian.
3. Calculate the eigenvalues of the Jacobian.

If the point is hyperbolic, Hartman-Grobman assures us that the local dynamics are completely characterized by the behavior of the linear system $\dot{\mathbf{z}} = J(\mathbf{x_0})\mathbf{z}$. The stability, the presence of saddle dynamics, and the spiral or nodal nature of the trajectories are all preserved. This allows us to use the well-understood tools of linear systems theory to draw robust conclusions about the behavior of highly nonlinear systems.


# Lecture 4

Multi-stability, nonlinear centers, limit cycles & nonlinear oscillations. (5.11.25)

Wilson–Cowan neural population model
  sigmoid function
  nullclines
  stable manifold of saddle -> separatrix, basins of attraction, bistability
Def. α/ω limit set
Def. attractor & basin of attraction
Closed orbits in state space
  Nonlinear centers
  Mention: Hamiltonian, conservative systems
  Limit cycles: intuition and 2d examples: van-der-Pol oscillator, Duffing oscillator
  (Hopfield networks, active memory patterns as point attractors)
  Def. limit cycle
Poincaré–Bendixson theorem
Mention: Poincaré map
Index theory in 2d, winding numbers, implications for unobserved regions
Hodgkin–Huxley model
  Separation of time scales
  Teaser: bifurcation diagrams

# Lecture 10

A Study Book on Dynamical Systems Theory

Table of Contents

Chapter 1: The General Solution to Linear ODE Systems

* 1.1 Recap: Eigenvalue-Based Solutions
* 1.2 Topological Equivalence and Similar Matrices
* 1.3 Canonical Forms for 2x2 Systems
* 1.4 The Fundamental Theorem of Linear Dynamical Systems
* 1.5 The Matrix Exponential
* 1.6 Reconciling the Solutions
* 1.7 The Degenerate Case

Chapter 2: Extensions of Linear Systems

* 2.1 Inhomogeneous (Affine) Systems
* 2.2 Non-Autonomous Systems with Forcing Functions

Chapter 3: Discrete-Time Linear Systems (Linear Maps)

* 3.1 Introduction to Linear Maps
* 3.2 The Scalar Case and Geometric Interpretation
* 3.3 Fixed Points: Stability Analysis
* 3.4 Higher-Dimensional Linear Maps

Chapter 4: Formal Foundations of Dynamical Systems

* 4.1 From Continuous to Discrete: The Flow Operator
* 4.2 A Formal Definition of a Dynamical System
* 4.3 Trajectories and Orbits
* 4.4 The Fundamental Existence-Uniqueness Theorem

Chapter 5: An Introduction to Dynamical Systems Reconstruction

* 5.1 The Problem of System Identification
* 5.2 Challenges in Recurrent Neural Networks: The Gradient Problem
* 5.3 A Modern Approach: Piecewise Linear Recurrent Networks (PL-RNNs)


--------------------------------------------------------------------------------


Chapter 1: The General Solution to Linear ODE Systems

This chapter transitions from a specific case of linear systems to a comprehensive, general solution. We begin by recapping the solution derived under the assumption of distinct eigenvalues and then build the necessary mathematical machinery—similar matrices and the matrix exponential—to state and prove the fundamental theorem of linear dynamical systems, which provides a solution for any linear system.

1.1 Recap: Eigenvalue-Based Solutions

We previously considered linear dynamical systems defined by a system of ordinary differential equations (ODEs) of the form:

$$\dot{\mathbf{x}} = A \mathbf{x}$$  

where $\mathbf{x} \in \mathbb{R}^m$ and $A$ is an $m \times m$ square matrix.

Under the strong assumption that the matrix $A$ has distinct eigenvalues $\lambda_i$, its corresponding eigenvectors $\mathbf{v}_i$ form a basis for the state space $\mathbb{R}^m$. In this scenario, we derived a general solution for the evolution of the system from an initial condition $\mathbf{x}_0$:

$$\mathbf{x}(t) = \sum_{i=1}^{m} c_i e^{\lambda_i t} \mathbf{v}_i$$  

The coefficients $c_i$ are determined by the initial conditions. By splitting the eigenvalues into their real ($\alpha_i$) and imaginary ($\omega_i$) parts, i.e., $\lambda_i = \alpha_i + j\omega_i$, we can analyze the system's behavior. The real part, $\alpha_i$, dictates the exponential growth or decay, while the imaginary part, $\omega_i$, governs oscillations.

This framework allowed us to classify various types of equilibria (fixed points):

* Stable/Unstable Nodes: Purely real eigenvalues, leading to convergence or divergence along eigenvector directions.
* Stable/Unstable Spirals: Complex conjugate eigenvalues where the real part is non-zero ($\alpha_i \neq 0$), causing trajectories to spiral into or away from the equilibrium.
* Centers: Purely imaginary eigenvalues ($\alpha_i = 0$), resulting in neutrally stable oscillations.
* Saddle Points: A mix of positive and negative real eigenvalues, creating directions of convergence and divergence.
* Line Attractors: At least one eigenvalue is zero, creating a continuous line (or plane) of equilibria.

However, this solution is incomplete. It relies on the assumption that the eigenvectors form a basis, which is not guaranteed if eigenvalues are repeated. To address this, we must develop a more general formulation.

1.2 Topological Equivalence and Similar Matrices

To generalize our understanding, we first introduce a concept that allows us to group matrices that produce qualitatively identical dynamics. This concept is similarity.

Definition: Similar Matrices

Two square matrices, $A_1$ and $A_2$, are called similar if there exists an invertible matrix $S$ such that the following relationship holds:

$$A_1 = S A_2 S^{-1}$$

The matrix $S$ represents an invertible transformation (a change of variables or basis) that maps the dynamics of one system onto the other.

Remark/Intuition

If two matrices are similar, their corresponding linear dynamical systems are considered topologically equivalent. This means that while the specific coordinate system may differ, the fundamental "shape" and stability of the trajectories (e.g., whether they form a spiral, a saddle, etc.) are identical. All similar matrices share the same eigenvalues and, consequently, the same dynamics. The eigendecomposition of a matrix, $A = V \Lambda V^{-1}$, is a prime example of this, where $A$ is similar to its diagonal matrix of eigenvalues, $\Lambda$.

1.3 Canonical Forms for 2x2 Systems

For any $2 \times 2$ matrix, it can be shown that through a similarity transformation, it can be converted into one of three fundamental or canonical forms. These three forms encompass all possible dynamics for two-dimensional linear systems.

1. Diagonal Form (Distinct Real Eigenvalues)
   
   $$A = \begin{pmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{pmatrix}$$  
   
   Dynamics: This form corresponds to systems without an oscillatory component. Depending on the signs of $\lambda_1$ and $\lambda_2$, this matrix describes stable nodes, unstable nodes, or saddle points.
2. Complex-Conjugate Form (Complex Eigenvalues)
   
   $$A = \begin{pmatrix} a & -b \\ b & a \end{pmatrix}$$
   
   Dynamics: This matrix has complex conjugate eigenvalues $\lambda = a \pm jb$. This form gives rise to spirals (if $a \neq 0$) or centers (if $a=0$). The matrix can be conceptually split into a component controlling amplitude ($a$, through a scaling matrix $aI$) and a component controlling rotation ($b$, through a skew-symmetric matrix).
3. Degenerate Form (Repeated Eigenvalues)
   
   $$A = \begin{pmatrix} a & 1 \\ 0 & a \end{pmatrix}$$
   
   Dynamics: This is the new case we have not yet covered. This matrix has a single eigenvalue, $a$, with a multiplicity of two, but it only has one corresponding eigenvector. This case is called degenerate because the eigenvectors do not form a basis for the space; they align, and the dynamics effectively collapse into a one-dimensional subspace. Trajectories will converge towards or diverge from the equilibrium while ultimately aligning with the single eigenvector direction.

1.4 The Fundamental Theorem of Linear Dynamical Systems

We now state the central theorem that provides a universal solution for any linear ODE system, regardless of its eigenvalues or whether it is diagonalizable.

Theorem: Fundamental Theorem of Linear Dynamical Systems

Let $A$ be an $m \times m$ matrix and let $\mathbf{x}_0 \in \mathbb{R}^m$ be an initial condition. The initial value problem defined by:

$$\dot{\mathbf{x}} = A \mathbf{x}, \quad \mathbf{x}(0) = \mathbf{x}_0$$

has a unique solution given by:

$$\mathbf{x}(t) = e^{At} \mathbf{x}_0$$

where $e^{At}$ is the matrix exponential.

1.5 The Matrix Exponential

The solution presented in the theorem relies on the concept of the matrix exponential, which is defined in direct analogy to the Taylor series expansion of the scalar exponential function.

Definition: The Matrix Exponential

For a square matrix $A$ and a scalar $t$, the matrix exponential $e^{At}$ is defined by the infinite series:

$$e^{At} = \sum_{k=0}^{\infty} \frac{(At)^k}{k!} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \dots$$ 

Proof (Partial): Showing $e^{At}\mathbf{x}_0$ is a Solution

To verify that $\mathbf{x}(t) = e^{At}\mathbf{x}_0$ is indeed a solution to $\dot{\mathbf{x}} = A \mathbf{x}$, we can take its temporal derivative. Differentiating the series term-by-term with respect to t:

$$
\begin{aligned} 
\frac{d}{dt} \mathbf{x}(t) \\ = \frac{d}{dt} \left( \sum_{k=0}^{\infty} \frac{A^k t^k}{k!} \right) \mathbf{x}0 \\ = \left( \sum{k=1}^{\infty} \frac{A^k (k t^{k-1})}{k!} \right) \mathbf{x}0 \\ = \left( \sum{k=1}^{\infty} \frac{A^k t^{k-1}}{(k-1)!} \right) \mathbf{x}0 \\ = A \left( \sum{k=1}^{\infty} \frac{A^{k-1} t^{k-1}}{(k-1)!} \right) \mathbf{x}0 \\ = A \left( \sum{j=0}^{\infty} \frac{A^j t^j}{j!} \right) \mathbf{x}_0 \quad (\text{letting } j=k-1) \\ = A e^{At} \mathbf{x}_0 \\ = A \mathbf{x}(t) 
\end{aligned}
$$

This confirms that the form satisfies the differential equation. The proof of uniqueness is more involved and relies on showing that any two potential solutions must be identical.

1.6 Reconciling the Solutions

At first glance, the matrix exponential solution $\mathbf{x}(t) = e^{At} \mathbf{x}_0$ looks quite different from the eigenvalue-based solution $\mathbf{x}(t) = \sum c_i e^{\lambda_i t} \mathbf{v}_i$. We will now show that for the case where the matrix $A$ is diagonalizable (i.e., has distinct eigenvalues), these two forms are perfectly equivalent.

Proof: Equivalence of Solutions for a Diagonalizable Matrix

1. Express the Eigenvalue Solution in Matrix Form: The initial condition $\mathbf{x}_0$ can be written as a linear combination of eigenvectors: $\mathbf{x}_0 = \sum c_i \mathbf{v}_i$. In matrix form, this is $\mathbf{x}_0 = V\mathbf{c}$, where $V$ is the matrix whose columns are the eigenvectors $\mathbf{v}_i$ and $\mathbf{c}$ is a vector of coefficients $c_i$. Since the eigenvectors form a basis, $V$ is invertible, so $\mathbf{c} = V^{-1} \mathbf{x}_0$.

Substituting this into the eigenvalue solution gives:

$$\mathbf{x}(t) = \sum_{i=1}^{m} (V^{-1}\mathbf{x}_0)_i e^{\lambda_i t} \mathbf{v}_i$$  

This can be more cleanly written in matrix form as:

$$\mathbf{x}(t) = V \begin{pmatrix} e^{\lambda_1 t} & & 0 \\ & \ddots & \\ 0 & & e^{\lambda_m t} \end{pmatrix} V^{-1} \mathbf{x}_0$$

Let's call the diagonal matrix $D(t) = \text{diag}(e^{\lambda_i t})$. So, $\mathbf{x}(t) = V D(t) V^{-1} \mathbf{x}_0.

2. Expand the Matrix Exponential Solution: We start with the fundamental solution $\mathbf{x}(t) = e^{At} \mathbf{x}_0$. Since $A$ is diagonalizable, we can write $A = V \Lambda V^{-1}$, where $\Lambda$ is the diagonal matrix of eigenvalues. Let's substitute this into the series definition of the matrix exponential:
   
   $$e^{At} = \sum_{k=0}^{\infty} \frac{(V \Lambda V^{-1} t)^k}{k!}$$
   
   Now, consider the term $(V \Lambda V^{-1})^k$. Due to the telescoping cancellation of the inner $V^{-1}V$ terms, this simplifies:
   
   $$(V \Lambda V^{-1})^k = (V \Lambda V^{-1})(V \Lambda V^{-1})\dots(V \Lambda V^{-1}) = V \Lambda^k V^{-1}$$
   
3. Substituting this back into the series:
   
   $$\begin{aligned} e^{At} &= \sum_{k=0}^{\infty} \frac{V (\Lambda t)^k V^{-1}}{k!} \\ &= V \left( \sum_{k=0}^{\infty} \frac{(\Lambda t)^k}{k!} \right) V^{-1} \end{aligned}$$
   
   The series inside the parentheses is the exponential of the diagonal matrix $\Lambda t$. The exponential of a diagonal matrix is simply the diagonal matrix of the exponentials of its elements:
   
  $$\sum_{k=0}^{\infty} \frac{(\Lambda t)^k}{k!} = e^{\Lambda t} = \begin{pmatrix} e^{\lambda_1 t} & & 0 \\ & \ddots & \\ 0 & & e^{\lambda_m t} \end{pmatrix} = D(t)$$  

1. Conclusion: We have shown that $e^{At} = V D(t) V^{-1}$. Therefore, the solution is:
   
   $$\mathbf{x}(t) = e^{At} \mathbf{x}_0 = V D(t) V^{-1} \mathbf{x}_0$$  
   
   This is exactly the same matrix form we derived from the eigenvalue-based approach. The two solutions are equivalent when $A$ is diagonalizable. The power of the matrix exponential formulation is that it also covers the non-diagonalizable (degenerate) case.

1.7 The Degenerate Case

The fundamental theorem holds even when eigenvalues are repeated and the matrix $A$ is not diagonalizable. In this degenerate case, the solution involves not just exponential terms but also polynomials of time $t$.

Example: A Degenerate 2x2 System

Consider the canonical degenerate matrix:

$$A = \begin{pmatrix} a & 1 \\ 0 & a \end{pmatrix}$$

The solution to $\dot{\mathbf{x}} = A\mathbf{x}$ is given by $\mathbf{x}(t) = e^{At}\mathbf{x}_0$. For this specific matrix, the matrix exponential can be calculated to be:

$$e^{At} = e^{at} \begin{pmatrix} 1 & t \\ 0 & 1 \end{pmatrix}$$

The solution is therefore:

$$\mathbf{x}(t) = e^{at} \begin{pmatrix} 1 & t \\ 0 & 1 \end{pmatrix} \mathbf{x}_0$$  

Notice the appearance of the linear term $t$ in the solution, a direct result of the degeneracy. In higher-dimensional degenerate systems, higher-order polynomials of $t$ can appear in the solution. This polynomial component is naturally handled by the matrix exponential formulation, highlighting its generality and power.


A Study of Dynamical Systems and Recurrent Neural Networks

Table of Contents

1. Revisiting Recurrent Networks and Their Challenges
  * 1.1. The Exploding and Vanishing Gradient Problem
  * 1.2. Dynamical Systems Reconstruction (DSR)
2. Piecewise Linear Recurrent Neural Networks (PL-RNNs)
  * 2.1. Architecture and Formulation
  * 2.2. Addressing the Gradient Problem
  * 2.3. Inducing Stability: Line and Manifold Attractors
  * 2.4. Learning Slow Time Constants
3. Training PL-RNNs through Regularization
  * 3.1. The Regularized Loss Function
4. Benchmark Application: The Addition Problem
  * 4.1. Task Definition


--------------------------------------------------------------------------------


1. Revisiting Recurrent Networks and Their Challenges

This chapter provides a brief review of a fundamental challenge in training Recurrent Neural Networks (RNNs) and introduces the problem setting of Dynamical Systems Reconstruction, which motivates the advanced architectures discussed later.

1.1 The Exploding and Vanishing Gradient Problem

Remark/Intuition

When training Recurrent Neural Networks using gradient descent, the primary objective is to minimize a loss function that quantifies the difference between the network's predictions and the true observed data. A common choice for this is the mean squared error over a time series.

The process of backpropagation through time, which is necessary to compute the gradients for an RNN, involves the chain rule. This results in a long product of Jacobian matrices, each representing the derivative of the latent state at one time step with respect to the latent state at the preceding time step.

The core of the exploding and vanishing gradient problem lies in this product. If the magnitudes (norms) of these Jacobian matrices are consistently greater than one, their product will grow exponentially, leading to an "exploding" gradient. Conversely, if their magnitudes are consistently less than one, the product will shrink exponentially towards zero, causing the gradient to "vanish." Both scenarios severely impede the network's ability to learn long-term dependencies in the data, as the influence of past states on the current loss becomes either overwhelmingly large or negligible.

Definition: Standard Loss Function

The optimization is typically performed on a loss function, $\mathcal{L}$, calculated as the sum of squared differences between the observed time series values, $X_t$, and the estimated values, $\hat{X}_t$, produced by the network's decoder or observation function.

$$\mathcal{L} = \sum_{t=1}^{T} \|X_t - \hat{X}_t\|^2$$

When minimizing this loss via gradient descent, we encounter terms derived from the chain rule involving products of Jacobians of the latent state transition function, $f$:

$$\frac{\partial z_t}{\partial z_{t-k}} = \frac{\partial z_t}{\partial z_{t-1}} \frac{\partial z_{t-1}}{\partial z_{t-2}} \cdots \frac{\partial z_{t-k+1}}{\partial z_{t-k}}$$

These products are the source of the exploding and vanishing gradient problem.

1.2 Dynamical Systems Reconstruction (DSR)

Remark/Intuition

Dynamical Systems Reconstruction (DSR) is the process of inferring the underlying mathematical model that generated an observed time series. The core assumption is that the data we collect is the product of some unknown, underlying dynamical system. We do not observe the true state of this system directly but rather through a measurement or observation function.

The goal of DSR is to use the observed time series data, $X_t$, to find a parameterized function, $f_\lambda$, that serves as a good approximation of the system's true, unknown flow operator. Simultaneously, we often need to estimate the observation function, $g_\lambda$, that maps the system's internal states to the measurements we can see. In the context of machine learning, an RNN is a natural candidate for this task: the network's recursive update rule acts as the approximate flow operator, and its output layer (or decoder) acts as the approximate observation function.

Definition: The DSR Setup

The problem of Dynamical Systems Reconstruction is formally defined by the following components:

1. Data Generating System: An unknown system whose state evolves over time according to a deterministic or stochastic rule. This evolution is governed by an unknown flow operator, $F$.
2. Observation Function: We do not have direct access to the true state of the system. Instead, we observe it through an unknown measurement function, $G$.
3. Observed Time Series: The sequence of measurements collected over time forms the time series data, denoted as $X_t$, for a duration of length $T$.
4. Modeling Goal: The objective is to estimate a parameterized function, $f_\lambda$, that approximates the true flow operator F, and a parameterized function, $g_\lambda$, that approximates the true observation function $G$. The parameters, denoted collectively by $\lambda$, are learned from the observed data $X_t$.

In the RNN framework, this translates to:

* The latent state update is an approximation of the flow operator: $z_t = f_\lambda(z_{t-1}, s_t)$, where $s_t$ are potential external inputs.
* The network output is generated by the observation function (decoder): $\hat{X}\_t = g_\lambda(z_t)$.

The overall aim is to estimate the functions $f_\lambda$ and $g_\lambda$ from the data.


A Study Book on Dynamical Systems and Machine Learning

Table of Contents

* Chapter 1: Reconstructing Dynamical Systems from Time Series
  * 1.1 The Goal of Systems Reconstruction
  * 1.2 Qualitative Hallmarks of a Successful Reconstruction
  * 1.3 A Formal Definition of Reconstruction via Topological Conjugacy
  * 1.4 Beyond Topology: Quantifying Geometric Similarity
    * 1.4.1 The Kullback-Leibler Divergence for State-Space Geometry
    * 1.4.2 Practical Estimation of KL Divergence


--------------------------------------------------------------------------------


Chapter 1: Reconstructing Dynamical Systems from Time Series

1.1 The Goal of Systems Reconstruction

Remark/Intuition

In the study of dynamical systems, a central challenge is to move beyond general machine learning benchmarks and develop models that can accurately recreate the specific system that generated an observed set of time series data. When a model, such as a Recurrent Neural Network (RNN), is trained on data from a system (e.g., a bursting neuron model or human ECG data), it must learn the underlying rules governing its evolution. This task is distinct from simple prediction; the goal is to build a generative model that embodies the system's dynamics.

This endeavor often requires specialized training techniques and amendments to standard optimization procedures like gradient descent. The following sections will explore what it means to successfully "reconstruct" a dynamical system and how we can formally define and measure this success.

1.2 Qualitative Hallmarks of a Successful Reconstruction

Remark/Intuition

Before diving into formal definitions, it is useful to establish an intuitive understanding of what a good reconstruction looks like. When we only have access to a finite, and often short, trajectory from a real system, a powerful model should be able to infer the global properties of the dynamics.

Key properties of a high-quality reconstruction include:

* Reconstruction of the Full Attractor: The model should be able to generate the complete extent of the system's attractor, even if it was only trained on a small portion of it. For instance, a network trained on a short segment of a trajectory from the chaotic Rössler system should learn to reproduce the entire Rössler attractor.
* Generalization to Nearby Initial Conditions: A successful model must not only replicate the trajectory it was trained on but also accurately predict the evolution of the system from initial conditions it has not seen before, provided they are within the same basin of attraction.

The ability to achieve this is non-trivial. It demonstrates that the model has learned the underlying governing equations, or a system topologically equivalent to them, rather than simply memorizing a single time series.

When working with real-world empirical data, a crucial prerequisite for these comparisons is to perform an optimal delay embedding. This ensures that the observed time series is represented in a state space that properly unfolds the system's dynamics, making a meaningful comparison to the model's generated state space possible. For developing and validating new methods, it is imperative to first test them on simulated data where the ground-truth governing equations are precisely known.

1.3 A Formal Definition of Reconstruction via Topological Conjugacy

Definition: Dynamical Systems Reconstruction

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

Remark/Intuition

Recall that topological conjugacy implies a deep structural equivalence between two systems. It means that there exists a homeomorphism $g: B \to \mathcal{R}^{\ast}$ (a continuous, invertible function with a continuous inverse) that maps the original state space to the reconstructed one.

This conjugacy ensures that for any initial condition $x_0 \in B$, the trajectory generated by the original system, $x(t) = \phi(t, x_0)$, is topologically equivalent to the trajectory generated by the reconstructed system from the mapped initial condition, $x^{\ast}(t) = \phi^{\ast}(t, g(x_0))$. This equivalence must also preserve the parameterization by time.

In simple terms, if a model is topologically conjugate to the real system, it has perfectly captured the qualitative structure of the dynamics—it's like a stretched or compressed, but unbroken, version of the original.

1.4 Beyond Topology: Quantifying Geometric Similarity

Remark/Intuition

While topological conjugacy provides a powerful definition of equivalence, it does not capture every aspect we might be interested in. Specifically, it is a topological definition and is insensitive to the geometric properties of the attractor. For a reconstruction to be truly useful, we often want the geometry of the generated attractor to be similar to that of the original.

To compare geometric properties, especially when the two systems may live in different state spaces ($\mathcal{R}$ and $\mathcal{R}^{\ast}$), we require a measure that can assess the similarity between their structures.

1.4.1 The Kullback-Leibler Divergence for State-Space Geometry

Remark/Intuition

Real-world data is invariably noisy. Even for deterministic chaotic systems, it is often fruitful to adopt a statistical perspective and describe the system's behavior using an invariant measure, which describes the long-term probability of finding the system in a particular region of its state space.

We can therefore quantify the geometric similarity between the true system and our model by comparing the probability distributions they induce over their respective state spaces. A powerful tool for this is the Kullback-Leibler (KL) divergence.

Definition: State-Space KL Divergence

The KL divergence measures the difference between two probability distributions. Let $P_{true}(x)$ be the probability distribution of states from the true underlying system, and let $P_{gen}(x)$ be the distribution of states generated by our model. The KL divergence is defined as:

$$D_{KL}(P_{true} \,\|\, P_{gen}) = \int_{\mathcal{D}} P_{true}(x) \log \left( \frac{P_{true}(x)}{P_{gen}(x)} \right) dx$$

where the integral is taken over the entire domain $\mathcal{D}$ of interest.

* If the distributions are identical ($P_{true}(x) = P_{gen}(x)$ everywhere), the fraction inside the logarithm is 1, making $\log(1) = 0$, and thus $D_{KL} = 0$.
* As the distributions diverge, the KL divergence becomes a positive value greater than zero.

1.4.2 Practical Estimation of KL Divergence

Remark/Intuition

Calculating the integral in the KL divergence definition is often intractable. We must therefore rely on numerical estimation methods.

Method 1: Grid-Based Estimation (Binning)

A straightforward approach, analogous to the box-counting method for fractal dimension, is to discretize the state space.

1. Place a grid of $K$ bins (or boxes) over the state space.
2. Approximate the continuous probabilities $P(x)$ by estimating the relative frequencies $\hat{p}$ of observed data points that fall into each bin.

Definition: Binned KL Divergence Estimator

The estimator for the KL divergence becomes a sum over the $K$ bins:

$$\hat{D}_{KL} \approx \sum_{k=1}^{K} \hat{p}_{true}(k) \log \left( \frac{\hat{p}_{true}(k)}{\hat{p}_{gen}(k)} \right)$$

where $\hat{p}\_{true}(k)$ is the estimated probability of the true data being in bin $k$, and $\hat{p}\_{gen}(k)$ is the same for the generated data.

Remark/Intuition

This method has a significant caveat: it performs poorly in high-dimensional state spaces due to the "curse of dimensionality," where the number of bins required to cover the space grows exponentially. Furthermore, one must be careful to ensure that no bin used in the calculation has a zero probability in the denominator ($\hat{p}_{gen}(k) = 0$), as this would cause the expression to diverge to infinity. This typically requires choosing a sufficiently large bin size or other regularization techniques.

Method 2: Gaussian Mixture Models (GMM)

An alternative and more sophisticated approach, particularly for higher-dimensional spaces, avoids a rigid global grid.

1. Instead of binning the entire space, define local $\epsilon$-neighborhoods along the observed trajectory.
2. Model the probability distribution within each of these neighborhoods using a Gaussian distribution.
3. The overall probability distribution is then represented as a Gaussian Mixture Model (GMM), which is a weighted sum of these individual Gaussians.

This GMM representation can then be used to estimate the probabilities $P_{true}(x)$ and $P_{gen}(x)$ needed for the KL divergence calculation. While the technical details are extensive, the core idea is to use a more flexible, data-driven method to model the probability distributions, which is more robust in high dimensions.

A Study Book on Dynamical Systems Reconstruction

Table of Contents

* Chapter 1: Evaluating Dynamical Systems Reconstructions
  * 1.1 The Challenge of Assessing Performance in Chaotic Systems
  * 1.2 Why Traditional Metrics Fail: The Case of Mean Squared Error
  * 1.3 A Dynamical Systems Approach to Evaluation
    * 1.3.1 Assessing Geometric Overlap
    * 1.3.2 Assessing Temporal Structure
    * 1.3.3 Other Dynamical Invariants
* Chapter 2: The Deep Connection Between Gradients and Chaos
  * 2.1 Revisiting the Exploding & Vanishing Gradient Problem
  * 2.2 Formalizing System Dynamics: The Lyapunov Spectrum
  * 2.3 Deconstructing the Learning Process: The Loss Gradient
  * 2.4 The Fundamental Link: Jacobians in Dynamics and Learning


--------------------------------------------------------------------------------


Chapter 1: Evaluating Dynamical Systems Reconstructions

When we train a model, such as a recurrent neural network, to replicate a dynamical system from data, a fundamental question arises: How do we measure success? When can we confidently say that our model is a good "dynamic systems reconstruction"? This chapter explores the nuances of this evaluation, highlighting the inadequacy of standard statistical measures for chaotic systems and introducing a more robust framework rooted in the principles of dynamical systems theory.

1.1 The Challenge of Assessing Performance in Chaotic Systems

The core challenge stems from a defining characteristic of many real-world systems: chaos. In chaotic systems, trajectories that begin from infinitesimally different initial conditions will diverge exponentially over time. This sensitivity to initial conditions has profound implications for evaluation.

* For a non-chaotic system, such as one that settles into a complex limit cycle, a well-trained model initialized on a true trajectory should be able to replicate that trajectory almost perfectly. A point-by-point comparison is meaningful.
* For a chaotic system, this is not the case. Even a perfect model of the Lorenz '63 system will produce a trajectory that quickly diverges from the original data it was trained on. Therefore, expecting a precise, long-term overlap in the time domain is not a sensible goal. Our evaluation must instead focus on whether the model has captured the underlying rules and structure of the system, not a specific path through its state space.

1.2 Why Traditional Metrics Fail: The Case of Mean Squared Error

In standard time series analysis and machine learning, the Mean Squared Error (MSE) is a ubiquitous metric for performance. However, for the reasons outlined above, it is fundamentally unsuited for evaluating reconstructions of chaotic systems.


--------------------------------------------------------------------------------


Remark/Intuition: The Deception of Mean Squared Error

The MSE is misleading when applied to chaotic systems. Consider two trajectories generated from the exact same Lorenz '63 system with the same parameters. Due to sensitivity to initial conditions, they will always diverge. Once this divergence occurs, the point-wise MSE between them will grow large, even though they both perfectly represent the same underlying dynamics.

An untrained or poorly trained recurrent neural network might produce a simple, non-chaotic oscillatory output. This simple output might, by chance, stay close to one major oscillation period of the true chaotic system for a short time, resulting in a deceptively low MSE. Conversely, a perfectly trained network that correctly captures the chaotic nature of the Lorenz attractor will produce a valid trajectory that, by definition, diverges from the training data, leading to a higher MSE.

This leads to a paradoxical and dangerous conclusion if one relies on MSE: the poorer model appears to perform better. This is an extremely important message to internalize. You cannot use out-of-the-box classical measures from statistics and machine learning to assess the quality of a dynamical systems reconstruction.


--------------------------------------------------------------------------------


Examples: Visualizing the MSE Paradox

Consider reconstructions of the Lorenz '63 system produced by a recurrent neural network, where blue represents the original simulated trajectories and red represents the network's output.

| Reconstruction Quality | State Space Plot | Time Series | Mean Squared Error | Analysis |
|------------------------|-----------------|-------------|-------------------|----------|
| Poor                   | The model fails to capture the classic "butterfly" geometry of the Lorenz attractor. | The model only captures a single major oscillation pattern. | Low | The MSE is deceptively low because the model's simple, non-chaotic output happens to align with the training data for a short period before the true system diverges elsewhere. |
| Excellent              | The model's output traces the geometry of the Lorenz attractor almost perfectly. | The model's output is qualitatively similar but quickly diverges from the specific path of the blue trajectory. | High | The MSE is high precisely because the model has successfully learned the chaotic dynamics. Both the model and the true system are chaotic and thus their trajectories rapidly separate. |

This illustrates that to truly evaluate our models, we need measures that assess the similarity of the underlying temporal and geometric structures, not the point-wise agreement of specific trajectories.

1.3 A Dynamical Systems Approach to Evaluation

A robust evaluation framework assesses agreement on multiple fronts: the geometry of the attractor in state space and the general temporal structure of the signals.

1.3.1 Assessing Geometric Overlap

We want to confirm that the attractor generated by our model has the same shape and density as the true system's attractor.


--------------------------------------------------------------------------------


Remark/Intuition: From Topology to Geometry

We often desire more than just topological agreement (e.g., both attractors have one hole). We want geometrical agreement, meaning the shapes are quantitatively similar. A probabilistic formulation is particularly well-suited for this task. In dynamical systems theory, chaotic attractors are often described by invariant measures, which capture the probability of finding the system in a particular region of its state space. This approach is also robust to the noise that is always present in real-world data.


--------------------------------------------------------------------------------


Metrics for assessing geometric overlap include:

* Vaserstein Distance: A measure of the distance between probability distributions.
* Kullback-Leibler (KL) Divergence: Another measure to quantify the difference between two probability distributions, often used to assess the overlap of the invariant measures on the attractors.

1.3.2 Assessing Temporal Structure

When we seek "temporal overlap," we do not mean the precise matching of time-domain patterns. Instead, we want to measure the similarity in the general temporal structure, such as characteristic frequencies and correlations.

Key tools for this include:

* Power Spectra: Derived from the Fourier Transform, the power spectrum reveals the dominant frequencies present in a signal.
* Autocorrelation Functions: These functions measure the correlation of a signal with a delayed copy of itself, revealing characteristic time scales and periodicities.

To quantify the similarity between the power spectra of a true signal and a reconstructed one, the Hellinger distance is a common and effective choice.

--------------------------------------------------------------------------------


Definition: Hellinger Distance for Power Spectra

Let $P_{true}(\omega)$ and $P_{recon}(\omega)$ be the power spectra of the true and reconstructed signals, respectively, as a function of frequency $\omega$. We first normalize both spectra such that their total area is one:

$$\int_{-\infty}^{\infty} P_{true}(\omega) \, d\omega = 1 \quad \text{and} \quad \int_{-\infty}^{\infty} P_{recon}(\omega) \, d\omega = 1$$

The Hellinger distance $H$ is then defined as:

$$H(P_{true}, P_{recon}) = \sqrt{1 - \int_{-\infty}^{\infty} \sqrt{P_{true}(\omega) P_{recon}(\omega)} \, d\omega}$$

Remark/Intuition: Understanding the Hellinger Distance

* The integral term acts like a correlation measure in Fourier space. It measures the product of the "power" (square root of the power spectrum) at each frequency.
* The entire distance is a normalized measure bounded between 0 and 1.
* If the power spectra are identical, the integral becomes 1, and the distance $H$ is 0 (perfect agreement).
* If the spectra have no overlap, the integral is 0, and the distance $H$ is 1 (maximum disagreement).
* In practice, the integral is approximated numerically.


--------------------------------------------------------------------------------


1.3.3 Other Dynamical Invariants

Beyond direct comparisons of geometry and spectra, we can compare fundamental quantitative properties—or invariants—of the dynamical systems.

* Lyapunov Exponents:
  * Maximum Lyapunov Exponent ($\lambda_{max}$): This measures the average exponential rate of divergence of nearby trajectories. A positive $\lambda_{max}$ is a hallmark of chaos. It can be estimated from real data (e.g., using a delay embedding) and computed directly from a trained model. Comparing these values is a powerful test of the model's learned dynamics.
  * Full Lyapunov Spectrum: This is the complete set of Lyapunov exponents. While very tricky to estimate from real data, it can be computed for a known model (like a trained RNN) from its Jacobians, providing a detailed fingerprint of the system's dynamics.
* Fractal Dimensionality: Chaotic attractors are often fractals. We can assess if our model's attractor has the same dimensionality as the true one.
  * Box-Counting Dimension: A straightforward method suitable for lower-dimensional data.
  * Correlation Dimension: A more practical estimator for higher-dimensional data, which describes the scaling behavior of data points within an $\epsilon$-ball placed along trajectories.
  * Kaplan-Yorke Dimension: An estimator for the fractal dimensionality that can be calculated from the Lyapunov spectrum.


--------------------------------------------------------------------------------


Examples: Successful Reconstructions

Well-trained recurrent neural networks have been shown to successfully reconstruct a variety of dynamical systems:

* Chaotic Lorenz System: Models can capture the chaotic attractor, exhibiting similar power spectra and Lyapunov exponents, even though the time-domain trajectories diverge. In some cases, these models even correctly recover system properties like equilibria that were not explicitly present in the training data (which consisted only of trajectories on the attractor).
* Bursting Neuron Model (Limit Cycle): For this complex but non-chaotic system, a trained model can achieve near-perfect overlap in the time domain when initialized correctly.
* Higher-Dimensional Neural Population Models (Chaotic): Similar to the Lorenz system, successful reconstructions capture the general temporal and geometric structure without precise trajectory matching.

As a general rule, a visual representation of these various measures (KL divergence, power spectra agreement, Kaplan-Yorke dimension, $\lambda_{max}$) shows a clear trend: as the reconstruction quality improves from poor to excellent, all these dynamical measures show progressively better agreement.


--------------------------------------------------------------------------------


Chapter 2: The Deep Connection Between Gradients and Chaos

In training recurrent neural networks, we often encounter the exploding and vanishing gradient problem. This issue, which can destabilize or stall the learning process, is not just a numerical quirk. For RNNs trained on dynamical systems, there is a profound and direct connection between the behavior of these gradients and the intrinsic stability properties of the system being modeled, as characterized by its Lyapunov spectrum. This connection was explored in detail in a 2020 NeurIPS paper by Yasin Abbasi-Asl, Zeinab Sadegh-Zadeh, and others.

2.1 Revisiting the Exploding & Vanishing Gradient Problem

The core of training an RNN involves backpropagation through time (BPTT), where the gradient of a loss function is calculated with respect to the network's parameters. This calculation involves a long chain of matrix multiplications, one for each time step. If the matrices in this chain are consistently larger than one, the gradients explode; if they are smaller than one, the gradients vanish. We will now show that the matrices involved are, in fact, the Jacobians of the system's dynamics.

2.2 Formalizing System Dynamics: The Lyapunov Spectrum

To understand a system's stability, we analyze how small perturbations evolve. The Lyapunov exponents quantify the average exponential rate of separation (or convergence) of nearby trajectories.


--------------------------------------------------------------------------------


Definition: Maximum Lyapunov Exponent

Consider a generic discrete-time dynamical system described by a recursive map $f$:

$$z_t = f(z_{t-1})$$

where $z_t \in \mathbb{R}^m$ is the state of the system at time $t$. The evolution of an infinitesimal perturbation is governed by the product of the system's Jacobians along a trajectory. The maximum Lyapunov exponent, $\lambda_{max}$, is defined as:

$$\lambda_{max} = \lim_{T \to \infty} \frac{1}{T} \log \left\| \prod_{r=0}^{T-2} J(z_r) \right\|$$

where:

* $J(z_r) = \frac{\partial f(z_r)}{\partial z_r} = \frac{\partial z_{r+1}}{\partial z_r}$ is the Jacobian matrix of the map $f$ evaluated at state $z_r$.
* The product $\prod_{r=0}^{T-2} J(z_r)$ represents the accumulated linearization of the dynamics over $T-1$ steps.
* $\|\| \cdot \|\|$ denotes a matrix norm, such as the spectral norm, which yields the maximum singular value. Taking the maximum singular value gives us the exponent corresponding to the fastest rate of expansion.

Remark/Intuition: The Product of Jacobians

The key takeaway here is the product series of Jacobians. This mathematical object is the heart of the Lyapunov exponent calculation. It tells us how the system expands or contracts volumes in its state space over long periods. As we are about to see, this exact same structure appears in the gradient calculation for RNNs.


--------------------------------------------------------------------------------


2.3 Deconstructing the Learning Process: The Loss Gradient

Now, let's analyze the training process. We have a generic RNN, which can be an LSTM, a PLRNN, or any other type, described by a map $F$ with parameters $\theta$:

$$z_t = F(z_{t-1}, \theta)$$

(We omit external inputs for simplicity). We define a total loss $L$ which is the sum of losses $L_t$ at each time step, $L = \sum_t L_t$. Our goal is to compute the gradient of this loss with respect to a parameter $\theta$.


--------------------------------------------------------------------------------


Proof: Deriving the Loss Gradient

Let's derive the gradient $\frac{\partial L}{\partial \theta}$. By linearity, this is the sum of the gradients of the per-time-step losses:

$$\frac{\partial L}{\partial \theta} = \sum_t \frac{\partial L_t}{\partial \theta}$$

Now, we focus on a single term $\frac{\partial L_t}{\partial \theta}$. The loss at time $t$, $L_t$, depends on the network's parameters $\theta$ through the entire history of states $\lbrace z_1, z_2, \dots, z_t\rbrace$. Using the chain rule, we can express this dependency as a sum over all prior time steps $r \le t$:

$$\frac{\partial L_t}{\partial \theta} = \sum_{r=1}^{t} \frac{\partial L_t}{\partial z_t} \frac{\partial z_t}{\partial z_r} \frac{\partial z_r}{\partial \theta}$$

The crucial term here is $\frac{\partial z_t}{\partial z_r}$, which describes how a change in an earlier state $z_r$ affects a later state $z_t$. We can "unwrap" this term by applying the chain rule recursively:

$$\frac{\partial z_t}{\partial z_r} = \frac{\partial z_t}{\partial z_{t-1}} \frac{\partial z_{t-1}}{\partial z_{t-2}} \cdots \frac{\partial z_{r+1}}{\partial z_r}$$

Recognizing that each term $\frac{\partial z_{k+1}}{\partial z_k}$ is simply the Jacobian of the network's transition function, $J(z_k)$, we can write this as a product series:

$$\frac{\partial z_t}{\partial z_r} = \prod_{k=r}^{t-1} J(z_k)$$

Substituting this back into our expression for the full gradient, we see the complete structure. The gradient of the loss with respect to the parameters is a sum over time, and each term in that sum contains a product series of Jacobians.


--------------------------------------------------------------------------------


2.4 The Fundamental Link: Jacobians in Dynamics and Learning

By comparing the definitions, the connection becomes strikingly clear:

| Concept                | Defining Formula                                                                 | Key Component           |
|------------------------|-----------------------------------------------------------------------------------|-------------------------|
| Max. Lyapunov Exponent | $\lambda_{max} = \lim_{T \to \infty} \frac{1}{T} \log \lvert \prod_{r=0}^{T-2} J(z_r) \rvert$ | Product of Jacobians   |
| Loss Gradient Term     | $\frac{\partial L_t}{\partial z_r} = \frac{\partial L_t}{\partial z_t} \left( \prod_{k=r}^{t-1} J(z_k) \right)$ | Product of Jacobians   |

Remark/Intuition: The Deep Connection

The very same mathematical structure—the product of Jacobians along a trajectory—governs two seemingly different phenomena:

1. System Dynamics: The long-term product of Jacobians determines the Lyapunov spectrum, which tells us if the system is stable, periodic, or chaotic. A system with exponents whose magnitudes are greater than zero will cause perturbations to grow exponentially.
2. Learning Dynamics: The product of Jacobians over finite time horizons determines how gradients propagate backward in time during training. If the Jacobians correspond to an expanding (chaotic) system, their product will grow exponentially, leading to exploding gradients. If they correspond to a highly contracting system, their product will shrink exponentially, leading to vanishing gradients.

Therefore, the exploding and vanishing gradient problem is not merely a numerical issue to be solved with clever tricks; it is a direct reflection of the intrinsic stability properties of the dynamical system the network is trying to learn. This insight is critical for designing architectures and training methods that are truly suited for the complex task of dynamical systems reconstruction.

A Study Book on Dynamical Systems Reconstruction

Table of Contents

1. The Challenge of Modeling Chaotic Systems with Recurrent Networks
  * 1.1 The Gradient Recursion and the Product of Jacobians
  * 1.2 The Inevitable Link Between Chaos and Exploding Gradients
2. Techniques for Gradient Stabilization
  * 2.1 Sparse Teacher Forcing: A Control-Theoretic Approach
    * 2.1.1 Core Idea and Formalism
    * 2.1.2 Optimal Forcing Interval and the Predictability Time
    * 2.1.3 Empirical Validation and Generative Properties
  * 2.2 Multiple Shooting: A Segmentation-Based Approach
    * 2.2.1 Core Idea and Formalism


--------------------------------------------------------------------------------


1. The Challenge of Modeling Chaotic Systems with Recurrent Networks

When attempting to use recurrent neural networks (RNNs) to reconstruct the dynamics of a system, a fundamental challenge arises if the underlying system is chaotic. The very nature of chaos, characterized by sensitive dependence on initial conditions, directly translates into a significant numerical problem during model training: the explosion of loss gradients. This section will detail the mathematical origins of this problem and explain why it is an unavoidable consequence of accurately modeling chaotic behavior.

1.1 The Gradient Recursion and the Product of Jacobians

To understand the problem, we must first recall how gradients are calculated in an RNN. The loss function, $L$, depends on the sequence of latent states, $\lbrace z_1, z_2, \dots, z_T\rbrace$, generated by the network. The gradient of the loss with respect to a state $z_t$ at a past time step involves a product of Jacobian matrices from that point forward in time. This relationship, derived from the chain rule of differentiation, can be expressed as:

$$\frac{\partial L}{\partial z_t} \propto \prod_{k=t}^{T-1} J_k$$


where $J_k$ is the Jacobian of the recurrent transition function $f$ at time step $k$. This product series is central to the issue of gradient stability.

1.2 The Inevitable Link Between Chaos and Exploding Gradients

Remark/Intuition

A defining characteristic of a chaotic system is that its maximum Lyapunov exponent is greater than zero ($\lambda_{max} > 0$). This positive exponent signifies that nearby trajectories in the state space diverge exponentially over time. For an RNN to successfully capture this chaotic behavior, its internal dynamics, represented by the function $f$, must also exhibit this exponential divergence.

When the model successfully learns the chaotic dynamics, the Jacobian matrices $J_k$ in the product series will have singular values with absolute values greater than one. Consequently, as the product series extends over time, its norm will grow exponentially. This leads directly to an explosion in the magnitude of the loss gradients.

This is not a flaw in the model or a bug in the training process; it is a fundamental and inevitable consequence. If we demand that our RNN genuinely reconstructs the underlying chaotic system, it must produce chaos. If it produces chaos, its loss gradients will explode. We cannot avoid this problem if our goal is a true dynamical systems reconstruction, such as modeling the Lorenz system, which would be incomplete without its characteristic chaotic behavior.


--------------------------------------------------------------------------------


2. Techniques for Gradient Stabilization

Given that exploding gradients are an intrinsic property of training recurrent models on chaotic data, we must employ specialized techniques to manage them. These methods aim to regularize the training process, allowing the network to learn the system's long-term structure without being overwhelmed by numerical instability.

2.1 Sparse Teacher Forcing: A Control-Theoretic Approach

One classical and effective technique for managing gradient explosion is teacher forcing. In our context, we use a specialized version known as sparse teacher forcing, which can be understood as a technique from control theory applied to the training of a dynamical model.

Remark/Intuition

The core idea behind sparse teacher forcing is to allow the model-generated trajectory to evolve freely for a period, letting it explore the dynamics of the state space, but then periodically "pulling it back" towards the true trajectory observed in the data. This prevents the model's trajectory from diverging too far, which in turn prevents the associated gradients from exploding.

This method strikes a crucial balance. If we correct the trajectory at every single time step (classical teacher forcing), the model only learns to make one-step-ahead predictions and fails to capture the essential long-term properties and geometric structure of the system's attractor. If we never correct it, we encounter the exploding gradient problem. Sparse teacher forcing, by applying corrections at well-chosen intervals, allows the model to learn about long-term behavior while keeping the training process stable.

This idea has historical roots in papers by Williams and Zipser (1989) and Pearlmutter (1990).

2.1.1 Core Idea and Formalism

Let us formalize the sparse teacher forcing mechanism.

Definition: Model and Control Series Setup

1. Observed Data: We assume an observed time series of length $T$, denoted as $\lbrace x_1, x_2, \dots, x_T\rbrace$, where each $x_t \in \mathbb{R}^n$.
2. Recurrent Model: Our RNN generates a sequence of latent states $\lbrace z_1, z_2, \dots, z_T\rbrace$, where $z_t \in \mathbb{R}^m$, according to the transition function $z_{t+1} = f(z_t)$.
3. Observation Model: The observed data $x_t$ is related to the latent state $z_t$ through a decoder or observation function, $g$. For simplicity, let's consider a linear mapping:
  
  $$x_t = C z_t$$
  
  where $C$ is a matrix mapping from the $m$-dimensional latent space to the $n$-dimensional observation space.
4. Control Series: To "force" the model back to the real trajectory, we need an estimate of the "true" latent states directly from the data. We construct a control series, $\lbrace\tilde{z}_1, \tilde{z}_2, \dots, \tilde{z}_T\rbrace$, by inverting the decoder model. For our linear example, this can be achieved using the pseudo-inverse $C^+$:  
   
  $$\tilde{z}_t = C^+ x_t  $$
  
  where $C^+$ is the pseudo-inverse of $C$.

More generally, this involves solving a regression problem to find an estimate of the latent state that corresponds to a given observation.

Definition: The Sparse Forcing Update Rule

We define a set of forcing times, $\mathcal{T}$, at regular intervals determined by a forcing interval $\tau \in \mathbb{N}^+$.

$$\mathcal{T} = \lbrace n \cdot \tau \mid n \in \mathbb{N}\rbrace$$  

During training, the latent state $z_{t+1}$ is computed using a conditional rule:  

$$z_{t+1} = \begin{cases} f(\tilde{z}_t) & \text{if } t \in \mathcal{T} \ f(z_t) & \text{if } t \notin \mathcal{T} \end{cases}$$     

Remark/Intuition: How Forcing Cuts Off Gradients

The key to stabilizing training lies in how this update rule affects backpropagation. At a forcing time step $t \in \mathcal{T}$, the input to the function $f$ is $\tilde{z}\_t$. Since $\tilde{z}\_t$ is derived directly from the data $x_t$, it is treated as a constant with respect to the previous latent state $z_{t-1}$. Therefore, the gradient chain is broken:  

$$\frac{\partial z_{t+1}}{\partial z_t} = 0 \quad \text{for } t \in \mathcal{T}$$  

This effectively "resets" the product of Jacobians every $\tau$ steps, preventing it from growing uncontrollably over long time horizons.

2.1.2 The Predictability Time: Choosing an Optimal Forcing Interval

The choice of the forcing interval $\tau$ is critical. An improperly chosen $\tau$ can either fail to solve the gradient problem or prevent the model from learning essential dynamics.

Theorem: Optimal Forcing Interval

The optimal choice for the forcing interval $\tau$ is approximately given by the system's predictability time, which is inversely related to the maximum Lyapunov exponent $\lambda_{max}$. 

$$\tau_{optimal} \approx \frac{\ln(2)}{\lambda_{max}}$$ 

Remark/Intuition

This relationship is highly intuitive.

* A larger $\lambda_{max}$ implies a more chaotic system where trajectories diverge more quickly. This requires a smaller $\tau$ to provide more frequent corrections and keep the model's trajectory from straying too far.
* A smaller $\lambda_{max}$ implies a less chaotic system with slower divergence, allowing for a larger $\tau$ and giving the model more freedom to evolve on its own.

Empirical evidence shows that choosing $\tau$ is a delicate balance.

* If $\tau$ is too small, the system is over-regularized. It only learns one-step-ahead predictions and fails to capture the long-term, global structure of the dynamics.
* If $\tau$ is too large, the forcing is too infrequent to prevent the product of Jacobians from diverging, and the exploding gradient problem persists.

2.1.3 Empirical Results and Model Validation

The effectiveness of choosing $\tau$ based on the predictability time has been demonstrated on real-world data.

Examples

* Brain Recordings: This technique has been successfully applied to the reconstruction of dynamics from complex, real-world time series such as human fMRI and EEG data.
* Performance Metrics: By plotting metrics of dynamical similarity (e.g., state-space divergence or Heller distance) against different values of $\tau$, studies show that the error indeed reaches a minimum when $\tau$ is chosen to be near the predictability time of the empirical time series.

Remark/Intuition: The Generative Nature of the Model

It is crucial to emphasize that a successfully trained RNN for dynamical systems reconstruction is a generative model. The patterns and trajectories it produces after training (e.g., simulated brain activity patterns) are not simply literal fits to the training data. Instead, they are novel generations synthesized by the model itself, demonstrating that it has learned the underlying rules and structure of the system's dynamics.

2.2 Multiple Shooting: A Segmentation-Based Approach

A related idea, originating from the dynamical systems literature, is a technique known as multiple shooting.

Remark/Intuition

The core concept of multiple shooting is to divide the full time series into smaller, more manageable segments. Instead of trying to propagate a single trajectory from one initial condition across the entire time series, we estimate a new initial condition for each segment. The optimization process then involves two simultaneous goals:

1. Fit the trajectory within each individual segment.
2. Ensure that the trajectories are continuous across the boundaries of the segments (i.e., the end state of one segment matches the initial state of the next).

This approach inherently breaks the long-term dependencies that lead to exploding gradients by limiting the backpropagation horizon to the length of a single segment. A key reference for this method is a 2004 paper by Fattal et al. in Physical Review.

2.2.1 Formalism: The Optimization Problem

Let's formalize the objective for multiple shooting.

Definition: Multiple Shooting Loss Function

1. We partition the time series into N segments.
2. For each segment $n \in \lbrace 1, \dots, N\rbrace$, we estimate a unique initial condition, $z_0^{(n)}$, which becomes a trainable parameter.
3. The overall objective is to minimize a loss function that sums the errors over all segments. A typical squared-error loss would be:

$$\min_{\theta, \{z_0^{(n)}\}_{n=1}^N} \sum_{n=1}^{N} \sum_{t=1}^{T_{seg}} \left\| x_t^{(n)} - g(z_t^{(n)}) \right\|^2$$

where $\theta$ represents the parameters of the recurrent model $f$, $T_{seg}$ is the length of each segment, and the evolution of $z_t^{(n)}$ depends on the initial condition $z_0^{(n)}$. The constraint of temporal continuity between segments must also be enforced, often through additional terms in the loss function or as part of the optimization procedure.

A Study Book on Reconstructing Dynamical Systems

Table of Contents

1. Multiple Shooting for Gradient Stability
2. A Control-Theoretic Perspective on Training
3. Generalized Teacher Forcing
4. Summary of Techniques


--------------------------------------------------------------------------------


Multiple Shooting for Gradient Stability

Remark/Intuition: The Challenge of Long Time Series

When training models of dynamical systems on long time series, propagating gradients back through many time steps can lead to the well-known problems of exploding or vanishing gradients. This makes it difficult for gradient descent-based optimizers to find good parameters. One effective strategy to mitigate this is Multiple Shooting. The core idea is to break the long time series into smaller, more manageable segments or "batches." We then train the model on these individual segments, but with an additional constraint that ensures the trajectory remains continuous across the segment boundaries.

Definition: Multiple Shooting with a Continuity Constraint

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

Remark/Intuition: Knitting Trajectories Together

This technique can be visualized as learning short trajectory pieces independently and then "knitting them together." By enforcing continuity across the boundaries of these shorter intervals, we can reconstruct a globally consistent long-term trajectory while keeping the backpropagation paths short and the gradients stable.


--------------------------------------------------------------------------------


A Control-Theoretic Perspective on Training

Remark/Intuition: From Observation to Control

Control Theory is a branch of mathematics focused on influencing the behavior of dynamical systems. Instead of merely observing a system's evolution, the goal is to find optimal control signals that can steer the system towards a desired state or behavior. For example, a control signal might be used to stabilize a chaotic system onto a stable limit cycle.

This concept is directly relevant to training neural models of dynamical systems. Techniques like teacher forcing can be viewed through a control-theoretic lens. In this context, the loss gradients act as the control signal, actively pushing the model's trajectory towards the true data trajectory.

Example: A Biophysical Neuron Model

Consider a biophysical model of a spiking neuron with a voltage signal and two gating variables. We can introduce a control term directly into the system's differential equations. This term is proportional to the difference between the observed voltage from a real cell and the voltage produced by the model:

$$\frac{dV}{dt} = \dots + \kappa (V_{\text{observed}} - V_{\text{model}})$$

Here, $\kappa$ is a parameter that determines the strength of the control. When attempting to estimate the system's parameters from real voltage recordings, this control term serves two purposes:

1. Guidance: Much like sparse teacher forcing, it guides the model's trajectory, preventing it from diverging wildly from the real data.
2. Loss Landscape Smoothing: For a sufficiently large $\kappa$, this control term can smooth out the loss function, making it more convex. A more convex loss landscape is significantly easier for gradient-based optimizers to navigate, reducing the risk of getting stuck in poor local minima.

During training, the control parameter $\kappa$ would need to be regulated, similar to how teacher forcing schedules are used. This approach provides a foundation for more advanced techniques.


--------------------------------------------------------------------------------


Generalized Teacher Forcing

Introduced in a 2023 ICML paper by Florian Hess et al., Generalized Teacher Forcing is a powerful technique that formalizes the control-theoretic approach in a graded, adaptive manner.

Definition: The Weighted Average State

Unlike sparse teacher forcing, which replaces the model's state with the true state at certain time steps, generalized teacher forcing creates a new state, $\tilde{z}_t$, which is a weighted average of the model-propagated state and an estimate from the true data.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Teacher Forcing)</span></p>

Generalized teacher forcing creates a new state, $\tilde{z}_t$, which is a weighted average of the model-propagated state and an estimate from the true data.

The new state at time $t$ is defined as:

$$\tilde{z}_t = (1 - \alpha) f_{\theta}(\tilde{z}_{t-1}) + \alpha \hat{z}_t$$

where:
* $\tilde{z}_t$ is the new, "controlled" state at time $t$.
* $f_{\theta}(\tilde{z}\_{t-1})$ is the one-step forward propagation of the previous state using the learned dynamics model $f_{\theta}$.
* $\hat{z}_t$ is the latent state estimated from the real data observation $x_t$ at time $t$, typically obtained by inverting a decoder model.

</div>



* $\alpha$ is a weighting parameter between 0 and 1 that balances the influence of the model's internal dynamics and the guidance from the real data.

Remark/Intuition: An Old Idea, Revisited

This idea of blending model dynamics with data was first introduced by Kenji Doya in 1992. However, it remained dormant for a long time, partly because a principled method for choosing the mixing parameter $\alpha$ was not established. The key contribution of the modern approach is to provide a smart, adaptive choice for $\alpha$.

Derivation: The Product Series of Jacobians

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

Theorem: An Optimal Choice for $\alpha$

The derivation above reveals a clear strategy for controlling gradient flow. The magnitude of the gradients is determined by the singular values of the Jacobian product. If the largest singular value is much greater than 1, gradients will explode. If it is much less than 1, they will vanish.

This suggests that $\alpha$ should be chosen at each time step to regulate the singular values of the individual Jacobians, $G_t$. A smart choice for $\alpha$ at time $t$ is one that forces the norm of the controlled Jacobian, $J_t$, to be close to 1. We can achieve this by setting:

$$1 - \alpha_t = \frac{1}{\sigma_{\max}(G_t)}$$

which implies:

$$\alpha_t = 1 - \frac{1}{\sigma_{\max}(G_t)}$$

where $\sigma_{\max}(G_t)$ is the maximum singular value of the dynamics Jacobian $G_t$. By choosing $\alpha$ in this way, we can adaptively keep the gradients in check, preventing both exploding and vanishing gradient problems.

Remark/Intuition: Practical Implementation

A significant caveat is that computing a full Singular Value Decomposition (SVD) to find $\sigma_{\max}$ at every single time step is computationally prohibitive. However, this theoretical result provides the foundation for practical approximations. In practice, one can:

* Use computationally efficient proxies for the SVD.
* Update the value of $\alpha$ less frequently, for instance, every 10 time steps, rather than at every step.

This makes the automatic, adaptive regulation of gradients feasible for training on real-world systems. Furthermore, an optimal choice of $\alpha$ not only stabilizes gradients but also tends to make the loss landscape nearly convex, greatly aiding the optimization process.


--------------------------------------------------------------------------------


Summary of Techniques

To effectively reconstruct dynamical systems, especially from real-world data, it is crucial to employ techniques that manage gradient flow during training and allow the model to explore the state space. The key methods discussed are:

* Multiple Shooting: This technique addresses the exploding/vanishing gradient problem in long sequences by breaking the trajectory into shorter segments and enforcing continuity between them with a regularization term.
* Sparse Teacher Forcing: (Referenced as a point of comparison) This involves occasionally replacing the model's predicted state with the ground truth state, providing a strong corrective signal.
* Generalized Teacher Forcing: A more sophisticated, control-theoretic method that creates a new state at each time step as a weighted average of the model's prediction and a data-derived estimate. The weighting parameter, $\alpha$, can be chosen adaptively to regulate the singular values of the system's Jacobians, ensuring stable gradients and leading to a smoother, more convex loss landscape.



[SINDy](/subpages/dynamical-systems/dynamical-systems-in-machine-learning/sindy/)



