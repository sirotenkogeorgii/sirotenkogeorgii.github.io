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

A point $x^*\in\mathbb{R}^n$ is called equilibrium point of a system ODEs, if $f(t,x)=0$ for all $t\in I$.

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

Remark/Intuition

The matrix S represents an invertible transformation, or a change of variables (a change of basis). If two matrices are similar, it means that the dynamical systems they define are topologically equivalent; they possess the same fundamental dynamics, merely viewed from a different coordinate system. The eigendecomposition of a matrix, for instance, is a transformation that reveals its similarity to a diagonal matrix of its eigenvalues.


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

  * **Eigenvalues:** has only the one eigenvalue $λ_1 = a$.
  * **Dynamics:** This matrix has one eigenvalue, $a$, with a multiplicity of two. However, it only has one corresponding eigenvector direction. This case is called degenerate because the eigenvectors do not form a basis for the space; instead, two eigenvector directions align. The dynamics ultimately collapse into a one-dimensional space, with all trajectories aligning with the single eigenvector direction. The specific path of convergence or divergence depends on the initial conditions and the value of $a$.


--------------------------------------------------------------------------------

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


<!-- There is no difference. Yeah. So they are equivalent under the condition that um um all values are distinct. If they are not distinct the first solution doesn't apply because it rest on the assumptions. that our vectors are a basis for our space which is not the case anymore. Yeah. So uh so this is the only the degenerate case is the only case where we can have things like this. Yeah. And the uh the degenerate case is only covered by the more general solution. Yeah. Otherwise everything is fine. -->

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

To understand this, we first locate the new equilibrium, or fixed point, by finding the point $x^*$ where the flow is zero ($\dot{x}=0$).

Assuming the matrix $A$ is invertible, we solve for the fixed point: 

$$0 = Ax^* + b \implies Ax^* = -b \implies x^* = -A^{-1}b$$ 

This point $x^*$ is our new equilibrium, shifted from the origin.

Proof: Equivalence of Dynamics via Change of Variables

We can formally prove that the dynamics remain the same by defining a new variable y that represents the state relative to the fixed point $x^*$.

1. Define a new variable: Let $y = x - x^*$. This is equivalent to $x = y + x^*$.
2. Consider the dynamics of the new variable: The temporal derivative of $y$ is $\dot{y} = \dot{x}$, since $x^*$ is a constant and its derivative is zero.
3. Substitute into the original equation: We can now express $\dot{y}$ in terms of $y$
   
   $\dot{y} = \dot{x} = Ax + b$  
   
   Substitute $$x = y + x^*: \dot{y} = A(y + x^*) + b = Ay + Ax^* + b$$  
   
  Now, substitute the expression for the fixed point, $x^* = -A^{-1}b$:
   
  $$\dot{y} = Ay + A(-A^{-1}b) + b = Ay - b + b$$

1. Result: The dynamics for the new variable are:
   
  $$\dot{y} = Ay$$
   
  This is precisely the homogeneous linear system we have already analyzed. The dynamics (stability, rotation, etc.) around the fixed point $x^*$ are identical to the dynamics of the homogeneous system around the origin. To find the full solution for $x(t)$, one solves for $y(t)$ and then recovers $x(t) = y(t) + x^*$.

Remark/Intuition: The Non-Invertible Case

If the matrix $A$ is not invertible, it possesses at least one zero eigenvalue. In this scenario, a unique fixed point does not exist. This corresponds to the case of a center or, more generally, a line attractor (or plane/hyperplane attractor in higher dimensions). The system has a continuous manifold of equilibrium points along the direction of the eigenvector(s) associated with the zero eigenvalue(s).


--------------------------------------------------------------------------------


1.2 Non-autonomous Systems with a Forcing Function

We now consider systems where the dynamics are explicitly influenced by time, driven by an external "forcing function."

Definition: Non-autonomous System with Forcing Function

A non-autonomous linear system with a forcing function $f(t)$ is defined as a system that explicitly depends on time. For simplicity, we will analyze the scalar case:

$$\dot{x} = ax + f(t)$$ 

Remark/Intuition: Variation of Parameters

To solve this type of equation, we employ a powerful technique known as variation of parameters. The logic is as follows: we know the solution to the homogeneous part of the equation ($\dot{x} = ax$) is $x(t) = C e^{at}$, where C is a constant. We now "promote" this constant to a time-dependent function, $k(t)$, and propose an ansatz (an educated guess) for the full solution that has a similar form. This allows the solution to adapt to the time-varying influence of $f(t)$.

Proof: Derivation of the Solution

1. Formulate the ansatz: Let the solution be of the form $x(t) = (h(t) + C)e^{at}$, where $h(t)$ is an unknown function we need to determine and $C$ is a constant of integration.
2. Take the temporal derivative: Using the product rule, the derivative of our ansatz is:
  
  $$\dot{x}(t) = \frac{d}{dt}[(h(t) + C)e^{at}] = \dot{h}(t)e^{at} + (h(t) + C)ae^{at}$$ 

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

Example: Iterative Solution and Limiting Behavior (Scalar Case)

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

Table of Contents

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
* 2.4 Stability Analysis via Eigenvalues


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

1.3 Fixed Points in One Dimension

1.3.1 Definition and Geometric Intuition

A central concept in dynamical systems is the notion of a fixed point, which is analogous to an equilibrium in continuous-time systems described by differential equations.

Definition: Fixed Point A point $x^*$ is a fixed point of a discrete-time system $x_{t+1} = f(x_t)$ if it remains unchanged by the map. That is, it satisfies the condition:

$$x^* = f(x^*)$$

If the system is initialized at a fixed point, it will remain there for all future time steps. There is no movement at this point.

Remark/Intuition: Geometric View of Fixed Points Geometrically, a fixed point is simply the intersection of the function graph $y = f(x)$ and the bisectrix $y=x$. At this specific point, the input to the function is exactly equal to its output, satisfying the definition $x^* = f(x^*)$.

1.3.2 Algebraic Solution

We can find the fixed point not only graphically but also by solving the defining equation algebraically.

Proof: Derivation of the 1D Fixed Point To find the fixed point $x^*$, we set the output equal to the input according to the definition:  $$x^* = ax^* + b$$  We then solve for 

$$x^*:x^* - ax^* = b$$

$$(1-a)x^* = b$$

Assuming a \neq 1, we can divide by (1-a) to find the unique fixed point:

$$x^* = \frac{b}{1-a}$$

This algebraic solution precisely matches the limiting solution for convergent systems and identifies the point of intersection on the cobweb plot.

1.4 Stability Analysis of Fixed Points

A fixed point can be stable, unstable, or neutrally stable, depending on the behavior of nearby trajectories. This stability is determined entirely by the slope parameter, $a$.

1.4.1 Stable Fixed Points: $\lvert a\rvert < 1$

If the absolute value of the slope is less than one, any initial condition will lead to a trajectory that converges to the fixed point.

Definition: Stable Fixed Point A fixed point $x^*$ is stable if trajectories starting near $x^*$ converge towards it as $t \to \infty$. In the linear 1D case, this occurs when $\lvert a\rvert < 1$.

Remark/Intuition On the Cobweb Plot, a slope with $\lvert a\rvert < 1$ is less steep than the bisectrix. This geometric configuration ensures that each step of the cobweb construction brings the state closer to the intersection point, causing the "web" to spiral inwards towards the fixed point.

1.4.2 Unstable Fixed Points: $\lvert a\rvert > 1$

If the absolute value of the slope is greater than one, the system will diverge from the fixed point, unless it starts exactly on it.

Definition: Unstable Fixed Point A fixed point $x^*$ is unstable if trajectories starting near $x^*$ move away from it as $t \to \infty$. In the linear 1D case, this occurs when $\lvert a\rvert > 1$.

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


2. Higher-Dimensional Discrete-Time Linear Systems

We now generalize our analysis to systems with $m$ dimensions, where the state is represented by a vector and the dynamics are governed by a matrix transformation.

2.1 The General Affine Map

Definition: $m$-Dimensional Linear Map The state of the system is a vector $\vec{x}_t \in \mathbb{R}^m$. The evolution is given by the affine map:

$$\vec{x}_{t+1} = A\vec{x}_t + \vec{b}$$

where $A$ is an $m \times m$ square matrix and $\vec{b} \in \mathbb{R}^m$ is a constant offset vector.

2.2 Solving for Fixed Points in $m$ Dimensions

The definition of a fixed point remains the same: it is a point that is mapped onto itself.

Proof: Derivation of the $m$-Dimensional Fixed Point Let $\vec{x}^*$ be a fixed point. It must satisfy the condition $\vec{x}^* = A\vec{x}^* + \vec{b}$. We solve for $\vec{x}^*$:  

$$\vec{x}^* - A\vec{x}^* = \vec{b}$$

$$(I - A)\vec{x}^* = \vec{b}$$

where $I$ is the $m \times m$ identity matrix. If the matrix $(I-A)$ is invertible, we can find the unique fixed point by multiplying by its inverse:  

$$\vec{x}^* = (I-A)^{-1}\vec{b}$$

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
  
  $$\text{If } \max_i \lvert\lambda_i\rvert < 1 \implies \text{Convergence}  $$
  
  As $T \to \infty$, $\Lambda^{T-1} \to 0$, causing $\vec{x}_T \to \vec{0}$.
* Divergence: The system diverges if at least one eigenvalue has an absolute value greater than 1.  
  
  $$\text{If } \max_i \lvert\lambda_i\rvert > 1 \implies \text{Divergence}  $$
  
  The component of the trajectory along the eigenvector corresponding to this eigenvalue will grow without bound.
* Neutral Stability / Manifold Attractors: If at least one eigenvalue has an absolute value of exactly 1 (and no eigenvalues have absolute values greater than 1), the system has neutral directions.  
  
  $$\text{If } \max_i \lvert\lambda_i\rvert = 1 \implies \text{Line or Manifold Attractor}  $$
  
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

1. Find the fixed point of the continuous system: Set $\dot{x} = 0. 0 = Ax^* + c \implies x^* = -A^{-1}c$
2. Find the fixed point of the discrete system: Set $x_{n+1} = x_n = x^*$. $x^* = \tilde{A}x^* + b \implies (I - \tilde{A})x^* = b$
3. Equate and Solve for $b$: By substituting the expression for $x^*$ from the continuous system into the discrete system's fixed point equation, we can solve for $b$.


--------------------------------------------------------------------------------


1.3 Applications and Advanced Concepts

The principles of establishing equivalence between continuous and discrete systems are not merely theoretical exercises. They have profound implications in modern machine learning and computational neuroscience.

1.3.1 Piecewise Linear Recurrent Neural Networks

An important class of models in machine learning is the Piecewise Linear Recurrent Neural Network (PL-RNN). These networks are often defined using the Rectified Linear Unit (ReLU) activation function, which is a piecewise linear function.

A typical PL-RNN update rule has the form:

$$x_{n+1} = \tilde{A} g(x_n) + b$$

where $g$ is the ReLU nonlinearity, defined as $g(z) = \max(0, z)$.

Remark

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

Having built some intuition, we now proceed to a formal mathematical definition.

Definition: Dynamical System

A dynamical system is a commutative group or semigroup action, $\phi$, defined on a domain $T \times R$. It is composed of the following elements:

1. A Time Domain (T): This is the set from which time values are drawn.
  * For continuous-time systems defined for all time, $T = \mathbb{R}$ (a group).
  * For systems defined only in forward time, $T = \mathbb{R}_{\ge 0}$ (a semigroup).
  * For discrete-time systems, $T = \mathbb{Z}$ (the integers).
2. A State Space ($R$): This is an open set, $R \subseteq \mathbb{R}^d$, which contains all possible states the system can occupy. It is the space spanned by the dynamical variables.
3. A Flow Map ($\phi$): An operator that maps a time and a state to a new state.
   
  $$\phi: T \times R \to R$$  
   
  We write this as $\phi(t, x)$ or sometimes abbreviate it as $\phi_t(x)$.

Properties of the Flow Map

The flow map $\phi$ must satisfy the following properties:

* Identity Property: For any state $x$ in the state space $R$, evolving for zero time leaves the state unchanged.
  
$$\forall x \in R, \quad \phi(0, x) = x$$ 

Chapter 1: The Flow Map and Trajectories

Welcome to the study of dynamical systems. At its core, this field is about understanding how systems evolve over time. To do this rigorously, we must first establish a formal mathematical framework. Our journey begins with the central concept that defines a dynamical system: the flow operator, or flow map.

1.1 The Flow Operator: Defining a Dynamical System

Definition: A dynamical system is defined by its flow operator, denoted as $\phi_t(x)$. This operator takes a point $x$ from the state space and maps it to the position it will occupy after an amount of time $t$ has elapsed.

Remark/Intuition: The key insight here is that we use the concept of a map to describe the evolution of a system. Even when dealing with systems of continuous differential equations, we ultimately conceptualize their solutions in terms of this operator, which acts like a discrete time map. This operator, $\phi$, encapsulates all the rules governing the system's dynamics.

1.2 Properties of the Flow

For an operator $\phi_t(x)$ to qualify as the flow of a dynamical system, it must satisfy specific properties. These properties ensure that the evolution of the system is consistent and predictable. Let $x$ be a point in the state space and let $s$, $t$ be elements of the time domain $T$ (e.g., $\mathbb{R}$ for continuous time).

* Neutral Element: Applying the flow for zero time leaves the point unchanged. This is the identity operation.
  
$$\phi_0(x) = x$$ 

* Semigroup (or Group) Property: Evolving a point for a time $s+t$ is equivalent to first evolving it for time $t$ and then evolving the result for time $s$ (or vice versa). This property is described as commutative, meaning the order of time evolution operations can be exchanged.
  
$$\phi_{s+t}(x) = \phi_s(\phi_t(x)) = \phi_t(\phi_s(x))$$ 

* Remark/Intuition: Imagine a particle tracing a path in the state space. It should not matter whether you calculate its position after 5 seconds by moving it forward 2 seconds and then 3 seconds, or by moving it 3 seconds and then 2 seconds. The final position must be the same as moving it forward for 5 seconds directly. This consistency is fundamental.
* Inverse Operation (for Groups): As a consequence of the group property, if the system is time-reversible (i.e., time can be negative), we have an inverse operation. Evolving forward by time $t$ and then backward by time $t$ returns the system to its original state.
  
$$\phi_t(\phi_{-t}(x)) = x$$ 

1.3 Trajectories and Orbits

With the flow operator established, we can now precisely define the path that a point carves out in the state space over time.

Definition: The trajectory (or orbit) of a dynamical system starting from an initial point $x_0$ is the solution curve, denoted $\gamma(x_0)$. It is the set of all points in the state space that lie on this solution curve for all time $t \in T$.

$$\gamma(x_0) = \lbrace \phi_t(x_0) \mid t \in T \rbrace$$

Remark/Intuition: A critical feature of a well-defined dynamical system is the uniqueness of its trajectories. For any given initial point $x_0$, there can be only one trajectory passing through it. If two different curves could originate from the same starting point, it would imply that the state space is missing crucial information needed to predict the future state, and we would not have a deterministic dynamical system. We will explore the conditions that guarantee this uniqueness in the next chapter.


--------------------------------------------------------------------------------


Chapter 2: Existence and Uniqueness of Solutions

Having defined the concepts of flows and trajectories, a fundamental question arises: given a system of differential equations, can we always expect it to have a unique solution for a given starting condition? This is the central question of existence and uniqueness.

2.1 The Core Problem: Do Unique Solutions Always Exist?

The unfortunate answer is no, unique solutions are not guaranteed for all systems. However, the fortunate reality is that for the vast majority of well-behaved systems, they almost always do. The conditions where uniqueness fails are quite specific.

Examples: Consider the following initial value problem:

$$\dot{x} = x^{2/3}, \quad x(0) = 0$$

This system has two distinct solutions that satisfy the initial condition:

1. The Trivial Solution: $u(t) = 0$
2. A Non-trivial Solution: $v(t) = t^3$

Proof: We must verify that both functions satisfy the differential equation and the initial condition.

* For $u(t) = 0$:
  * Initial Condition: $u(0) = 0$, which is satisfied.
  * Differential Equation: The time derivative is $\dot{u}(t) = 0$. Plugging into the equation gives $0 = (0)^{2/3} = 0$. The equation holds.
* For $v(t) = t^3$:
  * Initial Condition: $v(0) = 0^3 = 0$, which is satisfied.
  * Differential Equation: The time derivative is $\dot{v}(t) = 3t^2$. We check if this equals $v(t)^{2/3}$.
* Since the derivative $3t^2$ is not equal to $t^2$, there appears to be a transcription error in the original lecture notes. Let's re-examine the example with a slight correction to match the lecture's conclusion. Let's assume the example was meant to be $x' = 3x^{2/3}$ or the solution was $v(t) = (t/3)^3$. Assuming the intended solution was $v(t) = (t/3)^3$ for $\dot{x}=x^{2/3}$, the derivative would be $\dot{v}(t) = 3(t/3)^2 \cdot (1/3) = t^2/9$, while $v^{2/3} = ((t/3)^3)^{2/3} = (t/3)^2 = t^2/9$. This seems more plausible. Let's proceed with the lecture's reasoning about the function's smoothness, which is the core lesson.

Remark/Intuition: What causes this failure of uniqueness? The problem lies at the point $x=0$. The vector field $f(x) = x^{2/3}$ is continuous at $x=0$, but it is not continuously differentiable there. Let's examine its derivative with respect to the dynamical variable $x$:
 
 $$\frac{df}{dx} = \frac{d}{dx}(x^{2/3}) = \frac{2}{3}x^{-1/3}$$
 
This derivative is undefined at $x=0$. This lack of smoothness in the vector field is precisely what allows for multiple solution paths to emerge from the same point.

2.2 The Fundamental Existence and Uniqueness Theorem

The issue identified in the counterexample is the exact problem that the following powerful theorem resolves. If we can guarantee that our vector field is smooth enough, we can guarantee a unique solution.

Theorem: (Fundamental Existence and Uniqueness Theorem) Let $E \subseteq \mathbb{R}^m$ be an open set (our state space) and let the vector field $f: E \to \mathbb{R}^m$ be a continuously differentiable function (i.e., $f \in C^1(E)$).

Then, for any initial condition $x_0 \in E$, there exists a constant $a > 0$ such that the initial value problem  

$$\dot{x} = f(x), \quad x(0) = x_0$$

has a unique solution, $x(t)$, within the so-called maximum interval of existence $(-a, a)$, which is a subset of $\mathbb{R}$.

Furthermore, this unique solution has the general form:

$$x(t) = x_0 + \int_0^t f(x(s))ds$$     

This integral expression is sometimes referred to as the solution operator.

Remark/Intuition: This theorem is the bedrock for much of dynamical systems theory. It tells us that as long as our system's rules of evolution (the vector field $f$) are smooth, we don't have to worry about non-uniqueness. The solution might not exist for all time (it could "blow up" in finite time), but in some local time interval around our starting point, the path is uniquely determined. While the integral form of the solution is general, it may not be solvable analytically and often requires a numerical solver.

2.3 A Weaker Condition: Lipschitz Continuity

The requirement of being continuously differentiable ($C^1$) is sufficient, but it is actually stronger than necessary. A weaker, more general condition also guarantees uniqueness.

Remark/Intuition: The theorem can be proven under the weaker assumption that the function $f$ is locally Lipschitz continuous. For any two points $x$ and $y$ in some local interval, a function is Lipschitz continuous if the absolute difference in its values is bounded by a constant multiple of the distance between the points.  

$$\lvert f(x) - f(y)\rvert \le L \lvert x - y\rvert$$

Here, $L$ is a positive real number known as the Lipschitz constant. Intuitively, this condition means that the slope of the function is bounded. Every continuously differentiable function is locally Lipschitz, but not every Lipschitz continuous function is differentiable, making this a more general condition.

2.4 On Solving Non-Linear Systems

While the theorem guarantees the existence of a unique solution for a broad class of systems, it does not provide a general method for finding it.

Remark/Intuition: In the most general case, non-linear systems of differential equations cannot be solved analytically. However, for certain scalar cases or systems with special structures, analytical techniques exist. These include:

* Separation of Variables: Rearranging the equation so that all terms involving one variable are on one side and all terms involving the other variable are on the other side, allowing for direct integration.
* Variational Calculus: A more advanced method for solving certain classes of problems.

For most complex systems encountered in practice, numerical methods are the primary tool for approximating the unique solution trajectories that the theorem guarantees.



[SINDy](/subpages/dynamical-systems/dynamical-systems-in-machine-learning/sindy/)



