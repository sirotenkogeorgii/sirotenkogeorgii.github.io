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

The **state space** is the set of all possible states a system can occupy. For a system with p variables, the state space can be visualized as a p-dimensional space (typically \mathbb{R}^p) where each axis corresponds to one of the system's variables $(x_1, x_2, \dots, x_p)$.

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

### Analysis of Linear Systems

Linear systems are a cornerstone of dynamical systems theory. While they don't exhibit the complex behaviors of nonlinear systems (like chaos), they are fundamental for two reasons:

1. They are one of the few classes of systems that can be solved completely analytically.
2. The behavior of a nonlinear system in the close vicinity of an equilibrium point can often be accurately approximated by a linear system.

### Rewriting Higher-Order Systems

A key insight is that any higher-order ODE can be rewritten as a system of first-order ODEs. This simplifies our analysis, as we only need to develop tools for first-order systems.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Conversion of Higher-Order ODEs)</span></p>

An $m^{th}$-order one-dimensional ODE of the form $F(\frac{d^m x}{dt^m}, \dots, \frac{dx}{dt}, x, t) = 0$ can always be rewritten as a system of $m$ coupled first-order ODEs in an $m$-dimensional state space.

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

### The One-Dimensional Case: $$\dot{x} = ax$$

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

5. **Define a new constant** $\tilde{C} = \pm e^C$ to absorb the absolute value and the constant term. This gives the general solution:
  
  $$x(t) = \tilde{C} e^{at}$$

6. **Apply the initial condition** $x(0) = x_0$. At $t=0$, we have $x(0) = \tilde{C}e^0 = \tilde{C}$. Therefore, $\tilde{C} = x_0$.

The final solution is: 

$$x(t) = x_0 e^{at}$$

Remark/Intuition: Stability of the Equilibrium at $x=0$

The point $x=0$ is an **equilibrium point** (or **fixed point**) because if the system starts there ($x_0=0$), its derivative $\dot{x}$ is zero, and it remains there for all time. The stability of this equilibrium depends entirely on the sign of the coefficient $a$.

* **Case 1: $a > 0$ (Unstable Equilibrium)**
  * The solution $x(t) = x_0e^{at}$ grows exponentially.
  * If the system is perturbed even slightly from the origin, it will move away from it at an accelerating rate.
  * The vector field on the 1D line points away from the origin on both sides. This is also called a repeller or source.
* **Case 2: $a < 0$ (Stable Equilibrium)**
  * The solution $x(t) = x_0e^{at}$ decays exponentially to zero.
  * No matter where the system starts (besides the origin itself), it will always return to the equilibrium at $x=0$.
  * The vector field points towards the origin from both sides. This is also called an attractor or sink.
* **Case 3: $a = 0$ (Neutrally or Marginally Stable)**
  * The equation becomes $\dot{x} = 0$, meaning the velocity is always zero.
  * Wherever the system starts, it stays there forever. The entire $x$-axis is a continuum of fixed points.
  * It is called "marginally" stable because a small perturbation does not return to the original point, but it also doesn't grow unboundedly; it simply moves to a new fixed point.

### Higher-Dimensional Linear Systems: $$\dot{x} = Ax$$

Now we consider a system of $m$ coupled linear ODEs, where $x \in \mathbb{R}^m$ and $A$ is an $m \times m$ matrix. 

$$\dot{x} = Ax, \quad x(0) = x_0 $$

*Proof:* Pedagogical Derivation of the General Solution

To find the solution, we can use an ansatz inspired by the 1D case, proposing a solution of a similar exponential form.

1. Propose a solution form, where $v$ is a constant vector and $\lambda$ is a scalar: 
    
  $$x(t) = v e^{\lambda t}$$
  
  We assume $v \neq 0$ to find a non-trivial solution.
2. Substitute the ansatz into the ODE. First, find the derivative \dot{x}: 
   
  $$\dot{x} = \frac{d}{dt}(v e^{\lambda t}) = \lambda v e^{\lambda t}$$

3. Set the two sides of the ODE equal:
 
  $$\dot{x} = Ax \implies \lambda v e^{\lambda t} = A(v e^{\lambda t})$$

1. Simplify the equation. Since $e^{\lambda t}$ is a non-zero scalar, we can cancel it from both sides: 
  
  $$\lambda v = Av$$

This is the fundamental eigenvalue problem. To find the solutions to our differential equation, we need to find the eigenvalues $\lambda$ and corresponding eigenvectors $v$ of the matrix $A$.

Remark/Intuition: The General Solution

Assuming the matrix $A$ has $m$ distinct eigenvalues $\lambda_1, \dots, \lambda_m$ with corresponding eigenvectors $v_1, \dots, v_m$, the eigenvectors form a basis for the state space. Since the system is linear, any linear combination of individual solutions is also a solution. The general solution can therefore be written as a sum:

$$x(t) = \sum_{i=1}^{m} c_i v_i e^{\lambda_i t}$$

The coefficients c_i are determined by the initial condition $x(0) = x_0$:

$$x_0 = \sum_{i=1}^{m} c_i v_i$$

The behavior of the system is a superposition of simple exponential behaviors along each of the eigendirections.

Solution with Complex Eigenvalues

The eigenvalues of a real matrix $A$ can be complex. Since $A$ is real, its complex eigenvalues must come in conjugate pairs: $\lambda = \alpha \pm i\omega$.

Recalling Euler's formula: 

$$e^{i\theta} = \cos(\theta) + i\sin(\theta)$$

We can rewrite the exponential term for a complex eigenvalue $\lambda_i = \alpha_i + i\omega_i$:

$$e^{\lambda_i t} = e^{(\alpha_i + i\omega_i)t} = e^{\alpha_i t} e^{i\omega_i t} = e^{\alpha_i t}(\cos(\omega_i t) + i\sin(\omega_i t))$$

The solution form reveals that the system's behavior has two components:

* An exponential growth or decay component, governed by the real part of the eigenvalue, $e^{\alpha_i t}.$
* An oscillatory component, governed by the imaginary part of the eigenvalue, $\cos(\omega_i t) + i\sin(\omega_i t)$.

The overall behavior is a spiral: the system oscillates while its amplitude grows or shrinks exponentially.

### A Geometric Classification of 2D Linear Equilibria

The origin $x=0$ is always an equilibrium point for the system $\dot{x} = Ax$. We can classify the geometry of the flow around this equilibrium based on the eigenvalues of the matrix $A$. Let's consider a 2D system with eigenvalues $\lambda_1, \lambda_2$.

Definition: Nullclines

Nullclines are curves in the state space where the rate of change of one of the variables is zero.

* The $x_1$-nullcline is the set of points where $\dot{x}_1 = 0$.
* The $x_2$-nullcline is the set of points where $\dot{x}_2 = 0$.

Equilibrium points must lie at the intersection of all nullclines, as this is where all derivatives are zero simultaneously. For linear systems, nullclines are straight lines passing through the origin. They divide the state space into regions with different flow directions.

Case 1: Real Eigenvalues $\omega_1 = \omega_2 = 0$

* Stable Node: Both eigenvalues are real and negative (\lambda_1 < \lambda_2 < 0).
  * Geometry: All trajectories move directly toward the origin. The system decays exponentially along all directions.
  * Time Series: Both x_1(t) and x_2(t) decay exponentially to zero.
  * Stability: Stable. Also called a sink.
* Unstable Node: Both eigenvalues are real and positive (0 < \lambda_1 < \lambda_2).
  * Geometry: All trajectories move directly away from the origin. The system grows exponentially along all directions.
  * Time Series: Both variables diverge exponentially.
  * Stability: Unstable. Also called a source or repeller.
* Saddle Node: Eigenvalues are real and have opposite signs (\lambda_1 < 0 < \lambda_2).
  * Geometry: This is a critical configuration. There is one special direction (the eigenvector v_1 corresponding to \lambda_1 < 0) along which trajectories converge toward the origin. There is another special direction (the eigenvector v_2 corresponding to \lambda_2 > 0) along which trajectories diverge. All other trajectories approach the origin for a time before being swept away along the unstable direction.
  * Stability: Unstable.
  * Manifolds:
    * The line spanned by v_1 is the stable manifold (E^s): The set of all points that flow to the equilibrium.
    * The line spanned by v_2 is the unstable manifold (E^u): The set of all points that flow away from the equilibrium.
    * Manifolds are invariant: any trajectory starting on a manifold stays on that manifold forever.

Case 2: Complex Conjugate Eigenvalues (\lambda = \alpha \pm i\omega, with \omega \neq 0)

* Stable Spiral (or Focus): The real part is negative (\alpha < 0).
  * Geometry: Trajectories spiral inward toward the origin.
  * Time Series: Variables exhibit damped oscillations, converging to zero.
  * Stability: Stable.
* Unstable Spiral (or Focus): The real part is positive (\alpha > 0).
  * Geometry: Trajectories spiral outward, away from the origin.
  * Time Series: Variables exhibit oscillations with growing amplitude.
  * Stability: Unstable.
* Center: The real part is exactly zero (\alpha = 0, \lambda = \pm i\omega).
  * Geometry: This is a very special case. Trajectories are perfect, closed orbits (ellipses or circles) around the equilibrium. The state space is filled with a continuous family of these orbits.
  * Time Series: Variables exhibit perfect, sustained oscillations with constant amplitude.
  * Stability: Neutrally stable. A small perturbation moves the system to a new, nearby orbit; it neither returns nor diverges.

Definition: Hyperbolic Systems

An equilibrium point is called hyperbolic if none of its eigenvalues have a real part equal to zero. This means the system has no centers and no directions of marginal stability. Stable nodes, unstable nodes, saddle nodes, and spirals are all hyperbolic. This is an important property because the local behavior of hyperbolic equilibria is robust to small changes in the system.

Case 3: Eigenvalues with Zero Real Part

* Line or Plane of Equilibria: One eigenvalue is zero, and the others are negative (e.g., \lambda_1 < 0, \lambda_2 = 0).
  * Geometry: There is an entire line (or plane in higher dimensions) of fixed points. This line is the eigenspace corresponding to \lambda_2 = 0. Trajectories from off this line will converge toward it along the stable eigendirections.
  * Stability: Marginally stable.
  * Remark: This configuration, sometimes called a line attractor, is crucial in neuroscience and machine learning for modeling memory. The system can be placed at any point along the line and will stay there, effectively "remembering" that state.


<!-- # Dynamical Systems Theory in Machine Learning: A Foundational Overview

## Executive Summary

Dynamical Systems Theory (DST) provides a rigorous mathematical framework for analyzing systems that evolve over time. This field is fundamentally concerned with the geometric and topological properties of a system's "state space," aiming to understand and predict its long-term behavior. At its core, DST models systems using either continuous-time formulations, such as systems of Ordinary Differential Equations (ODEs) of the form $ \dot{x} = f(x) $, or discrete-time formulations, such as iterative maps like $ x_{t} = f(x_{t-1}) $. The function $ f $, known as the vector field, dictates the flow of the system, guiding trajectories through the state space. A primary goal is to identify and characterize "attractors"—the sets of states, such as stable points or cycles, to which the system converges over time.

### Key Connections to Machine Learning

- **Optimization as a Dynamical System.** The process of training a machine learning model, such as gradient descent, can be viewed as a dynamical system where the model's parameters evolve to minimize a loss function. The minima of this function correspond to stable equilibria or attractors of the system.
- **Recurrent Neural Networks (RNNs).** These architectures are explicitly defined as discrete-time dynamical systems. Their internal "hidden state" evolves at each time step based on the previous state and current input, making their behavior—including phenomena like memory, stability, and convergence—directly analyzable with DST.
- **Scientific Machine Learning.** A frontier in AI involves "dynamical systems reconstruction," where the goal is to reverse-engineer the governing equations $ f(x) $ of a physical, biological, or economic system directly from observational data. Models like Neural ODEs and Physics-Informed Neural Networks are designed for this purpose, aiming to automate a core part of the scientific discovery process.

---

## 1. Fundamentals of Dynamical Systems

### 1.1 What is a Dynamical System?

A dynamical system is a mathematical formalization for any system that evolves along a dimension, which is typically time. The theory provides a set of principles and tools to describe how a system's state changes according to a fixed rule.

**Definition (Dynamical system)**: Let $I \times R \subseteq \mathbb{R} \times \mathbb{R}^n$.
A map $\Phi : I \times R \to R$ is called a dynamical system or flow if:
1. $\Phi(0, x) = x, \quad \forall x \in M,$
2. $\Phi(t + s, x) = \Phi(t, \Phi(s, x)), \quad \forall s, t \in \mathbb{R}, , x \in M,$
3. $\Phi$ is continuous in $(t, x)$.

There is also a discrete formulation of a dynamical system. We will see later that a recurrent neural network is a discrete dynamical system.

Dynamical systems are broadly classified into two categories based on how time is treated:

**Continuous-Time Systems.** The system's state evolves continuously. These are typically described by differential equations.

**Discrete-Time Systems.** The system's state evolves in discrete steps. These are described by iterative functions or maps.

### 1.2 Continuous-Time Systems and Ordinary Differential Equations (ODEs)

The most common representation for continuous-time dynamical systems is a system of first-order ordinary differential equations (ODEs).

> **Definition: Continuous-Time Dynamical System.** A continuous-time dynamical system is often expressed in the form  
> $$
> \dot{x} = f(x)
> $$  
> where the state vector, derivative, and vector field satisfy the following:

- $ x $ is the state vector, a point in a $ p $-dimensional state space, typically $ \mathbb{R}^p $. Its components are $ x = (x_1, x_2, \ldots, x_p) $.
- $ \dot{x} $ represents the temporal derivative $ \frac{dx}{dt} $.
- $ f(x) $ is a function, often called the vector field, that maps the state space to itself ($ f: \mathbb{R}^p \to \mathbb{R}^p $). It determines the velocity of the system's state at every point $ x $.

This compact vector notation represents a system of $ p $ coupled differential equations:

$$
\begin{align*}
\dot{x}_1 &= f_1(x_1, x_2, \ldots, x_p) \\
\dot{x}_2 &= f_2(x_1, x_2, \ldots, x_p) \\
&\vdots \\
\dot{x}_p &= f_p(x_1, x_2, \ldots, x_p)
\end{align*}
$$

While more complex systems can be described by Partial Differential Equations (PDEs), which involve derivatives with respect to multiple variables (e.g., time and space), this course will focus primarily on ODEs.

---

## Exercises for Chapter 1

1. Conceptual: Explain in your own words the relationship between a vector field, a state space, and a trajectory. Why is the uniqueness of trajectories a critical assumption in dynamical systems theory?
2. ML Connection: A simple RNN without external input is defined by the update rule $ h_t = \tanh(W h_{t-1}) $, where $ h_t $ is the state vector and $ W $ is a weight matrix. Identify the components of this equation that correspond to the general form of a discrete-time dynamical system.

---

## 2. Fundamental Properties and Transformations

### 2.1 From Higher-Order to First-Order Systems

Many physical systems are naturally described by higher-order differential equations (involving second or higher derivatives). A powerful technique allows us to convert any such system into an equivalent, larger system of first-order ODEs.

> **Proposition:** Any $ m $-th order one-dimensional ODE can be rewritten as an $ m $-dimensional first-order system of ODEs.

**Example: The Damped Harmonic Oscillator.** Consider the second-order ODE for a damped harmonic oscillator:
$$\frac{d^2x}{dt^2} + a\frac{dx}{dt} + bx = 0.$$
To convert this into a first-order system, we introduce a new set of variables. Let:

- $ x_1 = x $ (position)
- $ x_2 = \dot{x} = \frac{dx}{dt} $ (velocity)

Now, we find the derivatives of these new variables:

- $ \dot{x}_1 = \frac{dx_1}{dt} = \frac{dx}{dt} = x_2 $
- $ \dot{x}_2 = \frac{dx_2}{dt} = \frac{d^2x}{dt^2} = -a\dot{x} - bx = -ax_2 - bx_1 $

This gives us the equivalent two-dimensional first-order system:
$$
\begin{align*}
\dot{x}_1 &= x_2 \\
\dot{x}_2 &= -bx_1 - ax_2
\end{align*}
$$

**Key Takeaway:** The ability to transform higher-order ODEs into first-order systems is crucial. It means that the theoretical tools developed for first-order systems like $ \dot{x} = f(x) $ are broadly applicable, and we can focus our analysis on this canonical form without loss of generality.

### 2.2 Autonomous vs. Non-Autonomous Systems

The vector field of a dynamical system may or may not depend explicitly on time. This distinction leads to two important classes of systems.

- **Autonomous System.** The vector field depends only on the state $ x $. The governing laws of the system do not change over time.  
  $$\dot{x} = f(x)$$
- **Non-Autonomous System.** The vector field depends explicitly on both the state $ x $ and time $ t $. This often occurs when a system is subject to an external influence or forcing.  
  $$\dot{x} = f(x, t)$$
  A common example is a system with a forcing function, such as a pendulum being pushed rhythmically:  
  $$\frac{d^2x}{dt^2} + a\frac{dx}{dt} + bx = F(t).$$
  The function $ F(t) $ represents an external drive. In climate science, the steady increase in atmospheric CO$_2$ acts as a forcing term on the climate system.

Interestingly, any non-autonomous system can be converted into an autonomous one by augmenting the state space. For a system $ \dot{x}_1 = f(x_1, t) $, we can define a new variable $ x_2 = t $. Its derivative is simply $ \dot{x}_2 = 1 $. The new autonomous system in the augmented state space $ (x_1, x_2) $ is:
$$
\begin{align*}
\dot{x}_1 &= f(x_1, x_2) \\
\dot{x}_2 &= 1
\end{align*}
$$
While this mathematical trick can be convenient, it can sometimes obscure the underlying physics by treating time as just another state variable.

---

## Exercises for Chapter 2

1. Transformation: Rewrite the third-order ODE $ \frac{d^3x}{dt^3} + 2\frac{d^2x}{dt^2} - \frac{dx}{dt} + 5x = \cos(t) $ as a system of first-order ODEs. Is the resulting system autonomous or non-autonomous?
2. Application: Convert the non-autonomous system from the previous exercise into an autonomous system using the state-space augmentation technique.

---

## 3. Linear Dynamical Systems

Linear systems are a class of dynamical systems that, despite their relative simplicity, are fundamental to the entire field. Their behavior is fully understood and can be solved analytically. Furthermore, they form the basis for analyzing the local behavior of more complex nonlinear systems.

### 3.1 The Canonical Form: $ \dot{x} = Ax $

A linear system of ODEs is defined by the equation
$$\dot{x} = Ax,$$
where $ x \in \mathbb{R}^m $ is the state vector and $ A $ is a constant $ m \times m $ matrix of coefficients. Each component equation is a linear combination of the state variables:
$$\dot{x}_i = \sum_{j=1}^{m} A_{ij} x_j.$$

### 3.2 The One-Dimensional Case: Stability of an Equilibrium

The simplest case is the one-dimensional linear system
$$\dot{x} = ax,$$
where $ a $ is a scalar. Given an initial condition $ x(0) = x_0 $, the solution can be found by separation of variables to be
$$x(t) = x_0 e^{at}.$$
The point $ x = 0 $ is an equilibrium point (or fixed point), because if the system starts there ($ x_0 = 0 $), then $ \dot{x} = 0 $ and the state remains at zero for all time. The stability of this equilibrium is determined entirely by the sign of $ a $.

#### Geometric Interpretation

**Geometric Interpretation.** By plotting $ \dot{x} $ versus $ x $, we get a line through the origin with slope $ a $.

- **Case 1: $ a > 0 $ (Unstable Equilibrium).**
  - If $ x > 0 $, then $ \dot{x} > 0 $, and $ x $ increases. If $ x < 0 $, then $ \dot{x} < 0 $, and $ x $ decreases.
  - Any small perturbation away from the origin results in the state moving exponentially away. This is also called a repeller or source.
- **Case 2: $ a < 0 $ (Stable Equilibrium).**
  - If $ x > 0 $, then $ \dot{x} < 0 $, and $ x $ decreases towards zero. If $ x < 0 $, then $ \dot{x} > 0 $, and $ x $ increases towards zero.
  - Any perturbation results in the state returning exponentially to the origin. This is a simple point attractor or sink.
- **Case 3: $ a = 0 $ (Neutrally or Marginally Stable).**
  - $ \dot{x} = 0 $ for all $ x $. Every point is a fixed point.
  - If the system is perturbed, it simply stays at its new position.

| Condition | Type of Equilibrium | Behavior of $ x(t) $ |
| --- | --- | --- |
| $ a > 0 $ | Unstable | Exponentially diverges from $ 0 $ |
| $ a < 0 $ | Stable | Exponentially converges to $ 0 $ |
| $ a = 0 $ | Marginally Stable | Remains constant at $ x_0 $ |

### 3.3 The Multi-Dimensional Case: The Eigenvalue Problem

To solve the multi-dimensional system $ \dot{x} = Ax $, we generalize the exponential solution from the 1D case. We make an ansatz (an educated guess) that the solution has the form
$$x(t) = v e^{\lambda t},$$
where $ v $ is a constant vector and $ \lambda $ is a scalar. Taking the derivative with respect to time gives
$$\dot{x}(t) = \lambda v e^{\lambda t}.$$
Substituting both into the original ODE yields
$$\lambda v e^{\lambda t} = A(v e^{\lambda t}).$$
Since $ e^{\lambda t} $ is a non-zero scalar, we can cancel it from both sides, yielding the fundamental eigenvalue problem
$$Av = \lambda v.$$
This shows that the solutions to the linear dynamical system are constructed from the eigenvalues ($ \lambda $) and eigenvectors ($ v $) of the matrix $ A $.

If the matrix $ A $ has $ m $ distinct eigenvalues $ \lambda_1, \ldots, \lambda_m $ with corresponding eigenvectors $ v_1, \ldots, v_m $, the general solution is a linear combination of these fundamental solutions:
$$x(t) = \sum_{i=1}^{m} c_i v_i e^{\lambda_i t}.$$
The coefficients $ c_i $ are constants determined by the system's initial condition, $ x(0) $.

### 3.4 The Role of Complex Eigenvalues

For a real-valued matrix $ A $, its eigenvalues can be complex. Complex eigenvalues always appear in conjugate pairs: $ \lambda = \alpha \pm i\omega $. Using Euler's formula, $ e^{i\theta} = \cos(\theta) + i\sin(\theta) $, we can analyze the solution corresponding to a complex eigenvalue pair:
$$e^{\lambda t} = e^{(\alpha + i\omega)t} = e^{\alpha t} e^{i\omega t} = e^{\alpha t} \bigl(\cos(\omega t) + i\sin(\omega t)\bigr).$$
This reveals that the solution has two components:

1. An exponential term $ e^{\alpha t} $, controlled by the real part ($ \alpha $) of the eigenvalue. This governs the growth ($ \alpha > 0 $) or decay ($ \alpha < 0 $) of the solution's amplitude.
2. An oscillatory term $ \cos(\omega t) + i\sin(\omega t) $, controlled by the imaginary part ($ \omega $). This governs rotation or oscillation in the state space.

**Key Takeaway:** The eigenvalues of the matrix $ A $ completely determine the qualitative behavior of a linear system near its equilibrium. The real part ($ \alpha $) dictates stability (convergence or divergence), while the imaginary part ($ \omega $) dictates rotation (oscillation).

---

## Exercises for Chapter 3

1. 1D Analysis: For the system $ \dot{x} = -2x $ with $ x(0) = 5 $, write the explicit solution $ x(t) $. Is the equilibrium at $ x = 0 $ stable, unstable, or marginally stable? Sketch the trajectory $ x(t) $ versus $ t $.
2. Eigenvalue Problem: Find the eigenvalues and eigenvectors of the matrix $ A = \begin{pmatrix} -3 & 1 \\ 2 & -2 \end{pmatrix} $. Based on the eigenvalues, what do you predict about the long-term behavior of the system $ \dot{x} = Ax $?

---

## 4. A Taxonomy of Linear Equilibria in Two Dimensions

The behavior of a two-dimensional linear system near its single equilibrium point (at the origin, assuming $ A $ is invertible) can be classified exhaustively based on the eigenvalues of the matrix $ A $.

### 4.1 Nullclines and Equilibria

An equilibrium point is a state $ x_{eq} $ where the system's velocity is zero: $ \dot{x} = f(x_{eq}) = 0 $. For a linear system $ \dot{x} = Ax $, this occurs at $ x = 0 $.

Nullclines are curves in the state space where one component of the vector field is zero. For a 2D system, the $ x_1 $-nullcline is the set of points where $ \dot{x}_1 = 0 $, and the $ x_2 $-nullcline is where $ \dot{x}_2 = 0 $. Equilibrium points must lie at the intersection of all nullclines.

### 4.2 Classification Based on Eigenvalues ($ \lambda = \alpha \pm i\omega $)

The following classification provides a complete "zoo" of behaviors for 2D linear systems.

#### Case 1: Real Eigenvalues ($ \omega = 0 $)

| Type | Eigenvalue Condition | Description | Phase Portrait |
| --- | --- | --- | --- |
| Stable Node | $ \lambda_2 < \lambda_1 < 0 $ | Trajectories converge directly to the origin. The system decays exponentially along all directions. Also called a sink. | Trajectories are straight lines or curves moving inward toward the origin. |
| Unstable Node | $ 0 < \lambda_1 < \lambda_2 $ | Trajectories diverge directly away from the origin. The system grows exponentially. Also called a source or repeller. | Trajectories are straight lines or curves moving outward from the origin. |
| Saddle Node | $ \lambda_1 < 0 < \lambda_2 $ | Trajectories converge toward the origin along one direction (the stable manifold) and diverge away along another (the unstable manifold). | Trajectories approach the origin and then are repelled away, except for those exactly on the stable manifold. |

**Stable and Unstable Manifolds.** For a saddle node, the directions of convergence and divergence are determined by the eigenvectors.

- **Stable Manifold ($ E^s $).** The subspace spanned by eigenvectors whose eigenvalues have negative real parts. Trajectories starting on this manifold converge to the equilibrium.
- **Unstable Manifold ($ E^u $).** The subspace spanned by eigenvectors with positive real parts. Trajectories on this manifold diverge. These manifolds are invariant under the flow, meaning a trajectory that starts on a manifold stays on that manifold for all time. Saddle nodes are of fundamental importance and are a key ingredient for chaotic behavior in nonlinear systems.

#### Case 2: Complex Conjugate Eigenvalues ($ \omega \neq 0 $)

| Type | Eigenvalue Condition | Description | Phase Portrait |
| --- | --- | --- | --- |
| Stable Spiral | $ \alpha < 0 $, $ \omega \neq 0 $ | Trajectories spiral inward towards the origin, corresponding to a damped oscillation in time. Also called a stable focus. | Spiraling curves that converge to the origin. |
| Unstable Spiral | $ \alpha > 0 $, $ \omega \neq 0 $ | Trajectories spiral outward away from the origin, corresponding to a growing oscillation. Also called an unstable focus. | Spiraling curves that diverge from the origin. |
| Center | $ \alpha = 0 $, $ \omega \neq 0 $ | Trajectories are closed, concentric orbits (ellipses) around the origin. This corresponds to a sustained, perfect oscillation. The system is neutrally stable. | A family of nested closed loops surrounding the origin. |

#### Case 3: Degenerate Cases (Zero Eigenvalues)

| Type | Eigenvalue Condition | Description | ML/Neuroscience Connection |
| --- | --- | --- | --- |
| Line Attractor | $ \lambda_1 < 0 $, $ \lambda_2 = 0 $ | The system has a line of fixed points. Trajectories converge to this line. Any point on the line is a marginally stable equilibrium. In higher dimensions, this generalizes to plane or hyperplane attractors. | This structure is hypothesized to be a mechanism for memory in neural networks and the brain. Information can be stored by placing the system's state at a specific point on the attractor, where it will remain. |

### 4.3 Hyperbolic Systems

An equilibrium point is called hyperbolic if none of the eigenvalues of the matrix $ A $ have a real part equal to zero ($ \alpha_i \neq 0 $ for all $ i $). This means the system has no centers and no lines or planes of fixed points. The behavior near a hyperbolic equilibrium is structurally stable—small changes to the matrix $ A $ do not qualitatively change the type of equilibrium.

---

## Exercises for Chapter 4

1. Classification: For each of the following matrices $ A $, determine the eigenvalues and classify the type of equilibrium at the origin ($ x = 0 $) for the system $ \dot{x} = Ax $. Sketch a rough phase portrait for each.  
   a) $ A = \begin{pmatrix} -1 & -4 \\ 1 & -1 \end{pmatrix} $  
   b) $ A = \begin{pmatrix} 1 & 1 \\ 4 & 1 \end{pmatrix} $  
   c) $ A = \begin{pmatrix} 0 & 2 \\ -2 & 0 \end{pmatrix} $
2. Manifolds: Consider a system with eigenvalues $ \lambda_1 = -2 $ and $ \lambda_2 = 1 $. Describe the behavior of a trajectory that starts very close to, but not exactly on, the stable manifold.

---

## Nonlinear Systems

### One-dimensional nonlinear ODEs

### Multi-dimensional nonlinear ODEs

$\text{Remark:}$ TODO: put remark between two lotka-volterras

$\text{Example (Lotka-Volterra):}$ ...

$\text{Definition (Homeomorphism):}$ Let $X$ be a topoligical space and $A,B \subseteq X$. A map $h: A \to B$ is called homeomorphism, if

* $h$ is bijective and continuous.
* the inverse $h^{-1}: B \to A$ is continuous.

Two topological spaces $A$ and $B$ are homeomorphic if there is a homeomorphism $h: A \to B$.

> Note: Homeomorphism is also called **a continuous deformation with a continuous inverse**, or **topological isomorphism**, or **bicontinuous function**. Examples: A square and a circle are homeomorphic because one can be continuously deformed into the other. Non-examples: A line segment cannot be continuously deformed into a point, so this is not a homeomorphism. Similarly, a sphere and a torus are not homeomorphic because they have different topological properties (e.g., the number of holes) and cannot be continuously deformed into each other. 

<div class="accordion">
  <details>
    <summary>Topological Space</summary>
    <p>
    A topological space is a set whose elements are called points, along with an additional structure called a topology, which can be defined as a set of neighbourhoods for each point that satisfy some axioms formalizing the concept of closeness. There are several equivalent definitions of a topology, the most commonly used of which is the definition through open sets.
    </p>
    <ul>
      <li>Each head can move at most <code>d</code> squares.</li>
      <li>Starting anywhere inside the current block, after <code>d</code> moves the head can only be in the same block or one of its immediate neighbors (it can’t skip two blocks).</li>
      <li>Moreover, M can read and write anywhere it visits during those <code>d</code> moves. That set of positions is contained within the current block plus at most one block to the left and one to the right.</li>
    </ul>
    <p><strong>Definition via neighbourhoods</strong></p>
    <p><strong>Definition via open sets</strong></p>
    <p><strong>Definition via closed sets</strong></p>
    <p><strong>Examples of topologies</strong></p>
    <p><strong>Examples of topological spaces</strong></p>
    <p>... Every manifold has a natural topology since it is locally Euclidean. Similarly, every simplex and every simplicial complex inherits a natural topology from. ...</p>
    <p><strong>Classification of topological spaces</strong></p>
    <p>Topological spaces can be broadly classified, up to homeomorphism, by their topological properties(?). A topological property is a property of spaces that is invariant under homeomorphisms. To prove that two spaces are not homeomorphic it is sufficient to find a topological property not shared by them. Examples of such properties include connectedness, compactness, and various separation axioms. For algebraic invariants see algebraic topology.</p>

  </details>
</div>


[SINDy](/subpages/dynamical-systems/dynamical-systems-in-machine-learning/sindy/) -->