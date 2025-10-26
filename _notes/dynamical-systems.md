---
title: Dynamical Systems Theory in Machine Learning
date: 2024-11-01
excerpt: Connecting differential equations, stability analysis, and attractor theory to the training dynamics of modern machine learning models.
tags:
  - dynamical-systems
  - machine-learning
  - theory
---

Dynamical Systems Theory in Machine Learning: A Foundational Overview

Executive Summary

Dynamical Systems Theory (DST) provides a rigorous mathematical framework for analyzing systems that evolve over time. This field is fundamentally concerned with the geometric and topological properties of a system's "state space," aiming to understand and predict its long-term behavior. At its core, DST models systems using either continuous-time formulations, such as systems of Ordinary Differential Equations (ODEs) of the form ( \dot{x} = f(x) ), or discrete-time formulations, such as iterative maps like ( x_{t} = f(x_{t-1}) ). The function (f), known as the vector field, dictates the flow of the system, guiding trajectories through the state space. A primary goal is to identify and characterize "attractors"—the sets of states, such as stable points or cycles, to which the system converges over time.

The conceptual tools of DST offer a powerful lens for understanding and designing machine learning models. The connection is not merely an analogy but a direct mathematical equivalence in many cases:

* Optimization as a Dynamical System: The process of training a machine learning model, such as gradient descent, can be viewed as a dynamical system where the model's parameters evolve to minimize a loss function. The minima of this function correspond to stable equilibria or attractors of the system.
* Recurrent Neural Networks (RNNs): These architectures are explicitly defined as discrete-time dynamical systems. Their internal "hidden state" evolves at each time step based on the previous state and current input, making their behavior—including phenomena like memory, stability, and convergence—directly analyzable with DST.
* Scientific Machine Learning: A frontier in AI involves "dynamical systems reconstruction," where the goal is to reverse-engineer the governing equations (( f(x) )) of a physical, biological, or economic system directly from observational data. Models like Neural ODEs and Physics-Informed Neural Networks are designed for this purpose, aiming to automate a core part of the scientific discovery process.

Key concepts from DST, such as the stability of equilibria, the existence of attractors, and the possibility of bifurcations (sudden, qualitative changes in system behavior, analogous to "tipping points"), are crucial for analyzing the performance, robustness, and failure modes of complex machine learning systems. For instance, the analysis of line attractors—continuous sets of stable points—provides a theoretical basis for how neural systems might implement memory by maintaining information as a persistent state of activity. By applying DST, we move from treating machine learning models as black boxes to understanding them as structured systems with predictable and analyzable dynamics.


--------------------------------------------------------------------------------


1. Fundamentals of Dynamical Systems

1.1 What is a Dynamical System?

A dynamical system is a mathematical formalization for any system that evolves along a dimension, which is typically time. The theory provides a set of principles and tools to describe how a system's state changes according to a fixed rule.

Dynamical systems are broadly classified into two categories based on how time is treated:

1. Continuous-Time Systems: The system's state evolves continuously. These are typically described by differential equations.
2. Discrete-Time Systems: The system's state evolves in discrete steps. These are described by iterative functions or maps.

1.2 Continuous-Time Systems and Ordinary Differential Equations (ODEs)

The most common representation for continuous-time dynamical systems is a system of first-order ordinary differential equations (ODEs).

Definition: Continuous-Time Dynamical System A continuous-time dynamical system is often expressed in the form:  \dot{x} = f(x)  where:

* ( x ) is the state vector, a point in a (p)-dimensional state space, typically ( \mathbb{R}^p ). Its components are ( x = (x_1, x_2, \ldots, x_p) ).
* ( \dot{x} ) represents the temporal derivative ( \frac{dx}{dt} ).
* ( f(x) ) is a function, often called the vector field, that maps the state space to itself ((f: \mathbb{R}^p \to \mathbb{R}^p)). It determines the velocity of the system's state at every point (x).

This compact vector notation represents a system of (p) coupled differential equations:  \begin{align*} \dot{x}_1 &= f_1(x_1, x_2, \ldots, x_p) \ \dot{x}_2 &= f_2(x_1, x_2, \ldots, x_p) \ & \vdots \ \dot{x}_p &= f_p(x_1, x_2, \ldots, x_p) \end{align*}  While more complex systems can be described by Partial Differential Equations (PDEs), which involve derivatives with respect to multiple variables (e.g., time and space), this course will focus primarily on ODEs.

1.3 The Geometric Perspective: State Space, Trajectories, and Vector Fields

A core insight of dynamical systems theory is to shift focus from finding explicit solutions (x(t)) to understanding the qualitative geometry of the system's evolution.

* State Space: The state space is the set of all possible states the system can occupy. For a system with (p) variables, this is typically a subset of ( \mathbb{R}^p ). Each point in the state space corresponds to a unique configuration of the system at a moment in time.
* Trajectory: As the system evolves over time, the state vector (x(t)) traces a curve in the state space. This curve is called a trajectory.
* Vector Field: The function ( f(x) ) defines a vector at each point in the state space. This vector indicates the instantaneous direction and speed of the trajectory passing through that point. The collection of all such vectors is the vector field.

A fundamental requirement for a well-defined dynamical system is that trajectories must be unique. At any given point in the state space, the future evolution of the system is strictly determined. A point cannot have multiple possible paths branching from it. If such branching is observed, it implies that the mathematical description of the system is incomplete—some variables are missing.

The long-term behavior of trajectories is of particular interest. DST investigates questions such as: Do trajectories converge to a single point? Do they settle into a repeating loop? These limiting sets are known as attractors.

1.4 Discrete-Time Systems and Maps

In many real-world scenarios, particularly those involving measurements or digital computation, time is measured in discrete intervals.

Definition: Discrete-Time Dynamical System A discrete-time dynamical system is described by an iterative function or map:  x_{t} = f(x_{t-1})  where (x_t) is the state of the system at time step (t), which is determined by its state at the previous time step (t-1).

Connection to Machine Learning: Discrete-time systems are prevalent in machine learning. Recurrent Neural Networks (RNNs) are a canonical example. The hidden state (h_t) of an RNN is updated at each step based on the previous state (h_{t-1}) and the current input, forming a discrete-time dynamical system.

1.5 The Link Between Continuous and Discrete Time

The distinction between continuous and discrete systems is often blurred in practice:

* Empirical Observation: Measurements of any real-world system (e.g., climate, neuroscience) are taken at discrete time points with a certain sampling frequency. Thus, empirically observed systems are always discrete.
* Numerical Simulation: When solving a system of ODEs on a computer, numerical methods (like Euler's method) convert the continuous system into a discrete-time approximation to compute the solution step-by-step.


--------------------------------------------------------------------------------


Exercises for Chapter 1

1. Conceptual: Explain in your own words the relationship between a vector field, a state space, and a trajectory. Why is the uniqueness of trajectories a critical assumption in dynamical systems theory?
2. ML Connection: A simple RNN without external input is defined by the update rule (h_t = \tanh(W h_{t-1})), where (h_t) is the state vector and (W) is a weight matrix. Identify the components of this equation that correspond to the general form of a discrete-time dynamical system.


--------------------------------------------------------------------------------


2. Fundamental Properties and Transformations

2.1 From Higher-Order to First-Order Systems

Many physical systems are naturally described by higher-order differential equations (involving second or higher derivatives). A powerful technique allows us to convert any such system into an equivalent, larger system of first-order ODEs.

Proposition: Any (m)-th order one-dimensional ODE can be rewritten as an (m)-dimensional first-order system of ODEs.

Example: The Damped Harmonic Oscillator Consider the second-order ODE for a damped harmonic oscillator:  \frac{d^2x}{dt^2} + a\frac{dx}{dt} + bx = 0  To convert this into a first-order system, we introduce a new set of variables. Let:

* ( x_1 = x ) (position)
* ( x_2 = \dot{x} = \frac{dx}{dt} ) (velocity)

Now, we find the derivatives of these new variables:

* ( \dot{x}_1 = \frac{dx_1}{dt} = \frac{dx}{dt} = x_2 )
* ( \dot{x}_2 = \frac{dx_2}{dt} = \frac{d^2x}{dt^2} = -a\dot{x} - bx = -ax_2 - bx_1 )

This gives us the equivalent two-dimensional first-order system:  \begin{align*} \dot{x}_1 &= x_2 \ \dot{x}_2 &= -bx_1 - ax_2 \end{align*} 

Key Takeaway: The ability to transform higher-order ODEs into first-order systems is crucial. It means that the theoretical tools developed for first-order systems like ( \dot{x} = f(x) ) are broadly applicable, and we can focus our analysis on this canonical form without loss of generality.

2.2 Autonomous vs. Non-Autonomous Systems

The vector field of a dynamical system may or may not depend explicitly on time. This distinction leads to two important classes of systems.

* Autonomous System: The vector field depends only on the state (x). The governing laws of the system do not change over time.  \dot{x} = f(x) 
* Non-Autonomous System: The vector field depends explicitly on both the state (x) and time (t). This often occurs when a system is subject to an external influence or forcing.  \dot{x} = f(x, t)  A common example is a system with a forcing function, such as a pendulum being pushed rhythmically:  \frac{d^2x}{dt^2} + a\frac{dx}{dt} + bx = F(t)  The function (F(t)) represents an external drive. In climate science, the steady increase in atmospheric CO2 acts as a forcing term on the climate system.

Interestingly, any non-autonomous system can be converted into an autonomous one by augmenting the state space. For a system ( \dot{x}_1 = f(x_1, t) ), we can define a new variable ( x_2 = t ). Its derivative is simply ( \dot{x}_2 = 1 ). The new autonomous system in the augmented state space ( (x_1, x_2) ) is:  \begin{align*} \dot{x}_1 &= f(x_1, x_2) \ \dot{x}_2 &= 1 \end{align*}  While this mathematical trick can be convenient, it can sometimes obscure the underlying physics by treating time as just another state variable.


--------------------------------------------------------------------------------


Exercises for Chapter 2

1. Transformation: Rewrite the third-order ODE ( \frac{d^3x}{dt^3} + 2\frac{d^2x}{dt^2} - \frac{dx}{dt} + 5x = \cos(t) ) as a system of first-order ODEs. Is the resulting system autonomous or non-autonomous?
2. Application: Convert the non-autonomous system from the previous exercise into an autonomous system using the state-space augmentation technique.


--------------------------------------------------------------------------------


3. Linear Dynamical Systems

Linear systems are a class of dynamical systems that, despite their relative simplicity, are fundamental to the entire field. Their behavior is fully understood and can be solved analytically. Furthermore, they form the basis for analyzing the local behavior of more complex nonlinear systems.

3.1 The Canonical Form: ( \dot{x} = Ax )

A linear system of ODEs is defined by the equation:  \dot{x} = Ax  where ( x \in \mathbb{R}^m ) is the state vector and ( A ) is a constant ( m \times m ) matrix of coefficients. Each component equation is a linear combination of the state variables:  \dot{x}_i = \sum_{j=1}^{m} A_{ij} x_j 

3.2 The One-Dimensional Case: Stability of an Equilibrium

The simplest case is the one-dimensional linear system:  \dot{x} = ax  where (a) is a scalar. Given an initial condition ( x(0) = x_0 ), the solution can be found by separation of variables to be:  x(t) = x_0 e^{at}  The point (x=0) is an equilibrium point (or fixed point), because if the system starts there ((x_0=0)), then ( \dot{x} = 0 ) and the state remains at zero for all time. The stability of this equilibrium is determined entirely by the sign of (a).

Geometric Interpretation: By plotting ( \dot{x} ) versus ( x ), we get a line through the origin with slope (a).

* Case 1: ( a > 0 ) (Unstable Equilibrium)
  * If (x > 0), then ( \dot{x} > 0 ), and (x) increases. If (x < 0), then ( \dot{x} < 0 ), and (x) decreases.
  * Any small perturbation away from the origin results in the state moving exponentially away. This is also called a repeller or source.
* Case 2: ( a < 0 ) (Stable Equilibrium)
  * If (x > 0), then ( \dot{x} < 0 ), and (x) decreases towards zero. If (x < 0), then ( \dot{x} > 0 ), and (x) increases towards zero.
  * Any perturbation results in the state returning exponentially to the origin. This is a simple point attractor or sink.
* Case 3: ( a = 0 ) (Neutrally or Marginally Stable)
  * ( \dot{x} = 0 ) for all (x). Every point is a fixed point.
  * If the system is perturbed, it simply stays at its new position.

Condition	Type of Equilibrium	Behavior of (x(t))
(a > 0)	Unstable	Exponentially diverges from (0)
(a < 0)	Stable	Exponentially converges to (0)
(a = 0)	Marginally Stable	Remains constant at (x_0)

3.3 The Multi-Dimensional Case: The Eigenvalue Problem

To solve the multi-dimensional system ( \dot{x} = Ax ), we generalize the exponential solution from the 1D case. We make an ansatz (an educated guess) that the solution has the form:  x(t) = v e^{\lambda t}  where (v) is a constant vector and (\lambda) is a scalar. Taking the derivative with respect to time gives:  \dot{x}(t) = \lambda v e^{\lambda t}  Substituting both into the original ODE:  \lambda v e^{\lambda t} = A(v e^{\lambda t})  Since (e^{\lambda t}) is a non-zero scalar, we can cancel it from both sides, yielding the fundamental eigenvalue problem:  Av = \lambda v  This shows that the solutions to the linear dynamical system are constructed from the eigenvalues ((\lambda)) and eigenvectors ((v)) of the matrix (A).

If the matrix (A) has (m) distinct eigenvalues (\lambda_1, \ldots, \lambda_m) with corresponding eigenvectors (v_1, \ldots, v_m), the general solution is a linear combination of these fundamental solutions:  x(t) = \sum_{i=1}^{m} c_i v_i e^{\lambda_i t}  The coefficients ( c_i ) are constants determined by the system's initial condition, ( x(0) ).

3.4 The Role of Complex Eigenvalues

For a real-valued matrix (A), its eigenvalues can be complex. Complex eigenvalues always appear in conjugate pairs: ( \lambda = \alpha \pm i\omega ). Using Euler's formula, ( e^{i\theta} = \cos(\theta) + i\sin(\theta) ), we can analyze the solution corresponding to a complex eigenvalue pair:  e^{\lambda t} = e^{(\alpha + i\omega)t} = e^{\alpha t} e^{i\omega t} = e^{\alpha t} (\cos(\omega t) + i\sin(\omega t))  This reveals that the solution has two components:

1. An exponential term (e^{\alpha t}), controlled by the real part ((\alpha)) of the eigenvalue. This governs the growth ((\alpha > 0)) or decay ((\alpha < 0)) of the solution's amplitude.
2. An oscillatory term ((\cos(\omega t) + i\sin(\omega t))), controlled by the imaginary part ((\omega)). This governs rotation or oscillation in the state space.

Key Takeaway: The eigenvalues of the matrix (A) completely determine the qualitative behavior of a linear system near its equilibrium. The real part ((\alpha)) dictates stability (convergence or divergence), while the imaginary part ((\omega)) dictates rotation (oscillation).


--------------------------------------------------------------------------------


Exercises for Chapter 3

1. 1D Analysis: For the system ( \dot{x} = -2x ) with (x(0)=5), write the explicit solution (x(t)). Is the equilibrium at (x=0) stable, unstable, or marginally stable? Sketch the trajectory (x(t)) versus (t).
2. Eigenvalue Problem: Find the eigenvalues and eigenvectors of the matrix ( A = \begin{pmatrix} -3 & 1 \ 2 & -2 \end{pmatrix} ). Based on the eigenvalues, what do you predict about the long-term behavior of the system ( \dot{x} = Ax )?


--------------------------------------------------------------------------------


4. A Taxonomy of Linear Equilibria in Two Dimensions

The behavior of a two-dimensional linear system near its single equilibrium point (at the origin, assuming (A) is invertible) can be classified exhaustively based on the eigenvalues of the matrix (A).

4.1 Nullclines and Equilibria

An equilibrium point is a state (x_{eq}) where the system's velocity is zero: ( \dot{x} = f(x_{eq}) = 0 ). For a linear system ( \dot{x}=Ax ), this occurs at (x=0).

Nullclines are curves in the state space where one component of the vector field is zero. For a 2D system, the (x_1)-nullcline is the set of points where ( \dot{x}_1 = 0 ), and the (x_2)-nullcline is where ( \dot{x}_2 = 0 ). Equilibrium points must lie at the intersection of all nullclines.

4.2 Classification Based on Eigenvalues (( \lambda = \alpha \pm i\omega ))

The following classification provides a complete "zoo" of behaviors for 2D linear systems.

Case 1: Real Eigenvalues (( \omega = 0 ))

Type	Eigenvalue Condition	Description	Phase Portrait
Stable Node	( \lambda_2 < \lambda_1 < 0 )	Trajectories converge directly to the origin. The system decays exponentially along all directions. Also called a sink.	Trajectories are straight lines or curves moving inward toward the origin.
Unstable Node	( 0 < \lambda_1 < \lambda_2 )	Trajectories diverge directly away from the origin. The system grows exponentially. Also called a source or repeller.	Trajectories are straight lines or curves moving outward from the origin.
Saddle Node	( \lambda_1 < 0 < \lambda_2 )	Trajectories converge toward the origin along one direction (the stable manifold) and diverge away along another (the unstable manifold).	Trajectories approach the origin and then are repelled away, except for those exactly on the stable manifold.

Stable and Unstable Manifolds: For a saddle node, the directions of convergence and divergence are determined by the eigenvectors.

* Stable Manifold ((E^s)): The subspace spanned by eigenvectors whose eigenvalues have negative real parts. Trajectories starting on this manifold converge to the equilibrium.
* Unstable Manifold ((E^u)): The subspace spanned by eigenvectors with positive real parts. Trajectories on this manifold diverge. These manifolds are invariant under the flow, meaning a trajectory that starts on a manifold stays on that manifold for all time. Saddle nodes are of fundamental importance and are a key ingredient for chaotic behavior in nonlinear systems.

Case 2: Complex Conjugate Eigenvalues (( \omega \neq 0 ))

Type	Eigenvalue Condition	Description	Phase Portrait
Stable Spiral	( \alpha < 0 ), ( \omega \neq 0 )	Trajectories spiral inward towards the origin, corresponding to a damped oscillation in time. Also called a stable focus.	Spiraling curves that converge to the origin.
Unstable Spiral	( \alpha > 0 ), ( \omega \neq 0 )	Trajectories spiral outward away from the origin, corresponding to a growing oscillation. Also called an unstable focus.	Spiraling curves that diverge from the origin.
Center	( \alpha = 0 ), ( \omega \neq 0 )	Trajectories are closed, concentric orbits (ellipses) around the origin. This corresponds to a sustained, perfect oscillation. The system is neutrally stable.	A family of nested closed loops surrounding the origin.

Case 3: Degenerate Cases (Zero Eigenvalues)

Type	Eigenvalue Condition	Description	ML/Neuroscience Connection
Line Attractor	( \lambda_1 < 0, \lambda_2 = 0 )	The system has a line of fixed points. Trajectories converge to this line. Any point on the line is a marginally stable equilibrium. In higher dimensions, this generalizes to plane or hyperplane attractors.	This structure is hypothesized to be a mechanism for memory in neural networks and the brain. Information can be stored by placing the system's state at a specific point on the attractor, where it will remain.

4.3 Hyperbolic Systems

An equilibrium point is called hyperbolic if none of the eigenvalues of the matrix (A) have a real part equal to zero ((\alpha_i \neq 0) for all (i)). This means the system has no centers and no lines/planes of fixed points. The behavior near a hyperbolic equilibrium is structurally stable—small changes to the matrix (A) do not qualitatively change the type of equilibrium.


--------------------------------------------------------------------------------


Exercises for Chapter 4

1. Classification: For each of the following matrices (A), determine the eigenvalues and classify the type of equilibrium at the origin ((x=0)) for the system ( \dot{x}=Ax ). Sketch a rough phase portrait for each. a) ( A = \begin{pmatrix} -1 & -4 \ 1 & -1 \end{pmatrix} ) b) ( A = \begin{pmatrix} 1 & 1 \ 4 & 1 \end{pmatrix} ) c) ( A = \begin{pmatrix} 0 & 2 \ -2 & 0 \end{pmatrix} )
2. Manifolds: Consider a system with eigenvalues ( \lambda_1 = -2 ) and ( \lambda_2 = 1 ). Describe the behavior of a trajectory that starts very close to, but not exactly on, the stable manifold.
