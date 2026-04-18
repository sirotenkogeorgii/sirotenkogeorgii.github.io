## Lecture 2

### Decoupling Technique

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Decoupling in Dynamical Systems)</span></p>

In dynamical systems, **decoupling** means rewriting a coupled system so that its variables evolve **independently** or in **smaller independent groups**. The point is to turn a hard multivariable problem into several easier one-variable problems. Imagine a system of two variables, $x_1$ and $x_2$. If the rate of change of $x_1$ depends only on $x_1$, and the rate of change of $x_2$ depends only on $x_2$, the system is already decoupled. You can solve them completely independently. But in the real world, things are "coupled." The rate of change of $x_1$ depends on what $x_2$ is doing, and vice versa.

Suppose you start with a coupled linear system

$$\dot{x} = Ax$$

where $x \in \mathbb{R}^n$ and the components of $x$ influence each other through the matrix $A$.

If you can find a change of variables $x = Py$ such that

$$\dot{y} = P^{-1}AP, y$$

and $P^{-1}AP$ is diagonal, then the system becomes

$$\dot{y}_i = \lambda_i y_i$$

Now each equation depends only on one variable $y_i$, so the system is decoupled.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Main idea)</span></p>

A system is **coupled** if one equation contains several state variables, like

$$
\dot{x}_1 = 2x_1 + x_2, \qquad
\dot{x}_2 = x_1 + 2x_2
$$

Here $x_1$ and $x_2$ affect each other.

A **decoupled** version would look like

$$\dot{y}_1 = 3y_1, \qquad \dot{y}_2 = y_2$$

Now each variable evolves on its own.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Decoupling in Linear Dynamical Systems)</span></p>

For linear systems, the usual technique is:

1. Write the system in matrix form $\dot{x}=Ax$.
2. Find eigenvalues and eigenvectors of $A$.
3. Use eigenvectors to build a transformation matrix $P$.
4. Change coordinates with $x=Py$.

If $A$ is diagonalizable, this gives independent scalar ODEs.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Main idea)</span></p>

Take

$$\dot{x}_1 = x_1 + x_2,\qquad \dot{x}_2 = x_1 + x_2$$

Define new variables

$$y_1 = x_1 + x_2,\qquad y_2 = x_1 - x_2$$

Then

$$\dot{y}_1 = 2y_1,\qquad \dot{y}_2 = 0$$

So the original coupled system becomes decoupled in the new coordinates.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why this is useful)</span></p>

Decoupling helps with:

* solving the system explicitly,
* understanding stability,
* identifying normal modes,
* separating fast and slow behavior,
* simplifying numerical simulation.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Important note)</span></p>

Decoupling is easiest for **linear systems**. For **nonlinear systems**, exact decoupling is usually harder, but one often tries to:

* decouple **locally** near an equilibrium by linearization,
* use **normal forms**,
* exploit symmetry or conserved quantities,
* separate variables into slow/fast subsystems.

So, in one sentence:

**Decoupling is a coordinate transformation that turns interacting state equations into independent ones, making the dynamics easier to analyze.**

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Physical Example)</span></p>

To really understand decoupling, it helps to step away from the abstract matrix $A$ and look at a physical example. 

Imagine two identical pendulums hanging side-by-side, connected by a spring.
* If you pull just the left pendulum back and let it go, the resulting motion is a chaotic, messy dance. The left pendulum swings, the spring pulls the right pendulum, the right pendulum swings back and pushes the left one. The energy sloshes back and forth between them.
* The position of the left pendulum ($x_1$) and the right pendulum ($x_2$) are deeply **coupled**.

However, if you observe carefully, you will notice there are two specific ways you can start the pendulums where the motion is incredibly simple:
1. **The "Together" Pattern:** If you pull both pendulums back by the exact same amount and let go, they will swing back and forth in perfect unison. The spring between them never stretches or compresses; it just goes along for the ride.
2. **The "Opposite" Pattern:** If you pull them apart by the exact same amount and let go, they will swing exactly opposite to each other, crashing inward and flying outward symmetrically.
  
These hidden, pure patterns are called **Normal Modes**. In these specific patterns, the complex system acts like a simple, single entity.

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/Two-pendulums-connected-by-a-spring.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Two pendulums connected by a spring.</figcaption>
</figure>

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Mathematical Translation Of Example above)</span></p>

* **The standard coordinates ($x_1, x_2$):** These represent the physical positions of the left and right pendulums. Tracking these is messy and coupled ($A$).
* **The eigenvectors ($P$):** These are the mathematical representation of our "Normal Modes." One eigenvector represents the "Together" pattern, and the other represents the "Opposite" pattern.
* **The eigenvalues ($D$):** These dictate the frequencies or decay rates of those specific patterns. The "Together" pattern will swing at one specific frequency ($\lambda_1$), and the "Opposite" pattern will swing at a different frequency ($\lambda_2$) because it has to fight the tension of the spring.
* **The decoupled coordinates ($y_1, y_2$):** These variables no longer ask "Where is the left pendulum?" Instead, $y_1$ asks "How much of the Together pattern is happening right now?" and $y_2$ asks "How much of the Opposite pattern is happening right now?"

**Key Moment:**

Here is the brilliant core of decoupling: **Any messy, chaotic state the pendulums can possibly be in is just a combination of those two simple patterns.**

By using the substitution $\mathbf{x} = P\mathbf{y}$, we stopped trying to track the left and right pendulums individually. Instead, we translated the entire problem into the language of the system's natural patterns. Because the "Together" pattern and the "Opposite" pattern do not interfere with each other, their mathematical equations ($\dot{y}_1 = \lambda_1 y_1$ and $\dot{y}_2 = \lambda_2 y_2$) are completely separated (decoupled).

We solve the simple patterns, and then translate them back into physical positions at the very end.

</div>

### General Solutions for Linear Systems

[Derivation of General Solutions for Linear Systems](/subpages/dynamical-systems/dynamical-systems-in-machine-learning/long-proofs/)

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

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Canonical Forms for 2x2 Systems)</span></p>

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

</div>

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
* But the eigenspace is only **1-dimensional** (**geometric multiplicity 1**): there is only **one independent eigenvector direction**. You *don’t* get a basis of eigenvectors in $\mathbb R^2$. That’s what they mean by "degenerate."

**Geometry/dynamics: it’s “scaling + shear”**

For the linear system $\dot x = Jx$, the matrix exponential is

$$e^{Jt}=e^{at}\begin{pmatrix}1&t\\0&1\end{pmatrix}.$$

So solutions are

$$x_2(t)=C_2 e^{at}, \qquad x_1(t)=e^{at}(C_1 + C_2 t).$$

**Geometric interpretation:**

* The factor $e^{at}$ is uniform expansion/decay (depending on sign of $a$).
* The $\begin{pmatrix} 1 & t \\\ 0 & 1 \end{pmatrix}$ part is a **shear**: it pushes points sideways in the $x_1$ direction at a rate proportional to $t$ and to their $x_2$-component.

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

The **matrix exponential** $e^{At}$ is defined in a manner analogous to the Taylor series expansion of the scalar exponential function:

$$e^{At} = \sum_{k=0}^{\infty} \frac{(At)^k}{k!} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \dots$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name"></span></p>

It is straightforward to see why **this form constitutes a solution to the ODE**. If we take the temporal derivative of the solution $\mathbf{x}(t) = e^{At} \mathbf{x}_0$, we differentiate the series term-by-term:

$$\frac{d}{dt} \mathbf{x}(t) = \frac{d}{dt} \left( \sum_{k=0}^{\infty} \frac{A^k t^k}{k!} \right) \mathbf{x}_0$$

$$= \left( \sum_{k=1}^{\infty} \frac{A^k k t^{k-1}}{k!} \right) \mathbf{x}_0 = A \left( \sum_{k=1}^{\infty} \frac{A^{k-1} t^{k-1}}{(k-1)!} \right) \mathbf{x}_0$$

By re-indexing the sum (let $j=k-1$), we recover the original series:

$$= A \left( \sum_{j=0}^{\infty} \frac{(At)^j}{j!} \right) \mathbf{x}_0 = A e^{At} \mathbf{x}_0 = A \mathbf{x}(t)$$

This confirms that $\dot{\mathbf{x}} = A\mathbf{x}(t)$, satisfying the differential equation. The full proof of the theorem also requires showing this solution is unique, which can be done by assuming two distinct solutions and demonstrating they must be identical.

</div>

#### Equivalence of Solutions for Diagonalizable Systems

While the matrix exponential provides a powerful general solution, it is important to verify that it is consistent with the eigenvector-based solution we derived earlier for the case where $A$ is diagonalizable (i.e., has distinct eigenvalues).

<div class="math-callout math-callout--info" markdown="1">
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

An **affine or inhomogeneous linear system of ordinary differential equations** is defined by:  

$$\dot{x} = Ax + b$$

where $x, b \in \mathbb{R}^m$ and $A$ is an $m \times m$ matrix.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Shifting the Equilibrium)</span></p>

The addition of the constant vector $b$ does not alter the fundamental dynamics of the system, which are dictated by the matrix $A$. Instead, its effect is to move the system's equilibrium point. The vector field remains unchanged relative to this new equilibrium.

To understand this, we first locate the new equilibrium, or fixed point, by finding the point $x^{\ast}$ where the flow is zero ($\dot{x}=0$).

Assuming the matrix $A$ is invertible, we solve for the fixed point: 

$$0 = Ax^{\ast} + b \implies Ax^{\ast} = -b \implies x^{\ast} = -A^{-1}b$$ 

This point $x^{\ast}$ is our new equilibrium, shifted from the origin.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Variation of Parameters)</span></p>

The addition of the constant vector $b$ does not alter the fundamental dynamics of the system, which are dictated by the matrix $A$. Instead, its effect is to move the system's equilibrium point. **The vector field remains unchanged relative to this new equilibrium.**

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We can formally prove that the dynamics remain the same by defining a new variable $y$ that represents the state relative to the fixed point $x^{\ast}$.

1. **Define a new variable:** Let $y = x - x^{\ast}$. This is equivalent to $x = y + x^{\ast}$.
2. **Consider the dynamics of the new variable:** The temporal derivative of $y$ is $\dot{y} = \dot{x}$, since $x^{\ast}$ is a constant and its derivative is zero.
3. **Substitute into the original equation:** We can now express $\dot{y}$ in terms of $y$
   
   $$\dot{y} = \dot{x} = Ax + b$$
   
   Substitute $x = y + x^{\ast}$: 
   
   $$\dot{y} = A(y + x^{\ast}) + b = Ay + Ax^{\ast} + b$$  
   
   Now, substitute the expression for the fixed point, $x^{\ast} = -A^{-1}b$:

4. **Result:** The dynamics for the new variable are:
   
   $$\dot{y} = Ay$$
   
   This is precisely the homogeneous linear system we have already analyzed. The dynamics (stability, rotation, etc.) around the fixed point $x^{\ast}$ are identical to the dynamics of the homogeneous system around the origin. To find the full solution for $x(t)$, one solves for $y(t)$ and then recovers 
   
   $$x(t) = y(t) + x^{\ast}$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection to the Hartman-Grobman Theorem)</span></p>

The coordinate shift $y = x - x^\ast$ used here is the same conceptual move that underlies the Hartman-Grobman theorem for nonlinear systems. In both cases, one translates the equilibrium to the origin and studies the dynamics of the deviation from the equilibrium's point of view (center is equilibrium). The difference is the strength of the result:

- **Linear case (here):** The system is already linear (affine), so the reduction $\dot{y} = Ay$ to linear from affine is **exact and global** — no approximation is involved, and it holds everywhere in state space.
- **Nonlinear case (Hartman-Grobman):** For $\dot{x} = f(x)$ near a hyperbolic equilibrium, the linearization $\dot{y} \approx Df(x^\ast)\, y$ is only a **topological conjugacy**, valid **locally** near $x^\ast$. The qualitative picture (stability type, saddle/node/spiral) is preserved, but the correspondence is through a continuous homeomorphism that is not necessarily smooth, and it breaks down far from the equilibrium.

In this sense, the Variation of Parameters result can be viewed as the exact, global special case of the Hartman-Grobman idea: when there are no higher-order terms to discard, the linearization trick works perfectly everywhere.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Non-Invertible Case)</span></p>

If the matrix $A$ is not invertible, it possesses at least one zero eigenvalue. In this scenario, a unique fixed point does not exist. This corresponds to the case of a center or, more generally, a line attractor (or plane/hyperplane attractor in higher dimensions). The system has a continuous manifold of equilibrium points along the direction of the eigenvector(s) associated with the zero eigenvalue(s).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Degenerate does not always mean manifold attractor)</span></p>

A zero eigenvalue can happen for different reasons.

1. **Case A: line/manifold of equilibria**

   Example:

   $$\dot{x}=0, \qquad \dot{y}=-y.$$

   Every point $(x,0)$ is an equilibrium, so the $x$-axis is a manifold of equilibria. Here the zero eigenvalue corresponds to genuine neutral motion along that equilibrium manifold.

1. **Case B: isolated equilibrium with zero eigenvalue**

   Example:

   $$\dot{x}=x^2, \qquad \dot{y}=-y.$$

   At $(0,0)$, one eigenvalue is zero, but there is no line of equilibria. The only equilibrium is $(0,0)$, and the motion in the $x$-direction is determined by the nonlinear term $x^2$.

   So:

   > Degeneracy does not automatically mean line attractor or manifold attractor.

</div>


#### Non-autonomous Systems with a Forcing Function

We now consider systems where the dynamics are explicitly influenced by time, driven by an external "forcing function."

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Non-autonomous System with Forcing Function)</span></p>

A **non-autonomous linear system with a forcing function** $f(t)$ is defined as a system that explicitly depends on time. For simplicity, we will analyze the scalar case:

$$\dot{x} = ax + f(t)$$ 

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Variation of Parameters)</span></p>

To solve this type of equation, we employ a powerful technique known as variation of parameters. The logic is as follows: we know the solution to the homogeneous part of the equation ($\dot{x} = ax$) is $x(t) = C e^{at}$, where $C$ is a constant. We now "promote" this constant to a time-dependent function, $k(t)$, and propose an ansatz (an educated guess) for the full solution that has a similar form. This allows the solution to adapt to the time-varying influence of $f(t)$.

</div>

<div class="math-callout math-callout--info" markdown="1">
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
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Discrete-time autonomous dynamical system)</span></p>

A **discrete-time autonomous dynamical system** is defined by a recursive prescription:  

$$\mathbf{x}_t = f(\mathbf{x}_{t-1})$$

We will focus on the affine linear map, which is the discrete-time analogue of the inhomogeneous systems discussed earlier:  

$$\mathbf{x}_t = A \mathbf{x}_{t-1} + \mathbf{b}$$  

Such a map generates a sequence of vector-valued numbers $\lbrace\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T\rbrace$ starting from an initial condition $\mathbf{x}_1$. A primary goal is to understand the limiting behavior of this sequence as $t \to \infty$.

</div>

#### Iterative Solution and Limiting Behavior (Scalar Case)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Iterative Solution For Scalar Discrete-time Autonomous DS)</span></p>

The scalar discrete-time autonomous dynamical system is defined as

$$x_t = ax_{t-1} + b$$

For the given time step $T$ the (iterative) solution is

$$x_T = a^{T-1}x_1 + b \sum_{i=0}^{T-2} a^i$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Recursive Expansion:** We can expand the expression recursively to understand its structure over time:

* At $t=2: x_2 = ax_1 + b$
* At $t=3: x_3 = a(x_2) + b = a(ax_1 + b) + b = a^2x_1 + ab + b$ 
* At $t=4: x_4 = a(x_3) + b = a(a^2x_1 + ab + b) + b = a^3x_1 + a^2b + ab + b$

**General Form:** Observing the pattern, the state at a general time step $T$ is:
  
  $$x_T = a^{T-1}x_1 + b(a^{T-2} + a^{T-3} + \dots + a^1 + a^0)$$
  
  The second term is a finite geometric series. We can write this more compactly as:
  
  $$x_T = a^{T-1}x_1 + b \sum_{i=0}^{T-2} a^i$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(A Different View)</span></p>

The fixed point or limiting solution of $x_t$ as $t \to \infty$ is

$$\lim_{t \to \infty} x_t = b \left( \frac{1}{1-a} \right)$$

converges for $\lvert a\rvert <1$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Limiting Behavior:** We are interested in what happens as $t \to \infty$. The convergence of this sequence depends entirely on the value of $a$.

* **Condition for Convergence:** The sequence converges only if the absolute value of a is less than one, i.e., $\lvert a\rvert < 1$.
* **Analysis of Terms:**
  * **Initial Condition Term:** For $\lvert a\rvert<1$, the term $a^{T-1}x_1$ decays to zero as $T \to \infty$. This means the system forgets about the initial condition exponentially fast.
  * **Geometric Series Term:** For $\lvert a\rvert <1$, the infinite geometric series converges to a fixed value:
* **The Limit:** Combining these results, the limit of $x_t$ as $t \to \infty$ is
  
  $$\lim_{t \to \infty} x_t = b \left( \frac{1}{1-a} \right)$$

</details>
</div>

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

where $a$ and $b$ are scalar constants. The parameter $a$ represents the slope, and $b$ is the intercept or offset.

</div>

#### Geometric Interpretation: The Cobweb Plot

To gain a deeper intuition for the system's evolution over time, we can visualize this recursive process graphically. We plot the function $x_{t+1} = ax_t + b$ against the bisectrix, which is the line $x_{t+1} = x_t$. The intersection of these two lines holds special significance, as we will see shortly.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Cobweb Plot)</span></p>

The Cobweb Plot is a powerful geometric technique for visualizing the trajectory of a discrete-time system. It provides an immediate feel for whether the system converges to a specific value, diverges to infinity, or exhibits other behaviors. The procedure is as follows:

1. **Initialization:** Start with an initial condition, $x_0$, on the horizontal axis.
2. **Evaluation:** Move vertically from $x_0$ to the function line $x_{t+1} = ax_t + b$. The height of this point gives the next state, $x_1$.
3. **Iteration:** To use $x_1$ as the next input, move horizontally from the point on the function line to the bisectrix ($x_{t+1} = x_t$). This transfers the output value $x_1$ to the horizontal axis, preparing it for the next iteration.
4. **Repeat:** From this new point on the bisectrix, move vertically again to the function line to find $x_2$, then horizontally to the bisectrix, and so on.

The path traced by these movements often resembles a spider's web, spiraling inwards or outwards, which gives the method its name.

</div>

#### Fixed Points in One Dimension

A central concept in dynamical systems is the notion of a fixed point, which is analogous to an equilibrium in continuous-time systems described by differential equations.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fixed Point of Discrete-Time System)</span></p>

A point $x^{\ast}$ is a **fixed point of a discrete-time system** $x_{t+1} = f(x_t)$ if it remains unchanged by the map. That is, it satisfies the condition:

$$x^{\ast} = f(x^{\ast})$$

If the system is initialized at a fixed point, it will remain there for all future time steps. There is no movement at this point.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric View of Fixed Points)</span></p>

Geometrically, a fixed point is simply the intersection of the function graph $y = f(x)$ and the bisectrix $y=x$. At this specific point, the input to the function is exactly equal to its output, satisfying the definition $x^{\ast} = f(x^{\ast})$.

</div>

We can find the fixed point not only graphically but also by solving the defining equation algebraically.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Fixed Point of 1D Discrete-Time LDS)</span></p>

Fixed point of 1D discrete-time LDS $x_t = ax_{t-1} + b$ is 

$$x^{\ast} = \frac{b}{1-a}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

To find the fixed point $x^{\ast}$, we set the output equal to the input according to the definition:

$$x^{\ast} = ax^{\ast} + b$$

We then solve for $x^{\ast}$:

$$x^{\ast} - ax^{\ast} = b$$

$$(1-a)x^{\ast} = b$$

Assuming $a \neq 1$, we can divide by $(1-a)$ to find the unique fixed point:

$$x^{\ast} = \frac{b}{1-a}$$

This algebraic solution precisely matches the limiting solution for convergent systems and identifies the point of intersection on the cobweb plot.

</details>
</div>

#### Stability Analysis of Fixed Points

A fixed point can be stable, unstable, or neutrally stable, depending on the behavior of nearby trajectories. This stability is determined entirely by the slope parameter, $a$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stable and Unstable Fixed Point of 1D Discrete-Time Linear Map)</span></p>

* A fixed point $x^{\ast}$ is **stable** if *trajectories starting near* $x^{\ast}$ converge towards it as $t \to \infty$. In the linear 1D case, this occurs when $\lvert a\rvert < 1$.
* A fixed point $x^{\ast}$ is **unstable** if *trajectories starting near* $x^{\ast}$ move away from it as $t \to \infty$. In the linear 1D case, this occurs when $\lvert a\rvert > 1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

On the Cobweb Plot, a slope with $\lvert a\rvert < 1$ is less steep than the bisectrix. This geometric configuration ensures that each step of the cobweb construction brings the state closer to the intersection point, causing the "web" to spiral inwards towards the fixed point.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

When $\lvert a\rvert > 1$, the function line is steeper than the bisectrix. The Cobweb Plot immediately reveals that each iteration throws the state further away from the intersection point, causing the "web" to spiral outwards. The system still possesses a fixed point, but any infinitesimal perturbation from it will lead to divergence.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Neutrally Stable Point of 1D Discrete-Time Linear Map)</span></p>

A point or system is **neutrally stable** if *nearby trajectories* neither converge towards nor diverge away from it, but instead remain in a bounded orbit. This occurs when $\lvert a\rvert = 1$.

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

Consider the system 

$$x_{t+1} = -x_t + 1$$

Let the initial state be $x_1 = 2$.

* $x_2 = -x_1 + 1 = -(2) + 1 = -1$
* $x_3 = -x_2 + 1 = -(-1) + 1 = 2$
* $x_4 = -x_3 + 1 = -(2) + 1 = -1$ 
  
The system enters a stable 2-cycle, oscillating between the values $2$ and $-1$. The amplitude of this oscillation depends on the initial value, but the oscillatory nature is preserved. This behavior is analogous to the center case in systems of linear differential equations, where solutions form a continuous set of stable orbits.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Analogy to Continuous Systems)</span></p>

The spectrum of solutions observed in discrete-time linear systems—stable and unstable fixed points, and mutually stable oscillations—is precisely the same class of solutions found in continuous-time linear systems of ordinary differential equations. This parallel provides a powerful conceptual bridge between the two domains.

</div>

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

### Higher-Dimensional Discrete-Time Linear Systems

We now generalize our analysis to systems with $m$ dimensions, where the state is represented by a vector and the dynamics are governed by a matrix transformation.

#### The General Affine Map

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($m$-Dimensional Affine Map)</span></p>

The state of the system is a vector $x_t \in \mathbb{R}^m$. The evolution is given by the **affine map**:

$$x_{t+1} = Ax_t + b$$

where $A$ is an $m \times m$ square matrix and $b \in \mathbb{R}^m$ is a constant offset vector.

</div>

#### Solving for Fixed Points in $m$ Dimensions

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">($m$-Dimensional Fixed Point of Linear Map)</span></p>

Fixed point of $m$-dimensional linear map $x_t = Ax_{t-1} + b$ is 

$$x^{\ast} = (I-A)^{-1}b$$

If $(I-A)$ is not invertible (i.e., it is singular), a unique fixed point does not exist. In this case, the system may have no fixed points or a continuous set of fixed points, such as a line attractor or a higher-dimensional manifold attractor.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $x^{\ast}$ be a fixed point. It must satisfy the condition $x^{\ast} = Ax^{\ast} + b$. We solve for $x^{\ast}$:

$$x^{\ast} - Ax^{\ast} = b$$

$$(I - A)x^{\ast} = b$$

where $I$ is the $m \times m$ identity matrix. If the matrix $(I-A)$ is invertible, we can find the unique fixed point by multiplying by its inverse:  

$$x^{\ast} = (I-A)^{-1}b$$

If $(I-A)$ is not invertible (i.e., it is singular), a unique fixed point does not exist. In this case, the system may have no fixed points or a continuous set of fixed points, such as a line attractor or a higher-dimensional manifold attractor.

</details>
</div>

#### System Dynamics and Diagonalization

To understand the system's trajectory, we analyze the behavior of the map when iterated over time. For simplicity, we first consider the homogeneous case where $b = 0$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Iterated Map Dynamics)</span></p>

For the system $x_{t+1} = Ax_t$, the state at time $T$ is related to the initial state $x_1$ by:

$$x_T = A^{T-1}x_1$$

If $A$ is diagonalizable, then for $A = V \Lambda V^{-1}$ we obtain

$$x_T = V \Lambda^{T-1} V^{-1}x_1$$  

</div>

#### Stability Analysis via Eigenvalues

The stability of the fixed point (in this case, the origin, since $b=0$) is determined by the magnitudes of the eigenvalues of the matrix $A$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stability Analysis of Multidimensional Linear Map)</span></p>

* **Convergence:** The system converges to the fixed point if all eigenvalues have an absolute value less than $1$.  
  
  $$\text{If } \max_i \lvert\lambda_i\rvert < 1 \implies \text{Convergence}$$
  
  As $T \to \infty$, $\Lambda^{T-1} \to 0$, causing $x_T \to 0$.

* **Divergence:** The system diverges if at least one eigenvalue has an absolute value greater than $1$.  
  
  $$\text{If } \max_i \lvert\lambda_i\rvert > 1 \implies \text{Divergence}$$
  
  The component of the trajectory along the eigenvector corresponding to this eigenvalue will grow without bound.

* **Neutral Stability / Manifold Attractors:** If at least one eigenvalue has an absolute value of exactly $1$ (and no eigenvalues have absolute values greater than $1$), the system has neutral directions.  
  
  $$\text{If } \max_i \lvert\lambda_i\rvert = 1 \implies \text{Line or Manifold Attractor}$$
  
  The system will neither converge to the origin nor diverge to infinity, but will instead move along a stable manifold defined by the eigenvectors associated with the eigenvalues of magnitude one.

* **Saddle-like Behavior:** If the matrix $A$ has a mix of eigenvalues with magnitudes greater than and less than one, the system exhibits behavior analogous to a saddle point. Trajectories will converge towards the fixed point along directions spanned by eigenvectors with $\lvert\lambda_i\rvert < 1$ but will diverge along directions spanned by eigenvectors with $\lvert\lambda_i\rvert > 1$.

</div>

<div id="ee-container" style="margin:2em auto;max-width:1060px;">
  <h4 style="text-align:center;margin:0 0 .2em;">Interactive: Eigenvalue Dynamics of \(x_{n+1}=Ax_n\)</h4>
  <p style="text-align:center;color:#888;font-size:.82em;margin:0 0 .8em;">
    Drag the orange eigenvalue on the left to explore dynamics. Snaps to the real axis for 1D cobweb view.
  </p>
  <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:12px;">
    <div style="text-align:center;">
      <div id="ee-ltitle" style="font-size:.85em;font-weight:600;margin-bottom:3px;">Eigenvalues in complex plane</div>
      <canvas id="ee-ec" style="border:1px solid #ddd;border-radius:3px;background:#fff;max-width:100%;"></canvas>
    </div>
    <div style="text-align:center;">
      <div id="ee-rtitle" style="font-size:.85em;font-weight:600;margin-bottom:3px;">Linear map: rotation by 60°</div>
      <canvas id="ee-dc" style="border:1px solid #ddd;border-radius:3px;background:#fff;max-width:100%;"></canvas>
    </div>
  </div>
  <div id="ee-info" style="text-align:center;font-size:.82em;margin-top:.5em;font-family:serif;color:#555;"></div>
</div>

<script>
(function(){
  var S=500, ER=1.5, DR=2.3, SNAP=0.06;
  var COL=['#1a1a2e','#1565C0','#c62828','#e65100','#1b5e20','#7b1fa2','#00838f'];
  var re=0.5, im=Math.sqrt(3)/2, drag=false, dw=0;
  var ec=document.getElementById('ee-ec'), dc=document.getElementById('ee-dc');
  var dpr=window.devicePixelRatio||1;

  function initC(c){c.width=S*dpr;c.height=S*dpr;c.style.width=S+'px';c.style.height=S+'px';var x=c.getContext('2d');x.scale(dpr,dpr);return x;}
  var E=initC(ec), D=initC(dc);

  function m2c(x,y,R){return[(x/R+1)*S/2,(1-y/R)*S/2];}
  function c2m(cx,cy,R){return[(cx/S*2-1)*R,(1-cy/S*2)*R];}

  function ln(c,x1,y1,x2,y2){c.beginPath();c.moveTo(x1,y1);c.lineTo(x2,y2);c.stroke();}
  function circ(c,x,y,r,f){c.beginPath();c.arc(x,y,r,0,Math.PI*2);if(f)c.fill();else c.stroke();}
  function arw(c,x,y,a,sz){c.save();c.translate(x,y);c.rotate(a);c.beginPath();c.moveTo(0,0);c.lineTo(-sz,-sz*.38);c.lineTo(-sz,sz*.38);c.closePath();c.fill();c.restore();}

  function grid(c,R,ticks){
    c.strokeStyle='#f0f0f0';c.lineWidth=.5;
    ticks.forEach(function(v){if(!v)return;
      var p=m2c(v,-R,R),q=m2c(v,R,R);ln(c,p[0],p[1],q[0],q[1]);
      p=m2c(-R,v,R);q=m2c(R,v,R);ln(c,p[0],p[1],q[0],q[1]);
    });
    c.strokeStyle='#81D4FA';c.lineWidth=1;
    var a=m2c(-R,0,R),b=m2c(R,0,R);ln(c,a[0],a[1],b[0],b[1]);
    a=m2c(0,-R,R);b=m2c(0,R,R);ln(c,a[0],a[1],b[0],b[1]);
    c.font='9px sans-serif';c.fillStyle='#bbb';
    ticks.forEach(function(v){if(!v)return;
      var t=m2c(v,0,R);c.fillText(v,t[0]-4,t[1]+13);
      t=m2c(0,v,R);c.fillText(v,t[0]+5,t[1]+3);
    });
  }

  function drawEigen(){
    E.clearRect(0,0,S,S);
    grid(E,ER,[-1,-.5,.5,1]);
    E.strokeStyle='#1976D2';E.lineWidth=2;
    var o=m2c(0,0,ER);circ(E,o[0],o[1],S/(2*ER),false);
    var p1=m2c(re,im,ER),p2=m2c(re,-im,ER);
    E.strokeStyle='#333';E.fillStyle='#333';E.lineWidth=1.5;
    ln(E,o[0],o[1],p1[0],p1[1]);
    arw(E,p1[0],p1[1],Math.atan2(p1[1]-o[1],p1[0]-o[0]),8);
    if(im>SNAP){
      ln(E,o[0],o[1],p2[0],p2[1]);
      arw(E,p2[0],p2[1],Math.atan2(p2[1]-o[1],p2[0]-o[0]),8);
    }
    E.fillStyle='#FF9800';circ(E,p1[0],p1[1],7,true);
    E.strokeStyle='#E65100';E.lineWidth=1.5;circ(E,p1[0],p1[1],7,false);
    E.fillStyle='#4CAF50';circ(E,p2[0],p2[1],7,true);
    E.strokeStyle='#2E7D32';E.lineWidth=1.5;circ(E,p2[0],p2[1],7,false);
    E.font='12px "Times New Roman",serif';E.fillStyle='#333';
    if(im>SNAP){
      E.fillText('\u03BB = '+re.toFixed(2)+(im>=0?'+':'')+im.toFixed(2)+'i',p1[0]+10,p1[1]-10);
      E.fillText('\u03BB\u0304 = '+re.toFixed(2)+(im>=0?'\u2212':'+')+Math.abs(im).toFixed(2)+'i',p2[0]+10,p2[1]+16);
    }else{
      E.fillText('\u03BB = '+re.toFixed(2),p1[0]+10,p1[1]-10);
    }
    E.font='12px "Times New Roman",serif';E.fillStyle='#888';
    E.fillText('\u211C(\u03BB)',S-42,S/2-6);
    E.fillText('\u2111(\u03BB)',S/2+6,13);
  }

  function draw2D(){
    D.clearRect(0,0,S,S);
    grid(D,DR,[-2,-1,1,2]);
    D.strokeStyle='#ece0f0';D.lineWidth=.7;
    var oc=m2c(0,0,DR);
    [.5,1,1.5,2].forEach(function(r){circ(D,oc[0],oc[1],r*S/(2*DR),false);});
    var rr=[.3,.5,.7,1,1.3,1.7];
    rr.forEach(function(r0,k){
      var x=r0,y=0,pts=[[x,y]];
      for(var i=0;i<50;i++){
        var nx=re*x-im*y,ny=im*x+re*y;x=nx;y=ny;
        if(x*x+y*y>DR*DR*4)break;
        pts.push([x,y]);
      }
      if(pts.length<2)return;
      var c=COL[k%COL.length];D.strokeStyle=c;D.fillStyle=c;D.lineWidth=1.5;
      D.beginPath();var s=m2c(pts[0][0],pts[0][1],DR);D.moveTo(s[0],s[1]);
      for(var i=1;i<pts.length;i++){var p=m2c(pts[i][0],pts[i][1],DR);D.lineTo(p[0],p[1]);}
      D.stroke();
      var step=Math.max(2,Math.floor(pts.length/5));
      for(var i=step;i<pts.length;i+=step){
        var cu=m2c(pts[i][0],pts[i][1],DR),pr=m2c(pts[i-1][0],pts[i-1][1],DR);
        var dx=cu[0]-pr[0],dy=cu[1]-pr[1];
        if(dx*dx+dy*dy>4)arw(D,cu[0],cu[1],Math.atan2(dy,dx),6);
      }
      circ(D,s[0],s[1],3.5,true);
    });
    D.font='12px "Times New Roman",serif';D.fillStyle='#888';
    D.fillText('x\u2081',S-18,S/2-6);D.fillText('x\u2082',S/2+6,13);
  }

  function drawCob(){
    D.clearRect(0,0,S,S);var R=DR,lam=re;
    grid(D,R,[-2,-1,1,2]);
    D.save();D.setLineDash([6,4]);D.strokeStyle='#FF9800';D.lineWidth=1.5;
    var b1=m2c(-R,-R,R),b2=m2c(R,R,R);ln(D,b1[0],b1[1],b2[0],b2[1]);D.restore();
    D.strokeStyle='#1976D2';D.lineWidth=2;
    var p1=m2c(-3,-lam*3,R),p2=m2c(3,lam*3,R);ln(D,p1[0],p1[1],p2[0],p2[1]);
    [-1.5,-.5,.5,1.5].forEach(function(x0,k){
      var c=COL[k%COL.length];D.strokeStyle=c;D.fillStyle=c;D.lineWidth=1.2;
      var x=x0,pts=[m2c(x,0,R)];
      for(var i=0;i<30;i++){
        var y=lam*x;
        if(Math.abs(y)>R*2){pts.push(m2c(x,Math.max(-R*2,Math.min(R*2,y)),R));break;}
        pts.push(m2c(x,y,R));pts.push(m2c(y,y,R));x=y;
        if(Math.abs(x)>R*2)break;
      }
      D.beginPath();D.moveTo(pts[0][0],pts[0][1]);
      for(var i=1;i<pts.length;i++)D.lineTo(pts[i][0],pts[i][1]);
      D.stroke();circ(D,pts[0][0],pts[0][1],3.5,true);
    });
    D.font='12px "Times New Roman",serif';
    D.fillStyle='#1976D2';D.fillText('f(x) = '+lam.toFixed(2)+'x',8,16);
    D.fillStyle='#FF9800';D.fillText('y = x',8,30);
    D.fillStyle='#888';D.fillText('x\u2099',S-20,S/2-6);
    D.fillText('x\u2099\u208A\u2081',S/2+6,13);
  }

  function updInfo(){
    var el=document.getElementById('ee-info');
    var mod=Math.sqrt(re*re+im*im),ang=Math.atan2(im,re);
    var beh=mod<.995?'Stable (converging)':mod>1.005?'Unstable (diverging)':'Neutral (|\u03BB|\u22481)';
    if(im<=SNAP)el.innerHTML='\u03BB = '+re.toFixed(3)+' &nbsp;|&nbsp; |\u03BB| = '+mod.toFixed(3)+' &nbsp;|&nbsp; '+beh;
    else el.innerHTML='\u03BB = '+re.toFixed(3)+(im>=0?' + ':' \u2212 ')+Math.abs(im).toFixed(3)+'i &nbsp;|&nbsp; |\u03BB| = '+mod.toFixed(3)+' &nbsp;|&nbsp; arg(\u03BB) = '+(ang*180/Math.PI).toFixed(1)+'\u00B0 &nbsp;|&nbsp; '+beh;
  }
  function updTitles(){
    var mod=Math.sqrt(re*re+im*im),ang=Math.atan2(im,re);
    var lt=document.getElementById('ee-ltitle'),rt=document.getElementById('ee-rtitle');
    lt.textContent=im>SNAP?'Eigenvalues in complex plane':'Eigenvalue in complex plane';
    if(im<=SNAP){
      if(Math.abs(re-1)<.03)rt.textContent='Neutral case: x\u2099\u208A\u2081 = x\u2099';
      else if(Math.abs(re+1)<.03)rt.textContent='Flip case: x\u2099\u208A\u2081 = \u2212x\u2099';
      else rt.textContent='Cobweb: x\u2099\u208A\u2081 = '+re.toFixed(2)+'\u00B7x\u2099';
    }else{
      var d=(ang*180/Math.PI).toFixed(0);
      if(mod<.995)rt.textContent='Contracting spiral (|\u03BB| = '+mod.toFixed(2)+')';
      else if(mod>1.005)rt.textContent='Expanding spiral (|\u03BB| = '+mod.toFixed(2)+')';
      else rt.textContent='Rotation by '+d+'\u00B0';
    }
  }
  function redraw(){drawEigen();if(im<=SNAP)drawCob();else draw2D();updInfo();updTitles();}

  function mpos(e){var r=ec.getBoundingClientRect();return c2m((e.clientX-r.left)*(S/r.width),(e.clientY-r.top)*(S/r.height),ER);}
  ec.addEventListener('mousedown',function(e){
    var m=mpos(e);
    var d1=(m[0]-re)*(m[0]-re)+(m[1]-im)*(m[1]-im);
    var d2=(m[0]-re)*(m[0]-re)+(m[1]+im)*(m[1]+im);
    if(d1<.04){drag=true;dw=1;}else if(d2<.04&&im>SNAP){drag=true;dw=2;}
  });
  ec.addEventListener('mousemove',function(e){
    if(!drag){var m=mpos(e);var d1=(m[0]-re)*(m[0]-re)+(m[1]-im)*(m[1]-im);var d2=(m[0]-re)*(m[0]-re)+(m[1]+im)*(m[1]+im);ec.style.cursor=(d1<.04||(d2<.04&&im>SNAP))?'grab':'crosshair';return;}
    ec.style.cursor='grabbing';var m=mpos(e);
    re=m[0];im=Math.abs(m[1])<SNAP?0:Math.abs(m[1]);
    re=Math.max(-ER+.05,Math.min(ER-.05,re));im=Math.max(0,Math.min(ER-.05,im));
    redraw();
  });
  window.addEventListener('mouseup',function(){drag=false;dw=0;ec.style.cursor='crosshair';});

  ec.addEventListener('touchstart',function(e){e.preventDefault();var t=e.touches[0],r=ec.getBoundingClientRect();var m=c2m((t.clientX-r.left)*(S/r.width),(t.clientY-r.top)*(S/r.height),ER);var d1=(m[0]-re)*(m[0]-re)+(m[1]-im)*(m[1]-im);var d2=(m[0]-re)*(m[0]-re)+(m[1]+im)*(m[1]+im);if(d1<.06){drag=true;dw=1;}else if(d2<.06&&im>SNAP){drag=true;dw=2;}},{passive:false});
  ec.addEventListener('touchmove',function(e){e.preventDefault();if(!drag)return;var t=e.touches[0],r=ec.getBoundingClientRect();var m=c2m((t.clientX-r.left)*(S/r.width),(t.clientY-r.top)*(S/r.height),ER);re=m[0];im=Math.abs(m[1])<SNAP?0:Math.abs(m[1]);re=Math.max(-ER+.05,Math.min(ER-.05,re));im=Math.max(0,Math.min(ER-.05,im));redraw();},{passive:false});
  ec.addEventListener('touchend',function(){drag=false;dw=0;});

  redraw();
})();
</script>

<div id="re-container" style="margin:2em auto;max-width:1060px;">
  <h4 style="text-align:center;margin:0 0 .2em;">Interactive: Two Independent Real Eigenvalues</h4>
  <p style="text-align:center;color:#888;font-size:.82em;margin:0 0 .8em;">
    Drag each eigenvalue independently on the real axis. The map is \(x_{n+1}=\mathrm{diag}(\lambda_1,\lambda_2)\,x_n\).
  </p>
  <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:12px;">
    <div style="text-align:center;">
      <div id="re-ltitle" style="font-size:.85em;font-weight:600;margin-bottom:3px;">Eigenvalues in complex plane</div>
      <canvas id="re-ec" style="border:1px solid #ddd;border-radius:3px;background:#fff;max-width:100%;"></canvas>
    </div>
    <div style="text-align:center;">
      <div id="re-rtitle" style="font-size:.85em;font-weight:600;margin-bottom:3px;">Phase portrait: stable node</div>
      <canvas id="re-dc" style="border:1px solid #ddd;border-radius:3px;background:#fff;max-width:100%;"></canvas>
    </div>
  </div>
  <div id="re-info" style="text-align:center;font-size:.82em;margin-top:.5em;font-family:serif;color:#555;"></div>
</div>

<script>
(function(){
  var S=500, ER=1.8, DR=2.3, MAX=50;
  var COL=['#1a1a2e','#1565C0','#c62828','#e65100','#1b5e20','#7b1fa2','#00838f','#4e342e'];
  var l1=0.5, l2=-0.3; // two independent real eigenvalues
  var drag=false, dw=0; // dw: 1=lambda1, 2=lambda2

  var ec=document.getElementById('re-ec'), dc=document.getElementById('re-dc');
  var dpr=window.devicePixelRatio||1;
  function initC(c){c.width=S*dpr;c.height=S*dpr;c.style.width=S+'px';c.style.height=S+'px';var x=c.getContext('2d');x.scale(dpr,dpr);return x;}
  var E=initC(ec), D=initC(dc);

  function m2c(x,y,R){return[(x/R+1)*S/2,(1-y/R)*S/2];}
  function c2m(cx,cy,R){return[(cx/S*2-1)*R,(1-cy/S*2)*R];}
  function ln(c,x1,y1,x2,y2){c.beginPath();c.moveTo(x1,y1);c.lineTo(x2,y2);c.stroke();}
  function circ(c,x,y,r,f){c.beginPath();c.arc(x,y,r,0,Math.PI*2);if(f)c.fill();else c.stroke();}
  function arw(c,x,y,a,sz){c.save();c.translate(x,y);c.rotate(a);c.beginPath();c.moveTo(0,0);c.lineTo(-sz,-sz*.38);c.lineTo(-sz,sz*.38);c.closePath();c.fill();c.restore();}

  function grid(c,R,ticks){
    c.strokeStyle='#f0f0f0';c.lineWidth=.5;
    ticks.forEach(function(v){if(!v)return;
      var p=m2c(v,-R,R),q=m2c(v,R,R);ln(c,p[0],p[1],q[0],q[1]);
      p=m2c(-R,v,R);q=m2c(R,v,R);ln(c,p[0],p[1],q[0],q[1]);
    });
    c.strokeStyle='#81D4FA';c.lineWidth=1;
    var a=m2c(-R,0,R),b=m2c(R,0,R);ln(c,a[0],a[1],b[0],b[1]);
    a=m2c(0,-R,R);b=m2c(0,R,R);ln(c,a[0],a[1],b[0],b[1]);
    c.font='9px sans-serif';c.fillStyle='#bbb';
    ticks.forEach(function(v){if(!v)return;
      var t=m2c(v,0,R);c.fillText(v,t[0]-4,t[1]+13);
      t=m2c(0,v,R);c.fillText(v,t[0]+5,t[1]+3);
    });
  }

  // === EIGENVALUE PLANE ===
  function drawEigen(){
    E.clearRect(0,0,S,S);
    grid(E,ER,[-1.5,-1,-.5,.5,1,1.5]);
    // Unit circle
    E.strokeStyle='#1976D2';E.lineWidth=2;
    var o=m2c(0,0,ER);circ(E,o[0],o[1],S/(2*ER),false);
    // Arrows from origin to eigenvalues
    var p1=m2c(l1,0,ER), p2=m2c(l2,0,ER);
    E.strokeStyle='#333';E.fillStyle='#333';E.lineWidth=1.5;
    ln(E,o[0],o[1],p1[0],p1[1]);
    arw(E,p1[0],p1[1],Math.atan2(0,p1[0]-o[0]),8);
    ln(E,o[0],o[1],p2[0],p2[1]);
    arw(E,p2[0],p2[1],Math.atan2(0,p2[0]-o[0]),8);
    // Orange dot (λ₁)
    E.fillStyle='#FF9800';circ(E,p1[0],p1[1],7,true);
    E.strokeStyle='#E65100';E.lineWidth=1.5;circ(E,p1[0],p1[1],7,false);
    // Green dot (λ₂)
    E.fillStyle='#4CAF50';circ(E,p2[0],p2[1],7,true);
    E.strokeStyle='#2E7D32';E.lineWidth=1.5;circ(E,p2[0],p2[1],7,false);
    // Labels (offset vertically to avoid overlap)
    E.font='12px "Times New Roman",serif';E.fillStyle='#333';
    E.fillText('\u03BB\u2081 = '+l1.toFixed(2),p1[0]+10,p1[1]-12);
    E.fillText('\u03BB\u2082 = '+l2.toFixed(2),p2[0]+10,p2[1]+20);
    // Axis labels
    E.fillStyle='#888';
    E.fillText('\u211C(\u03BB)',S-42,S/2-6);
    E.fillText('\u2111(\u03BB)',S/2+6,13);
  }

  // === 2D PHASE PORTRAIT ===
  function draw2D(){
    D.clearRect(0,0,S,S);
    grid(D,DR,[-2,-1,1,2]);
    // Reference circles
    D.strokeStyle='#ece0f0';D.lineWidth=.7;
    var oc=m2c(0,0,DR);
    [.5,1,1.5,2].forEach(function(r){circ(D,oc[0],oc[1],r*S/(2*DR),false);});

    // Eigenvector direction lines (along axes for diagonal matrix)
    D.save();D.setLineDash([4,4]);D.lineWidth=1;
    // e1 direction color based on |λ₁|
    D.strokeStyle=Math.abs(l1)<1?'#43A047':'#E53935';
    var a=m2c(-DR,0,DR),b=m2c(DR,0,DR);ln(D,a[0],a[1],b[0],b[1]);
    // e2 direction
    D.strokeStyle=Math.abs(l2)<1?'#43A047':'#E53935';
    a=m2c(0,-DR,DR);b=m2c(0,DR,DR);ln(D,a[0],a[1],b[0],b[1]);
    D.restore();

    // Initial conditions: points on circles at various angles
    var angles=[0,Math.PI/4,Math.PI/2,3*Math.PI/4,Math.PI,5*Math.PI/4,3*Math.PI/2,7*Math.PI/4];
    var radii=[.6,1.2,1.8];
    var ics=[];
    radii.forEach(function(r){angles.forEach(function(a){ics.push([r*Math.cos(a),r*Math.sin(a)]);});});

    ics.forEach(function(ic,k){
      var x=ic[0],y=ic[1],pts=[[x,y]];
      for(var i=0;i<MAX;i++){
        x*=l1;y*=l2;
        if(x*x+y*y>DR*DR*4)break;
        pts.push([x,y]);
        // Stop if converged to origin
        if(x*x+y*y<1e-8)break;
      }
      if(pts.length<2)return;
      var col=COL[k%COL.length];
      D.strokeStyle=col;D.fillStyle=col;D.lineWidth=1.2;
      D.beginPath();
      var s=m2c(pts[0][0],pts[0][1],DR);D.moveTo(s[0],s[1]);
      for(var i=1;i<pts.length;i++){var p=m2c(pts[i][0],pts[i][1],DR);D.lineTo(p[0],p[1]);}
      D.stroke();
      // Arrows
      var step=Math.max(2,Math.floor(pts.length/4));
      for(var i=step;i<pts.length;i+=step){
        var cu=m2c(pts[i][0],pts[i][1],DR),pr=m2c(pts[i-1][0],pts[i-1][1],DR);
        var dx=cu[0]-pr[0],dy=cu[1]-pr[1];
        if(dx*dx+dy*dy>3)arw(D,cu[0],cu[1],Math.atan2(dy,dx),5);
      }
      // Start dot
      circ(D,s[0],s[1],2.5,true);
    });

    // Axis labels
    D.font='12px "Times New Roman",serif';D.fillStyle='#888';
    D.fillText('x\u2081',S-18,S/2-6);D.fillText('x\u2082',S/2+6,13);
    // Eigenvector legend
    D.font='11px "Times New Roman",serif';
    D.fillStyle=Math.abs(l1)<1?'#43A047':'#E53935';
    D.fillText('e\u2081 (\u03BB\u2081='+l1.toFixed(2)+')',6,16);
    D.fillStyle=Math.abs(l2)<1?'#43A047':'#E53935';
    D.fillText('e\u2082 (\u03BB\u2082='+l2.toFixed(2)+')',6,30);
    D.fillStyle='#888';D.font='10px sans-serif';
    D.fillText('green = stable, red = unstable',6,44);
  }

  // === UI ===
  function classify(){
    var a1=Math.abs(l1),a2=Math.abs(l2);
    if(a1<1-1e-3&&a2<1-1e-3)return'Stable node';
    if(a1>1+1e-3&&a2>1+1e-3)return'Unstable node';
    if((a1<1-1e-3&&a2>1+1e-3)||(a1>1+1e-3&&a2<1-1e-3))return'Saddle point';
    if(Math.abs(a1-1)<1e-2&&Math.abs(a2-1)<1e-2)return'Both neutral';
    if(Math.abs(a1-1)<1e-2||Math.abs(a2-1)<1e-2){
      if(a1<1||a2<1)return'Center-stable';
      return'Center-unstable';
    }
    return'Mixed';
  }
  function updInfo(){
    var el=document.getElementById('re-info');
    var cls=classify();
    el.innerHTML='\u03BB\u2081 = '+l1.toFixed(3)+' &nbsp;|&nbsp; \u03BB\u2082 = '+l2.toFixed(3)+
      ' &nbsp;|&nbsp; |\u03BB\u2081| = '+Math.abs(l1).toFixed(3)+
      ' &nbsp;|&nbsp; |\u03BB\u2082| = '+Math.abs(l2).toFixed(3)+
      ' &nbsp;|&nbsp; '+cls;
  }
  function updTitles(){
    document.getElementById('re-ltitle').textContent='Eigenvalues on the real axis';
    document.getElementById('re-rtitle').textContent='Phase portrait: '+classify();
  }
  function redraw(){drawEigen();draw2D();updInfo();updTitles();}

  // === MOUSE ===
  function mpos(e){var r=ec.getBoundingClientRect();return c2m((e.clientX-r.left)*(S/r.width),(e.clientY-r.top)*(S/r.height),ER);}
  ec.addEventListener('mousedown',function(e){
    var m=mpos(e);
    var d1=Math.abs(m[0]-l1), d2=Math.abs(m[0]-l2);
    // Pick closest if both near, but prefer λ₁ if tied
    if(d1<.12&&(d1<=d2||d2>=.12)){drag=true;dw=1;}
    else if(d2<.12){drag=true;dw=2;}
  });
  ec.addEventListener('mousemove',function(e){
    var m=mpos(e);
    if(!drag){
      var d1=Math.abs(m[0]-l1),d2=Math.abs(m[0]-l2);
      ec.style.cursor=(d1<.12||d2<.12)?'grab':'crosshair';
      return;
    }
    ec.style.cursor='grabbing';
    var nx=Math.max(-ER+.05,Math.min(ER-.05,m[0]));
    if(dw===1)l1=nx; else l2=nx;
    redraw();
  });
  window.addEventListener('mouseup',function(){if(drag){drag=false;dw=0;ec.style.cursor='crosshair';}});

  // === TOUCH ===
  ec.addEventListener('touchstart',function(e){e.preventDefault();
    var t=e.touches[0],r=ec.getBoundingClientRect();
    var m=c2m((t.clientX-r.left)*(S/r.width),(t.clientY-r.top)*(S/r.height),ER);
    var d1=Math.abs(m[0]-l1),d2=Math.abs(m[0]-l2);
    if(d1<.15&&(d1<=d2||d2>=.15)){drag=true;dw=1;}
    else if(d2<.15){drag=true;dw=2;}
  },{passive:false});
  ec.addEventListener('touchmove',function(e){e.preventDefault();if(!drag)return;
    var t=e.touches[0],r=ec.getBoundingClientRect();
    var m=c2m((t.clientX-r.left)*(S/r.width),(t.clientY-r.top)*(S/r.height),ER);
    var nx=Math.max(-ER+.05,Math.min(ER-.05,m[0]));
    if(dw===1)l1=nx;else l2=nx;
    redraw();
  },{passive:false});
  ec.addEventListener('touchend',function(){drag=false;dw=0;});

  redraw();
})();
</script>

### The Flow of Dynamical Systems

#### From Continuous to Discrete Time: The Sampling Equivalence

In many scientific and engineering contexts, particularly in physics, systems are naturally modeled using continuous-time differential equations. However, our observation and measurement of these systems are almost always discrete, taken at specific moments in time. This raises a crucial question: can we find a discrete-time system that is equivalent to a continuous-time one?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Flow Discretization using Time Periods)</span></p>

Let us assume we have a continuous-time system that we sample at fixed time steps of duration $\Delta t$. The measurements are taken at times $0, \Delta t, 2\Delta t, \dots, n\Delta t$.

The flow map for this system transports a state from one sample point to the next:

$$x((n+1)\Delta t) = \phi(\Delta t, x(n\Delta t))$$

We can define a new matrix that encapsulates this discrete-time evolution.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/flow_discretization.jpeg' | relative_url }}" alt="a" loading="lazy">
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Discrete-Time Equivalent Matrix)</span></p>

Let a continuous-time linear system be defined by $\dot{x} = Ax$. Its equivalent **discrete-time evolution matrix**, $\tilde{A}$, for a sampling time step $\Delta t$ is defined as:

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

This concept of equivalence can be extended to affine systems of differential equations, which include a constant offset term.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Discrete-Time Equivalent Matrix For Affine Systems)</span></p>

For the continuous-time affine system:

$$\dot{x} = Ax + c$$ 

An equivalent discrete-time affine system has the form:

$$x_{n+1} = \tilde{A} x_n + b$$

where $\tilde{A}$, $b$ are defined: 

$$\tilde{A} = e^{A \Delta t}$$

$$b = (I - \tilde{A})x^{\ast} = -(I - \tilde{A})A^{-1}c$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof for discrete offset</summary>

To find the corresponding offset vector $b$, we can enforce the condition that the fixed points of both the continuous and discrete systems must be identical.

1. **Find the fixed point of the continuous system:** 
   1. Set $\dot{x} = 0$. 
   2. $0 = Ax^{\ast} + c \implies x^{\ast} = -A^{-1}c$
2. **Find the fixed point of the discrete system:** 
   1. Set $x_{n+1} = x_n = x^{\ast}$. 
   2. $x^{\ast} = \tilde{A}x^{\ast} + b \implies (I - \tilde{A})x^{\ast} = b$
3. **Equate and Solve for $b$:** By substituting the expression for $x^{\ast}$ from the continuous system into the discrete system's fixed point equation, we can solve for $b$.

</details>
</div>

#### Applications and Advanced Concepts

The principles of establishing equivalence between continuous and discrete systems are not merely theoretical exercises. They have profound implications in modern machine learning and computational neuroscience.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Piecewise Linear Recurrent Neural Networks)</span></p>

An important class of models in machine learning is the **Piecewise Linear Recurrent Neural Network (PL-RNN)**. These networks are often defined using the Rectified Linear Unit (ReLU) activation function, which is a piecewise linear function.

A typical PL-RNN update rule has the form:

$$x_{n+1} = Ax_{n} + Wg(x_n) + b$$

where $g$ is the ReLU nonlinearity, defined as $g(z) = \max(0, z)$.

The ideas of state-space dynamics and the equivalence between continuous and discrete forms can be extended to analyze these powerful computational models. For those interested in the details of this connection, the following resources are recommended:
* A paper by Monfared and Durstewitz presented at ICML 2020.
* The book Time Series Analysis (2013) by Ozaki, which contains a chapter on defining equivalent formulations for some nonlinear systems.

</div>

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

### Exact discretization vs. Numerical discretization

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Exact discretization vs. Numerical discretization)</span></p>

When passing from a continuous-time dynamical system

$$\dot{x}(t)=f(x(t))$$

to a discrete-time map with step size $h>0$, one should distinguish between the **exact discretization** and a **numerical discretization**. The exact discretization is given by the flow of the ODE over one time step,

$$x_{k+1}=\Phi_h(x_k)$$

where $\Phi_h$ maps the state $x_k$ at time $t_k$ to the exact state at time $t_{k+1}=t_k+h$. **In general, this map is not available in closed form**. Therefore, in practice one often replaces $\Phi_h$ by a numerical approximation $\Psi_h$, obtained for example by Euler or Runge–Kutta methods. This yields a discrete update of the form

$$x_{k+1}=\Psi_h(x_k)$$

for instance, in the explicit Euler scheme,

$$x_{k+1}=x_k+hf(x_k)$$

Hence, the exact discretization preserves the true continuous dynamics at the sampling times, whereas a numerical discretization only approximates them, with an error that depends on the method and the step size $h$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(General Exact Discretization)</span></p>

A **continuous-time dynamical system**

$$\dot{x}(t)=f(x(t),t)$$

is turned into a **discrete-time map**

$$x_{k+1}=F_h(x_k)$$

by choosing a time step $h>0$ and looking only at the state at times

$$t_k = kh$$

1. **Start from the continuous system**

   Suppose

   $$\frac{dx}{dt}=f(x(t))$$

   with initial value $x(0)=x_0$.

   This describes how the state changes continuously in time.

2. **Sample time at discrete points**

   Define

   $$x_k := x(t_k), \qquad t_k = kh.$$

   Now we want a rule that gives $x_{k+1}$ from $x_k$.

   That rule is the **map**.

3. **Exact discrete map**

   From the ODE, the exact solution after one time step is

   $$x_{k+1} = \Phi_h(x_k)$$

   where $\Phi_h$ is the **flow map** of the system over time $h$.

   So the continuous system induces the discrete map

   $$F_h = \Phi_h$$

   This is exact, but usually $\Phi_h$ is not available in closed form.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(General Numerical Discretization)</span></p>

Use the integral form:

$$x(t_{k+1}) = x(t_k) + \int_{t_k}^{t_{k+1}} f(x(t)) dt$$

Approximate the integral by assuming $f(x(t))$ stays roughly constant on the interval:

$$\int_{t_k}^{t_{k+1}} f(x(t)) dt \approx hf(x_k)$$

Then

$$x_{k+1} \approx x_k + h f(x_k)$$

So the discrete map becomes

$$\boxed{x_{k+1}=x_k+h f(x_k)}$$

This is the standard **forward Euler discretisation**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Exact discretization vs. Numerical discretization)</span></p>

Take

$$\dot{x}=ax$$

Then Euler gives

$$x_{k+1}=x_k+hax_k=(1+ha)x_k$$

So the continuous system

$$\dot{x}=ax$$

becomes the discrete map

$$\boxed{x_{k+1}=(1+ha)x_k}$$

The exact map in this case would be

$$x_{k+1}=e^{ah}x_k$$

So:

* **exact discretisation**: $x_{k+1}=e^{ah}x_k$
* **Euler discretisation**: $x_{k+1}=(1+ah)x_k$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Linear DS: Exact Discretization)</span></p>

For a linear continuous system

$$\dot{x}(t)=Ax(t)$$

the exact discrete-time map is

$$\boxed{x_{k+1}=e^{Ah}x_k}$$

If there is an input,

$$\dot{x}(t)=Ax(t)+Bu(t)$$

and the input is held constant on each interval $[t_k,t_{k+1})$, then

$$\boxed{x_{k+1}=A_d x_k + B_d u_k}$$

with

$$
A_d=e^{Ah},
\qquad
B_d=\int_0^h e^{A\tau}B,d\tau.
$$

</div>

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/exact_vs_euler_large_h.png' | relative_url }}" alt="Continuous function at 0" loading="lazy">
  </figure>

  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/exact_vs_euler_small_h.png' | relative_url }}" alt="Discontinuous derivative at 0" loading="lazy">
  </figure>
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

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We must verify that both functions satisfy the differential equation and the initial condition.

* For $u(t) = 0$:
  * **Initial Condition:** $u(0) = 0$, which is satisfied.
  * **Differential Equation:** The time derivative is $\dot{u}(t) = 0$. Plugging into the equation gives $0 = (0)^{2/3} = 0$. The equation holds.
* For $v(t) = t^3$:
  * **Initial Condition:** $v(0) = 0^3 = 0$, which is satisfied.
  * **Differential Equation:** The time derivative is $\dot{v}(t) = 3t^2 = 3v^{2/3}$. 

</details>
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

<div class="gd-grid">
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
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bedrock for much of dynamical systems theory)</span></p>

This theorem is the bedrock for much of dynamical systems theory. It tells us that as long as our system's rules of evolution (the vector field $f$) are smooth, we don't have to worry about non-uniqueness. The solution might not exist for all time (it could "blow up" in finite time), but in some local time interval around our starting point, the path is uniquely determined. While the integral form of the solution is general, it may not be solvable analytically and often requires a numerical solver.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Fundamental Existence and Uniqueness Theorem For Non-Autonomous Systems)</span></p>

The origional thoerem works not only for autonomous systems.

The real principle is:

If the ODE has a unique solution through each initial condition, then **two different trajectories cannot pass through the same state at the same time**.

For an autonomous system

$$\dot x = f(x),$$

this is usually phrased as “trajectories do not intersect in phase space,” because the velocity at a point $x$ is fixed by $f(x)$. If two trajectories met at some point $x_0$, they would have the same future and past by uniqueness, so they were not really different trajectories.

For a nonautonomous system

$$\dot x = f(t,x),$$

the situation is a bit subtler:

* In the **extended state space** $(t,x)$, trajectories still do not intersect, again assuming uniqueness.
* But if you look only at the **$x$-space** and ignore time, then yes, two solution curves can pass through the same point $x$ at different times.

Why? Because the vector field now depends on $t$. Being at the same position $x$ at time $t_1$ and at time $t_2$ are different dynamical situations.

Example:

$$\dot x = t.$$

Solutions are

$$x(t)=\frac{t^2}{2}+C.$$

These are parabolas. In the $t$-$x$ plane they do not intersect if $C$ is different. But if you only look at $x$ as a 1D state, different trajectories can hit the same value $x$ at different times.

An even more direct forced example:

$$\dot x = \cos t.$$

Then

$$x(t)=\sin t + C.$$

Different solutions repeatedly take the same $x$-values at different times.

So the clean statement is:

* **Autonomous system:** trajectories cannot intersect in state space, under uniqueness.
* **Nonautonomous system:** trajectories cannot intersect in $(t,x)$-space, but they **can** appear to intersect in $x$-space if the intersection happens at different times.
* They still cannot have the same $x$ at the same $t$, unless they are actually the same solution, assuming uniqueness.

A very standard trick is to make the nonautonomous system autonomous by adding time as a state:

$$\dot x = f(t,x), \qquad \dot t = 1$$

Then everything becomes autonomous in the enlarged space, and the no-intersection rule is restored there.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Continuously differentiable is sufficient but not necessary)</span></p>

The requirement of being continuously differentiable ($C^1$) is sufficient, but it is actually stronger than necessary. A weaker, more general condition also guarantees uniqueness.

The theorem can be proven under the weaker assumption that the function $f$ is locally Lipschitz continuous. For any two points $x$ and $y$ in some local interval, a function is Lipschitz continuous if the absolute difference in its values is bounded by a constant multiple of the distance between the points.  

$$\lvert f(x) - f(y)\rvert \le L \lvert x - y\rvert$$

Here, $L$ is a positive real number known as the Lipschitz constant. Intuitively, this condition means that the slope of the function is bounded. Every continuously differentiable function is locally Lipschitz, but not every Lipschitz continuous function is differentiable, making this a more general condition.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Picard-Lindelof)</span></p>
  
Let an IVP be given. Let $f$ be globally Lipschitz-continuous with respect to $x$:

$$\lvert f(x) - f(y)\rvert \le L \lvert x - y\rvert$$

For some $L \in \mathbb{R}^+$. Then there exists a unique solution $x: I \to R$ of the IVP for each $x_0 \in R$, where $R$ is a some subset of $\mathbb{R}^n$.

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

