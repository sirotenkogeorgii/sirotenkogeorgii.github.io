---
layout: default
title: Discrete and Continuous Optimization
date: 2025-03-21
excerpt: Notes on discrete and continuous optimization covering linear programming, convex optimization, unconstrained optimization, and more.
tags:
  - optimization
  - mathematics
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

# Chapter 1: Introduction

An optimization problem (or a mathematical programming problem) reads

$$\min f(x) \quad \text{subject to} \quad x \in M,$$

where $f \colon \mathbb{R}^n \to \mathbb{R}$ is the **objective function** and $M \subseteq \mathbb{R}^n$ is the **feasible set**. In general, this problem is undecidable (there is provably no algorithm that can solve it); another well-known undecidable problem is the *halting problem*. On the other hand, there are effectively solvable sub-classes of problems.

Depending on the character of the feasible set $M$, we distinguish two types:

- **Discrete optimization.** The set $M$ is (typically) finite, but usually large enough to inspect and process all feasible solutions. Usually $\lvert M \rvert \ge 2^n$. Examples include the shortest path problem, the minimum spanning tree problem or the minimum matching problem in a graph. These problems are effectively solvable. In contrast, some problems in discrete optimization are NP-hard: integer linear programming, the travelling salesman problem, the knapsack problem or finding the max cut in a graph.

- **Continuous optimization.** Here, the feasible set $M$ is uncountably infinite. Surprisingly, this may pay off: linear programming is polynomially solvable, but the additional integrality requirement makes it NP-hard. The typical problems are linear programming (LP) and diverse kinds of nonlinear programming (such as convex programming, quadratic programming or semidefinite programming).

### Relation Discrete vs. Continuous Optimization

Discrete and continuous optimization are not disjoint. They are closely related and techniques from one area are used in the other. Consider *integer programming*: most of the methods are based on a relaxation to a continuous problem and an iterative improvement. Conversely, an integer condition can easily be reduced to a continuous one. For example, the condition $x \in \lbrace 0, 1 \rbrace$ is equivalent to $x = x^2$ (in reality, however, this is not used).

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.1</span><span class="math-callout__name">(Flows in Networks)</span></p>

Consider a directed graph $G = (V, E)$, where $s \in V$ is the source vertex and $t \in V$ the terminal vertex. Each edge has a capacity, represented by a function $u \colon E \mapsto \mathbb{R}^+$. The objective is to find a maximum flow from $s$ to $t$. The flow coming into any intermediate vertex needs to equal the flow going out of it (flow in $=$ flow out, called the conservation law).

Including an artificial edge $(t, s)$ in the graph, the maximum flow problem is then equivalently formulated as finding the maximum flow through the additional edge:

$$\max \; x_{ts} \quad \text{subject to} \quad \sum_{j:(i,j)\in E} x_{ij} - \sum_{j:(j,i)\in E} x_{ji} = 0, \; \forall i \in V, \quad 0 \le x_{ij} \le u_{ij}, \; \forall (i,j) \in E,$$

which is an integer linear programming problem. Denoting $A$ the incidence matrix of graph $G$, the problem has a compact form

$$\max \; x_{t,s} \quad \text{subject to} \quad Ax = 0, \; 0 \le x \le u.$$

The best known algorithms utilize the discrete nature of the problem. On the other hand, the LP formulation is beneficial, too. Since matrix $A$ is totally unimodular, the resulting optimal solution is automatically integral, provided the capacities are integral. Hence the problem is efficiently solvable by means of linear programming, despite integer conditions.

Another advantage of the LP formulation is that we can easily modify it to different variants of the problem. Consider for example the problem of finding a *minimum-cost flow*. Denote by $c_{ij}$ the cost of sending a unit of flow along the edge $(i,j) \in E$ and by $d > 0$ the minimum required flow. Then the problem reads as an LP problem

$$\min \sum_{(i,j)\in E} c_{ij} x_{ij} \quad \text{subject to} \quad Ax = 0, \; 0 \le x \le u, \; x_{ts} \ge d.$$

</div>

## 1.1 Motivation Examples

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.2</span><span class="math-callout__name">(Theoretical: Eigenvalues)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ be a symmetric matrix and $\lambda_1 \ge \cdots \ge \lambda_n$ its (real) eigenvalues. Consider the unit ball $B$ in space $\mathbb{R}^n$; the ball is defined as $B := \lbrace x \in \mathbb{R}^n;\; \lVert x \rVert_2 \le 1 \rbrace$. The maximal eigenvalue $\lambda_1$ is attained as the maximal value of the quadratic form $x^T A x$ on ball $B$, and similarly the minimal eigenvalue $\lambda_n$ is attained as the minimal value of $x^T A x$ on $B$. Formally:

$$\lambda_1 = \max_{x:\lVert x\rVert_2 \le 1} x^T A x, \quad \lambda_n = \min_{x:\lVert x\rVert_2 \le 1} x^T A x.$$

This is a statement of the *Rayleigh–Ritz theorem*. Let us prove it for $\lambda_1$:

**Inequality "$\le$":** Let $x_1$ be an eigenvector corresponding to $\lambda_1$ and normalized such that $\lVert x_1 \rVert_2 = 1$. Then $Ax_1 = \lambda_1 x_1$. Multiplying by $x_1^T$ from the left yields

$$\lambda_1 = \lambda_1 x_1^T x_1 = x_1^T A x_1 \le \max_{x:\lVert x\rVert_2 = 1} x^T A x.$$

**Inequality "$\ge$":** Let $x \in \mathbb{R}^n$ be an arbitrary vector such that $\lVert x \rVert_2 = 1$. Let $A = Q\Lambda Q^T$ be a spectral decomposition of matrix $A$. Denoting $y := Q^T x$, we have $\lVert y \rVert_2 = 1$ and

$$x^T A x = x^T Q \Lambda Q^T x = y^T \Lambda y = \sum_{i=1}^n \lambda_i y_i^2 \le \sum_{i=1}^n \lambda_1 y_i^2 = \lambda_1 \lVert y \rVert_2^2 = \lambda_1.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.3</span><span class="math-callout__name">(Functional Optimization)</span></p>

In principle, the number of variables need not be finite. In a functional problem, we want to find a function satisfying certain constraints and minimizing a specified criterion. For illustration, imagine computing the best trajectory for a spacecraft traveling from Earth to Mercury; the variable is the curve of the trajectory described by a function, and the objective is to minimize travel time. Certain simple functional problems can be solved analytically, but in general they are solved by discretization of the unknown function and then application of classical optimization methods.

Isoperimetric problems belong to this area, too. It is well-known that the ball has the smallest surface area of all surfaces that enclose a given volume. But how is it when two volumes are given and we wish to minimize the surface area (including the separating surface)? This problem is known as the *double bubble problem*. The minimum area shape consists of two spherical surfaces meeting at angles $120° = \frac{2}{3}\pi$. The separating area is also a spherical surface; it is a disc in case of two equally sized volumes.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.4</span><span class="math-callout__name">(When the Nature Optimizes)</span></p>

Snell's law quantifies the bending of light as it passes through a boundary between two media. The less dense the medium, the faster light travels. The trajectory of light is such that it is traversed in the least time (the so called Fermat's principle of least time).

</div>

## 1.2 Continuous Optimization: First Steps

### Local and Global Minima

A point $x^* \in M$ is called

- a *(global) minimum* if $f(x^*) \le f(x)$ for every $x \in M$,
- a *strict (global) minimum* if $f(x^*) < f(x)$ for every $x^* \ne x \in M$,
- *local minimum* if $f(x^*) \le f(x)$ for every $x \in M \cap \mathcal{O}_\varepsilon(x^*)$,
- a *strict local minimum* if $f(x^*) < f(x)$ for every $x^* \ne x \in M \cap \mathcal{O}_\varepsilon(x^*)$.

Naturally, to solve a problem $\min_{x \in M} f(x)$ means to find its minimum, called the optimal solution. However, sometimes the problem is so hard that we are contented with an approximate solution instead. Be aware that the minimal value of function $f(x)$ on set $M$ need not be attained. Consider for example the problem $\min_{x \in \mathbb{R}} x$, which is unbounded from below, or the problem $\min_{x \in \mathbb{R}} e^x$, which is bounded from below. A sufficient condition for existence of a minimum is given by the Weierstrass theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1.5</span><span class="math-callout__name">(Weierstrass)</span></p>

If $f(x)$ is continuous and $M$ compact, then $f(x)$ attains a minimum on $M$.

</div>

Another problem appears when local minima exist. The basic methods for solving optimization problems are iterative. They start at an initial point and move in the decreasing direction of the objective function. When they approach a local minimum, they get stuck. This phenomenon does occur in linear programming, or more generally in convex optimization, since each local minimum is a global one.

### Classification

The feasible set $M$ is often defined by a system of equations and inequalities

$$g_j(x) \le 0, \quad j = 1, \ldots, J, \qquad h_\ell(x) = 0, \quad \ell = 1, \ldots, L,$$

where $g_j(x), h_\ell(x) \colon \mathbb{R}^n \to \mathbb{R}$. We will employ a short form

$$g(x) \le 0, \quad h(x) = 0,$$

where $g \colon \mathbb{R}^n \to \mathbb{R}^J$ and $h \colon \mathbb{R}^n \to \mathbb{R}^L$. Depending on the type of the objective function and the feasible set, we classify the optimization problems as follows:

- *Linear programming.* Functions $f(x)$, $g_j(x)$, $h_\ell(x)$ are linear.
- *Unconstrained optimization.* Here $M = \mathbb{R}^n$.
- *Convex optimization.* Functions $f(x)$, $g_j(x)$ are convex and $h_\ell(x)$ are linear.

### Basic Transformations

If one wants to find a maximum of $f(x)$ on set $M$, then the problem is easily reduced to the minimization problem

$$\max_{x \in M} f(x) = -\min_{x \in M} -f(x).$$

An equation constraint can be reduced to inequalities since $h(x) = 0$ is equivalent to $h(x) \le 0$, $h(x) \ge 0$, but this is not recommended in view of numerical issues.

**Transformations of functions.** The optimization problem

$$\min \; f(x) \quad \text{subject to} \quad g(x) \le 0, \; h(x) = 0$$

can be transformed to

$$\min \; \varphi(f(x)) \quad \text{subject to} \quad \psi(g(x)) \le 0, \; \eta(h(x)) = 0,$$

provided

- $\varphi(z)$ is increasing on its domain, e.g., $z^k$, $z^{1/k}$, $\log(z)$;
- $\psi(z)$ preserves nonnegativity, i.e., $z \le 0 \;\Leftrightarrow\; \psi(z) \le 0$, e.g., $z^3$;
- $\eta(z)$ preserves roots, i.e., $z = 0 \;\Leftrightarrow\; \eta(z) = 0$, e.g., $z^2$.

Both optimization problems then possess the same minima. The optimal values are different, but they can be easily computed from the optimal solutions.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.6</span><span class="math-callout__name">(Geometric Programming)</span></p>

The transformation turns out to be very convenient in geometric programming, for instance. To illustrate it, consider the particular example

$$\min \; x^2 y \quad \text{subject to} \quad 5xy^3 \le 1, \; 7x^{-3}y \le 1, \; x,y > 0.$$

The logarithm of both sides yields

$$\min \; 2\log(x) + \log(y) \quad \text{subject to} \quad \log(5) + \log(x) + 3\log(y) \le 0, \; \log(7) - 3\log(x) + \log(y) \le 0.$$

The substitution $x' := \log(x)$, $y' := \log(y)$ then leads to an LP problem

$$\min \; 2x' + y' \quad \text{subject to} \quad \log(5) + x' + 3y' \le 0, \; \log(7) - 3x' + y' \le 0.$$

</div>

**Moving the objective function to the constraints.** The frequently used transformation is to move the objective function to the constraints, that is, the problem $\min_{x \in M} f(x)$ is transformed to

$$\min \; z \quad \text{subject to} \quad f(x) \le z, \; x \in M.$$

The objective function is now linear, and all possible obstacles are hidden in the constraints.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.7</span><span class="math-callout__name">(A Finite Minimax)</span></p>

Consider the problem

$$\min_{x \in M} \max_{i=1,\ldots,s} f_i(x).$$

The problems of type min–max are very hard in general. However, in our situation, the outer objective function is the maximum on a finite set. The problem thus can be written as

$$\min \; z \quad \text{subject to} \quad f_i(x) \le z, \; i = 1, \ldots, s, \; x \in M.$$

In the original formulation, the outer objective function $\max_{i=1,\ldots,s} f_i(x)$ is nonsmooth. After the transformation, the objective function is linear.

</div>

**Elimination of equations and variables.** Consider the problem

$$\min \; f(x) \quad \text{subject to} \quad g(x) \le 0, \; Ax = b,$$

where $A \in \mathbb{R}^{m \times n}$ has full row rank. First, we solve the system of equations $Ax = b$. Suppose that the solution set is not empty, so it has the form of $x^0 + \text{Ker}(A)$, where $x^0$ is one (arbitrarily chosen) solution and $\text{Ker}(A)$ is the kernel of $A$. Construct matrix $B \in \mathbb{R}^{n \times (n-m)}$ such that its columns form a basis of $\text{Ker}(A)$. Then any solution of $Ax = b$ can be expressed as $x = x^0 + Bz$, where $z \in \mathbb{R}^{n-m}$. Substitution for $x$ results in optimization problem

$$\min \; f(x^0 + Bz) \quad \text{subject to} \quad g(x^0 + Bz) \le 0.$$

This approach eliminates the equations and reduces the dimension of the problem (i.e., the number of variables) by $m$.

## 1.3 Linear Regression

The problem of linear regression is to find a linear dependence in data $(a_1, b_1), \ldots, (a_m, b_m) \in \mathbb{R}^{n+1}$. Linear regression is widely used in many disciplines, including economy, biology and computer science. In pattern recognition, for example, one wants a computer system to predict and make decisions autonomously (e.g., spam filtering, books and movie recommendations, face recognition, credit card fraud detection). Of course, the true dependence need not be linear and there exist models for nonlinear regression; we focus to the linear case only.

Let the matrix $A \in \mathbb{R}^{m \times n}$ consist of rows $a_1, \ldots, a_m$. Then the goal is to find a vector $x \in \mathbb{R}^n$ such that $Ax \approx b$. Since usually $m \gg n$, the system of linear equations $Ax = b$ is overdetermined and has no solution. Therefore we will seek for an approximate solution.

Mathematically, we can model the problem as an optimization problem to find $x \in \mathbb{R}^n$ such that the difference between the left and right hand side is minimal in a certain norm:

$$\min_{x \in \mathbb{R}^n} \lVert Ax - b \rVert.$$

The geometric interpretation of this problem is to find the projection of vector $b \in \mathbb{R}^m$ to the column space $\mathcal{S}(A)$ of matrix $A$. The typical choices are the following norms:

- **Euclidean norm.** The problem then reads $\min_{x \in \mathbb{R}^n} \lVert Ax - b \rVert_2^2 = \min_{x \in \mathbb{R}^n} \sum_{i=1}^m (A_{i*}x - b_i)^2$, that is, it is the ordinary least squares problem. If matrix $A$ has full column rank, then the solution is unique and has the form $x^* = (A^T A)^{-1} A^T b$. This approach is also justified by statistics: suppose that the dependence is really linear and the entries of the right-hand side vector $b$ are affected by independent and normally distributed errors. Then $x^*$ is the best linear unbiased estimator and also the maximum likelihood estimator.

- **Manhattan norm.** The problem $\min_{x \in \mathbb{R}^n} \lVert Ax - b \rVert_1$ can be expressed as the linear program

  $$\min \; e^T z \quad \text{subject to} \quad -z \le Ax - b \le z, \; z \in \mathbb{R}^m, \; x \in \mathbb{R}^n.$$

  This case has also a statistical interpretation. The optimal solution produces the maximum likelihood estimator as long as the noise follows the Laplace distribution.

- **Maximum norm.** The problem $\min_{x \in \mathbb{R}^n} \lVert Ax - b \rVert_\infty$ is also equivalent to an LP problem

  $$\min \; z \quad \text{subject to} \quad -ze \le Ax - b \le ze, \; z \in \mathbb{R}, \; x \in \mathbb{R}^n.$$

### Outliers

An outlier is an observation that differs significantly from the others. Usually, it is caused by some experimental error. An outlier spoils the linear tendency in data and the resulting estimator can be distorted. The Manhattan norm is less sensitive to outliers than the other norms, but still outliers can cause problems.

If we expect or estimate that there are $k \ll m$ outliers in data, then we can solve the linear regression problem as follows

$$\min \; \lVert A_I x - b_I \rVert \quad \text{subject to} \quad x \in \mathbb{R}^n, \; I \subseteq \lbrace 1, \ldots, m \rbrace, \; \lvert I \rvert \ge m - k,$$

where $A_I, b_I$ denotes submatrices indexed by $I$. Nevertheless, this is a hard combinatorial optimization problem.

### Cardinality

The cardinality of a vector $x \in \mathbb{R}^n$ is the number of nonzero entries and it is denoted by

$$\lVert x \rVert_0 = \lvert \lbrace i;\; x_i \ne 0 \rbrace \rvert.$$

This notation resembles the vector $\ell_p$-norm $\lVert x \rVert_p = \sqrt[p]{\sum_{i=1}^n \lvert x_i \rvert^p}$. Indeed, the cardinality is obtained by the limit transition, neglecting the $p$-th roots

$$\lVert x \rVert_0 = \lim_{p \to 0^+} \sum_{i=1}^n \lvert x_i \rvert^p.$$

However, $\lVert x \rVert_0$ is not a vector norm.

In regression, we usually aim to explain $b$ by using a small number of variables (or regressors), that is, we also want to minimize $\lVert x \rVert_0$. We join both criteria by a weighted sum, resulting to a formulation

$$\min \; \lVert Ax - b \rVert_2 + \gamma \lVert x \rVert_0,$$

where $\gamma > 0$ is a suitably chosen constant. Again, this is a hard combinatorial problem. That is why $\lVert x \rVert_0$ is approximated by the Manhattan norm (in some sense, it is the best approximation). As a consequence, we get an effectively solvable optimization problem

$$\min \; \lVert Ax - b \rVert_2 + \gamma \lVert x \rVert_1.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.8</span><span class="math-callout__name">(Signal Reconstruction)</span></p>

Consider the problem of a signal reconstruction. Let a vector $\tilde{x} \in \mathbb{R}^n$ represent the unknown signal, and let $y = \tilde{x} + \text{err}$ represent the observed noisy signal. We want to smooth the noisy signal and find a good approximation of $\tilde{x}$. To this end, we will seek for a vector $x \in \mathbb{R}^n$ that is close to $y$ and that is also smoothed, i.e., there are not big oscillations.

This idea leads to multi-objective optimization problem

$$\min_{x \in \mathbb{R}^n} \; \lVert x - y \rVert_2, \; \lvert x_{i+1} - x_i \rvert \;\; \forall i.$$

A single-objective scalarization is obtained by a weighted sum of the objectives

$$\min_{x \in \mathbb{R}^n} \; \lVert x - y \rVert_2 + \gamma \sum_{i=1}^{n-1} \lvert x_{i+1} - x_i \rvert,$$

where $\gamma > 0$ is a parameter. A smaller value of $\gamma$ prioritizes the first objective and so the resulting signal is closer to the observed signal, while a larger $\gamma$ penalizes oscillations and produces more smoothed signals.

Denote by $D \in \mathbb{R}^{(n-1) \times n}$ the difference matrix with entries $D_{ii} = 1$, $D_{i,i+1} = -1$ and zeros elsewhere. Then the problem reads

$$\min_{x \in \mathbb{R}^n} \; \lVert x - y \rVert_2 + \gamma \lVert Dx \rVert_1.$$

This can again be viewed as an approximation of a cardinality problem

$$\min_{x \in \mathbb{R}^n} \; \lVert x - y \rVert_2 + \gamma \lVert Dx \rVert_0,$$

in which we aim to find a signal approximation in the form of a piecewise constant function. This approach is called *total variation reconstruction*, and it is used when processing digital signals.

</div>

# Chapter 2: Unconstrained Optimization

An unconstrained optimization problem reads

$$\min \; f(x) \quad \text{subject to} \quad x \in \mathbb{R}^n.$$

The objective function $f \colon \mathbb{R}^n \to \mathbb{R}$ is either general or we impose some differentiability assumptions later on. First we present the well-known first order necessary optimality condition.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.1</span><span class="math-callout__name">(First Order Necessary Optimality Condition)</span></p>

Let $f(x)$ be differentiable and let $x^* \in \mathbb{R}^n$ be a local extremal point. Then $\nabla f(x^*) = o$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Without loss of generality assume that $x^*$ is a local minimum. Recall that for any $i = 1, \ldots, n$

$$\nabla_i f(x) = \frac{\partial f(x^*)}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1^*, \ldots, x_{i-1}^*, x_i^* + h, x_{i+1}^*, \ldots, x_n^*) - f(x^*)}{h}.$$

The limit must be the same if we consider the limit from the left or from the right. In the first case,

$$\nabla_i f(x) = \lim_{h \to 0^+} \frac{f(x_1^*, \ldots, x_{i-1}^*, x_i^* + h, x_{i+1}^*, \ldots, x_n^*) - f(x^*)}{h} \ge 0,$$

and in the second case analogously $\nabla_i f(x) \le 0$. Therefore $\nabla_i f(x) = 0$.

</details>
</div>

Obviously, the above condition is only a necessary condition for optimality since it cannot distinguish between minima, maxima and inflection points. The point with zero gradient is called a **stationary point**.

We mention two second order optimality conditions, one is a necessary condition and one is a sufficient condition.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.2</span><span class="math-callout__name">(Second Order Necessary Optimality Condition)</span></p>

Let $f(x)$ be twice continuously differentiable and let $x^* \in M$ be a local minimum. Then the Hessian matrix $\nabla^2 f(x^*)$ is positive semidefinite.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The continuity of second partial derivatives implies that for every $\lambda \in \mathbb{R}$ and $y \in \mathbb{R}^n$ there is $\theta \in (0,1)$ such that

$$f(x^* + \lambda y) = f(x^*) + \lambda \nabla f(x^*)^T y + \frac{1}{2} \lambda^2 y^T \nabla^2 f(x^* + \theta \lambda y) y. \tag{2.1}$$

In other words, this is Taylor's expansion with Lagrange remainder. Due to minimality of $x^*$ we have $f(x^* + \lambda y) \ge f(x^*)$, and from Theorem 2.1 we have $\nabla f(x^*) = o$. Hence

$$\lambda^2 y^T \nabla^2 f(x^* + \theta \lambda y) y \ge 0.$$

By the limit transition $\lambda \to 0$ we get $y^T \nabla^2 f(x^*) y \ge 0$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.3</span><span class="math-callout__name">(Second Order Sufficient Optimality Condition)</span></p>

Let $f(x)$ be twice continuously differentiable. If $\nabla f(x^*) = o$ and $\nabla^2 f(x^*)$ is positive definite for a certain $x^* \in M$, then $x^*$ is a strict local minimum.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We proceed similarly as in the proof of Theorem 2.2. In equation (2.1) we have for $\lambda \ne 0$, $y \ne o$ and sufficiently small $\lambda$

$$\lambda \nabla f(x^*)^T y = 0, \quad \frac{1}{2} \lambda^2 y^T \nabla^2 f(x^* + \theta \lambda y) y > 0.$$

Therefore $f(x^* + \lambda y) > f(x^*)$.

</details>
</div>

We see that there is a quite tight gap between the necessary and the sufficient conditions. However, the example $f(x) = -x^4$ shows that the gap is not zero: the point $x = 0$ is a strict local maximum (and not a minimum), the sufficient condition is not satisfied (as expected), but the necessary condition is satisfied.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.4</span><span class="math-callout__name">(The Least Squares Method)</span></p>

Consider a system of linear equations $Ax = b$, where $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$ and matrix $A$ has rank $n$ (cf. Section 1.3). Usually, $m$ is much greater than $n$. Since this system has practically never an exact solution, we seek for an approximate solution by means of an optimization problem

$$\min_{x \in \mathbb{R}^n} \lVert Ax - b \rVert_2.$$

Here we aim to find such a vector $x$ that minimizes the Euclidean norm of the difference between the left and right hand sides of system $Ax = b$. Since the square is an increasing function, the minimum is attained at the same point as for the problem

$$\min_{x \in \mathbb{R}^n} \lVert Ax - b \rVert_2^2 = (Ax - b)^T(Ax - b) = x^T A^T A x - 2b^T A x + b^T b.$$

We now check for the assumptions of Theorem 2.3. The gradient of the objective function is $2A^T Ax - 2A^T b$. Since it should be zero, we get the condition $A^T Ax = A^T b$, whence $x = (A^T A)^{-1} A^T b$. The Hessian of the objective function is $2A^T A$, which is a positive definite matrix. Therefore the point $x = (A^T A)^{-1} A^T b$ is a strict local minimum. Moreover, since the objective function is convex, this solution is indeed the global minimum.

If matrix $A$ has not full column rank, then any solution of the system of linear equations $A^T Ax = A^T b$ is a candidate for an optimum. In fact, one can show that all these infinitely many solutions of the system of equations are optimal solutions of our problem.

</div>

# Chapter 3: Convexity

Convex sets and convex functions appeared more than 100 years ago and the topic was pioneered by Hölder (1889), Jensen (1906), Minkowski (1910) and other famous mathematicians.

## 3.1 Convex Sets

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.1</span><span class="math-callout__name">(Convex Set)</span></p>

A set $M \subseteq \mathbb{R}^n$ is *convex* if for every $x_1, x_2 \in M$ and every $\lambda_1, \lambda_2 \ge 0$, $\lambda_1 + \lambda_2 = 1$, the convex combination satisfies $\lambda_1 x_1 + \lambda_2 x_2 \in M$.

</div>

The empty set $\emptyset$ or a singleton $\lbrace x \rbrace$ are convex sets. From the geometric point of view, the convexity of a set $M$ means that for any two points in $M$ the set also includes the whole line segment connecting these two points. The line segment connecting points $x_1$ and $x_2$ will be denoted

$$u(x_1, x_2) := \lbrace x \in \mathbb{R}^n;\; x = \lambda_1 x_1 + \lambda_2 x_2, \; \lambda_1, \lambda_2 \ge 0, \; \lambda_1 + \lambda_2 = 1 \rbrace.$$

Convexity of a set can be equivalently characterized by using convex combinations of all $k$-tuples of its points.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.2</span></p>

Let $k \ge 2$. Then a set $M \subseteq \mathbb{R}^n$ is convex if and only if for any $x_1, \ldots, x_k \in M$ and any $\lambda_1, \ldots, \lambda_k \ge 0$, $\sum_{i=1}^k \lambda_i = 1$ one has $\sum_{i=1}^k \lambda_i x_i \in M$.

</div>

Obviously, the union of convex sets need not be convex. On the other hand, the intersection of convex sets is always convex.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.3</span></p>

If $M_i \subseteq \mathbb{R}^n$, $i \in I$, are convex, then $\cap_{i \in I} M_i$ is convex.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $x_1, x_2 \in \cap_{i \in I} M_i$. Then for every $i \in I$ we have $x_1, x_2 \in M_i$, and hence also their convex combination $\lambda_1 x_1 + \lambda_2 x_2 \in M_i$.

</details>
</div>

This property justifies introduction of the concept of the convex hull of a set $M$ as the minimal (with respect to inclusion) convex set containing $M$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.4</span><span class="math-callout__name">(Convex Hull)</span></p>

The *convex hull* of a set $M \subseteq \mathbb{R}^n$ is the intersection of all convex sets in $\mathbb{R}^n$ including $M$. We denote it by $\text{conv}(M)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.5</span></p>

A set $M \subseteq \mathbb{R}^n$ is convex if and only if $M = \text{conv}(M)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

"$\Rightarrow$" Since $M$ is convex, it is one of those convex sets that are intersected to $\text{conv}(M)$.

"$\Leftarrow$" Due to Theorem 3.3, the set $\text{conv}(M)$ is convex, so $M$ is also convex.

</details>
</div>

Recall that the **relative interior** of a set $M \subseteq \mathbb{R}^n$ is the interior of $M$ when restricted to the smallest affine subspace containing $M$. We denote it by $\text{ri}(M)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.6</span></p>

If $M \subseteq \mathbb{R}^n$ is convex, then $\text{ri}(M)$ is convex.

</div>

An important property of disjoint convex sets is their linear separability.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.7</span><span class="math-callout__name">(Separable Sets)</span></p>

Two nonempty sets $M, N \subseteq \mathbb{R}^n$ are *separable* if there exists a vector $o \ne a \in \mathbb{R}^n$ and a number $b \in \mathbb{R}$ such that

$$a^T x \le b \quad \forall x \in M, \qquad a^T x \ge b \quad \forall x \in N,$$

but not $a^T x = b \quad \forall x \in M \cup N$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.8</span><span class="math-callout__name">(Separation Theorem)</span></p>

Let $M, N \subseteq \mathbb{R}^n$ be nonempty and convex. Then they are separable if and only if $\text{ri}(M) \cap \text{ri}(N) = \emptyset$.

</div>

Let $M \subseteq \mathbb{R}^n$ be convex and closed. Using the separation property we can separate a boundary point $x^* \in M$ and the set $M$ by a hyperplane $a^T x = b$; we call this hyperplane a **supporting hyperplane** of $M$. We then have $a^T x^* = b$ (i.e., the hyperplane contains the point $x^*$) and set $M$ lies in the positive halfspace defined by the hyperplane, that is, $a^T x \le b$ for every $x \in M$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.9</span></p>

Let $M \subseteq \mathbb{R}^n$ be convex and closed. Then $M$ is equal to the intersection of the positive halfspaces determined by all supporting hyperplanes of $M$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

From property $a^T x \le b \;\forall x \in M$ we get that $M$ lies in the intersection of the halfspaces. We prove the converse inclusion by contradiction: If there is $x^* \notin M$ lying in the intersection of the halfspaces, then we can separate it (or more precisely, its neighbourhood) from $M$ by a supporting hyperplane. Thus we found a halfspace not containing $x^*$; a contradiction.

</details>
</div>

The above statement is not only of a theoretical importance. Using supporting hyperplanes we can enclose set $M$ to a convex polyhedron with an arbitrary precision. This property is used in certain algorithms, too; they start with an initial selection of supporting hyperplanes and then they iteratively include other ones when needed, in particular when one has to separate some points from set $M$.

## 3.2 Convex Functions

Convexity regards not only sets, but also functions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.10</span><span class="math-callout__name">(Convex Function)</span></p>

Let $M \subseteq \mathbb{R}^n$ be a convex set. Then a function $f \colon \mathbb{R}^n \to \mathbb{R}$ is *convex* on $M$ if for every $x_1, x_2 \in M$ and every $\lambda_1, \lambda_2 \ge 0$, $\lambda_1 + \lambda_2 = 1$, one has

$$f(\lambda_1 x_1 + \lambda_2 x_2) \le \lambda_1 f(x_1) + \lambda_2 f(x_2).$$

If we have $f(\lambda_1 x_1 + \lambda_2 x_2) < \lambda_1 f(x_1) + \lambda_2 f(x_2)$ for every convex combination with $x_1 \ne x_2$ and $\lambda_1, \lambda_2 > 0$, then $f$ is **strictly convex** on $M$.

</div>

Analogously we define a *concave* function: $f(x)$ is concave if $-f(x)$ is convex. A function is linear (or, more precisely, affine) if and only if it is both convex and concave.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.11</span></p>

Any vector norm is a convex function because by definition for any $x_1, x_2 \in \mathbb{R}^n$ and $\lambda_1, \lambda_2 \ge 0$, $\lambda_1 + \lambda_2 = 1$,

$$\lVert \lambda_1 x_1 + \lambda_2 x_2 \rVert \le \lVert \lambda_1 x_1 \rVert + \lVert \lambda_2 x_2 \rVert = \lambda_1 \lVert x_1 \rVert + \lambda_2 \lVert x_2 \rVert.$$

In particular, the smooth Euclidean norm $\lVert x \rVert_2$ is convex as well as the non-smooth norms $\lVert x \rVert_1$ and $\lVert x \rVert_\infty$, or any matrix norm.

</div>

Analogously as in Theorem 3.2 we can characterize convex functions by means of convex combinations of $k$-tuples of points.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.12</span><span class="math-callout__name">(Jensen's Inequality)</span></p>

Let $k \ge 2$ and let $M \subseteq \mathbb{R}^n$ be convex. Then a function $f \colon \mathbb{R}^n \to \mathbb{R}$ is convex on $M$ if and only if for any $x_1, \ldots, x_k \in M$ and $\lambda_1, \ldots, \lambda_k \ge 0$, $\sum_{i=1}^k \lambda_i = 1$, one has

$$f\!\left(\sum_{i=1}^k \lambda_i x_i\right) \le \sum_{i=1}^k \lambda_i f(x_i).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We will proceed by mathematical induction on $k$. The statement is obvious for $k = 2$, so we turn our attention to the induction step. Define $\alpha := \sum_{i=1}^{k-1} \lambda_i$. Since $\alpha + \lambda_k = 1$ and $\sum_{i=1}^{k-1} \alpha^{-1} \lambda_i = 1$, we get using the induction hypothesis

$$f\!\left(\sum_{i=1}^k \lambda_i x_i\right) = f\!\left(\alpha \sum_{i=1}^{k-1} \alpha^{-1} \lambda_i x_i + \lambda_k x_k\right) \le \alpha f\!\left(\sum_{i=1}^{k-1} \alpha^{-1} \lambda_i x_i\right) + \lambda_k f(x_k) \le \alpha \sum_{i=1}^{k-1} \alpha^{-1} \lambda_i f(x_i) + \lambda_k f(x_k) = \sum_{i=1}^k \lambda_i f(x_i).$$

</details>
</div>

The following observation is useful for practical verification of convexity of a function.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.13</span></p>

A function $f(x)$ is convex on $M$ if and only if it is convex on each segment in $M$. That is, the function $g(t) = f(x + ty)$ is convex on the corresponding compact interval domain of variable $t$ for every $x \in M$ and every $y$ of norm 1.

</div>

Another characterization of convex functions is by means of epigraphs.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.14</span><span class="math-callout__name">(Epigraph)</span></p>

The *epigraph* of a function $f \colon \mathbb{R}^n \to \mathbb{R}$ on a set $M \subseteq \mathbb{R}^n$ is the set

$$\lbrace (x, z) \in \mathbb{R}^{n+1};\; x \in M, \; z \ge f(x) \rbrace.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.15</span><span class="math-callout__name">(Fenchel, 1951)</span></p>

Let $M \subseteq \mathbb{R}^n$ be a convex set. Then a function $f \colon \mathbb{R}^n \to \mathbb{R}$ is convex if and only if its epigraph is a convex set.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

"$\Rightarrow$" Denote by $\mathcal{E}$ the epigraph of $f(x)$ on $M$, and let $(x_1, z_1), (x_2, z_2) \in \mathcal{E}$ be arbitrarily chosen. Consider their convex combination

$$\lambda_1(x_1, z_1) + \lambda_2(x_2, z_2) = (\lambda_1 x_1 + \lambda_2 x_2, \lambda_1 z_1 + \lambda_2 z_2).$$

Due to convexity of $M$ we have $\lambda_1 x_1 + \lambda_2 x_2 \in M$, and convexity of $f(x)$ then implies

$$f(\lambda_1 x_1 + \lambda_2 x_2) \le \lambda_1 f(x_1) + \lambda_2 f(x_2) \le \lambda_1 z_1 + \lambda_2 z_2.$$

"$\Leftarrow$" Let $\mathcal{E}$ be convex. For any $x_1, x_2 \in M$ we have $(x_1, f(x_1)), (x_2, f(x_2)) \in \mathcal{E}$. Consider a convex combination $\lambda_1(x_1, f(x_1)) + \lambda_2(x_2, f(x_2))$. Due to convexity of $\mathcal{E}$ we have

$$\lambda_1(x_1, f(x_1)) + \lambda_2(x_2, f(x_2)) = (\lambda_1 x_1 + \lambda_2 x_2, \lambda_1 f(x_1) + \lambda_2 f(x_2)) \in \mathcal{E},$$

whence $f(\lambda_1 x_1 + \lambda_2 x_2) \le \lambda_1 f(x_1) + \lambda_2 f(x_2)$.

</details>
</div>

The following property is frequently used in optimization. The feasible set $M$ of an optimization problem $\min_{x \in M} f(x)$ is usually described by a system of inequalities $g_j(x) \le 0$, $j = 1, \ldots, J$. If functions $g_j$ are convex, then the set $M$ is convex, too.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.16</span></p>

Let $M \subseteq \mathbb{R}^n$ be a convex set and $f \colon \mathbb{R}^n \to \mathbb{R}$ a convex function. For any $b \in \mathbb{R}$ the set $\lbrace x \in M;\; f(x) \le b \rbrace$ is convex.

</div>

Another nice property of convex functions is their continuity.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.17</span></p>

Let $M \subseteq \mathbb{R}^n$ be a nonempty convex set of dimension $n$, and let $f \colon \mathbb{R}^n \to \mathbb{R}$ be a convex function. Then $f(x)$ is continuous and locally Lipschitz on $\text{int}\, M$.

</div>

## 3.3 The First and Second Order Characterization of Convex Functions

The first order characterization of a convex function $f(x)$ can be viewed visually. The tangent line to the graph of $f(x)$ at any point $(x_1, f(x_1))$ must lie below the graph, that is, $f(x) \ge f(x_1) + \nabla f(x_1)^T (x - x_1)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.18</span><span class="math-callout__name">(First Order Characterization of a Convex Function)</span></p>

Let $\emptyset \ne M \subseteq \mathbb{R}^n$ be a convex set and let $f(x)$ be a function differentiable on an open superset of $M$. Then $f(x)$ is convex on $M$ if and only if for every $x_1, x_2 \in M$

$$f(x_2) - f(x_1) \ge \nabla f(x_1)^T (x_2 - x_1). \tag{3.1}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

"$\Rightarrow$" Let $x_1, x_2 \in M$ and $\lambda \in (0,1)$ be arbitrary. Then

$$f((1-\lambda)x_1 + \lambda x_2) \le (1-\lambda) f(x_1) + \lambda f(x_2),$$

$$\frac{f(x_1 + \lambda(x_2 - x_1)) - f(x_1)}{\lambda} \le f(x_2) - f(x_1).$$

By the limit transition $\lambda \to 0$ we get (3.1) utilizing the chain rule for the derivative of a composite function $g(\lambda) = f(x_1 + \lambda(x_2 - x_1))$ with respect to $\lambda$.

"$\Leftarrow$" Let $x_1, x_2 \in M$ and consider a convex combination $x = \lambda_1 x_1 + \lambda_2 x_2$. By (3.1) we have

$$f(x_1) - f(x) \ge \nabla f(x)^T(x_1 - x) = \lambda_2 \nabla f(x)^T(x_1 - x_2),$$

$$f(x_2) - f(x) \ge \nabla f(x)^T(x_2 - x) = \lambda_1 \nabla f(x)^T(x_2 - x_1).$$

Multiply the first inequality by $\lambda_1$, the second one by $\lambda_2$, and summing up we get

$$\lambda_1(f(x_1) - f(x)) + \lambda_2(f(x_2) - f(x)) \ge 0,$$

or $\lambda_1 f(x_1) + \lambda_2 f(x_2) \ge f(x)$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3.19</span></p>

For strict convexity we have an analogous characterization

$$\forall x_1, x_2 \in M, x_1 \ne x_2 : f(x_2) - f(x_1) > \nabla f(x_1)^T(x_2 - x_1). \tag{3.2}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.20</span><span class="math-callout__name">(Second Order Characterization of a Convex Function)</span></p>

Let $\emptyset \ne M \subseteq \mathbb{R}^n$ be an open convex set of dimension $n$, and suppose that a function $f \colon M \to \mathbb{R}$ is twice continuously differentiable on $M$. Then $f(x)$ is convex on $M$ if and only if the Hessian $\nabla^2 f(x)$ is positive semidefinite for every $x \in M$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $x^* \in M$ be arbitrary. Due to continuity of the second partial derivatives we have that for every $\lambda \in \mathbb{R}$ and $y \in \mathbb{R}^n$, $x^* + \lambda y \in M$, there is $\theta \in (0,1)$ such that

$$f(x^* + \lambda y) = f(x^*) + \lambda \nabla f(x^*)^T y + \frac{1}{2} \lambda^2 y^T \nabla^2 f(x^* + \theta \lambda y) y. \tag{3.3}$$

"$\Rightarrow$" From Theorem 3.18 we get

$$f(x^* + \lambda y) \ge f(x^*) + \lambda \nabla f(x^*)^T y,$$

so that (3.3) implies $y^T \nabla^2 f(x^* + \theta \lambda y) y \ge 0$. By the limit transition $\lambda \to 0$ we have $y^T \nabla^2 f(x^*) y \ge 0$.

"$\Leftarrow$" Due to positive semidefiniteness of the Hessian we have $y^T \nabla^2 f(x^* + \theta \lambda y) y \ge 0$ in the expression (3.3). Hence

$$f(x^* + \lambda y) \ge f(x^*) + \lambda \nabla f(x^*)^T y,$$

which shows convexity of $f(x)$ in view of Theorem 3.18.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3.21</span></p>

For strict convexity, we can state the following conditions:

1. If $f$ is strictly convex, then the Hessian $\nabla^2 f(x)$ is positive definite almost everywhere on $M$; in the remaining cases it is positive semidefinite there.
2. If the Hessian $\nabla^2 f(x)$ is positive definite on $M$, then $f$ is strictly convex.

In the first item, we cannot claim positive definiteness everywhere on $M$. Using an analogous reasoning as in the proof of Theorem 3.20, the limit transition $\lambda \to 0$ can turn the strict inequality to a non-strict one.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.22</span></p>

1. Function $f(x) = x^4$ is strictly convex on $\mathbb{R}$, but its Hessian $f(x)'' = 12x^2$ vanishes at $x = 0$.
2. Consider function $f(x_1, x_2) = -x_1^2$ on the set $M = \lbrace (0,t)^T;\; t \in [0,1] \rbrace$, which is the line segment between the origin and the point $(0,1)^T$. On the set $M$, function $f$ is constant, so also convex. Nevertheless, the Hessian matrix $\nabla^2 f(x) = \begin{pmatrix} -1 & 0 \\ 0 & 0 \end{pmatrix}$ is not positive semidefinite at no point. This justifies why the set $M$ has to be fully-dimensional in Theorem 3.20.
3. Function $f(x) = x^{-2}$ has the second derivatives positive everywhere on $\mathbb{R} \setminus \lbrace 0 \rbrace$, but it is not convex there. The reason is that $\mathbb{R} \setminus \lbrace 0 \rbrace$ is not a convex set, and also the definition of a convex function is not satisfied even when zero avoids the convex combinations. Therefore it is necessary that the domain is a convex set. Hence $f(x)$ is convex separately on $(0, \infty)$ and on $(-\infty, 0)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.23</span></p>

Consider a quadratic function $f \colon \mathbb{R}^n \to \mathbb{R}$ given by formula $f(x) = x^T Ax + b^T x + c$, where $A \in \mathbb{R}^{n \times n}$ is symmetric, $b \in \mathbb{R}^n$ and $c \in \mathbb{R}$. Then

- $f(x)$ is convex if and only if $A$ is positive semidefinite,
- $f(x)$ is strictly convex if and only if $A$ is positive definite.

</div>

## 3.4 Other Rules for Detecting Convexity of a Function

In this section we discuss if or how is convexity preserved under addition, product, composition and other operations.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.24</span></p>

Let $f_1, \ldots, f_k$ be convex functions on a convex set $M \subseteq \mathbb{R}^n$, and let $c_1, \ldots, c_k \ge 0$. Then $\sum_{i=1}^k c_i f_i(x)$ is a convex function on $M$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.25</span></p>

Let $f_1, \ldots, f_k$ be convex functions on a convex set $M \subseteq \mathbb{R}^n$. Then $\max_{i=1,\ldots,k} f_i(x)$ is a convex function on $M$.

</div>

As an example, a piecewise linear function of type $\max(a_1^T x + b_1, \ldots, a_k^T x + b_k)$ is convex. The statement can be extended to the maximum of infinitely many convex functions. We just need to ensure that the pointwise maxima exist.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.26</span></p>

Let $f \colon \mathbb{R}^m \to \mathbb{R}$ be a convex function, $A \in \mathbb{R}^{m \times n}$ and $b \in \mathbb{R}^m$. Then $g(y) = f(Ay + b)$ is a convex function on $\mathbb{R}^n$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.27</span></p>

1. Function $e^x$ is convex, so $e^{x_1 - 2x_2 + 3}$ is convex.
2. Function $x^2$ is convex, so $(2x_1 - x_2 + 5)^2$ is convex.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.28</span></p>

Let $f, g \colon \mathbb{R} \to \mathbb{R}$.

1. If $f(x), g(x)$ are both convex, nonnegative and nondecreasing (or both nonincreasing), then $f(x) \cdot g(x)$ is convex.
2. If $f(x)$ is convex, nonnegative and nondecreasing, and $g(x)$ is concave, positive and nonincreasing, then $f(x)/g(x)$ is convex.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.29</span></p>

All three functions $f_1(x) = x$, $f_2(x) = x^2$ and $g(x) = e^x$ are convex, nonnegative and nondecreasing on $M = \lbrace x \ge 0 \rbrace$. Hence the function products $f_1(x)g(x) = xe^x$ and $f_2(x)g(x) = x^2 e^x$ are convex on the set $M$. However, on the whole real line, none of the products is convex.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.30</span></p>

Both functions $f(x) = x$ and $g(y) = y$ are convex, but their product $h(x,y) = xy$ is not convex even when restricted to domain $(x,y) \in [0, \infty)^2$; it is strictly concave on the segment between points $(1,0)$ and $(0,1)$. Therefore, in order to apply Theorem 3.28, we need that both functions are of the same variable. Notice that function $f(x) \cdot g(x) = x^2$ is convex now.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.31</span></p>

Let $f \colon \mathbb{R}^n \to \mathbb{R}^k$ and $g \colon \mathbb{R}^k \to \mathbb{R}$.

1. If $f_i(x)$ is convex for each $i = 1, \ldots, k$ and $g(y)$ is convex and nondecreasing in each coordinate, then $(g \circ f)(x) = g(f(x))$ is convex.
2. If $f_i(x)$ is concave for each $i = 1, \ldots, k$ and $g(y)$ is convex and nonincreasing in each coordinate, then $(g \circ f)(x) = g(f(x))$ is convex.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.32</span></p>

1. If $f \colon \mathbb{R}^n \to \mathbb{R}$ is convex, then $e^{f(x)}$ is convex. For example, $e^{e^x}$, $e^{x^2 - x}$, $e^{x_1 - x_2}$, $\ldots$
2. If $f(x) \ge 0$ and convex, then $f(x)^p$ is convex for every $p \ge 1$.
3. If $f(x) \ge 0$ and concave, then $-\log(f(x))$ is convex.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.33</span></p>

The monotonicity assumption in Theorem 3.31 is necessary, indeed. For example, functions $f(x) = x^2 - 1$ and $g(y) = y^2$ are convex, but $g(f(x)) = (x^2 - 1)^2$ is not convex.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3.34</span></p>

Checking convexity of a function is a hard problem in general. Ahmadi et al. (2013) proved that it is an NP-hard problem for a class of multivariate polynomials of degree at most 4 (i.e., the sum of degrees in each term is at most 4). For a "general" function, it is still an open problem whether convexity testing is decidable.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.35</span></p>

Consider the function $f(x_1, x_2) = x_1^2 x_2^2$. How to check if it is convex? Let us try the second order characterization. The Hessian of $f(x)$ reads

$$\nabla^2 f(x) = \begin{pmatrix} 2x_2^2 & 4x_1 x_2 \\ 4x_1 x_2 & 2x_1^2 \end{pmatrix}.$$

It is not a positive semidefinite matrix for every $x \in \mathbb{R}^2$ since $\det(\nabla^2 f(x)) = -12x_1^2 x_2^2 \le 0$. For example on the line segment between points $(1,0)^T$ and $(0,1)^T$ the shape is more concave-like. Summary: $f(x)$ is not convex on the whole space $\mathbb{R}^2$.

</div>

# Chapter 4: Convex Optimization

The problem of *convex optimization* reads

$$\min \; f(x) \quad \text{subject to} \quad x \in M,$$

where $f \colon \mathbb{R}^n \to \mathbb{R}$ is a convex function and $M \subseteq \mathbb{R}^n$ is a convex set. Often the feasible set $M$ is described in the form as follows

$$M = \lbrace x \in \mathbb{R}^n;\; g_j(x) \le 0, \; j = 1, \ldots, J \rbrace,$$

where $g_j(x) \colon \mathbb{R}^n \to \mathbb{R}$, $j = 1, \ldots, J$, are convex functions. By Theorem 3.16 the set $M$ is convex then. In this chapter, however, we will deal with a general convex set $M$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.1</span></p>

An example of a convex optimization problem:

$$\min \; x_1 + x_2 \quad \text{subject to} \quad x_1^2 + x_2^2 \le 2.$$

Another example:

$$\min \; x_1^2 + x_2^2 + 2x_2 \quad \text{subject to} \quad x_1^2 + x_2^2 \le 2.$$

</div>

## 4.1 Basic Properties

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.2</span><span class="math-callout__name">(Fenchel, 1951)</span></p>

For a convex optimization problem we have:

1. Each local minimum is a global minimum.
2. The optimal solution set is convex.
3. If $f(x)$ is a strictly convex function, then the minimum is either unique or none.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**(1)** Let $x^0 \in M$ be a local minimum and suppose to the contrary that there is $x^* \in M$ such that $f(x^*) < f(x^0)$. Consider the convex combination $x = \lambda x^* + (1-\lambda)x^0 \in M$, $\lambda \in (0,1)$. Then

$$f(x) \le \lambda f(x^*) + (1-\lambda) f(x^0) < \lambda f(x^0) + (1-\lambda)f(x^0) = f(x^0).$$

This is in contradiction with local minimality of $x^0$ since for arbitrarily small $\lambda > 0$ we have $f(x) < f(x^0)$.

**(2)** Let $x_1, x_2 \in M$ be two optimal solutions and denote by $z = f(x_1) = f(x_2)$ the optimal value. The convex combination $x = \lambda_1 x_1 + \lambda_2 x_2 \in M$ then satisfies

$$f(x) \le \lambda_1 f(x_1) + \lambda_2 f(x_2) = \lambda_1 z + \lambda_2 z = z,$$

that is, $x$ is also an optimal solution.

**(3)** Suppose to the contrary that $x_1, x_2 \in M$, $x_1 \ne x_2$, are two optimal solutions. Denote by $z = f(x_1) = f(x_2)$ the optimal value. The convex combination $x = \lambda_1 x_1 + \lambda_2 x_2 \in M$, $\lambda_1, \lambda_2 > 0$, then satisfies

$$f(x) < \lambda_1 f(x_1) + \lambda_2 f(x_2) = \lambda_1 z + \lambda_2 z = z,$$

that is, $x$ is better than the optimal solution; a contradiction.

</details>
</div>

Notice that a convex optimization problem need not possess an optimal solution. Consider, for example, $\min_{x \in \mathbb{R}} e^x$. This situation may happen even if the feasible set is compact:

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.3</span></p>

Consider the function $f \colon [1,2] \to \mathbb{R}$ defined by

$$f(x) = \begin{cases} x & \text{if } 1 < x \le 2, \\ 2 & \text{if } x = 1. \end{cases}$$

This function is convex, but not continuous, and the minimum on $[1,2]$ is not attained.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.4</span></p>

Let $\emptyset \ne M \subseteq \mathbb{R}^n$ be an open convex set and $f \colon M \to \mathbb{R}$ a convex differentiable function on $M$. Then $x^* \in M$ is an optimal solution if and only if $\nabla f(x^*) = o$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

"$\Rightarrow$" Let $x^*$ be an optimal solution. Then it is a local minimum, too, and according to Theorem 2.1 we have $\nabla f(x^*) = o$.

"$\Leftarrow$" Let $\nabla f(x^*) = o$. By Theorem 3.18 we have $f(x) - f(x^*) \ge \nabla f(x^*)^T(x - x^*) = 0$ for any $x \in M$. Therefore $f(x) \ge f(x^*)$ and $x^*$ is an optimal solution.

</details>
</div>

We cannot remove the assumption that $M$ is open. For instance, for the problem $\min_{x \in [1,2]} x$ we have $M = [1,2]$ convex and the objective function $f(x) = x$ is differentiable on $\mathbb{R}$, but its derivative at the optimal point $x^* = 1$ is $f'(1) = 1$.

We can generalize the theorem as follows.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.5</span></p>

Let $\emptyset \ne M \subseteq \mathbb{R}^n$ be a convex set and $f \colon M' \to \mathbb{R}$ a convex function differentiable on an open set $M' \supseteq M$. Then $x^* \in M$ is an optimal solution if and only if $\nabla f(x^*)^T(y - x^*) \ge 0$ for every $y \in M$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

"$\Rightarrow$" Suppose to the contrary that there is $y \in M$ such that $\nabla f(x^*)^T(y - x^*) < 0$. Consider the convex combination $x_\lambda = \lambda y + (1-\lambda)x^* = x^* + \lambda(y - x^*) \in M$. Then

$$0 > \nabla f(x^*)^T(y - x^*) = \lim_{\lambda \to 0^+} \frac{f(x^* + \lambda(y-x^*)) - f(x^*)}{\lambda} = \lim_{\lambda \to 0^+} \frac{f(x_\lambda) - f(x^*)}{\lambda}.$$

Hence $f(x_\lambda) < f(x^*)$ for a sufficiently small $\lambda > 0$; a contradiction.

"$\Leftarrow$" By Theorem 3.18, for every $y \in M$ we have $f(y) - f(x^*) \ge \nabla f(x^*)^T(y - x^*) \ge 0$. Therefore $f(y) \ge f(x^*)$, and $x^*$ is an optimal solution.

</details>
</div>

The condition from Theorem 4.5 is particularly satisfied if $\nabla f(x^*) = o$. This means that each stationary point is a global minimum.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.6</span></p>

The first problem in Example 4.1 reads

$$\min \; x_1 + x_2 \quad \text{subject to} \quad x_1^2 + x_2^2 \le 2.$$

Obviously, the optimum is $x^* = (-1,-1)^T$. We can verify it by means of Theorem 4.5. First, compute $\nabla f(x^*) = (1,1)^T$. Now, we have to show that for each feasible $y$ we have

$$\nabla f(x^*)^T(y - x^*) = (1,1) \begin{pmatrix} y_1 + 1 \\ y_2 + 1 \end{pmatrix} \ge 0,$$

or $y_1 + y_2 \ge -2$. This is clearly true.

The second problem in Example 4.1 reads

$$\min \; x_1^2 + x_2^2 + 2x_2 \quad \text{subject to} \quad x_1^2 + x_2^2 \le 2.$$

We compute $\nabla f(x^*) = (2x_1, 2x_2 + 2)^T$, and this gradient is zero at point $x^\star = (0, -1)$. Since this point satisfies the constraint, it is the optimum.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.7</span><span class="math-callout__name">(Rating System)</span></p>

Many methods have been developed to provide ratings of sport teams or other entities. Consider $n$ teams that we want to rate by numbers $r_1, \ldots, r_n \in \mathbb{R}^n$. Let $A \in \mathbb{R}^{n \times n}$ be a known scoring matrix, where $a_{ij}$ gives the scoring of team $i$ against team $j$. This matrix is skew symmetric, that is $A = -A^T$, since $a_{ii} = 0$ and $a_{ij} = -a_{ji}$. The rating vector $r = (r_1, \ldots, r_n)^T$ should reflect the scorings, so ideally we have $a_{ij} = r_i - r_j$, or in matrix form $A = re^T - er^T$. This is hardly satisfied in practice, but we aim to find the best approximation, which leads to an optimization formulation

$$\min_{x \in \mathbb{R}^n} f(x) = \lVert A - (xe^T - ex^T) \rVert^2.$$

We choose the Frobenius matrix norm, defined for $M \in \mathbb{R}^{n \times n}$ as $\lVert M \rVert = \sqrt{\sum_{i,j} m_{ij}^2} = \sqrt{\text{tr}(M^T M)}$; that is why we minimize the square of the norm. The objective function then reads

$$f(x) = \text{tr}(A^T A) - 4x^T Ae + 2n(x^T x) - 2(e^T x)^2.$$

The gradient and the Hessian read

$$\nabla f(x) = -4Ae + 4nx - 4ee^T x, \quad \nabla^2 f(x) = 4(nI_n - ee^T).$$

Since the Hessian is positive semidefinite, function $f(x)$ is convex. The optimality condition $\nabla f(x) = 0$ yields the system of linear equations

$$(nI_n - ee^T)x = Ae.$$

The matrix has rank $n-1$ and so the solution set is the line $x = \frac{1}{n}Ae + \alpha e$, $\alpha \in \mathbb{R}$. Function $f(x)$ is constant on this line, so the whole line is the optimal solution set. In practice, we usually normalize the rating vector such that $e^T r = 0$. Since $e^T Ae = 0$, we obtain the resulting formula for the rating vector $r = \frac{1}{n}Ae$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.8</span></p>

Consider the problem $\min_{x \ge o} x^T Cx + d^T x$, where

$$C = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}, \quad d = \begin{pmatrix} 2 \\ -8 \end{pmatrix}.$$

If an optimal solution exists, then it lies in the interior or on the border of the feasible set. Thus we analyse particular cases:

1. **Case $x^* > o$:** By Theorem 4.5, it must hold $\nabla f(x^*) = o$. In our case it takes the form of $2Cx^* + d = 0$, and this system has the unique solution $x^0 = (-2,3)^T$. Nevertheless, it contradicts the condition $x^* > o$.
2. **Case $x_1^* > 0$, $x_2^* = 0$:** Here we have $0 = \nabla f(x^*)_1$, which leads to $0 = 4x_1^* + 2x_2^* + 2 = 4x_1^* + 2$. It has no solution such that $x_1^* > 0$.
3. **Case $x_1^* = 0$, $x_2^* > 0$:** Here we have $0 = \nabla f(x^*)_2$, which leads to $0 = 2x_1^* + 4x_2^* - 8 = 4x_2^* - 8$. In this case $x^* = (0,2)^T$.
4. **Case $x^* = o$:** Here we have $o \le \nabla f(x^*) = 2Cx^* + d = d$, which is not satisfied.

Therefore, the optimal solution is unique, the point $x^* = (0,2)^T$, and the optimal value is $-8$.

Let us have a look on a more general convex optimization problem $\min_{x \ge o} f(x)$. By Theorem 4.5, an optimal solution $x^*$ has to satisfy $\nabla f(x^*)^T(y - x^*) \ge 0$ for every $y \ge o$. To avoid unboundedness of the problem from below, we must have $\nabla f(x^*) \ge o$. The function $\nabla f(x^*)^T(y - x^*)$ has the minimal value for $y = o$, which leads to the constraint $\nabla f(x^*)^T(-x^*) \ge 0$. Since $\nabla f(x^*) \ge o$ and $x^* \ge o$, we have $\nabla f(x^*)^T x^* = 0$. This condition is called a **complementarity condition** since for every $i$ we have $\nabla f(x^*)_i = 0$ or $x_i^* = 0$. In total, we obtain the following optimality conditions: $x^* \ge o$, $\nabla f(x^*) \ge o$, $\nabla f(x^*)^T x^* = 0$ (cf. Chapter 5).

</div>

## 4.2 Quadratic Programming

A *quadratic programming* problem reads

$$\min \; x^T Cx + d^T x \quad \text{subject to} \quad x \in M,$$

where $C \in \mathbb{R}^{n \times n}$ is symmetric, $d \in \mathbb{R}^n$ and $M \subseteq \mathbb{R}^n$ is a convex polyhedral set. If matrix $C$ is positive semidefinite, then it is a convex problem, called a *convex quadratic program*.

Convex quadratic programs are effectively solvable in polynomial time. If $C$ is not positive semidefinite, then the problem is NP-hard, even finding a local minimum is NP-hard. It is interesting that NP-hardness remains valid even for the subclass of problems defined by matrix $C$ having exactly one eigenvalue negative.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.9</span></p>

The problem $\max_{x \in M} x^T Cx$ is NP-hard even when $C$ is positive definite.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We will construct a reduction from the NP-complete problem SET-PARTITIONING: Given a set of numbers $\alpha_1, \ldots, \alpha_n \in \mathbb{N}$, can we group them into two subsets such that the sums of the numbers in both subsets are the same? Equivalently, is there $x \in \lbrace \pm 1 \rbrace^n$ such that $\sum_{i=1}^n \alpha_i x_i = 0$? This problem can be formulated as follows

$$\max \sum_{i=1}^n x_i^2 \quad \text{subject to} \quad \sum_{i=1}^n \alpha_i x_i = 0, \; x \in [-1, 1]^n.$$

The optimal value of this problem is $n$ if and only if SET-PARTITIONING is solvable. This optimization problem follows the template since the constraints are linear and the objective function has the form of $x^T Cx + d^T x$ for $C = I_n$ and $d = o$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.10</span><span class="math-callout__name">(Portfolio Selection Problem)</span></p>

This is a textbook example of an application of convex quadratic programming. The pioneer in this area was Harry Markowitz, a Nobel Prize winner in Economics in 1990, awarding his results from 1952.

The problem is formulated as follows: capital $K$ is to be invested in $n$ investments. The return of investment $i$ is $c_i$. The mathematical formulation of the portfolio selection problem is as a linear program

$$\max \; c^T x \quad \text{subject to} \quad e^T x = K, \; x \ge o.$$

The returns of investments are usually not known exactly and they are modelled as random quantities. Suppose that the vector $c$ is random, its expected value is $\tilde{c} := \text{E}\,c$ and the covariance matrix is $\Sigma := \text{cov}\,c = \text{E}\,(c - \tilde{c})(c - \tilde{c})^T$, which is positive semidefinite. For a real vector $x \in \mathbb{R}^n$, the expected value of the objective function value $c^T x$ is $\text{E}(c^T x) = \tilde{c}^T x$, and the variance of $c^T x$ is $\text{var}(c^T x) = x^T \Sigma x$.

Maximizing the expected value of the reward leads to the linear programming problem

$$\max \; \tilde{c}^T x \quad \text{subject to} \quad e^T x = K, \; x \ge o.$$

Taking into account the risks of investments, we model the problem as a convex quadratic program

$$\max \; \tilde{c}^T x - \gamma x^T \Sigma x \quad \text{subject to} \quad e^T x = K, \; x \ge o,$$

where $\gamma > 0$ is the so called risk aversion coefficient.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.11</span><span class="math-callout__name">(Quadrocopter Trajectory Planning)</span></p>

We need to plan a trajectory for a quadrocopter fleet such that a collision is avoided and the fleet is transferred from an initial state to a terminal state with minimum effort. In our model, time is discretized into time slots of length $h$. The variables are the position $p_i(k)$, velocity $v_i(k)$ and acceleration $a_i(k)$ for quadrocopter $i$ in time step $k$. The constraints are:

- physical constraints: the relations between velocity and acceleration, position and velocity, $\ldots$ (e.g., $v_i(k) = v_i(k-1) + h \cdot a_i(k-1)$, $p_i(k) = p_i(k-1) + h \cdot v_i(k-1)$, $\ldots$)
- restrictions on the maximum velocity, acceleration and jerk (i.e., the derivative of acceleration),
- the initial and terminal state (positions etc.),
- the collision avoidance constraint is nonlinear ($\lVert p_i(k) - p_j(k) \rVert_2 \ge r$ $\forall i \ne j$), so we have to linearize it.

The objective function is given by the sum of norms of accelerations in particular time steps ($\sum_{i,k} \lVert a_i(k) + g \rVert_2^2$).

</div>

## 4.3 Geometric Programming

A *geometric programming* problem reads

$$\min \sum_{j=1}^k c_j x_1^{\gamma_{j1}} \cdots x_n^{\gamma_{jn}}$$

$$\text{subject to} \quad \sum_{j=1}^{k_i} a_{ij} x_1^{\alpha_{ij1}} \cdots x_n^{\alpha_{ijn}} \le 1, \quad i = 1, \ldots, m,$$

$$b_i x_1^{\beta_{i1}} \cdots x_n^{\beta_{in}} = 1, \quad i = 1, \ldots, \ell,$$

$$x > o,$$

where $c_j, a_{ij}, b_i > 0$.

In this form, it is not a convex problem since for example the problem $\min \; x^{1/2}$ subject to $xy = 1$, $x, y > 0$ has non-convex both the objective function and the feasible set.

Nevertheless, the problem can be formulated to a convex one by using the following transformation. Substitute $y_i := \log(x_i)$, $\varphi_j := \log(c_j)$, $\xi_{ij} := \log(a_{ij})$, $\eta_i := \log(b_i)$. Then

$$c_j x_1^{\gamma_{j1}} \cdots x_n^{\gamma_{jn}} = e^{\varphi_j} e^{\gamma_{j1} y_1} \cdots e^{\gamma_{jn} y_n} = e^{\gamma_j^T y + \varphi_j}$$

and the problem can be formulated as

$$\min \sum_{j=1}^k e^{\gamma_j^T y + \varphi_j}$$

$$\text{subject to} \quad \sum_{j=1}^{k_i} e^{\alpha_{ij}^T y + \xi_{ij}} \le 1, \quad i = 1, \ldots, m,$$

$$\beta_i^T y + \eta_i = 0, \quad i = 1, \ldots, \ell,$$

$$y \in \mathbb{R}^n.$$

The logarithm of the objective function and both sides of the inequalities leads to

$$\min \; \log\!\left(\sum_{j=1}^k e^{\gamma_j^T y + \varphi_j}\right)$$

$$\text{subject to} \quad \log\!\left(\sum_{j=1}^{k_i} e^{\alpha_{ij}^T y + \xi_{ij}}\right) \le 0, \quad i = 1, \ldots, m,$$

$$\beta_i^T y + \eta_i = 0, \quad i = 1, \ldots, \ell,$$

$$y \in \mathbb{R}^n,$$

which is a convex problem. Geometric programming appears e.g. in engineering problems of structural mechanics to design a beam. Notice that geometric programs are polynomially solvable by using interior-point methods.

## 4.4 Convex Cone Programming

This section comes mainly from Ben-Tal and Nemirovski (2001). The motivation is as follows. The linear programming problem

$$\min \; c^T x \quad \text{subject to} \quad Ax \ge b$$

can be generalized in several ways. In Section 4.2 we replaced linear functions with quadratic ones. Another way of a generalization is to generalize the relation "$\ge$".

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.12</span><span class="math-callout__name">(Convex Cone)</span></p>

A set $\emptyset \ne \mathcal{K} \subseteq \mathbb{R}^n$ is a *convex cone* if two conditions are satisfied:

1. for every $\alpha \ge 0$ and $x \in \mathcal{K}$ we have $\alpha x \in \mathcal{K}$,
2. for every $x, y \in \mathcal{K}$ we have $x + y \in \mathcal{K}$.

A cone is called *pointed* if it contains no complete line.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.13</span></p>

If $\mathcal{K}$ is a pointed convex cone, then it induces

1. a partial order by definition $x \ge_\mathcal{K} y \;\Leftrightarrow\; x - y \in \mathcal{K}$,
2. a strict partial order by definition $x >_\mathcal{K} y \;\Leftrightarrow\; x - y \in \text{int}\,\mathcal{K}$.

</div>

From now on we consider only a pointed convex closed cone $\mathcal{K}$ with nonempty interior.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.14</span><span class="math-callout__name">(Examples of Cones)</span></p>

The frequently used cones are:

- *The nonnegative orthant* $\mathbb{R}_+^n = \lbrace x \in \mathbb{R}^n;\; x \ge 0 \rbrace$. The corresponding partial order is the standard entrywise inequality $\ge$ for vectors.
- *Lorentz cone (ice cream cone)* $\mathcal{L} = \lbrace x \in \mathbb{R}^n;\; x_n \ge \sqrt{\sum_{i=1}^{n-1} x_i^2} \rbrace = \lbrace x \in \mathbb{R}^n;\; x_n \ge \lVert (x_1, \ldots, x_{n-1}) \rVert_2 \rbrace$.
- *Generalized Lorentz cone* $\mathcal{L} = \lbrace x \in \mathbb{R}^n;\; x_n \ge \lVert (x_1, \ldots, x_{n-1}) \rVert \rbrace$, where $\lVert \cdot \rVert$ is an arbitrary norm.
- *Convex polyhedral cone* is characterized by the system $Ax \le 0$. This category involves, for example, the nonnegative orthant or the generalized Lorentz cone with the Manhattan or maximum norm.
- *The cone of positive semidefinite matrices.*

</div>

Now we are ready to introduce cone programming. The **cone programming problem** reads

$$\min \; c^T x \quad \text{subject to} \quad Ax \ge_\mathcal{K} b. \tag{4.1}$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.15</span><span class="math-callout__name">(Examples of Cone Programs)</span></p>

- For $\mathcal{K} = \mathbb{R}_+^n$ we get the standard linear programming.
- Employing the Lorentz cone, we have a more interesting example

  $$\min \; c^T x \quad \text{subject to} \quad \lVert Bx - a \rVert_2 \le d^T x + f. \tag{4.2}$$

  This problem is not easily transformable to a problem with convex quadratic constraints since the squaring of both sides yields quadratic constraints which need not be convex (convexity of the constraint function was destroyed by the squaring).

- The cone constraints can be combined, so we can consider also problems such as

  $$\min \; c^T x \quad \text{subject to} \quad Ax \ge b, \; \lVert Bx - a \rVert_2 \le d^T x + f.$$

  The reason is that the Cartesian product of cones is again a cone. This problem belongs to a called second order cone programming.

- The cone of positive semidefinite matrices leads to the problems of type

  $$\min \; c^T x \quad \text{subject to} \quad \sum_{k=1}^n x_k A^{(k)} - B \text{ is positive semidefinite},$$

  where $A^{(1)}, \ldots, A^{(n)}, B$ are symmetric matrices. Such problems are called semidefinite programs.

</div>

### 4.4.1 Duality in Convex Cone Programming

**Motivation.** Recall the derivation of duality of a linear program $\min\lbrace c^T x;\; Ax \ge b \rbrace$: Let $x$ be a feasible solution. Then for every $y \ge 0$ we have $y^T Ax \ge y^T b$. If $y$ in addition satisfies $y^T A = c^T$, then we get $c^T x = y^T Ax \ge y^T b$. In other words, $y^T b$ is a lower bound on the optimal value for every $y \ge 0$ such that $A^T y = c$. This leads to the dual problem formulation and weak duality

$$\min\lbrace c^T x;\; Ax \ge b \rbrace \ge \max\lbrace b^T y;\; A^T y = c, \; y \ge 0 \rbrace.$$

Now the question is, in the case of convex cone programming (4.1), which relation should replace $y \ge 0$? We are interested in such $y$, for which we have $y^T a \ge 0$ for every $a \ge_\mathcal{K} 0$. Obviously, the set of such $y$s forms a cone — this cone is called the dual cone of $\mathcal{K}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.16</span><span class="math-callout__name">(Dual Cone)</span></p>

Let $\mathcal{K} \subseteq \mathbb{R}^n$ be a cone. Then its *dual cone* is the cone

$$\mathcal{K}^* = \lbrace y \in \mathbb{R}^n;\; y^T a \ge 0 \;\forall a \in \mathcal{K} \rbrace.$$

</div>

By using the dual cone, we formulate the dual problem to (4.1) as follows

$$\max \; b^T y \quad \text{subject to} \quad A^T y = c, \; y \ge_{\mathcal{K}^*} 0. \tag{4.3}$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.17</span><span class="math-callout__name">(Weak Duality)</span></p>

We have: $\min\lbrace c^T x;\; Ax \ge_\mathcal{K} b \rbrace \ge \max\lbrace b^T y;\; A^T y = c, \; y \ge_{\mathcal{K}^*} 0 \rbrace$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For every $y \ge_{\mathcal{K}^*} 0$ such that $A^T y = c$ and for every $x$ such that $Ax \ge_\mathcal{K} b$ we have

$$c^T x = y^T Ax \ge y^T b.$$

In other words, the objective value of each feasible solution is an upper bound on every objective value of the dual problem. Therefore the inequality holds true even for the extremal values.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.18</span></p>

- Nonnegative orthant is self-dual, that is, $(\mathbb{R}_+^n)^* = \mathbb{R}_+^n$.
- The Lorentz cone is self-dual as well, $\mathcal{L}^* = \mathcal{L}$.
- The cone of positive semidefinite matrices is also self-dual; herein, the scalar product of positive semidefinite matrices $A, B$ is defined by $\langle A, B \rangle := \text{tr}(AB) = \sum_{i,j} a_{ij} b_{ij}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.19</span></p>

We have:

1. $\mathcal{K}^*$ is a closed convex cone.
2. If $\mathcal{K}$ is a closed convex cone, then $(\mathcal{K}^*)^* = \mathcal{K}$.
3. If $\mathcal{K}_1, \mathcal{K}_2$ are cones, then $\mathcal{K}_1 \times \mathcal{K}_2$ is a cone and $(\mathcal{K}_1 \times \mathcal{K}_2)^* = \mathcal{K}_1^* \times \mathcal{K}_2^*$.
4. If $\mathcal{K}_1 \subseteq \mathcal{K}_2$ are cones, then $\mathcal{K}_1^* \supseteq \mathcal{K}_2^*$.

</div>

Based on the above properties, we can see that a convex cone program can also have the form of

$$\min \; c^T x \quad \text{subject to} \quad Ax \ge b, \; Bx \ge_\mathcal{K} d,$$

the dual problem of which is

$$\max \; b^T y + d^T z \quad \text{subject to} \quad A^T y + B^T z = c, \; y \ge 0, \; z \ge_{\mathcal{K}^*} 0.$$

Can we state strong duality in convex cone programming? In general not, but under mild assumptions the strong duality holds.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.20</span><span class="math-callout__name">(Strong Duality)</span></p>

The primal and dual optimal values are the same provided at least one of the following conditions holds

1. the primal problem is strictly feasible, that is, there is $x$ such that $Ax >_\mathcal{K} b$,
2. the dual problem is strictly feasible, that is, there is $y >_{\mathcal{K}^*} 0$ such that $A^T y = c$.

</div>

Notice that even when strong duality holds and both primal and dual optimal values are (the same and) finite, it may happen that the optimal value is not attained (formally, we should write "inf" instead of "min"). The following example illustrates this situation.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.21</span></p>

Consider the convex cone program of form (4.2)

$$\min \; x_1 \quad \text{subject to} \quad \sqrt{(x_1 - x_2)^2 + 1} \le x_1 + x_2.$$

By squaring both sides of the inequality we get

$$\min \; x_1 \quad \text{subject to} \quad 4x_1 x_2 \le 1, \; x_1 + x_2 > 0.$$

Even though the problem is strictly feasible, the optimal value is not attained.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.22</span></p>

Consider the second order cone program

$$\min \; x_2 \quad \text{subject to} \quad \sqrt{x_1^2 + x_2^2} \le x_1.$$

We express it equivalently as

$$\min \; x_2 \quad \text{subject to} \quad x_2 = 0, \; x_1 \ge 0.$$

We can see that the optimal value is 0 and each feasible solution is optimal, that is, the optimal solution set consists of the nonnegative part of the first axis.

To construct the dual problem, we rewrite the primal program into the canonical form

$$\min \; x_2 \quad \text{subject to} \quad (x_1, x_2, x_1)^T \ge_\mathcal{L} 0.$$

The dual problem then reads

$$\max \; 0 \quad \text{subject to} \quad y_1 + y_3 = 0, \; y_2 = 1, \; y \ge_\mathcal{L} 0.$$

The inequality $y \ge_\mathcal{L} 0$ takes the form of $y_3 \ge \sqrt{y_1^2 + y_2^2}$, which together with $y_1 + y_3 = 0$ leads to $y_2 = 1$; a contradiction. Hence the dual problem is infeasible, even though the primal problem has a finite optimal value.

</div>

### 4.4.2 Second Order Cone Programming

Second order cone programming deals with convex cone programs with linear constraints and constraints corresponding to the Lorentz cone. For the sake of simplicity we employ just one Lorentz cone, and so the problem reads

$$\min \; c^T x \quad \text{subject to} \quad Ax \ge b, \; Bx \ge_\mathcal{L} d. \tag{4.4}$$

We express

$$(B \mid d) = \begin{pmatrix} D & f \\ p^T & q \end{pmatrix}$$

so the condition $Bx \ge_\mathcal{L} d$ takes the form of $\lVert Dx - f \rVert_2 \le p^T x - q$. Thus we have an explicit description of problem (4.4)

$$\min \; c^T x \quad \text{subject to} \quad Ax \ge b, \; \lVert Dx - f \rVert_2 \le p^T x - q. \tag{4.5}$$

A lot of functions and nonlinear conditions can be expressed in the form of (4.5).

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.23</span><span class="math-callout__name">(Examples of Second Order Cone Programs)</span></p>

- *Quadratic constraints.* For example, the condition $x^T x \le z$ can be expressed as $x^T x + \frac{1}{4}(z-1)^2 \le \frac{1}{4}(z+1)^2$, the square root of which gives $\lVert (x^T, \frac{1}{2}(z-1)) \rVert_2 \le \frac{1}{4}(z+1)$.
- *Hyperbola.* The condition $x \cdot y \ge 1$ on $y \ge 0$ can be expressed as $\frac{1}{4}(x+y)^2 \ge 1 + \frac{1}{4}(x-y)^2$, the square root of which (notice $y \ge 0$) gives $\lVert (1, \frac{1}{2}(x-y)) \rVert_2 \le \frac{1}{2}(x+y)$.

On the other hand, condition $e^x \le z$ is not a second order cone constraint.

The dual problem is

$$\max \; b^T y + d^T z \quad \text{subject to} \quad A^T y + B^T z = c, \; y \ge 0, \; z \ge_{\mathcal{L}^*} 0.$$

Letting $z = (u^T, v)^T$ we get

$$\max \; b^T y + d^T z \quad \text{subject to} \quad A^T y + B^T z = c, \; y \ge 0, \; v \ge \lVert u \rVert_2.$$

The dual is thus also a second order cone program.

</div>

### 4.4.3 Semidefinite Programming

Employing the cone of positive semidefinite matrices in the convex cone programming problem (4.1), we obtain the class of **semidefinite programming** problems

$$\min \; c^T x \quad \text{subject to} \quad \sum_{k=1}^n x_k A^{(k)} \succeq B, \tag{4.6}$$

where $c \in \mathbb{R}^n$, matrices $A^{(1)}, \ldots, A^{(n)}, B \in \mathbb{R}^{m \times m}$ are symmetric and the relation $A \succeq B$ means that $A - B$ is positive semidefinite. Semidefinite programming is a large class of efficiently solvable optimization problems. It is often used to approximate NP-hard problems; indeed, it provides one of the best known approximation factors.

How to construct the dual problem? According to (4.3), the dual problem has $m^2$ variables, so that they constitute a matrix of variables $Y \in \mathbb{R}^{m \times m}$. The dual objective function is $\sum_{i,j} b_{ij} y_{ij}$, the equations have the form of $\sum_{i,j} a_{ij}^{(k)} y_{ij} = c_k$, and the condition $Y \ge_{\mathcal{K}^*} 0$ takes the form $Y \succeq 0$. In total, the dual problem reads

$$\max \; \text{tr}(BY) \quad \text{subject to} \quad \text{tr}(A^{(k)} Y) = c_k, \; k = 1, \ldots, n, \; Y \succeq 0. \tag{4.7}$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.24</span><span class="math-callout__name">(Examples of Semidefinite Programs)</span></p>

- *Linear constraints.* Linear inequalities $Ax \le b$ are expressed as semidefinite conditions as follows:

  $$\begin{pmatrix} b_1 - A_{1*}x & 0 & \cdots & 0 \\ 0 & b_2 - A_{2*}x & \ddots & \vdots \\ \vdots & \ddots & \ddots & 0 \\ 0 & \cdots & 0 & b_m - A_{m*}x \end{pmatrix} \succeq 0.$$

- *Second order cone constraints.* They can be expressed as semidefinite constraints. Basically, it is sufficient to show it for the condition $\lVert x \rVert_2 \le z$; the others can be handled by a linear transformation. We have

  $$\lVert x \rVert_2 \le z \quad \Leftrightarrow \quad \begin{pmatrix} z \cdot I_n & x \\ x^T & z \end{pmatrix} \succeq 0. \tag{4.8}$$

- *Eigenvalues.* Many conditions on eigenvalues can be expressed as semidefinite programs. For instance, the largest eigenvalue $\lambda_{\max}$ of a symmetric matrix $A \in \mathbb{R}^{n \times n}$:

  $$\lambda_{\max} = \min \; z \quad \text{subject to} \quad z \cdot I_n \succeq A.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.25</span><span class="math-callout__name">(Portfolio Selection with Interval Estimation)</span></p>

Consider the portfolio selection problem (Example 4.10)

$$\max \; c^T x \quad \text{subject to} \quad e^T x = K, \; x \ge o,$$

where $c$ is a random vector with the expected value $\tilde{c} := \text{E}\,c$ and the covariance matrix $\Sigma := \text{cov}\,c = \text{E}\,(c - \tilde{c})(c - \tilde{c})^T$. Assume that a portfolio $\tilde{x}$ is chosen, but for the covariance matrix we know only an interval estimation $\Sigma_1 \le \Sigma \le \Sigma_2$. What is the risk of portfolio $\tilde{x}$? The risk is given by the variance of the reward $c^T \tilde{x}$, which is equal to $\tilde{x}^T \Sigma \tilde{x}$. Thus the largest variance is computed by a semidefinite program

$$\max \; \tilde{x}^T \Sigma \tilde{x} \quad \text{subject to} \quad \Sigma_1 \le \Sigma \le \Sigma_2, \; \Sigma \succeq 0.$$

The objective function is linear in variable $\Sigma$, and the constraints are easily transformed to the basic form (4.7) by means of Example 4.24.

</div>

## 4.5 Computational Complexity

In general, convex optimization problems are considered to be tractable. Indeed, under some assumptions, they are solvable in polynomial time. On the other hand, there are some intractable convex optimization problems, too.

### 4.5.1 Good News — The Ellipsoid Method

A convex optimization problem $\min_{x \in M} f(x)$ is solvable in polynomial time by the ellipsoid method under general assumptions. This result is, however, rather theoretical. To solve the problem practically, other methods, such as interior point methods, are usually more convenient.

The ellipsoid method is designed to find a feasible solution, but the same idea works to find an optimal solution as well. Thus we focus on the problem of finding a point $x \in M$ or determining that $M$ is empty. First, we construct a sufficiently large ellipsoid $\mathcal{E}$ covering the whole set $M$. Then we check if the center $c$ of ellipsoid $\mathcal{E}$ lies in $M$. If yes, we are done. If not, then we construct a hyperplane containing point $c$ and being disjoint to $M$ such that $M$ lies in halfspace $a^T x \le b$. Then we construct a smaller (minimum volume) ellipsoid covering the intersection $\mathcal{E} \cap \lbrace x;\; a^T x \le b \rbrace$. We repeat this process until we find a feasible point or prove $M = \emptyset$. The convergence is guaranteed by the fact that the size of the ellipsoids exponentially decreases.

In order that the above algorithm is correct and runs in polynomial time, we need to ensure certain conditions:

- The feasible set $M$ shouldn't be too flat or too large. There must exist "reasonably" large numbers $r, R > 0$ such that $M$ contains a ball of radius $r$ and also $M$ lies in the ball $\lbrace x;\; \lVert x \rVert_2 \le R \rbrace$.
- **Separation oracle.** For every $x^* \in \mathbb{R}^n$ we need to check for $x^* \in M$ in polynomial time. If $x^* \notin M$, then we need to find a vector $a \ne o$ such that $a^T x^* \ge \sup_{x \in M} a^T x$. This gives us a hyperplane $a^T x = a^T x^*$ satisfying $a^T x \le a^T x^*$ for every $x \in M$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.26</span></p>

In some cases, the ellipsoid method provides a polynomial algorithm for problems with exponentially many or even infinitely many constraints. For example, let $M$ be a unit ball described by the tangent hyperplanes, that is,

$$M = \lbrace x \in \mathbb{R}^n;\; a^T x \le 1, \; \forall a : \lVert a \rVert_2 = 1 \rbrace.$$

To check if a given point $x^* \in \mathbb{R}^n$ belongs to the set $M$, we do not need to process all the infinitely many inequalities. It is sufficient to check the possibly violated constraint, which is the case of $a = \frac{1}{\lVert x^* \rVert_2} x^*$.

</div>

### 4.5.2 Bad News — Copositive Programming

Not every convex optimization problem is tractable. Here we present a convex problem that is NP-hard. Denote by

$$\mathcal{C} := \lbrace A \in \mathbb{R}^{n \times n};\; A = A^T, \; x^T Ax \ge 0 \;\forall x \ge 0 \rbrace$$

the convex cone of **copositive matrices** and by

$$\mathcal{C}^* := \text{conv}\lbrace xx^T;\; x \ge 0 \rbrace$$

its dual cone of **completely positive matrices**. The set $\mathcal{C}$ covers both nonnegative symmetric matrices and positive semidefinite matrices, but it contains other matrices, too. Similarly the matrices in $\mathcal{C}^*$ are nonnegative positive semidefinite, but not each such matrix belongs to $\mathcal{C}^*$. Notice that even to decide if a given matrix is copositive is a co-NP-complete problem. Checking complete positivity of a matrix is NP-hard, but if the problem is in NP is not known yet.

A *copositive program* is an optimization problem, in which one of the constraints is the condition that a variable matrix is copositive; the other objective and constraint functions are linear.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.27</span><span class="math-callout__name">(de Klerk and Pasechnik, 2002)</span></p>

Let $G = (V, E)$ be a graph with $n$ vertices and let $\alpha$ denote the size of a maximum independent set in graph $G$. Let $A$ be the adjacency matrix of $G$. We have

$$\alpha = \min \; \beta \quad \text{subject to} \quad \beta(I_n + A) - ee^T \in \mathcal{C}. \tag{4.9}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Without loss of generality suppose that the maximum independent set consists of vertices $1, \ldots, \alpha$. Any vector $x \in \mathbb{R}^n$ can be decomposed accordingly as $x = (u^T, v^T)^T$, where $u \in \mathbb{R}^\alpha$ and $v \in \mathbb{R}^{n-\alpha}$, we will use the fact that the constraint of problem (4.9) equivalently reads as

$$\beta x^T(I_n + A)x \ge \lVert x \rVert_1^2 \quad \forall x \ge 0. \tag{4.10}$$

"$\le$" It is sufficient to show that matrix $\alpha(I_n + A) - ee^T$ is copositive. If $\alpha = n$, then the copositivity condition takes the form $n\lVert x \rVert_2^2 \ge \lVert x \rVert_1^2$; however, this is a well-known vector norm inequality. If $\alpha < n$, then without loss of generality we can assume (by a permutation of vertices) that $a_{12} = 1$. Consider the worst-case scenario with $A_{*1} = A_{*2} = e_1 + e_2$. For $\varepsilon > 0$ put $y = x - \varepsilon e_1 + \varepsilon e_2$. Then $\lVert x \rVert_1 = \lVert y \rVert_1$, so that the right-hand side of inequality (4.10) does not change when substituting $y$. The left-hand side takes for $\beta = \alpha$ the form

$$\alpha y^T(I_n + A)y = \alpha x^T(I_n + A)x.$$

Since $(I_n + A)(e_2 - e_1) = 0$, it simplifies to $\alpha y^T(I_n + A)y = \alpha x^T(I_n + A)x$. Hence the left-hand side of (4.10) does not change, too. Putting $\varepsilon := x_1$ we make the first entry of vector $x$ to be zero. Thus, we reduced the dimension and recursively proceed further.

"$\ge$" We want to show that for $\beta < \alpha$ the matrix $\beta(I_n + A) - ee^T$ is not copositive. Define $x = (u^T, v^T)^T$ such that $u = \frac{1}{\alpha}e$, $v = 0$. Then

$$\beta x^T(I_n + A)x < \alpha x^T(I_n + A)x = \alpha u^T u = 1 = \lVert x \rVert_1^2,$$

Therefore, $\beta(I_n + A) - ee^T \notin \mathcal{C}$ since it violates condition (4.10).

</details>
</div>

## 4.6 Applications

### 4.6.1 Robust PCA

Let $A \in \mathbb{R}^{m \times n}$ be a matrix representing certain data. The problem is to determine some essential information hidden in the data. For example, if the matrix represents a picture, then we may want to recognize some pattern (e.g., a face) or to perform some operations such as reconstruction of a damaged picture.

To this end the SVD decomposition of $A$ may serve well, however, for some purposes it is not sufficient. We will formulate the problem as the so called robust PCA (principal component analysis):

$\to$ Decompose $A = L + S$ such that $L$ has low rank and $S$ is sparse.

Then $L$ represents the fundamental information in the data and $S$ can be interpreted as a noise. This problem is rather vaguely defined and that is why we consider the (approximate) optimization problem formulation

$$\min \; \lVert L \rVert_* + \lVert S \rVert_{\ell_1} \quad \text{subject to} \quad A = L + S, \tag{4.11}$$

where $\lVert S \rVert_{\ell_1} := \sum_{i,j} \lvert s_{ij} \rvert$ is the entrywise sum norm and $\lVert L \rVert_* := \sum_i \sigma_i(L)$ is the nuclear norm defined as the sum of the singular values.

Notice that the nuclear norm is a good approximation of the matrix rank since it is the best convex underestimator of the rank on a unit ball. Similarly, the entrywise sum norm is a good approximation of matrix sparsity.

Problem (4.11) is a convex optimization problem since a norm is always convex. Hence the problem is effectively solvable.

**Foreground and background detection in a video.** The Robust PCA technique can effectively be used to recognize foreground and background in a video or a sequence of pictures. The columns of matrix $A$ represent the particular video frames. Then we can expect that matrix $L$ corresponds to the background since it is static and the matrix has low rank. In contrast, matrix $S$ captures the foreground then.

### 4.6.2 Minimum Volume Enclosing Ellipsoid

The aim is to find an ellipsoid with minimum volume and covering a given convex polyhedron. Let $x_1, \ldots, x_m \in \mathbb{R}^n$ be vertices of a convex polyhedron that we want to enclose by an ellipsoid. For simplicity we restrict to a full-dimensional ellipsoid centered in the origin. Such an ellipsoid is described by $x^T Hx \le 1$, where $H \in \mathbb{R}^{n \times n}$ is a positive definite matrix (in short we write $H \succ 0$). The volume of the ellipsoid is inversely proportional to $\det(H)$. Therefore the problem can be formulated as

$$\min \; -\det(H) \quad \text{subject to} \quad H \succ 0, \; x_i^T Hx_i \le 1, \; i = 1, \ldots, m,$$

where the unknown matrix $H$ is variable. This problem is not convex, so take the logarithm of the objective function

$$\min \; -\log \det(H) \quad \text{subject to} \quad H \succ 0, \; x_i^T Hx_i \le 1, \; i = 1, \ldots, m.$$

Function $-\log \det(H)$ is strictly convex on the set of positive definite matrices, so we have a convex optimization problem, which is efficiently solvable.

# Chapter 5: Karush–Kuhn–Tucker Optimality Conditions

In this chapter we consider the following optimization problem

$$\min \; f(x) \quad \text{subject to} \quad x \in M,$$

where $f \colon \mathbb{R}^n \to \mathbb{R}$ is a differentiable function and the feasible set $M \subseteq \mathbb{R}^n$ is described by the system

$$g_j(x) \le 0, \quad j = 1, \ldots, J, \qquad h_\ell(x) = 0, \quad \ell = 1, \ldots, L,$$

where $g_j(x), h_\ell(x) \colon \mathbb{R}^n \to \mathbb{R}$.

By Theorem 2.1 we know that in case $M = \mathbb{R}^n$ the necessary condition for optimality of $x \in \mathbb{R}^n$ is $\nabla f(x) = 0$. If $f(x)$ is convex, then the condition is also sufficient (Theorem 4.4).

This chapter generalizes the above condition to a constrained optimization problem, which results to the so called Karush–Kuhn–Tucker conditions. First we consider only equality constraints, and then we extend the results to the general form.

## Equality Constraints

Consider for a while an equality constrained problem

$$\min \; f(x) \quad \text{subject to} \quad h(x) = 0. \tag{5.1}$$

Let $x^*$ be a feasible point. When is $x^*$ optimal? First we discuss the case when the constraints are linear.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.1</span></p>

If $x^* \in \mathbb{R}^n$ is a local optimum of

$$\min \; f(x) \quad \text{subject to} \quad Ax = b,$$

then $\nabla f(x^*) \in \mathcal{R}(A)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The feasible set is the solution set of the system $Ax = b$, so it is an affine subspace $x^* + \text{Ker}(A)$. Let $B$ be a matrix such that its columns form a basis of $\text{Ker}(A)$. Then the feasible set can be expressed as $x = x^* + Bv$, $v \in \mathbb{R}^k$. Substituting for $x$ we obtain an unconstrained optimization problem

$$\min \; f(x^* + Bv) \quad \text{subject to} \quad v \in \mathbb{R}^k.$$

By Theorem 2.1, the necessary condition for local optimality of $v = 0$ is zero gradient, that is, $\nabla f(x^*)^T B = 0^T$. In other words, $\nabla f(x^*) \in \text{Ker}(A)^\perp = \mathcal{R}(A)$.

</details>
</div>

Now, the idea is based on linearization of possibly nonlinear functions $h_\ell$. The equation $h_\ell(x) = 0$ will be replaced by the tangent hyperplane of the corresponding manifold at point $x^*$:

$$\nabla h_\ell(x^*)^T(x - x^*) = 0,$$

so that the linearized constraints can be expressed as $A(x - x^*) = 0$. In order that $x^*$ is optimal, the objective function gradient $\nabla f(x^*)$ must be perpendicular to the intersection of the tangent hyperplanes; in other words, $\nabla f(x^*)$ must be a linear combination of the gradients $\nabla h_\ell(x^*)$ of the tangent hyperplanes. According to Proposition 5.1 we have $\nabla f(x^*) \in \mathcal{R}(A)$. This leads to the condition

$$\nabla f(x^*) + \sum_{\ell=1}^L \nabla h_\ell(x^*) \mu_\ell = 0.$$

As illustrated by a degenerate case where the intersection of the curves is a point but the intersection of the tangent lines is a line, this idea can be *wrong* since a degenerate situation may appear. Thus we need to avoid such a degenerate case. This can be achieved by the assumption on linear independence of gradients $\nabla h_\ell(x^*)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.2</span></p>

Let $\nabla h_\ell(x^*)$, $\ell = 1, \ldots, L$, be linearly independent. If $x^*$ is a local optimum, then there is $\mu \in \mathbb{R}^L$ such that

$$\nabla f(x^*) + \nabla h(x^*) \mu = 0.$$

</div>

Coefficients $\mu_1, \ldots, \mu_L$ are called **Lagrange multipliers**. The condition stated in the theorem is a necessary condition. This is convenient for us since we can restrict the feasible set to a much smaller set of candidates for optima — ideally the candidate is unique.

## Equality and Inequality Constraints

Now we consider the general case with both equality and inequality constraints. The **active set** of a feasible point $x$ is the set of those inequalities that are satisfied as equations:

$$I(x) = \lbrace j;\; g_j(x) = 0 \rbrace.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.3</span><span class="math-callout__name">(KKT Conditions)</span></p>

Let $\nabla h_\ell(x^*)$, $\ell = 1, \ldots, L$, $\nabla g_j(x^*)$, $j \in I(x^*)$, be linearly independent. If $x^*$ is a local optimum, then there exist $\lambda \in \mathbb{R}^J$, $\lambda \ge 0$, and $\mu \in \mathbb{R}^L$ such that

$$\nabla f(x^*) + \nabla h(x^*)\mu + \nabla g(x^*)\lambda = 0, \tag{5.2}$$

$$\lambda^T g(x^*) = 0. \tag{5.3}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Condition (5.3) is called **complementarity condition** since it says that for every $j = 1, \ldots, J$ we have $\lambda_j = 0$ or $g_j(x^*) = 0$. If $g_j(x^*) < 0$, then $\lambda_j = 0$ and hence variable $\lambda_j$ does not act in the KKT conditions; this corresponds to the situation that $x^*$ does not lie on the border of the set described by this constraint. Conversely, if $g_j(x^*) = 0$, then the complementarity makes no restriction on $\lambda_j$. In summary, the complementarity condition enforces to consider the Lagrange multipliers $\lambda_j$ for the active constraints only.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 5.3 (Main idea)</summary>

We linearize the problem such that the objective function and the constraint functions are replaced by their tangent hyperplanes at point $x^*$. This results in a linear programming problem

$$\min \; \nabla f(x^*)^T x \quad \text{subject to} \quad \nabla g_j(x^*)^T(x - x^*) \le 0, \; j \in I(x^*), \quad \nabla h_\ell(x^*)^T(x - x^*) = 0, \; \ell = 1, \ldots, L.$$

Due to the linear independence assumption, the solution $x^*$ remains optimal. The dual problem to the linear program is

$$\max \sum_{\ell=1}^L (\nabla h_\ell(x^*)^T x^*)\mu_\ell + \sum_{j \in I(x^*)} (\nabla g_j(x^*)^T x^*)\lambda_j \quad \text{subject to}$$

$$\nabla f(x^*) + \sum_{\ell=1}^L \nabla h_\ell(x^*)\mu_\ell + \sum_{j \in I(x^*)} \nabla g_j(x^*)\lambda_j = 0, \quad \lambda_j \ge 0, \; j \in I(x^*).$$

Since the primal problem has an optimum, the dual problem must be feasible. Hence there exist $\lambda \ge 0, \mu$ satisfying (5.2). Condition (5.3) is fulfilled since for $j \in I(x^*)$ we have $g_j(x^*) = 0$ by definition, and for $j \notin I(x^*)$ we can put $\lambda_j = 0$.

</details>
</div>

Conditions (5.2)–(5.3) are called **Karush–Kuhn–Tucker conditions** (Karush, 1939; Kuhn and Tucker, 1951), or KKT conditions in short.

Since the linear independence assumption is hard to check in general (notice that $x^*$ is unknown), alternative assumptions were derived, too. Usually, they are more easy to verify but on account of stronger assumptions. One commonly used assumption is **Slater's condition**

$$\exists x^0 \in M : g(x^0) < 0.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.4</span></p>

Consider the optimization problem

$$\min \; f(x) \quad \text{subject to} \quad g(x) \le 0, \; x \in M,$$

where $f(x), g_j(x)$ are convex functions and $M$ is a convex set. Suppose that Slater's condition is satisfied. If $x^*$ is an optimum of the above problem, then there exists $\lambda \ge 0$ such that $x^*$ is an optimum of the problem

$$\min \; f(x) + \lambda^T g(x) \quad \text{subject to} \quad x \in M, \tag{5.4}$$

and, moreover, $\lambda^T g(x^*) = 0$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Define the sets

$$\mathcal{A} := \lbrace (r,s) \in \mathbb{R}^J \times \mathbb{R};\; r \ge g(x), \; s \ge f(x), \; x \in M \rbrace,$$

$$\mathcal{B} := \lbrace (r,s) \in \mathbb{R}^J \times \mathbb{R};\; r \le 0, \; s \le f(x^*) \rbrace.$$

Both sets are convex, and their interiors are disjoint since otherwise there is a point $x \in M$ such that $g(x) < 0$ and $f(x) < f(x^*)$. Therefore a separating hyperplane exists having the form of $\lambda^T r + \lambda_0 s = c$, where $(\lambda, \lambda_0) \ne 0$. The separability implies:

$$\forall (r,s) \in \mathcal{A} : \lambda^T r + \lambda_0 s \ge c, \qquad \forall (r,s) \in \mathcal{B} : \lambda^T r + \lambda_0 s \le c.$$

Since $(0, f(x^*)) \in \mathcal{A} \cap \mathcal{B}$, this point lies on the hyperplane, and thus $c = \lambda_0 f(x^*)$. Analogously $(g(x^*), f(x^*)) \in \mathcal{A} \cap \mathcal{B}$, so this point also lies on the hyperplane, yielding

$$\lambda^T g(x^*) + \lambda_0 f(x^*) = c = \lambda_0 f(x^*),$$

which gives the complementarity constraint $\lambda^T g(x^*) = 0$.

For every $i$ we have $(-e_i, f(x^*)) \in \mathcal{B}$, so this point lies in the negative halfspace. This means that $-\lambda^T e_i + \lambda_0 f(x^*) \le c$, from which $\lambda_i \ge 0$. Therefore $\lambda \ge 0$. Analogously we deduce $\lambda_0 \ge 0$. Since $(o, f(x^*) - 1) \in \mathcal{B}$, so $\lambda^T o + \lambda_0(f(x^*) - 1) \le c$, and hence $\lambda_0 \ge 0$.

Since $g(x^0) < 0$, we have $(r, f(x^0)) \in \mathcal{A}$ for every $r$ in the neighbourhood of $0$. Hence the separating hyperplane cannot be vertical, which means $\lambda_0 \ne 0$. Without loss of generality we normalize it such that $\lambda_0 = 1$.

For every $x \in M$ we have $(g(x), f(x)) \in \mathcal{A}$, which fulfills

$$\lambda^T g(x) + f(x) \ge c = \lambda^T g(x^*) + f(x^*) = f(x^*).$$

This proves that $x^*$ is the optimum of (5.4).

</details>
</div>

Applying the optimality conditions from Theorem 2.1 to problem (5.4), we obtain the KKT conditions as a corollary:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 5.5</span></p>

Suppose that Slater's condition is satisfied for the convex optimization problem

$$\min \; f(x) \quad \text{subject to} \quad g(x) \le 0.$$

If $x^*$ is an optimum, then there exists $\lambda \ge 0$ such that the KKT conditions are satisfied, i.e.,

$$\nabla f(x^*) + \nabla g(x^*)\lambda = 0, \tag{5.5a}$$

$$\lambda^T g(x^*) = 0. \tag{5.5b}$$

</div>

We obtain also a general form involving equality constraints.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 5.6</span></p>

Suppose that Slater's condition is satisfied for the convex optimization problem

$$\min \; f(x) \quad \text{subject to} \quad g(x) \le 0, \; Ax = b.$$

If $x^*$ is an optimum, then there exist $\lambda \ge 0$ and $\mu$ such that the KKT conditions are satisfied, i.e.,

$$\nabla f(x^*) + \nabla g(x^*)\lambda + A^T \mu = 0,$$

$$\lambda^T g(x^*) = 0.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.7</span></p>

If Slater's condition is not satisfied, then the KKT conditions property (Corollary 5.5) can fail. Consider an optimization problem $\min_{x \in M} x_1$. Two constraints describe the feasible set having the form of a half-line starting from point $x^*$. Point $x^*$ is optimal. The KKT conditions read $-\nabla f(x^*) = \nabla g(x^*)\lambda$, but the point $x^*$ does not fulfill them since the gradients $\nabla g_1(x^*) = (0,-1)^T$ and $\nabla g_2(x^*) = (0,1)^T$ span a vertical line, not containing the opposite of the objective function gradient $-\nabla f(x^*) = (-1,0)^T$.

</div>

In optimization, necessary optimality conditions are usually preferred to sufficient optimality conditions since they often help to restrict the feasible set to a smaller set of candidate optimal solutions. Anyway, sufficient optimality conditions are also of interest, and below we show that the KKT conditions do this job under general assumptions.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.8</span><span class="math-callout__name">(Sufficient KKT Conditions)</span></p>

Let $x^* \in \mathbb{R}^n$ be a feasible solution of

$$\min \; f(x) \quad \text{subject to} \quad g(x) \le 0,$$

let $f(x)$ be a convex function, and let $g_j(x)$, $j \in I(x^*)$, be convex functions, too. If KKT conditions (5.5) are satisfied for $x^*$ with certain $\lambda \ge 0$, then $x^*$ is an optimal solution.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Convexity of function $f(x)$ implies $f(x) - f(x^*) \ge \nabla f(x^*)^T(x - x^*)$ due to Theorem 3.18. Analogously, for functions $g_j(x)$, $j \in I(x^*)$, we have $g_j(x) - g_j(x^*) \ge \nabla g_j(x^*)^T(x - x^*)$. KKT conditions give $\nabla f(x^*) = -\nabla g(x^*)\lambda$, from which premultiplying by $(x - x^*)$ we get

$$f(x) - f(x^*) \ge \nabla f(x^*)^T(x - x^*) = -\lambda^T \nabla g(x^*)^T(x - x^*)$$

$$= -\sum_{j \in I(x^*)} \lambda_j \nabla g_j(x^*)^T(x - x^*) \ge -\sum_{j \in I(x^*)} \lambda_j (g_j(x) - g_j(x^*))$$

$$= -\sum_{j \in I(x^*)} \lambda_j g_j(x) \ge 0.$$

Therefore $f(x^*)$ is the optimal value and $x^*$ is an optimal solution.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.9</span></p>

Consider the problem

$$\min \; x^T Cx \quad \text{subject to} \quad x^T x \le 1,$$

where $C \in \mathbb{R}^{n \times n}$ is symmetric, but not necessarily positive semidefinite. The problem is not convex, so we should not employ the KKT conditions. However, in this case, it is possible to do it since for $\lambda = 0$ they state optimality in the interior of the feasible set, and for $\lambda > 0$ we get $(C + \lambda I_n)x = 0$, so $-\lambda$ is an eigenvalue of matrix $C$. For any eigenvalue $\lambda_i$ and the corresponding eigenvector $x_i$ of length 1 we have $x_i^T Cx_i = x_i^T \lambda_i x_i = \lambda_i$. Conclusion: If $C$ is positive semidefinite, then the optimal value is 0; otherwise it is the smallest eigenvalue of matrix $C$.

</div>

# Chapter 6: Methods

To solve an optimization problem is a very difficult task in general; indeed, it is undecidable (provably there cannot exist an algorithm)! Thus we can hardly hope to solve optimally every problem. Many algorithms thus produce approximate solutions only — KKT solutions, local optima etc. If the problem is large and hard, then we often use heuristic methods (genetic and evolutionary algorithms, simulated annealing, tabu search, $\ldots$). On the other hand, many hard optimization problems can be solved by using global optimization techniques. However, they work in small dimensions only since their computational complexity is high. The choice of a suitable method thus depends not only on the type of the problem, but also on the dimensions, time restrictions etc.

## 6.1 Line Search

By a line search we mean minimization of a univariate function $f(x) \colon \mathbb{R} \to \mathbb{R}$, that is, we have $n = 1$. Even this particular case is important since it often serves as an auxiliary sub-procedure in the general case.

Our goal is to find a local minimum (or its approximation) in the neighbourhood of the current point. We present two approaches: Armijo rule, which aims to move to a point of a significant decrease of the objective function, and the Newton method, which converges to a local minimum under certain conditions.

### Armijo Rule

We assume that $f(x)$ is differentiable and $f'(0) < 0$, so it locally decreases at $x = 0$. We want to decrease the objective function by moving to the right from point $x = 0$. We wish to decrease it significantly, that is, not to get stuck locally close to $x = 0$, but to move away from this current point if possible.

Consider the condition

$$f(x) \le f(0) + \varepsilon \cdot f'(0) \cdot x, \tag{6.1}$$

where $0 < \varepsilon < 1$ is a given parameter; usually we take $\varepsilon \approx 0.2$.

The condition is used as follows: Choose a value of parameter $\beta > 0$ (e.g., $\beta = 2$ or $\beta = 10$) and an arbitrary $x > 0$. Now

- if condition (6.1) is satisfied, then set $x := \beta x$ and while the condition holds, repeat this process;
- if condition (6.1) is not satisfied, then set $x := x/\beta$ and repeat until the condition holds.

This procedure ensures that we move to a point with smaller objective value and simultaneously we move far from the initial point.

Armijo rule is also used as the termination condition within other line search methods: Condition (6.1) cannot be violated (which ensures that $x$ is not too large) and simultaneously the converse inequality

$$f(x) \ge f(0) + \varepsilon' \cdot f'(0) \cdot x,$$

must be satisfied for certain parameter $\varepsilon' > \varepsilon$ (which ensures that $x$ is not too large small).

### Newton Method

It is the classical Newton method for finding a root of $f'(x) = 0$. Here we need $f(x)$ to be twice differentiable.

This method is iterative and we construct a sequence of points $x_0 = 0, x_1, x_2, \ldots$ that, under some assumptions, converge to a local minimum. The basic idea is to approximate function $f(x)$ by a function $q(x)$ such that they both have the same value and the first and second derivatives and the current point $x_k$ (in the $k$th iteration). Thus we want $q(x_k) = f(x_k)$, $q'(x_k) = f'(x_k)$ and $q''(x_k) = f''(x_k)$. This suggests that it is suitable for $q(x_k)$ to be a quadratic polynomial. Such a quadratic function is unique and it is described by

$$q(x) = f(x_k) + f'(x_k)(x - x_k) + \frac{1}{2}f''(x_k)(x - x_k)^2.$$

The minimum of quadratic function $q(x_k)$ is at the stationary point (where the derivative is zero), so

$$0 = f'(x_k) + f''(x_k)(x - x_k).$$

From this we get

$$x = x_k - \frac{f'(x_k)}{f''(x_k)},$$

which is the current point $x_{k+1}$ of the subsequent iteration.

## 6.2 Unconstrained Problems

Consider the optimization problem

$$\min \; f(x) \quad \text{subject to} \quad x \in \mathbb{R}^n,$$

where $f(x)$ is a differentiable function.

A basic approach is an iterative method, generating a sequence of points $x_0, x_1, x_2, \ldots$, which, under certain assumptions, converge to a local minimum. The initial point $x_0$ can be chosen arbitrarily, unless we have some additional information that we can utilize. The iterations terminate when the objective function values at points $x_k$ get stabilized.

### Gradient Methods

In $k$th iteration, the current point is $x_k$. We determine a direction $d_k$ in which the objective function locally decreases, that is, $\nabla f(x_k)^T d_k < 0$. Now we call a line search method applied to the function $\varphi(\alpha) := f(x_k + \alpha d_k)$. Denote by $\alpha_k$ the output. Then the next point is set as $x_{k+1} := x_k + \alpha_k d_k$.

How to choose $d_k$? The simplest way is the **steepest descent method**, which takes $d_k := -\nabla f(x_k)$, that is, the direction in which the objective function locally decreases the most rapidly. This choice need no be the best one. There are advanced methods that take into account also the Hessian $\nabla^2 f(x_k)$ or its approximation and they combine the steepest descent direction and the directions of the previous iteration(s); see also the conjugate gradient methods in Section 6.4.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.1</span><span class="math-callout__name">(Learning of Neural Networks)</span></p>

Basically, the steepest descent method is used in learning of artificial neural networks. The goal of the learning is to set up weights of inputs of particular neurons such that the neural network performs best on the training data. Mathematically speaking, the variables are the weights of inputs of the neurons. The objective function that we minimize is the distance between the actual output vector and the ideal output vector. It is hard to find the optimal solution since this optimization problem is nonlinear, nonconvex and high-dimensional. That is why the problem is solved iteratively and at each step the weights are refined by means of the steepest descent. To compute the gradient of the objective function is also computationally demanding since there are usually large training data, so we simplify further and we approximate the gradient by its partial value based on the gradient of a randomly chosen training sample point. This approach is called *stochastic gradient descent*.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.2</span></p>

Optimization techniques are also used to solve problems that are not optimization problems in the essence. Consider for example the problem of solving a system of linear equations $Ax = b$, where $A$ is a positive definite matrix. Then the optimal solution of the convex quadratic program

$$\min_{x \in \mathbb{R}^n} \; \frac{1}{2}x^T Ax - b^T x$$

is the point $A^{-1}b$, the same as the solution of the equations, since at this point the gradient $\nabla f(x) = Ax - b$ vanishes. Thus we can solve linear equations by using optimization techniques. This is really used in practice, in particular for large and sparse systems. There exist several ways how to choose the vector $d_k$ in this context. For instance, the conjugate gradient method combines the gradient and the previous direction, so it takes a linear combination of $\nabla f(x_k)$ and $d_{k-1}$; see Section 6.4.

</div>

### Newton Method

This works in a similar fashion as in the univariate case. We approximate the objective function by a quadratic function, whose minimum is the current point of the subsequent iteration.

In step $k$, the current point is $x_k$ and at this point we approximate $f(x)$ by using Taylor expansion

$$f(x) \approx f(x_k) + \nabla f(x_k)^T(x - x_k) + \frac{1}{2}(x - x_k)^T \nabla^2 f(x_k)(x - x_k).$$

This gives us a quadratic function. If its Hessian $\nabla^2 f(x_k)$ is positive definite, then its minimum is unique and it is the point with zero gradient. This leads us to the system

$$\nabla f(x_k) + \nabla^2 f(x_k)(x - x_k) = 0,$$

from which we express the solution

$$x = x_k - (\nabla^2 f(x_k))^{-1} \nabla f(x_k).$$

This point is set as the current point $x_{k+1}$ of the next iteration.

*Comment.* The expression $y := (\nabla^2 f(x_k))^{-1} \nabla f(x_k)$ is evaluated by solving the system of linear equations $\nabla^2 f(x_k) y = \nabla f(x_k)$, not by inverting the matrix.

The advantage of this method is a rapid convergence (if we are close to the minimum). The drawback is that the Hessian $\nabla^2 f(x_k)$ need not be positive definite. Another drawback is that the evaluation of the Hessian might be computationally demanding. Therefore, diverse variants of this method exist (quasi-Newton methods) that approximate the Hessian matrix or regularize it. *Quasi-Newton* methods utilize an approximation of the Hessian matrix, so that we replace it with a certain positive definite matrix $P$. If we take $P := I_n$, it reduces to the steepest descent method (albeit with no perfect step length).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.3</span><span class="math-callout__name">(Scaling)</span></p>

Quasi-Newton methods relate with a scaling. Consider a gradient method with the direction $d_k := -D_k \nabla f(x_k)$, where $D_k$ is a positive definite matrix. During the iterations we have $\nabla f(x_k) \ne 0$, so the derivative of function $f(x)$ in the direction of $d_k$ reads as

$$\nabla f(x_k)^T d_k = -\nabla f(x_k)^T D_k \nabla f(x_k) < 0.$$

It means that for any positive definite $D_k$, the vector $d_k$ represents a descent direction and we can use it for the iterations. Putting $D_k := I_n$, we get the steepest descent method, and putting $D_k := (\nabla^2 f(x_k))^{-1}$, we get the Newton method. The matrix $D_k$ is often chosen to be diagonal. For instance, it can be formed by the inverse elements of the diagonal of $\nabla^2 f(x_k)$, in which case we save computational time of computing the whole Hessian matrix.

</div>

## 6.3 Constrained Problems

Consider the optimization problem

$$\min \; f(x) \quad \text{subject to} \quad x \in M,$$

where $f \colon \mathbb{R}^n \to \mathbb{R}$ is a differentiable function and the feasible set $M \subseteq \mathbb{R}^n$ is characterized by the system

$$g_j(x) \le 0, \quad j = 1, \ldots, J, \qquad h_\ell(x) = 0, \quad \ell = 1, \ldots, L,$$

where $g_j(x), h_\ell(x) \colon \mathbb{R}^n \to \mathbb{R}$.

The solution methods are again iterative, where we construct a sequence of points $x_0, x_1, x_2, \ldots$ The initial point $x_0$ is chosen randomly, unless we have some additional knowledge about the problem. The iterations terminate when the objective function values at points $x_k$ get stabilized.

### 6.3.1 Methods of Feasible Directions

These methods naturally the generalize gradient methods from unconstrained optimization. The basic idea is the same and the only difference is in the line search, when we must stay within the feasible set $M$. The equality constraints $h(x) = 0$ are hard to deal with in this case.

These methods are particularly convenient when $M$ is a convex polyhedron. So in this section we assume that $M = \lbrace x \in \mathbb{R}^n;\; Ax \le b \rbrace$.

**Method by Frank and Wolfe (1956).** Let $x_k$ be the current feasible point in $k$th iteration. A feasible descent direction $d_k$ is computed by an auxiliary linear program

$$\min \; \nabla f(x_k)^T x \quad \text{subject to} \quad Ax \le b.$$

Denote by $x_k^*$ its optimal solution. Then we take $d_k := x_k^* - x_k$. This direction is feasible since $x_k^* \in M$. Moreover, $d_k$ corresponds to a steep descent since the objective function $\nabla f(x_k)^T(x - x_k)$ yields the derivative of function $f$ at point $x_k$ in the direction of $x - x_k$ (the term $\nabla f(x_k)^T x_k$ is negligible since it is constant).

**Method by Zoutendijk (1960).** This method is similar to the previous one, but the auxiliary problem has the form

$$\min \; \nabla f(x_k)^T x \quad \text{subject to} \quad Ax \le b, \; \lVert x - x_k \rVert \le 1.$$

If we use the Euclidean norm, then we are seeking for the steepest descent direction that is feasible. In order that the auxiliary problem is easy to solve, we usually employ the maximum or the Manhattan norm. For the latter, for example, the problem takes the form of a linear program, in which $\lVert x - x_k \rVert \le 1$ is replaced by

$$e^T z \le 1, \quad x - x_k \le z, \quad -x + x_k \le z.$$

### 6.3.2 Active-Set Methods

These methods reduce the problem to a sequence of optimization problems with equality constraints only.

Let $x_k$ be a current feasible solution and let

$$W := \lbrace j;\; g_j(x_k) = 0 \rbrace$$

be the active set. Then we solve an auxiliary problem

$$\min \; f(x) \quad \text{subject to} \quad h(x) = 0, \; g_j(x) = 0, \; j \in W.$$

If we move to the boundary of $M$ during the computation and another constraint becomes active, then we include it to $W$. If we achieve a local minimum $x^*$ during the computation of this auxiliary problem, then we assume that KKT conditions are satisfied. That is, there exists $\lambda$ such that

$$\nabla f(x^*) + \nabla h(x^*)\mu + \sum_{j \in W} \lambda_j \nabla g_j(x^*) = 0.$$

Now, if $\lambda_j \ge 0$, then $j$ remains in $W$; otherwise the index $j$ is removed from $W$. This treatment is based on the interpretation of Lagrange multipliers as the negative derivatives of the objective function with respect to the right-hand side of the constraints. Hence, $\lambda_j < 0$ implies that locally a decrease of $g_j(x)$ makes a decrease of $f(x)$.

The schema of this method resembles the simplex method in linear programming, in which we move from one feasible basis to another and dynamically change the active set. Therefore, the active-set method is primarily used in optimization problems with linear constraints.

### 6.3.3 Penalty and Barrier Methods

These methods transform the problem in such a way that the constraint functions are added to the objective function and the problem is reduced to an unconstrained problem (in fact, to a series of unconstrained problems). This transformation works such that we pay a penalization in the form of higher objective values for infeasible points (penalty methods), or we force the computed points to stay in the interior of the feasible set by increasing the objective function values to infinity on its boundary (barrier methods).

#### Penalty Methods

Consider the problem

$$\min \; f(x) \quad \text{subject to} \quad x \in M,$$

where $f(x)$ is a continuous function and $M \ne \emptyset$ is a closed set. A **penalty function** is any continuous nonnegative function $q \colon \mathbb{R}^n \to \mathbb{R}$ satisfying the conditions:

- $q(x) = 0$ for every $x \in M$,
- $q(x) > 0$ for every $x \notin M$.

Penalty methods are based on a transformation of the problem to an unconstrained problem

$$\min \; f(x) + c \cdot q(x) \quad \text{subject to} \quad x \in \mathbb{R}^n,$$

where $c > 0$ is a parameter.

Penalty methods are implemented such that $c$ is not constant, but it is increased during the iterations. Too high value of $c$ at the beginning leads to a numerically ill-conditioned problem. That is why in practice the values from a suitable sequence $c_k > 0$, where $c_k \to_{k \to \infty} \infty$, are used.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.4</span></p>

Let $x_k$ be an optimal solution of problem

$$\min \; f(x) + c_k \cdot q(x) \quad \text{subject to} \quad x \in \mathbb{R}^n.$$

If $x_k \to_{k \to \infty} x^*$, then $x^*$ is an optimal solution of the original problem $\min_{x \in M} f(x)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

If $x^* \notin M$, then for $k^*$ large enough we have $x_k \notin M$ $\forall k \ge k^*$, and thus the objective function grows without bound. Hence $f(x^*) + c_k \cdot q(x^*) \to_{k \to \infty} \infty$ and also $f(x_k) + c_k \cdot q(x_k) \to_{k \to \infty} \infty$, which contradicts optimality of $x_k$.

Consider now the case of $x^* \in M$ and suppose to the contrary that $x^*$ is not optimal. Then there is a point $x' \in M$ such that $f(x') < f(x^*)$. Since the penalization is zero within the feasible set $M$, we get

$$f(x') + c_k \cdot q(x') < f(x^*) + c_k \cdot q(x^*)$$

for every $k \in \mathbb{N}$. Due to continuity we have for sufficiently large $k$

$$f(x') + c_k \cdot q(x') < f(x_k) + c_k \cdot q(x_k),$$

which is a contradiction to the optimality of $x_k$ in iteration $k$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.5</span></p>

For constraints of type $g(x) \le 0$ we often use the penalty function

$$q(x) := \sum_{j=1}^J (g_j(x)^+)^2 = \sum_{j=1}^J \max(0, g_j(x))^2,$$

which preserves smoothness of the objective function, and for constraints of type $h(x) = 0$ we can use the penalty function

$$q(x) := \sum_{\ell=1}^L h_\ell(x)^2.$$

</div>

#### Barrier Methods

Consider again the problem

$$\min \; f(x) \quad \text{subject to} \quad x \in M,$$

where $f(x)$ is a continuous function. Suppose that $M$ is a connected set satisfying $M = \text{cl}(\text{int}\,M)$, that is, it is equal to the closure if its interior. A **barrier function** is any continuous nonnegative function $q \colon \text{int}\,M \to \mathbb{R}$ such that $q(x) \to \infty$ for every $x \to \partial M$. This means that when $x$ approaches the boundary of $M$, then the barrier function grows to infinity.

The original problem is then transformed to an unconstrained problem

$$\min \; f(x) + \frac{1}{c} q(x) \quad \text{subject to} \quad x \in \mathbb{R}^n, \tag{6.2}$$

where $c > 0$ is a parameter.

The algorithm is similar to penalty methods, that is, we iteratively seek for optimal solutions of auxiliary problems when $c \to \infty$. A drawback of this method is that we have to know an initial feasible solution at the beginning. The advantage is its simplicity.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.6</span></p>

For constraints of type $g(x) \le 0$ we often use the barrier function in the form

$$q(x) := -\sum_{j=1}^J \frac{1}{g_j(x)} \tag{6.3}$$

or in the form

$$q(x) := -\sum_{j=1}^J \log(-g_j(x)). \tag{6.4}$$

Both barrier functions preserve convexity: If functions $g_j(x)$ are convex, then $q(x)$ is convex (see Theorem 3.31). Barrier function (6.4) is utilized in the popular interior point methods, which implementations can solve linear programs and certain convex optimization problems (such as quadratic programs) in polynomial time. For example, the linear program

$$\min \; c^T x \quad \text{subject to} \quad Ax \le b$$

is transformed to the problem

$$\min \; c^T x - \frac{1}{c_k} \sum_{i=1}^m \log(b_i - A_{i*}x).$$

For semidefinite condition $X \succeq 0$ we can use the barrier function

$$q(X) := -\log(\det(X)).$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.7</span></p>

For concreteness, consider the constraint $x \in [-1, 2]$. The inverse barrier function (6.3) takes the form

$$q(x) := \frac{1}{x+1} + \frac{1}{2-x}, \tag{6.5}$$

and the logarithmic barrier function (6.4) reads as

$$q(x) := -\log(x+1) - \log(2-x). \tag{6.6}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.8</span></p>

Let $c_k > 0$ be a sequence of numbers such that $c_k \to_{k \to \infty} \infty$. Let $x_k$ be an optimal solution of problem

$$\min \; f(x) + \frac{1}{c_k} q(x) \quad \text{subject to} \quad x \in \mathbb{R}^n.$$

If $x_k \to_{k \to \infty} x^*$, then $x^*$ is an optimal solution of the original problem $\min_{x \in M} f(x)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Suppose to the contrary that $x^*$ is not optimal, that is, there is $x' \in M$ such that $f(x') < f(x^*)$. Due to continuity of $f(x)$ there is $x'' \in \text{int}\,M$ such that $f(x'') < f(x^*)$. Then for $k$ large enough we have

$$f(x'') + \frac{1}{c_k} q(x'') < f(x^*) + \frac{1}{c_k} q(x^*).$$

For $k$ large enough we also have

$$f(x'') + \frac{1}{c_k} q(x'') < f(x_k) + \frac{1}{c_k} q(x_k),$$

which is a contradiction to the optimality of $x_k$ in step $k$.

</details>
</div>

For convex optimization problems under general assumptions (e.g., strictly convex barrier function and $M$ bounded) the optimal solution $x(c)$ of (6.2) is unique and the points $x(c)$, $c > 0$, draw a smooth curve, called the **central path**, whose limit as $c \to \infty$ is the optimal solution of the original problem.

Certain algorithms use the same principle: For the increasing values of $c$ they find (approximation of) the optimal solutions $x(c)$. With a small change of $c$ the point $x(c)$ moves continuously, so it is easy and fast to reoptimize and find the new optimum. For theoretical analysis of polynomiality of certain convex optimization problems short steps are used, but in practice larger steps are convenient. Typically, we increase $c$ with a factor of 1.1.

A natural question is, why not to choose a large value of $c$ at the beginning? The numerical issues cause troubles then. Next, such a choice makes not the algorithm faster. The Newton method (or other methods used to solve (6.2)) is slow if we start far from the optimum. Therefore tracing the central path using fast steps is the most convenient way.

Eventually, draw attention to one practical aspect — tunneling through the barrier. This occurs because practical methods are iterative and proceed from one solution to another one. Then it may happen that one step is too long and penetrates the barrier. Therefore, one has to take this into account when implementing the barrier methods.

## 6.4 Conjugate Gradient Method

This method was derived to solve a system of linear equations $Ax = b$, where matrix $A \in \mathbb{R}^{n \times n}$ is positive definite. Its authors are Hestenes and Stiefel (1952), and it belongs to both optimization textbooks and textbooks on numerical mathematics. Even though the method is iterative, it converges to the solution in at most $n$ steps. Since it does not transform matrix $A$ and has low space complexity, it is convenient for very large systems in particular.

The basic idea is to consider the quadratic function

$$f(x) = \frac{1}{2}x^T Ax - b^T x.$$

Since $A$ is positive definite, the function is strictly convex and attains the unique minimum. The minimum is the point, in which the gradient $\nabla f(x) = Ax - b$ is zero. Hence the minimum of function $f(x)$ is the same as the solution of $Ax = b$. In this way we reduced the problem of solving linear equations to an optimization problem.

We will describe the method in a simplified way. First, instead of the standard basis of $\mathbb{R}^n$ we consider an orthonormal basis $d_1, \ldots, d_n$ and the inner product $\langle x, y \rangle := x^T Ay$ instead of the standard one; to avoid confusion, the corresponding orthogonality is called A-orthogonality and the orthonormal basis is called A-orthonormal. We will show later on how to choose the basis. Denote by $x^* := A^{-1}b$ the solution we are seeking for, and denote by $x_k$ an approximate solution obtained in $k$th iteration. At the beginning, the initial point $x_1$ is chosen arbitrarily.

**Basic scheme.** We express vector $x^* - x_1$ as a linear combination of the basis vectors

$$x^* - x_1 = \sum_{k=1}^n \alpha_k d_k.$$

The basic scheme of the method is simple — imagine we move from a vertex of a box to the opposite vertex by using the (mutually perpendicular) edges:

Iterate $x_{k+1} := x_k + \alpha_k d_k$, $k = 1, \ldots$

To implement the method we need to determine the basis $d_1, \ldots, d_n$ and show how to compute coefficients $\alpha_k$ effectively. Denote $g_k := \nabla f(x_k) = Ax_k - b$, which represents not only the gradient at point $x_k$ in $k$th iteration, but also the residual, that is, the difference between the left and right-hand sides of the system (when the residual is 0, then we get the solution). Notice that for any $j \in \lbrace 1, \ldots, k \rbrace$,

$$x_{k+1} = x_k + \alpha_k d_k = x_{k-1} + \alpha_k d_k + \alpha_{k-1} d_{k-1} = \ldots = x_j + \sum_{i=j}^k \alpha_i d_i.$$

**Computation of $\alpha_k$.** Since $d_1, \ldots, d_n$ is an A-orthonormal basis, the coordinates $\alpha_k$ are the Fourier coefficients and we compute them easily as $\alpha_k = \langle d_k, x^* - x_1 \rangle$. The problem is that $x^*$ is unknown. Since $x_k - x_1 = \sum_{i=1}^{k-1} \alpha_i d_i$, vector $x_k - x_1$ is A-orthogonal to $d_k$, that is, $\langle d_k, x_k - x_1 \rangle = 0$. We derive

$$\alpha_k = \langle d_k, x^* - x_1 \rangle = \langle d_k, x^* - x_k + x_k - x_1 \rangle = \langle d_k, x^* - x_k \rangle + \langle d_k, x_k - x_1 \rangle$$

$$= \langle d_k, x^* - x_k \rangle = d_k^T A(x^* - x_k) = d_k^T(b - Ax_k) = -d_k^T g_k.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.9</span></p>

Vector $x_{k+1}$ is the minimum of $f(x)$ on the affine subspace $x_1 + \text{span}\lbrace d_1, \ldots, d_k \rbrace$, that is, $g_{k+1}^T d_j = 0$ for $j = 1, \ldots, k$ (i.e., $g_{k+1} \perp d_j$ meaning the standard orthogonality).

</div>

**The choice of basis $d_1, \ldots, d_n$.** We choose the basis such that $\text{span}\lbrace d_1, \ldots, d_k \rbrace = \text{span}\lbrace g_1, \ldots, g_k \rbrace$ for every $k = 1, \ldots, n$. At the beginning we naturally put $d_1 := -g_1 / \sqrt{\langle g_1, g_1 \rangle}$. In $(k+1)$st iteration we construct vector $d_{k+1}$ from vector $-g_{k+1}$ by making it orthogonal to subspace $\text{span}\lbrace d_1, \ldots, d_k \rbrace$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.10</span></p>

$\text{span}\lbrace g_1, \ldots, g_k \rbrace = \text{span}\lbrace g_1, Ag_1, \ldots, A^{k-1}g_1 \rbrace$.

</div>

Since $g_{k+1}$ is orthogonal (in the standard sense) to vectors $d_1, \ldots, d_k$, it is also orthogonal to $g_1, \ldots, g_k$, and by Proposition 6.10 it is A-orthogonal to vectors $g_1, \ldots, g_{k-1}$, too. Thus, in order to compute $d_{k+1}$, it is sufficient to make $-g_{k+1}$ orthogonal to vector $d_k$. Notice that the resulting value of $d_{k+1}$ is not normalized, so we have to normalize it afterwards.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.11</span></p>

We have $d_{k+1} = -g_{k+1} + \beta_{k+1} d_k$, where $\beta_{k+1} = \langle d_k, g_{k+1} \rangle$.

</div>

### Summary — The Algorithm

Now we have all the ingredients to explicitly write the algorithm:

1. choose $x_1 \in \mathbb{R}^n$ and put $d_0 := 0$,
2. for $k = 1, \ldots, n$ do

$$g_k := Ax_k - b,$$

$$\beta_k := d_{k-1}^T A g_k,$$

$$d_k := -g_k + \beta_k d_{k-1}, \quad d_k := d_k / \sqrt{d_k^T A d_k},$$

$$\alpha_k := -d_k^T g_k,$$

$$x_{k+1} := x_k + \alpha_k d_k.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.12</span></p>

A few of comments to the conjugate gradient method:

1. The method has low memory requirement and makes no operations on matrix $A$. The method is beneficial particularly when matrix $A$ is large and sparse. The running time of one iteration is relatively low. Moreover, not all $n$ iterations are needed to perform in general — we can achieve the solution or its tight approximation much sooner.
2. Often the method is presented without the normalization of vector $d_k$. Then the expressions with $d_k$ must be adjusted accordingly.
3. If we choose $x_1 = 0$, then $\text{span}\lbrace d_1, \ldots, d_k \rbrace = \text{span}\lbrace b, Ab, A^2 b, \ldots, A^{k-1}b \rbrace$ is called the Krylov subspace and the theory behind is very interesting.

The basic idea of the conjugate gradient method can be used to minimize a general nonlinear function $f(x)$ over space $\mathbb{R}^n$. Herein, the key idea is to construct the improving direction $d_k$ as a linear combination of gradient $g_k$ and the previous direction $d_{k-1}$. Vector $g_k$ is then the gradient of function $f(x)$ at point $x_k$, and the coefficients are computed analogously. The resulting method is called the method of Fletcher–Reeves (1964). There exist several variants, which differ in the values of coefficients $\beta_k$.

There are also methods employing Krylov subspaces for solving systems $Ax = b$, where matrix $A$ is not necessarily symmetric positive definite. For example, let us mention GMRES (Generalized minimal residual method, Saad & Schultz, 1986), which in $k$th iteration computes vector $x_k$ that minimizes the Euclidean norm of the residual (i.e., $\lVert Ax - b \rVert$) over subspace $\text{span}\lbrace b, Ab, A^2 b, \ldots, A^{k-1}b \rbrace$.

</div>

# Chapter 7: Selected Topics

## 7.1 Robust Optimization

In practice, data are often inexact or subject to various uncertainties. This motivates us to seek for solutions that are *robust*. There is no precise definition, but basically it means that a robust solution is feasible and optimal even for specific data perturbations. We present two approaches to robustness, the interval one and the ellipsoidal one.

### Interval Uncertainty (I)

Consider first a linear program in the form

$$\min \; c^T x \quad \text{subject to} \quad Ax \le b, \; x \ge 0.$$

Suppose that $A$ and $b$ are not known exactly and the only information that we have are interval estimations of the values. That is, we know a matrix of intervals $[\underline{A}, \overline{A}]$ and the vector of interval right-hand sides $[\underline{b}, \overline{b}]$. We say that a vector $x$ is a robust feasible solution if it fulfills inequality $Ax \le b$ for each $A \in [\underline{A}, \overline{A}]$ and $b \in [\underline{b}, \overline{b}]$. Due to nonnegativity of variables we have that $x$ is robust feasible if and only if $\overline{A}x \le \underline{b}$. Hence the robust counterpart of the linear program reads

$$\min \; c^T x \quad \text{subject to} \quad \overline{A}x \le \underline{b}, \; x \ge 0.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.1</span><span class="math-callout__name">(Catfish Diet Problem)</span></p>

This is a simplified example of an optimization model of finding a minimum cost catfish diet in Thailand. The mathematical formulation reads

$$\min \; c^T x \quad \text{subject to} \quad Ax \ge b, \; x \ge 0, \tag{7.1}$$

where variable $x_j$ stands for the number of units of food $j$ to be consumed by the catfish, $b_i$ is the required minimal amount of nutrient $i$, $c_j$ is the price per unit of food $j$, and $a_{ij}$ is the amount of nutrient $i$ contained in one unit of food $j$. The data are:

$$A = \begin{pmatrix} 9 & 65 & 44 & 12 & 0 \\ 1.10 & 3.90 & 2.57 & 1.99 & 0 \\ 0.02 & 3.7 & 0.3 & 0.1 & 38.0 \end{pmatrix}, \quad b = \begin{pmatrix} 30 \\ 250 \\ 0.5 \end{pmatrix}, \quad c = \begin{pmatrix} 2.15 \\ 8.0 \\ 6.0 \\ 2.0 \\ 0.4 \end{pmatrix}.$$

Since the nutritive values are not known exactly, we assume that their accuracy is 5%. Hence the exact value of each entry of matrix $A$ lies in interval $[0.95 \cdot a_{ij}, 1.05 \cdot a_{ij}]$. According to the lines described above, the robust counterpart is obtained by setting the constraint matrix to be $\underline{A}$, that is,

$$\underline{A} = \begin{pmatrix} 8.550 & 61.75 & 41.800 & 11.400 & 0.00 \\ 1.045 & 3.705 & 2.4415 & 1.8905 & 0.00 \\ 0.019 & 3.515 & 0.2850 & 0.0950 & 36.1 \end{pmatrix}.$$

</div>

### Interval Uncertainty (II)

Consider now a linear program in the form with variables unrestricted in sign

$$\min \; c^T x \quad \text{subject to} \quad Ax \le b.$$

Let $a^T x \le d$ be a selected inequality. Let intervals $[\underline{a}, \overline{a}] = ([\underline{a}_1, \overline{a}_1], \ldots, [\underline{a}_n, \overline{a}_n])^T$ and $[\underline{d}, \overline{d}]$ be given. A solution $x$ is a robust solution of the selected inequality if it satisfies

$$a^T x \le d \quad \forall a \in [\underline{a}, \overline{a}], \; \forall d \in [\underline{d}, \overline{d}],$$

or, $\max_{a \in [\underline{a}, \overline{a}]} a^T x \le \underline{d}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 7.2</span></p>

Denote by $a_\Delta = \frac{1}{2}(\overline{a} - \underline{a})$ the vector of interval radii and by $a_c = \frac{1}{2}(\underline{a} + \overline{a})$ the vector of interval midpoints. Then

$$\max_{a \in [\underline{a}, \overline{a}]} a^T x = a_c^T x + a_\Delta^T \lvert x \rvert.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For every $a \in [\underline{a}, \overline{a}]$ we have

$$a^T x = a_c^T x + (a - a_c)^T x \le a_c^T x + \lvert a - a_c \rvert^T \lvert x \rvert \le a_c^T x + a_\Delta^T \lvert x \rvert.$$

The inequality is attained as equation for certain $a \in [\underline{a}, \overline{a}]$. If $x \ge 0$, then $a_c^T x + a_\Delta^T \lvert x \rvert = a_c^T x + a_\Delta^T x = \overline{a}^T x$. If $x \le 0$, then $a_c^T x + a_\Delta^T \lvert x \rvert = a_c^T x - a_\Delta^T x = \underline{a}^T x$. Otherwise we apply this idea entrywise, so that inequality is attained for $a$ each entry of which is the interval left or right endpoint.

</details>
</div>

We use this lemma to express the robust solution constraint as

$$a_c^T x + a_\Delta^T \lvert x \rvert \le \underline{d}.$$

The left-hand side function is convex, but not smooth. Nevertheless, we can rewrite the constraint as a linear constraint by introducing an auxiliary variable $y \in \mathbb{R}^n$

$$a_c^T x + a_\Delta^T y \le \underline{d}, \quad x \le y, \quad -x \le y.$$

Therefore linearity is preserved — the robust solutions of interval linear programs are also described by linear constraints.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.3</span><span class="math-callout__name">(Robust Classification)</span></p>

Consider two classes of data, the first one comprises given points $x_1, \ldots, x_p \in \mathbb{R}^n$, and the second one contains given points $y_1, \ldots, y_q \in \mathbb{R}^n$. We wish to construct a classifier that is able to predict to which class a new input belongs to. A basic linear classifier is based on data separation by a widest separating band. Mathematically, we seek for a hyperplane $a^T x + b = 0$ such that the first set of points belongs to the positive halfspace, the second set of points belongs to the negative halfspace, and the separating band is maximal. This leads to a convex quadratic program

$$\min \; \lVert a \rVert_2 \quad \text{subject to} \quad a^T x_i + b \ge 1 \; \forall i, \; a^T y_j + b \le -1 \; \forall j.$$

Suppose now that data are not measured exactly and one knows them with a specified accuracy only. Hence we are given vectors of intervals $[\underline{x}_i, \overline{x}_i] = [(x_c)_i - (x_\Delta)_i, (x_c)_i + (x_\Delta)_i]$, $i = 1, \ldots, p$, and $[\underline{y}_j, \overline{y}_j] = [(y_c)_j - (y_\Delta)_j, (y_c)_j + (y_\Delta)_j]$, $j = 1, \ldots, q$, comprising the true data. Using the approach described above, the robust counterpart model reads

$$\min \; \lVert a \rVert_2 \quad \text{subject to} \quad (x_c)_i^T a + (x_\Delta)_i^T a' + b \le 1 \; \forall i, \; (y_c)_j^T a + (y_\Delta)_j^T a' + b \le -1 \; \forall j, \; \pm a \le a'.$$

Again, it is a convex quadratic program (in variables $a, a' \in \mathbb{R}^n$ and $b \in \mathbb{R}$).

</div>

### Ellipsoidal Uncertainty

Consider again the linear program in the form with variables unrestricted in sign

$$\min \; c^T x \quad \text{subject to} \quad Ax \le b.$$

Let $a^T x \le d$ be a selected inequality. Consider an ellipsoid

$$\mathcal{E} = \lbrace a \in \mathbb{R}^n;\; a = p + Pu, \; \lVert u \rVert_2 \le 1 \rbrace,$$

which is expressed as the image of a unit ball under a linear (or more precisely affine) mapping. A point $x$ is a robust solution of the selected inequality if it satisfies

$$a^T x \le d \quad \forall a \in \mathcal{E},$$

or, $\max_{a \in \mathcal{E}} a^T x \le d$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 7.4</span></p>

We have

$$\max_{a \in \mathcal{E}} a^T x = p^T x + \lVert P^T x \rVert_2.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Write

$$\max_{a \in \mathcal{E}} a^T x = \max_{\lVert u \rVert_2 \le 1} (p + Pu)^T x = p^T x + \max_{\lVert u \rVert_2 \le 1} (P^T x)^T u = p^T x + (P^T x)^T \frac{1}{\lVert P^T x \rVert_2} P^T x = p^T x + \lVert P^T x \rVert_2.$$

</details>
</div>

Using the lemma, we can express the robust solution constraint as

$$p^T x + \lVert P^T x \rVert_2 \le d.$$

The left-hand side function is smooth and convex — indeed it is a second order cone constraint.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.5</span><span class="math-callout__name">(Portfolio Selection under Ellipsoidal Uncertainty)</span></p>

Consider again the portfolio selection problem (Example 4.10)

$$\max \; c^T x \quad \text{subject to} \quad e^T x = K, \; x \ge o, \tag{7.2}$$

where $c$ is a random Gaussian vector, its expected value is $\tilde{c} := \text{E}\,c$ and the covariance matrix is $\Sigma := \text{cov}\,c = \text{E}\,(c - \tilde{c})(c - \tilde{c})^T$. The level sets of the density function represent ellipsoids, so it is natural to work with them. For a random vector $c$ we have that the probability $P(c - \tilde{c} \in \mathcal{E}_\eta) = \eta$, where $\mathcal{E}_\eta$ is a certain ellipsoid (concretely, $\mathcal{E}_\eta = \lbrace d \in \mathbb{R}^n;\; d = F^{-1}(\eta)\sqrt{\Sigma}u, \; \lVert u \rVert_2 \le 1 \rbrace$, where $F^{-1}(\eta)$ is the quantile function of the normal distribution and $\sqrt{\Sigma}$ is the positive semidefinite square root of matrix $\Sigma$, i.e., $(\sqrt{\Sigma})^2 = \Sigma$).

One of the possible ways to solve (7.2) is to consider the deterministic counterpart

$$\max \; z \quad \text{subject to} \quad P(c^T x \ge z) \ge \eta, \; e^T x = K, \; x \ge o,$$

where $\eta \in [\frac{1}{2}, 1]$ is a fixed value, e.g., $\eta = 0.95$. Obviously, condition $P(c^T x \ge z) \ge \eta$ is fulfilled if $d^T x \ge z$ holds for every $d \in \mathcal{E}_\eta + \tilde{c}$. Hence we can approximate the problem as

$$\max \; z \quad \text{subject to} \quad d^T x \ge z \; \forall d \in \mathcal{E}_\eta + \tilde{c}, \; e^T x = K, \; x \ge o.$$

This optimization problem involves ellipsoidal uncertainty, so we can equivalently write it as

$$\max \; z \quad \text{subject to} \quad \tilde{c}^T x - F^{-1}(\eta) \lVert \sqrt{\Sigma} x \rVert_2 \ge z, \; e^T x = K, \; x \ge o.$$

Since $F^{-1}(\eta) \ge 0$ for any $\eta \ge \frac{1}{2}$, it is a second order cone programming problem.

</div>

## 7.2 Concave Programming

Concave programming means minimizing a concave function on a convex set, or equivalently maximizing a convex function

$$\max \; f(x) \quad \text{subject to} \quad x \in M,$$

where $M \subseteq \mathbb{R}^n$ is a convex set and $f \colon \mathbb{R}^n \to \mathbb{R}$ is a convex function.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7.6</span></p>

Let $M$ be a bounded convex polyhedron. Then the optimal solution exists and it is attained in at least one of the vertices of $M$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $v_1, \ldots, v_m$ be vertices of $M$, and without loss of generality assume that $f(v_1) = \max_{i=1,\ldots,m} f(v_i)$. Then every point $x \in M$ can be expressed as a convex combination $x = \sum_{i=1}^m \alpha_i v_i$, where $\alpha_i \ge 0$ and $\sum_{i=1}^m \alpha_i = 1$. Now

$$f(x) = f(\sum_{i=1}^m \alpha_i v_i) \le \sum_{i=1}^m \alpha_i f(v_i) \le \sum_{i=1}^m \alpha_i f(v_1) = f(v_1).$$

Therefore $v_1$ is an optimum.

</details>
</div>

*Remark.* The theorem can be extended as follows: Any continuous convex function on a compact set $M$ attains its maximum in an extreme point of $M$. This property holds even more generally, when function $f(x)$ is so called quasiconvex.

This property holds in linear programming, too. For computing an optimal solution, however, it is not very convenient since polyhedron $M$ may contain many vertices, and we do not know which one is optimal. By Theorem 4.9, concave programming is NP-hard.

Typical problems resulting in concave programming comprise

- *Fixed charged problems.* The objective function has the form $f(x) = \sum_{i=1}^k f_i(x_i)$, where $f_i(x_i) = 0$ for $x_i = 0$ and $f_i(x_i) = c_i + g_i(x_i)$ for $x_i > 0$. Herein, $f_i(x_i)$ represents a price (e.g., the price for the transport of goods of size $x_i$). Hence the price is naturally zero when $x_i = 0$. When $x_i > 0$, we pay a fixed charge $c_i$ plus the price $g_i(x_i)$ depending on the size. We can assume that $g_i(x_i)$ is concave since the larger $x_i$, the smaller relative price for the unit of goods (e.g., due to discounts).

- *Multiplicative programming.* The objective function has the form $f(x) = \prod_{i=1}^k x_i$. This is not a concave function in general, but its logarithm gives a concave function $\log(f(x)) = \sum_{i=1}^k \log(x_i)$. Such problems appear in geometry, where, for example, we minimize the volume of a body (e.g., a cuboid) subject to some constraints (e.g., the cuboid contains specified points).

# Appendix

## Derivative of Matrix Expressions

Let $A \in \mathbb{R}^{n \times n}$, $b \in \mathbb{R}^n$ and $c \in \mathbb{R}$. Consider the quadratic function $f \colon \mathbb{R}^n \to \mathbb{R}$ defined as

$$f(x) = x^T Ax + b^T x + c.$$

Its gradient reads

$$\nabla f(x) = (A + A^T)x + b,$$

and the Hessian matrix takes the form

$$\nabla^2 f(x) = A + A^T.$$

In particular, if matrix $A$ is symmetric, then

$$\nabla f(x) = 2Ax + b, \quad \nabla^2 f(x) = 2A.$$

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

First we consider the linear term

$$\frac{\partial}{\partial x_k} b^T x = \frac{\partial}{\partial x_k} \sum_{i=1}^n b_i x_i = b_k,$$

whence $\nabla b^T x = b$.

For the quadratic term, we get

$$\frac{\partial}{\partial x_k} x^T Ax = \frac{\partial}{\partial x_k} \sum_{i=1}^n \sum_{j=1}^n a_{ij} x_i x_j = \frac{\partial}{\partial x_k} \left( a_k x_k^2 + \sum_{i \ne k} (a_{ik} + a_{ki}) x_i x_k + \sum_{i,j \ne k} a_{ij} x_i x_j \right)$$

$$= 2a_k x_k + \sum_{i \ne k} (a_{ik} + a_{ki}) x_i = \sum_{i=1}^n (a_{ik} + a_{ki}) x_i = ((A + A^T)x)_i.$$

Hence the gradient reads $\nabla x^T Ax = (A + A^T)x$. Since this is a linear function, the particular coordinates are differentiated in the same way as for the linear term. Therefore $\nabla^2 x^T Ax = A + A^T$.

</details>
</div>
