---
layout: default
title: Linear Algebra I & II
date: 2025-03-10
excerpt: Introduction, notation, complex numbers, polynomials, analytic geometry, and optimization.
tags:
  - linear-algebra
  - mathematics
---

**Table of Contents**
- TOC
{:toc}

# Linear Algebra I & II

## Chapter 1 — Introduction

### About the book and linear algebra

Linear algebra is one of the fundamental branches of mathematics. Its main tools are vectors and matrices. Vectors live in some space, and linear algebra deals with linear objects in space — points, lines, planes, etc. It also studies linear mappings (rotations, projections, etc.).

Matrices represent data in matrix form and correspond to linear transformations in Euclidean space. Matrices can be viewed in two ways — algebraically and geometrically.

Main topics of the text:

- **Systems of linear equations** — the most fundamental problem of linear algebra. Gaussian elimination can in principle solve any system.
- **Affine subspaces** — every geometric object can be described using a system of equations.
- **Matrices** — a fundamental tool. Types of matrices and their relationship to systems of linear equations.
- **Groups and fields** — extending results to other number domains (complex numbers, $\mathbb{F}_2$, etc.).
- **Vectors and vector spaces** — axiomatically defined spaces, concepts of dimension and isomorphism.
- **Linear mappings** — transformations mapping lines to lines and the origin to the origin.
- **Inner product** — geometry (orthogonality, distances, projections).
- **Determinants and eigenvalues** — characteristics of matrices (volume, explicit formulas, finer information about matrix behavior).
- **Positive definite matrices and quadratic forms** — ellipsoids, optimization, statistics.
- **Matrix decompositions** — QR, SVD, and others.

### Concepts and notation

#### Number domains

- $\mathbb{N} = \lbrace 1, 2, \dots \rbrace$ — the set of natural numbers
- $\mathbb{Z}$ — the set of integers
- $\mathbb{Q}$ — the set of rational numbers
- $\mathbb{R}$ — the set of real numbers
- $\mathbb{C}$ — the set of complex numbers

#### Summation

The symbol $\sum$ represents the sum of all instances of the expression following the summation sign:

$$\sum_{i=1}^{n} a_i = a_1 + \ldots + a_n.$$

The sum over an empty index set is defined as $0$.

#### Product

The symbol $\prod$ is used for products:

$$\prod_{i=1}^{n} a_i = a_1 \cdot a_2 \cdot \ldots \cdot a_n.$$

#### Quantifiers

- $\forall$ — "for all" (universal quantifier)
- $\exists$ — "there exists" (existential quantifier)

For example, $\forall x \in \mathbb{R}: x + 1 \le e^x$ reads "for all real numbers $x$, the inequality $x + 1 \le e^x$ holds."

#### Modulo

The modulo $a \bmod n$ gives the remainder when dividing the integer $a$ by the natural number $n$. The remainder is defined as the number $b \in \lbrace 0, 1, \ldots, n-1 \rbrace$ such that $a = zn + b$ for some $z \in \mathbb{Z}$. For example:

$$17 \bmod 7 = 3, \qquad -17 \bmod 7 = 4.$$

#### Factorial

The factorial of a natural number $n$ is the product $1 \cdot 2 \cdot 3 \cdot \ldots \cdot n$ and is denoted $n!$.

#### Points and vectors

A point, as an $n$-tuple of real numbers $(v_1, \ldots, v_n)$, behaves algebraically the same as an arithmetic vector. The two concepts are interpreted differently only from a geometric perspective, where a nonzero vector corresponds to a particular direction.

#### Mappings

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Mapping)</span></p>

A mapping $f$ from the set $\mathcal{A}$ to the set $\mathcal{B}$ is denoted $f \colon \mathcal{A} \to \mathcal{B}$. For every $a \in \mathcal{A}$, $f(a)$ is defined and belongs to $\mathcal{B}$.

- **Injective** (one-to-one): $a_1 \neq a_2 \Rightarrow f(a_1) \neq f(a_2)$.
- **Surjective** (onto): for every $b \in \mathcal{B}$ there exists $a \in \mathcal{A}$ such that $f(a) = b$.
- **Bijective** (one-to-one correspondence): both injective and surjective.
- If $f$ is a bijection, then there exists an **inverse mapping** $f^{-1} \colon \mathcal{B} \to \mathcal{A}$ defined by $f^{-1}(b) = a$ if $f(a) = b$. We have $(f^{-1})^{-1} = f$.
- **Composition**: $(g \circ f)(a) = g(f(a))$. The composition of bijections is a bijection. Composition is associative: $h \circ (g \circ f) = (h \circ g) \circ f$.
- **Isomorphism**: a bijective mapping preserving structure. If an isomorphism between $\mathcal{A}$ and $\mathcal{B}$ exists, then the sets are called *isomorphic*.

</div>

#### Countable and uncountable sets

A set $M$ is **countable** if there exists a bijection between $M$ and $\mathbb{N}$ or a subset thereof. Thus every finite set is countable, as are $\mathbb{N}$, $\mathbb{Z}$, $\mathbb{Q}$. A set that is not countable is called **uncountable** — an example is $\mathbb{R}$.

#### Relations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Relation)</span></p>

A *binary relation* $R$ on a set $M$ is any subset of the Cartesian product $M \times M = \lbrace (x, y) ;\; x, y \in M \rbrace$. The relation $R$ is

- **reflexive**, if $(x, x) \in R$ for every $x \in M$,
- **symmetric**, if $(x, y) \in R \Rightarrow (y, x) \in R$,
- **anti-symmetric**, if $(x, y) \in R$ and $(y, x) \in R \Rightarrow x = y$,
- **transitive**, if $(x, y) \in R$ and $(y, z) \in R \Rightarrow (x, z) \in R$.

A relation is an **equivalence** if it is reflexive, symmetric, and transitive. A relation is a **(partial) order** if it is reflexive, anti-symmetric, and transitive.

</div>

Examples of equivalence: equality of numbers, equality of remainders upon division, congruence of geometric objects, matching of colors.

Examples of partial order: inequality $\le$, inclusion $\subseteq$, divisibility on natural numbers.

### Construction of mathematical theory

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Basic building blocks)</span></p>

- **Definition** is a precise specification of a concept using basic or previously defined concepts.
- **Proposition** is a statement whose truth must be confirmed by a proof.
- **Theorem** is a more significant proposition.
- **Lemma** is an auxiliary proposition serving to prove a more complex theorem.
- **Corollary** is a proposition that more or less simply follows from the preceding one.
- **Proof** is a sequence of logical steps formally establishing the validity of a proposition.

</div>

A proposition has the form of an implication "If $\mathcal{P}$ holds, then $\mathcal{T}$ holds." Two basic proof techniques:

- **Direct proof**: From the assumption $\mathcal{P}$ and a sequence of valid derivations, one arrives at the validity of $\mathcal{T}$.
- **Proof by contradiction**: One starts from the assumption $\mathcal{P}$ and the negation of $\mathcal{T}$ and arrives at a logical contradiction.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof by mathematical induction)</span></p>

Proof by mathematical induction is used for statements of the type "For all natural $n$, $\mathcal{T}(n)$ holds." It has two steps:

1. **Base case**: We verify the validity of $\mathcal{T}(n)$ for $n = 1$.
2. **Inductive step** ($n \leftarrow n - 1$): We show the validity of $\mathcal{T}(n)$ using the validity of $\mathcal{T}(n-1)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Proof by induction)</span></p>

*Claim:* For every natural number $n$, the number $n^3 + 2n$ is divisible by three.

*Base case ($n = 1$):* $1^3 + 2 \cdot 1 = 3$, which is divisible by three.

*Inductive step:* We assume validity for $n-1$ and want to show validity for $n$:

$$n^3 + 2n = ((n-1)+1)^3 + 2((n-1)+1) = (n-1)^3 + 2(n-1) + 3(n-1)^2 + 3(n-1) + 3.$$

By the inductive hypothesis, $(n-1)^3 + 2(n-1)$ is divisible by three, hence $n^3 + 2n$ is also divisible by three.

</div>

### Number representation

Due to limited memory, not all real numbers can be represented in a computer. Numbers are standardly represented in so-called floating-point form

$$m \times b^\top,$$

where $b$ is the base of the numeral system (usually $b = 2$), $m$ is the mantissa and $t$ the exponent. The mantissa is normalized so that $1 \le m < b$. Since both the mantissa and exponent have limited size on a computer, we can represent only finitely many real numbers.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Rounding errors)</span></p>

Rounding errors can accumulate when evaluating arithmetic operations. For example, the sum of $1/3$ and $1/3$ in four-digit decimal arithmetic leads to the sum of representations $0.3333$ and $0.3333$ with result $0.6666$. However, the closest representable number to the actual sum $2/3$ is $0.6667$.

The influence of rounding errors must be taken into account when designing algorithms. This topic is the subject of *numerical analysis*.

</div>

### Complex numbers

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Complex number)</span></p>

A *complex number* $z$ is introduced as the expression $a + bi$, where $a, b \in \mathbb{R}$ and the imaginary unit $i$ satisfies $i^2 = -1$. Here $a$ is the **real part** ($\operatorname{Re}(z)$) and $b$ is the **imaginary part** ($\operatorname{Im}(z)$). The set of complex numbers is denoted $\mathbb{C}$.

</div>

Basic operations for $z_1 = a + bi$, $z_2 = c + di$:

$$z_1 + z_2 = (a + c) + (b + d)i, \qquad z_1 z_2 = (ac - bd) + (cb + ad)i.$$

For $z_2 \neq 0$:

$$\frac{z_1}{z_2} = \frac{a + bi}{c + di} = \frac{a + bi}{c + di} \cdot \frac{c - di}{c - di} = \frac{ac + bd}{c^2 + d^2} + \frac{cb - ad}{c^2 + d^2}i.$$

Complex numbers have a geometric interpretation: the number $a + bi$ corresponds to the point $(a, b)$ in the complex (Gauss) plane.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Complex conjugate and absolute value)</span></p>

For $z = a + bi$ we define:

- **Complex conjugate**: $\overline{z} \coloneqq a - bi$.
- **Absolute value**: $\lvert z \rvert \coloneqq \sqrt{z \overline{z}} = \sqrt{a^2 + b^2}$.

The absolute value determines the Euclidean distance of the point $(a, b)$ from the origin. The complex conjugate represents reflection about the real axis.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Complex conjugates and absolute values)</span></p>

- $z = \overline{z}$ if and only if $z$ is a real number,
- $z + \overline{z} = 2\operatorname{Re}(z)$,
- $\overline{z_1 + z_2} = \overline{z_1} + \overline{z_2}$,
- $\overline{z_1 \cdot z_2} = \overline{z_1} \cdot \overline{z_2}$,
- $\lvert z_1 + z_2 \rvert \le \lvert z_1 \rvert + \lvert z_2 \rvert$ (triangle inequality),
- $\lvert z_1 \cdot z_2 \rvert = \lvert z_1 \rvert \cdot \lvert z_2 \rvert$,
- $\operatorname{Re}(z) \le \lvert z \rvert$.

Note: in general $\lvert z \rvert^2 \neq z^2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric meaning of operations)</span></p>

- **Addition** $z \mapsto z + v$ represents a shift in the direction of the vector $(\operatorname{Re}(v), \operatorname{Im}(v))$.
- **Multiplication** $z \mapsto vz$: if $v$ is real, this is a scaling by the factor $\lvert v \rvert$. If $v$ is complex and $\lvert v \rvert = 1$, this is a rotation by the angle $\alpha$ that $v$ makes with the real axis. In the general case, both properties are combined.

Example: $v = i$ represents a rotation by $90°$.

</div>

### Polynomials

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Polynomial)</span></p>

A *real polynomial* of degree $n$ is a function $p(x) = a_n x^n + a_{n-1} x^{n-1} + \ldots + a_1 x + a_0$, where $a_0, \ldots, a_n \in \mathbb{R}$ and $a_n \neq 0$. In addition to real polynomials, one can consider polynomials with complex coefficients.

</div>

Operations with polynomials $p(x) = a_n x^n + \ldots + a_1 x + a_0$ and $q(x) = b_m x^m + \ldots + b_1 x + b_0$ (let $n \ge m$):

- **Addition**: $p(x) + q(x) = a_n x^n + \ldots + (a_m + b_m) x^m + \ldots + (a_0 + b_0)$.
- **Multiplication**: $p(x)q(x) = a_n b_m x^{n+m} + \ldots + a_0 b_0$.
- **Division with remainder**: There exist a uniquely determined polynomial $r(x)$ of degree $n - m$ and a polynomial $s(x)$ of degree less than $m$ such that $p(x) = r(x)q(x) + s(x)$.

#### Roots

A *root* of a polynomial $p(x)$ is a value $x^* \in \mathbb{R}$ (resp. $\mathbb{C}$) such that $p(x^*) = 0$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Fundamental theorem of algebra)</span></p>

Every polynomial with complex coefficients has at least one complex root.

</div>

If $x_1$ is a root of the polynomial $p(x)$, then $p(x)$ is divisible by the factor $(x - x_1)$ without remainder and the quotient is a polynomial of degree $n - 1$. By repeated application of the fundamental theorem of algebra we obtain the factorization

$$p(x) = a_n (x - x_1)(x - x_2) \cdots (x - x_n),$$

where $x_1, \ldots, x_n$ are the roots (counting multiplicities). A polynomial of degree $n$ thus has exactly $n$ roots.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Finding roots)</span></p>

The roots of a quadratic polynomial $a_2 x^2 + a_1 x + a_0$ are found by the formula $x_{1,2} = \frac{1}{2a_2}(-a_1 \pm \sqrt{a_1^2 - 4a_2 a_0})$. For degree three, Cardano's formulas exist (but are much more complicated). Abel (1824) showed that for polynomials of degree higher than 4, no formula for computing roots exists. Therefore we find roots using iterative methods.

</div>

### Analytic geometry

#### Line in the plane

*Equation description* of a line in the plane:

$$a_1 x_1 + a_2 x_2 = b,$$

where $a_1, a_2, b \in \mathbb{R}$ and at least one of $a_1, a_2$ is nonzero. All points $(x_1, x_2)$ satisfying the equation form a line. The vector $(a_1, a_2)$ is called the **normal vector** and is perpendicular to the line.

*Parametric description* of a line in the plane:

$$(x_1, x_2) = (b_1, b_2) + t \cdot (v_1, v_2), \quad t \in \mathbb{R},$$

where $(b_1, b_2)$ is a given point on the line and $(v_1, v_2) \neq (0, 0)$ is the **direction vector** of the line.

#### Line in space

Parametric description of a line in space:

$$(x_1, x_2, x_3) = (b_1, b_2, b_3) + t \cdot (v_1, v_2, v_3), \quad t \in \mathbb{R},$$

where $(v_1, v_2, v_3) \neq (0, 0, 0)$ is the direction vector. In $n$-dimensional space analogously:

$$(x_1, \ldots, x_n) = (b_1, \ldots, b_n) + t \cdot (v_1, \ldots, v_n), \quad t \in \mathbb{R}.$$

The direction vector is determined up to a scalar multiple uniquely, while the point $(b_1, \ldots, b_n)$ can be chosen anywhere on the line.

#### Plane in space

A single equation $a_1 x_1 + a_2 x_2 + a_3 x_3 = b$, where $(a_1, a_2, a_3) \neq (0, 0, 0)$, describes a plane in space. The vector $(a_1, a_2, a_3)$ is its **normal vector** — it is perpendicular to the plane and is determined up to a scalar multiple uniquely.

The equation description of a line in space requires two equations:

$$a_1 x_1 + a_2 x_2 + a_3 x_3 = b_1, \qquad a_1' x_1 + a_2' x_2 + a_3' x_3 = b_2,$$

where the normals must be nonzero and must not point in the same direction (the planes must not be parallel).

### Optimization

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Optimization problem)</span></p>

Let $f \colon \mathbb{R}^n \to \mathbb{R}$ be a real function and $M \subseteq \mathbb{R}^n$ a set of points. The *optimization* problem is

$$\min f(x) \quad \text{subject to } x \in M.$$

We seek a point $x \in M$ such that $f(x) \le f(y)$ for all $y \in M$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Linear optimization)</span></p>

Consider the function $f(x_1, x_2) = x_1 - x_2$ and the set $M$ in the plane defined by the constraints $x_1 + 2x_2 \le 4$, $x_1 \ge 0$, $x_2 \ge 0$. The set $M$ is a triangle with vertices $(0, 0)$, $(4, 0)$ and $(0, 2)$. The minimum value of the function is attained at the point $(0, 2)$.

</div>

### Mathematical software

Functions for solving basic problems of linear algebra are a standard part of mathematical software systems:

- **Matlab** — a rich environment for numerical computations. **Octave** is an open-source alternative with nearly identical syntax.
- **Mathematica** and **Maple** — leading systems for symbolic computation. **SageMath** is a freely available alternative.
- **Julia** — a modern language for computationally intensive tasks with support for parallel and distributed computing.
- **Wolfram Alpha** — an online computational system based on Mathematica.

## Chapter 2 — Systems of Linear Equations

### Basic Concepts

Systems of linear equations are among the fundamental algebraic problems and we encounter them almost everywhere — if a problem does not lead to a system of equations directly, systems of equations often appear as a subproblem.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Historical problem — Jiuzhang Suanshu, ca 200 BC)</span></p>

*Three sheaves of good grain, two sheaves of average grain, and one sheaf of poor grain sell for a total of 39 dou. Two sheaves of good grain, three of average, and one of poor sell for 34 dou. One sheaf of good grain, two of average, and three of poor sell for 26 dou. What is the price of one sheaf of good / average / poor grain?*

Written in modern mathematics, we obtain a system of equations:

$$3x + 2y + z = 39, \qquad 2x + 3y + z = 34, \qquad x + 2y + 3z = 26,$$

where $x, y, z$ are the unknowns for the prices of one sheaf of good / average / poor grain.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Matrix)</span></p>

A real *matrix* of type $m \times n$ is a rectangular array (table) of real numbers

$$A = \begin{pmatrix} a_{11} & a_{12} & \ldots & a_{1n} \\ \vdots & \vdots & & \vdots \\ a_{m1} & a_{m2} & \ldots & a_{mn} \end{pmatrix}.$$

The element at position $(i, j)$ of matrix $A$ (i.e., in the $i$-th row and $j$-th column) is denoted $a_{ij}$ or $A_{ij}$. The set of all real matrices of type $m \times n$ is denoted $\mathbb{R}^{m \times n}$; similarly for complex, rational, etc. If $m = n$, the matrix is called *square*.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vector)</span></p>

A real $n$-dimensional arithmetic column *vector* is a matrix of type $n \times 1$:

$$x = \begin{pmatrix} x_1 \\ \vdots \\ x_n \end{pmatrix}$$

and a row vector is a matrix of type $1 \times n$: $x = (x_1, \ldots, x_n)$.

By default, unless stated otherwise, we consider column vectors. The set of all $n$-dimensional vectors is denoted $\mathbb{R}^n$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Notation for rows and columns)</span></p>

- The $i$-th row of matrix $A$ is denoted $A_{i*} = (a_{i1}, a_{i2}, \ldots, a_{in})$.
- The $j$-th column of matrix $A$ is denoted $A_{*j} = (a_{1j}, a_{2j}, \ldots, a_{mj})^\top$.

Matrix $A \in \mathbb{R}^{m \times n}$ can therefore be written column-wise as $A = (A_{*1} \; A_{*2} \; \ldots \; A_{*n})$ or row-wise. General matrices are denoted by uppercase letters and vectors by lowercase letters.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(System of linear equations)</span></p>

Consider a system of $m$ linear equations in $n$ unknowns:

$$a_{11}x_1 + a_{12}x_2 + \ldots + a_{1n}x_n = b_1, \quad \ldots, \quad a_{m1}x_1 + a_{m2}x_2 + \ldots + a_{mn}x_n = b_m,$$

where $a_{ij}, b_i$ are given coefficients and $x_1, \ldots, x_n$ are unknowns. By a *solution* we mean every vector $x \in \mathbb{R}^n$ satisfying all equations.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Coefficient matrix and augmented matrix)</span></p>

The *coefficient matrix* of the system is

$$A = \begin{pmatrix} a_{11} & a_{12} & \ldots & a_{1n} \\ \vdots & \vdots & & \vdots \\ a_{m1} & a_{m2} & \ldots & a_{mn} \end{pmatrix}$$

and the *augmented matrix* of the system is

$$(A \mid b) = \begin{pmatrix} a_{11} & a_{12} & \ldots & a_{1n} & b_1 \\ a_{21} & a_{22} & \ldots & a_{2n} & b_2 \\ \vdots & \vdots & & \vdots & \vdots \\ a_{m1} & a_{m2} & \ldots & a_{mn} & b_m \end{pmatrix}.$$

The vertical line in the augmented matrix symbolizes the equality between the left-hand side and the right-hand side of the system. Rows correspond to equations, columns on the left to the variables $x_1, \ldots, x_n$, and the last column to the values on the right-hand side.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric meaning of a system of equations)</span></p>

For $n = 2$ (two equations in two unknowns), each equation describes a line in the plane $\mathbb{R}^2$. The solution of the system lies at the intersection of the two lines.

For $n = 3$, each equation describes a plane in space $\mathbb{R}^3$. The solution of the system is the intersection of these planes — it can be a single point (general position), a line (the planes share a common line), or the empty set (parallel planes).

In general, for arbitrary $n$, the equations determine so-called hyperplanes and we seek the solution of the system at their intersection.

</div>

### Elementary Row Operations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Elementary row operations)</span></p>

The elementary row operations on a matrix are:

1. Multiplying the $i$-th row by a real number $\alpha \neq 0$ (i.e., all elements of the row are multiplied).
2. Adding the $\alpha$-multiple of the $j$-th row to the $i$-th row, where $i \neq j$ and $\alpha \in \mathbb{R}$.
3. Swapping the $i$-th and $j$-th rows.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Independence of operations)</span></p>

In fact, the above-mentioned operations are not all that elementary. For the second row operation, we only need $\alpha = 1$, and the third operation can be simulated using the previous two.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 2.10)</span></p>

Elementary row operations preserve the solution set of a system.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof idea</summary>

The key idea is to show that an elementary operation does not change the solution set. An elementary operation does not lose any solution, because if $x$ is a solution before the operation, it is also a solution after the operation. Conversely, an operation does not introduce any new solution, because every operation has its inverse — by applying a suitable elementary operation we can return to the original form of the system.

</details>
</div>

### Gaussian Elimination

The basic idea of the method is to transform the augmented matrix of the system using elementary row operations into a simpler matrix from which the solution can be easily read off.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gaussian elimination — demonstration)</span></p>

Consider the system of linear equations:

$$x_1 + 2x_2 + 3x_3 = 32, \quad x_1 + x_2 + 2x_3 = 21, \quad 3x_1 + x_2 + 3x_3 = 35.$$

**Forward elimination:** We successively eliminate variables. By subtracting the first equation from the second and three times the first from the third, followed by further operations, we obtain:

$$x_1 + 2x_2 + 3x_3 = 32, \quad -x_2 - x_3 = -11, \quad -x_3 = -6.$$

**Back substitution:** From the third equation $x_3 = 6$. Substituting into the second: $x_2 = 5$. Substituting into the first: $x_1 = 4$. The solution of the system is $(x_1, x_2, x_3) = (4, 5, 6)$.

</div>

#### Row Echelon Form (REF)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Row echelon form — REF)</span></p>

A matrix $A \in \mathbb{R}^{m \times n}$ is in *row echelon form* (REF) if there exists $r$ such that:

- rows $1, \ldots, r$ are nonzero (i.e., each contains at least one nonzero entry),
- rows $r+1, \ldots, m$ are zero,

and moreover, if we denote $p_i = \min\lbrace j;\; a_{ij} \neq 0 \rbrace$ the position of the first nonzero element in the $i$-th row, then $p_1 < p_2 < \cdots < p_r$.

The positions $(1, p_1), (2, p_2), \ldots, (r, p_r)$ are called **pivots**. The columns $p_1, p_2, \ldots, p_r$ are called **basic** (pivot columns) and the remaining columns **non-basic** (free columns).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Matrices in REF and not in REF)</span></p>

The following matrices are in row echelon form:

$$\begin{pmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 0 & 0 & 6 \end{pmatrix}, \quad \begin{pmatrix} 1 & 2 & 3 \\ 0 & 0 & 5 \\ 0 & 0 & 0 \end{pmatrix}, \quad \begin{pmatrix} 1 & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix}, \quad \begin{pmatrix} 0 & 1 & 2 & 3 \\ 0 & 0 & 4 & 5 \\ 0 & 0 & 0 & 0 \end{pmatrix}.$$

The following matrices are *not* in row echelon form:

$$\begin{pmatrix} 1 & 1 & 1 \\ 0 & 0 & 2 \\ 0 & 0 & 3 \end{pmatrix}, \quad \begin{pmatrix} 1 & 2 & 3 \\ 2 & 3 & 0 \\ 3 & 0 & 0 \end{pmatrix}, \quad \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}, \quad \begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 2 & 2 \\ 0 & 0 & 0 & 0 \end{pmatrix}.$$

</div>

#### Rank of a Matrix

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Rank of a matrix)</span></p>

The *rank* of a matrix $A$ is the number of nonzero rows after reduction to row echelon form, denoted $\operatorname{rank}(A)$.

</div>

The rank of a matrix is therefore equal to the number of pivots (i.e., the number $r$) after reduction to row echelon form. Even though the row echelon form is not unique, the pivot positions are unique (see Theorem 2.28 below). Therefore the notion of rank is well-defined.

#### REF Algorithm

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(REF(A) — Algorithm 2.15)</span></p>

**Input:** matrix $A \in \mathbb{R}^{m \times n}$.

1. $i := 1$, $j := 1$.
2. **if** $a_{k\ell} = 0$ for all $k \ge i$ and $\ell \ge j$ **then** stop.
3. $j := \min\lbrace \ell;\; \ell \ge j, \; a_{k\ell} \neq 0 \text{ for some } k \ge i \rbrace$ (skip zero subcolumns).
4. Find $k$ such that $a_{kj} \neq 0$, $k \ge i$, and swap rows $A_{i*}$ and $A_{k*}$.
5. For all $k > i$ set $A_{k*} := A_{k*} - \frac{a_{kj}}{a_{ij}} A_{i*}$ (2nd elementary operation).
6. Set $i := i + 1$, $j := j + 1$, and go to step 2.

**Output:** matrix $A$ in row echelon form.

In practice, **partial pivoting** is recommended in step 4 — choosing the candidate $a_{kj}$ with the maximum absolute value, which has better numerical properties.

</div>

The algorithm reduces the matrix to row echelon form in at most $\min(m, n)$ iterations of the main loop. In step 5, we zero out all elements below the pivot $(i, j)$.

#### Gaussian Elimination — Solving a System

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Gaussian elimination — Algorithm 2.17)</span></p>

**Input:** system of equations $(A \mid b)$, where $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$.

We reduce the augmented matrix $(A \mid b)$ to row echelon form $(A' \mid b')$ and denote $r = \operatorname{rank}(A \mid b)$. Exactly one of three cases occurs:

**(A) The system has no solution.** This occurs when the last column is a pivot column, i.e., $\operatorname{rank}(A) < \operatorname{rank}(A \mid b)$. The last nonzero row has the form $0x_1 + \ldots + 0x_n = b_r' \neq 0$.

**(B1) The system has exactly one solution.** This occurs when $r = n$ (the number of variables equals the number of pivots) and the last column is non-basic. The solution is found by **back substitution**: Successively for $k = n, n-1, \ldots, 1$ we compute

$$x_k := \frac{b_k' - \sum_{j=k+1}^{n} a_{kj}' x_j}{a_{kk}'}.$$

**(B2) The system has infinitely many solutions.** This occurs when $r < n$. The matrix has at least one non-basic column. The non-basic variables are free parameters; the basic variables are determined by back substitution. The number of non-basic variables $n - r > 0$ expresses the dimension of the solution set.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gaussian elimination with infinitely many solutions)</span></p>

We solve the system with augmented matrix:

$$\begin{pmatrix} 2 & 2 & -1 & 5 & 1 \\ 4 & 5 & 0 & 9 & 3 \\ 0 & 1 & 2 & 2 & 4 \\ 2 & 4 & 3 & 7 & 7 \end{pmatrix} \xrightarrow{\text{REF}} \begin{pmatrix} 2 & 2 & -1 & 5 & 1 \\ 0 & 1 & 2 & -1 & 1 \\ 0 & 0 & 0 & 3 & 3 \\ 0 & 0 & 0 & 0 & 0 \end{pmatrix}.$$

Basic columns: 1, 2, 4. Non-basic column: 3 ($x_3$ is a free variable). Back substitution:

1. $x_4 = 1$,
2. $x_3$ is a free (non-basic) variable,
3. $x_2 = 1 + x_4 - 2x_3 = 2 - 2x_3$,
4. $x_1 = \frac{1}{2}(1 - 5x_4 + x_3 - 2x_2) = -4 + \frac{5}{2}x_3$.

All solutions: $(-4 + \tfrac{5}{2}x_3,\; 2 - 2x_3,\; x_3,\; 1)$, where $x_3 \in \mathbb{R}$, or equivalently

$$(-4, 2, 0, 1) + x_3 \cdot (\tfrac{5}{2}, -2, 1, 0), \quad x_3 \in \mathbb{R}.$$

The solution set represents a line in $\mathbb{R}^4$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Computational complexity)</span></p>

The number of operations of the REF$(A)$ algorithm can be expressed as a polynomial in $n$. The leading term consists of $n^2$ multiplications and $n^2$ subtractions in the first iteration, $(n-1)^2$ in the second, etc. Using the formula $\sum_{k=1}^{n} k^2 = \frac{1}{6}n(n+1)(2n+1)$, we see that the total asymptotic complexity of the REF algorithm is on the order of $\frac{2}{3}n^3$ operations.

Back substitution requires on the order of $n^2$ operations, so it does not significantly affect the overall complexity.

</div>

### Gauss-Jordan Elimination

#### Reduced Row Echelon Form (RREF)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reduced row echelon form — RREF)</span></p>

A matrix $A \in \mathbb{R}^{m \times n}$ is in *reduced row echelon form* (RREF) if it is in REF and additionally:

- $a_{1p_1} = a_{2p_2} = \ldots = a_{rp_r} = 1$, i.e., the pivot positions contain ones, and
- for each $i = 1, \ldots, r$ we have $a_{1p_i} = a_{2p_i} = \ldots = a_{i-1,p_i} = 0$, i.e., all entries above each pivot are zero.

</div>

#### RREF Algorithm

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(RREF(A) — Algorithm 2.21)</span></p>

**Input:** matrix $A \in \mathbb{R}^{m \times n}$.

1. $i := 1$, $j := 1$.
2. **if** $a_{k\ell} = 0$ for all $k \ge i$ and $\ell \ge j$ **then** stop.
3. $j := \min\lbrace \ell;\; \ell \ge j, \; a_{k\ell} \neq 0 \text{ for some } k \ge i \rbrace$.
4. Find $a_{kj} \neq 0$, $k \ge i$, and swap rows $A_{i*}$ and $A_{k*}$.
5. Set $A_{i*} := \frac{1}{a_{ij}} A_{i*}$ (now the pivot position has value $1$).
6. For all $k \neq i$ set $A_{k*} := A_{k*} - a_{kj} A_{i*}$ (2nd elementary operation — eliminates also **above** the pivot).
7. Set $i := i + 1$, $j := j + 1$, and go to step 2.

**Output:** matrix $A$ in reduced row echelon form.

</div>

The difference from the REF algorithm is in steps 5 and 6: the pivots are normalized to one and elimination is performed not only below but also above the pivot.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Reduction to RREF)</span></p>

$$\begin{pmatrix} 2 & 2 & -1 & 5 \\ 4 & 5 & 0 & 9 \\ 0 & 1 & 2 & 2 \\ 2 & 4 & 3 & 7 \end{pmatrix} \xrightarrow{\text{RREF}} \begin{pmatrix} 1 & 0 & -2.5 & 0 \\ 0 & 1 & 2 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \end{pmatrix}.$$

</div>

#### Gauss-Jordan Elimination — Solving a System

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Gauss-Jordan elimination — Algorithm 2.23)</span></p>

**Input:** system of equations $(A \mid b)$, where $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$.

We reduce the augmented matrix of the system to reduced row echelon form $(A' \mid b')$ and denote $r = \operatorname{rank}(A \mid b)$. We distinguish three cases:

**(A) The system has no solution.** This occurs when the last column is a pivot column, i.e., $\operatorname{rank}(A) < \operatorname{rank}(A \mid b)$.

**(B1) The system has exactly one solution.** This occurs when the last column is non-basic and $r = n$. All columns $1, \ldots, n$ are basic and the solution is directly $(x_1, \ldots, x_n) = (b_1', \ldots, b_n')$.

**(B2) The system has infinitely many solutions.** This occurs when $r < n$. The non-basic variables $x_i$, $i \in N = \lbrace 1, \ldots, n \rbrace \setminus \lbrace p_1, \ldots, p_r \rbrace$, are free parameters. The basic variables are determined by back substitution:

$$x_{p_k} := b_k' - \sum_{j \in N,\; j > p_k} a_{kj}' x_j.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gauss-Jordan elimination)</span></p>

The system from the previous example solved by Gauss-Jordan elimination:

$$\begin{pmatrix} 2 & 2 & -1 & 5 & 1 \\ 4 & 5 & 0 & 9 & 3 \\ 0 & 1 & 2 & 2 & 4 \\ 2 & 4 & 3 & 7 & 7 \end{pmatrix} \xrightarrow{\text{RREF}} \begin{pmatrix} 1 & 0 & -2.5 & 0 & -4 \\ 0 & 1 & 2 & 0 & 2 \\ 0 & 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 0 & 0 \end{pmatrix}.$$

Back substitution steps: $x_4 = 1$, $x_3$ free, $x_2 = 2 - 2x_3$, $x_1 = -4 + \frac{5}{2}x_3$. The same result as with Gaussian elimination.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gaussian vs. Gauss-Jordan elimination)</span></p>

Both methods have the same asymptotic complexity on the order of $n^3$. Gaussian elimination is approximately one-third faster; on the other hand, Gauss-Jordan elimination (i.e., the RREF form) is needed for matrix inversion.

The computational complexity of RREF$(A)$ for a square matrix of order $n$ is on the order of $n^3$ arithmetic operations.

</div>

#### Uniqueness of RREF

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Uniqueness of RREF — Theorem 2.28)</span></p>

The RREF of a matrix is unique. Regardless of which elementary row operations are performed and in what order, the resulting RREF of the matrix is always the same.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof idea</summary>

For contradiction, assume that a matrix $A \in \mathbb{R}^{m \times n}$ has two different RREF forms $A_1$ and $A_2$. Let $i$ be the index of the first column in which $A_1, A_2$ differ. We remove from matrices $A, A_1, A_2$ all non-basic columns before the $i$-th one. The resulting matrices $B, B_1, B_2$ have RREF forms $B_1$ and $B_2$. If we interpret matrix $B$ as a system of linear equations, from the RREF form $B_1$ we read off a different solution than from $B_2$, which is a contradiction. If both systems are inconsistent, then their last columns $c, d$ are basic, and therefore identical.

</details>
</div>

#### Frobenius' Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Frobenius' theorem — Remark 2.25)</span></p>

The system $(A \mid b)$ has at least one solution if and only if $\operatorname{rank}(A) = \operatorname{rank}(A \mid b)$.

</div>

### Applications

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Electrical circuit)</span></p>

Consider an electrical circuit with resistors $10\,\Omega$, $10\,\Omega$, $20\,\Omega$ and voltage sources $10\,\text{V}$, $5\,\text{V}$. We want to determine the currents $I_1, I_2, I_3$. Using Kirchhoff's laws (current law: $I_1 + I_2 - I_3 = 0$; voltage law for loops) we obtain the system:

$$\begin{pmatrix} 1 & 1 & -1 & 0 \\ 10 & -10 & 0 & 10 \\ 0 & 10 & 20 & 5 \end{pmatrix}.$$

Solving, we get $I_1 = 0.7\,\text{A}$, $I_2 = -0.3\,\text{A}$, $I_3 = 0.4\,\text{A}$.

</div>

## Chapter 3 — Matrices

We introduced matrices in the previous chapter for the compact notation of systems of linear equations and the description of methods for solving them. However, matrices have much broader applications, so in this chapter we will examine them in more detail. We will introduce several types of matrices and basic operations that allow us to work with matrices more effectively and simply.

### Basic matrix operations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Equality, sum, and scalar multiple of matrices)</span></p>

- **Equality:** Two matrices are equal, $A = B$, if they have the same dimensions $m \times n$ and $A_{ij} = B_{ij}$ for $i = 1, \ldots, m$, $j = 1, \ldots, n$.
- **Sum:** Let $A, B \in \mathbb{R}^{m \times n}$. Then $A + B$ is a matrix of type $m \times n$ with entries $(A + B)_{ij} = A_{ij} + B_{ij}$.
- **Scalar multiple:** Let $\alpha \in \mathbb{R}$ and $A \in \mathbb{R}^{m \times n}$. Then $\alpha A$ is a matrix of type $m \times n$ with entries $(\alpha A)_{ij} = \alpha A_{ij}$.

Subtraction is naturally defined as $A - B := A + (-1)B$. A special matrix is the **zero matrix**, all of whose entries are zeros; we denote it by $0$ or $0_{m \times n}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Sum and scalar multiples of matrices)</span></p>

For real numbers $\alpha, \beta$ and matrices $A, B, C \in \mathbb{R}^{m \times n}$ the following hold:

1. $A + B = B + A$ (commutativity),
2. $(A + B) + C = A + (B + C)$ (associativity),
3. $A + 0 = A$,
4. $A + (-1)A = 0$,
5. $\alpha(\beta A) = (\alpha \beta) A$,
6. $1A = A$,
7. $\alpha(A + B) = \alpha A + \alpha B$ (distributivity),
8. $(\alpha + \beta)A = \alpha A + \beta A$ (distributivity).

</div>

#### Matrix product

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Matrix product)</span></p>

Let $A \in \mathbb{R}^{m \times p}$ and $B \in \mathbb{R}^{p \times n}$. Then $AB$ is a matrix of type $m \times n$ with entries

$$(AB)_{ij} = \sum_{k=1}^{p} A_{ik} B_{kj}.$$

The entry at position $(i, j)$ of the product $AB$ is computed as the dot product of the $i$-th row of matrix $A$ and the $j$-th column of matrix $B$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Identity matrix)</span></p>

The *identity matrix* of order $n$, denoted $I$ or $I_n$, is a square matrix with entries $I_{ij} = 1$ for $i = j$ and $I_{ij} = 0$ otherwise. It has ones on the diagonal and zeros elsewhere. The *standard unit vector* $e_i$ is the $i$-th column of the identity matrix, i.e., $e_i = I_{*i}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Matrix product)</span></p>

For a scalar $\alpha$ and matrices $A, B, C$ of compatible dimensions, the following hold:

1. $(AB)C = A(BC)$ (associativity),
2. $A(B + C) = AB + AC$ (left distributivity),
3. $(A + B)C = AC + BC$ (right distributivity),
4. $\alpha(AB) = (\alpha A)B = A(\alpha B)$,
5. $0A = A0 = 0$,
6. $I_m A = A I_n = A$ where $A \in \mathbb{R}^{m \times n}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of associativity</summary>

Let $A \in \mathbb{R}^{m \times p}$, $B \in \mathbb{R}^{p \times r}$, and $C \in \mathbb{R}^{r \times n}$. Then $AB$ has type $m \times r$, $BC$ has type $p \times n$, and both products $(AB)C$, $A(BC)$ have type $m \times n$. At position $(i, j)$:

$$((AB)C)_{ij} = \sum_{k=1}^{r} (AB)_{ik} C_{kj} = \sum_{k=1}^{r} \left(\sum_{\ell=1}^{p} A_{i\ell} B_{\ell k}\right) C_{kj} = \sum_{k=1}^{r} \sum_{\ell=1}^{p} A_{i\ell} B_{\ell k} C_{kj},$$

$$(A(BC))_{ij} = \sum_{\ell=1}^{p} A_{i\ell} (BC)_{\ell j} = \sum_{\ell=1}^{p} A_{i\ell} \left(\sum_{k=1}^{r} B_{\ell k} C_{kj}\right) = \sum_{\ell=1}^{p} \sum_{k=1}^{r} A_{i\ell} B_{\ell k} C_{kj}.$$

Both expressions are identical due to the commutativity of addition of real numbers.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Non-commutativity of the product)</span></p>

Matrix multiplication is generally not commutative: for many matrices $AB \neq BA$. For example, for $A = \bigl(\begin{smallmatrix} 0 & 1 \\ 0 & 0 \end{smallmatrix}\bigr)$, $B = \bigl(\begin{smallmatrix} 1 & 0 \\ 0 & 0 \end{smallmatrix}\bigr)$ we have $AB = \bigl(\begin{smallmatrix} 0 & 0 \\ 0 & 0 \end{smallmatrix}\bigr)$ but $BA = \bigl(\begin{smallmatrix} 0 & 1 \\ 0 & 0 \end{smallmatrix}\bigr)$.

Moreover, it can happen that the product $AB$ is defined but $BA$ is not (the matrices have incompatible dimensions).

</div>

#### Transpose

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Transpose)</span></p>

Let $A \in \mathbb{R}^{m \times n}$. Then the *transposed matrix* $A^\top$ has type $n \times m$ and is defined by $(A^\top)_{ij} := a_{ji}$.

Transposition means flipping along the main diagonal. Thanks to transposition, we can write column vectors $x \in \mathbb{R}^n$ as $x = (x_1, \ldots, x_n)^\top$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Transpose)</span></p>

For a scalar $\alpha$ and matrices $A, B$ of compatible dimensions:

1. $(A^\top)^\top = A$,
2. $(A + B)^\top = A^\top + B^\top$,
3. $(\alpha A)^\top = \alpha A^\top$,
4. $(AB)^\top = B^\top A^\top$.

Property (4) can be extended by mathematical induction to a product of $k$ matrices: $(A_1 A_2 \ldots A_k)^\top = A_k^\top \ldots A_2^\top A_1^\top$.

</div>

#### Symmetric matrix

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Symmetric matrix)</span></p>

A matrix $A \in \mathbb{R}^{n \times n}$ is *symmetric* if $A = A^\top$.

</div>

Symmetric matrices are invariant under transposition — visually they are symmetric about the main diagonal. Examples of symmetric matrices are $I_n$, the zero matrix $0_n$, or $\bigl(\begin{smallmatrix} 1 & 2 \\ 2 & 3 \end{smallmatrix}\bigr)$. The sum of symmetric matrices is again a symmetric matrix, but this does not generally hold for the product.

For any matrix $B \in \mathbb{R}^{m \times n}$, the matrix $B^\top B$ is symmetric, since $(B^\top B)^\top = B^\top (B^\top)^\top = B^\top B$.

Symmetric matrices frequently appear in geometric problems (distance matrices), statistics (covariance matrices), and optimization (the Hessian).

#### Special types of matrices

- **Diagonal matrix**: $A \in \mathbb{R}^{n \times n}$ is diagonal if $a_{ij} = 0$ for all $i \neq j$. A diagonal matrix with entries $v_1, \ldots, v_n$ on the diagonal is denoted $\operatorname{diag}(v_1, \ldots, v_n)$.

- **Upper triangular matrix**: $A \in \mathbb{R}^{m \times n}$ is upper triangular if $a_{ij} = 0$ for all $i > j$. An example is any matrix in REF form (pivots must be on or above the diagonal).

- **Lower triangular matrix**: a matrix with zeros above the diagonal.

The product of two upper triangular matrices is again an upper triangular matrix.

#### Dot product and outer product of vectors

Transposition and the product of vectors viewed as single-column matrices allow us to introduce two important products:

The **standard dot product** of vectors $x, y \in \mathbb{R}^n$:

$$x^\top y = \sum_{i=1}^{n} x_i y_i$$

(formally a $1 \times 1$ matrix, which we identify with a real number). The standard Euclidean norm:

$$\lVert x \rVert = \sqrt{x^\top x} = \sqrt{\sum_{i=1}^{n} x_i^2}.$$

The **outer product** of vectors $x \in \mathbb{R}^n$, $y \in \mathbb{R}^n$ is a square matrix of order $n$:

$$xy^\top = \begin{pmatrix} x_1 y_1 & x_1 y_2 & \ldots & x_1 y_n \\ x_2 y_1 & x_2 y_2 & \ldots & x_2 y_n \\ \vdots & \vdots & & \vdots \\ x_n y_1 & x_n y_2 & \ldots & x_n y_n \end{pmatrix}.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 3.17 — Rank-1 matrices)</span></p>

A matrix $A \in \mathbb{R}^{m \times n}$ has rank 1 if and only if it has the form $A = xy^\top$ for some nonzero vectors $x \in \mathbb{R}^m$, $y \in \mathbb{R}^n$.

</div>

#### Properties of matrix-vector multiplication

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Proposition 3.18)</span></p>

Let $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$, $x \in \mathbb{R}^n$, and $y \in \mathbb{R}^m$. Then:

1. $Ae_j = A_{*j}$ (multiplication by a unit vector gives the $j$-th column),
2. $e_i^\top A = A_{i*}$ (gives the $i$-th row),
3. $(AB)_{*j} = A B_{*j}$ (the $j$-th column of the product is $A$ times the $j$-th column of $B$),
4. $(AB)_{i*} = A_{i*} B$ (the $i$-th row of the product is the $i$-th row of $A$ times $B$),
5. $Ax = \sum_{j=1}^{n} x_j A_{*j}$ (column interpretation),
6. $y^\top A = \sum_{i=1}^{m} y_i A_{i*}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Writing a system of equations and its interpretations)</span></p>

A system of linear equations can be written in matrix form as $Ax = b$, where $x = (x_1, \ldots, x_n)^\top$ is the vector of unknowns and $b \in \mathbb{R}^m$ is the right-hand side vector.

- **Row interpretation**: the $i$-th equation has the form $A_{i*} x = b_i$ and describes a hyperplane. We seek the intersection of all hyperplanes.
- **Column interpretation** (from property 5): $Ax = b$ means $x_1 A_{*1} + x_2 A_{*2} + \ldots + x_n A_{*n} = b$. We seek to express the vector $b$ as a combination of columns of the matrix $A$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Matrix as a mapping)</span></p>

A matrix $A \in \mathbb{R}^{m \times n}$ can be understood as a mapping $x \mapsto Ax$ from $\mathbb{R}^n$ to $\mathbb{R}^m$. Solving the system $Ax = b$ then means finding all vectors $x$ that are mapped to the vector $b$.

The composition of two mappings $x \mapsto Ax$ and $y \mapsto By$ corresponds to matrix multiplication: $x \mapsto B(Ax) = (BA)x$. This is precisely why matrix multiplication is defined in this way.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Matrix as a mapping in the plane)</span></p>

- **Reflection across the $x_2$-axis**: $A = \bigl(\begin{smallmatrix} -1 & 0 \\ 0 & 1 \end{smallmatrix}\bigr)$, mapping $(x_1, x_2)^\top \mapsto (-x_1, x_2)^\top$.
- **Stretching in the $x_1$ direction**: $A = \bigl(\begin{smallmatrix} 2.5 & 0 \\ 0 & 1 \end{smallmatrix}\bigr)$, mapping $(x_1, x_2)^\top \mapsto (2.5 x_1, x_2)^\top$.
- **Rotation by angle $\alpha$**: $A = \bigl(\begin{smallmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{smallmatrix}\bigr)$.

Composition of mappings corresponds to matrix multiplication, but generally depends on the order (matrix multiplication is not commutative).

</div>

#### Block multiplication and computational complexity

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Block matrix multiplication)</span></p>

Matrices can be partitioned into blocks (submatrices) and then multiplied as if the submatrices were ordinary numbers:

$$AB = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix} \begin{pmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{pmatrix} = \begin{pmatrix} A_{11}B_{11} + A_{12}B_{21} & A_{11}B_{12} + A_{12}B_{22} \\ A_{21}B_{11} + A_{22}B_{21} & A_{21}B_{12} + A_{22}B_{22} \end{pmatrix}.$$

It is necessary that the submatrices have compatible dimensions so that the products and sums on the right-hand side are well-defined.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Computational complexity of matrix operations)</span></p>

- **Sum** $A + B$: $n^2$ operations.
- **Product** $AB$ for $A, B \in \mathbb{R}^{n \times n}$: a total of $2n^3$ arithmetic operations (standard method).

Strassen's algorithm (1969) reduces the complexity to $\approx n^{2.807}$ operations. Coppersmith--Winograd (1990) further reduces it to $\approx n^{2.376}$. However, these algorithms are only practical for large $n$. The smallest possible asymptotic complexity of matrix multiplication remains an open problem.

</div>

### Nonsingular matrices

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Nonsingular matrix)</span></p>

Let $A \in \mathbb{R}^{n \times n}$. The matrix $A$ is *nonsingular* (or *regular*) if the system $Ax = 0$ has the unique solution $x = 0$. Otherwise, the matrix $A$ is called *singular*.

The system $Ax = 0$ with zero right-hand side is called *homogeneous*. The zero vector is always a solution. For a nonsingular $A$, however, no other solution exists, i.e., $Ax \neq 0$ for all $x \neq 0$. A typical example of a nonsingular matrix is $I_n$, and of a singular matrix is $0_n$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 3.27 — Equivalent characterizations of nonsingularity)</span></p>

Let $A \in \mathbb{R}^{n \times n}$. Then the following are equivalent:

1. $A$ is nonsingular,
2. $\operatorname{RREF}(A) = I_n$,
3. $\operatorname{rank}(A) = n$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 3.28 — Nonsingular matrices and solvability)</span></p>

Let $A \in \mathbb{R}^{n \times n}$. Then the following are equivalent:

1. $A$ is nonsingular,
2. for some $b \in \mathbb{R}^n$, the system $Ax = b$ has a unique solution,
3. for **every** $b \in \mathbb{R}^n$, the system $Ax = b$ has a unique solution.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 3.29 — Product of nonsingular matrices)</span></p>

Let $A, B \in \mathbb{R}^{n \times n}$ be nonsingular matrices. Then $AB$ is also nonsingular.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $x$ be a solution of the system $ABx = 0$. We want to show that $x$ must be the zero vector. Denote $y := Bx$. Then the system becomes $Ay = 0$. By the nonsingularity of $A$, the only solution is $y = 0$, which gives $Bx = 0$. By the nonsingularity of $B$, we get $x = 0$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 3.30 — Singularity of a product)</span></p>

If at least one of the matrices $A, B \in \mathbb{R}^{n \times n}$ is singular, then $AB$ is also singular.

</div>

#### Elementary operation matrices

Elementary row operations can be represented using matrices — the result of an operation on a matrix $A$ can be expressed as $EA$ for some matrix $E$. The matrix $E$ is obtained by applying the given operation to the identity matrix $I_n$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Elementary operation matrices)</span></p>

1. **Multiplying the $i$-th row by $\alpha \neq 0$**: $E_i(\alpha) = I + (\alpha - 1)e_i e_i^\top$ (on the diagonal in the $i$-th row there is $\alpha$ instead of $1$).
2. **Adding $\alpha$ times the $j$-th row to the $i$-th row**: $E_{ij}(\alpha) = I + \alpha e_i e_j^\top$ ($i \neq j$; at position $(i, j)$ there is $\alpha$).
3. **Swapping the $i$-th and $j$-th rows**: $E_{ij} = I + (e_j - e_i)(e_i - e_j)^\top$.

All elementary operation matrices are nonsingular. Each elementary operation has its inverse operation, which transforms the matrix back to the identity matrix.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 3.31 — RREF as a product)</span></p>

Let $A \in \mathbb{R}^{m \times n}$. Then $\operatorname{RREF}(A) = QA$ for some nonsingular matrix $Q \in \mathbb{R}^{m \times m}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 3.32 — Nonsingular matrix as a product of elementary matrices)</span></p>

Every nonsingular matrix $A \in \mathbb{R}^{n \times n}$ can be expressed as a product of finitely many elementary matrices.

</div>

### Inverse matrix

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Inverse matrix)</span></p>

Let $A \in \mathbb{R}^{n \times n}$. Then $A^{-1}$ is the *inverse matrix* of $A$ if it satisfies $AA^{-1} = A^{-1}A = I_n$.

The inverse of $I_n$ is again $I_n$. The inverse of the zero matrix $0_n$ clearly does not exist.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 3.34 — Existence of the inverse matrix)</span></p>

Let $A \in \mathbb{R}^{n \times n}$. If $A$ is nonsingular, then its inverse matrix exists and is uniquely determined. Conversely, if an inverse of $A$ exists, then $A$ must be nonsingular.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Existence.** From the nonsingularity of $A$ it follows that the system $Ax = e_j$ has a unique solution for each $j = 1, \ldots, n$; denote it $x_j$. Form the matrix $A^{-1} = (x_1 \mid x_2 \mid \ldots \mid x_n)$. Then $(AA^{-1})_{*j} = Ax_j = e_j = I_{*j}$, so $AA^{-1} = I$. The second equality $A^{-1}A = I$ is proved by a trick: the matrix $A(A^{-1}A - I)$ is zero, so its $j$-th column satisfies $A(A^{-1}A - I)_{*j} = 0$. By the nonsingularity of $A$, we get $(A^{-1}A - I)_{*j} = 0$ for every $j$, hence $A^{-1}A = I$.

**Uniqueness.** Suppose for some matrix $B$ we have $AB = BA = I$. Then $B = BI = B(AA^{-1}) = (BA)A^{-1} = IA^{-1} = A^{-1}$.

**Converse.** Suppose an inverse of $A$ exists. Let $x$ be a solution of $Ax = 0$. Then $x = Ix = (A^{-1}A)x = A^{-1}(Ax) = A^{-1}0 = 0$. Hence $A$ is nonsingular.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 3.35)</span></p>

If $A$ is nonsingular, then $A^\top$ is nonsingular and $(A^\top)^{-1} = (A^{-1})^\top$, which is sometimes abbreviated as $A^{-T}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 3.36 — One equality suffices)</span></p>

Let $A, B \in \mathbb{R}^{n \times n}$. If $BA = I_n$, then both matrices $A, B$ are nonsingular and are inverses of each other, that is $B = A^{-1}$ and $A = B^{-1}$.

</div>

#### Computing the inverse matrix

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 3.37 — Computing the inverse matrix)</span></p>

Let $A \in \mathbb{R}^{n \times n}$. Suppose the matrix $(A \mid I_n)$ of type $n \times 2n$ has RREF form $(I_n \mid B)$. Then $B = A^{-1}$. If the first part of the RREF is not $I_n$, then $A$ is singular.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Computing the inverse matrix)</span></p>

Let $A = \begin{pmatrix} 1 & 1 & 3 \\ 0 & 2 & -1 \\ 3 & 5 & 7 \end{pmatrix}$. We compute the inverse matrix:

$$(A \mid I_3) = \begin{pmatrix} 1 & 1 & 3 & 1 & 0 & 0 \\ 0 & 2 & -1 & 0 & 1 & 0 \\ 3 & 5 & 7 & 0 & 0 & 1 \end{pmatrix} \xrightarrow{\text{RREF}} \begin{pmatrix} 1 & 0 & 0 & -9.5 & -4 & 3.5 \\ 0 & 1 & 0 & 1.5 & 1 & -0.5 \\ 0 & 0 & 1 & 3 & 1 & -1 \end{pmatrix} = (I_3 \mid A^{-1}).$$

Thus $A^{-1} = \begin{pmatrix} -9.5 & -4 & 3.5 \\ 1.5 & 1 & -0.5 \\ 3 & 1 & -1 \end{pmatrix}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Proposition 3.39 — Properties of the inverse matrix)</span></p>

Let $A, B \in \mathbb{R}^{n \times n}$ be nonsingular. Then:

1. $(A^{-1})^{-1} = A$,
2. $(A^{-1})^\top = (A^\top)^{-1}$,
3. $(\alpha A)^{-1} = \frac{1}{\alpha} A^{-1}$ pro $\alpha \neq 0$,
4. $(AB)^{-1} = B^{-1} A^{-1}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 3.40 — System of equations and the inverse matrix)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ be nonsingular. Then the solution of the system $Ax = b$ is given by the formula $x = A^{-1}b$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark on practical use)</span></p>

In practice, we do not use the formula $x = A^{-1}b$, since it is computationally more expensive than Gaussian elimination. The significance of Theorem 3.40 is rather theoretical — it provides an explicit formula for the solution.

Left-multiplying the system $Ax = b$ by a nonsingular matrix $Q$ does not change the solution set: from the system $(QA)x = Qb$ one can return to the original by multiplying by $Q^{-1}$. Every nonsingular matrix can be decomposed into elementary matrices (Proposition 3.32), so elementary row operations indeed do not change the solution set.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric interpretation of nonsingularity)</span></p>

The mapping $x \mapsto Ax$ with a nonsingular matrix $A \in \mathbb{R}^{n \times n}$ is a bijection — every vector in $\mathbb{R}^n$ has a unique preimage. The inverse mapping is $y \mapsto A^{-1}y$. Examples: rotation, reflection, stretching (nonsingular). Conversely, a singular matrix collapses the space — the projection $A = \bigl(\begin{smallmatrix} 1 & 0 \\ 0 & 0 \end{smallmatrix}\bigr)$ maps the entire plane onto the $x_1$-axis, and therefore has no inverse.

</div>

#### Sherman--Morrison formula

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Sherman--Morrison formula — Theorem 3.44)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ be nonsingular and let $b, c \in \mathbb{R}^n$. If $c^\top A^{-1} b = -1$, then $A + bc^\top$ is singular; otherwise

$$(A + bc^\top)^{-1} = A^{-1} - \frac{1}{1 + c^\top A^{-1} b} A^{-1} b c^\top A^{-1}.$$

</div>

This formula allows us to quickly recompute the inverse matrix when we make a "small" change to the original matrix (rank-one update). If we know $A^{-1}$, then computing $(A + bc^\top)^{-1}$ requires only $6n^2$ operations instead of $\sim n^3$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Computational complexity of inversion)</span></p>

The computational complexity of inverting a matrix $A \in \mathbb{R}^{n \times n}$ is determined by the complexity of the RREF algorithm on the matrix $(A \mid I_n)$, which is of order $3n^3$ operations. In fact, the procedure can be improved so that the total complexity is $2n^3$, i.e., the same asymptotic complexity as matrix multiplication.

</div>

### LU decomposition

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(LU decomposition)</span></p>

The *LU decomposition* of a square matrix $A \in \mathbb{R}^{n \times n}$ is a factorization $A = LU$, where $L$ is a lower triangular matrix with ones on the diagonal and $U$ is an upper triangular matrix.

</div>

The LU decomposition is closely related to the echelon form of a matrix. We can take $U$ to be the echelon form of $A$ and obtain $L$ from the elementary operations. If during elimination we only use addition of a multiple of a row to a row below it (without row swaps), then the matrices of such operations $E_{ij}(\alpha)$ are lower triangular and their product $L$ is again a lower triangular matrix.

Reduce $A$ to echelon form $U$: $E_k \ldots E_1 A = U$, from which $A = \underbrace{E_1^{-1} \ldots E_k^{-1}}_{L} U$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(LU decomposition)</span></p>

$$A = \begin{pmatrix} 2 & 1 & 3 \\ 4 & 1 & 7 \\ -6 & -2 & -12 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ -3 & -1 & 1 \end{pmatrix} \begin{pmatrix} 2 & 1 & 3 \\ 0 & -1 & 1 \\ 0 & 0 & -2 \end{pmatrix} = LU.$$

The matrix $L$ is constructed by writing the coefficients $-\alpha$ from the elementary operations in place of the zeros below the diagonal.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Using LU decomposition to solve a system)</span></p>

To solve the system $Ax = b$ (i.e., $LUx = b$):

1. Find the LU decomposition of the matrix $A$, i.e., $A = LU$.
2. Solve the system $Ly = b$ by forward substitution.
3. Solve the system $Ux = y$ by back substitution.

The computational complexity of the entire algorithm is asymptotically the same as Gaussian elimination ($\frac{2}{3}n^3$).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 3.49 — PA = LU)</span></p>

Every matrix $A \in \mathbb{R}^{n \times n}$ can be decomposed in the form $PA = LU$, where $P \in \mathbb{R}^{n \times n}$ is a *permutation matrix* (a matrix with ones on the diagonal after a suitable reordering of rows), $L \in \mathbb{R}^{n \times n}$ is a lower triangular matrix with ones on the diagonal, and $U \in \mathbb{R}^{n \times n}$ is an upper triangular matrix.

</div>

The LU decomposition without row swaps may not exist for every matrix (e.g., for $A = \bigl(\begin{smallmatrix} 0 & 1 \\ 1 & 0 \end{smallmatrix}\bigr)$), but after a suitable permutation of rows (permutation matrix $P$) it always does.

### Numerical stability in solving systems, iterative methods

#### Ill-conditioned matrices

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Numerical stability)</span></p>

When solving systems of linear equations numerically on computers, rounding errors occur and the computed result may differ drastically from the correct solution. Errors are particularly pronounced for so-called **ill-conditioned matrices** — matrices that are in some sense close to singular matrices.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Ill-conditioned system)</span></p>

Two systems differing in a single coefficient by $\frac{2}{30}$:

$$0.835 x_1 + 0.667 x_2 = 0.168, \quad 0.333 x_1 + 0.266 x_2 = 0.067$$

has the solution $(1, -1)$, while

$$0.835 x_1 + 0.667 x_2 = 0.168, \quad 0.333 x_1 + 0.266 x_2 = 0.066$$

has the solution $(-666, 834)$. Geometrically, this is the intersection of two nearly parallel lines — a small change in the data leads to a huge change in the intersection point.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Hilbert matrix)</span></p>

A typical example of an ill-conditioned matrix is the Hilbert matrix $H_n$ of order $n$, defined by $(H_n)_{ij} = \frac{1}{i + j - 1}$. Already for $n \approx 14$, numerical errors have an enormous impact on the accuracy of the solution (with double precision $\approx 10^{-16}$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Partial and complete pivoting)</span></p>

**Partial pivoting** (choosing the pivot with the largest absolute value in the column below the current position) often leads to a more accurate solution, although it is not a universal remedy either. **Complete pivoting** searches for the element with the largest absolute value in the entire lower-right submatrix — it improves numerical properties but is computationally more expensive and rarely used in practice.

</div>

#### Iterative methods

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sparse systems and iterative methods)</span></p>

Large practical problems (typically solving differential equations) lead to large but **sparse** systems $Ax = b$ (matrix order $n = 10^7$, but most entries are zero). Gaussian elimination is not suitable because elementary operations increase the proportion of nonzero entries ("fill in" the matrix) and, moreover, elimination does not exploit sparsity.

**Iterative methods** start from an initial vector and gradually converge to the solution. They are less sensitive to rounding errors, have lower memory requirements, and are well-suited for sparse matrices. One iteration requires on the order of $kn$ operations (where $k$ is the number of nonzero entries per row), while Gaussian elimination requires $\frac{2}{3}n^3$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gauss--Seidel method)</span></p>

Consider the system:

$$6x + 2y - z = 4, \quad x + 5y + z = 3, \quad 2x + y + 4z = 27.$$

We rewrite: $x = \frac{1}{6}(4 - 2y + z)$, $y = \frac{1}{5}(3 - x - z)$, $z = \frac{1}{4}(27 - 2x - y)$.

Starting with the initial guess $(1, 1, 1)$, we iterate, and after 6 iterations we have an approximate solution close to the exact one $(2, -1, 6)^\top$:

| iteration | $x$ | $y$ | $z$ |
| --- | --- | --- | --- |
| 0 | 1 | 1 | 1 |
| 1 | 0.5 | 0.3 | 6.425 |
| 2 | 1.6375 | $-1.0125$ | 6.184375 |
| 6 | 1.999624 | $-0.999895$ | 6.000012 |

Convergence is guaranteed only for certain classes of matrices, e.g., when in each row the diagonal entry is larger than the sum of the absolute values of the remaining entries.

</div>

### Applications

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Leontief input-output model)</span></p>

Consider an economy with $n$ sectors. Sector $i$ produces one commodity in quantity $x_i$. Producing one unit of the $j$-th commodity requires $a_{ij}$ units of the $i$-th commodity. Let $d_i$ denote the final demand for the output of sector $i$. The model:

$$x_i = a_{i1} x_1 + \ldots + a_{in} x_n + d_i, \quad \text{i.e.,} \quad (I_n - A)x = d.$$

The solution has the explicit expression $x = (I_n - A)^{-1} d$. Leontief applied this model to the US economy in the 1940s and received the Nobel Prize in 1973 for this input-output model.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Polynomial interpolation)</span></p>

Given $n + 1$ points $(x_0, y_0), (x_1, y_1), \ldots, (x_n, y_n)$ in the plane, where $x_i \neq x_j$ for $i \neq j$, we seek a polynomial $p(x) = a_n x^n + \ldots + a_1 x + a_0$ passing through these points. Substituting, we obtain a system of equations with the **Vandermonde matrix**:

$$\begin{pmatrix} x_0^n & \ldots & x_0 & 1 \\ x_1^n & \ldots & x_1 & 1 \\ \vdots & & \vdots & \vdots \\ x_n^n & \ldots & x_n & 1 \end{pmatrix} \begin{pmatrix} a_n \\ \vdots \\ a_1 \\ a_0 \end{pmatrix} = \begin{pmatrix} y_0 \\ y_1 \\ \vdots \\ y_n \end{pmatrix}.$$

The Vandermonde matrix is nonsingular (for distinct $x_i$), and therefore a polynomial of degree $n$ always exists and is uniquely determined.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Discrete and fast Fourier transform)</span></p>

A polynomial $p(x)$ can be represented in two ways: (1) by its coefficients $a_n, \ldots, a_0$, or (2) by a list of function values at $n + 1$ distinct points. In the first representation, addition is simple ($\sim n$ operations), but multiplication costs $\sim 2n^2$. In the second, both addition and multiplication take $\sim n$ operations.

The **fast Fourier transform** (FFT) allows switching between both representations in $\sim \alpha n \log(n)$ operations, and thus also multiplying polynomials (and hence real numbers) efficiently.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Image compression — Haar transform)</span></p>

An image represented by a matrix $M \in \mathbb{R}^{m \times n}$ can be compressed using the Haar transform. The matrix $M$ is divided into $8 \times 8$ submatrices, and to each we apply the transformation $A' = H^\top A H$, where $H$ is a nonsingular matrix. By averaging neighboring pixels, values close to zero are set to zero — this produces a sparse matrix that can be stored more efficiently. The *compression ratio* $k$ gives the ratio of nonzero numbers in $A'$ before and after zeroing out small values. A higher $k$ means greater compression, but also greater loss of information.

</div>

## Chapter 4 — Groups and Fields

This chapter is devoted to basic algebraic structures — groups and fields. These are abstract concepts generalizing the well-known domains of real (rational, complex, etc.) numbers with the operations of addition and multiplication.

### 4.1 Groups

A group is a very abstract algebraic structure. It consists of a set with a binary operation that must satisfy several basic properties. Groups are used to describe symmetries of (not only geometric) objects. Thanks to their generality and abstractness, groups can be found in many different areas: physics (Lie groups), architecture (frieze groups), geometry and molecular chemistry (symmetric groups), etc.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 4.1 — Group)</span></p>

Let $\circ \colon G^2 \to G$ be a binary operation on a set $G$. Then a *group* is a pair $(G, \circ)$ satisfying:

1. $\forall a, b, c \in G: a \circ (b \circ c) = (a \circ b) \circ c$ &emsp; (associativity),
2. $\exists e \in G\ \forall a \in G: e \circ a = a \circ e = a$ &emsp; (existence of an identity element),
3. $\forall a \in G\ \exists b \in G: a \circ b = b \circ a = e$ &emsp; (existence of an inverse element).

An **Abelian (commutative) group** is a group that additionally satisfies:

4. $\forall a, b \in G: a \circ b = b \circ a$ &emsp; (commutativity).

</div>

The definition of a group implicitly contains the closure condition, so that the result of the operation does not fall outside the set $G$. If the operation $\circ$ is addition, the identity element is usually denoted $0$ and the inverse $-a$; if it is multiplication, the identity element is denoted $1$ and the inverse $a^{-1}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 4.2 — Definitions by construction vs. axioms)</span></p>

A mathematical object can be introduced either by construction from some already created objects, or by specifying properties (axioms) that it must satisfy. The definition of a group falls into the second category. A group is then any object that satisfies the given properties. An axiomatic definition has the advantage that it does not tie us to one specific object — any property that we derive for an axiomatically defined object then automatically holds for every concrete instance.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 4.3 — Examples of groups)</span></p>

- The well-known domains of integers $(\mathbb{Z}, +)$, rational numbers $(\mathbb{Q}, +)$, real numbers $(\mathbb{R}, +)$, and complex numbers $(\mathbb{C}, +)$. The identity element is $0$, and the inverse of an element $a$ is $-a$. Commutativity and associativity of addition clearly hold.
- Matrix groups $(\mathbb{R}^{m \times n}, +)$. The identity element is the zero matrix $0$ of size $m \times n$, and the inverse of a matrix $A$ is $-A$. Commutativity and associativity of addition hold by Proposition 3.5.
- The finite group $(\mathbb{Z}_n, +)$, where the set $\mathbb{Z}_n := \lbrace 0, 1, \ldots, n - 1 \rbrace$ and addition is performed modulo $n$. The identity element is $0$, and the inverse of an element $a$ is $-a \bmod n$.
- Number domains with multiplication, e.g. $(\mathbb{Q} \setminus \lbrace 0 \rbrace, \cdot)$, $(\mathbb{R} \setminus \lbrace 0 \rbrace, \cdot)$. We must exclude zero because it has no inverse element. The identity element is now $1$, and the inverse of an element $a$ is $a^{-1}$.
- The set of real polynomials in the variable $x$ with addition.
- Mappings on a set with the operation of composition, e.g. rotations in $\mathbb{R}^n$ about the origin or the permutations discussed later (Section 4.2). Rotations in the plane $\mathbb{R}^2$ are still commutative, but in higher dimensions commutativity is lost. The identity element is rotation by zero angle, and the inverse element is rotation by the opposite angle back.
- Invertible matrices of fixed order $n$ with multiplication (the so-called matrix group). The identity element is $I_n$, and the inverse of a matrix $A$ is the inverse matrix $A^{-1}$. Associativity of matrix multiplication was established in Proposition 3.9.

**Examples of non-groups:** $(\mathbb{N}, +)$, $(\mathbb{Z}, -)$, $(\mathbb{R} \setminus \lbrace 0 \rbrace, :)$, ...

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 4.4 — Basic properties in a group)</span></p>

For elements of a group $(G, \circ)$, the following properties hold:

1. $a \circ c = b \circ c$ implies $a = b$ &emsp; (cancellation law),
2. the identity element $e$ is uniquely determined,
3. for each $a \in G$, its inverse element is uniquely determined,
4. the equation $a \circ x = b$ has exactly one solution for every $a, b \in G$,
5. $(a^{-1})^{-1} = a$,
6. $(a \circ b)^{-1} = b^{-1} \circ a^{-1}$.

*Proof.* (1) From $a \circ c = b \circ c$ we compose with $c^{-1}$ on the right: $a \circ (c \circ c^{-1}) = b \circ (c \circ c^{-1})$, so $a \circ e = b \circ e$, i.e. $a = b$. (2) If there exist two distinct identity elements $e_1, e_2$, then $e_1 = e_1 \circ e_2 = e_2$, a contradiction. (3) If there exist two distinct inverse elements $a_1, a_2$, then $a \circ a_1 = e = a \circ a_2$ and by the cancellation law $a_1 = a_2$, a contradiction. (4) We multiply $a \circ x = b$ on the left by $a^{-1}$ and obtain $x = a^{-1} \circ b$. Substituting back verifies that the equality holds.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 4.5 — Subgroup)</span></p>

A *subgroup* of a group $(G, \circ)$ is a group $(H, \diamond)$ such that $H \subseteq G$ and for all $a, b \in H$ we have $a \circ b = a \diamond b$. Notation: $(H, \circ) \le (G, \circ)$.

In other words, with the same operation, $H$ satisfies the properties of closure and existence of the identity and inverse elements. That is, for every $a, b \in H$ we have $a \circ b \in H$, furthermore $e \in H$, and for every $a \in H$ we have $a^{-1} \in H$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 4.6)</span></p>

- Every group $(G, \circ)$ has two trivial subgroups: itself $(G, \circ)$ and $(\lbrace e \rbrace, \circ)$.
- $(\mathbb{N}, +) \not\le (\mathbb{Z}, +) \le (\mathbb{Q}, +) \le (\mathbb{R}, +) \le (\mathbb{C}, +)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 4.7)</span></p>

Subgroups are closed under intersection, but not under union. In other words, the intersection of two subgroups of a group $(G, \circ)$ is again a subgroup, but the union of subgroups is not necessarily a subgroup.

</div>

### 4.2 Permutations

Another example of groups is the so-called symmetric group of permutations, so let us say more about permutations.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 4.8 — Permutation)</span></p>

A *permutation* on a finite set $X$ is a bijection $p \colon X \to X$.

</div>

We will mostly consider $X = \lbrace 1, \ldots, n \rbrace$. The set of all permutations on the set $\lbrace 1, \ldots, n \rbrace$ is denoted $S_n$. A permutation can be specified, for example, by:

- A **table**, where the top row contains the preimages and the bottom row their images.
- A **graph** indicating where each element maps to.
- **Decomposition into cycles**: $p = (1, 2)(3)(4, 5, 6)$, where each parenthesized expression $(a_1, \ldots, a_k)$ means that $a_1$ maps to $a_2$, $a_2$ maps to $a_3$, etc., up to $a_{k-1}$ maps to $a_k$ and $a_k$ maps to $a_1$. From the definition it is clear that every permutation can be decomposed into disjoint cycles. In the following text we will most often use the reduced notation, in which we omit cycles of length $1$.

An example of a simple but nontrivial permutation is a *transposition* $t = (i, j)$ — a permutation with a single cycle of length $2$ that swaps two elements. The simplest permutation is the identity $id$ mapping every element to itself.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 4.9 — Inverse permutation)</span></p>

Let $p \in S_n$. The *inverse permutation* of $p$ is the permutation $p^{-1}$ defined by $p^{-1}(i) = j$ if $p(j) = i$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 4.10)</span></p>

$(i, j)^{-1} = (i, j)$, $(i, j, k)^{-1} = (k, j, i)$, ...

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 4.11 — Composition of permutations)</span></p>

Let $p, q \in S_n$. The *composed permutation* $p \circ q$ is the permutation defined by $(p \circ q)(i) = p(q(i))$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 4.12)</span></p>

$id \circ p = p \circ id = p$, $p \circ p^{-1} = p^{-1} \circ p = id$, ...

</div>

Composition of permutations is associative (as is any mapping), but it is not commutative in general. For example, for $p = (1, 2)$, $q = (1, 3, 2)$ we have $p \circ q = (1, 3)$, but $q \circ p = (2, 3)$.

#### Sign of a permutation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 4.13 — Sign of a permutation)</span></p>

Let a permutation $p \in S_n$ consist of $k$ cycles. Then the *sign of the permutation* is the number $\operatorname{sgn}(p) = (-1)^{n-k}$.

</div>

The sign is always $1$ or $-1$. Accordingly, permutations are classified as *even* (those with sign $1$) and *odd* (those with sign $-1$).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 4.15 — On the sign of composing a permutation with a transposition)</span></p>

Let $p \in S_n$ and let $t = (i, j)$ be a transposition. Then $\operatorname{sgn}(p) = -\operatorname{sgn}(t \circ p) = -\operatorname{sgn}(p \circ t)$.

*Proof.* We prove $\operatorname{sgn}(p) = -\operatorname{sgn}(t \circ p)$; the second equality is analogous. The permutation $p$ consists of several cycles. We distinguish two cases:

- Suppose $i, j$ belong to the same cycle, denoted $(i, u_1, \ldots, u_r, j, v_1, \ldots, v_s)$. Then $(i, j) \circ (i, u_1, \ldots, u_r, j, v_1, \ldots, v_s) = (i, u_1, \ldots, u_r)(j, v_1, \ldots, v_s)$, so the number of cycles increases by one.
- Suppose $i, j$ belong to two different cycles, e.g. $(i, u_1, \ldots, u_r)(j, v_1, \ldots, v_s)$. Then $(i, j) \circ (i, u_1, \ldots, u_r)(j, v_1, \ldots, v_s) = (i, u_1, \ldots, u_r, j, v_1, \ldots, v_s)$, so the number of cycles decreases by one.

In either case, the number of cycles changes by one, and consequently so does the resulting sign.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 4.16 — Decomposition into transpositions)</span></p>

Every permutation can be decomposed into a composition of transpositions. Any cycle $(u_1, \ldots, u_r)$ decomposes as

$$(u_1, \ldots, u_r) = (u_1, u_2) \circ (u_2, u_3) \circ (u_3, u_4) \circ \ldots \circ (u_{r-1}, u_r).$$

The decomposition into transpositions is not unique, not even the number of transpositions. Only their parity remains the same.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 4.17)</span></p>

We have $\operatorname{sgn}(p) = (-1)^r$, where $r$ is the number of transpositions in the decomposition of $p$ into transpositions.

*Proof.* This is a consequence of Theorem 4.15. We start from the identity, which is even. Each transposition changes the sign, so the resulting sign is $(-1)^r$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 4.18)</span></p>

Let $p, q \in S_n$. Then $\operatorname{sgn}(p \circ q) = \operatorname{sgn}(p) \operatorname{sgn}(q)$.

*Proof.* Let $p$ be decomposable into $r_1$ transpositions and $q$ into $r_2$ transpositions. Then $p \circ q$ can be composed of $r_1 + r_2$ transpositions. Hence $\operatorname{sgn}(p \circ q) = (-1)^{r_1 + r_2} = (-1)^{r_1}(-1)^{r_2} = \operatorname{sgn}(p)\operatorname{sgn}(q)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 4.19)</span></p>

Let $p \in S_n$. Then $\operatorname{sgn}(p) = \operatorname{sgn}(p^{-1})$.

*Proof.* We have $1 = \operatorname{sgn}(id) = \operatorname{sgn}(p \circ p^{-1}) = \operatorname{sgn}(p)\operatorname{sgn}(p^{-1})$, so $p$ and $p^{-1}$ must have the same sign.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 4.20 — Inversions and sign)</span></p>

Besides the number of cycles and the number of transpositions, the sign of a permutation $p$ can also be defined, for example, using the number of inversions. By an inversion we mean an ordered pair $(i, j)$ such that $i < j$ and $p(i) > p(j)$. If we denote the number of inversions of a permutation $p$ as $I(p)$, then $\operatorname{sgn}(p) = (-1)^{I(p)}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 4.21 — Symmetric group)</span></p>

The set of permutations $S_n$ with the composition operation $\circ$ forms a non-commutative group $(S_n, \circ)$, called the *symmetric group*. It can be shown that every group is isomorphic to some subgroup of a symmetric group (the so-called Cayley representation, which even generalizes to infinite groups). A similar role is played by matrix groups, since every finite group is isomorphic to some matrix subgroup (linear representation).

The group $(S_n, \circ)$ is called symmetric because it and its subgroups describe symmetries of various objects. For instance, an isosceles triangle is symmetric about its vertical axis, and this symmetry corresponds to the permutation $(2, 3)$. The symmetries of an equilateral triangle are reflections about the medians (corresponding to transpositions $(1, 2)$, $(2, 3)$, and $(1, 3)$) as well as rotations by $0°$, $120°$, and $240°$ (corresponding to permutations $id$, $(1, 2, 3)$, and $(1, 3, 2)$). All symmetries thus constitute the entire group $(S_3, \circ)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 4.22 — The 15 puzzle)</span></p>

Symmetric groups and the sign of a permutation are also used in the analysis of puzzles such as the 15 puzzle or the Rubik's cube. The 15 puzzle is a game consisting of a $4 \times 4$ grid and tiles numbered $1$ to $15$. One cell is empty, and by sliding adjacent tiles into the empty cell we change the arrangement of tiles. The goal is to reach an ascending arrangement of tiles through these moves.

If we number the individual cells as $1$ to $16$, then the tile configuration corresponds to some permutation $p \in S_{16}$ and moving a tile corresponds to composing $p$ with some transposition. If we denote by $(r, s)$ the position of the empty cell, then the value $h = (-1)^{r+s}\operatorname{sgn}(p)$ remains the same throughout the game. The target configuration has value $h = 1$, so initial configurations with $h = -1$ cannot be solvable.

</div>

### 4.3 Fields

Algebraic fields generalize the class of traditional number domains, such as the set of real numbers, to an abstract set with two operations and a number of properties. This allows us to work with matrices (add, multiply, invert, solve systems of equations, ...) over domains other than just $\mathbb{R}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 4.23 — Motivational)</span></p>

Consider a system of linear equations $Ax = b$ with an invertible matrix $A$. The system therefore has a unique solution. If the entries of the matrix $(A \mid b)$ are integers, then the solution may not have integer components, because division occurs during the elimination. However, if the entries of the matrix $(A \mid b)$ are rational numbers, then the solution also has rational components, because ordinary matrix operations only involve arithmetic operations with numbers. The set of rational numbers $\mathbb{Q}$ thus has the property that ordinary matrix operations do not take us outside of $\mathbb{Q}$. A similar property is also enjoyed by the set of real numbers $\mathbb{R}$ and the set of complex numbers $\mathbb{C}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 4.24 — Field)</span></p>

A *field* is a set $\mathbb{T}$ together with two commutative binary operations $+$ and $\cdot$ satisfying

1. $(\mathbb{T}, +)$ is an Abelian group, whose identity element we denote $0$ and the inverse of $a$ by $-a$,
2. $(\mathbb{T} \setminus \lbrace 0 \rbrace, \cdot)$ is an Abelian group, whose identity element we denote $1$ and the inverse of $a$ by $a^{-1}$,
3. $\forall a, b, c \in \mathbb{T}: a \cdot (b + c) = a \cdot b + a \cdot c$ &emsp; (distributivity).

Every field has at least two elements, because the definition necessarily implies $0 \neq 1$. The operations $+$ and $\cdot$ do not necessarily represent classical addition and multiplication, but this notation is used for correspondence with the standard number domains. For this reason, we will also write $ab$ instead of $a \cdot b$ for brevity.

</div>

The inverse element property in the group $(\mathbb{T}, +)$ naturally introduces the operation "$-$" defined as addition of the inverse element, i.e. $a - b \equiv a + (-b)$. Analogously, the inverse element property in the group $(\mathbb{T} \setminus \lbrace 0 \rbrace, \cdot)$ naturally introduces the operation "$/$" defined as multiplication by the inverse element, i.e. $a / b \equiv ab^{-1}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 4.25)</span></p>

Examples of infinite fields include $\mathbb{Q}$, $\mathbb{R}$, and $\mathbb{C}$ with the usual operations of addition and multiplication. However, the set of integers $\mathbb{Z}$ does not form a field, because multiplicative inverses are missing (e.g. when we invert an integer matrix, fractions often arise, and thus we leave the domain $\mathbb{Z}$). Numbers represented on a computer in floating-point arithmetic also do not form a field — the operations of addition and multiplication are not closed (if the result were a very large or very small number), and they are not even associative (due to rounding).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 4.26 — Quaternions)</span></p>

Another example of fields are *quaternions*. They are a generalization of complex numbers by adding two more imaginary units $j$ and $k$, whose squares equal $-1$ and which are additionally linked by the relation $ijk = -1$. While addition is defined naturally, multiplication is somewhat more complicated and commutativity no longer holds. Quaternions therefore form a non-commutative field. Quaternions are well suited for describing rotations in three-dimensional space and have found applications in robotics and quantum physics.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 4.27 — Basic properties in a field)</span></p>

For elements of a field, the following properties hold:

1. $0a = 0$,
2. $ab = 0$ implies that $a = 0$ or $b = 0$,
3. $-a = (-1)a$.

*Proof.* (1) We derive $0a = (0 + 0)a = 0a + 0a$; adding $(-0a)$ yields $0 = 0a$. (2) If $a = 0$, the statement holds. If $a \neq 0$, then $a^{-1}$ exists. Multiplying both sides of $ab = 0$ on the left by $a^{-1}$ gives $a^{-1}ab = a^{-1}0$, i.e. $1b = 0$, so $b = 0$. (3) We have $0 = 0a = (1 - 1)a = 1a + (-1)a = a + (-1)a$, hence $-a = (-1)a$.

</div>

The second property (and its proof) also tell us that when determining whether a structure forms a field, we do not need to verify closure of multiplication on the set $\mathbb{T} \setminus \lbrace 0 \rbrace$ — this property follows from the others.

#### Finite fields

Let us now look at finite fields. Already in Example 4.3 we introduced the set $\mathbb{Z}_n = \lbrace 0, 1, \ldots, n - 1 \rbrace$. The operations $+$ and $\cdot$ on this set are defined modulo $n$. It is easy to see that $\mathbb{Z}_2$ and $\mathbb{Z}_3$ are fields, but $\mathbb{Z}_4$ is not, since the element $2$ has no inverse $2^{-1}$. This result can be generalized.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Lemma 4.28)</span></p>

Let $n$ be a prime and let $0 \neq a \in \mathbb{Z}_n$. Then using multiplication modulo $n$ we have

$$\lbrace 0, 1, \ldots, n - 1 \rbrace = \lbrace 0a, 1a, \ldots, (n - 1)a \rbrace.$$

*Note.* In the set $\lbrace 0a, 1a, \ldots, (n - 1)a \rbrace$, all numbers $0, 1, \ldots, n - 1$ appear (not necessarily in this order), each exactly once.

*Proof.* For contradiction, suppose $ak = a\ell$ for some $k, \ell \in \mathbb{Z}_n$, $k \neq \ell$. Then we get $a(k - \ell) = 0$, so either $a$ or $k - \ell$ is divisible by $n$. This means either $a = 0$ or $k - \ell = 0$. But neither can occur, which is a contradiction.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 4.29)</span></p>

$\mathbb{Z}_n$ is a field if and only if $n$ is a prime.

*Proof.* If $n$ is composite, then $n = pq$, where $1 < p, q < n$. If $\mathbb{Z}_n$ were a field, then $pq = 0$ would imply by Proposition 4.27 that either $p = 0$ or $q = 0$, but neither holds.

If $n$ is prime, then all axioms from the definition of a field are easily verified. The only one that may require more work is the existence of the inverse $a^{-1}$ for any $a \neq 0$. But this follows easily from Lemma 4.28. Since $\lbrace 0, 1, \ldots, n - 1 \rbrace = \lbrace 0a, 1a, \ldots, (n - 1)a \rbrace$, the element $1$ must appear in the set on the right, and therefore there exists $b \in \mathbb{Z}_n \setminus \lbrace 0 \rbrace$ such that $ba = 1$. Hence $b = a^{-1}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 4.30 — The field $\mathbb{Z}_5$)</span></p>

For illustration, we present in tables the explicit expressions of both operations over the field $\mathbb{Z}_5$:

| $+$ | 0 | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|---|
| **0** | 0 | 1 | 2 | 3 | 4 |
| **1** | 1 | 2 | 3 | 4 | 0 |
| **2** | 2 | 3 | 4 | 0 | 1 |
| **3** | 3 | 4 | 0 | 1 | 2 |
| **4** | 4 | 0 | 1 | 2 | 3 |

| $\cdot$ | 0 | 1 | 2 | 3 | 4 |
|---------|---|---|---|---|---|
| **0** | 0 | 0 | 0 | 0 | 0 |
| **1** | 0 | 1 | 2 | 3 | 4 |
| **2** | 0 | 2 | 4 | 1 | 3 |
| **3** | 0 | 3 | 1 | 4 | 2 |
| **4** | 0 | 4 | 3 | 2 | 1 |

Commutativity manifests as the symmetry of the tables, the identity element copies the table header into the corresponding row and column, and multiplication by zero gives zero. The inverse element property manifests in the fact that in every row and column (except multiplication by zero), each element of the field appears exactly once.

The inverse elements of the field $\mathbb{Z}_5$ are then:

| $x$ | 0 | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|---|
| $-x$ | 0 | 4 | 3 | 2 | 1 |

| $x$ | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|
| $x^{-1}$ | 1 | 3 | 2 | 4 |

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 4.31 — The field $\mathbb{Z}_2$ and bits)</span></p>

The field $\mathbb{Z}_2$ is of particular importance to computer scientists, since it works with two elements $0$ and $1$, which can be viewed as computer bits. Many common bit operations can then be translated into the language of operations in the field $\mathbb{Z}_2$. It is easy to see that the addition operation in $\mathbb{Z}_2$ corresponds to the computer operation XOR, and multiplication corresponds to AND. Similarly, other logical operations can be expressed using operations in the field $\mathbb{Z}_2$. Consequently, any logic gate of a digital circuit represents some arithmetic expression over $\mathbb{Z}_2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 4.32 — Matrices over a field)</span></p>

We introduced systems of equations and matrix operations over the field of real numbers. However, nothing prevents us from extending these concepts and working over any other field. If $\mathbb{T}$ is a field, then $\mathbb{T}^{m \times n}$ will denote an $m \times n$ matrix with entries in the field $\mathbb{T}$. The only properties of real numbers that we used are precisely those that appear in the definition of a field. Therefore, all procedures and theory built in the previous Chapters 2 and 3 remain valid. We can, for example, solve systems of linear equations over any field using Gaussian elimination, discuss the invertibility of a matrix from $\mathbb{T}^{n \times n}$, or find its inverse.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 4.33 — Computing an inverse matrix over $\mathbb{Z}_5$)</span></p>

$$(A \mid I_3) = \begin{pmatrix} 1 & 2 & 3 & 1 & 0 & 0 \\ 2 & 0 & 4 & 0 & 1 & 0 \\ 3 & 3 & 4 & 0 & 0 & 1 \end{pmatrix} \sim \ldots \sim \begin{pmatrix} 1 & 0 & 0 & 2 & 4 & 2 \\ 0 & 1 & 0 & 1 & 0 & 3 \\ 0 & 0 & 1 & 4 & 2 & 4 \end{pmatrix} = (I_3 \mid A^{-1})$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 4.34 — How to find an inverse)</span></p>

A natural question when computing over the field $\mathbb{Z}_p$ is how to find the inverse of an element $x \in \mathbb{Z}_p \setminus \lbrace 0 \rbrace$. For small values of $p$, we can try $1, 2, \ldots, p - 1$ successively until we find the inverse of $x$. If $p$ is a very large prime, this approach is no longer efficient and one uses the so-called *extended Euclidean algorithm*, which finds $a, b \in \mathbb{Z}$ such that $ax + bp = 1$. From this equation we see that the desired inverse $x^{-1}$ is the element $a$, taken modulo $p$.

</div>

#### Sizes of finite fields

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 4.35 — On sizes of finite fields)</span></p>

Finite fields exist precisely of sizes $p^n$, where $p$ is a prime and $n \ge 1$.

</div>

We omit the proof, but we show the basic idea of how to construct a field of size $p^n$. Such a field is denoted by the symbol $\operatorname{GF}(p^n)$ (Galois field) and its elements are polynomials of degree at most $n - 1$ with coefficients in the field $\mathbb{Z}_p$, that is

$$\operatorname{GF}(p^n) = \lbrace a_{n-1}x^{n-1} + \ldots + a_1 x + a_0 ;\ a_0, \ldots, a_{n-1} \in \mathbb{Z}_p \rbrace.$$

Addition is defined analogously to real polynomials. However, multiplication could produce polynomials of degree higher than $n - 1$. Therefore, we first choose an arbitrary fixed irreducible polynomial of degree $n$, multiply the polynomials in the usual way, and then take the remainder upon division by this irreducible polynomial.

Another interesting property is that every finite field of size $p^n$ is isomorphic to $\operatorname{GF}(p^n)$, meaning that such fields are essentially the same up to relabeling of elements.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 4.36 — The field $\operatorname{GF}(8)$)</span></p>

The set has as its elements polynomials of degree at most two with coefficients in $\mathbb{Z}_2$:

$$\operatorname{GF}(8) = \lbrace 0,\ 1,\ x,\ x + 1,\ x^2,\ x^2 + 1,\ x^2 + x,\ x^2 + x + 1 \rbrace.$$

Addition is defined as $(a_2 x^2 + a_1 x + a_0) + (b_2 x^2 + b_1 x + b_0) = (a_2 + b_2)x^2 + (a_1 + b_1)x + (a_0 + b_0)$, e.g. $(x + 1) + (x^2 + x) = x^2 + 1$. Consider the irreducible polynomial $x^3 + x + 1$. Then we multiply modulo this polynomial, e.g. $x^2 \cdot x = -x - 1 = x + 1$, or $x^2 \cdot (x^2 + 1) = -x = x$.

</div>

#### Characteristic of a field

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 4.37 — Characteristic of a field)</span></p>

The *characteristic of a field* $\mathbb{T}$ is the smallest $n$ such that

$$\underbrace{1 + 1 + \ldots + 1}_{n} = 0.$$

If no such $n$ exists, we define it to be $0$.

</div>

For example, the infinite fields $\mathbb{Q}$, $\mathbb{R}$, or $\mathbb{C}$ have characteristic $0$, and the field $\mathbb{Z}_p$ has characteristic $p$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 4.38)</span></p>

The characteristic of a field is either zero or a prime.

*Proof.* Since $0 \neq 1$, the characteristic cannot be $1$. If the characteristic were a composite number $n = pq$, then

$$0 = \underbrace{1 + 1 + \ldots + 1}\_{n = pq} = (\underbrace{1 + \ldots + 1}\_{p})(\underbrace{1 + \ldots + 1}\_{q}),$$

so the sum of $p$ or $q$ ones equals zero, which contradicts the minimality of $n$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 4.39 — Average in a field)</span></p>

If the characteristic of a field $\mathbb{T}$ is not $2$, then we can introduce something like an average. Denoting by $2$ the value $1 + 1$, for any $a, b \in \mathbb{T}$ the element $p = \tfrac{1}{2}(a + b)$ has the property $a - p = p - b$, so it is equally "distant" from $a$ and from $b$.

A field with characteristic $2$ is $\mathbb{Z}_2$ or more generally any field $\operatorname{GF}(2^n)$. In these fields, the average of $0$ and $1$ cannot be defined, whereas for example in the field $\mathbb{Z}_5$ the average of $0$ and $1$ is $3$.

</div>

#### Fermat's little theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 4.40 — Fermat's little theorem)</span></p>

Let $p$ be a prime and let $0 \neq a \in \mathbb{Z}_p$. Then $a^{p-1} = 1$ in the field $\mathbb{Z}_p$.

*Proof.* By Lemma 4.28, $\lbrace 0, 1, \ldots, p - 1 \rbrace = \lbrace 0a, 1a, \ldots, (p - 1)a \rbrace$. Since $0 = 0a$, we obtain $\lbrace 1, \ldots, p - 1 \rbrace = \lbrace 1a, \ldots, (p - 1)a \rbrace$. Therefore $1 \cdot 2 \cdot 3 \cdot \ldots \cdot (p - 1) = (1a) \cdot (2a) \cdot (3a) \cdot \ldots \cdot (p - 1)a$. Canceling both sides by $1, 2, \ldots, p - 1$ yields the desired equality $1 = a \cdot \ldots \cdot a = a^{p-1}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 4.41)</span></p>

What is the value of $2^{111}$ in the field $\mathbb{Z}_{11}$? By Fermat's little theorem, $2^{10} = 1$, and therefore $2^{110} = 1$. Hence $2^{111} = 2^{110+1} = 2^{110} \cdot 2^1 = 2$.

</div>

### 4.4 Applications

Finite fields are used, for example, in coding theory and cryptography. To conclude this chapter, we demonstrate a practical application of fields in coding.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 4.42 — Error-correcting codes — Hamming code $(7, 4, 3)$)</span></p>

Consider the problem of transmitting data consisting of a sequence of zeros and ones. While the task of encryption is to transform data so that no unauthorized person can read it, the task of coding is to improve transmission properties. By this we mean primarily the ability to detect and correct errors that naturally arise during transmission.

The Hamming code $(7, 4, 3)$ consists of splitting the transmitted data into segments of four bits, which are encoded into seven bits. This code can detect and correct one transmission error. Encoding and decoding can be elegantly represented by matrix multiplication. We view a segment of four bits as an arithmetic vector $a$ over the field $\mathbb{Z}_2$. Encoding is performed by multiplying the vector $a$ by the so-called generator matrix $H \in \mathbb{Z}_2^{7 \times 4}$:

$$Ha = b.$$

The receiver obtains the segment represented by the vector $b$. The bits of the original data are at the highlighted positions $b_3, b_5, b_6, b_7$; the remaining bits $b_1, b_2, b_4$ are check bits. For error detection and correction, the receiver uses a detection matrix $D \in \mathbb{Z}_2^{3 \times 7}$. If $Db = 0$, no error occurred during transmission (or more than two errors occurred). Otherwise, a transmission error has occurred and the erroneous bit is at position $Db$, interpreting this vector as a binary representation of a natural number.

</div>

### Summary of Chapter 4

Groups represent the first axiomatically defined abstract concept that we have encountered. A group is any set on which we have an operation satisfying several basic properties (associativity, identity and inverse elements, and possibly commutativity). It is precisely this abstract definition that allows us to encompass a wide range of objects, thereby broadening the scope. As a notable example of a non-commutative group, we discussed permutations with the composition operation, the so-called symmetric group, since it was invented precisely for describing symmetries.

Algebraic fields are richer than groups by an additional operation. A field is thus a set with two operations satisfying certain properties. The matrix operations discussed in previous chapters can therefore be freely extended to work over any field, not just over $\mathbb{R}$; all results essentially remain valid. We know infinite fields such as $\mathbb{R}$ or $\mathbb{C}$, and finite fields such as the two-element field $\mathbb{Z}_2$, which is close to computer scientists. Finite fields exist precisely of sizes $p^n$, where $p$ is a prime.

## Chapter 5 — Vector Spaces

Vector spaces (in some fields also referred to as *linear spaces*) generalize the well-known space of arithmetic vectors $\mathbb{R}^n$. Just as with groups and fields, we define them using abstract axioms.

### 5.1 Basic Concepts

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.1 — Motivational)</span></p>

An ordered $n$-tuple of real numbers $v = (v_1, \ldots, v_n)$ has two possible interpretations in the Euclidean $n$-dimensional space $\mathbb{R}^n$, and we will use both for geometric intuition. We can view it as a specific point or as a vector. A vector specifies the direction from the origin $(0, \ldots, 0)$ to the point $(v_1, \ldots, v_n)$. We can perform the following operations with vectors:

- *Addition.* The sum of vectors is again a vector; for $u, v \in \mathbb{R}^n$ we have $u + v = (u_1 + v_1, \ldots, u_n + v_n)$. Addition is commutative and associative.
- *Scalar multiplication.* A scalar multiple of a vector is again a vector; for $\alpha \in \mathbb{R}$, $v \in \mathbb{R}^n$ we have $\alpha v = (\alpha v_1, \ldots, \alpha v_n)$. A scalar multiple of a vector points in the same direction (if $\alpha > 0$) or the opposite direction (if $\alpha < 0$). Basic properties such as distributivity over addition are satisfied.

</div>

There are additional operations possible with real arithmetic vectors, but we do not consider them for now. In our effort to generalize the notion of a vector and a space of vectors, we will naturally require similar properties to those mentioned above. That is, we want to be able to add vectors and multiply them by scalars (numbers), and we want these operations to satisfy basic axioms.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 5.2 — Vector Space)</span></p>

Let $\mathbb{T}$ be a field with identity elements $0$ for addition and $1$ for multiplication. By a *vector space over the field* $\mathbb{T}$ we mean a set $V$ with operations of vector addition $+ \colon V^2 \to V$ and scalar multiplication $\cdot \colon \mathbb{T} \times V \to V$ satisfying for all $\alpha, \beta \in \mathbb{T}$ and $u, v \in V$:

1. $(V, +)$ is an Abelian group, whose identity element we denote by $o$ and the inverse of $v$ by $-v$,
2. $\alpha(\beta v) = (\alpha \beta)v$ &emsp; (associativity),
3. $1v = v$,
4. $(\alpha + \beta)v = \alpha v + \beta v$ &emsp; (distributivity),
5. $\alpha(u + v) = \alpha u + \alpha v$ &emsp; (distributivity).

The elements of the vector space $V$ are called *vectors* and will be denoted by Latin letters. We write vectors without arrows, i.e., $v$ and not $\vec{v}$. The elements of the field $\mathbb{T}$ are called *scalars*, and to distinguish them we will denote them by Greek letters.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.3 — Examples of Vector Spaces)</span></p>

- The arithmetic space $\mathbb{R}^n$ over $\mathbb{R}$, or more generally $\mathbb{T}^n$ over $\mathbb{T}$, where $\mathbb{T}$ is an arbitrary field; $n$-tuples of elements from the field $\mathbb{T}$ are added and multiplied by scalars componentwise, similarly to $\mathbb{R}^n$. The axioms from the definition of a vector space then follow from the properties of the field.
- The space of matrices $\mathbb{R}^{m \times n}$ over $\mathbb{R}$, or more generally $\mathbb{T}^{m \times n}$ over $\mathbb{T}$. The axioms from the definition of a vector space are easily verified from the properties of matrices and fields.
- The space of all real polynomials in the variable $x$ over the field $\mathbb{R}$, denoted $\mathcal{P}$.
- The space of all real polynomials over $\mathbb{R}$ in the variable $x$ of degree at most $n$, denoted $\mathcal{P}^n$. The operations are defined in the standard way:
  - Addition: $(a_n x^n + \ldots + a_1 x + a_0) + (b_n x^n + \ldots + b_1 x + b_0) = (a_n + b_n)x^n + \ldots + (a_1 + b_1)x + (a_0 + b_0)$
  - Scalar multiplication: $\alpha(a_n x^n + \ldots + a_1 x + a_0) = (\alpha a_n)x^n + \ldots + (\alpha a_1)x + (\alpha a_0)$
  - Zero vector: $0$. Additive inverse: $-(a_n x^n + \ldots + a_0) = (-a_n)x^n + \ldots + (-a_0)$.
- The space of all real functions $f \colon \mathbb{R} \to \mathbb{R}$, denoted $\mathcal{F}$. Functions $f, g \colon \mathbb{R} \to \mathbb{R}$ are added by summing the corresponding function values, i.e., $(f + g)(x) = f(x) + g(x)$. Similarly, a function $f \colon \mathbb{R} \to \mathbb{R}$ is multiplied by a scalar $\alpha \in \mathbb{R}$ by multiplying all function values, i.e., $(\alpha f)(x) = \alpha f(x)$.
- The space of all continuous functions $f \colon \mathbb{R} \to \mathbb{R}$, denoted $\mathcal{C}$. The space of all continuous functions $f \colon [a, b] \to \mathbb{R}$ on the interval $[a, b]$ is denoted $\mathcal{C}_{[a,b]}$. The operations are defined analogously as for $\mathcal{F}$.

Unless stated otherwise, the spaces $\mathbb{R}^n$ and $\mathbb{R}^{m \times n}$ will henceforth be implicitly considered over the field $\mathbb{R}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 5.4 — Basic Properties of Vectors)</span></p>

In a space $V$ over a field $\mathbb{T}$, the following hold for every scalar $\alpha \in \mathbb{T}$ and vector $v \in V$:

1. $0v = o$,
2. $\alpha o = o$,
3. $\alpha v = o$ implies that $\alpha = 0$ or $v = o$,
4. $(-1)v = -v$.

*Proof.* Analogous to the properties in a field.

</div>

### 5.2 Subspaces and Linear Combinations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 5.5 — Subspace)</span></p>

Let $V$ be a vector space over $\mathbb{T}$. Then $U \subseteq V$ is a *subspace* of $V$ if it forms a vector space over $\mathbb{T}$ with the same operations. Notation: $U \le V$.

</div>

As the following proposition shows, an equivalent definition of a subspace is that it must contain the zero vector and be closed under both operations.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 5.6)</span></p>

Let $U$ be a subset of a vector space $V$ over $\mathbb{T}$. Then $U$ is a subspace of $V$ if and only if:

1. $o \in U$,
2. $\forall u, v \in U: u + v \in U$,
3. $\forall \alpha \in \mathbb{T}\ \forall u \in U: \alpha u \in U$.

*Proof.* If $U$ is a subspace of $V$, then it must satisfy the required three properties from the definition of a vector space. Conversely, suppose that $U$ satisfies the given three properties. The remaining properties from the definition of a vector space (such as commutativity, associativity, distributivity) also hold, because they hold for the set $V$, and therefore automatically hold for any of its subsets. The fact that $U$ is closed under additive inverses follows from closure under scalar multiples, since by Proposition 5.4 we have $-v = (-1)v$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.7 — Examples of Vector Subspaces)</span></p>

- Two trivial subspaces of $V$ are: $V$ and $\lbrace o \rbrace$.
- Any line in the plane passing through the origin is a subspace of $\mathbb{R}^2$; any other line is not.
- $\mathcal{P}^n \le \mathcal{P} \le \mathcal{C} \le \mathcal{F}$.
- The set of symmetric real matrices of order $n$ is a subspace of $\mathbb{R}^{n \times n}$.
- $\mathbb{Q}^n$ over $\mathbb{Q}$ is a subspace of $\mathbb{R}^n$ over $\mathbb{Q}$, but it is not a subspace of $\mathbb{R}^n$ over $\mathbb{R}$, because it operates over a different field.

Some properties of vector subspaces:

- If $U, V$ are subspaces of $W$ and $U \subseteq V$, then $U \le V$.
- The property "being a subspace" is transitive, i.e., $U \le V \le W$ implies $U \le W$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 5.8 — Intersection of Subspaces)</span></p>

Let $V$ be a vector space over $\mathbb{T}$, and let $V_i$, $i \in I$, be an arbitrary system of subspaces of $V$. Then $\bigcap_{i \in I} V_i$ is again a subspace of $V$.

*Proof.* By Proposition 5.6 it suffices to verify three properties: Since $o \in V_i$ for every $i \in I$, it must also be in their intersection. Closure under addition: Let $u, v \in \bigcap_{i \in I} V_i$, i.e., for every $i \in I$ we have $u, v \in V_i$, hence also $u + v \in V_i$. Therefore $u + v \in \bigcap_{i \in I} V_i$. Closure under scalar multiples is analogous.

</div>

#### Linear Span

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 5.9 — Linear Span)</span></p>

Let $V$ be a vector space over $\mathbb{T}$, and $W \subseteq V$. Then the *linear span* of $W$, denoted $\operatorname{span}(W)$, is the intersection of all subspaces of $V$ containing $W$, that is $\operatorname{span}(W) = \bigcap_{U: W \subseteq U \le V} U$.

</div>

The linear span of a set of vectors $W$ is therefore the smallest space containing $W$ in the sense that any other space containing $W$ is a superset of it.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.10 — Examples of Linear Spans)</span></p>

Examples of linear spans in the vector space $\mathbb{R}^2$:

- $\operatorname{span}\lbrace (1, 0)^\top \rbrace$ is a line, specifically the $x_1$-axis.
- $\operatorname{span}\lbrace (1, 0)^\top, (2, 0)^\top \rbrace$ is the same.
- $\operatorname{span}\lbrace (1, 1)^\top, (1, 2)^\top \rbrace$ is the entire plane $\mathbb{R}^2$.
- $\operatorname{span}\lbrace \rbrace = \lbrace o \rbrace$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 5.11 — Generators and Finitely Generated Space)</span></p>

Let the space $U$ be the linear span of a set of vectors $W$, i.e., $U = \operatorname{span}(W)$. Then we say that $W$ *generates* the space $U$, and the elements of $W$ are *generators* of the space $U$. The space $U$ is called *finitely generated* if it is generated by some finite set of vectors.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.12)</span></p>

Consider the vector space $\mathbb{R}^2$ and its subspace $U$ represented by the $x_1$-axis. This subspace can be generated by the vector $(1, 0)^\top$, or by the vector $(-3, 0)^\top$, or by any other vector of the form $(a, 0)^\top$ where $a \neq 0$. However, $U$ can also be generated by the set of vectors $\lbrace (2, 0)^\top, (5, 0)^\top \rbrace$. We see that this set is not minimal — one vector can be removed and the remaining vector still generates the subspace $U$. This pursuit of a minimal representation and the elimination of redundancies will later lead to the concept of a basis (Section 5.4).

</div>

#### Linear Combinations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 5.13 — Linear Combination)</span></p>

Let $V$ be a vector space over $\mathbb{T}$ and $v_1, \ldots, v_n \in V$. Then by a *linear combination* of vectors $v_1, \ldots, v_n$ we mean an expression of the form $\sum_{i=1}^{n} \alpha_i v_i = \alpha_1 v_1 + \ldots + \alpha_n v_n$, where $\alpha_1, \ldots, \alpha_n \in \mathbb{T}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 5.14)</span></p>

It should be emphasized here that we only consider linear combinations of finitely many vectors. This fully suffices for our purposes, since we will mostly work with finitely generated vector spaces. Infinite linear combinations can also be introduced in some cases, but we would need stronger assumptions (e.g., working over $\mathbb{R}$) and stronger tools (limits, convergence, ...).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 5.15)</span></p>

A linear combination can be understood in two ways. The first way is to view it as the expression $\sum_{i=1}^{n} \alpha_i v_i$ and the second way is to consider its concrete value, i.e., the resulting vector. We will use both of these perspectives.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 5.17)</span></p>

Let $V$ be a vector space over $\mathbb{T}$, and let $v_1, \ldots, v_n \in V$. Then

$$\operatorname{span}\lbrace v_1, \ldots, v_n \rbrace = \lbrace \sum_{i=1}^{n} \alpha_i v_i ;\ \alpha_1, \ldots, \alpha_n \in \mathbb{T} \rbrace.$$

*Proof.* Inclusion "$\supseteq$". The linear span $\operatorname{span}\lbrace v_1, \ldots, v_n \rbrace$ is a subspace of $V$ containing the vectors $v_1, \ldots, v_n$, so it must be closed under scalar multiples and sums. Hence it also contains the multiples $\alpha_i v_i$, $i = 1, \ldots, n$, and also their sum $\sum_{i=1}^{n} \alpha_i v_i$.

Inclusion "$\subseteq$". It suffices to show that the set of linear combinations $M := \lbrace \sum_{i=1}^{n} \alpha_i v_i ;\ \alpha_1, \ldots, \alpha_n \in \mathbb{T} \rbrace$ is a vector subspace of $V$ containing the vectors $v_1, \ldots, v_n$, and therefore it is one of the sets whose intersection is $\operatorname{span}\lbrace v_1, \ldots, v_n \rbrace$. Each $v_i$ is contained in $M$; it suffices to take the linear combination with $\alpha_i = 1$ and $\alpha_j = 0$, $j \neq i$. The zero vector is also contained; take the linear combination with zero coefficients. Closure under sums: Take any two vectors $u = \sum_{i=1}^{n} \beta_i v_i$, $u' = \sum_{i=1}^{n} \beta_i' v_i$ from $M$. Then $u + u' = \sum_{i=1}^{n} (\beta_i + \beta_i') v_i$, which is an element of $M$. Similarly for scalar multiples.

</div>

The linear span of a single vector $v$ is given by the set of all its linear combinations, i.e., its scalar multiples. The linear span of two vectors $u, v$ (with different directions) in $\mathbb{R}^3$ represents a plane.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 5.19)</span></p>

Let $V$ be a vector space over $\mathbb{T}$ and let $M \subseteq V$. Then $\operatorname{span}(M)$ consists of all linear combinations of every finite system of vectors from $M$.

*Proof.* Analogous to the proof of Theorem 5.17; left as an exercise.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 5.20 — A Slightly Different View of a System of Equations)</span></p>

The expression $Ax = \sum_j x_j A_{*j}$ is actually a linear combination of the columns of matrix $A$ (cf. Remark 3.19), so solving the system $Ax = b$ means finding a linear combination of columns that equals $b$. A solution therefore exists if and only if $b$ belongs to the subspace generated by the columns of matrix $A$, i.e., $b \in \operatorname{span}\lbrace A_{*1}, \ldots, A_{*n} \rbrace$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 5.21 — A Slightly Different View of Matrix Multiplication)</span></p>

The previous reasoning can also be applied to matrix multiplication. Consider $A \in \mathbb{T}^{m \times p}$, $B \in \mathbb{T}^{p \times n}$. Let us first focus on the columns of the resulting matrix $AB$. Any $j$-th column can be expressed as $(AB)_{*j} = AB_{*j} = \sum_{k=1}^{p} b_{kj} A_{*k}$, which is a linear combination of the columns of matrix $A$. Every column of the matrix $AB$ is therefore formed as a linear combination of the columns of matrix $A$.

Similarly, matrix multiplication can be interpreted as forming linear combinations of rows. Any $i$-th row of the resulting matrix $AB$ can be expressed as $(AB)_{i*} = A_{i*}B = \sum_{k=1}^{p} a_{ik}B_{k*}$, and thus represents a linear combination of the rows of matrix $B$. Elementary row operations on a matrix $B$ can then be viewed as forming linear combinations of rows and replacing the original rows with these combinations.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 5.22 — Yet Another View of Matrix Multiplication)</span></p>

The product $A \in \mathbb{T}^{m \times p}$, $B \in \mathbb{T}^{p \times n}$ can also be expressed in another way as $AB = \sum_{k=1}^{p} A_{*k}B_{k*}$. Each term of the sum represents the outer product of two vectors, which produces a matrix of rank at most $1$. With this formula, we have decomposed the matrix into a sum of at most $k$ rank-$1$ matrices.

</div>

### 5.3 Linear Independence

A finitely generated space can typically be generated by different sets of vectors. The motivation for this section is the desire to find a generating set that is minimal both in size and in inclusion (meaning no proper subset still generates the space), cf. Example 5.12. This will then lead to concepts such as basis, coordinates, and dimension.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 5.23 — Linear Independence)</span></p>

Let $V$ be a vector space over $\mathbb{T}$ and let $v_1, \ldots, v_n \in V$. Then the vectors $v_1, \ldots, v_n$ are called *linearly independent* if the equality $\sum_{i=1}^{n} \alpha_i v_i = o$ holds only for $\alpha_1 = \ldots = \alpha_n = 0$. Otherwise the vectors are *linearly dependent*.

</div>

Thus vectors are linearly dependent if there exist $\alpha_1, \ldots, \alpha_n \in \mathbb{T}$, not all zero, such that $\sum_{i=1}^{n} \alpha_i v_i = o$.

We also generalize the concept of linear independence to infinite sets of vectors — however, infinity presents some difficulties (e.g., what would an infinite linear combination mean?), which is why it is defined as follows:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 5.24 — Linear Independence of an Infinite Set)</span></p>

Let $V$ be a vector space over $\mathbb{T}$ and let $M \subseteq V$ be an infinite set of vectors. Then $M$ is *linearly independent* if every finite subset of $M$ is linearly independent. Otherwise $M$ is *linearly dependent*.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.25 — Examples of Linearly (In)dependent Vectors in $\mathbb{R}^2$)</span></p>

- $(1, 0)^\top$ is linearly independent,
- $(1, 0)^\top$, $(2, 0)^\top$ are linearly dependent,
- $(1, 1)^\top$, $(1, 2)^\top$ are linearly independent,
- $(1, 0)^\top$, $(0, 1)^\top$, $(1, 1)^\top$ are linearly dependent,
- $(0, 0)^\top$ is linearly dependent,
- the empty set is linearly independent.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.26 — Testing Linear Independence)</span></p>

It is not hard to see that two vectors form a linearly dependent system if one of them is a scalar multiple of the other. For more vectors, however, linear dependence is not so easy to see. How can we practically determine whether given arithmetic vectors, e.g., $(1, 3, 2)^\top$, $(2, 5, 3)^\top$, $(2, 3, 1)^\top$, are linearly dependent or independent? By the definition, we seek when a linear combination of the vectors gives the zero vector:

$$\alpha \begin{pmatrix} 1 \\ 3 \\ 2 \end{pmatrix} + \beta \begin{pmatrix} 2 \\ 5 \\ 3 \end{pmatrix} + \gamma \begin{pmatrix} 2 \\ 3 \\ 1 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}.$$

We express this equivalently as a system of equations with unknowns $\alpha, \beta, \gamma$ and solve it by reducing the system matrix to echelon form:

$$\begin{pmatrix} 1 & 2 & 2 \\ 3 & 5 & 3 \\ 2 & 3 & 1 \end{pmatrix} \sim \begin{pmatrix} 1 & 0 & -4 \\ 0 & 1 & 3 \\ 0 & 0 & 0 \end{pmatrix}.$$

The system has infinitely many solutions and we can certainly find a nonzero one, e.g., $\alpha = 4$, $\beta = -3$, $\gamma = 1$. This means that the given vectors are linearly dependent.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Example 5.27 — Connection with Regularity)</span></p>

The definition of linear independence somewhat resembles the definition of regularity (Definition 3.26). This is no coincidence: the columns of a regular matrix (and consequently its rows) provide another example of linearly independent vectors. By definition, a square matrix $A$ is regular if the equality $\sum_j A_{*j} x_j = o$ holds only for $x = o$, and this precisely corresponds to the linear independence of the columns of matrix $A$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 5.28 — Characterization of Linear Dependence)</span></p>

Let $V$ be a vector space over $\mathbb{T}$, and let $v_1, \ldots, v_n \in V$. Then the vectors $v_1, \ldots, v_n$ are linearly dependent if and only if there exists $k \in \lbrace 1, \ldots, n \rbrace$ such that $v_k = \sum_{i \neq k} \alpha_i v_i$ for some $\alpha_1, \ldots, \alpha_n \in \mathbb{T}$, that is $v_k \in \operatorname{span}\lbrace v_1, \ldots, v_{k-1}, v_{k+1}, \ldots, v_n \rbrace$.

*Proof.* Implication "$\Rightarrow$". If the vectors are linearly dependent, then there exists a nontrivial linear combination equal to zero, i.e., $\sum_{i=1}^{n} \beta_i v_i = o$ for $\beta_1, \ldots, \beta_n \in \mathbb{T}$ with $\beta_k \neq 0$ for some $k \in \lbrace 1, \ldots, n \rbrace$. Here we may choose any $k$ such that $\beta_k \neq 0$. We express the $k$-th term $\beta_k v_k = -\sum_{i \neq k} \beta_i v_i$ and after dividing we obtain the required formula $v_k = \sum_{i \neq k} (-\beta_k^{-1}\beta_i)v_i$.

Implication "$\Leftarrow$". If $v_k = \sum_{i \neq k} \alpha_i v_i$, then $v_k - \sum_{i \neq k} \alpha_i v_i = o$, which is the required nontrivial combination equal to zero, since the coefficient of $v_k$ is $1 \neq 0$.

</div>

A consequence is yet another characterization of linear dependence. It says, among other things, that vectors are linearly dependent if and only if removing some (but not necessarily any, see Example 5.30) of them does not decrease their linear span. Thus there is some redundant vector among them. For a linearly independent system the opposite holds: removing any one of them strictly decreases their linear span, so none of them is redundant.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 5.29)</span></p>

Let $V$ be a vector space over $\mathbb{T}$, and let $v_1, \ldots, v_n \in V$. Then the vectors $v_1, \ldots, v_n$ are linearly dependent if and only if there exists $k \in \lbrace 1, \ldots, n \rbrace$ such that

$$\operatorname{span}\lbrace v_1, \ldots, v_n \rbrace = \operatorname{span}\lbrace v_1, \ldots, v_{k-1}, v_{k+1}, \ldots, v_n \rbrace.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.30)</span></p>

The vectors $(2, 3)^\top, (2, 1)^\top, (4, 2)^\top \in \mathbb{R}^2$ are linearly dependent, so their linear span can also be generated from a proper subset of these vectors. We can remove, for example, the second or the third vector (but not both at once) and the resulting two vectors will still generate the same space $\mathbb{R}^2$. However, the first vector cannot be removed — the remaining two vectors no longer generate $\mathbb{R}^2$!

</div>

### 5.4 Basis

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 5.31 — Basis)</span></p>

Let $V$ be a vector space over $\mathbb{T}$. Then by a *basis* we mean any linearly independent generating system of $V$.

</div>

In the definition, by a system we mean an ordered set; we will see later why the ordering is important (for coordinates, etc.). However, for simplicity of notation, we will denote a basis consisting of finitely many vectors $v_1, \ldots, v_n$ by $\lbrace v_1, \ldots, v_n \rbrace$.

A basis is therefore, by definition, a generating system of the space $V$ that is minimal in the sense of inclusion. Each generator serves a purpose; we cannot omit any of them, otherwise we would not generate the entire space $V$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.32 — Examples of Bases)</span></p>

- In $\mathbb{R}^2$ we have, for example, the basis $e_1 = (1, 0)^\top$, $e_2 = (0, 1)^\top$. Another basis is $(7, 5)^\top$, $(2, 3)^\top$.
- In $\mathbb{R}^n$ we have, for example, the basis $e_1, \ldots, e_n$, called the *canonical* basis and denoted by kan. Every vector $v = (v_1, \ldots, v_n)^\top \in \mathbb{R}^n$ can be expressed as a linear combination of the basis vectors simply as $v = \sum_{i=1}^{n} v_i e_i$.
- In $\mathcal{P}^n$, a basis is, for example, $1, x, x^2, \ldots, x^n$. Every polynomial $p \in \mathcal{P}^n$ in standard form $p(x) = a_n x^n + \ldots + a_1 x + a_0$ is already expressed as a linear combination of basis vectors (in reverse order). This is at first glance the simplest basis, but not the only possible one. The Bernstein basis consists of the vectors $\binom{n}{i}x^i(1 - x)^{n-i}$ for $i = 0, 1, \ldots, n$ and is used for various approximations, e.g., in computational geometry for approximating curves passing through or controlled by given points (so-called Bezier curves, used for instance in typography for describing fonts).
- In $\mathcal{P}$, a basis is, for example, the infinite countable system of polynomials $1, x, x^2, \ldots$
- In the space $\mathcal{C}_{[a,b]}$ a basis also exists, but it is not easy to express any one explicitly.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 5.33 — Uniqueness of Representation in a Basis)</span></p>

Let $v_1, \ldots, v_n$ be a basis of the space $V$. Then for every vector $u \in V$ there exist uniquely determined coefficients $\alpha_1, \ldots, \alpha_n \in \mathbb{T}$ such that $u = \sum_{i=1}^{n} \alpha_i v_i$.

*Proof.* The vectors $v_1, \ldots, v_n$ form a basis of $V$, so every $u \in V$ can be expressed as $u = \sum_{i=1}^{n} \alpha_i v_i$ for suitable scalars $\alpha_1, \ldots, \alpha_n \in \mathbb{T}$. We show uniqueness by contradiction. Suppose there also exists another representation $u = \sum_{i=1}^{n} \beta_i v_i$. Then $\sum_{i=1}^{n} \alpha_i v_i - \sum_{i=1}^{n} \beta_i v_i = u - u = o$, i.e., $\sum_{i=1}^{n} (\alpha_i - \beta_i)v_i = o$. Since $v_1, \ldots, v_n$ are linearly independent, we must have $\alpha_i = \beta_i$ for every $i = 1, \ldots, n$. This contradicts the assumption that the representations are different.

</div>

Thanks to the mentioned uniqueness, we can introduce the concept of coordinates.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 5.34 — Coordinates)</span></p>

Let $B = \lbrace v_1, \ldots, v_n \rbrace$ be a basis of the space $V$ and let the vector $u \in V$ have the representation $u = \sum_{i=1}^{n} \alpha_i v_i$. Then the *coordinates* of the vector $u$ with respect to the basis $B$ are the coefficients $\alpha_1, \ldots, \alpha_n$, and the coordinate vector is denoted $[u]_B := (\alpha_1, \ldots, \alpha_n)^\top$.

</div>

The concept of coordinates is more important than it might seem at first glance. It allows us to represent hard-to-grasp vectors and (finitely generated) spaces using coordinates, i.e., arithmetic vectors. Every vector has certain coordinates and conversely every $n$-tuple of scalars gives the coordinates of some vector. There is thus a bijective correspondence between vectors and coordinates, which we will later (Section 6.3) use to transfer many, e.g., computational, problems from the space $V$ to an arithmetic space where computations are easier.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.35 — Coordinates of a Vector with Respect to a Basis in $\mathbb{R}^2$)</span></p>

- Coordinates of the vector $(-2, 3)^\top$ with respect to the canonical basis: $[(-2, 3)^\top]_{\text{kan}} = (-2, 3)^\top$.
- Coordinates of the vector $(-2, 3)^\top$ with respect to the basis $B = \lbrace (-3, 1)^\top, (1, 1)^\top \rbrace$: $[(-2, 3)^\top]_B = (\tfrac{5}{4}, \tfrac{7}{4})^\top$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.36)</span></p>

For every $v \in \mathbb{R}^n$ we have $[v]_{\text{kan}} = v$, since the vector $v = (v_1, \ldots, v_n)^\top$ has the representation $v = \sum_{i=1}^{n} v_i e_i$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.37)</span></p>

Consider the basis $B = \lbrace 1, x, x^2 \rbrace$ of the space $\mathcal{P}^2$. Then $[3x^2 - 5]_B = (-5, 0, 3)^\top$. In general, every polynomial $p \in \mathcal{P}^n$ in standard form $p(x) = a_n x^n + \ldots + a_1 x + a_0$ has coordinates with respect to the basis $B = \lbrace 1, x, x^2, \ldots, x^n \rbrace$ given by $[p]_B = (a_0, a_1, \ldots, a_n)^\top$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.38)</span></p>

Let $B = \lbrace v_1, \ldots, v_n \rbrace$ be a basis of the space $V$. Then $[v_1]_B = (1, 0, \ldots, 0)^\top = e_1$, $[v_2]_B = e_2$, $\ldots$, $[v_n]_B = e_n$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.39 — Observation)</span></p>

Observe the following for a vector space $V$:

- If $v_1, \ldots, v_n \in V$ is a generating system of $V$, then every vector $u \in V$ can be expressed as a linear combination of the vectors $v_1, \ldots, v_n$ in at least one way.
- If $v_1, \ldots, v_n \in V$ are linearly independent, then every vector $u \in V$ can be expressed as a linear combination of the vectors $v_1, \ldots, v_n$ in at most one way.
- If $v_1, \ldots, v_n \in V$ is a basis of $V$, then every vector $u \in V$ can be expressed as a linear combination of the vectors $v_1, \ldots, v_n$ in exactly one way.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 5.40 — Linearity of Coordinates)</span></p>

For any basis $B$ of a finitely generated space $V$ over $\mathbb{T}$, vectors $u, v \in V$, and scalar $\alpha \in \mathbb{T}$, the following holds

$$[u + v]_B = [u]_B + [v]_B, \qquad [\alpha v]_B = \alpha [v]_B.$$

*Proof.* Let the basis $B$ consist of vectors $z_1, \ldots, z_n$, let $u = \sum_{i=1}^{n} \beta_i z_i$ and let $v = \sum_{i=1}^{n} \gamma_i z_i$. Then $u + v = \sum_{i=1}^{n} (\beta_i + \gamma_i) z_i$ and thus $[u + v]_B = (\beta_1 + \gamma_1, \ldots, \beta_n + \gamma_n)^\top = [u]_B + [v]_B$. Similarly for scalar multiples $\alpha[u]_B = \alpha(\beta_1, \ldots, \beta_n)^\top = (\alpha \beta_1, \ldots, \alpha \beta_n)^\top = [\alpha u]_B$.

</div>

The property from the proposition can be generalized: The coordinates of any linear combination of vectors equal the same linear combination of their coordinates. Coordinates thus preserve a certain structure and relationships between vectors (linear dependence, etc.). Later in Chapter 6 we will see that thanks to this property we can efficiently compute coordinates.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 5.41 — Existence of a Basis)</span></p>

Every vector space has a basis.

*Proof.* We carry out the proof only for a finitely generated space $V$. Let $v_1, \ldots, v_n$ be a generating system of $V$. If the vectors are linearly independent, they already form a basis. Otherwise, by Corollary 5.29 there exists an index $k$ such that $\operatorname{span}\lbrace v_1, \ldots, v_n \rbrace = \operatorname{span}\lbrace v_1, \ldots, v_{k-1}, v_{k+1}, \ldots, v_n \rbrace$. Thus, by removing $v_k$ the system of vectors still generates $V$. If the system of vectors is now linearly independent, it forms a basis. Otherwise we repeat the procedure until we find a basis. The procedure is finite because we have a finite set of generators, so we must eventually find a basis.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Lemma 5.42 — Exchange Lemma)</span></p>

Let $y_1, \ldots, y_n$ be a generating system of a vector space $V$ and let the vector $x \in V$ have the representation $x = \sum_{i=1}^{n} \alpha_i y_i$. Then for any $k$ such that $\alpha_k \neq 0$, the system $y_1, \ldots, y_{k-1}, x, y_{k+1}, \ldots, y_n$ is a generating system of $V$.

*Proof.* From the relation $x = \sum_{i=1}^{n} \alpha_i y_i$ we express $y_k = \tfrac{1}{\alpha_k}(x - \sum_{i \neq k} \alpha_i y_i)$. We want to prove that the vectors $y_1, \ldots, y_{k-1}, x, y_{k+1}, \ldots, y_n$ generate the space $V$. Take any vector $z \in V$. For suitable coefficients $\beta_i$ we can express $z$ as $z = \sum_{i=1}^{n} \beta_i y_i = \beta_k y_k + \sum_{i \neq k} \beta_i y_i = \tfrac{\beta_k}{\alpha_k}x + \sum_{i \neq k} (\beta_i - \tfrac{\beta_k}{\alpha_k}\alpha_i) y_i$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 5.44 — Steinitz Exchange Theorem)</span></p>

Let $V$ be a vector space, let $x_1, \ldots, x_m$ be a linearly independent system in $V$, and let $y_1, \ldots, y_n$ be a generating system of $V$. Then

1. $m \le n$,
2. there exist pairwise distinct indices $k_1, \ldots, k_{n-m}$ such that $x_1, \ldots, x_m, y_{k_1}, \ldots, y_{k_{n-m}}$ form a generating system of $V$.

*Proof.* We carry out the proof by mathematical induction on $m$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 5.45)</span></p>

All bases of a finitely generated vector space $V$ have the same size.

*Proof.* Let $x_1, \ldots, x_m$ and $y_1, \ldots, y_n$ be two bases of the space $V$. In particular, $x_1, \ldots, x_m$ are linearly independent and $y_1, \ldots, y_n$ are generators of $V$, hence $m \le n$. Analogously in reverse, $y_1, \ldots, y_n$ are linearly independent and $x_1, \ldots, x_m$ generate $V$, hence $n \le m$. Together we get $m = n$.

</div>

### 5.5 Dimension

Every finitely generated space has a basis (Theorem 5.41) and all bases have the same size (Corollary 5.45), which justifies introducing the dimension of a space as the size of (any) basis.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 5.46 — Dimension)</span></p>

The *dimension* of a finitely generated vector space is the size of any of its bases. The dimension of a space that is not finitely generated is $\infty$. The dimension of a space $V$ is denoted $\dim V$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.47 — Examples of Dimensions)</span></p>

- $\dim \mathbb{R}^n = n$, $\dim \mathbb{R}^{m \times n} = mn$, $\dim \lbrace o \rbrace = 0$, $\dim \mathcal{P}^n = n + 1$,
- the real spaces $\mathcal{P}$, $\mathcal{F}$, and the space $\mathbb{R}$ over $\mathbb{Q}$ are not finitely generated and have dimension $\infty$ (see Problem 5.1).

</div>

From now on, we will consider only finitely generated vector spaces.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 5.48 — Relationship Between System Size and Dimension)</span></p>

For a vector space $V$ the following holds:

1. Let $x_1, \ldots, x_m \in V$ be linearly independent. Then $m \le \dim V$. If $m = \dim V$, then $x_1, \ldots, x_m$ is a basis.
2. Let $y_1, \ldots, y_n$ be generators of $V$. Then $n \ge \dim V$. If $n = \dim V$, then $y_1, \ldots, y_n$ is a basis.

*Proof.* Denote $d = \dim V$ and let $z_1, \ldots, z_d$ be a basis of $V$, i.e., its linearly independent generators. (1) Since $x_1, \ldots, x_m$ are linearly independent and $z_1, \ldots, z_d$ are generators of $V$, by the Steinitz Exchange Theorem 5.44 we have $m \le d$. If $m = d$, then by the same theorem the system $x_1, \ldots, x_m$ can be extended by $d - m = 0$ vectors to a generating system of $V$. Hence they are necessarily generators and thus a basis. (2) Since $y_1, \ldots, y_n$ are generators of $V$ and $z_1, \ldots, z_d$ are linearly independent, by the Steinitz Exchange Theorem 5.44 we have $n \ge d$. Suppose $n = d$. If $y_1, \ldots, y_n$ are linearly independent, they form a basis. If they are linearly dependent, then one can be omitted to obtain a generating system of size $n - 1$ (Corollary 5.29). By the Steinitz theorem we would then have $d \le n - 1$, which leads to a contradiction.

</div>

The first part of Proposition 5.48 says, among other things, that a basis can be viewed as a maximal linearly independent system. The second part says that a basis is a minimal generating system (both in terms of inclusion and in terms of size).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 5.49 — Extension of a Linearly Independent System to a Basis)</span></p>

Every linearly independent system of a vector space $V$ can be extended to a basis of $V$.

*Proof.* Let $x_1, \ldots, x_m$ be linearly independent and $z_1, \ldots, z_d$ be a basis of $V$. By the Steinitz Exchange Theorem 5.44 there exist indices $k_1, \ldots, k_{d-m}$ such that $x_1, \ldots, x_m, z_{k_1}, \ldots, z_{k_{d-m}}$ are generators of $V$. Their count is $d$, so by Proposition 5.48 they also form a basis of $V$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 5.50 — Dimension of a Subspace)</span></p>

If $W$ is a subspace of $V$, then $\dim W \le \dim V$. If moreover $\dim W = \dim V$, then $W = V$.

*Proof.* Define $M := \emptyset$. If $\operatorname{span}(M) = W$, we are done. Otherwise there exists a vector $v \in W \setminus \operatorname{span}(M)$. We add the vector $v$ to the set $M$ and repeat the procedure. Since $M$ is a linearly independent set of vectors, by Proposition 5.48 the size of $M$ is bounded above by the dimension of $V$. The process is therefore finite. Since $\operatorname{span}(M) = W$, the set $M$ forms a basis of $W$, and hence $\dim W \le \dim V$.

If $\dim W = \dim V$, then the set $M$ must by Proposition 5.48 form a basis of $V$, and therefore $W = V$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.51 — Subspaces of $\mathbb{R}^2$)</span></p>

Let us find all subspaces of $\mathbb{R}^2$:

- dimension 2: this is only $\mathbb{R}^2$ (by Theorem 5.50),
- dimension 1: these are generated by a single vector, so they are all lines passing through the origin,
- dimension 0: this is only $\lbrace o \rbrace$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.52 — Structure of Subspaces)</span></p>

To illustrate the structure of subspaces, let us first consider all subsets of the set $\lbrace 1, \ldots, n \rbrace$ and the relation "being a subset", i.e., inclusion $\subseteq$. Some subsets are incomparable with respect to inclusion while others are not. Inclusion is thus a partial order and can be depicted by a so-called *Hasse diagram*, where edges indicate "adjacent" subsets in the inclusion.

In a similar way, we can also depict the structure of subspaces of a space $V$ of dimension $n$, since the relation "being a subspace" is also a partial order. The diagram will have $n + 1$ levels, where the $i$-th level contains the subspaces of dimension $i$. These are incomparable among themselves in terms of inclusion or in terms of "being a subspace", although they may share some vectors.

</div>

#### Sum of Subspaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 5.53 — Sum of Subspaces)</span></p>

Let $U, V$ be subspaces of a vector space $W$. Then the *sum of subspaces* $U, V$ is defined as $U + V := \lbrace u + v ;\ u \in U, v \in V \rbrace$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 5.54)</span></p>

Let $U, V$ be subspaces of a vector space $W$. Then

$$U + V = \operatorname{span}(U \cup V).$$

*Proof.* Inclusion "$\subseteq$": this is trivial, since the space $\operatorname{span}(U \cup V)$ is closed under sums. Inclusion "$\supseteq$": It suffices to show that $U + V$ contains the spaces $U, V$ and that it is a subspace of $W$. The first part is obvious; for the second, consider $x_1, x_2 \in U + V$. The vectors can be expressed as $x_1 = u_1 + v_1$, $u_1 \in U$, $v_1 \in V$, and $x_2 = u_2 + v_2$, $u_2 \in U$, $v_2 \in V$. Then $x_1 + x_2 = (u_1 + u_2) + (v_1 + v_2) \in U + V$, which proves closure under addition. For closure under scalar multiples, consider $x = u + v \in U + V$, $u \in U$, $v \in V$ and a scalar $\alpha$. Then $\alpha x = \alpha(u + v) = (\alpha u) + (\alpha v) \in U + V$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.55)</span></p>

- $\mathbb{R}^2 = \operatorname{span}\lbrace e_1 \rbrace + \operatorname{span}\lbrace e_2 \rbrace$,
- $\mathbb{R}^3 = \operatorname{span}\lbrace e_1 \rbrace + \operatorname{span}\lbrace e_2 \rbrace + \operatorname{span}\lbrace e_3 \rbrace$,
- $\mathbb{R}^3 = \operatorname{span}\lbrace e_1, e_2 \rbrace + \operatorname{span}\lbrace e_3 \rbrace$,
- $\mathbb{R}^2 = \operatorname{span}\lbrace (1, 2)^\top \rbrace + \operatorname{span}\lbrace (3, 4)^\top \rbrace$,
- but also $\mathbb{R}^2 = \operatorname{span}\lbrace (1, 2)^\top \rbrace + \operatorname{span}\lbrace (3, 4)^\top \rbrace + \operatorname{span}\lbrace (5, 6)^\top \rbrace$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 5.56 — Dimension of Sum and Intersection)</span></p>

Let $U, V$ be subspaces of a vector space $W$. Then

$$\dim(U + V) + \dim(U \cap V) = \dim U + \dim V.$$

*Proof.* $U \cap V$ is a subspace of $W$, so it has a finite basis $z_1, \ldots, z_p$. By Theorem 5.49 we can extend it to a basis of $U$ of the form $z_1, \ldots, z_p, x_1, \ldots, x_m$. Similarly, we can extend it to a basis of $V$ of the form $z_1, \ldots, z_p, y_1, \ldots, y_n$. It suffices to show that the vectors $z_1, \ldots, z_p, x_1, \ldots, x_m, y_1, \ldots, y_n$ together form a basis of $U + V$, and then the equality $(5.3)$ will follow.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 5.58 — Direct Sum of Subspaces)</span></p>

If $U \cap V = \lbrace o \rbrace$, then the sum of subspaces $W = U + V$ is called the *direct sum* of subspaces $U, V$ and is denoted $W = U \oplus V$. By Theorem 5.56 we have $\dim(U \oplus V) = \dim U + \dim V$. The condition $U \cap V = \lbrace o \rbrace$ additionally ensures that every vector $w \in W$ can be written in a unique way as $w = u + v$, where $u \in U$ and $v \in V$ (see Problem 5.2). For example, $\mathbb{R}^2 = \operatorname{span}\lbrace (1, 2)^\top \rbrace \oplus \operatorname{span}\lbrace (3, 4)^\top \rbrace$ and $\mathbb{R}^3 = \operatorname{span}\lbrace e_1 \rbrace \oplus \operatorname{span}\lbrace e_2 \rbrace \oplus \operatorname{span}\lbrace e_3 \rbrace$ are direct sums, but $\mathbb{R}^2 = \operatorname{span}\lbrace (1, 2)^\top \rbrace \oplus \operatorname{span}\lbrace (3, 4)^\top \rbrace \oplus \operatorname{span}\lbrace (5, 6)^\top \rbrace$ is not.

</div>

### 5.6 Matrix Spaces

We now combine matrix theory with vector spaces. Both subjects will mutually enrich each other: the vector space perspective will allow us to easily derive further properties of matrices, and conversely, methods from matrix theory will provide tools for testing linear independence, determining dimension, etc.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 5.59 — Matrix Spaces)</span></p>

Let $A \in \mathbb{T}^{m \times n}$. Then we define

1. the column space $\mathcal{S}(A) := \operatorname{span}\lbrace A_{*1}, \ldots, A_{*n} \rbrace$,
2. the row space $\mathcal{R}(A) := \mathcal{S}(A^\top)$,
3. the kernel $\operatorname{Ker}(A) := \lbrace x \in \mathbb{T}^n ;\ Ax = o \rbrace$.

</div>

The column space is therefore the space generated by the columns of matrix $A$, and it is a subspace of $\mathbb{T}^m$. Similarly, the row space is the space generated by the rows of matrix $A$, and it is a subspace of $\mathbb{T}^n$. The kernel $\operatorname{Ker}(A)$ consists of all solutions to the system $Ax = o$ and is also a subspace of $\mathbb{T}^n$, since the three basic properties are satisfied:

- The kernel contains the zero vector: $Ao = o$.
- The kernel is closed under addition: If $x, y \in \mathbb{T}^n$ are solutions to the system, then $Ax = o$, $Ay = o$. Adding the equations we get $A(x + y) = o$, so the vector $x + y$ also belongs to the kernel.
- The kernel is closed under scalar multiples: If the vector $x \in \mathbb{T}^n$ is a solution to the system, then $Ax = o$. For any $\alpha \in \mathbb{T}$ we have $A(\alpha x) = \alpha(Ax) = \alpha o = o$, so the vector $\alpha x$ also belongs to the kernel.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.60)</span></p>

Consider the real matrix

$$A = \begin{pmatrix} 1 & 1 & 1 \\ 0 & 1 & 0 \end{pmatrix}.$$

Then its column space is $\mathcal{S}(A) = \mathbb{R}^2$ and its row space is $\mathcal{R}(A) = \operatorname{span}\lbrace (1, 1, 1)^\top, (0, 1, 0)^\top \rbrace$. The kernel of matrix $A$ is determined by solving the system $Ax = o$. The matrix $A$ is already in echelon form, so using the free variable $x_3$ we describe the solution set as $\lbrace (x_3, 0, -x_3)^\top ;\ x_3 \in \mathbb{R} \rbrace$. The kernel thus has the form $\operatorname{Ker}(A) = \operatorname{span}\lbrace (1, 0, -1)^\top \rbrace$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 5.61)</span></p>

Let $A \in \mathbb{T}^{m \times n}$. Then

1. $\mathcal{S}(A) = \lbrace Ax ;\ x \in \mathbb{T}^n \rbrace$,
2. $\mathcal{R}(A) = \lbrace A^\top y ;\ y \in \mathbb{T}^m \rbrace$.

*Proof.* Obvious from the fact that $Ax = \sum_{j=1}^{n} x_j A_{*j}$ represents a linear combination of the columns of matrix $A$. In the second part, analogously $A^\top y$ represents a linear combination of the rows of matrix $A$.

</div>

We can represent any subspace $V$ of $\mathbb{T}^n$ using matrices. It suffices to take some of its generators $v_1, \ldots, v_m$ and construct a matrix $A \in \mathbb{T}^{m \times n}$ whose rows are precisely the vectors $v_1, \ldots, v_m$. Then $V = \mathcal{R}(A)$. Similarly, $V$ can be expressed as the column space of a suitable matrix from $\mathbb{T}^{n \times m}$. We can even represent the space $V$ as the kernel of a suitable matrix from $\mathbb{T}^{m \times n}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 5.62)</span></p>

Let $V$ be a subspace of $\mathbb{T}^n$. Then

1. $V = \mathcal{S}(A)$ for a suitable matrix $A \in \mathbb{T}^{n \times m}$,
2. $V = \mathcal{R}(A)$ for a suitable matrix $A \in \mathbb{T}^{m \times n}$,
3. $V = \operatorname{Ker}(A)$ for a suitable matrix $A \in \mathbb{T}^{m \times n}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 5.63 — Geometric View of Matrix Spaces)</span></p>

Consider the mapping $x \mapsto Ax$ with matrix $A \in \mathbb{T}^{m \times n}$. The kernel of matrix $A$ is thus formed by all vectors from $\mathbb{T}^n$ that map to the zero vector. The column space $\mathcal{S}(A)$ of matrix $A$ represents the set of all images, i.e., the image of the space $\mathbb{T}^n$ under this mapping. As we will show later, these spaces play a key role in analyzing the geometric structure of this mapping.

</div>

#### Spaces and Left Multiplication by a Matrix

Let us examine how matrix spaces change when a matrix is multiplied on the left by another matrix (this is essentially what Gaussian elimination does).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 5.64 — Spaces and Left Multiplication by a Matrix)</span></p>

Let $A \in \mathbb{T}^{m \times n}$, $Q \in \mathbb{T}^{p \times m}$. Then

1. $\mathcal{R}(QA)$ is a subspace of $\mathcal{R}(A)$,
2. If $A_{*k} = \sum_{j \neq k} \alpha_j A_{*j}$ for some $k \in \lbrace 1, \ldots, n \rbrace$ and some $\alpha_j \in \mathbb{T}$, $j \neq k$, then $(QA)_{*k} = \sum_{j \neq k} \alpha_j (QA)_{*j}$.

*Proof.* (1) It suffices to show $\mathcal{R}(QA) \subseteq \mathcal{R}(A)$. Let $x \in \mathcal{R}(QA)$, then there exists $y \in \mathbb{T}^p$ such that $x = (QA)^\top y = A^\top(Q^\top y) \in \mathcal{R}(A)$. (2) $(QA)_{*k} = QA_{*k} = Q(\sum_{j \neq k} \alpha_j A_{*j}) = \sum_{j \neq k} \alpha_j QA_{*j} = \sum_{j \neq k} \alpha_j (QA)_{*j}$.

</div>

The theorem says that row spaces are directly comparable — after multiplying by any matrix on the left we obtain a subspace. This is also easily seen from the fact that every row of the matrix $QA$ is actually a linear combination of the rows of matrix $A$ (see Remark 5.21), and by selected linear combinations one can generate only a subspace. Also the linear dependence relation is preserved: if the $i$-th column of matrix $A$ is dependent on the others, then the $i$-th column of matrix $QA$ is dependent on the others with the same linear combination (note that linear independence need not be preserved).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 5.66 — Spaces and Left Multiplication by a Regular Matrix)</span></p>

Let $Q \in \mathbb{T}^{m \times m}$ be regular and $A \in \mathbb{T}^{m \times n}$. Then

1. $\mathcal{R}(QA) = \mathcal{R}(A)$,
2. The equality $A_{*k} = \sum_{j \neq k} \alpha_j A_{*j}$ holds if and only if $(QA)_{*k} = \sum_{j \neq k} \alpha_j (QA)_{*j}$, where $k \in \lbrace 1, \ldots, n \rbrace$ and $\alpha_j \in \mathbb{T}$, $j \neq k$.

*Proof.* (1) By Proposition 5.64 we have $\mathcal{R}(QA) \subseteq \mathcal{R}(A)$. Applying Proposition 5.64 to the matrix $(QA)$ multiplied on the left by $Q^{-1}$, we get $\mathcal{R}(Q^{-1}QA) \subseteq \mathcal{R}(QA)$, i.e., $\mathcal{R}(A) \subseteq \mathcal{R}(QA)$. Together we have $\mathcal{R}(QA) = \mathcal{R}(A)$. (2) The left-to-right implication follows from Proposition 5.64. The reverse implication follows from Proposition 5.64 applied to the matrix $(QA)$ multiplied on the left by $Q^{-1}$.

</div>

A consequence of the preceding theorem is that if some columns of matrix $A$ are linearly independent, they remain so after multiplication by a regular matrix on the left.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.65)</span></p>

In matrix $A$, the second column is twice the first, and this property is preserved in the product $QA$:

$$QA = \begin{pmatrix} 1 & 2 & -1 \\ -2 & 1 & 1 \end{pmatrix} \begin{pmatrix} 1 & 2 & 4 \\ 2 & 4 & 5 \\ 1 & 2 & 7 \end{pmatrix} = \begin{pmatrix} 4 & 8 & 7 \\ 1 & 2 & 4 \end{pmatrix}.$$

In matrix $A'$, the third column is the sum of the first two, and this property is again preserved in the product $QA'$:

$$QA' = \begin{pmatrix} 1 & 2 & -1 \\ -2 & 1 & 1 \end{pmatrix} \begin{pmatrix} 1 & 1 & 2 \\ 1 & 2 & 3 \\ 1 & 3 & 4 \end{pmatrix} = \begin{pmatrix} 2 & 2 & 4 \\ 0 & 3 & 3 \end{pmatrix}.$$

</div>

#### Matrix Spaces and RREF

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 5.68 — Matrix Spaces and RREF)</span></p>

Let $A \in \mathbb{T}^{m \times n}$ and let $A^R$ be its RREF form with pivots at positions $(1, p_1), \ldots, (r, p_r)$, where $r = \operatorname{rank}(A)$. Then

1. the nonzero rows of $A^R$, i.e., the vectors $A^R_{1*}, \ldots, A^R_{r*}$, form a basis of $\mathcal{R}(A)$,
2. the columns $A_{*p_1}, \ldots, A_{*p_r}$ form a basis of $\mathcal{S}(A)$,
3. $\dim \mathcal{R}(A) = \dim \mathcal{S}(A) = r$.

*Proof.* We know from Theorem 3.31 that $A^R = QA$ for some regular matrix $Q$. (1) By Proposition 5.66 we have $\mathcal{R}(A) = \mathcal{R}(QA) = \mathcal{R}(A^R)$. The nonzero rows of $A^R$ are linearly independent, so they form a basis of both $\mathcal{R}(A^R)$ and $\mathcal{R}(A)$. (2) The vectors $A^R_{*p_1}, \ldots, A^R_{*p_r}$ form a basis of $\mathcal{S}(A^R)$. These vectors are certainly linearly independent (they are unit vectors). They generate $\mathcal{S}(A^R)$, since any non-pivot column can be expressed as a linear combination of the pivot ones: $A^R_{*j} = \sum_{i=1}^{r} a^R_{ij} e_i = \sum_{i=1}^{r} a^R_{ij} A^R_{*p_i}$. Now we use Proposition 5.66, which guarantees that $A_{*p_1}, \ldots, A_{*p_r}$ are also linearly independent and generate $\mathcal{S}(A)$, thus forming a basis of $\mathcal{S}(A)$. (3) The value $\dim \mathcal{R}(A)$ is the size of the basis of $\mathcal{R}(A)$, i.e., $r$, and similarly $\dim \mathcal{S}(A)$ is also $r$. Moreover $r = \operatorname{rank}(A)$.

</div>

Let us emphasize that a basis of the row space $\mathcal{R}(A)$ is found in the rows of the matrix $A^R$, while a basis of the column space $\mathcal{S}(A)$ is found in the columns of the original matrix $A$.

The third property of Theorem 5.68 yields a very nontrivial consequence for the rank of a matrix and its transpose, since

$$\operatorname{rank}(A) = \dim \mathcal{R}(A) = \dim \mathcal{S}(A) = \dim \mathcal{R}(A^\top) = \operatorname{rank}(A^\top).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 5.69)</span></p>

For every matrix $A \in \mathbb{T}^{m \times n}$ we have $\operatorname{rank}(A) = \operatorname{rank}(A^\top)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.70)</span></p>

Consider the space $V = \operatorname{span}\lbrace (1, 2, 3, 4, 5)^\top, (1, 1, 1, 1, 1)^\top, (1, 3, 5, 7, 9)^\top, (2, 1, 1, 0, 0)^\top \rbrace \le \mathbb{R}^5$. First, let us construct a matrix $A$ whose columns equal the given generators of $V$, so that $V = \mathcal{S}(A)$, and reduce it to reduced row echelon form:

$$A = \begin{pmatrix} 1 & 1 & 1 & 2 \\ 2 & 1 & 3 & 1 \\ 3 & 1 & 5 & 1 \\ 4 & 1 & 7 & 0 \\ 5 & 1 & 9 & 0 \end{pmatrix} \xrightarrow{\text{RREF}} \begin{pmatrix} 1 & 0 & 2 & 0 \\ 0 & 1 & -1 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}.$$

From the RREF form we see that $\dim(V) = \operatorname{rank}(A) = 3$ and a basis of $V$ is, for example, $(1, 2, 3, 4, 5)^\top$, $(1, 1, 1, 1, 1)^\top$, $(2, 1, 1, 0, 0)^\top$. The third generator is dependent on the others; specifically, it equals twice the first minus the second (the coefficients can be seen in the third column of the RREF matrix).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 5.71)</span></p>

Consider a system of linear equations $Ax = b$. Solvability of the system essentially means that the right-hand side vector $b$ can be expressed as a linear combination of the columns of matrix $A$ (cf. Remark 5.20). Hence the system is solvable if and only if $b \in \mathcal{S}(A)$, i.e., $\mathcal{S}(A) = \mathcal{S}(A \mid b)$. Theorem 5.68 then directly gives the statement of the Frobenius theorem from Remark 2.25.

</div>

#### Kernel Dimension and Matrix Rank

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 5.72 — Rank-Nullity Theorem)</span></p>

For every matrix $A \in \mathbb{T}^{m \times n}$ the following holds

$$\dim \operatorname{Ker}(A) + \operatorname{rank}(A) = n.$$

*Proof.* Let $\dim \operatorname{Ker}(A) = k$. Let the vectors $v_1, \ldots, v_k$ form a basis of $\operatorname{Ker}(A)$, which means in particular that $Av_1 = \ldots = Av_k = o$. We extend the vectors $v_1, \ldots, v_k$ to a basis of the entire space $\mathbb{T}^n$ by adding vectors $v_{k+1}, \ldots, v_n$. It suffices to show that the vectors $Av_{k+1}, \ldots, Av_n$ form a basis of $\mathcal{S}(A)$, since then $\operatorname{rank}(A) = \dim \mathcal{S}(A) = n - k$ and the equality from the theorem is satisfied.

"Generating property." Let $y \in \mathcal{S}(A)$, then $y = Ax$ for some $x \in \mathbb{T}^n$. This $x$ can be expressed as $x = \sum_{i=1}^{n} \alpha_i v_i$. Substituting, $y = Ax = A(\sum_{i=1}^{n} \alpha_i v_i) = \sum_{i=1}^{n} \alpha_i A v_i = \sum_{i=k+1}^{n} \alpha_i (Av_i)$.

"Linear independence." Let $\sum_{i=k+1}^{n} \alpha_i Av_i = o$. Then $A(\sum_{i=k+1}^{n} \alpha_i v_i) = o$, so $\sum_{i=k+1}^{n} \alpha_i v_i$ belongs to the kernel of matrix $A$. Therefore $\sum_{i=k+1}^{n} \alpha_i v_i = \sum_{i=1}^{k} \beta_i v_i$ for some scalars $\beta_1, \ldots, \beta_k$. Rewriting the equation as $\sum_{i=k+1}^{n} \alpha_i v_i + \sum_{i=1}^{k} (-\beta_i) v_i = o$ and using the linear independence of the vectors $v_1, \ldots, v_n$, we get $\alpha_{k+1} = \ldots = \alpha_n = \beta_1 = \ldots = \beta_k = 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 5.73 — Geometric View of Theorem 5.72)</span></p>

Consider the mapping $x \mapsto Ax$ with matrix $A \in \mathbb{T}^{m \times n}$, see Remark 5.63. The space $\mathbb{T}^n$ maps onto the space $\mathcal{S}(A)$, whose dimension is $r = \operatorname{rank}(A)$. Thus the mapping sends an $n$-dimensional space to an $r$-dimensional space. Precisely this deficit $n - r \ge 0$ equals, by formula (5.4), the dimension of the kernel of matrix $A$. For a regular matrix, the kernel is trivial ($\operatorname{Ker}(A) = \lbrace o \rbrace$), and so it maps $\mathbb{T}^n$ onto all of $\mathbb{T}^n$. However, the larger the kernel, the smaller the image of $\mathbb{T}^n$. The kernel dimension thus describes the degree of "degeneracy" of the mapping.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.74 — Computing a Basis of the Kernel)</span></p>

Consider the matrix and its RREF form

$$A = \begin{pmatrix} 2 & 4 & 4 & 4 \\ -3 & -4 & 2 & 0 \\ 5 & 7 & -2 & 1 \end{pmatrix} \xrightarrow{\text{RREF}} \begin{pmatrix} 1 & 0 & -6 & -4 \\ 0 & 1 & 4 & 3 \\ 0 & 0 & 0 & 0 \end{pmatrix}.$$

Thus $\dim \operatorname{Ker}(A) = 4 - 2 = 2$. The space $\operatorname{Ker}(A)$ represents all solutions of the system $Ax = o$, which have the form $(6x_3 + 4x_4, -4x_3 - 3x_4, x_3, x_4)^\top$, $x_3, x_4 \in \mathbb{R}$, i.e.,

$$x_3(6, -4, 1, 0)^\top + x_4(4, -3, 0, 1)^\top, \quad x_3, x_4 \in \mathbb{R}.$$

Thus the vectors $(6, -4, 1, 0)^\top$, $(4, -3, 0, 1)^\top$ form a basis of $\operatorname{Ker}(A)$. These vectors can also be found directly by substituting $1$ for one non-pivot variable, $0$ for the remaining non-pivot variables, and computing the values of the pivot variables. This procedure works universally for any matrix.

</div>

### 5.7 Applications

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.75 — More on Coding)</span></p>

Let us continue from Example 4.42 about the Hamming code $(7, 4, 3)$. For encoding we used the generating matrix $H$ of size $7 \times 4$ simply by encoding an input segment $a$ of length $4$ into a segment $b := Ha$ of length $7$. All encoded segments thus represent the column space of matrix $H$. Since $H$ has linearly independent columns, this is a subspace of dimension $4$ in the space $\mathbb{Z}_2^7$.

Error detection of a received segment $b$ is performed using a detection matrix $D$ of size $3 \times 7$. If $Db = o$, no error occurred (or at least two errors occurred). We therefore want the detection matrix to map (only) vectors from the column space of matrix $H$ to the zero vector. Hence we need $\mathcal{S}(H) = \operatorname{Ker}(D)$. We can now see why matrix $D$ has these dimensions — for its kernel to be a four-dimensional subspace, it must by Theorem 5.72 have rank $3$, and therefore $3$ linearly independent rows suffice.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.76 — Face Recognition)</span></p>

Detection and recognition of faces from a digital image is a modern task in computer graphics. We represent a digital image as a matrix $A \in \mathbb{R}^{m \times n}$, where $a_{ij}$ specifies the color of the pixel at position $i, j$. The set of images containing faces can, with some degree of simplification, be thought of as a subspace of the space of all images $\mathbb{R}^{m \times n}$. The basis of this subspace consists of so-called *eigenfaces*, i.e., certain basic types or features of a face from which we compose other faces.

If we want to determine whether an image corresponds to a face, we check whether the corresponding vector lies in the subspace of faces or near it. We proceed similarly if we want to recognize whether a given image corresponds to some known face: In the vector space $\mathbb{R}^{m \times n}$ we determine which of the vectors corresponding to known faces is closest to the vector of our image.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 5.77 — Lagrange Interpolation Polynomial)</span></p>

Let us now return to the problem of interpolating points by a polynomial. Suppose we have $n + 1$ points $(x_0, y_0), (x_1, y_1), \ldots, (x_n, y_n)$ in the plane, where $x_i \neq x_j$ for $i \neq j$. The task is to find a polynomial $p(x)$ passing through these points.

The polynomials $1, x, x^2, \ldots, x^n$ form the standard basis of the vector space $\mathcal{P}^n$, and our goal is essentially to find the coordinates $a_0, a_1, \ldots, a_n$ of the desired polynomial $p(x)$ with respect to this basis.

Now the question arises whether we could find the polynomial more easily if we chose a different basis of $\mathcal{P}^n$. The answer is "yes". We choose the following basis of $\mathcal{P}^n$. For $i = 0, 1, \ldots, n$ we define the polynomial

$$p_i(x) = \prod_{j=0, j \neq i}^{n} \frac{1}{x_i - x_j}(x - x_j).$$

This polynomial has the value $1$ at the point $x_i$ and the value $0$ at all other points $x_j$, $j \neq i$. It is easy to see that these polynomials are linearly independent: no polynomial $p_i(x)$ is a linear combination of the others, because the other polynomials have the value $0$ at the point $x_i$. Hence the polynomials $p_0(x), \ldots, p_n(x)$ form a basis of the space $\mathcal{P}^n$. The interpolation polynomial $p(x)$ can thus be uniquely expressed as a linear combination, and the coordinates are precisely the function values $y_0, \ldots, y_n$. This gives us an explicit expression of the interpolation polynomial in the so-called Lagrange form

$$p(x) = \sum_{i=0}^{n} y_i p_i(x).$$

</div>

### Summary of Chapter 5

Vector spaces represent another abstract concept. We can add vectors in a space and multiply each individually by a scalar (not necessarily with each other!). Applying both operations to $n$ vectors yields a linear combination of these vectors. The set of all linear combinations of given vectors forms a vector subspace. If no strictly smaller subset of vectors generates the same subspace, the vectors are linearly independent; otherwise they are linearly dependent. Alternatively, vectors are dependent if among them there is at least one that is a linear combination of the others. Linearly independent generators of a space are called a basis of that space. Every space has some basis, and if there are several, they all have the same size (Steinitz Exchange Theorem). This justifies introducing the dimension of a space as the number of vectors in a basis. A basis of a space then represents a kind of coordinate system in that space, because every vector can be uniquely expressed as a linear combination of the basis vectors; the corresponding coefficients are called coordinates.

Spaces are closely related to matrices in two ways. Several vector spaces are associated with every matrix $A$: the one generated by columns, the one generated by rows, and the kernel, i.e., the solution space of the system $Ax = o$. By examining how elementary and other matrix operations change these spaces, we can in turn use matrices to easily solve many problems: determine whether given vectors are linearly independent, find the dimension of the space they generate, select a suitable basis from them, compute the coordinates of a vector in a given basis, etc.

---

## Chapter 6 — Linear Maps

We have already briefly encountered linear maps as maps of the form $x \mapsto Ax$, where $A \in \mathbb{T}^{m \times n}$. We observed that the map is a bijection precisely for nonsingular matrices, and the inverse map has the form $y \mapsto A^{-1}y$. Furthermore, we know that the space $\mathbb{T}^n$ maps onto the space $\mathcal{S}(A)$, whose dimension is $r = \operatorname{rank}(A)$. The difference of dimensions $n - r$ between the domain and the image then corresponds to the dimension of the kernel of the matrix $A$.

For the linear map $x \mapsto Ax$, the following clearly holds:

$$(x + y) \mapsto A(x + y) = Ax + Ay, \qquad (\alpha x) \mapsto A(\alpha x) = \alpha(Ax).$$

It is precisely this property that we will use as the definition of a linear map for general spaces. In other words, this property says that the image of the sum of two vectors equals the sum of their images, and analogously for scalar multiples. Consequently, the image of a linear combination of vectors can be expressed as a linear combination of their images. A linear map therefore preserves the relationships between vectors: linearly dependent vectors map to linearly dependent images (but not vice versa!); a vector that depends on other vectors maps to a vector dependent on their images with the same linear combination, etc.

Throughout this chapter, we consider only finitely generated vector spaces.

### 6.1 Linear Maps Between General Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 6.1 — Linear Map)</span></p>

Let $U, V$ be vector spaces over a field $\mathbb{T}$. A map $f \colon U \to V$ is *linear* if for every $x, y \in U$ and $\alpha \in \mathbb{T}$:

- $f(x + y) = f(x) + f(y)$,
- $f(\alpha x) = \alpha f(x)$.

</div>

A linear map is also called a *homomorphism*. An injective homomorphism is a *monomorphism*, a surjective homomorphism is an *epimorphism*, a homomorphism from a set to itself is an *endomorphism*, a surjective and injective homomorphism is an *isomorphism*, and an isomorphic endomorphism is called an *automorphism*.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.2 — Examples of Linear Maps in the Plane)</span></p>

Already in Example 3.21 we showed several linear maps given by $x \mapsto Ax$, where $A \in \mathbb{R}^{2 \times 2}$. These maps represented various transformations in the plane, specifically reflection about an axis, stretching along an axis, and rotation about the origin. Projection as a linear map was introduced in Remark 3.43.

A linear map with matrix $A = \binom{v_1 \ 0}{0 \ v_2}$ represents scaling that stretches by a factor of $v_1$ in the direction of the $x_1$-axis and by a factor of $v_2$ in the direction of the $x_2$-axis. Specifically, for the value $v = (0.6, 0.6)^\top$ we obtain a map that uniformly shrinks objects.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.3 — Rotation Matrix)</span></p>

We derive the expression for the linear map that represents rotation in the plane about the origin by an angle $\alpha$ counterclockwise. We identify the point $(x_1, x_2)^\top \in \mathbb{R}^2$ with the complex number $z := x_1 + ix_2$ and denote the complex number $r := \cos(\alpha) + i\sin(\alpha)$. As we know from Section 1.4, multiplication by $r$ represents rotation by the angle $\alpha$. Thus the complex number $z$ is rotated to the complex number

$$r \cdot z = (\cos(\alpha) + i\sin(\alpha)) \cdot (x_1 + ix_2) = \cos(\alpha)x_1 - \sin(\alpha)x_2 + i(\sin(\alpha)x_1 + \cos(\alpha)x_2).$$

If we identify complex numbers back with points in the plane, we obtain that the point $(x_1, x_2)^\top$ maps to the point $(\cos(\alpha)x_1 - \sin(\alpha)x_2, \ \sin(\alpha)x_1 + \cos(\alpha)x_2)^\top$. Thus rotation forms a linear map and its matrix expression is $x \mapsto Ax$, where

$$A = \begin{pmatrix} \cos(\alpha) & -\sin(\alpha) \\ \sin(\alpha) & \cos(\alpha) \end{pmatrix}.$$

In particular, the vector $e_1 = (1, 0)^\top$ maps to $(\cos(\alpha), \sin(\alpha))^\top$ and the vector $e_2 = (0,1)^\top$ maps to $(-\sin(\alpha), \cos(\alpha))^\top$.

Specifically, the rotation matrix for $90°$ and the rotation matrix for $180°$ have the form

$$\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \quad \text{a} \quad \begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}.$$

The rotation matrix can be easily generalized to the case of rotation in the space $\mathbb{R}^n$, provided we restrict ourselves to rotation by an angle $\alpha$ in the plane of the axes $x_i, x_j$. Schematically (empty spaces correspond to zeros):

$$\begin{pmatrix} I & & & \\ & \cos(\alpha) & & -\sin(\alpha) & \\ & & I & & \\ & \sin(\alpha) & & \cos(\alpha) & \\ & & & & I \end{pmatrix}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.4 — Further Examples of Linear Maps)</span></p>

- A typical example of a linear map is $f \colon \mathbb{R}^n \to \mathbb{R}^m$ defined by $f(x) = Ax$, where $A \in \mathbb{R}^{m \times n}$ is a fixed matrix. As we will see later in Corollary 6.20, no other linear map between the spaces $\mathbb{R}^n$ and $\mathbb{R}^m$ exists.
- The trivial map $f \colon U \to V$ defined by $f(x) = o$ is obviously linear.
- The identity $id \colon U \to U$ defined by $id(x) = x$ is another example of a linear map.
- The map $f \colon \mathbb{T}^{m \times n} \to \mathbb{T}^{n \times m}$ given by $f(A) = A^\top$ is linear due to the properties of matrix transposition (Proposition 3.13).
- The derivative from the space of real differentiable functions to the space of real functions $\mathcal{F}$ also represents a linear map, because it satisfies the properties $(f + g)' = f' + g'$ and $(\alpha f)' = \alpha f'$ for any two functions $f, g$ and scalar $\alpha \in \mathbb{R}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 6.5 — Properties of Linear Maps)</span></p>

Let $f \colon U \to V$ be a linear map. Then

1. $f\!\left(\sum_{i=1}^{n} \alpha_i x_i\right) = \sum_{i=1}^{n} \alpha_i f(x_i)$ for every $\alpha_i \in \mathbb{T}$, $x_i \in U$, $i = 1, \ldots, n$,
2. $f(o) = o$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 6.6 — Geometric Property)</span></p>

One of the geometric properties of linear maps is that they map a line to a line or to a point. A line (see p. 120) determined by two distinct vectors $v_1, v_2$ is the set of vectors of the form $\lambda v_1 + (1 - \lambda)v_2$, where $\lambda \in \mathbb{T}$. The image of this set under a linear map $f$ is the set described by $f(\lambda v_1 + (1 - \lambda)v_2) = \lambda f(v_1) + (1 - \lambda)f(v_2)$, which is again a line or a point (if $f(v_1) = f(v_2)$). Note that the converse does not hold: not every map preserving lines is linear. For example, translation is nonlinear, but it maps lines to lines.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 6.7 — Image and Kernel)</span></p>

Let $f \colon U \to V$ be a linear map. We define

- **image** $f(U) := \lbrace f(x);\ x \in U \rbrace$,
- **kernel** $\operatorname{Ker}(f) := \lbrace x \in U;\ f(x) = o \rbrace$.

The image has the natural meaning as the range of the map. The definition can be extended to the image of any subset $M \subseteq U$ as follows: $f(M) := \lbrace f(x);\ x \in M \rbrace$.

</div>

The kernel describes certain features of the linear map. A trivial kernel (i.e., $\operatorname{Ker}(f) = \lbrace o \rbrace$) means that the map is injective, and therefore the dimensions of the domain $U$ and the image $f(U)$ are the same. Conversely, the larger the kernel, the more the map degenerates, more vectors map to the same value, and the smaller the dimension of the image $f(U)$ relative to the dimension of the domain $U$ (see Corollary 6.43).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 6.8)</span></p>

The kernel of a matrix and the kernel of a linear map are closely related. If we define the map $f$ by $f(x) = Ax$, then $\operatorname{Ker}(f) = \operatorname{Ker}(A)$ and $f(U) = \mathcal{S}(A)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.9 — Image and Kernel)</span></p>

Consider the linear map $x \mapsto Ax$, where $A \in \mathbb{R}^{2 \times 2}$, see Example 6.2.

- For the matrix $A = \binom{-1 \ 0}{\ 0 \ 1}$, the map represents reflection about the $x_2$-axis. The image is $f(\mathbb{R}^2) = \mathbb{R}^2$ and the kernel is $\operatorname{Ker}(f) = \lbrace o \rbrace$.
- For the matrix $A = \binom{1 \ 0}{0 \ 0}$, we obtain the projection onto the $x_1$-axis. The image is now $f(\mathbb{R}^2) = \operatorname{span}\lbrace (1, 0)^\top \rbrace$, i.e., the $x_1$-axis, and the kernel is $\operatorname{Ker}(f) = \operatorname{span}\lbrace (0, 1)^\top \rbrace$, i.e., the $x_2$-axis.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 6.10)</span></p>

Let $f \colon U \to V$ be a linear map. Then:

1. $f(U)$ is a subspace of $V$,
2. $\operatorname{Ker}(f)$ is a subspace of $U$,
3. for every $x_1, \ldots, x_n \in U$: $f(\operatorname{span}\lbrace x_1, \ldots, x_n \rbrace) = \operatorname{span}\lbrace f(x_1), \ldots, f(x_n) \rbrace$.

</div>

Part (3) of Proposition 6.10 also provides a method for determining the image of a subspace $W$ of $U$: we determine the images of a basis (or more generally, generators of $W$), and these form the generators of the image $f(W)$.

Let us recall two types of maps: injective and surjective. A linear map $f \colon U \to V$ is surjective if $f(U) = V$. In other words, for every vector $y \in V$ there exists a vector $x \in U$ that maps to it, i.e., $f(x) = y$. To determine whether the map $f$ is surjective, one can easily use part (3) of Proposition 6.10. It suffices to choose generators of the space $U$ and verify whether their images generate the space $V$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 6.11)</span></p>

A linear map $f \colon U \to V$ is surjective if and only if some generators of the space $U$ map to generators of the space $V$.

</div>

A linear map $f \colon U \to V$ is injective if $f(x) = f(y)$ occurs only for $x = y$. In other words, for any two vectors $x, y \in U$, $x \neq y$, we have $f(x) \neq f(y)$. The following theorem characterizes when a linear map is injective.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 6.12 — Injective Linear Map)</span></p>

Let $f \colon U \to V$ be a linear map. Then the following are equivalent:

1. $f$ is injective,
2. $\operatorname{Ker}(f) = \lbrace o \rbrace$,
3. the image of any linearly independent set is a linearly independent set.

</div>

*Proof.* We prove the implications $(1) \Rightarrow (2) \Rightarrow (3) \Rightarrow (1)$.

- Implication $(1) \Rightarrow (2)$: Since $f(o) = o$, we have $o \in \operatorname{Ker}(f)$. But since $f$ is injective, the kernel contains no other element.
- Implication $(2) \Rightarrow (3)$: Let $x_1, \ldots, x_n \in U$ be linearly independent and suppose $\sum_{i=1}^{n} \alpha_i f(x_i) = o$. Then $f(\sum_{i=1}^{n} \alpha_i x_i) = o$, so $\sum_{i=1}^{n} \alpha_i x_i$ belongs to the kernel $\operatorname{Ker}(f) = \lbrace o \rbrace$. Therefore $\sum_{i=1}^{n} \alpha_i x_i = o$, and by linear independence of the vectors we have $\alpha_i = 0$ for all $i$.
- Implication $(3) \Rightarrow (1)$: For contradiction, assume there exist two distinct vectors $x, y \in U$ such that $f(x) = f(y)$. Then $o = f(x) - f(y) = f(x - y)$. The vector $o$ represents a linearly dependent set of vectors, so $x - y$ must also be a linearly dependent set by assumption (3), and therefore $x - y = o$, i.e., $x = y$. This is a contradiction.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.13 — Injective Linear Map)</span></p>

Consider the linear map from Example 6.9, i.e., $x \mapsto Ax$, where $A \in \mathbb{R}^{2 \times 2}$.

- For the matrix $A = \binom{-1 \ 0}{\ 0 \ 1}$, the map represents reflection about the $x_2$-axis. Since the kernel is $\operatorname{Ker}(f) = \lbrace o \rbrace$, the map is injective.
- For the matrix $A = \binom{1 \ 0}{0 \ 0}$, the map represents projection onto the $x_1$-axis. Since the kernel is $\operatorname{Ker}(f) = \operatorname{span}\lbrace (0, 1)^\top \rbrace$, this is not an injective map.

</div>

In particular, part (3) of Theorem 6.12 says that an injective linear map $f \colon U \to V$ maps a basis of the space $U$ to a basis of $f(U)$. Consequently, an injective map satisfies $\dim U = \dim f(U)$. Later (Corollary 6.43) we will see that this equality fully characterizes injective maps.

Even an injective linear map need not always be surjective, as witnessed for example by the embedding of $\mathbb{R}^n$ into $\mathbb{R}^{n+1}$ defined by $(v_1, \ldots, v_n)^\top \mapsto (v_1, \ldots, v_n, 0)^\top$.

For vector spaces, we know that every (finitely generated) subspace is uniquely determined by some basis. We would like such a minimal representation for linear maps as well. As we will see, a certain analogy holds for linear maps, since every linear map is uniquely determined by where it sends the vectors of a basis.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.14)</span></p>

Consider a linear map $f \colon \mathbb{R}^2 \to V$. If we only know the image of a vector $x \neq o$, then we can determine the images of all its scalar multiples, i.e., vectors on the line $\operatorname{span}\lbrace x \rbrace$, simply from the relation $f(\alpha x) = \alpha f(x)$. However, we cannot reconstruct the entire map. For that, we need to also know the image of some other (linearly independent) vector $y$. Then we can compute the image not only of all scalar multiples of vectors $x$ and $y$, but also of their sums and all linear combinations, i.e., all vectors of the space $\mathbb{R}^2$ from the relation $f(\alpha x + \beta y) = \alpha f(x) + \beta f(y)$. Thus the linear map $f$ is characterized solely by the images of two linearly independent vectors, i.e., a basis.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 6.15 — Linear Map and Uniqueness with Respect to Basis Images)</span></p>

Let $U, V$ be spaces over $\mathbb{T}$ and $x_1, \ldots, x_n$ a basis of $U$. For arbitrary vectors $y_1, \ldots, y_n \in V$ there exists exactly one linear map such that $f(x_i) = y_i$, $i = 1, \ldots, n$.

</div>

*Proof.* "Existence." Let $x \in U$ be arbitrary. Then $x = \sum_{i=1}^{n} \alpha_i x_i$ for some scalars $\alpha_1, \ldots, \alpha_n \in \mathbb{T}$. We define the image of $x$ as $f(x) = \sum_{i=1}^{n} \alpha_i y_i$, because a linear map must satisfy $f(x) = f\!\left(\sum_{i=1}^{n} \alpha_i x_i\right) = \sum_{i=1}^{n} \alpha_i f(x_i) = \sum_{i=1}^{n} \alpha_i y_i$. That the map defined in this way is indeed linear is then easy to verify.

"Uniqueness." Suppose we have two distinct linear maps $f$ and $g$ satisfying $f(x_i) = g(x_i) = y_i$ for all $i = 1, \ldots, n$. Then for arbitrary $x \in U$, expressed as $x = \sum_{i=1}^{n} \alpha_i x_i$, we have $f(x) = \sum_{i=1}^{n} \alpha_i f(x_i) = \sum_{i=1}^{n} \alpha_i y_i = \sum_{i=1}^{n} \alpha_i g(x_i) = g(x)$. Thus $f(x) = g(x)$ $\forall x \in U$, which contradicts the assumption that they are distinct maps.

### 6.2 Matrix Representation of a Linear Map

Every linear map between (finitely generated) vector spaces can be represented by a matrix. Since vectors can be various objects, it is convenient to describe them in the language of coordinates. Then we can operate with them as with arithmetic vectors, which is often more convenient.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.16 — Introduction to the Matrix of a Linear Map)</span></p>

Consider a linear map $f \colon \mathbb{T}^n \to \mathbb{T}^m$. Then for arbitrary $x \in \mathbb{T}^n$ we have

$$f(x) = f\!\left(\sum_{i=1}^{n} x_i e_i\right) = \sum_{i=1}^{n} x_i f(e_i).$$

Denoting the matrix with columns $f(e_1), \ldots, f(e_n)$ as

$$A = \begin{pmatrix} | & & | \\ f(e_1) & \cdots & f(e_n) \\ | & & | \end{pmatrix},$$

we clearly have $f(x) = Ax$. Every linear map $f \colon \mathbb{T}^n \to \mathbb{T}^m$ can therefore be represented by a matrix as $f(x) = Ax$.

</div>

Now consider a linear map $f \colon U \to \mathbb{T}^m$ and a basis $B = \lbrace v_1, \ldots, v_n \rbrace$ of the space $U$. Let the vector $x \in U$ have the expression $x = \sum_{i=1}^{n} \alpha_i v_i$, i.e., $[x]_B = (\alpha_1, \ldots, \alpha_n)^\top$. Then

$$f(x) = f\!\left(\sum_{i=1}^{n} \alpha_i v_i\right) = \sum_{i=1}^{n} \alpha_i f(v_i).$$

Denoting the matrix with columns $f(v_1), \ldots, f(v_n)$ as

$$A = \begin{pmatrix} | & & | \\ f(v_1) & \cdots & f(v_n) \\ | & & | \end{pmatrix},$$

we clearly have $f(x) = A \cdot [x]_B$. Unlike the previous case, we multiply the matrix by the coordinate vector $[x]_B$ of the vector $x$, not by the vector itself.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 6.17 — Matrix of a Linear Map)</span></p>

Let $f \colon U \to V$ be a linear map, $B_U = \lbrace x_1, \ldots, x_n \rbrace$ a basis of the space $U$ over $\mathbb{T}$, and $B_V = \lbrace y_1, \ldots, y_m \rbrace$ a basis of the space $V$ over $\mathbb{T}$. Let $f(x_j) = \sum_{i=1}^{m} a_{ij} y_i$. Then the matrix $A \in \mathbb{T}^{m \times n}$ with entries $a_{ij}$, $i = 1, \ldots, m$, $j = 1, \ldots, n$, is called the *matrix of the linear map* $f$ with respect to the bases $B_U, B_V$ and is denoted ${}_{B_V}[f]_{B_U}$.

</div>

In other words, the matrix of a linear map is such that its $j$-th column consists of the coordinates of the image of the vector $x_j$ with respect to the basis $B_V$, that is

$${}_{B_V}[f]_{B_U} = \begin{pmatrix} | & & | \\ [f(x_1)]_{B_V} & \cdots & [f(x_n)]_{B_V} \\ | & & | \end{pmatrix}.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.18 — Matrix of a Linear Map)</span></p>

Consider the linear map $f \colon \mathbb{R}^2 \to \mathbb{R}^2$ given by $f(x) = Ax$, where

$$A = \begin{pmatrix} 1 & 2 \\ 3 & -4 \end{pmatrix}.$$

Let us choose the bases $B_U = \lbrace (1, 2)^\top, (2, 1)^\top \rbrace$, $B_V = \lbrace (1, -1)^\top, (0, 1)^\top \rbrace$ and find the matrix of the map $f$ with respect to the bases $B_U, B_V$.

The image of the first vector of the basis $B_U$ is $f(1, 2) = (5, -5)^\top$, and its coordinates with respect to the basis $B_V$ are $[f(1, 2)]_{B_V} = (5, 0)^\top$. Similarly, the image of the second vector of the basis $B_U$ is $f(2, 1) = (4, 2)^\top$, and its coordinates with respect to the basis $B_V$ are $[f(2, 1)]_{B_V} = (4, 6)^\top$. Therefore

$${}_{B_V}[f]_{B_U} = \begin{pmatrix} 5 & 4 \\ 0 & 6 \end{pmatrix}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 6.19 — Matrix Representation of a Linear Map)</span></p>

Let $f \colon U \to V$ be a linear map, $B_U = \lbrace x_1, \ldots, x_n \rbrace$ a basis of the space $U$, and $B_V = \lbrace y_1, \ldots, y_m \rbrace$ a basis of the space $V$. Then for every $x \in U$:

$$[f(x)]_{B_V} = {}_{B_V}[f]_{B_U} \cdot [x]_{B_U}. \tag{6.1}$$

</div>

*Proof.* Denote $A := {}_{B_V}[f]_{B_U}$. Let $x \in U$, so $x = \sum_{i=1}^{n} \alpha_i x_i$, i.e., $[x]_{B_U} = (\alpha_1, \ldots, \alpha_n)^\top$. Then

$$f(x) = f\!\left(\sum_{j=1}^{n} \alpha_j x_j\right) = \sum_{j=1}^{n} \alpha_j f(x_j) = \sum_{j=1}^{n} \alpha_j \left(\sum_{i=1}^{m} a_{ij} y_i\right) = \sum_{i=1}^{m} \left(\sum_{j=1}^{n} \alpha_j a_{ij}\right) y_i.$$

Thus the expression $\sum_{j=1}^{n} \alpha_j a_{ij}$ represents the $i$-th coordinate of the vector $[f(x)]_{B_V}$, and its value is $\sum_{j=1}^{n} \alpha_j a_{ij} = (A \cdot [x]_{B_U})_i$, which is the $i$-th component of the vector ${}_{B_V}[f]_{B_U} \cdot [x]_{B_U}$.

The matrix of a linear map thus converts the coordinates of a vector with respect to a given basis into the coordinates of its image, and moreover, the image of any vector can be expressed simply as matrix multiplication.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 6.20)</span></p>

Every linear map $f \colon \mathbb{T}^n \to \mathbb{T}^m$ can be expressed as $f(x) = Ax$ for some matrix $A \in \mathbb{T}^{m \times n}$.

</div>

*Proof.* For every $x \in \mathbb{T}^n$ we have $f(x) = [f(x)]_{\text{kan}} = {}_{\text{kan}}[f]_{\text{kan}} \cdot [x]_{\text{kan}} = {}_{\text{kan}}[f]_{\text{kan}} \cdot x$. Thus $f(x) = Ax$, where $A = {}_{\text{kan}}[f]_{\text{kan}}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 6.21 — Uniqueness of the Matrix of a Linear Map)</span></p>

Let $f \colon U \to V$ be a linear map, $B_U$ a basis of the space $U$, and $B_V$ a basis of the space $V$. Then the only matrix $A$ satisfying (6.1) is $A = {}_{B_V}[f]_{B_U}$.

</div>

*Proof.* Let the basis $B_U$ consist of vectors $z_1, \ldots, z_n$. For contradiction, assume that the linear map $f$ has two matrix representations (6.1) via matrices $A \neq A'$. Then there exists a vector $s \in \mathbb{T}^n$ such that $As \neq A's$; such a vector can be chosen, for example, as the unit vector with a one in the position where the columns of the matrices $A, A'$ differ. Define the vector $x := \sum_{i=1}^{n} s_i z_i$. Then $[f(x)]_{B_V} = As \neq A's = [f(x)]_{B_V}$, which contradicts the uniqueness of coordinates (Theorem 5.33).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 6.22)</span></p>

Not only can every linear map be represented by a matrix, but conversely, every matrix represents the matrix of some linear map. Let $B_U, B_V$ be bases of spaces $U, V$ of dimensions $n, m$, and let $A \in \mathbb{T}^{m \times n}$. Then there exists a unique linear map $f \colon U \to V$ such that $A = {}_{B_V}[f]_{B_U}$; the columns of the matrix $A$ give the coordinates of the images of the basis vectors $B_U$, which fully determines the map $f$ by Theorem 6.15. This means that there is a one-to-one correspondence between linear maps $f \colon U \to V$ and the space of matrices $\mathbb{T}^{m \times n}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 6.23 — Change-of-Basis Matrix)</span></p>

Let $V$ be a vector space and $B_1, B_2$ two of its bases. Then the *change-of-basis matrix* from $B_1$ to $B_2$ is the matrix ${}_{B_2}[id]_{B_1}$.

</div>

The change-of-basis matrix then has the following meaning according to the matrix representation: Let $x \in U$, then

$$[x]_{B_2} = {}_{B_2}[id]_{B_1} \cdot [x]_{B_1},$$

so by simple matrix multiplication we obtain coordinates with respect to a different basis. Clearly ${}_{B}[id]_{B} = I_n$ for any basis $B$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.24 — Change-of-Basis Matrix)</span></p>

Find the change-of-basis matrix in $\mathbb{R}^3$ from the basis

$$B_1 = \lbrace (1, 1, -1)^\top, (3, -2, 0)^\top, (2, -1, 1)^\top \rbrace$$

to the basis

$$B_2 = \lbrace (8, -4, 1)^\top, (-8, 5, -2)^\top, (3, -2, 1)^\top \rbrace.$$

Solution: we compute

$$[(1, 1, -1)^\top]_{B_2} = (2, 3, 3)^\top, \quad [(3, -2, 0)^\top]_{B_2} = (-1, -4, -7)^\top, \quad [(2, -1, 1)^\top]_{B_2} = (1, 3, 6)^\top.$$

Therefore

$${}_{B_2}[id]_{B_1} = \begin{pmatrix} 2 & -1 & 1 \\ 3 & -4 & 3 \\ 3 & -7 & 6 \end{pmatrix}.$$

If we know, for example, that the coordinates of the vector $(4, -1, -1)^\top$ with respect to the basis $B_1$ are $(1, 1, 0)^\top$, then the coordinates with respect to $B_2$ are obtained as

$$[(4, -1, -1)^\top]_{B_2} = {}_{B_2}[id]_{B_1} \cdot [(4, -1, -1)^\top]_{B_1} = {}_{B_2}[id]_{B_1} \cdot (1, 1, 0)^\top = (1, -1, -4)^\top.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.25)</span></p>

Let $B$ be a basis of the space $\mathbb{T}^n$. By the matrix representation of a linear map, we obtain in particular

$$[x]_B = {}_{B}[id]_{\text{kan}} \cdot [x]_{\text{kan}} = {}_{B}[id]_{\text{kan}} \cdot x.$$

The coordinates of any vector are thus obtained simply by multiplying the change-of-basis matrix by the vector $x$.

</div>

An essential role in the theory of linear maps is played by their composition. Recall that for maps $f \colon U \to V$ and $g \colon V \to W$, the composite map $g \circ f$ is defined by $(g \circ f)(x) := g(f(x))$, $x \in U$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 6.26 — Composition of Linear Maps)</span></p>

Let $f \colon U \to V$, $g \colon V \to W$ be linear maps. Then the composite map $g \circ f$ is again a linear map.

</div>

Consider two linear maps $f \colon \mathbb{T}^n \to \mathbb{T}^p$ and $g \colon \mathbb{T}^p \to \mathbb{T}^m$ represented by matrices $f(x) = Ax$, $g(y) = By$ for certain matrices $A \in \mathbb{T}^{p \times n}$, $B \in \mathbb{T}^{m \times p}$. Then the composite map has the form

$$(g \circ f)(x) = g(f(x)) = B(Ax) = (BA)x.$$

This is therefore a linear map represented by the matrix $BA$ (see Remark 3.20). This property holds more generally: the matrix of a composite linear map equals the product of the matrices of the corresponding maps.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 6.27 — Matrix of a Composite Linear Map)</span></p>

Let $f \colon U \to V$ and $g \colon V \to W$ be linear maps, let $B_U$ be a basis of $U$, $B_V$ a basis of $V$, and $B_W$ a basis of $W$. Then

$${}_{B_W}[g \circ f]_{B_U} = {}_{B_W}[g]_{B_V} \cdot {}_{B_V}[f]_{B_U}. \tag{6.2}$$

</div>

*Proof.* For every $x \in U$ we have

$$[(g \circ f)(x)]_{B_W} = [g(f(x))]_{B_W} = {}_{B_W}[g]_{B_V} \cdot [f(x)]_{B_V} = {}_{B_W}[g]_{B_V} \cdot {}_{B_V}[f]_{B_U} \cdot [x]_{B_U}.$$

By the uniqueness of the matrix of a linear map (Theorem 6.21), ${}_{B_W}[g]_{B_V} \cdot {}_{B_V}[f]_{B_U}$ is the desired matrix of the composite map.

In formula (6.2), the mnemonic in the notation of matrices of linear maps is again useful. Specifically, the matrix of the map $g \circ f$ has the same input basis $B_U$ as the matrix of the map $f$ and the same output basis $B_W$ as the matrix of the map $g$. Moreover, the output basis $B_V$ of the matrix of the map $f$ must be the same as the input basis of the matrix of the map $g$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.28 — Composition of Rotations and Addition Formulas for sin and cos)</span></p>

Rotation in the plane by an angle $\alpha$ counterclockwise has, with respect to the canonical basis, the matrix

$$\begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix},$$

see Example 6.3. Similarly for rotation by an angle $\beta$. The rotation matrix for the angle $\alpha + \beta$ can be obtained either by directly substituting the value $\alpha + \beta$ into the rotation matrix or by composing the rotation by the angle $\alpha$ and then the rotation by the angle $\beta$. By comparison, we obtain the addition formulas for sin and cos:

$$\begin{pmatrix} \cos(\alpha + \beta) & -\sin(\alpha + \beta) \\ \sin(\alpha + \beta) & \cos(\alpha + \beta) \end{pmatrix} = \begin{pmatrix} \cos\beta & -\sin\beta \\ \sin\beta & \cos\beta \end{pmatrix} \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix} = \begin{pmatrix} \cos\alpha\cos\beta - \sin\alpha\sin\beta & -\sin\alpha\cos\beta - \sin\beta\cos\alpha \\ \cos\alpha\sin\beta + \cos\beta\sin\alpha & -\sin\alpha\sin\beta + \cos\alpha\cos\beta \end{pmatrix}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.29 — Converting the Matrix of a Map Between Bases)</span></p>

Suppose we are given the matrix of a linear map $f$ with respect to bases $B_1, B_2$, i.e., ${}_{B_2}[f]_{B_1}$. How do we determine the matrix with respect to bases $B_3, B_4$, i.e., ${}_{B_4}[f]_{B_3}$? By the theorem on the matrix of a composite map applied to $f = id \circ f \circ id$ and the appropriate bases, we have

$${}_{B_4}[f]_{B_3} = {}_{B_4}[id]_{B_2} \cdot {}_{B_2}[f]_{B_1} \cdot {}_{B_1}[id]_{B_3}.$$

Thus all the work is done by the change-of-basis matrices.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 6.30 — Multivariable Derivatives and Composition of Maps)</span></p>

Consider differentiable functions $f(x) \colon \mathbb{R}^n \to \mathbb{R}^p$, $g(y) \colon \mathbb{R}^p \to \mathbb{R}^m$ and points $x^* \in \mathbb{R}^n$ and $y^* := f(x^*)$. From a course on multivariable calculus, we know the formula for partial derivatives of a composite map

$$\frac{\partial (g \circ f)_i}{\partial x_k} = \sum_{j=1}^{p} \frac{\partial g_i}{\partial y_j} \cdot \frac{\partial f_j}{\partial x_k}.$$

In the language of Jacobian matrices

$$\nabla f = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & & \vdots \\ \frac{\partial f_p}{\partial x_1} & \cdots & \frac{\partial f_p}{\partial x_n} \end{pmatrix}$$

the above formula takes the form

$$\nabla (g \circ f)(x^*) = \nabla g(y^*) \cdot \nabla f(x^*). \tag{6.3}$$

The Jacobian matrix is the matrix of the linear map that (locally best) approximates a smooth map. Formula (6.3) says that when composing smooth maps, their linear approximations compose in the corresponding way. The formula thus also illustrates Theorem 6.27 on the matrix of a composite linear map.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.31 — Translation as a Linear Map?)</span></p>

Let $v \in \mathbb{R}^n$ be fixed and consider the map $f \colon \mathbb{R}^n \to \mathbb{R}^n$ given by $f(x) = x + v$. This map is not linear because it does not map the zero vector to the zero vector. However, we can simulate it as a linear map using techniques from classical projective geometry. We embed the space $\mathbb{R}^n$ into a space of one higher dimension so that for a certain linear map $g \colon \mathbb{R}^{n+1} \to \mathbb{R}^{n+1}$ we have

$$g(x_1, \ldots, x_n, 1) = (x_1 + v_1, \ldots, x_n + v_n, 1).$$

We extend $g$ to the remaining points so that it forms a linear map

$$g(x_1, \ldots, x_n, x_{n+1}) = (x_1 + v_1 x_{n+1}, \ldots, x_n + v_n x_{n+1}, x_{n+1}).$$

The matrix of this map is

$$\begin{pmatrix} 1 & 0 & \cdots & 0 & v_1 \\ 0 & 1 & \cdots & 0 & v_2 \\ \vdots & & \ddots & & \vdots \\ 0 & 0 & \cdots & 1 & v_n \\ 0 & 0 & \cdots & 0 & 1 \end{pmatrix}.$$

</div>

### 6.3 Isomorphism

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 6.32 — Isomorphism)</span></p>

An *isomorphism* between spaces $U, V$ over a field $\mathbb{T}$ is a bijective linear map $f \colon U \to V$. If an isomorphism exists between spaces $U, V$, then we say that $U, V$ are *isomorphic*.

</div>

Isomorphic spaces behave the same from the perspective of linear algebra. An isomorphism maps linearly dependent vectors to linearly dependent ones with the same relations (because it is a linear map), maps linearly independent vectors to linearly independent ones (because it is an injective map), preserves dimension, maps a basis to a basis, etc.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.33)</span></p>

Examples of isomorphisms include scaling, reflection in $\mathbb{R}^2$ (Example 6.2), or rotation (Example 6.28). An example of a linear map that is not an isomorphism is projection (Example 6.2).

An example of isomorphic spaces is $\mathcal{P}^n$ and $\mathbb{R}^{n+1}$, where a suitable (and not the only) isomorphism is

$$a_n x^n + \ldots + a_1 x + a_0 \mapsto (a_n, \ldots, a_1, a_0).$$

Another example of isomorphic spaces is $\mathbb{R}^{m \times n}$ and $\mathbb{R}^{mn}$, where a suitable isomorphism is for instance

$$A \mapsto (a_{11}, \ldots, a_{1n}, a_{21}, \ldots, a_{2n}, \ldots, a_{m1}, \ldots, a_{mn}).$$

The vector space $\mathbb{C}^n$ over $\mathbb{R}$ is isomorphic to the space $\mathbb{R}^{2n}$ over $\mathbb{R}$. A specific isomorphism is, for example, the map that sends the vector $(a_1 + ib_1, a_n + ib_n)^\top \in \mathbb{C}^n$ to the vector $(a_1, b_1, \ldots, a_n, b_n)^\top \in \mathbb{R}^{2n}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 6.34 — Properties of Isomorphism)</span></p>

1. If $f \colon U \to V$ is an isomorphism, then $f^{-1} \colon V \to U$ exists and is also an isomorphism.
2. If $f \colon U \to V$ and $g \colon V \to W$ are isomorphisms, then $g \circ f \colon U \to W$ is also an isomorphism.
3. A linear map $f \colon U \to V$ is an isomorphism if and only if any basis of the space $U$ maps to a basis of the space $V$.
4. If $f \colon U \to V$ is an isomorphism, then $\dim U = \dim V$.

</div>

*Proof.*

1. The map $f$ is bijective, so $f^{-1}$ exists and is also bijective. It remains to prove linearity. Let $v_1, v_2 \in V$ and let $f^{-1}(v_1) = u_1$ and $f^{-1}(v_2) = u_2$. Then $f(u_1 + u_2) = f(u_1) + f(u_2) = v_1 + v_2$, so $f^{-1}(v_1 + v_2) = u_1 + u_2 = f^{-1}(v_1) + f^{-1}(v_2)$. Similarly for scalar multiples: Let $v \in V$ and $f^{-1}(v) = u$, then $f(\alpha u) = \alpha f(u) = \alpha v$, so $f^{-1}(\alpha v) = \alpha u = \alpha f^{-1}(v)$.
2. Easy from Proposition 6.26.
3. Let $x_1, \ldots, x_n$ be a basis of $U$. Since $f$ is injective, by Theorem 6.12(3) the images $f(x_1), \ldots, f(x_n)$ are linearly independent. Since $f$ is surjective, the vectors $f(x_1), \ldots, f(x_n)$ generate the space $f(U) = V$ by Proposition 6.10(3). Thus the vectors $f(x_1), \ldots, f(x_n)$ form a basis of $V$. — Conversely, let $x_1, \ldots, x_n$ be a basis of $U$ and $f(x_1), \ldots, f(x_n)$ a basis of $V$. Then the map $f$ is clearly surjective. That the map $f$ is injective can be seen by contradiction: Suppose the kernel $\operatorname{Ker}(f)$ contains a nonzero vector. Then for some nontrivial linear combination we have $f(\sum_{i=1}^{n} \alpha_i x_i) = o$. By linearity of the map we get $\sum_{i=1}^{n} \alpha_i f(x_i) = o$, which contradicts the linear independence of the vectors $f(x_1), \ldots, f(x_n)$.
4. Follows from the previous part.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 6.35)</span></p>

Let $f \colon U \to V$ be an isomorphism, $B_U$ a basis of $U$, and $B_V$ a basis of $V$. Then

$${}_{B_U}[f^{-1}]_{B_V} = {}_{B_V}[f]_{B_U}^{-1}.$$

</div>

*Proof.* Since $f^{-1} \circ f = id$, we obtain

$${}_{B_U}[f^{-1}]_{B_V} \cdot {}_{B_V}[f]_{B_U} = {}_{B_U}[f^{-1} \circ f]_{B_U} = {}_{B_U}[id]_{B_U} = I.$$

Since ${}_{B_V}[f]_{B_U}$ is square by Theorem 6.34(4), ${}_{B_U}[f^{-1}]_{B_V}$ is its inverse matrix.

The matrix of an isomorphism has an inverse matrix, so it must be nonsingular. This statement also holds in reverse: If the matrix of a linear map $f$ is nonsingular, then $f$ is an isomorphism, because the inverse matrix gives the formula for the inverse map $f^{-1}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 6.36)</span></p>

A linear map $f \colon U \to V$ is an isomorphism if and only if some (any) matrix representing $f$ is nonsingular.

</div>

A further corollary of Proposition 6.35 is obtained specifically for the change-of-basis matrix between bases $B_U$ and $B_V$, namely

$${}_{B_U}[id]_{B_V} = {}_{B_V}[id]_{B_U}^{-1}.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.37 — Mnemonic for Computing Change-of-Basis Matrices)</span></p>

For computing the change-of-basis matrix in $\mathbb{R}^n$ from basis $B_U$ to basis $B_V$, i.e., ${}_{B_V}[id]_{B_U}$, one can use the following mnemonic:

$$(B_V \mid B_U) \xrightarrow{\text{RREF}} (I_n \mid {}_{B_V}[id]_{B_U}).$$

The first matrix has the basis $B_V$ in its columns and then the basis $B_U$, which are essentially the matrices ${}_{\text{kan}}[id]_{B_V}$ and ${}_{\text{kan}}[id]_{B_U}$. By reducing to RREF form, we obtain the desired change-of-basis matrix on the right. The reason stems from the relation ${}_{B_V}[id]_{B_U} = {}_{B_V}[id]_{\text{kan}} \cdot {}_{\text{kan}}[id]_{B_U} = {}_{\text{kan}}[id]_{B_V}^{-1} \cdot {}_{\text{kan}}[id]_{B_U}$. Reducing the matrix to RREF form can be expressed as left-multiplication by the matrix ${}_{\text{kan}}[id]_{B_V}^{-1}$.

Specifically, for Example 6.24, we obtain

$$\begin{pmatrix} 8 & -8 & 3 & 1 & 3 & 2 \\ -4 & 5 & -2 & 1 & -2 & -1 \\ 1 & -2 & 1 & -1 & 0 & 1 \end{pmatrix} \xrightarrow{\text{RREF}} \begin{pmatrix} 1 & 0 & 0 & 2 & -1 & 1 \\ 0 & 1 & 0 & 3 & -4 & 3 \\ 0 & 0 & 1 & 3 & -7 & 6 \end{pmatrix}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 6.38)</span></p>

Let $V$ be a vector space over a field $\mathbb{T}$ of dimension $n$ with basis $B$. Then the map $x \mapsto [x]_B$ is an isomorphism between the spaces $V$ and $\mathbb{T}^n$.

</div>

*Proof.* Let the basis $B$ consist of vectors $v_1, \ldots, v_n$. It is easy to see that the map $x \mapsto [x]_B$ is linear, injective, and surjective, because every $n$-tuple $(\alpha_1, \ldots, \alpha_n) \in \mathbb{T}^n$ represents the coordinates of a specific vector $\sum_{i=1}^{n} \alpha_i v_i$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 6.39 — Isomorphism of $n$-dimensional Spaces)</span></p>

All $n$-dimensional vector spaces over a field $\mathbb{T}$ are mutually isomorphic.

</div>

*Proof.* By Proposition 6.38, all $n$-dimensional vector spaces over a field $\mathbb{T}$ are isomorphic to $\mathbb{T}^n$, and therefore also to each other, since the composition of isomorphisms is again an isomorphism.

The theorem says that all $n$-dimensional spaces over the same field are mutually isomorphic. This means that they are, in a certain sense, the same. Although each has its own specifics, special operations, etc., they exhibit a similar structure and we can approach them in a uniform way. Therefore, when finding dimension, verifying linear independence, etc., it suffices to pass via an isomorphism to the space $\mathbb{T}^n$ over $\mathbb{T}$, where computation is much easier. Indeed, an isomorphism preserves linear independence of vectors, preserves the dimension of the image of a subspace, and also preserves dependence of vectors.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.40)</span></p>

Consider the polynomials $2x^3 + x^2 + x + 3$, $x^3 + 2x^2 + 3x + 1$, $x^3 - x^2 - 2x + 2$, $4x^3 - x^2 - 3x + 7$ as vectors of the space $\mathcal{P}^3$. Are they linearly independent? What is the dimension of the space they generate? What is its basis? These questions are easily answered using the isomorphism $a_3 x^3 + a_2 x^2 + a_1 x + a_0 \mapsto (a_3, a_2, a_1, a_0)$. Under this isomorphism, the polynomials map to the vectors

$$(2, 1, 1, 3)^\top, \quad (1, 2, 3, 1)^\top, \quad (1, -1, -2, 2)^\top, \quad (4, -1, -3, 7)^\top.$$

Now in the standard way (Example 5.70) we find that the vectors (and therefore also the polynomials) are linearly dependent, they generate a two-dimensional subspace, and a basis is formed, for example, by the first two.

</div>

For the linear map $f \colon \mathbb{R}^n \to \mathbb{R}^m$ defined by $f(x) = Ax$, we have $\operatorname{Ker}(f) = \operatorname{Ker}(A)$ and $f(\mathbb{R}^n) = \mathcal{S}(A)$. In the general case as well, there is a close relationship between the kernel of a linear map and the kernel of the corresponding matrix, and similarly between the image and the column space of the matrix.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 6.41 — On the Dimension of the Kernel and Image)</span></p>

Let $f \colon U \to V$ be a linear map, $B_U$ a basis of the space $U$, and $B_V$ a basis of the space $V$. Denote $A = {}_{B_V}[f]_{B_U}$. Then:

1. $\dim \operatorname{Ker}(f) = \dim \operatorname{Ker}(A)$,
2. $\dim f(U) = \dim \mathcal{S}(A) = \operatorname{rank}(A)$.

</div>

*Proof.*

1. By Theorem 6.34(4), it suffices to construct an isomorphism between the spaces $\operatorname{Ker}(f)$ and $\operatorname{Ker}(A)$. An isomorphism can be, for example, the map $x \in \operatorname{Ker}(f) \mapsto [x]_{B_U}$. From Proposition 6.38, we know it is linear and injective. It remains to show that $[x]_{B_U} \in \operatorname{Ker}(A)$ and that the map is surjective. Let $x \in \operatorname{Ker}(f)$, then $o = [f(x)]_{B_V} = {}_{B_V}[f]_{B_U} \cdot [x]_{B_U}$, so $[x]_{B_U} \in \operatorname{Ker}(A)$. Conversely, for every $[x]_{B_U} \in \operatorname{Ker}(A)$ we have $f(x) = o$.
2. Denote $\dim U = n$, $\dim V = m$. Again we construct an isomorphism, now between $f(U)$ and $\mathcal{S}(A)$, namely $y \in f(U) \mapsto [y]_{B_V}$. Again, the map is linear and injective. Furthermore, for $y \in f(U)$ there exists $x \in U$ such that $f(x) = y$. Now $[y]_{B_V} = [f(x)]_{B_V} = A \cdot [x]_{B_U}$, so $[y]_{B_V}$ belongs to the column space $\mathcal{S}(A)$. Conversely, for every $b \in \mathcal{S}(A)$ there exists $a \in \mathbb{T}^n$ such that $b = Aa$. So for the vector $x \in U$ such that $[x]_{B_U} = a$, we have $y := f(x) \in f(U)$ and at the same time $[y]_{B_V} = [f(x)]_{B_V} = A \cdot [x]_{B_U} = Aa = b \in \mathcal{S}(A)$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 6.42)</span></p>

The proof of Theorem 6.41 is constructive -- it tells us not only how to compute the dimension of the kernel and image of $f$, but also how to find their bases. If $x_1, \ldots, x_k$ is a basis of $\operatorname{Ker}(A)$, then these vectors form the coordinates (with respect to the basis $B_U$) of a basis of $\operatorname{Ker}(f)$. Similarly, if $y_1, \ldots, y_r$ is a basis of the space $\mathcal{S}(A)$, then these vectors represent the coordinates of a basis of the space $f(U)$ with respect to $B_V$.

</div>

As a corollary of Theorem 6.41, we obtain the following generalization of the equality from Theorem 6.34(4), since for an isomorphism we have $\dim \operatorname{Ker}(f) = 0$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 6.43)</span></p>

Let $f \colon U \to V$ be a linear map. Then $\dim U = \dim \operatorname{Ker}(f) + \dim f(U)$.

</div>

*Proof.* By Theorem 5.72, for a matrix $A$ of type $m \times n$, the equality $n = \dim \operatorname{Ker}(A) + \operatorname{rank}(A)$ holds. In particular, for $A = {}_{B_V}[f]_{B_U}$ we obtain the desired identity, since $n = \dim U$, $\dim \operatorname{Ker}(f) = \dim \operatorname{Ker}(A)$, and $\dim f(U) = \operatorname{rank}(A)$.

Already on page 102 we observed that the kernel of a linear map describes how much the map degenerates. Corollary 6.43 then expresses the degree of degeneracy numerically. The dimension of the kernel gives the difference between the dimension of the space $U$ and the dimension of its image.

With regard to Theorems 6.12 and 6.41, we obtain that a linear map $f \colon U \to V$ is injective if and only if $\dim U = \dim f(U)$, i.e., $\dim U = \operatorname{rank}({}_{B_V}[f]_{B_U})$. A necessary and sufficient condition for $f$ to be injective is therefore that the matrix of the map $f$ with respect to any bases has linearly independent columns.

How do we determine that a linear map $f \colon U \to V$ is surjective? This situation can be expressed by the condition $\dim V = \dim f(U)$, i.e., $\dim V = \operatorname{rank}({}_{B_V}[f]_{B_U})$. Equivalently, the matrix of $f$ with respect to any bases must have linearly independent rows. We thus obtain the following proposition.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 6.44)</span></p>

Let $f \colon U \to V$ be a linear map, $B_U$ a basis of the space $U$, and $B_V$ a basis of the space $V$. Then:

1. $f$ is injective if and only if ${}_{B_V}[f]_{B_U}$ has linearly independent columns,
2. $f$ is surjective if and only if ${}_{B_V}[f]_{B_U}$ has linearly independent rows.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.45)</span></p>

Let us have a linear map $f \colon \mathbb{R}^3 \to \mathcal{P}^2$ given by the matrix

$${}_{B_V}[f]_{B_U} = A = \begin{pmatrix} 1 & 1 & 1 \\ 3 & 2 & 0 \\ 0 & 1 & 3 \end{pmatrix},$$

where

$$B_U = \lbrace (1, 2, 1)^\top, \ (0, 1, 1)^\top, \ (1, 2, 4)^\top \rbrace, \qquad B_V = \lbrace x^2 - 2x + 3, \ x - 1, \ 2x^2 + x \rbrace.$$

Since $\operatorname{rank}(A) = 2$, we immediately get $\dim \operatorname{Ker}(f) = 3 - \operatorname{rank}(A) = 1$ and $\dim f(\mathbb{R}^3) = \operatorname{rank}(A) = 2$. Since the kernel has positive dimension and is therefore nontrivial, by Theorem 6.12 this means the map $f$ is not injective. Since the image has dimension 2 but the space $\mathcal{P}^2$ has dimension 3, the map $f$ is not surjective.

A basis of $\operatorname{Ker}(A)$ is $(2, -3, 1)^\top$, which represents the coordinates of the desired vector in the basis $B_U$. Thus a basis of $\operatorname{Ker}(f)$ is formed by the vector $2(1, 2, 1)^\top - 3(0, 1, 1)^\top + 1(1, 2, 4)^\top = (3, 3, 3)^\top$.

A basis of $\mathcal{S}(A)$ is $(1, 3, 0)^\top$, $(1, 2, 1)^\top$, which again represents the coordinates of the desired vectors. Thus the basis of the image $f(\mathbb{R}^3)$ consists of two vectors

$$1(x^2 - 2x + 3) + 3(x - 1) + 0(2x^2 + x) = x^2 + x, \qquad 1(x^2 - 2x + 3) + 2(x - 1) + 1(2x^2 + x) = 3x^2 + x + 1.$$

</div>

### 6.4 The Space of Linear Maps

It is not difficult to see that the set of linear maps from a space $U$ over $\mathbb{T}$ of dimension $n$ to a space $V$ over $\mathbb{T}$ of dimension $m$ forms a vector space: the sum of linear maps $f, g \colon U \to V$ is again a linear map $(f + g) \colon U \to V$, and the scalar multiple $\alpha f$ of a linear map $f \colon U \to V$ is also a linear map. The zero vector is the map $u \mapsto o_V$ $\forall u \in U$.

Moreover, since every linear map is uniquely determined by a matrix with respect to given bases, this space of linear maps is isomorphic to the space of matrices $\mathbb{T}^{m \times n}$ and therefore has dimension $mn$. The corresponding isomorphism can be the map $f \mapsto {}_{B_V}[f]_{B_U}$, where $B_U$ is any fixed basis of the space $U$ and $B_V$ is any fixed basis of the space $V$. The linearity of this map follows easily (due to linearity of coordinates) from the properties

$${}_{B_V}[f + g]_{B_U} = {}_{B_V}[f]_{B_U} + {}_{B_V}[g]_{B_U}, \qquad {}_{B_V}[\alpha f]_{B_U} = \alpha \, {}_{B_V}[f]_{B_U}.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 6.46 — Linear Form and Dual Space)</span></p>

Let $V$ be a vector space over $\mathbb{T}$. A *linear form* (also called a linear functional) is any linear map from $V$ to $\mathbb{T}$. The *dual space*, denoted $V^*$, is the vector space of all linear forms.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.47)</span></p>

An example of a linear form on the space $\mathbb{R}^n$ over $\mathbb{R}$ is the map $f(x_1, \ldots, x_n) = \frac{1}{n}\sum_{i=1}^{n} x_i$ or the map $g(x_1, \ldots, x_n) = x_1$.

</div>

Nothing prevents us from considering the dual space of the dual space. This is the space $V^{**}$ of all linear maps $F \colon V^* \to \mathbb{T}$. In other words, $F$ maps every linear form on $V$ to a scalar from the field $\mathbb{T}$. For example, let $v^* \in V$ be a fixed vector and consider the map that sends a linear form $f$ to its function value $f(v^*)$. We have just defined a map $F_{v^*} \in V^{**}$ given by $F_{v^*}(f) = f(v^*)$. For every vector $v^* \in V$ we have thus found a vector $F_{v^*} \in V^{**}$. The map $v^* \mapsto F_{v^*}$ is called the *canonical embedding* of the space $V$ into the space $V^{**}$. It can be shown that this is an injective linear map.

If $\dim V = n$, then also $\dim V^* = n$. If $v_1, \ldots, v_n$ is a basis of $V$, then the dual space has, for example, the basis $f_1, \ldots, f_n$, where $f_i$ is determined by the images of the basis: $f_i(v_i) = 1$ and $f_i(v_j) = 0$ for $i \neq j$. This basis is called the *dual basis* to the basis $v_1, \ldots, v_n$.

For a finitely generated space, $V$ is therefore isomorphic to the dual space $V^*$, to the dual of the dual space $V^{**}$, etc. Nevertheless, there always exists a canonical embedding of $V$ into $V^{**}$. If moreover $V$ and $V^{**}$ are isomorphic, then $V$ has certain nice properties.

### 6.5 Applications

Linear maps have wide applications in computer graphics for data visualization, animation, 3D scene modeling, etc. Since linear maps allow performing basic transformations (scaling, rotation, projection, ...) using simple matrix operations, we obtain an elegant way to display two- and three-dimensional objects.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 6.48 — Visualization of Three-Dimensional Objects)</span></p>

Consider an object for visualization; in practice, this could be, for example, a 3D image of a human organ obtained via magnetic resonance or CT, which we want to display from a certain viewpoint and at a certain scale. The object is placed in a given coordinate system. In our case, we consider a cylinder-shaped object with the center of its base at the origin.

First, it is necessary to rescale the object to the desired size. We do this by the transformation $x \mapsto Ax$ with a diagonal matrix $A = \operatorname{diag}(\alpha_x, \alpha_y, \alpha_z)$. In the $x$-axis we scale with coefficient $\alpha_x$ and similarly for the other axes. For uniform scaling, $A = \alpha I_3$.

Next, the object needs to be placed in the correct position and rotated to the correct orientation. Every rotation in the space $\mathbb{R}^3$ can be composed of three rotations about the coordinate axes. According to Example 6.3, the rotation matrix about the $y$-axis by an angle $\varphi$ has the form

$$\begin{pmatrix} \cos(\varphi) & 0 & -\sin(\varphi) \\ 0 & 1 & 0 \\ \sin(\varphi) & 0 & \cos(\varphi) \end{pmatrix}.$$

Finally, we project the object onto the corresponding projection plane. For example, the projection onto the plane of the $x, z$ axes is represented by the matrix $\operatorname{diag}(1, 0, 1)$. This corresponds to an observer looking from the direction of the $y$-axis.

The resulting image for rendering was produced using several linear transformations, and the entire procedure can therefore be represented as a matrix product of the corresponding matrices of linear transformations.

</div>

### Summary of Chapter 6

A linear map between vector spaces preserves the structure of linear combinations: it maps a linear combination of vectors to the same linear combination of their images. To specify a linear map, it suffices to state where the vectors of some basis are sent, and this fully determines the images of all other vectors and thus of the entire space.

For arithmetic spaces (of the type $\mathbb{T}^n$), a linear map can be expressed in matrix form as $x \mapsto Ax$. Every matrix thus corresponds to some linear map, and conversely, every linear map has a matrix expression. This duality is absolutely key, because many problems can be viewed algebraically (operations with the matrix $A$) or geometrically (using the linear map $x \mapsto Ax$). Many properties of the linear map $x \mapsto Ax$ are again related to properties of the matrix $A$:

- composition of maps corresponds to matrix multiplication,
- the map is injective if and only if the kernel of the matrix contains only $o$,
- the rank of the matrix gives the dimension of the image,
- etc.

For general spaces, the situation is slightly more complicated, but things work similarly. One just has to work with coordinates instead of the vectors themselves. Coordinates are vectors from the space $\mathbb{T}^n$, so the observations mentioned above can be adapted to the general case. When working with coordinates, the change-of-basis matrix is used extensively, converting coordinates in one basis to coordinates in another basis; thus a change of coordinate system can again be efficiently represented by a matrix.

A linear map that is a bijection is called an isomorphism. Isomorphisms correspond to nonsingular matrices; therefore, inverses exist for both (isomorphisms and matrices). Spaces between which an isomorphism exists are called isomorphic. Isomorphic spaces are, from the perspective of linear algebra, essentially the same -- they have different elements but behave identically. They also have the same dimension, and this observation holds in reverse as well: All $n$-dimensional spaces over the same field are mutually isomorphic. This allows us to easily work with any space as if it were the space $\mathbb{T}^n$.

---

## Chapter 7 — Affine Subspaces

This is a brief introduction to affine subspaces. Vector spaces and subspaces are limited by the requirement that they must contain the zero vector. Affine subspaces generalize the notion of a subspace and avoid this restriction. An affine subspace of $\mathbb{R}^3$ can thus be any line or plane, not just one passing through the origin.

### 7.1 Basic Concepts

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 7.1 — Affine Subspace)</span></p>

Let $V$ be a vector space over $\mathbb{T}$. Then an *affine subspace* is any set $M \subseteq V$ of the form

$$M = U + a = \lbrace u + a;\ u \in U \rbrace,$$

where $a \in V$ and $U$ is a vector subspace of $V$.

</div>

An affine subspace (the terms affine space or affine set are also used with the same meaning) is thus any subspace $U$ "shifted" by some vector $a$.

Since $o \in U$, we have $a \in M$. This representative $a$ is not unique; we can choose any vector from $M$. Conversely, the subspace $U$ is uniquely determined for each affine subspace (Problem 7.1).

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 7.2)</span></p>

Let $V$ be a vector space. Every vector subspace $U$ of $V$ is also an affine subspace, since we can choose $a = o$, so that $U = U + o$ has the form of an affine subspace.

Furthermore, for every vector $a \in V$, the set $\lbrace a \rbrace$ is a one-element affine subspace of $V$. We obtain it by choosing $U := \lbrace o \rbrace$, because then $U + a = \lbrace a \rbrace$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 7.3 — Tiling by Affine Subspaces)</span></p>

Let $V$ be a vector space and $U$ its subspace. Then affine subspaces of the form $U + a$, $U + a'$ are either identical or disjoint. Moreover, every vector $v \in V$ lies in some affine subspace of this form, for example in the affine subspace $U + v$. Therefore, the space $V$ can be decomposed into a disjoint union of affine subspaces of the form $U + a$ for suitable choices of vectors $a$.

</div>

#### Affine Combinations

While vector subspaces are sets of vectors that are closed under linear combinations, affine subspaces are sets of vectors that are closed under so-called affine combinations.

An *affine combination* of two vectors $x, y \in V$ (a space over a field $T$) is an expression (vector) $\alpha x + (1 - \alpha)y$, where $\alpha \in \mathbb{T}$. The affine combination can be rewritten as $\alpha x + (1 - \alpha)y = y + \alpha(x - y)$, which is a parametric description of a line with point $y$ and direction $x - y$. In other words, an affine subspace containing any two points must also contain the line passing through them.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 7.4 — Characterization of an Affine Subspace)</span></p>

Let $V$ be a vector space over a field $\mathbb{T}$ of characteristic different from 2, and let $\emptyset \neq M \subseteq V$. Then $M$ is an affine subspace if and only if for every $x, y \in M$ and $\alpha \in \mathbb{T}$ we have $\alpha x + (1 - \alpha)y \in M$.

</div>

*Proof.* Implication "$\Rightarrow$": Let $M$ have the form $M = U + a$. Let $x, y \in M$, so they have the form $x = u + a$, $y = v + a$, where $u, v \in U$. Then $\alpha x + (1 - \alpha)y = \alpha(u + a) + (1 - \alpha)(v + a) = \alpha u + (1 - \alpha)v + a \in U + a = M$.

Implication "$\Leftarrow$": We show that it suffices to choose $a \in M$ arbitrarily fixed and $U := M - a = \lbrace x - a;\ x \in M \rbrace$. We need to verify that $M = U + a$ and that $U$ is a vector subspace. The equality $M = U + a$ follows from the definition of $U$, so we focus on the second part and show $o \in U$ and closure of $U$ under scalar multiples and sums. Clearly $o \in U$.

Closure under scalar multiples: Let $\alpha \in \mathbb{T}$ and $u \in U$, so $u = x - a$ for some $x \in M$. Then $\alpha u = \alpha(x - a) = (\alpha x + (1 - \alpha)a) - a \in M - a = U$, since $\alpha x + (1 - \alpha)a$ is an affine combination of vectors $x, a \in M$.

Closure under sums: Let $u, u' \in U$, so they have the form $u = x - a$, $u' = x' - a$ for some $x, x' \in M$. Their sum gives $u + u' = (x - a) + (x' - a) = (x + x' - a) - a$. It suffices to show that $x + x' - a \in M$. Since $x, x' \in M$, their affine combination $\frac{1}{2}x + \frac{1}{2}x' \in M$. Since $(\frac{1}{2}x + \frac{1}{2}x'), a \in M$, their affine combination $2(\frac{1}{2}x + \frac{1}{2}x') + (1 - 2)a = x + x' - a \in M$.

The implication "$\Rightarrow$" always holds, but the converse implication may not hold over a field of characteristic 2. As a counterexample, take the space $\mathbb{Z}_2^n$ over $\mathbb{Z}_2$, in which every set of vectors is closed under affine combinations of two vectors.

The theorem can be generalized to fields of characteristic 2 as well, but we must extend the notion of affine combination to a larger number of vectors. An *affine combination* of vectors $x_1, \ldots, x_n \in V$ is an expression (vector)

$$\sum_{i=1}^{n} \alpha_i x_i, \quad \text{where } \alpha_i \in \mathbb{T}, \quad \sum_{i=1}^{n} \alpha_i = 1.$$

This is a linear combination in which the sum of coefficients equals 1. For two vectors, we recover the original definition. The geometric interpretation for three vectors (points) says that their affine combinations describe the plane determined by these points.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 7.5)</span></p>

Let $V$ be a vector space over $\mathbb{T}$ and let $\emptyset \neq M \subseteq V$. Then $M$ is an affine subspace if and only if $M$ is closed under affine combinations.

</div>

*Proof.* Analogous to the proof of Theorem 7.4. The proof of closure of the set $U$ under sums follows directly from the fact that $x + x' - a \in M$, because this is an affine combination of three vectors $x, x', a$ (their coefficients sum to $1 + 1 - 1 = 1$). Therefore, there is no need to divide by two anywhere, and one can consider an arbitrary field.

From the proof, we see that it suffices for the set $M$ to be closed under affine combinations of three vectors. Then it is already closed under affine combinations of any finite number of vectors.

#### Affine Subspaces and Systems of Linear Equations

There is a very close relationship between affine subspaces and the solution sets of systems of linear equations.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 7.6 — Systems of Linear Equations and Affine Subspaces)</span></p>

The solution set of the system of equations $Ax = b$ is either empty or affine. If it is nonempty, we can express this solution set in the form $\operatorname{Ker}(A) + x_0$, where $x_0$ is any particular solution of the system.

</div>

*Proof.* If $x_1$ is a solution, then we can write $x_1 = x_1 - x_0 + x_0$. It suffices to show that $x_1 - x_0 \in \operatorname{Ker}(A)$. By substitution, $A(x_1 - x_0) = Ax_1 - Ax_0 = b - b = o$. Thus $x_1 \in \operatorname{Ker}(A) + x_0$. Conversely, if $x_2 \in \operatorname{Ker}(A)$, then $x_2 + x_0$ is a solution of the system, since $A(x_2 + x_0) = Ax_2 + Ax_0 = o + b = b$.

We will show that the converse implication also holds, i.e., every affine subspace of $\mathbb{T}^n$ over $\mathbb{T}$ can be described by a system of equations.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 7.7 — Affine Subspaces and Systems of Linear Equations)</span></p>

Let $U + a$ be an affine subspace of $\mathbb{T}^n$ over $\mathbb{T}$. Then there exists a matrix $A \in \mathbb{T}^{m \times n}$ and a vector $b \in \mathbb{T}^m$ such that the solution set of the system of linear equations $Ax = b$ equals $U + a$.

</div>

*Proof.* Let $v_1, \ldots, v_k \in \mathbb{T}^n$ be a basis of the subspace $U$. We construct a matrix $C \in \mathbb{T}^{k \times n}$ whose rows are the vectors $v_1, \ldots, v_k$. The dimension of its kernel is $\dim \operatorname{Ker}(C) = n - \operatorname{rank}(C) = n - k$. Let $w_1, \ldots, w_{n-k}$ be a basis of $\operatorname{Ker}(C)$. Thus $Cw_j = o$, which in particular for the rows of $C$ gives $v_i^\top w_j = 0$ for $i = 1, \ldots, k$, $j = 1, \ldots, n - k$. Now we construct a matrix $A \in \mathbb{T}^{(n-k) \times n}$ whose rows are formed by the vectors $w_1, \ldots, w_{n-k}$. The dimension of its kernel is $\dim \operatorname{Ker}(A) = n - \operatorname{rank}(A) = n - (n - k) = k$. Since the vectors $v_1, \ldots, v_k$ are linearly independent and there are the right number of them, they form a basis of $\operatorname{Ker}(A)$. Therefore $\operatorname{Ker}(A) = U$. It remains to determine the vector $b$ so that the vector $a$ is a solution of the system $Ax = b$. It suffices to choose $b := Aa$. By Theorem 7.6, the solution set of the system $Ax = b$ equals $\operatorname{Ker}(A) + a = U + a$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 7.8 — System of Linear Equations Under Change of the Right-Hand Side)</span></p>

Theorem 7.6 gives the following geometric perspective on systems of equations under perturbation of the right-hand side. Let $Ax = b$ be solvable, so it describes an affine subspace $\operatorname{Ker}(A) + x_0$, where $x_0'$ is one chosen solution. If we change the right-hand side from $b$ to $b'$, then either the system becomes unsolvable, or the affine subspace shifts to $\operatorname{Ker}(A) + x_0'$, where $x_0'$ is one chosen solution. If the rows of $A$ are linearly independent, then the system is solvable for any right-hand side, and therefore the solution set merely shifts in some direction when the right-hand side changes. If the rows of $A$ are linearly dependent, then for some $b$ the system is solvable and for others it is not. For those values of $b$ for which the system is solvable, the solution set is again the same up to a shift.

For concreteness, consider the system of linear equations with a general right-hand side

$$(A \mid b) = \begin{pmatrix} 1 & 1 & 3 & b_1 \\ 2 & 1 & 1 & b_2 \end{pmatrix}.$$

The rows of $A$ are linearly independent and for every $b = (b_1, b_2)^\top$ the solution set has the form $\operatorname{span}\lbrace (2, -5, 1)^\top \rbrace + (-b_1 + b_2, 2b_1 - b_2, 0)^\top$. It is therefore a line with the same direction every time.

Now consider the system of linear equations with linearly dependent rows of $A$

$$(A \mid b) = \begin{pmatrix} 1 & 2 & 3 & b_1 \\ 2 & 4 & 6 & b_2 \end{pmatrix}.$$

If $b_2 \neq 2b_1$, no solution exists. If $b_2 = 2b_1$, the solution set is a plane described by the equation $x_1 + 2x_2 + 3x_3 = b_1$ and its normal does not depend on the right-hand side.

</div>

#### Dimension of an Affine Subspace

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 7.9 — Dimension of an Affine Subspace)</span></p>

The *dimension* of an affine subspace $M = U + a$ is defined as $\dim(M) := \dim(U)$.

</div>

Since every vector subspace of $V$ is also an affine subspace, the definition generalizes the notion of dimension introduced for vector spaces. The definition naturally assigns dimension zero to a point, dimension one to a line in $\mathbb{R}^n$, and dimension two to a plane.

It also allows us to define a *line* $p$ in any vector space $V$ over $\mathbb{T}$ as an affine subspace of dimension one. In other words, $p = \operatorname{span}\lbrace v \rbrace + a$, where $a, v \in V$ and $v \neq o$. From this we also obtain the familiar parametric description of a line $p = \lbrace \alpha v + a;\ \alpha \in \mathbb{T} \rbrace$.

A *hyperplane* in a space of dimension $n$ is any affine subspace of dimension $n - 1$. For example, in $\mathbb{R}^2$ these are lines, in $\mathbb{R}^3$ they are planes, etc.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 7.10)</span></p>

The set $\lbrace e^x + \alpha \sin x;\ \alpha \in \mathbb{R} \rbrace$ is a line in the function space $\mathcal{F}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 7.11)</span></p>

For any $a \in \mathbb{R}^n \setminus \lbrace o \rbrace$ and $b \in \mathbb{R}$, the set described by the equation $a^\top x = b$ is a hyperplane in $\mathbb{R}^n$. Conversely, every hyperplane in $\mathbb{R}^n$ can be described by the equation $a^\top x = b$ for some $a \in \mathbb{R}^n \setminus \lbrace o \rbrace$ and $b \in \mathbb{R}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 7.12)</span></p>

Let $A \in \mathbb{T}^{m \times n}$, $b \in \mathbb{T}^m$. If the solution set of the system of equations $Ax = b$ is nonempty, then it forms an affine subspace of dimension $n - \operatorname{rank}(A)$.

</div>

*Proof.* By Proposition 7.6, the solution set can be expressed in the form $\operatorname{Ker}(A) + x_0$, where $x_0$ is any particular solution of the system. Its dimension therefore equals the dimension of the kernel, which by Theorem 5.72 equals $\dim \operatorname{Ker}(A) = n - \operatorname{rank}(A)$.

#### Affine Independence

Linear independence of vectors $x_1, \ldots, x_n$ meant that the subspace generated by these vectors cannot be generated by any proper subset of them. None of them is, in a certain sense, redundant. We would like to have an analogous property for affine subspaces as well, that is, to characterize the smallest set of vectors that uniquely determines a given affine subspace. This leads to the concept of affine independence.

We defined an affine subspace as $U + a$, i.e., a vector subspace $U$ shifted by a vector $a$. It is therefore natural to define affine independence of vectors by shifting them back and requiring linear independence of the resulting vectors.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 7.13 — Affine Independence)</span></p>

Vectors $x_0, x_1, \ldots, x_n$ of a vector space are *affinely independent* if $x_1 - x_0, \ldots, x_n - x_0$ are linearly independent. Otherwise, the vectors are called *affinely dependent*.

</div>

Vectors $x_0, x_1, \ldots, x_n \in V$ uniquely determine the smallest (in the sense of inclusion) affine subspace containing them. Denote it by $M = U + x_0$. The vectors $x_1 - x_0, \ldots, x_n - x_0$ are generators of the subspace $U$. These generators form a basis of $U$ if and only if the vectors $x_0, x_1, \ldots, x_n$ are affinely independent. If we were to remove any of the vectors $x_0, x_1, \ldots, x_n$, we would not generate the entire affine subspace $M$, but only a part of it.

It is not difficult to see that affine independence does not depend on the ordering of the vectors, and therefore not on the choice of $x_0$ either (prove this!).

Affine independence also provides a simple formalization of the notion of "points in general position": A set of points $x_1, \ldots, x_m \in \mathbb{R}^n$ is in general position if every subset of size at most $n + 1$ is affinely independent. For example, in the plane $\mathbb{R}^2$, points are in general position if no three of them lie on a common line.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 7.14)</span></p>

The vectors $(1, 1)^\top, (2, 2)^\top, (1, 2)^\top \in \mathbb{R}^2$ are linearly dependent but affinely independent.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 7.15)</span></p>

Two distinct points in $\mathbb{R}^n$ are affinely independent and the affine subspace they generate is a line. However, three points on a line are already affinely dependent, because a line is uniquely determined by just two points.

</div>

#### Coordinates in an Affine Subspace

Let $M = U + a$ be an affine subspace and $B = \lbrace v_1, \ldots, v_n \rbrace$ a basis of $U$. Then every $x \in M$ can be uniquely written in the form $x = a + \sum_{i=1}^{n} \alpha_i v_i$. Thus the system of vectors $S = \lbrace a, v_1, \ldots, v_n \rbrace$ can be regarded as a *coordinate system* and the vector $[x]_S := [x - a]_B = (\alpha_1, \ldots, \alpha_n)^\top$ as the corresponding *coordinates*.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 7.16)</span></p>

Let $v = (2, 1)^\top$, $a = (1, 2)^\top$ and consider the line $\operatorname{span}\lbrace v \rbrace + a$. Consider also the coordinate system $S = \lbrace a, v \rbrace$. Then the point $x = (5, 4)^\top$ can be expressed as $x = a + 2v$, and therefore its coordinates are $[x]_S = (2)$.

Now consider a different vector $a' = (-1, 1)^\top$. Then the expression for the vector $x$ changes to $x = a' + 3v$, and therefore its coordinates in the system $S' = \lbrace a', v \rbrace$ are $[x]_{S'} = (3)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 7.17)</span></p>

Let $M = U + a$ be an affine subspace and $B = \lbrace v_1, \ldots, v_n \rbrace$, $B' = \lbrace v_1', \ldots, v_n' \rbrace$ two bases of $U$.

1. For two given coordinate systems $S = \lbrace a, v_1, \ldots, v_n \rbrace$ and $S' = \lbrace a, v_1', \ldots, v_n' \rbrace$ we have

$$[x]_{S'} = {}_{B'}[id]_B \cdot [x]_S, \quad \forall x \in U + a.$$

2. For two given coordinate systems $S = \lbrace a, v_1, \ldots, v_n \rbrace$ and $S' = \lbrace a', v_1', \ldots, v_n' \rbrace$ we have

$$[x]_{S'} = [a - a']_{B'} + {}_{B'}[id]_B \cdot [x]_S, \quad \forall x \in U + a.$$

</div>

To transition between coordinate systems we can use our familiar change-of-basis matrix exactly as we are accustomed to. In the case where we also change the vector $a$, a constant additive term appears in addition.

#### Relationship Between Affine Subspaces

Affine subspaces $U + a$, $W + b$ are *parallel* if $U \subseteq W$ or $W \subseteq U$; *intersecting* if they are not parallel and have nonempty intersection; and *skew* if they are not parallel and have empty intersection.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 7.18)</span></p>

Let $v$ be any vector of a vector space $V$. Then the affine subspace $\lbrace v \rbrace$ is parallel to every affine subspace of $V$. The entire space $V$ has the same property.

</div>

#### Affine Maps

Let $g \colon U \to V$ be a linear map and let $b \in V$ be a fixed vector. Then an *affine map* has the form $f(u) = g(u) + b$. A simple example of an affine map is a translation, i.e., the map $g \colon V \to V$ given by $f(x) = x + b$, where $b \in V$ is fixed.

An affine map need not send the zero vector in $U$ to the zero vector in $V$, because the images are shifted by the additive term $b$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 7.19)</span></p>

Let $U + a$ be an affine subspace of dimension $k$ in the space $V$, and let $S$ be a coordinate system in $U + a$. Then the map $f(v) = [v]_S$ is an affine map that maps $U + a$ isomorphically onto $\mathbb{T}^k$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 7.20 — Properties of Affine Maps)</span></p>

1. The image of an affine subspace under an affine map is an affine subspace.
2. The composition of two affine maps is again an affine map.
3. Let $f \colon U \to V$ be a linear map and let $v \in V$. Then the preimage of the vector $v$

$$f^{-1}(v) := \lbrace u \in U;\ f(u) = v \rbrace$$

is either the empty set or an affine subspace of $U$.

</div>

*Proof.*

1. Let $f \colon U \to V$ be an affine map of the form $f(u) = g(u) + b$, where $g \colon U \to V$ is linear and $b \in V$. Then the image of the affine subspace $W + a \subseteq U$ is $f(W + a) = g(W + a) + b = g(W) + g(a) + b$. This is therefore an affine subspace of $V$, obtained by shifting the subspace $g(W)$ in the direction of the vector $g(a) + b$.
2. Let $f_1 \colon U \to V$, $f_2 \colon V \to W$ be affine maps of the form $f_1(u) = g_1(u) + b_1$, $f_2(v) = g_2(v) + b_2$, where $g_1 \colon U \to V$, $g_2 \colon V \to W$ are linear and $b_1 \in V$, $b_2 \in W$. Then the composed map has the form $(f_2 \circ f_1)(u) = f_2(f_1(u)) = g_2(g_1(u) + b_1) + b_2 = g_2(g_1(u)) + g_2(b_1) + b_2 = (g_2 \circ g_1)(u) + g_2(b_1) + b_2$. This is again an affine map, obtained from the linear map $g_2 \circ g_1$ by shifting by the additive term $g_2(b_1) + b_2$.
3. Let $U, V$ be spaces over a field $\mathbb{T}$ and let $u_1, \ldots, u_n \in f^{-1}(v)$. Consider their affine combination $\sum_{i=1}^{n} \alpha_i u_i$, where $\alpha_1, \ldots, \alpha_n \in \mathbb{T}$ and $\sum_{i=1}^{n} \alpha_i = 1$. Then $f(\sum_{i=1}^{n} \alpha_i u_i) = \sum_{i=1}^{n} \alpha_i f(u_i) = \sum_{i=1}^{n} \alpha_i v = v$. Therefore $\sum_{i=1}^{n} \alpha_i u_i \in f^{-1}(v)$, which shows that the set $f^{-1}(v)$ is closed under affine combinations.

Point (3) of Proposition 7.20 has an analogy with solving systems of linear equations. Consider the linear map $f \colon \mathbb{R}^n \to \mathbb{R}^m$ expressed in matrix form as $f(x) = Ax$ and let $b \in \mathbb{R}^m$ be given. Then finding all solutions of the system $Ax = b$ actually means finding the preimage of the vector $b$, i.e., the set

$$f^{-1}(b) = \lbrace x \in \mathbb{R}^n;\ f(x) = b \rbrace = \lbrace x \in \mathbb{R}^n;\ Ax = b \rbrace.$$

From Theorem 7.6 we know that the solution set of the system $Ax = b$ is either empty or an affine subspace. Yet another perspective on point (3) of Proposition 7.20 is through the kernel of a linear map. Similarly to Theorem 7.6, we can express the preimage of the vector $v$ as the affine subspace $\operatorname{Ker}(f) + u$, where $u \in U$ is one fixed preimage of the vector $v$.

### 7.2 Applications

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 7.21 — Equations Yes, but Inequalities?)</span></p>

In the previous chapters we studied systems of linear equations $Ax = b$, so it is natural to also consider systems of linear inequalities $Ax \le b$. Inequality between vectors means inequality in each component, i.e., $(Ax)_i \le b_i$ for all $i$. Furthermore, we restrict ourselves to the field $\mathbb{R}$, where an ordering is defined.

While a single equation defines a hyperplane in the space and a system of equations defines some affine subspace, a single inequality defines a half-space and a system of inequalities defines an intersection of half-spaces, which is a *convex polyhedron*.

A quadrilateral consists of four vertices, four edges, and an interior. The vertices lie at the intersection of hyperplanes corresponding to a system of equations, and by Theorem 7.6 this is an affine subspace of dimension zero. An edge connecting two adjacent vertices lies on a one-dimensional affine subspace. We characterize other vertices and edges analogously.

We continue in a similar fashion in higher dimensions. For example, three-dimensional polyhedra such as a cube, octahedron, etc., have the following structure. Vertices are zero-dimensional affine subspaces determined by the intersection of three hyperplanes, i.e., described by three equations. Edges lie in a one-dimensional affine subspace described by a system of two equations. Finally, faces lie in a hyperplane, i.e., in a two-dimensional affine subspace determined by a single equation.

Convex polyhedra are studied more extensively in the field of *linear programming*. This field examines not only convex polyhedra, but also solves optimization problems over them of the type

$$\min\ c^\top x \quad \text{subject to } Ax \le b,$$

where $c \in \mathbb{R}^n$, $A \in \mathbb{R}^{m \times n}$ and $b \in \mathbb{R}^m$ are given and $x \in \mathbb{R}^n$ is the vector of variables. Linear programming thus seeks the minimum of a linear function on a convex polyhedron. This problem is a fundamental task of optimization and appears in nearly all problems related to optimization: for example, in scheduling and planning, transportation problems (finding the shortest path), or in finding optimal strategies in game theory.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 7.22 — Affine Maps and Fractals)</span></p>

A fractal is a self-similar geometric shape that appears complex at first glance. We will show by example that even a rather complex fractal can have a simple description in terms of affine maps.

Using four affine maps we can draw a complex fractal in the plane. We start at the origin and with given probabilities we consider a transition according to the corresponding affine map.

$$T_1(x, y) = \begin{pmatrix} 0.86 & 0.03 \\ -0.03 & 0.86 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} + \begin{pmatrix} 0 \\ 1.5 \end{pmatrix} \quad \text{with probability 0.83}$$

$$T_2(x, y) = \begin{pmatrix} 0.2 & -0.25 \\ 0.21 & 0.23 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} + \begin{pmatrix} 0 \\ 1.5 \end{pmatrix} \quad \text{with probability 0.08}$$

$$T_3(x, y) = \begin{pmatrix} -0.15 & 0.27 \\ 0.25 & 0.26 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} + \begin{pmatrix} 0 \\ 0.45 \end{pmatrix} \quad \text{with probability 0.08}$$

$$T_4(x, y) = \begin{pmatrix} 0 & 0 \\ 0 & 0.17 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \end{pmatrix} \quad \text{with probability 0.01}$$

The visited points gradually draw the so-called Barnsley fern fractal in the shape of a fern leaf.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 7.23 — Stewart–Gough Platform in Robotics)</span></p>

The Stewart--Gough platform is a so-called parallel manipulator in the field of kinematic robotics. A fixed base is connected to a mobile platform by several (usually six) movable legs. These platforms are used as manipulators, in simulations (e.g., flight simulators), or in joint biomechanics for testing implants outside the human body.

Both the base and the mobile platform have their own coordinate systems, between which we can transition using an affine map. For example, if $x = (x_1, x_2, x_3)^\top$ are the coordinates of a point in the platform's system, then the coordinates with respect to the base are obtained as $x' = Px + c$, where $P$ is a matrix representing the tilt and $c$ is a fixed vector representing the displacement. Of course, $P$ and $c$ are not fixed but depend on the degree of extension of the movable legs. Moreover, it can be shown that the matrix $P$ depends on only three parameters, because the platform's system is merely rotated relative to the base and is not deformed in any way (stretched, skewed, etc.).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 7.24 — Linear Classifier and Neural Networks)</span></p>

Suppose we have data represented by vectors $v_1, \ldots, v_m \in \mathbb{R}^n$ and for each one we know whether it belongs to group A or group B. Let $v_i$ belong to group A for $i \in \mathcal{A}$ and to group B for $i \in \mathcal{B}$. We want to construct a classifier that can automatically decide for a new value $v \in \mathbb{R}^n$ which group it belongs to. A simple classifier can be constructed based on a linear separator. Its essence lies in constructing a hyperplane $a^\top x = b$ such that the vectors of group A lie in one half-space and the vectors of group B in the other.

Stated mathematically, we seek $a \in \mathbb{R}^n$, $b \in \mathbb{R}$ such that

$$a^\top v_i < b \quad \forall i \in \mathcal{A}, \qquad a^\top v_i > b \quad \forall i \in \mathcal{B}.$$

If we find such a hyperplane, the classification of a new data point $v \in \mathbb{R}^n$ works simply. If $a^\top v < b$, then we consider the value $v$ a member of group A, and otherwise a member of group B. If we cannot find a linear separator, the situation is somewhat more complicated. One way to deal with this is to embed the data into a higher-dimensional space, where there is a greater chance of success.

Linear classifiers are used, among other things, in neural networks. This is one of the ways in which perceptron learning algorithms search for the weight coefficients of neuron connections.

</div>

### Summary of Chapter 7

A subspace of a vector space must pass through the origin. However, if we shift a subspace in the direction of some vector, we obtain a new object -- an affine subspace. While vector subspaces are closed under linear combinations, affine subspaces are (under general assumptions) closed under affine combinations. The relationship with systems of linear equations is important -- the solution set of the system $Ax = b$ is an affine subspace, and conversely every affine subspace can be expressed as the solution set of a certain system.

Many concepts and properties from vector spaces naturally carry over to affine subspaces. Thus we easily introduce notions such as (affine) independence, basis, coordinates, or dimension. An affine map is then a map that has the form of a linear map with a constant additive term; for spaces of the type $\mathbb{T}^n$ it has the form $x \mapsto Ax + b$. This map again has similar properties to a linear map, only translated into the world of affine subspaces.

---

## Chapter 8 — Inner Product

By Definition 5.2, we must be able to add vectors and multiply them by scalars, but multiplication of vectors with each other has not been required so far. Nevertheless, introducing the inner product of vectors also allows us to naturally introduce the notions of orthogonality, magnitude, and distance of vectors (and thus limits), etc.

### 8.1 Inner product and norm

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.1 — Motivational)</span></p>

The standard inner product (p. 41) of vectors $x, y \in \mathbb{R}^n$ is defined as

$$x^\top y = \sum_{i=1}^{n} x_i y_i$$

and using it we can easily express the magnitude of a vector or the angle between two vectors. The Euclidean norm (magnitude) of a vector $x \in \mathbb{R}^n$ is defined as $\|x\| = \sqrt{x^\top x} = \sqrt{\sum_{i=1}^{n} x_i^2}$. Clearly $x^\top x \ge 0$. The only vector with zero norm is the zero vector.

Geometrically, the inner product expresses the relation

$$x^\top y = \|x\| \cdot \|y\| \cdot \cos(\varphi),$$

where $\varphi$ is the angle between vectors $x, y$. Thus, knowing the vectors $x, y$, we can easily compute the angle between them. In particular, $x, y$ are orthogonal if and only if $x^\top y = 0$.

From the definition it is immediately clear that the inner product is symmetric, i.e., $x^\top y = y^\top x$. The inner product is also a linear function in both the first and the second argument, but not in both simultaneously. This means that for every $x, x', y \in \mathbb{R}^n$ and $\alpha \in \mathbb{R}$,

$$(x + x')^\top y = x^\top y + x'^\top y, \qquad (\alpha y)^\top y = \alpha (x^\top y)$$

(and symmetrically for the second argument), but in general, for example, $(x + x')^\top(y + y') = x^\top y + x'^\top y'$ does not hold.

</div>

We introduce the inner product (just like groups, vector spaces, etc.) axiomatically, i.e., by listing the properties it should satisfy.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 8.2 — Inner product over $\mathbb{R}$)</span></p>

Let $V$ be a vector space over $\mathbb{R}$. Then an *inner product* is a mapping $\langle \cdot, \cdot \rangle \colon V^2 \to \mathbb{R}$ satisfying for all $x, y, z \in V$ and $\alpha \in \mathbb{R}$:

1. $\langle x, x \rangle \ge 0$ and equality holds only for $x = 0$,
2. $\langle x + y, z \rangle = \langle x, z \rangle + \langle y, z \rangle$,
3. $\langle \alpha x, y \rangle = \alpha \langle x, y \rangle$,
4. $\langle x, y \rangle = \langle y, x \rangle$.

</div>

Property (1) requires an ordering, which is why we defined the inner product over the field of real numbers. Properties (2) and (3) say that the inner product is a linear function in the first argument. Thanks to property (4), the inner product is symmetric, and therefore it is also a linear function in the second argument. This means that for every $x, y, z \in V$ and $\alpha, \beta \in \mathbb{R}$,

$$\langle x, \alpha y + \beta z \rangle = \alpha \langle x, y \rangle + \beta \langle x, z \rangle.$$

If we use property (3) with the value $\alpha = 0$, we get $\langle o, x \rangle = \langle x, o \rangle = 0$, so the product of any vector with the zero vector gives zero.

Now we extend the definition to the field of complex numbers. Recall that the complex conjugate of a number $a + bi \in \mathbb{C}$ is defined as $\overline{a + bi} = a - bi$. Due to the fourth property below, necessarily $\langle x, x \rangle = \overline{\langle x, x \rangle}$, so $\langle x, x \rangle$ is always a real number and can be compared with zero.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 8.3 — Inner product over $\mathbb{C}$)</span></p>

Let $V$ be a vector space over $\mathbb{C}$. Then an *inner product* is a mapping $\langle \cdot, \cdot \rangle \colon V^2 \to \mathbb{C}$ satisfying for all $x, y, z \in V$ and $\alpha \in \mathbb{C}$:

1. $\langle x, x \rangle \ge 0$ and equality holds only for $x = 0$,
2. $\langle x + y, z \rangle = \langle x, z \rangle + \langle y, z \rangle$,
3. $\langle \alpha x, y \rangle = \alpha \langle x, y \rangle$,
4. $\langle x, y \rangle = \overline{\langle y, x \rangle}$.

</div>

The second and third properties again say that the inner product is a linear function in the first argument. What about the second?

$$\langle x, y + z \rangle = \overline{\langle y + z, x \rangle} = \overline{\langle y, x \rangle} + \overline{\langle z, x \rangle} = \langle x, y \rangle + \langle x, z \rangle,$$

$$\langle x, \alpha y \rangle = \overline{\langle \alpha y, x \rangle} = \overline{\alpha} \overline{\langle y, x \rangle} = \overline{\alpha} \langle x, y \rangle.$$

In the second argument, the complex inner product is therefore no longer linear.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.4 — Examples of standard inner products)</span></p>

- In the space $\mathbb{R}^n$: the standard inner product $\langle x, y \rangle = x^\top y = \sum_{i=1}^{n} x_i y_i$.
- In the space $\mathbb{C}^n$: the standard inner product $\langle x, y \rangle = x^\top \overline{y} = \sum_{i=1}^{n} x_i \overline{y_i}$.
- In the space $\mathbb{R}^{m \times n}$: the standard inner product $\langle A, B \rangle = \sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij} b_{ij}$.
- In $\mathcal{C}_{[a,b]}$, the space of continuous functions on the interval $[a, b]$: the standard inner product $\langle f, g \rangle = \int_a^b f(x) g(x) \, dx$.

</div>

The inner products mentioned above are only examples of possible products on the given spaces; other operations can also serve as inner products. Later, in Theorem 11.18, we will describe all inner products on the space $\mathbb{R}^n$.

It is worth noting that the mapping $\langle x, y \rangle = x^\top y$ on the space $\mathbb{C}^n$ does not form an inner product, because for example for the vector $x = (i, i)^\top$ we would have $\langle x, x \rangle = x^\top x = -2$.

From now on, let us consider a vector space $V$ over $\mathbb{R}$ or $\mathbb{C}$ with an inner product. We first show that the inner product allows us to introduce a norm, i.e., the magnitude of a vector.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 8.5 — Norm induced by an inner product)</span></p>

The *norm induced by an inner product* is defined as $\|x\| := \sqrt{\langle x, x \rangle}$, where $x \in V$.

</div>

The norm is well-defined thanks to the first property of the inner product definition, and it is always a non-negative value. For the standard inner product in $\mathbb{R}^n$, we obtain the Euclidean norm $\|x\| = \sqrt{x^\top x} = \sqrt{\sum_{i=1}^{n} x_i^2}$.

As we recalled in Example 8.1, for the standard inner product in $\mathbb{R}^n$, $x, y$ are orthogonal if and only if $\langle x, y \rangle = 0$. In other vector spaces, such geometric intuition is lacking, so we define orthogonality precisely through the relation $\langle x, y \rangle = 0$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 8.6 — Orthogonality)</span></p>

Vectors $x, y \in V$ are *orthogonal* if $\langle x, y \rangle = 0$. Notation: $x \perp y$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.7 — Examples of orthogonal vectors for standard inner products)</span></p>

- In the space $\mathbb{R}^3$: $(1, 2, 3) \perp (1, 1, -1)$.
- In the space $\mathbb{R}^n$: the $i$-th row of a nonsingular matrix $A \in \mathbb{R}^{n \times n}$ and the $j$-th column of the matrix $A^{-1}$ for any $i \neq j$.
- In the space $\mathcal{C}_{[-\pi, \pi]}$: $\sin x \perp \cos x \perp 1$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 8.8 — Pythagorean)</span></p>

*If $x, y \in V$ are orthogonal, then $\|x + y\|^2 = \|x\|^2 + \|y\|^2$.*

*Proof.* $\|x + y\|^2 = \langle x + y, x + y \rangle = \langle x, x \rangle + \underbrace{\langle x, y \rangle}\_{=0} + \underbrace{\langle y, x \rangle}\_{=0} + \langle y, y \rangle = \|x\|^2 + \|y\|^2$.

</div>

Note that over $\mathbb{R}$ the converse implication also holds, but over $\mathbb{C}$ it generally does not (see Problem 8.2).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 8.9 — Cauchy–Schwarz inequality)</span></p>

*For every $x, y \in V$, $|\langle x, y \rangle| \le \|x\| \cdot \|y\|$.*

*Proof.* (Real version) We first show the real version. For $y = o$ the inequality holds trivially, so assume $y \neq o$. Consider the real function $f(t) = \langle x + ty, x + ty \rangle \ge 0$ of the variable $t \in \mathbb{R}$. Then

$$f(t) = \langle x, x \rangle + 2t\langle x, y \rangle + t^2 \langle y, y \rangle.$$

This is a quadratic function that is everywhere non-negative, so it cannot have two distinct roots. Therefore the corresponding discriminant is non-positive:

$$4\langle x, y \rangle^2 - 4\langle x, x \rangle \langle y, y \rangle \le 0.$$

From this we get $\langle x, y \rangle^2 \le \langle x, x \rangle \langle y, y \rangle$, and taking square roots, $|\langle x, y \rangle| \le \|x\| \cdot \|y\|$.

*Proof.* (Complex version) For $y = o$ the claim holds trivially. Let $y \neq o$ and without loss of generality assume that $\|y\| = 1$. Define the scalar $\alpha := \langle x, y \rangle$, the vector $z := x - \alpha y$, and compute

$$0 \le \langle z, z \rangle = \langle x - \alpha y, \, x - \alpha y \rangle = \langle x, x \rangle - \overline{\alpha}\langle x, y \rangle - \alpha \langle y, x \rangle + \alpha \overline{\alpha} \langle y, y \rangle.$$

Since $\alpha \overline{\alpha} = |\alpha|^2$, $\langle y, y \rangle = 1$, and $\alpha = \langle x, y \rangle$, we get

$$0 \le \langle x, x \rangle - |\alpha|^2 = \langle x, x \rangle - |\langle x, y \rangle|^2.$$

Thus $|\alpha|^2 \le \langle x, x \rangle$ and taking square roots of both sides we have $|\langle x, y \rangle| \le \|x\|$.

</div>

Sometimes the Cauchy–Schwarz inequality is stated in the equivalent form

$$|\langle x, y \rangle|^2 \le \langle x, x \rangle \langle y, y \rangle.$$

The Cauchy–Schwarz inequality is used for deriving further results in a general setting, or even for specific algebraic expressions. For example, for the standard inner product in $\mathbb{R}^n$ we obtain the inequality

$$\left(\sum_{i=1}^{n} x_i y_i\right)^2 \le \left(\sum_{i=1}^{n} x_i^2\right) \left(\sum_{i=1}^{n} y_i^2\right).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 8.10 — Triangle inequality)</span></p>

*For every $x, y \in V$, $\|x + y\| \le \|x\| + \|y\|$.*

*Proof.* First recall that for every complex number $z = a + bi$: $z + \overline{z} = 2a = 2\operatorname{Re}(z)$, and furthermore $a \le |z|$. Now we can derive:

$$\|x + y\|^2 = \langle x + y, x + y \rangle = \langle x, x \rangle + \langle y, y \rangle + 2\operatorname{Re}(\langle x, y \rangle) \le \|x\|^2 + \|y\|^2 + 2|\langle x, y \rangle| \le (\|x\| + \|y\|)^2,$$

where the last inequality follows from the Cauchy–Schwarz inequality.

</div>

#### Norm in general

The norm induced by an inner product is just one type of norm, but the concept of a norm is defined more generally. We will mostly work with the norm induced by an inner product, so the following section is only a brief digression.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 8.11 — Norm)</span></p>

Let $V$ be a vector space over $\mathbb{R}$ or $\mathbb{C}$. Then a *norm* is a mapping $\|\cdot\| \colon V \to \mathbb{R}$ satisfying:

1. $\|x\| \ge 0$ for all $x \in V$, and equality holds only for $x = 0$,
2. $\|\alpha x\| = |\alpha| \cdot \|x\|$ for all $x \in V$ and for all $\alpha \in \mathbb{R}$ resp. $\alpha \in \mathbb{C}$,
3. $\|x + y\| \le \|x\| + \|y\|$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 8.12)</span></p>

*The norm induced by an inner product is a norm.*

*Proof.* Property (1) is satisfied by the definition of the norm induced by an inner product. Property (3) is shown in Corollary 8.10. It remains to verify property (2):

$$\|\alpha x\| = \sqrt{\langle \alpha x, \alpha x \rangle} = \sqrt{\alpha \overline{\alpha} \langle x, x \rangle} = \sqrt{\alpha \overline{\alpha}} \sqrt{\langle x, x \rangle} = |\alpha| \cdot \|x\|.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.13 — Examples of norms in $\mathbb{R}^n$)</span></p>

A special class of norms are the so-called $p$-norms. For $p = 1, 2, \ldots$ we define the $p$-norm of a vector $x \in \mathbb{R}^n$ as

$$\|x\|_p = \left(\sum_{i=1}^{n} |x_i|^p\right)^{1/p}.$$

Special choices of $p$ lead to well-known norms:

- for $p = 2$: the Euclidean norm $\|x\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$, which is the norm induced by the standard inner product,
- for $p = 1$: the sum norm $\|x\|_1 = \sum_{i=1}^{n} |x_i|$; it is called the Manhattan norm because it corresponds to real distances when traversing a rectangular grid of streets in a city,
- for $p = \infty$ (by a limiting process): the maximum (Chebyshev) norm $\|x\|\_\infty = \max_{i=1,\ldots,n} |x_i|$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.14 — Unit ball)</span></p>

The unit ball is the set of vectors whose norm is at most 1, and thus they are at distance at most 1 from the origin. Formally, we define the unit ball as

$$\lbrace x \in V \,;\, \|x\| \le 1 \rbrace.$$

Different norms have different geometric objects as their unit balls. However, every unit ball in $\mathbb{R}^n$ must be closed, bounded, symmetric about the origin, convex (i.e., together with any two points it contains their connecting segment), and the origin lies in its interior. The converse also holds — every set satisfying these properties represents the unit ball of some norm.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.15 — Examples of norms in $\mathcal{C}_{[a,b]}$)</span></p>

The norm of a continuous function $f \colon [a, b] \to \mathbb{R}$ can be introduced analogously to the Euclidean space:

- analogue of the Euclidean norm: $\|f\|_2 = \sqrt{\int_a^b f(x)^2 \, dx}$,
- analogue of the sum norm: $\|f\|_1 = \int_a^b |f(x)| \, dx$,
- analogue of the maximum norm: $\|f\|\_\infty = \max_{x \in [a,b]} |f(x)|$,
- analogue of the $p$-norm: $\|f\|_p = \left(\int_a^b |f(x)|^p \, dx\right)^{1/p}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 8.16 — Parallelogram law)</span></p>

For the norm induced by an inner product, the so-called *parallelogram law* holds:

$$\|x - y\|^2 + \|x + y\|^2 = 2\|x\|^2 + 2\|y\|^2.$$

*Proof.* $\|x - y\|^2 + \|x + y\|^2 = \langle x - y, x - y \rangle + \langle x + y, x + y \rangle = 2\langle x, x \rangle + 2\langle y, y \rangle = 2\|x\|^2 + 2\|y\|^2$.

Using this, we can easily see that the sum norm and the maximum norm are not induced by any inner product. An even stronger statement holds: if the parallelogram law holds for a norm, then it is induced by some inner product; see Horn and Johnson [1985].

</div>

The norm allows us to introduce distance (or metric) between vectors $x, y$ as $\|x - y\|$. And once we have distance, we can introduce limits, etc. Moreover, to define a metric we do not even need a vector space; any set suffices.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 8.17 — Metric)</span></p>

A metric on a set $M$ is defined as a mapping $d \colon M^2 \to \mathbb{R}$ satisfying:

1. $d(x, y) \ge 0$ for all $x, y \in M$, and equality holds only for $x = y$,
2. $d(x, y) = d(y, x)$ for all $x, y \in M$,
3. $d(x, z) \le d(x, y) + d(y, z)$ for all $x, y, z \in M$.

Every norm determines a metric by the formula $d(x, y) := \|x - y\|$, i.e., it defines the distance of vectors $x, y$ as the magnitude of their difference. In the opposite direction, however, this is generally not true. There exist spaces with a metric that is not induced by any norm, e.g., the discrete metric $d(x, y) := \lceil \|x - y\|_2 \rceil$, or the discrete metric $d(x, y) := 1$ for $x \neq y$ and $d(x, y) := 0$ for $x = y$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.18 — Classification of handwritten digits)</span></p>

We demonstrate the use of distance for creating a simple classifier for automatic identification of handwritten digits. We assume that each digit is given as an image, represented by a matrix $A \in \mathbb{R}^{m \times n}$, so the pixel of the image at position $(i, j)$ has color number $a_{ij}$. As templates we use averaged values from the MNIST database.

We therefore need to introduce a metric on the space of matrices. For this purpose, we adapt the classical Euclidean distance and define the distance of matrices $A, B \in \mathbb{R}^{m \times n}$ as

$$\|A - B\| := \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} (a_{ij} - b_{ij})^2}.$$

If we compute the distances between the matrix representing the classified image and the individual templates, we classify according to the smallest distance. However, such a simple classifier can easily make mistakes. Our classifier cannot recognize the shapes of strokes (sign of curvature, etc.), and therefore there may also be a small distance to the image of the digit 3.

</div>

### 8.2 Orthonormal basis, Gram–Schmidt orthogonalization

Every vector space has a basis. For a space with an inner product, it is natural to ask whether there exists a basis consisting of mutually orthogonal vectors. In this section we show that this is indeed true, that such a basis has a number of remarkable properties, and we also derive an algorithm for finding it.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 8.19 — Orthogonal and orthonormal system)</span></p>

A system of vectors $z_1, \ldots, z_n$ is *orthogonal* if $\langle z_i, z_j \rangle = 0$ for all $i \neq j$. A system of vectors $z_1, \ldots, z_n$ is *orthonormal* if it is orthogonal and $\|z_i\| = 1$ for all $i = 1, \ldots, n$.

</div>

If a system $z_1, \ldots, z_n$ is orthonormal, then it is also orthogonal. The converse does not hold in general, but it is easy to orthonormalize an orthogonal system. If $z_1, \ldots, z_n$ are nonzero and orthogonal, then $\frac{1}{\|z_1\|} z_1, \ldots, \frac{1}{\|z_n\|} z_n$ is orthonormal.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.20)</span></p>

In the space $\mathbb{R}^n$ with the standard inner product, an example of an orthonormal system is the canonical basis $e_1, \ldots, e_n$. In particular, in the plane $\mathbb{R}^2$, the vectors $(1, 0)^\top$, $(0, 1)^\top$ form an orthonormal basis. Another example of an orthonormal basis in $\mathbb{R}^2$ is: $\frac{\sqrt{2}}{2}(1, 1)^\top$, $\frac{\sqrt{2}}{2}(-1, 1)^\top$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 8.21)</span></p>

*If a system of vectors $z_1, \ldots, z_n$ is orthonormal, then it is linearly independent.*

*Proof.* Consider a linear combination $\sum_{i=1}^{n} \alpha_i z_i = o$. Then for every $k = 1, \ldots, n$:

$$0 = \langle o, z_k \rangle = \left\langle \sum_{i=1}^{n} \alpha_i z_i, z_k \right\rangle = \sum_{i=1}^{n} \alpha_i \langle z_i, z_k \rangle = \alpha_k \langle z_k, z_k \rangle = \alpha_k.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 8.22 — Fourier coefficients)</span></p>

*Let $z_1, \ldots, z_n$ be an orthonormal basis of the space $V$. Then for every $x \in V$, $x = \sum_{i=1}^{n} \langle x, z_i \rangle z_i$.*

*Proof.* We know that $x = \sum_{i=1}^{n} \alpha_i z_i$ and the coordinates $\alpha_1, \ldots, \alpha_n$ are unique (Theorem 5.33). Now for every $k = 1, \ldots, n$:

$$\langle x, z_k \rangle = \left\langle \sum_{i=1}^{n} \alpha_i z_i, z_k \right\rangle = \sum_{i=1}^{n} \alpha_i \langle z_i, z_k \rangle = \alpha_k \langle z_k, z_k \rangle = \alpha_k.$$

</div>

The expression $x = \sum_{i=1}^{n} \langle x, z_i \rangle z_i$ is called the *Fourier expansion*, and the scalars $\langle x, z_i \rangle$, $i = 1, \ldots, n$ are called *Fourier coefficients*. The geometric meaning of the Fourier coefficient $\langle x, z_i \rangle$ is that $\langle x, z_i \rangle z_i$ gives the projection of the vector $x$ onto the line $\operatorname{span}\lbrace z_i \rbrace$. The vector $x$ can then be composed from these partial projections by simple summation $x = \sum_{i=1}^{n} \langle x, z_i \rangle z_i$ (we will discuss projections further in Section 8.3). If the basis $z_1, \ldots, z_n$ were not orthonormal, then this property would no longer hold in general.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Algorithm 8.23 — Gram–Schmidt orthogonalization)</span></p>

Input: linearly independent vectors $x_1, \ldots, x_n \in V$.

1. **for** $k := 1$ **to** $n$ **do**
2. $\qquad y_k := x_k - \sum_{j=1}^{k-1} \langle x_k, z_j \rangle z_j$, $\quad$ // compute the orthogonal component
3. $\qquad z_k := \frac{1}{\|y_k\|} y_k$, $\quad$ // normalize the length to 1
4. **end for**

Output: $z_1, \ldots, z_n$ orthonormal basis of the space $\operatorname{span}\lbrace x_1, \ldots, x_n \rbrace$.

*Proof.* (Correctness of Gram–Schmidt orthogonalization.) By mathematical induction on $n$ we prove that $z_1, \ldots, z_n$ is an orthonormal basis of the space $\operatorname{span}\lbrace x_1, \ldots, x_n \rbrace$. For $n = 1$, $y_1 = x_1 \neq o$, the vector $z_1 = \frac{1}{\|x_1\|} x_1$ is well-defined and $\operatorname{span}\lbrace x_1 \rbrace = \operatorname{span}\lbrace z_1 \rbrace$.

Induction step $n \leftarrow n - 1$. Assume that $z_1, \ldots, z_{n-1}$ is an orthonormal basis of the space $\operatorname{span}\lbrace x_1, \ldots, x_{n-1} \rbrace$. If $y_n = o$, then $x_n = \sum_{j=1}^{n-1} \langle x_n, z_j \rangle z_j$ and $x_n \in \operatorname{span}\lbrace z_1, \ldots, z_{n-1} \rbrace = \operatorname{span}\lbrace x_1, \ldots, x_{n-1} \rbrace$, which would be a contradiction with the linear independence of the vectors $x_1, \ldots, x_n$. Therefore $y_n \neq o$ and $z_n = \frac{1}{\|y_n\|} y_n$ is well-defined and has unit norm.

Now we prove that $z_1, \ldots, z_n$ is an orthonormal system. By the induction hypothesis, $z_1, \ldots, z_{n-1}$ is an orthonormal system and therefore $\langle z_i, z_j \rangle$ equals 0 for $i \neq j$ and equals 1 for $i = j$. It suffices to show that $z_n$ is orthogonal to all other $z_i$ for $i < n$:

$$\langle z_n, z_i \rangle = \frac{1}{\|y_n\|} \langle y_n, z_i \rangle = \frac{1}{\|y_n\|} \langle x_n, z_i \rangle - \frac{1}{\|y_n\|} \sum_{j=1}^{n-1} \langle x_n, z_j \rangle \langle z_j, z_i \rangle = \frac{1}{\|y_n\|} \langle x_n, z_i \rangle - \frac{1}{\|y_n\|} \langle x_n, z_i \rangle = 0.$$

It remains to verify $\operatorname{span}\lbrace z_1, \ldots, z_n \rbrace = \operatorname{span}\lbrace x_1, \ldots, x_n \rbrace$. From the algorithm it is clear that $z_n \in \operatorname{span}\lbrace z_1, \ldots, z_{n-1}, x_n \rbrace \subseteq \operatorname{span}\lbrace x_1, \ldots, x_n \rbrace$, and therefore $\operatorname{span}\lbrace z_1, \ldots, z_n \rbrace \subseteq \operatorname{span}\lbrace x_1, \ldots, x_n \rbrace$. Since both spaces have the same dimension, equality holds (Theorem 5.50).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.24 — Gram–Schmidt orthogonalization)</span></p>

With the standard inner product, we want to find an orthonormal basis of the space generated by the vectors

$$x_1 = (1, 0, 1, 0)^\top, \quad x_2 = (1, 1, 1, 1)^\top, \quad x_3 = (1, 0, 0, 1)^\top.$$

We proceed exactly according to the algorithm:

$$y_1 := x_1, \qquad z_1 := \frac{1}{\|y_1\|} y_1 = \frac{\sqrt{2}}{2}(1, 0, 1, 0)^\top,$$

$$y_2 := x_2 - \langle x_2, z_1 \rangle z_1 = (1, 1, 1, 1)^\top - \sqrt{2} \frac{\sqrt{2}}{2}(1, 0, 1, 0)^\top = (0, 1, 0, 1)^\top,$$

$$z_2 := \frac{1}{\|y_2\|} y_2 = \frac{\sqrt{2}}{2}(0, 1, 0, 1)^\top,$$

$$y_3 := x_3 - \langle x_3, z_1 \rangle z_1 - \langle x_3, z_2 \rangle z_2 = (1, 0, 0, 1)^\top - \frac{\sqrt{2}\sqrt{2}}{2}(1, 0, 1, 0)^\top - \frac{\sqrt{2}\sqrt{2}}{2}(0, 1, 0, 1)^\top = \frac{1}{2}(1, -1, -1, 1)^\top,$$

$$z_3 := \frac{1}{\|y_3\|} y_3 = \frac{1}{2}(1, -1, -1, 1)^\top.$$

The resulting orthonormal basis consists of the vectors $z_1, z_2, z_3$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 8.25 — Computational complexity)</span></p>

For the analysis of the computational complexity of Algorithm 8.23, consider vectors $x_1, \ldots, x_n \in \mathbb{R}^m$. The inner product of two vectors from the space $\mathbb{R}^m$ requires on the order of $2m$ arithmetic operations. Step 2 therefore has an asymptotic complexity of $4m(k-1)$ operations and in step 3 it is $3m$. In total we get

$$\sum_{k=1}^{n} (4mk - m) = 4m \frac{1}{2} n(n+1) - mn,$$

which gives a computational complexity of order $2mn^2$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 8.26 — Existence of an orthonormal basis)</span></p>

*Every finitely generated space (with an inner product) has an orthonormal basis.*

*Proof.* We know (Theorem 5.41) that every finitely generated space has a basis, and we can orthogonalize it using the Gram–Schmidt method.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 8.27 — Extension of an orthonormal system to an orthonormal basis)</span></p>

*Every orthonormal system of vectors in a finitely generated space can be extended to an orthonormal basis.*

*Proof.* We know (Theorem 5.49) that every orthonormal system of vectors $z_1, \ldots, z_m$ can be extended to a basis $z_1, \ldots, z_m, x_{m+1}, \ldots, x_n$, and we can orthogonalize it using the Gram–Schmidt method to obtain $z_1, \ldots, z_m, z_{m+1}, \ldots, z_n$. The orthogonalization does not change the first $m$ vectors.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 8.28 — Bessel's inequality and Parseval's equality)</span></p>

*Let $z_1, \ldots, z_n$ be an orthonormal system in $V$ and let $x \in V$. Then:*

1. *Bessel's inequality:* $\|x\|^2 \ge \sum_{j=1}^{n} |\langle x, z_j \rangle|^2$,
2. *Parseval's equality:* $\|x\|^2 = \sum_{j=1}^{n} |\langle x, z_j \rangle|^2$ *if and only if $x \in \operatorname{span}\lbrace z_1, \ldots, z_n \rbrace$.*

*Proof.*

(1) Follows from the computation

$$0 \le \left\langle x - \sum_{j=1}^{n} \langle x, z_j \rangle z_j, \, x - \sum_{j=1}^{n} \langle x, z_j \rangle z_j \right\rangle = \|x\|^2 - \sum_{j=1}^{n} |\langle x, z_j \rangle|^2.$$

(2) Follows from the above, since equality holds if and only if $x = \sum_{j=1}^{n} \langle x, z_j \rangle z_j$.

</div>

Bessel's inequality says that the norm of a vector $x$ can never be smaller than the norm of its projection into any subspace, here expressed as $\operatorname{span}\lbrace z_1, \ldots, z_n \rbrace$.

Parseval's equality shows that for vectors close to the origin, their coordinates must also be sufficiently small. Furthermore, the equality can be generalized to infinite-dimensional spaces such as $\mathcal{C}\_{[-\pi, \pi]}$, which among other things means that Fourier coefficients in an infinite expansion must converge to zero.

Parseval's equality also says, in other words, that in any finitely generated space $V$, the norm of any $x \in V$ can be expressed as the standard Euclidean norm of its coordinate vector:

$$\|x\| = \|[x]_B\|_2 = \sqrt{[x]_B^\top [x]_B},$$

where $B$ is an orthonormal basis of $V$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 8.29)</span></p>

*Let $B$ be an orthonormal basis of the space $V$ and let $x, y \in V$. Then $\langle x, y \rangle = [x]_B^\top \overline{[y]_B}$.*

*Proof.* Let $B = \lbrace z_1, \ldots, z_n \rbrace$. By Theorem 8.22, $[x]_B = (\langle x, z_1 \rangle, \ldots, \langle x, z_n \rangle)^\top$. Now

$$\langle x, y \rangle = \left\langle \sum_{j=1}^{n} \langle x, z_j \rangle z_j, y \right\rangle = \sum_{j=1}^{n} \langle x, z_j \rangle \overline{\langle y, z_j \rangle} = [x]_B^\top \overline{[y]_B}.$$

</div>

It is not hard to see that this theorem also holds in the converse direction. Thus we obtain that the mapping $\langle \cdot, \cdot \rangle$ is an inner product on the space $V$ if and only if it can be expressed as $\langle x, y \rangle = [x]_B^\top \overline{[y]_B}$ for some (or for any) orthonormal basis $B$. Every inner product is therefore the standard inner product when viewed from any orthonormal basis.

### 8.3 Orthogonal complement and projection

In this section we derive a method for computing the distance from a point to a subspace (for example, from a point to a line, from a point to a plane, ...) and also for determining the point in the subspace that is closest to a given point. This will allow us to solve both purely geometric problems and problems that seemingly have nothing to do with geometry.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 8.30 — Orthogonal complement)</span></p>

Let $V$ be a vector space and $M \subseteq V$. Then the *orthogonal complement* of the set $M$ is $M^\perp := \lbrace x \in V \,;\, \langle x, y \rangle = 0 \;\forall y \in M \rbrace$.

</div>

The orthogonal complement $M^\perp$ thus contains those vectors $x$ that are orthogonal to all vectors from $M$ (sometimes we say briefly that $x$ is orthogonal to $M$). Clearly $\lbrace o \rbrace^\perp = V$ and $V^\perp = \lbrace o \rbrace$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.31)</span></p>

The orthogonal complement of the vector $(2, 5)^\top$ is the line $\operatorname{span}\lbrace (5, -2)^\top \rbrace$. The orthogonal complement of the entire line $\operatorname{span}\lbrace (2, 5)^\top \rbrace$ is also the line $\operatorname{span}\lbrace (5, -2)^\top \rbrace$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 8.32 — Properties of the orthogonal complement of a set)</span></p>

*Let $V$ be a vector space and $M, N \subseteq V$. Then*

1. *$M^\perp$ is a subspace of $V$,*
2. *if $M \subseteq N$ then $M^\perp \supseteq N^\perp$,*
3. *$M^\perp = \operatorname{span}(M)^\perp$.*

*Proof.*

(1) We verify the subspace properties: $o \in M^\perp$ trivially. Now let $x_1, x_2 \in M^\perp$. Then $\langle x_1, y \rangle = \langle x_2, y \rangle = 0$ $\forall y \in M$, so also $\langle x_1 + x_2, y \rangle = \langle x_1, y \rangle + \langle x_2, y \rangle = 0$. Finally, let $x \in M^\perp$. Then for every scalar $\alpha$, $\langle \alpha x, y \rangle = \alpha \langle x, y \rangle = 0$.

(2) Let $x \in N^\perp$, i.e., $\langle x, y \rangle = 0$ $\forall y \in N$. A fortiori $\langle x, y \rangle = 0$ $\forall y \in M \subseteq N$, and therefore $x \in M^\perp$.

(3) $M \subseteq \operatorname{span}(M)$, so by the previous part $M^\perp \supseteq \operatorname{span}(M)^\perp$. The proof of the second inclusion relies on the fact that if a vector $x$ is orthogonal to certain vectors, then it is orthogonal to their linear combinations, and hence to their linear span.

</div>

Property (3) says that the orthogonal complement of a space or of its basis is the same. This simplifies the practical task of finding the orthogonal complement, because it suffices to verify orthogonality only against the basis vectors.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 8.33 — Properties of the orthogonal complement of a subspace)</span></p>

*Let $U$ be a subspace of a vector space $V$. Then:*

1. *If $z_1, \ldots, z_m$ is an orthonormal basis of $U$, and $z_1, \ldots, z_m, z_{m+1}, \ldots, z_n$ is its extension to an orthonormal basis of $V$, then $z_{m+1}, \ldots, z_n$ is an orthonormal basis of $U^\perp$.*
2. *$\dim V = \dim U + \dim U^\perp$,*
3. *$V = U + U^\perp$,*
4. *$(U^\perp)^\perp = U$,*
5. *$U \cap U^\perp = \lbrace o \rbrace$.*

*Proof.*

(1) Clearly $z_{m+1}, \ldots, z_n$ is an orthonormal system in $V$, and so it suffices to prove $\operatorname{span}\lbrace z_{m+1}, \ldots, z_n \rbrace = U^\perp$. Inclusion "$\supseteq$". Every $x \in V$ has a Fourier expansion $x = \sum_{i=1}^{n} \langle x, z_i \rangle z_i$. If $x \in U^\perp$, then $\langle x, z_i \rangle = 0$, $i = 1, \ldots, m$, and therefore $x = \sum_{i=m+1}^{n} \langle x, z_i \rangle z_i \in \operatorname{span}\lbrace z_{m+1}, \ldots, z_n \rbrace$. Inclusion "$\subseteq$". Let $x \in \operatorname{span}\lbrace z_{m+1}, \ldots, z_n \rbrace$, then $x = \sum_{i=m+1}^{n} \langle x, z_i \rangle z_i = \sum_{i=1}^{n} 0 z_i + \sum_{i=m+1}^{n} \langle x, z_i \rangle z_i$. By the uniqueness of coordinates we get $\langle x, z_i \rangle = 0$, $i = 1, \ldots, m$, and thus $x \in U^\perp$.

(2) From the first property we have $\dim V = n$, $\dim U = m$, $\dim U^\perp = n - m$.

(3) From the first property we have $x = \sum_{i=1}^{m} \langle x, z_i \rangle z_i + \sum_{i=m+1}^{n} \langle x, z_i \rangle z_i \in U + U^\perp$.

(4) From the first property, $z_{m+1}, \ldots, z_n$ is an orthonormal basis of $U^\perp$, so $z_1, \ldots, z_m$ is an orthonormal basis of $(U^\perp)^\perp$.

(5) From the above and by Theorem 5.56 on the dimension of join and intersection, $\dim(U \cap U^\perp) = \dim V - \dim U - \dim U^\perp = 0$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 8.35 — Orthogonal projection)</span></p>

Let $V$ be a vector space and $U$ its subspace. Then by the *projection* of a vector $x \in V$ into the subspace $U$ we mean such a vector $x_U \in U$ that satisfies

$$\|x - x_U\| = \min_{y \in U} \|x - y\|.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 8.36 — On orthogonal projection)</span></p>

*Let $U$ be a subspace of a vector space $V$. Then for every $x \in V$ there exists exactly one projection $x_U \in U$ into the subspace $U$. Moreover, if $z_1, \ldots, z_m$ is an orthonormal basis of $U$, then*

$$x_U = \sum_{i=1}^{m} \langle x, z_i \rangle z_i.$$

*Proof.* Let $z_1, \ldots, z_m, z_{m+1}, \ldots, z_n$ be an extension to an orthonormal basis of $V$. Define $x_U := \sum_{i=1}^{m} \langle x, z_i \rangle z_i \in U$ and show that this is the desired vector. Now

$$x - x_U = \sum_{i=1}^{n} \langle x, z_i \rangle z_i - \sum_{i=1}^{m} \langle x, z_i \rangle z_i = \sum_{i=m+1}^{n} \langle x, z_i \rangle z_i \in U^\perp.$$

Let $y \in U$ be arbitrary. Now we have $x - x_U \in U^\perp$ and $x_U - y \in U$. Thus $(x - x_U) \perp (x_U - y)$ and we can apply the Pythagorean theorem, which gives

$$\|x - y\|^2 = \|(x - x_U) + (x_U - y)\|^2 = \|x - x_U\|^2 + \|x_U - y\|^2 \ge \|x - x_U\|^2,$$

that is, $\|x - y\| \ge \|x - x_U\|$, which proves minimality. Uniqueness: equality holds only when $\|x_U - y\|^2 = 0$, i.e., when $x_U = y$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.37)</span></p>

We want to find the projection $x_U$ of the vector $x = (1, 2, 4, 5)^\top$ into the subspace $U$ generated by the vectors

$$x_1 = (1, 0, 1, 0)^\top, \quad x_2 = (1, 1, 1, 1)^\top, \quad x_3 = (1, 0, 0, 1)^\top$$

and determine the distance from $x$ to $U$ with the standard inner product.

First, we find an orthonormal basis of the subspace $U$. This was already done in Example 8.24, and the orthonormal basis consists of the vectors

$$z_1 = \frac{\sqrt{2}}{2}(1, 0, 1, 0)^\top, \quad z_2 = \frac{\sqrt{2}}{2}(0, 1, 0, 1)^\top, \quad z_3 = \frac{1}{2}(1, -1, -1, 1)^\top.$$

Now we find the projection using the formula

$$x_U = \sum_{i=1}^{3} \langle x, z_i \rangle z_i = \frac{1}{2}(5, 7, 5, 7)^\top.$$

The desired distance is $\|x - x_U\| = \|\frac{1}{2}(-3, -3, 3, 3)^\top\| = 3$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.38 — Projection onto a line)</span></p>

Let $a \in \mathbb{R}^n$ be a nonzero vector and consider the projection of a vector $x \in \mathbb{R}^n$ onto the line with direction $a$, i.e., the projection into the subspace $U = \operatorname{span}\lbrace a \rbrace$. The orthonormal basis of the space $U$ is the vector $z = \frac{1}{\|a\|} a$ and by formula (8.2), the projection of the vector $x$ has the form

$$x_U = \langle x, z \rangle z = \frac{1}{\|a\|^2} \langle x, a \rangle a = \frac{x^\top a}{a^\top a} a.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 8.39)</span></p>

We have already implicitly used the projection several times before we formally introduced it:

- In the proof of the complex version of the Cauchy–Schwarz inequality (Theorem 8.9). The vector $\langle x, y \rangle y$ expressed the projection of the vector $x$ onto the line $\operatorname{span}\lbrace y \rbrace$ and the vector $z$ represented the difference between $x$ and its projection.
- The Fourier expansion from Theorem 8.22 is essentially a decomposition of the vector $x$ into a sum of projections onto the individual lines $\operatorname{span}\lbrace z_i \rbrace$, $i = 1, \ldots, n$.
- The Gram–Schmidt orthogonalization in the $k$-th cycle of Algorithm 8.23 constructs the projection of the vector $x_k$ into the subspace $\operatorname{span}\lbrace x_1, \ldots, x_{k-1} \rbrace$. By subtracting the projection from the vector $x_k$, we obtain the desired orthogonal vector $y_k$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 8.40)</span></p>

By properties (3) and (5) of Theorem 8.33, the space $V$ can be expressed as a direct sum of the subspaces $U$ and $U^\perp$, i.e., $V = U \oplus U^\perp$ (see Remark 5.58). This means, among other things, that every vector $v \in V$ has a unique representation $v = u + u'$, where $u \in U$ and $u' \in U^\perp$. Moreover, by Theorem 8.36, the vector $u$ is the projection of $v$ into $U$, and the vector $u'$ is the projection of $v$ into $U^\perp$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 8.41)</span></p>

*Using the notation from Theorem 8.36, if some $y \in U$ satisfies $x - y \in U^\perp$, then $y = x_U$.*

*Proof.* Since $(x - y) \perp (y - x_U)$, we apply the Pythagorean theorem, which gives

$$\|x - x_U\|^2 = \|x - y\|^2 + \|y - x_U\|^2 \ge \|x - y\|^2.$$

We get $\|x - x_U\| \ge \|x - y\|$, so by the property and uniqueness of the projection, $y = x_U$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.42 — Legendre polynomials)</span></p>

Consider the space of polynomials $\mathcal{P}^n$. If we realize that $\mathcal{P}^n$ is a subspace of the space of continuous functions $\mathcal{C}\_{[a,b]}$, we can use the standard inner product of $\mathcal{C}\_{[a,b]}$ on $\mathcal{P}^n$. If we orthogonalize the vectors $1, x, x^2, \ldots$ specifically on $\mathcal{C}\_{[-1,1]}$, we obtain the so-called *Legendre polynomials*

$$p_0(x) = 1, \quad p_1(x) = x, \quad p_2(x) = \frac{1}{2}(3x^2 - 1), \quad p_3(x) = \frac{1}{2}(5x^3 - 3x), \quad \ldots$$

These polynomials are mutually orthogonal, but for the sake of certain applications they are normalized so that the $n$-th polynomial has norm $2/(2n+1)$.

Legendre polynomials can be used, for example, to approximate a function by a polynomial, cf. the method in Section 3.6. If we want to approximate a function $f$ by a polynomial of degree $n$, we compute the projection of $f$ into the subspace $\mathcal{P}^n$ in this inner product using Theorem 8.36, and as the orthonormal basis of $\mathcal{P}^n$ we use the Legendre polynomials. The resulting projection has the property that among all polynomials of degree $n$ it is closest to $f$ in the norm induced by the given inner product, which roughly corresponds to the goal of minimizing the area between $f$ and the polynomial.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.43 — Orthonormal system in a function space)</span></p>

In the space $\mathcal{C}\_{[-\pi, \pi]}$ there exists a countable orthonormal system $z_1, z_2, \ldots$ consisting of the vectors

$$\frac{1}{\sqrt{2\pi}}, \quad \frac{1}{\sqrt{\pi}} \cos x, \quad \frac{1}{\sqrt{\pi}} \sin x, \quad \frac{1}{\sqrt{\pi}} \cos 2x, \quad \frac{1}{\sqrt{\pi}} \sin 2x, \quad \frac{1}{\sqrt{\pi}} \cos 3x, \quad \frac{1}{\sqrt{\pi}} \sin 3x, \quad \ldots$$

Although this is not a basis in the true sense of the word, every function $f \in \mathcal{C}\_{[-\pi, \pi]}$ can be expressed as an infinite series $f(x) = \sum_{i=1}^{\infty} \langle f, z_i \rangle z_i$.

The expression of the first few terms $f(x) \approx \sum_{i=1}^{k} \langle f, z_i \rangle z_i$, which is essentially the projection into the space $\operatorname{span}\lbrace z_1, \ldots, z_k \rbrace$ of dimension $k$, gives a good approximation of the function $f(x)$. Such approximations are widely used in the field of signal processing (e.g., audio).

Specifically, let us compute the Fourier expansion of the function $f(x) = x$ on the interval $[-\pi, \pi]$

$$x = a_0 + \sum_{k=1}^{\infty} (a_k \sin(kx) + b_k \cos(kx)),$$

where

$$a_0 = \frac{1}{2\pi} \int_{-\pi}^{\pi} x \, dx = 0, \quad a_k = \frac{1}{\pi} \int_{-\pi}^{\pi} x \sin(kx) \, dx = (-1)^{k+1} \frac{2}{k}, \quad b_k = \frac{1}{\pi} \int_{-\pi}^{\pi} x \cos(kx) \, dx = 0.$$

Thus $x = \sum_{k=1}^{\infty} (-1)^{k+1} \frac{2}{k} \sin(kx)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 8.44 — Gram matrix)</span></p>

*Let $U$ be a subspace of a real vector space $V$. Let $U$ have a basis $B = \lbrace w_1, \ldots, w_m \rbrace$. Denote by the Gram matrix $G \in \mathbb{R}^{m \times m}$ the matrix with entries $G_{ij} = \langle w_i, w_j \rangle$. Then $G$ is a nonsingular matrix and the coordinate vector $s = [x_U]_B$ of the projection $x_U$ of any vector $x \in V$ into the subspace $U$ is the solution of the system*

$$Gs = (\langle w_1, x \rangle, \ldots, \langle w_m, x \rangle)^\top.$$

*Proof.* To prove the nonsingularity of $G$, let $s \in \mathbb{R}^m$ be a solution of the system $Gs = o$. Then the $i$-th row of the system has the form $\sum_{j=1}^{m} G_{ij} s_j = \langle w_i, \sum_{j=1}^{m} s_j w_j \rangle = 0$, i.e., $\sum_{j=1}^{m} s_j w_j \in U^\perp \cap U = \lbrace o \rbrace$. By the linear independence of $w_1, \ldots, w_m$, necessarily $s = o$.

We know that $x_U$ exists and is unique and can be written in the form $x_U = \sum_{j=1}^{m} \alpha_j w_j$ for suitable scalars $\alpha_j$. Since $x - x_U \in U^\perp$, we get in particular $\langle w_i, x - x_U \rangle = 0$ for all $i = 1, \ldots, m$. Substituting for $x_U$ we obtain $\langle w_i, x - \sum_{j=1}^{m} \alpha_j w_j \rangle = 0$, i.e.,

$$\sum_{j=1}^{m} \alpha_j \langle w_i, w_j \rangle = \langle w_i, x \rangle, \quad i = 1, \ldots, m.$$

Thus $s := [x_U]_B = (\alpha_1, \ldots, \alpha_m)^\top$ solves the system. By the nonsingularity of $G$, there exists only one solution of the system, and it corresponds to the given projection.

</div>

The Gram matrix is nonsingular if and only if the vectors $w_1, \ldots, w_m$ are linearly independent.

### 8.4 Orthogonal complement and projection in $\mathbb{R}^n$

From the previous section we know how to compute the orthogonal complement and projection for any finitely generated vector space with an inner product, using an orthonormal basis. Now we show that in $\mathbb{R}^n$ for the standard inner product, these transformations can be expressed explicitly and directly without computing an orthonormal basis.

#### Orthogonal complement

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 8.45 — Orthogonal complement in $\mathbb{R}^n$)</span></p>

*Let $A \in \mathbb{R}^{m \times n}$. Then $\mathcal{R}(A)^\perp = \operatorname{Ker}(A)$.*

*Proof.* From the properties of the orthogonal complement (Proposition 8.32(3)) we know $\mathcal{R}(A)^\perp = \lbrace A_{1*}, \ldots, A_{m*} \rbrace^\perp$. Thus $x \in \mathcal{R}(A)^\perp$ if and only if $x$ is orthogonal to the rows of the matrix $A$, i.e., $A_{i*} x = 0$ for all $i = 1, \ldots, m$. Equivalently, $Ax = o$, that is $x \in \operatorname{Ker}(A)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.46)</span></p>

Let $V$ be the space generated by the vectors $(1, 2, 3)^\top$ and $(1, -1, 0)^\top$. To determine $V^\perp$, we form the matrix

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 1 & -1 & 0 \end{pmatrix},$$

since $V = \mathcal{R}(A)$. Now it suffices to find a basis of $V^\perp = \operatorname{Ker}(A)$, which consists, for example, of the vector $(1, 1, -1)^\top$.

</div>

The characterization of the orthogonal complement also has theoretical consequences, for example the relationship between the matrix $A$ and the matrix $A^\top A$. Note that for column spaces the analogy does not hold!

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 8.47)</span></p>

*Let $A \in \mathbb{R}^{m \times n}$. Then*

1. *$\operatorname{Ker}(A^\top A) = \operatorname{Ker}(A)$,*
2. *$\mathcal{R}(A^\top A) = \mathcal{R}(A)$,*
3. *$\operatorname{rank}(A^\top A) = \operatorname{rank}(A)$.*

*Proof.*

(1) If $x \in \operatorname{Ker}(A)$, then $Ax = o$, and therefore also $A^\top Ax = A^\top o = o$, so $x \in \operatorname{Ker}(A^\top A)$. Conversely, if $x \in \operatorname{Ker}(A^\top A)$, then $A^\top Ax = o$. Multiplying by $x^\top$ we get $x^\top A^\top Ax = o$, i.e., $\|Ax\|^2 = 0$. By the property of the norm, $Ax = o$ and therefore $x \in \operatorname{Ker}(A)$.

(2) $\mathcal{R}(A^\top A) = \operatorname{Ker}(A^\top A)^\perp = \operatorname{Ker}(A)^\perp = \mathcal{R}(A)$.

(3) Trivially from the previous point.

</div>

#### Matrix spaces and linear maps

If the linear map given by $f(x) = Ax$, where $A \in \mathbb{R}^{m \times n}$, is injective, then we can introduce an inverse map from the space $f(\mathbb{R}^n) = \mathcal{S}(A)$ to the space $\mathbb{R}^n$.

If the linear map $f(x)$ is not injective, then $\dim f(\mathbb{R}^n) < n$. The only way to construct something like an inverse map is to choose a suitable subspace $U$ of $\mathbb{R}^n$ such that $\dim U = \dim f(\mathbb{R}^n)$ and simultaneously $f(U) = f(\mathbb{R}^n)$. Then the map $f(x)$ on the restricted domain $U$ represents an isomorphism between $U$ and $f(\mathbb{R}^n)$, and consequently an inverse map exists for it.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 8.48)</span></p>

*Consider the linear map $f(x) = Ax$, where $A \in \mathbb{R}^{m \times n}$. If we restrict the domain of $f(x)$ to the space $\mathcal{R}(A)$ only, we obtain an isomorphism between $\mathcal{R}(A)$ and $f(\mathbb{R}^n)$.*

*Proof.* Let $x \in \mathbb{R}^n$. Since $\mathcal{R}(A)^\perp = \operatorname{Ker}(A)$, by Remark 8.40 the vector $x$ can be decomposed as $x = x^R + x^K$, where $x^R \in \mathcal{R}(A)$ and $x^K \in \operatorname{Ker}(A)$. Then

$$f(x) = Ax = A(x^R + x^K) = Ax^R + Ax^K = Ax^R.$$

Every vector from $f(\mathbb{R}^n)$ is therefore the image of some vector from $\mathcal{R}(A)$, i.e., $f(\mathcal{R}(A)) = f(\mathbb{R}^n)$. Since both spaces $\mathcal{R}(A)$ and $f(\mathbb{R}^n)$ have the same dimension (equal to $\operatorname{rank}(A)$), the map $f(x)$ represents an isomorphism.

</div>

#### Orthogonal projection

Now we derive an explicit formula for the projection of a vector $x$ into a subspace $U$. If we place the basis vectors of the subspace $U$ into the columns of a matrix $A$, then the projection of the vector $x$ into $U$ can be formulated as the projection of $x$ into $\mathcal{S}(A)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 8.49 — Orthogonal projection in $\mathbb{R}^m$)</span></p>

*Let $A \in \mathbb{R}^{m \times n}$ have rank $n$. Then the projection of a vector $x \in \mathbb{R}^m$ into the column space $\mathcal{S}(A)$ is $x' = A(A^\top A)^{-1} A^\top x$.*

*Proof.* First we observe that $x'$ is well-defined. The matrix $A^\top A$ has rank $n$ (Corollary 8.47(3)), so it is nonsingular and has an inverse. By Proposition 8.41 it now suffices to show that $x' \in \mathcal{S}(A)$ and $x - x' \in \mathcal{S}(A)^\perp$. The first property holds since $x' = Az$ for $z = (A^\top A)^{-1} A^\top x$. For the second property it suffices to verify that $x - x' \in \mathcal{S}(A)^\perp = \mathcal{R}(A^\top)^\perp = \operatorname{Ker}(A^\top)$, and this follows from the computation

$$A^\top(x - x') = A^\top(x - A(A^\top A)^{-1} A^\top x) = A^\top x - A^\top A(A^\top A)^{-1} A^\top x = A^\top x - A^\top x = o.$$

</div>

Note that the projection is a linear map and by the previous theorem $P := A(A^\top A)^{-1} A^\top$ is its matrix (with respect to the canonical basis). Moreover, this matrix has several special properties:

- The matrix $P$ is symmetric.
- $P^2 = P$. The projection of a vector $x$ is the vector $Px$. The vector $Px$ already belongs to the subspace $\mathcal{S}(A)$, and therefore its projection is itself: $P^2 x = Px$.
- Since $P$ represents the projection into $\mathcal{S}(A)$, we have $\mathcal{S}(P) = \mathcal{S}(A)$. The rank of the matrix $P$ is therefore equal to the dimension of the space into which we project, i.e., $\operatorname{rank}(P) = \operatorname{rank}(A)$. The matrix $P$ is thus nonsingular only when $m = n$, i.e., $\mathcal{S}(A) = \mathbb{R}^n$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 8.50)</span></p>

*A matrix $P \in \mathbb{R}^{n \times n}$ is a projection matrix if and only if it is symmetric and $P = P^2$.*

*Proof.* One direction has already been observed. Now assume that $P$ is symmetric and satisfies $P = P^2$, and we want to show that it is the projection matrix onto the space $\mathcal{S}(A)$. In other words, we want to show that for every vector $x \in \mathbb{R}^n$, $Px$ is its projection into $\mathcal{S}(A)$. By Proposition 8.41 it suffices to show that $x - Px \in \mathcal{S}(A)^\perp$. That is, $x - Px$ must be orthogonal to all vectors from $\mathcal{S}(A)$, and these have the form $Py$, where $y \in \mathbb{R}^n$. But this is easily verified by expanding their inner product

$$((I_n - P)x)^\top Py = x^\top(I_n - P)^\top Py = x^\top(P - P^2)y = 0.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.51)</span></p>

Consider the problem from Example 8.37: computing the projection $x_U$ of the vector $x = (1, 2, 4, 5)^\top$ into the subspace $U$ generated by the vectors $x_1 = (1, 0, 1, 0)^\top$, $x_2 = (1, 1, 1, 1)^\top$, $x_3 = (1, 0, 0, 1)^\top$.

Since we are working with the standard inner product, we can compute the projection alternatively using Theorem 8.49. We form the matrix

$$A = \begin{pmatrix} 1 & 1 & 1 \\ 0 & 1 & 0 \\ 1 & 1 & 0 \\ 0 & 1 & 1 \end{pmatrix},$$

whose columns are formed by $x_1, x_2, x_3$, and the projection is computed by the formula

$$x_U = A(A^\top A)^{-1} A^\top x = \frac{1}{2}(5, 7, 5, 7)^\top.$$

Here moreover

$$P = A(A^\top A)^{-1} A^\top = \frac{1}{4} \begin{pmatrix} 3 & -1 & 1 & 1 \\ -1 & 3 & 1 & 1 \\ 1 & 1 & 3 & -1 \\ 1 & 1 & -1 & 3 \end{pmatrix}$$

represents the projection matrix as a linear map into the subspace $U$. So if we have it expressed explicitly like this, the projection $x_U$ of a vector $x$ is easily computed as $x_U = Px$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 8.52 — Projection with an orthonormal basis)</span></p>

Denote by $z_1, \ldots, z_n$ the columns of the matrix $A \in \mathbb{R}^{m \times n}$ and assume they form an orthonormal system. Then $(A^\top A)_{ij} = \langle z_i, z_j \rangle$ and therefore $A^\top A = I_n$. The projection matrix $P$ into the column space $\mathcal{S}(A)$ takes the simpler form $P = A(A^\top A)^{-1} A^\top = AA^\top$. Here we can notice a parallel with the projection formula (8.2), because the projection of a vector $x \in \mathbb{R}^n$ is

$$Px = AA^\top x = \begin{pmatrix} | & | & & | \\ z_1 & z_2 & \cdots & z_n \\ | & | & & | \end{pmatrix} \begin{pmatrix} z_1^\top x \\ z_2^\top x \\ \vdots \\ z_n^\top x \end{pmatrix} = \sum_{i=1}^{n} (z_i^\top x) z_i.$$

Recapitulation: Let the matrix $A$ have orthonormal columns. Then $AA^\top$ represents the projection matrix into $\mathcal{S}(A)$ and in general $AA^\top \neq I_m$, but the product in the opposite order gives $A^\top A = I_n$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.53 — Projection onto a line, revisited)</span></p>

In particular, the projection matrix onto a one-dimensional subspace (line) has the form $P = a(a^\top a)^{-1} a^\top$, where $a \in \mathbb{R}^n$ is the direction of the line. The projection of a vector $x$ onto the line is then the vector $Px = a(a^\top a)^{-1} a^\top x = \frac{a^\top x}{a^\top a} a$ (cf. Example 8.38). If moreover we normalize the direction so that $\|a\|_2 = 1$, then $a^\top a = 1$ and therefore the projection matrix takes the simple form $P = aa^\top$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 8.54 — Orthogonal projection into the complement)</span></p>

*Let $P \in \mathbb{R}^{n \times n}$ be the projection matrix into a subspace $V \subseteq \mathbb{R}^n$. Then $I - P$ is the projection matrix into $V^\perp$.*

*Proof.* By Theorem 8.33, every vector $x \in \mathbb{R}^n$ can be uniquely decomposed into the sum $x = y + z$, where $y \in V$ and $z \in V^\perp$. From the perspective of Theorem 8.36, $y$ is the projection of $x$ into $V$ and $z$ is the projection of $x$ into $V^\perp$. Thus $z = x - y = x - Px = (I - P)x$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.55 — Projection matrix into $\operatorname{Ker}(A)$)</span></p>

Theorem 8.54 allows us to elegantly express the projection into the kernel of a matrix $A \in \mathbb{R}^{m \times n}$. Assume that $\operatorname{rank}(A) = m$. Since $\operatorname{Ker}(A)^\perp = \mathcal{R}(A) = \mathcal{S}(A^\top)$, the projection matrix into $\operatorname{Ker}(A)$ is given by $I - A^\top(AA^\top)^{-1} A$, where $A^\top(AA^\top)^{-1} A$ is the projection matrix into $\mathcal{S}(A^\top)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 8.56 — Distances between subspaces)</span></p>

One of the elegant uses of projections in geometry is computing distances between affine subspaces — distance from a point to a line, distance from a point to a plane, distance between two lines, distance from a point to a plane, etc. By the distance between two affine subspaces $U + a$, $V + b$ we mean the smallest distance $\|x - y\|$, where $x \in U + a$, $y \in V + b$. Without proof we state that the minimum distance is always attained.

The universal procedure is the following. Let $U + a$, $V + b$ be two affine subspaces of $\mathbb{R}^n$, where $U = \operatorname{span}\lbrace u_1, \ldots, u_m \rbrace$ and $V = \operatorname{span}\lbrace v_1, \ldots, v_n \rbrace$. Let the minimum distance be attained for points $x \in U + a$, $y \in V + b$; these points can be expressed as $x = a + \sum_{i=1}^{m} \alpha_i u_i$, $y = b + \sum_{j=1}^{n} \beta_j v_j$. The distance between these two points is the same as the distance from the point $a$ to the point $b + \sum_{j=1}^{n} \beta_j v_j - \sum_{i=1}^{m} \alpha_i u_i$. So the desired distance can equivalently be expressed as the distance from the point $a - b$ to the affine subspace $U + V$. By shifting in the direction $-b$, the distance is then computed as the distance from the point $a - b$ to the subspace $U + V = \operatorname{span}\lbrace u_1, \ldots, u_m, v_1, \ldots, v_n \rbrace$. This is already a standard problem, which we solve using Theorem 8.36 or Theorem 8.49 as the distance from the point $a - b$ to its projection into the subspace $U + V$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 8.57 — Computational complexity)</span></p>

What is the complexity of computing the projection matrix $A(A^\top A)^{-1} A^\top$ for $A \in \mathbb{R}^{m \times n}$? By Remarks 3.24 and 3.45, the computation of the matrix $A^\top A$ costs on the order of $2mn^2$ operations, its inversion $3n^3$ operations, and the remaining two matrix products $2n^2 m + 2nm^2$ operations. The total asymptotic complexity is then $3n^3 + 4n^2 m + 2nm^2$.

If we are only interested in the projection of a vector $x \in \mathbb{R}^m$, the expression $A(A^\top A)^{-1} A^\top x$ can be evaluated more efficiently by parenthesizing as $A\bigl((A^\top A)^{-1} (A^\top x)\bigr)$. Computing the matrix $(A^\top A)^{-1}$ again has complexity on the order of $2mn^2 + 3n^3$, but for the product $A^\top x$ we get only $2mn$, and for the rest $2n^2 + 2nm$. In total we have on the order of $2mn^2 + 2nm + 3n^3$ operations, which is significantly less than for the projection matrix, especially when $m$ is much larger than $n$. However, if we only want to compute the projection of a single vector, the Gram–Schmidt orthogonalization is computationally slightly more efficient, which by Remark 8.25 costs asymptotically only $2mn^2$ operations. The projection itself, given the orthonormal basis, does not asymptotically increase the complexity.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 8.58 — Projection and the Gram matrix)</span></p>

The formula for the projection can also be derived from Theorem 8.44 about the Gram matrix. If we denote by $w_1, \ldots, w_n$ the basis of $\mathcal{S}(A)$ given in the columns of the matrix $A \in \mathbb{R}^{m \times n}$, then $\langle w_i, w_j \rangle = (A^\top A)_{ij}$. The Gram matrix is now $A^\top A$ and equation (8.4) has the form $A^\top As = A^\top x$. From the equation we express $s = (A^\top A)^{-1} A^\top x$, which is the coordinate vector of the desired projection $x'$. Thus

$$x' = \sum_{i=1}^{n} s_i w_i = As = A(A^\top A)^{-1} A^\top x.$$

</div>

### 8.5 Least squares method

The least squares method illustrates another application of the projection theorem. Consider a system $Ax = b$ that has no solution (typically when $m$ is much larger than $n$). In that case we would like some good approximation, i.e., such a vector $x$ that the left and right sides are as close as possible. Formally,

$$\min_{x \in \mathbb{R}^n} \|Ax - b\|.$$

This approach is studied for various norms, but for the Euclidean norm we get

$$\min_{x \in \mathbb{R}^n} \|Ax - b\|_2^2 = \min_{x \in \mathbb{R}^n} \sum_{j=1}^{n} (A_{j*} x - b_j)^2.$$

Hence the name *least squares method*. Using the projection theorem, we find the solution easily. The following theorem states that least squares solutions are simultaneously solutions of the system of equations

$$A^\top Ax = A^\top b. \tag{8.5}$$

This system is called the *system of normal equations*. Interestingly, we obtain this system from the original system $Ax = b$ simply by multiplying by the matrix $A^\top$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 8.59 — Set of least squares solutions)</span></p>

*Let $A \in \mathbb{R}^{m \times n}$. Then the set of approximate solutions of the system $Ax = b$ by the least squares method is nonempty and equals the set of solutions of the normal equations (8.5).*

*Proof.* We are essentially looking for the projection of the vector $b$ into the subspace $\mathcal{S}(A)$, and this projection is a vector of the form $Ax$, where $x \in \mathbb{R}^n$. By Proposition 8.41, $Ax$ is the projection if and only if $Ax - b \in \mathcal{S}(A)^\perp = \operatorname{Ker}(A^\top)$. In other words, $A^\top(Ax - b) = 0$ must hold, i.e., $A^\top Ax = A^\top b$. This system has a solution because the projection must exist.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 8.60)</span></p>

*Let $A \in \mathbb{R}^{m \times n}$ have rank $n$. Then the approximate solution of the system $Ax = b$ by the least squares method is $x^* = (A^\top A)^{-1} A^\top b$, and it is unique.*

</div>

If the matrix $A$ is nonsingular, then the solution of the system $Ax = b$ is $x = A^{-1} b$. If the matrix $A$ is rectangular with linearly independent columns, then the least squares solution of the system $Ax = b$ is $x = (A^\top A)^{-1} A^\top b$. The matrix $(A^\top A)^{-1} A^\top$ can be viewed as a generalized inverse matrix (see Section 13.6 for more). Indeed, if we multiply it by the matrix $A$ from the right, we get $(A^\top A)^{-1} A^\top A = I_n$. In the opposite order this property does not hold, so $(A^\top A)^{-1} A^\top$ represents only a so-called left inverse of $A$.

The least squares method has applications in many fields, especially in statistics for linear regression. It studies the behavior and estimates the future development of various quantities, e.g., global temperature, GDP, stock or oil prices over time.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.61 — Linear regression: world population growth)</span></p>

The data on world population growth are as follows:

| year | 1950 | 1960 | 1970 | 1980 | 1990 | 2000 |
| --- | --- | --- | --- | --- | --- | --- |
| population (bn.) | 2.519 | 2.982 | 3.692 | 4.435 | 5.263 | 6.070 |

We want to find the dependence of the population size on time. Let us assume that the dependence is linear. We describe the linear relationship by a line $y = px + q$, where $x$ is time and $y$ is the population size. After substituting the data into the equations, the parameters $p, q$ should satisfy the conditions

$$2.519 = p \cdot 1950 + q, \quad \ldots, \quad 6.070 = p \cdot 2000 + q.$$

An exact solution does not exist, but the least squares solution is $p^* = 0.0724$, $q^* = -138.84$.

The resulting dependence can be used for predictions for the following years. The estimate for the year 2010 is 6.6943 billion inhabitants, while the actual number was 6.853 billion. However, note that it only makes sense to create short-term estimates — in 1900 the population size was certainly not negative.

</div>

### 8.6 Orthogonal matrices

Consider a linear map on the space $\mathbb{R}^n$. What must this map (and consequently its matrix) be like so that it does not deform geometric objects in any way? Rotation around an axis or reflection across a hyperplane are examples of such maps. We will show that this property is related to so-called orthogonal matrices. However, they have far-reaching significance. Because they have good numerical properties (see Sections 1.3 and 3.5), we encounter them frequently in various numerical algorithms.

In this section we consider the standard inner product in $\mathbb{R}^n$ resp. $\mathbb{C}^n$ and the Euclidean norm.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 8.62 — Orthogonal and unitary matrix)</span></p>

A matrix $Q \in \mathbb{R}^{n \times n}$ is *orthogonal* if $Q^\top Q = I_n$. A matrix $Q \in \mathbb{C}^{n \times n}$ is *unitary* if $\overline{Q}^\top Q = I_n$.

</div>

The concept of a unitary matrix is a generalization of orthogonal matrices to complex numbers. From now on, however, we will mostly work only with orthogonal matrices.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 8.63 — Characterization of orthogonal matrices)</span></p>

*Let $Q \in \mathbb{R}^{n \times n}$. Then the following are equivalent:*

1. *$Q$ is orthogonal,*
2. *$Q$ is nonsingular and $Q^{-1} = Q^\top$,*
3. *$QQ^\top = I_n$,*
4. *$Q^\top$ is orthogonal,*
5. *$Q^{-1}$ exists and is orthogonal,*
6. *the columns of $Q$ form an orthonormal basis of $\mathbb{R}^n$,*
7. *the rows of $Q$ form an orthonormal basis of $\mathbb{R}^n$.*

*Proof.* Briefly. (1)--(5) If $Q$ is orthogonal, then $Q^\top Q = I$ and thus $Q^{-1} = Q^\top$; similarly in the other direction. By the property of the inverse we also have $QQ^\top = I$, i.e., $(Q^\top)^\top Q^\top = I$, so $Q^\top$ is orthogonal. (6): From the equality $Q^\top Q = I$, by comparing entries at position $i, j$, we get $\langle Q_{*i}, Q_{*j} \rangle = 1$ if $i = j$ and $\langle Q_{*i}, Q_{*j} \rangle = 0$ if $i \neq j$. Thus the columns of $Q$ form an orthonormal system. Analogously in the other direction.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 8.64 — Product of orthogonal matrices)</span></p>

*If $Q_1, Q_2 \in \mathbb{R}^{n \times n}$ are orthogonal, then $Q_1 Q_2$ is orthogonal.*

*Proof.* $(Q_1 Q_2)^\top Q_1 Q_2 = Q_2^\top Q_1^\top Q_1 Q_2 = Q_2^\top Q_2 = I_n$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 8.65 — Examples of orthogonal matrices)</span></p>

- The identity matrix $I_n$, or its negative $-I_n$.
- *Householder matrix*: $H(a) := I_n - \frac{2}{a^\top a} a a^\top$, where $o \neq a \in \mathbb{R}^n$. Its geometric meaning is the following. Let $x'$ be the projection of the point $x$ onto the line $\operatorname{span}\lbrace a \rbrace$, and consider the linear map that rotates the point $x$ about the line $\operatorname{span}\lbrace a \rbrace$ by an angle of 180°. Using Theorem 8.49 on projection, we obtain that the point $x$ is mapped to the vector

  $$x + 2(x' - x) = 2x' - x = 2a(a^\top a)^{-1} a^\top x - x = \left(2 \frac{aa^\top}{a^\top a} - I\right) x.$$

  Thus the rotation matrix is $\frac{2}{a^\top a} aa^\top - I_n$. Now consider reflection across the hyperplane with normal $a$. This can be represented as a 180° rotation about $a$, followed by a reflection through the origin. Thus the matrix of this map is $I_n - \frac{2}{a^\top a} aa^\top = H(a)$.

  Moreover, it can be shown that every orthogonal matrix of order $n$ can be decomposed as a product of at most $n$ suitable Householder matrices. Therefore, a linear map with an orthogonal matrix geometrically represents the composition of at most $n$ reflections.

- *Givens matrix*: For $n = 2$ it is the rotation matrix by an angle $\alpha$ counterclockwise

  $$\begin{pmatrix} \cos \alpha & -\sin \alpha \\ \sin \alpha & \cos \alpha \end{pmatrix}.$$

  It is thus a matrix of the form $\binom{c\ {-s}}{s\ c}$, where $c^2 + s^2 = 1$ and every such matrix corresponds to some rotation matrix. In general, for dimension $n$ it is a matrix representing rotation by an angle $\alpha$ in the plane of axes $x_i, x_j$.

  Every orthogonal matrix can also be composed from Givens matrices, but up to $\binom{n}{2}$ of them are needed in the product, and possibly one additional diagonal matrix with $\pm 1$ on the diagonal. Geometrically this means that every linear map with an orthogonal matrix represents the composition of at most $\binom{n}{2}$ simple rotations and possibly one reflection along the coordinate axes.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 8.66 — Properties of orthogonal matrices)</span></p>

*Let $Q \in \mathbb{R}^{n \times n}$ be orthogonal. Then:*

1. *$\langle Qx, Qy \rangle = \langle x, y \rangle$ for every $x, y \in \mathbb{R}^n$,*
2. *$\|Qx\| = \|x\|$ for every $x \in \mathbb{R}^n$,*
3. *$|Q_{ij}| \le 1$ and $|Q_{ij}^{-1}| \le 1$ for every $i, j = 1, \ldots, n$,*
4. *$\begin{pmatrix} 1 & o^\top \\ o & Q \end{pmatrix}$ is an orthogonal matrix.*

*Proof.*

(1) $\langle Qx, Qy \rangle = (Qx)^\top Qy = x^\top Q^\top Qy = x^\top Iy = \langle x, y \rangle$.

(2) $\|Qx\| = \sqrt{\langle Qx, Qx \rangle} = \sqrt{\langle x, x \rangle} = \|x\|$.

(3) By property (6) of Proposition 8.63, $\|Q_{*j}\| = 1$ for every $j = 1, \ldots, n$. Thus $1 = \|Q_{*j}\|^2 = \sum_{i=1}^{n} q_{ij}^2$, from which $q_{ij}^2 \le 1$, and therefore $|q_{ij}| \le 1$. The matrix $Q^{-1}$ is orthogonal, so the claim holds for it as well.

(4) By definition $\begin{pmatrix} 1 & o^\top \\ o & Q \end{pmatrix}^\top \begin{pmatrix} 1 & o^\top \\ o & Q \end{pmatrix} = \begin{pmatrix} 1 & o^\top \\ o & Q^\top Q \end{pmatrix} = I_{n+1}$.

</div>

If we view $Q$ as the matrix of the corresponding linear map $x \mapsto Qx$, then property (1) of Theorem 8.66 says that angles are preserved under this map, and property (2) says that lengths are preserved. The converse also holds: the matrix of a map preserving the inner product must necessarily be orthogonal (cf. Theorem 8.68) and in fact the matrix of a map preserving the Euclidean norm must be orthogonal. Property (3) is valued in numerical mathematics because $Q$ and $Q^{-1}$ have bounded entry magnitudes. An important property for numerical computation is also (2), because when multiplying by an orthogonal matrix, the entries (and thus also rounding errors) do not tend to grow.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 8.67 — Orthogonal matrices and Fourier coefficients)</span></p>

Orthogonal matrices give a slightly different perspective on the Fourier coefficients from Theorem 8.22. Let $z_1, \ldots, z_n$ be a basis of the space $\mathbb{R}^n$ and let $v \in \mathbb{R}^n$. The coordinates of the vector $v$ with respect to the given basis are given by the relation $v = \sum_{i=1}^{n} x_i z_i$. The coordinates are thus the solution of the system $Qx = v$, where the columns of the matrix $Q$ are formed by the basis vectors, i.e., $Q_{*i} = z_i$ for $i = 1, \ldots, n$. If the basis is orthonormal, the matrix $Q$ is orthogonal and we can simply write

$$x = Q^{-1} v = Q^\top v = \begin{pmatrix} z_1^\top \\ z_2^\top \\ \vdots \\ z_n^\top \end{pmatrix} v = \begin{pmatrix} z_1^\top v \\ z_2^\top v \\ \vdots \\ z_n^\top v \end{pmatrix}.$$

Once again we obtain that the $i$-th coordinate $x_i$ of the vector $v$ has the value $\langle z_i, v \rangle = z_i^\top v$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 8.68 — Orthogonal matrices and linear maps)</span></p>

*Let $U, V$ be spaces over $\mathbb{R}$ with an arbitrary inner product and $f \colon U \to V$ a linear map. Let $B_U$ resp. $B_V$ be an orthonormal basis of $U$ resp. $V$. Then the matrix of the map ${}_{B_V}[f]\_{B_U}$ is orthogonal if and only if $\langle f(x), f(y) \rangle = \langle x, y \rangle$ for every $x, y \in U$.*

*Proof.* By Proposition 8.29 and the properties of the matrix of a map,

$$\langle x, y \rangle = [x]\_{B_U}^\top \cdot [y]\_{B_U},$$

$$\langle f(x), f(y) \rangle = [f(x)]\_{B_V}^\top \cdot [f(y)]\_{B_V} = \left({}_{B_V}[f]\_{B_U} \cdot [x]\_{B_U}\right)^\top \cdot {}_{B_V}[f]\_{B_U} \cdot [y]\_{B_U}.$$

Therefore, if ${}_{B_V}[f]\_{B_U}$ is orthogonal, then the equality $\langle f(x), f(y) \rangle = \langle x, y \rangle$ holds for every $x, y \in U$, since the coordinates are unit vectors. Conversely, if the equality $\langle f(x), f(y) \rangle = \langle x, y \rangle$ holds for every $x, y \in U$, then substituting for $x$ and $y$ specifically the $i$-th and $j$-th vector of the basis $B_U$, we have $[x]\_{B_U} = e_i$, $[y]\_{B_U} = e_j$, and therefore

$$(I_n)_{ij} = e_i^\top e_j = [x]\_{B_U}^\top [y]\_{B_U} = \langle x, y \rangle = \langle f(x), f(y) \rangle = e_i^\top \cdot {}_{B_V}[f]\_{B_U}^\top \cdot {}_{B_V}[f]\_{B_U} \cdot e_j = \left({}_{B_V}[f]\_{B_U}^\top \cdot {}_{B_V}[f]\_{B_U}\right)\_{ij}.$$

Thus, entry by entry, we obtain the equality $I_n = {}_{B_V}[f]\_{B_U}^\top \cdot {}_{B_V}[f]\_{B_U}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 8.69 — Orthogonal matrices and change-of-basis matrices)</span></p>

*Let $V$ be a space over $\mathbb{R}$ with an arbitrary inner product and $B_1, B_2$ two of its bases. Any two of the following properties imply the third:*

1. *$B_1$ is an orthonormal basis,*
2. *$B_2$ is an orthonormal basis,*
3. *${}_{B_2}[id]\_{B_1}$ is an orthogonal matrix.*

*Proof.* Implication "(1), (2) $\Rightarrow$ (3)". Follows from Theorem 8.68, since the identity preserves the inner product. Implication "(2), (3) $\Rightarrow$ (1)". Let $B_1 = \lbrace x_1, \ldots, x_n \rbrace$. By definition, the columns of ${}_{B_2}[id]\_{B_1}$ are formed by the vectors $[x_i]\_{B_2}$, which are (due to the orthogonality of the change-of-basis matrix) orthonormal with respect to the standard inner product in $\mathbb{R}^n$. By Proposition 8.29, $\langle x_i, x_j \rangle = [x_i]\_{B_2}^\top [x_j]\_{B_2}$, which equals 1 for $i = j$ and 0 otherwise. Implication "(3), (1) $\Rightarrow$ (2)". Follows from the previous by symmetry, since ${}_{B_1}[id]\_{B_2} = {}_{B_2}[id]\_{B_1}^{-1}$.

</div>

### Summary of Chapter 8

The inner product introduces a special product of two vectors, where the result is a scalar. If a vector space is equipped with an inner product, then this inner product naturally also defines a norm on the space, i.e., the magnitude of a vector. And the norm then defines the distance of vectors as the norm of their difference. We need both concepts in order to be able to measure in the space, but also, for example, to express that a sequence of vectors converges.

The inner product further naturally introduces orthogonality of vectors. An orthonormal basis is a basis consisting of vectors of magnitude 1 that are mutually orthogonal. With such a basis, coordinates, projections, etc. are then easy to compute. We can construct an orthonormal basis using the Gram–Schmidt orthogonalization method. Although we defined the concept of inner product abstractly, it turned out that every inner product has the form of the standard inner product in the coordinate system of (any) orthonormal basis.

Orthogonal projection is a map that sends a vector to the nearest vector in a given subspace. The line from the vector to its projection must be perpendicular to the subspace (hence "orthogonal" projection). The projection is easily computed if we know an orthonormal basis of the subspace. Otherwise we use a matrix formula. Projection is a very useful tool not only in geometry, where it allows us to elegantly express distances between various objects. As a non-geometric application we presented the least squares method, which computes the best approximate solution of an overdetermined system of equations.

Algebraically, orthogonal matrices are those matrices whose inverse is simply expressed as the transpose. Orthogonal matrices then geometrically represent linear maps that do not deform objects — they preserve angles and distances. These maps can always be expressed as the composition of finitely many rotations and reflections. The geometric essence is also reflected in numerical properties — computation with orthogonal matrices is advantageous because rounding errors are not amplified as much.

---

## Chapter 9 — Determinants

Determinants were developed for the purpose of solving square systems of linear equations and provide an explicit formula for their solution (see Theorem 9.15). Gottfried Wilhelm Leibniz is considered the author of the determinant, and independently of him, the Japanese mathematician Seki Kōwa discovered it in the same year 1683. The term "determinant" itself comes from Gauss (*Disquisitiones arithmeticae*, 1801). However, it turned out that the determinant itself is a certain characteristic of a square matrix with a number of different applications.

Recall that $S_n$ denotes the set of all permutations on the set $\lbrace 1, \ldots, n \rbrace$, see Section 4.2.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 9.1 — Determinant)</span></p>

Let $A \in \mathbb{T}^{n \times n}$. Then the *determinant* of matrix $A$ is the number

$$\det(A) = \sum_{p \in S_n} \operatorname{sgn}(p) \prod_{i=1}^{n} a_{i, p(i)} = \sum_{p \in S_n} \operatorname{sgn}(p) \, a_{1, p(1)} \ldots a_{n, p(n)}.$$

Notation: $\det(A)$ or $|A|$.

</div>

What does the formula from the definition of the determinant actually say? Each summand has the form $\operatorname{sgn}(p) \, a_{1,p(1)} \ldots a_{n,p(n)}$, which corresponds to selecting $n$ elements from the matrix $A$ such that exactly one element is chosen from each row and each column. These elements are then multiplied together, and the summand is assigned a positive or negative sign depending on the sign of the permutation that determined these elements.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 9.2 — Determinant of a matrix of order 2 and 3)</span></p>

A matrix of order 2 has determinant

$$\det \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix} = a_{11} a_{22} - a_{21} a_{12}.$$

A matrix of order 3 has determinant

$$\det \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix} = a_{11} a_{22} a_{33} + a_{21} a_{32} a_{13} + a_{31} a_{12} a_{23} - a_{31} a_{22} a_{13} - a_{11} a_{32} a_{23} - a_{21} a_{12} a_{33}.$$

</div>

Computing determinants from the definition for larger matrices is generally very inefficient, because it requires processing $n!$ summands. The computation is simpler only for special matrices. Such a matrix is, for example, an upper triangular matrix, i.e., a matrix $A \in \mathbb{T}^{n \times n}$ for which $a_{ij} = 0$ for $i > j$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 9.3 — Determinant of a triangular matrix)</span></p>

*Let $A \in \mathbb{T}^{n \times n}$ be an upper triangular matrix. Then $\det(A) = a_{1,p(1)} \ldots a_{n,p(n)}$.*

*Proof.* Since the matrix $A$ is upper triangular, the factor $a_{n,p(n)}$ is nonzero only if $p(n) = n$. For the factor $a_{n-1,p(n-1)}$ to be nonzero, either $p(n-1) = n$ or $p(n-1) = n-1$. The first option is excluded since $p(n) = n$, so $p(n-1) = n-1$. By repeating this procedure, we conclude that the term (9.1) is nonzero only for the identity permutation. Therefore $\det(A) = a_{1,1} \ldots a_{n,n}$, i.e., the determinant equals the product of the diagonal entries. As a consequence, in particular $\det(I_n) = 1$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 9.4 — Determinant of the transpose)</span></p>

*Let $A \in \mathbb{T}^{n \times n}$. Then $\det(A^\top) = \det(A)$.*

*Proof.*

$$\det(A^\top) = \sum_{p \in S_n} \operatorname{sgn}(p) \prod_{i=1}^{n} A^\top_{i,p(i)} = \sum_{p \in S_n} \operatorname{sgn}(p) \prod_{i=1}^{n} a_{p(i),i} = \sum_{p \in S_n} \operatorname{sgn}(p^{-1}) \prod_{i=1}^{n} a_{i,p^{-1}(i)} = \sum_{q \in S_n} \operatorname{sgn}(q) \prod_{i=1}^{n} a_{i,q(i)} = \det(A).$$

</div>

For determinants, in general $\det(A + B) \neq \det(A) + \det(B)$, nor is there a known simple formula for the determinant of a sum of matrices. The exception is the following special case of row linearity.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 9.5 — Row linearity of the determinant)</span></p>

*Let $A \in \mathbb{T}^{n \times n}$, $b \in \mathbb{T}^n$. Then for any $i = 1, \ldots, n$ it holds that:*

$$\det(A + e_i b^\top) = \det(A) + \det(A + e_i(b^\top - A_{i*})).$$

*In other words,*

$$\det \begin{pmatrix} A_{1*} \\ \vdots \\ a_{i1} + b_1 & \ldots & a_{in} + b_n \\ \vdots \\ A_{n*} \end{pmatrix} = \det \begin{pmatrix} A_{1*} \\ \vdots \\ a_{i1} & \ldots & a_{in} \\ \vdots \\ A_{n*} \end{pmatrix} + \det \begin{pmatrix} A_{1*} \\ \vdots \\ b_1 & \ldots & b_n \\ \vdots \\ A_{n*} \end{pmatrix}.$$

</div>

By virtue of Proposition 9.4, the determinant is not only row-linear but also column-linear.

### 9.1 Determinant and elementary operations

Our plan is to use Gaussian elimination to compute the determinant. For this, we first need to be able to compute the determinant of a matrix in row echelon form, and to know how the value of the determinant is affected by elementary row operations. The first question has a simple answer, because a matrix in row echelon form is also upper triangular, and therefore its determinant equals the product of the diagonal entries. The second question is answered by analyzing the individual elementary operations. Let the matrix $A'$ be obtained from $A$ by some elementary operation:

1. **Multiplying the $i$-th row by a scalar $\alpha \in \mathbb{T}$:** $\det(A') = \alpha \det(A)$.

2. **Swapping the $i$-th and $j$-th rows:** $\det(A') = -\det(A)$.

3. **Adding $\alpha$ times the $j$-th row to the $i$-th row, where $i \neq j$:** $\det(A') = \det(A)$.

*Proofs.*

(1) Follows directly from the definition — we factor out $\alpha$ from the $i$-th row.

(2) Denote the transposition $t = (i, j)$. Then $\det(A') = \sum_{p \in S_n} \operatorname{sgn}(p) a'_{1,p(1)} \ldots a'_{n,p(n)}$, where $a'_{i,p(i)} = a_{j,p(i)}$, $a'_{j,p(j)} = a_{i,p(j)}$, etc. By the substitution $q = p \circ t$ we obtain $\det(A') = -\det(A)$.

(3) From the row linearity of the determinant and Corollary 9.6 (a matrix with two identical rows has zero determinant), we get $\det(A') = \det(A) + \alpha \cdot 0 = \det(A)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 9.6)</span></p>

*If the matrix $A \in \mathbb{T}^{n \times n}$ has two identical rows, then $\det(A) = 0$.*

*Proof (for fields of characteristic $\neq 2$).* By swapping these two identical rows, we get $\det(A) = -\det(A)$, and therefore $\det(A) = 0$.

*Proof (for general fields).* For $\mathbb{Z}_2$ we have $1 = -1$, and therefore we must proceed differently. Define the transposition $t := (i, j)$, where $i, j$ are the indices of the identical rows. Let $S'_n$ be the set of even permutations in $S_n$. Then $S_n$ can be decomposed disjointly into $S'_n$ and $\lbrace p \circ t;\, p \in S'_n \rbrace$. Therefore

$$\det(A) = \sum_{p \in S'_n} \operatorname{sgn}(p) \prod_{i=1}^{n} a_{i,p(i)} + \sum_{p \in S'_n} \operatorname{sgn}(p \circ t) \prod_{i=1}^{n} a_{i,p \circ t(i)} = \sum_{p \in S'_n} \operatorname{sgn}(p) \prod_{i=1}^{n} a_{i,p(i)} - \sum_{p \in S'_n} \operatorname{sgn}(p) \prod_{i=1}^{n} a_{i,p(i)} = 0.$$

</div>

The above observations have several consequences: For any matrix $A \in \mathbb{T}^{n \times n}$, we have $\det(\alpha A) = \alpha^n \det(A)$. Furthermore, if $A$ contains a zero row or column, then $\det(A) = 0$.

The main significance of the effect of elementary operations on the determinant is that we can compute determinants using Gaussian elimination:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Algorithm 9.7 — Computing the determinant via REF)</span></p>

Reduce matrix $A$ to row echelon form $A'$ and keep track of the changes to the determinant in a coefficient $c$; then $\det(A)$ equals the product of $c^{-1}$ and the diagonal entries of the matrix $A'$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 9.8 — Computing the determinant using elementary row operations)</span></p>

$$|A| = \begin{vmatrix} 1 & 2 & 3 & 4 \\ 1 & 2 & 1 & 3 \\ 2 & 5 & 5 & 5 \\ 0 & 2 & -3 & -4 \end{vmatrix} = \begin{vmatrix} 1 & 2 & 3 & 4 \\ 0 & 0 & -2 & -1 \\ 0 & 1 & -1 & -3 \\ 0 & 2 & -3 & -4 \end{vmatrix} = -\begin{vmatrix} 1 & 2 & 3 & 4 \\ 0 & 1 & -1 & -3 \\ 0 & 0 & -2 & -1 \\ 0 & 0 & -1 & 2 \end{vmatrix}$$

$$= -\begin{vmatrix} 1 & 2 & 3 & 4 \\ 0 & 1 & -1 & -3 \\ 0 & 0 & -2 & -1 \\ 0 & 0 & -1 & 2 \end{vmatrix} = 2 \begin{vmatrix} 1 & 2 & 3 & 4 \\ 0 & 1 & -1 & -3 \\ 0 & 0 & 1 & 0.5 \\ 0 & 0 & -1 & 2 \end{vmatrix} = 2 \begin{vmatrix} 1 & 2 & 3 & 4 \\ 0 & 1 & -1 & -3 \\ 0 & 0 & 1 & 0.5 \\ 0 & 0 & 0 & 2.5 \end{vmatrix} = 5.$$

</div>

### 9.2 Further properties of the determinant

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 9.9 — Regularity criterion)</span></p>

*A matrix $A \in \mathbb{T}^{n \times n}$ is nonsingular if and only if $\det(A) \neq 0$.*

*Proof.* We reduce the matrix $A$ to row echelon form $A'$ using elementary operations; these may change the value of the determinant, but not its (non)zeroness. Then $A$ is nonsingular if and only if $A'$ has nonzero entries on the diagonal.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 9.10 — Measure of regularity)</span></p>

Theorem 9.9 allows us to introduce a certain measure of regularity. The closer $\det(A)$ is to 0, the closer the matrix $A$ is to some singular matrix. An example is the Hilbert matrix $H_n$ (see Example 3.51), which is ill-conditioned because it is "almost" singular. Indeed, as the table shows, the determinant of the matrix is very close to zero.

| $n$ | $\det(H_n)$ |
| --- | --- |
| 4 | $\approx 10^{-7}$ |
| 6 | $\approx 10^{-18}$ |
| 8 | $\approx 10^{-33}$ |
| 10 | $\approx 10^{-53}$ |

However, this measure is not ideal (a better one uses, e.g., eigenvalues or singular values, see Section 13.5), because it is very sensitive to scaling. Consider, for example, the matrix $0.1 I_n$, for which $\det(0.1 I_n) = 10^{-n}$. Although $10^{-n}$ can be an arbitrarily small number, the matrix itself is relatively far from being singular.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 9.11 — Multiplicativity of the determinant)</span></p>

*For every $A, B \in \mathbb{T}^{n \times n}$ it holds that $\det(AB) = \det(A) \det(B)$.*

*Proof.* (1) First consider the special case when $A$ is an elementary operation matrix:

1. $A = E_i(\alpha)$, multiplying the $i$-th row by the scalar $\alpha$. Then $\det(AB) = \alpha \det(B)$ and $\det(A) \det(B) = \alpha \det(B)$.
2. $A = E_{ij}$, swapping the $i$-th and $j$-th rows. Then $\det(AB) = -\det(B)$ and $\det(A) \det(B) = -1 \det(B)$.
3. $A = E_{ij}(\alpha)$, adding $\alpha$ times the $j$-th row to the $i$-th. Then $\det(AB) = \det(B)$ and $\det(A) \det(B) = 1 \det(B)$.

Thus the equality holds in all cases.

(2) Now consider the general case. If $A$ is singular, then $AB$ is also singular (Proposition 3.30), and therefore by Theorem 9.9 we have $\det(AB) = 0 = \det(A) \det(B)$. If $A$ is nonsingular, then it can be decomposed as a product of elementary matrices $A = E_1 \ldots E_k$. We now proceed by mathematical induction on $k$. The case $k = 1$ was handled in part (1). By the induction hypothesis and from part (1), we get

$$\det(AB) = \det(E_1(E_2 \ldots E_k B)) = \det(E_1) \det((E_2 \ldots E_k) B) = \det(E_1) \det(E_2 \ldots E_k) \det(B) = \det(A) \det(B).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 9.12)</span></p>

*Let $A \in \mathbb{T}^{n \times n}$ be nonsingular. Then $\det(A^{-1}) = \det(A)^{-1}$.*

*Proof.* $1 = \det(I_n) = \det(A A^{-1}) = \det(A) \det(A^{-1})$.

</div>

We now present a recursive formula for computing the determinant.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 9.13 — Laplace expansion along the $i$-th row)</span></p>

*Let $A \in \mathbb{T}^{n \times n}$, $n \ge 2$. Then for every $i = 1, \ldots, n$ it holds that*

$$\det(A) = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} \det(A^{ij}),$$

*where $A^{ij}$ is the matrix obtained from $A$ by deleting the $i$-th row and the $j$-th column.*

*Note.* Similarly to expanding along a row, we can expand along any column.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 9.14 — Laplace expansion along the 4th row)</span></p>

$$\begin{vmatrix} 1 & 2 & 3 & 4 \\ 1 & 2 & 1 & 2 \\ 2 & 5 & 5 & 5 \\ 0 & 2 & -4 & -4 \end{vmatrix} = (-1)^{4+1} \cdot 0 \cdot \begin{vmatrix} 2 & 3 & 4 \\ 2 & 1 & 2 \\ 5 & 5 & 5 \end{vmatrix} + (-1)^{4+2} \cdot 2 \cdot \begin{vmatrix} 1 & 3 & 4 \\ 1 & 1 & 2 \\ 2 & 5 & 5 \end{vmatrix}$$

$$+ (-1)^{4+3} \cdot (-4) \cdot \begin{vmatrix} 1 & 2 & 4 \\ 1 & 2 & 2 \\ 2 & 5 & 5 \end{vmatrix} + (-1)^{4+4} \cdot (-4) \cdot \begin{vmatrix} 1 & 2 & 3 \\ 1 & 2 & 1 \\ 2 & 5 & 5 \end{vmatrix}$$

$$= 0 + 2 \cdot 4 + 4 \cdot 2 - 4 \cdot 2 = 8.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 9.15 — Cramer's rule)</span></p>

*Let $A \in \mathbb{T}^{n \times n}$ be nonsingular, $b \in \mathbb{T}^n$. Then the solution of the system $Ax = b$ is given by the formula*

$$x_i = \frac{\det(A + (b - A_{*i}) e_i^\top)}{\det(A)}, \quad i = 1, \ldots, n.$$

*Proof.* Let $x$ be the solution of the system $Ax = b$; due to the nonsingularity of $A$, the solution exists and is unique. We expand the equation as $\sum_{j=1}^{n} A_{*j} x_j = b$. From the column linearity of the determinant, we get

$$\det(A + (b - A_{*i}) e_i^\top) = \det(A_{*1} | \ldots | b | \ldots | A_{*n}) = \det(A_{*1} | \ldots | \sum_{j=1}^{n} A_{*j} x_j | \ldots | A_{*n})$$

$$= \sum_{j=1}^{n} \det(A_{*1} | \ldots | A_{*j} | \ldots | A_{*n}) x_j = \det(A_{*1} | \ldots | A_{*i} | \ldots | A_{*n}) x_i = \det(A) x_i.$$

It now suffices to divide both sides by $\det(A) \neq 0$.

</div>

Cramer's rule from 1750 is named after the Swiss mathematician Gabriel Cramer. In its time, it was a popular tool for solving systems of linear equations. Today, it is no longer used for practical computations, because computing the solution of a system using $n + 1$ determinants is not very efficient in terms of computational time. Moreover, it has worse numerical properties. The significance of the determinant is rather theoretical; among other things, it shows and provides:

- An explicit expression for the solution of a system of linear equations.
- Continuity of the solution with respect to the entries of the matrix $A$ and the vector $b$. Formally, the mapping $(A, b) \mapsto A^{-1} b$ is continuous on the domain of nonsingular matrices $A$.
- An estimate of the description size of the solution from the description size of the input values.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 9.16 — Cramer's rule)</span></p>

The solution of the system of equations

$$\left(\begin{array}{ccc|c} 1 & 2 & 3 & 1 \\ 1 & 2 & 1 & 3 \\ 2 & 5 & 5 & 4 \end{array}\right)$$

is computed component by component

$$x_1 = \frac{\begin{vmatrix} 1 & 2 & 3 \\ 3 & 2 & 1 \\ 4 & 5 & 5 \end{vmatrix}}{\begin{vmatrix} 1 & 2 & 3 \\ 1 & 2 & 1 \\ 2 & 5 & 5 \end{vmatrix}} = \frac{4}{2} = 2, \quad x_2 = \frac{\begin{vmatrix} 1 & 1 & 3 \\ 1 & 3 & 1 \\ 2 & 4 & 5 \end{vmatrix}}{\begin{vmatrix} 1 & 2 & 3 \\ 1 & 2 & 1 \\ 2 & 5 & 5 \end{vmatrix}} = \frac{2}{2} = 1, \quad x_3 = \frac{\begin{vmatrix} 1 & 2 & 1 \\ 1 & 2 & 3 \\ 2 & 5 & 4 \end{vmatrix}}{\begin{vmatrix} 1 & 2 & 3 \\ 1 & 2 & 1 \\ 2 & 5 & 5 \end{vmatrix}} = \frac{-2}{2} = -1.$$

The solution is therefore the vector $x = (2, 1, -1)^\top$.

</div>

### 9.3 Adjugate matrix

The adjugate matrix is closely related to determinants and matrix inversion. We will use it when deriving the Cayley--Hamilton theorem (Theorem 10.20), but the reader may also encounter it, for example, in cryptography or when deriving the formula for the derivative of the determinant.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 9.17 — Adjugate matrix)</span></p>

Let $A \in \mathbb{T}^{n \times n}$ and $n \ge 2$. Then the *adjugate matrix* $\operatorname{adj}(A) \in \mathbb{T}^{n \times n}$ has entries

$$\operatorname{adj}(A)_{ij} = (-1)^{i+j} \det(A^{ji}), \quad i, j = 1, \ldots, n,$$

where $A^{ji}$ again denotes the matrix obtained from $A$ by deleting the $j$-th row and the $i$-th column.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 9.18 — On the adjugate matrix)</span></p>

*For every matrix $A \in \mathbb{T}^{n \times n}$ it holds that $A \operatorname{adj}(A) = \det(A) I_n$.*

*Proof.* We derive

$$(A \operatorname{adj}(A))_{ij} = \sum_{k=1}^{n} A_{ik} \operatorname{adj}(A)_{kj} = \sum_{k=1}^{n} A_{ik} (-1)^{k+j} \det(A^{jk}) = \begin{cases} \det(A), & \text{pro } i = j, \\ 0, & \text{pro } i \neq j. \end{cases}$$

The justification of the last equality is that for $i = j$, this is the Laplace expansion of $\det(A)$ along the $j$-th row. For $i \neq j$, this is the expansion along the $j$-th row of matrix $A$, but with the $j$-th row first replaced by the $i$-th row. This matrix will have two identical rows and hence zero determinant.

</div>

For a nonsingular matrix $A$, we have $\det(A) \neq 0$, and dividing by $\det(A)$ yields an explicit formula for the inverse matrix $A^{-1}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 9.19)</span></p>

*If $A \in \mathbb{T}^{n \times n}$ is nonsingular, then $A^{-1} = \frac{1}{\det(A)} \operatorname{adj}(A)$.*

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 9.20 — Adjugate matrix)</span></p>

Let

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 1 & 2 & 1 \\ 2 & 5 & 5 \end{pmatrix}.$$

Then:

$$\operatorname{adj}(A)_{12} = (-1)^{1+2} \begin{vmatrix} 2 & 3 \\ 5 & 5 \end{vmatrix} = 5, \quad \ldots$$

In total:

$$\operatorname{adj}(A) = \begin{pmatrix} 5 & 5 & -4 \\ -3 & -1 & 2 \\ 1 & -1 & 0 \end{pmatrix}.$$

Therefore:

$$A^{-1} = \frac{1}{\det(A)} \operatorname{adj}(A) = \frac{1}{2} \begin{pmatrix} 5 & 5 & -4 \\ -3 & -1 & 2 \\ 1 & -1 & 0 \end{pmatrix}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 9.21 — Derivative of the determinant)</span></p>

Consider the determinant as a function $\det(A) \colon \mathbb{R}^{n \times n} \to \mathbb{R}$. The problem now is to determine the partial derivative of $\det(A)$ with respect to $a_{ij}$ and to assemble the matrix of partial derivatives.

For this purpose, we start from the Laplace expansion $\det(A) = \sum_{k=1}^{n} (-1)^{i+k} a_{ik} \det(A^{ik})$ and simply derive

$$\frac{\partial \det(A)}{\partial a_{ij}} = (-1)^{i+j} \det(A^{ij}).$$

Therefore the matrix of partial derivatives is $\partial \det(A) = \operatorname{adj}(A)^\top$.

Using specifically the matrix from Example 9.20, we have $\det(A) = 2$. Since $\operatorname{adj}(A)_{33} = 0$, the determinant of matrix $A$ does not change when the entry $a_{33}$ is changed. On the other hand, a small increase in the entry $a_{11}$ causes the determinant to increase significantly (since $\operatorname{adj}(A)_{11} = 5$), and an increase in the entry $a_{13}$ causes the determinant to increase less significantly (since $\operatorname{adj}(A)_{31} = 1$).

</div>

### 9.4 Applications

The theorem on the adjugate matrix gives the following characterization of when the inverse matrix has integer entries.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 9.22)</span></p>

*Let $A \in \mathbb{Z}^{n \times n}$. Then $A^{-1}$ has integer entries if and only if $\det(A) = \pm 1$.*

*Proof.* Implication "$\Rightarrow$". We know $1 = \det(A) \det(A^{-1})$. If the matrices $A, A^{-1}$ are integer, then their determinants are also integers and therefore must equal $\pm 1$.

Implication "$\Leftarrow$". We know $A^{-1}_{ij} = \frac{1}{\det(A)} (-1)^{i+j} \det(A^{ji})$. This is an integer value provided that $\det(A) = \pm 1$ and $\det(A^{ji})$ is an integer.

</div>

Another example of the use of the determinant is in polynomials. The determinant from the following proposition is called the *resultant* and is used, for example, for solving nonlinear equations.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 9.23 — Resultant)</span></p>

*The polynomials $p(x) = a_n x^n + \ldots + a_1 x + a_0$, $q(x) = b_m x^m + \ldots + b_1 x + b_0$ have a common root if and only if*

$$\begin{vmatrix} a_n & a_{n-1} & \ldots & & a_0 & & \\ & a_n & a_{n-1} & \ldots & a_0 & & \\ & & \ddots & \ddots & & \ddots & \\ & & & a_n & a_{n-1} & \ldots & a_0 \\ b_m & b_{m-1} & \ldots & b_1 & b_0 & & \\ & \ddots & \ddots & & & \ddots & \\ & & b_m & b_{m-1} & \ldots & & b_1 & b_0 \end{vmatrix} = 0.$$

</div>

#### Geometric interpretation of the determinant

The determinant has a nice geometric meaning. If we consider the linear map $x \mapsto Ax$ with matrix $A \in \mathbb{R}^{n \times n}$, then geometric bodies change their volume under this map by the factor $|\det(A)|$. We do not formally define the notion of "volume" in the space $\mathbb{R}^n$ and rely on intuitive understanding. The volume of common geometric objects in $\mathbb{R}^1$ corresponds to lengths, in $\mathbb{R}^2$ corresponds to areas, and in $\mathbb{R}^3$ corresponds to volume in the usual sense.

We first consider the special case of a parallelepiped. A *parallelepiped* with linearly independent edges $a_1, \ldots, a_m$ is defined as the set $\lbrace x \in \mathbb{R}^n;\, x = \sum_{i=1}^{m} \alpha_i a_i, \, 0 \le \alpha_i \le 1 \rbrace$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 9.24 — Volume of a parallelepiped)</span></p>

*Let $A \in \mathbb{R}^{m \times n}$ and consider the parallelepiped with edges given by the rows of matrix $A$. Then its volume (as an $m$-dimensional object) is $\sqrt{\det(AA^\top)}$. In particular, for $m = n$ the volume is $|\det(A)|$.*

*Proof.* By mathematical induction on $m$. For $m = 1$ this is obvious; let us proceed to the induction step. Denote the $i$-th row of matrix $A$ as the vector $a_i^\top$ and define the matrix $D$ obtained from $A$ by removing the last row. Decompose $a_m = b_m + c_m$, where $c_m \in \mathcal{R}(D)$ and $b_m \in \mathcal{R}(D)^\perp$ according to Remark 8.40. The rows of matrix $D$ generate a parallelepiped of lower dimension, which forms the base of the overall parallelepiped. By the induction hypothesis, the content of the base is $\sqrt{\det(DD^\top)}$. The vector $b_m$ is perpendicular to the base and its length corresponds to the height $\|b_m\|$ of the parallelepiped.

Furthermore,

$$A' A'^\top = \begin{pmatrix} D \\ b_m^\top \end{pmatrix} (D^\top \quad b_m) = \begin{pmatrix} DD^\top & Db_m \\ b_m^\top D^\top & b_m^\top b_m \end{pmatrix} = \begin{pmatrix} DD^\top & o \\ o^\top & b_m^\top b_m \end{pmatrix}$$

Thus $\det(A' A'^\top) = b_m^\top b_m \det(DD^\top)$ and taking the square root we obtain

$$\sqrt{\det(A' A'^\top)} = \|b_m\| \sqrt{\det(DD^\top)}.$$

This corresponds to the intuitive notion of volume as height times the content of the base. From $A'$ to $A$ one can pass via elementary row operations, since it suffices to add $c_m$ to the last row, which is a linear combination of $a_1, \ldots, a_{m-1}$. Thus there exist elementary matrices $E_1, \ldots, E_k$ such that $A = E_1 \ldots E_k A'$; moreover, their determinant is 1, because they only add a multiple of one row to another. Now

$$\det(AA^\top) = \det(E_1 \ldots E_k A' A'^\top E_k^\top \ldots E_1^\top) = \det(A' A'^\top).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 9.25 — Volume of a parallelepiped and elementary operations)</span></p>

The validity of Theorem 9.24 can be seen geometrically by analyzing the effect of elementary operations. Consider the parallelepiped generated by the rows of matrix $A \in \mathbb{R}^{n \times n}$ and we want to show that its volume is $|\det(A)|$. We know that the determinant does not change when we apply the third elementary operation to the matrix (adding a multiple of one row to another). The parallelepiped itself does change, however. Understanding why the volume remains preserved is easy from a geometric viewpoint: adding a multiple of a row to another (for example, the last one) means that the parallelepiped is sheared or straightened, but both the base and the height remain the same.

Visualizing the other elementary operations is even easier. Swapping rows of matrix $A$ means flipping the parallelepiped, and therefore its volume does not change. Multiplying a row of matrix $A$ by a scalar $\alpha$ stretches the parallelepiped in one direction, and thus the volume changes by a factor of $\alpha$. Multiplying the entire matrix $A$ by a scalar $\alpha$ stretches the parallelepiped in all directions, and the volume changes by a factor of $\alpha^n$.

If the matrix $A$ is singular, then the corresponding parallelepiped lies in some subspace of dimension less than $n$, and therefore its volume is zero. If the matrix $A$ is nonsingular, then we reduce it to the identity matrix using elementary operations -- the corresponding parallelepiped is the unit cube, which has volume 1.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 9.26 — Explanation of the definition of the determinant)</span></p>

The previous remark provides an alternative way to introduce the determinant and explain its definition. If we wanted to define the determinant of a matrix $A$ as the volume of the corresponding parallelepiped, we would encounter the problem of sign, since volume is always nonnegative. We therefore introduce something like an oriented volume, using the basic properties that the volume should satisfy:

1. The determinant of the identity matrix $I_n$ equals 1, corresponding to the volume of the unit cube.
2. Swapping rows changes the sign of the determinant. This corresponds to the property that the volume of a parallelepiped does not change by changing the order of the edges, i.e., by flipping, yet the sign change introduces a certain orientation into the definition of the determinant.
3. Multiplying a row of matrix $A$ by a scalar $\alpha \in \mathbb{R}$ changes the determinant by the factor $\alpha$. This corresponds to stretching the parallelepiped in the direction of the given edge, and thus the corresponding change in volume.
4. Row linearity of the determinant in the sense of Theorem 9.5. A consequence of this property is, for example, that shearing does not change the volume of a parallelepiped, and therefore the determinant remains the same.

From these basic properties, all other properties of the determinant can be derived, and the original definition $\det(A) = \sum_{p \in S_n} \operatorname{sgn}(p) a_{1,p(1)} \ldots a_{n,p(n)}$ can also be explained.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 9.27 — Volume of other geometric bodies)</span></p>

Let $A \in \mathbb{R}^{n \times n}$. As we have already mentioned, the volume of geometric bodies changes under the map $x \mapsto Ax$ by the factor $|\det(A)|$. A cube with edge length 1 is mapped to a parallelepiped with edges corresponding to the columns of matrix $A$, and its volume is therefore $|\det(A^\top)| = |\det(A)|$.

This property can be generalized to other commonly used geometric bodies, such as spheres, ellipsoids, polyhedra, etc. Such a body can be covered by cubes, and its image is therefore approximated by parallelepipeds, with the volume change being approximately $|\det(A)|$. By successively refining the approximation (shrinking the cubes), we obtain the resulting ratio in the limit.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 9.29 — Substitution in multidimensional integrals)</span></p>

The geometric interpretation of the determinant also allows us to easily see the validity of the substitution theorem for multidimensional integrals. Under fairly general assumptions, the theorem states that

$$\int_{\varphi(M)} f(y) \, \mathrm{d}y = \int_M f(\varphi(x)) \cdot |\det(\nabla \varphi(x))| \, \mathrm{d}x,$$

where $M \subseteq \mathbb{R}^n$ is an open set, $\varphi \colon M \to \mathbb{R}^n$ is an injective function with continuous partial derivatives, and $\nabla \varphi(x)$ is the Jacobian matrix of partial derivatives of the function $\varphi(x)$ (see Remark 6.30), which must be nonsingular for all $x \in M$. The explanation of the equality is then clear from the geometric viewpoint. The map $\varphi(x)$ is not linear, but it can be locally linearized precisely by the Jacobian matrix $\nabla \varphi(x)$. The map then locally changes volumes by a factor corresponding to the determinant of the Jacobian matrix. Therefore, the integral also changes by the same factor.

</div>

Determinants are used in solving many other geometric problems. For example, by computing a determinant we can easily decide whether a given point in the plane lies inside or outside a circle defined by its three points, and similarly in higher dimensions.

Let us mention a problem related to the volume of a parallelepiped, namely determining the volume of a simplex with $n + 1$ vertices in $\mathbb{R}^n$. Without loss of generality, let one vertex $a_0$ be at the origin and the others have positions $a_1, \ldots, a_n \in \mathbb{R}^n$. Define the matrix $A \in \mathbb{R}^{n \times n}$ so that its columns are the vectors $a_1, \ldots, a_n$. Then the volume of the simplex is $\frac{1}{n!} |\det(A)|$, i.e., it constitutes only a fraction of the parallelepiped given by the factor $1 : n!$.

The previous method of computing the volume of a simplex assumed that we know the positions of the individual vertices in the space $\mathbb{R}^n$. In some cases (e.g., molecular biology), however, only the distances $d_{ij} = \|a_i - a_j\|$ between the individual vertices are known, i.e., the edge lengths of the simplex. In this case, we compute the volume using the so-called Cayley--Menger determinant as

$$\frac{(-1)^{n-1}}{2^n (n!)^2} \begin{vmatrix} 0 & 1 & 1 & 1 & \ldots & 1 \\ 1 & 0 & d_{01}^2 & d_{02}^2 & \ldots & d_{0n}^2 \\ 1 & d_{10}^2 & 0 & d_{12}^2 & \ldots & d_{1n}^2 \\ 1 & d_{20}^2 & d_{21}^2 & 0 & \ldots & d_{2n}^2 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & d_{n0}^2 & d_{n1}^2 & d_{n2}^2 & \ldots & 0 \end{vmatrix}.$$

### Summary of Chapter 9

The determinant of a matrix $A$ is a numerical characteristic of the matrix that can be computed efficiently using Gaussian elimination -- one only needs to determine how elementary operations change the determinant of the matrix. The determinant indicates, among other things, whether the matrix is nonsingular or singular, and using the determinant we can also explicitly express the solution of the system $Ax = b$ with a nonsingular matrix $A$. Similarly, we can explicitly express the inverse of a nonsingular matrix, which leads to the concept of the adjugate matrix. Geometrically, the determinant represents the factor by which the volume of bodies changes under the linear map $x \mapsto Ax$; in particular, it gives the volume of the parallelepiped whose edges are given by the rows of matrix $A$.

## Chapter 10 — Eigenvalues

Eigenvalues (formerly also called "characteristic numbers"), similarly to the determinant, represent a certain characteristic of a matrix. They provide much important information about the matrix and the corresponding linear map.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 10.1 — Eigenvalues and eigenvectors)</span></p>

Let $A \in \mathbb{C}^{n \times n}$. Then $\lambda \in \mathbb{C}$ is an *eigenvalue* of the matrix $A$ and $x \in \mathbb{C}^n$ is the corresponding *eigenvector* if $Ax = \lambda x$, $x \neq o$.

</div>

The condition $x \neq o$ is necessary because for $x = o$ the equality would be trivially satisfied for every $\lambda \in \mathbb{C}$. On the other hand, $\lambda = 0$ can certainly occur. The eigenvector for a given eigenvalue is not uniquely determined — any nonzero scalar multiple of it is also an eigenvector. Sometimes, the eigenvector is therefore normalized so that $\|x\| = 1$.

Naturally, eigenvalues and eigenvectors can be defined in the same way over any other field. We will stay with $\mathbb{R}$ and $\mathbb{C}$, respectively. As we will see later, complex numbers cannot be avoided even when the matrix $A$ is real.

Eigenvalues can also be introduced more generally. Let $V$ be a vector space and $f \colon V \to V$ a linear map. Then $\lambda$ is an eigenvalue and $x \neq o$ the corresponding eigenvector if $f(x) = \lambda x$. However, we will mostly deal with eigenvalues of matrices, because due to the matrix representation of linear maps we can reduce the problem of finding eigenvalues and eigenvectors of linear maps on finitely generated spaces to matrices.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.2 — Geometric interpretation of eigenvalues and eigenvectors)</span></p>

An eigenvector represents an invariant direction under the map $x \mapsto Ax$, that is, a direction that maps to itself. In other words, if $v$ is an eigenvector, then the line $\operatorname{span}\lbrace v \rbrace$ maps to itself. The eigenvalue then represents the scaling factor in this invariant direction.

- Reflection across the line $y = -x$, mapping matrix $A = \begin{pmatrix} 0 & -1 \\ -1 & 0 \end{pmatrix}$: eigenvalue 1, eigenvector $(-1, 1)^\top$; eigenvalue $-1$, eigenvector $(1, 1)^\top$.
- Rotation by angle $90°$, mapping matrix $A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$: no real eigenvalues.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 10.3 — Characterization of eigenvalues and eigenvectors)</span></p>

Let $A \in \mathbb{C}^{n \times n}$. Then

1. $\lambda \in \mathbb{C}$ is an eigenvalue of $A$ if and only if $\det(A - \lambda I_n) = 0$,
2. $x \in \mathbb{C}^n$ is an eigenvector corresponding to eigenvalue $\lambda \in \mathbb{C}$ if and only if $o \neq x \in \operatorname{Ker}(A - \lambda I_n)$.

</div>

*Proof.* (1) $\lambda \in \mathbb{C}$ is an eigenvalue of $A$ if and only if $Ax = \lambda I_n x$, $x \neq o$, that is $(A - \lambda I_n)x = o$, $x \neq o$, which is equivalent to the singularity of the matrix $A - \lambda I_n$, and that in turn to the condition $\det(A - \lambda I_n) = 0$. (2) Analogously, $x \in \mathbb{C}^n$ is an eigenvector for eigenvalue $\lambda \in \mathbb{C}$ if and only if $(A - \lambda I_n)x = o$, $x \neq o$, that is, $x$ lies in the kernel of the matrix $A - \lambda I_n$.

A consequence of the theorem is that for a given eigenvalue $\lambda$ there are $\dim \operatorname{Ker}(A - \lambda I_n) = n - \operatorname{rank}(A - \lambda I_n)$ linearly independent eigenvectors.

### Characteristic polynomial

The first part of Theorem 10.3 says that $\lambda \in \mathbb{C}$ is an eigenvalue of matrix $A$ if and only if the matrix $A - \lambda I_n$ is singular, that is, $\det(A - \lambda I_n) = 0$. If we view $\lambda$ as a complex variable, then finding an eigenvalue is the same as finding a solution of the equation $\det(A - \lambda I_n) = 0$. Expanding the determinant from the definition yields a polynomial of degree at most $n$ in the variable $\lambda$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 10.4 — Characteristic polynomial)</span></p>

The *characteristic polynomial* of a matrix $A \in \mathbb{C}^{n \times n}$ with respect to the variable $\lambda$ is $p_A(\lambda) = \det(A - \lambda I_n)$.

</div>

From the definition of the determinant it is clear that the characteristic polynomial can be expressed in the form

$$p_A(\lambda) = \det(A - \lambda I_n) = (-1)^n \lambda^n + a_{n-1}\lambda^{n-1} + \ldots + a_1 \lambda + a_0.$$

Thus it is indeed a polynomial and has degree $n$. We can easily see that $a_{n-1} = (-1)^{n-1}(a_{11} + \ldots + a_{nn})$ and by substituting $\lambda = 0$ we obtain $a_0 = \det(A)$.

By the fundamental theorem of algebra, this polynomial has $n$ complex roots (counting multiplicities); denote them $\lambda_1, \ldots, \lambda_n$. Then

$$p_A(\lambda) = (-1)^n (\lambda - \lambda_1) \ldots (\lambda - \lambda_n).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 10.5)</span></p>

The eigenvalues of a matrix $A \in \mathbb{C}^{n \times n}$ are precisely the roots of its characteristic polynomial $p_A(\lambda)$, and there are $n$ of them counting multiplicities.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.6)</span></p>

Consider the matrix $A = \begin{pmatrix} 0 & -2 \\ 2 & 0 \end{pmatrix}$, similar to the matrix from Example 10.2. Then

$$p_A(\lambda) = \det(A - \lambda I_n) = \det \begin{pmatrix} -\lambda & -2 \\ 2 & -\lambda \end{pmatrix} = \lambda^2 + 4.$$

The roots of the polynomial, and hence the eigenvalues of matrix $A$, are $\pm 2i$. The eigenvector corresponding to $2i$ is $(1, -i)^\top$ and the eigenvector corresponding to $-2i$ is $(1, i)^\top$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 10.7 — Spectrum and spectral radius)</span></p>

Let $A \in \mathbb{C}^{n \times n}$ have eigenvalues $\lambda_1, \ldots, \lambda_n$. Then the *spectrum* of matrix $A$ is the set of its eigenvalues $\lbrace \lambda_1, \ldots, \lambda_n \rbrace$ and the *spectral radius* is $\rho(A) = \max_{i=1,\ldots,n} |\lambda_i|$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 10.8 — Algebraic and geometric multiplicity of an eigenvalue)</span></p>

Let $\lambda \in \mathbb{C}$ be an eigenvalue of matrix $A \in \mathbb{C}^{n \times n}$. The *algebraic multiplicity* of $\lambda$ equals the multiplicity of $\lambda$ as a root of $p_A(\lambda)$. The *geometric multiplicity* of $\lambda$ equals $n - \operatorname{rank}(A - \lambda I_n)$, i.e., the number of linearly independent eigenvectors corresponding to $\lambda$.

</div>

The algebraic multiplicity is always greater than or equal to the geometric multiplicity, which will follow in Section 10.4.

Computing eigenvalues as roots of the characteristic polynomial is not very efficient. Merely determining the individual coefficients of this polynomial is not a trivial task. Moreover, as we know, there is no formula or finite procedure for roots of polynomials and they are computed by iterative methods. The same holds for eigenvalues. Nevertheless, for certain special matrices, such as triangular matrices, we can determine the eigenvalues easily.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.9 — Eigenvalues of a triangular matrix)</span></p>

- Let $A \in \mathbb{C}^{n \times n}$ be a triangular matrix. Then its eigenvalues are the diagonal entries, since $\det(A - \lambda I_n) = (a_{11} - \lambda) \ldots (a_{nn} - \lambda)$.
- In particular, $I_n$ has eigenvalue 1, which has multiplicity $n$. The set of corresponding eigenvectors is $\mathbb{R}^n \setminus \lbrace o \rbrace$.
- In particular, $0_n$ has eigenvalue 0, which has multiplicity $n$. The set of corresponding eigenvectors is $\mathbb{R}^n \setminus \lbrace o \rbrace$.
- In particular, the matrix $\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ has eigenvalue 1, which is a double eigenvalue (algebraically). The corresponding eigenvector is, up to a scalar multiple, only $(1, 0)^\top$, so the geometric multiplicity of eigenvalue 1 is only one.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.10 — Geometric interpretation: shearing and stretching)</span></p>

Consider the matrix $A = \begin{pmatrix} 1.5 & 0.75 \\ 0 & 1 \end{pmatrix}$. The corresponding linear map $x \mapsto Ax$ geometrically represents a shear and a stretch in the $x_1$-axis by $50\%$, while it does not stretch in the direction of the $x_2$-axis.

The eigenvalues of matrix $A$ are 1.5 and 1, and the corresponding eigenvectors are $(1, 0)^\top$ and $(-1.5, 1)^\top$. The first eigenvalue and eigenvector say that the image is stretched by $50\%$ in the direction of the $x_1$-axis. The second eigenvalue and eigenvector say that the image is not deformed in the direction of the vector $(-1.5, 1)^\top$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 10.11 — Product and sum of eigenvalues)</span></p>

Let $A \in \mathbb{C}^{n \times n}$ with eigenvalues $\lambda_1, \ldots, \lambda_n$. Then

1. $\det(A) = \lambda_1 \ldots \lambda_n$,
2. $\operatorname{trace}(A) = \lambda_1 + \ldots + \lambda_n$.

</div>

*Proof.* (1) We know that $p_A(\lambda) = (-1)^n(\lambda - \lambda_1) \ldots (\lambda - \lambda_n)$. Substituting $\lambda = 0$ we get $\det(A) = (-1)^n(-\lambda_1) \ldots (-\lambda_n) = \lambda_1 \ldots \lambda_n$. (2) By comparing the coefficients of $\lambda^{n-1}$ in different expressions of the characteristic polynomial.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 10.12 — Properties of eigenvalues)</span></p>

Let $A \in \mathbb{C}^{n \times n}$ have eigenvalues $\lambda_1, \ldots, \lambda_n$ and corresponding eigenvectors $x_1, \ldots, x_n$. Then:

1. $A$ is nonsingular if and only if $0$ is not its eigenvalue,
2. if $A$ is nonsingular, then $A^{-1}$ has eigenvalues $\lambda_1^{-1}, \ldots, \lambda_n^{-1}$ and eigenvectors $x_1, \ldots, x_n$,
3. $A^2$ has eigenvalues $\lambda_1^2, \ldots, \lambda_n^2$ and eigenvectors $x_1, \ldots, x_n$,
4. $\alpha A$ has eigenvalues $\alpha \lambda_1, \ldots, \alpha \lambda_n$ and eigenvectors $x_1, \ldots, x_n$,
5. $A + \alpha I_n$ has eigenvalues $\lambda_1 + \alpha, \ldots, \lambda_n + \alpha$ and eigenvectors $x_1, \ldots, x_n$,
6. $A^\top$ has eigenvalues $\lambda_1, \ldots, \lambda_n$, but generally different eigenvectors.

</div>

*Proof.* (1) $A$ has eigenvalue 0 if and only if $0 = \det(A - 0 \cdot I_n) = \det(A)$, that is, when $A$ is singular. (2) For each $i$ we have $Ax_i = \lambda_i x_i$. Multiplying by $A^{-1}$ we get $x_i = \lambda_i A^{-1} x_i$ and dividing by $\lambda_i \neq 0$ we obtain $A^{-1} x_i = \lambda_i^{-1} x_i$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.13)</span></p>

Consider the matrices

$$A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}, \quad B = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}.$$

Both have all eigenvalues equal to zero. The sum of the matrices

$$A + B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

has eigenvalues $-1$ and $1$. The product of the matrices

$$AB = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$

has eigenvalues 0 and 1. From this simple example one can conclude that adding and multiplying matrices changes the eigenvalues, and they cannot be easily estimated from the eigenvalues of the original matrices. Matrices of certain types (e.g., diagonalizable, Section 10.3) behave more reasonably, and the eigenvalues of the product or sum of such matrices cannot grow completely arbitrarily.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 10.14 — Topology of the set of nonsingular matrices)</span></p>

The set of nonsingular matrices is a so-called *dense set* in the space $\mathbb{R}^{n \times n}$. This means that every matrix $A \in \mathbb{R}^{n \times n}$ can be expressed as the limit of a suitable sequence of nonsingular matrices. For a nonsingular matrix $A$ the observation is obvious; it suffices to consider the constant sequence consisting of $A$. If $A$ is singular, then $A + \frac{1}{k}I_n$ is nonsingular for sufficiently large $k$, since it will not have a zero eigenvalue. Thus we have a sequence of nonsingular matrices $A + \frac{1}{k}I_n$ converging to $A$ as $k \to \infty$.

The set of nonsingular matrices is also a so-called *open set* in the space $\mathbb{R}^{n \times n}$ (and consequently the set of singular matrices is closed). This property says that for every nonsingular matrix $A \in \mathbb{R}^{n \times n}$, the matrices in its neighborhood are also nonsingular. The claim follows from the fact that $\det(A) \neq 0$ and that the determinant is a continuous function. Therefore $\det(A') \neq 0$ also for matrices $A'$ from a sufficiently small neighborhood of $A$.

</div>

We know that even a real matrix can have some complex eigenvalues. However, these complex eigenvalues can always be paired into conjugate pairs.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 10.15)</span></p>

If $\lambda \in \mathbb{C}$ is an eigenvalue of a matrix $A \in \mathbb{R}^{n \times n}$, then the complex conjugate $\overline{\lambda}$ is also an eigenvalue of $A$.

</div>

*Proof.* We know that $\lambda$ is a root of $p_A(\lambda) = (-1)^n \lambda^n + a_{n-1}\lambda^{n-1} + \ldots + a_1\lambda + a_0 = 0$. Taking the complex conjugate of both sides we get $(-1)^n \overline{\lambda}^n + a_{n-1}\overline{\lambda}^{n-1} + \ldots + a_1\overline{\lambda} + a_0 = 0$, so $\overline{\lambda}$ is also a root of $p_A(\lambda)$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.16)</span></p>

The spectrum of a real matrix is therefore a set symmetric with respect to the real axis. Complex matrices can have any $n$ complex numbers as their spectrum.

</div>

Now we will show that computing the roots of a polynomial can be reduced to the problem of finding eigenvalues of a certain matrix.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 10.17 — Companion matrix)</span></p>

Let $p(x) = x^n + a_{n-1}x^{n-1} + \ldots + a_1 x + a_0$. Then the *companion matrix* of the polynomial $p(x)$ is the square matrix of order $n$ defined by

$$C(p) \coloneqq \begin{pmatrix} 0 & \ldots & \ldots & 0 & -a_0 \\ 1 & \ddots & & \vdots & -a_1 \\ 0 & \ddots & \ddots & \vdots & -a_2 \\ \vdots & \ddots & \ddots & 0 & \vdots \\ 0 & \ldots & 0 & 1 & -a_{n-1} \end{pmatrix}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 10.18 — On the companion matrix)</span></p>

For the characteristic polynomial of the matrix $C(p)$ we have $p_{C(p)}(\lambda) = (-1)^n p(\lambda)$, so the eigenvalues of $C(p)$ correspond to the roots of the polynomial $p(\lambda)$.

</div>

The theorem has, among other things, the consequence that the problems of finding roots of real polynomials and eigenvalues of matrices are mutually reducible: Theorem 10.5 reduces finding eigenvalues of a matrix to finding roots of a polynomial, and Theorem 10.18 does the opposite.

### Cayley--Hamilton theorem

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.19 — Polynomial matrix)</span></p>

To better understand the following theorem, we present an example of a polynomial matrix and a matrix polynomial

$$\begin{pmatrix} \lambda^2 - \lambda & 2\lambda - 3 \\ 7 & 5\lambda^2 - 4 \end{pmatrix} = \lambda^2 \begin{pmatrix} 1 & 0 \\ 0 & 5 \end{pmatrix} + \lambda \begin{pmatrix} -1 & 2 \\ 0 & 0 \end{pmatrix} + \begin{pmatrix} 0 & -3 \\ 7 & -4 \end{pmatrix}.$$

These are two representations of the same matrix with parameter $\lambda$, and we can easily convert one representation to the other.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 10.20 — Cayley--Hamilton)</span></p>

Let $A \in \mathbb{C}^{n \times n}$ and $p_A(\lambda) = (-1)^n \lambda^n + a_{n-1}\lambda^{n-1} + \ldots + a_1\lambda + a_0$. Then

$$(-1)^n A^n + a_{n-1}A^{n-1} + \ldots + a_1 A + a_0 I_n = 0.$$

</div>

*Proof.* We know that for adjugate matrices $(A - \lambda I_n) \operatorname{adj}(A - \lambda I_n) = \det(A - \lambda I_n)I_n$. Each entry of $\operatorname{adj}(A - \lambda I_n)$ is a polynomial of degree at most $n - 1$ in the variable $\lambda$, so it can be expressed in the form $\operatorname{adj}(A - \lambda I_n) = \lambda^{n-1}B_{n-1} + \ldots + \lambda B_1 + B_0$ for certain $B_{n-1}, \ldots, B_0 \in \mathbb{C}^{n \times n}$. Substituting and comparing coefficients of powers of $\lambda$ we obtain a system of equations, and summing them after multiplying by the corresponding powers of $A$ yields the desired result.

In short, the statement of the Cayley--Hamilton theorem can be expressed as $p_A(A) = 0$, i.e., a matrix is itself a root of its characteristic polynomial.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 10.21)</span></p>

Let $A \in \mathbb{C}^{n \times n}$. Then:

1. For every $k \in \mathbb{N}$, $A^k \in \operatorname{span}\lbrace I_n, A, \ldots, A^{n-1} \rbrace$, that is, $A^k$ is a linear combination of the matrices $I_n, A, \ldots, A^{n-1}$.
2. If $A$ is nonsingular, then $A^{-1} \in \operatorname{span}\lbrace I_n, A, \ldots, A^{n-1} \rbrace$.

</div>

*Proof.* (1) It suffices to consider $k \ge n$. Dividing the polynomial $\lambda^k$ by the polynomial $p_A(\lambda)$ with remainder, we decompose $\lambda^k = r(\lambda) p_A(\lambda) + s(\lambda)$. Then $A^k = r(A) p_A(A) + s(A) = s(A) = b_{n-1}A^{n-1} + \ldots + b_1 A + b_0 I_n$. (2) From $p_A(A) = (-1)^n A^n + a_{n-1}A^{n-1} + \ldots + a_1 A + a_0 I_n = 0$ and $a_0 \neq 0$ by the nonsingularity of $A$.

According to this corollary, a large power $A^k$ of a matrix $A$ can alternatively be computed by finding the appropriate coefficients of the characteristic polynomial and expressing $A^k$ as a linear combination of $I_n, A, \ldots, A^{n-1}$. Similarly, we can express $A^{-1}$, and consequently the solution of the system $Ax = b$ with a nonsingular matrix as $A^{-1}b = \frac{1}{a_0}(-(-1)^n A^{n-1}b - \ldots - a_1 b)$.

### Diagonalizability

When solving systems of linear equations using Gauss--Jordan elimination, we used elementary row operations. These do not change the solution set and transform the system matrix into a form from which the solution can be easily read off. It is natural to look for analogous, spectrum-preserving, transformations for the problem of computing eigenvalues as well. Elementary row operations cannot be used because they change the spectrum. A suitable transformation is the so-called *similarity*, because it preserves the spectrum. And if we manage to use this transformation to bring the matrix to diagonal form, we are done -- the sought eigenvalues are on the diagonal.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 10.22 — Similarity)</span></p>

Matrices $A, B \in \mathbb{C}^{n \times n}$ are *similar* if there exists a nonsingular $S \in \mathbb{C}^{n \times n}$ such that $A = SBS^{-1}$.

</div>

Similarity can equivalently be defined by the relation $AS = SB$ for some nonsingular matrix $S$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.23)</span></p>

The matrices $\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ and $\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$ are similar via the matrix $S = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 10.24 — Eigenvalues of similar matrices)</span></p>

Similar matrices have the same eigenvalues.

</div>

*Proof.* From the similarity of the matrices, there exists a nonsingular $S$ such that $A = SBS^{-1}$. Then

$$p_A(\lambda) = \det(A - \lambda I_n) = \det(SBS^{-1} - \lambda S I_n S^{-1}) = \det(S(B - \lambda I_n)S^{-1}) = \det(S)\det(B - \lambda I_n)\det(S^{-1}) = \det(B - \lambda I_n) = p_B(\lambda).$$

Both matrices have the same characteristic polynomials, and hence the same eigenvalues.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.25)</span></p>

Show that similarity as a binary relation is reflexive, symmetric, and transitive. Thus it is an equivalence relation.

</div>

The theorem says nothing about the eigenvectors, which may change. What does remain invariant, however, is the number of linearly independent eigenvectors.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 10.26)</span></p>

Let the matrices $A, B \in \mathbb{C}^{n \times n}$ be similar and let $\lambda$ be their eigenvalue. Then the number of eigenvectors corresponding to $\lambda$ is the same for both matrices.

</div>

*Proof.* Let $A = SBS^{-1}$. Since the rank of a matrix does not change when multiplied by a nonsingular matrix, $\operatorname{rank}(A - \lambda I_n) = \operatorname{rank}(S(B - \lambda I_n)S^{-1}) = \operatorname{rank}(B - \lambda I_n)$. Therefore the dimensions of the kernels of both matrices $A - \lambda I_n$ and $B - \lambda I_n$ are the same, and hence so is the number of eigenvectors.

Eigenvalues do not change under a similarity transformation, so if we transform matrix $A$ by a similarity transformation to diagonal or more generally triangular form, we find its eigenvalues on the diagonal. In particular, matrices that can be transformed to diagonal form have especially nice properties.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 10.27 — Diagonalizability)</span></p>

A matrix $A \in \mathbb{C}^{n \times n}$ is *diagonalizable* if it is similar to some diagonal matrix.

</div>

A diagonalizable matrix $A$ can therefore be expressed in the form $A = S \Lambda S^{-1}$, where $S$ is nonsingular and $\Lambda$ is diagonal. This form is called the *spectral decomposition*, because the diagonal of $\Lambda$ contains the spectrum of matrix $A$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.28)</span></p>

Not every matrix is diagonalizable, e.g., $A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$. Its eigenvalue (of multiplicity two) is 0. If $A$ were diagonalizable, it would be similar to the zero matrix, so $A = S 0 S^{-1} = 0$, which is a contradiction.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 10.29 — Characterization of diagonalizability)</span></p>

A matrix $A \in \mathbb{C}^{n \times n}$ is diagonalizable if and only if it has $n$ linearly independent eigenvectors.

</div>

*Proof.* Implication "$\Rightarrow$": If $A$ is diagonalizable, then it has the spectral decomposition $A = S \Lambda S^{-1}$, where $S$ is nonsingular and $\Lambda$ is diagonal. From the equality $AS = S\Lambda$ and comparing $j$-th columns we get $AS_{*j} = (S\Lambda)_{*j} = S\Lambda_{jj}e_j = \Lambda_{jj}S_{*j}$. Thus $\Lambda_{jj}$ is an eigenvalue and $S_{*j}$ is the corresponding eigenvector. The columns of $S$ are linearly independent due to its nonsingularity.

Implication "$\Leftarrow$": Let $A$ have eigenvalues $\lambda_1, \ldots, \lambda_n$ with corresponding linearly independent eigenvectors $x_1, \ldots, x_n$. Construct the nonsingular matrix $S \coloneqq (x_1 \mid \cdots \mid x_n)$ and the diagonal matrix $\Lambda \coloneqq \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$. Then $(AS)_{*j} = Ax_j = \lambda_j x_j = \Lambda_{jj} S_{*j} = (S\Lambda)_{*j}$. Thus $AS = S\Lambda$, from which $A = S\Lambda S^{-1}$.

Non-diagonalizable matrices are those for which certain pathological situations arise, but diagonalizable matrices have a whole range of natural properties. If a matrix $A$ is diagonalizable, then:

- The algebraic and geometric multiplicity of every eigenvalue of $A$ are the same.
- The rank of matrix $A$ equals the number of nonzero eigenvalues of $A$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 10.30 — Geometric interpretation of diagonalization)</span></p>

Another perspective on diagonalization is geometric: we know that an eigenvector represents an invariant direction under the map $x \mapsto Ax$. Now imagine that $A$ represents the matrix of some linear map $f \colon \mathbb{C}^n \to \mathbb{C}^n$ with respect to a basis $B$. Let $S = {}_{B'}[id]_B$ be the change-of-basis matrix from $B$ to another basis $B'$. Then $SAS^{-1} = {}_{B'}[id]_B \cdot {}_B[f]_B \cdot {}_B[id]_{B'} = {}_{B'}[f]_{B'}$ is the matrix of the map $f$ with respect to the new basis $B'$. Now diagonalizability can be understood as finding a suitable basis $B'$ so that the corresponding matrix is diagonal, and thus simply describes the behavior of the map.

Thanks to this geometric perspective, we can easily see the validity of Theorem 10.24. Similarity means a change of basis, but it does not change the linear map $f$ itself, so the eigenvalues must remain the same.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.31 — Geometric interpretation of diagonalization)</span></p>

Let $A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$. The eigenvalues and eigenvectors of matrix $A$ are: $\lambda_1 = 4$, $x_1 = (1, 1)^\top$; $\lambda_2 = 2$, $x_2 = (-1, 1)^\top$.

The diagonalization has the form:

$$A = S \Lambda S^{-1} = \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 4 & 0 \\ 0 & 2 \end{pmatrix} \begin{pmatrix} \frac{1}{2} & \frac{1}{2} \\ -\frac{1}{2} & \frac{1}{2} \end{pmatrix}.$$

Geometric interpretation: In the coordinate system of eigenvectors, the mapping matrix is diagonal and the map represents only scaling along the axes.

</div>

Now we will show that distinct eigenvalues correspond to linearly independent eigenvectors.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 10.32 — Eigenvectors of distinct eigenvalues)</span></p>

Let $\lambda_1, \ldots, \lambda_k$ be pairwise distinct eigenvalues (not necessarily all) of a matrix $A \in \mathbb{C}^{n \times n}$. Then the corresponding eigenvectors $x_1, \ldots, x_k$ are linearly independent.

</div>

*Proof.* By mathematical induction on $k$. For $k = 1$ obvious, since an eigenvector is nonzero. Induction step $k \leftarrow k - 1$. Consider the linear combination $\alpha_1 x_1 + \ldots + \alpha_k x_k = o$. Multiplying by matrix $A$ we get $\alpha_1 \lambda_1 x_1 + \ldots + \alpha_k \lambda_k x_k = o$. Subtracting $\lambda_k$ times the first equation from the second we get $\alpha_1(\lambda_1 - \lambda_k) x_1 + \ldots + \alpha_{k-1}(\lambda_{k-1} - \lambda_k) x_{k-1} = o$. By the induction hypothesis, $x_1, \ldots, x_{k-1}$ are linearly independent, so $\alpha_1 = \ldots = \alpha_{k-1} = 0$. Substituting back we have $\alpha_k x_k = o$, i.e., $\alpha_k = 0$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 10.33)</span></p>

If a matrix $A \in \mathbb{C}^{n \times n}$ has $n$ pairwise distinct eigenvalues, then it is diagonalizable.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 10.34)</span></p>

Let $A, B \in \mathbb{C}^{n \times n}$. Then the matrices $AB$ and $BA$ have the same eigenvalues including multiplicities.

</div>

*Proof.* The matrices $\begin{pmatrix} AB & 0 \\ B & 0 \end{pmatrix}$ and $\begin{pmatrix} 0 & 0 \\ B & BA \end{pmatrix}$ are block triangular, so they have the same eigenvalues as $AB$ and $BA$, respectively, plus an additional eigenvalue 0 of multiplicity $n$. These matrices are similar via the matrix $S = \begin{pmatrix} I & A \\ 0 & I \end{pmatrix}$. The above proposition also holds for rectangular matrices $A, B^\top \in \mathbb{R}^{m \times n}$, but the statement holds only for nonzero eigenvalues; the multiplicity of zero eigenvalues may be (and typically is) different.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.35 — Matrix power)</span></p>

Let $A = S \Lambda S^{-1}$ be the spectral decomposition of a matrix $A \in \mathbb{C}^{n \times n}$. Then $A^2 = S \Lambda S^{-1} S \Lambda S^{-1} = S \Lambda^2 S^{-1}$. More generally:

$$A^k = S \Lambda^k S^{-1} = S \begin{pmatrix} \lambda_1^k & 0 & 0 \\ 0 & \ddots & 0 \\ 0 & 0 & \lambda_n^k \end{pmatrix} S^{-1}.$$

We can also study the asymptotic behavior. In simplified form:

$$\lim_{k \to \infty} A^k = \begin{cases} 0, & \text{if } \rho(A) < 1, \\ \text{diverges}, & \text{if } \rho(A) > 1, \\ \text{converges / diverges}, & \text{if } \rho(A) = 1. \end{cases}$$

Geometrically: Raising the matrix $A$ to a power corresponds to composing the map with itself. If all eigenvalues are less than 1 in absolute value, the linear map contracts distances and therefore converges to zero. If at least one eigenvalue is greater than 1 in absolute value, the linear map stretches distances in the direction of the corresponding eigenvector, and therefore diverges.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.36 — Recurrence relations and Fibonacci)</span></p>

Consider the sequence $a_1, a_2, \ldots$ defined by the recurrence relation $a_n = p a_{n-1} + q a_{n-2}$, where $a_1, a_2$ are the given initial values of the sequence and $p, q$ are constants. We express the recurrence in matrix form:

$$\begin{pmatrix} a_n \\ a_{n-1} \end{pmatrix} = \begin{pmatrix} p & q \\ 1 & 0 \end{pmatrix} \begin{pmatrix} a_{n-1} \\ a_{n-2} \end{pmatrix}.$$

Denoting $x_n \coloneqq \begin{pmatrix} a_n \\ a_{n-1} \end{pmatrix}$ and $A = \begin{pmatrix} p & q \\ 1 & 0 \end{pmatrix}$, the recurrence takes the form $x_n = A x_{n-1} = A^2 x_{n-2} = \ldots = A^{n-2} x_2$.

We therefore need to determine a higher power of matrix $A$. Diagonalization serves this purpose: $A = S \Lambda S^{-1}$, and then $x_n = S \Lambda^{n-2} S^{-1} x_2$. The explicit expression for $a_n$ is hidden in the first component of the vector $x_n$.

For the Fibonacci sequence with $a_n = a_{n-1} + a_{n-2}$ and $a_1 = a_2 = 1$ and the golden ratio $\varphi \coloneqq \frac{1}{2}(1 + \sqrt{5})$ we get

$$a_n = -\frac{\sqrt{5}}{5}(1 - \varphi)^n + \frac{\sqrt{5}}{5}\varphi^n.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.37 — Discrete and fast Fourier transform)</span></p>

Matrices that have the same form as matrix $A$ (i.e., circulants) have remarkable properties. One of them is that their eigenvectors do not depend on the specific values $a_0, \ldots, a_{n-1}$, but only on the structure of the circulant. The eigenvalues of matrix $A$ can then be easily computed from the knowledge of the eigenvectors.

Moreover, for the matrix $S$ we have $S^{-1} = \frac{1}{n}\overline{S}^\top$. The product $Ab$ can now be expressed as

$$Ab = S \Lambda \frac{1}{n} \overline{S}^\top b.$$

The product $Ab$ can therefore be expressed using three operations: successive multiplication by the three matrices $\frac{1}{n}\overline{S}^\top$, $\Lambda$, and $S$. If we think of the matrix $S$ as the change-of-basis matrix from the eigenvector basis to the canonical basis, then the first operation converts the vector $b$ to the coordinate system of the eigenvectors (the so-called *discrete Fourier transform*), the second performs the main operation, and the third converts the resulting vector back to the canonical basis (the so-called *inverse Fourier transform*). The second operation is obviously trivial because $\Lambda$ is diagonal.

Using suitable algorithms (based, e.g., on the divide and conquer principle), multiplying the vector $b$ by the matrix $\frac{1}{n}\overline{S}^\top$ can be performed in time proportional to $n \log(n)$. Similarly for multiplication by $S$. This is an asymptotically significant improvement over the ordinary matrix product $Ab$, which requires on the order of $2n^2$ arithmetic operations. This improved method is called the *fast Fourier transform* and is one of the most important numerical algorithms.

</div>

### Jordan normal form

The simplest form of a matrix that can be achieved using elementary row operations is the reduced row echelon form. But what is the simplest form of a matrix that can be achieved using similarity? It is not a diagonal matrix, because we already know that not all matrices are diagonalizable. Nevertheless, every matrix can be transformed by a similarity transformation into a relatively simple form called the Jordan normal form.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 10.38 — Jordan block)</span></p>

Let $\lambda \in \mathbb{C}$, $k \in \mathbb{N}$. A *Jordan block* $J_k(\lambda)$ is a square matrix of order $k$ defined by

$$J_k(\lambda) = \begin{pmatrix} \lambda & 1 & 0 & \ldots & 0 \\ 0 & \ddots & \ddots & \ddots & \vdots \\ \vdots & \ddots & \ddots & \ddots & 0 \\ \vdots & & \ddots & \ddots & 1 \\ 0 & \ldots & \ldots & 0 & \lambda \end{pmatrix}.$$

A Jordan block has eigenvalue $\lambda$, which has multiplicity $k$, and it has only one eigenvector $e_1 = (1, 0, \ldots, 0)^\top$, because the matrix $J_k(\lambda) - \lambda I_k$ has rank $k - 1$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 10.39 — Jordan normal form)</span></p>

A matrix $J \in \mathbb{C}^{n \times n}$ is in *Jordan normal form* if it has the block diagonal form

$$J = \begin{pmatrix} J_{k_1}(\lambda_1) & 0 & \ldots & 0 \\ 0 & \ddots & \ddots & \vdots \\ \vdots & \ddots & \ddots & 0 \\ 0 & \ldots & 0 & J_{k_m}(\lambda_m) \end{pmatrix}$$

and the diagonal contains the Jordan blocks $J_{k_1}(\lambda_1), \ldots, J_{k_m}(\lambda_m)$.

</div>

The values $\lambda_i$ and $k_i$ need not be pairwise distinct. Likewise, a given Jordan block may appear more than once.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 10.40 — On the Jordan normal form)</span></p>

Every matrix $A \in \mathbb{C}^{n \times n}$ is similar to a matrix in Jordan normal form. This matrix is uniquely determined up to the ordering of the blocks.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.41)</span></p>

The matrix

$$A = \begin{pmatrix} 5 & -2 & 2 & -2 & 0 \\ 0 & 6 & -1 & 3 & 2 \\ 2 & 2 & 7 & -2 & -2 \\ 2 & 3 & 1 & 2 & -4 \\ -2 & -2 & -2 & 6 & 11 \end{pmatrix}$$

has eigenvalue 5 (of multiplicity two) and 7 (of multiplicity three). Since $3 = \operatorname{rank}(A - 5I_5) = \operatorname{rank}(A - 5I_5)^2$, we will look for two chains of length 1. We find two linearly independent vectors $x_1, x_2 \in \operatorname{Ker}(A - 5I_5)$, for example $x_1 = (-2, 1, 1, 0, 0)^\top$ and $x_2 = (-1, 1, 0, -1, 1)^\top$.

Let us proceed to eigenvalue 7. Now we have $\operatorname{rank}(A - 7I_5) = 3$ and $\operatorname{rank}(A - 7I_5)^2 = \operatorname{rank}(A - 7I_5)^3 = 2$. We therefore choose $x_4 \in \operatorname{Ker}(A - 7I_5)^2 \setminus \operatorname{Ker}(A - 7I_5)$, for example $x_4 = (1, 0, 1, 0, 0)^\top$, and then the corresponding part of the basis is formed by the chain $x_3 = (A - 7I_5)x_4 = (0, -1, 2, 3, -4)^\top$, $x_4$. The last basis vector will be a vector from $\operatorname{Ker}(A - 7I_5)$ linearly independent of $x_3$, for example $x_5 = (0, 1, 1, 0, 1)^\top$.

Placing these vectors $x_1, \ldots, x_5$ into the columns of the matrix $S$, we get

$$J = S^{-1}AS = \begin{pmatrix} 5 & 0 & 0 & 0 & 0 \\ 0 & 5 & 0 & 0 & 0 \\ 0 & 0 & 7 & 1 & 0 \\ 0 & 0 & 0 & 7 & 0 \\ 0 & 0 & 0 & 0 & 7 \end{pmatrix}$$

is the sought Jordan normal form of matrix $A$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 10.42)</span></p>

The total number of Jordan blocks corresponding to $\lambda$ equals the number of eigenvectors for $\lambda$.

</div>

As a consequence, we further obtain that the (algebraic) multiplicity of every eigenvalue $\lambda$ is always greater than or equal to the number of eigenvectors corresponding to it (i.e., the geometric multiplicity).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 10.43)</span></p>

The multiplicity of an eigenvalue is greater than or equal to the number of eigenvectors corresponding to it.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 10.44 — Sizes and number of blocks)</span></p>

The number of blocks $J_k(\lambda)$ of a matrix $A \in \mathbb{C}^{n \times n}$ in the resulting Jordan normal form equals

$$\operatorname{rank}(\bar{A}^{k-1}) - 2\operatorname{rank}(\bar{A}^k) + \operatorname{rank}(\bar{A}^{k+1}),$$

kde $\bar{A} = A - \lambda I_n$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 10.45 — Generalized eigenvectors)</span></p>

Let $J$ be the Jordan normal form of a matrix $A$, that is, $A = SJS^{-1}$ for some nonsingular matrix $S$. If the matrix $J$ is diagonal, then the columns of $S$ are eigenvectors that correspond in order to the eigenvalues of $A$ placed on the diagonal of $J$. The columns of the matrix $S$ are called generalized eigenvectors.

In summary, let $A \in \mathbb{C}^{n \times n}$ and $\lambda \in \mathbb{C}$ be an eigenvalue. The space of eigenvectors corresponding to $\lambda$ is the kernel of $A - \lambda I_n$. The space of generalized eigenvectors corresponding to $\lambda$ is the kernel of $(A - \lambda I_n)^n$. There is a full set of eigenvectors (i.e., $n$) only when the matrix $A$ is diagonalizable. On the other hand, there are always $n$ (linearly independent) generalized eigenvectors. This naturally relates to the fact that every matrix is similar to a matrix in Jordan normal form.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.46 — Matrix powers)</span></p>

Already in Example 10.35 we mentioned the use of diagonalization for computing matrix powers. Using the Jordan normal form, we can generalize the statement for arbitrary $A \in \mathbb{C}^{n \times n}$: Let $A = SJS^{-1}$, then

$$A^k = SJ^kS^{-1} = S \begin{pmatrix} J_{k_1}(\lambda_1)^k & 0 & \ldots & 0 \\ 0 & \ddots & \ddots & \vdots \\ \vdots & \ddots & \ddots & 0 \\ 0 & \ldots & 0 & J_{k_m}(\lambda_m)^k \end{pmatrix} S^{-1}.$$

Asymptotically, we then get the same result as for diagonalizable matrices:

$$\lim_{k \to \infty} A^k = \begin{cases} 0, & \text{if } \rho(A) < 1, \\ \text{diverges}, & \text{if } \rho(A) > 1, \\ \text{converges / diverges}, & \text{if } \rho(A) = 1. \end{cases}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.47 — Matrix function)</span></p>

Let us ask the question: How do we define a matrix function such as $\cos(A)$, $e^A$, etc.? For a real function $f \colon \mathbb{R} \to \mathbb{R}$ and a matrix $A \in \mathbb{R}^{n \times n}$, one can define $f(A)$ by applying the function to each entry of the matrix separately, but such an approach does not have very nice properties.

Assume that the function $f \colon \mathbb{R} \to \mathbb{R}$ can be expressed by an infinite series $f(x) = \sum_{i=0}^{\infty} a_i x^i$; real analytic functions such as $\sin(x)$, $\exp(x)$, etc. satisfy this assumption. Then it is natural to define $f(A) = \sum_{i=0}^{\infty} a_i A^i$. We already know how to compute matrix powers, so if $A = SJS^{-1}$, then

$$f(A) = \sum_{i=0}^{\infty} a_i S J^i S^{-1} = S \left( \sum_{i=0}^{\infty} a_i J^i \right) S^{-1} = S f(J) S^{-1}.$$

We can further easily see that

$$f(J) \coloneqq \begin{pmatrix} f(J_{k_1}(\lambda_1)) & 0 & \ldots & 0 \\ 0 & \ddots & \ddots & \vdots \\ \vdots & \ddots & \ddots & 0 \\ 0 & \ldots & 0 & f(J_{k_m}(\lambda_m)) \end{pmatrix}.$$

For $k_i = 1$ this is trivial, as it is a matrix of order 1. For $k_i > 1$ the formula is more involved:

$$f(J_{k_i}(\lambda_i)) \coloneqq \begin{pmatrix} f(\lambda_i) & f'(\lambda_i) & \ldots & \frac{f^{(k_i - 1)}(\lambda_i)}{(k_i - 1)!} \\ 0 & \ddots & \ddots & \vdots \\ \vdots & \ddots & \ddots & f'(\lambda_i) \\ 0 & \ldots & 0 & f(\lambda_i) \end{pmatrix}.$$

For example, the function $f(x) = x^2$ has the matrix extension $f(A) = A^2$, which is simply the standard matrix squaring. Another example of a matrix function is the matrix exponential $e^A = \sum_{i=0}^{\infty} \frac{1}{i!} A^i$. One of its many applications is for expressing rotations in the space $\mathbb{R}^3$. The matrix $e^R$, where

$$R = \alpha \begin{pmatrix} 0 & -z & y \\ z & 0 & -x \\ -y & x & 0 \end{pmatrix}$$

describes the rotation matrix around the axis with direction $(x, y, z)^\top$ by angle $\alpha$ according to the right-hand rule.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.48 — System of linear differential equations)</span></p>

Consider the so-called system of linear differential equations:

$$u(t)' = Au(t),$$

where $A \in \mathbb{R}^{n \times n}$. The goal is to find the unknown function $u \colon \mathbb{R} \to \mathbb{R}^n$ satisfying this system for a given initial condition of the form $u(t_0) = u_0$.

For the case $n = 1$, the solution of the differential equation $u(t)' = au(t)$ is the function $u(t) = v \cdot e^{at}$, where $v \in \mathbb{R}^n$ is arbitrary. This motivates us to look for a solution of the general case in the form $u(t) = (v_1 e^{\lambda_1 t}, \ldots, v_n e^{\lambda_n t}) = e^{\lambda t} v$, where $v_i, \lambda$ are unknowns. Substituting $u(t) \coloneqq e^{\lambda t} v$ into the system we get $\lambda e^{\lambda t} v = e^{\lambda t} Av$, that is, $\lambda v = Av$.

This is precisely the problem of computing eigenvalues and eigenvectors. Let the matrix $A$ have eigenvalues $\lambda_1, \ldots, \lambda_n$ and eigenvectors $x_1, \ldots, x_n$. Then the solution is $u(t) = \sum_{i=1}^n \alpha_i e^{\lambda_i t} x_i$, where $\alpha_i \in \mathbb{R}$ is obtained from the initial conditions.

Consider a specific example: $u_1'(t) = 7u_1(t) - 4u_2(t)$, $u_2'(t) = 5u_1(t) - 2u_2(t)$. The matrix $A = \begin{pmatrix} 7 & -4 \\ 5 & -2 \end{pmatrix}$ has eigenvalues 2 and 3, with corresponding eigenvectors $(4, 5)^\top$ and $(1, 1)^\top$. The solutions are of the form

$$\begin{pmatrix} u_1(t) \\ u_2(t) \end{pmatrix} = \alpha_1 e^{2t} \begin{pmatrix} 4 \\ 5 \end{pmatrix} + \alpha_2 e^{3t} \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \quad \alpha_1, \alpha_2 \in \mathbb{R}.$$

The eigenvalues also determine how the solution $u(t)$ behaves over longer time. If the eigenvalues are negative, $e^{\lambda_i t}$ converges to zero as $t \to \infty$. In this case the problem is so-called asymptotically stable. The problem is unstable if some eigenvalue is positive, because then $e^{\lambda_i t}$ diverges as $t \to \infty$.

</div>

### Symmetric matrices

Real symmetric matrices have a number of remarkable properties concerning eigenvalues. Among the key properties is the fact that they are always diagonalizable, their eigenvalues are real, and the eigenvectors can be chosen to be mutually orthogonal.

First, let us look at the generalization of transposition and symmetric matrices for complex matrices.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 10.49 — Hermitian matrix and conjugate transpose)</span></p>

The *conjugate transpose* (Hermitian transpose) of a matrix $A \in \mathbb{C}^{n \times n}$ is the matrix $A^* \coloneqq \overline{A}^\top$. A matrix $A \in \mathbb{C}^{n \times n}$ is called *Hermitian* if $A^* = A$.

</div>

The conjugate transpose has similar properties to the classical transpose, e.g., $(A^\ast)^\ast = A$, $(\alpha A)^\ast = \overline{\alpha} A^\ast$, $(A + B)^\ast = A^\ast + B^\ast$, $(AB)^\ast = B^\ast A^\ast$.

Using the conjugate transpose, we can define *unitary matrices* (extending the concept of orthogonal matrices to complex matrices) as matrices $Q \in \mathbb{C}^{n \times n}$ satisfying $Q^\ast Q = I_n$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.50)</span></p>

Of the matrices $\begin{pmatrix} 2 & 1 + i \\ 1 + i & 5 \end{pmatrix}$ and $\begin{pmatrix} 2 & 1 + i \\ 1 - i & 5 \end{pmatrix}$, the first is symmetric but not Hermitian, and the second is Hermitian but not symmetric. For real matrices, both concepts coincide.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 10.51 — Eigenvalues of symmetric matrices)</span></p>

The eigenvalues of real symmetric (or more generally complex Hermitian) matrices are real.

</div>

*Proof.* Let $A \in \mathbb{C}^{n \times n}$ be Hermitian and let $\lambda \in \mathbb{C}$ be any eigenvalue of $A$ and $x \in \mathbb{C}^n$ the corresponding eigenvector of unit size, i.e., $\|x\|_2 = 1$. Multiplying the equation $Ax = \lambda x$ by $x^\ast$ we have $x^\ast Ax = \lambda x^\ast x = \lambda$. Now

$$\lambda = x^* Ax = x^* A^* x = (x^* Ax)^* = \lambda^*.$$

Thus $\lambda = \lambda^\ast$, and therefore $\lambda$ must be real.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.52 — Eigenvalues of a projection matrix)</span></p>

Let $P \in \mathbb{R}^{n \times n}$ be the projection matrix onto a subspace $U$ of dimension $d$. For every vector $x \in U$ we have $Px = x$. Therefore 1 is an eigenvalue and it has $d$ eigenvectors from a basis of the space $U$. For every vector $x \in U^\perp$ we have $Px = o$. Thus 0 is an eigenvalue and it has $n - d$ eigenvectors from a basis of the space $U^\perp$. The matrix $P$ has no other eigenvalues, since we have found $n$ linearly independent eigenvectors. In summary, a projection matrix has only eigenvalues 0 and 1.

</div>

The following theorem says that symmetric matrices are diagonalizable. Moreover, they are diagonalizable in a specific way: the eigenvectors can be chosen to form an orthonormal system, meaning the similarity matrix is orthogonal.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 10.53 — Spectral decomposition of symmetric matrices)</span></p>

For every symmetric matrix $A \in \mathbb{R}^{n \times n}$ there exist an orthogonal $Q \in \mathbb{R}^{n \times n}$ and a diagonal $\Lambda \in \mathbb{R}^{n \times n}$ such that

$$A = Q \Lambda Q^\top$$

</div>

*Proof.* By mathematical induction on $n$. The case $n = 1$ is trivial: $\Lambda = A$, $Q = 1$. Induction step $n \leftarrow n - 1$. Let $\lambda$ be an eigenvalue of $A$ and $x$ the corresponding eigenvector normalized so that $\|x\|_2 = 1$. Extend $x$, as an orthonormal system, to an orthogonal matrix $S \coloneqq (x \mid \cdots)$. Since $(A - \lambda I_n)x = o$, we have $(A - \lambda I_n)S = (o \mid \cdots)$, and therefore $S^\top(A - \lambda I_n)S = S^\top(o \mid \cdots) = (o \mid \cdots)$. And since this matrix is symmetric, we have

$$S^\top(A - \lambda I_n)S = \begin{pmatrix} 0 & o^\top \\ o & A' \end{pmatrix},$$

where $A'$ is some symmetric matrix of order $n - 1$. By the induction hypothesis, it has the spectral decomposition $A' = Q' \Lambda' Q'^\top$, where $\Lambda'$ is diagonal and $Q'$ is orthogonal. We extend the matrices and the equality by one row and column, and finally obtain the desired decomposition $A = Q \Lambda Q^\top$, where $Q \coloneqq SR$ is an orthogonal matrix and $\Lambda \coloneqq \Lambda'' + \lambda I_n$ is diagonal.

Similarly, we can spectrally decompose Hermitian matrices as $A = Q \Lambda Q^*$, where $Q$ is a unitary matrix.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 10.54 — Alternative form of the spectral decomposition)</span></p>

Let a symmetric $A \in \mathbb{R}^{n \times n}$ have eigenvalues $\lambda_1, \ldots, \lambda_n$ and corresponding orthonormal eigenvectors $x_1, \ldots, x_n$. Thus in the spectral decomposition $A = Q \Lambda Q^\top$ we have $\Lambda_{ii} = \lambda_i$ and $Q_{*i} = x_i$. If we decompose $\Lambda$ as a sum of simpler diagonal matrices

$$\Lambda = \sum_{i=1}^n \lambda_i e_i e_i^\top,$$

then the matrix $A$ can be expressed as

$$A = Q \Lambda Q^\top = Q \left( \sum_{i=1}^n \lambda_i e_i e_i^\top \right) Q^\top = \sum_{i=1}^n \lambda_i Q e_i e_i^\top Q^\top = \sum_{i=1}^n \lambda_i Q_{*i} Q_{*i}^\top = \sum_{i=1}^n \lambda_i x_i x_i^\top.$$

The form $A = \sum_{i=1}^n \lambda_i x_i x_i^\top$ is thus an alternative expression of the spectral decomposition, in which we decompose the matrix $A$ as a sum of $n$ matrices of rank 0 or 1. Moreover, $x_i x_i^\top$ is the projection matrix onto the line $\operatorname{span}\lbrace x_i \rbrace$, so from a geometric perspective we can view the map $x \mapsto Ax$ as a sum of $n$ maps, where in each one we project onto a line (orthogonal to the others) and scale by the value $\lambda_i$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 10.55 — Courant--Fischer)</span></p>

Let $\lambda_1 \ge \cdots \ge \lambda_n$ be the eigenvalues of a symmetric matrix $A \in \mathbb{R}^{n \times n}$. Then

$$\lambda_1 = \max_{x: \|x\|_2 = 1} x^\top A x, \quad \lambda_n = \min_{x: \|x\|_2 = 1} x^\top A x.$$

</div>

*Proof.* Only for $\lambda_1$; the second part is analogous. Inequality "$\le$": Let $x_1$ be the eigenvector corresponding to $\lambda_1$ normalized so that $\|x_1\|_2 = 1$. Then $Ax_1 = \lambda_1 x_1$. Multiplying by $x_1^\top$ from the left we get $\lambda_1 = \lambda_1 x_1^\top x_1 = x_1^\top A x_1 \le \max_{x: \|x\|_2 = 1} x^\top Ax$. Inequality "$\ge$": Let $x \in \mathbb{R}^n$ be any vector such that $\|x\|_2 = 1$. Denote $y \coloneqq Q^\top x$, then $\|y\|_2 = 1$. Using the spectral decomposition $A = Q\Lambda Q^\top$ we get

$$x^\top Ax = x^\top Q \Lambda Q^\top x = y^\top \Lambda y = \sum_{i=1}^n \lambda_i y_i^2 \le \sum_{i=1}^n \lambda_1 y_i^2 = \lambda_1 \|y\|_2^2 = \lambda_1.$$

### Theory of nonnegative matrices

The Perron--Frobenius theory of nonnegative matrices is an advanced theory concerning eigenvalues of nonnegative matrices. We will only state the basic Perron result without proof. A matrix $A \in \mathbb{R}^{n \times n}$ is called *nonnegative* if it is nonnegative in every entry ($a_{ij} \ge 0$ for all $i, j$), and is called *positive* if it is positive in every entry ($a_{ij} > 0$ for all $i, j$).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 10.56 — Perron's theorem)</span></p>

1. Let $A \in \mathbb{R}^{n \times n}$ be a nonnegative matrix. Then the largest eigenvalue in absolute value is real and nonnegative, and the corresponding eigenvector is nonnegative (in all entries).
2. Let $A \in \mathbb{R}^{n \times n}$ be a positive matrix. Then the largest eigenvalue in absolute value is real and positive, it is unique (the others have smaller absolute value), it has multiplicity 1, and the corresponding eigenvector is positive (in all entries). Moreover, no other eigenvalue has a nonnegative eigenvector.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.57 — Markov chains)</span></p>

One application of matrix powers (Example 10.46) and, to some extent, of the theory of nonnegative matrices is Markov chains. Let $x \in \mathbb{R}^n$ be a state vector, where $x_i$ gives the value of state $i$. Let $A \in \mathbb{R}^{n \times n}$ be a matrix with values $a_{ij} \in [0, 1]$ such that the sum of values in each column equals 1. We view the matrix $A$ as a transition matrix, that is, the value $a_{ij}$ is the probability of transition from state $j$ to state $i$. Then $Ax$ gives the new state vector after one step of the process. We are interested in how the state vector evolves over time and whether it stabilizes.

Directly from the definition, $A^\top e = e$, where $e = (1, \ldots, 1)^\top$. Therefore $e$ is an eigenvector of $A^\top$ and 1 is an eigenvalue of $A^\top$ (and hence also of $A$). Moreover, it can be shown that no other eigenvalue is larger in absolute value (see Remark 10.60), so 1 is the eigenvalue of $A$ from Perron's theorem and the corresponding eigenvector is nonnegative.

A specific example: Migration of US residents in the city--suburb--rural sectors occurs annually according to the following pattern:

| | from city | from suburb | from rural |
|---|---|---|---|
| stays in city | 96% | 1% | 1.5% |
| to suburb | 3% | 98% | 0.5% |
| to rural | 1% | 1% | 98% |

Initial state: 58 million residents in the city, 142 million in suburbs, and 60 million in rural areas. Denote

$$A \coloneqq \begin{pmatrix} 0.96 & 0.01 & 0.015 \\ 0.03 & 0.98 & 0.005 \\ 0.01 & 0.01 & 0.98 \end{pmatrix}, \quad x_0 = (58, 142, 60)^\top.$$

By diagonalization we compute $A^\infty x_0 = (0.23 e^\top x_0, \; 0.43 e^ x_0, \; 0.33 e^\top x_0)^\top$. Thus (regardless of the initial state $x_0$) the population distribution stabilizes at: 23% in the city, 43% in suburbs, 33% in rural areas.

</div>

### Computing eigenvalues

As we already mentioned, eigenvalues are computed only by numerical iterative methods and finding them as roots of the characteristic polynomial is not an efficient approach. In this section we will show a simple estimate for eigenvalues and a simple method for computing the largest eigenvalue.

Since numerical methods are iterative and compute eigenvalues only with a certain precision, it is difficult to express a priori the exact number of operations they will perform. Nevertheless, current methods for both symmetric and non-symmetric matrices have practically cubic complexity. This means they require asymptotically $\alpha n^3$ operations, where $n$ is the dimension of the matrix and $\alpha > 0$ is the corresponding coefficient.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 10.58 — Gershgorin discs)</span></p>

Every eigenvalue $\lambda$ of a matrix $A \in \mathbb{C}^{n \times n}$ lies in a disc centered at $a_{ii}$ with radius $\sum_{j \neq i} |a_{ij}|$ for some $i \in \lbrace 1, \ldots, n \rbrace$.

</div>

*Proof.* Let $\lambda$ be an eigenvalue and $x$ the corresponding eigenvector, so $Ax = \lambda x$. Let the $i$-th component of $x$ have the largest absolute value, i.e., $|x_i| = \max_{k=1,\ldots,n} |x_k|$. Since the $i$-th equation has the form $\sum_{j=1}^n a_{ij} x_j = \lambda x_i$, dividing by $x_i \neq 0$ we get $\lambda = a_{ii} + \sum_{j \neq i} a_{ij} \frac{x_j}{x_i}$, and therefore $|\lambda - a_{ii}| = \left| \sum_{j \neq i} a_{ij} \frac{x_j}{x_i} \right| \le \sum_{j \neq i} |a_{ij}| \frac{|x_j|}{|x_i|} \le \sum_{j \neq i} |a_{ij}|$.

The theorem gives a simple but coarse estimate on the magnitude of eigenvalues (there also exist improvements, e.g., Cassini ovals, etc.). Nevertheless, in some applications such an estimate may suffice.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.59)</span></p>

Consider $A = \begin{pmatrix} 2 & 1 & 0 \\ -2 & 5 & 1 \\ -1 & -2 & -3 \end{pmatrix}$. The eigenvalues of $A$ are $\lambda_1 = -2.78$, $\lambda_2 = 3.39 + 0.6i$, $\lambda_3 = 3.39 - 0.6i$. The Gershgorin discs are: the disc centered at 2 with radius 1, the disc centered at 5 with radius 3, and the disc centered at $-3$ with radius 3. All eigenvalues lie inside one of the discs.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 10.60 — Three applications of Gershgorin discs)</span></p>

1. *Stopping criterion for iterative methods.* For example, the Jacobi method for computing eigenvalues consists of gradually reducing the off-diagonal entries of a symmetric matrix so that the matrix converges to a diagonal matrix. The Gershgorin discs then give an upper bound on the accuracy of the computed eigenvalues. If, for instance, a matrix $A \in \mathbb{R}^{n \times n}$ is nearly diagonal in the sense that all off-diagonal entries are less than $10^{-k}$ for some $k \in \mathbb{N}$, then the diagonal entries approximate the eigenvalues with accuracy $10^{-k}(n-1)$.
2. *Diagonally dominant matrices.* Gershgorin discs also give the following sufficient condition for the nonsingularity of a matrix $A \in \mathbb{C}^{n \times n}$: $\lvert a_{ii}\rvert > \sum_{j \neq i} \lvert a_{ij}\rvert$ $\forall i = 1, \ldots, n$. In this case the discs do not contain the origin, and therefore zero is not an eigenvalue of $A$. Matrices with this property are called diagonally dominant.
3. *Markov matrices.* Let $A$ be the Markov matrix from Example 10.57. All Gershgorin discs of the matrix $A^\top$ have their center at a point in the interval $[0, 1]$ and their right edge touches the value 1 on the real axis. This proves that $\rho(A) \le 1$, and therefore 1 is indeed the largest eigenvalue of the matrix $A$ in absolute value.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Algorithm 10.61 — Power method)</span></p>

Input: matrix $A \in \mathbb{C}^{n \times n}$.

1. Choose $o \neq x_0 \in \mathbb{C}^n$, $i \coloneqq 1$,
2. **while not** stopping criterion satisfied **do**
3. &emsp; $y_i \coloneqq A x_{i-1}$,
4. &emsp; $x_i \coloneqq \frac{1}{\|y_i\|_2} y_i$,
5. &emsp; $i \coloneqq i + 1$,
6. **end while**

Output: $\lambda_1 \coloneqq x_{i-1}^\top y_i$ is an estimate of the eigenvalue, $v_1 \coloneqq x_i$ is an estimate of the corresponding eigenvector.

</div>

The method terminates when the value $x_{i-1}^\top y_i$ or the vector $x_i$ stabilizes; then $x_i \approx x_{i-1}$ is an estimate of the eigenvector and $x_{i-1}^\top y_i = x_{i-1}^\top A x_{i-1} \approx x_{i-1}^\top \lambda x_{i-1} \approx \lambda$ is an estimate of the corresponding eigenvalue. The method can be slow, the error and convergence rate are difficult to estimate, and furthermore the initial choice of $x_0$ matters significantly. On the other hand, it is robust (rounding errors have little effect) and easily applicable to large sparse matrices. It does not always converge, but under certain assumptions convergence can be guaranteed.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 10.63 — Convergence of the power method)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ with eigenvalues $|\lambda_1| > |\lambda_2| \ge \ldots \ge |\lambda_n|$ and corresponding linearly independent eigenvectors $v_1, \ldots, v_n$ of unit size. Let $x_0$ have a nonzero component in the direction of $v_1$. Then $x_i$ converges (up to a scalar multiple) to the eigenvector $v_1$ and $x_{i-1}^\top y_i$ converges to the eigenvalue $\lambda_1$.

</div>

*Proof.* Since the vectors $v_1, \ldots, v_n$ form a basis of $\mathbb{R}^n$, we can express the vector $x_0$ as $x_0 = \sum_{j=1}^n \alpha_j v_j$, where $\alpha_1 \neq 0$ by assumption. Then $A^i x_0 = \sum_{j=1}^n \alpha_j \lambda_j^i v_j = \lambda_1^i \left( \alpha_1 v_1 + \sum_{j \neq 1} \alpha_j \left(\frac{\lambda_j}{\lambda_1}\right)^i v_j \right)$. Since the vectors $x_i$ are successively normalized, the factor $\lambda_1^i$ does not matter. The remaining vector gradually converges to $\alpha_1 v_1$, because $\left|\frac{\lambda_j}{\lambda_1}\right| < 1$ and therefore $\left|\frac{\lambda_j}{\lambda_1}\right|^i \to 0$ as $i \to \infty$.

From the proof we see that the convergence rate depends strongly on the ratio $\left|\frac{\lambda_2}{\lambda_1}\right|$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 10.64 — On eigenvalue deflation)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ be symmetric, $\lambda_1, \ldots, \lambda_n$ its eigenvalues, and $v_1, \ldots, v_n$ the corresponding orthonormal eigenvectors. Then the matrix $A - \lambda_1 v_1 v_1^\top$ has eigenvalues $0, \lambda_2, \ldots, \lambda_n$ and eigenvectors $v_1, \ldots, v_n$.

</div>

*Proof.* By Remark 10.54 we can write $A = \sum_{i=1}^n \lambda_i v_i v_i^\top$. Then $A - \lambda_1 v_1 v_1^\top = 0 v_1 v_1^\top + \sum_{i=2}^n \lambda_i v_i v_i^\top$, which is the spectral decomposition of the matrix $A - \lambda_1 v_1 v_1^\top$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 10.65 — On eigenvalue deflation for a general matrix)</span></p>

Let $\lambda$ be an eigenvalue and $x$ the corresponding eigenvector of a matrix $A \in \mathbb{R}^{n \times n}$. Extend $x$ to a nonsingular matrix $S$ so that $S_{*1} = x$. Then

$$S^{-1}AS = S^{-1}A(x \mid \cdots) = S^{-1}(\lambda x \mid \cdots) = (\lambda e_1 \mid \cdots) = \begin{pmatrix} \lambda & \cdots \\ o & A' \end{pmatrix}.$$

By similarity, the matrix $A'$ has the same eigenvalues as $A$, only $\lambda$ has multiplicity reduced by one. Therefore the remaining eigenvalues of matrix $A$ can be found using $A'$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 10.66 — Google search engine and PageRank)</span></p>

Consider a web network with $N$ web pages. The goal is to determine the importances $x_1, \ldots, x_N$ of individual pages. The basic idea of the authors of Google's PageRank is to set the importance of the $i$-th page to be proportional to the sum of the importances of the pages linking to it. We therefore solve the equation $x_i = \sum_{j=1}^N \frac{a_{ij}}{b_j} x_j$, $i = 1, \ldots, N$, where $a_{ij} = 1$ if the $j$-th page links to the $i$-th page (otherwise 0) and $b_j$ is the number of links from the $j$-th page. In matrix form $A'x = x$, where $a'_{ij} \coloneqq \frac{a_{ij}}{b_j}$.

Thus $x$ is an eigenvector of the matrix $A'$ corresponding to eigenvalue 1. The eigenvalue 1 is dominant, which is easily seen from the Gershgorin discs for the matrix $A'^\top$ (the column sums of $A'$ equal 1, so all Gershgorin discs have their rightmost point at 1). By Perron's theorem 10.56, the eigenvector $x$ is nonnegative.

In practice, the matrix $A'$ is huge, on the order of $\approx 10^{10}$, and at the same time sparse (most values are zeros). Therefore the power method is well-suited for computing $x$, requiring approximately $\approx 100$ iterations. In practice, the matrix $A'$ is also slightly modified to make it stochastic, aperiodic, and irreducible.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 10.67 — Further applications in graph theory)</span></p>

In conclusion, let us mention the broad use of eigenvalues in graph theory. The eigenvalues of the adjacency matrix and the Laplacian matrix of a graph reveal much about the structure of the graph. They are used to estimate the size of the so-called "bottleneck" in a graph, which is a set of vertices with relatively few outgoing edges. They also provide various estimates on the size of independent sets in graphs and other characteristics.

</div>

### Summary of Chapter 10

Eigenvalues and eigenvectors of a matrix $A$ provide essential information about the matrix and the linear map $x \mapsto Ax$. Geometrically, eigenvectors represent invariant directions that map to themselves, and eigenvalues represent the scaling factor in these directions. Eigenvalues thus describe quite well how much the linear map $x \mapsto Ax$ degenerates objects and what happens when the map is iterated.

A matrix $A \in \mathbb{C}^{n \times n}$ has exactly $n$ eigenvalues counting multiplicities, and at most $n$ (linearly independent) eigenvectors. If an eigenvalue has multiplicity $k$, then it has at most $k$ eigenvectors. Distinct eigenvalues have linearly independent eigenvectors.

Elementary operations change the eigenvalues of a matrix, but a transformation that preserves them is similarity. Geometrically, similarity means only a change of coordinate system. Every matrix is similar to a matrix in a simpler form -- the Jordan normal form. This can even take the form of a diagonal matrix (in which case the original matrix is diagonalizable). This happens if and only if the matrix has a full set of eigenvectors (the number of Jordan blocks equals the number of eigenvectors).

An important class of matrices are symmetric matrices. They have three essential properties: (1) they are always diagonalizable, (2) they have real eigenvalues, (3) the eigenvectors can be chosen to be mutually orthogonal. The corresponding spectral decomposition is then a very useful tool.

Another class of matrices with special properties are nonnegative matrices: the largest eigenvalue in absolute value lies on the real axis to the right of the origin, and the corresponding eigenvector is nonnegative.

The problem of computing eigenvalues and the problem of computing roots of polynomials are mutually reducible (via the characteristic polynomial and the companion matrix). Gaussian elimination cannot be simply used for computing eigenvalues -- all methods in use are iterative, such as the power method. Various estimates, such as the Gershgorin discs, are also useful.

## Chapter 11 — Positive (Semi-)Definite Matrices

Already in Theorem 10.55 we encountered the function $f \colon \mathbb{R}^n \to \mathbb{R}$ given by $f(x) = x^\top Ax = \sum_{i=1}^n \sum_{j=1}^n a_{ij} x_i x_j$, where $A \in \mathbb{R}^{n \times n}$ is a fixed matrix. This function represents a polynomial in the variables $x_1, \ldots, x_n$ and we will analyze it further in Chapter 12. Here we focus on the situation when the function $f(x)$ is nonnegative or positive, and for which matrices this is satisfied.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 11.1 — Positive (semi-)definite matrix)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ be symmetric. Then $A$ is *positive semidefinite* if $x^\top Ax \ge 0$ for all $x \in \mathbb{R}^n$, and $A$ is *positive definite* if $x^\top Ax > 0$ for all $x \neq o$.

</div>

Clearly, if $A$ is positive definite, then it is also positive semidefinite.

Positive definiteness and semidefiniteness need not be tested for all vectors $x \in \mathbb{R}^n$; it suffices to restrict, for example, to the unit sphere. If $x^\top Ax > 0$ for all vectors $x$ with unit norm $\|x\|_2 = 1$, then this holds for all other nonzero vectors as well. Indeed, every vector $x \neq o$ is a positive multiple of a unit-length vector, specifically the $\|x\|_2$-multiple of the vector $\frac{1}{\|x\|_2} x$.

Besides positive (semi-)definite matrices, one can also introduce negative (semi-)definite matrices using the reversed inequality. We will not deal with them, since $A$ is negative (semi-)definite if and only if $-A$ is positive (semi-)definite, so everything reduces to the basic case.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 11.2)</span></p>

The definition also makes sense for non-symmetric matrices, but these can easily be symmetrized by the transformation $\frac{1}{2}(A + A^\top)$, since

$$x^\top \tfrac{1}{2}(A + A^\top)x = \tfrac{1}{2}x^\top Ax + \tfrac{1}{2}x^\top A^\top x = \tfrac{1}{2}x^\top Ax + \left(\tfrac{1}{2}x^\top Ax\right)^\top = x^\top Ax.$$

Thus, for testing the condition one can equivalently use the symmetric matrix $\frac{1}{2}(A + A^\top)$. The restriction to symmetric matrices is therefore without loss of generality. The reason we restrict to symmetric matrices is that many of the testing conditions work only for symmetric matrices.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 11.3)</span></p>

An example of a positive semidefinite matrix is $0_n$. An example of a positive definite matrix is $I_n$, since $x^\top I_n x = x^\top x = \|x\|_2^2 > 0$ for all $x \neq o$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 11.4 — Necessary condition for positive (semi-)definiteness)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ be a symmetric matrix. For it to be positive semidefinite, by definition $x^\top Ax \ge 0$ must hold for all $x \in \mathbb{R}^n$. Substituting successively $x = e_i$, $i = 1, \ldots, n$, we obtain $x^\top Ax = e_i^\top A e_i = a_{ii} \ge 0$. Therefore, a positive semidefinite matrix must have a nonnegative diagonal, and a positive definite matrix must even have a positive diagonal.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 11.5)</span></p>

The matrix $A = (a) \in \mathbb{R}^{1 \times 1}$ is positive semidefinite if and only if $a \ge 0$, and positive definite if and only if $a > 0$. Therefore, we can view positive semidefiniteness as a generalization of the concept of nonnegativity from numbers to matrices. This is also why positive semidefiniteness of a matrix $A \in \mathbb{R}^{n \times n}$ is denoted $A \succeq 0$ (as opposed to $A \ge 0$, which is used for entrywise nonnegativity).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 11.6 — Properties of positive definite matrices)</span></p>

1. If $A, B \in \mathbb{R}^{n \times n}$ are positive definite, then $A + B$ is also positive definite,
2. If $A \in \mathbb{R}^{n \times n}$ is positive definite and $\alpha > 0$, then $\alpha A$ is also positive definite,
3. If $A \in \mathbb{R}^{n \times n}$ is positive definite, then it is nonsingular and $A^{-1}$ is positive definite.

</div>

*Proof.* The first two properties are trivial; we only prove the third one. First we verify the nonsingularity of $A$. Let $x$ be a solution of the system $Ax = o$. Then $x^\top Ax = x^\top o = 0$. By assumption, $x = o$. Now we show positive definiteness. By contradiction, suppose there exists $x \neq o$ such that $x^\top A^{-1}x \le 0$. Then $x^\top A^{-1}x = x^\top A^{-1}AA^{-1}x = y^\top Ay \le 0$, where $y = A^{-1}x \neq o$. This is a contradiction, since $A$ is positive definite.

The analogue of the proposition also holds for positive semidefinite matrices. Part (1) holds without change, part (2) holds for all $\alpha \ge 0$, but part (3) no longer holds in general, since a positive semidefinite matrix can be singular.

The product of positive definite matrices is discussed in Remark 12.20; we have already implicitly encountered this expression, for example in the method of least squares (Section 8.5) in the system of normal equations $A^\top Ax = A^\top b$, or more generally in the explicit expression for orthogonal projection in $\mathbb{R}^n$ (Theorem 8.49).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 11.7 — Characterization of positive definiteness)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ be symmetric. Then the following conditions are equivalent:

1. $A$ is positive definite,
2. the eigenvalues of $A$ are positive,
3. there exists a matrix $U \in \mathbb{R}^{m \times n}$ of rank $n$ such that $A = U^\top U$.

</div>

*Proof.* Implication (1) $\Rightarrow$ (2): By contradiction, suppose there exists an eigenvalue $\lambda \le 0$, and let $x$ be the corresponding eigenvector with Euclidean norm equal to 1. Then $Ax = \lambda x$ implies $x^\top Ax = \lambda x^\top x = \lambda \le 0$. This contradicts the positive definiteness of $A$.

Implication (2) $\Rightarrow$ (3): Since $A$ is symmetric, it has a spectral decomposition $A = Q\Lambda Q^\top$, where $\Lambda$ is a diagonal matrix with entries $\lambda_1, \ldots, \lambda_n > 0$. Define the matrix $\Lambda'$ as diagonal with entries $\sqrt{\lambda_1}, \ldots, \sqrt{\lambda_n} > 0$. Then the desired matrix is, for example, $U = \Lambda' Q^\top$, since $U^\top U = Q\Lambda' \Lambda' Q^\top = Q\Lambda'^2 Q^\top = Q\Lambda Q^\top = A$. Note that $U$ has rank $n$ and is therefore nonsingular, being the product of two nonsingular matrices.

Implication (3) $\Rightarrow$ (1): By contradiction, suppose $x^\top Ax \le 0$ for some $x \neq o$. Then $0 \ge x^\top Ax = x^\top U^\top Ux = (Ux)^\top Ux = \|Ux\|_2^2$. Hence $Ux = o$, but the columns of $U$ are linearly independent, so $x = o$, a contradiction.

For positive semidefiniteness we have the following characterization (the proof is analogous):

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 11.8 — Characterization of positive semidefiniteness)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ be symmetric. Then the following conditions are equivalent:

1. $A$ is positive semidefinite,
2. the eigenvalues of $A$ are nonnegative,
3. there exists a matrix $U \in \mathbb{R}^{m \times n}$ such that $A = U^\top U$.

</div>

### Methods for testing positive definiteness

We now focus on specific methods for testing positive definiteness. Many of them are based on the following recurrence relation.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 11.9 — Recurrence formula for testing positive definiteness)</span></p>

A symmetric matrix $A = \begin{pmatrix} \alpha & a^\top \\ a & \tilde{A} \end{pmatrix}$, where $\alpha \in \mathbb{R}$, $a \in \mathbb{R}^{n-1}$, $\tilde{A} \in \mathbb{R}^{(n-1) \times (n-1)}$, is positive definite if and only if $\alpha > 0$ and $\tilde{A} - \frac{1}{\alpha}aa^\top$ is positive definite.

</div>

*Proof.* Implication "$\Rightarrow$": Let $A$ be positive definite. Then $x^\top Ax > 0$ for all $x \neq o$, so in particular for $x = e_1$ we get $\alpha = e_1^\top A e_1 > 0$. Next, let $\tilde{x} \in \mathbb{R}^{n-1}$, $\tilde{x} \neq o$. Then

$$\tilde{x}^\top \left(\tilde{A} - \tfrac{1}{\alpha}aa^\top\right)\tilde{x} = \tilde{x}^\top \tilde{A}\tilde{x} - \tfrac{1}{\alpha}(a^\top\tilde{x})^2 = \begin{pmatrix} -\tfrac{1}{\alpha}a^\top\tilde{x} & \tilde{x}^\top \end{pmatrix} \begin{pmatrix} \alpha & a^\top \\ a & \tilde{A} \end{pmatrix} \begin{pmatrix} -\tfrac{1}{\alpha}a^\top\tilde{x} \\ \tilde{x} \end{pmatrix} > 0.$$

Implication "$\Leftarrow$": Let $x = \begin{pmatrix} \beta \\ \tilde{x} \end{pmatrix} \in \mathbb{R}^n$. Then

$$x^\top Ax = \begin{pmatrix} \beta & \tilde{x}^\top \end{pmatrix} \begin{pmatrix} \alpha & a^\top \\ a & \tilde{A} \end{pmatrix} \begin{pmatrix} \beta \\ \tilde{x} \end{pmatrix} = \alpha\beta^2 + 2\beta a^\top\tilde{x} + \tilde{x}^\top\tilde{A}\tilde{x} = \tilde{x}^\top(\tilde{A} - \tfrac{1}{\alpha}aa^\top)\tilde{x} + \left(\sqrt{\alpha}\beta + \tfrac{1}{\sqrt{\alpha}}a^\top\tilde{x}\right)^2 \ge 0.$$

Equality holds only when $\tilde{x} = o$ and the second square is zero, i.e., $\beta = 0$.

Although the recurrence formula can be used for testing positive definiteness, a more important role is played by the following Cholesky decomposition.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 11.10 — Cholesky decomposition)</span></p>

For every positive definite matrix $A \in \mathbb{R}^{n \times n}$, there exists a unique lower triangular matrix $L \in \mathbb{R}^{n \times n}$ with positive diagonal such that $A = LL^\top$.

</div>

*Proof.* By mathematical induction on $n$. For $n = 1$ we have $A = (a_{11})$ and $L = (\sqrt{a_{11}})$.

Inductive step $n \leftarrow n - 1$. Let $A = \begin{pmatrix} \alpha & a^\top \\ a & \tilde{A} \end{pmatrix}$. By Theorem 11.9, $\alpha > 0$ and $\tilde{A} - \frac{1}{\alpha}aa^\top$ is positive definite. By the inductive hypothesis, there exists a lower triangular matrix $\tilde{L} \in \mathbb{R}^{(n-1) \times (n-1)}$ with positive diagonal such that $\tilde{A} - \frac{1}{\alpha}aa^\top = \tilde{L}\tilde{L}^\top$. Then $L = \begin{pmatrix} \sqrt{\alpha} & o^\top \\ \frac{1}{\sqrt{\alpha}}a & \tilde{L} \end{pmatrix}$, since

$$LL^\top = \begin{pmatrix} \sqrt{\alpha} & o^\top \\ \frac{1}{\sqrt{\alpha}}a & \tilde{L} \end{pmatrix} \begin{pmatrix} \sqrt{\alpha} & \frac{1}{\sqrt{\alpha}}a^\top \\ o & \tilde{L}^\top \end{pmatrix} = \begin{pmatrix} \alpha & a^\top \\ a & \frac{1}{\alpha}aa^\top + \tilde{L}\tilde{L}^\top \end{pmatrix} = A.$$

The Cholesky decomposition also exists for positive semidefinite matrices, but it is not unique.

**Algorithm of the Cholesky decomposition.** Theorem 11.10 was primarily of an existential character; however, constructing the Cholesky decomposition is very simple. The basic idea is to start from the equation $A = LL^\top$ and successively compare elements from the top in the first column of the matrix on the left and right, then in the second column, etc.

Assume that we have computed the first $k - 1$ columns of the matrix $L$. From the relation $A = LL^\top$ we derive for the element at position $(k, k)$:

$$a_{kk} = \sum_{j=1}^{k} \ell_{kj}^2, \quad \text{tedy} \quad \ell_{kk} = \sqrt{a_{kk} - \sum_{j=1}^{k-1} \ell_{kj}^2}.$$

And for the element at position $(i, k)$, where $i > k$:

$$a_{ik} = \sum_{j=1}^{k} \ell_{ij}\ell_{kj}, \quad \text{tedy} \quad \ell_{ik} = \frac{1}{\ell_{kk}} \left(a_{ik} - \sum_{j=1}^{k-1} \ell_{ij}\ell_{kj}\right).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Algorithm 11.11 — Cholesky decomposition)</span></p>

Input: symmetric matrix $A \in \mathbb{R}^{n \times n}$.

1. $L \coloneqq 0_n$,
2. **for** $k \coloneqq 1$ **to** $n$ **do** &emsp;&emsp; // in the $k$-th iteration we determine the values $L_{*k}$
3. &emsp; **if** $a_{kk} - \sum_{j=1}^{k-1} \ell_{kj}^2 \le 0$ **then return** "$A$ is not positive definite",
4. &emsp; $\ell_{kk} \coloneqq \sqrt{a_{kk} - \sum_{j=1}^{k-1} \ell_{kj}^2}$,
5. &emsp; **for** $i \coloneqq k + 1$ **to** $n$ **do**
6. &emsp;&emsp; $\ell_{ik} \coloneqq \frac{1}{\ell_{kk}} \left(a_{ik} - \sum_{j=1}^{k-1} \ell_{ij}\ell_{kj}\right)$,
7. &emsp; **end for**
8. **end for**

Output: matrix $L$ satisfying $A = LL^\top$, or the information that $A$ is not positive definite.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 11.12)</span></p>

Cholesky decomposition of the matrix $A$:

$$\begin{pmatrix} 2 & 0 & 0 \\ -1 & 3 & 0 \\ 2 & 1 & 1 \end{pmatrix} \begin{pmatrix} 2 & -1 & 2 \\ 0 & 3 & 1 \\ 0 & 0 & 1 \end{pmatrix} = \begin{pmatrix} 4 & -2 & 4 \\ -2 & 10 & 1 \\ 4 & 1 & 6 \end{pmatrix}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 11.13 — Using the Cholesky decomposition for solving systems)</span></p>

Using the Cholesky decomposition for solving the system $Ax = b$ with a positive definite matrix $A$. If we have the decomposition $A = LL^\top$, then the system has the form $L(L^\top x) = b$. First we solve the system $Ly = b$ by forward substitution, then $L^\top x = y$ by back substitution. The procedure is thus as follows:

1. Find the Cholesky decomposition $A = LL^\top$.
2. Find the solution $y^*$ of the system $Ly = b$ by forward substitution.
3. Find the solution $x^*$ of the system $L^\top x = y^*$ by back substitution.

This procedure is roughly $50\%$ faster than solving by Gaussian elimination.

The Cholesky decomposition can also be used to invert positive definite matrices, since $A^{-1} = (LL^\top)^{-1} = (L^{-1})^\top L^{-1}$ and the inverse of the lower triangular matrix $L$ is easy to find.

</div>

The recurrence formula has further consequences that express how to test positive definiteness using Gauss-Jordan elimination and using determinants.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 11.14 — Gaussian elimination and positive definiteness)</span></p>

A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is positive definite if and only if Gaussian elimination reduces it to row echelon form with a positive diagonal using only the elementary operation of adding a multiple of the pivot row to another row below it.

</div>

*Proof.* Let $A = \begin{pmatrix} \alpha & a^\top \\ a & \tilde{A} \end{pmatrix}$ be positive definite. The first step of Gaussian elimination transforms the matrix to the form $\begin{pmatrix} \alpha & a^\top \\ o & \tilde{A} - \frac{1}{\alpha}aa^\top \end{pmatrix}$; it suffices to subtract the $\frac{1}{\alpha}a$-multiple of the first row from the second block row. By Theorem 11.9, $\alpha > 0$ and $\tilde{A} - \frac{1}{\alpha}aa^\top$ is again positive definite, so we can continue inductively.

Conversely, suppose that Gaussian elimination transforms the matrix $A$ into the required form. In the first step it again transforms it to the form $\begin{pmatrix} \alpha & a^\top \\ o & \tilde{A} - \frac{1}{\alpha}aa^\top \end{pmatrix}$, where $\alpha > 0$. By mathematical induction on the size of the matrix, we may assume that the matrix $\tilde{A} - \frac{1}{\alpha}aa^\top$ is positive definite. Therefore the matrix $A$ is also positive definite by Theorem 11.9.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 11.15 — Sylvester's criterion for positive definiteness)</span></p>

A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is positive definite if and only if the determinants of all leading principal submatrices $A_1, \ldots, A_n$ are positive, where $A_i$ is the upper-left submatrix of $A$ of size $i$ (i.e., obtained from $A$ by removing the last $n - i$ rows and columns).

</div>

*Proof.* Implication "$\Rightarrow$": Let $A$ be positive definite. Then for each $i = 1, \ldots, n$, $A_i$ is positive definite, since if $x^\top A_i x \le 0$ for some $x \neq o$, then $(x^\top \; o^\top) A \binom{x}{o} = x^\top A_i x \le 0$. Hence $A_i$ has positive eigenvalues and its determinant is also positive (it equals the product of the eigenvalues).

Implication "$\Leftarrow$": During Gaussian elimination of the matrix $A$, all pivots are positive, since if the $i$-th pivot is nonpositive, then $\det(A_i) \le 0$. By Proposition 11.14, $A$ is therefore positive definite.

Nonnegativity of the determinants of all leading principal submatrices does not yet imply positive semidefiniteness (find such an example!). The analogue of Sylvester's criterion for positive semidefinite matrices is as follows:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 11.16 — Sylvester's criterion for positive semidefiniteness)</span></p>

A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is positive semidefinite if and only if the determinants of all principal submatrices are nonnegative, where a principal submatrix is a matrix obtained from $A$ by removing a certain number (possibly zero) of rows and columns with the same indices.

</div>

*Proof.* If $A$ is positive semidefinite, then clearly the principal submatrices are also positive semidefinite, and therefore have nonnegative determinant ($=$ product of eigenvalues).

We prove the reverse implication by mathematical induction. For $n = 1$ the claim is obvious. Inductive step $n \leftarrow n - 1$. For contradiction, let $\lambda < 0$ be an eigenvalue of $A$, and let $x$ be the corresponding eigenvector normalized so that $\|x\|_2 = 1$. If all other eigenvalues are positive, then $\det(A) < 0$, and we are done. Otherwise, let $\mu \le 0$ be another eigenvalue of $A$ and let $y$, $\|y\|_2 = 1$, be the corresponding eigenvector. Since $x \perp y$, we now find $\alpha \in \mathbb{R}$ such that the vector $z \coloneqq x + \alpha y$ has at least one zero component; let it be the $i$-th. Then $z^\top Az = (x + \alpha y)^\top A(x + \alpha y) = \lambda x^\top x + \alpha^2 \mu y^\top y = \lambda + \alpha^2 \mu < 0$. Let $A'$ be obtained from $A$ by removing the $i$-th row and column, and let $z'$ be obtained from the vector $z$ by removing the $i$-th component. Then $z'^\top A' z' = z^\top Az < 0$, so the principal submatrix $A'$ is not positive semidefinite and we apply the inductive hypothesis.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 11.17 — Computational complexity)</span></p>

We compare the computational complexity of the individual methods for testing positive definiteness. By Remark 2.19, the asymptotic complexity of Gaussian elimination is $\frac{2}{3}n^3$. Computing the determinant has the same complexity, so Sylvester's criterion requires on the order of

$$\sum_{k=1}^{n} \frac{2}{3}k^3 = \frac{2}{3} \cdot \frac{1}{4} n^2(n+1)^2$$

operations, which is asymptotically $\frac{1}{6}n^4$. Sylvester's criterion is therefore unsuitable for practical use. The recurrence formula costs on the order of

$$\sum_{k=1}^{n} 2k^2 = \frac{2}{6}n(n+1)(2n+1)$$

operations, which gives the same complexity as for Gaussian elimination, i.e., $\frac{2}{3}n^3$. Finally, the Cholesky decomposition requires

$$\sum_{k=1}^{n} 2k + (n-k)2k = n(n+1) + n^2(n+1) - \frac{2}{6}n(n+1)(2n+1)$$

operations. Asymptotically it thus costs only $\frac{1}{3}n^3$ operations and is therefore the computationally best method.

</div>

Although we have presented several methods for testing positive definiteness, some of them are quite similar. The proof of Proposition 11.14 shows that the recurrence formula and Gaussian elimination work essentially the same way. And if we compute determinants via Gaussian elimination, then Sylvester's criterion is also a variant of the first two. In contrast, the Cholesky decomposition is a fundamentally different method.

### Applications

First we show that using positive definite matrices we can describe all possible inner products on the space $\mathbb{R}^n$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 11.18 — Inner product and positive definiteness)</span></p>

The operation $\langle x, y \rangle$ is an inner product on $\mathbb{R}^n$ if and only if it has the form $\langle x, y \rangle = x^\top Ay$ for some positive definite matrix $A \in \mathbb{R}^{n \times n}$.

</div>

*Proof.* Implication "$\Rightarrow$": Define the matrix $A \in \mathbb{R}^{n \times n}$ by $a_{ij} = \langle e_i, e_j \rangle$, where $e_i, e_j$ are the standard unit vectors. The matrix $A$ is obviously symmetric. Now, by linearity of the inner product in both the first and second argument, we can write

$$\langle x, y \rangle = \left\langle \sum_{i=1}^n x_i e_i, \sum_{j=1}^n y_j e_j \right\rangle = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \langle e_i, e_j \rangle = \sum_{i=1}^n \sum_{j=1}^n x_i y_j a_{ij} = x^\top Ay.$$

The matrix $A$ must be positive definite, since by the definition of the inner product $x^\top Ax = \langle x, x \rangle \ge 0$ and equals zero only for $x = o$.

Implication "$\Leftarrow$": Let $A$ be positive definite. Then $\langle x, y \rangle = x^\top Ay$ forms an inner product: $\langle x, x \rangle = x^\top Ax \ge 0$ and equals zero only for $x = o$, it is linear in the first argument, and it is symmetric since $\langle x, y \rangle = x^\top Ay = (x^\top Ay)^\top = y^\top A^\top x = y^\top Ax = \langle y, x \rangle$.

We know that an inner product induces a norm (Definition 8.5). The norm induced by the above inner product is $\|x\| = \sqrt{x^\top Ax}$. In this norm, the unit ball is an ellipsoid (see Example 12.22). For $A = I_n$ we obtain the standard inner product on $\mathbb{R}^n$ and the Euclidean norm.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 11.19)</span></p>

Although the nonstandard inner product $\langle x, y \rangle = x^\top Ay$ may look strange, its relationship to the standard one is very close. Since the matrix $A$ is positive definite, it can be decomposed as $A = R^\top R$, where $R$ is nonsingular. Let $B$ be the basis formed by the columns of the matrix $R^{-1}$, so $R = {}_B[id]_{\text{kan}}$ is the change-of-basis matrix from $B$ to the canonical basis. Now $x^\top Ay = x^\top R^\top R y = (Rx)^\top(Ry) = [x]_B^\top [y]_B$. This shows that the nonstandard inner product can be expressed as the standard inner product with respect to a certain basis.

</div>

Another application is the matrix square root. For positive semidefinite matrices we can introduce the positive semidefinite square root $\sqrt{A}$. The square root is even unique.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 11.20 — Matrix square root)</span></p>

For every positive semidefinite matrix $A \in \mathbb{R}^{n \times n}$, there exists a positive semidefinite matrix $B \in \mathbb{R}^{n \times n}$ such that $B^2 = A$.

</div>

*Proof.* Let $A$ have the spectral decomposition $A = Q\Lambda Q^\top$, where $\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$, $\lambda_1, \ldots, \lambda_n \ge 0$. Define the diagonal matrix $\Lambda' = \operatorname{diag}(\sqrt{\lambda_1}, \ldots, \sqrt{\lambda_n})$ and the matrix $B = Q\Lambda' Q^\top$. Then $B^2 = Q\Lambda' Q^\top Q\Lambda' Q^\top = Q\Lambda'^2 Q^\top = Q\Lambda Q^\top = A$.

It is appropriate here to compare the matrix square root with matrix functions from Example 10.47. The square root can be expressed by an infinite series only in a small neighborhood of a given positive number; however, where it exists, both definitions will agree.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 11.21 — Positive definiteness and optimization)</span></p>

Positive (semi-)definiteness appears in optimization when determining the minimum of a function $f \colon \mathbb{R}^n \to \mathbb{R}$. The matrix that arises here is the so-called Hessian, the matrix of second partial derivatives. Assuming we are at a point $x^* \in \mathbb{R}^n$ with zero gradient, positive definiteness gives a sufficient condition for $x^*$ to be a local minimum, while conversely positive semidefiniteness gives a necessary condition. This is a generalization of the one-dimensional case, where a real smooth function $f \colon \mathbb{R} \to \mathbb{R}$ has a local minimum at a point $x^* \in \mathbb{R}$ if its derivative at $a$ is zero and the second derivative is positive.

The Hessian is similarly used in determining the convexity of a function. Positive definiteness on some open convex set implies convexity of the function $f$.

Positive definite matrices play yet another important role in optimization. A semidefinite program is an optimization problem in which we seek the minimum of a linear function subject to the constraint that a matrix whose entries are linear functions of the variables is positive semidefinite. Formally, it is the problem

$$\min \; c^\top x \quad \text{subject to } A_0 + \sum_{i=1}^m A_i x_i \text{ being positive semidefinite},$$

where $c \in \mathbb{R}^m$, $A_0, A_1, \ldots, A_m \in \mathbb{R}^{n \times n}$ are given and $x = (x_1, \ldots, x_m)^\top$ is the vector of variables. Semidefinite programs can not only model a larger class of problems than linear programs, but are still solvable efficiently in reasonable time. They have enabled, among other things, great progress in the field of combinatorial optimization, since many computationally hard problems can be quickly and tightly approximated using suitable semidefinite programs.

</div>

The occurrence of positive (semi-)definite matrices is even broader. For example, in statistics we encounter the so-called covariance and correlation matrices. Both provide certain information about the dependence among $n$ random variables and, not coincidentally, are always positive semidefinite.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 11.22 — Determining protein structure)</span></p>

One of the fundamental tasks in protein modeling is determining the three-dimensional structure of proteins. A typical procedure is to determine the distance matrix of individual atoms using nuclear magnetic resonance and then derive the structure.

Let $X \in \mathbb{R}^{n \times 3}$ be the matrix of positions of individual atoms, that is, the row $X_{i*}$ gives the coordinates of the $i$-th atom in space. We denote by $D \in \mathbb{R}^{n \times n}$ the distances between individual atoms, so $d_{ij} = $ the distance between the $i$-th and $j$-th atom. If we know $X$, then we can compute $D$. We shift the coordinate system so that the $n$-th atom is at the origin, i.e., $X_{n*} = (0, 0, 0)$, and remove the last row from the matrix $X$, which is now redundant. Denote the auxiliary matrix $D^* \coloneqq XX^\top$ and the coordinates of any two atoms $u \coloneqq X_{i*}$, $v \coloneqq X_{j*}$. The relationship between the matrices $D$ and $D^*$ is as follows:

$$d_{ij}^2 = \|u - v\|^2 = \langle u - v, u - v \rangle = \langle u, u \rangle + \langle v, v \rangle - 2\langle u, v \rangle = d_{ii}^* + d_{jj}^* - 2d_{ij}^*.$$

Using this formula we compute the matrix $D^*$ from the matrix $D$:

$$d_{ij}^* = \frac{1}{2}(d_{ii}^* + d_{jj}^* - d_{ij}^2) = \frac{1}{2}(d_{in}^2 + d_{jn}^2 - d_{ij}^2).$$

Since $D^*$ is symmetric and positive semidefinite, from the spectral decomposition $D^* = Q\Lambda Q^\top$, where $\Lambda = \operatorname{diag}(\lambda_1, \lambda_2, \lambda_3)$, we construct the desired matrix $X = Q \cdot \operatorname{diag}(\sqrt{\lambda_1}, \sqrt{\lambda_2}, \sqrt{\lambda_3})$. Then indeed $D^* = XX^\top$.

Another problem from this field is the so-called *Procrustes problem*, in which we compare two protein structures to determine how similar they are. Denote the matrices of the two structures by $X, Y$. A linear mapping $f$ by an orthogonal matrix $Q$ preserves angles and distances, so $YQ$ corresponds to the same structure as $Y$, only rotated or reflected in some way. If we want to determine the similarity of both structures, we seek an orthogonal matrix $Q \in \mathbb{R}^{3 \times 3}$ such that the matrices $X$ and $YQ$ are "as close as possible." The mathematical formulation leads to the optimization problem of minimizing the matrix norm $\|X - YQ\|$ over the set of orthogonal matrices $Q$.

</div>

### Summary of Chapter 11

A positive definite matrix is a special type of matrix that nevertheless appears in various situations:

- every inner product on the space $\mathbb{R}^n$ has the form $\langle x, y \rangle = x^\top Ay$ for some positive definite matrix $A$,
- a function $f \colon \mathbb{R}^n \to \mathbb{R}$ is convex if its Hessian is positive definite,
- and others.

We define a positive definite matrix as a symmetric matrix $A$ for which the function $f(x) = x^\top Ax$ is nonnegative and zero only at the origin. Alternatively, we can characterize it as a matrix that has positive eigenvalues. Yet another way to view positive definite matrices is that they have a decomposition $A = U^\top U$, where $U$ is nonsingular. One can even require the matrix $U$ to be upper triangular with positive diagonal (and then it is also unique). This gives rise to an efficient method for testing positive definiteness, the so-called Cholesky decomposition (it is customary to write it in the form $A = LL^\top$, where $L$ is a lower triangular matrix). Moreover, the decomposition $A = U^\top U$ finds application in solving systems of linear equations and other computations with the matrix $A$.

## Chapter 12 — Quadratic Forms

The word *linear* in the expression "linear algebra" does not mean that the field deals only with linear objects such as lines and planes in geometric applications. In this chapter we take a closer look at quadratic forms. In principle, we have already encountered them with the Euclidean norm (p. 130) and positive definiteness (Definition 11.1). In this chapter we discuss quadratic forms in more detail. We show connections with positive (semi-)definiteness, the signs of eigenvalues, and the description of certain geometric objects.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 12.1 — Motivation for quadratic forms)</span></p>

To begin with, we can think of a quadratic form as a polynomial in $n$ variables where the sum of degrees of each term is exactly two. That is,

$$f(x) = 5x^2$$

is a quadratic form in one variable, and

$$f(x_1, x_2) = 5x_1^2 - 3x_1 x_2 + 12x_2^2$$

is a quadratic form in two variables. The general formula for such a polynomial in $n$ variables $x = (x_1, \ldots, x_n)^\top$ is

$$f(x) = \sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij} x_i x_j = x^\top A x,$$

where $A$ is the $n \times n$ matrix of the corresponding coefficients. We will frequently use the compact matrix notation $x^\top Ax$.

</div>

### 12.1 Bilinear and quadratic forms

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 12.2 — Bilinear and quadratic form)</span></p>

Let $V$ be a vector space over $\mathbb{T}$. A *bilinear form* is a mapping $b \colon V^2 \to \mathbb{T}$ that is linear in both the first and second argument, i.e.,

$$b(\alpha u + \beta v, w) = \alpha b(u, w) + \beta b(v, w), \quad \forall \alpha, \beta \in \mathbb{T}, \; \forall u, v, w \in V,$$

$$b(w, \alpha u + \beta v) = \alpha b(w, u) + \beta b(w, v), \quad \forall \alpha, \beta \in \mathbb{T}, \; \forall u, v, w \in V.$$

A bilinear form is called *symmetric* if $b(u, v) = b(v, u)$ for all $u, v \in V$. A mapping $f \colon V \to \mathbb{T}$ is a *quadratic form* if it can be expressed as $f(u) = b(u, u)$ for some symmetric bilinear form $b$.

It is easy to see that $b(o, v) = b(v, o) = 0$ and $f(o) = 0$ always hold.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 12.3)</span></p>

- Every real inner product is a bilinear form, so for any matrix $A \in \mathbb{R}^{n \times n}$ the mapping $b(x, y) = x^\top Ay$ is a bilinear form. For a symmetric matrix $A$, the mapping $f(x) = x^\top Ax$ is then a quadratic form.

  In particular, in the space $\mathbb{R}^1$ any mapping $b \colon \mathbb{R}^2 \to \mathbb{R}$ given by $b(x, y) = axy$, where $a \in \mathbb{R}$ is a constant, is a bilinear form. The corresponding quadratic form is then the quadratic function of one variable $f(x) = ax^2$. Quadratic forms on $\mathbb{R}^n$ can thus be understood as a generalization of the quadratic function from one to $n$ variables.

- The complex inner product is not a bilinear form, since it is not linear in the second argument. Forms of this type are called sesquilinear.

- Let $V = \mathbb{R}^2$. Then $b(x, y) = x_1 y_1 + 2x_1 y_2 + 4x_2 y_1 + 10x_2 y_2$ is an example of a bilinear form, $b'(x, y) = x_1 y_1 + 3x_1 y_2 + 3x_2 y_1 + 10x_2 y_2$ is an example of a symmetric bilinear form, and $f(x) = b'(x, x) = x_1^2 + 6x_1 x_2 + 10x_2^2$ is the corresponding quadratic form.

</div>

In this chapter we will primarily deal with quadratic forms, although bilinear forms are also interesting, for example precisely through their relationship with the inner product.

In analogy with the theory of linear mappings, bilinear forms are also uniquely determined by the images of bases and can be expressed in matrix form. We therefore recommend that the reader compare the concepts and results of this section with Section 6.2 and recognize a certain parallel.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 12.5 — Motivation for matrices of forms)</span></p>

Let $b \colon V^2 \to \mathbb{T}$ be a bilinear form and $B = \lbrace w_1, \ldots, w_n \rbrace$ a basis of the space $V$. Let $u, v \in V$ be arbitrary vectors with representations $u = \sum_{i=1}^n x_i w_i$, $v = \sum_{i=1}^n y_i w_i$ in basis $B$. From the definition of a bilinear form, the image of the vectors is

$$b(u, v) = b\!\left(\sum_{i=1}^n x_i w_i, \sum_{j=1}^n y_j w_j\right) = \sum_{i=1}^n \sum_{j=1}^n x_i y_j b(w_i, w_j).$$

We see that the entire bilinear form is in fact determined by the images of all pairs of basis vectors. This moreover motivates us to place these values $b(w_i, w_j)$ into a matrix and work with the matrix representation.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 12.6 — Matrix of a bilinear and quadratic form)</span></p>

Let $b \colon V^2 \to \mathbb{T}$ be a bilinear form and $B = \lbrace w_1, \ldots, w_n \rbrace$ a basis of the space $V$. We define the matrix $A \in \mathbb{T}^{n \times n}$ of the bilinear form with respect to basis $B$ by $a_{ij} = b(w_i, w_j)$. The matrix of a quadratic form $f \colon V \to \mathbb{T}$ is defined as the matrix of any symmetric bilinear form inducing $f$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 12.7 — Matrix representation of forms)</span></p>

Let $B$ be a basis of a vector space $V$ and let $b$ be a bilinear form on $V$. Then $A$ is the matrix of the form $b$ with respect to basis $B$ if and only if for every $u, v \in V$ we have

$$b(u, v) = [u]_B^\top A [v]_B.$$

Furthermore, if $b$ is a symmetric form, then the corresponding quadratic form $f$ satisfies for every $u \in V$

$$f(u) = [u]_B^\top A [u]_B.$$

</div>

*Proof.* Denote $x \coloneqq [u]_B$, $y \coloneqq [v]_B$, and let $B$ consist of vectors $w_1, \ldots, w_n$. If $A$ is the matrix of the form $b$, then

$$b(u, v) = b\!\left(\sum_{i=1}^n x_i w_i, \sum_{j=1}^n y_j w_j\right) = \sum_{i=1}^n \sum_{j=1}^n x_i y_j b(w_i, w_j) = \sum_{i=1}^n \sum_{j=1}^n x_i y_j a_{ij} = x^\top Ay.$$

Conversely, if (12.1) holds for every $u, v \in V$, then substituting $u \coloneqq w_i$, $v \coloneqq w_j$ we get $b(w_i, w_j) = [w_i]_B^\top A [w_j]_B = e_i^\top A e_j = a_{ij}$ for all $i, j = 1, \ldots, n$.
Finally, $f(u) = b(u, u) = x^\top Ax$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 12.8)</span></p>

Let $B = \lbrace w_1, \ldots, w_n \rbrace$ be a basis of a vector space $V$ over $\mathbb{T}$ and let $A \in \mathbb{T}^{n \times n}$. Then there exists a unique bilinear form $b \colon V^2 \to \mathbb{T}$ such that $b(w_i, w_j) = a_{ij}$ for all $i, j = 1, \ldots, n$.

</div>

*Proof.* "Existence." It suffices to verify that the mapping $b \colon V^2 \to \mathbb{T}$ given by $b(u, v) = [u]_B^\top A [v]_B$ satisfies the conditions of a bilinear form. This is easy to see, since the mapping $u \mapsto [u]_B$ is linear (cf. Proposition 6.38). "Uniqueness." From (12.1) it follows that for every $u, v \in V$ we have $b(u, v) = [u]_B^\top A [v]_B$, so the images are uniquely determined.

Let $B$ be a fixed basis of a space $V$ of dimension $n$. Each bilinear form thus uniquely corresponds to a matrix $A \in \mathbb{T}^{n \times n}$, and conversely each matrix $A \in \mathbb{T}^{n \times n}$ uniquely corresponds to a bilinear form. There is therefore a bijective correspondence between the set of bilinear forms and the space of matrices $\mathbb{T}^{n \times n}$. Moreover, this is an isomorphism, since bilinear forms form a vector space with naturally defined addition and scalar multiples (cf. the space $\mathcal{F}$ from p. 79).

In the vector space $\mathbb{T}^n$, bilinear forms have a special form.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 12.9)</span></p>

Suppose the characteristic of the field $\mathbb{T}$ is not 2. Then every bilinear form on $\mathbb{T}^n$ can be expressed in the form

$$b(x, y) = x^\top Ay$$

for a certain matrix $A \in \mathbb{T}^{n \times n}$, and every quadratic form on $\mathbb{T}^n$ can be expressed in the form

$$f(x) = x^\top Ax$$

for a certain symmetric matrix $A \in \mathbb{T}^{n \times n}$.

</div>

*Proof.* It suffices to take $A$ as the matrix of the form with respect to the canonical basis. Then $b(x, y) = [x]_{\text{kan}}^\top A [y]_{\text{kan}} = x^\top Ay$. For the quadratic form we then have $f(x) = b(x, x) = x^\top Ax$. If $A$ is not symmetric, we replace it by the symmetric matrix $\frac{1}{2}(A + A^\top)$ in the sense of Remark 11.2, because $x^\top Ax = x^\top \frac{1}{2}(A + A^\top)x$. We use here the convention $2 \equiv 1 + 1$. Since the characteristic of the field is not 2, we have $1 + 1 \neq 0$ and we can construct the matrix.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 12.10)</span></p>

Consider the bilinear form on $\mathbb{R}^2$

$$b(x, y) = x_1 y_1 + 2x_1 y_2 + 4x_2 y_1 + 10x_2 y_2.$$

The matrix of $b$ with respect to the canonical basis is $A = \begin{pmatrix} 1 & 2 \\ 4 & 10 \end{pmatrix}$, which we can also easily see from the expression

$$b(x, y) = x^\top Ay = \begin{pmatrix} x_1 & x_2 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 4 & 10 \end{pmatrix} \begin{pmatrix} y_1 \\ y_2 \end{pmatrix}.$$

This bilinear form is not symmetric, unlike the bilinear form

$$b'(x, y) = x_1 y_1 + 3x_1 y_2 + 3x_2 y_1 + 10x_2 y_2.$$

The matrix of $b'$ with respect to the canonical basis is $A' = \begin{pmatrix} 1 & 3 \\ 3 & 10 \end{pmatrix}$, so $b'(x, y) = x^\top A' y$. The corresponding quadratic form satisfies

$$f'(x) = b'(x, x) = x^\top A' x = \begin{pmatrix} x_1 & x_2 \end{pmatrix} \begin{pmatrix} 1 & 3 \\ 3 & 10 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 12.11 — Why symmetric bilinear forms)</span></p>

One might ask why we define quadratic forms only using symmetric bilinear forms. After all, nothing prevents us from defining $f(u) = b(u, u)$ also for a non-symmetric bilinear form $b$. The reason is similar to what we stated for positive definite matrices, see Remark 11.2. We can introduce the bilinear form $b_s(u, v) \coloneqq \frac{1}{2}\bigl(b(u, v) + b(v, u)\bigr)$, which will be symmetric. Moreover, as is easy to see, both forms $b$, $b_s$ induce the same quadratic form $f$. Here, however, the field $\mathbb{T}$ must not have characteristic 2, otherwise the fraction would not make sense. By restricting to the symmetric case, we also consequently have uniqueness of the matrix of the quadratic form, again under the assumption that the field $\mathbb{T}$ does not have characteristic 2.

</div>

The matrices of forms depend on the choice of basis. How does the matrix change when we switch to a different basis?

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 12.12 — Matrix of a quadratic form under change of basis)</span></p>

Let $A \in \mathbb{T}^{n \times n}$ be the matrix of a quadratic form $f$ with respect to basis $B$ of a space $V$. Let $B'$ be another basis and $S = {}_B[id]_{B'}$ the change-of-basis matrix from $B'$ to $B$. Then the matrix of the form $f$ with respect to basis $B'$ is $S^\top AS$ and corresponds to the same symmetric bilinear form.

</div>

*Proof.* Let $u, v \in V$ and $b$ be the symmetric bilinear form inducing $f$. Then

$$b(u, v) = [u]_B^\top A [v]_B = ({}_B[id]_{B'} \cdot [u]_{B'})^\top A ({}_B[id]_{B'} \cdot [v]_{B'}) = [u]_{B'}^\top S^\top A S [v]_{B'}.$$

By Theorem 12.7, $S^\top AS$ is the matrix of the form $b$, and hence of $f$, with respect to basis $B'$.

By varying the choice of basis of the space $V$, we obtain different matrix representations. Our goal will be to find a basis with respect to which the matrix is as simple as possible, namely diagonal.

There is a certain parallel with diagonalization for eigenvalues, where we transformed the matrix using similarity. Now we transform the matrix by the operation $S^\top AS$, where $S$ is nonsingular. Instead of similarity, we now have so-called *congruence*. As we shall see, for quadratic forms the situation is simpler -- every matrix can be diagonalized.

### 12.2 Sylvester's law of inertia

In this section we continue to consider the real space $\mathbb{R}^n$ over $\mathbb{R}$. From the definition it is clear that the matrix of a quadratic form is symmetric. We will very much need this property.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 12.13 — Sylvester's law of inertia)</span></p>

Let $f(x) = x^\top Ax$ be a quadratic form on $\mathbb{R}^n$. Then there exists a basis with respect to which $f$ has a diagonal matrix with entries $1, -1, 0$. Moreover, this matrix is unique up to the order of entries.

</div>

*Proof.* "Existence." Since $A$ is symmetric, it has a spectral decomposition $A = Q\Lambda Q^\top$, where $\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$. Thus $\Lambda = Q^\top AQ$ is the diagonalization of the form. To achieve $\pm 1$ on the diagonal, we perform the additional transformation $\Lambda' Q^\top A Q \Lambda'$, where $\Lambda'$ is the diagonal matrix with entries $\Lambda'_{ii} = |\lambda_i|^{-1/2}$ if $\lambda_i \neq 0$ and $\Lambda'_{ii} = 1$ otherwise. We can now regard $Q\Lambda'$ as the matrix ${}\_{\text{kan}}[id]_B$ of change of basis from the desired basis $B$ to the canonical basis. Thus the basis $B$ can be read from the columns of the matrix $Q\Lambda'$.

"Uniqueness." By contradiction, suppose we have two different diagonalizations $D$, $D'$:

$$D = \operatorname{diag}(\underbrace{1, \ldots, 1}_{p}, \underbrace{-1, \ldots, -1}_{q-p}, \underbrace{0, \ldots, 0}_{n-q}), \qquad D' = \operatorname{diag}(\underbrace{1, \ldots, 1}_{s}, \underbrace{-1, \ldots, -1}_{t-s}, \underbrace{0, \ldots, 0}_{n-t}).$$

Let the first correspond to basis $B = \lbrace w_1, \ldots, w_n \rbrace$ and the second to basis $B' = \lbrace w'_1, \ldots, w'_n \rbrace$. Let $u \in \mathbb{R}^n$ be arbitrary with coordinates $y = [u]_B$, $z = [u]_{B'}$. Then by Theorem 12.7

$$f(u) = [u]_B^\top D [u]_B = y^\top D y = y_1^2 + \ldots + y_p^2 - y_{p+1}^2 - \ldots - y_q^2 + 0 y_{q+1}^2 + \ldots + 0 y_n^2,$$

$$f(u) = [u]_{B'}^\top D' [u]_{B'} = z^\top D' z = z_1^2 + \ldots + z_s^2 - z_{s+1}^2 - \ldots - z_t^2 + 0 z_{t+1}^2 + \ldots + 0 z_n^2.$$

First, notice that $q = t$. Since $D = S^\top D' S$ for some nonsingular $S$, specifically for $S = {}_{B'}[id]_B$, the matrices $D, D'$ have the same rank. Hence $q = t$. It remains to show that necessarily $p = s$. Without loss of generality, assume $p > s$. Define the spaces $P = \operatorname{span}\lbrace w_1, \ldots, w_p \rbrace$ and $R = \operatorname{span}\lbrace w'_{s+1}, \ldots, w'_n \rbrace$. Then

$$\dim P \cap R = \dim P + \dim R - \dim(P + R) \ge p + (n - s) - n = p - s \ge 1.$$

Hence there exists a nonzero vector $u \in P \cap R$ and for it we have $u = \sum_{i=1}^p y_i w_i = \sum_{j=s+1}^n z_j w'_j$, from which we get

$$f(u) = \begin{cases} y_1^2 + \ldots + y_p^2 > 0, \\ -z_{s+1}^2 - \ldots - z_t^2 \le 0. \end{cases}$$

This is a contradiction.

A basis with respect to which the matrix of the quadratic form is diagonal is called a *polar basis*. Thus the basis from Theorem 12.13 is an example of a polar basis, but typically there are others. It can also be shown that a polar basis exists not only for real spaces, but also for spaces over any field of characteristic different from 2.

The significance of Sylvester's law of inertia lies not only in the existence of the diagonalization, but especially in its uniqueness (hence the name "inertia"). This uniqueness justifies the introduction of the concept of *signature* as the triple $(p, q, z)$, where $p$ is the number of ones, $q$ the number of minus ones, and $z$ the number of zeros in the resulting diagonal matrix. Moreover, it has a number of consequences concerning, among other things, positive (semi-)definiteness and eigenvalues.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 12.15)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ be symmetric and $S^\top AS$ be the reduction to diagonal form. Then the number of ones, minus ones, and zeros on the diagonal corresponds to the number of positive, negative, and zero eigenvalues of the matrix $A$, respectively.

</div>

*Proof.* It suffices to consider the quadratic form $f(x) = x^\top Ax$ with matrix $A$. From the proof of Theorem 12.13 (the "existence" part), it is clear that one diagonalization is obtained from the spectral decomposition and the claim holds for it. Thanks to the uniqueness in Sylvester's law of inertia, the counts must then agree for any other diagonalization as well.

By diagonalizing the matrix $A$ we thus do not find the eigenvalues, but we determine how many of them are positive and how many are negative.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 12.16)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ be symmetric and $S^\top AS$ be the reduction to diagonal form. Then

1. $A$ is positive definite if and only if $S^\top AS$ has a positive diagonal,
2. $A$ is positive semidefinite if and only if $S^\top AS$ has a nonnegative diagonal.

</div>

*Proof.* From Corollary 12.15 and the relationship between positive (semi-)definiteness and eigenvalues (Theorem 11.7).

Sylvester's law thus provides a recipe for deciding positive definiteness, positive semidefiniteness, and negative (semi-)definiteness all at once with a single method.

#### Diagonalization of a matrix using elementary operations

The remaining question is how to reduce the matrix of a quadratic form to diagonal form. The proof of Sylvester's law of inertia does give a recipe (via the spectral decomposition), but we can simply adapt elementary matrix operations. What happens when we transform a symmetric matrix $A$ to $EAE^\top$, where $E$ is an elementary row operation matrix? The product $EA$ performs the row operation, and multiplication by $E^\top$ on the right also performs the analogous column operation. The basic idea of the diagonalization method is therefore to apply row operations and the corresponding column operations to the matrix. This will zero out elements both below and above the diagonal until the matrix is reduced to diagonal form.

First assume that there is always a nonzero number in the pivot position, so we can restrict ourselves to the second elementary operation only -- adding an $\alpha$-multiple of the $j$-th row to the $i$-th row below it. This operation symmetrically zeros out elements below and to the right of the pivot. Moreover, it does not spoil the part that has already been processed. As a result, we necessarily obtain a diagonal matrix.

A minor difficulty arises if a zero appears in the pivot position during the matrix operations. However, we can add the second row to the first row and analogously for columns, which leads to a matrix with a nonzero pivot. This procedure can be applied in general. If there is a zero in the pivot position, we add a suitable row below it and analogously for columns.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 12.17 — Diagonalization of a quadratic form matrix)</span></p>

We diagonalize the matrix $A$ by alternately applying a row operation and then the corresponding column operation:

$$A = \begin{pmatrix} 1 & 2 & -1 \\ 2 & 5 & -3 \\ -1 & -3 & 2 \end{pmatrix} \sim \begin{pmatrix} 1 & 2 & -1 \\ 0 & 1 & -1 \\ -1 & -3 & 2 \end{pmatrix} \sim \begin{pmatrix} 1 & 0 & -1 \\ 0 & 1 & -1 \\ -1 & -1 & 2 \end{pmatrix} \sim$$

$$\sim \begin{pmatrix} 1 & 0 & -1 \\ 0 & 1 & -1 \\ 0 & -1 & 1 \end{pmatrix} \sim \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & -1 \\ 0 & -1 & 1 \end{pmatrix} \sim \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & -1 \\ 0 & 0 & 0 \end{pmatrix} \sim \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}.$$

We see that the matrix $A$ has two positive eigenvalues and one zero eigenvalue, so it is positive semidefinite.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 12.18 — Finding a polar basis)</span></p>

For simplicity, consider the quadratic form $f(x) = x^\top Ax$, where $A \in \mathbb{R}^{n \times n}$ is symmetric. If we find a matrix $S \in \mathbb{R}^{n \times n}$ such that $S^\top AS$ is diagonal, then the polar basis is contained in the columns of the matrix $S$. But how do we find the matrix $S$? If we diagonalize the matrix $A$ using elementary operations, then the matrix $S$ represents the accumulated column operations. The method is now straightforward: we transform the augmented matrix $(A \mid I_n)$ by applying row and column operations to the matrix $A$ to diagonalize it, and applying only column operations to the identity matrix. The polar basis is then read from the columns of the matrix on the right.

We apply the procedure to the matrix from Example 12.17, and the individual steps are

$$(A \mid I_3) = \begin{pmatrix} 1 & 2 & -1 \mid 1 & 0 & 0 \\ 2 & 5 & -3 \mid 0 & 1 & 0 \\ -1 & -3 & 2 \mid 0 & 0 & 1 \end{pmatrix} \sim \begin{pmatrix} 1 & 0 & -1 \mid 1 & -2 & 0 \\ 0 & 1 & -1 \mid 0 & 1 & 0 \\ -1 & -1 & 2 \mid 0 & 0 & 1 \end{pmatrix} \sim$$

$$\sim \begin{pmatrix} 1 & 0 & 0 \mid 1 & -2 & 1 \\ 0 & 1 & -1 \mid 0 & 1 & 0 \\ 0 & -1 & 1 \mid 0 & 0 & 1 \end{pmatrix} \sim \begin{pmatrix} 1 & 0 & 0 \mid 1 & -2 & -1 \\ 0 & 1 & 0 \mid 0 & 1 & 1 \\ 0 & 0 & 0 \mid 0 & 0 & 1 \end{pmatrix}.$$

The corresponding polar basis thus consists of the vectors $(1, 0, 0)^\top$, $(-2, 1, 0)^\top$, $(-1, 1, 1)^\top$. If we arrange these vectors as columns of a matrix $S$, then $S^\top AS$ is the diagonal matrix with entries $1, 1, 0$ on the diagonal.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 12.19 — Sum of squares of linear forms)</span></p>

Consider the quadratic form $f(x) = x^\top Ax$ with a symmetric matrix $A \in \mathbb{R}^{n \times n}$. If we can express $x^\top Ax$ with variables $x_1, \ldots, x_n$ as a sum of squares of linear forms, then obviously $f(x) \ge 0$ for all $x \in \mathbb{R}^n$ and the matrix $A$ is positive semidefinite. Interestingly, the converse also holds: Every quadratic form with a positive semidefinite matrix can be expressed as a sum of squares of linear forms.

We find a matrix $S$ for which $S^\top AS = D$ is diagonal. Then $A = S^{-T} D S^{-1}$ and with the substitution $y \coloneqq S^{-1}x$ we obtain the desired form

$$x^\top Ax = x^\top S^{-T} D S^{-1} x = y^\top Dy = \sum_{i=1}^n d_{ii} y_i^2 = \sum_{i=1}^n d_{ii} (S_{i*}^{-1} x)^2.$$

Specifically, consider the matrix from Example 12.18, where we already observed that $S^\top AS = D$ for

$$S = \begin{pmatrix} 1 & -2 & -1 \\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{pmatrix}, \quad D = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}.$$

We compute

$$S^{-1} = \begin{pmatrix} 1 & 2 & -1 \\ 0 & 1 & -1 \\ 0 & 0 & 1 \end{pmatrix}.$$

Now $\sum_{i=1}^n d_{ii}(S_{i*}^{-1}x)^2 = (x_1 + 2x_2 - x_3)^2 + (x_2 - x_3)^2$. We have thus shown that

$$x^\top Ax = x_1^2 + 4x_1 x_2 - 2x_1 x_3 + 5x_2^2 - 6x_2 x_3 + 2x_3^2 = (x_1 + 2x_2 - x_3)^2 + (x_2 - x_3)^2.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 12.20 — Product of positive definite matrices)</span></p>

The product of positive definite matrices $A, B \in \mathbb{R}^{n \times n}$ need not be a positive definite matrix. The product $AB$ is generally not a symmetric matrix, and moreover even symmetrization in the sense of Remark 11.2 need not yield a positive definite matrix.

Interestingly, however, the matrix $AB$, although not necessarily symmetric, still has real positive eigenvalues. This is easily seen from the expression $AB = \sqrt{A} \sqrt{A} B$. Multiplying by $\sqrt{A}^{-1}$ on the left and $\sqrt{A}$ on the right, we obtain the similar matrix $\sqrt{A} B \sqrt{A}$. This matrix is symmetric, hence has real eigenvalues, and by inertia has the same signature as $B$, which means positive eigenvalues.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 12.21 — Block test for positive definiteness)</span></p>

There is a close relationship between diagonalization of a matrix using elementary operations and the recurrence formula for testing positive definiteness (Theorem 11.9). If we consider the matrix $A = \begin{pmatrix} \alpha & a^\top \\ a & \tilde{A} \end{pmatrix}$ as the matrix of a quadratic form and use elementary operations to zero out elements below and to the right of the pivot, the resulting block diagonal matrix can be expressed in matrix form as

$$\begin{pmatrix} \alpha & o^\top \\ o & \tilde{A} - \frac{1}{\alpha}aa^\top \end{pmatrix}.$$

This matrix is positive definite if and only if all blocks are positive definite matrices, i.e., $\alpha > 0$ and $\tilde{A} - \frac{1}{\alpha}aa^\top$ is positive definite. This gives us another derivation of the recurrence formula.

</div>

### Conics and quadrics

Quadratic forms can be used to describe geometric objects called *quadrics*. These are (briefly speaking) sets described by the equation $x^\top Ax + b^\top x + c = 0$, where $A \in \mathbb{R}^{n \times n}$ is symmetric, $b \in \mathbb{R}^n$, $c \in \mathbb{R}$. As we can see, this equation is no longer linear precisely because of the quadratic term $x^\top Ax$. Using various characteristics, such as the eigenvalues or the signature of the matrix $A$, we can then easily classify the individual geometric shapes of quadrics. These shapes include ellipsoids, paraboloids, hyperboloids, etc., see the examples below.

A special case of quadrics in the space $\mathbb{R}^2$ are the *conics*. These include ellipses, parabolas, and hyperbolas.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 12.22 — Ellipsoids)</span></p>

The equation $\frac{1}{a^2}x_1^2 + \frac{1}{b^2}x_2^2 = 1$ describes in the plane $\mathbb{R}^2$ an ellipse centered at the origin, with semi-axes in the directions of the coordinate axes $x_1, x_2$ of lengths $a$ and $b$, respectively.

Now consider the equation $x^\top Ax = 1$, where $A \in \mathbb{R}^{n \times n}$ is positive definite and $x = (x_1, \ldots, x_n)^\top$ is the vector of variables. We show that this equation describes an ellipsoid in the space $\mathbb{R}^n$. Let $A = Q\Lambda Q^\top$ be the spectral decomposition. With the substitution $y \coloneqq Q^\top x$ we get

$$1 = x^\top Ax = x^\top Q \Lambda Q^\top x = y^\top \Lambda y = \sum_{i=1}^n \lambda_i y_i^2 = \sum_{i=1}^n \frac{1}{(\lambda_i^{-1/2})^2} y_i^2.$$

We thus obtain a description of an ellipsoid centered at the origin, with semi-axes in the directions of the coordinate axes of lengths $\frac{1}{\sqrt{\lambda_1}}, \ldots, \frac{1}{\sqrt{\lambda_n}}$. However, this description is in the space after the transformation $y = Q^\top x$. We transform back via $x = Qy$. Since $Q$ is an orthogonal matrix, we get the same ellipsoid centered at the origin, only rotated or reflected in some way. Since the canonical basis $e_1, \ldots, e_n$ maps to the columns of the matrix $Q$ (which are the eigenvectors of $A$), the semi-axes of the original ellipsoid will be in the directions of the eigenvectors of the matrix $A$.

If $A$ is symmetric but not positive definite, the analysis will be the same. We just do not get an ellipsoid, but a different geometric object (hyperboloid, etc.)

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 12.23 — Some quadrics in $\mathbb{R}^3$)</span></p>

The following figures illustrate some three-dimensional quadrics:

- **Ellipsoid:** $\frac{x_1^2}{a^2} + \frac{x_2^2}{b^2} + \frac{x_3^2}{c^2} = 1$
- **Cone:** $\frac{x_1^2}{a^2} + \frac{x_2^2}{b^2} - \frac{x_3^2}{c^2} = 0$
- **Hyperbolic paraboloid:** $\frac{x_1^2}{a^2} - \frac{x_2^2}{b^2} - x_3 = 0$
- **Hyperbolic cylinder:** $\frac{x_1^2}{a^2} - \frac{x_2^2}{b^2} = 1$
- **Hyperboloid of one sheet:** $\frac{x_1^2}{a^2} + \frac{x_2^2}{b^2} - \frac{x_3^2}{c^2} = 1$
- **Hyperboloid of two sheets:** $-\frac{x_1^2}{a^2} - \frac{x_2^2}{b^2} + \frac{x_3^2}{c^2} = 1$

</div>

### Summary of Chapter 12

A quadratic form on the space $\mathbb{R}^n$ is a polynomial $f(x) = x^\top Ax$, where $A \in \mathbb{R}^{n \times n}$ is symmetric. For general spaces we work in coordinates similarly to linear mappings. If we change the coordinate system, the matrix changes to $S^\top AS$, where $S$ is the change-of-basis matrix between coordinate systems. The central theorem of this chapter, Sylvester's law of inertia, asserts that we can always find a coordinate system in which the matrix is diagonal. Moreover, this diagonal matrix always has the same signature for a given quadratic form -- the same number of positive and negative entries (this is the main message of the theorem!). This allows us to simply classify and describe quadratic forms. The signature for a given matrix is determined simply using elementary operations applied both to rows and symmetrically to columns. This also gives us, incidentally, an efficient method for testing positive (semi-)definiteness, among other things. The theory of quadratic forms also provides us with an effective tool for analyzing polynomial equations of the type $x^\top Ax + b^\top x + c = 0$ and for characterizing various conics and quadrics, such as ellipsoids.

Quadratic forms are closely related to bilinear forms. A real inner product is always a bilinear form. In particular, in the space $\mathbb{R}^n$, bilinear forms have the form $b(x, y) = x^\top Ay$. In general spaces they have a similar expression, but in the language of coordinates. Specifically, $b(x, y) = [x]_B^\top A [y]_B$ for a given basis $B$.

## Chapter 13 — Matrix Decompositions

Top 10 algorithms of the 20th century according to [Dongarra and Sullivan, 2000; Cipra, 2000]:

1. *Monte Carlo method* (1946, J. von Neumann, S. Ulam, and N. Metropolis) — using simulations with random numbers we compute approximate solutions to problems that are very hard to solve exactly.
2. *Simplex method for linear programming* (1946, G. Dantzig) — a method for solving optimization problems with linear objective and constraints.
3. *Krylov subspace iteration methods* (1950, M. Hestenes, E. Stiefel, C. Lanczos) — methods for solving large and sparse systems of linear equations.
4. *Matrix decompositions* (1951, A. Householder) — matrix factorizations such as the Cholesky decomposition, LU decomposition, QR decomposition, spectral decomposition, Schur triangularization, or SVD decomposition.
5. *Fortran compiler* (1957, J. Backus) — the programming language Fortran, which was one of the first to enable demanding numerical computations.
6. *QR algorithm* (1961, J. Francis) — an algorithm for computing eigenvalues.
7. *Quicksort* (1962, A. Hoare) — a practically fast algorithm for sorting elements.
8. *Fast Fourier transform* (1965, J. Cooley, J. Tukey) — for fast multiplication of polynomials and numbers, signal processing, and many other things.
9. *Integer relation detection algorithm* (1977, H. Ferguson, R. Forcade) — a generalization of the Euclidean algorithm of successive division to the problem: Given $n$ real numbers $x_1, \ldots, x_n$, does a nontrivial integer combination $a_1 x_1 + \ldots + a_n x_n = 0$ exist?
10. *Fast multipole algorithm* (1987, L. Greengard, V. Rokhlin) — simulation in the problem of computing long-range forces in the $n$-body problem. It groups nearby sources together and treats them as a single one.

We see that matrix decompositions (factorizations) were included in the list. We have already encountered several decompositions (LU decomposition, spectral decomposition, Cholesky decomposition, ...). The QR decomposition, which we will discuss in Section 13.2, appears in the list implicitly one more time, since it is the basis of the QR algorithm. Its importance is thus evident.

In this chapter, we will consider the standard inner product in $\mathbb{R}^n$ and the Euclidean norm (unless explicitly stated otherwise).

### 13.1 Householder Transformation

The motivation for this section is the following. Let two vectors $x, y \in \mathbb{R}^n$ be given and we want to find a linear mapping $x \mapsto Hx$ that maps the vector $x$ to the vector $y$. In order for the linear mapping to have nice properties, we additionally require the matrix $H$ to be orthogonal. Since such a linear mapping preserves lengths (Theorem 8.66), the two vectors $x, y$ must necessarily have the same norm.

Such a mapping always exists, and one can choose a suitable Householder matrix for $H$. Recall (Example 8.65) that the Householder matrix is defined as $H(x) \coloneqq I_n - \frac{2}{x^\top x} x x^\top$, where $o \neq x \in \mathbb{R}^n$. This matrix is orthogonal and symmetric. By a suitable choice of vectors $x, y$, it can replace elementary matrices in the computation of the echelon form of a matrix. This procedure is called the Householder transformation.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 13.1 — Householder Transformation)</span></p>

For every $x, y \in \mathbb{R}^n$, $x \neq y$, $\|x\|_2 = \|y\|_2$, we have $y = H(x - y)x$.

</div>

*Proof.* Let us compute

$$H(x - y)x = \left(I_n - \frac{2}{(x - y)^\top(x - y)}(x - y)(x - y)^\top\right) x =$$

$$= x - \frac{2(x - y)^\top x}{(x - y)^\top(x - y)}(x - y) = x - \frac{2\|x\|_2^2 - 2y^\top x}{(x - y)^\top(x - y)}(x - y) =$$

$$= x - \frac{\|x\|_2^2 + \|y\|_2^2 - 2y^\top x}{\|x - y\|_2^2}(x - y) = x - \frac{\|x - y\|_2^2}{\|x - y\|_2^2}(x - y) = x - (x - y) = y.$$

The Householder matrix thus transforms a chosen vector $x$ into another vector $y$ with the same norm by multiplying the vector $x$ from the left by a suitable Householder matrix. In particular, every vector can be transformed into a suitable multiple of a unit vector:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Corollary 13.2)</span></p>

Let $x \in \mathbb{R}^n$ and define

$$H \coloneqq \begin{cases} H(x - \|x\|_2 e_1), & \text{if } x \neq \|x\|_2 e_1, \\ I_n, & \text{otherwise.} \end{cases}$$

Then $Hx = \|x\|_2 e_1$.

</div>

*Proof.* The case $x = \|x\|_2 e_1$ is obvious. Otherwise we apply Theorem 13.1; the vectors $x$, $\|x\|_2 e_1$ have the same norm.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 13.3)</span></p>

Let $x = (2, 2, 1)^\top$. Then $\|x\|_2 = 3$ and thus

$$H = H(x - 3e_1) = \frac{1}{3}\begin{pmatrix} 2 & 2 & 1 \\ 2 & -1 & -2 \\ 1 & -2 & 2 \end{pmatrix}.$$

Now $Hx = (3, 0, 0)^\top$, so the linear mapping $x \mapsto Hx$ maps the vector $x = (2, 2, 1)^\top$ to a multiple of the unit vector.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 13.4 — Extension to an Orthonormal Basis)</span></p>

Let $u \in \mathbb{R}^n$ be a vector of unit length, i.e., $\|u\|_2 = 1$. Construct an orthogonal matrix $Q$ with its first column equal to $u$. In other words, extend $u$ to an orthonormal basis of $\mathbb{R}^n$.

*Solution:* If $u = e_1$, then it clearly suffices to choose $Q = I_n$. Otherwise, we can choose $Q \coloneqq H(e_1 - u)$, i.e., the Householder matrix for the vector $e_1 - u$. The justification is simple: the first column of this matrix equals, by the Householder transformation, $Qe_1 = H(e_1 - u)e_1 = u$.

</div>

Let $A \in \mathbb{R}^{m \times n}$. We construct the Householder matrix $H$ so that the Householder transformation according to Corollary 13.2 maps the first column of $A$ to a multiple of $e_1$. By multiplying $HA$, we thus zero out the entries in the first column of $A$ except for the first one. By recursively applying the transformation, we then reduce the matrix to echelon form. This procedure is therefore an alternative to elementary row operations. However, we gain something extra here, namely the so-called QR decomposition, which we discuss in more detail in the following section.

Similarly, one can also use Givens matrices (Problem 13.2). By multiplying the matrix $A$ from the left by a suitable Givens matrix, we can zero out any (but only one) entry below the pivot; this is a property shared with elementary operation matrices. To zero out all entries below the pivot, we therefore need to apply the corresponding Givens matrices multiple times. However, we will not discuss Givens matrices in detail.

### 13.2 QR Decomposition

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 13.5 — QR Decomposition)</span></p>

For every matrix $A \in \mathbb{R}^{m \times n}$, there exists an orthogonal $Q \in \mathbb{R}^{m \times m}$ and an upper triangular matrix $R \in \mathbb{R}^{m \times n}$ with nonnegative diagonal such that $A = QR$.

</div>

*Proof.* By mathematical induction on $n$, i.e., the number of columns. If $n = 1$, then $A = a \in \mathbb{R}^m$ and for the matrix $H$ constructed according to Corollary 13.2 we have $Ha = \|a\|_2 e_1$. It suffices to set $Q \coloneqq H^\top$ and $R \coloneqq \|a\|_2 e_1$.

Inductive step $n \leftarrow n - 1$. Applying Corollary 13.2 to the first column of $A$, we get $HA_{*1} = \|A_{*1}\|_2 e_1$. Thus $HA$ has the form

$$HA = \begin{pmatrix} \alpha & b^\top \\ o & B \end{pmatrix},$$

where $B \in \mathbb{R}^{(m-1) \times (n-1)}$ and $\alpha = \|A_{*1}\|_2 \ge 0$. By the inductive hypothesis, there exists a decomposition $B = Q'R'$, where $Q' \in \mathbb{R}^{(m-1) \times (m-1)}$ is orthogonal and $R' \in \mathbb{R}^{(m-1) \times (n-1)}$ is upper triangular with nonnegative diagonal. We compute

$$\begin{pmatrix} 1 & o^\top \\ o & Q'^\top \end{pmatrix} HA = \begin{pmatrix} 1 & o^\top \\ o & Q'^\top \end{pmatrix} \begin{pmatrix} \alpha & b^\top \\ o & B \end{pmatrix} = \begin{pmatrix} \alpha & b^\top \\ o & R' \end{pmatrix}.$$

Denote

$$Q \coloneqq H^\top \begin{pmatrix} 1 & o^\top \\ o & Q' \end{pmatrix}, \quad R \coloneqq \begin{pmatrix} \alpha & b^\top \\ o & R' \end{pmatrix}.$$

The matrix $Q$ is orthogonal and $R$ is upper triangular with nonnegative diagonal. Now equation (13.1) has the form $Q^\top A = R$, that is, $A = QR$ is the desired decomposition.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Algorithm 13.6 — QR Decomposition)</span></p>

**Input:** matrix $A \in \mathbb{R}^{m \times n}$.

1. $Q \coloneqq I_m$, $R \coloneqq A$,
2. **for** $j \coloneqq 1$ **to** $\min(m, n)$ **do**
3. &emsp; $x \coloneqq R(j : m, j)$,
4. &emsp; **if** $x \neq \|x\|_2 e_1$ **then**
5. &emsp;&emsp; $x \coloneqq x - \|x\|_2 e_1$,
6. &emsp;&emsp; $H(x) \coloneqq I_{m-j+1} - \frac{2}{x^\top x} x x^\top$,
7. &emsp;&emsp; $H \coloneqq \begin{pmatrix} I_{j-1} & 0 \\ 0 & H(x) \end{pmatrix}$,
8. &emsp;&emsp; $R \coloneqq HR$, $Q \coloneqq QH$,
9. &emsp; **end if**
10. **end for**

**Output:** matrices $Q, R$ from the QR decomposition of $A$ (we have $A = QR$).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 13.7 — QR Decomposition)</span></p>

Let

$$A = \begin{pmatrix} 0 & -20 & -14 \\ 3 & 27 & -4 \\ 4 & 11 & -2 \end{pmatrix}.$$

**First iteration:**

$x = A_{*1} - \|A_{*1}\|e_1 = (-5, 3, 4)^\top$,

$$Q_1 = I_3 - 2\frac{xx^\top}{x^\top x} = \frac{1}{25}\begin{pmatrix} 0 & 15 & 20 \\ 15 & 16 & -12 \\ 20 & -12 & 9 \end{pmatrix}, \quad Q_1 A = \begin{pmatrix} 5 & 25 & -4 \\ 0 & 0 & -10 \\ 0 & -25 & -10 \end{pmatrix}.$$

**Second iteration:**

$x = (0, -25)^\top - 25e_1 = (-25, -25)^\top$,

$$Q_2 = I_2 - 2\frac{xx^\top}{x^\top x} = \begin{pmatrix} 0 & -1 \\ -1 & 0 \end{pmatrix}, \quad \tilde{Q}_2 = \begin{pmatrix} 25 & 10 \\ 0 & 10 \end{pmatrix}.$$

**Result:**

$$Q = Q_1 \begin{pmatrix} 1 & 0 \\ 0 & Q_2 \end{pmatrix} = \frac{1}{25}\begin{pmatrix} 0 & -20 & -15 \\ 15 & 12 & -16 \\ 20 & -9 & 12 \end{pmatrix}, \quad R = \begin{pmatrix} 5 & 25 & -4 \\ 0 & 25 & 10 \\ 0 & 0 & 10 \end{pmatrix}.$$

</div>

The QR decomposition is unique only under certain assumptions. For example, for the zero matrix $A = 0$ we have $R = 0$ and $Q$ is an arbitrary orthogonal matrix, so uniqueness does not hold here.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 13.8 — Uniqueness of the QR Decomposition)</span></p>

For a nonsingular matrix $A \in \mathbb{R}^{n \times n}$, the QR decomposition is unique and the matrix $R$ has positive diagonal entries.

</div>

*Proof.* From the relation $A = QR$ it follows that $R$ is nonsingular, and therefore must have a nonzero, hence positive, diagonal. We prove uniqueness by contradiction. Suppose $A$ has two different decompositions $A = Q_1 R_1 = Q_2 R_2$. Then $Q_2^\top Q_1 = R_2 R_1^{-1}$, and we denote this matrix by $U$. Clearly $U$ is orthogonal (it is a product of orthogonal matrices $Q_2^\top$ and $Q_1$) and upper triangular (it is a product of upper triangular matrices $R_2$ and $R_1^{-1}$). In particular, $U$ has the form $U_{*1} = (u_{11}, 0, \ldots, 0)^\top$, where $u_{11} > 0$. For it to have unit norm, we must have $u_{11} = 1$ and therefore $U_{*1} = e_1$. The second column is orthogonal to the first, so $u_{21} = 0$, and for it to have unit norm, we must have $u_{22} = 1$. Thus $U_{*2} = e_2$. Etc., we continue until we obtain $U = I_n$, from which $Q_1 = Q_2$ and $R_1 = R_2$. This is a contradiction.

The theorem can be generalized to the case when $A \in \mathbb{R}^{m \times n}$ has linearly independent columns. Then the matrix $R$ and the first $n$ columns of $Q$ are uniquely determined, and the diagonal of $R$ is positive.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 13.9 — QR Decomposition via Gram–Schmidt Orthogonalization)</span></p>

The QR decomposition of a matrix $A \in \mathbb{R}^{m \times n}$ can also be constructed using Gram–Schmidt orthogonalization. In practice this is not done, because the use of orthogonal matrices makes the Householder transformation numerically superior, but it allows us to better understand the relationship between the two methods.

The basic idea is the following: While Householder transformations (i.e., a sequence of orthogonal matrices) reduce the matrix $A$ to upper triangular form, Gram–Schmidt orthogonalization works in exactly the opposite way — using a sequence of suitable upper triangular matrices we transform $A$ into a matrix with orthonormal columns.

Specifically, we describe Gram–Schmidt orthogonalization as follows. Let $A \in \mathbb{R}^{m \times n}$ be a matrix whose columns we want to orthonormalize. We proceed according to Algorithm 8.23, where the vectors $x_1, \ldots, x_n$ are the columns of $A$ and during the computation we replace the columns of $A$ by the vectors $y_k$ and $z_k$. Step 2 of the algorithm, which has the form

$$y_k \coloneqq x_k - \sum_{j=1}^{k-1} \langle x_k, z_j \rangle z_j,$$

we express in matrix form by multiplying $A$ from the right by the matrix

$$\begin{pmatrix} 1 & & \alpha_1 \\ & \ddots & \vdots \\ & & 1 & \alpha_{k-1} \\ & & & 1 \\ & & & & 1 \end{pmatrix},$$

where $\alpha_j = -\langle x_k, z_j \rangle$, $j = 1, \ldots, k - 1$. Similarly, Step 3, which has the form $z_k \coloneqq \frac{1}{\|y_k\|} y_k$, we express by multiplying $A$ from the right by a diagonal matrix with entries $1, \ldots, 1, \frac{1}{\|y_k\|}, 1, \ldots, 1$ on the diagonal. Since both matrices by which we multiplied $A$ from the right are upper triangular, we can express the entire orthogonalization as

$$AR_1 \ldots R_\ell = Q,$$

where $R_1, \ldots, R_\ell$ are upper triangular matrices and $Q$ has orthonormal columns. The desired QR decomposition is now obtained by setting $R \coloneqq (R_1 \ldots R_\ell)^{-1}$, which is also an upper triangular matrix.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 13.10 — QR Decomposition via Cholesky Decomposition)</span></p>

Let $A \in \mathbb{R}^{m \times n}$ have rank $n$ and let $A = QR$ be its desired QR decomposition. We express the matrices $Q, R$ in block form as

$$A = QR = (\tilde{Q} \quad \tilde{Q}') \begin{pmatrix} \tilde{R} \\ 0 \end{pmatrix} = \tilde{Q}\tilde{R},$$

where $\tilde{Q} \in \mathbb{R}^{m \times n}$ consists of the first $n$ columns of $Q$ and $\tilde{R}$ consists of the first $n$ rows of $R$. The matrix $\tilde{R}$ is a nonsingular upper triangular matrix with positive diagonal. Then $A^\top A = R^\top Q^\top QR = R^\top R = \tilde{R}^\top \tilde{R}$. The matrix $\tilde{R}$ can therefore be constructed by the Cholesky decomposition of the matrix $A^\top A$. From the equation $A = \tilde{Q}\tilde{R}$ we then simply express $\tilde{Q} = A\tilde{R}^{-1}$. The remaining columns of $Q$ are computed arbitrarily so that $Q$ is orthogonal (that this is possible is guaranteed by Corollary 8.27).

</div>

### 13.3 Applications of the QR Decomposition

The QR decomposition can be used to solve many problems that we have encountered so far. Its main advantage is that it works with the orthogonal matrix $Q$. Since orthogonal matrices preserve the norm (Theorem 8.66(2)), rounding errors do not grow excessively. This is the reason why orthogonal matrices are widely used in numerical methods.

#### QR Decomposition and Systems of Equations

Consider a system of linear equations $Ax = b$, where $A \in \mathbb{R}^{n \times n}$ is nonsingular. We compute the solution as follows: Compute the QR decomposition $A = QR$. Then the system has the form $QRx = b$, i.e., $Rx = Q^\top b$. Since $R$ is an upper triangular matrix, the solution is easily obtained by back substitution.

The asymptotic complexity of the QR decomposition, and hence of solving a system of equations, is $\frac{4}{3}n^3$. Compared to Gaussian elimination, this approach is therefore approximately twice as slow (see Remark 2.19), but it is numerically more stable and more accurate.

#### QR Decomposition and Orthogonalization

For what follows, we first introduce the so-called *reduced QR decomposition*. Let $A \in \mathbb{R}^{m \times n}$ have linearly independent columns. Then we write the QR decomposition in block form

$$A = QR = (\tilde{Q} \quad \tilde{Q}') \begin{pmatrix} \tilde{R} \\ 0 \end{pmatrix} = \tilde{Q}\tilde{R},$$

where $\tilde{Q} \in \mathbb{R}^{m \times n}$ consists of the first $n$ columns of $Q$ and $\tilde{R}$ consists of the first $n$ rows of $R$. The matrix $\tilde{R}$ is nonsingular.

Now let us see how to apply the QR decomposition to find an orthonormal basis of a given space; it is therefore an alternative to Gram–Schmidt orthogonalization in $\mathbb{R}^m$. Let $A \in \mathbb{R}^{m \times n}$ have linearly independent columns and suppose we want to construct an orthonormal basis of the column space $\mathcal{S}(A)$. From the equality $A = \tilde{Q}\tilde{R}$ and the nonsingularity of $\tilde{R}$ it follows (Proposition 5.66) that $\mathcal{S}(A) = \mathcal{S}(\tilde{Q})$. Thus an orthonormal basis of $\mathcal{S}(A)$ is formed by the columns of $\tilde{Q}$.

Due to the properties of orthogonal matrices, $\tilde{Q}'$ then forms an orthonormal basis of $\operatorname{Ker}(A^\top)$, since $\operatorname{Ker}(A^\top)$ is the orthogonal complement of $\mathcal{S}(A)$. From the QR decomposition of $A$ or $A^\top$, we can thus read off an orthonormal basis of all fundamental matrix spaces — the row space, column space, and kernel.

#### QR Decomposition and Extension to an Orthonormal Basis

Let $a \in \mathbb{R}^n$, $\|a\|_2 = 1$, and let $a = Qr$ be its QR decomposition. For $r$ to be a vector $r$, viewed as a matrix with one column, in upper triangular form with nonnegative diagonal, it must have the form $r = (\alpha, 0, \ldots, 0)^\top$ for some $\alpha \ge 0$. Since $\|a\|_2 = 1$ and $Q$ is orthogonal, we must have $\|r\|_2 = 1$. Therefore $r = e_1$, from which $a = Q_{*1}$. The vector $a$ thus lies in the first column of $Q$ and the remaining columns represent its extension to an orthonormal basis.

#### QR Decomposition and Projection onto a Subspace

Let $A \in \mathbb{R}^{m \times n}$ have linearly independent columns. We know (Theorem 8.49) that the projection of a vector $x \in \mathbb{R}^m$ onto the column space $\mathcal{S}(A)$ is of the form $x' = A(A^\top A)^{-1}A^\top x$. The expression can be simplified using the reduced QR decomposition $A = \tilde{Q}\tilde{R}$. Since $\mathcal{S}(A) = \mathcal{S}(\tilde{Q})$, we seek the projection onto a space with an orthonormal basis given by the columns of $\tilde{Q}$. By Remark 8.52, the projection matrix is $\tilde{Q}\tilde{Q}^\top$ and the vector $x$ is projected onto the vector $x' = \tilde{Q}\tilde{Q}^\top x$.

#### QR Decomposition and the Least Squares Method

The least squares method (Section 8.5) consists of approximately solving an overdetermined system of equations $Ax = b$, where $A \in \mathbb{R}^{m \times n}$, $m > n$. Let $A$ have rank $n$; then the approximate solution by the least squares method is

$$x^* = (A^\top A)^{-1}A^\top b = \tilde{R}^{-1}(\tilde{R}^\top)^{-1}\tilde{R}^\top \tilde{Q}^\top b = \tilde{R}^{-1}\tilde{Q}^\top b.$$

In other words, $x^*$ is obtained as the solution of the nonsingular system $\tilde{R}x = \tilde{Q}^\top b$, by back substitution since the matrix $\tilde{R}$ is upper triangular. Note the analogy with solving the nonsingular system $Ax = b$, which led to $Rx = Q^\top b$; now we have the truncated system $\tilde{R}x = \tilde{Q}^\top b$.

#### QR Algorithm

The *QR algorithm* is a method for computing eigenvalues of a matrix $A \in \mathbb{R}^{n \times n}$, which has become the foundation of modern efficient methods.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Algorithm 13.11 — QR Algorithm)</span></p>

**Input:** matrix $A \in \mathbb{R}^{n \times n}$.

1. $A_0 \coloneqq A$, $i \coloneqq 0$,
2. **while not** termination condition met **do**
3. &emsp; construct the QR decomposition of $A_i$, i.e., $A_i = QR$,
4. &emsp; $A_{i+1} \coloneqq RQ$,
5. &emsp; $i \coloneqq i + 1$,
6. **end while**

**Output:** matrix $A_i$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 13.12)</span></p>

The matrices $A_0, A_1, \ldots$ are mutually similar.

</div>

*Proof.* $A_{i+1} = RQ = I_n RQ = Q^\top QRQ = Q^\top A_i Q$.

The output matrix $A_i$ is similar to $A$, and therefore has the same eigenvalues. How do we find them? The algorithm generally converges (cases where it does not converge are rare, almost artificial; for a long time no case of non-convergence was known) to a block upper triangular matrix with blocks of size 1 and 2. Blocks of size 1 are eigenvalues, and from blocks of size 2 we easily compute pairs of complex conjugate eigenvalues.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 13.13 — Iterations of the QR Algorithm)</span></p>

Iterations of the QR algorithm for the given matrix $A$:

$$A = \begin{pmatrix} 2 & 4 & 2 \\ 4 & 2 & 2 \\ 2 & 2 & -1 \end{pmatrix} \to \begin{pmatrix} 6.1667 & -2.4623 & 0.8616 \\ -2.4623 & -1.2576 & -0.2598 \\ 0.8616 & -0.2598 & -1.9091 \end{pmatrix} \to$$

$$\to \begin{pmatrix} 6.9257 & 0.7725 & 0.2586 \\ 0.7725 & -1.9331 & 0.0224 \\ 0.2586 & 0.0224 & -1.9925 \end{pmatrix} \to \begin{pmatrix} 6.9939 & -0.2225 & 0.0742 \\ -0.2225 & -1.9945 & -0.0018 \\ 0.0742 & -0.0018 & -1.9994 \end{pmatrix} \to$$

$$\to \begin{pmatrix} 6.9995 & 0.0636 & 0.0212 \\ 0.0636 & -1.9996 & 0.0001 \\ 0.0212 & 0.0001 & -1.9999 \end{pmatrix} \to \begin{pmatrix} 7.0000 & -0.0182 & 0.0061 \\ -0.0182 & -2.0000 & -10^{-5} \\ 0.0061 & -10^{-5} & -2.0000 \end{pmatrix}.$$

A symmetric matrix converges to a diagonal one. The accuracy of the computed eigenvalues is determined by Theorem 10.58 on Gershgorin discs.

</div>

### 13.4 SVD Decomposition

Just like the QR decomposition, the SVD decomposition is one of the fundamental techniques in numerical computations.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 13.14 — SVD Decomposition)</span></p>

Let $A \in \mathbb{R}^{m \times n}$, $q \coloneqq \min\lbrace m, n \rbrace$. Then there exists a diagonal matrix $\Sigma \in \mathbb{R}^{m \times n}$ with entries $\sigma_{11} \ge \ldots \ge \sigma_{qq} \ge 0$ and orthogonal matrices $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$ such that $A = U\Sigma V^\top$.

</div>

The idea of the proof is given after Algorithm 13.17, which constructs the SVD decomposition. The positive numbers on the diagonal $\sigma_{11}, \ldots, \sigma_{rr}$ are called the *singular values* of the matrix $A$ and are usually denoted $\sigma_1, \ldots, \sigma_r$. Clearly $r = \operatorname{rank}(A)$. The singular values are unique, but the matrices $U, V$, and hence the SVD decomposition, need not be.

The transpose of an orthogonal matrix is again an orthogonal matrix, so at first glance it seems pointless to transpose the matrix $V$ in the decomposition $A = U\Sigma V^\top$. The reason for this is rather convention. This way we can find bases of certain spaces in the columns of matrices $U, V$ (see Theorem 13.20); otherwise, without the transpose, these would be the columns of $U$ and the rows of $V$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 13.15 — Relationship Between Singular and Eigenvalues)</span></p>

Let $A \in \mathbb{R}^{m \times n}$, $r = \operatorname{rank}(A)$, and let $A^\top A$ have eigenvalues $\lambda_1 \ge \ldots \ge \lambda_n$. Then the singular values of $A$ are $\sigma_i = \sqrt{\lambda_i}$, $i = 1, \ldots, r$.

</div>

*Proof.* Let $A = U\Sigma V^\top$ be the SVD decomposition of $A$. Then

$$A^\top A = V\Sigma^\top U^\top U \Sigma V^\top = V\Sigma^\top \Sigma V^\top = V \operatorname{diag}(\sigma_1^2, \ldots, \sigma_q^2, 0, \ldots, 0)V^\top,$$

which is the spectral decomposition of the positive semidefinite matrix $A^\top A$. Therefore the determining elements of the diagonal matrix are its eigenvalues, i.e., $\lambda_i = \sigma_i^2$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 13.16)</span></p>

Let $Q \in \mathbb{R}^{n \times n}$ be orthogonal. Then $Q^\top Q = I_n$ has all eigenvalues equal to one. Therefore the orthogonal matrix $Q$ has all singular values equal to one as well.

</div>

The proof of Proposition 13.15 additionally revealed that $V$ is the orthogonal matrix from the spectral decomposition of $A^\top A$. Similarly, $U$ is the orthogonal matrix from the spectral decomposition of $AA^\top$:

$$AA^\top = U\Sigma V^\top V\Sigma^\top U^\top = U\Sigma\Sigma^\top U^\top = U \operatorname{diag}(\sigma_1^2, \ldots, \sigma_q^2, 0, \ldots, 0)U^\top.$$

Unfortunately, we cannot use the spectral decompositions of $A^\top A$ and $AA^\top$ to construct the SVD decomposition, because they are not unique. We can only use one and compute the other in a slightly different way.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Algorithm 13.17 — SVD Decomposition)</span></p>

**Input:** matrix $A \in \mathbb{R}^{m \times n}$.

1. Construct the spectral decomposition $V\Lambda V^\top$ of the matrix $A^\top A$;
2. $r \coloneqq \operatorname{rank}(A)$;
3. $\sigma_i \coloneqq \sqrt{\lambda_i}$, $i = 1, \ldots, r$;
4. $S \coloneqq \operatorname{diag}(\sigma_1, \ldots, \sigma_r)$, $\Sigma \coloneqq \begin{pmatrix} S & 0 \\ 0 & 0 \end{pmatrix}$;
5. let $V_1$ be the matrix formed by the first $r$ columns of $V$;
6. $U_1 \coloneqq AV_1 S^{-1}$;
7. extend $U_1$ to an orthogonal matrix $U = (U_1 \mid U_2)$;

**Output:** matrices $U, \Sigma, V^\top$ from the SVD decomposition of $A$ (we have $A = U\Sigma V^\top$).

</div>

*Proof.* From Proposition 13.15 we know that $\sigma_1, \ldots, \sigma_r$ are the desired singular values and clearly $V$ is orthogonal. We must prove that $U_1$ has orthonormal columns and $A = U\Sigma V^\top$.

From the equality $A^\top A = V\Lambda V^\top$ we derive $\Lambda = V^\top A^\top A V$ and by removing the last $n - r$ rows and columns we get the matrix $\operatorname{diag}(\lambda_1, \ldots, \lambda_r) = V_1^\top A^\top A V_1$. Now it is clear that $U_1$ has orthonormal columns, since

$$U_1^\top U_1 = (S^{-1})^\top V_1^\top A^\top A V_1 S^{-1} = (S^{-1})^\top S^2 S^{-1} = I_r.$$

It remains to show that $A = U\Sigma V^\top$, i.e., $\Sigma = U^\top AV$. Decompose $V = (V_1 \mid V_2)$. By removing the first $r$ rows and columns from the matrix $\Lambda = V^\top A^\top AV$ we get $0 = V_2^\top A^\top AV_2$, from which $AV_2 = 0$ (Corollary 8.47(1)). Now using the equality $AV_1 = U_1 S$ we have

$$U^\top AV = U^\top A(V_1 \mid V_2) = (U^\top U_1 S \mid U^\top AV_2) = \begin{pmatrix} S & 0 \\ 0 & 0 \end{pmatrix} = \Sigma.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 13.18 — SVD Decomposition)</span></p>

Let

$$A = \begin{pmatrix} 1 & 1 \\ 2 & 0 \\ 0 & -2 \end{pmatrix}.$$

Spectral decomposition of $A^\top A$:

$$A^\top A = \begin{pmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \end{pmatrix} \begin{pmatrix} 6 & 0 \\ 0 & 4 \end{pmatrix} \begin{pmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \end{pmatrix} \equiv V\Lambda V^\top.$$

Determination of $S$: $S = \begin{pmatrix} \sqrt{6} & 0 \\ 0 & 2 \end{pmatrix}$.

Determination of $U_1$ (in this example we have $V_1 = V$):

$$U_1 = AV_1 S^{-1} = \begin{pmatrix} \frac{\sqrt{3}}{3} & 0 \\ \frac{\sqrt{3}}{3} & \frac{\sqrt{2}}{2} \\ -\frac{\sqrt{3}}{3} & \frac{\sqrt{2}}{2} \end{pmatrix}.$$

Extension of $U_1$ to an orthogonal matrix $U$:

$$U = \begin{pmatrix} \frac{\sqrt{3}}{3} & 0 & -\frac{\sqrt{6}}{3} \\ \frac{\sqrt{3}}{3} & \frac{\sqrt{2}}{2} & \frac{\sqrt{6}}{6} \\ -\frac{\sqrt{3}}{3} & \frac{\sqrt{2}}{2} & -\frac{\sqrt{6}}{6} \end{pmatrix}.$$

Resulting SVD decomposition:

$$A = \begin{pmatrix} \frac{\sqrt{3}}{3} & 0 & -\frac{\sqrt{6}}{3} \\ \frac{\sqrt{3}}{3} & \frac{\sqrt{2}}{2} & \frac{\sqrt{6}}{6} \\ -\frac{\sqrt{3}}{3} & \frac{\sqrt{2}}{2} & -\frac{\sqrt{6}}{6} \end{pmatrix} \begin{pmatrix} \sqrt{6} & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \end{pmatrix} \equiv U\Sigma V^\top.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 13.19 — SVD Decomposition of Symmetric Matrices)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ be symmetric and let $A = Q\Lambda Q^\top$ be its spectral decomposition, where the eigenvalues on the diagonal of $\Lambda$ are sorted in decreasing order of absolute value. If moreover $A$ is positive definite, then the spectral decomposition is also its SVD decomposition $A = U\Sigma V^\top$, since we can choose $U = Q$, $\Sigma = \Lambda$, and $V = Q$. The singular values and eigenvalues of $A$ then coincide. If $A$ is not positive definite, then its singular values are the absolute values of the eigenvalues. The SVD decomposition can have the form $A = U\Sigma V^\top$, where $U = Q'$, $\Sigma = |\Lambda|$, $V = Q$, and the matrix $Q'$ is obtained from $Q$ by multiplying those columns corresponding to negative eigenvalues by $-1$.

</div>

Similarly to the QR decomposition, there also exists a reduced version for the SVD decomposition, the so-called *reduced* (or *thin*) SVD decomposition. Let $A = U\Sigma V^\top$ have rank $r > 0$. Decompose $U = (U_1 \mid U_2)$, $V = (V_1 \mid V_2)$ into the first $r$ columns and the rest, and further denote $S \coloneqq \operatorname{diag}(\sigma_1, \ldots, \sigma_r)$. Then

$$A = U\Sigma V^\top = (U_1 \quad U_2) \begin{pmatrix} S & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} V_1^\top \\ V_2^\top \end{pmatrix} = U_1 S V_1^\top.$$

The reduced SVD decomposition uses only part of the information from the SVD decomposition, but the essential part, from which we can reconstruct the full SVD decomposition (by extending $U_1$, $V_1$ to orthogonal matrices). We have already implicitly used the reduced SVD in the proof of Algorithm 13.17.

### 13.5 Applications of the SVD Decomposition

#### SVD and Orthogonalization

The SVD decomposition can be used to find an orthonormal basis of (not only) the column space $\mathcal{S}(A)$. Unlike previous approaches, we do not need to assume linear independence of the columns of $A$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 13.20)</span></p>

Let $A = U\Sigma V^\top = U_1 SV_1^\top$ be the full or reduced SVD decomposition of $A \in \mathbb{R}^{m \times n}$. Then

1. The columns of $U_1$ form an orthonormal basis of $\mathcal{S}(A)$.
2. The columns of $V_1$ form an orthonormal basis of $\mathcal{R}(A)$.
3. The columns of $V_2$ form an orthonormal basis of $\operatorname{Ker}(A)$.

</div>

*Proof.*

1. By multiplying the equation $A = U_1 SV_1^\top$ by $V_1$ from the right we get $AV_1 = U_1 S$. Now, $\mathcal{S}(A) \ni \mathcal{S}(AV_1) = \mathcal{S}(U_1 S) = \mathcal{S}(U_1)$ due to the nonsingularity of $S$. Since $\operatorname{rank}(A) = \operatorname{rank}(U_1)$, we have the equality $\mathcal{S}(A) = \mathcal{S}(U_1)$.
2. Follows from the previous part since $\mathcal{R}(A) = \mathcal{S}(A^\top)$ and $A^\top = V_1 S U_1^\top$ is the reduced SVD decomposition of the transposed matrix.
3. From the previous part we know that the columns of $V_1$ form an orthonormal basis of $\mathcal{R}(A) = \operatorname{Ker}(A)^\perp$. Therefore the columns of $V_2$, which extend the columns of $V_1$ to an orthonormal basis of $\mathbb{R}^n$, represent an orthonormal basis of $\operatorname{Ker}(A)$.

#### SVD and Projection onto a Subspace

Using the SVD decomposition, we can easily express the projection matrix onto the column (and row) space of a given matrix. Moreover, we do not need the assumption of linear independence of the columns of the matrix, which was required in Theorem 8.49.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 13.21)</span></p>

Let $A = U_1 SV_1^\top$ be the reduced SVD decomposition of $A \in \mathbb{R}^{m \times n}$. Then the projection matrix onto the

1. column space $\mathcal{S}(A)$ is $U_1 U_1^\top$,
2. row space $\mathcal{R}(A)$ is $V_1 V_1^\top$.

</div>

*Proof.*

1. By Theorem 13.20, $\mathcal{S}(A) = \mathcal{S}(U_1)$. The columns of $U_1$ form an orthonormal system, and therefore the projection matrix has by Remark 8.52 the form $U_1 U_1^\top$.
2. Follows from the previous part since $\mathcal{R}(A) = \mathcal{S}(A^\top)$.

#### SVD and the Geometry of Linear Mappings

Let $A \in \mathbb{R}^{n \times n}$ be a nonsingular matrix and let us study the image of the unit sphere under the mapping $x \mapsto Ax$. From the SVD decomposition $A = U\Sigma V^\top$ it follows that the linear mapping can be decomposed into a composition of three basic mappings: an orthogonal mapping with matrix $V^\top$, scaling by $\Sigma$, and an orthogonal mapping with matrix $U$. Specifically, the mapping with matrix $V^\top$ maps the sphere onto itself, $\Sigma$ deforms it into an ellipsoid, and $U$ rotates/reflects it. Thus the result will be an ellipsoid centered at the origin, with semi-axes in the directions of the columns of $U$ and lengths equal to $\sigma_1, \ldots, \sigma_n$.

The value $\frac{\sigma_1}{\sigma_n} \ge 1$ is called the *distortion measure* and quantitatively indicates how much the mapping deforms geometric objects. If the value equals 1, the ellipsoid will have the shape of a sphere, and conversely, the larger the value, the more elongated the ellipsoid will be. The significance of this value is not only geometric, however. In numerical mathematics, the ratio $\frac{\sigma_1}{\sigma_n}$ is called the *condition number*, and the larger it is, the more ill-conditioned the matrix $A$ is, in the sense that it exhibits poor numerical properties — rounding in floating-point computer arithmetic causes errors.

A rule of thumb says that if the condition number is of order $10^k$, then in computations with the matrix (inversion, solving systems, etc.) we lose accuracy by $k$ decimal places. Orthogonal matrices have condition number equal to 1, and therefore they are frequently used in numerical mathematics. On the other hand, for example, the Hilbert matrices from Example 3.51 have a very high condition number:

| $n$ | condition number of $H_n$ |
|-----|---------------------------|
| 3   | $\approx 500$ |
| 5   | $\approx 10^5$ |
| 10  | $\approx 10^{13}$ |
| 15  | $\approx 10^{17}$ |

#### SVD and Numerical Rank

The rank of a matrix $A$ equals the number of (positive) singular values. However, for computational purposes a very small positive number is treated as a practical zero. Let $\varepsilon > 0$; then the *numerical rank* of $A$ is $\max\lbrace s; \; \sigma_s > \varepsilon \rbrace$, i.e., the number of singular values greater than $\varepsilon$, the rest being treated as zero. For example, Matlab / Octave defines $\varepsilon \coloneqq \max\lbrace m, n \rbrace \cdot \sigma_1 \cdot eps$, where $eps \approx 2 \cdot 10^{-16}$ is the precision of computer arithmetic.

#### SVD and Low-Rank Approximation

Let $A \in \mathbb{R}^{m \times n}$ and let $A = U\Sigma V^\top$ be its SVD decomposition. If we keep the $k$ largest singular values and set the rest to zero $\sigma_{k+1} \coloneqq 0, \ldots, \sigma_r \coloneqq 0$, we obtain the matrix

$$A' = U \operatorname{diag}(\sigma_1, \ldots, \sigma_k, 0, \ldots, 0) V^\top$$

of rank $k$, which well approximates $A$. Moreover, this approximation is in a certain sense the best possible. That is, in a certain norm (see Section 13.7), among all matrices of rank $k$, it is precisely $A'$ that is closest to $A$.

#### SVD and Data Compression

We use the low-rank approximation from the previous paragraph for a simple method of lossy data compression. Suppose that the matrix $A \in \mathbb{R}^{m \times n}$ represents data that we want to compress. If $\operatorname{rank}(A) = r$, then for the reduced SVD decomposition $A = U_1 SV_1^\top$ we need to store $mr + r + nr = (m + n + 1)r$ values. With low-rank approximation $A \approx U \operatorname{diag}(\sigma_1, \ldots, \sigma_k, 0, \ldots, 0)V^\top$ it suffices to store only $(m + n + 1)k$ values. Thus the compression ratio is $k : r$. The smaller $k$, the less data we need to store. But on the other hand, a smaller $k$ means a worse approximation.

#### SVD and Measure of Regularity

As we mentioned in Remark 9.10, the determinant is not well suited as a measure of regularity. Singular values, on the other hand, are tailor-made for this purpose. Let $A \in \mathbb{R}^{n \times n}$. Then $\sigma_n$ gives the distance (in a certain norm, see Section 13.7) to the nearest singular matrix. So it is consistent with what we would expect from such a measure. Orthogonal matrices have measure 1, while Hilbert matrices have a small measure of regularity, i.e., they are nearly singular:

| $n$ | $\sigma_n(H_n)$ |
|-----|------------------|
| 3   | $\approx 0.0027$ |
| 5   | $\approx 10^{-6}$ |
| 10  | $\approx 10^{-13}$ |
| 15  | $\approx 10^{-18}$ |

### 13.6 Pseudoinverse Matrix

It is a natural endeavor to generalize the concept of matrix inverse to singular or rectangular matrices. Such a generalized inverse is called a *pseudoinverse* and several types exist. The most common is the so-called Moore–Penrose pseudoinverse, which is based on the SVD decomposition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 13.23 — Moore–Penrose Pseudoinverse)</span></p>

Let $A \in \mathbb{R}^{m \times n}$ be a matrix with reduced SVD decomposition $A = U_1 SV_1^\top$. If $A \neq 0$, then its *pseudoinverse* is $A^\dagger = V_1 S^{-1} U_1^\top \in \mathbb{R}^{n \times m}$. For $A = 0$ we define the pseudoinverse by $A^\dagger = A^\top$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 13.24)</span></p>

The pseudoinverse of a nonzero vector $a \in \mathbb{R}^n$ is $a^\dagger = \frac{1}{a^\top a} a^\top$, in particular e.g. $((1, 1, 1, 1)^\top)^\dagger = \frac{1}{4}(1, 1, 1, 1)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 13.25 — Properties of the Pseudoinverse)</span></p>

For a matrix $A \in \mathbb{R}^{m \times n}$ the following hold:

1. If $A$ is nonsingular, then $A^{-1} = A^\dagger$,
2. $(A^\dagger)^\dagger = A$,
3. $(A^\top)^\dagger = (A^\dagger)^\top$,
4. $A = AA^\dagger A$,
5. $A^\dagger = A^\dagger A A^\dagger$,
6. $AA^\dagger$ is symmetric,
7. $A^\dagger A$ is symmetric,
8. $A^\dagger = (A^\top A)^\dagger A^\top$,
9. if $A$ has linearly independent columns, then $A^\dagger = (A^\top A)^{-1} A^\top$,
10. if $A$ has linearly independent rows, then $A^\dagger = A^\top (AA^\top)^{-1}$.

</div>

*Proof.* The properties are proved simply from the definition. For illustration, we show only two properties, leaving the rest to the reader.

(4) From the definition $AA^\dagger A = U_1 SV_1^\top V_1 S^{-1} U_1^\top U_1 SV_1^\top = U_1 SS^{-1}SV_1^\top = U_1 SV_1^\top = A$.

(9) By assumption, $V_1$ is square, hence orthogonal. Then

$$(A^\top A)^{-1} = (V_1 SU_1^\top U_1 SV_1^\top)^{-1} = (V_1 S^2 V_1^\top)^{-1} = V_1 S^{-2} V_1^\top,$$

from which $(A^\top A)^{-1}A^\top = V_1 S^{-2} V_1^\top V_1 S U_1^\top = V_1 S^{-1} U_1^\top = A^\dagger$.

The first property says that this is indeed a generalization of the classical inverse. Properties (4)–(7) are also interesting in that they provide an alternative definition of the pseudoinverse; it can equivalently be defined as the matrix satisfying conditions (4)–(7), and surprisingly such a matrix always exists and is unique.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 13.26)</span></p>

Let $A \in \mathbb{R}^{m \times n}$. Then the projection matrix onto the

1. column space $\mathcal{S}(A)$ is $AA^\dagger$,
2. row space $\mathcal{R}(A)$ is $A^\dagger A$,
3. kernel $\operatorname{Ker}(A)$ is $I_n - A^\dagger A$.

</div>

*Proof.*

1. Using the reduced SVD decomposition $A = U_1 SV_1^\top$ we compute $AA^\dagger = U_1 SV_1^\top V_1 S^{-1}U_1^\top = U_1 U_1^\top$. By Theorem 13.21, this is the desired projection matrix.
2. Analogously to the previous part, $A^\dagger A = V_1 V_1^\top$, which is the projection matrix onto $\mathcal{R}(A)$.
3. Follows from Theorem 8.54 and the property $\operatorname{Ker}(A) = \mathcal{R}(A)^\perp$ (Theorem 8.45).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 13.27 — Pseudoinverse Matrix and Linear Mappings)</span></p>

Consider the linear mapping $f(x) = Ax$, where $A \in \mathbb{R}^{m \times n}$.

1. If we restrict the domain of $f(x)$ to the space $\mathcal{R}(A)$ only, we obtain an isomorphism between $\mathcal{R}(A)$ and $f(\mathbb{R}^n)$.
2. The inverse mapping to this isomorphism has the form $y \mapsto A^\dagger y$.

</div>

*Proof.*

1. This part was proved in Proposition 8.48, but we present a different proof. The mapping with restricted domain is surjective, because by Corollary 8.47(2), $f(\mathbb{R}^n) = \mathcal{S}(A) = \mathcal{R}(A^\top) = \mathcal{R}(AA^\top) = \lbrace Ay; \; y \in \mathcal{R}(A) \rbrace = f(\mathcal{R}(A))$. Since the spaces $f(\mathbb{R}^n)$ and $\mathcal{R}(A)$ have the same dimension (Theorem 5.68), the mapping must be an isomorphism.
2. By Theorem 13.20(2), every vector $x \in \mathcal{R}(A)$ under the mapping $x \mapsto A^\dagger Ax$ is mapped to $A^\dagger Ax = x$. Therefore $A^\dagger$ is the matrix of the inverse mapping to $x \mapsto Ax$.

The most notable property of the pseudoinverse lies in the description of the solution set of consistent systems and the set of approximate solutions by the least squares method for inconsistent systems. In both cases, $A^\dagger b$ is in a certain sense a distinguished solution.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 13.28 — Pseudoinverse Matrix and Solving Systems of Equations)</span></p>

Let $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$, and let $X$ be the solution set of the system $Ax = b$. If $X \neq \emptyset$, then

$$X = A^\dagger b + \operatorname{Ker}(A).$$

where $\operatorname{Ker}(A) = \mathcal{S}(I_n - A^\dagger A)$.

Moreover, among all vectors in $X$, $A^\dagger b$ has the smallest Euclidean norm, and it is the only solution with this property.

</div>

*Proof.* "$=$" Let $x \in X$, i.e., $Ax = b$. Then by Proposition 13.25(4), $AA^\dagger b = AA^\dagger Ax = Ax = b$, so $A^\dagger b \in X$. By Theorem 7.6, $X = x_0 + \operatorname{Ker}(A)$, where $x_0$ is any solution. By Theorem 13.20(3), $\operatorname{Ker}(A) = \mathcal{S}(I_n - A^\dagger A)$ and we can choose $x_0 = A^\dagger b$.

"Norm." Let $x \in X$. By Theorem 13.27(2), $A^\dagger b = A^\dagger Ax \in \mathcal{R}(A)$ and furthermore $\mathcal{R}(A) = \operatorname{Ker}(A)^\perp$. Now by the Pythagorean theorem, for every $y \in \mathcal{S}(I_n - A^\dagger A)$ we have

$$\|A^\dagger b + y\|_2^2 = \|A^\dagger b\|_2^2 + \|y\|_2^2 \ge \|A^\dagger b\|_2^2.$$

Thus $A^\dagger b$ has the smallest Euclidean norm. Every other vector in $X$ has a larger norm, because $y \neq 0$ implies $\|y\|_2 > 0$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 13.29 — Pseudoinverse Matrix and the Least Squares Method)</span></p>

Let $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$, and let $X$ be the set of approximate solutions of the system $Ax = b$ by the least squares method. Then

$$X = A^\dagger b + \operatorname{Ker}(A).$$

Moreover, among all vectors in $X$, $A^\dagger b$ has the smallest Euclidean norm, and it is the only solution with this property.

</div>

*Proof.* The set of approximate solutions of the system $Ax = b$ by the least squares method is described by the system $A^\top Ax = A^\top b$ and is nonempty, see Theorem 8.59. By Theorem 13.28 we have

$$X = (A^\top A)^\dagger (A^\top b) + \operatorname{Ker}(A^\top A).$$

Since by Proposition 13.25(8), $(A^\top A)^\dagger A^\top = A^\dagger$ and by Corollary 8.47, $\operatorname{Ker}(A^\top A) = \operatorname{Ker}(A)$, the set $X$ has the required description and the required property.

The preceding two theorems thus say, among other things, that $A^\dagger b$ is a distinguished vector. If the system $Ax = b$ has a solution, then it is its solution with minimal norm. And if the system $Ax = b$ has no solution, then it is its approximate solution (again with minimal norm) by the least squares method. Moreover, the assumption of linear independence of the columns of $A$ is not needed.

### 13.7 Matrix Norm

Now we return to norms and look at how to introduce a norm for matrices. Although we discussed norms in Section 8.1, the largest singular value of a matrix represents an important matrix norm, which is why we place this part alongside the SVD decomposition.

In principle, matrices from $\mathbb{R}^{m \times n}$ form a vector space, so we can view matrices as vectors. However, for a matrix norm one additional property is considered, hence we have a special definition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definition 13.30 — Matrix Norm)</span></p>

A class of mappings $\|\cdot\| \colon \mathbb{R}^{m \times n} \to \mathbb{R}$ is a real *matrix norm* if it is a norm for all $m, n$ and additionally satisfies:

$$\|AB\| \le \|A\| \cdot \|B\| \quad \text{for all } A \in \mathbb{R}^{m \times p}, \; B \in \mathbb{R}^{p \times n}.$$

</div>

The first example of a matrix norm is the *Frobenius norm*:

$$\|A\|_F \coloneqq \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}^2}.$$

It is essentially the Euclidean norm of the vector formed by all entries of $A$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 13.31 — Frobenius Norm)</span></p>

Let $A \in \mathbb{R}^{m \times n}$ have singular values $\sigma_1, \ldots, \sigma_r$. Then $\|A\|_F = \sqrt{\sum_{i=1}^{r} \sigma_i^2}$.

</div>

*Proof.* By Proposition 10.11 and Proposition 13.15 we have $\|A\|_F = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n} a_{ij}^2} = \sqrt{\operatorname{trace}(A^\top A)} = \sqrt{\sum_{i=1}^{r}\sigma_i^2}$.

The second example of a matrix norm is the *matrix $p$-norm*:

$$\|A\|_p = \max_{x:\|x\|_p=1} \|Ax\|_p.$$

In this definition we use the vector $p$-norm. We can think of the resulting norm as follows: We map the unit sphere (in the $p$-norm, i.e., vectors satisfying $\|x\|_p = 1$) by the linear mapping $x \mapsto Ax$ and in the image we select the vector with the largest norm. For different values of $p$ we obtain different matrix norms.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 13.32 — Matrix $p$-Norms)</span></p>

Let $A \in \mathbb{R}^{m \times n}$. Then the matrix $p$-norms for $p \in \lbrace 1, 2, \infty \rbrace$ have the form

1. $\|A\|_2 = \sigma_1(A)$ (the largest singular value),
2. $\|A\|_1 = \max_{j=1,\ldots,n} \sum_{i=1}^{m} |a_{ij}|$,
3. $\|A\|_\infty = \max_{i=1,\ldots,m} \sum_{j=1}^{n} |a_{ij}| = \|A^\top\|_1$.

</div>

*Proof.*

1. As we already mentioned, $\|A\|_2$ is the magnitude of the largest point of the ellipse formed by the image of the unit sphere under the mapping $x \mapsto Ax$. From Section 13.5 (SVD and the geometry of linear mappings) we know that this value is $\sigma_1(A)$.
2. Denote $c \coloneqq \max_{j=1,\ldots,n} \sum_{i=1}^m |a_{ij}|$. For any $x$ such that $\|x\|_1 = 1$, we have $\|Ax\|_1 = \sum_{i=1}^m \left|\sum_{j=1}^n a_{ij}x_j\right| \le \sum_{i=1}^m \sum_{j=1}^n |a_{ij}||x_j| = \sum_{j=1}^n |x_j|\left(\sum_{i=1}^m |a_{ij}|\right) \le \sum_{j=1}^n |x_j| c = c$. At the same time, equality $\|Ax\|_1 = c$ is attained by a suitable choice of a unit vector $x = e_i$.
3. Denote $c \coloneqq \max_{i=1,\ldots,m} \sum_{j=1}^n |a_{ij}|$. For any $x$ such that $\|x\|_\infty = 1$, we have $\|Ax\|_\infty = \max_{i=1,\ldots,m} \left|\sum_{j=1}^n a_{ij}x_j\right| \le \max_{i=1,\ldots,m} \sum_{j=1}^n |a_{ij}||x_j| \le \max_{i=1,\ldots,m} \sum_{j=1}^n |a_{ij}| = c$. At the same time, equality $\|Ax\|_\infty = c$ is attained by a suitable choice of a vector $x \in \lbrace \pm 1 \rbrace^n$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 13.33)</span></p>

Let $v \in \mathbb{R}^n$ and let $p$ be arbitrary. The expression $\|v\|_p$ can denote both the vector and the matrix $p$-norm, if we regard $v$ as a matrix with one column. This is not a problem, however, since both give the same value:

$$\|v\|_p = \max_{x \in \mathbb{R}:\|x\|=1} \|vx\|_p = \max_{x \in \lbrace \pm 1 \rbrace} \|\pm v\|_p = \|v\|_p.$$

</div>

We know from Theorem 8.66 that multiplying a vector by an orthogonal matrix does not change its Euclidean norm. We now generalize this statement to the Frobenius and matrix 2-norm.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Proposition 13.34)</span></p>

Let $A \in \mathbb{R}^{m \times n}$ and let $Q \in \mathbb{R}^{m \times m}$, $R \in \mathbb{R}^{n \times n}$ be orthogonal. Then

1. $\|QAR\|_F = \|A\|_F$,
2. $\|QAR\|_2 = \|A\|_2$.

</div>

*Proof.*

1. Analogously to the proof of Proposition 13.31, we have $\|QAR\|_F^2 = \operatorname{trace}((QAR)^\top(QAR)) = \operatorname{trace}(R^\top A^\top Q^\top QAR) = \operatorname{trace}(R^\top A^\top AR) = \operatorname{trace}(A^\top ARR^\top) = \operatorname{trace}(A^\top A) = \|A\|_F^2$, where we additionally used the fact that $\operatorname{trace}(BC) = \operatorname{trace}(CB)$ for every $B, C \in \mathbb{R}^{n \times n}$.
2. Using the substitution $x \coloneqq R^\top y$ we derive $\|QAR\|_2 = \max_{x:\|x\|_2=1} \|QARx\|_2 = \max_{x:\|x\|_2=1} \|ARx\|_2 = \max_{y:\|R^\top y\|_2=1} \|Ay\|_2 = \max_{y:\|y\|_2=1} \|Ay\|_2 = \|A\|_2$.

Matrix norms appear in various contexts. First we show that they give bounds on the magnitude of eigenvalues.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 13.35 — Bound on the Spectral Radius via Norm)</span></p>

Let $A \in \mathbb{R}^{n \times n}$. Then for every matrix norm we have $\rho(A) \le \|A\|$.

</div>

*Proof.* Let $\lambda \in \mathbb{C}$ be any eigenvalue and $x$ the corresponding eigenvector of $A$, i.e., $Ax = \lambda x$. Define the matrix $X \coloneqq (x \mid o \mid \cdots \mid o)$. Since $AX = \lambda X$, we can derive

$$|\lambda| \cdot \|X\| = \|\lambda X\| = \|AX\| \le \|A\| \cdot \|X\|.$$

Dividing by $\|X\| \neq 0$ we obtain $|\lambda| \le \|A\|$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 13.36)</span></p>

Consider the matrix

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 1 & 2 & 3 \\ 3 & 6 & 9 \end{pmatrix}.$$

Its spectral radius and various types of norms have the values:

$$\rho(A) = 12, \quad \|A\|_F = \sqrt{154} \approx 12.4097, \quad \|A\|_2 = \sqrt{154} \approx 12.4097, \quad \|A\|_1 = 15, \quad \|A\|_\infty = 18.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 13.37 — Computational Complexity)</span></p>

Let $A \in \mathbb{R}^{n \times n}$. It is not hard to see that computing the Frobenius norm and the $p$-norm for $p \in \lbrace 1, \infty \rbrace$ has asymptotic complexity $2n^2$. The matrix 2-norm is computed only iteratively and commonly used methods have cubic complexity (with a certain coefficient), similarly to eigenvalues of a matrix (see the beginning of Section 10.7). Nevertheless, it is the default matrix norm used by, e.g., Matlab or Octave.

</div>

Another very interesting property of singular values is that $\sigma_i$ gives, in the 2-norm, the distance of the matrix to the nearest matrix of rank at most $i - 1$. Hidden in the proof of the following theorem is also how to construct this matrix — not coincidentally, it is the matrix from the low-rank approximation (cf. Section 13.5).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 13.38 — Interpretation of Singular Values)</span></p>

Let $A \in \mathbb{R}^{m \times n}$ have singular values $\sigma_1, \ldots, \sigma_r$. Then

$$\sigma_i = \min \lbrace \|A - B\|_2; \; B \in \mathbb{R}^{m \times n}, \; \operatorname{rank}(B) \le i - 1 \rbrace$$

for every $i = 1, \ldots, r$.
]`
</div>

*Proof.* Inequality "$\ge$". Let $A = U\Sigma V^\top$ be the SVD decomposition of $A$. Define the matrix $B \coloneqq U \operatorname{diag}(\sigma_1, \ldots, \sigma_{i-1}, 0, \ldots, 0) V^\top$. Then

$$\|A - B\|_2 = \|U \operatorname{diag}(0, \ldots, 0, \sigma_i, \ldots, \sigma_n) V^\top\|_2 = \|\operatorname{diag}(0, \ldots, 0, \sigma_i, \ldots, \sigma_n)\|_2 = \sigma_i.$$

Inequality "$\le$". Let $B \in \mathbb{R}^{n \times n}$ be any matrix of rank at most $i - 1$ and we show that $\|A - B\|_2 \ge \sigma_i$. Let $V_1$ consist of the first $i$ columns of $V$. Let $o \neq z \in \operatorname{Ker}(B) \cap \mathcal{S}(V_1)$, i.e., $Bz = o$, and furthermore normalize $z$ so that $\|z\|_2 = 1$. Such a vector exists because $\dim\operatorname{Ker}(B) \ge n - i + 1$ and $\dim\mathcal{S}(V_1) = i$. Then

$$\|A - B\|_2^2 = \max_{x:\|x\|_2=1} \|(A - B)x\|_2^2 \ge \|(A - B)z\|_2^2 = \|Az\|_2^2 = \|U\Sigma V^\top z\|_2^2.$$

Since $z \in \mathcal{S}(V_1)$, we can write $z = Vy$ for some vector $y = (y_1, \ldots, y_i, 0, \ldots, 0)^\top$, where $\|y\|_2 = \|V^\top z\|_2 = \|z\|_2 = 1$. Now

$$\|U\Sigma V^\top z\|_2^2 = \|U\Sigma V^\top Vy\|_2^2 = \|\Sigma y\|_2^2 = \sum_{j=1}^{i} \sigma_j^2 y_j^2 \ge \sum_{j=1}^{i} \sigma_i^2 y_j^2 = \sigma_i^2 \|y\|_2^2 = \sigma_i^2.$$

In particular, the smallest singular value $\sigma_n$ of a matrix $A \in \mathbb{R}^{n \times n}$ gives the distance to the nearest singular matrix. This means that the matrix $A + C$ is nonsingular for all matrices $C \in \mathbb{R}^{n \times n}$ satisfying $\|C\|_2 < \sigma_n$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example 13.39)</span></p>

Consider the matrix

$$A = \begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & 1 & -1 & -1 \\ 1 & -1 & 1 & -1 \\ 1 & -1 & -1 & 1 \end{pmatrix}.$$

Its smallest singular value is $\sigma_4 = 2$ (in fact, all singular values are equal, since $A$ is twice an orthogonal matrix). Therefore the matrix remains nonsingular even if we add any matrix with 2-norm less than 2.

</div>

### Summary of Chapter 13

Matrix decompositions are a very powerful tool in both theoretical and computational computer science. There exist many decompositions; among the most notable are the QR and SVD decompositions. It is no coincidence that both use orthogonal matrices in the decomposition.

The QR decomposition expresses any real matrix as a product of an orthogonal and an upper triangular matrix. This decomposition can be computationally easily obtained using Gram–Schmidt orthogonalization or Householder transformations. The applications of the QR decomposition are countless: solving systems of linear equations, finding an orthonormal basis, constructing the orthogonal projection matrix, solving by the least squares method, computing eigenvalues, ...

The SVD decomposition has analogous properties. It decomposes a given real matrix into a product of an orthogonal, a diagonal (but not necessarily square!) and an orthogonal matrix. Its applications are similar to those of the QR decomposition, but it additionally tells us something about the geometry of linear mappings and provides a tool for data approximation and compression. It also allows us to naturally extend the concept of matrix inverse to not necessarily nonsingular matrices.

The singular values (= the numbers on the diagonal matrix from the SVD decomposition) then provide essential information about the linear mapping $x \mapsto Ax$, about the matrix $A$ itself, and also about the data it represents. The singular values tell us how much the linear mapping degenerates objects, what the distance to the nearest singular matrix is and what numerical properties the matrix has, and the largest singular value represents a commonly used matrix norm.
