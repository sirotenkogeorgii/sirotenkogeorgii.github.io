---
layout: default
title: "Calculus of Variations — Gelfand & Fomin"
date: 2025-01-01
excerpt: Notes on the classical text by Gelfand and Fomin covering functionals, Euler's equation, and variational methods.
tags:
  - calculus-of-variations
  - analysis
  - mathematics
---

# Calculus of Variations — Gelfand & Fomin

## Chapter 1: Elements of the Theory

### 1. Functionals. Some Simple Variational Problems

Variable quantities called *functionals* play an important role in many problems arising in analysis, mechanics, geometry, etc. By a *functional*, we mean a correspondence which assigns a definite (real) number to each function (or curve) belonging to some class. Thus, a functional is a kind of function where the independent variable is itself a function (or curve).

Examples of functionals:

1. The **length of a curve** is a functional defined on the set of all rectifiable plane curves.
2. The **ordinate of the center of mass** of a homogeneous curve is a functional on the set of rectifiable plane curves.
3. Given all paths joining two points $A$ and $B$ in the plane, and a velocity field $v(x, y)$, the **time to traverse the path** is a functional.
4. For any continuously differentiable function $y(x)$ on $[a, b]$, the expression

$$J[y] = \int_a^b y'^2(x)\,dx$$

defines a functional on the set of all such functions $y(x)$.

More generally, let $F(x, y, z)$ be a continuous function of three variables. Then

$$J[y] = \int_a^b F[x, y(x), y'(x)]\,dx,$$

where $y(x)$ ranges over all continuously differentiable functions on $[a, b]$, defines a functional. By choosing different $F(x, y, z)$, we obtain different functionals. For example, if $F(x, y, z) = \sqrt{1 + z^2}$, then $J[y]$ is the length of the curve $y = y(x)$, while if $F(x, y, z) = z^2$, then $J[y]$ reduces to the fourth example above.

All the above problems involve functionals of the form

$$\int_a^b F(x, y, y')\,dx.$$

Such functionals have a "localization property": if we divide the curve $y = y(x)$ into parts and compute the functional for each part, the sum of the values equals the value for the whole curve.

We now indicate some typical *variational problems*, i.e., problems involving the determination of maxima and minima of functionals.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Shortest Curve / Geodesic)</span></p>

*Find the shortest plane curve joining two given points $A$ and $B$*, i.e., find the curve $y = y(x)$ for which the functional

$$\int_a^b \sqrt{1 + y'^2}\,dx$$

achieves its minimum. The solution is the straight line segment joining $A$ and $B$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Brachistochrone Problem)</span></p>

Let $A$ and $B$ be two fixed points. Find the curve along which a particle slides under gravity from $A$ to $B$ in the least time. This is the *brachistochrone problem*, posed by John Bernoulli in 1696. It played an important role in the development of the calculus of variations and was solved by John Bernoulli, James Bernoulli, Newton, and L'Hospital. The brachistochrone turns out to be a **cycloid**, lying in the vertical plane and passing through $A$ and $B$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Isoperimetric Problem)</span></p>

*Among all closed curves of a given length $l$, find the curve enclosing the greatest area.* This was solved by Euler: the required curve is a **circle**.

</div>

To understand the basic meaning of the calculus of variations, it is important to see how variational problems relate to problems of classical analysis, i.e., to the study of functions of $n$ variables. Consider a functional

$$J[y] = \int_a^b F(x, y, y')\,dx, \quad y(a) = A, \quad y(b) = B.$$

Using the points $a = x\_0, x\_1, \ldots, x\_n, x\_{n+1} = b$, we divide $[a, b]$ into $n + 1$ equal parts and approximate $y(x)$ by a polygonal line with vertices $(x\_i, y(x\_i))$. Then

$$J(y_1, \ldots, y_n) = \sum_{i=1}^{n+1} F\!\left(x_i, y_i, \frac{y_i - y_{i-1}}{h}\right) h,$$

where $y\_i = y(x\_i)$ and $h = x\_i - x\_{i-1}$. Thus, as an approximation, the variational problem reduces to finding extrema of a function of $n$ variables $J(y\_1, \ldots, y\_n)$. Euler made extensive use of this *method of finite differences*. By passing to the limit as $n \to \infty$, functionals can be regarded as "functions of infinitely many variables," and the calculus of variations can be regarded as the corresponding analog of differential calculus.

### 2. Function Spaces

Just as in the study of functions of $n$ variables it is convenient to use geometric language by regarding $(y\_1, \ldots, y\_n)$ as a point in $n$-dimensional space, we regard each function $y(x)$ belonging to some class as a point in some space. Spaces whose elements are functions are called *function spaces*.

In the case of function spaces, there is no single "universal" space. The nature of the problem determines the choice of the function space.

The concept of continuity plays an important role for functionals, just as for ordinary functions. To formulate this, we need a concept of "closeness" for elements in a function space, which is done by introducing a *norm*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Normed Linear Space)</span></p>

A **linear space** $\mathscr{R}$ is a set of elements $x, y, z, \ldots$ for which operations of addition and multiplication by real numbers $\alpha, \beta, \ldots$ are defined and obey the usual axioms (commutativity, associativity, existence of zero and additive inverses, distributivity, and $1 \cdot x = x$).

A linear space $\mathscr{R}$ is said to be **normed** if each element $x \in \mathscr{R}$ is assigned a nonnegative number $\lVert x \rVert$, called the **norm** of $x$, such that

1. $\lVert x \rVert = 0$ if and only if $x = 0$;
2. $\lVert \alpha x \rVert = \lvert \alpha \rvert \, \lVert x \rVert$;
3. $\lVert x + y \rVert \leqslant \lVert x \rVert + \lVert y \rVert$ (triangle inequality).

The distance between $x$ and $y$ is defined as $\lVert x - y \rVert$.

</div>

The following normed linear spaces are important for the calculus of variations:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Space $\mathscr{C}(a, b)$)</span></p>

The space $\mathscr{C} = \mathscr{C}(a, b)$ consists of all continuous functions $y(x)$ defined on $[a, b]$, with the norm

$$\lVert y \rVert_0 = \max_{a \leqslant x \leqslant b} |y(x)|.$$

Two functions in $\mathscr{C}$ are "close" if their values are uniformly close on $[a, b]$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Space $\mathscr{D}\_1(a, b)$)</span></p>

The space $\mathscr{D}\_1 = \mathscr{D}\_1(a, b)$ consists of all functions $y(x)$ defined on $[a, b]$ which are continuous and have continuous first derivatives. The norm is

$$\lVert y \rVert_1 = \max_{a \leqslant x \leqslant b} |y(x)| + \max_{a \leqslant x \leqslant b} |y'(x)|.$$

Two functions in $\mathscr{D}\_1$ are close if both the functions themselves and their first derivatives are close together.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Space $\mathscr{D}\_n(a, b)$)</span></p>

The space $\mathscr{D}\_n = \mathscr{D}\_n(a, b)$ consists of all functions $y(x)$ on $[a, b]$ which are continuous and have continuous derivatives up to order $n$ inclusive. The norm is

$$\lVert y \rVert_n = \sum_{i=0}^{n} \max_{a \leqslant x \leqslant b} |y^{(i)}(x)|,$$

where $y^{(0)}(x)$ denotes $y(x)$ itself. Two functions in $\mathscr{D}\_n$ are close if the functions and all their derivatives up to order $n$ are close together.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Continuity of a Functional)</span></p>

The functional $J[y]$ is said to be **continuous** at the point $\hat{y} \in \mathscr{R}$ if for any $\varepsilon > 0$, there is a $\delta > 0$ such that

$$|J[y] - J[\hat{y}]| < \varepsilon,$$

provided that $\lVert y - \hat{y} \rVert < \delta$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Semicontinuity)</span></p>

The inequality $\|J[y] - J[\hat{y}]\| < \varepsilon$ is equivalent to the pair $J[y] - J[\hat{y}] > -\varepsilon$ and $J[y] - J[\hat{y}] < \varepsilon$. If only the first holds, $J[y]$ is called *lower semicontinuous* at $\hat{y}$; if only the second, *upper semicontinuous*.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Choice of Function Space)</span></p>

It might seem that the largest space $\mathscr{C}$ would suffice for all variational problems. However, this is not the case. For example, a functional of the form $\int\_a^b F(x, y, y')\,dx$ (e.g., arc length) will be continuous if we use the norm of $\mathscr{D}\_1$, but in general will not be continuous in the norm of $\mathscr{C}$. Therefore, in studying functionals of various types, it is reasonable to use various function spaces.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Admissible Functions)</span></p>

In many variational problems, we deal with functionals defined on sets of functions which do not form linear spaces. The set of functions (or curves) satisfying the constraints of a given variational problem is called the set of *admissible functions* (or *admissible curves*). For example, the admissible curves for the "simplest" variational problem are the smooth plane curves passing through two fixed points, and the sum of two such curves does not pass through the two points.

</div>

### 3. The Variation of a Functional. A Necessary Condition for an Extremum

#### 3.1. Linear Functionals and the Variation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Linear Functional)</span></p>

Given a normed linear space $\mathscr{R}$, let each element $h \in \mathscr{R}$ be assigned a number $\varphi[h]$, i.e., let $\varphi[h]$ be a functional defined on $\mathscr{R}$. Then $\varphi[h]$ is said to be a **(continuous) linear functional** if

1. $\varphi[\alpha h] = \alpha \varphi[h]$ for any $h \in \mathscr{R}$ and any real number $\alpha$;
2. $\varphi[h\_1 + h\_2] = \varphi[h\_1] + \varphi[h\_2]$ for any $h\_1, h\_2 \in \mathscr{R}$;
3. $\varphi[h]$ is continuous (for all $h \in \mathscr{R}$).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Linear Functionals)</span></p>

1. If $h(x) \in \mathscr{C}(a, b)$ and we fix a point $x\_0 \in [a, b]$, then $\varphi[h] = h(x\_0)$ is a linear functional on $\mathscr{C}(a, b)$.
2. The integral $\varphi[h] = \int\_a^b h(x)\,dx$ is a linear functional on $\mathscr{C}(a, b)$.
3. The integral $\varphi[h] = \int\_a^b \alpha(x) h(x)\,dx$, where $\alpha(x)$ is a fixed function in $\mathscr{C}(a, b)$, is a linear functional on $\mathscr{C}(a, b)$.
4. More generally, $\varphi[h] = \int\_a^b [\alpha\_0(x)h(x) + \alpha\_1(x)h'(x) + \cdots + \alpha\_n(x)h^{(n)}(x)]\,dx$, where the $\alpha\_i(x)$ are fixed functions in $\mathscr{C}(a, b)$, defines a linear functional on $\mathscr{D}\_n(a, b)$.

</div>

#### Fundamental Lemmas

Suppose the linear functional from Example 4 vanishes for all $h(x)$ belonging to some class. What can be said about the functions $\alpha\_i(x)$? Some typical results in this direction are given by the following lemmas.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Fundamental Lemma 1)</span></p>

If $\alpha(x)$ is continuous in $[a, b]$, and if

$$\int_a^b \alpha(x) h(x)\,dx = 0$$

for every function $h(x) \in \mathscr{C}(a, b)$ such that $h(a) = h(b) = 0$, then $\alpha(x) = 0$ for all $x$ in $[a, b]$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Suppose $\alpha(x)$ is nonzero, say positive, at some point in $[a, b]$. Then $\alpha(x)$ is also positive in some interval $[x\_1, x\_2] \subset [a, b]$. Set $h(x) = (x - x\_1)(x\_2 - x)$ for $x \in [x\_1, x\_2]$ and $h(x) = 0$ otherwise. Then $h(x)$ satisfies the conditions of the lemma. However,

$$\int_a^b \alpha(x)h(x)\,dx = \int_{x_1}^{x_2} \alpha(x)(x - x_1)(x_2 - x)\,dx > 0,$$

since the integrand is positive (except at $x\_1$ and $x\_2$). This contradiction proves the lemma.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Fundamental Lemma 2)</span></p>

If $\alpha(x)$ is continuous in $[a, b]$, and if

$$\int_a^b \alpha(x) h'(x)\,dx = 0$$

for every function $h(x) \in \mathscr{D}\_1(a, b)$ such that $h(a) = h(b) = 0$, then $\alpha(x) = c$ for all $x$ in $[a, b]$, where $c$ is a constant.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $c$ be the constant defined by the condition $\int\_a^b [\alpha(x) - c]\,dx = 0$, and let

$$h(x) = \int_a^x [\alpha(\xi) - c]\,d\xi,$$

so that $h(x)$ automatically belongs to $\mathscr{D}\_1(a, b)$ and satisfies $h(a) = h(b) = 0$. Then on one hand,

$$\int_a^b [\alpha(x) - c] h'(x)\,dx = \int_a^b \alpha(x) h'(x)\,dx - c[h(b) - h(a)] = 0,$$

while on the other hand,

$$\int_a^b [\alpha(x) - c] h'(x)\,dx = \int_a^b [\alpha(x) - c]^2\,dx.$$

It follows that $\alpha(x) - c = 0$, i.e., $\alpha(x) = c$, for all $x$ in $[a, b]$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Fundamental Lemma 3)</span></p>

If $\alpha(x)$ is continuous in $[a, b]$, and if

$$\int_a^b \alpha(x) h''(x)\,dx = 0$$

for every function $h(x) \in \mathscr{D}\_2(a, b)$ such that $h(a) = h(b) = 0$ and $h'(a) = h'(b) = 0$, then $\alpha(x) = c\_0 + c\_1 x$ for all $x$ in $[a, b]$, where $c\_0$ and $c\_1$ are constants.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Fundamental Lemma 4)</span></p>

If $\alpha(x)$ and $\beta(x)$ are continuous in $[a, b]$, and if

$$\int_a^b [\alpha(x)h(x) + \beta(x)h'(x)]\,dx = 0$$

for every function $h(x) \in \mathscr{D}\_1(a, b)$ such that $h(a) = h(b) = 0$, then $\beta(x)$ is differentiable and $\beta'(x) = \alpha(x)$ for all $x$ in $[a, b]$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Lemma 4</summary>

Setting $A(x) = \int\_a^x \alpha(\xi)\,d\xi$ and integrating by parts, we find that

$$\int_a^b \alpha(x)h(x)\,dx = -\int_a^b A(x)h'(x)\,dx,$$

so the equation becomes

$$\int_a^b [-A(x) + \beta(x)]h'(x)\,dx = 0.$$

According to Lemma 2, this implies that $\beta(x) - A(x) = \mathrm{const}$, and hence by the definition of $A(x)$, $\beta'(x) = \alpha(x)$, as asserted. Note that the differentiability of $\beta(x)$ was not assumed in advance.

</details>
</div>

#### 3.2. The Variation (Differential) of a Functional

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Increment and Variation of a Functional)</span></p>

Let $J[y]$ be a functional defined on some normed linear space, and let

$$\Delta J[h] = J[y + h] - J[y]$$

be its *increment*, corresponding to the increment $h = h(x)$ of the "independent variable" $y = y(x)$. Suppose that

$$\Delta J[h] = \varphi[h] + \varepsilon \lVert h \rVert,$$

where $\varphi[h]$ is a linear functional and $\varepsilon \to 0$ as $\lVert h \rVert \to 0$. Then the functional $J[y]$ is said to be **differentiable**, and the principal linear part of the increment $\Delta J[h]$, i.e., the linear functional $\varphi[h]$, is called the **variation** (or **differential**) of $J[y]$ and is denoted by $\delta J[h]$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Uniqueness of the Differential)</span></p>

The differential of a differentiable functional is unique.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

If $\varphi[h]$ is a linear functional and $\varphi[h]/\lVert h \rVert \to 0$ as $\lVert h \rVert \to 0$, then $\varphi[h] \equiv 0$. In fact, suppose $\varphi[h\_0] \neq 0$ for some $h\_0 \neq 0$. Setting $h\_n = h\_0/n$ and $\lambda = \varphi[h\_0]/\lVert h\_0 \rVert$, we have $\lVert h\_n \rVert \to 0$ as $n \to \infty$, but

$$\lim_{n \to \infty} \frac{\varphi[h_n]}{\lVert h_n \rVert} = \lim_{n \to \infty} \frac{n \varphi[h_0]}{n \lVert h_0 \rVert} = \lambda \neq 0,$$

a contradiction. Now, if $\Delta J[h] = \varphi\_1[h] + \varepsilon\_1 \lVert h \rVert = \varphi\_2[h] + \varepsilon\_2 \lVert h \rVert$, then $\varphi\_1[h] - \varphi\_2[h] = \varepsilon\_2 \lVert h \rVert$, and since $\varphi\_1 - \varphi\_2$ is linear with $(\varphi\_1[h] - \varphi\_2[h])/\lVert h \rVert \to 0$, by the above it follows that $\varphi\_1[h] - \varphi\_2[h] \equiv 0$.

</details>
</div>

#### Weak and Strong Extrema

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Weak and Strong Extremum)</span></p>

We shall say that the functional $J[y]$ has a **weak extremum** for $y = \hat{y}$ if there exists an $\varepsilon > 0$ such that $J[y] - J[\hat{y}]$ has the same sign for all $y$ in the domain of definition of the functional which satisfy $\lVert y - \hat{y} \rVert\_1 < \varepsilon$, where $\lVert \cdot \rVert\_1$ denotes the norm in $\mathscr{D}\_1$.

We say that $J[y]$ has a **strong extremum** for $y = \hat{y}$ if there exists an $\varepsilon > 0$ such that $J[y] - J[\hat{y}]$ has the same sign for all $y$ satisfying $\lVert y - \hat{y} \rVert\_0 < \varepsilon$, where $\lVert \cdot \rVert\_0$ denotes the norm in $\mathscr{C}$.

Every strong extremum is simultaneously a weak extremum, since if $\lVert y - \hat{y} \rVert\_1 < \varepsilon$, then $\lVert y - \hat{y} \rVert\_0 < \varepsilon$, *a fortiori*. However, the converse is not true in general: a weak extremum may not be a strong extremum.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Necessary Condition for an Extremum)</span></p>

A necessary condition for the differentiable functional $J[y]$ to have an extremum for $y = \hat{y}$ is that its variation vanish for $y = \hat{y}$, i.e., that

$$\delta J[h] = 0$$

for all admissible $h$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Suppose $J[y]$ has a minimum for $y = \hat{y}$. By definition of the variation $\delta J[h]$, we have $\Delta J[h] = \delta J[h] + \varepsilon \lVert h \rVert$, where $\varepsilon \to 0$ as $\lVert h \rVert \to 0$. Thus, for sufficiently small $\lVert h \rVert$, $\Delta J[h]$ has the same sign as $\delta J[h]$. If $\delta J[h\_0] \neq 0$ for some admissible $h\_0$, then $\delta J[-\alpha h\_0] = -\delta J[\alpha h\_0]$, so $\Delta J$ can be made to have either sign for arbitrarily small $\lVert h \rVert$. This contradicts the assumption that $J[y]$ has a minimum for $y = \hat{y}$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

In elementary analysis, for a function to have a minimum it is necessary not only that its first differential vanish ($df = 0$), but also that its second differential be nonnegative. The analogous problem for functionals will be postponed until Chapter 5.

</div>

### 4. The Simplest Variational Problem. Euler's Equation

#### 4.1. Derivation of Euler's Equation

We begin our study of concrete variational problems by considering the "simplest" variational problem:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Simplest Variational Problem)</span></p>

Let $F(x, y, z)$ be a function with continuous first and second (partial) derivatives with respect to all its arguments. Then, among all functions $y(x)$ which are continuously differentiable for $a \leqslant x \leqslant b$ and satisfy the boundary conditions

$$y(a) = A, \qquad y(b) = B,$$

find the function for which the functional

$$J[y] = \int_a^b F(x, y, y')\,dx$$

has a weak extremum.

</div>

Suppose we give $y(x)$ an increment $h(x)$, where $h(a) = h(b) = 0$. Then

$$\Delta J = J[y + h] - J[y] = \int_a^b [F(x, y + h, y' + h') - F(x, y, y')]\,dx.$$

By Taylor's theorem,

$$\Delta J = \int_a^b [F_y(x, y, y')h + F_{y'}(x, y, y')h']\,dx + \cdots,$$

where the subscripts denote partial derivatives and the dots denote higher-order terms. The integral is the principal linear part of the increment $\Delta J$, and hence the variation of $J[y]$ is

$$\delta J = \int_a^b [F_y(x, y, y')h + F_{y'}(x, y, y')h']\,dx.$$

According to Theorem 2 (necessary condition for an extremum), a necessary condition for $J[y]$ to have an extremum for $y = y(x)$ is that

$$\delta J = \int_a^b (F_y h + F_{y'} h')\,dx = 0$$

for all admissible $h$. But according to Lemma 4, this implies that

$$F_y - \frac{d}{dx} F_{y'} = 0,$$

a result known as **Euler's equation**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Euler's Equation)</span></p>

Let $J[y]$ be a functional of the form

$$\int_a^b F(x, y, y')\,dx,$$

defined on the set of functions $y(x)$ which have continuous first derivatives in $[a, b]$ and satisfy the boundary conditions $y(a) = A$, $y(b) = B$. Then a necessary condition for $J[y]$ to have an extremum for a given function $y(x)$ is that $y(x)$ satisfy Euler's equation

$$F_y - \frac{d}{dx} F_{y'} = 0.$$

</div>

The integral curves of Euler's equation are called **extremals**. Since Euler's equation is a second-order differential equation, its solution will in general depend on two arbitrary constants, which are determined from the boundary conditions $y(a) = A$, $y(b) = B$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Necessary vs. Sufficient)</span></p>

Euler's equation gives a necessary condition for an extremum, not a sufficient one. Sufficient conditions will be considered in Chapter 5. In many cases, however, the existence of an extremum is clear from the physical or geometric meaning of the problem. If in such a case there exists only one extremal satisfying the boundary conditions, then this extremal must be the curve for which the extremum is achieved.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bernstein's Existence and Uniqueness Theorem)</span></p>

If the functions $F$, $F\_y$ and $F\_{y'}$ are continuous at every finite point $(x, y)$ for any finite $y'$, and if a constant $k > 0$ and functions

$$\alpha = \alpha(x, y) \geqslant 0, \qquad \beta = \beta(x, y) \geqslant 0$$

(which are bounded in every finite region of the plane) can be found such that

$$F_{y'y'}(x, y, y') > k, \qquad |F(x, y, y')| \leqslant \alpha y'^2 + \beta,$$

then one and only one integral curve of the equation $y'' = F(x, y, y')$ passes through any two points $(a, A)$ and $(b, B)$ with different abscissas ($a \neq b$).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Regularity of Extremals)</span></p>

Suppose $y = y(x)$ has a continuous first derivative and satisfies Euler's equation

$$F_y - \frac{d}{dx} F_{y'} = 0.$$

Then, if $F(x, y, y')$ has continuous first and second derivatives with respect to all its arguments at all points $(x, y)$ where $F\_{y'y'}[x, y(x), y'(x)] \neq 0$, it follows that $y(x)$ has a continuous second derivative.

</div>

#### 4.2. Special Cases of Euler's Equation

Euler's equation is in general a second-order differential equation. In certain special cases it can be reduced to a first-order equation, or solved entirely in terms of quadratures.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Special Cases of Euler's Equation)</span></p>

**Case 1.** $F$ does not depend on $y$, i.e., the functional is $\int\_a^b F(x, y')\,dx$. Then Euler's equation becomes $\frac{d}{dx} F\_{y'} = 0$, which has the first integral

$$F_{y'} = C,$$

a first-order ODE. Solving for $y'$, we get $y' = f(x, C)$, and $y$ is found by a quadrature.

**Case 2.** $F$ does not depend on $x$, i.e., the functional is $\int\_a^b F(y, y')\,dx$. Then Euler's equation has the first integral

$$F - y' F_{y'} = C.$$

This follows from multiplying the expanded Euler equation $F\_y - F\_{y'y}y' - F\_{y'y'}y'' = 0$ by $y'$ to obtain $\frac{d}{dx}(F - y'F\_{y'}) = 0$.

**Case 3.** $F$ does not depend on $y'$. Then Euler's equation becomes $F\_y(x, y) = 0$, which is not a differential equation but a "finite" equation whose solution consists of one or more curves $y = y(x)$.

**Case 4.** Functionals of the form $\int\_a^b f(x, y)\sqrt{1 + y'^2}\,dx$, representing integrals with respect to arc length $s$ ($ds = \sqrt{1 + y'^2}\,dx$). Euler's equation simplifies to

$$f_y - f_x y' - f \frac{y''}{1 + y'^2} = 0.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Shortest Curve Revisited)</span></p>

Consider $J[y] = \int\_1^2 \frac{\sqrt{1 + y'^2}}{x}\,dx$, with $y(1) = 0$, $y(2) = 1$. The integrand does not contain $y$, so by Case 1, $F\_{y'} = C$. Thus

$$\frac{y'}{x\sqrt{1 + y'^2}} = C,$$

so that $y'^2(1 - C^2 x^2) = C^2 x^2$, giving $y' = \frac{Cx}{\sqrt{1 - C^2 x^2}}$. Integrating:

$$y = \int \frac{Cx\,dx}{\sqrt{1 - C^2 x^2}} = \frac{1}{C}\sqrt{1 - C^2 x^2} + C_1.$$

This gives $(y - C\_1)^2 + x^2 = 1/C^2$, a circle with center on the $y$-axis. From $y(1) = 0$ and $y(2) = 1$, we find $C = 1/\sqrt{5}$ and $C\_1 = 2$, so the final solution is

$$(y - 2)^2 + x^2 = 5.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Surface of Revolution of Minimum Area — Catenary)</span></p>

Among all curves joining two given points $(x\_0, y\_0)$ and $(x\_1, y\_1)$, find the one which generates the surface of revolution of minimum area when rotated about the $x$-axis. The area is

$$2\pi \int_{x_0}^{x_1} y\sqrt{1 + y'^2}\,dx.$$

Since the integrand does not depend explicitly on $x$, by Case 2, $F - y'F\_{y'} = C$, i.e.,

$$y\sqrt{1 + y'^2} - y \cdot \frac{y'^2}{\sqrt{1 + y'^2}} = C,$$

which simplifies to $y = C\sqrt{1 + y'^2}$, so $y' = \sqrt{y^2/C^2 - 1}$. Separating variables and integrating:

$$x + C_1 = C \ln \frac{y + \sqrt{y^2 - C^2}}{C},$$

giving

$$y = C \cosh \frac{x + C_1}{C}.$$

The curve is a **catenary**, and the surface of revolution is a **catenoid**. The constants $C$ and $C\_1$ are determined by the boundary conditions $y(x\_0) = y\_0$ and $y(x\_1) = y\_1$.

Depending on the positions of the two points, three cases are possible:

1. If a single catenary passes through both points, it is the solution.
2. If two extremals pass through both points, one corresponds to the minimum area surface and the other does not.
3. If no catenary passes through both points, there is no smooth surface of revolution of minimum area; the minimum is achieved by the "broken extremal" (Goldschmidt discontinuous solution).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Functional $(x - y)^2$)</span></p>

For the functional

$$J[y] = \int_a^b (x - y)^2\,dx,$$

Euler's equation reduces to the finite equation $F\_y = -2(x - y) = 0$ (Case 3), whose solution is the straight line $y = x$. The integral $J[y]$ vanishes along this line.

</div>

### 5. The Case of Several Variables

So far we have considered functionals depending on functions of one variable, i.e., on curves. In many problems, one encounters functionals depending on functions of several independent variables, i.e., on surfaces. Such multidimensional problems will be considered in detail in Chapter 7. For now, we give an idea of how the formulation and solution carries over to the case of functionals depending on surfaces.

Consider two independent variables (the considerations remain the same for $n$ variables). Let $F(x, y, z, p, q)$ be a function with continuous first and second (partial) derivatives with respect to all its arguments, and consider the functional

$$J[z] = \iint_R F(x, y, z, z_x, z_y)\,dx\,dy,$$

where $R$ is a closed region, $z\_x$ and $z\_y$ are the partial derivatives of $z$. We seek a function $z(x, y)$ such that:

1. $z(x, y)$ and its first and second derivatives are continuous in $R$;
2. $z(x, y)$ takes given values on the boundary $\Gamma$ of $R$;
3. The functional $J[z]$ has an extremum for $z = z(x, y)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Two-Dimensional Fundamental Lemma)</span></p>

If $\alpha(x, y)$ is a fixed continuous function in a closed region $R$, and if the integral

$$\iint_R \alpha(x, y) h(x, y)\,dx\,dy$$

vanishes for every function $h(x, y)$ which has continuous first and second derivatives in $R$ and equals zero on the boundary $\Gamma$ of $R$, then $\alpha(x, y) = 0$ everywhere in $R$.

</div>

The variation of the functional $J[z]$ is computed as follows. Let $h(x, y)$ be an arbitrary function with continuous first and second derivatives, vanishing on the boundary $\Gamma$. Then

$$\delta J = \iint_R (F_z h + F_{z_x} h_x + F_{z_y} h_y)\,dx\,dy.$$

Using Green's theorem (integration by parts in two dimensions) and the fact that $h = 0$ on $\Gamma$, one obtains

$$\delta J = \iint_R \left(F_z - \frac{\partial}{\partial x} F_{z_x} - \frac{\partial}{\partial y} F_{z_y}\right) h(x, y)\,dx\,dy.$$

Setting $\delta J = 0$ and applying the two-dimensional fundamental lemma, we obtain **Euler's equation for several variables**:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Euler's Equation for Several Variables)</span></p>

A necessary condition for the functional

$$J[z] = \iint_R F(x, y, z, z_x, z_y)\,dx\,dy$$

to have an extremum is that $z(x, y)$ satisfy the second-order partial differential equation

$$F_z - \frac{\partial}{\partial x} F_{z_x} - \frac{\partial}{\partial y} F_{z_y} = 0.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Minimal Surface / Plateau's Problem)</span></p>

*Find the surface of least area spanned by a given contour.* This reduces to minimizing the functional

$$J[z] = \iint_R \sqrt{1 + z_x^2 + z_y^2}\,dx\,dy.$$

Euler's equation has the form

$$r(1 + q^2) - 2spq + t(1 + p^2) = 0,$$

where $p = z\_x$, $q = z\_y$, $r = z\_{xx}$, $s = z\_{xy}$, $t = z\_{yy}$. This equation states that the **mean curvature** of the required surface is zero:

$$M = \frac{1}{2}\left(\frac{1}{\kappa_1} + \frac{1}{\kappa_2}\right) = \frac{Eg - 2Ff + Ge}{2(EG - F^2)} = 0.$$

Surfaces with zero mean curvature are called **minimal surfaces**.

</div>

### 6. A Simple Variable End Point Problem

There are many other kinds of variational problems besides the "simplest" variational problem considered so far. Here we consider the *variable end point problem*, a particular case of which can be stated as follows: *Among all curves whose end points lie on two given vertical lines $x = a$ and $x = b$, find the curve for which the functional*

$$J[y] = \int_a^b F(x, y, y')\,dx$$

*has an extremum.*

We calculate the variation $\delta J$. Unlike the fixed end point problem, $h(x)$ need no longer vanish at $a$ and $b$, so integration by parts gives

$$\delta J = \int_a^b \!\left(F_y - \frac{d}{dx} F_{y'}\right) h(x)\,dx + F_{y'} h(x)\Big|_{x=a}^{x=b}.$$

We first consider functions $h(x)$ such that $h(a) = h(b) = 0$. Then, as in the simplest variational problem, the condition $\delta J = 0$ implies that

$$F_y - \frac{d}{dx} F_{y'} = 0,$$

i.e., $y$ must be an extremal (a solution of Euler's equation). But if $y$ is an extremal, the integral in $\delta J$ vanishes, and the condition $\delta J = 0$ reduces to

$$F_{y'}\big|_{x=b}\, h(b) - F_{y'}\big|_{x=a}\, h(a) = 0.$$

Since $h(x)$ is arbitrary, this gives the **natural boundary conditions**:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Natural Boundary Conditions)</span></p>

$$F_{y'}\big|_{x=a} = 0, \qquad F_{y'}\big|_{x=b} = 0.$$

Thus, to solve the variable end point problem, we must first find the general integral of Euler's equation, and then use the natural boundary conditions to determine the values of the arbitrary constants.

In the **mixed case**, where one end is fixed (say $y(a) = A$) and the other is variable (the end point lies on $x = b$), the conditions reduce to Euler's equation, the fixed condition $y(a) = A$, and the single natural boundary condition $F\_{y'}\big\|\_{x=b} = 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Brachistochrone with Variable End Point)</span></p>

Starting from the point $P = (a, A)$, a heavy particle slides down a curve in the vertical plane. Find the curve such that the particle reaches the vertical line $x = b$ ($\neq a$) in the shortest time. (This is a variant of the brachistochrone problem.)

Assuming the origin coincides with the starting point, the velocity of motion along the curve equals $v = ds/dt = \sqrt{1 + y'^2}\,(dx/dt)$, and by energy conservation $v = \sqrt{2gy}$, so

$$T = \int \frac{\sqrt{1 + y'^2}}{\sqrt{2gy}}\,dx.$$

The general solution of the corresponding Euler equation consists of a family of cycloids

$$x = r(\theta - \sin \theta) + c, \qquad y = r(1 - \cos \theta).$$

Since the curve passes through the origin, $c = 0$. The natural boundary condition $F\_{y'}\big\|\_{x=b} = 0$ gives $y' = 0$ at $x = b$, meaning the tangent to the curve at the right end point must be horizontal. It follows that $r = b/\pi$, so the required curve is

$$x = \frac{b}{\pi}(\theta - \sin \theta), \qquad y = \frac{b}{\pi}(1 - \cos \theta).$$

</div>

### 7. The Variational Derivative

In Sec. 3.2 we introduced the concept of the differential of a functional. We now introduce the concept of the *variational* (or *functional*) *derivative*, which plays the same role for functionals as the concept of the partial derivative plays for functions of $n$ variables.

Consider functionals of the type

$$J[y] = \int_a^b F(x, y, y')\,dx, \qquad y(a) = A, \quad y(b) = B,$$

corresponding to the simplest variational problem. Our approach is to go from the variational problem to an $n$-dimensional problem, and then pass to the limit $n \to \infty$.

We divide $[a, b]$ into $n + 1$ equal subintervals by introducing points $x\_0 = a, x\_1, \ldots, x\_n, x\_{n+1} = b$ with $x\_{i+1} - x\_i = \Delta x$, and replace the smooth function $y(x)$ by a polygonal line with vertices $(x\_i, y\_i)$, where $y\_i = y(x\_i)$. Then $J[y]$ is approximated by

$$J(y_1, \ldots, y_n) \approx \sum_{i=0}^{n} F\!\left(x_i, y_i, \frac{y_{i+1} - y_i}{\Delta x}\right) \Delta x.$$

Computing the partial derivative $\partial J / \partial y\_k$, only the terms with $i = k$ and $i = k - 1$ contribute:

$$\frac{\partial J}{\partial y_k} = F_y\!\left(x_k, y_k, \frac{y_{k+1} - y_k}{\Delta x}\right)\Delta x + F_{y'}\!\left(x_{k-1}, y_{k-1}, \frac{y_k - y_{k-1}}{\Delta x}\right) - F_{y'}\!\left(x_k, y_k, \frac{y_{k+1} - y_k}{\Delta x}\right).$$

Dividing by $\Delta x$:

$$\frac{\partial J}{\partial y_k \Delta x} = F_y\!\left(x_k, y_k, \frac{y_{k+1} - y_k}{\Delta x}\right) - \frac{1}{\Delta x}\!\left[F_{y'}\!\left(x_k, y_k, \frac{y_{k+1} - y_k}{\Delta x}\right) - F_{y'}\!\left(x_{k-1}, y_{k-1}, \frac{y_k - y_{k-1}}{\Delta x}\right)\right].$$

As $\Delta x \to 0$, this converges to the limit

$$\frac{\delta J}{\delta y} = F_y(x, y, y') - \frac{d}{dx} F_{y'}(x, y, y'),$$

called the **variational derivative** of the functional $J[y]$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Variational Derivative)</span></p>

Let $J[y]$ be a functional depending on $y(x)$, and suppose we give $y(x)$ an increment $h(x)$ which is different from zero only in a neighborhood of a point $x\_0$. Dividing the corresponding increment $J[y + h] - J[y]$ by the area $\Delta\sigma$ lying between the curve $y = h(x)$ and the $x$-axis, we obtain the ratio

$$\frac{J[y + h] - J[y]}{\Delta\sigma}.$$

If, as $\Delta\sigma \to 0$ (with both $\max \|h(x)\|$ and the length of the interval where $h(x)$ is nonvanishing going to zero), this ratio converges to a limit, that limit is called the **variational derivative** of $J[y]$ at the point $x\_0$ [for the curve $y = y(x)$], and is denoted by

$$\frac{\delta J}{\delta y}\bigg|_{x = x_0}.$$

</div>

The variational derivative obeys all the familiar rules of ordinary derivatives (e.g., sums, products, composite functions, etc.).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Variation via the Variational Derivative)</span></p>

If $h(x)$ is different from zero in a neighborhood of $x\_0$ and $\Delta\sigma$ is the area between $y = h(x)$ and the $x$-axis, then

$$\Delta J = J[y + h] - J[y] = \left\lbrace\frac{\delta J}{\delta y}\bigg|_{x = x_0} + \varepsilon\right\rbrace \Delta\sigma,$$

where $\varepsilon \to 0$ as both $\max \|h(x)\|$ and the length of the interval in which $h(x)$ is nonvanishing go to zero. It follows that the differential (or variation) of $J[y]$ at the point $x\_0$ is

$$\delta J = \frac{\delta J}{\delta y}\bigg|_{x = x_0} \Delta\sigma.$$

Thus, the meaning of Euler's equation is that the variational derivative of the functional under consideration should vanish at every point. This is the analog of the condition in elementary analysis that for a function of $n$ variables to have an extremum, all its partial derivatives must vanish.

</div>

### 8. Invariance of Euler's Equation

Suppose that instead of the rectangular plane coordinates $x$ and $y$, we introduce curvilinear coordinates $u$ and $v$, where

$$x = x(u, v), \qquad y = y(u, v), \qquad \begin{vmatrix} x_u & x_v \\ y_u & y_v \end{vmatrix} \neq 0.$$

Then the curve $y = y(x)$ in the $xy$-plane corresponds to a curve $v = v(u)$ in the $uv$-plane. Under this change of variables, the functional

$$J[y] = \int_a^b F(x, y, y')\,dx$$

goes into the functional

$$J_1[v] = \int_{a_1}^{b_1} F\!\left[x(u, v),\, y(u, v),\, \frac{y_u + y_v v'}{x_u + x_v v'}\right](x_u + x_v v')\,du = \int_{a_1}^{b_1} F_1(u, v, v')\,du.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Invariance of Euler's Equation)</span></p>

If $y = y(x)$ satisfies the Euler equation

$$\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'} = 0$$

corresponding to the original functional $J[y]$, then the corresponding curve $v = v(u)$ satisfies the Euler equation

$$\frac{\partial F_1}{\partial v} - \frac{d}{du}\frac{\partial F_1}{\partial v'} = 0$$

corresponding to the new functional $J\_1[v]$. In other words, whether or not a curve is an extremal is a property which is independent of the choice of coordinate system.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (sketch)</summary>

Let $\Delta\sigma$ denote the area bounded by the curves $y = y(x)$ and $y = y(x) + h(x)$, and let $\Delta\sigma\_1$ denote the area bounded by the corresponding curves $v = v(u)$ and $v = v(u) + \eta(u)$ in the $uv$-plane. By the standard formula for the transformation of areas, in the limit as $\Delta\sigma, \Delta\sigma\_1 \to 0$ the ratio $\Delta\sigma/\Delta\sigma\_1$ approaches the Jacobian

$$\begin{vmatrix} x_u & x_v \\ y_u & y_v \end{vmatrix},$$

which by hypothesis is nonzero. Thus, if

$$\lim_{\Delta\sigma \to 0} \frac{J[y + h] - J[y]}{\Delta\sigma} = 0,$$

then

$$\lim_{\Delta\sigma_1 \to 0} \frac{J_1[v + \eta] - J_1[v]}{\Delta\sigma_1} = 0$$

as well. It follows that $v(u)$ satisfies Euler's equation for $J\_1[v]$ whenever $y(x)$ satisfies Euler's equation for $J[y]$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Use of Invariance)</span></p>

Because of the invariance property, changes of variables can be made directly in the integral representing the functional rather than in Euler's equation itself. One then writes Euler's equation for the new integral.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Geodesics in Polar Coordinates)</span></p>

Consider the functional

$$J[r] = \int_{\varphi_0}^{\varphi_1} \sqrt{r^2 + r'^2}\,d\varphi,$$

where $r = r(\varphi)$. The corresponding Euler equation has the form

$$\frac{r}{\sqrt{r^2 + r'^2}} - \frac{d}{d\varphi}\frac{r'}{\sqrt{r^2 + r'^2}} = 0.$$

Making the change of variables $x = r\cos\varphi$, $y = r\sin\varphi$, the functional transforms into

$$\int_{x_0}^{x_1} \sqrt{1 + y'^2}\,dx,$$

which has the Euler equation $y'' = 0$, with general solution $y = \alpha x + \beta$. Therefore, the solution of the original problem in polar coordinates is

$$r\sin\varphi = \alpha r\cos\varphi + \beta,$$

i.e., straight lines, as expected for geodesics in the plane.

</div>

## Chapter 2: Further Generalizations

This chapter extends the simplest variational problem to several settings: functionals depending on several unknown functions (Sec. 9), problems in parametric form (Sec. 10), problems involving higher derivatives (Sec. 11), and problems with subsidiary conditions (Sec. 12).

### 9. The Fixed End Point Problem for $n$ Unknown Functions

Let $F(x, y\_1, \ldots, y\_n, z\_1, \ldots, z\_n)$ be a function with continuous first and second (partial) derivatives with respect to all its arguments. Consider the problem of finding necessary conditions for an extremum of a functional of the form

$$J[y_1, \ldots, y_n] = \int_a^b F(x, y_1, \ldots, y_n, y_1', \ldots, y_n')\,dx,$$

which depends on $n$ continuously differentiable functions $y\_1(x), \ldots, y\_n(x)$ satisfying the boundary conditions

$$y_i(a) = A_i, \quad y_i(b) = B_i \qquad (i = 1, \ldots, n).$$

In other words, we are looking for an extremum of the functional defined on the set of smooth curves joining two fixed points in $(n + 1)$-dimensional Euclidean space $\mathscr{E}\_{n+1}$. The problem of finding *geodesics*, i.e., shortest curves joining two points of some manifold, is of this type. The same kind of problem arises in geometric optics, in finding the paths along which light rays propagate in an inhomogeneous medium. In fact, according to *Fermat's principle*, light goes from a point $P\_0$ to a point $P\_1$ along the path for which the transit time is the smallest.

To find necessary conditions for the functional to have an extremum, we replace each $y\_i(x)$ by a "varied" function $y\_i(x) + h\_i(x)$. The *variation* $\delta J$ of the functional $J[y\_1, \ldots, y\_n]$, i.e., the expression linear in $h\_i$, $h\_i'$ $(i = 1, \ldots, n)$ which differs from the increment $\Delta J$ by a quantity of order higher than 1, is

$$\delta J = \int_a^b \sum_{i=1}^{n} (F_{y_i} h_i + F_{y_i'} h_i')\,dx.$$

Since all the increments $h\_i(x)$ are independent, we can choose one of them quite arbitrarily (as long as the boundary conditions are satisfied), setting all the others equal to zero. Therefore, the necessary condition $\delta J = 0$ for an extremum implies

$$\int_a^b (F_{y_i} h_i + F_{y_i'} h_i')\,dx = 0 \qquad (i = 1, \ldots, n).$$

Using Lemma 4 of Sec. 3.1, we obtain the following system of Euler equations:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Euler Equations for $n$ Unknown Functions)</span></p>

A necessary condition for the curve $y\_i = y\_i(x)$ $(i = 1, \ldots, n)$ to be an extremal of the functional

$$\int_a^b F(x, y_1, \ldots, y_n, y_1', \ldots, y_n')\,dx$$

is that the functions $y\_i(x)$ satisfy the Euler equations

$$F_{y_i} - \frac{d}{dx} F_{y_i'} = 0 \qquad (i = 1, \ldots, n).$$

Since this is a system of $n$ second-order differential equations, its general solution contains $2n$ arbitrary constants, which are determined from the boundary conditions.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1</span><span class="math-callout__name">(Equivalent Functionals)</span></p>

Two different integrands $F$ can lead to the same system of Euler equations. In fact, let $\Phi = \Phi(x, y\_1, \ldots, y\_n)$ be any twice differentiable function, and let

$$\Psi(x, y_1, \ldots, y_n, y_1', \ldots, y_n') = \frac{\partial \Phi}{\partial x} + \sum_{i=1}^{n} \frac{\partial \Phi}{\partial y_i} y_i'.$$

Then by direct calculation $\frac{\partial}{\partial y\_i} - \frac{d}{dx}\!\left(\frac{\partial}{\partial y\_i'}\right)$ applied to $\Psi$ gives zero. Hence the functionals

$$\int_a^b F(x, y_1, \ldots, y_n, y_1', \ldots, y_n')\,dx$$

and

$$\int_a^b [F(x, y_1, \ldots, y_n, y_1', \ldots, y_n') + \Psi(x, y_1, \ldots, y_n, y_1', \ldots, y_n')]\,dx$$

lead to the same system of Euler equations. Two functionals are said to be *equivalent* if they have the same extremals.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1</span><span class="math-callout__name">(Propagation of Light in an Inhomogeneous Medium)</span></p>

Suppose that three-dimensional space is filled with an optically inhomogeneous medium, such that the velocity of propagation of light at each point is some function $v(x, y, z)$ of the coordinates of the point. According to Fermat's principle, light goes from one point to another along the curve for which the transit time of the light is the smallest. If the curve joining two points $A$ and $B$ is specified by the equations $y = y(x)$, $z = z(x)$, the time it takes light to traverse the curve equals

$$\int_a^b \frac{\sqrt{1 + y'^2 + z'^2}}{v(x, y, z)}\,dx.$$

Writing the system of Euler equations for this functional, i.e.,

$$\frac{\partial v}{\partial y}\frac{\sqrt{1 + y'^2 + z'^2}}{v^2} + \frac{d}{dx}\frac{y'}{v\sqrt{1 + y'^2 + z'^2}} = 0,$$

$$\frac{\partial v}{\partial z}\frac{\sqrt{1 + y'^2 + z'^2}}{v^2} + \frac{d}{dx}\frac{z'}{v\sqrt{1 + y'^2 + z'^2}} = 0,$$

we obtain the differential equations for the curves along which the light propagates.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2</span><span class="math-callout__name">(Geodesics on a Surface)</span></p>

Suppose we have a surface $\sigma$ specified by a vector equation $\mathbf{r} = \mathbf{r}(u, v)$. The shortest curve lying on $\sigma$ and connecting two points of $\sigma$ is called the *geodesic* connecting the two points. The equations for the geodesics of $\sigma$ are the Euler equations of the corresponding variational problem, i.e., the problem of finding the minimum distance (measured along $\sigma$) between two points of $\sigma$.

A curve lying on the surface can be specified by the equations $u = u(t)$, $v = v(t)$. The arc length between the points corresponding to $t\_1$ and $t\_2$ equals

$$J[u, v] = \int_{t_1}^{t_2} \sqrt{Eu'^2 + 2Fu'v' + Gv'^2}\,dt,$$

where $E$, $F$ and $G$ are the coefficients of the first fundamental (quadratic) form of the surface, i.e., $E = \mathbf{r}\_u \cdot \mathbf{r}\_u$, $F = \mathbf{r}\_u \cdot \mathbf{r}\_v$, $G = \mathbf{r}\_v \cdot \mathbf{r}\_v$.

Writing the Euler equations for this functional, we obtain

$$\frac{E_u u'^2 + 2F_u u'v' + G_u v'^2}{\sqrt{Eu'^2 + 2Fu'v' + Gv'^2}} - \frac{d}{dt}\frac{2(Eu' + Fv')}{\sqrt{Eu'^2 + 2Fu'v' + Gv'^2}} = 0,$$

$$\frac{E_v u'^2 + 2F_v u'v' + G_v v'^2}{\sqrt{Eu'^2 + 2Fu'v' + Gv'^2}} - \frac{d}{dt}\frac{2(Fu' + Gv')}{\sqrt{Eu'^2 + 2Fu'v' + Gv'^2}} = 0.$$

As a simple illustration, consider the geodesics of the circular cylinder $\mathbf{r} = (a\cos\varphi, a\sin\varphi, z)$, where $\varphi$ and $z$ play the role of the parameters $u$ and $v$. Since $E = a^2$, $F = 0$, $G = 1$, the geodesics of the cylinder have the equations

$$\frac{d}{dt}\frac{a^2\varphi'}{\sqrt{a^2\varphi'^2 + z'^2}} = 0, \qquad \frac{d}{dt}\frac{z'}{\sqrt{a^2\varphi'^2 + z'^2}} = 0,$$

i.e., $\frac{a^2\varphi'}{\sqrt{a^2\varphi'^2 + z'^2}} = C\_1$ and $\frac{z'}{\sqrt{a^2\varphi'^2 + z'^2}} = C\_2$. Dividing the second equation by the first, we obtain $\frac{dz}{d\varphi} = c\_1$, which has the solution $z = c\_1\varphi + c\_2$, representing a two-parameter family of helical lines lying on the cylinder.

</div>

### 10. Variational Problems in Parametric Form

So far, we have considered functionals of curves given by explicit equations, e.g., $y = y(x)$. However, it is often more convenient to consider functionals of curves given in parametric form, and in fact we have already encountered this case in Example 2 of Section 9 (geodesics on a surface). Moreover, in problems involving closed curves (like the isoperimetric problem), it is usually impossible to get along without representing the curves in parametric form.

Suppose that the functional

$$\int_{t_0}^{t_1} F(x, y, y')\,dx$$

has its argument $y$ given as a curve in parametric form, rather than in the form $y = y(x)$. Then the functional can be written as

$$\int_{t_0}^{t_1} F\!\left[x(t), y(t), \frac{\dot{y}(t)}{\dot{x}(t)}\right]\dot{x}(t)\,dt = \int_{t_0}^{t_1} \Phi(x, y, \dot{x}, \dot{y})\,dt$$

(where the overdot denotes differentiation with respect to $t$), i.e., as a functional depending on two unknown functions $x(t)$ and $y(t)$. The function $\Phi$ appearing in the right-hand side does not involve $t$ explicitly, and is *positive-homogeneous of degree 1* in $\dot{x}$ and $\dot{y}$, which means that

$$\Phi(x, y, \lambda\dot{x}, \lambda\dot{y}) \equiv \lambda\Phi(x, y, \dot{x}, \dot{y})$$

for every $\lambda > 0$.

Conversely, let $\int\_{t\_0}^{t\_1} \Phi(x, y, \dot{x}, \dot{y})\,dt$ be a functional whose integrand $\Phi$ does not involve $t$ explicitly and is positive-homogeneous of degree 1 in $\dot{x}$ and $\dot{y}$. We now show that the value of such a functional depends only on the curve in the $xy$-plane defined by the parametric equations $x = x(t)$, $y = y(t)$, and not on the functions $x(t)$, $y(t)$ themselves, i.e., that if we go from $t$ to some new parameter $\tau$ by setting $t = t(\tau)$, where $dt/d\tau > 0$, the value of the functional does not change.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Parametric Invariance)</span></p>

A necessary and sufficient condition for the functional

$$\int_{t_0}^{t_1} \Phi(t, x, y, \dot{x}, \dot{y})\,dt$$

to depend only on the curve in the $xy$-plane defined by the parametric equations $x = x(t)$, $y = y(t)$, and not on the choice of the parametric representation of the curve, is that the integrand $\Phi$ should not involve $t$ explicitly and should be a positive-homogeneous function of degree 1 in $\dot{x}$ and $\dot{y}$.

</div>

Now, suppose some parameterization of the curve $y = y(x)$ reduces the functional to the form

$$\int_{t_0}^{t_1} F\!\left(x, y, \frac{\dot{y}}{\dot{x}}\right)\dot{x}\,dt = \int_{t_0}^{t_1} \Phi(x, y, \dot{x}, \dot{y})\,dt.$$

The variational problem for the right-hand side leads to the pair of Euler equations

$$\Phi_x - \frac{d}{dt}\Phi_{\dot{x}} = 0, \qquad \Phi_y - \frac{d}{dt}\Phi_{\dot{y}} = 0,$$

which must be equivalent to the single Euler equation $F\_y - \frac{d}{dx}F\_{y'} = 0$ corresponding to the variational problem for the original functional. Hence, the equations cannot be independent, and in fact it is easily verified that they are connected by the identity

$$\dot{x}\!\left(\Phi_x - \frac{d}{dt}\Phi_{\dot{x}}\right) + \dot{y}\!\left(\Phi_y - \frac{d}{dt}\Phi_{\dot{y}}\right) = 0.$$

### 11. Functionals Depending on Higher-Order Derivatives

So far, we have considered functionals of the form $\int\_a^b F(x, y, y')\,dx$, depending on the function $y(x)$ and its first derivative $y'(x)$. However, many problems (e.g., in the theory of elasticity) involve functionals whose integrands contain not only $y\_i(x)$ and $y\_i'(x)$, but also higher-order derivatives $y\_i''(x), y\_i'''(x), \ldots$ The method given above for finding extrema of functionals can be carried over to this more general case without essential changes. For simplicity, we confine ourselves to the case of a single unknown function $y(x)$.

Thus, let $F(x, y, z\_1, \ldots, z\_n)$ be a function with continuous first and second (partial) derivatives with respect to all its arguments, and consider a functional of the form

$$J[y] = \int_a^b F(x, y, y', \ldots, y^{(n)})\,dx.$$

Then we pose the following problem: *Among all functions $y(x)$ belonging to the space $\mathscr{D}\_n(a, b)$* and satisfying the conditions

$$y(a) = A_0, \quad y'(a) = A_1, \quad \ldots, \quad y^{(n-1)}(a) = A_{n-1},$$

$$y(b) = B_0, \quad y'(b) = B_1, \quad \ldots, \quad y^{(n-1)}(b) = B_{n-1},$$

*find the function for which $J[y]$ has an extremum.* We replace $y(x)$ by the "varied" function $y(x) + h(x)$, where $h(x)$ belongs to $\mathscr{D}\_n(a, b)$. By the *variation* $\delta J$ of the functional $J[y]$, we mean the expression which is linear in $h, h', \ldots, h^{(n)}$, and which differs from the increment by a quantity of order higher than 1. Then

$$\delta J = \int_a^b (F_y h + F_{y'} h' + \cdots + F_{y^{(n)}} h^{(n)})\,dx.$$

The necessary condition $\delta J = 0$ for an extremum implies that

$$\int_a^b (F_y h + F_{y'} h' + \cdots + F_{y^{(n)}} h^{(n)})\,dx = 0.$$

Repeatedly integrating by parts and using the boundary conditions $h(a) = h'(a) = \cdots = h^{(n-1)}(a) = 0$, $h(b) = h'(b) = \cdots = h^{(n-1)}(b) = 0$, we find that

$$\int_a^b \left[F_y - \frac{d}{dx}F_{y'} + \frac{d^2}{dx^2}F_{y''} - \cdots + (-1)^n \frac{d^n}{dx^n}F_{y^{(n)}}\right] h(x)\,dx = 0$$

for any function $h$ which has $n$ continuous derivatives and satisfies the boundary conditions. It follows from a generalization of Lemma 1 of Sec. 3.1 that

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Euler–Poisson Equation)</span></p>

A necessary condition for the functional

$$J[y] = \int_a^b F(x, y, y', \ldots, y^{(n)})\,dx$$

to have an extremum is that its extremals satisfy the **Euler equation**

$$F_y - \frac{d}{dx}F_{y'} + \frac{d^2}{dx^2}F_{y''} - \cdots + (-1)^n \frac{d^n}{dx^n}F_{y^{(n)}} = 0.$$

Since this is a differential equation of order $2n$, its general solution contains $2n$ arbitrary constants, which can be determined from the boundary conditions.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The derivation of the Euler equation above is not completely rigorous, since the transition from the integral condition to the pointwise equation presupposes the existence of the derivatives $\frac{d}{dx}F\_{y'}, \frac{d^2}{dx^2}F\_{y''}, \ldots, \frac{d^n}{dx^n}F\_{y^{(n)}}$. However, by a somewhat more elaborate argument, it can be shown that the result holds without this additional hypothesis.

</div>

### 12. Variational Problems with Subsidiary Conditions

#### 12.1. The Isoperimetric Problem

In the simplest variational problem considered in Chapter 1, the class of admissible curves was specified (apart from certain smoothness requirements) by conditions imposed on the end points of the curves. However, many applications of the calculus of variations lead to problems in which not only boundary conditions, but also conditions of quite a different type known as *subsidiary conditions* (synonymously, *side conditions* or *constraints*) are imposed on the admissible curves. As an example, we first consider the *isoperimetric problem*, which can be stated as follows: *Find the curve $y = y(x)$ for which the functional*

$$J[y] = \int_a^b F(x, y, y')\,dx$$

*has an extremum, where the admissible curves satisfy the boundary conditions*

$$y(a) = A, \qquad y(b) = B,$$

*and are such that another functional*

$$K[y] = \int_a^b G(x, y, y')\,dx$$

*takes a fixed value $l$.*

To solve this problem, we assume that the functions $F$ and $G$ defining the functionals have continuous first and second derivatives in $[a, b]$ for arbitrary values of $y$ and $y'$. Then we have

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Isoperimetric Problem)</span></p>

Given the functional

$$J[y] = \int_a^b F(x, y, y')\,dx,$$

let the admissible curves satisfy the conditions

$$y(a) = A, \qquad y(b) = B, \qquad K[y] = \int_a^b G(x, y, y')\,dx = l,$$

where $K[y]$ is another functional, and let $J[y]$ have an extremum for $y = y(x)$. Then, if $y = y(x)$ is not an extremal of $K[y]$, there exists a constant $\lambda$ such that $y = y(x)$ is an extremal of the functional

$$\int_a^b (F + \lambda G)\,dx,$$

i.e., $y = y(x)$ satisfies the differential equation

$$F_y - \frac{d}{dx}F_{y'} + \lambda\!\left(G_y - \frac{d}{dx}G_{y'}\right) = 0.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $J[y]$ have an extremum for the curve $y = y(x)$, subject to the conditions. We choose two points $x\_1$ and $x\_2$ in the interval $[a, b]$, where $x\_1$ is arbitrary and $x\_2$ satisfies a condition to be stated below. Then we give $y(x)$ an increment $\delta\_1 y(x) + \delta\_2 y(x)$, where $\delta\_1 y(x)$ is nonzero only in a neighborhood of $x\_1$, and $\delta\_2 y(x)$ is nonzero only in a neighborhood of $x\_2$. Using variational derivatives, we can write the corresponding increment $\Delta J$ of the functional $J$ in the form

$$\Delta J = \left\lbrace\frac{\delta F}{\delta y}\bigg\vert_{x=x_1} + \varepsilon_1\right\rbrace\!\Delta\sigma_1 + \left\lbrace\frac{\delta F}{\delta y}\bigg\vert_{x=x_2} + \varepsilon_2\right\rbrace\!\Delta\sigma_2,$$

where $\varepsilon\_1, \varepsilon\_2 \to 0$ as $\Delta\sigma\_1, \Delta\sigma\_2 \to 0$.

We now require that the "varied" curve $y^\ast(x) = y(x) + \delta\_1 y(x) + \delta\_2 y(x)$ satisfy the condition $K[y^\ast] = K[y]$. Writing $\Delta K$ in a form similar to the above, we obtain

$$\Delta K = \left\lbrace\frac{\delta G}{\delta y}\bigg\vert_{x=x_1} + \varepsilon_1'\right\rbrace\!\Delta\sigma_1 + \left\lbrace\frac{\delta G}{\delta y}\bigg\vert_{x=x_2} + \varepsilon_2'\right\rbrace\!\Delta\sigma_2 = 0.$$

Next, we choose $x\_2$ to be a point for which $\frac{\delta G}{\delta y}\big\vert\_{x=x\_2} \neq 0$. Such a point exists, since by hypothesis $y = y(x)$ is not an extremal of the functional $K$. With this choice of $x\_2$, we can write the condition in the form

$$\Delta\sigma_2 = -\left\lbrace\frac{\frac{\delta G}{\delta y}\big\vert_{x=x_1}}{\frac{\delta G}{\delta y}\big\vert_{x=x_2}} + \varepsilon'\right\rbrace\!\Delta\sigma_1,$$

where $\varepsilon' \to 0$ as $\Delta\sigma\_1 \to 0$. Setting $\lambda = -\frac{\frac{\delta F}{\delta y}\big\vert\_{x=x\_2}}{\frac{\delta G}{\delta y}\big\vert\_{x=x\_2}}$ and substituting into the formula for $\Delta J$, we obtain

$$\Delta J = \left\lbrace\frac{\delta F}{\delta y}\bigg\vert_{x=x_1} + \lambda\frac{\delta G}{\delta y}\bigg\vert_{x=x_1}\right\rbrace\!\Delta\sigma_1 + \varepsilon\,\Delta\sigma_1,$$

where $\varepsilon \to 0$ as $\Delta\sigma\_1 \to 0$. Since a necessary condition for an extremum is that $\delta J = 0$, and since $\Delta\sigma\_1$ is nonzero while $x\_1$ is arbitrary, we finally have

$$\frac{\delta F}{\delta y} + \lambda\frac{\delta G}{\delta y} = F_y - \frac{d}{dx}F_{y'} + \lambda\!\left(G_y - \frac{d}{dx}G_{y'}\right) = 0,$$

which is precisely the stated equation. This completes the proof.

</details>
</div>

To use Theorem 1 to solve a given isoperimetric problem, we first write the general solution of the Euler equation for $\int\_a^b (F + \lambda G)\,dx$, which will contain two arbitrary constants in addition to the parameter $\lambda$. We then determine these three quantities from the boundary conditions $y(a) = A$, $y(b) = B$ and the subsidiary condition $K[y] = l$.

Everything just said generalizes immediately to the case of functionals depending on several functions $y\_1, \ldots, y\_n$ and subject to several subsidiary conditions of the form

$$\int_a^b G_j(x, y_1, \ldots, y_n, y_1', \ldots, y_n')\,dx = l_j \qquad (j = 1, \ldots, k).$$

In this case a necessary condition for an extremum is that

$$\frac{\partial}{\partial y_i}\!\left(F + \sum_{j=1}^{k} \lambda_j G_j\right) - \frac{d}{dx}\!\left\lbrace\frac{\partial}{\partial y_i'}\!\left(F + \sum_{j=1}^{k} \lambda_j G_j\right)\right\rbrace = 0 \qquad (i = 1, \ldots, n).$$

The $2n$ arbitrary constants appearing in the solution of the system, and the values of the $k$ parameters $\lambda\_1, \ldots, \lambda\_k$, sometimes called *Lagrange multipliers*, are determined from the boundary conditions and the subsidiary conditions.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1</span><span class="math-callout__name">(Largest Area under a Curve of Given Length)</span></p>

Among all curves of length $l$ in the upper half-plane passing through the points $(-a, 0)$ and $(a, 0)$, find the one which together with the interval $[-a, a]$ encloses the largest area. We are looking for the function $y = y(x)$ for which the integral

$$J[y] = \int_{-a}^{a} y\,dx$$

takes the largest value subject to the conditions

$$y(-a) = y(a) = 0, \qquad K[y] = \int_{-a}^{a} \sqrt{1 + y'^2}\,dx = l.$$

Thus, we are dealing with an isoperimetric problem. Using Theorem 1, we form the functional $J[y] + \lambda K[y] = \int\_{-a}^{a} (y + \lambda\sqrt{1 + y'^2})\,dx$, and write the corresponding Euler equation

$$1 + \lambda\frac{d}{dx}\frac{y'}{\sqrt{1 + y'^2}} = 0,$$

which implies $x + \lambda\frac{y'}{\sqrt{1 + y'^2}} = C\_1$. Integrating, we obtain the equation

$$(x - C_1)^2 + (y - C_2)^2 = \lambda^2$$

of a family of circles. The values of $C\_1$, $C\_2$ and $\lambda$ are then determined from the conditions $y(-a) = y(a) = 0$ and $K[y] = l$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2</span><span class="math-callout__name">(Shortest Curve on a Sphere)</span></p>

Among all curves lying on the sphere $x^2 + y^2 + z^2 = a^2$ and passing through two given points $(x\_0, y\_0, z\_0)$ and $(x\_1, y\_1, z\_1)$, find the one which has the least length. The length of the curve $y = y(x)$, $z = z(x)$ is given by the integral

$$\int_{x_0}^{x_1} \sqrt{1 + y'^2 + z'^2}\,dx.$$

Using Theorem 2 (below), we form the auxiliary functional

$$\int_{x_0}^{x_1} [\sqrt{1 + y'^2 + z'^2} + \lambda(x)(x^2 + y^2 + z^2)]\,dx,$$

and write the corresponding Euler equations

$$2y\lambda(x) - \frac{d}{dx}\frac{y'}{\sqrt{1 + y'^2 + z'^2}} = 0,$$

$$2z\lambda(x) - \frac{d}{dx}\frac{z'}{\sqrt{1 + y'^2 + z'^2}} = 0.$$

Solving these equations, we obtain a family of curves depending on four constants, whose values are determined from the boundary conditions.

</div>

#### 12.2. Finite Subsidiary Conditions

In the isoperimetric problem, the subsidiary conditions which must be satisfied by the functions $y\_1, \ldots, y\_n$ are of the form of integral (functional) conditions. We now consider a problem of a different type, which can be stated as follows: *Find the functions $y\_i(x)$ for which the functional*

$$J[y_1, \ldots, y_n] = \int_a^b F(x, y_1, \ldots, y_n, y_1', \ldots, y_n')\,dx$$

*has an extremum, where the admissible functions satisfy the boundary conditions*

$$y_i(a) = A_i, \quad y_i(b) = B_i \qquad (i = 1, \ldots, n)$$

*and $k$ "finite" subsidiary conditions ($k < n$)*

$$g_j(x, y_1, \ldots, y_n) = 0 \qquad (j = 1, \ldots, k).$$

In other words, the functional is not considered for all curves satisfying the boundary conditions, but only for those which lie in the $(n - k)$-dimensional manifold defined by the system of constraints. For simplicity, we confine ourselves to the case $n = 2$, $k = 1$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2</span><span class="math-callout__name">(Finite Subsidiary Conditions)</span></p>

Given the functional

$$J[y, z] = \int_a^b F(x, y, z, y', z')\,dx,$$

let the admissible curves lie on the surface

$$g(x, y, z) = 0$$

and satisfy the boundary conditions $y(a) = A\_1$, $y(b) = B\_1$, $z(a) = A\_2$, $z(b) = B\_2$, and moreover, let $J[y]$ have an extremum for the curve $y = y(x)$, $z = z(x)$.

Then, if $g\_y$ and $g\_z$ do not vanish simultaneously at any point of the surface, there exists a function $\lambda(x)$ such that the curve is an extremal of the functional

$$\int_a^b [F + \lambda(x) g]\,dx,$$

i.e., satisfies the differential equations

$$F_y + \lambda g_y - \frac{d}{dx}F_{y'} = 0, \qquad F_z + \lambda g_z - \frac{d}{dx}F_{z'} = 0.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $J[y, z]$ have an extremum for the curve $y = y(x)$, $z = z(x)$, subject to the conditions, and let $x\_1$ be an arbitrary point of the interval $[a, b]$. Then we give $y(x)$ an increment $\delta y(x)$ and $z(x)$ an increment $\delta z(x)$, where both $\delta y(x)$ and $\delta z(x)$ are nonzero only in a neighborhood $[\alpha, \beta]$ of $x\_1$. Using variational derivatives, we can write the corresponding increment $\Delta J$ of the functional $J[y, z]$ in the form

$$\Delta J = \left\lbrace\frac{\delta F}{\delta y}\bigg\vert_{x=x_1} + \varepsilon_1\right\rbrace\!\Delta\sigma_1 + \left\lbrace\frac{\delta F}{\delta z}\bigg\vert_{x=x_1} + \varepsilon_2\right\rbrace\!\Delta\sigma_2,$$

where $\Delta\sigma\_1 = \int\_a^b \delta y(x)\,dx$, $\Delta\sigma\_2 = \int\_a^b \delta z(x)\,dx$, and $\varepsilon\_1, \varepsilon\_2 \to 0$ as $\Delta\sigma\_1, \Delta\sigma\_2 \to 0$.

We now require that the "varied" curve $y^\ast(x) = y(x) + \delta y(x)$, $z^\ast(x) = z(x) + \delta z(x)$ satisfy the condition $g(x, y^\ast, z^\ast) = 0$. In view of this, we have

$$0 = \int_a^b [g(x, y^*, z^*) - g(x, y, z)]\,dx = \lbrace\bar{g}_y\vert_{x=x_1} + \varepsilon_1'\rbrace\Delta\sigma_1 + \lbrace\bar{g}_z\vert_{x=x_1} + \varepsilon_2'\rbrace\Delta\sigma_2,$$

where the overbar indicates that the corresponding derivatives are evaluated along certain intermediate curves. By hypothesis, either $g\_y\vert\_{x=x\_1}$ or $g\_z\vert\_{x=x\_1}$ is nonzero. If $g\_z\vert\_{x=x\_1} \neq 0$, we can write the condition in the form

$$\Delta\sigma_2 = -\left\lbrace\frac{\bar{g}_y\vert_{x=x_1}}{\bar{g}_z\vert_{x=x_1}} + \varepsilon'\right\rbrace\!\Delta\sigma_1,$$

where $\varepsilon' \to 0$ as $\Delta\sigma\_1 \to 0$. Substituting into the formula for $\Delta J$, we obtain

$$\Delta J = \left\lbrace\frac{\delta F}{\delta y}\bigg\vert_{x=x_1} - \frac{g_y}{g_z}\frac{\delta F}{\delta z}\bigg\vert_{x=x_1}\right\rbrace\!\Delta\sigma_1 + \varepsilon\,\Delta\sigma_1,$$

where $\varepsilon \to 0$ as $\Delta\sigma\_1 \to 0$. Since a necessary condition for an extremum is that $\delta J = 0$, and since $\Delta\sigma\_1$ is nonzero while $x\_1$ is arbitrary, we finally have

$$\frac{\delta F}{\delta y} - \frac{g_y}{g_z}\frac{\delta F}{\delta z} = 0$$

or

$$\frac{F_y - \frac{d}{dx}F_{y'}}{g_y} = \frac{F_z - \frac{d}{dx}F_{z'}}{g_z}.$$

Along the curve $y = y(x)$, $z = z(x)$, the common value of the ratios is some function of $x$. If we denote this function by $-\lambda(x)$, then the system reduces to precisely $F\_y + \lambda g\_y - \frac{d}{dx}F\_{y'} = 0$ and $F\_z + \lambda g\_z - \frac{d}{dx}F\_{z'} = 0$. This completes the proof.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1</span><span class="math-callout__name">(Nonholonomic Constraints)</span></p>

Theorem 2 remains valid when the class of admissible curves consists of smooth space curves satisfying the differential equation $g(x, y, z, y', z') = 0$. More precisely, if the functional $J$ has an extremum for a curve $\gamma$, subject to the condition $g(x, y, z, y', z') = 0$, and if the derivatives $g\_{y'}$, $g\_{z'}$ do not vanish simultaneously along $\gamma$, then there exists a function $\lambda(x)$ such that $\gamma$ is an integral curve of the system $\Phi\_y - \frac{d}{dx}\Phi\_{y'} = 0$, $\Phi\_z - \frac{d}{dx}\Phi\_{z'} = 0$, where $\Phi = F + \lambda G$. In mechanics, conditions like $g(x, y, z, y', z') = 0$, which contain derivatives, are called *nonholonomic constraints*, and conditions like $g(x, y, z) = 0$ are called *holonomic constraints*.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2</span><span class="math-callout__name">(Finite Conditions as Limiting Isoperimetric Problems)</span></p>

In a certain sense, we can consider a variational problem with a finite subsidiary condition to be a limiting case of an isoperimetric problem. In fact, if we assume that the condition $g(x, y, z) = 0$ does not hold everywhere, but only at some fixed point $g(x\_1, y, z) = 0$, we obtain a condition whose left side can be regarded as a functional of $y$ and $z$, i.e., a condition of the type appearing in the isoperimetric problem. Thus, the condition $g(x, y, z) = 0$ can be regarded as an infinite set of conditions, each of which is a functional. As we have seen, in the isoperimetric problem the number of Lagrange multipliers $\lambda\_1, \ldots, \lambda\_k$ equals the number of conditions of constraint. In the same way, the function $\lambda(x)$ appearing in the problem with a finite subsidiary condition can be interpreted as a "Lagrange multiplier for each point $x$."

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Constrained vs. Unconstrained Extrema)</span></p>

As is familiar from elementary analysis, in finding an extremum of a function of $n$ variables subject to $k$ constraints ($k < n$), we can use the constraints to express $k$ variables in terms of the other $n - k$ variables. In this way, the problem is reduced to that of finding an *unconstrained* extremum of a function of $n - k$ variables. The situation is the same in the calculus of variations. For example, the problem of finding geodesics on a given surface can be regarded as a problem subject to a constraint, as in Example 2 of this section. On the other hand, if we express the coordinates $x$, $y$ and $z$ as functions of two parameters, we can reduce the problem to that of finding an unconstrained extremum, as in Example 2 of Sec. 9.

</div>

## Chapter 3: The General Variation of a Functional

### 13. Derivation of the Basic Formula

In this section, we derive the general formula for the variation of a functional of the form

$$J[y_1, \ldots, y_n] = \int_{x_0}^{x_1} F(x, y_1, \ldots, y_n, y_1', \ldots, y_n')\,dx,$$

beginning with the case where it depends on a single function $y$ and hence reduces to

$$J[y] = \int_{x_0}^{x_1} F(x, y, y')\,dx.$$

We assume that all admissible curves are smooth, but, departing from our previous hypothesis, we assume that the end points of the curves for which $J[y]$ is defined can move in an arbitrary way. By the *distance* between two curves $y = y(x)$ and $y = y^\ast(x)$ is meant the quantity

$$\rho(y, y^*) = \max \lvert y - y^*\rvert + \max \lvert y' - y^{*\prime}\rvert + \rho(P_0, P_0^*) + \rho(P_1, P_1^*),$$

where $P\_0$, $P\_0^\ast$ denote the left-hand end points of the curves $y = y(x)$, $y = y^\ast(x)$, respectively, and $P\_1$, $P\_1^\ast$ denote their right-hand end points. In general, the functions $y$ and $y^\ast$ are defined on different intervals $I$ and $I^\ast$. Thus, in order for the distance to make sense, we have to extend $y$ and $y^\ast$ onto some interval containing both $I$ and $I^\ast$. For example, this can be done by drawing tangents to the curves at their end points.

Now let $y = y(x)$ and $y = y^\ast(x)$ be two neighboring curves, in the sense of the distance, and let $h(x) = y^\ast(x) - y(x)$. Moreover, let $P\_0 = (x\_0, y\_0)$, $P\_1 = (x\_1, y\_1)$ denote the end points of the curve $y = y(x)$, while the end points of the curve $y = y^\ast(x) = y(x) + h(x)$ are denoted by

$$P_0^* = (x_0 + \delta x_0, y_0 + \delta y_0), \qquad P_1^* = (x_1 + \delta x_1, y_1 + \delta y_1).$$

The corresponding variation $\delta J$ of the functional $J[y]$ is defined as the expression which is linear in $h$, $h'$, $\delta x\_0$, $\delta y\_0$, $\delta x\_1$, $\delta y\_1$, and which differs from the increment

$$\Delta J = J[y + h] - J[y]$$

by a quantity of order higher than 1 relative to $\rho(y, y + h)$. Since

$$\Delta J = \int_{x_0 + \delta x_0}^{x_1 + \delta x_1} F(x, y + h, y' + h')\,dx - \int_{x_0}^{x_1} F(x, y, y')\,dx,$$

we can write

$$\Delta J = \int_{x_0}^{x_1} [F(x, y + h, y' + h') - F(x, y, y')]\,dx + \int_{x_1}^{x_1 + \delta x_1} F(x, y + h, y' + h')\,dx - \int_{x_0}^{x_0 + \delta x_0} F(x, y + h, y' + h')\,dx.$$

It follows by using Taylor's theorem and letting the symbol $\sim$ denote equality except for terms of order higher than 1 relative to $\rho(y, y + h)$ that

$$\Delta J \sim \int_{x_0}^{x_1} [F_y(x, y, y')h + F_{y'}(x, y, y')h']\,dx + F(x, y, y')\vert_{x=x_1}\,\delta x_1 - F(x, y, y')\vert_{x=x_0}\,\delta x_0.$$

After integrating the term containing $h'$ by parts, and noting from the geometry that $h(x\_0) \sim \delta y\_0 - y'(x\_0)\,\delta x\_0$ and $h(x\_1) \sim \delta y\_1 - y'(x\_1)\,\delta x\_1$, we obtain

$$\delta J = \int_{x_0}^{x_1} \left[F_y - \frac{d}{dx}F_{y'}\right] h(x)\,dx + F_{y'}\vert_{x=x_1}\,\delta y_1 + (F - F_{y'}y')\vert_{x=x_1}\,\delta x_1 - F_{y'}\vert_{x=x_0}\,\delta y_0 - (F - F_{y'}y')\vert_{x=x_0}\,\delta x_0,$$

or more concisely,

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(General Variation of $J[y]$)</span></p>

The general variation of the functional $J[y] = \int\_{x\_0}^{x\_1} F(x, y, y')\,dx$ is

$$\delta J = \int_{x_0}^{x_1} \left[F_y - \frac{d}{dx}F_{y'}\right] h(x)\,dx + F_{y'}\,\delta y\bigg\vert_{x=x_0}^{x=x_1} + (F - F_{y'}y')\,\delta x\bigg\vert_{x=x_0}^{x=x_1},$$

where we *define*

$$\delta x\vert_{x=x_i} = \delta x_i, \qquad \delta y\vert_{x=x_i} = \delta y_i \qquad (i = 0, 1).$$

</div>

This is the basic formula for the general variation of the functional $J[y]$. If the end points of the admissible curves are constrained to lie on the straight lines $x = x\_0$, $x = x\_1$, as in the simple variable end point problem considered in Sec. 6, then $\delta x\_0 = \delta x\_1 = 0$, while in the case of the fixed end point problem, $\delta x\_0 = \delta x\_1 = 0$ and $\delta y\_0 = \delta y\_1 = 0$.

#### The General Case of $n$ Functions

Next, we return to the more general functional depending on $n$ functions $y\_1, \ldots, y\_n$. Since any system of $n$ functions can be interpreted as a curve in $(n + 1)$-dimensional Euclidean space $\mathscr{E}\_{n+1}$, we can regard the functional as defined on some set of curves in $\mathscr{E}\_{n+1}$. Paralleling the treatment just given for $n = 1$, we now calculate the variation of the functional when there are no restrictions on the end points of the admissible curves. As before, we write

$$h_i(x) = y_i^*(x) - y_i(x) \qquad (i = 1, \ldots, n),$$

where for each $i$, the function $y\_i^\ast(x)$ is close to $y\_i(x)$ in the sense of the distance. Moreover, we let

$$P_0 = (x_0, y_1^0, \ldots, y_n^0), \qquad P_1 = (x_1, y_1^1, \ldots, y_n^1)$$

denote the end points of the curve $y\_i = y\_i(x)$, $i = 1, \ldots, n$, while the end points of the curve $y\_i = y\_i^\ast(x) = y\_i(x) + h\_i(x)$, $i = 1, \ldots, n$, are denoted by

$$P_0^* = (x_0 + \delta x_0, y_1^0 + \delta y_1^0, \ldots, y_n^0 + \delta y_n^0), \qquad P_1^* = (x_1 + \delta x_1, y_1^1 + \delta y_1^1, \ldots, y_n^1 + \delta y_n^1).$$

The corresponding variation $\delta J$ of the functional $J[y\_1, \ldots, y\_n]$ is defined as the expression which is linear in $\delta x\_0$, $\delta x\_1$ and all $h\_i$, $h\_i'$, $\delta y\_i^0$, $\delta y\_i^1$ $(i = 1, \ldots, n)$, and which differs from the increment $\Delta J = J[y\_1 + h\_1, \ldots, y\_n + h\_n] - J[y\_1, \ldots, y\_n]$ by a quantity of order higher than 1. After computations similar to the single-function case (integrating by parts the terms containing $h\_i'$ and using the approximations $h\_i(x\_0) \sim \delta y\_i^0 - y\_i'(x\_0)\,\delta x\_0$ and $h\_i(x\_1) \sim \delta y\_i^1 - y\_i'(x\_1)\,\delta x\_1$), we obtain

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(General Variation of $J[y\_1, \ldots, y\_n]$)</span></p>

The general variation of the functional $J[y\_1, \ldots, y\_n] = \int\_{x\_0}^{x\_1} F(x, y\_1, \ldots, y\_n, y\_1', \ldots, y\_n')\,dx$ is

$$\delta J = \int_{x_0}^{x_1} \sum_{i=1}^{n} \left(F_{y_i} - \frac{d}{dx}F_{y_i'}\right) h_i(x)\,dx + \sum_{i=1}^{n} F_{y_i'}\,\delta y_i\bigg\vert_{x=x_0}^{x=x_1} + \left(F - \sum_{i=1}^{n} y_i' F_{y_i'}\right)\delta x\bigg\vert_{x=x_0}^{x=x_1},$$

where we *define* $\delta x\vert\_{x=x\_j} = \delta x\_j$, $\delta y\_i\vert\_{x=x\_j} = \delta y\_i^j$ $(j = 0, 1)$.

</div>

#### Canonical Variables and the Hamiltonian

We now write an even more concise formula for the variation, at the same time introducing some important new ideas, to be discussed in more detail in the next chapter. Let

$$p_i = F_{y_i'} \qquad (i = 1, \ldots, n),$$

and suppose that the Jacobian

$$\frac{\partial(p_1, \ldots, p_n)}{\partial(y_1', \ldots, y_n')} = \det \lVert F_{y_i' y_k'}\rVert$$

is nonzero. Then we can solve the equations $p\_i = F\_{y\_i'}$ for $y\_1', \ldots, y\_n'$ as functions of the variables $x, y\_1, \ldots, y\_n, p\_1, \ldots, p\_n$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hamiltonian)</span></p>

Next, we express the function $F(x, y\_1, \ldots, y\_n, y\_1', \ldots, y\_n')$ appearing in the functional in terms of a new function $H(x, y\_1, \ldots, y\_n, p\_1, \ldots, p\_n)$ related to $F$ by the formula

$$H = -F + \sum_{i=1}^{n} y_i' F_{y_i'} \equiv -F + \sum_{i=1}^{n} y_i' p_i,$$

where the $y\_i'$ are regarded as functions of the variables $x, y\_1, \ldots, y\_n, p\_1, \ldots, p\_n$. The function $H$ is called the **Hamiltonian** (function) corresponding to the functional $J[y\_1, \ldots, y\_n]$. In this way, we can make a local transformation from the "variables" $x, y\_1, \ldots, y\_n, y\_1', \ldots, y\_n'$, $F$ appearing in the functional to the new quantities $x, y\_1, \ldots, y\_n, p\_1, \ldots, p\_n$, $H$, called the **canonical variables** (corresponding to the functional $J[y\_1, \ldots, y\_n]$).

</div>

In terms of the canonical variables, the general variation can be written in the form

$$\delta J = \int_{x_0}^{x_1} \sum_{i=1}^{n} \left(F_{y_i} - \frac{dp_i}{dx}\right) h_i(x)\,dx + \left(\sum_{i=1}^{n} p_i\,\delta y_i - H\,\delta x\right)\bigg\vert_{x=x_0}^{x=x_1},$$

or, if the integral in the variation vanishes (which happens when the curve for which $J[y\_1, \ldots, y\_n]$ has an extremum is an extremal and then satisfies the condition that the boundary terms vanish — see Problem 1, p. 63),

$$\delta J = \left(\sum_{i=1}^{n} p_i\,\delta y_i - H\,\delta x\right)\bigg\vert_{x=x_0}^{x=x_1}.$$

Thus, regardless of the boundary conditions defining our variable end point problem, the curve for which $J[y\_1, \ldots, y\_n]$ has an extremum must first be an extremal and then satisfy the condition that the boundary terms in the variation vanish.

### 14. End Points Lying on Two Given Curves or Surfaces

The first two chapters of this book were devoted mainly to fixed end point problems, where the boundary conditions require that all admissible curves have two given end points. The only exception is the simple variable end point problem considered in Sec. 6, where the end points of the admissible curves are free to move along two fixed straight lines parallel to the $y$-axis. We now consider a more general variable end point problem. To keep matters simple, we start with the case where there is only one unknown function. Our problem can be stated as follows: *Among all smooth curves whose end points $P\_0$ and $P\_1$ lie on two given curves $y = \varphi(x)$ and $y = \psi(x)$, find the curve for which the functional*

$$J[y] = \int_{x_0}^{x_1} F(x, y, y')\,dx$$

*has an extremum.* For example, the problem of finding the distance between two plane curves is of this type, with $F(x, y, y') = \sqrt{1 + y'^2}$.

As shown in Sec. 13, the general variation of the functional $J[y]$ is given by formula (5). If $J[y]$ has an extremum for the curve $y = y(x)$, then, as noted at the end of Sec. 13, this curve must first of all be an extremal, i.e., a solution of Euler's equation. Hence the integral in the variation vanishes and we have

$$\delta J = F_{y'}\vert_{x=x_1}\,\delta y_1 + (F - y'F_{y'})\vert_{x=x_1}\,\delta x_1 - F_{y'}\vert_{x=x_0}\,\delta y_0 - (F - y'F_{y'})\vert_{x=x_0}\,\delta x_0,$$

which must vanish if $J[y]$ is to have an extremum for $y = y(x)$.

Next, we observe that since the end points lie on the given curves, the increments $\delta y\_0$, $\delta y\_1$ are related to $\delta x\_0$, $\delta x\_1$ by

$$\delta y_0 = [\varphi'(x) + \varepsilon_0]\,\delta x_0, \qquad \delta y_1 = [\psi'(x_1) + \varepsilon_1]\,\delta x_1,$$

where $\varepsilon\_0 \to 0$ as $\delta x\_0 \to 0$, and $\varepsilon\_1 \to 0$ as $\delta x\_1 \to 0$. Thus, in the present case, the condition $\delta J = 0$ becomes

$$\delta J = (F_{y'}\psi' + F - y'F_{y'})\vert_{x=x_1}\,\delta x_1 - (F_{y'}\varphi' + F - y'F_{y'})\vert_{x=x_0}\,\delta x_0 = 0.$$

Since $\delta x\_0$ and $\delta x\_1$ are independent, this implies the boundary conditions

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Transversality Conditions)</span></p>

For the functional $J[y] = \int\_{x\_0}^{x\_1} F(x, y, y')\,dx$, suppose the end points of the admissible curves lie on two given curves $y = \varphi(x)$ and $y = \psi(x)$. Then a necessary condition for an extremum is that the extremal $y = y(x)$ satisfies Euler's equation

$$F_y - \frac{d}{dx}F_{y'} = 0,$$

together with the **transversality conditions**

$$[F + (\varphi' - y')F_{y'}]\vert_{x=x_0} = 0,$$

$$[F + (\psi' - y')F_{y'}]\vert_{x=x_1} = 0.$$

The curve $y = y(x)$ satisfying these conditions is said to be a *transversal* of the curves $y = \varphi(x)$ and $y = \psi(x)$.

</div>

Thus, to solve this kind of variable end point problem, we must first solve Euler's equation and then use the transversality conditions to determine the values of the two arbitrary constants appearing in the general solution.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Transversality Reduces to Orthogonality)</span></p>

In solving variational problems, we often encounter functionals of the form

$$\int_{x_0}^{x_1} f(x, y)\sqrt{1 + y'^2}\,dx.$$

For such functionals, the transversality conditions have a particularly simple appearance. In fact, in this case,

$$F_{y'} = f(x, y)\frac{y'}{\sqrt{1 + y'^2}} = \frac{y'F}{1 + y'^2},$$

so that the transversality conditions become

$$F + (\varphi' - y')F_{y'} = \frac{(1 + y'\varphi')F}{1 + y'^2} = 0, \qquad F + (\psi' - y')F_{y'} = \frac{(1 + y'\psi')F}{1 + y'^2} = 0.$$

It follows that $y' = -1/\varphi'$ at the left-hand end point, and $y' = -1/\psi'$ at the right-hand end point, i.e., for functionals of this form, transversality reduces to **orthogonality**.

</div>

#### Variable End Points on Surfaces

The same kind of variable end point problem can be posed for functionals depending on several functions. For example, consider the following problem: *Among all smooth curves whose end points lie on two given surfaces $x = \varphi(y, z)$ and $x = \psi(y, z)$, find the curve for which the functional*

$$J[y, z] = \int_{x_0}^{x_1} F(x, y, z, y', z')\,dx$$

*has an extremum.* Setting $n = 2$ in the general variation formula, we obtain by the same argument as in the case of one independent function, we find that the required curve $y = y(x)$, $z = z(x)$ must again be an extremal, i.e., satisfy the Euler equations

$$F_y - \frac{d}{dx}F_{y'} = 0, \qquad F_z - \frac{d}{dx}F_{z'} = 0.$$

The boundary conditions are now

$$\begin{aligned}
&[F_{y'} + \tfrac{\partial\varphi}{\partial y}(F - y'F_{y'} - z'F_{z'})]\vert_{x=x_0} = 0, \\\\
&[F_{z'} + \tfrac{\partial\varphi}{\partial z}(F - y'F_{y'} - z'F_{z'})]\vert_{x=x_0} = 0, \\\\
&[F_{y'} + \tfrac{\partial\psi}{\partial y}(F - y'F_{y'} - z'F_{z'})]\vert_{x=x_1} = 0, \\\\
&[F_{z'} + \tfrac{\partial\psi}{\partial z}(F - y'F_{y'} - z'F_{z'})]\vert_{x=x_1} = 0,
\end{aligned}$$

and are again called the *transversality conditions*.

### 15. Broken Extremals. The Weierstrass-Erdmann Conditions

So far, we have only considered functions defined for *smooth* curves, and hence we have only permitted smooth solutions of variational problems. However, it is easy to give examples of variational problems which have no solutions in the class of smooth curves, but which have solutions if we extend the class of admissible curves to include *piecewise smooth* curves. Thus, consider the functional

$$J[y] = \int_{-1}^{1} y^2(1 - y')^2\,dx, \qquad y(-1) = 0, \quad y(1) = 1.$$

The greatest lower bound of the values of $J[y]$ for smooth $y = y(x)$ satisfying the boundary conditions is obviously zero, but it does not achieve this value for any smooth curve. In fact, the minimum is achieved for the curve

$$y = y(x) = \begin{cases} 0 & \text{for } -1 \leqslant x \leqslant 0, \\\\ x & \text{for } 0 < x \leqslant 1, \end{cases}$$

which has a *corner* (i.e., a discontinuous first derivative) at the point $x = 0$. Such a piecewise smooth extremal with corners is called a **broken extremal**.

Guided by the above considerations, we enlarge the class of admissible functions, relaxing the requirement that they be smooth everywhere. Thus, we pose the following problem: *Among all functions $y(x)$ which are continuously differentiable for $a \leqslant x \leqslant b$ except possibly at some point $c$ $(a < c < b)$, and which satisfy the boundary conditions*

$$y(a) = A, \qquad y(b) = B,$$

*find the function for which the functional*

$$J[y] = \int_a^b F(x, y, y')\,dx$$

*has a weak extremum.* It is clear that on each of the intervals $[a, c]$ and $[c, b]$ the function for which $J[y]$ has an extremum must satisfy the Euler equation

$$F_y - \frac{d}{dx}F_{y'} = 0.$$

Writing $J[y]$ as a sum of two functionals, i.e.,

$$J[y] = \int_a^c F(x, y, y')\,dx + \int_c^b F(x, y, y')\,dx \equiv J_1[y] + J_2[y],$$

we calculate the variations $\delta J\_1$ and $\delta J\_2$ of the two terms separately. The end points $x = a$, $x = b$ are fixed, and we require that the two "pieces" of the function $y(x)$ join continuously at $x = c$, but otherwise the point $x = c$ can move freely. Using formula (5) of Sec. 13 to write $\delta J\_1$ and $\delta J\_2$, and recalling that $y(x)$ is an extremal, we find that

$$\delta J_1 = F_{y'}\vert_{x=c-0}\,\delta y_1 + (F - y'F_{y'})\vert_{x=c-0}\,\delta x_1,$$

$$\delta J_2 = -F_{y'}\vert_{x=c+0}\,\delta y_1 - (F - y'F_{y'})\vert_{x=c+0}\,\delta x_1.$$

[The condition that $y(x)$ be continuous at $x = c$ implies that $\delta J\_1$ and $\delta J\_2$ involve the same increments $\delta x\_1$ and $\delta y\_1$.] At an extremum we must have $\delta J = \delta J\_1 + \delta J\_2 = 0$, and hence

$$(F_{y'}\vert_{x=c-0} - F_{y'}\vert_{x=c+0})\,\delta y_1 + [(F - y'F_{y'})\vert_{x=c-0} - (F - y'F_{y'})\vert_{x=c+0}]\,\delta x_1 = 0.$$

Since $\delta x\_1$ and $\delta y\_1$ are arbitrary, the conditions

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Weierstrass-Erdmann Corner Conditions)</span></p>

At a point $c$ where the extremal has a corner, the following conditions must hold:

$$F_{y'}\vert_{x=c-0} = F_{y'}\vert_{x=c+0},$$

$$(F - y'F_{y'})\vert_{x=c-0} = (F - y'F_{y'})\vert_{x=c+0}.$$

These are called the **Weierstrass-Erdmann (corner) conditions**.

</div>

In each of the intervals $[a, c]$ and $[c, b]$, the extremal $y = y(x)$ must satisfy the Euler equation, a second-order differential equation. Solving these two equations, we obtain four arbitrary constants, which can then be found from the boundary conditions $y(a) = A$, $y(b) = B$ and the Weierstrass-Erdmann conditions.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Canonical Variables Interpretation)</span></p>

The Weierstrass-Erdmann conditions take a particularly simple form if we use the canonical variables

$$p = F_{y'}, \qquad H = -F + y'F_{y'}.$$

introduced in Sec. 13. In fact, then the conditions just mean that *the canonical variables are continuous at a point where the extremal has a corner*.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric Interpretation via the Indicatrix)</span></p>

The Weierstrass-Erdmann conditions have the following simple geometric interpretation: Let $x$ and $y$ take fixed values, plot the value of $y'$ along one coordinate axis, and plot the values of $F(x, y, y')$ along the other. The result is a curve, called the *indicatrix*, representing $F(x, y, y')$ as a function of $y'$. Then the first of the conditions means that the tangents to the indicatrix at the points $y'(c - 0)$ and $y'(c + 0)$ are parallel, while the second condition, which can be written in the form

$$F\vert_{x=c+0} - F\vert_{x=c-0} = F_{y'}y'\vert_{x=c+0} - F_{y'}y'\vert_{x=c-0},$$

means that the two tangents are not only parallel, but in fact **coincide**.

</div>

## Chapter 4: The Canonical Form of the Euler Equations and Related Topics

Many physical laws can be expressed as *variational principles*, i.e., in terms of extremal properties of certain functionals. In this chapter, we illustrate this by using variational methods to study the classical mechanics of a system consisting of a finite number of particles. We show how the trajectories in phase space can be found as the extremals of a certain functional, and how conserved quantities arise from symmetries. We return to the subject of canonical variables (introduced in Sec. 13), discuss the reduction of the Euler equations to canonical form, and present Noether's theorem connecting invariance of a functional to first integrals.

### 16. The Canonical Form of the Euler Equations

The Euler equations corresponding to the functional

$$J[y_1, \ldots, y_n] = \int_a^b F(x, y_1, \ldots, y_n, y_1', \ldots, y_n')\,dx$$

(which depends on $n$ functions) form a system of $n$ second-order differential equations

$$F_{y_i} - \frac{d}{dx} F_{y_i'} = 0 \qquad (i = 1, \ldots, n).$$

This system can be reduced (in various ways) to a system of $2n$ first-order differential equations. For example, regarding $y\_1', \ldots, y\_n'$ as $n$ new functions, independent of $y\_1, \ldots, y\_n$, we can write the system in the form

$$\frac{dy_i}{dx} = y_i', \qquad F_{y_i} - \frac{d}{dx} F_{y_i'} = 0 \qquad (i = 1, \ldots, n),$$

where $y\_1, \ldots, y\_n, y\_1', \ldots, y\_n'$ are $2n$ unknown functions and $x$ is the independent variable. However, we obtain a much more convenient and symmetric form of the Euler equations if we replace $x, y\_1, \ldots, y\_n, y\_1', \ldots, y\_n'$ by the *canonical variables* introduced in Sec. 13.

Recall from Sec. 13 the equations

$$p_i = F_{y_i'} \qquad (i = 1, \ldots, n)$$

to write $y\_1', \ldots, y\_n'$ as functions of the variables $x, y\_1, \ldots, y\_n, p\_1, \ldots, p\_n$. We then express the function $F$ appearing in the functional in terms of a new function $H(x, y\_1, \ldots, y\_n, p\_1, \ldots, p\_n)$ related to $F$ by the formula

$$H = -F + \sum_{i=1}^{n} y_i' p_i,$$

where the $y\_i'$ are regarded as functions of the variables $x, y\_1, \ldots, y\_n, p\_1, \ldots, p\_n$. The function $H$ is called the **Hamiltonian** corresponding to the functional $J[y\_1, \ldots, y\_n]$. Finally, we introduce the new variables $x, y\_1, \ldots, y\_n, p\_1, \ldots, p\_n, H$.

We now show how the Euler equations transform when we go over to canonical variables. By the definition of $H$, we have

$$dH = -dF + \sum_{i=1}^{n} p_i\,dy_i' + \sum_{i=1}^{n} y_i'\,dp_i.$$

Expanding $dF$ and using the fact that $\frac{\partial F}{\partial y\_i'} = p\_i$, the terms containing $dy\_i'$ cancel, and we obtain

$$dH = -\frac{\partial F}{\partial x}\,dx - \sum_{i=1}^{n} \frac{\partial F}{\partial y_i}\,dy_i + \sum_{i=1}^{n} y_i'\,dp_i.$$

It follows that the partial derivatives of $H$ are connected with the partial derivatives of $F$ by the formulas

$$y_i' = \frac{\partial H}{\partial p_i}, \qquad \frac{\partial F}{\partial y_i} = -\frac{\partial H}{\partial y_i}.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Canonical Euler Equations)</span></p>

Using these relations, the Euler equations can be written in the form

$$\frac{dy_i}{dx} = \frac{\partial H}{\partial p_i}, \qquad \frac{dp_i}{dx} = -\frac{\partial H}{\partial y_i} \qquad (i = 1, \ldots, n).$$

These $2n$ first-order differential equations form a system which is equivalent to the original system of $n$ second-order Euler equations. It is called the **canonical system of Euler equations** (or simply the **canonical Euler equations**) for the functional $J[y\_1, \ldots, y\_n]$.

</div>

### 17. First Integrals of the Euler Equations

Recall that a *first integral* of a system of differential equations is a function which has a constant value along each integral curve of the system. We now look for first integrals of the canonical system, and hence of the original Euler equations.

First, we consider the case where the function $F$ defining the functional does not depend on $x$ explicitly, i.e., is of the form $F(y\_1, \ldots, y\_n, y\_1', \ldots, y\_n')$. Then the function

$$H = -F + \sum_{i=1}^{n} y_i' p_i$$

also does not depend on $x$ explicitly, and hence

$$\frac{dH}{dx} = \sum_{i=1}^{n} \left(\frac{\partial H}{\partial y_i}\frac{dy_i}{dx} + \frac{\partial H}{\partial p_i}\frac{dp_i}{dx}\right).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hamiltonian as First Integral)</span></p>

Using the Euler equations in canonical form, we find that

$$\frac{dH}{dx} = \sum_{i=1}^{n} \left(\frac{\partial H}{\partial y_i}\frac{\partial H}{\partial p_i} - \frac{\partial H}{\partial p_i}\frac{\partial H}{\partial y_i}\right) = 0,$$

along each extremal. Thus, *if $F$ does not depend on $x$ explicitly, the function $H(y\_1, \ldots, y\_n, p\_1, \ldots, p\_n)$ is a first integral of the Euler equations*.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

If $H$ depends on $x$ explicitly, the formula becomes $\frac{dH}{dx} = \frac{\partial H}{\partial x}$.

</div>

Next, we consider an arbitrary function of the form

$$\Phi = \Phi(y_1, \ldots, y_n, p_1, \ldots, p_n),$$

and examine the conditions under which $\Phi$ will be a first integral of the canonical system. Along each integral curve, we have

$$\frac{d\Phi}{dx} = \sum_{i=1}^{n} \left(\frac{\partial \Phi}{\partial y_i}\frac{dy_i}{dx} + \frac{\partial \Phi}{\partial p_i}\frac{dp_i}{dx}\right) = \sum_{i=1}^{n} \left(\frac{\partial \Phi}{\partial y_i}\frac{\partial H}{\partial p_i} - \frac{\partial \Phi}{\partial p_i}\frac{\partial H}{\partial y_i}\right) = [\Phi, H],$$

where the expression

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Poisson Bracket)</span></p>

$$[\Phi, H] = \sum_{i=1}^{n} \left(\frac{\partial \Phi}{\partial y_i}\frac{\partial H}{\partial p_i} - \frac{\partial \Phi}{\partial p_i}\frac{\partial H}{\partial y_i}\right)$$

is called the **Poisson bracket** of the functions $\Phi$ and $H$. Thus we have proved the formula

$$\frac{d\Phi}{dx} = [\Phi, H].$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(First Integrals via Poisson Brackets)</span></p>

It follows that *a necessary and sufficient condition for a function $\Phi = \Phi(y\_1, \ldots, y\_n, p\_1, \ldots, p\_n)$ to be a first integral of the system of canonical Euler equations is that the Poisson bracket $[\Phi, H]$ vanish identically*.

</div>

### 18. The Legendre Transformation

We now consider another method of reducing the Euler equations to canonical form, a method which differs from that presented in Sec. 16. The idea of this new method is to replace the variational problem under consideration by another, equivalent problem, such that the Euler equations for the new problem are the same as the *canonical* Euler equations for the original problem.

#### 18.1. Extrema and the Legendre Transform for Functions

We begin by discussing some related topics from the theory of extrema of functions of $n$ variables. First, we consider the case $n = 1$.

Suppose we are looking for an extremum, say a minimum, of the function $f(\xi)$, and suppose $f(\xi)$ is *(strictly) convex*, which means that

$$f''(\xi) > 0$$

wherever $f(\xi)$ is defined. We introduce a new independent variable

$$p = f'(\xi),$$

called the **tangential coordinate**, which is just the slope of the tangent passing through a given point of the curve $\eta = f(\xi)$. Since by hypothesis $\frac{dp}{d\xi} = f''(\xi) > 0$, the function $f(\xi)$ being convex, we can use this to express $\xi$ in terms of $p$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Legendre Transformation)</span></p>

We now introduce the new function

$$H(p) = -f(\xi) + p\xi,$$

where $\xi$ is regarded as the function of $p$ obtained by solving $p = f'(\xi)$. The transformation from the variable and function pair $\xi$, $f(\xi)$ to the variable and function pair $p$, $H(p)$, defined by the formulas $p = f'(\xi)$ and $H(p) = -f(\xi) + p\xi$, is called the **Legendre transformation**. Since $f(\xi)$ is convex, so is $H(p)$. The convex functions $H(p)$ and $f(\xi)$ are sometimes said to be *conjugate*.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Key Properties of the Legendre Transformation)</span></p>

1. From the definition, $dH = -f'(\xi)\,d\xi + p\,d\xi + \xi\,dp$, and since $p = f'(\xi)$, this simplifies to $\frac{dH}{dp} = \xi$, and moreover $\frac{d^2H}{dp^2} = \frac{d\xi}{dp} = \frac{1}{f''(\xi)} > 0$.

2. The Legendre transformation is an **involution**, i.e., a transformation which is its own inverse. If the Legendre transformation is applied to the pair $p$, $H(p)$, we get back the pair $\xi$, $f(\xi)$. This follows from the relation $-H(p) + pH'(p) = f(\xi) - p\xi + p\xi = f(\xi)$.

3. The same considerations apply to a (strictly) *concave* function, i.e., a function such that $f''(\xi) < 0$ everywhere.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Legendre Transform of a Power Function)</span></p>

If $f(\xi) = \frac{\xi^a}{a}$ with $a > 1$, then $f'(\xi) = p = \xi^{a-1}$, i.e., $\xi = p^{1/(a-1)}$. It follows that

$$H = -\frac{\xi^a}{a} + p\xi = -\frac{p^{a/(a-1)}}{a} + p \cdot p^{1/(a-1)} = p^{a/(a-1)}\left(-\frac{1}{a} + 1\right),$$

and therefore $H(p) = \frac{p^b}{b}$, where $b$ is related to $a$ by the formula $\frac{1}{a} + \frac{1}{b} = 1$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Young's Inequality)</span></p>

If $-H(p) + \xi p$ is regarded as a function of two variables, then

$$f(\xi) = \max_p\,[-H(p) + \xi p].$$

It follows that $-H(p\_1, \ldots, p\_n) + \sum\_{i=1}^{n} p\_i \xi\_i \leqslant f(\xi\_1, \ldots, \xi\_n)$ for arbitrary $p\_1, \ldots, p\_n$, i.e.,

$$\sum_{i=1}^{n} p_i \xi_i \leqslant H(p_1, \ldots, p_n) + f(\xi_1, \ldots, \xi_n),$$

a result known as **Young's inequality**. Moreover, if the matrix $\lVert f\_{\xi\_i \xi\_k}\rVert$ is *positive definite*, then

$$f(\xi_1, \ldots, \xi_n) = \max_{p_1, \ldots, p_n}\!\left[-H(p_1, \ldots, p_n) + \sum_{i=1}^{n} p_i \xi_i\right].$$

</div>

#### 18.2. Application to Functionals

We now apply the considerations of Sec. 18.1 to functionals. Given a functional

$$J[y] = \int_a^b F(x, y, y')\,dx,$$

we set

$$p = F_{y'}(x, y, y')$$

and

$$H(x, y, p) = -F + py'.$$

Here we assume that $F\_{y'y'} \neq 0$, so that the equation $p = F\_{y'}$ defines $y'$ as a function of $x$, $y$ and $p$. Then we introduce the new functional

$$J[y, p] = \int_a^b [-H(x, y, p) + py']\,dx,$$

where $y$ and $p$ are regarded as two independent functions, and $y'$ is the derivative of $y$. This functional is obviously the same as the original functional $J[y]$, if we choose $p$ to be given by the expression $F\_{y'}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Equivalence via the Legendre Transformation)</span></p>

The Euler equations for the functional $J[y, p] = \int\_a^b [-H(x, y, p) + py']\,dx$ are

$$-\frac{\partial H}{\partial y} - \frac{dp}{dx} = 0, \qquad -\frac{\partial H}{\partial p} + \frac{dy}{dx} = 0,$$

i.e., just the canonical equations for the original functional. Moreover, since $J[y, p]$ does not contain $p'$, to find an extremum of $J[y, p]$ it is sufficient to find an extremum of the integrand $-H + py'$ at every point, giving $y' = \frac{\partial H}{\partial p}$ and $-H + p\frac{\partial H}{\partial p} = F$. Thus, the variational problems for $J[y]$ and $J[y, p]$ have their extrema for the same curves, providing a new derivation of the canonical equations independent of the derivation given in Sec. 16.

</div>

The fact that the Legendre transformation is an involution means that if we subject $H(x, y, p)$ to a Legendre transformation, we get back the function $F(x, y, y')$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Sturm-Liouville Functional)</span></p>

Consider the functional

$$\int_a^b (Py'^2 + Qy^2)\,dx,$$

where $P$ and $Q$ are functions of $x$. In this case, $p = 2Py'$ and $H = Py'^2 - Qy^2$, hence

$$H = \frac{p^2}{4P} - Qy^2.$$

The corresponding canonical equations are

$$\frac{dp}{dx} = 2Qy, \qquad \frac{dy}{dx} = \frac{p}{2P},$$

while the usual form of the Euler equation for this functional is $2yQ - \frac{d}{dx}(2Py') = 0$.

</div>

### 19. Canonical Transformations

Next, we look for transformations under which the canonical Euler equations preserve their canonical form. Recall from Sec. 8 that the Euler equation $F\_y - \frac{d}{dx}F\_{y'} = 0$ is invariant under coordinate transformations of the form $u = u(x, y)$, $v = v(x, y)$ with nonvanishing Jacobian. The canonical Euler equations also have this invariance property. Furthermore, because of the symmetry between the variables $y\_i$ and $p\_i$ in the canonical equations, they permit even more general changes of variables, i.e., we can transform the variables $x, y\_i, p\_i$ into new variables $x, Y\_i, P\_i$:

$$Y_i = Y_i(x, y_1, \ldots, y_n, p_1, \ldots, p_n),$$

$$P_i = P_i(x, y_1, \ldots, y_n, p_1, \ldots, p_n).$$

However, the canonical equations do not preserve their form under all transformations of this type. We now study the conditions which have to be imposed on the transformations for the canonical equations to remain in canonical form, i.e., to transform into new equations

$$\frac{dY_i}{dx} = \frac{\partial H^*}{\partial P_i}, \qquad \frac{dP_i}{dx} = -\frac{\partial H^*}{\partial Y_i},$$

where $H^\ast = H^\ast(x, Y\_1, \ldots, Y\_n, P\_1, \ldots, P\_n)$ is some new function. Transformations which preserve the canonical form of the Euler equations are called **canonical transformations**.

The Euler equations of the functional

$$J[y_1, \ldots, y_n, p_1, \ldots, p_n] = \int_a^b \left(\sum_{i=1}^{n} p_i y_i' - H\right)dx$$

are the canonical Euler equations. We want the new variables $Y\_i$ and $P\_i$ to satisfy the equations with $H^\ast$ as its Euler equations. This functional is

$$J^*[Y_1, \ldots, Y_n, P_1, \ldots, P_n] = \int_a^b \left(\sum_{i=1}^{n} P_i Y_i' - H^*\right)dx.$$

As was shown in Sec. 9 (Remark 1), two variational problems are equivalent (i.e., have the same extremals) if the integrands of the corresponding functionals differ from each other by a total differential, which in this case means that

$$\sum_{i=1}^{n} p_i\,dy_i - H\,dx = \sum_{i=1}^{n} P_i\,dY_i - H^*\,dx + d\Phi(x, y_1, \ldots, y_n, p_1, \ldots, p_n)$$

for some function $\Phi$. Thus, if a given transformation from the variables $x, y\_i, p\_i$ to the variables $x, Y\_i, P\_i$ is such that there exists a function $\Phi$ satisfying this condition, then the transformation is canonical. The function $\Phi$ defined by this condition is called the **generating function** of the canonical transformation.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Generating Function of a Canonical Transformation)</span></p>

Writing the condition in the form

$$d\Phi = \sum_{i=1}^{n} p_i\,dy_i - \sum_{i=1}^{n} P_i\,dY_i + (H^* - H)\,dx,$$

and assuming $\Phi$ is a function of $x$, $y\_i$ and $Y\_i$, we find that the $2n + 1$ equations

$$p_i = \frac{\partial \Phi}{\partial y_i}, \qquad P_i = -\frac{\partial \Phi}{\partial Y_i}, \qquad H^* = H + \frac{\partial \Phi}{\partial x}$$

establish the connection between the old variables $y\_i$, $p\_i$ and the new variables $Y\_i$, $P\_i$, and also give an expression for the new Hamiltonian $H^\ast$. If the generating function $\Phi$ does not depend on $x$ explicitly, then $H^\ast = H$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Alternative Generating Functions)</span></p>

It may be more convenient to express the generating function in terms of $y\_i$ and $P\_i$ instead of $y\_i$ and $Y\_i$. To this end, we rewrite the condition as

$$d\!\left(\Phi + \sum_{i=1}^{n} P_i Y_i\right) = \sum_{i=1}^{n} p_i\,dy_i + \sum_{i=1}^{n} Y_i\,dP_i + (H^* - H)\,dx,$$

thereby obtaining a new generating function $\Psi = \Phi + \sum\_{i=1}^{n} P\_i Y\_i$, which is to be regarded as a function of $x$, $y\_i$ and $P\_i$. The corresponding canonical transformation is given by the form

$$p_i = \frac{\partial \Psi}{\partial y_i}, \qquad Y_i = \frac{\partial \Psi}{\partial P_i}, \qquad H^* = H + \frac{\partial \Psi}{\partial x}.$$

</div>

### 20. Noether's Theorem

In Sec. 17 we proved that the system of Euler equations corresponding to the functional

$$\int_a^b F(y_1, \ldots, y_n, y_1', \ldots, y_n')\,dx,$$

where $F$ does not depend on $x$ explicitly, has the first integral $H = -F + \sum\_{i=1}^{n} y\_i' F\_{y\_i'}$. The statement "$F$ does not depend on $x$ explicitly" is equivalent to the statement that the integral remains the same if we replace $x$ by the new variable $x^\ast = x + \varepsilon$, where $\varepsilon$ is an arbitrary constant. We now show that even in the general case, there is a connection between the existence of first integrals of a system of Euler equations and the invariance of the corresponding functional under certain transformations.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Invariance of a Functional)</span></p>

Suppose we are given a functional

$$J[y] = \int_{x_0}^{x_1} F(x, y, y')\,dx,$$

where now $y$ indicates the $n$-dimensional vector $(y\_1, \ldots, y\_n)$ and $y'$ the $n$-dimensional vector $(y\_1', \ldots, y\_n')$. Consider the transformation

$$x^* = \Phi(x, y, y'), \qquad y_i^* = \Psi_i(x, y, y'),$$

where $i = 1, \ldots, n$. The transformation carries the curve $\gamma$, with vector equation $y = y(x)$ $(x\_0 \leqslant x \leqslant x\_1)$, into another curve $\gamma^\ast$, with vector equation $y^\ast = y^\ast(x^\ast)$ $(x\_0^\ast \leqslant x^\ast \leqslant x\_1^\ast)$.

The functional $J[y]$ is said to be **invariant** under the transformation if $J[\gamma^\ast] = J[\gamma]$, i.e., if

$$\int_{x_0^*}^{x_1^*} F\!\left(x^*, y^*, \frac{dy^*}{dx^*}\right)dx^* = \int_{x_0}^{x_1} F\!\left(x, y, \frac{dy}{dx}\right)dx.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1</span><span class="math-callout__name">(Translation Invariance)</span></p>

The functional $J[y] = \int\_{x\_0}^{x\_1} y'^2\,dx$ is invariant under the transformation $x^\ast = x + \varepsilon$, $y^\ast = y$, where $\varepsilon$ is an arbitrary constant. In fact, the "transformed" curve $\gamma^\ast$, obtained by shifting $\gamma$ a distance $\varepsilon$ along the $x$-axis, satisfies $y^\ast(x^\ast - \varepsilon) = y(x^\ast - \varepsilon) = y^\ast(x^\ast)$, and so $J[\gamma^\ast] = J[\gamma]$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2</span><span class="math-callout__name">(Non-Invariance)</span></p>

The integral $J[y] = \int\_{x\_0}^{x\_1} xy'^2\,dx$ is *not* invariant under the translation $x^\ast = x + \varepsilon$, $y^\ast = y$. In fact, $J[\gamma^\ast] = \int\_{x\_0 + \varepsilon}^{x\_1 + \varepsilon} (x + \varepsilon)\left[\frac{dy(x)}{dx}\right]^2 dx \neq J[\gamma]$.

</div>

Now suppose that we have a *family* of transformations

$$x^* = \Phi(x, y, y'; \varepsilon), \qquad y_i^* = \Psi_i(x, y, y'; \varepsilon),$$

depending on a parameter $\varepsilon$, where the functions $\Phi$ and $\Psi\_i$ are differentiable with respect to $\varepsilon$, and the value $\varepsilon = 0$ corresponds to the identity transformation:

$$\Phi(x, y, y'; 0) = x, \qquad \Psi_i(x, y, y'; 0) = y_i.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Noether's Theorem)</span></p>

If the functional

$$J[y] = \int_{x_0}^{x_1} F(x, y, y')\,dx$$

is invariant under the family of transformations $x^\ast = \Phi(x, y, y'; \varepsilon)$, $y\_i^\ast = \Psi\_i(x, y, y'; \varepsilon)$ for arbitrary $x\_0$ and $x\_1$, then

$$\sum_{i=1}^{n} F_{y_i'}\psi_i + \left(F - \sum_{i=1}^{n} y_i' F_{y_i'}\right)\varphi = \mathrm{const}$$

along each extremal of $J[y]$, where

$$\varphi(x, y, y') = \left.\frac{\partial \Phi(x, y, y'; \varepsilon)}{\partial \varepsilon}\right\vert_{\varepsilon = 0}, \qquad \psi_i(x, y, y') = \left.\frac{\partial \Psi_i(x, y, y'; \varepsilon)}{\partial \varepsilon}\right\vert_{\varepsilon = 0}.$$

In other words, *every one-parameter family of transformations leaving $J[y]$ invariant leads to a first integral of its system of Euler equations*.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Suppose $\varepsilon$ is a small quantity. Then, by Taylor's theorem,

$$x^* = \Phi(x, y, y'; 0) + \varepsilon\,\frac{\partial \Phi}{\partial \varepsilon}\bigg\vert_{\varepsilon = 0} + o(\varepsilon) = x + \varepsilon\varphi(x, y, y') + o(\varepsilon),$$

$$y_i^* = \Psi_i(x, y, y'; 0) + \varepsilon\,\frac{\partial \Psi_i}{\partial \varepsilon}\bigg\vert_{\varepsilon = 0} + o(\varepsilon) = y_i + \varepsilon\psi_i(x, y, y') + o(\varepsilon).$$

Assuming that the curve $y\_i = y\_i(x)$ $(1 \leqslant i \leqslant n)$ is an extremal of $J[y]$, we can use the formula for the general variation of $J[y]$ from Sec. 13 corresponding to the transformation. In the present case,

$$\delta x = \varepsilon\varphi, \qquad \delta y_i = \varepsilon\psi_i.$$

Since by hypothesis $J[y]$ is invariant under the transformation, $\delta J$ vanishes, i.e.,

$$\left[\sum_{i=1}^{n} F_{y_i'}\psi_i + \left(F - \sum_{i=1}^{n} y_i' F_{y_i'}\right)\varphi\right]\bigg\vert_{x = x_0}^{x = x_1} = 0$$

(the integral term vanishes because $y$ is an extremal). The fact that this holds along each extremal now follows from the arbitrariness of $x\_0$ and $x\_1$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Canonical Variables Form)</span></p>

In terms of the canonical variables $p\_i$ and $H$, the first integral from Noether's theorem becomes simply

$$\sum_{i=1}^{n} p_i \psi_i - H\varphi = \mathrm{const}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3</span><span class="math-callout__name">(Hamiltonian as a Special Case of Noether's Theorem)</span></p>

Consider the functional $J[y] = \int\_{x\_0}^{x\_1} F(y, y')\,dx$, whose integrand does not depend on $x$ explicitly. Then, by the same argument as in Example 1, $J[y]$ is invariant under the one-parameter family of transformations $x^\ast = x + \varepsilon$, $y\_i^\ast = y\_i$. In this case, $\varphi = 1$, $\psi\_i = 0$, and the first integral reduces to just $H = \mathrm{const}$, i.e., the Hamiltonian $H$ is constant along each extremal of $J[y]$. This recovers the result already obtained in Sec. 17.

</div>

### 21. The Principle of Least Action

We now apply the general results obtained in the preceding sections to some mechanical problems. Suppose we are given a system of $n$ particles (mass points), where no constraints whatsoever are imposed on the system. Let the $i$th particle have mass $m\_i$ and coordinates $x\_i, y\_i, z\_i$ $(i = 1, \ldots, n)$. Then the *kinetic energy* of the system is

$$T = \frac{1}{2}\sum_{i=1}^{n} m_i(\dot{x}_i^2 + \dot{y}_i^2 + \dot{z}_i^2),$$

where $t$ denotes the time and the overdot denotes differentiation with respect to $t$.

We assume that the system has *potential energy* $U$, i.e., that there exists a function

$$U = U(t, x_1, y_1, z_1, \ldots, x_n, y_n, z_n)$$

such that the force acting on the $i$th particle has components $X\_i = -\frac{\partial U}{\partial x\_i}$, $Y\_i = -\frac{\partial U}{\partial y\_i}$, $Z\_i = -\frac{\partial U}{\partial z\_i}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lagrangian)</span></p>

Next, we introduce the expression

$$L = T - U,$$

called the **Lagrangian** (function) of the system of particles. Obviously, $L$ is a function of the time $t$ and of the positions $(x\_i, y\_i, z\_i)$ and velocities $(\dot{x}\_i, \dot{y}\_i, \dot{z}\_i)$ of the $n$ particles in the system.

</div>

Suppose that at time $t\_0$ the system is in some fixed position. Then subsequent evolution of the system in time is described by a curve

$$x_i = x_i(t), \quad y_i = y_i(t), \quad z_i = z_i(t) \qquad (i = 1, \ldots, n)$$

in a space of $3n$ dimensions. It can be shown that among all curves passing through the point corresponding to the initial position of the system, the curve which actually describes the motion of the given system, under the influence of the forces acting upon it, satisfies the following condition, known as the **principle of least action**:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Principle of Least Action)</span></p>

*The motion of a system of $n$ particles during the time interval $[t\_0, t\_1]$ is described by those functions $x\_i(t), y\_i(t), z\_i(t)$, $1 \leqslant i \leqslant n$, for which the integral*

$$\int_{t_0}^{t_1} L\,dt,$$

*called the **action**, is a minimum.*

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (sketch)</summary>

We show that the principle of least action implies the usual equations of motion for a system of $n$ particles. If the functional $\int\_{t\_0}^{t\_1} L\,dt$ has a minimum, then the Euler equations

$$\frac{\partial L}{\partial x_i} - \frac{d}{dt}\frac{\partial L}{\partial \dot{x}_i} = 0, \qquad \frac{\partial L}{\partial y_i} - \frac{d}{dt}\frac{\partial L}{\partial \dot{y}_i} = 0, \qquad \frac{\partial L}{\partial z_i} - \frac{d}{dt}\frac{\partial L}{\partial \dot{z}_i} = 0$$

must be satisfied for $i = 1, \ldots, n$. Bearing in mind that the potential energy $U$ depends only on $t, x\_i, y\_i, z\_i$, and not on $\dot{x}\_i, \dot{y}\_i, \dot{z}\_i$, while $T$ is a sum of squares of the velocity components (with coefficients $\frac{1}{2}m\_i$), we can write these equations in the form

$$-\frac{\partial U}{\partial x_i} - \frac{d}{dt}m_i\dot{x}_i = 0, \quad -\frac{\partial U}{\partial y_i} - \frac{d}{dt}m_i\dot{y}_i = 0, \quad -\frac{\partial U}{\partial z_i} - \frac{d}{dt}m_i\dot{z}_i = 0,$$

which reduce to $m\_i\ddot{x}\_i = X\_i$, $m\_i\ddot{y}\_i = Y\_i$, $m\_i\ddot{z}\_i = Z\_i$, i.e., just Newton's equations of motion for a system of $n$ particles, subject to no constraints.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1</span></p>

The principle of least action remains valid in the case where the system of particles is subject to constraints, except that then the admissible curves for which the functional $\int\_{t\_0}^{t\_1} L\,dt$ is considered have to satisfy the constraints. In other words, in this case, application of the principle of least action leads to a variational problem with subsidiary conditions.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2</span></p>

Actually, the principle of least action only holds for sufficiently small time intervals $[t\_0, t\_1]$, and has to be modified for continuous systems.

</div>

### 22. Conservation Laws

We have just seen that the equations of motion of a mechanical system consisting of $n$ particles, with kinetic energy $T$, potential energy $U$ and Lagrangian $L = T - U$, can be obtained from the principle of least action, i.e., by minimizing the integral

$$\int_{t_0}^{t_1} L\,dt = \int_{t_0}^{t_1} (T - U)\,dt.$$

The canonical variables corresponding to this functional turn out to be

$$p_{ix} = \frac{\partial L}{\partial \dot{x}_i} = m_i\dot{x}_i, \qquad p_{iy} = \frac{\partial L}{\partial \dot{y}_i} = m_i\dot{y}_i, \qquad p_{iz} = \frac{\partial L}{\partial \dot{z}_i} = m_i\dot{z}_i,$$

which are just the components of the momentum of the $i$th particle. In terms of $p\_{ix}$, $p\_{iy}$ and $p\_{iz}$, we have

$$H = \sum_{i=1}^{n} (\dot{x}_i p_{ix} + \dot{y}_i p_{iy} + \dot{z}_i p_{iz}) - L = 2T - (T - U) = T + U,$$

so that $H$ is the **total energy** of the system.

Using the form of the integrand in the action functional, we can find various functions which maintain constant values along each trajectory of the system, thereby obtaining so-called *conservation laws*.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Conservation of Energy)</span></p>

Suppose the given system is *conservative*, which means that the Lagrangian $L$ (or more precisely, the potential energy $U$) does not depend on time explicitly. Then, as shown in Sec. 17 (see also Sec. 20, Example 3), $H = \mathrm{const}$ along each extremal, i.e., the total energy of a conservative system does not change during the motion of the system.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Conservation of Momentum)</span></p>

First, we recall that according to Noether's theorem (Sec. 20), invariance of the functional $J[y]$ under the family of transformations

$$x^* = \Phi(x, y, y'; \varepsilon) = x, \qquad y_i^* = \Psi_i(x, y, y'; \varepsilon)$$

implies that the corresponding system of Euler equations has the first integral $\sum\_{i=1}^{n} F\_{y\_i'}\psi\_i = \mathrm{const}$, where $\psi\_i = \frac{\partial \Psi\_i}{\partial \varepsilon}\big\vert\_{\varepsilon=0}$.

Therefore, the invariance of the functional $\int\_{t\_0}^{t\_1} L\,dt$ under the transformation

$$x_i^* = x_i + \varepsilon, \qquad y_i^* = y_i, \qquad z_i^* = z_i$$

implies that

$$\sum_{i=1}^{n} \frac{\partial L}{\partial \dot{x}_i} = \mathrm{const},$$

i.e., $\sum\_{i=1}^{n} p\_{ix} = \mathrm{const}$. The same argument applies to translations in the $y$- and $z$-directions, so that the total momentum of the system is conserved in each direction where the Lagrangian is translation-invariant.

</div>

## Chapter 5: The Second Variation. Sufficient Conditions for a Weak Extremum

Until now, in studying extrema of functionals, we have only considered a particular *necessary* condition for a functional to have a weak (relative) extremum for a given curve $\gamma$, i.e., the condition that the variation of the functional vanish for the curve $\gamma$. In this chapter, we shall derive *sufficient* conditions for a functional to have a weak extremum. To find these sufficient conditions, we must first introduce a new concept, namely, the *second variation* of a functional. We then study the properties of the second variation, and at the same time, we derive some new necessary conditions for an extremum.

As will soon be apparent, there exist sufficient conditions for an extremum which closely resemble the necessary conditions and are easy to apply. These sufficient conditions differ from the necessary conditions (also derived in this chapter) in much the same way as the sufficient conditions $y' = 0$, $y'' > 0$ for a function of one variable to have a minimum differ from the corresponding necessary conditions $y' = 0$, $y'' \geqslant 0$.

### 24. Quadratic Functionals. The Second Variation of a Functional

We begin by introducing some general concepts that will be needed later.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bilinear and Quadratic Functionals)</span></p>

A functional $B[x, y]$ depending on two elements $x$ and $y$, belonging to some normed linear space $\mathscr{R}$, is said to be **bilinear** if it is a linear functional of $y$ for any fixed $x$ and a linear functional of $x$ for any fixed $y$. Thus,

$$B[x + y, z] = B[x, z] + B[y, z], \qquad B[\alpha x, y] = \alpha B[x, y],$$

$$B[x, y + z] = B[x, y] + B[x, z], \qquad B[x, \alpha y] = \alpha B[x, y]$$

for any $x, y, z \in \mathscr{R}$ and any real number $\alpha$.

If we set $y = x$ in a bilinear functional, we obtain an expression called a **quadratic functional**. A quadratic functional $A[x] = B[x, x]$ is said to be **positive definite** if $A[x] > 0$ for every nonzero element $x$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1</span><span class="math-callout__name">(Inner Product as Bilinear Functional)</span></p>

The expression $B[x, y] = \int\_a^b x(t)y(t)\,dt$ is a bilinear functional defined on the space $\mathscr{C}$ of all functions which are continuous in the interval $a \leqslant t \leqslant b$. The corresponding quadratic functional is $A[x] = \int\_a^b x^2(t)\,dt$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2</span><span class="math-callout__name">(Weighted Bilinear Functional)</span></p>

A more general bilinear functional defined on $\mathscr{C}$ is $B[x, y] = \int\_a^b \alpha(t)x(t)y(t)\,dt$, where $\alpha(t)$ is a fixed function. If $\alpha(t) > 0$ for all $t$ in $[a, b]$, then the corresponding quadratic functional $A[x] = \int\_a^b \alpha(t)x^2(t)\,dt$ is positive definite.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3</span><span class="math-callout__name">(Quadratic Functional on $\mathscr{D}\_1$)</span></p>

The expression $A[x] = \int\_a^b [\alpha(t)x^2(t) + \beta(t)x(t)x'(t) + \gamma(t)x'^2(t)]\,dt$ is a quadratic functional defined on the space $\mathscr{D}\_1$ of all functions which are continuously differentiable in $[a, b]$.

</div>

We now introduce the concept of the *second variation* (or *second differential*) of a functional. Let $J[y]$ be a functional defined on some normed linear space $\mathscr{R}$. In Chapter 1 we called the functional $J[y]$ *differentiable* if its increment

$$\Delta J[h] = J[y + h] - J[y]$$

can be written in the form $\Delta J[h] = \varphi[h] + \varepsilon\lVert h\rVert$, where $\varphi[h]$ is a linear functional and $\varepsilon \to 0$ as $\lVert h\rVert \to 0$. The quantity $\varphi[h]$ is the (first) variation, denoted by $\delta J[h]$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Second Variation)</span></p>

Similarly, we say that the functional $J[y]$ is **twice differentiable** if its increment can be written in the form

$$\Delta J[h] = \varphi_1[h] + \varphi_2[h] + \varepsilon\lVert h\rVert^2,$$

where $\varphi\_1[h]$ is a linear functional (in fact, the first variation), $\varphi\_2[h]$ is a quadratic functional, and $\varepsilon \to 0$ as $\lVert h\rVert \to 0$. The quadratic functional $\varphi\_2[h]$ is called the **second variation** (or **second differential**) of the functional $J[y]$, and is denoted by $\delta^2 J[h]$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Necessary Condition for a Minimum)</span></p>

*A necessary condition for the functional $J[y]$ to have a minimum for $y = \hat{y}$ is that*

$$\delta^2 J[y] \geqslant 0$$

*for $y = \hat{y}$ and all admissible $h$. For a maximum, the sign $\geqslant$ in (1) is replaced by $\leqslant$.*

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By definition, we have $\Delta J[h] = \delta J[h] + \delta^2 J[h] + \varepsilon\lVert h\rVert^2$, where $\varepsilon \to 0$ as $\lVert h\rVert \to 0$. According to the necessary condition for an extremum (Theorem 2 of Sec. 3.2), $\delta J[h] = 0$ for $y = \hat{y}$ and all admissible $h$, and hence $\Delta J[h] = \delta^2 J[h] + \varepsilon\lVert h\rVert^2$. Thus, for sufficiently small $\lVert h\rVert$, the sign of $\Delta J[h]$ will be the same as the sign of $\delta^2 J[h]$. Now suppose that $\delta^2 J[h\_0] < 0$ for some admissible $h\_0$. Then for any $\alpha \neq 0$, no matter how small, we have $\delta^2 J[\alpha h\_0] = \alpha^2 \delta^2 J[h\_0] < 0$. Hence $\Delta J[h] = J[\hat{y} + h] - J[\hat{y}] \geqslant 0$ for all sufficiently small $\lVert h\rVert$ is contradicted. This proves the theorem.

</details>
</div>

The condition $\delta^2 J \geqslant 0$ is necessary but of course not sufficient for the functional $J[y]$ to have a minimum for a given function. To obtain a sufficient condition, we introduce the following concept:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Strongly Positive Quadratic Functional)</span></p>

We say that a quadratic functional $\varphi\_2[h]$ defined on some normed linear space $\mathscr{R}$ is **strongly positive** if there exists a constant $k > 0$ such that

$$\varphi_2[h] \geqslant k\lVert h\rVert^2$$

for all $h$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2</span><span class="math-callout__name">(Sufficient Condition for a Minimum)</span></p>

*A sufficient condition for a functional $J[y]$ to have a minimum for $y = \hat{y}$, given that the first variation $\delta J[h]$ vanishes for $y = \hat{y}$, is that its second variation $\delta^2 J[h]$ be strongly positive for $y = \hat{y}$.*

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

For $y = \hat{y}$, we have $\delta J[h] = 0$ for all admissible $h$, and hence $\Delta J[h] = \delta^2 J[h] + \varepsilon\lVert h\rVert^2$, where $\varepsilon \to 0$ as $\lVert h\rVert \to 0$. Moreover, for $y = \hat{y}$, $\delta^2 J[h] \geqslant k\lVert h\rVert^2$, where $k = \mathrm{const} > 0$. Thus, for sufficiently small $\varepsilon\_1$, $\|\varepsilon\| < \frac{1}{2}k$ if $\lVert h\rVert < \varepsilon\_1$. It follows that $\Delta J[h] = \delta^2 J[h] + \varepsilon\lVert h\rVert^2 > \frac{1}{2}k\lVert h\rVert^2 > 0$ if $\lVert h\rVert < \varepsilon\_1$, i.e., $J[y]$ has a minimum for $y = \hat{y}$, as asserted.

</details>
</div>

### 25. The Formula for the Second Variation. Legendre's Condition

Let $F(x, y, z)$ be a function with continuous partial derivatives up to order three with respect to all its arguments. We now find an expression for the second variation in the case of the simplest variational problem, i.e., for functionals of the form

$$J[y] = \int_a^b F(x, y, y')\,dx,$$

defined for curves $y = y(x)$ with fixed end points $y(a) = A$, $y(b) = B$.

First, we give the function $y(x)$ an increment $h(x)$ satisfying the boundary conditions $h(a) = 0$, $h(b) = 0$. Then, using Taylor's theorem with remainder, we write the increment of the functional $J[y]$ as

$$\Delta J[h] = J[y + h] - J[y] = \int_a^b (F_y h + F_{y'} h')\,dx + \frac{1}{2}\int_a^b (F_{yy}h^2 + 2F_{yy'}hh' + F_{y'y'}h'^2)\,dx + \varepsilon,$$

where the first term on the right-hand side is $\delta J[h]$, and the second term, which is quadratic in $h$, is the second variation $\delta^2 J[h]$. Thus, for the functional under consideration,

$$\delta^2 J[h] = \frac{1}{2}\int_a^b (F_{yy}h^2 + 2F_{yy'}hh' + F_{y'y'}h'^2)\,dx.$$

We now transform this into a more convenient form. Integrating the term $2F\_{yy'}hh'$ by parts and taking account of the boundary conditions $h(a) = h(b) = 0$, we obtain

$$\delta^2 J[h] = \int_a^b (Ph'^2 + Qh^2)\,dx,$$

where

$$P = P(x) = \frac{1}{2}F_{y'y'}, \qquad Q = Q(x) = \frac{1}{2}\!\left(F_{yy} - \frac{d}{dx}F_{yy'}\right).$$

This is the expression for the second variation which will be used below.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Increment Formula)</span></p>

The following consequence of the above formulas should be noted. If $J[y]$ has an extremum for the curve $y = y(x)$, and if $y = y(x) + h(x)$ is an admissible curve, then

$$\Delta J[h] = \int_a^b (Ph'^2 + Qh^2)\,dx + \int_a^b (\xi h^2 + \eta h'^2)\,dx,$$

where $\xi, \eta \to 0$ as $\lVert h\rVert\_1 \to 0$. In fact, since $J[y]$ has an extremum for $y = y(x)$, the linear terms in the right-hand side of the expansion vanish, while the remainder $\varepsilon$ can be written in the form $\int\_a^b (\xi h^2 + \eta h'^2)\,dx$. This formula will be used later, when we derive sufficient conditions for a weak extremum (see Sec. 28).

</div>

We now use the expression for the second variation to derive a new necessary condition for a minimum.

The term $Ph'^2$ plays the dominant role in the quadratic functional $\int\_a^b (Ph'^2 + Qh^2)\,dx$, in the sense that $Ph'^2$ can be much larger than the second term $Qh^2$ (this is because we can construct a function $h(x)$ which is itself small but has a large derivative $h'(x)$ in $[a, b]$). Therefore, it might be expected that the coefficient $P(x)$ determines whether the functional takes values with just one sign or values with both signs. We now make this qualitative argument precise:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Necessary Condition on $P(x)$)</span></p>

*A necessary condition for the quadratic functional*

$$\delta^2 J[h] = \int_a^b (Ph'^2 + Qh^2)\,dx,$$

*defined for all functions $h(x) \in \mathscr{D}\_1(a, b)$ such that $h(a) = h(b) = 0$, to be nonnegative is that*

$$P(x) \geqslant 0 \qquad (a \leqslant x \leqslant b).$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Suppose the condition does not hold, i.e., suppose (without loss of generality) that $P(x\_0) = -2\beta$ $(\beta > 0)$ at some point $x\_0$ in $[a, b]$. Then, since $P(x)$ is continuous, there exists an $\alpha > 0$ such that $a \leqslant x\_0 - \alpha$, $x\_0 + \alpha \leqslant b$, and $P(x\_0) < -\beta$ for $x\_0 - \alpha \leqslant x \leqslant x\_0 + \alpha$. We now construct a function $h(x) \in \mathscr{D}\_1(a, b)$ such that the functional is negative. In fact, let

$$h(x) = \begin{cases} \sin^2 \frac{\pi(x - x_0)}{\alpha} & \text{for } x_0 - \alpha \leqslant x \leqslant x_0 + \alpha, \\\\ 0 & \text{otherwise}. \end{cases}$$

Then $\int\_a^b (Ph'^2 + Qh^2)\,dx < -\frac{2\beta\pi^2}{\alpha} + 2M\alpha$, where $M = \max\_{a \leqslant x \leqslant b} \|Q(x)\|$. For sufficiently small $\alpha$, the right side is negative, and hence the functional is negative for the corresponding function $h(x)$. This proves the lemma.

</details>
</div>

Using the lemma and the necessary condition for a minimum proved in Sec. 24, we immediately obtain

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Legendre's Necessary Condition)</span></p>

*A necessary condition for the functional*

$$J[y] = \int_a^b F(x, y, y')\,dx, \qquad y(a) = A, \quad y(b) = B$$

*to have a minimum for the curve $y = y(x)$ is that the inequality*

$$F_{y'y'} \geqslant 0$$

*(Legendre's condition) be satisfied at every point of the curve.*

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Legendre's Attempted Sufficient Condition)</span></p>

Legendre attempted (unsuccessfully) to show that a sufficient condition for $J[y]$ to have a (weak) minimum for the curve $y = y(x)$ is that the *strict* inequality $F\_{y'y'} > 0$ (the **strengthened Legendre condition**) be satisfied at every point of the curve. His approach was to first write the second variation in the form

$$\delta^2 J[h] = \int_a^b [Ph'^2 + 2whh' + (Q + w')h^2]\,dx,$$

where $w(x)$ is an arbitrary differentiable function, using the fact that $\int\_a^b \frac{d}{dx}(wh^2)\,dx = 0$ since $h(a) = h(b) = 0$. Next, he observed that the condition $F\_{y'y'} > 0$ would indeed be sufficient if it were possible to find a function $w(x)$ for which the integrand in the above is a perfect square. However, this is not always possible, as was first shown by Legendre himself, since $w(x)$ would have to satisfy the equation $P(Q + w') = w^2$, and although this equation is "locally solvable," it may not have a solution on a sufficiently large interval.

</div>

Actually, the following argument shows that the requirement $F\_{y'y'}[x, y(x), y'(x)] > 0$ cannot be a sufficient condition for the extremal to be a minimum of the functional $J[y]$. The condition $F\_{y'y'} > 0$, like the condition $F\_y - \frac{d}{dx}F\_{y'} = 0$, is of a "local" character, i.e., it does not pertain to the curve as a whole, but only to individual points of the curve. Therefore, if the condition holds for any two curves $AB$ and $BC$, it also holds for the curve $AC$ formed by joining $AB$ and $BC$. On the other hand, the fact that a functional has an extremum for each part $AB$ and $BC$ of some curve $AC$ does not imply that it has an extremum for the whole curve $AC$. For example, a great circle arc on a given sphere is the shortest curve joining its end points if the arc consists of less than half a circle, but it is not the shortest curve if the arc consists of more than half a circle.

Although the condition $F\_{y'y'} > 0$ alone does not guarantee a minimum, the idea of completing the square of the integrand in the formula for the second variation, with the aim of finding sufficient conditions for an extremum, turns out to be very fruitful. In fact, the differential equation $P(Q + w') = w^2$, which comes to the fore when trying to implement this idea, leads to new necessary conditions for an extremum (which are no longer local!). We shall discuss these matters further in the next two sections.

### 26. Analysis of the Quadratic Functional $\int\_a^b (Ph'^2 + Qh^2)\,dx$

As shown in the preceding section, to pursue our study of the "simplest" variational problem, i.e., that of finding the extrema of the functional

$$J[y] = \int_a^b F(x, y, y')\,dx,$$

where $y(a) = A$, $y(b) = B$, we have to analyze the quadratic functional

$$\int_a^b (Ph'^2 + Qh^2)\,dx,$$

defined on the set of functions $h(x) \in \mathscr{D}\_1(a, b)$ satisfying the conditions $h(a) = 0$, $h(b) = 0$. Here the functions $P$ and $Q$ are related to the function $F$ appearing in the integrand of the original functional by the formulas

$$P = \tfrac{1}{2}F_{y'y'}, \qquad Q = \tfrac{1}{2}(F_{yy} - \tfrac{d}{dx}F_{yy'}).$$

In this section, it will be assumed that the strengthened inequality $P(x) > 0$ $(a \leqslant x \leqslant b)$ holds. We then proceed to find conditions which are both necessary and sufficient for the quadratic functional to be $> 0$ for all admissible $h(x) \neq 0$, i.e., to be *positive definite*. We begin by writing the Euler equation

$$-\frac{d}{dx}(Ph') + Qh = 0$$

corresponding to the quadratic functional. This is a linear differential equation of the second order, which is satisfied, together with the boundary conditions $h(a) = 0$, $h(c) = 0$ $(a < c \leqslant b)$, by the function $h(x) \equiv 0$. However, in general, the equation can have other, nontrivial solutions satisfying the same boundary conditions. In this connection, we introduce the following important concept:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Conjugate Point)</span></p>

The point $\tilde{a}$ $(\neq a)$ is said to be **conjugate** to the point $a$ if the equation $-\frac{d}{dx}(Ph') + Qh = 0$ has a solution which vanishes for $x = a$ and $x = \tilde{a}$ but is not identically zero.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Normalization)</span></p>

If $h(x)$ is a solution of the Euler equation which is not identically zero and satisfies the conditions $h(a) = h(c) = 0$, then $Ch(x)$ is also such a solution, where $C = \mathrm{const} \neq 0$. Therefore, for definiteness, we can impose some kind of normalization on $h(x)$, and in fact we shall usually assume that the constant $C$ has been chosen to make $h'(a) = 1$.

</div>

The following theorem effectively realizes Legendre's idea, mentioned on p. 104.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Positive Definiteness and Conjugate Points)</span></p>

*If $P(x) > 0$ $(a \leqslant x \leqslant b)$, and if the interval $[a, b]$ contains no points conjugate to $a$, then the quadratic functional*

$$\int_a^b (Ph'^2 + Qh^2)\,dx$$

*is positive definite for all $h(x)$ such that $h(a) = h(b) = 0$.*

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

The fact that the functional is positive definite will be proved if we can reduce it to the form $\int\_a^b P\varphi^2(\cdots)\,dx$, where $\varphi^2(\cdots)$ is some expression which cannot be identically zero unless $h(x) \equiv 0$. To achieve this, we add a quantity of the form $d(wh^2)$ to the integrand of the functional, where $w(x)$ is a differentiable function. This will not change the value of the functional, since $h(a) = h(b) = 0$ implies $\int\_a^b d(wh^2) = 0$.

We now select a function $w(x)$ such that the expression $Ph'^2 + Qh^2 + \frac{d}{dx}(wh^2)$ is a perfect square. This will be the case if $w(x)$ is chosen to be a solution of the equation $P(Q + w') = w^2$, which is a Riccati equation. By setting $w = -\frac{u'}{u}P$, where $u$ is a new unknown function, we obtain the equation $-\frac{d}{dx}(Pu') + Qu = 0$, which is just the Euler equation. If there are no points conjugate to $a$ in $[a, b]$, then the Euler equation has a solution which does not vanish anywhere in $[a, b]$, and then there exists a solution of the Riccati equation, given by $w = -\frac{u'}{u}P$, which is defined on the whole interval $[a, b]$. This completes the proof of the theorem.

The quadratic functional can be transformed into $\int\_a^b P\!\left(h' + \frac{w}{P}h\right)^2 dx$, which is nonnegative. Moreover, if this expression vanishes for some function $h(x)$, then $h'(x) + \frac{w}{P}h(x) \equiv 0$, and since $P(x) > 0$, the boundary condition $h(a) = 0$ implies $h(x) \equiv 0$ by the uniqueness theorem. It follows that the functional is actually positive definite.

</details>
</div>

The reduction of the quadratic functional to the form $\int\_a^b P\!\left(h' + \frac{w}{P}h\right)^2 dx$ is the continuous analog of the reduction of a quadratic form to a sum of squares. The absence of points conjugate to $a$ in the interval $[a, b]$ is the analog of the familiar criterion for a quadratic form to be positive definite.

Next, we show that the absence of points conjugate to $a$ in $[a, b]$ is not only sufficient but also necessary for the quadratic functional to be positive definite.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2</span><span class="math-callout__name">(Converse: Conjugate Points Destroy Positive Definiteness)</span></p>

*If the quadratic functional $\int\_a^b (Ph'^2 + Qh^2)\,dx$, where $P(x) > 0$ $(a \leqslant x \leqslant b)$, is positive definite for all $h(x)$ such that $h(a) = h(b) = 0$, then the interval $[a, b]$ contains no points conjugate to $a$.*

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (sketch)</summary>

The idea of the proof is the following: We construct a family of positive definite quadratic functionals, depending on a parameter $t$, which for $t = 1$ gives the functional $\int\_a^b (Ph'^2 + Qh^2)\,dx$ and for $t = 0$ gives the very simple quadratic functional $\int\_a^b h'^2\,dx$, for which there can certainly be no points in $[a, b]$ conjugate to $a$. We then prove that as the parameter $t$ is varied continuously from 0 to 1, no conjugate points can appear in the interval $[a, b]$.

Consider the functional $\int\_a^b [t(Ph'^2 + Qh^2) + (1 - t)h'^2]\,dx$, which is positive definite for all $t$, $0 \leqslant t \leqslant 1$, since (by assumption) the original functional is positive definite. The Euler equation corresponding to this mixed functional is $-\frac{d}{dx}\lbrace [tP + (1 - t)]h'\rbrace  + tQh = 0$. Let $h(x, t)$ be the solution satisfying the initial conditions $h(a, t) = 0$, $h\_x(a, t) = 1$ for all $t$, $0 \leqslant t \leqslant 1$. This solution is a continuous function of the parameter $t$. If $h(x\_0, t\_0) = 0$ at some point $x = x\_0$ in $[a, b]$ for some $t\_0$, this would contradict the assumption that the functional is positive definite. Therefore, no such zeros (and hence no conjugate points) can exist in $[a, b]$.

</details>
</div>

If we replace the condition that the functional be positive definite by the condition that it be nonnegative for all admissible $h(x)$, we obtain the following result:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2'</span><span class="math-callout__name">(Nonnegativity and Interior Conjugate Points)</span></p>

*If the quadratic functional $\int\_a^b (Ph'^2 + Qh^2)\,dx$, where $P(x) > 0$ $(a \leqslant x \leqslant b)$, is nonnegative for all $h(x)$ such that $h(a) = h(b) = 0$, then the interval $[a, b]$ contains no interior points conjugate to $a$.*

</div>

Combining Theorems 1 and 2, we finally obtain

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3</span><span class="math-callout__name">(Complete Characterization of Positive Definiteness)</span></p>

*The quadratic functional $\int\_a^b (Ph'^2 + Qh^2)\,dx$, where $P(x) > 0$ $(a \leqslant x \leqslant b)$, is positive definite for all $h(x)$ such that $h(a) = h(b) = 0$ if and only if the interval $[a, b]$ contains no points conjugate to $a$.*

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Vanishing of the Quadratic Functional)</span></p>

*If the function $h = h(x)$ satisfies the equation $-\frac{d}{dx}(Ph') + Qh = 0$ and the boundary conditions $h(a) = h(b) = 0$, then*

$$\int_a^b (Ph'^2 + Qh^2)\,dx = 0.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

The lemma is an immediate consequence of the formula

$$0 = \int_a^b \left[-\frac{d}{dx}(Ph') + Qh\right]h\,dx = \int_a^b (Ph'^2 + Qh^2)\,dx,$$

which is obtained by integrating by parts and using the boundary conditions.

</details>
</div>

### 27. Jacobi's Necessary Condition. More on Conjugate Points

We now apply the results obtained in the preceding section to the simplest variational problem, i.e., to the functional

$$\int_a^b F(x, y, y')\,dx,$$

with the boundary conditions $y(a) = A$, $y(b) = B$. It will be recalled from Sec. 25 that the second variation of this functional [in the neighborhood of some extremal $y = y(x)$] is given by

$$\int_a^b (Ph'^2 + Qh^2)\,dx,$$

where $P = \frac{1}{2}F\_{y'y'}$, $Q = \frac{1}{2}(F\_{yy} - \frac{d}{dx}F\_{yy'})$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1</span><span class="math-callout__name">(Jacobi Equation)</span></p>

The Euler equation

$$-\frac{d}{dx}(Ph') + Qh = 0$$

of the quadratic functional representing the second variation is called the **Jacobi equation** of the original functional $\int\_a^b F(x, y, y')\,dx$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2</span><span class="math-callout__name">(Conjugate Point for a Functional)</span></p>

The point $\tilde{a}$ is said to be **conjugate** to the point $a$ with respect to the functional $\int\_a^b F(x, y, y')\,dx$ if it is conjugate to $a$ with respect to the quadratic functional representing the second variation, i.e., if the Jacobi equation has a solution which vanishes for $x = a$ and $x = \tilde{a}$ but is not identically zero.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Jacobi's Necessary Condition)</span></p>

*If the extremal $y = y(x)$ corresponds to a minimum of the functional $\int\_a^b F(x, y, y')\,dx$, and if $F\_{y'y'} > 0$ along this extremal, then the open interval $(a, b)$ contains no points conjugate to $a$.*

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

In Sec. 24 it was proved that nonnegativity of the second variation is a necessary condition for a functional $J[y]$ to have a minimum. Moreover, according to Theorem 2' of Sec. 26, if the quadratic functional representing the second variation is nonnegative, the interval $(a, b)$ can contain no points conjugate to $a$. The theorem follows at once from these two facts taken together.

</details>
</div>

We have just defined the Jacobi equation as the Euler equation of the quadratic functional which represents the second variation. We can also derive Jacobi's equation by the following argument: Given that $y = y(x)$ is an extremal, let us examine the conditions which have to be imposed on $h(x)$ if the varied curve $y = y^\ast(x) = y(x) + h(x)$ is to be an extremal also. Substituting $y(x) + h(x)$ into Euler's equation, using Taylor's formula, and bearing in mind that $y(x)$ is already a solution of Euler's equation, we find that

$$F_{yy}h + F_{yy'}h' - \frac{d}{dx}(F_{yy'}h + F_{y'y'}h') = o(h),$$

where $o(h)$ denotes an infinitesimal of order higher than 1 relative to $h$ and its derivative. Neglecting $o(h)$ and combining terms, we obtain the linear differential equation

$$(F_{yy} - \frac{d}{dx}F_{yy'})h - \frac{d}{dx}(F_{y'y'}h') = 0;$$

this is just Jacobi's equation, which we previously wrote in the form $-\frac{d}{dx}(Ph') + Qh = 0$ using the notation of the preceding section. In other words, *Jacobi's equation, except for infinitesimals of order higher than 1, is the differential equation satisfied by the difference between two neighboring (i.e., "infinitely close") extremals*. An equation which is satisfied to within terms of the first order by the difference between two neighboring solutions of a given differential equation is called the **variational equation** (of the original differential equation). Thus, we have just proved that Jacobi's equation is the variational equation of Euler's equation.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Variational Equations in General)</span></p>

These considerations are easily extended to the case of an arbitrary differential equation $F(x, y, y', \ldots, y^{(n)}) = 0$ of order $n$. Let $y(x)$ and $y(x) + \delta y(x)$ be two neighboring solutions. Using Taylor's formula, and bearing in mind that $y(x)$ satisfies the equation, we obtain the linear differential equation

$$F_y(\delta y) + F_{y'}(\delta y)' + \cdots + F_{y^{(n)}}(\delta y)^{(n)} = 0,$$

satisfied by the variation $\delta y$; as before, this equation is called the **variational equation** of the original equation.

</div>

We now return to the concept of a *conjugate point*. Recall that in Sec. 26 the point $\tilde{a}$ was said to be conjugate to the point $a$ if $h(\tilde{a}) = 0$, where $h(x)$ is a solution of Jacobi's equation satisfying the initial conditions $h(a) = 0$, $h'(a) = 1$. As just shown, the difference $z(x) = y^\ast(x) - y(x)$ corresponding to two neighboring extremals $y = y(x)$ and $y = y^\ast(x)$ drawn from the same initial point must satisfy the condition $-\frac{d}{dx}(Pz') + Qz = o(z)$, i.e., to within such an infinitesimal, $y^\ast(x) - y(x)$ is a nonzero solution of Jacobi's equation. This leads to another definition of a conjugate point:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3</span><span class="math-callout__name">(Conjugate Point via Neighboring Extremals)</span></p>

Given an extremal $y = y(x)$, the point $\tilde{M} = (\tilde{a}, y(\tilde{a}))$ is said to be conjugate to the point $M = (a, y(a))$ if at $\tilde{M}$ the difference $y^\ast(x) - y(x)$ is an infinitesimal of order higher than 1 relative to $\lVert y^\ast(x) - y(x)\rVert\_1$, where $y = y^\ast(x)$ is any neighboring extremal drawn from the same initial point $M$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4</span><span class="math-callout__name">(Conjugate Point as Limit of Intersections)</span></p>

Given an extremal $y = y(x)$, the point $\tilde{M} = (\tilde{a}, y(\tilde{a}))$ is said to be conjugate to the point $M = (a, y(a))$ if $\tilde{M}$ is the limit as $\lVert y^\ast(x) - y(x)\rVert\_1 \to 0$ of the points of intersection of $y = y(x)$ and the neighboring extremals $y = y^\ast(x)$ drawn from the same initial point $M$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Equivalence of Definitions)</span></p>

It is clear that if the point $\tilde{M}$ is conjugate to the point $M$ in the sense of Definition 4 (i.e., if the extremals intersect in the way described), then $\tilde{M}$ is also conjugate to $M$ in the sense of Definition 3. The converse is also true, thereby establishing the equivalence of Definitions 3 and 4.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Conjugate Points on a Sphere)</span></p>

Consider the geodesics on a sphere, i.e., the great circle arcs. Each such arc is an extremal of the functional which gives arc length on the sphere. The conjugate of any point $M$ on the sphere is the diametrically opposite point $\tilde{M}$. In fact, given an extremal, *all* extremals with the same initial point $M$ (and not just the neighboring extremals) intersect the given extremal at $\tilde{M}$. This property stems from the fact that a sphere has constant curvature, and is no longer true if the sphere is replaced by a "neighboring" ellipsoid (for example).

</div>

We conclude this section by summarizing the necessary conditions for an extremum found so far: If the functional

$$\int_a^b F(x, y, y')\,dx, \qquad y(a) = A, \quad y(b) = B$$

has a weak extremum for the curve $y = y(x)$, then

1. The curve $y = y(x)$ is an extremal, i.e., satisfies Euler's equation $F\_y - \frac{d}{dx}F\_{y'} = 0$ (see Sec. 4);
2. Along the curve $y = y(x)$, $P(x) \equiv \frac{1}{2}F\_{y'y'} \geqslant 0$ for a minimum and $F\_{y'y'} \leqslant 0$ for a maximum (see Sec. 25);
3. The interval $(a, b)$ contains no points conjugate to $a$ (see Sec. 27).

### 28. Sufficient Conditions for a Weak Extremum

In this section, we formulate a set of conditions which is sufficient for a functional of the form

$$J[y] = \int_a^b F(x, y, y')\,dx, \qquad y(a) = A, \quad y(b) = B$$

to have a weak extremum for the curve $y = y(x)$. It should be noted that the sufficient conditions to be given below closely resemble the necessary conditions given at the end of the preceding section. The necessary conditions were considered separately, since each of them is necessary by itself. However, the sufficient conditions have to be considered as a set, since the presence of an extremum is assured only if all the conditions are satisfied simultaneously.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Sufficient Conditions for a Weak Minimum)</span></p>

*Suppose that for some admissible curve $y = y(x)$, the functional $J[y] = \int\_a^b F(x, y, y')\,dx$, $y(a) = A$, $y(b) = B$ satisfies the following conditions:*

1. *The curve $y = y(x)$ is an extremal, i.e., satisfies Euler's equation $F\_y - \frac{d}{dx}F\_{y'} = 0$;*

2. *Along the curve $y = y(x)$, the **strengthened Legendre condition** $P(x) \equiv \frac{1}{2}F\_{y'y'}[x, y(x), y'(x)] > 0$ holds;*

3. *The interval $[a, b]$ contains no points conjugate to the point $a$ (the **strengthened Jacobi condition**).*

*Then the functional $J[y]$ has a weak minimum for $y = y(x)$.*

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

If $[a, b]$ contains no points conjugate to $a$, and if $P(x) > 0$ in $[a, b]$, then because of the continuity of the solution of Jacobi's equation and the function $P(x)$, we can find a larger interval $[a, b + \varepsilon]$ which also contains no points conjugate to $a$, and such that $P(x) > 0$ in $[a, b + \varepsilon]$. Consider the quadratic functional

$$\int_a^b (Ph'^2 + Qh^2)\,dx - \alpha^2 \int_a^b h'^2\,dx$$

with the Euler equation $-\frac{d}{dx}\lbrace (P - \alpha^2)h'\rbrace  + Qh = 0$. Since $P(x)$ is positive in $[a, b + \varepsilon]$ and hence has a positive greatest lower bound on this interval, and since the solution of the Euler equation satisfying the initial conditions $h(a) = 0$, $h'(0) = 1$ depends continuously on the parameter $\alpha$, for all sufficiently small $\alpha$ we have:

1. $P(x) - \alpha^2 > 0$, $a \leqslant x \leqslant b$;
2. The solution of the Euler equation satisfying the boundary conditions $h(a) = 0$, $h'(a) = 1$ does not vanish for $a < x \leqslant b$.

As shown in Theorem 1 of Sec. 26, these two conditions imply that the quadratic functional $\int\_a^b (Ph'^2 + Qh^2)\,dx$ is positive definite for all sufficiently small $\alpha$. In other words, there exists a positive number $c > 0$ such that

$$\int_a^b (Ph'^2 + Qh^2)\,dx > c\int_a^b h'^2\,dx.$$

Now, from the increment formula established in Sec. 25, we have $\Delta J[h] = \int\_a^b (Ph'^2 + Qh^2)\,dx + \int\_a^b (\xi h^2 + \eta h'^2)\,dx$. Since $\xi, \eta \to 0$ as $\lVert h\rVert\_1 \to 0$, for sufficiently small $\lVert h\rVert\_1$ the second integral is small compared to $c\int\_a^b h'^2\,dx$, and it follows that $\Delta J[h] > 0$. Thus $J[y]$ has a weak minimum for $y = y(x)$.

</details>
</div>

### 29. Generalization to $n$ Unknown Functions

The concept of a conjugate point and the related Jacobi conditions can be generalized to the case where the functional under consideration depends on $n$ functions $y\_1(x), \ldots, y\_n(x)$. We carry over to such functionals the definitions and results given earlier for functionals depending on a single function. To keep the notation simple, we write

$$J[y] = \int_a^b F(x, y, y')\,dx, \tag{50}$$

as before, where now $y$ denotes the $n$-dimensional vector $(y\_1, \ldots, y\_n)$ and $y'$ denotes $(y\_1', \ldots, y\_n')$. By the *scalar product* of two vectors $y = (y\_1, \ldots, y\_n)$ and $z = (z\_1, \ldots, z\_n)$, we mean

$$(y, z) = y_1 z_1 + \cdots + y_n z_n.$$

Whenever the transition from the case of a single function to the case of $n$ functions is straightforward, we shall omit details.

#### 29.1. The Second Variation. The Legendre Condition

If the increment $\Delta J[h]$ of the functional (50), corresponding to the change from $y$ to $y + h$, can be written in the form

$$\Delta J[h] = \varphi_1[h] + \varphi_2[h] + \varepsilon\lVert h\rVert^2,$$

where $\varphi\_1[h]$ is a linear functional, $\varphi\_2[h]$ is a quadratic functional, and $\varepsilon \to 0$ as $\lVert h\rVert \to 0$, then $\varphi\_2[h]$ is called the **second variation** of the original functional (50) and is denoted by $\delta^2 J[h]$. In the case of fixed end points, where

$$h(a) = h(b) = 0,$$

we easily find, applying Taylor's formula, that the second variation of (50) is given by

$$\delta^2 J[h] = \frac{1}{2}\int_a^b \left[\sum_{i,\,k=1}^{n} F_{y_i y_k} h_i h_k + 2\sum_{i,\,k=1}^{n} F_{y_i y_k'} h_i h_k' + \sum_{i,\,k=1}^{n} F_{y_i' y_k'} h_i' h_k'\right] dx. \tag{51}$$

Introducing the matrices

$$F_{yy} = \lVert F_{y_i y_k}\rVert, \qquad F_{yy'} = \lVert F_{y_i y_k'}\rVert, \qquad F_{y'y'} = \lVert F_{y_i' y_k'}\rVert, \tag{52}$$

we can write (51) in the compact form

$$\delta^2 J[h] = \frac{1}{2}\int_a^b \left[(F_{yy}h, h) + 2(F_{yy'}h, h') + (F_{y'y'}h', h')\right] dx, \tag{53}$$

where each term in the integrand is the scalar product of the vector $h$ or $h'$ and the vector obtained by applying one of the matrices (52) to $h$ or $h'$. Then, integrating by parts, we can reduce (53) to the form

$$\int_a^b \left[(Ph', h') + (Qh, h)\right] dx, \tag{54}$$

where $P = P(x)$ and $Q = Q(x)$ are the matrices

$$P = \lVert P_{ik}\rVert = \frac{1}{2}F_{y'y'}, \qquad Q = \lVert Q_{ik}\rVert = \frac{1}{2}\!\left(F_{yy} - \frac{d}{dx}F_{yy'}\right). \tag{*}$$

In deriving (54), we assume that $F\_{yy'}$ is a *symmetric* matrix, i.e., that $F\_{y\_i y\_k'} = F\_{y\_k y\_i'}$ for all $i, k = 1, \ldots, n$ ($F\_{yy'}$ and $F\_{y'y'}$ are automatically symmetric, because of the tacitly assumed smoothness of $F$). Just as in the case of one unknown function, it is easily verified that the term $(Ph', h')$ makes the "main contribution" to the quadratic functional (54). More precisely, we have the following result:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Necessary Condition — Legendre)</span></p>

*A necessary condition for the quadratic functional (54) to be nonnegative for all $h(x)$ such that $h(a) = h(b) = 0$ is that the matrix $P$ be nonnegative definite.*

</div>

#### 29.2. Investigation of the Quadratic Functional

As in Sec. 26, we can investigate the functional (54) without reference to the original functional (50), assuming, however, that $P$ and $Q$ are symmetric matrices. As before (see Sec. 26), we begin by writing the system of Euler equations

$$-\frac{d}{dx}\sum_{i=1}^{n} P_{ik}h_i' + \sum_{i=1}^{n} Q_{ik}h_i = 0 \qquad (k = 1, \ldots, n), \tag{55}$$

corresponding to the functional (54). The equations (55) can be written more concisely as

$$-\frac{d}{dx}(Ph') + Qh = 0. \tag{56}$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1</span><span class="math-callout__name">(Conjugate Point for $n$ Unknown Functions)</span></p>

Let

$$h^{(1)} = (h_{11}, h_{12}, \ldots, h_{1n}),$$

$$h^{(2)} = (h_{21}, h_{22}, \ldots, h_{2n}),$$

$$\vdots$$

$$h^{(n)} = (h_{n1}, h_{n2}, \ldots, h_{nn})$$

be a set of $n$ solutions of the system (55), where the $i$th solution satisfies the initial conditions

$$h_{ik}(a) = 0 \qquad (k = 1, \ldots, n) \tag{58}$$

and

$$h_{ii}(a) = 1, \qquad h_{ik}'(a) = 0 \qquad (k \neq i). \tag{59}$$

Then the point $\tilde{a}$ ($\neq a$) is said to be **conjugate** to the point $a$ if the determinant

$$\begin{vmatrix} h_{11}(x) & h_{12}(x) & \cdots & h_{1n}(x) \\\ h_{21}(x) & h_{22}(x) & \cdots & h_{2n}(x) \\\ \vdots & & \ddots & \vdots \\\ h_{n1}(x) & h_{n2}(x) & \cdots & h_{nn}(x) \end{vmatrix} \tag{60}$$

vanishes for $x = \tilde{a}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2</span><span class="math-callout__name">(Positive Definite $P$ and No Conjugate Points $\Rightarrow$ Positive Definite Functional)</span></p>

*If $P$ is a positive definite symmetric matrix, and if the interval $[a, b]$ contains no points conjugate to $a$, then the quadratic functional (54) is positive definite for all $h(x)$ such that $h(a) = h(b) = 0$.*

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

The proof follows the same plan as the proof of Theorem 1 of Sec. 26. Let $W$ be an arbitrary differentiable symmetric matrix. Then

$$0 = \int_a^b \frac{d}{dx}(Wh, h)\,dx = \int_a^b (W'h, h)\,dx + 2\int_a^b (Wh, h')\,dx$$

for every vector $h$ satisfying the boundary conditions (58). Therefore, we can add the expression $(W'h, h) + 2(Wh, h')$ to the integrand of (54), obtaining

$$\int_a^b \left[(Ph', h') + 2(Wh, h') + (Qh, h) + (W'h, h)\right] dx, \tag{61}$$

without changing the value of (54).

We now try to select a matrix $W$ such that the integrand of (61) is a perfect square. This will be the case if $W$ is chosen to be a solution of the equation

$$Q + W' = WP^{-1}W, \tag{62}$$

which we call the **matrix Riccati equation** (cf. p. 108). In fact, if we use (62), the integrand of (61) becomes

$$(Ph', h') + 2(Wh, h') + (WP^{-1}Wh, h). \tag{63}$$

Since $P$ is a positive definite symmetric matrix, the square root $P^{1/2}$ exists, is itself positive definite and symmetric, and has the inverse $P^{-1/2}$. Therefore, we can write (63) as the "perfect square"

$$(P^{1/2}h' + P^{-1/2}Wh,\; P^{1/2}h' + P^{-1/2}Wh).$$

Since the integrand of (61) is nonnegative, and since a continuous nonnegative function which integrates to zero must vanish identically, we find that $P^{1/2}h' + P^{-1/2}Wh = 0$ everywhere, which implies $h \equiv 0$ unless the matrix Riccati equation fails to have a solution. It follows that if the matrix Riccati equation (62) has a solution $W$ defined on the whole interval $[a, b]$, then the functional (54) is positive definite.

Thus, the proof of the theorem reduces to showing that the absence of points in $[a, b]$ which are conjugate to $a$ guarantees that (62) has a solution defined on the whole interval $[a, b]$. Making the substitution

$$W = -PU'U^{-1} \tag{64}$$

in (62), where $U$ is a new unknown matrix, we obtain the equation

$$-\frac{d}{dx}(PU') + QU = 0, \tag{65}$$

which is just the matrix form of equation (56). The solution of (65) satisfying the initial conditions $U(0) = \theta$, $U'(0) = I$ (where $\theta$ is the zero matrix and $I$ the unit matrix of order $n$) is precisely the set of solutions (57) of the system (55) which satisfy the initial conditions (58) and (59). If $[a, b]$ contains no points conjugate to $a$, we can show that (65) has a solution $U(x)$ whose determinant does not vanish anywhere in $[a, b]$, and then there exists a solution of (62), given by (64), which is defined on the whole interval $[a, b]$. In other words, we can actually find a matrix $W$ which converts the integrand of the functional (61) into a perfect square, in the way described. This completes the proof of the theorem.

</details>
</div>

Next we show, as in Sec. 26, that the absence of points conjugate to $a$ in the interval $[a, b]$ is not only sufficient but also necessary for the functional (53) to be positive definite.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Solutions of the Euler System and the Quadratic Functional)</span></p>

If $h(x) = (h\_1(x), \ldots, h\_n(x))$ satisfies the system (55) and the boundary conditions $h(a) = h(b) = 0$, then

$$\int_a^b \left[(Ph', h') + (Qh, h)\right] dx = 0. \tag{66}$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

The lemma is an immediate consequence of the formula

$$0 = \int_a^b \!\left(-\frac{d}{dx}(Ph') + Qh,\, h\right) dx = \int_a^b \left[(Ph', h') + (Qh, h)\right] dx,$$

which is obtained by integrating by parts and using (66).

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3</span><span class="math-callout__name">(Positive Definite Functional $\Rightarrow$ No Conjugate Points)</span></p>

If the quadratic functional

$$\int_a^b \left[(Ph', h') + (Qh, h)\right] dx, \tag{67}$$

where $P$ is a positive definite symmetric matrix, is positive definite for all $h(x)$ such that $h(a) = h(b) = 0$, then the interval $[a, b]$ contains no points conjugate to $a$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

The proof follows the same plan as the proof of the corresponding theorem for the case of one unknown function (Theorem 2 of Sec. 26). We consider the positive definite quadratic functional

$$\int_a^b \left\lbrace t\left[(Ph', h') + (Qh, h)\right] + (1 - t)(h', h')\right\rbrace  dx. \tag{68}$$

The system of Euler equations corresponding to (68) is

$$-\frac{d}{dx}\!\left[t\sum_{i=1}^{n} P_{ik}h_i' + (1-t)h_k'\right] + t\sum_{i=1}^{n} Q_{ik}h_i = 0 \qquad (k = 1, \ldots, n), \tag{69}$$

which for $t = 1$ reduces to the system (55), and for $t = 0$ reduces to $h\_k'' = 0$ $(k = 1, \ldots, n)$.

Suppose the interval $[a, b]$ contains a point $\tilde{a}$ conjugate to $a$, i.e., suppose the determinant (60) vanishes for $x = \tilde{a}$. Then there exists a linear combination $h(x)$ of the solutions (57) which is not identically zero such that $h(\tilde{a}) = 0$. Moreover, there exists a nontrivial solution $h(x, t)$ of the system (69) which depends continuously on $t$ and reduces to $h(x)$ for $t = 1$. It is clear that $\tilde{a} \neq b$, since otherwise, according to the lemma, the positive definite functional (67) would vanish for $h(x) \neq 0$, which is impossible. The fact that $\tilde{a}$ cannot be an interior point of $[a, b]$ is proved by the same kind of argument as used in Theorem 2 of Sec. 26, for the case of a scalar function $h(x)$. Further details are left to the reader.

</details>
</div>

Suppose now that we only require that the functional (67) be nonnegative. Then, by the same argument as used to prove Theorem 2' of Sec. 26, we have

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3'</span><span class="math-callout__name">(Nonnegative Functional $\Rightarrow$ No Interior Conjugate Points)</span></p>

*If the quadratic functional $\int\_a^b [(Ph', h') + (Qh, h)]\,dx$, where $P$ is a positive definite symmetric matrix, is nonnegative for all $h(x)$ such that $h(a) = h(b) = 0$, then the interval $[a, b]$ contains no interior points conjugate to $a$.*

</div>

Finally, combining Theorems 2 and 3, we obtain

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4</span><span class="math-callout__name">(Characterization of Positive Definite Quadratic Functionals)</span></p>

*The quadratic functional $\int\_a^b [(Ph', h') + (Qh, h)]\,dx$, where $P$ is a positive definite symmetric matrix, is positive definite for all $h(x)$ such that $h(a) = h(b) = 0$ if and only if the interval $[a, b]$ contains no point conjugate to $a$.*

</div>

#### 29.3. Jacobi's Necessary Condition. More on Conjugate Points

We now apply the results just obtained to the original functional

$$J[y] = \int_a^b F(x, y, y')\,dx, \qquad y(a) = M_0, \quad y(b) = M_1, \tag{70}$$

where $M\_0$ and $M\_1$ are two fixed points, recalling that the second variation of (70) is given by

$$\int_a^b \left[(Ph', h') + (Qh, h)\right] dx, \tag{71}$$

where

$$P = \frac{1}{2}F_{y'y'}, \qquad Q = \frac{1}{2}\!\left(F_{yy} - \frac{d}{dx}F_{yy'}\right). \tag{72}$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2</span><span class="math-callout__name">(Jacobi System)</span></p>

The system of Euler equations

$$-\frac{d}{dx}\sum_{i=1}^{n} P_{ik}h_i' + \sum_{i=1}^{n} Q_{ik}h_i = 0 \qquad (k = 1, \ldots, n),$$

or more concisely

$$-\frac{d}{dx}(Ph') + Qh = 0, \tag{73}$$

of the quadratic functional (71) is called the **Jacobi system** of the original functional (70).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3</span><span class="math-callout__name">(Conjugate Point for the Original Functional)</span></p>

The point $\tilde{a}$ is said to be **conjugate** to the point $a$ with respect to the functional (70) if it is conjugate to $a$ with respect to the quadratic functional (71) which is the second variation of (70), i.e., if it is conjugate to $a$ in the sense of Definition 1.

</div>

Since nonnegativity of the second variation is a necessary condition for the functional (70) to have a minimum (see Theorem 1 of Sec. 24), Theorem 3' immediately implies

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5</span><span class="math-callout__name">(Jacobi's Necessary Condition for $n$ Functions)</span></p>

*If the extremal $y\_1 = y\_1(x), \ldots, y\_n = y\_n(x)$ corresponds to a minimum of the functional (70), and if the matrix $F\_{y'y'}[x, y(x), y'(x)]$ is positive definite, then the open interval $(a, b)$ contains no points conjugate to $a$.*

</div>

So far, we have said that the point $\tilde{a}$ is conjugate to $a$ if the determinant formed from $n$ linearly independent solutions of the Jacobi system, satisfying certain initial conditions, vanishes for $x = \tilde{a}$. As in the case $n = 1$, this basic definition is equivalent to two others, which involve only extremals of the functional (70), and not solutions of the Jacobi system:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4</span><span class="math-callout__name">(Conjugate Point via Neighboring Extremals)</span></p>

Suppose $n$ neighboring extremals

$$y_1 = y_{i1}(x), \ldots, y_n = y_{in}(x) \qquad (i = 1, \ldots, n)$$

start from the same $n$-dimensional point, with directions which are close together but linearly independent. Then the point $\tilde{a}$ is said to be **conjugate** to the point $a$ if the value of the determinant

$$\begin{vmatrix} y_{11}(x) & y_{12}(x) & \cdots & y_{1n}(x) \\\ y_{21}(x) & y_{22}(x) & \cdots & y_{2n}(x) \\\ \vdots & & \ddots & \vdots \\\ y_{n1}(x) & y_{n2}(x) & \cdots & y_{nn}(x) \end{vmatrix}$$

for $x = \tilde{a}$ is an infinitesimal whose order is higher than its values for $a < x < \tilde{a}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5</span><span class="math-callout__name">(Conjugate Point via Limit of Intersections)</span></p>

Given an extremal $\gamma$ with equations $y\_1 = y\_1(x), \ldots, y\_n = y\_n(x)$, the point

$$\tilde{M} = (\tilde{a},\; y_1(\tilde{a}), \ldots, y_n(\tilde{a}))$$

is said to be **conjugate** to the point $M = (a,\; y\_1(a), \ldots, y\_n(a))$ if $\gamma$ has a sequence of neighboring extremals drawn from the same initial point $M$, such that each neighboring extremal intersects $\gamma$ and the points of intersection have $\tilde{M}$ as their limit.

</div>

The equivalence of all these definitions of a conjugate point is proved by using considerations similar to those given for the case of a single unknown function (see Sec. 27).

#### 29.4. Sufficient Conditions for a Weak Extremum

Theorem 2 and an argument like that used to prove the corresponding theorem of Sec. 28 (for the scalar case) imply

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6</span><span class="math-callout__name">(Sufficient Conditions for a Weak Minimum — $n$ Functions)</span></p>

*Suppose that for some admissible curve $\gamma$ with equations $y\_1 = y\_1(x), \ldots, y\_n = y\_n(x)$, the functional (70) satisfies the following conditions:*

1. *The curve $\gamma$ is an extremal, i.e., satisfies the system of Euler equations*

   $$F_{y_i} - \frac{d}{dx}F_{y_i'} = 0 \qquad (i = 1, \ldots, n);$$

2. *Along $\gamma$ the matrix $P(x) = \frac{1}{2}F\_{y'y'}[x, y(x), y'(x)]$ is positive definite;*

3. *The interval $[a, b]$ contains no points conjugate to the point $a$.*

*Then the functional (70) has a weak minimum for the curve $\gamma$.*

</div>

### 30. Connection between Jacobi's Condition and the Theory of Quadratic Forms

According to Theorem 3 of Sec. 26, the quadratic functional

$$\int_a^b (Ph'^2 + Qh^2)\,dx, \tag{74}$$

where $P(x) > 0$ for $a \leqslant x \leqslant b$, is positive definite for all $h(x)$ such that $h(a) = h(b) = 0$ if and only if the interval $[a, b]$ contains no points conjugate to $a$. The functional (74) is the infinite-dimensional analog of a quadratic form. Therefore, to obtain conditions for (74) to be positive definite, it is natural to start from the conditions for a quadratic form defined on an $n$-dimensional space to be positive definite, and then take the limit as $n \to \infty$.

This may be done as follows: By introducing the points $a = x\_0, x\_1, \ldots, x\_n, x\_{n+1} = b$, we divide the interval $[a, b]$ into $n + 1$ equal parts of length

$$\Delta x = x_{i+1} - x_i = \frac{b - a}{n + 1} \qquad (i = 0, 1, \ldots, n).$$

Then we consider the quadratic form

$$\sum_{i=0}^{n} \left[P_i\!\left(\frac{h_{i+1} - h_i}{\Delta x}\right)^{\!2} + Q_i h_i^2\right] \Delta x, \tag{75}$$

where $P\_i$, $Q\_i$ and $h\_i$ are the values of the functions $P(x)$, $Q(x)$ and $h(x)$ at the point $x = x\_i$. This quadratic form is a "finite-dimensional approximation" to the functional (74). Grouping similar terms and bearing in mind that $h\_0 = h(a) = 0$, $h\_{n+1} = h(b) = 0$, we can write (75) as

$$\sum_{i=1}^{n} \left[\!\left(Q_i \Delta x + \frac{P_{i-1} + P_i}{\Delta x}\right)h_i^2 - 2\frac{P_{i-1}}{\Delta x}\, h_{i-1} h_i\right]. \tag{76}$$

In other words, the quadratic functional (74) can be approximated by a quadratic form in $n$ variables $h\_1, \ldots, h\_n$, with the $n \times n$ matrix

$$\begin{pmatrix} a_1 & b_1 & 0 & \cdots & 0 & 0 & 0 \\\ b_1 & a_2 & b_2 & \cdots & 0 & 0 & 0 \\\ 0 & b_2 & a_3 & \cdots & 0 & 0 & 0 \\\ \vdots & & & \ddots & & & \vdots \\\ 0 & 0 & 0 & \cdots & b_{n-2} & a_{n-1} & b_{n-1} \\\ 0 & 0 & 0 & \cdots & 0 & b_{n-1} & a_n \end{pmatrix}, \tag{77}$$

where

$$a_i = Q_i\,\Delta x + \frac{P_{i-1} + P_i}{\Delta x} \qquad (i = 1, \ldots, n) \tag{78}$$

and

$$b_i = -\frac{P_i}{\Delta x} \qquad (i = 1, \ldots, n - 1). \tag{79}$$

A symmetric matrix like (77), all of whose elements vanish except those appearing on the principal diagonal and on the two adjoining diagonals, is called a **Jacobi matrix**, and a quadratic form with such a matrix is called a **Jacobi form**. For any Jacobi matrix, there is a recurrence relation between the *descending principal minors*, i.e., between the determinants

$$D_i = \begin{vmatrix} a_1 & b_1 & 0 & \cdots & 0 & 0 \\\ b_1 & a_2 & b_2 & \cdots & 0 & 0 \\\ 0 & b_2 & a_3 & \cdots & 0 & 0 \\\ \vdots & & & \ddots & & \vdots \\\ 0 & 0 & 0 & \cdots & a_{i-1} & b_{i-1} \\\ 0 & 0 & 0 & \cdots & b_{i-1} & a_i \end{vmatrix} \tag{80}$$

where $i = 1, \ldots, n$. In fact, expanding $D\_i$ with respect to the elements of the last row, we obtain the recursion relation

$$D_i = a_i D_{i-1} - b_{i-1}^2\,D_{i-2}, \tag{81}$$

which allows us to determine the minors $D\_3, \ldots, D\_n$ in terms of the first two minors $D\_1$ and $D\_2$. Moreover, if we set $D\_0 = 1$, $D\_{-1} = 0$, then (81) is valid for all $i = 1, \ldots, n$, and uniquely determines $D\_1, \ldots, D\_n$.

According to a familiar result, sometimes called the **Sylvester criterion**, a quadratic form

$$\sum_{i,\,k=1}^{n} a_{ik}\,\xi_i\,\xi_k \qquad (a_{ki} = a_{ik})$$

is positive definite if and only if the descending principal minors

$$a_{11}, \quad \begin{vmatrix} a_{11} & a_{12} \\\ a_{21} & a_{22}\end{vmatrix}, \quad \begin{vmatrix} a_{11} & a_{12} & a_{13} \\\ a_{21} & a_{22} & a_{23} \\\ a_{31} & a_{32} & a_{33}\end{vmatrix}, \quad \ldots, \quad \det\lVert a_{ik}\rVert$$

are all positive. Applied to the present problem, this criterion states that the Jacobi form (76), with matrix (77), is positive definite if and only if all the quantities defined by (81) are positive, where $i = 1, \ldots, n$ and $D\_0 = 1$, $D\_{-1} = 0$.

We now use this result to obtain a criterion for the quadratic functional (74) to be positive definite. Thus, we examine what happens to the recurrence relation (81) as $n \to \infty$. Substituting for the coefficients $a\_i$ and $b\_i$ from (78) and (79), we can write (81) in the form

$$D_i = \left(Q_i \Delta x + \frac{P_{i-1} + P_i}{\Delta x}\right)D_{i-1} - \frac{P_{i-1}^2}{(\Delta x)^2}\,D_{i-2} \qquad (i = 1, \ldots, n). \tag{82}$$

It is obviously impossible to pass directly to the limit $n \to \infty$ (i.e., $\Delta x \to 0$) in (82), since the coefficients of $D\_{i-1}$ and $D\_{i-2}$ become infinite. To avoid this difficulty, we make the "change of variables"

$$D_i = \frac{P_1 \cdots P_i\, Z_{i+1}}{(\Delta x)^{i+1}} \qquad (i = 1, \ldots, n),$$

$$D_0 = \frac{Z_1}{\Delta x} = 1,$$

$$D_{-1} = Z_0 = 0. \tag{83}$$

In terms of the variables $Z\_i$, the recurrence relation (82) becomes

$$Q_i Z_i(\Delta x)^2 + P_{i-1}Z_i + P_i Z_i - P_i Z_{i+1} - P_{i-1}Z_{i-1} = 0$$

or

$$Q_i Z_i - \frac{1}{\Delta x}\!\left(P_i\,\frac{Z_{i+1} - Z_i}{\Delta x} - P_{i-1}\,\frac{Z_i - Z_{i-1}}{\Delta x}\right) = 0 \qquad (i = 1, \ldots, n). \tag{84}$$

Passing to the limit $\Delta x \to 0$ in (84), we obtain the differential equation

$$-\frac{d}{dx}(PZ') + QZ = 0, \tag{85}$$

which is just the Jacobi equation! The condition that the quantities $D\_i$ satisfying the relation (82) be positive is equivalent to the condition that the quantities $Z\_i$ satisfying the difference equation (84) be positive, since the factor $P\_1 \cdots P\_i / (\Delta x)^{i+1}$ is always positive [because $P(x) > 0$]. Thus, we have proved that *the quadratic form (76) is positive definite if and only if all but the first of the $n + 2$ quantities $Z\_0, Z\_1, \ldots, Z\_{n+1}$ satisfying the difference equation (84) are positive*.

If we consider the polygonal line $\Pi\_n$ with vertices $(a, Z\_0)$, $(x\_1, Z\_1)$, $\ldots$, $(b, Z\_{n+1})$, recall that $a = x\_0$, $b = x\_{n+1}$, the condition that $Z\_0 = 0$ and $Z\_i > 0$ for $i = 1, \ldots, n + 1$ means that $\Pi\_n$ does not intersect the interval $[a, b]$ except at the end point $a$. As $\Delta x \to 0$, the difference equation (84) goes into the Jacobi differential equation (85), and the polygonal line $\Pi\_n$ goes into a nontrivial solution of (85) which satisfies the initial condition

$$Z(a) = Z_0 = 0, \qquad Z'(a) = \lim_{\Delta x \to 0} \frac{Z_1 - Z_0}{\Delta x} = \lim_{\Delta x \to 0} \frac{\Delta x}{\Delta x} = 1$$

and does not vanish for $a < x \leqslant b$. In other words, as $n \to \infty$, the Jacobi form (76) goes into the quadratic functional (74), and the condition that (76) be positive definite goes into precisely the condition for (74) to be positive definite given in Theorem 3 of Sec. 26, the condition that $[a, b]$ contain no points conjugate to $a$. The legitimacy of this passage to the limit can be made completely rigorous, but we omit the details.

## Chapter 6: Fields. Sufficient Conditions for a Strong Extremum

In our study of sufficient conditions for a weak extremum, we introduced the important concept of a conjugate point. The simplest and most natural way to introduce this concept is based on the use of families of neighboring extremals (see Sec. 27). The conjugate of a point $M$ lying on an extremal $\gamma$ is defined as the limit of the points of intersection of $\gamma$ with the neighboring extremals drawn from $M$.

The utility of studying families of extremals rather than individual extremals is particularly apparent when we turn to the problem of finding sufficient conditions for a *strong* extremum. The study of such families of extremals is intimately connected with the important concept of a *field*, which we introduce in the next section. Since the concept of a field is useful in many problems, we first give a general definition of a field, which is not directly related to variational problems.

### 31. Consistent Boundary Conditions. General Definition of a Field

Consider a system of second-order differential equations

$$y_i'' = f_i(x, y_1, \ldots, y_n, y_1', \ldots, y_n') \qquad (i = 1, \ldots, n), \tag{1}$$

solved explicitly for the second derivatives. In order to single out a definite solution of this system, we have to specify $2n$ conditions, e.g., boundary conditions of the form

$$y_i' = \psi_i(y_1, \ldots, y_n) \qquad (i = 1, \ldots, n) \tag{2}$$

for two values of $x$, say $x\_1$ and $x\_2$. Boundary conditions of this kind are commonly encountered in variational problems. If we require that the boundary conditions (2) hold only at one point, they determine a solution of the system (1) which depends on $n$ parameters.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1</span><span class="math-callout__name">(Mutually Consistent Boundary Conditions)</span></p>

The boundary conditions

$$y_i' = \psi_i^{(1)}(y_1, \ldots, y_n) \qquad (i = 1, \ldots, n), \tag{3}$$

prescribed for $x = x\_1$, and the boundary conditions

$$y_i' = \psi_i^{(2)}(y_1, \ldots, y_n) \qquad (i = 1, \ldots, n), \tag{4}$$

prescribed for $x = x\_2$, are said to be **(mutually) consistent** if every solution of the system (1) satisfying the boundary conditions (3) at $x = x\_1$ also satisfies the boundary conditions (4) at $x = x\_2$, and conversely.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2</span><span class="math-callout__name">(Field of Directions)</span></p>

Suppose the boundary conditions

$$y_i' = \psi_i(x, y_1, \ldots, y_n) \qquad (i = 1, \ldots, n) \tag{5}$$

(where the $\psi\_i$ are continuously differentiable functions) are prescribed for every $x$ in the interval $[a, b]$, and suppose they are consistent for every pair of points $x\_1$, $x\_2$ in $[a, b]$. Then the family of mutually consistent boundary conditions (5) is called a **field** (of directions) for the given system (1).

</div>

As is clear from (5), boundary conditions prescribed for every value of $x$ define a system of first-order differential equations. The requirement that the boundary conditions be consistent for different values of $x$ means that the solutions of the system (5) must also satisfy the system (1), i.e., that (1) is implied by (5).

Because of the existence and uniqueness theorem for systems of differential equations, one and only one integral curve of the system (5) passes through each point $(x, y\_1, \ldots, y\_n)$ of the region $R$ where the functions $\psi\_i(x, y\_1, \ldots, y\_n)$ are defined. According to what has just been said, each of these curves is at the same time a solution of the system (1). Thus, specifying a field (5) of the system (1) in some region $R$ defines an $n$-parameter family of solutions of (1), such that one and only one curve from the family passes through each point of $R$. The curves of the family will be called *trajectories* of the field.

The following theorem gives conditions which must be satisfied by the functions $\psi\_i(x, y\_1, \ldots, y\_n)$, $1 \leqslant i \leqslant n$, if the system (5) is to be a field for the system (1):

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hamilton-Jacobi System Characterization of a Field)</span></p>

The first-order system

$$y_i' = \psi_i(x, y_1, \ldots, y_n) \qquad (a \leqslant x \leqslant b;\; 1 \leqslant i \leqslant n) \tag{6}$$

is a field for the second-order system

$$y_i'' = f_i(x, y_1, \ldots, y_n, y_1', \ldots, y_n') \tag{7}$$

if and only if the functions $\psi\_i(x, y\_1, \ldots, y\_n)$ satisfy the following system of partial differential equations, called the **Hamilton-Jacobi system** for the original system (7):

$$\frac{\partial \psi_i}{\partial x} + \sum_{k=1}^{n} \frac{\partial \psi_i}{\partial y_k}\,\psi_k = f_i(x, y_1, \ldots, y_n, \psi_1, \ldots, \psi_n). \tag{8}$$

Thus, every solution of the Hamilton-Jacobi system (8) gives a field for the original system (7).

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Differentiating (6) with respect to $x$, we obtain

$$y_i'' = \frac{\partial \psi_i}{\partial x} + \sum_{k=1}^{n} \frac{\partial \psi_i}{\partial y_k}\,\frac{dy_k}{dx},$$

i.e.,

$$y_i'' = \frac{\partial \psi_i}{\partial x} + \sum_{k=1}^{n} \frac{\partial \psi_i}{\partial y_k}\,\psi_k.$$

Thus, the system (7) is a consequence of the system (6) if and only if (8) holds.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1</span><span class="math-callout__name">(Field for a Single Linear Equation)</span></p>

Consider a single linear differential equation $y'' = p(x)y$. The corresponding Hamilton-Jacobi system reduces to a single equation

$$\frac{\partial \psi}{\partial x} + \frac{\partial \psi}{\partial y}\,\psi = p(x)y,$$

i.e.,

$$\frac{\partial \psi}{\partial x} + \frac{1}{2}\frac{\partial \psi^2}{\partial y} = p(x)y. \tag{10}$$

The set of solutions of (10) depends on an arbitrary function, and according to the theorem, each of these solutions is a field for the equation $y'' = p(x)y$. The simplest solutions of (10) are those that are linear in $y$: $\psi(x, y) = \alpha(x)y$. Substituting into (10), we obtain $\alpha'(x)y + \alpha^2(x)y = p(x)y$. Thus, $\alpha(x)$ satisfies the **Riccati equation** $\alpha' + \alpha^2 = p(x)$. Solving this and setting $y' = \alpha(x)y$, we obtain a field (which is linear in $y$) for the equation $y'' = p(x)y$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2</span><span class="math-callout__name">(Field for a System of Linear Equations)</span></p>

In the same way, we can find the simplest field for a system of linear differential equations $Y'' = P(x)Y$, where $Y = (y\_1, \ldots, y\_n)$ and $P(x) = \lVert p\_{ik}(x)\rVert$ is a matrix. The system of Hamilton-Jacobi equations corresponding to this system is

$$\frac{\partial \psi_i}{\partial x} + \sum_{k=1}^{n} \frac{\partial \psi_i}{\partial y_k}\,\psi_k = \sum_{k=1}^{n} p_{ik}(x)y_k \qquad (i = 1, \ldots, n). \tag{14}$$

Looking for a solution which is linear in $Y$, i.e., $\psi\_i(x, y\_1, \ldots, y\_n) = \sum\_{k=1}^{n} \alpha\_{ik}(x)y\_k$, or in vector notation $\Psi = AY$, and substituting into (14), we find that the matrix $A(x) = \lVert \alpha\_{ik}\rVert$ satisfies the **matrix Riccati equation**

$$\frac{d}{dx}A(x) + A^2(x) = P(x),$$

and the functions $\Psi = AY$ define a field (linear in $y$) for the system $Y'' = P(x)Y$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection with the Sweep Method)</span></p>

The concept of a field is intimately related to the solution of boundary value problems for systems of second-order differential equations by the so-called "sweep method." Consider the simple case of $y'' = p(x)y + f(x)$ with boundary conditions $y'(a) = c\_0 y(a) + d\_0$ and $y'(b) = c\_1 y(b) + d\_1$. We construct a first-order equation $y' = \alpha(x)y + \beta(x)$ such that all its solutions satisfy the boundary conditions and the original equation. Setting $\alpha(a) = c\_0$, $\beta(a) = d\_0$ to meet the first requirement, differentiation gives $y'' = \alpha' y + \alpha y' + \beta' = [\alpha' + \alpha^2]y + \alpha\beta + \beta'$. This equals $p(x)y + f(x)$ if and only if $\alpha' + \alpha^2 = p(x)$ and $\beta' + \alpha\beta = f(x)$. The process of shifting the boundary condition from $x = a$ to every other point in $[a, b]$ is called the "forward sweep." Setting $x = b$ gives $y'(b) = \alpha(b)y(b) + \beta(b)$, which together with the boundary condition at $b$ determines $y(b)$ and $y'(b)$. Finding $y(b)$ is the "backward sweep." The forward sweep is nothing but the construction of a field linear in $y$, and the system the $\alpha$, $\beta$ must satisfy is precisely the Hamilton-Jacobi system for this case.

</div>

### 32. The Field of a Functional

#### 32.1. Euler Equations and Boundary Conditions

We now apply the considerations of the preceding section to variational problems. The Euler equations

$$F_{y_i} - \frac{d}{dx}F_{y_i'} = 0 \qquad (i = 1, \ldots, n),$$

corresponding to the functional

$$\int_a^b F(x, y_1, \ldots, y_n, y_1', \ldots, y_n')\,dx, \tag{22}$$

form a system of $n$ second-order differential equations. In order to single out a definite solution of this system, we have to specify $2n$ supplementary conditions, which are usually given in the form of boundary conditions, i.e., relations connecting the values of $y\_i$ and $y\_i'$ at the end points of the interval $[a, b]$ (there are $n$ such relations at each end point). In many cases, the boundary conditions are determined by the very functional under consideration.

For example, consider the variable end point problem for the functional

$$\int_a^b F(x, y_1, \ldots, y_n, y_1', \ldots, y_n')\,dx + g^{(1)}(a, y_1, \ldots, y_n) + g^{(2)}(b, y_1, \ldots, y_n), \tag{23}$$

differing from (22) by two functions $g^{(1)}$ and $g^{(2)}$ of the coordinates of the end points of the path along which the functional is considered. Calculating the variation of the functional (23), we obtain

$$\int_a^b \sum_{i=1}^{n}\!\left(F_{y_i} - \frac{d}{dx}F_{y_i'}\right)h_i\,dx + \sum_{i=1}^{n} F_{y_i'}\,h_i\Big|_{x=a}^{x=b} + \sum_{i=1}^{n} g_{y_i}^{(1)} h_i(a) + \sum_{i=1}^{n} g_{y_i}^{(2)} h_i(b). \tag{24}$$

Setting (24) equal to zero, and assuming that the curve $y\_i = y\_i(x)$, $1 \leqslant i \leqslant n$, is an extremal, we find that

$$\sum_{i=1}^{n} F_{y_i'}\,h_i\Big|_{x=a}^{x=b} + \sum_{i=1}^{n} g_{y_i}^{(1)} h_i(a) + \sum_{i=1}^{n} g_{y_i}^{(2)} h_i(b) = 0. \tag{25}$$

Since $h\_i(a)$ and $h\_i(b)$ are arbitrary, (25) implies that

$$(F_{y_i'} - g_{y_i}^{(1)})\big|_{x=a} = 0 \qquad (i = 1, \ldots, n) \tag{26}$$

and

$$(F_{y_i'} - g_{y_i}^{(2)})\big|_{x=b} = 0 \qquad (i = 1, \ldots, n), \tag{27}$$

i.e., the natural boundary conditions for a variable end point problem like the one considered in Sec. 6. If $g^{(1)} = g^{(2)} \equiv 0$, (25) implies $F\_{y\_i'}\big\|\_{x=a} = F\_{y\_i'}\big\|\_{x=b} = 0$, i.e., the transversality conditions (see Sec. 15).

Next, we examine in more detail the boundary conditions corresponding to one end point, say $x = a$. For simplicity, we write $g$ instead of $g^{(1)}$, and adopt the vector notation $y = (y\_1, \ldots, y\_n)$, $y' = (y\_1', \ldots, y\_n')$. As usual, we introduce the "momenta" (see Sec. 15)

$$p_i(x, y, y') = F_{y_i'}(x, y, y') \qquad (i = 1, \ldots, n), \tag{28}$$

and then write the boundary conditions (26) in the form

$$p_i(x, y, y')\big|_{x=a} = g_{y_i}(x, y)\big|_{x=a} \qquad (i = 1, \ldots, n). \tag{29}$$

The relations (28) determine $y\_1'(a), \ldots, y\_n'(a)$ as functions of $y\_1(a), \ldots, y\_n(a)$:

$$y_i'(a) = \psi_i(y) \qquad (i = 1, \ldots, n). \tag{30}$$

Boundary conditions that can be derived in this way merit a special name:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1</span><span class="math-callout__name">(Self-Adjoint Boundary Conditions)</span></p>

Given a functional $\int\_a^b F(x, y, y')\,dx$, with momenta (28), the boundary conditions (30), prescribed for $x = a$, are said to be **self-adjoint** if there exists a function $g(x, y)$ such that

$$p_i[x, y, \psi(y)]\big|_{x=a} \equiv g_{y_i}(x, y)\big|_{x=a} \qquad (i = 1, \ldots, n). \tag{31}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Characterization of Self-Adjointness)</span></p>

*The boundary conditions (30) are self-adjoint if and only if they satisfy the conditions*

$$\frac{\partial p_i[x, y, \psi(y)]}{\partial y_k}\bigg|_{x=a} = \frac{\partial p_k[x, y, \psi(y)]}{\partial y_i}\bigg|_{x=a} \qquad (i, k = 1, \ldots, n), \tag{32}$$

*called the **self-adjointness conditions**.*

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

If the boundary conditions (30) are self-adjoint, then (31) holds, and hence

$$\frac{\partial p_i[x, y, \psi(y)]}{\partial y_k} = \frac{\partial^2 g(x, y)}{\partial y_i\,\partial y_k} = \frac{\partial p_k[x, y, \psi(y)]}{\partial y_i},$$

which is just (32). Conversely, if the boundary conditions (30) are such that the functions $p\_i[x, y, \psi(y)]$ satisfy (32), then, for $x = a$, the $p\_i$ are the partial derivatives with respect to $y\_i$ of some function $g(y)$, so that the boundary conditions (30) are self-adjoint in the sense of Definition 1.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Self-Adjointness for $n = 1$)</span></p>

It is immediately clear that for $n = 1$, i.e., in the case of variational problems involving a single unknown function, any boundary condition is self-adjoint, and in fact, the self-adjointness conditions (32) disappear for $n = 1$.

</div>

#### 32.2. Field of a Functional

In the preceding section, we introduced the concept of a field for a system of second-order differential equations. We now define the field of a functional:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2</span><span class="math-callout__name">(Field of a Functional)</span></p>

Given a functional

$$\int_a^b F(x, y, y')\,dx, \tag{33}$$

with the system of Euler equations

$$F_{y_i} - \frac{d}{dx}F_{y_i'} = 0 \qquad (i = 1, \ldots, n), \tag{34}$$

we say that boundary conditions

$$y_i' = \psi_i^{(1)}(y) \qquad (i = 1, \ldots, n), \tag{35}$$

prescribed for $x = x\_1$, and the boundary conditions

$$y_i' = \psi_i^{(2)}(y) \qquad (i = 1, \ldots, n), \tag{36}$$

prescribed for $x = x\_2$, are **(mutually) consistent** with respect to the functional (33) if they are consistent with respect to the system (34), i.e., if every extremal satisfying the boundary conditions (35) at $x = x\_1$ also satisfies the boundary conditions (36) at $x = x\_2$, and conversely.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3</span><span class="math-callout__name">(Field of Boundary Conditions)</span></p>

The family of boundary conditions

$$y_i' = \psi_i(x, y) \qquad (i = 1, \ldots, n), \tag{37}$$

prescribed for every $x$ in the interval $[a, b]$, is said to be a **field** of the functional (33) if

1. The conditions (37) are self-adjoint for every $x$ in $[a, b]$;
2. The conditions (37) are consistent for every pair of points $x\_1$, $x\_2$ in $[a, b]$.

</div>

In other words, by a field of the functional (33) is meant a field for the corresponding system of Euler equations (34) which satisfies the self-adjointness conditions at every point $x$. The equations (37) represent a system of first-order differential equations. Its general solution (the family of *trajectories* of the field) is an $n$-parameter family of extremals such that one and only one extremal passes through each point $(x, y\_1, \ldots, y\_n)$ of the region where the field is defined.

We now give an effective criterion for a given family of boundary conditions to be the field of a functional:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2</span><span class="math-callout__name">(Necessary and Sufficient Conditions for a Field)</span></p>

*A necessary and sufficient condition for the family of boundary conditions (37) to be a field of the functional (33) is that the **self-adjointness conditions***

$$\frac{\partial p_i[x, y, \psi(x, y)]}{\partial y_k} = \frac{\partial p_k[x, y, \psi(x, y)]}{\partial y_i} \qquad (i, k = 1, \ldots, n) \tag{38}$$

*and the **consistency conditions***

$$\frac{\partial p_i[x, y, \psi(x, y)]}{\partial x} = -\frac{\partial H[x, y, \psi(x, y)]}{\partial y_i} \qquad (i = 1, \ldots, n) \tag{39}$$

*be satisfied at every point $x$ in $[a, b]$, where*

$$p_i(x, y, y') = F_{y_i'}(x, y, y') \tag{40}$$

*and $H$ is the Hamiltonian corresponding to the functional (33):*

$$H(x, y, y') = -F(x, y, y') + \sum_{i=1}^{n} p_i(x, y, y')\,y_i'. \tag{41}$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

We have already shown in Theorem 1 that the conditions (38) are necessary and sufficient for the boundary conditions (42) given by $y\_i' = \psi\_i(x, y)$ to be self-adjoint at every point $x$ in $[a, b]$. Therefore, it only remains to show that if (38) holds at every point in $[a, b]$, then the conditions (39) are necessary and sufficient for the boundary conditions (42) to be consistent for $a \leqslant x \leqslant b$.

To prove this, we set $y\_i' = \psi\_i(x, y)$, $y'' = \psi(x, y)$ in (40) and (41), and substitute the right-hand sides of the resulting equations into (39). Performing the indicated differentiations and dropping arguments (to keep the notation concise), we obtain

$$F_{y_ix} + \sum_{k=1}^{n} F_{y_iy_k}\frac{\partial \psi_k}{\partial x} = F_{y_i} + \sum_{k=1}^{n} F_{y_k}\frac{\partial \psi_k}{\partial y_i} - \sum_{k=1}^{n}\psi_k\frac{\partial F_{y_k}}{\partial y_i} - \sum_{k=1}^{n} F_{y_k}\frac{\partial \psi_k}{\partial y_i}. \tag{43}$$

Using the self-adjointness conditions $\frac{\partial F\_{y\_i'}}{\partial y\_k} = \frac{\partial F\_{y\_k'}}{\partial y\_i}$, we can write (43) in the form

$$F_{y_i} = F_{y_ix} + \sum_{k=1}^{n} F_{y_iy_k}\frac{\partial \psi_k}{\partial x} + \sum_{k=1}^{n} F_{y_iy_k'}\!\left(\frac{\partial \psi_k}{\partial x} + \sum_{j=1}^{n}\frac{\partial \psi_k}{\partial y_j}\,\psi_j\right). \tag{45}$$

Along the trajectories of the field, we have $\frac{dy\_k}{dx} = \psi\_k$ and $\frac{d^2 y\_k}{dx^2} = \frac{\partial \psi\_k}{\partial x} + \sum\_j \frac{\partial \psi\_k}{\partial y\_j}\psi\_j$. Therefore, (45) reduces to

$$F_{y_i} - \frac{d}{dx}F_{y_i'} = 0, \tag{46}$$

along the trajectories of the field. This means that the trajectories of the field directions (42) are extremals, i.e., (42) is a field of the functional. Since the calculations leading from (39) to (46) are reversible, the conditions (39) are also necessary, and the theorem is proved.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3</span><span class="math-callout__name">(Constancy of Self-Adjointness Along Extremals)</span></p>

The expression

$$\frac{\partial p_i(x, y, y')}{\partial y_k} - \frac{\partial p_k(x, y, y')}{\partial y_i} \tag{48}$$

has a constant value along each extremal.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Using (46), we find that

$$\frac{d}{dx}\!\left(\frac{\partial p_i}{\partial y_k}\right) = \frac{\partial F_{y_i}}{\partial y_k} - \frac{\partial F_{y_k}}{\partial y_i} = 0.$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Extremals as Field Trajectories)</span></p>

Suppose the boundary conditions $y\_i' = \psi\_i(x, y)$ $(a \leqslant x \leqslant b;\; 1 \leqslant i \leqslant n)$ are consistent, i.e., suppose the solutions of the system (49) are extremals of the functional (47). Then, to prove that the conditions (49) define a field of the functional (47), it is only necessary to verify that they are self-adjoint at a single (arbitrary) point in $[a, b]$.

</div>

According to Definition 1, the boundary conditions are self-adjoint if there exists a function $g(x, y)$ such that $p\_i[x, y, \psi(x, y)] = g\_{y\_i}(x, y)$ for $a \leqslant x \leqslant b$. It follows from (50) that the Hamilton-Jacobi equation (51) can be written in the form

$$\frac{\partial g}{\partial x} + H\!\left(x, y_1, \ldots, y_n, \frac{\partial g}{\partial y_1}, \ldots, \frac{\partial g}{\partial y_n}\right) = 0. \tag{51}$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4</span><span class="math-callout__name">(Consistency via the Hamilton-Jacobi Equation)</span></p>

*The boundary conditions (49) defined by the relations (50) are consistent if and only if the function $g(x, y)$ satisfies the **Hamilton-Jacobi equation** (51).*

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

It follows from (50) that the Hamilton-Jacobi equation (51) can be written in the form $\frac{\partial g}{\partial x} = -H(x, y\_1, \ldots, y\_n, p\_1, \ldots, p\_n)$. Differentiating (52) with respect to $y\_i$, we obtain

$$\frac{\partial^2 g}{\partial x\,\partial y_i} = -\frac{\partial H[x, y_1, \ldots, y_n, \psi_1(x, y), \ldots, \psi_n(x, y)]}{\partial y_i},$$

i.e.,

$$\frac{\partial p_i}{\partial x} = -\frac{\partial H[x, y, \psi(x, y)]}{\partial y_i},$$

which is just the set of consistency conditions (39).

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection between the Two Hamilton-Jacobi Systems)</span></p>

The connection between the Hamilton-Jacobi system introduced in Sec. 31 and the Hamilton-Jacobi equation introduced in Sec. 23 is now apparent. As we saw in Sec. 31, in the case of an arbitrary system of $n$ second-order differential equations, a field is a system of $n$ first-order differential equations of the form (49), where the functions $\psi\_i(x, y)$ satisfy the Hamilton-Jacobi system (8). When we deal with the field of a functional, and in this case, we impose the additional requirement that the boundary conditions defining the field be self-adjoint at every point. This means that the field of a functional is not really determined by $n$ functions $\psi\_i(x, y)$, but rather by a single function $g(x, y)$ from which the functions $\psi\_i(x, y)$ are derived by using the relations (50). In other words, the function $g(x, y)$ is a kind of *potential* for the field of a functional. Since the field of a functional is determined by a single function, instead of $n$ consistency conditions, the set of $n$ consistency conditions for such a field should reduce to a single equation, i.e., that the Hamilton-Jacobi system should be replaced by the Hamilton-Jacobi equation.

</div>

#### 32.3. Central Fields

Once more, we consider a functional

$$\int_a^b F(x, y, y')\,dx, \tag{53}$$

whose extremals are curves in the $(n + 1)$-dimensional space of points $(x, y) = (x, y\_1, \ldots, y\_n)$. Let $R$ be a simply connected region in this space, and let $c = (c\_0, c\_1, \ldots, c\_n)$ be a point lying outside $R$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4</span><span class="math-callout__name">(Central Field)</span></p>

Let $(x, y)$ be an arbitrary point of $R$, and suppose that one and only one extremal of the functional (53) leaves $c$ and passes through $(x, y)$, thereby defining a direction

$$y_i' = \psi_i(x, y) \qquad (i = 1, \ldots, n) \tag{54}$$

at every point of $R$. Then the field of directions (54) is called a **central field**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5</span><span class="math-callout__name">(Every Central Field is a Field)</span></p>

*Every central field (54) is a field of the functional (53), i.e., satisfies the consistency and self-adjointness conditions.*

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Consider the function

$$g(x, y) = \int_c^{(x,y)} F(x, y, y')\,dx, \tag{55}$$

where the integral is taken along the extremal of (53) joining the point $c$ to the point $(x, y)$. We define a field of directions in $R$ by setting

$$F_{y_i'}(x, y, y') \equiv p_i(x, y, y') = g_{y_i}(x, y) \qquad (i = 1, \ldots, n). \tag{56}$$

The theorem will be proved if it can be shown that this field coincides with the original field (54), since then the original field will satisfy the self-adjointness conditions [since its trajectories are extremals] and also the consistency conditions [this follows from Theorem 1 applied to the field defined by (56)]. But (55) is just the function $S(x, y\_1, \ldots, y\_n)$ of Sec. 23, and hence $g\_{y\_i}(x, y) = p\_i(x, y, z)$, where $z$ denotes the slope of the extremal joining $c$ to $(x, y)$, evaluated at $(x, y)$. This shows that the field of directions (56) actually coincides with the original field (54). The proof is now completed by using Theorem 5.

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5</span><span class="math-callout__name">(Imbedding in a Field)</span></p>

Given an extremal $\gamma$ of the functional (53), suppose there exists a simply connected (open) region $R$ containing $\gamma$ such that

1. A field of the functional (53) covers $R$, i.e., is defined at every point of $R$;
2. One of the trajectories of the field is $\gamma$.

Then we say that $\gamma$ **can be imbedded in a field** (of the functional (53)).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6</span><span class="math-callout__name">(Imbedding an Extremal in a Field)</span></p>

*Let $\gamma$ be an extremal of the functional (53), with equation $y = y(x)$ $(a \leqslant x \leqslant b)$, in vector form. Moreover, suppose that $\det\lVert F\_{y\_i'y\_k'}\rVert$ is nonvanishing in $[a, b]$, and that no points conjugate to $(a, y(a))$ lie on $\gamma$. Then $\gamma$ can be imbedded in a field.*

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By hypothesis, the following two conditions are satisfied for sufficiently small $\varepsilon > 0$:

1. The extremal $\gamma$ can be extended onto the whole interval $[a - \varepsilon, b]$;
2. The interval $[a - \varepsilon, b]$ contains no points conjugate to $a$.

Now consider the family of extremals leaving the point $(a - \varepsilon, y(a - \varepsilon))$. Since there are no points conjugate to $a - \varepsilon$ in the interval $[a - \varepsilon, b]$, it follows that for $a \leqslant x \leqslant b$ no two extremals in this family which are sufficiently close to the original extremal $\gamma$ can intersect. Thus, in some region $R$ containing $\gamma$, the extremals sufficiently close to $\gamma$ define a central field in which $\gamma$ is imbedded. The proof is now completed by using Theorem 5.

</details>
</div>

### 33. Hilbert's Invariant Integral

As before, let $R$ be a simply connected region in the $(n + 1)$-dimensional space of points $(x, y) = (x, y\_1, \ldots, y\_n)$, and let

$$y_i' = \psi_i(x, y) \qquad (i = 1, \ldots, n) \tag{57}$$

define a field of the functional

$$\int_a^b F(x, y, y')\,dx \tag{58}$$

in $R$. It was proved in the preceding section (see Theorem 2) that the field of directions (57) is a field of the functional (58) if and only if the functions $\psi\_i(x, y)$ satisfy the self-adjointness conditions

$$\frac{\partial p_i[x, y, \psi(x, y)]}{\partial y_k} = \frac{\partial p_k[x, y, \psi(x, y)]}{\partial y_i} \tag{59}$$

and the consistency conditions

$$\frac{\partial H[x, y, \psi(x, y)]}{\partial y_i} = -\frac{\partial p_i[x, y, \psi(x, y)]}{\partial x}. \tag{60}$$

Taken together, the conditions (59) and (60) imply that the quantity

$$-H[x, y, \psi(x, y)]\,dx + \sum_{i=1}^{n} p_i[x, y, \psi(x, y)]\,dy_i$$

is the exact differential of some function $g(x, y) = g(x, y\_1, \ldots, y\_n)$. As is familiar from elementary analysis, this function, which is determined to within an additive constant, can be written as a line integral

$$g(x, y) = \int_\Gamma \!\left(-H\,dx + \sum_{i=1}^{n} p_i\,dy_i\right), \tag{61}$$

evaluated along the curve $\Gamma$ going from some fixed point $M\_0 = (x\_0, y(x\_0))$ to the variable point $M = (x, y)$. Since the integrand of (61) is an exact differential, the choice of the curve $\Gamma$ does not matter; in fact, the value of the integral depends only on the points $M\_0$, $M\_1$, and not on the curve $\Gamma$. The right-hand side of (61) is known as **Hilbert's invariant integral**.

Using the equations (57) defining the field, and explicitly introducing the integrand $F$ of the functional (58), we can write the integral in (61) as

$$\int_\Gamma \!\left(\left\lbrace F[x, y, \psi(x, y)] - \sum_{i=1}^{n} \psi_i(x, y)\,F_{y_i'}[x, y, \psi(x, y)]\right\rbrace dx + \sum_{i=1}^{n} F_{y_i'}[x, y, \psi(x, y)]\,dy_i\right). \tag{62}$$

This expression is Hilbert's invariant integral, in the form corresponding to the field defined by the functions $\psi\_i(x, y)$. If the curve $\Gamma$ along which the integral (62) is evaluated is one of the trajectories of the field, then $dy\_i = \psi\_i(x, y)\,dx$ along $\Gamma$, and hence (62) reduces to

$$\int_\Gamma F(x, y, y')\,dx$$

evaluated along this trajectory.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hilbert's Integral Along Extremals)</span></p>

If $\gamma$ is an extremal which is a trajectory of the field, Hilbert's invariant integral can be used to write the value of the functional for this extremal as an integral evaluated along *any* curve joining the end points of $\gamma$. This important fact will be used in the next section.

</div>

### 34. The Weierstrass $E$-Function. Sufficient Conditions for a Strong Extremum

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Weierstrass $E$-Function)</span></p>

By the **Weierstrass $E$-function** of the functional

$$J[y] = \int_a^b F(x, y, y')\,dx, \qquad y(a) = A, \quad y(b) = B \tag{63}$$

we mean the following function of $3n + 1$ variables:

$$E(x, y, z, w) = F(x, y, w) - F(x, y, z) - \sum_{i=1}^{n}(w_i - z_i)\,F_{y_i'}(x, y, z). \tag{64}$$

In other words, $E(x, y, z, w)$ is the difference between the value of the function $F$ (regarded as a function of its last $n$ arguments) at the point $w$ and the first two terms of its Taylor's series expansion about the point $z$. Thus, $E(x, y, z, w)$ can also be written as the remainder of a Taylor's series:

$$E(x, y, z, w) = \frac{1}{2}\sum_{i,\,k=1}^{n}(w_i - z_i)(w_k - z_k)\,F_{y_i'y_k'}[x, y, z + \theta(w - z)]$$

$$(0 < \theta < 1).$$

</div>

For $n = 1$, the Weierstrass $E$-function has a simple geometric interpretation, since if we regard $F(x, y, z)$ as a function of $z$, then $F(x, y, w) - F(x, y, z) - (w - z)F\_{y'}(x, y, z)$ is just the vertical distance from the curve $\Gamma$ representing $F(x, y, z)$ to the tangent to $\Gamma$ drawn through a fixed point of $\Gamma$.

Our goal in this section is to derive sufficient conditions for the functional (63) to have a strong extremum. It will be recalled from Secs. 28 and 29 that the following set of conditions is sufficient for the functional (63) to have a weak minimum for the admissible curve $\gamma$:

**Condition 1.** The curve $\gamma$ is an extremal;

**Condition 2.** The matrix $\lVert F\_{y\_i'y\_k'}\rVert$ is positive definite along $\gamma$;

**Condition 3.** The interval $[a, b]$ contains no points conjugate to $a$.

Every strong extremum is simultaneously a weak extremum, but the converse is in general false (see p. 13). Therefore, in looking for sufficient conditions for a strong extremum, it is natural to assume from the outset that the three conditions just listed are satisfied. We then try to supplement them in such a way as to obtain a set of conditions guaranteeing a strong extremum as well as a weak extremum. To find such supplementary conditions, we first recall that Conditions 2 and 3 imply that the given extremal $\gamma$ can be imbedded in a field

$$y_i' = \psi_i(x, y) \qquad (i = 1, \ldots, n) \tag{65}$$

of the functional (63) (see Theorem 6 of Sec. 32). Let $\gamma$ have the equations $y\_i = y\_i(x)$ $(i = 1, \ldots, n)$, and let $\gamma^\ast$ be an arbitrary curve with the same end points as $\gamma$, lying in the $(n + 1)$-dimensional region $R$ containing $\gamma$ and covered by the field (65). Then, according to the remark at the end of Sec. 33, we have

$$\int_\gamma F(x, y, y')\,dx = \int_{\gamma^*}\!\left(\left\lbrace F[x, y, \psi] - \sum_{i=1}^{n}\psi_i\,F_{y_i'}[x, y, \psi]\right\rbrace dx + \sum_{i=1}^{n} F_{y_i'}[x, y, \psi]\,dy_i\right), \tag{66}$$

where for simplicity we omit the arguments of the functions $\psi$ and $\psi\_i$. The right-hand side of (66) is just Hilbert's invariant integral, in the form corresponding to the field (65). As usual, we are interested in the increment

$$\Delta J = \int_{\gamma^*} F(x, y, y')\,dx - \int_\gamma F(x, y, y')\,dx.$$

Using (66), we find that

$$\Delta J = \int_{\gamma^*}\!\left(F(x, y, y') - F(x, y, \psi) - \sum_{i=1}^{n}(y_i' - \psi_i)\,F_{y_i'}(x, y, \psi)\right) dx,$$

or in terms of the Weierstrass $E$-function,

$$\Delta J = \int_{\gamma^*} E(x, y, \psi, y')\,dx. \tag{67}$$

We are now in a position to state sufficient conditions for a strong extremum:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Sufficient Conditions for a Strong Minimum)</span></p>

*Let $\gamma$ be an extremal, and let*

$$y_i' = \psi_i(x, y) \qquad (i = 1, \ldots, n) \tag{68}$$

*be a field of the functional*

$$J[y] = \int_a^b F(x, y, y')\,dx, \qquad y(a) = A, \quad y(b) = B. \tag{69}$$

*Suppose that at every point $(x, y) = (x, y\_1, \ldots, y\_n)$ of some (open) region containing $\gamma$ and covered by the field (68), the condition*

$$E(x, y, \psi, w) \geqslant 0 \tag{70}$$

*is satisfied for every finite vector $w = (w\_1, \ldots, w\_n)$. Then $J[y]$ has a **strong minimum** for the extremal $\gamma$.*

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

To say that the functional $J[y]$ has a strong minimum for the extremal $\gamma$ means that $\Delta J$ is nonnegative for any admissible curve $\gamma^\ast$ which is sufficiently close to $\gamma$ in the norm of the space $\mathscr{C}\_1(a, b)$. But the condition (70) guarantees that the increment $\Delta J$, given by (67), is nonnegative for all such curves. Note that we do not impose any restrictions at all on the slope of the curve $\gamma^\ast$, i.e., $\gamma^\ast$ need not be close to $\gamma$ in the norm of the space $\mathscr{D}\_1(a, b)$. In fact, $\gamma^\ast$ need not even belong to $\mathscr{D}\_1(a, b)$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1</span><span class="math-callout__name">(Imbedding Replaces Conditions 2 and 3)</span></p>

As already noted, the hypothesis that the extremal $\gamma$ can be imbedded in a field can be replaced by Conditions 2 and 3.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2</span><span class="math-callout__name">(Weierstrass $E$-Function and Nonnegative Definiteness)</span></p>

Since the Weierstrass $E$-function can be written in the form

$$E(x, y, \psi, w) = \frac{1}{2}\sum_{i,\,k=1}^{n}(w_i - \psi_i)(w_k - \psi_k)\,F_{y_i'y_k'}[x, y, \psi + \theta(w - \psi)]$$

$$(0 < \theta < 1),$$

we can replace (70) by the condition that at every point of some region containing $\gamma$, the matrix $\lVert F\_{y\_i'y\_k'}(x, y, z)\rVert$ be nonnegative definite for every finite $z$.

</div>

We conclude this section by indicating the following *necessary* condition for a strong extremum:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2</span><span class="math-callout__name">(Weierstrass' Necessary Condition)</span></p>

*If the functional $J[y] = \int\_a^b F(x, y, y')\,dx$, $y(a) = A$, $y(b) = B$ has a strong minimum for the extremal $\gamma$, then*

$$E(x, y, y', w) \geqslant 0 \tag{71}$$

*along $\gamma$ for every finite $w$.*

</div>

The idea of the proof is the following: If (71) is not satisfied, there exists a point $\xi$ in $[a, b]$ and a vector $q$ such that $E[\xi, y(\xi), y'(\xi), q] < 0$, where $y = y(x)$ is the equation of the extremal $\gamma$. It can then be shown that a suitable modification of $\gamma$ leads to an admissible curve $\gamma^\ast$ close to $\gamma$ in the norm of the space $\mathscr{C}(a, b)$ such that $\Delta J = \int\_{\gamma^\ast} F(x, y, y')\,dx - \int\_\gamma F(x, y, y')\,dx < 0$, which contradicts the hypothesis that $J[y]$ has a strong minimum for $\gamma$. However, the construction of $\gamma^\ast$ must be carried out carefully, since all we know is that (72) holds for a suitable $q$.

## Chapter 7: Variational Problems Involving Multiple Integrals

In this chapter, we discuss topics pertaining to functionals which depend on functions of two or more variables. Such functionals arise, for example, in mechanical problems involving systems with infinitely many degrees of freedom (strings, membranes, etc.). In our treatment of systems consisting of a finite number of particles (see Chapter 4), we derived the principle of least action and a general method for obtaining conservation laws (Noether's theorem). These methods will now be applied to systems with infinitely many degrees of freedom.

### 35. Variation of a Functional Defined on a Fixed Region

Consider the functional

$$J[u] = \int \cdots \int_R F(x_1, \ldots, x_n, u, u_{x_1}, \ldots, u_{x_n})\,dx_1 \cdots dx_n,$$

depending on $n$ independent variables $x\_1, \ldots, x\_n$, an unknown function $u$ of these variables, and the partial derivatives $u\_{x\_1}, \ldots, u\_{x\_n}$ of $u$. As usual, it is assumed that the integrand $F$ has continuous first and second derivatives with respect to all its arguments.

We calculate the variation of $J[u]$, assuming the region $R$ stays fixed, while the function $u(x\_1, \ldots, x\_n)$ goes into

$$u^*(x_1, \ldots, x_n) = u(x_1, \ldots, x_n) + \varepsilon\psi(x_1, \ldots, x_n) + \cdots$$

For simplicity, we write $u(x)$, $\psi(x)$ instead of $u(x\_1, \ldots, x\_n)$, $\psi(x\_1, \ldots, x\_n)$, $dx$ instead of $dx\_1 \cdots dx\_n$, etc. Then, using Taylor's theorem, we find that

$$J[u^*] - J[u] = \varepsilon \int_R \left(F_u + \sum_{i=1}^{n} F_{u_{x_i}} \psi_{x_i}\right) dx + \cdots,$$

where the dots denote terms of order higher than 1 relative to $\varepsilon$. It follows that the **variation** of the functional is

$$\delta J = \varepsilon \int_R \left(F_u + \sum_{i=1}^{n} F_{u_{x_i}} \psi_{x_i}\right) dx.$$

Next, we try to represent the variation as an integral of the form $G(x)\psi(x) + \operatorname{div}(\cdots)$, so that the derivatives $\psi\_{x\_i}$ only appear in a divergence. Replacing $F\_{u\_{x\_i}} \psi\_{x\_i}$ by

$$\frac{\partial}{\partial x_i}[F_{u_{x_i}} \psi(x)] - \frac{\partial F_{u_{x_i}}}{\partial x_i}\,\psi(x)$$

we obtain

$$\delta J = \varepsilon \int_R \left(F_u - \sum_{i=1}^{n} \frac{\partial}{\partial x_i} F_{u_{x_i}}\right)\psi(x)\,dx + \varepsilon \int_R \sum_{i=1}^{n} \frac{\partial}{\partial x_i}[F_{u_{x_i}} \psi(x)]\,dx.$$

The second term is the integral of a divergence, and by the $n$-dimensional version of Green's theorem can be reduced to an integral over the boundary $\Gamma$ of the region $R$:

$$\int_R \sum_{i=1}^{n} \frac{\partial}{\partial x_i}[F_{u_{x_i}} \psi(x)]\,dx = \int_\Gamma \psi(x)(G, \nu)\,d\sigma,$$

where $G = (F\_{u\_{x\_1}}, \ldots, F\_{u\_{x\_n}})$, $\nu = (\nu\_1, \ldots, \nu\_n)$ is the unit outward normal to $\Gamma$, and $(G, \nu)$ denotes the scalar product. Using this, we can write the variation as

$$\delta J = \varepsilon \int_R \left(F_u - \sum_{i=1}^{n} \frac{\partial}{\partial x_i} F_{u_{x_i}}\right)\psi(x)\,dx + \varepsilon \int_\Gamma \psi(x)(G, \nu)\,d\sigma.$$

For the functional to have an extremum, $\delta J = 0$ for all admissible $\psi(x)$, in particular for all $\psi(x)$ which vanish on the boundary $\Gamma$. For such functions, the boundary integral vanishes and we obtain

$$\delta J = \int_R \left(F_u - \sum_{i=1}^{n} \frac{\partial}{\partial x_i} F_{u_{x_i}}\right)\psi(x)\,dx,$$

and by the arbitrariness of $\psi(x)$ inside $R$, $\delta J = 0$ implies

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Euler Equation for Multiple Integrals)</span></p>

*A necessary condition for the functional $J[u] = \int \cdots \int\_R F(x\_1, \ldots, x\_n, u, u\_{x\_1}, \ldots, u\_{x\_n})\,dx\_1 \cdots dx\_n$ to have an extremum is that $u$ satisfies the Euler equation*

$$F_u - \sum_{i=1}^{n} \frac{\partial}{\partial x_i} F_{u_{x_i}} = 0$$

*for all $x \in R$. This is the $n$-dimensional generalization of the Euler equation from Sec. 5.*

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

In deriving the Euler equation, we assumed that the region of integration $R$ appearing in the functional is fixed. Generalization to the case where the region of integration is variable will be made in Sec. 37.

</div>

### 36. Variational Derivation of the Equations of Motion of Continuous Mechanical Systems

As we saw in Sec. 21, the equations of motion of a mechanical system consisting of $n$ particles can be derived from the *principle of least action*, which states that the actual trajectory of the system in phase space minimizes the *action functional*

$$\int_{t_0}^{t_1} (T - U)\,dt,$$

where $T$ is the kinetic energy and $U$ the potential energy of the system of particles. We now use this principle, together with the basic formula for the first variation, to derive the equations of motion and the appropriate boundary conditions for some simple mechanical systems with infinitely many degrees of freedom, namely the vibrating string, membrane, and plate.

#### 36.1. The Vibrating String

Consider the transverse motion of a *string* (i.e., a homogeneous flexible cord) of length $l$ and linear mass density $\rho$. Suppose the ends of the string (at $x = 0$ and $x = l$) are *fastened elastically*, which means that if either end is displaced from its equilibrium position, a restoring force proportional to the displacement appears.

Let the equilibrium position of the string lie along the $x$-axis, and let $u(x, t)$ denote the displacement at the point $x$ and time $t$ from its equilibrium position. Then, at time $t$, the kinetic energy of the element of string which initially lies between $x\_0$ and $x\_0 + \Delta x$ is clearly

$$\tfrac{1}{2}\,\rho\, u_t^2(x_0, t)\,\Delta x.$$

Integrating from $0$ to $l$, we find that the kinetic energy of the whole string at time $t$ equals

$$T = \frac{1}{2}\,\rho \int_0^l u_t^2(x, t)\,dx.$$

To find the potential energy of the string, we use the following argument: The potential energy of the string in the position described by the function $u(x, t)$, where $t$ is fixed, is just the work required to move the string from its equilibrium position $u \equiv 0$ into the given position $u(x, t)$. Let $\tau$ denote the tension in the spring, and consider the element of string indicated by $AB$. The work needed to deform $\Delta A$ equals the product of $\tau$ and the increase in length of the element of string, i.e., the quantity

$$\tau\sqrt{(\Delta x)^2 + (\Delta u)^2} - \tau\,\Delta x = \frac{1}{2}\,\tau\!\left(\frac{\Delta u}{\Delta x}\right)^2 \Delta x + \cdots = \frac{1}{2}\,\tau\,u_x^2(x_0, t)\,\Delta x + \cdots,$$

where the dots indicate terms of order higher than those written ($\Delta u / \Delta x \ll 1$ for all $t$, since the vibrations are small).

Integrating from $0$ to $l$, we find that the potential energy of the whole string is

$$U_1 = \frac{1}{2}\,\tau \int_0^l u_x^2(x, t)\,dx,$$

except for the work expended in displacing the elastically fastened ends of the string from their equilibrium positions. This work equals

$$U_2 = \frac{1}{2}\,\varkappa_1 u^2(0, t) + \frac{1}{2}\,\varkappa_2 u^2(l, t),$$

where $\varkappa\_1$ and $\varkappa\_2$ are positive constants (the *elastic moduli* of the springs). In fact, the force $f\_1$ acting on the end point $P\_1$ is proportional to the displacement $\xi$ of $P\_1$ from its equilibrium position $x = 0$, $u = 0$, i.e., $\lvert f\_1 \rvert = \varkappa\_1 \xi$. Integration of this shows that the work required to move $P\_1$ from $(0, 0)$ to $(0, u(0, t))$ is $\frac{1}{2}\,\varkappa\_1 u^2(0, t)$, and similarly for $P\_2$.

Then, adding, the total potential energy of the string in the position described by $u(x, t)$ is

$$U = U_1 + U_2 = \frac{1}{2}\,\tau \int_0^l u_x^2(x, t)\,dx + \frac{1}{2}\,\varkappa_1 u^2(0, t) + \frac{1}{2}\,\varkappa_2 u^2(l, t).$$

Finally, using the expressions for $T$ and $U$, we write the action functional for the vibrating string:

$$J[u] = \frac{1}{2}\int_{t_0}^{t_1}\!\int_0^l [\rho\, u_t^2(x, t) - \tau\, u_x^2(x, t)]\,dx\,dt - \frac{1}{2}\,\varkappa_1 \int_{t_0}^{t_1} u^2(0, t)\,dt - \frac{1}{2}\,\varkappa_2 \int_{t_0}^{t_1} u^2(l, t)\,dt.$$

According to the principle of least action, $\delta J$ must vanish for the function $u(x, t)$ which describes the actual motion of the string. Suppose we go from the function $u(x, t)$ to the "varied" function $u^\ast(x, t) = u(x, t) + \varepsilon\psi(x, t) + \cdots$ Then, using formula (4) of Sec. 35 and the fact that the variation of a sum equals the sum of the variations of the separate terms, we find that

$$\delta J = \varepsilon\!\left\lbrace \int_{t_0}^{t_1}\!\int_0^l [-\rho\, u_{tt}(x, t) + \tau\, u_{xx}(x, t)]\,\psi(x, t)\,dx\,dt - \varkappa_1 \int_{t_0}^{t_1} u(0, t)\,\psi(0, t)\,dt - \varkappa_2 \int_{t_0}^{t_1} u(l, t)\,\psi(l, t)\,dt \right\rbrace$$

$${}+ \varepsilon\int_{t_0}^{t_1}\!\int_0^l \frac{\partial}{\partial x}[-\tau\, u_x(x, t)\,\psi(x, t)]\,dx\,dt + \varepsilon\int_{t_0}^{t_1}\!\int_0^l \frac{\partial}{\partial t}[\rho\, u_t(x, t)\,\psi(x, t)]\,dx\,dt.$$

If we assume that the admissible functions $\psi(x, t)$ are such that

$$\psi(x, t_0) = \psi(x, t_1) = 0 \quad (0 \leqslant x \leqslant l),$$

i.e., that $u(x, t)$ is not varied at the initial and final times, then the last term in the variation vanishes, and the next to last term reduces to

$$\varepsilon \int_{t_0}^{t_1}[\tau u_x(0, t)\psi(0, t) - \tau u_x(l, t)\psi(l, t)]\,dt.$$

It follows that the variation can be written in the form

$$\delta J = \varepsilon\!\left\lbrace \int_{t_0}^{t_1}\!\int_0^l [-\rho\, u_{tt} + \tau\, u_{xx}]\,\psi\,dx\,dt - \int_{t_0}^{t_1}[\varkappa_1 u(0, t) - \tau u_x(0, t)]\psi(0, t)\,dt + \int_{t_0}^{t_1}[\varkappa_2 u(l, t) + \tau u_x(l, t)]\psi(l, t)\,dt\right\rbrace.$$

Suppose first that $\psi(x, t)$ vanishes at the ends of the string, i.e., that $\psi(0, t) = 0$, $\psi(l, t) = 0$ for $t\_0 \leqslant t \leqslant t\_1$. Then $\delta J$ reduces to just the double integral. Setting it equal to zero, and using the arbitrariness of $\psi(x, t)$ for $0 < x < l$, $t\_0 < t < t\_1$, we find that

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Equation</span><span class="math-callout__name">(Vibrating String)</span></p>

$$u_{tt} = a^2\, u_{xx} \qquad \left(a^2 = \frac{\tau}{\rho}\right)$$

for $0 \leqslant x \leqslant l$ and all $t$. This is the **equation of the vibrating string**, i.e., the Euler equation of the action functional.

</div>

Next, we remove the restriction that $\psi$ vanishes at the ends. Since $u(x, t)$ must satisfy the string equation, the first term in $\delta J$ vanishes, and we have

$$\delta J = -\varepsilon\!\left\lbrace \int_{t_0}^{t_1}[\varkappa_1 u(0, t) - \tau u_x(0, t)]\,\psi(0, t)\,dt + \int_{t_0}^{t_1}[\varkappa_2 u(l, t) + \tau u_x(l, t)]\,\psi(l, t)\,dt \right\rbrace.$$

This expression must also vanish. Since $[t\_0, t\_1]$ is arbitrary and $\psi(0, t)$, $\psi(l, t)$ are arbitrary admissible functions, equating to zero leads to the boundary conditions:

$$\varkappa_1 u(0, t) - \tau u_x(0, t) = 0 \qquad \left(\alpha = -\frac{\varkappa_1}{\tau}\right)$$

and

$$\varkappa_2 u(l, t) + \tau u_x(l, t) = 0 \qquad \left(\beta = \frac{\varkappa_2}{\tau}\right),$$

which connect the displacement from equilibrium and the direction of the tangent at each end of the string.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Special Boundary Conditions)</span></p>

**Free ends:** If the ends of the string are free (springs absent, $\varkappa\_1 = \varkappa\_2 = 0$), the boundary conditions become $u\_x(0, t) = 0$, $u\_x(l, t) = 0$. Thus, at a free end point, the tangent to the string always preserves the same slope (zero) as it had in the equilibrium position.

**Fixed ends:** If the ends of the string are fixed, corresponding to $\varkappa\_1 \to \infty$, $\varkappa\_2 \to \infty$, the boundary conditions become $u(0, t) = 0$, $u(l, t) = 0$. This can be regarded as a limit of the elastically fastened case.

</div>

#### 36.2. Least Action vs. Stationary Action

The principle of least action is widely used not only in mechanics, but also in other branches of physics, e.g., in electrodynamics and field theory. However, as already noted (see Remark 2, p. 85), in a certain sense the principle is not quite true. For example, consider a *simple harmonic oscillator*, i.e., a particle of mass $m$ oscillating about an equilibrium position under the action of an elastic restoring force. The equation of motion of the particle is

$$m\ddot{x} + \varkappa x = 0,$$

with solution

$$x = C \sin(\omega t + \theta),$$

where $\omega = \sqrt{\varkappa/m}$ and the values of the constants $C$, $\theta$ are determined from the initial conditions. Moreover, the particle has kinetic energy $T = \frac{1}{2}m\dot{x}^2$ and potential energy $U = \frac{1}{2}\varkappa x^2$, so that the action is

$$\frac{1}{2}\int_{t_0}^{t_1}(m\dot{x}^2 - \varkappa x^2)\,dt.$$

The Euler equation for this functional gives $m\ddot{x} + \varkappa x = 0$, but in general we cannot assert that its solution actually minimizes the action. In fact, consider the solution

$$x = \frac{1}{\omega}\sin \omega t,$$

which passes through the point $x = 0$, $t = 0$ and satisfies $\dot{x}(0) = 1$. The point $(\pi/\omega, 0)$ is conjugate to the point $(0, 0)$, since every extremal satisfying $x(0) = 0$ intersects the extremal at $(\pi/\omega, 0)$. Since $F\_{\dot{x}\dot{x}} = m > 0$, the extremal satisfies the sufficient conditions for a minimum, *provided that* $0 \leqslant t \leqslant t\_0 < \pi/\omega$. However, if we consider time intervals greater than $\pi/\omega$, we can no longer guarantee that the extremal minimizes the action functional.

Next, consider a system of $n$ coupled oscillators, with kinetic energy

$$T = \sum_{i,k=1}^{n} a_{ik}\dot{x}_i\dot{x}_k$$

(a quadratic form in the velocities $\dot{x}\_i$) and potential energy

$$U = \sum_{i,k=1}^{n} b_{ik} x_i x_k$$

(a quadratic form in the coordinates $x\_i$). The quadratic form for $T$ is positive definite (since it is a kinetic energy); therefore, both forms can be simultaneously reduced to sums of squares by a suitable linear transformation $x\_i = \sum\_{k=1}^{n} c\_{ik} q\_k$ ($i = 1, \ldots, n$), i.e., substitution gives

$$T = \sum_{i=1}^{n} \dot{q}_i^2, \qquad U = \sum_{i=1}^{n} \lambda_i q_i^2.$$

Then the equations of motion of the system of oscillators are given by the Euler equations

$$\ddot{q}_i + \lambda_i q_i = 0 \qquad (i = 1, \ldots, n),$$

corresponding to the action functional $\int\_{t\_0}^{t\_1} \sum\_{i=1}^{n}(\dot{q}\_i^2 - \lambda\_i q\_i^2)\,dt$.

Suppose all the $\lambda\_i$ are positive, which means that we are considering oscillations about a position of stable equilibrium. Then the solution has the form

$$q_i = C_i \sin \omega_i(t + \theta_i) \qquad (i = 1, \ldots, n),$$

where $\omega\_i = \sqrt{\lambda\_i}$, and the values of $C\_i$, $\theta\_i$ are determined from the initial conditions. An argument like that made for the simple harmonic oscillator ($n = 1$) shows that a trajectory of the system [i.e., a curve given by the $q\_i(t)$ in a space of $n + 1$ dimensions] whose projection on the time axis is of length no greater than $\pi/\omega$, where

$$\omega = \max_{1 \leqslant i \leqslant n} \omega_i,$$

contains no conjugate points and satisfies the sufficient conditions for a minimum. However, just as before, we cannot guarantee that a trajectory whose projection on the time axis is of length greater than $\pi/\omega$ actually minimizes the action.

Finally, consider a vibrating string of length $l$ with fixed ends. As shown above, the function $u(x, t)$ describing the oscillations of the string satisfies

$$u_{tt} = a^2 u_{xx}$$

and the boundary conditions $u(0, t) = 0$, $u(l, t) = 0$. It follows that

$$u(x, t) = \sum_{k=1}^{\infty} C_k(x) \sin \omega_k(t + \theta_k),$$

where $\omega\_k = k a\pi / l$. Thus, in a certain sense, a vibrating string can be regarded as a system of infinitely many coupled oscillators, with natural frequencies $\omega\_k$. However, the numbers $\omega\_k$ have no finite upper bound, and hence the analogy with the case of $n$ coupled oscillators leads us to believe that for a vibrating string, there is no time interval short enough to guarantee that $u(x, t)$ actually minimizes the action functional.

Guided by the above considerations, we shall henceforth replace the principle of *least* action by the principle of *stationary* action. In other words, the actual trajectory of a given mechanical system will not be required to minimize the action but only to cause its first variation to vanish.

#### 36.3. The Vibrating Membrane

Consider the transverse motion of a *membrane* (i.e., a homogeneous flexible sheet) of surface mass density $\rho$. Let $u(x, y, t)$ denote the displacement from equilibrium of the point $(x, y)$ of the membrane, at time $t$. The kinetic energy of the membrane at time $t$ is given by

$$T = \frac{1}{2}\,\rho \iint_R u_t^2(x, y, t)\,dx\,dy,$$

where $R$ is the region of the $xy$-plane occupied by the membrane at rest.

The potential energy of the membrane in the position described by the function $u(x, y, t)$, where $t$ is fixed, is just the work required to move the membrane from its equilibrium position $u \equiv 0$ into the given position $u(x, y, t)$. This work is the sum of the work $U\_1$ expended in deforming the membrane and the work $U\_2$ expended in moving the boundary of the membrane, which we assume to be elastically fastened to its equilibrium position.

To calculate $U\_1$, let $\tau$ denote the tension in the membrane, and consider the element $\Delta A$ of the membrane initially occupying the region $x\_0 \leqslant x \leqslant x\_0 + \Delta x$, $y\_0 \leqslant y \leqslant y\_0 + \Delta y$. The work needed to deform $\Delta A$ equals the product of $\tau$ and the increase in area of $\Delta A$ under deformation, i.e.,

$$\tau\sqrt{(\Delta x)^2 + (\Delta u)^2}\sqrt{(\Delta y)^2 + (\Delta u)^2} - \tau\,\Delta x\,\Delta y = \frac{1}{2}\,\tau[u_x^2(x_0, y_0, t) + u_y^2(x_0, y_0, t)]\,\Delta x\,\Delta y + \cdots$$

Integrating over $R$, we find that the work required to deform the whole membrane is

$$U_1 = \frac{1}{2}\,\tau \iint_R [u_x^2(x, y, t) + u_y^2(x, y, t)]\,dx\,dy.$$

To calculate $U\_2$, if $\Gamma$ is the boundary of the region $R$, and $s$ is arc length measured along $\Gamma$ from some fixed point on $\Gamma$, then

$$U_2 = \frac{1}{2}\int_\Gamma \varkappa(s)\,u^2(s, t)\,ds,$$

where $u(s, t)$ is the displacement of the membrane from equilibrium at the point $s$ and time $t$, and $\varkappa(s)$ is the linear density of the elastic modulus of the forces retaining the boundary of the membrane.

Combining $T$, $U\_1$, and $U\_2$, we find that the action functional for the vibrating membrane is

$$J[u] = \frac{1}{2}\int_{t_0}^{t_1}\!\iint_R \lbrace\rho\, u_t^2(x, y, t) - \tau[u_x^2(x, y, t) + u_y^2(x, y, t)]\rbrace\,dx\,dy\,dt - \frac{1}{2}\int_{t_0}^{t_1}\!\int_\Gamma \varkappa(s)\,u^2(s, t)\,ds\,dt.$$

Suppose we go from the function $u(x, y, t)$ to the "varied" function $u^\ast(x, y, t) = u(x, y, t) + \varepsilon\psi(x, y, t) + \cdots$ Then, using formula (4) of Sec. 35, the variation of the functional is

$$\delta J = \varepsilon \int_{t_0}^{t_1}\!\iint_R [-\rho u_{tt} + \tau(u_{xx} + u_{yy})]\,\psi\,dx\,dy\,dt - \varepsilon \int_{t_0}^{t_1}\!\int_\Gamma xu\psi\,ds\,dt + \varepsilon \int_{t_0}^{t_1}\!\iint_R \left[\frac{\partial}{\partial x}(u_x\psi) + \frac{\partial}{\partial y}(u_y\psi)\right] dx\,dy\,dt + \varepsilon\int_{t_0}^{t_1}\!\iint_R \frac{\partial}{\partial t}(u_t\psi)\,dx\,dy\,dt.$$

Just as in the case of the vibrating string, we assume that $u(x, y, t)$ is not varied at the initial and final times, i.e., that $\psi(x, y, t\_0) = \psi(x, y, t\_1) \equiv 0$. Then the last integral vanishes. Moreover, using Green's theorem in two dimensions, the boundary term evaluates as $\int\_\Gamma \frac{\partial u}{\partial n}\,\psi\,ds$, where $\partial/\partial n$ denotes differentiation with respect to the outward normal to $\Gamma$.

Thus, we can write the variation as

$$\delta J = \varepsilon \int_{t_0}^{t_1}\!\iint_R [-\rho u_{tt} + \tau(u_{xx} + u_{yy})]\,\psi\,dx\,dy\,dt - \varepsilon \int_{t_0}^{t_1}\!\int_\Gamma \left[\varkappa u + \tau \frac{\partial u}{\partial n}\right]\psi\,ds\,dt.$$

We first assume that $\psi(s, t) = 0$ ($s \in \Gamma$), i.e., that $u$ does not vary on the boundary of the membrane. Then $\delta J$ reduces to the volume integral. Setting it equal to zero, we find that

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Equation</span><span class="math-callout__name">(Vibrating Membrane)</span></p>

$$u_{tt} = a^2[u_{xx}(x, y, t) + u_{yy}(x, y, t)] \qquad \left(a^2 = \frac{\tau}{\rho}\right)$$

for $(x, y) \in R$ and all $t$. Equation can also be written as $u\_{tt} = a^2 \nabla^2 u(x, y, t)$, in terms of the **Laplacian** (operator)

$$\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}.$$

</div>

Next, we remove the restriction on $\psi$ at the boundary. Since $u(x, y, t)$ must satisfy the membrane equation, the first term in $\delta J$ vanishes, and we are left with

$$\delta J = -\varepsilon \int_{t_0}^{t_1}\!\int_\Gamma \left[\varkappa(s)\,u(s, t) + \tau \frac{\partial u(s, t)}{\partial n}\right]\psi(s, t)\,ds\,dt.$$

Then, since $\psi(s, t)$ is an arbitrary admissible function, equating to zero leads to the **boundary condition** for the vibrating membrane:

$$\varkappa(s)\,u(s, t) + \tau \frac{\partial u(s, t)}{\partial n} = 0 \qquad (s \in \Gamma).$$

In particular, if the boundary of the membrane is free, $\varkappa(s) = 0$ and the boundary condition becomes $\frac{\partial u(s, t)}{\partial n} = 0$ ($s \in \Gamma$), while if the boundary is fixed, $\varkappa(s) = \infty$ and the boundary condition becomes $u(s, t) = 0$ ($s \in \Gamma$).

#### 36.4. The Vibrating Plate

Finally, we use the principle of stationary action to derive the equation of the transverse vibrations of a *plate* (i.e., a homogeneous two-dimensional elastic body) with surface mass density $\rho$. As in the case of the vibrating membrane, let $u(x, y, t)$ denote the displacement from equilibrium of the point $(x, y)$ of the plate, at time $t$. Then the kinetic energy of the plate at time $t$ is given by

$$T = \frac{1}{2}\,\rho \iint_R u_t^2(x, y, t)\,dx\,dy,$$

where $R$ is the region of the $xy$-plane occupied by the plate at rest.

The potential energy of deformation of the plate, denoted by $U\_1$, depends on how the plate is bent, and hence involves the second derivatives $u\_{xx}$, $u\_{xy}$ and $u\_{yy}$. Unlike the case of the membrane, it is assumed that no work is done in stretching the plate, so that $U\_1$ does not involve $u\_x$ and $u\_y$. Moreover, we require $U\_1$ to be a quadratic functional in $u\_{xx}$, $u\_{xy}$ and $u\_{yy}$, which does not depend on the orientation of the coordinate system. Then, since the matrix

$$\begin{Vmatrix} u_{xx} & u_{xy} \\ u_{yx} & u_{yy} \end{Vmatrix}$$

has just two invariants under rotations, i.e., its trace and its determinant, it follows that

$$U_1 = \iint_R [A(u_{xx} + u_{yy})^2 + B(u_{xx}u_{yy} - u_{xy}^2)]\,dx\,dy,$$

where $A$ and $B$ are constants. This is usually written in the form

$$U_1 = \frac{1}{2}\,c \iint_R [(u_{xx}^2 + u_{yy}^2) - 2(1 - \mu)(u_{xx}u_{yy} - u_{xy}^2)]\,dx\,dy,$$

where $c$ is a constant depending on the choice of units, and $\mu$ is an absolute constant (*Poisson's ratio*) characterizing the material from which the plate is made. For simplicity, we set $c = 1$.

In addition to $U\_1$, the total potential energy of the plate may also contain a contribution $U\_2$ due to bending moments with density $m(s, t)$, prescribed on the boundary $\Gamma$ of $R$, and a contribution $U\_3$ due to external forces acting on $R$ with surface density $f(x, y, t)$ and on $\Gamma$ with linear density $p(s, t)$. This would give

$$U_2 = \int_\Gamma m(s, t)\frac{\partial u(s, t)}{\partial n}\,ds,$$

$$U_3 = \iint_R f(x, y, t)\,u(x, y, t)\,dx\,dy + \int_\Gamma p(s, t)\,u(s, t)\,ds.$$

Combining all terms, the action functional for the vibrating plate is

$$J[u] = \int_{t_0}^{t_1} (T - U_1 - U_2 - U_3)\,dt$$

$$= \frac{1}{2}\int_{t_0}^{t_1}\!\iint_R [\rho u_t^2 - (u_{xx} + u_{yy})^2 + 2(1 - \mu)(u_{xx}u_{yy} - u_{xy}^2) - 2fu]\,dx\,dy\,dt$$

$$- \int_{t_0}^{t_1}\!\int_\Gamma \left(\rho u + m\frac{\partial u}{\partial n}\right) ds\,dt.$$

Unlike the corresponding expressions for the vibrating string and the vibrating membrane, this functional contains second derivatives of the unknown function $u$. The variation of the functional, corresponding to the transition from $u(x, y, t)$ to $u^\ast(x, y, t) = u(x, y, t) + \varepsilon\psi(x, y, t) + \cdots$, turns out to be

$$\delta J = \varepsilon \int_{t_0}^{t_1}\!\iint_R (-\rho u_{tt} - \nabla^4 u - f)\,\psi\,dx\,dy\,dt + \varepsilon \int_{t_0}^{t_1}\!\int_\Gamma \left[(P - p)\psi + (M - m)\frac{\partial\psi}{\partial n}\right] ds\,dt.$$

Here,

$$M = -[\mu\nabla^2 u + (1 - \mu)(u_{xx}x_n^2 + 2u_{xy}x_n y_n + u_{yy}y_n^2)]$$

and

$$P = \frac{\partial}{\partial n}\nabla^2 u + (1 - \mu)\frac{\partial}{\partial s}[u_{xx}x_n x_s + u_{xy}(x_n y_s + x_s y_n) + u_{yy}y_n y_s],$$

where $\partial/\partial n$ denotes differentiation in the direction of the outward normal to $\Gamma$, with direction cosines $x\_n$, $y\_n$, and $\partial/\partial s$ denotes differentiation in the direction of the tangent to $\Gamma$, with direction cosines $x\_s$, $y\_s$. Moreover,

$$\nabla^4 u = \nabla^2(\nabla^2 u) = \frac{\partial^4 u}{\partial x^4} + 2\frac{\partial^4 u}{\partial x^2\,\partial y^2} + \frac{\partial^4 u}{\partial y^4}$$

according to the definition of the Laplacian.

We first assume that $\psi(s, t) = 0$, $\frac{\partial\psi(s, t)}{\partial n} = 0$ ($s \in \Gamma$), i.e., that $u$ and its normal derivative do not vary on the boundary of the plate. Then $\delta J$ reduces to the volume integral. Setting it equal to zero, we obtain the equation for *forced* vibrations of the plate:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Equation</span><span class="math-callout__name">(Vibrating Plate)</span></p>

$$\rho\, u_{tt}(x, y, t) + \nabla^4 u(x, y, t) + f(x, y, t) = 0.$$

If we set $f \equiv 0$, so that there are no external forces acting on the plate, this reduces to the equation for **free** vibrations of the plate: $\rho\, u\_{tt} + \nabla^4 u = 0$.

Finally, if we set $u\_{tt} \equiv 0$ and assume that $f = f(x, y)$ is independent of time, we obtain the equation for the equilibrium position of the plate under the action of external forces: $\nabla^4 u(x, y) + f(x, y) = 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1</span><span class="math-callout__name">(Poisson's Ratio)</span></p>

The Euler equation for the plate does not involve the coefficient $\mu$. This is explained by the fact that the expression $u\_{xx}u\_{yy} - u\_{xy}^2$ is the divergence of the vector $(u\_x u\_{yy}, -u\_x u\_{xy})$, and hence has no effect on the Euler equation. However, it does have a decisive effect on the boundary conditions, via the functions $M(s, t)$ and $P(s, t)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2</span><span class="math-callout__name">(Minimum Potential Energy)</span></p>

For a mechanical system to be in equilibrium, its kinetic energy $T$ must vanish and its potential energy $U$ must be independent of time. Under these conditions, the principle of stationary action reduces to the assertion that $\delta U = 0$. Thus, the equilibrium position of the system corresponds to a stationary value of $U$. Moreover, it can be shown that this stationary value must be a minimum if the equilibrium is to be stable and hence physically realizable. In elasticity theory, this *principle of minimum potential energy* is often replaced by *Castigliano's principle*, which states that the equilibrium position of an elastic body corresponds to a minimum of the work of deformation.

</div>

Next, we remove the restriction on the boundary. Since $u(x, y, t)$ must satisfy the plate equation, the first term in $\delta J$ vanishes, and we are left with

$$\delta J = \varepsilon \int_{t_0}^{t_1}\!\int_\Gamma \left[(P - p)\psi + (M - m)\frac{\partial\psi}{\partial n}\right] ds\,dt.$$

Then, since the functions $\psi$, $\partial\psi/\partial n$ and the interval $[t\_0, t\_1]$ are arbitrary, equating to zero leads to the **natural boundary conditions**

$$P(s, t) - p(s, t) = 0, \quad M(s, t) - m(s, t) = 0 \qquad (s \in \Gamma).$$

If the boundary of the plate is **clamped**, the natural boundary conditions are replaced by the "imposed" boundary conditions $u(s, t) = 0$, $\frac{\partial u(s, t)}{\partial n} = 0$ ($s \in \Gamma$). If the plate is **supported**, i.e., if the boundary of the plate is held fixed while the tangent plane at the boundary can vary, we obtain the boundary conditions $u(s, t) = 0$, $M(s, t) - m(s, t) = 0$ ($s \in \Gamma$).

### 37. Variation of a Functional Defined on a Variable Region

#### 37.1. Statement of the Problem

In Sec. 35, we derived the variation formula for the functional

$$J[u] = \int \cdots \int_R F(x_1, \ldots, x_n, u, u_{x_1}, \ldots, u_{x_n})\,dx_1 \cdots dx_n,$$

allowing only the function $u$ (and hence its derivatives) to vary, while leaving the independent variables (and hence the region of integration $R$) unchanged. We now find the variation of the functional in the general case where the independent variables $x\_1, \ldots, x\_n$ are varied, as well as the function $u$ and its derivatives. For simplicity, we use vector notation, writing $x = (x\_1, \ldots, x\_n)$, $dx = dx\_1 \cdots dx\_n$ and $\operatorname{grad} u \equiv \nabla u = (u\_{x\_1}, \ldots, u\_{x\_n})$. With this notation, the functional becomes

$$J[u] = \int_R F(x, u, \nabla u)\,dx.$$

Now consider the family of transformations

$$x_i^* = \Phi_i(x, u, \nabla u;\, \varepsilon), \qquad u^* = \Psi(x, u, \nabla u;\, \varepsilon),$$

$(i = 1, \ldots, n)$ depending on a parameter $\varepsilon$, where the functions $\Phi\_i$ and $\Psi$ are differentiable with respect to $\varepsilon$, and $\varepsilon = 0$ corresponds to the identity transformation: $\Phi\_i(x, u, \nabla u; 0) = x\_i$, $\Psi(x, u, \nabla u; 0) = u$.

The transformation carries the surface $\sigma$, with equation $u = u(x)$ ($x \in R$), into another surface $\sigma^\ast$ with equation $u^\ast = u^\ast(x^\ast)$ ($x^\ast \in R^\ast$), and $R^\ast$ is a new $n$-dimensional region. Thus, the transformation carries the functional $J[u(x)]$ into

$$J[u^*(x^*)] = \int_{R^*} F(x^*, u^*, \nabla^* u^*)\,dx^*,$$

where $\nabla^\ast u^\ast = (u^\ast\_{x\_1^\ast}, \ldots, u^\ast\_{x\_n^\ast})$.

Our goal in this section is to calculate the variation of the functional corresponding to the transformation from $x$, $u(x)$ to $x^\ast$, $u^\ast(x^\ast)$, i.e., the principal linear part (relative to $\varepsilon$) of the difference $J[u^\ast(x^\ast)] - J[u(x)]$.

#### 37.2. Calculation of $\delta x\_i$ and $\delta u$

Writing

$$x_i^* = \Phi_i(x, u, \nabla u;\, \varepsilon) \sim x_i + \varepsilon\varphi_i(x, u, \nabla u), \qquad u^* = \Psi(x, u, \nabla u;\, \varepsilon) \sim u + \varepsilon\psi(x, u, \nabla u),$$

where $\varphi\_i = \frac{\partial \Phi\_i}{\partial \varepsilon}\big\vert\_{\varepsilon=0}$ and $\psi = \frac{\partial \Psi}{\partial \varepsilon}\big\vert\_{\varepsilon=0}$, the increments and variations are defined as the principal linear parts (relative to $\varepsilon$):

$$\Delta x_i = x_i^* - x_i = \varepsilon\varphi_i(x) + o(\varepsilon), \qquad \Delta u = u^*(x^*) - u(x) = \varepsilon\psi(x) + o(\varepsilon),$$

and the **variations** $\delta x\_i = \varepsilon\varphi\_i$, $\delta u = \varepsilon\psi$.

We must also consider the increment $\overline{\Delta u} = u^\ast(x^\ast) - u(x)$, i.e., the change in the $u$-coordinate as we go from the point $(x, u(x))$ on the surface $\sigma$ to its image $(x^\ast, u^\ast(x^\ast))$ on the surface $\sigma^\ast$ under the transformation. Introducing a new function $\bar{\psi}(x)$ and a corresponding variation $\bar{\delta}u$:

$$\overline{\Delta u} = u^*(x^*) - u(x^*) = \varepsilon\bar{\psi}(x) + o(\varepsilon), \qquad \bar{\delta}u = \varepsilon\bar{\psi}(x).$$

To find the relation between $\delta u$ and $\bar{\delta}u$, we write

$$\Delta u = u^*(x^*) - u(x) = [u^*(x^*) - u^*(x)] + [u^*(x) - u(x)] = \sum_{i=1}^{n} \frac{\partial u^*}{\partial x_i}(x_i^* - x_i) + \bar{\delta}u + o(\varepsilon).$$

Since $\partial u^\ast/dx\_i$ and $\partial u/\partial x\_i$ differ only by a quantity of order $\varepsilon$, we have $u^\ast(x) = \varepsilon x + u(x) + o(\varepsilon)$, and:

$$\delta u = \bar{\delta}u + \sum_{i=1}^{n} u_{x_i}\,\delta x_i,$$

or equivalently,

$$\psi = \bar{\psi} + \sum_{i=1}^{n} u_{x_i}\,\varphi_i.$$

#### 37.3. Calculation of $\delta u\_{x\_i}$

We now derive an expression for the quantity $\Delta u\_{x\_i} = \frac{\partial u^\ast(x^\ast)}{\partial x\_i^\ast} - \frac{\partial u(x)}{\partial x\_i}$, or more precisely, its principal part $\delta u\_{x\_i}$. Using the chain rule and the expressions for $\delta x\_k$ from above, after careful analysis involving the Kronecker delta $\delta\_{ik}$ and the relation

$$\frac{\partial}{\partial x_i} - \frac{\partial}{\partial x_i^*} \sim \varepsilon \sum_{k=1}^{n} \frac{\partial\varphi_k}{\partial x_i} \frac{\partial}{\partial x_k^*},$$

we arrive at

$$\delta u_{x_i} = (\bar{\delta}u)_{x_i} + \sum_{k=1}^{n} u_{x_i x_k}\,\delta x_k.$$

This formula shows that **the operations $\delta$ and $\partial/\partial x\_i$ commute when applied to $\bar{\delta}u$**, i.e., $\delta u\_{x\_i} - (\bar{\delta}u)\_{x\_i} = \sum\_k u\_{x\_i x\_k}\,\delta x\_k$, which is analogous to the one-variable formula from Sec. 13.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Rotation in the $xu$-Plane)</span></p>

Let $u$ be a function of a single independent variable $x$, and let the transformation be the counterclockwise rotation of the $xu$-plane about the small angle $\alpha = \varepsilon$:

$$x^* = x\cos\varepsilon - u(x)\sin\varepsilon = x - \varepsilon u(x) + o(\varepsilon), \qquad u^*(x^*) = x\sin\varepsilon + u(x)\cos\varepsilon = \varepsilon x + u(x) + o(\varepsilon).$$

Then $\delta x = -\varepsilon u(x)$, $\delta u = \varepsilon x$, and $\varphi(x) = -u(x)$, $\psi(x) = x$. It follows from $\delta u = \bar{\delta}u + u'\,\delta x$ that $\bar{\delta}u = \varepsilon[x + u(x)u'(x)]$, i.e., $\bar{\psi}(x) = x + u(x)u'(x)$.

</div>

#### 37.4. Calculation of $\delta J$

We are now in a position to calculate the variation of a functional defined on a variable domain.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Variation of a Functional on a Variable Domain)</span></p>

*The variation of the functional*

$$J[u] = \int_R F(x, u, \nabla u)\,dx$$

*corresponding to the transformation*

$$x_i^* = \Phi_i(x, u, \nabla u;\, \varepsilon) \sim x_i + \varepsilon\varphi_i(x, u, \nabla u) \qquad (i = 1, \ldots, n),$$
$$u^* = \Psi(x, u, \nabla u;\, \varepsilon) \sim u + \varepsilon\psi(x, u, \nabla u)$$

*($i = 1, \ldots, n$) is given by the formula*

$$\delta J = \varepsilon \int_R \left(F_u - \sum_{i=1}^{n} \frac{\partial}{\partial x_i} F_{u_{x_i}}\right)\bar{\psi}\,dx + \varepsilon \int_R \sum_{i=1}^{n} \frac{\partial}{\partial x_i}(F_{u_{x_i}}\bar{\psi} + F\,\delta x_i)\,dx,$$

*where $\bar{\psi} = \psi - \sum\_{i=1}^{n} u\_{x\_i}\varphi\_i$.*

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1</span><span class="math-callout__name">(Fixed Independent Variables)</span></p>

In the special case where the function $u$ and its derivatives are varied, but not the independent variables $x\_i$, we have $\varphi\_i = 0$, $\bar{\psi} = \psi - \sum u\_{x\_i}\varphi\_i = \psi$, and the formula reduces to

$$\delta J = \varepsilon \int_R \left(F_u - \sum_{i=1}^{n} \frac{\partial}{\partial x_i} F_{u_{x_i}}\right)\psi(x)\,dx + \varepsilon \int_R \sum_{i=1}^{n} \frac{\partial}{\partial x_i}[F_{u_{x_i}}\psi(x)]\,dx,$$

which is identical with formula (4) of Sec. 35.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2</span><span class="math-callout__name">(Extremal Surface)</span></p>

The formula for the variation of $J[u]$ is ordinarily used in the case where $u = u(x)$ is an extremal surface of $J[u]$, i.e., satisfies the Euler equation $F\_u - \sum \frac{\partial}{\partial x\_i} F\_{u\_{x\_i}} = 0$. Then the formula reduces to

$$\delta J = \varepsilon \int_R \sum_{i=1}^{n} \frac{\partial}{\partial x_i}(F_{u_{x_i}}\bar{\psi} + F\varphi_i)\,dx$$

in the general case, and to $\delta J = \varepsilon \int\_R \sum \frac{\partial}{\partial x\_i}(F\_{u\_{x\_i}}\psi)\,dx$ in the case where the independent variables $x\_i$ are not varied.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3</span><span class="math-callout__name">(Multiple Unknown Functions)</span></p>

Consider the functional involving $m$ unknown functions $u\_1, \ldots, u\_m$ and their derivatives:

$$J[u_1, \ldots, u_m] = \int_R F\!\left(x, u_1, \ldots, u_m, \frac{\partial u_1}{\partial x_1}, \ldots, \frac{\partial u_m}{\partial x_n}\right) dx.$$

Introducing the vector $u = (u\_1, \ldots, u\_m)$ and interpreting $\nabla u$ as the tensor with components $\frac{\partial u\_j}{\partial x\_i}$ ($i = 1, \ldots, n$; $j = 1, \ldots, m$), the formula for the variation generalizes to

$$\delta J = \varepsilon \int_R \sum_{j=1}^{m} \left(F_{u_j} - \sum_{i=1}^{n} \frac{\partial}{\partial x_i} \frac{\partial F}{\partial\!\left(\frac{\partial u_j}{\partial x_i}\right)}\right)\bar{\psi}_j\,dx + \varepsilon \int_R \sum_{i=1}^{n} \frac{\partial}{\partial x_i}\left(\sum_{j=1}^{m} \frac{\partial F}{\partial\!\left(\frac{\partial u_j}{\partial x_i}\right)}\bar{\psi}_j + F\varphi_i\right) dx,$$

where $\bar{\psi}\_j = \psi\_j - \sum\_{i=1}^{n} \frac{\partial u\_j}{\partial x\_i}\varphi\_i$ ($j = 1, \ldots, m$).

</div>

#### 37.5. Noether's Theorem

Using the formula for the variation of a functional, we can deduce an important theorem due to Noether, concerning "invariant variational problems." This theorem has already been proved in Sec. 20 for the case of a single independent variable.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Invariance under a Transformation)</span></p>

The functional $J[u] = \int\_R F(x, u, \nabla u)\,dx$ is said to be **invariant** under the transformation $x\_i^\ast = \Phi\_i(x, u, \nabla u)$, $u^\ast = \Psi(x, u, \nabla u)$ if $J[\sigma^\ast] = J[\sigma]$, i.e., if

$$\int_{R^*} F(x^*, u^*, \nabla^* u^*)\,dx^* = \int_R F(x, u, \nabla u)\,dx.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Invariance under Rotation)</span></p>

The functional

$$J[u] = \iint_R \left[\left(\frac{\partial u}{\partial x}\right)^2 + \left(\frac{\partial u}{\partial y}\right)^2\right] dx\,dy$$

is invariant under the rotation $x^\ast = x\cos\varepsilon - y\sin\varepsilon$, $y^\ast = x\sin\varepsilon + y\cos\varepsilon$, $u^\ast = u$, where $\varepsilon$ is an arbitrary constant. In fact, since the inverse of the transformation is $x = x^\ast\cos\varepsilon + y^\ast\sin\varepsilon$, $y = -x^\ast\sin\varepsilon + y^\ast\cos\varepsilon$, $u = u^\ast$, a surface $\sigma$ with equation $u = u(x, y)$ has "transformed" surface $\sigma^\ast$ with equation $u^\ast = u^\ast(x^\ast, y^\ast)$, and

$$J[\sigma^*] = \iint_{R^*}\left[\left(\frac{\partial u^*}{\partial x^*}\right)^2 + \left(\frac{\partial u^*}{\partial y^*}\right)^2\right] dx^*\,dy^* = \iint_R \left[\left(\frac{\partial u}{\partial x}\right)^2 + \left(\frac{\partial u}{\partial y}\right)^2\right] dx\,dy = J[\sigma].$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2</span><span class="math-callout__name">(Noether's Theorem for Multiple Integrals)</span></p>

*If the functional*

$$J[u] = \int_R F(x, u, \nabla u)\,dx$$

*is invariant under the family of transformations*

$$x_i^* = \Phi_i(x, u, \nabla u;\, \varepsilon) \sim x_i + \varepsilon\varphi_i(x, u, \nabla u), \qquad u^* = \Psi(x, u, \nabla u;\, \varepsilon) \sim u + \varepsilon\psi(x, u, \nabla u),$$

*$(i = 1, \ldots, n)$ for an arbitrary region $R$, then*

$$\sum_{i=1}^{n} \frac{\partial}{\partial x_i}(F_{u_{x_i}}\bar{\psi} + F\varphi_i) = 0$$

*on each extremal surface of $J[u]$, where $\bar{\psi} = \psi - \sum\_{i=1}^{n} u\_{x\_i}\varphi\_i$.*

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

According to the variation formula, $\delta J = \varepsilon \int\_R \sum\_{i=1}^{n} \frac{\partial}{\partial x\_i}(F\_{u\_{x\_i}}\bar{\psi} + F\varphi\_i)\,dx$, if $u = u(x)$ is an extremal surface. Since $J[u]$ is invariant under the given transformation, $\delta J = 0$, and since $R$ is arbitrary, this implies the asserted divergence-free condition.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1</span><span class="math-callout__name">(Fixed Independent Variables)</span></p>

If we drop the requirement that $u = u(x)$ be an extremal surface of $J[u]$, then, using the full variation formula, we find that the invariance condition $\delta J = 0$ implies

$$\left(F_u - \sum_{i=1}^{n} \frac{\partial}{\partial x_i} F_{u_{x_i}}\right)\bar{\psi} + \sum_{i=1}^{n} \frac{\partial}{\partial x_i}(F_{u_{x_i}}\bar{\psi} + F\varphi_i) = 0.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2</span><span class="math-callout__name">(Multiple Unknown Functions)</span></p>

If there are $m$ unknown functions $u = (u\_1, \ldots, u\_m)$ and we continue to write the functional as $J[u] = \int\_R F(x, u, \nabla u)\,dx$, then invariance of $J[u]$ under the family of transformations

$$x_i^* \sim x_i + \varepsilon\varphi_i(x, u, \nabla u), \qquad u_j^* \sim u_j + \varepsilon\psi_j(x, u, \nabla u)$$

implies that

$$\sum_{i=1}^{n} \frac{\partial}{\partial x_i}\left(\sum_{j=1}^{m} \frac{\partial F}{\partial\!\left(\frac{\partial u_j}{\partial x_i}\right)}\bar{\psi}_j + F\varphi_i\right) = 0,$$

where $\bar{\psi}\_j = \psi\_j - \sum\_{i=1}^{n} \frac{\partial u\_j}{\partial x\_i}\varphi\_i$.

When $n = 1$, this reduces to $\frac{d}{dx}\!\left(\sum\_{j=1}^{m} F\_{u\_j'}\bar{\psi}\_j + F\varphi\right) = 0$, or equivalently $\sum\_{j=1}^{m} F\_{u\_j'}\psi\_j + \left(F - \sum\_{j=1}^{m} u\_j' F\_{u\_j'}\right)\varphi = \text{const}$, which is precisely the version of Noether's theorem proved in Sec. 20.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3</span><span class="math-callout__name">(Multi-Parameter Invariance)</span></p>

Invariance of the functional $J[u] = \int\_R F(x, u, \nabla u)\,dx$ under an $r$-parameter family of transformations

$$x_i^* \sim x_i + \sum_{k=1}^{r} \varepsilon_k\,\varphi_i^{(k)}(x, u, \nabla u), \qquad u_j^* \sim u_j + \sum_{k=1}^{r} \varepsilon_k\,\psi_j^{(k)}(x, u, \nabla u)$$

implies the existence of $r$ linearly independent relations

$$\sum_{i=1}^{n} \frac{\partial}{\partial x_i}\left(\sum_{j=1}^{m} \frac{\partial F}{\partial\!\left(\frac{\partial u_j}{\partial x_i}\right)}\bar{\psi}_j^{(k)} + F\varphi_i^{(k)}\right) = 0 \qquad (k = 1, \ldots, r),$$

where $\bar{\psi}\_j^{(k)} = \psi\_j^{(k)} - \sum\_{i=1}^{n} \frac{\partial u\_j}{\partial x\_i}\,\varphi\_i^{(k)}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4</span><span class="math-callout__name">(Identities from Function-Dependent Invariance)</span></p>

Suppose the functional $J[u]$ is invariant under a family of transformations depending on $r$ arbitrary functions instead of $r$ arbitrary parameters. Then, according to another theorem of Noether (which will not be proved here), there are $r$ identities connecting the left-hand sides of the Euler equations corresponding to $J[u]$. For example, consider the simplest variational problem in parametric form, involving a functional $J[x, y] = \int\_{t\_0}^{t\_1} \Phi(x, y, \dot{x}, \dot{y})\,dt$, where $\Phi$ is a positive-homogeneous function of degree 1 in $\dot{x}(t)$ and $\dot{y}(t)$. Then $J[x, y]$ does not change if we introduce a new parameter $\tau$ by setting $t = t(\tau)$, where $dt/d\tau > 0$. In fact, the left-hand sides of the Euler equations $\Phi\_x - \frac{d}{dt}\Phi\_{\dot{x}} = 0$, $\Phi\_y - \frac{d}{dt}\Phi\_{\dot{y}} = 0$ are connected by the identity

$$\dot{x}\!\left(\Phi_x - \frac{d}{dt}\Phi_{\dot{x}}\right) + \dot{y}\!\left(\Phi - \frac{d}{dt}\Phi_{\dot{y}}\right) = 0.$$

</div>

### 38. Applications to Field Theory

#### 38.1. The Principle of Stationary Action for Fields

In Sec. 36, we discussed the application of the principle of stationary action to vibrating systems with infinitely many degrees of freedom. These systems were characterized by a function $u(x, t)$ or $u(x, y, t)$ giving the transverse displacement of the system from its equilibrium position. More generally, consider a physical system (not necessarily mechanical) characterized by one function

$$u(t, x_1, \ldots, x_n)$$

or by a set of functions

$$u_j(t, x_1, \ldots, x_n) \qquad (j = 1, \ldots, m),$$

depending on the time $t$ and the space coordinates $x\_1, \ldots, x\_n$. Such a system is called a *field*, and the functions $u\_j$ are called the *field functions*. As usual, we can simplify the notation by interpreting $u = (u\_1, \ldots, u\_m)$ as a vector function in the case where $m > 1$. It is also convenient to write

$$t = x_0, \quad x = (x_0, x_1, \ldots, x_n), \quad dx = dx_0\,dx_1 \cdots dx_n.$$

Then the field function becomes simply $u(x)$.

In the case of the simple vibrating systems studied in Sec. 36, the equations of motion for the system were derived by first calculating the action functional $\int\_a^b (T - U)\,dt$, where $T$ is the kinetic energy and $U$ the potential energy of the system, and then invoking the principle of stationary action. Similarly, many other physical fields can be derived from a suitably defined action functional. By analogy with the vibrating string and the vibrating membrane, we write the action in the form

$$J[u, \nabla u] = \int_a^b dx_0 \int \cdots \int_R L(u, \nabla u)\,dx_1 \cdots dx_n = \int_\Omega \mathscr{L}(u, \nabla u)\,dx,$$

where $\nabla$ is the operator $\left(\frac{\partial}{\partial x\_0}, \frac{\partial}{\partial x\_1}, \ldots, \frac{\partial}{\partial x\_n}\right)$, $R$ is some $n$-dimensional region, and $\Omega$ is the "cylindrical space-time region" $R \times [a, b]$, i.e., the Cartesian product of $R$ and the interval $[a, b]$. The functions $L(u, \nabla u)$ and $\mathscr{L}(u, \nabla u)$ are called the **Lagrangian** and **Lagrangian density** of the field, respectively.

Applying the principle of stationary action to the functional, we require that $\delta J = 0$. This leads to the Euler equations

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Equation</span><span class="math-callout__name">(Field Equations)</span></p>

$$\frac{\partial \mathscr{L}}{\partial u_j} - \sum_{i=0}^{3} \frac{\partial}{\partial x_i}\frac{\partial \mathscr{L}}{\partial\!\left(\frac{\partial u_j}{\partial x_i}\right)} = 0 \qquad (j = 1, \ldots, m),$$

which are the desired field equations.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1</span><span class="math-callout__name">(Vibrating String with Free Ends)</span></p>

For the vibrating string with free ends ($\varkappa\_1 = \varkappa\_2 = 0$), we have $m = n = 1$, and

$$\mathscr{L} = \tfrac{1}{2}(\rho u_t^2 - \tau u_x^2) = \tfrac{1}{2}(\rho u_{x_0}^2 - \tau u_{x_1}^2)$$

[cf. formula (16) of the vibrating string].

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2</span><span class="math-callout__name">(Vibrating Membrane with Free Boundary)</span></p>

For the vibrating membrane with a free boundary [$\varkappa(s) = 0$], we have $m = 1$, $n = 2$, and

$$\mathscr{L} = \tfrac{1}{2}[\rho u_t^2 - \tau(u_x^2 + u_y^2)] = \tfrac{1}{2}[\rho u_{x_0}^2 - \tau(u_{x_1}^2 + u_{x_2}^2)]$$

[cf. formula (42) of the vibrating membrane].

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3</span><span class="math-callout__name">(Klein-Gordon Equation)</span></p>

Consider the *Klein-Gordon equation*

$$(\Box - M^2)u(x) = 0,$$

describing the scalar field corresponding to uncharged particles of mass $M$ with spin zero (e.g., $\pi^0$-mesons). Here, $\Box$ denotes the *D'Alembertian* (operator)

$$\Box = -\frac{\partial^2}{\partial x_0^2} + \frac{\partial^2}{\partial x_1^2} + \frac{\partial^2}{\partial x_2^2} + \frac{\partial^2}{\partial x_3^2}.$$

It is easy to see that the Klein-Gordon equation is the Euler equation corresponding to the Lagrangian density

$$\mathscr{L} = \tfrac{1}{2}(u_{x_0}^2 - u_{x_1}^2 - u_{x_2}^2 - u_{x_3}^2 - M^2 u^2).$$

</div>

#### 38.2. Conservation Laws for Fields

Noether's theorem (derived in Sec. 37.5) affords a general method of deriving *conservation laws* for fields, i.e., for constructing combinations of field functions, called **field invariants**, which do not change in time.

Thus, suppose the integral

$$\int_\Omega \mathscr{L}(u, \nabla u)\,dx$$

is invariant under an $r$-parameter family of transformations

$$x_i^* = \Phi_i(x, u, \nabla u;\, \varepsilon) \sim x_i + \sum_{k=1}^{r} \varepsilon_k\,\varphi_i^{(k)} \qquad (i = 0, 1, 2, 3),$$

$$u_j^* = \Psi_j(x, u, \nabla u;\, \varepsilon) \sim u_j + \sum_{k=1}^{r} \varepsilon_k\,\psi_j^{(k)} \qquad (j = 1, \ldots, m),$$

where $\varepsilon = (\varepsilon\_1, \ldots, \varepsilon\_r)$. Then, according to Remark 3 of Sec. 37.5, we have $r$ relations of the form

$$\operatorname{div} I^{(k)} = \sum_{i=0}^{3} \frac{\partial I_i^{(k)}}{\partial x_i} = 0,$$

where

$$I_i^{(k)} = \sum_{j=1}^{m} \frac{\partial \mathscr{L}}{\partial\!\left(\frac{\partial u_j}{\partial x_i}\right)} \bar{\psi}_j^{(k)} + \mathscr{L}\,\varphi_i^{(k)} \qquad (k = 1, \ldots, r)$$

and $\bar{\psi}\_j^{(k)} = \psi\_j^{(k)} - \sum\_{i=0}^{n} \frac{\partial u\_j}{\partial x\_i}\,\varphi\_i^{(k)}$.

These equations have the following interesting consequence: Suppose the cylinder $\Omega = R \times [a, b]$, where $R$ is the three-dimensional sphere defined by $x\_1^2 + x\_2^2 + x\_3^2 \leqslant c^2$. Let $\Gamma$ be the boundary of $\Omega$, and let $\nu$ be the unit outward normal to $\Gamma$. Then, integrating each of the relations $\operatorname{div} I^{(k)} = 0$ over $\Omega$ and using Green's theorem [formula (5) of Sec. 35], we obtain

$$\int_\Omega \operatorname{div} I^{(k)}\,dx = \int_\Gamma (I^{(k)}, \nu)\,d\sigma = 0 \qquad (k = 1, \ldots, r).$$

The surface integral is the sum of an integral over the lateral surface of the cylinder $\Gamma$ and an integral over the two end surfaces cut off by the planes $x\_0 = a$, $x\_0 = b$. As $c \to \infty$, the integral over the lateral surfaces goes to zero (by the usual argument requiring that the field fall off at infinity "sufficiently rapidly"), and we are left with the integral over the end surfaces. On these surfaces, the scalar product $(I^{(k)}, \nu)$ reduces to $I\_0^{(k)}$, where the plus sign refers to the "top" surface and the minus sign to the "bottom" surface. Therefore, taking the limit as $c \to \infty$, we find that

$$\int I_0^{(k)}(a, x_1, x_2, x_3)\,dx_1\,dx_2\,dx_3 = \int I_0^{(k)}(b, x_1, x_2, x_3)\,dx_1\,dx_2\,dx_3 \qquad (k = 1, \ldots, r),$$

where $I\_0^{(k)}$ denotes the $x\_0$-component of the vector $I^{(k)}$, and the integrations extend over all of three-dimensional space. Since $a$ and $b$ are arbitrary, it follows from this that the quantities

$$P_k = \int I_0^{(k)}\,dx_1\,dx_2\,dx_3 = \int \left(\sum_{j=1}^{m} \frac{\partial \mathscr{L}}{\partial\!\left(\frac{\partial u_j}{\partial x_0}\right)} \bar{\psi}_j^{(k)} + \mathscr{L}\,\varphi_0^{(k)}\right) dx_1\,dx_2\,dx_3 \qquad (k = 1, \ldots, r)$$

are independent of time. The $r$ quantities $P\_k$ are the required field invariants, whose existence is implied by the invariance of the action functional under the $r$-parameter family of transformations.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Of course, all the functions in the field invariant expressions are supposed to be evaluated on an extremal surface of the action functional, corresponding to a solution $u(x)$ of the field equations.

</div>

#### 38.3. Conservation of Energy and Momentum

The action functional of any physical field is invariant under parallel displacements, i.e., under the family of transformations

$$x_i^* = x_i + \varepsilon_i \quad (i = 0, 1, 2, 3), \qquad u_j^* = u_j \quad (j = 1, \ldots, m),$$

where the $\varepsilon\_i$ are arbitrary. In this case, we have $\delta x\_i = \varepsilon\_i$, $\delta u\_j = 0$, which implies

$$\varphi_i^{(k)} = \delta_{ik}, \qquad \bar{\psi}_j^{(k)} = -\sum_{i=0}^{n} \frac{\partial u_j}{\partial x_i}\,\delta_{ik} = -\frac{\partial u_j}{\partial x_k},$$

where $\delta\_{ik}$ is the Kronecker delta. It is convenient to introduce the second-rank tensor

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Energy-Momentum Tensor)</span></p>

The **energy-momentum tensor** is defined as

$$T_{ik} = \sum_{j=1}^{m} \frac{\partial \mathscr{L}}{\partial\!\left(\frac{\partial u_j}{\partial x_i}\right)} \frac{\partial u_j}{\partial x_k} - \mathscr{L}\,\delta_{ik}.$$

In terms of $T\_{ik}$, the field invariants corresponding to translational symmetry are

$$P_k = \int T_{0k}\,dx_1\,dx_2\,dx_3 \qquad (k = 0, 1, 2, 3).$$

The vector $P = (P\_0, P\_1, P\_2, P\_3)$ is called the **energy-momentum vector**, and in fact it can be shown that $P\_0$ is the energy and $P\_1, P\_2, P\_3$ the momentum components of the field. Thus, since $P$ is a field invariant, we have just proved that *the energy and momentum of the field are conserved*.

</div>

#### 38.4. Conservation of Angular Momentum

According to the special theory of relativity, the action functional of any physical field is invariant under *orthochronous Lorentz transformations*, i.e., under transformations of four-dimensional space-time which leave the quadratic form

$$-x_0^2 + x_1^2 + x_2^2 + x_3^2$$

invariant and preserve the time direction. For simplicity, we consider the case where $u(x)$ is a scalar field ($m = 1$). Then the action functional must be invariant under the family of (infinitesimal) transformations

$$x_i^* \sim x_i + \sum_{l \neq i} g_{il}\,\varepsilon_{il}\,x_l, \qquad u^* = u,$$

where $g\_{00} = -1$, $g\_{11} = g\_{22} = g\_{33} = 1$, and $\varepsilon\_{kl} = -\varepsilon\_{ik}$ ($k \neq l$). Since the twelve parameters $\varepsilon\_{kl}$ are connected by the relations $\varepsilon\_{kl} = -\varepsilon\_{lk}$, only six of them are independent, and we choose the independent parameters to be those for which $k < l$.

Corresponding to the transformations, we have

$$\delta x_i = \sum_{l \neq i} g_{il}\,\varepsilon_{il}\,x_l = \sum_{l < k} \sum_{k=0}^{3} \varepsilon_{kl}(g_{il}\,\delta_{ik}\,x_l - g_{ik}\,\delta_{il}\,x_k),$$

and $\bar{\delta}u = -\sum\_{i=0}^{3} \frac{\partial u}{\partial x\_i}\,\delta x\_i$, where the pair of indices $k$, $l$ plays the same role as the single index $k$ in the general theory, and ranges over the six combinations $0, 1;\; 0, 2;\; 0, 3;\; 1, 2;\; 1, 3;\; 2, 3$.

According to the field invariant formula, the corresponding field invariants are

$$\int \left(\frac{\partial\mathscr{L}}{\partial\!\left(\frac{\partial u}{\partial x_i}\right)}\left[\frac{\partial u}{\partial x_l}\,g_{kk}\,x_k - \frac{\partial u}{\partial x_k}\,g_{ll}\,x_l\right] + \mathscr{L}[g_{il}\,\delta_{ik}\,x_l - g_{ik}\,\delta_{il}\,x_k]\right) dx_1\,dx_2\,dx_3 \qquad (k < l).$$

It is convenient to introduce the third-rank tensor

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Angular Momentum Tensor)</span></p>

The **angular momentum tensor** (for a scalar field) is defined as

$$M_{ikl} = \frac{\partial \mathscr{L}}{\partial\!\left(\frac{\partial u}{\partial x_i}\right)}\left[\frac{\partial u}{\partial x_l}\,g_{kk}\,x_k - \frac{\partial u}{\partial x_k}\,g_{ll}\,x_l\right] + \mathscr{L}[g_{il}\,\delta_{ik}\,x_l - g_{ik}\,\delta_{il}\,x_k] \qquad (k < l),$$

$$M_{ikl} = -M_{ilk} \qquad (k > l),$$

i.e., $M\_{ikl}$ is antisymmetric in the indices $k$ and $l$. Using the expression for the energy-momentum tensor (specialized to the case of scalar fields), we can write $M\_{ikl}$ as

$$M_{ikl} = g_{kk}\,x_k\,T_{il} - g_{ll}\,x_l\,T_{ik}.$$

In terms of $M\_{ikl}$, the field invariants are

$$\int M_{0kl}\,dx_1\,dx_2\,dx_3 \qquad (k < l),$$

a fact summarized by saying that *the angular momentum of the field is conserved*.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Klein-Gordon Field)</span></p>

Using the quantities $g\_{ii}$, we can write the Lagrangian density corresponding to the Klein-Gordon equation in the form

$$\mathscr{L} = -\frac{1}{2}\sum_{i=0}^{3} g_{ii}\!\left(\frac{\partial u}{\partial x_i}\right)^2 - \frac{1}{2}M^2 u^2.$$

This leads to the energy-momentum tensor

$$T_{ik} = -g_{ii}\,\frac{\partial u}{\partial x_i}\,\frac{\partial u}{\partial x_k} - \mathscr{L}\,\delta_{ik}$$

and the angular momentum tensor

$$M_{ikl} = g_{ii}\,\frac{\partial u}{\partial x_i}\!\left(g_{ll}\,x_l\,\frac{\partial u}{\partial x_k} - g_{kk}\,x_k\,\frac{\partial u}{\partial x_l}\right) + \mathscr{L}(g_{il}\,\delta_{ik}\,x_l - g_{ik}\,\delta_{il}\,x_k).$$

The energy density corresponding to the Klein-Gordon field is

$$T_{00} = \frac{1}{2}\sum_{i=0}^{3}\!\left(\frac{\partial u}{\partial x_i}\right)^2 + \frac{1}{2}M^2 u^2,$$

while the momentum has the components

$$T_{0k} = \frac{\partial u}{\partial x_0}\,\frac{\partial u}{\partial x_k} \qquad (k = 1, 2, 3).$$

</div>

#### 38.5. The Electromagnetic Field

To illustrate the methods developed above, we now derive the equations of the electromagnetic field from a suitable Lagrangian density. The electromagnetic field is described by two three-dimensional vectors, the *electric field vector* $E = (E\_1, E\_2, E\_3)$ and the *magnetic field vector* $H = (H\_1, H\_2, H\_3)$. In the absence of electric charges, $E$ and $H$ are related by the familiar *Maxwell equations*

$$\operatorname{curl} E = -\frac{\partial H}{\partial x_0}, \qquad \operatorname{curl} H = \frac{\partial E}{\partial x_0},$$

$$\operatorname{div} H = 0, \qquad \operatorname{div} E = 0,$$

where

$$\operatorname{div} E = \frac{\partial E_1}{\partial x_1} + \frac{\partial E_2}{\partial x_2} + \frac{\partial E_3}{\partial x_3},$$

$$\operatorname{curl} E = \left(\frac{\partial E_3}{\partial x_2} - \frac{\partial E_2}{\partial x_3},\; \frac{\partial E_1}{\partial x_3} - \frac{\partial E_3}{\partial x_1},\; \frac{\partial E_2}{\partial x_1} - \frac{\partial E_1}{\partial x_2}\right),$$

and similarly for $\operatorname{div} H$, $\operatorname{curl} H$. It is convenient to express $E$ and $H$ in terms of a four-dimensional *electromagnetic potential* $\lbrace A\_i \rbrace = (A\_0, A\_1, A\_2, A\_3)$, by setting

$$E = \operatorname{grad} A_0 - \frac{\partial A}{\partial x_0}, \qquad H = \operatorname{curl} A,$$

where $A = (A\_1, A\_2, A\_3)$ and $\operatorname{grad} A\_0 = \left(\frac{\partial A\_0}{\partial x\_1}, \frac{\partial A\_0}{\partial x\_2}, \frac{\partial A\_0}{\partial x\_3}\right)$.

The potential $\lbrace A\_i \rbrace$ is not uniquely determined by the vectors $E$ and $H$. In fact, $E$ and $H$ do not change if we make a *gauge transformation*, i.e., if we replace $\lbrace A\_i \rbrace$ by a new potential $\lbrace A\_i' \rbrace$ with components $A\_j'(x) = A\_j(x) + \frac{\partial f(x)}{\partial x\_j}$ ($j = 0, 1, 2, 3$), where $f(x)$ is an arbitrary function. To avoid this lack of uniqueness, an extra condition can be imposed on $\lbrace A\_i \rbrace$. The condition usually chosen is

$$-\frac{\partial A_0}{\partial x_0} + \operatorname{div} A = \sum_{j=0}^{3} g_{jj}\,\frac{\partial A_j}{\partial x_j} = 0,$$

and is known as the *Lorentz condition*.

Next, we prove that the Maxwell equations reduce to a single equation determining the electromagnetic potential $\lbrace A\_i \rbrace$. First, we introduce the antisymmetric tensor $H\_{ij}$, whose matrix

$$\begin{Vmatrix} 0 & -E_1 & -E_2 & -E_3 \\ E_1 & 0 & H_3 & -H_2 \\ E_2 & -H_3 & 0 & H_1 \\ E_3 & H_2 & -H_1 & 0 \end{Vmatrix}$$

is formed from the components of $E$ and $H$. It is easily verified that the formula relating $H\_{ij}$ to the potential $\lbrace A\_i \rbrace$ is

$$H_{ij} = \frac{\partial A_j}{\partial x_i} - \frac{\partial A_i}{\partial x_j}.$$

In terms of $H\_{ij}$, we can write the Maxwell equations in the form

$$\sum_{i=0}^{3} g_{ii}\,\frac{\partial H_{ij}}{\partial x_i} = 0 \qquad (j = 0, 1, 2, 3),$$

$$\frac{\partial H_{ij}}{\partial x_k} + \frac{\partial H_{ki}}{\partial x_j} + \frac{\partial H_{jk}}{\partial x_i} = 0,$$

where in the second equation $i, j, k$ ranges over the cyclic combinations $0, 1, 2;\; 1, 2, 3;\; 2, 3, 0;\; 3, 0, 1$.

Substituting the expression for $H\_{ij}$ in terms of $\lbrace A\_i \rbrace$ into these equations, and using the Lorentz condition, we find that the second equation is an identity, while the first reduces to

$$\Box A_j = 0 \qquad (j = 0, 1, 2, 3),$$

where $\Box$ is the D'Alembertian $\Box = -\frac{\partial^2}{\partial x\_0^2} + \frac{\partial^2}{\partial x\_1^2} + \frac{\partial^2}{\partial x\_2^2} + \frac{\partial^2}{\partial x\_3^2}$.

Finally, we show that $\Box A\_j = 0$ is a consequence of the principle of stationary action, if we choose the Lagrangian density of the electromagnetic field to be

$$\mathscr{L} = \frac{1}{8\pi}(E^2 - H^2).$$

Replacing $E$ and $H$ by their expressions in terms of the electromagnetic potential $\lbrace A\_i \rbrace$, we obtain

$$\mathscr{L} = \frac{1}{8\pi}\!\left[\left(\operatorname{grad} A_0 - \frac{\partial A}{\partial x_0}\right)^2 - (\operatorname{curl} A)^2\right].$$

We verify that the Euler equations

$$\frac{\partial \mathscr{L}}{\partial A_j} - \sum_{i=0}^{3} \frac{\partial}{\partial x_i}\frac{\partial \mathscr{L}}{\partial\!\left(\frac{\partial A_j}{\partial x_i}\right)} = 0 \qquad (j = 0, 1, 2, 3)$$

corresponding to this Lagrangian density can be reduced to $\Box A\_j = 0$. For example, for $j = 0$ the Euler equation becomes

$$\frac{\partial \mathscr{L}}{\partial A_0} - \sum_{i=0}^{3}\frac{\partial}{\partial x_i}\frac{\partial \mathscr{L}}{\partial\!\left(\frac{\partial A_0}{\partial x_i}\right)} = -\frac{1}{4\pi}\!\left[\frac{\partial^2 A_0}{\partial x_1^2} + \frac{\partial^2 A_0}{\partial x_2^2} + \frac{\partial^2 A_0}{\partial x_3^2} - \frac{\partial}{\partial x_0}\!\left(\frac{\partial A_1}{\partial x_1} + \frac{\partial A_2}{\partial x_2} + \frac{\partial A_3}{\partial x_3}\right)\right] = 0.$$

According to the Lorentz condition, $\frac{\partial A\_1}{\partial x\_1} + \frac{\partial A\_2}{\partial x\_2} + \frac{\partial A\_3}{\partial x\_3} = \frac{\partial A\_0}{\partial x\_0}$, and hence the expression reduces to

$$-\frac{\partial^2 A_0}{\partial x_0^2} + \frac{\partial^2 A_0}{\partial x_1^2} + \frac{\partial^2 A_0}{\partial x_2^2} + \frac{\partial^2 A_0}{\partial x_3^2} = \Box A_0 = 0,$$

which is just $\Box A\_j = 0$ for $j = 0$. The calculations for $A\_1$, $A\_2$, $A\_3$ are completely analogous.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1</span><span class="math-callout__name">(Lorentz Condition)</span></p>

In deriving $\Box A\_j = 0$ from the Lagrangian density, we made use of the Lorentz condition. Instead, we could have introduced an additional term into the Lagrangian density by writing

$$\mathscr{L} = \frac{1}{8\pi}\!\left\lbrace\left(\operatorname{grad} A_0 - \frac{\partial A}{\partial x_0}\right)^2 - (\operatorname{curl} A)^2 - \left(\operatorname{div} A - \frac{\partial A_0}{\partial x_0}\right)^2\right\rbrace,$$

which reduces to the previous expression if the Lorentz condition is satisfied. The Euler equations corresponding to this modified Lagrangian density reduce to $\Box A\_j = 0$ for *arbitrary* $\lbrace A\_i \rbrace$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2</span><span class="math-callout__name">(Symmetries of the Electromagnetic Field)</span></p>

The Lagrangian density of the electromagnetic field, and hence its action functional, is invariant under parallel displacements, Lorentz transformations, and gauge transformations. According to Sec. 38.3, the invariance under parallel displacements implies conservation of energy and momentum of the field, while according to Sec. 38.4, the invariance under Lorentz transformations implies conservation of angular momentum of the field. Moreover, according to Remark 4 of Sec. 37.5, the invariance under gauge transformations (which depend on one arbitrary function) implies the existence of a relation between the left-hand sides of the corresponding Euler equations $\Box A\_j = 0$. Therefore, these equations do not uniquely determine the electromagnetic potential $\lbrace A\_i \rbrace$. In fact, to determine $\lbrace A\_i \rbrace$ uniquely, we need an extra equation, which is usually chosen to be the Lorentz condition.

</div>

## Chapter 8: Direct Methods in the Calculus of Variations

Up to this point, the standard strategy for solving a variational problem has been to convert it into a differential equation (or a system of differential equations). While powerful, this indirect approach has limitations: the boundary value problems that arise may require solutions over an entire region $R$ rather than just in a local neighborhood, and when multiple independent variables are involved the Euler equation becomes a partial differential equation, which can be very hard to solve. These difficulties motivate a fundamentally different strategy known as *direct methods*, which bypass the reduction to differential equations altogether.

Interestingly, once direct variational methods have been developed, they can be turned around and used to prove the existence of solutions to differential equations. If a given differential equation happens to be the Euler equation of some functional, and if one can show that this functional attains its extremum on a sufficiently smooth admissible function, then the differential equation must have a solution satisfying the corresponding boundary conditions. Moreover, the approximations constructed along the way can be used to compute solutions to any desired accuracy.

### 39. Minimizing Sequences

The direct methods considered here all rest on a single general idea. Consider the problem of minimizing a functional $J[y]$ on a space $\mathscr{M}$ of admissible functions $y$. We assume there exist functions in $\mathscr{M}$ for which $J[y] < +\infty$ and that the infimum is finite:

$$\inf_y J[y] = \mu > -\infty,$$

where the greatest lower bound is taken over all admissible $y$. By the definition of $\mu$, we can find an infinite sequence of functions $y\_1, y\_2, \ldots$ such that

$$\lim_{n \to \infty} J[y_n] = \mu.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Minimizing Sequence)</span></p>

A sequence of admissible functions $\lbrace y\_n \rbrace$ is called a **minimizing sequence** for the functional $J[y]$ if $\lim\_{n \to \infty} J[y\_n] = \mu$, where $\mu = \inf\_y J[y]$.

</div>

If the minimizing sequence $\lbrace y\_n \rbrace$ converges to a limit function $\hat{y}$, and if interchanging the functional and the limit is justified, then

$$J[\hat{y}] = \lim_{n \to \infty} J[y_n] = \mu,$$

and $\hat{y}$ solves the variational problem. The functions of the minimizing sequence serve as approximate solutions.

Thus, solving a variational problem by the direct method requires three steps:

1. Construct a minimizing sequence $\lbrace y\_n \rbrace$.
2. Show that $\lbrace y\_n \rbrace$ has a limit function $\hat{y}$.
3. Justify passing the functional through the limit.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1</span><span class="math-callout__name">(Existence of Minimizing Sequences)</span></p>

A minimizing sequence can always be constructed whenever $\mu > -\infty$. The two principal direct methods for building such sequences — the *Ritz method* and the *method of finite differences* — are described in Sec. 40.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2</span><span class="math-callout__name">(A Minimizing Sequence May Have No Limit)</span></p>

Even when a minimizing sequence exists, it may fail to converge to a limit function. For instance, consider

$$J[y] = \int_{-1}^{1} x^2 y'^2\,dx, \qquad y(-1) = -1,\quad y(1) = 1.$$

Since $J[y]$ is always nonnegative, $\inf J[y] = 0$. The sequence

$$y_n(x) = \frac{\tan^{-1}(nx)}{\tan^{-1}(n)} \qquad (n = 1, 2, \ldots)$$

is a minimizing sequence with $J[y\_n] \to 0$. However, this sequence has no limit in the class of continuous functions satisfying the boundary conditions.

</div>

Even when a minimizing sequence does converge in the $\mathscr{C}$-norm (i.e., uniformly), passing the functional through the limit is nontrivial because typical functionals in the calculus of variations are not continuous in the $\mathscr{C}$-norm. However, the interchange is still valid under a weaker condition: it suffices for $J[y]$ to be *lower semicontinuous*.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Lower Semicontinuity and Minimizing Sequences)</span></p>

If $\lbrace y\_n \rbrace$ is a minimizing sequence of the functional $J[y]$ with limit function $\hat{y}$, and if $J[y]$ is lower semicontinuous at $\hat{y}$, then

$$J[\hat{y}] = \lim_{n \to \infty} J[y_n].$$

</div>

*Sketch of proof.* On one hand, $J[\hat{y}] \geqslant \lim\_{n\to\infty} J[y\_n] = \inf J[y]$, since $\hat{y}$ is admissible. On the other hand, for every $\varepsilon > 0$ and $n$ large enough, lower semicontinuity gives $J[y\_n] - J[\hat{y}] > -\varepsilon$, so letting $n \to \infty$ yields $J[\hat{y}] \leqslant \lim\_{n\to\infty} J[y\_n]$. Combining both inequalities gives equality.

### 40. The Ritz Method and the Method of Finite Differences

#### 40.1. The Ritz Method

The Ritz method is one of the most widely used direct variational methods. Suppose we want to minimize $J[y]$ on a normed linear space $\mathscr{M}$ of admissible functions. Choose an infinite sequence of functions

$$\varphi_1, \quad \varphi_2, \quad \ldots$$

in $\mathscr{M}$, and let $\mathscr{M}\_n$ be the $n$-dimensional subspace spanned by the first $n$ of these functions, consisting of all linear combinations

$$\alpha_1 \varphi_1 + \cdots + \alpha_n \varphi_n.$$

Restricting $J[y]$ to $\mathscr{M}\_n$ turns it into an ordinary function of $n$ real variables:

$$J[\alpha_1 \varphi_1 + \cdots + \alpha_n \varphi_n],$$

which we minimize over $\alpha\_1, \ldots, \alpha\_n$. Denote the minimum value on $\mathscr{M}\_n$ by $\mu\_n$ and the minimizing element by $y\_n$. Since $\mathscr{M}\_n \subset \mathscr{M}\_{n+1}$, the minima form a non-increasing sequence:

$$\mu_1 \geqslant \mu_2 \geqslant \cdots$$

The key question is whether this procedure actually produces a minimizing sequence.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Complete Sequence)</span></p>

The sequence $\lbrace \varphi\_n \rbrace$ is said to be **complete** in $\mathscr{M}$ if, given any $y \in \mathscr{M}$ and any $\varepsilon > 0$, there exists a linear combination $\eta\_n = \alpha\_1 \varphi\_1 + \cdots + \alpha\_n \varphi\_n$ (for some $n$ depending on $\varepsilon$) such that $\lVert \eta\_n - y \rVert < \varepsilon$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Convergence of the Ritz Method)</span></p>

If the functional $J[y]$ is continuous (in the norm of $\mathscr{M}$) and the sequence $\lbrace \varphi\_n \rbrace$ is complete, then $\lim\_{n \to \infty} \mu\_n = \mu$, where $\mu = \inf\_y J[y]$.

</div>

*Sketch of proof.* Given $\varepsilon > 0$, pick $y^\ast$ with $J[y^\ast] < \mu + \varepsilon$. By continuity, $\lvert J[y] - J[y^\ast] \rvert < \varepsilon$ whenever $\lVert y - y^\ast \rVert < \delta$. By completeness, there exists a linear combination $\eta\_n$ in $\mathscr{M}\_n$ with $\lVert \eta\_n - y^\ast \rVert < \delta$. Then

$$\mu \leqslant J[y_n] \leqslant J[\eta_n] < \mu + 2\varepsilon.$$

Since $\varepsilon$ is arbitrary, $\mu\_n \to \mu$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1</span><span class="math-callout__name">(Geometric Interpretation)</span></p>

If $\lbrace \varphi\_n \rbrace$ is complete, every element of the infinite-dimensional space $\mathscr{M}$ can be approximated arbitrarily closely by elements of the finite-dimensional subspaces $\mathscr{M}\_n$, i.e., $\lim\_{n \to \infty} \mathscr{M}\_n = \mathscr{M}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2</span><span class="math-callout__name">(Rate of Convergence)</span></p>

The speed of convergence of the Ritz method depends on the problem and on the choice of the basis functions $\varphi\_n$. In many cases, a very small number of basis functions already gives a remarkably good approximation.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3</span><span class="math-callout__name">(Generalized Setting)</span></p>

The spaces $\mathscr{M}$ and $\mathscr{M}\_n$ need not themselves be normed linear spaces — it suffices that the admissible functions belong to a normed linear space $\mathscr{R}$. For example, one may impose boundary conditions $y(a) = A$, $y(b) = B$ or subsidiary integral conditions such as $\int\_a^b y^2\,dx = 1$, with appropriate modifications to the Ritz procedure.

</div>

#### 40.2. The Method of Finite Differences

An alternative direct method approximates the variational problem by a finite-dimensional optimization problem through discretization. To find an extremum of

$$J[y] = \int_a^b F(x, y, y')\,dx, \qquad y(a) = A,\quad y(b) = B,$$

we subdivide $[a, b]$ into $n + 1$ equal parts by introducing nodes

$$x_0 = a,\quad x_1, \ldots, x_n,\quad x_{n+1} = b, \qquad x_{i+1} - x_i = \Delta x,$$

and replace the unknown function $y(x)$ by a polygonal line through the vertices $(x\_0, y\_0), (x\_1, y\_1), \ldots, (x\_{n+1}, y\_{n+1})$, where $y\_i = y(x\_i)$, and $y\_0 = A$, $y\_{n+1} = B$ are fixed. The functional is then approximated by the sum

$$J(y_1, \ldots, y_n) = \sum_{i=0}^{n} F\!\left(x_i,\; y_i,\; \frac{y_{i+1} - y_i}{\Delta x}\right) \Delta x,$$

which is an ordinary function of the $n$ free variables $y\_1, \ldots, y\_n$. Minimizing this function for each $n$ yields a sequence of polygonal approximations to the solution of the original variational problem.

### 41. The Sturm–Liouville Problem

As an application of direct variational methods to differential equations, we study the *Sturm–Liouville problem*: given $P(x) > 0$ continuously differentiable and $Q(x)$ continuous, find the nontrivial solutions (eigenfunctions) and corresponding eigenvalues $\lambda$ of

$$-(Py')' + Qy = \lambda y, \tag{SL}$$

subject to the boundary conditions $y(a) = 0$, $y(b) = 0$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Sturm–Liouville Eigenvalue Problem)</span></p>

The Sturm–Liouville problem $\text{(SL)}$ with $y(a) = 0$, $y(b) = 0$ possesses an infinite sequence of eigenvalues $\lambda^{(1)}, \lambda^{(2)}, \ldots$, and to each eigenvalue $\lambda^{(n)}$ there corresponds an eigenfunction $y^{(n)}$ which is unique up to a constant factor.

</div>

The proof is carried out in stages using the Ritz method. The central observation is that the Euler equation of the quadratic functional

$$J[y] = \int_a^b (Py'^2 + Qy^2)\,dx$$

subject to $y(a) = 0$, $y(b) = 0$ and the normalization $\int\_a^b y^2\,dx = 1$ is precisely the Sturm–Liouville equation $\text{(SL)}$, and the extremal value of $J$ equals the corresponding eigenvalue $\lambda$.

#### 41.1. Setting Up the Ritz Approximation

Since $P(x) > 0$, the functional $J[y]$ is bounded from below. For simplicity, take $a = 0$, $b = \pi$ and use the complete orthogonal system $\lbrace \sin nx \rbrace$ as the Ritz basis, which automatically satisfies the boundary conditions. An admissible linear combination

$$y(x) = \sum_{k=1}^{n} \alpha_k \sin kx$$

satisfies the normalization constraint $\int\_0^\pi y^2\,dx = 1$ if and only if $\frac{\pi}{2}\sum\_{k=1}^{n} \alpha\_k^2 = 1$, so the coefficients lie on an $n$-dimensional sphere $\sigma\_n$. On $\sigma\_n$, the functional becomes

$$J_n(\alpha_1, \ldots, \alpha_n) = \int_0^\pi \!\left[P(x)\!\left(\sum_{k=1}^{n} \alpha_k \cos kx \cdot k\right)^{\!2} + Q(x)\!\left(\sum_{k=1}^{n} \alpha_k \sin kx\right)^{\!2}\right] dx,$$

a quadratic form in $\alpha\_1, \ldots, \alpha\_n$, which attains a minimum $\lambda\_n^{(1)}$ on the compact set $\sigma\_n$. Since $\sigma\_n \subset \sigma\_{n+1}$ (by setting $\alpha\_{n+1} = 0$), the minima form a non-increasing sequence

$$\lambda_{n+1}^{(1)} \leqslant \lambda_n^{(1)},$$

and because $J[y]$ is bounded from below, the limit

$$\lambda^{(1)} = \lim_{n \to \infty} \lambda_n^{(1)}$$

exists and is finite.

#### 41.2. Convergence of the Eigenfunctions

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1</span><span class="math-callout__name">(Uniform Convergence of a Subsequence)</span></p>

The sequence of Ritz minimizers $\lbrace y\_n^{(1)}(x) \rbrace$ contains a uniformly convergent subsequence.

</div>

*Sketch of proof.* From the boundedness of $J[y\_n]$ and the positivity of $P$, one obtains uniform bounds on both $\int y\_n'^2\,dx$ and $\int y\_n^2\,dx$. By Schwarz's inequality and the boundary condition $y\_n(0) = 0$, the sequence $\lbrace y\_n \rbrace$ is uniformly bounded. A second application of Schwarz's inequality shows equicontinuity:

$$\lvert y_n(x_2) - y_n(x_1) \rvert^2 \leqslant M_2 \lvert x_2 - x_1 \rvert.$$

By the Arzelà–Ascoli theorem, a uniformly convergent subsequence exists.

We set $y^{(1)}(x) = \lim\_{m \to \infty} y\_{n\_m}(x)$, where $\lbrace y\_{n\_m} \rbrace$ is the convergent subsequence.

#### 41.3. The Limit Function Solves the Sturm–Liouville Equation

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2</span><span class="math-callout__name">(Variational Characterization)</span></p>

Let $y(x)$ be continuous on $[0, \pi]$ and suppose

$$\int_0^\pi [-(Ph')' + Q_1 h]\,y\,dx = 0$$

for every $h(x) \in \mathscr{D}\_2(0, \pi)$ with $h(0) = h(\pi) = 0$ and $h'(0) = h'(\pi) = 0$. Then $y \in \mathscr{D}\_2(0, \pi)$ and $-(Py')' + Q\_1 y = 0$.

</div>

Using this lemma, one shows that the limit function $y^{(1)}$ belongs to $\mathscr{D}\_2(0, \pi)$ and satisfies the Sturm–Liouville equation with $\lambda = \lambda^{(1)}$:

$$-(Py^{(1)})' + Qy^{(1)} = \lambda^{(1)} y^{(1)}.$$

The proof proceeds by showing that the Ritz optimality conditions for each finite-dimensional approximation pass to the limit. In particular, convergence in the mean of $y\_n \to y^{(1)}$, $y\_n' \to (y^{(1)})'$, and $y\_n'' \to (y^{(1)})''$ is established, which allows the integral identity to transfer to the limit function.

#### 41.4. Higher Eigenvalues and Eigenfunctions

Having found the first eigenvalue $\lambda^{(1)}$ and eigenfunction $y^{(1)}$, the subsequent eigenvalues are obtained by adding orthogonality constraints. The second eigenvalue $\lambda^{(2)}$ is found by minimizing the same quadratic functional

$$J[y] = \int_0^\pi (Py'^2 + Qy^2)\,dx$$

subject to the same boundary and normalization conditions as before, plus the additional orthogonality condition

$$\int_0^\pi y^{(1)}(x)\,y(x)\,dx = 0.$$

In the Ritz framework with basis $\lbrace \sin kx \rbrace$, this orthogonality constraint cuts the sphere $\sigma\_n$ down to a lower-dimensional sphere $\hat{\sigma}\_{n-1}$. The minimum $\lambda\_n^{(2)}$ on $\hat{\sigma}\_{n-1}$ satisfies $\lambda\_{n+1}^{(2)} \leqslant \lambda\_n^{(2)}$ (non-increasing in $n$), and the limit

$$\lambda^{(2)} = \lim_{n \to \infty} \lambda_n^{(2)}$$

exists with $\lambda^{(1)} \leqslant \lambda^{(2)}$. The corresponding limit function $y^{(2)}$ solves the Sturm–Liouville equation with eigenvalue $\lambda^{(2)}$ and is orthogonal to $y^{(1)}$.

Since the eigenfunctions are orthogonal, they cannot be linearly dependent, and because each eigenvalue admits only one eigenfunction (up to a constant factor), we obtain the strict ordering

$$\lambda^{(1)} < \lambda^{(2)}.$$

Repeating this procedure — minimizing $J[y]$ on successively smaller subspaces defined by orthogonality to all previously found eigenfunctions — yields an infinite sequence of eigenvalues $\lambda^{(1)} < \lambda^{(2)} < \lambda^{(3)} < \cdots$ and corresponding eigenfunctions $y^{(1)}, y^{(2)}, y^{(3)}, \ldots$, completing the proof of the theorem.

## Appendix I: Propagation of Disturbances and the Canonical Equations

This appendix develops the basic concepts of the calculus of variations — the canonical equations, the Hamiltonian function, and the Hamilton–Jacobi equation — from a purely geometric model of wave propagation in an inhomogeneous, anisotropic medium. The key assumptions are:

1. Each point can be in only one of two states, *excitation* or *rest* — there is no concept of intensity.
2. If a disturbance arrives at a point $P$ at time $t$, then from time $t$ onward, $P$ itself serves as a source of further disturbances propagating in the medium.

This construction is essentially a mathematical formulation of *Huygens' principle*.

### I.1. Statement of the Problem

Let the medium fill an $n$-dimensional Euclidean space $\mathscr{X}$, so that every point $x \in \mathscr{X}$ is specified by $n$ real numbers $x^1, \ldots, x^n$. Fix a point $x\_0 \in \mathscr{X}$ and consider all smooth curves

$$x = x(s)$$

passing through $x\_0$. The set of vectors tangent to such a curve at $x\_0$, i.e., the set of vectors

$$x' = \frac{dx}{ds},$$

forms the *tangent space* $\mathscr{T}(x\_0)$.

Since the medium is inhomogeneous and anisotropic, the velocity of propagation depends on both position and direction. Let $f(x, x')$ denote the reciprocal of the velocity. Then if $x(s)$ and $x(s + ds)$ are two neighboring points on a curve $x = x(s)$, the time $dt$ for the disturbance to travel between them is

$$dt = f\!\left(x, \frac{dx}{ds}\right) ds,$$

and the total propagation time along an infinite path joining $x\_0 = x(s\_0)$ and $x\_1 = x(s\_1)$ is

$$\int_{s_0}^{s_1} f\!\left(x, \frac{dx}{ds}\right) ds.$$

Because only the fastest path matters (excited points are "on or off"), the propagation time from $x\_0$ to $x\_1$ is

$$\tau = \min \int_{s_0}^{s_1} f\!\left(x, \frac{dx}{ds}\right) ds,$$

where the minimum is taken over all curves $x = x(s)$ joining $x\_0$ and $x\_1$. Thus, disturbances in the medium obey the *Fermat principle*: among all paths joining $x\_0$ and $x\_1$, the disturbance always propagates along the path which it traverses in the least time. Such paths are called the *trajectories* of the disturbance.

### I.2. Properties of the Function $f(x, x')$

We state physically plausible properties for $f(x, x')$:

1. The propagation time along any curve is positive:

$$f(x, x') > 0 \quad \text{if } x' \neq 0.$$

2. The propagation time depends only on the curve $\gamma$ and not on its parameterization. It follows that $f(x, x')$ is positive-homogeneous of degree 1 in $x'$:

$$f(x, \lambda x') = \lambda f(x, x') \quad \text{for every } \lambda > 0.$$

   In particular, this implies the *decomposition property*

$$f(x, x' + \bar{x}') = f(x, x') + f(x, \bar{x}'),$$

   whenever $\bar{x}' = \lambda x'$ with $\lambda > 0$.

3. The time to traverse a curve $\gamma$ from $x\_0$ to $x\_1$ equals the time to traverse $\gamma$ in the opposite direction from $x\_1$ to $x\_0$:

$$f(x, -x') = f(x, x').$$

4. In a homogeneous medium, $f$ depends on direction only, and the disturbance propagates in straight lines. The *convexity condition* then holds:

$$f(x' + \bar{x}') \leqslant f(x') + f(\bar{x}').$$

   More generally, if $f$ depends on $x$ smoothly, the convexity condition becomes

$$f(x, x' + \bar{x}') \leqslant f(x, x') + f(x, \bar{x}').$$

5. We strengthen this to the *strict convexity condition*: equality in the above holds only if $\bar{x}' = \lambda x'$, where $\lambda > 0$.

### I.3. Introduction of a Norm in $\mathscr{T}(x)$

We use $f(x, x')$ to introduce a norm in the $n$-dimensional tangent space $\mathscr{T}(x)$ by setting

$$\|x'\| = f(x, x')$$

for all vectors $x' \neq 0$ in $\mathscr{T}(x)$. The requirements for a norm — positivity, homogeneity, and the triangle inequality — are immediate consequences of properties (1), (4)/(7), and (6) above.

The set of all vectors in $\mathscr{T}(x)$ with $f(x, x') = \|x'\| = \alpha$ is called a *sphere of radius $\alpha$* in $\mathscr{T}(x)$, with center at the point $x$. The sphere of radius $\alpha$ is the boundary of the closed region of $\mathscr{T}(x)$ (and hence of $\mathscr{X}$) excited during the time $\alpha$ by a disturbance originally concentrated at $x$.

Our problem can now be rephrased as follows: *Suppose a tangent space $\mathscr{T}(x)$, equipped with the norm $\|x'\| = f(x, x')$ satisfying the strict convexity condition, is defined at each point $x$ of the $n$-dimensional space $\mathscr{X}$. Find the equations describing the propagation of disturbances in $\mathscr{X}$, if during the time $dt$ the disturbance originally at $x$ "spreads out and fills" the sphere*

$$f(x, dx) = dt.$$

### I.4. The Conjugate Space $\widetilde{\mathscr{T}}(x)$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Conjugate Space)</span></p>

Let $\varphi[x']$ be a linear functional on $\mathscr{T}(x)$. Then there is a unique vector $p = (p\_1, \ldots, p\_n)$ such that

$$\varphi[x'] = (p, x') = \sum_{i=1}^n p_i x^{i'}.$$

The set of all such vectors $p$, equivalently the set of all linear functionals on $\mathscr{T}(x)$, is itself an $n$-dimensional linear space called the *conjugate space* of $\mathscr{T}(x)$ and denoted $\widetilde{\mathscr{T}}(x)$.

</div>

We define the norm of a vector $p \in \widetilde{\mathscr{T}}(x)$ by

$$\|p\| = \sup_{x'} \frac{(p, x')}{\|x'\|},$$

where the supremum is over all $x' \neq 0$ in $\mathscr{T}(x)$. Writing $H(x, p)$ instead of $\|p\|$:

$$H(x, p) = \sup_{x'} \frac{(p, x')}{\|x'\|}.$$

The transition from $f(x, x')$ to $H(x, p)$ via this formula is precisely the parametric form of the *Legendre transformation* (see Sec. 18).

### I.5. The Propagation Process

Suppose the wave front at time $t$ is the surface $\sigma\_t$ with equation

$$S(x, t) = 0.$$

Each point of $\sigma\_t$ serves as a source of new disturbances, which during time $dt$ excite the region bounded by the sphere

$$f(x, dx) = dt.$$

Since $f(x, x')$ is differentiable and strictly convex, there is a unique hyperplane tangent to each point of the sphere, and this hyperplane has only one point in common with the sphere. Constructing a family of spheres, one for each $x \in \sigma\_t$, the wave front $\sigma\_{t+dt}$ at time $t + dt$ has the equation

$$S(x, t + dt) = 0$$

and is the *envelope* $E$ of this family of spheres — the interface separating points reachable from $\sigma\_t$ in times $\leqslant dt$ from those reachable only in times $> dt$.

Two important implications follow:

1. Given $x \in \sigma\_t$, there is a unique point $x + dx \in \sigma\_{t+dt}$ excited after time $dt$ by a disturbance at $x$, namely the point lying on the unique hyperplane tangent to both $\sigma\_{t+dt}$ and the sphere. Thus there is a unique *direction of propagation* at each $x \in \sigma\_t$.
2. Conversely, given $x + dx \in \sigma\_{t+dt}$, there is a unique point $x \in \sigma\_t$ that was the source at time $t$.

### I.6. The Hamilton–Jacobi Equation

Every hyperplane in the tangent space $\mathscr{T}(x)$ can be written as $\sum p\_i x^{i'} = \text{const}$ for some $p \in \widetilde{\mathscr{T}}(x)$. Let $x + dx$ be the point on $\sigma\_{t+dt}$ whose "source" is $x \in \sigma\_t$. The hyperplane in $\mathscr{T}(x)$ tangent to $\sigma\_{t+dt}$ at $x + dx$ has the equation

$$\sum_{i=1}^n \frac{\partial S}{\partial x^i}\,dx^i = c,$$

where, if the hyperplane is also tangent to the sphere $f(x, dx) = dt$, then $c$ equals the norm of the gradient vector $\nabla S$ multiplied by the radius of the sphere:

$$c = H(x, \nabla S)\,dt.$$

Therefore

$$\sum_{i=1}^n \frac{\partial S}{\partial x^i}\,dx^i = H(x, \nabla S)\,dt.$$

But since $S$ is constant on the wave front,

$$\sum_{i=1}^n \frac{\partial S}{\partial x^i}\,dx^i + \frac{\partial S}{\partial t}\,dt = 0.$$

Comparing these two expressions, we obtain the **Hamilton–Jacobi equation**:

$$\frac{\partial S}{\partial t} + H(x, \nabla S) = 0.$$

### I.7. Relation Between Trajectories and Wave Fronts

The general solution of the Hamilton–Jacobi equation describes how the wave front evolves in time. As the front evolves, each of its points traces out a succession of uniquely defined points on neighboring fronts, thereby "sweeping out" a trajectory $\gamma$ which automatically minimizes the functional $\int f(x, dx/ds)\,ds$. If we specify a one-parameter family of wave fronts

$$S(x, t) = 0,$$

where $t$ is the parameter, every point $x\_0$ on some initial surface generates a trajectory, and the one-parameter family determines an $(n - 1)$-parameter family of trajectories, one through each point of $\mathscr{X}$. More generally, let

$$S(x, t, \alpha_1, \ldots, \alpha_n)$$

be a *complete integral* of the Hamilton–Jacobi equation depending on $n$ parameters $\alpha\_1, \ldots, \alpha\_n$. This determines an $(n + 1)$-parameter family of surfaces

$$S(x, t, \alpha_1, \ldots, \alpha_n) = 0,$$

which in turn determines a $(2n - 1)$-parameter family of trajectories. Since the trajectories of the disturbances are the extremals of the functional $\int f\,ds$, this leads to a geometric interpretation of Jacobi's theorem (p. 91) concerning the construction of the general solution of the Euler equations from a complete integral of the corresponding Hamilton–Jacobi equation.

### I.8. The Canonical Equations

To derive the differential equations satisfied by the trajectories, we use $t$ as the parameter along each trajectory. From $f(x, dx) = dt$ and the homogeneity of $f$ in $dx$:

$$f\!\left(x, \frac{dx}{dt}\right) = 1,$$

so the norm of $dx/dt$ is identically 1. Using the tangency relation between the trajectory and the wave front, the covariant vector $p$ (determining the hyperplane tangent to the wave front) is related to $dx/dt$ by

$$\sum_{i=1}^n p_i \frac{dx^i}{dt} = H(x, p).$$

By the definition of $H$ in $\widetilde{\mathscr{T}}(x)$, the expression

$$\sum_{i=1}^n p_i \frac{dx^i}{dt} - H(x, p),$$

regarded as a function of $p$, achieves its maximum when $p$ is the vector determining the hyperplane tangent to the wave front. Therefore, along the trajectories,

$$\frac{dx^i}{dt} = \frac{\partial H(x, p)}{\partial p_i} \qquad (i = 1, \ldots, n).$$

This gives $n$ first-order equations. To find the remaining $n$ equations for $p\_1, \ldots, p\_n$, note that along each trajectory $p\_i(t) = \partial S / \partial x^i$, and hence

$$\frac{dp_i}{dt} = \frac{d}{dt}\frac{\partial S}{\partial x^i} = \frac{\partial}{\partial t}\frac{\partial S}{\partial x^i} + \sum_{k=1}^n \frac{\partial^2 S}{\partial x^k \partial x^i}\frac{dx^k}{dt}.$$

Using the Hamilton–Jacobi equation $\partial S / \partial t = -H(x, \nabla S)$ and simplifying, one obtains

$$\frac{dp_i}{dt} = -\frac{\partial H(x, p)}{\partial x^i} \qquad (i = 1, \ldots, n).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Canonical Equations / Hamilton's Equations)</span></p>

Combining these, the trajectories of the disturbance satisfy the system of $2n$ first-order ordinary differential equations

$$\frac{dx^i}{dt} = \frac{\partial H(x, p)}{\partial p_i}, \qquad \frac{dp_i}{dt} = -\frac{\partial H(x, p)}{\partial x^i},$$

where $i = 1, \ldots, n$. The integral curves of this system are precisely the extremals of the variational functional $\int f(x, dx/ds)\,ds$.

</div>

This is the *canonical system* of Euler equations for the variational problem, and represents the *characteristic system* associated with the Hamilton–Jacobi equation (cf. Sec. 16, p. 90).

## Appendix II: Variational Methods in Problems of Optimal Control

This appendix sketches results obtained by L. S. Pontryagin and his students on the theory of *optimal control processes*, and discusses the connection between optimal control and classical variational theory.

### II.1. Statement of the Problem

Suppose the state of a physical system is characterized by $n$ real numbers $x^1, \ldots, x^n$, forming a vector $x = (x^1, \ldots, x^n)$ in $n$-dimensional "phase space" $\mathscr{X}$, and the state evolves according to the system of differential equations

$$\frac{dx^i}{dt} = f^i(x^1, \ldots, x^n, u^1, \ldots, u^k) \qquad (i = 1, \ldots, n),$$

where $u = (u^1, \ldots, u^k) \in \Omega$ is the *control vector*, belonging to a fixed "control region" $\Omega \subset \mathbb{R}^k$, and the $f^i(x, u)$ are continuous functions defined for all $x \in \mathscr{X}$ and $u \in \Omega$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Control Function, Control Process, Trajectory)</span></p>

A *control function* is a vector function $u(t)$ with values in $\Omega$, defined for $t\_0 \leqslant t \leqslant t\_1$. Substituting $u = u(t)$ into the system gives

$$\frac{dx^i}{dt} = f^i[x^1, \ldots, x^n, u^1(t), \ldots, u^k(t)] \qquad (i = 1, \ldots, n).$$

For every initial value $x\_0 = x(t\_0)$, this system has a definite solution called a *trajectory*. The aggregate

$$U = \lbrace u(t),\, t_0,\, t_1,\, x_0 \rbrace$$

is called a *control process*.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Optimal Control)</span></p>

Let

$$f^0(x^1, \ldots, x^n, u^1, \ldots, u^k)$$

be a function defined, together with its partial derivatives $\partial f^0 / \partial x^i$, for all $x \in \mathscr{X}$ and $u \in \Omega$. To every control process $U$, we assign the number

$$J[U] = \int_{t_0}^{t_1} f^0(x, u)\,dt.$$

The control process $U$ (with corresponding trajectory carrying $x\_0$ into $x\_1$) is said to be *optimal* if

$$J[U] \leqslant J[U^*]$$

for any other control process $U^\ast$ carrying $x\_0$ into $x\_1$. The corresponding trajectory is called the *optimal trajectory*.

</div>

The components $u^1(t), \ldots, u^k(t)$ of any admissible control process take values in $\Omega$, are bounded, and are piecewise continuous (with left- and right-hand limits at every point of discontinuity).

An important special case is the *time-optimal problem*, where $J[U] = \int\_{t\_0}^{t\_1} dt = t\_1 - t\_0$, and optimality means reaching $x\_1$ from $x\_0$ in the least time.

### II.2. Relation to the Calculus of Variations

The integral $\int\_{t\_0}^{t\_1} f^0(x, u)\,dt$ can be viewed as a functional depending on the $n + k$ functions $x^1, \ldots, x^n, u^1, \ldots, u^k$, i.e., a functional defined on some class of curves in $n + k + 1$ dimensions. Since the $x^i$ and $u^\alpha$ are connected by the equations of motion, this is a problem of finding a minimum subject to *nonholonomic constraints* (see p. 48). The boundary conditions require the optimal trajectory $x(t)$ to pass through the fixed endpoints $x\_0$ and $x\_1$, which means the endpoints of the admissible curves must lie on two $(k+1)$-dimensional hyperplanes.

The problem of optimal control is thus a variant of finding a minimum subject to subsidiary conditions, with the special feature that the control functions $u^1(t), \ldots, u^k(t)$ take values in a fixed region $\Omega$ but are in general *not required to be continuous*.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Classical Variational Problems as Special Cases)</span></p>

When the integrand does not depend on $t$ explicitly, the simplest $n$-dimensional variational problem of finding the curve through two fixed points minimizing

$$\int_{t_0}^{t_1} f^0\!\left(x^1, \ldots, x^n, \frac{dx^1}{dt}, \ldots, \frac{dx^n}{dt}\right) dt$$

is a special case of optimal control. One simply takes the system $dx^i/dt = u^i$ for $i = 1, \ldots, n$ with no constraint on $u$.

</div>

### II.3. Necessary Conditions for Optimality

To find necessary conditions for a given control process and trajectory to be optimal, we supplement the system $dx^i/dt = f^i(x, u)$ with the extra equation

$$\frac{dx^0}{dt} = f^0(x, u),$$

where $f^0(x, u)$ is the integrand of $J[U]$. We impose the initial conditions

$$x^i(t_0) = x_0^i \qquad (i = 1, \ldots, n), \qquad x^0(t_0) = 0.$$

Introducing the $(n+1)$-dimensional vector $\mathbf{x}(t) = (x^0(t), x^1(t), \ldots, x^n(t))$, the augmented system becomes

$$\frac{dx^i}{dt} = f^i(x, u) \qquad (i = 0, 1, \ldots, n),$$

and $J[U] = \int\_{t\_0}^{t\_1} f^0(x, u)\,dt = x^0(t\_1)$. The problem reduces to finding the admissible control $U$ for which $x^0(t\_1)$ is as small as possible.

Next, we introduce new variables $\psi\_0, \psi\_1, \ldots, \psi\_n$ satisfying the *conjugate* (or *adjoint*) system of differential equations

$$\frac{d\psi_i}{dt} = -\sum_{\alpha=0}^{n} \frac{\partial f^\alpha(x, u)}{\partial x^i}\,\psi_\alpha \qquad (i = 0, 1, \ldots, n).$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Function $\Pi$ and the Hamiltonian $\mathscr{H}$)</span></p>

Let $\boldsymbol{\psi}(t) = (\psi\_0(t), \psi\_1(t), \ldots, \psi\_n(t))$ and consider the function

$$\Pi(\boldsymbol{\psi}, x, u) = \sum_{\alpha=0}^{n} \psi_\alpha f^\alpha(x, u).$$

In terms of $\Pi$, the equations of motion and the conjugate system can be written as

$$\frac{dx^i}{dt} = \frac{\partial \Pi}{\partial \psi_i}, \qquad \frac{d\psi_i}{dt} = -\frac{\partial \Pi}{\partial x^i} \qquad (i = 0, 1, \ldots, n).$$

These resemble the canonical equations of Euler (cf. Sec. 16, formula (11), p. 70), but form a closed system only when $u$ is specified. To close the system, define the *Hamiltonian*

$$\mathscr{H}(\boldsymbol{\psi}, x) = \sup_{u \in \Omega}\, \Pi(\boldsymbol{\psi}, x, u).$$

</div>

### II.4. The Maximum Principle

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Pontryagin's Maximum Principle)</span></p>

Let $U = \lbrace u(t), t\_0, t\_1, x\_0 \rbrace$ be an admissible control process, and let $\mathbf{x}(t)$ be the corresponding integral curve of the augmented system passing through $(0, x\_0^1, \ldots, x\_0^n)$ for $t = 0$, and satisfying $x^i(t\_1) = x\_1^i$ for $i = 1, \ldots, n$. If the control process $U$ is optimal, then there exists a continuous vector function $\boldsymbol{\psi}(t) = (\psi\_0(t), \psi\_1(t), \ldots, \psi\_n(t))$ such that:

1. $\boldsymbol{\psi}(t)$ satisfies the conjugate system for $x = \mathbf{x}(t)$, $u = u(t)$;
2. For all $t$ in $[t\_0, t\_1]$, the function $\Pi(\boldsymbol{\psi}(t), \mathbf{x}(t), u(t))$ achieves its maximum for $u = u(t)$, i.e.,

$$\Pi[\boldsymbol{\psi}(t), \mathbf{x}(t), u(t)] = \mathscr{H}[\boldsymbol{\psi}(t), \mathbf{x}(t)];$$

3. The relations

$$\psi_0(t_1) \leqslant 0, \qquad \mathscr{H}[\boldsymbol{\psi}(t_1), \mathbf{x}(t_1)] = 0$$

   hold at the time $t\_1$. In fact, since $\psi\_0$ and $\mathscr{H}$ turn out to be constants along the optimal trajectory, we can replace $t\_1$ by any $t \in [t\_0, t\_1]$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Using the Maximum Principle)</span></p>

The maximum principle can often be used as a prescription for constructing the optimal trajectory: for every fixed $\boldsymbol{\psi}$ and $x$, we find the value of $u$ for which

$$\sum_{\alpha=0}^{n} \psi_\alpha f^\alpha(x, u)$$

achieves its maximum. If this determines $u$ as a single-valued function $u = u(\boldsymbol{\psi}, x)$, then substituting into the equations of motion and the conjugate system yields a closed system of $2(n+1)$ equations in $2(n+1)$ unknowns — precisely the equations satisfied by the optimal trajectory.

</div>

### II.5. Relation to Weierstrass' Necessary Condition

For the simple variational functional

$$\int_{t_0}^{t_1} f^0(x^1, \ldots, x^n, x^{1'}, \ldots, x^{n'})\,dt,$$

where $u^i = dx^i/dt$ and $\Omega = \mathbb{R}^n$, the function $\Pi$ becomes

$$\Pi(\boldsymbol{\psi}, x, u) = \psi_0 f^0(x, u) + \sum_{\alpha=1}^{n} \psi_\alpha u^\alpha.$$

The Weierstrass $E$-function for this functional is

$$E(x, x', z) = f^0(x, z) - f^0(x, x') - \sum_{i=1}^{n} (z_i - x_i') f^0_{z_i}(x, x').$$

One can show that

$$\psi_0 E = \Pi(\boldsymbol{\psi}, x, z) - \Pi(\boldsymbol{\psi}, x, x') - \sum_{i=1}^{n} (z_i - x^{i'}) \frac{\partial}{\partial u^i}\Pi(\boldsymbol{\psi}, x, x').$$

If $\Pi$ achieves its maximum for values of $u$ which are interior points of $\Omega$, then $\partial \Pi / \partial u^i = 0$ at these points, and condition (2) of the maximum principle (that $\Pi$ is maximized at $u = x'$) is equivalent to

$$E(x, x', z) \geqslant 0.$$

This is precisely *Weierstrass' necessary condition* (see p. 149). Thus the maximum principle and Weierstrass' necessary condition are equivalent when the control region $\Omega$ is open. When $u(t)$ takes values on the boundary of $\Omega$, the Weierstrass condition is in general no longer valid, but the maximum principle continues to apply.
