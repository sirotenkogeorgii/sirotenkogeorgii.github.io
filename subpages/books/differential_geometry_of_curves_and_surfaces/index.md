---
layout: default
title: Differential Geometry of Curves and Surfaces
date: 2026-03-17
excerpt: Notes on the classical differential geometry of curves and surfaces following do Carmo.
tags:
  - differential-geometry
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

# Chapter 1 — Curves

## 1-1. Introduction

The differential geometry of curves and surfaces has two aspects:

- **Classical differential geometry** studies *local* properties — those that depend only on the behavior of the curve or surface in a neighborhood of a point. The methods of differential calculus are the natural tools here, so all curves and surfaces will be defined by functions that can be differentiated sufficiently many times.

- **Global differential geometry** studies the influence of local properties on the behavior of the entire curve or surface.

The most interesting part of classical differential geometry is the study of surfaces. However, some local properties of curves appear naturally while studying surfaces, so the first chapter provides a brief treatment of curves.

## 1-2. Parametrized Curves

We denote by $R^3$ the set of triples $(x, y, z)$ of real numbers. Our goal is to characterize certain subsets of $R^3$ (to be called curves) that are, in a certain sense, one-dimensional and to which the methods of differential calculus can be applied.

A natural way of defining such subsets is through differentiable functions. We say that a real function of a real variable is *differentiable* (or *smooth*) if it has, at all points, derivatives of all orders (which are automatically continuous).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Parametrized Differentiable Curve)</span></p>

A **parametrized differentiable curve** is a differentiable map $\alpha\colon I \to R^3$ of an open interval $I = (a, b)$ of the real line $R$ into $R^3$.

</div>

The word *differentiable* means that $\alpha$ is a correspondence which maps each $t \in I$ into a point $\alpha(t) = (x(t), y(t), z(t)) \in R^3$ in such a way that the functions $x(t)$, $y(t)$, $z(t)$ are differentiable. The variable $t$ is called the **parameter** of the curve. The word *interval* is taken in a generalized sense, so that we do not exclude the cases $a = -\infty$, $b = +\infty$.

The vector $(x'(t), y'(t), z'(t)) = \alpha'(t) \in R^3$ is called the **tangent vector** (or **velocity vector**) of the curve $\alpha$ at $t$. The image set $\alpha(I) \subset R^3$ is called the **trace** of $\alpha$. One should carefully distinguish a parametrized curve, which is a map, from its trace, which is a subset of $R^3$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Helix)</span></p>

The parametrized differentiable curve given by

$$\alpha(t) = (a\cos t,\; a\sin t,\; bt), \qquad t \in R,$$

has as its trace in $R^3$ a helix of pitch $2\pi b$ on the cylinder $x^2 + y^2 = a^2$. The parameter $t$ measures the angle which the $x$ axis makes with the line joining the origin to the projection of the point $\alpha(t)$ over the $xy$ plane.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Cusp Curve)</span></p>

The map $\alpha\colon R \to R^2$ given by $\alpha(t) = (t^3, t^2)$, $t \in R$, is a parametrized differentiable curve. Notice that $\alpha'(0) = (0, 0)$; that is, the velocity vector is zero for $t = 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Self-Intersecting Curve)</span></p>

The map $\alpha\colon R \to R^2$ given by $\alpha(t) = (t^3 - 4t,\; t^2 - 4)$, $t \in R$, is a parametrized differentiable curve. Notice that $\alpha(2) = \alpha(-2) = (0, 0)$; that is, the map $\alpha$ is not one-to-one.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Non-Differentiable Map)</span></p>

The map $\alpha\colon R \to R^2$ given by $\alpha(t) = (t, \|t\|)$, $t \in R$, is *not* a parametrized differentiable curve, since $\|t\|$ is not differentiable at $t = 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Same Trace, Different Parametrizations)</span></p>

The two distinct parametrized curves

$$\alpha(t) = (\cos t,\; \sin t), \qquad \beta(t) = (\cos 2t,\; \sin 2t),$$

where $t \in (0 - \epsilon,\; 2\pi + \epsilon)$, $\epsilon > 0$, have the same trace, namely the circle $x^2 + y^2 = 1$. Notice that the velocity vector of the second curve is the double of the first one.

</div>

### Inner Product in $R^3$

Let $u = (u\_1, u\_2, u\_3) \in R^3$ and define its **norm** (or **length**) by

$$|u| = \sqrt{u_1^2 + u_2^2 + u_3^2}.$$

Geometrically, $\|u\|$ is the distance from the point $(u\_1, u\_2, u\_3)$ to the origin $0 = (0, 0, 0)$. Let $u = (u\_1, u\_2, u\_3)$ and $v = (v\_1, v\_2, v\_3)$ belong to $R^3$, and let $\theta$, $0 \le \theta \le \pi$, be the angle formed by the segments $0u$ and $0v$. The **inner product** $u \cdot v$ is defined by

$$u \cdot v = |u|\,|v|\cos\theta.$$

The following properties hold:

1. Assume that $u$ and $v$ are nonzero vectors. Then $u \cdot v = 0$ if and only if $u$ is orthogonal to $v$.
2. $u \cdot v = v \cdot u$.
3. $\lambda(u \cdot v) = \lambda u \cdot v = u \cdot \lambda v$.
4. $u \cdot (v + w) = u \cdot v + u \cdot w$.

Using the standard basis $e\_1 = (1, 0, 0)$, $e\_2 = (0, 1, 0)$, $e\_3 = (0, 0, 1)$ with $e\_i \cdot e\_j = 1$ if $i = j$ and $e\_i \cdot e\_j = 0$ if $i \neq j$, one obtains the useful expression

$$u \cdot v = u_1 v_1 + u_2 v_2 + u_3 v_3.$$

If $u(t)$ and $v(t)$, $t \in I$, are differentiable curves, then $u(t) \cdot v(t)$ is a differentiable function, and

$$\frac{d}{dt}(u(t) \cdot v(t)) = u'(t) \cdot v(t) + u(t) \cdot v'(t).$$

## 1-3. Regular Curves; Arc Length

Let $\alpha\colon I \to R^3$ be a parametrized differentiable curve. For each $t \in I$ where $\alpha'(t) \neq 0$, there is a well-defined straight line, which contains the point $\alpha(t)$ and the vector $\alpha'(t)$. This line is called the **tangent line** to $\alpha$ at $t$. For the study of the differential geometry of a curve it is essential that there exists such a tangent line at every point. Therefore, we call any point $t$ where $\alpha'(t) = 0$ a **singular point** of $\alpha$ and restrict our attention to curves without singular points.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Regular Curve)</span></p>

A parametrized differentiable curve $\alpha\colon I \to R^3$ is said to be **regular** if $\alpha'(t) \neq 0$ for all $t \in I$.

</div>

From now on we shall consider only regular parametrized differentiable curves (and, for convenience, shall usually omit the word differentiable).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Arc Length)</span></p>

Given $t \in I$, the **arc length** of a regular parametrized curve $\alpha\colon I \to R^3$, from the point $t\_0$, is by definition

$$s(t) = \int_{t_0}^{t} |\alpha'(t)|\,dt,$$

where

$$|\alpha'(t)| = \sqrt{(x'(t))^2 + (y'(t))^2 + (z'(t))^2}.$$

Since $\alpha'(t) \neq 0$, the arc-length $s$ is a differentiable function of $t$ and $ds/dt = \|\alpha'(t)\|$.

</div>

It can happen that the parameter $t$ is already the arc length measured from some point. In this case, $ds/dt = 1 = \|\alpha'(t)\|$; that is, the velocity vector has constant length equal to 1. Conversely, if $\|\alpha'(t)\| \equiv 1$, then

$$s = \int_{t_0}^{t} dt = t - t_0;$$

i.e., $t$ is the arc length of $\alpha$ measured from some point.

To simplify our exposition, we shall restrict ourselves to curves parametrized by arc length; we shall see later (Sec. 1-5) that this restriction is not essential.

It is convenient to set still another convention. Given the curve $\alpha$ parametrized by arc length $s \in (a, b)$, we may consider the curve $\beta$ defined in $(-b, -a)$ by $\beta(-s) = \alpha(s)$, which has the same trace as the first one but is described in the opposite direction. We say that these two curves differ by a **change of orientation**.

## 1-4. The Vector Product in $R^3$

In this section, we shall present some properties of the vector product in $R^3$.

### Orientation

It is convenient to begin by reviewing the notion of orientation of a vector space. Two ordered bases $e = \lbrace e\_i \rbrace$ and $f = \lbrace f\_i \rbrace$, $i = 1, \dots, n$, of an $n$-dimensional vector space $V$ have the **same orientation** if the matrix of change of basis has positive determinant. We denote this relation by $e \sim f$. This is an equivalence relation.

The set of all ordered bases of $V$ is thus decomposed into two equivalence classes. Each of the equivalence classes determined by this relation is called an **orientation** of $V$. Therefore, $V$ has two orientations, and if we fix one of them arbitrarily, the other one is called the opposite orientation.

In the case $V = R^3$, there exists a natural ordered basis $e\_1 = (1, 0, 0)$, $e\_2 = (0, 1, 0)$, $e\_3 = (0, 0, 1)$, and we shall call the orientation corresponding to this basis the **positive orientation** of $R^3$, the other one being the **negative orientation**.

### Definition and Properties of the Vector Product

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vector Product)</span></p>

Let $u, v \in R^3$. The **vector product** $u \wedge v$ (also written $u \times v$ and called the **cross product**) is the unique vector $u \wedge v \in R^3$ characterized by

$$(u \wedge v) \cdot w = \det(u, v, w) \qquad \text{for all } w \in R^3.$$

If we express $u$, $v$, and $w$ in the natural basis $\lbrace e\_i \rbrace$, then

$$\det(u, v, w) = \begin{vmatrix} u_1 & u_2 & u_3 \\ v_1 & v_2 & v_3 \\ w_1 & w_2 & w_3 \end{vmatrix}.$$

It is immediate from the definition that

$$u \wedge v = \begin{vmatrix} u_2 & u_3 \\ v_2 & v_3 \end{vmatrix} e_1 - \begin{vmatrix} u_1 & u_3 \\ v_1 & v_3 \end{vmatrix} e_2 + \begin{vmatrix} u_1 & u_2 \\ v_1 & v_2 \end{vmatrix} e_3.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Vector Product)</span></p>

1. $u \wedge v = -v \wedge u$ (anticommutativity).
2. $u \wedge v$ depends linearly on $u$ and $v$; i.e., for any real numbers $a$, $b$:
   $(au + bw) \wedge v = a\,u \wedge v + b\,w \wedge v.$
3. $u \wedge v = 0$ if and only if $u$ and $v$ are linearly dependent.
4. $(u \wedge v) \cdot u = 0$, $(u \wedge v) \cdot v = 0$.

</div>

### Geometric Interpretation

It follows from property 4 that the vector product $u \wedge v \neq 0$ is normal to a plane generated by $u$ and $v$. Its geometric interpretation is as follows:

- **Direction:** $\lbrace u, v, u \wedge v \rbrace$ is a positive basis, since $(u \wedge v) \cdot (u \wedge v) = \|u \wedge v\|^2 > 0$.

- **Norm:** We have the relation

$$(u \wedge v) \cdot (x \wedge y) = \begin{vmatrix} u \cdot x & v \cdot x \\ u \cdot y & v \cdot y \end{vmatrix},$$

from which it follows that

$$|u \wedge v|^2 = |u|^2\,|v|^2(1 - \cos^2\theta) = A^2,$$

where $\theta$ is the angle of $u$ and $v$, and $A$ is the area of a parallelogram generated by $u$ and $v$.

In short, the vector product of $u$ and $v$ is a vector $u \wedge v$ perpendicular to a plane generated by $u$ and $v$, with a norm equal to the area of a parallelogram generated by $u$ and $v$, and a direction such that $\lbrace u, v, u \wedge v \rbrace$ is a positive basis.

### Further Identities

The vector product is not associative. In fact, we have the following identity:

$$(u \wedge v) \wedge w = (u \cdot w)v - (v \cdot w)u. \tag{2}$$

Finally, let $u(t) = (u\_1(t), u\_2(t), u\_3(t))$ and $v(t) = (v\_1(t), v\_2(t), v\_3(t))$ be differentiable maps from the interval $(a, b)$ to $R^3$. Then $u(t) \wedge v(t)$ is also differentiable and

$$\frac{d}{dt}(u(t) \wedge v(t)) = \frac{du}{dt} \wedge v(t) + u(t) \wedge \frac{dv}{dt}.$$

## 1-5. The Local Theory of Curves Parametrized by Arc Length

This section contains the main results of curves which will be used in later parts of the book.

Let $\alpha\colon I = (a, b) \to R^3$ be a curve parametrized by arc length $s$. Since the tangent vector $\alpha'(s)$ has unit length, the norm $\|\alpha''(s)\|$ of the second derivative measures the rate of change of the angle which neighboring tangents make with the tangent at $s$. $\|\alpha''(s)\|$ gives, therefore, a measure of how rapidly the curve pulls away from the tangent line at $s$, in a neighborhood of $s$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Curvature)</span></p>

Let $\alpha\colon I \to R^3$ be a curve parametrized by arc length $s \in I$. The number $\|\alpha''(s)\| = k(s)$ is called the **curvature** of $\alpha$ at $s$.

</div>

If $\alpha$ is a straight line, $\alpha(s) = us + v$, where $u$ and $v$ are constant vectors ($\|u\| = 1$), then $k \equiv 0$. Conversely, if $k = \|\alpha''(s)\| \equiv 0$, then by integration $\alpha(s) = us + v$, and the curve is a straight line.

Notice that by a change of orientation, the tangent vector changes its direction; that is, if $\beta(-s) = \alpha(s)$, then

$$\frac{d\beta}{d(-s)}(-s) = -\frac{d\alpha}{ds}(s).$$

Therefore, $\alpha''(s)$ and the curvature remain invariant under a change of orientation.

### The Frenet Trihedron

At points where $k(s) \neq 0$, a unit vector $n(s)$ in the direction of $\alpha''(s)$ is well defined by the equation $\alpha''(s) = k(s)n(s)$. Moreover, $\alpha''(s)$ is normal to $\alpha'(s)$, because by differentiating $\alpha'(s) \cdot \alpha'(s) = 1$ we obtain $\alpha''(s) \cdot \alpha'(s) = 0$. Thus, $n(s)$ is normal to $\alpha'(s)$ and is called the **normal vector** at $s$. The plane determined by the unit tangent and normal vectors, $\alpha'(s)$ and $n(s)$, is called the **osculating plane** at $s$.

At points where $k(s) = 0$, the normal vector (and therefore the osculating plane) is not defined. It is convenient to say that $s \in I$ is a **singular point of order 1** if $\alpha''(s) = 0$ (in this context, the points where $\alpha'(s) = 0$ are called singular points of order 0).

In what follows, we shall restrict ourselves to curves parametrized by arc length without singular points of order 1. We shall denote by $t(s) = \alpha'(s)$ the unit tangent vector of $\alpha$ at $s$. Thus, $t'(s) = k(s)n(s)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Binormal Vector)</span></p>

The unit vector $b(s) = t(s) \wedge n(s)$ is normal to the osculating plane and will be called the **binormal vector** at $s$.

</div>

Since $b(s)$ is a unit vector, the length $\|b'(s)\|$ measures the rate of change of the neighboring osculating planes with the osculating plane at $s$; that is, $b'(s)$ measures how rapidly the curve pulls away from the osculating plane at $s$, in a neighborhood of $s$.

To compute $b'(s)$ we observe that, on the one hand, $b'(s)$ is normal to $b(s)$ and that, on the other hand,

$$b'(s) = t'(s) \wedge n(s) + t(s) \wedge n'(s) = t(s) \wedge n'(s);$$

that is, $b'(s)$ is normal to $t(s)$. It follows that $b'(s)$ is parallel to $n(s)$, and we may write

$$b'(s) = \tau(s)\,n(s)$$

for some function $\tau(s)$. (*Warning*: Many authors write $-\tau(s)$ instead of our $\tau(s)$.)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Torsion)</span></p>

Let $\alpha\colon I \to R^3$ be a curve parametrized by arc length such that $\alpha''(s) \neq 0$, $s \in I$. The number $\tau(s)$ defined by $b'(s) = \tau(s)\,n(s)$ is called the **torsion** of $\alpha$ at $s$.

</div>

If $\alpha$ is a plane curve (that is, $\alpha(I)$ is contained in a plane), then the plane of the curve agrees with the osculating plane; hence, $\tau \equiv 0$. Conversely, if $\tau \equiv 0$ (and $k \neq 0$), we have that $b(s) = b\_0 = \text{constant}$, and therefore

$$(\alpha(s) \cdot b_0)' = \alpha'(s) \cdot b_0 = 0.$$

It follows that $\alpha(s) \cdot b\_0 = \text{constant}$; hence, $\alpha(s)$ is contained in a plane normal to $b\_0$. The condition that $k \neq 0$ everywhere is essential here.

In contrast to the curvature, the torsion may be either positive or negative. The sign of the torsion has a geometric interpretation, to be given later (Sec. 1-6).

Notice that by changing orientation the binormal vector changes sign, since $b = t \wedge n$. It follows that $b'(s)$, and therefore the torsion, remains invariant under a change of orientation.

### The Frenet Formulas

To each value of the parameter $s$, we have associated three orthogonal unit vectors $t(s)$, $n(s)$, $b(s)$. The trihedron thus formed is referred to as the **Frenet trihedron** at $s$. The derivatives $t'(s) = kn$, $b'(s) = \tau n$, when expressed in the basis $\lbrace t, n, b \rbrace$, yield geometrical entities (curvature $k$ and torsion $\tau$) which give us information about the behavior of $\alpha$ in a neighborhood of $s$.

The search for other local geometrical entities would lead us to compute $n'(s)$. However, since $n = b \wedge t$, we have

$$n'(s) = b'(s) \wedge t(s) + b(s) \wedge t'(s) = -\tau b - kt,$$

and we obtain again the curvature and the torsion. For later use, we shall call the equations

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Frenet Formulas)</span></p>

$$\begin{aligned}
t' &= k\,n, \\
n' &= -k\,t - \tau\,b, \\
b' &= \tau\,n
\end{aligned}$$

the **Frenet formulas** (we have omitted the $s$, for convenience).

</div>

In this context, the following terminology is usual. The $tb$ plane is called the **rectifying plane**, and the $nb$ plane the **normal plane**. The lines which contain $n(s)$ and $b(s)$ and pass through $\alpha(s)$ are called the **principal normal** and the **binormal**, respectively. The inverse $R = 1/k$ of the curvature is called the **radius of curvature** at $s$. Of course, a circle of radius $r$ has radius of curvature equal to $r$.

Physically, we can think of a curve in $R^3$ as being obtained from a straight line by bending (curvature) and twisting (torsion). After reflecting on this construction, we are led to conjecture the following statement, which, roughly speaking, shows that $k$ and $\tau$ describe completely the local behavior of the curve.

### The Fundamental Theorem of the Local Theory of Curves

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Fundamental Theorem of the Local Theory of Curves)</span></p>

Given differentiable functions $k(s) > 0$ and $\tau(s)$, $s \in I$, there exists a regular parametrized curve $\alpha\colon I \to R^3$ such that $s$ is the arc length, $k(s)$ is the curvature, and $\tau(s)$ is the torsion of $\alpha$. Moreover, any other curve $\tilde{\alpha}$, satisfying the same conditions, differs from $\alpha$ by a rigid motion; that is, there exists an orthogonal linear map $\rho$ of $R^3$, with positive determinant, and a vector $c$ such that $\tilde{\alpha} = \rho \circ \alpha + c$.

</div>

A complete proof involves the theorem of existence and uniqueness of solutions of ordinary differential equations and will be given in the appendix to Chap. 4.

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of the Uniqueness Part</summary>

We first remark that arc length, curvature, and torsion are invariant under rigid motions; that means, for instance, that if $M\colon R^3 \to R^3$ is a rigid motion and $\alpha = \alpha(t)$ is a parametrized curve, then

$$\int_a^b \left|\frac{d\alpha}{dt}\right| dt = \int_a^b \left|\frac{d(M \circ \alpha)}{dt}\right| dt.$$

Now, assume that two curves $\alpha = \alpha(s)$ and $\tilde{\alpha} = \tilde{\alpha}(s)$ satisfy the conditions $k(s) = \tilde{k}(s)$ and $\tau(s) = \tilde{\tau}(s)$, $s \in I$. Let $t\_0, n\_0, b\_0$ and $\tilde{t}\_0, \tilde{n}\_0, \tilde{b}\_0$ be the Frenet trihedrons at $s = s\_0$ of $\alpha$ and $\tilde{\alpha}$, respectively. Clearly, there is a rigid motion which takes $\tilde{\alpha}(s\_0)$ into $\alpha(s\_0)$ and $\tilde{t}\_0, \tilde{n}\_0, \tilde{b}\_0$ into $t\_0, n\_0, b\_0$. Thus, after performing this rigid motion on $\tilde{\alpha}$, we have that $\tilde{\alpha}(s\_0) = \alpha(s\_0)$ and that the Frenet trihedrons $t(s), n(s), b(s)$ and $\tilde{t}(s), \tilde{n}(s), \tilde{b}(s)$ of $\alpha$ and $\tilde{\alpha}$, respectively, satisfy the Frenet equations:

$$\frac{dt}{ds} = kn, \qquad \frac{d\tilde{t}}{ds} = k\tilde{n},$$

$$\frac{dn}{ds} = -kt - \tau b, \qquad \frac{d\tilde{n}}{ds} = -k\tilde{t} - \tau\tilde{n},$$

$$\frac{db}{ds} = \tau n, \qquad \frac{d\tilde{b}}{ds} = \tau\tilde{n},$$

with $t(s\_0) = \tilde{t}(s\_0)$, $n(s\_0) = \tilde{n}(s\_0)$, $b(s\_0) = \tilde{b}(s\_0)$. We now observe, by using the Frenet equations, that

$$\frac{1}{2}\frac{d}{ds}\lbrace |t - \tilde{t}|^2 + |n - \tilde{n}|^2 + |b - \tilde{b}|^2 \rbrace$$

$$= \langle t - \tilde{t},\; t' - \tilde{t}'\rangle + \langle b - \tilde{b},\; b' - \tilde{b}'\rangle + \langle n - \tilde{n},\; n' - \tilde{n}'\rangle$$

$$= k\langle t - \tilde{t},\; n - \tilde{n}\rangle + \tau\langle b - \tilde{b},\; n - \tilde{n}\rangle - k\langle n - \tilde{n},\; t - \tilde{t}\rangle$$

$$\quad - \tau\langle n - \tilde{n},\; b - \tilde{b}\rangle$$

$$= 0$$

for all $s \subset I$. Thus, the above expression is constant, and, since it is zero for $s = s\_0$, it is identically zero. It follows that $t(s) = \tilde{t}(s)$, $n(s) = \tilde{n}(s)$, $b(s) = \tilde{b}(s)$ for all $s \in I$. Since

$$\frac{d\alpha}{ds} = t = \tilde{t} = \frac{d\tilde{\alpha}}{ds},$$

we obtain $(d/ds)(\alpha - \tilde{\alpha}) = 0$. Thus, $\alpha(s) = \tilde{\alpha}(s) + a$, where $a$ is a constant vector. Since $\alpha(s\_0) = \tilde{\alpha}(s\_0)$, we have $a = 0$; hence, $\alpha(s) = \tilde{\alpha}(s)$ for all $s \in I$. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1</span><span class="math-callout__name">(Signed Curvature for Plane Curves)</span></p>

In the particular case of a plane curve $\alpha\colon I \to R^2$, it is possible to give the curvature $k$ a sign. For that, let $\lbrace e\_1, e\_2 \rbrace$ be the natural basis (see Sec. 1-4) of $R^2$ and define the normal vector $n(s)$, $s \in I$, by requiring the basis $\lbrace t(s), n(s) \rbrace$ to have the same orientation as the basis $\lbrace e\_1, e\_2 \rbrace$. The curvature $k$ is then *defined* by

$$\frac{dt}{ds} = kn$$

and might be either positive or negative. It is clear that $\|k\|$ agrees with the previous definition and that $k$ changes sign when we change either the orientation of $\alpha$ or the orientation of $R^2$.

</div>

It should also be remarked that, in the case of plane curves ($\tau = 0$), the proof of the fundamental theorem, referred to above, is actually very simple (see Exercise 9).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2</span><span class="math-callout__name">(Reparametrization by Arc Length)</span></p>

Given a regular parametrized curve $\alpha\colon I \to R^3$ (not necessarily parametrized by arc length), it is possible to obtain a curve $\beta\colon J \to R^3$ parametrized by arc length which has the same trace as $\alpha$. In fact, let

$$s = s(t) = \int_{t_0}^{t} |\alpha'(t)|\,dt, \qquad t, t_0 \in I.$$

Since $ds/dt = \|\alpha'(t)\| \neq 0$, the function $s = s(t)$ has a differentiable inverse $t = t(s)$, $s \in s(I) = J$. Now set $\beta = \alpha \circ t\colon J \to R^3$. Clearly, $\beta(J) = \alpha(I)$ and $\|\beta'(s)\| = \|\alpha'(t) \cdot (dt/ds)\| = 1$. This shows that $\beta$ has the same trace as $\alpha$ and is parametrized by arc length. It is usual to say that $\beta$ is a **reparametrization of $\alpha(I)$ by arc length**.

This fact allows us to extend all local concepts previously defined to regular curves with an arbitrary parameter. Thus, we say that the curvature $k(t)$ of $\alpha\colon I \to R^3$ at $t \in I$ is the curvature of a reparametrization $\beta\colon J \to R^3$ of $\alpha(I)$ by arc length at the corresponding point $s = s(t)$. This is clearly independent of the choice of $\beta$ and shows that the restriction, made at the end of Sec. 1-3, of considering only curves parametrized by arc length is not essential.

</div>

## 1-6. The Local Canonical Form

One of the most effective methods of solving problems in geometry consists of finding a coordinate system which is adapted to the problem. In the study of local properties of a curve, in the neighborhood of the point $s$, we have a natural coordinate system, namely the Frenet trihedron at $s$. It is therefore convenient to refer the curve to this trihedron.

Let $\alpha\colon I \to R^3$ be a curve parametrized by arc length without singular points of order 1. We shall write the equations of the curve, in a neighborhood of $s\_0$, using the trihedron $t(s\_0)$, $n(s\_0)$, $b(s\_0)$ as a basis for $R^3$. We may assume, without loss of generality, that $s\_0 = 0$, and we shall consider the (finite) Taylor expansion

$$\alpha(s) = \alpha(0) + s\alpha'(0) + \frac{s^2}{2}\alpha''(0) + \frac{s^3}{6}\alpha'''(0) + R,$$

where $\lim R/s^3 = 0$ as $s \to 0$. Since $\alpha'(0) = t$, $\alpha''(0) = kn$, and

$$\alpha'''(0) = (kn)' = k'n + kn' = k'n - k^2t - k\tau b,$$

we obtain

$$\alpha(s) - \alpha(0) = \left(s - \frac{k^2 s^3}{3!}\right)t + \left(\frac{s^2 k}{2} + \frac{s^3 k'}{3!}\right)n - \frac{s^3}{3!}k\tau\,b + R,$$

where all terms are computed at $s = 0$.

Let us now take the system $Oxyz$ in such a way that the origin $O$ agrees with $\alpha(0)$ and that $t = (1, 0, 0)$, $n = (0, 1, 0)$, $b = (0, 0, 1)$. Under these conditions, $\alpha(s) = (x(s), y(s), z(s))$ is given by

$$x(s) = s - \frac{k^2 s^3}{6} + R_x,$$

$$y(s) = \frac{k}{2}s^2 + \frac{k'}{6}s^3 + R_y,$$

$$z(s) = -\frac{k\tau}{6}s^3 + R_z,$$

where $R = (R\_x, R\_y, R\_z)$. The representation above is called the **local canonical form** of $\alpha$, in a neighborhood of $s = 0$.

### Geometric Interpretation of the Sign of Torsion

From the third equation of the local canonical form it follows that if $\tau < 0$ and $s$ is sufficiently small, then $z(s)$ increases with $s$. Let us make the convention of calling the "positive side" of the osculating plane that side toward which $b$ is pointing. Then, since $z(0) = 0$, when we describe the curve in the direction of increasing arc length, the curve will cross the osculating plane at $s = 0$, pointing toward the positive side. If, on the contrary, $\tau > 0$, the curve (described in the direction of increasing arc length) will cross the osculating plane pointing to the side opposite the positive side.

The helix of Exercise 1 of Sec. 1-5 has negative torsion. An example of a curve with positive torsion is the helix

$$\alpha(s) = \left(a\cos\frac{s}{c},\; a\sin\frac{s}{c},\; -b\frac{s}{c}\right)$$

obtained from the first one by a reflection in the $xz$ plane.

### Other Consequences of the Local Canonical Form

Another consequence of the canonical form is the existence of a neighborhood $J \subset I$ of $s = 0$ such that $\alpha(J)$ is entirely contained in the one side of the rectifying plane toward which the vector $n$ is pointing. In fact, since $k > 0$, we obtain, for $s$ sufficiently small, $y(s) \ge 0$, and $y(s) = 0$ if and only if $s = 0$. This proves our claim.

As a last application of the canonical form, we mention the following property of the osculating plane. The osculating plane at $s$ is the limit position of the plane determined by the tangent line at $s$ and the point $\alpha(s + h)$ when $h \to 0$. To prove this, let us assume that $s = 0$. Thus, every plane containing the tangent at $s = 0$ is of the form $z = cy$ or $y = 0$. The plane $y = 0$ is the rectifying plane that, as seen above, contains no points near $\alpha(0)$ (except $\alpha(0)$ itself) and may therefore be discarded from our considerations. The condition for the plane $z = cy$ to pass through $s + h$ is ($s = 0$)

$$c = \frac{z(h)}{y(h)} = \frac{-\frac{k}{6}\tau h^3 + \cdots}{\frac{k}{2}h^2 + \frac{k'}{6}h^3 + \cdots}.$$

Letting $h \to 0$, we see that $c \to 0$. Therefore, the limit position of the plane $z(s) = c(h)y(s)$ is the plane $z = 0$, that is, the osculating plane, as we wished.

## 1-7. Global Properties of Plane Curves

In this section we want to describe some results that belong to the global differential geometry of curves. Even in the simple case of plane curves, the subject already offers examples of nontrivial theorems and interesting questions. To develop this material here, we must assume some plausible facts without proofs; we shall try to be careful by stating these facts precisely.

This section contains three topics in order of increasing difficulty: (A) the isoperimetric inequality, (B) the four-vertex theorem, and (C) the Cauchy-Crofton formula.

### Closed Plane Curves

A *differentiable function on a closed interval* $[a, b]$ is the restriction of a differentiable function defined on an open interval containing $[a, b]$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Closed Plane Curve)</span></p>

A **closed plane curve** is a regular parametrized curve $\alpha\colon [a, b] \to R^2$ such that $\alpha$ and all its derivatives agree at $a$ and $b$; that is,

$$\alpha(a) = \alpha(b), \qquad \alpha'(a) = \alpha'(b), \qquad \alpha''(a) = \alpha''(b), \quad \dots$$

The curve $\alpha$ is **simple** if it has no further self-intersections; that is, if $t\_1, t\_2 \in [a, b)$, $t\_1 \neq t\_2$, then $\alpha(t\_1) \neq \alpha(t\_2)$.

</div>

We usually consider the curve $\alpha\colon [0, l] \to R^2$ parametrized by arc length $s$; hence, $l$ is the length of $\alpha$. Sometimes we refer to a simple closed curve $C$, meaning the trace of such an object. The curvature of $\alpha$ will be taken with a sign, as in Remark 1 of Sec. 1-5.

We assume that *a simple closed curve $C$ in the plane bounds a region of this plane* that is called the **interior** of $C$. This is part of the so-called Jordan curve theorem (a proof will be given in Sec. 5-6, Theorem 1), which does not hold, for instance, for simple curves on a torus (the surface of a doughnut). Whenever we speak of the area bounded by a simple closed curve $C$, we mean the area of the interior of $C$.

We assume further that the parameter of a simple closed curve can be so chosen that if one is going along the curve in the direction of increasing parameters, then the interior of the curve remains to the left. Such a curve will be called **positively oriented**.

### A. The Isoperimetric Inequality

This is perhaps the oldest global theorem in differential geometry and is related to the following (isoperimetric) problem. *Of all simple closed curves in the plane with a given length $l$, which one bounds the largest area?* In this form, the problem was known to the Greeks, who also knew the solution, namely, the circle.

A satisfactory proof of the fact that the circle is a solution to the isoperimetric problem took, however, a long time to appear. The earliest proofs assumed that a solution should exist. It was only in 1870 that K. Weierstrass pointed out that many similar questions did not have solutions and gave a complete proof of the existence of a solution. The simple proof we shall present is due to E. Schmidt (1939).

We shall make use of the following formula for the area $A$ bounded by a positively oriented simple closed curve $\alpha(t) = (x(t), y(t))$, where $t \in [a, b]$ is an arbitrary parameter:

$$A = -\int_a^b y(t)\,x'(t)\,dt = \int_a^b x(t)\,y'(t)\,dt = \frac{1}{2}\int_a^b (xy' - yx')\,dt.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Isoperimetric Inequality)</span></p>

Let $C$ be a simple closed plane curve with length $l$, and let $A$ be the area of the region bounded by $C$. Then

$$l^2 - 4\pi A \ge 0,$$

and equality holds if and only if $C$ is a circle.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of the Isoperimetric Inequality</summary>

Let $E$ and $E'$ be two parallel lines which do not meet the closed curve $C$, and move them together until they first meet $C$. We thus obtain two parallel tangent lines to $C$, $L$ and $L'$, so that the curve is entirely contained in the strip bounded by $L$ and $L'$. Consider a circle $S^1$ which is tangent to both $L$ and $L'$ and does not meet $C$. Let $O$ be the center of $S^1$ and take a coordinate system with origin at $O$ and the $x$ axis perpendicular to $L$ and $L'$.

Parametrize $C$ by arc length, $\alpha(s) = (x(s), y(s))$, so that it is positively oriented and the tangency points of $L$ and $L'$ are $s = 0$ and $s = s\_1$, respectively. We can assume that the equation of $S^1$ is

$$\tilde{\alpha}(s) = (\tilde{x}(s), \tilde{y}(s)) = (x(s), \bar{y}(s)), \quad s \in [0, l]$$

where $2r$ is the distance between $L$ and $L'$. By using the area formula and denoting by $\tilde{A}$ the area bounded by $S^1$, we have

$$A = \int_0^l x y'\,ds, \qquad \tilde{A} = \pi r^2 = -\int_0^l \bar{y} x'\,ds.$$

Thus,

$$A + \pi r^2 = \int_0^l (xy' - \bar{y}x')\,ds \le \int_0^l \sqrt{(xy' - \bar{y}x')^2}\,ds$$

$$\le \int_0^l \sqrt{(x^2 + \bar{y}^2)((x')^2 + (y')^2)}\,ds = \int_0^l \sqrt{x^2 + \bar{y}^2}\,ds$$

$$= lr.$$

We now notice the fact that the geometric mean of two positive numbers is smaller than or equal to their arithmetic mean, and equality holds only if they are equal. It follows that

$$\sqrt{A \cdot \pi r^2} \le \frac{1}{2}(A + \pi r^2) \le \frac{1}{2}lr.$$

Therefore, $4\pi A r^2 \le l^2 r^2$, and this gives the inequality $l^2 - 4\pi A \ge 0$.

Now, assume that equality holds in the isoperimetric inequality. Then equality must hold everywhere in the above chain of inequalities. From equality in the AM-GM step, $A = \pi r^2$. Thus, $l = 2\pi r$ and $r$ does not depend on the choice of the direction of $L$. Furthermore, equality in the Cauchy-Schwarz step implies that

$$(xy' - \bar{y}x')^2 = (x^2 + \bar{y}^2)((x')^2 + (y')^2)$$

or

$$(xx' + \bar{y}y')^2 = 0;$$

that is,

$$\frac{x}{y'} = \frac{\bar{y}}{x'} = \frac{\sqrt{x^2 + \bar{y}^2}}{\sqrt{(y')^2 + (x')^2}} = \pm r.$$

Thus, $x = \pm r y'$. Since $r$ does not depend on the choice of the direction of $L$, we can interchange $x$ and $y$ in the last relation and obtain $y = \pm r x'$. Thus,

$$x^2 + y^2 = r^2((x')^2 + (y')^2) = r^2$$

and $C$ is a circle, as we wished. **Q.E.D.**

</details>
</div>

### B. The Four-Vertex Theorem

We shall need further general facts on plane closed curves.

Let $\alpha\colon [0, l] \to R^2$ be a plane closed curve given by $\alpha(s) = (x(s), y(s))$. Since $s$ is the arc length, the tangent vector $t(s) = (x'(s), y'(s))$ has unit length. It is convenient to introduce the **tangent indicatrix** $t\colon [0, l] \to R^2$ that is given by $t(s) = (x'(s), y'(s))$; this is a differentiable curve, the trace of which is contained in a circle of radius 1.

The velocity vector of the tangent indicatrix is

$$\frac{dt}{ds} = (x''(s), y''(s)) = \alpha''(s) = kn,$$

where $n$ is the normal vector, oriented as in Remark 1 of Sec. 1-5, and $k$ is the curvature of $\alpha$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Rotation Index)</span></p>

Let $\theta(s)$, $0 < \theta(s) < 2\pi$, be the angle that $t(s)$ makes with the $x$ axis; that is, $x'(s) = \cos\theta(s)$, $y'(s) = \sin\theta(s)$. Then $\theta = \theta(s)$ is locally well defined as a differentiable function and

$$\theta(s) = \int_0^s k(s)\,ds.$$

Since $\alpha$ is closed, this angle is an integer multiple $I$ of $2\pi$; that is,

$$\int_0^l k(s)\,ds = \theta(l) - \theta(0) = 2\pi I.$$

The integer $I$ is called the **rotation index** of the curve $\alpha$.

</div>

The rotation index changes sign when we change the orientation of the curve. Furthermore, the definition is so set that the rotation index of a positively oriented simple closed curve is positive.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem of Turning Tangents)</span></p>

The rotation index of a simple closed curve is $\pm 1$, where the sign depends on the orientation of the curve.

</div>

A regular, plane (not necessarily closed) curve $\alpha\colon [a, b] \to R^2$ is **convex** if, for all $t \in [a, b]$, the trace $\alpha([a, b])$ of $\alpha$ lies entirely on one side of the closed half-plane determined by the tangent line at $t$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vertex)</span></p>

A **vertex** of a regular plane curve $\alpha\colon [a, b] \to R^2$ is a point $t \in [a, b]$ where $k'(t) = 0$. For instance, an ellipse with unequal axes has exactly four vertices, namely the points where the axes meet the ellipse.

</div>

It is an interesting global fact that four is the least number of vertices for all closed convex curves.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Four-Vertex Theorem)</span></p>

A simple closed convex curve has at least four vertices.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of the Four-Vertex Theorem</summary>

Before starting the proof, we need a lemma.

**Lemma.** Let $\alpha\colon [0, l] \to R^2$ be a plane closed curve parametrized by arc length and let $A$, $B$, $C$ be arbitrary real numbers. Then

$$\int_0^l (Ax + By + C)\frac{dk}{ds}\,ds = 0,$$

where the functions $x = x(s)$, $y = y(s)$ are given by $\alpha(s) = (x(s), y(s))$, and $k$ is the curvature of $\alpha$.

*Proof of the Lemma.* Recall that there exists a differentiable function $\theta\colon [0, l] \to R$ such that $x'(s) = \cos\theta$, $y'(s) = \sin\theta$. Thus, $k(s) = \theta'(s)$ and

$$x'' = -ky', \qquad y'' = kx'.$$

Therefore, since the functions involved agree at $0$ and $l$,

$$\int_0^l k'\,ds = 0,$$

$$\int_0^l xk'\,ds = -\int_0^l kx'\,ds = -\int_0^l y''\,ds = 0,$$

$$\int_0^l yk'\,ds = -\int_0^l ky'\,ds = \int_0^l x''\,ds = 0. \qquad \textbf{Q.E.D.}$$

*Proof of the Theorem.* Parametrize the curve by arc length, $\alpha\colon [0, l] \to R^2$. Since $k = k(s)$ is a continuous function on the closed interval $[0, l]$, it reaches a maximum and a minimum on $[0, l]$. Thus, $\alpha$ has at least two vertices, $\alpha(s\_1) = p$ and $\alpha(s\_2) = q$. Let $L$ be the straight line passing through $p$ and $q$, and let $\beta$ and $\gamma$ be the two arcs of $C$ which are determined by the points $p$ and $q$.

We claim that each of these arcs lies on a definite side of $L$. Otherwise, it meets $L$ in a point $r$ distinct from $p$ and $q$. By convexity, and since $p$, $q$, $r$ are distinct points on $C$, the tangent line at the intermediate point, say $p$, has to agree with $L$. Again, by convexity, this implies that $L$ is tangent to $C$ at the three points $p$, $q$, and $r$. But then the tangent to a point near $p$ (the intermediate point) will have $q$ and $r$ on distinct sides, unless the whole segment $rq$ of $L$ belongs to $C$. This implies that $k = 0$ at $p$ and $q$. Since these are points of maximum and minimum for $k$, $k \equiv 0$ on $C$, a contradiction.

Let $Ax + By + C = 0$ be the equation of $L$. If there are no further vertices, $k'(s)$ keeps a constant sign on each of the arcs $\beta$ and $\gamma$. We can then arrange the sign of all the coefficients $A$, $B$, $C$ so that the integral in the Lemma is positive. This contradiction shows that there is a third vertex and that $k'(s)$ changes sign on $\beta$ or $\gamma$, say, on $\beta$. Since $p$ and $q$ are points of maximum and minimum, $k'(s)$ changes sign twice on $\beta$. Thus, there is a fourth vertex. **Q.E.D.**

</details>
</div>

The four-vertex theorem has been the subject of many investigations. The theorem also holds for simple, closed (not necessarily convex) curves, but the proof is harder.

Later (Sec. 5-6, Prop. 1) we shall prove that *a plane closed curve is convex if and only if it is simple and can be oriented so that its curvature is positive or zero*. From that, and the proof given above, we see that we can reformulate the statement of the four-vertex theorem as follows. *The curvature function of a closed convex curve is (nonnegative and) either constant or else has at least two maxima and two minima.*

### C. The Cauchy-Crofton Formula

Our last topic in this section will be dedicated to finding a theorem which, roughly speaking, describes the following situation. Let $C$ be a regular curve in the plane. We look at all straight lines in the plane that meet $C$ and assign to each such line a **multiplicity** which is the number of its intersection points with $C$.

We first want to find a way of assigning a measure to a given subset of straight lines in the plane. A straight line $L$ in the plane is determined by the distance $p \ge 0$ from $L$ to the origin $O$ of the coordinates and by the angle $\theta$, $0 \le \theta < 2\pi$, which a half-line starting at $0$ and normal to $L$ makes with the $x$ axis. The equation of $L$ in terms of these parameters is easily seen to be

$$x\cos\theta + y\sin\theta = p.$$

Thus, we can replace the set of all straight lines in the plane by the set

$$\mathfrak{L} = \lbrace (p, \theta) \in R^2;\; p \ge 0,\; 0 \le \theta < 2\pi \rbrace.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Invariance of the Measure on Lines)</span></p>

Let $f(x, y)$ be a continuous function defined in $R^2$. For any set $S \subset R^2$, define the area of $S$ by

$$A(S) = \iint_S f(x, y)\,dx\,dy.$$

Assume that $A$ is invariant under rigid motions; that is, if $S$ is any set and $\bar{S} = F^{-1}(S)$, where $F$ is the rigid motion $(x, y) \to (\bar{x}, \bar{y})$ given by

$$x = a + \bar{x}\cos\varphi - \bar{y}\sin\varphi, \qquad y = b + \bar{x}\sin\varphi + \bar{y}\cos\varphi,$$

we have $A(\bar{S}) = A(S)$. Then $f(x, y) = \text{const}$.

</div>

Since the Jacobian of the rigid motion is 1, and rigid motions are transitive on points of the plane, this proposition shows that $dx\,dy$ is, up to a constant factor, the only element of area that is invariant under rigid motions. With these preparations, we can finally define the measure of a set $\mathfrak{S} \subset \mathfrak{L}$ as

$$\iint_{\mathfrak{S}} dp\,d\theta.$$

In the same way as in the proposition above, one can then prove that this is, up to a constant factor, the only measure on $\mathfrak{L}$ that is invariant under rigid motions. This measure is, therefore, as reasonable as it can be.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Cauchy-Crofton Formula)</span></p>

Let $C$ be a regular plane curve with length $l$. The measure of the set of straight lines (counted with multiplicities) which meet $C$ is equal to $2l$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Sketch of Proof of the Cauchy-Crofton Formula</summary>

First assume that the curve $C$ is a segment of a straight line with length $l$. Since our measure is invariant under rigid motions, we can assume that the coordinate system has its origin $0$ in the middle point of $C$ and that the $x$ axis is in the direction of $C$. Then the measure of the set of straight lines that meet $C$ is

$$\iint dp\,d\theta = \int_0^{2\pi}\left(\int_0^{|\cos\theta|\,(l/2)} dp\right)d\theta = \int_0^{2\pi} \frac{l}{2}|\cos\theta|\,d\theta = 2l.$$

Next, let $C$ be a polygonal line composed of a finite number of segments $C\_i$ with length $l\_i$ ($\sum l\_i = l$). Let $n = n(p, \theta)$ be the number of intersection points of the straight line $(p, \theta)$ with $C$. Then, by summing up the results for each segment $C\_i$, we obtain

$$\iint n\,dp\,d\theta = 2\sum_i l_i = 2l,$$

which is the Cauchy-Crofton formula for a polygonal line.

Finally, by a limiting process, it is possible to extend the above formula to any regular curve, and this will prove the theorem. **Q.E.D.**

</details>
</div>

The Cauchy-Crofton formula can be used in many ways. For instance, if a curve is not rectifiable (see Exercise 9, Sec. 1-3) but the left-hand side of the formula has a meaning, this can be used to define the "length" of such a curve. The formula can also be used to obtain an efficient way of estimating lengths of curves. Indeed, a good approximation for the integral is given as follows. Consider a family of parallel straight lines such that two consecutive lines are at a distance $r$. Rotate this family by angles of $\pi/4$, $2\pi/4$, $3\pi/4$ in order to obtain four families of straight lines. Let $n$ be the number of intersection points of a curve $C$ with all these lines. Then

$$\frac{1}{2}\,n\,r\,\frac{\pi}{4}$$

is an approximation to the integral

$$\frac{1}{2}\iint n\,dp\,d\theta = \text{length of } C$$

and therefore gives an estimate for the length of $C$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Estimating DNA Length via Cauchy-Crofton)</span></p>

Consider an electron micrograph of a circular DNA molecule. Four families of straight lines at a distance of 7 millimeters and angles of $\pi/4$ are drawn over the picture. The number of intersection points is found to be 153. Thus,

$$\frac{1}{2}\,n\,r\,\frac{\pi}{4} = \frac{1}{2} \cdot 153 \cdot \frac{3.14}{4} \approx 60.$$

Since the reference line in the picture represents 1 micrometer ($= 10^{-6}$ meter) and measures, in our scale, 25 millimeters, $r = \frac{25}{7}$, and thus the length of this DNA molecule, from our values, is approximately

$$60\left(\frac{25}{7}\right) \approx 16.6 \text{ micrometers}.$$

The actual value is 16.3 micrometers.

</div>

The general ideas of this topic belong to a branch of geometry known under the name of **integral geometry**. A survey of the subject can be found in L. A. Santaló, "Integral Geometry," in *Studies in Global Geometry and Analysis*, edited by S. S. Chern, The Mathematical Association of America, 1967, 147–193.

# Chapter 2 — Regular Surfaces

## 2-1. Introduction

In this chapter, we shall begin the study of surfaces. Whereas in the first chapter we used mainly elementary calculus of one variable, we shall now need some knowledge of calculus of several variables. Specifically, we need to know some facts about continuity and differentiability of functions and maps in $R^2$ and $R^3$.

In Sec. 2-2 we shall introduce the basic concept of a regular surface in $R^3$. In contrast to the treatment of curves in Chap. 1, regular surfaces are defined as sets rather than maps. The goal of Sec. 2-2 is to describe some criteria that are helpful in trying to decide whether a given subset of $R^3$ is a regular surface.

In Sec. 2-3 we shall show that it is possible to define what it means for a function on a regular surface to be differentiable, and in Sec. 2-4 we shall show that the usual notion of differential in $R^2$ can be extended to such functions. Thus, regular surfaces in $R^3$ provide a natural setting for two-dimensional calculus.

Sections 2-6 through 2-8 are optional on a first reading. In Sec. 2-6, we shall treat the idea of orientation on regular surfaces, which will be needed in Chaps. 3 and 4.

## 2-2. Regular Surfaces; Inverse Images of Regular Values

In this section we shall introduce the notion of a regular surface in $R^3$. Roughly speaking, a regular surface in $R^3$ is obtained by taking pieces of a plane, deforming them, and arranging them in such a way that the resulting figure has no sharp points, edges, or self-intersections and so that it makes sense to speak of a tangent plane at points of the figure. The idea is to define a set that is, in a certain sense, two-dimensional and that also is smooth enough so that the usual notions of calculus can be extended to it.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Regular Surface)</span></p>

A subset $S \subset R^3$ is a **regular surface** if, for each $p \in S$, there exists a neighborhood $V$ in $R^3$ and a map $\mathbf{x}\colon U \to V \cap S$ of an open set $U \subset R^2$ onto $V \cap S \subset R^3$ such that

1. $\mathbf{x}$ is **differentiable**. This means that if we write

   $$\mathbf{x}(u, v) = (x(u, v),\; y(u, v),\; z(u, v)), \qquad (u, v) \in U,$$

   the functions $x(u, v)$, $y(u, v)$, $z(u, v)$ have continuous partial derivatives of all orders in $U$.

2. $\mathbf{x}$ is a **homeomorphism**. Since $\mathbf{x}$ is continuous by condition 1, this means that $\mathbf{x}$ has an inverse $\mathbf{x}^{-1}\colon V \cap S \to U$ which is continuous; that is, $\mathbf{x}^{-1}$ is the restriction of a continuous map $F\colon W \subset R^3 \to R^2$ defined on an open set $W$ containing $V \cap S$.

3. **(The regularity condition.)** For each $q \in U$, the differential $d\mathbf{x}\_q\colon R^2 \to R^3$ is one-to-one.

</div>

The mapping $\mathbf{x}$ is called a **parametrization** or a **system of (local) coordinates** in (a neighborhood of) $p$. The neighborhood $V \cap S$ of $p$ in $S$ is called a **coordinate neighborhood**.

### Explaining the Regularity Condition

To give condition 3 a more familiar form, let us compute the matrix of the linear map $d\mathbf{x}\_q$ in the canonical bases $e\_1 = (1, 0)$, $e\_2 = (0, 1)$ of $R^2$ and $f\_1 = (1, 0, 0)$, $f\_2 = (0, 1, 0)$, $f\_3 = (0, 0, 1)$ of $R^3$.

Let $q = (u\_0, v\_0)$. The vector $e\_1$ is tangent to the curve $u \to (u, v\_0)$ whose image under $\mathbf{x}$ is the curve (called the *coordinate curve* $v = v\_0$) lying on $S$ with tangent vector

$$\left(\frac{\partial x}{\partial u},\; \frac{\partial y}{\partial u},\; \frac{\partial z}{\partial u}\right) = \frac{\partial \mathbf{x}}{\partial u}.$$

Similarly, $d\mathbf{x}\_q(e\_2) = \partial \mathbf{x}/\partial v$. Thus, the matrix of $d\mathbf{x}\_q$ in the referred basis is

$$d\mathbf{x}_q = \begin{pmatrix} \frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} \\[4pt] \frac{\partial y}{\partial u} & \frac{\partial y}{\partial v} \\[4pt] \frac{\partial z}{\partial u} & \frac{\partial z}{\partial v} \end{pmatrix}.$$

Condition 3 of the definition may now be expressed by requiring the two column vectors of this matrix to be linearly independent; or, equivalently, that the vector product $\partial \mathbf{x}/\partial u \wedge \partial \mathbf{x}/\partial v \neq 0$; or, in still another way, that one of the Jacobian determinants

$$\frac{\partial(x, y)}{\partial(u, v)}, \qquad \frac{\partial(y, z)}{\partial(u, v)}, \qquad \frac{\partial(z, x)}{\partial(u, v)}$$

be different from zero at $q$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1</span><span class="math-callout__name">(On the Definition of Regular Surface)</span></p>

In contrast to our treatment of curves in Chap. 1, we have defined a surface as a *subset $S$ of $R^3$*, and not as a map. This is achieved by covering $S$ with the traces of parametrizations which satisfy conditions 1, 2, and 3.

Condition 1 is very natural if we expect to do some differential geometry on $S$. The one-to-oneness in condition 2 has the purpose of preventing self-intersections in regular surfaces. The continuity of the inverse in condition 2 has a more subtle purpose which can be fully understood only in the next section. For the time being, we shall mention that this condition is essential to proving that certain objects defined in terms of a parametrization do not depend on this parametrization but only on the set $S$ itself. Finally, condition 3 will guarantee the existence of a "tangent plane" at all points of $S$.

</div>

### Examples of Regular Surfaces

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Sphere $S^2$)</span></p>

Let us show that the unit sphere

$$S^2 = \lbrace (x, y, z) \in R^3;\; x^2 + y^2 + z^2 = 1 \rbrace$$

is a regular surface. We first verify that the map $\mathbf{x}\_1\colon U \subset R^2 \to R^3$ given by

$$\mathbf{x}_1(x, y) = (x,\; y,\; +\sqrt{1 - (x^2 + y^2)}), \qquad (x, y) \in U,$$

where $R^2 = \lbrace (x, y, z) \in R^3;\; z = 0 \rbrace$ and $U = \lbrace (x, y) \in R^2;\; x^2 + y^2 < 1 \rbrace$, is a parametrization of $S^2$. The image $\mathbf{x}\_1(U)$ is the (open) part of $S^2$ above the $xy$ plane. Since $x^2 + y^2 < 1$, the function $+\sqrt{1 - (x^2 + y^2)}$ has continuous partial derivatives of all orders. Thus, $\mathbf{x}\_1$ is differentiable and condition 1 holds. Condition 3 is easily verified, since $\partial(x, y)/\partial(x, y) \equiv 1$.

We similarly define $\mathbf{x}\_2(x, y) = (x, y, -\sqrt{1 - (x^2 + y^2)})$ and use the $xz$ and $yz$ planes to obtain the parametrizations $\mathbf{x}\_3$ through $\mathbf{x}\_6$. Together, $\mathbf{x}\_1$ through $\mathbf{x}\_6$ cover $S^2$ completely and show that $S^2$ is a regular surface.

For most applications, it is convenient to relate parametrizations to **geographical coordinates** on $S^2$. Let $V = \lbrace (\theta, \varphi);\; 0 < \theta < \pi,\; 0 < \varphi < 2\pi \rbrace$ and let $\mathbf{x}\colon V \to R^3$ be given by

$$\mathbf{x}(\theta, \varphi) = (\sin\theta\cos\varphi,\; \sin\theta\sin\varphi,\; \cos\theta).$$

Here $\theta$ is called the **colatitude** (the complement of the latitude) and $\varphi$ the **longitude**. One can verify that $\mathbf{x}$ is a parametrization of $S^2$ that covers $S^2$ minus a semicircle (including the two poles). Two such parametrizations cover the whole sphere.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Ellipsoid)</span></p>

The ellipsoid

$$\frac{x^2}{a^2} + \frac{y^2}{b^2} + \frac{z^2}{c^2} = 1$$

is a regular surface. In fact, it is the set $f^{-1}(0)$ where

$$f(x, y, z) = \frac{x^2}{a^2} + \frac{y^2}{b^2} + \frac{z^2}{c^2} - 1$$

is a differentiable function and $0$ is a regular value of $f$. This follows from the fact that the partial derivatives $f\_x = 2x/a^2$, $f\_y = 2y/b^2$, $f\_z = 2z/c^2$ vanish simultaneously only at the point $(0, 0, 0)$, which does not belong to $f^{-1}(0)$. This example includes the sphere as a particular case ($a = b = c = 1$).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Hyperboloid of Two Sheets)</span></p>

The hyperboloid of two sheets $-x^2 - y^2 + z^2 = 1$ is a regular surface, since it is given by $S = f^{-1}(0)$, where $f(x, y, z) = -x^2 - y^2 + z^2 - 1$ and $0$ is a regular value of $f$. Note that the surface $S$ is not connected; that is, given two points in two distinct sheets ($z > 0$ and $z < 0$), it is not possible to join them by a continuous curve $\alpha(t) = (x(t), y(t), z(t))$ contained in the surface.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Torus)</span></p>

The torus $T$ is the "surface" generated by rotating a circle $S^1$ of radius $r$ about a straight line belonging to the plane of the circle and at a distance $a > r$ away from the center of the circle. Let $S^1$ be the circle in the $yz$ plane with its center at the point $(0, a, 0)$. Then $S^1$ is given by $(y - a)^2 + z^2 = r^2$, and the points of the figure $T$ obtained by rotating this circle about the $z$ axis satisfy the equation

$$z^2 = r^2 - (\sqrt{x^2 + y^2} - a)^2.$$

Therefore, $T$ is the inverse image of $r^2$ by the function

$$f(x, y, z) = z^2 + (\sqrt{x^2 + y^2} - a)^2.$$

This function is differentiable for $(x, y) \neq (0, 0)$, and since

$$\frac{\partial f}{\partial z} = 2z, \qquad \frac{\partial f}{\partial y} = \frac{2y(\sqrt{x^2 + y^2} - a)}{\sqrt{x^2 + y^2}}, \qquad \frac{\partial f}{\partial x} = \frac{2x(\sqrt{x^2 + y^2} - a)}{\sqrt{x^2 + y^2}},$$

$r^2$ is a regular value of $f$. It follows that the torus $T$ is a regular surface.

A parametrization for the torus $T$ can be given by

$$\mathbf{x}(u, v) = ((r\cos u + a)\cos v,\; (r\cos u + a)\sin v,\; r\sin u),$$

where $0 < u < 2\pi$, $0 < v < 2\pi$. The torus can be covered by three such coordinate neighborhoods.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The One-Sheeted Cone is Not a Regular Surface)</span></p>

The one-sheeted cone $C$, given by $z = +\sqrt{x^2 + y^2}$, $(x, y) \in R^2$, is *not* a regular surface. By Prop. 3 below, if $C$ were a regular surface, it would be, in a neighborhood of $(0, 0, 0)$, the graph of a differentiable function having one of three forms: $y = h(x, z)$, $x = g(y, z)$, $z = f(x, y)$. The first two forms can be discarded by the simple fact that the projections of $C$ over the $xz$ and $yz$ planes are not one-to-one. The last form would have to agree, in a neighborhood of $(0, 0, 0)$, with $z = +\sqrt{x^2 + y^2}$. Since $z = +\sqrt{x^2 + y^2}$ is not differentiable at $(0, 0)$, this is impossible.

</div>

### Propositions for Recognizing Regular Surfaces

Deciding whether a given subset of $R^3$ is a regular surface directly from the definition may be quite tiresome. The following propositions simplify this task.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Graphs are Regular Surfaces)</span></p>

If $f\colon U \to R$ is a differentiable function in an open set $U$ of $R^2$, then the graph of $f$, that is, the subset of $R^3$ given by $(x, y, f(x, y))$ for $(x, y) \in U$, is a regular surface.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

It suffices to show that the map $\mathbf{x}\colon U \to R^3$ given by

$$\mathbf{x}(u, v) = (u,\; v,\; f(u, v))$$

is a parametrization of the graph whose coordinate neighborhood covers every point. Condition 1 is clearly satisfied, and condition 3 also offers no difficulty since $\partial(x, y)/\partial(u, v) \equiv 1$. Finally, each point $(x, y, z)$ of the graph is the image under $\mathbf{x}$ of the unique point $(u, v) = (x, y) \in U$. $\mathbf{x}$ is therefore one-to-one, and since $\mathbf{x}^{-1}$ is the restriction to $f$ of the (continuous) projection of $R^3$ onto the $xy$ plane, $\mathbf{x}^{-1}$ is continuous. **Q.E.D.**

</details>
</div>

Before stating the next proposition, we need a definition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Critical Point, Critical Value, Regular Value)</span></p>

Given a differentiable map $F\colon U \subset R^n \to R^m$ defined in an open set $U$ of $R^n$, we say that $p \in U$ is a **critical point** of $F$ if the differential $dF\_p\colon R^n \to R^m$ is not a surjective (or onto) mapping. The image $F(p) \in R^m$ of a critical point is called a **critical value** of $F$. A point of $R^m$ which is not a critical value is called a **regular value** of $F$.

Note, in particular, that if $f\colon U \subset R^3 \to R$ is a differentiable function, then $a \in f(U)$ is a regular value of $f$ if and only if $f\_x$, $f\_y$, and $f\_z$ do not vanish simultaneously at any point in the inverse image $f^{-1}(a) = \lbrace (x, y, z) \in U\colon f(x, y, z) = a \rbrace$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Inverse Images of Regular Values)</span></p>

If $f\colon U \subset R^3 \to R$ is a differentiable function and $a \in f(U)$ is a regular value of $f$, then $f^{-1}(a)$ is a regular surface in $R^3$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Sketch</summary>

Let $p = (x\_0, y\_0, z\_0)$ be a point of $f^{-1}(a)$. Since $a$ is a regular value, it is possible to assume, by renaming the axis if necessary, that $f\_z \neq 0$ at $p$. We define a mapping $F\colon U \subset R^3 \to R^3$ by

$$F(x, y, z) = (x,\; y,\; f(x, y, z)),$$

and the differential of $F$ at $p$ is given by

$$dF_p = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ f_x & f_y & f_z \end{pmatrix},$$

whence $\det(dF\_p) = f\_z \neq 0$. We can therefore apply the inverse function theorem, which guarantees the existence of neighborhoods $V$ of $p$ and $W$ of $F(p)$ such that $F\colon V \to W$ is invertible and the inverse $F^{-1}\colon W \to V$ is differentiable. It follows that the coordinate functions of $F^{-1}$, i.e., the functions

$$x = u, \qquad y = v, \qquad z = g(u, v, t), \qquad (u, v, t) \in W,$$

are differentiable. In particular, $z = g(u, v, a) = h(x, y)$ is a differentiable function defined in the projection of $V$ onto the $xy$ plane. Since

$$F(f^{-1}(a) \cap V) = W \cap \lbrace (u, v, t);\; t = a \rbrace,$$

we conclude that the graph of $h$ is $f^{-1}(a) \cap V$. By Prop. 1, $f^{-1}(a) \cap V$ is a coordinate neighborhood, and so $f^{-1}(a)$ can be covered by such neighborhoods and is a regular surface. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Regular Surfaces are Locally Graphs)</span></p>

Let $S \subset R^3$ be a regular surface and $p \in S$. Then there exists a neighborhood $V$ of $p$ in $S$ such that $V$ is the graph of a differentiable function which has one of the following three forms: $z = f(x, y)$, $y = g(x, z)$, $x = h(y, z)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4</span><span class="math-callout__name">(Sufficient Conditions for Parametrization)</span></p>

Let $p \in S$ be a point of a regular surface $S$ and let $\mathbf{x}\colon U \subset R^2 \to R^3$ be a map with $p \in \mathbf{x}(U)$ such that conditions 1 and 3 of the definition hold. Assume that $\mathbf{x}$ is one-to-one. Then $\mathbf{x}^{-1}$ is continuous.

</div>

This proposition means that if we already know that $S$ is a regular surface and we have a candidate $\mathbf{x}$ for a parametrization, we do not have to check that $\mathbf{x}^{-1}$ is continuous, provided that the other conditions hold and $\mathbf{x}$ is one-to-one.

### Connected Surfaces and Sign-Preserving Functions

A surface $S \subset R^3$ is said to be **connected** if any two of its points can be joined by a continuous curve in $S$. In the definition of a regular surface we made no restrictions on the connectedness of the surfaces, and the examples of regular surfaces given by Prop. 2 may not be connected.

An important property of connected surfaces is: *If $f\colon S \subset R^3 \to R$ is a nonzero continuous function defined on a connected surface $S$, then $f$ does not change sign on $S$.*

### Parametrized Surfaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Parametrized Surface)</span></p>

A **parametrized surface** $\mathbf{x}\colon U \subset R^2 \to R^3$ is a differentiable map $\mathbf{x}$ from an open set $U \subset R^2$ into $R^3$. The set $\mathbf{x}(U) \subset R^3$ is called the **trace** of $\mathbf{x}$. $\mathbf{x}$ is **regular** if the differential $d\mathbf{x}\_q\colon R^2 \to R^3$ is one-to-one for all $q \in U$ (i.e., the vectors $\partial \mathbf{x}/\partial u$, $\partial \mathbf{x}/\partial v$ are linearly independent for all $q \in U$). A point $p \in U$ where $d\mathbf{x}\_q$ is not one-to-one is called a **singular point** of $\mathbf{x}$.

</div>

Observe that a parametrized surface, even when regular, may have self-intersections in its trace.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Tangent Surface)</span></p>

Let $\alpha\colon I \to R^3$ be a regular parametrized curve. Define

$$\mathbf{x}(t, v) = \alpha(t) + v\alpha'(t), \qquad (t, v) \in I \times R.$$

$\mathbf{x}$ is a parametrized surface called the **tangent surface** of $\alpha$. Assume now that the curvature $k(t)$, $t \in I$, of $\alpha$ is nonzero for all $t \in I$, and restrict the domain of $\mathbf{x}$ to $U = \lbrace (t, v) \in I \times R;\; v \neq 0 \rbrace$. Then

$$\frac{\partial \mathbf{x}}{\partial t} = \alpha'(t) + v\alpha''(t), \qquad \frac{\partial \mathbf{x}}{\partial v} = \alpha'(t)$$

and

$$\frac{\partial \mathbf{x}}{\partial t} \wedge \frac{\partial \mathbf{x}}{\partial v} = v\,\alpha''(t) \wedge \alpha'(t) \neq 0, \qquad (t, v) \in U,$$

since the curvature $k(t) = \|\alpha''(t) \wedge \alpha'(t)\|/\|\alpha'(t)\|^3$ is nonzero. It follows that the restriction $\mathbf{x}\colon U \to R^3$ is a regular parametrized surface, the trace of which consists of two connected pieces whose common boundary is the set $\alpha(I)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Surfaces of Revolution)</span></p>

Let $S \subset R^3$ be the set obtained by rotating a regular plane curve $C$ about an axis in the plane which does not meet the curve; we shall take the $xz$ plane as the plane of the curve and the $z$ axis as the rotation axis. Let

$$x = f(v), \qquad z = g(v), \qquad a < v < b, \qquad f(v) > 0,$$

be a parametrization for $C$ and denote by $u$ the rotation angle about the $z$ axis. Thus, we obtain a map

$$\mathbf{x}(u, v) = (f(v)\cos u,\; f(v)\sin u,\; g(v))$$

from the open set $U = \lbrace (u, v) \in R^2;\; 0 < u < 2\pi,\; a < v < b \rbrace$ into $S$. Since $S$ can be entirely covered by similar parametrizations, it follows that $S$ is a regular surface which is called a **surface of revolution**. The curve $C$ is called the **generating curve** of $S$, the $z$ axis is the **rotation axis**, the circles described by the points of $C$ are called the **parallels** of $S$, and the various positions of $C$ on $S$ are called the **meridians** of $S$.

</div>

## 2-3. Change of Parameters; Differentiable Functions on Surfaces

Differential geometry is concerned with those properties of surfaces which depend on their behavior in a neighborhood of a point. The definition of a regular surface given in Sec. 2-2 is adequate for this purpose. According to this definition, each point $p$ of a regular surface belongs to a coordinate neighborhood. The points of such a neighborhood are characterized by their coordinates, and we should be able, therefore, to define the local properties which interest us in terms of these coordinates.

For example, it is important that we be able to define what it means for a function $f\colon S \to R$ to be differentiable at a point $p$ of a regular surface $S$. The same point of $S$ can, however, belong to various coordinate neighborhoods. Moreover, other coordinate systems could be chosen in a neighborhood of $p$. For the above definition to make sense, it is necessary that it does not depend on the chosen system of coordinates.

### Change of Parameters

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Change of Parameters)</span></p>

Let $p$ be a point of a regular surface $S$, and let $\mathbf{x}\colon U \subset R^2 \to S$, $\mathbf{y}\colon V \subset R^2 \to S$ be two parametrizations of $S$ such that $p \in \mathbf{x}(U) \cap \mathbf{y}(V) = W$. Then the **change of coordinates** $h = \mathbf{x}^{-1} \circ \mathbf{y}\colon \mathbf{y}^{-1}(W) \to \mathbf{x}^{-1}(W)$ is a **diffeomorphism**; that is, $h$ is differentiable and has a differentiable inverse $h^{-1}$.

</div>

In other words, if $\mathbf{x}$ and $\mathbf{y}$ are given by

$$\mathbf{x}(u, v) = (x(u, v), y(u, v), z(u, v)), \qquad (u, v) \in U,$$

$$\mathbf{y}(\xi, \eta) = (x(\xi, \eta), y(\xi, \eta), z(\xi, \eta)), \qquad (\xi, \eta) \in V,$$

then the change of coordinates $h$, given by

$$u = u(\xi, \eta), \qquad v = v(\xi, \eta), \qquad (\xi, \eta) \in \mathbf{y}^{-1}(W),$$

where the functions $\xi$ and $\eta$ also have partial derivatives of all orders. Since

$$\frac{\partial(u, v)}{\partial(\xi, \eta)} \cdot \frac{\partial(\xi, \eta)}{\partial(u, v)} = 1,$$

this implies that the Jacobian determinants of both $h$ and $h^{-1}$ are nonzero everywhere.

### Differentiable Functions on Surfaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Differentiable Function on a Surface)</span></p>

Let $f\colon V \subset S \to R$ be a function defined in an open subset $V$ of a regular surface $S$. Then $f$ is said to be **differentiable** at $p \in V$ if, for some parametrization $\mathbf{x}\colon U \subset R^2 \to S$ with $p \in \mathbf{x}(U) \subset V$, the composition $f \circ \mathbf{x}\colon U \subset R^2 \to R$ is differentiable at $\mathbf{x}^{-1}(p)$. $f$ is **differentiable in $V$** if it is differentiable at all points of $V$.

</div>

It follows immediately from the last proposition that the definition given does not depend on the choice of the parametrization $\mathbf{x}$. In fact, if $\mathbf{y}\colon V \subset R^2 \to S$ is another parametrization with $p \in \mathbf{x}(V)$, and if $h = \mathbf{x}^{-1} \circ \mathbf{y}$, then $f \circ \mathbf{y} = f \circ \mathbf{x} \circ h$ is also differentiable, whence the asserted independence.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Differentiable Functions on Surfaces)</span></p>

Let $S$ be a regular surface and $V \subset R^3$ be an open set such that $S \subset V$. Let $f\colon V \subset R^3 \to R$ be a differentiable function. Then the restriction of $f$ to $S$ is a differentiable function on $S$. In fact, for any $p \in S$ and any parametrization $\mathbf{x}\colon U \subset R^2 \to S$ in $p$, the function $f \circ \mathbf{x}\colon U \to R$ is differentiable. In particular, the following are differentiable functions:

1. The **height function** relative to a unit vector $v \in R^3$, $h\colon S \to R$, given by $h(p) = p \cdot v$, $p \in S$, where the dot denotes the usual inner product in $R^3$. $h(p)$ is the height of $p \in S$ relative to a plane normal to $v$ and passing through the origin of $R^3$.

2. The square of the distance from a fixed point $p\_0 \in R^3$, $f(p) = \|p - p\_0\|^2$, $p \in S$. The need for taking the square comes from the fact that the distance $\|p - p\_0\|$ is not differentiable at $p = p\_0$.

</div>

### Diffeomorphisms between Surfaces

The definition of differentiability can be easily extended to mappings between surfaces. A continuous map $\varphi\colon V\_1 \subset S\_1 \to S\_2$ of an open set $V\_1$ of a regular surface $S\_1$ to a regular surface $S\_2$ is said to be **differentiable at** $p \in V\_1$ if, given parametrizations

$$\mathbf{x}_1\colon U_1 \subset R^2 \to S_1, \qquad \mathbf{x}_2\colon U_2 \subset R^2 \to S_2,$$

with $p \in \mathbf{x}\_1(U\_1)$ and $\varphi(\mathbf{x}\_1(U\_1)) \subset \mathbf{x}\_2(U\_2)$, the map

$$\mathbf{x}_2^{-1} \circ \varphi \circ \mathbf{x}_1\colon U_1 \to U_2$$

is differentiable at $q = \mathbf{x}\_1^{-1}(p)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Diffeomorphism)</span></p>

Two regular surfaces $S\_1$ and $S\_2$ are **diffeomorphic** if there exists a differentiable map $\varphi\colon S\_1 \to S\_2$ with a differentiable inverse $\varphi^{-1}\colon S\_2 \to S\_1$. Such a $\varphi$ is called a **diffeomorphism** from $S\_1$ to $S\_2$.

</div>

The notion of diffeomorphism plays the same role in the study of regular surfaces that the notion of isomorphism plays in the study of vector spaces or the notion of congruence plays in Euclidean geometry. In other words, from the point of view of differentiability, two diffeomorphic surfaces are indistinguishable.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Parametrizations are Local Diffeomorphisms)</span></p>

If $\mathbf{x}\colon U \subset R^2 \to S$ is a parametrization, $\mathbf{x}^{-1}\colon \mathbf{x}(U) \to R^2$ is differentiable. In fact, for any $p \in \mathbf{x}(U)$ and any parametrization $\mathbf{y}\colon V \subset R^2 \to S$ in $p$, we have that $\mathbf{x}^{-1} \circ \mathbf{y}\colon \mathbf{y}^{-1}(W) \to \mathbf{x}^{-1}(W)$, where $W = \mathbf{x}(U) \cap \mathbf{y}(V)$, is differentiable. This shows that $U$ and $\mathbf{x}(U)$ are diffeomorphic (i.e., every regular surface is locally diffeomorphic to a plane) and justifies the identification made in Remark 1.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Differentiable Maps from Ambient Space)</span></p>

Let $S\_1$ and $S\_2$ be regular surfaces. Assume that $S\_1 \subset V \subset R^3$, where $V$ is an open set of $R^3$, and that $\varphi\colon V \to R^3$ is a differentiable map such that $\varphi(S\_1) \subset S\_2$. Then the restriction $\varphi\|\_{S\_1}\colon S\_1 \to S\_2$ is a differentiable map. In particular:

1. If $S$ is symmetric relative to the $xy$ plane, then the map $\sigma\colon S \to S$, $\sigma(x, y, z) = (x, y, -z)$, which takes each $p \in S$ into its symmetrical point, is differentiable. This generalizes to surfaces symmetric relative to any plane of $R^3$.

2. Let $R\_{z,\theta}\colon R^3 \to R^3$ be the rotation of angle $\theta$ about the $z$ axis, and let $S \subset R^3$ be a regular surface invariant by this rotation. Then the restriction $R\_{z,\theta}\|\_S\colon S \to S$ is a differentiable map.

3. Let $\varphi\colon R^3 \to R^3$ be given by $\varphi(x, y, z) = (xa, yb, zc)$, where $a$, $b$, and $c$ are nonzero real numbers. $\varphi$ is clearly differentiable, and the restriction $\varphi\|\_{S^2}$ is a differentiable map from the sphere $S^2$ into the ellipsoid $x^2/a^2 + y^2/b^2 + z^2/c^2 = 1$.

</div>

### Regular Curves on Surfaces

At this stage we could return to the theory of curves and treat them from the point of view of this chapter, i.e., as subsets of $R^3$. A **regular curve** in $R^3$ is a subset $C \subset R^3$ with the following property: For each point $p \in C$ there is a neighborhood $V$ of $p$ in $R^3$ and a differentiable homeomorphism $\alpha\colon I \subset R \to V \cap C$ such that the differential $d\alpha\_t$ is one-to-one for each $t \in I$.

It is possible to prove that the change of parameters is given by a diffeomorphism. From this fundamental result, it is possible to decide when a given property obtained by means of a parametrization is independent of that parametrization. Such a property will then be a local property of the set $C$.

For example, the arc length, defined in Chap. 1, is independent of the parametrization chosen and is, therefore, a property of the set $C$. Since it is always possible to locally parametrize a regular curve $C$ by arc length, the properties (curvature, torsion, etc.) determined by means of this parametrization are local properties of $S$. This shows that the local theory of curves developed in Chap. 1 is valid for regular curves.

## 2-4. The Tangent Plane; The Differential of a Map

In this section we shall show that condition 3 in the definition of a regular surface $S$ guarantees that for every $p \in S$ the set of tangent vectors to the parametrized curves of $S$, passing through $p$, constitutes a plane.

By a **tangent vector** to $S$, at a point $p \in S$, we mean the tangent vector $\alpha'(0)$ of a differentiable parametrized curve $\alpha\colon (-\epsilon, \epsilon) \to S$ with $\alpha(0) = p$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Tangent Plane)</span></p>

Let $\mathbf{x}\colon U \subset R^2 \to S$ be a parametrization of a regular surface $S$ and let $q \in U$. The vector subspace of dimension 2,

$$d\mathbf{x}_q(R^2) \subset R^3,$$

coincides with the set of tangent vectors to $S$ at $\mathbf{x}(q)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $w$ be a tangent vector at $\mathbf{x}(q)$, that is, let $w = \alpha'(0)$, where $\alpha\colon (-\epsilon, \epsilon) \to \mathbf{x}(U) \subset S$ is differentiable and $\alpha(0) = \mathbf{x}(q)$. By Example 2 of Sec. 2-3, the curve $\beta = \mathbf{x}^{-1} \circ \alpha\colon (-\epsilon, \epsilon) \to U$ is differentiable. By definition of the differential, we have $d\mathbf{x}\_q(\beta'(0)) = w$. Hence, $w \in d\mathbf{x}\_q(R^2)$.

On the other hand, let $w = d\mathbf{x}\_q(v)$, where $v \in R^2$. It is clear that $v$ is the velocity vector of the curve $\gamma\colon (-\epsilon, \epsilon) \to U$ given by $\gamma(t) = tv + q$, $t \in (-\epsilon, \epsilon)$. By definition of the differential, $w = \alpha'(0)$, where $\alpha = \mathbf{x} \circ \gamma$. This shows that $w$ is a tangent vector. **Q.E.D.**

</details>
</div>

By the above proposition, the plane $d\mathbf{x}\_q(R^2)$, which passes through $\mathbf{x}(q) = p$, does not depend on the parametrization $\mathbf{x}$. This plane will be called the **tangent plane** to $S$ at $p$ and will be denoted by $T\_p(S)$. The choice of the parametrization $\mathbf{x}$ determines a basis $\lbrace (\partial \mathbf{x}/\partial u)(q),\; (\partial \mathbf{x}/\partial v)(q) \rbrace$ of $T\_p(S)$, called the **basis associated to $\mathbf{x}$**. Sometimes it is convenient to write $\partial \mathbf{x}/\partial u = \mathbf{x}\_u$ and $\partial \mathbf{x}/\partial v = \mathbf{x}\_v$.

The coordinates of a vector $w \in T\_p(S)$ in the basis associated to a parametrization $\mathbf{x}$ are determined as follows. $w$ is the velocity vector $\alpha'(0)$ of a curve $\alpha = \mathbf{x} \circ \beta$, where $\beta\colon (-\epsilon, \epsilon) \to U$ is given by $\beta(t) = (u(t), v(t))$, with $\beta(0) = q = \mathbf{x}^{-1}(p)$. Thus,

$$\alpha'(0) = \frac{d}{dt}(\mathbf{x} \circ \beta)(0) = \frac{d}{dt}\mathbf{x}(u(t), v(t))(0) = \mathbf{x}_u(q)\,u'(0) + \mathbf{x}_v(q)\,v'(0) = w.$$

Thus, in the basis $\lbrace \mathbf{x}\_u(q), \mathbf{x}\_v(q) \rbrace$, $w$ has coordinates $(u'(0), v'(0))$, where $(u(t), v(t))$ is the expression, in the parametrization $\mathbf{x}$, of a curve whose velocity vector at $t = 0$ is $w$.

### The Differential of a Map between Surfaces

With the notion of a tangent plane, we can talk about the differential of a (differentiable) map between surfaces. Let $S\_1$ and $S\_2$ be two regular surfaces and let $\varphi\colon V \subset S\_1 \to S\_2$ be a differentiable mapping of an open set $V$ of $S\_1$ into $S\_2$. If $p \in V$, we know that every tangent vector $w \in T\_p(S\_1)$ is the velocity vector $\alpha'(0)$ of a differentiable parametrized curve $\alpha\colon (-\epsilon, \epsilon) \to V$ with $\alpha(0) = p$. The curve $\beta = \varphi \circ \alpha$ is such that $\beta(0) = \varphi(p)$, and therefore $\beta'(0)$ is a vector of $T\_{\varphi(p)}(S\_2)$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Differential of a Map)</span></p>

In the discussion above, given $w$, the vector $\beta'(0)$ does not depend on the choice of $\alpha$. The map $d\varphi\_p\colon T\_p(S\_1) \to T\_{\varphi(p)}(S\_2)$ defined by $d\varphi\_p(w) = \beta'(0)$ is linear.

</div>

The linear map $d\varphi\_p$ defined by Prop. 2 is called the **differential** of $\varphi$ at $p \in S\_1$. In a similar way we define the differential of a (differentiable) function $f\colon U \subset S \to R$ at $p \in U$ as a linear map $df\_p\colon T\_p(S) \to R$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Differential of the Height Function)</span></p>

Let $v \in R^3$ be a unit vector and let $h\colon S \to R$, $h(p) = v \cdot p$, $p \in S$, be the height function defined in Example 1 of Sec. 2-3. To compute $dh\_p(w)$, $w \in T\_p(S)$, choose a differentiable curve $\alpha\colon (-\epsilon, \epsilon) \to S$ with $\alpha(0) = p$, $\alpha'(0) = w$. Since $h(\alpha(t)) = \alpha(t) \cdot v$, we obtain

$$dh_p(w) = \frac{d}{dt}h(\alpha(t))\bigg|_{t=0} = \alpha'(0) \cdot v = w \cdot v.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Differential of Rotation on the Sphere)</span></p>

Let $S^2 \subset R^2$ be the unit sphere and let $R\_{z,\theta}\colon R^3 \to R^3$ be the rotation of angle $\theta$ about the $z$ axis. Then $R\_{z,\theta}$ restricted to $S^2$ is a differentiable map of $S^2$. We shall compute $(dR\_{z,\theta})\_p(w)$, $p \in S^2$, $w \in T\_p(S^2)$. Choose $\alpha\colon (-\epsilon, \epsilon) \to S^2$ with $\alpha(0) = p$, $\alpha'(0) = w$. Then, since $R\_{z,\theta}$ is linear,

$$(dR_{z,\theta})_p(w) = \frac{d}{dt}(R_{z,\theta} \circ \alpha(t))\bigg|_{t=0} = R_{z,\theta}(\alpha'(0)) = R_{z,\theta}(w).$$

Observe that $R\_{z,\theta}$ leaves the north pole $N = (0, 0, 1)$ fixed, and that $(dR\_{z,\theta})\_N\colon T\_N(S) \to T\_N(S)$ is just a rotation of angle $\theta$ in the plane $T\_N(S)$.

</div>

### The Normal Vector and the Normal Line

Given a point $p$ on a regular surface $S$, there are two unit vectors of $R^3$ that are normal to the tangent plane $T\_p(S)$; each of them is called a **unit normal vector** at $p$. The straight line that passes through $p$ and contains a unit normal vector at $p$ is called the **normal line** at $p$. The **angle** of two intersecting surfaces at an intersection point $p$ is the angle of their tangent planes (or their normal lines) at $p$.

By fixing a parametrization $\mathbf{x}\colon U \subset R^2 \to S$ at $p \in S$, we can make a definite choice of a unit normal vector at each point $q \in \mathbf{x}(U)$ by the rule

$$N(q) = \frac{\mathbf{x}_u \wedge \mathbf{x}_v}{|\mathbf{x}_u \wedge \mathbf{x}_v|}(q).$$

Thus, we obtain a differentiable map $N\colon \mathbf{x}(U) \to R^3$. We shall see later (Secs. 2-6 and 3-1) that it is not always possible to extend this map differentiably to the whole surface $S$.

### The Inverse Function Theorem for Surfaces

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Inverse Function Theorem for Surfaces)</span></p>

If $S\_1$ and $S\_2$ are regular surfaces and $\varphi\colon U \subset S\_1 \to S\_2$ is a differentiable mapping of an open set $U \subset S\_1$ such that the differential $d\varphi\_p$ of $\varphi$ at $p \in U$ is an isomorphism, then $\varphi$ is a local diffeomorphism at $p$.

</div>

Of course, all other concepts of calculus, like critical points, regular values, etc., do extend naturally to functions and maps defined on regular surfaces.

## 2-5. The First Fundamental Form; Area

So far we have looked at surfaces from the point of view of differentiability. In this section we shall begin the study of further geometric structures carried by the surface. The most important of these is perhaps the first fundamental form, which we shall now describe.

The natural inner product of $R^3 \supset S$ induces on each tangent plane $T\_p(S)$ of a regular surface $S$ an inner product, to be denoted by $\langle\;,\;\rangle\_p$: If $w\_1, w\_2 \in T\_p(S) \subset R^3$, then $\langle w\_1, w\_2 \rangle\_p$ is equal to the inner product of $w\_1$ and $w\_2$ as vectors in $R^3$. To this inner product, which is a symmetric bilinear form (i.e., $\langle w\_1, w\_2 \rangle = \langle w\_2, w\_1 \rangle$ and $\langle w\_1, w\_2 \rangle$ is linear in both $w\_1$ and $w\_2$), there corresponds a quadratic form $I\_p\colon T\_p(S) \to R$ given by

$$I_p(w) = \langle w, w \rangle_p = |w|^2 > 0.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(First Fundamental Form)</span></p>

The quadratic form $I\_p$ on $T\_p(S)$, defined by $I\_p(w) = \langle w, w \rangle\_p = \|w\|^2 > 0$, is called the **first fundamental form** of the regular surface $S \subset R^3$ at $p \in S$.

</div>

Therefore, the first fundamental form is merely the expression of how the surface $S$ inherits the natural inner product of $R^3$. Geometrically, the first fundamental form allows us to make measurements on the surface (lengths of curves, angles of tangent vectors, areas of regions) without referring back to the ambient space $R^3$ where the surface lies.

### Coefficients of the First Fundamental Form

We shall now express the first fundamental form in the basis $\lbrace \mathbf{x}\_u, \mathbf{x}\_v \rbrace$ associated to a parametrization $\mathbf{x}(u, v)$ at $p$. Since a tangent vector $w \in T\_p(S)$ *is the tangent vector to a parametrized curve* $\alpha(t) = \mathbf{x}(u(t), v(t))$, $t \in (-\epsilon, \epsilon)$, with $p = \alpha(0) = \mathbf{x}(u\_0, v\_0)$, we obtain

$$I_p(\alpha'(0)) = \langle \alpha'(0), \alpha'(0) \rangle_p = \langle \mathbf{x}_u u' + \mathbf{x}_v v',\; \mathbf{x}_u u' + \mathbf{x}_v v' \rangle_p$$

$$= E(u')^2 + 2F u'v' + G(v')^2,$$

where the values of the functions involved are computed for $t = 0$, and

$$E(u_0, v_0) = \langle \mathbf{x}_u, \mathbf{x}_u \rangle_p, \qquad F(u_0, v_0) = \langle \mathbf{x}_u, \mathbf{x}_v \rangle_p, \qquad G(u_0, v_0) = \langle \mathbf{x}_v, \mathbf{x}_v \rangle_p$$

are the **coefficients of the first fundamental form** in the basis $\lbrace \mathbf{x}\_u, \mathbf{x}\_v \rbrace$ of $T\_p(S)$. By letting $p$ run in the coordinate neighborhood corresponding to $\mathbf{x}(u, v)$ we obtain differentiable functions $E(u, v)$, $F(u, v)$, $G(u, v)$ in that neighborhood.

### Metric Applications

As we mentioned before, the importance of the first fundamental form $I$ comes from the fact that by knowing $I$ we can treat metric questions on a regular surface without further references to the ambient space $R^3$. Thus, the arc length $s$ of a parametrized curve $\alpha\colon I \to S$ is given by

$$s(t) = \int_0^t |\alpha'(t)|\,dt = \int_0^t \sqrt{I(\alpha'(t))}\,dt.$$

In particular, if $\alpha(t) = \mathbf{x}(u(t), v(t))$ is contained in a coordinate neighborhood corresponding to the parametrization $\mathbf{x}(u, v)$, we can compute the arc length of $\alpha$ between, say, $0$ and $t$ by

$$s(t) = \int_0^t \sqrt{E(u')^2 + 2Fu'v' + G(v')^2}\,dt.$$

Also, the angle $\theta$ under which two parametrized regular curves $\alpha\colon I \to S$, $\beta\colon I \to S$ intersect at $t = t\_0$ is given by

$$\cos\theta = \frac{\langle \alpha'(t_0), \beta'(t_0) \rangle}{|\alpha'(t_0)|\,|\beta'(t_0)|}.$$

In particular, the angle $\varphi$ of the coordinate curves of a parametrization $\mathbf{x}(u, v)$ is

$$\cos\varphi = \frac{\langle \mathbf{x}_u, \mathbf{x}_v \rangle}{|\mathbf{x}_u|\,|\mathbf{x}_v|} = \frac{F}{\sqrt{EG}};$$

it follows that *the coordinate curves of a parametrization are orthogonal if and only if* $F(u, v) = 0$ *for all* $(u, v)$. Such a parametrization is called an **orthogonal parametrization**.

Because of the expression for arc length, many mathematicians talk about the "element" of arc length, $ds$, of $S$, and write

$$ds^2 = E\,du^2 + 2F\,du\,dv + G\,dv^2,$$

meaning that if $\alpha(t) = \mathbf{x}(u(t), v(t))$ is a curve on $S$ and $s = s(t)$ is its arc length, then

$$\left(\frac{ds}{dt}\right)^2 = E\left(\frac{du}{dt}\right)^2 + 2F\frac{du}{dt}\frac{dv}{dt} + G\left(\frac{dv}{dt}\right)^2.$$

### Examples

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(First Fundamental Form of the Plane)</span></p>

A coordinate system for a plane $P = R^3$ passing through $p\_0 = (x\_0, y\_0, z\_0)$ and containing the orthonormal vectors $w\_1 = (a\_1, a\_2, a\_3)$, $w\_2 = (b\_1, b\_2, b\_3)$ is given as follows:

$$\mathbf{x}(u, v) = p_0 + uw_1 + vw_2, \qquad (u, v) \in R^2.$$

Since $w\_1$ and $w\_2$ are unit orthogonal vectors, $\mathbf{x}\_u = w\_1$, $\mathbf{x}\_v = w\_2$; the functions $E$, $F$, $G$ are constant and given by

$$E = 1, \qquad F = 0, \qquad G = 1.$$

In this trivial case, the first fundamental form is essentially the Pythagorean theorem in $P$; i.e., the square of the length of a vector $w$ which has coordinates $a$, $b$ in the basis $\lbrace \mathbf{x}\_u, \mathbf{x}\_v \rbrace$ is equal to $a^2 + b^2$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(First Fundamental Form of the Cylinder)</span></p>

The right cylinder over the circle $x^2 + y^2 = 1$ admits the parametrization $\mathbf{x}\colon U \to R^3$, where

$$\mathbf{x}(u, v) = (\cos u, \sin u, v), \qquad U = \lbrace (u, v) \in R^2;\; 0 < u < 2\pi,\; -\infty < v < \infty \rbrace.$$

We compute $\mathbf{x}\_u = (-\sin u, \cos u, 0)$, $\mathbf{x}\_v = (0, 0, 1)$, and therefore

$$E = \sin^2 u + \cos^2 u = 1, \qquad F = 0, \qquad G = 1.$$

We remark that, although the cylinder and the plane are distinct surfaces, we obtain the same result in both cases. We shall return to this subject later (Sec. 4-2).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(First Fundamental Form of the Helicoid)</span></p>

Consider a helix that is given by (see Example 1, Sec. 1-2) $(\cos u, \sin u, au)$. Through each point of the helix, draw a line parallel to the $xy$ plane and intersecting the $z$ axis. The surface generated by these lines is called a **helicoid** and admits the following parametrization:

$$\mathbf{x}(u, v) = (v\cos u,\; v\sin u,\; au), \qquad 0 < u < 2\pi,\quad -\infty < v < \infty.$$

The computation of the coefficients of the first fundamental form in the above parametrization gives

$$E(u, v) = v^2 + a^2, \qquad F(u, v) = 0, \qquad G(u, v) = 1.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(First Fundamental Form of the Sphere — Geographical Coordinates)</span></p>

We shall compute the first fundamental form of a sphere at a point of the coordinate neighborhood given by the parametrization (cf. Example 1, Sec. 2-2)

$$\mathbf{x}(\theta, \varphi) = (\sin\theta\cos\varphi,\; \sin\theta\sin\varphi,\; \cos\theta).$$

First, observe that

$$\mathbf{x}_\theta = (\cos\theta\cos\varphi,\; \cos\theta\sin\varphi,\; -\sin\theta), \qquad \mathbf{x}_\varphi = (-\sin\theta\sin\varphi,\; \sin\theta\cos\varphi,\; 0).$$

Hence,

$$E(\theta, \varphi) = 1, \qquad F(\theta, \varphi) = 0, \qquad G(\theta, \varphi) = \sin^2\theta.$$

Thus, if $w$ is a tangent vector to the sphere at the point $\mathbf{x}(\theta, \varphi)$, given in the basis associated to $\mathbf{x}(\theta, \varphi)$ by $w = a\mathbf{x}\_\theta + b\mathbf{x}\_\varphi$, then the square of the length of $w$ is given by

$$|w|^2 = I(w) = Ea^2 + 2Fab + Gb^2 = a^2 + b^2\sin^2\theta.$$

As an application, let us determine the curves in this coordinate neighborhood of the sphere which make a constant angle $\beta$ with the meridians $\varphi = \text{const.}$ These curves are called **loxodromes** (rhumb lines) of the sphere. The equation of the loxodromes is obtained from

$$\cos\beta = \frac{\langle \mathbf{x}_\theta,\; \alpha'(t) \rangle}{|\mathbf{x}_\theta|\,|\alpha'(t)|} = \frac{\theta'}{\sqrt{(\theta')^2 + (\varphi')^2\sin^2\theta}},$$

which yields $\theta'/\sin\theta = \pm \varphi'/\tan\beta$, and by integration,

$$\log\tan\left(\frac{\theta}{2}\right) = \pm(\varphi + c)\cot\beta,$$

where $c$ is the constant of integration determined by giving one point $\mathbf{x}(\theta\_0, \varphi\_0)$ through which the curve passes.

</div>

### Area of a Bounded Region

Another metric question that can be treated by the first fundamental form is the computation (or definition) of the area of a bounded region of a regular surface $S$. A (regular) **domain** of $S$ is an open and connected subset of $S$ such that its boundary is the image of a circle by a differentiable homeomorphism which is regular (that is, its differential is nonzero) except at a finite number of points. A **region** of $S$ is the union of a domain with its boundary. A region of $S \subset R^3$ is **bounded** if it is contained in some ball of $R^3$.

We shall consider bounded regions $R$ which are contained in a coordinate neighborhood $\mathbf{x}(U)$ of a parametrization $\mathbf{x}\colon U \subset R^2 \to S$. In other words, $R$ is the image by $\mathbf{x}$ of a bounded region $Q \subset U$. The function $\|\mathbf{x}\_u \wedge \mathbf{x}\_v\|$, defined in $U$, measures the area of the parallelogram generated by the vectors $\mathbf{x}\_u$ and $\mathbf{x}\_v$. One can first show that the integral

$$\int_Q |\mathbf{x}_u \wedge \mathbf{x}_v|\,du\,dv$$

does not depend on the parametrization $\mathbf{x}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Area of a Region)</span></p>

Let $R \subset S$ be a bounded region of a regular surface contained in the coordinate neighborhood of the parametrization $\mathbf{x}\colon U \subset R^2 \to S$. The positive number

$$\iint_Q |\mathbf{x}_u \wedge \mathbf{x}_v|\,du\,dv = A(R), \qquad Q = \mathbf{x}^{-1}(R),$$

is called the **area** of $R$.

</div>

It is convenient to observe that

$$|\mathbf{x}_u \wedge \mathbf{x}_v|^2 + \langle \mathbf{x}_u, \mathbf{x}_v \rangle^2 = |\mathbf{x}_u|^2\,|\mathbf{x}_v|^2,$$

which shows that the integrand of $A(R)$ can be written as

$$|\mathbf{x}_u \wedge \mathbf{x}_v| = \sqrt{EG - F^2}.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Area of the Torus)</span></p>

Let us compute the area of the torus of Example 6, Sec. 2-2. For that, we consider the coordinate neighborhood corresponding to the parametrization

$$\mathbf{x}(u, v) = ((a + r\cos u)\cos v,\; (a + r\cos u)\sin v,\; r\sin u),$$

$$0 < u < 2\pi, \qquad 0 < v < 2\pi,$$

which covers the torus, except for a meridian and a parallel. The coefficients of the first fundamental form are

$$E = r^2, \qquad F = 0, \qquad G = (r\cos u + a)^2;$$

hence,

$$\sqrt{EG - F^2} = r(r\cos u + a).$$

Now, consider the region $R\_\epsilon$ obtained as the image by $\mathbf{x}$ of the region $Q\_\epsilon$ ($\epsilon > 0$ and small),

$$Q_\epsilon = \lbrace (u, v) \in R^2;\; 0 + \epsilon \le u \le 2\pi - \epsilon,\; 0 + \epsilon \le v \le 2\pi - \epsilon \rbrace.$$

Then

$$A(R_\epsilon) = \iint_{Q_\epsilon} r(r\cos u + a)\,du\,dv.$$

Letting $\epsilon \to 0$, we obtain

$$A(T) = \lim_{\epsilon \to 0} A(R_\epsilon) = 4\pi^2 ra.$$

This agrees with the value found by using the theorem of Pappus for the area of surfaces of revolution.

</div>

## 2-6. Orientation of Surfaces

In this section we shall discuss in what sense, and when, it is possible to orient a surface. Intuitively, since every point $p$ of a regular surface $S$ has a tangent plane $T\_p(S)$, the choice of an orientation of $T\_p(S)$ induces an orientation in a neighborhood of $p$, that is, a notion of positive movement along sufficiently small closed curves about each point of the neighborhood. If it is possible to make this choice for each $p \in S$ so that in the intersection of any two neighborhoods the orientations coincide, then $S$ is said to be orientable. If this is not possible, $S$ is called nonorientable.

We shall now make these ideas precise. By fixing a parametrization $\mathbf{x}(u, v)$ of a neighborhood of a point $p$ of a regular surface $S$, we determine an orientation of the tangent plane $T\_p(S)$, namely, the orientation of the associated ordered basis $\lbrace \mathbf{x}\_u, \mathbf{x}\_v \rbrace$. If $p$ belongs to the coordinate neighborhood of another parametrization $\tilde{\mathbf{x}}(\tilde{u}, \tilde{v})$, the new basis $\lbrace \tilde{\mathbf{x}}\_{\tilde{u}}, \tilde{\mathbf{x}}\_{\tilde{v}} \rbrace$ is expressed in terms of the first one by

$$\tilde{\mathbf{x}}_{\tilde{u}} = \mathbf{x}_u \frac{\partial u}{\partial \tilde{u}} + \mathbf{x}_v \frac{\partial v}{\partial \tilde{u}}, \qquad \tilde{\mathbf{x}}_{\tilde{v}} = \mathbf{x}_u \frac{\partial u}{\partial \tilde{v}} + \mathbf{x}_v \frac{\partial v}{\partial \tilde{v}},$$

where $u = u(\tilde{u}, \tilde{v})$ and $v = v(\tilde{u}, \tilde{v})$ are the expressions of the change of coordinates. The bases $\lbrace \mathbf{x}\_u, \mathbf{x}\_v \rbrace$ and $\lbrace \tilde{\mathbf{x}}\_{\tilde{u}}, \tilde{\mathbf{x}}\_{\tilde{v}} \rbrace$ determine, therefore, the same orientation of $T\_p(S)$ if and only if the Jacobian

$$\frac{\partial(u, v)}{\partial(\tilde{u}, \tilde{v})}$$

of the coordinate change is positive.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Orientable Surface)</span></p>

A regular surface $S$ is called **orientable** if it is possible to cover it with a family of coordinate neighborhoods in such a way that if a point $p \in S$ belongs to two neighborhoods of this family, then the change of coordinates has positive Jacobian at $p$. The choice of such a family is called an **orientation** of $S$, and $S$, in this case, is called **oriented**. If such a choice is not possible, the surface is called **nonorientable**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Graphs and Spheres are Orientable)</span></p>

1. A surface which is the graph of a differentiable function (cf. Sec. 2-2, Prop. 1) is an orientable surface. In fact, all surfaces which can be covered by one coordinate neighborhood are trivially orientable.

2. The sphere is an orientable surface. Instead of proceeding by a direct calculation, let us resort to a general argument. The sphere can be covered by two coordinate neighborhoods (using stereographic projection; see Exercise 16, Sec. 2-2), with parameters $(u, v)$ and $(\tilde{u}, \tilde{v})$, in such a way that the intersection $W$ of these neighborhoods (the sphere minus two points) is a connected set. Fix a point $p$ in $W$. If the Jacobian of the coordinate change at $p$ is negative, we interchange $u$ and $v$ in the first system, and the Jacobian becomes positive. Since the Jacobian is different from zero in $W$ and positive at $p \in W$, it follows from the connectedness of $W$ that the Jacobian is everywhere positive. There exists, therefore, a family of coordinate neighborhoods satisfying Def. 1, and so the sphere is orientable.

</div>

By the argument just used, it is clear that *if a regular surface can be covered by two coordinate neighborhoods whose intersection is connected, then the surface is orientable*.

### Orientability via the Normal Vector

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Orientability and the Normal Field)</span></p>

A regular surface $S \subset R^3$ is orientable if and only if there exists a differentiable field of unit normal vectors $N\colon S \to R^3$ on $S$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Sketch</summary>

If $S$ is orientable, it is possible to cover it with a family of coordinate neighborhoods so that, in the intersection of any two of them, the change of coordinates has a positive Jacobian. At the points $p = \mathbf{x}(u, v)$ of each coordinate neighborhood, we define $N(p) = N(u, v)$ by

$$N = \frac{\mathbf{x}_u \wedge \mathbf{x}_v}{|\mathbf{x}_u \wedge \mathbf{x}_v|}.$$

$N(p)$ is well defined, since if $p$ belongs to two coordinate neighborhoods, with parameters $(u, v)$ and $(\tilde{u}, \tilde{v})$, the normal vectors $N(u, v)$ and $N(\tilde{u}, \tilde{v})$ coincide because $\tilde{\mathbf{x}}\_{\tilde{u}} \wedge \tilde{\mathbf{x}}\_{\tilde{v}} = (\mathbf{x}\_u \wedge \mathbf{x}\_v)\,\partial(u, v)/\partial(\tilde{u}, \tilde{v})$ and the Jacobian is positive. Moreover, the coordinates of $N(u, v)$ in $R^3$ are differentiable functions, and thus $N\colon S \to R^3$ is differentiable, as desired.

On the other hand, let $N\colon S \to R^3$ be a differentiable field of unit normal vectors, and consider a family of *connected* coordinate neighborhoods covering $S$. For the points $p = \mathbf{x}(u, v)$ of each coordinate neighborhood $\mathbf{x}(U)$, $U \subset R^2$, it is possible, by the continuity of $N$ and, if necessary, by interchanging $u$ and $v$, to arrange that

$$N(p) = \frac{\mathbf{x}_u \wedge \mathbf{x}_v}{|\mathbf{x}_u \wedge \mathbf{x}_v|}.$$

Proceeding in this manner with all the coordinate neighborhoods, we have that in the intersection of any two of them, say $\mathbf{x}(u, v)$ and $\tilde{\mathbf{x}}(\tilde{u}, \tilde{v})$, the Jacobian $\partial(u, v)/\partial(\tilde{u}, \tilde{v})$ is certainly positive; otherwise, we would have $N(p) = -N(p)$, which is a contradiction. Hence, the given family of coordinate neighborhoods, after undergoing certain interchanges of $u$ and $v$, satisfies the conditions of Def. 1, and $S$ is, therefore, orientable. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Continuous vs. Differentiable Normal Fields)</span></p>

As the proof shows, we need only to require the existence of a *continuous* unit vector field on $S$ for $S$ to be orientable. Such a vector field will be automatically differentiable.

</div>

### The Möbius Strip — A Nonorientable Surface

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Möbius Strip)</span></p>

The **Möbius strip** is obtained by considering the circle $S^1$ given by $x^2 + y^2 = 4$ and the open segment $AB$ given in the $yz$ plane by $y - 2$, $\vert z\vert  < 1$. We move the center $c$ of $AB$ along $S^1$ and turn $AB$ about $c$ in the $cz$ plane in such a manner that when $c$ has passed through an angle $u$, $AB$ has been rotated by an angle $u/2$. When $c$ completes one trip around the circle, $AB$ returns to its initial position, with its end points inverted.

A system of coordinates $\mathbf{x}\colon U \to M$ for the Möbius strip is given by

$$\mathbf{x}(u, v) = \left(\left(2 - v\sin\frac{u}{2}\right)\sin u,\; \left(2 - v\sin\frac{u}{2}\right)\cos u,\; v\cos\frac{u}{2}\right),$$

where $0 < u < 2\pi$ and $-1 < v < 1$. By taking the origin of the $u$'s at the $x$ axis, we obtain another parametrization $\tilde{\mathbf{x}}(\tilde{u}, \tilde{v})$ whose coordinate neighborhood omits the points of the open interval $u = \pi/2$. These two coordinate neighborhoods cover the Möbius strip and can be used to show that it is a regular surface.

The intersection of the two coordinate neighborhoods is not connected but consists of two connected components:

$$W_1 = \left\lbrace \mathbf{x}(u, v)\colon \frac{\pi}{2} < u < 2\pi \right\rbrace, \qquad W_2 = \left\lbrace \mathbf{x}(u, v)\colon 0 < u < \frac{\pi}{2} \right\rbrace.$$

The change of coordinates is given by

$$\tilde{u} = u - \frac{\pi}{2},\quad \tilde{v} = v \qquad \text{in } W_1,$$

and

$$\tilde{u} = \frac{3\pi}{2} + u,\quad \tilde{v} = -v \qquad \text{in } W_2.$$

It follows that the Jacobian is $+1 > 0$ in $W\_1$ and $-1 < 0$ in $W\_2$.

To show that the Möbius strip is nonorientable, we suppose that it is possible to define a differentiable field of unit normal vectors $N\colon M \to R^3$. Interchanging $u$ and $v$ if necessary, we can assume that $N(p) = \mathbf{x}\_u \wedge \mathbf{x}\_v / \|\mathbf{x}\_u \wedge \mathbf{x}\_v\|$ for any $p$ in the coordinate neighborhood of $\mathbf{x}(u, v)$. Analogously, we may assume that $N(p) = \tilde{\mathbf{x}}\_{\tilde{u}} \wedge \tilde{\mathbf{x}}\_{\tilde{v}} / \|\tilde{\mathbf{x}}\_{\tilde{u}} \wedge \tilde{\mathbf{x}}\_{\tilde{v}}\|$ at all points of the coordinate neighborhood of $\tilde{\mathbf{x}}(\tilde{u}, \tilde{v})$. However, the Jacobian of the change of coordinates must be $-1$ in either $W\_1$ or $W\_2$ (depending on what changes of the type $u \to v$, $\tilde{u} \to \tilde{v}$ has to be made). If $p$ is a point of that component of the intersection, then $N(p) = -N(p)$, which is a contradiction.

</div>

### Orientability of Level Surfaces

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Level Surfaces are Orientable)</span></p>

If a regular surface is given by $S = \lbrace (x, y, z) \in R^3;\; f(x, y, z) = a \rbrace$, where $f\colon U \subset R^3 \to R$ is differentiable and $a$ is a regular value of $f$, then $S$ is orientable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Given a point $(x\_0, y\_0, z\_0) = p \in S$, consider the parametrized curve $(x(t), y(t), z(t))$, $t \in I$, on $S$ passing through $p$ for $t = t\_0$. Since the curve is on $S$, we have $f(x(t), y(t), z(t)) = a$ for all $t \in I$. By differentiating both sides with respect to $t$, we see at $t = t\_0$

$$f_x(p)\left(\frac{dx}{dt}\right)_{t_0} + f_y(p)\left(\frac{dy}{dt}\right)_{t_0} + f_z(p)\left(\frac{dz}{dt}\right)_{t_0} = 0.$$

This shows that the tangent vector to the curve at $t = t\_0$ is perpendicular to the vector $(f\_x, f\_y, f\_z)$ at $p$. Since the curve and the point are arbitrary, we conclude that

$$N(x, y, z) = \frac{(f_x,\; f_y,\; f_z)}{\sqrt{f_x^2 + f_y^2 + f_z^2}}$$

is a differentiable field of unit normal vectors on $S$. Together with Prop. 1, this implies that $S$ is orientable as desired. **Q.E.D.**

</details>
</div>

A final remark. Orientation is definitely not a local property of a regular surface. Locally, every regular surface is diffeomorphic to an open set in the plane, and hence orientable. Orientation is a global property, in the sense that it involves the whole surface. We shall have more to say about global properties later in this book (Chap. 5).

## 2-7. A Characterization of Compact Orientable Surfaces

The converse of Prop. 2 of Sec. 2-6, namely, that *an orientable surface in $R^3$ is the inverse image of a regular value of some differentiable function*, is true and nontrivial to prove. Even in the particular case of compact surfaces (defined in this section), the proof is instructive and offers an interesting example of a global theorem in differential geometry. This section will be dedicated entirely to the proof of this converse statement.

### Tubular Neighborhoods

Let $S \subset R^3$ be an orientable surface. The crucial point of the proof consists of showing that one may choose, on the normal line through $p \in S$, an open interval $I\_p$ around $p$ of length, say, $2\epsilon\_p$ ($\epsilon\_p$ varies with $p$) in such a way that if $p \neq q \subset S$, then $I\_p \cap I\_q = \emptyset$. Thus, the union $\bigcup I\_p$, $p \in S$, constitutes an open set $V$ of $R^3$ which contains $S$ and has the property that through each point of $V$ there passes a unique normal line to $S$; $V$ is then called a **tubular neighborhood** of $S$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Local Tubular Neighborhood)</span></p>

Let $S$ be a regular surface and $\mathbf{x}\colon U \to S$ be a parametrization of a neighborhood of a point $p = \mathbf{x}(u\_0, v\_0) \in S$. Then there exists a neighborhood $W \subset \mathbf{x}(U)$ of $p$ in $S$ and a number $\epsilon > 0$ such that the segments of the normal lines passing through points $q \in W$, with center at $q$ and length $2\epsilon$, are disjoint (that is, $W$ has a tubular neighborhood).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Consider the map $F\colon U \times R \to R^3$ given by

$$F(u, v;\; t) = \mathbf{x}(u, v) + tN(u, v), \qquad (u, v) \in U,\; t \in R,$$

where $N(u, v) = (N\_x, N\_y, N\_z)$ is the unit normal vector at $\mathbf{x}(u, v) = (x(u, v), y(u, v), z(u, v))$. Geometrically, $F$ maps the point $(u, v; t)$ of the "cylinder" $U \times R$ in the point of the normal line to $S$ at a distance $t$ from $\mathbf{x}(u, v)$. $F$ is clearly differentiable and its Jacobian at $t = 0$ is given by

$$\begin{vmatrix} \frac{\partial x}{\partial u} & \frac{\partial y}{\partial u} & \frac{\partial z}{\partial u} \\[4pt] \frac{\partial x}{\partial v} & \frac{\partial y}{\partial v} & \frac{\partial z}{\partial v} \\[4pt] N_x & N_y & N_z \end{vmatrix} = |\mathbf{x}_u \wedge \mathbf{x}_v| \neq 0.$$

By the inverse function theorem, there exists a parallelepiped in $U \times R$,

$$u_0 - \delta < u < u_0 + \delta, \qquad v_0 - \delta < v < v_0 + \delta, \qquad -\epsilon < t < \epsilon,$$

restricted to which $F$ is one-to-one. But this means that in the image $W$ by $F$ of the rectangle $u\_0 - \delta < u < u\_0 + \delta$, $v\_0 - \delta < v < v\_0 + \delta$, the segments of the normal lines with centers $q \in W$ and of length $< 2\epsilon$ do not meet. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(The Signed Distance Function)</span></p>

Assume the existence of a tubular neighborhood $V$ of an orientable surface $S \subset R^3$, and choose an orientation for $S$. Then the function $g\colon V \to R$, defined as the oriented distance from a point of $V$ to the foot of the unique normal line passing through this point, is differentiable and has zero as a regular value.

</div>

### Compact Sets and Compact Surfaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Compact Set)</span></p>

Let $A$ be a subset of $R^3$. We say that $p \in R^3$ is a **limit point** of $A$ if every neighborhood of $p$ in $R^3$ contains a point of $A$ distinct from $p$. $A$ is said to be **closed** if it contains all its limit points. $A$ is **bounded** if it is contained in some ball of $R^3$. If $A$ is closed and bounded, it is called a **compact set**.

</div>

The sphere and the torus are compact surfaces. The paraboloid of revolution $z = x^2 + y^2$, $(x, y) \in R^2$, is a closed surface, but, being unbounded, it is not a compact surface. The disk $x^2 + y^2 < 1$ in the plane and the Möbius strip are bounded but not closed and therefore are noncompact.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Compact Sets in $R^3$)</span></p>

1. **(Bolzano-Weierstrass)** Let $A \subset R^3$ be a compact set. Then every infinite subset of $A$ has at least one limit point in $A$.

2. **(Heine-Borel)** Let $A \subset R^3$ be a compact set and $\lbrace U\_\alpha \rbrace$ be a family of open sets of $A$ such that $\bigcup\_\alpha U\_\alpha = A$. Then it is possible to choose a finite number $U\_{k\_1}, U\_{k\_2}, \dots, U\_{k\_n}$ of $U\_\alpha$ such that $\bigcup U\_{k\_i} = A$, $i = 1, \dots, n$.

3. **(Lebesgue)** Let $A \subset R^3$ be a compact set and $\lbrace U\_\alpha \rbrace$ a family of open sets of $A$ such that $\bigcup\_\alpha U\_\alpha = A$. Then there exists a number $\delta > 0$ (the **Lebesgue number** of the family $\lbrace U\_\alpha \rbrace$) such that whenever two points $p, q \in A$ are at a distance $d(p, q) < \delta$ then $p$ and $q$ belong to some $U\_\alpha$.

</div>

### The Main Theorem

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Global Tubular Neighborhood for Compact Surfaces)</span></p>

Let $S \subset R^3$ be a regular, compact, orientable surface. Then there exists a number $\epsilon > 0$ such that whenever $p, q \in S$ the segments of the normal lines of length $2\epsilon$, centered in $p$ and $q$, are disjoint (that is, $S$ has a tubular neighborhood).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By Prop. 1, for each $p \in S$ there exists a neighborhood $W\_p$ and a number $\epsilon\_p > 0$ such that the proposition holds for points of $W\_p$ with $\epsilon = \epsilon\_p$. Letting $p$ run through $S$, we obtain a family $\lbrace W\_p \rbrace$ with $\bigcup\_{p \in S} W\_p = S$. By compactness (Property 2), it is possible to choose a finite number of the $W\_p$'s, say, $W\_1, \dots, W\_k$ (corresponding to $\epsilon\_1, \dots, \epsilon\_k$) such that $\bigcup W\_i = S$, $i = 1, \dots, k$. We shall show that the required $\epsilon$ is given by

$$\epsilon < \min\left(\epsilon_1, \dots, \epsilon_k,\; \frac{\delta}{2}\right),$$

where $\delta$ is the Lebesgue number of the family $\lbrace W\_i \rbrace$ (Property 3).

In fact, let two points $p, q \in S$. If both belong to some $W\_i$, $i = 1, \dots, k$, the segments of the normal lines with centers in $p$ and $q$ and of length $2\epsilon$ do not meet, since $\epsilon < \epsilon\_i$. If $p$ and $q$ do not belong to the same $W\_i$, then $d(p, q) \ge \delta$; were the segments of the normal lines, centered in $p$ and $q$ and of length $2\epsilon$, to meet at point $Q \in R^3$, we would have

$$2\epsilon \ge d(p, Q) + d(Q, q) \ge d(p, q) \ge \delta,$$

which contradicts the definition of $\epsilon$. **Q.E.D.**

</details>
</div>

Putting together Props. 1, 2, and 3, we obtain the main goal of this section.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Characterization of Compact Orientable Surfaces)</span></p>

Let $S \subset R^3$ be a regular compact orientable surface. Then there exists a differentiable function $g\colon V \to R$, defined in an open set $V \subset R^3$, with $V \supset S$ (precisely a tubular neighborhood of $S$), which has zero as a regular value and is such that $S = g^{-1}(0)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1</span><span class="math-callout__name">(Noncompact Case)</span></p>

It is possible to prove the existence of a tubular neighborhood of an orientable surface, even if the surface is not compact; the theorem is true, therefore, without the restriction of compactness. The proof is, however, more technical. In this general case, the $\epsilon(p) > 0$ is not constant as in the compact case but may vary with $p$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2</span><span class="math-callout__name">(Compact Surfaces are Automatically Orientable)</span></p>

It is possible to prove that a regular compact surface in $R^3$ is orientable; the hypothesis of orientability in the theorem (the compact case) is therefore unnecessary. A proof of this fact can be found in H. Samelson, "Orientability of Hypersurfaces in $R^n$," *Proc. A.M.S.* 22 (1969), 301-302.

</div>

## 2-8. A Geometric Definition of Area

In this section we shall present a geometric justification for the definition of area given in Sec. 2-5. More precisely, we shall give a geometric definition of area and shall prove that in the case of a bounded region of a regular surface such a definition leads to the formula given for the area in Sec. 2-5.

To define the area of a region $R \subset S$ we shall start with a **partition** $\mathcal{P}$ of $R$ into a finite number of regions $R\_i$, that is, we write $R = \bigcup\_i R\_i$, where the intersection of two such regions $R\_i$ is either empty or made up of boundary points of both regions. The **diameter** of $R\_i$ is the supremum of the distances (in $R^3$) of any two points in $R\_i$; the largest diameter of the $R\_i$'s of a given partition $\mathcal{P}$ is called the **norm** $\mu$ of $\mathcal{P}$.

Given a partition $R = \bigcup\_i R\_i$, we choose arbitrarily points $p\_i \in R\_i$ and project $R\_i$ onto the tangent plane at $p\_i$ in the direction of the normal line at $p\_i$; this projection is denoted by $\bar{R}\_i$ and its area by $A(\bar{R}\_i)$. The sum $\sum\_i A(\bar{R}\_i)$ is an approximation of what we understand intuitively by the area of $R$.

If, by choosing partitions $\mathcal{P}\_1, \dots, \mathcal{P}\_n, \dots$ more and more refined and such that the norm $\mu\_n$ of $\mathcal{P}\_n$ converges to zero, there exists a limit of $\sum\_i A(\bar{R}\_i)$ and this limit is independent of all choices, then we say that $R$ has an **area** $A(R)$ defined by

$$A(R) = \lim_{\mu_n \to 0} \sum_i A(\bar{R}_i).$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Geometric Area Equals Integral Formula)</span></p>

Let $\mathbf{x}\colon U \to S$ be a coordinate system in a regular surface $S$ and let $R = \mathbf{x}(Q)$ be a bounded region of $S$ contained in $\mathbf{x}(U)$. Then $R$ has an area given by

$$A(R) = \iint_Q |\mathbf{x}_u \wedge \mathbf{x}_v|\,du\,dv.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Sketch</summary>

Consider a partition $R = \bigcup\_i R\_i$ and choose a point $p\_i \in R\_i = \mathbf{x}(Q\_i)$. We want to compute the area of the normal projection $\bar{R}\_i$ of $R\_i$ onto the tangent plane at $p\_i$. To do this, consider a new system of axes $p\_i \bar{x}\bar{y}\bar{z}$ in $R^3$, obtained from $Oxyz$ by a translation followed by a rotation which takes the $z$ axis into the normal line at $p\_i$ in such a way that both systems have the same orientation. In the new axes, the parametrization can be written

$$\bar{\mathbf{x}}(u, v) = (\bar{x}(u, v),\; \bar{y}(u, v),\; \bar{z}(u, v)),$$

where $\bar{\mathbf{x}}(u, v)$ is obtained from $\mathbf{x}(u, v)$ by a translation followed by an orthogonal linear map. At $p\_i$, the vectors $\bar{\mathbf{x}}\_u$ and $\bar{\mathbf{x}}\_v$ belong to the $\bar{x}\bar{y}$ plane; therefore, $\partial\bar{z}/\partial u = \partial\bar{z}/\partial v = 0$ at $p\_i$; hence,

$$\left|\frac{\partial(\bar{x}, \bar{y})}{\partial(u, v)}\right| = \left|\frac{\partial \bar{\mathbf{x}}}{\partial u} \wedge \frac{\partial \bar{\mathbf{x}}}{\partial v}\right| \qquad \text{at } p_i.$$

The area of the projection is $A(\bar{R}\_i) = \iint\_{R\_i} d\bar{x}\,d\bar{y}$. Since $\partial(\bar{x}, \bar{y})/\partial(u, v) \neq 0$, we can use the change of coordinates $\bar{x} = \bar{x}(u, v)$, $\bar{y} = \bar{y}(u, v)$ and transform this into

$$A(\bar{R}_i) = \iint_{Q_i} \frac{\partial(\bar{x}, \bar{y})}{\partial(u, v)}\,du\,dv.$$

Since the length of a vector is preserved by translations and orthogonal linear maps, we obtain

$$\left|\frac{\partial \mathbf{x}}{\partial u} \wedge \frac{\partial \mathbf{x}}{\partial v}\right| = \left|\frac{\partial \bar{\mathbf{x}}}{\partial u} \wedge \frac{\partial \bar{\mathbf{x}}}{\partial v}\right| = \left|\frac{\partial(\bar{x}, \bar{y})}{\partial(u, v)}\right| - \epsilon_i(u, v),$$

where $\epsilon\_i(u, v)$ is a continuous function in $Q\_i$ with $\epsilon\_i(\mathbf{x}^{-1}(p\_i)) = 0$. Let $M\_i$ and $m\_i$ be the maximum and the minimum of $\epsilon\_i$ in $Q\_i$; then

$$m_i \le \left|\frac{\partial(\bar{x}, \bar{y})}{\partial(u, v)}\right| - \left|\frac{\partial \mathbf{x}}{\partial u} \wedge \frac{\partial \mathbf{x}}{\partial v}\right| \le M_i.$$

Summing over all $R\_i$, we obtain

$$\sum_i m_i A(Q_i) \le \sum_i A(\bar{R}_i) - \iint_Q |\mathbf{x}_u \wedge \mathbf{x}_v|\,du\,dv \le \sum_i M_i A(Q_i).$$

Now, refine more and more the given partition in such a way that the norm $\mu \to 0$. Then $M\_i \to m\_i$. Therefore, there exists the limit of $\sum\_i A(\bar{R}\_i)$, given by

$$A(R) = \iint_Q \left|\frac{\partial \mathbf{x}}{\partial u} \wedge \frac{\partial \mathbf{x}}{\partial v}\right| du\,dv,$$

which is clearly independent of the choice of the partitions and of the point $p\_i$ in each partition. **Q.E.D.**

</details>
</div>

## Appendix to Chapter 2 — A Brief Review of Continuity and Differentiability

$R^n$ will denote the set of $n$-tuples $(x\_1, \dots, x\_n)$ of real numbers. Although we use only the cases $R^1 = R$, $R^2$, and $R^3$, the more general notion of $R^n$ unifies the definitions and brings in no additional difficulties.

### A. Continuity in $R^n$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Open Ball)</span></p>

A **ball** (or **open ball**) in $R^n$ with center $p\_0 = (x\_1^0, \dots, x\_n^0)$ and radius $\epsilon > 0$ is the set

$$B_\epsilon(p_0) = \lbrace (x_1, \dots, x_n) \in R^n;\; (x_1 - x_1^0)^2 + \cdots + (x_n - x_n^0)^2 < \epsilon^2 \rbrace.$$

Thus, in $R$, $B\_\epsilon(p\_0)$ is an open interval with center $p\_0$ and length $2\epsilon$; in $R^2$, $B\_\epsilon(p\_0)$ is the interior of a disk with center $p\_0$ and radius $\epsilon$; in $R^3$, $B\_\epsilon(p\_0)$ is the interior of a region bounded by a sphere of center $p\_0$ and radius $\epsilon$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Open Set, Neighborhood)</span></p>

A set $U \subset R^n$ is an **open set** if for each $p \in U$ there is a ball $B\_\epsilon(p) \subset U$; intuitively this means that points in $U$ are entirely surrounded by points of $U$, or that points sufficiently close to points of $U$ still belong to $U$. An open set in $R^n$ containing a point $p \in R^n$ is called a **neighborhood** of $p$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Continuity of Maps)</span></p>

A map $F\colon U \subset R^n \to R^m$ is **continuous** at $p \in U$ if given $\epsilon > 0$, there exists $\delta > 0$ such that

$$F(B_\delta(p)) \subset B_\epsilon(F(p)).$$

In other words, $F$ is continuous at $p$ if points arbitrarily close to $p$ are images of points sufficiently close to $p$. We say that $F$ is **continuous in $U$** if $F$ is continuous for all $p \in U$.

</div>

Given a map $F\colon U \subset R^n \to R^m$, we can determine $m$ functions of $n$ variables as follows. Let $p = (x\_1, \dots, x\_n) \in U$ and $F(p) = (y\_1, \dots, y\_m)$. Then we can write

$$y_1 = f_1(x_1, \dots, x_n), \quad \dots, \quad y_m = f_m(x_1, \dots, x_n).$$

The functions $f\_i\colon U \to R$, $i = 1, \dots, m$, are the **component functions** of $F$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Continuity via Components)</span></p>

$F\colon U \subset R^n \to R^m$ is continuous if and only if each component function $f\_i\colon U \subset R^n \to R$, $i = 1, \dots, m$, is continuous.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Continuity via Neighborhoods)</span></p>

A map $F\colon U \subset R^n \to R^m$ is continuous at $p \in U$ if and only if, given a neighborhood $V$ of $F(p)$ in $R^m$, there exists a neighborhood $W$ of $p$ in $R^n$ such that $F(W) \subset V$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Composition of Continuous Maps)</span></p>

Let $F\colon U \subset R^n \to R^m$ and $G\colon V \subset R^m \to R^k$ be continuous maps, where $U$ and $V$ are open sets such that $F(U) \subset V$. Then $G \circ F\colon U \subset R^n \to R^k$ is a continuous map.

</div>

### Continuity on Arbitrary Sets

It is often necessary to deal with maps defined on arbitrary (not necessarily open) sets of $R^n$. Let $F\colon A \subset R^n \to R^m$ be a map, where $A$ is an arbitrary set in $R^n$. We say that $F$ is **continuous in $A$** if there exists an open set $U \subset R^n$, $U \supset A$, and a continuous map $\bar{F}\colon U \to R^m$ such that the restriction $\bar{F}\vert \_A = F$. In other words, $F$ is continuous in $A$ if it is the restriction of a continuous map defined in an open set containing $A$.

Given a neighborhood $V$ of $F(p)$ in $R^m$, $p \in A$, there exists a neighborhood $W$ of $p$ in $R^n$ such that $F(W \cap A) \subset V$. For this reason, it is convenient to call the set $W \cap A$ a **neighborhood of $p$ in $A$**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Homeomorphism)</span></p>

We say that a continuous map $F\colon A \subset R^n \to R^n$ is a **homeomorphism** if $F$ is one-to-one and the inverse $F^{-1}\colon F(A) \subset R^n \to R^n$ is continuous. In this case $A$ and $F(A)$ are **homeomorphic sets**.

</div>

### Properties of Real Continuous Functions

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4</span><span class="math-callout__name">(Intermediate Value Theorem)</span></p>

Let $f\colon [a, b] \to R$ be a continuous function defined on the closed interval $[a, b]$. Assume that $f(a)$ and $f(b)$ have opposite signs; that is, $f(a)f(b) < 0$. Then there exists a point $c \in (a, b)$ such that $f(c) = 0$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5</span><span class="math-callout__name">(Extreme Value Theorem)</span></p>

Let $f\colon [a, b] \to R$ be a continuous function defined in the closed interval $[a, b]$. Then $f$ reaches its maximum and its minimum in $[a, b]$; that is, there exist points $x\_1, x\_2 \in [a, b]$ such that $f(x\_1) \le f(x) \le f(x\_2)$ for all $x \in [a, b]$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6</span><span class="math-callout__name">(Heine-Borel for Intervals)</span></p>

Let $[a, b]$ be a closed interval and let $I\_\alpha$, $\alpha \in A$, be a collection of open intervals in $[a, b]$ such that $\bigcup\_\alpha I\_\alpha = [a, b]$. Then it is possible to choose a finite number $I\_{k\_1}, I\_{k\_2}, \dots, I\_{k\_n}$ of $I\_\alpha$ such that $\bigcup I\_{k\_i} = [a, b]$, $i = 1, \dots, n$.

</div>

### B. Differentiability in $R^n$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Differentiable Function)</span></p>

Let $f\colon U \subset R \to R$. The **derivative** of $f$ at $x\_0 \in U$ is the limit (when it exists)

$$f'(x_0) = \lim_{h \to 0} \frac{f(x_0 + h) - f(x_0)}{h}, \qquad x_0 + h \in U.$$

When $f$ has derivatives at all points of a neighborhood $V$ of $x\_0$, we can consider the derivative of $f'\colon V \to R$ at $x\_0$, which is called the **second derivative** $f''(x\_0)$ of $f$ at $x\_0$, and so forth. $f$ is **differentiable** at $x\_0$ if it has continuous derivatives of all orders at $x\_0$. $f$ is **differentiable** in $U$ if it is differentiable at all points in $U$.

</div>

We use the word differentiable for what is sometimes called infinitely differentiable (or of class $C^\infty$). Our usage should not be confused with the usage of elementary calculus, where a function is called differentiable if its first derivative exists.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Partial Derivatives)</span></p>

Let $f\colon U \subset R^2 \to R$. The **partial derivative** of $f$ with respect to $x$ at $(x\_0, y\_0) \in U$, denoted by $(\partial f/\partial x)(x\_0, y\_0)$, is (when it exists) the derivative at $x\_0$ of the function of one variable $x \to f(x, y\_0)$. Similarly, the partial derivative with respect to $y$, $(\partial f/\partial y)(x\_0, y\_0)$, is defined as the derivative at $y\_0$ of $y \to f(x\_0, y)$.

When $f$ has partial derivatives at all points of a neighborhood $V$ of $(x\_0, y\_0)$, we can consider the **second partial derivatives** at $(x\_0, y\_0)$:

$$\frac{\partial^2 f}{\partial x^2}, \qquad \frac{\partial^2 f}{\partial x\,\partial y}, \qquad \frac{\partial^2 f}{\partial y\,\partial x}, \qquad \frac{\partial^2 f}{\partial y^2},$$

and so forth. $f$ is **differentiable** at $(x\_0, y\_0)$ if it has continuous partial derivatives of all orders at $(x\_0, y\_0)$. $f$ is **differentiable** in $U$ if it is differentiable at all points of $U$. We sometimes denote partial derivatives by $f\_x$, $f\_y$, $f\_{xx}$, $f\_{xy}$, $f\_{yy}$, etc.

</div>

It is an important fact that when $f$ is differentiable the partial derivatives of $f$ are independent of the order in which they are performed; that is,

$$\frac{\partial^2 f}{\partial x\,\partial y} = \frac{\partial^2 f}{\partial y\,\partial x}, \qquad \frac{\partial^3 f}{\partial^2 x\,\partial y} = \frac{\partial^3 f}{\partial x\,\partial y\,\partial x}, \qquad \text{etc.}$$

A further important fact is that partial derivatives obey the so-called **chain rule**. For instance, if $x = x(u, v)$, $y = y(u, v)$, $z = z(u, v)$ are real differentiable functions in $U \subset R^2$ and $f(x, y, z)$ is a real differentiable function in $R^3$, then the composition $f(x(u, v), y(u, v), z(u, v))$ is a differentiable function in $U$, and the partial derivative of $f$ with respect to, say, $u$ is given by

$$\frac{\partial f}{\partial u} = \frac{\partial f}{\partial x}\frac{\partial x}{\partial u} + \frac{\partial f}{\partial y}\frac{\partial y}{\partial u} + \frac{\partial f}{\partial z}\frac{\partial z}{\partial u}.$$

### Differentiable Maps and the Differential

We say that $F\colon U \subset R^n \to R^m$ is **differentiable** at $p \in U$ if its component functions $f\_i$, $i = 1, \dots, m$, have continuous partial derivatives of all orders at $p$. $F$ is **differentiable** in $U$ if it is differentiable at all points of $U$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Differential of a Map)</span></p>

Let $F\colon U \subset R^n \to R^m$ be a differentiable map. To each $p \in U$ we associate a linear map $dF\_p\colon R^n \to R^m$ which is called the **differential** of $F$ at $p$ and is defined as follows. Let $w \in R^n$ and let $\alpha\colon (-\epsilon, \epsilon) \to U$ be a differentiable curve such that $\alpha(0) = p$, $\alpha'(0) = w$. By the chain rule, the curve $\beta = F \circ \alpha\colon (-\epsilon, \epsilon) \to R^m$ is also differentiable. Then

$$dF_p(w) = \beta'(0).$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7</span><span class="math-callout__name">(The Differential is Well-Defined and Linear)</span></p>

The above definition of $dF\_p$ does not depend on the choice of the curve which passes through $p$ with tangent vector $w$, and $dF\_p$ is, in fact, a linear map.

</div>

The matrix of $dF\_p\colon R^n \to R^m$ in the canonical bases of $R^n$ and $R^m$, that is, the matrix $(\partial f\_i / \partial x\_j)$, $i = 1, \dots, m$, $j = 1, \dots, n$, is called the **Jacobian matrix** of $F$ at $p$. When $n = m$, this is a square matrix and its determinant is called the **Jacobian determinant**; it is usual to denote it by

$$\det\left(\frac{\partial f_i}{\partial x_j}\right) = \frac{\partial(f_1, \dots, f_n)}{\partial(x_1, \dots, x_n)}.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8</span><span class="math-callout__name">(Chain Rule for Maps)</span></p>

Let $F\colon U \subset R^n \to R^m$ and $G\colon V \subset R^m \to R^k$ be differentiable maps, where $U$ and $V$ are open sets such that $F(U) \subset V$. Then $G \circ F\colon U \to R^k$ is a differentiable map, and

$$d(G \circ F)_p = dG_{F(p)} \circ dF_p, \qquad p \in U.$$

</div>

Note that, for the particular situation where $F\colon U \subset R^2 \to R^3$ and $G\colon V \subset R^3 \to R^2$, the relation $d(G \circ F)\_p = dG\_{F(p)} \circ dF\_p$ is equivalent to the following product of Jacobian matrices:

$$\begin{pmatrix} \frac{\partial \xi}{\partial u} & \frac{\partial \xi}{\partial v} \\[4pt] \frac{\partial \eta}{\partial u} & \frac{\partial \eta}{\partial v} \end{pmatrix} = \begin{pmatrix} \frac{\partial \xi}{\partial x} & \frac{\partial \xi}{\partial y} & \frac{\partial \xi}{\partial z} \\[4pt] \frac{\partial \eta}{\partial x} & \frac{\partial \eta}{\partial y} & \frac{\partial \eta}{\partial z} \end{pmatrix} \begin{pmatrix} \frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} \\[4pt] \frac{\partial y}{\partial u} & \frac{\partial y}{\partial v} \\[4pt] \frac{\partial z}{\partial u} & \frac{\partial z}{\partial v} \end{pmatrix},$$

which contains the expressions of all partial derivatives $\partial \xi / \partial u$, $\partial \xi / \partial v$, $\partial \eta / \partial u$, $\partial \eta / \partial v$.

### Connectedness and Constant Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Connected Set)</span></p>

An open set $U \subset R^n$ is **connected** if given two points $p, q \in U$ there exists a continuous map $\alpha\colon [a, b] \to U$ such that $\alpha(a) = p$ and $\alpha(b) = q$. This means that two points of $U$ can be joined by a continuous curve in $U$ or that $U$ is made up of one single "piece."

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9</span><span class="math-callout__name">(Zero Differential Implies Constant)</span></p>

Let $f\colon U \subset R^n \to R$ be a differentiable function defined on a connected open subset $U$ of $R^n$. Assume that $df\_p\colon R^n \to R$ is zero at every point $p \in U$. Then $f$ is constant on $U$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Sketch</summary>

The proof proceeds in two steps. First, one shows the result *locally*: Let $p \in U$ and let $B\_\delta(p) \subset U$ be an open ball contained in $U$. Any point $q \in B\_\delta(p)$ can be joined to $p$ by the "radial" segment $\beta(t) = tp + (1-t)q$, $t \in [0, 1]$. Now, $f \circ \beta\colon [0, 1] \to R$ is a function defined in an open interval, and

$$d(f \circ \beta)_t = (df \circ d\beta)_t = 0,$$

since $df \equiv 0$. Thus, $(d/dt)(f \circ \beta) = 0$ for all $t$, and hence $f \circ \beta$ is constant, so $f(p) = f(q)$. Thus, $f$ is constant on $B\_\delta(p)$.

Second, one uses **connectedness** to globalize. Let $r$ be an arbitrary point of $U$. Since $U$ is connected, there exists a continuous curve $\alpha\colon [a, b] \to U$ with $\alpha(a) = p$, $\alpha(b) = r$. The function $f \circ \alpha\colon [a, b] \to R$ is continuous. By the first part, for each $t \in [a, b]$ there exists an interval $I\_t$ open in $[a, b]$ such that $f \circ \alpha$ is constant on $I\_t$. Since $\bigcup\_t I\_t = [a, b]$, we can apply the Heine-Borel theorem (Prop. 6) to choose a finite subcover $I\_1, \dots, I\_k$. Since consecutive intervals overlap, $f \circ \alpha$ is constant on the union of two consecutive intervals, hence on all of $[a, b]$. It follows that $f(\alpha(a)) = f(p) = f(\alpha(b)) = f(r)$. Since $r$ is arbitrary, $f$ is constant on $U$. **Q.E.D.**

</details>
</div>

### The Inverse Function Theorem

One of the most important theorems of differential calculus is the so-called inverse function theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Inverse Function Theorem)</span></p>

Let $F\colon U \subset R^n \to R^n$ be a differentiable mapping and suppose that at $p \in U$ the differential $dF\_p\colon R^n \to R^n$ is an isomorphism. Then there exists a neighborhood $V$ of $p$ in $U$ and a neighborhood $W$ of $F(p)$ in $R^n$ such that $F\colon V \to W$ has a differentiable inverse $F^{-1}\colon W \to V$.

</div>

A differentiable mapping $F\colon V \subset R^n \to W \subset R^n$, where $V$ and $W$ are open sets, is called a **diffeomorphism** of $V$ with $W$ if $F$ has a differentiable inverse. The inverse function theorem asserts that if at a point $p \in U$ the differential $dF\_p$ is an isomorphism, then $F$ is a diffeomorphism in a neighborhood of $p$. In other words, an assertion about the differential of $F$ at a point implies a similar assertion about the behavior of $F$ in a neighborhood of the point.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Local but Not Global Diffeomorphism)</span></p>

Let $F\colon R^2 \to R^2$ be given by

$$F(x, y) = (e^x \cos y,\; e^x \sin y), \qquad (x, y) \in R^2.$$

The component functions $u(x, y) = e^x\cos y$, $v(x, y) = e^x\sin y$ have continuous partial derivatives of all orders, so $F$ is differentiable. Its Jacobian matrix is

$$dF_{(x,y)} = \begin{pmatrix} e^x\cos y & -e^x\sin y \\ e^x\sin y & e^x\cos y \end{pmatrix},$$

and the Jacobian determinant $\det(dF\_{(x,y)}) = e^{2x} \neq 0$ for all $(x, y) \in R^2$. By the inverse function theorem, $F$ is locally a diffeomorphism. Geometrically, the vertical line $x = x\_0$ is mapped into the circle $u^2 + v^2 = e^{2x\_0}$, and the horizontal line $y = y\_0$ is mapped into the half-line from the origin with slope $\tan y\_0$.

However, $F(x, y) = F(x, y + 2\pi)$, so $F$ is not one-to-one globally. For each $p \in R^2$, the inverse function theorem gives neighborhoods $V$ of $p$ and $W$ of $F(p)$ so that $F\colon V \to W$ is a diffeomorphism. In our case, $V$ may be taken as the strip $\lbrace -\infty < x < \infty,\; 0 < y < 2\pi \rbrace$ and $W$ as $R^2 - \lbrace (0, 0) \rbrace$. But a global inverse of $F$ may fail to exist even though the conditions of the theorem are satisfied everywhere.

</div>

# Chapter 3 — The Geometry of the Gauss Map

## 3-1. Introduction

As we have seen in Chap. 1, the consideration of the rate of change of the tangent line to a curve $C$ led us to an important geometric entity, namely, the curvature of $C$. In this chapter we shall extend this idea to regular surfaces; that is, we shall try to measure how rapidly a surface $S$ pulls away from the tangent plane $T\_p(S)$ in a neighborhood of a point $p \in S$. This is equivalent to measuring the rate of change at $p$ of a unit normal vector field $N$ on a neighborhood of $p$. As we shall see shortly, this rate of change is given by a linear map on $T\_p(S)$ which happens to be self-adjoint (see the appendix to Chap. 3). A surprisingly large number of local properties of $S$ at $p$ can be derived from the study of this linear map.

In Sec. 3-2, we shall introduce the relevant definitions (the Gauss map, principal curvatures and principal directions, Gaussian and mean curvatures, etc.) without using local coordinates. In this way, the geometric content of the definitions is clearly brought up. However, for computational as well as for theoretical purposes, it is important to express all concepts in local coordinates. This is taken up in Sec. 3-3.

Section 3-4 contains a proof of the fact that at each point of a regular surface there exists an orthogonal parametrization, that is, a parametrization such that its coordinate curves meet orthogonally.

In Sec. 3-5 we shall take up two interesting special cases of surfaces, namely, the ruled surfaces and the minimal surfaces.

## 3-2. The Definition of the Gauss Map and Its Fundamental Properties

We shall begin by briefly reviewing the notion of orientation for surfaces. As we have seen in Sec. 2-4, given a parametrization $\mathbf{x}\colon U \subset R^2 \to S$ of a regular surface $S$ at a point $p \in S$, we can choose a unit normal vector at each point of $\mathbf{x}(U)$ by the rule

$$N(q) = \frac{\mathbf{x}_u \wedge \mathbf{x}_v}{|\mathbf{x}_u \wedge \mathbf{x}_v|}(q), \qquad q \in \mathbf{x}(U).$$

Thus, we have a differentiable map $N\colon \mathbf{x}(U) \to R^3$ that associates to each $q \in \mathbf{x}(U)$ a unit normal vector $N(q)$.

More generally, if $V \subset S$ is an open set in $S$ and $N\colon V \to R^3$ is a differentiable map which associates to each $q \in V$ a unit normal vector at $q$, we say that $N$ is a **differentiable field of unit normal vectors** on $V$. Not all surfaces admit a differentiable field of unit normal vectors defined on the *whole* surface. For instance, on the Möbius strip one cannot define such a field: After one turn, the vector field $N$ would come back as $-N$, a contradiction to the continuity of $N$.

We shall say that a regular surface is **orientable** if it admits a differentiable field of unit normal vectors defined on the whole surface; the choice of such a field $N$ is called an **orientation** of $S$.

Throughout this chapter, $S$ will denote a regular orientable surface in which an orientation $N$ has been chosen; this will be simply called a surface $S$ with an orientation $N$.

### The Gauss Map

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gauss Map)</span></p>

Let $S \subset R^3$ be a surface with an orientation $N$. The map $N\colon S \to S^2$, which takes its values in the unit sphere

$$S^2 = \lbrace (x, y, z) \in R^3;\; x^2 + y^2 + z^2 = 1 \rbrace,$$

is called the **Gauss map** of $S$.

</div>

It is straightforward to verify that the Gauss map is differentiable. The differential $dN\_p$ of $N$ at $p \in S$ is a linear map from $T\_p(S)$ to $T\_{N(p)}(S^2)$. Since $T\_p(S)$ and $T\_{N(p)}(S^2)$ are parallel planes, $dN\_p$ can be looked upon as a linear map on $T\_p(S)$.

The linear map $dN\_p\colon T\_p(S) \to T\_p(S)$ operates as follows. For each parametrized curve $\alpha(t)$ in $S$ with $\alpha(0) = p$, we consider the parametrized curve $N \circ \alpha(t) = N(t)$ in the sphere $S^2$; this amounts to restricting the normal vector $N$ to the curve $\alpha(t)$. The tangent vector $N'(0) = dN\_p(\alpha'(0))$ is a vector in $T\_p(S)$. It measures the rate of change of the normal vector $N$, restricted to the curve $\alpha(t)$, at $t = 0$. Thus, $dN\_p$ measures how $N$ pulls away from $N(p)$ in a neighborhood of $p$.

### Examples

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Gauss Map of a Plane)</span></p>

For a plane $P$ given by $ax + by + cz + d = 0$, the unit normal vector $N = (a, b, c)/\sqrt{a^2 + b^2 + c^2}$ is constant, and therefore $dN \equiv 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Gauss Map of the Unit Sphere)</span></p>

Consider the unit sphere $S^2 = \lbrace (x, y, z) \in R^3;\; x^2 + y^2 + z^2 = 1 \rbrace$. If $\alpha(t) = (x(t), y(t), z(t))$ is a parametrized curve in $S^2$, then $2xx' + 2yy' + 2zz' = 0$, which shows that the vector $(x, y, z)$ is normal to the sphere at the point $(x, y, z)$. Thus, $\bar{N} = (x, y, z)$ and $N = (-x, -y, -z)$ are fields of unit normal vectors in $S^2$. We fix an orientation in $S^2$ by choosing $N = (-x, -y, -z)$. Notice that $N$ points toward the center of the sphere.

Restricted to the curve $\alpha(t)$, the normal vector $N(t) = (-x(t), -y(t), -z(t))$ is a vector function of $t$, and therefore

$$dN(x'(t), y'(t), z'(t)) = N'(t) = (-x'(t), -y'(t), -z'(t));$$

that is, $dN\_p(v) = -v$ for all $p \in S^2$ and all $v \in T\_p(S^2)$. Notice that with the choice of the opposite orientation $\bar{N}$ as a normal field we would have obtained $d\bar{N}\_p(v) = v$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Gauss Map of the Cylinder)</span></p>

Consider the cylinder $\lbrace (x, y, z) \in R^3;\; x^2 + y^2 = 1 \rbrace$, and $N = (-x, -y, 0)$ are unit normal vectors at $(x, y, z)$. We fix an orientation by choosing $N = (-x, -y, 0)$ as the normal vector field.

By considering a curve $(x(t), y(t), z(t))$ contained in the cylinder, that is, with $(x(t))^2 + (y(t))^2 = 1$, we are able to see that, along this curve, $N(t) = (-x(t), -y(t), 0)$ and therefore

$$dN(x'(t), y'(t), z'(t)) = N'(t) = (-x'(t), -y'(t), 0).$$

We conclude the following: If $v$ is a vector tangent to the cylinder and parallel to the $z$ axis, then $dN(v) = 0 = 0v$; if $w$ is a vector tangent to the cylinder and parallel to the $xy$ plane, then $dN(w) = -w$. It follows that $v$ and $w$ are eigenvectors of $dN$ with eigenvalues $0$ and $-1$, respectively.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Gauss Map of the Hyperbolic Paraboloid)</span></p>

Let us analyze the point $p = (0, 0, 0)$ of the hyperbolic paraboloid $z = y^2 - x^2$. We consider a parametrization $\mathbf{x}(u, v) = (u, v, v^2 - u^2)$ and compute the normal vector $N(u, v)$. We obtain successively

$$\mathbf{x}_u = (1, 0, -2u), \qquad \mathbf{x}_v = (0, 1, 2v),$$

$$N = \frac{(2u, -2v, 1)}{\sqrt{4u^2 + 4v^2 + 1}}.$$

Notice that at $p = (0, 0, 0)$, $\mathbf{x}\_u$ and $\mathbf{x}\_v$ agree with the unit vectors along the $x$ and $y$ axes, respectively. Therefore, the tangent vector at $p$ to the curve $\alpha(t) = \mathbf{x}(u(t), v(t))$, with $\alpha(0) = p$, has, in $R^3$, coordinates $(u'(0), v'(0), 0)$. Restricting $N(u, v)$ to this curve and computing $N'(0)$, we obtain

$$N'(0) = (2u'(0), -2v'(0), 0).$$

It follows that the vectors $(1, 0, 0)$ and $(0, 1, 0)$ are eigenvectors of $dN\_p$ with eigenvalues $2$ and $-2$, respectively.

</div>

### Self-Adjointness of $dN\_p$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Self-Adjointness of $dN\_p$)</span></p>

The differential $dN\_p\colon T\_p(S) \to T\_p(S)$ of the Gauss map is a self-adjoint linear map (cf. the appendix to Chap. 3).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $dN\_p$ is linear, it suffices to verify that $\langle dN\_p(w\_1), w\_2 \rangle = \langle w\_1, dN\_p(w\_2) \rangle$ for a basis $\lbrace w\_1, w\_2 \rbrace$ of $T\_p(S)$. Let $\mathbf{x}(u, v)$ be a parametrization of $S$ at $p$ and $\lbrace \mathbf{x}\_u, \mathbf{x}\_v \rbrace$ the associated basis of $T\_p(S)$. If $\alpha(t) = \mathbf{x}(u(t), v(t))$ is a parametrized curve in $S$, with $\alpha(0) = p$, we have

$$dN_p(\alpha'(0)) = dN_p(\mathbf{x}_u u'(0) + \mathbf{x}_v v'(0)) = \frac{d}{dt}N(u(t), v(t))\bigg|_{t=0} = N_u u'(0) + N_v v'(0);$$

in particular, $dN\_p(\mathbf{x}\_u) = N\_u$ and $dN\_p(\mathbf{x}\_v) = N\_v$. Therefore, to prove that $dN\_p$ is self-adjoint, it suffices to show that

$$\langle N_u, \mathbf{x}_v \rangle = \langle \mathbf{x}_u, N_v \rangle.$$

To see this, take the derivatives of $\langle N, \mathbf{x}\_u \rangle = 0$ and $\langle N, \mathbf{x}\_v \rangle = 0$, relative to $v$ and $u$, respectively, and obtain

$$\langle N_v, \mathbf{x}_u \rangle + \langle N, \mathbf{x}_{uv} \rangle = 0, \qquad \langle N_u, \mathbf{x}_v \rangle + \langle N, \mathbf{x}_{vu} \rangle = 0.$$

Thus, $\langle N\_u, \mathbf{x}\_v \rangle = -\langle N, \mathbf{x}\_{uv} \rangle = \langle N\_v, \mathbf{x}\_u \rangle$. **Q.E.D.**

</details>
</div>

### The Second Fundamental Form

The fact that $dN\_p\colon T\_p(S) \to T\_p(S)$ is a self-adjoint linear map allows us to associate to $dN\_p$ a quadratic form $Q$ in $T\_p(S)$, given by $Q(v) = \langle dN\_p(v), v \rangle$, $v \in T\_p(S)$. For reasons that will be clear shortly, we shall use the quadratic form $-Q$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Second Fundamental Form)</span></p>

The quadratic form $II\_p$, defined in $T\_p(S)$ by $II\_p(v) = -\langle dN\_p(v), v \rangle$, is called the **second fundamental form** of $S$ at $p$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Normal Curvature)</span></p>

Let $C$ be a regular curve in $S$ passing through $p \in S$, and $\cos\theta = \langle n, N \rangle$, where $n$ is the normal vector to $C$ and $N$ is the normal vector to $S$ at $p$. The number $k\_n = k\cos\theta$ is then called the **normal curvature** of $C \subset S$ at $p$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Normal Curvature and Orientation)</span></p>

The normal curvature of $C$ does not depend on the orientation of $C$ but changes sign with a change of orientation for the surface.

</div>

To give an interpretation of the second fundamental form $II\_p$, consider a regular curve $C \subset S$ parametrized by $\alpha(s)$, where $s$ is the arc length of $C$, and with $\alpha(0) = p$. If we denote by $N(s)$ the restriction of the normal vector $N$ to the curve $\alpha(s)$, we have $\langle N(s), \alpha'(s) \rangle = 0$. Hence,

$$\langle N(s), \alpha''(s) \rangle = -\langle N'(s), \alpha'(s) \rangle.$$

Therefore,

$$II_p(\alpha'(0)) = -\langle dN_p(\alpha'(0)), \alpha'(0) \rangle = -\langle N'(0), \alpha'(0) \rangle = \langle N(0), \alpha''(0) \rangle = \langle N, kn \rangle(p) = k_n(p).$$

In other words, the value of the second fundamental form $II\_p$ for a unit vector $v \in T\_p(S)$ is equal to the normal curvature of a regular curve passing through $p$ and tangent to $v$. In particular, we obtained the following result.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Meusnier)</span></p>

All curves lying on a surface $S$ and having at a given point $p \in S$ the same tangent line have at this point the same normal curvatures.

</div>

The above proposition allows us to speak of the **normal curvature along a given direction** at $p$. Given a unit vector $v \in T\_p(S)$, the intersection of $S$ with the plane containing $v$ and $N(p)$ is called the **normal section** of $S$ at $p$ along $v$. In a neighborhood of $p$, a normal section is a regular plane curve on $S$ whose normal vector $n$ is $\perp N(p)$ or zero; its curvature is therefore equal to the absolute value of the normal curvature along $v$ at $p$.

### Principal Curvatures and Principal Directions

Since $dN\_p$ is a self-adjoint linear map, the theorem in the appendix to Chap. 3 shows that for each $p \in S$ there exists an orthonormal basis $\lbrace e\_1, e\_2 \rbrace$ of $T\_p(S)$ such that $dN\_p(e\_1) = -k\_1 e\_1$, $dN\_p(e\_2) = -k\_2 e\_2$. Moreover, $k\_1$ and $k\_2$ ($k\_1 \ge k\_2$) are the maximum and minimum of the second fundamental form $II\_p$ restricted to the unit circle of $T\_p(S)$; that is, they are the extreme values of the normal curvature at $p$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Principal Curvatures and Principal Directions)</span></p>

The maximum normal curvature $k\_1$ and the minimum normal curvature $k\_2$ are called the **principal curvatures** at $p$; the corresponding directions, that is, the directions given by the eigenvectors $e\_1$, $e\_2$, are called **principal directions** at $p$.

</div>

### The Euler Formula

The knowledge of the principal curvatures at $p$ allows us to compute easily the normal curvature along a given direction of $T\_p(S)$. In fact, let $v \in T\_p(S)$ with $\vert v\vert  = 1$. Since $e\_1$ and $e\_2$ form an orthonormal basis of $T\_p(S)$, we have

$$v = e_1\cos\theta + e_2\sin\theta,$$

where $\theta$ is the angle from $e\_1$ to $v$ in the orientation of $T\_p(S)$. The normal curvature $k\_n$ along $v$ is given by

$$k_n = II_p(v) = -\langle dN_p(e_1\cos\theta + e_2\sin\theta),\; e_1\cos\theta + e_2\sin\theta \rangle = k_1\cos^2\theta + k_2\sin^2\theta.$$

The last expression is known classically as the **Euler formula**; actually, it is just the expression of the second fundamental form in the basis $\lbrace e\_1, e\_2 \rbrace$.

### Lines of Curvature

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Line of Curvature)</span></p>

If a regular connected curve $C$ on $S$ is such that for all $p \in C$ the tangent line of $C$ is a principal direction at $p$, then $C$ is said to be a **line of curvature** of $S$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Olinde Rodrigues)</span></p>

A necessary and sufficient condition for a connected regular curve $C$ on $S$ to be a line of curvature of $S$ is that

$$N'(t) = \lambda(t)\,\alpha'(t),$$

for any parametrization $\alpha(t)$ of $C$, where $N(t) = N \circ \alpha(t)$ and $\lambda(t)$ is a differentiable function of $t$. In this case, $-\lambda(t)$ is the (principal) curvature along $\alpha'(t)$.

</div>

### Gaussian Curvature and Mean Curvature

Given a linear map $A\colon V \to V$ of a vector space of dimension 2 and given a basis $\lbrace v\_1, v\_2 \rbrace$ of $V$, recall that

$$\det A = a_{11}a_{22} - a_{12}a_{21}, \qquad \text{trace of } A = a_{11} + a_{22},$$

where $(a\_{ij})$ is the matrix of $A$ in the basis $\lbrace v\_1, v\_2 \rbrace$. These numbers do not depend on the choice of the basis and are, therefore, attached to the linear map $A$.

In our case, the determinant of $dN$ is the product $(-k\_1)(-k\_2) = k\_1 k\_2$ of the principal curvatures, and the trace of $dN$ is the negative $-(k\_1 + k\_2)$ of the sum of principal curvatures. If we change the orientation of the surface, the determinant does not change (the fact that the dimension is even is essential here); the trace, however, changes sign.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gaussian Curvature and Mean Curvature)</span></p>

Let $p \in S$ and let $dN\_p\colon T\_p(S) \to T\_p(S)$ be the differential of the Gauss map. The determinant of $dN\_p$ is the **Gaussian curvature** $K$ of $S$ at $p$. The negative of half of the trace of $dN\_p$ is called the **mean curvature** $H$ of $S$ at $p$. In terms of the principal curvatures we can write

$$K = k_1 k_2, \qquad H = \frac{k_1 + k_2}{2}.$$

</div>

### Classification of Points

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Classification of Surface Points)</span></p>

A point of a surface $S$ is called

1. **Elliptic** if $\det(dN\_p) > 0$ (equivalently, $K > 0$).
2. **Hyperbolic** if $\det(dN\_p) < 0$ (equivalently, $K < 0$).
3. **Parabolic** if $\det(dN\_p) = 0$, with $dN\_p \neq 0$ (equivalently, $K = 0$ but not both $k\_1, k\_2$ are zero).
4. **Planar** if $dN\_p = 0$ (equivalently, $k\_1 = k\_2 = 0$).

</div>

This classification does not depend on the choice of the orientation. At an **elliptic** point, the Gaussian curvature is positive, both principal curvatures have the same sign, and all curves through this point have their normal vectors pointing toward the same side of the tangent plane. The points of a sphere are elliptic. At a **hyperbolic** point, the Gaussian curvature is negative, the principal curvatures have opposite signs, and there are curves through $p$ whose normal vectors at $p$ point toward any of the sides of the tangent plane. At a **parabolic** point, one of the principal curvatures is not zero. The points of a cylinder are parabolic. At a **planar** point, all principal curvatures are zero. The points of a plane trivially satisfy this condition. A nontrivial example of a planar point was given in Example 6 (the surface of revolution of $z = y^4$ around the $z$ axis, at $p = (0, 0, 0)$).

### Umbilical Points

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Umbilical Point)</span></p>

If at $p \in S$, $k\_1 = k\_2$, then $p$ is called an **umbilical point** of $S$; in particular, the planar points ($k\_1 = k\_2 = 0$) are umbilical points.

</div>

All the points of a sphere and a plane are umbilical points. Using the method of Example 6, we can verify that the point $(0, 0, 0)$ of the paraboloid $z = x^2 + y^2$ is a (nonplanar) umbilical point.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4</span><span class="math-callout__name">(Umbilical Surfaces are Spheres or Planes)</span></p>

If all points of a connected surface $S$ are umbilical points, then $S$ is either contained in a sphere or in a plane.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $p \in S$ and let $\mathbf{x}(u, v)$ be a parametrization of $S$ at $p$ such that the coordinate neighborhood $V$ is connected. Since each $q \in V$ is an umbilical point, we have, for any vector $w = a\_1 \mathbf{x}\_u + a\_2 \mathbf{x}\_v$ in $T\_q(S)$,

$$dN(w) = \lambda(q)\,w,$$

where $\lambda = \lambda(q)$ is a real differentiable function in $V$. We first show that $\lambda(q)$ is constant in $V$. For that, we write the above equation as $N\_u a\_1 + N\_v a\_2 = \lambda(\mathbf{x}\_u a\_1 + \mathbf{x}\_v a\_2)$; hence, since $w$ is arbitrary, $N\_u = \lambda \mathbf{x}\_u$ and $N\_v = \lambda \mathbf{x}\_v$. Differentiating the first equation in $u$ and the second one in $v$ and subtracting the resulting equations, we obtain $\lambda\_u \mathbf{x}\_v - \lambda\_v \mathbf{x}\_u = 0$. Since $\mathbf{x}\_u$ and $\mathbf{x}\_v$ are linearly independent, we conclude that $\lambda\_u = \lambda\_v = 0$ for all $q \in V$. Since $V$ is connected, $\lambda$ is constant in $V$, as we claimed.

If $\lambda \equiv 0$, $N\_u = N\_v = 0$ and therefore $N = N\_0 = \text{constant}$ in $V$. Thus, $\langle \mathbf{x}(u, v), N\_0 \rangle\_u = \langle \mathbf{x}(u, v), N\_0 \rangle\_v = 0$; hence, $\langle \mathbf{x}(u, v), N\_0 \rangle = \text{const}$., and all points $\mathbf{x}(u, v)$ of $V$ belong to a plane.

If $\lambda \neq 0$, then the point $\mathbf{y}(u, v) = \mathbf{x}(u, v) - (1/\lambda)N(u, v)$ is fixed, because $(\mathbf{x}(u, v) - (1/\lambda)N(u, v))\_u = (\mathbf{x}(u, v) - (1/\lambda)N(u, v))\_v = 0$. Since $\|\mathbf{x}(u, v) - \mathbf{y}\|^2 = 1/\lambda^2$, all points of $V$ are contained in a sphere of center $\mathbf{y}$ and radius $1/\vert \lambda\vert $.

This proves the proposition locally, that is, for a neighborhood of a point $p \in S$. To complete the proof we observe that, since $S$ is connected, given any other point $r \in S$, there exists a continuous curve $\alpha\colon [0, 1] \to S$ with $\alpha(0) = p$, $\alpha(1) = r$. For each point $\alpha(t)$ of this curve there exists a neighborhood $V\_t$ in $S$ contained in a sphere or in a plane. The union $\bigcup \alpha^{-1}(V\_t)$, $t \in [0, 1]$, covers $[0, 1]$ and since $[0, 1]$ is a closed interval, it is covered by finitely many of the neighborhoods $V\_t$. By renumbering the intervals, if necessary, that two consecutive intervals overlap. Thus, $f \circ \alpha$ is constant in the union of two consecutive intervals. It follows that $f$ is constant on $[0, 1]$; that is, $f(\alpha(0)) = f(p) = f(\alpha(1)) = f(r)$. Since $r$ is arbitrary, $f$ is constant on $U$. **Q.E.D.**

</details>
</div>

### Asymptotic Directions and the Dupin Indicatrix

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Asymptotic Direction and Asymptotic Curve)</span></p>

Let $p$ be a point in $S$. An **asymptotic direction** of $S$ at $p$ is a direction of $T\_p(S)$ for which the normal curvature is zero. An **asymptotic curve** of $S$ is a regular connected curve $C \subset S$ such that for each $p \in C$ the tangent line of $C$ at $p$ is an asymptotic direction.

</div>

It follows at once from the definition that at an elliptic point there are no asymptotic directions. At a hyperbolic point there are exactly two asymptotic directions (since the Dupin indicatrix is a pair of conjugate hyperbolas whose asymptotes give the asymptotic directions). At a parabolic point, there is exactly one asymptotic direction.

The **Dupin indicatrix** at $p$ is the set of vectors $w$ of $T\_p(S)$ such that $II\_p(w) = \pm 1$. To write the equations of the Dupin indicatrix in a more convenient form, let $(\xi, \eta)$ be the Cartesian coordinates of $T\_p(S)$ in the orthonormal basis $\lbrace e\_1, e\_2 \rbrace$, where $e\_1$ and $e\_2$ are eigenvectors of $dN\_p$. Given $w \in T\_p(S)$, let $\rho$ and $\theta$ be "polar coordinates" defined by $w = \rho v$, with $\vert v\vert  = 1$ and $v = e\_1\cos\theta + e\_2\sin\theta$, if $\rho \neq 0$. By Euler's formula,

$$\pm 1 = II_p(w) = \rho^2 II_p(v) = k_1 \rho^2 \cos^2\theta + k_2 \rho^2 \sin^2\theta,$$

where $w = \xi e\_1 + \eta e\_2$. Thus, the coordinates $(\xi, \eta)$ of a point of the Dupin indicatrix satisfy the equation

$$k_1 \xi^2 + k_2 \eta^2 = \pm 1.$$

For an **elliptic** point, the Dupin indicatrix is an ellipse ($k\_1$ and $k\_2$ have the same sign); this ellipse degenerates into a circle if the point is an umbilical nonplanar point ($k\_1 = k\_2 \neq 0$). For a **hyperbolic** point, $k\_1$ and $k\_2$ have opposite signs. The Dupin indicatrix is therefore made up of two conjugate hyperbolas with a common pair of asymptotic lines. Along the directions of the asymptotes, the normal curvature is zero; they are therefore asymptotic directions. This justifies the terminology and shows that a hyperbolic point has *exactly two* asymptotic directions. For a **parabolic** point, one of the principal curvatures is zero, and the Dupin indicatrix degenerates into a pair of parallel lines. The common direction of these lines is the only asymptotic direction at the given point.

### Conjugate Directions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Conjugate Directions)</span></p>

Let $p$ be a point on a surface $S$. Two nonzero vectors $w\_1, w\_2 \in T\_p(S)$ are **conjugate** if $\langle dN\_p(w\_1), w\_2 \rangle = \langle w\_1, dN\_p(w\_2) \rangle = 0$. Two directions $r\_1$, $r\_2$ at $p$ are **conjugate** if a pair of nonzero vectors $w\_1$, $w\_2$ parallel to $r\_1$ and $r\_2$, respectively, are conjugate.

</div>

It is immediate that the definition of conjugate directions does not depend on the choice of the vectors $w\_1$ and $w\_2$ on $r\_1$ and $r\_2$. It follows from the definition that the principal directions are conjugate and that an asymptotic direction is conjugate to itself. Furthermore, at a nonplanar umbilic, every orthogonal pair of directions is a pair of conjugate directions, and at a planar umbilic each direction is conjugate to any other direction.

Assuming that $p \in S$ is not an umbilical point, let $\lbrace e\_1, e\_2 \rbrace$ be the orthonormal basis of $T\_p(S)$ determined by $dN\_p(e\_1) = -k\_1 e\_1$, $dN\_p(e\_2) = -k\_2 e\_2$. Let $\theta$ and $\varphi$ be the angles that a pair of directions $r\_1$ and $r\_2$ make with $e\_1$. Then $r\_1$ and $r\_2$ are conjugate if and only if

$$k_1\cos\theta\cos\varphi + k_2\sin\theta\sin\varphi = 0.$$

## 3-3. The Gauss Map in Local Coordinates

In the preceding section, we introduced some concepts related to the local behavior of the Gauss map. To emphasize the geometry of the situation, the definitions were given without the use of a coordinate system. Some simple examples were then computed directly from the definitions; this procedure, however, is inefficient in handling general situations. In this section, we shall obtain the expressions of the second fundamental form and of the differential of the Gauss map in a coordinate system. This will give us a systematic method for computing specific examples.

All parametrizations $\mathbf{x}\colon U \subset R^2 \to S$ considered in this section are assumed to be compatible with the orientation $N$ of $S$; that is, in $\mathbf{x}(U)$,

$$N = \frac{\mathbf{x}_u \wedge \mathbf{x}_v}{|\mathbf{x}_u \wedge \mathbf{x}_v|}.$$

### The Coefficients $e$, $f$, $g$ and the Weingarten Equations

Let $\mathbf{x}(u, v)$ be a parametrization at a point $p \in S$ of a surface $S$, and let $\alpha(t) = \mathbf{x}(u(t), v(t))$ be a parametrized curve on $S$, with $\alpha(0) = p$. To simplify the notation, all functions to appear below denote their values at the point $p$. The tangent vector to $\alpha(t)$ at $p$ is $\alpha' = \mathbf{x}\_u u' + \mathbf{x}\_v v'$ and

$$dN(\alpha') = N'(u(t), v(t)) = N_u u' + N_v v'.$$

Since $N\_u$ and $N\_v$ belong to $T\_p(S)$, we may write

$$N_u = a_{11}\mathbf{x}_u + a_{21}\mathbf{x}_v, \qquad N_v = a_{12}\mathbf{x}_u + a_{22}\mathbf{x}_v,$$

and therefore,

$$dN\begin{pmatrix} u' \\ v' \end{pmatrix} = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}\begin{pmatrix} u' \\ v' \end{pmatrix}.$$

This shows that in the basis $\lbrace \mathbf{x}\_u, \mathbf{x}\_v \rbrace$, $dN$ is given by the matrix $(a\_{ij})$, $i, j = 1, 2$. Notice that this matrix is not necessarily symmetric, unless $\lbrace \mathbf{x}\_u, \mathbf{x}\_v \rbrace$ is an orthonormal basis.

On the other hand, the expression of the second fundamental form in the basis $\lbrace \mathbf{x}\_u, \mathbf{x}\_v \rbrace$ is given by

$$II_p(\alpha') = -\langle dN(\alpha'), \alpha' \rangle = e(u')^2 + 2fu'v' + g(v')^2,$$

where, since $\langle N, \mathbf{x}\_u \rangle = \langle N, \mathbf{x}\_v \rangle = 0$,

$$e = -\langle N_u, \mathbf{x}_u \rangle = \langle N, \mathbf{x}_{uu} \rangle, \qquad f = -\langle N_v, \mathbf{x}_u \rangle = \langle N, \mathbf{x}_{uv} \rangle, \qquad g = -\langle N_v, \mathbf{x}_v \rangle = \langle N, \mathbf{x}_{vv} \rangle.$$

The relations between the coefficients $e$, $f$, $g$ and the matrix entries $a\_{ij}$ can be expressed in matrix form as

$$-\begin{pmatrix} e & f \\ f & g \end{pmatrix} = \begin{pmatrix} a_{11} & a_{21} \\ a_{12} & a_{22} \end{pmatrix}\begin{pmatrix} E & F \\ F & G \end{pmatrix},$$

where $E$, $F$, and $G$ are the coefficients of the first fundamental form in the basis $\lbrace \mathbf{x}\_u, \mathbf{x}\_v \rbrace$ (cf. Sec. 2-5). Hence,

$$\begin{pmatrix} a_{11} & a_{21} \\ a_{12} & a_{22} \end{pmatrix} = -\begin{pmatrix} e & f \\ f & g \end{pmatrix}\begin{pmatrix} E & F \\ F & G \end{pmatrix}^{-1},$$

and since

$$\begin{pmatrix} E & F \\ F & G \end{pmatrix}^{-1} = \frac{1}{EG - F^2}\begin{pmatrix} G & -F \\ -F & E \end{pmatrix},$$

we obtain the following expressions for the coefficients $(a\_{ij})$ of the matrix of $dN$ in the basis $\lbrace \mathbf{x}\_u, \mathbf{x}\_v \rbrace$:

$$a_{11} = \frac{fF - eG}{EG - F^2}, \qquad a_{12} = \frac{gF - fG}{EG - F^2},$$

$$a_{21} = \frac{eF - fE}{EG - F^2}, \qquad a_{22} = \frac{fF - gE}{EG - F^2}.$$

These relations, together with Eqs. for $N\_u$ and $N\_v$, are known as the **equations of Weingarten**.

### Gaussian and Mean Curvatures in Local Coordinates

From the expressions above we immediately obtain

$$K = \det(a_{ij}) = \frac{eg - f^2}{EG - F^2},$$

$$H = -\frac{1}{2}(a_{11} + a_{22}) = \frac{1}{2}\frac{eG - 2fF + gE}{EG - F^2}.$$

To compute the mean curvature, we recall that $-k\_1$, $-k\_2$ are the eigenvalues of $dN$. Therefore, $k\_1$ and $k\_2$ satisfy the equation

$$k^2 + k(a_{11} + a_{22}) + a_{11}a_{22} - a_{21}a_{12} = 0,$$

that is,

$$k^2 - 2Hk + K = 0,$$

and therefore,

$$k = H \pm \sqrt{H^2 - K}.$$

From this relation, it follows that if we choose $k\_1(q) \ge k\_2(q)$, $q \in S$, then the functions $k\_1$ and $k\_2$ are continuous in $S$. Moreover, $k\_1$ and $k\_2$ are differentiable in $S$, except perhaps at the umbilical points ($H^2 = K$) of $S$.

In the computations of this chapter, it will be convenient to write for short

$$\langle u \wedge v, w \rangle = (u, v, w) \qquad \text{for any } u, v, w \in R^3.$$

We recall that this is merely the determinant of the $3 \times 3$ matrix whose columns (or lines) are the components of the vectors $u$, $v$, $w$ in the canonical basis of $R^3$.

### Examples in Local Coordinates

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gaussian Curvature of the Torus)</span></p>

We shall compute the Gaussian curvature of the points of the torus covered by the parametrization (cf. Example 6 of Sec. 2-2)

$$\mathbf{x}(u, v) = ((a + r\cos u)\cos v,\; (a + r\cos u)\sin v,\; r\sin u), \qquad 0 < u < 2\pi,\; 0 < v < 2\pi.$$

For the computation of the coefficients $e$, $f$, $g$, we need to know $N$ (and thus $\mathbf{x}\_u$ and $\mathbf{x}\_v$), $\mathbf{x}\_{uu}$, $\mathbf{x}\_{uv}$, and $\mathbf{x}\_{vv}$:

$$\mathbf{x}_u = (-r\sin u\cos v,\; -r\sin u\sin v,\; r\cos u), \qquad \mathbf{x}_v = (-(a + r\cos u)\sin v,\; (a + r\cos u)\cos v,\; 0).$$

From these, $E = r^2$, $F = 0$, $G = (a + r\cos u)^2$, and since $\|\mathbf{x}\_u \wedge \mathbf{x}\_v\| = \sqrt{EG - F^2}$,

$$e = \left\langle \frac{\mathbf{x}_u \wedge \mathbf{x}_v}{|\mathbf{x}_u \wedge \mathbf{x}_v|},\; \mathbf{x}_{uu} \right\rangle = \frac{(\mathbf{x}_u, \mathbf{x}_v, \mathbf{x}_{uu})}{\sqrt{EG - F^2}} = r,$$

$$f = \frac{(\mathbf{x}_u, \mathbf{x}_v, \mathbf{x}_{uv})}{\sqrt{EG - F^2}} = 0, \qquad g = \frac{(\mathbf{x}_u, \mathbf{x}_v, \mathbf{x}_{vv})}{\sqrt{EG - F^2}} = \cos u(a + r\cos u).$$

Finally, since $K = (eg - f^2)/(EG - F^2)$, we have that

$$K = \frac{\cos u}{r(a + r\cos u)}.$$

From this expression, it follows that $K = 0$ along the parallels $u = \pi/2$ and $u = 3\pi/2$; the points of such parallels are therefore parabolic points. In the region of the torus given by $\pi/2 < u < 3\pi/2$, $K$ is negative (notice that $r > 0$ and $a > r$); the points in this region are therefore hyperbolic points. In the region given by $0 < u < \pi/2$ or $3\pi/2 < u < 2\pi$, the curvature is positive and the points are elliptic points.

</div>

### Position of a Surface Near Elliptic and Hyperbolic Points

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Position Relative to the Tangent Plane)</span></p>

Let $p \in S$ be an elliptic point of a surface $S$. Then there exists a neighborhood $V$ of $p$ in $S$ such that all points in $V$ belong to the same side of the tangent plane $T\_p(S)$. Let $p \in S$ be a hyperbolic point. Then in each neighborhood of $p$ there exist points of $S$ in both sides of $T\_p(S)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $\mathbf{x}(u, v)$ be a parametrization in $p$, with $\mathbf{x}(0, 0) = p$. The distance $d$ from a point $q = \mathbf{x}(u, v)$ to the tangent plane $T\_p(S)$ is given by

$$d = \langle \mathbf{x}(u, v) - \mathbf{x}(0, 0),\; N(p) \rangle.$$

Since $\mathbf{x}(u, v)$ is differentiable, we have Taylor's formula:

$$\mathbf{x}(u, v) = \mathbf{x}(0, 0) + \mathbf{x}_u u + \mathbf{x}_v v + \tfrac{1}{2}(\mathbf{x}_{uu}u^2 + 2\mathbf{x}_{uv}uv + \mathbf{x}_{vv}v^2) + \bar{R},$$

where the derivatives are taken at $(0, 0)$ and the remainder $\bar{R}$ satisfies $\lim \bar{R}/(u^2 + v^2) = 0$. It follows that

$$d = \tfrac{1}{2}(eu^2 + 2fuv + gv^2) + R = \tfrac{1}{2}\,II_p(w) + R,$$

where $w = \mathbf{x}\_u u + \mathbf{x}\_v v$, $R = \langle \bar{R}, N(p) \rangle$, and $\lim R/\vert w\vert ^2 = 0$.

For an elliptic point $p$, $II\_p(w)$ has a fixed sign. Therefore, for all sufficiently small $(u, v)$, $d$ has the same sign as $II\_p(w)$; that is, all such $(u, v)$ belong to the same side of $T\_p(S)$.

For a hyperbolic point $p$, in each neighborhood of $p$ there exist points $(u, v)$ and $(\tilde{u}, \tilde{v})$ such that $II\_p(w/\|w\|)$ and $II\_p(\tilde{w}/\|\tilde{w}\|)$ have opposite signs (here $\tilde{w} = \mathbf{x}\_u \tilde{u} + \mathbf{x}\_v \tilde{v}$); such points belong therefore to distinct sides of $T\_p(S)$. **Q.E.D.**

</details>
</div>

No such statement as Prop. 1 can be made in a neighborhood of a parabolic or a planar point. In the above examples of parabolic and planar points (cf. Examples 3 and 6 of Sec. 3-2) the surface lies on one side of the tangent plane and may have a line in common with this plane.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Monkey Saddle)</span></p>

The "monkey saddle" is given by $x = u$, $y = v$, $z = u^3 - 3v^2 u$. A direct computation shows that at $(0, 0)$ the coefficients of the second fundamental form are $e = f = g = 0$; the point $(0, 0)$ is therefore a planar point. In any neighborhood of this point, however, there are points in both sides of its tangent plane.

</div>

### Asymptotic and Principal Directions in Local Coordinates

The expression of the second fundamental form in local coordinates is particularly useful for the study of the asymptotic and principal directions.

Let $\mathbf{x}(u, v)$ be a parametrization at $p \in S$, with $\mathbf{x}(0, 0) = p$, and let $e(u, v) = e$, $f(u, v) = f$, and $g(u, v) = g$ be the coefficients of the second fundamental form in this parametrization. A connected regular curve $C$ in the coordinate neighborhood of $\mathbf{x}$ is an asymptotic curve if and only if for any parametrization $\alpha(t) = \mathbf{x}(u(t), v(t))$, $t \in I$, of $C$ we have $II(\alpha'(t)) = 0$, for all $t \in I$, that is, if and only if

$$e(u')^2 + 2fu'v' + g(v')^2 = 0, \qquad t \in I.$$

This is called the **differential equation of the asymptotic curves**. A necessary and sufficient condition for a parametrization in a neighborhood of a hyperbolic point (e.g. $-f^2 < 0$) to be such that the coordinate curves of the parametrization are asymptotic curves is that $e = g = 0$.

Similarly, a connected regular curve $C$ in the coordinate neighborhood of $\mathbf{x}$ is a line of curvature if and only if for any parametrization $\alpha(t) = \mathbf{x}(u(t), v(t))$, $t \in I$, of $C$, $t \in I$, we have (cf. Prop. 3 of Sec. 3-2):

$$dN(\alpha'(t)) = \lambda(t)\alpha'(t).$$

By eliminating $\lambda$ in the above system, we obtain the **differential equation of the lines of curvature**,

$$(fE - eF)(u')^2 + (gE - eG)u'v' + (gF - fG)(v')^2 = 0,$$

which may be written, in a more symmetric way, as

$$\begin{vmatrix} (v')^2 & -u'v' & (u')^2 \\ E & F & G \\ e & f & g \end{vmatrix} = 0.$$

Using the fact that the principal directions are orthogonal to each other, it follows easily from this equation that *a necessary and sufficient condition for the coordinate curves of a parametrization to be lines of curvature in a neighborhood of a nonumbilical point is that* $F = f = 0$.

### Surfaces of Revolution in Local Coordinates

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Surfaces of Revolution — Curvatures)</span></p>

Consider a surface of revolution parametrized by (cf. Example 4 of Sec. 2-3; we have changed $f$ and $g$ by $\varphi$ and $\psi$, respectively)

$$\mathbf{x}(u, v) = (\varphi(v)\cos u,\; \varphi(v)\sin u,\; \psi(v)), \qquad 0 < u < 2\pi,\; a < v < b,\; \varphi(v) \neq 0.$$

The coefficients of the first fundamental form are given by $E = \varphi^2$, $F = 0$, $G = (\varphi')^2 + (\psi')^2$. It is convenient to assume that the rotating curve is parametrized by arc length, that is, $(\varphi')^2 + (\psi')^2 = G = 1$.

The computation of the coefficients of the second fundamental form is straightforward and yields

$$e = -\varphi\psi', \qquad f = 0, \qquad g = \psi'\varphi'' - \psi''\varphi'.$$

Since $F = f = 0$, we conclude that the parallels ($v = \text{const.}$) and the meridians ($u = \text{const.}$) of a surface of revolution are lines of curvature (this fact was used in Example 3 of Sec. 3-2).

Because $K = (eg - f^2)/(EG - F^2)$ and $\varphi$ is always positive, the parabolic points are given by either $\psi' = 0$ (the tangent line to the generator curve is perpendicular to the axis of rotation) or $\varphi'\psi'' - \psi'\varphi'' = 0$ (the curvature of the generator curve is zero). A point which satisfies both conditions is a planar point, since these conditions imply that $e = f = g = 0$.

It is convenient to put the Gaussian curvature in still another form. By differentiating $(\varphi')^2 + (\psi')^2 = 1$ we obtain $\varphi'\varphi'' = -\psi'\psi''$. Thus,

$$K = -\frac{\psi'(\psi'\varphi'' - \psi''\varphi')}{\varphi} = -\frac{(\psi')^2\varphi'' + (\varphi')^2\varphi''}{\varphi} = -\frac{\varphi''}{\varphi}.$$

If a parametrization of a regular surface is such that $F = f = 0$, then the principal curvatures are given by $e/E$ and $g/G$. Thus, the principal curvatures of a surface of revolution are given by

$$\frac{e}{E} = -\frac{\psi'\varphi}{\varphi^2} = -\frac{\psi'}{\varphi}, \qquad \frac{g}{G} = \psi'\varphi'' - \psi''\varphi';$$

hence, the mean curvature of such a surface is

$$H = \frac{1}{2}\frac{-\psi' + \varphi(\psi'\varphi'' - \psi''\varphi')}{\varphi}.$$

</div>

### Formulas for Graphs $z = h(x, y)$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Curvatures for Graphs)</span></p>

Very often a surface is given as the graph of a differentiable function $z = h(x, y)$, where $(x, y)$ belong to an open set $U \subset R^2$. It is therefore convenient to have at hand formulas for the relevant concepts in this case. Using the parametrization $\mathbf{x}(u, v) = (u, v, h(u, v))$, with $u = x$, $v = y$, we obtain

$$\mathbf{x}_u = (1, 0, h_u), \quad \mathbf{x}_v = (0, 1, h_v), \quad \mathbf{x}_{uu} = (0, 0, h_{uu}), \quad \mathbf{x}_{uv} = (0, 0, h_{uv}), \quad \mathbf{x}_{vv} = (0, 0, h_{vv}).$$

Thus,

$$N(x, y) = \frac{(-h_x, -h_y, 1)}{(1 + h_x^2 + h_y^2)^{1/2}}$$

is a unit normal field on the surface, and the coefficients of the second fundamental form in this orientation are given by

$$e = \frac{h_{xx}}{(1 + h_x^2 + h_y^2)^{1/2}}, \qquad f = \frac{h_{xy}}{(1 + h_x^2 + h_y^2)^{1/2}}, \qquad g = \frac{h_{yy}}{(1 + h_x^2 + h_y^2)^{1/2}}.$$

From the above expressions, any needed formula can be easily computed. For instance, the Gaussian and mean curvatures are:

$$K = \frac{h_{xx}h_{yy} - h_{xy}^2}{(1 + h_x^2 + h_y^2)^2},$$

$$H = \frac{(1 + h_x^2)h_{yy} - 2h_x h_y h_{xy} + (1 + h_y^2)h_{xx}}{(1 + h_x^2 + h_y^2)^{3/2}}.$$

There is another important reason to study surfaces given by $z = h(x, y)$. It comes from the fact that locally any surface is the graph of a differentiable function (cf. Prop. 3, Sec. 2-2). Given a point $p$ of a surface $S$, we can choose the coordinate axes of $R^3$ so that the origin $O$ is at $p$ and the $z$ axis is directed along the positive normal of $S$ at $p$ (thus, the $xy$ plane agrees with $T\_p(S)$). It follows that a neighborhood of $p$ in $S$ can be represented in the form $z = h(x, y)$, $(x, y) \in U \subset R^2$, with $h(0, 0) = p$, $h\_x(0, 0) = 0$, $h\_y(0, 0) = 0$.

The second fundamental form of $S$ at $p$ applied to the vector $(x, y) \in R^2$ becomes

$$h_{xx}(0, 0)x^2 + 2h_{xy}(0, 0)xy + h_{yy}(0, 0)y^2.$$

In elementary calculus of two variables, this quadratic form is known as the **Hessian** of $h$ at $(0, 0)$. Thus, the Hessian of $h$ at $(0, 0)$ is the second fundamental form of $S$ at $p$.

</div>

### Geometric Interpretation of the Gaussian Curvature

The Gauss map $N$ preserves or reverses orientation depending on the sign of $K$. More precisely, an orientation $N$ on $S$ induces an orientation on $S^2$. Let $p \in S$ be such that $dN\_p$ is nonsingular. Since for a basis $\lbrace w\_1, w\_2 \rbrace$ in $T\_p(S)$

$$dN_p(w_1) \wedge dN_p(w_2) = \det(dN_p)(w_1 \wedge w_2) = K\,w_1 \wedge w_2,$$

the Gauss map $N$ will be orientation-preserving at $p \in S$ if $K(p) > 0$ and orientation-reversing at $p \in S$ if $K(p) < 0$. Intuitively, this means the following: An orientation of $T\_p(S)$ induces an orientation of small closed curves in $S$ around $p$; the image by $N$ of these curves will have the same or the opposite orientation to the initial one, depending on whether $p$ is an elliptic or a hyperbolic point, respectively.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Gaussian Curvature as Area Ratio)</span></p>

Let $p$ be a point of a surface $S$ such that the Gaussian curvature $K(p) \neq 0$, and let $V$ be a connected neighborhood of $p$ where $K$ does not change sign. Then

$$K(p) = \lim_{A \to 0} \frac{A'}{A},$$

where $A$ is the area of a region $B \subset V$ containing $p$, $A'$ is the area of the image of $B$ by the Gauss map $N\colon S \to S^2$, and the limit is taken through a sequence of regions $B\_n$ that converges to $p$, in the sense that any sphere around $p$ contains all $B\_n$, for $n$ sufficiently large.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The area $A$ of $B$ is given by (cf. Sec. 2-5)

$$A = \iint_R |\mathbf{x}_u \wedge \mathbf{x}_v|\,du\,dv,$$

where $\mathbf{x}(u, v)$ is a parametrization in $p$, whose coordinate neighborhood contains $V$ (we can assume $V$ to be sufficiently small) and $R$ is the region in the $uv$ plane corresponding to $B$. The area $A'$ of $N(B)$ is

$$A' = \iint_R |N_u \wedge N_v|\,du\,dv.$$

Using the expression $N\_u = a\_{11}\mathbf{x}\_u + a\_{21}\mathbf{x}\_v$, $N\_v = a\_{12}\mathbf{x}\_u + a\_{22}\mathbf{x}\_v$, the definition of $K$, and the above convention, we can write

$$A' = \iint_R K\,|\mathbf{x}_u \wedge \mathbf{x}_v|\,du\,dv.$$

Going to the limit and denoting also by $R$ the area of the region $R$, we obtain

$$\lim_{A \to 0} \frac{A'}{A} = \lim_{R \to 0} \frac{(1/R)\iint_R K\,|\mathbf{x}_u \wedge \mathbf{x}_v|\,du\,dv}{(1/R)\iint_R |\mathbf{x}_u \wedge \mathbf{x}_v|\,du\,dv} = \frac{K\,|\mathbf{x}_u \wedge \mathbf{x}_v|}{|\mathbf{x}_u \wedge \mathbf{x}_v|} = K$$

(notice that we have used the mean value theorem for double integrals), and this proves the proposition. **Q.E.D.**

</details>
</div>

Comparing the proposition with the expression of the curvature $k$ of a plane curve $C$ at $p$, namely $k = \lim \sigma / s$ (here $s$ is the arc length of a small segment of $C$ containing $p$, and $\sigma$ is the arc length of its image in the indicatrix of tangents; cf. Exercise 3 of Sec. 1-5), we see that the Gaussian curvature $K$ is the analogue, for surfaces, of the curvature $k$ of plane curves.

## 3-4. Vector Fields

In this section we shall use the fundamental theorems of ordinary differential equations (existence, uniqueness, and dependence on the initial conditions) to prove the existence of certain coordinate systems on surfaces.

### Vector Fields in $R^2$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vector Field in $R^2$)</span></p>

A **vector field** in an open set $U \subset R^2$ is a map which assigns to each $q \in U$ a vector $w(q) \in R^2$. The vector field $w$ is said to be **differentiable** if writing $q = (x, y)$ and $w(q) = (a(x, y), b(x, y))$, the functions $a$ and $b$ are differentiable functions in $U$.

</div>

Given a vector field $w$, it is natural to ask whether there exists a **trajectory** of this field, that is, whether there exists a differentiable parametrized curve $\alpha(t) = (x(t), y(t))$, $t \in I$, such that $\alpha'(t) = w(\alpha(t))$.

In the language of ordinary differential equations, one says that the vector field $w$ determines a system of **differential equations**,

$$\frac{dx}{dt} = a(x, y), \qquad \frac{dy}{dt} = b(x, y),$$

and that a trajectory of $w$ is a **solution** of this system.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Existence and Uniqueness of Trajectories)</span></p>

Let $w$ be a vector field in an open set $U \subset R^2$. Given $p \in U$, there exists a trajectory $\alpha\colon I \to U$ of $w$ (i.e., $\alpha'(t) = w(\alpha(t))$, $t \in I$) with $\alpha(0) = p$. This trajectory is unique in the following sense: Any other trajectory $\beta\colon J \to U$ with $\beta(0) = p$ agrees with $\alpha$ in $I \cap J$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2</span><span class="math-callout__name">(Smooth Dependence on Initial Conditions — Local Flow)</span></p>

Let $w$ be a vector field in an open set $U \subset R^2$. For each $p \in U$ there exist a neighborhood $V \subset U$ of $p$, an interval $I$, and a mapping $\alpha\colon V \times I \to U$ such that

1. For a fixed $q \in V$, the curve $\alpha(q, t)$, $t \in I$, is the trajectory of $w$ passing through $q$; that is, $\alpha(q, 0) = q$, $\frac{\partial \alpha}{\partial t}(q, t) = w(\alpha(q, t))$.

2. $\alpha$ is differentiable.

</div>

The map $\alpha$ is called the **(local) flow** of $w$ at $p$. Geometrically, Theorem 2 means that all trajectories which pass, for $t = 0$, in a certain neighborhood $V$ of $p$ may be "collected" into a single differentiable map. It is in this sense that we say that the trajectories depend differentiably on $p$.

### First Integrals

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Existence of a First Integral)</span></p>

Let $w$ be a vector field in an open set $U \subset R^2$ and let $p \in U$ be such that $w(p) \neq 0$. Then there exist a neighborhood $W \subset U$ of $p$ and a differentiable function $f\colon W \to R$ such that $f$ is constant along each trajectory of $w$ and $df\_q \neq 0$ for all $q \in W$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Choose a Cartesian coordinate system in $R^2$ such that $p = (0, 0)$ and $w(p)$ is in the direction of the $x$ axis. Let $\alpha\colon V \times I \to U$ be the local flow at $p$, and let $\tilde{\alpha}$ be the restriction of $\alpha$ to the rectangle $(V \times I) \cap \lbrace (x, y, t) \in R^3;\; x = 0 \rbrace$.

By the definition of local flow, $d\tilde{\alpha}\_p$ maps the unit vector of the $t$ axis into $w$ and maps the unit vector of the $y$ axis into itself. Therefore, $d\tilde{\alpha}\_p$ is nonsingular. It follows that there exists a neighborhood $W \subset U$ of $p$, where $\tilde{\alpha}^{-1}$ is defined and differentiable. The projection of $\tilde{\alpha}^{-1}(x, y)$ onto the $y$ axis is a differentiable function $\xi = f(x, y)$, which has the same value $\xi$ for all points of the trajectory passing through $(0, \xi)$. Since $d\tilde{\alpha}\_p$ is nonsingular, $W$ may be taken sufficiently small so that $df\_q \neq 0$ for all $q \in W$. $f$ is therefore the required function. **Q.E.D.**

</details>
</div>

The function $f$ of the above lemma is called a (local) **first integral** of $w$ in a neighborhood of $p$. For instance, if $w(x, y) = (y, -x)$ is defined in $R^2$, a first integral $f\colon R^2 - \lbrace (0, 0) \rbrace \to R$ is $f(x, y) = x^2 + y^2$.

### Fields of Directions and Integral Curves

Closely associated with the concept of vector field is the concept of field of directions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Field of Directions, Integral Curve)</span></p>

A **field of directions** $r$ in an open set $U \subset R^2$ is a correspondence which assigns to each $p \in U$ a line $r(p)$ in $R^2$ passing through $p$. $r$ is said to be **differentiable** at $p \in U$ if there exists a nonzero differentiable vector field $w$, defined in a neighborhood $V \subset U$ of $p$, such that for each $q \in V$, $w(q) \neq 0$ is a basis of $r(q)$; $r$ is **differentiable in $U$** if it is differentiable for every $p \in U$.

A regular connected curve $C$ is an **integral curve** of a field of directions $r$ defined in $U \subset R^2$ if $r(q)$ is the tangent line to $C$ at $q$ for all $q \in C$.

</div>

In the language of differential equations, a field of directions $r$ is usually given by

$$a(x, y)\,dx + b(x, y)\,dy = 0,$$

which simply means that at a point $q = (x, y)$ we associate the line through $q$ that contains the vector $(b, -a)$ or any of its nonzero multiples.

### Vector Fields on Surfaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vector Field on a Surface)</span></p>

A **vector field** $w$ in an open set $U \subset S$ of a regular surface $S$ is a correspondence which assigns to each $p \in U$ a vector $w(p) \in T\_p(S)$. The vector field $w$ is **differentiable** at $p \in U$ if, for some parametrization $\mathbf{x}(u, v)$ at $p$, the functions $a(u, v)$ and $b(u, v)$ given by

$$w(p) = a(u, v)\mathbf{x}_u + b(u, v)\mathbf{x}_v$$

are differentiable functions at $p$; it is clear that this definition does not depend on the choice of $\mathbf{x}$.

</div>

We can define, similarly, trajectories, field of directions, and integral curves on surfaces. Theorems 1 and 2 and the Lemma above extend easily to the present situation; up to a change of $R^2$ by $S$, the statements are exactly the same.

### The Main Theorem: Parametrization by Two Vector Fields

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Parametrization by Two Linearly Independent Vector Fields)</span></p>

Let $w\_1$ and $w\_2$ be two vector fields in an open set $U \subset S$, which are linearly independent at some point $p \in U$. Then it is possible to parametrize a neighborhood $V \subset U$ of $p$ in such a way that for each $q \in V$ the coordinate lines of this parametrization passing through $q$ are tangent to the lines determined by $w\_1(q)$ and $w\_2(q)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $W$ be a neighborhood of $p$ where the first integrals $f\_1$ and $f\_2$ of $w\_1$ and $w\_2$, respectively, are defined. Define a map $\varphi\colon W \to R^2$ by

$$\varphi(q) = (f_1(q), f_2(q)), \qquad q \in W.$$

Since $f\_1$ is constant on the trajectories of $w\_1$ and $(df\_1)\_p(w\_1) \neq 0$, we have at $p$

$$d\varphi_p(w_1) = ((df_1)_p(w_1),\; (df_2)_p(w_1)) = (0, a),$$

where $a = (df\_2)\_p(w\_1) \neq 0$, since $w\_1$ and $w\_2$ are linearly independent. Similarly,

$$d\varphi_p(w_2) = (b, 0),$$

where $b = (df\_1)\_p(w\_2) \neq 0$. It follows that $d\varphi\_p$ is nonsingular, and hence $\varphi$ is a local diffeomorphism. There exists, therefore, a neighborhood $\bar{U} \subset R^2$ of $\varphi(p)$ which is mapped diffeomorphically by $\mathbf{x} = \varphi^{-1}$ onto a neighborhood $V = \mathbf{x}(\bar{U})$ of $p$; that is, $\mathbf{x}$ is a parametrization of $S$ at $p$, whose coordinate curves

$$f_1(q) = \text{const.}, \qquad f_2(q) = \text{const.},$$

are tangent at $q$ to the lines determined by $w\_1(q)$, $w\_2(q)$, respectively. **Q.E.D.**

</details>
</div>

It should be remarked that the theorem does not imply that the coordinate curves can be so parametrized that their velocity vectors are $w\_1(q)$ and $w\_2(q)$. The statement of the theorem applies to the coordinate curves as regular (point set) curves; more precisely, we have

### Corollaries

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 1</span><span class="math-callout__name">(Integral Curves of Two Fields of Directions)</span></p>

Given two fields of directions $r$ and $r'$ in an open set $U \subset S$ such that at $p \in U$, $r(p) \neq r'(p)$, there exists a parametrization $\mathbf{x}$ in a neighborhood of $p$ such that the coordinate curves of $\mathbf{x}$ are the integral curves of $r$ and $r'$.

</div>

A first application of the above theorem is the proof of the existence of an orthogonal parametrization at any point of a regular surface.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2</span><span class="math-callout__name">(Existence of Orthogonal Parametrization)</span></p>

For all $p \in S$ there exists a parametrization $\mathbf{x}(u, v)$ in a neighborhood $V$ of $p$ such that the coordinate curves $u = \text{const.}$, $v = \text{const.}$ intersect orthogonally for each $q \in V$ (such an $\mathbf{x}$ is called an **orthogonal parametrization**).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Consider an arbitrary parametrization $\tilde{\mathbf{x}}\colon \tilde{U} \to S$ at $p$, and define two vector fields $w\_1 = \tilde{\mathbf{x}}\_{\tilde{u}}$, $w\_2 = -(\bar{F}/\bar{E})\tilde{\mathbf{x}}\_{\tilde{u}} + \tilde{\mathbf{x}}\_{\tilde{v}}$, where $\bar{E}$, $\bar{F}$, $\bar{G}$ are the coefficients of the first fundamental form in $\tilde{\mathbf{x}}$. Since $w\_1(q)$, $w\_2(q)$ are orthogonal vectors, for each $q \in \tilde{\mathbf{x}}(\tilde{U})$, an application of the theorem yields the required parametrization. **Q.E.D.**

</details>
</div>

A second application of the theorem (more precisely, of Corollary 1) is the existence of coordinates given by the asymptotic and principal directions.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3</span><span class="math-callout__name">(Asymptotic Parametrization at Hyperbolic Points)</span></p>

Let $p \in S$ be a hyperbolic point of $S$. Then it is possible to parametrize a neighborhood of $p$ in such a way that the coordinate curves of this parametrization are the asymptotic curves of $S$.

</div>

This follows because at a hyperbolic point $eg - f^2 < 0$, so the differential equation of the asymptotic curves $e(u')^2 + 2fu'v' + g(v')^2 = 0$ can be factored into two distinct linear equations, each determining a differentiable field of directions. By applying Corollary 1, we see that coordinate curves can be chosen as the integral curves of these two fields.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4</span><span class="math-callout__name">(Principal Parametrization at Nonumbilical Points)</span></p>

Let $p \in S$ be a nonumbilical point of $S$. Then it is possible to parametrize a neighborhood of $p$ in such a way that the coordinate curves of this parametrization are the lines of curvature of $S$.

</div>

Similarly, in a neighborhood of a nonumbilical point, it is possible to decompose the differential equation of the lines of curvature into distinct linear factors. By an analogous argument we obtain Corollary 4.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Asymptotic Curves of the Hyperbolic Paraboloid)</span></p>

Let $S$ be the hyperbolic paraboloid $z = x^2 - y^2$. As usual we parametrize the entire surface by $\mathbf{x}(u, v) = (u, v, u^2 - v^2)$. A simple computation shows that

$$e = \frac{2}{(1 + 4u^2 + 4v^2)^{1/2}}, \qquad f = 0, \qquad g = \frac{-2}{(1 + 4u^2 + 4v^2)^{1/2}}.$$

Thus, the equation of the asymptotic curves can be written as

$$\frac{2}{(1 + 4u^2 + 4v^2)^{1/2}}\bigl((u')^2 - (v')^2\bigr) = 0,$$

which can be factored into two linear equations and gives two fields of directions:

$$r_1\colon\; u' + v' = 0, \qquad r_2\colon\; u' - v' = 0.$$

The integral curves of these fields of directions are given by the two families of curves:

$$r_1\colon\; u + v = \text{const.}, \qquad r_2\colon\; u - v = \text{const.}$$

Now, the functions $f\_1(u, v) = u + v$, $f\_2(u, v) = u - v$ are clearly first integrals of the vector fields associated to $r\_1$ and $r\_2$, respectively. Thus, by setting

$$\tilde{u} = u + v, \qquad \tilde{v} = u - v,$$

we obtain a new parametrization for the entire surface $z = x^2 - y^2$ in which the coordinate curves are the asymptotic curves of the surface.

</div>

## 3-5. Ruled Surfaces and Minimal Surfaces

In differential geometry one finds quite a number of special cases (surfaces of revolution, parallel surfaces, ruled surfaces, minimal surfaces, etc.) which may either become interesting in their own right (like minimal surfaces), or give a beautiful example of the power and limitations of differentiable methods in geometry. We shall use this section to present some of these topics in more detail, specifically the theory of ruled surfaces and an introduction to the theory of minimal surfaces.

### A. Ruled Surfaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Ruled Surface)</span></p>

A (differentiable) **one-parameter family of (straight) lines** $\lbrace \alpha(t), w(t) \rbrace$ is a correspondence that assigns to each $t \in I$ a point $\alpha(t) \in R^3$ and a vector $w(t) \in R^3$, $w(t) \neq 0$, so that both $\alpha(t)$ and $w(t)$ depend differentiably on $t$. For each $t \in I$, the line $L\_t$ which passes through $\alpha(t)$ and is parallel to $w(t)$ is called the **line of the family at $t$**.

Given a one-parameter family of lines $\lbrace \alpha(t), w(t) \rbrace$, the parametrized surface

$$\mathbf{x}(t, v) = \alpha(t) + vw(t), \qquad t \in I,\; v \in R,$$

is called the **ruled surface** generated by the family $\lbrace \alpha(t), w(t) \rbrace$. The lines $L\_t$ are called the **rulings**, and the curve $\alpha(t)$ is called a **directrix** of the surface $\mathbf{x}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Basic Ruled Surfaces)</span></p>

1. The simplest examples of ruled surfaces are the tangent surfaces to a regular curve (cf. Example 4, Sec. 2-3), the cylinders, and the cones. A **cylinder** is a ruled surface generated by a one-parameter family $\lbrace \alpha(t), w(t) \rbrace$, $t \in I$, where $\alpha(I) \subset P$ a plane and $w(t)$ is parallel to a fixed direction in $R^3$. A **cone** is a ruled surface generated by a family $\lbrace \alpha(t), w(t) \rbrace$, $t \in I$, where $\alpha(I) \subset P$ and the rulings $L\_t$ all pass through a point $p \notin P$.

2. Let $S^1$ be the unit circle $x^2 + y^2 = 1$ in the $xy$ plane, and let $\alpha(s)$ be a parametrization of $S^1$ by arc length. For each $s$, let $w(s) = \alpha'(s) + e\_3$, where $e\_3$ is the unit vector of the $z$ axis. Then

$$\mathbf{x}(s, v) = \alpha(s) + v(\alpha'(s) + e_3)$$

is a ruled surface. It can be put into a more familiar form if we write $\mathbf{x}(s, v) = (\cos s - v\sin s,\; \sin s + v\cos s,\; v)$ and notice that $x^2 + y^2 - z^2 = 1 + v^2 - v^2 = 1$. This shows that the trace of $\mathbf{x}$ is a **hyperboloid of revolution**. If we take $w(s) = -\alpha'(s) + e\_3$, we again obtain the same surface, showing that the hyperboloid of revolution has two sets of rulings.

</div>

### The Line of Striction and the Distribution Parameter

We shall now study general ruled surfaces. We can assume, without loss of generality, that $\vert w(t)\vert  = 1$, $t \in I$. To develop the theory, we need the nontrivial assumption that $w'(t) \neq 0$ for all $t \in I$; if the zeros of $w'(t)$ are isolated, we can divide our surface into pieces such that the theory can be applied to each of them. The assumption $w'(t) \neq 0$, $t \in I$, is usually expressed by saying that the ruled surface $\mathbf{x}$ is **noncylindrical**.

We first want to find a parametrized curve $\beta(t)$ such that $\langle \beta'(t), w'(t) \rangle = 0$, $t \in I$, and $\beta(t)$ lies on the trace of $\mathbf{x}$; that is,

$$\beta(t) = \alpha(t) + u(t)w(t),$$

for some real-valued function $u = u(t)$. From the condition $\langle \beta', w' \rangle = 0$ and $\langle w, w' \rangle = 0$, we obtain

$$u = -\frac{\langle \alpha', w' \rangle}{\langle w', w' \rangle}.$$

The curve $\beta$ is then called the **line of striction**, and its points are called the **central points** of the ruled surface. The line of striction does not depend on the choice of the directrix $\alpha$.

We now take the line of striction as the directrix of the ruled surface and write

$$\mathbf{x}(t, u) = \beta(t) + uw(t).$$

With this choice, $\mathbf{x}\_t = \beta' + uw'$, $\mathbf{x}\_u = w$, and since $\langle w', w \rangle = 0$ and $\langle w', \beta' \rangle = 0$, we conclude that $\beta' \wedge w = \lambda w'$ for some function $\lambda = \lambda(t)$. Thus,

$$|\mathbf{x}_t \wedge \mathbf{x}_u|^2 = |\lambda w' + uw' \wedge w|^2 = \lambda^2|w'|^2 + u^2|w'|^2 = (\lambda^2 + u^2)|w'|^2.$$

It follows that the only singular points of the ruled surface are along the line of striction $u = 0$, and they will occur if and only if $\lambda(t) = 0$. The Gaussian curvature at regular points is

$$K = \frac{eg - f^2}{EG - F^2} = -\frac{\lambda^2}{(\lambda^2 + u^2)^2},$$

since $g = 0$ (because $\mathbf{x}\_{uu} = 0$).

This shows that, at regular points, *the Gaussian curvature $K$ of a ruled surface satisfies $K \le 0$, and $K$ is zero only along those rulings which meet the line of striction at a singular point*.

The function $\lambda(t)$ is called the **distribution parameter** of $\mathbf{x}$. Since the line of striction is independent of the choice of the directrix, the same holds for $\lambda$. The distribution parameter has a geometric interpretation: if $\theta$ is the angle which the normal vector at a point of a ruling makes with the normal vector at the central point of this ruling, then $\tan\theta = u/\lambda$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Hyperbolic Paraboloid as a Ruled Surface)</span></p>

Let $S$ be the hyperbolic paraboloid $z = kxy$, $k \neq 0$. To show that $S$ is a ruled surface, we observe that the lines $y = z/tk$, $x = t$, for each $t \neq 0$ belong to $S$. If we take the intersection of this family of lines with the plane $z = 0$, we obtain the curve $x = t$, $y = 0$, $z = 0$. Taking this curve as the directrix and $w(t)$ parallel to the lines $y = z/tk$, $x = t$, we obtain

$$\alpha(t) = (t, 0, 0), \qquad w(t) = \left(0, \frac{1}{k}, t\right).$$

This gives a ruled surface $\mathbf{x}(t, v) = (t, v/k, vt)$, $t \in R$, $v \in R$, the trace of which clearly agrees with $S$.

Since $\alpha'(t) = (1, 0, 0)$, the line of striction is $\alpha$ itself. The distribution parameter is

$$\lambda = \frac{(\beta', w, w')}{|w'|^2} = \frac{1 + k^2 t^2}{k^2}.$$

</div>

### Developable Surfaces

Among the ruled surfaces, the developables play a distinguished role.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Developable Surface)</span></p>

A ruled surface $\mathbf{x}(t, v) = \alpha(t) + vw(t)$ generated by the family $\lbrace \alpha(t), w(t) \rbrace$ with $\vert w(t)\vert  \equiv 1$ is said to be **developable** if

$$(w, w', \alpha') \equiv 0.$$

</div>

To find a geometric interpretation of this condition, we compute the Gaussian curvature of a developable surface at a regular point. A computation entirely similar to the one for general ruled surfaces gives

$$g = 0, \qquad f = \frac{(w, w', \tilde{\alpha}')}{|\mathbf{x}_t \wedge \mathbf{x}_u|^2}.$$

By the developability condition, $f \equiv 0$; hence,

$$K = \frac{eg - f^2}{EG - F^2} \equiv 0.$$

This implies that, *at regular points, the Gaussian curvature of a developable surface is identically zero*.

We can now distinguish two nonexhaustive cases of developable surfaces:

1. $w(t) \wedge w'(t) \equiv 0$. This implies that $w'(t) \equiv 0$. Thus, $w(t)$ is constant and the ruled surface is a cylinder over a curve obtained by intersecting the cylinder with a plane normal to $w(t)$.

2. $w(t) \wedge w'(t) \neq 0$ for all $t \in I$. In this case, the surface is noncylindrical, and we can apply our previous work. We can determine the line of striction $\beta(t)$ and check that the distribution parameter is

$$\lambda = \frac{(\beta', w, w')}{|w'|^2} \equiv 0.$$

Therefore, the line of striction will be the locus of singular points of the developable surface. If $\beta'(t) \neq 0$ for all $t \in I$, it follows from $\lambda \equiv 0$ and $\langle \beta', w' \rangle \equiv 0$ that $w$ is parallel to $\beta'$. Thus, the ruled surface is the tangent surface of $\beta$. If $\beta'(t) = 0$ for all $t \in I$, then the line of striction is a point, and the ruled surface is a **cone** with vertex at this point.

Of course, the above cases do not exhaust all possibilities. As usual, if there is a clustering of zeros of the functions involved, the analysis may become rather complicated. At any rate, away from these cluster points, a developable surface is a union of pieces of cylinders, cones, and tangent surfaces.

### The Envelope of Tangent Planes along a Curve

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Envelope of Tangent Planes along a Curve)</span></p>

Let $S$ be a regular surface and $\alpha = \alpha(s)$ a curve on $S$ parametrized by arc length. Assume that $\alpha$ is nowhere tangent to an asymptotic direction. Consider the ruled surface

$$\mathbf{x}(s, v) = \alpha(s) + v\frac{N(s) \wedge N'(s)}{|N'(s)|},$$

where by $N(s)$ we denote the unit normal vector of $S$ restricted to the curve $\alpha(s)$ (since $\alpha'(s)$ is not an asymptotic direction, $N'(s) \neq 0$ for all $s$). We shall show that $\mathbf{x}$ is a developable surface which is regular in a neighborhood of $v = 0$ and is tangent to $S$ along $\alpha(s)$.

The rulings of $\mathbf{x}$ are the limiting positions of the intersection of neighboring tangent planes of the family $\lbrace T\_{\alpha(s)}(S) \rbrace$. $\mathbf{x}$ is called the **envelope of the family of tangent planes** to $S$ along $\alpha(s)$.

To show that $\mathbf{x}$ is a developable surface, we check condition $(w, w', \alpha') \equiv 0$. By a straightforward computation,

$$\left\langle \frac{N \wedge N'}{|N'|} \wedge \left(\frac{N \wedge N'}{|N'|}\right)', \alpha' \right\rangle = \frac{1}{|N'|^2}\langle N \wedge N', (N \wedge N')' \rangle \cdot \langle N, \alpha' \rangle = 0.$$

This proves our claim. The surface $\mathbf{x}$ is regular in a neighborhood of $v = 0$ and the unit normal vector of $\mathbf{x}$ at $\mathbf{x}(s, 0)$ agrees with $N(s)$. Thus, $\mathbf{x}$ is tangent to $S$ along $\alpha(s)$.

For instance, if $\alpha$ is a parametrization of a parallel of a sphere $S^2$, then the envelope of tangent planes of $S^2$ along $\alpha$ is either a cylinder (if the parallel is an equator) or a cone (if the parallel is not an equator).

</div>

### B. Minimal Surfaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Minimal Surface)</span></p>

A regular parametrized surface is called **minimal** if its mean curvature vanishes everywhere. A regular surface $S \subset R^3$ is **minimal** if each of its parametrizations is minimal.

</div>

### Normal Variations and Minimality

To explain why we use the word *minimal* for such surfaces, we need to introduce the notion of a **variation**. Let $\mathbf{x}\colon U \subset R^2 \to R^3$ be a regular parametrized surface. Choose a bounded domain $D \subset U$ and a differentiable function $h\colon \bar{D} \to R$, where $\bar{D}$ is the union of the domain $D$ with its boundary $\partial D$. The **normal variation** of $\mathbf{x}(\bar{D})$, determined by $h$, is the map

$$\varphi(u, v, t) = \mathbf{x}(u, v) + th(u, v)N(u, v), \qquad (u, v) \in \bar{D},\; t \in (-\epsilon, \epsilon).$$

For each fixed $t \in (-\epsilon, \epsilon)$, the map $\mathbf{x}^t(u, v) = \varphi(u, v, t)$ is a parametrized surface with

$$\frac{\partial \mathbf{x}^t}{\partial u} = \mathbf{x}_u + thN_u + th_u N, \qquad \frac{\partial \mathbf{x}^t}{\partial v} = \mathbf{x}_v + thN_v + th_v N.$$

If we denote by $E^t$, $F^t$, $G^t$ the coefficients of the first fundamental form of $\mathbf{x}^t$, then using $\langle \mathbf{x}\_u, N\_u \rangle = -e$, $\langle \mathbf{x}\_u, N\_v \rangle + \langle \mathbf{x}\_v, N\_u \rangle = -2f$, and $\langle \mathbf{x}\_v, N\_v \rangle = -g$, one obtains

$$E^t G^t - (F^t)^2 = (EG - F^2)(1 - 4thH) + R,$$

where $\lim\_{t \to 0}(R/t) = 0$. It follows that if $\epsilon$ is small, $A$ is a differentiable function and its derivative at $t = 0$ is

$$A'(0) = -\int_D 2hH\sqrt{EG - F^2}\,du\,dv.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Variational Characterization of Minimal Surfaces)</span></p>

Let $\mathbf{x}\colon U \to R^3$ be a regular parametrized surface and let $D \subset U$ be a bounded domain in $U$. Then $\mathbf{x}$ is minimal if and only if $A'(0) = 0$ for all such $D$ and all normal variations of $\mathbf{x}(\bar{D})$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

If $\mathbf{x}$ is minimal, $H \equiv 0$ and the condition is clearly satisfied. Conversely, assume that the condition is satisfied and that $H(q) \neq 0$ for some $q \in D$. Choose $h\colon \bar{D} \to R$ such that $h(q) = H(q)$ and $h$ is identically zero outside a small neighborhood of $q$. Then $A'(0) < 0$ for the variation determined by this $h$, and that is a contradiction. **Q.E.D.**

</details>
</div>

Thus, any bounded region $\mathbf{x}(\bar{D})$ of a minimal surface $\mathbf{x}$ is a critical point for the area function of any normal variation of $\mathbf{x}(\bar{D})$. It should be noticed that this critical point may not be a minimum and that this makes the word *minimal* seem somewhat awkward. It is, however, a time-honored terminology which was introduced by Lagrange (who first defined a minimal surface) in 1760.

Minimal surfaces are usually associated with soap films that can be obtained by dipping a wire frame into a soap solution and withdrawing it carefully.

### The Mean Curvature Vector and Isothermal Parametrizations

The **mean curvature vector** is defined by $\mathbf{H} = HN$. If we deform $\mathbf{x}(\bar{D})$ in the direction of the vector $\mathbf{H}$, that is, if we choose $h = H$, then

$$A'(0) = -2\int_D \langle \mathbf{H}, \mathbf{H} \rangle\sqrt{EG - F^2}\,du\,dv < 0.$$

This means that *if we deform $\mathbf{x}(\bar{D})$ in the direction of the vector $\mathbf{H}$, the area is initially decreasing*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Isothermal Parametrization)</span></p>

A regular parametrized surface $\mathbf{x} = \mathbf{x}(u, v)$ is said to be **isothermal** if $\langle \mathbf{x}\_u, \mathbf{x}\_u \rangle = \langle \mathbf{x}\_v, \mathbf{x}\_v \rangle$ and $\langle \mathbf{x}\_u, \mathbf{x}\_v \rangle = 0$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Laplacian and Mean Curvature for Isothermal Parametrizations)</span></p>

Let $\mathbf{x} = \mathbf{x}(u, v)$ be a regular parametrized surface and assume that $\mathbf{x}$ is isothermal. Then

$$\mathbf{x}_{uu} + \mathbf{x}_{vv} = 2\lambda^2 \mathbf{H},$$

where $\lambda^2 = \langle \mathbf{x}\_u, \mathbf{x}\_u \rangle = \langle \mathbf{x}\_v, \mathbf{x}\_v \rangle$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $\mathbf{x}$ is isothermal, $\langle \mathbf{x}\_u, \mathbf{x}\_u \rangle = \langle \mathbf{x}\_v, \mathbf{x}\_v \rangle$ and $\langle \mathbf{x}\_u, \mathbf{x}\_v \rangle = 0$. By differentiation,

$$\langle \mathbf{x}_{uu}, \mathbf{x}_u \rangle = \langle \mathbf{x}_{vu}, \mathbf{x}_v \rangle = -\langle \mathbf{x}_u, \mathbf{x}_{vv} \rangle.$$

Thus, $\langle \mathbf{x}\_{uu} + \mathbf{x}\_{vv}, \mathbf{x}\_u \rangle = 0$. Similarly, $\langle \mathbf{x}\_{uu} + \mathbf{x}\_{vv}, \mathbf{x}\_v \rangle = 0$. It follows that $\mathbf{x}\_{uu} + \mathbf{x}\_{vv}$ is parallel to $N$. Since $\mathbf{x}$ is isothermal,

$$H = \frac{1}{2}\frac{g + e}{\lambda^2}.$$

Thus, $2\lambda^2 H = g + e = \langle N, \mathbf{x}\_{uu} + \mathbf{x}\_{vv} \rangle$; hence, $\mathbf{x}\_{uu} + \mathbf{x}\_{vv} = 2\lambda^2 \mathbf{H}$. **Q.E.D.**

</details>
</div>

The **Laplacian** $\Delta f$ of a differentiable function $f\colon U \subset R^2 \to R$ is defined by $\Delta f = (\partial^2 f / \partial u^2) + (\partial^2 f / \partial v^2)$, $(u, v) \in U$. We say that $f$ is **harmonic** in $U$ if $\Delta f = 0$. From Prop. 2, we obtain

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Minimal Isothermal Surfaces Have Harmonic Coordinates)</span></p>

Let $\mathbf{x}(u, v) = (x(u, v), y(u, v), z(u, v))$ be a parametrized surface and assume that $\mathbf{x}$ is isothermal. Then $\mathbf{x}$ is minimal if and only if its coordinate functions $x$, $y$, $z$ are harmonic.

</div>

### Examples of Minimal Surfaces

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Catenoid)</span></p>

The **catenoid** is the surface generated by rotating the catenary $y = a\cosh(z/a)$ about the $z$ axis. It is given by

$$\mathbf{x}(u, v) = (a\cosh v\cos u,\; a\cosh v\sin u,\; av), \qquad 0 < u < 2\pi,\; -\infty < v < \infty.$$

It is easily checked that $E = G = a^2\cosh^2 v$, $F = 0$, and that $\mathbf{x}\_{uu} + \mathbf{x}\_{vv} = 0$. Thus, the catenoid is a minimal surface. It can be characterized as the only surface of revolution which is minimal.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof that the Catenoid is the Only Minimal Surface of Revolution</summary>

We want to find a curve $y = f(x)$ such that, when rotated about the $x$ axis, it describes a minimal surface. Since the parallels and meridians of a surface of revolution are lines of curvature, we must have that the curvature of $y = f(x)$ is the negative of the normal curvature of the circle generated by the point $f(x)$ (both are principal curvatures). The curvature of $y = f(x)$ is $y''/(1 + (y')^2)^{3/2}$ and the normal curvature of the circle is the projection $(= 1/y)\cos\varphi$ over the normal $N$ to the surface. Since $-\cos\varphi = \cos\theta$ and $\tan\theta = y'$, we obtain

$$\frac{y''}{(1 + (y')^2)^{3/2}} = \frac{1}{y}\frac{1}{(1 + (y')^2)^{1/2}}.$$

Setting $1 + (y')^2 = z$ (hence, $2y''y' = z'$), we have $z'/z = 2y'/y$, which gives $\log z = \log(yk)^2$, or $1 + (y')^2 = (yk)^2$. This separates to

$$\frac{k\,dy}{\sqrt{(yk)^2 - 1}} = k\,dx,$$

which integrates to $\cosh^{-1}(yk) = kx + c$ or $y = (1/k)\cosh(kx + c)$. Thus, in the neighborhood of a point where $f' \neq 0$, the curve $y = f(x)$ is a catenary. But then $y'$ can only be zero at $x = 0$, and if the surface is to be connected, it is by continuity a catenoid, as we claimed. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Helicoid)</span></p>

The **helicoid** (cf. Example 3, Sec. 2-5) is given by

$$\mathbf{x}(u, v) = (a\sinh v\cos u,\; a\sinh v\sin u,\; au).$$

It is easily checked that $E = G = a^2\cosh^2 v$, $F = 0$, and $\mathbf{x}\_{uu} + \mathbf{x}\_{vv} = 0$. Thus, the helicoid is a minimal surface. It has the additional property that it is the only minimal surface, other than the plane, which is also a ruled surface (assuming that the zeros of the Gaussian curvature are isolated).

</div>

The helicoid and the catenoid were discovered in 1776 by Meusnier, who also proved that Lagrange's definition of minimal surfaces as critical points of a variational problem is equivalent to the vanishing of the mean curvature. For a long time, they were the only known examples of minimal surfaces. Only in 1835 did Scherk find further examples.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Enneper's Minimal Surface)</span></p>

**Enneper's surface** is the parametrized surface

$$\mathbf{x}(u, v) = \left(u - \frac{u^3}{3} + uv^2,\; v - \frac{v^3}{3} + vu^2,\; u^2 - v^2\right), \qquad (u, v) \in R^2,$$

which is easily seen to be minimal. Notice that by changing $(u, v)$ into $(-v, u)$ we change, in the surface, $(x, y, z)$ into $(-y, x, -z)$. Thus, if we perform a positive rotation of $\pi/2$ about the $z$ axis and follow it by a symmetry in the $xy$ plane, the surface remains invariant.

An interesting feature of Enneper's surface is that it has self-intersections. By setting $u = \rho\cos\theta$, $v = \rho\sin\theta$, and computing when $\mathbf{x}(\rho\_1, \theta\_1) = \mathbf{x}(\rho\_2, \theta\_2)$, one can show that the self-intersection curves lie on the planes $y = 0$ and $x = 0$.

</div>

### Connection with Analytic Functions

A useful relation between minimal surfaces and analytic functions of a complex variable can be established as follows. Let $\mathbb{C}$ denote the complex plane, identified with $R^2$ by setting $\zeta = u + iv$. Let $\mathbf{x}\colon U \subset R^2 \to R^3$ be a regular parametrized surface and define complex functions $\varphi\_1$, $\varphi\_2$, $\varphi\_3$ by

$$\varphi_1(\zeta) = \frac{\partial x}{\partial u} - i\frac{\partial x}{\partial v}, \qquad \varphi_2(\zeta) = \frac{\partial y}{\partial u} - i\frac{\partial y}{\partial v}, \qquad \varphi_3(\zeta) = \frac{\partial z}{\partial u} - i\frac{\partial z}{\partial v},$$

where $x$, $y$, and $z$ are the component functions of $\mathbf{x}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Isothermal Minimal Surfaces and Analytic Functions)</span></p>

$\mathbf{x}$ is isothermal if and only if $\varphi\_1^2 + \varphi\_2^2 + \varphi\_3^2 \equiv 0$. If this last condition is satisfied, $\mathbf{x}$ is minimal if and only if $\varphi\_1$, $\varphi\_2$, and $\varphi\_3$ are analytic functions.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By a simple computation, we obtain that $\varphi\_1^2 + \varphi\_2^2 + \varphi\_3^2 = E - G + 2iF$, whence the first part of the lemma. Furthermore, $\mathbf{x}\_{uu} + \mathbf{x}\_{vv} = 0$ if and only if

$$\frac{\partial}{\partial u}\left(\frac{\partial x}{\partial v}\right) = -\frac{\partial}{\partial v}\left(\frac{\partial x}{\partial v}\right),$$

and similarly for $y$ and $z$, which give one-half of the Cauchy-Riemann equations for $\varphi\_1$, $\varphi\_2$, $\varphi\_3$. Since the other half is automatically satisfied, we conclude that $\mathbf{x}\_{uu} + \mathbf{x}\_{vv} = 0$ if and only if $\varphi\_1$, $\varphi\_2$, and $\varphi\_3$ are analytic. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Scherk's Minimal Surface)</span></p>

**Scherk's surface** is given by

$$\mathbf{x}(u, v) = \left(\arg\frac{\zeta + i}{\zeta - i},\; \arg\frac{\zeta + 1}{\zeta - 1},\; \log\left|\frac{\zeta^2 + 1}{\zeta^2 - 1}\right|\right), \qquad \zeta \neq \pm 1,\; \zeta \neq \pm i,$$

where $\zeta = u + iv$. One computes that

$$\varphi_1 = \frac{-2}{1 + \zeta^2}, \qquad \varphi_2 = \frac{-2i}{1 - \zeta^2}, \qquad \varphi_3 = \frac{4\zeta}{1 - \zeta^4}.$$

Since $\varphi\_1^2 + \varphi\_2^2 + \varphi\_3^2 \equiv 0$ and $\varphi\_1$, $\varphi\_2$, $\varphi\_3$ are analytic, $\mathbf{x}$ is an isothermal parametrization of a minimal surface. It is easily seen from the expressions of $x$, $y$, and $z$ that

$$z = \log\frac{\cos y}{\cos x}.$$

This representation shows that Scherk's surface is defined on the chess-board pattern of the $xy$ plane (except at the vertices of the squares, where the surface is actually a vertical line).

</div>

### Osserman's Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Osserman)</span></p>

Let $S \subset R^3$ be a regular, closed (as a subset of $R^3$) minimal surface which is not a plane. Then the image of the Gauss map $N\colon S \to S^2$ is dense in the sphere $S^2$ (that is, arbitrarily close to any point of $S^2$ there is a point of $N(S) \subset S^2$).

</div>

Minimal surfaces are perhaps the best-studied surfaces in differential geometry, and we have barely touched the subject. A very readable introduction can be found in R. Osserman, *A Survey of Minimal Surfaces*. The theory has developed into a rich branch of differential geometry in which interesting and nontrivial questions are still being investigated. It has deep connections with analytic functions of complex variables and partial differential equations.

# Chapter 4 — The Intrinsic Geometry of Surfaces

## 4-1. Introduction

In Chap. 2 we introduced the first fundamental form of a surface $S$ and showed how it can be used to compute simple metric concepts on $S$ (length, angle, area, etc.). The important point is that such computations can be made without "leaving" the surface, once the first fundamental form is known. Because of this, these concepts are said to be **intrinsic** to the surface $S$.

The geometry of the first fundamental form, however, does not exhaust itself with the simple concepts mentioned above. As we shall see in this chapter, many other important local properties of a surface can be expressed only in terms of the first fundamental form. The study of such properties is called the **intrinsic geometry** of the surface. This chapter is dedicated to intrinsic geometry.

In Sec. 4-2 we shall define the notion of isometry, which essentially makes precise the intuitive idea of two surfaces having "the same" first fundamental forms.

In Sec. 4-3 we shall prove the celebrated Gauss formula that expresses the Gaussian curvature $K$ as a function of the coefficients of the first fundamental form and its derivatives. This means that $K$ is an intrinsic concept, a very striking fact if we consider that $K$ was defined using the second fundamental form.

In Sec. 4-5 we shall study the Gauss-Bonnet theorem both in its local and global versions. This is probably the most important theorem of this book.

In Sec. 4-6 we shall define the exponential map and use it to introduce two special coordinate systems, namely, the normal coordinates and the geodesic polar coordinates.

## 4-2. Isometries; Conformal Maps

Examples 1 and 2 of Sec. 2-5 display an interesting peculiarity. Although the cylinder and the plane are distinct surfaces, their first fundamental forms are "equal" (at least in the coordinate neighborhoods that we have considered). This means that insofar as intrinsic metric questions are concerned (length, angle, area), the plane and the cylinder behave locally in the same way. (This is intuitively clear, since by cutting a cylinder along a generator we may unroll it onto a part of a plane.)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Isometry)</span></p>

A diffeomorphism $\varphi\colon S \to \bar{S}$ is an **isometry** if for all $p \in S$ and all pairs $w\_1, w\_2 \in T\_p(S)$ we have

$$\langle w_1, w_2 \rangle_p = \langle d\varphi_p(w_1),\; d\varphi_p(w_2) \rangle_{\varphi(p)}.$$

The surfaces $S$ and $\bar{S}$ are then said to be **isometric**.

</div>

In other words, a diffeomorphism $\varphi$ is an isometry if the differential $d\varphi$ preserves the inner product. It follows that, $d\varphi$ being an isometry,

$$I_p(w) = \langle w, w \rangle_p = \langle d\varphi_p(w), d\varphi_p(w) \rangle_{\varphi(p)} = I_{\varphi(p)}(d\varphi_p(w))$$

for all $w \in T\_p(S)$. Conversely, if a diffeomorphism $\varphi$ preserves the first fundamental form, then by the identity $2\langle w\_1, w\_2 \rangle = I(w\_1 + w\_2) - I(w\_1) - I(w\_2)$, $\varphi$ is an isometry.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Local Isometry)</span></p>

A map $\varphi\colon V \to \bar{S}$ of a neighborhood $V$ of $p \in S$ is a **local isometry** at $p$ if there exists a neighborhood $\bar{V}$ of $\varphi(p) \in \bar{S}$ such that $\varphi\colon V \to \bar{V}$ is an isometry. If there exists a local isometry into $\bar{S}$ at every $p \in S$, $S$ and $\bar{S}$ are **locally isometric**. If $S$ is locally isometric to $\bar{S}$ and $\bar{S}$ is locally isometric to $S$, then the relation is symmetric.

</div>

It is clear that if $\varphi\colon S \to \bar{S}$ is a diffeomorphism and a local isometry for every $p \in S$, then $\varphi$ is an isometry (globally). It may, however, happen that two surfaces are locally isometric without being (globally) isometric.

### Criterion for Local Isometry

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Local Isometry via First Fundamental Form)</span></p>

Assume the existence of parametrizations $\mathbf{x}\colon U \to S$ and $\bar{\mathbf{x}}\colon U \to \bar{S}$ such that $E = \bar{E}$, $F = \bar{F}$, $G = \bar{G}$ in $U$. Then the map $\varphi = \bar{\mathbf{x}} \circ \mathbf{x}^{-1}\colon \mathbf{x}(U) \to \bar{S}$ is a local isometry.

</div>

### Examples of Isometries

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Cylinder and Plane are Locally Isometric)</span></p>

Let $\varphi$ be a map of the coordinate neighborhood $\bar{\mathbf{x}}(U)$ of the cylinder given in Example 2 of Sec. 2-5 into the plane $\mathbf{x}(R^2)$ of Example 1 of Sec. 2-5, defined by $\varphi = \mathbf{x} \circ \bar{\mathbf{x}}^{-1}$. Then $\varphi$ is a local isometry. In fact, each vector $w$, tangent to the cylinder at a point $p \in \bar{\mathbf{x}}(U)$, is tangent to a curve $\bar{\mathbf{x}}(u(t), v(t))$, where $(u(t), v(t))$ is a curve in $U \subset R^2$. Thus, $w = \bar{\mathbf{x}}\_u u' + \bar{\mathbf{x}}\_v v'$. On the other hand, $d\varphi(w)$ is tangent to the curve $\mathbf{x}(u(t), v(t))$. Thus, $d\varphi(w) = \mathbf{x}\_u u' + \mathbf{x}\_v v'$. Since $E = \bar{E}$, $F = \bar{F}$, $G = \bar{G}$, we obtain

$$I_p(w) = E(u')^2 + 2Fu'v' + G(v')^2 = \bar{E}(u')^2 + 2\bar{F}u'v' + \bar{G}(v')^2 = I_{\varphi(p)}(d\varphi_p(w)),$$

and it follows that the cylinder $x^2 + y^2 = 1$ is locally isometric to a plane. The isometry cannot be extended to the entire cylinder because the cylinder is not even homeomorphic to a plane.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Catenoid and Helicoid are Locally Isometric)</span></p>

Let $S$ be a surface of revolution and let

$$\mathbf{x}(u, v) = (f(v)\cos u,\; f(v)\sin u,\; g(v)),$$

be a parametrization of $S$. In particular, the surface of revolution of the catenary, $x = a\cosh v$, $z = av$, $-\infty < v < \infty$, has the following parametrization:

$$\mathbf{x}(u, v) = (a\cosh v\cos u,\; a\cosh v\sin u,\; av), \qquad 0 < u < 2\pi,\; -\infty < v < \infty,$$

relative to which the coefficients of the first fundamental form are $E = a^2\cosh^2 v$, $F = 0$, $G = a^2(1 + \sinh^2 v) = a^2\cosh^2 v$. This surface of revolution is the **catenoid**. We shall show that the catenoid is locally isometric to the helicoid.

A parametrization for the helicoid is given by $\bar{\mathbf{x}}(\bar{u}, \bar{v}) = (\bar{v}\cos\bar{u},\; \bar{v}\sin\bar{u},\; a\bar{u})$, $0 < \bar{u} < 2\pi$, $-\infty < \bar{v} < \infty$. Let us make the following change of parameters: $\bar{u} = u$, $\bar{v} = a\sinh v$. This gives a new parametrization of the helicoid:

$$\bar{\mathbf{x}}(u, v) = (a\sinh v\cos u,\; a\sinh v\sin u,\; au),$$

relative to which the coefficients of the first fundamental form are given by $\bar{E} = a^2\cosh^2 v$, $\bar{F} = 0$, $\bar{G} = a^2\cosh^2 v$. Using Prop. 1, we conclude that the catenoid and the helicoid are locally isometric.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Cone Minus a Generator is Locally Isometric to a Plane)</span></p>

The one-sheeted cone (minus the vertex) $z = +k\sqrt{x^2 + y^2}$, $(x, y) \neq (0, 0)$, is locally isometric to a plane. To show this, let $U \subset R^2$ be the open set given in polar coordinates $(\rho, \theta)$ by $0 < \rho < \infty$, $0 < \theta < 2\pi\sin\alpha$, where $2\alpha$ ($0 < 2\alpha < \pi$) is the angle at the vertex of the cone (i.e., $\cot\alpha = k$), and let $F\colon U \to R^3$ be the map

$$F(\rho, \theta) = \left(\rho\sin\alpha\cos\left(\frac{\theta}{\sin\alpha}\right),\; \rho\sin\alpha\sin\left(\frac{\theta}{\sin\alpha}\right),\; \rho\cos\alpha\right).$$

It is clear that $F(U)$ is contained in the cone since $k\sqrt{x^2 + y^2} = \cot\alpha \cdot \rho\sin\alpha = \rho\cos\alpha = z$. Furthermore, when $\theta$ describes the interval $(0, 2\pi\sin\alpha)$, $\theta/\sin\alpha$ describes the interval $(0, 2\pi)$. Thus, all points of the cone except the generator $\theta = 0$ are covered by $F(U)$.

The first fundamental form of $U$ in the parametrization $\bar{\mathbf{x}}(\rho, \theta) = (\rho\cos\theta,\; \rho\sin\theta,\; 0)$ gives $\bar{E} = 1$, $\bar{F} = 0$, $\bar{G} = \rho^2$. On the other hand, the coefficients of the first fundamental form of the cone in the parametrization $F \circ \mathbf{x}$ are $E = 1$, $F = 0$, $G = \rho^2$. From Prop. 1 we conclude that $F$ is a local isometry.

</div>

### Intrinsic Distance

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intrinsic Distance)</span></p>

The fact that we can compute lengths of curves on a surface $S$ by using only its first fundamental form allows us to introduce a notion of "intrinsic" distance for points in $S$. Roughly speaking, we define the (intrinsic) **distance** $d(p, q)$ between two points of $S$ as the infimum of the length of curves on $S$ joining $p$ and $q$. (We shall go into that in more detail in Sec. 5-3.) This distance is clearly greater than or equal to the distance $\|p - q\|$ of $p$ to $q$ as points in $R^3$. The distance $d$ is invariant under isometries; that is, if $\varphi\colon S \to \bar{S}$ is an isometry, then $d(p, q) = d(\varphi(p), \varphi(q))$, $p, q \in S$.

</div>

### Conformal Maps

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Conformal Map)</span></p>

A diffeomorphism $\varphi\colon S \to \bar{S}$ is called a **conformal map** if for all $p \in S$ and all $v\_1, v\_2 \in T\_p(S)$ we have

$$\langle d\varphi_p(v_1),\; d\varphi_p(v_2) \rangle = \lambda^2(p)\langle v_1, v_2 \rangle_p,$$

where $\lambda^2$ is a nowhere-zero differentiable function on $S$; the surfaces $S$ and $\bar{S}$ are then said to be **conformal**. A map $\varphi\colon V \to \bar{S}$ of a neighborhood $V$ of $p \in S$ into $\bar{S}$ is a **local conformal map** at $p$ if there exists a neighborhood $\bar{V}$ of $\varphi(p)$ such that $\varphi\colon V \to \bar{V}$ is a conformal map. If for each $p \in S$ there exists a local conformal map at $p$, the surface $S$ is said to be **locally conformal** to $\bar{S}$.

</div>

The geometric meaning of this definition is that the angles (but not necessarily the lengths) are preserved by conformal maps.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Local Conformality via First Fundamental Form)</span></p>

Let $\mathbf{x}\colon U \to S$ and $\bar{\mathbf{x}}\colon U \to \bar{S}$ be parametrizations such that $E = \lambda^2 \bar{E}$, $F = \lambda^2 \bar{F}$, $G = \lambda^2 \bar{G}$ in $U$, where $\lambda^2$ is a nowhere-zero differentiable function in $U$. Then the map $\varphi = \bar{\mathbf{x}} \circ \mathbf{x}^{-1}\colon \mathbf{x}(U) \to \bar{S}$ is a local conformal map.

</div>

Local conformality is easily seen to be an equivalence relation; that is, if $S\_1$ is locally conformal to $S\_2$ and $S\_2$ is locally conformal to $S\_3$, then $S\_1$ is locally conformal to $S\_3$. The most important property of conformal maps is given by the following theorem, which we shall not prove.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Any Two Regular Surfaces are Locally Conformal)</span></p>

Any two regular surfaces are locally conformal.

</div>

The proof is based on the possibility of parametrizing a neighborhood of any point of a regular surface in such a way that the coefficients of the first fundamental form are $E = \lambda^2(u, v) > 0$, $F = 0$, $G = \lambda^2(u, v)$. Such a coordinate system is called **isothermal**. Once the existence of an isothermal coordinate system of a regular surface $S$ is assumed, $S$ is clearly locally conformal to a plane, and by composition locally conformal to any other surface.

## 4-3. The Gauss Theorem and the Equations of Compatibility

The properties of Chap. 3 were obtained from the study of the variation of the tangent plane in a neighborhood of a point. Proceeding with the analogy with curves, we are going to assign to each point of a surface a trihedron (the analogue of Frenet's trihedron) and study the derivatives of its vectors.

$S$ will denote, as usual, a regular, orientable, and oriented surface. Let $\mathbf{x}\colon U \subset R^2 \to S$ be a parametrization in the orientation of $S$. It is possible to assign to each point of $\mathbf{x}(U)$ a natural trihedron given by the vectors $\mathbf{x}\_u$, $\mathbf{x}\_v$, and $N$. The study of this trihedron will be the subject of this section.

### The Christoffel Symbols

By expressing the derivatives of the vectors $\mathbf{x}\_u$, $\mathbf{x}\_v$, and $N$ in the basis $\lbrace \mathbf{x}\_u, \mathbf{x}\_v, N \rbrace$, we obtain

$$\mathbf{x}_{uu} = \Gamma_{11}^1 \mathbf{x}_u + \Gamma_{11}^2 \mathbf{x}_v + L_1 N,$$

$$\mathbf{x}_{uv} = \Gamma_{12}^1 \mathbf{x}_u + \Gamma_{12}^2 \mathbf{x}_v + L_2 N,$$

$$\mathbf{x}_{vu} = \Gamma_{21}^1 \mathbf{x}_u + \Gamma_{21}^2 \mathbf{x}_v + \bar{L}_2 N,$$

$$\mathbf{x}_{vv} = \Gamma_{22}^1 \mathbf{x}_u + \Gamma_{22}^2 \mathbf{x}_v + L_3 N,$$

$$N_u = a_{11}\mathbf{x}_u + a_{21}\mathbf{x}_v, \qquad N_v = a_{12}\mathbf{x}_u + a_{22}\mathbf{x}_v,$$

where the $a\_{ij}$ were obtained in Chap. 3. The coefficients $\Gamma\_{ij}^k$, $i, j, k = 1, 2$, are called the **Christoffel symbols** of $S$ in the parametrization $\mathbf{x}$. Since $\mathbf{x}\_{uv} = \mathbf{x}\_{vu}$, we conclude that $\Gamma\_{12}^1 = \Gamma\_{21}^1$ and $\Gamma\_{12}^2 = \Gamma\_{21}^2$; that is, the Christoffel symbols are symmetric relative to the lower indices.

By taking the inner product of the first four relations with $N$, we immediately obtain $L\_1 = e$, $L\_2 = \bar{L}\_2 = f$, $L\_3 = g$, where $e$, $f$, $g$ are the coefficients of the second fundamental form of $S$.

To determine the Christoffel symbols, we take the inner product of the first four relations with $\mathbf{x}\_u$ and $\mathbf{x}\_v$, obtaining the system

$$\Gamma_{11}^1 E + \Gamma_{11}^2 F = \langle \mathbf{x}_{uu}, \mathbf{x}_u \rangle = \tfrac{1}{2}E_u, \qquad \Gamma_{11}^1 F + \Gamma_{11}^2 G = \langle \mathbf{x}_{uu}, \mathbf{x}_v \rangle = F_u - \tfrac{1}{2}E_v,$$

$$\Gamma_{12}^1 E + \Gamma_{12}^2 F = \langle \mathbf{x}_{uv}, \mathbf{x}_u \rangle = \tfrac{1}{2}E_v, \qquad \Gamma_{12}^1 F + \Gamma_{12}^2 G = \langle \mathbf{x}_{uv}, \mathbf{x}_v \rangle = \tfrac{1}{2}G_u,$$

$$\Gamma_{22}^1 E + \Gamma_{22}^2 F = \langle \mathbf{x}_{vv}, \mathbf{x}_u \rangle = F_v - \tfrac{1}{2}G_u, \qquad \Gamma_{22}^1 F + \Gamma_{22}^2 G = \langle \mathbf{x}_{vv}, \mathbf{x}_v \rangle = \tfrac{1}{2}G_v.$$

Note that for each pair the determinant of the system is $EG - F^2 \neq 0$. Thus, it is possible to solve the above system and *to compute the Christoffel symbols in terms of the coefficients of the first fundamental form, $E$, $F$, $G$, and their derivatives*. We shall not obtain the explicit expressions of the $\Gamma\_{ij}^k$, since it is easier to work in each particular case with the system above.

An important consequence: *All geometric concepts and properties expressed in terms of the Christoffel symbols are invariant under isometries.*

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Christoffel Symbols for a Surface of Revolution)</span></p>

We shall compute the Christoffel symbols for a surface of revolution parametrized by $\mathbf{x}(u, v) = (f(v)\cos u, f(v)\sin u, g(v))$, $f(v) \neq 0$. Since $E = (f(v))^2$, $F = 0$, $G = (f'(v))^2 + (g'(v))^2$, we obtain $E\_u = 0$, $E\_v = 2ff'$, $F\_u = F\_v = 0$, $G\_u = 0$, $G\_v = 2(f'f'' + g'g'')$, where prime denotes derivative with respect to $v$. The first two equations of the system give

$$\Gamma_{11}^1 = 0, \qquad \Gamma_{11}^2 = -\frac{ff'}{(f')^2 + (g')^2}.$$

Next, the second pair of equations yields $\Gamma\_{12}^1 = ff'/ f^2$, $\Gamma\_{12}^2 = 0$. Finally, from the last two equations we obtain $\Gamma\_{22}^1 = 0$, $\Gamma\_{22}^2 = (f'f'' + g'g'')/((f')^2 + (g')^2)$.

</div>

### The Gauss and Mainardi-Codazzi Equations

A way of obtaining relations between the coefficients is to consider the expressions

$$(\mathbf{x}_{uu})_v - (\mathbf{x}_{uv})_u = 0, \qquad (\mathbf{x}_{vv})_u - (\mathbf{x}_{vu})_v = 0, \qquad N_{uv} - N_{vu} = 0.$$

Since the vectors $\mathbf{x}\_u$, $\mathbf{x}\_v$, $N$ are linearly independent, each of these equations yields three relations: $A\_i = 0$, $B\_i = 0$, $C\_i = 0$, $i = 1, 2, 3$, for a total of nine relations.

As an example, we shall determine $A\_1 = 0$, $B\_1 = 0$, $C\_1 = 0$. By equating the coefficients of $\mathbf{x}\_v$ in the first relation, we obtain

$$(\Gamma_{12}^2)_u - (\Gamma_{11}^2)_v + \Gamma_{12}^1\Gamma_{12}^2 + \Gamma_{12}^2\Gamma_{22}^2 - \Gamma_{11}^2\Gamma_{22}^1 - \Gamma_{11}^1\Gamma_{12}^2 = -E\frac{eg - f^2}{EG - F^2} = -EK.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorema Egregium — Gauss)</span></p>

The Gaussian curvature $K$ of a surface is invariant by local isometries.

</div>

In fact if $\mathbf{x}\colon U \subset R^2 \to S$ is a parametrization at $p \in S$ and if $\varphi\colon V \subset S \to \bar{S}$, where $V \subset \mathbf{x}(U)$, is a local isometry at $p$, then $\mathbf{y} = \mathbf{x} \circ \varphi$ is a parametrization of $\bar{S}$ at $\varphi(p)$. Since $\varphi$ is an isometry, the coefficients of the first fundamental form in the parametrizations $\mathbf{x}$ and $\mathbf{y}$ agree at corresponding points $q$ and $\varphi(q)$, $q \in V$; thus, the corresponding Christoffel symbols also agree. By the Gauss formula, $K$ can be computed at a point as a function of the Christoffel symbols in a given parametrization at the point. It follows that $K(q) = K(\varphi(q))$ for all $q \in V$.

The above expression, which yields the value of $K$ in terms of the coefficients of the first fundamental form and its derivatives, is known as the **Gauss formula**. It was first proved by Gauss in a famous paper.

Actually, it is a remarkable fact that a concept such as the Gaussian curvature, the definition of which made essential use of the position of a surface in the space, does not depend on this position but only on the metric structure (first fundamental form) of the surface. It thus makes sense to talk about a geometry of the first fundamental form, which we call intrinsic geometry, since it may be developed without any reference to the space that contains the surface (once the first fundamental form is given).

By equating also the coefficients of $N$, we obtain the **Mainardi-Codazzi equations**:

$$e_v - f_u = e\Gamma_{12}^1 + f(\Gamma_{12}^2 - \Gamma_{11}^1) - g\Gamma_{11}^2,$$

$$f_v - g_u = e\Gamma_{22}^1 + f(\Gamma_{22}^2 - \Gamma_{12}^1) - g\Gamma_{12}^2.$$

The Gauss formula and the Mainardi-Codazzi equations are known under the name of **compatibility equations** of the theory of surfaces.

### The Bonnet Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bonnet)</span></p>

Let $E$, $F$, $G$, $e$, $f$, $g$ be differentiable functions defined in an open set $V \subset R^2$, with $E > 0$ and $G > 0$. Assume that the given functions satisfy formally the Gauss and Mainardi-Codazzi equations and $EG - F^2 > 0$. Then, for every $q \in V$ there exists a neighborhood $U \subset V$ of $q$ and a diffeomorphism $\mathbf{x}\colon U \to \mathbf{x}(U) \subset R^3$ such that the regular surface $\mathbf{x}(U) \subset R^3$ has $E$, $F$, $G$ and $e$, $f$, $g$ as coefficients of the first and second fundamental forms, respectively. Furthermore, if $U$ is connected and if

$$\bar{\mathbf{x}}\colon U \to \bar{\mathbf{x}}(U) \subset R^3$$

is another diffeomorphism satisfying the same conditions, then there exist a translation $T$ and a proper linear orthogonal transformation $\rho$ in $R^3$ such that $\bar{\mathbf{x}} = T \circ \rho \circ \mathbf{x}$.

</div>

A proof of this theorem may be found in the appendix to Chap. 4.

For later use, it is convenient to observe how the Mainardi-Codazzi equations simplify when the coordinate neighborhood contains no umbilical points and the coordinate curves are lines of curvature ($F = 0 = f$). Then, the equations may be written

$$e_v = \frac{E_v}{2}\left(\frac{e}{E} + \frac{g}{G}\right), \qquad g_u = \frac{G_u}{2}\left(\frac{e}{E} + \frac{g}{G}\right).$$

## 4-4. Parallel Transport. Geodesics

We shall now proceed to a systematic exposition of the intrinsic geometry. We shall start with the definition of covariant derivative of a vector field, which is the analogue for surfaces of the usual differentiation of vectors in the plane.

### The Covariant Derivative

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Covariant Derivative)</span></p>

Let $w$ be a differentiable vector field in an open set $U \subset S$ and let $y \in T\_p(S)$. Consider a parametrized curve $\alpha\colon (-\epsilon, \epsilon) \to U$, with $\alpha(0) = p$ and $\alpha'(0) = y$, and let $w(t)$, $t \in (-\epsilon, \epsilon)$, be the restriction of the vector field $w$ to the curve $\alpha$. The vector obtained by the normal projection of $(dw/dt)(0)$ onto the plane $T\_p(S)$ is called the **covariant derivative** at $p$ of the vector field $w$ relative to the vector $y$. This covariant derivative is denoted by $(Dw/dt)(0)$ or $(D\_y w)(p)$.

</div>

The above definition makes use of the normal vector of $S$ and of a particular curve $\alpha$. To show that covariant differentiation is a concept of the intrinsic geometry and that it does not depend on the choice of the curve $\alpha$, we shall obtain its expression in terms of a parametrization $\mathbf{x}(u, v)$ of $S$ in $p$.

Let $\mathbf{x}(u(t), v(t)) = \alpha(t)$ be the expression of the curve $\alpha$ and let $w(t) = a(t)\mathbf{x}\_u + b(t)\mathbf{x}\_v$. Then

$$\frac{dw}{dt} = a(\mathbf{x}_{uu}u' + \mathbf{x}_{uv}v') + b(\mathbf{x}_{vu}u' + \mathbf{x}_{vv}v') + a'\mathbf{x}_u + b'\mathbf{x}_v.$$

Since $Dw/dt$ is the component of $dw/dt$ in the tangent plane, we use the expressions for $\mathbf{x}\_{uu}$, $\mathbf{x}\_{uv}$, $\mathbf{x}\_{vv}$ from Sec. 4-3, and, by dropping the normal component, we obtain

$$\frac{Dw}{dt} = (a' + \Gamma_{11}^1 au' + \Gamma_{12}^1 av' + \Gamma_{12}^1 bu' + \Gamma_{22}^1 bv')\mathbf{x}_u + (b' + \Gamma_{11}^2 au' + \Gamma_{12}^2 av' + \Gamma_{12}^2 bu' + \Gamma_{22}^2 bv')\mathbf{x}_v.$$

This expression shows that $Dw/dt$ depends only on the vector $(u', v') = y$ and not on the curve $\alpha$. Furthermore, the surface makes its appearance in Eq. (1) through the Christoffel symbols, that is, through the first fundamental form. Our assertions are, therefore, proved.

If, in particular, $S$ is a plane, we know that it is possible to find a parametrization in such a way that $E = G = 1$ and $F = 0$. The equations that give the Christoffel symbols show that in this case the $\Gamma\_{ij}^k$ all become zero. It follows from Eq. (1) that the covariant derivative agrees with the usual derivative of vectors in the plane. The covariant derivative is, therefore, a generalization of the usual derivative of vectors in the plane.

### Vector Fields along Curves. Parallel Transport

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vector Field along a Curve)</span></p>

Let $\alpha\colon I \to S$ be a parametrized curve in $S$. A **vector field $w$ along $\alpha$** is a correspondence that assigns to each $t \in I$ a vector $w(t) \in T\_{\alpha(t)}(S)$. The vector field $w$ is **differentiable** at $t\_0 \in I$ if for some parametrization $\mathbf{x}(u, v)$ in $\alpha(t\_0)$ the components $a(t)$, $b(t)$ of $w(t) = a\mathbf{x}\_u + b\mathbf{x}\_v$ are differentiable functions of $t$ at $t\_0$. $w$ is **differentiable** in $I$ if it is differentiable for every $t \in I$.

</div>

An example of a (differentiable) vector field along $\alpha$ is given by the field $\alpha'(t)$ of the tangent vectors of $\alpha$.

The expression (1) of $(Dw/dt)(t)$, $t \in I$, is well defined and is called the **covariant derivative** of $w$ at $t$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Parallel Vector Field)</span></p>

A vector field $w$ along a parametrized curve $\alpha\colon I \to S$ is said to be **parallel** if $Dw/dt = 0$ for every $t \in I$.

</div>

In the particular case of the plane, the notion of parallel field along a parametrized curve reduces to that of a constant field along the curve; that is, the length of the vector and its angle with a fixed direction are constant. Those properties are partially reobtained on any surface as the following proposition shows.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Parallel Fields Preserve Inner Products)</span></p>

Let $w$ and $v$ be parallel vector fields along $\alpha\colon I \to S$. Then $\langle w(t), v(t) \rangle$ is constant. In particular, $\|w(t)\|$ and $\|v(t)\|$ are constant, and the angle between $v(t)$ and $w(t)$ is constant.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

To say that the vector field $w$ is parallel along $\alpha$ means that $dw/dt$ is normal to the plane which is tangent to the surface at $\alpha(t)$; that is, $\langle v(t), w'(t) \rangle = 0$, $t \in I$. On the other hand, $v'(t)$ is also normal to the tangent plane at $\alpha(t)$. Thus,

$$\langle v(t), w(t) \rangle' = \langle v'(t), w(t) \rangle + \langle v(t), w'(t) \rangle = 0;$$

that is, $\langle v(t), w(t) \rangle = \text{constant}$. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Existence and Uniqueness of Parallel Transport)</span></p>

Let $\alpha\colon I \to S$ be a parametrized curve in $S$ and let $w\_0 \in T\_{\alpha(t\_0)}(S)$, $t\_0 \in I$. Then there exists a unique parallel vector field $w(t)$ along $\alpha(t)$, with $w(t\_0) = w\_0$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Parallel Transport)</span></p>

Let $\alpha\colon I \to S$ be a parametrized curve and $w\_0 \in T\_{\alpha(t\_0)}(S)$, $t\_0 \in I$. Let $w$ be the parallel vector field along $\alpha$, with $w(t\_0) = w\_0$. The vector $w(t\_1)$, $t\_1 \in I$, is called the **parallel transport** of $w\_0$ along $\alpha$ at the point $t\_1$.

</div>

It should be remarked that if $\alpha\colon I \to S$, $t \in I$, is regular, then the parallel transport does not depend on the parametrization of $\alpha(I)$. By Prop. 1, $P\_\alpha\colon T\_p(S) \to T\_q(S)$ is an isometry.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Parallel Transport on the Sphere)</span></p>

Let $C$ be a parallel of colatitude $\varphi$ of an oriented unit sphere and let $w\_0$ be a unit vector, tangent to $C$ at some point $p$ of $C$. Let us determine the parallel transport of $w\_0$ along $C$, parametrized by arc length $s$, with $s = 0$ at $p$.

Consider the cone which is tangent to the sphere along $C$. The angle $\psi$ at the vertex of this cone is given by $\psi = (\pi/2) - \varphi$. By the property that two surfaces tangent along a curve have the same parallel transport along that curve, the problem reduces to the determination of the parallel transport of $w\_0$ along $C$, relative to the tangent cone.

The cone minus one generator is, however, isometric to an open set $U \subset R^2$, given in polar coordinates $(\rho, \theta)$ by $0 < \rho < \infty$, $0 < \theta < 2\pi\sin\psi$. Since in the plane the parallel transport coincides with the usual notion, we obtain, for a displacement $s$ of $\rho$, corresponding to the central angle $\theta$, that the oriented angle formed by the tangent vector $t(s)$ with the parallel transport $w(s)$ is given by $2\pi - \theta$.

</div>

### Geodesics

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Geodesic — Parametrized Curve)</span></p>

A nonconstant, parametrized curve $\gamma\colon I \to S$ is said to be **geodesic** at $t \in I$ if the field of its tangent vectors $\gamma'(t)$ is parallel along $\gamma$ at $t$; that is,

$$\frac{D\gamma'(t)}{dt} = 0.$$

$\gamma$ is a **parametrized geodesic** if it is geodesic for all $t \in I$.

</div>

By Prop. 1, we obtain immediately that $\|\gamma'(t)\| = \text{const.} = c \neq 0$. Therefore, we may introduce the arc length $s = ct$ as a parameter, and we conclude that the parameter $t$ of a parametrized geodesic $\gamma$ is proportional to the arc length of $\gamma$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Geodesic — Regular Curve)</span></p>

A regular connected curve $C$ in $S$ is said to be a **geodesic** if, for every $p \in S$, the parametrization $\alpha(s)$ of a coordinate neighborhood of $p$ by the arc length $s$ is a parametrized geodesic; that is, $\alpha'(s)$ is a parallel vector field along $\alpha(s)$.

</div>

Observe that every straight line contained in a surface satisfies the geodesic condition. From a point of view exterior to the surface, saying that $\alpha''(s) = kn$ is normal to the tangent plane, that is, parallel to the normal to the surface, is equivalent to saying that the principal normal at each point $p \in C$ is parallel to the normal to $S$ at $p$. In other words, a regular curve $C \subset S$ ($k \neq 0$) is a geodesic if and only if its principal normal at each point $p \in C$ is parallel to the normal to $S$ at $p$.

### Examples of Geodesics

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Geodesics of the Sphere)</span></p>

The great circles of a sphere $S^2$ are geodesics. Indeed, the great circles $C$ are obtained by intersecting the sphere with a plane that passes through the center $O$ of the sphere. The principal normal at a point $p \in C$ lies in the direction of the line that connects $p$ to $O$ because $C$ is a circle of center $O$. Since $S^2$ is a sphere, the normal lies in the same direction, which verifies our assertion.

Later in this section we shall prove the general fact that for each point $p \in S$ and each direction in $T\_p(S)$ there exists exactly one geodesic $C \subset S$ passing through $p$ and tangent to this direction. For the case of the sphere, through each point and to each direction there passes exactly one great circle, which, as we proved before, is a geodesic. Therefore, by uniqueness, the great circles are the only geodesics of a sphere.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Geodesics of the Cylinder)</span></p>

For the right circular cylinder over the circle $x^2 + y^2 = 1$, the circles obtained by the intersection of the cylinder with planes that are normal to the axis of the cylinder are geodesics. That is so because the principal normal to any of its points is parallel to the normal to the surface at this point.

On the other hand, by the observation after Def. 8a, the straight lines of the cylinder (generators) are also geodesics.

To verify the existence of other geodesics on the cylinder $C$, we shall consider a parametrization $\mathbf{x}(u, v) = (\cos u, \sin u, v)$ of the cylinder in a neighborhood of $p$ in $C$, with $\mathbf{x}(0, 0) = p$. Since the cylinder is a local isometry (cf. Example 1, Sec. 4-2) which maps a neighborhood $U$ of $(0, 0)$ of the $uv$ plane into the cylinder, the geodesic of the plane are the straight lines. Therefore, excluding the cases already obtained,

$$u(s) = as, \qquad v(s) = bs, \qquad a^2 + b^2 = 1.$$

It follows that when a regular curve $C$ (which is neither a circle nor a line) is a geodesic of the cylinder it is locally of the form

$$(\cos as, \sin as, bs),$$

and thus it is a helix. In this way, all the geodesics of a right circular cylinder are determined.

</div>

### The Geodesic Curvature

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Algebraic Value of the Covariant Derivative)</span></p>

Let $w$ be a differentiable field of unit vectors along a parametrized curve $\alpha\colon I \to S$ on an oriented surface $S$. Since $w(t)$, $t \in I$, is a unit vector field, $(dw/dt)(t)$ is normal to $w(t)$, and therefore

$$\frac{Dw}{dt} = \lambda(N \wedge w(t)).$$

The real number $\lambda = \lambda(t)$, denoted by $[Dw/dt]$, is called the **algebraic value** of the covariant derivative of $w$ at $t$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Geodesic Curvature)</span></p>

Let $C$ be an oriented regular curve contained on an oriented surface $S$, and let $\alpha(s)$ be a parametrization of $C$, in a neighborhood of $p \in S$, by the arc length $s$. The algebraic value of the covariant derivative $[D\alpha'(s)/ds] = k\_g$ of $\alpha'(s)$ at $p$ is called the **geodesic curvature** of $C$ at $p$.

</div>

The geodesics which are regular curves are thus characterized as curves whose geodesic curvature is zero. From a point of view external to the surface, the absolute value of the geodesic curvature $k\_g$ at $p$ is the absolute value of the tangential component of the vector $\alpha''(s) = kn$, where $k$ is the curvature of $C$ at $p$ and $n$ is the normal vector of $C$ at $p$. Recalling that the absolute value of the normal component of $kn$ is the absolute value of the normal curvature $k\_n$ of $C \subset S$ in $p$, we have immediately

$$k^2 = k_g^2 + k_n^2.$$

The geodesic curvature $k\_g$ of $C \subset S$ changes sign when we change the orientation of either $C$ or $S$.

The geodesic curvature is the rate of change of the angle that the tangent to the curve makes with a parallel direction along the curve. More precisely, if $w = \alpha'(s)$ and $v(s)$ is a parallel field along $\alpha(s)$, then by taking $\varphi$ as a determination of the angle from $v$ to $w$,

$$k_g(s) = \left[\frac{D\alpha'(s)}{ds}\right] = \frac{d\varphi}{ds}.$$

### Computing the Covariant Derivative and Geodesic Curvature

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Covariant Derivative in Orthogonal Coordinates)</span></p>

Let $\mathbf{x}(u, v)$ be an orthogonal parametrization (that is, $F = 0$) of a neighborhood of an oriented surface $S$, and $w(t)$ be a differentiable field of unit vectors along the curve $\mathbf{x}(u(t), v(t))$. Then

$$\left[\frac{Dw}{dt}\right] = \frac{1}{2\sqrt{EG}}\left\lbrace G_u \frac{dv}{dt} - E_v \frac{du}{dt} \right\rbrace + \frac{d\varphi}{dt},$$

where $\varphi(t)$ is the angle from $\mathbf{x}\_u$ to $w(t)$ in the given orientation.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4</span><span class="math-callout__name">(Liouville's Formula)</span></p>

Let $\alpha(s)$ be a parametrization by arc length of a regular oriented curve $C$ on an oriented surface $S$. Let $\mathbf{x}(u, v)$ be an orthogonal parametrization of $S$ in $p$ and $\varphi(s)$ be the angle that $\alpha'(s)$ makes with $\mathbf{x}\_u$ in the given orientation. Then

$$k_g = (k_g)_1\cos\varphi + (k_g)_2\sin\varphi + \frac{d\varphi}{ds},$$

where $(k\_g)\_1$ and $(k\_g)\_2$ are the geodesic curvatures of the coordinate curves $v = \text{const.}$ and $u = \text{const.}$, respectively.

</div>

In an orthogonal parametrization the geodesic curvatures of the coordinate curves are

$$(k_g)_1 = -\frac{E_v}{2E\sqrt{G}}, \qquad (k_g)_2 = \frac{G_u}{2G\sqrt{E}}.$$

### Differential Equations of the Geodesics

The equations of a geodesic in a coordinate neighborhood can be obtained from the covariant derivative formula. Let $\gamma\colon I \to S$ be a parametrized curve of $S$ and let $\mathbf{x}(u(t), v(t))$, $t \in I$, be a parametrization of $S$ in a neighborhood $V$ of $\gamma(t\_0)$. The tangent vector field $\gamma'(t) = u'(t)\mathbf{x}\_u + v'(t)\mathbf{x}\_v$. Setting $a = u'$ and $b = v'$ in Eq. (1) and equating to zero the coefficients of $\mathbf{x}\_u$ and $\mathbf{x}\_v$, we obtain the **differential equations of the geodesics**:

$$u'' + \Gamma_{11}^1(u')^2 + 2\Gamma_{12}^1 u'v' + \Gamma_{22}^1(v')^2 = 0,$$

$$v'' + \Gamma_{11}^2(u')^2 + 2\Gamma_{12}^2 u'v' + \Gamma_{22}^2(v')^2 = 0.$$

In other words, $\gamma\colon I \to S$ is a geodesic if and only if the above system is satisfied for every interval $J \subset I$ such that $\gamma(J)$ is contained in a coordinate neighborhood.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5</span><span class="math-callout__name">(Local Existence and Uniqueness of Geodesics)</span></p>

Given a point $p \in S$ and a vector $w \in T\_p(S)$, $w \neq 0$, there exist an $\epsilon > 0$ and a unique parametrized geodesic $\gamma\colon (-\epsilon, \epsilon) \to S$ such that $\gamma(0) = p$, $\gamma'(0) = w$.

</div>

### Geodesics on Surfaces of Revolution

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Geodesics of a Surface of Revolution)</span></p>

We shall use the differential equations of geodesics to study locally the geodesics of a surface of revolution parametrized by $\mathbf{x}(u, v) = (f(v)\cos u, f(v)\sin u, g(v))$, $f(v) \neq 0$. Using the Christoffel symbols computed in Example 1 of Sec. 4-3, the geodesic equations become

$$u'' + \frac{2ff'}{f^2}u'v' = 0,$$

$$v'' - \frac{ff'}{(f')^2 + (g')^2}(u')^2 + \frac{f'f'' + g'g''}{(f')^2 + (g')^2}(v')^2 = 0.$$

First, as expected, the meridians $u = \text{const.}$, $v = v(s)$, parametrized by arc length $s$, are geodesics. Indeed, the first equation is trivially satisfied by $u = \text{const.}$, and since the first fundamental form along the meridian gives $((f')^2 + (g')^2)(v')^2 = 1$, we obtain $(v')^2 = 1/((f')^2 + (g')^2)$, which after differentiation gives the second equation.

Now we are going to determine which parallels $v = \text{const.}$, $u = u(s)$, parametrized by arc length, are geodesics. The first of the equations gives $ff'(u')^2/((f')^2 + (g')^2) = 0$. In order that the parallel $v = \text{const.}$ be a geodesic, it is necessary that $u' \neq 0$. Since $(f')^2 + (g')^2 \neq 0$ and $f \neq 0$, we conclude from the above equation that $f' = 0$.

In other words, a necessary condition for a parallel of a surface of revolution to be a geodesic is that such a parallel be generated by the rotation of a point of the generating curve where the tangent is parallel to the axis of revolution.

</div>

### Clairaut's Relation

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Clairaut's Relation)</span></p>

Let $\gamma$ be a geodesic on a surface of revolution $S$ parametrized by $\mathbf{x}(u, v) = (f(v)\cos u, f(v)\sin u, g(v))$. Let $\theta$, $0 \le \theta \le \pi/2$, be the angle at which $\gamma$ intersects a parallel, and let $r = f$ be the radius of the parallel at the intersection point. Then

$$r\cos\theta = |c| = \text{const.}$$

</div>

This relation follows from the first geodesic equation, which can be rewritten as $(f^2 u')' = 0$, hence $f^2 u' = c = \text{const.}$ Since $\cos\theta = \|\langle \mathbf{x}\_u, \mathbf{x}\_u u' + \mathbf{x}\_v v' \rangle\|/\|\mathbf{x}\_u\| = \|fu'\|$, we obtain $r\cos\theta = \|c\|$.

The geodesic equations can also be integrated by means of primitives. Let $u = u(s)$, $v = v(s)$ be a geodesic parametrized by arc length, which we shall assume not to be a meridian or a parallel of the surface. The first equation gives $f^2 u' = \text{const.} = c \neq 0$. The first fundamental form along $(u(s), v(s))$ gives

$$1 = f^2\left(\frac{du}{ds}\right)^2 + ((f')^2 + (g')^2)\left(\frac{dv}{ds}\right)^2,$$

together with the geodesic equations, is equivalent to the second of the geodesic equations. By substituting $f^2 u' = c$ in the above, we obtain

$$\left(\frac{dv}{ds}\right)^2 = \frac{f^2 - c^2}{f^2((f')^2 + (g')^2)},$$

and by writing $du/dv = u'/v'$,

$$\frac{du}{dv} = \frac{c\sqrt{(f')^2 + (g')^2}}{f\sqrt{f^2 - c^2}},$$

which, by integration, gives $u$ as a function of $v$; that is, $u = u(v)$.

## 4-5. The Gauss-Bonnet Theorem and Its Applications

In this section, we shall present the Gauss-Bonnet theorem and some of its consequences. The geometry involved in this theorem is fairly simple, and the difficulty of its proof lies in certain topological facts. These facts will be presented without proofs.

The Gauss-Bonnet theorem is probably the deepest theorem in the differential geometry of surfaces. A first version was presented by Gauss in a famous paper and deals with geodesic triangles on surfaces (that is, triangles whose sides are arcs of geodesics). Roughly speaking, it asserts that the excess over $\pi$ of the sum of the interior angles $\varphi\_1$, $\varphi\_2$, $\varphi\_3$ of a geodesic triangle $T$ is equal to the integral of the Gaussian curvature $K$ over $T$; that is,

$$\sum_{i=1}^3 \varphi_i - \pi = \iint_T K\,d\sigma.$$

For instance, if $K \equiv 0$, we obtain that $\sum \varphi\_i = \pi$, an extension of Thales' theorem to surfaces of zero curvature. Also, if $K \equiv 1$, we obtain $\sum \varphi\_i - \pi = \text{area}(T) > 0$. Thus, on a unit sphere, the sum of the interior angles of any geodesic triangle is greater than $\pi$.

### Simple Closed Piecewise Regular Curves

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Simple Closed Piecewise Regular Curve)</span></p>

Let $\alpha\colon [0, l] \to S$ be a continuous map from the closed interval $[0, l]$ into the regular surface $S$. We say that $\alpha$ is a **simple, closed, piecewise regular, parametrized curve** if

1. $\alpha(0) = \alpha(l)$.
2. $t\_1 \neq t\_2$, $t\_1, t\_2 \in [0, l)$, implies that $\alpha(t\_1) \neq \alpha(t\_2)$.
3. There exists a subdivision $0 = t\_0 < t\_1 < \cdots < t\_k < t\_{k+1} = l$ of $[0, l]$ such that $\alpha$ is differentiable and regular in each $[t\_i, t\_{i+1}]$, $i = 0, \dots, k$.

</div>

The points $\alpha(t\_i)$, $i = 0, \dots, k$, are called the **vertices** of $\alpha$ and the traces $\alpha([t\_i, t\_{i+1}])$ are called the **regular arcs** of $\alpha$. It is usual to call the trace $\alpha([0, l])$ of $\alpha$ a **closed piecewise regular curve**.

For each vertex $\alpha(t\_i)$ there exist the limits $\alpha'(t\_i - 0) \neq 0$ and $\alpha'(t\_i + 0) \neq 0$. Let $\|\theta\_i\|$, $0 < \|\theta\_i\| \le \pi$, be the smallest determination of the angle from $\alpha'(t\_i - 0)$ to $\alpha'(t\_i + 0)$. The signed angle $\theta\_i$, $-\pi < \theta\_i < \pi$, is called the **external angle** at the vertex $\alpha(t\_i)$.

### The Theorem of Turning Tangents

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem of Turning Tangents)</span></p>

Let $\mathbf{x}\colon U \to S$ be a parametrization compatible with the orientation of $S$, where $U$ is homeomorphic to an open disk in the plane. Let $\alpha\colon [0, l] \to \mathbf{x}(U)$ be a simple closed, piecewise regular, parametrized curve, with vertices $\alpha(s\_0), \dots, \alpha(s\_k)$ and external angles $\theta\_0, \dots, \theta\_k$. Let $\varphi\_i\colon [t\_i, t\_{i+1}] \to R$ be differentiable functions which measure at each $t \in [t\_i, t\_{i+1}]$ the positive angle from $\mathbf{x}\_u$ to $\alpha'(t)$. Then

$$\sum_{i=0}^{k} (\varphi_i(t_{i+1}) - \varphi_i(t_i)) + \sum_{i=0}^{k} \theta_i = \pm 2\pi,$$

where the sign plus or minus depends on the orientation of $\alpha$.

</div>

The theorem states that the total variation of the angle of the tangent vector to $\alpha$ with a given direction plus the "jumps" at the vertices is equal to $2\pi$.

### Simple Regions and the Integral of a Function

Let $S$ be an oriented surface. A region $R \subset S$ (union of a connected open set with its boundary) is called a **simple region** if $R$ is homeomorphic to a disk and the boundary $\partial R$ of $R$ is the trace of a simple, closed, piecewise regular, parametrized curve $\alpha\colon I \to S$. We say that $\alpha$ is **positively oriented** if for each $\alpha(t)$, belonging to a regular arc, the positive orthonormal basis $\lbrace \alpha'(t), h(t) \rbrace$ satisfies the condition that $h(t)$ "points toward" $R$.

If $f$ is a differentiable function on $S$, the **integral of $f$ over the regular region** $R$ is defined by

$$\iint_R f\,d\sigma,$$

which does not depend on the parametrization $\mathbf{x}$, chosen in the class of orientation of $S$.

### The Local Gauss-Bonnet Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Gauss-Bonnet — Local)</span></p>

Let $\mathbf{x}\colon U \to S$ be an orthogonal parametrization (that is, $F = 0$) of an oriented surface $S$, where $U \subset R^2$ is homeomorphic to an open disk and $\mathbf{x}$ is compatible with the orientation of $S$. Let $R \subset \mathbf{x}(U)$ be a simple region and let $\alpha\colon I \to S$ be such that $\partial R = \alpha(I)$. Assume that $\alpha$ is positively oriented, parametrized by arc length $s$, and let $\alpha(s\_0), \dots, \alpha(s\_k)$ and $\theta\_0, \dots, \theta\_k$ be, respectively, the vertices and the external angles of $\alpha$. Then

$$\sum_{i=0}^{k} \int_{s_i}^{s_{i+1}} k_g(s)\,ds + \iint_R K\,d\sigma + \sum_{i=0}^{k} \theta_i = 2\pi,$$

where $k\_g(s)$ is the geodesic curvature of the regular arcs of $\alpha$ and $K$ is the Gaussian curvature of $S$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $u = u(s)$, $v = v(s)$ be the expression of $\alpha$ in the parametrization $\mathbf{x}$. By using Prop. 3 of Sec. 4-4, we have

$$k_g(s) = \frac{1}{2\sqrt{EG}}\left\lbrace G_u \frac{dv}{ds} - E_v \frac{du}{ds} \right\rbrace + \frac{d\varphi_i}{ds},$$

where $\varphi\_i = \varphi\_i(s)$ is a differentiable function which measures the positive angle from $\mathbf{x}\_u$ to $\alpha'(s)$ in $[s\_i, s\_{i+1}]$. By integrating the above expression in every interval $[s\_i, s\_{i+1}]$ and adding up the results,

$$\sum_{i=0}^{k} \int_{s_i}^{s_{i+1}} k_g(s)\,ds = \sum_{i=0}^{k} \int_{s_i}^{s_{i+1}} \left(\frac{G_u}{2\sqrt{EG}} \frac{dv}{ds} - \frac{E_v}{2\sqrt{EG}} \frac{du}{ds}\right) ds + \sum_{i=0}^{k} \int_{s_i}^{s_{i+1}} \frac{d\varphi_i}{ds}\,ds.$$

From the Gauss formula for $F = 0$ (cf. Exercise 1, Sec. 4-3), we know that

$$\iint_{\mathbf{x}^{-1}(R)} \left\lbrace \left(\frac{E_v}{2\sqrt{EG}}\right)_v + \left(\frac{G_u}{2\sqrt{EG}}\right)_u \right\rbrace du\,dv = -\iint_{\mathbf{x}^{-1}(R)} K\sqrt{EG}\,du\,dv = -\iint_R K\,d\sigma.$$

Now we use the Gauss-Green theorem in the $uv$ plane. It follows that

$$\sum_{i=0}^{k} \int_{s_i}^{s_{i+1}} k_g(s)\,ds = -\iint_R K\,d\sigma + \sum_{i=0}^{k} \int_{s_i}^{s_{i+1}} \frac{d\varphi_i}{ds}\,ds.$$

On the other hand, by the theorem of turning tangents,

$$\sum_{i=0}^{k} \int_{s_i}^{s_{i+1}} \frac{d\varphi_i}{ds}\,ds = \sum_{i=0}^{k} (\varphi_i(s_{i+1}) - \varphi_i(s_i)) = \pm 2\pi - \sum_{i=0}^{k} \theta_i.$$

Since the curve $\alpha$ is positively oriented, the sign should be plus. By putting these facts together, we obtain

$$\sum_{i=0}^{k} \int_{s_i}^{s_{i+1}} k_g(s)\,ds + \iint_R K\,d\sigma + \sum_{i=0}^{k} \theta_i = 2\pi. \qquad \textbf{Q.E.D.}$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation of $K$ via Parallel Transport)</span></p>

The local Gauss-Bonnet theorem gives an interpretation of $K$ in terms of parallel transport. Let $R \subset \mathbf{x}(U)$ be a simple region without vertices, containing $p$ in its interior, and let $\alpha\colon [0, l] \to S$ be a curve parametrized by arc length $s$ such that the trace of $\alpha$ is the boundary of $R$. Let $w\_0$ be a unit vector tangent to $S$ at $\alpha(0)$ and let $w(s)$, $s \in [0, l]$, be the parallel transport of $w\_0$ along $\alpha$. If $\varphi = \varphi(s)$ is a differentiable determination of the angle from $\mathbf{x}\_u$ to $w(s)$, then $\Delta\varphi = \varphi(l) - \varphi(0)$ is given by

$$\Delta\varphi = \iint_R K\,d\sigma.$$

Now, $\Delta\varphi$ does not depend on the choice of $w\_0$, and it follows that $\Delta\varphi$ does not depend on the choice of $\alpha(0)$ either. By taking the limit (in the sense of Prop. 2, Sec. 3-3)

$$\lim_{R \to p} \frac{\Delta\varphi}{A(R)} = K(p),$$

where $A(R)$ denotes the area of the region $R$, we obtain the desired interpretation of $K$.

</div>

### Topological Preliminaries

To globalize the Gauss-Bonnet theorem, we need some topological preliminaries.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Regular Region, Triangle, Triangulation)</span></p>

Let $S$ be a regular surface. A region $R \subset S$ is said to be **regular** if $R$ is compact and its boundary $\partial R$ is the finite union of (simple) closed piecewise regular curves which do not intersect. For convenience, we shall consider a compact surface as a regular region, the boundary of which is empty. A simple region which has only three vertices with external angles $\alpha\_i \neq 0$, $i = 1, 2, 3$, is called a **triangle**.

A **triangulation** of a regular region $R \subset S$ is a finite family $\mathfrak{T}$ of triangles $T\_i$, $i = 1, \dots, n$, such that

1. $\bigcup\_{i=1}^n T\_i = R$.
2. If $T\_i \cap T\_j \neq \emptyset$, then $T\_i \cap T\_j$ is either a common edge of $T\_i$ and $T\_j$ or a common vertex of $T\_i$ and $T\_j$.

</div>

### The Euler-Poincaré Characteristic

Given a triangulation $\mathfrak{T}$ of a regular region $R \subset S$, we shall denote by $F$ the number of triangles (faces), by $E$ the number of sides (edges), and by $V$ the number of vertices of the triangulation.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Euler-Poincaré Characteristic)</span></p>

The number

$$\chi(R) = F - E + V$$

is called the **Euler-Poincaré characteristic** of the triangulation $\mathfrak{T}$.

</div>

The Euler-Poincaré characteristic is a topological invariant of the regular region $R$:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Existence of Triangulations)</span></p>

Every regular region of a regular surface admits a triangulation.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Invariance of $\chi$)</span></p>

If $R \subset S$ is a regular region of a surface $S$, the Euler-Poincaré characteristic $\chi(R)$ does not depend on the triangulation of $R$. It is convenient, therefore, to denote it by $\chi(R)$.

</div>

By direct computation: the Euler-Poincaré characteristic of the sphere is 2, that of the torus (sphere with one handle) is 0, that of the double torus (sphere with two handles) is $-2$, and, in general, that of the $n$-torus (sphere with $n$ handles) is $-2(n - 1)$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4</span><span class="math-callout__name">(Classification of Compact Surfaces)</span></p>

Let $S \subset R^3$ be a compact connected surface; then one of the values $2, 0, -2, \dots, -2n, \dots$ is assumed by the Euler-Poincaré characteristic $\chi(S)$. Furthermore, if $S' \subset R^3$ is another compact surface and $\chi(S) = \chi(S')$, then $S$ is homeomorphic to $S'$.

</div>

In other words, every compact connected surface $S \subset R^3$ is homeomorphic to a sphere with a certain number $g$ of handles. The number $g = (2 - \chi(S))/2$ is called the **genus** of $S$.

### The Global Gauss-Bonnet Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Gauss-Bonnet — Global)</span></p>

Let $R \subset S$ be a regular region of an oriented surface and let $C\_1, \dots, C\_n$ be the closed, simple, piecewise regular curves which form the boundary $\partial R$ of $R$. Suppose that each $C\_i$ is positively oriented and let $\theta\_1, \dots, \theta\_p$ be the set of all external angles of all curves $C\_1, \dots, C\_n$. Then

$$\sum_{i=1}^{n} \int_{C_i} k_g(s)\,ds + \iint_R K\,d\sigma + \sum_{l=1}^{p} \theta_l = 2\pi\chi(R),$$

where $s$ denotes the arc length of $C\_i$, and the integral over $C\_i$ means the sum of integrals in every regular arc of $C\_i$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Sketch</summary>

Consider a triangulation $\mathfrak{T}$ of the region $R$ such that every triangle $T \in \mathfrak{T}$ is contained in a coordinate neighborhood of a family of orthogonal parametrizations compatible with the orientation of $S$. Furthermore, if the boundary of every triangle of $\mathfrak{T}$ is positively oriented, adjacent triangles determine opposite orientations in the edges which are common to adjacent triangles.

By applying the local Gauss-Bonnet theorem to every triangle $T \in \mathfrak{T}$, and adding up the results (using the fact that each "interior" side is described twice in opposite orientations),

$$\sum_i \int_{C_i} k_g(s)\,ds + \iint_R K\,d\sigma + \sum_{j,k} \theta_{jk} = 2\pi F,$$

where $F$ denotes the number of triangles of $\mathfrak{T}$, and $\theta\_{j1}$, $\theta\_{j2}$, $\theta\_{j3}$ are the external angles of the triangle $T\_j$.

We now introduce the interior angles of the triangle $T\_j$, given by $\varphi\_{jk} = \pi - \theta\_{jk}$. Using the notation $E\_e$, $E\_i$, $V\_e$, $V\_i$ for the number of external/internal edges and vertices, we count: $3F = 2E\_i + E\_e$, and $E\_e = V\_e$ since the curves $C\_i$ are closed. After careful bookkeeping, we obtain:

$$\sum_{j,k} \theta_{jk} = 2\pi E - 2\pi V + \sum_l \theta_l$$

and therefore

$$\sum_{i=1}^{n} \int_{C_i} k_g(s)\,ds + \iint_R K\,d\sigma + \sum_{l=1}^{p} \theta_l = 2\pi(F - E + V) = 2\pi\chi(R). \qquad \textbf{Q.E.D.}$$

</details>
</div>

### Corollaries

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 1</span><span class="math-callout__name">(Gauss-Bonnet for Simple Regions)</span></p>

If $R$ is a simple region of $S$, then

$$\sum_{i=0}^{k} \int_{s_i}^{s_{i+1}} k_g(s)\,ds + \iint_R K\,d\sigma + \sum_{i=0}^{k} \theta_i = 2\pi.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2</span><span class="math-callout__name">(Gauss-Bonnet for Compact Surfaces)</span></p>

Let $S$ be an orientable compact surface; then

$$\iint_S K\,d\sigma = 2\pi\chi(S).$$

</div>

Corollary 2 is most striking. We have only to think of all possible shapes of a surface homeomorphic to a sphere to find it very surprising that in each case the curvature function distributes itself in such a way that the "total curvature," i.e., $\iint K\,d\sigma$, is the same for all cases.

### Applications

The following are remarkable consequences of the Gauss-Bonnet theorem.

**Application 1.** *A compact surface of positive curvature is homeomorphic to a sphere.* The Euler-Poincaré characteristic of such a surface is positive and the sphere is the only compact surface of $R^3$ which satisfies this condition.

**Application 2.** *Let $S$ be an orientable surface of negative or zero curvature. Then two geodesics $\gamma\_1$ and $\gamma\_2$ which start from a point $p \in S$ cannot meet again at a point $q \in S$ in such a way that the traces of $\gamma\_1$ and $\gamma\_2$ constitute the boundary of a simple region $R$ of $S$.* Assume the contrary. By the Gauss-Bonnet theorem ($R$ is simple), $\iint\_R K\,d\sigma + \theta\_1 + \theta\_2 = 2\pi$, where $\theta\_1$ and $\theta\_2$ are the external angles of the region $R$. Since the geodesics cannot be mutually tangent, $\theta\_i < \pi$. On the other hand, $K \le 0$, whence the contradiction.

**Application 3.** *Let $S$ be a surface homeomorphic to a cylinder with $K < 0$. Then $S$ has at most one simple closed geodesic.*

**Application 4.** *If there exist two simple closed geodesics $\Gamma\_1$ and $\Gamma\_2$ on a compact surface $S$ of positive curvature, then $\Gamma\_1$ and $\Gamma\_2$ intersect.*

### Geodesic Triangles

**Application 6.** *Let $T$ be a geodesic triangle (that is, the sides of $T$ are geodesics) in an oriented surface $S$. Let $\theta\_1$, $\theta\_2$, $\theta\_3$ be the external angles of $T$ and let $\varphi\_1 = \pi - \theta\_1$, $\varphi\_2 = \pi - \theta\_2$, $\varphi\_3 = \pi - \theta\_3$ be its interior angles.* By the Gauss-Bonnet theorem,

$$\iint_T K\,d\sigma + \sum_{i=1}^3 \theta_i = 2\pi.$$

Thus,

$$\iint_T K\,d\sigma = 2\pi - \sum_{i=1}^3 (\pi - \varphi_i) = -\pi + \sum_{i=1}^3 \varphi_i.$$

It follows that the sum of the interior angles $\sum\_{i=1}^3 \varphi\_i$ of a geodesic triangle is

1. *Equal to $\pi$* if $K = 0$.
2. *Greater than $\pi$* if $K > 0$.
3. *Smaller than $\pi$* if $K < 0$.

Furthermore, the difference $\sum\_{i=1}^3 \varphi\_i - \pi$ (the **excess** of $T$) is given precisely by $\iint\_T K\,d\sigma$. If $K \neq 0$ on $T$, this is the area of the image $N(T)$ of $T$ by the Gauss map $N\colon S \to S^2$. This was the form in which Gauss himself stated his theorem: *The excess of a geodesic triangle $T$ is equal to the area of its spherical image $N(T)$.*

This fact is related to the historical controversy about the possibility of proving Euclid's fifth axiom (the axiom of the parallels). Surfaces of constant negative curvature constitute a (local) model of a geometry where Euclid's axioms hold, except for the fifth.

### The Poincaré Theorem on Indices of Vector Fields

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Index of a Vector Field at an Isolated Singular Point)</span></p>

Let $v$ be a differentiable vector field on an oriented surface $S$. We say that $p \in S$ is a **singular point** of $v$ if $v(p) = 0$. The singular point $p$ is **isolated** if there exists a neighborhood $V$ of $p$ in $S$ such that $v$ has no singular points in $V$ other than $p$.

Let $\mathbf{x}\colon U \to S$ be an orthogonal parametrization at $p = \mathbf{x}(0, 0)$ compatible with the orientation of $S$, and let $\alpha\colon [0, l] \to S$ be a simple, closed, piecewise regular parametrized curve such that $\alpha([0, l]) \subset \mathbf{x}(U)$ is the boundary of a simple region $R$ containing $p$ as its only singular point. Let $v = v(t)$, $t \in [0, l]$, be the restriction of $v$ along $\alpha$, and let $\varphi = \varphi(t)$ be some differentiable determination of the angle from $\mathbf{x}\_u$ to $v(t)$. Since $\alpha$ is closed, there is an integer $I$ defined by

$$2\pi I = \varphi(l) - \varphi(0) = \int_0^l \frac{d\varphi}{dt}\,dt.$$

$I$ is called the **index** of $v$ at $p$.

</div>

The index does not depend on the choices made (the curve $\alpha$, the parametrization $\mathbf{x}$). If $p$ is not a singular point of $v$, the index is then zero.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Poincaré)</span></p>

The sum of the indices of a differentiable vector field $v$ with isolated singular points on a compact oriented surface $S$ is equal to the Euler-Poincaré characteristic of $S$:

$$\sum_i I_i = \frac{1}{2\pi}\iint_S K\,d\sigma = \chi(S).$$

</div>

This is a remarkable result. It implies that $\sum I\_i$ does not depend on $v$ but only on the topology of $S$. For instance, in any surface homeomorphic to a sphere, all vector fields with isolated singularities must have the sum of their indices equal to 2. In particular, no such surface can have a differentiable vector field without singular points (since the torus has $\chi = 0$, the torus is the only compact orientable surface that can have a nonvanishing vector field).

## 4-6. The Exponential Map. Geodesic Polar Coordinates

In this section we shall introduce some special coordinate systems with an eye toward their geometric applications. The natural way of introducing such coordinates is by means of the exponential map, which we shall now describe.

### The Exponential Map

As we learned in Sec. 4-4, Prop. 5, given a point $p$ of a regular surface $S$ and a nonzero vector $v \in T\_p(S)$ there exists a unique parametrized geodesic $\gamma\colon (-\epsilon, \epsilon) \to S$, with $\gamma(0) = p$ and $\gamma'(0) = v$. To indicate the dependence of this geodesic on the vector $v$, it is convenient to denote it by $\gamma(t, v) = \gamma$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1</span><span class="math-callout__name">(Homogeneity of Geodesics)</span></p>

If the geodesic $\gamma(t, v)$ is defined for $t \in (-\epsilon, \epsilon)$, then the geodesic $\gamma(t, \lambda v)$, $\lambda \in R$, $\lambda \neq 0$, is defined for $t \in (-\epsilon/\lambda, \epsilon/\lambda)$, and $\gamma(t, \lambda v) = \gamma(\lambda t, v)$.

</div>

Intuitively, Lemma 1 means that since the speed of a geodesic is constant, we can go over its trace within a prescribed time by adjusting our speed appropriately.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Exponential Map)</span></p>

If $v \in T\_p(S)$, $v \neq 0$, is such that $\gamma(\|v\|, v/\|v\|) = \gamma(1, v)$ is defined, we set

$$\exp_p(v) = \gamma(1, v) \qquad \text{and} \qquad \exp_p(0) = p.$$

</div>

Geometrically, the construction corresponds to laying off (if possible) a length equal to $\|v\|$ along the geodesic that passes through $p$ in the direction of $v$; the point of $S$ thus obtained is denoted by $\exp\_p(v)$.

For example, $\exp\_p(v)$ is defined on the unit sphere $S^2$ for every $v \in T\_p(S^2)$. The points of the circles of radii $\pi, 3\pi, \dots, (2n+1)\pi$ are mapped into the antipodal point $q$ of $p$. The points of the circles of radii $2\pi, 4\pi, \dots, 2n\pi$ are mapped back into $p$.

On the other hand, on the regular surface $C$ formed by the one-sheeted cone minus the vertex, $\exp\_p(v)$ is not defined for a vector $v \in T\_p(C)$ in the direction of the meridian that connects $p$ to the vertex, when $\|v\| \ge d$ and $d$ is the distance from $p$ to the vertex.

The important point is that $\exp\_p$ is always defined and differentiable in some neighborhood of $p$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Local Existence of the Exponential Map)</span></p>

Given $p \in S$ there exists an $\epsilon > 0$ such that $\exp\_p$ is defined and differentiable in the interior $B\_\epsilon$ of a disk of radius $\epsilon$ of $T\_p(S)$, with center in the origin.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">($\exp\_p$ is a Local Diffeomorphism)</span></p>

$\exp\_p\colon B\_\epsilon \subset T\_p(S) \to S$ is a diffeomorphism in a neighborhood $U \subset B\_\epsilon$ of the origin $0$ of $T\_p(S)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We shall show that the differential $d(\exp\_p)$ is nonsingular at $0 \in T\_p(S)$. To do this, we identify the space of tangent vectors to $T\_p(S)$ at $0$ with $T\_p(S)$ itself. Consider the curve $\alpha(t) = tv$, $v \in T\_p(S)$. It is obvious that $\alpha(0) = 0$ and $\alpha'(0) = v$. The curve $(\exp\_p \circ \alpha)(t) = \exp\_p(tv)$ has at $t = 0$ the tangent vector

$$\frac{d}{dt}(\exp_p(tv))\bigg|_{t=0} = \frac{d}{dt}(\gamma(t, v))\bigg|_{t=0} = v.$$

It follows that $(d\exp\_p)\_0(v) = v$, which shows that $d\exp\_p$ is nonsingular at 0. By applying the inverse function theorem (cf. Prop. 3, Sec. 2-4), we complete the proof of the proposition. **Q.E.D.**

</details>
</div>

### Normal Neighborhood and Coordinate Systems

It is convenient to call $V = \exp\_p(U)$ a **normal neighborhood** of $p \in S$ if $V$ is the image $\exp\_p(U)$ of a neighborhood $U$ of the origin of $T\_p(S)$ restricted to which $\exp\_p$ is a diffeomorphism.

Since the exponential map at $p \in S$ is a diffeomorphism on $U$, it may be used to introduce coordinates in $V$. Among the coordinate systems thus introduced, the most usual are:

1. The **normal coordinates** which correspond to a system of rectangular coordinates in the tangent plane $T\_p(S)$.
2. The **geodesic polar coordinates** which correspond to polar coordinates in the tangent plane $T\_p(S)$.

### Normal Coordinates

We choose in the plane $T\_p(S)$, $p \in S$, two orthogonal unit vectors $e\_1$ and $e\_2$. Since $\exp\_p\colon U \to V \subset S$ is a diffeomorphism, it satisfies the conditions for a parametrization in $p$. If $q \in V$, then $q = \exp\_p(w)$, where $w = ue\_1 + ve\_2 \in U$, and we say that $q$ has **normal coordinates** $(u, v)$.

In a system of normal coordinates centered in $p$, the geodesics through $p$ are the images by $\exp\_p$ of the lines $u = at$, $v = bt$ which pass through the origin of $T\_p(S)$. Observe also that at $p$ the coefficients of the first fundamental form are given by $E(p) = G(p) = 1$, $F(p) = 0$.

### Geodesic Polar Coordinates

Choose in the plane $T\_p(S)$, $p \in S$, a system of polar coordinates $(\rho, \theta)$, where $\rho$ is the polar radius and $\theta$, $0 < \theta < 2\pi$, is the polar angle. Since the polar coordinates are not defined in the closed half-line $l$ which corresponds to $\theta = 0$, set $\exp\_p(l) = L$. Since $\exp\_p\colon U - l \to V - L$ is still a diffeomorphism, we may parametrize the points of $V - L$ by the coordinates $(\rho, \theta)$, which are called **geodesic polar coordinates**.

The images by $\exp\_p$ of the circles in $U$ centered in $0$ will be called **geodesic circles** of $V$, and the images of the lines through $0$ will be called **radial geodesics** of $V$. In $V - L$ these are the curves $\rho = \text{const.}$ and $\theta = \text{const.}$, respectively.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(First Fundamental Form in Geodesic Polar Coordinates)</span></p>

Let $\mathbf{x}\colon U - l \to V - L$ be a system of geodesic polar coordinates $(\rho, \theta)$. Then the coefficients $E = E(\rho, \theta)$, $F = F(\rho, \theta)$, and $G = G(\rho, \theta)$ of the first fundamental form satisfy the conditions

$$E = 1, \qquad F = 0, \qquad \lim_{\rho \to 0} G = 0, \qquad \lim_{\rho \to 0} (\sqrt{G})_\rho = 1.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By definition of the exponential map, $\rho$ measures the arc length along the curve $\theta = \text{const.}$ It follows immediately that $E = 1$.

By introducing in the differential equation of a geodesic (Eq. (4), Sec. 4-4) the fact that $\theta = \text{const.}$ is a geodesic, we conclude that $\Gamma\_{11}^2 = 0$. By using the first of the relations (2) of Sec. 4-3 that define the Christoffel symbols, we first obtain $0 = \tfrac{1}{2}E\_\rho = \Gamma\_{11}^1 E = \Gamma\_{11}^1$. By introducing in the second of the equations that $\Gamma\_{11}^2 = 0$, we conclude that $F\_\rho = 0$, and, therefore, $F(\rho, \theta)$ does not depend on $\rho$.

For each $q \in V$, we shall denote by $\alpha(\sigma)$ the geodesic circle that passes through $q$, where $\sigma \in [0, 2\pi]$ (if $q = p$, $\alpha(\sigma)$ is the constant curve $\alpha(\sigma) = p$). We shall denote by $\gamma(s)$, where $s$ is the arc length of $\gamma$, the radial geodesic that passes through $q$. With this notation we may write

$$F(\rho, \theta) = \left\langle \frac{d\alpha}{d\sigma},\; \frac{d\gamma}{ds} \right\rangle.$$

The coefficient $F(\rho, \theta)$ is not defined at $p$. However, if we fix the radial geodesic $\theta = \theta\_0$ and let $\rho \to 0$, $\alpha$ approaches the constant curve $p$ and therefore $d\alpha/d\sigma \to 0$. Thus, $\lim\_{\rho \to 0} F = 0$. Together with the fact that $F$ does not depend on $\rho$, this implies that $F = 0$.

To prove the last assertion of the proposition, we choose a system of normal coordinates $(\bar{u}, \bar{v})$ in $p$ in such a way that the change of coordinates is given by $\bar{u} = \rho\cos\theta$, $\bar{v} = \rho\sin\theta$, $\rho \neq 0$, $0 < \theta < 2\pi$. By recalling that

$$\sqrt{EG - F^2} = \sqrt{\bar{E}\bar{G} - \bar{F}^2}\;\frac{\partial(\bar{u}, \bar{v})}{\partial(\rho, \theta)},$$

and that $\bar{E}$, $\bar{F}$, $\bar{G}$ are the coefficients of the first fundamental form in the normal coordinates $(\bar{u}, \bar{v})$, we have

$$\sqrt{G} = \rho\sqrt{\bar{E}\bar{G} - \bar{F}^2}, \qquad \rho \neq 0.$$

Since at $p$, $\bar{E} = \bar{G} = 1$, $\bar{F} = 0$ (the normal coordinates are defined at $p$), we conclude that $\lim\_{\rho \to 0} \sqrt{G} = 0$ and $\lim\_{\rho \to 0} (\sqrt{G})\_\rho = 1$. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gauss Lemma)</span></p>

The geometric meaning of the fact that $F = 0$ is that in a normal neighborhood the family of geodesic circles is orthogonal to the family of radial geodesics. This fact is known as the **Gauss lemma**.

</div>

### Surfaces of Constant Curvature — Minding's Theorem

Since in a polar system $E = 1$ and $F = 0$, the Gaussian curvature $K$ can be written

$$K = -\frac{(\sqrt{G})_{\rho\rho}}{\sqrt{G}}.$$

This expression may be considered as the differential equation which $\sqrt{G}(\rho, \theta)$ should satisfy if we want the surface to have (in the coordinate neighborhood in question) curvature $\bar{K}(\rho, \theta)$. If $\bar{K}$ is constant, the above expression becomes

$$(\sqrt{G})_{\rho\rho} + \bar{K}\sqrt{G} = 0,$$

which is a linear differential equation of second order with constant coefficients.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Minding)</span></p>

Any two regular surfaces with the same constant Gaussian curvature $K$ are locally isometric. More precisely, let $S\_1$, $S\_2$ be two such surfaces. Choose points $p\_1 \in S\_1$, $p\_2 \in S\_2$, and orthonormal bases $\lbrace e\_1, e\_2 \rbrace \in T\_{p\_1}(S\_1)$, $\lbrace f\_1, f\_2 \rbrace \in T\_{p\_2}(S\_2)$. Then there exist neighborhoods $V\_1$ of $p\_1$, $V\_2$ of $p\_2$ and an isometry $\psi\colon V\_1 \to V\_2$ such that $d\psi(e\_1) = f\_1$, $d\psi(e\_2) = f\_2$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let us first consider the equation $(\sqrt{G})\_{\rho\rho} + \bar{K}\sqrt{G} = 0$ and study separately the cases (1) $K = 0$, (2) $K > 0$, and (3) $K < 0$.

1. If $K = 0$, $(\sqrt{G})\_{\rho\rho} = 0$. Thus, $(\sqrt{G})\_\rho = g(\theta)$, where $g(\theta)$ is a function of $\theta$. Since $\lim\_{\rho \to 0} (\sqrt{G})\_\rho = 1$, we conclude that $(\sqrt{G})\_\rho \equiv 1$ and $\sqrt{G} = \rho + f(\theta)$. Since $\lim\_{\rho \to 0} \sqrt{G} = 0$, we finally have, in this case,

$$E = 1, \qquad F = 0, \qquad G(\rho, \theta) = \rho^2.$$

2. If $K > 0$, the general solution of the equation is

$$\sqrt{G} = A(\theta)\cos(\sqrt{K}\,\rho) + B(\theta)\sin(\sqrt{K}\,\rho).$$

Since $\lim \sqrt{G} = 0$, we obtain $A(\theta) = 0$. Since $\lim (\sqrt{G})\_\rho = 1$, we obtain $B(\theta) = 1/\sqrt{K}$. Therefore, in this case,

$$E = 1, \qquad F = 0, \qquad G = \frac{1}{K}\sin^2(\sqrt{K}\,\rho).$$

3. Finally, if $K < 0$, the general solution is

$$\sqrt{G} = A(\theta)\cosh(\sqrt{-K}\,\rho) + B(\theta)\sinh(\sqrt{-K}\,\rho).$$

By using the initial conditions, we verify that in this case

$$E = 1, \qquad F = 0, \qquad G = \frac{1}{-K}\sinh^2(\sqrt{-K}\,\rho).$$

We are now prepared to prove Minding's theorem. Let $V\_1$ and $V\_2$ be normal neighborhoods of $p\_1$ and $p\_2$, respectively. Let $\varphi$ be the linear isometry of $T\_{p\_1}(S\_1)$ onto $T\_{p\_2}(S\_2)$ given by $\varphi(e\_1) = f\_1$, $\varphi(e\_2) = f\_2$. Take a polar coordinate system $(\rho, \theta)$ in $T\_{p\_1}(S\_1)$ with axis $l$ and set $L\_1 = \exp\_{p\_1}(l)$, $L\_2 = \exp\_{p\_2}(\varphi(l))$. Let $\psi$ be defined by

$$\psi = \exp_{p_2} \circ \varphi \circ \exp_{p_1}^{-1}.$$

We claim that $\psi$ is the required isometry. In fact, the restriction $\bar{\psi}$ of $\psi$ to $V\_1 - L\_1$ maps a polar coordinate neighborhood with coordinates $(\rho, \theta)$ centered in $p\_1$ into a polar coordinate neighborhood with coordinates $(\rho, \theta)$ centered in $p\_2$. By the above study of Eq. (2), the coefficients of the first fundamental forms at corresponding points are equal. By Prop. 1 of Sec. 4-2, $\bar{\psi}$ is an isometry. By continuity, $\psi$ still preserves inner products at points of $L\_1$ and thus is an isometry. It is immediate to check that $d\psi(e\_1) = f\_1$, $d\psi(e\_2) = f\_2$, and this concludes the proof. **Q.E.D.**

</details>
</div>

### Geometric Interpretation of $K$ via Geodesic Circles

The expression $K = -(\sqrt{G})\_{\rho\rho}/\sqrt{G}$ in geodesic polar coordinates provides a nice geometric interpretation of the Gaussian curvature $K$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Spreading of Geodesics and the Sign of $K$)</span></p>

If $K < 0$, since $\lim\_{\rho \to 0} (\sqrt{G})\_\rho = 1$ and $(\sqrt{G})\_{\rho\rho} = -K\sqrt{G} > 0$, the function $L(\rho)$ (arc length of a geodesic circle of radius $\rho$) increases with $\rho$; that is, the geodesics $\theta = \theta\_0$ and $\theta = \theta\_1$ get farther and farther apart. On the other hand, if $K > 0$, $L(\rho)$ behaves as in the sphere: the geodesics may (case I) or may not (case II) come closer together after a certain value of $\rho$, and this depends on the Gaussian curvature.

</div>

By expanding $\sqrt{G}$ as a Taylor series in $\rho$ and using the initial conditions, we can compute the arc length $L$ of a geodesic circle $S\_r(p)$ around $p$ of radius $r$:

$$L = \lim_{\epsilon \to 0} \int_{0+\epsilon}^{2\pi - \epsilon} \sqrt{G(r, \theta)}\,d\theta = 2\pi r - \frac{\pi}{3}r^3 K(p) + R_1,$$

where $\lim\_{r \to 0} R\_1/r^3 = 0$. It follows that

$$K(p) = \lim_{r \to 0} \frac{3}{\pi}\frac{2\pi r - L}{r^3},$$

which gives an intrinsic interpretation of $K(p)$ in terms of the radius $r$ of a geodesic circle $S\_r(p)$ around $p$ and the arc lengths $L$ of $S\_r(p)$ and $2\pi r$ of the circle of radius $r$ in the plane (that is, the image $\exp\_p^{-1}(S\_r(p))$).

### Geodesics Minimize Arc Length Locally

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4</span><span class="math-callout__name">(Geodesics Locally Minimize Arc Length)</span></p>

Let $p$ be a point on a surface $S$. Then, there exists a neighborhood $W \subset S$ of $p$ such that if $\gamma\colon I \to W$ is a parametrized geodesic with $\gamma(0) = p$, $\gamma(t\_1) = q$, $t\_1 \in I$, and $\alpha\colon [0, t\_1] \to S$ is a parametrized regular curve joining $p$ to $q$, we have

$$l_\gamma \le l_\alpha,$$

where $l\_\alpha$ denotes the length of the curve $\alpha$. Moreover, if $l\_\gamma = l\_\alpha$, then the trace of $\alpha$ coincides with the trace of $\gamma$ between $p$ and $q$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Sketch</summary>

Let $V$ be a normal neighborhood of $p$, and let $\bar{W}$ be the closed region bounded by a geodesic circle of radius $r$ contained in $V$. Let $(\rho, \theta)$ be geodesic polar coordinates in $\bar{W} - L$ centered in $p$ such that $q \in L$.

Suppose first that $\alpha((0, t\_1)) \subset \bar{W} - L$, and set $\alpha(t) = (\rho(t), \theta(t))$. Observe initially that

$$\sqrt{(\rho')^2 + G(\theta')^2} \ge \sqrt{(\rho')^2},$$

and equality holds if and only if $\theta' \equiv 0$; that is, $\theta = \text{const.}$ Therefore, the length $l\_\alpha(\epsilon)$ of $\alpha$ between $\epsilon$ and $t\_1 - \epsilon$ satisfies

$$l_\alpha(\epsilon) = \int_\epsilon^{t_1-\epsilon} \sqrt{(\rho')^2 + G(\theta')^2}\,dt \ge \int_\epsilon^{t_1-\epsilon} \sqrt{(\rho')^2}\,dt \ge \int_\epsilon^{t_1-\epsilon} \rho'\,dt = l_\gamma - 2\epsilon,$$

and equality holds only if $\alpha$ is the radial geodesic $\theta = \text{const.}$ with a parametrization $\rho = \rho(t)$, where $\rho'(t) > 0$. Making $\epsilon \to 0$, we obtain $l\_\alpha \ge l\_\gamma$ and that equality holds only if $\alpha$ is the radial geodesic from $p$ to $q$. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5</span><span class="math-callout__name">(Shortest Regular Curves are Geodesics)</span></p>

Let $\alpha\colon I \to S$ be a regular parametrized curve with a parameter proportional to arc length. Suppose that the arc length of $\alpha$ between any two points $t$, $\tau \in I$, is smaller than or equal to the arc length of any regular parametrized curve joining $\alpha(t)$ to $\alpha(\tau)$. Then $\alpha$ is a geodesic.

</div>

The previous proposition is not true globally, as is shown by the example of the sphere. Two nonantipodal points of a sphere may be connected by two meridians and only the smaller one satisfies the conclusions of the proposition. In other words, a geodesic, if sufficiently extended, may not be the shortest path between its end points.

## 4-7. Further Properties of Geodesics; Convex Neighborhoods

In this section we shall show how certain facts on geodesics (in particular, Prop. 5 of Sec. 4-4) follow from the general theorem of existence, uniqueness, and dependence on the initial condition of vector fields.

### Geodesics as a System in $R^4$

The geodesics in a parametrization $\mathbf{x}(u, v)$ are given by the system

$$u'' + \Gamma_{11}^1(u')^2 + 2\Gamma_{12}^1 u'v' + \Gamma_{22}^1(v')^2 = 0,$$

$$v'' + \Gamma_{11}^2(u')^2 + 2\Gamma_{12}^2 u'v' + \Gamma_{22}^2(v')^2 = 0,$$

where the $\Gamma\_{ij}^k$ are functions of the local coordinates $u$ and $v$. By setting $u' = \xi$ and $v' = \eta$, we may write the above system in the general form

$$\xi' = F_1(u, v, \xi, \eta), \qquad \eta' = F_2(u, v, \xi, \eta), \qquad u' = F_3(u, v, \xi, \eta), \qquad v' = F_4(u, v, \xi, \eta),$$

where $F\_3 = \xi$ and $F\_4 = \eta$.

It is convenient to use the following notation: $(u, v, \xi, \eta)$ will denote a point of $R^4 = R^2 \times R^2$; $(u, v)$ will denote a point of the first factor and $(\xi, \eta)$ a point of the second factor.

The system above is equivalent to a vector field in an open set of $R^4$, entirely analogous to vector fields in $R^2$ (cf. Sec. 3-4). The theorem of existence and uniqueness of trajectories (Theorem 1, Sec. 3-4) still holds in this case, and is stated as follows.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Existence, Uniqueness, and Smooth Dependence of Geodesics)</span></p>

Given $p \in S$ there exist numbers $\epsilon\_1 > 0$, $\epsilon\_2 > 0$ and a differentiable map

$$\gamma\colon (-\epsilon_2, \epsilon_2) \times B_{\epsilon_1} \to S, \qquad B_{\epsilon_1} \subset T_p(S)$$

such that for $v \in B\_{\epsilon\_1}$, $v \neq 0$, $t \in (-\epsilon\_2, \epsilon\_2)$ the curve $t \mapsto \gamma(t, v)$ is the geodesic of $S$ with $\gamma(0, v) = p$, $\gamma'(0, v) = v$, and for $v = 0$, $\gamma(t, 0) = p$.

</div>

The theorem of the dependence on the initial conditions for the vector field defined by the geodesic equations is also important. It is essentially the same as that for vector fields in $R^2$.

*Given the system in an open set $U \subset R^4$ and given a point $(u\_0, v\_0, \xi\_0, \eta\_0) \in U$, there exist a neighborhood $V = V\_1 \times V\_2$ of $p$ (where $V\_1$ is a neighborhood of $(u\_0, v\_0)$ and $V\_2$ is a neighborhood of $(\xi\_0, \eta\_0)$), an open interval $I$, and a differentiable mapping $\alpha\colon I \times V\_1 \times V\_2 \to U$ such that, fixed $(q, v) \in V\_1 \times V\_2$, then $\alpha(t, q, v)$, $t \in I$, is the trajectory of the system passing through $(q, v)$.*

### Uniform Neighborhoods

Applying this to a regular surface $S$, we introduce a parametrization $\mathbf{x}(u, v)$ in $p \in S$, with coordinate neighborhood $V$, and identify, as above, the set of pairs $(q, v)$, $q \in V$, $v \in T\_q(S)$, may be identified with an open set $V \times R^2 = U \subset R^4$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1a</span><span class="math-callout__name">(Uniform Geodesic Neighborhood)</span></p>

Given $p \in S$, there exist positive numbers $\epsilon$, $\epsilon\_1$, $\epsilon\_2$ and a differentiable map

$$\gamma\colon (-\epsilon_2, \epsilon_2) \times \mathfrak{U} \to S,$$

where

$$\mathfrak{U} = \lbrace (q, v);\; q \in B_\epsilon(p),\; v \in B_{\epsilon_1}(0) \subset T_q(S) \rbrace,$$

such that $\gamma(t, q, 0) = q$, and for $v \neq 0$ the curve $t \longmapsto \gamma(t, q, v)$, $t \in (-\epsilon\_2, \epsilon\_2)$

is the geodesic of $S$ with $\gamma(0, q, v) = q$, $\gamma'(0, q, v) = v$.

</div>

### Normal Neighborhoods for All Points

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Uniform Normal Neighborhood)</span></p>

Given $p \in S$ there exist a neighborhood $W$ of $p$ in $S$ and a number $\delta > 0$ such that for every $q \in W$, $\exp\_q$ is a diffeomorphism on $B\_\delta(0) \subset T\_q(S)$ and $\exp\_q(B\_\delta(0)) \supset W$; that is, $W$ is a normal neighborhood of all its points.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $V$ be a coordinate neighborhood of $p$. Let $\epsilon$, $\epsilon\_1$, $\epsilon\_2$ and $\gamma\colon (-\epsilon\_2, \epsilon\_2) \times \mathfrak{U} \to V$ be as in Theorem 1a. By choosing $\epsilon\_1 < \epsilon\_2$, we can make sure that, for $(q, v) \in \mathfrak{U}$, $\exp\_q(v) = \gamma(\|v\|, q, v)$ is well defined. Thus, we can define a differentiable map $\varphi\colon \mathfrak{U} \to V \times V$ by

$$\varphi(q, v) = (q, \exp_q(v)).$$

We first show that $d\varphi$ is nonsingular at $(p, 0)$. For that, we investigate how $\varphi$ transforms the curves in $\mathfrak{U}$ given by

$$t \longmapsto (p, tw), \qquad t \longmapsto (\alpha(t), 0),$$

where $w \in T\_p(S)$ and $\alpha(t)$ is a curve in $S$ with $\alpha(0) = p$. Observe that the tangent vectors at $t = 0$ are $(0, w)$ and $(\alpha'(0), 0)$, respectively. Thus,

$$d\varphi_{(p,0)}(0, w) = (0, w), \qquad d\varphi_{(p,0)}(\alpha'(0), 0) = (\alpha'(0), \alpha'(0)),$$

and $d\varphi\_{(p,0)}$ takes linearly independent vectors into linearly independent vectors. Hence, $d\varphi$ is nonsingular at $(p, 0)$. By applying the inverse function theorem, and concluding the existence of the desired neighborhood $W$ and $\delta$. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1</span><span class="math-callout__name">(Unique Geodesic Joining Nearby Points)</span></p>

From Prop. 1, it follows that given two points $q\_1, q\_2 \in W$ there exists a unique geodesic $\gamma$ of length less than $\delta$ joining $q\_1$ and $q\_2$. Furthermore, the proof also shows that $\gamma$ "depends differentiably" on $q\_1$ and $q\_2$ in the following sense: Given $(q\_1, q\_2) \in W \times W$, a unique $v \in T\_{q\_1}(S)$ is determined (precisely, the $v$ given by $\varphi^{-1}(q\_1, q\_2) = (q\_1, v)$) which depends differentiably on $(q\_1, q\_2)$ and is such that $\gamma'(0) = v$.

</div>

### Convex Neighborhoods

A natural question about Prop. 1 is whether the geodesic of length less than $\delta$ which joins two points $q\_1$, $q\_2$ of $W$ is contained in $W$. If this is the case for every pair of points in $W$, we say that $W$ is **convex**. We say that a parametrized geodesic joining two points is **minimal** if its length is smaller than or equal to that of any other parametrized piecewise regular curve joining these two points.

When $W$ is convex, we have by Prop. 4 of Sec. 4-6 (see also Remark 3 of Sec. 4-6) that the geodesic joining $q\_1 \in W$ to $q\_2 \in W$ is the unique minimal geodesic in $B\_c(p)$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Piecewise Geodesics that Minimize Length are Geodesics)</span></p>

Let $\alpha\colon I \to S$ be a parametrized, piecewise regular curve such that in each regular arc the parameter is proportional to the arc length. Suppose that the arc length between any two of its points is smaller than or equal to the arc length of any parametrized piecewise regular curve joining these points. Then $\alpha$ is a geodesic; in particular, $\alpha$ is regular everywhere.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Geodesics Tangent to Small Circles Leave the Disk)</span></p>

For each point $p \in S$ there exists a positive number $\epsilon$ with the following property: If a geodesic $\gamma(t)$ is tangent to the geodesic circle $S\_r(p)$, $r < \epsilon$, at $\gamma(0)$, then, for $t \neq 0$ small, $\gamma(t)$ is outside $B\_r(p)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4</span><span class="math-callout__name">(Existence of Convex Neighborhoods)</span></p>

For each point $p \in S$ there exists a number $c > 0$ such that $B\_c(p)$ is convex; that is, any two points of $B\_c(p)$ can be joined by a unique minimal geodesic in $B\_c(p)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $\epsilon$ be given as in Prop. 3 and $W$ in Prop. 1 in such a way that $\delta < \epsilon/2$. Choose $c < \delta$ and such that $B\_c(p) \subset W$. We shall prove that $B\_c(p)$ is convex.

Let $q\_1, q\_2 \in B\_c(p)$ and let $\gamma\colon I \to S$ be the geodesic with length less than $\delta < \epsilon/2$ joining $q\_1$ to $q\_2$. $\gamma(I)$ is clearly contained in $B\_\epsilon(p)$, and we want to prove that $\gamma(I)$ is contained in $B\_c(p)$. Assume the contrary. Then there is a point $m \in B\_c(p)$ where the maximum distance $v$ of $\gamma(I)$ to $p$ is attained. In a neighborhood of $m$, the points of $\gamma(I)$ will be in $B\_r(p)$. But this contradicts Prop. 3. **Q.E.D.**

</details>
</div>

### A Geodesic Cannot Be Asymptotic to a Non-Geodesic Parallel

As a further application of Prop. 1, we sketch a proof that a geodesic $\gamma(t)$ of a surface of revolution cannot be asymptotic to a parallel $P\_0$ which is not itself a geodesic. In Example 6 of Sec. 4-4 we have used this fact.

Assume the contrary, and let $p$ be a point in the parallel $P\_0$. Let $W$ and $\delta$ be the neighborhood and the number given by Prop. 1. Because $\gamma(t)$ is asymptotic to $P\_0$, the point $p$ is a limit of points $\gamma(t\_i)$, where $\lbrace t\_i \rbrace \to \infty$, and the tangents of $\gamma$ at $t\_i$ converge to the tangent of $P\_0$ at $p$. By Remark 1, the geodesic $\bar{\gamma}(t)$ with length smaller than $\delta$ joining $p$ to $q$ must be tangent to $P\_0$ at $p$. By Clairaut's relation (cf. Example 5, Sec. 4-4), a small arc of $\bar{\gamma}(t)$ around $p$ will be in the region of $W$ where $\gamma(t)$ lies. It follows that, sufficiently close to $p$, there is a pair of points in $W$ joined by two geodesics of length smaller than $\delta$. This is a contradiction, and proves our claim.

## Appendix — Proofs of the Fundamental Theorems of the Local Theory of Curves and Surfaces

In this appendix we shall show how the fundamental theorems of existence and uniqueness of curves and surfaces (Secs. 1-5 and 4-3) may be obtained from theorems on differential equations.

### Proof of the Fundamental Theorem of the Local Theory of Curves

The starting point is to observe that Frenet's equations

$$\frac{dt}{ds} = kn, \qquad \frac{dn}{ds} = -kt - \tau b, \qquad \frac{db}{ds} = \tau n$$

may be considered as a differential system in $I \times R^9$, where $(\xi\_1, \xi\_2, \xi\_3) = t$, $(\xi\_4, \xi\_5, \xi\_6) = n$, $(\xi\_7, \xi\_8, \xi\_9) = b$, and $f\_i$, $i = 1, \dots, 9$, are linear functions (with coefficients that depend on $s$) of the coordinates $\xi\_j$.

Given initial conditions $s\_0 \in I$, $(\xi\_1)\_0, \dots, (\xi\_9)\_0$, there exist an open interval $J \subset I$ containing $s\_0$ and a unique differentiable mapping $\alpha\colon J \to R^9$ with $\alpha(s\_0) = ((\xi\_1)\_0, \dots, (\xi\_9)\_0)$ and $\alpha'(s) = (f\_1, \dots, f\_9)$. Furthermore, if the system is linear, $J = I$.

It follows that given an orthonormal, positively oriented trihedron $\lbrace t\_0, n\_0, b\_0 \rbrace$ in $R^3$ and a value $s\_0 \in I$, there exists a family of trihedrons $\lbrace t(s), n(s), b(s) \rbrace$, $s \in I$, with $t(s\_0) = t\_0$, $n(s\_0) = n\_0$, $b(s\_0) = b\_0$.

One first shows that the family $\lbrace t(s), n(s), b(s) \rbrace$ thus obtained remains orthonormal for every $s \in I$, by using the system to express the derivatives of the six quantities $\langle t, n \rangle$, $\langle n, b \rangle$, $\langle t, b \rangle$, $\langle t, t \rangle$, $\langle n, n \rangle$, $\langle b, b \rangle$ and verifying that the orthonormality conditions constitute a solution with the initial conditions $0, 0, 0, 1, 1, 1$. By uniqueness, the family is orthonormal for all $s$.

From the family $\lbrace t(s), n(s), b(s) \rbrace$ it is possible to obtain a curve by setting $\alpha(s) = \int t(s)\,ds$, $s \in I$. It is clear that $\alpha'(s) = t(s)$ and that $\alpha''(s) = kn$. Therefore, $k(s)$ is the curvature of $\alpha$ at $s$. Moreover, since $\alpha'''(s) = k'n + kn' = k'n - k^2 t - k\tau b$, the torsion of $\alpha$ will be given by $\tau$. $\alpha$ is, therefore, the required curve.

The uniqueness up to translations and rotations follows from the fact that if $\bar{\alpha}\colon \bar{I} \to R^3$ is another curve with the same curvature and torsion, the trihedrons $\lbrace \bar{t}\_0, \bar{n}\_0, \bar{b}\_0 \rbrace$ and $\lbrace t\_0, n\_0, b\_0 \rbrace$ can be made to coincide by a translation $A$ and a rotation $\rho$, after which both solutions satisfy the same system with the same initial conditions. By uniqueness, $\mathbf{x}\_u = \bar{\mathbf{x}}\_u$ and $\mathbf{x}\_v = \bar{\mathbf{x}}\_v$, hence $\mathbf{x} = \bar{\mathbf{x}} + C$, and since they agree at $(u\_0, v\_0)$, $C = 0$.

### Proof of the Fundamental Theorem of the Local Theory of Surfaces (Bonnet)

The idea of the proof is analogous: we search for a family of trihedrons $\lbrace \mathbf{x}\_u, \mathbf{x}\_v, N \rbrace$, depending on $u$ and $v$, which satisfies the system

$$\mathbf{x}_{uu} = \Gamma_{11}^1 \mathbf{x}_u + \Gamma_{11}^2 \mathbf{x}_v + eN, \qquad \mathbf{x}_{uv} = \Gamma_{12}^1 \mathbf{x}_u + \Gamma_{12}^2 \mathbf{x}_v + fN, \qquad \mathbf{x}_{vv} = \Gamma_{22}^1 \mathbf{x}_u + \Gamma_{22}^2 \mathbf{x}_v + gN,$$

$$N_u = a_{11}\mathbf{x}_u + a_{21}\mathbf{x}_v, \qquad N_v = a_{12}\mathbf{x}_u + a_{22}\mathbf{x}_v,$$

where the coefficients $\Gamma\_{ij}^k$, $a\_{ij}$, $i, j = 1, 2$, are obtained from $E$, $F$, $G$, $e$, $f$, $g$ as if they were on a surface.

The above system defines a system of partial differential equations in $V \times R^9$. In contrast to ordinary differential equations, a system of partial differential equations is not generally integrable. For the case in question, the conditions which guarantee the existence and uniqueness of a local solution, for given initial conditions, are

$$\xi_{uv} = \xi_{vu}, \qquad \eta_{uv} = \eta_{vu}, \qquad \zeta_{uv} = \zeta_{vu}.$$

These integrability conditions are equivalent to the equations of Gauss and Mainardi-Codazzi, which are, by hypothesis, satisfied. Therefore, the system is integrable.

Let $\lbrace \xi, \eta, \zeta \rbrace$ be a solution defined in a neighborhood of $(u\_0, v\_0)$ with initial conditions satisfying $\xi\_0^2 = E(u\_0, v\_0)$, $\eta\_0^2 = G(u\_0, v\_0)$, $\langle \xi\_0, \eta\_0 \rangle = F(u\_0, v\_0)$, $\zeta\_0^2 = 1$, $\langle \xi\_0, \zeta\_0 \rangle = \langle \eta\_0, \zeta\_0 \rangle = 0$.

One then shows that these inner product relations are preserved for all $(u, v)$ where the solution is defined, by expressing the partial derivatives of $\xi^2$, $\eta^2$, $\langle \xi, \eta \rangle$, $\zeta^2$, $\langle \xi, \zeta \rangle$, $\langle \eta, \zeta \rangle$ as a system of 12 partial differential equations and using uniqueness.

With this solution we form a new system $\mathbf{x}\_u = \xi$, $\mathbf{x}\_v = \eta$, which is integrable since $\xi\_v = \eta\_u$. Let $\mathbf{x}\colon \bar{V} \to R^3$ be a solution, with $\mathbf{x}(u\_0, v\_0) = p\_0 \in R^3$. We conclude that $\mathbf{x}\_u \wedge \mathbf{x}\_v \neq 0$ (since $EG - F^2 > 0$); hence $\mathbf{x}(\bar{V}) \subset R^3$ is a regular surface. From the preserved inner products it follows that $E$, $F$, $G$ are the coefficients of the first fundamental form and that $\zeta$ is a unit vector normal to the surface. The coefficients of the second fundamental form are computed to be $e$, $f$, $g$, concluding the first part.

The uniqueness up to translations and rotations of $R^3$ follows by showing that if $\bar{\mathbf{x}}$ is another surface with the same first and second fundamental forms, the trihedrons can be matched at $(u\_0, v\_0)$, and by uniqueness of the system the two surfaces agree everywhere in a connected neighborhood. Since $\mathbf{x}(u\_0, v\_0) = \bar{\mathbf{x}}(u\_0, v\_0)$, we have $\mathbf{x} = \bar{\mathbf{x}}$, completing the proof.

# Chapter 5 — Global Differential Geometry

## 5-1. Introduction

The goal of this chapter is to provide an introduction to global differential geometry. Global differential geometry deals with the relations between local (in general, topological) properties of curves and surfaces. We have already met global theorems (the characterization of compact orientable surfaces in Sec. 2-7 and the Gauss-Bonnet theorem in Sec. 4-5). Now, with the local theory out of the way, we can start a more systematic study of global properties.

The chapter covers the following topics:

- **Sec. 5-2:** The rigidity of the sphere — a compact, connected, regular surface with constant Gaussian curvature $K$ is a sphere.
- **Sec. 5-3:** Complete surfaces and the Hopf-Rinow theorem — the existence of a minimal geodesic joining any two points of a complete surface.
- **Sec. 5-4:** First and second variations of arc length; Bonnet's theorem.
- **Sec. 5-5:** Jacobi fields and conjugate points.
- **Sec. 5-6:** Covering spaces and Hadamard's theorems.
- **Sec. 5-7:** Global theorems for curves.
- **Sec. 5-8:** Surfaces of zero Gaussian curvature.
- **Sec. 5-9:** Jacobi's theorem on conjugate points.
- **Sec. 5-10:** Abstract surfaces and Riemannian manifolds.
- **Sec. 5-11:** Hilbert's theorem on surfaces of constant negative curvature.

## 5-2. The Rigidity of the Sphere

We shall prove that the sphere is *rigid* in the following sense. Let $\varphi\colon \Sigma \to S$ be an isometry of a sphere $\Sigma \subset R^3$ onto a regular surface $S = \varphi(\Sigma) \subset R^3$. Then $S$ is a sphere. Intuitively, this means that it is not possible to deform a sphere made of a flexible but inelastic material.

Actually, we shall prove the following stronger theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Rigidity of the Sphere — Liebmann)</span></p>

Let $S$ be a compact, connected, regular surface with constant Gaussian curvature $K$. Then $S$ is a sphere.

</div>

The rigidity of the sphere follows immediately: if $\varphi\colon \Sigma \to S$ is an isometry of a sphere $\Sigma$ onto $S$, then $S$ has constant curvature (since $K$ is invariant under isometries). Furthermore, $\varphi(\Sigma) = S$ is compact and connected as a continuous image of a compact and connected set $\Sigma$. It follows from Theorem 1 that $S$ is a sphere.

The proof of Theorem 1 is based on the following local lemma, which uses the Mainardi-Codazzi equations.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1</span><span class="math-callout__name">(Local Lemma for Rigidity)</span></p>

Let $S$ be a regular surface and $p \in S$ a point satisfying the following conditions:

1. $K(p) > 0$; that is, the Gaussian curvature in $p$ is positive.
2. $p$ is simultaneously a point of local maximum for the function $k\_1$ and a point of local minimum for the function $k\_2$ ($k\_1 \ge k\_2$).

Then $p$ is an umbilical point of $S$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 1</summary>

Assume that $p$ is not an umbilical point and obtain a contradiction. If $p$ is not umbilical, parametrize a neighborhood of $p$ by coordinates $(u, v)$ so that the coordinate lines are lines of curvature (Sec. 3-4). In this system, $F = f = 0$, and the principal curvatures are $k\_1 = e/E$, $k\_2 = g/G$.

The Mainardi-Codazzi equations (Sec. 4-3, Eqs. (7) and (7a)) become

$$E(k_1)_v = \frac{E_v}{2}(-k_1 + k_2), \qquad G(k_2)_u = \frac{G_u}{2}(k_1 - k_2).$$

Differentiating the first equation with respect to $v$ and the second with respect to $u$, and using the Gauss formula for $F = 0$:

$$-2KEG = E_{vv} + G_{uu} + ME_v + NG_u,$$

we arrive at

$$-(k_1 - k_2)KEG = -2E(k_1)_{vv} + 2G(k_2)_{uu} + \tilde{M}(k_1)_v + \tilde{N}(k_2)_u.$$

Since $K > 0$ and $k\_1 > k\_2$ at $p$, the first member is strictly negative. Since $k\_1$ reaches a local maximum at $p$ and $k\_2$ reaches a local minimum at $p$, we have $(k\_1)\_v = (k\_2)\_u = 0$, $(k\_1)\_{vv} \le 0$, $(k\_2)\_{uu} \ge 0$ at $p$. However, this implies that the second member of the equation is positive or zero, which is a contradiction. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2</span><span class="math-callout__name">(Compact Surfaces Have Elliptic Points)</span></p>

A regular compact surface $S \subset R^3$ has at least one elliptic point.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 2</summary>

Since $S$ is compact, $S$ is bounded. Therefore, there are spheres of $R^3$, centered in a fixed point $O \in R^3$, such that $S$ is contained in the interior of the region bounded by any of them. Consider the set of all such spheres and let $r$ be the infimum of their radii. Let $\Sigma \subset R^3$ be a sphere of radius $r$ centered in $O$. It is clear that $\Sigma$ and $S$ have at least one common point, say $p$. The tangent plane to $\Sigma$ at $p$ has only the common point $p$ with $S$, in a neighborhood of $p$. Therefore, $\Sigma$ and $S$ are tangent at $p$. By observing the normal sections at $p$, it is easy to conclude that any normal curvature of $S$ at $p$ is greater than or equal to the corresponding curvature of $\Sigma$ at $p$. Therefore, $K\_S(p) \ge K\_\Sigma(p) > 0$, and $p$ is an elliptic point. **Q.E.D.**

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 1</summary>

Since $S$ is compact, there is an elliptic point by Lemma 2. Because $K = k\_1 k\_2$ is constant and positive, $K > 0$ on $S$.

By compactness, the continuous function $k\_1$ on $S$ reaches a maximum at a point $p \in S$. Since $K = k\_1 k\_2$ is a positive constant, $k\_2$ is a decreasing function of $k\_1$, and therefore it reaches a minimum at $p$.

Now let $q$ be any given point of $S$. Since we assumed $k\_1(q) \ge k\_2(q)$, we have

$$k_1(p) \ge k_1(q) \ge k_2(q) \ge k_2(p) = k_1(p),$$

the last equality holding because $K$ is constant. Therefore, $k\_1(q) = k\_2(q)$ for every $q \in S$.

It follows that all the points of $S$ are umbilical points and, by Prop. 4 of Sec. 3-2, $S$ is contained in a sphere or a plane. Since $K > 0$, $S$ is contained in a sphere $\Sigma$. By compactness, $S$ is closed in $\Sigma$, and since $S$ is a regular surface, $S$ is open in $\Sigma$. Since $\Sigma$ is connected, $S = \Sigma$. Therefore, the surface $S$ is a sphere. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1a</span><span class="math-callout__name">(Constant $K > 0$ and $H$ Implies Sphere)</span></p>

Let $S$ be a regular, compact, and connected surface with Gaussian curvature $K > 0$ and mean curvature $H$ constant. Then $S$ is a sphere.

</div>

The proof is entirely analogous to that of Theorem 1: the argument applies whenever $k\_2 = f(k\_1)$, where $f$ is a decreasing function of $k\_1$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Ovaloids and Hopf's Theorem)</span></p>

The compact, connected surfaces in $R^3$ for which $K > 0$ are called **ovaloids**. Thus Theorem 1a may be stated as: *An ovaloid of constant mean curvature is a sphere*.

It is a simple consequence of the Gauss-Bonnet theorem that an ovaloid is homeomorphic to a sphere (cf. Sec. 4-5, application 1). H. Hopf proved the following (stronger) statement: *A regular surface of constant mean curvature that is homeomorphic to a sphere is a sphere*. A. Alexandroff extended this further by replacing the condition of being homeomorphic to a sphere by compactness: *A regular, compact, and connected surface of constant mean curvature is a sphere*.

The rigidity of the sphere may be obtained as a consequence of the Cohn-Vossen theorem: *Two isometric ovaloids differ by an orthogonal linear transformation of $R^3$*.

</div>

## 5-3. Complete Surfaces. Theorem of Hopf-Rinow

All the surfaces to be considered from now on will be regular and connected, except when otherwise stated.

The considerations at the end of Sec. 5-1 have shown that in order to obtain global theorems we require, besides connectedness, some global hypothesis to ensure that the surface cannot be "extended" further as a regular surface. It is clear that compactness serves this purpose. However, it would be useful to have a hypothesis weaker than compactness which could still have the same effect.

### Extendable and Complete Surfaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Extendable Surface)</span></p>

A regular (connected) surface $S$ is said to be **extendable** if there exists a regular (connected) surface $\bar{S}$ such that $S \subset \bar{S}$ as a proper subset. $S$ is said to be **nonextendable** if there exists no such $\bar{S}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Complete Surface)</span></p>

A regular surface $S$ is said to be **complete** when for every point $p \in S$, any parametrized geodesic $\gamma\colon [0, \epsilon) \to S$ of $S$, starting from $p = \gamma(0)$, may be extended into a parametrized geodesic $\bar{\gamma}\colon R \to S$, defined on the entire line $R$. In other words, $S$ is complete when for every $p \in S$ the mapping $\exp\_p\colon T\_p(S) \to S$ (Sec. 4-6) is defined for every $v \in T\_p(S)$.

</div>

The plane is clearly a complete surface. The cone minus the vertex is a noncomplete surface, since by extending a generator (which is a geodesic) sufficiently we reach the vertex, which does not belong to the surface. A sphere is a complete surface, since its parametrized geodesics (the great circles) may be defined for every real value. The cylinder is also a complete surface since its geodesics are circles, lines, and helices, which are defined for all real values.

On the other hand, a surface $S - \lbrace p \rbrace$ obtained by removing a point $p$ from a complete surface $S$ is not complete. In fact, a geodesic $\gamma$ of $S$, starting from a nearby point $q$, can be found that cannot be extended past $p$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Complete Surfaces are Nonextendable)</span></p>

A complete surface $S$ is nonextendable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Assume that $S$ is extendable, meaning there exists a regular surface $\bar{S}$ with $S \subset \bar{S}$. Since $S$ is a regular surface, $S$ is open in $\bar{S}$. Since $\bar{S} \neq S$, the boundary $\text{Bd}\,S$ of $S$ in $\bar{S}$ is nonempty (otherwise, $\bar{S} = S \cup (\bar{S} - S)$ would be the union of two disjoint open sets, contradicting the connectedness of $\bar{S}$).

Therefore, there exists a point $p \in \text{Bd}\,S$, $p \notin S$. Let $\bar{V}$ be a neighborhood of $p$ in $\bar{S}$ such that every point $q \in \bar{V}$ may be joined to $p$ by a unique geodesic of $\bar{S}$. Since $p \in \text{Bd}\,S$, there exists $q \in \bar{V} \cap S$. Let $\bar{\gamma}\colon [0, 1] \to \bar{S}$ be a geodesic of $\bar{S}$ with $\bar{\gamma}(0) = p$ and $\bar{\gamma}(1) = q$. It is clear that $\alpha(t) = \bar{\gamma}(1 - t)$, $t \in [0, \epsilon)$, is a geodesic of $S$, with $\alpha(0) = q$, the extension of which to the line $R$ would pass through $p$ for $t = 1$. Since $p \notin S$, this geodesic cannot be extended, which contradicts the hypothesis of completeness and concludes the proof. **Q.E.D.**

</details>
</div>

The converse of Prop. 1 is false: the cone minus the vertex is nonextendable but not complete (Example 1 in the text).

### The Intrinsic Distance

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Intrinsic Distance)</span></p>

The **intrinsic distance** $d(p, q)$ from the point $p \in S$ to the point $q \in S$ is the number

$$d(p, q) = \inf l(\alpha_{p,q}),$$

where the infimum is taken over all piecewise differentiable curves joining $p$ to $q$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Properties of the Intrinsic Distance)</span></p>

The distance $d$ defined above has the following properties:

1. $d(p, q) = d(q, p)$,
2. $d(p, q) + d(q, r) \ge d(p, r)$,
3. $d(p, q) \ge 0$,
4. $d(p, q) = 0$ if and only if $p = q$,

where $p$, $q$, $r$ are arbitrary points of $S$.

</div>

The intrinsic distance $d$ gives $S$ the structure of a **metric space**. It is an important fact that the intrinsic metric $d$ and the metric $\bar{d}$ induced by $R^3$ on $S$ (where $\bar{d}(p, q) = \|p - q\|$) determine the same topology on $S$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5</span><span class="math-callout__name">(Closed Surfaces are Complete)</span></p>

A closed surface $S \subset R^3$ is complete.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $\gamma\colon [0, \epsilon) \to S$, $\gamma(0) = p \in S$, be a parametrized geodesic of $S$, parametrized by arc length. We need to show that it is possible to extend $\gamma$ to a geodesic $\bar{\gamma}\colon R \to S$, defined on the entire line $R$. Observe first that when $\bar{\gamma}(s\_0)$, $s\_0 \in R$, is defined, then, by the theorem of existence and uniqueness of geodesics (Sec. 4-4, Prop. 5), it is possible to extend $\bar{\gamma}$ to a neighborhood of $s\_0$ in $R$. Therefore, the set of all $s \in R$ where $\bar{\gamma}$ is defined is open in $R$.

Let us assume that $\bar{\gamma}$ is defined for $s < s\_0$ and let us show that $\bar{\gamma}$ is defined for $s = s\_0$. Consider a sequence $\lbrace s\_n \rbrace \to s\_0$, with $s\_n < s\_0$. We shall first prove that the sequence $\lbrace \bar{\gamma}(s\_n) \rbrace$ converges in $S$. Given $\epsilon > 0$, there exists $n\_0$ such that if $n, m > n\_0$, then $\|s\_n - s\_m\| < \epsilon$. Denote by $\bar{d}$ the distance in $R^3$, and observe that $\bar{d}(p, q) \le d(p, q)$ for $p, q \in S$. Thus,

$$\bar{d}(\bar{\gamma}(s_n), \bar{\gamma}(s_m)) \le d(\bar{\gamma}(s_n), \bar{\gamma}(s_m)) \le |s_n - s_m| < \epsilon.$$

It follows that $\lbrace \bar{\gamma}(s\_n) \rbrace$ is a Cauchy sequence in $R^3$; hence, it converges to a point $q \in R^3$. Since $q$ is a limit point of $\lbrace \bar{\gamma}(s\_n) \rbrace$ and $S$ is closed, $q \in S$, which proves our assertion. The proof is concluded by the theorem of existence and uniqueness, applied at $s = s\_0$. **Q.E.D.**

</details>
</div>

**Corollary.** *A compact surface is complete.* (Since compact subsets of $R^3$ are closed.)

The converse of Prop. 5 does not hold: a right cylinder erected over a plane curve that is asymptotic to a circle is easily seen to be complete but not closed.

### The Hopf-Rinow Theorem

We say that a geodesic $\gamma$ joining two points $p, q \in S$ is **minimal** if its length $l(\gamma)$ is smaller than or equal to the length of any piecewise regular curve joining $p$ to $q$. This is equivalent to saying that $l(\gamma) = d(p, q)$.

A minimal geodesic may not exist: on $S^2 - \lbrace p \rbrace$, two points $p\_1$ and $p\_2$ symmetric relative to $p$ and sufficiently near to $p$ cannot be joined by a minimal geodesic.

On the other hand, there may exist an infinite number of minimal geodesics: all the meridians that join two antipodal points of a sphere are minimal geodesics.

The main result of this section is that in a complete surface there always exists a minimal geodesic joining two given points.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hopf-Rinow)</span></p>

Let $S$ be a complete surface. Given two points $p, q \in S$, there exists a minimal geodesic joining $p$ to $q$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Sketch</summary>

Let $r = d(p, q)$ be the distance between $p$ and $q$. Let $B\_\delta(0) \subset T\_p(S)$ be a disk of radius $\delta$, centered in the origin $0$ of the tangent plane and contained in a neighborhood $U \subset T\_p(S)$ of $0$, where $\exp\_p$ is a diffeomorphism. Let $B\_\delta(p) = \exp\_p(B\_\delta(0))$. Observe that the boundary $\text{Bd}\,B\_\delta(p) = \Sigma$ is compact since it is the continuous image of the compact set $\text{Bd}\,B\_\delta(0)$.

If $x \in \Sigma$, the continuous function $d(x, q)$ reaches a minimum at a point $x\_0 \in \Sigma$. The point $x\_0$ may be written as $x\_0 = \exp\_p(\delta v)$, $\|v\| = 1$, $v \in T\_p(S)$.

Let $\gamma$ be the geodesic parametrized by arc length, given by $\gamma(s) = \exp\_p(sv)$. Since $S$ is complete, $\gamma$ is defined for every $s \in R$. In particular, $\gamma$ is defined in the interval $[0, r]$. If we show that $\gamma(r) = q$, then $\gamma$ must be a geodesic joining $p$ to $q$ which is minimal, since $l(\gamma) = r = d(p, q)$, and this will conclude the proof.

To prove this, we shall show that if $s \in [\delta, r]$, then

$$d(\gamma(s), q) = r - s. \tag{1}$$

Equation (1) implies, for $s = r$, that $\gamma(r) = q$ (by property 4 of distance).

We first prove (1) for $s = \delta$: since every curve joining $p$ to $q$ intersects $\Sigma$, denoting by $x$ an arbitrary point of $\Sigma$,

$$d(p, q) = \inf_{x \in \Sigma}\lbrace d(p, x) + d(x, q) \rbrace = \inf_{x \in \Sigma}(\delta + d(x, q)) = \delta + d(x_0, q).$$

Hence, $d(\gamma(\delta), q) = r - \delta$.

Next we show that if (1) holds for $s\_0 \in [\delta, r]$, then for $\delta' > 0$ and sufficiently small, it holds for $s\_0 + \delta'$. This is done by repeating the same argument at $\gamma(s\_0)$: a new geodesic sphere $B\_{\delta'}(\gamma(s\_0))$ has boundary $\Sigma'$, and the minimum of $d(x', q)$ on $\Sigma'$ is achieved along $\gamma$, giving $d(\gamma(s\_0 + \delta'), q) = r - s\_0 - \delta'$.

The set $A = \lbrace s \in [\delta, r] : \text{Eq. (1) holds} \rbrace$ is clearly closed in $[0, r]$. By the above, if $s\_0 \in A$ and $s\_0 < r$, then it holds for $s\_0 + \delta'$. Thus $A$ is also relatively open in $[\delta, r]$. Since $[\delta, r]$ is connected and $A$ is nonempty (it contains $\delta$), $A = [\delta, r]$ and Eq. (1) is proved. **Q.E.D.**

</details>
</div>

## 5-4. First and Second Variations of Arc Length; Bonnet's Theorem

The goal of this section is to prove that a complete surface $S$ with Gaussian curvature $K \ge \delta > 0$ is compact (Bonnet's theorem).

The crucial point of the proof is to show that if $K \ge \delta > 0$, a geodesic $\gamma$ joining two arbitrary points $p, q \in S$ and having length $l(\gamma) > \pi/\sqrt{\delta}$ is no longer minimal; that is, there exists a parametrized curve joining $p$ and $q$, the length of which is smaller than $l(\gamma)$.

Once this is proved, it follows that every minimal geodesic has length $l \le \pi/\sqrt{\delta}$; thus, $S$ is bounded in the distance $d$. Since $S$ is complete, $S$ is compact (Corollary 2, Sec. 5-3). We remark that, in addition, we obtain an estimate for the diameter $\rho$ of $S$, namely, $\rho(S) \le \pi/\sqrt{\delta}$.

To compare the arc length of a parametrized curve with the arc length of "neighboring curves," we shall introduce several ideas which are adaptations to the purposes of differential geometry of more general concepts found in calculus of variations.

### Variations and the Variational Vector Field

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Variation of a Curve)</span></p>

Let $\alpha\colon [0, l] \to S$ be a regular parametrized curve, where the parameter $s \in [0, l]$ is the arc length. A **variation** of $\alpha$ is a differentiable map $h\colon [0, l] \times (-\epsilon, \epsilon) \subset R^2 \to S$ such that

$$h(s, 0) = \alpha(s), \qquad s \in [0, l].$$

For each $t \in (-\epsilon, \epsilon)$, the curve $h\_t\colon [0, l] \to S$, given by $h\_t(s) = h(s, t)$, is called a **curve of the variation** $h$. A variation $h$ is said to be **proper** if

$$h(0, t) = \alpha(0), \qquad h(l, t) = \alpha(l), \qquad t \in (-\epsilon, \epsilon).$$

</div>

It follows that a variation $h$ of $\alpha$ determines a differentiable vector field $V(s)$ along $\alpha$ by

$$V(s) = \frac{\partial h}{\partial t}(s, 0), \qquad s \in [0, l].$$

$V$ is called the **variational vector field** of $h$; we remark that if $h$ is proper, then $V(0) = V(l) = 0$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Realization of a Variational Vector Field)</span></p>

If we let $V(s)$ be a differentiable vector field along a parametrized regular curve $\alpha\colon [0, l] \to S$, then there exists a variation $h\colon [0, l] \times (-\epsilon, \epsilon) \to S$ of $\alpha$ such that $V(s)$ is the variational vector field of $h$. Furthermore, if $V(0) = V(l) = 0$, then $h$ can be chosen to be proper.

</div>

### The Arc Length Function and the First Variation

We define a function $L\colon (-\epsilon, \epsilon) \to R$ by

$$L(t) = \int_0^l \left|\frac{\partial h}{\partial s}(s, t)\right| ds, \qquad t \in (-\epsilon, \epsilon).$$

The study of $L$ in a neighborhood of $t = 0$ will inform us of the "arc length behavior" of curves neighboring $\alpha$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Formula for the First Variation of Arc Length)</span></p>

Let $h\colon [0, l] \times (-\epsilon, \epsilon) \to S$ be a proper variation of the curve $\alpha\colon [0, l] \to S$ and let $V(s) = (\partial h/\partial t)(s, 0)$, $s \in [0, l]$, be the variational vector field of $h$. Then

$$L'(0) = -\int_0^l \langle A(s),\; V(s) \rangle\,ds,$$

where $A(s) = (D/\partial s)(\partial h/\partial s)(s, 0)$ is the acceleration vector of the curve $\alpha$.

</div>

The vector $A(s)$ is called the **acceleration vector** of the curve $\alpha$, and its norm is nothing but the absolute value of the geodesic curvature of $\alpha$. Expression $L'(0)$ is usually called the **formula for the first variation** of the arc length of $\alpha$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Variational Characterization of Geodesics)</span></p>

A regular parametrized curve $\alpha\colon [0, l] \to S$, where the parameter $s \subset [0, l]$ is the arc length of $\alpha$, is a geodesic if and only if, for every proper variation $h\colon [0, l] \times (-\epsilon, \epsilon) \to S$ of $\alpha$, $L'(0) = 0$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The necessity is trivial since the acceleration vector $A(s) = (D/\partial s)(\partial \alpha/\partial s)$ of a geodesic $\alpha$ is identically zero. Therefore, $L'(0) = 0$ for every proper variation.

Suppose now that $L'(0) = 0$ for every proper variation of $\alpha$ and consider a vector field $V(s) = f(s)A(s)$, where $f\colon [0, l] \to R$ is a real differentiable function, with $f(s) \ge 0$, $f(0) = f(l) = 0$, and $A(s)$ is the acceleration vector of $\alpha$. By constructing a variation corresponding to $V(s)$, we have

$$L'(0) = -\int_0^l \langle f(s)A(s),\; A(s) \rangle\,ds = -\int_0^l f(s)|A(s)|^2\,ds = 0.$$

Therefore, since $f(s)\|A(s)\|^2 \ge 0$, we obtain $f(s)\vert A(s)\vert ^2 \equiv 0$.

We shall prove that $A(s) = 0$, $s \in [0, l]$. In fact, if $\|A(s\_0)\| \neq 0$, $s\_0 \in (0, l)$, there exists an interval $I = (s\_0 - \epsilon, s\_0 + \epsilon)$ such that $\|A(s)\| \neq 0$ for $s \in I$. By choosing $f$ such that $f(s\_0) > 0$, we contradict $f(s\_0)\|A(s\_0)\| = 0$. Therefore, $\|A(s)\| = 0$ when $s \in (0, l)$. By continuity, $A(0) = A(l) = 0$ as asserted. Since the acceleration vector of $\alpha$ is identically zero, $\alpha$ is geodesic. **Q.E.D.**

</details>
</div>

### The Second Variation of Arc Length

From now on, we shall only consider proper variations of geodesics. To simplify the computations, we shall restrict ourselves to **orthogonal variations**; that is, we shall assume that the variational field $V(s)$ satisfies the condition $\langle V(s), \gamma'(s) \rangle = 0$, $s \in [0, l]$.

The key technical tools are the following lemmas.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5</span><span class="math-callout__name">(Curvature and Covariant Derivatives)</span></p>

Let $\mathbf{x}\colon U \to S$ be a parametrization at a point $p \in S$ of a regular surface $S$, with parameters $u$, $v$, and let $K$ be the Gaussian curvature of $S$. Then

$$\frac{D}{\partial v}\frac{D}{\partial u}\mathbf{x}_u - \frac{D}{\partial u}\frac{D}{\partial v}\mathbf{x}_u = K(\mathbf{x}_u \wedge \mathbf{x}_v) \wedge \mathbf{x}_u.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 6</span><span class="math-callout__name">(Curvature and Variations)</span></p>

Let $h\colon [0, l] \times (-\epsilon, \epsilon) \to S$ be a differentiable mapping and let $V(s, t)$, $(s, t) \in [0, l] \times (-\epsilon, \epsilon)$, be a differentiable vector field along $h$. Then

$$\frac{D}{\partial t}\frac{D}{\partial s}V - \frac{D}{\partial s}\frac{D}{\partial t}V = K(s, t)\left(\frac{\partial h}{\partial s} \wedge \frac{\partial h}{\partial t}\right) \wedge V,$$

where $K(s, t)$ is the curvature of $S$ at the point $h(s, t)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4</span><span class="math-callout__name">(Formula for the Second Variation of Arc Length)</span></p>

Let $h\colon [0, l] \times (-\epsilon, \epsilon) \to S$ be a proper orthogonal variation of a geodesic $\gamma\colon [0, l] \to S$ parametrized by the arc length $s \in [0, l]$. Let $V(s) = (\partial h/\partial t)(s, 0)$ be the variational vector field of $h$. Then

$$L''(0) = \int_0^l \left(\left|\frac{D}{ds}V(s)\right|^2 - K(s)|V(s)|^2\right) ds,$$

where $K(s) = K(s, 0)$ is the Gaussian curvature of $S$ at $\gamma(s) = h(s, 0)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Sketch</summary>

Starting from the first variation formula and differentiating again, using the symmetry Lemma 4 ($\frac{D}{\partial s}\frac{\partial h}{\partial t} = \frac{D}{\partial t}\frac{\partial h}{\partial s}$), Lemma 3, and Lemma 6, one arrives at

$$L''(t) = \int_0^l \frac{d}{dt}\left\langle \frac{D}{\partial s}\frac{\partial h}{\partial t},\; \frac{\partial h}{\partial s}\right\rangle ds - \int_0^l \frac{\left(\left\langle \frac{D}{\partial s}\frac{\partial h}{\partial t},\; \frac{\partial h}{\partial s}\right\rangle\right)^2}{|\partial h/\partial s|^{3/2}}\,ds.$$

Evaluating at $t = 0$, using the facts that $\|\partial h/\partial s(s, 0)\| = 1$, $(D/\partial s)(\partial h/\partial s)(s, 0) = 0$ (since $\gamma$ is a geodesic), and $\langle \partial h/\partial s, \partial h/\partial t \rangle = 0$ (orthogonal variation), and applying Lemma 6 to handle the commutator of covariant derivatives, one obtains the stated formula. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Alternative Form of the Second Variation)</span></p>

It is often convenient to have the formula for the second variation written as follows:

$$L''(0) = -\int_0^l \left\langle \frac{D^2 V}{ds^2} + KV,\; V \right\rangle ds.$$

This form is obtained from Prop. 4 by integration by parts, using $V(0) = V(l) = 0$.

</div>

### Bonnet's Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bonnet)</span></p>

Let the Gaussian curvature $K$ of a complete surface $S$ satisfy the condition

$$K \ge \delta > 0.$$

Then $S$ is compact and the diameter $\rho$ of $S$ satisfies the inequality

$$\rho \le \frac{\pi}{\sqrt{\delta}}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $S$ is complete, given two points $p, q \in S$, there exists, by the Hopf-Rinow theorem, a minimal geodesic $\gamma$ of $S$ joining $p$ to $q$. We shall prove that the length $l = d(p, q)$ of this geodesic satisfies the inequality $l \le \pi/\sqrt{\delta}$.

We shall assume that $l > \pi/\sqrt{\delta}$ and consider a variation of the geodesic $\gamma\colon [0, l] \to S$, defined as follows. Let $w\_0$ be a unit vector of $T\_{\gamma(0)}(S)$ such that $\langle w\_0, \gamma'(0) \rangle = 0$ and let $w(s)$, $s \in [0, l]$, be the parallel transport of $w\_0$ along $\gamma$. It is clear that $\|w(s)\| = 1$ and that $\langle w(s), \gamma'(s) \rangle = 0$, $s \in [0, l]$. Consider the vector field $V(s)$ defined by

$$V(s) = w(s)\sin\frac{\pi}{l}s, \qquad s \in [0, l].$$

Since $V(0) = V(l) = 0$ and $\langle V(s), \gamma'(s) \rangle = 0$, the vector field $V(s)$ determines a proper, orthogonal variation of $\gamma$. By Prop. 4,

$$L_V''(0) = \int_0^l \left(\left|\frac{D}{\partial s}V(s)\right|^2 - K|V(s)|^2\right) ds.$$

Since $w(s)$ is a parallel vector field,

$$\frac{D}{\partial s}V(s) = \left(\frac{\pi}{l}\cos\frac{\pi}{l}s\right)w(s).$$

Thus, since $l > \pi/\sqrt{\delta}$, so that $K \ge \delta > \pi^2/l^2$, we obtain

$$L_V''(0) = \int_0^l \left(\frac{\pi^2}{l^2}\cos^2\frac{\pi}{l}s - K\sin^2\frac{\pi}{l}s\right) ds < \int_0^l \frac{\pi^2}{l^2}\left(\cos^2\frac{\pi}{l}s - \sin^2\frac{\pi}{l}s\right) ds$$

$$= \frac{\pi^2}{l^2}\int_0^l \cos\frac{2\pi}{l}s\,ds = 0.$$

Therefore, there exists a variation of $\gamma$ for which $L''(0) < 0$. However, since $\gamma$ is a minimal geodesic, its length is smaller than or equal to that of any curve joining $p$ to $q$. Thus, for every variation of $\gamma$ we should have $L'(0) = 0$ and $L''(0) \ge 0$. We obtained therefore a contradiction, which shows that $l = d(p, q) \le \pi/\sqrt{\delta}$, as we asserted.

Since $d(p, q) \le \pi/\sqrt{\delta}$ for any two given points of $S$, we have that $S$ is bounded and its diameter $\rho \le \pi/\sqrt{\delta}$. Moreover, since $S$ is complete and bounded, $S$ is compact. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sharpness and Weakening of Bonnet's Theorem)</span></p>

The estimate of the diameter $\rho \le \pi/\sqrt{\delta}$ given by Bonnet's theorem is the best possible, as shown by the example of the unit sphere: $K \equiv 1$ and $\rho = \pi$. The hypothesis $K \ge \delta > 0$ may not be weakened to $K > 0$. In fact, the paraboloid $(x, y, z) \in R^3;\; z = x^2 + y^2$ has Gaussian curvature $K > 0$, is complete, and is not compact. Note that the curvature of the paraboloid tends toward zero when the distance of the point $(x, y) \in R^2$ to the origin $(0, 0)$ becomes arbitrarily large.

</div>

## 5-5. Jacobi Fields and Conjugate Points

In this section we shall explore some details of the variational techniques which were used to prove Bonnet's theorem. We are interested in obtaining information on the behavior of geodesics neighboring a given geodesic $\gamma$. The natural way to proceed is to consider variations of $\gamma$ which satisfy the further condition that the curves of the variation are themselves geodesics.

To simplify the exposition we shall assume that the surfaces are complete, although this assumption may be dropped with further work. The notation $\gamma\colon [0, l] \to S$ will denote a geodesic parametrized by arc length on the complete surface $S$.

### Jacobi Fields

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Jacobi Field)</span></p>

Let $\gamma\colon [0, l] \to S$ be a parametrized geodesic on $S$ and let $h\colon [0, l] \times (-\epsilon, \epsilon) \to S$ be a variation of $\gamma$ such that for every $t \in (-\epsilon, \epsilon)$, the curve $h\_t(s) = h(s, t)$, $s \in [0, l]$, is a parametrized geodesic (not necessarily parametrized by arc length). The variational field $(\partial h/\partial t)(s, 0) = J(s)$ is called a **Jacobi field** along $\gamma$.

</div>

A trivial example of a Jacobi field is given by the field $\gamma'(s)$, $s \in [0, l]$, of tangent vectors to the geodesic $\gamma$. In fact, by taking $h(s, t) = \gamma(s + t)$, we have $J(s) = d\gamma/ds$.

We are particularly interested in studying the behavior of geodesics neighboring $\gamma\colon [0, l] \to S$, which start from $\gamma(0)$. Thus, we shall consider variations $h\colon [0, l] \times (-\epsilon, \epsilon) \to S$ that satisfy the condition $h(0, t) = \gamma(0)$, $t \in (-\epsilon, \epsilon)$. Therefore, the corresponding Jacobi field satisfies the condition $J(0) = 0$.

### The Jacobi Equation

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(The Jacobi Equation)</span></p>

Let $J(s)$ be a Jacobi field along $\gamma\colon [0, l] \to S$, $s \in [0, l]$. Then $J$ satisfies the so-called **Jacobi equation**

$$\frac{D}{ds}\frac{D}{ds}J(s) + K(s)(\gamma'(s) \wedge J(s)) \wedge \gamma'(s) = 0,$$

where $K(s)$ is the Gaussian curvature of $S$ at $\gamma(s)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By the definition of $J(s)$, there exists a variation $h\colon [0, l] \times (-\epsilon, \epsilon) \to S$ of $\gamma$ such that $(\partial h/\partial t)(s, 0) = J(s)$ and $h\_t(s)$ is a geodesic, $t \in (-\epsilon, \epsilon)$. It follows that $(D/\partial s)(\partial h/\partial s)(s, t) = 0$. Therefore,

$$\frac{D}{\partial t}\frac{D}{\partial s}\frac{\partial h}{\partial s}(s, t) = 0, \qquad (s, t) \in [0, l] \times (-\epsilon, \epsilon).$$

On the other hand, by using Lemma 6 of Sec. 5-4 we have

$$\frac{D}{\partial t}\frac{D}{\partial s}\frac{\partial h}{\partial s} = \frac{D}{\partial s}\frac{D}{\partial t}\frac{\partial h}{\partial s} + K(s, t)\left(\frac{\partial h}{\partial s} \wedge \frac{\partial h}{\partial t}\right) \wedge \frac{\partial h}{\partial s} = 0.$$

Since $(D/\partial t)(\partial h/\partial t) = (D/\partial s)(\partial h/\partial t)$, we have, for $t = 0$,

$$\frac{D}{\partial s}\frac{D}{\partial s}J(s) + K(s)(\gamma'(s) \wedge J(s)) \wedge \gamma'(s) = 0. \qquad \textbf{Q.E.D.}$$

</details>
</div>

To draw consequences from the Jacobi equation, it is convenient to put it in a more familiar form. For that, let $e\_1(0)$ and $e\_2(0)$ be unit orthogonal vectors in the tangent plane $T\_{\gamma(0)}(S)$ and let $e\_1(s)$ and $e\_2(s)$ be the parallel transport of $e\_1(0)$ and $e\_2(0)$, respectively, along $\gamma(s)$. Assume that

$$J(s) = a_1(s)e_1(s) + a_2(s)e_2(s)$$

for some functions $a\_1 = a\_1(s)$, $a\_2 = a\_2(s)$. Then, since $e\_1$ and $e\_2$ are parallel, $DJ/ds = a\_1'e\_1 + a\_2'e\_2$ and $D^2J/ds^2 = a\_1''e\_1 + a\_2''e\_2$. Writing $(\gamma' \wedge J) \wedge \gamma' = \lambda\_1 e\_1 + \lambda\_2 e\_2$, and setting $\langle (\gamma' \wedge e\_i) \wedge \gamma', e\_j \rangle = \alpha\_{ij}$, the Jacobi equation becomes a system of linear, second-order ordinary differential equations:

$$a_1'' + K(\alpha_{11}a_1 + \alpha_{21}a_2) = 0, \qquad a_2'' + K(\alpha_{12}a_1 + \alpha_{22}a_2) = 0.$$

The solutions $(a\_1(s), a\_2(s)) = J(s)$ of such a system are defined for every $s \in [0, l]$ and constitute a vector space. Moreover, a solution $J(s)$ of the Jacobi equation is completely determined by the initial conditions $J(0)$, $(DJ/ds)(0)$, and the space of the solutions has $2 \times 2 = 4$ dimensions.

One can show that every vector field $J(s)$ along a geodesic $\gamma\colon [0, l] \to S$ which satisfies the Jacobi equation, with $J(0) = 0$, is in fact a Jacobi field. Since we are interested only in Jacobi fields $J(s)$ which satisfy the condition $J(0) = 0$, we shall prove the proposition only for this particular case.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Converse: Solutions of the Jacobi Equation are Jacobi Fields)</span></p>

If we let $J(s)$ be a differentiable vector field along $\gamma\colon [0, l] \to S$, $s \in [0, l]$, satisfying the Jacobi equation, with $J(0) = 0$, then $J(s)$ is a Jacobi field along $\gamma$.

</div>

### Jacobi Fields via the Exponential Map

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1</span><span class="math-callout__name">(Jacobi Fields from the Exponential Map)</span></p>

Let $p \in S$ and choose $v, w \in T\_p(S)$, with $\vert v\vert  = l$. Let $\gamma\colon [0, l] \to S$ be the geodesic on $S$ given by $\gamma(s) = \exp\_p(sv)$, $s \in [0, l]$. Then, the vector field $J(s)$ along $\gamma$ given by

$$J(s) = s(d\exp_p)_{sv}(w), \qquad s \in [0, l],$$

is a Jacobi field. Furthermore, $J(0) = 0$, $(DJ/ds)(0) = w$.

</div>

### Conjugate Points

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Conjugate Point)</span></p>

Let $\gamma\colon [0, l] \to S$ be a geodesic of $S$ with $\gamma(0) = p$. We say that the point $q = \gamma(s\_0)$, $s\_0 \in [0, l]$, is **conjugate** to $p$ relative to the geodesic $\gamma$ if there exists a Jacobi field $J(s)$ which is not identically zero along $\gamma$ with $J(0) = J(s\_0) = 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Conjugate Points on the Sphere)</span></p>

Let $S^2 = \lbrace (x, y, z) \in R^3;\; x^2 + y^2 + z^2 = 1 \rbrace$ be the unit sphere and $\mathbf{x}(\theta, \varphi)$ be a parametrization at $p \in S$ by the colatitude $\theta$ and the longitude $\varphi$. Consider on the parallel $\theta = \pi/2$ the segment between $\varphi\_0 = \pi/2$ and $\varphi\_1 = 3\pi/2$. This segment is a geodesic $\gamma$, which we assume to be parametrized by $\varphi - \varphi\_0 = s$. Let $w(s)$ be the parallel transport along $\gamma$ of a vector $w(0) \in T\_{\gamma(0)}(S)$, with $\vert w(0)\vert  = 1$ and $\langle w(0), \gamma'(0) \rangle = 0$. We shall prove that the vector field

$$J(s) = (\sin s)\,w(s), \qquad s \in [0, \pi],$$

is a Jacobi field along $\gamma$. In fact, since $J(0) = 0$, it suffices to verify that $J$ satisfies the Jacobi equation. By using the fact that $K = 1$ and $w$ is a parallel field, we obtain, successively,

$$\frac{DJ}{ds} = (\cos s)\,w(s), \qquad \frac{D^2J}{ds^2} = (-\sin s)\,w(s),$$

$$\frac{D^2J}{ds^2} + K(\gamma' \wedge J) \wedge \gamma' = (-\sin s)w(s) + (\sin s)w(s) = 0,$$

which shows that $J$ is a Jacobi field. Observe that $J(\pi) = 0$.

</div>

In general, given a point $p$ of a surface $S$, the "first" conjugate point $q$ to $p$ varies as we change the direction of the geodesic passing through $p$ and describes a parametrized curve. The trace of such a curve is called the **conjugate locus** to $p$ and is denoted by $C(p)$.

### Conjugate Points and the Exponential Map

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5</span><span class="math-callout__name">(Conjugate Points as Critical Points of $\exp\_p$)</span></p>

Let $p, q \in S$ be two points of $S$ and let $\gamma\colon [0, l] \to S$ be a geodesic joining $p = \gamma(0)$ to $q = \exp\_p(l\gamma'(0))$. Then $q$ is conjugate to $p$ relative to $\gamma$ if and only if $v = l\gamma'(0)$ is a critical point of $\exp\_p\colon T\_p(S) \to S$.

</div>

### No Conjugate Points under Nonpositive Curvature

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(No Conjugate Points for $K \le 0$)</span></p>

Assume that the Gaussian curvature $K$ of a surface $S$ satisfies the condition $K \le 0$. Then, for every $p \in S$, the conjugate locus of $p$ is empty. In short, a surface of curvature $K \le 0$ does not have conjugate points.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $p \in S$ and let $\gamma\colon [0, l] \to S$ be a geodesic of $S$ with $\gamma(0) = p$. Assume that there exists a nonvanishing Jacobi field $J(s)$, with $J(0) = J(l) = 0$. We shall prove that this gives a contradiction.

In fact, since $J(s)$ is a Jacobi field and $J(0) = J(l) = 0$, we have, by the corollary of Prop. 4, that $\langle J(s), \gamma'(s) \rangle = 0$, $s \in [0, l]$. Therefore,

$$\frac{D^2J}{ds^2} + KJ = 0, \qquad \left\langle \frac{D^2J}{ds^2}, J \right\rangle = -K\langle J, J \rangle \ge 0,$$

since $K \le 0$. It follows that

$$\frac{d}{ds}\left\langle \frac{DJ}{ds}, J \right\rangle = \left\langle \frac{D^2J}{ds^2}, J \right\rangle + \left\langle \frac{DJ}{ds}, \frac{DJ}{ds} \right\rangle \ge 0.$$

Therefore, the function $\langle DJ/ds, J \rangle$ does not decrease in the interval $[0, l]$. Since this function is zero for $s = 0$ and $s = l$, we conclude that

$$\left\langle \frac{DJ}{ds}, J(s) \right\rangle = 0, \qquad s \in [0, l].$$

Finally, by observing that $\frac{d}{ds}\langle J, J \rangle = 2\langle DJ/ds, J \rangle = 0$, we have $\vert J\vert ^2 = \text{const.}$ Since $J(0) = 0$, we conclude that $\vert J(s)\vert  = 0$, $s \in [0, l]$; that is, $J$ is identically zero in $[0, l]$. This is a contradiction. **Q.E.D.**

</details>
</div>

**Corollary.** *Assume that the Gaussian curvature $K$ of $S$ is negative or zero. Then for every $p \in S$, the mapping $\exp\_p\colon T\_p(S) \to S$ is a local diffeomorphism.*

This follows immediately from the theorem and Prop. 5 (conjugate points = critical points of $\exp\_p$).

### Properties of Jacobi Fields

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Wronskian-type Identity for Jacobi Fields)</span></p>

Let $J\_1(s)$ and $J\_2(s)$ be Jacobi fields along $\gamma\colon [0, l] \to S$, $s \in [0, l]$. Then

$$\left\langle \frac{DJ_1}{ds}, J_2(s) \right\rangle - \left\langle J_1(s), \frac{DJ_2}{ds} \right\rangle = \text{const.}$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4</span><span class="math-callout__name">(Orthogonality of Jacobi Fields)</span></p>

Let $J(s)$ be a Jacobi field along $\gamma\colon [0, l] \to S$, with $\langle J(s\_1), \gamma'(s\_1) \rangle = \langle J(s\_2), \gamma'(s\_2) \rangle = 0$, $s\_1, s\_2 \in [0, l]$, $s\_1 \neq s\_2$. Then $\langle J(s), \gamma'(s) \rangle = 0$, $s \in [0, l]$.

</div>

**Corollary.** *Let $J(s)$ be a Jacobi field along $\gamma\colon [0, l] \to S$, with $J(0) = J(l) = 0$. Then $\langle J(s), \gamma'(s) \rangle = 0$, $s \in [0, l]$.*

### The Gauss Lemma (Generalized)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2</span><span class="math-callout__name">(Gauss Lemma — Generalized)</span></p>

Let $p \in S$ be a point of a (complete) surface $S$ and let $u \in T\_p(S)$ and $w \in (T\_p(S))\_u$. Then

$$\langle u, w \rangle = \langle (d\exp_p)_u(u),\; (d\exp_p)_u(w) \rangle,$$

where the identification $T\_p(S) \approx (T\_p(S))\_u$ is being used.

</div>

This is a generalization of the fact that, in a normal neighborhood of $p$, the geodesic circles are orthogonal to the radial geodesics (Sec. 4-6, Prop. 3 and Remark 1). It says that this orthogonality holds *globally* on a complete surface, even beyond conjugate points.

## 5-6. Covering Spaces; The Theorems of Hadamard

We saw in the last section that when the curvature $K$ of a complete surface $S$ satisfies the condition $K \le 0$ then the mapping $\exp\_p\colon T\_p(S) \to S$, $p \in S$, is a local diffeomorphism. It is natural to ask when this local diffeomorphism is a global diffeomorphism. It is convenient to put this question in a more general setting for which we need the notion of covering space.

### A. Covering Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Covering Map)</span></p>

Let $\tilde{B}$ and $B$ be subsets of $R^3$. We say that $\pi\colon \tilde{B} \to B$ is a **covering map** if

1. $\pi$ is continuous and $\pi(\tilde{B}) = B$.
2. Each point $p \in B$ has a neighborhood $U$ in $B$ (to be called a **distinguished neighborhood** of $p$) such that $\pi^{-1}(U) = \bigcup\_\alpha V\_\alpha$, where the $V\_\alpha$'s are pairwise disjoint open sets such that the restriction of $\pi$ to $V\_\alpha$ is a homeomorphism of $V\_\alpha$ onto $U$.

$\tilde{B}$ is then called a **covering space** of $B$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Plane Covering the Cylinder)</span></p>

Let $P \subset R^3$ be a plane. By fixing a point $q\_0 \in P$ and two orthogonal unit vectors $e\_1, e\_2 \in P$, with origin in $q\_0$, every point $q \in P$ is characterized by coordinates $(u, v)$ given by $q - q\_0 = ue\_1 + ve\_2$. Now let $S = \lbrace (x, y, z) \in R^3;\; x^2 + y^2 = 1 \rbrace$ be the right circular cylinder whose axis is the $z$ axis, and let $\pi\colon P \to S$ be the map defined by $\pi(u, v) = (\cos u, \sin u, v)$.

We shall prove that $\pi$ is a covering map. For each $(u\_0, v\_0) \in P$, the mapping $\pi$ restricted to the band $R = \lbrace (u, v) \in P;\; u\_0 - \pi \le u \le u\_0 + \pi \rbrace$ covers $S$ entirely. To verify condition 2, let $p \in S$ and $U = S - r$, where $r$ is the generator opposite to the generator passing through $p$. Then $\pi^{-1}(U) = \bigcup\_n V\_n$, where $V\_n = \lbrace (u, v) \in P;\; u\_0 + (2n-1)\pi < u < u\_0 + (2n+1)\pi \rbrace$, $n = 0, \pm 1, \pm 2, \dots$ The $V\_n$'s are pairwise disjoint, and $\pi$ restricted to any $V\_n$ is a homeomorphism onto $U$. This verifies condition 2 and shows that the plane $P$ is a covering space of the cylinder $S$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Helix Covering the Circle)</span></p>

Let $H$ be the helix $H = \lbrace (x, y, z) \in R^3;\; x = \cos t,\; y = \sin t,\; z = bt,\; t \in R \rbrace$ and let $S^1 = \lbrace (x, y, 0) \in R^3;\; x^2 + y^2 = 1 \rbrace$ be a unit circle. Let $\pi\colon H \to S^1$ be defined by $\pi(x, y, z) = (x, y, 0)$. Then $\pi$ is a covering map: for each $p \in S^1$, $U = S^1 - \lbrace q \rbrace$ (where $q$ is the point diametrically opposite to $p$) is a distinguished neighborhood.

</div>

### Local Homeomorphisms vs. Covering Maps

Not every local homeomorphism is a covering map. For instance, a segment $\tilde{H}$ of the helix $H$ corresponding to the interval $(\pi, 4\pi) \subset R$ projects onto $S^1$ by a local homeomorphism $\tilde{\pi}$, but no neighborhood of $\pi(\cos 3\pi, \sin 3\pi, b\cdot 3\pi) = (-1, 0, 0) \in S^1$ can be a distinguished neighborhood. So $\tilde{\pi}$ is a local homeomorphism but not a covering map.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Compact Local Homeomorphism is a Covering Map)</span></p>

Let $\pi\colon \tilde{B} \to B$ be a local homeomorphism, $\tilde{B}$ compact and $B$ connected. Then $\pi$ is a covering map.

</div>

### Lifting of Arcs and Homotopies

The most important property of a covering map is the possibility of "lifting" continuous curves of $B$ into $\tilde{B}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lifting of an Arc)</span></p>

Let $B \subset R^3$, $\pi\colon \tilde{B} \to B$ be a continuous map, and $\alpha\colon [0, l] \to B$ be an arc of $B$. If there exists an arc $\tilde{\alpha}\colon [0, l] \to \tilde{B}$ with $\pi \circ \tilde{\alpha} = \alpha$, $\tilde{\alpha}$ is said to be a **lifting** of $\alpha$ with origin at $\tilde{\alpha}(0) \in \tilde{B}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Uniqueness and Existence of Liftings)</span></p>

Let $\pi\colon \tilde{B} \to B$ be a covering map, $\alpha\colon [0, l] \to B$ an arc in $B$, and $\tilde{p}\_0 \in \tilde{B}$ a point of $\tilde{B}$ such that $\pi(\tilde{p}\_0) = \alpha(0) = p\_0$. Then there exists a unique lifting $\tilde{\alpha}\colon [0, l] \to \tilde{B}$ of $\alpha$ with origin at $\tilde{p}\_0$, that is, with $\tilde{\alpha}(0) = \tilde{p}\_0$.

</div>

An interesting consequence of the arc lifting property is that when $B$ is arcwise connected there exists a one-to-one correspondence between the sets $\pi^{-1}(p)$ and $\pi^{-1}(q)$, where $p$ and $q$ are two arbitrary points of $B$. If this number is finite, it is called the **number of sheets** of the covering.

### Homotopies and Simply Connected Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Homotopy)</span></p>

Let $B \subset R^3$ and let $\alpha\_0\colon [0, l] \to B$, $\alpha\_1\colon [0, l] \to B$ be two arcs of $B$, joining the points $p = \alpha\_0(0) = \alpha\_1(0)$ and $q = \alpha\_0(l) = \alpha\_1(l)$. We say that $\alpha\_0$ and $\alpha\_1$ are **homotopic** if there exists a continuous map $H\colon [0, l] \times [0, 1] \to B$ such that

1. $H(s, 0) = \alpha\_0(s)$, $H(s, 1) = \alpha\_1(s)$, $s \in [0, l]$.
2. $H(0, t) = p$, $H(l, t) = q$, $t \in [0, 1]$.

The map $H$ is called a **homotopy** between $\alpha\_0$ and $\alpha\_1$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Lifting of Homotopies)</span></p>

Let $\pi\colon \tilde{B} \to B$ be a local homeomorphism with the property of lifting arcs. Let $\alpha\_0, \alpha\_1\colon [0, l] \to B$ be two arcs of $B$ joining the points $p$ and $q$, let $H\colon [0, l] \times [0, 1] \to B$ be a homotopy between $\alpha\_0$ and $\alpha\_1$, and let $\tilde{p} \in \tilde{B}$ be such that $\pi(\tilde{p}) = p$. Then there exists a unique lifting $\tilde{H}\colon [0, l] \times [0, 1] \to \tilde{B}$ of $H$ with origin at $\tilde{p}$.

</div>

A consequence of Prop. 3 is that if $\pi\colon \tilde{B} \to B$ is a covering map, then homotopic arcs of $B$ are lifted into homotopic arcs of $\tilde{B}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Simply Connected)</span></p>

An arcwise connected set $B \subset R^3$ is **simply connected** if given two points $p, q \in B$ and two arcs $\alpha\_0\colon [0, l] \to B$, $\alpha\_1\colon [0, l] \to B$ joining $p$ to $q$, there exists a homotopy in $B$ between $\alpha\_0$ and $\alpha\_1$. In particular, any closed arc $\alpha\colon [0, l] \to B$ (closed means that $\alpha(0) = \alpha(l) = p$) is homotopic to the "constant" arc $\alpha(s) = p$, $s \in [0, l]$.

</div>

Intuitively, an arcwise connected set $B$ is simply connected if every closed arc in $B$ can be continuously deformed into a point. The plane and the sphere are simply connected, while the cylinder and the torus are not simply connected.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5</span><span class="math-callout__name">(Covering of a Simply Connected Space is a Homeomorphism)</span></p>

Let $\pi\colon \tilde{B} \to B$ be a local homeomorphism with the property of lifting arcs. Let $\tilde{B}$ be arcwise connected and $B$ simply connected. Then $\pi$ is a homeomorphism.

</div>

**Corollary.** *Let $\pi\colon \tilde{B} \to B$ be a covering map, $\tilde{B}$ arcwise connected, and $B$ simply connected. Then $\pi$ is a homeomorphism.*

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6</span><span class="math-callout__name">(Locally Simply Connected Local Homeomorphism is a Covering Map)</span></p>

Let $\pi\colon \tilde{B} \to B$ be a local homeomorphism with the property of lifting arcs. Assume that $B$ is locally simply connected and that $\tilde{B}$ is locally arcwise connected. Then $\pi$ is a covering map.

</div>

We remark that a regular surface is locally simply connected, since $p \in S$ has arbitrarily small neighborhoods homeomorphic to the interior of a disk in the plane. Thus, in the proposition below, the hypotheses on $B$ and $\tilde{B}$ are satisfied when both $B$ and $\tilde{B}$ are regular surfaces.

### B. The Hadamard Theorems

We shall now return to the question posed in the beginning of this chapter: namely, under what conditions is the local diffeomorphism $\exp\_p\colon T\_p(S) \to S$, where $p$ is a point of a complete surface $S$ of curvature $K \le 0$, a global diffeomorphism of $T\_p(S)$ onto $S$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1</span><span class="math-callout__name">($\exp\_p$ is Length-Increasing when $K \le 0$)</span></p>

Let $S$ be a complete surface of curvature $K \le 0$. Then $\exp\_p\colon T\_p(S) \to S$, $p \in S$, is length-increasing in the following sense: If $u, w \in T\_p(S)$, we have

$$\langle (d\exp_p)_u(w),\; (d\exp_p)_u(w) \rangle \ge \langle w, w \rangle,$$

where, as usual, $w$ denotes a vector in $(T\_p(S))\_u$ that is obtained from $w$ by the translation of vector $u$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For the case $u = 0$, the equality is trivially verified. Thus, let $v = u/\vert u\vert $, $u \neq 0$, and let $\gamma\colon [0, l] \to S$, $l = \vert u\vert $, be the geodesic $\gamma(s) = \exp\_p(sv)$, $s \in [0, l]$.

By the Gauss lemma, we may assume that $\langle w, v \rangle = 0$. Let $J(s) = s(d\exp\_p)\_{sv}(w)$ be the Jacobi field along $\gamma$ given by Lemma 1 of Sec. 5-5. We know that $J(0) = 0$, $(DJ/ds)(0) = w$, and $\langle J(s), \gamma'(s) \rangle = 0$, $s \in [0, l]$.

Observe now that, since $K \le 0$ (cf. Eq. (1), Sec. 5-5),

$$\frac{d}{ds}\left\langle J, \frac{DJ}{ds} \right\rangle = \left\langle \frac{DJ}{ds}, \frac{DJ}{ds} \right\rangle + \left\langle J, \frac{D^2J}{ds^2} \right\rangle = \left|\frac{DJ}{ds}\right|^2 - K\langle J, J \rangle \ge 0.$$

This implies that $\langle J, DJ/ds \rangle \ge 0$; hence, $\frac{d}{ds}\langle DJ/ds, DJ/ds \rangle = -2K\langle DJ/ds, J \rangle \ge 0$, giving $\vert DJ/ds\vert ^2 \ge \vert DJ/ds(0)\vert ^2 = \langle w, w \rangle = C$. Thus,

$$\frac{d^2}{ds^2}\langle J, J \rangle = 2\left\langle \frac{DJ}{ds}, \frac{DJ}{ds} \right\rangle + 2\left\langle J, \frac{D^2J}{ds^2} \right\rangle \ge 2\left\langle \frac{DJ}{ds}, \frac{DJ}{ds} \right\rangle \ge 2C.$$

By integrating both sides, $\frac{d}{ds}\langle J, J \rangle \ge 2Cs$, and integrating again, $\langle J, J \rangle \ge Cs^2$. By setting $s = l$ in the above expression and noticing that $C = \langle w, w \rangle$, we obtain

$$\langle J(l), J(l) \rangle \ge l^2 \langle w, w \rangle.$$

Since $J(l) = l(d\exp\_p)\_{lv}(w) = l(d\exp\_p)\_u(w)$, we finally conclude that

$$\langle (d\exp_p)_u(w),\; (d\exp_p)_u(w) \rangle \ge \langle w, w \rangle. \qquad \textbf{Q.E.D.}$$

</details>
</div>

**Corollary** *(of the proof)*. Let $K \equiv 0$. Then $\exp\_p\colon T\_p(S) \to S$, $p \in S$, is a local isometry.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7</span><span class="math-callout__name">($\exp\_p$ is a Covering Map when $K \le 0$)</span></p>

Let $S$ be a complete surface with Gaussian curvature $K \le 0$. Then the map $\exp\_p\colon T\_p(S) \to S$, $p \in S$, is a covering map.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Hadamard — Nonpositive Curvature)</span></p>

Let $S$ be a simply connected, complete surface with Gaussian curvature $K \le 0$. Then $\exp\_p\colon T\_p(S) \to S$, $p \in S$, is a diffeomorphism; that is, $S$ is diffeomorphic to a plane.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By Prop. 7, $\exp\_p\colon T\_p(S) \to S$ is a covering map. By the corollary of Prop. 5, $\exp\_p$ is a homeomorphism. Since $\exp\_p$ is a local diffeomorphism, its inverse map is differentiable, and $\exp\_p$ is a diffeomorphism. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2</span><span class="math-callout__name">(Hadamard — Ovaloids)</span></p>

Let $S$ be an ovaloid (i.e., a connected, compact, regular surface with $K > 0$). Then the Gauss map $N\colon S \to S^2$ is a diffeomorphism. In particular, $S$ is diffeomorphic to a sphere.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since for every $p \in S$ the Gaussian curvature $K = \det(dN\_p)$ is positive, $N$ is a local diffeomorphism. By Prop. 1, $N$ is a covering map. Since $S^2$ is simply connected, we conclude from the corollary of Prop. 5 that $N\colon S \to S^2$ is a homeomorphism of $S$ onto the unit sphere $S^2$. Since $N$ is a local diffeomorphism, its inverse map is differentiable. Therefore, $N$ is a diffeomorphism. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Consequences of Hadamard's Theorems)</span></p>

Since the Gauss map $N$ is a diffeomorphism, each unit vector $v$ of $R^3$ appears exactly once as a unit normal vector to $S$. Taking a plane normal to $v$, away from the surface, and displacing it parallel to itself until it meets the surface, we conclude that $S$ lies on one side of each of its tangent planes. This is expressed by saying that an ovaloid is **locally convex**. It can be proved from this that $S$ is actually the boundary of a convex set (i.e., a set $K \subset R^3$ such that the line segment joining any two points $p, q \in K$ belongs entirely to $K$).

The fact that compact surfaces with $K > 0$ are homeomorphic to spheres was extended to compact surfaces with $K \ge 0$ by S. S. Chern and R. K. Lashof and, for complete surfaces with $K > 0$, by J. J. Stoker, who proved: *A complete surface with $K > 0$ is homeomorphic to a sphere or a plane.*

</div>

## 5-7. Global Theorems for Curves; The Fary-Milnor Theorem

In this section, some global theorems for closed curves will be presented. The main tool used here is the degree theory for continuous maps of the circle, which uses some properties of covering maps developed in Sec. 5-6.

### The Degree of a Map of the Circle

Let $S^1 = \lbrace (x, y) \in R^2;\; x^2 + y^2 = 1 \rbrace$ and let $\pi\colon R \to S^1$ be the covering of $S^1$ by the real line $R$ given by $\pi(x) = (\cos x, \sin x)$, $x \in R$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Degree of a Map $S^1 \to S^1$)</span></p>

Let $\varphi\colon S^1 \to S^1$ be a continuous map. The **degree** of $\varphi$ is defined as follows. Think of $\varphi$ as a continuous map $\bar{\varphi}\colon [0, l] \to S^1$, with $\bar{\varphi}(0) = \bar{\varphi}(l) = p \in S^1$. This closed arc $\bar{\varphi}$ can be lifted into a unique arc $\tilde{\varphi}\colon [0, l] \to R$, starting at a point $x \in R$ with $\pi(x) = p$. Since $\pi(\tilde{\varphi}(0)) = \pi(\tilde{\varphi}(l))$, the difference $\tilde{\varphi}(l) - \tilde{\varphi}(0)$ is an integral multiple of $2\pi$. The integer $\deg\varphi$ given by

$$\tilde{\varphi}(l) - \tilde{\varphi}(0) = (\deg\varphi)2\pi$$

is called the **degree** of $\varphi$.

</div>

The definition of degree is independent of the choices of $p$ and $x$. The most important property of degree is its **invariance under homotopy**: if $\varphi\_1$ and $\varphi\_2$ are homotopic, then $\deg\varphi\_1 = \deg\varphi\_2$.

In the differentiable case, if $\varphi = (a(t), b(t))$ satisfies $a^2 + b^2 = 1$, then the lifting is $\tilde{\varphi}(t) = \tilde{\varphi}\_0 + \int\_0^t (ab' - ba')\,dt$, and the degree can be expressed by the integral

$$\deg\varphi = \frac{1}{2\pi}\int_0^l \frac{d\tilde{\varphi}}{dt}\,dt.$$

### Winding Number and Rotation Index

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Winding Number)</span></p>

Let $\alpha\colon [0, l] \to R^2$ be a plane, continuous closed curve. Choose a point $p\_0 \in R^2$, $p\_0 \notin \alpha([0, l])$, and let $\varphi\colon [0, l] \to S^1$ be given by

$$\varphi(t) = \frac{\alpha(t) - p_0}{|\alpha(t) - p_0|}, \qquad t \in [0, l].$$

Clearly $\varphi(0) = \varphi(l)$, and $\varphi$ may be thought of as a map of $S^1$ into $S^1$; it is called the **position map** of $\alpha$ relative to $p\_0$. The degree of $\varphi$ is called the **winding number** (or the **index**) of the curve $\alpha$ relative to $p\_0$.

The winding number is constant when $q$ runs in a connected component of $R^2 - \alpha([0, l])$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Rotation Index)</span></p>

Let $\alpha\colon [0, l] \to R^2$ be a regular plane closed curve, and let $\varphi\colon [0, l] \to S^1$ be given by

$$\varphi(t) = \frac{\alpha'(t)}{|\alpha'(t)|}, \qquad t \in [0, l].$$

$\varphi$ is called the **tangent map** of $\alpha$, and the degree of $\varphi$ is called the **rotation index** of $\alpha$. Intuitively, the rotation index is the number of complete turns given by the tangent vector field along the curve.

</div>

### The Differentiable Jordan Curve Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Differentiable Jordan Curve Theorem)</span></p>

Let $\alpha\colon [0, l] \to R^2$ be a plane, regular, closed, simple curve. Then $R^2 - \alpha([0, l])$ has exactly two connected components, and $\alpha([0, l])$ is their common boundary.

</div>

The two connected components given by Theorem 1 can easily be distinguished. If $p\_0$ is outside a closed disk $D$ containing $\alpha([0, l])$, then the winding number of $\alpha$ relative to $p\_0$ is zero. Thus, the connected component with winding number zero is unbounded and contains all points outside a certain disk. It is usual to call the remaining connected component (which has winding number $\pm 1$) the **interior** of $\alpha$, and the unbounded component the **exterior** of $\alpha$.

### The Theorem of Turning Tangents (Differentiable Version)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2</span><span class="math-callout__name">(Theorem of Turning Tangents)</span></p>

Let $\beta\colon [0, l] \to R^2$ be a plane, regular, simple, closed curve. Then the rotation index of $\beta$ is $\pm 1$ (depending on the orientation of $\beta$).

</div>

### Convex Curves

A plane, regular, closed curve $\alpha\colon [0, l] \to R^2$ is **convex** if, for each $t \in [0, l]$, the curve lies in one of the closed half-planes determined by the tangent line at $t$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Characterization of Convex Curves)</span></p>

A plane, regular, closed curve is convex if and only if it is simple and its curvature $k$ does not change sign.

</div>

### Fenchel's Theorem and the Fary-Milnor Theorem

We now turn our attention to space curves. In what follows the word *curve* will mean a parametrized regular curve $\alpha\colon [0, l] \to R^3$ with arc length $s$ as parameter. The **total curvature** of $\alpha$ is $\int\_0^l \vert k(s)\vert \,ds$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3</span><span class="math-callout__name">(Fenchel's Theorem)</span></p>

The total curvature of a simple closed curve is $\ge 2\pi$, and equality holds if and only if the curve is a plane convex curve.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Sketch</summary>

The **tube** of radius $r$ around the curve $\alpha$ is the parametrized surface

$$\mathbf{x}(s, v) = \alpha(s) + r(n\cos v + b\sin v), \qquad s \in [0, l],\; v \in [0, 2\pi],$$

where $n = n(s)$ and $b = b(s)$ are the normal and the binormal vector of $\alpha$, respectively. It is easily checked that $\vert \mathbf{x}\_s \wedge \mathbf{x}\_v\vert  = r^2(1 - rk\cos v)^2$. For $r$ sufficiently small, the trace $T$ of $\mathbf{x}$ is a regular surface (if $\alpha$ is simple, $T$ is a torus).

The Gaussian curvature of the tube is $K(s, v) = -k\cos v / (r(1 - rk\cos v))$. Let $R \subset T$ be the region of $T$ where $K \ge 0$. Then

$$\iint_R K\,d\sigma = \int_0^l k\,ds \int_{\pi/2}^{3\pi/2} \cos v\,dv = 2\int_0^l k(s)\,ds.$$

On the other hand, since $R$ has nonnegative curvature and every half-line $L$ through the origin of $R^3$ meets $T$ at a point where $K \ge 0$ (by moving a plane perpendicular to $L$ toward $T$), the Gauss map $N$ of $R$ covers $S^2$ at least once; hence, $\iint\_R K\,d\sigma \ge 4\pi$. Therefore, the total curvature of $\alpha$ is $\ge 2\pi$.

If the total curvature equals $2\pi$, the Gauss map covers $S^2$ exactly once, which implies that all $\Gamma\_s^+$ (the half-great-circles corresponding to $K \ge 0$ at each $s$) have the same end points. This forces $\alpha$ to be a plane convex curve. **Q.E.D.**

</details>
</div>

A simple closed continuous curve $C \subset R^3$ is **unknotted** if there exists a homotopy $H\colon S^1 \times I \to R^3$, $I = [0, 1]$, such that $H(S^1 \times \lbrace 0 \rbrace) = S^1$, $H(S^1 \times \lbrace 1 \rbrace) = C$, and $H(S^1 \times \lbrace t \rbrace) = C\_t \subset R^3$ is homeomorphic to $S^1$ for all $t \in [0, 1]$. Such a homotopy is called an **isotopy**; an unknotted curve is then a curve isotopic to $S^1$. When this is not the case, $C$ is said to be **knotted**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4</span><span class="math-callout__name">(Fary-Milnor)</span></p>

The total curvature of a knotted simple closed curve is greater than or equal to $4\pi$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Sketch</summary>

Let $C = \alpha([0, l])$, let $T$ be a tube around $\alpha$, and let $R \subset T$ be the region where $K \ge 0$. Let $b = b(s)$ be the binormal vector of $\alpha$, and let $v \in R^3$ be a unit vector, $v \neq b(s)$ for all $s \in [0, l]$. Let $h\_v\colon [0, l] \to R$ be the height function of $\alpha$ in the direction of $v$; that is, $h\_v(s) = \langle \alpha(s) - 0, v \rangle$. Clearly, $s$ is a critical point of $h\_v$ if and only if $v$ is perpendicular to the tangent line at $\alpha(s)$. Furthermore, at a critical point, $d^2/ds^2(h\_v) = k\langle n, v \rangle \neq 0$, since $v \neq b(s)$ for all $s$ and $k > 0$. Thus, the critical points of $h\_v$ are either maxima or minima.

Now, assume that the total curvature of $\alpha$ is smaller than $4\pi$. This means that $\iint\_R K\,d\sigma < 8\pi$. We claim that, for some $v\_0 \notin b([0, l])$, $h\_{v\_0}$ has exactly two critical points (which correspond to the maximum and minimum of $h\_{v\_0}$). If this were not the case, then for every $v \notin b([0, l])$, $h\_v$ has at least three critical points, hence at least three minima (which, together with maxima, give at least six critical points overall). By an argument involving the Gauss map on the tube, this leads to $\iint\_R K\,d\sigma \ge 8\pi$, a contradiction.

Since $h\_{v\_0}$ has exactly two critical points $s\_1$ and $s\_2$, corresponding to the max and min, the planes perpendicular to $v\_0$ between these two values each meet $C$ in exactly two points. Joining these pairs of points by line segments, we generate a surface bounded by $C$ which is homeomorphic to a disk. Thus, $C$ is unknotted, and this contradiction completes the proof. **Q.E.D.**

</details>
</div>

## 5-8. Surfaces of Zero Gaussian Curvature

We have already seen (Sec. 4-6) that the regular surfaces with identically zero Gaussian curvature are locally isometric to the plane. In this section, we shall look upon such surfaces from the point of view of their position in $R^3$ and prove the following global theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Complete Surfaces of Zero Curvature are Cylinders or Planes)</span></p>

Let $S \subset R^3$ be a complete surface with zero Gaussian curvature. Then $S$ is a cylinder or a plane.

</div>

By definition, a **cylinder** is a regular surface $S$ such that through each point $p \in S$ there passes a unique line $R(p) \subset S$ (the generator through $p$) which satisfies the condition that if $q \neq p$, then the lines $R(p)$ and $R(q)$ are parallel or equal.

### Local Structure

Let $S \subset R^3$ be a regular surface with Gaussian curvature $K \equiv 0$. Since $K = k\_1 k\_2$, where $k\_1$ and $k\_2$ are the principal curvatures, the points of $S$ are either parabolic or planar. We denote by $P$ the set of planar points and by $U = S - P$ the set of parabolic points.

$P$ is closed in $S$ (the points of $P$ satisfy the condition that the mean curvature $H = \frac{1}{2}(k\_1 + k\_2)$ is zero, which is a closed condition). Therefore, $U = S - P$ is open in $S$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1</span><span class="math-callout__name">(Surface of Zero Curvature with Planar and Parabolic Points)</span></p>

Consider an open triangle $ABC$ and attach to each side a cylindrical surface with generators parallel to the given side. For instance, along the open segment $BC$, attach a cylindrical band $BCDE$ whose generators are parallel to $BC$. To ensure regularity along $BC$, it suffices that the section $FG$ of the cylindrical band by a plane normal to $BC$ is a curve of the form $\exp(-1/x^2)$.

The vertices $A$, $B$, $C$ of the triangle and the edges $BE$, $CD$, etc., of the cylindrical bands do not belong to $S$. The surface $S$ so constructed has curvature $K \equiv 0$. The set $P$ is the closed triangle $ABC$ minus its vertices. Observe that $P$ is closed in $S$ but not in $R^3$. The set $U$ is formed by the points which are interior to the cylindrical bands. Through each point of $U$ there passes a unique line which will never meet $P$. The boundary of $P$ is formed by the open segments $AB$, $BC$, and $CA$.

</div>

In the following, we shall prove that the relevant properties of this example appear in the general case. First, let $p \in U$. Since $p$ is a parabolic point, one of the principal directions at $p$ is an asymptotic direction, and there is no other asymptotic direction at $p$. We shall prove that the unique asymptotic curve that passes through $p$ is a segment of a straight line.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Asymptotic Lines are Straight Lines)</span></p>

The unique asymptotic line that passes through a parabolic point $p \in U \subset S$ of a surface $S$ of curvature $K \equiv 0$ is an (open) segment of a (straight) line in $S$.

</div>

An important intermediate result concerns the maximal asymptotic lines through points of $U$. By parametrizing the asymptotic line $r$ through $q \in U$ by arc length, we see that $r$ is a geodesic and $r \cap P = \phi$. By completeness, $r$ extends to an entire line $R(q)$ in $S$, with $R(q) \subset U$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Maximal Asymptotic Lines are Complete Lines)</span></p>

On a complete surface $S$ of curvature $K \equiv 0$, through each parabolic point $q \in U$ there passes a unique complete line $R(q) \subset S$ (the maximal asymptotic line through $q$), and $R(q) \subset U$. Furthermore, the segment $C(p)$ does not contain its extreme points.

</div>

We also have that $\mathrm{Bd}(U) = \mathrm{Bd}(P)$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Massey)</span></p>

Let $p \in \mathrm{Bd}(U) \subset S$ be a point of the boundary of the set $U$ of parabolic points of a surface $S$ of curvature $K \equiv 0$. Then through $p$ there passes a unique open segment of line $C(p) \subset S$. Furthermore, $C(p) \subset \mathrm{Bd}(U)$; that is, the boundary of $U$ is formed by segments of lines.

</div>

### Proof of the Theorem

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Complete Surfaces of Zero Curvature are Cylinders or Planes)</summary>

Assume that $S$ is not a plane. Then (Sec. 3-2, Prop. 5) $S$ contains parabolic points. Let $U$ be the (open) set of parabolic points of $S$ and $P$ be the (closed) set of planar points of $S$. We shall denote by $\mathrm{int}\, P$ the interior of $P$, the set of points which have a neighborhood entirely contained in $P$. Since $\mathrm{int}\, P$ is an open set in $S$ which contains only planar points, each connected component of $\mathrm{int}\, P$ is contained in a plane (Sec. 3-2, Prop. 5).

We shall first prove that if $q \in S$ and $q \notin \mathrm{int}\, P$, then through $q$ there passes a unique line $R(q) \subset S$, and two such lines are either equal or do not intersect.

When $q \in U$, there exists a unique maximal asymptotic line $r$ passing through $q$, which is a geodesic with $r \cap P = \phi$ (cf. Props. 1 and 2). By completeness, $r$ is an entire line $R(q)$, and $R(q) \subset U$. When $q \in \mathrm{Bd}(U) = \mathrm{Bd}(P)$, then (cf. Prop. 3) through $q$ there passes a unique open segment of line which is contained in $\mathrm{Bd}(U)$. By the same completeness argument, this segment extends to an entire line $R(q)$, and if $p \in \mathrm{Bd}(U)$, $p \notin R(q)$, then $R(p) \cap R(q) = \phi$.

In this way, through each point of $S - \mathrm{int}\, P = U \cup \mathrm{Bd}(U)$ there passes a unique line contained in $S - \mathrm{int}\, P$, and two such lines are either equal or do not intersect. If we prove that these lines are parallel, we shall conclude that $\mathrm{Bd}(U)$ ($= \mathrm{Bd}(P)$) is formed by parallel lines and that each connected component of $\mathrm{int}\, P$ is an open set of a plane, bounded by two parallel lines. Thus, through each point $t \subset \mathrm{int}\, P$ parallel to the common direction, and $R(t) \subset \mathrm{int}\, P$. It follows that through each point of $S$ there passes a unique generator and that the generators are parallel, that is, $S$ is a cylinder, as we wish.

To prove that the lines passing through the points of $U \cup \mathrm{Bd}(U)$ are parallel, we proceed as follows. Since $S$ is connected, there exists an arc $\alpha\colon [0, l] \to S$, with $\alpha(0) = q$, $\alpha(l) = p$. The map $\exp\_p\colon T\_p(S) \to S$ is a covering map and a local isometry. Let $\tilde{\alpha}\colon [0, l] \to T\_p(S)$ be the lifting of $\alpha$, with origin at the origin $0 \in T\_p(S)$. For each $\tilde{\alpha}(t)$, with $\exp\_p\, \tilde{\alpha}(t) = \alpha(t) \in U \cup \mathrm{Bd}(U)$, let $r\_t$ be the lifting of $R(\alpha(t))$ with origin at $\tilde{\alpha}(t)$. Since $\exp\_p$ is a local isometry, $r$ is a line in $T\_p(S)$.

By covering the interval $[0, l]$ with finitely many neighborhoods on which $\exp\_p$ is an isometry, one proves step by step that the lines $R(\alpha(t))$, $t \in [0, l]$, are all parallel. In particular, the line $R(q)$ is parallel to $R(p)$. If $s$ is another point in $U \cup \mathrm{Bd}(U)$, then by the same argument, $R(s)$ is parallel to $R(p)$ and hence parallel to $R(q)$. In this way, it is proved that all the lines that pass through $U \cup \mathrm{Bd}(U)$ are parallel, and this concludes the proof of the theorem. **Q.E.D.**

</details>
</div>

## 5-9. Jacobi's Theorems

It is a fundamental property of a geodesic $\gamma$ (Sec. 4-6, Prop. 4) that when two points $p$ and $q$ of $\gamma$ are sufficiently close, then $\gamma$ minimizes the arc length between $p$ and $q$. Suppose now that we follow a geodesic $\gamma$ starting from a point $p$. It is then natural to ask how far the geodesic $\gamma$ minimizes arc length. In the case of a sphere, for instance, a geodesic (a meridian) starting from a point $p$ minimizes arc length up to the first conjugate point of $p$ relative to $\gamma$ (that is, up to the antipodal point of $p$). Past the antipodal point of $p$, the geodesic stops being minimal.

For simplicity, the surfaces in this section are assumed to be complete and the geodesics are parametrized by arc length.

### Preliminary Results

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1</span><span class="math-callout__name">(Arc Length Comparison)</span></p>

Let $p \in S$, $u \in T\_p(S)$, $l = \|u\| \neq 0$, and let $\tilde{\gamma}\colon [0, l] \to T\_p(S)$ be the line of $T\_p(S)$ given by $\tilde{\gamma}(s) = sv$, $s \in [0, l]$, $v = u/\vert u\vert $. Let $\tilde{\alpha}\colon [0, l] \to T\_p(S)$ be a differentiable parametrized curve of $T\_p(S)$, with $\tilde{\alpha}(0) = 0$, $\tilde{\alpha}(l) = u$, and $\tilde{\alpha}(s) \neq 0$ if $s \neq 0$. Furthermore, let

$$\alpha(s) = \exp_p \tilde{\alpha}(s) \quad \text{and} \quad \gamma(s) = \exp_p \tilde{\gamma}(s).$$

Then:

1. $l(\alpha) \ge l(\gamma)$, where $l(\cdot)$ denotes the arc length of the corresponding curve.

2. In addition, if $\tilde{\alpha}(s)$ is not a critical point of $\exp\_p$, $s \in [0, l]$, and if the traces of $\alpha$ and $\gamma$ are distinct, then $l(\alpha) > l(\gamma)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 1</summary>

Let $\tilde{\alpha}(s)/\|\tilde{\alpha}(s)\| = r$, and let $n$ be a unit vector of $T\_p(S)$ with $\langle r, n \rangle = 0$. In the basis $\lbrace r, n \rbrace$ of $T\_p(S)$ we can write $\tilde{\alpha}'(s) = ar + bn$, where $a = \langle \tilde{\alpha}'(s), r \rangle$ and $b = \langle \tilde{\alpha}'(s), n \rangle$.

By definition,

$$\alpha'(s) = (d\exp_p)_{\tilde{\alpha}(s)}(\tilde{\alpha}'(s)) = a(d\exp_p)_{\tilde{\alpha}(s)}(r) + b(d\exp_p)_{\tilde{\alpha}(s)}(n).$$

By using the Gauss lemma (cf. Sec. 5-5, Lemma 2) we obtain $\langle \alpha'(s), \alpha'(s) \rangle = a^2 + c^2$, where $c^2 = b^2 \vert (d\exp\_p)\_{\tilde{\alpha}(s)}(n)\vert ^2$. It follows that $\langle \alpha'(s), \alpha'(s) \rangle \ge a^2$.

On the other hand, $\frac{d}{ds}\langle \tilde{\alpha}(s), \tilde{\alpha}(s) \rangle^{1/2} = \langle \tilde{\alpha}'(s), r \rangle = a$. Therefore,

$$l(\alpha) = \int_0^l \langle \alpha'(s), \alpha'(s) \rangle^{1/2}\,ds \ge \int_0^l a\,ds = \int_0^l \frac{d}{ds}\langle \tilde{\alpha}(s), \tilde{\alpha}(s) \rangle^{1/2}\,ds = |\tilde{\alpha}(l)| = l = l(\gamma),$$

and this proves part 1.

To prove part 2, let us assume that $l(\alpha) = l(\gamma)$. Then $\int\_0^l \langle \alpha'(s), \alpha'(s) \rangle^{1/2}\,ds = \int\_0^l a\,ds$, and since $\langle \alpha'(s), \alpha'(s) \rangle^{1/2} \ge a$, the equality must hold for every $s \in [0, l]$. Therefore, $c = \|b\|\,\|(d\exp\_p)\_{\tilde{\alpha}(s)}(n)\| = 0$. Since $\tilde{\alpha}(s)$ is not a critical point of $\exp\_p$, we conclude that $b \equiv 0$. It follows that the tangent lines to the curve $\tilde{\alpha}$ all pass through the origin $O$ of $T\_p(S)$. Thus, $\tilde{\alpha}$ is a line of $T\_p(S)$ which passes through $O$. Since $\tilde{\alpha}(l) = \tilde{\gamma}(l)$, the lines $\tilde{\alpha}$ and $\tilde{\gamma}$ coincide, thus contradicting the assumption that the traces of $\alpha$ and $\gamma$ are distinct. From this contradiction it follows that $l(\alpha) > l(\gamma)$, which proves part 2. **Q.E.D.**

</details>
</div>

### Jacobi's First Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Jacobi — Geodesics without Conjugate Points Minimize Arc Length)</span></p>

Let $\gamma\colon [0, l] \to S$, $\gamma(0) = p$, be a geodesic without conjugate points; that is, $\exp\_p\colon T\_p(S) \to S$ is regular at the points of the line $\tilde{\gamma}(s) = s\gamma'(0)$ of $T\_p(S)$, $s \in [0, l]$. Let $h\colon [0, l] \times (-\epsilon, \epsilon) \to S$ be a proper variation of $\gamma$. Then

1. There exists a $\delta > 0$, $\delta \le \epsilon$, such that if $t \in (-\delta, \delta)$,

$$L(t) \ge L(0),$$

where $L(t)$ is the length of the curve $h\_t\colon [0, l] \to S$ given by $h\_t(s) = h(s, t)$.

2. If, in addition, the trace of $h\_t$ is distinct from the trace of $\gamma$, $L(t) > L(0)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Sketch</summary>

The proof consists of lifting the variation curves $h\_t$ into curves $\tilde{h}\_t$ of $T\_p(S)$ such that $\tilde{h}\_t(0) = 0$, $\tilde{h}\_t(l) = \tilde{\gamma}(l)$, and then applying Lemma 1 to this situation.

Since $\exp\_p$ is regular at the points of the line $\tilde{\gamma}$ of $T\_p(S)$, for each $s \in [0, l]$ there exists a neighborhood $U\_s$ of $\tilde{\gamma}(s)$ such that $\exp\_p$ restricted to $U\_s$ is a diffeomorphism. By compactness, a finite subfamily $U\_1, \ldots, U\_n$ covers $\tilde{\gamma}([0, l])$. It follows that we may divide the interval $[0, l]$ by points $0 = s\_1 < s\_2 < \cdots < s\_n < s\_{n+1} = l$ in such a way that $\tilde{\gamma}([s\_i, s\_{i+1}]) \subset U\_i$. For $\delta$ sufficiently small, the curves $h\_t$, $t \in (-\delta, \delta)$, may be lifted into curves $\tilde{h}\_t\colon [0, l] \to T\_p(S)$ with $\tilde{h}\_t(0) = 0$. By the covering space technique (cf. Prop. 2, Sec. 5-6), we can extend $\tilde{h}\_t$ for all $s \in [0, l]$ and obtain $\tilde{h}\_t(l) = \tilde{\gamma}(l)$. We then apply Lemma 1 to obtain the desired conclusions. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1</span></p>

A geodesic $\gamma$ containing no conjugate points may well not be minimal relative to the curves which are not in a neighborhood of $\gamma$. Such a situation occurs, for instance, in the cylinder (which has no conjugate points), as the reader will easily verify by observing a closed geodesic of the cylinder. This is related to the fact that conjugate points inform us only about the differential of the exponential map, that is, about the rate of "spreading out" of the geodesics neighboring a given geodesic. The global behavior of the geodesics is controlled by the exponential map itself, which may not be globally one-to-one even when its differential is nonsingular everywhere.

</div>

### Jacobi's Second Theorem

We shall now prove that a geodesic $\gamma$ containing conjugate points is not a local minimum for the arc length; that is, "arbitrarily near" to $\gamma$ there exists a curve, joining its extreme points, the length of which is smaller than that of $\gamma$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1</span><span class="math-callout__name">(Broken Variation of a Geodesic)</span></p>

Let $\gamma\colon [0, l] \to S$ be a geodesic and let

$$h\colon [0, l] \times (-\epsilon, \epsilon) \to S$$

be a continuous map with $h(s, 0) = \gamma(s)$, $s \in [0, l]$. $h$ is said to be a **broken variation** of $\gamma$ if there exists a partition $0 = s\_0 < s\_1 < s\_2 < \cdots < s\_{n-1} < s\_n = l$ of $[0, l]$ such that $h\colon [s\_i, s\_{i+1}] \times (-\epsilon, \epsilon) \to S$, $i = 0, 1, \ldots, n-1$, is differentiable. The broken variation is said to be **proper** if $h(0, t) = \gamma(0)$, $h(l, t) = \gamma(l)$ for every $t \in (-\epsilon, \epsilon)$.

The curves $h\_t(s)$, $s \in [0, l]$, of the variation are now piecewise differentiable curves. The variational vector field $V(s) = (\partial h/\partial t)(s, 0)$ is a piecewise differentiable vector field along $\gamma$; that is, $V\colon [0, l] \to R^3$ is a continuous map, differentiable in each $[t\_i, t\_{i+1}]$. The broken variation $h$ is said to be **orthogonal** if $\langle V(s), \gamma'(s) \rangle = 0$, $s \in [0, l]$.

</div>

The second variation of the arc length $L''(0)$ for proper and orthogonal broken variations is given by

$$L''(0) = \int_0^l \left( \left\langle \frac{DV}{ds}, \frac{DV}{ds} \right\rangle - K(s)\langle V(s), V(s) \rangle \right) ds.$$

This defines the **index form** $I(V, V)$ of the geodesic $\gamma$.

Let $\gamma\colon [0, l] \to S$ be a geodesic and let us denote by $\mathcal{V}$ the set of piecewise differentiable vector fields along $\gamma$ which are orthogonal to $\gamma$; that is, if $V \in \mathcal{V}$, then $\langle V(s), \gamma'(s) \rangle = 0$ for all $s \in [0, l]$. Define a map $I\colon \mathcal{V} \times \mathcal{V} \to R$ by

$$I(V, W) = \int_0^l \left( \left\langle \frac{DV}{ds}, \frac{DW}{ds} \right\rangle - K(s)\langle V(s), W(s) \rangle \right) ds,$$

where $V, W \in \mathcal{V}$. It is immediate to verify that $I$ is a symmetric bilinear map; that is, $I$ is linear in each variable and $I(V, W) = I(W, V)$. Therefore, $I$ determines a quadratic form in $\mathcal{V}$, given by $I(V, V)$. This quadratic form is called the **index form** of $\gamma$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2</span><span class="math-callout__name">(Index Form Identity)</span></p>

Let $V \in \mathcal{V}$ be a Jacobi field along a geodesic $\gamma\colon [0, l] \to S$ and $W \in \mathcal{V}$. Then

$$I(V, W) = \left\langle \frac{DV}{ds}(l),\; W(l) \right\rangle - \left\langle \frac{DV}{ds}(0),\; W(0) \right\rangle.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By observing that

$$\frac{d}{ds}\left\langle \frac{DV}{ds}, W \right\rangle = \left\langle \frac{D^2V}{ds^2}, W \right\rangle + \left\langle \frac{DV}{ds}, \frac{DW}{ds} \right\rangle,$$

we may write $I$ in the form (cf. Remark 4, Sec. 5-4):

$$I(V, W) = \left\langle \frac{DV}{ds}, W \right\rangle \bigg|_0^l - \int_0^l \left\langle \left( \frac{D^2V}{ds^2} + K(s)V(s) \right),\; W(s) \right\rangle ds.$$

From the fact that $V$ is a Jacobi field orthogonal to $\gamma$, we conclude that the integrand of the second term is zero. Therefore,

$$I(V, W) = \left\langle \frac{DV}{ds}(l),\; W(l) \right\rangle - \left\langle \frac{DV}{ds}(0),\; W(0) \right\rangle. \qquad \textbf{Q.E.D.}$$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2</span><span class="math-callout__name">(Jacobi — Conjugate Points and Non-Minimality)</span></p>

If we let $\gamma\colon [0, l] \to S$ be a geodesic of $S$ and we let $\gamma(s\_0) \in \gamma((0, l))$ be a point conjugate to $\gamma(0) = p$ relative to $\gamma$, then there exists a proper broken variation $h\colon [0, l] \times (-\epsilon, \epsilon) \to S$ of $\gamma$ and a real number $\delta > 0$, $\delta \le \epsilon$, such that if $t \in (-\delta, \delta)$, $t \neq 0$, we have $L(t) < L(0)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Sketch</summary>

Since $\gamma(s\_0)$ is conjugate to $p$ relative to $\gamma$, there exists a Jacobi field $J$ along $\gamma$, not identically zero, with $J(0) = J(s\_0) = 0$. Furthermore, $(DJ/ds)(s\_0) \neq 0$; otherwise, $J(s) \equiv 0$.

Let $\tilde{Z}$ be a parallel vector field along $\gamma$, with $\tilde{Z}(s\_0) = -(DJ/ds)(s\_0)$, and let $f\colon [0, l] \to R$ be a differentiable function with $f(0) = f(l) = 0$, $f(s\_0) = 1$. Define $Z(s) = f(s)\tilde{Z}(s)$, $s \in [0, l]$.

For each real number $\eta > 0$, define a vector field $Y\_\eta$ along $\gamma$ by

$$Y_\eta = \begin{cases} J(s) + \eta Z(s), & s \in [0, s_0], \\ \eta Z(s), & s \in [s_0, l]. \end{cases}$$

$Y\_\eta$ is a piecewise differentiable vector field orthogonal to $\gamma$. Since $Y\_\eta(0) = Y\_\eta(l) = 0$, it gives rise to a proper, orthogonal, broken variation of $\gamma$. We shall compute $L''(0) = I(Y\_\eta, Y\_\eta)$.

Using the bilinearity of $I$ and Lemma 2, one shows:

$$I(Y_\eta, Y_\eta) = -2\eta \left|\frac{DJ}{ds}(s_0)\right|^2 + \eta^2 I(Z, Z).$$

Observe now that if $\eta = \eta\_0$ is sufficiently small, the above expression is negative. Therefore, by taking $Y\_{\eta\_0}$, we shall obtain a proper broken variation with $L''(0) < 0$. Since $L'(0) = 0$, this means that $0$ is a point of local maximum for $L$; that is, there exists $\delta > 0$ such that if $t \in (-\delta, \delta)$, $t \neq 0$, then $L(t) < L(0)$. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2</span><span class="math-callout__name">(Morse Index Theorem)</span></p>

The index form of a geodesic was introduced by M. Morse, who proved the following result. Let $\gamma(s\_0)$ be a conjugate point of $\gamma(0) = p$, $\gamma\colon [0, l] \to S$, $s\_0 \in [0, l]$. The **multiplicity** of the conjugate point $\gamma(s\_0)$ is the dimension of the largest subspace $E$ of $T\_p(S)$ such that $(d\exp\_p)\_{\gamma(s\_0)}(u) = 0$ for every $u \in E$. The **index** of a quadratic form $Q\colon E \to R$ in a vector space $E$ is the maximum dimension of a subspace $L$ of $E$ such that $Q(u) < 0$, $u \in L$. With this terminology, the **Morse index theorem** is stated as follows: *Let $\gamma\colon [0, l] \to S$ be a geodesic. Then the index of the quadratic form $I$ of $\gamma$ is finite, and it is equal to the number of conjugate points to $\gamma(0)$ in $\gamma((0, l))$, each one counted with its multiplicity.*

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3</span></p>

Jacobi's theorem is a particular case of the Morse index theorem. Actually, the crucial point of the proof of the index theorem is essentially an extension of the ideas presented in the proof of Theorem 2.

</div>

## 5-10. Abstract Surfaces; Further Generalizations

In Sec. 5-11, we shall prove a theorem, due to Hilbert, which asserts that there exists no complete regular surface in $R^3$ with constant negative Gaussian curvature. To understand the correct statement and the proof of Hilbert's theorem, it will be convenient to introduce the notion of an abstract geometric surface.

So far the surfaces we have dealt with are subsets $S$ of $R^3$ on which differentiable functions make sense. We defined a tangent plane $T\_p(S)$ at each $p \in S$ and developed the differential geometry around $p$ as the study of the variation of $T\_p(S)$. We have, however, observed that all the notions of the intrinsic geometry (Gaussian curvature, geodesics, completeness, etc.) only depended on the choice of an inner product on each $T\_p(S)$. If we are able to define abstractly (that is, with no reference to $R^3$) a set $S$ on which differentiable functions make sense, we might eventually extend the intrinsic geometry to such sets.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1</span><span class="math-callout__name">(Abstract Surface / Differentiable Manifold of Dimension 2)</span></p>

An **abstract surface** (differentiable manifold of dimension 2) is a set $S$ together with a family of one-to-one maps $\mathbf{x}\_\alpha\colon U\_\alpha \to S$ of open sets $U\_\alpha \subset R^2$ into $S$ such that

1. $\bigcup\_\alpha \mathbf{x}\_\alpha(U\_\alpha) = S$.

2. For each pair $\alpha$, $\beta$ with $\mathbf{x}\_\alpha(U\_\alpha) \cap \mathbf{x}\_\beta(U\_\beta) = W \neq \phi$, we have that $\mathbf{x}\_\alpha^{-1}(W)$, $\mathbf{x}\_\beta^{-1}(W)$ are open sets in $R^2$, and $\mathbf{x}\_\beta^{-1} \circ \mathbf{x}\_\alpha$, $\mathbf{x}\_\alpha^{-1} \circ \mathbf{x}\_\beta$ are differentiable maps.

The pair $(U\_\alpha, \mathbf{x}\_\alpha)$ with $p \in \mathbf{x}\_\alpha(U\_\alpha)$ is called a **parametrization** (or coordinate system) of $S$ around $p$. $\mathbf{x}\_\alpha(U\_\alpha)$ is called a **coordinate neighborhood**, and if $q = \mathbf{x}\_\alpha(u\_\alpha, v\_\alpha) \in S$, we say that $(u\_\alpha, v\_\alpha)$ are the **coordinates** of $q$ in this coordinate system. The family $\lbrace U\_\alpha, \mathbf{x}\_\alpha \rbrace$ is called a **differentiable structure** for $S$.

</div>

It follows immediately from condition 2 that the "change of parameters" $\mathbf{x}\_\beta^{-1} \circ \mathbf{x}\_\alpha\colon \mathbf{x}\_\alpha^{-1}(W) \to \mathbf{x}\_\beta^{-1}(W)$ is a diffeomorphism.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1</span><span class="math-callout__name">(Real Projective Plane $P^2$)</span></p>

Let $S^2 = \lbrace (x, y, z) \in R^3;\; x^2 + y^2 + z^2 = 1 \rbrace$ be the unit sphere and let $A\colon S^2 \to S^2$ be the antipodal map; i.e., $A(x, y, z) = (-x, -y, -z)$. Let $P^2$ be the set obtained from $S^2$ by identifying $p$ with $A(p)$ and denote by $\pi\colon S^2 \to P^2$ the natural map $\pi(p) = \lbrace p, A(p) \rbrace$. Cover $S^2$ with parametrizations $\mathbf{x}\_\alpha\colon U\_\alpha \to S^2$ such that $\mathbf{x}\_\alpha(U\_\alpha) \cap A \circ \mathbf{x}\_\alpha(U\_\alpha) = \phi$. From the fact that $S^2$ is a regular surface and $A$ is a diffeomorphism, it follows that $P^2$ together with the family $\lbrace U\_\alpha, \pi \circ \mathbf{x}\_\alpha \rbrace$ is an abstract surface, to be denoted again by $P^2$. $P^2$ is called the **real projective plane**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2</span><span class="math-callout__name">(Klein Bottle)</span></p>

Let $T \subset R^3$ be a torus of revolution (Sec. 2-2, Example 4) with center in $(0, 0, 0) \in R^3$ and let $A\colon T \to T$ be defined by $A(x, y, z) = (-x, -y, -z)$. Let $K$ be the quotient space of $T$ by the equivalence relation $p \sim A(p)$ and denote by $\pi\colon T \to K$ the map $\pi(p) = \lbrace p, A(p) \rbrace$. Cover $T$ with parametrizations $\mathbf{x}\_\alpha\colon U\_\alpha \to T$ such that $\mathbf{x}\_\alpha(U\_\alpha) \cap A \circ \mathbf{x}\_\alpha(U\_\alpha) = \phi$. As before, it is possible to prove that $K$ with the family $\lbrace U\_\alpha, \pi \circ \mathbf{x}\_\alpha \rbrace$ is an abstract surface, which is called the **Klein bottle**.

</div>

### Differentiable Maps and Tangent Vectors

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2</span><span class="math-callout__name">(Differentiable Map between Abstract Surfaces)</span></p>

Let $S\_1$ and $S\_2$ be abstract surfaces. A map $\varphi\colon S\_1 \to S\_2$ is **differentiable** at $p \in S\_1$ if given a parametrization $\mathbf{y}\colon V \subset R^2 \to S\_2$ around $\varphi(p)$ there exists a parametrization $\mathbf{x}\colon U \subset R^2 \to S\_1$ around $p$ such that $\varphi(\mathbf{x}(U)) \subset \mathbf{y}(V)$ and the map

$$\mathbf{y}^{-1} \circ \varphi \circ \mathbf{x}\colon \mathbf{x}^{-1}(U) \subset R^2 \to R^2$$

is differentiable at $\mathbf{x}^{-1}(p)$. $\varphi$ is **differentiable on $S\_1$** if it is differentiable at every $p \in S\_1$.

</div>

Now we need to associate a tangent plane to each point of an abstract surface $S$. Since we do not have the support of $R^3$, we must search for a characteristic property of tangent vectors to curves which is independent of $R^3$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3</span><span class="math-callout__name">(Tangent Vector on an Abstract Surface)</span></p>

A differentiable map $\alpha\colon (-\epsilon, \epsilon) \to S$ is called a **curve** on $S$. Assume that $\alpha(0) = p$ and let $D$ be the set of functions on $S$ which are differentiable at $p$. The **tangent vector** to the curve $\alpha$ at $t = 0$ is the function $\alpha'(0)\colon D \to R$ given by

$$\alpha'(0)(f) = \frac{d(f \circ \alpha)}{dt}\bigg|_{t=0}, \qquad f \in D.$$

A **tangent vector** at a point $p \in S$ is the tangent vector at $t = 0$ of some curve $\alpha\colon (-\epsilon, \epsilon) \to S$ with $\alpha(0) = p$.

</div>

By choosing a parametrization $\mathbf{x}\colon U \to S$ around $p = \mathbf{x}(0, 0)$ we may express both the function $f$ and the curve $\alpha$ in $\mathbf{x}$ by $f(u, v)$ and $(u(t), v(t))$, respectively. Therefore,

$$\alpha'(0)(f) = \left\lbrace u'(0)\frac{\partial}{\partial u} + v'(0)\frac{\partial}{\partial v} \right\rbrace (f).$$

This suggests that we denote by $(\partial/\partial u)\_0$ the operator that maps a function $f$ into $(\partial f/\partial u)\_0$; a similar meaning will be attached to the symbol $(\partial/\partial v)\_0$. We remark that $(\partial/\partial u)\_0$, $(\partial/\partial v)\_0$ may be interpreted as the tangent vectors at $p$ of the "coordinate curves" $u \mapsto \mathbf{x}(u, 0)$ and $v \mapsto \mathbf{x}(0, v)$, respectively.

The set of tangent vectors at $p$, with the usual operations for functions, is a two-dimensional vector space $T\_p(S)$ called the **tangent space** of $S$ at $p$. A parametrization $\mathbf{x}\colon U \to S$ around $p$ determines an **associated basis** $\lbrace (\partial/\partial u)\_q,\; (\partial/\partial v)\_q \rbrace$ of $T\_p(S)$ for any $q \in \mathbf{x}(U)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4</span><span class="math-callout__name">(Differential of a Map)</span></p>

Let $S\_1$ and $S\_2$ be abstract surfaces and let $\varphi\colon S\_1 \to S\_2$ be a differentiable map. For each $p \in S\_1$ and each $w \in T\_p(S\_1)$, consider a differentiable curve $\alpha\colon (-\epsilon, \epsilon) \to S\_1$, with $\alpha(0) = p$, $\alpha'(0) = w$. Set $\beta = \varphi \circ \alpha$. The map $d\varphi\_p\colon T\_p(S\_1) \to T\_{\varphi(p)}(S\_2)$ given by $d\varphi\_p(w) = \beta'(0)$ is a well-defined linear map, called the **differential** of $\varphi$ at $p$.

</div>

### The Riemannian Metric

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5</span><span class="math-callout__name">(Riemannian Manifold of Dimension 2)</span></p>

A geometric surface $S$ (Riemannian manifold of dimension 2) is an abstract surface $S$ together with the choice of an inner product $\langle\; ,\; \rangle\_p$ at each $T\_p(S)$, $p \in S$, which varies differentiably with $p$ in the following sense. For some (and hence all) parametrization $\mathbf{x}\colon U \to S$ around $p$, the functions

$$E(u, v) = \left\langle \frac{\partial}{\partial u}, \frac{\partial}{\partial u} \right\rangle, \qquad F(u, v) = \left\langle \frac{\partial}{\partial u}, \frac{\partial}{\partial v} \right\rangle, \qquad G(u, v) = \left\langle \frac{\partial}{\partial v}, \frac{\partial}{\partial v} \right\rangle$$

are differentiable functions in $U$. The inner product $\langle\; ,\; \rangle$ is often called a **(Riemannian) metric** on $S$.

</div>

With the functions $E$, $F$, $G$ we define Christoffel symbols for $S$ by system 2 of Sec. 4-3. Since the notions of intrinsic geometry were all defined in terms of the Christoffel symbols, they can now be defined in $S$: covariant derivatives, parallel transport, geodesics, Gaussian curvature (via Eq. (5) of Sec. 4-3 or in terms of parallel transport as in Sec. 4-5).

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3</span><span class="math-callout__name">(The Hyperbolic Plane)</span></p>

Let $S = R^2$ be a plane with coordinates $(u, v)$ and define an inner product at each point $q = (u, v) \in R^2$ by setting

$$\left\langle \frac{\partial}{\partial u}, \frac{\partial}{\partial u} \right\rangle_q = E = 1, \qquad \left\langle \frac{\partial}{\partial u}, \frac{\partial}{\partial v} \right\rangle_q = F = 0, \qquad \left\langle \frac{\partial}{\partial v}, \frac{\partial}{\partial v} \right\rangle_q = G = e^{2u}.$$

$R^2$ with this inner product is a geometric surface $H$ called the **hyperbolic plane**. The geometry of $H$ is different from the usual geometry of $R^2$. For instance, the curvature of $H$ is

$$K = -\frac{1}{2\sqrt{EG}}\left\lbrace \left(\frac{E_v}{\sqrt{EG}}\right)_v + \left(\frac{G_u}{\sqrt{EG}}\right)_u \right\rbrace = -\frac{1}{2e^u}\left(\frac{2e^{2u}}{e^u}\right)_u = -1.$$

The geometry of $H$ is an exact model for the non-euclidean geometry of Lobachevski, in which all the axioms of Euclid, except the axiom of parallels, are assumed (cf. Sec. 4-5).

</div>

### Geodesics of the Poincare Half-Plane

To compute the geodesics of $H$, define a map

$$\phi\colon H \to R_+^2 = \lbrace (x, y) \in R^2;\; y > 0 \rbrace$$

by $\phi(u, v) = (v, e^{-u})$. Then $\phi$ is a diffeomorphism, and we can induce an inner product in $R\_+^2$ by setting $\langle d\phi(w\_1), d\phi(w\_2) \rangle\_{\phi(q)} = \langle w\_1, w\_2 \rangle\_q$. Computing:

$$\frac{\partial}{\partial x} = \frac{\partial}{\partial v}, \qquad \frac{\partial}{\partial y} = -e^u \frac{\partial}{\partial u},$$

hence,

$$\left\langle \frac{\partial}{\partial x}, \frac{\partial}{\partial x} \right\rangle = e^{2u} = \frac{1}{y^2}, \qquad \left\langle \frac{\partial}{\partial x}, \frac{\partial}{\partial y} \right\rangle = 0, \qquad \left\langle \frac{\partial}{\partial y}, \frac{\partial}{\partial y} \right\rangle = \frac{1}{y^2}.$$

$R\_+^2$ with this inner product is isometric to $H$, and it is sometimes called the **Poincare half-plane**.

Looking at the differential equations for the geodesics when $E = 1$, $F = 0$, the curves $v = \text{const.}$ are geodesics. Using polar coordinates $(x - x\_0) = \rho \cos\theta$, $y = \rho \sin\theta$ centered at a point $(x\_0, 0)$ on the boundary $y = 0$, one finds that the geodesics $\rho\_1 = \rho = \text{const.}$ are also geodesics. Collecting our observations, we conclude that **the lines and the half-circles which are perpendicular to the axis $y = 0$ are geodesics of the Poincare half-plane** $R\_+^2$. Since through each point $q \in R\_+^2$ and each direction issuing from $q$ there passes either a circle tangent to that line and normal to the axis $y = 0$ or a vertical line, these are all the geodesics of $R\_+^2$.

The geometric surface $R\_+^2$ is complete; that is, geodesics can be defined for all values of the parameter. It is now easy to see that, if we define a straight line of $R\_+^2$ to be a geodesic, all the axioms of Euclid but the axiom of parallels hold true in this geometry. The axiom of parallels in the Euclidean plane $P$ asserts that from a point not in a straight line $r \subset P$ one can draw a unique straight line parallel to $r$ (that is, a straight line that does not meet $r$). In the geometry of the Poincare half-plane, from a point not in a geodesic $\gamma$ we can draw infinitely many geodesics that do not meet $\gamma$.

### Immersions and Embeddings

The question then arises whether such a surface can be found as a regular surface in $R^3$. The natural context for this question is the following definition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6</span><span class="math-callout__name">(Isometric Immersion)</span></p>

A differentiable map $\varphi\colon S \to R^3$ of an abstract surface $S$ into $R^3$ is an **immersion** if the differential $d\varphi\_p\colon T\_p(S) \to T\_p(R^3)$ is injective. If, in addition, $S$ has a metric $\langle\; ,\; \rangle$ and

$$\langle d\varphi_p(v),\; d\varphi_p(w) \rangle_{\varphi(p)} = \langle v, w \rangle_p, \qquad v, w \in T_p(S),$$

$\varphi$ is said to be an **isometric immersion**.

</div>

The first inner product above is the usual inner product of $R^3$, whereas the second one is the given Riemannian metric on $S$. This means that in an isometric immersion, the metric "induced" by $R^3$ on $S$ agrees with the given metric on $S$.

Hilbert's theorem, to be proved in Sec. 5-11, states that there is no isometric immersion into $R^3$ of the complete hyperbolic plane. In particular, one cannot find a model of the geometry of Lobachevski as a regular surface in $R^3$. The above definition of isometric immersion makes perfect sense when we replace $R^3$ by $R^4$ or, for that matter, by an arbitrary $R^n$. Thus, we can ask: *For what values of $n$ is there an isometric immersion of the complete hyperbolic plane into $R^n$?* Hilbert's theorem says that $n \ge 4$. As far as we know, the case $n = 4$ is still unsettled.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4</span><span class="math-callout__name">(The Flat Torus)</span></p>

Let $R^2$ be a plane with coordinates $(x, y)$ and $T\_{m,n}\colon R^2 \to R^2$ be the map (translation) $T\_{m,n}(x, y) = (x + m, y + n)$, where $m$ and $n$ are integers. Define an equivalence relation in $R^2$ by $(x, y) \sim (x\_1, y\_1)$ if there exist integers $m$, $n$ such that $T\_{m,n}(x, y) = (x\_1, y\_1)$. Let $T$ be the quotient space of $R^2$ by this equivalence relation, and let $\pi\colon R^2 \to T$ be the natural projection map $\pi(x, y) = \lbrace T\_{m,n}(x, y);\; \text{all integers } m, n \rbrace$.

$T$ may be thought of as a closed square with opposite sides identified. Since $T\_{m,n}$ is an isometry of $R^2$, we can introduce a geometric (Riemannian) structure on $T$ so that $\pi$ is a local isometry. The coefficients of the first fundamental form of $T$, in any of the parametrizations of the family $\lbrace U\_\alpha, \pi \circ i\_\alpha \rbrace$, are $E = G = 1$, $F = 0$. Thus, this torus behaves locally like a Euclidean space. Its Gaussian curvature is identically zero. This accounts for the name **flat torus**.

Clearly the flat torus cannot be isometrically immersed in $R^3$, since by compactness it would have a point of positive curvature (cf. Exercise 16, Sec. 3-3, or Lemma 1, Sec. 5-2). However, it can be isometrically immersed in $R^4$.

In fact, let $F\colon R^2 \to R^4$ be given by

$$F(x, y) = \frac{1}{2\pi}(\cos 2\pi x,\; \sin 2\pi x,\; \cos 2\pi y,\; \sin 2\pi y).$$

Since $F(x + m, y + n) = F(x, y)$ for all $m$, $n$, we can define a map $\varphi\colon T \to R^4$ by $\varphi(p) = F(q)$, where $q \in \pi^{-1}(p)$. Since $\pi$ is a local diffeomorphism, $\varphi$ is differentiable. Furthermore, the rank of $dF$ is easily computed to be 2. Thus, $\varphi$ is an immersion. To see that the immersion is isometric, we observe that $\langle dF(e\_i), dF(e\_j) \rangle = \langle e\_i, e\_j \rangle$, $i, j = 1, 2$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5</span><span class="math-callout__name">(Orientability and Nonorientable Surfaces)</span></p>

The definition of orientability (cf. Sec. 2-6, Def. 1) can be extended, without changing a single word, to abstract surfaces. The real projective plane $P^2$ of Example 1 is **nonorientable**. To prove this, we observe that whenever an abstract surface $S$ contains an open set $M$ diffeomorphic to a Mobius strip (Sec. 2-6, Example 3), it is nonorientable. Now, $P^2$ is obtained from the sphere $S^2$ by identifying antipodal points. Consider on $S^2$ a thin strip $B$ made up of open segments of meridians whose centers lay on half an equator. Under identification of antipodal points, $B$ clearly becomes an open Mobius strip in $P^2$. Thus, $P^2$ is nonorientable.

By a similar argument, it can be shown that the Klein bottle $K$ of Example 2 is also nonorientable. It can be proved that a compact regular surface in $R^3$ is orientable (cf. Remark 2, Sec. 2-7). Thus, $P^2$ and $K$ cannot be embedded in $R^3$, and the same happens to the compact orientable surfaces generated by identifying symmetric points on surfaces symmetric relative to the origin of $R^3$.

$P^2$ and $K$ can, however, be embedded in $R^4$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7</span><span class="math-callout__name">(Embedding)</span></p>

Let $S$ be an abstract surface. A differentiable map $\varphi\colon S \to R^n$ is an **embedding** if $\varphi$ is an immersion and a homeomorphism onto its image.

</div>

A regular surface in $R^3$ can be characterized as the image of an abstract surface $S$ by an embedding $\varphi\colon S \to R^3$. This means that only those abstract surfaces which can be embedded in $R^3$ could have been detected in our previous study of regular surfaces in $R^3$.

### Higher-Dimensional Manifolds

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1a</span><span class="math-callout__name">(Differentiable Manifold of Dimension $n$)</span></p>

A **differentiable manifold of dimension $n$** is a set $M$ together with a family of one-to-one maps $\mathbf{x}\_\alpha\colon U\_\alpha \to M$ of open sets $U\_\alpha \subset R^n$ into $M$ such that

1. $\bigcup\_\alpha \mathbf{x}\_\alpha(U\_\alpha) = M$.

2. For each pair $\alpha$, $\beta$ with $\mathbf{x}\_\alpha(U\_\alpha) \cap \mathbf{x}\_\beta(U\_\beta) = W \neq \phi$, we have that $\mathbf{x}\_\alpha^{-1}(W)$, $\mathbf{x}\_\beta^{-1}(W)$ are open sets in $R^n$ and that $\mathbf{x}\_\beta^{-1} \circ \mathbf{x}\_\alpha$, $\mathbf{x}\_\alpha^{-1} \circ \mathbf{x}\_\beta$ are differentiable maps.

3. The family $\lbrace U\_\alpha, \mathbf{x}\_\alpha \rbrace$ is maximal relative to conditions 1 and 2.

</div>

The definitions of differentiable maps and tangent vector carry over, word by word, to differentiable manifolds. The tangent space is now an $n$-dimensional vector space. The definitions of differential and orientability also extend straightforwardly to the present situation.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6</span><span class="math-callout__name">(The Tangent Bundle)</span></p>

Let $S$ be an abstract surface and let $T(S) = \lbrace (p, w);\; p \in S,\; w \in T\_p(S) \rbrace$. We shall show that $T(S)$ can be given a differentiable structure (of dimension 4) to be called the **tangent bundle** of $S$.

Let $\lbrace U\_\alpha, \mathbf{x}\_\alpha \rbrace$ be a differentiable structure for $S$. Denote by $(u\_\alpha, v\_\alpha)$ the coordinates of $U\_\alpha$ and by $\lbrace \partial/\partial u\_\alpha,\; \partial/\partial v\_\alpha \rbrace$ the associated bases in the tangent planes of $\mathbf{x}\_\alpha(U\_\alpha)$. For each $\alpha$, define a map $\mathbf{y}\_\alpha\colon U\_\alpha \times R^2 \to T(S)$ by

$$\mathbf{y}_\alpha(u_\alpha, v_\alpha, x, y) = \left(\mathbf{x}_\alpha(u_\alpha, v_\alpha),\; x\frac{\partial}{\partial u_\alpha} + y\frac{\partial}{\partial v_\alpha}\right), \qquad (x, y) \in R^2.$$

This means that we take as coordinates of a point $(p, w) \in T(S)$ the coordinates $u\_\alpha$, $v\_\alpha$ of $p$ plus the coordinates of $w$ in the basis $\lbrace \partial/\partial u\_\alpha,\; \partial/\partial v\_\alpha \rbrace$. Then $\lbrace U\_\alpha \times R^2, \mathbf{y}\_\alpha \rbrace$ is a differentiable structure for $T(S)$.

The tangent bundle is the natural space to work with when dealing with second-order differential equations on $S$. For instance, the equations of a geodesic on a geometric surface $S$ can be written in a coordinate neighborhood as (cf. Sec. 4-7)

$$u'' = f_1(u, v, u', v'), \qquad v'' = f_2(u, v, u', v').$$

Introducing new variables $x = u'$, $y = v'$ reduces this to a first-order system

$$x' = f_1(u, v, x, y), \quad y' = f_2(u, v, x, y), \quad u' = f_3(u, v, x, y), \quad v' = f_4(u, v, x, y),$$

which may be interpreted as bringing into consideration the tangent bundle $T(S)$ with coordinates $(u, v, x, y)$, and looking upon the geodesics as trajectories of a vector field given locally by the system above. This field (or rather its trajectories) is called the **geodesic flow** on $T(S)$.

</div>

### Riemannian Manifolds and Sectional Curvature

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5a</span><span class="math-callout__name">(Riemannian Manifold of Dimension $n$)</span></p>

A **Riemannian manifold** is an $n$-dimensional differentiable manifold $M$ together with a choice, for each $p \in M$, of an inner product $\langle\; ,\; \rangle\_p$ in $T\_p(M)$ that varies differentiably with $p$ in the following sense. For some (and hence all) parametrization $\mathbf{x}\colon U\_\alpha \to M$ with $p \in \mathbf{x}\_\alpha(U\_\alpha)$, the functions

$$g_{ij}(u_1, \ldots, u_n) = \left\langle \frac{\partial}{\partial u_i}, \frac{\partial}{\partial u_j} \right\rangle, \qquad i, j = 1, \ldots, n,$$

are differentiable at $\mathbf{x}\_\alpha^{-1}(p)$; here $(u\_1, \ldots, u\_n)$ are the coordinates of $U\_\alpha \subset R^n$.

The differentiable family $\lbrace \langle\; ,\; \rangle\_p,\; p \in M \rbrace$ is called a **Riemannian structure** (or **Riemannian metric**) for $M$.

</div>

Starting from the covariant derivative, we can define parallel transport, geodesics, geodesic curvature, the exponential map, completeness, etc. The definitions are exactly the same as those we have given previously. The notion of curvature, however, requires more elaboration.

Let $p \in M$ and let $\sigma \subset T\_p(M)$ be a two-dimensional subspace of the tangent space $T\_p(M)$. Consider all those geodesics of $M$ that start from $p$ and are tangent to $\sigma$. From the fact that the exponential map is a local diffeomorphism at the origin of $T\_p(M)$, it can be shown that small segments of such geodesics make up an abstract surface $S$ containing $p$. $S$ has a natural geometric structure induced by the Riemannian structure of $M$. The Gaussian curvature of $S$ at $p$ is called the **sectional curvature** $K(p, \sigma)$ of $M$ at $p$ along $\sigma$.

It is possible to formalize the sectional curvature in terms of the Levi-Civita connection but that is too technical to be described here. We shall only mention that most of the theorems in this chapter can be posed as natural questions in Riemannian geometry. Some of them are true with little or no modification of the given proofs (the Hopf-Rinow theorem, the Bonnet theorem, the first Hadamard theorem, and the Jacobi theorems are all in this class). Some others, however, require further assumptions to hold true (the second Hadamard theorem, for instance) and were seeds for further developments.

## 5-11. Hilbert's Theorem

Hilbert's theorem can be stated as follows.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hilbert)</span></p>

A complete geometric surface $S$ with constant negative curvature cannot be isometrically immersed in $R^3$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1</span><span class="math-callout__name">(Historical Note)</span></p>

Hilbert's theorem was first treated in D. Hilbert, "Uber Flachen von konstanter Gausscher Krummung," *Trans. Amer. Math. Soc.* 2 (1901), 87–99. A different proof was given shortly after by E. Holmgren (1902). The proof we shall present here follows Hilbert's original ideas. The local part is essentially the same as in Hilbert's paper; the global part, however, is substantially different.

</div>

We shall start with some observations. By multiplying the inner product by a constant factor, we may assume that the curvature $K \equiv -1$. Moreover, since $\exp\_p\colon T\_p(S) \to S$ is a local diffeomorphism (corollary of the theorem of Sec. 5-5), it induces an inner product in $T\_p(S)$. Denote by $S'$ the geometric surface $T\_p(S)$ with this inner product. If $\psi\colon S \to R^3$ is an isometric immersion, the same holds for $\varphi = \psi \circ \exp\_p\colon S' \to R^3$. Thus, we are reduced to proving that there exists no isometric immersion $\varphi\colon S' \to R^3$ of a plane $S'$ with an inner product such that $K \equiv -1$.

### Area of $S'$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1</span><span class="math-callout__name">(The Area of $S'$ is Infinite)</span></p>

The area of $S'$ is infinite.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We shall prove that $S'$ is (globally) isometric to the hyperbolic plane $H$. Since the area of $H$ is (cf. Example 3, Sec. 5-10) $\int\_{-\infty}^{+\infty}\int\_{-\infty}^{+\infty} e^u\,du\,dv = \infty$, this will prove the lemma.

Let $p \in H$, $p' \in S'$, and choose a linear isometry $\psi\colon T\_p(H) \to T\_{p'}(S')$ between their tangent spaces. Define a map $\varphi\colon H \to S'$ by $\varphi = \exp\_{p'} \circ \psi \circ \exp\_p^{-1}$. Since each point of $H$ is joined to $p$ by a unique minimal geodesic, $\varphi$ is well defined. We now use polar coordinates $(\rho, \theta)$ and $(\rho', \theta')$ around $p$ and $p'$, respectively, requiring that $\varphi$ maps the axis $\theta = 0$ into the axis $\theta' = 0$. By the results of Sec. 4-6, $\varphi$ preserves the first fundamental form; hence, it is locally an isometry. Since $S'$ is simply connected, $\varphi$ is a homeomorphism, and hence a (global) isometry. **Q.E.D.**

</details>
</div>

### Tchebyshef Nets and Asymptotic Curves

For the rest of this section we shall assume that there exists an isometric immersion $\varphi\colon S' \to R^3$, where $S'$ is a geometric surface homeomorphic to a plane and with $K \equiv -1$.

To avoid the difficulties associated with possible self-intersections of $\varphi(S')$, we shall work with $S'$ and use the immersion $\varphi$ to induce on $S'$ the local extrinsic geometry of $\varphi(S') \subset R^3$. More precisely, since $\varphi$ is an immersion, for each $p \in S'$ there exists a neighborhood $V' \subset S'$ of $p$ such that the restriction $\varphi\|\_{V'} = \tilde{\varphi}$ is a diffeomorphism. At each $\tilde{\varphi}(q) \in \tilde{\varphi}(V')$, there exist two asymptotic directions. Through $\tilde{\varphi}$, these directions induce two directions at $q$, which will be called the **asymptotic directions on $S'$**, and the same procedure can be applied to any other local entity of $\varphi(S')$.

We now recall that the coordinate curves of a parametrization constitute a **Tchebyshef net** if the opposite sides of any quadrilateral formed by them have equal length (cf. Exercise 7, Sec. 2-5). If this is the case, it is possible to reparametrize the coordinate neighborhood in such a way that $E = G = 1$, $F = \cos\theta$, where $\theta$ is the angle formed by the coordinate curves (Sec. 2-5, Exercise 8). Furthermore, in this situation, $K = -(\theta\_{uv}/\sin\theta)$ (Sec. 4-3, Exercise 5).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2</span><span class="math-callout__name">(Asymptotic Curves Form a Tchebyshef Net)</span></p>

For each $p \in S'$ there is a parametrization $\mathbf{x}\colon U \subset R^2 \to S'$, $p \in \mathbf{x}(U)$, such that the coordinate curves of $\mathbf{x}$ are the asymptotic curves of $S'$ and form a Tchebyshef net (we shall express this by saying that the asymptotic curves form a Tchebyshef net).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $K < 0$, a neighborhood $V' \subset S'$ of $p$ can be parametrized by $\mathbf{x}(u, v)$ in such a way that the coordinate curves of $\mathbf{x}$ are the asymptotic curves of $V'$. Thus, if $e$, $f$, and $g$ are the coefficients of the second fundamental form of $S'$ in this parametrization, we have $e = g = 0$. Notice that we are using the convention of referring to the second fundamental form of $S'$ rather than the second fundamental form of $\varphi(S') \subset R^3$.

Now in $\varphi(V') \subset R^3$, setting $D = \sqrt{EG - F^2}$:

$$N \wedge N_v = K(\mathbf{x}_u \wedge \mathbf{x}_v)$$

and

$$N \wedge N_u = \frac{1}{D}(f\mathbf{x}_u - e\mathbf{x}_v), \qquad N \wedge N_v = \frac{1}{D}(g\mathbf{x}_u - f\mathbf{x}_v).$$

Since $K = -1$ and $e = g = 0$, we obtain $N \wedge N\_u = \pm \mathbf{x}\_v$, $N \wedge N\_v = \pm \mathbf{x}\_u$; hence $2KDN = \pm \mathbf{x}\_{uv} \pm \mathbf{x}\_{uv}$. It follows that $\mathbf{x}\_{uv}$ is parallel to $N$; hence, $E\_v = 2\langle \mathbf{x}\_{uv}, \mathbf{x}\_u \rangle = 0$ and $G\_u = 2\langle \mathbf{x}\_{uv}, \mathbf{x}\_v \rangle = 0$. But $E\_v = G\_u = 0$ implies (Sec. 2-5, Exercise 7) that the coordinate curves form a Tchebyshef net. **Q.E.D.**

</details>
</div>

### Area Bound from Asymptotic Coordinates

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3</span><span class="math-callout__name">(Area of Quadrilaterals in a Tchebyshef Net is Less Than $2\pi$)</span></p>

Let $V' \subset S'$ be a coordinate neighborhood of $S'$ such that the coordinate curves are the asymptotic curves in $V'$. Then the area $A$ of any quadrilateral formed by the coordinate curves is smaller than $2\pi$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $(\tilde{u}, \tilde{v})$ be the coordinates of $V'$. By the argument of Lemma 2, it is possible to reparametrize $V'$ by, say, $(u, v)$ so that $E = G = 1$ and $F = \cos\theta$. Let $R$ be a quadrilateral formed by the coordinate curves with vertices $(u\_1, v\_1)$, $(u\_2, v\_1)$, $(u\_2, v\_2)$, $(u\_1, v\_2)$ and interior angles $\alpha\_1$, $\alpha\_2$, $\alpha\_3$, $\alpha\_4$, respectively. Since $E = G = 1$, $F = \cos\theta$, and $\theta\_{uv} = \sin\theta$, we obtain

$$A = \int_R dA = \int_R \sin\theta\,du\,dv = \int_R \theta_{uv}\,du\,dv$$

$$= \theta(u_1, v_1) - \theta(u_2, v_1) + \theta(u_2, v_2) - \theta(u_1, v_2)$$

$$= \alpha_1 + \alpha_3 - (\pi - \alpha_2) - (\pi - \alpha_4) = \sum_{i=1}^4 \alpha_i - 2\pi < 2\pi,$$

since $\alpha\_i < \pi$. **Q.E.D.**

</details>
</div>

### Global Parametrization by Asymptotic Curves

The next step is to construct a global parametrization of $S'$ using asymptotic coordinates.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 4</span><span class="math-callout__name">(Asymptotic Curves with Arc Length)</span></p>

For a fixed $t$, the curve $\mathbf{x}(s, t)$, $-\infty < s < \infty$, is an asymptotic curve with $s$ as arc length.

</div>

The map $\mathbf{x}\colon R^2 \to S'$ is constructed as follows. Fix a point $O \in S'$ and choose orientations on the asymptotic curves passing through $O$. Make a definite choice of one of these asymptotic curves, to be called $a\_1$, and denote the other one by $a\_2$. For each $(s, t) \in R^2$, lay off on $a\_1$ a length equal to $s$ starting from $O$. Let $p'$ be the point thus obtained. Through $p'$ there pass two asymptotic curves, one of which is $a\_1$. Choose the other one and give it the orientation obtained by the continuous extension, along $a\_1$, of the orientation of $a\_2$. Over this oriented asymptotic curve lay off a length equal to $t$ starting from $p'$. The point so obtained is $\mathbf{x}(s, t)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5</span><span class="math-callout__name">($\mathbf{x}$ is a Local Diffeomorphism)</span></p>

$\mathbf{x}$ is a local diffeomorphism.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 6</span><span class="math-callout__name">($\mathbf{x}$ is Surjective)</span></p>

$\mathbf{x}$ is surjective.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $Q = \mathbf{x}(R^2)$. Since $\mathbf{x}$ is a local diffeomorphism, $Q$ is open in $S'$. We also remark that if $p' = \mathbf{x}(s\_0, t\_0)$, then the two asymptotic curves which pass through $p'$ are entirely contained in $Q$.

Let us assume that $Q \neq S'$. Since $S'$ is connected, the boundary $\mathrm{Bd}\, Q \neq \phi$. Let $p \in \mathrm{Bd}\, Q$. Since $Q$ is open in $S'$, $p \notin Q$. Now consider a rectangular neighborhood $R$ of $p$ in which the asymptotic curves form a Tchebyshef net. Let $q \in Q \cap R$. Then one of the asymptotic curves through $q$ intersects one of the asymptotic curves through $p$. By the above remark, this is a contradiction. **Q.E.D.**

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 7</span><span class="math-callout__name">(Two Independent Asymptotic Vector Fields)</span></p>

On $S'$ there are two differentiable linearly independent vector fields which are tangent to the asymptotic curves of $S'$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 8</span><span class="math-callout__name">($\mathbf{x}$ is Injective)</span></p>

$\mathbf{x}$ is injective.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Sketch</summary>

We want to show that $\mathbf{x}(s\_0, t\_0) = \mathbf{x}(s\_1, t\_1)$ implies $(s\_0, t\_0) = (s\_1, t\_1)$.

We first assume that $\mathbf{x}(s\_0, t\_0) = \mathbf{x}(s\_1, t\_0)$, $s\_1 > s\_0$, and show that this leads to a contradiction. By Lemma 7, an asymptotic curve cannot intersect itself unless the tangent lines agree at the intersection point. Since $\mathbf{x}$ is a local diffeomorphism, there exists an $\epsilon > 0$ such that $\mathbf{x}(s\_0, t) \neq \mathbf{x}(s\_1, t)$ for $t\_0 - \epsilon < t < t\_0$. By the same reason, the points of the curve $\mathbf{x}(s\_0, t)$ for which $\mathbf{x}(s\_0, t) = \mathbf{x}(s\_1, t)$ form an open and closed set of this curve; hence, either:

1. $\mathbf{x}(s\_0, t\_0) \neq \mathbf{x}(s\_0, t)$ for all $t > t\_0$, or

2. There exists $t = t\_1 > t\_0$ such that $\mathbf{x}(s\_0, t\_0) = \mathbf{x}(s\_0, t\_1)$; by a similar argument, $\mathbf{x}(s, t\_0 + b) = \mathbf{x}(s, t\_1 + b)$ for all $s$, $0 \le b \le t\_1 - t\_0$.

In case 1, $\mathbf{x}$ maps each strip of $R^2$ between two vertical lines at distance $s\_1 - s\_0$ onto $S'$ and identifies points in these lines with the same $t$. This implies that $S'$ is homeomorphic to a cylinder, and this is a contradiction (since $S'$ is homeomorphic to a plane).

In case 2, $\mathbf{x}$ maps each square formed by two horizontal lines at distance $s\_1 - s\_0$ and two vertical lines at distance $t\_1 - t\_0$ onto $S$ and identifies the opposite sides. But this contradicts the fact that the area of $S'$ is infinite (Lemma 1), since the area of any such quadrilateral is at most $2\pi$ (Lemma 3). **Q.E.D.**

</details>
</div>

### Conclusion of the Proof

Since $\mathbf{x}\colon R^2 \to S'$ is a local diffeomorphism (Lemma 5), surjective (Lemma 6), and injective (Lemma 8), $\mathbf{x}$ is a global diffeomorphism and therefore a global parametrization of $S'$. It follows that $S'$ can be covered by a single coordinate neighborhood in which the asymptotic curves form a Tchebyshef net with $E = G = 1$, $F = \cos\theta$. The area of $S'$ in this parametrization is

$$\text{Area}(S') = \iint_{R^2} \sin\theta\,du\,dv.$$

By Lemma 3, the area of any quadrilateral in this parametrization is less than $2\pi$. Since $\mathbf{x}$ is a global parametrization, this means that the total area of $S'$ is at most $2\pi$. But this contradicts Lemma 1, which states that the area of $S'$ is infinite. This contradiction proves that no such isometric immersion can exist. **Q.E.D.**

# Appendix — Point-Set Topology of Euclidean Spaces

In Chapter 5 we have used more freely some elementary topological properties of $R^n$. The usual properties of compact and connected subsets of $R^n$, as they appear in courses of advanced calculus, are essentially all that is needed. For completeness, we shall make a brief presentation of this material here, with proofs.

## A. Preliminaries

In what follows $U \subset R^n$ will denote an open set in $R^n$. The index $i$ varies in the range $1, 2, \ldots, m, \ldots$, and if $p = (x\_1, \ldots, x\_n)$, $q = (y\_1, \ldots, y\_n)$, $\|p - q\|$ will denote the distance from $p$ to $q$; that is,

$$|p - q|^2 = \sum_j (x_j - y_j)^2, \qquad j = 1, \ldots, n.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1</span><span class="math-callout__name">(Convergence)</span></p>

A sequence $p\_1, \ldots, p\_i, \ldots \in R^n$ **converges** to $p\_0 \in R^n$ if given $\epsilon > 0$, there exists an index $i\_0$ of the sequence such that $p\_i \in B\_\epsilon(p\_0)$ for all $i > i\_0$. In this situation, $p\_0$ is the **limit** of the sequence $\lbrace p\_i \rbrace$, and this is denoted by $\lbrace p\_i \rbrace \to p\_0$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Continuity and Convergence)</span></p>

A map $F\colon U \subset R^n \to R^m$ is continuous at $p\_0 \in U$ if and only if for each converging sequence $\lbrace p\_i \rbrace \to p\_0$ in $U$, the sequence $\lbrace F(p\_i) \rbrace$ converges to $F(p\_0)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2</span><span class="math-callout__name">(Limit Point)</span></p>

A point $p \in R^n$ is a **limit point** of a set $A \subset R^n$ if every neighborhood of $p$ in $R^n$ contains one point of $A$ distinct from $p$. A limit point is sometimes called a **cluster point** or an **accumulation point**.

</div>

Definition 2 is equivalent to saying that every neighborhood $V$ of $p$ contains infinitely many points of $A$. A set which contains only one limit point (as a set) is convergent if and only if it contains only one limit point.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3</span><span class="math-callout__name">(Closed Set, Closure)</span></p>

A set $F \subset R^n$ is **closed** if every limit point of $F$ belongs to $F$. The **closure** of $A \subset R^n$, denoted by $\bar{A}$, is the union of $A$ with its limit points.

</div>

Intuitively, $F$ is closed if it contains the limit of all its convergent sequences, or, in other words, it is invariant under the operation of passing to the limit. The closure of a set is a closed set, and the empty set $\phi$ is both open and closed.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Closed Sets and Open Complements)</span></p>

$F \subset R^n$ is closed if and only if the complement $R^n - F$ of $F$ is open.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Continuity and Open Sets)</span></p>

A map $F\colon U \subset R^n \to R^m$ is continuous if and only if for each open set $V \subset R^m$, $F^{-1}(V)$ is an open set.

</div>

**Corollary.** $F\colon U \subset R^n \to R^m$ is continuous if and only if for every closed set $A \subset R^m$, $F^{-1}(A)$ is a closed set.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4</span><span class="math-callout__name">(Boundary)</span></p>

Let $A \subset R^n$. The **boundary** $\mathrm{Bd}\, A$ of $A$ is the set of points $p$ in $R^n$ such that every neighborhood of $p$ contains points in $A$ and points in $R^n - A$.

</div>

$A \subset R^n$ is open if and only if no point of $\mathrm{Bd}\, A$ belongs to $A$, and $B \subset R^n$ is closed if and only if all points of $\mathrm{Bd}\, B$ belong to $B$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5</span><span class="math-callout__name">(Open Sets in Subsets)</span></p>

Let $A \subset R^n$. We say that $V \subset A$ is an **open set in $A$** if there exists an open set $U$ in $R^n$ such that $V = U \cap A$. A **neighborhood** of $p \in A$ in $A$ is an open set in $A$ containing $p$.

</div>

### Completeness of the Real Numbers

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6</span><span class="math-callout__name">(Supremum and Infimum)</span></p>

A subset $A \subset R$ of the real line $R$ is **bounded above** if there exists $M \in R$ such that $M \ge a$ for all $a \in A$. The number $M$ is called an **upper bound** for $A$. When $A$ is bounded above, a **supremum** or **least upper bound** $\sup A$ (or l.u.b. $A$) is an upper bound $M$ which satisfies the following condition: Given $\epsilon > 0$, there exists $a \in A$ such that $M - \epsilon < a$. By changing the sign of the above inequalities, we define similarly a **lower bound** and an **infimum** (or a greatest lower bound) of $A$, $\inf A$ (or g.l.b. $A$).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Axiom</span><span class="math-callout__name">(Completeness of Real Numbers)</span></p>

Let $A \subset R$ be nonempty and bounded above (below). Then there exists $\sup A$ ($\inf A$).

</div>

With the convention that $\sup A = +\infty$ if $A$ is not bounded above ($\inf A = -\infty$ if not bounded below), the above axiom can be stated as: *Every nonempty set of real numbers has a sup and an inf.*

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1</span><span class="math-callout__name">(Cauchy Sequences)</span></p>

Call a sequence $\lbrace x\_i \rbrace$ of real numbers a **Cauchy sequence** if given $\epsilon < 0$, there exists $i\_0$ such that $\|x\_i - x\_j\| < \epsilon$ for all $i, j > i\_0$. A sequence is convergent if and only if it is a Cauchy sequence.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4</span><span class="math-callout__name">(Cauchy Sequences in $R^n$)</span></p>

A sequence $\lbrace p\_i \rbrace$, $p\_i \in R^n$, converges if and only if it is a Cauchy sequence.

</div>

A **Cauchy sequence** $\lbrace p\_i \rbrace$ in $R^n$ is one for which given $\epsilon > 0$, there exists an index $i\_0$ such that the distance $\|p\_i - p\_j\| < \epsilon$ for all $i, j > i\_0$.

## B. Connected Sets

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8</span><span class="math-callout__name">(Arc)</span></p>

A continuous curve $\alpha\colon [a, b] \to A \subset R^n$ is called an **arc** in $A$ joining $\alpha(a)$ to $\alpha(b)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 9</span><span class="math-callout__name">(Arcwise Connected)</span></p>

$A \subset R^n$ is **arcwise connected** if, given two points $p, q \in A$, there exists an arc in $A$ joining $p$ to $q$.

</div>

Earlier in the book we used the word connected to mean arcwise connected. For a general subset of $R^n$, however, the notion of arcwise connectedness is much too restrictive, and it is more convenient to use the following definition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10</span><span class="math-callout__name">(Connected Set)</span></p>

$A \subset R^n$ is **connected** when it is not possible to write $A = U\_1 \cup U\_2$, where $U\_1$ and $U\_2$ are nonempty open sets in $A$ and $U\_1 \cap U\_2 = \phi$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5</span><span class="math-callout__name">(Open-Closed Subsets of Connected Sets)</span></p>

Let $A \subset R^n$ be connected and let $B \subset A$ be simultaneously open and closed in $A$. Then either $B = \phi$ or $B = A$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6</span><span class="math-callout__name">(Continuous Image of a Connected Set)</span></p>

Let $F\colon A \subset R^n \to R^m$ be continuous and $A$ be connected. Then $F(A)$ is connected.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 11</span><span class="math-callout__name">(Interval)</span></p>

An **interval** of the real line $R$ is any of the sets $a < x < b$, $a < x \le b$, $a \le x < b$, $a \le x \le b$, $x \in R$. The cases $a = b$, $a = -\infty$, $b = +\infty$ are not excluded, so that an interval may be a point, a half-line, or $R$ itself.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7</span><span class="math-callout__name">(Connected Subsets of $R$)</span></p>

$A \subset R$ is connected if and only if $A$ is an interval.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8</span><span class="math-callout__name">(Sign Preservation on Connected Sets)</span></p>

Let $f\colon A \subset R^n \to R$ be continuous and $A$ be connected. Assume that $f(q) \neq 0$ for all $q \in A$. Then $f$ does not change sign in $A$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9</span><span class="math-callout__name">(Arcwise Connected Implies Connected)</span></p>

Let $A \subset R^n$ be arcwise connected. Then $A$ is connected.

</div>

The converse is, in general, not true. However, there is an important special case where the converse holds.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12</span><span class="math-callout__name">(Locally Arcwise Connected)</span></p>

A set $A \subset R^n$ is **locally arcwise connected** if for each $p \in A$ and each neighborhood $V$ of $p$ in $A$ there exists an arcwise connected neighborhood $U \subset V$ of $p$ in $A$.

</div>

A simple example of a locally arcwise connected set in $R^3$ is a regular surface.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 10</span><span class="math-callout__name">(Locally Arcwise Connected: Connected iff Arcwise Connected)</span></p>

Let $A \subset R^n$ be a locally arcwise connected set. Then $A$ is connected if and only if it is arcwise connected.

</div>

## C. Compact Sets

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 13</span><span class="math-callout__name">(Bounded and Compact Sets)</span></p>

A set $A \subset R^n$ is **bounded** if it is contained in some ball of $R^n$. A set $K \subset R^n$ is **compact** if it is closed and bounded.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14</span><span class="math-callout__name">(Open Cover, Subcover)</span></p>

An **open cover** of a set $A \subset R^n$ is a family of open sets $\lbrace U\_\alpha \rbrace$, $\alpha \in \mathfrak{A}$, such that $\bigcup\_\alpha U\_\alpha = A$. When there are only finitely many $U\_\alpha$ in the family, we say that the cover is **finite**. If the subfamily $\lbrace U\_\beta \rbrace$, $\beta \in \mathfrak{B} \subset \mathfrak{A}$, still covers $A$, we say that $\lbrace U\_\beta \rbrace$ is a **subcover** of $\lbrace U\_\alpha \rbrace$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11</span><span class="math-callout__name">(Characterizations of Compactness)</span></p>

For a set $K \subset R^n$ the following assertions are equivalent:

1. $K$ is compact.

2. **(Heine-Borel)** Every open cover of $K$ has a finite subcover.

3. **(Bolzano-Weierstrass)** Every infinite subset of $K$ has a limit point in $K$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 12</span><span class="math-callout__name">(Continuous Image of a Compact Set)</span></p>

Let $F\colon K \subset R^n \to R^m$ be continuous and let $K$ be compact. Then $F(K)$ is compact.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 13</span><span class="math-callout__name">(Extreme Value Theorem)</span></p>

Let $f\colon K \subset R^n \to R$ be a continuous function defined on a compact set $K$. Then there exists $p\_1, p\_2 \in K$ such that

$$f(p_2) \le f(p) \le f(p_1) \qquad \text{for all } p \in K;$$

that is, $f$ reaches a maximum at $p\_1$ and a minimum at $p\_2$.

</div>

It is an important fact that on compact sets the two notions agree: *let $F\colon K \subset R^n \to R^m$ be continuous and $K$ be compact. Then $F$ is uniformly continuous in $K$.*

## D. Connected Components

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 14</span><span class="math-callout__name">(Union of Intersecting Connected Sets)</span></p>

Let $C\_\alpha \subset R^n$ be a family of connected sets such that $\bigcap\_\alpha C\_\alpha \neq \phi$. Then $C = \bigcup\_\alpha C\_\alpha$ is a connected set.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 15</span><span class="math-callout__name">(Connected Component)</span></p>

Let $A \subset R^n$ and $p \in A$. The union of all connected subsets of $A$ which contain $p$ is called the **connected component** of $A$ containing $p$.

</div>

By Proposition 14, a connected component is a connected set. Intuitively, the connected component of $A$ containing $p \in A$ is the largest connected subset of $A$ that contains $p$ (that is, it is contained in no other connected subset of $A$ that contains $p$).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 15</span><span class="math-callout__name">(Closure of a Connected Set is Connected)</span></p>

Let $C \subset A \subset R^n$ be a connected set. Then the closure $\bar{C}$ of $C$ in $A$ is connected.

</div>

**Corollary.** A connected component $C \subset A \subset R^n$ of a set $A$ is closed in $A$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16</span><span class="math-callout__name">(Connected Components of Locally Arcwise Connected Sets are Open)</span></p>

Let $C \subset A \subset R^n$ be a connected component of a locally arcwise connected set $A$. Then $C$ is open in $A$.

</div>
