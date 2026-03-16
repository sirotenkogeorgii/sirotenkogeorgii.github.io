---
layout: default
title: Differential Geometry
date: 2024-11-01
# excerpt: Connecting differential equations, stability analysis, and attractor theory to the training dynamics of modern machine learning models.
# tags:
#   - dynamical-systems
#   - machine-learning
#   - theory
---

# Differential Geometry

## Multivariable Calculus Done Correctly

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Setting and Notation)</span></p>

* all vector spaces are **finite-dimensional normed real vector spaces**;
* therefore they are also metric spaces;
* we can talk about **open sets** in the usual way;
* $U \subseteq V$ is usually an open set;
* $p \in U$ is a point of evaluation;
* $v \in V$ is a small displacement vector.

A useful conceptual distinction:

* elements of **$U$** are the actual **inputs**;
* elements of **$V$** are the allowed **small changes/displacements**.

Even though $U \subseteq V$, it helps to think of these as playing different roles.

</div>

### The Big Picture

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Better Intuition on Derivatives: from numbers to functions)</span></p>

First, let $f: [a,b] \to \mathcal{R}$. You might recall from high school calculus that for every point $p\in\mathcal{R}$, we defined $f'(p)$ as the derivative at the point $p$ (if it existed), which we interpreted as the **slope** of the "tangent line".

<figure>
  <img src="{{ '/assets/images/notes/differential_geometry/tangent_line.png' | relative_url }}" alt="a" loading="lazy">
</figure>

That’s fine, but I claim that the “better” way to interpret the derivative at that point is as a **linear map**, that is, as a **function**. If $f'(p) = 1.5$, then the derivative tells me that if I move $\epsilon$ away from $p$ then I should expect $f$ to change by about $1.5\epsilon$. In other words,

> **The derivative of $f$ at $p$ approximates $f$ near $p$ by a linear function.**

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(More General)</span></p>

Suppose we have a function $f:\mathcal{R}^2\to\mathcal{R},$ say

$$f(x,y)=x^2+y^2$$

for concreteness or something. For a point $p\in\mathcal{R}^2$, the “derivative” of $f$ at $p$ ought to represent a linear map that approximates $f$ at that point $p$. That means I want a **linear map** $T: \mathcal{R}^2\to\mathcal{R}$ such that 

$$f(p+ v) \approx f(p) + T(v)$$

for small displacements $v\in\mathcal{R}^2$.

<figure>
  <img src="{{ '/assets/images/notes/differential_geometry/tangent_plane.png' | relative_url }}" alt="a" loading="lazy">
</figure>

Even more generally, if $f: U\to W$ with $U \subseteq V$ open (in the $\lVert\cdot\rVert_V$ metric as usual), then the derivative at $p\in U$ ought to be so that 

$$f(p+ v) \approx f(p) + T(v) \in W.$$

(We need $U$ open so that for small enough $v,p+ v\in U$ as well.) In fact this is exactly what we were doing earlier with $f'(p)$ in high school.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The fundamental idea of Calculus)</span></p>

> **The fundamental idea of Calculus is the local approximation of functions by linear functions. The derivative does exactly this.**

By an unfortunate coincidence, a linear map $\mathcal{R} \to \mathcal{R}$ can be represented by just its slope. And in the unending quest to make everything a number so that it can be AP tested, we immediately forgot all about what we were trying to do in the first place and just defined the derivative of $f$ to be a **number** instead of a **function**.

Jean Dieudonné continues:
> In the classical teaching of Calculus, this idea is immediately obscured by the accidental fact that, on a one-dimensional vector space, there is a one-to-one correspondence between linear forms and numbers, and therefore the derivative at a point is defined as a number instead of a linear form. This **slavish subservience to the shibboleth of numerical interpretation at any cost** becomes much worse $\dots$

</div>

### Total Derivative

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Total Derivative)</span></p>

Let $U \subseteq V$ be open, let $f:U\to W$ be continuous, and let $p\in U$. If there exists a linear map $T:V\to W$ such that

$$
\lim_{\| v\|_V \to 0}
\frac{\| f(p+v)-f(p)-T(v)\|_W}{\| v\|_V}=0,
$$

then $T$ is called the **total derivative** of $f$ at $p$, denoted

$$(Df)_p$$

In that case, $f$ is **differentiable at $p$**. If this exists at every point, then $f$ is differentiable.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation)</span></p>

This means:

$$f(p+v)=f(p)+(Df)_p(v)+\text{error}$$

where the error is **small compared to $\| v\|$** as $v\to 0$. So the derivative is the **best first-order linear approximation** of $f$ near $p$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Ordinary Calculus)</span></p>

As in ordinary calculus:

* **differentiability implies continuity**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Relation to the One-Variable Derivative)</span></p>

If $V=W=\mathbb R$, then a linear map $T:\mathbb R\to\mathbb R$ has the form

$$T(v)=cv$$

for some real number $c$.

So in one variable, the total derivative is equivalent to the usual derivative number $f'(p)$. The usual derivative is therefore just a special case of the more general "derivative as a linear map" viewpoint.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($f(x,y)=x^2+y^2$)</span></p>

Let

$$f:\mathbb R^2\to\mathbb R,\qquad f(x,y)=x^2+y^2$$

Take $p=(a,b)$. Then

$$(Df)_p(v)=2a\,e_1^\vee(v)+2b\,e_2^\vee(v)$$

where $e_1^\vee,e_2^\vee$ are the dual basis functionals on $\mathbb R^2$. Equivalently, if $v=(x,y)$, then

$$(Df)_p(v)=2ax+2by$$

So the derivative at $(a,b)$ is the linear functional

$$(Df)_p = 2a\,e_1^\vee + 2b\,e_2^\vee$$

This is the main prototype: the derivative is not "the pair $(2a,2b)$" by itself; rather, it is the **linear map**

$$(x,y)\mapsto 2ax+2by$$

The coefficients $(2a,2b)$ only appear after choosing a basis.

</div>

### The Projection Principle

When passing from single-variable calculus to multivariable calculus, there are two apparent changes:

* the domain changes from $\mathbb R$ to $\mathbb R^n$;
* the codomain changes from $\mathbb R$ to $\mathbb R^m$.

The second upgrade is **super easy** in comparison to the first, and basically does not require doing anything new.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Projection Principle)</span></p>

Let $U$ be an open subset of the vector space $V$. Let $W$ be an $m$-dimensional real vector space with basis $w_1,\dots,w_m$, then any function

$$f:U\to W$$

can be uniquely written as

$$f(v)=f_1(v)w_1+\cdots+f_m(v)w_m$$

where each $f_i:U\to\mathbb R$. This gives a bijection between continuous functions $f:U\to W$ and $m$-tuples $f_1,f_2,\dots: U\to \mathbb{R}$ by projection onto the $i$th basis element $w_i$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Each Coordinate Can Be Studied Independently)</span></p>

**To think about a function $f: U \to \mathbb{R}^m$, it suffices to think about each coordinate separately.**

To understand $f:U\to\mathbb R^m$, it is enough to understand its coordinate functions one by one.

So the real conceptual difficulty in multivariable calculus comes from the **domain** being multidimensional, not the codomain.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Focusing on 1D Codomain)</span></p>

**For this reason, we will most often be interested in functions $f:U\to\mathbb R$. That's why the dual space $V^\vee$ is so important.**

</div>

### Total Derivative as a Function

If $f:U\to\mathbb R$ is differentiable at every point, then the total derivative itself becomes a function:

$$Df:U\to V^\vee,\qquad p\mapsto (Df)_p$$

This is important:

* $Df$ is not a number;
* $Df$ is not even just one linear map;
* $Df$ assigns to every point $p$ a linear functional $(Df)_p\in V^\vee$.

So for scalar-valued functions, the derivative naturally lives in the **dual space** $V^\vee$.

### Partial Derivatives

Assume $V$ has basis $e_1,\dots,e_n$. Then there is a dual basis $e_1^\vee,\dots,e_n^\vee$ of $V^\vee$.

Since $Df$ takes values in $V^\vee$, after choosing a basis we can write

$$Df = \psi_1 e_1^\vee + \cdots + \psi_n e_n^\vee$$

for some real-valued functions $\psi_1,\dots,\psi_n$. These functions are the **partial derivatives**.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Partial Derivative)</span></p>

The **$i$-th partial derivative** of $f:U\to\mathbb{R}$, denoted as

$$\frac{\partial f}{\partial e_i}(p): U\to\mathbb{R}$$

is defined by

$$\frac{\partial f}{\partial e_i}(p) := \lim_{t\to 0}\frac{f(p+t e_i)-f(p)}{t}$$

This is the derivative of $f$ along the basis direction $e_i$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Partial Derivatives and the Total Derivative)</span></p>

If the total derivative exists, then

$$(Df)_p(e_i)=\frac{\partial f}{\partial e_i}(p)$$

Therefore,

$$Df = \frac{\partial f}{\partial e_1}e_1^\vee+\cdots+\frac{\partial f}{\partial e_n}e_n^\vee$$

So once a basis is chosen, the total derivative is encoded by the list of partial derivatives. But conceptually the derivative is still the whole linear functional, not just the list of numbers.

</div>

### Distinguishing $Df$ from $(Df)_p$

This is easy to confuse.

* **$Df$** is a function $Df:U\to V^\vee$.
* **$(Df)_p$** is the value of that function at a specific point: $(Df)_p\in V^\vee$.

If

$$Df = \frac{\partial f}{\partial e_1}e_1^\vee+\cdots+\frac{\partial f}{\partial e_n}e_n^\vee$$

then at a point $p$,

$$(Df)_p= \frac{\partial f}{\partial e_1}(p)e_1^\vee+\cdots+\frac{\partial f}{\partial e_n}(p)e_n^\vee$$

So the partial derivatives $\frac{\partial f}{\partial e_i}$ are themselves functions $U\to\mathbb R$. At a particular point $p$, they become numbers.

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($f(x,y)=x^2+y^2$)</span></p>

For $f(x,y)=x^2+y^2$ we have

$$Df(x,y)=2x\,e_1^\vee+2y\,e_2^\vee$$

Thus the partial derivatives are

$$
\frac{\partial f}{\partial x}(x,y)=2x,\qquad
\frac{\partial f}{\partial y}(x,y)=2y
$$

At a specific point $(a,b)$,

$$(Df)_{(a,b)}=2a\,e_1^\vee+2b\,e_2^\vee$$

So:

* $Df$ is the full derivative field over the plane;
* $(Df)_{(a,b)}$ is the derivative at one point;
* $\partial f/\partial x$ and $\partial f/\partial y$ are the coordinate functions of that field.

</div>

### Computing Derivatives in Practice

The limit definition of total derivative is conceptually correct, but often hard to use directly. Fortunately, the standard computational method from calculus still works in most ordinary situations.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Continuous Partials Imply Differentiability)</span></p>

Let $U\subseteq V$ be open, choose a basis $e_1,\dots,e_n$, and let $f:U\to\mathbb R$. If each partial derivative

$$\frac{\partial f}{\partial e_i}$$

exists and is continuous, then $f$ is differentiable and

$$Df=\sum_{i=1}^n \frac{\partial f}{\partial e_i}\, e_i^\vee$$

</div>

The idea behind this: given $v=t_1e_1+\cdots+t_ne_n$, one moves from $p$ to $p+v$ one coordinate direction at a time:

$$p \to p+t_1e_1 \to p+t_1e_1+t_2e_2 \to \cdots \to p+v$$

The accumulated first-order changes are controlled by the partial derivatives.

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Computing a Nontrivial Total Derivative)</span></p>

Let

$$f(x,y)=x\sin y + x^2 y^4$$

Then

$$\frac{\partial f}{\partial x}(x,y)=\sin y + 2xy^4$$

and

$$\frac{\partial f}{\partial y}(x,y)=x\cos y + 4x^2 y^3$$

Since these partial derivatives are continuous, $f$ is differentiable and

$$Df = \frac{\partial f}{\partial x}e_1^\vee + \frac{\partial f}{\partial y}e_2^\vee$$

So explicitly,

$$Df(x,y)=\big(\sin y + 2xy^4\big)e_1^\vee + \big(x\cos y + 4x^2 y^3\big)e_2^\vee$$

At a point $(a,b)$, this means

$$(Df)_{(a,b)}(u,v) = (\sin b + 2ab^4)u + (a\cos b + 4a^2 b^3)v$$

The key lesson is that once the basis is fixed, we compute the total derivative from the familiar partial derivatives.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Partial Derivatives Alone Are Not Enough)</span></p>

The continuity assumption in the theorem above is important. Consider the counterexample:

$$
f(x,y)=
\begin{cases}
\dfrac{xy}{x^2+y^2}, & (x,y)\neq (0,0) \\
0, & (x,y)=(0,0).
\end{cases}
$$

At $(0,0)$, both partial derivatives exist, but the total derivative does **not** exist there. So:

* existence of partial derivatives at a point does **not** guarantee differentiability;
* continuity of the partials is a sufficient condition that fixes this.

</div>

### Higher Derivatives

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Higher Derivatives)</span></p>

If $f:U\to W$, then

$$Df:U\to \operatorname{Hom}(V,W)$$

Since $\operatorname{Hom}(V,W)$ is itself a normed vector space (with operator norm), it makes sense to differentiate again:

$$D^2f = D(Df): U\to \operatorname{Hom}(V,\operatorname{Hom}(V,W))$$

Similarly, $D^3f, D^4f, \dots$ can be defined iteratively.

In cleaner tensor notation:

$$D^k f : V \to (V^\vee)^{\otimes k}\otimes W$$

The $k$-th derivative is a $k$-linear object.

</div>

### The Hessian

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hessian)</span></p>

When $k=2$ and $W=\mathbb R$,

$$D^2f(p)\in (V^\vee)^{\otimes 2}$$

and after choosing a basis it can be represented by an $n\times n$ matrix. This is the **Hessian**.

The Hessian is just the matrix representation of the second derivative after a basis has been chosen.

</div>

### Symmetry of the Second Derivative

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Symmetry of the Second Derivative)</span></p>

If $f:U\to W$, $U\subseteq V$ open, and $(D^2f)_p$ exists, then it is **symmetric**:

$$(D^2f)_p(v_1,v_2)=(D^2f)_p(v_2,v_1)$$

This is the abstract version of the familiar "mixed partials commute" fact.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Clairaut's Theorem)</span></p>

If $f:U\to \mathbb R$ is twice differentiable, then for any basis vectors $e_i,e_j$,

$$\frac{\partial}{\partial e_i}\frac{\partial}{\partial e_j}f(p) = \frac{\partial}{\partial e_j}\frac{\partial}{\partial e_i}f(p)$$

So mixed partial derivatives are equal whenever the second derivative exists appropriately.

</div>

### Toward Differential Forms

The derivative of a scalar-valued function should be thought of as

$$Df:U\to V^\vee$$

This viewpoint sets up the next topic: **differential forms**. The familiar symbols $dx,dy,\dots$ are not decorative; they are related to dual vectors / covectors and integration in the correct coordinate-free language.

### Further Results

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Chain Rule and Higher Symmetry)</span></p>

1. **Chain Rule**: For differentiable maps $U_1 \xrightarrow{f} U_2 \xrightarrow{g} U_3$, if $h=g\circ f$, then

   $$(Dh)_p=(Dg)_{f(p)}\circ (Df)_p$$

2. **Symmetry of higher derivatives**: If $f$ is $k$-times differentiable, then $(D^k f)_p$ is symmetric in its $k$ arguments.

</div>

### Summary

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Core Definitions)</span></p>

* **Derivative at a point** = a linear map $(Df)_p:V\to W$ giving the best local linear approximation.
* **Differentiability at $p$** means the approximation error is $o(\|v\|)$.
* For scalar-valued functions, $(Df)_p\in V^\vee$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Core Formulas)</span></p>

If $f:U\to\mathbb R$ and $V$ has basis $e_1,\dots,e_n$, then

$$(Df)_p(e_i)=\frac{\partial f}{\partial e_i}(p)$$

and

$$Df=\sum_{i=1}^n \frac{\partial f}{\partial e_i} e_i^\vee$$

If the partial derivatives exist and are continuous, then this formula is valid and $f$ is differentiable.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Common Confusions to Avoid)</span></p>

1. **Confusing $Df$ with $(Df)_p$**: $Df$ is a function on $U$; $(Df)_p$ is one linear map at one point.
2. **Thinking the derivative is "just the gradient vector"**: The gradient depends on coordinates and inner product conventions. The more fundamental object is the linear functional $(Df)_p\in V^\vee$.
3. **Assuming partial derivatives imply differentiability**: False in general. You need stronger conditions, such as continuity of the partials.

</div>

---

## Differential Forms

### Overview

<div class="math-callout math-callout--info" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Setup)</span></p>

Throughout:

* $V$ is a **finite-dimensional real inner product space**
* $U \subseteq V$ is an **open set**
* all functions are assumed **smooth** (infinitely differentiable)
* $V^\vee$ denotes the **dual space** of linear functionals on $V$
* $\bigwedge^k(V^\vee)$ is the space of **alternating $k$-linear forms** on $V$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The key idea)</span></p>

The guiding idea is:

> **A differential form is something you can integrate.**

This develops in two stages:

1. **Geometric intuition**: Differential forms are introduced by pictures:
   * **0-forms** integrate over points
   * **1-forms** integrate over curves
   * **2-forms** integrate over surfaces

2. **Algebraic definition**: A differential $k$-form on an open set $U \subseteq V$ is a smooth map

   $$\alpha: U \to \bigwedge^k(V^\vee)$$

   where $V^\vee$ is the dual space of $V$.

</div>

### Pictures of Differential Forms

#### 0-Forms

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(0-Form)</span></p>

A **0-form** on $U$ is just a smooth function

$$f: U \to \mathbb{R}$$

A 0-form assigns a real number to every point of $U$. It can be integrated over a **0-dimensional cell** (a finite set of points) by simply summing its values.

* 0-form = scalar field
* Input: a point
* Output: a number
* Integrates over: points

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(0-Form)</span></p>

Thus, if we specify a finite set $S$ of points in $U$ we can “integrate” over $S$ by just adding up the values of the points:

$$0 + \sqrt{2} + 3 + (-1) = 2 + \sqrt{2}$$

So, a **0-form f lets us integrate over 0-dimensional “cells”.**

<figure>
  <img src="{{ '/assets/images/notes/differential_geometry/0-form-integration.png' | relative_url }}" alt="a" loading="lazy">
</figure>

</div>

#### 1-Forms

But this is quite boring, because as we know we like to integrate over things like curves, not single points. So, by analogy, we want a 1-form to let us integrate over 1-dimensional cells: i.e. over curves. What information would we need to do that? To answer this, let’s draw a picture of a curve c, which can be thought of as a function $c: [0,1] \to U$.

<figure>
  <img src="{{ '/assets/images/notes/differential_geometry/1-form-integration.png' | relative_url }}" alt="a" loading="lazy">
</figure>

If we want to integrate over a **curve**, then knowing only the value of a function at each point is not enough. Along a curve, at each point there is also a **tangent vector** telling us the direction of motion. So, we can define a 1-form $\alpha$ as follows. A 0-form just took a point and gave a real number, but a **1-form will take both a point *and* a tangent vector at that point, and spit out a real number**. So a 1-form $\alpha$ is a smooth function on pairs $(p,v)$, where $v$ is a tangent vector at $p$, to $\mathcal{R}$. Hence

$$\alpha: U\times V \to \mathcal{R}$$

Actually, for any point $p$, we will require that $\alpha(p,−)$ is a linear function in terms of the vectors: i.e. we want for example that $\alpha(p,2v) = 2\alpha(p,v)$. So it is more customary to think of $\alpha$ as:

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(1-Form)</span></p>

A **1-form** is a smooth map

$$\alpha: U \to V^\vee$$

Like with $Df$, we’ll use αp instead of $\alpha$(p). At each point $p$, the value 

$$\alpha_p \in V^\vee$$

i.e. a linear functional that eats a tangent vector and produces a real number. 

Informally, a 1-form takes a point $p$ and a tangent vector $v$ at that point, and returns a real number. For each fixed point, the dependence on $v$ is **linear**.

* 1-form = something that measures vectors along curves
* Input: point + tangent vector
* Output: a number
* Integrates over: curves

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Arc-Length Is Not a 1-Form)</span></p>

The **arc-length element $ds$** ($\int_c ds$ for the curve $c$) is **not** a 1-form, because it is **not linear** in the tangent vector. The expression

$$ds = \sqrt{dx^2 + dy^2}$$

already suggests nonlinearity. A 1-form does **not** measure "how long a vector is." It measures a vector **linearly**.

</div>

#### 2-Forms

A **2-form** should integrate over a **2-dimensional cell**, such as a square or blob of the form:

$$c: [0,1]\times [0,1]\to U$$

A surface is approximated by tiny **parallelograms** spanned by pairs of tangent vectors in $U$. In the same sense that lots of tangent vectors approximate the entire curve,
lots of tiny squares will approximate the big square in $U$.

<figure>
  <img src="{{ '/assets/images/notes/differential_geometry/2-form-integration.png' | relative_url }}" alt="a" loading="lazy">
</figure>

So what should a 2-form $\beta$ be? As before, it should start by taking a point $p\in U$, so $\beta_p$ is now a linear functional: but this time, it should be a linear map on two vectors $v$ and $w$. Here $v$ and w are not tangent so much as their span cuts out a small parallelogram. So, the right thing to do is in fact consider

$$\beta_p \in V^\vee \wedge V^\vee$$
.
That is, to use the wedge product to get a handle on the idea that $v$ and $w$ span a parallelogram. Another valid choice would have been $(V \wedge V)^\vee$; in fact, the two are isomorphic, but it will be more convenient to write it in the former.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2-Form)</span></p>

At a point $p$, a 2-form takes two vectors $v,w$ and outputs a real number associated to the parallelogram they span. Algebraically,

$$\beta_p \in V^\vee \wedge V^\vee = \bigwedge^2(V^\vee)$$

So a 2-form at a point is an alternating bilinear functional on pairs of vectors.

* 2-form = something that measures oriented area / flux / signed parallelograms
* Input: point + two vectors
* Output: a number
* Integrates over: surfaces

</div>

### Pictures of Exterior Derivatives

#### From a 0-Form to a 1-Form: $df$

Suppose $f$ is a 0-form, i.e. a smooth function $f:U\to\mathbb R$. Then there is a natural associated 1-form called **$df$**.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Exterior Derivative of a 0-Form)</span></p>

At a point $p$, and for a tangent vector $v$,

$$(df)_p(v) = (Df)_p(v)$$

the total derivative of $f$ at $p$ applied to $v$.

$df$ measures the **infinitesimal change** of $f$ in the direction $v$:

* $f$ tells you the height/value
* $df$ tells you how that value changes locally in a direction

</div>

<figure>
  <img src="{{ '/assets/images/notes/differential_geometry/from-0-form-to-1-form.png' | relative_url }}" alt="a" loading="lazy">
</figure>

Thus, $df$ measures "the change in $f$". Now, even if I haven’t defined integration yet, given a curve $c$ from a point $a$ to $b$, what do you think

$$\int_c df$$

should be equal to? Remember that $df$ is the 1-form that measures "infinitesimal change in $f$". So if we add up all the change in $f$ along a path from $a$ to $b$, then the answer we get should just be

$$\int_c df = f(b) - f(a)$$

This is the first case of something we call Stokes’ theorem.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(First Stokes-Type Result)</span></p>

If $c$ is a curve from point $a$ to point $b$, then

$$\int_c df = f(b)-f(a)$$

The integral of an exact 1-form $df$ over a path depends only on the endpoints. This is the **first case of Stokes' theorem**.

</div>

#### From a 1-Form to a 2-Form: $d\alpha$

Generalizing, how should we get from a 1-form to a 2-form? At each point $p$, the 2-form $\beta$ gives a $\beta p$ which takes in a “parallelogram” and returns a real number. Now suppose we have a 1-form $\alpha$. Then along each of the edges of a parallelogram, with an appropriate sign convention the 1-form $\alpha$ gives us a real number. So, given a 1-form $\alpha$, we define $d\alpha$ to be the 2-form that takes in a parallelogram spanned by $v$ and $w$, and returns the measure of $\alpha$ along the boundary.

Given a 1-form $\alpha$, the exterior derivative $d\alpha$ should be a 2-form. At each point, a small parallelogram has four boundary edges. The 1-form $\alpha$ can be integrated along those boundary edges with signs, so $d\alpha$ measures the **total boundary contribution** around the tiny parallelogram.

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stokes' Theorem Pattern)</span></p>

If you integrate $d\alpha$ over a whole square, then the contributions from internal boundaries cancel in pairs, leaving only the contribution from the outer boundary. This is another instance of **Stokes' theorem**.

The core pattern:

* $d$ turns "local measurement" into "boundary accumulation"
* Global integral over a region = boundary integral

</div>

<figure>
  <img src="{{ '/assets/images/notes/differential_geometry/from-1-form-to-2-form.png' | relative_url }}" alt="a" loading="lazy">
</figure>

### Formal Definition of Differential Forms

**Prototypical example for this section:** Algebraically, something that looks like $fe_1^\vee \wedge e_2^\vee + \dots$, and geometrically, see the previous section.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Differential $k$-Form)</span></p>

Let $V$ be an $n$-dimensional real vector space and $U\subseteq V$ open.

A **differential $k$-form** on $U$ is a smooth map

$$\alpha: U \to \bigwedge^k(V^\vee)$$

At each point $p\in U$, the value $\alpha_p$ is an alternating $k$-linear functional on vectors in $V$.

At each point, a $k$-form is a rule that takes $k$ vectors and returns a number interpreted as a **signed $k$-dimensional volume**.

</div>

### Low-Dimensional Examples

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Forms in Low Dimensions)</span></p>

**$k=0$:** A 0-form is just a smooth function $U\to \mathbb R$.

**$k=1$:** A 1-form is a smooth function $U\to V^\vee$. In particular, if $f:V\to \mathbb R$ is smooth, then its total derivative $Df$ is a 1-form.

**2-forms in $\mathbb R^3$:** If $V=\mathbb R^3$ with standard basis $e_1,e_2,e_3$, then a general 2-form has the shape

$$
\alpha_p = f(p)\,e_1^\vee\wedge e_2^\vee
+
g(p)\,e_1^\vee\wedge e_3^\vee
+
h(p)\,e_2^\vee\wedge e_3^\vee
$$

where $f,g,h:V\to\mathbb R$ are smooth functions. A 2-form in $\mathbb R^3$ has one coefficient for each basis 2-plane:

* $e_1^\vee\wedge e_2^\vee$
* $e_1^\vee\wedge e_3^\vee$
* $e_2^\vee\wedge e_3^\vee$

</div>

### Coordinate Expression of a General $k$-Form

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Coordinate Expression)</span></p>

Choose a basis $e_1,\dots,e_n$ of $V$. Then a basis of $\bigwedge^k(V^\vee)$ is given by

$$
e_{i_1}^\vee\wedge e_{i_2}^\vee\wedge \cdots \wedge e_{i_k}^\vee
\quad\text{for }1\le i_1<\cdots<i_k\le n
$$

Hence a general $k$-form can be written as

$$
\alpha_p = \sum_{1\le i_1<\cdots<i_k\le n}
f_{i_1,\dots,i_k}(p)\,
e_{i_1}^\vee\wedge \cdots \wedge e_{i_k}^\vee
$$

Abbreviated as $\alpha=\sum_I f_I\,de_I$ where $I=(i_1,\dots,i_k)$ and $de_I = e_{i_1}^\vee\wedge\cdots\wedge e_{i_k}^\vee$.

A $k$-form is like a field of alternating $k$-linear algebraic gadgets, with smooth coefficient functions.

</div>

### Evaluating Differential Forms

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Wedge Product Evaluation)</span></p>

For linear functionals $\xi_1,\dots,\xi_k\in V^\vee$ and vectors $v_1,\dots,v_k\in V$, the wedge product is evaluated by

$$
(\xi_1\wedge\cdots\wedge \xi_k)(v_1,\dots,v_k) = \det
\begin{bmatrix}
\xi_1(v_1) & \cdots & \xi_1(v_k)\\
\vdots & \ddots & \vdots\\
\xi_k(v_1) & \cdots & \xi_k(v_k)
\end{bmatrix}
$$

This determinant formula ensures alternation, e.g. $\alpha_p(v_1,v_2)=-\alpha_p(v_2,v_1)$.

Determinants measure **signed volume**: swap two vectors → sign changes; linearly dependent vectors → volume $=0$; scaling one vector scales the value linearly. This is exactly why differential forms are suitable for integration.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Evaluating a 2-Form in $\mathbb R^3$)</span></p>

Let $V=\mathbb R^3$. Suppose at a point $p$,

$$\alpha_p = 2e_1^\vee\wedge e_2^\vee + e_1^\vee\wedge e_3^\vee$$

Take

$$v_1 = 3e_1 + e_2 + 4e_3, \qquad v_2 = 8e_1 + 9e_2 + 5e_3$$

Then

$$
\alpha_p(v_1,v_2) = 2\det
\begin{bmatrix}
3 & 8\\
1 & 9
\end{bmatrix}
+
\det
\begin{bmatrix}
3 & 8\\
4 & 5
\end{bmatrix} = 2(27-8)+(15-32)=38-17=21
$$

The 2-form $\alpha_p$ assigns to the ordered pair $(v_1,v_2)$ the signed area-like quantity $21$.

</div>

### Geometric Meaning of a $k$-Form

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation of $k$-Forms)</span></p>

At each point $p$, a $k$-form takes $k$ vectors and outputs a number interpreted as a **signed volume** of the parallelepiped they span:

* **0-form**: value
* **1-form**: signed linear measurement along a direction
* **2-form**: signed area / circulation density / flux-type measurement
* **$k$-form**: signed $k$-dimensional volume measurement

</div>

### What Makes Forms Special

Differential forms are built using the **wedge product**, which forces **alternation**:

* exchanging two input vectors changes the sign
* repeated / dependent directions collapse to zero

This matches geometric intuition about oriented area and volume, which is why forms are the natural objects for integration.

### The Exterior Derivative

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What the Exterior Derivative Does)</span></p>

* $df$ records infinitesimal change of a function $f$
* $d\alpha$ records infinitesimal boundary accumulation of a 1-form $\alpha$ around a tiny parallelogram

So the operator $d$ raises degree by 1:

* 0-form $\to$ 1-form
* 1-form $\to$ 2-form
* in general $k$-form $\to$ $(k+1)$-form

</div>

### Summary

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Forms to Know)</span></p>

1. **0-form**: smooth function $f:U\to\mathbb R$
2. **1-form**: smooth map $U\to V^\vee$
3. **$k$-form**: smooth map $U\to \bigwedge^k(V^\vee)$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Key Results)</span></p>

* $(df)_p(v)=(Df)_p(v)$
* $\int_c df = f(b)-f(a)$ for a curve from $a$ to $b$
* $d\alpha$ measures the boundary accumulation of $\alpha$ around infinitesimal parallelograms
* wedge products are evaluated via determinants

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Common Pitfalls)</span></p>

1. **Thinking every infinitesimal quantity is a differential form**: Not true -- $ds$ is not a 1-form because it is not linear.
2. **Forgetting orientation**: Since forms are alternating, reversing order changes sign.
3. **Confusing a form with a function only on points**: For $k\ge1$, a form needs vectors as inputs too.

</div>

A differential $k$-form on $U\subseteq V$ is a smooth field of alternating $k$-linear maps:

$$
\alpha_p: V^k\to\mathbb R,
\qquad
\alpha_p\in \bigwedge^k(V^\vee).
$$

It assigns a signed $k$-dimensional volume-type number to $k$ vectors at each point.

* 0-forms integrate over points
* 1-forms integrate over curves
* 2-forms integrate over surfaces

The exterior derivative $d$ raises degree by $1$:

* $df$ = infinitesimal change of $f$
* $d\alpha$ = infinitesimal boundary accumulation of $\alpha$

and Stokes' theorem says global integrals of $d(\cdot)$ become boundary integrals.

---

## Integrating Differential Forms

We now show how to integrate differential forms over cells, and state Stokes' theorem in this context. Throughout, all vector spaces are finite-dimensional and real.

### Motivation: Line Integrals

Given a function $g\colon [a,b]\to\mathbb R$, the fundamental theorem of calculus says

$$\int_{[a,b]} g(t)\,dt = f(b)-f(a)$$

where $f$ is a function such that $g=df/dt$. Equivalently, for $f\colon [a,b]\to\mathbb R$,

$$\int_{[a,b]} g\,dt = \int_{[a,b]} df = f(b)-f(a)$$

where $df$ is the exterior derivative defined earlier.

Now suppose more generally we have $U$ an open subset of our real vector space $V$ and a 1-form $\alpha\colon U\to V^\vee$.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Parametrized Curve)</span></p>

A **parametrized curve** is a smooth function $c\colon [a,b]\to U$.

</div>

We want to define $\int_c\alpha$ such that the integral "adds up all the $\alpha$ along the curve $c$."

Our differential form $\alpha$ first takes in a point $p$ to get $\alpha_p\in V^\vee$. Then it eats a tangent vector $v\in V$ to finally give a real number $\alpha_p(v)\in\mathbb R$. We would like to "add all these numbers up," using only the notion of an integral over $[a,b]$.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Line Integral of a 1-Form)</span></p>

The integral of a 1-form $\alpha$ along a parametrized curve $c\colon [a,b]\to U$ is

$$\int_c\alpha := \int_{[a,b]} \alpha_{c(t)}(c'(t))\,dt$$

Here $c'(t)$ is shorthand for $(Dc)_t(1)$. It represents the **tangent vector** to the curve $c$ at the point $p=c(t)$, at time $t$.

</div>

### Pullbacks

Now that definition was a pain to write, so we define a differential 1-form $c^\ast\alpha$ on $[a,b]$ to swallow the entire thing: specifically, we define $c^\ast\alpha$ to be

$$(c^\ast\alpha)_t(\varepsilon) = \alpha_{c(t)}\cdot (Dc)_t(\varepsilon)$$

where $\varepsilon$ is some displacement in time. Thus, we can more succinctly write

$$\int_c\alpha := \int_{[a,b]}c^\ast\alpha$$

This is a special case of a **pullback**: roughly, if $\phi\colon U\to U'$ (where $U\subseteq V$, $U'\subseteq V'$), we can change any differential $k$-form $\alpha$ on $U'$ to a $k$-form on $U$. In particular, if $U=[a,b]$, we can resort to our old definition of an integral.

Let $V$ and $V'$ be finite dimensional real vector spaces (possibly different dimensions) and suppose $U$ and $U'$ are open subsets of each; next, consider a $k$-form $\alpha$ on $U'$.

Given a map $\phi\colon U\to U'$, we now want to define a pullback in much the same way as before. Specifically: $\alpha$ accepts a point in $U'$ and $k$ tangent vectors in $V'$, and returns a real number. We want $\phi^\ast\alpha$ to accept a point in $U$ and $k$ tangent vectors in $V$, and feed the corresponding information to $\alpha$.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Pullback)</span></p>

Given $\phi\colon U\to U'$ and $\alpha$ a $k$-form, we define the **pullback**

$$(\phi^\ast\alpha)_p(v_1,\dots,v_k) := \alpha_{\phi(p)}\left((D\phi)_p(v_1),\dots,(D\phi)_p(v_k)\right)$$

We give the point $q=\phi(p)$. As for the tangent vectors, since we are interested in volume, we take the derivative of $\phi$ at $p$, $(D\phi)_p$, which will scale each of our vectors $v_i$ into some vector in the target $V'$.

</div>

There is a more concrete way to define the pullback using bases. Suppose $w_1,\dots,w_n$ is a basis of $V'$ and $e_1,\dots,e_m$ is a basis of $V$. Thus, the map $\phi\colon V\to V'$ can be thought of as

$$\phi(v)=\phi_1(v)w_1+\cdots+\phi_n(v)w_n$$

where each $\phi_i$ takes in a $v\in V$ and returns a real number. We know also that $\alpha$ can be written concretely as

$$\alpha = \sum_{I\subseteq\lbrace 1,\dots,n\rbrace} f_I\,dw_I$$

Then we define

$$\phi^\ast\alpha = \sum_{I\subseteq\lbrace 1,\dots,n\rbrace}(f_I\circ\phi)(D\phi_{i_1}\wedge\cdots\wedge D\phi_{i_k})$$

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Computation of a Pullback)</span></p>

Let $V=\mathbb R^2$ with basis $\mathbf e_1$ and $\mathbf e_2$, and suppose $\phi\colon V\to V'$ is given by sending

$$\phi(a\mathbf e_1+b\mathbf e_2) = (a^2+b^2)w_1+\log(a^2+1)w_2+b^3w_3$$

where $w_1,w_2,w_3$ is a basis for $V'$. Consider the form $\alpha_q=f(q)w_1^\vee\wedge w_3^\vee$, where $f\colon V'\to\mathbb R$. Then

$$(\phi^\ast\alpha)_p = f(\phi(p))\cdot(2a\mathbf e_1^\vee+2b\mathbf e_2^\vee)\wedge(3b^2\mathbf e_2^\vee) = f(\phi(p))\cdot 6ab^2\cdot\mathbf e_1^\vee\wedge\mathbf e_2^\vee$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Pullback)</span></p>

The pullback basically behaves nicely as possible:

* $\phi^\ast(c\alpha+\beta)=c\phi^\ast\alpha+\phi^\ast\beta$ (linearity)
* $\phi^\ast(\alpha\wedge\beta)=(\phi^\ast\alpha)\wedge(\phi^\ast\beta)$
* $\phi_1^\ast(\phi_2^\ast(\alpha))=(\phi_2\circ\phi_1)^\ast(\alpha)$ (naturality)

</div>

### Cells

*Prototypical example: A disk in $\mathbb R^2$ can be thought of as the cell $[0,R]\times[0,2\pi]\to\mathbb R^2$ by $(r,\theta)\mapsto (r\cos\theta)\mathbf e_1+(r\sin\theta)\mathbf e_2$.*

Now that we have the notion of a pullback, we can define the notion of an integral for more general spaces.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($k$-Cell)</span></p>

A **$k$-cell** is a smooth function $c\colon [a_1,b_1]\times[a_2,b_2]\times\dots[a_k,b_k]\to V$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Examples of Cells)</span></p>

Let $V=\mathbb R^2$ for convenience.

* A 0-cell consists of a single point.
* As we saw, a 1-cell is an arbitrary curve.
* A 2-cell corresponds to a 2-dimensional surface. For example, the map $c\colon [0,R]\times[0,2\pi]\to V$ by

  $$c\colon (r,\theta)\mapsto (r\cos\theta, r\sin\theta)$$

  can be thought of as a disk of radius $R$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Integrating Differential $k$-Forms)</span></p>

Take a differential $k$-form $\alpha$ and a $k$-cell $c\colon [0,1]^k\to V$. Define the integral $\int_c\alpha$ using the pullback

$$\int_c\alpha := \int_{[0,1]^k}c^\ast\alpha$$

Since $c^\ast\alpha$ is a $k$-form on the $k$-dimensional unit box, it can be written as $f(x_1,\dots,x_n)\,dx_1\wedge\cdots\wedge dx_n$. So the above integral could also be written as

$$\int_0^1\cdots\int_0^1 f(x_1,\dots,x_n)\,dx_1\wedge\cdots\wedge dx_n$$

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Area of a Circle)</span></p>

Consider $V=\mathbb R^2$ and let $c\colon (r,\theta)\mapsto (r\cos\theta)\mathbf e_1+(r\sin\theta)\mathbf e_2$ on $[0,R]\times[0,2\pi]$ as before. Take the 2-form $\alpha$ which gives $\alpha_p=\mathbf e_1^\vee\wedge\mathbf e_2^\vee$ at every point $p$. Then

$$c^\ast\alpha = (\cos\theta\,dr - r\sin\theta\,d\theta)\wedge(\sin\theta\,dr + r\cos\theta\,d\theta)$$

$$= r(\cos^2\theta+\sin^2\theta)(dr\wedge d\theta) = r\,dr\wedge d\theta$$

Thus,

$$\int_c\alpha = \int_0^R\int_0^{2\pi} r\,dr\wedge d\theta = \pi R^2$$

which is the area of a circle.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric Intuition for Integration)</span></p>

Given a $k$-cell in $V$, a differential $k$-form $\alpha$ accepts a point $p$ and some tangent vectors $v_1,\dots,v_k$ and spits out a number $\alpha_p(v_1,\dots,v_k)$, which we view as a signed hypervolume. The integral **adds up all these infinitesimals across the entire cell**. In particular, if $V=\mathbb R^k$ and we take the form 

$$\alpha\colon p\mapsto\mathbf e_1^\vee\wedge\cdots\wedge\mathbf e_k^\vee$$

then these $\alpha$'s give the $k$th hypervolume of the cell. For this reason, this $\alpha$ is called the **volume form** on $\mathbb R^k$.

</div>

#### Reparametrization and Orientation

Let $\alpha$ be a $k$-form on $U$ and $c\colon [a_1,b_1]\times\cdots\times[a_k,b_k]\to U$ a $k$-cell. Suppose $\phi\colon [a_1',b_1']\times\dots[a_k',b_k']\to [a_1,b_1]\times\cdots\times[a_k,b_k]$; it is a **reparametrization** if $\phi$ is bijective and $(D\phi)_p$ is always invertible (think "change of variables"); thus

$$c\circ\phi\colon [a_1',b_1']\times\cdots\times[a_k',b_k']\to U$$

is a $k$-cell as well. Then it is said to **preserve orientation** if $\det(D\phi)_p>0$ for all $p$ and **reverse orientation** if $\det(D\phi)_p<0$ for all $p$.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Changing Variables Doesn't Affect Integrals)</span></p>

Let $c$ be a $k$-cell, $\alpha$ a $k$-form, and $\phi$ a reparametrization. Then

$$\int_{c\circ\phi}\alpha = \begin{cases}\int_c\alpha & \phi\text{ preserves orientation}\\[4pt] -\int_c\alpha & \phi\text{ reverses orientation.}\end{cases}$$

*Proof.* Use naturality of the pullback to reduce it to the corresponding theorem in normal calculus.

</div>

### Boundaries

*Prototypical example: The boundary of $[a,b]$ is $\lbrace b\rbrace - \lbrace a\rbrace$. The boundary of a square goes around its edge counterclockwise.*

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($k$-Chain)</span></p>

A **$k$-chain** $U$ is a formal linear combination of $k$-cells over $U$, i.e. a sum of the form

$$c=a_1c_1+\cdots+a_mc_m$$

where each $a_i\in\mathbb R$ and $c_i$ is a $k$-cell. We define $\int_c\alpha=\sum_i a_i\int_{c_i}\alpha$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Boundary of a $k$-Cell)</span></p>

Suppose $c\colon [0,1]^k\to U$ is a $k$-cell. Then the **boundary** of $c$, denoted $\partial c\colon [0,1]^{k-1}\to U$, is the $(k-1)$-chain defined as follows. For each $i=1,\dots,k$ define $(k-1)$-chains by

$$c_i^{\text{start}}\colon (t_1,\dots,t_{k-1})\mapsto c(t_1,\dots,t_{i-1},0,t_i,\dots,t_{k-1})$$

$$c_i^{\text{stop}}\colon (t_1,\dots,t_{k-1})\mapsto c(t_1,\dots,t_{i-1},1,t_i,\dots,t_{k-1})$$

Then

$$\partial c := \sum_{i=1}^k (-1)^{i+1}\left(c_i^{\text{stop}}-c_i^{\text{start}}\right)$$

Finally, the boundary of a chain is the sum of the boundaries of each cell (with the appropriate weights). That is, $\partial\left(\sum a_i c_i\right)=\sum a_i\,\partial c_i$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Boundaries of a Square)</span></p>

Consider the 2-cell $c\colon [0,1]^2\to\mathbb R^2$. Let $p_1,p_2,p_3,p_4$ be the images of $(0,0)$, $(0,1)$, $(1,1)$, $(1,0)$ respectively. Then

$$\partial c = [p_2,p_3]-[p_1,p_4]-[p_4,p_3]+[p_1,p_2]$$

where each "interval" represents a 1-cell on the right. We can take the boundary of this as well, and obtain an empty chain since

$$\partial(\partial c) = \sum_{i=1}^4 \lbrace p_{i+1}\rbrace - \lbrace p_i\rbrace = 0$$

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Boundary of a Unit Disk)</span></p>

Consider the unit disk given by

$$c\colon [0,1]\times[0,1]\to\mathbb R^2 \quad\text{by}\quad (r,\theta)\mapsto r\cos(2\pi\theta)\mathbf e_1+r\sin(2\pi\theta)\mathbf e_2$$

The four parts of the boundary are: two of the arrows more or less cancel each other out when they are integrated. Moreover, we interestingly have a *degenerate* 1-cell at the center of the circle; it is a constant function $[0,1]\to\mathbb R^2$ which always gives the origin.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Boundary of the Boundary Is Empty)</span></p>

$\partial^2=0$, in the sense that for any $k$-chain $c$ we have $\partial^2(c)=0$.

</div>

### Stokes' Theorem

*Prototypical example: $\int_{[a,b]}dg = g(b)-g(a).$*

We now have all the ingredients to state Stokes' theorem for cells.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Stokes' Theorem for Cells)</span></p>

Take $U\subseteq V$ as usual, let $c\colon [0,1]^k\to U$ be a $k$-cell and let $\alpha\colon U\to\bigwedge^{k-1}(V^\vee)$ be a $(k-1)$-form. Then

$$\int_c d\alpha = \int_{\partial c}\alpha$$

In particular, if $d\alpha=0$ then the left-hand side vanishes.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stokes' Theorem Recovers the FTC)</span></p>

If $c$ is the interval $[a,b]$ then $\partial c=\lbrace b\rbrace -\lbrace a\rbrace$, and thus we obtain the fundamental theorem of calculus.

</div>

### Back to Earth: Comparison with Vector Calculus

Now that we've done everything abstractly, let's compare what we've learned to what you might see in $\mathbb R^3$ in a typical vector calculus course.

#### 0-Forms and $n$-Forms

A 0-form is the same as just a function, so the column of $0-D$ integrals is easy to understand: it's just evaluation at a point, or a sum of points.

The case $n=d$ is also easy: we know it's possible to integrate an $n$-form in $\mathbb R^n$ and get a number. That is:

* A normal integral $\int_a^b dx$ is the integral across a 1-cell $[a,b]$ of the 1-form $f\cdot\mathbf e_1^\vee$.
* An area integral $\int_{a_1}^{b_1}\int_{a_2}^{b_2}f(x,y)\,dx\,dy$ corresponds to integrating the 2-form $f\cdot\mathbf e_1^\vee\wedge\mathbf e_2^\vee$.
* A volume integral $\int_{a_1}^{b_1}\int_{a_2}^{b_2}\int_{a_3}^{b_3}f(x,y,z)\,dx\,dy\,dz$ corresponds to integrating the 3-form $f\cdot\mathbf e_1^\vee\wedge\mathbf e_2^\vee\wedge\mathbf e_3^\vee$.

#### 1-Forms and Work Integrals

Consider $d=1$ and $n=3$, i.e. the 3-D line integral. We have a vector-valued function $\mathbf F\colon\mathbb R^3\to\mathbb R^3$. By projection principle, it's the same as the data of

$$\mathbf F(p)=F_1(p)\mathbf e_1+F_2(p)\mathbf e_2+F_3(p)\mathbf e_3$$

To convert the 18.02 notation $\mathbf F(p)$ into Napkin notation, we identify $\mathbf F$ with the differential form

$$\alpha_p = F_1(p)\mathbf e_1^\vee+F_2(p)\mathbf e_2^\vee+F_3(p)\mathbf e_3^\vee$$

Meanwhile, the path $\mathbf r(t)$ parametrized by time $t\in[t_0,t_1]$ matches the concept of a 1-form $c\colon [t_0,t_1]\to\mathbb R^3$. The "work" in the integral is written as

$$\mathbf F(\mathbf r(t))\cdot\mathbf r'(t)$$

but that dot product is exactly the pullback $c^\ast\alpha$.

#### 2-Forms and the Flux Integral

The flux integral, for $d=2$ and $n=3$, is the weirdest case. The parametrization $\mathbf r(u,v)$ is fine, and it corresponds to a 2-cell $c$. But $\mathbf F(p)$ seems to have the wrong type: $\mathbf F(p)=F_1(p)\mathbf e_1+F_2(p)\mathbf e_2+F_3(p)\mathbf e_3$.

There is a fairly weird hack to convert this into Napkin notation: the form desired is

$$\alpha_p = F_1(p)\mathbf e_2^\vee\wedge\mathbf e_3^\vee + F_2(p)\mathbf e_3^\vee\wedge\mathbf e_1^\vee + F_3(p)\mathbf e_1^\vee\wedge\mathbf e_2^\vee$$

For this to be possible, we had to exploit that $\binom{3}{1}=\binom{3}{2}$. That is, the three-dimensional space $\bigwedge^2(\mathbb R^3)$ happens to have the same number of basis elements as $\bigwedge^1(\mathbb R^3)\cong\mathbb R^3$, so the

$$\star\colon\bigwedge^2(\mathbb R^3)\to\mathbb R^3$$

defined by $\mathbf e_1\wedge\mathbf e_2\mapsto\mathbf e_3$, $\mathbf e_2\wedge\mathbf e_3\mapsto\mathbf e_1$, $\mathbf e_3\wedge\mathbf e_1\mapsto\mathbf e_2$ is really an isomorphism. We denote this map by $\star$, because it turns out this map generalizes to the so-called **Hodge star operator** in higher dimensions.

With this, the weird dot-cross product $\mathbf F(\mathbf r(u,v))\cdot(\mathbf r_u\times\mathbf r_v)\,du\,dv$ is now rigged to correspond to the pullback $c^\ast\alpha$. So using this Hodge star, we find that **flux is actually the integration of a 2-form**.

#### Exterior Derivatives

Every red arrow in the 18.02 chart corresponds to the exterior derivative of the corresponding form. That is:

* The "grad" operation takes a 0-form $f$ and outputs a vector field corresponding to the 1-form $df$.
* The "curl" operation takes a 1-form $\alpha$ and outputs a vector field corresponding to the 2-form $d\alpha$. When $n=3$, this checks out because the space of 1-forms is $\binom{3}{1}$-dimensional, and the space of 2-forms is $\binom{3}{2}$, and thankfully $\binom{3}{1}=3=\binom{3}{2}$.

  The weird notation $\nabla\times\mathbf F$ can be checked to correspond to the exterior derivative. On the 18.02 side, if we have

  $$\mathbf F=F_1\mathbf e_1+F_2\mathbf e_2+F_3\mathbf e_3$$

  then the 18.02 definition of curl is

  $$\operatorname{curl}(\mathbf F):=\nabla\times\mathbf F:=\left(\frac{\partial F_3}{\partial y}-\frac{\partial F_2}{\partial z}\right)\mathbf e_1+\left(\frac{\partial F_1}{\partial z}-\frac{\partial F_3}{\partial x}\right)\mathbf e_2+\left(\frac{\partial F_2}{\partial x}-\frac{\partial F_1}{\partial y}\right)\mathbf e_3$$

  Now to convert $\mathbf F$ into Napkin notation, remember we identified $\mathbf F$ with the differential form $\alpha=F_1\mathbf e_1^\vee+F_2\mathbf e_2^\vee+F_3\mathbf e_3^\vee$. Computing the exterior derivative,

  $$d\alpha = dF_1\wedge\mathbf e_1^\vee+dF_2\wedge\mathbf e_2^\vee+dF_3\wedge\mathbf e_3^\vee$$

  $$= \left(\frac{\partial F_3}{\partial y}-\frac{\partial F_2}{\partial z}\right)\mathbf e_2^\vee\wedge\mathbf e_3^\vee+\left(\frac{\partial F_1}{\partial z}-\frac{\partial F_3}{\partial x}\right)\mathbf e_3^\vee\wedge\mathbf e_1^\vee+\left(\frac{\partial F_2}{\partial x}-\frac{\partial F_1}{\partial y}\right)\mathbf e_1^\vee\wedge\mathbf e_2^\vee$$

  Taking the Hodge star and dropping all the $\vee$'s gives the same thing as $\nabla\times\mathbf F$, so this completes the correspondence between the 18.02 notation and the Napkin notation.

* In 18.02 terminology, the divergence div is defined by

  $$\operatorname{div}(\mathbf F):=\nabla\cdot\mathbf F:=\frac{\partial F_1}{\partial x}+\frac{\partial F_2}{\partial y}+\frac{\partial F_3}{\partial z}$$

  which is a scalar-valued function for input points $p\in\mathbb R^3$.

#### Double Derivative

We know that $d^2=0$, which in the 18.02 chart means composing two arrows gives zero. You'll see this as:

* The curl of a gradient is zero.
* The flux of a curl is zero.

But really they're the same theorem.

#### Stokes' Theorem in 18.02

Each red arrow also gives an instance of Stokes' theorem for cells. So Stokes' theorem even for cells is really great, because we get six 18.02 theorems as special cases:

* The three arrows from 0-D integrals to 1-D integrals are all called "Fundamental Theorem of Calculus". Some authors say "Fundamental Theorem of Calculus for line integrals" instead for $n>1$.
* For $n=2$, the other red arrow is called "**Green's theorem**."
* For $n=3$, the arrow from work to flux is confusingly also called "**Stokes' theorem**"; it says the flux of a 2-D surface equals the work on the 1-D boundary.
* The rightmost red arrow for $n=3$ is called the "**divergence theorem**"; it says the divergence of a 3-D volume equals the flux of the 2-D boundary surface.

---

## A Bit of Manifolds

Last chapter, we stated Stokes' theorem for cells. It turns out there is a much larger class of spaces, the so-called *smooth manifolds*, for which this makes sense.

### Topological Manifolds

*Prototypical example: $S^2$: "the Earth looks flat."*

Long ago, people thought the Earth was flat, i.e. homeomorphic to a plane, and in particular they thought that $\pi_2(\text{Earth})=0$. But in fact, the Earth is actually a sphere, which is not contractible and in particular $\pi_2(\text{Earth})\cong\mathbb Z$. This observation underlies the definition of a manifold:

> **An $n$-manifold is a space which locally looks like $\mathbb R^n$.**

There are two ways to think about a topological manifold $M$:

* **"Locally"**: at every point $p\in M$, some open neighborhood of $p$ looks like an open set of $\mathbb R^n$. For example, to someone standing on the surface of the Earth, the Earth looks much like $\mathbb R^2$.
* **"Globally"**: there exists an open cover of $M$ by open sets $\lbrace U_i\rbrace_i$ (possibly infinite) such that each $U_i$ is homeomorphic to some open subset of $\mathbb R^n$. For example, from outer space, the Earth can be covered by two hemispherical pancakes.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Topological $n$-Manifold)</span></p>

A **topological $n$-manifold** $M$ is a Hausdorff space with an open cover $\lbrace U_i\rbrace$ of sets homeomorphic to subsets of $\mathbb R^n$, say by homeomorphisms

$$\phi_i\colon U_i\xrightarrow{\cong} E_i\subseteq\mathbb R^n$$

where each $E_i$ is an open subset of $\mathbb R^n$. Each $\phi_i\colon U_i\to E_i$ is called a **chart**, and together they form a so-called **atlas**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intrinsic Definition)</span></p>

This definition is nice because it doesn't depend on embeddings: a manifold is an *intrinsic* space $M$, rather than a subset of $\mathbb R^N$ for some $N$. Analogy: an abstract group $G$ is an intrinsic object rather than a subgroup of $S_n$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(An Atlas on $S^1$)</span></p>

An atlas for $S^1$ can be built with just two open sets $U_1$ and $U_2$, each homeomorphic to an open interval in $\mathbb R$. The two sets overlap in two disjoint arcs.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Some Examples of Topological Manifolds)</span></p>

* The sphere $S^2$ is a 2-manifold: every point in the sphere has a small open neighborhood that looks like $D^2$. One can cover the Earth with just two hemispheres, and each hemisphere is homeomorphic to a disk.
* The circle $S^1$ is a 1-manifold; every point has an open neighborhood that looks like an open interval.
* The torus, Klein bottle, $\mathbb{RP}^2$ are all 2-manifolds.
* $\mathbb R^n$ is trivially a manifold, as are its open sets.

All these spaces are compact except $\mathbb R^n$.

A non-example of a manifold is $D^n$, because it has a **boundary**; points on the boundary do not have open neighborhoods that look Euclidean.

</div>

### Smooth Manifolds

*Prototypical example: All the topological manifolds.*

Let $M$ be a topological $n$-manifold with atlas $\lbrace U_i\xrightarrow{\phi_i}E_i\rbrace$.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Transition Map)</span></p>

For any $i,j$ such that $U_i\cap U_j\neq\varnothing$, the **transition map** $\phi_{ij}$ is the composed map

$$\phi_{ij}\colon E_i\cap\phi_i^{\text{img}}(U_i\cap U_j)\xrightarrow{\phi_i^{-1}}U_i\cap U_j\xrightarrow{\phi_j}E_j\cap\phi_j^{\text{img}}(U_i\cap U_j)$$

The transition map is just the natural way to go from $E_i\to E_j$, restricted to overlaps.

</div>

We want to add enough structure so that we can use differential forms.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Smooth Manifold)</span></p>

We say $M$ is a **smooth manifold** if all its transition maps are smooth.

This definition makes sense, because we know what it means for a map between two open sets of $\mathbb R^n$ to be differentiable.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Porting Definitions to Manifolds)</span></p>

With smooth manifolds we can try to port over definitions that we built for $\mathbb R^n$ onto our manifolds. So in general, all definitions involving smooth manifolds will reduce to something on each of the coordinate charts, with a compatibility condition.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Smooth Map)</span></p>

**(a)** Let $M$ be a smooth manifold. A continuous function $f\colon M\to\mathbb R$ is called **smooth** if the composition

$$E_i\xrightarrow{\phi_i^{-1}} U_i\hookrightarrow M\xrightarrow{f}\mathbb R$$

is smooth as a function $E_i\to\mathbb R$.

**(b)** Let $M$ and $N$ be smooth with atlases $\lbrace U_i^M\xrightarrow{\phi_i}E_i^M\rbrace_i$ and $\lbrace U_j^N\xrightarrow{\phi_j}E_j^N\rbrace_j$. A map $f\colon M\to N$ is **smooth** if for every $i$ and $j$, the composed map

$$E_i\xrightarrow{\phi_i^{-1}}U_i\hookrightarrow M\xrightarrow{f}N\twoheadrightarrow U_j\xrightarrow{\phi_j}E_j$$

is smooth, as a function $E_i\to E_j$.

</div>

### Regular Value Theorem

*Prototypical example: $x^2+y^2=1$ is a circle!*

Despite all that we've written about general manifolds, it would be sort of mean to leave you without telling you how to actually construct manifolds in practice, even though we know the circle $x^2+y^2=1$ is a great example of a one-dimensional manifold embedded in $\mathbb R^2$.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Regular Value Theorem)</span></p>

Let $V$ be an $n$-dimensional real normed vector space, let $U\subseteq V$ be open and let $f_1,\dots,f_m\colon U\to\mathbb R$ be smooth functions. Let $M$ be the set of points $p\in U$ such that $f_1(p)=\cdots=f_m(p)=0$.

Assume $M$ is nonempty and that the map

$$V\to\mathbb R^m\quad\text{by}\quad v\mapsto\left((Df_1)_p(v),\dots,(Df_m)_p(v)\right)$$

has rank $m$, for every point $p\in M$. Then $M$ is a manifold of dimension $n-m$.

</div>

One very common special case is to take $m=1$ above.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Level Hypersurfaces)</span></p>

Let $V$ be a finite-dimensional real normed vector space, let $U\subseteq V$ be open and let $f\colon U\to\mathbb R$ be smooth. Let $M$ be the set of points $p\in U$ such that $f(p)=0$. If $M\neq\varnothing$ and $(Df)_p$ is not the zero map for any $p\in M$, then $M$ is a manifold of dimension $\dim V-1$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Circle $x^2+y^2-c=0$)</span></p>

Let $f(x,y)=x^2+y^2-c$, $f\colon\mathbb R^2\to\mathbb R$, where $c$ is a positive real number. Note that

$$Df = 2x\cdot dx + 2y\cdot dy$$

which in particular is nonzero as long as $(x,y)\neq(0,0)$, i.e. as long as $c\neq 0$. Thus:

* When $c>0$, the resulting curve -- a circle with radius $\sqrt{c}$ -- is a one-dimensional manifold, as we knew.
* When $c=0$, the result fails. Indeed, $M$ is a single point, which is actually a zero-dimensional manifold!

</div>

### Differential Forms on Manifolds

We already know what a differential form is on an open set $U\subseteq\mathbb R^n$. So, we naturally try to port over the definition of differentiable form on each subset, plus a compatibility condition.

Let $M$ be a smooth manifold with atlas $\lbrace U_i\xrightarrow{\phi_i}E_i\rbrace_i$.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Differential $k$-Form on a Manifold)</span></p>

A **differential $k$-form** $\alpha$ on a smooth manifold $M$ is a collection $\lbrace\alpha_i\rbrace_i$ of differential $k$-forms on each $E_i$, such that for any $j$ and $i$ we have that

$$\alpha_j = \phi_{ij}^\ast(\alpha_i)$$

In English: we specify a differential form on each chart, which is compatible under pullbacks of the transition maps.

</div>

### Orientations

*Prototypical example: Left versus right, clockwise vs. counterclockwise.*

This still isn't enough to integrate on manifolds. We need one more definition: that of an orientation.

The main issue is the observation from standard calculus that

$$\int_a^b f(x)\,dx = -\int_b^a f(x)\,dx$$

Consider then a space $M$ which is homeomorphic to an interval. If we have a 1-form $\alpha$, how do we integrate it over $M$? Since $M$ is just a topological space (rather than a subset of $\mathbb R$), there is no default "left" or "right" that we can pick. As another example, if $M=S^1$ is a circle, there is no default "clockwise" or "counterclockwise" unless we decide to embed $M$ into $\mathbb R^2$.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Orientable Manifold, Volume Form)</span></p>

A smooth $n$-manifold is **orientable** if there exists a differential $n$-form $\omega$ on $M$ such that for every $p\in M$,

$$\omega_p\neq 0$$

Recall here that $\omega_p$ is an element of $\bigwedge^n(V^\vee)$. In that case we say $\omega$ is a **volume form** of $M$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Picturing Orientability)</span></p>

If we recall that a differential form is supposed to take tangent vectors of $M$ and return real numbers, we can think of each point $p\in M$ as having a **tangent plane** $T_p(M)$ which is $n$-dimensional. Now since the volume form $\omega$ is $n$-dimensional, it takes an entire basis of the $T_p(M)$ and gives a real number. So a manifold is orientable if there exists a consistent choice of sign for the basis of tangent vectors at every point of the manifold.

For "embedded manifolds," this just amounts to being able to pick a nonzero field of normal vectors to each point $p\in M$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Orientable Surfaces)</span></p>

* Spheres $S^n$, planes, and the torus $S^1\times S^1$ are orientable.
* The Mobius strip and Klein bottle are *not* orientable: they are "one-sided."
* $\mathbb{CP}^n$ is orientable for any $n$.
* $\mathbb{RP}^n$ is orientable only for odd $n$.

</div>

### Stokes' Theorem for Manifolds

Stokes' theorem in the general case is based on the idea of a **manifold with boundary** $M$, which we won't define, other than to say its boundary $\partial M$ is an $n-1$ dimensional manifold, and that it is oriented if $M$ is oriented. An example is $M=D^2$, which has boundary $\partial M=S^1$.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Support of a Differential Form)</span></p>

The **support** of a differential form $\alpha$ on $M$ is the closure of the set

$$\lbrace p\in M\mid\alpha_p\neq 0\rbrace$$

If this support is compact as a topological space, we say $\alpha$ is **compactly supported**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Stokes' Theorem for Manifolds)</span></p>

Let $M$ be a smooth oriented $n$-manifold with boundary and let $\alpha$ be a compactly supported $(n-1)$-form. Then

$$\int_M d\alpha = \int_{\partial M}\alpha$$

</div>

### (Optional) The Tangent and Cotangent Space

*Prototypical example: Draw a line tangent to a circle, or a plane tangent to a sphere.*

Let $M$ be a smooth manifold and $p\in M$ a point. One of the points of all this manifold stuff is that we really want to see the manifold as an *intrinsic object*, in its own right, rather than as embedded in $\mathbb R^n$. So, we would like our notion of a tangent vector to not refer to an ambient space, but only to intrinsic properties of the manifold $M$ in question.

#### Tangent Space

To motivate this construction, let us start with an embedded case for which we know the answer already: a sphere.

Suppose $f\colon S^2\to\mathbb R$ is a function on a sphere, and take a point $p$. Near the point $p$, $f$ looks like a function on some open neighborhood of the origin. Thus we can think of taking a *directional derivative* along a vector $\vec{v}$ in the imagined tangent plane (i.e. some partial derivative). For a fixed $\vec{v}$ this partial derivative is a linear map

$$D_{\vec{v}}\colon C^\infty(M)\to\mathbb R$$

It turns out this goes the other way: if you know what $D_{\vec{v}}$ does to every smooth function, then you can recover $v$. This is the trick we use in order to create the tangent space. Rather than trying to specify a vector $\vec{v}$ directly (which we can't do because we don't have an ambient space),

> **The vectors *are* partial-derivative-like maps.**

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Derivation)</span></p>

A **derivation** $D$ at $p$ is a linear map $D\colon C^\infty(M)\to\mathbb R$ (i.e. assigning a real number to every smooth $f$) satisfying the following Leibniz rule: for any $f,g$ we have the equality

$$D(fg)=f(p)\cdot D(g)+g(p)\cdot D(f)\in\mathbb R$$

This is just a "product rule."

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Tangent Vector, Tangent Space)</span></p>

A **tangent vector** is just a derivation at $p$, and the **tangent space** $T_p(M)$ is simply the set of all these tangent vectors.

</div>

#### Cotangent Space

The product rule for $D$ is equivalent to the following three conditions:

1. $D$ is linear, meaning $D(af+bg)=aD(f)+bD(g)$.
2. $D(1_M)=0$, where $1_M$ is the constant function on $M$.
3. $D(fg)=0$ whenever $f(p)=g(p)=0$. Intuitively, this means that if a function $h=fg$ vanishes to second order at $p$, then its derivative along $D$ should be zero.

This suggests a third equivalent definition: suppose we define

$$\mathfrak m_p := \lbrace f\in C^\infty M\mid f(p)=0\rbrace$$

to be the set of functions which vanish at $p$ (this is called the *maximal ideal* at $p$). In that case,

$$\mathfrak m_p^2 = \left\lbrace\sum_i f_i\cdot g_i\mid f_i(p)=g_i(p)=0\right\rbrace$$

is the set of functions vanishing to second order at $p$. Thus, a tangent vector is really just a linear map

$$\mathfrak m_p/\mathfrak m_p^2\to\mathbb R$$

In other words, the tangent space is actually the dual space of $\mathfrak m_p/\mathfrak m_p^2$; for this reason, the space $\mathfrak m_p/\mathfrak m_p^2$ is defined as the **cotangent space** (the dual of the tangent space). This definition is even more abstract than the one with derivations above, but has some nice properties:

* it is coordinate-free, and
* it's defined only in terms of the smooth functions $M\to\mathbb R$, which will be really helpful later on in algebraic geometry when we have varieties or schemes and can repeat this definition.

#### Sanity Check

With all these equivalent definitions, the last thing we should do is check that this definition of tangent space actually gives a vector space of dimension $n$. To do this it suffices to show verify this for open subsets of $\mathbb R^n$, which will imply the result for general manifolds $M$ (which are locally open subsets of $\mathbb R^n$). Using some real analysis, one can prove the following result:

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Tangent Space of $\mathbb R^n$)</span></p>

Suppose $M\subset\mathbb R^n$ is open and $0\in M$. Then

$$\mathfrak m_0 = \lbrace\text{smooth functions } f\mid f(0)=0\rbrace$$

$$\mathfrak m_0^2 = \lbrace\text{smooth functions } f\mid f(0)=0, (\nabla f)_0=0\rbrace$$

In other words $\mathfrak m_0^2$ is the set of functions which vanish at $0$ and such that all first derivatives of $f$ vanish at zero.

Thus, it follows that there is an isomorphism

$$\mathfrak m_0/\mathfrak m_0^2\cong\mathbb R^n\quad\text{by}\quad f\mapsto\left[\frac{\partial f}{\partial x_1}(0),\dots,\frac{\partial f}{\partial x_n}(0)\right]$$

and so the cotangent space, hence tangent space, indeed has dimension $n$.

</div>
