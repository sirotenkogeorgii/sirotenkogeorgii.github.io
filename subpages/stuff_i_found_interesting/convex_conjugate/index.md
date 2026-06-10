---
layout: default
title: "Convex Conjugate"
tags:
  - convex-analysis
  - optimization
  - legendre-transform
  - duality
  - optimal-transport
  - subgradients
---
<figure>
  <img src="{{ '/assets/images/notes/stuff_i_found_interesting/Drini-nonuniformconvergence.png' | relative_url }}"" loading="lazy">
  <figcaption>The limit of a sequence of continuous functions does not have to be continuous: the sequence of functions $f_{n}(x)=\sin ^{n}(x)$ (marked in green and blue) converges pointwise over the entire domain, but the limit function is discontinuous (marked in red).</figcaption>
</figure>

The **convex conjugate** of a function

$$f:\mathbb{R}^d\to (-\infty,+\infty]$$

is the function $f^\ast:\mathbb{R}^d\to (-\infty,+\infty]$ defined by

$$f^*(p)=\sup_{x\in\mathbb{R}^d}\bigl(\langle p,x\rangle-f(x)\bigr).$$

It is also called the **Legendre–Fenchel transform**.

Intuitively, $f^\ast(p)$ asks:

> Among all points $x$, how much can the linear function $\langle p,x\rangle$ exceed $f(x)$?

So $f^\ast(p)$ measures the **best possible gap** between the linear function with slope $p$ and the function $f$.

## Geometric interpretation

Rewrite the definition:

$$f^*(p)=\sup_x \bigl(\langle p,x\rangle-f(x)\bigr).$$

This means that for every $x$,

$$\langle p,x\rangle-f(x)\leq f^*(p).$$

Equivalently,

$$f(x)\geq \langle p,x\rangle-f^*(p).$$

So for each $p$, the expression

$$x\mapsto \langle p,x\rangle-f^*(p)$$

is an affine function lying below $f$.

Thus $f^\ast(p)$ tells you **how low you must shift the affine function with slope $p$** so that it lies below $f$.

For one-dimensional $f$, this becomes:

$$f^*(p) = \sup_x (px-f(x))$$

The line

$$x \mapsto px-f^*(p)$$

has slope $p$ and supports $f$ from below.

## Main intuition

The convex conjugate describes a convex function not by its values at points $x$, but by its **supporting hyperplanes**.

Instead of asking:

> What is the value of $f$ at position $x$?

the conjugate asks:

> What affine lower bound with slope $p$ can support $f$?

So $f$ is described in the “position variable” $x$, while $f^\ast$ is described in the “slope variable” $p$.

## Example

Take

$$f(x)=\frac12 x^2.$$

Then

$$f^*(p)=\sup_x \left(px-\frac12x^2\right).$$

Maximize in $x$:

$$\frac{d}{dx}\left(px-\frac12x^2\right)=p-x.$$

So the maximum occurs at $x=p$. Therefore

$$f^*(p)=p^2-\frac12p^2=\frac12p^2.$$

Hence

$$f^*(p)=\frac12p^2.$$

The quadratic is self-dual under convex conjugation.

## Why it is useful

The convex conjugate converts a convex function into a dual object. It is central in optimization because many problems can be rewritten in terms of their conjugates.

The key inequality is the **Fenchel inequality**:

$$\langle p,x\rangle \leq f(x)+f^*(p).$$

This follows immediately from the definition:

$$f^*(p)=\sup_z(\langle p,z\rangle-f(z)) \geq \langle p,x\rangle-f(x).$$

Rearranging gives

$$\langle p,x\rangle\leq f(x)+f^\ast(p).$$

Equality holds when $p$ is a subgradient of $f$ at $x$:

$$p\in \partial f(x).$$

So the conjugate captures the relationship between points $x$ and slopes/subgradients $p$.

## In optimal transport

For the quadratic cost, convex conjugates appear because inequalities such as

$$\phi(x)+\psi(y)\leq x\cdot y$$

can be rewritten using conjugacy.

For example,

$$\phi(x)+\psi(y)\leq x\cdot y$$

is equivalent to

$$\psi(y)\leq x\cdot y-\phi(x)$$

for all $x$. Therefore

$$\psi(y)\leq \inf_x(x\cdot y-\phi(x)).$$

This is closely related to the negative of a convex conjugate:

$$\inf_x(x\cdot y-\phi(x)) = -\sup_x(\phi(x)-x\cdot y).$$

So convex conjugates naturally appear when optimizing over pairs of functions constrained by affine inequalities.

## Conjugating twice

For a convex function, taking the conjugate twice brings you right back to the original function $f^{**} = f$. It means the function can be perfectly defined by its set of tangent lines, just like a polygon can be defined by its flat sides.

## Comments

> can we say that $f^\ast(p)$ gives the minimal shift below of the linear function with slope $p$, such that the linear function is below the function $f$? The mapping $x \mapsto \langle p,x\rangle-f^\ast(p)$ gives the value on the value of the shifted linear function with slope $p$ at the point $x$?

Yes. More precisely:

For fixed $p$, consider affine functions with slope $p$:

$$x \mapsto \langle p,x\rangle - c.$$

We want this affine function to lie below $f$, i.e.

$$
\langle p,x\rangle - c \leq f(x)
\qquad \text{for all }x.
$$

Rearranging,

$$
\langle p,x\rangle - f(x) \leq c
\qquad \text{for all }x.
$$

Therefore $c$ must satisfy

$$c \geq \sup_x \bigl(\langle p,x\rangle - f(x)\bigr).$$

But the right-hand side is exactly

$$f^*(p)=\sup_x\bigl(\langle p,x\rangle-f(x)\bigr).$$

So yes:

$$f^*(p)$$

is the **minimal vertical shift amount** $c$ such that

$$x\mapsto \langle p,x\rangle - c$$

lies below $f$.

Equivalently,

$$x\mapsto \langle p,x\rangle - f^*(p)$$

is the **highest affine function with slope $p$** that still lies below $f$.

Your second sentence is also correct:

$$x\mapsto \langle p,x\rangle - f^*(p)$$

gives the value at $x$ of the shifted linear function with slope $p$.

One small nuance: it is technically an **affine** function, not purely linear, because it has a vertical shift/intercept:

$$\langle p,x\rangle - f^*(p).$$

So the clean interpretation is:

> $f^\ast(p)$ tells us how much we must shift the linear function $x\mapsto \langle p,x\rangle$ downward so that it becomes the highest affine lower bound of $f$ with slope $p$.

If the supremum is attained at some $x_0$, then this affine function touches $f$ at $x_0$:

$$f(x_0)=\langle p,x_0\rangle-f^*(p).$$

If the supremum is not attained, then it lies below $f$ but may not touch it anywhere.

## The supporting-hyperplane interpretation is naturally convex

## Connection to subgradient 

For the statement

$$
\langle p,x\rangle=f(x)+f^*(p)
\quad\Longleftrightarrow\quad
p\in \partial f(x),
$$

you are right to focus on convexity.

More precisely:

For a **proper convex function** $f$, the convex-analysis subgradient is defined by

$$p\in \partial f(x)$$

iff

$$f(z)\geq f(x)+\langle p,z-x\rangle\qquad \text{for all }z.$$

This means: the affine function

$$z\mapsto f(x)+\langle p,z-x\rangle$$

is a **global supporting affine lower bound** to $f$ at $x$.

Now compare this with Fenchel equality.

Starting from

$$f^*(p)=\sup_z(\langle p,z\rangle-f(z)).$$

Equality

$$\langle p,x\rangle=f(x)+f^*(p)$$

means

$$f^*(p)=\langle p,x\rangle-f(x).$$

So the supremum defining $f^*(p)$ is attained at $x$. Therefore, for every $z$,

$$
\langle p,z\rangle-f(z)
\leq
\langle p,x\rangle-f(x).
$$

Rearrange:

$$f(z)\geq f(x)+\langle p,z-x\rangle.$$

But this is exactly

$$p\in\partial f(x).$$

So the equivalence is:

$$
\boxed{
\langle p,x\rangle=f(x)+f^*(p)
\iff
p\in\partial f(x)
}
$$

where $\partial f(x)$ means the **convex subdifferential**, i.e. global supporting slopes.

The subtle point is:

* The algebraic equivalence with a **global supporting affine lower bound** does not really need convexity.
* But calling this object the usual **subgradient** is mainly a convex-analysis notion.
* For convex functions, local slope information and global support fit together nicely.
* For non-convex functions, a derivative/gradient at $x$ generally does **not** give such a global lower support.

So yes: the clean “$p$ is a subgradient” statement is meant in the convex setting.


<figure>
  <img src="{{ '/assets/images/notes/stuff_i_found_interesting/Drini-convex_linear_support.png' | relative_url }}"" loading="lazy">
  <!-- <figcaption>The limit of a sequence of continuous functions does not have to be continuous: the sequence of functions $f_{n}(x)=\sin ^{n}(x)$ (marked in green and blue) converges pointwise over the entire domain, but the limit function is discontinuous (marked in red).</figcaption> -->
</figure>

<figure>
  <img src="{{ '/assets/images/notes/stuff_i_found_interesting/Geometrical-interpretation-of-the-conjugate-function-and-the-convex-envelope-Color.gif.png' | relative_url }}"" loading="lazy">
  <figcaption>Geometrical interpretation of the conjugate function and the convex envelope (Color figure online)</figcaption>
</figure>