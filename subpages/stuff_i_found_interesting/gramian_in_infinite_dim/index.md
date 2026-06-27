---
layout: note
title: "The Gramian in Infinite Dimensions"
date: 2026-06-27
math: true
tags:
  - functional-analysis
  - linear-algebra
  - kernel-methods
  - rkhs
  - inner-product-spaces
  - machine-learning
---

This started as an innocent question — what is the infinite-dimensional analogue of a Gram matrix? — and quietly turned into something more interesting. Chasing the answer forces a clean separation between two ideas that finite-dimensional intuition keeps glued together: *features* and *basis vectors*. By the end it also re-casts the inner product itself as a Gramian, and drops the whole construction into the taxonomy of bilinear forms. These are the notes I wish I had had at the start.

## From a Gram matrix to a Gram operator

In finite dimensions the Gramian of a list of vectors $v\_1, \dots, v\_n$ is the matrix of all their pairwise inner products,

$$G_{ij} = \langle v_i, v_j \rangle.$$

To pass to infinite dimensions, replace the discrete list by a continuum of elements and the matrix by a continuous operator.

| Feature | Finite (Gram matrix) | Infinite (Gram operator) |
|---|---|---|
| Elements | a finite set $\{v\_1, \dots, v\_n\}$ | a continuum of profiles $\{\phi\_x\}\_{x \in \Omega}$ |
| Indexing | integer rows/columns $(i, j)$ | continuous indices $(x, y)$ |
| Entry / action | $G\_{ij} = \langle v\_i, v\_j \rangle$ | $(Kf)(x) = \int k(x,y)\, f(y)\, dy$ |
| Structure | symmetric (Hermitian) PSD matrix | self-adjoint, positive operator (compact under Mercer-type hypotheses) |

When the "vectors" are functions in a space like $L^2$, the discrete entries $G\_{ij}$ become a continuous **kernel** $k(x,y)$, and matrix multiplication becomes integration against that kernel:

$$(Kf)(x) = \int k(x,y)\, f(y)\, dy, \qquad k(x,y) = \langle \phi_x, \phi_y \rangle.$$

### The integral is a matrix–vector product, not a single inner product

A natural first guess is that the infinite analogue should just be $\langle x, y\rangle\, y$. It is not. The way to see this is to discretise: replace the continuous variables by indices and the integral by a sum.

* functions become vectors: $f(y) \to v\_j$ and $(Kf)(x) \to (Gv)\_i$;
* the kernel becomes the matrix: $k(x,y) \to G\_{ij} = \langle \phi\_i, \phi\_j \rangle$;
* the integral becomes the sum:

$$(Gv)_i = \sum_j G_{ij}\, v_j.$$

So the kernel integral is the *entire* matrix–vector multiplication. The value at $x$ is the continuous dot product of the "row" $k(x,\cdot)$ with the input $f(\cdot)$ — not a single rank-one term $\langle v, u\rangle\, u$ — a rank-one operator (an orthogonal projection only when $\|u\| = 1$), an altogether different object.

$$
\begin{array}{ccc}
\textbf{Infinite dimensions} & & \textbf{Finite dimensions} \\
(Kf)(x) & \longleftrightarrow & (Gv)_i \\
\big\downarrow & & \big\downarrow \\
\displaystyle\int k(x,y)\, f(y)\, dy & \longleftrightarrow & \displaystyle\sum_j G_{ij}\, v_j
\end{array}
$$

## What the kernel actually represents: feature maps

The kernel is an inner product of **features**:

$$k(x,y) = \langle \Phi(x), \Phi(y)\rangle.$$

A feature map $\Phi$ sends an input to a vector (or function) in some Hilbert space $\mathcal H$, and the kernel reads off the overlap of two such features. The finite Gramian is the same statement with a discrete index:

* **Finite.** The index set is $\{1, \dots, n\}$. The feature map evaluated at $i$ returns the $i$-th vector, $\Phi(i) = \phi\_i$, so $G\_{ij} = \langle \Phi(i), \Phi(j)\rangle = \langle \phi\_i, \phi\_j\rangle$.
* **Infinite.** The index set is continuous, say $\mathbb R$. At a coordinate $x$ the feature map returns a continuous profile $\Phi(x) = \phi\_x \in \mathcal H$, and $k(x,y) = \langle \phi\_x, \phi\_y\rangle$.

### The reproducing-kernel reading

In a **reproducing kernel Hilbert space (RKHS)** the kernel acquires a concrete meaning. Evaluating a function at a point, $f \mapsto f(x)$, is a bounded linear functional, so by the Riesz representation theorem there is a unique element $k(\cdot, x) \in \mathcal H$ with $f(x) = \langle f, k(\cdot, x)\rangle$. Taking $f = k(\cdot, y)$ gives

$$k(x,y) = \langle k(\cdot, x), k(\cdot, y)\rangle.$$

So the feature of $x$ is literally its point-evaluation representer. If $k(x,y)$ is large, the function values at $x$ and $y$ are strongly coupled; if it is zero, those two positions are orthogonal — a function can do whatever it likes at one without any constraint from the other.

### A concrete example: polynomial features

Let $\Phi$ extract powers of a scalar $x$. Stop at degree two, $\Phi(x) = (1,\ x,\ x^2)^{\top}$, and the kernel is a finite dot product:

$$k(x,y) = 1 + xy + x^2 y^2.$$

Now let the powers run to infinity with Taylor weights,

$$\Phi(x) = \Big(1,\ \tfrac{x}{\sqrt{1!}},\ \tfrac{x^2}{\sqrt{2!}},\ \tfrac{x^3}{\sqrt{3!}},\ \dots\Big)^{\top}.$$

The inner product becomes an infinite series that sums in closed form:

$$k(x,y) = \sum_{n=0}^{\infty} \frac{(xy)^n}{n!} = e^{xy}.$$

Same recipe, finite or infinite: plug the index into $\Phi$, take the inner product, read off the Gramian entry.

The same machinery explains the popular **Gaussian RBF kernel** $k(x,y) = e^{-\|x - y\|^2}$. Factoring

$$e^{-\|x - y\|^2} = e^{-\|x\|^2}\, e^{-\|y\|^2}\, e^{2\langle x, y\rangle}$$

and Taylor-expanding the last factor shows its feature map carries *every* polynomial power of the input, each damped by a Gaussian envelope. A tempting but incorrect gloss is that this space therefore "contains all smooth functions" — it does not; the Gaussian RKHS is a comparatively small space of real-analytic functions. What is true is **universality**: the space is *dense* in the continuous functions on any compact set, so it can approximate them arbitrarily well rather than represent them exactly.

#TODO: add visualizations (a finite feature vector growing into an infinite feature profile)

## The basis illusion: features are not basis vectors

Here is the question that drives the whole topic. In the finite picture the "feature of $i$" was just the basis vector $\phi\_i$ — rigid, one per axis. In the infinite picture the "feature of $y$" was *any* function of $y$. Why the asymmetry?

There is none. The asymmetry is an artifact of how we chose to index, not a fact about dimension.

### You can use arbitrary features in finite dimensions too

Nothing stops you from mapping a continuous input $y$ into a finite feature space with whatever functions you like:

$$\Phi(y) = \begin{bmatrix} f_1(y) \\ \vdots \\ f_d(y) \end{bmatrix}.$$

For instance $f\_1(y) = \sin y$, $f\_2(y) = y^2$, $f\_3(y) = e^y$ gives a perfectly good $3$-dimensional Gramian

$$G(x,y) = \langle \Phi(x), \Phi(y)\rangle = \sin x \sin y + x^2 y^2 + e^x e^y,$$

with features that are *functions*, in a space that is firmly finite-dimensional. The "feature equals basis vector" picture only ever held because we indexed by integers, which makes $\Phi(i) = \phi\_i$ look like a coordinate axis.

### Features are not constrained like a basis

Once the two ideas are separated, the constraints fall away. A basis must be linearly independent and span the space, with exactly $\dim V$ elements. Features obey none of this:

* they can be **linearly dependent** — area in square feet and area in square metres are perfectly correlated, yet both are legitimate features with a perfectly good Gramian;
* there can be **more features than dimensions** — map a 2-D point into a 100-dimensional feature space;
* or **fewer** — extract 2 features from a 10-dimensional space.

In every case $\Phi(x\_i)$ is a data point written in "feature language," not a coordinate axis. The Gramian $G\_{ij} = \langle \Phi(x\_i), \Phi(x\_j)\rangle$ is written exactly like the Gramian of a basis, which is precisely why the two get conflated.

### What features look like in infinite dimensions

In infinite-dimensional spaces, features come in three flavours, only the first of which behaves like a basis.

1. **A genuine basis (discrete).** In a separable Hilbert space indexed by integers (the model space is $\ell^2$) — e.g. $L^2$ of the circle with the Fourier system $\phi\_n(x) = e^{i n x}$, an orthogonal basis (orthonormal after the $1/\sqrt{2\pi}$ scaling) — the features form a Riesz or orthonormal basis. They act just like finite basis vectors; there are merely countably many.
2. **Overcomplete frames.** You can have a spanning set carrying redundancy, e.g. Morlet wavelets. A frame is not a basis — there is no unique way to expand a function in it — yet it still represents functions stably. In finite dimensions redundancy only gives you a linearly dependent set; in infinite dimensions a frame is a tool in its own right.
3. **Continuous "no-basis" systems.** When the features are indexed by a continuum they often cannot be a basis at all. The point-evaluation features $\Phi(x) = \delta(\cdot - x)$ have formal "Gramian" $\langle \delta\_x, \delta\_y\rangle = \delta(x - y)$, but the Dirac delta is a distribution — it does not even live in $L^2$. These features sit *outside* the space they describe; making them rigorous is exactly what **rigged Hilbert spaces** are for.

So in infinite dimensions a "feature" is better pictured as a probe or a continuous field than as a rigid building block.

#TODO: add visualizations (the three regimes: orthonormal basis vs. overcomplete frame vs. continuous deltas)

## The Gramian is more general than a basis — and the inner product is a Gramian

Two facts close the loop, and they point in opposite directions.

### Any set of vectors has a Gramian

Given *any* $v\_1, \dots, v\_k$ in an inner product space, $G\_{ij} = \langle v\_i, v\_j\rangle$ is a Gramian — no independence or spanning required.

* If the vectors are a basis, $G$ is positive-**definite** (invertible).
* If they are redundant or dependent, $G$ is positive-**semidefinite** (some eigenvalues vanish) — still a perfectly valid Gramian.

For instance, in $\mathbb R^2$ take $v\_1 = (1,0)$, $v\_2 = (0,1)$, and $v\_3 = (1,1) = v\_1 + v\_2$. These are not a basis, yet

$$G = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}$$

is a legitimate $3 \times 3$ Gramian. This is exactly the kernel-matrix situation in machine learning: the rows are indexed by data points, almost never by a basis — 10000 points sitting in a 10-dimensional space still give a $10000 \times 10000$ PSD Gramian.

### Conversely, an inner product *is* a Gramian

Run the logic backwards. On a real finite-dimensional space every inner product can be written as a bilinear form

$$\langle x, y\rangle_M = x^\top M y,$$

and the axioms of an inner product force $M$ to be symmetric and positive-definite. That matrix is exactly the Gramian of the chosen basis,

$$M_{ij} = \langle e_i, e_j\rangle.$$

* An **orthonormal** basis has $M = I$, recovering the standard dot product $x^{\top} y$.
* A **skewed** (non-orthogonal) basis gives an $M$ that records the warping of the axes — the metric tensor — and $x^{\top} M y$ corrects for it so lengths and angles come out right.

The two appearances of "Gramian" are therefore the same object seen from opposite ends: a *basis* produces a positive-definite Gramian that **defines** the metric of the space, while an arbitrary collection of *data vectors*, dropped into that fixed metric, produces a positive-semidefinite Gramian that merely **describes their layout** within it.

#TODO: add visualizations (orthonormal vs. skewed basis — the unit ball as a circle vs. an ellipse on square grid paper; a degenerate Gramian collapsing an axis)

## Where it lands: the taxonomy of bilinear forms

The final step is to name what we have built. A Gramian is a special bilinear form $x^{\top} A y$; relaxing the conditions on $A$ walks us through the whole family.

### When the generating vectors are dependent

If the $v\_i$ are linearly dependent, $G$ is positive-semidefinite but no longer definite, and the form $x^{\top} G x$ is **degenerate**: there is a nonzero $x$ with $x^{\top} G y = 0$ for all $y$. Equivalently it is a **semi-inner product** (or pseudo-inner product) — it satisfies every inner-product axiom except strict positivity, so it carries a "blind spot," a subspace of nonzero vectors that register zero length. The same object shows up under different names across fields:

* **Statistics** — a covariance matrix of perfectly correlated variables; the degenerate direction is a deterministic combination with exactly zero variance.
* **Differential geometry** — a rank-deficient metric tensor; the coordinate grid has collapsed along an axis, so the effective dimension is below the matrix size.

### When the matrix is arbitrary

Drop all constraints and $x^{\top} A y$ is a **general bilinear form**, classified by the structure of $A$.

* **Square $A$** (a form on $V \times V$):
  * **symmetric** ($A = A^{\top}$): the order does not matter, $x^{\top} A y = y^{\top} A x$;
  * **skew-symmetric** ($A = -A^{\top}$): the diagonal vanishes, $x^{\top} A x = 0$ for all $x$;
  * **asymmetric**: no internal symmetry, but always uniquely splittable into a symmetric and a skew part,

$$A = \tfrac{1}{2}(A + A^\top) + \tfrac{1}{2}(A - A^\top).$$

* **Rectangular $A$** (a pairing $V \times W \to \mathbb R$): a **dual / pairing form**. It no longer measures lengths or angles, since $x$ and $y$ live in different spaces; it measures how a vector in one space acts on a vector in the other.

And that rectangular case is the doorway back to where we began: its infinite-dimensional version is a **bounded linear operator between two different spaces** — the very move that turned the Gram matrix into the Gram operator, now applied to forms that pair two distinct worlds.

## Appendix

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gram matrix and Gram operator)</span></p>

For vectors $v\_1, \dots, v\_n$ in an inner product space, their **Gram matrix** is $G\_{ij} = \langle v\_i, v\_j\rangle$. Its continuous analogue is the **Gram (integral) operator** on $L^2(\Omega)$,

$$(Kf)(x) = \int_\Omega k(x,y)\, f(y)\, dy,$$

with symmetric kernel $k(x,y) = \langle \phi\_x, \phi\_y\rangle$. For a continuous positive-semidefinite kernel on a compact domain, $K$ is self-adjoint, positive, and compact.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reproducing kernel Hilbert space)</span></p>

A Hilbert space $\mathcal H$ of functions on a set $\mathcal X$ is a **reproducing kernel Hilbert space** if every point evaluation $f \mapsto f(x)$ is a bounded linear functional. By the Riesz representation theorem there is then a unique $k(\cdot, x) \in \mathcal H$ with the **reproducing property**

$$f(x) = \langle f, k(\cdot, x)\rangle \quad \text{for all } f \in \mathcal H,$$

and consequently $k(x,y) = \langle k(\cdot, x), k(\cdot, y)\rangle$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Moore–Aronszajn)</span></p>

A symmetric kernel $k\colon \mathcal X \times \mathcal X \to \mathbb R$ is positive semidefinite if and only if there exist a Hilbert space $\mathcal H$ and a feature map $\Phi\colon \mathcal X \to \mathcal H$ with $k(x,y) = \langle \Phi(x), \Phi(y)\rangle$. Moreover, to every positive-semidefinite kernel there corresponds a *unique* RKHS in which $k$ is the reproducing kernel.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Mercer)</span></p>

Let $k$ be a continuous positive-semidefinite kernel on a compact domain. Then the associated integral operator has an orthonormal system of eigenfunctions $\varphi\_n$ with eigenvalues $\lambda\_n \ge 0$, and

$$k(x,y) = \sum_{n} \lambda_n\, \varphi_n(x)\, \varphi_n(y),$$

converging absolutely and uniformly. The feature map is then explicit: $\Phi(x) = \big(\sqrt{\lambda\_1}\,\varphi\_1(x),\ \sqrt{\lambda\_2}\,\varphi\_2(x),\ \dots\big)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bilinear forms and their Gramians)</span></p>

For a matrix $A$, the map $(x, y) \mapsto x^{\top} A y$ is a **bilinear form**. With $A = G$ a Gramian:

* **basis vectors** $\Rightarrow$ $G$ positive-definite, $\det G > 0$, full rank — a genuine inner product (a non-degenerate form);
* **dependent vectors** $\Rightarrow$ $G$ positive-semidefinite, $\det G = 0$, rank-deficient — a **semi-inner product** (a degenerate form).

For a general $A$: **symmetric** ($A = A^{\top}$), **skew-symmetric** ($A = -A^{\top}$), or, when $A$ is rectangular, a **pairing** $V \times W \to \mathbb R$ between two different spaces.

</div>

### References

* C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006 — Chapter 4 (Covariance Functions) is the most readable route into Moore–Aronszajn, Mercer, and feature spaces. Free online.
* B. Schölkopf and A. J. Smola, *Learning with Kernels*, MIT Press, 2002 — Chapter 2 (A Primer on Kernel Methods) covers feature maps and RKHS for the machine-learning reader.
* A. Berlinet and C. Thomas-Agnan, *Reproducing Kernel Hilbert Spaces in Probability and Statistics*, Springer, 2004 — Chapter 1 gives the rigorous proofs without the ML framing.
* N. Aronszajn, "Theory of Reproducing Kernels," *Transactions of the American Mathematical Society* 68 (1950) — the dense but historic primary source.
