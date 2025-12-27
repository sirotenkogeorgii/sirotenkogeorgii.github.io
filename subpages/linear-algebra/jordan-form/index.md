---
title: Jordan Normal Form
layout: default
noindex: true
---

# Jordan Normal Form

## Definition

But what is the simplest form of a matrix that can be arrived at using similarity? It is not a **diagonal matrix**, because we already know that not all matrices are diagonalizable. However, any matrix can be transformed into a relatively simple form, called **Jordan normal form**, by a similarity transformation. Jordan normal form (JNF) is useful because it’s the **“closest thing to diagonalization”** you can get for every complex square matrix, and it turns a lot of matrix questions into simple block-by-block calculations.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Jordan block)</span></p>

Let $\lambda \in \mathbb{C}$ and $k \in \mathbb{N}$. The **Jordan block** $J_k(\lambda)$ is the $k\times k$ matrix

$$
J_k(\lambda)=
\begin{pmatrix}
\lambda & 1 & 0 & \cdots & 0\\
0 & \lambda & 1 & \ddots & \vdots\\
\vdots & \ddots & \ddots & \ddots & 0\\
\vdots & & \ddots & \lambda & 1\\
0 & \cdots & \cdots & 0 & \lambda
\end{pmatrix}.
$$

</div>

The Jordan block has eigenvalue $\lambda$ with algebraic multiplicity $k$, and it has only one eigenvector,

$$e_1=(1,0,\dots,0)^\top,$$

because $\operatorname{rank}\big(J_k(\lambda)-\lambda I_k\big)=k-1$.

* **Number of Jordan blocks for (\lambda)** = $\dim \ker(A-\lambda I)$ (geometric multiplicity).
* **Sizes of blocks** encode the “chains” of generalized eigenvectors and how big the nilpotent part is.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Jordan normal form)</span></p>

A matrix $J \in \mathbb{C}^{n\times n}$ is in **Jordan normal form** if it is block diagonal, with Jordan blocks on the diagonal:

$$
J=
\begin{pmatrix}
J_{k_1}(\lambda_1) & 0 & \cdots & 0\\
0 & \ddots & \ddots & \vdots\\
\vdots & \ddots & \ddots & 0\\
0 & \cdots & 0 & J_{k_m}(\lambda_m)
\end{pmatrix},
$$

where the diagonal blocks are $J_{k_1}(\lambda_1),\dots,J_{k_m}(\lambda_m)$.

</div>

Values $\lambda_i$ and $k_i$ are not necessarily different. In the same way the same Jordan block could appear several times.


<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(About JNF)</span></p>

Every square matrix $A \in \mathbb{C}^{n\times n}$ is similar to some matrix in JNF. This matrix is unique up to order of Jordan blocks.

</div>

**Jordan normal form is unstable** in the sense that even a small change in the original matrix can cause a sudden change in the Jordan form – **it is not a continuous function with respect to the elements of matrix $A$**. This is also the reason why the `Maple` computing system refused to include the calculation of Jordan form in its library. In contrast, **the eigenvalues ​​themselves are stable**, because they represent continuous functions with respect to the elements of matrix $A$.

## It makes matrix functions easy: powers, exponentials, polynomials

If $A = PJP^{-1}$, then $f(A)=Pf(J)P^{-1}$, and $f(J)$ is computed blockwise.

For a Jordan block $J=\lambda I+N$ with $N$ nilpotent $(N^k=0)$:

* **Powers**:
  
  $$(\lambda I+N)^m=\sum_{r=0}^{k-1}\binom{m}{r}\lambda^{m-r}N^r.$$
  
* **Exponential**:
  
  $$e^{\lambda I+N}=e^\lambda \left(I+N+\frac{N^2}{2!}+\cdots+\frac{N^{k-1}}{(k-1)!}\right).$$

This is hugely useful in:

* solving linear ODEs $x'(t)=Ax(t)$ via $e^{tA}$,
* analyzing discrete systems $x_{n+1}=Ax_n$ via $A^n$,
* computing any analytic $f(A)$ (log, sqrt, etc., when defined).

## It gives sharp stability / asymptotic behavior of linear dynamical systems

Eigenvalues alone tell you growth rates, but Jordan blocks tell you **extra polynomial factors**.

If a block for eigenvalue $\lambda$ has size $k$, then trajectories can involve terms like

$$
t^{k-1}e^{\lambda t} \quad \text{(continuous time)},\qquad
n^{k-1}\lambda^n \quad \text{(discrete time)}.
$$

So even if $\lvert\lambda\rvert=1$ (or $\Re(\lambda)=0$), nontrivial Jordan blocks can create polynomial growth.

## Examples

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Powers of matrix)</span></p>

We already mentioned the use of diagonalization for computing a power of a matrix. Using the Jordan normal form, we can generalize the statement to an arbitrary matrix $A \in \mathbb{C}^{n\times n}$. If

$$A = SJS^{-1},$$

then

$$
A^k = SJ^kS^{-1}
= S\begin{pmatrix}
J_{k_1}(\lambda_1)^k & 0 & \cdots & 0\\
0 & \ddots & \ddots & \vdots\\
\vdots & \ddots & \ddots & 0\\
0 & \cdots & 0 & J_{k_m}(\lambda_m)^k
\end{pmatrix}S^{-1}.
$$

Here one needs to think a bit about how the powers of Jordan blocks $J_{k_i}(\lambda_i)^k$, $i=1,\dots,m$, behave. Asymptotically we obtain the same conclusion as for diagonalizable matrices:

$$
\lim_{k\to\infty} A^k =
\begin{cases}
0, & \text{if } \rho(A) < 1,\\
\text{diverges}, & \text{if } \rho(A) > 1,\\
\text{converges / diverges}, & \text{if } \rho(A) = 1,
\end{cases}
$$

where $\rho(A)$ denotes the spectral radius of $A$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Matrix function)</span></p>

Let us ask: how can we define a matrix function such as $\cos(A)$, $e^{A}$, etc.?
For a real function $f:\mathbb{R}\to\mathbb{R}$ and a matrix $A\in\mathbb{R}^{n\times n}$, one possible definition is to apply $f$ to each entry of $A$ separately:

$$
f(A)=
\begin{pmatrix}
f(a_{11}) & \cdots & f(a_{1n})\\
\vdots & \ddots & \vdots\\
f(a_{n1}) & \cdots & f(a_{nn})
\end{pmatrix},
\qquad \text{(10.4)}
$$

which is possible, but it will not have many nice properties. Let us try a different approach.

Assume that $f$ can be expressed as a power series

$$f(x)=\sum_{i=0}^{\infty} a_i x^i,$$

as is the case for real analytic functions such as $\sin(x)$, $\exp(x)$, etc. Then it is natural to define

$$f(A)=\sum_{i=0}^{\infty} a_i A^i.$$

We already know how to compute powers of matrices, so if $A=SJS^{-1}$, then

$$
f(A)=\sum_{i=0}^{\infty} a_i S J^i S^{-1}
= S\left(\sum_{i=0}^{\infty} a_i J^i\right)S^{-1}
= S f(J) S^{-1}.
$$

It is then easy to see that

$$
f(J):=
\begin{pmatrix}
f(J_{k_1}(\lambda_1)) & 0 & \cdots & 0\\
0 & \ddots & \ddots & \vdots\\
\vdots & \ddots & \ddots & 0\\
0 & \cdots & 0 & f(J_{k_m}(\lambda_m))
\end{pmatrix}.
$$

So it remains to define $f$ on Jordan blocks $J_{k_i}(\lambda_i)$. For $k_i=1$ this is trivial (a $(1\times 1)$ matrix). For $k_i>1$ the formula is more complicated:

$$
f!\left(J_{k_i}(\lambda_i)\right):=
\begin{pmatrix}
f(\lambda_i) & f'(\lambda_i) & \cdots & \dfrac{f^{(k_i-1)}(\lambda_i)}{(k_i-1)!}\\
0 & f(\lambda_i) & \ddots & \vdots\\
\vdots & \ddots & \ddots & f'(\lambda_i)\\
0 & \cdots & 0 & f(\lambda_i)
\end{pmatrix}.
$$

For example, for $f(x)=x^2$ we get the matrix extension $f(A)=A^2$, i.e. ordinary matrix squaring. In contrast, definition (10.4) would suggest squaring each entry separately, which is not what we want.

Another example of a matrix function is the matrix exponential

$$e^A=\sum_{i=0}^{\infty}\frac{1}{i!}A^i.$$

One of its many uses is to describe rotations in $\mathbb{R}^3$. The matrix $e^R$, where

$$
R=\alpha
\begin{pmatrix}
0 & -z & y\\
z & 0 & -x\\
-y & x & 0
\end{pmatrix},
$$

describes a rotation about the axis in the direction $(x,y,z)^\top$ by the angle $\alpha$, according to the right-hand rule.


</div>