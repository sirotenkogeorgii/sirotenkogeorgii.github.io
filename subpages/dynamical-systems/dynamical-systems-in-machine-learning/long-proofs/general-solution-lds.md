---
title: General Solution Derivation For LDS
layout: default
noindex: true
---

# General Solution Derivation For LDS 

## $A$ is Diagonalizable

Deriving the solution for a system of linear differential equations is a fundamental concept in dynamical systems. The key to solving $\dot{\mathbf{x}} = A\mathbf{x}$ when $A$ is diagonalizable is to use a coordinate transformation that turns a complex, interconnected system into a set of simple, independent equations.

Here is the step-by-step derivation of the general solution.

### **1. Define the Diagonalization**

Since the matrix $A$ is diagonalizable, we can decompose it into:

$$A = PDP^{-1}$$

Where:
* $P$ is an invertible matrix whose columns are the eigenvectors of $A$ ($\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n$).
* $D$ is a diagonal matrix containing the corresponding eigenvalues of $A$ ($\lambda_1, \lambda_2, \dots, \lambda_n$) along its main diagonal.
* $P^{-1}$ is the inverse of the eigenvector matrix.

### **2. Substitute and Decouple**

Substitute the diagonalized form of $A$ back into the original differential equation:

$$\dot{\mathbf{x}} = (PDP^{-1})\mathbf{x}$$

To decouple this system, we introduce a change of variables. Let's define a new state vector $\mathbf{y}$ such that:

$$\mathbf{x} = P\mathbf{y}$$

Because $P$ is a matrix of constants, taking the time derivative of both sides gives us:

$$\dot{\mathbf{x}} = P\dot{\mathbf{y}}$$

Now, substitute $P\mathbf{y}$ for $\mathbf{x}$ and $P\dot{\mathbf{y}}$ for $\dot{\mathbf{x}}$ in our substituted differential equation:

$$P\dot{\mathbf{y}} = (PDP^{-1})(P\mathbf{y})$$

### **3. Simplify the New System**

Notice that $P^{-1}P = I$ (the identity matrix). The equation simplifies to:

$$P\dot{\mathbf{y}} = PD\mathbf{y}$$

Multiply both sides from the left by $P^{-1}$:

$$\dot{\mathbf{y}} = D\mathbf{y}$$

This is the crucial step. Because $D$ is a diagonal matrix, this new system is completely decoupled. It simply represents $n$ independent, first-order linear differential equations of the form:

$$\dot{y}_i = \lambda_i y_i$$

### **4. Solve the Decoupled System**

The solution to the scalar equation $\dot{y}_i = \lambda_i y_i$ is standard:

$$y_i(t) = c_i e^{\lambda_i t}$$

where $c_i$ is an arbitrary constant determined by the initial conditions.

In matrix form, the solution for the vector $\mathbf{y}(t)$ is:

$$\mathbf{y}(t) = e^{Dt}\mathbf{c}$$

Where $e^{Dt}$ is a diagonal matrix with $e^{\lambda_i t}$ on the diagonal, and $\mathbf{c}$ is a column vector of arbitrary constants $[c_1, c_2, \dots, c_n]^\top$.

### **5. Transform Back to Original Coordinates**

Finally, we convert our solution back to the original variables using our relationship $\mathbf{x} = P\mathbf{y}$:

$$\mathbf{x}(t) = P e^{Dt} \mathbf{c}$$

By multiplying out the matrix $P$ (whose columns are the eigenvectors $\mathbf{v}_i$) and the vector $\mathbf{y}(t)$, we can express the general solution as a linear combination of the eigenvectors:

$$\mathbf{x}(t) = c_1 e^{\lambda_1 t}\mathbf{v}_1 + c_2 e^{\lambda_2 t}\mathbf{v}_2 + \dots + c_n e^{\lambda_n t}\mathbf{v}_n$$

## $A$ is Defective

For a non-diagonalizable matrix $A$, the idea is the same, except we use **Jordan form** instead of a diagonal form.

There exists an invertible matrix $P$ such that

$$A = PJP^{-1}$$

where $J$ is the Jordan normal form of $A$. Then for

$$\dot x = Ax$$

set again

$$x = Py$$

Then

$$\dot x = P\dot y = APy = PJP^{-1}Py = PJy$$

so

$$\dot y = Jy$$

Thus the problem reduces to solving

$$\dot y = Jy$$

with $J$ block diagonal, made of Jordan blocks.

### 1. A single Jordan block

Suppose first that $J$ is one Jordan block of size $m$ with eigenvalue $\lambda$:

$$
J =
\begin{pmatrix}
\lambda & 1 & 0 & \cdots & 0\\
0 & \lambda & 1 & \cdots & 0\\
\vdots & & \ddots & \ddots & \vdots\\
0 & \cdots & 0 & \lambda & 1\\
0 & \cdots & \cdots & 0 & \lambda
\end{pmatrix}
$$

Write this as

$$J = \lambda I + N$$

where $N$ is nilpotent:

$$
N =
\begin{pmatrix}
0 & 1 & 0 & \cdots & 0\\
0 & 0 & 1 & \cdots & 0\\
\vdots & & \ddots & \ddots & \vdots\\
0 & \cdots & 0 & 0 & 1\\
0 & \cdots & \cdots & 0 & 0
\end{pmatrix},
\qquad N^m=0.
$$

Then

$$e^{Jt}=e^{(\lambda I+N)t}=e^{\lambda t}e^{Nt}$$

Since $N$ is nilpotent,

$$e^{Nt}=I+tN+\frac{t^2}{2!}N^2+\cdots+\frac{t^{m-1}}{(m-1)!}N^{m-1}$$

So for one Jordan block,

$$y(t)=e^{Jt}c = e^{\lambda t}\left(I+tN+\frac{t^2}{2!}N^2+\cdots+\frac{t^{m-1}}{(m-1)!}N^{m-1}\right)c$$

That is why, in the non-diagonalizable case, solutions contain terms like

$$e^{\lambda t},\quad t e^{\lambda t},\quad t^2 e^{\lambda t},\dots$$

instead of only pure exponentials.

### 2. General case

If $J$ has several Jordan blocks, solve each block separately. Altogether,

$$y(t)=e^{Jt}c$$

hence

$$x(t)=Pe^{Jt}c$$

Since $A=PJP^{-1}$, this is also

$$x(t)=e^{At}x(0), \qquad e^{At}=Pe^{Jt}P^{-1}$$

### 3. Form of the general solution

So the general solution for a non-diagonalizable $A$ is

$$x(t)=Pe^{Jt}c$$

where each Jordan block for eigenvalue $\lambda$ contributes terms of the form

$$e^{\lambda t}(v_0 + t v_1 + t^2 v_2 + \cdots + t^{k} v_k)$$

Equivalently, along a Jordan chain, solutions look like

$$
e^{\lambda t}v_1,\qquad
e^{\lambda t}(t v_1 + v_2),\qquad
e^{\lambda t}\left(\frac{t^2}{2}v_1 + t v_2 + v_3\right),
$$

and so on.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($2\times 2$ Jordan block)</span></p>

If

$$
J=
\begin{pmatrix}
\lambda & 1\\
0 & \lambda
\end{pmatrix},
$$

then

$$
N=
\begin{pmatrix}
0 & 1\\
0 & 0
\end{pmatrix},
\qquad N^2=0,
$$

so

$$
e^{Jt}=e^{\lambda t}(I+tN) = e^{\lambda t}
\begin{pmatrix}
1 & t\\
0 & 1
\end{pmatrix}.
$$

Thus

$$
y(t)=
e^{\lambda t}
\begin{pmatrix}
1 & t\\
0 & 1
\end{pmatrix}
\begin{pmatrix}
c_1\\
c_2
\end{pmatrix}
=
e^{\lambda t}
\begin{pmatrix}
c_1+t c_2\\
c_2
\end{pmatrix}.
$$

Then

$$x(t)=Py(t)$$

So compared with the diagonalizable case, the new feature is:

$$\boxed{\text{non-diagonalizable }A ;\Rightarrow; \text{polynomials in } t \text{ multiplying } e^{\lambda t}.}$$

I can also show how this comes directly from generalized eigenvectors, which is often the most intuitive derivation.

</div>
