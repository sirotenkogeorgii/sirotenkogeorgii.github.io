---
layout: default
title: Linear Algebra III
date: 2025-03-10
excerpt: Eigenvalues, singular value decomposition, pseudoinverse matrices, and matrix norms.
tags:
  - linear-algebra
  - mathematics
---

# Linear Algebra III

## Eigenvalues

### Eigenvalues of General Matrices

The Schur decomposition is a key result in eigenvalue theory. Like the Jordan normal form, it exists for every square matrix, but it leverages unitary matrices, which is where its power lies.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Schur Decomposition)</span></p>

For every matrix $A \in \mathbb{C}^{n \times n}$ there exists a unitary $U \in \mathbb{C}^{n \times n}$ and an upper triangular $T \in \mathbb{C}^{n \times n}$ such that $A = UTU^*$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By induction on $n$ (analogous to the proof of the spectral decomposition of a symmetric matrix). For $n=1$ the statement is trivial, so assume the inductive step.

Let $\lambda \in \mathbb{C}$ be an eigenvalue of $A$ and $x \in \mathbb{C}^n$, $\lVert x \rVert_2 = 1$, a corresponding eigenvector. Extend $x$ to a unitary matrix $\tilde{V} = (x \mid V)$. Then

$$
\tilde{V}^* A \tilde{V} = \begin{pmatrix} x^* \\ V^* \end{pmatrix} (Ax \mid AV) = \begin{pmatrix} x^* \\ V^* \end{pmatrix} (\lambda x \mid AV) = \begin{pmatrix} \lambda & x^* AV \\ 0 & V^* AV \end{pmatrix}.
$$

By the inductive hypothesis there exists a unitary $W \in \mathbb{C}^{(n-1) \times (n-1)}$ such that $\tilde{T} \coloneqq W^* V^* AVW$ is upper triangular. Setting

$$
U = \tilde{V} \begin{pmatrix} 1 & 0^T \\ 0 & W \end{pmatrix}, \quad T = \begin{pmatrix} \lambda & x^* AVW \\ 0 & \tilde{T} \end{pmatrix}
$$

gives $A = UTU^*$.

</details>
</div>

Since $T$ is similar to $A$, the eigenvalues of $A$ appear on the diagonal of $T$. For real matrices $A$, $U$ and $T$ may be complex, but there are real variants:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Schur Decomposition, Real Variant I)</span></p>

For every matrix $A \in \mathbb{R}^{n \times n}$ there exists an orthogonal $Q \in \mathbb{R}^{n \times n}$ and a block upper triangular $T \in \mathbb{R}^{n \times n}$ such that $A = QTQ^T$. The blocks of $T$ are of size $1$ or $2$, where blocks of size $2$ contain complex conjugate eigenvalues.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Schur Decomposition, Real Variant II)</span></p>

For every matrix $A \in \mathbb{R}^{n \times n}$ there exists a regular $S \in \mathbb{R}^{n \times n}$ and a block upper triangular $T \in \mathbb{R}^{n \times n}$ such that $A = STS^{-1}$. The blocks of $T$ are of size $1$ or $2$, where blocks of size $2$ have the form $\begin{pmatrix} a & b \\ -b & a \end{pmatrix}$ and contain complex conjugate eigenvalues $a \pm ib$.

</div>

One consequence of the Schur decomposition is the observation that any matrix that is arbitrarily close to diagonal is diagonalizable (see Theorem 5.27 in later sections).

Matrices whose Schur decomposition has a diagonal $T$ are called **normal**. A typical example of a normal matrix is a real symmetric matrix, since its spectral decomposition $QAQ^T$ ($Q$ orthogonal and $A$ diagonal) is precisely its Schur decomposition. Other symmetric matrices do not have $QAQ^T$ normal since the result is not symmetric. Nevertheless, thanks to complex Schur decomposition, both real and complex normal matrices (including antisymmetric $A = -A^T$ and orthogonal matrices) can arise. Normal matrices share many nice properties with symmetric matrices, and some results from Section 2.2 apply analogously to them.

### Eigenvalues of Symmetric Matrices

The Rayleigh--Ritz formula gives an elegant expression for the largest and smallest eigenvalue of a symmetric matrix --- the largest and smallest values of the quadratic form $f(x) = x^T A x$ on the unit Euclidean sphere.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Rayleigh--Ritz)</span></p>

Let $\lambda_1 \geq \ldots \geq \lambda_n$ be the eigenvalues of a symmetric matrix $A \in \mathbb{R}^{n \times n}$. Then

$$
\lambda_1 = \max_{\lVert x \rVert_2 = 1} x^T A x, \quad \lambda_n = \min_{\lVert x \rVert_2 = 1} x^T A x.
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Only for $\lambda_1$; the second part is analogous.

**"$\leq$":** Let $x_1$ be the eigenvector for $\lambda_1$ normalized to $\lVert x_1 \rVert_2 = 1$. Then $Ax_1 = \lambda_1 x_1$. Multiplying by $x_1^T$ from the left gives $\lambda_1 = \lambda_1 x_1^T x_1 = x_1^T A x_1 \leq \max_{\lVert x \rVert_2 = 1} x^T A x$.

**"$\geq$":** Let $x \in \mathbb{R}^n$ be arbitrary with $\lVert x \rVert_2 = 1$. Set $y \coloneqq Q^T x$, so $\lVert y \rVert_2 = 1$. Using the spectral decomposition $A = Q \Lambda Q^T$:

$$
x^T A x = x^T Q \Lambda Q^T x = y^T \Lambda y = \sum_{i=1}^n \lambda_i y_i^2 \leq \sum_{i=1}^n \lambda_1 y_i^2 = \lambda_1 \lVert y \rVert_2^2 = \lambda_1.
$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Symmetric matrices not only have specific properties but also generally behave more "reasonably" than general matrices. Consider, for example, the mapping $x \mapsto Ax$ where

$$
A = \begin{pmatrix} 1 & K \\ 0 & 1 \end{pmatrix},
$$

with $K$ arbitrarily large. The matrix $A$ has only one eigenvalue, equal to $1$. Yet the vector $e_2 = (0,1)^T$ can be mapped to an arbitrarily large vector, since $Ae_2 = (K,1)^T$. Thus eigenvalues do not bound the size of images.

For symmetric matrices it is different. Let $A \in \mathbb{R}^{n \times n}$ be symmetric and $x \in \mathbb{R}^n$, $\lVert x \rVert_2 = 1$. By the Rayleigh--Ritz formula:

$$
\lVert Ax \rVert_2 = \sqrt{(Ax)^T(Ax)} = \sqrt{x^T A^2 x} \leq \sqrt{\lambda_1(A^2)} = \sqrt{\max\lbrace|\lambda_1(A)|, |\lambda_n(A)|\rbrace^2} \leq \rho(A).
$$

Thus the image size is bounded, at most $\rho(A)$-times larger.

</div>

The following theorem is a direct consequence of the Rayleigh--Ritz formula. It gives a formula for any eigenvalue, not just the smallest and largest.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Courant--Fischer)</span></p>

Let $\lambda_1 \geq \ldots \geq \lambda_n$ be the eigenvalues of a symmetric matrix $A \in \mathbb{R}^{n \times n}$. Then

$$
\lambda_k = \max_{V \in \mathbb{R}^n:\, \dim V = k} \min_{x \in V:\, \lVert x \rVert_2 = 1} x^T A x = \min_{V \in \mathbb{R}^n:\, \dim V = n-k+1} \max_{x \in V:\, \lVert x \rVert_2 = 1} x^T A x.
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Equality (2.1).** Let $q_1, \ldots, q_n$ be an orthonormal eigenbasis with eigenvalues $\lambda_1, \ldots, \lambda_n$. Set $Q \coloneqq (q_1 \mid \cdots \mid q_n)$.

**"$\leq$":** Define $U \coloneqq \operatorname{span}\lbrace q_1, \ldots, q_k \rbrace$. Every $x \in U$ with $\lVert x \rVert_2 = 1$ can be written as $x = \sum_{i=1}^k \alpha_i q_i$ with $\sum_{i=1}^k \alpha_i^2 = 1$. Then

$$
x^T A x = \sum_{i=1}^k \alpha_i^2 \lambda_i \geq \sum_{i=1}^k \alpha_i^2 \lambda_k = \lambda_k.
$$

**"$\geq$":** Let $V$ be an arbitrary subspace of $\mathbb{R}^n$ of dimension $k$. Then $V \cap \operatorname{span}\lbrace q_k, \ldots, q_n \rbrace$ has dimension at least $1$, so there exists a unit vector $v$ expressible as $x = \sum_{i=k}^n \alpha_i q_i$ with $\sum_{i=k}^n \alpha_i^2 = 1$. Then

$$
x^T A x = \sum_{i=k}^n \alpha_i^2 \lambda_i \leq \sum_{i=k}^n \alpha_i^2 \lambda_k = \lambda_k.
$$

**Equality (2.2).** Using $\lambda_k(A) = -\lambda_{n-k+1}(-A)$:

$$
\lambda_k(A) = -\max_{V:\,\dim V = n-k+1} \min_{x \in V:\, \lVert x \rVert_2=1} (-x^T A x) = \min_{V:\,\dim V = n-k+1} \max_{x \in V:\, \lVert x \rVert_2=1} x^T A x.
$$

</details>
</div>

One consequence is the Cauchy interlacing property, connecting eigenvalues of a matrix and the matrix obtained by removing one row and column.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Cauchy Interlacing Property)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ be symmetric and let $B$ be the matrix obtained from $A$ by deleting the $i$-th row and column. Denote $\lambda_1 \geq \ldots \geq \lambda_n$ the eigenvalues of $A$ and $\mu_1 \geq \ldots \geq \mu_{n-1}$ the eigenvalues of $B$. Then

$$
\lambda_1 \geq \mu_1 \geq \lambda_2 \geq \ldots \geq \lambda_{n-1} \geq \mu_{n-1} \geq \lambda_n.
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Without loss of generality, assume $i = n$. Set $W \coloneqq \operatorname{span}\lbrace e_1, \ldots, e_{n-1} \rbrace$.

**"$\lambda_k \geq \mu_k$":** By Courant--Fischer,

$$
\lambda_k = \max_{V \in \mathbb{R}^n:\, \dim V = k} \min_{x \in V:\, \lVert x \rVert_2 = 1} x^T A x \geq \max_{V \in W:\, \dim V = k} \min_{x \in V:\, \lVert x \rVert_2 = 1} x^T A x = \mu_k.
$$

**"$\mu_k \geq \lambda_{k+1}$":** Let $V_{k+1}$ be the subspace achieving the maximum in the Courant--Fischer formula for $\lambda_{k+1}$. Then $V_{k+1} \cap W$ has dimension at least $k$, so

$$
\lambda_{k+1} = \min_{x \in V_{k+1}:\, \lVert x \rVert_2 = 1} x^T A x \leq \max_{V \in W:\, \dim V = k} \min_{x \in V_{k+1} \cap W:\, \lVert x \rVert_2 = 1} x^T A x \leq \mu_k.
$$

</details>
</div>

---

## SVD Decomposition

### Construction and Connection to Eigenvalues

The SVD (Singular Value Decomposition) is one of the most important numerical techniques. It was discovered in 1873 independently by several authors, including Eugenio Beltrami, Camille Jordan, James Sylvester, Erhard Schmidt, and Hermann Weyl. Its practical implementations were developed by Gene Golub and William Kahan.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(SVD Decomposition)</span></p>

Let $A \in \mathbb{R}^{m \times n}$, $q \coloneqq \min\lbrace m,n \rbrace$. Then there exists a diagonal matrix $\Sigma \in \mathbb{R}^{m \times n}$ with entries $\sigma_{11} \geq \ldots \geq \sigma_{qq} \geq 0$ and orthogonal matrices $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$ such that $A = U \Sigma V^T$.

</div>

The diagonal entries $\sigma_1, \ldots, \sigma_q$ of $\Sigma$ are called the **singular values** of $A$, denoted $\sigma_1, \ldots, \sigma_q$. The number of positive singular values equals $r = \operatorname{rank}(A)$, where $\sigma_r > 0$ and $\sigma_{r+1} = 0$.

The **reduced form** of the SVD: Decompose $U = (U_1 \mid U_2)$, $V = (V_1 \mid V_2)$, and $S \coloneqq \operatorname{diag}(\sigma_1, \ldots, \sigma_r)$. Then

$$
A = U \Sigma V^T = \begin{pmatrix} U_1 & U_2 \end{pmatrix} \begin{pmatrix} S & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} V_1^T \\ V_2^T \end{pmatrix} = U_1 S V_1^T.
$$

The reduced SVD uses only a portion of the full SVD but contains the essential information, since the full SVD can be reconstructed by completing $U$, $V$ to orthogonal matrices.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Relationship Between Singular Values and Eigenvalues I)</span></p>

Let $A \in \mathbb{R}^{m \times n}$, $r = \operatorname{rank}(A)$, and let $A^T A$ have eigenvalues $\lambda_1 \geq \ldots \geq \lambda_n$. Then the positive singular values of $A$ are $\sigma_i = \sqrt{\lambda_i}$, $i = 1, \ldots, r$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $A = U \Sigma V^T$ be the SVD of $A$. Then

$$
A^T A = V \Sigma^T U^T U \Sigma V^T = V \Sigma^T \Sigma V^T = V \operatorname{diag}(\sigma_1^2, \ldots, \sigma_q^2, 0, \ldots, 0) V^T,
$$

which is the spectral decomposition of the positive semidefinite matrix $A^T A$. Thus $\lambda_i = \sigma_i^2$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

Let $Q \in \mathbb{R}^{n \times n}$ be orthogonal. Since $Q^T Q = I_n$, all eigenvalues are ones. Thus the singular values of an orthogonal matrix are also all ones.

Conversely, if $A \in \mathbb{R}^{n \times n}$ has all singular values equal to one, then $\Sigma = I_n$ and $A = U \Sigma V^T = U I_n V^T = UV^T$, which is a product of orthogonal matrices and hence orthogonal.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

Let $A \in \mathbb{R}^{n \times n}$ be symmetric. Then $A^T A = A^2$ has eigenvalues that are squares of eigenvalues $\lambda_1, \ldots, \lambda_n$ of $A$. Hence the singular values of $A$ are $|\lambda_1|, \ldots, |\lambda_n|$ in descending order.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of the SVD Theorem</summary>

Let $A^T A = V \Lambda V^T$ be the spectral decomposition of $A^T A$, with columns $v_1, \ldots, v_m$ of $V$. Set $r = \operatorname{rank}(A)$. Then

$$
v_j^T A^T A v_i = v_j^T \lambda_i v_i = \begin{cases} \lambda_i & \text{if } i = j, \\ 0 & \text{if } i \neq j. \end{cases}
$$

Define $u_i \coloneqq \frac{1}{\sqrt{\lambda_i}} A v_i$ for $i = 1, \ldots, r$. Then $v_i \perp v_j$ and

$$
\lVert u_i \rVert_2^2 = \langle u_i, u_i \rangle = \left\langle \frac{1}{\sqrt{\lambda_i}} A v_i, \frac{1}{\sqrt{\lambda_i}} A v_i \right\rangle = \frac{1}{\lambda_i} v_i^T A^T A v_i = 1.
$$

Thus $u_1, \ldots, u_r$ form an orthonormal system. Extend to an orthonormal basis $u_1, \ldots, u_r, \ldots, u_m$ of $\mathbb{R}^m$. Since $u_i \in \mathcal{S}(A)$ and $\operatorname{rank}(A) = r$, we have $u_i^T A = 0$ for $i = r+1, \ldots, n$. The equation can be rewritten as $U^T A V = \Sigma$, giving $A = U \Sigma V^T$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The proof reveals that $V$ is the orthogonal matrix from the spectral decomposition of $A^T A$. Similarly, $U$ is the orthogonal matrix from the spectral decomposition of $AA^T$:

$$
AA^T = U \Sigma V^T V \Sigma^T U^T = U \Sigma \Sigma^T U^T = U \operatorname{diag}(\sigma_1^2, \ldots, \sigma_q^2, 0, \ldots, 0) U^T.
$$

Unfortunately, the spectral decompositions of $A^T A$ and $AA^T$ alone cannot be used to uniquely construct the SVD, since we can only use one and must compute the other slightly differently.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

Let $A = (1\ 1\ 1\ 1)$. The spectral decomposition of $A^T A$ is

$$
A^T A = \begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \end{pmatrix} = \begin{pmatrix} 1/2 & \cdots \\ 1/2 & \cdots \\ 1/2 & \cdots \\ 1/2 & \cdots \end{pmatrix} \begin{pmatrix} 4 & & \\ & 0 & \\ & & 0 \\ & & & 0 \end{pmatrix} \begin{pmatrix} 1/2 & 1/2 & 1/2 & 1/2 \\ \vdots & \vdots & \vdots & \vdots \end{pmatrix}.
$$

Now $u_1 = \frac{1}{\sqrt{\lambda_i}} A v_i = 1$. Thus the full and reduced SVD is

$$
A = (1)(2\ 0\ 0\ 0) \begin{pmatrix} 1/2 & 1/2 & 1/2 & 1/2 \\ \vdots & \vdots & \vdots & \vdots \end{pmatrix} = (1)(2)(1/2\ \ 1/2\ \ 1/2\ \ 1/2).
$$

</div>

There is another relationship between singular values and eigenvalues:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Relationship Between Singular Values and Eigenvalues II)</span></p>

A matrix $A \in \mathbb{R}^{m \times n}$ has singular values $\sigma_1, \ldots, \sigma_r$ exactly when the nonzero eigenvalues of the matrix

$$
\begin{pmatrix} 0 & A \\ A^T & 0 \end{pmatrix}
$$

are $\sigma_1, \ldots, \sigma_r, -\sigma_r, \ldots, -\sigma_1$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**"$\Leftarrow$":** Let $\lambda$ be an eigenvalue and $(y^T, x^T)^T$ the corresponding eigenvector of $\begin{pmatrix} 0 & A \\ A^T & 0 \end{pmatrix}$, i.e.,

$$
\begin{pmatrix} 0 & A \\ A^T & 0 \end{pmatrix} \begin{pmatrix} y \\ x \end{pmatrix} = \lambda \begin{pmatrix} y \\ x \end{pmatrix}.
$$

This gives two equations: $A^T y = \lambda x$ and $Ax = \lambda y$. Thus $A^T A x = A^T(\lambda y) = \lambda^2 x$. By Theorem 3.2, $\sigma \coloneqq \sqrt{\lambda^2} = |\lambda|$ is a singular value of $A$.

**"$\Rightarrow$":** Let $\sigma$ be a singular value of $A$. By Theorem 3.2, $A^T A x = \sigma^2 x$ for some $x \neq 0$. Set $y \coloneqq \frac{1}{\sigma} Ax$. Then $Ax = \sigma y$ and $A^T y = \sigma x$, so

$$
\begin{pmatrix} 0 & A \\ A^T & 0 \end{pmatrix} \begin{pmatrix} y \\ x \end{pmatrix} = \sigma \begin{pmatrix} y \\ x \end{pmatrix}.
$$

Thus $\sigma$ is an eigenvalue. Analogously, setting $y' \coloneqq -\frac{1}{\sigma} Ax$, we get $-\sigma$ as an eigenvalue.

</details>
</div>

As a consequence, we obtain the interlacing property for singular values:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Interlacing Property for Singular Values)</span></p>

Let $A \in \mathbb{R}^{m \times n}$ have rank $r$ and let $B$ be the matrix obtained from $A$ by removing one row or column. Then

$$
\sigma_1(A) \geq \sigma_1(B) \geq \sigma_2(A) \geq \ldots \geq \sigma_{r-1}(A) \geq \sigma_{r-1}(B) \geq \sigma_r(A).
$$

</div>

Another consequence of Theorem 3.7 is the min-max representation of singular values (an adaptation of the Rayleigh--Ritz and Courant--Fischer formulas):

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Min-Max Representation of Singular Values)</span></p>

Let $A \in \mathbb{R}^{m \times n}$. Then

$$
\sigma_1(A) = \max_{\lVert x \rVert_2 = \lVert y \rVert_2 = 1} x^T A y.
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Applying the Rayleigh--Ritz theorem to the matrix $\begin{pmatrix} 0 & A \\ A^T & 0 \end{pmatrix}$:

$$
\sigma_1(A) = \max_{\lVert(x,y)\rVert_2 = 1} \begin{pmatrix} x^T & y^T \end{pmatrix} \begin{pmatrix} 0 & A \\ A^T & 0 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = 2 \max_{\lVert(x,y)\rVert_2 = 1} x^T A y.
$$

This is $\geq 2 \max_{\lVert x \rVert_2 = \lVert y \rVert_2 = 1} x^T A y \cdot \frac{1}{1} = \max_{\lVert x \rVert_2 = \lVert y \rVert_2 = 1} x^T A y$.

On the other hand, let $A = U\Sigma V^T$ and let $u$, $v$ be the first columns of $U$ and $V$. Then $u^T A v = e_1^T \Sigma e_1 = \sigma_1(A)$, so the equality is achieved for $x \coloneqq u$ and $y \coloneqq v$.

</details>
</div>

### Applications of SVD Decomposition

#### SVD and Orthogonalization

The SVD can be used to find an orthonormal basis of (not only) the column space $\mathcal{S}(A)$. Unlike previous approaches, we do not need to assume that the columns of $A$ are linearly independent.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

Let $A = U\Sigma V^T = U_1 S V_1^T$ be the SVD of $A \in \mathbb{R}^{m \times n}$. Then

1. Columns of $U_1$ form an orthonormal basis of $\mathcal{S}(A)$.
2. Columns of $U_2$ form an orthonormal basis of $\operatorname{Ker}(A^T)$.
3. Columns of $V_1$ form an orthonormal basis of $\mathcal{R}(A)$.
4. Columns of $V_2$ form an orthonormal basis of $\operatorname{Ker}(A)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

1. The reduced SVD is $A = U_1 S V_1^T$. Multiplying $V_1$ from the right gives $AV_1 = U_1 S$. Thus $\mathcal{S}(A) \ni \mathcal{S}(AV_1) = \mathcal{S}(U_1 S) = \mathcal{S}(U_1)$ by regularity of $S$. Since $\operatorname{rank}(A) = \operatorname{rank}(U_1)$, we have $\mathcal{S}(A) = \mathcal{S}(U_1)$.
2. Analogous, using (4).
3. Follows from $\mathcal{R}(A) = \mathcal{S}(A^T)$.
4. From the transpose $A^T = V_1 S U_1^T$, the columns of $V_1$ form an orthonormal basis of $\mathcal{S}(A^T) = \mathcal{R}(A) = \operatorname{Ker}(A)^\perp$. Hence columns of $V_2$ represent an orthonormal basis of $\operatorname{Ker}(A)$.

</details>
</div>

#### SVD and Projection onto Subspaces

Using the SVD we can easily express the projection matrix onto the column space (and row space) of a given matrix.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

Let $A = U\Sigma V^T = U_1 S V_1^T$ be the SVD of $A \in \mathbb{R}^{m \times n}$. Then the projection matrix onto

1. the column space $\mathcal{S}(A)$ is $U_1 U_1^T$,
2. the row space $\mathcal{R}(A)$ is $V_1 V_1^T$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

1. By Theorem 3.11, $\mathcal{S}(A) = \mathcal{S}(U_1)$. The columns of $U_1$ are linearly independent, so the projection matrix has the well-known form $U_1(U_1^T U_1)^{-1} U_1^T = U_1 (I_r)^{-1} U_1^T = U_1 U_1^T$.
2. Follows from $\mathcal{R}(A) = \mathcal{S}(A^T)$.

</details>
</div>

#### SVD and Geometry of Linear Maps

Let $A \in \mathbb{R}^{n \times n}$ be a regular matrix and consider the image of the unit ball under $x \mapsto Ax$. From the SVD $A = U\Sigma V^T$ it follows that the linear map decomposes into three elementary maps: orthogonal transformation by $V^T$, scaling by $\Sigma$, and orthogonal transformation by $U$. Specifically, $V^T$ maps the ball onto itself, $\Sigma$ deforms it into an ellipsoid, and $U$ rotates/reflects it. The resulting ellipsoid has semi-axes in the directions of the columns of $U$ with lengths $\sigma_1, \ldots, \sigma_n$.

The ratio $\frac{\sigma_1}{\sigma_n} \geq 1$ is called the **condition number** and quantifies how much the map deforms geometric shapes. If the condition number equals $1$, the ellipsoid is a ball; the larger it is, the more elongated the ellipsoid becomes. In numerical mathematics, $\frac{\sigma_1}{\sigma_n}$ is also called the condition number of the matrix $A$: the larger it is, the worse the matrix $A$ is conditioned, causing issues with rounding in computer arithmetic.

#### SVD and Numerical Rank

The rank of $A$ equals the number of positive singular values. However, for computational purposes, very small positive numbers are treated as effectively zero. Given $\varepsilon > 0$, the **numerical rank** of $A$ is $\max\lbrace s \colon \sigma_s > \varepsilon \rbrace$, counting singular values larger than $\varepsilon$ while treating the rest as zero. For example, **Matlab** / **Octave** use $\varepsilon \coloneqq \max\lbrace m, n \rbrace \cdot \sigma_1 \cdot eps$, where $eps \approx 2 \cdot 10^{-16}$ is the machine precision.

#### SVD and Low-Rank Approximation

Let $A \in \mathbb{R}^{m \times n}$ with $A = U\Sigma V^T$. By keeping only the $k$ largest singular values and setting $\sigma_{k+1} \coloneqq 0, \ldots, \sigma_r \coloneqq 0$, we obtain a matrix

$$
A' = U \operatorname{diag}(\sigma_1, \ldots, \sigma_k, 0, \ldots, 0) V^T
$$

of rank $k$ that well approximates $A$. Moreover, this approximation is optimal in a certain sense --- in a specific norm (see Theorem 5.26), $A'$ is the closest rank-$k$ matrix to $A$.

#### SVD and Data Compression

If $A \in \mathbb{R}^{m \times n}$ with $\operatorname{rank}(A) = r$, the reduced SVD $A = U_1 S V_1^T$ requires storing $(m+r+n)r$ values. A low-rank approximation $A \approx U \operatorname{diag}(\sigma_1, \ldots, \sigma_k, 0, \ldots, 0) V^T$ requires only $(m+n+1)k$ values. The compression ratio is $k : r$. The smaller $k$, the smaller the data to store, but the worse the approximation.

#### SVD and Regularity Measure

The determinant is not a good measure of matrix regularity. Singular values are better suited. For $A \in \mathbb{R}^{n \times n}$, the smallest singular value $\sigma_n$ gives the distance (in a certain norm, see Theorem 5.26) to the nearest singular matrix. Orthogonal matrices have regularity measure $1$, while Hilbert matrices have very small regularity measure, i.e., they are nearly singular:

| $n$ | $\sigma_n(H_n)$ |
| --- | --- |
| 3 | $\approx 0.0027$ |
| 5 | $\approx 10^{-6}$ |
| 10 | $\approx 10^{-13}$ |
| 15 | $\approx 10^{-18}$ |

---

## Pseudoinverse Matrices

The natural desire to generalize the concept of matrix inverse so that the generalized inverse (= pseudoinverse) exists even for singular and rectangular matrices has led to several concepts. The most well-known is the Moore--Penrose pseudoinverse, which exists for every matrix and appears in the orthogonal least squares problem. The Drazin pseudoinverse exists only for square matrices and is more common in non-orthogonal problems.

It is useful to keep in mind that pseudoinverses are rarely computed explicitly. They are used more for establishing and expressing certain properties. Similar to the classical inverse for solving systems $Ax = b$ --- we do not solve via $x = A^{-1}b$, yet the explicit formula is very useful.

### Moore--Penrose Pseudoinverse

The most common pseudoinverse is the Moore--Penrose pseudoinverse, which is based on the SVD.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Moore--Penrose Pseudoinverse)</span></p>

Let $A \in \mathbb{R}^{m \times n}$ with reduced SVD $A = U_1 S V_1^T$. If $A \neq 0$, its **pseudoinverse** is $A^\dagger = V_1 S^{-1} U_1^T \in \mathbb{R}^{n \times m}$. For $A = 0$ we define $A^\dagger = A^T$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

The pseudoinverse of a nonzero vector $a \in \mathbb{R}^n$ is $a^\dagger = \frac{1}{a^T a} a^T$. For instance, $(1,1,1,1)^\dagger = \frac{1}{4}(1,1,1,1)^T$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Properties of the Pseudoinverse)</span></p>

Let $A \in \mathbb{R}^{m \times n}$, then

1. If $A$ is regular, then $A^{-1} = A^\dagger$.
2. $(A^\dagger)^\dagger = A$.
3. $(A^T)^\dagger = (A^\dagger)^T$.
4. $A = AA^\dagger A$.
5. $A^\dagger = A^\dagger A A^\dagger$.
6. $AA^\dagger$ is symmetric.
7. $A^\dagger A$ is symmetric.
8. If $A$ has linearly independent columns, then $A^\dagger = (A^T A)^{-1} A^T$.
9. If $A$ has linearly independent rows, then $A^\dagger = A^T (A A^T)^{-1}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (selected)</summary>

**(4):** From the definition, $AA^\dagger A = U_1 S V_1^T V_1 S^{-1} U_1^T U_1 S V_1^T = U_1 S S^{-1} S V_1^T = U_1 S V_1^T = A$.

**(8):** Here $V_1$ is square, hence orthogonal. Then

$$
(A^T A)^{-1} = (V_1 S U_1^T U_1 S V_1^T)^{-1} = (V_1 S^2 V_1^T)^{-1} = V_1 S^{-2} V_1^T,
$$

and so $(A^T A)^{-1} A^T = V_1 S^{-2} V_1^T V_1 S U_1^T = V_1 S^{-1} U_1^T = A^\dagger$.

</details>
</div>

Properties (4) and (7) are particularly interesting: they provide an alternative definition of the pseudoinverse. A matrix satisfying properties (4)--(7) always exists and is unique.

Note that some properties we might expect to hold do not hold in general. For instance, $AA^\dagger \neq A^\dagger A$ and $(AB)^\dagger \neq B^\dagger A^\dagger$.

Using the pseudoinverse, we can elegantly express projection matrices onto matrix subspaces:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Pseudoinverse and Projections)</span></p>

Let $A \in \mathbb{R}^{m \times n}$. The projection matrix onto

1. the column space $\mathcal{S}(A)$ is $AA^\dagger$,
2. the row space $\mathcal{R}(A)$ is $A^\dagger A$,
3. the kernel $\operatorname{Ker}(A)$ is $I_n - A^\dagger A$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

1. Using the reduced SVD $A = U_1 S V_1^T$: $AA^\dagger = U_1 S V_1^T V_1 S^{-1} U_1^T = U_1 U_1^T$. By Theorem 3.12, this is the projection onto $\mathcal{S}(A)$.
2. Analogously, $A^\dagger A = V_1 V_1^T$, the projection onto $\mathcal{R}(A)$.
3. From the property $\operatorname{Ker}(A) = \mathcal{R}(A)^\perp$.

</details>
</div>

An interesting interpretation of the pseudoinverse comes from the perspective of linear maps:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Pseudoinverse and Linear Maps)</span></p>

Consider the linear map $f(x) = Ax$, where $A \in \mathbb{R}^{m \times n}$.

1. If we restrict the domain of $f(x)$ to the space $\mathcal{R}(A)$, we obtain an isomorphism between $\mathcal{R}(A)$ and $f(\mathbb{R}^n)$.
2. The inverse of this isomorphism is $y \mapsto A^\dagger y$.

</div>

The most significant property of the pseudoinverse is its role in describing the solution set of solvable systems and the set of approximate solutions of unsolvable systems.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Pseudoinverse and Systems of Equations)</span></p>

Let $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$, and $X$ the solution set of the system $Ax = b$. If $X \neq \emptyset$, then

$$
X = A^\dagger b + \operatorname{Ker}(A),
$$

where $\operatorname{Ker}(A) = \mathcal{S}(I_n - A^\dagger A)$.

Moreover, $A^\dagger b$ has the smallest Euclidean norm among all vectors in $X$, and it is the unique solution with this property.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**"$=$":** Let $x \in X$, i.e., $Ax = b$. Then $AA^\dagger b = AA^\dagger A x = Ax = b$, so $A^\dagger b \in X$. We have $X = x_0 + \operatorname{Ker}(A)$, where $x_0$ is any solution. By Theorem 4.4(3), $\operatorname{Ker}(A) = \mathcal{S}(I_n - A^\dagger A)$, and choosing $x_0 = A^\dagger b$ gives the result.

**"Norm":** By Theorem 4.4(2), $A^\dagger b \in \mathcal{R}(A)$, and $\mathcal{R}(A)^\perp = \operatorname{Ker}(A)$. By the Pythagorean theorem, for every $y \in \operatorname{Ker}(A) = \mathcal{S}(I_n - A^\dagger A)$:

$$
\lVert A^\dagger b + y \rVert_2^2 = \lVert A^\dagger b \rVert_2^2 + \lVert y \rVert_2^2 \geq \lVert A^\dagger b \rVert_2^2.
$$

Thus $A^\dagger b$ has the smallest Euclidean norm, since any other vector in $X$ has a larger norm (because $y \neq 0$ implies $\lVert y \rVert_2 > 0$).

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Pseudoinverse and Least Squares)</span></p>

Let $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$, and $X$ the set of approximate solutions of $Ax = b$ by the least squares method. Then

$$
X = A^\dagger b + \operatorname{Ker}(A).
$$

Moreover, $A^\dagger b$ has the smallest Euclidean norm among all vectors in $X$, and it is the unique solution with this property.

</div>

The two preceding theorems together say that $A^\dagger b$ is the **distinguished vector**: if $Ax = b$ has a solution, $A^\dagger b$ is the minimum-norm solution; if $Ax = b$ has no solution, $A^\dagger b$ is the minimum-norm approximate solution by least squares. Neither result requires the assumption of linearly independent columns of $A$.

### Drazin Pseudoinverse

The second type of pseudoinverse is the **Drazin pseudoinverse**. It exists only for square matrices.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Drazin Pseudoinverse)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ and let $A = SJS^{-1}$, where $J$ is the Jordan normal form of $A$. Suppose $J$ has the form $J = \begin{pmatrix} C & 0 \\ 0 & N \end{pmatrix}$, where $C$ is regular and $N$ contains the Jordan blocks for the zero eigenvalue. Then the **Drazin pseudoinverse** is $A^D = S \begin{pmatrix} C^{-1} & 0 \\ 0 & 0 \end{pmatrix} S^{-1}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

Let $A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$. Then $A^\dagger = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$, but $A^D = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$. Thus the Moore--Penrose and Drazin pseudoinverses generally differ.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Properties of the Drazin Pseudoinverse)</span></p>

Let $A \in \mathbb{R}^{m \times n}$, then

1. If $A$ is regular, then $A^{-1} = A^D$.
2. $A^D A A^D = A^D$.
3. $AA^D = A^D A$.
4. $A^{k+1} A^D = A^k$, where $k$ is the smallest integer such that $\operatorname{rank}(A^{k+1}) = \operatorname{rank}(A^k)$ (the so-called *index*).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (selected)</summary>

**(1):** If $A$ is regular, then $A = SCS^{-1}$. Hence $A^{-1} = SC^{-1}S^{-1} = A^D$.

**(2):** $A^D A A^D = S \begin{pmatrix} C^{-1} & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} C & 0 \\ 0 & N \end{pmatrix} \begin{pmatrix} C^{-1} & 0 \\ 0 & 0 \end{pmatrix} S^{-1} = S \begin{pmatrix} C^{-1}CC^{-1} & 0 \\ 0 & 0 \end{pmatrix} S^{-1} = S \begin{pmatrix} C^{-1} & 0 \\ 0 & 0 \end{pmatrix} S^{-1} = A^D$.

</details>
</div>

Drazin originally defined the pseudoinverse as the unique matrix satisfying properties (2)--(4). Thus, like the Moore--Penrose pseudoinverse, the Drazin pseudoinverse can be introduced axiomatically.

The Drazin pseudoinverse appears, for example, in solving systems of linear differential equations and also as the group inverse:

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Matrix Group)</span></p>

It is well known that regular matrices in $\mathbb{R}^{n \times n}$ with multiplication form a group, with the inverse being the classical matrix inverse. In $\mathbb{R}^{n \times n}$ we can find other groups with multiplication.

Let $S \in \mathbb{R}^{n \times n}$ be fixed and regular, and let $k < n$. Define the set $\mathcal{G}$ of all $n \times n$ matrices of the form $S \begin{pmatrix} C & 0 \\ 0 & 0 \end{pmatrix} S^{-1}$, where $C \in \mathbb{R}^{k \times k}$ is regular. All matrices in $\mathcal{G}$ are singular, yet $\mathcal{G}$ forms a group with multiplication. The identity element is $S \begin{pmatrix} I_k & 0 \\ 0 & 0 \end{pmatrix} S^{-1}$ and the inverse of $A = S \begin{pmatrix} C & 0 \\ 0 & 0 \end{pmatrix} S^{-1}$ is $S \begin{pmatrix} C^{-1} & 0 \\ 0 & 0 \end{pmatrix} S^{-1} = A^D$. Thus the group inverses correspond exactly to the Drazin pseudoinverse.

</div>

---

## Matrix Norms

The norm of a matrix is important because it allows us to say how "large" a matrix is, what the distance between two matrices is, whether a sequence of matrices converges, or to define infinite sums of matrices.

Recall that a **vector norm** on a real or complex vector space $V$ is a mapping $\lVert \cdot \rVert \colon V \to \mathbb{R}$ satisfying:

1. $\lVert x \rVert \geq 0$ for all $x \in V$, with equality only for $x = 0$,
2. $\lVert \alpha x \rVert = |\alpha| \cdot \lVert x \rVert$ for all $x \in V$ and all scalars $\alpha$,
3. $\lVert x + y \rVert \leq \lVert x \rVert + \lVert y \rVert$.

Important examples of vector norms on $\mathbb{R}^n$ are the $\ell_p$-norms:

$$
\lVert x \rVert_{\ell_p} = \left( \sum_{i=1}^n |x_i|^p \right)^{1/p},
$$

where $p \geq 1$. For $p \in \lbrace 1, 2 \rbrace$ and the limiting case $p = \infty$ we get the sum, Euclidean, and maximum norms:

$$
\lVert x \rVert_{\ell_1} = \sum_{i=1}^n |x_i|, \quad \lVert x \rVert_{\ell_2} = \sqrt{\sum_{i=1}^n x_i^2}, \quad \lVert x \rVert_{\ell_\infty} = \max_{i=1,\ldots,n} |x_i|.
$$

Since the space of matrices $\mathbb{R}^{n \times n}$ forms a vector space, we can apply vector norms to matrices as well. This gives:

$$
\lVert A \rVert_{\ell_1} = \sum_{i,j=1}^n |a_{ij}|, \quad \lVert A \rVert_{\ell_2} = \sqrt{\sum_{i,j=1}^n a_{ij}^2}, \quad \lVert A \rVert_{\ell_\infty} = \max_{i,j=1,\ldots,n} |a_{ij}|.
$$

In particular, the norm $\lVert A \rVert_{\ell_2}$ is often called the **Frobenius norm** and denoted $\lVert A \rVert_F$. It can be equivalently expressed as

$$
\lVert A \rVert_F = \sqrt{\operatorname{trace}(A^T A)}.
$$

### Definition, Examples, and the Induced Norm

For matrix norms, we additionally require **consistency** (or sub-multiplicativity): $\lVert AB \rVert \leq \lVert A \rVert \cdot \lVert B \rVert$. This is what distinguishes a matrix norm from a mere vector norm applied to the entries.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Matrix Norm)</span></p>

A mapping $\lVert \cdot \rVert \colon \mathbb{R}^{n \times n} \to \mathbb{R}$ is a real **matrix norm** if for all $A \in \mathbb{R}^{n \times n}$ and $\alpha \in \mathbb{R}$:

1. $\lVert A \rVert \geq 0$, with equality only for $A = 0$,
2. $\lVert \alpha A \rVert = |\alpha| \cdot \lVert A \rVert$,
3. $\lVert A + B \rVert \leq \lVert A \rVert + \lVert B \rVert$,
4. $\lVert AB \rVert \leq \lVert A \rVert \cdot \lVert B \rVert$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Claim</span></p>

For every matrix norm and $k \in \mathbb{N}$: $\lVert A^k \rVert \leq \lVert A \rVert^k$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

1. $\lVert A \rVert_{\ell_1}$ is a matrix norm.
2. $\lVert A \rVert_{\ell_2} = \lVert A \rVert_F$ is a matrix norm.
3. $\lVert A \rVert_{\ell_\infty}$ is **not** a matrix norm, but $n \cdot \max_{i,j} |a_{ij}|$ is.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

It suffices to verify property (4):

1. $\lVert AB \rVert_{\ell_1} = \sum_{i,j} |\sum_k a_{ik} b_{kj}| \leq \sum_{i,j,k} |a_{ik} b_{kj}| = (\sum_{i,k} |a_{ik}|)(\sum_{\ell,j} |b_{\ell j}|) = \lVert A \rVert_{\ell_1} \lVert B \rVert_{\ell_1}$.
2. Using the Cauchy--Schwarz inequality: $\lVert AB \rVert_F^2 = \sum_{i,j} (\sum_k a_{ik} b_{kj})^2 \leq \sum_{i,j} (\sum_k a_{ik}^2)(\sum_\ell b_{\ell j}^2) = (\sum_{i,k} a_{ik}^2)(\sum_{\ell,j} b_{\ell j}^2) = \lVert A \rVert_F^2 \lVert B \rVert_F^2$.
3. $\lVert A \rVert_{\ell_\infty}$ is not a matrix norm since for $A = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$ we get $2 = \lVert A^2 \rVert \not\leq \lVert A \rVert^2 = 1$. However, for the scaled variant $n \cdot \max_{i,j} |a_{ij}|$: $n \cdot \max_{i,j} |\sum_k a_{ik} b_{kj}| \leq n \cdot \max_{i,j} \sum_k |a_{ik}| |b_{kj}| \leq n \cdot \lVert A \rVert_{\ell_\infty} \cdot n \cdot \lVert B \rVert_{\ell_\infty}$.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Claim</span></p>

If $\lVert A \rVert$ is a matrix norm, then $\alpha \lVert A \rVert$ is also a matrix norm for all $\alpha \geq 1$.

</div>

If $\lVert v \rVert$ is a vector norm, then scaling $\alpha \lVert v \rVert$ is a vector norm for all $\alpha > 0$. This property does not transfer directly to matrix norms, as the counterexample for $\lVert A \rVert_{\ell_\infty}$ shows. However, a weaker version holds.

Although we already have several examples of matrix norms, the most important ones are the **induced** matrix norms.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Induced Matrix Norm)</span></p>

Let $\lVert \cdot \rVert \colon \mathbb{R}^n \to \mathbb{R}$ be a vector norm. The **induced matrix norm** on $\mathbb{R}^{n \times n}$ is

$$
\lVert A \rVert = \max_{\lVert x \rVert = 1} \lVert Ax \rVert.
$$

</div>

Since the unit sphere is compact and the norm is continuous, the maximum is attained and the induced norm is well-defined. Alternatively:

$$
\lVert A \rVert = \max_{\lVert x \rVert = 1} \lVert Ax \rVert = \max_{\lVert x \rVert \leq 1} \lVert Ax \rVert = \max_{x \neq 0} \frac{\lVert Ax \rVert}{\lVert x \rVert}.
$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric Meaning)</span></p>

Consider a linear map $x \mapsto Ax$ with regular $A$. The unit sphere $\lbrace x \in \mathbb{R}^n \colon \lVert x \rVert = 1 \rbrace$ maps to $\lbrace Ax \in \mathbb{R}^n \colon \lVert x \rVert = 1 \rbrace$. The induced norm $\lVert A \rVert$ is the largest magnitude among vectors in this image, i.e., the maximal stretching of the unit sphere. As we will see in Theorem 5.10, the smallest contraction is $\lVert A^{-1} \rVert^{-1}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

The induced norm is a matrix norm, and moreover $\lVert Ax \rVert \leq \lVert A \rVert \cdot \lVert x \rVert$ (i.e., it is consistent with the vector norm that induces it).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

1. Clearly $\lVert A \rVert \geq 0$. Equality holds iff $\lVert Ax \rVert = 0$ for all $x$, which happens only when $A = 0$.
2. $\lVert \alpha A \rVert = |\alpha| \cdot \lVert A \rVert$ is clear.
3. $\lVert A + B \rVert = \max_{\lVert x \rVert = 1} \lVert (A+B)x \rVert \leq \max_{\lVert x \rVert = 1} (\lVert Ax \rVert + \lVert Bx \rVert) \leq \lVert A \rVert + \lVert B \rVert$.
4. For $Bx \neq 0$: $\lVert AB \rVert = \max_{x \neq 0} \frac{\lVert ABx \rVert}{\lVert x \rVert} = \max_{x \neq 0} \frac{\lVert ABx \rVert}{\lVert Bx \rVert} \cdot \frac{\lVert Bx \rVert}{\lVert x \rVert} \leq \max_{y \neq 0} \frac{\lVert Ay \rVert}{\lVert y \rVert} \cdot \max_{x \neq 0} \frac{\lVert Bx \rVert}{\lVert x \rVert} = \lVert A \rVert \cdot \lVert B \rVert$.
5. Consistency: from the equivalent expression $\lVert A \rVert \geq \frac{\lVert Ax \rVert}{\lVert x \rVert}$ for all $x \neq 0$, we get $\lVert Ax \rVert \leq \lVert A \rVert \cdot \lVert x \rVert$.

</details>
</div>

We denote the matrix norm induced by the $\ell_p$-vector norm as $\lVert A \rVert_p$:

$$
\lVert A \rVert_p = \max_{\lVert x \rVert_{\ell_p} = 1} \lVert Ax \rVert_{\ell_p}.
$$

For $p = 2$, the induced matrix norm is called the **spectral norm** and is one of the most important matrix norms (it is the default norm in **Matlab** / **Octave**).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span></p>

The spectral norm $\lVert A \rVert_2$ is orthogonally invariant, i.e., $\lVert UAV \rVert_2 = \lVert A \rVert_2$ for all orthogonal matrices $U, V$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Using the fact that the Euclidean vector norm is orthogonally invariant:

$$
\lVert UAV \rVert_2 = \max_{\lVert x \rVert_{\ell_2}=1} \lVert UAVx \rVert_{\ell_2} = \max_{\lVert x \rVert_{\ell_2}=1} \lVert AVx \rVert_{\ell_2} = \max_{\lVert Vx \rVert_{\ell_2}=1} \lVert AVx \rVert_{\ell_2} = \max_{\lVert y \rVert_{\ell_2}=1} \lVert Ay \rVert_{\ell_2} = \lVert A \rVert_2.
$$

</details>
</div>

For $p \in \lbrace 1, 2, \infty \rbrace$ the $p$-norm has a simple closed-form expression. For other values of $p$ no simple formula is known, and for $p \notin \lbrace 1, 2, \infty \rbrace$ the computation of $\lVert A \rVert_p$ or its approximation is NP-hard.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

1. $\lVert A \rVert_1 = \max_j \sum_i |a_{ij}|$ (maximum column sum),
2. $\lVert A \rVert_2 = \sigma_1(A)$ (largest singular value of $A$),
3. $\lVert A \rVert_\infty = \max_i \sum_j |a_{ij}| = \lVert A^T \rVert_1$ (maximum row sum).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

It suffices to verify property (4) --- the other axioms follow from Theorem 5.7.

**(1):** Set $h \coloneqq \max_j \sum_i |a_{ij}| = \max_j \lVert A_{\ast j} \rVert_{\ell_1}$.

"$\geq$": Choose $x = e_k$, where $k$ achieves the maximum in $h$. Then $\lVert Ax \rVert_{\ell_1} = \lVert A_{\ast k} \rVert_{\ell_1} = h$.

"$\leq$": $\lVert A \rVert_1 = \max_{\lVert x \rVert_{\ell_1}=1} \lVert Ax \rVert_{\ell_1} = \max_{\lVert x \rVert_{\ell_1}=1} \lVert \sum_j x_j A_{\ast j} \rVert_{\ell_1} \leq \max_{\lVert x \rVert_{\ell_1}=1} \sum_j |x_j| \cdot \lVert A_{\ast j} \rVert_{\ell_1} \leq \max_{\lVert x \rVert_{\ell_1}=1} \sum_j |x_j| h = h$.

**(2):** Let $A = U\Sigma V^T$ be the SVD. By Lemma 5.8, $\lVert A \rVert_2 = \lVert \Sigma \rVert_2 = \max_{\lVert x \rVert_{\ell_2}=1} \lVert \sigma_1 I_n x \rVert_{\ell_2} \leq \max_{\lVert x \rVert_{\ell_2}=1} \sigma_1 \lVert x \rVert_{\ell_2} = \sigma_1$, with equality for $x = e_1$.

**(3):** Set $h \coloneqq \max_i \sum_j |a_{ij}|$.

"$\geq$": Let $k$ achieve the maximum. Choose $x \coloneqq \operatorname{sgn}(A_{k\ast})$, the sign vector of the $k$-th row of $A$. Then $\lVert A \rVert_\infty \geq \lVert Ax \rVert_{\ell_\infty} = h$.

"$\leq$": $\lVert A \rVert_\infty = \max_{\lVert x \rVert_{\ell_\infty}=1} \max_i |\sum_j a_{ij} x_j| \leq \max_{\lVert x \rVert_{\ell_\infty}=1} \max_i \sum_j |a_{ij}| |x_j| \leq \max_i \sum_j |a_{ij}| = h$.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Claim</span></p>

For an induced matrix norm and a regular matrix $A$: $\min_{\lVert x \rVert=1} \lVert Ax \rVert = \lVert A^{-1} \rVert^{-1}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Equivalence of Matrix Norms)</span></p>

All matrix norms on $\mathbb{R}^{n \times n}$ are equivalent in the following sense: for any two norms $\lVert \cdot \rVert_\alpha$, $\lVert \cdot \rVert_\beta$, there exist $\gamma, \delta > 0$ such that for all $A \in \mathbb{R}^{n \times n}$:

$$
\gamma \lVert A \rVert_\alpha \leq \lVert A \rVert_\beta \leq \delta \lVert A \rVert_\alpha.
$$

For specific norms, the conversion constants are known, for example:

$$
\frac{1}{\sqrt{n}} \lVert A \rVert_1 \leq \lVert A \rVert_2 \leq \sqrt{n} \lVert A \rVert_1, \quad \frac{1}{\sqrt{n}} \lVert A \rVert_\infty \leq \lVert A \rVert_2 \leq \sqrt{n} \lVert A \rVert_\infty, \quad \lVert A \rVert_2 \leq \lVert A \rVert_F \leq \sqrt{n} \lVert A \rVert_2.
$$

</div>

### Spectral Radius vs. Matrix Norm

One important relationship between matrix norms is their connection to the spectral radius. The spectral radius is **not** a matrix norm, as the following example shows.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

The spectral radius is not a matrix norm because:

1. $\rho(A) = 0$ even for $A \neq 0$, e.g., $A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$.
2. $\rho(A+B) \leq \rho(A) + \rho(B)$ does not hold, e.g., $A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$, $B = A^T$.
3. $\rho(AB) \leq \rho(A)\rho(B)$ does not hold, e.g., $A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, $B = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$, since $\rho(A) = \rho(B) = 1$ but $\rho(AB) = (1+\sqrt{5})/2$.

</div>

Even if we restrict to symmetric matrices, the spectral radius is still not a matrix norm (the counterexamples are non-diagonalizable, but the product of symmetric matrices need not be symmetric either).

Nevertheless, the spectral radius can be approximated from above by matrix norms, and from below as well:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bounding the Spectral Radius)</span></p>

Let $A \in \mathbb{R}^{n \times n}$. For every matrix norm: $\rho(A) \leq \lVert A \rVert$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $\lambda \in \mathbb{C}$ be any eigenvalue and $x$ the corresponding eigenvector, so $Ax = \lambda x$. Define $X \coloneqq (x \mid 0 \mid \cdots \mid 0)$. Since $AX = \lambda X$:

$$
|\lambda| \cdot \lVert X \rVert = \lVert \lambda X \rVert = \lVert AX \rVert \leq \lVert A \rVert \cdot \lVert X \rVert.
$$

Dividing by $\lVert X \rVert \neq 0$ gives $|\lambda| \leq \lVert A \rVert$.

</details>
</div>

The inequality can be strict --- for instance, $A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$ has $\rho(A) = 0 < \lVert A \rVert$ for any matrix norm. Moreover, $\lVert A \rVert - \rho(A)$ can be arbitrarily large (consider $A = \begin{pmatrix} 0 & \alpha \\ 0 & 0 \end{pmatrix}$).

On the other hand, we can get arbitrarily close to the spectral radius by choosing a suitable norm:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span></p>

Let $S \in \mathbb{R}^{n \times n}$ be regular and $\lVert \cdot \rVert$ a matrix norm on $\mathbb{R}^{n \times n}$. Then $\lVert A \rVert_S \coloneqq \lVert SAS^{-1} \rVert$ is also a matrix norm.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

Let $A \in \mathbb{R}^{n \times n}$ and $\varepsilon > 0$. Then there exists a matrix norm such that $\rho(A) \leq \lVert A \rVert \leq \rho(A) + \varepsilon$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Every square matrix is similar to a matrix in Jordan normal form. There exists a regular $R \in \mathbb{C}^{n \times n}$ such that $RAR^{-1} = J$. Define $D$ as a diagonal matrix with entries $t, t^2, \ldots, t^n$ for a parameter $t > 0$. Then

$$
DJD^{-1} = \begin{pmatrix} \lambda_1 & t^{-1}\lbrace 0,1 \rbrace & 0 & \ldots & 0 \\ 0 & \lambda_2 & t^{-1}\lbrace 0,1 \rbrace & \ldots & 0 \\ \vdots & \ddots & \ddots & \ddots & \vdots \\ 0 & \ldots & & \lambda_{n-1} & t^{-1}\lbrace 0,1 \rbrace \\ 0 & \ldots & \ldots & 0 & \lambda_n \end{pmatrix},
$$

where $\lbrace 0,1 \rbrace$ denotes either $0$ or $1$ (the exact Jordan block structure is not important here). For large $t > 0$, the off-diagonal entries are arbitrarily small and the column sums are arbitrarily close to $\lambda_i$. Setting $S = DR$ and using $\lVert A \rVert_S \coloneqq \lVert SAS^{-1} \rVert_1 = \lVert DJD^{-1} \rVert_1 \leq \rho(A) + \varepsilon$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Bounds for Polynomial Roots)</span></p>

Consider a polynomial $p(x) = x^n + a_{n-1}x^{n-1} + \ldots + a_1 x + a_0$. Its **companion matrix** is

$$
C(p) \coloneqq \begin{pmatrix} 0 & \ldots & \ldots & 0 & -a_0 \\ 1 & \ddots & & \vdots & -a_1 \\ 0 & \ddots & \ddots & \vdots & -a_2 \\ \vdots & \ddots & \ddots & 0 & \vdots \\ 0 & \ldots & 0 & 1 & -a_{n-1} \end{pmatrix},
$$

and the roots of $p(x)$ equal the eigenvalues of $C(p)$. For any root $x^*$ and any matrix norm: $|x^*| \leq \rho(C(p)) \leq \lVert C(p) \rVert$. Using $\lVert \cdot \rVert_\infty$:

$$
|x^*| \leq \lVert C(p) \rVert_\infty = \max\lbrace |a_0|, 1 + |a_1|, \ldots, 1 + |a_{n-1}| \rbrace \leq 1 + \max_{i=0,\ldots,n-1} |a_i|,
$$

which is the Cauchy bound.

</div>

### Power Sequences

We now turn to power sequences $A, A^2, A^3, \ldots$ and their convergence. By $\lim_{k \to \infty} B_k = A$ we mean $\lVert B_k - A \rVert \to_{k \to \infty} 0$. Since all norms on $\mathbb{R}^{n \times n}$ are equivalent, matrix convergence is equivalent to entrywise convergence.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Sufficient Condition for Convergence)</span></p>

Let $A \in \mathbb{R}^{n \times n}$. If $\lVert A \rVert < 1$ for some matrix norm, then $\lim_{k \to \infty} A^k = 0$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By the sub-multiplicativity property, $\lVert A^k \rVert \leq \lVert A \rVert^k \to_{k \to \infty} 0$. Thus $A^k \to_{k \to \infty} 0$.

</details>
</div>

The converse does not hold for every norm (consider $A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$ and certain norms; also $A = \begin{pmatrix} 0 & \alpha \\ 0 & 0 \end{pmatrix}$ with $\alpha$ large). On the other hand, via the spectral radius we reach a full equivalence --- bringing power sequences and Neumann series together:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Neumann Series)</span></p>

Let $A \in \mathbb{R}^{n \times n}$. The following are equivalent:

1. $\rho(A) < 1$,
2. $\lim_{k \to \infty} A^k = 0$,
3. $\sum_{k=0}^\infty A^k$ converges.

If any of these holds, then $(I_n - A)^{-1} = \sum_{k=0}^\infty A^k$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**(1)$\Rightarrow$(2):** Since $\rho(A) < 1$, by Theorem 5.15 there exists a matrix norm with $\lVert A \rVert < 1$. By Theorem 5.17, $A^k \to 0$.

**(1)$\Leftarrow$(2):** Let $\lambda$ be an eigenvalue and $x$ the eigenvector. Then $A^k x = \lambda^k x$. Since $A^k \to 0$, we need $\lambda^k \to 0$, hence $|\lambda| < 1$. Thus $\rho(A) < 1$.

**(3)$\Rightarrow$(2):** Clear (terms of a convergent series tend to zero).

**(2)$\Rightarrow$(3):** $(I_n - A)(\sum_{k=0}^m A^k) = I_n - A^{m+1}$ converges to $I_n$ as $m \to \infty$. Since $I_n - A^{m+1}$ is regular for large $m$, so is $I_n - A$. Thus $\sum_{k=0}^m A^k = (I_n - A)^{-1}(I_n - A^{m+1}) \to (I_n - A)^{-1}$.

</details>
</div>

Note that $(I_n - A)^{-1}$ may exist without any of the three conditions holding. The Neumann series is an immediate application: it can be used to approximate the inverse $(I_n - A)^{-1}$ using partial sums $\sum_{k=0}^m A^k$. If $\rho(A)$ is small, the approximation $(I_n - A)^{-1} \approx I + A$ may already be useful.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Gelfand's Formula)</span></p>

For every matrix norm $a$ and $A \in \mathbb{R}^{n \times n}$: $\rho(A) = \lim_{k \to \infty} \lVert A^k \rVert^{1/k}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

"$\leq$": $\rho(A)^k = \rho(A^k) \leq \lVert A^k \rVert$, so $\rho(A) \leq \lVert A^k \rVert^{1/k}$. This holds in the limit as well.

"$\geq$": Let $\varepsilon > 0$ and define $B \coloneqq \frac{1}{\rho(A)+\varepsilon} A$. Then $\rho(B) = \frac{\rho(A)}{\rho(A)+\varepsilon} < 1$, so $B^k \to 0$. This gives $\lVert A^k \rVert = (\rho(A)+\varepsilon)^k \lVert B^k \rVert$, so $\lVert A^k \rVert^{1/k} < \rho(A) + \varepsilon$ for all sufficiently large $k$. Since $\varepsilon$ is arbitrary, $\lVert A^k \rVert^{1/k}$ is eventually close to $\rho(A)$.

</details>
</div>

### Orthogonally Invariant Norms

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Orthogonally Invariant Norm)</span></p>

A matrix norm $\lVert A \rVert$ is **orthogonally invariant** if $\lVert UAV \rVert = \lVert A \rVert$ for all orthogonal matrices $U, V$.

</div>

For any orthogonally invariant norm and $A \in \mathbb{R}^{m \times n}$ with $A = U\Sigma V^T$: $\lVert A \rVert = \lVert U\Sigma V^T \rVert = \lVert \Sigma \rVert$, so the norm depends only on the singular values of $A$.

We already know that the spectral norm $\lVert A \rVert_2$ is orthogonally invariant (Lemma 5.8). It is not the only such norm. The Frobenius norm is also orthogonally invariant since it depends only on singular values:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

Let $A \in \mathbb{R}^{m \times n}$ have singular values $\sigma_1, \ldots, \sigma_r$. Then $\lVert A \rVert_F = \sqrt{\sum_{i=1}^r \sigma_i^2}$.

</div>

Other examples of orthogonally invariant norms include:

- The **Ky Fan norm**: $\lVert A \rVert_{K_k} \coloneqq \sum_{i=1}^k \sigma_i$ (sum of the $k$ largest singular values). For $k=1$ this is the spectral norm and for $k = r$ the **nuclear norm** (also called the **trace norm**): $\lVert A \rVert_* = \sum_{i=1}^r \sigma_i$.
- The **Schatten $p$-norm**: $\lVert A \rVert_{S_p} \coloneqq \left( \sum_{i=1}^r \sigma_i^p \right)^{1/p}$.

The nuclear norm is the best convex lower bound to the rank of a matrix within the unit ball. It can be used for matrix rank minimization problems --- for instance, in robust PCA, where instead of minimizing the rank we minimize the nuclear norm. Its relationship with the Frobenius norm is:

$$
\lVert A \rVert_F = \sqrt{\operatorname{trace}(A^T A)}, \quad \lVert A \rVert_* = \operatorname{trace}\left(\sqrt{A^T A}\right).
$$

### Further Applications of Matrix Norms

#### Interpretation of Singular Values

An interesting property of singular values is that $\sigma_i$ gives, in the spectral norm, the distance from the matrix to the nearest matrix of rank at most $i-1$. This is not a coincidence --- it is how low-rank approximation via SVD works.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Interpretation of Singular Values)</span></p>

Let $A \in \mathbb{R}^{m \times n}$ have singular values $\sigma_1, \ldots, \sigma_r$. Then $\sigma_i = \min\lbrace \lVert A - B \rVert_2 \colon B \in \mathbb{R}^{m \times n},\ \operatorname{rank}(B) \leq i-1 \rbrace$ for each $i = 1, \ldots, r$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**"$\geq$":** Let $A = U\Sigma V^T$ be the SVD. Define the rank-$(i-1)$ matrix $B \coloneqq U \operatorname{diag}(\sigma_1, \ldots, \sigma_{i-1}, 0, \ldots, 0) V^T$. Then

$$
\lVert A - B \rVert_2 = \lVert U \operatorname{diag}(0, \ldots, 0, \sigma_i, \ldots, \sigma_n) V^T \rVert_2 = \lVert \operatorname{diag}(0, \ldots, 0, \sigma_i, \ldots, \sigma_n) \rVert_2 = \sigma_i.
$$

**"$\leq$":** Let $B \in \mathbb{R}^{n \times n}$ have rank at most $i-1$. Let $V_1$ consist of the first $i$ columns of $V$. Take $0 \neq z \in \operatorname{Ker}(B) \cap \mathcal{S}(V_1)$ (such $z$ exists since $\dim \operatorname{Ker}(B) \geq n - i + 1$ and $\dim \mathcal{S}(V_1) = i$). Normalize $\lVert z \rVert_2 = 1$. Since $z \in \mathcal{S}(V_1)$, write $z = V y$ for $y = (y_1, \ldots, y_i, 0, \ldots, 0)^T$ with $\lVert y \rVert_2 = 1$. Then

$$
\lVert A - B \rVert_2^2 \geq \lVert (A-B)z \rVert_2^2 \geq \lVert Az \rVert_2^2 = \lVert U\Sigma V^T z \rVert_2^2 = \lVert \Sigma y \rVert_2^2 = \sum_{j=1}^i \sigma_j^2 y_j^2 \geq \sigma_i^2 \lVert y \rVert_2^2 = \sigma_i^2.
$$

</details>
</div>

In particular, the smallest singular value $\sigma_n$ of $A \in \mathbb{R}^{n \times n}$ is the distance to the nearest singular matrix. This means $A + C$ is regular for all $C \in \mathbb{R}^{n \times n}$ with $\lVert C \rVert_2 < \sigma_n$. The result holds analogously for other orthogonally invariant norms, so the distance in the Frobenius norm is also $\sigma_n$.

#### Prokrustes Problem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Prokrustes Problem)</span></p>

Let $A, B \in \mathbb{R}^{m \times n}$. The question is whether $A$ differs from $B$ by a rotation, i.e., whether there exists an orthogonal $Q \in \mathbb{R}^{m \times m}$ such that $A = QB$. We solve the more general optimization problem

$$
\min\lbrace \lVert A - QB \rVert_F \colon Q \in \mathbb{R}^{m \times m} \text{ orthogonal} \rbrace.
$$

Let $AB^T = U\Sigma V^T$ be the SVD of $AB^T$. Then the optimum is $Q = UV^T$. Moreover, there exists an orthogonal $Q$ with $A = QB$ exactly when $\lVert A \rVert_F = \lVert B \rVert_F = \sqrt{\sum_i \sigma_i}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The Frobenius norm is induced by the scalar product $\langle A, B \rangle = \operatorname{trace}(A^T B)$. Expanding:

$$
\lVert A - QB \rVert_F^2 = \lVert A \rVert_F^2 + \lVert QB \rVert_F^2 - 2 \langle QB, A \rangle = \lVert A \rVert_F^2 + \lVert B \rVert_F^2 - 2\langle QB, A \rangle.
$$

Since $\lVert A \rVert_F^2 + \lVert B \rVert_F^2$ is constant, we maximize $\langle QB, A \rangle = \operatorname{trace}(B^T Q^T A) = \operatorname{trace}(U\Sigma V^T Q^T)$ over orthogonal $Q$. Setting $H = V^T Q^T U$, we get $\langle QB, A \rangle = \operatorname{trace}(\Sigma H) = \sum_i \sigma_i h_{ii}$. Since $h_{ii} \in [-1,1]$ for orthogonal $H$, the maximum $\sum_i \sigma_i$ is achieved when $H = I_m$, i.e., $Q = UV^T$.

For the second part, $A = QB$ requires $\lVert A \rVert_F = \lVert B \rVert_F$ by orthogonal invariance of the Frobenius norm. The first part shows $Q$ exists iff $0 = \lVert A - QB \rVert_F^2 = 2\lVert A \rVert_F^2 - 2\langle QB, A \rangle$, i.e., $\lVert A \rVert_F^2 = \sum_i \sigma_i$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

For $B = I$, the best orthogonal approximation of $A$ is $Q = UV^T$, where $A = U\Sigma V^T$ is the SVD of $A$.

</div>

#### Interpretation of the Moore--Penrose Pseudoinverse

A similar optimization leads to an interpretation of the pseudoinverse: $AA^\dagger$ is the best approximation of the identity $I_m$ in the Frobenius norm, among matrices of the form $AX$:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Interpretation of the Moore--Penrose Pseudoinverse)</span></p>

Let $A \in \mathbb{R}^{m \times n}$. The matrix $X = A^\dagger$ is the optimal solution of

$$
\min\lbrace \lVert I_m - AX \rVert_F \colon X \in \mathbb{R}^{n \times m} \rbrace.
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By a matrix Pythagorean theorem (Lemma 5.24): for $X \coloneqq I_m - AA^\dagger$ and $Y \coloneqq A(A^\dagger - X)$, if $\mathcal{S}(X) \perp \mathcal{S}(Y)$, then $\lVert X + Y \rVert_F^2 = \lVert X \rVert_F^2 + \lVert Y \rVert_F^2$.

Since $\mathcal{S}(I_m - AA^\dagger) \subseteq \mathcal{S}(A) = \operatorname{Ker}(A^T)^\perp$ and $\mathcal{S}(AA^\dagger - AX) \subseteq \mathcal{S}(A)$ (by Theorem 4.4, $I_m - AA^\dagger$ projects onto $\operatorname{Ker}(A^T)$):

$$
\lVert I_m - AX \rVert_F^2 = \lVert I_m - AA^\dagger + AA^\dagger - AX \rVert_F^2 = \lVert I_m - AA^\dagger \rVert_F^2 + \lVert AA^\dagger - AX \rVert_F^2 \geq \lVert I_m - AA^\dagger \rVert_F^2.
$$

Thus $X = A^\dagger$ minimizes $\lVert I_m - AX \rVert_F$.

</details>
</div>

#### Density of Diagonalizable Matrices

Using matrix norms, we can formally express the fact that any matrix is arbitrarily close to a diagonalizable one. This means the set of diagonalizable matrices is dense in $\mathbb{R}^{n \times n}$ (or $\mathbb{C}^{n \times n}$).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

For every matrix $A \in \mathbb{C}^{n \times n}$ and every $\varepsilon > 0$ there exists a diagonalizable matrix $A' \in \mathbb{C}^{n \times n}$ such that $\lVert A - A' \rVert < \varepsilon$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By the Schur decomposition (Theorem 2.1), $A = UTU^*$, where $U$ is unitary and $T$ is upper triangular. Set $A' = UT'U^*$, where $T' = T + \operatorname{diag}(\delta, \ldots, \delta^n)$ for small $\delta > 0$. If $\delta < |t_{ii} - t_{jj}|/2$ for all $i \neq j$, then $T'$ has distinct diagonal entries, hence distinct eigenvalues, and $A'$ is diagonalizable. Moreover,

$$
\lVert A - A' \rVert_2 = \lVert U \operatorname{diag}(\delta, \ldots, \delta^n) U^* \rVert_2 = \lVert \operatorname{diag}(\delta, \ldots, \delta^n) \rVert_2 = \delta,
$$

so for the spectral norm it suffices to take $\delta < \min_{i \neq j} \lbrace \varepsilon, \frac{1}{2}|t_{ii} - t_{jj}| \rbrace$.

</details>
</div>

---

## Condition Number

The condition number characterizes a square matrix and roughly indicates how inaccurately numerical computations involving this matrix behave, i.e., how large numerical errors it causes. The smallest condition number is $1$, achieved by well-conditioned matrices (typically orthogonal matrices). The larger the condition number, the worse the matrix is conditioned (i.e., the closer it is to a singular matrix).

The concept was introduced by Turing in 1948 for the Frobenius norm, though a similar idea had been used by von Neumann and Goldstine in 1947.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Condition Number)</span></p>

The **condition number** of a regular matrix $A \in \mathbb{R}^{n \times n}$ is $k(A) = \lVert A \rVert \cdot \lVert A^{-1} \rVert$, where $\lVert \cdot \rVert$ is an induced matrix norm. For the $p$-norm specifically, $k_p(A) = \lVert A \rVert_p \cdot \lVert A^{-1} \rVert_p$.

</div>

Although the definition also makes sense for non-induced matrix norms, induced norms are preferred due to the following properties:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Claim</span></p>

For an induced matrix norm: $\lVert I_n \rVert = 1$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Claim</span></p>

$k(A) \geq 1$. Equality holds, in particular, when $A$ is symmetric and $k_2$ is used.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Claim</span></p>

$k(A) \geq \frac{|\lambda_{\max}(A)|}{|\lambda_{\min}(A)|}$. Equality holds when $A$ is symmetric and $k_2(A)$ is used.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Claim</span></p>

1. $k(AB) \leq k(A) k(B)$.
2. $k(\alpha A) = k(A)$ for $\alpha \neq 0$.

</div>

Thus the condition number is independent of scaling. On the other hand, multiplying matrices can increase the condition number, and in particular, powers of a matrix can lead to exponential growth of the condition number.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The condition number serves mainly to derive theoretical properties. It is rarely computed directly, since computing $A^{-1}$ introduces numerical errors. Instead, it is estimated. The reciprocal of the condition number gives the relative distance to the nearest singular matrix in the corresponding matrix norm (Gastinel & Kahan, 1966). Hence matrices with large condition numbers behave poorly numerically.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Matrices occurring in various applications are not random, so it is hard to determine the distribution of their condition numbers. An interesting result by Demmel (1988) shows that a randomly chosen matrix (under uniform distribution) of order $n$ has condition number of value at most $n(n^2 - 1)h^{-2}$ with probability at most $h$. Thus the probability of a large condition number decreases quadratically with its reciprocal.

</div>

### Condition Number for the Spectral Norm

The spectral norm is the most commonly used norm for the condition number, because many of its nice properties transfer to $k(A)$ under other norms by equivalence.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

1. $k_2(A) = \sigma_1 / \sigma_n$,
2. $k_2(A) = k_2(A^T)$,
3. $k_2(A^T A) = k_2(A)^2$,
4. $k_2(A) = 1$ exactly when $A$ is a nonzero scalar multiple of an orthogonal matrix.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $A = U\Sigma V^T$ be the SVD.

1. $k_2(A) = \lVert A \rVert_2 \cdot \lVert A^{-1} \rVert_2 = \sigma_1 / \sigma_n$.
2. $A$ and $A^T$ have the same singular values.
3. $A^T A = (U\Sigma V^T)^T (U\Sigma V^T) = V \Sigma^2 V^T$, so the singular values of $A^T A$ are $\sigma_1^2, \ldots, \sigma_n^2$.
4. "$\Leftarrow$": Orthogonal matrices have all singular values equal to $1$, so $k_2 = 1$.
   "$\Rightarrow$": If $k_2(A) = 1$, all singular values must be equal, say $\sigma_1 = \sigma_n = c > 0$. Then $A = U \Sigma V^T = U(cI_n) V^T = c(UV^T)$, where $UV^T$ is orthogonal.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric Interpretation I)</span></p>

Geometrically, $k_2(A) = \cot(\varphi/2)$, where $\varphi$ is the minimum angle between $Ax$ and $Ay$ over all $x, y \in \mathbb{R}^n \setminus \lbrace 0 \rbrace$ with $x \perp y$. If $A$ is orthogonal, the minimum angle is $90°$ and $k_2(A) = 1$. The smaller the angle, the larger the condition number.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric Interpretation II)</span></p>

For $A \in \mathbb{R}^{n \times n}$ regular, the SVD shows that the image of the unit ball under $x \mapsto Ax$ is an ellipsoid with semi-axes of lengths $\sigma_1, \ldots, \sigma_n$. The condition number $k_2(A) = \sigma_1/\sigma_n$ measures how much the map deforms circles into ellipses. If $k_2(A) = 1$, the ellipsoid is a ball; the larger the condition number, the more elongated it is.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Numerical Interpretation)</span></p>

An empirical rule says that if the condition number is approximately $10^k$, then computations involving the matrix (inverses, solving systems, etc.) lose about $k$ decimal digits of precision. Orthogonal matrices ($k_2 = 1$) are optimal numerically, which is why they are often used in numerical linear algebra. In contrast, Hilbert matrices are very ill-conditioned:

| $n$ | condition number of $H_n$ |
| --- | --- |
| 3 | $\approx 500$ |
| 5 | $\approx 10^5$ |
| 10 | $\approx 10^{13}$ |
| 15 | $\approx 10^{17}$ |

</div>

### Condition Number for Solving Linear Systems

Consider the system $Ax = b$ and suppose $\tilde{x}$ is a numerically computed solution. We want to know how accurate it is. A natural idea is to compute the **residual**:

$$
r \coloneqq b - A\tilde{x}.
$$

One might expect that small $r$ means $\tilde{x}$ is close to $x$, and large $r$ means it is far. However, this depends on the condition number of $A$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

Let $x \coloneqq A^{-1} b$ with $b \neq 0$. Then:

$$
k(A)^{-1} \frac{\lVert r \rVert}{\lVert b \rVert} \leq \frac{\lVert x - \tilde{x} \rVert}{\lVert x \rVert} \leq k(A) \frac{\lVert r \rVert}{\lVert b \rVert}.
$$

(Using any vector norm and $k(A)$ with the corresponding induced matrix norm.)

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Upper bound:** We have $x - \tilde{x} = A^{-1}b - \tilde{x} = A^{-1}(b - A\tilde{x}) = A^{-1}r$ and $\lVert b \rVert = \lVert Ax \rVert \leq \lVert A \rVert \cdot \lVert x \rVert$. Thus

$$
\lVert x - \tilde{x} \rVert = \lVert A^{-1}r \rVert \leq \lVert A^{-1} \rVert \cdot \lVert r \rVert \leq \frac{\lVert A \rVert \cdot \lVert x \rVert}{\lVert b \rVert} \lVert A^{-1} \rVert \cdot \lVert r \rVert = k(A) \cdot \lVert x \rVert \frac{\lVert r \rVert}{\lVert b \rVert}.
$$

**Lower bound:** $\lVert r \rVert = \lVert b - A\tilde{x} \rVert = \lVert A(x - \tilde{x}) \rVert \leq \lVert A \rVert \cdot \lVert x - \tilde{x} \rVert$ and $\lVert x \rVert = \lVert A^{-1}b \rVert \leq \lVert A^{-1} \rVert \cdot \lVert b \rVert$. Hence

$$
\lVert r \rVert \leq \lVert A \rVert \cdot \lVert x - \tilde{x} \rVert \frac{\lVert x \rVert}{\lVert x \rVert} \leq \lVert A \rVert \cdot \lVert x - \tilde{x} \rVert \frac{\lVert A^{-1} \rVert \cdot \lVert b \rVert}{\lVert x \rVert} = k(A) \cdot \frac{\lVert b \rVert}{\lVert x \rVert} \lVert x - \tilde{x} \rVert.
$$

</details>
</div>

The more important bound is the upper one: $\tilde{x}$ is a good approximation when the residual $r$ is small **and** the condition number $k(A)$ is small. If the matrix is ill-conditioned, even a small residual can correspond to a large error. If $k(A) = 1$, the normalized residual $\frac{\lVert r \rVert}{\lVert b \rVert}$ equals the relative distance $\frac{\lVert x - \tilde{x} \rVert}{\lVert x \rVert}$ exactly.

### Condition Number for Eigenvalue Computation

Let $\tilde{\lambda}$ be an estimate of an eigenvalue and $\tilde{x}$ (with $\lVert \tilde{x} \rVert = 1$) an estimate of the eigenvector of $A \in \mathbb{R}^{n \times n}$. How good are these estimates? Again, we examine the residual:

$$
r \coloneqq A\tilde{x} - \tilde{\lambda}\tilde{x}.
$$

As in the previous section, the quality of the approximation depends on the condition number of $A$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bauer--Fike, 1960)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ be diagonalizable with $A = S\Lambda S^{-1}$ (spectral decomposition). Then there exists an eigenvalue $\lambda$ of $A$ such that

$$
|\lambda - \tilde{\lambda}| \leq k_p(S) \cdot \lVert r \rVert_{\ell_p}.
$$

(Using any $\ell_p$-vector norm and the corresponding induced matrix norm for $k(A)$.)

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $r = A\tilde{x} - \tilde{\lambda}\tilde{x} = (S\Lambda S^{-1} - \tilde{\lambda}I_n)\tilde{x} = S(\Lambda - \tilde{\lambda}I_n)S^{-1}\tilde{x}$, we get $\tilde{x} = S(\Lambda - \tilde{\lambda}I_n)^{-1}S^{-1}r$. Let $\lambda$ be the eigenvalue of $A$ closest to $\tilde{\lambda}$. Then

$$
1 = \lVert \tilde{x} \rVert = \lVert S(\Lambda - \tilde{\lambda}I_n)^{-1}S^{-1}r \rVert \leq \lVert S \rVert \cdot \lVert (\Lambda - \tilde{\lambda}I_n)^{-1} \rVert \cdot \lVert S^{-1} \rVert \cdot \lVert r \rVert = k_p(S) \cdot \lVert r \rVert \cdot \lVert (\Lambda - \tilde{\lambda}I_n)^{-1} \rVert \leq k_p(S) \cdot \lVert r \rVert \cdot |\lambda - \tilde{\lambda}|^{-1},
$$

from which $|\lambda - \tilde{\lambda}| \leq k_p(S) \cdot \lVert r \rVert$.

</details>
</div>

For symmetric matrices, the eigenvector matrix $S$ is orthogonal, so $k_2(S) = 1$. The Bauer--Fike theorem then takes a simpler form:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span></p>

If $A \in \mathbb{R}^{n \times n}$ is symmetric, then $|\lambda - \tilde{\lambda}| \leq \lVert r \rVert_2$.

</div>

---

## Perturbations

In this chapter we ask how the solution of a given problem (systems of linear equations, eigenvalues of a matrix, etc.) changes when we slightly modify the input values. In the chapter on condition numbers we were interested in numerical properties of a matrix. Now we focus more on how much the solution can change, both for discrete perturbations and for infinitesimal changes.

### Perturbation of Eigenvalues

For eigenvalue perturbation of general matrices, we can use the following version of the Bauer--Fike theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bauer--Fike, 1960)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ be diagonalizable with spectral decomposition $A = S\Lambda S^{-1}$. Let $B \in \mathbb{R}^{n \times n}$ be a perturbation matrix. Then for every eigenvalue $\mu$ of $A + B$ there exists an eigenvalue $\lambda$ of $A$ such that

$$
|\lambda - \mu| \leq k_p(S) \cdot \lVert B \rVert_p.
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Apply Theorem 6.13 (Bauer--Fike for eigenvalue computation) with $\tilde{\lambda} = \mu$ and $\tilde{x}$ the eigenvector of $A + B$ of norm $1$. Then

$$
|\lambda - \mu| \leq k_p(S) \cdot \lVert r \rVert_p = k_p(S) \cdot \lVert A\tilde{x} - \mu\tilde{x} \rVert_p = k_p(S) \cdot \lVert (A+B)\tilde{x} - (A+B)\tilde{x} + A\tilde{x} - \mu\tilde{x} \rVert_p = k_p(S) \cdot \lVert B\tilde{x} \rVert_p \leq k_p(S) \cdot \lVert B \rVert_p.
$$

</details>
</div>

If $A$ is symmetric, then $S$ is orthogonal and $k_2(S) = 1$, giving the bound

$$
|\lambda - \mu| \leq \lVert B \rVert_2.
$$

For symmetric matrices we can derive stronger results --- we can compare individual eigenvalues of $A$ and $A + B$. Denote eigenvalues of a symmetric matrix in decreasing order: $\lambda_1 \geq \ldots \geq \lambda_n$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Weyl, 1912, Special Version)</span></p>

Let $A, B \in \mathbb{R}^{n \times n}$ be symmetric. Then for every $i = 1, \ldots, n$:

$$
\lambda_i(A) + \lambda_n(B) \leq \lambda_i(A+B) \leq \lambda_i(A) + \lambda_1(B).
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Fix $i \in \lbrace 1, \ldots, n \rbrace$ and set $C \coloneqq A + B$. Let $a_1, \ldots, a_n$, resp. $c_1, \ldots, c_n$ be the orthonormal eigenvector bases of $A$, resp. $C$, corresponding to eigenvalues in decreasing order. Define

$$
U = \operatorname{span}\lbrace a_1, \ldots, a_i \rbrace, \quad V = \operatorname{span}\lbrace c_i, \ldots, c_n \rbrace.
$$

By dimension counting, $\dim(U \cap V) \geq i + (n - i + 1) - n = 1$. So there exists $x \in U \cap V$ with $\lVert x \rVert_2 = 1$. Using $x \in V$:

$$
x^T C x = \sum_{k=i}^n \gamma_k^2 \lambda_k(C) \leq \sum_{k=i}^n \gamma_k^2 \lambda_i(C) = \lambda_i(C).
$$

Using $x \in U$:

$$
x^T C x = x^T(A+B)x = x^T A x + x^T B x \geq \lambda_i(A) + \lambda_n(B).
$$

This gives the lower bound. The upper bound is proved analogously.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Weyl, 1912, General Version)</span></p>

Let $A, B \in \mathbb{R}^{n \times n}$ be symmetric. Then for every $i = 1, \ldots, n$:

$$
\lambda_i(A+B) \leq \lambda_j(A) + \lambda_{i-j+1}(B), \quad \forall\, j \leq i,
$$

$$
\lambda_i(A+B) \geq \lambda_j(A) + \lambda_{i-j+n}(B), \quad \forall\, j \geq i.
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Fix $i \in \lbrace 1, \ldots, n \rbrace$. Let $a_1, \ldots, a_n$, resp. $b_1, \ldots, b_n$, resp. $c_1, \ldots, c_n$ be orthonormal eigenvector bases of $A$, resp. $B$, resp. $C \coloneqq A + B$. Define

$$
V_1 = \operatorname{span}\lbrace a_1, \ldots, a_j \rbrace, \quad V_2 = \operatorname{span}\lbrace b_1, \ldots, b_{i-j+n} \rbrace, \quad V_3 = \operatorname{span}\lbrace c_i, \ldots, c_n \rbrace.
$$

By an intersection dimension argument: $\dim(V_1 \cap V_2 \cap V_3) \geq j + (i-j+n) + (n-i+1) - n - n = 1$. So there exists $x \in V_1 \cap V_2 \cap V_3$ with $\lVert x \rVert_2 = 1$. Using $x \in V_3$: $x^T C x \leq \lambda_i(C)$. Using $x \in V_1 \cap V_2$: $x^T C x = x^T A x + x^T B x \geq \lambda_j(A) + \lambda_{i-j+n}(B)$.

This gives the lower bound. The upper bound is proved analogously.

</details>
</div>

As a consequence, we get perturbation bounds on all eigenvalues simultaneously, including a bound on the spectral radius:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span></p>

Let $A, B \in \mathbb{R}^{n \times n}$ be symmetric. Then $\rho(A+B) \leq \rho(A) + \rho(B)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By the special Weyl theorem:

$$
\lambda_1(A+B) \leq \lambda_1(A) + \lambda_1(B) \leq \rho(A) + \rho(B),
$$

$$
\lambda_n(A+B) \geq \lambda_n(A) + \lambda_n(B) \leq -\rho(A) - \rho(B).
$$

Thus $\rho(A+B) = \max\lbrace \lambda_1(A+B), -\lambda_n(A+B) \rbrace \leq \rho(A) + \rho(B)$.

</details>
</div>

Note that this bound does **not** hold for general (non-symmetric) matrices. For example, $A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$, $B = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$ have $\rho(A) = \rho(B) = 0$ but $\rho(A+B) = 1$.

Another consequence of the Weyl theorem is a uniform bound on the change of eigenvalues:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span></p>

Let $A, B \in \mathbb{R}^{n \times n}$ be symmetric. Then for every $i = 1, \ldots, n$:

$$
|\lambda_i(A+B) - \lambda_i(A)| \leq \rho(B).
$$

Moreover, by Theorem 5.13 we can also use matrix norms: $|\lambda_i(A+B) - \lambda_i(A)| \leq \lVert B \rVert$. For the spectral norm, this result is stronger than the Bauer--Fike bound, because unlike the latter, it compares the $i$-th eigenvalues directly.

</div>

### Continuity of Eigenvalues

We now move from discrete perturbations (represented by additive matrices $B$) to infinitesimally small perturbations.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

The eigenvalues of a matrix $A \in \mathbb{C}^{n \times n}$ are continuous functions of the entries of $A$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Eigenvalues are roots of the characteristic polynomial, and roots are continuous functions with respect to the coefficients of the polynomial.

</details>
</div>

Eigenvectors are generally **not** continuous. For example, the matrix

$$
A = \begin{pmatrix} 0 & 1 \\ 0 & \alpha \end{pmatrix}
$$

has, for $\alpha = 0$, eigenvalue $0$ with eigenvector $(1,0)^T$ only. For $\alpha > 0$, it has eigenvalue $\alpha$ with eigenvector $(1,\alpha)^T$. So the eigenvectors are indeed two for $\alpha > 0$ but only one for $\alpha = 0$.

Under additional conditions, eigenvectors can be made continuous:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span></p>

Let $A \in \mathbb{C}^{n \times n}$ be regular and $b \in \mathbb{C}^n$. Then there exists a neighborhood of $A$ and $b$ on which the solution of $Ax = b$ is a continuous function of the entries of $A$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

If $A \in \mathbb{C}^{n \times n}$ has mutually distinct eigenvalues, then the eigenvectors can be chosen as continuous functions of the entries of $A$ in a neighborhood of $A$.

</div>

Continuity of eigenvalues is useful in many contexts. A noteworthy application is the **Gershgorin disc theorem**: every eigenvalue $\lambda$ of $A \in \mathbb{C}^{n \times n}$ lies in a disc centered at $a_{ii}$ with radius $\sum_{j \neq i} |a_{ij}|$ for some $i \in \lbrace 1, \ldots, n \rbrace$. Moreover:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Gershgorin Disc Refinement)</span></p>

Each connected component of the union of Gershgorin discs contains exactly as many eigenvalues as the number of discs forming that component.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Decompose $A = D + N$ into diagonal and off-diagonal parts. Define $A(t) = D + tN$ for $t \in [0,1]$. Then $A(0) = D$ has eigenvalues $a_{11}, \ldots, a_{nn}$ and $A(1) = A$. Since eigenvalues vary continuously, as $t$ goes from $0$ to $1$, no eigenvalue can jump from one connected component of Gershgorin discs to another, disjoint one.

</details>
</div>

### Matrix Derivatives

Consider a matrix $A \in \mathbb{R}^{n \times n}$ depending on a parameter $t \in \mathbb{R}$, i.e., each entry $a_{ij}(t) \colon \mathbb{R} \to \mathbb{R}$ is a real function. The **derivative** of $A$ is the matrix of derivatives of individual entries:

$$
A' = A(t)' = \begin{pmatrix} a_{11}(t)' & \ldots & a_{1n}(t)' \\ \vdots & & \vdots \\ a_{n1}(t)' & \ldots & a_{nn}(t)' \end{pmatrix}.
$$

Recall also that the Jacobian of a vector function $f \colon \mathbb{R}^n \to \mathbb{R}^n$ is

$$
\frac{\partial f}{\partial x} = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \ldots & \frac{\partial f_1}{\partial x_n} \\ \vdots & & \vdots \\ \frac{\partial f_n}{\partial x_1} & \ldots & \frac{\partial f_n}{\partial x_n} \end{pmatrix},
$$

with gradients of $f_1, \ldots, f_n$ in the columns.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Claim</span></p>

For two parametric matrices: $(AB)' = A'B + AB'$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Claim</span></p>

For a parametric regular matrix: $(A^{-1})' = -A^{-1}A'A^{-1}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By differentiating $0 = I_n' = (AA^{-1})' = A'A^{-1} + A(A^{-1})'$, we get $A(A^{-1})' = -A'A^{-1}$ and thus $(A^{-1})' = -A^{-1}A'A^{-1}$.

</details>
</div>

#### Derivatives of Solutions of Linear Systems

Consider the system $Ax = b$ with regular $A$, so $x = A^{-1}b$. Three cases arise:

1. **$A$ is fixed, $b$ is parametric.** Then $x' = (A^{-1}b)' = A^{-1}b'$. If only entry $b_k$ changes ($b_k(t) = b_k + t$), then $x' = A^{-1}e_k = A^{-1}_{\ast k}$. In summary, the Jacobian is $\frac{\partial x}{\partial b} = A^{-T}$.

2. **$A$ is parametric, $b$ is fixed.** Then $x' = (A^{-1})'b = -A^{-1}A'A^{-1}b = -A^{-1}A'x$. If only entry $a_{ij}$ changes ($a_{ij}(t) = a_{ij} + t$), then $x' = -A^{-1}e_i e_j^T x = -A^{-1}_{\ast i} x_j$.

3. **Both $A$ and $b$ are parametric.** Then $x' = (A^{-1}b)' = (A^{-1})'b + A^{-1}b' = -A^{-1}A'A^{-1}b + A^{-1}b' = -A^{-1}A'x + A^{-1}b'$.

Note that $A^{-1}$ always appears in the derivative formulas. This is not a coincidence --- it is directly connected to the condition number in Definition 6.1.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

Consider the parametric system

$$
\begin{pmatrix} 1+t & 2+t \\ 3+t & 4+t \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} 1 \\ 5 \end{pmatrix}.
$$

For $t = 0$, the solution is $x = (3, -1)^T$ and $A^{-1} = \frac{1}{2}\begin{pmatrix} -4 & 2 \\ 3 & -1 \end{pmatrix}$. Computing $x' = (2, -2)^T$. So as $t$ increases, the first component grows and the second decreases at the same rate.

</div>

#### Derivatives of Eigenvalues

Recall that the **left eigenvector** of $A$ is defined as a (right) eigenvector of $A^T$. The following theorem gives an elegant formula for the derivative of an eigenvalue.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

Let $A \in \mathbb{C}^{n \times n}$ have mutually distinct eigenvalues. Let $\lambda \in \mathbb{C}$ be an eigenvalue, and $x, y \in \mathbb{C}^n$ its right and left eigenvectors normalized so that $x^T y = 1$. Then

$$
\lambda' = y^T A' x.
$$

In particular, if only entry $a_{ij}$ changes ($a_{ij}(t) = a_{ij} + t$), then $\lambda' = y_i x_j$.

If $A$ is moreover real symmetric, then $\lambda' = x_i x_j$ (since left and right eigenvectors coincide).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $A$ is diagonalizable, $A = X\Lambda X^{-1}$, where $\Lambda$ is diagonal with eigenvalues, $X$ has right eigenvectors in columns, and $X^{-1}$ has left eigenvectors in rows. The normalization $x^T y = 1$ is ensured by $X^{-1}X = I_n$.

Differentiating $AX = X\Lambda$:

$$
A'X + AX' = X'\Lambda + X\Lambda'.
$$

Thus $A'X - X\Lambda' = X'\Lambda - AX'$, which gives

$$
X^{-1}A'X - \Lambda' = X^{-1}X'\Lambda - X^{-1}AXX^{-1}X' = X^{-1}X'\Lambda - \Lambda X^{-1}X'.
$$

The diagonal entries of the right-hand side are zero (since $(X^{-1}X'\Lambda - \Lambda X^{-1}X')_{kk} = 0$), so we get $(X^{-1}A'X - \Lambda')_{kk} = 0$ for each $k$.

For a specific eigenvalue, this gives $y^T A' x = \lambda'$.

If only $a_{ij}$ changes, $A' = e_i e_j^T$, so $\lambda' = y^T e_i e_j^T x = y_i x_j$. For a real symmetric matrix, left and right eigenvectors coincide, so $\lambda' = x_i x_j$.

</details>
</div>

If $A$ is symmetric and only a diagonal entry $a_{ii}$ changes, then $\lambda' = x_i^2 \geq 0$. This means that increasing any diagonal entry of a symmetric matrix does not decrease any eigenvalue --- each eigenvalue either increases or stays the same.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

Consider $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$. It has eigenvalues $3$ and $1$ with eigenvectors $\frac{\sqrt{2}}{2}(1,1)^T$ and $\frac{\sqrt{2}}{2}(1,-1)^T$. By the theorem:

$$
\frac{\partial \lambda_1}{\partial a_{11}} = \frac{1}{2}, \quad \frac{\partial \lambda_1}{\partial a_{12}} = \frac{1}{2}, \quad \frac{\partial \lambda_2}{\partial a_{11}} = \frac{1}{2}, \quad \frac{\partial \lambda_2}{\partial a_{12}} = -\frac{1}{2}.
$$

So the matrix $\begin{pmatrix} 2.1 & 1 \\ 1 & 2 \end{pmatrix}$ should have eigenvalues $\approx 3.05$ and $\approx 1.05$, and the matrix $\begin{pmatrix} 2 & 1.1 \\ 1.1 & 2 \end{pmatrix}$ should have eigenvalues $\approx 3.1$ and $\approx 0.9$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Magnus, 1985)</span></p>

Consider the symmetric matrix

$$
A = \begin{pmatrix} 1 + \varepsilon & \delta \\ \delta & 1 + \varepsilon \end{pmatrix}.
$$

Its eigenvalues are $\lambda_{1,2} = 1 \pm \sqrt{\varepsilon^2 + \delta^2}$, which form a double cone in the $(\varepsilon, \delta, \lambda)$ space. To choose the two eigenvalues $\lambda_{1,2}$ as continuous functions of $(\varepsilon, \delta)$, the only option is for one eigenvalue to correspond to the upper half of the cone and the other to the lower half. Then the eigenvalues are not smooth at the origin. Note that both parameters $(\varepsilon, \delta)$ are needed; with a single parameter, the eigenvalue set has a "$\chi$" shape and the eigenvalues can be chosen as smooth "$\backslash$" and "$/$" functions.

</div>

#### Other Applications of Matrix Derivatives

Matrix derivatives appear in many areas, for instance in optimization. A well-known application is the **least squares method**.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Least Squares Method)</span></p>

Consider the system $Ax = b$ where $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$, and $A$ has rank $n$. Typically $m \gg n$, so the system has no exact solution. We seek the approximate solution by minimizing

$$
\min_{x \in \mathbb{R}^n} \lVert Ax - b \rVert_2.
$$

Since $\lVert \cdot \rVert_2$ is a monotone function and its square attains the minimum at the same point, we minimize

$$
\lVert Ax - b \rVert_2^2 = (Ax - b)^T(Ax-b) = x^T A^T A x - 2b^T A x + b^T b.
$$

This is a convex function, so the minimum is attained where the gradient vanishes. The gradient is $2A^T Ax - 2A^T b$. Setting it to zero gives the **normal equations** $A^T Ax = A^T b$. Since $A$ has linearly independent columns, $A^T A$ is regular and the unique solution is $x = (A^T A)^{-1} A^T b$.

</div>

---

## Nonnegative and Positive Matrices

Nonnegative and positive matrices have interesting eigenvalue properties, which is why they merit special attention. A matrix is **nonnegative** (resp. **positive**) if all its entries are $\geq 0$ (resp. $> 0$). We write $A \geq 0$ resp. $A > 0$. Similarly, componentwise operations and absolute value are applied entrywise.

The theory of nonnegative and positive matrices is often called **Perron's theory** after the German mathematician Oskar Perron. In summary:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Perron, 1907)</span></p>

1. Let $A \in \mathbb{R}^{n \times n}$ be nonnegative. Then the largest eigenvalue (in absolute value) is real nonnegative and the corresponding eigenvector is nonnegative (in all components).
2. Let $A \in \mathbb{R}^{n \times n}$ be positive. Then the largest eigenvalue (in absolute value) is real positive, has multiplicity $1$, and the corresponding eigenvector is positive (in all components). Moreover, no other eigenvalue has a nonnegative eigenvector.

</div>

We build up the theory step by step.

### Basic Results for Nonnegative Matrices

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

If $|A| \leq B$ (entrywise), then $\rho(A) \leq \rho(B)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

From $|A| \leq B$ we deduce $|A|^k \leq B^k$ entrywise, so $\lVert A^k \rVert_F \leq \lVert |A|^k \rVert_F \leq \lVert B^k \rVert_F$. Thus $\lVert A^k \rVert_F^{1/k} \leq \lVert B^k \rVert_F^{1/k}$ and by Gelfand's formula (Theorem 5.19), $\rho(A) \leq \rho(B)$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span></p>

If $A \geq 0$ and $\tilde{A}$ is a principal submatrix of $A$ (obtained by deleting some rows and corresponding columns), then $\rho(\tilde{A}) \leq \rho(A)$. In particular, $a_{ii} \leq \rho(A)$ for all $i$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span></p>

Let $A \geq 0$ with constant row sums. Then $\rho(A) = \lVert A \rVert_\infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

If $A \geq 0$, then $\min_i \sum_j a_{ij} \leq \rho(A) \leq \max_i \sum_j a_{ij}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Define $\alpha \coloneqq \min_i \sum_j a_{ij}$ and the matrix $B$ with entries $b_{ij} \coloneqq a_{ij} \frac{\alpha}{\sum_j a_{ij}}$ (for $\alpha = 0$ the claim is trivial, so assume $\alpha > 0$). Then $0 \leq B \leq A$ and the row sums of $B$ are all $\alpha$. By the Lemma and Theorem 8.2: $\alpha = \rho(B) \leq \rho(A)$.

The second inequality is directly $\rho(A) \leq \lVert A \rVert_\infty$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

Let $A \geq 0$ and $x > 0$. Then the following implications hold:

$$
\alpha x \leq Ax \;\Rightarrow\; \alpha \leq \rho(A), \qquad \alpha x < Ax \;\Rightarrow\; \alpha < \rho(A),
$$

$$
Ax \leq \beta x \;\Rightarrow\; \rho(A) \leq \beta, \qquad Ax < \beta x \;\Rightarrow\; \rho(A) < \beta.
$$

</div>

Choosing the best possible $\alpha, \beta$ gives the following bounds for $\rho(A)$. The quality of the bounds depends on the choice of the positive vector $x > 0$. If $\rho(A)$ is an eigenvalue with a positive eigenvector (which is always the case for positive matrices, as we will see), the Collatz inequalities become equalities. Hence, the closer $x$ is to the eigenvector estimate, the tighter the bounds.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Collatz, 1942)</span></p>

If $A \geq 0$ and $x > 0$, then

$$
\min_i \frac{(Ax)_i}{x_i} \leq \rho(A) \leq \max_i \frac{(Ax)_i}{x_i}.
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span></p>

If $A \geq 0$ has an eigenvector $x > 0$, then the corresponding eigenvalue is $\rho(A)$.

</div>

### Specific Results for Positive Matrices

We now prove the essential part of Perron's theorem --- that $\rho(A) > 0$ is an eigenvalue with a positive eigenvector.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

Let $A > 0$, $Ax = \lambda x$, $x \neq 0$, $|\lambda| = \rho(A)$. Then $A|x| = \rho(A)|x|$ and $|x| > 0$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

$\rho(A)|x| = |\lambda| \cdot |x| = |\lambda x| = |Ax| \leq A|x|$ (entrywise). If equality $A|x| = \rho(A)|x|$ holds, we are done since $A|x| > 0$ implies $|x| > 0$.

Otherwise, define $z \coloneqq A|x| > 0$. Then $\rho(A)z < Az$ and by Theorem 8.6, $\rho(A) < \rho(A)$ --- a contradiction.

</details>
</div>

The next part of Perron's theorem says the largest eigenvalue is unique --- there are no two distinct eigenvalues $\lambda_1, \lambda_2$ with $|\lambda_1| = |\lambda_2| = \rho(A)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

Let $A > 0$ and $\lambda \neq \rho(A)$ be an eigenvalue. Then $|\lambda| < \rho(A)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Suppose $|\lambda| = \rho(A)$. Let $x$ be the eigenvector for $\lambda$. By Theorem 8.9, $|Ax| = A|x|$, so for every $k$: $|\sum_j a_{kj} x_j| = \sum_j a_{kj} |x_j|$. Triangle inequality for complex numbers becomes equality exactly when all numbers lie on the same ray from the origin. Thus there exists $\gamma \in \mathbb{C}$ such that $\gamma a_{kj} x_j > 0$ for all $j$ (with $\gamma$ aligning the ray with the positive real axis). For all $j$, $\gamma x_j$ is real positive, and hence from $A(\gamma x) = \lambda(\gamma x)$, the eigenvector $\gamma x > 0$ is positive. By Corollary 8.8, $\lambda = \rho(A)$ --- a contradiction.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

Let $A > 0$ and let $x \geq 0$, $x \neq 0$ be its eigenvector. Then $Ax = \rho(A) x$.

</div>

To complete Perron's theorem for positive matrices, it remains to show that $\rho(A)$ has multiplicity $1$.

### Specific Results for Nonnegative Matrices

Part of Perron's theorem for nonnegative matrices can be shown simply by a limiting argument from positive matrices: for $A \geq 0$, consider $A + \varepsilon e e^T > 0$ for $\varepsilon > 0$; by the limit $\varepsilon \to 0$ and continuity of eigenvalues/eigenvectors (Section 7.2), we obtain the desired properties.

This shows that the largest eigenvalue is real nonnegative and the corresponding eigenvector is nonnegative. However, the remaining properties of positive matrices are generally lost:

- $\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ has the largest eigenvalue of multiplicity greater than one.
- $\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ has the largest eigenvalue $1$ with eigenvector $(1,0)^T$ which is not positive. Another eigenvalue $0$ also has a nonnegative eigenvector $(0,1)^T$.
- $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ has the largest eigenvalue $0$, even though it is nonnegative and nonzero.

For certain nonnegative matrices, the properties of positive matrices can be strengthened. This leads to the theory of **irreducible** matrices. A matrix $A$ is irreducible if no permutation matrix $P$ exists such that $P^T AP$ is block upper triangular. For a nonnegative irreducible matrix, $\rho(A) > 0$ is an eigenvalue of multiplicity $1$ with a positive eigenvector. Thus nearly all properties of positive matrices hold; the only exception is that $\rho(A)$ may be attained by multiple eigenvalues (e.g., $A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ has eigenvalues $\pm 1$).

### Application --- Verification of Linear Systems

Throughout this text, we studied numerical problems when solving systems of linear equations. We identified matrices prone to numerical errors via the condition number and studied changes in solutions under perturbations. The goal of **verification** is to numerically determine a rigorous (upper) bound on the distance of a numerically computed solution from the actual one.

Consider $Ax = b$ where $A \in \mathbb{R}^{n \times n}$ and $b \in \mathbb{R}^n$. Let $x^*$ be the numerically computed solution. The goal is to find a vector $y^\Delta > 0$ such that $|x - x^*| \leq y^\Delta$, i.e., $x \in x^* + [-y^\Delta, y^\Delta]$. Geometrically, we seek a hyperrectangle around $x^*$ containing the true solution.

Systems are often preconditioned: if $C \in \mathbb{R}^{n \times n}$ is regular, $CAx = Cb$ is an equivalent system. For our purposes, $C \coloneqq A^{-1}$ or its approximation is the best choice.

The approach uses **interval arithmetic** --- operations on intervals $[\underline{a}, \overline{a}]$, $[\underline{b}, \overline{b}]$ defined as the image of the interval under the given operation.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Krawczyk, 1969; Moore, 1977; Rump, 1980)</span></p>

If

$$
C(b - Ax^*) + (I - CA)[-y^\Delta, y^\Delta] \subseteq \operatorname{int}[-y^\Delta, y^\Delta],
$$

then $A$ and $C$ are regular and $A^{-1}b \in x^* + [-y^\Delta, y^\Delta]$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Consider the map $y \mapsto C(b - Ax^*) + (I - CA)y$. The inclusion condition says that $[-y^\Delta, y^\Delta]$ maps into its own interior. By Brouwer's fixed-point theorem (continuous maps of convex compact sets into themselves have a fixed point), there exists a fixed point $y$: $y = C(b - Ax^*) + (I - CA)y$, i.e., $CA(x^* + y) = Cb$. By Theorems 8.2 and 8.6, the inclusion implies $|I - CA| y^\Delta < y^\Delta$, so $\rho(|I - CA|) < 1$. The eigenvalues of $CA$ are all nonzero, hence $CA$ (and individually $A$, $C$) are regular.

</details>
</div>

In practice: $C$ is a numerical approximation of $A^{-1}$, $x^*$ is the approximate solution, and $[-y^\Delta, y^\Delta]$ is a small initial hyperrectangle. If condition (8.1) holds (verified with interval arithmetic and correct rounding), we have a rigorous bound. Otherwise, we enlarge $[-y^\Delta, y^\Delta]$ and repeat.

## Matrix Functions and Power Series

We have already briefly encountered matrix power series in the Neumann series theorem. Now we study matrix series more generally, within the framework of matrix functions.

### Matrix Functions

The question is: how to define a matrix function such as $\cos(A)$, $e^A$, etc.? For a real function $f \colon \mathbb{R} \to \mathbb{R}$ and a matrix $A \in \mathbb{R}^{n \times n}$, defining $f(A)$ by applying $f$ to each entry individually,

$$
f(A) = \begin{pmatrix} f(a_{11}) & \ldots & f(a_{1n}) \\ \vdots & & \vdots \\ f(a_{n1}) & \ldots & f(a_{nn}) \end{pmatrix},
$$

is possible but lacks nice properties. Instead, assume $f \colon \mathbb{R} \to \mathbb{R}$ is smooth enough to be expressed as a power series $f(x) = \sum_{k=0}^{\infty} a_k x^k$ (i.e., $f$ is analytic --- real analytic functions like $\sin(x)$, $e^x$ satisfy this). Then we naturally define

$$
f(A) = \sum_{k=0}^{\infty} a_k A^k.
$$

This series is evaluated using the Jordan normal form. Let $A = SJS^{-1}$ where $J$ is the Jordan normal form of $A$. Since $A^k = SJ^kS^{-1}$, we get

$$
f(A) = \sum_{k=0}^{\infty} a_k SJ^kS^{-1} = Sf(J)S^{-1}.
$$

We have reduced the problem to evaluating $f$ for matrices in Jordan normal form. Starting with the simplest case: when $J$ is diagonal (i.e., $A$ is diagonalizable) with diagonal entries $\lambda_1, \ldots, \lambda_n$ (eigenvalues of $A$), then

$$
J^k = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)^k = \operatorname{diag}(\lambda_1^k, \ldots, \lambda_n^k),
$$

and therefore

$$
f(J) = \sum_{k=0}^{\infty} a_k \begin{pmatrix} \lambda_1^k & & \\ & \ddots & \\ & & \lambda_n^k \end{pmatrix} = \begin{pmatrix} f(\lambda_1) & & \\ & \ddots & \\ & & f(\lambda_n) \end{pmatrix}.
$$

In the general case, when $J$ is block diagonal with Jordan blocks $J_{k_1}(\lambda_1), \ldots, J_{k_m}(\lambda_m)$, we have

$$
f(J) \coloneqq \begin{pmatrix} f(J_{k_1}(\lambda_1)) & & \\ & \ddots & \\ & & f(J_{k_m}(\lambda_m)) \end{pmatrix}.
$$

It remains to define the image of a single Jordan block $J_{k_i}(\lambda_i)$. For $k_i = 1$ this is simply $f(\lambda_i)$. For $k_i > 1$, one shows:

$$
f(J_{k_i}(\lambda_i)) \coloneqq \begin{pmatrix} f(\lambda_i) & f'(\lambda_i) & \ldots & \frac{f^{(k_i-1)}(\lambda_i)}{(k_i-1)!} \\ 0 & \ddots & \ddots & \vdots \\ \vdots & \ddots & \ddots & f'(\lambda_i) \\ 0 & \ldots & 0 & f(\lambda_i) \end{pmatrix}.
$$

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof sketch (for $k_i = 2$)</summary>

Consider a Jordan block of size $2$: $A = J_2(\lambda) = \begin{pmatrix} \lambda & 1 \\ 0 & \lambda \end{pmatrix}$. Define $A_\varepsilon \coloneqq \begin{pmatrix} \lambda & 1 \\ 0 & \lambda + \varepsilon \end{pmatrix}$ where $\varepsilon \neq 0$. This matrix is diagonalizable:

$$
A_\varepsilon = S_\varepsilon \Lambda_\varepsilon S_\varepsilon^{-1}, \quad S_\varepsilon = \begin{pmatrix} 1 & 1 \\ 0 & \varepsilon \end{pmatrix}, \quad \Lambda_\varepsilon = \begin{pmatrix} \lambda & 0 \\ 0 & \lambda + \varepsilon \end{pmatrix}, \quad S_\varepsilon^{-1} = \begin{pmatrix} 1 & -1/\varepsilon \\ 0 & 1/\varepsilon \end{pmatrix}.
$$

Therefore

$$
f(A_\varepsilon) = S_\varepsilon \begin{pmatrix} f(\lambda) & 0 \\ 0 & f(\lambda + \varepsilon) \end{pmatrix} S_\varepsilon^{-1} = \begin{pmatrix} f(\lambda) & (f(\lambda + \varepsilon) - f(\lambda))/\varepsilon \\ 0 & f(\lambda + \varepsilon) \end{pmatrix}.
$$

Taking the limit $\varepsilon \to 0$ gives $f(A) = \begin{pmatrix} f(\lambda) & f'(\lambda) \\ 0 & f(\lambda) \end{pmatrix}$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

1. The function $f(x) = x^2$ gives the matrix function $f(A) = A^2$, which is the standard matrix squaring. Note that applying $f$ entry-wise would square individual entries --- not what we want.
2. For $f(x) = \sqrt{x}$ and a positive semidefinite matrix, $f(A) = \sqrt{A}$ yields the standard matrix square root.
3. For $f(x) = \frac{1}{1-x}$ and a matrix $A \in \mathbb{R}^{n \times n}$ with $\rho(A) < 1$, we recover the Neumann series: $(I_n - A)^{-1} = \sum_{k=0}^{\infty} A^k$.

</div>

The above approach allows defining matrix functions for many real functions, especially analytic ones. Thus we can consider $\sin(A)$, $\cos(A)$, $e^A$, and study their properties. Some well-known identities generalize, for example

$$
\sin(2A) = 2\sin(A)\cos(A), \qquad \sin^2(A) + \cos^2(A) = I_n.
$$

Others, however, do not generalize straightforwardly (see Theorem 9.4 below).

### Matrix Exponential

The exponential is one of the most important matrix functions. For $A \in \mathbb{R}^{n \times n}$, the matrix exponential is defined by the power series

$$
e^A = \sum_{k=0}^{\infty} \frac{1}{k!} A^k.
$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Derivation of $e^A$)</span></p>

We already know that a matrix function can be evaluated on individual Jordan blocks. Let $A = \lambda I_n + N$ where $N = J_n(0)$ is nilpotent ($N^m = 0$ for $m \geq n$). Then

$$
A^k = (\lambda I_n + N)^k = \sum_{m=0}^{k} \binom{k}{m} \lambda^{k-m} N^m,
$$

since $(N^m)_{i,i+m} = 1$ for $i = 1, \ldots, n-m$ and zero elsewhere, and $N^m = 0$ for $m \geq n$. Thus

$$
e^A = \sum_{k=0}^{\infty} \frac{1}{k!} \sum_{m=0}^{k} \binom{k}{m} \lambda^{k-m} N^m = \sum_{m=0}^{\infty} \frac{1}{m!} N^m \sum_{k=m}^{\infty} \frac{1}{(k-m)!} \lambda^{k-m} = \sum_{m=0}^{\infty} \frac{1}{m!} N^m e^\lambda
$$

$$
= e^\lambda \sum_{m=0}^{n-1} \frac{1}{m!} N^m = e^\lambda \begin{pmatrix} 1 & \frac{1}{1!} & \cdots & \frac{1}{(n-1)!} \\ 0 & \ddots & \ddots & \vdots \\ \vdots & \ddots & \ddots & \frac{1}{1!} \\ 0 & \ldots & 0 & 1 \end{pmatrix}.
$$

</div>

We now state several important properties of the matrix exponential.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

$\det(e^A) = e^{\operatorname{trace}(A)}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $A = SJS^{-1}$ where $J$ is in Jordan normal form. Then $\det(e^A) \equiv \det(Se^JS^{-1}) = \det(e^J)$, and

$$
e^J = \sum_{k=0}^{\infty} \frac{1}{k!} J^k = \begin{pmatrix} e^{\lambda_1} & ? \\ & \ddots \\ 0 & & e^{\lambda_n} \end{pmatrix}.
$$

Therefore $\det(e^J) = e^{\lambda_1} \cdots e^{\lambda_n} = e^{\lambda_1 + \cdots + \lambda_n} = e^{\operatorname{trace}(A)}$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

If $AB = BA$, then $e^{A+B} = e^A e^B$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We have

$$
e^{A+B} = \sum_{k=0}^{\infty} \frac{1}{k!}(A+B)^k = \sum_{k=0}^{\infty} \frac{1}{k!} \sum_{m=0}^{k} \binom{k}{m} A^m B^{k-m},
$$

where the binomial theorem applies because $AB = BA$. Reindexing:

$$
= \sum_{m=0}^{\infty} \sum_{k=m}^{\infty} \frac{1}{m!} A^m \frac{1}{(k-m)!} B^{k-m} = \sum_{m=0}^{\infty} \frac{1}{m!} A^m e^B = e^A e^B.
$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The commutativity condition $AB = BA$ is essential. For non-commuting matrices, the identity $e^{A+B} = e^A e^B$ generally fails. A counterexample: $A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$, $B = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$; then $e^{A+B} = \begin{pmatrix} e & e-1 \\ 0 & 1 \end{pmatrix} \neq \begin{pmatrix} e & e \\ 0 & 1 \end{pmatrix} = e^A e^B$.

</div>

#### Applications

**Differential equations.** Consider the linear ODE $y' = ay$ where $y = y(t) \colon \mathbb{R} \to \mathbb{R}$ and $a \in \mathbb{R}$ is a constant. The solution is $y = c \cdot e^{at}$ for any $c \in \mathbb{R}$ (determined by the initial condition $y(0) = c$).

Now consider a system of $n$ linear ODEs

$$
y' = Ay,
$$

where $y = y(t) \colon \mathbb{R}^n \to \mathbb{R}$ is the unknown vector function and $A \in \mathbb{R}^{n \times n}$ is a fixed matrix. The solution is $y(t) = e^{At}c$, where $c \in \mathbb{R}^n$ is an arbitrary vector. Indeed:

$$
y(t)' = \Big(\sum_{k=0}^{\infty} \frac{1}{k!} A^k t^k c\Big)' = \sum_{k=1}^{\infty} \frac{1}{k!} A^k k t^{k-1} c = A \sum_{k=1}^{\infty} \frac{1}{(k-1)!} A^{k-1} t^{k-1} c = A e^{At} c = Ay.
$$

**Graph centrality.** Let $A$ be the adjacency matrix of an undirected graph, i.e., $A_{ij} = 1$ if vertices $i,j$ are connected by an edge and $A_{ij} = 0$ otherwise. Then $(A^k)_{ij}$ counts the number of walks of length $k$ between vertices $i$ and $j$. The matrix exponential $e^A = \sum_{k=0}^{\infty} \frac{1}{k!} A^k$ can be interpreted as a weighted average count of walks. The largest diagonal entry of $e^A$ is called the **graph centrality** and identifies the most important vertex in a certain sense.

**Rotation matrices.** The matrix exponential can be used to express rotations in $\mathbb{R}^3$. The matrix $e^R$, where

$$
R = \alpha \begin{pmatrix} 0 & -z & y \\ z & 0 & -x \\ -y & x & 0 \end{pmatrix},
$$

describes a rotation around the axis $(x,y,z)^T$ by angle $\alpha$ (right-hand rule).

## Nonstandard Matrix Products

Besides the standard matrix product, there are other useful matrix products. Their significance lies mainly in simplifying complex expressions and they appear in certain situations. Even though they are not as important as the standard product, it is good to know about them.

### Kronecker Product

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Kronecker Product)</span></p>

Let $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{p \times q}$. The **Kronecker product** (also called the tensor product) of matrices $A, B$ is the matrix $A \otimes B \in \mathbb{R}^{mp \times nq}$ defined blockwise as

$$
A \otimes B = \begin{pmatrix} a_{11}B & \ldots & a_{1n}B \\ \vdots & & \vdots \\ a_{m1}B & \ldots & a_{mn}B \end{pmatrix}.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric Meaning of the Kronecker Product)</span></p>

Let $f \colon U \to V$, $\tilde{f} \colon \tilde{U} \to \tilde{V}$ be two linear mappings between vector spaces. Consider the product space $U \times \tilde{U}$, where we require bilinearity-like properties for pairs $(u, \tilde{u})$:

$$
\alpha(u, \tilde{u}) = (\alpha u, \tilde{u}) = (u, \alpha \tilde{u}), \quad (u_1, \tilde{u}) + (u_2, \tilde{u}) = (u_1 + u_2, \tilde{u}), \quad (u, \tilde{u}_1) + (u, \tilde{u}_2) = (u, \tilde{u}_1 + \tilde{u}_2).
$$

For $u = \sum_i \alpha_i u_i \in U$ and $\tilde{u} = \sum_j \beta_j \tilde{u}_j \in \tilde{U}$, these properties imply $(u, \tilde{u}) = \sum_{i,j} \alpha_i \beta_j (u_i, \tilde{u}_j)$. The space $U \times \tilde{U}$ is then viewed as the set of expressions $\sum_{i,j} \alpha_i \beta_j (u_i, \tilde{u}_j)$.

The mapping $f \otimes \tilde{f}$ acts by $(u_j, \tilde{u}_\ell) \mapsto (f(u_j), \tilde{f}(\tilde{u}_\ell))$. If $A, B$ are the matrices of $f, \tilde{f}$ with respect to chosen bases, then the matrix of $f \otimes \tilde{f}$ is exactly $A \otimes B$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Properties of the Kronecker Product)</span></p>

For matrices of compatible dimensions:
1. $(A \otimes B)(C \otimes D) = AC \otimes BD$ (provided $AC, BD$ make sense),
2. $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$ if $A, B$ are regular,
3. $(A \otimes B)^T = A^T \otimes B^T$,
4. $(A \otimes B) \otimes C = A \otimes (B \otimes C)$,
5. In general $A \otimes B \neq B \otimes A$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (selected parts)</summary>

**(1)** Comparing blocks at position $i, j$: $\big(a_{i1}B \;\cdots\; a_{in}B\big) \begin{pmatrix} c_{1j}D \\ \vdots \\ c_{nj}D \end{pmatrix} = \big(\sum_{k=1}^n a_{ik}c_{kj}\big)BD = (AC)_{ij}BD$.

**(2)** From (1): $(A \otimes B)(A^{-1} \otimes B^{-1}) = AA^{-1} \otimes BB^{-1} = I_{mn}$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Eigenvalues of the Kronecker Product)</span></p>

Let $A \in \mathbb{R}^{m \times m}$ have eigenvalues $\lambda_1, \ldots, \lambda_m$ and $B \in \mathbb{R}^{n \times n}$ have eigenvalues $\mu_1, \ldots, \mu_n$. Then $A \otimes B$ has eigenvalues $\lambda_i \mu_j$ for all $i, j$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Convert $A, B$ (independently) to Jordan normal form via basis changes $S, \tilde{S}$, giving $S^{-1}AS \otimes \tilde{S}^{-1}B\tilde{S}$. By property (1), this equals $(S^{-1} \otimes \tilde{S}^{-1})(A \otimes B)(S \otimes \tilde{S})$, i.e., $A \otimes B$ is similar to $J_A \otimes J_B$. This is an upper triangular matrix with diagonal entries $\lambda_i \mu_j$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span></p>

Let $A \in \mathbb{R}^{m \times m}$, $B \in \mathbb{R}^{n \times n}$ with eigenvalues $\lambda_1, \ldots, \lambda_m$ and $\mu_1, \ldots, \mu_n$ respectively:
1. If $A, B$ are positive (semi-)definite, then $A \otimes B$ is positive (semi-)definite.
2. $\det(A \otimes B) = \det(A)^n \det(B)^m$.
3. The eigenvalues of $A \otimes B$ are the same as those of $B \otimes A$.
4. The eigenvalues of $A \otimes I_n + I_m \otimes B$ are $\lambda_i + \mu_j$ for all $i, j$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**(1)** Symmetry of $A \otimes B$ follows from property (3) of Theorem 10.3. Nonnegativity (resp. positivity) of eigenvalues $\lambda_i \mu_j$ follows since products of nonnegative (resp. positive) numbers are nonnegative (resp. positive).

**(2)** Clear from the eigenvalue formula.

**(3)** Clear from the eigenvalue formula.

**(4)** By a change of basis, $A \otimes I_n + I_m \otimes B$ becomes $J_A \otimes I_n + I_m \otimes J_B$. This is upper triangular with diagonal entries $\lambda_i + \mu_j$.

</details>
</div>

The matrix $A \otimes I_n + I_m \otimes B$ is called the **Kronecker sum** of $A$ and $B$.

#### Matrix Systems of Equations

The Kronecker product is a useful tool for working with matrix systems of linear equations. Consider the system $AX = B$, where $A, B$ are given matrices and $X$ is the unknown matrix. Using the $\operatorname{vec}(\cdot)$ operator, which stacks columns of a matrix into a single long vector:

$$
\operatorname{vec}(B) = (b_{11}, \ldots, b_{m1}, b_{12}, \ldots, b_{m2}, \ldots, b_{1n}, \ldots, b_{mn})^T,
$$

we can rewrite $AX = B$ as $(I \otimes A)\operatorname{vec}(X) = \operatorname{vec}(B)$.

Similarly, the system $AX + XC = B$ can be rewritten as $(I \otimes A + C^T \otimes I)\operatorname{vec}(X) = \operatorname{vec}(B)$, and $AXC = B$ becomes $(C^T \otimes A)\operatorname{vec}(X) = \operatorname{vec}(B)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

The system $AXC = B$ is equivalent to the system $(C^T \otimes A)\operatorname{vec}(X) = \operatorname{vec}(B)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The $(i,j)$-th equation in $AXC = B$ is $\sum_{k,\ell} a_{ik} x_{k\ell} c_{\ell j} = (AXC)_{ij} = b_{ij}$. On the other hand, the equations in the $j$-th block of $(C^T \otimes A)\operatorname{vec}(X) = \operatorname{vec}(B)$ have the form $(c_{1j}A \;\cdots\; c_{nj}A)\operatorname{vec}(X) = B_{*j}$, i.e., $\sum_\ell c_{\ell j} A X_{*\ell} = B_{*j}$. The $i$-th equation in this block is $\sum_{k,\ell} c_{\ell j} a_{ik} x_{k\ell} = b_{ij}$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Rewriting a matrix system in the classical form is useful for establishing properties of different systems. However, for actually solving the system, it may not be the best approach because the system size grows: if $AX = B$ has a square matrix of order $n$, then $(I \otimes A)\operatorname{vec}(X) = \operatorname{vec}(B)$ is a system of order $n^2$.

There also exist explicit solutions for matrix systems. For example, the Sylvester equation $AX - XB = C$ can be solved using the matrix exponential as $X = -\int_0^{\infty} e^{At} C e^{-Bt} dt$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Sylvester, 1884; Rosenblum, 1956)</span></p>

Let $A, B \in \mathbb{R}^{n \times n}$ have no common eigenvalue. Then the Sylvester equation $AX - XB = C$ has exactly one solution for any $C \in \mathbb{R}^{n \times n}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The equation can be rewritten as $(I_n \otimes A - B^T \otimes I_n)\operatorname{vec}(X) = \operatorname{vec}(C)$. By property (4) of Corollary 10.5, the eigenvalues of $I_n \otimes A - B^T \otimes I_n$ are $\lambda_i - \mu_j$ for all $i, j$. Since $A$ and $B$ have no common eigenvalue, all these are nonzero, so the matrix is regular and the solution is unique.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Lyapunov, 1892)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ have eigenvalues only in the open left half-plane (i.e., their real parts are negative). Then the Lyapunov equation $AX + XA^T = -I_n$ has exactly one solution. Moreover, the solution is symmetric and positive definite.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $A$ and $-A^T$ share no common eigenvalue (eigenvalues of $-A^T$ are the negatives of eigenvalues of $A$, which lie in the right half-plane), the Sylvester--Rosenblum theorem guarantees a unique solution.

By transposing the equation: $X^T A^T + AX^T = -I_n$, so $X^T$ is also a solution. By uniqueness, $X = X^T$, i.e., $X$ is symmetric.

For positive definiteness (in the special case when $A$ is symmetric): since $A$ is negative definite, let $\lambda$ be an eigenvalue of $X$ and $v$ the corresponding eigenvector. Then $0 > -v^T v = v^T(AX + XA^T)v = v^T AXv + v^T XA^Tv = 2v^T AXv = 2\lambda v^T Av$. Since $v^T Av < 0$ (by the Rayleigh--Ritz theorem), we get $\lambda > 0$.

</details>
</div>

The solvability of the Lyapunov equation is important for stability in control theory. A matrix whose eigenvalues all lie in the open left half-plane is called **stable**.

### Hadamard Product

The Hadamard product multiplies matrices entry-wise, as one might intuitively define matrix multiplication.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hadamard Product)</span></p>

Let $A, B \in \mathbb{R}^{m \times n}$. The **Hadamard product** (also called the Schur product) of $A$ and $B$ is the matrix $A \circ B \in \mathbb{R}^{m \times n}$ defined as $(A \circ B)_{ij} = a_{ij} b_{ij}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Schur, 1911)</span></p>

Let $A, B \in \mathbb{R}^{n \times n}$.
1. If $A, B$ are positive semidefinite, then $A \circ B$ is positive semidefinite.
2. If $A, B$ are positive definite, then $A \circ B$ is positive definite.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**(1)** Let $A = Q \Lambda Q^T$ be the spectral decomposition, so $a_{ij} = \sum_{k=1}^n q_{ik} \lambda_k q_{jk}$ with $\lambda_k \geq 0$. For any nonzero $x \neq 0$:

$$
x^T(A \circ B)x = \sum_{i,j=1}^n x_i x_j a_{ij} b_{ij} = \sum_{k=1}^n \lambda_k \sum_{i,j=1}^n b_{ij}(q_{ik}x_i)(q_{jk}x_j) = \sum_{k=1}^n \lambda_k (y^{(k)})^T B y^{(k)} \geq 0,
$$

where $y^{(k)} \coloneqq (q_{1k}x_1, \ldots, q_{nk}x_n)^T$.

**(2)** For positive definite $A, B$, the equality $x^T(A \circ B)x = 0$ holds only when $y^{(k)} = 0$ for all $k$ (since $B$ is positive definite). Since $0 = \sum_{i,k=1}^n (y_i^{(k)})^2 = \sum_{i,k=1}^n (q_{ik}x_i)^2 = \sum_{i=1}^n x_i^2 \sum_{k=1}^n q_{ik}^2 = \sum_{i=1}^n x_i^2$, we must have $x = 0$.

</details>
</div>

The Hadamard product has different properties from the standard matrix product. For instance:
- If $A, B$ are symmetric, then $A \circ B$ is symmetric.
- If $A, B$ are regular, $A \circ B$ may be singular: e.g., $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$, $B = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

Let $A, B \in \mathbb{R}^{n \times n}$. Then $\lVert A \circ B \rVert_2 \leq \lVert A \rVert_2 \cdot \lVert B \rVert_2$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For any square matrix $C$, $\lVert C \rVert_2^2 = \rho(C^TC) \leq \lVert C^TC \rVert_1 \leq \lVert C \rVert_1$. Setting $C \coloneqq A \circ B$, we get $\lVert A \circ B \rVert_2^2 \leq \lVert A^T \circ B^T \rVert_1 \cdot \lVert A \circ B \rVert_1$.

Now $\lVert A \circ B \rVert_1 = \max_j \sum_i |a_{ij} b_{ij}| = \max_j \langle |A_{*j}|, |B_{*j}| \rangle \leq \max_j \lVert A_{*j} \rVert_2 \cdot \lVert B_{*j} \rVert_2 \leq \lVert A \rVert_2 \cdot \lVert B \rVert_2$ by Cauchy--Schwarz and the fact that $\max_j \sqrt{e_j^T A^T A e_j} \leq \sqrt{\lambda_1(A^TA)} = \lVert A \rVert_2$.

Similarly $\lVert A^T \circ B^T \rVert_1 \leq \lVert A \rVert_2 \cdot \lVert B \rVert_2$, giving $\lVert A \circ B \rVert_2^2 \leq \lVert A \rVert_2^2 \cdot \lVert B \rVert_2^2$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(JPEG Compression)</span></p>

The Hadamard product is used in JPEG lossy image compression. During quantization, the image block (transformed by DCT) is Hadamard-multiplied by a quantization matrix and then the values are rounded. This performs part of the compression (rounding discards many values, achieving compression) and simultaneously introduces information loss --- according to the compression ratio. During decompression, the individual steps are performed in reverse order, and again the Hadamard product is used for multiplication with the quantization matrix.

</div>

## Positive Semidefiniteness

The set of positive semidefinite matrices has a specific geometric structure that we explore in this chapter. Denote by $\mathcal{S}_+$ the set of positive semidefinite matrices from $\mathbb{R}^{n \times n}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span></p>

Let $A \in \mathcal{S}_+$ and $x \in \mathbb{R}^n$. If $x^T A x = 0$, then $Ax = 0$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We have $0 = x_i^T \sqrt{A} \sqrt{A} x_i = \lVert \sqrt{A} x_i \rVert_2^2$, which implies $\sqrt{A} x_i = 0$ and hence $Ax = 0$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

The set $\mathcal{S}_+$ forms a convex cone in the space of symmetric matrices in $\mathbb{R}^{n \times n}$. The interior of $\mathcal{S}_+$ consists of positive definite matrices. The boundary of $\mathcal{S}_+$ consists of singular matrices. The extreme rays of the cone are formed by positive semidefinite matrices of rank $1$, i.e., matrices of the form $xx^T$ for $x \neq 0$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The set $\mathcal{S}_+$ is closed under addition and nonnegative scalar multiplication, so it forms a convex cone. Its interior consists of positive definite matrices (since these are exactly the ones with all eigenvalues strictly positive, forming an open set).

Every $A \in \mathcal{S}_+$ can be written via its spectral decomposition as $A = Q\Lambda Q^T = \sum_{i=1}^n \lambda_i q_i q_i^T$, where $q_i = Q_{*i}$. Thus every positive semidefinite matrix is a nonnegative combination of rank-$1$ matrices $xx^T$.

It remains to show that matrices of the form $xx^T$ are extreme. Suppose $xx^T = \alpha A + \beta B$ where $A, B \in \mathcal{S}_+$ and $\alpha, \beta > 0$. Without loss of generality let $\lVert x \rVert_2 = 1$ and extend to an orthonormal basis $x, x_2, \ldots, x_n$. For each $i = 2, \ldots, n$: $0 = x_i^T(xx^T)x_i = \alpha x_i^T A x_i + \beta x_i^T B x_i \geq 0$, so $x_i^T A x_i = x_i^T B x_i = 0$. By Lemma 11.1, $Ax_i = Bx_i = 0$. This means $A$ and $B$ have rank at most $1$ and $\operatorname{Ker}(A) = \operatorname{Ker}(B) = \operatorname{Ker}(xx^T)$, hence $A$ and $B$ must be positive multiples of $xx^T$.

</details>
</div>

If we extend the set $\mathcal{S}_+$ to a full space (by taking the intersection with a hyperplane), we obtain a geometric body called a **spectrahedron**.

### Löwner Ordering

Positive (semi-)definiteness allows us to define an interesting matrix relation.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Löwner Ordering)</span></p>

Let $A, B \in \mathbb{R}^{n \times n}$ be symmetric. We write $A \succeq B$ if $A - B$ is positive semidefinite, and $A \succ B$ if $A - B$ is positive definite.

</div>

Positive semidefiniteness of $A$ can be written as $A \succeq 0$. The relation $\succeq$ defines a partial ordering and $\succ$ a strict ordering on the class of symmetric matrices of order $n$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

The relation $\succeq$ defines a partial ordering and the relation $\succ$ a strict ordering on the class of symmetric matrices of order $n$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Reflexivity:* $A \succeq A$ trivially.

*Transitivity:* If $A \succeq B \succeq C$, then $A - B$ and $B - C$ are positive semidefinite, so their sum $A - C$ is also positive semidefinite.

*Anti-symmetry:* If $A \succeq B$ and $B \succeq A$, then $A - B$ has nonnegative eigenvalues and nonpositive eigenvalues simultaneously, so all eigenvalues are zero and $A - B = 0$.

Analogously for $\succ$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span></p>

Let $A, B, C, D \in \mathbb{R}^{n \times n}$ be symmetric. Then:
1. $A \succeq B \;\Rightarrow\; SAS^T \succeq SBS^T$ for every $S \in \mathbb{R}^{n \times n}$,
2. $A \succ B \;\Rightarrow\; SAS^T \succ SBS^T$ for every regular $S \in \mathbb{R}^{n \times n}$,
3. $A \succeq B,\; C \succeq D \;\Rightarrow\; A + C \succeq B + D$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

Let $A, B \in \mathbb{R}^{n \times n}$ be symmetric and $a = A \succ 0$. Then $A, B$ are simultaneously diagonalizable (in the sense of quadratic forms): there exists a regular $U$ such that $U^T A U$ and $U^T B U$ are both diagonal.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $V \coloneqq \sqrt{A}^{-1}$, so $V^T AV = I_n$. The matrix $V^T BV$ is symmetric and hence has a spectral decomposition $V^T BV = Q \Lambda Q^T$. Setting $U \coloneqq VQ$ gives $U^T AU = Q^T V^T A V Q = Q^T I_n Q = I_n$ and $U^T BU = Q^T \Lambda Q^T Q = \Lambda$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

Let $A \succ 0$, $B \succeq 0$. Then:
1. $A \succeq B$ if and only if $\rho(A^{-1}B) \leq 1$,
2. $A \succ B$ if and only if $\rho(A^{-1}B) < 1$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By Theorem 11.6, $A = UU^T$ and $B = UDU^T$ where $D$ is diagonal with nonnegative entries. Then $A \succeq B$ iff $U(I-D)U^T \succeq 0$, iff $I - D \succeq 0$, iff $d_{ii} \leq 1$. The eigenvalues of $A^{-1}B = U^{-T}DU^T$ are $d_{11}, \ldots, d_{nn}$, so $\rho(A^{-1}B) \leq 1$ iff $d_{ii} \leq 1$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

Let $A, B \succ 0$. Then $A \succeq B \;\Leftrightarrow\; A^{-1} \preceq B^{-1}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By Theorem 11.7, $A \succeq B$ iff $\rho(A^{-1}B) \leq 1$. Since $\rho(A^{-1}B) = \rho(AA^{-1}BA^{-1}) = \rho((B^{-1})^{-1}A^{-1})$, the same condition reads $B^{-1} \succeq A^{-1}$, i.e., $A^{-1} \preceq B^{-1}$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

Let $A, B \in \mathbb{R}^{n \times n}$ be symmetric and $A \succeq B$. Then $\lambda_i(A) \geq \lambda_i(B)$ for all $i = 1, \ldots, n$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By the Courant--Fischer theorem:

$$
\lambda_i(A) = \max_{V \in \mathbb{R}^n:\, \dim V = i} \min_{x \in V:\, \lVert x \rVert_2 = 1} x^T A x \geq \max_{V \in \mathbb{R}^n:\, \dim V = i} \min_{x \in V:\, \lVert x \rVert_2 = 1} x^T B x = \lambda_i(B).
$$

</details>
</div>

The converse does not hold in general. For example: $A = \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}$, $B = \begin{pmatrix} 0 & 0 \\ 0 & 2 \end{pmatrix}$ --- here $\lambda_1(A) \geq \lambda_1(B)$ and $\lambda_2(A) \geq \lambda_2(B)$, but $A - B$ is indefinite.

From the properties above we can derive further results. For instance, $A \succeq B \succeq 0$ implies $\det(A) \geq \det(B)$, $\operatorname{trace}(A) \geq \operatorname{trace}(B)$, etc.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hadamard Inequality)</span></p>

1. If $A \succ 0$, then $\det(A) \leq \prod_{i=1}^n a_{ii}$.
2. If $B \in \mathbb{R}^{n \times n}$, then $|\det(B)| \leq \prod_{i=1}^n \lVert B_{*i} \rVert_2$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**(1)** Define $D \coloneqq \operatorname{diag}(a_{11}^{-1/2}, \ldots, a_{nn}^{-1/2})$. The condition $\det(A) \leq \prod_{i=1}^n a_{ii}$ is equivalent to $\det(DAD) \leq 1$, since $\det(DAD) = \det(A) \det(D)^2 = \det(A) \prod_{i=1}^n a_{ii}^{-1}$. Let $\lambda_1, \ldots, \lambda_n$ be the eigenvalues of $DAD$. By the AM--GM inequality:

$$
\det(DAD) = \prod_{i=1}^n \lambda_i \leq \Big(\frac{1}{n}\sum_{i=1}^n \lambda_i\Big)^n = \Big(\frac{1}{n}\operatorname{trace}(DAD)\Big)^n = 1.
$$

**(2)** For singular $B$ the bound is trivial, so assume $B$ is regular. Define $A \coloneqq B^T B \succ 0$. Then $\det(A) \leq \prod_{i=1}^n a_{ii}$ gives $\det(B)^2 = \det(B^TB) \leq \prod_{i=1}^n \lVert B_{*i} \rVert_2^2$.

</details>
</div>

The geometric meaning of the second inequality: the determinant is the volume of the parallelepiped spanned by the columns of $B$, while the right-hand side is the volume of the rectangular box with sides matching the column norms. Equality holds when the columns of $B$ are mutually orthogonal.

## Special Matrices

We frequently encounter special types of matrices and exploit their specific properties. Types such as regular, orthogonal, positive definite, etc., are already well known; in this chapter we focus on another important class of matrices.

### M-Matrices

**Motivation.** Consider a square system of linear equations $Ax = b$. The $i$-th equation, solved for the $i$-th variable $x_i$, reads

$$
x_i = \frac{1}{a_{ii}} \Big( b_i - \sum_{j \neq i} a_{ij} x_j \Big), \quad i = 1, \ldots, n.
$$

This form leads to the **Jacobi iterative method** for solving systems. Starting from an arbitrary initial vector $x^0 \in \mathbb{R}^n$, at the $k$-th iteration we produce from $x^{k-1}$ the vector $x^k$:

$$
x_i^k \coloneqq \frac{1}{a_{ii}} \Big( b_i - \sum_{j \neq i} a_{ij} x_j^{k-1} \Big), \quad i = 1, \ldots, n.
$$

Under certain conditions, the sequence $x^0, x^1, \ldots$ converges to the solution $A^{-1}b$.

We can express this iteration in matrix form. Decompose $A = D + A'$ into a diagonal and an off-diagonal part:

$$
D = \begin{pmatrix} a_{11} & 0 & \ldots & 0 \\ 0 & \ddots & \ddots & \vdots \\ \vdots & \ddots & \ddots & 0 \\ 0 & \ldots & 0 & a_{nn} \end{pmatrix}, \quad A' = \begin{pmatrix} 0 & a_{12} & \ldots & a_{1n} \\ a_{21} & \ddots & \ddots & \vdots \\ \vdots & \ddots & \ddots & a_{n-1,n} \\ a_{n1} & \ldots & a_{n,n-1} & 0 \end{pmatrix}.
$$

Then the equation $Ax = b$ becomes $Dx + A'x = b$, i.e., $Dx = b - A'x$, and the $k$-th iteration is

$$
x^k \coloneqq D^{-1}(b - A'x^{k-1}) = -D^{-1}A'x^{k-1} + D^{-1}b.
$$

More generally, we can consider any decomposition $A = M - N$ where $M$ is easily invertible (diagonal, triangular, etc.). Then the iteration takes the form

$$
x^k \coloneqq M^{-1}Nx^{k-1} + M^{-1}b.
$$

A sufficient condition for convergence is given by the following theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span></p>

If $\rho(M^{-1}N) < 1$, then $A$ is regular and $x^k \to_{k \to \infty} A^{-1}b$ for every initial vector $x^0$ and for every $b$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

From $A = M - N$ we get $M^{-1}A = I_n - M^{-1}N$. Since $\rho(M^{-1}N) < 1$, the matrix $I_n - M^{-1}N$ has all eigenvalues with positive real part, hence is regular. Therefore $A$ is regular.

Let $x^* = A^{-1}b$ and consider

$$
x^1 - x^* = M^{-1}Nx^0 + M^{-1}b - x^* = M^{-1}Nx^0 + M^{-1}(A-M)x^* - x^* = M^{-1}N(x^0 - x^*).
$$

By induction, $\lVert x^k - x^* \rVert \leq \lVert M^{-1}N \rVert^k \cdot \lVert x^0 - x^* \rVert$ for any induced matrix norm. By Theorem 5.15, there exists an induced norm such that $\lVert M^{-1}N \rVert < 1$, so the distance $x^k$ from $x^*$ shrinks geometrically.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The converse also holds. If $\rho(M^{-1}N) \geq 1$, then $A$ need not be regular. Even if $A$ is regular, the sequence $x^0, x^1, \ldots$ typically does not converge for all $x^0$ and $b$. As a concrete example, let $M = N = I_n$; then $\rho(M^{-1}N) = 1$ and the sequence does not converge.

If $\rho(M^{-1}N)$ is close to $1$ from below, convergence is slow. The spectral radius only provides convergence for real eigenvalues $|\lambda| \geq 1$; for complex eigenvalues $\lambda \pm i\nu$ the situation is more involved (via the real Schur decomposition and a careful analysis of the $2 \times 2$ blocks).

</div>

**M-matrices defined.** M-matrices represent a class of matrices for which the Jacobi method converges, and for which every splitting $A = M - N$ with $M$ regular and $N$ nonnegative achieves convergence. The significance of M-matrices goes far beyond this, however.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(M-Matrix)</span></p>

A matrix $A \in \mathbb{R}^{n \times n}$ is an **M-matrix** if $a_{ij} \leq 0$ for $i \neq j$ and $A^{-1} \geq 0$.

</div>

The name originates from Alexander Ostrowski (1937) and is meant to honor Hermann Minkowski, who first showed the initial properties. M-matrices can be characterized by many equivalent conditions.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Equivalent Conditions for M-Matrices)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ with $a_{ij} \leq 0$ for $i \neq j$. The following conditions are equivalent:
1. $A$ is an M-matrix, i.e., $A^{-1} \geq 0$,
2. $A = \alpha I_n - B$ where $B \geq 0$ and $\rho(B) < \alpha$,
3. $A = M - N$ where $M^{-1} \geq 0$, $N \geq 0$, and $\rho(M^{-1}N) < 1$,
4. the real parts of all eigenvalues of $A$ are positive,
5. all real eigenvalues of $A$ are positive,
6. there exists an LU decomposition $A = LU$ where $L, U$ are M-matrices,
7. there exists $x > 0$ such that $Ax > 0$,
8. $A^{-1}e > 0$,
9. if $Ax \geq 0$, then $x \geq 0$,
10. all leading principal submatrices of $A$ have positive determinant.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (selected implications)</summary>

**(1) $\Rightarrow$ (2):** Define $\alpha \coloneqq \max_i |a|_{ii}$ and $B \coloneqq \alpha I_n - A \geq 0$. Set $x \coloneqq A^{-1}e > 0$. Then $(\alpha I_n - B)x = Ax = e > 0$, so $Bx < \alpha x$. By Perron theory (Theorem 8.6), $\rho(B) < \alpha$.

**(2) $\Rightarrow$ (1):** By the Neumann series, $A^{-1} = (\alpha I_n - B)^{-1} = \frac{1}{\alpha}(I_n - \frac{1}{\alpha}B)^{-1} = \frac{1}{\alpha}\sum_{k=0}^{\infty}(\frac{1}{\alpha}B)^k \geq 0$.

**(1) $\Rightarrow$ (3):** Follows from property (2) by setting $M \coloneqq \alpha I_n$, $N \coloneqq B$.

**(3) $\Rightarrow$ (1):** By the Neumann series, $A^{-1} = (M-N)^{-1} = (I_n - M^{-1}N)^{-1}M^{-1} = \big(\sum_{k=0}^{\infty}(M^{-1}N)^k\big)M^{-1} \geq 0$.

**(2) $\Rightarrow$ (4):** Eigenvalues of $A$ are essentially eigenvalues of $-B$ shifted by $\alpha$, so they lie to the right of the origin.

**(4) $\Rightarrow$ (2):** Define $\alpha \coloneqq \max_i |a|_{ii}$ and $B \coloneqq \alpha I_n - A \geq 0$. The matrix $B$ has nonnegative off-diagonal entries with real parts of eigenvalues less than $\alpha$. By Perron theory, $\rho(B) < \alpha$.

**(1) $\Rightarrow$ (6):** During Gaussian elimination, no row permutations are needed --- it suffices to add positive multiples of rows. M-matrix structure is preserved for both $L$ and $U$.

**(6) $\Rightarrow$ (1):** $A^{-1} = U^{-1}L^{-1} \geq 0$.

**(1) $\Rightarrow$ (7):** Set $x \coloneqq A^{-1}e > 0$.

**(7) $\Rightarrow$ (2):** Define $\alpha \coloneqq \max_i |a|_{ii}$ and $B \coloneqq \alpha I_n - A \geq 0$. Since $Ax > 0$, we have $Bx < \alpha x$. By Perron theory, $\rho(B) < \alpha$.

**(1) $\Rightarrow$ (8):** Clear from nonnegativity of $A^{-1}$.

**(8) $\Rightarrow$ (7):** Set $x \coloneqq A^{-1}e > 0$.

**(1) $\Rightarrow$ (9):** If $Ax \geq 0$, then $x = A^{-1}Ax \geq 0$ since $A^{-1} \geq 0$.

**(9) $\Rightarrow$ (1):** $A$ must be regular (otherwise $Ax = 0$ for some $x \neq 0$, and one of $\pm x$ is not nonnegative). For each $i = 1, \ldots, n$, define $x^i \coloneqq A^{-1}e_i$. Since $Ax^i = e_i \geq 0$, property (9) gives $x^i \geq 0$. The columns of $A^{-1}$ are thus nonnegative.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span></p>

Symmetric M-matrices are positive definite.

</div>

Regarding the location of eigenvalues of M-matrices, one can show even stronger results. For $n > 2$, the angle at the origin between the real axis and the line to any eigenvalue is less than $\frac{\pi}{2} - \frac{\pi}{n}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Laplacian Matrix of a Graph)</span></p>

The **Laplacian matrix** of a graph $G = (V, E)$ with $n$ vertices is $L \in \mathbb{R}^{n \times n}$ defined as

$$
L_{ij} = \begin{cases} \deg(i), & \text{if } i = j, \\ -1, & \text{if } (i,j) \in E, \\ 0, & \text{if } (i,j) \notin E. \end{cases}
$$

The matrix $L$ fully describes the graph $G$ and many graph properties can be easily determined from it. For example, the number of spanning trees of $G$ equals the determinant of $L$ after deleting any one row and column.

The Laplacian matrix is not an M-matrix in the strict sense (it is called a **singular M-matrix**), but it has nonnegative off-diagonal entries and satisfies the equality condition $\rho(B) = \alpha$ in property (2) of Theorem 12.4. Thus it shares many properties with M-matrices.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Linear Complementarity Problem)</span></p>

The **linear complementarity problem** is the feasibility problem for the system

$$
y = Ax + b, \quad x, y \geq 0, \quad x^T y = 0,
$$

where $A \in \mathbb{R}^{n \times n}$ and $b \in \mathbb{R}^n$ are given and $x, y \in \mathbb{R}^n$ are the unknowns. The constraints are linear except for the complementarity condition $x^T y = 0$, which equivalently states that for each $i$, either $x_i = 0$ or $y_i = 0$.

This problem is NP-hard in general and arises in optimality conditions (quadratic programming, integer programming feasibility) and in expressing Nash equilibria of bimatrix games.

If $A$ is an M-matrix, the linear complementarity problem has exactly one solution for every $b \in \mathbb{R}^n$ (in fact, a larger class of matrices called P-matrices satisfies this property). Moreover, the solution can be found efficiently.

</div>

## Further Topics

This chapter briefly surveys several additional topics in linear algebra.

### Applications of the SVD

The SVD decomposition has numerous applications, including:
- **Principal component analysis** in statistics and data science,
- **Latent semantic analysis** for text mining and information retrieval,
- **Image compression** and low-rank approximations,
- **Recommender systems** (e.g., the Netflix Prize challenge, which offered \$1M for a 10% improvement in recommendation quality).

### Numerical Range

The **numerical range** (also called field of values) of a matrix $A \in \mathbb{C}^{n \times n}$ is the set of complex numbers

$$
F(A) \coloneqq \lbrace x^* A x ;\; x \in \mathbb{C}^n,\; x^* x = 1 \rbrace.
$$

Since this set contains the eigenvalues of $A$, it is useful for estimating eigenvalues and bounding the spectral radius $\rho(A)$. For symmetric (resp. normal) matrices, $F(A)$ has a specific, well-characterized shape.

### Generalized Eigenvalues

Given matrices $A, B \in \mathbb{R}^{n \times n}$, a scalar $\lambda \in \mathbb{C}$ is a **generalized eigenvalue** of the pair $(A, B)$ if there exists $0 \neq x \in \mathbb{C}^n$ such that $Ax = \lambda Bx$. For $B = I_n$ this reduces to the standard eigenvalue problem.

### Stable Matrices

A matrix $A \in \mathbb{R}^{n \times n}$ is **(Hurwitz) stable** if the real parts of all its eigenvalues are negative. The motivation comes from convergence of solutions of differential equations: consider

$$
x' = Ax.
$$

The solution is $x(t) = e^{At}x(0)$. The state $x(t)$ converges to the equilibrium (or stays bounded near it) precisely when $A$ is stable.

An equivalent characterization of stability is given by the Lyapunov theorem: $A$ is stable if and only if the Lyapunov equation $XA + A^TX = -I$ has a positive definite solution $X$.

### Iterative Methods for Solving Linear Systems

To solve $Ax = b$, decompose $A = M - N$ where $M$ is easily invertible. The iteration

$$
x^{k+1} \coloneqq M^{-1}(Nx^k + b)
$$

converges to the solution under the condition $\rho(M^{-1}N) < 1$. The choice of $M$ determines the specific method:
- $M = D$ (diagonal part of $A$): **Jacobi method**,
- $M = D + L$ (lower triangular part): **Gauss--Seidel method**,
- $M = \frac{1}{\omega}(D + \omega L)$: **SOR (Successive Over-Relaxation) method**.

### Normal Matrices

A matrix is **normal** if its Schur decomposition (Theorem 2.1) has a diagonal matrix. Equivalently, $A$ is normal if and only if $A^T A = AA^T$. Normal matrices have many nice properties --- similar to symmetric matrices but forming a larger class.

### Total Least Squares

The **ordinary least squares** method seeks an approximate solution to the (typically overdetermined) system $Ax = b$ by minimizing $\lVert Ax - b \rVert_2$ over $x \in \mathbb{R}^n$.

The **total least squares** method generalizes this by allowing perturbations in both $A$ and $b$. It solves

$$
\min \lbrace \lVert (A' \mid b') \rVert ;\; (A + A')x = b + b',\; x \in \mathbb{R}^n \rbrace.
$$

In other words, we seek the smallest perturbation (in the Frobenius or spectral norm) of the augmented matrix $(A \mid b)$ that makes the system solvable.
