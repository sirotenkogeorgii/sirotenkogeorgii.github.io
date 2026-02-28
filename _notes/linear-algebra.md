---
layout: default
title: Linear Algebra
date: 2024-11-01
# excerpt: Connecting differential equations, stability analysis, and attractor theory to the training dynamics of modern machine learning models.
# tags:
#   - dynamical-systems
#   - machine-learning
#   - theory
---

# Linear Algebra

<figure>
  <img src="{{ 'assets/images/notes/linear-algebra/hierarchy_of_vector_spaces.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Hierarchy of vector spaces</figcaption>
</figure>

## Vector Space

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Field)</span></p>

A **field** is a set $\mathbb{T}$ with two **commutative** binary operations $+$ and $\cdot$ such that:

1. $(\mathbb{T},+)$ is an **abelian group** (neutral element is $0$; inverse of $a$ is $-a$).
2. $(\mathbb{T}\setminus{0},\cdot)$ is an **abelian group** (neutral element is $1$; inverse of $a\neq 0$ is $a^{-1}$).
3. **Distributivity:** for all $a,b,c\in \mathbb{T}$,

   $$a\cdot(b+c)=a\cdot b+a\cdot c$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vector Space)</span></p>

Let $\mathbb{T}$ be a **field** with neutral elements $0$ (for addition) and $1$ (for multiplication).

A **vector space over $\mathbb{T}$** is a set $V$ equipped with

* **vector addition** $+\colon V\times V \to V$,
* **scalar multiplication** $\cdot\colon \mathbb{T}\times V \to V$,

such that for all $\alpha,\beta \in \mathbb{T}$ and all $u,v\in V$:

1. $(V,+)$ is an **abelian group** (there is a zero vector $o\in V$, every $v\in V$ has an additive inverse $-v$, and addition is associative and commutative).
2. $\alpha(\beta v) = (\alpha\beta)v$  (associativity of scalar multiplication).
3. $1v = v$  (multiplicative identity acts trivially).
4. $(\alpha+\beta)v = \alpha v + \beta v$  (distributivity over scalar addition).
5. $\alpha(u+v) = \alpha u + \alpha v$  (distributivity over vector addition).

Elements of $V$ are called **vectors**, and elements of $\mathbb{T}$ are called **scalars**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Examples of vector spaces)</span></p>

### 1) Arithmetic space $\mathbb{R}^n$ (more generally $\mathbb{T}^n$ over a field $\mathbb{T}$)

* Vectors are $n$-tuples $(x_1,\dots,x_n)$.
* Addition and scalar multiplication are **componentwise**:
  
  $$(x_1,\dots,x_n)+(y_1,\dots,y_n)=(x_1+y_1,\dots,x_n+y_n)$$
  
  $$\alpha(x_1,\dots,x_n)=(\alpha x_1,\dots,\alpha x_n)$$
  
### 2) Matrices $\mathbb{R}^{m\times n}$ (more generally $\mathbb{T}^{m\times n}$)

* Vectors are matrices.
* Addition and scalar multiplication are **entrywise**:
  
  $$(A+B)_{ij}=A_{ij}+B_{ij}, \quad (\alpha A)_{ij}=\alpha A_{ij}$$

### 3) All real polynomials in one variable $x$ (denoted $\mathcal P$)

* Vector space over $\mathbb{R}$.
* Addition and scalar multiplication are the standard polynomial operations.

### 4) Polynomials of degree at most $n$: $\mathcal P^n$

For

$$p(x)=a_nx^n+a_{n-1}x^{n-1}+\cdots+a_1x+a_0,\quad
q(x)=b_nx^n+b_{n-1}x^{n-1}+\cdots+b_1x+b_0,$$

* Addition:
  
  $$p(x)+q(x)=(a_n+b_n)x^n+(a_{n-1}+b_{n-1})x^{n-1}+\cdots+(a_0+b_0)$$
  
* Scalar multiplication $(\alpha\in\mathbb{R})$:
  
  $$\alpha p(x)=(\alpha a_n)x^n+(\alpha a_{n-1})x^{n-1}+\cdots+(\alpha a_0)$$
  
* Zero vector is the zero polynomial $0$.
* Additive inverse:
  
  $$-p(x)=(-a_n)x^n+(-a_{n-1})x^{n-1}+\cdots+(-a_0)$$

### 5) All real functions $f:\mathbb{R}\to\mathbb{R}$ (denoted $\mathcal F$)

* Addition is pointwise:
  
  $$(f+g)(x)=f(x)+g(x)$$
  
* Scalar multiplication is pointwise:
  
  $$(\alpha f)(x)=\alpha f(x)$$

### 6) All continuous real functions $f:\mathbb{R}\to\mathbb{R}$ (denoted $\mathcal C$)

* Operations are defined like for $\mathcal F$.

### 7) All continuous real functions on interval $f:[a,b]\to\mathbb{R}$ (denoted $\mathcal C_{[a,b]}$)

* Operations are defined like for $\mathcal F$.
  
</div>

<figure>
  <img src="{{ '/assets/images/notes/linear-algebra/real_functions_space.png' | relative_url }}" alt="a" loading="lazy">
</figure>

## Inner Product Spaces

### 1. Inner Products

#### Real Inner Products

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Real inner product)</span></p>

Let $V$ be a real vector space. A **real inner product** is a function

$$\langle \cdot , \cdot \rangle : V \times V \to \mathbb{R}$$

satisfying:

1. **Symmetry**:
   
   $$\langle v, w \rangle = \langle w, v \rangle$$
   
2. **Bilinearity**:
   
   $$\langle cx + y, z \rangle = c \langle x, z \rangle + \langle y, z \rangle$$
   
3. **Positive definiteness**:
   
   $$\langle v, v \rangle \ge 0, \quad \langle v, v \rangle = 0 \iff v = 0$$

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\mathbb{R}^n$)</span></p>

For vectors

$$v = \sum_{i=1}^n a_i e_i, \quad w = \sum_{i=1}^n b_i e_i$$

define

$$\langle v, w \rangle = \sum_{i=1}^n a_i b_i.$$

This is symmetric, bilinear, and positive definite.

</div>

---

#### Complex Inner Products

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Complex inner product)</span></p>

Let $V$ be a complex vector space. A **complex inner product** is a function

$$\langle \cdot , \cdot \rangle : V \times V \to \mathbb{C}$$

satisfying:

1. **Conjugate symmetry**:
   
   $$\langle v, w \rangle = \overline{\langle w, v \rangle}$$
   
2. **Sesquilinearity**:
   
   $$\langle cx + y, v \rangle = c \langle x, v \rangle + \langle y, v \rangle$$
   
   
   $$\langle v, cx \rangle = \overline{c}\langle v, x \rangle$$
   
3. **Positive definiteness**:
   
   $$\langle v, v \rangle \in \mathbb{R}_{\ge 0}, \quad \langle v, v \rangle = 0 \iff v = 0$$
   

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\mathbb{C}^n$)</span></p>

For

$$w = (w_1,\dots,w_n), \quad z = (z_1,\dots,z_n)$$

define

$$\langle w, z \rangle = \sum_{i=1}^n w_i \overline{z_i}.$$

This defines a complex inner product.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why sesquilinear?)</span></p>

If bilinearity were used over $\mathbb{C}$, then

$$\langle ie, ie \rangle = -\langle e, e \rangle,$$

violating positivity. Conjugate symmetry ensures

$$\langle v, v \rangle \in \mathbb{R}.$$

</div>

#### Inner Product Spaces

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Inner product space)</span></p>

An **inner product space** is either:
* a real vector space with a real inner product, or
* a complex vector space with a complex inner product.

</div>

### 2. Norms and Metric Structure

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Norm)</span></p>

For $v \in V$, define the **norm**

$$\lvert v\rvert = \sqrt{\langle v, v \rangle}.$$


</div>

<div class="math-callout math-callout--lemma" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Cauchy–Schwarz)</span></p>

For all $v, w \in V$,

$$\lvert \langle v, w \rangle\rvert \le \lvert v \rvert \lvert w\rvert.$$

Equality holds iff $v$ and $w$ are linearly dependent.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Triangle inequality)</span></p>

For all $v, w \in V$,

$$\lvert v + w\rvert \le \lvert v\rvert + \lvert w\rvert.$$

Equality holds iff $v$, $w$ are linearly dependent and point in the same direction.

</div>

### 3. Orthogonality

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Orthogonality)</span></p>

Two nonzero vectors $v, w \in V$ are **orthogonal** if

$$\langle v, w \rangle = 0.$$

</div>

<div class="math-callout math-callout--lemma" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Orthogonal vectors are independent)</span></p>

If $v_1,\dots,v_n$ are pairwise orthogonal with $\lvert v_i\rvert \neq 0$, then they are linearly independent.

</div>

#### Orthonormal Bases

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Orthonormal basis)</span></p>

A basis $e_1,\dots,e_n$ is **orthonormal** if

$$\langle e_i, e_j \rangle = \delta_{ij}.$$


</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Gram–Schmidt)</span></p>

Every finite-dimensional inner product space admits an orthonormal basis.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Uniqueness of dot product form)</span></p>

If $\lbrace e_i \rbrace$ is orthonormal, then

$$\left\langle \sum a_i e_i, \sum b_i e_i \right\rangle = \sum a_i b_i.$$

Thus all finite-dimensional inner products look like the dot product in an orthonormal basis.

</div>


### 4. Hilbert Spaces

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hilbert space)</span></p>

A **Hilbert space** is an inner product space whose induced metric is complete.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Convergence criterion)</span></p>

Let $\lbrace e_i\rbrace$ be orthonormal and define

$$v_n = \sum_{i=1}^n c_i e_i.$$

Then $(v_n)$ converges iff

$$\sum_{i=1}^\infty \lvert c_i\rvert^2 < \infty.$$


</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hilbert basis)</span></p>

An **orthonormal basis** of a Hilbert space is a (possibly infinite) sequence
$\lbrace e_i \rbrace$ such that every $v \in V$ has a unique expansion

$$v = \sum_i c_i e_i, \quad \sum_i \lvert c_i\rvert^2 < \infty.$$


</div>

### 5. General Normed Vector Spaces

Inner product spaces give rise to norms, but **not every norm comes from an inner product**. This motivates the following generalization.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Normed vector space)</span></p>

A **normed vector space** is a vector space $V$ over $\mathbb{R}$ or $\mathbb{C}$ equipped with a function

$$\lvert \cdot \rvert : V \to \mathbb{R}$$

satisfying:

1. **Positivity**:
   
   $$\lvert v\rvert \ge 0, \quad \lvert v\rvert = 0 \iff v = 0$$
   
2. **Homogeneity**:
   
   $$\lvert \lambda v\rvert = \lvert \lambda\rvert , \lvert v\rvert \quad \text{for all scalars } \lambda$$

3. **Triangle inequality**:
   
   $$\lvert v + w\rvert \le \lvert v|\rvert + \lvert w\rvert$$
   
</div>

### Norms Induced by Inner Products

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Inner product induces a norm)</span></p>

If $(V, \langle \cdot, \cdot \rangle)$ is an inner product space, then

$$\lvert v\rvert := \sqrt{\langle v, v \rangle}$$

defines a norm on $V$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(One-way implication)</span></p>

Every inner product space is a normed vector space, but the converse is false:
there exist norms which do **not** arise from any inner product.

</div>

### Example of a Non–Inner-Product Norm

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\ell^1$-norm on $\mathbb{R}^n$)</span></p>

For $x = (x_1,\dots,x_n) \in \mathbb{R}^n$, define

$$\lvert x\rvert_1 := \sum_{i=1}^n \lvert x_i\rvert.$$

This satisfies all norm axioms, hence makes $\mathbb{R}^n$ a normed vector space.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Not every norm comes from an inner product)</span></p>

For $n \ge 2$, the norm

$$\lvert (x_1,\dots,x_n)\rvert = \sum_{i=1}^n \lvert x_i\rvert$$

on $\mathbb{R}^n$ does **not** arise from any inner product.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Parallelogram law)</span></p>

A norm $\lvert \cdot\rvert$ arises from an inner product $\iff$ it satisfies the
parallelogram identity:

$$\lvert v+w\rvert^2 + \lvert v-w\rvert^2 = 2\lvert v\rvert^2 + 2\lvert w\rvert^2.$$

The $\ell^1$-norm violates this identity.

</div>

## 6. Banach Spaces

Completeness is a purely **metric** notion and does not require an inner product.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Banach space)</span></p>

A **Banach space** is a normed vector space $(V, \lvert \cdot\rvert)$ such that the induced metric

$$d(v,w) := \lvert v-w\rvert$$

is complete.

</div>

### Relationship Between Hilbert and Banach Spaces

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hilbert vs Banach)</span></p>

Every Hilbert space is a Banach space, but not every Banach space is a Hilbert space.
The distinction is whether the norm arises from an inner product.

</div>

### Finite-Dimensional Case

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Finite-dimensional spaces are complete)</span></p>

Every finite-dimensional normed vector space is a Banach space.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Consequence)</span></p>

In finite dimensions, completeness does **not** depend on the choice of norm.
All norms on a finite-dimensional vector space are equivalent.

</div>

### Example: A Classic Banach Space

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\ell^2$ space)</span></p>

Let

$$\ell^2 = \lbrace (a_1,a_2,\dots) \mid \sum_{i=1}^\infty a_i^2 < \infty \rbrace$$

with norm

$$\lvert a \rvert = \left( \sum_{i=1}^\infty a_i^2 \right)^{1/2}.$$

Then $\ell^2$ is a Hilbert space, hence also a Banach space.

</div>

### Metric Embedding Theorem

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Kuratowski embedding)</span></p>

For any metric space $(M,d)$, there exists a Banach space $X$ and an injective map

$$f : M \hookrightarrow X$$

such that

$$d(x,y) = \lvert f(x) - f(y)\rvert  \quad \text{for all } x,y \in M.$$

</div>


## Dual Space and Trace

## 1. Big picture

This chapter answers a natural question:

> A matrix depends on a basis, so why does the **trace** (sum of diagonal entries) make sense independently of basis?

The answer is to define trace **intrinsically**, without coordinates. To do that, the chapter introduces two major constructions:

* the **tensor product** $V \otimes W$
* the **dual space** $V^\vee$

Then it shows that for finite-dimensional spaces,

$$V^\vee \otimes W \cong \mathrm{Hom}(V,W)$$

so linear maps can be viewed as tensors. In particular, for $T:V\to V$, the trace comes from the natural **evaluation map** $V^\vee \otimes V \to k$. 

---

## 2. Tensor product $V \otimes W$

## 2.1 Motivation

We already know:

$$\dim(V \oplus W)=\dim V+\dim W$$

But sometimes we want a real “product” of vector spaces whose dimension multiplies:

$$\dim(V \otimes W)=\dim V \cdot \dim W$$

The motivating example in the chapter is that if

* $V$ is the space of quadratic polynomials in $x$,
* $W$ is the space of linear polynomials in $y$,

then instead of storing pairs $(f(x),g(y))$, we want a space of products like

$$4x^2y+5xy+y+3$$

which behaves like a 2-variable polynomial space. This is the role of the tensor product. 

---

## 2.2 Definition

For vector spaces $V,W$ over the same field $k$, the **tensor product** $V\otimes_k W$ is generated by formal symbols $v\otimes w$, subject to the relations

$$(v_1+v_2)\otimes w = v_1\otimes w + v_2\otimes w$$

$$v\otimes (w_1+w_2)=v\otimes w_1 + v\otimes w_2$$

$$(c v)\otimes w = v\otimes (c w)$$

Scalars can move through the tensor wall:

$$c(v\otimes w)=(cv)\otimes w=v\otimes (cw)$$

An element of the form $v\otimes w$ is called a **pure tensor**. General elements of $V\otimes W$ are **sums** of pure tensors. 

---

## 2.3 Important warning

Not every element of $V\otimes W$ is a single pure tensor.

That is, many tensors are sums like

$$\xi_1\otimes w_1+\xi_2\otimes w_2+\cdots$$

and cannot be factored into one $v\otimes w$.

The chapter emphasizes this with the example

$$\mathbb R[x]\otimes_{\mathbb R}\mathbb R[y]=\mathbb R[x,y]$$

A two-variable polynomial does **not** always factor as $f(x)g(y)$, but it can always be written as a sum of monomials $x^a\otimes y^b$. 

---

## 2.4 Basis of a tensor product

If

* $e_1,\dots,e_m$ is a basis of $V$,
* $f_1,\dots,f_n$ is a basis of $W$,

then a basis of $V\otimes W$ is

$$e_i\otimes f_j \quad (1\le i\le m,;1\le j\le n)$$

Therefore,

$$\dim(V\otimes W)= (\dim V)(\dim W)$$

So the tensor product really behaves like a multiplicative version of combining spaces. 

---

## 2.5 Concrete computation

If $V$ has basis $e_1,e_2$, $W$ has basis $f_1,f_2$, and

$$v=3e_1+4e_2,\qquad w=5f_1+6f_2,$$

then

$$v\otimes w=(3e_1+4e_2)\otimes(5f_1+6f_2)$$

expands by bilinearity to

$$15(e_1\otimes f_1)+20(e_2\otimes f_1)+18(e_1\otimes f_2)+24(e_2\otimes f_2)$$

This is the tensor-product analogue of multiplying out brackets. 

---

## 2.6 Key intuition

Think of $\otimes$ as a **wall**:

* it keeps the two vector spaces conceptually separate,
* but scalars can pass through.

Also, $V\otimes W$ depends only on the spaces $V$ and $W$, not on any relationship between them. Even if $V=W$, in general

$$v\otimes 1 \ne 1\otimes v$$

So tensor product is content-agnostic: it is a formal construction, not identification. 

---

# 3. Dual space $V^\vee$

## 3.1 Definition

For a vector space $V$ over $k$, the **dual space**

$$V^\vee$$

is the vector space of all linear maps

$$V \to k$$

Its elements are called **linear functionals**. Addition and scalar multiplication are pointwise. 

---

## 3.2 Concrete interpretation

If $V=\mathbb R^3$, vectors can be viewed as column vectors:

$$v=\begin{bmatrix}2\\5\\9\end{bmatrix}$$

A linear functional $f:V\to \mathbb R$ can be viewed as a row vector:

$$f=\begin{bmatrix}3&4&5\end{bmatrix}$$

Then

$$
f(v)=
\begin{bmatrix}3&4&5\end{bmatrix}
\begin{bmatrix}2\5\9\end{bmatrix}
=71
$$

So dual vectors act on vectors by “row times column.” 

---

## 3.3 Dual basis

If $V$ has basis $e_1,\dots,e_n$, then define $e_1^\vee,\dots,e_n^\vee$ by

$$
e_i^\vee(e_j)=
\begin{cases}
1,& i=j,\\
0,& i\ne j
\end{cases}
$$

These are the **dual basis vectors**.

Interpretation: $e_i^\vee(v)$ picks out the coefficient of $e_i$ when $v$ is written in the basis $e_1,\dots,e_n$. The set

$$e_1^\vee,\dots,e_n^\vee$$

forms a basis of (V^\vee). Therefore,

$$\dim V^\vee = \dim V$$

for finite-dimensional $V$. 

---

## 3.4 Example

If

$$f=3e_1^\vee+4e_2^\vee+5e_3^\vee$$

then for example

$$f(e_1)=3e_1^\vee(e_1)+4e_2^\vee(e_1)+5e_3^\vee(e_1)=3$$

So the notation behaves exactly as expected: the coefficients of the dual-basis expansion tell you the images of basis vectors. 

---

## 3.5 Is $V \cong V^\vee$?

Yes, for finite-dimensional spaces they are isomorphic because they have the same dimension.

But the chapter makes a very important point:

> The isomorphism $V\to V^\vee$ is **not natural** in general.

Why not? Because the identification

$$e_i \mapsto e_i^\vee$$

depends on the chosen basis.

If two people choose different bases of the same vector space, they generally get **different** identifications $V\to V^\vee$. So although $V\cong V^\vee$, there is usually no canonical way to say a vector *is* a covector unless extra structure is given. 

This is one of the most important conceptual points in the chapter.

---

# 4. Why $V^\vee \otimes W$ represents linear maps $V\to W$

## 4.1 Goal

For finite-dimensional vector spaces $V$ and $W$,

$$V^\vee\otimes W$$

can be interpreted as the space of linear maps from $V$ to $W$. In other words,

$$V^\vee\otimes W \cong \mathrm{Hom}(V,W)$$

This is the central structural result of the chapter. 

---

## 4.2 Intuition

Suppose $\dim V=3$ and $\dim W=5$. Then a linear map $V\to W$ is like a $5\times 3$ matrix, so the space of such maps should have dimension (15).

Meanwhile,

$$\dim(V^\vee\otimes W)= (\dim V^\vee)(\dim W)=3\cdot 5=15$$

So the dimensions match, and the tensor product is the right candidate. 

---

## 4.3 How a tensor gives a linear map

Take a tensor such as

$$\xi_1\otimes w_1+\cdots+\xi_m\otimes w_m \in V^\vee\otimes W$$

Each $\xi_i$ is a functional $V\to k$, so if you plug in $v\in V$, each $\xi_i(v)$ is a scalar. Hence define

$$v \mapsto \xi_1(v)w_1+\cdots+\xi_m(v)w_m$$

This gives an element of $W$, and the map is linear. So every tensor naturally defines a linear map $V\to W$. 

---

## 4.4 The theorem

Define

$$\Psi: V^\vee\otimes W \to \mathrm{Hom}(V,W)$$

by

$$\Psi(\xi_1\otimes w_1+\cdots+\xi_m\otimes w_m)(v) = \xi_1(v)w_1+\cdots+\xi_m(v)w_m$$

Then $\Psi$ is an **isomorphism of vector spaces**. So every linear map $V\to W$ can be represented uniquely as an element of $V^\vee\otimes W$. 

---

## 4.5 Proof idea

The proof in the chapter has two steps.

### Step 1: Surjectivity

Take any $T:V\to W$. If $e_1,\dots,e_n$ is a basis of $V$, write

$$T(e_i)=w_i$$

Then the tensor

$$e_1^\vee\otimes w_1+\cdots+e_n^\vee\otimes w_n$$

maps under $\Psi$ to $T$. So every linear map has at least one representation. 

### Step 2: Dimension count

We know

$$\dim(V^\vee\otimes W)=\dim V \cdot \dim W$$

Also $\mathrm{Hom}(V,W)$ has the same dimension, because linear maps $V\to W$ are represented by $(\dim W)\times(\dim V)$ matrices. Since $\Psi$ is surjective between spaces of equal dimension, it is an isomorphism. 

---

## 4.6 Concrete matrix example

If $V=\mathbb R^2$ with basis $e_1,e_2$, and

$$
T=
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix},
$$

then the corresponding tensor is

$$
e_1^\vee\otimes e_1
+2e_2^\vee\otimes e_1
+3e_1^\vee\otimes e_2
+4e_2^\vee\otimes e_2.
$$

The nice part is that although this expression changes with basis, the underlying tensor in $V^\vee\otimes V$ is basis-independent. 

---

# 5. Trace

## 5.1 Coordinate-free definition

For a finite-dimensional vector space $V$, a linear map $T:V\to V$ is an element of

$$\mathrm{Hom}(V,V)\cong V^\vee\otimes V$$

There is also a natural **evaluation map**

$$\mathrm{ev}: V^\vee\otimes V \to k$$

defined on pure tensors by

$$f\otimes v \mapsto f(v)$$

Compose the isomorphism $\mathrm{Hom}(V,V)\cong V^\vee\otimes V$ with evaluation. The resulting scalar is **the trace** of $T$. 

So:

$$\operatorname{Tr}(T)=\mathrm{ev}(\text{tensor corresponding to }T)$$

This is the intrinsic definition.

---

## 5.2 Why this becomes “sum of diagonal entries”

Using the earlier example

$$
T=
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix},
$$

its tensor is

$$
e_1^\vee\otimes e_1
+2e_2^\vee\otimes e_1
+3e_1^\vee\otimes e_2
+4e_2^\vee\otimes e_2
$$

Applying evaluation gives

$$e_1^\vee(e_1)+2e_2^\vee(e_1)+3e_1^\vee(e_2)+4e_2^\vee(e_2) =1+0+0+4=5$$

So only the “matching” terms survive, and that is exactly why trace equals the sum of diagonal entries. 

---

# 6. Core formulas to memorize

## Tensor product

$$(v_1+v_2)\otimes w=v_1\otimes w+v_2\otimes w$$

$$v\otimes(w_1+w_2)=v\otimes w_1+v\otimes w_2$$

$$(cv)\otimes w=v\otimes(cw)=c(v\otimes w)$$

## Dual basis

$$e_i^\vee(e_j)=\delta_{ij}$$

## Main isomorphism

$$V^\vee\otimes W \cong \mathrm{Hom}(V,W)$$

given by

$$\xi\otimes w \mapsto (v\mapsto \xi(v)w).$$

## Trace

$$
\operatorname{Tr}(T)=\mathrm{ev}(T)
\quad\text{under}\quad
\mathrm{Hom}(V,V)\cong V^\vee\otimes V
$$

---

# 7. Most important conceptual takeaways

### 1. Tensor product is not the same as direct sum

* direct sum adds dimensions
* tensor product multiplies dimensions

### 2. General tensors are sums of pure tensors

Do **not** assume every tensor is a single $v\otimes w$.

### 3. Dual vectors are linear functionals

Elements of $V^\vee$ do not live in $V$; they act on $V$.

### 4. $V\cong V^\vee$ is not usually canonical

The identification depends on basis unless extra structure exists.

### 5. Linear maps are tensors

$$V^\vee\otimes W$$

is the natural home of linear maps $V\to W$.

### 6. Trace is evaluation

Trace is not just a coordinate trick; it comes from a natural map

$$V^\vee\otimes V \to k$$

---

# 8. Problem section: what the chapter wants you to think about

The chapter ends with four deeper problems:

* **11A:** show that trace equals the sum of eigenvalues over an algebraically closed field
* **11B:** prove
  $$\operatorname{Tr}(T\otimes S)=\operatorname{Tr}(T)\operatorname{Tr}(S)$$
  
* **11C:** prove
  $$\operatorname{Tr}(T\circ S)=\operatorname{Tr}(S\circ T)$$
  
  for compatible maps $T$ and $S$
* **11D:** a Putnam problem about many highly independent eigenvectors implying the map may be scalar 

These are good indicators of what the author thinks is structurally important about trace.

---

# 9. Ultra-short summary

* $V\otimes W$ is a bilinear product space with basis $e_i\otimes f_j$.
* $V^\vee$ is the space of linear maps $V\to k$, with dual basis $e_i^\vee$.
* For finite-dimensional spaces,
  
  $$V^\vee\otimes W \cong \mathrm{Hom}(V,W)$$
  
* A linear operator $T:V\to V$ can therefore be seen as an element of $V^\vee\otimes V$.
* Applying evaluation $f\otimes v\mapsto f(v)$ gives $\operatorname{Tr}(T)$.
* This explains, in a basis-free way, why trace is the sum of diagonal entries. 

---





[Jordan Normal Form](/subpages/linear-algebra/jordan-form/)

---

## Eigen-things

This section develops the theory of eigenvalues and eigenvectors, culminating in the Jordan canonical form.

### Why You Should Care

A square matrix $T$ is really a linear map $V \to V$. The simplest such map is multiplication by a scalar $\lambda$, giving a diagonal matrix $\lambda I$. More generally, if we had a basis $e_1, \dots, e_n$ where $T$ just scales each $e_i$ by $\lambda_i$, the matrix would be diagonal:

$$T = \begin{pmatrix} \lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n \end{pmatrix}.$$

Computing $T^{100}$ with such a matrix is trivial: $e_i \mapsto \lambda_i^{100} e_i$.

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Getting lucky)</span></p>

Let $V$ be two-dimensional with basis $e_1, e_2$. Consider the map $T: V \to V$ defined by $e_1 \mapsto 2e_1$ and $e_2 \mapsto e_1 + 3e_2$, so

$$T = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix} \quad \text{in basis } e_1, e_2.$$

Rewriting: $e_1 \mapsto 2e_1$ and $e_1 + e_2 \mapsto 3(e_1 + e_2)$. Changing to the basis $e_1, e_1 + e_2$ gives

$$T = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix} \quad \text{in basis } e_1, e_1 + e_2.$$

Under a suitable change of basis, the map becomes diagonal.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Assumptions)</span></p>

Most theorems in this section require:
* **Finite-dimensional** vector spaces $V$,
* over a field $k$ which is **algebraically closed**.

The definitions work fine without these assumptions.

</div>

### Eigenvectors and Eigenvalues

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Eigenvector and Eigenvalue)</span></p>

Let $T: V \to V$ and $v \in V$ a **nonzero** vector. We say $v$ is an **eigenvector** if $T(v) = \lambda v$ for some $\lambda \in k$ (possibly zero, but $v \neq 0$). The value $\lambda$ is called an **eigenvalue** of $T$.

We abbreviate "$v$ is an eigenvector with eigenvalue $\lambda$" to "$v$ is a $\lambda$-eigenvector".

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Eigenvectors of a $2 \times 2$ matrix)</span></p>

Consider $T = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix}$.

* $e_1$ and $e_1 + e_2$ are $2$-eigenvectors and $3$-eigenvectors respectively.
* $5e_1$ is also a $2$-eigenvector.
* $7e_1 + 7e_2$ is also a $3$-eigenvector.

</div>

The $\lambda$-eigenvectors together with $\lbrace 0 \rbrace$ form a subspace.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($\lambda$-Eigenspace)</span></p>

For any $\lambda$, the **$\lambda$-eigenspace** is the set of $\lambda$-eigenvectors together with $0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Eigenvalues need not exist)</span></p>

Let $V = \mathbb{R}^2$ and let $T$ be the map which rotates a vector by $90°$ around the origin. Then $T(v)$ is not a multiple of $v$ for any nonzero $v \in V$.

</div>

However, over algebraically closed fields eigenvalues always exist:

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Eigenvalues always exist over algebraically closed fields)</span></p>

Suppose $k$ is an **algebraically closed** field. Let $V$ be a finite-dimensional $k$-vector space. Then if $T: V \to V$ is a linear map, there exists an eigenvalue $\lambda \in k$.

**Proof.** Fix any nonzero $v \in V$. The $n + 1$ vectors $v, T(v), \dots, T^n(v)$ (where $n = \dim V$) cannot be linearly independent, so there exists a nonzero monic polynomial $P$ with $P(T)(v) = 0$. Factor $P(z) = (z - r_1)\cdots(z - r_m)$ over $k$. Then

$$0 = (T - r_1 \mathrm{id}) \circ (T - r_2 \mathrm{id}) \circ \cdots \circ (T - r_m \mathrm{id})(v)$$

so at least one $T - r_i \mathrm{id}$ is not injective, giving an eigenvector. $\square$

</div>

### The Jordan Form

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Jordan Block)</span></p>

A **Jordan block** is an $n \times n$ matrix with $\lambda$ on the diagonal and $1$ on the superdiagonal:

$$J_n(\lambda) = \begin{pmatrix} \lambda & 1 & 0 & \cdots & 0 \\ 0 & \lambda & 1 & \cdots & 0 \\ \vdots & & \ddots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda & 1 \\ 0 & 0 & \cdots & 0 & \lambda \end{pmatrix}.$$

We allow $n = 1$, so $[\lambda]$ is a Jordan block.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Jordan Canonical Form)</span></p>

Let $T: V \to V$ be a linear map of finite-dimensional vector spaces over an algebraically closed field $k$. Then we can choose a basis of $V$ such that the matrix of $T$ is **block-diagonal** with each block being a Jordan block.

Such a matrix is said to be in **Jordan form**. This form is unique up to rearranging the order of the blocks.

</div>

This means $V$ decomposes as a direct sum

$$V = J_1 \oplus J_2 \oplus \cdots \oplus J_m$$

where $T$ acts on each subspace $J_i$ independently. In the simplest case $\dim J_i = 1$, so $J_i$ has a basis element $e$ with $T(e) = \lambda_i e$ (a simple eigenvalue). When $\dim J_i \geq 2$, we get $1$'s above the diagonal ("descending staircases").

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Concrete Jordan form)</span></p>

Let $T: k^6 \to k^6$ be given by

$$T = \begin{pmatrix} 5 & 0 & 0 & 0 & 0 & 0 \\ 0 & 2 & 1 & 0 & 0 & 0 \\ 0 & 0 & 2 & 0 & 0 & 0 \\ 0 & 0 & 0 & 7 & 0 & 0 \\ 0 & 0 & 0 & 0 & 3 & 0 \\ 0 & 0 & 0 & 0 & 0 & 3 \end{pmatrix}.$$

The eigenvectors and eigenvalues: for any $a, b \in k$,

$$T(a \cdot e_1) = 5a \cdot e_1, \quad T(a \cdot e_2) = 2a \cdot e_2, \quad T(a \cdot e_4) = 7a \cdot e_4, \quad T(a \cdot e_5 + b \cdot e_6) = 3[a \cdot e_5 + b \cdot e_6].$$

The element $e_3$ is **not** an eigenvector since $T(e_3) = e_2 + 2e_3$.

</div>

### Nilpotent Maps

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Nilpotent map)</span></p>

A map $T: V \to V$ is **nilpotent** if $T^m$ is the zero map for some integer $m$. (Here $T^m$ means "$T$ applied $m$ times".)

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The descending staircase)</span></p>

Let $V = k^{\oplus 3}$ with basis $e_1, e_2, e_3$. The map $T$ sending

$$e_3 \mapsto e_2 \mapsto e_1 \mapsto 0$$

is nilpotent since $T(e_1) = T^2(e_2) = T^3(e_3) = 0$, hence $T^3(v) = 0$ for all $v \in V$. Its matrix is

$$T = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix},$$

which is a Jordan block with $\lambda = 0$. The only eigenvalue is $0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Double staircase)</span></p>

Let $V = k^{\oplus 5}$ with basis $e_1, \dots, e_5$. The map

$$e_3 \mapsto e_2 \mapsto e_1 \mapsto 0 \quad \text{and} \quad e_5 \mapsto e_4 \mapsto 0$$

is nilpotent. It consists of two independent staircases.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Nilpotent Jordan)</span></p>

Let $V$ be a finite-dimensional vector space over an algebraically closed field $k$. Let $T: V \to V$ be a nilpotent map. Then we can write $V = \bigoplus_{i=1}^{m} V_i$ where each $V_i$ has a basis of the form $v_i, T(v_i), \dots, T^{\dim V_i - 1}(v_i)$ for some $v_i \in V_i$, and $T^{\dim V_i}(v_i) = 0$.

Hence: **every nilpotent map can be viewed as independent staircases.**

</div>

The key observation: if $S$ is a nilpotent staircase matrix, then $S + \lambda \mathrm{id}$ is a Jordan block with eigenvalue $\lambda$. This gives the strategy: decompose $V$ into subspaces where $T - \lambda \mathrm{id}$ is nilpotent, then apply Nilpotent Jordan.

### Reducing to the Nilpotent Case

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($T$-invariant subspace)</span></p>

Let $T: V \to V$. A subspace $W \subseteq V$ is called **$T$-invariant** if $T(w) \in W$ for any $w \in W$. In this way, $T$ can be thought of as a map $W \to W$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Indecomposable map)</span></p>

A map $T: V \to V$ is called **indecomposable** if it is impossible to write $V = W_1 \oplus W_2$ where both $W_1$ and $W_2$ are nontrivial $T$-invariant subspaces.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Invariant subspace decomposition)</span></p>

Let $V$ be a finite-dimensional vector space. Given any map $T: V \to V$, we can write

$$V = V_1 \oplus V_2 \oplus \cdots \oplus V_m$$

where each $V_i$ is $T$-invariant, and $T: V_i \to V_i$ is indecomposable for every $i$.

**Proof.** Same as the proof that every integer is a product of primes. If $V$ is not decomposable, we are done. Otherwise write $V = W_1 \oplus W_2$ and repeat on each factor. $\square$

</div>

With this decomposition, consider $T: V_1 \to V_1$ indecomposable. It has an eigenvalue $\lambda_1$, so let $S = T - \lambda_1 \mathrm{id}$, giving $\ker S \neq \lbrace 0 \rbrace$. Since $V_1 = \ker S^N \oplus \operatorname{im} S^N$ for some $N$, and $T$ is indecomposable, we must have $\operatorname{im} S^N = \lbrace 0 \rbrace$ and $\ker S^N = V_1$. Hence $S$ is nilpotent, and since $T$ is indecomposable there is only one staircase. Thus $V_1$ is a Jordan block.

### Algebraic and Geometric Multiplicity

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Geometric and Algebraic Multiplicity)</span></p>

Let $T: V \to V$ be a linear map and $\lambda$ a scalar.

* The **geometric multiplicity** of $\lambda$ is the dimension $\dim V_\lambda$ of the $\lambda$-eigenspace.
* The **generalized eigenspace** $V^\lambda$ is the subspace of $V$ for which $(T - \lambda \mathrm{id})^n(v) = 0$ for some $n \geq 1$. The **algebraic multiplicity** of $\lambda$ is $\dim V^\lambda$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Eigenspaces via Jordan form)</span></p>

Consider the matrix in Jordan form

$$T = \begin{pmatrix} 7 & 1 & & & & \\ 0 & 7 & & & & \\ & & 9 & & & \\ & & & 7 & 1 & 0 \\ & & & 0 & 7 & 1 \\ & & & 0 & 0 & 7 \end{pmatrix}$$

and let $\lambda = 7$.

* The eigenspace $V_\lambda$ has basis $e_1$ and $e_4$, so the **geometric multiplicity** is $2$.
* The generalized eigenspace $V^\lambda$ has basis $e_1, e_2, e_4, e_5, e_6$, so the **algebraic multiplicity** is $5$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Geometric and algebraic multiplicity vs Jordan blocks)</span></p>

Assume $T: V \to V$ is written in Jordan form. Let $\lambda$ be a scalar. Then

* The **geometric multiplicity** of $\lambda$ is the number of Jordan blocks with eigenvalue $\lambda$; the eigenspace has one basis element per Jordan block.
* The **algebraic multiplicity** of $\lambda$ is the sum of the dimensions of the Jordan blocks with eigenvalue $\lambda$; the generalized eigenspace is the direct sum of the corresponding subspaces.

The geometric multiplicity is always $\leq$ the algebraic multiplicity.

</div>

This gives tentative definitions:

* The **trace** is the sum of the eigenvalues, counted with algebraic multiplicity.
* The **determinant** is the product of the eigenvalues, counted with algebraic multiplicity.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Diagonalizable)</span></p>

A linear map $T: V \to V$ (where $\dim V$ is finite) is **diagonalizable** if it has a basis $e_1, \dots, e_n$ such that each $e_i$ is an eigenvector. Over an algebraically closed field, $T$ is diagonalizable if and only if for every $\lambda$, the geometric multiplicity equals the algebraic multiplicity.

</div>