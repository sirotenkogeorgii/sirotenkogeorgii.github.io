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

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Field)</span></p>

A **field** is a set $\mathbb{T}$ with two **commutative** binary operations $+$ and $\cdot$ such that:

1. $(\mathbb{T},+)$ is an **abelian group** (neutral element is $0$; inverse of $a$ is $-a$).
2. $(\mathbb{T}\setminus{0},\cdot)$ is an **abelian group** (neutral element is $1$; inverse of $a\neq 0$ is $a^{-1}$).
3. **Distributivity:** for all $a,b,c\in \mathbb{T}$,

   $$a\cdot(b+c)=a\cdot b+a\cdot c$$

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
  
  $(A+B)*{ij}=A*{ij}+B_{ij}, \quad (\alpha A)*{ij}=\alpha A*{ij}$

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

<div class="math-callout math-callout--example" markdown="1">
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

<div class="math-callout math-callout--example" markdown="1">
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

<div class="math-callout math-callout--example" markdown="1">
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

<div class="math-callout math-callout--example" markdown="1">
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

<div class="math-callout math-callout--example" markdown="1">
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





[Jordan Normal Form](/subpages/linear-algebra/jordan-form/)