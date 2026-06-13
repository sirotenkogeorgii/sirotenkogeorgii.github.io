---
layout: default
title: Cookbook Tricks
tags:
  - mathematics
  - inequalities
  - analysis
---

## Inequalities and Identities

1. **Difference of squares** 
   
   $$\lVert a \rVert^2 - \lVert b \rVert^2 = \langle a-b, a+b \rangle$$

2. **Cauchy-Schwarz (for any inner product)**
   
   $$\lvert \langle a, b \rangle \rvert \leq \lVert a \rVert \cdot \lVert b \rVert$$

3. Convexity for $\alpha \in [0, h]$

4. **Young inequality (general)**
   
   $$ab \leq \frac{a^p}{p} + \frac{b^q}{q} \qquad \forall a,b\in\mathbb{R},$$
   
   $$\frac{1}{p} + \frac{1}{q} = 1 \qquad \forall p,q > 1$$

5. **Young inequality (frequent instance)**
   
   $$ab \leq \frac{1}{2}a^2 + \frac{1}{2}b^2 \qquad \forall a,b\in\mathbb{R}$$

6. Gronwall inequality
7. Bessel inequality
8. Minkowski's Inequality
9. Minkowski's Integral Inequality
10. Fenchel Inequality
11. Cauchy-Schwarz inequality for $L^2$ complex-valued functions
12. **The Matrix Exponential**
    
    $$e^A = \sum_{k=0}^{\infty} \frac{A^k}{k!}$$

13. **The Scalar Exponential**

    $$e^x = \sum_{k=0}^{\infty} \frac{x^k}{k!}$$

15. **Log and sqrt:**
    
    $$\sqrt{x} - 1 \geq \frac{\log x}{2} \qquad \forall x\geq 0$$

16. **$l1 < \sqrt n l2$**
    
    $$n\sum_{i=1}^n a_i^2 \;-\; \Big(\sum_{i=1}^n a_i\Big)^2 \;=\; \sum_{1\le i<j\le n} (a_i - a_j)^2$$

17. For $p,q\geq 0$
    
    $$\lvert p - q \rvert = \lvert \sqrt{p} - \sqrt{q} \rvert (\sqrt{p} + \sqrt{q})$$

    * derived from $a^2-b^2 = (a-b)(a+b)$

18. Some algebraic inequality: 

    $$1−(1−t)^k \leq \min(1,kt) \text{for } t\in[0,1]$$

    <figure>
      <img src="{{ '/assets/images/notes/random/roof_of_the_curve_ineq.png' | relative_url }}" alt="Left: vectors e1=(1,0), e2=(1,1) and their dual basis covectors ě1=(1,-1), ě2=(0,1) plotted in R² with level sets of the covectors as faded diagonal and horizontal lines. Right: the canonical basis is its own dual under the canonical Euclidean inner product." loading="lazy">
      <!-- <figcaption>Left (b): the dual basis $\check e^1 = (1, -1)$, $\check e^2 = (0, 1)$ for the oblique basis $e_1, e_2$. Faded red lines are level sets of $\check e^1$ (diagonals along direction $e_2$); faded orange lines are level sets of $\check e^2$ (horizontals along $e_1$). Each covector $\check e^i$ kills $e_j$ for $j \ne i$. Right (c): the canonical basis under the canonical inner product is self-dual.</figcaption> -->
    </figure>
    
19. **Apollonius's Theorem**
    
    $$\text{Side}_1^2 + \text{Side}_2^2 = 2 \cdot (\text{Median})^2 + 2 \cdot \left(\frac{\text{Base}}{2}\right)^2$$

20. **Corollary of Appollonius's Theorem (useful for Gaussian kernels decomposition)**

    $$\lVert t-x \rVert^2 + \lVert t-y \rVert^2 = 2 \left\lVert t - \frac{x+y}{2} \right\rVert^2 + 2 \left( \frac{\lVert x-y \rVert}{2} \right)^2$$

21. **Euler–Poisson integral**

    $$\int_{\infty}^{\infty} e^{-ax^2} = \sqrt{\frac{\pi}{a}}$$

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

This is one of the most famous—and cleverest—tricks in all of calculus. You cannot compute the antiderivative of $e^{-ax^2}$ using standard elementary functions, so trying to solve the 1D integral directly is a dead end.

Instead, the trick is to **square the integral** and push it into two dimensions, where the geometry of the plane makes it trivial to solve.

Here is the step-by-step mathematical derivation.

**Step 1: Square the Integral**

Let the integral we want to solve be $I$:

$$I = \int_{-\infty}^\infty e^{-ax^2} dx$$

Since $e^{-ax^2}$ is strictly positive, we know that $I > 0$. Let's square $I$. To keep the variables distinct, we will use $x$ for the first integral and $y$ for the second:

$$I^2 = \left( \int_{-\infty}^\infty e^{-ax^2} dx \right) \left( \int_{-\infty}^\infty e^{-ay^2} dy \right)$$

**Step 2: Combine into a Double Integral**

Because the integrals are independent, we can combine them into a single double integral over the entire 2D Cartesian plane ($\mathbb{R}^2$):

$$I^2 = \int_{-\infty}^\infty \int_{-\infty}^\infty e^{-ax^2} e^{-ay^2} dx dy$$

Using the property of exponents ($e^A e^B = e^{A+B}$), this becomes:

$$I^2 = \int_{-\infty}^\infty \int_{-\infty}^\infty e^{-a(x^2 + y^2)} dx dy$$

**Step 3: Switch to Polar Coordinates**

This is where the magic happens. The term $(x^2 + y^2)$ perfectly maps to the squared radius in polar coordinates.
We make the standard substitution:

* $r^2 = x^2 + y^2$
* The area element $dx dy$ becomes $r dr d\theta$

We also need to change the bounds. Integrating over the entire Cartesian plane ($x$ and $y$ from $-\infty$ to $\infty$) is exactly the same as integrating the radius from $0$ to $\infty$, and the angle from $0$ to $2\pi$:

$$I^2 = \int_0^{2\pi} \int_0^\infty e^{-ar^2} r dr d\theta$$

**Step 4: Evaluate the Integral**

The problem is now trivial because that extra $r$ from the polar area element allows us to use simple u-substitution.

Let's evaluate the inner integral with respect to $r$:

$$\int_0^\infty e^{-ar^2} r dr$$

Let $u = -ar^2$. Then $du = -2ar dr$, which means $r dr = -\frac{du}{2a}$.
The integral becomes:

$$\left[ -\frac{1}{2a} e^{-ar^2} \right]_0^\infty$$

Evaluating this at the boundaries:

* As $r \to \infty$, $e^{-ar^2} \to 0$.
* At $r = 0$, $e^0 = 1$.

So the inner integral evaluates to:

$$0 - \left( -\frac{1}{2a} (1) \right) = \frac{1}{2a}$$

Now, substitute this result back into the outer integral with respect to $\theta$:

$$I^2 = \int_0^{2\pi} \frac{1}{2a} d\theta$$

Since $\frac{1}{2a}$ is just a constant, evaluating the integral over $2\pi$ gives:

$$I^2 = \frac{1}{2a} \cdot 2\pi = \frac{\pi}{a}$$

**Step 5: Take the Square Root**

We now have the value for $I^2$. Since we established at the very beginning that $I$ must be positive, we simply take the positive square root:

$$I = \sqrt{\frac{\pi}{a}}$$

</details>
</div>

## Tricks

1. **Constant function to integral to derivative:**
   
    $$h \lVert \dfrac{x_l - x_{l-1}}{h} \rVert^2 = \int_{(l-1)h}^{lh} \lVert \dfrac{x_l - x_{l-1}}{h} \rVert^2 dt = \int_{(l-1)h}^{lh} \lVert \dfrac{d}{dt} \tilde{x}_h(t) \rVert^2 dt$$

2. **The standard trick is to split the kernel mass evenly between two Cauchy–Schwarz factors:**
   * **The natural attempt — Cauchy–Schwarz in $y$ — fails because $K\in L^1$, not $L^2$**
     
     $$\lvert K(x-y) f(y)\rvert = \lvert K(x-y)\rvert^{1/2} \cdot \lvert K(x-y)\rvert^{1/2} \lvert f(y)\rvert$$

## Criteria

### Space Compactness Criteria

Key criteria:

* **Arzelà-Ascoli Theorem:** A classic criterion in real analysis that determines if a family of continuous functions in $C(X)$ is compact. It requires the family to be uniformly bounded and equicontinuous.
* **Kolmogorov-Riesz Theorem:** A criterion applied to $L^{p}$ spaces that determines when a subset of functions is compact, primarily relying on tightness and $L^{p}$-bounded translation conditions.
* **Dunford-Pettis Theorem:** A standard weak compactness criterion in probability and measure theory that asserts a subset of $L^{1}$ is weakly compact if and only if it is uniformly integrable

### Operator Compactness Criteria

## Definition

A compact operator is a bounded linear operator $T: X \rightarrow Y$ between Banach spaces where the image of the closed unit ball is relatively compact.

## Core Criteria (Equivalent Conditions)

* **Sequential Definition:** Every bounded sequence $\lbrace x_n\rbrace$ in $X$ has a subsequence $\lbrace x_{n_k}\rbrace$ such that $\lbrace Tx_{n_k}\rbrace$ converges in $Y$.
* **Relative Compactness:** The closure of the image of the closed unit ball, $\overline{T(B_1(0))}$, is a compact set in $Y$.
* **Finite Rank Approximation:** The operator $T$ is the uniform limit of a sequence of finite-rank operators (when $Y$ has the approximation property, which includes all Hilbert spaces).
* **Schauder's Theorem:** The operator $T$ is compact if and only if its adjoint operator $T^\ast$ is compact.

## Hilbert Space Characterization

* **Singular Values:** $T$ is compact if and only if its sequence of singular values $s_n(T)$ converges to $0$ as $n \rightarrow \infty$.
* **Hilbert-Schmidt Condition:** If $\sum s_n(T)^2 < \infty$, the operator is Hilbert-Schmidt, which automatically makes it compact.

## Essential Spectral Properties

* **Eigenvalues:** Non-zero eigenvalues are isolated and have finite multiplicity.
* **Accumulation Point:** The only possible accumulation point for the spectrum is $0$.
* **Zero Value:** If the space is infinite-dimensional, $0$ is always in the spectrum.



### Boundedness Criteria

For a **linear operator**

$$T:X\to Y$$

between normed vector spaces, the main criterion is:

$$\boxed{\exists C\ge 0 \text{ such that } |Tx|_Y \le C|x|_X \quad \forall x\in X.}$$

If such a constant exists, then $T$ is called **bounded**.

Equivalently,

$$\boxed{\sup_{|x|_X\le 1}|Tx|_Y <\infty.}$$

This supremum is the **operator norm**:

$$|T|_{\mathcal L(X,Y)} = \sup_{|x|\le 1}|Tx| = \sup_{|x|=1}|Tx|.$$

So the practical test is:

$$T \text{ is bounded} \iff T \text{ sends the unit ball to a bounded set.}$$

## Equivalent criteria for linear operators

For a linear operator $T:X\to Y$, the following are equivalent:

$$\boxed{T \text{ is bounded}}$$

$$\boxed{\exists C>0:|Tx|\le C|x|\ \forall x}$$

$$\boxed{T \text{ is continuous everywhere}}$$

$$\boxed{T \text{ is continuous at }0}$$

$$\boxed{T \text{ maps bounded sets to bounded sets}}$$

$$\boxed{\sup_{|x|\le 1}|Tx|<\infty}$$

The most important fact is:

$$\boxed{\text{For linear maps, boundedness } \Longleftrightarrow \text{ continuity}.}$$

This is special to **linear** operators.

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Matrix operator is always bounded)</span></p>

If $A:\mathbb R^n\to \mathbb R^m$ is given by a matrix, then

$$T(x)=Ax$$

is always bounded.

Why? In finite dimensions, all linear maps are continuous, hence bounded.

So in finite-dimensional spaces:

$$\boxed{\text{Every linear operator is bounded.}}$$

The difficulty appears mainly in infinite-dimensional spaces.

</div>

## Example of an unbounded operator

Consider

$$D:C^1[0,1]\to C[0,1], \qquad Df=f',$$

with the norm

$$|f|_\infty=\sup_{x\in[0,1]}|f(x)|.$$

Take

$$f_n(x)=\sin(nx).$$

Then

$$|f_n|_\infty=1,$$

but

$$Df_n(x)=n\cos(nx),$$

so

$$|Df_n|_\infty=n.$$

Therefore,

$$\sup_{|f|_\infty\le 1}|Df|_\infty=\infty.$$

So $D$ is **not bounded** with this choice of norm.

The derivative can make functions oscillate more and more wildly without increasing their sup norm.

---

## Important Hilbert space criterion

If $H,K$ are Hilbert spaces and $T:H\to K$ is linear, then $T$ is bounded iff

$$\exists C>0:|Tx|_K^2\le C^2|x|_H^2 \quad \forall x\in H.$$

Same idea, just squared.

Also, for a linear functional $L:H\to \mathbb R$ or $\mathbb C$,

$$L \text{ is bounded} \iff \exists C>0:\ |L(x)|\le C|x|.$$

For example, if

$$L(x)=\langle x,a\rangle,$$

then by Cauchy-Schwarz,

$$|L(x)|=|\langle x,a\rangle| \le |x||a|.$$

So $L$ is bounded and

$$|L|=|a|.$$

This is the basic mechanism behind the Riesz representation theorem.

---

## Practical checklist

To prove $T$ is bounded, try to show:

$$|Tx|\le C|x|.$$

Usually you use inequalities such as:

$$|\langle x,y\rangle|\le |x||y|,$$

or

$$|Ax|\le |A||x|,$$

or estimates involving integrals, expectations, or supremums.

To prove $T$ is unbounded, construct a sequence $x_n$ such that

$$|x_n|\le 1$$

but

$$|Tx_n|\to\infty.$$

That shows the image of the unit ball is not bounded.

## Being \mathcal{L}(X,Y) operator

Convention: \mathcal{L}(X,Y) is ...

---
