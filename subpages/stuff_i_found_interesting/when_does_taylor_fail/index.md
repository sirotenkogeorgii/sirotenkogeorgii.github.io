## Taylor Approximation from Mat. Analyza 1

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Taylor Approximation)</span></p>

If $n \in \mathbb{N}\_0$ and $f\colon U(b, \delta) \to \mathbb{R}$ is a function with finite $f^{(n)}(b) \in \mathbb{R}$ (i.e., $f$ is continuous at $b$ for $n = 0$), then for $x \in U(b, \delta)$ and $x \to b$,

$$f(x) = T_n^{f,b}(x) + o((x - b)^n) = \sum_{j=0}^{n} \frac{f^{(j)}(b)}{j!}(x - b)^j + \underbrace{o((x - b)^n)}_{e(x)}.$$

The notation $o(\dots)$ means that $\lim\_{x \to b} e(x)/(x - b)^n = 0$.

</div>

## When Taylor approximations fall apart?

A Taylor approximation fails when the error (or "remainder") term does not shrink to zero as you include more terms. This breakdown primarily happens under specific mathematical and structural conditions, rather than being a random event.
Here is exactly when Taylor approximations fall apart:

### 1. Moving Too Far from the Base Point

Taylor polynomials are strictly local approximations. The further you move away from the center point of expansion (let's call it $x=a$), the larger the error becomes. Even if the series mathematically converges, it might take a massive number of terms to be accurate just a short distance away.

### 2. Lack of Smoothness (Non-Differentiable Points)

To build a Taylor polynomial of degree $n$, you must be able to take $n$ continuous derivatives of the function. The approximation instantly fails at:

* **Corners or cusps:** (e.g., $f(x) = \vert{}x\vert{}$ at $x=0$) where the slope changes abruptly.
* **Vertical tangents:** (e.g., $f(x) = x^{1/3}$ at $x=0$) where the derivative goes to infinity.
* **Discontinuities:** Points where the function itself has a jump or break.

### 3. Non-Analytic Functions
Just because a function is infinitely differentiable does not mean a Taylor series will perfectly approximate it. If a function's higher-order derivatives grow in magnitude too quickly, the Taylor series might converge to a completely different value than the original function. A classic example is the bump function:
$f(x) = e^{-1/x^2}$ for $x \neq 0$, and $0$ for $x = 0$.
All of its derivatives at $x = 0$ equal zero. Therefore, its Taylor series approximation is just $0$ everywhere, which fails to capture the actual "bumping" behavior of the function.

### 4. Hitting a Radius of Convergence

For complex functions, Taylor series have a strictly defined "radius of convergence" (an exact distance limit) governed by singularities—usually points of division by zero or the presence of complex/imaginary numbers.

* **Example:** For the function $f(x) = \frac{1}{1+x^2}$ centered at $x=0$, the Taylor series fails entirely at $x = 1$ and beyond, blowing up to infinity because of the complex singularity at $x = i$ and $x = -i$.

### How to check for failure

To mathematically verify if your approximation is failing, you must analyze the Taylor Remainder (Error) bound, denoted as $R_n(x)$. The approximation only works if the error limit approaches zero as the number of terms goes to infinity:

$$\lim_{n \to \infty} R_n(x) = 0$$

## Assume that at the point the function has all derivatives. Could it still be non-approximable by Taylor approximation?

Yes, a function can have a derivative at a point and still be completely non-approximable by a Taylor approximation.
To build a Taylor polynomial of degree $n$, the function must have continuous derivatives up to order $n$. Having just a first derivative is not enough to guarantee higher-order approximations. Furthermore, even if a function has infinite derivatives at a point, the Taylor series can still completely fail to approximate it.
Here is a breakdown of why and how this happens.

<figure>
  <img src="{{ '/assets/images/notes/stuff_i_found_interesting/bump_function_taylor_notes.png' | relative_url }}"" loading="lazy">
  <!-- <figcaption>The limit of a sequence of continuous functions does not have to be continuous: the sequence of functions $f_{n}(x)=\sin ^{n}(x)$ (marked in green and blue) converges pointwise over the entire domain, but the limit function is discontinuous (marked in red).</figcaption> -->
</figure>

### 1. Insufficient Higher-Order Derivatives

If a function only has a first derivative at a point $x = a$, you can only construct the first-degree linear approximation:

$$T_1(x) = f(a) + f'(a)(x-a)$$ 

If the second derivative $f^{''}(a)$ does not exist, it is impossible to create a quadratic ($n=2$) or any higher-degree Taylor approximation. The math simply breaks down because the required coefficients cannot be calculated.

### 2. Discontinuous Higher Derivatives

Even if the derivatives exist, they must be continuous in an open interval around the point for the error bounds to hold. If the derivative exists exactly at the base point but is wildly discontinuous everywhere else, the Taylor approximation will fail to accurately predict the behavior of the function even an infinitesimal distance away.

### 3. The Infinitely Differentiable Trap (Non-Analytic Smooth Functions)

The most surprising failure occurs when a function has every higher-order derivative ($f', f^{''}, f^{'''}, \dots$) at a point, yet the Taylor series still fails.
Consider the classic "bump-like" function centered at $x = 0$:

$$f(x) = \begin{cases} e^{-1/x^2} & \text{if } x \neq 0 \\ 0 & \text{if } x = 0 \end{cases}$$ 

* **The Derivatives:** If you calculate the derivatives at $x = 0$ using the limit definition, you find that $f'(0) = 0$, $f^{''}(0) = 0$, and every subsequent derivative $f^{(n)}(0) = 0$.
* **The Taylor Series:** Because every single derivative is exactly zero, the resulting Maclaurin series is:

  $$T(x) = 0 + 0x + \frac{0}{2!}x^2 + \dots = 0$$ 

* **The Failure:** The Taylor series is just a flat line ($y = 0$). While it technically "converges," it converges to $0$ everywhere. However, for any $x \neq 0$, $e^{-1/x^2}$ is strictly greater than $0$. The approximation fails completely for every point other than the origin itself.

### Summary of Conditions

For a Taylor approximation to succeed, the function must be analytic at that point. This requires not just the existence of a derivative, but that the function is infinitely differentiable and that its Taylor series actually equals the function within a neighborhood around the point.

### Conclusion

Even if a function has a derivative at a specific point, it can completely defy Taylor approximation if higher-order derivatives do not exist, or if the function belongs to the class of non-analytic smooth functions where the Taylor series converges to the wrong values.
Would you like to see how to mathematically check if a specific function is analytic using the Taylor remainder theorem?

## Appendix

## Analytic Function

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Analytic function)</span></p>

Let $U \subseteq \mathbb{R}$ be open. A function $f:U\to\mathbb{R}$ is **analytic** at the point $p\in U$ if there exists an open neighborhood $V\subseteq U$ of $p$ and coefficients $a_n\in\mathbb{R}$ such that

$$
f(x)=\sum_{n=0}^{\infty} a_n (x-p)^n
$$

for every $x\in V$, where the power series converges on $V$. The function $f$ is **analytic on** $U$ if it is analytic at every point of $U$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Taylor series)</span></p>

For a smooth function $f$, the power series

$$
\sum_{n=0}^{\infty} \frac{f^{(n)}(p)}{n!}(x-p)^n
$$

is called the **Taylor series** of $f$ at $p$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Analytic functions)</span></p>

* $e^x$, $\sin x$, $\cos x$, and polynomials are analytic.
* $\frac{1}{1-x}$ is analytic on $\mathbb R\setminus\{1\}$. Around $0$, we have

  $$
  \frac{1}{1-x}=\sum_{n=0}^{\infty}x^n
  $$

  for $\lvert x\rvert<1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Analytic is stronger than smooth)</span></p>

A useful contrast:

* **Smooth** means infinitely differentiable.
* **Analytic** is stronger: the function must equal its Taylor series near each point.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Smooth, but not analytic functions)</span></p>

So a function can be smooth but **not** analytic. A classic example is

$$
f(x)=
\begin{cases}
e^{-1/x^2}, & x\neq 0 \\
0, & x=0
\end{cases}
$$

This function is infinitely differentiable at $0$, and all its derivatives at $0$ are $0$. Therefore its Taylor series at $0$ is identically $0$, but the function itself is nonzero for $x\neq 0$. Hence it is smooth but not analytic at $0$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Closure properties of analytic functions)</span></p>

Sums, products, scalar multiples, and compositions of analytic functions are analytic. Also, if $f$ and $g$ are analytic and $g$ is nonzero on the relevant domain, then $f/g$ is analytic.

</div>