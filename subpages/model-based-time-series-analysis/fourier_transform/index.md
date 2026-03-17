---
title: Fourier Transform
layout: default
noindex: true
---

<style>
  .accordion summary {
    font-weight: 600;
    color: var(--accent-strong, #2c3e94);
    background-color: var(--accent-soft, #f5f6ff);
    padding: 0.35rem 0.6rem;
    border-left: 3px solid var(--accent-strong, #2c3e94);
    border-radius: 0.25rem;
  }
</style>

# Fourier Transform

What does polynomial multiplication have in common with audio compression? Or with image recognition? In this chapter, we will show that behind all these questions lies a common algebraic structure, known to mathematicians as the Discrete Fourier Transform. We will derive an efficient algorithm for calculating this transform and show some of its interesting consequences.

## Polynomials and Their Multiplication

First, let us briefly recall how to work with polynomials.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Polynomial)</span></p>

A **polynomial** is an expression of the type $P(x) = \sum_{i=0}^{n-1} p_i \cdot x^i$, where $x$ is a variable and $p_0$ to $p_{n-1}$ are numbers, which we call the **coefficients** of the polynomial. We denote polynomials with capital letters and their coefficients with the corresponding lowercase letters with indices. For now, we assume that all numbers are real, though in general, they could be elements of any commutative ring.

</div>

In algorithms, we usually represent polynomials by a vector of coefficients $(p_0, \dots, p_{n-1})$; contrary to the conventions of linear algebra, we will index the components of vectors from 0 throughout this chapter. The number of coefficients $n$ will be called the **size** of the polynomial, denoted $\|P\|$. We will express the time complexity of an algorithm in relation to the sizes of the input polynomials. We will assume that we can work with real numbers in constant time per operation.

If we add a new coefficient $p_n = 0$, the value of the polynomial does not change for any $x$. Similarly, if the highest coefficient $p_{n-1}$ is zero, we can omit it. In this way, we can reduce any polynomial to a **normal form**, in which it either has a non-zero highest coefficient or has no coefficients at all — this is the so-called **zero polynomial**, which is equal to zero for every $x$. The highest power with a non-zero coefficient is called the **degree** of the polynomial, $\deg P$; we assign a degree of $-1$ to the zero polynomial.

We treat polynomials as expressions. Addition and subtraction are straightforward, but let's look at what happens with multiplication:

$$P(x) \cdot Q(x) = \left( \sum_{i=0}^{n-1} p_i \cdot x^i \right) \cdot \left( \sum_{j=0}^{m-1} q_j \cdot x^j \right) = \sum_{i,j} p_i q_j x^{i+j}.$$

We can write this product as a polynomial $R(x)$, whose coefficient for $x^k$ is equal to $r_k = p_0q_k + p_1q_{k-1} + \dots + p_kq_0$. We can see that the polynomial $R$ has a degree of $\deg P + \deg Q$ and a size of $\|P\| + \|Q\| - 1$.

An algorithm that computes the product of two polynomials of size $n$ directly from the definition therefore requires $\Theta(n)$ time to compute each coefficient, for a total of $\Theta(n^2)$. Similar to number multiplication, we will try to find a more efficient method here as well.

### Graphs of Polynomials

Let's digress for a moment and consider when we regard two polynomials as being the same. This can be viewed in several ways. We can look at polynomials as expressions and compare their symbolic notations. In this case, two polynomials are equal if and only if they have the same vectors of coefficients after normalization. We then say they are **identical** and usually denote this as $P \equiv Q$.

Alternatively, we can compare polynomials as real functions. Polynomials $P$ and $Q$ are equal ($P = Q$) if and only if $P(x) = Q(x)$ for all $x \in \mathbb{R}$. Identically equal polynomials are also equal as functions, but does the reverse have to be true? The following theorem will show that it is, and that equality for a finite number of $x$ is sufficient.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Polynomial Identity)</span></p>

Let $P$ and $Q$ be polynomials of degree at most $d$. If $P(x_i) = Q(x_i)$ for mutually distinct numbers $x_0, \dots, x_d$, then $P$ and $Q$ are identical.

</div>

Let us first recall the following standard lemma about the roots of polynomials:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Roots of a Polynomial)</span></p>

A polynomial $R$ of degree $t \ge 0$ has at most $t$ roots (numbers $\alpha$ for which $R(\alpha) = 0$).

</div>

<div class="accordion">
  <details>
    <summary>Proof of Lemma</summary>
    <p>If we divide the polynomial $R$ by the polynomial $x - \alpha$ (see exercise 1), we get $R(x) \equiv (x - \alpha) \cdot R'(x) + \beta$, where $\beta$ is a constant. If $\alpha$ is a root of $R$, then $\beta$ must be 0. Furthermore, the polynomial $R'$ has degree $t-1$ and the same roots as $R$, with the possible exception of the root $\alpha$.</p>
    <p>If we repeat this process $t$ times, we will either run out of roots in the process (in which case the lemma certainly holds), or we will obtain the equality $R(x) \equiv (x - \alpha_1) \cdot \dots \cdot (x - \alpha_t) \cdot R''(x)$, where $R''$ is a polynomial of degree zero. Such a polynomial, however, cannot have any roots, and therefore $R$ cannot have any additional roots either. $\square$</p>
  </details>
</div>

<div class="accordion">
  <details>
    <summary>Proof of Theorem</summary>
    <p>It is sufficient to consider the polynomial $R(x) \equiv P(x) - Q(x)$. This polynomial has a degree of at most $d$, but each of the numbers $x_0, \dots, x_d$ is its root. According to the lemma, it must therefore be identically zero, and thus $P \equiv Q$. $\square$</p>
  </details>
</div>

Thanks to the preceding theorem, we can represent polynomials not only by a vector of coefficients but also by a vector of function values at some agreed-upon points — we will call this vector the **graph** of the polynomial. If we choose a sufficient number of points, the polynomial is uniquely determined by its graph.

In this representation, multiplying polynomials is trivial: The product of polynomials $P$ and $Q$ at a point $x$ has the value $P(x) \cdot Q(x)$. It is therefore sufficient to multiply the graphs component-wise, which we can do in linear time. We just need to be careful that the product has a higher degree than the individual factors, so we must evaluate the polynomials at twice the number of points.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(PolynomialMultiplication)</span></p>

1. Given polynomials $P$ and $Q$ of size $n$, defined by their coefficients. Without loss of generality, let's assume that the upper $n/2$ coefficients are zero for both polynomials, so the product $R \equiv P \cdot Q$ will also be a polynomial of size $n$.
2. Choose mutually distinct numbers $x_0, \dots, x_{n-1}$.
3. Compute the graphs of polynomials $P$ and $Q$, i.e., the vectors $(P(x_0), \dots, P(x_{n-1}))$ and $(Q(x_0), \dots, Q(x_{n-1}))$.
4. From this, calculate the graph of the product $R$ by component-wise multiplication: $R(x_i) = P(x_i) \cdot Q(x_i)$.
5. Find the coefficients of the polynomial $R$ that correspond to the graph.

</div>

Step 4 takes $\Theta(n)$ time, so the speed of the entire algorithm depends on the efficiency of conversions between the coefficient and value representations of polynomials. In general, we cannot do this in better than quadratic time, but here we have the option to choose the points $x_0, \dots, x_{n-1}$, so we will choose them cleverly so that the conversion can be done quickly.

### Evaluating a Polynomial using the Divide and Conquer Method

Now let's try to construct an algorithm for polynomial evaluation based on the Divide and Conquer method. Although this attempt will ultimately fail, it will be instructive to see why and how it failed.

Consider a polynomial $P$ of size $n$ that we want to evaluate at $n$ points. We will choose the points to be paired, i.e., to form pairs differing only in sign: $\pm x_0, \pm x_1, \dots, \pm x_{n/2-1}$.

We can decompose the polynomial $P$ into terms with even exponents and those with odd exponents:

$$P(x) = (p_0x^0 + p_2x^2 + \dots + p_{n-2}x^{n-2}) + (p_1x^1 + p_3x^3 + \dots + p_{n-1}x^{n-1}).$$

Furthermore, we can factor out $x$ from the second parenthesis:

$$P(x) = (p_0x^0 + p_2x^2 + \dots + p_{n-2}x^{n-2}) + x \cdot (p_1x^0 + p_3x^2 + \dots + p_{n-1}x^{n-2}).$$

Now, only even powers of $x$ appear in both parentheses. Therefore, we can consider each parenthesis as the evaluation of some polynomial of size $n/2$ at the point $x^2$:

$$P(x) = P_s(x^2) + x \cdot P_\ell(x^2),$$

where:

$$P_s(t) = p_0t^0 + p_2t^1 + \dots + p_{n-2}t^{\frac{n-2}{2}}, \qquad P_\ell(t) = p_1t^0 + p_3t^1 + \dots + p_{n-1}t^{\frac{n-2}{2}}.$$

Moreover, if we substitute the value $-x$ into $P$ in a similar way, we get:

$$P(-x) = P_s(x^2) - x \cdot P_\ell(x^2).$$

Evaluating the polynomial $P$ at the points $\pm x_0, \dots, \pm x_{n/2-1}$ can thus be reduced to evaluating the polynomials $P_s$ and $P_\ell$ of half the size at the points $x_0^2, \dots, x_{n/2-1}^2$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Failure of Divide and Conquer over $\mathbb{R}$)</span></p>

This suggests an algorithm with a time complexity of $T(n) = 2T(n/2) + \Theta(n)$, and from the Master Theorem, we know that such a recurrence has the solution $T(n) = \Theta(n \log n)$. But alas, this algorithm does not work: the squares that we pass to the recursive call are always non-negative, so they can no longer be properly paired. That is... at least as long as we are computing with real numbers.

We will show that in the domain of complex numbers, we can choose points that will remain correctly paired even after being squared several times.

</div>

---

## Intermezzo on Complex Numbers

In this chapter, we will be computing with complex numbers. Let's therefore review how to handle them.

### Basic Operations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Complex Numbers)</span></p>

$\mathbb{C} = \lbrace a + bi \mid a, b \in \mathbb{R}\rbrace$, where $i^2 = -1$.

* **Addition and Subtraction:** $(a + bi) \pm (p + qi) = (a \pm p) + (b \pm q)i$.
* **Multiplication:** $(a + bi)(p + qi) = (ap - bq) + (aq + bp)i$. For $\alpha \in \mathbb{R}$, $\alpha(a + bi) = \alpha a + \alpha bi$.
* **Complex Conjugation:** $\overline{a + bi} = a - bi$. Properties: $\overline{\overline{x}} = x$, $\overline{x \pm y} = \overline{x} \pm \overline{y}$, $\overline{x \cdot y} = \overline{x} \cdot \overline{y}$, $x \cdot \overline{x} = a^2 + b^2 \in \mathbb{R}$.
* **Absolute Value:** $\|x\| = \sqrt{x \cdot \overline{x}}$, so $\|a + bi\| = \sqrt{a^2 + b^2}$. Also, $\|\alpha x\| = \|\alpha\| \cdot \|x\|$ for $\alpha \in \mathbb{R}$.
* **Division:** To compute the quotient $x/y$, we multiply the numerator and denominator by $\overline{y}$ to get $(x \cdot \overline{y}) / (y \cdot \overline{y})$. The denominator is now real, so we can divide each component of the numerator separately.

</div>

### Gaussian Plane and Trigonometric Form

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Geometric Interpretation)</span></p>

* We associate complex numbers with points in $\mathbb{R}^2$: $a + bi \leftrightarrow (a, b)$.
* $\|x\|$ expresses the distance from the point $(0, 0)$.
* $\|x\| = 1$ for numbers lying on the unit circle (complex units). Then $x = \cos\phi + i \sin\phi$ for some $\phi \in [0, 2\pi)$.
* For any $x \in \mathbb{C}$: $x = \|x\| \cdot (\cos\phi(x) + i \sin\phi(x))$. The number $\phi(x)$ is called the **argument** of $x$ and is sometimes also denoted as $\arg x$. Arguments are periodic with a period of $2\pi$ and are often normalized to the interval $[0, 2\pi)$.
* Additionally, $\phi(\overline{x}) = -\phi(x)$, if we consider arguments modulo $2\pi$.

</div>

### Exponential Form

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Exponential Form of Complex Numbers)</span></p>

* **Euler's Formula:** $e^{i\phi} = \cos\phi + i \sin\phi$.
* Any $x \in \mathbb{C}$ can thus be written as $\|x\| \cdot e^{i\phi(x)}$.
* **Multiplication:** $xy = \|x\| \cdot \|y\| \cdot e^{i(\phi(x)+\phi(y))}$ (absolute values are multiplied, arguments are added).
* **Exponentiation:** For $\alpha \in \mathbb{R}$, $x^\alpha = \|x\|^\alpha \cdot e^{i\alpha\phi(x)}$.

</div>

### Roots of Unity

Finding roots in complex numbers is not generally unique: for example, if we look for the fourth root of unity, i.e., solve the equation $x^4 = 1$, we find four solutions: $1$, $-1$, $i$, and $-i$.

Let us now examine more generally how the n-th roots of unity behave, i.e., the complex roots of the equation $x^n = 1$:

* Since $\|x^n\| = \|x\|^n$, it must be that $\|x\| = 1$. Therefore, $x = e^{i\phi}$ for some $\phi$.
* It must hold that $1 = x^n = e^{i\phi n} = \cos(\phi n) + i \sin(\phi n)$. This occurs whenever $\phi n = 2k\pi$ for some $k \in \mathbb{Z}$.

We thus obtain $n$ distinct n-th roots of 1, namely $e^{2k\pi i / n}$ for $k = 0, \dots, n-1$. Some of these roots are special:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Primitive Root of Unity)</span></p>

A complex number $x$ is a **primitive n-th root of unity** if $x^n = 1$ and none of the numbers $x^1, x^2, \dots, x^{n-1}$ are equal to 1.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Fourth Roots of Unity)</span></p>

Of the four fourth roots of 1, $i$ and $-i$ are primitive, while $1$ and $-1$ are not (verify this by substitution). For a general $n > 2$, there are always at least two primitive roots, namely the numbers $\omega = e^{2\pi i / n}$ and $\overline{\omega} = e^{-2\pi i / n}$. This is because $\omega^j = e^{2\pi ij / n}$, which is equal to 1 if and only if $j$ is a multiple of $n$ (the individual powers of $\omega$ successively traverse the unit circle). The same applies to $\overline{\omega}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Properties of Primitive Roots)</span></p>

For an even $n$ and any number $\omega$ that is a primitive n-th root of unity, the following holds:

* $\omega^j \ne \omega^k$ whenever $0 \le j < k < n$. It is sufficient to look at the quotient $\omega^k / \omega^j = \omega^{k-j}$. This cannot be equal to one, because $0 < k - j < n$ and $\omega$ is primitive.
* $\omega^{n/2} = -1$. This is because $(\omega^{n/2})^2 = \omega^n = 1$, so $\omega^{n/2}$ is a square root of 1. There are only two such roots: 1 and $-1$. However, it cannot be 1, because $\omega$ is primitive.

</div>

---

## The Fast Fourier Transform (FFT)

The challenge of efficiently evaluating polynomials can be addressed by leveraging primitive roots of unity. This section demonstrates how these mathematical constructs rescue the pairing algorithm for polynomial evaluation.

First, we augment the polynomials with zero coefficients so that their size, $n$, becomes a power of two. We then select a primitive n-th root of unity, $\omega$, and evaluate the polynomial at the points $\omega^0, \omega^1, \dots, \omega^{n-1}$. These points are distinct complex numbers and are conveniently paired: the values $\omega^{n/2}, \dots, \omega^{n-1}$ differ from $\omega^0, \dots, \omega^{n/2-1}$ only by their sign. This is easily verified, as for $0 \le j < n/2$:

$$\omega^{n/2+j} = \omega^{n/2}\omega^j = -\omega^j.$$

Furthermore, $\omega^2$ is a primitive $(n/2)$-th root of unity. This allows for a recursive call on a problem of the same type, which is also correctly paired.

This successful application of the Divide and Conquer method yields an algorithm with a complexity of $\Theta(n \log n)$ for polynomial evaluation. We will now adapt this algorithm to operate on vectors of coefficients and values. This algorithm is known as the **Fast Fourier Transform (FFT)**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Fast Fourier Transform)</span></p>

**Input:**
* An integer $n = 2^k$.
* A primitive n-th root of unity, $\omega$.
* A vector $(p_0, \dots, p_{n-1})$ of coefficients for a polynomial $P$.

**Algorithm:**
1. If $n = 1$, set $y_0 \leftarrow p_0$ and terminate.
2. Otherwise, recursively call the algorithm on the even and odd parts of the coefficients:
   * $(s_0, \dots, s_{n/2-1}) \leftarrow \text{FFT}(n/2, \omega^2, (p_0, p_2, p_4, \dots, p_{n-2}))$.
   * $(\ell_0, \dots, \ell_{n/2-1}) \leftarrow \text{FFT}(n/2, \omega^2, (p_1, p_3, p_5, \dots, p_{n-1}))$.
3. Combine the results from both parts to construct the values of the full polynomial:
   * For $j = 0, \dots, n/2 - 1$:
     * $y_j \leftarrow s_j + \omega^j \cdot \ell_j$ (the power $\omega^j$ is recalculated iteratively)
     * $y_{j+n/2} \leftarrow s_j - \omega^j \cdot \ell_j$

**Output:**
* The values of the polynomial $P$, i.e., the vector $(y_0, \dots, y_{n-1})$ where $y_j = P(\omega^j)$.

</div>

### The Discrete Fourier Transform (DFT)

While we can efficiently evaluate a polynomial at the powers of $\omega$, we also need to perform the reverse operation — converting values back to coefficients — just as quickly. To achieve this, we can view polynomial evaluation more abstractly as a mapping from one vector of complex numbers to another. This mapping is known as the **Fourier Transform**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Discrete Fourier Transform)</span></p>

The **DFT** is a mapping $F: \mathbb{C}^n \to \mathbb{C}^n$ that transforms a vector $\mathbf{x}$ into a vector $\mathbf{y}$ according to the rule:

$$y_j = \sum_{k=0}^{n-1} x_k \cdot \omega^{jk}$$

where $\omega$ is a fixed, chosen primitive n-th root of unity. The vector $\mathbf{y}$ is called the **Fourier image** of the vector $\mathbf{x}$.

</div>

If we denote $\mathbf{p}$ as the vector of coefficients of a polynomial $P$, its Fourier transform $F(\mathbf{p})$ is precisely the set of values of this polynomial at the points $\omega^0, \dots, \omega^{n-1}$. The FFT algorithm, therefore, computes the Discrete Fourier Transform in $\Theta(n \log n)$ time.

The DFT is a linear transformation and can be expressed as a matrix multiplication. Let $\Omega$ be a matrix where $\Omega_{jk} = \omega^{jk}$. Then $F(\mathbf{p}) = \Omega \mathbf{p}$. To convert the values back to coefficients, we need to find the inverse mapping, which is determined by the inverse matrix $\Omega^{-1}$.

Let $\bar{\omega} = \omega^{-1}$. Let us examine if the conjugate matrix $\bar{\Omega}$ is related to the inverse we seek.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Inverse of the DFT Matrix)</span></p>

$\Omega \cdot \bar{\Omega} = n \cdot E$, where $E$ is the identity matrix. Therefore, $\Omega^{-1} = \frac{1}{n} \cdot \bar{\Omega}$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>By substituting the definitions and performing elementary manipulations:</p>
    $$(\Omega \cdot \bar{\Omega})_{jk} = \sum_{\ell=0}^{n-1} \Omega_{j\ell} \cdot \bar{\Omega}_{\ell k} = \sum_{\ell=0}^{n-1} \omega^{j\ell} \cdot \bar{\omega}^{\ell k}$$
    <p>Since $\bar{\omega} = \omega^{-1}$:</p>
    $$\sum_{\ell=0}^{n-1} \omega^{j\ell} \cdot \omega^{-\ell k} = \sum_{\ell=0}^{n-1} (\omega^{(j-k)})^\ell$$
    <p>This final sum is a geometric series.</p>
    <ul>
      <li>If $j = k$, then $j-k=0$, and all terms in the series are $\omega^0 = 1$. The sum is therefore $n$.</li>
      <li>If $j \neq k$, we use the formula for the sum of a geometric series with ratio $q = \omega^{j-k}$:
        $$\sum_{\ell=0}^{n-1} q^\ell = \frac{q^n - 1}{q - 1} = \frac{\omega^{(j-k)n} - 1}{\omega^{j-k} - 1}$$
        The numerator can be simplified: $\omega^{(j-k)n} = (\omega^n)^{j-k} = 1^{j-k} = 1$. The numerator is therefore zero. The denominator is non-zero because $\omega$ is a primitive root and $0 < |j-k| < n$, so $\omega^{j-k} \neq 1$. Thus, the sum is 0.</li>
    </ul>
    <p>$\square$</p>
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Inverse Transform)</span></p>

The matrix $\Omega$ is regular, and its inverse is simply its complex conjugate scaled by $1/n$. Furthermore, the number $\bar{\omega} = \omega^{-1}$ is also a primitive n-th root of unity. This means that, apart from the scaling factor of $1/n$, the inverse transform is also a Fourier transform and can be computed using the same FFT algorithm.

</div>

### Applications and Theorems

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(DFT Complexity)</span></p>

If $n$ is a power of two, the Discrete Fourier Transform in $\mathbb{C}^n$ and its inverse can be computed in $\Theta(n \log n)$ time.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Fast Polynomial Multiplication)</span></p>

Polynomials of size $n$ over the field $\mathbb{C}$ can be multiplied in $\Theta(n \log n)$ time.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>First, the coefficient vectors of the two polynomials are padded with at least $n$ zeros, so that the total number of coefficients is a power of two. Then, using the DFT (computed via FFT) in $\Theta(n \log n)$ time, both polynomials are converted from their coefficient representations to their value representations (graphs). The graphs are multiplied component-wise in $\Theta(n)$ time. Finally, the resulting graph is converted back to the coefficient representation of the product polynomial using the inverse DFT in $\Theta(n \log n)$ time. $\square$</p>
  </details>
</div>

The Fourier transform has applications beyond polynomial multiplication. It is fundamental in other algebraic algorithms and in physics, where it corresponds to the spectral decomposition of a signal into sines and cosines of various frequencies. This principle underlies algorithms for audio filtering, audio and image compression (e.g., MP3, JPEG), and speech recognition.

---

## Spectral Decomposition

This section explores the connection between FFT and digital signal processing, focusing on one-dimensional signals like audio for simplicity.

Consider a real function $f$ defined on the interval $[0, 1)$. If we sample its values at $n$ regularly spaced points, we obtain a real vector $\mathbf{f} \in \mathbb{R}^n$ with components $f_j = f(j/n)$. We now ask what the Fourier image of $\mathbf{f}$ reveals about the function $f$.

### Properties of DFT for Real Vectors

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Conjugate Symmetry of Real DFT)</span></p>

If $\mathbf{x}$ is a real vector from $\mathbb{R}^n$, its Fourier image $\mathbf{y} = F(\mathbf{x})$ is conjugate symmetric: $y_j = \bar{y}_{n-j}$ for all $j$.

Specifically, $y_0 = \bar{y}\_0$ and $y_{n/2} = \bar{y}\_{n/2}$, which implies that both of these values are real.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>From the definition of the DFT, we know that:</p>
    $$y_{n-j} = \sum_k x_k \omega^{(n-j)k} = \sum_k x_k \omega^{nk - jk} = \sum_k x_k \omega^{-jk} = \sum_k x_k \overline{\omega^{jk}}$$
    <p>Since the complex conjugate distributes over arithmetic operations, we can write:</p>
    $$y_{n-j} = \overline{\sum_k x_k \omega^{jk}}$$
    <p>For a real vector $\mathbf{x}$, the components $x_k$ are equal to their conjugates ($\bar{x}_k = x_k$). Therefore:</p>
    $$y_{n-j} = \overline{\sum_k \bar{x}_k \omega^{jk}} = \overline{\sum_k x_k \omega^{jk}} = \bar{y}_j$$
    <p>The equality $y_j = \bar{y}_{n-j}$ follows by taking the conjugate of both sides. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Dimension of Conjugate Symmetric Vectors)</span></p>

The conjugate symmetric vectors in $\mathbb{C}^n$ form a vector space of dimension $n$ over the field of real numbers, $\mathbb{R}$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>We verify the vector space axioms. It is important that the space is constructed over $\mathbb{R}$ and not $\mathbb{C}$, as multiplying a conjugate symmetric vector by a complex scalar does not generally preserve the symmetry.</p>
    <p>Regarding the dimension: in a conjugate symmetric vector $\mathbf{y}$, the components $y_0$ and $y_{n/2}$ are real. For the components $y_1, \dots, y_{n/2-1}$, we can choose both their real and imaginary parts freely. The remaining components are then uniquely determined by the symmetry property. The vector is therefore determined by $2 + 2(n/2-1) = n$ independent real parameters. $\square$</p>
  </details>
</div>

### The Fourier Basis

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fourier Basis Vectors)</span></p>

For the remainder of this text, we fix $n$ and set $\omega = e^{2\pi i/n}$. We denote by $\mathbf{e}_k$, $\mathbf{s}_k$, and $\mathbf{c}_k$ the vectors obtained by sampling the functions $e^{2k\pi ix}$, $\sin(2k\pi x)$, and $\cos(2k\pi x)$ (the complex exponential, sine, and cosine with frequency $k$) at $n$ points in the interval $[0, 1)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Fourier Images of Basis Vectors)</span></p>

For $0 < k < n/2$, the Fourier images of the vectors $\mathbf{e}_k$, $\mathbf{s}_k$, and $\mathbf{c}_k$ are as follows:

* $F(\mathbf{e}_k) = (0, \dots, 0, n, 0, \dots, 0)$ where the non-zero element is at position $k$.
* $F(\mathbf{s}_k) = (0, \dots, 0, -\frac{n}{2}i, 0, \dots, 0, \frac{n}{2}i, 0, \dots, 0)$ where the non-zero elements are at positions $k$ and $n-k$.
* $F(\mathbf{c}_k) = (0, \dots, 0, \frac{n}{2}, 0, \dots, 0, \frac{n}{2}, 0, \dots, 0)$ where the non-zero elements are at positions $k$ and $n-k$.

While the formula for $F(\mathbf{e}\_k)$ works for $k=0$ and $k=n/2$, the sines and cosines behave differently at these boundaries. $\mathbf{s}\_0$ and $\mathbf{s}\_{n/2}$ are zero vectors, so their transforms are also zero. $\mathbf{c}\_0$ is a vector of all ones, with $F(\mathbf{c}\_0) = (n, 0, \dots, 0)$. The vector $\mathbf{c}\_{n/2}$ is $(1, -1, \dots, 1, -1)$, with $F(\mathbf{c}\_{n/2}) = (0, \dots, 0, n, 0, \dots, 0)$, with $n$ at position $n/2$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>For $\mathbf{e}_k$, we observe that the $j$-th component is $(e_k)_j = e^{2k\pi i \cdot j/n} = (\omega^k)^j = \omega^{jk}$. The $t$-th component of its Fourier image is:</p>
    $$\sum_j (\mathbf{e}_k)_j \omega^{-jt} = \sum_j \omega^{jk} \omega^{-jt} = \sum_j \omega^{j(k-t)}$$
    <p>This is a geometric series, similar to the one in the derivation of the inverse FFT. For $t=k$, it sums to $n$. For all other $t$, it sums to 0. The proofs for $\mathbf{s}_k$ and $\mathbf{c}_k$ are left as an exercise. $\square$</p>
  </details>
</div>

### The Discrete Fourier Series

Any conjugate symmetric vector can be formed by a real linear combination of the vectors $F(\mathbf{s}\_1), \dots, F(\mathbf{s}\_{n/2-1})$ and $F(\mathbf{c}\_0), \dots, F(\mathbf{c}\_{n/2})$. Since the DFT is a linear transformation, it follows that any real vector can be obtained by a linear combination of $\mathbf{s}\_1, \dots, \mathbf{s}\_{n/2-1}$ and $\mathbf{c}\_0, \dots, \mathbf{c}\_{n/2}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Discrete Fourier Series)</span></p>

For every vector $\mathbf{x} \in \mathbb{R}^n$, there exist real coefficients $\alpha_0, \dots, \alpha_{n/2}$ and $\beta_1, \dots, \beta_{n/2-1}$ such that:

$$\mathbf{x} = \sum_{k=0}^{n/2} \alpha_k \mathbf{c}_k + \sum_{k=1}^{n/2-1} \beta_k \mathbf{s}_k \quad (*)$$

Furthermore, these coefficients can be calculated from the Fourier image $\mathbf{y} = F(\mathbf{x}) = (a_0 + i b_0, \dots, a_{n-1} + i b_{n-1})$ as follows:

* $\alpha_0 = a_0/n$
* $\alpha_k = 2a_k/n \quad \text{for } k = 1, \dots, n/2$
* $\beta_k = -2b_k/n \quad \text{for } k = 1, \dots, n/2-1$

(Note that $\beta_0$ and $\beta_{n/2}$ are implicitly zero, corresponding to $\mathbf{s}\_0$ and $\mathbf{s}\_{n/2}$ being zero vectors.)

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Since the DFT is invertible, we can apply the Fourier transform to both sides of equation (*). We want to show that $\mathbf{y} = F(\sum_k \alpha_k \mathbf{c}_k + \sum_k \beta_k \mathbf{s}_k)$. Due to the linearity of $F$, the right side is equal to $\sum_k (\alpha_k F(\mathbf{c}_k) + \beta_k F(\mathbf{s}_k))$. Let's denote this vector by $\mathbf{z}$ and calculate its components using the Lemma on Fourier Images of Basis Vectors:</p>
    <ul>
      <li>The 0-th component $z_0$ receives a contribution only from $\mathbf{c}_0$. So, $z_0 = \alpha_0 F(\mathbf{c}_0)_0 = (a_0/n) \cdot n = a_0$.</li>
      <li>For $j = 1, \dots, n/2 - 1$, the component $z_j$ receives contributions only from $\mathbf{c}_j$ and $\mathbf{s}_j$:
        $$z_j = \alpha_j F(\mathbf{c}_j)_j + \beta_j F(\mathbf{s}_j)_j = \frac{2a_j}{n} \cdot \frac{n}{2} + \frac{-2b_j}{n} \cdot \left(-\frac{n}{2} i\right) = a_j + i b_j$$</li>
      <li>The component $z_{n/2}$ receives a contribution only from $\mathbf{c}_{n/2}$, so analogously, $z_{n/2} = \alpha_{n/2} F(\mathbf{c}_{n/2})_{n/2} = (2a_{n/2}/n) \cdot (n/2) = a_{n/2}$.</li>
    </ul>
    <p>The vectors $\mathbf{z}$ and $\mathbf{y}$ thus agree on their first $n/2+1$ components. Since both are conjugate symmetric (as linear combinations of such vectors), they must agree on all remaining components as well. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Spectral Decomposition of Functions)</span></p>

For any real function $f$ on the interval $[0, 1)$, there exists a linear combination of the functions $\sin(2k\pi x)$ and $\cos(2k\pi x)$ for $k = 0, \dots, n/2$ that is indistinguishable from the function $f$ when sampled at $n$ points.

</div>

This is the discrete equivalent of the well-known theorem from continuous Fourier analysis, which states that any "sufficiently smooth" periodic function can be expressed as a linear combination of sines and cosines with integer frequencies.

This is useful, for instance, in audio processing. Since $\alpha \cos x + \beta \sin x = A \sin(x + \phi)$ for suitable $A$ and $\phi$, any sound can be decomposed into sinusoidal tones of different frequencies. For each tone, we obtain its amplitude $A$ and phase shift $\phi$, which correspond (up to a scaling factor of $n$) to the absolute value and argument of the original complex Fourier coefficient. This is called the **spectral decomposition** of a signal, and thanks to the FFT, it can be calculated very quickly from a sampled signal.

---

## Other FFT Variants

### FFT as a Gate Network

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Circuit Interpretation)</span></p>

The execution of the FFT algorithm can be visualized graphically. The input vector is on the left, and the output vector is on the right. Reading the algorithm backwards, the output is computed from the results of "half-size" transforms of the even-indexed and odd-indexed inputs. The circles in the diagram correspond to the computation of a linear combination $a + \omega^k b$, known as a **butterfly** operation. The entire computation proceeds in $\log_2 n$ layers, each with $\Theta(n)$ operations.

This visualization can be seen as a schematic for a circuit or "gate network" that computes the DFT. The operations within a single layer are independent of each other, allowing for parallel computation. The network thus operates in $\Theta(\log n)$ time and $\Theta(n \log n)$ space. The permutation of the inputs on the left-hand side is known as the **bit-reversal permutation**.

</div>

### Non-Recursive FFT

The circuit diagram can be evaluated layer by layer from left to right. This gives rise to an elegant non-recursive algorithm for computing the FFT, which operates in $\Theta(n \log n)$ time and $\Theta(n)$ space.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Non-recursive Fast Fourier Transform — FFT2)</span></p>

**Input:**
* Complex numbers $x_0, \dots, x_{n-1}$.
* A primitive n-th root of unity, $\omega$.

**Algorithm:**
1. Precompute a table of values $\omega^0, \omega^1, \dots, \omega^{n-1}$.
2. For $k = 0, \dots, n-1$, set $y_k \leftarrow x_{r(k)}$, where $r$ is the bit-reversal function.
3. $b \leftarrow 1$ (initial block size)
4. While $b < n$:
   * For $j = 0, \dots, n-1$ with a step of $2b$: (start of a block)
     * For $k = 0, \dots, b-1$: (position within the block)
       * $\alpha \leftarrow \omega^{nk/(2b)}$
       * $(y_{j+k},\ y_{j+k+b}) \leftarrow (y_{j+k} + \alpha \cdot y_{j+k+b},\ y_{j+k} - \alpha \cdot y_{j+k+b})$
   * $b \leftarrow 2b$

**Output:**
* $y_0, \dots, y_{n-1}$

</div>

### FFT in Finite Fields

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(FFT over Finite Fields)</span></p>

The Fourier transform can be defined not only over the field of complex numbers but also in certain finite fields, provided a primitive n-th root of unity exists.

An algebraic theorem states that the multiplicative group of any finite field $\mathbb{Z}_p$ (the set of its $p-1$ non-zero elements with the operation of multiplication) is cyclic. This means that all $p-1$ non-zero elements can be written as powers of some number $g$, a generator of the group. Since every non-zero element appears exactly once among $g^0, g^1, \dots, g^{p-2}$, $g$ is a primitive $(p-1)$-th root of unity.

If we choose a prime $p$ such that $p-1$ is divisible by $n$ (where $n$ is a power of 2), then $\omega = g^{(p-1)/n}$ is a primitive n-th root of unity, and we can perform the FFT.

Practical values include:

* $p = 2^{16} + 1 = 65537$, $g = 3$. This works for $n=2^{16}$.
* $p = 15 \cdot 2^{27} + 1$, $g=31$. For $n=2^{27}$, we can use $\omega = g^{15} \pmod p$.
* $p = 3 \cdot 2^{30} + 1$, $g=5$. For $n=2^{30}$, we can use $\omega = g^3 \pmod p$.

A closer examination reveals that a field is not strictly necessary. Any commutative ring in which the required primitive root of unity exists, along with its multiplicative inverse and the multiplicative inverse of $n$, will suffice.

The advantage of these forms of the Fourier transform is that, unlike the classic complex version, they are not burdened by floating-point rounding errors. This is particularly useful in algorithms for multiplying large numbers.

</div>
