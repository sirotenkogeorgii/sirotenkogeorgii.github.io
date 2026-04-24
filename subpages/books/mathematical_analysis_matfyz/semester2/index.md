---
layout: default
title: Mathematical Analysis (Pultr) - Semester 2
date: 2026-03-16
excerpt: Notes from the second semester of A Course of Analysis for Computer Scientists by A. Pultr.
tags:
  - analysis
  - mathematics
---

**Table of Contents**
- TOC
{:toc}

# 2nd Semester

## IX. Polynomials and Their Roots

### 1. Polynomials

We are interested in real analysis but we will still need some basic facts about polynomials with coefficients and variables in the field $\mathbb{C}$ of complex numbers.

From Chapter I, 3.4, recall the absolute value $\lvert a \rvert = \sqrt{a_1^2 + a_2^2}$ of the complex number $a = a_1 + a_2 i$ and the triangle inequality

$$\lvert a + b \rvert \le \lvert a \rvert + \lvert b \rvert.$$

Further recall the complex conjugate $\overline{a} = a_1 - a_2 i$ of $a = a_1 + a_2 i$ and the facts that

$$\overline{a + b} = \overline{a} + \overline{b}, \quad \overline{ab} = \overline{a}\,\overline{b} \quad \text{and} \quad \lvert a \rvert = \sqrt{a\overline{a}}.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(1.1.1)</span></p>

$a + \overline{a}$ and $a\overline{a}$ are always real numbers.

</div>

**Degree of a polynomial.** If the coefficient $a_n$ in the polynomial

$$p \equiv a_n x^n + \cdots + a_1 x + a_0$$

is not $0$ we say that the degree of $p$ is $n$ and write $\deg(p) = n$. This leaves out $p = \text{const}_0$ which is usually not given a degree.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(1.2.1)</span></p>

$$\deg(pq) = \deg(p) + \deg(q).$$

</div>

#### 1.3. Dividing Polynomials

Consider polynomials $p, q$ with degrees $n = \deg(p) \ge k = \deg(q)$,

$$p \equiv a_n x^n + \cdots + a_1 x + a_0, \qquad q \equiv b_k x^k + \cdots + b_1 x + b_0.$$

Subtracting $\frac{a_n}{b_k} x^{n-k} q(x)$ from $p(x)$ we obtain zero or a polynomial $p_1$ with $\deg(p_1) < n$, and

$$p(x) = c_1 x^{n_1} q(x) + p_1(x).$$

Repeating this procedure we finish with

$$p(x) = s(x)q(x) + r(x)$$

with $r = \text{const}_0$ or $\deg(r) < \deg(q)$. One speaks of the $r$ as of the **remainder** when dividing $p$ by $q$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(1.3.1 - Important)</span></p>

If the coefficients of $p$ and $q$ are real then also the coefficients of $s$ and $r$ are real.

</div>

### 2. Fundamental Theorem of Algebra. Roots and Decomposition

A **root** of a polynomial $p$ is a number $x$ such that $p(x) = 0$. A polynomial with real coefficients does not have to have a real root (consider for example $p \equiv x^2 + 1$) but in the field of complex numbers we have

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Fundamental Theorem of Algebra)</span></p>

Each polynomial $p$ of $\deg(p) > 0$ with complex coefficients has a complex root.

</div>

#### 2.2. Decomposition of Complex Polynomials

Recall the obvious formula

$$x^k - \alpha^k = (x - \alpha)(x^{k-1} + x^{k-2}\alpha + \cdots + x\alpha^{k-2} + \alpha^{k-1})$$

and denote the polynomial $x^{k-1} + x^{k-2}\alpha + \cdots + x\alpha^{k-2} + \alpha^{k-1}$ (in $x$) of degree $k - 1$ by $s_k(x, \alpha)$. If $\alpha_1$ is a root of $p(x) = \sum_{k=0}^{n} a_k x^k$ of degree $n$ we have

$$p(x) = p(x) - p(\alpha_1) = \sum_{k=0}^{n} a_k x^k - \sum_{k=0}^{n} a_k \alpha_1^k = \sum_{k=0}^{n} a_k (x^k - \alpha_1^k) = (x - \alpha_1) \sum_{k=0}^{n} a_k s_k(x, \alpha_1)$$

where the polynomial $p_1(x) = \sum_{k=0}^{n} a_k s_k(x, \alpha)$ has by 1.2.1 degree precisely $n - 1$. Repeating the procedure we obtain

$$p_1(x) = (x - \alpha_2) p_2(x), \quad p_2(x) = (x - \alpha_3) p_3(x), \quad \text{etc.}$$

with $\deg(p_k) = n - k$, and ultimately

$$p(x) = a(x - \alpha_1)(x - \alpha_2) \cdots (x - \alpha_n) \qquad (*)$$

with $a \neq 0$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.3 - Maximum number of roots)</span></p>

A polynomial of degree $n$ has at most $n$ roots.

*Proof.* Let $x$ be a root of $p(x) = a(x - \alpha_1)(x - \alpha_2) \cdots (x - \alpha_n)$. Then $(x - \alpha_1)(x - \alpha_2) \cdots (x - \alpha_n) = 0$ and hence some of the $x - \alpha_k$ has to be zero, that is, $x = \alpha_k$. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.3.1 - Unicity of the coefficients)</span></p>

The coefficients $a_k$ in the expression $p(x) = a_n x^n + \cdots + a_1 x + a_0$ are uniquely determined by the function $(x \mapsto p(x))$. Consequently, this function also determines $\deg(p)$.

*Proof.* Let $p(x) = a_n x^n + \cdots + a_1 x + a_0 = b_n x^n + \cdots + b_1 x + b_0$ (any of $a_k, b_k$ may be zero). Then $(a_n - b_n) x^n + \cdots + (a_1 - b_1) x + (a_0 - b_0)$ has infinitely many roots and hence cannot have a degree. Thus, $a_k = b_k$ for all $k$. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.3.2 - Unicity of division)</span></p>

The polynomials $s, r$ obtained when dividing polynomial $p$ by a polynomial $q$ as in 1.3 are uniquely determined.

*Proof.* Let $p(x) = s_1(x)q(x) + r_1(x) = s_2(x)q(x) + r_2(x)$. Then $q(x)(s_1(x) - s_2(x)) + (r_1(x) - r_2(x))$ is a zero polynomial and since $\deg(q) > \deg(r_1 - r_2)$ (if the last is determined at all), $s_1 = s_2$. Then $r_1 - r_2 \equiv 0$ and hence also $r_1 = r_2$. $\square$

</div>

#### 2.4. Multiple Roots

On the other hand, $p(x)$ does not have to have $\deg(p)$ many distinct roots: see for instance $p(x) = x^n$ with only one root, namely zero. The roots $\alpha_k$ in the decomposition $(*)$ can appear several times, and after suitable permutation of the factors, $(*)$ can be rewritten as

$$p(x) = a(x - \beta_1)^{k_1}(x - \beta_2)^{k_2} \cdots (x - \beta_r)^{k_r} \quad \text{with } \beta_k \text{ distinct.} \qquad (**)$$

The power $k_j$ is called **multiplicity** of the root $\beta_j$ and we have $\sum_{j=1}^{r} k_j = n$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.4.1 - Unicity of multiplicity)</span></p>

The multiplicity of a root is uniquely defined. Consequently, the decomposition $(**)$ is determined up to the permutation of the factors.

*Proof.* Suppose we have $p(x) = (x - \beta)^k q(x) = (x - \beta)^\ell r(x)$ such that $\beta$ is not a root of neither $q$ nor $r$. Suppose $k < \ell$. Dividing $p(x)$ by $(x - \beta)^k$ we obtain (using the unicity of division, see 2.3.2 above) that $q(x) = (x = \beta)^{\ell - k} r(x)$ so that $\beta$ is a root of $p$, a contradiction. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(2.5 - Integral domain of polynomials)</span></p>

The set of all complex polynomials forms an integral domain (similarly like the set of integers). Now $q \mid p$ ($q$ divides $p$) if $p(x) = s(x)q(x)$ and both $q \mid p$ and $q \mid p$ iff there is a number $c \neq 0$ such that $p(x) = c \cdot q(x)$. The primes in this division are the (equivalence classes of) binoms $x - \alpha$. In the propositions above we have seen that in the integral domain of complex polynomials we have unique prime decomposition.

</div>

### 3. Decomposition of Polynomials with Real Coefficients

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.1 - Conjugate roots)</span></p>

Let the coefficients $a_n$ of a polynomial $p(x) = a_n x^n + \cdots + a_1 x + a_0$ be real. Let $\alpha$ be a root of $p$. Then the complex conjugate $\overline{\alpha}$ is also a root of $p$.

*Proof.* We have (recall 1.1) $p(\overline{\alpha}) = a_n \overline{\alpha}^n + \cdots + a_1 \overline{\alpha} + a_0 = \overline{a_n} \overline{\alpha}^n + \cdots + \overline{a_1} \overline{\alpha} + \overline{a_0} = \overline{a_n \alpha^n + \cdots + a_1 \alpha + a_0} = \overline{0} = 0.$ $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.2 - Conjugate multiplicity)</span></p>

Let $\alpha$ be a root of multiplicity $k$ of a polynomial $p$ with real coefficients. Then the multiplicity of the root $\overline{\alpha}$ is also $k$.

*Proof.* If $\alpha$ is real there is nothing to prove. Now let $\alpha$ not be real. Then we have

$$p(x) = (x - \alpha)(x - \overline{\alpha})q(x) = (x^2 - (\alpha + \overline{\alpha})x + \alpha\overline{\alpha})q(x)$$

and since $x^2 - (\alpha + \overline{\alpha})x + \alpha\overline{\alpha}$ has real coefficients (recall 1.1.1), $q$ also has real coefficients (recall 1.3.1). Now if $\alpha$ is a root of $q$ again we have another root $\overline{\alpha}$ of $q$, and the statement follows inductively. $\square$

</div>

**Irreducible trinomials.** The trinomials $x^2 + \beta x + \gamma = x^2 - (\alpha + \overline{\alpha})x + \alpha\overline{\alpha}$ have no real roots: they already have roots $\alpha$ and $\overline{\alpha}$, and cannot have more by 2.3. They are called **irreducible** trinomials.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(3.4.1 - Real polynomial decomposition)</span></p>

Let $p$ be a polynomial of degree $n$ with real coefficients. Then

$$p(x) = a(x - \beta_1)^{k_1}(x - \beta_2)^{k_2} \cdots (x - \beta_r)^{k_r}(x^2 + \gamma_1 x + \delta_1)^{\ell_1} \cdots (x^2 + \gamma_s x + \delta_s)^{\ell_s}$$

with $\beta_j, \gamma_j, \delta_j$ real, $x^2 + \gamma_j x + \delta_j$ irreducible and $\sum_{j=1}^{r} k_j + 2\sum_{j=1}^{s} \ell_j = n$ ($s$ can be equal to $0$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(3.4.1 - Primes in real polynomials)</span></p>

Thus, in the integrity domain of real polynomials we have a greater variety of primes. Besides the $x - \beta$ we also have the irreducible $x^2 + \gamma x + \delta$.

</div>

### 4. Sum Decomposition of Rational Functions

We have already used the term **integral domain** in Notes 2.5 and 3.4.1. To be more specific, an integral domain is a commutative ring $J$ with unit 1 and such that for $a, b \in J$, $a, b \neq 0$ implies $ab \neq 0$.

As in the domain $\mathbb{Z}$ of integers, in a general integral domain (and in particular in the domain of polynomials with coefficients in $\mathbb{C}$ resp. $\mathbb{R}$) we say that $a$ divides $b$ and write $a \mid b$ if there is an $x$ such that $b = xa$. Elements $a$ and $b$ are equivalent if $a \mid b$ and $b \mid a$; we write $a \sim b$.

The **greatest common divisor** $a, b$ is a $d$ such that $d \mid a$ and $d \mid b$ and such that whenever $x \mid a$ and $x \mid b$ then $x \mid d$. The unit divides every $a$; elements $a$ and $b$ are **coprime** (or **relatively prime**) if they have (up to equivalence) no other common divisor.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(4.2 - Bézout-type theorem)</span></p>

Let $J$ be an integrity domain and let us have a function $\nu : J \to \mathbb{N}$ and a rule of division with remainder for $a, b \neq 0$ and $b$ not dividing $a$,

$$a = sb + r \quad \text{with} \quad \nu(r) > \nu(b).$$

Then for any $a, b \neq 0$ there exist $x, y$ such that $xa + yb$ is the greatest common divisor of $a, b$.

*Proof.* Let $d = xa + yb$ with the least possible $\nu(d)$. Suppose $d$ does not divide $a$. Then

$$a = sd + r \quad \text{with} \quad \nu(r) < \nu(d).$$

But then $(1 - sx)a - syb = r$ and $\nu((1 - sx)a - syb) = \nu(r) < \nu(d)$, a contradiction. Thus, $d \mid a$ and for the same reason $d \mid b$. On the other hand, if $c \mid a$ and $c \mid b$ then obviously $c \mid (xa + yb)$. Thus, $d$ is the greatest common divisor. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(4.2.1)</span></p>

For the integrity domain of integers (with $\nu(n) = \lvert n \rvert$) this was proved by Bachet (16.–17. century), in the more general form -- in particular for our polynomials -- this is by Bézout (18. century). One usually speaks of Bézout lemma; Bachet-Bézout Theorem should be appropriate.

</div>

A **rational function** (in one variable) is a complex or real function of one (complex resp. real) variable that can be written as

$$P(x) = \frac{p(x)}{q(x)}$$

where $p, q$ are polynomials.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(4.3.1 - Partial fraction decomposition, complex case)</span></p>

A complex rational function $P(x) = \frac{p(x)}{q(x)}$ can be written as

$$P_1(x) + \sum_j V_j(x)$$

where each of the expression is of the form

$$\frac{A}{(x - \alpha)^k}$$

where $A$ is a number and $\alpha$ is a root of the polynomial $q$ with multiplicity at least $k$.

*Proof.* By induction on $\deg(q)$. The statement is trivial for $\deg(q) = 0$. For $\deg(q) = 1$ (and hence $q(x) = C(x - \alpha)$) we obtain from 1.3 that $p(x) = s(x)q(x) + B$ and

$$\frac{p(x)}{q(x)} = s(x) + \frac{B'}{x - \alpha} \quad \text{where} \quad B' = \frac{B}{C}.$$

Now let the theorem hold for $\deg(q) < n$. It suffices to prove it for $\frac{p(x)}{(x - \alpha)q(x)}$ with $\deg q < n$. This can be written, by the induction hypothesis as

$$\frac{P_1(x)}{x - \alpha} + \sum_j \frac{V_j(x)}{x - \alpha}.$$

If $V_j = \frac{A}{(x - \alpha)^k}$ the corresponding summand will be $\frac{A}{(x - \alpha)^{k+1}}$. If it is $\frac{A}{(x - \beta)^k}$ with $\beta \neq \alpha$ we realize first that the greatest common divisor of $(x - \alpha)$ and $(x - \beta)^k$ is 1 and hence by 4.2 there are polynomials $u, v$ such that

$$u(x)(x - \alpha) + v(x)(x - \beta)^k = 1$$

so that

$$\frac{A}{(x - \alpha)(x - \beta)^k} = \frac{Au(x)}{(x - \beta)^k} + \frac{Av(x)}{(x - \alpha)}$$

and by induction hypothesis both of the last summands can be written as desired. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(4.3.2 - Partial fraction decomposition, real case)</span></p>

A real rational function $P(x) = \frac{p(x)}{q(x)}$ can be written as

$$P_1(x) + \sum_j V_j(x)$$

where each of the expression is of the form

$$\frac{A}{(x - \alpha)^k}$$

where $A$ is a number and $\alpha$ is a root of $q$ with multiplicity at least $k$, or of the form

$$\frac{Ax + B}{(x^2 + ax + b)^k}$$

where $x^2 + ax + b$ is some of the irreducible trinomials from 3.4.1 and $k$ is less or equal to the corresponding $\ell$.

*Proof* can be done following the lines of the proof of 4.3.1, only distinguishing more cases of the relative primeness of the $x - \alpha$ and $x^2 + ax + b$. With careful checking it can also be deduced from 4.3.1: namely, if a root $\alpha$ is not real we have to have with each $\frac{A}{(x - \alpha)^k}$ a summand $\frac{B}{(x - \overline{\alpha})^k}$ with the same power $k$: else, the sum would not be real. Now we have

$$\frac{A}{(x - \alpha)^k} + \frac{B}{(x - \overline{\alpha})^k} = \frac{A(x - \overline{\alpha}) + B(x - \alpha)}{(x^2 - (\alpha + \overline{\alpha})x + \alpha\overline{\alpha})^k} = \frac{A_1 x + B_1}{(x^2 + ax + b)^k}$$

and again we have to check that the $A_1, B_1$ have to be real. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(4.3.3 - Practical computation)</span></p>

In practical computing one simply takes into account that the expression as in 4.3.1 or 4.3.2 is possible and obtains the coefficients $A$ resp. $A$ and $B$ as solutions of a system of linear equations.

</div>

## X. Primitive Function (Indefinite Integral)

### 1. Reversing Differentiation

In Chapter VI we defined a derivative of a function and learned how to compute the derivatives of elementary functions. Now we will reverse the task. Given a function $f$ we will be interested in a function $F$ such that $F' = f$. Such a function $F$ will be called the **primitive function**, or **indefinite integral** of $f$ (in the next chapter we will discuss a basic definite one, the Riemann integral).

In differentiation we had, first, a derivative of a function at a point, which was a number, and then we defined a derivative of a function $f$ as a function $f': D \to \mathbb{R}$, provided $f$ had a derivative $f'(x)$ in every point $x$ of a domain $D$. In taking the primitive function we have nothing like the former. It will be always a search of a function (the $F$ above) associated with a given one.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Fact</span><span class="math-callout__name">(1.2.1 - Uniqueness up to a constant)</span></p>

If $F$ and $G$ are primitive functions of $F$ on an interval $J$ then there is a constant $C$ such that

$$F(x) = G(x) + C$$

for all $x \in J$.

</div>

Unlike a derivative $f'$ that is uniquely determined by the function $f$, the primitive function is not, for obvious reasons: the derivative of a constant $C$ is zero so that if $F(x)$ is a primitive function of $f(x)$ then so is any $F(x) + C$. But the situation is not much worse than that, as we have already proved in VIII.3.3.

**Notation.** Primitive function of a function $f$ is often denoted by

$$\int f$$

Instead of this concise symbol we equally often use a more explicit

$$\int f(x)\mathrm{d}x.$$

This latter is not just an elaborate indication of what the variable in question is. In Section 4 it will be of a great advantage in computing an integral by means of the substitution method.

Since a primitive function is not uniquely determined, the expression "$F = \int f$" should be understood as "$F$ is a primitive function of $f$", not as an equality of two entities. To be safer one usually writes

$$\int f(x)\mathrm{d}x = F(x) + C \quad \text{or} \quad \int f = F(x) + C,$$

but even this can be misleading: the statement 1.2.1 holds for an interval only and the domains of very natural functions are not always intervals; see 2.2.2.2 below. One has to be careful.

### 2. A Few Simple Formulas

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.1 - Linearity of integration)</span></p>

Let $f, g$ be functions with the same domain $D$ and let $a, b$ be numbers. Let $\int f$ and $\int g$ exist on $D$. Then $\int (af + bg)$ exists and we have

$$\int (af + bg) = a \int f + b \int g.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(2.1.1)</span></p>

This is the only arithmetic rule for integration. For principial reasons there cannot be a formula for $\int f(x)g(x)\mathrm{d}x$ or for $\int \frac{f(x)}{g(x)}\mathrm{d}x$, see 2.2.2.1 and 2.3.1.

</div>

**Basic integration formulas.** Reversing the rule for differentiating $x^n$ with $n \neq -1$ we obtain

$$\int x^n \mathrm{d}x = \frac{1}{n+1} x^{n+1}.$$

(In fact, this does not hold for integers $n$ only. If $D$ is $\lbrace x \in \mathbb{R} \mid x > 0 \rbrace$ then we have by VI.3.3 the formula $\int x^a \mathrm{d}x = \frac{1}{a+1} x^{a+1}$ for any real $a \neq -1$.)

Hence, using 2.1 we have for a polynomial $p(x) = \sum_{k=0}^{n} a_k x^k$,

$$\int p(x)\mathrm{d}x = \sum_{k=0}^{n} \frac{a_k}{k+1} x^{k+1}.$$

For $n = -1$ (and domain $\mathbb{R} \setminus \lbrace 0 \rbrace$) we have the formula

$$\int \frac{1}{x}\mathrm{d}x = \lg \lvert x \rvert.$$

(Indeed, for $x > 0$ we have $\lvert x \rvert = x$ and hence $(\lg \lvert x \rvert)' = \frac{1}{x}$. For $x < 0$ we have $\lvert x \rvert = -x$ and hence $(\lg \lvert x \rvert)' = (\lg(-x))' = \frac{1}{-x} \cdot (-1) = \frac{1}{x}$ again.)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(2.2.2 - Warning about domains)</span></p>

1. The last formula indicates that there can hardly be a simple rule for integration $\frac{f(x)}{g(x)}$ in terms of $\int f$ and $\int g$: this would mean an arithmetic formula producing $\lg x$ from $x = \int 1$ and $\frac{1}{2}x^2 = \int x$.

2. The domain of function $\frac{1}{x}$ is not an interval. Note that we have, a.o.,

$$\int \frac{1}{x}\mathrm{d}x = \begin{cases} \lg \lvert x \rvert + 2 & \text{for } x < 0, \\ \lg \lvert x \rvert + 5 & \text{for } x > 0. \end{cases}$$

which shows that using the expression $\int f(x)\mathrm{d}x = F(x) + C$ is not without danger.

</div>

For goniometric functions we immediately obtain

$$\int \sin x = -\cos x \quad \text{and} \quad \int \cos x = \sin x.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(2.3.1 - Non-elementary primitives)</span></p>

In general, a primitive function of an elementary function (although it always exists as we will see in the next chapter) may not be elementary. One such is

$$\int \frac{\sin x}{x}$$

(proving this is far beyond our means, you have to believe it). Now we have an easy $\int \frac{1}{x}$ and $\int \sin x$; thus there cannot be a rule for computing $\int f(x)g(x)\mathrm{d}x$ in terms of $\int f$ and $\int g$.

</div>

For the exponential we have, trivially,

$$\int e^x \mathrm{d}x = e^x \quad \text{and by VI.3.3 more generally} \quad \int a^x \mathrm{d}x = \frac{1}{\lg a} a^x.$$

Let us add two more obvious formulas:

$$\int \frac{\mathrm{d}x}{1 + x^2} = \arctan x \quad \text{and} \quad \int \frac{\mathrm{d}x}{\sqrt{1 - x^2}} = \arcsin x.$$

---

In the following two sections we will learn two useful methods for finding primitive functions in more involved cases.

### 3. Integration Per Partes

Let $f, g$ have derivatives. From the rule of differentiating products we immediately obtain

$$\int f' \cdot g = f \cdot g - \int f \cdot g'. \qquad (*)$$

At the first sight we have not achieved much: we wish to integrate the product $f' \cdot g$ and we are left with integrating a similar one, $f \cdot g'$. But

1. $\int f \cdot g'$ can be much simpler than $\int f' \cdot g$, or
2. the formula can result in an equation from which the desired integral can be easily computed, or
3. the formula may yield a recursive one that leads to our goal.

Using the formula $(*)$ is called **integration per partes**.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(3.2 - Illustration of 3.1.(1): $\int x^a \lg x$)</span></p>

Let us compute $J = \int x^a \lg x$ with $x > 0$ and $a \neq -1$.

If we set $f(x) = \frac{1}{a+1} x^{a+1}$ and $g(x) = \lg x$ we obtain $f'(x) = x^a$ and $g'(x) = \frac{1}{x}$ so that

$$J = \frac{1}{a+1} x^{a+1} \lg x - \frac{1}{a+1} \int x^{a+1} \cdot \frac{1}{x} = \frac{1}{a+1}(x^{a+1} \lg x - \int x^a) = \frac{x^{a+1}}{a+1}\left(\lg x - \frac{1}{a+1}\right)$$

and hence for instance for $a = 1$ we obtain

$$\int \lg x \mathrm{d}x = x(\lg x - 1).$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(3.3 - Illustration of 3.1.(2): $\int e^x \sin x$)</span></p>

Let us compute $J = \int e^x \sin x \mathrm{d}x$.

Setting $f(x) - f'(x) = e^x$ and $g(x) = \sin x$ we obtain

$$J = e^x \sin x - \int e^x \cos x \mathrm{d}x.$$

Now the new integral on the left hand side is about as complex as the one we have started with. But let us repeat the procedure, this time with $g(x) = \cos x$. We obtain

$$\int e^x \cos x \mathrm{d}x = e^x \cos x - \int e^x(-\sin x)\mathrm{d}x$$

and hence

$$J = e^x \sin x - (e^x \cos x - \int e^x(-\sin x)\mathrm{d}x) = e^x \sin x - e^x \cos x - J$$

and conclude that

$$J = \frac{e^x}{2}(\sin x - \cos x).$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(3.4 - Illustration of 3.1.(3): $\int x^n e^x$)</span></p>

Let us compute $J_n = \int x^n e^x \mathrm{d}x$ for integers $n \ge 0$.

Setting $f(x) = x^n$ and $g(x) = g'(x) = e^x$ we obtain

$$J_n = x^n e^x - \int n x^{n-1} e^x = x^n e^x - n J_{n-1}.$$

Iterating the procedure we get

$$J_n = x^n e^x - n x^{n-1} + n(n-1) J_{n-2} = \cdots = x^n e^x - n x^{n-1} + n(n-1) x^{n-2} e^x + \cdots \pm n! J_0$$

and since $J_0 = \int e^x = e^x$ this makes

$$J_n = e^x \cdot \sum_{k=0}^{n} \frac{n!}{(n-k)!} (-1)^k \cdot x^{n-k}.$$

</div>

### 4. Substitution Method

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Fact</span><span class="math-callout__name">(4.1 - Substitution rule)</span></p>

Let $\int f = F$, let a function $\phi$ have derivative $\phi'$, and let the composition $F \circ \phi$ make sense. Then

$$\int f(\phi(x)) \cdot \phi'(x) \mathrm{d}x = F(\phi(x)).$$

</div>

Thus, to obtain $\int f(\phi(x)) \cdot \phi'(x)\mathrm{d}x$ we compute $\int f(y)\mathrm{d}y$ and in the result substitute $\phi(x)$ for all the occurrences of $y$. Using this trick is called the **substitution method**.

Here the notation $\int f(x)\mathrm{d}x$ instead of the plain $\int f$ is of a great help. Recall the notation $\frac{\mathrm{d}\phi(x)}{\mathrm{d}x}$ for the derivative $\phi'(x)$. Now the expression $\frac{\mathrm{d}\phi(x)}{\mathrm{d}x}$ is not really a fraction with numerator $\mathrm{d}\phi(x)$ and denominator $\mathrm{d}x$, but let us pretend for a moment it is. Thus,

$$\mathrm{d}\phi(x) = \phi'(x)\mathrm{d}x \quad \text{or} \quad \text{"}\mathrm{d}y = \phi'(x)\mathrm{d}x \text{ where } \phi(x) \text{ is substituted for } y\text{."}$$

Hence, using the substitution method (substituting $\phi(x)$ for $y$) consists of computing

$$\int f(y)\mathrm{d}y$$

as an integral in variable $y$, and when substituting $\phi(x)$ for $y$ writing

$$\mathrm{d}y = \phi'(x)\mathrm{d}x \quad \text{as obtained from} \quad \frac{\mathrm{d}y}{\mathrm{d}x} = \phi'(x).$$

This is very easy to remember.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(4.2 - $\int \frac{\lg x}{x}\mathrm{d}x$)</span></p>

To determine $\int \frac{\lg x}{x}\mathrm{d}x$ substitute $y = \lg x$. Then $\mathrm{d}y = \frac{\mathrm{d}x}{x}$ and we obtain

$$\int \frac{\lg x}{x}\mathrm{d}x = \int y\mathrm{d}y = \frac{1}{2}y^2 = \frac{1}{2}(\lg x)^2.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(4.3 - $\int \tan x \mathrm{d}x$)</span></p>

To compute $\int \tan x \mathrm{d}x$ recall that $\tan x = \frac{\sin x}{\cos x}$ and that $(-\cos x)' = \sin x$. Hence, substituting $y = -\cos x$ we obtain

$$\int \tan x \mathrm{d}x = \int \frac{\sin x}{\cos x}\mathrm{d}x = \int \frac{\mathrm{d}y}{-y} = -\lg \lvert y \rvert = -\lg \lvert \cos x \rvert.$$

</div>

### 5. Integrals of Rational Functions

In view of 2.1 and IX.4.3.2 it suffices to find the integrals

$$\int \frac{1}{(x - a)^k}\mathrm{d}x \qquad (5.1.1)$$

and

$$\int \frac{Ax + B}{(x^2 + ax + b)^k}\mathrm{d}x \quad \text{with } x^2 + ax + b \text{ irreducible} \qquad (5.1.2)$$

for natural numbers $k$.

The first, (5.1.1), is very easy. If we substitute $y = x - a$ then $\mathrm{d}y = \mathrm{d}x$ and we compute our integral as $\int \frac{1}{y^k}$ and by 2.2 and 2.2.1 (substituting back $x - a$ for $y$)

$$\int \frac{1}{(x - a)^k}\mathrm{d}x = \begin{cases} \frac{1}{1-k} \cdot \frac{1}{(x-a)^{k-1}} & \text{for } k \neq 1, \\ \lg \lvert x - a \rvert & \text{for } k = 1. \end{cases}$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(5.3 - Reduction of $\int \frac{Ax+B}{(x^2+ax+b)^k}$)</span></p>

Set

$$J(a, b, x, k) = \int \frac{1}{(x^2 + ax + b)^k}\mathrm{d}x.$$

Then we have

$$\int \frac{Ax + B}{(x^2 + ax + b)^k}\mathrm{d}x = \begin{cases} \frac{A}{2(1-k)} \cdot \frac{1}{(x^2+ax+b)^{k-1}} + (B - \frac{Aa}{2})J(a, b, x, k) & \text{for } k \neq 1, \\ \frac{A}{2}\lg\lvert x^2 + ax + b \rvert + (B - \frac{Aa}{2})J(a, b, x, k) & \text{for } k = 1. \end{cases}$$

*Proof.* We have

$$\frac{Ax + B}{x^2 + ax + b} = \frac{A}{2} \frac{2x + a}{x^2 + ax + b} + \left(B - \frac{Aa}{2}\right) \frac{1}{x^2 + ax + b}$$

Now in the first we can compute $\int \frac{2x + a}{x^2 + ax + b}\mathrm{d}x$ substituting $y = x^2 + ax + b$; then we have $\mathrm{d}y = (2x + a)\mathrm{d}x$ and the task, as in 5.2, reduces to determining $\int \frac{1}{y^k}\mathrm{d}y$. $\square$

</div>

Hence, (5.1.2) will be solved by computing $\int \frac{1}{(x^2 + ax + b)^k}\mathrm{d}x$ with irreducible $x^2 + ax + b$.

First observe that because of the irreducibility we have $b - \frac{a^2}{4} > 0$ (otherwise, $x^2 + ax + b$ would have real roots). Therefore we have a real $c$ with $c^2 = b - \frac{a^2}{4}$ and

$$x^2 + ax + b = c^2\left(\left(\frac{x + \frac{1}{2}a}{c}\right)^2 + 1\right).$$

Thus, if we substitute $y = \frac{x + \frac{1}{2}a}{c}$ (hence, $\mathrm{d}y = \frac{1}{c}\mathrm{d}x$) in $\int \frac{1}{(x^2+ax+b)^k}\mathrm{d}x$ we obtain

$$\frac{1}{c^{2k-1}} \int \frac{1}{(y^2 + 1)^k}\mathrm{d}y$$

and we have further reduced our task to finding $\int \frac{1}{(x^2+1)^k}\mathrm{d}x$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.4.2 - Recursive formula for $J_k = \int \frac{1}{(x^2+1)^k}\mathrm{d}x$)</span></p>

The integral

$$J_k = \int \frac{1}{(x^2 + 1)^k}\mathrm{d}x$$

can be computed recursively from the formula

$$J_{k+1} = \frac{1}{2k} \cdot \frac{x}{x^2 + 1} + \frac{2k - 1}{2k} J_k \qquad (*)$$

with $J_1 = \arctan x$.

*Proof.* First set $f(x) = \frac{1}{(x^2 + 1)^k}$ and $g(x) = x$. Then $f'(x) = -k\frac{2x}{(x^2+1)^{k+1}}$ and $g'(x) = 1$ and from the per partes formula we obtain

$$J_k = \frac{x}{(x^2+1)^k} + 2k \int \frac{x^2}{(x^2+1)^{k+1}} = \frac{x}{(x^2+1)^k} + 2k\left(\int \frac{x^2 + 1}{(x^2+1)^{k+1}} - \int \frac{1}{(x^2+1)^{k+1}}\right) = \frac{x}{(x^2+1)^k} + 2kJ_k - 2kJ_{k+1}$$

and the formula $(*)$ follows; the $J_1 = \arctan x$ was already mentioned in 2.5. $\square$

</div>

### 6. A Few Standard Substitutions

First let us extend the terminology from Chapter IX. An expression $\sum_{r,s \le n} a_{rs} x^r y^s$ will be called a **polynomial in two variables** $x, y$. If $p(x,y), q(x,y)$ are polynomials in two variables we speak of

$$R(x, y) = \frac{p(x, y)}{q(x, y)}$$

as of **rational function in two variables**.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(6.1.2)</span></p>

Let $P(x), Q(x)$ be rational functions as in Chapter IX. Then $S(x) = R(P(x), Q(x))$ is a rational function.

</div>

#### 6.2. The integral $\int R\big(x, \sqrt{\frac{ax+b}{cx+d}}\big)\mathrm{d}x$

Substitute $y = \sqrt{\frac{ax+b}{cx+d}}$. Then $y^2 = \frac{ax+b}{cx+d}$ from which we obtain

$$x = \frac{b - dy^2}{ay^2 + a}$$

and hence $\frac{\mathrm{d}x}{\mathrm{d}y} = S(y)$ where $S(y)$ is a rational function. Hence, the substitution transforms

$$\int R\left(x, \sqrt{\frac{ax+b}{cx+d}}\right)\mathrm{d}x \quad \text{to} \quad \int R\left(\frac{b - dy^2}{ay^2 + a}, y\right)S(y)\mathrm{d}y$$

and this we can compute using the procedures from the previous section.

#### 6.3. Euler Substitution: the integral $\int R(x, \sqrt{ax^2 + bx + c})\mathrm{d}x$

First let us dismiss the case of $a \le 0$. Since we assume that the function makes sense, we have to have $ax^2 + bx + c \ge 0$ on its domain which implies (in case of $a \le 0$) real roots $\alpha, \beta$ and

$$R(x, \sqrt{ax^2 + bx + c}) = R(x, \sqrt{-a}\sqrt{(x - \alpha)(x - \beta)}) = R\left(x, \sqrt{-a}(x - \alpha)\sqrt{\frac{x - \beta}{x - \alpha}}\right)$$

and this is a case already dealt with in 5.2.

But if $a > 0$ the situation is new. Then let us substitute the $t$ from the equation

$$\sqrt{ax^2 + bx + c} = \sqrt{a}x + t$$

(this is the **Euler substitution**). The squares of both sides yield

$$ax^2 + bx + c = ax^2 + 2\sqrt{a}xt + t^2$$

and we obtain

$$x = \frac{t^2 - c}{b - 2t\sqrt{a}} \quad \text{and hence} \quad \frac{\mathrm{d}x}{\mathrm{d}t} = S(t)$$

where $S(t)$ is a rational function. Thus we can compute our integral as

$$\int R\left(\frac{t^2 - c}{b - 2t\sqrt{a}},\, \sqrt{a}\frac{t^2 - c}{b - 2t\sqrt{a}} + t\right) S(t)\mathrm{d}t.$$

#### 6.4. Goniometric Functions in a Rational One: $\int R(\sin x, \cos x)\mathrm{d}x$

To compute $\int R(\sin x, \cos x)\mathrm{d}x$ we will be helped by the substitution

$$y = \tan \frac{x}{2}.$$

Recall the standard formula $\cos^2 x = \frac{1}{1 + \tan^2 x}$ from which we obtain

$$\sin x = 2\sin\frac{x}{2}\cos\frac{x}{2} = 2\tan\frac{x}{2}\cos^2\frac{x}{2} = \frac{2\tan\frac{x}{2}}{1 + \tan^2\frac{x}{2}} = \frac{2y}{1 + y^2},$$

$$\cos x = \cos^2\frac{x}{2} - \sin^2\frac{x}{2} = 2\cos^2\frac{x}{2} - 1 = \frac{2}{1 + y^2} - 1 = \frac{1 - y^2}{1 + y^2}.$$

Further we have

$$\frac{\mathrm{d}y}{\mathrm{d}x} = \frac{1}{2} \cdot \frac{1}{\cos^2\frac{x}{2}} = \frac{1}{2} \cdot (1 + \tan^2\frac{x}{2}) = \frac{1}{2}(1 + y^2)$$

and hence

$$\mathrm{d}x = \frac{2}{1 + y^2}\mathrm{d}y$$

so that we can solve our task by computing

$$\int R\left(\frac{2y}{1 + y^2},\, \frac{1 - y^2}{1 + y^2}\right) \frac{2}{1 + y^2}\mathrm{d}y.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(6.5)</span></p>

The procedures in Section 4 and Section 5 are admittedly very laborious and time consuming. This is because they should cover fairly general cases. In a concrete case we sometimes can find a combination of the per partes and substitution methods leading to our goal in a much shorter procedure. Compare for instance $\int \tan x \mathrm{d}x$ as computed in 4.3 with 5.4.

</div>

## XI. Riemann Integral

### 1. The Area of a Planar Figure

Let us denote by $\mathsf{vol}(M)$ the area of a planar figure $M \subseteq \mathbb{R}^2$. A figure may be too exotic to be assigned an area, but we will not work with such here. Using the symbol $\mathsf{vol}$ includes the claim that the area in question makes sense.

The reader may wonder why we use the abbreviation $\mathsf{vol}$ and not something like "ar". This is because later we will work in higher dimensions and referring to $M \subseteq \mathbb{R}^n$ with general $n$, "volume" is used rather than "area".

The following are rules we can certainly easily agree upon.

1. $\mathsf{vol}(M) \ge 0$ whenever it makes sense,
2. if $M \subseteq N$ then $\mathsf{vol}(M) \le \mathsf{vol}(N)$,
3. if $M$ and $N$ are disjoint then $\mathsf{vol}(M \cup N) = \mathsf{vol}(M) + \mathsf{vol}(N)$, and
4. if $M$ is a rectangle with sides $a, b$ then $\mathsf{vol}(M) = a \cdot b$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(1.3)</span></p>

1. $\mathsf{vol}(\emptyset) = 0$.
2. Let $M$ be a segment. Then $\mathsf{vol}(M) = 0$.

*Proof.* 1: $\emptyset$ is a subset of any rectangle, hence the statement follows from (1),(2) and (4).

2 follows similarly: a segment of length $a$ is a subset of a rectangle with sides $a, b$ with arbitrarily small positive $b$. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(1.3.1)</span></p>

Thus we see that it was not necessary to specify whether we included in 1.2(4) the border segments, or just parts of them.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(1.4 - Inclusion-exclusion for areas)</span></p>

If the areas make sense we have

$$\mathsf{vol}(M \cup N) = \mathsf{vol}(M) + \mathsf{vol}(N) - \mathsf{vol}(M \cap N).$$

In particular we have

$$\mathsf{vol}(M \cup N) = \mathsf{vol}(M) + \mathsf{vol}(N) \quad \text{whenever} \quad \mathsf{vol}(M \cap N) = 0.$$

*Proof.* Follows from 1.2(4) taking into account the disjoint unions $M \cup N = M \cup (N \setminus M)$ and $N = (N \setminus M) \cup (N \cap M)$. $\square$

</div>

**Step figures.** In the sequel the areas of figures of the type consisting of unions of rectangles with sides parallel to the axes (step figures) will play a fundamental role. By the previous trivial statements, their areas are simply the sums of the areas of the rectangles involved. In particular, the area of such a figure with rectangles on the $x$-axis at positions $x_0 < x_1 < \cdots < x_4$ and heights $y_0, y_1, y_2, y_3$ is

$$y_0(x_1 - x_0) + y_1(x_2 - x_1) + y_2(x_3 - x_2) + y_3(x_4 - x_3).$$

### 2. Definition of the Riemann Integral

**Convention.** In this chapter we will be interested in **bounded** real functions $f : J \to \mathbb{R}$ defined on compact intervals $J$, that is, functions such that there are constants $m, M$ such that for all $x \in J$, $m \le f(x) \le M$. Recall that (because of the compactness) a continuous function on $J$ is always bounded. But our functions will not be always necessarily continuous.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2.2 - Partition)</span></p>

A **partition** of a compact interval $\langle a, b \rangle$ is a sequence

$$P: \quad a = t_0 < t_1 < \cdots < t_{n-1} < t_n = b.$$

Another partition $P'$: $a = t'_0 < t'_1 < \cdots < t'_{n-1} < t'_m = b$ is said to **refine** $P$ (or to be a **refinement** of $P$) if the set $\lbrace t_j \mid j = 1, \ldots, n-1 \rbrace$ is contained in $\lbrace t'_j \mid j = 1, \ldots, m-1 \rbrace$.

The **mesh** of $P$, denoted $\mu(P)$, is defined as the maximum of the differences $t_j - t_{j-1}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2.3 - Upper and lower sums)</span></p>

For a bounded function $f : J = \langle a, b \rangle \to \mathbb{R}$ and a partition $P : a = t_0 < t_1 < \cdots < t_n = b$ define the **lower** resp. **upper sum** of $f$ in $P$ by setting

$$s(f, P) = \sum_{j=1}^{n} m_j (t_j - t_{j-1}) \quad \text{resp.} \quad S(f, P) = \sum_{j=1}^{n} M_j (t_j - t_{j-1})$$

where 

* $m_j = \inf\lbrace f(x) \mid t_{j-1} \le x \le t_j \rbrace$
* $M_j = \sup\lbrace f(x) \mid t_{j-1} \le x \le t_j \rbrace$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.3.1 - Refinement monotonicity)</span></p>

Let $P'$ refine $P$. Then

$$s(f, P) \le s(f, P') \quad \text{and} \quad S(f, P) \ge S(f, P').$$

*Proof* will be done for the upper sum. Let $t_{k-1} = t'_l < t'_{l+1} < \cdots < t'_{l+r} = t_k$. For $M'_{l+j} = \sup\lbrace f(x) \mid t'_{l+j-1} \le x \le t'_{l+j} \rbrace$ and $M_k = \sup\lbrace f(x) \mid t_{k-1} \le x \le t_k \rbrace$ we have $\sum_j M'_j(t'_{l+j} - t'_{l+j-1}) \le \sum_j M_k(t'_{l+j} - t'_{l+j-1}) = M_k(t_k - t_{k-1})$ and the statement follows. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.3.2 - Lower sums bounded by upper sums)</span></p>

For any two partitions $P_1, P_2$ we have

$$s(f, P_1) \le S(f, P_2).$$

*Proof.* Obviously, $s(f, P) \le S(f, P)$ for any partition. Further, for any two partitions $P_1, P_2$ there is a common refinement $P$: it suffices to take the union of the dividing points of the two partitions. Thus, by 2.3.1,

$$s(f, P_1) \le s(f, P) \le S(f, P) \le S(f, P_2). \quad \square$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2.4 - Lower and upper Riemann integrals)</span></p>

By 2.3.2 we have the set of real numbers $\lbrace s(f, P) \mid P \text{ a partition} \rbrace$ bounded from above and $\lbrace S(f, P) \mid P \text{ a partition} \rbrace$ bounded from below. Hence there are finite

$$\underline{\int_a^b} f(x)\mathrm{d}x = \sup\lbrace s(f, P) \mid P \text{ a partition} \rbrace$$

$$\overline{\int_a^b} f(x)\mathrm{d}x = \inf\lbrace S(f, P) \mid P \text{ a partition} \rbrace.$$

The first is called the **lower Riemann integral** of $f$ over $\langle a, b \rangle$, the second is the **upper Riemann integral** of $f$.

</div>

From 2.3.2 again we see that

$$\underline{\int_a^b} f(x)\mathrm{d}x \le \overline{\int_a^b} f(x)\mathrm{d}x;$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Riemann integral)</span></p>

If $\underline{\int_a^b} f(x)\mathrm{d}x = \overline{\int_a^b} f(x)\mathrm{d}x$ the common value is denoted by

$$\int_a^b f(x)\mathrm{d}x$$

and called the **Riemann integral** of $f$ over $\langle a, b \rangle$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(2.4.1)</span></p>

Set $m = \inf\lbrace f(x) \mid a \le x \le b \rbrace$ and $M = \sup\lbrace f(x) \mid a \le x \le b \rbrace$. We have

$$m(b - a) \le \underline{\int_a^b} f(x)\mathrm{d}x \le \overline{\int_a^b} f(x)\mathrm{d}x \le M(b - a).$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.4.2 - $\varepsilon$-characterization of integrability)</span></p>

The Riemann integral $\int_a^b f(x)\mathrm{d}x$ exists if and only if for every $\varepsilon > 0$ there is a partition $P$ such that

$$S(f, P) - s(f, P) < \varepsilon.$$

*Proof.* I. Let $\int_a^b f(x)\mathrm{d}x$ exist and let $\varepsilon > 0$. Then there are partitions $P_1$ and $P_2$ such that

$$S(f, P_1) < \int_a^b f(x)\mathrm{d}x + \frac{\varepsilon}{2} \quad \text{and} \quad s(f, P_2) > \int_a^b f(x)\mathrm{d}x - \frac{\varepsilon}{2}.$$

Then we have, by 2.3.1, for the common refinement $P$ of $P_1, P_2$,

$$S(f, P) - s(f, P) < \int_a^b f(x)\mathrm{d}x + \frac{\varepsilon}{2} - \int_a^b f(x)\mathrm{d}x + \frac{\varepsilon}{2} = \varepsilon.$$

II. Let the statement hold. Choose an $\varepsilon > 0$ such that $S(f, P) - s(f, P) < \varepsilon$. Then

$$\overline{\int_a^b} f(x)\mathrm{d}x \le S(f, P) < s(f, P) + \varepsilon \le \underline{\int_a^b} f(x)\mathrm{d}x + \varepsilon,$$

and since $\varepsilon$ was arbitrary we conclude that $\overline{\int_a^b} f(x)\mathrm{d}x = \underline{\int_a^b} f(x)\mathrm{d}x$. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(2.5 - Geometric interpretation)</span></p>

1. We will see best what is happening if we analyse the case of a non-negative function $f$. Consider $F = \lbrace (x, y) \mid x \in \langle a, b \rangle,\, 0 \le f(x) \rbrace$, that is, the figure bordered by the $x$-axis, the graph of $f$ and the vertical lines passing through $(a, 0)$ and $(b, 0)$. Take the largest union $F_l(P)$ of rectangles with the lower horizontal sides $\langle t_{j-1}, t_j \rangle$ (recall the picture in 1.5) that is contained in $F$; obviously $\mathsf{vol}(F_l(P)) = s(f, P)$. The similar smallest union of rectangles $F_u(P)$ that contains $F$ has $\mathsf{vol}(F_u(P)) = S(f, P)$. Thus, if the area of $F$ makes sense we have to have

$$s(f, P) = \mathsf{vol}(F_l(P)) \le \mathsf{vol}(F) \le \mathsf{vol}(F_u(P)) = S(f, P),$$

and if $\int_a^b f(x)\mathrm{d}x$ exists then this number is the only candidate for $\mathsf{vol}(F)$ and it is only natural to take it for the definition of the area.

2. The notation $\int_a^b f(x)\mathrm{d}x$ comes from not quite correct but useful intuition. Think of $\mathrm{d}x$ as of a very small interval (one would like to say "infinitely small, but with non-zero length", which is not quite such a nonsense as it sounds); anyway, the $\mathrm{d}x$ are disjoint and cover the segment $\langle a, b \rangle$, and $\int$ stands for "sum" of the areas of the "very thin rectangles" with the horizontal side $\mathrm{d}x$ and height $f(x)$. Note how close this intuition is to the more correct view from 1 if we take $P$ with a very small mesh.

</div>

**Notation.** If there is no danger of confusion we abbreviate (in analogy with the notation in Chapter X) the expressions

$$\underline{\int_a^b} f(x)\mathrm{d}x, \quad \int_a^b f(x)\mathrm{d}x, \quad \int_a^b f(x)\mathrm{d}x \quad \text{to} \quad \underline{\int_a^b} f, \quad \overline{\int_a^b} f, \quad \int_a^b f.$$

### 3. Continuous Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(3.1 - Uniform continuity)</span></p>

A real function $f : D \to \mathbb{R}$ is said to be **uniformly continuous** if

$$\forall \varepsilon > 0 \;\; \exists \delta > 0 \;\; \text{such that} \;\; \forall x, y \in D, \;\; \lvert x - y \rvert < \delta \;\Rightarrow\; \lvert f(x) - f(y) \rvert < \varepsilon.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(3.1.1 - Continuity vs. uniform continuity)</span></p>

Note the subtle difference between continuity and uniform continuity. In the former the $\delta$ depends not only on the $\varepsilon$ but also on the $x$, while in the latter it does not. A uniformly continuous function is obviously continuous, but the reverse implication does not hold even in very simple cases. Take for instance

$$f(x) = (x \mapsto x^2) : \mathbb{R} \to \mathbb{R}.$$

We have $\lvert x^2 - y^2 \rvert = \lvert x - y \rvert \cdot \lvert x + y \rvert$; thus, if we wish to have $\lvert x^2 - y^2 \rvert < \varepsilon$ in the neighbourhood of $x = 1$ it suffices to take $\delta$ close to $\varepsilon$ itself, in the neighbourhood of $x = 100$ one needs something like $\delta = \frac{\varepsilon}{100}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.1.2 - Uniform continuity on compact domains)</span></p>

A function $f : \langle a, b \rangle \to \mathbb{R}$ is continuous if and only if it is uniformly continuous.

*Proof.* Let $f$ not be uniformly continuous. We will prove it is not continuous either.

Since the formula for uniform continuity does not hold we have an $\varepsilon_0 > 0$ such that for every $\delta > 0$ there are $x(\delta), y(\delta)$ such that $\lvert x(\delta) - y(\delta) \rvert < \delta$ while $\lvert f(x(\delta)) - f(y(\delta)) \rvert \ge \varepsilon_0$. Set $x_n = x(\frac{1}{n})$ and $y_n = y(\frac{1}{n})$. By IV.1.3.1 we can choose convergent subsequences $(\widetilde{x}\_n)\_n$, $(\widetilde{y}\_n)\_n$ (first choose a convergent subsequence $(x_{k_n})\_n$ of $(x_n)\_n$ then a convergent subsequence $(y_{k_{l_n}})\_n$ of $(y_{n_k})\_k$ and finally set $\widetilde{x}\_n = x_{k_{l_n}}$ and $\widetilde{y}\_n = y_{k_{l_n}}$). Then $\lvert \widetilde{x}\_n - \widetilde{y}\_n \rvert < \frac{1}{n}$ and hence $\lim \widetilde{x}\_n = \lim \widetilde{y}\_n$. Because of $\lvert f(\widetilde{x}\_n) - f(\widetilde{y}\_n) \rvert \ge \varepsilon_0$, however, we cannot have $\lim f(\widetilde{x}\_n) = \lim f(\widetilde{y}\_n)$ so that by IV.5.1 $f$ is not continuous. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.2 - Integrability of continuous functions)</span></p>

For every continuous function $f : \langle a, b \rangle \to \mathbb{R}$ the Riemann integral $\int_a^b f$ exists.

*Proof.* Since $f$ is by 3.1.2 uniformly continuous we can choose, for $\varepsilon > 0$ a $\delta > 0$ such that

$$\lvert x - y \rvert < \delta \;\Rightarrow\; \lvert f(x) - f(y) \rvert < \frac{\varepsilon}{b - a}.$$

Recall the mesh $\mu(P) = \max_j(t_j - t_{j-1})$ of $P : t_0 < t_1 < \cdots < t_k$. If $\mu(P) < \delta$ we have $t_j - t_{j-1} < \delta$ for all $j$, and hence

$$M_j - m_j = \sup\lbrace f(x) \mid t_{j-1} \le x \le t_j \rbrace - \inf\lbrace f(x) \mid t_{j-1} \le x \le t_j \rbrace \le \sup\lbrace \lvert f(x) - f(y) \rvert \mid t_{j-1} \le x, y \le t_j \rbrace \le \frac{\varepsilon}{b - a}$$

so that

$$S(f, P) - s(f, P) = \sum (M_j - m_j)(t_j - t_{j-1}) \le \frac{\varepsilon}{b - a} \sum (t_j - t_{j-1}) = \frac{\varepsilon}{b - a}(b - a) = \varepsilon.$$

Now use 2.4.2. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.2.1 - Limit of Riemann sums)</span></p>

Let $f : \langle a, b \rangle \to \mathbb{R}$ be a continuous function and let $P_1, P_2, \ldots$ be a sequence of partitions such that $\lim_n \mu(P_n) = 0$. Then

$$\lim_n s(f, P_n) = \lim_n S(f, P_n) = \int_a^b f.$$

(Indeed, with $\varepsilon$ and $\delta$ as above choose an $n_0$ such that for $n \ge n_0$ we have $\mu(P_n) < \delta$.)

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.3 - Integral Mean Value Theorem)</span></p>

Let $f : \langle a, b \rangle \to \mathbb{R}$ be continuous. Then there exists a $c \in \langle a, b \rangle$ such that

$$\int_a^b f(x)\mathrm{d}x = f(c)(b - a).$$

*Proof.* Set $m = \min\lbrace f(x) \mid a \le x \le b \rbrace$ and $M = \max\lbrace f(x) \mid a \le x \le b \rbrace$ (recall IV.5.2). Then

$$m(b - a) \le \int_a^b f(x)\mathrm{d}x \le M(b - a).$$

Hence there is a $K$ with $m \le K \le M$ such that $\int_a^b f(x)\mathrm{d}x = K(b - a)$. By IV.3.2 there exists a $c \in \langle a, b \rangle$ such that $K = f(c)$. $\square$

</div>

### 4. Fundamental Theorem of Calculus

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(4.1 - Additivity of the integral)</span></p>

Let $a < b < c$ and let $f$ be bounded on $\langle a, c \rangle$. Then

$$\underline{\int_a^b} f + \underline{\int_b^c} f = \underline{\int_a^c} f \quad \text{and} \quad \overline{\int_a^b} f + \overline{\int_b^c} f = \overline{\int_a^c} f.$$

*Proof* for the lower integral. Denote by $\mathcal{P}(u, v)$ the set of all partitions of $\langle u, v \rangle$. For $P_1 \in \mathcal{P}(a, b)$ and $P_2 \in \mathcal{P}(b, c)$ define $P_1 + P_2 \in \mathcal{P}(a, c)$ as the union of the two sequences. Then obviously

$$s(f, P_1 + P_2) = s(f, P_1) + s(f, P_2)$$

and hence

$$\underline{\int_a^b} f + \underline{\int_b^c} f = \sup_{P_1 \in \mathcal{P}(a,b)} s(f, P_1) + \sup_{P_2 \in \mathcal{P}(b,c)} s(f, P_2) = $$

$$= \sup\lbrace s(f, P_1) + s(f, P_2) \mid P_1 \in \mathcal{P}(a, b),\, P_2 \in \mathcal{P}(b, c) \rbrace = \sup\lbrace s(f, P_1 + P_2) \mid P_1 \in \mathcal{P}(a, b),\, P_2 \in \mathcal{P}(b, c) \rbrace.$$

Now every $P \in \mathcal{P}(a, c)$ can be refined to a $P_1 + P_2$: it suffices to add $b$ into the sequence. Thus, by 2.3.1 this last supremum is equal to

$$\sup\lbrace s(f, P) \mid P \in \mathcal{P}(a, c) \rbrace = \underline{\int_a^c} f. \quad \square$$

</div>

**Convention.** For $a = b$ we set $\int_a^a f = 0$ and for $a > b$ we set $\int_a^b f = \int_b^a f$. Then by straightforward checking we obtain

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(4.2.1)</span></p>

For any $a, b, c$,

$$\int_a^b f + \int_b^c f = \int_a^c f.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(4.3 - Fundamental Theorem of Calculus)</span></p>

Let $f : \langle a, b \rangle \to \mathbb{R}$ be continuous. For $x \in \langle a, b \rangle$ set

$$F(x) = \int_a^x f(t)\mathrm{d}t.$$

Then $F'(x) = f(x)$ (to be precise, the derivative in $a$ is from the right and the one in $b$ is from the left).

*Proof.* By 4.2.1 and 3.3 we have for $h \neq 0$

$$\frac{1}{h}(F(x+h) - F(x)) = \frac{1}{h}\left(\int_a^{x+h} f - \int_a^x f\right) = \frac{1}{h}\int_x^{x+h} f = \frac{1}{h}f(x + \theta h)h = f(x + \theta h)$$

where $0 < \theta < 1$ and as $f$ is continuous, $\lim_{h \to 0} f(x + \theta h) = f(x)$. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(4.3.1 - Newton-Leibniz formula)</span></p>

Let $f : \langle a, b \rangle \to \mathbb{R}$ be continuous. Then it has a primitive function on $(a, b)$ continuous on $\langle a, b \rangle$. If $G$ is any primitive function of $f$ on $(a, b)$ continuous on $\langle a, b \rangle$ then

$$\int_a^b f(t)\mathrm{d}t = G(b) - G(a).$$

(By 4.3 we have $\int_a^b f(t)\mathrm{d}t = F(b) - F(a)$. Recall X.1.2.)

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(4.3.2 - Derivatives vs. primitive functions)</span></p>

Note the contrast between derivatives and primitive functions. Having a derivative is a very strong property of a continuous function, but differentiating of elementary functions -- that is, the functions we typically encounter -- is very easy. On the other hand, each continuous function has a primitive one, but it is hard to compute.

</div>

Recall the Integral Mean Value Theorem (3.3). The fundamental theorem of calculus puts it in a close connection with the Mean Value Theorem of differential calculus. Indeed if we denote by $F$ the primitive function of $f$, the formula in 3.3 reads

$$F(b) - F(a) = F'(c)(b - a).$$

### 5. A Few Simple Facts

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.1 - Invariance under finitely many changes)</span></p>

Let $g$ and $f$ differ in finitely many points. Then

$$\underline{\int_a^b} f = \underline{\int_a^b} g \quad \text{and} \quad \overline{\int_a^b} f = \overline{\int_a^b} g.$$

In particular, if $\int_a^b f$ exists then also $\int_a^b g$ exists and $\int_a^b f = \int_a^b g$.

*Proof* for the lower integral. Recall the mesh $\mu(P)$ from 2.2. If $\lvert f(x) \rvert$ and $\lvert g(x) \rvert$ are $\le A$ for all $x$ and if $f$ and $g$ differ in $n$ points then

$$\lvert s(f, P) - s(g, P) \rvert \le n \cdot A \cdot \mu(P),$$

and $\mu(P)$ can be arbitrarily small. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.2 - Integrability with finitely many discontinuities)</span></p>

Let $f$ have only finitely many points of discontinuity in $\langle a, b \rangle$, all of them of the first kind. Then the Riemann integral $\int_a^b f$ exists.

*Proof.* Let the discontinuity points be $c_1 < c_2 < \cdots < c_n$. Then we have

$$\int_a^b f = \int_a^{c_1} f + \int_{c_1}^{c_2} f + \cdots + \int_{c_n}^b f. \quad \square$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.3 - Linearity of the Riemann integral)</span></p>

Let $\int_a^b f$ and $\int_a^b g$ exist and let $\alpha, \beta$ be real numbers. Then $\int_a^b (\alpha f + \beta g)$ exists and we have

$$\int_a^b (\alpha f + \beta g) = \alpha \int_a^b f + \beta \int_a^b g.$$

*Proof.* I. First we easily see that $\underline{\int_a^b} \alpha f = \alpha \int_a^b f$. Indeed, for $\alpha \ge 0$ we obviously have $s(\alpha f, P) = \alpha s(f, P)$ and $S(\alpha f, P) = \alpha S(f, P)$, and for $\alpha \le 0$ we have $s(\alpha f, P) = \alpha S(f, P)$ and $S(\alpha f, P) = \alpha s(f, P)$.

II. Thus, it suffices to prove the statement for the sum $f + g$. Set $m_i = \inf\lbrace f(x) + g(x) \mid x \in \langle t_{i-1}, t_i \rangle \rbrace$, $m'\_i = \inf\lbrace f(x) \mid x \in \langle t_{i-1}, t_i \rangle \rbrace$ and $m''\_i = \inf\lbrace g(x) \mid x \in \langle t_{i-1}, t_i \rangle \rbrace$. Obviously $m'\_i + m''\_i \le m_i$ and consequently

$$s(f, P) + s(g, P) \le s(f + g, P), \quad \text{and similarly} \quad S(f + g, P) \le S(f, P) + S(g, P)$$

and we easily conclude that

$$\underline{\int_a^b} f + \underline{\int_a^b} g \le \underline{\int_a^b} (f + g) \le \overline{\int_a^b} (f + g) \le \overline{\int_a^b} f + \overline{\int_a^b} g. \quad \square$$

</div>

#### 5.4. Per Partes

Set $[h]_a^b = h(b) - h(a)$. Then we trivially obtain from 4.3 and X.3.1

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(5.4 - Integration by parts for Riemann integral)</span></p>

$$\int_a^b f \cdot g' = [f \cdot g]_a^b - \int_a^b f' \cdot g.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(5.5 - Substitution theorem for Riemann integral)</span></p>

Let $f : \langle a, b \rangle \to \mathbb{R}$ be continuous and let $\phi : \langle a, b \rangle \to \mathbb{R}$ be a one-to-one map with derivative. Then

$$\int_a^b f(\phi(x))\phi'(x)\mathrm{d}x = \int_{\phi(a)}^{\phi(b)} f(x)\mathrm{d}x.$$

*Proof.* Recall 4.4 including the definition of $F$. We immediately have

$$\int_{\phi(a)}^{\phi(b)} f(x)\mathrm{d}x = F(\phi(b)) - F(\phi(a)).$$

But from X.4.1 and 4.4 we also have

$$F(\phi(b)) - F(\phi(a)) = \int_a^b f(\phi(x))\phi'(x)\mathrm{d}x,$$

and the statement follows. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(5.5.1 - Geometric intuition of the substitution formula)</span></p>

There is a strong geometric intuition behind the substitution formula. Recall 2.5 and 2.6. Think of $\phi$ as of a deformation of the interval $\langle a, b \rangle$ to obtain $\langle \phi(a), \phi(b) \rangle$. The derivative $\phi'(x)$ is a measure of how a very small interval around $x$ is stretched resp. compressed. Thus, if we compute the integral $\int_{\phi(a)}^{\phi(b)} f$ as an integral over the original $\langle a, b \rangle$ we have to adjust the "small element" of length $\mathrm{d}x$ by the stretch or compression and obtain a corrected "small element" of length $\phi'(x)\mathrm{d}x$.

</div>

## XII. A Few Applications of Riemann Integral

In this short chapter we will present a few applications of Riemann integral. Some of them will concern computing volumes and similar, but there will be also two theoretical ones.

### 1. The Area of a Planar Figure Again

We motivated the definition of Riemann integral by the idea of the area of the planar figure

$$F = \lbrace (x, y) \mid x \in \langle a, b \rangle, 0 \le y \le f(x) \rbrace$$

where $f$ was a non-negative continuous function. Given a partition $P : a = t_0 < t_1 \cdots < t_n = b$ of $\langle a, b \rangle$ this $F$ was minorized by the union of rectangles

$$\bigcup_{j=1}^{n} \langle t_{j-1}, t_j \rangle \times \langle 0, m_j \rangle \quad \text{with} \quad m_j = \inf\lbrace f(x) \mid t_{j-1} \le x \le t_j \rbrace,$$

with the area $s(f, D) = \sum_{j=1}^{n} m_j(t_j - t_{j-1})$, and majorized by the union of rectangles

$$\bigcup_{j=1}^{n} \langle t_{j-1}, t_j \rangle \times \langle 0, M_j \rangle \quad \text{with} \quad M_j = \sup\lbrace f(x) \mid t_{j-1} \le x \le t_j \rbrace,$$

with the area $S(f, D) = \sum_{j=1}^{n} M_j(t_j - t_{j-1})$. Thus (recall XI.2.5), the only candidate for the area of $F$ is

$$\mathsf{vol}(F) = \int_a^b f(x)\mathrm{d}x,$$

the common value of the supremum of the former and the infimum of the latter.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(1.2 - Area under a parabola)</span></p>

The area of the section of parabola $F = \lbrace (x, y) \mid -1 \le x \le 1, 0 \le y \le 1 - x^2 \rbrace$ is

$$\int_{-1}^{1} (1 - x^2)\mathrm{d}x = \left[x - \frac{1}{3}x^3\right]_{-1}^{1} = 1 - \frac{1}{3} + 1 - \frac{1}{3} = \frac{4}{3}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(1.3 - Area of a circle)</span></p>

Let us compute the area of the circle with radius $r$. A half of it is given by

$$J = \int_{-r}^{r} \sqrt{r^2 - x^2}\mathrm{d}x.$$

Substitute $x = r\sin y$. Then $\mathrm{d}x = r\cos y\mathrm{d}y$ and $\sqrt{r^2 - x^2} = r\cos y$ so that we have $J$ transformed to

$$J = r^2 \int_{-\frac{\pi}{2}}^{\frac{\pi}{2}} \cos^2 y \,\mathrm{d}y.$$

Now $\cos^2 y = \frac{1}{2}(\cos 2y + 1)$, and we proceed

$$\frac{J}{r^2} = \frac{1}{2}\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}} \cos 2y\,\mathrm{d}y + \frac{1}{2}\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}} \mathrm{d}y = \frac{1}{2}\left(\left[\frac{1}{2}\sin 2y\right]_{-\frac{\pi}{2}}^{\frac{\pi}{2}} + [y]_{-\frac{\pi}{2}}^{\frac{\pi}{2}}\right) = \frac{1}{2}(0 + \pi)$$

and hence the area in question is $2J = \pi r^2$.

</div>

### 2. Volume of a Rotating Body

Consider again a non-negative continuous function $f$ and the curve

$$C = \lbrace (x, f(x), 0) \mid a \le x \le b \rbrace$$

in the three-dimensional Euclidean space. Now rotate $C$ around the $x$-axis $\lbrace x, 0, 0) \mid x \in \mathbb{R} \rbrace$ and consider the set $F$ surrounded by the result.

It is easy to compute the volume of $F$. Instead of the union of rectangles $\bigcup_{j=1}^{n} \langle t_{j-1}, t_j \rangle \times \langle 0, m_j \rangle$ as in 1.1, we will now minorize the set $F$ by the union of discs (cylinders)

$$\bigcup_{j=1}^{n} \langle t_{j-1}, t_j \rangle \times \lbrace (y, z) \mid y^2 + z^2 \le m_j^2 \rbrace \quad \text{with} \quad m_j = \inf\lbrace f(x) \mid t_{j-1} \le x \le t_j \rbrace$$

with the volume $\sum_{j=1}^{n} \pi m_j^2(t_j - t_{j-1})$ and similarly we obtain the upper estimate of the volume by

$$\sum_{j=1}^{n} \pi M_j^2(t_j - t_{j-1}) \quad \text{with} \quad M_j = \sup\lbrace f(x) \mid t_{j-1} \le x \le t_j \rbrace.$$

Thus, we compute the volume of $F$ as

$$\mathsf{vol}(F) = \pi \int_a^b f^2(x)\mathrm{d}x.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(2.2 - Volume of a ball)</span></p>

For instance we obtain the three-dimensional ball $B_3$ as bounded by the rotating curve $\lbrace (x, \sqrt{r^2 - x^2}) \mid -r \le x \le r \rbrace$ and hence obtain

$$\mathsf{vol}(B_3) = \pi \int_{-r}^{r} (r^2 - x^2)\mathrm{d}x = \pi \left[r^2 x - \frac{1}{3}x^3\right]_{-r}^{r} = 2\pi\left(r^3 - \frac{1}{3}r^3\right) = \frac{4}{3}\pi r^3.$$

</div>

### 3. Length of a Planar Curve and Surface of a Rotating Body

Let $f$ be a continuous function on $\langle a, b \rangle$ (later, we will assume it to have a derivative) and the curve

$$C = \lbrace (x, f(x)) \mid a \le x \le b \rbrace.$$

Take a partition $P : a = t_0 < t_1 < \cdots < t_{n-1} < t_n = b$ of the interval $\langle a, b \rangle$, and approximate $C$ by the system of segments $S(P)$ connecting

$$(t_{j-1}, f(t_{j-1})) \quad \text{with} \quad (t_j, f(t_j)).$$

The length $L(P)$ of this approximation, the overall sum of the lengths of these segments, is

$$L(P) = \sum_{j=1}^{n} \sqrt{(t_j - t_{j-1})^2 + (f(t_j) - f(t_{j-1}))^2}.$$

Now suppose $f$ has a derivative. Then we can use the Mean Value Theorem (VII.2.2) to obtain

$$L(P) = \sum_{j=1}^{n} \sqrt{(t_j - t_{j-1})^2 + f'(\theta_i)^2(t_j - t_{j-1})^2} = \sum_{j=1}^{n} \sqrt{1 + f'(\theta_i)^2}\,(t_j - t_{j-1}).$$

Obviously if $P_1$ refines $P$ we have from the triangle inequality

$$L(P_1) \ge L(P)$$

so that

$$L(C) = \sup\lbrace L(P) \mid P \text{ partition of } \langle a, b \rangle \rbrace$$

can be naturally viewed as the length of the curve $C$. By XI.3.2.1 the sums converge to

$$L(C) = \int_a^b \sqrt{1 + f'(x)^2}\,\mathrm{d}x.$$

**Surface of a rotating body.** Similarly, approximating the surface of a rotating body by the relevant parts of truncated cones with heights $(t_j - t_{j-1})$ and radii $f(t_i)$ and $f(t_{j-1})$ of the bases, we obtain the formula

$$2\pi \int_a^b f(x)\sqrt{1 + f'(x)^2}\,\mathrm{d}x.$$

### 4. Logarithm

In V.1.1 we introduced logarithm axiomatically as a function $L$ that

1. increases in $\langle 0, +\infty)$,
2. satisfies $L(xy) = L(x) + L(y)$,
3. and such that $\lim_{x \to 0} \frac{L(x)}{x - 1} = 1$.

The existence of such a function (which we had to believe in in V.1.1) will be now proven by a simple construction.

Set

$$L(x) = \int_1^x \frac{1}{t}\mathrm{d}t.$$

If $x > 0$ this is correct: the function $\frac{1}{t}$ is well defined and continuous on the closed interval between 1 and $x$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(4.2.1 - $L$ is increasing)</span></p>

If $x < y$ then $L(y) - L(x) = \int_x^y \frac{1}{t}\mathrm{d}t$ is an integral of a positive function over $\langle x, y \rangle$ and hence a positive number. Hence $L(x)$ increases.

</div>

**Verification of the product rule.** We have

$$L(xy) = \int_1^{xy} \frac{1}{t}\mathrm{d}t = \int_1^{x} \frac{1}{t}\mathrm{d}t + \int_x^{xy} \frac{1}{t}\mathrm{d}t. \qquad (*)$$

In the last summand substitute $z = \phi(t) = xt$ to obtain

$$\int_x^{xy} \frac{1}{z}\mathrm{d}z = \int_1^{y} \frac{1}{xt}\phi'(t)\mathrm{d}t = \int_1^{y} \frac{x}{xt}\mathrm{d}t = \int_1^{y} \frac{1}{t}\mathrm{d}t$$

so that $(*)$ yields

$$L(x, y) = \int_1^{x} \frac{1}{t}\mathrm{d}t + \int_1^{y} \frac{1}{t}\mathrm{d}t = L(x) + L(y).$$

**Verification of the limit condition.** Finally we have

$$\lim_{x \to 0} \frac{L(x)}{x - 1} = \lim_{x \to 0} \frac{L(x) - L(1)}{x - 1} = L'(1) = \frac{1}{1} = 1$$

by XI.4.3.

### 5. Integral Criterion of Convergence of a Series

Consider a series $\sum a_n$ with $a_1 \ge a_2 \ge a_3 \ge \cdots \ge 0$. Let $f$ be a non-increasing continuous function defined on the interval $\langle 1, +\infty)$ such that $a_n = f(n)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(5.2 - Integral Criterion of Convergence)</span></p>

The series $\sum a_n$ converges if and only if the limit

$$\lim_{n \to \infty} \int_1^n f(x)\mathrm{d}x$$

is finite.

*Proof.* The trivial estimate of Riemann integral yields

$$a_{n+1} = f(n+1) \le \int_n^{n+1} f(x)\mathrm{d}x \le f(n) = a_n.$$

Thus,

$$a_2 + a_3 + \cdots + a_n \le \int_1^n f(x)\mathrm{d}x \le a_1 + a_2 + \cdots + a_{n-1}.$$

Hence, if $L = \lim_{n \to \infty} \int_1^n f(x)\mathrm{d}x$ is finite then

$$\sum_1^n a_k \le a_1 + L$$

and the series converges. On the other hand, if the sequence $(\int_1^n f(x)\mathrm{d}x)_n$ is not bounded then also $(\sum_1^n a_n)_n$ is not bounded. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(5.3)</span></p>

Note that unlike the criteria in III.2.5, the Integral Criterion is a necessary and sufficient condition. Hence, of course, it is much finer. This will be illustrated by the following example.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.4 - Convergence of $p$-series)</span></p>

Let $\alpha > 1$ be a real number. Then the series

$$\frac{1}{1^\alpha} + \frac{1}{2^\alpha} + \frac{1}{3^\alpha} + \cdots + \frac{1}{n^\alpha} + \cdots \qquad (*)$$

converges.

*Proof.* We have

$$\int_1^n x^{-\alpha}\mathrm{d}x = \left[\frac{1}{1 - \alpha} \cdot x^{1-\alpha}\right] = \frac{1}{1 - \alpha}\left(\frac{1}{n^{\alpha - 1}} - 1\right) \le \frac{1}{\alpha - 1}. \quad \square$$

Note that the convergence of the series $(*)$ does not follow from the criteria III.2.5 even for big $\alpha$.

</div>

## XIII. Metric Spaces: Basics

### 1. An Example

In the following chapters we will study real functions of several real variables. Hence, domains of such functions will be subsets of Euclidean spaces. We will need to understand better the basic notions like convergence or continuity: as we will see in the following example they cannot be reduced to the behaviour of functions in the individual variables. In this chapter we will discuss some concepts to be used in the general context of metric spaces.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(1.2 - Continuity in individual variables does not imply joint continuity)</span></p>

Define a function of two real variables $f : \mathbb{E}_2 \to \mathbb{R}$ by setting

$$f(x, y) = \begin{cases} \frac{xy}{x^2 + y^2} & \text{for } (x, y) \neq (0, 0), \\ 0 & \text{for } (x, y) = (0, 0). \end{cases}$$

For any fixed $y_0$ the function $\phi : \mathbb{R} \to \mathbb{R}$ defined by $\phi(x) = f(x, y_0)$ is evidently a continuous one (if $y_0 \neq 0$ it is defined by an arithmetic expression, and for $y_0 = 0$ it is the constant 0) and similarly for any fixed $x_0$ the formula $\psi(y) = f(x_0)$ defines a continuous function $\psi : \mathbb{R} \to \mathbb{R}$. But the function $f$ as a whole behaves weirdly: if we approach $(0, 0)$ in the arguments $(x, x)$ with $x \neq 0$ the values of $f$ are constantly $\frac{1}{2}$ and at $x = 0$ we jump to 0, an evident discontinuity in any reasonable intuitive meaning of the word.

</div>

### 2. Metric Spaces, Subspaces, Continuity

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2.1 - Metric space)</span></p>

A **metric** (or **distance function**, or briefly **distance**) on a set $X$ is a function

$$d : X \times X \to \mathbb{R}$$

such that

1. $\forall x, y$, $d(x, y) \ge 0$ and $d(x, y) = 0$ iff $x = y$,
2. $\forall x, y$, $d(x, y) = d(y, x)$ and
3. $\forall x, y, z$, $d(x, z) \le d(x, y) + d(y, z)$ (triangle inequality).

A **metric space** $(X, d)$ is a set $X$ endowed with a metric $d$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Intuition behind the axioms)</span></p>

The assumptions (1) and (3) are rather intuitive: (1) requires that the distance of two distinct points is not zero, (3) says that the shortest path between $x$ and $z$ cannot be longer than the one subjected to the condition that we visit a point $y$ on the way. The symmetry condition (2) is somewhat less satisfactory (consider the distances between two places in town one has to cover by car), but for our purposes is quite acceptable.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(2.2 - Examples of metric spaces)</span></p>

1. The **real line**, that is, $\mathbb{R}$ with the distance $d(x, y) = \lvert x - y \rvert$.

2. The **Gauss plane**, that is, the set of complex numbers $\mathbb{C}$ with the distance $d(x, y) = \lvert x - y \rvert$. Note that the fact that this formula is a distance in $\mathbb{C}$ is less trivial than the fact about the $\lvert x - y \rvert$ in $\mathbb{R}$.

3. The $n$**-dimensional Euclidean space** $\mathbb{E}_n$: The set $\lbrace (x_1, \ldots, x_n) \mid x_i \in \mathbb{R} \rbrace$ with the metric

$$d((x_1, \ldots, x_n), (y_1, \ldots, y_n)) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}. \qquad (*)$$

4. Let $J$ be an interval. Consider the set $F(J) = \lbrace f \mid f : J \to \mathbb{R} \text{ bounded} \rbrace$ endowed with the distance

$$d(f, g) = \sup\lbrace \lvert f(x) - g(x) \rvert \mid x \in J \rbrace.$$

</div>

**More about $\mathbb{E}_n$.** The Euclidean space $\mathbb{E}_n$ (and its subsets) will play a fundamental role in the sequel. It deserves a few comments.

(a) The reader knows from linear algebra the $n$-dimensional vector space $V_n$, the scalar product $x \cdot y = (x_1, \ldots, x_n) \cdot (y_1, \ldots, y_n) = \sum_{i=1}^{n} x_i y_i$, the norm $\lVert x \rVert = \sqrt{x \cdot x}$, and the Cauchy-Schwarz inequality

$$\lvert x \cdot y \rvert \le \lVert x \rVert \cdot \lVert y \rVert.$$

From this inequality one easily infers that $d(x, y) = \lVert x - y \rVert$ is a distance on $V_n$. Now $\mathbb{E}\_n$ is nothing else than $(V_n, d)$ with the structure of vector space neglected.

(b) The Gauss plane is the Euclidean plane $\mathbb{E}_2$. Only, similarly as $V_n$ as compared with $\mathbb{E}_n$, it has more structure.

(c) The (Pythagorean) metric $(\ast)$ in $\mathbb{E}\_n$ is in accordance with the standard Euclidean geometry. It can be, however, somewhat inconvenient to work with. More expedient distances (equivalent with $(\ast)$ for our purposes) will be introduced in 4.3 below.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2.3 - Continuous and uniformly continuous maps)</span></p>

Let $(X_1, d_1)$ and $(X_2, d_2)$ be metric spaces. A mapping $f : X_1 \to X_2$ is said to be **continuous** if

$$\forall x \in X_1 \;\forall \varepsilon > 0 \;\exists \delta > 0 \text{ such that } \forall y \in X_1, \; d_1(x, y) < \delta \;\Rightarrow\; d_2(f(x), f(y)).$$

It is said to be **uniformly continuous** if

$$\forall \varepsilon > 0 \;\exists \delta > 0 \text{ such that } \forall x \in X_1 \;\forall y \in X_1, \; d_1(x, y) < \delta \;\Rightarrow\; d_2(f(x), f(y)).$$

Note that obviously each uniformly continuous mapping is continuous.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(2.3.1)</span></p>

(1) The identity mapping $\mathrm{id} : (X, d) \to (X, d)$ is continuous.

(2) The composition $g \circ f : (X_1, d_1) \to (X_3, d_3)$ of (uniformly) continuous maps $f : (X_1, d_1) \to (X_2, d_2)$ and $g : (X_2, d_2) \to (X_3, d_3)$ is (uniformly) continuous.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2.4 - Subspaces)</span></p>

Let $(X, d)$ be a metric space and let $Y \subseteq X$ be a subset. Defining $d_Y(x, y) = d(x, y)$ for $x, y \in Y$ we obtain a metric on $Y$; the resulting metric space $(Y, d_Y)$ is said to be a **subspace** of $(X, d)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(2.4.1)</span></p>

Let $f : (X_1, d_1) \to (X_2, d_2)$ be a (uniformly) continuous mapping. Let $Y_i \subseteq X_i$ be such that $f[Y_1] \subseteq Y_2$. Then the mapping $g : (Y_1, d\_{1\_{Y\_1}}) \to (Y_2, d\_{2\_{Y\_2}})$ defined by $g(x) = f(x)$ is (uniformly) continuous.

</div>

**Conventions.** 1. Often, if there is no danger of confusion, we use the same symbol for distinct metrics. In particular we will mostly omit the subscript $Y$ in the subspace metric $d_Y$.

2. Unless stated otherwise, we will endow a subset of a metric space automatically with the subspace metric. We will speak of subspaces as of the corresponding subsets, and of subsets as of the corresponding subspaces. Thus we will speak of a "finite subspace", an "open subspace" (see 3.4 below) or, on the other hand, of a "compact subset" (see Section 7), etc.

### 3. Several Topological Concepts

#### 3.1. Convergence

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(3.1 - Convergence in a metric space)</span></p>

A sequence $(x_n)_n$ in a metric space $(X, d)$ **converges** to $x \in X$ if

$$\forall \varepsilon > 0 \;\exists n_0 \text{ such that } \forall n \ge n_0, \; d(x_n, x) < \varepsilon.$$

We then speak of a **convergent sequence** and the $x$ is called its **limit**, and we write $x = \lim_n x_n$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(3.1.1)</span></p>

Let $(x_n)\_n$ be a convergent sequence and let $x$ be its limit. Then each subsequence $(x_{k_n})\_n$ of $(x_n)\_n$ converges and we have $\lim_n x_{k_n} = x$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.1.2 - Sequential characterization of continuity)</span></p>

A mapping $f : (X_1, d_1) \to (X_2, d_2)$ is continuous if and only if for each convergent sequence $(x_n)_n$ in $(X_1, d_1)$ the sequence $(f(x_n))_n$ converges in $(X_2, d_2)$ and $\lim_n f(x_n) = f(\lim_n x_n)$.

*Proof.* I. Let $f$ be continuous and let $\lim_n x_n = x$. For $\varepsilon > 0$ choose by continuity a $\delta > 0$ such that $d_2(f(y), f(x)) < \varepsilon$ for $d_1(x, y) < \delta$. Now by the definition of the convergence of sequences there is an $n_0$ such that for $n \ge n_0$, $d_1(x_n, x) < \delta$. Thus, if $n \le n_0$ we have $d_2(f(x_n), f(x)) < \varepsilon$ so that $\lim_n f(x_n) = f(\lim_n x_n)$.

II. Let $f$ not be continuous. Then there is an $x \in X_1$ and an $\varepsilon_0 > 0$ such that for every $\delta > 0$ there is an $x(\delta)$ such that $d_1(x, x(\delta)) < \delta$ but $d_2(f(x), f(x(\delta))) \ge \varepsilon_0$. Set $x_n = x(\frac{1}{n})$. Then $\lim_n x_n = x$ but $(f(x_n))_n$ cannot converge to $f(x)$. $\square$

Note that the proof is the same as that in IV.5.1, only with the $\lvert u - v \rvert$ substituted by the distances in the two spaces.

</div>

#### 3.2. Neighbourhoods

For a point $x$ in a metric space $(X, d)$ and $\varepsilon > 0$ set

$$\Omega_{(X,d)}(x, \varepsilon) = \lbrace y \mid d(x, y) < \varepsilon \rbrace$$

(if there is no danger of confusion, the subscript "$(X, d)$" is often omitted, or replaced just by "$X$").

A **neighbourhood** of a point $x$ in $(X, d)$ is any $U \subseteq X$ such that there is an $\varepsilon > 0$ with $\Omega(x, \varepsilon) \subseteq U$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.3.1 - Properties of neighbourhoods)</span></p>

1. If $U$ is a neighbourhood of $x$ and $U \subseteq V$ then $V$ is a neighbourhood of $x$.

2. If $U$ and $V$ are neighbourhoods of $x$ then the intersection $U \cap V$ is a neighbourhood of $x$.

*Proof.* 1 is trivial. 2: If $\Omega(x, \varepsilon_1) \subseteq U$ and $\Omega(x, \varepsilon_2) \subseteq V$ then $\Omega(x, \min(\varepsilon_1, \varepsilon_2)) \subseteq U \cap V$. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.3.2 - Neighbourhoods in subspaces)</span></p>

Let $Y$ be a subspace of a metric space $(X, d)$. Then $\Omega_Y(x, \varepsilon) = \Omega_X(x, \varepsilon) \cap Y$ and $U \subseteq Y$ is a neighbourhood of $x \in Y$ iff there is a neighbourhood $V$ of $x$ in $(X, d)$ such that $U = V \cap Y$.

*Proof* is straightforward. $\square$

</div>

#### 3.4. Open Sets

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(3.4 - Open set)</span></p>

A subset $U \subseteq (X, d)$ is **open** if it is a neighbourhood of each of its points.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.4.1)</span></p>

Each $\Omega_X(x, \varepsilon)$ is open in $(X, d)$.

*Proof.* Let $y \in \Omega_X(x, \varepsilon)$. Then $d(x, y) < \varepsilon$. Set $\delta = \varepsilon - d(x, y)$. By triangle inequality, $\Omega(y, \delta) \subseteq \Omega(x, \varepsilon)$. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(3.4.2 - Properties of open sets)</span></p>

$\emptyset$ and $X$ are open. If $U_i$, $i \in J$, are open then $\bigcup_{i \in J} U_i$ is open, and if $U$ and $V$ are open then $U \cap V$ is open.

*Proof.* The first three statements are obvious and the third one immediately follows from 2.3.1. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.4.3 - Open sets in subspaces)</span></p>

Let $Y$ be a subspace of a metric space $(X, d)$. Then $U$ is open in $Y$ iff there is a $V$ open in $X$ such that $U = V \cap Y$.

*Proof.* For every $V$ open in $X$, $U \cap Y$ is open in $Y$ by 3.3.2. On the other hand, if $U$ is open in $Y$ choose for each $x \in U$ an $\Omega_Y(x, \varepsilon_x) \subseteq U$ and set $V = \bigcup_{x \in U} \Omega_X(x, \varepsilon_x)$. $\square$

</div>

#### 3.5. Closed Sets

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(3.5 - Closed set)</span></p>

A subset $A \subseteq (X, d)$ is **closed** in $(X, d)$ if for every sequence $(x_n)_n \subseteq A$ convergent in $X$ the limit $\lim_n x_n$ is in $A$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.5.1 - Closed iff complement is open)</span></p>

A subset $A \subseteq (X, d)$ is closed in $(X, d)$ iff the complement $X \setminus A$ is open.

*Proof.* I. Let $X \setminus A$ not be open. Then there is a point $x \in X \setminus A$ such that for every $n$, $\Omega(x, \frac{1}{n}) \nsubseteq X \setminus A$, that is, $\Omega(x, \frac{1}{n}) \cap A \neq \emptyset$. Choose $x_n \in \Omega(x, \frac{1}{n}) \cap A$. Then $(x_n)_n \subseteq A$ and the sequence converges to $x \notin A$ and hence $A$ is not closed.

II. Let $X \setminus A$ be open and let $(x_n)_n \subseteq A$ converge to $x \in X \setminus A$. Then for some $\varepsilon > 0$, $\Omega(x, \varepsilon) \subseteq X \setminus A$ and hence for sufficiently large $n$, $x_n \in \Omega(x, \varepsilon) \subseteq X \setminus A$, a contradiction. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(3.5.2 - Properties of closed sets)</span></p>

$\emptyset$ and $X$ are closed. If $A_i$, $i \in J$, are closed then $\bigcap_{i \in J} A_i$ is closed, and if $A$ and $B$ are closed then $A \cup B$ is closed.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(3.5.3 - Closed sets in subspaces)</span></p>

Let $Y$ be a subspace of a metric space $(X, d)$. Then $A$ is closed in $Y$ iff there is a $B$ closed in $X$ such that $A = B \cap Y$.

</div>

#### 3.6. Distance of a Point from a Subset. Closure

Let $x$ be a point and $A \subseteq X$ be a subset of a metric space $(X, d)$. Define the **distance** of $x$ from $A$ as

$$d(x, A) = \inf\lbrace d(x, a) \mid a \in A \rbrace.$$

The **closure** of a set $A$ is

$$\overline{A} = \lbrace x \mid d(x, A) = 0 \rbrace.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.6.1 - Properties of closure)</span></p>

(1) $\overline{\emptyset} = \emptyset$.
(2) $A \subseteq \overline{A}$.
(3) $A \subseteq B \;\Rightarrow\; \overline{A} \subseteq \overline{B}$.
(4) $\overline{A \cup B} = \overline{A} \cup \overline{B}$, and
(5) $\overline{\overline{A}} = \overline{A}$.

*Proof.* (1): $d(x, \emptyset) = +\infty$. (2) and (3) are trivial.

(4): By (3) we have $\overline{A \cup B} \supseteq \overline{A} \cup \overline{B}$. Now let $x \in \overline{A \cup B}$ but not $x \in \overline{A}$. Then $\alpha = d(x, A) > 0$ and hence all the $y \in A \cup B$ such that $d(x, y) < \alpha$ are in $B$; hence $x \in \overline{B}$.

(5): Let $d(x, \overline{A})$ be 0. Choose $\varepsilon > 0$. There is a $z \in \overline{A}$ such that $d(x, z) < \frac{\varepsilon}{2}$ and for this $z$ we can choose a $y \in A$ such that $d(z, y) < \frac{\varepsilon}{2}$. Thus, by triangle inequality, $d(x, y) < \frac{\varepsilon}{2} + \frac{\varepsilon}{2} = \varepsilon$ and we see that $x \in \overline{A}$. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.6.2 - Closure as set of limits)</span></p>

$\overline{A}$ is the set of all limits of convergent sequences $(x_n)_n \subseteq A$.

*Proof.* A limit of a convergent $(x_n)_n \subseteq A$ is obviously in $\overline{A}$. Now let $x \in \overline{A}$. If $x \in A$ then it is the limit of the constant sequence $x, x, x, \ldots$. If $x \in \overline{A} \setminus A$ then for each $n$ there is an $x_n \in A$ such that $d(x, x_n) < \frac{1}{n}$. Obviously $x = \lim_n x_n$. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.6.3 - Closure is the smallest closed set)</span></p>

$\overline{A}$ is closed and it is the least closed set containing $A$. That is,

$$\overline{A} = \bigcap\lbrace B \mid A \subseteq B, \; B \text{ closed} \rbrace.$$

*Proof.* Let $(x_n)_n \subseteq \overline{A}$ converge to $x$. For each $n$ choose $y_n \in A$ such that $d(x_n, y_n) < \frac{1}{n}$. Then $\lim_n y_n = x$ and $x$ is in $\overline{A}$ by 3.5.1.

Now let $B$ be closed and let $A \subseteq B$. If $x \in \overline{A}$ we can choose, by 3.5.1, a convergent sequence $(x_n)_n$ in $A$, and hence in $B$, such that $\lim x_n = x$. Thus, $x \in B$. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(3.6.4 - Closure in subspaces)</span></p>

Let $Y$ be a subspace of a metric space $(X, d)$. Then the closure of $A$ in $Y$ is equal to $\overline{A} \cap Y$ (where $\overline{A}$ is the closure in $X$).

</div>

#### 3.7. Characterizations of continuity

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.7 - Equivalent characterizations of continuity)</span></p>

Let $(X_1, d_1), (X_2, d_2)$ be metric spaces and let $f : X_1 \to X_2$ be a mapping. Then the following statements are equivalent.

1. $f$ is continuous.
2. For every $x \in X_1$ and for every neighbourhood $V$ of $f(x)$ there is a neighbourhood $U$ of $x$ such that $f[U] \subseteq V$.
3. For every open $U$ in $X_2$ the preimage $f^{-1}[U]$ is open in $X_1$.
4. For every closed $A$ in $X_2$ the preimage $f^{-1}[A]$ is closed in $X_1$.
5. For every $A \subseteq X_1$, $f[\overline{A}] \subseteq \overline{f[A]}$.

*Proof.* $(1)\Rightarrow(2)$: There is an $\varepsilon > 0$ such that $\Omega(f(x), \varepsilon) \subseteq V$. Take the $\delta$ from the definition of continuity and set $U = \Omega(x, \delta)$. Then $f[U] \subseteq \Omega(f(x), \varepsilon) \subseteq V$.

$(2)\Rightarrow(3)$: Let $U$ be open and $x \in f^{-1}[U]$. Thus, $f(x) \in U$ and $U$ is a neighbourhood of $f(x)$. There is a neighbourhood $V$ of $x$ such that $f[V] \subseteq U$. Consequently $x \in V \subseteq f^{-1}[U]$ and $f^{-1}[U]$ is a neighbourhood of $x$. Since $x \in f^{-1}[U]$ was arbitrary, the preimage is open.

$(3)\Leftrightarrow(4)$ by 3.5.1 since preimage preserves complements.

$(4)\Rightarrow(5)$: We have $A \subseteq f^{-1}[f[A]] \subseteq f^{-1}[\overline{f[A]}]$. By (4), $f^{-1}[\overline{f[A]}]$ is closed and hence by 3.5.3, $\overline{A} \subseteq f^{-1}[\overline{f[A]}]$ and finally $f[\overline{A}] \subseteq \overline{f[A]}$.

$(5)\Rightarrow(1)$: Let $\varepsilon > 0$. Set $B = X_2 \setminus \Omega(f(x), \varepsilon)$ and $A = f^{-1}[B]$. Then $f[\overline{A}] \subseteq f[f^{-1}[B]] \subseteq \overline{B}$. Hence $x \notin \overline{A}$ (the distance $d(f(x), B)$ is at least $\varepsilon$) and hence there is a $\delta > 0$ such that $\Omega(x, \delta) \cap A = \emptyset$ and we easily conclude that $f[\Omega(x, \delta)] \subseteq \Omega(f(x), \varepsilon)$. $\square$

</div>

#### 3.8. Homeomorphism. Topological Concepts

A continuous mapping $f : (X, d) \to (Y, d')$ is called **homeomorphism** if there is a continuous $g : (Y, d') \to (X, d)$ such that $f \circ g = \mathrm{id}_Y$ and $g \circ f = \mathrm{id}_X$. If there exists a homeomorphism $f : (X, d) \to (Y, d')$ we say that the spaces $(X, d)$ and $(Y, d')$ are **homeomorphic**.

A property or definition is said to be **topological** if it is preserved by homeomorphisms. Thus we have the following topological properties:

* convergence (see 3.1.2),
* openness (see 3.7),
* closedness (see 3.7),
* closure (although $d(x, A)$ is not topological; see, however, 3.6.3),
* neighbourhood (although $\Omega(x, \varepsilon)$ is not topological; but realize that $A$ is a neighbourhood of $x$ if there is an open $U$ such that $x \in U \subseteq A$),
* or continuity itself.

On the other hand, for instance uniform continuity is not a topological property.

#### 3.9. Isometry

An onto mapping $f : (X, d) \to (Y, d')$ is called **isometry** if $d'(f(x), f(y)) = d(x, y)$ for all $x, y \in X$. Then, trivially,

* $f$ is one-to-one and continuous, and
* its inverse is also an isometry; thus, $f$ is a homeomorphism.

If there is an isometry $f : (X, d) \to (Y, d')$ the spaces $(X, d)$ and $(Y, d')$ are said to be **isometric**. Of course, an isometry preserves all topological concepts, but much more, indeed everything that can be defined in terms of distance.

### 4. Equivalent and Strongly Equivalent Metrics

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(4.1 - Equivalent metrics)</span></p>

Two metrics $d_1, d_2$ on a set are said to be **equivalent** if $\mathrm{id}_X : (X, d_1) \to (X, d_2)$ is a homeomorphism. Thus, replacing a metric by an equivalent one we obtain a space in which all topological notions from the original space are preserved.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(4.2 - Strongly equivalent metrics)</span></p>

A much stronger concept is that of a strong equivalence. We say that $d_1, d_2$ on a set are **strongly equivalent** if there are positive constants $\alpha$ and $\beta$ such that for all $x, y \in X$

$$\alpha \cdot d_1(x, y) \le d_2(x, y) \le \beta \cdot d_1(x, y)$$

(this relation is of course symmetric: consider $\frac{1}{\alpha}$ and $\frac{1}{\beta}$).

Note that replacing a metric by a strongly equivalent one preserves not only topological properties but also for instance the uniform convergence.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(4.3.1 - Equivalent metrics on $\mathbb{E}_n$)</span></p>

The metrics $d$, $\lambda$ and $\sigma$ on $\mathbb{E}_n$ defined by

$$d((x_i), (y_i)) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}, \quad \lambda((x_i), (y_i)) = \sum_{i=1}^{n} \lvert x_i - y_i \rvert, \quad \sigma((x_i), (y_i)) = \max_i \lvert x_i - y_i \rvert$$

are strongly equivalent.

*Proof.* It is easy to see that $\lambda$ and $\sigma$ are metrics. Now we have

$$\lambda((x_i), (y_i)) = \sum_{i=1}^{n} \lvert x_i - y_i \rvert \le n\sigma((x_j), (y_j))$$

since for each $i$, $\lvert x_i - y_i \rvert \le \sigma((x_j), (y_j))$, and for the same reason

$$d((x_i), (y_i)) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2} \le \sqrt{n}\,\sigma((x_j), (y_j)).$$

On the other hand obviously

$$\sigma((x_i), (y_i)) \le \lambda((x_i), (y_i)) \quad \text{and} \quad \sigma((x_i), (y_i)) \le d((x_i), (y_i)). \quad \square$$

</div>

In the sequel we will mostly work with the Euclidean space as with $(\mathbb{E}_n, \sigma)$.

### 5. Products

Let $(X_i, d_i)$, $i = 1, \ldots, n$ be metric spaces. On the cartesian product $\prod_{i=1}^{n} X_i$ define a metric

$$d((x_1, \ldots, x_n), (y_1, \ldots, y_n)) = \max_i d_i(x_i, y_i).$$

The resulting metric space will be denoted by $\prod_{i=1}^{n}(X_i, d_i)$.

**Notation.** We will also write $(X_1, d_1) \times (X_2, d_2)$ or $(X_1, d_1) \times (X_2, d_2) \times (X_3, d_3)$ for products of two or three spaces, and sometimes also $(X_1, d_1) \times \cdots \times (X_n, d_n)$ for the general $\prod_{i=1}^{n}(X_i, d_i)$. Further, if $(X_i, d_i) = (X, d)$ for all $i$ we write $\prod_{i=1}^{n}(X_i, d_i) = (X, d)^n$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(5.1.2)</span></p>

1. Thus, $(\mathbb{E}_n, \sigma)$ is the product $\overbrace{\mathbb{R} \times \cdots \times \mathbb{R}}^{n \text{ times}} = \mathbb{R}^n$.

2. For all purposes we could have defined the metric in the product by

$$d((x_i), (y_i)) = \sqrt{\sum_{i=1}^{n} d_i(x_i, y_i)^2} \quad \text{or} \quad d((x_i), (y_i)) = \sum_{i=1}^{n} d_i(x_i, y_i),$$

but working with the $d$ above is much easier.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(5.2 - Convergence in products)</span></p>

A sequence $(x_1^k, \ldots, x_n^k), \ldots$ **converges** to $(x_1, \ldots, x_n)$ in $\prod(X_i, d_i)$ if and only if each of the sequences $(x_i^k)_k$ converges to $x_i$ in $(X_i, d_i)$.

(Caution: the superscripts $k$ are indices, not powers.)

*Proof.* $\Rightarrow$ immediately follows from the fact that $d_i(u_i, v_i) \le d((u_j)_j, (v_j)_j)$.

$\Leftarrow$: Let each of the $(x_i^k)_k$ converge to $x_i$. For an $\varepsilon > 0$ and $i$ we have $k_i$ such that for $k \ge k_i$, $d_i(x_i^k, x_i) < \varepsilon$. Then for $k \ge \max_i k_i$ we have $d((x_1^k, \ldots, x_n^k), (x_1, \ldots, x_n)) < \varepsilon$. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(5.3 - Universal property of products)</span></p>

1. The projection mappings $p_j : \prod_{i=1}^{n}(X_i, d_i) \to (X_j, d_j)$ defined by $p_j((x_i)_i) \mapsto x_j$ are continuous.

2. Let $f_j : (Y, d') \to (X_j, d_j)$ be arbitrary continuous mappings. Then the unique mapping $f : (Y, d') \to \prod_{i=1}^{n}(X_i, d_i)$ such that $p_j \circ f = f_j$, namely that defined by $f(y) = (f_1(y), \ldots, f_n(y))$, is continuous.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* 1 immediately follows from the fact that $d_j(x_j, y_j) \le d((x_i)_i, (y_i)_i)$.

2: Follows from 3.1.2 and 5.2. If $\lim_k y_k = y$ in $(Y, d')$ then $\lim_k f_j(y_k) = f_j(y)$ in $(X_j, d_j)$ for all $j$ and hence $(f(y_k))_k$, that is, $(f_1(y_1), \ldots, f_n(y_1)), (f_1(y_2), \ldots, f_n(y_2)), \ldots$ converges to $(f_1(y), \ldots, f_n(y))$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(5.4)</span></p>

Obviously $\prod_{i=1}^{n+1}(X_i, d_i)$ is isometric (recall 3.9) with $\prod_{i=1}^{n}(X_i, d_i) \times (X_{n+1}, d_{n+1})$. Consequently, it usually suffices to prove a statement on finite products for products of two spaces only.

</div>

### 6. Cauchy Sequences. Completeness

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(6.1 - Cauchy sequence)</span></p>

A sequence $(x_n)_n$ in a metric space $(X, d)$ is said to be **Cauchy** if

$$\forall \varepsilon > 0 \;\exists n_0 \text{ such that } m, n \ge n_0 \;\Rightarrow\; d(x_m, x_n) < \varepsilon.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(6.1.1)</span></p>

Each convergent sequence is Cauchy. (Just like in $\mathbb{R}$: if $d(x_n, x) < \varepsilon$ for $n \ge n_0$ then for $m, n \ge n_0$, $d(x_n, x_m) \le d(x_n, x) + d(x, x_m) < 2\varepsilon$.)

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(6.2 - Cauchy with convergent subsequence)</span></p>

Let a Cauchy sequence have a convergent subsequence. Then it converges (to the limit of the subsequence).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* Let $(x_n)\_n$ be Cauchy and let $\lim_n x_{k_n} = x$. Let $d(x_m, x_n) < \varepsilon$ for $m, n \ge n_1$ and $d(x_{k_n}, x) \le \varepsilon$ for $n \ge n_2$. If we set $n_0 = \max(n_1, n_2)$ we have for $n \ge n_0$ (since $k_n \ge n$)

$$d(x_n, x) \le d(x_n, x_{k_n}) + d(x_{k_n}, x) < 2\varepsilon. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(6.3 - Complete metric space)</span></p>

A metric space $(X, d)$ is **complete** if each Cauchy sequence in $(X, d)$ converges.

</div>

Thus, by Bolzano-Cauchy Theorem (II.3.4) the real line $\mathbb{R}$ with the standard metric is complete.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(6.4 - Complete subspaces are closed)</span></p>

A subspace of a complete space is complete if and only if it is closed.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* I. Let $Y \subseteq (X, d)$ be closed. Let $(y_n)_n$ be Cauchy in $Y$. Then it is Cauchy and hence convergent in $X$, and the limit, by closedness, is in $Y$.

II. Let $Y$ not be closed. Then there is a sequence $(y_n)_n$ in $Y$ convergent in $X$ such that $\lim_n y_n \notin Y$. Then $(y_n)_n$ is Cauchy in $X$, but since the distance is the same, also in $Y$. But in $Y$ it does not converge. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(6.5 - Cauchy sequences in products)</span></p>

A sequence $(x_1^k, \ldots, x_n^k), \ldots$ is Cauchy in $\prod_{i=1}^{n}(X_i, d_i)$ if and only if each of the sequences $(x_i^k)_k$ is Cauchy in $(X_i, d_i)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* $\Rightarrow$ immediately follows from the fact that $d_i(u_i, v_i) \le d((u_j)_j, (v_j)_j)$.

$\Leftarrow$: Let each of the $(x_i^k)_k$ be Cauchy. For an $\varepsilon > 0$ and $i$ we have $k_i$ such that for $k, l \ge k_i$, $d_i(x_i^k, x_i^l) < \varepsilon$. Then for $k, l \ge \max_i k_i$ we have $d((x_1^k, \ldots, x_n^k), (x_1^l, \ldots, x_n^l)) < \varepsilon$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(6.6 - Products of complete spaces)</span></p>

A product of complete spaces is complete. In particular, the Euclidean space $\mathbb{E}_n$ is complete.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(6.7 - Complete subspaces of $\mathbb{E}_n$)</span></p>

A subspace $Y$ of the Euclidean space $\mathbb{E}\_n$ is complete if and only if it is closed.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(6.8 - Completeness is not topological)</span></p>

Neither the Cauchy property nor completeness is a topological property. Consider $\mathbb{R}$ and any bounded open interval $J$ in $\mathbb{R}$. They are homeomorphic (if for instance $J = (-\frac{\pi}{2}, +\frac{\pi}{2})$ we have the mutually inverse homeomorphisms $\tan : J \to \mathbb{R}$ and $\arctan : \mathbb{R} \to J$). But $\mathbb{R}$ is complete and $J$ is not.

But it is easy to see that the properties are preserved when replacing a metric by a strongly equivalent one. This concerns, of course, in particular the metrics in $\mathbb{E}\_n$ mentioned in Section 4.

</div>

### 7. Compact Metric Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(7.1 - Compact metric space)</span></p>

A metric space $(X, d)$ is said to be **compact** if each sequence in $(X, d)$ contains a convergent subsequence.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(7.1.1)</span></p>

Thus the compact intervals, that is the bounded closed intervals $\langle a, b \rangle$ are compact in this definition, and they are the only compact ones among the various types of intervals.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.2 - Compact subspaces are closed)</span></p>

A subspace of a compact space is compact if and only if it is closed.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* I. Let $Y \subseteq (X, d)$ be closed. Let $(y_n)_n$ be a sequence in $Y$. In $X$ it has a convergent subsequence $(y_{k_n})_n$ convergent in $X$, and the limit, by closedness, is in $Y$.

II. Let $Y$ not be closed. Then there is a sequence $(y_n)_n$ in $Y$ convergent in $X$ such that $y = \lim_n y_n \notin Y$. Then $(y_n)_n$ cannot have a subsequence convergent in $Y$ since each subsequence converges to $y$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.3 - Compact subspaces of arbitrary spaces are closed)</span></p>

Let $(X, d)$ be arbitrary and let a subspace $Y$ of $X$ be compact. Then $Y$ is closed in $(X, d)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* Let $(y_n)_n$ be a sequence in $Y$ convergent in $X$ to a limit $y$. Then each subsequence of $(y_n)_n$ converges to $y$ and hence $y \in Y$. $\square$

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(7.4 - Bounded metric space)</span></p>

A metric space $(X, d)$ is said to be **bounded** if there is a constant $K$ such that $\forall x, y \in X$, $d(x, y) < K$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.4.1 - Compact spaces are bounded)</span></p>

Each compact metric space is bounded.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* Suppose not. Choose $x_1$ arbitrarily and then $x_n$ so that $d(x_1, x_n) > n$. The sequence $(x_n)_n$ has no convergent subsequence: if $x$ were a limit of such a subsequence we would have infinitely many members of this subsequence closer to $x_1$ than $d(x_1, n) + 1$, a contradiction. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(7.5 - Products of compact spaces)</span></p>

A product of finitely many compact metric spaces is compact.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* By 5.4 it suffices to prove the statement for two spaces.

Let $(X, d_1)$, $(Y, d_2)$ be compact and let $((x_n, y_n))_n$ be a sequence in $X \times Y$. Choose a convergent subsequence $(x_{k_n})_n$ of $(x_n)_n$ and a convergent subsequence $(y_{k_{l_n}})_n$ of $(y_{k_n})_n$. Then by 5.2

$$((x_{k_{l_n}}, y_{k_{l_n}}))_n$$

is a convergent subsequence of $((x_n, y_n))_n$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(7.6 - Heine-Borel for $\mathbb{E}_n$)</span></p>

A subspace of the Euclidean space $\mathbb{E}\_n$ is compact if and only if it is bounded and closed.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* I. A compact subspace of any metric space is closed by 7.3 and bounded by 7.4.1.

II. Now let $Y \subseteq \mathbb{E}_n$ be bounded and closed. Since it is bounded we have for a sufficiently large compact interval $Y \subseteq J^n \subseteq \mathbb{E}_n$. Now by 7.5 $J^n$ is compact and since $Y$ is closed in $\mathbb{E}_n$ it is also closed in $J^n$ and hence compact by 7.2. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.7 - Compact implies complete)</span></p>

Each compact space is complete.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* A Cauchy sequence has by compactness a convergent subsequence and hence it converges, by 6.2. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.8 - Continuous image of compact is compact)</span></p>

Let $f : (X, d) \to (Y, d')$ be a continuous mapping and let $A \subseteq X$ be compact. Then $f[A]$ is compact.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* Let $(y_n)\_n$ be a sequence in $f[A]$. Choose $x_n \in A$ such that $y_n = f(x_n)$. Let $(x\_{k\_n})\_n$ be a convergent subsequence of $(x_n)\_n$. Then $(y\_{k\_n})\_n = (f(x_\{k\_n}))\_n$ is by 3.1.2 a convergent subsequence of $(x_n)_n$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.9 - Extreme value theorem)</span></p>

Let $(X, d)$ be compact. Then a continuous function $f : (X, d) \to \mathbb{R}$ attains a maximum and a minimum.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* By 7.8, $Y = f[X] \subseteq \mathbb{R}$ is compact. Hence it is bounded by 7.4.1 and it has to have a supremum $M$ and an infimum $m$. We have obviously $d(m, Y) = d(M, Y) = 0$ and since $Y$ is closed, $m, M \in Y$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(7.9.1)</span></p>

Let all the values of a continuous function on a compact space be positive. Then there is a $c > 0$ such that all the values of $f$ are greater or equal $c$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(7.10 - Continuous bijection on compact is homeomorphism)</span></p>

Let $(X, d)$ be compact and let $f : (X, d) \to (Y, d')$ be a one-to-one and onto continuous map. Then $f$ is a homeomorphism.

More generally, let $f : (X, d) \to (Y, d')$ be an onto continuous map, let $g : (X, d) \to (Z, d'')$ be a continuous map, and let $h : (Y, d') \to (Z, d'')$ be such that $h \circ f = g$. Then $h$ is continuous.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* We will prove the second statement: the first one follows by setting $g = \mathrm{id}_Y$.

Let $B$ be closed in $Z$. Then $A = g^{-1}[B]$ is closed and hence compact in $X$ and hence $f[A]$ is compact and hence closed in $Y$. Since $f$ is onto we have $f[f^{-1}[C]] = C$ for any $C$. Thus,

$$h^{-1}[B] = f[f^{-1}[h^{-1}[B]]] = f[(h \circ f)^{-1}[B]] = f[g^{-1}[B]] = f[A]$$

is closed. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(7.11 - Uniform continuity on compact spaces)</span></p>

Let $(X, d)$ be a compact space. Then a mapping $f : (X, d) \to (Y, d')$ is continuous if and only if it is uniformly continuous.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* Let $f$ not be uniformly continuous. We will prove it is not continuous either.

Since the formula for uniform continuity does not hold we have an $\varepsilon_0 > 0$ such that for every $\delta > 0$ there are $x(\delta), y(\delta)$ such that $d(x(\delta), y(\delta)) < \delta$ while $d'(f(x(\delta)), f(y(\delta))) \ge \varepsilon_0$. Set $x_n = x(\frac{1}{n})$ and $y_n = y(\frac{1}{n})$. Choose convergent subsequences $(\widetilde{x}\_n)\_n$, $(\widetilde{y}\_n)\_n$ (first choose a convergent subsequence $(x_{k_n})\_n$ of $(x_n)\_n$ then a convergent subsequence $(y\_{k\_{l\_n}})_n$ of $(y\_{n\_k})\_k$ and finally set $\widetilde{x}\_n = x\_{k\_{l\_n}}$ and $\widetilde{y}\_n = y\_{k\_{l\_n}}$). Then $d(\widetilde{x}\_n, \widetilde{y}\_n) < \frac{1}{n}$ and hence $\lim \widetilde{x}\_n = \lim \widetilde{y}\_n$. Because of $d'(f(\widetilde{x}\_n), f(\widetilde{y}\_n)) \ge \varepsilon_0$, however, we cannot have $\lim f(\widetilde{x}\_n) = \lim f(\widetilde{y}\_n)$ so that by 3.1.2, $f$ is not continuous. $\square$

</details>
</div>

## XIV. Partial Derivatives and Total Differential. Chain Rule

### 1. Conventions

We will work with real functions of several real variables, that is, with mappings $f : D \to \mathbb{R}$ where the domain $D$ is a subset of $\mathbb{E}_n$. When taking derivatives, $D$ will be typically open. Sometimes we will also have closed domains, usually closures of open sets with transparent boundaries.

We already know (recall XIII.1) that the behaviour of such functions cannot be reduced to that of functions of one variable obtained by fixing all the variables but one. But this will not prevent us from such fixings in some constructions (for instance already in the definition of partial derivative in the next section).

**Convention.** To simplify notation, we will often use bold-face letters to indicate points of the Euclidean space $\mathbb{E}_n$ (that is, $n$-tuples of real numbers, real arithmetic vectors). For example, we will write

$$\mathbf{x} \text{ for } (x_1, \ldots, x_n) \quad \text{or} \quad \mathbf{A} \text{ for } (A_1, \ldots, A_n).$$

We will also write $\mathbf{o}$ for $(0, 0, \ldots, 0)$. The scalar product of vectors $\mathbf{x}$, $\mathbf{y}$, that is, $\sum_{j=1}^{n} x_j y_j$, can be written as $\mathbf{xy}$.

**Extending the convention.** The "bold face" convention will be also used for **vector functions**, that is,

$$\mathbf{f} = (f_1, \ldots, f_m) : D \to \mathbb{E}_m, \quad f_j : D \to \mathbb{R}.$$

Note that here there is no problem with continuity: $\mathbf{f}$ is continuous iff all the $f_i$ are continuous (recall XIII.5.3).

**Composition.** Vector functions $\mathbf{f} : D \to \mathbb{E}_m$, $D \subseteq \mathbb{E}_n$, and $\mathbf{g} : D' \to \mathbb{E}_k$, $D \subseteq \mathbb{E}_n$ can be composed if $\mathbf{f}[D] \subseteq D'$, and we shall write

$$\mathbf{g} \circ \mathbf{f} : D \to \mathbb{E}_k, \quad \text{(if there is no danger of confusion, just } \mathbf{gf} : D \to \mathbb{E}_k\text{)},$$

Note that, similarly like with real functions of one real variable, we refrain from pedantic renaming the $\mathbf{f}$ when restricted to a map $D \to D'$.

### 2. Partial Derivatives

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2.1 - Partial derivative)</span></p>

Let $f : D \to \mathbb{R}$ be a real function of $n$ variables. Consider the functions

$$\phi_k(t) = f(x_1, \ldots, x_{k-1}, t, x_{k+1}, \ldots, x_n), \quad \text{all } x_j \text{ with } j \neq k \text{ fixed.}$$

The **partial derivative** of $f$ by $x_k$ (at the point $(x_1, \ldots, x_n)$) is the (ordinary) derivative of the function $\phi_k$, that is, the limit

$$\lim_{h \to 0} \frac{f(x_1, \ldots, x_{k-1}, x_k + h, x_{k+1}, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}.$$

One sometimes speaks of the $k$-th partial derivative of $f$ but one has to be careful not to confuse this expression with a derivative of higher order.

The standard notation is

$$\frac{\partial f(x_1, \ldots, x_n)}{\partial x_k} \quad \text{or} \quad \frac{\partial f}{\partial x_k}(x_1, \ldots, x_n),$$

in case of denoting variables by different letters, say $f(x, y)$, we write, of course, $\frac{\partial f(x,y)}{\partial x}$ and $\frac{\partial f(x,y)}{\partial y}$, etc.

</div>

Similarly as with the standard derivative it can happen (and typically it does) that a partial derivative $\frac{\partial f(x_1, \ldots, x_n)}{\partial x_k}$ exists for all $(x_1, \ldots, x_n)$ in some domain $D'$. In such a case, we have a function $\frac{\partial f}{\partial x_k} : D' \to \mathbb{R}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(2.3 - Partial derivatives do not imply continuity)</span></p>

The function $f$ from XIII.1.2 has both partial derivatives in every point $(x, y)$. Thus we see that unlike the standard derivative of a real function with one real variable, the existence of partial derivatives does not imply continuity. For calculus in several variables we will need a stronger concept. It will be discussed in the next section.

</div>

### 3. Total Differential

Recall VI.1.5. The formula $f(x + h) - f(x) = Ah$ (we are neglecting the "small part" $\lvert h \rvert \cdot \mu(h)$) expresses the line tangent to the curve $\lbrace (t, f(t)) \mid t \in D \rbrace$ at the point $(x, f(x))$. Or, it can be viewed as a linear approximation of the function in the vicinity of this point.

Now think of a function $f(x, y)$ in this vein (the problem with more than two variables is the same) and consider the surface

$$S = \lbrace (t, u, f(t, u)) \mid (t, u) \in D \rbrace.$$

The two partial derivatives express the directions of two tangent lines to $S$ in the point $(x, y, f(x, y))$,

* but not the tangent plane (and only that would be a desirable extension of the fact in VI.1.5),
* and do not provide any linear approximation of the function.

This will be mended by the concept of total differential.

**The norm.** For a point $\mathbf{x} \in \mathbb{E}\_n$ we define the norm $\lVert \mathbf{x} \rVert$ as the distance of $\mathbf{x}$ from $\mathbf{o}$. Thus, we will typically use the formula $\lVert \mathbf{x} \rVert = \max_i \lvert x_i \rvert$ (but $\lVert \mathbf{x} \rVert = \sum_{i=1}^{n} \lvert x_i \rvert$ or the standard Pythagorean $\lVert \mathbf{x} \rVert = \sqrt{\mathbf{x} \cdot \mathbf{x}}$ would yield the same results, recall XIII.4).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(3.3 - Total differential)</span></p>

We say that $f(x_1, \ldots, x_n)$ has a **total differential** at a point $\mathbf{a} = (a_1, \ldots, a_n)$ if there exists a function $\mu$ continuous in a neighborhood $U$ of $\mathbf{o}$ which satisfies $\mu(\mathbf{o}) = 0$ (in another, equivalent, formulation, one requires $\mu$ to be defined in $U \setminus \lbrace \mathbf{o} \rbrace$ and satisfy $\lim_{\mathbf{h} \to \mathbf{o}} \mu(\mathbf{h}) = 0$), and numbers $A_1, \ldots, A_n$ such that

$$f(\mathbf{a} + \mathbf{h}) - f(\mathbf{a}) = \sum_{k=1}^{n} A_k h_k + \lVert \mathbf{h} \rVert \mu(\mathbf{h}).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(3.3.1)</span></p>

1. Using the scalar product we may write $f(\mathbf{a} + \mathbf{h}) - f(\mathbf{a}) = \mathbf{A}\mathbf{a} + \lVert \mathbf{h} \rVert \mu(\mathbf{h})$.

2. Note that we have not defined a total differential as an entity, only the property of a function "to have a total differential". We will leave it at that.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.4 - Total differential implies continuity and partial derivatives)</span></p>

Let a function $f$ have a total differential at a point $\mathbf{a}$. Then

1. $f$ is continuous in $\mathbf{a}$.
2. $f$ has all the partial derivatives in $\mathbf{a}$, with values $\frac{\partial f(\mathbf{a})}{\partial x_k} = A_k$.


</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

1. We have 

$$\lvert f(\mathbf{x} - \mathbf{y}) \rvert \le \lvert \mathbf{A}(\mathbf{x} - \mathbf{y}) \rvert + \lvert \mu(\mathbf{x} - \mathbf{y}) \rvert \lVert \mathbf{x} - \mathbf{y} \rVert$$ 

and the limit of the right hand side for $\mathbf{y} \to \mathbf{x}$ is obviously 0.

2. We have 
  
$$\frac{1}{h}(f(x_1, \ldots, x_{k-1}, x_k + h, x_{k+1}, \ldots, x_n) - f(x_1, \ldots, x_n)) = A_k + \mu((0, \ldots, 0, h, 0, \ldots, 0))\frac{\lVert(0, \ldots, h, \ldots, 0)\rVert}{h}$$

and the limit of the right hand side is clearly $A_k$. $\square$

</details>
</div>

**Linear approximation.** Now we have a linear approximation: the formula

$$f(x_1 + h_1, \ldots, x_n + h_n) - f(x_1, \ldots, x_n) = f(\mathbf{a} + \mathbf{h}) - f(\mathbf{a}) = \sum_{k=1}^{n} A_k h_k + \lVert \mathbf{h} \rVert \mu(\mathbf{h})$$

can be interpreted as saying that in a small neighborhood of $\mathbf{a}$, the function $f$ is well approximated by the linear function

$$L(x_1, \ldots, x_n) = f(a_1, \ldots, a_n) + \sum A_k(x_k - a_k).$$

By the required properties of $\mu$, the error is much smaller than the difference $\mathbf{x} - \mathbf{a}$.

In case of just one variable, there is no difference between having a derivative at a point $a$ and having a total differential at the same point (recall VI.1.5). In case of more than one variable, however, the difference between having all partial derivatives and having a total differential at a point is tremendous.

What is happening geometrically is this: If we think of a function $f$ as represented by its "graph", the hypersurface

$$S = \lbrace (x_1, \ldots, x_n, f(x_1, \ldots, x_n)) \mid (x_1, \ldots, x_n) \in D \rbrace \subseteq \mathbb{E}_{n+1},$$

the partial derivatives describe just the tangent lines in the directions of the coordinate axes (recall 3.1), while the total differential describes the entire tangent hyperplane.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.6 - Continuous partial derivatives imply total differential)</span></p>

Let $f$ have continuous partial derivatives in a neighborhood of a point $\mathbf{a}$. Then $f$ has a total differential at $\mathbf{a}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let 

$$\mathbf{h}^{(0)} = \mathbf{h}$$

$$\mathbf{h}^{(1)} = (0, h_2, \ldots, h_n)$$

$$\mathbf{h}^{(2)} = (0, 0, h_3, \ldots, h_n)$$

$$\dots$$

(so that $\mathbf{h}^{(n)} = \mathbf{o}$). Then we have

$$f(\mathbf{a} + \mathbf{h}) - f(\mathbf{a}) = \sum_{k=1}^{n} (f(\mathbf{a} + \mathbf{h}^{(k-1)}) - f(\mathbf{a} + \mathbf{h}^{(k)})) = M.$$

By Lagrange Theorem (VII.2.2), there are $0 \le \theta_k \le 1$ such that

$$f(\mathbf{a} + \mathbf{h}^{(k-1)}) - f(\mathbf{a} + \mathbf{h}^{(k)}) = \frac{\partial f(a_1, \ldots, a_k + \theta_k h_k, \ldots, a_n)}{\partial x_k} h_k$$

and hence we can proceed

$$
\begin{aligned}
M &= \sum \frac{\partial f(a_1, \ldots, a_k + \theta_k h_k, \ldots, a_n)}{\partial x_k} h_k \\
&= \sum \frac{\partial f(\mathbf{a})}{\partial x_k} h_k + \lVert \mathbf{h} \rVert \sum \left(\frac{\partial f(a_1, \ldots, a_k + \theta_k h_k, \ldots, a_n)}{\partial x_k} - \frac{\partial f(\mathbf{a})}{\partial x_k}\right)\frac{h_k}{\lVert \mathbf{h} \rVert}
\end{aligned}
$$

Set 

$$
\mu(\mathbf{h}) = \sum \left(\frac{\partial f(a_1, \ldots, a_k + \theta_k h_k, \ldots, a_n)}{\partial x_k} - \frac{\partial f(\mathbf{a})}{\partial x_k}\right)\frac{h_k}{\lVert \mathbf{h} \rVert}
$$

Since $\left\lvert \frac{h_k}{\lVert \mathbf{h} \rVert}\right\rvert \le 1$ and since the functions $\frac{\partial f}{\partial x_k}$ are continuous, $\lim_{\mathbf{h} \to \mathbf{o}} \mu(\mathbf{h}) = 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(3.7 - Hierarchy of differentiability concepts)</span></p>

Thus, we can write schematically

$$\text{continuous PD} \;\Rightarrow\; \text{TD} \;\Rightarrow\; \text{PD}$$

(where PD stands for all partial derivatives and TD for total differential). Note that neither of the implications can be reversed. We have already discussed the second one; for the first one, recall that for functions of one variable the existence of a derivative at a point coincides with the existence of a total differential, while a derivative is not necessarily a continuous function even when it exists at every point of an open set.

In the rest of this chapter, simply assuming that partial derivatives exist will almost never be enough. Sometimes the existence of the total differential will suffice, but more often than not we will assume the stronger existence of continuous partial derivatives.

</div>

### 4. Higher Order Partial Derivatives. Interchangeability

Recall 2.2. When we have a function $g(\mathbf{x}) = \frac{\partial f(\mathbf{x})}{\partial x_k}$ then similarly as taking the second derivative of a function of one variable, we may consider partial derivatives of $g(\mathbf{x})$, that is, $\frac{\partial g(\mathbf{x})}{\partial x_l}$. The result, if it exists, is then denoted by

$$\frac{\partial^2 f(\mathbf{x})}{\partial x_k \partial x_l}.$$

More generally, iterating this procedure we may obtain

$$\frac{\partial^r f(\mathbf{x})}{\partial x_{k_1} \partial x_{k_2} \ldots \partial x_{k_r}},$$

the **partial derivatives of order** $r$.

Note that the order is given by the number of taking derivatives and does not depend on repeated individual variables. Thus for example, $\frac{\partial^3 f(x, y, x)}{\partial x \partial y \partial z}$ and $\frac{\partial^3 f(x, y, x)}{\partial x \partial x \partial x}$ are derivatives of third order (even though in the former case we have taken a partial derivative by each variable only once).

To simplify notation, taking a partial derivatives by the same variable more than once consecutively may be indicated by an exponent, e.g. $\frac{\partial^5 f(x, y)}{\partial x^2 \partial y^3} = \frac{\partial^5 f(x, y)}{\partial x \partial x \partial y \partial y \partial y}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(4.2 - Mixed second order derivatives)</span></p>

Compute the "mixed" second order derivatives of the function $f(x, y) = x\sin(y^2 + x)$. We obtain, first,

$$\frac{\partial f(x,y)}{\partial x} = \sin(y^2 + x) + x\cos(y^2 + x) \quad \text{and} \quad \frac{\partial f(x,y)}{\partial y} = 2xy\cos(y^2 + x).$$

Now for the second order derivatives we get

$$\frac{\partial^2 f}{\partial x \partial y} = 2y\cos(y^2 + x) - 2xy\sin(y^2 + x) = \frac{\partial^2 f}{\partial y \partial x}.$$

</div>

Whether it is surprising or not, it suggests that higher order partial derivatives may not depend on the order of differentiation. In effect this is true -- provided all the derivatives in question are continuous (it should be noted, though, that without this assumption the equality does not necessarily hold).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(4.2.1 - Interchangeability of mixed partials)</span></p>

Let $f(x, y)$ be a function such that the partial derivatives $\frac{\partial^2 f}{\partial x \partial y}$ and $\frac{\partial^2 f}{\partial y \partial x}$ are defined and continuous in a neighborhood of a point $(x, y)$. Then we have

$$\frac{\partial^2 f(x, y)}{\partial x \partial y} = \frac{\partial^2 f(x, y)}{\partial y \partial x}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The idea of the proof is easy: we compute the second derivative in one step. This leads, as one easily sees, to computing the limit $\lim_{h \to 0} F(h)$ of the function

$$F(h) = \frac{f(x + h, y + h) - f(x, y + h) - f(x + h, y) + f(x, y)}{h^2}$$

and this is what we are going to do.

Setting 

$$\varphi_h(y) = f(x + h, y) - f(x, y)$$

$$\psi_k(x) = f(x, y + k) - f(x, y)$$

we obtain two expressions for $F(h)$:

$$F(h) = \frac{1}{h^2}(\varphi_h(y + h) - \varphi_h(y)) \quad \text{and} \quad F(h) = \frac{1}{h^2}(\psi_h(x + h) - \psi_h(x)).$$

Let us compute the first one. The function $\varphi_h$, which is a function of one variable $y$, has the derivative 

$$\varphi'_h(y) = \frac{\partial f(x + h, y)}{\partial y} - \frac{\partial f(x, y)}{\partial y}$$ 

and hence by Lagrange Formula VI.2.2, we have

$$F(h) = \frac{1}{h^2}(\varphi_h(y + h) - \varphi_h(y)) = \frac{1}{h}\varphi'_h(y + \theta_1 h) = \frac{\frac{\partial f(x + h, y + \theta_1 h)}{\partial y} - \frac{\partial f(x, y + \theta_1 h)}{\partial y}}{h}.$$

Then, using VI.2.2 again, we obtain

$$F(h) = \frac{\partial}{\partial x}\left(\frac{\partial f(x + \theta_2 h, y + \theta_1 h)}{\partial y}\right) \qquad (*)$$

for some $\theta_1, \theta_2$ between 0 and 1. Similarly, computing 

$$\frac{1}{h^2}(\psi_h(x + h) - \psi_h(x))$ $

we obtain

$$F(h) = \frac{\partial}{\partial y}\left(\frac{\partial f(x + \theta_4 h, y + \theta_2 h)}{\partial x}\right). \qquad (**)$$

Now since both $\frac{\partial}{\partial y}(\frac{\partial f}{\partial x})$ and $\frac{\partial}{\partial x}(\frac{\partial f}{\partial y})$ are continuous at the point $(x, y)$, we can compute $\lim_{h \to 0} F(h)$ from either of the formulas $(*)$ or $(**)$ and obtain

$$\lim_{h \to 0} F(h) = \frac{\partial^2 f(x, y)}{\partial x \partial y} = \frac{\partial^2 f(x, y)}{\partial y \partial x}. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(4.3 - General interchangeability)</span></p>

Let a function $f$ of $n$ variables possess continuous partial derivatives up to the order $k$. Then the values of these derivatives depend only on the number of times a partial derivative is taken in each of the individual variables $x_1, \ldots, x_n$.

</div>

Thus, under the assumption of Theorem 4.3, we can write a general partial derivative of the order $r \le k$ as

$$\frac{\partial^r f}{\partial x_1^{r_1} \partial x_2^{r_2} \ldots \partial x_n^{r_n}} \quad \text{with} \quad r_1 + r_2 + \cdots + r_n = r$$

where, of course, $r_j = 0$ is allowed and indicates the absence of the symbol $\partial x_j$.

### 5. Composed Functions and the Chain Rule

Recall the proof of the rule of the derivative for composed functions in VI.2.2.1. It was based on the "total differential formula for one variable". By an analogous procedure we will obtain the following

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(5.1 - Chain Rule, simplest form)</span></p>

Let $f(\mathbf{x})$ have a total differential at a point $\mathbf{a}$. Let real functions $g_k(t)$ have derivatives at a point $b$ and let $g_k(b) = a_k$ for all $k = 1, \ldots, n$. Put

$$F(t) = f(\mathbf{g}(t)) = f(g_1(t), \ldots, g_n(t)).$$

Then $F$ has a derivative at $b$, and

$$F'(b) = \sum_{k=1}^{n} \frac{\partial f(\mathbf{a})}{\partial x_k} \cdot g'_k(b).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Applying the formula from 3.3 we get

$$
\begin{aligned}
\frac{1}{h}(F(b + h) - F(b)) &= \frac{1}{h}(f(\mathbf{g}(b + h)) - f(\mathbf{g}(b))) \\ &= \frac{1}{h}(f(\mathbf{g}(b) + (\mathbf{g}(b + h) - \mathbf{g}(b))) - f(\mathbf{g}(b))) \\ &= \sum_{k=1}^{n} A_k \frac{g_k(b + h) - g_k(b)}{h} + \mu(\mathbf{g}(b + h) - \mathbf{g}(b))\max_k \frac{\lvert g_k(b + h) - g_k(b) \rvert}{h}
\end{aligned}
$$

We have $\lim_{h \to 0} \mu(\mathbf{g}(b + h) - \mathbf{g}(b)) = 0$ since the functions $g_k$ are continuous at $b$. Since the functions $g_k$ have derivatives, the values $\max_k \frac{\lvert g_k(b+h) - g_k(b) \rvert}{h}$ are bounded in a sufficiently small neighborhood of 0. Thus, the limit of the last summand is zero and we have

$$
\begin{aligned}
\lim_{h \to 0} \frac{1}{h}(F(b + h) - F(b)) &= \lim_{h \to 0} \sum_{k=1}^{n} A_k \frac{g_k(b + h) - g_k(b)}{h} \\ &= \sum_{k=1}^{n} A_k \lim_{h \to 0} \frac{g_k(b + h) - g_k(b)}{h} \\ &= \sum_{k=1}^{n} \frac{\partial f(\mathbf{a})}{\partial x_k} g'_k(b)
\end{aligned}
$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(5.1.1 - Chain Rule for partial derivatives)</span></p>

Let $f(\mathbf{x})$ have a total differential at a point $\mathbf{a}$. Let real functions $g_k(t_1, \ldots, t_r)$ have partial derivatives at $\mathbf{b} = (b_1, \ldots, b_r)$ and let $g_k(\mathbf{b}) = a_k$ for all $k = 1, \ldots, n$. Then the function

$$(f \circ \mathbf{g})(t_1, \ldots, t_r) = f(\mathbf{g}(t)) = f(g_1(t), \ldots, g_n(t))$$

has all the partial derivatives at $b$, and

$$\frac{\partial(f \circ \mathbf{g})(\mathbf{b})}{\partial t_j} = \sum_{k=1}^{n} \frac{\partial f(\mathbf{a})}{\partial x_k} \cdot \frac{\partial g_k(\mathbf{b})}{\partial t_j}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(5.1.2 - Why total differential is essential)</span></p>

Just possessing partial derivatives would not suffice. The assumption of the existence of total differential in 5.1 is essential and it is easy to see why. Recall the geometric intuition from 3.1 and the last paragraph of 3.5. The $n$-tuple of functions $\mathbf{g} = (g_1, \ldots, g_n)$ represents a parametrized curve in $D$, and $f \circ \mathbf{g}$ is then a curve on the hypersurface $S$. The partial derivatives of $f$ (or the tangent lines of $S$ in the directions of the coordinate axes) have in general nothing to do with the behaviour on this curve.

</div>

**The rules for multiplication and division as a consequence of the chain rule.** As we have already mentioned, the Chain Rule (including its proof) is a more or less immediate extension of the composition rule in one variable. It may come as a surprise that it includes the rules for multiplication and division.

Consider $f(x, y) = xy$. Then $\frac{\partial f}{\partial x} = y$ and $\frac{\partial f}{\partial y} = x$ and hence

$$(u(t)v(t))' = f(u(t), v(t))' = \frac{\partial f(u(t), v(t))}{\partial x}v'(t) + \frac{\partial f(u(t), v(t))}{\partial y}u'(t) = v(t) \cdot u'(t) + u(t) \cdot v'(t).$$

Similarly for $f(x, y) = \frac{x}{y}$ we have $\frac{\partial f}{\partial x} = \frac{1}{y}$ and $\frac{\partial f}{\partial y} = -\frac{x}{y^2}$ and consequently

$$\frac{u(t)}{v(t)}' = \frac{1}{v(t)}u'(t) - \frac{u(t)}{v^2(t)} = \frac{v(t)u'(t) - u(t)v'(t)}{v^2(t)}.$$

#### 5.3. Chain Rule for Vector Functions

Let us make one more step and consider in 5.1.1 a mapping $\mathbf{f} = (f_1, \ldots, f_s) : D \to \mathbb{E}_s$. Take its composition $\mathbf{f} \circ \mathbf{g}$ with a mapping $\mathbf{g} : D' \to \mathbb{E}_n$ (recall the convention in 1.4). Then we have

$$\frac{\partial(\mathbf{f} \circ \mathbf{g})}{\partial t_j} = \sum_k \frac{\partial f_i}{\partial x_k} \cdot \frac{\partial g_k}{\partial x_j}. \qquad (*)$$

It certainly has not escaped the reader's attention that the right hand side is the product of matrices

$$\left(\frac{\partial f_i}{\partial x_k}\right)_{i,k} \left(\frac{\partial g_k}{\partial x_j}\right)_{k,j}. \qquad (**)$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(5.3.1 - Jacobian matrix)</span></p>

For an $\mathbf{f} = (f_1, \ldots, f_s) : U \to \mathbb{E}_s$, $D \subseteq \mathbb{E}_n$, define $D\mathbf{f}$ as the matrix

$$D\mathbf{f} = \left(\frac{\partial f_i}{\partial x_k}\right)_{i,k}.$$

Then the chain rule in matrix form reads

$$D(\mathbf{f} \circ \mathbf{g}) = D\mathbf{f} \cdot D\mathbf{g}.$$

More explicitly, in a concrete argument $\mathbf{t}$ we have

$$D(\mathbf{f} \circ \mathbf{g})(\mathbf{t}) = D(\mathbf{f}(\mathbf{g}))(\mathbf{t}) \cdot D\mathbf{g}(\mathbf{t}).$$

Compare it with the one variable rule $(f \circ g)'(t) = f'(g(t)) \cdot g'(t)$; for $1 \times 1$ matrices we of course have $(a)(b) = (ab)$.

</div>

#### 5.4. Lagrange Formula in Several Variables

Recall that a subset $U \subseteq \mathbb{E}_n$ is said to be **convex** if

$$\mathbf{x}, \mathbf{y} \in U \;\Rightarrow\; \forall t, \; 0 \le t \le 1, \; (1 - t)\mathbf{x} + t\mathbf{y} = \mathbf{x} + t(\mathbf{y} - \mathbf{x}) \in U.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.4.1 - Lagrange formula in several variables)</span></p>

Let $f$ have continuous partial derivatives in a convex open set $U \subseteq \mathbb{E}_n$. Then for any two points $\mathbf{x}, \mathbf{y} \in D$, there exists a $\theta$ with $0 \le \theta \le 1$ such that

$$f(\mathbf{y}) - f(\mathbf{x}) = \sum_{j=1}^{n} \frac{\partial f(\mathbf{x} + \theta(\mathbf{y} - \mathbf{x}))}{\partial x_j}(y_j - x_j).$$


</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Set 

$$F(t) = f(\mathbf{x} + t(\mathbf{y} - \mathbf{x}))$$

Then $F = f \circ \mathbf{g}$ where $\mathbf{g}$ is defined by 

$$g_j(t) = x_j + t(y_j - x_j)$$

and

$$F'(t) = \sum_{j=1}^{n} \frac{\partial f(\mathbf{g}(t))}{\partial x_j} g'_j(t) = \sum_{j=1}^{n} \frac{\partial f(\mathbf{g}(t))}{\partial x_j}(y_j - x_j).$$

Hence by VII.2.2, 

$$f(\mathbf{y}) - f(\mathbf{x}) = F(1) - F(0) = F'(\theta)$$

which yields the statement. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Lagrange formula, alternative form)</span></p>

The formula is often used in the form

$$f(\mathbf{x} + \mathbf{h}) - f(\mathbf{x}) = \sum_{j=1}^{n} \frac{\partial f(\mathbf{x} + \theta\mathbf{h})}{\partial x_j} h_j.$$

Compare this with the formula for total differential.

</div>

## XV. Implicit Function Theorems

### 1. The Task

Suppose we have $m$ real functions $F_k(x_1, \ldots, x_n, y_1, \ldots, y_m)$, $k = 1, \ldots, m$, of $n + m$ variables each. Consider the system of equations

$$F_1(x_1, \ldots, x_n, y_1, \ldots, y_m) = 0$$

$$\vdots$$

$$F_m(x_1, \ldots, x_n, y_1, \ldots, y_m) = 0$$

We would like to find a solution $y_1, \ldots, y_m$. Better, using the convention of XIV.1, we have a system of $m$ equations of $m$ unknowns (the number $n$ of the variables $x_j$ is inessential)

$$F_k(\mathbf{x}, y_1, \ldots, y_m) = 0, \quad k = 1, \ldots, m \qquad (*)$$

and we are looking for solutions $y_k = f_k(\mathbf{x})$ $(= f(x_1, \ldots, x_n))$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(1.2 - The circle equation)</span></p>

Even in simplest cases we cannot expect to have necessarily a solution, not to speak of a unique one. Take for example the following single equation

$$F(x, y) = x^2 + y^2 - 1 = 0.$$

For $\lvert x \rvert > 1$ there is no $y$ with $f(x, y) = 0$. For $\lvert x_0 \rvert < 1$, we have in a sufficiently small open interval containing $x_0$ two solutions

$$f(x) = \sqrt{1 - x^2} \quad \text{and} \quad g(x) = -\sqrt{1 - x^2}.$$

This is better, but we have *two* values in each point, contradicting the definition of a function. To achieve uniqueness, we have to restrict not only the values of $x$, but *also the values of* $y$ to an interval $(y_0 - \Delta, y_0 + \Delta)$ (where $F(x_0, y_0) = 0$). That is, if we have a particular solution $(x_0, y_0)$ we have a "window"

$$(x_0 - \delta, x_0 + \delta) \times (y_0 - \Delta, y_0 + \Delta)$$

through which we see a unique solution.

But in our example there is also the case $(x_0, y_0) = (1, 0)$, where there is a unique solution, but no suitable window as above, since in every neighborhood of $(1, 0)$, there are no solutions on the right hand side of $(1, 0)$, and two solutions on the left hand side.

Note that in the critical points $(1, 0)$ and $(-1, 0)$ we have

$$\frac{\partial F}{\partial y}(1, 0) = \frac{\partial F}{\partial y}(-1, 0) = 0. \qquad (**)$$

</div>

**Summary of the situation.** In this chapter we will show that for functions $F_k$ with continuous partial derivatives the situation is not worse than in the example above:

* we will have to have some points $\mathbf{x}^0, \mathbf{y}^0$ such that $F_k(\mathbf{x}^0, \mathbf{y}^0) = 0$ to start with;
* with certain exceptions we then have "windows" $U \times V$ such that for $\mathbf{x} \in U$ there is precisely one $\mathbf{y} \in V$, that is, $y_k = f(x_1, \ldots, x_n)$, satisfying the system of equations;
* and the exceptions are natural extensions of the condition associated with the $(**)$ above: instead of $\frac{\partial F}{\partial y}(x^0, y^0) \neq 0$ we will have $\frac{\mathsf{D}(\mathbf{F})}{\mathsf{D}(\mathbf{y})}(\mathbf{x}^0, \mathbf{y}^0) \neq 0$ for something related, called Jacobian.

Furthermore, the solutions will have continuous partial derivatives as long as the $F_j$ have them.

### 2. One Equation

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.1 - Implicit Function Theorem, one equation)</span></p>

Let $F(\mathbf{x}, y)$ be a function of $n + 1$ variables defined in a neighbourhood of a point $(\mathbf{x}^0, y_0)$. Let $F$ have continuous partial derivatives up to the order $k \ge 1$ and let

$$F(\mathbf{x}^0, y_0) = 0 \quad \text{and} \quad \left\lvert \frac{\partial F(\mathbf{x}^0, y_0)}{\partial y}\right\rvert \neq 0.$$

Then there exist $\delta > 0$ and $\Delta > 0$ such that for every $\mathbf{x}$ with $\lVert \mathbf{x} - \mathbf{x}^0 \rVert < \delta$ there exists precisely one $y$ with $\lvert y - y_0 \rvert < \Delta$ such that

$$F(\mathbf{x}, y) = 0.$$

Furthermore, if we write $y = f(\mathbf{x})$ for this unique solution $y$, then the function

$$f : (x_1^0 - \delta, x_1^0 + \delta) \times \cdots \times (x_n^0 - \delta, x_n^0 + \delta) \to \mathbb{R}$$

has continuous partial derivatives up to the order $k$.

*Proof.* The norm $\lVert \mathbf{x} \rVert$ will be as in XIV.3.2, that is $\max_i \lvert x_i \rvert$. Set $U(\gamma) = \lbrace \mathbf{x} \mid \lVert \mathbf{x} - \mathbf{x}^0 \rVert < \gamma \rbrace$ and $A(\gamma) = \lbrace \mathbf{x} \mid \lVert \mathbf{x} - \mathbf{x}^0 \rVert \le \gamma \rbrace$ (the "window" we are seeking will turn out to be $U(\delta) \times (y_0 - \Delta, y_0 + \delta)$). Without loss of generality let, say, $\frac{\partial F(\mathbf{x}^0, y_0)}{\partial y} > 0$.

The first partial derivatives of $F$ are continuous and $A(\delta)$ is closed and bounded and hence compact by XIII.7.6. Hence, by XIII.7.9 there exist $a > 0$, $K$, $\delta_1 > 0$ and $\Delta > 0$ such that for all $(\mathbf{x}, y) \in U(\delta_1) \times \langle y_0 - \Delta, y_0 + \Delta \rangle$ we have

$$\frac{\partial F(\mathbf{x}, y)}{\partial y} \ge a \quad \text{and} \quad \left\lvert \frac{\partial F(\mathbf{x}, y)}{\partial x_i}\right\rvert \le K. \qquad (*)$$

**I. The function $f$:** Fix an $\mathbf{x} \in U(\delta_1)$, and define a function of one variable $y \in (y_0 - \Delta, y_0 + \Delta)$ by $\varphi_{\mathbf{x}}(y) = F(\mathbf{x}, y)$. Then $\varphi'_{\mathbf{x}}(y) = \frac{\partial F(\mathbf{x},y)}{\partial y} > 0$ and hence all $\varphi_{\mathbf{x}}(y)$ are increasing functions of $y$, and $\varphi_{\mathbf{x}_0}(y_0 - \Delta) < \varphi_{\mathbf{x}_0}(y_0) = 0 < \varphi_{\mathbf{x}_0}(y_0 + \Delta)$.

By XIV.2.5 and XIV.3.4, $F$ is continuous, and hence there is a $\delta$, $0 < \delta \le \delta_1$, such that $\forall \mathbf{x} \in U(\delta)$, $\varphi_{\mathbf{x}}(y_0 - \Delta) < 0 < \varphi_{\mathbf{x}}(y_0 + \Delta)$.

Now $\varphi_{\mathbf{x}}$ is increasing and hence one-to-one. Thus, by IV.3 there is precisely one $y \in (y_0 - \Delta, y_0 + \Delta)$ such that $\varphi_{\mathbf{x}}(y) = 0$ -- that is, $F(\mathbf{x}, y) = 0$. Denote this $y$ by $f(\mathbf{x})$.

Note this $f$ is so far just a function; we know nothing about its properties, in particular, we do not know whether it is continuous or not.

**II. The first derivatives.** Fix an index $j$, abbreviate the sequence $x_1, \ldots, x_{j-1}$ by $\mathbf{x}_b$ and the sequence $x_{j+1}, \ldots, x_n$ by $\mathbf{x}_a$; thus, we have $\mathbf{x} = (\mathbf{x}_b, x_j, \mathbf{x}_a)$. We will compute $\frac{\partial f}{\partial x_j}$ as the derivative of $\psi(t) = f(\mathbf{x}_b, t, \mathbf{x}_a)$.

By XIV.5.4.1 we have

$$0 = F(\mathbf{x}_b, t + \theta h, \mathbf{x}_a, \psi(t + h)) - F(\mathbf{x}_b, t, \mathbf{x}_a, \psi(t)) = \frac{\partial F(\ldots)}{x_j}h + \frac{\partial F(\ldots)}{\partial y}(\psi(t + h) - \psi(t))$$

and hence

$$\psi(t + h) - \psi(t) = -h \cdot \frac{\frac{\partial F(\mathbf{x}_b, t + \theta h, \mathbf{x}_a, \psi(t) + \theta(\psi(t + h) - \psi(t)))}{\partial x_j}}{\frac{\partial F(\mathbf{x}_b, t + \theta h, \mathbf{x}_a, \psi(t) + \theta(\psi(t + h) - \psi(t)))}{\partial y}} \qquad (**)$$

for some $\theta$ between 0 and 1.

Now we can infer that $f$ is continuous. From $(*)$ we obtain $\lvert \psi(t + h) - \psi(t) \rvert \le \lvert h \rvert \cdot \left\lvert \frac{K}{a}\right\rvert$.

Using this fact we can compute from $(**)$ further

$$\lim_{h \to 0} \frac{\psi(t + h) - \psi(t)}{h} = -\frac{\frac{\partial F(\mathbf{x}_b, t, \mathbf{x}_a, \psi(t))}{\partial x_j}}{\frac{\partial F(\mathbf{x}_b, t, \mathbf{x}_a, \psi(t))}{\partial y}} = -\frac{\frac{\partial F(\mathbf{x}, f(\mathbf{x}))}{\partial x_j}}{\frac{\partial F(\mathbf{x}, f(\mathbf{x}))}{\partial y}}$$

**III. The higher derivatives.** Note that we have not only proved the *existence* of the first derivative of $f$, but also the formula

$$\frac{\partial f(\mathbf{x})}{\partial x_j} = -\frac{\partial F(\mathbf{x}, f(\mathbf{x}))}{\partial x_j} \cdot \left(\frac{\partial F(\mathbf{x}, f(\mathbf{x}))}{\partial y}\right)^{-1}. \qquad (***)$$

From this we can inductively compute the higher derivatives of $f$ (using the standard rules of differentiation) as long as the derivatives $\frac{\partial^r F}{\partial x_1^{r_1} \cdots \partial x_n^{r_n} \partial y^{r_{n+1}}}$ exist and are continuous. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(2.2 - Derivative formula via chain rule)</span></p>

We have obtained the formula $(***)$ as a by-product of the proof that $f$ has a derivative (it was useful further on, but this is not the point). Note that if we knew beforehand that $f$ had one we could deduce $(***)$ immediately from the Chain Rule. In effect, we have

$$0 \equiv F(\mathbf{x}, f(\mathbf{x}));$$

taking a derivative of both sides we obtain

$$0 = \frac{\partial F(\mathbf{x}, f(\mathbf{x}))}{\partial x_j} + \frac{\partial F(\mathbf{x}, f(\mathbf{x}))}{\partial y} \cdot \frac{\partial f(\mathbf{x})}{\partial x_j}.$$

Differentiating further, we obtain inductively linear equations from which we can compute the values of all the derivatives guaranteed by the theorem.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(2.3 - Smoothness of solutions)</span></p>

The solution $f$ in 2.1 has as many derivatives as the initial $F$ -- provided $F$ has at least the first ones. One sometimes thinks of the function itself as of its 0-th derivative. The theorem, however, *does not guarantee a continuous solution $f$ of an equation $F(x, f(x)) = 0$ with continuous $F$*. We had to use the first derivatives already for the existence of the $f$.

</div>

### 3. A Warm-Up: Two Equations

Consider a pair of equations

$$F_1(\mathbf{x}, y_1, y_2) = 0, \qquad F_2(\mathbf{x}, y_1, y_2) = 0$$

and try to find a solution $y_i = f_i(\mathbf{x})$, $i = 1, 2$, in a neighborhood of a point $(\mathbf{x}^0, y_1^0, y_2^0)$ (at which the equalities hold). We will apply the "substitution method" based on Theorem 2.1. First think of the second equation as an equation for the $y_2$; in a neighborhood of $(\mathbf{x}^0, y_1^0, y_2^0)$ we then obtain $y_2$ as a function $\psi(\mathbf{x}, y_1)$. Substitute this into the first equation to obtain

$$G(\mathbf{x}, y_1) = F_1(\mathbf{x}, y_1, \psi(\mathbf{x}, y_1));$$

if we find a solution $y_1 = f_1(\mathbf{x})$ in a neighborhood of $(\mathbf{x}^0, y_1^0)$ we can substitute it into $\psi$ and obtain $y_2 = f_2(\mathbf{x}) = \psi(\mathbf{x}, f_1(\mathbf{x}))$.

Now we have a solution let us summarize what exactly we have assumed:

-- First we had to have the continuous partial derivatives of the functions $F_i$.

-- Then, to be able to obtain $\psi$ by 2.1 the way we did, we needed to have $\frac{\partial F_2}{\partial y_2}(\mathbf{x}^0, y_1^0, y_2^0) \neq 0$.

-- Finally, we also need to have (use the Chain Rule) $0 \neq \frac{\partial G}{\partial y_1}(\mathbf{x}^0, x^0) = \frac{\partial F_1}{\partial y_1} + \frac{\partial F_1}{\partial y_2}\frac{\partial \psi}{\partial y_1}$.

Use the formula for the first derivative $\frac{\partial \psi}{\partial y_1} = -\left(\frac{\partial F_1}{\partial y_2}\right)^{-1}\frac{\partial F_2}{\partial y_1}$ from the proof of 2.1 and transform to

$$\left(\frac{\partial F_1}{\partial y_2}\right)^{-1}\left(\frac{\partial F_1}{\partial y_1}\frac{\partial F_1}{\partial y_2} - \frac{\partial F_1}{\partial y_2}\frac{\partial F_2}{\partial y_1}\right) \neq 0,$$

that is,

$$\frac{\partial F_1}{\partial y_1}\frac{\partial F_1}{\partial y_2} - \frac{\partial F_1}{\partial y_2}\frac{\partial F_2}{\partial y_1} \neq 0.$$

This is a familiar formula, namely that for a determinant. Thus we have in fact assumed that

$$\begin{vmatrix} \frac{\partial F_1}{\partial y_1}, & \frac{\partial F_1}{\partial y_2} \\ \frac{\partial F_2}{\partial y_1}, & \frac{\partial F_2}{\partial y_2} \end{vmatrix} = \det\left(\frac{\partial F_i}{\partial y_j}\right)_{i,j} \neq 0.$$

And this condition suffices: if we assume that this determinant is non-zero we have *either* $\frac{\partial F_2}{\partial y_2}(\mathbf{x}^0, y_1^0, y_2^0) \neq 0$ *and/or* $\frac{\partial F_2}{\partial y_1}(\mathbf{x}^0, y_1^0, y_2^0) \neq 0$, so if the latter holds, we can start by solving $F_2(\mathbf{x}, y_1, y_2) = 0$ for $y_1$ instead of $y_2$.

### 4. The General System

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(4.1 - Jacobi determinant)</span></p>

Let $\mathbf{F}$ be a sequence of functions $\mathbf{F}(\mathbf{x}, \mathbf{y}) = (F_1(\mathbf{x}, y_1, \ldots, y_m), \ldots, F_m(\mathbf{x}, y_1, \ldots, y_m))$. For this $\mathbf{F}$ and the sequence $\mathbf{y} = (y_1, \ldots, y_m)$ define the **Jacobi determinant** (briefly, the **Jacobian**)

$$\frac{\mathsf{D}(\mathbf{F})}{\mathsf{D}(\mathbf{y})} = \det\left(\frac{\partial F_i}{\partial y_j}\right)_{i,j=1,\ldots,m}$$

Note that if $m = 1$, that is if we have one function $F$ and one $y$, we have $\frac{\mathsf{D}(F)}{\mathsf{D}(y)} = \frac{\partial F}{\partial y}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(4.2 - Implicit Function Theorem, general system)</span></p>

Let $F_i(\mathbf{x}, y_1, \ldots, y_m)$, $i = 1, \ldots, m$, be functions of $n + m$ variables with continuous partial derivatives up to an order $k \ge 1$. Let

$$\mathbf{F}(\mathbf{x}^0, \mathbf{y}^0) = \mathbf{o}$$

and let

$$\frac{\mathsf{D}(\mathbf{F})}{\mathsf{D}(\mathbf{y})}(\mathbf{x}^0, \mathbf{y}^0) \neq 0.$$

Then there exist $\delta > 0$ and $\Delta > 0$ such that for every

$$\mathbf{x} \in (x_1^0 - \delta, x_1^0 + \delta) \times \cdots \times (x_n^0 - \delta, x_n^0 + \delta)$$

there exists precisely one

$$\mathbf{y} \in (y_1^0 - \Delta, y_1^0 + \Delta) \times \cdots \times (y_m^0 - \Delta, y_m^0 + \Delta)$$

such that $\mathbf{F}(\mathbf{x}, \mathbf{y}) = \mathbf{o}$.

Furthermore, if we write this $\mathbf{y}$ as a vector function $\mathbf{f}(\mathbf{x}) = (f_1(\mathbf{x}), \ldots, f_m(\mathbf{x}))$, then the functions $f_i$ have continuous partial derivatives up to the order $k$.

*Proof.* The procedure will follow the idea of the substitution method from Section 3. Only, we will have to do something more with determinants (but this is linear algebra, well known to the reader) and at the end we will have to tidy up the $\Delta$ and $\delta$ (which we have so far neglected).

*Proof* will be done by induction. The statement holds for $m = 1$ (see 2.1). Now let it hold for $m$, and let us have a system of equations $F_i(\mathbf{x}, \mathbf{y})$, $i = 1, \ldots, m + 1$ satisfying the assumptions (note that the unknown vector $\mathbf{y}$ is $m+1$-dimensional, too). Then, in particular, in the Jacobian determinant we cannot have a column consisting entirely of zeros, and hence, after possibly reshuffling the $F_i$'s, we can assume that

$$\frac{\partial F_{m+1}}{\partial y_{m+1}}(\mathbf{x}^0, \mathbf{y}^0) \neq 0.$$

Write $\widetilde{\mathbf{y}} = (y_1, \ldots, y_m)$; then, by the induction hypothesis, we have $\delta_1 > 0$ and $\Delta_1 > 0$ such that for $(\mathbf{x}, \widetilde{\mathbf{y}}) \in (x_1^0 - \delta_1, x_1^0 + \delta_1) \times \cdots \times (y_m^0 - \delta_1, y_m^0 + \delta_1)$, there exists precisely one $y_{m+1} = \psi(\mathbf{x}, \widetilde{\mathbf{y}})$ satisfying $F_{m+1}(\mathbf{x}, \widetilde{\mathbf{y}}, y_{m+1}) = 0$ and $\lvert y_{m+1} - y_{m+1}^0 \rvert < \Delta_1$.

This $\psi$ has continuous partial derivatives up to the order $k$ and hence so have the functions

$$G_i(\mathbf{x}, \widetilde{\mathbf{y}}) = F_i(\mathbf{x}, \widetilde{\mathbf{y}}, \psi(\mathbf{x}, \widetilde{\mathbf{y}})), \quad i = 1, \ldots, m+1$$

(the last $G_{m+1}$ is constant 0). By the Chain Rule we obtain

$$\frac{\partial G_j}{\partial y_i} = \frac{\partial F_j}{\partial y_i} + \frac{\partial F_j}{\partial y_{m+1}}\frac{\partial \psi}{\partial y_i}. \qquad (*)$$

Now consider the determinant. Multiply the last column by $\frac{\partial \psi}{\partial y_i}$ and add it to the $i$th one. By $(*)$, taking into account that $G_{m+1} \equiv 0$ and hence $\frac{\partial G_{m+1}}{\partial y_i} = \frac{\partial F_{m+1}}{\partial y_i} + \frac{\partial F_{m+1}}{\partial y_{m+1}}\frac{\partial \psi}{\partial y_i} = 0$, we obtain

$$\frac{\mathsf{D}(\mathbf{F})}{\mathsf{D}(\mathbf{y})} = \frac{\partial F_{m+1}}{\partial y_{m+1}} \cdot \frac{\mathsf{D}(G_1, \ldots, G_m)}{\mathsf{D}(y_1, \ldots, y_m)}.$$

Thus, $\frac{\mathsf{D}(G_1, \ldots, G_m)}{\mathsf{D}(y_1, \ldots, y_m)} \neq 0$ and hence by the induction hypothesis there are $\delta_2 > 0$, $\Delta_2 > 0$ such that for $\lvert x_i - x_i^0 \rvert < \delta_2$ there is a uniquely determined $\widetilde{\mathbf{y}}$ with $\lvert y_i - y_i^0 \rvert < \Delta_2$ such that $G_i(\mathbf{x}, \widetilde{\mathbf{y}}) = 0$ for $i = 1, \ldots, m$ and that the resulting $f_i(\mathbf{x})$ have continuous partial derivatives up to the order $k$. Finally defining

$$f_{i+1}(\mathbf{x}) = \psi(\mathbf{x}, f_1(\mathbf{x}), \ldots, f_m(\mathbf{x}))$$

we obtain a solution $\mathbf{f}$ of the original system of equations $\mathbf{F}(\mathbf{x}, \mathbf{y}) = \mathbf{o}$.

To finish the proof we need the constraints $\lVert \mathbf{x} - \mathbf{x}^0 \rVert < \delta$ and $\lVert \mathbf{y} - \mathbf{y}^0 \rVert < \Delta$ within which the solution is correct (that is, unique).

Choose $0 < \Delta \le \delta_1, \Delta_1, \Delta_2$ and then $0 < \delta < \delta_1, \delta_2$ and sufficiently small so that for $\lvert x_1 - x_i^0 \rvert < \delta$ one has $\lvert f_j(\mathbf{x}) - f_j(\mathbf{x}^0) \rvert < \Delta$ (the last condition makes sure to have in the $\Delta$-interval *at least one* solution). Now let

$$\mathbf{F}(\mathbf{x}, \mathbf{y}) = \mathbf{o}, \quad \lVert \mathbf{x} - \mathbf{x}^0 \rVert < \delta \text{ and } \lVert \mathbf{y} - \mathbf{y}^0 \rVert < \Delta. \qquad (**)$$

We have to prove that then necessarily $y_i = f_i(\mathbf{x})$ for all $i$. Since $\lvert x_i - x_i^0 \rvert < \delta \le \delta_1$ for $i = 1, \ldots, n$, $\lvert y_i - y_i^0 \rvert < \Delta \le \delta_1$ for $i = 1, \ldots, m$ and $\lvert y_{m+1} - y_{m+1}^0 \rvert < \Delta \le \Delta_1$ we have necessarily $y_{m+1} = \psi(\mathbf{x}, \widetilde{\mathbf{y}})$. Thus, by $(**)$, $\mathbf{G}(\mathbf{x}, \widetilde{\mathbf{y}}) = \mathbf{o}$ and since $\lvert x_i - x_i^0 \rvert < \delta \le \delta_2$ and $\lvert y_i - y_i^0 \rvert < \Delta \le \Delta_2$ we have indeed $y_i = f_i(\mathbf{x})$. $\square$

</div>

### 5. Two Simple Applications: Regular Mappings

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(5.1 - Regular mapping)</span></p>

Let $U \subseteq \mathbb{E}_n$ be an open set. Let $f_i$, $i = 1, \ldots, n$, be mappings with continuous partial derivatives (and hence continuous themselves). The resulting (continuous) mapping $\mathbf{f} = (f_1, \ldots, f_n) : U \to \mathbb{E}_n$ is said to be **regular** if

$$\frac{\mathsf{D}(\mathbf{f})}{\mathsf{D}(\mathbf{x})}(\mathbf{x}) \neq 0$$

for all $\mathbf{x} \in U$.

</div>

Recall that continuous mappings are characterized by preserving openness (or closedness) by *preimage* (recall XIII.3.7). Also recall the very special fact (XIII.7.10) that if the domain is compact, also *images* of closed sets are closed. For regular maps we have something similar.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.2 - Regular mappings are open)</span></p>

If $\mathbf{f} : U \to \mathbb{E}_n$ is regular then the image $\mathbf{f}[V]$ of every open $V \subseteq U$ is open.

*Proof.* Let $f(\mathbf{x}^0) = \mathbf{y}^0$. Define $\mathbf{F} : V \times \mathbb{E}_n \to \mathbb{E}_n$ by setting $F_i(\mathbf{x}, \mathbf{y}) = f_i(\mathbf{x}) - y_i$; then $\mathbf{F}(\mathbf{x}^0, \mathbf{y}^0) = \mathbf{o}$ and $\frac{\mathsf{D}(\mathbf{F})}{\mathsf{D}(\mathbf{x})} \neq 0$, and hence we can apply 4.2 to obtain $\delta > 0$ and $\Delta > 0$ such that for every $\mathbf{y}$ with $\lVert \mathbf{y} - \mathbf{y}^0 \rVert < \delta$, there exists a $\mathbf{x}$ such that $\lVert \mathbf{x} - \mathbf{x}^0 \rVert < \Delta$ and $F_i(\mathbf{x}, \mathbf{y}) = f_i(\mathbf{x}) - y_i = 0$. This means that we have $\mathbf{f}(\mathbf{x}) = \mathbf{y}$ (do not get confused by the reversed roles of the $x_i$ and the $y_i$: the $y_i$ are here the independent variables), and

$$\Omega(\mathbf{y}^0, \delta) = \lbrace \mathbf{y} \mid \lVert \mathbf{y} - \mathbf{y}^0 \rVert < \delta \rbrace \subseteq \mathbf{f}[V]. \quad \square$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.3 - Local invertibility of regular mappings)</span></p>

Let $\mathbf{f} : U \to \mathbb{E}_n$ be a regular mapping. Then for each $\mathbf{x}^0 \in U$ there exists an open neighborhood $V$ such that the restriction $\mathbf{f}\vert_V$ is one-to-one. Moreover, the mapping $\mathbf{g} : \mathbf{f}[V] \to \mathbb{E}_n$ inverse to $\mathbf{f}\vert_V$ is regular.

*Proof.* We will use again the mapping $\mathbf{F} = (F_1, \ldots, F_n)$ from $(*)$. For a sufficiently small $\Delta > 0$ we have precisely one $\mathbf{x} = \mathbf{g}(\mathbf{y})$ such that $\mathbf{F}(\mathbf{x}, \mathbf{y}) = 0$ and $\lVert \mathbf{x} - \mathbf{x}^0 \rVert < \Delta$. This $\mathbf{g}$ has, furthermore, continuous partial derivatives. By XIV.5.3 we have

$$D(\mathrm{id}) = D(\mathbf{f} \circ \mathbf{g}) = D(\mathbf{f}) \cdot D(\mathbf{g}).$$

By the Chain Rule (and the theorem on product of determinants)

$$\frac{\mathsf{D}(\mathbf{f})}{\mathsf{D}(\mathbf{x})} \cdot \frac{\mathsf{D}(\mathbf{g})}{\mathsf{D}(\mathbf{y})} = \det D(\mathbf{f}) \cdot \det D(\mathbf{g}) = 1$$

and hence for each $\mathbf{y} \in \mathbf{f}[V]$, $\frac{\mathsf{D}(\mathbf{g})}{\mathsf{D}(\mathbf{y})}(\mathbf{y}) \neq 0$. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(5.3.1)</span></p>

A one-to-one regular mapping $\mathbf{f} : U \to \mathbb{E}_n$ has a regular inverse $\mathbf{g} : \mathbf{f}[U] \to \mathbb{E}_n$.

</div>

### 6. Local Extremes and Extremes with Constraints

Recall looking for local extremes of a real-valued function of one real variable $f$ in VII.1. If $f$ was defined on an interval $\langle a, b \rangle$ and had a derivative in $(a, b)$ we learned by an easy application of the formula VI.1.5 that in the local extremes the derivative had to be zero. Then it sufficed to check the values in the boundary points $a$ and $b$ and we had a complete list of candidates.

Now consider the local extremes of a function of several real variables. Pinpointing possible local extremes *in the interior of its domain* is equally easy: similarly as in the function of one variable we deduce from the total differential formula (but we really do not even need that, partial derivatives would suffice) that at the points of local extreme $\mathbf{a}$, we must have

$$\frac{\partial f}{\partial x_i}(\mathbf{a}) = 0, \quad i = 1, \ldots, n. \qquad (*)$$

But the boundary is now another matter. Typically it does not consist of finitely many isolated points to be checked one at a time.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(6.1.1 - Extremes on a ball)</span></p>

Suppose we want to find the local extremes of the function $f(x, y) = x + 2y$ on the ball $B = \lbrace (x, y) \mid x^2 + y^2 \le 1 \rbrace$. The domain $B$ is compact, and hence the function $f$ certainly attains a minimum and a maximum on $B$. They cannot be in the interior of $B$: we have constantly $\frac{\partial f}{\partial x} = 1$ and $\frac{\partial f}{\partial y} = 2$; thus, the extremes must be located somewhere in the infinite set $\lbrace (x, y) \mid x^2 + y^2 = 1 \rbrace$, and the rule $(*)$ is of no use.

</div>

Hence we will try to find local extremes of a function $f(x_1, \ldots, x_n)$ *subject to certain constraints* $g_i(x_1, \ldots, x_n) = 0$, $i = 1, \ldots, k$. We have the following

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(6.2 - Lagrange Multipliers)</span></p>

Let $f, g_1, \ldots, g_k$ be real functions defined in an open set $D \subseteq \mathbb{E}_n$, and let them have continuous partial derivatives. Suppose that the rank of the matrix

$$M = \begin{pmatrix} \frac{\partial g_1}{\partial x_1}, & \ldots, & \frac{\partial g_1}{\partial x_n} \\ \vdots & & \vdots \\ \frac{\partial g_k}{\partial x_1}, & \ldots, & \frac{\partial g_k}{\partial x_n} \end{pmatrix}$$

is the largest possible, that is $k$, at each point of $D$.

If the function $f$ achieves at a point $\mathbf{a} = (a_1, \ldots, a_n)$ a local extreme subject to the constraints

$$g_i(x_1, \ldots, x_n) = 0, \quad i = 1, \ldots, k$$

then there exist numbers $\lambda_1, \ldots, \lambda_k$ such that for each $i = 1, \ldots, n$ we have

$$\frac{\partial f(\mathbf{a})}{\partial x_i} + \sum_{j=1}^{k} \lambda_j \cdot \frac{\partial g_j(\mathbf{a})}{\partial x_i} = 0.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Notes on the Lagrange Multipliers Theorem)</span></p>

1. The functions $f, g_i$ were assumed to be defined in an open $D$ so that we can take derivatives whenever we need them. In typical applications one works with functions that can be extended to an open set containing the area in question.

2. The force of the statement is in asserting the existence of $\lambda_1, \ldots, \lambda_k$ that satisfy *more than* $k$ equations. See the solution of 6.1.1 in 6.3 below.

3. The numbers $\lambda_i$ are known as **Lagrange multipliers**.

</div>

*Proof.* From linear algebra we know that a matrix $M$ has rank $k$ iff at least one of the $k \times k$ submatrices of $M$ is regular (and hence has a non-zero determinant). Without loss of generality we can assume that at the extremal point we have

$$\begin{vmatrix} \frac{\partial g_1}{\partial x_1}, & \ldots, & \frac{\partial g_1}{\partial x_k} \\ \vdots & & \vdots \\ \frac{\partial g_k}{\partial x_1}, & \ldots, & \frac{\partial g_k}{\partial x_k} \end{vmatrix} \neq 0. \qquad (1)$$

If this holds, we have by the Implicit Function Theorem in a neighborhood of the point $\mathbf{a}$ functions $\phi_i(x_{k+1}, \ldots, x_n)$ with continuous partial derivatives such that (we write $\widetilde{\mathbf{x}}$ for $(x_{k+1}, \ldots, x_n)$)

$$g_i(\phi_1(\widetilde{\mathbf{x}}), \ldots, \phi_k(\widetilde{\mathbf{x}}), \widetilde{\mathbf{x}}) = 0 \quad \text{for} \quad i = 1, \ldots, k,$$

Thus, a local maximum or a local minimum of $f(\mathbf{x})$ at $\mathbf{a}$, subject to the given constraints, implies the corresponding extreme property (without constraints) of the function

$$F(\widetilde{\mathbf{x}}) = f(\phi_1(\widetilde{\mathbf{x}}), \ldots, \phi_k(\widetilde{\mathbf{x}}), \widetilde{\mathbf{x}}),$$

at $\widetilde{\mathbf{a}}$, and hence by 5.1

$$\frac{\partial F(\widetilde{\mathbf{a}})}{\partial x_i} = 0 \quad \text{for} \quad i = k + 1, \ldots, n,$$

that is, by the Chain Rule,

$$\sum_{r=1}^{k} \frac{\partial f(\mathbf{a})}{\partial x_r}\frac{\partial \phi_r(\widetilde{\mathbf{a}})}{\partial x_i} + \frac{\partial f(\mathbf{a})}{\partial x_i} \quad \text{for} \quad i = k + 1, \ldots, n. \qquad (2)$$

Taking derivatives of the constant functions $g_i(\phi_1(\widetilde{\mathbf{x}}), \ldots, \phi_k(\widetilde{\mathbf{x}}), \widetilde{\mathbf{x}}) = 0$ we obtain for $j = 1, \ldots, k$,

$$\sum_{r=1}^{k} \frac{\partial g_j(\mathbf{a})}{\partial x_r}\frac{\partial \phi_r(\widetilde{\mathbf{a}})}{\partial x_i} + \frac{\partial g_j(\mathbf{a})}{\partial x_i} \quad \text{for} \quad i = k + 1, \ldots, n. \qquad (3)$$

Now we will use (1) again, for another purpose. Because of the rank of the matrix, the system of linear equations

$$\frac{\partial f(\mathbf{a})}{\partial x_i} + \sum_{j=1}^{n} \lambda_j \cdot \frac{\partial g_j(\mathbf{a})}{\partial x_i} = 0, \quad i = 1, \ldots, k,$$

has a unique solution $\lambda_1, \ldots, \lambda_k$. These are the equalities from the statement, but so far for $i \le k$ only. It remains to be shown that the same equalities hold also for $i > k$. In effect, by (2) and (3), for $i > k$ we obtain

$$\frac{\partial f(\mathbf{a})}{\partial x_i} + \sum_{j=1}^{n} \lambda_j \cdot \frac{\partial g_j(\mathbf{a})}{\partial x_i} = -\sum_{r=1}^{k} \frac{\partial f(\mathbf{a})}{\partial x_r}\frac{\partial \phi_r(\widetilde{\mathbf{a}})}{\partial x_i} - \sum_{j=1}^{k}\lambda_j\sum_{r=1}^{k}\frac{\partial g_j(\mathbf{a})}{\partial x_r}\frac{\partial \phi_r(\widetilde{\mathbf{a}})}{\partial x_i} = -\sum_{r=1}^{n}\left(\frac{\partial f(\mathbf{a})}{\partial x_i} + \sum_{j=1}^{n}\lambda_j \cdot \frac{\partial g_j(\mathbf{a})}{\partial x_i}\right)\frac{\partial \phi_r(\widetilde{\mathbf{a}})}{\partial x_i} = -\sum_{r=1}^{n} 0 \cdot \frac{\partial \phi_r(\widetilde{\mathbf{a}})}{\partial x_i} = 0. \quad \square$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(6.3 - Solution of 6.1.1)</span></p>

We have $\frac{\partial f}{\partial x} = 1$ and $\frac{\partial f}{\partial y} = 2$, $g(x, y) = x^2 + y^2 - 1$ and hence $\frac{\partial g}{\partial x} = 2x$ and $\frac{\partial g}{\partial y} = 2y$. There is *one* $\lambda$ that satisfies *two* equations

$$1 + \lambda \cdot 2x = 0 \quad \text{and} \quad 2 + \lambda \cdot 2y = 0.$$

This is possible only if $y = 2x$. Thus, as $x^2 + y^2 = 1$ we obtain $5x^2 = 1$ and hence $x = \pm\frac{1}{\sqrt{5}}$; this localizes the extremes to $(\frac{1}{\sqrt{5}}, \frac{2}{\sqrt{5}})$ and $(-\frac{1}{\sqrt{5}}, \frac{-2}{\sqrt{5}})$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(6.4 - Parallelepiped of largest volume for given surface area)</span></p>

The constraints $g_i$ do not necessarily come from describing boundaries. Here is an example of another nature.

Let us ask the question which rectangular parallelepiped of a given surface area has the largest volume. Denoting the lengths of the edges by $x_1, \ldots, x_n$, the surface area is

$$S(x_1, \ldots, x_n) = 2x_1 \cdots x_n\left(\frac{1}{x_1} + \cdots + \frac{1}{x_n}\right)$$

and the volume is $V(x_1, \ldots, x_n) = x_1 \cdots x_n$. Hence

$$\frac{\partial V}{\partial x_i} = \frac{1}{x_i} \cdot x_1 \cdots x_n \quad \text{and} \quad \frac{\partial S}{\partial x_i} = \frac{2}{x_i}(x_1 \cdots x_n)\left(\frac{1}{x_1} + \cdots + \frac{1}{x_n}\right) - 2x_1 \cdots x_n \frac{1}{x_i^2}.$$

If we denote $y_i = \frac{1}{x_i}$ and $s = y_1 + \cdots + y_n$, and divide the equation from the theorem by $x_1 \cdots x_n$, we obtain

$$2y_i(s - y_i) + \lambda y_i = 0 \quad \text{resulting in} \quad y_i = s + \frac{\lambda}{2}.$$

Thus, all the $x_i$ are equal and the solution is the cube.

</div>

## XVI. Multivariable Riemann Integral

The idea of Riemann integral in several variables is the same as that in one variable. The only difference is that we will have $n$-dimensional intervals instead of the standard ones, and that the partitions will have to divide such intervals in all dimensions so that the resulting intervals of the partition will not be so tidily ordered as the small intervals $\langle t_0, t_1 \rangle, \langle t_1, t_2 \rangle, \ldots$. But a finite sum is a finite sum and we will see that the ordering is not important.

What is new is the Fubini theorem (Section 4) allowing to compute multivariable integrals using integrals of one variable. All what will be done before that will be modifications of facts from Chapter XI.

### 1. Intervals and Partitions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(1.1 - $n$-dimensional compact interval)</span></p>

In this chapter, an $n$**-dimensional compact interval** is a product

$$J = \langle a_1, b_1 \rangle \times \cdots \times \langle a_n, b_n \rangle$$

(such a $J$ is indeed compact, recall XIII.7.6); if there will be no danger of confusion we will simply speak of an **interval**. We will also speak of **bricks**, in particular when they will be parts of bigger intervals.

A **partition** of $J$ is a sequence $P = (P^1, \ldots, P^n)$ of partitions

$$P^j: \quad a_j = t_{j0} < t_{j1} < \cdots < t_{j,n_j-1} < t_{j,n_j} = b_j, \quad j = 1, \ldots n.$$

The intervals $\langle t_{1,i_1}, t_{1,i_1+1} \rangle \times \cdots \times \langle t_{n,i_n}, t_{n,i_n+1} \rangle$ will be called the **bricks** of $P$ and the set of all bricks of $P$ will be denoted by $\mathcal{B}(P)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(1.2 - Volume of an interval)</span></p>

The **volume** of an interval $J = \langle a_1, b_1 \rangle \times \cdots \times \langle a_n, b_n \rangle$ is the number

$$\mathsf{vol}(J) = (b_1 - a_1)(b_2 - a_2) \cdots (b_n - a_n).$$

</div>

Since distinct bricks in $\mathcal{B}(P)$ obviously meet in a set of volume 0 (recall XI.1 applied for not necessarily planar figures) we have an

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(1.2.1)</span></p>

$\mathsf{vol}(J) = \sum\lbrace \mathsf{vol}(B) \mid B \in \mathcal{B}(J) \rbrace$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(1.3 - Diameter and mesh)</span></p>

The **diameter** of $J = \langle a_1, b_1 \rangle \times \cdots \times \langle a_n, b_n \rangle$ is

$$\mathsf{diam}(J) = \max_i(b_i - a_i)$$

and the **mesh** of a partition $P$ is

$$\mu(P) = \max\lbrace \mathsf{diam}(B) \mid B \in \mathcal{B}(P) \rbrace.$$

</div>

**Refinement.** Recall XI.2.2. A partition $Q = (Q^1, \ldots, Q^n)$ **refines** a partition $P = (P^1, \ldots, P^n)$ if every $Q^j$ refines $P^j$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(1.4.1 - Refinement and bricks)</span></p>

A refinement $Q$ of a partition $P$ induces partitions $Q_B$ of the bricks $B \in \mathcal{B}(P)$ and we have a disjoint union

$$\mathcal{B}(Q) = \bigcup\lbrace \mathcal{B}(Q_B) \mid B \in \mathcal{B}(P) \rbrace.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(1.5 - Common refinement)</span></p>

For any two partitions $P, Q$ of an $n$-dimensional compact interval $J$ there is a common refinement.

(Indeed, recall the proof of XI.2.3.2. If $P = (P^1, \ldots, P^n)$ and $Q = (Q^1, \ldots, Q^n)$ are partitions of $J$ consider the partition $R = (R^1, \ldots, R^n)$ with $R^j$ common refinements of $P^j$ and $Q^j$.)

</div>

### 2. Lower and Upper Sums. Definition of Riemann Integral

Let $f$ be a bounded real function on an $n$-dimensional compact interval $J$ and let $B \subseteq J$ be an $n$-dimensional compact subinterval of $J$ (a brick). Set

$$m(f, B) = \inf\lbrace f(\mathbf{x}) \mid \mathbf{x} \in B \rbrace \quad \text{and} \quad M(f, B) = \sup\lbrace f(\mathbf{x}) \mid \mathbf{x} \in B \rbrace.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Fact</span><span class="math-callout__name">(2.1.1)</span></p>

$m(f, B) \le M(f, B)$ and if $C \subseteq B$ then $m(f, C) \ge m(f, B)$ and $M(f, C) \le M(f, B)$.

($\lbrace f(\mathbf{x}) \mid \mathbf{x} \in C \rbrace$ is a subset of $\lbrace f(\mathbf{x}) \mid \mathbf{x} \in B \rbrace$ and hence each lower (upper) bound of the latter is a lower (upper) bound of the former.)

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2.2 - Lower and upper sums)</span></p>

Let $P$ be a partition of an interval $J$ and let $f : J \to \mathbb{R}$ be a bounded function. Set

$$s_J(f, P) = \sum\lbrace m(f, B) \cdot \mathsf{vol}(B) \mid B \in \mathcal{B}(P) \rbrace$$

$$S_J(f, P) = \sum\lbrace M(f, B) \cdot \mathsf{vol}(B) \mid B \in \mathcal{B}(P) \rbrace.$$

The subscript $J$ will be usually omitted.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.2.1 - Refinement monotonicity)</span></p>

Let a partition $Q$ refine $P$. Then

$$s(f, Q) \ge s(f, P) \quad \text{and} \quad S(f, Q) \le S(f, P).$$

*Proof.* We have (the statement used is indicated over $=$ or $\le$)

$$S(f, Q) = \sum\lbrace M(f, C) \cdot \mathsf{vol}(C) \mid C \in \mathcal{B}(Q) \rbrace \overset{1.4.1}{=} \sum\lbrace \sum\lbrace M(f, C) \cdot \mathsf{vol}(C) \mid C \in \mathcal{B}(Q_B) \rbrace \mid B \in \mathcal{B}(P) \rbrace \overset{2.1.1}{\le} \sum\lbrace M(f, B) \sum\lbrace \mathsf{vol}(C) \mid C \in \mathcal{B}(Q_B) \rbrace \mid B \in \mathcal{B}(P) \rbrace \overset{1.2.1}{=} \sum\lbrace M(f, B) \cdot \mathsf{vol}(B) \mid B \in \mathcal{B}(P) \rbrace = S(f, P).$$

Similarly for $s(f, Q)$. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.2.2)</span></p>

Let $P, Q$ be partitions of $J$. We have $s(f, P) \le S(f, Q)$.

*Proof.* For a common partition $R$ of $P, Q$ (recall 1.5) we have by 2.2.1 $s(f, P) \le s(f, R) \le S(f, R) \le S(f, Q)$. $\square$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2.3 - Riemann integral in $n$ dimensions)</span></p>

By 2.2.2 the set $\lbrace s(f, P) \mid P \text{ a partition} \rbrace$ is bounded from above and we can define the **lower Riemann integral** of $f$ over $J$ by

$$\underline{\int_J} f(\mathbf{x})\mathrm{d}\mathbf{x} = \sup\lbrace s(f, P) \mid P \text{ a partition} \rbrace;$$

similarly, the set $\lbrace S(f, P) \mid P \text{ a partition} \rbrace$ is bounded from below and we can define the **upper Riemann integral** of $f$ over $J$ by

$$\overline{\int_J} f(\mathbf{x})\mathrm{d}\mathbf{x} = \inf\lbrace S(f, P) \mid P \text{ a partition} \rbrace.$$

If the lower and upper integrals are equal we call the common value the **Riemann integral of $f$ over $J$** and denote it by

$$\int_J f(\mathbf{x})\mathrm{d}\mathbf{x} \quad \text{or simply} \quad \int_J f.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(2.3.1 - Notation)</span></p>

The integral can be also denoted e.g. by $\int_J f(x_1, \ldots, x_n)\mathrm{d}x_1, \ldots x_n$ which certainly does not surprise. The reader may encounter also symbols like

$$\int_J f(x_1, \ldots, x_n)\mathrm{d}x_1\mathrm{d}x_2 \cdots \mathrm{d}x_n.$$

This may look peculiar, but it makes more sense than meets the eyes. See 4.2 below.

</div>

Obviously we have the simple estimate

$$\inf\lbrace f(\mathbf{x}) \mid \mathbf{x} \in J \rbrace \cdot \mathsf{vol}(J) \le \underline{\int_J} f \le \overline{\int_J} f \le \sup\lbrace f(\mathbf{x}) \mid \mathbf{x} \in J \rbrace \cdot \mathsf{vol}(J).$$

### 3. Continuous Mappings

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.1 - $\varepsilon$-characterization of integrability)</span></p>

The Riemann integral $\int_J f(\mathbf{x})\mathrm{d}\mathbf{x}$ exists if and only if for every $\varepsilon > 0$ there is a partition $P$ such that

$$S_J(f, P) - s_J(f, P) < \varepsilon.$$

**Note instead of a proof.** The statement can be proved by repeating the proof of XI.2.4.2. But the reader may realize that rather than having here an easy generalization of IX.2.4.2, the statements are both special cases of a general simple statement on suprema and infima. Suppose you have a set $(X, \le)$ partially ordered by $\le$ such that for any two $x, y \in X$ there is a $z \le x, y$. If we have $\alpha : X \to \mathbb{R}$ such that $x \le y$ implies $\alpha(x) \ge \alpha(y)$ and $\beta : X \to \mathbb{R}$ such that $x \le y$ implies $\beta(x) \le \beta(y)$, and if $\alpha(x) \le \beta(x)$ for all $x, y$ then $\sup_x \alpha(x) = \inf_x \beta(x)$ iff for every $\varepsilon > 0$ there is an $x$ such that $\beta(x) < \alpha(x) + \varepsilon$. This is a trivial fact that has nothing to do with sums and such. But of course the criterion is very useful.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.2 - Integrability of continuous functions)</span></p>

For every continuous function $f : J \to \mathbb{R}$ on an $n$-dimensional compact interval the Riemann integral $\int_J f$ exists.

*Proof.* We will use the distance $\sigma$ in $\mathbb{E}_n$ defined by $\sigma(\mathbf{x}, \mathbf{y}) = \max_i \lvert x_i - y_i \rvert$.

Since $f$ is uniformly continuous we can choose for $\varepsilon > 0$ a $\delta > 0$ such that

$$\sigma(\mathbf{x}, \mathbf{y}) < \delta \;\Rightarrow\; \lvert f(\mathbf{x}) - f(\mathbf{y}) \rvert < \frac{\varepsilon}{\mathsf{vol}(J)}.$$

Recall the mesh $\mu(P)$ from 1.3. If $\mu(P) < \delta$ then $\mathsf{diam}(B) < \delta$ for all $B \in \mathcal{B}(P)$ and hence

$$M(f, B) - m(f, B) \le \sup\lbrace \lvert f(\mathbf{x}) - f(\mathbf{y}) \rvert \mid \mathbf{x}, \mathbf{y} \in B \rbrace < \frac{\varepsilon}{\mathsf{vol}(J)}$$

so that

$$S(f, P) - s(f, P) = \sum\lbrace (M(f, B) - m(f, B)) \cdot \mathsf{vol}(B) \mid B \in \mathcal{B}(P) \rbrace \le \frac{\varepsilon}{\mathsf{vol}(J)}\sum\lbrace \mathsf{vol}(B) \mid B \in \mathcal{B}(P) \rbrace = \frac{\varepsilon}{\mathsf{vol}(J)}\mathsf{vol}(J) = \varepsilon$$

by 1.2.1. Now use 3.1. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.2.1 - Limit of Riemann sums)</span></p>

Let $f : J \to \mathbb{R}$ be a continuous function and let $P_1, P_2, \ldots$ be a sequence of partitions such that $\lim_n \mu(P_n) = 0$. Then

$$\lim_n s(f, P_n) = \lim_n S(f, P_n) = \int_J f.$$

(Indeed, with $\varepsilon$ and $\delta$ as above choose an $n_0$ such that for $n \ge n_0$ we have $\mu(P_n) < \delta$.)

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(3.2.2 - Riemann sums with sample points)</span></p>

Let $f : J \to \mathbb{R}$ be a continuous function on an $n$-dimensional compact interval $J$. For every brick $B \subseteq J$ choose an element $\mathbf{x}_B \in B$ and define for a partition $P$ of $J$

$$\Sigma(f, P) = \sum\lbrace f(\mathbf{x}_B) \cdot \mathsf{vol}(B) \mid B \in \mathcal{B}(P) \rbrace.$$

Let $P_1, P_2, \ldots$ be a sequence of partitions such that $\lim_n \mu(P_n) = 0$. Then

$$\lim_n \Sigma(f, P_n) = \int_J f.$$

</div>

### 4. Fubini Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(4.1 - Fubini Theorem)</span></p>

Consider the product $J = J' \times J'' \subseteq \mathbb{E}_{m+n}$ of intervals $J' \subseteq \mathbb{E}_m$, $J'' \subseteq \mathbb{E}_n$. Let $f : J \to \mathbb{R}$ be such that $\int_J f(\mathbf{x}, \mathbf{y})\mathrm{d}\mathbf{x}\mathbf{y}$ exists and that for every $\mathbf{x} \in J'$ (resp. every $\mathbf{y} \in J''$) the integral $\int_{J''} f(\mathbf{x}, \mathbf{y})\mathrm{d}\mathbf{y}$ (resp. $\int_{J'} f(\mathbf{x}, \mathbf{y})\mathrm{d}\mathbf{x}$) exists (this holds in particular for every continuous function). Then

$$\int_J f(\mathbf{x}, \mathbf{y})\mathrm{d}\mathbf{x}\mathbf{y} = \int_{J'}\left(\int_{J''} f(\mathbf{x}, \mathbf{y})\mathrm{d}\mathbf{y}\right)\mathrm{d}\mathbf{x} = \int_{J''}\left(\int_{J'} f(\mathbf{x}, \mathbf{y})\mathrm{d}\mathbf{x}\right)\mathrm{d}\mathbf{y}.$$

*Proof.* We will discuss the first equality, the second one is analogous. Set

$$F(\mathbf{x}) = \int_{J''} f(\mathbf{x}, \mathbf{y})\mathrm{d}\mathbf{y}.$$

We will prove that $\int_{J'} F$ exists and that $\int_J f = \int_{J'} F$.

Choose a partition $P$ of $J$ such that $\int_J f - \varepsilon \le s(f, P) \le S(f, P) \le \int_J f + \varepsilon$.

This partition $P$ is obviously constituted of a partition $P'$ of $J'$ and a partition $P''$ of $J''$. We have $\mathcal{B}(P) = \lbrace B' \times B'' \mid B' \in \mathcal{B}(P'), B'' \in \mathcal{B}(P'') \rbrace$, and each brick of $P$ appears as precisely one $B' \times B''$. By 2.4

$$F(\mathbf{x}) \le \sum_{B'' \in \mathcal{B}(P'')} \max_{\mathbf{y} \in B''} f(\mathbf{x}, \mathbf{y}) \cdot \mathsf{vol}(B'')$$

and hence

$$S(F, P') \le \sum_{B' \in \mathcal{B}(P')} \left(\max_{\mathbf{x} \in B'} \sum_{B'' \in \mathcal{B}(P'')} \max_{\mathbf{y} \in B''} f(\mathbf{x}, \mathbf{y}) \cdot \mathsf{vol}(B'')\right) \cdot \mathsf{vol}(B') \le \sum_{B' \times B'' \in \mathcal{B}(P)} \max_{(\mathbf{x},\mathbf{y}) \in B' \times B''} f(\mathbf{x}, \mathbf{y}) \cdot \mathsf{vol}(B' \times B'') \le S(f, P)$$

and similarly $s(f, P) \le s(F, P')$.

Hence we have

$$\int_J f - \varepsilon \le s(F, P') \le \int_{J'} F \le S(F, P') \le \int_J f + \varepsilon$$

and therefore $\int_{J'} F$ is equal to $\int_J f$. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(4.2 - Iterated integrals)</span></p>

Let $f : J = \langle a_1, b_1 \rangle \times \cdots \times \langle a_n, b_n \rangle \to \mathbb{R}$ be a continuous function. Then

$$\int_J f(\mathbf{x})\mathrm{d}\mathbf{x} = \int_{a_n}^{b_n}\left(\cdots\left(\int_{a_2}^{b_2}\left(\int_{a_1}^{b_1} f(x_1, x_2, \ldots, x_n)\mathrm{d}x_1\right)\mathrm{d}x_2\right)\cdots\right)\mathrm{d}x_n.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Notation for iterated integrals)</span></p>

The notation $\int_J f(x_1, \ldots, x_n)\mathrm{d}x_1\mathrm{d}x_2 \cdots \mathrm{d}x_n$ mentioned in 2.3 comes, of course, from omitting the brackets in the iterated integral above.

</div>
