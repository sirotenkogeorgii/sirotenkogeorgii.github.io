---
layout: default
title: "Proof Techniques"
tags:
  - mathematics
  - analysis
  - proof-techniques
  - uniform-convergence
  - inequalities
---
## $\varepsilon /3$ trick

This theorem is proved by the "⁠$\varepsilon /3$ trick", and is the archetypal example of this trick: to prove a given inequality (that a desired quantity is less than ⁠$\varepsilon$), one uses the definitions of continuity and uniform convergence to produce 3 inequalities (demonstrating three separate quantities are each less than ⁠$\varepsilon /3$), and then combines them via the triangle inequality to produce the desired inequality.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Uniform limit theorem)</span></p>

Suppose $E$ is a topological space, $M$ is a metric space, and $(f_n)$ is a sequence of continuous functions $f_n:E\to M$. If $f_n \rightrightarrows f$ on $E$, then $f$ is also continuous. 

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $x_{0}\in E$ be an arbitrary point. We will prove that $f$ is continuous at $x_0$. Let $\varepsilon >0$. By uniform convergence, there exists a natural number $N$ such that

$$\forall x\in E\quad d{\bigl (}f_{N}(x),f(x){\bigr )}\leq {\tfrac {1}{3}}\varepsilon$$

(uniform convergence shows that the above statement is true for all $n\geq N$, but we will only use it for one function of the sequence, namely $f_{N}$).

It follows from the continuity of $f_{N}$ at $x_{0}\in E$ that there exists an open set $U$ containing $x_{0}$ such that 

$$\forall x\in U\quad d{\bigl (}f_{N}(x),f_{N}(x_{0}){\bigr )}\leq {\tfrac {1}{3}}\varepsilon$$.

Hence, using the triangle inequality, 

$$ \forall x\in U\quad d{\bigl (}f(x),f(x_{0}){\bigr )}\leq d{\bigl (}f(x),f_{N}(x){\bigr )}+d{\bigl (}f_{N}(x),f_{N}(x_{0}){\bigr )}+d{\bigl (}f_{N}(x_{0}),f(x_{0}){\bigr )}\leq \varepsilon$$,

which gives us the continuity of $f$ at $x_0$. $\square$

</details>
</div>

## Generalized a^2-b^2 = (a-b)(a+b)

This equality could be extended to have more general form, specifically adding the absolute value:

$$\lvert a^p - b^p \rvert = \lvert a^{p-1} - b^{p-1} \rvert (a^{p-1} + b^{p-1})$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Uniform limit theorem)</span></p>



</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>



</details>
</div>