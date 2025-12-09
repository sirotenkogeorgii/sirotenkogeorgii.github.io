---
layout: default
title: Non-Linear Dynamics and Chaos
date: 2024-10-20
# excerpt: ...
# tags:
#   - modeling
#   - statistical-physics
#   - artificial-intelligence
---

# Non-Linear Dynamics and Chaos

## 11. Fractals

### 1. Introduction

We saw that solutions of the Lorenz equations settle onto a complicated set in phase space. It is the strange attractor. As Lorenz (1963) noted, its geometry is highly unusual, like an “infinite complex of surfaces.” This chapter introduces ideas for describing such sets more precisely, drawing on **fractal geometry**.

**Fractals** are complex shapes with structure at arbitrarily small scales and often exhibit self-similarity: zooming in on a small part reveals features reminiscent of the whole, sometimes exactly but more often approximately or statistically. Our aims here are modest: to meet the simplest fractals and understand the main notions of fractal dimension. These ideas will help clarify the geometry of strange attractors.

### 1.1 Countable and Uncountable Sets

*Are some infinities larger than others?* Surprisingly, the answer is *yes*. In the late 1800s, Georg Cantor invented a clever way to compare different infinite sets. Two sets $X$ and $Y$ are said to have the same cardinality (or number of elements) if there is an invertible mapping that pairs each element $x \in X$ with precisely one $y \in Y$. Such a mapping is called a one-to-one correspondence. A familiar infinite set is the set of natural numbers $\mathbb{N}$. This set provides a basis for comparison — if another set $X$ can be put into one-to-one correspondence with the natural numbers, then $X$ is said to be **countable**. Otherwise $X$ is **uncountable**.

These definitions lead to some surprising conclusions, as the following examples show: 

**The set of even natural numbers $E = \lbrace 2,4,6,\dots\rbrace$ is countable.**   
<div class="accordion">
  <details markdown="1">
    <summary>Solution</summary>
We need to find a one-to-one correspondence between $E$ and $N$. Such a correspondence is given by the invertible mapping that pairs each natural number $n$ with the even number $2n$; thus $1 \Leftrightarrow 2, 2 \Leftrightarrow 4, 3 \Leftrightarrow 6$, and so on. 

Hence there are exactly as many even numbers as natural numbers. You might have thought that there would be only **half** as many, since all the odd numbers are missing! 
  </details>
</div>

There is an equivalent characterization of countable sets which is frequently useful. A set $X$ is countable if it can be written as a list $\lbrace x_1, x_2, x_3, \dots \rbrace$, with every $x \in X$ appearing somewhere in the list. In other words, given any $x$, there is some finite $n$ such that $x_n = x$. A convenient way to exhibit such a list is to give an algorithm that systematically counts the elements of $X$. This strategy is used in the next two examples.

**The integers are countable.**
<div class="accordion">
  <details markdown="1">
    <summary>Solution</summary>
Here’s an algorithm for listing all the integers: We start with $0$ and then work in order of increasing absolute value. Thus the list is $\lbrace 0, 1, 1, 2, 2, 3, –3, \dots \rbrace$. Any particular integer appears eventually, so the integers are countable.
  </details>
</div>

**The positive rational numbers are countable.**
<div class="accordion">
  <details markdown="1">
    <summary>Solution</summary>
Here’s a wrong way: we start listing the numbers $\frac{1}{1},\frac{1}{2},\frac{1}{3},\dots$ Unfortunately we never finish the $\frac{1}{n}$’s and so numbers like $\frac{2}{3}$ are never counted! The right way is to make a table where the $pq$-th entry is $p/q$. Then the rationals can be counted by the weaving procedure shown in Figure 11.1.1. Any given $p/q$ is reached after a finite number of steps, so the rationals are countable. 
  </details>
</div>

**Example of an uncountable set.**

### 1.2 Cantor Set