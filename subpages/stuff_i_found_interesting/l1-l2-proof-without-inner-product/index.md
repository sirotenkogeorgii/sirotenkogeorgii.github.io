---
layout: note
title: "A sum-of-squares proof of the l1–l2 comparison inequality"
date: 2024-10-20
math: true
tags:
  - mathematics
  - analysis
  - linear-algebra
  - inequalities
  - norms
---

A very standard exercise asks one to establish the comparison

$$\|x\|_1 \le \sqrt{n}\,\|x\|_2, \qquad x \in \mathbb{R}^n \text{ (or } \mathbb{C}^n),$$

between the $\ell^1$ and $\ell^2$ norms

$$\|x\|_1 = \sum_{i=1}^n |x_i|, \qquad \|x\|_2 = \Big(\sum_{i=1}^n |x_i|^2\Big)^{1/2}.$$

The reflexive move is to write the left side as the inner product $\langle \lvert x\rvert, \mathbf{1}\rangle$ of the vector of moduli against the all-ones vector $\mathbf{1}$, and then invoke Cauchy–Schwarz: $\langle \lvert x\rvert, \mathbf{1}\rangle \le \lVert x\rVert_2 \lvert\mathbf{1}\rvert_2 = \sqrt{n}\,\lvert x\rvert_2$. This is clean, but it quietly imports a fair amount of structure — an inner product, and the Cauchy–Schwarz inequality that comes with it. It is worth recording that the bound is genuinely more elementary than that, and rests on nothing beyond the fact that a real square is nonnegative.

## Reduction

Since only the moduli $a_i := \lvert x_i\rvert \ge 0$ enter either side, the real and complex cases are identical, and we may work with nonnegative reals throughout. Both sides being nonnegative, the inequality is equivalent to its square,

$$\Big(\sum_{i=1}^n a_i\Big)^2 \;\le\; n \sum_{i=1}^n a_i^2. \tag{$\ast$}$$

## The identity

The whole content of $(\ast)$ is the exact identity

$$n\sum_{i=1}^n a_i^2 \;-\; \Big(\sum_{i=1}^n a_i\Big)^2 \;=\; \sum_{1\le i<j\le n} (a_i - a_j)^2. \tag{$\dagger$}$$

To see this, expand the right-hand side. Each index $k$ is paired with the other $n-1$ indices, so $a_k^2$ occurs $n-1$ times in $\sum_{i<j}(a_i^2 + a_j^2) = (n-1)\sum_k a_k^2$; and squaring the sum gives $\big(\sum_i a_i\big)^2 = \sum_i a_i^2 + 2\sum_{i<j} a_i a_j$, so that

$$\sum_{i<j}(a_i - a_j)^2 = (n-1)\sum_k a_k^2 \;-\; \Big[\Big(\sum_i a_i\Big)^2 - \sum_i a_i^2\Big] = n\sum_i a_i^2 - \Big(\sum_i a_i\Big)^2.$$

The right-hand side of $(\dagger)$ is a sum of real squares, hence nonnegative, and $(\ast)$ follows on the spot. Taking square roots gives the claim, and the identity even hands us the equality case for free: equality holds precisely when $a_i = a_j$ for all $i,j$, i.e. when all the coordinates $\lvert x_i\rvert$ are equal. $\blacksquare$

## Remarks

1. **The constant is sharp.** Taking $x = \mathbf{1}$ gives $\lvert x\rvert_1 = n$ and $\lvert x\rvert_2 = \sqrt{n}$, so the ratio $\lvert x\rvert_1/\lvert x\rvert_2$ attains $\sqrt{n}$ and the factor cannot be improved. The opposite, trivial direction $\lvert x\rvert_2 \le \lvert x\rvert_1$ has constant $1$, and together the two bounds exhibit the equivalence of $\lvert\cdot\rvert_1$ and $\lvert\cdot\rvert_2$ explicitly. The growth of the constant with $n$ is the finite-dimensional shadow of the fact that $\ell^1$ and $\ell^2$ cease to be comparable in infinite dimensions.

2. **What $(\dagger)$ really is.** The identity is just the statement that the (scaled) variance of the $a_i$ is nonnegative: $\frac1n\sum a_i^2 - \big(\frac1n\sum a_i\big)^2 \ge 0$, which is the AM–QM inequality. So the sum-of-squares argument is nothing more than QM–AM proved from scratch, rather than cited — and Cauchy–Schwarz never appears because here it would be overkill.

3. **Jensen, if one prefers.** The same inequality drops out of convexity of $t \mapsto t^2$ via Jensen, $\big(\frac1n\sum a_i\big)^2 \le \frac1n\sum a_i^2$. This is shorter to write but less self-contained; $(\dagger)$ has the appeal of being a finite, checkable algebraic identity with the error term made fully visible.

4. **The general pattern.** For $1 \le p \le q \le \infty$ on $\mathbb{R}^n$ one has $\lvert x\rvert_q \le \lvert x\rvert_p \le n^{1/p - 1/q}\,\lvert x\rvert_q$, and the inequality above is the case $(p,q) = (1,2)$, where $n^{1/p-1/q} = n^{1/2} = \sqrt{n}$. The upper bound in general is again a convexity (power-mean) statement, and again the worst case is the constant vector.