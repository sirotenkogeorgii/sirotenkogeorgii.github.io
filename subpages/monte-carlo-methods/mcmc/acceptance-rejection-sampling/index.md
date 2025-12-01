---
title: Acceptance−Rejection Sampling
layout: default
noindex: true
---

# Acceptance−Rejection Sampling

Suppose we want to generate samples from a continuous random variable $X$ with density $f_X(x)$. One approach, is the *inversion method*: we apply the inverse cumulative distribution function $F_X^{-1}$ to a uniform random variable $U$ on $(0,1)$. This works provided we can efficiently compute $F_X^{-1}(p)$ for any $p \in (0,1)$. In practice, however, a closed-form expression for $F_X$ is often unavailable, and its inverse is even harder to obtain, so one typically has to rely on expensive numerical methods. Extending the inversion method to joint distributions of several random variables is even more difficult. Another technique called *change-of-variables* for constructing specific distributions, but finding appropriate transformations usually requires some creativity and good fortune.

A more broadly applicable—though not always efficient—approach is **acceptance–rejection sampling**. Assume there is another random variable $Z$ with density $f_Z(x)$ that we already know how to sample from efficiently, and that $f_Z(x)$ and $f_X(x)$ are reasonably similar. More precisely, suppose we can find a constant $c \ge 1$ such that

$$f_X(x) \le c, f_Z(x) \quad \text{for all } x \in \mathbb{R}$$

Then we can obtain samples from $X$ using the following simple algorithm. Here `sample_z` is a procedure that generates samples from $Z$, and `accept_probability` is the function

$$x \mapsto \frac{f_X(x)}{c \cdot f_Z(x)}$$

Geometrically, this **vertically scales (lifts)** the curve $f_Z(x)$ so it sits **above** $f_X(x)$ everywhere. The function $cf_Z(x)$ is then called the **envelope** or **majorizing** density.

In rejection sampling, we:

1. Sample ($Z \sim f_Z$),
2. Accept that sample with probability $\frac{f_X(Z)}{c f_Z(Z)} \in [0,1]$.

$c$ is exactly the factor that "lifts" $f_Z$ enough so that $cf_Z$ dominates $f_X$ everywhere. Smaller $c$ (as close as possible to $\sup_x \frac{f_X(x)}{f_Z(x)}$) means fewer rejections and a more efficient algorithm.

**Algorithm 4.5 — Acceptance–rejection sampling**

1. **repeat**
   * Draw $Y \sim g$.
   * Draw $U \sim \mathcal{U}(0,1)$.
2. **until** $U \le \dfrac{f(Y)}{c,g(Y)}$.
3. Set $X \leftarrow Y$.
4. Output $X$.


<div class="accordion">
  <details>
    <summary>Code</summary>

```python
def rejection_sample(sample_z, f_x, f_z, c, n_samples):
    samples = []
    while len(samples) < n_samples:
        z = sample_z()          # 1. sample from proposal Z
        u = random.random()     # 2. sample U ~ Uniform(0,1)

        # 3. accept with probability f_x(z) / (c * f_z(z))
        if u < f_x(z) / (c * f_z(z)):
            samples.append(z)

    return samples
```
  </details>
</div>
