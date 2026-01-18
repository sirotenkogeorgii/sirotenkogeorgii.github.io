---
title: Acceptance−Rejection Sampling
layout: default
noindex: true
---

# Acceptance−Rejection Sampling

This sampling method can be applied to both discrete and continuous distributions, and even to multidimensional distributions — although its eﬃciency rapidly decreases in the number of dimensions.

Suppose we want to generate samples from a continuous random variable $X$ with density $f_X(x)$. One approach, is the *inversion method*: we apply the inverse cumulative distribution function $F_X^{-1}$ to a uniform random variable $U$ on $(0,1)$. This works provided we can efficiently compute $F_X^{-1}(p)$ for any $p \in (0,1)$. In practice, however, a closed-form expression for $F_X$ is often unavailable, and its inverse is even harder to obtain, so one typically has to rely on expensive numerical methods. Extending the inversion method to joint distributions of several random variables is even more difficult. Another technique called *change-of-variables* for constructing specific distributions, but finding appropriate transformations usually requires some creativity and good fortune.

A more broadly applicable—though not always efficient—approach is **acceptance–rejection sampling**. Assume there is another random variable $Z$ with density $f_Z(x)$ that we already know how to sample from efficiently, and that $f_Z(x)$ and $f_X(x)$ are reasonably similar. More precisely, suppose we can find a constant $c \ge 1$ such that

$$f_X(x) \le c \cdot f_Z(x) \quad \text{for all } x \in \mathbb{R}$$

Then we can obtain samples from $X$ using the following simple algorithm. Here `sample_z` is a procedure that generates samples from $Z$, and `accept_probability` is the function

$$x \mapsto \frac{f_X(x)}{c \cdot f_Z(x)}$$

Geometrically, this **vertically scales (lifts)** the curve $f_Z(x)$ so it sits **above** $f_X(x)$ everywhere. The function $cf_Z(x)$ is then called the **envelope** or **majorizing** density.

In rejection sampling, we:

1. Sample ($Z \sim f_Z$),
2. Accept that sample with probability $\frac{f_X(Z)}{c f_Z(Z)} \in [0,1]$.

$c$ is exactly the factor that "lifts" $f_Z$ enough so that $cf_Z$ dominates $f_X$ everywhere. Smaller $c$ (as close as possible to $\sup_x \frac{f_X(x)}{f_Z(x)}$) means fewer rejections and a more efficient algorithm.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Acceptance–rejection sampling)</span></p>

1. **repeat**
   * Draw $Y \sim f_Z$.
   * Draw $U \sim \mathcal{U}(0,1)$.
2. **until** $U \le \dfrac{f_X(Y)}{cf_Z(Y)}$.
3. Set $X \leftarrow Y$.
4. Output $X$.

</div>

<!-- <div class="accordion">
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
</div> -->


<div class="accordion">
  <details>
    <summary>Code</summary>
    <pre><code class="language-python">
def rejection_sample(sample_z, f_x, f_z, c, n_samples):
    samples = []
    while len(samples) < n_samples:
        z = sample_z()          # 1. sample from proposal Z
        u = random.random()     # 2. sample U ~ Uniform(0,1)

        # 3. accept with probability f_x(z) / (c * f_z(z))
        if u < f_x(z) / (c * f_z(z)):
            samples.append(z)

    return samples
    </code></pre>
  </details>
</div>

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/monte-carlo-methods/rejection-sampling/rejection_sampling1.png' | relative_url }}" alt="a" loading="lazy">
    <figcaption>Relatively good application of rejection sampling</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/monte-carlo-methods/rejection-sampling/rejection_sampling2.png' | relative_url }}" alt="b" loading="lazy">
    <figcaption>Less good application of rejection sampling</figcaption>
  </figure>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Is acceptance-rejection sampling a Markov chain algorithm?)</span></p>

Not in the usual sense.

**Classic acceptance–rejection (AR)** sampling is typically an **i.i.d. sampling** method: each proposal is generated independently, and once you accept, that accepted draw does **not** depend on previously accepted draws. So the sequence of accepted samples is not a Markov chain you “run”; it’s just independent samples from the target.

That said, you *can* force a Markov-chain viewpoint:

* If you define a “state” that includes the current proposal and whether it was accepted (or include the running count / the RNG state), then the *full process* is Markov. But that’s kind of trivial—almost any algorithm becomes Markov if you include enough internal variables.
* If you look only at the **accepted samples**, they are i.i.d., so they *do* satisfy the Markov property (the next sample is independent of the past), but then calling it “a Markov chain algorithm” is misleading because there’s no dependence structure to exploit.

**In contrast, MCMC (e.g., Metropolis–Hastings)** is specifically designed so that

$$X_{t+1} \sim \kappa(\cdot \mid X_t),$$

and dependence between successive $X_t$’s is essential—stationarity and convergence arguments rely on that Markov structure.

So: **AR is primarily an independent sampling method, not an MCMC/Markov-chain sampling algorithm**, even though it can be described as a Markov process in an expanded state space.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Basic rejection sampling suffers fromthe curse of dimensionality)</span></p>

Most simple **rejection sampling** techniques run into the **curse of dimensionality**: as the number of dimensions increases, the rejection rate typically rises **exponentially**, making these methods quickly impractical.

Metropolis–Hastings and other **MCMC** approaches are generally **less severely affected** by this issue. Because of that, when the target distribution is high-dimensional, MCMC methods are often among the few workable options. For this reason, they are widely used to generate samples for **hierarchical Bayesian models** and many other modern high-dimensional statistical models across a range of fields.

</div>

## Why it works

To understand why the sampling procedure works, it helps to look at its **geometric interpretation**.

### 1. Uniform sampling under the density curve

Let $U$ be a uniform random variable on $(0,1)$, independent of $X$.
Consider the point $(X, Y) = (X, U f_X(X))$.

This point is uniformly distributed in the region $S_X = \{(x,y) : x \in \mathbb{R}, 0 < y < f_X(x)\}$, which is the area under the graph of $f_X$.

This follows from the mapping $(x, u) \mapsto (x, u f_X(x))$, which is differentiable, invertible (where $f_X(x) \neq 0$), and transforms the density $f_X(x)dxdu$ into $dxdy$. Thus, $f_{X,Y} = 1_{S_X}$: uniform over the region $S_X$.

### 2. Sampling by rejection from a larger region

One way to sample $(X,Y)$ is to first sample from a **larger, simpler region** that contains $S_X$ and reject points that fall outside $S_X$.

Suppose we have another random variable $Z$ and a constant $c \ge 1$ such that $f_X(x) \le c f_Z(x) \quad \text{for all } x$.

Then the region $S_Z = \{(x,y) : x \in \mathbb{R}, 0 < y < f_Z(x)\}$ satisfies $S_X \subset c S_Z$.

### 3. Using $Z$ to simulate $X$

Sample $Z$ and an independent $U' \sim \text{Uniform}(0,1)$. Then look at the point $(Z, U' \cdot c f_Z(Z))$.

Condition on the event that this point lies in $S_X$. Given this event, the conditional distribution is uniform on $S_X$. Therefore, the resulting $Z$ has the **same distribution as $X$**.

### 4. Acceptance probability

Once we observe $Z = z$, the event $U' \cdot c f_Z(z) < f_X(z)$

occurs with probability $\mathbb{P}(U' c f_Z(Z) < f_X(Z) \mid Z=z) = \frac{f_X(z)}{c f_Z(z)}$.

Thus, we should repeatedly sample $Z$ and **accept it** with this probability.
This is exactly what the rejection sampling algorithm does.

Here is a **clean, clear, and well-structured rephrasing** of the text:


## Algorithm Efficiency

At each iteration, the probability of accepting a sample is

$$\mathbb{P}(U' , c f_Z(Z) < f_X(Z))= \int_{-\infty}^{\infty} dzf_Z(z)\mathbb{P}(U' , c f_Z(z) < f_X(z)\mid Z=z)$$

$$= \int_{-\infty}^{\infty} dz\frac{f_X(z)}{c}= \frac{1}{c}$$

Therefore, the number of iterations required to obtain a single accepted sample is a random variable $N \ge 1$ with probability mass function

$$p_N(n) = \frac{1}{c}\left(1 - \frac{1}{c}\right)^{n-1}$$

This is a **geometric distribution** with success probability $1/c$.
Its expected value is


$$\mathbb{E}[N] = c$$

Thus, the algorithm requires on average $c$ samples from $Z$ to produce one sample from $X$. Consequently, it is important that the constant $c$ not be too large—otherwise, the computational cost of sampling **$X$ scales linearly with $c$**.

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/monte-carlo-methods/rejection-sampling/geometric_rejection.png' | relative_url }}" alt="a" loading="lazy">
    <figcaption>Geometric distribution for a rejection sampling problem with acceptance probability 1/c</figcaption>
  </figure>
</div>

> Corollary: The acceptance probability and the number of iterations required to obtain a single accepted sample does not depend on the shapes of $f_X$ and $f_Z$ and depends only on the constant $c$.


* The **local acceptance behavior** depends on $f_X$ and $f_Z$.
* The **overall acceptance probability** depends only on $c$.
* This is fundamental to rejection sampling: once you find an envelope $cf_Z$, the acceptance rate is exactly $1/c$.
  
The acceptance probability is exactly: $$\frac{\text{area under } f_X}{\text{area under } c f_Z} = \frac{1}{c}$$

---

> Note: As the number of dimensions increases it will become increasingly difficult to keep the constant $c$ relatively small.