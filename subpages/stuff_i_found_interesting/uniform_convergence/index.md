---
layout: default
title: "Uniform Convergence"
tags:
  - analysis
  - real-analysis
  - topology
  - uniform-convergence
  - function-spaces
  - uniform-spaces
---
Uniform convergence is a strong mode of convergence for a sequence of functions where the functions approach their limit at the same rate across the entire domain. Unlike pointwise convergence, where the speed of convergence can vary dramatically from point to point, uniform convergence guarantees that the entire graph of the function eventually fits within a tight vertical band around the target limit function.


<figure>
  <img src="{{ '/assets/images/notes/stuff_i_found_interesting/Uniform_convergence.svg' | relative_url }}"" loading="lazy">
  <figcaption>A sequence of functions $f_{n}$ converges uniformly to $f$ when for arbitrary small $\varepsilon$ there is an index $N$ such that the graph of $f_n$ is in the $\varepsilon$-tube around $f$ whenever $n\geq N$.</figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/notes/stuff_i_found_interesting/Drini-nonuniformconvergence.png' | relative_url }}"" loading="lazy">
  <figcaption>The limit of a sequence of continuous functions does not have to be continuous: the sequence of functions $f_{n}(x)=\sin ^{n}(x)$ (marked in green and blue) converges pointwise over the entire domain, but the limit function is discontinuous (marked in red).</figcaption>
</figure>

## Formal Definition

A sequence of real-valued functions $f_n: X \to \mathbb{R}$ converges uniformly to a limit function $f: X \to \mathbb{R}$ on a set $X$ if for every $\epsilon > 0$, there exists a single natural number $N$ such that:

$$\vert{}f_n(x) - f(x)\vert{} < \epsilon \quad \text{for all } n \geq N \text{ and for all } x \in X$$ 

The defining aspect of this definition is that the choice of $N$ depends only on $\epsilon$ and is entirely independent of $x$.

## Pointwise vs. Uniform Convergence

The primary mathematical distinction lies in the placement of the quantifiers:

* **Pointwise Convergence:** For each individual $x \in X$ and $\epsilon > 0$, there is an $N(x, \epsilon)$. The sequence converges at each isolated point, meaning some points can converge much slower than others.
* **Uniform Convergence:** For each $\epsilon > 0$, there is a universal $N(\epsilon)$ that works simultaneously for all $x \in X$.

## The Sup-Norm Characterization

An elegant way to test for uniform convergence is by checking if the maximum distance between $f_n$ and $f$ shrinks to zero. Formally, $f_n \to f$ uniformly on $X$ if and only if:

$$\lim_{n \to \infty} \Vert{}f_n - f\Vert{}_\infty = 0$$ 

Where $\Vert{}g\Vert{}\_\infty = \sup_{x \in X} \vert{}g(x)\vert{}$ represents the supremum norm (the absolute maximum vertical distance).

## Classic Example: The Failure of Uniformity

Consider the sequence of functions $f_n(x) = x^n$ on the closed interval $X = [0, 1]$.

<div style="margin: 1.2rem 0 1.5rem;">

<p style="margin: 0 0 0.4rem 0; font-style: italic; color: #5F5E5A;">Interactive: slide to increase $n$ and watch $f_n(x)=x^n$ on $[0,1]$. The graph collapses toward $0$ on every fixed $x<1$, while staying pinned at $f_n(1)=1$.</p>

<div style="margin: 1rem 0;">
  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">
    <label for="xn-n-slider" style="font-size: 14px; color: var(--color-text-secondary, #5F5E5A); min-width: 70px;">Exponent n</label>
    <input type="range" min="0" max="11" value="3" step="1" id="xn-n-slider" style="flex: 1; min-width: 200px;" />
    <span id="xn-n-value" style="font-size: 14px; font-weight: 500; min-width: 56px; text-align: right;">10</span>
  </div>

  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 1rem;">
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">Selected exponent n</div>
      <div id="xn-stat-n" style="font-size: 20px; font-weight: 500;">10</div>
    </div>
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">sup error on [0,1]</div>
      <div id="xn-stat-sup" style="font-size: 20px; font-weight: 500;">1</div>
    </div>
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">sup on [0,0.9]</div>
      <div id="xn-stat-restricted" style="font-size: 20px; font-weight: 500;">0.349</div>
    </div>
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">width where x<sup>n</sup> &ge; 0.1</div>
      <div id="xn-stat-layer" style="font-size: 20px; font-weight: 500;">0.206</div>
    </div>
  </div>

  <div style="display: flex; flex-wrap: wrap; gap: 18px; margin-bottom: 8px; font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 2px; background: #534AB7;"></span>
      x<sup>n</sup>
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 0; border-top: 1.5px dashed rgba(194,82,87,0.75);"></span>
      pointwise limit
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 0; border-top: 1.5px dashed rgba(95,94,90,0.55);"></span>
      &epsilon; = 0.1
    </span>
  </div>

  <div style="position: relative; width: 100%; height: 340px;">
    <canvas id="xn-chart" role="img" aria-label="Plot of x to the power n on the interval from 0 to 1, controlled by a slider for n, with the discontinuous pointwise limit and epsilon guide shown.">Plot of x^n on [0,1].</canvas>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
(function () {
  const nValues = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
  const sampleCount = 900;
  const epsilon = 0.1;
  const pathColor = '#534AB7';
  const limitColor = 'rgba(194, 82, 87, 0.75)';
  const guideColor = 'rgba(95, 94, 90, 0.55)';

  const slider = document.getElementById('xn-n-slider');
  const nValueEl = document.getElementById('xn-n-value');
  const statN = document.getElementById('xn-stat-n');
  const supEl = document.getElementById('xn-stat-sup');
  const restrictedEl = document.getElementById('xn-stat-restricted');
  const layerEl = document.getElementById('xn-stat-layer');

  const limitZero = [{ x: 0, y: 0 }, { x: 1, y: 0 }];
  const limitPoint = [{ x: 1, y: 1 }];
  const epsilonLine = [{ x: 0, y: epsilon }, { x: 1, y: epsilon }];

  function formatNumber(value) {
    if (value === 0) return '0';
    if (Math.abs(value) < 0.001) return value.toExponential(2);
    return value.toFixed(3);
  }

  function buildPowerPath(n) {
    const pts = new Array(sampleCount + 1);
    for (let i = 0; i <= sampleCount; i++) {
      const x = i / sampleCount;
      pts[i] = { x: x, y: Math.pow(x, n) };
    }
    return pts;
  }

  function buildLayerLine(n) {
    const x = Math.pow(epsilon, 1 / n);
    return [{ x: x, y: 0 }, { x: x, y: epsilon }];
  }

  let chart = null;

  function update() {
    const idx = parseInt(slider.value, 10);
    const n = nValues[idx];
    const layerWidth = 1 - Math.pow(epsilon, 1 / n);
    const restrictedSup = Math.pow(0.9, n);

    nValueEl.textContent = n.toLocaleString();
    statN.textContent = n.toLocaleString();
    supEl.textContent = '1';
    restrictedEl.textContent = formatNumber(restrictedSup);
    layerEl.textContent = formatNumber(layerWidth);

    const pathData = buildPowerPath(n);
    const layerLine = buildLayerLine(n);
    const lineWidth = n <= 50 ? 1.9 : (n <= 500 ? 1.35 : 0.9);

    if (chart) {
      chart.data.datasets[0].data = pathData;
      chart.data.datasets[0].borderWidth = lineWidth;
      chart.data.datasets[4].data = layerLine;
      chart.update('none');
      return;
    }

    chart = new Chart(document.getElementById('xn-chart'), {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'x^n',
            data: pathData,
            borderColor: pathColor,
            backgroundColor: 'transparent',
            borderWidth: lineWidth,
            pointRadius: 0,
            order: 1,
          },
          {
            label: 'limit: 0 on [0,1)',
            data: limitZero,
            borderColor: limitColor,
            backgroundColor: 'transparent',
            borderWidth: 1.4,
            borderDash: [5, 4],
            pointRadius: 0,
            order: 2,
          },
          {
            label: 'limit value at x=1',
            data: limitPoint,
            type: 'scatter',
            borderColor: limitColor,
            backgroundColor: limitColor,
            pointRadius: 4,
            pointHoverRadius: 4,
            order: 0,
          },
          {
            label: 'epsilon = 0.1',
            data: epsilonLine,
            borderColor: guideColor,
            backgroundColor: 'transparent',
            borderWidth: 1.2,
            borderDash: [4, 4],
            pointRadius: 0,
            order: 3,
          },
          {
            label: 'x^n = epsilon',
            data: layerLine,
            borderColor: guideColor,
            backgroundColor: 'transparent',
            borderWidth: 1,
            borderDash: [2, 3],
            pointRadius: 0,
            order: 4,
          },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        parsing: false,
        normalized: true,
        scales: {
          x: {
            type: 'linear',
            min: 0,
            max: 1,
            title: { display: true, text: 'x', color: '#5F5E5A' },
            grid: { color: 'rgba(127,127,127,0.12)' },
            ticks: { color: '#5F5E5A' }
          },
          y: {
            min: -0.05,
            max: 1.05,
            title: { display: true, text: 'value', color: '#5F5E5A' },
            grid: { color: 'rgba(127,127,127,0.12)' },
            ticks: { color: '#5F5E5A' }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false }
        }
      }
    });
  }

  slider.addEventListener('input', update);
  update();
})();
</script>

</div>

 1. **Find the pointwise limit:** As $n \to \infty$, $x^n \to 0$ for all $x \in [0, 1)$, and $1^n \to 1$. Thus, the pointwise limit function is:
 
    $$f(x) = \begin{cases} 0 & \text{if } 0 \leq x < 1 \\ 1 & \text{if } x = 1 \end{cases}$$ 

 2. **Evaluate uniform convergence:** The functions $f_n(x) = x^n$ are entirely continuous, but the limit function $f(x)$ jumps abruptly at $x=1$. Because a uniform limit of continuous functions must be continuous, this sequence cannot converge uniformly on $[0,1]$.

If you restrict the domain to a smaller, closed subset like $[0, 0.9]$, the sequence does converge uniformly because it avoids the problematic boundary at $x=1$.

## Critical Preservation Theorems

Uniform convergence is highly prized in mathematical analysis because it preserves the structural properties of the sequence members across the limit operation:

* **Continuity (Uniform Limit Theorem):** If a sequence of continuous functions $f_n$ converges uniformly to $f$, then the limit function $f$ is also guaranteed to be continuous.
* **Integrability:** If each $f_n$ is Riemann integrable on $[a,b]$ and $f_n \to f$ uniformly, then $f$ is also integrable. Furthermore, you can swap the limit and the integral:

  $$\lim_{n \to \infty} \int_a^b f_n(x) \,dx = \int_a^b f(x) \,dx$$ 

* **Differentiability:** Swapping limits with derivatives is more delicate. If $f_n(x)$ converges at a single point, and their derivatives $f_n'(x)$ converge uniformly to a continuous function $g(x)$, then $f_n \to f$ uniformly, and $f'(x) = g(x)$.

## Key Tests for Uniform Convergence

When evaluating a series of functions $\sum_{n=1}^\infty u_n(x)$, two primary analytical tools are used:

   1. **Weierstrass M-Test:** If you can find a sequence of positive constants $M_n$ such that $\vert{}u_n(x)\vert{} \leq M_n$ for all $x \in X$, and the numerical series $\sum M_n$ converges, then the function series $\sum u_n(x)$ converges uniformly on $X$.
   2. **Dini's Theorem:** If a monotone sequence ($f_n \leq f_{n+1}$) of continuous functions converges pointwise to a continuous function $f$ on a compact (closed and bounded) set $X$, then the convergence is automatically uniform.

---

> What about uniform topology?

Uniform topology is the topology generated by a uniform structure, which generalizes the concept of a metric space to capture geometric concepts like "sameness of size," "closeness," and "uniform convergence" without relying on an actual distance function [1, 2].
Every metric space naturally induces a uniform topology, but uniform topologies can also exist on more general topological spaces (such as topological groups) where a metric is not available [1, 2].

## Why Do We Need It?

Standard topological spaces are excellent for defining local properties like continuity and pointwise convergence. However, standard topology is too weak to define global behaviors: 

* You cannot define uniform continuity because you cannot say "points $x$ and $y$ are close" globally across the entire space.
* You cannot define uniform convergence for functions mapping into a general topological space.
* You cannot define completeness (Cauchy sequences) because you cannot measure if points are "getting closer together" in a meaningful way.

A uniform topology bridges this gap by adding just enough mathematical structure (called a uniformity) to make these concepts rigorous without requiring a strict metric [1, 2].

## Formal Definition via Entourages

A uniform space is a pair $(X, \Phi)$, where $X$ is a set and $\Phi$ is a collection of subsets of $X \times X$ (called entourages or neighborhoods of the diagonal) satisfying:

   1. Contains the Diagonal: Every entourage $U \in \Phi$ contains the set $\lbrace(x,x) \mid x \in X\rbrace$. (Points are always close to themselves).
   2. Symmetry: If $U \in \Phi$, then $U^{-1} = \lbrace(y,x) \mid (x,y) \in U\rbrace \in \Phi$.
   3. Composite Property: If $U \in \Phi$, there is a $V \in \Phi$ such that $V \circ V \subseteq U$. (This is the abstract version of the triangle inequality).
   4. Intersection: If $U, V \in \Phi$, then $U \cap V \in \Phi$.
   5. Superset: If $U \in \Phi$ and $U \subseteq W \subseteq X \times X$, then $W \in \Phi$.

## Generating the Topology

The uniform topology on $X$ is uniquely defined by declaring a subset $O \subseteq X$ to be open if, for every $x \in O$, there is an entourage $U \in \Phi$ such that:

$$U[x] = \lbrace y \in X \mid (x,y) \in U\rbrace \subseteq O$$ 

The sets $U[x]$ form a neighborhood base for each point $x$ in the uniform topology.

## The Uniform Topology on Function Spaces

If you are looking at the specific space of functions $Y^X$ (all functions from a set $X$ to a metric space $Y$), the term uniform topology (or the topology of uniform convergence) refers to the topology induced by the supremum metric.

* Basis: A basic open neighborhood around a function $f$ consists of all functions $g$ such that the maximum vertical distance between $f$ and $g$ is strictly less than $\epsilon$.
* Convergence: A sequence of functions $f_n$ converges to $f$ in this specific topology if and only if $f_n$ converges to $f$ uniformly on $X$.

## Key Properties of Uniform Topologies

Uniform topologies possess highly regular geometric properties that general topological spaces often lack:

* Completely Regular: Every uniform topology is completely regular ($T_{3.5}$). If it is also a Hausdorff space ($T_2$), it can be embedded into a compact Hausdorff space (Stone–Čech compactification).
* Metrizability: A uniform topology can be turned into a standard metric space if and only if its uniformity $\Phi$ has a countable base.
* Compactness Relation: If a topological space is compact and Hausdorff, it admits exactly one unique uniformity that generates its topology.

## Summary of Geometric Hierarchy

To see where uniform topology fits in mathematics, consider this structural hierarchy from most strict to most general:

$$\text{Metric Space} \implies \text{Uniform Space} \implies \text{Topological Space}$$ 

* Metric spaces use numbers ($d(x,y) < \epsilon$) to define closeness.
* Uniform spaces use sets ($(x,y) \in U$) to define closeness globally.
* Topological spaces use neighborhoods ($x \in O$) to define closeness locally.

> Is it related to the concept of uniform convergence? Is topology generated by uniform metric?

## Relationship to Uniform Convergence

Yes, the uniform topology is directly connected to uniform convergence. In fact, it was invented to turn the concept of uniform convergence into a geometric space.
When you look at a space of functions $Y^X$ (all functions from a domain $X$ to a target metric space $Y$):

* Pointwise convergence corresponds to the Product Topology (also called the topology of pointwise convergence).
* Uniform convergence corresponds exactly to the Uniform Topology (also called the topology of uniform convergence).

A sequence of functions $f_n$ converges uniformly to $f$ if and only if $f_n$ converges to $f$ as a sequence of points inside the uniform topology.

## Is it Generated by a Uniform Metric?

Yes, but only under certain conditions. The answer depends entirely on the domain set $X$.

## 1. When the Domain is Bounded or Compact
If the target space $Y$ is bounded (or if you restrict yourself only to the space of bounded functions $\mathcal{B}(X, Y)$), the uniform topology is exactly generated by the uniform metric (commonly known as the supremum metric):

$$d_\infty(f, g) = \sup_{x \in X} d_Y(f(x), g(x))$$ 

In this scenario, the uniform topology is perfectly metrizable. The open balls of this metric are precisely the tubes or bands of radius $\epsilon$ wrapped around a function.

## 2. When the Domain is Arbitrary or Unbounded

If the target space $Y$ is unbounded (like $\mathbb{R}$) and you allow any arbitrary functions, the supremum can equal infinity ($\infty$), meaning $d_\infty$ fails to be a valid metric.

To fix this and generate the true uniform topology on all functions, mathematicians do one of two things:

 1. Standard Bounded Metric: They replace the target metric $d_Y$ with an equivalent bounded metric, such as $\bar{d}\_Y(a,b) = \min(1, d_Y(a,b))$, and then take the supremum.
 2. Pure Uniformity (No Metric): They bypass metrics completely and define the topology using the entourages discussed previously. An entourage $W_\epsilon$ is defined as the set of all function pairs that stay close everywhere:
 
    $$W_\epsilon = \lbrace (f, g) \in Y^X \times Y^X \mid d_Y(f(x), g(x)) < \epsilon \text{ for all } x \in X\rbrace$$ 

These sets $W_\epsilon$ generate the uniform topology even when a standard global metric cannot be cleanly written down.

---

> what about comparison with point-wise convergence?

To truly understand uniform convergence, it helps to compare it directly to pointwise convergence. The fundamental difference lies in whether the "speed" of convergence depends on the specific point $x$ you choose.
Here is a breakdown of how they compare conceptually, logically, and behaviorally.

## 1. The Core Intuition

* Pointwise Convergence: You pick a specific point $x$ first, and then watch the sequence of numbers $f_1(x), f_2(x), f_3(x), \dots$ converge to $f(x)$. Every point on the graph is moving toward the limit at its own individual speed. Some regions might converge instantly, while other regions lag far behind.
* Uniform Convergence: The entire function graph $f_n$ must move toward the limit function $f$ as a unified whole. It requires the convergence speed to be bounded globally across the entire domain.

## 2. The Logical Switch (Order of Quantifiers)

The mathematical distinction boils down to swapping the order of two logical statements.

* Pointwise: For every $x$ and every $\varepsilon > 0$, there exists an $N(x, \varepsilon)$...
* Uniform: For every $\varepsilon > 0$, there exists an $N(\varepsilon)$ for every $x$...

In pointwise convergence, the index $N$ depends on both how close you want to get ($\varepsilon$) and where you are looking ($x$). In uniform convergence, the index $N$ depends only on $\varepsilon$; it must work universally for every single $x$ in the domain.

## 3. Visual Comparison: The $\varepsilon$-Tube

Imagine drawing an "error boundary" or tube of radius $\varepsilon$ around your target limit function $f(x)$.

* Pointwise: As $n \to \infty$, any individual point $x$ will eventually enter the tube and stay there. However, you can always find other points on the graph that are still sitting far outside the tube.
* Uniform: There is a specific cutoff step $N$ where the entire graph of $f_n$ completely enters the $\varepsilon$-tube and never leaves it for any subsequent step.

## 4. Summary Table of Properties

Uniform convergence is a much stronger condition than pointwise convergence. Uniform convergence always implies pointwise convergence, but pointwise convergence does not guarantee uniform convergence.

| Feature | Pointwise Convergence | Uniform Convergence |
|---|---|---|
| Strictness | Weaker condition | Stronger condition |
| Dependence of $N$ | Depends on $\varepsilon$ and $x$ | Depends only on $\varepsilon$ |
| Preserves Continuity? | ❌ No. Continuous functions can converge to a discontinuous limit. | Yes. If $f_n$ are continuous, the limit $f$ is guaranteed continuous. |
| Preserves Integrals? | ❌ No. $\lim \int f_n(x) \, dx$ does not always equal $\int f(x) \, dx$. | Yes. You can safely swap the limit and the integral. |
| Preserves Derivatives? | ❌ No. Even if $f_n \to f$ pointwise, $f'_n$ might not converge to $f'$. | ⚠️ Conditionally. Requires $f'_n$ to converge uniformly. |

## 5. A Classic Analytic Example

Consider the functions $f_n(x) = \frac{x}{n}$ on the entire real line $\mathbb{R}$.

   1. Pointwise check: If you fix any specific number $x$ (e.g., $x = 1,000,000$), taking the limit as $n \to \infty$ yields $\frac{1,000,000}{\infty} = 0$. The pointwise limit is $f(x) = 0$.
   2. Uniform check: If you pick an error tolerance $\varepsilon = 1$, can you find a single $N$ that works for the whole real line? No. No matter how large you make $n$, you can always choose an $x$ that is even larger (e.g., $x = 2n$), making $f_n(x) = \frac{2n}{n} = 2$, which misses the target $0$ by more than $\varepsilon$.

Therefore, $f_n(x) = \frac{x}{n}$ converges pointwise to $0$ on $\mathbb{R}$, but it does not converge uniformly.

Would you like to see how restricting a domain (like changing $\mathbb{R}$ to a closed interval $[0, 1]$) can sometimes turn a pointwise-only sequence into a uniformly converging one?

## Historical remark

The difference between uniform convergence and pointwise convergence was not fully appreciated early in the history of calculus, leading to instances of faulty reasoning. The concept, which was first formalized by Karl Weierstrass, is important because several properties of the functions $f_{n}$ such as continuity, Riemann integrability, and, with additional hypotheses, differentiability, are transferred to the limit $f$ if the convergence is uniform, but not necessarily if the convergence is not uniform.

---

**Uniform metric:**

$$d(f, h) = \sup_{x\in E} \lVert f(x) - g(x) \rVert$$

We define uniform convergence of $(f_n)\_{n\in\mathcal{N}}$ on $E$ as simple convergence of $(f_n)\_{n\in\mathcal{N}}$ in the functional space with respect to the uniform metric (also called the supremum metric) as defined above, and symbolically it is defined by

$$f_{n}\rightrightarrows f\iff d(f_{n},f)\to 0.$$

The sequence $(f_{n})\_{n\in \mathbb {N} }$ is said to be **locally uniformly convergent** with limit $f$ if $E$ is a metric space and for every $x\in E$, there exists an $r>0$ such that $(f_{n})$ converges uniformly on $B(x,r)\cap E$ where $B(x,r)$ is a ball centered at $x$ with the radius $r$. It is clear that uniform convergence implies local uniform convergence, which implies pointwise convergence.

## To continuity

The following result states that **continuity is preserved by uniform convergence**:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Uniform limit theorem)</span></p>

Suppose $E$ is a topological space, $M$ is a metric space, and $(f_n)$ is a sequence of continuous functions $f_n:E\to M$. If $f_n \rightrightarrows f$ on $E$, then $f$ is also continuous. 

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

> This theorem is proved by the "⁠$\varepsilon /3$ trick", and is the archetypal example of this trick: to prove a given inequality (that a desired quantity is less than ⁠$\varepsilon$), one uses the definitions of continuity and uniform convergence to produce 3 inequalities (demonstrating three separate quantities are each less than ⁠$\varepsilon /3$), and then combines them via the triangle inequality to produce the desired inequality.

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

This theorem is an important one in the history of real and Fourier analysis, since many 18th century mathematicians had the intuitive understanding that a sequence of continuous functions always converges to a continuous function. The image above shows a counterexample, and **many discontinuous functions could, in fact, be written as a Fourier series of continuous functions**. The erroneous claim that the pointwise limit of a sequence of continuous functions is continuous (originally stated in terms of convergent series of continuous functions) is infamously known as "Cauchy's wrong theorem". The uniform limit theorem shows that a stronger form of convergence, uniform convergence, is needed to ensure the preservation of continuity in the limit function.

More precisely, this theorem states that the uniform limit of uniformly continuous functions is uniformly continuous; for a locally compact space, continuity is equivalent to local uniform continuity, and thus the uniform limit of continuous functions is continuous.