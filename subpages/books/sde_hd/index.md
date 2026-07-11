---
layout: default
title: "SDEs & Diffusion Models"
date: 2026-04-20
excerpt: "Notes on 'SDEs & Diffusion Models' (SS 2026) — starting with a measure-theoretic outline of Brownian motion, Donsker's theorem, Markov properties, and a panoramic introduction to stochastic differential equations, Itô calculus, the Fokker–Planck equation, time-reversal of diffusions, and score-based generative modelling."
tags:
  - stochastic-differential-equations
  - brownian-motion
  - diffusion-models
  - markov-processes
  - score-matching
  - fokker-planck
  - time-reversal
  - ito-calculus
  - stochastic-integration
  - martingales
  - girsanov
  - ornstein-uhlenbeck
# math: true
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

**Table of Contents**
- TOC
{:toc}

# SDEs & Diffusion Models

## Problems

[Selected Problems](/subpages/books/sde_hd/problems/)

## 0 Brownian Motion (Outline)

### 0.1 Introduction

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">0.1 (Historical and mathematical context)</span></p>

* **Origin**: The erratic movement of pollen particles in water, first described by the botanist Robert Brown (1827).
* **Mathematical formulation**: Developed rigorously by Albert Einstein (1905) to model diffusion; further formalised by Norbert Wiener (1923), who constructed Brownian motion as a well-defined mathematical object.
* **Key significance**:
  * "Infinite-dimensional" version of the normal distribution;
  * canonical model of a continuous-time stochastic process with stationary, independent increments;
  * foundational for stochastic calculus, financial mathematics (e.g. the Black–Scholes model), PDE theory (Feynman–Kac formula), and statistical physics;
  * appears as the scaling limit of random walks (Donsker's theorem).

</div>

For simplicity, we consider **real-valued** processes throughout this chapter; all concepts generalise directly to $\mathbb{R}^m$.

### 0.2 Real-valued Gaussian Processes and Definition of Brownian Motion

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Assumption</span><span class="math-callout__name">(Probability space and Index set)</span></p>

Throughout, let 
* $(\Omega, \mathcal{A}, \mathbb{P})$ be an underlying **probability space** and 
* fix an **index set** $I \in \lbrace [0,T],\ [0,\infty) \rbrace$ ("time"), where $T > 0$ denotes a possible finite time horizon.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">0.1 (Gaussian Process)</span></p>

A real-valued stochastic process $(X_t)\_{t \in I}$ on $(\Omega, \mathcal{A}, \mathbb{P})$ is called a **Gaussian process** iff there exist 
* a **mean function** $\mu : I \to \mathbb{R}$ and 
* a **covariance function** $\gamma : I \times I \to \mathbb{R}$ such that for *any* $n \in \mathbb{N}$ and $t_1, \dots, t_n \in I$, the matrix $\bigl(\gamma(t_i, t_j)\bigr)\_{1 \le i, j \le n}$ is positive semidefinite and

$$
\begin{pmatrix} X_{t_1} \\ \vdots \\ X_{t_n} \end{pmatrix}
\sim
\mathcal{N}\!\left(
\begin{pmatrix} \mu(t_1) \\ \vdots \\ \mu(t_n) \end{pmatrix},\
\begin{pmatrix} \gamma(t_1,t_1) & \cdots & \gamma(t_1,t_n) \\ \vdots & \ddots & \vdots \\ \gamma(t_n,t_1) & \cdots & \gamma(t_n,t_n) \end{pmatrix}
\right).
$$

* If $\mu \equiv 0$ on $I$, the Gaussian process $(X_t)\_{t \in I}$ is called **centred**. 
* the index set $I$ can be anything: finite, countable, or uncountable. The reason is that the only quantifier ranges over **finite** subsets of $I$. The uncountability of $I$ is inherited from the surrounding assumption, not from the definition.

</div>

<!-- <div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Matrix Notation for Continuous Time)</span></p>

At first glance it may look suspicious to speak of a "covariance matrix" when $I$ is a continuous index set — a matrix is a finite object, while $I$ has uncountably many instants. The resolution is that the definition does not assert a single matrix for the whole process.

**What the definition actually says.** For **any** finite choice of times $t_1, \dots, t_n \in I$, the random vector $(X_{t_1}, \dots, X_{t_n})$ is multivariate Gaussian with covariance matrix $\bigl(\gamma(t_i, t_j)\bigr)\_{1 \le i, j \le n}$. The quantifier runs over *finite* subsets of $I$; each such subset yields its own finite-dimensional marginal.

**The global object is the kernel.** On the whole of continuous $I$, the fundamental object is the **covariance function** (or *kernel*)

$$
\gamma : I \times I \to \mathbb{R},
$$

which encodes the covariance of *any* two time instants. Matrices appear only as *finite-dimensional marginals* of this kernel: restricting $\gamma$ to the finite grid $\lbrace t_1, \dots, t_n \rbrace \times \lbrace t_1, \dots, t_n \rbrace$ produces the $n \times n$ matrix above.

**Link to Kolmogorov's extension theorem.** A Gaussian process is thus fully characterised by its *finite-dimensional distributions*, and each such distribution is genuinely finite-dimensional (hence a matrix). This is precisely the setup of Kolmogorov's extension theorem (Lemma 0.2), which assembles a continuous-time process from a consistent family of finite-dimensional marginals.

</div> -->

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The existence of Gaussian Process guaranteed by Kolmogorov's extension theorem)</span></p>

The existence of such processes on a suitable probability space is guaranteed by **Kolmogorov's extension theorem**, whose discussion we omit at this point.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">0.2 (Existence of Gaussian Processes)</span></p>

For any mean function $\mu : I \to \mathbb{R}$ and covariance function $\gamma : I \times I \to \mathbb{R}$ (with $\gamma$ positive semidefinite), there exist a probability space and a Gaussian process on it with mean $\mu$ and covariance $\gamma$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation for Brownian motion</span><span class="math-callout__name">(From Discrete Random Walk To Brownian Motion)</span></p>

We would like a random walk with "infinitesimal" time steps modelling the movement of a particle. Let $Z_i \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0,1)$ for $i \in \mathbb{N}$. For any $n \in \mathbb{N}$, define the process $W^n := (W^n_t)\_{t \in I}$ by

$$
W^n_t := \frac{1}{\sqrt{n}} \sum_{i=1}^{\lfloor nt \rfloor} Z_i \sim \mathcal{N}\!\left(0, \frac{\lfloor nt \rfloor}{n}\right) \approx \mathcal{N}(0, t), \qquad t \in I,
$$

which we interpret as a random walk with step size $Z_i/\sqrt{n}$ and variance $1/n$. Here $t \in I$ plays the role of time, while $n \in \mathbb{N}$ is only an auxiliary parameter indicating how finely we partition time.

**Two ways to read the parameters.** The definition above involves two knobs; keeping one fixed and varying the other gives complementary pictures of what $W^n_t$ *is*.

* **Fix $t$, vary $n$ (grid refinement at a fixed time).** For a fixed horizon $t$:
  * the number of steps executed up to time $t$ is $\lfloor n t \rfloor$, growing linearly in $n$;
  * each individual step has magnitude $Z_i/\sqrt{n}$, shrinking like $1/\sqrt{n}$;
  * the per-step variance is $1/n \to 0$;
  * yet the *total* variance at $t$ is $\lfloor n t \rfloor / n \to t$ — **invariant under refinement**.
  
  So $n$ controls *how finely we subdivide* the interval $[0,t]$: larger $n$ = more, smaller steps, but the aggregate variance at $t$ stays $\approx t$. This is the **Brownian scaling**: $\sqrt{n}$ in the denominator is the unique factor that keeps $\sum_{i=1}^{\lfloor nt \rfloor} Z_i / \sqrt{n}$ bounded in distribution (with variance $\to t$) as $n \to \infty$, preventing the walk from either diverging or collapsing to a point.

* **Fix $n$, vary $t$ (walking through time on a fixed grid).** With $n$ held fixed, $t \mapsto W^n_t$ is a *step function*: on each plateau $t \in \bigl[k/n,\,(k+1)/n\bigr)$ we have $\lfloor n t \rfloor = k$, so
  
  $$
  W^n_t \;=\; \frac{1}{\sqrt{n}} \sum_{i=1}^{k} Z_i \qquad \text{is constant on the plateau,}
  $$
  
  and at $t = (k+1)/n$ a new i.i.d. Gaussian increment $Z_{k+1}/\sqrt{n}$ is added, producing a jump. So with $n$ fixed, the variable $t$ indexes the **partial sum** of the walk up to position $\lfloor n t \rfloor$; advancing $t$ by $1/n$ appends exactly one step.

Taken together, $n$ is the *refinement* dial and $t$ is the *walk-so-far* dial; the limit $n \to \infty$ sends step size and per-step variance to zero while keeping the aggregate variance at each $t$ equal to $t$. We then wish to take

$$
\text{``}W_t := \lim_{n \to \infty} W^n_t\text{''},
$$

so that the particle can change direction at *any* instant $t \in I$. Mathematically, we define a stochastic process that inherits every property this formal limit should possess.

</div>

<div style="margin: 1.2rem 0 1.5rem;">

<p style="margin: 0 0 0.4rem 0; font-style: italic; color: #5F5E5A;">Interactive: slide to refine $n$; each setting redraws the step-function path $W^n_t$ on $[0,1]$ against the limit's $\pm\sqrt{t}$ and $\pm 2\sqrt{t}$ envelopes. Click "Resample $Z_i$" to draw a new i.i.d. sequence.</p>

<!-- <h2 class="sr-only">Interactive visualization of the discrete process $W_t^n$ as a step function on $[0,1]$, with a slider controlling $n$ and reference bands at $\pm sqrt(t)$ and plus/minus two $sqrt(t)$.</h2> -->

<div style="margin: 1rem 0;">
  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">
    <label for="n-slider" style="font-size: 14px; color: var(--color-text-secondary, #5F5E5A); min-width: 70px;">Steps n</label>
    <input type="range" min="0" max="11" value="4" step="1" id="n-slider" style="flex: 1; min-width: 200px;" />
    <span id="n-value" style="font-size: 14px; font-weight: 500; min-width: 56px; text-align: right;">20</span>
    <button id="resample" style="padding: 6px 14px;">Resample Z<sub>i</sub></button>
  </div>

  <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 1rem;">
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">Number of steps n</div>
      <div id="stat-n" style="font-size: 20px; font-weight: 500;">20</div>
    </div>
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">Step size 1/&radic;n</div>
      <div id="stat-step" style="font-size: 20px; font-weight: 500;">0.224</div>
    </div>
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">Per-step variance 1/n</div>
      <div id="stat-var" style="font-size: 20px; font-weight: 500;">0.0500</div>
    </div>
  </div>

  <div style="display: flex; flex-wrap: wrap; gap: 18px; margin-bottom: 8px; font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 2px; background: #534AB7;"></span>
      path of W<sub>t</sub><sup>n</sup>
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 0; border-top: 1.5px dashed rgba(83,74,183,0.65);"></span>
      &plusmn;&radic;t (1&sigma; of limit)
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 0; border-top: 1.5px dashed rgba(83,74,183,0.3);"></span>
      &plusmn;2&radic;t (2&sigma;)
    </span>
  </div>

  <div style="position: relative; width: 100%; height: 340px;">
    <canvas id="wchart" role="img" aria-label="Step-function plot of the random walk W_t^n on the interval from 0 to 1, with dashed reference bands at plus and minus square root of t and plus and minus two square root of t.">Plot of the discrete random walk W_t^n as a step function on [0,1].</canvas>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
(function () {
  const nValues = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
  const maxN = nValues[nValues.length - 1];

  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  let Z = new Array(maxN);
  function resample() {
    for (let i = 0; i < maxN; i++) Z[i] = randn();
  }
  resample();

  const slider = document.getElementById('n-slider');
  const nValueEl = document.getElementById('n-value');
  const statN = document.getElementById('stat-n');
  const statStep = document.getElementById('stat-step');
  const statVar = document.getElementById('stat-var');
  const resampleBtn = document.getElementById('resample');

  const bandPts = 120;
  const upperBand2 = [], lowerBand2 = [], upperBand1 = [], lowerBand1 = [];
  for (let i = 0; i <= bandPts; i++) {
    const t = i / bandPts;
    const sd = Math.sqrt(t);
    upperBand2.push({ x: t, y: 2 * sd });
    lowerBand2.push({ x: t, y: -2 * sd });
    upperBand1.push({ x: t, y: sd });
    lowerBand1.push({ x: t, y: -sd });
  }

  function buildPath(n) {
    const invSqrtN = 1 / Math.sqrt(n);
    const pts = new Array(n + 1);
    pts[0] = { x: 0, y: 0 };
    let S = 0;
    for (let i = 1; i <= n; i++) {
      S += Z[i - 1];
      pts[i] = { x: i / n, y: S * invSqrtN };
    }
    return pts;
  }

  const pathColor = '#534AB7';
  const band1Color = 'rgba(83, 74, 183, 0.6)';
  const band2Color = 'rgba(83, 74, 183, 0.28)';

  let chart = null;

  function update() {
    const idx = parseInt(slider.value, 10);
    const n = nValues[idx];
    nValueEl.textContent = n.toLocaleString();
    statN.textContent = n.toLocaleString();
    statStep.textContent = (1 / Math.sqrt(n)).toFixed(3);
    statVar.textContent = (1 / n).toFixed(4);

    const pathData = buildPath(n);
    const lineWidth = n <= 20 ? 1.8 : (n <= 200 ? 1.3 : 0.9);

    if (chart) {
      chart.data.datasets[0].data = pathData;
      chart.data.datasets[0].borderWidth = lineWidth;
      chart.update('none');
      return;
    }

    chart = new Chart(document.getElementById('wchart'), {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'W_t^n',
            data: pathData,
            borderColor: pathColor,
            backgroundColor: 'transparent',
            borderWidth: lineWidth,
            pointRadius: 0,
            stepped: 'after',
            order: 1,
          },
          { label: '+2√t', data: upperBand2, borderColor: band2Color, borderWidth: 1, borderDash: [5, 4], pointRadius: 0, fill: false, order: 4 },
          { label: '-2√t', data: lowerBand2, borderColor: band2Color, borderWidth: 1, borderDash: [5, 4], pointRadius: 0, fill: false, order: 5 },
          { label: '+√t',  data: upperBand1, borderColor: band1Color, borderWidth: 1, borderDash: [3, 3], pointRadius: 0, fill: false, order: 2 },
          { label: '-√t',  data: lowerBand1, borderColor: band1Color, borderWidth: 1, borderDash: [3, 3], pointRadius: 0, fill: false, order: 3 },
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
            title: { display: true, text: 't', color: '#5F5E5A' },
            grid: { color: 'rgba(127,127,127,0.12)' },
            ticks: { color: '#5F5E5A' }
          },
          y: {
            min: -3,
            max: 3,
            title: { display: true, text: 'W_t^n', color: '#5F5E5A' },
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
  resampleBtn.addEventListener('click', () => { resample(); update(); });

  update();
})();
</script>

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">0.3 (Standard Brownian Motion)</span></p>

A real-valued stochastic process $W := (W_t)\_{t \in I}$ on $(\Omega, \mathcal{A}, \mathbb{P})$ is called a **one-dimensional standard Brownian motion** iff it satisfies:

* **(W1) Starting in $0$:** $W_0 = 0$, $\mathbb{P}$-a.s.
* **(W2) Independent increments:** For any $n \in \mathbb{N}$ and $0 =: t_0 < t_1 < \dots < t_n$ in $I$, the increments $W_{t_1} - W_{t_0},\, W_{t_2} - W_{t_1},\, \dots,\, W_{t_n} - W_{t_{n-1}}$ are mutually stochastically independent.
* **(W3) Normal increments:** For any $s < t$ in $I$, $W_t - W_s \sim \mathcal{N}(0, t - s)$.
* **(W4) Continuous paths:** The paths $t \mapsto W_t$ are $\mathbb{P}$-a.s. continuous on $I$.

</div>

<!-- <div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Notation: $\mathbb{P}$-a.s.)</span></p>

The abbreviation "$\mathbb{P}$-a.s." stands for **"$\mathbb{P}$-almost surely"**, i.e. *with probability $1$ under the measure $\mathbb{P}$*. Formally, a statement holds $\mathbb{P}$-a.s. iff the set of $\omega \in \Omega$ on which it **fails** is contained in a measurable set of probability zero — that is, there exists $N \in \mathcal{A}$ with $\mathbb{P}(N) = 0$ such that the statement holds for every $\omega \in \Omega \setminus N$.

**Applied to (W4).** A Brownian motion $W$ is a collection of random variables $\lbrace W_t : t \in I \rbrace$ on $(\Omega, \mathcal{A}, \mathbb{P})$. For each fixed $\omega \in \Omega$ one obtains a deterministic realisation — the **sample path** —

$$
t \;\longmapsto\; W_t(\omega), \qquad t \in I,
$$

which is an ordinary function $I \to \mathbb{R}$. The continuity statement

> the paths $t \mapsto W_t$ are $\mathbb{P}$-a.s. continuous on $I$

is an assertion about this random function: there exists a measurable null set $N \in \mathcal{A}$ with $\mathbb{P}(N) = 0$ such that for **every** $\omega \in \Omega \setminus N$, the path $t \mapsto W_t(\omega)$ is continuous on $I$ in the usual $\varepsilon$–$\delta$ sense. Equivalently,

$$
\mathbb{P}\!\left(\, \lbrace \omega \in \Omega \,:\, t \mapsto W_t(\omega) \text{ is continuous on } I \rbrace \,\right) = 1.
$$

**Why "almost" and not "every"?** A null set of "bad" $\omega$ is unavoidable and harmless:

* Measure theory is blind to sets of probability zero — modifying $W$ on a null set yields a statistically indistinguishable process.
* In infinite-dimensional settings one generally cannot construct processes with *every* sample path continuous while preserving all other desired properties. The standard convention is therefore to demand continuity only on a set of full measure.

**Contrast with other "almost" notions.** The qualifier $\mathbb{P}$ matters: almost-sure statements always refer to a specific probability measure — changing the measure (e.g. via Girsanov's theorem) can change which events are negligible. Moreover, the *order of quantifiers* is crucial:

* *Pointwise a.s. continuity at a single $t$*: $\mathbb{P}(W_s \to W_t \text{ as } s \to t) = 1$. The null set is allowed to depend on $t$ — strictly weaker.
* *Uniform (path-wise) a.s. continuity on $I$*: one null set $N$ works for **all** $t \in I$ simultaneously. This is the stronger statement demanded by (W4).

</div> -->

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">0.4 (Standard vs. Non-Standard; Wiener Process)</span></p>

* The term "standard" refers to the absence of an additional variance factor. A real-valued process $W$ is a **(non-standard) Brownian motion** if it satisfies (W1), (W2), (W4) from Definition 0.3 together with

  **(W3')** for any $s < t$ in $I$ and some $\sigma > 0$,
  
  $$
  W_t - W_s \sim \mathcal{N}(0, \sigma \cdot (t - s)).
  $$

* The letter $W$ reflects the alternative — and very common — name **Wiener process**.

<figure>
  <img src="{{ 'assets/images/notes/sdes_diffusion_models/wiener_process_different_sigmas.png' | relative_url }}" alt="a" loading="lazy">
</figure>

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name"></span></p>

&rarr; **Main issue:** *existence*. Note that (W1)–(W3) characterise only the *finite-dimensional* distributions, whereas (W4) imposes a condition on *every* $t \in I$, a stronger requirement.

</div>

### 0.3 Existence of Brownian Motion and Path Properties

The starting point is the characterisation of Brownian motion as a Gaussian process.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">0.5 (Characterisation via Gaussian Processes)</span></p>

A real-valued stochastic process $W := (W_t)\_{t \in I}$ on $(\Omega, \mathcal{A}, \mathbb{P})$ is a one-dimensional standard Brownian motion if and only if

* **(W123)** $W$ is a centred Gaussian process with covariance function 
  
  $$\gamma(s, t) := \mathrm{Cov}[W_s, W_t] = s \wedge t = \min(s,t) \quad \forall s, t \in I$$

* **(W4)** the paths $t \mapsto W_t$ are $\mathbb{P}$-a.s. continuous on $I$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* See Exercise 1.1.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Centred Gaussian Process must be modified)</span></p>

Combining Proposition 0.5 with Lemma 0.2, we obtain a stochastic process that fulfils (W1), (W2), (W3). However, this process is *not* automatically guaranteed to satisfy (W4) — we must *modify* it.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">0.6 (Versions / Modifications / Indistinguishability)</span></p>

Let $X := (X_t)\_{t \in I}$ and $Y := (Y_t)\_{t \in I}$ be two stochastic processes on $(\Omega, \mathcal{A}, \mathbb{P})$.

(a) $X$ and $Y$ are **versions** (or **modifications**) of each other iff

$$\mathbb{P}(X_t = Y_t) = 1 \quad \text{for all } t \in I.$$

(b) $X$ and $Y$ are **indistinguishable** iff

$$\mathbb{P}(X_t = Y_t \text{ for all } t \in I) = 1.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Indistinguishability is stronger)</span></p>

Note that **indistinguishability is strictly stronger**: it requires $\mathbb{P}$-almost-sure equality *simultaneously* on the (uncountable) whole index set, whereas a version only gives almost-sure equality *pointwise* in $t$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Versions but not indistinguishable)</span></p>

Take $(\Omega, \mathcal{A}, \mathbb{P}) := ([0,1], \mathcal{B}([0,1]), \mathrm{Leb})$ and let $T(\omega) := \omega$, so $T \sim \mathrm{Unif}([0,1])$. On $I := [0,1]$ define

$$
X_t(\omega) := 0, \qquad Y_t(\omega) := \mathbf{1}_{\lbrace t \,=\, T(\omega)\rbrace} = \mathbf{1}_{\lbrace t \,=\, \omega\rbrace}.
$$

* **Versions** *(pointwise in $t$).* For every fixed $t \in I$ the bad set $\lbrace \omega : Y_t(\omega) \neq 0 \rbrace = \lbrace t \rbrace$ is a single point, hence Lebesgue-null:
  
  $$
  \mathbb{P}(X_t = Y_t) \;=\; \mathbb{P}(T \neq t) \;=\; 1.
  $$

* **Not indistinguishable** *(uniformly in $t$).* For *every* $\omega \in \Omega$ the path $Y_\cdot(\omega)$ has a spike at $t = T(\omega) = \omega$, so $Y_{T(\omega)}(\omega) = 1 \neq 0 = X_{T(\omega)}(\omega)$. The event $\lbrace Y_\cdot = X_\cdot \text{ on } I \rbrace$ is therefore empty:
  
  $$
  \mathbb{P}\bigl(X_t = Y_t \text{ for all } t \in I\bigr) \;=\; 0.
  $$

<figure>
  <img src="{{ 'assets/images/notes/sdes_diffusion_models/version_vs_indistinguishability.png' | relative_url }}" alt="Version vs. indistinguishability: P(X_t = Y_t) = 1 for every t, but P(X_. = Y_.) = 0" loading="lazy">
</figure>

</div>

<!-- Even when a process has finite-dimensional distributions fulfilling (W1)–(W3), the path property (W4) cannot be inferred on an uncountable index $I$, because uncountable unions of $\mathbb{P}$-null sets need no longer be $\mathbb{P}$-null (example above shows that version is not always indistinguishable). Nevertheless, we **can** define a modification preserving the finite-dimensional distributions (cf. Kolmogorov's extension theorem). -->

Even when a process has finite-dimensional distributions fulfilling (W1)–(W3), the path property (W4) cannot be inferred on an uncountable index $I$, because **uncountable** unions of $\mathbb{P}$-null sets need no longer be $\mathbb{P}$-null. Countable unions of null sets *are* null — that is precisely $\sigma$-additivity (the third Kolmogorov axiom) — but $\sigma$-additivity is by design restricted to countable families and does not extend to uncountable ones. The example above is the canonical illustration: each slice $\lbrace \omega : Y_t(\omega) \neq X_t(\omega) \rbrace = \lbrace t \rbrace$ is null, yet the union over $t \in I$ is the whole diagonal, of full measure.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Kolmogorov's extension theorem and the limits of finite-dimensional data)</span></p>

The same uncountability obstruction explains *why* (W4) is invisible to Kolmogorov's extension theorem. The theorem provides existence of a process from a consistent family of finite-dimensional distributions, realised on the **canonical path space** $\Omega := \mathbb{R}^I$, with $X_t(\omega) := \omega(t)$ and $\mathcal{A}$ the σ-algebra generated by the cylinder sets

$$
\bigl\lbrace \omega \in \mathbb{R}^I : (\omega(t_1), \dots, \omega(t_n)) \in B \bigr\rbrace, \qquad B \in \mathcal{B}(\mathbb{R}^n).
$$

Crucially, every $A \in \mathcal{A}$ depends on **only countably many coordinates**: for each $A$ there is a countable $S \subseteq I$ such that membership of $\omega$ in $A$ is determined by $\omega\|\_S$ alone. Continuity, by contrast, is a genuinely uncountable condition, so the event

$$
\bigl\lbrace \omega \in \mathbb{R}^I : t \mapsto \omega(t) \text{ is continuous on } I \bigr\rbrace
$$

is **not** in $\mathcal{A}$ when $I$ is uncountable. The extension theorem can therefore deliver (W1)–(W3) on $\mathbb{R}^I$ but cannot even assign a probability to (W4), let alone guarantee it. To upgrade the process to one with continuous paths one needs a *modification* — produced here by Kolmogorov's continuity theorem (Theorem 0.8 below).

Nevertheless, we **can** define a modification preserving the finite-dimensional distributions, which is precisely what the strategy below executes.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why (W4) is non-trivial given (W1)–(W3))</span></p>

At first glance one might hope that, since (W1)–(W3) already specify the entire Gaussian-process structure, continuity of paths should follow for free — perhaps as a limiting consequence of the Gaussian increments shrinking with $t - s$. This is **not** the case, and the reason is structural rather than technical.

**(W1)–(W3) only constrain finite-dimensional marginals.** Each of the three conditions is a statement about the joint law of finitely many random variables $W_{t_1}, \dots, W_{t_n}$ — a starting value, the law of an increment $W_t - W_s$, the independence of disjoint increments. The Gaussian-process viewpoint (Definition 0.1) is, by design, finite-dimensional: a process is specified by the family of joint laws on every finite tuple of times.

**(W4) is a path-level condition that consults uncountably many times simultaneously.** Continuity at $t_0$ requires $\|W_t - W_{t_0}\| < \varepsilon$ for *all* $t$ in some neighbourhood of $t_0$, and continuity on $I$ quantifies this over every $t_0 \in I$. There is no way to rephrase "$t \mapsto W_t(\omega)$ is continuous" as a condition on countably many coordinates — between any two times $t_i, t_j$ on a finite grid lies a continuum of unconsulted instants whose joint behaviour is left free by (W1)–(W3).

**The gap is exactly countable vs. uncountable.** Countable families of finite-dimensional constraints can be combined via σ-additivity (the third Kolmogorov axiom), so any property determined by countably many coordinates is reachable from finite-dimensional data. Uncountable families are not, and the $\sigma$-additivity argument that secures the countable case provably fails for uncountable unions (cf. the version-vs-indistinguishability example: each slice $\lbrace t \rbrace$ is null in $\Omega$, yet the union over $t \in I$ has full measure).

**Consequence: finite-dimensional distributions do not determine path behaviour.** Two processes with identical marginals — hence equally good as "centred Gaussian processes with covariance $\min(s,t)$" — can have radically different paths. The canonical illustration is

$$
\widetilde{W}_t := W_t + \mathbf{1}_{\lbrace t = T \rbrace}, \qquad T \sim \mathrm{Unif}([0,T]) \text{ indep.\ of } W,
$$

which satisfies (W1)–(W3) since $\mathbb{P}(t = T) = 0$ at every fixed $t$, yet whose every sample path has a discontinuity at the random instant $T(\omega)$. So (W1)–(W3) cannot single out the continuous representative.

**The same obstruction at the σ-algebra level.** The existence step (Lemma 0.2) realises the process on the canonical path space $\mathbb{R}^I$ with the cylinder σ-algebra $\mathcal{A}$. Every $A \in \mathcal{A}$ depends on only countably many coordinates, so the event

$$
\bigl\lbrace \omega \in \mathbb{R}^I : t \mapsto \omega(t) \text{ is continuous on } I \bigr\rbrace
$$

is **not** in $\mathcal{A}$ when $I$ is uncountable. Kolmogorov's extension theorem cannot even *assign a probability* to (W4), let alone make it $1$.

**Upshot.** (W4) is a genuinely additional requirement — not a derivable corollary of the Gaussian-process specification. Securing it requires a separate ingredient: a quantitative regularity input on increments (the moment bound (0.1)) feeding into Kolmogorov's continuity theorem, which produces a *modification* of the extension-theorem process whose paths are continuous. The two-stage strategy below is therefore forced by the structure of the problem, not a matter of pedagogical taste.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">Strategy for creating real-valued (standard) Brownian motion</span></p>

**Principle strategy for proving existence** of a real-valued (standard) Brownian motion:

1. Define a stochastic process $\widetilde{W}$ with properties (W1), (W2), (W3). (important: we say nothing about measure)
2. Define a modification $W$ of $\widetilde{W}$ that additionally fulfils (W4).

&rarr; $W$ satisfies (W1)–(W4), i.e. $W$ is a real-valued (standard) Brownian motion.

The main tool behind the modification argument is **Kolmogorov's continuity theorem**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">0.7 (Completion)</span></p>

For any $A \subseteq \Omega$ such that there exists $A' \in \mathcal{A}$ with $A' \supseteq A$ and $\mathbb{P}(A') = 0$, we set

$$
\mathbb{P}(A) := 0 \quad \text{and} \quad \mathbb{P}(A^c) := 1.
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">0.8 (Kolmogorov's Continuity Theorem)</span></p>

Let $\widetilde{X} := (\widetilde{X}_t)\_{t \in [0,T]}$ be a real-valued stochastic process on $(\Omega, \mathcal{A}, \mathbb{P})$ such that there exist $\alpha > 0$, $\beta > 1$, and $C > 0$ with

$$
\mathbb{E}\!\left[\,\lvert \widetilde{X}_t - \widetilde{X}_s \rvert^{\alpha}\,\right] \le C \cdot \lvert t - s \rvert^{\beta} \quad \text{for all } s, t \in [0, T]. \tag{0.1}
$$

Then there exists a modification $X := (X_t)\_{t \in [0,T]}$ of $\widetilde{X}$ that has $\gamma$-Hölder-continuous paths for all $\gamma \in \bigl(0, \tfrac{\beta-1}{\alpha}\bigr)$, with a random Hölder constant $K_\gamma : \Omega \to (0, \infty)$:

$$
\lvert X_t(\omega) - X_s(\omega) \rvert \le K_\gamma(\omega) \cdot \lvert t - s \rvert^{\gamma} \quad \text{for all } \omega \in \Omega.
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Thematic Classification</span><span class="math-callout__name">(Kolmogorov's Continuity Theorem 0.8)</span></p>

* **Historical background.** Discovered by Andrey N. Kolmogorov (1940) within the emerging measure-theoretic foundations of probability; addressed the gap between probabilistic laws and pathwise regularity of sample paths; enabled rigorous analysis of continuous-time processes such as Brownian motion.
* **Mathematical importance.** Converts moment conditions into pathwise regularity; guarantees existence of Hölder-continuous modifications; a standard tool in stochastic analysis.
* **Applications.** Brownian motion and Gaussian processes; existence / uniqueness theory for SDEs; regularity theory for SPDEs.

</div>

"Gluing together" compact subintervals yields the version on $[0, \infty)$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">0.9 (Continuity on $[0, \infty)$)</span></p>

Let $\widetilde{X} := (\widetilde{X}_t)\_{t \ge 0}$ be a real-valued stochastic process on $(\Omega, \mathcal{A}, \mathbb{P})$ satisfying condition (0.1) for all $s, t \ge 0$ with some $\alpha > 0$, $\beta > 1$, $C > 0$. Then there exists a continuous modification $X := (X_t)\_{t \ge 0}$ of $\widetilde{X}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">0.10 (Existence of Brownian Motion)</span></p>

There exists a Brownian motion: one can find a probability space and a stochastic process $W := (W_t)_{t \in I}$ on it satisfying all conditions (W1)–(W4) of Definition 0.3.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* By Lemma 0.2 and Proposition 0.5 there is $\widetilde{W} = (\widetilde{W}_t)\_{t \in I}$ with properties (W1), (W2), (W3). Since $\widetilde{W}\_t - \widetilde{W}\_s \sim \mathcal{N}(0, t-s)$ by (W3), for any $s < t$

$$
\mathbb{E}\!\left[\,\lvert \widetilde{W}_t - \widetilde{W}_s \rvert^4\,\right] = 3 \cdot (t - s)^2,
$$

so condition (0.1) holds with $\alpha := 4$, $\beta := 2 > 1$, $C := 3$. Kolmogorov's continuity theorem 0.8 and Corollary 0.9 yield a modification $W$ of $\widetilde{W}$ with continuous paths that inherits (W1), (W2), (W3); hence $W$ satisfies (W1)–(W4). $\square$

</details>
</div>

As Theorem 0.8 hints, Brownian motion enjoys many striking path properties.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">0.11 (Path Properties)</span></p>

Let $W := (W_t)\_{t \ge 0}$ be a standard Brownian motion on $(\Omega, \mathcal{A}, \mathbb{P})$. Define $l, u : [0,1] \to \mathbb{R}$ by $l(t) := t$ and $u(t) := 2t$. Then $W$ has the following path properties:

(i) On every interval $[a, b] \subseteq [0, \infty)$ with $a < b$, the paths $t \mapsto W_t$ are $\mathbb{P}$-a.s. $\gamma$-Hölder continuous with exponent $\gamma \in \bigl(0, \tfrac{1}{2}\bigr)$.

(ii) On every interval $[a, b] \subseteq [0, \infty)$ with $a < b$, the paths are $\mathbb{P}$-a.s. *not* Lipschitz continuous (with any constant $L > 0$); in particular, they are $\mathbb{P}$-a.s. nowhere differentiable.

(iii) On every interval $[a, b] \subseteq [0, \infty)$ with $a < b$, the paths are $\mathbb{P}$-a.s. *not* monotonically increasing.

(iv) The paths $t \mapsto W_t$ on $[0,1]$ are $\mathbb{P}$-a.s. not surrounded by $l$ and $u$, i.e. $\mathbb{P}$-a.s.

$$
l(t) \le W_t \le u(t) \quad \text{for all } t \in [0,1]
$$

fails.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* See Exercise 1.4.

</details>
</div>

<div style="margin: 1.2rem 0 1.5rem;">

<p style="margin: 0 0 0.4rem 0; font-style: italic; color: #5F5E5A;">Interactive: one Brownian sample path on $[0,1]$ with the linear envelopes $l(t)=t$ and $u(t)=2t$ overlaid. Slide the zoom center and shrink the window — the path stays equally jagged at every magnification, visualising the failure of Lipschitz continuity (Prop 0.11 (ii)) and the inability to be trapped between $l$ and $u$ on $[0,1]$ (Prop 0.11 (iv)). The right panel shows the full path with the current zoom window highlighted.</p>

<div style="margin: 1rem 0;">
  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px; flex-wrap: wrap;">
    <label for="path-center" style="font-size: 14px; color: var(--color-text-secondary, #5F5E5A); min-width: 110px;">Zoom center t<sub>0</sub></label>
    <input type="range" min="0" max="1000" value="500" step="1" id="path-center" style="flex: 1; min-width: 200px;" />
    <span id="path-center-val" style="font-size: 14px; font-weight: 500; min-width: 56px; text-align: right;">0.500</span>
  </div>
  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">
    <label for="path-width" style="font-size: 14px; color: var(--color-text-secondary, #5F5E5A); min-width: 110px;">Window width</label>
    <input type="range" min="-3" max="0" value="0" step="0.05" id="path-width" style="flex: 1; min-width: 200px;" />
    <span id="path-width-val" style="font-size: 14px; font-weight: 500; min-width: 56px; text-align: right;">1.000</span>
    <button id="path-resample" style="padding: 6px 14px;">Resample path</button>
  </div>

  <div style="display: flex; flex-wrap: wrap; gap: 18px; margin-bottom: 8px; font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 2px; background: #534AB7;"></span>
      W<sub>t</sub>
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 0; border-top: 1.5px dashed rgba(83,74,183,0.55);"></span>
      l(t)=t and u(t)=2t
    </span>
  </div>

  <div style="display: grid; grid-template-columns: 3fr 1fr; gap: 12px;">
    <div style="position: relative; height: 320px;">
      <canvas id="path-zoom" role="img" aria-label="Brownian motion path zoomed in to a chosen subinterval, with the linear envelopes l(t)=t and u(t)=2t overlaid."></canvas>
    </div>
    <div style="position: relative; height: 320px;">
      <canvas id="path-overview" role="img" aria-label="Full Brownian motion path on the unit interval with the current zoom window highlighted."></canvas>
    </div>
  </div>
</div>

<script>
(function () {
  const N = 4096;
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  const times = new Array(N + 1);
  const path = new Array(N + 1);
  function resample() {
    times[0] = 0; path[0] = 0;
    const dt = 1 / N, sdt = Math.sqrt(dt);
    for (let i = 1; i <= N; i++) {
      times[i] = i * dt;
      path[i] = path[i - 1] + sdt * randn();
    }
  }
  resample();

  const overviewData = [];
  function recomputeOverview() {
    overviewData.length = 0;
    for (let i = 0; i <= N; i++) overviewData.push({ x: times[i], y: path[i] });
  }
  recomputeOverview();

  const centerEl = document.getElementById('path-center');
  const widthEl = document.getElementById('path-width');
  const centerValEl = document.getElementById('path-center-val');
  const widthValEl = document.getElementById('path-width-val');
  const resampleBtn = document.getElementById('path-resample');

  const accent = '#534AB7';
  const envColor = 'rgba(83, 74, 183, 0.55)';
  const highlightColor = 'rgba(83, 74, 183, 0.18)';

  let lo = 0, hi = 1;
  let zoomChart = null;
  let overviewChart = null;

  function buildZoomData() {
    const arr = [];
    for (let i = 0; i <= N; i++) {
      const t = times[i];
      if (t >= lo && t <= hi) arr.push({ x: t, y: path[i] });
    }
    return arr;
  }

  function update() {
    const c = parseInt(centerEl.value, 10) / 1000;
    const w = Math.pow(10, parseFloat(widthEl.value));
    const half = w / 2;
    lo = Math.max(0, c - half);
    hi = Math.min(1, c + half);
    if (hi - lo < 1e-5) hi = Math.min(1, lo + 1e-5);
    centerValEl.textContent = c.toFixed(3);
    widthValEl.textContent = w.toFixed(3);

    const zoomData = buildZoomData();
    const lEnv = [{ x: lo, y: lo }, { x: hi, y: hi }];
    const uEnv = [{ x: lo, y: 2 * lo }, { x: hi, y: 2 * hi }];

    if (!zoomChart) {
      zoomChart = new Chart(document.getElementById('path-zoom'), {
        type: 'line',
        data: { datasets: [
          { label: 'W', data: zoomData, borderColor: accent, borderWidth: 1.2, pointRadius: 0, fill: false, order: 1 },
          { label: 'l', data: lEnv, borderColor: envColor, borderWidth: 1, borderDash: [4, 4], pointRadius: 0, fill: false, order: 3 },
          { label: 'u', data: uEnv, borderColor: envColor, borderWidth: 1, borderDash: [4, 4], pointRadius: 0, fill: false, order: 4 }
        ]},
        options: {
          animation: false, responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: {
            x: { type: 'linear', min: lo, max: hi, title: { display: true, text: 't' }, ticks: { maxTicksLimit: 6 } },
            y: { title: { display: true, text: 'value' } }
          }
        }
      });
    } else {
      zoomChart.data.datasets[0].data = zoomData;
      zoomChart.data.datasets[1].data = lEnv;
      zoomChart.data.datasets[2].data = uEnv;
      zoomChart.options.scales.x.min = lo;
      zoomChart.options.scales.x.max = hi;
      zoomChart.update('none');
    }

    if (!overviewChart) {
      overviewChart = new Chart(document.getElementById('path-overview'), {
        type: 'line',
        data: { datasets: [
          { label: 'W', data: overviewData, borderColor: accent, borderWidth: 0.6, pointRadius: 0, fill: false, order: 1 }
        ]},
        options: {
          animation: false, responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: {
            x: { type: 'linear', min: 0, max: 1, ticks: { maxTicksLimit: 3 } },
            y: { ticks: { maxTicksLimit: 4 } }
          }
        },
        plugins: [{
          id: 'window-highlight',
          beforeDatasetsDraw(chart) {
            const { ctx, chartArea, scales } = chart;
            const xLo = scales.x.getPixelForValue(lo);
            const xHi = scales.x.getPixelForValue(hi);
            ctx.save();
            ctx.fillStyle = highlightColor;
            ctx.fillRect(xLo, chartArea.top, Math.max(2, xHi - xLo), chartArea.bottom - chartArea.top);
            ctx.restore();
          }
        }]
      });
    } else {
      overviewChart.update('none');
    }
  }

  centerEl.addEventListener('input', update);
  widthEl.addEventListener('input', update);
  resampleBtn.addEventListener('click', () => { resample(); recomputeOverview(); update(); });
  update();
})();
</script>

</div>

### 0.4 Donsker's Theorem (Classical Version)

In this subsection we make the random-walk heuristic from §0.2 rigorous. Let $(X_i)\_{i \in \mathbb{N}}$ be i.i.d., real-valued, centred random variables on $(\Omega, \mathcal{A}, \mathbb{P})$ with $\sigma^2 := \mathrm{Var}[X_1] \in (0, \infty)$. For any $n \in \mathbb{N}$ define the **partial sum process** $W^n := (W^n_t)\_{t \in [0, T]}$ via

$$
W^n_t := \frac{1}{\sigma\sqrt{n}} \sum_{i=1}^{\lfloor n t \rfloor} X_i, \quad t \in [0, T].
$$

To simplify the analysis, introduce the *continuous approximation* $\widehat{W^n} := (\widehat{W^n}\_t)\_{t \in [0,T]}$:

$$
\widehat{W^n}_t := \frac{1}{\sigma\sqrt{n}} \sum_{i=1}^{\lfloor n t \rfloor} X_i + \frac{n t - \lfloor n t \rfloor}{\sigma \sqrt{n}} \cdot X_{\lfloor n t \rfloor + 1}, \quad t \in [0,T].
$$

This ensures $\widehat{W^n}$ takes values in the Banach space $\bigl(C([0, T]),\ \lVert \cdot \rVert_\infty\bigr)$ of real-valued continuous functions on the compact interval $[0, T]$ equipped with the supremum norm $\lVert f \rVert_\infty := \sup_{t \in [0,T]} \lvert f(t) \rvert$. Weak convergence in this space is controlled by the following general principle.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">0.12 (Weak Convergence in $C([0,T])$)</span></p>

Let $(Y^n)\_{n \in \mathbb{N}}$ be a sequence of continuous, real-valued processes $Y^n := (Y^n_t)\_{t \in [0,T]}$ on $(\Omega, \mathcal{A}, \mathbb{P})$. Suppose both conditions below hold:

(i) **Fidi convergence.** The finite-dimensional distributions of $Y^n$ converge weakly to those of some continuous process $Y := (Y_t)\_{t \in [0,T]}$, that is, for any $k \in \mathbb{N}$ and $t_1, \dots, t_k \in [0, T]$

$$
(Y^n_{t_1}, \dots, Y^n_{t_k})^\top \xrightarrow{d} (Y_{t_1}, \dots, Y_{t_k})^\top \quad \text{as } n \to \infty.
$$

(ii) **Kolmogorov's criterion.** There exist $\alpha > 0$, $\beta > 1$, and $C > 0$ such that

$$
\mathbb{E}\!\left[\,\lvert Y^n_s - Y^n_t \rvert^\alpha \,\right] \le C \cdot \lvert s - t \rvert^\beta \quad \text{for all } s, t \in [0, T].
$$

Then $(Y^n)\_{n \in \mathbb{N}}$ converges weakly in $\bigl(C([0,T]),\ \lVert \cdot \rVert_\infty\bigr)$ to $Y$.

</div>

This leads to the central result, often called **Donsker's invariance principle**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">0.13 (Donsker's Theorem in $C([0,T])$)</span></p>

In the above setup,

$$
\widehat{W^n} \xrightarrow{d} W \quad \text{in } \bigl(C([0,T]),\ \lVert \cdot \rVert_\infty\bigr),\ \text{as } n \to \infty,
$$

where $W$ is a one-dimensional standard Brownian motion on $(\Omega, \mathcal{A}, \mathbb{P})$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Thematic Classification</span><span class="math-callout__name">(Donsker's Theorem 0.13)</span></p>

* **Historical context.** First proved by Monroe D. Donsker (1951); deeply influenced by the work of Khinchin, Lévy, and Kolmogorov; marked a turning point in the understanding of pathwise convergence.
* **Mathematical importance.** Cornerstone of modern probability theory, bridging discrete and continuous stochastic processes; shows that rescaled random walks converge to Brownian motion; generalises the classical central limit theorem to the functional level.
* **Applications.** Functional limit theorems; empirical process theory; mathematical statistics; financial mathematics.

</div>

### 0.5 Markov Properties of Brownian Motion

For the rest of this section, let $\mathbb{F} := (\mathcal{F}\_t)\_{t \in I}$ be a filtration in $\mathcal{A}$ and consider a one-dimensional (standard) Brownian motion $W := (W_t)\_{t \in I}$ on $(\Omega, \mathcal{A}, \mathbb{P})$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">0.14 ($\mathbb{F}$-Brownian Motion)</span></p>

The stochastic process $(W_t, \mathcal{F}\_t)\_{t \in I}$ (or simply $W$, for short) is called a one-dimensional **(standard) $\mathbb{F}$-Brownian motion** (equivalently: standard Brownian motion with respect to $\mathbb{F}$) iff, additionally:

* $W_t$ is $\mathcal{F}\_t$-measurable for every $t \in I$;
* the process $(W_{t+s} - W_t)\_{s \in I}$ is independent of $\mathcal{F}\_t$.
  * future increment is independent of the past.
  * it says the future Brownian noise is independent of **all information available up to time $t$**.

</div>

The most common choice of $\mathbb{F}$ is the **canonical / natural filtration** induced by $W$:

$$
\mathbb{F}^W := (\mathcal{F}^W_t)_{t \in I}, \qquad \mathcal{F}^W_t := \sigma\bigl(\,W_s : s \in [0, t]\,\bigr).
$$

For any fixed $a \in \mathbb{R}$, define the probability measure on Borel sets $B \subseteq C(I)$

$$
\mathbf{P}^a_W : \mathcal{B}\bigl(C(I)\bigr) \to [0,1], \qquad \mathbf{P}^a_W(B) := \mathbb{P}(W + a \in B).
$$

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">($\mathbf{P}^a_W$)</span></p>

$\mathbf{P}^a_W$ is the law of a Brownian path shifted vertically by $a$: under $\mathbf{P}^a_W$, the process $W$ has the same distribution as the shifted process $W + a$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">0.15 (Markov Property)</span></p>

Let $W$ be a one-dimensional (standard) $\mathbb{F}$-Brownian motion. Then, for any bounded measurable functional $\Phi : C(I) \to \mathbb{R}$,

$$
\mathbb{E}\!\left[\, \Phi\bigl(W_{s+\cdot}\bigr) \,\big|\, \mathcal{F}_s \,\right]
\;=\;
\int_{C(I)} \Phi(w)\, \mathrm{d}\mathbf{P}^{W_s}_W(w)
\qquad \mathbb{P}\text{-a.s., for all } s \in I.
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Markov Property)</span></p>

Conditionally on $\mathcal{F}\_s$, the future path of $W$ behaves in mean like a *fresh* Brownian motion starting at $W_s$: the conditional expectation of a functional $\Phi$ is the integral of $\Phi$ against the shifted law $\mathbf{P}^{W_s}\_W$ — i.e. the integral over all future paths $w$ shifted by the current value $W_s$.

</div>

<div class="accordion" markdown="1">

<details markdown="1">
<summary>Proof sketch of Proposition 0.15</summary>

Let $k \in \mathbb{N}$, $0 \le t_1 < \dots < t_k$ in $I$, and $\lambda_1, \dots, \lambda_k \in \mathbb{R}$. Test the identity first on exponential functionals

$$
\Phi(w) := \exp\!\left(\sum_{i=1}^k \lambda_i \cdot w(t_i)\right), \qquad w \in C(I).
$$

For $a \in \mathbb{R}$, a measure change gives

$$
\int_{C(I)} \exp\!\left(\sum_{i=1}^k \lambda_i w(t_i)\right) \mathrm{d}\mathbf{P}^a_W(w)
= \exp\!\left(a \sum_{i=1}^k \lambda_i\right) \int_{C(I)} \exp\!\left(\sum_{i=1}^k \lambda_i w(t_i)\right) \mathrm{d}\mathbf{P}^0_W(w).
$$

On the other hand, writing $\overline{W}\_t := W_{s+t} - W_s$ — which is independent of $\mathcal{F}\_s$, while $W_s$ is $\mathcal{F}\_s$-measurable — properties of conditional expectation give

$$\mathbb{E}\!\left[\, \exp\!\left(\sum_i \lambda_i W_{s+t_i}\right) \,\Big|\, \mathcal{F}_s \,\right]$$

$$= \exp\!\left(W_s \sum_i \lambda_i \right) \cdot \mathbb{E}\!\left[\exp\!\left(\sum_i \lambda_i \overline{W}_{t_i}\right)\right]$$

$$= \exp\!\left(W_s \sum_i \lambda_i \right) \cdot \mathbb{E}\!\left[\exp\!\left(\sum_i \lambda_i W_{t_i}\right)\right]$$

Setting $a := W_s$ in the measure-change relation above matches both expressions:

$$
\mathbb{E}\!\left[\,\exp\!\left(\sum_i \lambda_i W_{s+t_i}\right) \,\Big|\, \mathcal{F}_s \,\right] = \int_{C(I)} \exp\!\left(\sum_i \lambda_i w(t_i)\right) \mathrm{d}\mathbf{P}^{W_s}_W(w).
$$

Finally, the Stone–Weierstraß theorem, together with the fact that finite-dimensional distributions uniquely determine the law of a stochastic process, extends the identity to every bounded measurable $\Phi : C(I) \to \mathbb{R}$. $\square$

</details>

</div>

Before strengthening Proposition 0.15 to a version allowing random times, we need two refined filtrations that avoid measurability pathologies.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">0.16 (Right-Continuous and Stopped $\sigma$-Algebras)</span></p>

(a) Define the **right-continuous filtration** $\mathbb{F}^{+} := (\mathcal{F}_{t+})\_{t \in I}$ by

$$
\mathcal{F}_{t+} := \bigcap_{s > t} \mathcal{F}_s, \qquad t \in I.
$$

Let $\tau : \Omega \to \mathbb{R} \cup \lbrace \infty \rbrace$ be a stopping time with respect to $\mathbb{F}$.

(b) The **stopped $\sigma$-algebra** (or **$\sigma$-algebra of the $\tau$-past**) is

$$
\mathcal{F}_\tau := \bigl\lbrace A \in \mathcal{A} : A \cap \lbrace \tau \le t \rbrace \in \mathcal{F}_t\ \text{ for all } t \in I \bigr\rbrace.
$$

(c) Its right-continuous analogue is

$$
\mathcal{F}_\tau^{+} := \bigl\lbrace A \in \mathcal{A} : A \cap \lbrace \tau \le t \rbrace \in \mathcal{F}_{t+}\ \text{ for all } t \in I \bigr\rbrace.
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">0.17 (Elementary Properties)</span></p>

It holds that:

(i) $\mathcal{F}\_t \subseteq \mathcal{F}\_{t+}$ for all $t \in I$.

Let $\tau : \Omega \to \mathbb{R} \cup \lbrace \infty \rbrace$ be a stopping time with respect to $\mathbb{F}$. Then:

(ii) $\mathcal{F}\_\tau \subseteq \mathcal{F}\_\tau^+$;

(iii) $\tau$ is also a stopping time with respect to $\mathbb{F}^+$;

(iv) $\mathcal{F}\_\tau^{+} = \bigl\lbrace A \in \mathcal{A} : A \cap \lbrace \tau < t \rbrace \in \mathcal{F}\_t\ \text{ for all } t \in I \bigr\rbrace$.

</div>

Equipped with Lemma 0.17, Proposition 0.15 can be extended to *random* times.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">0.18 (Strong Markov Property)</span></p>

Let $\tau : \Omega \to \mathbb{R} \cup \lbrace \infty \rbrace$ be a $\mathbb{P}$-a.s. finite $\mathbb{F}^+$-stopping time. Then the stochastic process $\overline{W} := (\overline{W}\_t)\_{t \in I}$ on $(\Omega, \mathcal{A}, \mathbb{P})$ defined by

$$
\overline{W}_t := W_{\tau + t} - W_\tau, \qquad t \in I,
$$

is a one-dimensional (standard) Brownian motion, independent of $\mathcal{F}\_\tau^{+}$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Strong Markov Property)</span></p>

The strong Markov property allows one to restart a Brownian motion at *random* times as though beginning anew — with complete independence from the entire past up to (and, via $\mathcal{F}\_\tau^{+}$, just after) $\tau$.

</div>

The proof is essentially a localisation at dyadic stopping times combined with a density-based approximation argument, analogous to the proof of Proposition 0.15.

### 0.6 Distributional Properties of Brownian Motion

Complementing the path properties from Proposition 0.11, we record distributional consequences of Theorem 0.18.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">0.19 (Reflection Principle)</span></p>

Let $\tau : \Omega \to \mathbb{R} \cup \lbrace \infty \rbrace$ be an $\mathbb{F}$-stopping time. Then $W$ has the same distribution as the **reflected process** $\widetilde{W} := (\widetilde{W}\_t)\_{t \in I}$ defined by

$$
\widetilde{W}_t := W_{t \wedge \tau} - \bigl(W_t - W_{t \wedge \tau}\bigr), \qquad t \in I.
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Reflection Principle)</span></p>

If a Brownian path crosses a level $x \in \mathbb{R}$, the "reflected" path — obtained by mirroring the trajectory after the crossing time — has the same law as the original process.

</div>

<div class="accordion" markdown="1">

<details markdown="1">
<summary>Proof sketch of Proposition 0.19</summary>

We prove it for $\mathbb{P}$-a.s. finite stopping times $\tau$ (the general case follows by localisation). Verify (W1)–(W4) for $\widetilde{W}$:

* **(W1) Starting in 0.** Clearly $\widetilde{W}\_0 = 0$, $\mathbb{P}$-a.s.
* **(W4) Continuous paths.** $W$ has $\mathbb{P}$-a.s. continuous paths, so does $\widetilde{W}$; in particular, at $t = \tau$, $\lim_{t \uparrow \tau} \widetilde{W}\_t = W_\tau = \lim_{t \downarrow \tau} \widetilde{W}\_t$.

For $t < \tau$, $\widetilde{W}\_t = W_t$, so (W2), (W3) hold trivially. Consider the "post-$\tau$" process $\overline{W} = (\overline{W}\_t)\_{t \in I}$, $\overline{W}\_t := W_{\tau + t} - W_\tau$; by Theorem 0.18, $\overline{W}$ is a standard Brownian motion independent of $\mathcal{F}\_\tau \subseteq \mathcal{F}\_\tau^+$ (Lemma 0.17 (ii)). For $t \ge \tau$,

$$
\widetilde{W}_t = 2 W_\tau - W_t = W_\tau - (W_t - W_\tau) = W_\tau - \overline{W}_{t - \tau},
$$

so $\widetilde{W}\_{\tau + t} = W_\tau - \overline{W}\_t$ for any $t \in I$.

* **(W2) Independent increments.** Increments of $\widetilde{W}$ after $\tau$ only depend on increments of $\overline{W}$, independent of $\mathcal{F}\_\tau$. Hence increments after $\tau$ are independent of the $\tau$-past, and increments among themselves are independent by (W2) of $\overline{W}$.
* **(W3) Normal increments.** Let $s < t$ in $I$. Distinguish the two remaining cases:
  * If $t > s \ge \tau$: by (W3) of $\overline{W}$
  
    $$\widetilde{W}_t - \widetilde{W}_s = -(\overline{W}_{t-\tau} - \overline{W}_{s-\tau}) \sim -\mathcal{N}(0, t - s) \stackrel{d}{=} \mathcal{N}(0, t - s)$$
  
  * If $s < \tau < t$: 
    
    $$\widetilde{W}_t - \widetilde{W}_s = (W_\tau - W_s) - \overline{W}_{t - \tau}$$
    
    The first summand is $\mathcal{F}\_\tau$-measurable, the second is independent of $\mathcal{F}\_\tau$. Both are Gaussian conditionally on $\tau$, and unconditionally,

    $$
    \widetilde{W}_t - \widetilde{W}_s = (W_\tau - W_s) - \overline{W}_{t - \tau} \sim \mathcal{N}(0, t - s). \qquad \square
    $$

</details>

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Summary</span><span class="math-callout__name">(Two Crucial Ideas Behind the Reflection Principle)</span></p>

* **Strong Markov property.** After the stopping time $\tau$, the process $\overline{W}$ behaves like a *fresh* Brownian motion independent of the past.
* **Symmetry.** A Brownian motion and its negative are identically distributed, so flipping the sign after $\tau$ does not change the law of the entire process.

</div>

In the next two corollaries, for $x > 0$ we make use of the $\mathbb{F}^W$-stopping time

$$
\tau_x := \inf\lbrace t \in I : W_t = x \rbrace,
$$

the first time $W$ reaches the value $x$ in $I$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">0.20 (Maximum Process of Brownian Motion)</span></p>

Define the **maximum process** $M := (M_t)\_{t \in I}$ of $W$ by $M_t := \max_{s \in [0, t]} W_s$ for $t \in I$. Then

$$
M_t \stackrel{d}{=} \lvert W_t \rvert, \qquad t \in I.
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* See Exercise 2.3.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">0.21 (Hitting-Time and Maximum Bounds)</span></p>

For any $x > 0$:

(i) $\displaystyle \mathbb{P}(\tau_x \le t) = 2 \cdot \mathbb{P}(W_t \ge x) = \frac{x}{\sqrt{2\pi}} \int_0^t \frac{1}{s^{3/2}} \cdot \exp\left(-\frac{x^2}{2s}\right) \mathrm{d}s$ for all $t \in I$;

(ii) $\displaystyle \mathbb{P}(M_t \ge x) \le \sqrt{\frac{2}{\pi}} \cdot \frac{\sqrt{t}}{x} \cdot \exp\left(-\frac{x^2}{2 t}\right)$ for all $t \in I$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Corollary 0.21 (i))</span></p>

The reflection principle doubles the probability of paths crossing a level $x \in \mathbb{R}$ before time $t \in I$, compared to simply ending above $x$, as any path crossing $x$ can be paired with a mirrored path not crossing it.

</div>

<div style="margin: 1.2rem 0 1.5rem;">

<p style="margin: 0 0 0.4rem 0; font-style: italic; color: #5F5E5A;">Interactive: a single Brownian path $W_t$ on $[0,1]$ together with its reflection $\widetilde W_t = W_{t \wedge \tau_x} - (W_t - W_{t \wedge \tau_x})$ about the level $x$ from the first hitting time $\tau_x$ onward (the two coincide before $\tau_x$). Slide $x$ and resample to verify visually that $\widetilde W$ has the same law as $W$. Below: empirical histogram of $\tau_x$ over $5{,}000$ independent paths against the theoretical density $\frac{x}{\sqrt{2\pi t^3}} \exp\!\left(-\frac{x^2}{2t}\right)$.</p>

<div style="margin: 1rem 0;">
  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">
    <label for="refl-x" style="font-size: 14px; color: var(--color-text-secondary, #5F5E5A); min-width: 80px;">Level x</label>
    <input type="range" min="0.1" max="2.5" value="0.7" step="0.01" id="refl-x" style="flex: 1; min-width: 200px;" />
    <span id="refl-x-val" style="font-size: 14px; font-weight: 500; min-width: 56px; text-align: right;">0.70</span>
    <button id="refl-resample" style="padding: 6px 14px;">Resample</button>
  </div>

  <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 1rem;">
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">Level x</div>
      <div id="refl-stat-x" style="font-size: 20px; font-weight: 500;">0.70</div>
    </div>
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">Hitting time &tau;<sub>x</sub> (this path)</div>
      <div id="refl-stat-tau" style="font-size: 20px; font-weight: 500;">&mdash;</div>
    </div>
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">P(&tau;<sub>x</sub> &le; 1) (theory)</div>
      <div id="refl-stat-prob" style="font-size: 20px; font-weight: 500;">&mdash;</div>
    </div>
  </div>

  <div style="display: flex; flex-wrap: wrap; gap: 18px; margin-bottom: 8px; font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 2px; background: #534AB7;"></span>
      original W<sub>t</sub>
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 2px; background: #D97757;"></span>
      reflected W&#771;<sub>t</sub>
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 0; border-top: 1.5px dashed rgba(83,74,183,0.55);"></span>
      level x
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 0; border-top: 1.5px dashed rgba(217,119,87,0.55);"></span>
      &tau;<sub>x</sub>
    </span>
  </div>

  <div style="position: relative; width: 100%; height: 280px; margin-bottom: 1rem;">
    <canvas id="refl-paths" role="img" aria-label="Brownian path and its reflection about the chosen level x, drawn against time on the unit interval."></canvas>
  </div>

  <p style="margin: 0 0 0.4rem 0; font-style: italic; color: #5F5E5A;">Empirical density of $\tau_x$ over $5{,}000$ independent paths vs. theoretical density (truncated to $\tau_x \le 1$).</p>
  <div style="position: relative; width: 100%; height: 240px;">
    <canvas id="refl-hist" role="img" aria-label="Histogram of simulated hitting times for the chosen level x, overlaid with the theoretical hitting-time density."></canvas>
  </div>
</div>

<script>
(function () {
  const N = 1024;
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function buildBM(steps) {
    const dt = 1 / steps, sdt = Math.sqrt(dt);
    const arr = new Array(steps + 1);
    arr[0] = 0;
    for (let i = 1; i <= steps; i++) arr[i] = arr[i - 1] + sdt * randn();
    return arr;
  }
  function erfc(z) {
    const a = Math.abs(z);
    const t = 1 / (1 + 0.5 * a);
    const ans = t * Math.exp(-a * a - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
      t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
      t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return z >= 0 ? ans : 2 - ans;
  }

  let path = buildBM(N);

  const M = 5000;
  const hits = new Float64Array(M);

  const xEl = document.getElementById('refl-x');
  const xValEl = document.getElementById('refl-x-val');
  const statX = document.getElementById('refl-stat-x');
  const statTau = document.getElementById('refl-stat-tau');
  const statProb = document.getElementById('refl-stat-prob');
  const resampleBtn = document.getElementById('refl-resample');

  const accent = '#534AB7';
  const reflColor = '#D97757';
  const levelColor = 'rgba(83, 74, 183, 0.55)';
  const tauColor = 'rgba(217, 119, 87, 0.55)';

  let pathsChart = null;
  let histChart = null;

  function rebuildHits(x) {
    const dt = 1 / N, sdt = Math.sqrt(dt);
    for (let k = 0; k < M; k++) {
      let w = 0, t = -1;
      for (let i = 1; i <= N; i++) {
        w += sdt * randn();
        if (w >= x) { t = i / N; break; }
      }
      hits[k] = t;
    }
  }

  function findTauIndex(arr, x) {
    for (let i = 0; i < arr.length; i++) if (arr[i] >= x) return i;
    return -1;
  }

  function update() {
    const x = parseFloat(xEl.value);
    xValEl.textContent = x.toFixed(2);
    statX.textContent = x.toFixed(2);

    const idx = findTauIndex(path, x);
    const tau = idx < 0 ? -1 : idx / (path.length - 1);
    statTau.textContent = tau < 0 ? 'not before T = 1' : tau.toFixed(3);

    statProb.textContent = erfc(x / Math.SQRT2).toFixed(3);

    const wPath = new Array(path.length);
    const refl = new Array(path.length);
    const wTau = idx >= 0 ? path[idx] : 0;
    for (let i = 0; i < path.length; i++) {
      const t = i / (path.length - 1);
      wPath[i] = { x: t, y: path[i] };
      refl[i] = { x: t, y: (idx < 0 || i <= idx) ? path[i] : 2 * wTau - path[i] };
    }
    const levelData = [{ x: 0, y: x }, { x: 1, y: x }];
    const yMin = Math.min(-x - 0.5, -2);
    const yMax = Math.max(x + 1, 2.5);
    const tauLine = tau < 0 ? [] : [{ x: tau, y: yMin }, { x: tau, y: yMax }];

    if (!pathsChart) {
      pathsChart = new Chart(document.getElementById('refl-paths'), {
        type: 'line',
        data: { datasets: [
          { label: 'W', data: wPath, borderColor: accent, borderWidth: 1.4, pointRadius: 0, fill: false, order: 1 },
          { label: 'reflected', data: refl, borderColor: reflColor, borderWidth: 1.4, pointRadius: 0, fill: false, order: 2 },
          { label: 'level', data: levelData, borderColor: levelColor, borderWidth: 1, borderDash: [5, 4], pointRadius: 0, fill: false, order: 3 },
          { label: 'tau', data: tauLine, borderColor: tauColor, borderWidth: 1, borderDash: [3, 3], pointRadius: 0, fill: false, order: 4 }
        ]},
        options: {
          animation: false, responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: {
            x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 't' } },
            y: { title: { display: true, text: 'value' } }
          }
        }
      });
    } else {
      pathsChart.data.datasets[0].data = wPath;
      pathsChart.data.datasets[1].data = refl;
      pathsChart.data.datasets[2].data = levelData;
      pathsChart.data.datasets[3].data = tauLine;
      pathsChart.update('none');
    }

    rebuildHits(x);
    const bins = 40;
    const counts = new Array(bins).fill(0);
    let nHit = 0;
    for (let k = 0; k < M; k++) {
      if (hits[k] >= 0 && hits[k] <= 1) {
        nHit++;
        const b = Math.min(bins - 1, Math.floor(hits[k] * bins));
        counts[b]++;
      }
    }
    const binWidth = 1 / bins;
    const denom = Math.max(1, M) * binWidth;
    const histData = [{ x: 0, y: 0 }];
    for (let b = 0; b < bins; b++) {
      const left = b / bins, right = (b + 1) / bins;
      const h = counts[b] / denom;
      histData.push({ x: left, y: h });
      histData.push({ x: right, y: h });
    }
    histData.push({ x: 1, y: 0 });

    const NP = 240;
    const theoryData = [];
    let yMaxTheory = 0;
    for (let i = 1; i <= NP; i++) {
      const t = i / NP;
      const d = (x / Math.sqrt(2 * Math.PI * t * t * t)) * Math.exp(-x * x / (2 * t));
      if (d > yMaxTheory) yMaxTheory = d;
      theoryData.push({ x: t, y: d });
    }
    let histPeak = 0;
    for (let b = 0; b < bins; b++) histPeak = Math.max(histPeak, counts[b] / denom);
    const yMaxAxis = Math.max(yMaxTheory, histPeak) * 1.15 + 1e-9;

    if (!histChart) {
      histChart = new Chart(document.getElementById('refl-hist'), {
        type: 'line',
        data: { datasets: [
          { label: 'empirical', data: histData, borderColor: 'rgba(83,74,183,0.7)', backgroundColor: 'rgba(83,74,183,0.18)', borderWidth: 1, pointRadius: 0, fill: 'origin', tension: 0, order: 2 },
          { label: 'theory', data: theoryData, borderColor: '#D97757', borderWidth: 1.6, pointRadius: 0, fill: false, tension: 0, order: 1 }
        ]},
        options: {
          animation: false, responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: {
            x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 't' } },
            y: { title: { display: true, text: 'density' }, beginAtZero: true, max: yMaxAxis }
          }
        }
      });
    } else {
      histChart.data.datasets[0].data = histData;
      histChart.data.datasets[1].data = theoryData;
      histChart.options.scales.y.max = yMaxAxis;
      histChart.update('none');
    }
  }

  xEl.addEventListener('input', update);
  resampleBtn.addEventListener('click', () => { path = buildBM(N); update(); });
  update();
})();
</script>

</div>

More involved distributional consequences of the strong Markov property are collected in the last proposition of this chapter. For $T > 0$, define

$$
A(T) := \underset{t \in [0,T]}{\operatorname*{argmax}} W_t, \qquad L(T) := \max\lbrace t \in [0, T] : W_t = 0 \rbrace,
$$

the argmax and the last zero of $W$ on $[0, T]$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">0.22 (Lévy's Arcsine Laws)</span></p>

With the above definitions, the following hold:

(i) **Lévy's second arcsine law.**

$$
\mathbb{P}\bigl(L(T) \le t\bigr) = \frac{2}{\pi} \cdot \arcsin\!\left(\sqrt{\frac{t}{T}}\right), \qquad t \in [0, T].
$$

(ii) **Lévy's third arcsine law.**

$$
\mathbb{P}\bigl(A(1) \le t\bigr) = \frac{2}{\pi} \cdot \arcsin\!\left(\sqrt{t}\right), \qquad t \in [0, 1].
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* See Exercise 2.4.

</details>
</div>

<div style="margin: 1.2rem 0 1.5rem;">

<p style="margin: 0 0 0.4rem 0; font-style: italic; color: #5F5E5A;">Interactive: empirical histograms of the last zero $L(T)/T$ and the argmax $A(1)$ over $N$ independent Brownian paths, against the common arcsine density $\dfrac{1}{\pi \sqrt{t(1-t)}}$ (the derivative of the CDF in Prop 0.22). Increase $N$ and resample — the bimodal U-shape that emerges is what makes the arcsine law so counterintuitive: the typical Brownian path on $[0,1]$ is far more likely to take its maximum (and to last touch zero) near the endpoints than near the middle.</p>

<div style="margin: 1rem 0;">
  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">
    <label for="arc-n" style="font-size: 14px; color: var(--color-text-secondary, #5F5E5A); min-width: 110px;">Sample size N</label>
    <input type="range" min="0" max="4" value="2" step="1" id="arc-n" style="flex: 1; min-width: 200px;" />
    <span id="arc-n-val" style="font-size: 14px; font-weight: 500; min-width: 56px; text-align: right;">2,000</span>
    <button id="arc-resample" style="padding: 6px 14px;">Resample</button>
  </div>

  <div style="display: flex; flex-wrap: wrap; gap: 18px; margin-bottom: 8px; font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 14px; height: 8px; background: rgba(83,74,183,0.18); border: 1px solid rgba(83,74,183,0.7);"></span>
      empirical histogram
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 2px; background: #D97757;"></span>
      arcsine density 1/(&pi;&radic;(t(1&minus;t)))
    </span>
  </div>

  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
    <div>
      <div style="font-size: 13px; font-weight: 500; margin-bottom: 4px; color: #2c3e94;">Last zero L(T)/T on [0,1]</div>
      <div style="position: relative; height: 260px;">
        <canvas id="arc-last-zero" role="img" aria-label="Histogram of the rescaled last zero of Brownian motion on the unit interval, against the arcsine density."></canvas>
      </div>
    </div>
    <div>
      <div style="font-size: 13px; font-weight: 500; margin-bottom: 4px; color: #2c3e94;">Argmax A(1) on [0,1]</div>
      <div style="position: relative; height: 260px;">
        <canvas id="arc-argmax" role="img" aria-label="Histogram of the location of the maximum of Brownian motion on the unit interval, against the arcsine density."></canvas>
      </div>
    </div>
  </div>
</div>

<script>
(function () {
  const Nsteps = 1024;
  const NSchedule = [200, 500, 2000, 5000, 10000];

  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const nEl = document.getElementById('arc-n');
  const nValEl = document.getElementById('arc-n-val');
  const resampleBtn = document.getElementById('arc-resample');

  let lastZeroChart = null;
  let argmaxChart = null;

  function simulate(N) {
    const dt = 1 / Nsteps, sdt = Math.sqrt(dt);
    const lastZeros = new Float64Array(N);
    const argmaxes = new Float64Array(N);
    for (let k = 0; k < N; k++) {
      let w = 0, prev = 0;
      let maxVal = 0, maxIdx = 0;
      let lastZeroIdx = 0;
      for (let i = 1; i <= Nsteps; i++) {
        prev = w;
        w += sdt * randn();
        if ((prev <= 0 && w >= 0) || (prev >= 0 && w <= 0)) {
          // approximate zero crossing time linearly
          const denom = (w - prev);
          const frac = denom !== 0 ? -prev / denom : 0;
          lastZeroIdx = (i - 1 + frac);
        }
        if (w > maxVal) { maxVal = w; maxIdx = i; }
      }
      lastZeros[k] = lastZeroIdx / Nsteps;
      argmaxes[k] = maxIdx / Nsteps;
    }
    return { lastZeros, argmaxes };
  }

  function histogram(samples, bins) {
    const counts = new Array(bins).fill(0);
    for (let i = 0; i < samples.length; i++) {
      const v = samples[i];
      if (v >= 0 && v <= 1) {
        const b = Math.min(bins - 1, Math.floor(v * bins));
        counts[b]++;
      }
    }
    const binWidth = 1 / bins;
    const denom = samples.length * binWidth;
    const data = [{ x: 0, y: 0 }];
    let peak = 0;
    for (let b = 0; b < bins; b++) {
      const left = b / bins, right = (b + 1) / bins;
      const h = counts[b] / denom;
      data.push({ x: left, y: h });
      data.push({ x: right, y: h });
      if (h > peak) peak = h;
    }
    data.push({ x: 1, y: 0 });
    return { data, peak };
  }

  const NP = 240;
  const arcsineData = [];
  for (let i = 1; i < NP; i++) {
    const t = i / NP;
    arcsineData.push({ x: t, y: 1 / (Math.PI * Math.sqrt(t * (1 - t))) });
  }

  function makeChart(canvasId, histData, peak) {
    return new Chart(document.getElementById(canvasId), {
      type: 'line',
      data: { datasets: [
        { label: 'empirical', data: histData, borderColor: 'rgba(83,74,183,0.7)', backgroundColor: 'rgba(83,74,183,0.18)', borderWidth: 1, pointRadius: 0, fill: 'origin', tension: 0, order: 2 },
        { label: 'arcsine', data: arcsineData, borderColor: '#D97757', borderWidth: 1.6, pointRadius: 0, fill: false, tension: 0, order: 1 }
      ]},
      options: {
        animation: false, responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: {
          x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 't' } },
          y: { title: { display: true, text: 'density' }, beginAtZero: true, max: Math.max(peak, 3.2) }
        }
      }
    });
  }

  function update() {
    const idx = parseInt(nEl.value, 10);
    const N = NSchedule[idx];
    nValEl.textContent = N.toLocaleString();

    const { lastZeros, argmaxes } = simulate(N);
    const bins = 40;
    const lz = histogram(lastZeros, bins);
    const am = histogram(argmaxes, bins);
    const peak = Math.max(lz.peak, am.peak, 3.2);

    if (!lastZeroChart) {
      lastZeroChart = makeChart('arc-last-zero', lz.data, peak);
    } else {
      lastZeroChart.data.datasets[0].data = lz.data;
      lastZeroChart.options.scales.y.max = peak;
      lastZeroChart.update('none');
    }
    if (!argmaxChart) {
      argmaxChart = makeChart('arc-argmax', am.data, peak);
    } else {
      argmaxChart.data.datasets[0].data = am.data;
      argmaxChart.options.scales.y.max = peak;
      argmaxChart.update('none');
    }
  }

  nEl.addEventListener('input', update);
  resampleBtn.addEventListener('click', update);
  update();
})();
</script>

</div>

## 1 Introduction

Before formalising the theory of continuous-time stochastic processes, we briefly outline the two central concepts upon which this course is built: **stochastic differential equations (SDEs)** and **diffusion models**. The aim is to clarify both the mathematical necessity of SDEs and their application in recent generative modelling.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stochastic Differential Equations)</span></p>

To model dynamical systems subject to continuous random fluctuations, one naturally considers differential equations driven by a noise process, where the canonical mathematical object for continuous noise is **Brownian motion**. However, the sample paths of Brownian motion are almost surely nowhere differentiable and possess infinite first variation (see Proposition 0.11 (ii) and §1.1), rendering classical Riemann–Stieltjes integration inapplicable. (TODO: why?)

SDEs, formalised via *Itô calculus*, provide the rigorous measure-theoretic framework required to integrate against such processes. By doing so, an SDE defines a flow not merely on a state space, but on the underlying probability space — allowing for rigorous analysis of stochastic dynamics in continuous time.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Generative Modeling)</span></p>

In the context of machine learning, *generative modelling* poses the problem:

> *Given an empirical distribution of samples, how can we construct a mechanism to draw novel samples from the underlying, unknown probability measure $p_{\mathrm{data}}$ on $\mathbb{R}^d$?*

Diffusion models solve this by constructing a continuous-time *stochastic transport map* via a two-step procedure:

1. A **forward SDE** acts as a smoothing operator, continuously transporting the intractable measure $p_{\mathrm{data}}$ into a mathematically tractable reference measure — typically a standard Gaussian $\mathcal{N}(0, I)$.
2. Exploiting the measure-theoretic properties of Markov process time-reversal (Kolmogorov 1936, Nelson 1967), one derives a **reverse SDE**. If the *score function* (the logarithmic gradient of the marginal density) can be estimated, this reverse SDE transports the reference measure back to $p_{\mathrm{data}}$.

Thus diffusion models provide a constructive, mathematically rigorous algorithm for generative synthesis. The aim of this course is therefore twofold: (i) build the stochastic analysis required to understand SDEs, and (ii) apply these tools to analyse the mechanics and convergence of diffusion-based generative models.

</div>

To build a rigorous theory of continuous stochastic dynamics, we must construct a framework capable of handling random motion that is nowhere differentiable. We base the exposition on the classical works of *Øksendal* (2013), *Revuz & Yor* (1999), and *Karatzas & Shreve* (1991).

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Foundation and a Problem</span><span class="math-callout__name">(BM is a cornerstone of continuous-time stochastic processes, but classical RS integration fails there)</span></p>

The cornerstone of continuous-time stochastic processes is the Brownian motion from Chapter 0. It is the canonical model for purely random, unpredictable continuous motion. As observed earlier (Paley–Wiener–Zygmund 1933), the paths of Brownian motion are $\mathbb{P}$-almost surely nowhere differentiable, with *infinite first variation* — hence classical Riemann–Stieltjes integration fails entirely for integrals of the form $\int f(t) \mathrm{d} W_t$. (TODO: why do we integrate over $W_t$?)

</div>

<div style="margin: 1.2rem 0 1.5rem;">

<p style="margin: 0 0 0.4rem 0; font-style: italic; color: #5F5E5A;">Interactive: refine the partition $0 = t_0 < t_1 < \dots < t_n = 1$ and watch the partition sums for a Brownian sample path $W_t$ (top) versus the smooth function $f(t) = \sin(2\pi t)$ (bottom). As $n \to \infty$ the <em>quadratic variation</em> of $W$ converges to $1$ while its <em>first variation</em> blows up to $\infty$; for $f$ both stay bounded, with quadratic variation collapsing to $0$. This is the precise reason classical Riemann–Stieltjes integration against $W$ fails — and why $(\mathrm{d}W_t)^2 = \mathrm{d}t$ in Itô calculus.</p>

<div style="margin: 1rem 0;">
  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">
    <label for="qv-n" style="font-size: 14px; color: var(--color-text-secondary, #5F5E5A); min-width: 110px;">Partition size n</label>
    <input type="range" min="0" max="10" value="4" step="1" id="qv-n" style="flex: 1; min-width: 200px;" />
    <span id="qv-n-val" style="font-size: 14px; font-weight: 500; min-width: 56px; text-align: right;">64</span>
    <button id="qv-resample" style="padding: 6px 14px;">Resample W</button>
  </div>

  <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 1rem;">
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">&Sigma;(&Delta;W)<sup>2</sup> &nbsp;&rarr; T = 1</div>
      <div id="qv-stat-w2" style="font-size: 20px; font-weight: 500; color: #534AB7;">&mdash;</div>
    </div>
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">&Sigma;|&Delta;W| &nbsp;&rarr; &infin;</div>
      <div id="qv-stat-w1" style="font-size: 20px; font-weight: 500; color: #534AB7;">&mdash;</div>
    </div>
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">&Sigma;(&Delta;sin)<sup>2</sup> &nbsp;&rarr; 0</div>
      <div id="qv-stat-s2" style="font-size: 20px; font-weight: 500; color: #D97757;">&mdash;</div>
    </div>
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">&Sigma;|&Delta;sin| &nbsp;&rarr; 4</div>
      <div id="qv-stat-s1" style="font-size: 20px; font-weight: 500; color: #D97757;">&mdash;</div>
    </div>
  </div>

  <div style="display: flex; flex-wrap: wrap; gap: 18px; margin-bottom: 8px; font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 2px; background: #534AB7;"></span>
      Brownian path W<sub>t</sub>
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 2px; background: #D97757;"></span>
      sin(2&pi;t)
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 5px; height: 5px; background: #2c3e94; border-radius: 50%;"></span>
      partition points
    </span>
  </div>

  <div style="position: relative; width: 100%; height: 220px; margin-bottom: 8px;">
    <canvas id="qv-w" role="img" aria-label="Brownian path on the unit interval with partition nodes marked, illustrating the divergence of first variation."></canvas>
  </div>
  <div style="position: relative; width: 100%; height: 220px;">
    <canvas id="qv-s" role="img" aria-label="The smooth function sin(2 pi t) on the unit interval with the same partition nodes marked, illustrating bounded first variation and vanishing quadratic variation."></canvas>
  </div>
</div>

<script>
(function () {
  const Nfine = 4096;
  const partitions = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const wPath = new Array(Nfine + 1);
  function resampleW() {
    wPath[0] = 0;
    const sdt = Math.sqrt(1 / Nfine);
    for (let i = 1; i <= Nfine; i++) wPath[i] = wPath[i - 1] + sdt * randn();
  }
  resampleW();

  const f = (t) => Math.sin(2 * Math.PI * t);

  const wFineData = [];
  const sFineData = [];
  for (let i = 0; i <= Nfine; i++) {
    const t = i / Nfine;
    wFineData.push({ x: t, y: wPath[i] });
    sFineData.push({ x: t, y: f(t) });
  }
  function rebuildWFine() {
    for (let i = 0; i <= Nfine; i++) wFineData[i] = { x: i / Nfine, y: wPath[i] };
  }

  const nEl = document.getElementById('qv-n');
  const nValEl = document.getElementById('qv-n-val');
  const resampleBtn = document.getElementById('qv-resample');
  const statW2 = document.getElementById('qv-stat-w2');
  const statW1 = document.getElementById('qv-stat-w1');
  const statS2 = document.getElementById('qv-stat-s2');
  const statS1 = document.getElementById('qv-stat-s1');

  let wChart = null;
  let sChart = null;

  function update() {
    const idx = parseInt(nEl.value, 10);
    const n = partitions[idx];
    nValEl.textContent = n.toLocaleString();

    const stride = Nfine / n;
    const wNodes = new Array(n + 1);
    const sNodes = new Array(n + 1);
    for (let k = 0; k <= n; k++) {
      const i = Math.round(k * stride);
      const t = i / Nfine;
      wNodes[k] = { x: t, y: wPath[i] };
      sNodes[k] = { x: t, y: f(t) };
    }

    let qvW = 0, fvW = 0, qvS = 0, fvS = 0;
    for (let k = 0; k < n; k++) {
      const dW = wNodes[k + 1].y - wNodes[k].y;
      const dS = sNodes[k + 1].y - sNodes[k].y;
      qvW += dW * dW; fvW += Math.abs(dW);
      qvS += dS * dS; fvS += Math.abs(dS);
    }

    statW2.textContent = qvW.toFixed(3);
    statW1.textContent = fvW.toFixed(2);
    statS2.textContent = qvS.toFixed(3);
    statS1.textContent = fvS.toFixed(3);

    if (!wChart) {
      wChart = new Chart(document.getElementById('qv-w'), {
        type: 'line',
        data: { datasets: [
          { label: 'W', data: wFineData, borderColor: '#534AB7', borderWidth: 0.8, pointRadius: 0, fill: false, order: 2 },
          { label: 'nodes', data: wNodes, borderColor: '#2c3e94', borderWidth: 0, pointRadius: 2.5, pointBackgroundColor: '#2c3e94', showLine: false, order: 1 }
        ]},
        options: {
          animation: false, responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: {
            x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 't' } },
            y: { title: { display: true, text: 'W' } }
          }
        }
      });
    } else {
      wChart.data.datasets[0].data = wFineData;
      wChart.data.datasets[1].data = wNodes;
      wChart.update('none');
    }

    if (!sChart) {
      sChart = new Chart(document.getElementById('qv-s'), {
        type: 'line',
        data: { datasets: [
          { label: 'sin', data: sFineData, borderColor: '#D97757', borderWidth: 1, pointRadius: 0, fill: false, order: 2 },
          { label: 'nodes', data: sNodes, borderColor: '#2c3e94', borderWidth: 0, pointRadius: 2.5, pointBackgroundColor: '#2c3e94', showLine: false, order: 1 }
        ]},
        options: {
          animation: false, responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: {
            x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 't' } },
            y: { title: { display: true, text: 'sin(2&#960;t)' }, min: -1.2, max: 1.2 }
          }
        }
      });
    } else {
      sChart.data.datasets[1].data = sNodes;
      sChart.update('none');
    }
  }

  nEl.addEventListener('input', update);
  resampleBtn.addEventListener('click', () => { resampleW(); rebuildWFine(); update(); });
  update();
})();
</script>

</div>

**Diffusion processes.** A general *diffusion process* may be characterised as a continuous-time, continuous-state Markov process with almost surely continuous sample paths. The Markov property ensures that future evolution depends only on the present state, not on history. Since paths are continuous and memoryless, the process is fully determined by its local, *infinitesimal* behaviour, captured by the **infinitesimal generator** (TODO: what? The definition below looks like a generalization of the derivative, where the next values is not deterministic, but defined by some distribution) $\mathscr{L}$:

$$
(\mathscr{L} f)(x) := \lim_{t \searrow 0} \frac{\mathbb{E}[\,f(X_t)\,\mid X_0 = x\,] - f(x)}{t}.
$$

For sufficiently regular diffusions (TODO: wdim "sufficiently regular diffusion"?) in $\mathbb{R}^d$, $\mathscr{L}$ takes the form of a second-order elliptic differential operator (TODO: what? What is differential operator? What is elliptic differential operator?),

$$
\mathscr{L} = \sum_{i=1}^d \mu_i(x) \frac{\partial}{\partial x_i} + \frac{1}{2} \sum_{i,j=1}^d \Sigma_{ij}(x) \frac{\partial^2}{\partial x_i \partial x_j}. \tag{1.1}
$$

### 1.1 Stochastic Differential Equations and Their Interpretation

While the generator describes a diffusion abstractly via its action on functions, *stochastic differential equations* (SDEs) provide a constructive, pathwise representation. They are the central mathematical tool for modelling systems subjected to noise. Let $W_t$ be an $m$-dimensional Brownian motion; we represent the state $X_t \in \mathbb{R}^d$ via the SDE

$$
\mathrm{d} X_t = \mu(t, X_t)\, \mathrm{d} t + \sigma(t, X_t)\, \mathrm{d} W_t. \tag{1.2}
$$

Rigorously, (1.2) has no independent meaning; it is shorthand for the continuous **Itô integral equation**

$$
X_t = X_0 + \int_0^t \mu(s, X_s)\, \mathrm{d} s + \int_0^t \sigma(s, X_s)\, \mathrm{d} W_s,
$$

where the second integral is constructed in the **Itô sense**, capitalising on the martingale property of $W_t$. The physical meaning of the coefficients $\mu \in \mathbb{R}^d$, $\sigma \in \mathbb{R}^{d \times m}$ becomes evident by analysing the infinitesimal moments of the increments $\Delta X_t := X_{t+h} - X_t$, conditional on $X_t = x$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Drift Vector and Diffusion Matrix)</span></p>

* **The drift vector $\mu(t, x)$** represents the deterministic tendency, or expected macroscopic velocity, of the process. Because the Itô integral of Brownian motion is a martingale with expectation zero,

  $$
  \lim_{h \searrow 0} \frac{1}{h} \mathbb{E}[\,X_{t+h} - X_t \mid X_t = x\,] = \mu(t, x).
  $$

* **The diffusion matrix $\Sigma(t, x) := \sigma(t, x) \sigma(t, x)^\top$.** The function $\sigma(t, x)$ determines the *volatility*. Due to the Brownian scaling $(\mathrm{d} W_t)^2 = \mathrm{d} t$ (TODO: what and why?), the local covariance is dominated entirely by the stochastic term. Applying *Itô's isometry* yields

  $$
  \lim_{h \searrow 0} \frac{1}{h} \mathbb{E}\!\left[(X_{t+h} - X_t)(X_{t+h} - X_t)^\top \,\big|\, X_t = x\right] = \sigma(t, x) \sigma(t, x)^\top =: \Sigma(t, x).
  $$

</div>

Notice that $\Sigma(t, x)$ precisely matches the second-order coefficient matrix in the generator (1.1).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Itô's Lemma — The Chain Rule for SDEs)</span></p>

To manipulate SDEs, classical calculus is insufficient because $(\mathrm{d} W_t)^2 = \mathrm{d} t$. If $f(t, x)$ is twice continuously differentiable, then $Y_t := f(t, X_t)$ is again an Itô process with dynamics governed by **Itô's formula**,

$$
\mathrm{d} f(t, X_t) = \left(\frac{\partial f}{\partial t} + \mathscr{L} f\right)\! \mathrm{d} t + \Bigl(\nabla_x f(t, X_t)^\top \sigma(t, X_t)\Bigr) \mathrm{d} W_t.
$$

</div>

Itô's formula lies at the heart of stochastic analysis, connecting probability and partial differential equations.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Existence and Uniqueness of Solutions)</span></p>

Whether an SDE like (1.2) admits a valid solution is highly non-trivial.

* **Strong solutions (Itô's approach).** Given a specific Brownian motion $W_t$, we seek a pathwise unique process $X_t$. Itô's theorem guarantees existence and uniqueness provided the coefficients are *globally Lipschitz*,

  $$
  \lVert \mu(t, x) - \mu(t, y) \rVert + \lVert \sigma(t, x) - \sigma(t, y) \rVert \le K \lVert x - y \rVert,
  $$

  and satisfy a *linear growth* bound to prevent finite-time explosion,

  $$
  \lVert \mu(t, x) \rVert + \lVert \sigma(t, x) \rVert \le K (1 + \lVert x \rVert).
  $$

* **Weak solutions (martingale problem).** Dropping the Lipschitz assumption typically kills strong uniqueness. However, Stroock and Varadhan showed one can still find a probability measure $\mathbb{P}$ on path space under which the coordinate process solves the SDE — by proving that, for any regular test function $f$,

  $$
  M^f_t := f(X_t) - f(X_0) - \int_0^t (\mathscr{L} f)(X_s)\, \mathrm{d} s
  $$

  is a local martingale. This deep connection allows construction of diffusions even with highly irregular drifts.

</div>

**The macroscopic view: the Fokker–Planck equation.** So far our treatment has been strictly pathwise (microscopic), tracking a single random trajectory $X_t$. To understand the statistical ensemble, we shift perspective to the *macroscopic evolution* of the probability distribution.

Assume the law of $X_t$ admits a density $p(t, x)$ with respect to the Lebesgue measure on $\mathbb{R}^d$, and that we seek a governing equation for $p(t, x)$. Recall the generator $\mathscr{L}$ from (1.1). For any smooth, compactly supported test function $f \in C_c^\infty(\mathbb{R}^d)$, the expectation $\mathbb{E}[f(X_t)]$ evolves according to **Kolmogorov's backward equation**,

$$
\frac{\mathrm{d}}{\mathrm{d} t} \mathbb{E}[f(X_t)] = \mathbb{E}\!\left[(\mathscr{L} f)(X_t)\right].
$$

Rewriting this in terms of $p(t, x)$ yields

$$
\int_{\mathbb{R}^d} f(x)\, \frac{\partial p(t, x)}{\partial t}\, \mathrm{d} x = \int_{\mathbb{R}^d} (\mathscr{L} f)(x)\, p(t, x)\, \mathrm{d} x.
$$

To isolate the evolution of $p(t, x)$, transfer $\mathscr{L}$ from $f$ onto $p$ via integration by parts (boundary terms vanish at infinity). This introduces the **formal adjoint operator** $\mathscr{L}^*$:

$$
\int_{\mathbb{R}^d} f(x)\, \frac{\partial p(t, x)}{\partial t}\, \mathrm{d} x = \int_{\mathbb{R}^d} f(x)\, (\mathscr{L}^* p)(t, x)\, \mathrm{d} x. \tag{1.3}
$$

Since this holds for *all* test functions $f$, the density itself must satisfy the **Fokker–Planck equation** (or Kolmogorov forward equation):

$$
\frac{\partial p(t, x)}{\partial t} = \mathscr{L}^* p(t, x) = -\sum_{i=1}^d \frac{\partial}{\partial x_i}\!\left[\mu_i(t, x)\, p(t, x)\right] + \frac{1}{2} \sum_{i,j=1}^d \frac{\partial^2}{\partial x_i \partial x_j}\!\left[\Sigma_{ij}(t, x)\, p(t, x)\right].
$$

<div style="margin: 1.2rem 0 1.5rem;">

<p style="margin: 0 0 0.4rem 0; font-style: italic; color: #5F5E5A;">Interactive: micro/macro view of the Ornstein–Uhlenbeck SDE $\mathrm{d}X_t = -X_t\, \mathrm{d}t + \sqrt{2}\, \mathrm{d}W_t$ with $X_0 = 2$. <strong>Left:</strong> $30$ pathwise (microscopic) Euler–Maruyama trajectories on $[0, 4]$ with a vertical line at the current time. <strong>Right:</strong> the macroscopic density $p(t, x) = \mathcal{N}\!\bigl(x;\, 2 e^{-t},\, 1 - e^{-2t}\bigr)$ — the closed-form solution of the Fokker–Planck equation above for these coefficients — alongside a histogram of the path values at time $t$. Scrub $t$ to watch the $\delta_{X_0}$ initial mass smear into the stationary $\mathcal{N}(0,1)$.</p>

<div style="margin: 1rem 0;">
  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">
    <label for="fp-t" style="font-size: 14px; color: var(--color-text-secondary, #5F5E5A); min-width: 80px;">Time t</label>
    <input type="range" min="0" max="4" value="0.5" step="0.01" id="fp-t" style="flex: 1; min-width: 200px;" />
    <span id="fp-t-val" style="font-size: 14px; font-weight: 500; min-width: 56px; text-align: right;">0.50</span>
    <button id="fp-resample" style="padding: 6px 14px;">Resample paths</button>
  </div>

  <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 1rem;">
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">Mean m(t) = 2 e<sup>&minus;t</sup></div>
      <div id="fp-stat-m" style="font-size: 20px; font-weight: 500;">&mdash;</div>
    </div>
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">Variance v(t) = 1 &minus; e<sup>&minus;2t</sup></div>
      <div id="fp-stat-v" style="font-size: 20px; font-weight: 500;">&mdash;</div>
    </div>
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">Stationary law</div>
      <div style="font-size: 20px; font-weight: 500;">&Nscr;(0, 1)</div>
    </div>
  </div>

  <div style="display: grid; grid-template-columns: 3fr 2fr; gap: 12px;">
    <div>
      <div style="font-size: 13px; font-weight: 500; margin-bottom: 4px; color: #2c3e94;">Sample paths X<sub>t</sub></div>
      <div style="position: relative; height: 300px;">
        <canvas id="fp-paths" role="img" aria-label="Thirty Euler-Maruyama sample paths of the Ornstein-Uhlenbeck process on the time interval from 0 to 4, with a vertical line marking the current time."></canvas>
      </div>
    </div>
    <div>
      <div style="font-size: 13px; font-weight: 500; margin-bottom: 4px; color: #2c3e94;">Density p(t, x)</div>
      <div style="position: relative; height: 300px;">
        <canvas id="fp-dens" role="img" aria-label="Gaussian density p(t, x) at the current time, alongside a histogram of the simulated path values at the current time."></canvas>
      </div>
    </div>
  </div>
</div>

<script>
(function () {
  const T = 4;
  const Nsteps = 400;
  const dt = T / Nsteps;
  const sdt = Math.sqrt(dt);
  const sigma = Math.SQRT2;
  const x0 = 2;
  const numPaths = 30;
  const histPaths = 2000;

  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Visible (small) paths for left panel.
  let paths = [];
  // Many lightweight paths for right-panel histogram (only need value at current time).
  let bigEnsemble = new Float64Array(histPaths * (Nsteps + 1));

  function simulate() {
    paths = [];
    for (let p = 0; p < numPaths; p++) {
      const arr = new Float64Array(Nsteps + 1);
      arr[0] = x0;
      for (let i = 1; i <= Nsteps; i++) arr[i] = arr[i - 1] - arr[i - 1] * dt + sigma * sdt * randn();
      paths.push(arr);
    }
    for (let p = 0; p < histPaths; p++) {
      let prev = x0;
      bigEnsemble[p * (Nsteps + 1)] = x0;
      for (let i = 1; i <= Nsteps; i++) {
        prev = prev - prev * dt + sigma * sdt * randn();
        bigEnsemble[p * (Nsteps + 1) + i] = prev;
      }
    }
  }
  simulate();

  const tEl = document.getElementById('fp-t');
  const tValEl = document.getElementById('fp-t-val');
  const statM = document.getElementById('fp-stat-m');
  const statV = document.getElementById('fp-stat-v');
  const resampleBtn = document.getElementById('fp-resample');

  let pathsChart = null;
  let densChart = null;

  // x-grid for density plot
  const xGrid = [];
  const NX = 240;
  const xMin = -4, xMax = 4;
  for (let i = 0; i <= NX; i++) xGrid.push(xMin + (xMax - xMin) * i / NX);

  function gauss(x, m, v) {
    return Math.exp(-(x - m) * (x - m) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
  }

  function update() {
    const t = parseFloat(tEl.value);
    tValEl.textContent = t.toFixed(2);
    const m = x0 * Math.exp(-t);
    const v = 1 - Math.exp(-2 * t);
    statM.textContent = m.toFixed(3);
    statV.textContent = v.toFixed(3);

    // Path datasets for left panel
    const pathDatasets = paths.map((arr) => {
      const data = [];
      for (let i = 0; i <= Nsteps; i++) data.push({ x: i * dt, y: arr[i] });
      return { data, borderColor: 'rgba(83, 74, 183, 0.45)', borderWidth: 0.7, pointRadius: 0, fill: false };
    });
    pathDatasets.push({ data: [{ x: t, y: -4 }, { x: t, y: 4 }], borderColor: '#D97757', borderWidth: 1.4, borderDash: [4, 4], pointRadius: 0, fill: false });

    if (!pathsChart) {
      pathsChart = new Chart(document.getElementById('fp-paths'), {
        type: 'line',
        data: { datasets: pathDatasets },
        options: {
          animation: false, responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: {
            x: { type: 'linear', min: 0, max: T, title: { display: true, text: 't' } },
            y: { min: -4, max: 4, title: { display: true, text: 'X' } }
          }
        }
      });
    } else {
      pathsChart.data.datasets = pathDatasets;
      pathsChart.update('none');
    }

    // Histogram of bigEnsemble at index = t/dt
    const idx = Math.min(Nsteps, Math.max(0, Math.round(t / dt)));
    const stride = Nsteps + 1;
    const bins = 50;
    const counts = new Array(bins).fill(0);
    let inRange = 0;
    for (let p = 0; p < histPaths; p++) {
      const val = bigEnsemble[p * stride + idx];
      if (val >= xMin && val <= xMax) {
        const b = Math.min(bins - 1, Math.floor((val - xMin) / (xMax - xMin) * bins));
        counts[b]++; inRange++;
      }
    }
    const binWidth = (xMax - xMin) / bins;
    const denom = histPaths * binWidth;
    const histData = [{ x: xMin, y: 0 }];
    for (let b = 0; b < bins; b++) {
      const left = xMin + b * binWidth, right = left + binWidth;
      const h = counts[b] / denom;
      histData.push({ x: left, y: h });
      histData.push({ x: right, y: h });
    }
    histData.push({ x: xMax, y: 0 });

    const safeV = Math.max(v, 1e-4);
    const densData = xGrid.map((xx) => ({ x: xx, y: gauss(xx, m, safeV) }));
    let peak = 0;
    for (const d of densData) if (d.y > peak) peak = d.y;
    for (let b = 0; b < bins; b++) peak = Math.max(peak, counts[b] / denom);

    if (!densChart) {
      densChart = new Chart(document.getElementById('fp-dens'), {
        type: 'line',
        data: { datasets: [
          { label: 'empirical', data: histData, borderColor: 'rgba(83,74,183,0.55)', backgroundColor: 'rgba(83,74,183,0.16)', borderWidth: 1, pointRadius: 0, fill: 'origin', tension: 0, order: 2 },
          { label: 'p(t,x)', data: densData, borderColor: '#D97757', borderWidth: 1.6, pointRadius: 0, fill: false, tension: 0, order: 1 }
        ]},
        options: {
          animation: false, responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: {
            x: { type: 'linear', min: xMin, max: xMax, title: { display: true, text: 'x' } },
            y: { title: { display: true, text: 'density' }, beginAtZero: true, max: Math.max(peak * 1.1, 0.6) }
          }
        }
      });
    } else {
      densChart.data.datasets[0].data = histData;
      densChart.data.datasets[1].data = densData;
      densChart.options.scales.y.max = Math.max(peak * 1.1, 0.6);
      densChart.update('none');
    }
  }

  tEl.addEventListener('input', update);
  resampleBtn.addEventListener('click', () => { simulate(); update(); });
  update();
})();
</script>

</div>

**Reversing time: the bridge to generative models.** We have now described how a diffusion process $(X_t)\_{t \in [0, T]}$ evolves forwards from an initial distribution $p_0$ to a terminal distribution $p_T$. A natural and fundamental question is:

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Question</span><span class="math-callout__name">(Time-Reversal)</span></p>

If we observe this stochastic process backwards in time, starting from $T$ and moving towards $0$, what are its dynamics?

</div>

Define the reversed time $\bar{t} := T - t$ and the reversed process $\bar{X}\_{\bar t} := X_{T - \bar t}$. A priori, reversing a Markov process need not yield a process with a continuous local-martingale part. However, a foundational result by *Anderson (1982)* shows that — under mild regularity on the density $p(t, x)$ — the time-reversed process is itself a diffusion.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">1.1 (Time-Reversal of Diffusions)</span></p>

Let $X_t$ be the solution to the forward SDE (1.2), and let $\bar{\mathcal{F}}\_{\bar t}$ be the reversed filtration generated by $\sigma(X_s : s \ge T - \bar t)$. Then the reversed process $\bar{X}\_{\bar t}$ satisfies the **reverse SDE**

$$
\mathrm{d}\bar{X}_{\bar t} = \bar{\mu}(\bar t, \bar{X}_{\bar t})\, \mathrm{d} \bar t + \sigma(T - \bar t, \bar{X}_{\bar t})\, \mathrm{d} \bar{W}_{\bar t},
$$

where $\bar{W}\_{\bar t}$ is a standard Brownian motion with respect to the reversed filtration $\bar{\mathcal{F}}\_{\bar t}$, and the reverse drift $\bar{\mu}$ is given by

$$
\bar{\mu}(\bar t, x) = -\mu(T - \bar t, x) + \frac{1}{p(T - \bar t, x)} \nabla_x \cdot \bigl[\Sigma(T - \bar t, x)\, p(T - \bar t, x)\bigr],
$$

where the divergence operator $\nabla_x \cdot$ acts column-wise on the matrix.

</div>

### 1.2 Diffusion Models

For the specific class of SDEs predominantly used in recent applications, the diffusion coefficient depends only on time, i.e. $\sigma(t, x) \equiv g(t) I$ for a scalar function $g$. In this case, the diffusion matrix simplifies to $\Sigma(t) = g^2(t) I$. Substituting into Anderson's formula yields a simplified reverse drift,

$$
\bar{\mu}(\bar t, x) = -\mu(T - \bar t, x) + g^2(T - \bar t)\, \nabla_x \log p(T - \bar t, x).
$$

Rewriting the reverse SDE purely in terms of the original forward time $t$ (integrating backwards from $T$ down to $0$) gives

$$
\mathrm{d} X_t = \bigl[\,\mu(t, X_t) - g^2(t)\, \underbrace{\nabla_x \log p(t, X_t)}_{\text{Score Function}}\,\bigr]\, \mathrm{d} t + g(t)\, \mathrm{d} \bar{W}_t. \tag{1.4}
$$

Equation (1.4) is the bridge between classical stochastic analysis and modern generative AI. It states that to perfectly simulate the backward trajectory — e.g. transforming pure noise $p_T$ back into complex data $p_0$ — we need only know:

1. The analytical forward coefficients $\mu(t, x)$ and $g(t)$ (which *we* choose);
2. The logarithmic gradient of the marginal density, $\nabla_x \log p(t, x)$.

The second term, known in statistics as the **Stein score function**, is the sole unknown. If we can estimate this vector field, we can reverse the arrow of time.

<div style="margin: 1.2rem 0 1.5rem;">

<p style="margin: 0 0 0.4rem 0; font-style: italic; color: #5F5E5A;">Interactive: the score field $\nabla_x \log p(x)$ for a 2D mixture $p = \tfrac{1}{2}\mathcal{N}\!\bigl(\mu_-, \tfrac{1}{2} I\bigr) + \tfrac{1}{2}\mathcal{N}\!\bigl(\mu_+, \tfrac{1}{2} I\bigr)$ with modes $\mu_\pm = (\pm d, 0)$. The shaded background is the density $p$; arrows are the score, which always points <em>uphill</em> on $\log p$ — toward the nearest mode of the data distribution. Reverse-time diffusion is, in essence, gradient ascent along this field with added Brownian noise.</p>

<div style="margin: 1rem 0;">
  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">
    <label for="score-d" style="font-size: 14px; color: var(--color-text-secondary, #5F5E5A); min-width: 130px;">Mode separation d</label>
    <input type="range" min="0.3" max="2.5" value="1.2" step="0.05" id="score-d" style="flex: 1; min-width: 200px;" />
    <span id="score-d-val" style="font-size: 14px; font-weight: 500; min-width: 56px; text-align: right;">1.20</span>
  </div>

  <div style="display: flex; flex-wrap: wrap; gap: 18px; margin-bottom: 8px; font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 14px; height: 8px; background: linear-gradient(to right, #f5f6ff, #534AB7); border: 1px solid rgba(83,74,183,0.4);"></span>
      density p(x)
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 2px; background: #D97757;"></span>
      score &nabla;<sub>x</sub> log p
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 7px; height: 7px; background: #2c3e94; border-radius: 50%;"></span>
      modes &mu;<sub>&plusmn;</sub>
    </span>
  </div>

  <div style="position: relative; width: 100%; aspect-ratio: 3 / 2; max-width: 760px; margin: 0 auto; background: #f5f6ff;">
    <canvas id="score-canvas" role="img" aria-label="Two-dimensional density of a Gaussian mixture with two modes, overlaid by the score vector field which points toward the nearest mode."></canvas>
  </div>
</div>

<script>
(function () {
  const canvas = document.getElementById('score-canvas');
  const dEl = document.getElementById('score-d');
  const dValEl = document.getElementById('score-d-val');

  const xMin = -3, xMax = 3, yMin = -2, yMax = 2;
  const sigma2 = 0.5;
  const wts = [0.5, 0.5];

  function modes(d) { return [[-d, 0], [d, 0]]; }
  function densityAt(x, y, mu) {
    return (1 / (2 * Math.PI * sigma2)) * Math.exp(-((x - mu[0]) * (x - mu[0]) + (y - mu[1]) * (y - mu[1])) / (2 * sigma2));
  }
  function totalDensity(x, y, mus) {
    let s = 0;
    for (let i = 0; i < mus.length; i++) s += wts[i] * densityAt(x, y, mus[i]);
    return s;
  }
  function score(x, y, mus) {
    const ps = [];
    let p = 0;
    for (let i = 0; i < mus.length; i++) { const pi = densityAt(x, y, mus[i]); ps.push(pi); p += wts[i] * pi; }
    if (p === 0) return [0, 0];
    let sx = 0, sy = 0;
    for (let i = 0; i < mus.length; i++) {
      const w = wts[i] * ps[i] / p;
      sx += w * (-(x - mus[i][0]) / sigma2);
      sy += w * (-(y - mus[i][1]) / sigma2);
    }
    return [sx, sy];
  }

  function colorFor(t) {
    const tt = Math.pow(t, 0.7);
    const r = Math.round(245 + (83 - 245) * tt);
    const g = Math.round(246 + (74 - 246) * tt);
    const b = Math.round(255 + (183 - 255) * tt);
    return `rgb(${r},${g},${b})`;
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 10) { requestAnimationFrame(draw); return; }
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = rect.width, H = rect.height;

    const d = parseFloat(dEl.value);
    dValEl.textContent = d.toFixed(2);
    const mus = modes(d);

    const cell = 4;
    const cols = Math.ceil(W / cell), rows = Math.ceil(H / cell);
    let pmax = 0;
    const grid = new Float64Array(cols * rows);
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const x = xMin + (i + 0.5) / cols * (xMax - xMin);
        const y = yMax - (j + 0.5) / rows * (yMax - yMin);
        const p = totalDensity(x, y, mus);
        grid[j * cols + i] = p;
        if (p > pmax) pmax = p;
      }
    }
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const t = Math.min(1, grid[j * cols + i] / (pmax + 1e-9));
        ctx.fillStyle = colorFor(t);
        ctx.fillRect(i * cell, j * cell, cell + 0.6, cell + 0.6);
      }
    }

    const aCols = 22, aRows = 14;
    let smax = 0;
    const sxs = new Float64Array(aCols * aRows);
    const sys = new Float64Array(aCols * aRows);
    for (let j = 0; j < aRows; j++) {
      for (let i = 0; i < aCols; i++) {
        const x = xMin + (i + 0.5) / aCols * (xMax - xMin);
        const y = yMax - (j + 0.5) / aRows * (yMax - yMin);
        const [sx, sy] = score(x, y, mus);
        sxs[j * aCols + i] = sx;
        sys[j * aCols + i] = sy;
        const m = Math.hypot(sx, sy);
        if (m > smax) smax = m;
      }
    }
    const arrowMaxLen = Math.min(W / aCols, H / aRows) * 0.7;
    ctx.strokeStyle = 'rgba(217, 119, 87, 0.9)';
    ctx.fillStyle = 'rgba(217, 119, 87, 0.9)';
    ctx.lineWidth = 1;
    for (let j = 0; j < aRows; j++) {
      for (let i = 0; i < aCols; i++) {
        const px = (i + 0.5) / aCols * W;
        const py = (j + 0.5) / aRows * H;
        const sx = sxs[j * aCols + i], sy = sys[j * aCols + i];
        const m = Math.hypot(sx, sy);
        if (m < 1e-3) continue;
        const lenScale = Math.pow(m / (smax + 1e-9), 0.7) * arrowMaxLen;
        const dx = sx / m * lenScale;
        const dy = -sy / m * lenScale;
        const ex = px + dx, ey = py + dy;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        const ang = Math.atan2(dy, dx);
        const ah = 4.5;
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - ah * Math.cos(ang - Math.PI / 6), ey - ah * Math.sin(ang - Math.PI / 6));
        ctx.lineTo(ex - ah * Math.cos(ang + Math.PI / 6), ey - ah * Math.sin(ang + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
      }
    }

    ctx.fillStyle = '#2c3e94';
    for (const mu of mus) {
      const px = (mu[0] - xMin) / (xMax - xMin) * W;
      const py = (yMax - mu[1]) / (yMax - yMin) * H;
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  dEl.addEventListener('input', draw);
  window.addEventListener('resize', draw);
  window.addEventListener('load', draw);
  requestAnimationFrame(draw);
})();
</script>

</div>

**Score matching and generative synthesis.** The challenge of generative modelling reduces to estimating the time-dependent vector field $\nabla_x \log p(t, x)$. Assuming access to an i.i.d. dataset $X_1, \dots, X_n \sim p_0$ (the unknown, highly complex data distribution), introduce a parametric family — typically a neural network — $s_\theta(x, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$. We aim to find parameters $\theta$ such that $s_\theta(x, t) \approx \nabla_x \log p(t, x)$ for all $t \in [0, T]$. The natural loss for this regression task is the expected $L^2$-distance, integrated over time — **explicit score matching (ESM)**:

$$
\mathcal{J}_{\mathrm{ESM}}(\theta) = \frac{1}{2} \int_0^T \lambda(t)\, \mathbb{E}_{X_t \sim p_t}\!\left[\,\bigl\lVert s_\theta(X_t, t) - \nabla_{X_t} \log p(t, X_t) \bigr\rVert_2^2 \,\right] \mathrm{d} t,
$$

with a positive weighting $\lambda(t) > 0$. Minimising $\mathcal{J}\_{\mathrm{ESM}}$ directly is impossible in practice: the marginal density at time $t$ is 

$$p(t, x) = \int_{\mathbb{R}^d} p_{t\mid 0}(x \mid x_0)\, p_0(x_0)\, \mathrm{d} x_0$$

since $p_0$ is unknown and the integral is high-dimensional, the true marginal score $\nabla_x \log p(t, x)$ is fundamentally intractable.

To circumvent this intractability one relies on a profound equivalence (Hyvärinen 2005; extended to conditional distributions by Vincent 2011) allowing one to bypass the marginal score entirely by substituting the *conditional* score of the forward process.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">1.2 (Denoising Score Matching)</span></p>

Assume the transition density $p_{t \mid 0}(x \mid x_0)$ is differentiable in $x$ and that boundary terms vanish at infinity. Then, up to an additive constant $C$ independent of $\theta$, the explicit score matching objective is equivalent to the **denoising score matching (DSM)** objective,

$$
\mathcal{J}_{\mathrm{DSM}}(\theta) = \frac{1}{2} \int_0^T \lambda(t)\, \mathbb{E}_{X_0 \sim p_0}\, \mathbb{E}_{X_t \sim p_{t \mid 0}(\cdot \mid X_0)}\!\left[\,\bigl\lVert s_\theta(X_t, t) - \nabla_{X_t} \log p_{t \mid 0}(X_t \mid X_0) \bigr\rVert_2^2 \,\right] \mathrm{d} t. \tag{1.5}
$$

That is,

$$
\arg\min_\theta\, \mathcal{J}_{\mathrm{ESM}}(\theta) = \arg\min_\theta\, \mathcal{J}_{\mathrm{DSM}}(\theta).
$$

</div>

Notice that the intractable *global* score $\nabla_{X_t} \log p(t, X_t)$ is replaced by the local *conditional* score $\nabla_{X_t} \log p_{t \mid 0}(X_t \mid X_0)$. Because the forward SDE is chosen by us, the transition kernel $p_{t \mid 0}$ is a known mathematical object.

To make (1.5) analytically computable, one must carefully design the forward SDE. Restricting the drift $\mu(t, x)$ to be an affine function of $x$ and the diffusion coefficient $\sigma(t, x)$ to depend only on time makes the process Gaussian. A ubiquitous choice in modern diffusion models is the **variance-preserving (VP) SDE**, a time-scaled Ornstein–Uhlenbeck process:

$$
\mathrm{d} X_t = -\tfrac{1}{2} \beta(t)\, X_t\, \mathrm{d} t + \sqrt{\beta(t)}\, \mathrm{d} W_t,
$$

where $\beta(t) > 0$ is a deterministically scheduled noise rate. For this linear SDE the transition kernel is known exactly: given $X_0 = x_0$, $X_t$ is normally distributed,

$$
p_{t \mid 0}(x \mid x_0) = \mathcal{N}\bigl(x;\, m(t) x_0,\, v(t) I\bigr),
$$

with
* **mean scaling** $m(t) = \exp\bigl(-\tfrac{1}{2} \int_0^t \beta(s)\, \mathrm{d} s\bigr)$ and 
* **variance** $v(t) = 1 - \exp\bigl(-\int_0^t \beta(s)\, \mathrm{d} s\bigr)$. 

<div style="margin: 1.2rem 0 1.5rem;">

<p style="margin: 0 0 0.4rem 0; font-style: italic; color: #5F5E5A;">Interactive: VP-SDE transition kernel under the linear schedule $\beta(t) = \beta_{\min} + \tfrac{t}{T}(\beta_{\max} - \beta_{\min})$ on $[0, T = 1]$ with $x_0 = 1.5$. <strong>Top:</strong> the mean-scaling $m(t)$ decaying from $1$ toward $0$ alongside the variance $v(t)$ rising toward $1$ — together they implement the <em>variance-preserving</em> property $m(t)^2 + v(t) = 1$ for the case $x_0 \sim \mathcal{N}(0, 1)$. <strong>Bottom:</strong> at the chosen time $t$, an empirical histogram of $5{,}000$ noised samples $X_t = m(t) x_0 + \sqrt{v(t)}\,\varepsilon$ alongside the analytic Gaussian density. As $t \to T$ the cloud collapses to standard noise — exactly what the reverse SDE of (1.4) starts from.</p>

<div style="margin: 1rem 0;">
  <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 8px;">
    <div style="display: flex; align-items: center; gap: 8px;">
      <label for="vp-bmin" style="font-size: 13px; color: var(--color-text-secondary, #5F5E5A); min-width: 60px;">&beta;<sub>min</sub></label>
      <input type="range" min="0.05" max="2" value="0.1" step="0.01" id="vp-bmin" style="flex: 1;" />
      <span id="vp-bmin-val" style="font-size: 13px; font-weight: 500; min-width: 40px; text-align: right;">0.10</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
      <label for="vp-bmax" style="font-size: 13px; color: var(--color-text-secondary, #5F5E5A); min-width: 60px;">&beta;<sub>max</sub></label>
      <input type="range" min="2" max="40" value="20" step="0.5" id="vp-bmax" style="flex: 1;" />
      <span id="vp-bmax-val" style="font-size: 13px; font-weight: 500; min-width: 40px; text-align: right;">20.0</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
      <label for="vp-t" style="font-size: 13px; color: var(--color-text-secondary, #5F5E5A); min-width: 60px;">time t</label>
      <input type="range" min="0" max="1" value="0.4" step="0.01" id="vp-t" style="flex: 1;" />
      <span id="vp-t-val" style="font-size: 13px; font-weight: 500; min-width: 40px; text-align: right;">0.40</span>
    </div>
  </div>
  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">
    <button id="vp-resample" style="padding: 6px 14px;">Resample &epsilon;</button>
    <span style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">x<sub>0</sub> = 1.5, T = 1</span>
  </div>

  <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 1rem;">
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">m(t)</div>
      <div id="vp-stat-m" style="font-size: 20px; font-weight: 500;">&mdash;</div>
    </div>
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">v(t)</div>
      <div id="vp-stat-v" style="font-size: 20px; font-weight: 500;">&mdash;</div>
    </div>
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">m(T), v(T)</div>
      <div id="vp-stat-end" style="font-size: 16px; font-weight: 500;">&mdash;</div>
    </div>
  </div>

  <div style="display: flex; flex-wrap: wrap; gap: 18px; margin-bottom: 8px; font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 2px; background: #534AB7;"></span>
      m(t)
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 2px; background: #D97757;"></span>
      v(t)
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 0; border-top: 1.5px dashed rgba(83,74,183,0.55);"></span>
      current t
    </span>
  </div>

  <div style="position: relative; width: 100%; height: 220px; margin-bottom: 12px;">
    <canvas id="vp-mv" role="img" aria-label="Plot of the mean-scaling m of t and variance v of t over the unit time interval, with a dashed vertical line at the current time."></canvas>
  </div>
  <p style="margin: 0 0 0.4rem 0; font-style: italic; color: #5F5E5A;">Histogram of $X_t$ at current $t$ (5,000 samples) vs. $\mathcal{N}\!\bigl(m(t) x_0,\, v(t)\bigr)$.</p>
  <div style="position: relative; width: 100%; height: 240px;">
    <canvas id="vp-kernel" role="img" aria-label="Histogram of the noised samples X_t at the current time, overlaid with the analytic Gaussian density of the transition kernel."></canvas>
  </div>
</div>

<script>
(function () {
  const T = 1, x0 = 1.5;
  const Mhist = 5000;
  const epsilons = new Float64Array(Mhist);

  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function resampleEps() { for (let i = 0; i < Mhist; i++) epsilons[i] = randn(); }
  resampleEps();

  const bminEl = document.getElementById('vp-bmin');
  const bmaxEl = document.getElementById('vp-bmax');
  const tEl = document.getElementById('vp-t');
  const bminValEl = document.getElementById('vp-bmin-val');
  const bmaxValEl = document.getElementById('vp-bmax-val');
  const tValEl = document.getElementById('vp-t-val');
  const statM = document.getElementById('vp-stat-m');
  const statV = document.getElementById('vp-stat-v');
  const statEnd = document.getElementById('vp-stat-end');
  const resampleBtn = document.getElementById('vp-resample');

  let mvChart = null;
  let kernelChart = null;

  // integral of beta from 0 to t for linear schedule
  function intBeta(t, bmin, bmax) {
    return bmin * t + 0.5 * (bmax - bmin) * t * t / T;
  }
  function mAt(t, bmin, bmax) { return Math.exp(-0.5 * intBeta(t, bmin, bmax)); }
  function vAt(t, bmin, bmax) { return 1 - Math.exp(-intBeta(t, bmin, bmax)); }

  const Ng = 200;
  const tGrid = [];
  for (let i = 0; i <= Ng; i++) tGrid.push(i * T / Ng);

  const xMin = -3.5, xMax = 3.5;
  const Nx = 240;
  const xGrid = [];
  for (let i = 0; i <= Nx; i++) xGrid.push(xMin + (xMax - xMin) * i / Nx);

  function update() {
    const bmin = parseFloat(bminEl.value);
    const bmax = parseFloat(bmaxEl.value);
    const t = parseFloat(tEl.value);
    bminValEl.textContent = bmin.toFixed(2);
    bmaxValEl.textContent = bmax.toFixed(1);
    tValEl.textContent = t.toFixed(2);

    const mNow = mAt(t, bmin, bmax);
    const vNow = vAt(t, bmin, bmax);
    statM.textContent = mNow.toFixed(3);
    statV.textContent = vNow.toFixed(3);
    statEnd.textContent = `${mAt(T, bmin, bmax).toFixed(3)},  ${vAt(T, bmin, bmax).toFixed(3)}`;

    const mData = tGrid.map((tt) => ({ x: tt, y: mAt(tt, bmin, bmax) }));
    const vData = tGrid.map((tt) => ({ x: tt, y: vAt(tt, bmin, bmax) }));
    const tLine = [{ x: t, y: -0.05 }, { x: t, y: 1.1 }];

    if (!mvChart) {
      mvChart = new Chart(document.getElementById('vp-mv'), {
        type: 'line',
        data: { datasets: [
          { label: 'm', data: mData, borderColor: '#534AB7', borderWidth: 1.6, pointRadius: 0, fill: false, tension: 0.1, order: 1 },
          { label: 'v', data: vData, borderColor: '#D97757', borderWidth: 1.6, pointRadius: 0, fill: false, tension: 0.1, order: 2 },
          { label: 'tline', data: tLine, borderColor: 'rgba(83,74,183,0.55)', borderWidth: 1, borderDash: [4, 4], pointRadius: 0, fill: false, order: 3 }
        ]},
        options: {
          animation: false, responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: {
            x: { type: 'linear', min: 0, max: T, title: { display: true, text: 't' } },
            y: { min: -0.05, max: 1.1, title: { display: true, text: 'value' } }
          }
        }
      });
    } else {
      mvChart.data.datasets[0].data = mData;
      mvChart.data.datasets[1].data = vData;
      mvChart.data.datasets[2].data = tLine;
      mvChart.update('none');
    }

    // histogram of X_t = m * x0 + sqrt(v) * eps
    const sigma = Math.sqrt(Math.max(vNow, 1e-10));
    const center = mNow * x0;
    const bins = 50;
    const counts = new Array(bins).fill(0);
    for (let i = 0; i < Mhist; i++) {
      const xv = center + sigma * epsilons[i];
      if (xv >= xMin && xv <= xMax) {
        const b = Math.min(bins - 1, Math.floor((xv - xMin) / (xMax - xMin) * bins));
        counts[b]++;
      }
    }
    const binWidth = (xMax - xMin) / bins;
    const denom = Mhist * binWidth;
    const histData = [{ x: xMin, y: 0 }];
    let peakHist = 0;
    for (let b = 0; b < bins; b++) {
      const left = xMin + b * binWidth, right = left + binWidth;
      const h = counts[b] / denom;
      histData.push({ x: left, y: h });
      histData.push({ x: right, y: h });
      if (h > peakHist) peakHist = h;
    }
    histData.push({ x: xMax, y: 0 });

    const densData = xGrid.map((xx) => ({ x: xx, y: Math.exp(-(xx - center) * (xx - center) / (2 * Math.max(vNow, 1e-10))) / Math.sqrt(2 * Math.PI * Math.max(vNow, 1e-10)) }));
    let peakD = 0;
    for (const d of densData) if (d.y > peakD) peakD = d.y;
    const yMax = Math.max(peakD, peakHist) * 1.1 + 1e-9;

    if (!kernelChart) {
      kernelChart = new Chart(document.getElementById('vp-kernel'), {
        type: 'line',
        data: { datasets: [
          { label: 'empirical', data: histData, borderColor: 'rgba(83,74,183,0.7)', backgroundColor: 'rgba(83,74,183,0.18)', borderWidth: 1, pointRadius: 0, fill: 'origin', tension: 0, order: 2 },
          { label: 'analytic', data: densData, borderColor: '#D97757', borderWidth: 1.6, pointRadius: 0, fill: false, tension: 0, order: 1 }
        ]},
        options: {
          animation: false, responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: {
            x: { type: 'linear', min: xMin, max: xMax, title: { display: true, text: 'x' } },
            y: { title: { display: true, text: 'density' }, beginAtZero: true, max: yMax }
          }
        }
      });
    } else {
      kernelChart.data.datasets[0].data = histData;
      kernelChart.data.datasets[1].data = densData;
      kernelChart.options.scales.y.max = yMax;
      kernelChart.update('none');
    }
  }

  bminEl.addEventListener('input', update);
  bmaxEl.addEventListener('input', update);
  tEl.addEventListener('input', update);
  resampleBtn.addEventListener('click', () => { resampleEps(); update(); });
  update();
})();
</script>

</div>

Consequently, the conditional score is simply the score of a Gaussian, with trivial closed form:

$$
\nabla_x \log p_{t \mid 0}(x \mid x_0) = -\frac{x - m(t) x_0}{v(t)}. \tag{1.6}
$$

Inserting (1.6) into the DSM objective (1.5), the neural network $s_\theta$ is trained purely by predicting the analytic noise vector that was added to the data $x_0$ to produce $x$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(The Generative Procedure)</span></p>

Once the network is optimally trained such that $s_{\theta^\ast}(x, t) \approx \nabla_x \log p(t, x)$, the theoretical generative framework is complete. Synthesis proceeds via numerical integration of the reverse SDE. For the VP SDE, choose the terminal time $T$ large enough so that $m(T) \approx 0$ and $v(T) \approx 1$; then the terminal distribution $p(T, x)$ is well-approximated by the standard Gaussian prior $\mathcal{N}(0, I)$. The generative procedure is:

1. **Initialisation.** Draw a sample of pure white noise $X_T \sim \mathcal{N}(0, I)$.
2. **Reverse evolution.** Solve the approximated reverse SDE backwards from $t = T$ down to $t = 0$:

   $$
   \mathrm{d} X_t = \Bigl[-\tfrac{1}{2} \beta(t) X_t - \beta(t)\, s_{\theta^\ast}(X_t, t)\Bigr]\, \mathrm{d} t + \sqrt{\beta(t)}\, \mathrm{d} \bar{W}_t. \tag{1.7}
   $$

</div>

By integrating (1.7) numerically (e.g. via the Euler–Maruyama scheme), the path of $X_t$ is drawn deterministically towards the regions of high data density, eventually producing a sample $X_0$ that follows the original, unknown distribution $p_0$. The theory of stochastic processes therefore provides a rigorous, constructive mechanism for generative modelling.

<div style="margin: 1.2rem 0 1.5rem;">

<p style="margin: 0 0 0.4rem 0; font-style: italic; color: #5F5E5A;">Interactive: a 1D toy diffusion model. The data distribution is the bimodal mixture $p_0 = 0.4\,\mathcal{N}(-1.5,\,0.16) + 0.6\,\mathcal{N}(1.5,\,0.25)$. The forward VP SDE smears it into $\mathcal{N}(0,1)$ over $[0,T]$; the reverse SDE — driven by the <em>exact</em> score $\nabla_x \log p_t$ (closed-form for a Gaussian mixture under VP dynamics) — transports $\mathcal{N}(0,1)$ back into $p_0$. <strong>Top:</strong> forward trajectories $X_0 \to X_T$. <strong>Bottom:</strong> reverse trajectories $X_T \to X_0$. The orange curve is the analytic marginal $p_t$; both histograms should match it at every $t$. Press <strong>Play</strong> to ping-pong $t$ across $[0,T]$.</p>

<div style="margin: 1rem 0;">
  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">
    <label for="diff-t" style="font-size: 14px; color: var(--color-text-secondary, #5F5E5A); min-width: 60px;">time t</label>
    <input type="range" min="0" max="1" value="0" step="0.01" id="diff-t" style="flex: 1; min-width: 200px;" />
    <span id="diff-t-val" style="font-size: 14px; font-weight: 500; min-width: 56px; text-align: right;">0.00</span>
    <button id="diff-play" style="padding: 6px 14px;">&#9658; Play</button>
    <button id="diff-resample" style="padding: 6px 14px;">Resample</button>
  </div>

  <div>
    <div style="font-size: 13px; font-weight: 500; margin-bottom: 4px; color: #2c3e94;">Forward: data &rarr; noise</div>
    <div style="position: relative; height: 220px; margin-bottom: 12px;">
      <canvas id="diff-fwd" role="img" aria-label="Histogram of forward-diffused samples and analytic marginal density at the current time."></canvas>
    </div>
    <div style="font-size: 13px; font-weight: 500; margin-bottom: 4px; color: #2c3e94;">Reverse: noise &rarr; data (exact score)</div>
    <div style="position: relative; height: 220px;">
      <canvas id="diff-rev" role="img" aria-label="Histogram of reverse-diffused samples driven by the exact score, and analytic marginal density at the current time."></canvas>
    </div>
  </div>
</div>

<script>
(function () {
  const T = 1, Nsteps = 100, dt = T / Nsteps;
  const bmin = 0.1, bmax = 10;
  const Np = 600;

  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const muData = [-1.5, 1.5];
  const sigData = [0.4, 0.5];
  const wData = [0.4, 0.6];
  function sampleData() {
    const i = Math.random() < wData[0] ? 0 : 1;
    return muData[i] + sigData[i] * randn();
  }
  function beta(t) { return bmin + (bmax - bmin) * t; }
  function intBeta(t) { return bmin * t + 0.5 * (bmax - bmin) * t * t; }
  function mAt(t) { return Math.exp(-0.5 * intBeta(t)); }
  function vAt(t) { return 1 - Math.exp(-intBeta(t)); }
  function pAt(x, t) {
    const m = mAt(t), v = vAt(t);
    let p = 0;
    for (let i = 0; i < 2; i++) {
      const mu = m * muData[i];
      const s2 = m * m * sigData[i] * sigData[i] + v;
      p += wData[i] * Math.exp(-(x - mu) * (x - mu) / (2 * s2)) / Math.sqrt(2 * Math.PI * s2);
    }
    return p;
  }
  function scoreAt(x, t) {
    const m = mAt(t), v = vAt(t);
    let p = 0;
    const ps = new Array(2), mus = new Array(2), s2s = new Array(2);
    for (let i = 0; i < 2; i++) {
      mus[i] = m * muData[i];
      s2s[i] = m * m * sigData[i] * sigData[i] + v;
      ps[i] = Math.exp(-(x - mus[i]) * (x - mus[i]) / (2 * s2s[i])) / Math.sqrt(2 * Math.PI * s2s[i]);
      p += wData[i] * ps[i];
    }
    if (p === 0) return 0;
    let s = 0;
    for (let i = 0; i < 2; i++) {
      const r = wData[i] * ps[i] / p;
      s += r * (-(x - mus[i]) / s2s[i]);
    }
    return s;
  }

  let fwd = new Array(Np);
  let rev = new Array(Np);

  function precompute() {
    for (let pIdx = 0; pIdx < Np; pIdx++) {
      const f = new Float64Array(Nsteps + 1);
      const r = new Float64Array(Nsteps + 1);
      f[0] = sampleData();
      for (let k = 1; k <= Nsteps; k++) {
        const t = (k - 1) * dt;
        const b = beta(t);
        f[k] = f[k - 1] - 0.5 * b * f[k - 1] * dt + Math.sqrt(b * dt) * randn();
      }
      r[Nsteps] = randn();
      for (let k = Nsteps - 1; k >= 0; k--) {
        const t = (k + 1) * dt;
        const b = beta(t);
        const s = scoreAt(r[k + 1], t);
        r[k] = r[k + 1] + 0.5 * b * r[k + 1] * dt + b * s * dt + Math.sqrt(b * dt) * randn();
      }
      fwd[pIdx] = f;
      rev[pIdx] = r;
    }
  }
  precompute();

  const tEl = document.getElementById('diff-t');
  const tValEl = document.getElementById('diff-t-val');
  const playBtn = document.getElementById('diff-play');
  const resampleBtn = document.getElementById('diff-resample');

  const xMin = -4, xMax = 4;
  const Nx = 240;
  const xGrid = [];
  for (let i = 0; i <= Nx; i++) xGrid.push(xMin + (xMax - xMin) * i / Nx);

  let fwdChart = null, revChart = null;

  function buildHist(samples) {
    const bins = 50;
    const counts = new Array(bins).fill(0);
    for (let i = 0; i < samples.length; i++) {
      const v = samples[i];
      if (v >= xMin && v <= xMax) {
        const b = Math.min(bins - 1, Math.floor((v - xMin) / (xMax - xMin) * bins));
        counts[b]++;
      }
    }
    const binWidth = (xMax - xMin) / bins;
    const denom = samples.length * binWidth;
    const data = [{ x: xMin, y: 0 }];
    let peak = 0;
    for (let b = 0; b < bins; b++) {
      const left = xMin + b * binWidth, right = left + binWidth;
      const h = counts[b] / denom;
      data.push({ x: left, y: h });
      data.push({ x: right, y: h });
      if (h > peak) peak = h;
    }
    data.push({ x: xMax, y: 0 });
    return { data, peak };
  }

  function update() {
    const t = parseFloat(tEl.value);
    tValEl.textContent = t.toFixed(2);
    const k = Math.min(Nsteps, Math.max(0, Math.round(t / dt)));

    const fwdSamples = new Float64Array(Np);
    const revSamples = new Float64Array(Np);
    for (let i = 0; i < Np; i++) {
      fwdSamples[i] = fwd[i][k];
      revSamples[i] = rev[i][k];
    }
    const fHist = buildHist(fwdSamples);
    const rHist = buildHist(revSamples);

    const densData = xGrid.map((x) => ({ x, y: pAt(x, Math.max(t, 1e-4)) }));
    let densPeak = 0;
    for (const d of densData) if (d.y > densPeak) densPeak = d.y;

    const yMax = Math.max(densPeak, fHist.peak, rHist.peak) * 1.1 + 1e-9;

    if (!fwdChart) {
      fwdChart = new Chart(document.getElementById('diff-fwd'), {
        type: 'line',
        data: { datasets: [
          { label: 'fwd hist', data: fHist.data, borderColor: 'rgba(83,74,183,0.7)', backgroundColor: 'rgba(83,74,183,0.18)', borderWidth: 1, pointRadius: 0, fill: 'origin', tension: 0, order: 2 },
          { label: 'p_t', data: densData, borderColor: '#D97757', borderWidth: 1.6, pointRadius: 0, fill: false, tension: 0, order: 1 }
        ]},
        options: {
          animation: false, responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: {
            x: { type: 'linear', min: xMin, max: xMax, title: { display: true, text: 'x' } },
            y: { title: { display: true, text: 'density' }, beginAtZero: true, max: yMax }
          }
        }
      });
    } else {
      fwdChart.data.datasets[0].data = fHist.data;
      fwdChart.data.datasets[1].data = densData;
      fwdChart.options.scales.y.max = yMax;
      fwdChart.update('none');
    }

    if (!revChart) {
      revChart = new Chart(document.getElementById('diff-rev'), {
        type: 'line',
        data: { datasets: [
          { label: 'rev hist', data: rHist.data, borderColor: 'rgba(83,74,183,0.7)', backgroundColor: 'rgba(83,74,183,0.18)', borderWidth: 1, pointRadius: 0, fill: 'origin', tension: 0, order: 2 },
          { label: 'p_t', data: densData, borderColor: '#D97757', borderWidth: 1.6, pointRadius: 0, fill: false, tension: 0, order: 1 }
        ]},
        options: {
          animation: false, responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: {
            x: { type: 'linear', min: xMin, max: xMax, title: { display: true, text: 'x' } },
            y: { title: { display: true, text: 'density' }, beginAtZero: true, max: yMax }
          }
        }
      });
    } else {
      revChart.data.datasets[0].data = rHist.data;
      revChart.data.datasets[1].data = densData;
      revChart.options.scales.y.max = yMax;
      revChart.update('none');
    }
  }

  let playing = false;
  let dir = 1;
  let last = 0;
  function frame(now) {
    if (!playing) return;
    if (!last) last = now;
    const elapsed = (now - last) / 1000;
    last = now;
    let t = parseFloat(tEl.value) + dir * 0.4 * elapsed;
    if (t >= T) { t = T; dir = -1; }
    if (t <= 0) { t = 0; dir = 1; }
    tEl.value = t.toFixed(3);
    update();
    requestAnimationFrame(frame);
  }
  playBtn.addEventListener('click', () => {
    playing = !playing;
    playBtn.innerHTML = playing ? '&#10074;&#10074; Pause' : '&#9658; Play';
    last = 0;
    if (playing) requestAnimationFrame(frame);
  });

  tEl.addEventListener('input', update);
  resampleBtn.addEventListener('click', () => { precompute(); update(); });
  update();
})();
</script>

</div>

## 2 Brownian Motion and Transition Semigroups

Having established the pathwise and distributional properties of one-dimensional Brownian motion (Chapter 0), we now extend the framework to $\mathbb{R}^d$. As generative diffusion models operate by systematically perturbing high-dimensional data distributions via *multivariate* noise, we need to understand the macroscopic spatial evolution of such systems. To rigorously analyse this probability flow — and ultimately derive the score function required for time-reversal — we must identify the foundational analytical object governing the forward process: the **transition density**.

### 2.1 Multivariate Brownian Motion

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">2.1 (Multidimensional Standard Brownian Motion)</span></p>

Let $\mathbb{F} = (\mathcal{F}_t)\_{t \ge 0}$ be a filtration. A $d$-dimensional stochastic process $B := (B_t)\_{t \ge 0}$ taking values in $\mathbb{R}^d$ is called a **$d$-dimensional (standard) $\mathbb{F}$-Brownian motion** if it is of the form

$$
B_t = \bigl(B_t^{(1)}, B_t^{(2)}, \dots, B_t^{(d)}\bigr)^{\!\top} \quad \text{for all } t \ge 0,
$$

where the coordinate processes $B^{(1)}, \dots, B^{(d)}$ are independent one-dimensional standard $\mathbb{F}$-Brownian motions.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Multidimensional Standard Brownian Motion)</span></p>

* **Multivariate increments.** As an immediate consequence of component-wise independence, the spatial increments are independent of the past and multivariate normal: for any $s < t$,

  $$
  B_t - B_s \sim \mathcal{N}\bigl(0,\ (t-s)\, I_d\bigr),
  $$

  where $I_d$ denotes the $d \times d$ identity matrix.

* **Renewal property (splitting the path).** A crucial conceptual implication of stationary, independent increments is the ability to *split* the Brownian path. Fix any deterministic time $s \ge 0$ and define the post-$s$ process

  $$
  \widetilde{B}_t := B_{s+t} - B_s, \qquad t \ge 0.
  $$

  Then $\widetilde{B}$ is again a standard $d$-dimensional Brownian motion, entirely *independent of the history* $\mathcal{F}_s$. Informally, the process "starts afresh" at any given time $s$.

* **Why this matters for diffusion models.** This renewal feature is what allows the forward corruption process to be implemented as a chain of identically distributed noise injections — the basic algorithmic primitive behind training a score network.

</div>

<figure>
  <img src="{{ 'assets/images/notes/sdes_diffusion_models/bm2d_renewal.png' | relative_url }}" alt="Two panels. Left: a two-dimensional Brownian path colour-graded from light to dark blue by time, starting at a green dot at the origin and ending at a red dot, with a colour bar for time. Right: a one-dimensional Brownian path split at time s equals one by a dashed vertical line; the history is blue, the continuation grey, and the shifted process, drawn in green, restarts from the origin and looks like an entirely fresh Brownian motion." loading="lazy">
  <figcaption>Definition 2.1 in pictures. <strong>Left.</strong> A two-dimensional Brownian path $B_t = (B_t^{(1)}, B_t^{(2)})^{\top}$, colour-graded by time: two independent one-dimensional Brownian motions drive the coordinates. <strong>Right.</strong> The renewal property. Past the deterministic time $s$ (dashed line), the shifted process $\widetilde{B}_t = B_{s+t} - B_s$ (green) is exactly the grey continuation translated back to the origin — again a standard Brownian motion, independent of the history $\mathcal{F}_s$ (blue).</figcaption>
</figure>

To formalise this powerful renewal behaviour into the Markov property, we must introduce notation that "restarts" the process at potentially non-zero spatial locations. For any fixed $x \in \mathbb{R}^d$, we define the **shifted probability measure** $\mathbb{P}^x_B$ associated with the Brownian motion $B$ by

$$
\mathbb{P}^x_B(A) := \mathbb{P}(B + x \in A).
$$

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">($\mathbb{P}^x_B$ — Brownian motion started at $x$)</span></p>

* Under $\mathbb{P}^x_B$, the process behaves *exactly* like a standard Brownian motion that has been spatially shifted to start at $x \in \mathbb{R}^d$ instead of the origin.
* We denote the corresponding expectation operator by $\mathbb{E}_x$. When $x = 0$, we recover the standard measure: $\mathbb{P}_0 = \mathbb{P}$ and $\mathbb{E}_0 = \mathbb{E}$.
* This notation is the multivariate analogue of the "fresh-start" measures encountered in §0.5 (Definition 0.14, Interpretation of $\mathbf{P}^a_W$); it is the standard device for stating the strong Markov property in $\mathbb{R}^d$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">2.2 (Markov Property)</span></p>

Let $B$ be a $d$-dimensional standard $\mathbb{F}$-Brownian motion. Then, for any bounded measurable functional $\Phi : C([0,\infty), \mathbb{R}^d) \to \mathbb{R}$, it holds that

$$
\mathbb{E}\bigl[\Phi(B_{s+\cdot})\,\big|\, \mathcal{F}_s\bigr] = \int \Phi(w)\, \mathrm{d}\mathbb{P}^{B_s}_B(w), \qquad \mathbb{P}\text{-a.s., for all } s \ge 0.
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Markov Property in $\mathbb{R}^d$)</span></p>

* Conditionally on $\mathcal{F}_s$, the future path of the process behaves exactly like a fresh Brownian motion *originating from the random current state* $B_s$.
* For a functional depending only on a single future time $s + t$, this seamlessly simplifies to the familiar form

  $$
  \mathbb{E}[\,f(B_{s+t}) \mid \mathcal{F}_s\,] = \mathbb{E}_{B_s}[\,f(B_t)\,].
  $$

* This is a *measure-theoretic strengthening* of the elementary Markov property of §0.5 (Proposition 0.15): we now condition not just on a value, but on the entire history $\sigma$-algebra $\mathcal{F}_s$.

</div>

This property allows us to systematically construct the finite-dimensional distributions of the process by chaining together independent transitions. Below we demonstrate this rigorously for three points in time.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Three-Time Marginal via Iterated Conditioning)</span></p>

Let $0 \le s < t < u$ and let $\varphi, \psi, \chi : \mathbb{R}^d \to \mathbb{R}$ be bounded measurable test functions. Iteratively conditioning on the intermediate times and exploiting the Markov property yields

$$
\begin{aligned}
\mathbb{E}_0\bigl[\varphi(B_s)\psi(B_t)\chi(B_u)\bigr]
&= \mathbb{E}_0\bigl[\varphi(B_s)\psi(B_t)\, \mathbb{E}_0[\chi(B_u) \mid \mathcal{F}_t]\bigr] \\
&= \mathbb{E}_0\bigl[\varphi(B_s)\psi(B_t)\, \mathbb{E}_{B_t}[\chi(B_{u-t})]\bigr] \\
&= \mathbb{E}_0\Bigl[\varphi(B_s)\, \mathbb{E}_{B_s}\bigl[\psi(B_{t-s})\, \mathbb{E}_{B_{t-s}}[\chi(B_{u-t})]\bigr]\Bigr].
\end{aligned}
$$

Rewriting these nested expectations as spatial integrals gives the fundamental identity for the finite-dimensional distributions,

$$
\iiint_{\mathbb{R}^{3d}} \varphi(x)\psi(y)\chi(z)\, \mathbb{P}_0\bigl(B_s \in \mathrm{d}x,\, B_t \in \mathrm{d}y,\, B_u \in \mathrm{d}z\bigr)
$$

$$
= \int_{\mathbb{R}^d}\!\!\!\varphi(x)\!\left[\int_{\mathbb{R}^d}\!\!\!\psi(y)\!\left(\int_{\mathbb{R}^d}\!\!\!\chi(z)\, \mathbb{P}_y(B_{u-t} \in \mathrm{d}z)\right)\!\mathbb{P}_x(B_{t-s} \in \mathrm{d}y)\right]\!\mathbb{P}_0(B_s \in \mathrm{d}x).
$$

* The joint distribution over any sequence of times is *completely determined* by the isolated, pairwise transition probabilities $\mathbb{P}\_x(B_{\Delta t} \in \mathrm{d}y)$.
* In diffusion models, this is precisely why the entire forward trajectory factorises into a computationally tractable Markov chain.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">2.3 (Chapman–Kolmogorov Equations)</span></p>

For any $s, t > 0$, $x \in \mathbb{R}^d$, and Borel set $A \in \mathcal{B}(\mathbb{R}^d)$, the transition probabilities satisfy

$$
\mathbb{P}_x(B_{s+t} \in A) = \int_{\mathbb{R}^d} \mathbb{P}_z(B_t \in A)\, \mathbb{P}_x(B_s \in \mathrm{d}z).
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Consistency Across Time)</span></p>

The probability of moving from $x$ to a set $A$ over a time interval $s + t$ must account for *all* possible intermediate states $z$ at time $s$. Chapman–Kolmogorov is the mathematical statement that "splitting time" is consistent: marginalising over the bridge state $z$ at time $s$ recovers the direct transition over $s+t$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.3</summary>

Let $A \in \mathcal{B}(\mathbb{R}^d)$ and consider the indicator $\mathbf{1}_A$. Taking the expectation under $\mathbb{P}_x$ and conditioning on the intermediate time $s$, the tower property and the Markov property (Proposition 2.2) yield

$$
\begin{aligned}
\mathbb{P}_x(B_{s+t} \in A)
&= \mathbb{E}_x\bigl[\mathbf{1}_A(B_{s+t})\bigr]
= \mathbb{E}_x\bigl[\mathbb{E}_x[\mathbf{1}_A(B_{s+t}) \mid \mathcal{F}_s]\bigr] \\
&= \mathbb{E}_x\bigl[\mathbb{P}_{B_s}(B_t \in A)\bigr]
= \int_{\mathbb{R}^d} \mathbb{P}_z(B_t \in A)\, \mathbb{P}_x(B_s \in \mathrm{d}z). \qquad\square
\end{aligned}
$$

</details>
</div>

To formalise this transition mechanism analytically and make it computationally useful, we introduce the explicit transition density of the process.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">2.4 (Gaussian Transition Density)</span></p>

For any $t > 0$ and $x, y \in \mathbb{R}^d$, the **transition density** of the $d$-dimensional standard Brownian motion is

$$
p_t(x, y) := \frac{1}{(2\pi t)^{d/2}} \exp\!\left(-\frac{\lvert x - y \rvert^2}{2t}\right).
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Density Form of Chapman–Kolmogorov)</span></p>

* The function $p_t(x, y)$ is the Lebesgue density of $B_t$ under $\mathbb{P}_x$, so the transition measure is

  $$
  \mathbb{P}_x(B_t \in A) = \int_A p_t(x, y)\, \mathrm{d}y.
  $$

* Substituting this density representation into Theorem 2.3, we instantly obtain the algebraic counterpart of Chapman–Kolmogorov,

  $$
  p_{s+t}(x, y) = \int_{\mathbb{R}^d} p_s(x, z)\, p_t(z, y)\, \mathrm{d}z.
  $$

* Convolving two Gaussian densities yields another Gaussian whose variance is the sum of the variances — the familiar accumulation of variance under independent Brownian increments, reflecting the continuous spatial diffusion over time.

</div>

<figure>
  <img src="{{ 'assets/images/notes/sdes_diffusion_models/gaussian_transition_density.png' | relative_url }}" alt="Two panels. Left: four centred Gaussian curves in shades of blue, from a tall narrow peak at t equals 0.1 to a flat wide bell at t equals 3, each directly labelled with its time. Right: the densities for s equals 0.5 in blue and t equals 1 in green, the analytic density for s plus t as a thick dark curve, and orange circles from a numerical convolution landing exactly on that curve." loading="lazy">
  <figcaption>The Gaussian transition density of Definition 2.4. <strong>Left.</strong> $p_t(0, y)$ flattens and spreads as $t$ grows — the variance accumulates linearly in time. <strong>Right.</strong> Chapman–Kolmogorov in density form: numerically convolving $p_s$ with $p_t$ (here $s = 0.5$, $t = 1$; orange circles) lands exactly on the analytic $p_{s+t}$ — convolving Gaussians adds their variances.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Bridge</span><span class="math-callout__name">(From Pathwise to Operator-Theoretic Viewpoint)</span></p>

Thus far our focus has been *microscopic and pathwise*: the irregularities, infinite variation, and continuity of $t \mapsto B_t(\omega)$. To analyse the *macroscopic* evolution of the process, we shift perspective from classical probability to functional analysis. Because Brownian motion is memoryless, time-evolution behaves *algebraically like operator composition*. We therefore abstract away the noise and ask:

> If we start at position $x$, what is the expected value of an observable $u$ at time $t$?

By translating "time" into the application of a linear operator $P_t$ that maps $u$ to this new expected value, we establish an operator-theoretic approach. This perspective ultimately connects Brownian motion to the rigorous PDE theory underlying generative diffusion models.

</div>

### 2.2 The Transition Semigroup

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Brownian Transition Operator)</span></p>

To formalise this intuition, we associate a family of linear operators with a given Brownian motion. For $t \ge 0$, define the **transition operator** $P_t$ acting on a function $u$ by

$$
P_t u(x) := \mathbb{E}_x[u(B_t)], \qquad x \in \mathbb{R}^d. \tag{2.1}
$$

Throughout this chapter, we work under the standing assumption that $(B_t)\_{t \ge 0}$ is a $d$-dimensional Brownian motion adapted to a *right-continuous* filtration $(\mathcal{F}\_t)\_{t \ge 0}$, meaning $\mathcal{F}\_t = \mathcal{F}\_{t+}$ for all $t \ge 0$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Operator Semigroup)</span></p>

In functional analysis, an **operator semigroup** $(P_t)\_{t \ge 0}$ on a Banach space $(\mathbf{B}, \lVert \cdot \rVert)$ is a family of linear operators $P_t : \mathbf{B} \to \mathbf{B}$, $t \ge 0$, satisfying the two algebraic conditions

$$
P_t P_s = P_{t+s} \quad \text{and} \quad P_0 = \mathrm{id}. \tag{2.2}
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Convention</span><span class="math-callout__name">(Used Stochastic Processes)</span></p>

In the context of stochastic processes we work predominantly with two specific Banach spaces:

* $\mathcal{B}\_b(\mathbb{R}^d)$, the space of bounded Borel measurable functions $f : \mathbb{R}^d \to \mathbb{R}$ equipped with the uniform norm $\lVert f \rVert_\infty$;
* $C_\infty(\mathbb{R}^d)$, the space of continuous functions vanishing at infinity, similarly equipped with the uniform norm.

Unless otherwise stated, we abbreviate $\mathcal{B}\_b = \mathcal{B}\_b(\mathbb{R}^d)$ and $C_\infty = C_\infty(\mathbb{R}^d)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Misconception about $C_\infty(\mathbb{R}^d)$)</span></p>

In the context of Feller processes, $C_\infty(\mathbb{R}^d)$ does *not* denote the space of infinitely differentiable (smooth) functions—which is usually written with the infinity symbol as a superscript, $C^\infty(\mathbb{R}^d)$. Instead, the subscript infinity in $C_\infty(\mathbb{R}^d)$ denotes the space of **continuous functions that vanish at infinity**.

For a function $f$ to belong to $C_\infty(\mathbb{R}^d)$, it must satisfy two distinct properties:

* **Continuity:** The function is continuous at every point $x \in \mathbb{R}^d$.
* **Vanishing at infinity:** As you move infinitely far away from the origin in any direction, the function's value goes to zero. Mathematically:

$$\lim_{|x| \to \infty} f(x) = 0$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">2.5 (Brownian Transition Operators Form a Semigroup)</span></p>

Let $(B_t)\_{t \ge 0}$ be a $d$-dimensional Brownian motion. Then the family $(P_t)\_{t \ge 0}$ defined in (2.1) is a semigroup of operators on $\mathcal{B}\_b(\mathbb{R}^d)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The linearity of $P_t$ follows from linearity of expectation. Expressing the expectation via the explicit Gaussian density,

$$
P_t u(x) = \int_{\mathbb{R}^d} u(y)\, \mathbb{P}_x(B_t \in \mathrm{d}y) = \frac{1}{(2\pi t)^{d/2}} \int_{\mathbb{R}^d} u(y)\, e^{-\lvert x - y \rvert^2 / (2t)}\, \mathrm{d}y.
$$

This integral representation is a convolution with a smooth Gaussian kernel, and so $P_t u$ is again bounded and Borel measurable: $P_t : \mathcal{B}_b \to \mathcal{B}_b$. The semigroup property (2.2) is a direct implication of the Markov property: for $s, t \ge 0$ and $u \in \mathcal{B}\_b(\mathbb{R}^d)$, the tower property of conditional expectation gives

$$
\begin{aligned}
P_{t+s} u(x)
&= \mathbb{E}_x[u(B_{t+s})]
= \mathbb{E}_x\bigl[\mathbb{E}_x[u(B_{t+s}) \mid \mathcal{F}_s]\bigr] \\
&= \mathbb{E}_x[\mathbb{E}_{B_s}[u(B_t)]]
= \mathbb{E}_x[P_t u(B_s)]
= P_s(P_t u)(x).
\end{aligned}
$$

Thus $P_{t+s} = P_s P_t$, completing the proof. $\square$

</details>
</div>

To further characterise the semigroup, we require a basic continuity property of the underlying Brownian paths.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">2.6 (Uniform Stochastic Continuity)</span></p>

A $d$-dimensional Brownian motion $(B_t)\_{t \ge 0}$ is **uniformly stochastically continuous**: for every $\delta > 0$,

$$
\lim_{t \to 0} \sup_{x \in \mathbb{R}^d} \mathbb{P}_x\bigl(\lvert B_t - x \rvert > \delta\bigr) = 0.
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By spatial translation invariance and Chebyshev's inequality,

$$
\mathbb{P}_x(\lvert B_t - x \rvert > \delta)
= \mathbb{P}_0(\lvert B_t \rvert > \delta)
\le \frac{\mathbb{E}_0[\lvert B_t \rvert^2]}{\delta^2}
= \frac{d \cdot t}{\delta^2}.
$$

Sending $t \to 0$ yields the claim. $\square$

</details>
</div>

We are now in a position to investigate the functional-analytic properties of the Brownian transition semigroup.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">2.7 (Analytical Properties of the Brownian Semigroup)</span></p>

Let $(P_t)\_{t \ge 0}$ be the transition semigroup of a $d$-dimensional Brownian motion. Then, for all $t > 0$:

a) $P_t \mathbf{1} = \mathbf{1}$ (**conservativity**);

b) $\lVert P_t u \rVert_\infty \le \lVert u \rVert_\infty$ (**contraction** on $\mathcal{B}_b$);

c) $u \ge 0 \implies P_t u \ge 0$ (**positivity preserving**);

d) $0 \le u \le 1 \implies 0 \le P_t u \le 1$ (**sub-Markovian**);

e) $u \in C_\infty \implies P_t u \in C_\infty$ (**Feller property**);

f) $\lim_{t \to 0} \lVert P_t u - u \rVert_\infty = 0$ for all $u \in C_\infty$ (**strong continuity** on $C_\infty$);

g) $u \in \mathcal{B}_b \implies P_t u \in C_b(\mathbb{R}^d)$ (**strong Feller property**).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">2.8 (Semigroup Classification)</span></p>

Let $(P_t)\_{t \ge 0}$ be a general operator semigroup. Depending on which properties from Proposition 2.7 it satisfies, it is classified as follows:

| Semigroup classification | Required properties |
| --- | --- |
| **Markov semigroup** | a), b), c), d) |
| **sub-Markov semigroup** | b), c), d) |
| **Feller semigroup** | b), c), d), e), f) |
| **strong Feller semigroup** | b), c), d), g) |

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Probabilistic and Analytical Significance)</span></p>

Each of these classifications carries a probabilistic and analytical meaning that determines the macroscopic behaviour of the underlying process:

* **Markov vs. sub-Markov.** A *Markov* semigroup is **conservative** ($P_t \mathbf{1} = \mathbf{1}$): probability mass is preserved over time and the process never "dies" or escapes the state space. A *sub-Markov* semigroup drops conservativity ($P_t \mathbf{1} \le \mathbf{1}$), which models processes that can be absorbed, killed, or pushed to infinity in finite time — the "missing" mass represents the likelihood that the process has terminated.

* **Feller semigroup.** This is the foundational class for rigorously constructing well-behaved Markov processes. The Feller property (e) and strong continuity (f) ensure that *continuous* initial conditions yield continuous spatial evolutions vanishing at infinity. By the Hille–Yosida theorem, any Feller semigroup is uniquely generated by a closed, densely defined operator (the **infinitesimal generator**, §2.4), which guarantees the existence of a corresponding strong Markov process with right-continuous paths possessing left limits (càdlàg).

* **Strong Feller semigroup.** The strong Feller property is a *very powerful regularisation effect*: even if we begin with a highly discontinuous, bounded measurable observable $u \in \mathcal{B}_b$, applying the semigroup $P_t$ *instantly* (for any $t > 0$) smooths it into a continuous function. For Brownian motion, this instantaneous smoothing is driven directly by the convolution with the Gaussian transition density.

* **Connection to diffusion models.** The strong Feller property is mathematically central for generative diffusion. When high-dimensional data (such as a manifold of natural images) is corrupted by forward Brownian motion, the highly singular initial data distribution is *instantly smoothed into a continuous, strictly positive density*. This guaranteed regularisation is a prerequisite for ensuring that the score field $\nabla_x \log p_t(x)$ is well-defined and differentiable everywhere in the state space for $t > 0$.

</div>

<figure>
  <img src="{{ 'assets/images/notes/sdes_diffusion_models/strong_feller_smoothing.png' | relative_url }}" alt="Two panels. Left: a discontinuous black step function with two blocks of heights one and 0.6, overlaid with three blue curves of increasing smoothness for t equal to 0.02, 0.2 and 1; even the smallest time gives a continuous curve, and the curves flatten below the maximum of u as t grows. Right: the same step function faint in grey, the function P one u as a blue curve, and orange circles for applying P one-half twice, matching the blue curve exactly." loading="lazy">
  <figcaption>Proposition 2.7 in action, computed by exact Gaussian quadrature. <strong>Left.</strong> A discontinuous observable $u \in \mathcal{B}_b$ (two blocks) becomes continuous <em>instantly</em> — already at $t = 0.02$ — illustrating the strong Feller property (g); no curve ever exceeds $\lVert u \rVert_\infty = 1$, illustrating contraction (b). <strong>Right.</strong> The semigroup identity of Lemma 2.5: applying $P_{1/2}$ twice (orange circles) reproduces $P_1 u$ (blue) exactly.</figcaption>
</figure>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 2.7</summary>

We address each property sequentially.

**a)** For conservativity, $P_t \mathbf{1}(x) = \mathbb{E}\_x[\mathbf{1}\lbrace B_t \in \mathbb{R}^d\rbrace] = \mathbb{P}\_x(B_t \in \mathbb{R}^d) = 1$, so probability mass is not lost.

**b)** $\lvert P_t u(x) \rvert \le \mathbb{E}\_x[\lvert u(B_t) \rvert] \le \lVert u \rVert_\infty \mathbb{E}\_x[1] = \lVert u \rVert_\infty$.

**c)** If $u \ge 0$, then $P_t u(x) = \mathbb{E}\_x[u(B_t)] \ge 0$ trivially.

**d)** Combining (b) and (c).

**e)** Write $P_t u(x) = \mathbb{E}\_x[u(B_t)] = \mathbb{E}\_0[u(B_t + x)]$. Since $\lvert u(B_t + x) \rvert \le \lVert u \rVert_\infty$, dominated convergence applies. As $x \to y$, continuity of $u$ gives $\lim_{x \to y} \mathbb{E}\_0[u(B_t + x)] = \mathbb{E}\_0[u(B_t + y)]$. Similarly, as $\lvert x \rvert \to \infty$, $u(B_t + x) \to 0$ almost surely, yielding $\lim_{\lvert x \rvert \to \infty} P_t u(x) = 0$. Thus $P_t u \in C_\infty$.

**f)** Fix $\varepsilon > 0$. Since $u \in C_\infty$ is uniformly continuous, there exists $\delta > 0$ such that $\lvert u(x) - u(y) \rvert < \varepsilon$ whenever $\lvert x - y \rvert < \delta$. Splitting the expectation,

$$
\begin{aligned}
\lVert P_t u - u \rVert_\infty
&\le \sup_{x} \mathbb{E}_x[\lvert u(B_t) - u(x) \rvert] \\
&\le \sup_{x}\! \left(\int_{\lbrace \lvert B_t - x \rvert < \delta \rbrace}\!\lvert u(B_t) - u(x) \rvert\, \mathrm{d}\mathbb{P}_x + \int_{\lbrace \lvert B_t - x \rvert \ge \delta \rbrace}\!\lvert u(B_t) - u(x) \rvert\, \mathrm{d}\mathbb{P}_x \right) \\
&\le \varepsilon + 2\lVert u \rVert_\infty \sup_{x} \mathbb{P}_x(\lvert B_t - x \rvert \ge \delta).
\end{aligned}
$$

By Lemma 2.6 the second term vanishes as $t \to 0$. Since $\varepsilon$ was arbitrary, strong continuity holds.

**g)** Let $R > 0$ and consider $x, z \in B(0, R)$. For $u \in \mathcal{B}\_b$, $P_t u$ is the convolution with the Gaussian density. For $\lvert y \rvert \ge 2R$ we have the geometric bound $\lvert z - y \rvert^2 \ge (\lvert y \rvert - \lvert z \rvert)^2 \ge \tfrac{1}{4} \lvert y \rvert^2$. Thus the integrand is dominated by

$$
\lvert u(y) \rvert\, e^{-\lvert z - y \rvert^2 / (2t)}
\le \lVert u \rVert_\infty \!\left(\mathbf{1}_{B(0,2R)}(y) + e^{-\lvert y \rvert^2 / (8t)} \mathbf{1}_{B(0,2R)^c}(y)\right).
$$

This dominating function is independent of $z$ and integrable, so dominated convergence pulls the limit $z \to x$ inside the integral, proving $P_t u \in C_b(\mathbb{R}^d)$. $\square$

</details>
</div>

The semigroup notation is not merely abstract; it offers an algebraic framework to express the finite-dimensional distributions of a Markov process. For $s < t$ and observables $f, g \in \mathcal{B}_b(\mathbb{R}^d)$, iterated tower property gives

$$
\mathbb{E}_x\bigl[f(B_s) g(B_t)\bigr]
= \mathbb{E}_x\bigl[f(B_s) \mathbb{E}_{B_s}[g(B_{t-s})]\bigr]
= \mathbb{E}_x\bigl[f(B_s) (P_{t-s} g)(B_s)\bigr]
= P_s\bigl[\,f \cdot P_{t-s} g\,\bigr](x).
$$

Iterating over an arbitrary number of time steps yields the following theorem, which fully characterises the finite-dimensional distributions of Brownian motion in *purely operator-theoretic terms*.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">2.9 (Finite-Dimensional Distributions via the Semigroup)</span></p>

Let $(B_t)\_{t \ge 0}$ be a $d$-dimensional Brownian motion evaluated at discrete times $0 = t_0 < t_1 < \dots < t_n$. For Borel sets $C_1, \dots, C_n \in \mathcal{B}(\mathbb{R}^d)$ and starting state $x \in \mathbb{R}^d$,

$$
\mathbb{P}_x\bigl(B_{t_1} \in C_1, \dots, B_{t_n} \in C_n\bigr)
= P_{t_1}\!\Bigl[\mathbf{1}_{C_1} P_{t_2 - t_1}\!\bigl[\dots P_{t_n - t_{n-1}}[\mathbf{1}_{C_n}]\bigr]\Bigr](x).
$$

</div>

### 2.3 From Semigroups to Processes

In §2.2 we observed how a given Brownian motion $(B_t)\_{t \ge 0}$ naturally induces a transition semigroup $(P_t)\_{t \ge 0}$. We now address the inverse problem:

> Given a family of operators satisfying certain analytical properties, can we construct a corresponding Markov process?

This transition from analytical description to stochastic process is facilitated by the concept of a **transition function** (or probability kernel): the probability of moving from a state $x$ to a set $A$ in time $t$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">2.10 (Transition Function)</span></p>

A function $p_t(x, A)$ with $t > 0$, $x \in \mathbb{R}^d$, $A \in \mathcal{B}(\mathbb{R}^d)$, is called a **transition function** if:

(i) For fixed $t$ and $x$, the map $A \mapsto p_t(x, A)$ is a probability measure on $\mathcal{B}(\mathbb{R}^d)$.

(ii) For fixed $t$ and $A$, the map $x \mapsto p_t(x, A)$ is a Borel measurable function.

(iii) The family satisfies the **Chapman–Kolmogorov equations**:

$$
p_{s+t}(x, A) = \int_{\mathbb{R}^d} p_s(y, A)\, p_t(x, \mathrm{d}y) \quad \text{for all } s, t > 0. \tag{2.3}
$$

</div>

<figure>
  <img src="{{ 'assets/images/notes/sdes_diffusion_models/KolmogorovChapmanConsistency.png' | relative_url }}" alt="a" loading="lazy">
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Consistency and the Diffusion-Model Reading)</span></p>

* **Consistency.** Property (iii) is a *consistency* condition: a direct jump from $x$ to $A$ in time $s + t$ is the same as jumping to some intermediate point $y$ in time $t$ and then from $y$ to $A$ in time $s$, integrated over all possible intermediate points $y$.
* **In diffusion models.** $p_t(x_0, A)$ corresponds to the forward diffusion kernel $q(x_t \mid x_0)$, and the Chapman–Kolmogorov property is precisely what allows us to define the diffusion in a chain-like manner ($x_0 \to x_s \to x_{s+t}$).

</div>

**Existence of the process.** The existence of a Markov process $(X_t)\_{t \ge 0}$ for a given transition function is guaranteed by Kolmogorov's extension theorem. However, a general such process may exhibit very irregular sample paths. To ensure path regularity, we focus on a specifically well-behaved class.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">2.11 (Feller Process)</span></p>

A Markov process $(X_t)\_{t \ge 0}$ is called a **Feller process** if its associated transition semigroup $(P_t)\_{t \ge 0}$, defined by $P_t u(x) := \int u(y)\, p_t(x, \mathrm{d}y)$, is a Feller semigroup on $C_\infty(\mathbb{R}^d)$, that is,

(i) $P_t : C_\infty \to C_\infty$ for all $t \ge 0$ (Feller property);

(ii) $\lim_{t \to 0} \lVert P_t u - u \rVert_\infty = 0$ for all $u \in C_\infty$ (strong continuity).

</div>

One of the most profound results in the theory of Markov processes connects the analytical Feller properties to the topological properties — i.e. the "look" — of the random paths.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">2.12 (Path Properties of Feller Processes)</span></p>

Let $(P_t)\_{t \ge 0}$ be a Feller semigroup. Then there exists a Markov process $(X_t)\_{t \ge 0}$ such that

1. for every $x \in \mathbb{R}^d$, $X_0 = x$ almost surely under $\mathbb{P}_x$;
2. the process $(X_t)\_{t \ge 0}$ possesses a **càdlàg** version, i.e., for almost all $\omega$, the map $t \mapsto X_t(\omega)$ is right-continuous and possesses left limits for all $t \ge 0$;
3. $(X_t)\_{t \ge 0}$ satisfies the **strong Markov property** with respect to the right-continuous filtration $(\mathcal{F}_{t+})\_{t \ge 0}$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Why this Theorem is so Central)</span></p>

* **BM is the prototype.** As proved in Proposition 2.7, the Brownian semigroup satisfies the Feller properties. Brownian motion already has continuous paths, but Feller theory provides a *more general framework* that includes processes with jumps (such as Lévy processes), provided they satisfy the càdlàg property.
* **In denoising diffusion models.** The Feller property ensures that if we start with a continuous data distribution, the diffused versions remain continuous and analytically manageable. This regularity is exactly *why* a reverse-time score function $\nabla \log p_t(x)$ can be learned reliably.

</div>

<figure>
  <img src="{{ 'assets/images/notes/sdes_diffusion_models/cadlag_feller_path.png' | relative_url }}" alt="Two panels. Left: a continuous, jagged Brownian path in blue over the time interval zero to three. Right: a green piecewise-linear path with four jumps; at each jump time an open circle marks the left limit and a filled circle marks the value the path actually takes, with dotted vertical connectors and grey annotations reading right-continuous, filled dot, and left limit exists, open dot." loading="lazy">
  <figcaption>Theorem 2.12: Feller theory delivers a càdlàg version. <strong>Left.</strong> Brownian motion, the continuous prototype of a Feller process. <strong>Right.</strong> A compound-Poisson-with-drift path — also Feller — shows exactly what càdlàg permits: at each jump the path is right-continuous (filled dot) while the left limit still exists (open dot). Jumps are allowed; wild oscillation is not.</figcaption>
</figure>

### 2.4 The Generator

We begin with a simple observation from deterministic calculus. Let $\varphi : [0, \infty) \to \mathbb{R}$ be continuous and satisfy the functional equation $\varphi(t) \varphi(s) = \varphi(t + s)$ with $\varphi(0) = 1$. Then there exists a unique $a \in \mathbb{R}$ with $\varphi(t) = e^{a t}$, and we recover this parameter via the derivative at zero,

$$
a = \left.\frac{\mathrm{d}}{\mathrm{d}t}\varphi(t)\right|_{t=0} = \lim_{t \to 0} \frac{\varphi(t) - 1}{t}.
$$

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Heuristic for the Generator)</span></p>

Since a strongly continuous semigroup $(P_t)\_{t \ge 0}$ on a Banach space satisfies the *exact same* algebraic functional equation $P_t P_s u = P_{t+s} u$, it is mathematically reasonable to guess that, in an appropriate sense, $P_t u = e^{tA} u$ for some operator $A$ acting on $\mathbf{B}$. This operator $A$ acts as the *"derivative" of the semigroup at $t = 0$*.

</div>

To keep things rigorous, we restrict attention to Feller semigroups.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">2.13 (Feller Generator)</span></p>

Let $(P_t)\_{t \ge 0}$ be a Feller semigroup on $C_\infty(\mathbb{R}^d)$. The **(infinitesimal) generator** $A$ of $(P_t)\_{t \ge 0}$ is

$$
A u := \lim_{t \to 0} \frac{P_t u - u}{t}, \qquad \text{(limit in } (C_\infty(\mathbb{R}^d), \lVert \cdot \rVert_\infty)), \tag{2.4}
$$

with explicit domain

$$
\mathcal{D}(A) := \left\lbrace u \in C_\infty(\mathbb{R}^d) \;\middle|\; \exists\, g \in C_\infty(\mathbb{R}^d): \lim_{t \to 0} \left\lVert \tfrac{P_t u - u}{t} - g \right\rVert_\infty = 0 \right\rbrace. \tag{2.5}
$$

This defines a linear operator $A : \mathcal{D}(A) \to C_\infty(\mathbb{R}^d)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name">(What the Generator Encodes)</span></p>

If $P_t$ tells us *where* the process expects to be in the future, then $A$ evaluates *in which direction and how fast* the expectation is moving exactly at the present moment $t = 0$. The semigroup is the "global flow"; the generator is the "local vector field" that drives it.

</div>

For Brownian motion, this abstract generator reduces to a simple differential operator. The following example (and its proof) is central to all that follows.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">2.14 (Generator of Brownian Motion)</span></p>

Let $(B_t)\_{t \ge 0}$ with $B_t = (B_t^{(1)}, \dots, B_t^{(d)})^{\top}$ be a $d$-dimensional Brownian motion, and let $P_t u(x) := \mathbb{E}\_x[u(B_t)]$ be the associated transition semigroup. Let $(A, \mathcal{D}(A))$ denote its generator. Then $C_\infty^2(\mathbb{R}^d) \subset \mathcal{D}(A)$, and the operator acts as

$$
A\bigr|_{C_\infty^2(\mathbb{R}^d)} = \tfrac{1}{2} \Delta = \tfrac{1}{2} \sum_{j=1}^d \partial_j^2,
$$

where $\partial_j := \partial / \partial x_j$ and the function space is

$$
C_\infty^2(\mathbb{R}^d) := \bigl\lbrace u \in C_\infty(\mathbb{R}^d) \,\big|\, \partial_j u, \partial_j \partial_k u \in C_\infty(\mathbb{R}^d) \text{ for all } j, k = 1, \dots, d \bigr\rbrace.
$$

In fact, $\mathcal{D}(A) = C_\infty^2(\mathbb{R})$ and $A = \tfrac{1}{2}\Delta$ if $d = 1$. For $d > 1$, $\mathcal{D}(A) \subsetneq C_\infty^2(\mathbb{R}^d)$, and in general

$$
\mathcal{D}(A) = \lbrace u \in L^2(\mathbb{R}^d) \,\mid\, u, \Delta u \in C_\infty(\mathbb{R}^d) \rbrace, \tag{2.6}
$$

where $\Delta u$ is understood in the sense of distributions.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $u \in C_\infty^2(\mathbb{R}^d)$. By Taylor's formula, for some random intermediate point $\xi_t = \xi(x, t, \omega) \in [x, x + B_t(\omega)]$ we have

$$
\begin{aligned}
\left\lvert \frac{\mathbb{E}[u(B_t + x) - u(x)]}{t} - \tfrac{1}{2} \Delta u(x) \right\rvert
&= \left\lvert \mathbb{E}\!\left[\sum_{j=1}^d \partial_j u(x) \frac{B_t^{(j)}}{t} + \frac{1}{2t} \sum_{j,k=1}^d \bigl(\partial_j \partial_k u(\xi_t) - \partial_j \partial_k u(x)\bigr) B_t^{(j)} B_t^{(k)}\right] \right\rvert \\
&= \left\lvert \frac{1}{2t} \sum_{j,k=1}^d \mathbb{E}\!\left[\bigl(\partial_j \partial_k u(\xi_t) - \partial_j \partial_k u(x)\bigr) B_t^{(j)} B_t^{(k)}\right] \right\rvert.
\end{aligned}
$$

The last equality uses the moments of Brownian motion: the linear term vanishes because increments are centred ($\mathbb{E}[B_t^{(j)}] = 0$), and the subtracted Laplacian cancels with the second-order expectation since $\mathbb{E}[B_t^{(j)} B_t^{(k)}] = t\, \delta_{jk}$. From the integral form of the Taylor remainder,

$$
\tfrac{1}{2} \partial_j \partial_k u(\xi_t) = \int_0^1 \partial_j \partial_k u(x + r B_t)(1 - r)\, \mathrm{d}r,
$$

we see that $\omega \mapsto \partial_j \partial_k u(\xi_t(\omega))$ is measurable. Separating the spatial variations from the noise via Cauchy–Schwarz (twice — first to the sum, then to the expectation),

$$
\begin{aligned}
\left\lvert \frac{1}{2t} \sum_{j,k=1}^d \mathbb{E}\!\left[\bigl(\partial_j \partial_k u(\xi_t) - \partial_j \partial_k u(x)\bigr) B_t^{(j)} B_t^{(k)}\right] \right\rvert
&\le \frac{1}{2t} \mathbb{E}\!\left[\Bigl(\sum_{j,k=1}^d \lvert \partial_j \partial_k u(\xi_t) - \partial_j \partial_k u(x) \rvert^2\Bigr)^{\!1/2} \Bigl(\sum_{j,k=1}^d (B_t^{(j)} B_t^{(k)})^2\Bigr)^{\!1/2}\right] \\
&\le \frac{1}{2t} \Bigl(\mathbb{E}\!\left[\sum_{j,k=1}^d \lvert \partial_j \partial_k u(\xi_t) - \partial_j \partial_k u(x) \rvert^2\right]\Bigr)^{\!1/2} \bigl(\mathbb{E}[\lvert B_t \rvert^4]\bigr)^{1/2}.
\end{aligned}
$$

Now invoke Brownian scaling, $\mathbb{E}[\lvert B_t \rvert^4] = t^2\, \mathbb{E}[\lvert B_1 \rvert^4] < \infty$. Furthermore, the spatial differences are globally bounded by 

$$\lvert \partial_j \partial_k u(\xi_t) - \partial_j \partial_k u(x) \rvert^2 \le 4 \lVert \partial_j \partial_k u \rVert_\infty^2$$

Since $u \in C_\infty^2$, second derivatives are uniformly continuous; as $t \to 0$, $B_t \to 0$ a.s., causing $\xi_t \to x$. By dominated convergence,

$$
\limsup_{t \to 0} \sup_{x \in \mathbb{R}^d} \left\lvert \frac{\mathbb{E}[u(B_t + x) - u(x)]}{t} - \tfrac{1}{2} \sum_{j=1}^d \partial_j^2 u(x) \right\rvert = 0.
$$

This shows $C_\infty^2(\mathbb{R}^d) \subset \mathcal{D}(A)$ and $A = \tfrac{1}{2}\Delta$ on $C_\infty^2(\mathbb{R}^d)$. $\square$

</details>
</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Connection</span><span class="math-callout__name">(Brownian Noise = Laplacian Smoothing)</span></p>

The proof above highlights a profound mechanism: *pure, mean-zero Gaussian noise injected over time locally acts exactly like the Laplacian operator $\tfrac{1}{2}\Delta$*.

* In score-based diffusion models, this is the very reason adding Brownian noise inherently smoothens the data manifold *according to the heat equation* $\partial_t p = \tfrac{1}{2} \Delta p$.
* The Laplacian measures *local curvature*: where probability mass forms a sharp peak ($\Delta p < 0$), the generator diffuses it outwards; where it forms a valley ($\Delta p > 0$), probability mass flows in.
* This deterministic, geometric smoothing is precisely what makes the score function $\nabla_x \log p_t(x)$ tractable and computable.

</div>

<figure>
  <img src="{{ 'assets/images/notes/sdes_diffusion_models/generator_difference_quotient.png' | relative_url }}" alt="Two panels. Left: a black double-bump function u, three blue difference-quotient curves for t equal to 1, 0.3 and 0.05 that progressively approach a dashed red curve showing one half of the second derivative of u. Right: two curves of P t u evaluated at a fixed point against time; the blue curve, started at a peak, decreases, and the orange curve, started at a valley, increases, each with a dashed tangent line at time zero labelled with its slope, minus 1.38 for the peak and plus 0.71 for the valley." loading="lazy">
  <figcaption>Example 2.14: the generator of Brownian motion is $\frac12\Delta$. <strong>Left.</strong> For $u$ a sum of two Gaussian bumps (black), the difference quotients $(P_t u - u)/t$ converge uniformly to $\frac12\Delta u$ (dashed red) as $t \to 0$; here $P_t u$ is the exact heat flow, so no simulation error enters. <strong>Right.</strong> The generator is the derivative at $t = 0$ of $t \mapsto P_t u(x_0)$: at a peak $\Delta u(x_0) < 0$ and the expected observable initially falls, at a valley $\Delta u(x_0) > 0$ and it rises — the dashed tangents have slope $\frac12\Delta u(x_0)$, exactly the peak-flattening / valley-filling mechanism described above.</figcaption>
</figure>

A straightforward extension of the previous computation handles affine transformations of Brownian motion.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">2.15 (Generator of Brownian Motion with Drift)</span></p>

Let $X_t = C B_t + b t$, where $C \in \mathbb{R}^{d \times d}$ is a constant matrix and $b \in \mathbb{R}^d$ a constant vector. Then the generator $A$ associated with $(X_t)\_{t \ge 0}$ acts on $u \in C_\infty^2(\mathbb{R}^d)$ as

$$
A u(x) = b \cdot \nabla u(x) + \tfrac{1}{2}\, \mathrm{Tr}\bigl(C C^{\!\top} D^2 u(x)\bigr),
$$

or in coordinate notation,

$$
A u(x) = \sum_{i=1}^d b_i \partial_i u(x) + \tfrac{1}{2} \sum_{i,j=1}^d (C C^{\!\top})\_{ij}\, \partial_i \partial_j u(x).
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Connection</span><span class="math-callout__name">(SDE Generator → Diffusion Models)</span></p>

This example is of paramount importance for modern generative AI:

* The process $X_t = C B_t + b t$ represents a Brownian motion with constant **drift $b$** and **diffusion matrix $C$**.
* In score-based generative modelling, the forward corruption process is formulated precisely as an SDE of the form $\mathrm{d}x = f(x, t)\, \mathrm{d}t + g(t)\, \mathrm{d}B_t$.
* The generator explicitly reveals how the deterministic drift $(b \cdot \nabla u)$ *translates probability mass*, while the diffusion term $\tfrac{1}{2}\,\mathrm{Tr}(C C^{\top} D^2 u)$ *smooths it out*.

</div>

<figure>
  <img src="{{ 'assets/images/notes/sdes_diffusion_models/drift_diffusion_generator.png' | relative_url }}" alt="A single two-dimensional plot. Thirty-five thin grey sample paths fan out diagonally from a green dot at the origin. Three blue one-sigma ellipses, light to dark for times 0.5, 1 and 2, are tilted along the diagonal and grow while their centres move along a dashed red drift arrow; a red annotation reads drift b t translates the mean." loading="lazy">
  <figcaption>Example 2.15 for $b = (1.2,\, 0.5)^{\top}$ and a non-diagonal $C$. Sample paths of $X_t = C B_t + b\,t$ (grey) with the $1\sigma$ ellipses of the exact law $\mathcal{N}(b\,t,\; t\,CC^{\top})$ at $t = 0.5, 1, 2$: the drift part $b \cdot \nabla$ translates the mean along $b$ (red arrow), while the diffusion part $\frac12\,\mathrm{Tr}(CC^{\top} D^2)$ spreads mass anisotropically along the eigendirections of $CC^{\top}$ — the tilt of the ellipses.</figcaption>
</figure>

Since every Feller semigroup $(P_t)\_{t \ge 0}$ is given by a family of measurable kernels, we can define linear operators of the form

$$
L = \int_0^t P_s\, \mathrm{d}s
$$

acting on $u \in \mathcal{B}_b(\mathbb{R}^d)$ via

$$
L u(x) := \int_0^t \!\!\left(\int u(y)\, p_s(x, \mathrm{d}y)\right)\!\mathrm{d}s.
$$

Integrating the semigroup over time acts as a profound *"smoothing"* operation. As we will see, this integral operator allows us to take non-smooth functions and map them directly into the domain $\mathcal{D}(A)$ of the generator.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">2.16 (Generator–Semigroup Relationship)</span></p>

Let $(P_t)\_{t \ge 0}$ be a Feller semigroup with generator $(A, \mathcal{D}(A))$.

a) For all $u \in \mathcal{D}(A)$ and $t > 0$, $P_t u \in \mathcal{D}(A)$. Moreover, the operator and the semigroup *commute*:

$$
\frac{\mathrm{d}}{\mathrm{d}t} P_t u = A P_t u = P_t A u, \qquad \forall u \in \mathcal{D}(A),\ t > 0.
$$

b) For all $u \in C_\infty(\mathbb{R}^d)$, the time-integral $\int_0^t P_s u\, \mathrm{d}s$ belongs to $\mathcal{D}(A)$, and

$$
P_t u - u = A \int_0^t P_s u\, \mathrm{d}s, \qquad \forall u \in C_\infty(\mathbb{R}^d),\ t > 0.
$$

If additionally $u \in \mathcal{D}(A)$, then

$$
P_t u - u = \int_0^t A P_s u\, \mathrm{d}s = \int_0^t P_s A u\, \mathrm{d}s, \qquad \forall u \in \mathcal{D}(A),\ t > 0.
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Three Lenses on Lemma 2.16)</span></p>

* **Kolmogorov's backward equation.** Part (a) says the expected value $v(t, x) := P_t u(x)$ is the unique solution to the PDE $\partial_t v = A v$. In the context of Brownian motion where $A = \tfrac{1}{2}\Delta$, this is simply the *heat equation*.
* **Reverse flow in generative modelling.** If $P_t$ describes the forward "blurring" of an image, the generator $A$ provides the *local rule* for how that blurring happens at every point in space and time — exactly the object whose adjoint we numerically integrate to denoise.
* **Operator-theoretic FTC.** Part (b) is the operator-theoretic version of the *fundamental theorem of calculus*: the total change in the observable ($P_t u - u$) is the accumulated effect of the generator acting along the path of the semigroup.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 2.16</summary>

**a)** Let $0 < \varepsilon < t$ and $u \in \mathcal{D}(A)$. The semigroup and contraction properties give

$$
\left\lVert \frac{P_\varepsilon P_t u - P_t u}{\varepsilon} - P_t A u \right\rVert_\infty
= \left\lVert \frac{P_{t+\varepsilon} u - P_t u}{\varepsilon} - P_t A u \right\rVert_\infty
= \left\lVert P_t\!\left(\frac{P_\varepsilon u - u}{\varepsilon} - A u\right) \right\rVert_\infty
\le \left\lVert \frac{P_\varepsilon u - u}{\varepsilon} - A u \right\rVert_\infty \xrightarrow{\varepsilon \to 0} 0.
$$

By definition of the generator, $P_t u \in \mathcal{D}(A)$ and $\mathrm{d}^+/\mathrm{d}t\, P_t u = A P_t u = P_t A u$. The left-sided derivative is handled symmetrically via strong continuity:

$$
\left\lVert \frac{P_t u - P_{t-\varepsilon} u}{\varepsilon} - P_t A u \right\rVert_\infty
\le \left\lVert P_{t-\varepsilon}\!\left(\frac{P_\varepsilon u - u}{\varepsilon} - A u\right) \right\rVert_\infty + \lVert P_{t-\varepsilon} A u - P_t A u \rVert_\infty \xrightarrow{\varepsilon \to 0} 0.
$$

**b)** Initially consider $u \in \mathcal{B}\_b(\mathbb{R}^d)$ and $t, \varepsilon > 0$. Starting in this broader space emphasises that interchanging the transition operator $P_\varepsilon$ and the time integral relies purely on Fubini's theorem for non-negative kernels and does not yet require the continuity of $u$. By Fubini,

$$
P_\varepsilon\!\left(\int_0^t P_s u\, \mathrm{d}s\right)
= \int\!\int_0^t \!\int u(y)\, p_s(z, \mathrm{d}y)\, \mathrm{d}s\, p_\varepsilon(x, \mathrm{d}z)
= \int_0^t P_\varepsilon P_s u(x)\, \mathrm{d}s.
$$

The difference quotient becomes

$$
\frac{P_\varepsilon - \mathrm{id}}{\varepsilon} \int_0^t P_s u(x)\, \mathrm{d}s
= \frac{1}{\varepsilon}\!\left(\int_t^{t + \varepsilon} P_s u(x)\, \mathrm{d}s - \int_0^\varepsilon P_s u(x)\, \mathrm{d}s\right),
$$

obtained via the change of variables $r = s + \varepsilon$. Now require $u \in C_\infty(\mathbb{R}^d)$ to invoke strong continuity: for $r \ge 0$,

$$
\left\lvert \frac{1}{\varepsilon} \int_r^{r+\varepsilon} P_s u\, \mathrm{d}s - P_r u \right\rvert
\le \frac{1}{\varepsilon} \int_r^{r+\varepsilon} \lvert P_s u - P_r u \rvert\, \mathrm{d}s
\le \sup_{r \le s \le r + \varepsilon} \lVert P_s u - P_r u \rVert_\infty \xrightarrow{\varepsilon \to 0} 0.
$$

Hence the difference quotient converges, so $\int_0^t P_s u\, \mathrm{d}s \in \mathcal{D}(A)$ and the first claim holds. If $u \in \mathcal{D}(A)$, applying part (a) twice lets us pull the generator inside the integral:

$$
\int_0^t P_s A u\, \mathrm{d}s = \int_0^t A P_s u\, \mathrm{d}s = \int_0^t \frac{\mathrm{d}}{\mathrm{d}s} P_s u\, \mathrm{d}s = P_t u - u = A\!\left(\int_0^t P_s u\, \mathrm{d}s\right). \qquad\square
$$

</details>
</div>

We have shown that the generator acts as the infinitesimal derivative of the transition semigroup. The natural follow-up question is:

> Does this infinitesimal operator encode enough information to *uniquely reconstruct* the global flow of the entire semigroup?

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">2.17 (Generator Encodes the Semigroup)</span></p>

Let $(P_t)\_{t \ge 0}$ be a Feller semigroup with generator $(A, \mathcal{D}(A))$.

a) $\mathcal{D}(A)$ is a *dense* subset of $C_\infty(\mathbb{R}^d)$.

b) $(A, \mathcal{D}(A))$ is a *closed* operator: if $(u_n)\_{n \ge 1} \subset \mathcal{D}(A)$ satisfies $u_n \to u$ in $C_\infty$ and $(A u_n)$ converges uniformly, then $u \in \mathcal{D}(A)$ and $A u = \lim_n A u_n$.

c) If $(T_t)\_{t \ge 0}$ is another Feller semigroup with the same generator $(A, \mathcal{D}(A))$, then $P_t = T_t$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Three Take-Aways)</span></p>

* **Density** ensures the generator is defined on a space large enough to approximate any continuous function.
* **Closedness** guarantees that limit operations (e.g. integrating over time) do not accidentally fall out of the generator's domain.
* **Uniqueness.** The generator acts as a *unique mathematical fingerprint*: two Feller processes sharing the same generator have the *exact same* transition probabilities.
* **Diffusion-model implication.** In score-based diffusion, we define the data corruption via a forward SDE (e.g. $\mathrm{d}X = f(X, t)\, \mathrm{d}t + g(t)\, \mathrm{d}B_t$); this SDE explicitly defines an infinitesimal generator. Corollary 2.17 guarantees that this *local formulation uniquely and unambiguously defines the macroscopic transition kernels* $q(x_t \mid x_0)$ used during training.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Corollary 2.17</summary>

**a)** Let $u \in C_\infty(\mathbb{R}^d)$. By Lemma 2.16, the time-averaged function

$$
u_\varepsilon := \frac{1}{\varepsilon} \int_0^\varepsilon P_s u\, \mathrm{d}s
$$

belongs to $\mathcal{D}(A)$. By an argument identical to the strong continuity proof, $\lim_{\varepsilon \to 0} \lVert u_\varepsilon - u \rVert_\infty = 0$. Hence any function in $C_\infty$ can be approximated by elements of $\mathcal{D}(A)$, so $\mathcal{D}(A)$ is dense.

**b)** Let $(u_n)\_{n \ge 1} \subset \mathcal{D}(A)$ with $u_n \to u$ and $A u_n \to w$ uniformly. By Lemma 2.16 (b) and dominated convergence, we may pass the limit through the integral,

$$
P_t u(x) - u(x) = \lim_{n \to \infty} \bigl(P_t u_n(x) - u_n(x)\bigr) = \lim_{n \to \infty} \int_0^t P_s A u_n(x)\, \mathrm{d}s = \int_0^t P_s w(x)\, \mathrm{d}s.
$$

Dividing by $t$ and sending $t \downarrow 0$, the continuity of the semigroup yields

$$
\lim_{t \to 0} \frac{P_t u(x) - u(x)}{t} = \lim_{t \to 0} \frac{1}{t} \int_0^t P_s w(x)\, \mathrm{d}s = w(x).
$$

Thus $u \in \mathcal{D}(A)$ and $A u = w$, proving closedness.

**c)** Let $0 < s < t$ and assume $0 < \lvert h \rvert < \min\lbrace t-s, s\rbrace$. We evaluate the difference quotient of $P_{t-s} T_s$ applied to $u \in \mathcal{D}(A)$. Using a product-rule style decomposition,

$$
\frac{P_{t-s} T_s u - P_{t-s+h} T_{s-h} u}{h} = \frac{P_{t-s} - P_{t-s+h}}{h} T_s u + P_{t-s+h}\, \frac{T_s u - T_{s-h} u}{h}.
$$

Sending $h \to 0$ via Lemma 2.16 (a) gives

$$
\frac{\mathrm{d}}{\mathrm{d}s} P_{t-s} T_s u = -P_{t-s} A T_s u + P_{t-s} A T_s u = 0.
$$

So the composition is constant in $s \in [0, t]$, and integrating yields

$$
0 = \int_0^t \frac{\mathrm{d}}{\mathrm{d}s} P_{t-s} T_s u\, \mathrm{d}s = P_{t-s} T_s u(x)\bigl|_{s=0}^{s=t} = T_t u(x) - P_t u(x).
$$

Therefore $T_t u = P_t u$ for $u \in \mathcal{D}(A)$. Since $\mathcal{D}(A)$ is dense in $C_\infty$ (part a), approximate any $u \in C_\infty$ by $(u_j) \subset \mathcal{D}(A)$ and pass to the limit: $T_t u = \lim_j T_t u_j = \lim_j P_t u_j = P_t u$. $\square$

</details>
</div>

## 3 Diffusions

*Diffusion* is a physical phenomenon describing the tendency of two (or more) substances — e.g. gases or liquids — to reach equilibrium. When particles of one type ($B$) move into another substance ($S$), their movement is influenced by various temporal and spatial inhomogeneities. Since the particles are physical objects, it is reasonable to assume their trajectories are continuous (in fact, differentiable). Diffusion phenomena are governed by **Fick's law**. Let $p = p(t, x)$ denote the concentration of the $B$-particles at $(t, x)$, and $J = J(t, x)$ the particle flux:

$$
J = -D\, \frac{\partial p}{\partial x}, \qquad \frac{\partial p}{\partial t} = -\frac{\partial J}{\partial x},
$$

where $D$ is the **diffusion constant** depending on the geometry and properties of the substances. In 1905 Einstein considered the particle motion observed by Brown and showed that, under temporal and spatial homogeneity, it can be described as a diffusion phenomenon yielding the Gaussian density,

$$
\frac{\partial p(t,x)}{\partial t} = D\, \frac{\partial^2 p(t,x)}{\partial x^2} \quad \implies \quad p(t, x) = \frac{1}{\sqrt{4\pi D t}}\, e^{-x^2 / (4 D t)}.
$$

The diffusion coefficient $D = \frac{R T}{N \cdot 6 \pi k P}$ depends on physical constants (absolute temperature $T$, universal gas constant $R$, Avogadro's number $N$, friction coefficient $k$, particle radius $P$). If the diffusion coefficient depends on time and space, $D = D(t, x)$, Fick's law leads to

$$
\frac{\partial p(t,x)}{\partial t} = D(t, x)\, \frac{\partial^2 p(t, x)}{\partial x^2} + \frac{\partial D(t, x)}{\partial x}\, \frac{\partial p(t, x)}{\partial x}.
$$

In a *mathematical model* of diffusion we adopt a microscopic point of view and describe the random position of particles by a stochastic process $(X_t)\_{t \ge 0}$. In view of the physical discussion, it is reasonable to require this stochastic process to:

1. have **continuous trajectories** $t \mapsto X_t(\omega)$;
2. have a **generator which is a second-order differential operator**.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Connection</span><span class="math-callout__name">(Why Diffusion Processes Are Built for Score-Based Generative AI)</span></p>

* In score-based diffusion models, the data distribution (e.g. a manifold of natural images) acts as the *initial concentration*.
* We construct a microscopic stochastic process $(X_t)\_{t \ge 0}$ to artificially diffuse this distribution into pure noise.
* The two requirements above (continuous paths + second-order generator) ensure the transition probabilities are governed by *tractable PDEs*, and this is precisely what allows us to formulate and learn the **reverse-time score function**.

</div>

Our motivation stems from Example 2.15. The classical $d$-dimensional Brownian motion with drift, $X_t = C B_t + b t$, is the prototypical example of a diffusion process. Its generator takes the form

$$
A u(x) = \tfrac{1}{2} \sum_{i,j=1}^d a_{ij}\, \partial_i \partial_j u(x) + \sum_{i=1}^d b_i\, \partial_i u(x),
$$

where $(a_{ij}) = C C^{\top}$. Thus the generator involves *both first- and second-order* derivatives with constant coefficients, with the second-order part encoding the covariance structure of the noise. In general, a diffusion process can be thought of as a Markov process that *behaves locally like a Brownian motion with drift and diffusion coefficients that may depend on the current position*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">3.1 (Diffusion Process)</span></p>

A **diffusion process** $(X_t)\_{t \ge 0}$ is a Feller process with values in $\mathbb{R}^d$, continuous trajectories, and an infinitesimal generator $(A, \mathcal{D}(A))$ such that $C_c^\infty(\mathbb{R}^d) \subset \mathcal{D}(A)$ and for $u \in C_c^\infty(\mathbb{R}^d)$,

$$
A u(x) = L(x, D) u(x) := \tfrac{1}{2} \sum_{i,j=1}^d a_{ij}(x)\, \frac{\partial^2 u(x)}{\partial x_i \partial x_j} + \sum_{j=1}^d b_j(x)\, \frac{\partial u(x)}{\partial x_j}. \tag{3.1}
$$

The symmetric, positive semidefinite matrix $a(x) = (a_{ij}(x))\_{i,j=1}^d \in \mathbb{R}^{d \times d}$ is called the **diffusion matrix**, and $b(x) = (b_1(x), \dots, b_d(x))^{\!\top} \in \mathbb{R}^d$ is called the **drift vector**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">3.2 (Continuity of Coefficients)</span></p>

Since $(X_t)\_{t \ge 0}$ is a Feller process, $A$ maps $C_c^\infty(\mathbb{R}^d)$ into $C_\infty(\mathbb{R}^d)$. Therefore the coefficient functions $a(\cdot)$ and $b(\cdot)$ have to be *continuous*.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">3.3 (Time-Inhomogeneous Diffusions)</span></p>

Sometimes one considers processes for which the diffusion matrix and drift vector depend not only on the spatial variable $x$ but also explicitly on time $t$. In such cases we speak of a **time-inhomogeneous diffusion process** $(X_t)\_{t \ge 0}$, governed by a family of second-order differential operators

$$
L(t, x, D_x) = \tfrac{1}{2} \sum_{i,j=1}^d a_{ij}(t, x)\, \frac{\partial^2}{\partial x_i \partial x_j} + \sum_{j=1}^d b_j(t, x)\, \frac{\partial}{\partial x_j},
$$

where $a_{ij}(t, x)$ and $b_j(t, x)$ are continuous in both variables.

* **Importance for generative modelling.** Forward corruption processes in generative modelling are time-inhomogeneous, since they rely on time-dependent variance schedules $\beta(t)$ to systematically degrade the data structure (cf. §1.2).
* **Space-time embedding.** This situation can be incorporated into the framework of Definition 3.1 by considering the associated *space-time process* $(t, X_t)\_{t \ge 0}$ taking values in $\mathbb{R}^{1+d}$. This is itself a diffusion in the extended space, with infinitesimal generator $\widetilde{A}$ acting on $u \in C_c^\infty(\mathbb{R}^{1+d})$ via

  $$
  \widetilde{A} u(t, x) = \partial_t u(t, x) + L(t, x, D_x) u(t, x).
  $$

* **Notation.** It is convenient to use the notation $p((t, x), (s, y))$ to emphasise the dependence of transition probabilities on the full space-time points.

</div>

### 3.1 Kolmogorov's Theory

Kolmogorov's seminal paper *"Über die analytischen Methoden der Wahrscheinlichkeitsrechnung"* (1931) marks the beginning of the mathematically rigorous theory of stochastic processes in continuous time. In sections 13 and 14 of this paper, Kolmogorov develops the analytical foundations of diffusion processes; we now follow his ideas.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">3.4 (Kolmogorov 1933)</span></p>

Let $(X_t)\_{t \ge 0}$ with $X_t = (X_t^1, \dots, X_t^d)^{\!\top}$ be a Feller process taking values in $\mathbb{R}^d$. Assume that for all $\delta > 0$ and $i, j = 1, \dots, d$, the following limits hold uniformly in $x \in \mathbb{R}^d$:

$$
\lim_{t \to 0} \frac{1}{t} \sup_{x \in \mathbb{R}^d} \mathbb{P}_x(\lvert X_t - x \rvert > \delta) = 0, \tag{3.2}
$$

$$
\lim_{t \to 0} \frac{1}{t}\, \mathbb{E}_x\!\left[(X_t^j - x^j)\, \mathbf{1}_{\lbrace \lvert X_t - x \rvert \le \delta \rbrace}\right] = b_j(x), \tag{3.3}
$$

$$
\lim_{t \to 0} \frac{1}{t}\, \mathbb{E}_x\!\left[(X_t^i - x^i)(X_t^j - x^j)\, \mathbf{1}_{\lbrace \lvert X_t - x \rvert \le \delta \rbrace}\right] = a_{ij}(x). \tag{3.4}
$$

Then $(X_t)\_{t \ge 0}$ is a diffusion process in the sense of Definition 3.1.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(What Kolmogorov's Conditions Mean)</span></p>

* Condition **(3.2)** is a *non-explosion / continuity condition*: small spatial jumps occur with probability vanishing faster than $t$. Via the Dynkin–Kinney criterion, this guarantees the continuity of trajectories.
* Condition **(3.3)** identifies the *infinitesimal mean* of displacements (the drift $b(x)$).
* Condition **(3.4)** identifies the *infinitesimal covariance matrix* of displacements (the diffusion matrix $a(x)$).
* Together, these three conditions show that the *first two moments of small increments* completely determine the local generator of a continuous-path Feller process.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Condition (3.2) guarantees the continuity of trajectories via the Dynkin–Kinney criterion (omitted technical step). We demonstrate that $C_c^\infty(\mathbb{R}^d) \subset \mathcal{D}(A)$ and that $A$ restricted to $C_c^\infty(\mathbb{R}^d)$ is the differential operator from Definition 3.1.

Let $u \in C_c^\infty(\mathbb{R}^d)$. Since the second derivatives $\partial_i \partial_j u$ are uniformly continuous, for every $\varepsilon > 0$ there exists $\delta > 0$ such that

$$
\max_{1 \le i, j \le d} \sup_{\lvert x - y \rvert \le \delta} \lvert \partial_i \partial_j u(x) - \partial_i \partial_j u(y) \rvert \le \varepsilon.
$$

Fix $\varepsilon > 0$, choose $\delta > 0$ accordingly, and define the transition semigroup $T_t u(x) := \mathbb{E}_x[u(X_t)]$ along with the local neighbourhood event $\Omega_\delta := \lbrace \lvert X_t - x \rvert \le \delta \rbrace$. Decompose the difference quotient:

$$
\frac{T_t u(x) - u(x)}{t} = \frac{1}{t} \mathbb{E}_x\!\left[(u(X_t) - u(x))\, \mathbf{1}_{\Omega_\delta}\right] + \frac{1}{t} \mathbb{E}_x\!\left[(u(X_t) - u(x))\, \mathbf{1}_{\Omega_\delta^c}\right] := J + J'.
$$

The far-field term $J'$ is bounded using condition (3.2),

$$
\lvert J' \rvert \le \frac{2 \lVert u \rVert_\infty}{t}\, \mathbb{P}_x(\lvert X_t - x \rvert > \delta) \xrightarrow{t \to 0} 0.
$$

For $J$, apply Taylor's theorem: there exists $\theta \in (0, 1)$ such that, for the intermediate point $\Theta = (1 - \theta)x + \theta X_t$,

$$
u(X_t) - u(x) = \sum_{j=1}^d (X_t^j - x^j) \partial_j u(x) + \tfrac{1}{2} \sum_{j,k=1}^d (X_t^j - x^j)(X_t^k - x^k) \partial_j \partial_k u(x) + R_t,
$$

with remainder

$$
R_t = \tfrac{1}{2} \sum_{j,k=1}^d (X_t^j - x^j)(X_t^k - x^k)\bigl(\partial_j \partial_k u(\Theta) - \partial_j \partial_k u(x)\bigr).
$$

Partitioning $J$ into the corresponding terms $J_1, J_2, J_3$, the uniform continuity of second derivatives and the inequality $2 \lvert ab \rvert \le a^2 + b^2$ bound the remainder term:

$$
\frac{1}{t} \mathbb{E}_x[\lvert J_3 \rvert\, \mathbf{1}_{\Omega_\delta}]
\le \frac{\varepsilon}{2t} \sum_{i,j=1}^d \mathbb{E}_x\!\left[(X_t^i - x^i)(X_t^j - x^j)\, \mathbf{1}_{\Omega_\delta}\right]
\xrightarrow[t \to 0]{(3.4)} \frac{\varepsilon}{2} \sum_{j=1}^d a_{jj}(x).
$$

Applying conditions (3.4) and (3.3) to the second- and first-order terms,

$$
\frac{1}{t} \mathbb{E}_x[J_2\, \mathbf{1}_{\Omega_\delta}] \xrightarrow{t \to 0} \tfrac{1}{2} \sum_{j,k=1}^d a_{jk}(x)\, \partial_j \partial_k u(x), \qquad \frac{1}{t} \mathbb{E}_x[J_1\, \mathbf{1}_{\Omega_\delta}] \xrightarrow{t \to 0} \sum_{j=1}^d b_j(x)\, \partial_j u(x).
$$

Since $\varepsilon > 0$ was arbitrary, the remainder vanishes in the limit, and the limit converges to $L(x, D) u(x)$. $\square$

</details>
</div>

If the process $(X_t)\_{t \ge 0}$ admits a transition density, $\mathbb{P}_x(X_t \in A) = \int_A p(t, x, y)\, \mathrm{d}y$ for $t > 0$, the dynamics can be formulated as a **Cauchy problem**. Depending on whether one perturbs the *start* or the *end* of the time interval, this leads to *Kolmogorov's first and second differential equations*.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">3.5 (Backward Kolmogorov Equation, 1931)</span></p>

Let $(X_t)\_{t \ge 0}$ be a diffusion process with generator $(A, \mathcal{D}(A))$ whose restriction to $C_c^\infty(\mathbb{R}^d)$ is given by Definition 3.1, with bounded drift $b \in C_b(\mathbb{R}^d, \mathbb{R}^d)$ and bounded diffusion matrix $a \in C_b(\mathbb{R}^d, \mathbb{R}^{d \times d})$. Assume that $(X_t)\_{t \ge 0}$ admits a transition density $p(t, x, y)$, $t > 0$, $x, y \in \mathbb{R}^d$, satisfying

$$
p,\, \frac{\partial p}{\partial t},\, \frac{\partial p}{\partial x_j},\, \frac{\partial^2 p}{\partial x_j \partial x_k} \in C\bigl((0, \infty) \times \mathbb{R}^d \times \mathbb{R}^d\bigr) \quad \text{for all } j, k,
$$

as well as

$$
p(t, \cdot, \cdot),\, \frac{\partial p(t, \cdot, \cdot)}{\partial x_j},\, \frac{\partial^2 p(t, \cdot, \cdot)}{\partial x_j \partial x_k} \in C_\infty(\mathbb{R}^d \times \mathbb{R}^d) \quad \text{for all } t > 0.
$$

Then the transition density satisfies the **backward Kolmogorov equation**,

$$
\frac{\partial p(t, x, y)}{\partial t} = L(x, D_x)\, p(t, x, y), \quad \text{for all } t > 0,\ x, y \in \mathbb{R}^d,
$$

where $L(x, D_x)$ is the differential operator associated with the generator $A$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Why "Backward"?)</span></p>

* The equation describes the evolution of the transition density with respect to the *initial spatial variable* $x$.
* If we fix a target state $y$ in the future and step infinitesimally backward in time, this PDE governs how the probability of reaching that target changes depending on *where the process started*.
* The differential operator $L(x, D_x)$ acts *exclusively on the starting coordinates* $x$, treating the target coordinates $y$ as fixed parameters.
* **Generative-modelling reading.** Backward Kolmogorov is the PDE governing how an observable's expectation evolves under forward diffusion — a sister equation to the Fokker–Planck (forward) equation that governs how the *density itself* evolves with respect to the terminal coordinate $y$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 3.5</summary>

Denote by $L(x, D_x)$ the second-order differential operator. By assumption, $A = L(x, D_x)$ on $C_c^\infty(\mathbb{R}^d)$. Since $p(t, x, y)$ is the transition density, the transition semigroup is

$$
T_t u(x) = \int_{\mathbb{R}^d} p(t, x, y)\, u(y)\, \mathrm{d}y.
$$

Using dominated convergence, the analytical conditions on $p(t, x, y)$ ensure that $T_t$ maps $C_c^\infty(\mathbb{R}^d)$ into $C_\infty^2(\mathbb{R}^d)$. Since the drift and diffusion coefficients are bounded, the global estimate

$$
\lVert A u \rVert_\infty \le \kappa\, \lVert u \rVert_{(2)}, \qquad \lVert u \rVert_{(2)} := \lVert u \rVert_\infty + \sum_{j=1}^d \lVert \partial_j u \rVert_\infty + \sum_{j,k=1}^d \lVert \partial_j \partial_k u \rVert_\infty,
$$

holds for $u \in C_\infty^2(\mathbb{R}^d)$, with $\kappa = \kappa(b, a)$. Since $C_c^\infty(\mathbb{R}^d) \subset \mathcal{D}(A)$ and $(A, \mathcal{D}(A))$ is closed (Corollary 2.17 (b)), the larger space $C_\infty^2(\mathbb{R}^d)$ is also contained in $\mathcal{D}(A)$, and the generator coincides with $L(x, D_x)$ there:

$$
A\bigr|_{C_\infty^2(\mathbb{R}^d)} = L(x, D_x)\bigr|_{C_\infty^2(\mathbb{R}^d)}.
$$

Therefore Lemma 2.16 implies the time derivative of the semigroup is generated by $A$:

$$
\frac{\mathrm{d}}{\mathrm{d}t} T_t u = A T_t u = L(\cdot, D) T_t u.
$$

Expanding via the integral representation,

$$
\frac{\partial}{\partial t} \int_{\mathbb{R}^d} p(t, x, y)\, u(y)\, \mathrm{d}y = L(x, D_x) \int_{\mathbb{R}^d} p(t, x, y)\, u(y)\, \mathrm{d}y = \int_{\mathbb{R}^d} L(x, D_x)\, p(t, x, y)\, u(y)\, \mathrm{d}y,
$$

for all $u \in C_c^\infty(\mathbb{R}^d)$. The last equality follows from the smoothness assumptions on $p$, which permit the differentiation–integration interchange via the differentiation lemma for parameter-dependent integrals. Rearranging,

$$
\int_{\mathbb{R}^d}\!\!\left(\frac{\partial p(t, x, y)}{\partial t} - L(x, D_x) p(t, x, y)\right)\!u(y)\, \mathrm{d}y = 0 \quad \text{for all } u \in C_c^\infty(\mathbb{R}^d).
$$

By the fundamental lemma of the calculus of variations, since this integral vanishes for any smooth test function with compact support, the continuous integrand must vanish identically, proving the claim. $\square$

</details>
</div>

While the backward equation analyses how the transition density changes with respect to the *initial starting position*, the **forward equation** describes its evolution concerning the *target destination*. In physics and generative modelling, this is often referred to as the **Fokker–Planck equation**: it tracks how probability mass drifts and diffuses forward through space over time.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">3.6 (Forward Kolmogorov Equation, 1931)</span></p>

Let $(X_t)\_{t \ge 0}$ be a diffusion process with generator $(A, \mathcal{D}(A))$ such that its restriction to $C_c^\infty(\mathbb{R}^d)$ is the second-order differential operator of Definition 3.1. Assume that the drift and diffusion coefficients exhibit higher regularity,

$$
b \in C^1(\mathbb{R}^d, \mathbb{R}^d), \qquad a \in C^2(\mathbb{R}^d, \mathbb{R}^{d \times d}).
$$

Suppose $X_t$ admits a transition density $p(t, x, y) \in C\bigl((0, \infty) \times \mathbb{R}^d \times \mathbb{R}^d\bigr)$ whose spatial derivatives with respect to the *target* variable $y$ satisfy

$$
\frac{\partial p}{\partial y_j},\ \frac{\partial^2 p}{\partial y_j \partial y_k} \in C\bigl((0, \infty) \times \mathbb{R}^d \times \mathbb{R}^d\bigr) \quad \text{for all } j, k.
$$

Then the transition density satisfies the **forward Kolmogorov equation**,

$$
\frac{\partial p(t, x, y)}{\partial t} = L^{\ast}(y, D_y)\, p(t, x, y), \quad \text{for all } t > 0,\ x, y \in \mathbb{R}^d,
$$

where $L^{\ast}(y, D_y)$ is the *formal adjoint* of the differential operator $L$, defined as

$$
L^{\ast}(y, D_y)\, u(y) = -\sum_{j=1}^d \partial_{y_j}\!\bigl(b_j(y)\, u(y)\bigr) + \frac{1}{2} \sum_{i,j=1}^d \partial_{y_i} \partial_{y_j}\!\bigl(a_{ij}(y)\, u(y)\bigr). \tag{3.5}
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Reading the Adjoint Operator)</span></p>

* The adjoint operator $L^{\ast}$ describes the **divergence of the probability flow**: the first term (advection/drift) rigidly *shifts* the probability mass, whilst the second term (diffusion) *scatters* it.
* The derivatives in $L^{\ast}$ are evaluated with respect to the *target* variable $y$, thus answering the question: *"how does the probability of arriving exactly at $y$ change as time progresses?"*
* In one line: the backward equation differentiates where you *start*; the forward equation differentiates where you *land*.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 3.6</summary>

The proof structure closely mirrors that of the backward equation, but we exploit the *alternate* commutation identity from Lemma 2.16 (a),

$$
\frac{\mathrm{d}}{\mathrm{d}t} T_t u = T_t A u = T_t L(\cdot, D) u \quad \text{for } u \in C_c^\infty(\mathbb{R}^d).
$$

Expressing both sides as integrals with respect to the transition density $p(t, x, y)$, we obtain the integral identity

$$
\frac{\partial}{\partial t} \int_{\mathbb{R}^d} p(t, x, y)\, u(y)\, \mathrm{d}y = \int_{\mathbb{R}^d} p(t, x, y)\, L(y, D_y)\, u(y)\, \mathrm{d}y.
$$

To isolate $\partial_t p(t, x, y)$, we must shift the differential operator $L$ away from the test function $u(y)$ and onto the density $p(t, x, y)$. This is achieved via multidimensional **integration by parts** (no boundary terms arise, since $u$ has compact support). Each spatial derivative shifted from $u(y)$ to $p(t, x, y)$ incurs a change of sign and necessitates differentiating the coefficient functions ($b$ and $a$) alongside the density; this mechanical shifting is exactly what yields the formal adjoint $L^{\ast}$ of (3.5):

$$
\int_{\mathbb{R}^d} p(t, x, y)\, L(y, D_y) u(y)\, \mathrm{d}y = \int_{\mathbb{R}^d} L^{\ast}(y, D_y)\, p(t, x, y)\, u(y)\, \mathrm{d}y.
$$

Rearranging,

$$
\int_{\mathbb{R}^d}\!\!\left(\frac{\partial p(t, x, y)}{\partial t} - L^{\ast}(y, D_y)\, p(t, x, y)\right)\! u(y)\, \mathrm{d}y = 0 \quad \text{for all } u \in C_c^\infty(\mathbb{R}^d),
$$

and the fundamental lemma of the calculus of variations forces the continuous integrand to vanish identically.

Unlike the backward case, this approach avoids the need to interchange the differential operator and the integral, allowing us to relax the boundedness assumptions on $p$, $b$, and $a$. However, we pay a direct theoretical price: the integration by parts requires the drift $b$ and the diffusion matrix $a$ to be continuously differentiable ($C^1$ and $C^2$, respectively). $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Random Initial Conditions)</span></p>

We have thus far formulated both equations assuming a fixed, deterministic initial state $x$ (i.e., $X_0 = x$ with certainty). In practice, processes often start from a random initial state distributed according to a probability measure $\mu$. The **marginal density** of the process at time $t$ is defined via the convolution

$$
p^\mu(t, y) := \int_{\mathbb{R}^d} p(t, x, y)\, \mu(\mathrm{d}x).
$$

Because the forward operator $L^{\ast}(y, D_y)$ is linear and acts solely on $y$, we can integrate both sides of the forward equation against $\mu(\mathrm{d}x)$ and safely interchange the integration with the derivatives. Consequently, the marginal density strictly obeys the exact same forward equation,

$$
\partial_t\, p^\mu = L^{\ast} p^\mu.
$$

* **Generative-modelling reading.** This $p^\mu(t, \cdot)$ — with $\mu$ the data distribution — is precisely the marginal $p(t, \cdot)$ of §1.2 whose score $\nabla_x \log p(t, x)$ drives the reverse SDE (1.4). The Fokker–Planck equation is the deterministic law that this sole unknown obeys.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Stochastic Reading: Perturbing the Start or the End)</span></p>

Using probability theory and the Markov property, we can derive a geometric intuition for the difference between the two equations. Run the process over $[0, t]$ and analyse the probability of landing in a set $C$; this trajectory can be perturbed in two distinct ways.

1. **Backward (initial perturbation).** Perturb the movement at the very beginning: let the process run on $[0, h]$, then evaluate the remaining journey of length $t - h$. By the Markov property, the change in probability is

   $$
   \mathbb{P}_x(X_t \in C) - \mathbb{P}_x(X_{t-h} \in C) = \mathbb{E}_x\!\left[\mathbb{P}_{X_h}(X_{t-h} \in C)\right] - \mathbb{P}_x(X_{t-h} \in C) = (T_h - \mathrm{id})\, \mathbb{P}_{(\cdot)}(X_{t-h} \in C)(x).
   $$

   Substituting the transition densities and dividing by $h$, one obtains

   $$
   \frac{p(t, x, y) - p(t-h, x, y)}{h} = \frac{(T_h - \mathrm{id})\, p(t-h, \cdot, y)(x)}{h}.
   $$

   Taking $h \to 0$ naturally applies the generator to the *starting* coordinate $x$, yielding the **backward equation**, $\partial_t p = A_x\, p$.

2. **Forward (terminal perturbation).** Alternatively, let the process run completely unperturbed up to time $t - h$, and apply the perturbation at the end, during $[t-h, t]$. Conditioning on the state at $t - h$ yields

   $$
   \mathbb{P}_x(X_t \in C) - \mathbb{P}_x(X_{t-h} \in C) = \mathbb{E}_x\!\left[\mathbb{P}_{X_{t-h}}(X_h \in C)\right] - \mathbb{P}_x(X_{t-h} \in C) = \mathbb{E}_x\!\left[(T_h - \mathrm{id})\, \mathbf{1}_C(X_{t-h})\right].
   $$

   Passing to transition densities and dividing by $h$, one shifts the focus to the *target* variable,

   $$
   \frac{p(t, x, y) - p(t-h, x, y)}{h} = \frac{(T_h^{\ast} - \mathrm{id})\, p(t-h, x, \cdot)(y)}{h}.
   $$

   As $h \to 0$, this yields the **forward equation**, $\partial_t p = A_y^{\ast}\, p$, where $T_t^{\ast}$ and $A^{\ast}$ denote the formal adjoints of the semigroup and the generator.

</div>

### 3.2 Outlook: Itô's Theory of Diffusion Processes

While Kolmogorov's framework is fundamentally *analytical* — characterising diffusions via transition semigroups, generators, and PDEs — Kiyosi Itô developed a parallel, *path-wise* approach. Instead of tracking the evolution of probability distributions globally, Itô's theory models the trajectories of individual particles explicitly in continuous time.

> *The formal construction of the stochastic machinery used in this section — including the rigorous definition of the stochastic integral and Itô's calculus — will be developed from first principles in the coming weeks. For the present discussion, we introduce these concepts conceptually, to establish the continuous-time link between path trajectories and the infinitesimal generator.*

In Itô's framework, a $d$-dimensional diffusion process $(X_t)\_{t \ge 0}$ is conceptualised as the unique solution to a *stochastic differential equation* (SDE) of the form

$$
\mathrm{d}X_t = b(X_t)\, \mathrm{d}t + \sigma(X_t)\, \mathrm{d}B_t, \qquad X_0 = x_0, \tag{3.6}
$$

where $(B_t)\_{t \ge 0}$ is a standard $m$-dimensional Brownian motion, serving as the continuous source of independent, erratic random fluctuations; $b : \mathbb{R}^d \to \mathbb{R}^d$ is the measurable *drift vector*, representing the local deterministic velocity of the process; and $\sigma : \mathbb{R}^d \to \mathbb{R}^{d \times m}$ is the *dispersion matrix* (or coefficient matrix), which scales and modulates the random ambient noise. The profound link between Itô's path-wise SDE (3.6) and Kolmogorov's analytical operator $L(x, D)$ is established via the *diffusion matrix* $a(x) \in \mathbb{R}^{d \times d}$, defined algebraically as

$$
a(x) := \sigma(x)\, \sigma(x)^{\!\top}. \tag{3.7}
$$

By constructing $a(x)$ in this manner, the components of (3.6) directly prescribe the coefficients of the second-order differential operator

$$
L(x, D)\, u(x) = \frac{1}{2} \sum_{i,j=1}^d a_{ij}(x)\, \frac{\partial^2 u(x)}{\partial x_i \partial x_j} + \sum_{j=1}^d b_j(x)\, \frac{\partial u(x)}{\partial x_j}. \tag{3.8}
$$

Intuitively, (3.6) implies that over an infinitesimal time interval $\mathrm{d}t$, the process behaves locally like a Brownian motion with a mean displacement of $b(x)\, \mathrm{d}t$ and a local covariance structure given by $a(x)\, \mathrm{d}t$. To guarantee that the path-wise formulation (3.6) is well-defined and indeed yields a valid Feller process matching our analytical definition, certain regularity conditions must be imposed on the coefficients.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">3.7 (Existence and Uniqueness of Strong Solutions)</span></p>

Suppose the drift vector $b(\cdot)$ and the dispersion matrix $\sigma(\cdot)$ satisfy the following two conditions for all $x, y \in \mathbb{R}^d$ and a constant $K > 0$:

1. **Global Lipschitz condition:**

   $$
   \lvert b(x) - b(y) \rvert + \lVert \sigma(x) - \sigma(y) \rVert \le K\, \lvert x - y \rvert.
   $$

2. **Linear growth condition:**

   $$
   \lvert b(x) \rvert + \lVert \sigma(x) \rVert \le K\, (1 + \lvert x \rvert).
   $$

Then, for any initial condition $X_0 = x_0$ independent of the Brownian motion, the SDE (3.6) admits a unique, path-wise continuous **strong solution** $(X_t)\_{t \ge 0}$. Furthermore, this solution possesses the strong Markov property and is a Feller process whose infinitesimal generator coincides with the differential operator $L(x, D)$ on $C_c^\infty(\mathbb{R}^d)$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(What the Two Conditions Rule Out)</span></p>

* The **Lipschitz condition** plays the same role as in the Picard–Lindelöf theorem for ODEs: it prevents two solutions started at the same point from *branching apart*, and it is what makes the fixed-point (Picard) iteration underlying the existence proof contract.
* The **linear growth condition** prevents *explosion in finite time* — compare the deterministic equation $\dot{x} = x^2$, whose solutions blow up; coefficients growing at most linearly keep the process alive on all of $[0, \infty)$.
* The conclusion is the bridge promised at the start of the chapter: the path-wise object (3.6) *is* an analytical diffusion in the sense of Definition 3.1, with dictionary $a = \sigma \sigma^{\!\top}$.

</div>

Before moving on to the mechanics of reversing these processes in time, we can summarise the complete symmetry between the two mathematical approaches established so far.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Summary</span><span class="math-callout__name">(The Kolmogorov–Itô Dictionary)</span></p>

* **Analytical viewpoint (Kolmogorov):** focuses on the transition probability densities $p(t, x, y)$ evolving *deterministically* via PDEs ($\partial_t p = L p$, respectively $\partial_t p = L^{\ast} p$).
* **Probabilistic viewpoint (Itô):** focuses on individual sample paths $X_t(\omega)$ evolving *stochastically* via differential trajectories driven by noise ($\mathrm{d}X_t = b\, \mathrm{d}t + \sigma\, \mathrm{d}B_t$).

The algebraic link $a(x) = \sigma(x)\, \sigma(x)^{\!\top}$ ensures that these two perspectives are completely equivalent characterisations of the same physical phenomenon.

</div>

### 3.3 The Time-Reversed Process

The Markov property can be formulated as follows:

> Given the present state, the future and the past are independent.

It was observed early on that this property is *time-symmetric*: if a process $(X_t)\_{t \in [0, T]}$ satisfies the Markov property, then so does the time-reversed process $Y_t := X_{T-t}$. However, this does *not* imply that the reversed process is a homogeneous Markov process — at time $T$, the reversed process must necessarily follow the distribution of $X_0$, so time genuinely plays a role. For diffusion processes, it turns out that the time-reversed process is again a diffusion, but typically with a *time-dependent* drift that depends on the marginal distributions, as the following result shows.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">3.8 (Time Reversal of Diffusions; Haussmann–Pardoux)</span></p>

Let $(X_t)\_{t \ge 0}$ be a diffusion process in $\mathbb{R}^d$ with smooth coefficients $b \in C^1(\mathbb{R}^d; \mathbb{R}^d)$, $a \in C^2(\mathbb{R}^d; \mathbb{R}^{d \times d})$ and transition density $p(t, x, y)$ fulfilling the assumptions of Proposition 3.6. Assume $X_0 \sim \mu$, and let $p^\mu(t, y) := \int_{\mathbb{R}^d} p(t, x, y)\, \mu(\mathrm{d}x) > 0$ denote the marginal density of $X_t$, which we also assume to be smooth for $t > 0$. Let $T > 0$. Then the time-reversed process

$$
Y_t := X_{T-t}, \qquad t \in [0, T],
$$

is again a diffusion process — in general *time-inhomogeneous* — with the **same diffusion matrix** $a$ and the time-dependent drift

$$
\widetilde{b}(t, y) = -\, b(y) + \frac{1}{p^\mu(T-t, y)}\, \nabla_y \cdot \bigl(a(y)\, p^\mu(T-t, y)\bigr),
$$

or, in coordinate form,

$$
\widetilde{b}_j(t, y) = -\, b_j(y) + \sum_{i=1}^d a_{ji}(y)\, \partial_{y_i} \log p^\mu(T-t, y) + \sum_{i=1}^d \partial_{y_i} a_{ji}(y).
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Connection</span><span class="math-callout__name">(The Reverse SDE of Generative Modelling, Proved)</span></p>

* This is exactly **Anderson's theorem** previewed as Theorem 1.1 in §1.1 — now derived from Kolmogorov's analytical machinery (stated here for time-homogeneous forward coefficients; the inhomogeneous case is analogous).
* The only ingredient of $\widetilde{b}$ that is not chosen by the modeller is the **score** $\nabla_y \log p^\mu$: the forward coefficients $b$ and $a$ are design choices, and the diffusion matrix survives reversal *unchanged*. Reversing time therefore costs no new noise structure — only a drift correction by the score. This is the precise mathematical reason why learning $\nabla \log p^\mu(t, \cdot)$ (score matching, §1.2) is *sufficient* to generate.
* **Mind the clock.** If one rewrites the dynamics of $Y$ in the *original* time $t \mapsto T - t$ (integrating backwards from $T$ to $0$), every drift picks up a factor $-1$, giving $b - a \nabla \log p^\mu - \nabla \cdot a$; for $\sigma(t, x) = g(t) I$ the divergence term vanishes and this is precisely the score-SDE form (1.4) of §1.2. The overall sign is pure bookkeeping of which clock one integrates against; the invariant statement is the theorem above.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Sanity Check</span><span class="math-callout__name">(Stationary Ornstein–Uhlenbeck)</span></p>

Let $d = 1$, $b(y) = -y$, $\sigma = \sqrt{2}$ (so $a = 2$), and start in the stationary law $X_0 \sim \mu = \mathcal{N}(0, 1)$. Then $p^\mu(t, y)$ is the standard normal density for *all* $t$, so $\partial_y \log p^\mu = -y$ and

$$
\widetilde{b}(t, y) = -\,(-y) + 2 \cdot (-y) = -y = b(y).
$$

The reversed process is *the same* OU process — as it must be: a stationary, reversible diffusion is statistically indistinguishable run forwards or backwards. Note how the two contributions conspire: the flipped drift $+y$ alone would be explosive, and the score term $-2y$ overcompensates to restore mean reversion. Getting either sign wrong destroys this balance — a useful memory hook for the formula.

</div>

<figure>
  <img src="{{ 'assets/images/notes/sdes_diffusion_models/time_reversal_ou.png' | relative_url }}" alt="Two panels of stochastic paths. Left: forward Ornstein-Uhlenbeck paths starting from two sharp clusters at plus and minus two that merge into a single Gaussian cloud; the bimodal initial density and the Gaussian terminal density are drawn sideways at the edges. Right: paths of the time-reversed diffusion starting from the Gaussian cloud and re-condensing onto the two original clusters, with the same densities mirrored." loading="lazy">
  <figcaption>Theorem 3.8 in action for the OU process $\mathrm{d}X_t = -X_t\,\mathrm{d}t + \sqrt{2}\,\mathrm{d}B_t$ started from the bimodal mixture $\mu = \frac12\mathcal{N}(-2, 0.15^2) + \frac12\mathcal{N}(2, 0.15^2)$, for which the marginal $p^\mu(t,\cdot)$ — and hence the exact score — is available in closed form. <strong>Left.</strong> The forward process transports "data" into (near-)Gaussian noise. <strong>Right.</strong> The reversed diffusion with drift $\widetilde b(\tau, y) = y + 2\,\partial_y \log p^\mu(T-\tau, y)$ transports the noise back onto the two data modes, reproducing the forward marginals in reverse — no learning involved, since the score is exact here.</figcaption>
</figure>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 3.8</summary>

The computation is carried out at the formal level; all $o(h)$-expansions and interchanges below are justified by the standing smoothness assumptions.

**Step 1: Structure of the reversed process.** As noted above, the reversed process $Y_t := X_{T-t}$ is again a (time-inhomogeneous) Markov process with continuous paths. We will compute its generator and show that it is a second-order differential operator with the coefficients stated in the theorem.

**Step 2: Transition density of the reversed process.** We aim to compute the transition density $q((s, y), (t, x))$ of

$$
\mathbb{P}(Y_t \in \mathrm{d}x \mid Y_s = y) \quad \text{for } 0 < s < t < T.
$$

By definition of the time reversal, $Y_s = X_{T-s}$ and $Y_t = X_{T-t}$, so the joint law of $(Y_s, Y_t)$ coincides with the joint law of $(X_{T-s}, X_{T-t})$. Since $X$ is a Markov process with transition densities $p(t, x, y)$ and $X_0 \sim \mu$, the joint density of $(X_u, X_v)$ for $u < v$ is

$$
\mathbb{P}(X_u \in \mathrm{d}x,\ X_v \in \mathrm{d}y) = p^\mu(u, x)\, p(v - u, x, y)\, \mathrm{d}x\, \mathrm{d}y.
$$

In our situation, setting $u = T - t$ and $v = T - s$, we obtain

$$
\mathbb{P}(Y_s \in \mathrm{d}y,\ Y_t \in \mathrm{d}x) = p^\mu(T-t, x)\, p(t-s, x, y)\, \mathrm{d}x\, \mathrm{d}y. \tag{3.9}
$$

Applying Bayes' rule for densities, $q((s, y), (t, x)) = f_{Y_s, Y_t}(y, x) / f_{Y_s}(y)$, with the marginal $f_{Y_s}(y) = p^\mu(T-s, y)$, we find

$$
q((s, y), (t, x)) = \frac{p^\mu(T-t, x)\, p(t-s, x, y)}{p^\mu(T-s, y)}. \tag{3.10}
$$

**Step 3: Generator of the reversed process.** Let $f \in C_c^\infty(\mathbb{R}^d)$ be a test function. The time-dependent generator $\widetilde{\mathcal{L}}\_t$ of $Y$ is defined as

$$
\widetilde{\mathcal{L}}_t f(y) := \lim_{h \to 0} \frac{1}{h}\bigl(\mathbb{E}[f(Y_{t+h}) \mid Y_t = y] - f(y)\bigr).
$$

Using the expression for $q$ from (3.10),

$$
\mathbb{E}[f(Y_{t+h}) \mid Y_t = y] = \frac{1}{p^\mu(T-t, y)} \int_{\mathbb{R}^d} f(x)\, p^\mu(T-t-h, x)\, p(h, x, y)\, \mathrm{d}x.
$$

Two expansions are needed as $h \downarrow 0$. In the *time* slot,

$$
p^\mu(T-t-h, x) = p^\mu(T-t, x) - h\, \partial_t p^\mu(T-t, x) + o(h).
$$

In the *space* slot, the kernel $p(h, \cdot, y)$ — the transition density integrated over its *starting* point — concentrates at $y$ and satisfies, for any smooth $g$, the **dual small-time expansion**

$$
\int_{\mathbb{R}^d} g(x)\, p(h, x, y)\, \mathrm{d}x = g(y) + h\, L^{\ast}(y, D_y)\, g(y) + o(h),
$$

the adjoint counterpart of the semigroup expansion $T_h g = g + h L g + o(h)$: differentiating in $h$ with the backward equation $\partial_h p = L(x, D_x)\, p$ (Proposition 3.5) and shifting $L$ onto $g$ by integration by parts gives $\frac{\mathrm{d}}{\mathrm{d}h} \int g\, p\, \mathrm{d}x = \int (L^{\ast} g)\, p\, \mathrm{d}x \to L^{\ast} g(y)$.

Applying both expansions with $g := f \cdot p^\mu(T-t, \cdot)$, and noting $g(y) / p^\mu(T-t, y) = f(y)$,

$$
\mathbb{E}[f(Y_{t+h}) \mid Y_t = y] = f(y) + \frac{h}{p^\mu(T-t, y)} \Bigl[ L^{\ast}\bigl(f\, p^\mu(T-t, \cdot)\bigr)(y) - f(y)\, \partial_t p^\mu(T-t, y) \Bigr] + o(h).
$$

Since the marginal density satisfies the forward Kolmogorov (Fokker–Planck) equation $\partial_t p^\mu = L^{\ast} p^\mu$ (Proposition 3.6 and the remark on random initial conditions), we arrive at the compact *conjugation formula*

$$
\widetilde{\mathcal{L}}_t f = \frac{L^{\ast}(f\, p^\mu) - f\, L^{\ast} p^\mu}{p^\mu}\bigg|_{(T-t,\, \cdot)}.
$$

As a first sanity check: constants are killed, $\widetilde{\mathcal{L}}\_t 1 = 0$, as any generator must satisfy.

It remains to expand. Write $p := p^\mu(T-t, \cdot)$. For the drift part of $L^{\ast}$,

$$
-\sum_{j=1}^d \partial_j(b_j\, f\, p) + f \sum_{j=1}^d \partial_j(b_j\, p) = -\sum_{j=1}^d b_j\, p\, \partial_j f.
$$

For the diffusion part, set $g_{ij} := a_{ij}\, p$; the product rule gives

$$
\partial_i \partial_j (g_{ij}\, f) - f\, \partial_i \partial_j g_{ij} = \partial_i f\, \partial_j g_{ij} + \partial_j f\, \partial_i g_{ij} + g_{ij}\, \partial_i \partial_j f,
$$

and summing over $i, j$ — using the symmetry $a_{ij} = a_{ji}$ to merge the two cross terms — yields

$$
\frac{L^{\ast}(f p) - f\, L^{\ast} p}{p} = \frac{1}{2} \sum_{i,j=1}^d a_{ij}\, \partial_i \partial_j f + \sum_{j=1}^d \Bigl[ -b_j + \frac{1}{p} \sum_{i=1}^d \partial_i (a_{ji}\, p) \Bigr] \partial_j f.
$$

Thus $\widetilde{\mathcal{L}}\_t$ is again a second-order differential operator with the *same* diffusion matrix $a$ and drift

$$
\widetilde{b}_j(t, y) = -\, b_j(y) + \frac{1}{p^\mu(T-t, y)} \sum_{i=1}^d \partial_{y_i}\bigl(a_{ji}(y)\, p^\mu(T-t, y)\bigr);
$$

expanding the derivative of the product, $\frac{1}{p} \partial_i(a_{ji} p) = \partial_i a_{ji} + a_{ji}\, \partial_i \log p$, gives the coordinate form in the statement. $\square$

</details>
</div>

## 4 Stochastic Integration and Stochastic Differential Equations

In Chapter 3, we analysed diffusion processes from a *macroscopic* perspective: we established that the temporal evolution of their transition densities $p(t, x)$ is governed deterministically by the Fokker–Planck (forward Kolmogorov) equation,

$$
\partial_t p = -\nabla \cdot (b\, p) + \frac{1}{2} \sum_{i,j} \partial_i \partial_j (a_{ij}\, p).
$$

However, in applications such as *score-based generative diffusion models*, manipulating the global probability density is insufficient. To actively generate new data (e.g., sampling an image from noise), we must be able to simulate the *individual sample paths* $X_t(\omega)$ backwards in time. This requires the path-wise, microscopic formulation given by a stochastic differential equation,

$$
\mathrm{d}X_t = b(X_t)\, \mathrm{d}t + \sigma(X_t)\, \mathrm{d}B_t. \tag{4.1}
$$

While physically intuitive, this differential notation presents a severe mathematical anomaly: by definition, the paths of Brownian motion $t \mapsto B_t(\omega)$ are *nowhere differentiable* (§0.3), so the derivative $\mathrm{d}B_t / \mathrm{d}t$ does not exist in any classical sense, rendering the SDE formally meaningless. To resolve this, we reinterpret (4.1) strictly as an *integral equation*,

$$
X_t = X_0 + \int_0^t b(X_s)\, \mathrm{d}s + \int_0^t \sigma(X_s)\, \mathrm{d}B_s. \tag{4.2}
$$

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Why Classical Integration Fails Here)</span></p>

* The first integral in (4.2) is a standard, path-wise Riemann–Lebesgue integral — no difficulty there.
* The second integral integrates against Brownian motion, a process of **infinite total variation** on every interval. The Riemann–Stieltjes theory requires an integrator of *bounded variation*, so classical integration theory inherently fails.
* Infinite total variation is not an accident but a *theorem*: any continuous process with non-vanishing quadratic variation (as we will see, $\langle B \rangle\_t = t$) must have infinite total variation — otherwise its squared increments would be dominated by $\max_i \lvert \Delta B \rvert \cdot \sum_i \lvert \Delta B \rvert \to 0$.
* The way out is to abandon path-wise definitions and exploit the **martingale structure** of $B$ instead: an $L^2$-theory of integration, built in this section, whose engine is Itô's isometry.

</div>

Therefore, to rigorously define diffusion trajectories — and to establish the foundational mechanics of reverse-time sampling in generative AI — we must first develop a completely new mathematical framework.

### 4.1 Stochastic Integration

In this section, we consider martingales $M = (M_t)\_{t \ge 0}$ with respect to a filtration $(\mathcal{F}\_t)\_{t \ge 0}$ that satisfy

$$
\sup_{t \ge 0} \mathbb{E}\bigl[M_t^2\bigr] < +\infty, \tag{4.3}
$$

$$
t \mapsto M_t \ \text{is continuous a.s.} \tag{4.4}
$$

We define the space of such martingales as

$$
\mathcal{M}_2^C := \bigl\lbrace M \,:\, M \text{ is a martingale with properties (4.3) and (4.4)} \bigr\rbrace.
$$

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Convention</span><span class="math-callout__name">(Standing Assumptions for §4.1)</span></p>

Throughout this section, we assume that $(X_t, \mathcal{F}\_t)\_{t \ge 0}$ is progressively measurable, that the filtration $(\mathcal{F}\_t)\_{t \ge 0}$ is right-continuous, and that $\mathcal{F}\_t$ contains all $\mathbb{P}$-null sets for all $t \ge 0$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">4.1 (Simple Process)</span></p>

A stochastic process $(X_t)\_{t \ge 0}$ is called a **simple process** if there exist real numbers $0 = t_0 < t_1 < t_2 < \cdots$ and $\mathcal{F}\_{t_i}$-measurable random variables $\xi_i$ for $i \ge 0$ such that

$$
X_t = \xi_0 \cdot \mathbb{1}_{\lbrace 0 \rbrace}(t) + \sum_{i=0}^{\infty} \xi_i \cdot \mathbb{1}_{(t_i,\, t_{i+1}]}(t), \qquad 0 \le t < \infty, \tag{4.5}
$$

with $t_n \to \infty$ as $n \to \infty$, and

$$
\sup_{n \ge 0}\, \lvert \xi_n \rvert \le C \quad \text{a.s. for some constant } C > 0. \tag{4.6}
$$

If a process $(X_t)\_{t \ge 0}$ is simple, we also write $X \in \mathcal{L}\_0$.

</div>

We now define stochastic integrals of the form $\int_0^t X_s\, \mathrm{d}M_s$ for integrands $X \in \mathcal{L}\_0$ and integrators $M \in \mathcal{M}\_2^C$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">4.2 (Stochastic Integral for Simple Processes)</span></p>

For $X \in \mathcal{L}\_0$ and $M \in \mathcal{M}\_2^C$, we define the stochastic integral path-wise as

$$
\int_0^t X_s\, \mathrm{d}M_s := \sum_{i=0}^{n-1} \xi_i \bigl(M_{t_{i+1}} - M_{t_i}\bigr) + \xi_n \bigl(M_t - M_{t_n}\bigr),
$$

where $n$ is the unique index chosen so that $t_n \le t < t_{n+1}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name">(The Gambling Reading)</span></p>

Think of $M$ as the fluctuating value of a fair game and of $\xi_i$ as the *stake* you hold during the round $(t_i, t_{i+1}]$. The integral is your cumulative winnings. Two features of Definition 4.1 are then no longer technicalities:

* $\xi_i$ is $\mathcal{F}\_{t_i}$-measurable — the stake must be chosen *before* the increment $M_{t_{i+1}} - M_{t_i}$ is revealed. No clairvoyance: this is exactly why the integral of a bounded strategy against a martingale remains a martingale (fair games stay fair under non-anticipating betting).
* The uniform bound (4.6) keeps expectations finite; it will be relaxed in several stages below.

</div>

We will subsequently extend the definition to vastly larger classes of processes $X$ and $M$. For the next generalisation, we must formalise the concept of *variance accumulation over time*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">4.3 (Quadratic Variation)</span></p>

A process $X$ is said to have **bounded quadratic variation** if there exists a process $\langle X \rangle\_t$ such that

$$
\sum_{i=0}^{n-1} \bigl(X_{t_{i+1}} - X_{t_i}\bigr)^2 \xrightarrow{\ \mathbb{P}\ } \langle X \rangle_t \quad \text{in probability},
$$

for all $t \ge 0$, where the sum is taken over any partition $0 = t_0 < \cdots < t_n = t$ with mesh size satisfying

$$
\sup_{0 \le i \le n-1} \bigl(t_{i+1} - t_i\bigr) \to 0.
$$

The process $\langle X \rangle\_t$ is called the **quadratic variation** of $X$. One frequently writes $\langle X, X \rangle\_t$ instead of $\langle X \rangle\_t$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">4.4 (Existence and Characterisation of Quadratic Variation)</span></p>

The following statements hold:

1. Every process $M \in \mathcal{M}\_2^C$ is of bounded quadratic variation.
2. The process $M^2 - \langle M, M \rangle$ is a martingale.
3. The process $V = \langle M, M \rangle$ is the *unique* process satisfying the following properties:
   1. $V$ is adapted to $(\mathcal{F}\_t)\_{t \ge 0}$,
   2. $V_t$ is continuous and non-decreasing,
   3. $V_0 = 0$,
   4. $M^2 - V$ is a martingale.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Roughness, Measured)</span></p>

* This theorem precisely characterises the "roughness" of continuous martingales via their quadratic variation: unlike processes of bounded variation, martingales accumulate variance *in a systematic way that can be measured and tracked* — this is what makes them viable integrators for Itô calculus.
* Note that if $M$ were of bounded variation, the integral $\int X_t\, \mathrm{d}M_t$ could be trivially defined path-wise via Riemann–Stieltjes. Our goal is precisely the broader, rougher class.
* Part 2 is the quantitative version of "variance accumulates": $\langle M \rangle$ is exactly the compensator that must be subtracted from $M^2$ (a submartingale) to restore the martingale property.
* $\langle M \rangle\_t$ will play the role of the *clock* of the integration theory: integrands are measured in the norm $\mathbb{E}\int_0^t X_s^2\, \mathrm{d}\langle M \rangle\_s$ (Itô isometry, Theorem 4.6).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 4.4</summary>

**Part 1.** First assume that $M$ is bounded, i.e., $\lvert M_t \rvert \le C$ a.s. for a constant $C > 0$. Fix $T > 0$ and consider a sequence of *nested* partitions $0 = t_0^n < t_1^n < \cdots < t_n^n = T$ such that

$$
\lbrace t_0^m, \dots, t_m^m \rbrace \subset \lbrace t_0^n, \dots, t_n^n \rbrace \quad \text{for all } m \le n, \qquad \max_{0 \le i \le n-1} \bigl(t_{i+1}^n - t_i^n\bigr) \to 0.
$$

Define the sum of squared increments

$$
V_n(T) := \sum_{i=0}^{n-1} \bigl(M_{t_{i+1}^n} - M_{t_i^n}\bigr)^2.
$$

We will show that

$$
V_n(T) \ \text{is a Cauchy sequence in } L^2(\mathbb{P}). \tag{*}
$$

This implies the existence of a limit $V(T) \in L^2(\mathbb{P})$ with $\mathbb{E}[(V_n(T) - V(T))^2] \to 0$; moreover, one can show that $V(T)$ does not depend on the choice of the partition sequence almost surely (merge two sequences into a common refinement). Hence (\*) implies part 1 for bounded $M$.

*Proof of (\*).* Take $m > n$ and refine: for each interval $[t_i^n, t_{i+1}^n]$, let

$$
t_i^n = t_{i,1}^m < t_{i,2}^m < \cdots < t_{i,J_i}^m = t_{i+1}^n
$$

be the subdivision induced by the finer partition $\lbrace t_j^m \rbrace$. Using the identity

$$
\Bigl(\sum_{k=1}^K c_k\Bigr)^2 - \sum_{k=1}^K c_k^2 = 2 \sum_{k=1}^K c_k \Bigl(\sum_{l=1}^{k-1} c_l\Bigr)
$$

on each block — with $c_j$ the fine increments inside block $i$, so that $\sum_{l < j} c_l = M_{t_{i,j}^m} - M_{t_i^n}$ telescopes — one computes

$$
V_n(T) - V_m(T) = \sum_{i=0}^{n-1} \Biggl[\bigl(M_{t_{i+1}^n} - M_{t_i^n}\bigr)^2 - \sum_{j=1}^{J_i - 1} \bigl(M_{t_{i,j+1}^m} - M_{t_{i,j}^m}\bigr)^2\Biggr] = 2 \sum_{i,j} \bigl(M_{t_{i,j+1}^m} - M_{t_{i,j}^m}\bigr)\bigl(M_{t_{i,j}^m} - M_{t_i^n}\bigr).
$$

Squaring and taking expectations, the *orthogonality of martingale increments* (together with boundedness of $M$) kills every term in which the four increments do not overlap suitably, leaving

$$
\mathbb{E}\bigl[(V_m(T) - V_n(T))^2\bigr] \le 4\, \mathbb{E}\Bigl[\Delta_n^2 \sum_{i=0}^{m-1} \bigl(M_{t_{i+1}^m} - M_{t_i^m}\bigr)^2\Bigr], \qquad \Delta_n := \max_{i,j} \bigl\lvert M_{t_{i,j}^m} - M_{t_i^n} \bigr\rvert.
$$

By continuity of $M$ we have $\Delta_n \to 0$ a.s. as $n \to \infty$ (uniform continuity on $[0,T]$), and $\lvert \Delta_n \rvert \le 2C$, so dominated convergence gives $\mathbb{E}[\Delta_n^4] \to 0$. For the second factor, write $Z_i := M_{t_{i+1}^m} - M_{t_i^m}$; then

$$
\mathbb{E}\Bigl[\Bigl(\sum_i Z_i^2\Bigr)^2\Bigr] = \mathbb{E}\Bigl[\sum_i Z_i^4\Bigr] + 2 \sum_{i < j} \mathbb{E}\bigl[Z_i^2 Z_j^2\bigr] \le 8 C^4,
$$

using $\mathbb{E}[Z_i^4] \le 4C^2\, \mathbb{E}[Z_i^2]$, the tower property with orthogonality of increments for the cross terms ($\mathbb{E}[\sum_{j>i} Z_j^2 \mid \mathcal{F}\_{t_{i+1}^m}] \le 4C^2$), and $\sum_i \mathbb{E}[Z_i^2] = \mathbb{E}[(M_T - M_0)^2] \le 4C^2$. Cauchy–Schwarz now combines the two factors:

$$
\mathbb{E}\bigl[(V_m(T) - V_n(T))^2\bigr] \le 4 \sqrt{\mathbb{E}[\Delta_n^4]}\, \sqrt{8 C^4} \longrightarrow 0,
$$

showing (\*) and proving part 1 for bounded $M$.

For processes $M$ that are not bounded, consider the stopped processes

$$
M_t^N = M_{t \wedge \tau_N}, \qquad \tau_N = \inf \lbrace t : \lvert M_t \rvert \ge N \rbrace,
$$

which are bounded continuous martingales. The first part yields, for partitions $0 = t_0^n < \cdots < t_n^n = T$,

$$
\Delta_T^{N,n} := \sum_{i=0}^{n-1} \bigl(M_{t_{i+1}^n}^N - M_{t_i^n}^N\bigr)^2 \to \langle M^N \rangle_T \quad \text{in } L^2(\mathbb{P}).
$$

On the event $\lbrace T \le \tau_N \rbrace$ we have $\Delta_T^{N,n} = \Delta_T^{N+k,n}$ for all $k \ge 0$, hence $\langle M^{N+k} \rangle\_T$ does not depend on $k \ge 0$ on $\lbrace T \le \tau_N \rbrace$, a.s. Since $\tau_N \uparrow \infty$, there exists a process $\Delta_T$ such that

$$
\sum_{i=0}^{n-1} \bigl(M_{t_{i+1}^n} - M_{t_i^n}\bigr)^2 \to \Delta_T \quad \text{in probability},
$$

which shows part 1 for arbitrary $M \in \mathcal{M}\_2^C$.

**Part 2.** Let $0 = t_0^n < t_1^n < \cdots$ be nested partitions with mesh $\to 0$, and define

$$
R_t^n := \sum_{i=0}^{k_n - 1} \bigl(M_{t_{i+1}^n} - M_{t_i^n}\bigr)^2 + \bigl(M_t - M_{t_{k_n}^n}\bigr)^2,
$$

where $k_n$ is chosen so that $t_{k_n}^n \le t < t_{k_n + 1}^n$. By arguments as in part 1 (orthogonality of increments), one finds for $s < t$ that

$$
\mathbb{E}\bigl[R_t^n - R_s^n \mid \mathcal{F}_s\bigr] = \mathbb{E}\bigl[M_t^2 - M_s^2 \mid \mathcal{F}_s\bigr],
\qquad\text{i.e.}\qquad
\mathbb{E}\bigl[M_t^2 - R_t^n \mid \mathcal{F}_s\bigr] = M_s^2 - R_s^n.
$$

By part 1, $R_t^n \to \langle M \rangle\_t$; for bounded $M$ the convergence holds in $L^2$, which suffices to pass to the limit inside the conditional expectation, and the general case follows by localisation. Hence $\mathbb{E}[M_t^2 - \langle M \rangle\_t \mid \mathcal{F}\_s] = M_s^2 - \langle M \rangle\_s$, i.e., $M^2 - \langle M \rangle$ is a martingale.

**Part 3.** Suppose that both $M^2 - V$ and $M^2 - Z$ are martingales, where $V$ and $Z$ are continuous, non-decreasing, and adapted. Then the difference $V - Z$ is an adapted martingale with continuous paths of *bounded variation* (a difference of monotone processes), started at $0$. Statement 3 then follows immediately from the following classical lemma. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">4.5 (Continuous Martingales of Bounded Variation Are Constant)</span></p>

Every martingale $N$ with continuous paths of bounded variation is a.s. constant.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 4.5</summary>

Without loss of generality, suppose $N_0 = 0$. Let $V_t$ denote the total variation process of $N$ on $[0, t]$, and for each $n \in \mathbb{N}$ define the stopping time

$$
S_n := \inf \bigl\lbrace t \ge 0 \,\big|\, V_t \ge n \bigr\rbrace.
$$

Set $X_t := N_{t \wedge S_n}$. Since $N$ is continuous, $X$ is a stopped martingale — hence itself a martingale — with continuous paths whose total variation is bounded by $n$ (so in particular $\lvert X_t \rvert \le n$).

Now take any partition $0 = s_0 < s_1 < \cdots < s_k = t$ with mesh $\to 0$. By orthogonality of martingale increments,

$$
\mathbb{E}\bigl[X_t^2\bigr] = \sum_{i=0}^{k-1} \mathbb{E}\Bigl[\bigl(X_{s_{i+1}} - X_{s_i}\bigr)^2\Bigr] \le \mathbb{E}\Bigl[\max_i \bigl\lvert X_{s_{i+1}} - X_{s_i} \bigr\rvert \cdot \underbrace{\sum_i \bigl\lvert X_{s_{i+1}} - X_{s_i} \bigr\rvert}_{\le\, V_{t \wedge S_n}\, \le\, n}\Bigr] \le n\, \mathbb{E}\Bigl[\max_i \bigl\lvert X_{s_{i+1}} - X_{s_i} \bigr\rvert\Bigr].
$$

By continuity, $\max_i \lvert X_{s_{i+1}} - X_{s_i} \rvert \to 0$ a.s. as the mesh vanishes, and it is bounded by $2n$; dominated convergence makes the right-hand side vanish. Hence $X_t = 0$ a.s. for every $t$ — the *bounded variation* budget is too small to sustain any *quadratic* variation, and a continuous martingale without quadratic variation cannot move. As $n$ was arbitrary, letting $n \to \infty$ yields that $N$ is a.s. constant. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Quadratic Variation of Brownian Motion)</span></p>

A paramount example is the standard Brownian motion $W$. Since it is a well-known fact that $W_t^2 - t$ is a martingale, the uniqueness statement of Theorem 4.4 (part 3, with $V_t = t$: adapted, continuous, non-decreasing, $V_0 = 0$) immediately dictates that

$$
\langle W \rangle_t = t.
$$

Brownian motion accumulates variance at unit rate — its quadratic variation is *deterministic*, even though the path is random.

</div>

<figure>
  <img src="{{ 'assets/images/notes/sdes_diffusion_models/qv_ito_correction.png' | relative_url }}" alt="Two panels. Left: staircase curves of cumulative squared increments of one Brownian path for three partition resolutions; the coarse staircase is jagged, the fine one hugs the diagonal line t. Right: three curves for the same path — the squared Brownian motion, the stochastic integral two times integral of W dW, and their difference, which follows the straight line t almost exactly." loading="lazy">
  <figcaption>One Brownian path, two readings of $\langle W\rangle_t = t$. <strong>Left.</strong> The sums of squared increments $V_n(t)$ (Definition 4.3) over dyadic partitions with $n = 2^3, 2^6, 2^{12}$ points converge to the deterministic line $t$. <strong>Right.</strong> The Itô correction previewed: $W_t^2$ and the stochastic integral $2\int_0^t W_s\,\mathrm{d}W_s$ differ by exactly the quadratic variation — the difference curve is glued to $t$. This is Itô's rule (Theorem 4.15) for $f(x)=x^2$, and equivalently the statement of Theorem 4.4 part 2 that $W_t^2 - t$ is a martingale.</figcaption>
</figure>

With this variance tracker $\langle M \rangle$ established, we can greatly enlarge our class of permissible integrands. We define the space $\mathcal{L}^{\ast}$ as

$$
\mathcal{L}^{\ast} := \biggl\lbrace (X_t, \mathcal{F}_t) \ \text{progressively measurable with}\ \mathbb{E}\Bigl[\int_0^t X_s^2\, \mathrm{d}\langle M \rangle_s\Bigr] < \infty \ \text{for all } t \ge 0 \biggr\rbrace.
$$

Here, the integral $\int_0^t X_s^2\, \mathrm{d}\langle M \rangle\_s$ is defined path-wise as a standard Lebesgue–Stieltjes integral, since $\langle M \rangle$ is non-decreasing. To establish the existence of stochastic integrals $\int X_s\, \mathrm{d}M_s$ for this larger class $\mathcal{L}^{\ast}$, we rely on the fundamental *Itô isometry* and a *density argument*, outlined in the following two theorems.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">4.6 (Itô Isometry for Simple Processes)</span></p>

For $X \in \mathcal{L}\_0$ and $M \in \mathcal{M}\_2^C$, it holds that

$$
\mathbb{E}\Biggl[\biggl(\int_0^t X_s\, \mathrm{d}M_s\biggr)^{\!2}\Biggr] = \mathbb{E}\Biggl[\int_0^t X_s^2\, \mathrm{d}\langle M \rangle_s\Biggr].
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 4.6</summary>

Let the simple process be given by $X_s = \xi_0 \mathbf{1}\_{\lbrace 0 \rbrace}(s) + \sum_{i=0}^\infty \xi_i \mathbf{1}\_{(t_i, t_{i+1}]}(s)$, and assume without loss of generality that $t = t_n$ for some $n \in \mathbb{N}$. Then, by definition,

$$
\int_0^t X_s\, \mathrm{d}M_s = \sum_{i=0}^{n-1} \xi_i \bigl(M_{t_{i+1}} - M_{t_i}\bigr).
$$

Squaring this sum and taking the expectation, the cross-terms vanish under expectation due to the martingale property of $M$ (condition on $\mathcal{F}\_{t_j}$ for the later increment). We are left with the diagonal terms:

$$
\mathbb{E}\Biggl[\biggl(\int_0^t X_s\, \mathrm{d}M_s\biggr)^{\!2}\Biggr]
= \mathbb{E}\Biggl[\sum_{i=0}^{n-1} \xi_i^2 \bigl(M_{t_{i+1}} - M_{t_i}\bigr)^2\Biggr]
= \mathbb{E}\Biggl[\sum_{i=0}^{n-1} \xi_i^2 \bigl(\langle M \rangle_{t_{i+1}} - \langle M \rangle_{t_i}\bigr)\Biggr]
= \mathbb{E}\Biggl[\int_0^t X_s^2\, \mathrm{d}\langle M \rangle_s\Biggr],
$$

where the middle equality uses that $M^2 - \langle M \rangle$ is a martingale (Theorem 4.4, part 2), so conditionally on $\mathcal{F}\_{t_i}$ the squared increment of $M$ and the increment of $\langle M \rangle$ have the same expectation. $\square$

</details>
</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(What the Isometry Buys)</span></p>

The map $X \mapsto \int_0^{\cdot} X\, \mathrm{d}M$ is an **isometry** from (a dense subspace of) the Hilbert-type space with norm $\mathbb{E}\int X^2\, \mathrm{d}\langle M \rangle$ into the space of martingales $\mathcal{M}\_2^C$. This is the exact analogue of extending the Riemann integral from step functions to $L^2$: once an isometry is established on a *dense* class (Theorem 4.7) and the target space is *complete* (Theorem 4.8), the integral extends uniquely by continuity. The next results supply precisely these two missing ingredients.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">4.7 (Density of Simple Processes)</span></p>

For any process $X \in \mathcal{L}^{\ast}$, there exists a sequence of simple processes $X^n \in \mathcal{L}\_0$ such that

$$
\mathbb{E}\Biggl[\int_0^t \bigl(X_s - X_s^n\bigr)^2\, \mathrm{d}\langle M \rangle_s\Biggr] \to 0 \quad \text{for all } t \ge 0. \tag{4.7}
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 4.7</summary>

We prove this strictly for the case $\langle M \rangle\_t = t$, which corresponds to standard Brownian motion; for the general semimartingale case, we refer to standard texts such as Karatzas and Shreve (1998). We want to show (4.7) for a fixed $t$, implying the existence of $X_s^n$ such that $\mathbb{E} \int_0^t (X_s - X_s^n)^2\, \mathrm{d}s \le 1/n$.

**Case 1: $X$ is continuous and bounded.** Define the piecewise constant approximation

$$
X_s^n = X_0 \mathbf{1}_{\lbrace 0 \rbrace}(s) + \sum_{k=0}^{2^n - 1} X_{k t / 2^n}\, \mathbf{1}_{(k t / 2^n,\, (k+1) t / 2^n]}(s).
$$

Clearly, $X^n \in \mathcal{L}\_0$. Because $X$ is continuous, $X_s^n \to X_s$ pointwise; since $X$ is bounded, (4.7) follows directly by dominated convergence.

**Case 2: $X$ is only bounded.** We employ a smoothing argument. Define the local average

$$
\widetilde{X}_s^m = m \bigl[F_s - F_{(s - 1/m)^+}\bigr], \qquad \text{where } F_s := \int_0^{s \wedge t} X_u(\omega)\, \mathrm{d}u.
$$

By construction, $\widetilde{X}\_s^m$ is $\mathcal{F}\_s$-adapted and continuous (hence progressively measurable). As established in Case 1, there exist simple processes $\widetilde{X}\_s^{m,n} \in \mathcal{L}\_0$ with

$$
\mathbb{E}\Biggl[\int_0^t \bigl(\widetilde{X}_s^{m,n} - \widetilde{X}_s^m\bigr)^2 \mathrm{d}s\Biggr] \to 0 \quad \text{as } n \to \infty.
$$

We must now demonstrate that the smoothed process converges to the original process,

$$
\mathbb{E}\Biggl[\int_0^t \bigl(\widetilde{X}_s^m - X_s\bigr)^2 \mathrm{d}s\Biggr] \to 0 \quad \text{as } m \to \infty. \tag{4.8}
$$

If true, we can choose a suitable diagonal sequence $m_n \to \infty$ such that setting $X^n = \widetilde{X}^{m_n, n}$ satisfies (4.7) for bounded $X$. To prove (4.8), define the exceptional set of times for a given path $\omega$:

$$
A_\omega := \Bigl\lbrace s \in [0, t] \,:\, \lim_{m \to \infty} \widetilde{X}_s^m(\omega) \ne X_s(\omega) \Bigr\rbrace.
$$

By Lebesgue's differentiation theorem (the fundamental theorem of calculus for merely integrable integrands), the Lebesgue measure of this set is zero, $\lambda(A_\omega) = 0$. Lifting this to the product space, define $A := \lbrace (s, \omega) : s \in A_\omega \rbrace$; by Fubini's theorem,

$$
(\lambda \times \mathbb{P})(A) = \int_\Omega \lambda(A_\omega)\, \mathbb{P}(\mathrm{d}\omega) = 0.
$$

Consequently, $\widetilde{X}\_s^m \to X_s$ almost everywhere on $[0,t] \times \Omega$; dominated convergence then immediately yields (4.8).

**Case 3: $X \in \mathcal{L}^{\ast}$ is arbitrary.** For an arbitrary integrand, we truncate: let $X_s^k := X_s \mathbf{1}\_{\lbrace \lvert X_s \rvert \le k \rbrace}$. Then

$$
\mathbb{E}\Biggl[\int_0^t \bigl(X_s^k - X_s\bigr)^2 \mathrm{d}s\Biggr] \to 0 \quad \text{as } k \to \infty
$$

by dominated convergence (the integrand is dominated by $X_s^2$, which is integrable by definition of $\mathcal{L}^{\ast}$). Combining this with the results from Case 2 via a final diagonal sequence argument completes the proof. $\square$

</details>
</div>

Consider now $X \in \mathcal{L}^{\ast}$. According to Theorem 4.7, we can choose $X^n \in \mathcal{L}\_0$ satisfying (4.7), and define

$$
I_t(X) := \lim_{n \to \infty} \int_0^t X_s^n\, \mathrm{d}M_s \quad \text{a.s.}
$$

This limit is a.s. well-defined by the following completeness argument: for a sequence $Z_n$ of random variables,

$$
\mathbb{E}\bigl[(Z_n - Z_m)^2\bigr] \to 0 \ \text{ for all } m > n \to \infty \quad \Longrightarrow \quad \exists\, Z \ \text{with}\ \mathbb{E}\bigl[(Z_n - Z)^2\bigr] \to 0 \ \text{as } n \to \infty,
$$

and the sequence $\int X^n \mathrm{d}M$ is Cauchy in $L^2$ precisely by the Itô isometry applied to the differences $X^n - X^m$. Up to $\mathbb{P}$-null sets, $I_t(X)$ does not depend on the choice of the approximating sequence $X^n$ (exercise). Below, we will use one particular choice of $I_t(X)$ as *the* definition of the stochastic integral $\int_0^t X_s\, \mathrm{d}M_s$ for $X \in \mathcal{L}^{\ast}$; for this choice, we make use of the following two theorems.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">4.8 (Completeness of $\mathcal{M}\_2^C$)</span></p>

For $M \in \mathcal{M}\_2^C$, define

$$
\vert\kern-0.25ex\vert\kern-0.25ex\vert M \vert\kern-0.25ex\vert\kern-0.25ex\vert_t := \sqrt{\mathbb{E}\bigl[M_t^2\bigr]}, \qquad \vert\kern-0.25ex\vert\kern-0.25ex\vert M \vert\kern-0.25ex\vert\kern-0.25ex\vert := \sum_{n=1}^\infty \frac{\vert\kern-0.25ex\vert\kern-0.25ex\vert M \vert\kern-0.25ex\vert\kern-0.25ex\vert_n \wedge 1}{2^n}.
$$

Then $\bigl(\mathcal{M}\_2^C, \vert\kern-0.25ex\vert\kern-0.25ex\vert \cdot \vert\kern-0.25ex\vert\kern-0.25ex\vert\bigr)$ is a complete metric space.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 4.8</summary>

Let $(M^n)$ be a Cauchy sequence in $\mathcal{M}\_2^C$. Then, for each $t$, $M_t^n$ is Cauchy in $L^2(\Omega, \mathcal{F}\_t, \mathbb{P})$, so there exists $\widetilde{M}\_t \in L^2(\Omega, \mathcal{F}\_t, \mathbb{P})$ with $\mathbb{E}\bigl[(M_t^n - \widetilde{M}\_t)^2\bigr] \to 0$. For $A \in \mathcal{F}\_s$, by Cauchy–Schwarz,

$$
\mathbb{E}\bigl[\mathbf{1}_A \bigl(M_t^n - \widetilde{M}_t\bigr)\bigr] \le \sqrt{\mathbb{P}(A)} \cdot \Bigl(\mathbb{E}\bigl[(M_t^n - \widetilde{M}_t)^2\bigr]\Bigr)^{1/2} \to 0,
$$

so $\mathbb{E}[\mathbf{1}\_A M_t^n] \to \mathbb{E}[\mathbf{1}\_A \widetilde{M}\_t]$ and likewise $\mathbb{E}[\mathbf{1}\_A M_s^n] \to \mathbb{E}[\mathbf{1}\_A \widetilde{M}\_s]$; passing to the limit in the martingale property shows $\widetilde{M}$ is a martingale. Using that if $X_t$ is a martingale w.r.t. $\mathcal{F}\_t$, then $X_t = \mathbb{E}[X_{t+} \mid \mathcal{F}\_t]$ a.s., hence $X_{t+}$ is a martingale w.r.t. $\mathcal{F}\_{t+}$, and since $\mathcal{F}\_t = \mathcal{F}\_{t+}$ (right-continuity of the filtration), we may set $M_t := \widetilde{M}\_{t+}$; then $M_t = \widetilde M_t$ a.s. and $M$ is right-continuous.

It remains to prove that $M$ is also continuous from the left. From Doob's maximal inequality,

$$
\mathbb{P}\Bigl(\sup_{0 \le s \le t} \bigl\lvert M_s^n - M_s \bigr\rvert \ge \varepsilon\Bigr) \le \frac{1}{\varepsilon^2}\, \mathbb{E}\bigl\lvert M_t^n - M_t \bigr\rvert^2 \to 0
$$

as $n \to \infty$. Passing to suitable subsequences $n_k$, we obtain

$$
\mathbb{P}\Bigl(\sup_{0 \le s \le t} \bigl\lvert M_s^{n_k} - M_s \bigr\rvert \ge \tfrac{1}{k}\Bigr) \le 2^{-k},
$$

and by the Borel–Cantelli lemma, $\sup_{0 \le s \le t} \lvert M_s^{n_k} - M_s \rvert \to 0$ a.s. — the convergence is *uniform* on compacts along the subsequence. A uniform limit of continuous paths is continuous, so $M_s$ is continuous in $s$ almost surely. Hence $M \in \mathcal{M}\_2^C$ and the space is complete. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">4.9 (Simple Integrals Live in $\mathcal{M}_2^C$)</span></p>

For $X \in \mathcal{L}\_0$ and $M \in \mathcal{M}\_2^C$, it holds that

$$
\int_0^t X_s\, \mathrm{d}M_s \in \mathcal{M}_2^C.
$$

</div>

Indeed: for simple $X$ the integral is an explicit martingale transform of $M$ with bounded, non-anticipating stakes, so it inherits the martingale property and path continuity from $M$; the uniform $L^2$-bound follows from the Itô isometry together with (4.6). We conclude from Theorems 4.8 and 4.9 that the following definition is well-posed.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">4.10 (Stochastic Integral on $\mathcal{L}^{\ast}$)</span></p>

For $X \in \mathcal{L}^{\ast}$ and $M \in \mathcal{M}\_2^C$, we define the stochastic integral

$$
I_t(X) := \int_0^t X_s\, \mathrm{d}M_s
$$

as the unique element of $\mathcal{M}\_2^C$ such that $\vert\kern-0.25ex\vert\kern-0.25ex\vert I(X) - I(X^n) \vert\kern-0.25ex\vert\kern-0.25ex\vert \to 0$ for every sequence $(X^n) \subset \mathcal{L}\_0$ with $\mathbb{E} \int_0^t (X_s - X_s^n)^2\, \mathrm{d}\langle M \rangle\_s \to 0$.

</div>

**Extension to local martingales.** We now generalise stochastic integration to broader classes of integrators and integrands. Let $\mathcal{M}^{c,\mathrm{loc}}$ denote the class of continuous **local martingales** — processes $M$ for which there exists a localising sequence of stopping times $T_n \uparrow \infty$ such that each stopped process $M_{\cdot \wedge T_n}$ is a martingale — and define

$$
\mathcal{P}^{\ast} := \biggl\lbrace (X_t, \mathcal{F}_t) \ \text{progressively measurable} \,:\, \int_0^t X_s^2\, \mathrm{d}\langle M \rangle_s < +\infty \ \text{a.s. for all } t \biggr\rbrace.
$$

Note the relaxation relative to $\mathcal{L}^{\ast}$: only *almost-sure* finiteness of the accumulated variance is required, no expectations. Choose $M \in \mathcal{M}^{c,\mathrm{loc}}$ and $X \in \mathcal{P}^{\ast}$. Let $R_n$ and $S_n$ be non-decreasing stopping times defined by

$$
R_n(\omega) := n \wedge \inf \biggl\lbrace 0 \le t < \infty \,:\, \int_0^t X_s^2(\omega)\, \mathrm{d}\langle M \rangle_s(\omega) \ge n \biggr\rbrace, \qquad M_{t \wedge S_n} \in \mathcal{M}_2^C.
$$

Set $T_n := R_n \wedge S_n$, and define the processes

$$
M_t^n(\omega) := M_{t \wedge T_n(\omega)}(\omega), \qquad X_t^n(\omega) := X_t(\omega) \cdot \mathbf{1}_{\lbrace T_n(\omega) \ge t \rbrace}.
$$

By construction $X^n \in \mathcal{L}^{\ast}$ and $M^n \in \mathcal{M}\_2^C$, so $\int X^n \mathrm{d}M^n$ is defined by Definition 4.10. The point is that these localised integrals are *consistent*:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">4.11 (Consistency of Localised Integrals; Karatzas–Shreve, Cor. 3.2.21)</span></p>

For $0 \le t \le T_n$ and $n \le m$, it holds almost surely that

$$
\int_0^t X_s^n\, \mathrm{d}M_s^n = \int_0^t X_s^m\, \mathrm{d}M_s^m.
$$

</div>

The lemma allows the following definition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">4.12 (Stochastic Integral for Local Martingales)</span></p>

Let $M \in \mathcal{M}^{c,\mathrm{loc}}$, $X \in \mathcal{P}^{\ast}$. Then the stochastic integral $\int_0^t X_s\, \mathrm{d}M_s$ is defined as the unique process in $\mathcal{M}^{c,\mathrm{loc}}$ satisfying

$$
\int_0^t X_s\, \mathrm{d}M_s = \int_0^t X_s^n\, \mathrm{d}M_s^n \quad \text{for } 0 \le t \le T_n.
$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">4.13 (Multivariate Integrals)</span></p>

Let $M^1, \dots, M^k \in \mathcal{M}^{c,\mathrm{loc}}$ and $X^{ij} \in \mathcal{P}^{\ast}$ for $1 \le i \le m$, $1 \le j \le k$. Define the vector-valued stochastic integral

$$
\int X_s\, \mathrm{d}M_s := \begin{pmatrix} \int X_s^{11}\, \mathrm{d}M_s^1 + \cdots + \int X_s^{1k}\, \mathrm{d}M_s^k \\ \vdots \\ \int X_s^{m1}\, \mathrm{d}M_s^1 + \cdots + \int X_s^{mk}\, \mathrm{d}M_s^k \end{pmatrix}.
$$

</div>

For the following processes, it is now easy to define integrals.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">4.14 (Continuous Semimartingale)</span></p>

Suppose a process $(X_t, \mathcal{F}\_t)$ admits the decomposition

$$
X_t = X_0 + M_t + B_t, \qquad 0 \le t < \infty,
$$

with $M \in \mathcal{M}^{c,\mathrm{loc}}$ and $B_t = B_t^{+} - B_t^{-}$, where $B_t^{\pm}$ are continuous, non-decreasing, and $\mathcal{F}\_t$-adapted with $B_0^{\pm} = 0$ a.s. Then $X$ is called a **continuous semimartingale**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Integrals Against Semimartingales — and a Notation Warning)</span></p>

For a continuous semimartingale $X$ and $Y \in \mathcal{P}^{\ast}$, the integral $\int Y_s\, \mathrm{d}X_s$ is defined by

$$
\int Y_s\, \mathrm{d}X_s := \int Y_s\, \mathrm{d}M_s + \int Y_s\, \mathrm{d}B_s,
$$

where the second integral is a path-wise Lebesgue–Stieltjes integral against the bounded-variation part. *Mind the notation clash:* from Definition 4.14 onwards, $B$ denotes the **bounded-variation part** of a semimartingale — not a Brownian motion, which we will denote by $W$ in the remainder of this chapter.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Summary</span><span class="math-callout__name">(The Construction Ladder)</span></p>

The stochastic integral was built in four rungs, each trading explicitness for generality:

1. **Simple processes $\mathcal{L}\_0$** (Definition 4.2): explicit martingale-transform sums — the "gambling winnings".
2. **$\mathcal{L}^{\ast}$** (Definition 4.10): extension by continuity — the Itô *isometry* (Theorem 4.6) maps a *dense* class (Theorem 4.7) into a *complete* space (Theorem 4.8). Structurally identical to extending $\int$ from step functions to $L^2$.
3. **$\mathcal{P}^{\ast}$ against local martingales** (Definition 4.12): *localisation* by stopping times removes all expectation requirements — only a.s.-finite accumulated variance $\int X^2\, \mathrm{d}\langle M \rangle$ remains.
4. **Semimartingales** (Definition 4.14 + Remark): integrate against the martingale part by rung 3 and against the bounded-variation part path-wise.

Quadratic variation is the fuel gauge of the whole construction: it decides which integrands are admissible and what the integral's variance is.

</div>

### 4.2 The Itô Rule

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">4.15 (Itô's Rule)</span></p>

Let $f : \mathbb{R} \to \mathbb{R}$ be twice continuously differentiable, and let $X_t = X_0 + M_t + B_t$ be a continuous semimartingale. Then, for all $t \ge 0$, it holds almost surely that

$$
f(X_t) = f(X_0) + \int_0^t f'(X_s)\, \mathrm{d}X_s + \frac{1}{2} \int_0^t f''(X_s)\, \mathrm{d}\langle M \rangle_s.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Second-Order Correction Term)</span></p>

Note the additional term involving the second derivative,

$$
\frac{1}{2} \int_0^t f''(X_s)\, \mathrm{d}\langle M \rangle_s,
$$

which has no counterpart in the classical chain rule. The heuristic: over a short interval, $(\mathrm{d}X)^2 \approx (\mathrm{d}M)^2 \approx \mathrm{d}\langle M \rangle$ is of order $\mathrm{d}t$ — *not* negligible — so the second-order Taylor term survives the limit. The classical chain rule is recovered exactly when $\langle M \rangle \equiv 0$, i.e., for bounded-variation paths (Lemma 4.5 territory). Two consequences worth internalising:

* **This is where the generator's $\frac{1}{2}$ comes from.** For an SDE solution ($\mathrm{d}\langle M \rangle\_s = a(X_s)\, \mathrm{d}s$), taking expectations in Itô's rule yields $\frac{\mathrm{d}}{\mathrm{d}t} \mathbb{E}[f(X_t)] = \mathbb{E}\bigl[\frac{1}{2} a f'' + b f'\bigr] (X_t)$ — precisely the differential operator $L(x, D)$ of Definition 3.1 and the Brownian generator $\frac{1}{2}\Delta$ of Example 2.14. Itô's rule is the path-wise engine behind the entire semigroup calculus of Chapters 2–3.
* The right panel of the figure above shows the correction path-wise for $f(x) = x^2$: $W_t^2 - 2\int_0^t W\, \mathrm{d}W = \langle W \rangle\_t = t$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 4.15</summary>

**Step 1 (Localisation).** Define

$$
T_n := \begin{cases} 0, & \text{if } \lvert X_0 \rvert \ge n, \\ \inf \bigl\lbrace t \ge 0 : \lvert M_t \rvert \ge n \ \text{or}\ B_t^{+} + B_t^{-} \ge n \ \text{or}\ \langle M \rangle_t \ge n \bigr\rbrace, & \text{if } \lvert X_0 \rvert < n, \end{cases}
$$

with $\inf \varnothing := \infty$, and put $X_t^{(n)} := X_{t \wedge T_n}$. We show the Itô rule for $X_t^{(n)}$; letting $n \to \infty$ then implies the theorem. Note that $X_t^{(n)}$, $M_t^{(n)}$, $\langle M^{(n)} \rangle\_t$, $B_t^{+(n)}$, and $B_t^{-(n)}$ are all bounded, so without loss of generality we may also assume that $f$, $f'$, and $f''$ are absolutely bounded. From now on, we drop the index $(n)$.

**Step 2 (Taylor expansion).** For a partition $0 = t_0 < \cdots < t_m = t$ with mesh $\to 0$, Taylor's theorem gives

$$
f(X_t) - f(X_0) = \sum_{k=1}^m \bigl(f(X_{t_k}) - f(X_{t_{k-1}})\bigr) = \sum_{k=1}^m f'(X_{t_{k-1}}) \bigl(X_{t_k} - X_{t_{k-1}}\bigr) + \frac{1}{2} \sum_{k=1}^m f''(\eta_k) \bigl(X_{t_k} - X_{t_{k-1}}\bigr)^2,
$$

where $\eta_k$ lies between $X_{t_{k-1}}$ and $X_{t_k}$. Splitting $X = X_0 + B + M$ and expanding the square,

$$
\begin{aligned}
f(X_t) - f(X_0)
&= \sum_{k} f'(X_{t_{k-1}}) \bigl(B_{t_k} - B_{t_{k-1}}\bigr) + \sum_{k} f'(X_{t_{k-1}}) \bigl(M_{t_k} - M_{t_{k-1}}\bigr) \\
&\quad + \frac{1}{2} \sum_{k} f''(\eta_k) \bigl(B_{t_k} - B_{t_{k-1}}\bigr)^2 + \sum_{k} f''(\eta_k) \bigl(B_{t_k} - B_{t_{k-1}}\bigr)\bigl(M_{t_k} - M_{t_{k-1}}\bigr) \\
&\quad + \frac{1}{2} \sum_{k} f''(\eta_k) \bigl(M_{t_k} - M_{t_{k-1}}\bigr)^2 =: J_1 + J_2 + J_3 + J_4 + J_5.
\end{aligned}
$$

For $\max_k \lvert t_k - t_{k-1} \rvert \to 0$:

* $J_1 \to \int_0^t f'(X_s)\, \mathrm{d}B_s$ a.s. (Riemann–Stieltjes sums against the BV part) and $J_2 \to \int_0^t f'(X_s)\, \mathrm{d}M_s$ a.s., since $f'(X_t)$ is $\mathcal{F}\_t$-adapted, continuous, and bounded.
* The mixed and pure-BV second-order terms vanish:

  $$
  \lvert J_3 \rvert \le \frac{1}{2} \max \lvert f'' \rvert \cdot \underbrace{\max_k \bigl\lvert B_{t_k} - B_{t_{k-1}} \bigr\rvert}_{\to\, 0 \ \text{a.s.}} \cdot \underbrace{\sum_k \bigl\lvert B_{t_k} - B_{t_{k-1}} \bigr\rvert}_{\le\, B_t^{+} + B_t^{-}\, \le\, 2n} \longrightarrow 0 \quad \text{a.s.},
  $$

  and by the same argument (Cauchy–Schwarz across the two factors), $J_4 \to 0$ a.s.
* For the crucial term $J_5$, we show

  $$
  \lvert J_5 - J_6 \rvert \to 0, \tag{4.9}
  $$

  where $J_6 := \frac{1}{2} \sum_k f''(X_{t_{k-1}}) \bigl(M_{t_k} - M_{t_{k-1}}\bigr)^2$ replaces $\eta_k$ by the left endpoint. Indeed,

  $$
  \lvert J_5 - J_6 \rvert \le \frac{1}{2} \sum_k \bigl(M_{t_k} - M_{t_{k-1}}\bigr)^2 \cdot \max_k \bigl\lvert f''(\eta_k) - f''(X_{t_{k-1}}) \bigr\rvert,
  $$

  which converges to zero because the first factor is bounded (in probability) by $\langle M \rangle\_t \le n$ while the second converges to zero by continuity of $X$ and (uniform) continuity of $f''$.
* Next, put $J_7 := \frac{1}{2} \sum_k f''(X_{t_{k-1}}) \bigl(\langle M \rangle\_{t_k} - \langle M \rangle\_{t_{k-1}}\bigr)$ — the same sum with the squared increments replaced by quadratic-variation increments. We claim

  $$
  \lvert J_6 - J_7 \rvert \to 0 \quad \text{a.s. (along a subsequence)}. \tag{4.10}
  $$

  Since $M^2 - \langle M \rangle$ is a martingale (Theorem 4.4), the increments $(M_{t_k} - M_{t_{k-1}})^2 - (\langle M \rangle\_{t_k} - \langle M \rangle\_{t_{k-1}})$ are pairwise orthogonal given the left endpoints, so the non-diagonal terms cancel in expectation:

  $$
  4\, \mathbb{E}\bigl[\lvert J_6 - J_7 \rvert^2\bigr] = \mathbb{E}\Biggl[\sum_k f''(X_{t_{k-1}})^2 \Bigl(\bigl(M_{t_k} - M_{t_{k-1}}\bigr)^2 - \bigl(\langle M \rangle_{t_k} - \langle M \rangle_{t_{k-1}}\bigr)\Bigr)^2\Biggr],
  $$

  and hence

  $$
  4\, \mathbb{E}\bigl[\lvert J_6 - J_7 \rvert^2\bigr] \le 2 \max \lvert f'' \rvert^2 \Biggl(\mathbb{E} \sum_k \bigl(M_{t_k} - M_{t_{k-1}}\bigr)^4 + \mathbb{E} \sum_k \bigl(\langle M \rangle_{t_k} - \langle M \rangle_{t_{k-1}}\bigr)^2\Biggr) \to 0,
  $$

  where the last convergence follows by dominated convergence (both sums are bounded and their maximal summands vanish by continuity).
* Finally, $J_7 \to \frac{1}{2} \int_0^t f''(X_s)\, \mathrm{d}\langle M \rangle\_s$: this is an ordinary Lebesgue–Stieltjes limit, since $\langle M \rangle\_t$ is of bounded variation.

Combining the limits of $J_1, J_2, J_7$ and the vanishing of the error terms, we obtain, for each fixed $t \ge 0$ a.s.,

$$
f(X_t) = f(X_0) + \int_0^t f'(X_s)\, \mathrm{d}X_s + \frac{1}{2} \int_0^t f''(X_s)\, \mathrm{d}\langle M \rangle_s.
$$

Because both sides are continuous in $t$, the identity holds a.s. *simultaneously* for all $t \ge 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">4.16 (Multivariate Itô Rule)</span></p>

Suppose that $M_t^1, \dots, M_t^d \in \mathcal{M}^{c,\mathrm{loc}}$ are $\mathcal{F}\_t$-adapted processes and that $B_t^1, \dots, B_t^d$ are $\mathcal{F}\_t$-adapted processes of bounded variation with $B_0^j = 0$. Define $M_t = (M_t^1, \dots, M_t^d)^{\!\top}$, $B_t = (B_t^1, \dots, B_t^d)^{\!\top}$, and put

$$
X_t = X_0 + M_t + B_t,
$$

with $X_0 \in \mathbb{R}^d$, $\mathcal{F}\_0$-measurable. Suppose that $f(t, x) : [0, \infty) \times \mathbb{R}^d \to \mathbb{R}$ has one continuous derivative with respect to $t$ and two continuous derivatives with respect to $x$. Then a.s.,

$$
\begin{aligned}
f(t, X_t) = f(0, X_0) &+ \int_0^t \frac{\partial}{\partial s} f(s, X_s)\, \mathrm{d}s + \sum_{i=1}^d \int_0^t \frac{\partial}{\partial x_i} f(s, X_s)\, \mathrm{d}B_s^i \\
&+ \sum_{i=1}^d \int_0^t \frac{\partial}{\partial x_i} f(s, X_s)\, \mathrm{d}M_s^i + \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d \int_0^t \frac{\partial^2}{\partial x_i \partial x_j} f(s, X_s)\, \mathrm{d}\langle M^i, M^j \rangle_s,
\end{aligned}
$$

for $0 \le t < \infty$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">4.17 (Quadratic Covariation)</span></p>

For two processes $M^1, M^2 \in \mathcal{M}^{c,\mathrm{loc}}$, the term

$$
\langle M^1, M^2 \rangle = \frac{1}{4} \Bigl(\bigl\langle M^1 + M^2 \bigr\rangle - \bigl\langle M^1 - M^2 \bigr\rangle\Bigr)
$$

is called the **quadratic covariation**. We also write $\langle M, M \rangle$ for $\langle M \rangle$, and $\langle X^1, X^2 \rangle$ for $\langle M^1, M^2 \rangle$ if $X = B + M$ is a semimartingale as above (the bounded-variation parts contribute nothing).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name">(Covariation by Polarisation)</span></p>

The definition mimics how any quadratic quantity is turned into a bilinear one:

$$
ab = \frac{1}{4}\bigl((a+b)^2 - (a-b)^2\bigr), \qquad \langle a, b \rangle = \frac{1}{4}\bigl(\lVert a + b \rVert^2 - \lVert a - b \rVert^2\bigr).
$$

Quadratic covariation is to quadratic variation what an inner product is to a norm: symmetric, bilinear, and it inherits existence directly from Theorem 4.4. It measures the *co-accumulation* of variance of two martingales — for independent Brownian motions it vanishes, $\langle W^i, W^j \rangle\_t = \delta_{ij} t$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">4.18 (Partial Integration)</span></p>

Suppose that $X, Y$ are $\mathcal{F}\_t$-adapted semimartingales. Then

$$
X_t Y_t = X_0 Y_0 + \int_0^t Y_s\, \mathrm{d}X_s + \int_0^t X_s\, \mathrm{d}Y_s + \int_0^t \mathrm{d}\langle X, Y \rangle_s.
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Corollary 4.18</summary>

Apply the multivariate Itô rule (Theorem 4.16) with $X^{(1)} = X$, $X^{(2)} = Y$, and $f(t, x_1, x_2) = x_1 x_2$: the mixed second derivative equals $1$, all others vanish. $\square$

*(For $X = Y = W$ this is exactly the right panel of the earlier figure: $W_t^2 = 2\int_0^t W\, \mathrm{d}W + t$.)*

</details>
</div>

We now consider an exemplary class of semimartingales.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">4.19 (Itô Process)</span></p>

Let $K(t)$ and $(H_1(t), \dots, H_d(t))$ be progressively measurable processes satisfying

$$
\int_0^t \lvert K(s) \rvert\, \mathrm{d}s < +\infty, \qquad \int_0^t H_i^2(s)\, \mathrm{d}s < +\infty \quad \text{a.s.}
$$

for $t \ge 0$ and $i = 1, \dots, d$. Furthermore, let $W_1, \dots, W_d$ be independent Brownian motions. Then the process

$$
X(t) = X(0) + \int_0^t K(s)\, \mathrm{d}s + \int_0^t H(s)\, \mathrm{d}W(s) = X(0) + \int_0^t K(s)\, \mathrm{d}s + \sum_{j=1}^d \int_0^t H_j(s)\, \mathrm{d}W_j(s)
$$

is called an **Itô process**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">4.20 (Itô Processes Are Semimartingales)</span></p>

Itô processes are semimartingales.

</div>

*Proof: exercise.* (The drift integral is the bounded-variation part; the stochastic integrals are continuous local martingales by construction.)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">4.21 (Covariation and Associativity for Itô Processes)</span></p>

Let $X, \widetilde{X}$ be $\mathcal{F}\_t$-adapted Itô processes with $X(t)$ as in Definition 4.19 and

$$
\widetilde{X}(t) = \widetilde{X}(0) + \int_0^t \widetilde{K}(s)\, \mathrm{d}s + \int_0^t \widetilde{H}(s)\, \mathrm{d}W(s),
$$

where $\widetilde{K}$ and $\widetilde{H}$ fulfil the same conditions as $K$ and $H$ in Definition 4.19. Then

$$
\bigl\langle X, \widetilde{X} \bigr\rangle_t = \sum_{i=1}^d \int_0^t H_i(s)\, \widetilde{H}_i(s)\, \mathrm{d}s,
$$

and

$$
\int_0^t \widetilde{X}(s)\, \mathrm{d}X(s) = \int_0^t \widetilde{X}(s)\, K(s)\, \mathrm{d}s + \int_0^t \widetilde{X}(s)\, H(s)\, \mathrm{d}W(s).
$$

</div>

*Proof: exercise.*

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Convention</span><span class="math-callout__name">(Differential Notation)</span></p>

Instead of $Z(t) = Z_0 + \int_0^t \widetilde{X}(s)\, \mathrm{d}X(s)$, we also write

$$
\mathrm{d}Z(t) = \widetilde{X}(t)\, \mathrm{d}X(t), \qquad Z(0) = Z_0.
$$

The differential notation is pure shorthand — every "$\mathrm{d}$-equation" in what follows *means* the corresponding integral equation.

</div>

We now discuss some stochastic differential equations.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">4.22 (Homogeneous Linear SDE — the Stochastic Exponential)</span></p>

Let $K$ and $H = (H_1, \dots, H_d)$ be progressively measurable processes satisfying

$$
\int_0^t \lvert K(s) \rvert\, \mathrm{d}s < +\infty, \qquad \int_0^t H_i^2(s)\, \mathrm{d}s < +\infty \quad \text{a.s.},
$$

and let $W_1, \dots, W_d$ be independent Brownian motions. Then

$$
Y(t) = \exp\biggl(\int_0^t \Bigl[K(s) - \frac{1}{2} \lVert H(s) \rVert^2\Bigr]\, \mathrm{d}s + \int_0^t H(s)\, \mathrm{d}W(s)\biggr)
$$

is a solution of the homogeneous SDE

$$
\begin{cases} \mathrm{d}Y(t) = Y(t) \bigl[K(t)\, \mathrm{d}t + H(t)\, \mathrm{d}W(t)\bigr], \\ Y(0) = 1. \end{cases} \tag{4.11}
$$

Moreover, if $\widetilde{Y}$ is another solution of (4.11), then $\widetilde{Y}(t) = Y(t)$ a.s., i.e., the solution is unique.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 4.22</summary>

By applying the Itô rule (Theorem 4.16, with $f = \exp$ applied to the Itô process in the exponent), one easily checks that $Y$ solves (4.11): the $\frac{1}{2} f''$-term contributes $\frac{1}{2} \lVert H \rVert^2 Y\, \mathrm{d}t$, which cancels exactly against the $-\frac{1}{2} \lVert H \rVert^2$ built into the exponent.

We show uniqueness. Suppose $\widetilde{Y}$ is another solution of (4.11). Define $\bar{Y}(t) := 1 / Y(t)$ (well-defined, as the exponential never vanishes). By Itô's formula,

$$
\mathrm{d}\bar{Y}(t) = \bar{Y}(t) \Bigl[-K(t)\, \mathrm{d}t + \frac{1}{2} \lVert H(t) \rVert^2\, \mathrm{d}t - H(t)\, \mathrm{d}W(t)\Bigr] + \frac{1}{2} \lVert H(t) \rVert^2\, \bar{Y}(t)\, \mathrm{d}t.
$$

Furthermore, by partial integration (Corollary 4.18) and the covariation formula (Lemma 4.21),

$$
\begin{aligned}
\mathrm{d}\bigl(\widetilde{Y}(t)\, \bar{Y}(t)\bigr) &= \widetilde{Y}(t)\, \mathrm{d}\bar{Y}(t) + \bar{Y}(t)\, \mathrm{d}\widetilde{Y}(t) + \mathrm{d}\bigl\langle \widetilde{Y}, \bar{Y} \bigr\rangle_t \\
&= \widetilde{Y}(t)\, \bar{Y}(t) \Bigl[-K\, \mathrm{d}t + \lVert H \rVert^2\, \mathrm{d}t - H\, \mathrm{d}W + K\, \mathrm{d}t + H\, \mathrm{d}W\Bigr] + \widetilde{Y}(t)\, \bar{Y}(t) \bigl(-H^{\!\top} H\bigr)\, \mathrm{d}t \\
&= 0 \cdot \mathrm{d}t.
\end{aligned}
$$

Thus $\widetilde{Y}(t)\, \bar{Y}(t)$ is constant a.s., and since $\widetilde{Y}(0)\, \bar{Y}(0) = 1$, we have $\widetilde{Y}(t)\, \bar{Y}(t) = 1$ a.s., hence $\widetilde{Y}(t) = Y(t)$ a.s. $\square$

</details>
</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Connection</span><span class="math-callout__name">(Geometric Brownian Motion and the $-\frac{1}{2}\lVert H\rVert^2$ Correction)</span></p>

* For constant scalar coefficients $K \equiv \mu$, $H \equiv \sigma$, the theorem gives **geometric Brownian motion** $Y(t) = \exp\bigl((\mu - \tfrac{\sigma^2}{2}) t + \sigma W_t\bigr)$ — the Black–Scholes asset model, and the left panel of the figure below.
* The $-\frac{1}{2} \sigma^2$ is the Itô correction in action: the *mean* grows like $\mathbb{E}[Y(t)] = e^{\mu t}$, but the *typical path* grows only at rate $\mu - \frac{\sigma^2}{2}$ — multiplicative noise drags the median below the mean (the gap is the variance of the log).
* In the general form, $Y$ is known as the **stochastic (Doléans-Dade) exponential** of the Itô process $\int K\, \mathrm{d}s + \int H\, \mathrm{d}W$; for $K \equiv 0$ it is the prototypical *positive local martingale*, the object underlying changes of measure (Girsanov's theorem) later in the theory.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">4.23 (Inhomogeneous Linear SDE — Variation of Constants)</span></p>

Let $K$, $H$, and $W$ be as in Theorem 4.22. Furthermore, let $k$ and $h = (h_1, \dots, h_m)$ be progressively measurable processes such that

$$
\int_0^t \lvert k(s) \rvert\, \mathrm{d}s < +\infty, \qquad \int_0^t h_i^2(s)\, \mathrm{d}s < +\infty \quad \text{a.s.}
$$

Define $Y(t)$ as in Theorem 4.22 and, for $x \in \mathbb{R}$, define

$$
Z(t) = x + \int_0^t \frac{1}{Y(s)} \Bigl[k(s) - \sum_{j=1}^m H_j(s)\, h_j(s)\Bigr]\, \mathrm{d}s + \sum_{j=1}^m \int_0^t \frac{h_j(s)}{Y(s)}\, \mathrm{d}W_j(s).
$$

Then

$$
X(t) = Y(t)\, Z(t)
$$

is the unique solution to the inhomogeneous SDE

$$
\mathrm{d}X(t) = \bigl[K(t)\, X(t) + k(t)\bigr]\, \mathrm{d}t + \sum_{j=1}^m \bigl[H_j(t)\, X(t) + h_j(t)\bigr]\, \mathrm{d}W_j(t), \qquad X(0) = x. \tag{4.12}
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 4.23</summary>

Apply partial integration (Corollary 4.18) to show that $X = YZ$ solves (4.12). Indeed, using $\mathrm{d}Y = Y[K\, \mathrm{d}t + H\, \mathrm{d}W]$ and the covariation formula (Lemma 4.21),

$$
\begin{aligned}
\mathrm{d}X(t) &= \mathrm{d}\bigl(Y(t)\, Z(t)\bigr) = Y(t)\, \mathrm{d}Z(t) + Z(t)\, \mathrm{d}Y(t) + \mathrm{d}\langle Y, Z \rangle_t \\
&= Y(t) \Biggl[\frac{1}{Y(t)} \Bigl(k(t) - \sum_j H_j(t)\, h_j(t)\Bigr)\, \mathrm{d}t + \sum_j \frac{h_j(t)}{Y(t)}\, \mathrm{d}W_j(t)\Biggr] \\
&\quad + Z(t)\, Y(t) \bigl[K(t)\, \mathrm{d}t + H(t)\, \mathrm{d}W(t)\bigr] + Y(t) \sum_j \frac{H_j(t)\, h_j(t)}{Y(t)}\, \mathrm{d}t \\
&= k(t)\, \mathrm{d}t + \sum_j h_j(t)\, \mathrm{d}W_j(t) + X(t) \bigl[K(t)\, \mathrm{d}t + H(t)\, \mathrm{d}W(t)\bigr],
\end{aligned}
$$

which is (4.12). For uniqueness, suppose $\widetilde{X}$ is another solution. Then $(\widetilde{X} - X)$ solves the homogeneous SDE (4.11) with initial condition $(\widetilde{X} - X)(0) = 0$; hence $Y + \widetilde{X} - X$ solves (4.11) with $(Y + \widetilde{X} - X)(0) = 1$, and uniqueness in Theorem 4.22 implies

$$
\bigl(Y + \widetilde{X} - X\bigr)(t) = Y(t) \quad \text{a.s.}
$$

Therefore $\widetilde{X}(t) = X(t)$ a.s. $\square$

</details>
</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Connection</span><span class="math-callout__name">(Linear SDEs Run Diffusion Models)</span></p>

Theorem 4.23 is the reason the forward half of a diffusion model is *analytically free*:

* **Ornstein–Uhlenbeck.** With $K \equiv -1$, $H \equiv 0$, $k \equiv 0$, $h \equiv \sigma$: $\mathrm{d}X = -X\, \mathrm{d}t + \sigma\, \mathrm{d}W$, and the theorem yields the explicit solution $X(t) = e^{-t} x + \sigma \int_0^t e^{-(t-s)}\, \mathrm{d}W_s$ — a Gaussian process (Wiener integrals of deterministic integrands are Gaussian). This is the process of the time-reversal figure in §3.3 and of Chapter 1's Langevin examples.
* **Forward corruption SDEs.** The VP-type forward process of §1.2, $\mathrm{d}X = -\frac{1}{2}\beta(t) X\, \mathrm{d}t + \sqrt{\beta(t)}\, \mathrm{d}W$, is an inhomogeneous linear SDE with time-dependent (deterministic) coefficients — the identical variation-of-constants computation applies. Consequently the transition kernels $p_{t \mid 0}(x \mid x_0)$ are *explicit Gaussians*, which is precisely what makes denoising score matching (§1.2) tractable: the conditional score $\nabla \log p_{t \mid 0}$ is available in closed form.
* Only the *reverse* dynamics (Theorem 3.8) is nonlinear — through the score $\nabla \log p^\mu$ — and that single nonlinear ingredient is what the neural network learns.

</div>

### 4.3 Simulation of Diffusion Processes

Given a diffusion process $(X_t)\_{t \ge 0}$ characterised by its infinitesimal generator $(A, \mathcal{D}(A))$ as in Definition 3.1, a natural question arises: how can one simulate sample paths of $(X_t)\_{t \ge 0}$ when only the generator is known? Recall that, for any $u \in C_c^\infty(\mathbb{R}^d)$, the generator acts as

$$
A u(x) = \frac{1}{2} \sum_{i,j=1}^d a_{ij}(x)\, \frac{\partial^2 u(x)}{\partial x_i \partial x_j} + \sum_{j=1}^d b_j(x)\, \frac{\partial u(x)}{\partial x_j},
$$

where $a(x) = (a_{ij}(x))\_{i,j=1}^d$ is the diffusion matrix and $b(x) = (b_1(x), \dots, b_d(x))^{\!\top}$ is the drift vector.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Euler–Maruyama Scheme)</span></p>

**Local approximation.** Over a small time step $\Delta t$, the increment $X_{t + \Delta t} - X_t$ can be approximated by a random variable whose conditional mean and covariance match the drift and diffusion coefficients,

$$
\mathbb{E}\bigl[X_{t+\Delta t} - X_t \mid X_t = x\bigr] \approx b(x)\, \Delta t, \qquad \operatorname{Cov}\bigl(X_{t+\Delta t} - X_t \mid X_t = x\bigr) \approx a(x)\, \Delta t.
$$

(These are exactly Kolmogorov's infinitesimal conditions (3.3)–(3.4), read forwards as a recipe rather than backwards as a characterisation.)

**Constructing increments.** Simulate the next position by a normally distributed random variable with the same mean and covariance,

$$
X_{t+\Delta t} \approx x + b(x)\, \Delta t + \sqrt{\Delta t}\; \Sigma(x)\, Z,
$$

where $\Sigma(x)$ is a matrix square root of $a(x)$, i.e., $\Sigma(x)\, \Sigma(x)^{\!\top} = a(x)$, and $Z \sim \mathcal{N}(0, I_d)$ is a standard normal vector in $\mathbb{R}^d$.

**Iterative simulation.** Starting from an initial state $X_0 = x_0$, iterate this scheme over discrete time steps to produce an approximate sample path $\lbrace X_{k \Delta t} \rbrace\_{k = 0, 1, \dots}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Considerations)</span></p>

* **Matrix square root.** Computing $\Sigma(x)$ requires $a(x)$ to be positive semidefinite. Common methods for obtaining such a square root include the Cholesky decomposition or the spectral decomposition.
* **Step size.** The time step $\Delta t$ must be chosen sufficiently small to maintain accuracy, particularly when the coefficients $a(x)$ or $b(x)$ vary rapidly. For SDE coefficients as in Theorem 3.7, the scheme converges *strongly* with order $\frac{1}{2}$ in general ($\mathbb{E}\lvert X_T - X_T^{\Delta t} \rvert = O(\sqrt{\Delta t})$; see the figure below), and with order $1$ for additive noise.
* **Domains and boundaries.** If the diffusion is constrained to a domain with boundary conditions, appropriate modifications (such as reflection or absorption) are required in order to respect these constraints.

</div>

<figure>
  <img src="{{ 'assets/images/notes/sdes_diffusion_models/euler_maruyama_gbm.png' | relative_url }}" alt="Two panels. Left: one exact path of geometric Brownian motion in grey with two Euler-Maruyama approximations overlaid, a coarse one with eight steps in orange visibly deviating and a finer one with sixty-four steps in green tracking the exact path closely. Right: a log-log plot of the strong error against the step size, following a straight dashed guide line of slope one half." loading="lazy">
  <figcaption>Euler–Maruyama for geometric Brownian motion $\mathrm{d}X = X\,\mathrm{d}t + 0.8\,X\,\mathrm{d}W$, whose exact solution is the stochastic exponential of Theorem 4.22. <strong>Left.</strong> Exact path vs. the scheme at $\Delta t = 2^{-3}$ and $2^{-6}$, driven by the same Brownian increments. <strong>Right.</strong> The strong error $\mathbb{E}\lvert X_T - X_T^{\Delta t}\rvert$ (4000 Monte-Carlo paths) decays at the theoretical order $\frac12$ — halving the error costs four times the steps.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Connection</span><span class="math-callout__name">(Sampling in Generative Models Is Euler–Maruyama)</span></p>

Both halves of a score-based diffusion model reduce to this scheme:

* The *forward* corruption rarely needs simulating at all — its transition kernels are explicit Gaussians by Theorem 4.23, so training pairs $(X_0, X_t)$ are sampled in one shot.
* The *generative* direction integrates the reverse SDE (1.4)/Theorem 3.8, whose drift contains the learned score $s_\theta \approx \nabla \log p^\mu$: DDPM-style "ancestral sampling" is, up to reparametrisation, an Euler–Maruyama discretisation of the reverse SDE. The step-size considerations above become the sampling-speed-versus-fidelity trade-off of few-step samplers.

With stochastic integration, the Itô rule, solvable linear SDEs, and a numerical scheme in hand, the microscopic toolkit promised at the start of this chapter is complete.

</div>

## 5 Diffusion Models

We are now in a position to explain the core idea behind diffusion models for generative modelling, using the Ornstein–Uhlenbeck (OU) process as a pedagogical example. This will serve as a bridge between the stochastic processes discussed before and recent machine learning approaches — and, in §5.4, we will see that the theory delivers more than intuition: a quantitative, polynomial-time error bound for the whole sampling pipeline.

### 5.1 The Ornstein–Uhlenbeck Process

The Ornstein–Uhlenbeck (OU) process is a fundamental example of a diffusion process. It is widely used to model mean-reverting behaviour in various applications, such as in finance, physics, and machine learning, and it is one of the rare examples of a diffusion with an explicitly known semigroup. In this section, we derive its transition densities and analyse its asymptotic behaviour.

For parameters $\sigma > 0$, $\theta \in \mathbb{R}$, the Ornstein–Uhlenbeck process is defined by its generator,

$$
L(x, D) f(x) = \frac{1}{2} \sigma^2 \Delta f(x) - \theta x \cdot \nabla f(x),
$$

where $\Delta$ denotes the Laplacian and $\nabla$ the gradient with respect to $x$. In terms of drift and diffusion coefficients,

$$
b(x) = -\theta x = (-\theta x_1, \dots, -\theta x_d)^{\!\top}, \qquad a(x) = \sigma^2 \mathrm{Id}_d,
$$

or, in coordinate notation, $a_{ij}(x) = \sigma^2 \delta_{ij}$ and $b_j(x) = -\theta x_j$. Thus, it is also described as the solution of the SDE

$$
\mathrm{d}X_t = -\theta X_t\, \mathrm{d}t + \sigma\, \mathrm{d}W_t,
$$

where $W = (W_t)\_{t \ge 0}$ is a standard $d$-dimensional Brownian motion.

**Transition densities.** We now apply Proposition 3.6 to compute the transition density. For the Ornstein–Uhlenbeck process, the adjoint operator from the proposition takes the form

$$
L^{\ast}(y, D_y)\, u(y) = \theta\, \nabla_y \cdot \bigl(y\, u(y)\bigr) + \frac{\sigma^2}{2} \Delta_y u(y).
$$

Hence, the transition density $p(t, x, y)$ must satisfy the PDE (forward Kolmogorov equation),

$$
\frac{\partial}{\partial t} p(t, x, y) = \theta\, \nabla_y \cdot \bigl(y\, p(t, x, y)\bigr) + \frac{\sigma^2}{2} \Delta_y p(t, x, y).
$$

We now make the ansatz that the process is a Gaussian process, i.e.,

$$
p(t, x, y) = \frac{1}{\sqrt{(2\pi)^d \det \Sigma_t}} \exp\biggl(-\frac{1}{2} \bigl(y - m_t(x)\bigr)^{\!\top} \Sigma_t^{-1} \bigl(y - m_t(x)\bigr)\biggr),
$$

and determine $m_t(x)$ and $\Sigma_t$ such that this expression solves the forward equation. By direct computation, we find

$$
m_t(x) = e^{-\theta t} x, \qquad \Sigma_t = \frac{\sigma^2}{2\theta} \bigl(\mathrm{Id}_d - e^{-2\theta t}\bigr).
$$

Substituting this expression into the forward equation confirms that the PDE is satisfied. The initial condition is also satisfied in the limit $t \to 0$, as the Gaussian converges to a Dirac delta at $y = x$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Two Routes to the Same Gaussian)</span></p>

The transition density can also be obtained *path-wise*, without any PDE: the OU equation is an inhomogeneous linear SDE, so Theorem 4.23 (with $K \equiv -\theta$, $H \equiv 0$, $k \equiv 0$, $h \equiv \sigma$) yields the explicit solution

$$
X_t = e^{-\theta t} x + \sigma \int_0^t e^{-\theta (t - s)}\, \mathrm{d}W_s.
$$

The Wiener integral of a deterministic integrand is Gaussian; its mean is $e^{-\theta t} x$ and, by the Itô isometry (Theorem 4.6), its variance is $\sigma^2 \int_0^t e^{-2\theta(t-s)}\, \mathrm{d}s = \frac{\sigma^2}{2\theta}\bigl(1 - e^{-2\theta t}\bigr)$ per coordinate — exactly the $m_t$ and $\Sigma_t$ found by the PDE ansatz. The analytical machinery of Chapter 3 and the path-wise machinery of Chapter 4 land on the same answer, as they must.

</div>

**Asymptotic behaviour as $t \to \infty$.** We now assume $\theta > 0$. From the transition distribution

$$
\mathcal{N}\biggl(y;\, e^{-\theta t} x,\ \frac{\sigma^2}{2\theta}\bigl(\mathrm{Id}_d - e^{-2\theta t}\bigr)\biggr),
$$

we observe that, as $t \to \infty$, the mean $m_t(x) = e^{-\theta t} x$ decays to zero exponentially fast, implying that the process tends to the origin, while the covariance matrix $\Sigma_t$ converges to the constant matrix $\frac{\sigma^2}{2\theta} \mathrm{Id}\_d$, indicating that the variance of the process stabilises. This means that for large $t$, the distribution of $X_t$ becomes *independent of the initial condition* $x$, and the process approaches a stationary Gaussian distribution with mean $0$ and covariance $\frac{\sigma^2}{2\theta} \mathrm{Id}\_d$. Hence, the Ornstein–Uhlenbeck process is stationary in the long run, with the stationary distribution being $\mathcal{N}\bigl(0, \frac{\sigma^2}{2\theta} \mathrm{Id}\_d\bigr)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">5.1 (Invariant Measure)</span></p>

A probability measure $\mu$ on $\mathbb{R}^d$ is called an **invariant measure** for the Markov semigroup $(P_t)\_{t \ge 0}$ if it satisfies

$$
\int_{\mathbb{R}^d} P_t g(x)\, \mu(\mathrm{d}x) = \int_{\mathbb{R}^d} g(x)\, \mu(\mathrm{d}x)
$$

for all bounded measurable functions $g : \mathbb{R}^d \to \mathbb{R}$ and all $t \ge 0$. It can be checked that this is equivalent to $L^{\ast} \mu = 0$ (understood in the weak sense).

</div>

**Reverse drift.** According to Theorem 3.8, the time-reversed process $Y_t := X_{T-t}$ is again a diffusion process with diffusion coefficient $a(y) = \sigma^2 \mathrm{Id}\_d$, and time-dependent drift

$$
\widetilde{b}(t, y) = \theta y + \sigma^2\, \nabla_y \log p^\mu(T - t, y).
$$

Thus, the reverse drift depends explicitly on the marginal density of the forward process at time $T - t$. In general, $p^\mu(t, y)$ has no closed-form expression if $\mu$ is arbitrary, as it requires evaluating the convolution

$$
p^\mu(t, y) = \int_{\mathbb{R}^d} \phi_{e^{-\theta t} x,\ \frac{\sigma^2}{2\theta}(\mathrm{Id}_d - e^{-2\theta t})}(y)\, \mu(\mathrm{d}x).
$$

This is a Gaussian smoothing of the initial density $\mu$, which is typically not analytically tractable. One of the rare exceptions is when $\mu$ equals the stationary distribution

$$
\mu(\mathrm{d}y) = \mathcal{N}\biggl(0, \frac{\sigma^2}{2\theta} \mathrm{Id}_d\biggr)(\mathrm{d}y);
$$

the convolution structure implies that $p^\mu(t, y) \equiv \mu(y)$ for all $t \ge 0$, since the Ornstein–Uhlenbeck process is stationary under this law. In this special case,

$$
\nabla_y \log p^\mu(T - t, y) = \nabla_y \log \mu(y) = -\frac{2\theta}{\sigma^2}\, y,
$$

and thus

$$
\widetilde{b}(t, y) = \theta y - 2\theta y = -\theta y.
$$

The time-reversed process is again *the same* Ornstein–Uhlenbeck process — as it must be: a stationary reversible diffusion is statistically indistinguishable run forwards or backwards. This is precisely the Sanity Check of §3.3 (which treated the case $\theta = 1$, $\sigma^2 = 2$), now for general parameters. In the *reversed-clock* convention of the score-SDE literature (integrating backwards in the original time), the drift instead reads $-\theta y - \sigma^2 \nabla_y \log p^\mu$, which at stationarity becomes $+\theta y$, "pointing away from the origin" — the same dynamics, read against the opposite clock.

### 5.2 Core Idea Based on Diffusion Processes

In generative diffusion models, one begins with an initial data distribution $\mu$ on $\mathbb{R}^d$, which is typically highly complex. A canonical example is the distribution of natural images (e.g., images of dogs) represented in pixel space.

The forward process $X$ is initialised with $X_0 \sim \mu$ and is designed to gradually inject noise into the data so that, as time progresses, the distribution of $X_T$ converges to a simple, known reference distribution, usually a standard Gaussian. The Ornstein–Uhlenbeck process exhibits precisely this behaviour: it transforms *any* initial distribution into a Gaussian distribution as $T \to \infty$. More precisely, $X_T$ converges in distribution to the stationary Gaussian

$$
\pi(\mathrm{d}y) = \mathcal{N}\biggl(0, \frac{\sigma^2}{2\theta} \mathrm{Id}_d\biggr)(\mathrm{d}y),
$$

independently of the form of $\mu$. Thus, the OU process can be viewed as a "noising" mechanism that progressively converts structured data into pure noise via stochastic perturbations.

In practice, the distribution $\mu$ is not available in closed form, but we have access to i.i.d. samples from it — namely, the training dataset. To simulate the forward process, one draws $X_0 \sim \mu$ from the dataset and simulates the stochastic process $(X_t)\_{t \in [0, T]}$ forward in time. This yields progressively noisier versions of the data at different time steps, which serve as training inputs for learning the reverse process.

**The inverse problem: sampling from complex distributions.** The objective of generative modelling is to *reverse* this process: starting from samples drawn from the simple Gaussian reference distribution, one aims to generate new samples from the original complex data distribution $\mu$. By the time-reversal result in Theorem 3.8, the time-reversed process $Y_t := X_{T-t}$ is again a diffusion with generator

$$
\widetilde{L}(t, y, D) f(y) = \frac{\sigma^2}{2} \Delta f(y) + \widetilde{b}(t, y) \cdot \nabla f(y),
\qquad
\widetilde{b}(t, y) = \theta y + \sigma^2\, \nabla_y \log p^\mu(T - t, y).
$$

In general, this drift depends on the so-called **score function** $\nabla_y \log p^\mu(t, y)$ of the forward process, which is not explicitly known for complex distributions $\mu$. Assuming that the reverse drift $\widetilde{b}(t, y)$ is known, approximate samples from $\mu$ can be generated by simulating the reverse-time diffusion $Y_t$ forward in time, starting from $Y_0 \sim \pi$, the known reference distribution (typically Gaussian, for large $T$). The stochastic process $Y$ is then simulated as described in §4.3, yielding a terminal sample $Y_T$ that approximates a draw from $\mu$.

**The key challenge: score estimation.** The main practical challenge in implementing the reverse-time dynamics lies in estimating the score function $\nabla_y \log p^\mu(t, y)$ for various time steps $t$. Since the exact marginals $p^\mu(t, y)$ are unknown in general, direct computation of the score is infeasible. In recent diffusion models, this problem is addressed through **score matching**: a time-dependent neural network is trained to approximate the score function using data generated from the forward process. Once the score model has been trained, the reverse-time process can be simulated as described above, yielding approximate samples from the target distribution $\mu$.

### 5.3 The Statistical Problem: Score Matching

In the previous section, we saw that reversing the diffusion process requires knowledge of the score function $\nabla_y \log p^\mu(t, y)$ at various times $t$. Since this function is typically intractable, it must be estimated from data. The standard approach used in diffusion models is *score matching*, a technique originally introduced by Hyvärinen (2005). We now present its mathematical formulation.

**The general problem.** Let $p(y)$ be a probability density on $\mathbb{R}^d$, from which we have access to samples but not to the density itself. The goal is to estimate the score function

$$
s(y) := \nabla_y \log p(y).
$$

Instead of attempting to estimate $p(y)$ directly, score matching proposes to estimate $s(y)$ by solving the optimisation problem

$$
\min_{s_\theta}\ \mathbb{E}\biggl[\frac{1}{2} \bigl\lVert s_\theta(Y) - s(Y) \bigr\rVert^2\biggr],
$$

where $s_\theta : \mathbb{R}^d \to \mathbb{R}^d$ is a parametrised function (e.g., a neural network) approximating the score, $\lVert \cdot \rVert$ denotes the Euclidean norm, and $Y$ has distribution $p$. Since $s(y)$ is unknown, this objective cannot be computed directly. Instead, Hyvärinen's key observation is that the above objective can be rewritten, up to an additive constant independent of $s_\theta$, as

$$
J(s_\theta) := \mathbb{E}\biggl[\frac{1}{2} \bigl\lVert s_\theta(Y) \bigr\rVert^2 + \nabla \cdot s_\theta(Y)\biggr],
$$

where $\nabla \cdot s_\theta = \sum_i \partial_{y_i} s_\theta^i(\cdot)$ denotes the divergence of the vector field $s_\theta$. We sketch the derivation of this identity. Expanding the square,

$$
\mathbb{E}\biggl[\frac{1}{2} \bigl\lVert s_\theta(Y) - s(Y) \bigr\rVert^2\biggr] = \frac{1}{2} \mathbb{E}\bigl[\lVert s_\theta(Y) \rVert^2\bigr] - \mathbb{E}\bigl[s_\theta(Y) \cdot s(Y)\bigr] + \frac{1}{2} \mathbb{E}\bigl[\lVert s(Y) \rVert^2\bigr],
$$

and observe that the last term is independent of $\theta$ and can be neglected for optimisation. The middle term can be rewritten using integration by parts (assuming $p\, s_\theta$ decays at infinity),

$$
\mathbb{E}\bigl[s_\theta(Y) \cdot s(Y)\bigr] = \int_{\mathbb{R}^d} s_\theta(y) \cdot \nabla \log p(y)\ p(y)\, \mathrm{d}y = \int_{\mathbb{R}^d} s_\theta(y) \cdot \nabla p(y)\, \mathrm{d}y = -\int_{\mathbb{R}^d} \nabla \cdot s_\theta(y)\ p(y)\, \mathrm{d}y = -\mathbb{E}\bigl[\nabla \cdot s_\theta(Y)\bigr].
$$

Thus, the objective function becomes $J(\theta) = \mathbb{E}\bigl[\frac{1}{2} \lVert s_\theta(Y) \rVert^2 + \nabla \cdot s_\theta(Y)\bigr]$, which depends only on $s_\theta$, but *not* on the true score $s$. This expectation can be approximated by a Monte Carlo average over samples $y_1, \dots, y_N \sim p$, yielding the empirical loss

$$
\widehat{J}(\theta) = \frac{1}{N} \sum_{k=1}^N \biggl[\frac{1}{2} \bigl\lVert s_\theta(y_k) \bigr\rVert^2 + \nabla \cdot s_\theta(y_k)\biggr],
$$

which can be minimised using standard techniques such as stochastic gradient descent.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Connection</span><span class="math-callout__name">(Three Faces of Score Matching)</span></p>

We have now met all three classical formulations, which coincide up to $\theta$-independent constants:

* **Explicit score matching (ESM)**, §1.2: the ideal $L^2$-regression onto $\nabla \log p$ — intractable, since the true score is unknown.
* **Implicit score matching (ISM)**, this section (Hyvärinen 2005): trades the unknown score for the divergence $\nabla \cdot s_\theta$. Exact, but computing the divergence of a deep network in high dimension is expensive (in practice it is estimated stochastically, e.g. by Hutchinson-type estimators).
* **Denoising score matching (DSM)**, §1.2 (Vincent 2011): replaces the marginal score by the *conditional* score of the forward transition kernel — which is an explicit Gaussian precisely because the forward corruption is a linear SDE (Theorem 4.23). This is the variant used by modern diffusion models.

</div>

**Time-dependent score matching in diffusion models.** In generative diffusion models, the data distribution evolves under the forward diffusion process into a family of progressively noised distributions $\lbrace p^\mu(t, y) \rbrace\_{t \in [0, T]}$. To learn how to reverse this process, one must approximate the time-dependent score function

$$
s_t(y) := \nabla_y \log p^\mu(t, y)
$$

for each time $t$. This is achieved by training a neural network $s_\theta(t, y)$ to match $s_t(y)$, using a generalised score matching objective averaged over time:

$$
\int_0^T \lambda(t)\, \mathbb{E}\biggl[\frac{1}{2} \bigl\lVert s_\theta(t, X_t) \bigr\rVert^2 + \nabla_y \cdot s_\theta(t, X_t)\biggr]\, \mathrm{d}t,
$$

where $X_t \sim p^\mu(t, \cdot)$ and $\lambda(t) \ge 0$ is a weighting function that emphasises certain noise levels over others. In practice, samples $X_t$ are obtained by simulating the forward diffusion process starting from empirical data points $X_0 \sim \mu$; this constructs the training data for score matching at each time $t \in [0, T]$. Once the score model $s_\theta(t, y)$ is trained, it is used to approximate the reverse-time dynamics by replacing the unknown score in the time-reversed SDE with the learned function.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(The Score-Based Generative Pipeline)</span></p>

The theory developed above provides a rigorous foundation for modern score-based generative models. In practice, the modelling pipeline proceeds as follows:

1. **Data and forward process:**
   * One is given i.i.d. data samples $x^{(1)}, \dots, x^{(N)} \sim \mu \subset \mathbb{R}^d$.
   * A forward diffusion process $(X_t)\_{t \in [0, T]}$ is chosen, e.g., an Ornstein–Uhlenbeck process, such that $X_0 \sim \mu$ and $X_T \approx \pi$ is a known reference distribution (e.g., Gaussian).
2. **Score estimation:**
   * The time-dependent score function $\nabla_y \log p^\mu(t, y)$ of the evolving distribution is estimated using score matching.
   * A neural network $s_\theta(t, y)$ is trained using the Hyvärinen objective applied to noised samples $X_t$.
3. **Reverse-time generation:**
   * Given the estimated score function, one simulates (as in §4.3) the diffusion with generator

     $$
     \widetilde{L}(t, y, D) f(y) = \frac{\sigma^2}{2} \Delta f(y) + \bigl[\theta y + \sigma^2 s_\theta(T - t, y)\bigr] \cdot \nabla f(y),
     $$

     starting from $Y_0 \sim \pi$, to generate approximate samples $Y_T \sim \mu$.

In this way, stochastic processes provide the theoretical and algorithmic backbone for generative modelling via diffusion.

</div>

### 5.4 Error Analysis for Diffusion Models

We have seen that the Ornstein–Uhlenbeck process interpolates between the data distribution and a Gaussian, and that its time reversal is governed by the score function of the intermediate marginals. The next question is *algorithmic*: if we discretise the reverse SDE and replace the exact score by an estimated score, how far is the output law from the target distribution? Do we still get a polynomial-time algorithm? As will be shown now, the sampling stage of a diffusion model (under mild assumptions) can be analysed by comparing two *path measures*: the exact reverse diffusion and the discretised reverse process. Since these two processes have the same diffusion coefficient, Girsanov's theorem converts the discrepancy into an integral of squared *drift mismatch*. This path-space KL argument is what yields a polynomial-time discretisation bound.

**Setup.** Let $p$ be the target density on $\mathbb{R}^d$, and let $\mu(\mathrm{d}x) = p(x)\, \mathrm{d}x$. We run the forward OU process with $\theta = 1$, $\sigma = \sqrt{2}$ (so that the invariant measure is the standard Gaussian $\gamma_d := \mathcal{N}(0, \mathrm{Id}\_d)$),

$$
\mathrm{d}X_t = -X_t\, \mathrm{d}t + \sqrt{2}\, \mathrm{d}W_t, \qquad X_0 \sim p.
$$

Let $p_t := p^\mu(t, \cdot)$ denote the density of $X_t$. By the explicit solution of the OU process (Theorem 4.23),

$$
X_t = e^{-t} X_0 + \sqrt{1 - e^{-2t}}\, Z, \qquad Z \sim \mathcal{N}(0, \mathrm{Id}_d) \ \text{independent}.
$$

For a fixed time horizon $T > 0$, the reverse process $Y_t = X_{T-t}$ satisfies (Theorem 3.8, with $\theta = 1$, $\sigma^2 = 2$)

$$
\mathrm{d}Y_t = \bigl[Y_t + 2 \nabla \log p_{T-t}(Y_t)\bigr]\, \mathrm{d}t + \sqrt{2}\, \mathrm{d}\widetilde{W}_t, \qquad Y_0 \sim p_T. \tag{5.1}
$$

If we could simulate (5.1) exactly starting from $p_T$, then $Y_T \sim p$.

**The practical reverse sampler.** In practice, the score $\nabla \log p_t$ is unknown and must be replaced by an estimator $s_t(\cdot)$. We also discretise time: let

$$
0 = t_0 < t_1 < \cdots < t_K = T, \qquad t_k = k h, \quad h = \frac{T}{K}.
$$

On each interval $[t_k, t_{k+1}]$ we "freeze" the estimated score at the left endpoint and consider

$$
\mathrm{d}\widehat{X}_t = \bigl[\widehat{X}_t + 2\, s_{T - t_k}(\widehat{X}_{t_k})\bigr]\, \mathrm{d}t + \sqrt{2}\, \mathrm{d}\widetilde{W}_t, \qquad t \in [t_k, t_{k+1}]. \tag{5.2}
$$

Because the drift is affine in $\widehat{X}\_t$, its one-step update is explicit — variation of constants (Theorem 4.23) on each interval gives the *exponential integrator*

$$
\widehat{X}_{t_{k+1}} = e^h\, \widehat{X}_{t_k} + 2 \bigl(e^h - 1\bigr)\, s_{T - t_k}(\widehat{X}_{t_k}) + \sqrt{e^{2h} - 1}\; \xi_k,
$$

where $\xi_k \sim \mathcal{N}(0, \mathrm{Id}\_d)$ are i.i.d. and independent of the past.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Three Sources of Error)</span></p>

There are three conceptually different errors in the practical sampler:

* **Initialisation error:** the algorithm initialises from the stationary Gaussian $\gamma_d = \mathcal{N}(0, \mathrm{Id}\_d)$ instead of the true (unknown) terminal law $p_T$.
* **Score estimation error:** we use the estimator $s_t$ instead of the true score $\nabla \log p_t$.
* **Discretisation error:** even with the exact score, we freeze it on each interval.

The remainder of this section quantifies all three terms.

</div>

**Statistical distances and information theory.** To quantify the approximation errors, we require appropriate distance measures between probability distributions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Total Variation Distance and KL Divergence)</span></p>

* **Total variation (TV) distance:** measures the maximum discrepancy over all Borel sets and takes values in $[0, 1]$; for two probability measures $P$ and $Q$ on $\mathbb{R}^d$,

  $$
  \mathrm{TV}(P, Q) := \sup_{A \in \mathcal{B}(\mathbb{R}^d)} \bigl\lvert P(A) - Q(A) \bigr\rvert,
  $$

  and if $P$ and $Q$ admit densities $p$ and $q$ with respect to the Lebesgue measure, this simplifies to

  $$
  \mathrm{TV}(P, Q) = \frac{1}{2} \int_{\mathbb{R}^d} \bigl\lvert p(x) - q(x) \bigr\rvert\, \mathrm{d}x.
  $$

* **Kullback–Leibler (KL) divergence:** strictly non-negative and asymmetric. For general probability measures, if $P$ is absolutely continuous with respect to $Q$ ($P \ll Q$), it is defined via the Radon–Nikodym derivative as

  $$
  \mathrm{KL}(P \,\Vert\, Q) := \int_{\mathbb{R}^d} \log\biggl(\frac{\mathrm{d}P}{\mathrm{d}Q}\biggr)\, \mathrm{d}P.
  $$

  If $P \not\ll Q$, we set $\mathrm{KL}(P \,\Vert\, Q) := \infty$. In the special case where both measures admit densities $p$ and $q$,

  $$
  \mathrm{KL}(P \,\Vert\, Q) = \int_{\mathbb{R}^d} p(x) \log\biggl(\frac{p(x)}{q(x)}\biggr)\, \mathrm{d}x.
  $$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">5.2 (Pinsker's Inequality)</span></p>

For any two probability measures $P$ and $Q$,

$$
\mathrm{TV}(P, Q) \le \sqrt{\frac{1}{2} \mathrm{KL}(P \,\Vert\, Q)}.
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 5.2</summary>

If $\mathrm{KL}(P \,\Vert\, Q) = \infty$, the inequality is trivially satisfied, so we may assume $\mathrm{KL}(P \,\Vert\, Q) < \infty$; this finiteness implies $P \ll Q$. By the Radon–Nikodym theorem, the density $f := \frac{\mathrm{d}P}{\mathrm{d}Q}$ exists and is $Q$-almost everywhere non-negative. Using this density,

$$
\mathrm{TV}(P, Q) = \frac{1}{2} \int \lvert f - 1 \rvert\, \mathrm{d}Q \qquad \text{and} \qquad \mathrm{KL}(P \,\Vert\, Q) = \int f \log f\, \mathrm{d}Q.
$$

Consider the strictly convex function $\psi(x) = x \log x - x + 1$ for $x \ge 0$. Since $\int (f - 1)\, \mathrm{d}Q = P(\Omega) - Q(\Omega) = 1 - 1 = 0$,

$$
\mathrm{KL}(P \,\Vert\, Q) = \int (f \log f - f + 1)\, \mathrm{d}Q = \int \psi(f)\, \mathrm{d}Q.
$$

Define the measurable set $A = \lbrace f \ge 1 \rbrace$. Then

$$
\mathrm{TV}(P, Q) = \int_A (f - 1)\, \mathrm{d}Q = \int_{A^c} (1 - f)\, \mathrm{d}Q =: \delta.
$$

Let $\alpha = Q(A)$ and $\beta = Q(A^c) = 1 - \alpha$. We split the KL integral over the partition $\lbrace A, A^c \rbrace$ and apply Jensen's inequality to the convex function $\psi$ on each part, with respect to the normalised probability measures $Q(\cdot \mid A)$ and $Q(\cdot \mid A^c)$:

$$
\mathrm{KL}(P \,\Vert\, Q) = \alpha \int_A \psi(f)\, \frac{\mathrm{d}Q}{\alpha} + \beta \int_{A^c} \psi(f)\, \frac{\mathrm{d}Q}{\beta} \ge \alpha\, \psi\biggl(\int_A f\, \frac{\mathrm{d}Q}{\alpha}\biggr) + \beta\, \psi\biggl(\int_{A^c} f\, \frac{\mathrm{d}Q}{\beta}\biggr).
$$

Observe that $\int_A f\, \mathrm{d}Q = P(A) = Q(A) + \delta = \alpha + \delta$, and similarly $\int_{A^c} f\, \mathrm{d}Q = P(A^c) = \beta - \delta$. Thus,

$$
\mathrm{KL}(P \,\Vert\, Q) \ge \alpha\, \psi\Bigl(1 + \frac{\delta}{\alpha}\Bigr) + \beta\, \psi\Bigl(1 - \frac{\delta}{\beta}\Bigr) = (\alpha + \delta) \log \frac{\alpha + \delta}{\alpha} + (\beta - \delta) \log \frac{\beta - \delta}{\beta},
$$

where the equality expands $\psi(x) = x \log x - x + 1$ (the linear parts telescope to zero). It remains to show that, for $p := \alpha + \delta$ and $q := \alpha$, the function $g(\delta) := p \log \frac{p}{q} + (1 - p) \log \frac{1 - p}{1 - q}$ is bounded below by $2\delta^2$. Computing derivatives with respect to $\delta$,

$$
g(0) = 0, \qquad g'(0) = \log 1 - \log 1 = 0, \qquad g''(\delta) = \frac{1}{\alpha + \delta} + \frac{1}{\beta - \delta} = \frac{1}{(\alpha + \delta)\bigl(1 - (\alpha + \delta)\bigr)}.
$$

Since the maximum of $x(1 - x)$ for $x \in [0, 1]$ is $\frac{1}{4}$, we have $g''(\delta) \ge 4$ uniformly. By Taylor's theorem, there exists $\xi \in (0, \delta)$ such that

$$
g(\delta) = g(0) + g'(0)\, \delta + \frac{1}{2} g''(\xi)\, \delta^2 \ge 0 + 0 + \frac{1}{2} \cdot 4 \cdot \delta^2 = 2 \delta^2.
$$

Thus $\mathrm{KL}(P \,\Vert\, Q) \ge 2\delta^2 = 2\, \mathrm{TV}(P, Q)^2$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">5.3 (Data Processing Inequalities)</span></p>

1. Let $P$ and $Q$ be probability measures, and let $\Phi$ be *any* measurable mapping (such as a deterministic function, the evolution of an SDE up to time $T$, or a projection onto marginals). Then the general data processing inequality (DPI) states that

   $$
   \mathrm{TV}(\Phi_{\#} P, \Phi_{\#} Q) \le \mathrm{TV}(P, Q) \qquad \text{and} \qquad \mathrm{KL}(\Phi_{\#} P \,\Vert\, \Phi_{\#} Q) \le \mathrm{KL}(P \,\Vert\, Q),
   $$

   where $\Phi_{\sharp} P$ denotes the pushforward measure under $\Phi$ (defined by $(\Phi_{\sharp} P)(A) := P(\Phi^{-1}(A))$ — the distribution of $\Phi(X)$ when $X \sim P$).
2. Let $p_t$ be the distribution of the forward OU process at time $t$, and let $\gamma_d = \mathcal{N}(0, \mathrm{Id}\_d)$ be its invariant measure. Then

   $$
   \mathrm{KL}(p_t \,\Vert\, \gamma_d) \le e^{-2t}\, \mathrm{KL}(p_0 \,\Vert\, \gamma_d).
   $$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof sketch of Lemma 5.3</summary>

**Part 1.** Both TV and KL are $f$-divergences of the form $D_f(P \,\Vert\, Q) = \mathbb{E}\_Q[f(Z)]$ with $Z := \frac{\mathrm{d}P}{\mathrm{d}Q}$, where $f$ is strictly convex ($f(x) = \frac{1}{2}\lvert x - 1 \rvert$ and $f(x) = x \log x$, respectively). By the properties of conditional expectation, the Radon–Nikodym derivative of the pushforward measures satisfies

$$
\frac{\mathrm{d}\Phi_{\#} P}{\mathrm{d}\Phi_{\#} Q}(Y) = \mathbb{E}_Q\bigl[Z \mid \Phi(X) = Y\bigr].
$$

Applying Jensen's inequality for conditional expectations yields

$$
D_f(\Phi_{\#} P \,\Vert\, \Phi_{\#} Q) = \mathbb{E}_{\Phi_{\#} Q}\Bigl[f\bigl(\mathbb{E}_Q[Z \mid \Phi(X)]\bigr)\Bigr] \le \mathbb{E}_{\Phi_{\#} Q}\Bigl[\mathbb{E}_Q\bigl[f(Z) \mid \Phi(X)\bigr]\Bigr] = \mathbb{E}_Q\bigl[f(Z)\bigr] = D_f(P \,\Vert\, Q).
$$

**Part 2.** The time derivative of the KL divergence along the OU flow follows the *de Bruijn identity*,

$$
\frac{\mathrm{d}}{\mathrm{d}t} \mathrm{KL}(p_t \,\Vert\, \gamma_d) = -I(p_t \,\Vert\, \gamma_d) := -\int_{\mathbb{R}^d} \Bigl\lVert \nabla \log \frac{p_t(x)}{\gamma_d(x)} \Bigr\rVert_2^2\ p_t(x)\, \mathrm{d}x,
$$

where $I$ is the *relative Fisher information*. The standard Gaussian measure $\gamma_d$ satisfies the logarithmic Sobolev inequality with constant $1$, providing the entropy bound

$$
\mathrm{KL}(p_t \,\Vert\, \gamma_d) \le \frac{1}{2}\, I(p_t \,\Vert\, \gamma_d).
$$

Substituting this into the derivative yields the differential inequality $\frac{\mathrm{d}}{\mathrm{d}t} \mathrm{KL}(p_t \,\Vert\, \gamma_d) \le -2\, \mathrm{KL}(p_t \,\Vert\, \gamma_d)$; integrating over $[0, t]$ (Gronwall) directly implies the exponential contraction $\mathrm{KL}(p_t \,\Vert\, \gamma_d) \le e^{-2t}\, \mathrm{KL}(p_0 \,\Vert\, \gamma_d)$. $\square$

</details>
</div>

**Main theorem: error bounds for diffusion models.** The analysis relies on three structural assumptions, due to Chen, Chewi, Li, Li, Salim, and Zhang ("Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions", ICLR 2023) — henceforth [CCL22].

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Assumption</span><span class="math-callout__name">(Structural Assumptions of [CCL22])</span></p>

1. **Lipschitz score:** for every $t \ge 0$, the exact score $\nabla \log p_t$ is $L$-Lipschitz.
2. **Second moment bound:** the target distribution $p$ satisfies

   $$
   m_2^2 := \mathbb{E}_{X \sim p}\bigl[\lVert X \rVert^2\bigr] < \infty.
   $$

3. **Discrete score accuracy:** for every grid point $t_k = kh$, the estimator $s_{t_k}$ achieves an expected $L^2$-error bounded by $\varepsilon_{\mathrm{score}}$,

   $$
   \mathbb{E}_{X \sim p_{t_k}}\Bigl[\bigl\lVert s_{t_k}(X) - \nabla \log p_{t_k}(X) \bigr\rVert_2^2\Bigr] \le \varepsilon_{\mathrm{score}}^2.
   $$

To ensure convergence in total variation, we additionally assume $\mathrm{KL}(p \,\Vert\, \gamma_d) < \infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">5.4 ([CCL22], Theorem 1)</span></p>

Suppose the three assumptions hold, with $L \ge 1$ and $h \le 1/L$. Let $\widehat{p}\_T$ be the law of the discretised reverse process (5.2) initialised at the standard Gaussian $\gamma_d$. Then

$$
\mathrm{TV}(\widehat{p}_T, p) \le \underbrace{\sqrt{\mathrm{KL}(p \,\Vert\, \gamma_d)}\; e^{-T}}_{\text{initialisation error}} + \underbrace{\bigl(L\sqrt{d h} + L m_2 h\bigr) \sqrt{T}}_{\text{discretisation error}} + \underbrace{\varepsilon_{\mathrm{score}} \sqrt{T}}_{\text{score estimation error}}.
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Polynomial-Time Sampling)</span></p>

* Assume $\mathrm{KL}(p \,\Vert\, \gamma_d)$ is polynomial in $d$ and $m_2^2 \le d$ — mild conditions on the data distribution: no log-concavity, no isoperimetry, no smoothness of $p$ itself beyond the Lipschitz score.
* Choose the terminal time $T \approx \log(\mathrm{KL}/\varepsilon)$ and the step size $h \approx \varepsilon^2 / (L^2 d)$.
* Then the total error is bounded by $\widetilde{\mathcal{O}}(\varepsilon + \varepsilon_{\mathrm{score}})$, requiring $K = T/h = \widetilde{\mathcal{O}}(L^2 d / \varepsilon^2)$ steps.
* Conclusion: the reverse-time generative *sampling* stage has polynomial complexity — the entire difficulty of generative modelling is compressed into learning the score. Hence the title of [CCL22]: *sampling is as easy as learning the score*.

</div>

<figure>
  <img src="{{ 'assets/images/notes/sdes_diffusion_models/diffusion_error_analysis.png' | relative_url }}" alt="Two panels of log-scale error curves. Left: measured total variation distance against the horizon T, dropping quickly from about 0.09 and flattening into a floor slightly above 0.01, with a dashed theoretical guide line decaying exponentially above it. Right: measured total variation against the step size h on a log-log scale, decreasing from about 0.5 at coarse steps down to about 0.008, roughly following a dashed guide line of slope one half before flattening at small h." loading="lazy">
  <figcaption>Theorem 5.4, measured. The exponential-integrator sampler (5.2) is run in $d=1$ with the <em>exact</em> score ($\varepsilon_{\mathrm{score}}=0$) for the bimodal target $p = \frac12\mathcal{N}(-2, 0.5^2) + \frac12\mathcal{N}(2, 0.5^2)$ under the forward OU process $\mathrm{d}X = -X\,\mathrm{d}t + \sqrt{2}\,\mathrm{d}W$; TV is estimated from $3\cdot 10^5$ samples. <strong>Left.</strong> At fixed $h = 0.01$, the error drops with the horizon $T$ at least as fast as the initialisation bound $\sqrt{\mathrm{KL}(p\,\|\,\gamma)}\,e^{-T}$ (dashed) predicts, then hits the discretisation/estimation floor — running the chain longer buys nothing once the other error terms dominate. <strong>Right.</strong> At fixed $T = 4$, the error decays with the step size $h$ consistently with the $\sqrt{h}$ discretisation term of Theorem 5.4 (dashed guide), until the Monte-Carlo estimation floor takes over.</figcaption>
</figure>

**A roadmap of the proof.** Let $\mathbb{Q}\_T$ be the path measure of the exact reverse SDE (5.1), let $\mathbb{P}\_T^{\circ}$ be the path measure of the approximate reverse process (5.2) initialised at $p_T$, and let $\mathbb{P}\_T$ be the same approximate process initialised at $\gamma_d$. The final law $\widehat{p}\_T$ is the time-$T$ marginal of $\mathbb{P}\_T$, while $p$ is the time-$T$ marginal of $\mathbb{Q}\_T$. Therefore,

$$
\mathrm{TV}(\widehat{p}_T, p) \le \mathrm{TV}(\mathbb{P}_T, \mathbb{P}_T^{\circ}) + \mathrm{TV}(\mathbb{P}_T^{\circ}, \mathbb{Q}_T). \tag{5.3}
$$

Indeed, we first apply the triangle inequality on path space, and then use the data processing inequality (Lemma 5.3, part 1) for the map $\omega \mapsto \omega_T$ which sends a path to its terminal point. The first term is purely an initialisation mismatch; the second term is the core of the argument and is controlled by path-space KL divergence.

**Analysis of the initialisation error.** The measures $\mathbb{P}\_T$ and $\mathbb{P}\_T^{\circ}$ share the identical transition rule and differ only in their initial distributions ($\gamma_d$ versus $p_T$). Applying Lemma 5.3, part 1 (the transition rule is a pushforward acting identically on both initial laws) together with Pinsker's inequality (Lemma 5.2),

$$
\mathrm{TV}(\mathbb{P}_T, \mathbb{P}_T^{\circ}) \le \mathrm{TV}(\gamma_d, p_T) \le \sqrt{\frac{1}{2} \mathrm{KL}(p_T \,\Vert\, \gamma_d)},
$$

and finally, applying Lemma 5.3, part 2,

$$
\mathrm{KL}(p_T \,\Vert\, \gamma_d) \le e^{-2T}\, \mathrm{KL}(p_0 \,\Vert\, \gamma_d) = e^{-2T}\, \mathrm{KL}(p \,\Vert\, \gamma_d).
$$

Combining these three steps establishes the initialisation bound from Theorem 5.4:

$$
\mathrm{TV}(\mathbb{P}_T, \mathbb{P}_T^{\circ}) \le \sqrt{\frac{1}{2} e^{-2T}\, \mathrm{KL}(p \,\Vert\, \gamma_d)} \le e^{-T} \sqrt{\mathrm{KL}(p \,\Vert\, \gamma_d)}.
$$

**Path-space KL divergence via Girsanov's theorem.** We now compare $\mathbb{Q}\_T$ and $\mathbb{P}\_T^{\circ}$; this is the heart of the proof. Under $\mathbb{Q}\_T$, the process $X_t$ solves

$$
\mathrm{d}X_t = \bigl[X_t + 2 \nabla \log p_{T-t}(X_t)\bigr]\, \mathrm{d}t + \sqrt{2}\, \mathrm{d}W_t,
$$

while under $\mathbb{P}\_T^{\circ}$, on the interval $[t_k, t_{k+1}]$, it solves

$$
\mathrm{d}X_t = \bigl[X_t + 2\, s_{T - t_k}(X_{t_k})\bigr]\, \mathrm{d}t + \sqrt{2}\, \mathrm{d}W_t.
$$

Thus, the drift mismatch on this interval is (up to the factor $2$ in front of both scores)

$$
\Delta_t := s_{T - t_k}(X_{t_k}) - \nabla \log p_{T-t}(X_t), \qquad t \in [t_k, t_{k+1}]. \tag{5.4}
$$

For analysing the drift mismatch, we require the following classical result from stochastic analysis.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">5.5 (Girsanov's Theorem)</span></p>

Let $B = (B_t)\_{t \in [0, T]}$ be a standard $d$-dimensional Brownian motion on a filtered probability space $(\Omega, \mathcal{F}, (\mathcal{F}\_t)\_{t \in [0, T]}, \mathbb{P})$. Let $\Delta = (\Delta_t)\_{t \in [0, T]}$ be an $\mathbb{R}^d$-valued, adapted process. Define the exponential process $Z = (Z_t)\_{t \in [0, T]}$ via

$$
Z_T := \exp\biggl(\int_0^T \langle \Delta_t, \mathrm{d}B_t \rangle - \frac{1}{2} \int_0^T \lVert \Delta_t \rVert_2^2\, \mathrm{d}t\biggr).
$$

If $\mathbb{E}\_{\mathbb{P}}[Z_T] = 1$, then $Z_T$ defines a valid Radon–Nikodym derivative for a new probability measure, $\frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mathbb{P}} = Z_T$. Under the equivalent measure $\mathbb{Q}$, the process

$$
\widetilde{B}_t := B_t - \int_0^t \Delta_s\, \mathrm{d}s, \qquad t \in [0, T],
$$

is a standard $d$-dimensional Brownian motion.

</div>

Two remarks on the statement: by construction, the process $Z$ is a non-negative *local* martingale (it is a stochastic exponential in the sense of Theorem 4.22, with $K \equiv 0$); Girsanov's theorem requires it to be a *true* martingale (i.e., $\mathbb{E}\_{\mathbb{P}}[Z_T] = 1$) to ensure that $\mathbb{Q}$ is a well-defined probability measure. A standard sufficient analytical criterion guaranteeing this integrability is the following condition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">5.6 (Novikov Condition)</span></p>

An adapted process $\Delta = (\Delta_t)\_{t \in [0, T]}$ satisfies the **Novikov condition** if

$$
\mathbb{E}_{\mathbb{P}}\biggl[\exp\biggl(\frac{1}{2} \int_0^T \lVert \Delta_t \rVert_2^2\, \mathrm{d}t\biggr)\biggr] < \infty.
$$

</div>

Now, by Girsanov's theorem, under mild assumptions, the Radon–Nikodym derivative of $\mathbb{Q}\_T$ with respect to $\mathbb{P}\_T^{\circ}$ is given by the exponential martingale built from the (rescaled) drift mismatch $\sqrt{2}\, \Delta_t$ — the two SDEs share the diffusion coefficient $\sqrt{2}$, so the Girsanov shift is $(\text{drift difference})/\sqrt{2} = \sqrt{2}\, \Delta_t$:

$$
\frac{\mathrm{d}\mathbb{Q}_T}{\mathrm{d}\mathbb{P}_T^{\circ}} = \exp\biggl(\int_0^T \bigl\langle \sqrt{2}\, \Delta_t, \mathrm{d}B_t \bigr\rangle - \frac{1}{2} \int_0^T \bigl\lVert \sqrt{2}\, \Delta_t \bigr\rVert_2^2\, \mathrm{d}t\biggr).
$$

Then,

$$
\mathrm{KL}(\mathbb{Q}_T \,\Vert\, \mathbb{P}_T^{\circ}) = \mathbb{E}_{\mathbb{Q}_T}\biggl[\int_0^T \lVert \Delta_t \rVert_2^2\, \mathrm{d}t\biggr] = \sum_{k=0}^{K-1} \int_{t_k}^{t_{k+1}} \mathbb{E}_{\mathbb{Q}_T}\bigl[\lVert \Delta_t \rVert_2^2\bigr]\, \mathrm{d}t. \tag{5.5}
$$

(The diffusion coefficient $\sqrt{2}$ turns the usual $\frac{1}{2}\int \lVert \cdot \rVert^2$ into $\int \lVert \cdot \rVert^2$; from here on, absolute constants are not tracked precisely, following [CCL22].) So the entire discretisation analysis reduces to controlling the integral of the squared drift mismatch.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">5.7 (Why a Stopping Argument Is Needed)</span></p>

Under the minimal assumptions of [CCL22], one cannot directly justify the exponential martingale in Girsanov's theorem by a simple Novikov condition. The workaround is to *stop* the process: define

$$
\tau_R := \inf\biggl\lbrace t \le T \,:\, \int_0^t \lVert \Delta_s \rVert_2^2\, \mathrm{d}s \ge R \biggr\rbrace \wedge T.
$$

For the stopped process, the quadratic variation of the stochastic exponential's martingale part is bounded by ($2$ times) $R$, so the corresponding exponential martingale is integrable and Girsanov applies. One obtains the KL identity up to time $\tau_R$ and then passes to the limit $R \to \infty$ using the lower semicontinuity of the KL divergence.

</div>

Pinsker's inequality on path space now gives

$$
\mathrm{TV}(\mathbb{P}_T^{\circ}, \mathbb{Q}_T)^2 \le \frac{1}{2}\, \mathrm{KL}(\mathbb{Q}_T \,\Vert\, \mathbb{P}_T^{\circ}).
$$

Since total variation decreases under marginalisation (Lemma 5.3, part 1), the same bound controls the TV distance of the time-$T$ marginals.

**Bounding the drift mismatch.** Under the exact reverse path law $\mathbb{Q}\_T$,

$$
X_{t_k} \sim p_{T - t_k}, \qquad X_t \sim p_{T - t}, \qquad t \in [t_k, t_{k+1}].
$$

For $t \in [t_k, t_{k+1}]$, we decompose

$$
\Delta_t = \underbrace{s_{T-t_k}(X_{t_k}) - \nabla \log p_{T-t_k}(X_{t_k})}_{\text{score estimation}} + \underbrace{\nabla \log p_{T-t_k}(X_{t_k}) - \nabla \log p_{T-t}(X_{t_k})}_{\text{time variation}} + \underbrace{\nabla \log p_{T-t}(X_{t_k}) - \nabla \log p_{T-t}(X_t)}_{\text{spatial variation}}.
$$

Using $\lVert a + b + c \rVert_2^2 \le 3 \bigl(\lVert a \rVert_2^2 + \lVert b \rVert_2^2 + \lVert c \rVert_2^2\bigr)$, we bound these three components independently. Proof strategy: the first term uses Assumption 3, the third exploits Lipschitz continuity (Assumption 1), and the second relies on the fact that $p_{T-t_k}$ and $p_{T-t}$ differ solely by a *short OU step*.

**I) Score estimation term.** Since $T - t_k \in \lbrace h, 2h, \dots, T \rbrace$, Assumption 3 directly applies to the reverse-time index $T - t_k$, giving

$$
\mathbb{E}_{\mathbb{Q}_T}\Bigl[\bigl\lVert s_{T-t_k}(X_{t_k}) - \nabla \log p_{T-t_k}(X_{t_k}) \bigr\rVert_2^2\Bigr] \le \varepsilon_{\mathrm{score}}^2;
$$

integrating over an interval of length $h$ and summing over all $K = T/h$ intervals yields

$$
\sum_{k=0}^{K-1} \int_{t_k}^{t_{k+1}} \mathbb{E}_{\mathbb{Q}_T}\Bigl[\bigl\lVert s_{T-t_k}(X_{t_k}) - \nabla \log p_{T-t_k}(X_{t_k}) \bigr\rVert_2^2\Bigr]\, \mathrm{d}t \le T\, \varepsilon_{\mathrm{score}}^2.
$$

**III) Spatial variation term.** By Assumption 1, $\nabla \log p_{T-t}$ is globally $L$-Lipschitz, giving

$$
\bigl\lVert \nabla \log p_{T-t}(X_{t_k}) - \nabla \log p_{T-t}(X_t) \bigr\rVert_2^2 \le L^2\, \lVert X_t - X_{t_k} \rVert_2^2,
$$

so we need a bound on how far the reverse process moves over one interval. Under $\mathbb{Q}\_T$ the process is the time reversal of the forward OU process, so it is enough to estimate the movement of the *forward* process $\overline{X}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">5.8 (Movement Bound)</span></p>

Let $0 \le s \le t \le T$ with $\delta := t - s \le 1$. Then, up to absolute constants,

$$
\mathbb{E}\bigl\lVert \overline{X}_t - \overline{X}_s \bigr\rVert^2 \lesssim d\, \delta + m_2^2\, \delta^2. \tag{5.6}
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 5.8</summary>

From the explicit OU formula (Theorem 4.23),

$$
\overline{X}_t = e^{-\delta}\, \overline{X}_s + \sqrt{1 - e^{-2\delta}}\, Z, \qquad Z \sim \mathcal{N}(0, \mathrm{Id}_d),
$$

where $Z$ is independent of $\overline{X}\_s$. Therefore,

$$
\overline{X}_t - \overline{X}_s = \bigl(e^{-\delta} - 1\bigr)\, \overline{X}_s + \sqrt{1 - e^{-2\delta}}\, Z.
$$

Taking expectations and using $1 - e^{-\delta} \le \delta$ and $1 - e^{-2\delta} \le 2\delta$, we obtain

$$
\mathbb{E}\bigl\lVert \overline{X}_t - \overline{X}_s \bigr\rVert^2 = \bigl(1 - e^{-\delta}\bigr)^2\, \mathbb{E}\bigl\lVert \overline{X}_s \bigr\rVert^2 + \bigl(1 - e^{-2\delta}\bigr) d \le \delta^2\, \mathbb{E}\bigl\lVert \overline{X}_s \bigr\rVert^2 + 2 \delta d.
$$

Finally, from the forward process bounds (interpolation between $p$ and $\gamma_d$),

$$
\mathbb{E}\bigl\lVert \overline{X}_s \bigr\rVert^2 = e^{-2s}\, \mathbb{E}\bigl\lVert \overline{X}_0 \bigr\rVert^2 + \bigl(1 - e^{-2s}\bigr) d \le m_2^2 + d,
$$

so $\mathbb{E}\lVert \overline{X}\_t - \overline{X}\_s \rVert^2 \le 2\delta d + (m_2^2 + d)\, \delta^2$; absorbing the term $d\, \delta^2 \le d\, \delta$ (valid for $\delta \le 1$) proves (5.6) up to constants. $\square$

</details>
</div>

Using the lemma with $s = T - t$ and $t = T - t_k$ under the forward process (so $\delta \le h$), we get under $\mathbb{Q}\_T$

$$
\mathbb{E}_{\mathbb{Q}_T}\Bigl[\bigl\lVert \nabla \log p_{T-t}(X_{t_k}) - \nabla \log p_{T-t}(X_t) \bigr\rVert_2^2\Bigr] \le L^2 \bigl(d h + m_2^2 h^2\bigr).
$$

After integrating over $t \in [t_k, t_{k+1}]$ and summing over $k$,

$$
\sum_{k=0}^{K-1} \int_{t_k}^{t_{k+1}} \mathbb{E}_{\mathbb{Q}_T}\Bigl[\bigl\lVert \nabla \log p_{T-t}(X_{t_k}) - \nabla \log p_{T-t}(X_t) \bigr\rVert_2^2\Bigr]\, \mathrm{d}t \le L^2 T \bigl(d h + m_2^2 h^2\bigr).
$$

**II) Time variation term.** The distributions $p_{T-t_k}$ and $p_{T-t}$ differ by a short OU step: if $\delta = t - t_k$, then

$$
p_{T-t_k} = S_{\delta\#}\, p_{T-t} \ast \mathcal{N}\bigl(0, (1 - e^{-2\delta})\, \mathrm{Id}_d\bigr), \qquad \text{where } S_\delta(x) = e^{-\delta} x.
$$

Here $S_{\delta\sharp}\, p_{T-t}$ denotes the pushforward of $p_{T-t}$ under the map $x \mapsto e^{-\delta} x$. So the difference between the two scores comes from a *small contraction* $x \mapsto e^{-\delta} x$ and a *small Gaussian smoothing*. We first record the score perturbation lemma proved by Chen et al. [CCL22, Lemma 16], stated in the $d$-dimensional form relevant here.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">5.9 (Score Perturbation; [CCL22], Lemma 16)</span></p>

Let $0 < \zeta < 1$. Suppose $M_0, M_1 \in \mathbb{R}^{d \times d}$, where $M_1$ is symmetric positive definite and $\lVert M_0 - \mathrm{Id}\_d \rVert\_{\mathrm{op}} \le \zeta$. Let $q = e^{-H}$ be a probability density on $\mathbb{R}^d$ such that $\nabla H$ is $L$-Lipschitz, and assume

$$
L \le \frac{1}{4 \lVert M_1 \rVert_{\mathrm{op}}}.
$$

Then, for every $\theta \in \mathbb{R}^d$,

$$
\bigl\lVert \nabla \log\bigl((M_0)_{\#} q \ast \mathcal{N}(0, M_1)\bigr)(\theta) - \nabla \log q(\theta) \bigr\rVert_2 \le L \sqrt{\lVert M_1 \rVert_{\mathrm{op}}\, d} + L \zeta \lvert \theta \rvert + \bigl(\zeta + L \lVert M_1 \rVert_{\mathrm{op}}\bigr) \bigl\lVert \nabla H(\theta) \bigr\rVert_2.
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof sketch of Lemma 5.9</summary>

First, consider the pure smoothing case $M_0 = \mathrm{Id}\_d$. For fixed $\theta$, define the tilted probability measure

$$
p_\theta(\mathrm{d}\theta') \propto \exp\biggl(-\frac{1}{2} \bigl\langle \theta - \theta', M_1^{-1} (\theta - \theta') \bigr\rangle\biggr)\, q(\mathrm{d}\theta'),
$$

where $M_1^{-1}$ is understood on $\operatorname{range}(M_1)$. Differentiating under the integral,

$$
\nabla \log\bigl(q \ast \mathcal{N}(0, M_1)\bigr)(\theta) - \nabla \log q(\theta) = \mathbb{E}_{p_\theta}\bigl[\nabla H(\theta')\bigr] - \nabla H(\theta).
$$

Taking norms and using the Lipschitzness of $\nabla H$,

$$
\bigl\lVert \nabla \log\bigl(q \ast \mathcal{N}(0, M_1)\bigr)(\theta) - \nabla \log q(\theta) \bigr\rVert_2 \le L\, \mathbb{E}_{p_\theta} \lvert \theta' - \theta \rvert.
$$

Now, the negative log-density of $p_\theta$ is

$$
H_\theta(\theta') := H(\theta') + \frac{1}{2} \bigl\langle \theta - \theta', M_1^{-1} (\theta - \theta') \bigr\rangle.
$$

Since $\nabla H$ is $L$-Lipschitz, the assumption $L \le \frac{1}{4\lVert M_1 \rVert_{\mathrm{op}}}$ makes $H_\theta$ strongly convex,

$$
\nabla^2 H_\theta \succeq \bigl(\lVert M_1 \rVert_{\mathrm{op}}^{-1} - L\bigr) \mathrm{Id}_d \succeq \frac{1}{2 \lVert M_1 \rVert_{\mathrm{op}}}\, \mathrm{Id}_d.
$$

So $p_\theta$ is strongly log-concave. Let $\theta_{\ast}$ be its mode; standard concentration results for strongly log-concave measures yield

$$
\mathbb{E}_{p_\theta} \lvert \theta' - \theta_{\ast} \rvert \le \sqrt{\lVert M_1 \rVert_{\mathrm{op}}\, d}.
$$

On the other hand, the optimality condition at the mode gives $\nabla H(\theta_{\ast}) + M_1^{-1}(\theta_{\ast} - \theta) = 0$, and hence

$$
\lvert \theta_{\ast} - \theta \rvert \le \lVert M_1 \rVert_{\mathrm{op}}\, \bigl\lVert \nabla H(\theta_{\ast}) \bigr\rVert_2 \le \dots
$$

which, combined with the triangle inequality $\mathbb{E}\lvert \theta' - \theta \rvert \le \mathbb{E}\lvert \theta' - \theta_{\ast} \rvert + \lvert \theta_{\ast} - \theta \rvert$ and one more Lipschitz step, leads to the stated bound. For general $M_0$, one decomposes the difference using the triangle inequality and a Taylor expansion for the pushforward operation, which produces the remaining terms involving $\zeta$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Where the Proof Stands)</span></p>

Let us take stock of the three components of the drift mismatch (5.4):

* **I (score estimation)** contributes $T \varepsilon_{\mathrm{score}}^2$ to the path-space KL (5.5) — after Pinsker, the $\varepsilon_{\mathrm{score}} \sqrt{T}$ term of Theorem 5.4.
* **III (spatial variation)** contributes $L^2 T (d h + m_2^2 h^2)$ — after Pinsker, the $(L\sqrt{dh} + L m_2 h)\sqrt{T}$ discretisation term.
* **II (time variation)** reduces to Lemma 5.9 applied with $M_0 = e^{-\delta} \mathrm{Id}\_d$ and $M_1 = (1 - e^{-2\delta})\, \mathrm{Id}\_d$, so that $\zeta \approx \delta \le h$ and $\lVert M_1 \rVert\_{\mathrm{op}} \le 2h$ — note that the lemma's hypothesis $L \le \frac{1}{4\lVert M_1 \rVert\_{\mathrm{op}}}$ is exactly where the step-size restriction $h \lesssim 1/L$ of Theorem 5.4 enters. Carrying out this application (bounding the resulting $\lvert \theta \rvert$- and $\lVert \nabla H \rVert$-moments under $p_{T-t}$) shows that the time-variation term is of the same order as term III; assembling I–III through (5.5), Pinsker on path space, and the decomposition (5.3) completes the proof of Theorem 5.4.

</div>
