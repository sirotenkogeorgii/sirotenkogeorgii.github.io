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

If $\mu \equiv 0$ on $I$, the Gaussian process $(X_t)\_{t \in I}$ is called **centred**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Matrix Notation for Continuous Time)</span></p>

At first glance it may look suspicious to speak of a "covariance matrix" when $I$ is a continuous index set — a matrix is a finite object, while $I$ has uncountably many instants. The resolution is that the definition does not assert a single matrix for the whole process.

**What the definition actually says.** For **any** finite choice of times $t_1, \dots, t_n \in I$, the random vector $(X_{t_1}, \dots, X_{t_n})$ is multivariate Gaussian with covariance matrix $\bigl(\gamma(t_i, t_j)\bigr)\_{1 \le i, j \le n}$. The quantifier runs over *finite* subsets of $I$; each such subset yields its own finite-dimensional marginal.

**The global object is the kernel.** On the whole of continuous $I$, the fundamental object is the **covariance function** (or *kernel*)

$$
\gamma : I \times I \to \mathbb{R},
$$

which encodes the covariance of *any* two time instants. Matrices appear only as *finite-dimensional marginals* of this kernel: restricting $\gamma$ to the finite grid $\lbrace t_1, \dots, t_n \rbrace \times \lbrace t_1, \dots, t_n \rbrace$ produces the $n \times n$ matrix above.

**Link to Kolmogorov's extension theorem.** A Gaussian process is thus fully characterised by its *finite-dimensional distributions*, and each such distribution is genuinely finite-dimensional (hence a matrix). This is precisely the setup of Kolmogorov's extension theorem (Lemma 0.2), which assembles a continuous-time process from a consistent family of finite-dimensional marginals.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The existence of Gaussian Process guaranteed by Kolmogorov's extension theorem)</span></p>

The existence of such processes on a suitable probability space is guaranteed by **Kolmogorov's extension theorem**, whose discussion we omit at this point.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
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

<h2 class="sr-only">Interactive visualization of the discrete process W_t^n as a step function on [0,1], with a slider controlling n and reference bands at plus/minus sqrt(t) and plus/minus two sqrt(t).</h2>

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

* **(W1) Starting in 0:** $W_0 = 0$, $\mathbb{P}$-a.s.
* **(W2) Independent increments:** For any $n \in \mathbb{N}$ and $0 =: t_0 < t_1 < \dots < t_n$ in $I$, the increments $W_{t_1} - W_{t_0},\, W_{t_2} - W_{t_1},\, \dots,\, W_{t_n} - W_{t_{n-1}}$ are mutually stochastically independent.
* **(W3) Normal increments:** For any $s < t$ in $I$, $W_t - W_s \sim \mathcal{N}(0, t - s)$.
* **(W4) Continuous paths:** The paths $t \mapsto W_t$ are $\mathbb{P}$-a.s. continuous on $I$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
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

</div>

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

Combining Proposition 0.5 with Lemma 0.2, we obtain a stochastic process that fulfils (W1), (W2), (W3). However, this process is *not* automatically guaranteed to satisfy (W4) — we must *modify* it.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">0.6 (Versions / Modifications / Indistinguishability)</span></p>

Let $X := (X_t)\_{t \in I}$ and $Y := (Y_t)\_{t \in I}$ be two stochastic processes on $(\Omega, \mathcal{A}, \mathbb{P})$.

(a) $X$ and $Y$ are **versions** (or **modifications**) of each other iff

$$\mathbb{P}(X_t = Y_t) = 1 \quad \text{for all } t \in I.$$

(b) $X$ and $Y$ are **indistinguishable** iff

$$\mathbb{P}(X_t = Y_t \text{ for all } t \in I) = 1.$$

</div>

Note that indistinguishability is strictly stronger: it requires $\mathbb{P}$-almost-sure equality *simultaneously* on the (uncountable) whole index set, whereas a version only gives almost-sure equality *pointwise* in $t$.

Even when a process has finite-dimensional distributions fulfilling (W1)–(W3), the path property (W4) cannot be inferred on an uncountable index $I$, because countable unions of $\mathbb{P}$-null sets need no longer be $\mathbb{P}$-null. Nevertheless, we *can* define a modification preserving the finite-dimensional distributions (cf. Kolmogorov's extension theorem).

**Principle strategy for proving existence** of a real-valued (standard) Brownian motion:

1. Define a stochastic process $\widetilde{W}$ with properties (W1), (W2), (W3).
2. Define a modification $W$ of $\widetilde{W}$ that additionally fulfils (W4).

&rarr; $W$ satisfies (W1)–(W4), i.e. $W$ is a real-valued (standard) Brownian motion.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">0.7 (Completion)</span></p>

For any $A \subseteq \Omega$ such that there exists $A' \in \mathcal{A}$ with $A' \supseteq A$ and $\mathbb{P}(A') = 0$, we set

$$
\mathbb{P}(A) := 0 \quad \text{and} \quad \mathbb{P}(A^c) := 1.
$$

</div>

The main tool behind the modification argument (accepted without proof) is Kolmogorov's continuity theorem.

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

(i) $\displaystyle \mathbb{P}(\tau_x \le t) = 2 \cdot \mathbb{P}(W_t \ge x) = \frac{x}{\sqrt{2\pi}} \int_0^t \frac{1}{s^{3/2}} \cdot \exp\!\left(-\frac{x^2}{2s}\right) \mathrm{d}s$ for all $t \in I$;

(ii) $\displaystyle \mathbb{P}(M_t \ge x) \le \sqrt{\frac{2}{\pi}} \cdot \frac{\sqrt{t}}{x} \cdot \exp\!\left(-\frac{x^2}{2 t}\right)$ for all $t \in I$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Corollary 0.21 (i))</span></p>

The reflection principle doubles the probability of paths crossing a level $x \in \mathbb{R}$ before time $t \in I$, compared to simply ending above $x$, as any path crossing $x$ can be paired with a mirrored path not crossing it.

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

## 1 Introduction

Before formalising the theory of continuous-time stochastic processes, we briefly outline the two central concepts upon which this course is built: **stochastic differential equations (SDEs)** and **diffusion models**. The aim is to clarify both the mathematical necessity of SDEs and their application in recent generative modelling.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stochastic Differential Equations)</span></p>

To model dynamical systems subject to continuous random fluctuations, one naturally considers differential equations driven by a noise process, where the canonical mathematical object for continuous noise is **Brownian motion**. However, the sample paths of Brownian motion are almost surely nowhere differentiable and possess infinite first variation (see Proposition 0.11 (ii) and §1.1), rendering classical Riemann–Stieltjes integration inapplicable.

SDEs, formalised via *Itô calculus*, provide the rigorous measure-theoretic framework required to integrate against such processes. By doing so, an SDE defines a flow not merely on a state space, but on the underlying probability space — allowing for rigorous analysis of stochastic dynamics in continuous time.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Diffusion Models)</span></p>

In the context of machine learning, *generative modelling* poses the problem:

> *Given an empirical distribution of samples, how can we construct a mechanism to draw novel samples from the underlying, unknown probability measure $p_{\mathrm{data}}$ on $\mathbb{R}^d$?*

Diffusion models solve this by constructing a continuous-time *stochastic transport map* via a two-step procedure:

1. A **forward SDE** acts as a smoothing operator, continuously transporting the intractable measure $p_{\mathrm{data}}$ into a mathematically tractable reference measure — typically a standard Gaussian $\mathcal{N}(0, I)$.
2. Exploiting the measure-theoretic properties of Markov process time-reversal (Kolmogorov 1936, Nelson 1967), one derives a **reverse SDE**. If the *score function* (the logarithmic gradient of the marginal density) can be estimated, this reverse SDE transports the reference measure back to $p_{\mathrm{data}}$.

Thus diffusion models provide a constructive, mathematically rigorous algorithm for generative synthesis. The aim of this course is therefore twofold: (i) build the stochastic analysis required to understand SDEs, and (ii) apply these tools to analyse the mechanics and convergence of diffusion-based generative models.

</div>

To build a rigorous theory of continuous stochastic dynamics, we must construct a framework capable of handling random motion that is nowhere differentiable. We base the exposition on the classical works of *Øksendal* (2013), *Revuz & Yor* (1999), and *Karatzas & Shreve* (1991).

**Foundations.** The cornerstone of continuous-time stochastic processes is the Brownian motion from Chapter 0. It is the canonical model for purely random, unpredictable continuous motion. As observed earlier (Paley–Wiener–Zygmund 1933), the paths of Brownian motion are $\mathbb{P}$-almost surely nowhere differentiable, with *infinite first variation* — hence classical Riemann–Stieltjes integration fails entirely for integrals of the form $\int f(t) \mathrm{d} W_t$.

**Diffusion processes.** A general *diffusion process* may be characterised as a continuous-time, continuous-state Markov process with almost surely continuous sample paths. The Markov property ensures that future evolution depends only on the present state, not on history. Since paths are continuous and memoryless, the process is fully determined by its local, *infinitesimal* behaviour, captured by the **infinitesimal generator** $\mathscr{L}$:

$$
(\mathscr{L} f)(x) := \lim_{t \searrow 0} \frac{\mathbb{E}[\,f(X_t)\,\mid X_0 = x\,] - f(x)}{t}.
$$

For sufficiently regular diffusions in $\mathbb{R}^d$, $\mathscr{L}$ takes the form of a second-order elliptic differential operator,

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

* **The diffusion matrix $\Sigma(t, x) := \sigma(t, x) \sigma(t, x)^\top$.** The function $\sigma(t, x)$ determines the *volatility*. Due to the Brownian scaling $(\mathrm{d} W_t)^2 = \mathrm{d} t$, the local covariance is dominated entirely by the stochastic term. Applying *Itô's isometry* yields

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

**Score matching and generative synthesis.** The challenge of generative modelling reduces to estimating the time-dependent vector field $\nabla_x \log p(t, x)$. Assuming access to an i.i.d. dataset $X_1, \dots, X_n \sim p_0$ (the unknown, highly complex data distribution), introduce a parametric family — typically a neural network — $s_\theta(x, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$. We aim to find parameters $\theta$ such that $s_\theta(x, t) \approx \nabla_x \log p(t, x)$ for all $t \in [0, T]$. The natural loss for this regression task is the expected $L^2$-distance, integrated over time — **explicit score matching (ESM)**:

$$
\mathcal{J}_{\mathrm{ESM}}(\theta) = \frac{1}{2} \int_0^T \lambda(t)\, \mathbb{E}_{X_t \sim p_t}\!\left[\,\bigl\lVert s_\theta(X_t, t) - \nabla_{X_t} \log p(t, X_t) \bigr\rVert_2^2 \,\right] \mathrm{d} t,
$$

with a positive weighting $\lambda(t) > 0$. Minimising $\mathcal{J}\_{\mathrm{ESM}}$ directly is impossible in practice: the marginal density at time $t$ is $p(t, x) = \int_{\mathbb{R}^d} p_{t\mid 0}(x \mid x_0)\, p_0(x_0)\, \mathrm{d} x_0$; since $p_0$ is unknown and the integral is high-dimensional, the true marginal score $\nabla_x \log p(t, x)$ is fundamentally intractable.

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

with mean scaling $m(t) = \exp\!\bigl(-\tfrac{1}{2} \int_0^t \beta(s)\, \mathrm{d} s\bigr)$ and variance $v(t) = 1 - \exp\!\bigl(-\int_0^t \beta(s)\, \mathrm{d} s\bigr)$. Consequently, the conditional score is simply the score of a Gaussian, with trivial closed form:

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
