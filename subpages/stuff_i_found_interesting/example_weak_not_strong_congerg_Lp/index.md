---
layout: note
title: "Weak but Not Strong Convergence in Lp: sin(nt)"
date: 2024-10-20
math: true
tags:
  - mathematics
  - functional-analysis
  - weak-convergence
  - lp-spaces
  - riemann-lebesgue-lemma
---

<div style="margin: 1.2rem 0 1.5rem;">

<p style="margin: 0 0 0.4rem 0; font-style: italic; color: #5F5E5A;">Interactive: slide to increase $n$ and watch $t \mapsto \sin(nt)$ on $[0,1]$. The graph oscillates faster, while its amplitude stays between $-1$ and $1$.</p>

<div style="margin: 1rem 0;">
  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">
    <label for="sin-n-slider" style="font-size: 14px; color: var(--color-text-secondary, #5F5E5A); min-width: 70px;">Frequency n</label>
    <input type="range" min="0" max="11" value="3" step="1" id="sin-n-slider" style="flex: 1; min-width: 200px;" />
    <span id="sin-n-value" style="font-size: 14px; font-weight: 500; min-width: 56px; text-align: right;">10</span>
  </div>

  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 1rem;">
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">Oscillation scale 2&pi;/n</div>
      <div id="sin-stat-period" style="font-size: 20px; font-weight: 500;">0.628</div>
    </div>
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">&int;<sub>0</sub><sup>1</sup> sin(nt) dt</div>
      <div id="sin-stat-integral" style="font-size: 20px; font-weight: 500;">0.184</div>
    </div>
    <div style="background: var(--color-background-secondary, #f5f6ff); border-radius: var(--border-radius-md, 8px); padding: 10px 12px;">
      <div style="font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">L<sup>2</sup> norm</div>
      <div id="sin-stat-l2" style="font-size: 20px; font-weight: 500;">0.690</div>
    </div>
  </div>

  <div style="display: flex; flex-wrap: wrap; gap: 18px; margin-bottom: 8px; font-size: 12px; color: var(--color-text-secondary, #5F5E5A);">
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 2px; background: #534AB7;"></span>
      sin(nt)
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 0; border-top: 1.5px dashed rgba(95,94,90,0.55);"></span>
      zero level
    </span>
    <span style="display: inline-flex; align-items: center; gap: 6px;">
      <span style="display: inline-block; width: 18px; height: 0; border-top: 1.5px dashed rgba(83,74,183,0.28);"></span>
      amplitude envelope
    </span>
  </div>

  <div style="position: relative; width: 100%; height: 340px;">
    <canvas id="sin-chart" role="img" aria-label="Plot of sine of n times t on the interval from 0 to 1, controlled by a slider for n.">Plot of sin(nt) on [0,1].</canvas>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
(function () {
  const nValues = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
  const sampleCount = 5000;
  const pathColor = '#534AB7';
  const guideColor = 'rgba(95, 94, 90, 0.55)';
  const envelopeColor = 'rgba(83, 74, 183, 0.28)';

  const slider = document.getElementById('sin-n-slider');
  const nValueEl = document.getElementById('sin-n-value');
  const periodEl = document.getElementById('sin-stat-period');
  const integralEl = document.getElementById('sin-stat-integral');
  const l2El = document.getElementById('sin-stat-l2');

  const zeroLine = [{ x: 0, y: 0 }, { x: 1, y: 0 }];
  const upperEnvelope = [{ x: 0, y: 1 }, { x: 1, y: 1 }];
  const lowerEnvelope = [{ x: 0, y: -1 }, { x: 1, y: -1 }];

  function formatNumber(value) {
    if (Math.abs(value) < 0.001) return value.toExponential(2);
    return value.toFixed(3);
  }

  function buildSinPath(n) {
    const pts = new Array(sampleCount + 1);
    for (let i = 0; i <= sampleCount; i++) {
      const t = i / sampleCount;
      pts[i] = { x: t, y: Math.sin(n * t) };
    }
    return pts;
  }

  let chart = null;

  function update() {
    const idx = parseInt(slider.value, 10);
    const n = nValues[idx];
    const integral = (1 - Math.cos(n)) / n;
    const l2Squared = 0.5 - Math.sin(2 * n) / (4 * n);

    nValueEl.textContent = n.toLocaleString();
    periodEl.textContent = formatNumber((2 * Math.PI) / n);
    integralEl.textContent = formatNumber(integral);
    l2El.textContent = formatNumber(Math.sqrt(l2Squared));

    const pathData = buildSinPath(n);
    const lineWidth = n <= 50 ? 1.8 : (n <= 500 ? 1.2 : 0.8);

    if (chart) {
      chart.data.datasets[0].data = pathData;
      chart.data.datasets[0].borderWidth = lineWidth;
      chart.update('none');
      return;
    }

    chart = new Chart(document.getElementById('sin-chart'), {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'sin(nt)',
            data: pathData,
            borderColor: pathColor,
            backgroundColor: 'transparent',
            borderWidth: lineWidth,
            pointRadius: 0,
            order: 1,
          },
          { label: 'y = 0', data: zeroLine, borderColor: guideColor, borderWidth: 1.2, borderDash: [4, 4], pointRadius: 0, fill: false, order: 2 },
          { label: 'y = 1', data: upperEnvelope, borderColor: envelopeColor, borderWidth: 1, borderDash: [5, 4], pointRadius: 0, fill: false, order: 3 },
          { label: 'y = -1', data: lowerEnvelope, borderColor: envelopeColor, borderWidth: 1, borderDash: [5, 4], pointRadius: 0, fill: false, order: 4 },
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
            min: -1.15,
            max: 1.15,
            title: { display: true, text: 'sin(nt)', color: '#5F5E5A' },
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

> Does $\sin(nt)$ converges weakly to $0$ in $L^p((0,1))$?

Yes, for the usual range

$$1 \le p < \infty,$$

we have

$$\sin(nt) \rightharpoonup 0 \quad\text{weakly in }L^p((0,1))$$

Let $q$ be the conjugate exponent:

$$\frac1p+\frac1q=1.$$

To prove weak convergence, we need to show that for every test function $g\in L^q((0,1))$,

$$\int_0^1 \sin(nt) g(t)dt \to 0.$$

Since $(0,1)$ has finite measure, every $g\in L^q((0,1))$ is also in $L^1((0,1))$. Therefore the **Riemann–Lebesgue lemma** applies:

$$\int_0^1 g(t)\sin(nt)dt \to 0.$$

So indeed,

$$\sin(nt) \rightharpoonup 0 \quad\text{in } L^p((0,1)), \qquad 1\le p<\infty.$$

But it does **not** converge strongly to $0$, because

$$|\sin(nt)|_{L^p}^p = \int_0^1 |\sin(nt)|^pdt$$

does not go to $0$. The functions only oscillate faster; their size does not disappear.

For $p=\infty$: it converges to $0$ in the weak-* topology $\sigma(L^\infty,L^1)$, but the full weak topology of $L^\infty$ is more subtle.

> Does $sin(nt)$ converge strongly to $0$ in $L^p((0,1))$?

No.

Strong convergence to $0$ in $L^p((0,1))$ would mean

$$|\sin(nt)|_{L^p((0,1))}\to 0.$$

For $1\le p<\infty$,

$$|\sin(nt)|_{L^p}^p = \int_0^1 |\sin(nt)|^pdt$$

Make the change of variables $s=nt$. Then

$$\int_0^1 |\sin(nt)|^pdt = \frac1n\int_0^n |\sin s|^pds$$

But $\|\sin s\|^p$ is periodic and has positive average over one period. Therefore

$$\frac1n\int_0^n |\sin s|^pds \to \frac{1}{2\pi}\int_0^{2\pi}|\sin s|^pds > 0.$$

So

$$|\sin(nt)|_{L^p} \not\to 0$$

Hence

$$\sin(nt)\not\to 0 \quad\text{strongly in }L^p((0,1))$$

For $p=\infty$, also no, because

$$|\sin(nt)|_{L^\infty((0,1))}\to 1,$$

so it definitely does not go to $0$.

> could you tell me what is Riemann–Lebesgue lemma?

The **Riemann–Lebesgue lemma** says:

If $f\in L^1((a,b))$, then

$$\int_a^b f(t)\sin(nt),dt \to 0$$

and

$$\int_a^b f(t)\cos(nt),dt \to 0$$

as (n\to\infty).

Equivalently, using complex exponentials,

$$\int_a^b f(t)e^{int},dt \to 0.$$

So the high-frequency oscillations of $\sin(nt)$, $\cos(nt)$, or $e^{int}$ “average out” against any integrable function $f$.

For example, if $f(t)=1$, then

$$\int_0^1 \sin(nt)dt = \left[-\frac{\cos(nt)}{n}\right]_0^1 = \frac{1-\cos n}{n}\to 0$$

The lemma says that this remains true even for any $f\in L^1$, not just smooth or constant $f$.

In your earlier question, we used it with

$$f(t)=g(t),$$

where $g\in L^q((0,1))$. Since $(0,1)$ has finite measure, $g\in L^q$ implies $g\in L^1$ for $q\ge 1$. Therefore

$$\int_0^1 g(t)\sin(nt),dt \to 0.$$

That is exactly why

$$\sin(nt)\rightharpoonup 0$$

weakly in $L^p((0,1))$ for $1\le p<\infty$.

> it suggests the functions $\cos(nt)$ and $\sin(nt)$ cross ox axes totally in points of total measure zero?

Yes, for each fixed $n$, the zeros of $\sin(nt)$ and $\cos(nt)$ in $(0,1)$ form a finite set, hence a set of Lebesgue measure zero.

For example,

$$\sin(nt)=0$$

when

$$nt=k\pi,$$

so

$$t=\frac{k\pi}{n}.$$

There are only finitely many such points inside $(0,1)$. Therefore

$$\lbrace t\in(0,1):\sin(nt)=0\rbrace$$

has measure zero.

Similarly,

$$\cos(nt)=0$$

when

$$nt=\frac{\pi}{2}+k\pi,$$

so

$$t=\frac{\frac{\pi}{2}+k\pi}{n}.$$

Again, finitely many points, hence measure zero.

But this is **not** the main reason behind the Riemann–Lebesgue lemma.

The important point is not that the functions cross the $x$-axis at measure-zero sets. The important point is that they oscillate faster and faster, so positive and negative parts cancel out in the integral:

$$\int_0^1 f(t)\sin(nt),dt \to 0.$$

The zero-crossing points themselves do not contribute anything to the integral, because points have measure zero. What matters are the intervals between the zeros, where $\sin(nt)$ is positive or negative.

So:

$$\text{zeros have measure zero} \quad \neq \quad \text{weak convergence to }0.$$

Weak convergence comes from **oscillatory cancellation**, not from the fact that the function is zero at some points. This is also why

$$\sin(nt)\not\to 0$$

strongly in $L^p$: its magnitude does not disappear; it only oscillates faster.

> I am new to functional analysis, could you explain to me, why you take a test function from $L^q((0,1))$ to test weak convergence in $L^p((0,1))$? Why not in the same space?

Because **weak convergence is tested by continuous linear functionals**, not by arbitrary elements of the same space.

For a normed space $X$,

$$x_n \rightharpoonup x$$

means:

$$\ell(x_n)\to \ell(x) \quad\text{for every continuous linear functional } \ell\in X^*.$$

So if

$$X=L^p((0,1)),$$

then we need to know what the dual space $(L^p)^*$ is.

For

$$1\le p<\infty,$$

the dual is

$$(L^p)^* \cong L^q,$$

where

$$\frac1p+\frac1q=1.$$

This means: every continuous linear functional on $L^p$ can be written as

$$\ell_g(f)=\int_0^1 f(t)g(t),dt$$

for some $g\in L^q((0,1))$.

So to test whether

$$f_n \rightharpoonup 0 \quad\text{in }L^p,$$

we check whether

$$\ell_g(f_n)\to 0$$

for all $g\in L^q$, i.e.

$$\int_0^1 f_n(t)g(t)dt\to 0 \quad\text{for all }g\in L^q.$$

In your example,

$$f_n(t)=\sin(nt),$$

so weak convergence to $0$ means

$$\int_0^1 \sin(nt)g(t)dt\to 0 \quad\text{for all }g\in L^q.$$

That is why the test function is from $L^q$, not generally from $L^p$.

The reason $L^q$ appears is **Hölder’s inequality**:

$$
\left|\int_0^1 f(t)g(t)dt\right|
\le
|f|_{L^p}|g|_{L^q}.
$$

This guarantees that

$$f\mapsto \int_0^1 f(t)g(t)dt$$

is a continuous linear functional on $L^p$.

In a Hilbert space, for example $L^2$, we do test using the same space because

$$(L^2)^*\cong L^2.$$

Indeed, for $p=2$, we have $q=2$. So weak convergence in $L^2$ is tested by

$$\int_0^1 f_n(t)g(t)dt$$

with $g\in L^2$.

But for (p\neq 2), the dual space is different:

$$(L^p)^* \cong L^q,$$

so we test using $L^q$, not $L^p$.
