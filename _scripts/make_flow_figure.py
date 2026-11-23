#!/usr/bin/env python3
"""Generate _includes/flow-figure.svg: the landing-page figure.

It draws the displacement (McCann) interpolation between a bimodal density
rho_0 and a unimodal density rho_1 on the real line. In 1-D the optimal
transport geodesic is linear in quantile functions:
    Q_t(u) = (1 - t) Q_0(u) + t Q_1(u),   rho_t(Q_t(u)) = 1 / Q_t'(u).
Pure standard library; rerun after changing parameters:
    python3 _scripts/make_flow_figure.py
"""
import math
import os

W, H, PAD_TOP, BASE = 640, 150, 8, 142
X_MIN, X_MAX = -3.3, 3.6
STEPS = 9  # number of curves, t = 0, 1/8, ..., 1

def mix_pdf(x):
    g = lambda x, m, s: math.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    return 0.55 * g(x, -1.7, 0.45) + 0.45 * g(x, 1.5, 0.6)

def target_pdf(x):
    m, s = 0.25, 0.85
    return math.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))

def quantile_table(pdf, n=6000):
    xs = [X_MIN - 2 + (X_MAX - X_MIN + 4) * i / n for i in range(n + 1)]
    cdf, acc = [0.0], 0.0
    for a, b in zip(xs, xs[1:]):
        acc += 0.5 * (pdf(a) + pdf(b)) * (b - a)
        cdf.append(acc)
    cdf = [c / acc for c in cdf]
    def q(u):
        lo, hi = 0, n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if cdf[mid] < u: lo = mid
            else: hi = mid
        c0, c1 = cdf[lo], cdf[hi]
        return xs[lo] + (xs[hi] - xs[lo]) * ((u - c0) / (c1 - c0) if c1 > c0 else 0.0)
    return q

q0, q1 = quantile_table(mix_pdf), quantile_table(target_pdf)
us = [1e-4 + (1 - 2e-4) * i / 2400 for i in range(2401)]
Q0, Q1 = [q0(u) for u in us], [q1(u) for u in us]

def curve(t):
    qt = [(1 - t) * a + t * b for a, b in zip(Q0, Q1)]
    pts = []
    for i in range(1, len(us) - 1):
        dq = (qt[i + 1] - qt[i - 1]) / (us[i + 1] - us[i - 1])
        if dq > 0:
            pts.append((qt[i], 1.0 / dq))
    return pts

curves = [curve(k / (STEPS - 1)) for k in range(STEPS)]
y_max = max(y for c in curves for _, y in c)

def sx(x): return (x - X_MIN) / (X_MAX - X_MIN) * W
def sy(y): return BASE - y / y_max * (BASE - PAD_TOP)

def path(pts, every=20):
    pts = [p for p in pts if X_MIN <= p[0] <= X_MAX][::every]
    d = "M" + " L".join(f"{sx(x):.1f} {sy(y):.1f}" for x, y in pts)
    return d

out = [f'<svg class="flow-figure__svg" viewBox="0 0 {W} {H}" role="img" '
       'aria-labelledby="flow-title" xmlns="http://www.w3.org/2000/svg">',
       '<title id="flow-title">Nine probability densities morphing from a bimodal to a unimodal shape</title>',
       f'<line class="flow-figure__base" x1="0" y1="{BASE + 0.5}" x2="{W}" y2="{BASE + 0.5}"/>']
for k, c in enumerate(curves):
    t = k / (STEPS - 1)
    end = " flow-figure__curve--end" if k in (0, STEPS - 1) else ""
    out.append(f'<path class="flow-figure__curve{end}" style="--t:{t:.3f};--i:{k}" '
               f'pathLength="1" d="{path(c)}"/>')
out.append("</svg>")
dest = os.path.join(os.path.dirname(__file__), "..", "_includes", "flow-figure.svg")
with open(dest, "w") as f:
    f.write("\n".join(out) + "\n")
print("wrote", os.path.normpath(dest))
