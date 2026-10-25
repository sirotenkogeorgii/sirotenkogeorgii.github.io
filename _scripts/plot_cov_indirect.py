"""
Figures for Chapters 1-2 of subpages/books/calculus_of_variations_hd/index.md
(Calculus of Variations, Heidelberg WS 2026/27, T. Laux after H. Knuepfer).

Writes PNGs into assets/images/notes/books/calculus_of_variations_hd/.

Figures
-------
 1. cov_brachistochrone.png            -- Ex 1.1: the cycloid beats the straight line
 2. cov_growth_envelopes.png           -- 1.2: p-growth envelopes of three integrands
 3. cov_fundamental_lemma.png          -- Lemma 1.4: bumps probe v, dipoles probe jumps
 4. cov_affine_admissible_class.png    -- 2.1: A = g + X_0 and its tangent space
 5. cov_differentiability_ladder.png   -- Def 2.1: first variation / Gateaux / Frechet
 6. cov_extrema.png                    -- Fig. 3 of the script: extrema of cos(3 pi x)/x
 7. cov_second_variation_coercivity.png-- Thm 2.4(ii): why the constant c matters
 8. cov_boundary_conditions.png        -- Thms 2.5/2.6: Dirichlet vs natural BC
 9. cov_double_well.png                -- Ex 2.8: non-convex integrand, sawtooth minimizers
10. cov_oscillating_test_function.png  -- Thm 2.9: the test function eps psi a sin(x.b/eps)
11. cov_rank_one_cone.png              -- Def 2.10: rank-1 directions, det along them
12. cov_cofactor_geometry.png          -- Lemma 2.15: det scales volume, cof moves normals
13. cov_det_null_lagrangian.png        -- Thm 2.16: int det Du is fixed by u on the boundary
14. cov_weak_continuity_det.png        -- Thm 2.18: det survives weak limits, |Du|^2 does not
15. cov_brouwer_retraction.png         -- Lemma 2.19 / Thm 2.20: the retraction construction
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, FancyArrowPatch  # noqa: E402

OUT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "assets",
        "images",
        "notes",
        "books",
        "calculus_of_variations_hd",
    )
)
os.makedirs(OUT, exist_ok=True)

C_BLUE = "#2c3e94"
C_GREEN = "#1f9d55"
C_ORANGE = "#e67e22"
C_RED = "#c0392b"
C_PURPLE = "#7d3c98"
C_TEAL = "#117a8b"
C_GREY = "#5a6171"
C_LIGHT = "#cfd5e2"
C_INK = "#222222"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig, name, tight=True):
    out = os.path.join(OUT, name)
    # tight=False for figures with 3-D axes: the tight cropper mis-measures them
    fig.savefig(out, dpi=160, bbox_inches="tight" if tight else None)
    plt.close(fig)
    print("wrote", out)


def _t(p):
    return (float(p[0]), float(p[1]))


def _panel_title(ax, text):
    ax.set_title(text, loc="left", fontweight="semibold", color=C_INK)


# --------------------------------------------------------------------------
# 1. Brachistochrone
# --------------------------------------------------------------------------
def fig_brachistochrone():
    g = 1.0
    a = 1.0
    xB, yB = np.pi * a, 2.0 * a  # the cycloid's lowest point for theta_B = pi

    def travel_time(x, y):
        # T = int ds / sqrt(2 g y), y measured downward, y(0) = 0.  On each segment
        # y is taken linear in arclength, which integrates exactly to
        #   ds * 2 (sqrt(y1) - sqrt(y0)) / (sqrt(2g) (y1 - y0)),
        # so the 1/sqrt(y) singularity at the start is handled exactly.
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        y0, y1 = y[:-1], y[1:]
        flat = np.abs(y1 - y0) < 1e-12
        seg = np.empty_like(ds)
        seg[~flat] = ds[~flat] * 2 * (np.sqrt(y1[~flat]) - np.sqrt(y0[~flat])) / (np.sqrt(2 * g) * (y1[~flat] - y0[~flat]))
        seg[flat] = ds[flat] / np.sqrt(2 * g * np.maximum(y0[flat], 1e-12))
        return float(np.sum(seg))

    th = np.linspace(0, np.pi, 20000)
    xc, yc = a * (th - np.sin(th)), a * (1 - np.cos(th))

    xs = np.linspace(0, xB, 20000)
    ys = yB * xs / xB

    # a circular arc through A and B, bulging below the chord
    cx, cy = xB / 2 + 0.7, yB / 2 - 0.7 * xB / yB  # centre on the perpendicular bisector
    r = np.hypot(cx, cy)
    phi0, phi1 = np.arctan2(0 - cy, 0 - cx), np.arctan2(yB - cy, xB - cx)
    phis = np.linspace(phi0, phi1, 20000)
    xa, ya = cx + r * np.cos(phis), cy + r * np.sin(phis)

    # a "go down first, then across" path
    xp = np.linspace(0, xB, 20000)
    yp = yB * (1 - (1 - xp / xB) ** 6)

    curves = [
        (xc, yc, C_GREEN, "cycloid (brachistochrone)", 3.0),
        (xs, ys, C_BLUE, "straight line", 2.0),
        (xa, ya, C_ORANGE, "circular arc", 2.0),
        (xp, yp, C_PURPLE, "steep drop, then flat", 2.0),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for x, y, c, lab, lw in curves:
        T = travel_time(x, y)
        ax.plot(x, -y, color=c, lw=lw, label=f"{lab}:  $T = {T:.3f}$")
    ax.plot([0, xB], [0, -yB], "o", color=C_INK, ms=6, zorder=5)
    ax.annotate("$A$", (0, 0), xytext=(-14, 6), textcoords="offset points", fontsize=12)
    ax.annotate("$B$", (xB, -yB), xytext=(6, -12), textcoords="offset points", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, xB + 0.4)
    ax.set_ylim(-yB - 0.5, 0.35)
    ax.set_xlabel("$x$")
    ax.set_ylabel("height")
    ax.legend(loc="upper right", frameon=False)
    _panel_title(ax, r"Travel time $T[y]=\int_0^{x_B}\sqrt{\frac{1+y'(x)^2}{2gy(x)}}\,\mathrm{d}x$ of a bead sliding from $A$ to $B$ (with $g=1$)")
    _save(fig, "cov_brachistochrone.png")


# --------------------------------------------------------------------------
# 2. Growth envelopes
# --------------------------------------------------------------------------
def fig_growth_envelopes():
    xi = np.linspace(-2.6, 2.6, 800)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    items = [
        (np.sqrt(1 + xi**2), 1, C_BLUE, r"minimal surface $\sqrt{1+\xi^2}$", r"$p=1$"),
        (0.5 * xi**2, 2, C_GREEN, r"Dirichlet $\frac{1}{2}\xi^2$", r"$p=2$"),
        ((xi**2 - 1) ** 2, 4, C_RED, r"double well $(\xi^2-1)^2$", r"$p=4$"),
    ]
    for f, p, c, lab, plab in items:
        ax.plot(xi, f, color=c, lw=2.2, label=lab)
        env = 1.0 * (1 + np.abs(xi) ** p)
        ax.plot(xi, env, color=c, lw=1.2, ls="--", alpha=0.75, label=r"envelope $1+|\xi|^{%d}$  (%s)" % (p, plab))
    ax.set_ylim(0, 28)
    ax.set_xlim(-2.6, 2.6)
    ax.set_xlabel(r"$\xi$ (gradient variable)")
    ax.set_ylabel(r"$f(\xi)$")
    ax.legend(loc="upper center", frameon=False, ncol=2)
    _panel_title(ax, r"Growth condition $|f(x,z,\xi)|\leq C(1+|z|^p+|\xi|^p)$: each integrand sits below a $p$-envelope (dashed)")
    _save(fig, "cov_growth_envelopes.png")


# --------------------------------------------------------------------------
# 3. Fundamental lemma / Du Bois-Reymond
# --------------------------------------------------------------------------
def _bump(x, c, w):
    t = (x - c) / w
    out = np.zeros_like(x)
    m = np.abs(t) < 1
    out[m] = np.exp(-1.0 / (1 - t[m] ** 2)) * np.e
    return out


def fig_fundamental_lemma():
    x = np.linspace(0, 1, 1200)
    v = 0.9 * np.sin(3 * np.pi * x) * np.exp(-((x - 0.5) ** 2) / 0.4)
    phi = _bump(x, 0.5, 0.11)
    phi /= phi.max()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    ax.axhline(0, color=C_GREY, lw=0.8)
    ax.plot(x, v, color=C_BLUE, lw=2.2, label="$v$")
    ax.plot(x, phi, color=C_ORANGE, lw=2.0, label=r"test bump $\varphi\in\mathcal{D}(\Omega)$")
    ax.fill_between(x, 0, v * phi, color=C_GREEN, alpha=0.35, label=r"$v\varphi$,  $\int v\varphi\,\mathrm{d}x<0$")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlabel("$x$")
    _panel_title(ax, r"(a) Fundamental lemma: a bump placed where $v\neq 0$ detects it")

    ax = axes[1]
    vj = np.where(x < 0.55, 0.3, 0.9)
    w = 0.1
    psi = _bump(x, 0.55, w)
    dpsi = np.gradient(psi, x)
    dpsi /= np.abs(dpsi).max()
    ax.axhline(0, color=C_GREY, lw=0.8)
    ax.plot(x, vj, color=C_BLUE, lw=2.2, label="$v$ (with a jump)")
    ax.plot(x, dpsi, color=C_ORANGE, lw=2.0, label=r"dipole $\varphi'$")
    ax.fill_between(x, 0, vj * dpsi, color=C_GREEN, alpha=0.35, label=r"$v\varphi'$,  $\int v\varphi'\,\mathrm{d}x = -[v]\,\varphi(x_0)\neq 0$")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.15, 1.25)
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlabel("$x$")
    _panel_title(ax, r"(b) Du Bois-Reymond: a derivative of a bump detects changes of $v$")
    fig.tight_layout()
    _save(fig, "cov_fundamental_lemma.png")


# --------------------------------------------------------------------------
# 4. Affine admissible class and tangent space
# --------------------------------------------------------------------------
def fig_affine_admissible_class():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) abstract picture in the "plane" X
    ax = axes[0]
    ax.set_aspect("equal")
    d = np.array([1.0, 0.45])
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    ug = 1.5 * n  # a point of A off the subspace
    t = np.linspace(-3.2, 3.2, 2)
    ax.plot(t * d[0], t * d[1], color=C_GREY, lw=1.6)
    ax.plot(ug[0] + t * d[0], ug[1] + t * d[1], color=C_BLUE, lw=2.4)
    ax.plot(0, 0, "o", color=C_INK, ms=4)
    ax.annotate("$0$", (0, 0), xytext=(4, -12), textcoords="offset points")
    ax.text(2.55 * d[0] + 0.1, 2.55 * d[1] - 0.35, r"$X_0=W^{1,p}_0(\Omega)^m$", color=C_GREY, fontsize=11)
    ax.text(ug[0] + 1.3 * d[0] + 0.1, ug[1] + 1.3 * d[1] + 0.95, r"$\mathcal{A}=W^{1,p}_g(\Omega)^m = u_g + X_0$", color=C_BLUE, fontsize=11)
    u = ug - 0.6 * d
    ax.plot(*u, "o", color=C_RED, ms=7, zorder=5)
    ax.annotate("$u$", _t(u), xytext=(-2, -18), textcoords="offset points", color=C_RED, fontsize=12)
    v = 2.1 * d
    ax.add_patch(FancyArrowPatch(_t(u), _t(u + v), arrowstyle="-|>", mutation_scale=16, color=C_GREEN, lw=2.2))
    ax.text(*(u + 0.45 * v - 0.42 * n), r"$v\in X_0$", color=C_GREEN, fontsize=12)
    ax.plot(*(u + v), "o", color=C_GREEN, ms=6, zorder=5)
    ax.annotate("$u+tv$", _t(u + v), xytext=(4, -18), textcoords="offset points", color=C_GREEN, fontsize=11)
    # a direction leaving A
    wv = 1.1 * n + 0.2 * d
    ax.add_patch(FancyArrowPatch(_t(u), _t(u + wv), arrowstyle="-|>", mutation_scale=16, color=C_ORANGE, lw=1.8, ls="--"))
    ax.text(*(u + wv + np.array([-0.55, 0.12])), r"$w\notin X_0$: leaves $\mathcal{A}$", color=C_ORANGE, fontsize=10)
    ax.set_xlim(-3.2, 4.4)
    ax.set_ylim(-1.8, 3.4)
    ax.axis("off")
    _panel_title(ax, r"(a) $\mathcal{A}$ is an affine subspace of $X=W^{1,p}(\Omega)^m$; its tangent space is $X_0$")

    # (b) concrete functions
    ax = axes[1]
    x = np.linspace(0, 1, 400)
    g0, g1 = 0.2, 0.7
    uu = g0 + (g1 - g0) * x + 0.25 * np.sin(np.pi * x) - 0.15 * np.sin(2 * np.pi * x)
    vv = np.sin(np.pi * x) ** 2 * np.cos(3 * x)
    for tt, al in [(-0.35, 0.35), (0.35, 0.35)]:
        ax.plot(x, uu + tt * vv, color=C_GREEN, lw=1.6, alpha=al + 0.3)
    ax.plot(x, uu, color=C_RED, lw=2.4, label="$u\\in\\mathcal{A}$")
    ax.plot(x, uu + 0.35 * vv, color=C_GREEN, lw=1.6, label=r"$u+tv\in\mathcal{A}$")
    ax.plot(x, 0.35 * vv, color=C_GREEN, lw=1.4, ls="--", label=r"$tv$,  $v=0$ on $\partial\Omega$")
    ax.plot([0, 1], [g0, g1], "s", color=C_BLUE, ms=7, zorder=5, label=r"boundary data $g$")
    ax.axhline(0, color=C_GREY, lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("$x\\in\\Omega=(0,1)$")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    _panel_title(ax, r"(b) Varying $u$ inside $\mathcal{A}$: the boundary values stay pinned")
    fig.tight_layout()
    _save(fig, "cov_affine_admissible_class.png")


# --------------------------------------------------------------------------
# 5. Differentiability ladder
# --------------------------------------------------------------------------
def fig_differentiability_ladder():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) first variation: slope along a line
    ax = axes[0]
    t = np.linspace(-1.2, 1.2, 300)
    F = lambda p, q: 0.5 * p**2 + 1.5 * q**2 + 0.6 * p * q**2 - 0.8 * p  # a smooth landscape
    u = np.array([0.3, 0.2])
    dirs = [(np.array([1.0, 0.0]), C_BLUE, "$v_1$"), (np.array([0.0, 1.0]), C_GREEN, "$v_2$"), (np.array([0.6, -0.8]), C_ORANGE, "$v_3$")]
    for v, c, lab in dirs:
        vals = np.array([F(*(u + s * v)) for s in t])
        ax.plot(t, vals, color=c, lw=2.0, label=f"$t\\mapsto I[u+t{lab[1:-1]}]$")
        h = 1e-5
        slope = (F(*(u + h * v)) - F(*(u - h * v))) / (2 * h)
        ax.plot(t, F(*u) + slope * t, color=c, lw=1.0, ls="--", alpha=0.8)
    ax.axvline(0, color=C_GREY, lw=0.8)
    ax.plot(0, F(*u), "o", color=C_INK, ms=5, zorder=5)
    ax.set_xlabel("$t$")
    ax.legend(loc="upper center", frameon=False)
    _panel_title(ax, r"(a) First variation: one slope per direction $v$")

    # (b) Gateaux: slopes must assemble into a LINEAR functional of v
    ax = axes[1]
    th = np.linspace(0, 2 * np.pi, 400)
    lin = np.cos(th)  # directional derivative of a differentiable function with gradient e_1
    cub = np.cos(th) ** 3  # directional derivative of x^3/(x^2+y^2) at 0
    ax.plot(th, lin, color=C_GREEN, lw=2.2, label=r"$v\mapsto\delta I[u,v]$ linear: a pure sinusoid in $\theta$")
    ax.plot(th, cub, color=C_RED, lw=2.2, label=r"$h(x,y)=\frac{x^3}{x^2+y^2}$: $\delta h[0,v]=\cos^3\theta$, not linear")
    ax.axhline(0, color=C_GREY, lw=0.8)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlabel(r"direction $v=(\cos\theta,\sin\theta)$")
    ax.set_ylim(-1.25, 1.75)
    ax.legend(loc="upper center", frameon=False)
    _panel_title(ax, r"(b) Gâteaux: the slopes depend linearly on $v$")

    # (c) Frechet: the remainder must be o(|v|) uniformly in direction
    ax = axes[2]
    n = 500
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, ys)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = X**3 * Y / (X**6 + Y**2)
    G = np.nan_to_num(G)
    im = ax.imshow(G, extent=(-1, 1, -1, 1), origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5, interpolation="bilinear")
    for ang in np.linspace(0, np.pi, 9)[:-1]:
        ax.plot([-np.cos(ang), np.cos(ang)], [-np.sin(ang), np.sin(ang)], color=C_GREY, lw=0.7, alpha=0.8)
    xr = np.linspace(-1, 1, 300)
    ax.plot(xr, xr**3, color=C_INK, lw=2.2, label=r"ridge $y=x^3$: $g\equiv\frac{1}{2}$")
    ax.plot(0, 0, "o", color=C_INK, ms=5)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$g(x,y)=\frac{x^3 y}{x^6+y^2}$")
    ax.legend(loc="upper left", frameon=False)
    _panel_title(ax, r"(c) Gâteaux but not Fréchet: rays are blind to a ridge")
    fig.tight_layout()
    _save(fig, "cov_differentiability_ladder.png")


# --------------------------------------------------------------------------
# 6. Extrema of cos(3 pi x)/x
# --------------------------------------------------------------------------
def fig_extrema():
    x = np.linspace(0.1, 1.1, 2000)
    f = np.cos(3 * np.pi * x) / x
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, f, color=C_BLUE, lw=2.2)
    ax.axhline(0, color=C_GREY, lw=0.8)
    # locate the interior extrema numerically
    df = np.gradient(f, x)
    idx = np.where(np.diff(np.sign(df)) != 0)[0]
    pts = [(x[i], f[i]) for i in idx]
    pts.sort(key=lambda p: p[0])
    labels = []
    for (px, py) in pts:
        if py > 0:
            labels.append((px, py, "local maximum", C_TEAL))
        else:
            labels.append((px, py, "local minimum", C_TEAL))
    # endpoints
    ends = [(x[0], f[0], "global maximum\n(at the boundary)", C_RED), (x[-1], f[-1], "", C_GREY)]
    gm = min(pts, key=lambda p: p[1])
    for px, py, lab, c in labels:
        if (px, py) == gm:
            continue
        ax.plot(px, py, "o", color=c, ms=7, zorder=5)
        ax.annotate(lab, (px, py), xytext=(14, 10 if py > 0 else -18), textcoords="offset points", color=c, fontsize=10, arrowprops=dict(arrowstyle="-", color=c, lw=0.8))
    # global minimum = the lowest interior point
    ax.annotate("global minimum", _t(gm), xytext=(14, -26), textcoords="offset points", color=C_RED, fontsize=10, arrowprops=dict(arrowstyle="-", color=C_RED, lw=0.8))
    ax.plot(gm[0], gm[1], "o", color=C_RED, ms=8, zorder=6)
    ax.plot(x[0], f[0], "o", color=C_RED, ms=8, zorder=6)
    ax.annotate(ends[0][2], _t(ends[0][:2]), xytext=(18, -4), textcoords="offset points", color=C_RED, fontsize=10, arrowprops=dict(arrowstyle="-", color=C_RED, lw=0.8))
    ax.set_xlim(0, 1.2)
    ax.set_ylim(-4.3, 6.5)
    ax.set_xlabel("$x$")
    _panel_title(ax, r"Extrema of $f(x)=\cos(3\pi x)/x$ on $\frac{1}{10}\leq x\leq\frac{11}{10}$ (Figure 3 of the script)")
    _save(fig, "cov_extrema.png")


# --------------------------------------------------------------------------
# 7. Second variation: why coercivity needs a uniform constant
# --------------------------------------------------------------------------
def fig_second_variation_coercivity():
    t = np.linspace(-0.85, 0.85, 600)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ks = [1, 2, 5, 20]
    cols = [C_BLUE, C_TEAL, C_GREEN, C_ORANGE]
    for k, c in zip(ks, cols):
        ax.plot(t, t**2 / k - t**4, color=c, lw=2.0, label=f"$k={k}$:  $I[te_k]=t^2/{k}-t^4$")
        r = np.sqrt(1.0 / (2 * k))
        ax.plot([-r, r], [r**2 / k - r**4] * 2, "|", color=c, ms=10, mew=1.8)
    ax.axhline(0, color=C_GREY, lw=0.8)
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.33, 0.2)
    ax.set_xlabel("$t$")
    ax.legend(loc="lower center", frameon=False, ncol=2)
    _panel_title(ax, r"$I[x]=\sum_k (x_k^2/k - x_k^4)$ on $\ell^2$: $\delta^2 I[0;v,v]=2\sum v_k^2/k>0$ for $v\neq 0$, but $0$ is no local minimizer")
    _save(fig, "cov_second_variation_coercivity.png")


# --------------------------------------------------------------------------
# 8. Dirichlet vs natural boundary conditions
# --------------------------------------------------------------------------
def fig_boundary_conditions():
    x = np.linspace(0, 1, 400)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    axes[0].set_ylim(-0.2, 1.0)

    def energy(u):
        du = np.gradient(u, x)
        return np.trapezoid(0.5 * du**2 - u, x)

    # (a) Dirichlet: u(0) = u(1) = 0
    ax = axes[0]
    uD = 0.5 * x * (1 - x)
    phiD = np.sin(np.pi * x) * np.sin(2.2 * x + 0.3)
    phiD = phiD / np.abs(phiD).max()
    for tt in [-0.25, 0.25]:
        comp = uD + tt * phiD
        ax.plot(x, comp, color=C_GREEN, lw=1.4, alpha=0.8, label=f"competitor $u+t\\varphi$,  $I={energy(comp):.3f}$")
    ax.plot(x, uD, color=C_RED, lw=2.6, label=f"minimizer $u=x(1-x)/2$,  $I={energy(uD):.3f}$")
    ax.plot([0, 1], [0, 0], "s", color=C_BLUE, ms=7, zorder=5, label="pinned ends")
    ax.axhline(0, color=C_GREY, lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("$x$")
    ax.legend(loc="upper left", frameon=False)
    _panel_title(ax, r"(a) $\mathcal{A}=W^{1,2}_0$: $-u''=1$,  $u(0)=u(1)=0$ (Dirichlet)")

    # (b) natural: u(0) = 0, right end free
    ax = axes[1]
    uN = x - 0.5 * x**2
    phiN = np.sin(0.5 * np.pi * x) * np.cos(2.5 * x - 0.5)
    phiN = phiN / np.abs(phiN).max()
    for tt in [-0.25, 0.25]:
        comp = uN + tt * phiN
        ax.plot(x, comp, color=C_GREEN, lw=1.4, alpha=0.8, label=f"competitor $u+t\\varphi$,  $I={energy(comp):.3f}$")
    ax.plot(x, uN, color=C_RED, lw=2.6, label=f"minimizer $u=x-x^2/2$,  $I={energy(uN):.3f}$")
    ax.plot([0], [0], "s", color=C_BLUE, ms=7, zorder=5, label="pinned end")
    ax.plot([0.8, 1.0], [uN[-1], uN[-1]], color=C_ORANGE, lw=2.2, ls="--", label=r"natural condition $u'(1)=0$")
    ax.axhline(0, color=C_GREY, lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("$x$")
    ax.legend(loc="upper left", frameon=False)
    _panel_title(ax, r"(b) $\mathcal{A}=\{u\in W^{1,2}: u(0)=0\}$: $-u''=1$,  $u'(1)=0$ (natural)")
    fig.suptitle(r"$I[u]=\int_0^1 \frac{1}{2}u'(x)^2 - u(x)\,\mathrm{d}x$: the same Euler-Lagrange equation, two boundary conditions", y=1.02, fontsize=11)
    fig.tight_layout()
    _save(fig, "cov_boundary_conditions.png")


# --------------------------------------------------------------------------
# 9. Non-convex integrand: double well and sawtooth minimizers
# --------------------------------------------------------------------------
def fig_double_well():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    xi = np.linspace(-1.7, 1.7, 500)
    ax.plot(xi, (xi**2 - 1) ** 2, color=C_BLUE, lw=2.4, label=r"$f(\xi)=(\xi^2-1)^2$")
    ax.plot([-1, 1], [0, 0], "o", color=C_GREEN, ms=8, zorder=5, label=r"wells $\xi=\pm 1$ (where $f=0$)")
    ax.plot([0], [1], "o", color=C_RED, ms=8, zorder=5, label=r"$\xi=0$: $f''(0)=-4<0$ (concave bump)")
    ax.plot([-1, 1], [0, 0], color=C_ORANGE, lw=1.6, ls="--", label="chord between the wells lies below $f$")
    ax.axhline(0, color=C_GREY, lw=0.8)
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-0.2, 3.2)
    ax.set_xlabel(r"$\xi=u'$")
    ax.legend(loc="upper center", frameon=False)
    _panel_title(ax, "(a) The integrand is not convex")

    ax = axes[1]
    x = np.linspace(-1, 1, 1001)

    def sawtooth(corners):
        # piecewise linear, slopes +-1, zero at the ends: corners = interior kinks
        pts = [-1.0] + list(corners) + [1.0]
        y = [0.0]
        s = 1.0
        for i in range(1, len(pts)):
            y.append(y[-1] + s * (pts[i] - pts[i - 1]))
            s = -s
        return np.interp(x, pts, y)

    for corners, c, lab in [
        ([0.0], C_GREEN, r"$u_1$"),
        ([-0.5, 0.0, 0.5], C_TEAL, r"$u_2$"),
        ([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75], C_PURPLE, r"$u_3$"),
    ]:
        y = sawtooth(corners)
        ax.plot(x, y, color=c, lw=2.0, label=f"{lab}: $|u'|=1$ a.e., $I[{lab[1:-1]}]=0$")
    ax.plot(x, 0 * x, color=C_RED, lw=2.6, label=r"$u\equiv 0$: the only $C^2$ solution of the EL equation, $I[0]=2$")
    ax.plot([-1, 1], [0, 0], "s", color=C_BLUE, ms=7, zorder=5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.3, 1.75)
    ax.set_xlabel("$x$")
    ax.legend(loc="upper center", frameon=False, fontsize=8.5)
    _panel_title(ax, r"(b) $I[u]=\int_{-1}^1(u'^2-1)^2\,\mathrm{d}x$ on $W^{1,4}_0$: uncountably many non-smooth minimizers")
    fig.tight_layout()
    _save(fig, "cov_double_well.png")


# --------------------------------------------------------------------------
# 10. Oscillating test function
# --------------------------------------------------------------------------
def fig_oscillating_test_function():
    x = np.linspace(0, 1, 4000)
    psi = _bump(x, 0.5, 0.42)
    psi /= psi.max()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    for eps, c in [(0.08, C_BLUE), (0.02, C_RED)]:
        ax.plot(x, eps * psi * np.sin(x / eps), color=c, lw=1.4, label=rf"$\varphi^{{(\varepsilon)}}=\varepsilon\psi\sin(x/\varepsilon)$,  $\varepsilon={eps}$")
    ax.plot(x, 0.08 * psi, color=C_GREY, lw=1.0, ls="--", label=r"envelope $\pm\varepsilon\psi$")
    ax.plot(x, -0.08 * psi, color=C_GREY, lw=1.0, ls="--")
    ax.axhline(0, color=C_GREY, lw=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("$x$")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    _panel_title(ax, r"(a) $\|\varphi^{(\varepsilon)}\|_{L^\infty}\to 0$, but the slopes $\partial\varphi^{(\varepsilon)}\approx\psi\cos(x/\varepsilon)$ stay $O(1)$")

    ax = axes[1]
    eps = 0.02
    prod = (psi * np.cos(x / eps)) ** 2
    ax.plot(x, prod, color=C_RED, lw=0.9, alpha=0.9, label=r"$(\partial\varphi^{(\varepsilon)})^2\approx\psi^2\cos^2(x/\varepsilon)$")
    win = int(0.06 / (x[1] - x[0]))
    kern = np.ones(win) / win
    avg = np.convolve(prod, kern, mode="same")
    ax.plot(x, avg, color=C_ORANGE, lw=2.2, label="local average over a window of width 0.06")
    ax.plot(x, 0.5 * psi**2, color=C_INK, lw=2.0, ls="--", label=r"weak-$\ast$ limit $\frac{1}{2}\psi^2$")
    ax.set_xlim(0, 1)
    ax.set_xlabel("$x$")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    _panel_title(ax, r"(b) $\cos^2(x/\varepsilon)\,\overset{\ast}{\rightharpoonup}\,\frac{1}{2}$: integrals against $L^1$ functions see only the mean")
    fig.tight_layout()
    _save(fig, "cov_oscillating_test_function.png")


# --------------------------------------------------------------------------
# 11. Rank-1 cone and det along rank-1 lines
# --------------------------------------------------------------------------
def fig_rank_one_cone():
    fig = plt.figure(figsize=(12, 4.8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    # symmetric 2x2 matrices (a b; b c): det = ac - b^2.  With p=(a+c)/2, q=(a-c)/2: det = p^2 - q^2 - b^2.
    # rank-1 directions: det = 0  <=>  p^2 = q^2 + b^2  (double cone, axis p)
    pp = np.linspace(-1, 1, 40)
    ang = np.linspace(0, 2 * np.pi, 60)
    P, T = np.meshgrid(pp, ang)
    Q = np.abs(P) * np.cos(T)
    B = np.abs(P) * np.sin(T)
    ax.plot_surface(Q, B, P, color=C_LIGHT, alpha=0.55, linewidth=0, antialiased=True)
    # rank-1 direction A = a (x) a with a = (1, 0.6): (a b; b c) = (1, .6; .6, .36)
    a_ = np.array([1.0, 0.6])
    A = np.outer(a_, a_)
    Ap, Aq, Ab = (A[0, 0] + A[1, 1]) / 2, (A[0, 0] - A[1, 1]) / 2, A[0, 1]
    s = 0.9 / np.sqrt(Ap**2 + Aq**2 + Ab**2)
    ax.plot([0, s * Aq], [0, s * Ab], [0, s * Ap], color=C_GREEN, lw=3)
    ax.text(s * Aq, s * Ab, s * Ap + 0.08, r"$A=a\otimes a$ (rank 1)", color=C_GREEN, fontsize=10)
    # full-rank direction: B = identity-like (p-axis)
    ax.plot([0, 0], [0, 0], [0, 0.9], color=C_RED, lw=3)
    ax.text(0.02, 0.02, 0.98, r"$B=\mathrm{Id}$ (rank 2)", color=C_RED, fontsize=10)
    ax.set_xlabel(r"$q=(\xi_{11}-\xi_{22})/2$")
    ax.set_ylabel(r"$b=\xi_{12}$")
    ax.set_zlabel(r"$p=(\xi_{11}+\xi_{22})/2$")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=18, azim=-55)
    ax.set_title(r"(a) Rank-1 cone $\{\det=0\}$ in symmetric $2\times 2$ matrices", loc="left", fontsize=10, fontweight="semibold")

    ax2 = fig.add_subplot(1, 2, 2)
    t = np.linspace(-1.5, 1.5, 300)
    xi = np.array([[1.0, 0.2], [0.2, 0.5]])
    detA = np.array([np.linalg.det(xi + tt * A) for tt in t])
    Bm = np.eye(2)
    detB = np.array([np.linalg.det(xi + tt * Bm) for tt in t])
    ax2.plot(t, detA, color=C_GREEN, lw=2.4, label=r"$t\mapsto\det(\xi+tA)$, $A$ rank 1: affine")
    ax2.plot(t, detB, color=C_RED, lw=2.4, label=r"$t\mapsto\det(\xi+tB)$, $B$ rank 2: quadratic")
    ax2.axhline(0, color=C_GREY, lw=0.8)
    ax2.axvline(0, color=C_GREY, lw=0.8)
    ax2.set_xlabel("$t$")
    ax2.legend(loc="upper left", frameon=False)
    ax2.set_title(r"(b) $\det$ along a rank-1 line vs. a rank-2 line", loc="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    _save(fig, "cov_rank_one_cone.png", tight=False)


# --------------------------------------------------------------------------
# 12. Cofactor geometry: volumes vs surface elements
# --------------------------------------------------------------------------
def fig_cofactor_geometry():
    A = np.array([[1.6, 0.6], [0.25, 1.0]])
    cof = np.array([[A[1, 1], -A[1, 0]], [-A[0, 1], A[0, 0]]])
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    img = sq @ A.T
    normals = [(np.array([0.5, 0.0]), np.array([0, -1.0])), (np.array([1.0, 0.5]), np.array([1.0, 0])), (np.array([0.5, 1.0]), np.array([0, 1.0])), (np.array([0.0, 0.5]), np.array([-1.0, 0]))]
    cols = [C_BLUE, C_GREEN, C_ORANGE, C_PURPLE]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.add_patch(Polygon(sq, closed=True, fc=C_LIGHT, ec=C_INK, lw=1.5, alpha=0.6))
    for (m, nvec), c in zip(normals, cols):
        ax.add_patch(FancyArrowPatch(_t(m), _t(m + 0.5 * nvec), arrowstyle="-|>", mutation_scale=14, color=c, lw=2))
    ax.text(0.5, 0.45, r"$|\Omega|=1$", ha="center", fontsize=12)
    ax.set_xlim(-0.9, 2.2)
    ax.set_ylim(-0.9, 2.1)
    ax.set_aspect("equal")
    ax.axis("off")
    _panel_title(ax, r"(a) Unit square, unit outward normals $\nu$")

    ax = axes[1]
    ax.add_patch(Polygon(img, closed=True, fc=C_LIGHT, ec=C_INK, lw=1.5, alpha=0.6))
    for (m, nvec), c in zip(normals, cols):
        mi = A @ m
        cn = cof @ nvec
        ax.add_patch(FancyArrowPatch(_t(mi), _t(mi + 0.5 * cn), arrowstyle="-|>", mutation_scale=14, color=c, lw=2))
    cen = A @ np.array([0.5, 0.5])
    ax.text(cen[0], cen[1] - 0.05, r"$|A\Omega|=\det A=%.2f$" % np.linalg.det(A), ha="center", fontsize=12)
    ax.set_xlim(-0.9, 2.9)
    ax.set_ylim(-0.9, 2.1)
    ax.set_aspect("equal")
    ax.axis("off")
    _panel_title(ax, r"(b) Image under $A$: area $\times\det A$, normals $\nu\mapsto\mathrm{cof}(A)\,\nu$")
    fig.tight_layout()
    _save(fig, "cov_cofactor_geometry.png")


# --------------------------------------------------------------------------
# 13. det is a Null Lagrangian: same boundary values => same integral
# --------------------------------------------------------------------------
def _grid_image(ax, umap, color, nr=7, nt=16):
    th = np.linspace(0, 2 * np.pi, 300)
    for r in np.linspace(0.14, 1.0, nr):
        pts = umap(np.c_[r * np.cos(th), r * np.sin(th)])
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.2 if r < 1 else 2.4, alpha=0.9 if r < 1 else 1)
    rr = np.linspace(0, 1, 120)
    for t in np.linspace(0, 2 * np.pi, nt, endpoint=False):
        pts = umap(np.c_[rr * np.cos(t), rr * np.sin(t)])
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=0.9, alpha=0.8)


def _int_det(umap, n=600):
    xs = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, xs)
    h = xs[1] - xs[0]
    P = np.c_[X.ravel(), Y.ravel()]
    e = 1e-5
    Ux = (umap(P + [e, 0]) - umap(P - [e, 0])) / (2 * e)
    Uy = (umap(P + [0, e]) - umap(P - [0, e])) / (2 * e)
    det = Ux[:, 0] * Uy[:, 1] - Ux[:, 1] * Uy[:, 0]
    inside = (P**2).sum(1) <= 1
    return np.sum(det[inside]) * h * h


def fig_det_null_lagrangian():
    def cut(P):
        r2 = (P**2).sum(1)
        return np.clip(1 - r2, 0, None) ** 2

    def u1(P):
        c = cut(P)
        return P + np.c_[0.55 * c * np.sin(2.5 * P[:, 1]), 0.45 * c * np.cos(2.0 * P[:, 0])]

    def u2(P):
        c = cut(P)
        ang = 2.2 * c
        rot = np.c_[np.cos(ang) * P[:, 0] - np.sin(ang) * P[:, 1], np.sin(ang) * P[:, 0] + np.cos(ang) * P[:, 1]]
        return rot + np.c_[0.35 * c * P[:, 0] ** 2, -0.5 * c]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, um, c, name in [(axes[0], u1, C_BLUE, "u"), (axes[1], u2, C_PURPLE, "v")]:
        _grid_image(ax, um, c)
        val = _int_det(um)
        ax.set_aspect("equal")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.axis("off")
        _panel_title(ax, rf"${name}=\mathrm{{id}}$ on $\partial B_1$:  $\int_{{B_1}}\det D{name}\,\mathrm{{d}}x={val:.3f}\approx\pi$")
    fig.tight_layout()
    _save(fig, "cov_det_null_lagrangian.png")


# --------------------------------------------------------------------------
# 14. Weak continuity of det vs failure for |Du|^2
# --------------------------------------------------------------------------
def fig_weak_continuity_det():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    k = 6
    ax = axes[0]
    L = np.pi
    for yv in np.linspace(0, L, 9):
        xs = np.linspace(0, L, 400)
        ax.plot(xs, yv + np.sin(k * xs) / k, color=C_BLUE, lw=1.1)
    for xv in np.linspace(0, L, 13):
        ys = np.linspace(0, L, 2)
        ax.plot([xv, xv], ys + np.sin(k * xv) / k, color=C_BLUE, lw=1.1)
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, L + 0.1)
    ax.set_ylim(-0.3, L + 0.3)
    ax.axis("off")
    _panel_title(ax, rf"(a) $u_k(x)=x+\frac{{1}}{{k}}\sin(kx_1)\,e_2$,  $k={k}$")

    ax = axes[1]
    x1 = np.linspace(0, L, 2000)
    det = np.ones_like(x1)
    nrm = 2 + np.cos(k * x1) ** 2
    ax.plot(x1, nrm, color=C_RED, lw=1.4, label=r"$|Du_k|^2=2+\cos^2(kx_1)$")
    ax.axhline(2.5, color=C_RED, lw=2.0, ls="--", label=r"weak limit of $|Du_k|^2$:  $2.5$")
    ax.axhline(2.0, color=C_ORANGE, lw=2.0, ls=":", label=r"$|D\,\mathrm{id}|^2=2$   (so $|Du_k|^2\not\rightharpoonup|Du|^2$)")
    ax.plot(x1, det, color=C_GREEN, lw=2.6, label=r"$\det Du_k\equiv 1=\det D\,\mathrm{id}$")
    ax.set_xlim(0, L)
    ax.set_ylim(0.5, 3.6)
    ax.set_xlabel("$x_1$")
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=8.5)
    _panel_title(ax, r"(b) $\det Du_k$ passes to the weak limit, $|Du_k|^2$ does not")
    fig.tight_layout()
    _save(fig, "cov_weak_continuity_det.png")


# --------------------------------------------------------------------------
# 15. Brouwer: the retraction construction, and area that cannot vanish
# --------------------------------------------------------------------------
def fig_brouwer_retraction():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    ax = axes[0]
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=C_INK, lw=1.8)
    ax.fill(np.cos(th), np.sin(th), color=C_LIGHT, alpha=0.35)

    def exit_point(x, ux):
        d = x - ux
        # |ux + t d|^2 = 1, larger root
        a, b, c = d @ d, 2 * ux @ d, ux @ ux - 1
        t = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
        return ux + t * d

    pts = [(np.array([0.15, 0.35]), np.array([-0.45, -0.25]), C_BLUE), (np.array([-0.55, 0.3]), np.array([0.2, 0.55]), C_GREEN), (np.array([0.5, -0.5]), np.array([0.1, -0.1]), C_PURPLE)]
    for x, ux, c in pts:
        w = exit_point(x, ux)
        ax.plot([ux[0], w[0]], [ux[1], w[1]], color=c, lw=1.2, ls="--")
        ax.add_patch(FancyArrowPatch(_t(ux), _t(w), arrowstyle="-|>", mutation_scale=14, color=c, lw=1.2))
        ax.plot(*x, "o", color=c, ms=6)
        ax.plot(*ux, "s", color=c, ms=6)
        ax.plot(*w, "D", color=c, ms=7, mfc="white", mew=1.8)
    # annotations for the first point
    x, ux, c = pts[0]
    w = exit_point(x, ux)
    ax.annotate("$x$", _t(x), xytext=(8, -14), textcoords="offset points", color=c, fontsize=12)
    ax.annotate("$u(x)$", _t(ux), xytext=(-30, -14), textcoords="offset points", color=c, fontsize=12)
    ax.annotate("$w(x)$", _t(w), xytext=(6, 4), textcoords="offset points", color=c, fontsize=12)
    bx = np.array([np.cos(2.4), np.sin(2.4)])
    ax.plot(*bx, "o", color=C_RED, ms=6)
    ax.annotate("$x\\in\\partial B_1$:  $w(x)=x$", _t(bx), xytext=(-110, 8), textcoords="offset points", color=C_RED, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")
    _panel_title(ax, r"(a) The ray from $u(x)$ through $x$ exits at $w(x)\in\partial B_1$")

    ax = axes[1]
    r = np.linspace(0, 1, 1000)
    for dlt, c in [(0.6, C_BLUE), (0.3, C_TEAL), (0.12, C_ORANGE)]:
        dens = np.where(r < dlt, 1.0 / dlt**2, 0.0)
        ax.plot(r, dens, color=c, lw=2.2, label=rf"$\delta={dlt}$:  $\det Dw_\delta=\delta^{{-2}}$ on $|x|<\delta$, $0$ outside")
        ax.fill_between(r, 0, dens, color=c, alpha=0.12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 75)
    ax.set_xlabel("$|x|$")
    ax.set_ylabel(r"$\det Dw_\delta$")
    ax.legend(loc="upper right", frameon=False)
    _panel_title(ax, r"(b) $w_\delta(x)=x/\max(|x|,\delta)$: area concentrates but never vanishes")
    fig.tight_layout()
    _save(fig, "cov_brouwer_retraction.png")


if __name__ == "__main__":
    fig_brachistochrone()
    fig_growth_envelopes()
    fig_fundamental_lemma()
    fig_affine_admissible_class()
    fig_differentiability_ladder()
    fig_extrema()
    fig_second_variation_coercivity()
    fig_boundary_conditions()
    fig_double_well()
    fig_oscillating_test_function()
    fig_rank_one_cone()
    fig_cofactor_geometry()
    fig_det_null_lagrangian()
    fig_weak_continuity_det()
    fig_brouwer_retraction()
