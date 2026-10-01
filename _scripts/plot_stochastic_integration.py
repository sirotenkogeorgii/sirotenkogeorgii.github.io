"""
Figures for Chapter 4 (Stochastic Integration and SDEs) of
subpages/books/sde_hd/index.md.

Writes into assets/images/notes/sdes_diffusion_models/:

  bm_variation_scaling.png       - why Riemann-Stieltjes fails: Brownian zoom
                                   cascade and the p-variation sums of a
                                   Brownian vs. a smooth path
  simple_integral_gambling.png   - Def 4.1/4.2: stakes, increments, winnings;
                                   non-anticipating vs. clairvoyant betting
  evaluation_point.png           - left / mid / right Riemann sums of int W dW
                                   converge to three different limits
  qv_clock_compensator.png       - Thm 4.4: <M> as variance clock, M^2 - <M>
                                   as a martingale, M on its own clock
  ito_isometry_pythagoras.png    - Thm 4.6: diagonal Gram matrix of the
                                   increments; isometry holds iff adapted
  density_extension.png          - Thm 4.7 + Def 4.10: simple approximations
                                   and the isometry transporting their error
  localisation.png               - Def 4.12: an integrand in P* but not L*,
                                   stopping times R_n and consistent integrals
  semimartingale_decomposition.png - Def 4.14: X = X0 + B + M; only M is seen
                                   by quadratic variation
  construction_ladder.png        - Summary: the four rungs of the construction
  ito_rule_taylor.png            - Thm 4.15: convexity gap, decomposition of
                                   f(W_t), the dt/dW multiplication table
  quadratic_covariation.png      - Def 4.17: correlated increments, <W1,W2>,
                                   polarisation
  partial_integration_rectangle.png - Cor 4.18: area bookkeeping and the
                                   surviving corner terms
  gbm_mean_vs_median.png         - Thm 4.22: mean grows while typical paths die
  ou_variation_of_constants.png  - Thm 4.23: OU = decayed start + fading memory
                                   of noise; one-shot Gaussian sampling
  euler_maruyama_mechanics.png   - Sec 4.3: one EM step = drift shift + Gaussian
                                   kick; step-size explosion
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Ellipse
from matplotlib.colors import to_rgb
from math import erf

OUT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "assets",
        "images",
        "notes",
        "sdes_diffusion_models",
    )
)
os.makedirs(OUT, exist_ok=True)

C_BLUE = "#2c3e94"
C_ORANGE = "#e67e22"
C_GREEN = "#1f9d55"
C_RED = "#c0392b"
C_PURPLE = "#534AB7"
C_GREY = "#5a6171"
C_INK = "#222222"


def tint(color, f):
    """Blend a color toward white; f = 0 gives the color, f = 1 gives white."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def gauss(x, m, v):
    return np.exp(-((x - m) ** 2) / (2 * v)) / np.sqrt(2 * np.pi * v)


def Phi(z):
    return 0.5 * (1 + erf(z / np.sqrt(2)))


def bm(rng, N, n_paths=None, T=1.0):
    """Brownian path(s) on a uniform grid with N steps; shape (N+1,) or (N+1, n)."""
    shape = (N,) if n_paths is None else (N, n_paths)
    dW = np.sqrt(T / N) * rng.standard_normal(shape)
    zero = np.zeros((1,) + shape[1:])
    return np.linspace(0, T, N + 1), np.concatenate([zero, np.cumsum(dW, axis=0)])


def _save(fig, name):
    out = os.path.join(OUT, name)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _grid(ax):
    ax.grid(True, lw=0.3, alpha=0.4)


def _panel(ax, letter):
    ax.text(0.012, 0.985, f"({letter})", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left", color=C_INK)


# ----------------------------------------------------------------------
# Figure 1: why classical integration fails
# ----------------------------------------------------------------------
def fig_bm_variation_scaling():
    rng = np.random.default_rng(11)
    N = 2**18
    t, W = bm(rng, N)
    g = 0.55 * np.sin(2 * np.pi * t) + 0.3 * np.sin(5 * np.pi * t + 0.4)

    fig = plt.figure(figsize=(12.6, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.42)
    top = gs[0].subgridspec(1, 3, wspace=0.25)
    bot = gs[1].subgridspec(1, 2, wspace=0.22)

    c = 0.4313
    ic = int(round(c * N))
    widths = [1.0, 1 / 16, 1 / 256]
    windows = []
    for w in widths:
        lo, hi = (0.0, 1.0) if w == 1.0 else (c - w / 2, c + w / 2)
        i0, i1 = int(round(lo * N)), int(round(hi * N))
        half = 2.3 * np.sqrt(w)            # diffusive scaling: space ~ sqrt(time)
        windows.append((lo, hi, i0, i1, W[ic] - half, W[ic] + half))

    titles = ["the whole path on $[0,1]$",
              r"zoom: time $\times 16$, space $\times 4$",
              r"zoom again: time $\times 16$, space $\times 4$"]
    letters = "abc"
    for k in range(3):
        ax = fig.add_subplot(top[k])
        lo, hi, i0, i1, ylo, yhi = windows[k]
        ax.plot(t[i0:i1 + 1], W[i0:i1 + 1], color=C_BLUE, lw=0.6)
        if k < 2:
            nlo, nhi, _, _, nylo, nyhi = windows[k + 1]
            ax.add_patch(Rectangle((nlo, nylo), nhi - nlo, nyhi - nylo, fill=False,
                                   ec=C_ORANGE, lw=1.4, ls="--"))
        ax.set_xlim(lo, hi)
        ax.set_ylim(ylo, yhi)
        slope = (yhi - ylo) / (hi - lo)
        ax.text(0.97, 0.05, f"box height / width = {slope:.0f}",
                transform=ax.transAxes, ha="right", fontsize=8.5, color=C_RED,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        ax.set_title(titles[k], fontsize=10.5)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_xlabel(r"$t$", fontsize=9)
        _grid(ax)
        _panel(ax, letters[k])

    # p-variation sums over dyadic partitions
    ks = np.arange(1, 19)
    ns = 2**ks

    def pvar(path, p):
        return np.array([np.sum(np.abs(np.diff(path[::N // n])) ** p) for n in ns])

    ax_d = fig.add_subplot(bot[0])
    ax_e = fig.add_subplot(bot[1], sharey=ax_d)
    cols = {1: C_RED, 2: C_BLUE, 3: C_GREEN}
    labels = {1: r"$\sum |\Delta|$  (total variation)",
              2: r"$\sum |\Delta|^2$  (quadratic variation)",
              3: r"$\sum |\Delta|^3$"}
    for p in (1, 2, 3):
        ax_d.loglog(ns, pvar(W, p), "o-", ms=3, color=cols[p], lw=1.5, label=labels[p])
        ax_e.loglog(ns, pvar(g, p), "o-", ms=3, color=cols[p], lw=1.5, label=labels[p])
    ax_d.loglog(ns, np.sqrt(2 * ns / np.pi), color=C_RED, ls=":", lw=1.1)
    ax_d.text(60, 40, r"$\approx\sqrt{2n/\pi}\to\infty$", color=C_RED, fontsize=9.5)
    ax_d.axhline(1.0, color=C_BLUE, ls=":", lw=1.1)
    ax_d.text(3e3, 0.25, r"$\to\langle W\rangle_1 = 1$", color=C_BLUE, fontsize=9.5)
    ax_d.text(3e3, 2.5e-3, r"$\to 0$", color=C_GREEN, fontsize=9.5)
    ax_d.set_title("Brownian path: only the square sum has a finite, non-zero limit",
                   fontsize=10.5)
    ax_e.text(3e3, 12, r"$\to \mathrm{TV}(g) < \infty$", color=C_RED, fontsize=9.5)
    ax_e.text(3e3, 2.5e-4, r"$\sim 1/n \to 0$", color=C_BLUE, fontsize=9.5)
    ax_e.set_title(r"smooth path $g$: total variation finite, squares vanish",
                   fontsize=10.5)
    for ax, L in ((ax_d, "d"), (ax_e, "e")):
        ax.set_xlabel(r"number $n$ of partition intervals of $[0,1]$")
        _grid(ax)
        _panel(ax, L)
    ax_d.set_ylim(1e-9, 1e3)
    ax_d.legend(fontsize=8.5, loc="lower left", framealpha=0.92)
    _save(fig, "bm_variation_scaling.png")


# ----------------------------------------------------------------------
# Figure 2: the simple integral as gambling
# ----------------------------------------------------------------------
def fig_simple_integral_gambling():
    rng = np.random.default_rng(5)
    N, n_int = 2**12, 8
    t, M = bm(rng, N)
    tk = np.linspace(0, 1, n_int + 1)
    idx = (tk * N).astype(int)

    xi = np.empty(n_int)
    xi[0] = 1.0
    for i in range(1, n_int):
        xi[i] = np.clip(-2.0 * M[idx[i]], -1, 1)

    I = np.zeros(N + 1)
    for i in range(n_int):
        a, b = idx[i], idx[i + 1]
        I[a:b + 1] = I[a] + xi[i] * (M[a:b + 1] - M[a])

    fig = plt.figure(figsize=(12.6, 6.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.45, 1], height_ratios=[1.2, 0.8, 1.1],
                          hspace=0.12, wspace=0.2)
    ax_m = fig.add_subplot(gs[0, 0])
    ax_x = fig.add_subplot(gs[1, 0], sharex=ax_m)
    ax_i = fig.add_subplot(gs[2, 0], sharex=ax_m)
    ax_h = fig.add_subplot(gs[:, 1])

    for i in range(n_int):
        col = C_GREEN if xi[i] > 0 else C_RED
        for ax in (ax_m, ax_x, ax_i):
            ax.axvspan(tk[i], tk[i + 1], color=col, alpha=0.06 + 0.14 * abs(xi[i]), lw=0)
        ax_m.plot(t[idx[i]:idx[i + 1] + 1], M[idx[i]:idx[i + 1] + 1], color=C_BLUE, lw=0.9)
        ax_m.plot(tk[i], M[idx[i]], "o", color=C_INK, ms=4, zorder=5)
        ax_x.plot([tk[i], tk[i + 1]], [xi[i], xi[i]], color=col, lw=2.4)
        ax_x.plot(tk[i], xi[i], "o", mfc="white", mec=col, ms=5, zorder=5)
        ax_x.plot(tk[i + 1], xi[i], "o", color=col, ms=5, zorder=5)
    for x in tk:
        for ax in (ax_m, ax_x, ax_i):
            ax.axvline(x, color=C_GREY, lw=0.5, alpha=0.6)
    ax_m.axhline(0, color=C_INK, lw=0.5)
    ax_m.annotate(r"stake for round 4 decided here,"
                  "\n" r"from $M_{t_3}$ only", (tk[3], M[idx[3]]),
                  xytext=(0.47, 0.82), textcoords="axes fraction", fontsize=8.5,
                  arrowprops=dict(arrowstyle="->", color=C_INK, lw=0.8))
    ax_m.set_ylabel(r"game $M_t$")
    ax_m.set_title(r"a non-anticipating strategy: $\xi_i = \mathrm{clip}(-2M_{t_i}, -1, 1)$"
                   "  (bet on a return to 0)", fontsize=10.5)
    ax_x.axhline(0, color=C_INK, lw=0.5)
    ax_x.set_ylim(-1.3, 1.3)
    ax_x.set_ylabel(r"stake $X_t$")
    ax_x.text(0.995, 0.86, r"constant on $(t_i, t_{i+1}]$", transform=ax_x.transAxes,
              ha="right", fontsize=8.5, color=C_GREY)
    ax_i.plot(t, I, color=C_INK, lw=1.2)
    ax_i.axhline(0, color=C_INK, lw=0.5)
    ax_i.set_ylabel(r"$\int_0^t X_s\,\mathrm{d}M_s$")
    ax_i.set_xlabel(r"$t$")
    ax_i.text(0.01, 0.06, "winnings: stake × increment, round by round",
              transform=ax_i.transAxes, fontsize=8.5, color=C_GREY)
    for ax in (ax_m, ax_x):
        plt.setp(ax.get_xticklabels(), visible=False)
    _panel(ax_m, "a")
    _panel(ax_x, "b")
    _panel(ax_i, "c")
    ax_m.set_xlim(0, 1)

    # many games: honest vs clairvoyant
    n_mc = 40000
    dM = np.sqrt(1 / n_int) * rng.standard_normal((n_mc, n_int))
    Mk = np.concatenate([np.zeros((n_mc, 1)), np.cumsum(dM, axis=1)], axis=1)
    xi_h = np.clip(-2.0 * Mk[:, :-1], -1, 1)
    xi_h[:, 0] = 1.0
    win_h = np.sum(xi_h * dM, axis=1)
    win_c = np.sum(np.sign(dM) * dM, axis=1)
    bins = np.linspace(-3, 6, 91)
    ax_h.hist(win_h, bins=bins, density=True, color=tint(C_BLUE, 0.35),
              label="non-anticipating stakes")
    ax_h.hist(win_c, bins=bins, density=True, color=tint(C_RED, 0.35), alpha=0.85,
              label=r"clairvoyant: $\xi_i = \mathrm{sign}(M_{t_{i+1}} - M_{t_i})$")
    ax_h.axvline(win_h.mean(), color=C_BLUE, lw=2)
    ax_h.axvline(win_c.mean(), color=C_RED, lw=2)
    ax_h.text(win_h.mean() - 0.95, 0.36, f"mean {win_h.mean():+.3f}\n(fair game\nstays fair)",
              color=C_BLUE, fontsize=9, ha="right")
    ax_h.text(win_c.mean() + 1.0, 0.4, f"mean {win_c.mean():.2f}\n(peeking pays)",
              color=C_RED, fontsize=9)
    ax_h.set_xlabel(r"final winnings $\int_0^1 X_s\,\mathrm{d}M_s$")
    ax_h.set_ylabel("density over 40 000 games")
    ax_h.set_title("why the stake must be $\\mathcal{F}_{t_i}$-measurable", fontsize=10.5)
    ax_h.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.06, 1.0))
    _grid(ax_h)
    _panel(ax_h, "d")
    _save(fig, "simple_integral_gambling.png")


# ----------------------------------------------------------------------
# Figure 3: the evaluation point matters
# ----------------------------------------------------------------------
def _three_sums(path, n):
    """Running left / mid / right Riemann sums of int path d path, n intervals."""
    N = len(path) - 1
    s = N // n
    L = path[0:N:s]
    Mi = path[s // 2:N:s]
    R = path[s:N + 1:s]
    d = R - L
    z = np.zeros(1)
    return (np.concatenate([z, np.cumsum(L * d)]),
            np.concatenate([z, np.cumsum(Mi * d)]),
            np.concatenate([z, np.cumsum(R * d)]))


def fig_evaluation_point():
    rng = np.random.default_rng(23)
    N, n = 2**14, 2**9
    t, W = bm(rng, N)
    g = 0.8 * np.sin(2 * np.pi * t) + 0.5 * t
    tn = np.linspace(0, 1, n + 1)

    fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.3),
                            gridspec_kw={"width_ratios": [1.15, 1, 1]})
    cols = (C_BLUE, C_GREEN, C_RED)
    names = ("left point", "midpoint", "right point")
    ax = axs[0]
    for S, c, nm in zip(_three_sums(W, n), cols, names):
        ax.plot(tn, S, color=c, lw=1.6, label=nm)
    ax.plot(t, (W**2 - t) / 2, color=C_BLUE, ls=":", lw=1.2)
    ax.plot(t, W**2 / 2, color=C_GREEN, ls=":", lw=1.2)
    ax.plot(t, (W**2 + t) / 2, color=C_RED, ls=":", lw=1.2)
    ax.text(1.01, (W[-1] ** 2 - 1) / 2, r"$\frac{W_t^2 - t}{2}$  (Itô)", color=C_BLUE,
            fontsize=9, va="center")
    ax.text(1.01, W[-1] ** 2 / 2, r"$\frac{W_t^2}{2}$", color=C_GREEN, fontsize=9,
            va="center")
    ax.text(1.01, (W[-1] ** 2 + 1) / 2, r"$\frac{W_t^2 + t}{2}$", color=C_RED,
            fontsize=9, va="center")
    ax.set_xlim(0, 1)
    ax.set_title(r"$\sum W_{\tau_i}(W_{t_{i+1}} - W_{t_i})$ along one Brownian path",
                 fontsize=10.5)
    ax.set_xlabel(r"$t$")
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.06, 1.0), title=r"$\tau_i$ =", title_fontsize=8.5)
    _grid(ax)
    _panel(ax, "a")

    ax = axs[1]
    for S, c, nm, ls, lw in zip(_three_sums(g, n), cols, names, ("-", "--", ":"),
                                (3.4, 2.0, 1.6)):
        ax.plot(tn, S, color=c, lw=lw, ls=ls, label=nm)
    ax.set_xlim(0, 1)
    gap = np.sum(np.diff(g[:: N // n]) ** 2)
    ax.text(0.97, 0.62, f"right $-$ left $= \\sum(\\Delta g)^2 = {gap:.3f}$\n"
            r"$\to 0$ like $1/n$", transform=ax.transAxes, ha="right", fontsize=9,
            color=C_GREY)
    ax.set_title(r"smooth integrator $g$: all three agree, $= g_t^2/2$", fontsize=10.5)
    ax.set_xlabel(r"$t$")
    ax.legend(fontsize=8.5, loc="lower left")
    _grid(ax)
    _panel(ax, "b")

    ax = axs[2]
    n_mc, m = 30000, 64
    h = np.sqrt(0.5 / m) * rng.standard_normal((n_mc, 2 * m))
    P = np.concatenate([np.zeros((n_mc, 1)), np.cumsum(h, axis=1)], axis=1)
    L, Mi, R = P[:, 0:2 * m:2], P[:, 1:2 * m:2], P[:, 2::2]
    d = R - L
    bins = np.linspace(-1, 4, 101)
    for S, c, nm in zip((np.sum(L * d, 1), np.sum(Mi * d, 1), np.sum(R * d, 1)),
                        cols, names):
        ax.hist(S, bins=bins, density=True, histtype="step", color=c, lw=1.6, label=nm)
        ax.axvline(S.mean(), color=c, ls="--", lw=1.2)
        ax.text(S.mean() + 0.04, 1.55, f"{S.mean():.2f}", color=c, fontsize=9)
    ax.set_xlim(-0.8, 3)
    ax.set_title(r"30 000 paths at $t = 1$: means $0,\ \frac{1}{2},\ 1$", fontsize=10.5)
    ax.set_xlabel(r"value of the sum at $t=1$")
    ax.legend(fontsize=8.5, loc="upper right")
    _grid(ax)
    _panel(ax, "c")
    fig.tight_layout()
    _save(fig, "evaluation_point.png")


# ----------------------------------------------------------------------
# Figure 4: quadratic variation as a clock, M^2 - <M> as a martingale
# ----------------------------------------------------------------------
def fig_qv_clock_compensator():
    rng = np.random.default_rng(8)
    N, n_mc, n_show = 2000, 4000, 30
    t = np.linspace(0, 1, N + 1)
    dt = 1 / N
    sig = lambda s: 0.35 + 1.3 * np.exp(-(((s - 0.62) / 0.09) ** 2))
    s_left = sig(t[:-1])
    A = np.concatenate([[0], np.cumsum(s_left**2 * dt)])
    dW = np.sqrt(dt) * rng.standard_normal((N, n_mc))
    M = np.concatenate([np.zeros((1, n_mc)), np.cumsum(s_left[:, None] * dW, axis=0)])
    burst = (t >= 0.5) & (t <= 0.74)

    fig = plt.figure(figsize=(13.4, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 3], width_ratios=[1.1, 1, 1],
                          hspace=0.08, wspace=0.24)
    ax_s = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[1, 0], sharex=ax_s)
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[:, 2])

    ax_s.fill_between(t, sig(t) ** 2, color=tint(C_ORANGE, 0.45), lw=0)
    ax_s.plot(t, sig(t) ** 2, color=C_ORANGE, lw=1.4)
    ax_s.set_ylabel(r"$\sigma(t)^2$", fontsize=9)
    ax_s.text(0.02, 0.55, r"speed of the clock $\mathrm{d}\langle M\rangle_t/\mathrm{d}t$",
              transform=ax_s.transAxes, fontsize=8.5, color=C_ORANGE)
    plt.setp(ax_s.get_xticklabels(), visible=False)
    ax_s.tick_params(labelsize=8)
    _panel(ax_s, "a")

    ax_a.fill_between(t, -2 * np.sqrt(A), 2 * np.sqrt(A), color=tint(C_BLUE, 0.85), lw=0)
    ax_a.plot(t, 2 * np.sqrt(A), color=C_BLUE, ls="--", lw=1)
    ax_a.plot(t, -2 * np.sqrt(A), color=C_BLUE, ls="--", lw=1)
    ax_a.plot(t, M[:, :n_show], color=C_BLUE, lw=0.5, alpha=0.35)
    for j in range(3):
        ax_a.plot(t, M[:, j], color=C_BLUE, lw=1)
        ax_a.plot(t[burst], M[burst, j], color=C_ORANGE, lw=1.3)
    ax_a.text(0.03, 0.9, r"band $\pm 2\sqrt{\langle M\rangle_t}$", transform=ax_a.transAxes,
              fontsize=9, color=C_BLUE)
    ax_a.set_xlabel(r"real time $t$")
    ax_a.set_ylabel(r"$M_t = \int_0^t \sigma(s)\,\mathrm{d}W_s$")
    ax_a.set_xlim(0, 1)
    _grid(ax_a)

    C = M**2 - A[:, None]
    ax_b.plot(t, C[:, :n_show], color=C_GREY, lw=0.5, alpha=0.4)
    ax_b.plot(t, (M**2).mean(1), color=C_BLUE, lw=2.2, label=r"$\mathbb{E}[M_t^2]$ (4000 paths)")
    ax_b.plot(t, A, color=C_ORANGE, lw=1.6, ls="--", label=r"$\langle M\rangle_t$")
    ax_b.plot(t, C.mean(1), color=C_GREEN, lw=2.2,
              label=r"$\mathbb{E}[M_t^2 - \langle M\rangle_t] \approx 0$")
    ax_b.set_ylim(-1.2, 1.8)
    ax_b.set_xlim(0, 1)
    ax_b.set_xlabel(r"$t$")
    ax_b.set_title(r"grey: paths of $M_t^2 - \langle M\rangle_t$ — a martingale",
                   fontsize=10.5)
    ax_b.legend(fontsize=8.5, loc="lower left", framealpha=0.92)
    _grid(ax_b)
    _panel(ax_b, "b")

    u = np.linspace(0, A[-1], 200)
    ax_c.fill_between(u, -2 * np.sqrt(u), 2 * np.sqrt(u), color=tint(C_BLUE, 0.85), lw=0)
    ax_c.plot(A, M[:, :n_show], color=C_BLUE, lw=0.5, alpha=0.35)
    for j in range(3):
        ax_c.plot(A, M[:, j], color=C_BLUE, lw=1)
        ax_c.plot(A[burst], M[burst, j], color=C_ORANGE, lw=1.3)
    ax_c.set_xlabel(r"intrinsic time $u = \langle M\rangle_t$")
    ax_c.set_title("the same paths on their own clock:\nplain Brownian motion",
                   fontsize=10.5)
    ax_c.text(0.03, 0.9, r"band $\pm 2\sqrt{u}$", transform=ax_c.transAxes,
              fontsize=9, color=C_BLUE)
    ax_c.text(0.45, 0.04, "orange = the turbulent stretch\n$0.5 \\leq t \\leq 0.74$",
              transform=ax_c.transAxes, fontsize=8.5, color=C_ORANGE,
              bbox=dict(fc="white", ec="none", alpha=0.85))
    ax_c.set_xlim(0, A[-1])
    _grid(ax_c)
    _panel(ax_c, "c")
    _save(fig, "qv_clock_compensator.png")


# ----------------------------------------------------------------------
# Figure 5: Ito isometry = Pythagoras for orthogonal increments
# ----------------------------------------------------------------------
def fig_ito_isometry():
    rng = np.random.default_rng(3)
    n, n_mc = 12, 200000
    dt = 1 / n
    dW = np.sqrt(dt) * rng.standard_normal((n_mc, n))
    Wk = np.concatenate([np.zeros((n_mc, 1)), np.cumsum(dW, axis=1)], axis=1)
    Z_ad = Wk[:, :-1] * dW
    Z_in = Wk[:, -1:] * dW
    G_ad = Z_ad.T @ Z_ad / n_mc
    G_in = Z_in.T @ Z_in / n_mc
    v = max(np.abs(G_ad).max(), np.abs(G_in).max())

    fig = plt.figure(figsize=(14.2, 4.7))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 0.22, 1.2], wspace=0.08)
    axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[4])]
    cax = fig.add_subplot(gs[2])
    im = None
    for ax, G, ttl, L in ((axs[0], G_ad, r"adapted: $\xi_i = W_{t_i}$", "a"),
                          (axs[1], G_in, r"insider: $\xi_i = W_1$ (knows the end)", "b")):
        im = ax.imshow(G, cmap="RdBu_r", vmin=-v, vmax=v)
        diag, off = np.trace(G), G.sum() - np.trace(G)
        ax.set_title(ttl + "\n" + rf"diagonal sum {diag:.2f},  off-diagonal sum {off:+.2f}",
                     fontsize=10)
        ax.set_xlabel(r"round $j$")
        if L == "a":
            ax.set_ylabel(r"round $i$")
        ax.set_xticks(range(0, n, 3))
        ax.set_yticks(range(0, n, 3))
        if L == "b":
            ax.set_yticklabels([])
        ax.text(0.97, 0.97, f"({L})", transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top", ha="right")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\mathbb{E}[\xi_i \Delta W_i\, \xi_j \Delta W_j]$", fontsize=9)

    # scatter: isometry check for several integrands
    N, n_mc2 = 256, 20000
    t, W = bm(rng, N, n_mc2)
    dW2 = np.diff(W, axis=0)
    tl = t[:-1, None]
    Wl = W[:-1]
    shift = int(0.25 * N)
    Wf = np.concatenate([W[shift:-1], np.repeat(W[-1:], shift, axis=0)])
    cases = [
        (r"$s\,W_s$", tl * Wl, True), (r"$W_s$", Wl, True), (r"$\cos W_s$", np.cos(Wl), True),
        (r"$\mathrm{sign}\, W_s$", np.sign(Wl) + (Wl == 0), True),
        (r"$e^{W_s/2}$", np.exp(Wl / 2), True), (r"$2W_s$", 2 * Wl, True),
        (r"$1+s$", np.broadcast_to(1 + tl, Wl.shape), True),
        (r"$W_1$", np.broadcast_to(W[-1], Wl.shape), False),
        (r"$W_{s+1/4}$", Wf, False),
        (r"$W_1 - W_s$", W[-1] - Wl, False),
    ]
    ax = axs[2]
    lim = 0
    for nm, X, ok in cases:
        x = np.mean(np.sum(X**2, axis=0) / N)
        y = np.mean(np.sum(X * dW2, axis=0) ** 2)
        lim = max(lim, x, y)
        ax.plot(x, y, "o" if ok else "X", color=C_BLUE if ok else C_RED, ms=8 if ok else 9)
        ax.annotate(nm, (x, y), xytext=(6, -3 if ok else 3), textcoords="offset points",
                    fontsize=8.5, color=C_BLUE if ok else C_RED)
    ax.plot([0, 1.15 * lim], [0, 1.15 * lim], color=C_GREY, ls="--", lw=1)
    ax.set_xlim(0, 1.15 * lim)
    ax.set_ylim(0, 1.15 * lim)
    ax.set_xlabel(r"$\mathbb{E}\int_0^1 X_s^2\,\mathrm{d}s$")
    ax.set_ylabel(r"$\mathbb{E}\left[\left(\int_0^1 X_s\,\mathrm{d}W_s\right)^2\right]$")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_title("isometry: adapted (blue) on the diagonal,\nanticipating (red) off it",
                 fontsize=10.5)
    _grid(ax)
    _panel(ax, "c")
    _save(fig, "ito_isometry_pythagoras.png")


# ----------------------------------------------------------------------
# Figure 6: density of simple processes + extension by isometry
# ----------------------------------------------------------------------
def fig_density_extension():
    rng = np.random.default_rng(31)
    N, n_mc = 2**12, 3000
    t, W = bm(rng, N, n_mc)
    X = np.sin(4 * W) + W
    dW = np.diff(W, axis=0)
    I_ref = np.concatenate([np.zeros((1, n_mc)), np.cumsum(X[:-1] * dW, axis=0)])

    def simple(Xp, n):
        s = N // n
        return np.repeat(Xp[:-1:s], s, axis=0)   # value on (t_k, t_{k+1}] = X_{t_k}

    fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.3))
    shows = [(4, tint(C_ORANGE, 0.35)), (16, C_ORANGE), (64, C_RED)]
    ax = axs[0]
    ax.plot(t, X[:, 0], color=C_INK, lw=1.2, label=r"$X_s = \sin 4W_s + W_s$")
    for n, c in shows:
        Xn = simple(X[:, 0], n)
        ax.step(t[:-1], Xn, where="post", color=c, lw=1.3, label=rf"$X^n$, $n={n}$")
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$s$")
    ax.set_title("simple approximations (frozen at left endpoints)", fontsize=10.5)
    ax.legend(fontsize=8.3, loc="lower center", framealpha=0.92)
    _grid(ax)
    _panel(ax, "a")

    ax = axs[1]
    ax.plot(t, I_ref[:, 0], color=C_INK, lw=1.4, label=r"$\int_0^t X\,\mathrm{d}W$")
    for n, c in shows:
        In = np.concatenate([[0], np.cumsum(simple(X[:, 0], n) * dW[:, 0])])
        ax.plot(t, In, color=c, lw=1.2, label=rf"$\int_0^t X^n\,\mathrm{{d}}W$, $n={n}$")
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$t$")
    ax.set_title("their integrals converge to a limit", fontsize=10.5)
    ax.legend(fontsize=8.3, loc="best", framealpha=0.92)
    _grid(ax)
    _panel(ax, "b")

    ax = axs[2]
    ns = 2 ** np.arange(1, 10)
    e_int, e_out = [], []
    for n in ns:
        Xn = simple(X, n)
        e_int.append(np.mean(np.sum((X[:-1] - Xn) ** 2, axis=0) / N))
        e_out.append(np.mean((I_ref[-1] - np.sum(Xn * dW, axis=0)) ** 2))
    ax.loglog(ns, e_int, "o-", color=C_BLUE, lw=2.6, ms=6,
              label=r"$\mathbb{E}\int_0^1 (X - X^n)^2\,\mathrm{d}s$  (integrands)")
    ax.loglog(ns, e_out, "s--", color=C_ORANGE, lw=1.4, ms=4,
              label=r"$\mathbb{E}\left[(I(X) - I(X^n))^2\right]$  (integrals)")
    ax.loglog(ns, 2.5 / ns, color=C_GREY, ls=":", lw=1.1)
    ax.text(25, 2.5 / 40 * 0.45, "slope $-1$", color=C_GREY, fontsize=9)
    ax.set_xlabel(r"number $n$ of steps of $X^n$")
    ax.set_title("the isometry carries the error over exactly", fontsize=10.5)
    ax.legend(fontsize=8.3, loc="lower left")
    _grid(ax)
    _panel(ax, "c")
    fig.tight_layout()
    _save(fig, "density_extension.png")


# ----------------------------------------------------------------------
# Figure 7: localisation
# ----------------------------------------------------------------------
def fig_localisation():
    rng = np.random.default_rng(2)
    N = 20000
    while True:
        t, W = bm(rng, N)
        if 1.65 < np.abs(W).max() < 1.95 and np.abs(W[: N // 2]).max() < 1.2:
            break
    dt = 1 / N
    X = np.exp(W**2)
    A = np.concatenate([[0], np.cumsum(X[:-1] ** 2 * dt)])
    I = np.concatenate([[0], np.cumsum(X[:-1] * np.diff(W))])

    fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.3))
    ax = axs[0]
    s = np.linspace(0, 0.2499, 400)
    ax.semilogy(s, 1 / np.sqrt(1 - 4 * s), color=C_RED, lw=2.4,
                label=r"$\mathbb{E}[X_s^2] = (1-4s)^{-1/2}$")
    ax.axvspan(0.25, 1, color=tint(C_RED, 0.88), lw=0)
    ax.text(0.5, 3e3, r"$\mathbb{E}[X_s^2] = \infty$" "\n" r"$\Rightarrow X \notin \mathcal{L}^{\ast}$",
            color=C_RED, fontsize=10, ha="center")
    rng2 = np.random.default_rng(9)
    _, Ws = bm(rng2, 2000, 12)
    ts = np.linspace(0, 1, 2001)
    ax.semilogy(ts, np.exp(2 * Ws**2), color=C_BLUE, lw=0.7, alpha=0.7)
    ax.semilogy(t, X**2, color=C_BLUE, lw=1.2, label=r"paths $X_s^2$: all finite")
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.06, 1.0), framealpha=0.92)
    ax.set_ylim(0.8, 1e5)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$s$")
    ax.set_title(r"integrand $X_s = e^{W_s^2}$", fontsize=10.5)
    _grid(ax)
    _panel(ax, "a")

    levels, R = [], []
    for L in (1, 2, 4, 8, 16, 32, 64, 128):
        if L >= A[-1]:
            break
        r = t[np.argmax(A >= L)]
        if R and r - R[-1] < 0.05:      # keep the R_n visibly apart
            levels[-1], R[-1] = L, r
        else:
            levels.append(L)
            R.append(r)
    levels, R = levels[-4:], R[-4:]
    cols = [tint(C_GREEN, 0.55), tint(C_GREEN, 0.3), C_GREEN, "#11603a"][-len(levels):]
    ax = axs[1]
    ax.plot(t, A, color=C_INK, lw=1.6)
    for L, r, c in zip(levels, R, cols):
        ax.axhline(L, color=c, ls="--", lw=1)
        ax.plot([r, r], [0, L], color=c, lw=1)
        ax.plot(r, L, "o", color=c, ms=6)
        ax.text(0.02, L * 1.03, rf"$n = {L}$", color=c, fontsize=9, va="bottom")
        ax.text(r, -0.06 * A[-1], rf"$R_{{{L}}}$", color=c, fontsize=9, ha="center", va="top")
    ax.set_ylim(0, 1.08 * A[-1])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_xlabel(r"$t$", labelpad=12)
    ax.set_title(r"accumulated variance $\int_0^t X_s^2\,\mathrm{d}s$ and the times $R_n$",
                 fontsize=10.5)
    _grid(ax)
    _panel(ax, "b")

    ax = axs[2]
    ax.plot(t, I, color=C_INK, lw=2.8, alpha=0.25, label=r"$\int_0^t X\,\mathrm{d}W$ (Def. 4.12)")
    for L, r, c in zip(levels, R, cols):
        k = int(round(r * N))
        In = np.concatenate([I[: k + 1], np.full(N - k, I[k])])
        ax.plot(t, In, color=c, lw=1.2, label=rf"$\int_0^t X^{{({L})}}\,\mathrm{{d}}W$, stopped at $R_{{{L}}}$")
        ax.plot(r, I[k], "o", color=c, ms=6)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$t$")
    ax.set_title("localised integrals agree until they freeze", fontsize=10.5)
    ax.legend(fontsize=8.3, loc="upper left", bbox_to_anchor=(0.06, 1.0), framealpha=0.92)
    _grid(ax)
    _panel(ax, "c")
    fig.tight_layout()
    _save(fig, "localisation.png")


# ----------------------------------------------------------------------
# Figure 8: semimartingale decomposition
# ----------------------------------------------------------------------
def fig_semimartingale_decomposition():
    rng = np.random.default_rng(12)
    T, N = 6.0, 2**14
    dt = T / N
    t = np.linspace(0, T, N + 1)
    b = lambda x: x - x**3
    sg = lambda x: 0.25 + 0.25 * x**2
    dW = np.sqrt(dt) * rng.standard_normal(N)
    X = np.empty(N + 1)
    X[0] = 0.2
    dB = np.empty(N)
    dM = np.empty(N)
    for k in range(N):
        dB[k] = b(X[k]) * dt
        dM[k] = sg(X[k]) * dW[k]
        X[k + 1] = X[k] + dB[k] + dM[k]
    B = np.concatenate([[0], np.cumsum(dB)])
    M = np.concatenate([[0], np.cumsum(dM)])
    QV = np.concatenate([[0], np.cumsum(sg(X[:-1]) ** 2 * dt)])

    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.3),
                            gridspec_kw={"width_ratios": [1, 1, 1]})
    ax = axs[0]
    ax.plot(t, X, color=C_INK, lw=0.9)
    ax.axhline(1, color=C_GREY, ls=":", lw=0.8)
    ax.axhline(-1, color=C_GREY, ls=":", lw=0.8)
    ax.set_title(r"$\mathrm{d}X = (X - X^3)\,\mathrm{d}t + \sigma(X)\,\mathrm{d}W$",
                 fontsize=10.5)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$X_t$")
    _grid(ax)
    _panel(ax, "a")

    ax = axs[1]
    ax.plot(t, M, color=C_BLUE, lw=0.8, label=r"$M_t = \int_0^t \sigma(X_s)\,\mathrm{d}W_s$ (rough)")
    ax.plot(t, B, color=C_ORANGE, lw=2.2, label=r"$B_t = \int_0^t b(X_s)\,\mathrm{d}s$ (smooth)")
    ax.plot(t, B + M, color=C_INK, lw=0.6, ls=":", label=r"$B_t + M_t = X_t - X_0$")
    ax.axhline(0, color=C_INK, lw=0.5)
    ax.set_xlabel(r"$t$")
    ax.set_title("the two parts pull against each other", fontsize=10.5)
    ax.legend(fontsize=8.3, loc="lower left", framealpha=0.92)
    _grid(ax)
    _panel(ax, "b")

    ax = axs[2]
    s = 2
    for path, c, nm, lw in ((X, C_INK, r"$X$", 2.8), (M, C_BLUE, r"$M$", 1.3),
                            (B, C_ORANGE, r"$B$", 1.8)):
        q = np.concatenate([[0], np.cumsum(np.diff(path[::s]) ** 2)])
        ax.plot(t[::s], q, color=c, lw=lw, label=rf"$\sum (\Delta {nm[1:-1]})^2$",
                alpha=0.45 if path is X else 1)
    ax.plot(t, QV, color=C_GREEN, ls="--", lw=1.4,
            label=r"$\langle M\rangle_t = \int_0^t \sigma(X_s)^2\,\mathrm{d}s$")
    tvB = np.sum(np.abs(dB))
    tvM = np.sum(np.abs(np.diff(M[::s])))
    ax.text(0.03, 0.62, f"total variation on [0, 6]:\n  B: {tvB:.1f} (finite)\n"
            f"  M: {tvM:.0f} (grows with the grid)",
            transform=ax.transAxes, fontsize=8.5, color=C_GREY, va="top")
    ax.set_xlabel(r"$t$")
    ax.set_title(r"quadratic variation sees only $M$: $\langle X\rangle = \langle M\rangle$",
                 fontsize=10.5)
    ax.legend(fontsize=8.3, loc="upper left", bbox_to_anchor=(0.06, 1.0), framealpha=0.92)
    _grid(ax)
    _panel(ax, "c")
    fig.tight_layout()
    _save(fig, "semimartingale_decomposition.png")


# ----------------------------------------------------------------------
# Figure 9: the construction ladder (schematic)
# ----------------------------------------------------------------------
def fig_construction_ladder():
    fig, ax = plt.subplots(figsize=(13.4, 5.2))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    w, h, dx, dy = 3.15, 1.55, 3.35, 0.95
    rungs = [
        (C_BLUE, "1   simple  $\\mathcal{L}_0$",
         "integrator $M \\in \\mathcal{M}_2^C$",
         "explicit sums $\\sum_i \\xi_i\\,\\Delta M_i$  (Def. 4.2)"),
        (C_GREEN, "2   $\\mathcal{L}^{\\ast}$",
         "$\\mathbb{E}\\int X^2\\,\\mathrm{d}\\langle M\\rangle < \\infty$",
         "$L^2$-limit of simple integrals  (Def. 4.10)"),
        (C_ORANGE, "3   $\\mathcal{P}^{\\ast}$,  local martingales",
         "$\\int X^2\\,\\mathrm{d}\\langle M\\rangle < \\infty$  a.s.",
         "glue stopped integrals  (Def. 4.12)"),
        (C_PURPLE, "4   semimartingales",
         "$X = X_0 + M + B$",
         "$\\int Y\\,\\mathrm{d}M + \\int Y\\,\\mathrm{d}B$  (Def. 4.14)"),
    ]
    tools = ["isometry 4.6 + density 4.7\n+ completeness 4.8",
             "localise with stopping\ntimes (Lemma 4.11)",
             "BV part: path-wise\nStieltjes integral"]
    for k, (c, head, cond, how) in enumerate(rungs):
        x, y = 0.2 + k * dx, 0.3 + k * dy
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=tint(c, 0.88), ec=c, lw=1.6))
        ax.text(x + 0.15, y + h - 0.2, head, fontsize=10.5, color=c, va="top",
                fontweight="bold")
        ax.text(x + 0.15, y + h / 2 - 0.05, cond, fontsize=10, color=C_INK, va="center")
        ax.text(x + 0.15, y + 0.14, how, fontsize=8.6, color=C_GREY, va="bottom")
        if k < 3:
            ax.annotate("", xy=(x + dx - 0.04, y + dy + 1.05), xytext=(x + 2.3, y + h + 0.04),
                        arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=1.4,
                                        connectionstyle="arc3,rad=-0.35"))
            ax.text(x + 1.45, y + h + 0.2, tools[k], fontsize=8.8, color=C_INK,
                    ha="center", va="bottom", style="italic")
    ax.text(0.3, 4.85, "each rung: more integrands and integrators, fewer explicit formulas",
            fontsize=10, color=C_GREY, va="top")
    ax.text(11.6, 1.15, "fuel gauge on every rung:\n"
            r"$\int X^2\,\mathrm{d}\langle M\rangle$ decides admissibility"
            "\nand the variance of the integral",
            fontsize=9.5, color=C_RED, ha="center", va="center",
            bbox=dict(fc="white", ec=C_RED, lw=0.8, boxstyle="round,pad=0.35"))
    _save(fig, "construction_ladder.png")


# ----------------------------------------------------------------------
# Figure 10: Ito's rule
# ----------------------------------------------------------------------
def fig_ito_rule_taylor():
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.4),
                            gridspec_kw={"width_ratios": [1, 1.1, 1.05]})
    ax = axs[0]
    x = np.linspace(-1.1, 1.1, 300)
    ax.plot(x, np.exp(x), color=C_INK, lw=1.8, label=r"$f(x) = e^x$")
    ax.plot(x, 1 + x, color=C_ORANGE, lw=1.4, ls="--", label="tangent (first order)")
    hh = 0.6
    for s in (-hh, hh):
        ax.plot([s, s], [1 + s, np.exp(s)], color=C_RED, lw=2)
        ax.plot(s, np.exp(s), "o", color=C_INK, ms=5)
        ax.plot(s, 1 + s, "o", mfc="white", mec=C_ORANGE, ms=5)
    avg = np.cosh(hh)
    ax.plot([-hh, hh], [np.exp(-hh), np.exp(hh)], color=C_GREY, lw=0.8, ls=":")
    ax.plot(0, avg, "D", color=C_RED, ms=6)
    ax.plot(0, 1, "o", color=C_ORANGE, ms=5)
    ax.annotate(rf"average of $f(\pm h)$: {avg:.3f}" "\n"
                rf"$\approx f(0) + \frac{{1}}{{2}}f''(0)h^2 = {1 + hh**2 / 2:.3f}$",
                (0, avg), xytext=(-1.08, 2.2), fontsize=8.8, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.8))
    ax.text(0.15, 0.3, r"steps $\Delta W = \pm h$:" "\n" r"first-order terms cancel,"
            "\n" r"red gaps $\frac{1}{2}f''(\Delta W)^2 > 0$ do not",
            fontsize=8.8, color=C_INK)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(0, 3.05)
    ax.set_xlabel(r"$x$")
    ax.set_title("one step: convexity bias of size $\\frac{1}{2}f''\\,\\Delta t$",
                 fontsize=10.5)
    ax.legend(fontsize=8.3, loc="upper left", bbox_to_anchor=(0.06, 1.0))
    _grid(ax)
    _panel(ax, "a")

    rng = np.random.default_rng(19)
    N = 2**14
    t, W = bm(rng, N, T=2.0)
    dt = 2.0 / N
    eW = np.exp(W)
    stoch = 1 + np.concatenate([[0], np.cumsum(eW[:-1] * np.diff(W))])
    corr = np.concatenate([[0], np.cumsum(0.5 * eW[:-1] * dt)])
    ax = axs[1]
    ax.plot(t, eW, color=C_INK, lw=2.6, alpha=0.35, label=r"$f(W_t) = e^{W_t}$")
    ax.plot(t, stoch, color=C_ORANGE, lw=1.2,
            label=r"chain-rule guess $1 + \int_0^t e^{W_s}\,\mathrm{d}W_s$")
    ax.plot(t, corr, color=C_GREEN, lw=1.6,
            label=r"Itô correction $\frac{1}{2}\int_0^t e^{W_s}\,\mathrm{d}s$")
    ax.plot(t, stoch + corr, color=C_INK, lw=1, ls="--", label="guess + correction")
    ax.set_xlim(0, 2)
    ax.set_xlabel(r"$t$")
    ax.set_title(r"Theorem 4.15 for $f = \exp$ along one path", fontsize=10.5)
    ax.legend(fontsize=8.2, loc="lower left", framealpha=0.92)
    _grid(ax)
    _panel(ax, "b")

    ax = axs[2]
    N2 = 2**16
    _, W2 = bm(rng, N2)
    ns = 2 ** np.arange(2, 17)
    rows = []
    for n in ns:
        d = np.diff(W2[:: N2 // n])
        rows.append((n * (1 / n) ** 2, np.sum(np.abs(d)) / n, np.sum(d**2), np.sum(np.abs(d) ** 3)))
    rows = np.array(rows)
    specs = [(r"$\sum (\Delta t)^2$", C_GREY), (r"$\sum |\Delta t\,\Delta W|$", C_ORANGE),
             (r"$\sum (\Delta W)^2$", C_BLUE), (r"$\sum |\Delta W|^3$", C_GREEN)]
    for j, (nm, c) in enumerate(specs):
        ax.loglog(ns, rows[:, j], "o-", ms=3, color=c, lw=1.6, label=nm)
    ax.set_xlabel(r"number $n$ of steps on $[0,1]$")
    ax.set_title("which second-order sums survive?", fontsize=10.5)
    ax.legend(fontsize=8.3, loc="lower left")
    ax.text(0.97, 0.97,
            r"$\mathrm{d}t\cdot\mathrm{d}t = 0$" "\n"
            r"$\mathrm{d}t\cdot\mathrm{d}W = 0$" "\n"
            r"$\mathrm{d}W\cdot\mathrm{d}W = \mathrm{d}t$",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color=C_INK,
            bbox=dict(fc="white", ec=C_BLUE, lw=1, boxstyle="round,pad=0.4"))
    _grid(ax)
    _panel(ax, "c")
    fig.tight_layout()
    _save(fig, "ito_rule_taylor.png")


# ----------------------------------------------------------------------
# Figure 11: quadratic covariation and polarisation
# ----------------------------------------------------------------------
def _corr_bm(rng, N, rho, T=1.0):
    t, W1 = bm(rng, N, T=T)
    _, Wp = bm(rng, N, T=T)
    return t, W1, rho * W1 + np.sqrt(1 - rho**2) * Wp


def fig_quadratic_covariation():
    rng = np.random.default_rng(4)
    fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.4),
                            gridspec_kw={"width_ratios": [0.95, 1, 1]})
    rho = 0.7
    ax = axs[0]
    z1 = rng.standard_normal(1500)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(1500)
    ax.scatter(z1, z2, s=4, color=C_BLUE, alpha=0.35, lw=0)
    for nstd in (1, 2):
        ax.add_patch(Ellipse((0, 0), 2 * nstd * np.sqrt(1 + rho), 2 * nstd * np.sqrt(1 - rho),
                             angle=45, fill=False, ec=C_BLUE, lw=1.2))
    L = 3.3
    ax.annotate("", xy=(2.3, 2.3), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=2))
    ax.annotate("", xy=(1.1, -1.1), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_PURPLE, lw=2))
    ax.text(1.3, 2.75, r"sum: variance $2(1+\rho)$", color=C_GREEN, fontsize=9)
    ax.text(0.9, -1.75, r"difference: $2(1-\rho)$", color=C_PURPLE, fontsize=9)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta W^1/\sqrt{\Delta t}$")
    ax.set_ylabel(r"$\Delta W^2/\sqrt{\Delta t}$")
    ax.set_title(rf"increments of a correlated pair, $\rho = {rho}$", fontsize=10.5)
    _grid(ax)
    _panel(ax, "a")

    N = 2**12
    ax = axs[1]
    for r, c in ((0.8, C_GREEN), (0.0, C_GREY), (-0.6, C_RED)):
        t, W1, W2 = _corr_bm(rng, N, r)
        cs = np.concatenate([[0], np.cumsum(np.diff(W1) * np.diff(W2))])
        ax.plot(t, cs, color=c, lw=1.4, label=rf"$\rho = {r}$")
        ax.plot([0, 1], [0, r], color=c, ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$t$")
    ax.set_title(r"$\sum \Delta W^1 \Delta W^2 \to \langle W^1, W^2\rangle_t = \rho t$",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.06, 1.0))
    _grid(ax)
    _panel(ax, "b")

    ax = axs[2]
    t, W1, W2 = _corr_bm(rng, N, rho)
    qs = np.concatenate([[0], np.cumsum(np.diff(W1 + W2) ** 2)]) / 4
    qd = np.concatenate([[0], np.cumsum(np.diff(W1 - W2) ** 2)]) / 4
    ax.plot(t, qs, color=C_GREEN, lw=1.5, label=r"$\frac{1}{4}\langle W^1 + W^2\rangle_t$")
    ax.plot(t, qd, color=C_PURPLE, lw=1.5, label=r"$\frac{1}{4}\langle W^1 - W^2\rangle_t$")
    ax.plot(t, qs - qd, color=C_BLUE, lw=2, label="difference")
    ax.plot([0, 1], [0, rho], color=C_BLUE, ls="--", lw=1)
    ax.text(0.5, rho * 0.5 - 0.09, rf"$\rho t$, $\rho = {rho}$", color=C_BLUE, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$t$")
    ax.set_title("polarisation: covariation from two variations", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.06, 1.0))
    _grid(ax)
    _panel(ax, "c")
    fig.tight_layout()
    _save(fig, "quadratic_covariation.png")


# ----------------------------------------------------------------------
# Figure 12: partial integration as area bookkeeping
# ----------------------------------------------------------------------
def fig_partial_integration():
    fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.4),
                            gridspec_kw={"width_ratios": [0.95, 1.05, 1]})
    ax = axs[0]
    X, Y, dX, dY = 2.0, 1.4, 0.55, 0.45
    ax.add_patch(Rectangle((0, 0), X, Y, fc=tint(C_GREY, 0.8), ec=C_GREY, lw=1))
    ax.add_patch(Rectangle((X, 0), dX, Y, fc=tint(C_BLUE, 0.6), ec=C_BLUE, lw=1))
    ax.add_patch(Rectangle((0, Y), X, dY, fc=tint(C_ORANGE, 0.55), ec=C_ORANGE, lw=1))
    ax.add_patch(Rectangle((X, Y), dX, dY, fc=tint(C_RED, 0.35), ec=C_RED, lw=1.4))
    ax.text(X / 2, Y / 2, r"$X_t Y_t$", ha="center", va="center", fontsize=12, color=C_INK)
    ax.text(X + dX / 2, Y / 2, r"$Y\,\Delta X$", ha="center", va="center", fontsize=10,
            color=C_BLUE, rotation=90)
    ax.text(X / 2, Y + dY / 2, r"$X\,\Delta Y$", ha="center", va="center", fontsize=10,
            color=C_ORANGE)
    ax.text(X + dX / 2, Y + dY / 2, r"$\Delta X\Delta Y$", ha="center", va="center",
            fontsize=8.5, color=C_RED)
    ax.text(0.02, -0.33, "smooth paths: corner $= O(\\Delta t^2)$, negligible\n"
            "Brownian paths: corner $\\sim \\Delta t$, piles up to $\\langle X, Y\\rangle$",
            fontsize=8.8, color=C_INK, va="top")
    ax.set_xlim(-0.1, 2.75)
    ax.set_ylim(-0.95, 2.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"one step of $X_{t+\Delta t}Y_{t+\Delta t} - X_tY_t$", fontsize=10.5)
    _panel(ax, "a")

    rng = np.random.default_rng(14)
    rho, n = 0.6, 2**10
    t, W1, W2 = _corr_bm(rng, n, rho)
    Xp, Yp = 1 + W1, 1 + W2
    dXp, dYp = np.diff(Xp), np.diff(Yp)
    strips = 1 + np.concatenate([[0], np.cumsum(Yp[:-1] * dXp + Xp[:-1] * dYp)])
    corners = np.concatenate([[0], np.cumsum(dXp * dYp)])
    ax = axs[1]
    ax.plot(t, Xp * Yp, color=C_INK, lw=2.8, alpha=0.3, label=r"$X_tY_t$")
    ax.plot(t, strips, color=C_ORANGE, lw=1.2,
            label=r"$1 + \int_0^t Y\,\mathrm{d}X + \int_0^t X\,\mathrm{d}Y$")
    ax.plot(t, strips + corners, color=C_INK, lw=1, ls="--", label="... + corners")
    ax.plot(t, corners, color=C_RED, lw=1.5, label=r"corners $\sum \Delta X\Delta Y$")
    ax.plot([0, 1], [0, rho], color=C_RED, ls=":", lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$t$")
    ax.set_title(rf"$X = 1 + W^1$, $Y = 1 + W^2$, $\rho = {rho}$", fontsize=10.5)
    ax.legend(fontsize=8.3, loc="upper left", bbox_to_anchor=(0.06, 1.0), framealpha=0.92)
    _grid(ax)
    _panel(ax, "b")

    ax = axs[2]
    N = 2**16
    tt = np.linspace(0, 1, N + 1)
    xs, ys = 1 + 0.6 * np.sin(np.pi * tt), 1 + tt**2
    _, B1, B2 = _corr_bm(rng, N, rho)
    ns = 2 ** np.arange(2, 17)
    cs_s = [np.sum(np.diff(xs[:: N // k]) * np.diff(ys[:: N // k])) for k in ns]
    cs_b = [np.sum(np.diff(B1[:: N // k]) * np.diff(B2[:: N // k])) for k in ns]
    ax.loglog(ns, np.abs(cs_s), "o-", ms=3, color=C_GREEN, lw=1.6, label="smooth pair")
    ax.loglog(ns, np.abs(cs_b), "o-", ms=3, color=C_RED, lw=1.6, label="Brownian pair")
    ax.axhline(rho, color=C_RED, ls=":", lw=1)
    ax.text(12, rho * 2.6, rf"$\to \langle X, Y\rangle_1 = {rho}$", color=C_RED, fontsize=9)
    ax.set_ylim(1e-6, 5)
    ax.text(300, 2e-4, r"$\sim 1/n \to 0$", color=C_GREEN, fontsize=9)
    ax.set_xlabel(r"number $n$ of steps on $[0,1]$")
    ax.set_title(r"total corner area $|\sum \Delta X\Delta Y|$", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower left")
    _grid(ax)
    _panel(ax, "c")
    fig.tight_layout()
    _save(fig, "partial_integration_rectangle.png")


# ----------------------------------------------------------------------
# Figure 13: geometric Brownian motion: mean vs median
# ----------------------------------------------------------------------
def fig_gbm_mean_vs_median():
    rng = np.random.default_rng(6)
    mu, sigma, T, N = 0.2, 1.0, 8.0, 800
    t = np.linspace(0, T, N + 1)
    _, Wp = bm(rng, N, 150, T=T)
    Y = np.exp((mu - sigma**2 / 2) * t[:, None] + sigma * Wp)
    gr = mu - sigma**2 / 2

    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.4),
                            gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axs[0]
    lo = np.exp(gr * t - 1.645 * sigma * np.sqrt(t))
    hi = np.exp(gr * t + 1.645 * sigma * np.sqrt(t))
    ax.fill_between(t, lo, hi, color=tint(C_BLUE, 0.85), lw=0, label="5–95 % of paths")
    ax.semilogy(t, Y, color=C_GREY, lw=0.4, alpha=0.4)
    ax.semilogy(t, np.exp(mu * t), color=C_RED, lw=2.2, label=r"mean $e^{\mu t}$")
    ax.semilogy(t, np.exp(gr * t), color=C_BLUE, lw=2.2,
                label=r"median $e^{(\mu - \sigma^2/2) t}$")
    ax.axhline(1, color=C_INK, lw=0.6, ls=":")
    ax.set_ylim(1e-5, 1e3)
    ax.set_xlim(0, T)
    ax.set_xlabel(r"$t$")
    ax.set_title(rf"$\mathrm{{d}}Y = Y(\mu\,\mathrm{{d}}t + \sigma\,\mathrm{{d}}W)$, "
                 rf"$\mu = {mu}$, $\sigma = {sigma:g}$", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.92)
    _grid(ax)
    _panel(ax, "a")

    ax = axs[1]
    m, v = gr * T, sigma**2 * T
    z = np.linspace(m - 4 * np.sqrt(v), m + 4 * np.sqrt(v), 400)
    pdf = gauss(z, m, v)
    ax.plot(z, pdf, color=C_INK, lw=1.6)
    zl = z[z < 0]
    ax.fill_between(zl, gauss(zl, m, v), color=tint(C_RED, 0.6), lw=0)
    p_loss = Phi(-m / np.sqrt(v))
    ax.text(m - 3.3, 0.55 * pdf.max(), f"{100 * p_loss:.0f} % of paths\nend below $Y_0 = 1$",
            color=C_RED, fontsize=9, ha="center",
            bbox=dict(fc="white", ec="none", alpha=0.85))
    for val, c, nm in ((m, C_BLUE, "median"), (mu * T, C_RED, "mean")):
        ax.axvline(val, color=c, lw=1.8)
        ax.text(val + 0.15, 1.02 * pdf.max(), nm, color=c, fontsize=9)
    ax.axvline(0, color=C_INK, lw=0.6, ls=":")
    ax.set_ylim(0, 1.12 * pdf.max())
    ax.set_xlabel(r"$\log Y_T$  at  $T = 8$")
    ax.set_title(r"log-normal: $\log Y_T \sim \mathcal{N}((\mu - \sigma^2/2)T,\ \sigma^2 T)$",
                 fontsize=10.5)
    _grid(ax)
    _panel(ax, "b")

    ax = axs[2]
    n_max = 200000
    for j, c in enumerate((C_BLUE, C_GREEN, C_ORANGE)):
        YT = np.exp(gr * T + sigma * np.sqrt(T) * rng.standard_normal(n_max))
        ax.semilogx(np.arange(1, n_max + 1), np.cumsum(YT) / np.arange(1, n_max + 1),
                    color=c, lw=1.1, label=f"run {j + 1}")
    ax.axhline(np.exp(mu * T), color=C_RED, ls="--", lw=1.4)
    ax.text(1.2, np.exp(mu * T) * 1.1, rf"$\mathbb{{E}}[Y_T] = e^{{{mu * T:g}}} \approx "
            rf"{np.exp(mu * T):.2f}$", color=C_RED, fontsize=9,
            bbox=dict(fc="white", ec="none", alpha=0.85))
    ax.set_ylim(0, 2.2 * np.exp(mu * T))
    ax.set_xlabel("number of simulated paths")
    ax.set_title("running sample mean: carried by rare\nhuge outcomes", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    _grid(ax)
    _panel(ax, "c")
    fig.tight_layout()
    _save(fig, "gbm_mean_vs_median.png")


# ----------------------------------------------------------------------
# Figure 14: OU via variation of constants
# ----------------------------------------------------------------------
def fig_ou_variation_of_constants():
    rng = np.random.default_rng(17)
    sigma, x0, T, N = 0.8, 2.5, 5.0, 5000
    dt = T / N
    t = np.linspace(0, T, N + 1)
    dW = np.sqrt(dt) * rng.standard_normal(N)
    noise = np.zeros(N + 1)
    em = np.zeros(N + 1)
    em[0] = x0
    for k in range(N):
        noise[k + 1] = np.exp(-dt) * (noise[k] + sigma * dW[k])
        em[k + 1] = em[k] - em[k] * dt + sigma * dW[k]
    X = np.exp(-t) * x0 + noise

    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.4))
    ax = axs[0]
    ax.plot(t, em, color=C_GREY, lw=2.6, alpha=0.35, label="Euler–Maruyama, same noise")
    ax.plot(t, X, color=C_INK, lw=1, label=r"$X_t$ from the formula")
    ax.plot(t, np.exp(-t) * x0, color=C_ORANGE, lw=2, ls="--",
            label=r"$e^{-t}x$ (forgotten start)")
    ax.plot(t, noise, color=C_BLUE, lw=0.9,
            label=r"$\sigma\int_0^t e^{-(t-s)}\,\mathrm{d}W_s$")
    ax.axhline(0, color=C_INK, lw=0.5)
    ax.set_xlim(0, T)
    ax.set_xlabel(r"$t$")
    ax.set_title(r"$\mathrm{d}X = -X\,\mathrm{d}t + \sigma\,\mathrm{d}W$: deterministic + noise",
                 fontsize=10.5)
    ax.legend(fontsize=8.2, loc="upper center", bbox_to_anchor=(0.62, 1.0), framealpha=0.92)
    _grid(ax)
    _panel(ax, "a")

    ax = axs[1]
    ts = 4.0
    m = 80
    sk = np.linspace(0, ts, m + 1)[:-1]
    kicks = sigma * np.sqrt(ts / m) * rng.standard_normal(m)
    wts = np.exp(-(ts - sk))
    ax.vlines(sk, 0, kicks, color=tint(C_GREY, 0.45), lw=2.4, label=r"kicks $\sigma\,\Delta W_s$")
    ax.vlines(sk + 0.012, 0, wts * kicks, color=C_BLUE, lw=2.4,
              label=r"weighted $e^{-(t-s)}\sigma\,\Delta W_s$")
    ss = np.linspace(0, ts, 300)
    for sgn in (1, -1):
        ax.plot(ss, sgn * 0.5 * np.exp(-(ts - ss)), color=C_ORANGE, lw=1.4,
                label=r"memory kernel $e^{-(t-s)}$ (scaled)" if sgn > 0 else None)
    ax.axhline(0, color=C_INK, lw=0.5)
    ax.set_xlim(0, ts + 0.1)
    ax.set_xlabel(r"time $s$ of the kick (observed at $t = 4$)")
    ax.set_title("fading memory: old kicks are forgotten", fontsize=10.5)
    ax.legend(fontsize=8.2, loc="upper left", bbox_to_anchor=(0.06, 1.0), framealpha=0.92)
    _grid(ax)
    _panel(ax, "b")

    ax = axs[2]
    n_mc, dt2 = 20000, 0.01
    Xs = np.full(n_mc, x0)
    snaps = {25: None, 100: None, 400: None}
    for k in range(1, 401):
        Xs = Xs - Xs * dt2 + sigma * np.sqrt(dt2) * rng.standard_normal(n_mc)
        if k in snaps:
            snaps[k] = Xs.copy()
    y = np.linspace(-2.5, 3.6, 400)
    for (k, S), c in zip(snaps.items(), (C_ORANGE, C_GREEN, C_BLUE)):
        tk = k * dt2
        mk, vk = x0 * np.exp(-tk), sigma**2 * (1 - np.exp(-2 * tk)) / 2
        ax.hist(S, bins=70, density=True, color=tint(c, 0.55), lw=0)
        ax.plot(y, gauss(y, mk, vk), color=c, lw=1.8, label=rf"$t = {tk:g}$")
    ax.set_xlabel(r"$x$")
    ax.set_title("400 Euler steps (bars) vs. one exact\nGaussian draw (curves)",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.06, 1.0))
    _grid(ax)
    _panel(ax, "c")
    fig.tight_layout()
    _save(fig, "ou_variation_of_constants.png")


# ----------------------------------------------------------------------
# Figure 15: Euler-Maruyama mechanics and step-size explosion
# ----------------------------------------------------------------------
def fig_euler_maruyama_mechanics():
    rng = np.random.default_rng(21)
    fig, axs = plt.subplots(1, 2, figsize=(13.0, 4.6),
                            gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    b = lambda x: x - x**3
    sig, dt, K = 0.6, 0.35, 8
    xs = [-0.15]
    for k in range(K):
        x = xs[-1]
        m = x + b(x) * dt
        xn = m + sig * np.sqrt(dt) * rng.standard_normal()
        tk1 = (k + 1) * dt
        yy = np.linspace(m - 3.2 * sig * np.sqrt(dt), m + 3.2 * sig * np.sqrt(dt), 120)
        dens = gauss(yy, m, sig**2 * dt)
        width = 0.13 * dens / dens.max()
        ax.fill_betweenx(yy, tk1, tk1 + width, color=tint(C_BLUE, 0.7), lw=0)
        ax.plot(tk1 + width, yy, color=C_BLUE, lw=0.9)
        ax.annotate("", xy=(tk1, m), xytext=(k * dt, x),
                    arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=1.6))
        ax.plot([tk1, tk1], [m, xn], color=C_RED, lw=1.4)
        ax.plot([k * dt, tk1], [x, xn], color=C_INK, lw=0.8, ls=":")
        xs.append(xn)
    ax.plot(np.arange(K + 1) * dt, xs, "o", color=C_INK, ms=5, zorder=5)
    for yv in (-1, 1):
        ax.axhline(yv, color=C_GREY, ls=":", lw=0.8)
    ax.text(0.01, 0.03,
            r"orange: drift shift $b(x)\,\Delta t$;  blue: $\mathcal{N}(x + b\Delta t,\ a\Delta t)$;"
            "\n" r"red: the sampled kick $\sigma\sqrt{\Delta t}\,Z$",
            transform=ax.transAxes, fontsize=8.8, color=C_INK,
            bbox=dict(fc="white", ec="none", alpha=0.85))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    ax.set_title(r"Euler–Maruyama for $b = x - x^3$, $\sigma = 0.6$: shift, then shake",
                 fontsize=10.5)
    ax.set_xlim(-0.05, (K + 1) * dt + 0.1)
    ax.set_ylim(-1.75, 1.95)
    _grid(ax)
    _panel(ax, "a")

    ax = axs[1]
    T, x0, s2 = 3.0, 2.5, 0.5
    for dtt, c, nm in ((0.01, C_GREEN, r"$\Delta t = 0.01$"), (0.3, C_RED, r"$\Delta t = 0.3$")):
        n = int(T / dtt)
        tt = np.arange(n + 1) * dtt
        for j in range(5):
            x = np.empty(n + 1)
            x[0] = x0
            for k in range(n):
                x[k + 1] = x[k] - x[k] ** 3 * dtt + s2 * np.sqrt(dtt) * rng.standard_normal()
                if abs(x[k + 1]) > 1e12:
                    x[k + 2:] = np.nan
                    x[k + 1] = np.sign(x[k + 1]) * 1e12
                    break
            ax.plot(tt, x, color=c, lw=1.1, marker="o" if dtt > 0.1 else None, ms=3,
                    label=nm if j == 0 else None)
    thr = np.sqrt(2 / 0.3)
    ax.axhspan(-thr, thr, color=tint(C_GREEN, 0.9), lw=0)
    ax.annotate(r"band $|x| < \sqrt{2/\Delta t}$ for $\Delta t = 0.3$:" "\n"
                "outside it, one Euler step overshoots 0\nby more than it started",
                (2.2, 1.0), xytext=(1.75, 1e6), fontsize=8.8, color=C_GREEN,
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=0.9),
                bbox=dict(fc="white", ec="none", alpha=0.85))
    ax.set_yscale("symlog", linthresh=3)
    ax.set_ylim(-1e12, 1e12)
    ax.set_yticks([-1e12, -1e8, -1e4, -10, 0, 10, 1e4, 1e8, 1e12])
    ax.set_xlim(0, T)
    ax.set_xlabel(r"$t$")
    ax.set_title(r"$\mathrm{d}X = -X^3\,\mathrm{d}t + 0.5\,\mathrm{d}W$, $X_0 = 2.5$:"
                 "\ntoo coarse a step overshoots and explodes", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    _grid(ax)
    _panel(ax, "b")
    fig.tight_layout()
    _save(fig, "euler_maruyama_mechanics.png")


if __name__ == "__main__":
    fig_bm_variation_scaling()
    fig_simple_integral_gambling()
    fig_evaluation_point()
    fig_qv_clock_compensator()
    fig_ito_isometry()
    fig_density_extension()
    fig_localisation()
    fig_semimartingale_decomposition()
    fig_construction_ladder()
    fig_ito_rule_taylor()
    fig_quadratic_covariation()
    fig_partial_integration()
    fig_gbm_mean_vs_median()
    fig_ou_variation_of_constants()
    fig_euler_maruyama_mechanics()
