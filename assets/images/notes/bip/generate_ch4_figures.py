"""Generate the Chapter 4 (Bayesian Inversion) figures for the
numerical_methods_for_bip notes.

Run from anywhere:  python3 generate_ch4_figures.py
PNGs are written next to this script. All randomness is seeded.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import kv, gamma as gamma_fn
from scipy.stats import beta as beta_dist

OUT = os.path.dirname(os.path.abspath(__file__))

# Site palette
BLUE = "#1d4ed8"
GREEN = "#0f9b6c"
AMBER = "#a86f00"
RED = "#d65336"
GRAY = "#5b6270"
DARK = "#1f2430"

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "font.size": 11,
        "axes.edgecolor": DARK,
        "axes.labelcolor": DARK,
        "axes.titlesize": 11,
        "xtick.color": DARK,
        "ytick.color": DARK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 170,
        "savefig.bbox": "tight",
    }
)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------------------
# 1. Prior x likelihood -> posterior (Remark 4.2.3)
# ----------------------------------------------------------------------
def fig_bayes_update():
    x = np.linspace(-4, 5, 600)
    prior = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    y, sig = 2.0, 0.7
    lik = np.exp(-0.5 * ((y - x) / sig) ** 2)  # as a function of x, unnormalised
    post_un = lik * prior
    post = post_un / np.trapezoid(post_un, x)
    lik_scaled = lik / np.trapezoid(lik, x)  # scaled only for display

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(x, prior, color=BLUE, lw=2, label=r"prior $\pi_X(x)$")
    ax.fill_between(x, prior, color=BLUE, alpha=0.10)
    ax.plot(x, lik_scaled, color=AMBER, lw=2, ls="--",
            label=r"likelihood $\pi_{Y|X}(y\,|\,x)$ (scaled)")
    ax.plot(x, post, color=GREEN, lw=2.4, label=r"posterior $\pi_{X|Y}(x\,|\,y)$")
    ax.fill_between(x, post, color=GREEN, alpha=0.15)
    ax.axvline(y, color=GRAY, lw=1, ls=":")
    ax.annotate(r"data $y$", xy=(y, 0.56), xytext=(y + 0.65, 0.56),
                color=GRAY, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    ax.set_xlabel(r"$x$")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "bip_bayes_update_1d.png")


# ----------------------------------------------------------------------
# 2. Credible interval / posterior variance (Example 4.2.4)
# ----------------------------------------------------------------------
def fig_credible():
    x = np.linspace(-1.5, 5, 2000)

    def mix(x, w, m1, s1, m2, s2):
        g1 = np.exp(-0.5 * ((x - m1) / s1) ** 2) / (s1 * np.sqrt(2 * np.pi))
        g2 = np.exp(-0.5 * ((x - m2) / s2) ** 2) / (s2 * np.sqrt(2 * np.pi))
        return w * g1 + (1 - w) * g2

    # narrow posterior (informative data) and wide skewed posterior
    p_narrow = mix(x, 0.85, 1.0, 0.25, 1.6, 0.35)
    p_wide = mix(x, 0.6, 0.8, 0.45, 2.6, 0.85)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1), sharey=True)
    for ax, p, ttl in (
        (axes[0], p_narrow, "small posterior variance"),
        (axes[1], p_wide, "large posterior variance"),
    ):
        p = p / np.trapezoid(p, x)
        cdf = np.cumsum(p) * (x[1] - x[0])
        lo, hi = x[np.searchsorted(cdf, 0.05)], x[np.searchsorted(cdf, 0.95)]
        x_map = x[np.argmax(p)]
        x_cm = np.trapezoid(x * p, x)
        var = np.trapezoid((x - x_cm) ** 2 * p, x)
        band = (x >= lo) & (x <= hi)
        ax.plot(x, p, color=DARK, lw=1.8)
        ax.fill_between(x[band], p[band], color=BLUE, alpha=0.15,
                        label="90% credible interval")
        ax.axvline(x_map, color=BLUE, lw=1.6, label=r"$x_{\mathrm{MAP}}$")
        ax.axvline(x_cm, color=RED, lw=1.6, ls="--", label=r"$x_{\mathrm{CM}}$")
        ax.set_title(ttl + rf"  ($\mathbb{{V}}[X|Y=y] \approx {var:.2f}$)",
                     fontsize=10.5)
        ax.set_xlabel(r"$x$")
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9)
    save(fig, "bip_credible_interval.png")


# ----------------------------------------------------------------------
# 3. Additive noise model in 2D: prior, likelihood, posterior (Ex. 4.2.10)
# ----------------------------------------------------------------------
def fig_additive_2d():
    n = 400
    g = np.linspace(-2.5, 2.5, n)
    X1, X2 = np.meshgrid(g, g)
    alpha, sig, y = 1.0, 0.4, 2.0
    prior = np.exp(-0.5 * alpha * (X1**2 + X2**2))
    lik = np.exp(-0.5 * ((X1 + X2 - y) / sig) ** 2)
    post = prior * lik

    # MAP = Tikhonov solution of min ||Ax-y||^2 + alpha sigma^2 ||x||^2, A=[1 1]
    a_eff = alpha * sig**2
    xm = y / (2 + a_eff)  # both coordinates equal by symmetry

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, Z, ttl, cmap in (
        (axes[0], prior, r"prior $\pi_X(x) \propto e^{-\alpha\|x\|^2/2}$", "Blues"),
        (axes[1], lik, r"likelihood $\pi_E(y - Ax)$", "YlOrBr"),
        (axes[2], post, r"posterior $\propto$ likelihood $\cdot$ prior", "Greens"),
    ):
        ax.contourf(X1, X2, Z, levels=12, cmap=cmap)
        ax.contour(X1, X2, Z, levels=6, colors=DARK, linewidths=0.4, alpha=0.5)
        ax.plot(g, y - g, color=GRAY, lw=1.0, ls=":")
        ax.set_title(ttl, fontsize=10.5)
        ax.set_xlabel(r"$x_1$")
        ax.set_xlim(g[0], g[-1])
        ax.set_ylim(g[0], g[-1])
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"$x_2$")
    axes[1].annotate(r"$\{x : Ax = y\}$", xy=(1.6, 0.4), xytext=(0.3, -2.0),
                     fontsize=10, color=DARK,
                     arrowprops=dict(arrowstyle="->", color=DARK, lw=0.8))
    axes[2].plot(xm, xm, "o", ms=6, color=RED, zorder=5)
    axes[2].annotate(r"$x_{\mathrm{MAP}}$ (= Tikhonov)", xy=(xm, xm),
                     xytext=(-2.1, 1.8), fontsize=10, color=RED,
                     arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    save(fig, "bip_additive_noise_2d.png")


# ----------------------------------------------------------------------
# 4. Posterior concentration as noise -> 0 (Section 4.2.3)
# ----------------------------------------------------------------------
def fig_noise_concentration():
    x = np.linspace(-3, 3.5, 800)
    y = 1.5
    prior = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    sigmas = [1.5, 0.8, 0.4, 0.15]
    colors = [BLUE, GREEN, AMBER, RED]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(x, prior, color=GRAY, lw=1.8, ls="--", label=r"prior $\mathcal{N}(0,1)$")
    for sig, col in zip(sigmas, colors):
        m = y / (1 + sig**2)
        v = sig**2 / (1 + sig**2)
        p = np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2 * np.pi * v)
        ax.plot(x, p, color=col, lw=1.9, label=rf"$\sigma = {sig}$")
    ax.axvline(y, color=DARK, lw=1.0, ls=":")
    ax.text(y + 0.07, 2.45, r"data $y$", color=DARK, fontsize=10)
    ax.set_xlabel(r"$x$")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9.5)
    save(fig, "bip_noise_concentration.png")


# ----------------------------------------------------------------------
# 5. Hellinger stability w.r.t. the data (Theorem 4.4.1)
# ----------------------------------------------------------------------
def fig_hellinger():
    x = np.linspace(-4, 4, 1500)
    prior = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    sig = 0.6
    phi = x + 0.4 * np.sin(2.0 * x)  # nonlinear forward map

    def posterior(y):
        p = np.exp(-0.5 * ((y - phi) / sig) ** 2) * prior
        return p / np.trapezoid(p, x)

    def hellinger(p, q):
        return np.sqrt(0.5 * np.trapezoid((np.sqrt(p) - np.sqrt(q)) ** 2, x))

    y0 = 0.8
    p0 = posterior(y0)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    ax = axes[0]
    for y, col, ls in ((y0, DARK, "-"), (1.1, BLUE, "--"), (1.6, AMBER, "-.")):
        lbl = rf"$y' = {y}$" if y != y0 else rf"$y = {y0}$"
        ax.plot(x, posterior(y), color=col, lw=1.9, ls=ls, label=lbl)
    ax.set_xlabel(r"$x$")
    ax.set_xlim(-2.5, 3.5)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9.5)
    ax.set_title("posteriors for perturbed data", fontsize=10.5)

    ax = axes[1]
    dy = np.linspace(0, 1.6, 60)
    dh = np.array([hellinger(p0, posterior(y0 + d)) for d in dy])
    slope = dh[1] / dy[1]
    ax.plot(dy, dh, color=GREEN, lw=2.0,
            label=r"$D_{\mathrm{H}}(\mu_{X|y},\, \mu_{X|y'})$")
    ax.plot(dy, slope * dy, color=GRAY, lw=1.2, ls=":",
            label=r"linear bound $C\,\|y - y'\|$")
    ax.set_xlabel(r"$\|y - y'\|$")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9.5)
    ax.set_title("Hellinger distance vs. data perturbation", fontsize=10.5)
    save(fig, "bip_hellinger_stability.png")


# ----------------------------------------------------------------------
# 6 & 7. Matern covariance functions and sample paths (PDF Figs 4.3 / 4.4)
# ----------------------------------------------------------------------
def matern(r, nu, sigma2=1.0, ell=1.0):
    r = np.asarray(r, dtype=float)
    if np.isinf(nu):
        return sigma2 * np.exp(-(r**2) / ell**2)
    out = np.full_like(r, sigma2)
    pos = r > 0
    z = 2.0 * np.sqrt(nu) * r[pos] / ell
    out[pos] = sigma2 / (2 ** (nu - 1) * gamma_fn(nu)) * z**nu * kv(nu, z)
    return out


NUS = [0.5, 1.5, 2.5, np.inf]
NU_LABELS = [r"$\nu = 1/2$", r"$\nu = 3/2$", r"$\nu = 5/2$", r"$\nu = \infty$"]
NU_COLORS = [BLUE, GREEN, AMBER, RED]


def fig_matern_cov():
    r = np.linspace(0, 3, 400)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for nu, lbl, col in zip(NUS, NU_LABELS, NU_COLORS):
        ax.plot(r, matern(r, nu), color=col, lw=2.0, label=lbl)
    ax.set_xlabel(r"$r = \|x - y\|$")
    ax.set_ylabel(r"$\rho(r)$")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 3)
    ax.legend(frameon=False)
    save(fig, "bip_figure4.3.png")


def fig_matern_paths():
    rng = np.random.default_rng(7)
    n = 500
    xs = np.linspace(0, 1, n)
    z = rng.standard_normal((n, 2))  # same driving noise for every nu

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.7), sharey=True)
    for ax, nu, lbl, col in zip(axes, NUS, NU_LABELS, NU_COLORS):
        C = matern(np.abs(xs[:, None] - xs[None, :]), nu, ell=0.4)
        C += 1e-8 * np.eye(n)
        L = np.linalg.cholesky(C)
        paths = L @ z
        ax.plot(xs, paths[:, 0], color=col, lw=1.5)
        ax.plot(xs, paths[:, 1], color=GRAY, lw=1.2, alpha=0.85)
        ax.set_title(lbl, fontsize=10.5)
        ax.set_xlabel(r"$x$")
        ax.set_xlim(0, 1)
    save(fig, "bip_figure4.4.png")


# ----------------------------------------------------------------------
# 8. Anatomy of the KL expansion (Theorem 4.5.13)
# ----------------------------------------------------------------------
def fig_kl_expansion():
    rng = np.random.default_rng(3)
    n = 500
    xs = np.linspace(0, 1, n)
    h = xs[1] - xs[0]
    ell_corr = 0.3
    C = np.exp(-np.abs(xs[:, None] - xs[None, :]) / ell_corr)
    # Nystrom: eigenpairs of the integral operator T_c
    evals, evecs = np.linalg.eigh(C * h)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    evecs /= np.sqrt(h)  # L2(0,1)-normalised eigenfunctions
    # sign convention: positive at left end
    evecs *= np.sign(evecs[0, :])

    xi = rng.standard_normal(n)
    coeff = np.sqrt(np.maximum(evals, 0)) * xi

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))

    ax = axes[0]
    ax.semilogy(np.arange(1, 51), evals[:50], "o", ms=3.5, color=BLUE)
    ax.set_xlabel(r"$j$")
    ax.set_ylabel(r"$\ell_j$")
    ax.set_title("eigenvalue decay of $T_c$", fontsize=10.5)

    ax = axes[1]
    for j, col in enumerate([BLUE, GREEN, AMBER, RED]):
        ax.plot(xs, evecs[:, j], color=col, lw=1.7,
                label=rf"$\varphi_{{{j + 1}}}$")
    ax.set_xlabel(r"$x$")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.set_title("leading eigenfunctions", fontsize=10.5)

    ax = axes[2]
    ref = evecs @ coeff
    for s, col, lw in ((2, BLUE, 1.3), (8, GREEN, 1.3), (32, AMBER, 1.4)):
        ax.plot(xs, evecs[:, :s] @ coeff[:s], color=col, lw=lw,
                label=rf"$s = {s}$")
    ax.plot(xs, ref, color=DARK, lw=1.6, alpha=0.9, label=r"$s = 500$")
    ax.set_xlabel(r"$x$")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.set_title(r"truncated KL realisation $\sum_{j \leq s} a_j \varphi_j$",
                 fontsize=10.5)
    save(fig, "bip_kl_expansion.png")


# ----------------------------------------------------------------------
# 9. Uniform random fields and the elliptic PDE (Example 4.5.17)
# ----------------------------------------------------------------------
def fig_uniform_field_pde():
    rng = np.random.default_rng(11)
    n = 400
    xs = np.linspace(0, 1, n)
    J = 60
    js = np.arange(1, J + 1)
    ells = 0.45 / js**2
    phis = np.sin(np.pi * js[None, :] * xs[:, None])  # ||phi_j||_inf = 1
    m = 1.0
    a_minus = m - ells.sum()

    def solve_pde(a_vals, f=10.0):
        # -(a u')' = f, u(0)=u(1)=0, FD with a at midpoints
        h = xs[1] - xs[0]
        am = 0.5 * (a_vals[:-1] + a_vals[1:])  # midpoints, len n-1
        N = n - 2  # interior nodes
        main = (am[:-1] + am[1:]) / h**2
        off = -am[1:-1] / h**2
        A = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
        u = np.zeros(n)
        u[1:-1] = np.linalg.solve(A, np.full(N, f))
        return u

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    cols = [BLUE, GREEN, AMBER, RED]
    for col in cols:
        xi = rng.uniform(-1, 1, size=J)
        a = m + phis @ (ells * xi)
        u = solve_pde(a)
        axes[0].plot(xs, a, color=col, lw=1.5)
        axes[1].plot(xs, u, color=col, lw=1.5)
    axes[0].axhline(a_minus, color=GRAY, lw=1.1, ls="--")
    axes[0].text(0.62, a_minus + 0.03, r"$a_- > 0$", color=GRAY, fontsize=10)
    axes[0].axhline(m, color=GRAY, lw=0.8, ls=":")
    axes[0].text(0.02, m + 0.03, r"$m \equiv 1$", color=GRAY, fontsize=10)
    axes[0].set_xlabel(r"$x$")
    axes[0].set_title(r"samples $a(\omega, \cdot) = m + \sum_j \ell_j \xi_j \varphi_j$",
                      fontsize=10.5)
    axes[0].set_xlim(0, 1)
    axes[1].set_xlabel(r"$x$")
    axes[1].set_title(r"solutions $u(a(\omega), \cdot)$ of the elliptic PDE",
                      fontsize=10.5)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(bottom=0)
    save(fig, "bip_uniform_field_pde.png")


# ----------------------------------------------------------------------
# 10. Uniform is not uninformative; Jeffreys prior (Section 4.5.5)
# ----------------------------------------------------------------------
def fig_jeffreys():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))

    ax = axes[0]
    x = np.linspace(1e-4, 1, 600)
    ax.plot(x, np.ones_like(x), color=BLUE, lw=2.0,
            label=r"$\pi_X \equiv 1$  ($X \sim$ uniform$(0,1)$)")
    ax.plot(x, 0.5 / np.sqrt(x), color=RED, lw=2.0,
            label=r"$\pi_{X^2}(x) = \frac{1}{2\sqrt{x}}$")
    ax.set_ylim(0, 4)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$x$")
    ax.legend(frameon=False, fontsize=9.5)
    ax.set_title(r"uniform prior is not invariant (Ex. 4.5.26)", fontsize=10.5)

    ax = axes[1]
    ax.plot(x, beta_dist.pdf(x, 0.5, 0.5), color=GREEN, lw=2.0,
            label=r"Jeffreys: Beta$(\frac{1}{2}, \frac{1}{2})$")
    ax.plot(x, np.ones_like(x), color=GRAY, lw=1.6, ls="--",
            label=r"uniform: Beta$(1, 1)$")
    ax.set_ylim(0, 4)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$x$ (success probability)")
    ax.legend(frameon=False, fontsize=9.5)
    ax.set_title(r"Jeffreys prior for the Bernoulli likelihood", fontsize=10.5)
    save(fig, "bip_jeffreys_prior.png")


if __name__ == "__main__":
    fig_bayes_update()
    fig_credible()
    fig_additive_2d()
    fig_noise_concentration()
    fig_hellinger()
    fig_matern_cov()
    fig_matern_paths()
    fig_kl_expansion()
    fig_uniform_field_pde()
    fig_jeffreys()
