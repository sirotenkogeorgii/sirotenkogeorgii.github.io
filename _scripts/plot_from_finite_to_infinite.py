"""
Plots for subpages/math_topics/from-finite-to-infinite/index.md.
Writes PNGs into assets/images/notes/math_topics/from-finite-to-infinite/.

Figures (one per machine / mechanism in the note):
  ffi_skeleton.png          -- the four-move skeleton and the completeness/compactness fork
  ffi_escape_vs_tails.png   -- Fatou can be strict (mass escapes to k = infinity) vs the uniform tail test
  ffi_blt.png               -- extension by continuity: approximate x from a dense set, the images are Cauchy
  ffi_dirichlet.png         -- Dirichlet kernels and Lebesgue constants ~ log N (no uniform bound)
  ffi_nested.png            -- nested compact sets meet, nested unbounded sets can escape (why compactness)
  ffi_konig.png             -- Koenig's lemma: a finitely branching infinite tree has an infinite branch
  ffi_diagonal.png          -- the diagonal argument: nested subsequences, then take the diagonal
  ffi_peano.png             -- Peano: Euler polygons converge, but to which solution?
  ffi_donsker.png           -- Donsker: the same random walk at three resolutions
  ffi_metastability.png     -- metastability: an eps-stable window [N, F(N)] for a monotone sequence
  ffi_lax.png               -- Lax: a consistent scheme converges iff it is stable; the rate when stable
  ffi_modica_mortola.png    -- Gamma-convergence: the recovery sequence must adapt its width
  ffi_lost_compactness.png  -- three ways a bounded sequence loses compactness
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

OUT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "images", "notes",
                 "math_topics", "from-finite-to-infinite"))
os.makedirs(OUT, exist_ok=True)

C_BLUE = "#2c3e94"
C_GREEN = "#1f9d55"
C_ORANGE = "#e67e22"
C_RED = "#c0392b"
C_PURPLE = "#534AB7"
C_GRAY = "#7f8c8d"
C_INK = "#222222"
C_BLUE_FILL = "#dde8f7"
C_ORANGE_FILL = "#fdebd0"
BLUE_RAMP = ["#b9c2d4", "#7f93c4", C_BLUE]  # one hue, light -> dark, for a magnitude

plt.rcParams.update({
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "font.size": 11,
})


def _save(fig, name):
    out = os.path.join(OUT, name)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, alpha=0.18)


# --------------------------------------------------------------------------- #
# 1. The skeleton                                                             #
# --------------------------------------------------------------------------- #
def plot_skeleton():
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec, tc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.16",
                                    fc=fc, ec=ec, lw=1.8))
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=11.5,
                fontweight="bold", color=tc)
        ax.text(x + w / 2, y + 0.17, body, ha="center", va="bottom", fontsize=9.3,
                color=C_INK, linespacing=1.4)

    def arrow(p, q, color=C_INK):
        ax.annotate("", xy=q, xytext=p,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                    shrinkA=1, shrinkB=1, mutation_scale=14))

    steps = [
        ("1  Approximate", "solve the finite problem\n(Galerkin space, grid,\npartial sum, finite subset)"),
        ("2  Estimate", "a bound on $X_N$ that\ndoes not depend on $N$"),
        ("3  Extract", "a limit $X$ from the $X_N$\n(along a subsequence,\nor along the whole sequence)"),
        ("4  Identify", "the property $P$ is closed\nunder that convergence"),
    ]
    xs = [0.2, 3.1, 6.0, 8.9]
    w, y0, h = 2.6, 3.55, 1.85
    for x, (t, b) in zip(xs, steps):
        box(x, y0, w, h, t, b, "#f6f7fb", C_INK, C_INK)
    for x in xs[:-1]:
        arrow((x + w + 0.03, y0 + h / 2), (x + w + 0.28, y0 + h / 2))
    ax.text(xs[1] + w / 2, y0 - 0.1, "where all the analysis lives", ha="center", va="top",
            fontsize=9.5, style="italic", color=C_GRAY)

    bw, bh, by = 4.6, 1.75, 0.6
    box(1.0, by, bw, bh, "step 3 by completeness",
        "Cauchy ⇒ a unique limit, step 4 is automatic\n"
        "continuity (§1) · consistency (§2) · stability (§5)",
        C_BLUE_FILL, C_BLUE, C_BLUE)
    box(6.6, by, bw, bh, "step 3 by compactness",
        "subsequence ⇒ a limit point, no uniqueness,\nand step 4 is the real work\n"
        "compactness (§3) · transfer (§4)",
        C_ORANGE_FILL, C_ORANGE, C_ORANGE)
    xc = xs[2] + w / 2
    arrow((xc - 0.3, y0 - 0.08), (1.0 + bw - 0.6, by + bh + 0.1), C_BLUE)
    arrow((xc + 0.3, y0 - 0.08), (6.6 + 1.3, by + bh + 0.1), C_ORANGE)
    ax.set_title("The skeleton: approximate, estimate, extract, identify — and the fork at step 3",
                 fontsize=13)
    _save(fig, "ffi_skeleton.png")


# --------------------------------------------------------------------------- #
# 2. Escape to infinity vs uniform tails                                      #
# --------------------------------------------------------------------------- #
def plot_escape_vs_tails():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 5.2),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})
    n = 8
    axL.set_xlim(-0.6, n + 2.0)
    axL.set_ylim(n + 2.0, -0.9)
    axL.set_aspect("equal")
    axL.axis("off")
    for i in range(1, n + 1):
        for k in range(1, n + 1):
            on = (i == k)
            axL.add_patch(Rectangle((k - 0.5, i - 0.5), 1, 1, fc=C_BLUE if on else "white",
                                    ec="#c8cdd6", lw=0.8))
            axL.text(k, i, "1" if on else "0", ha="center", va="center",
                     color="white" if on else "#9aa3b2", fontsize=9.5,
                     fontweight="bold" if on else "normal")
    for i in range(1, n + 1):
        axL.text(n + 1.3, i, "1", ha="center", va="center", color=C_BLUE, fontsize=10.5,
                 fontweight="bold")
    for k in range(1, n + 1):
        axL.text(k, n + 1.3, "0", ha="center", va="center", color=C_ORANGE, fontsize=10.5,
                 fontweight="bold")
    axL.text(n + 1.3, -0.35, r"$\sum_k a_{n,k}$", ha="center", va="center", color=C_BLUE,
             fontsize=10.5)
    axL.text(-0.05, n + 1.3, r"$\lim_n a_{n,k}$", ha="right", va="center", color=C_ORANGE,
             fontsize=10.5)
    axL.text(n + 1.3, n + 1.3, r"$0<1$", ha="center", va="center", color=C_RED, fontsize=11,
             fontweight="bold")
    axL.text(0.5, -0.35, r"$k\ \rightarrow$", ha="left", va="center", fontsize=10.5)
    axL.text(-0.05, 0.5, r"$n$", ha="right", va="center", fontsize=10.5)
    axL.text(-0.05, 1.3, r"$\downarrow$", ha="right", va="center", fontsize=10.5)
    axL.set_title(r"$a_{n,k}=\delta_{nk}$: Fatou is strict, the mass escapes to $k=\infty$",
                  fontsize=11)

    K = np.arange(0, 13)
    axR.semilogy(K, np.ones_like(K, dtype=float), "o-", color=C_ORANGE, ms=4.5, lw=1.6,
                 label=r"$x^{(n)}=e_n$:  $\sup_n\sum_{k>K}|x^{(n)}_k|^2 = 1$ for every $K$")
    axR.semilogy(K, 4.0 ** (-K) / 3.0, "s-", color=C_BLUE, ms=4.5, lw=1.6,
                 label=r"$|x^{(n)}_k|\leq 2^{-k}$:  tails $\leq 4^{-K}/3$, uniformly in $n$")
    axR.set_xlabel("$K$ (where the tail starts)")
    axR.set_ylabel(r"$\sup_n \sum_{k>K} |x^{(n)}_k|^2$")
    axR.set_title("The uniform tail test: coordinatewise null + uniform tails ⇔ "
                  r"$\ell^2$-null", fontsize=11)
    axR.legend(loc="lower left", framealpha=0.95)
    _clean(axR)
    fig.tight_layout()
    _save(fig, "ffi_escape_vs_tails.png")


# --------------------------------------------------------------------------- #
# 3. Extension by continuity (BLT)                                            #
# --------------------------------------------------------------------------- #
def plot_blt():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 5.6))
    lv = 4
    g = np.arange(0, 1 + 1e-9, 2.0 ** -lv)
    GX, GY = np.meshgrid(g, g)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    x = np.array([0.5825, 0.7161])
    approx = [np.round(x * 2 ** j) / 2 ** j for j in (1, 2, 4)]
    theta = np.deg2rad(35.0)
    C = 1.3
    T = C * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    axL.plot(pts[:, 0], pts[:, 1], ".", color="#b9c2d4", ms=4, label="$D$: dyadic points (dense in $X$)")
    for a, lab, off in zip(approx, ["$x_1$", "$x_2$", "$x_3$"], [(-18, -14), (-20, 4), (6, -16)]):
        axL.plot(*a, "o", color=C_BLUE, ms=7, zorder=4)
        axL.annotate(lab, a, xytext=off, textcoords="offset points", fontsize=11, color=C_BLUE)
    axL.annotate("$x$", x, xytext=(8, 6), textcoords="offset points", fontsize=11, color=C_ORANGE)
    for a, b in zip(approx[:-1], approx[1:]):
        axL.annotate("", xy=b, xytext=a,
                     arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.2, shrinkA=5, shrinkB=5))
    axL.annotate("", xy=x, xytext=approx[-1],
                 arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.2, shrinkA=5, shrinkB=5))
    axL.plot(*x, "*", color=C_ORANGE, ms=15, zorder=5, label="$x\\in X$, not in $D$")
    axL.set_xlim(-0.04, 1.04)
    axL.set_ylim(-0.04, 1.04)
    axL.set_aspect("equal")
    axL.set_title("In $X$: approximate $x$ by $x_n\\in D$, where $T$ is defined", fontsize=11)
    axL.legend(loc="lower right", framealpha=0.95)
    _clean(axL)

    tp = pts @ T.T
    axR.plot(tp[:, 0], tp[:, 1], ".", color="#b9c2d4", ms=4, label="$T(D)$")
    ta = [T @ a for a in approx]
    tx = T @ x
    for a, lab, off in zip(ta, ["$Tx_1$", "$Tx_2$", "$Tx_3$"], [(8, -12), (-36, -4), (8, -13)]):
        axR.plot(*a, "o", color=C_BLUE, ms=7, zorder=4)
        axR.annotate(lab, a, xytext=off, textcoords="offset points", fontsize=11, color=C_BLUE)
    axR.annotate("$Tx$", tx, xytext=(8, 6), textcoords="offset points", fontsize=11, color=C_ORANGE)
    for a, b in zip(ta[:-1], ta[1:]):
        axR.annotate("", xy=b, xytext=a,
                     arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.2, shrinkA=5, shrinkB=5))
    for a in ta[:2]:
        r = C * np.linalg.norm(a - tx)
        axR.add_patch(Circle(tx, r, fc="none", ec=C_ORANGE, ls="--", lw=1.1))
    axR.plot(*tx, "*", color=C_ORANGE, ms=15, zorder=5, mfc="white", mew=1.8,
             label="$Tx := \\lim Tx_n$ (defined by the extension)")
    axR.text(0.03, 0.17, "$\\|Tx_n - Tx\\| \\leq C\\|x_n - x\\|$", transform=axR.transAxes,
             fontsize=10.5, color=C_ORANGE)
    lo, hi = tp.min(0) - 0.08, tp.max(0) + 0.08
    axR.set_xlim(lo[0], hi[0])
    axR.set_ylim(lo[1], hi[1])
    axR.set_aspect("equal")
    axR.set_title("In $Y$: the uniform bound $\\|Td\\|\\leq C\\|d\\|$ makes $(Tx_n)$ Cauchy",
                  fontsize=11)
    axR.legend(loc="lower right", framealpha=0.95)
    _clean(axR)
    fig.tight_layout()
    _save(fig, "ffi_blt.png")


# --------------------------------------------------------------------------- #
# 4. Dirichlet kernels and Lebesgue constants                                  #
# --------------------------------------------------------------------------- #
def _dirichlet(N, t):
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.sin((N + 0.5) * t) / np.sin(t / 2)
    return np.where(np.abs(t) < 1e-12, 2 * N + 1, d)


def plot_dirichlet():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 5.0))
    t = np.linspace(-np.pi, np.pi, 4001)
    axL.plot(t, _dirichlet(4, t), color=C_BLUE, lw=1.7, label="$N=4$")
    axL.plot(t, _dirichlet(16, t), color=C_ORANGE, lw=1.1, label="$N=16$")
    axL.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axL.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "$0$", r"$\pi/2$", r"$\pi$"])
    axL.set_title(r"Dirichlet kernels $D_N(t)=\frac{\sin((N+\frac{1}{2})t)}{\sin(t/2)}$:"
                  "\nthe side lobes do not shrink", fontsize=11)
    axL.legend(loc="upper right", framealpha=0.95)
    _clean(axL)

    tt = np.linspace(-np.pi, np.pi, 40001)
    Ns = np.unique(np.round(np.logspace(0, np.log10(400), 36)).astype(int))
    L = np.array([np.trapezoid(np.abs(_dirichlet(N, tt)), tt) / (2 * np.pi) for N in Ns])
    axR.semilogx(Ns, L, "o", color=C_BLUE, ms=4.5, label=r"$L_N = \frac{1}{2\pi}\|D_N\|_{L^1} = \|S_N\|_{C(\mathbb{T})\to\mathbb{R}}$")
    axR.semilogx(Ns, 4 / np.pi ** 2 * np.log(Ns) + 1.2706, "--", color=C_GRAY, lw=1.4,
                 label=r"$\frac{4}{\pi^2}\ln N + 1.27$")
    axR.set_xlabel("$N$")
    axR.set_ylabel("$L_N$")
    axR.set_title(r"Lebesgue constants: $\sup_N\|S_N\| = \infty$, so convergence on the dense"
                  "\nset of trigonometric polynomials cannot transfer to all of $C(\\mathbb{T})$",
                  fontsize=11)
    axR.legend(loc="upper left", framealpha=0.95)
    _clean(axR)
    fig.tight_layout()
    _save(fig, "ffi_dirichlet.png")


# --------------------------------------------------------------------------- #
# 5. Nested compact sets meet; nested unbounded sets can escape                #
# --------------------------------------------------------------------------- #
def plot_nested():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)
    ns = np.arange(1, 8)
    c = 0.0
    for n in ns:
        r = 1.6 / (n + 0.6)
        axL.plot([c - r, c + r], [n, n], color=C_BLUE, lw=6, solid_capstyle="butt", alpha=0.9)
        axL.text(c + r + 0.06, n, f"$K_{n}$", va="center", fontsize=10, color=C_BLUE)
    axL.axvline(c, color=C_ORANGE, ls=":", lw=1.5)
    axL.plot([c], [7.9], "o", color=C_ORANGE, ms=7)
    axL.text(c + 0.1, 7.9, r"$\bigcap_n K_n \neq \emptyset$: the thread exists", va="center",
             fontsize=10.5, color=C_ORANGE)
    axL.set_xlim(-1.4, 1.9)
    axL.set_ylim(8.6, 0.3)
    axL.set_yticks(ns)
    axL.set_ylabel("$n$")
    axL.set_title("Compact stages: nested nonempty compact $K_n$ always meet", fontsize=11)
    _clean(axL)

    for n in ns:
        axR.plot([n, 9.6], [n, n], color=C_RED, lw=6, solid_capstyle="butt", alpha=0.85)
        axR.annotate("", xy=(10.2, n), xytext=(9.5, n),
                     arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.4, mutation_scale=12))
        axR.text(n - 0.15, n, f"$X_{n}$", va="center", ha="right", fontsize=10, color=C_RED)
    axR.text(0.6, 7.9, r"$\bigcap_n X_n = \emptyset$: nonempty at every finite stage,"
             "\nbut the thread escapes to infinity", va="center", fontsize=10.5, color=C_RED)
    axR.set_xlim(0.2, 10.6)
    axR.set_xlabel("the sets $X_n=\\{n, n+1, \\dots\\}$")
    axR.set_title("Non-compact stages: nested nonempty $X_n$ with empty intersection", fontsize=11)
    _clean(axR)
    fig.tight_layout()
    _save(fig, "ffi_nested.png")


# --------------------------------------------------------------------------- #
# 6. Koenig's lemma                                                           #
# --------------------------------------------------------------------------- #
def plot_konig(seed=2, name="ffi_konig.png"):
    DEPTH = 6
    rng = np.random.default_rng(seed)
    kind = {}
    edges = []

    def grow_finite(d, i, left):
        kind[(d, i)] = "fin"
        if left <= 0 or d >= DEPTH:
            return
        for c_ in (0, 1):
            if rng.random() < 0.75:
                edges.append(((d, i), (d + 1, 2 * i + c_)))
                grow_finite(d + 1, 2 * i + c_, left - 1)

    def grow_inf(d, i):
        kind[(d, i)] = "inf"
        if d == DEPTH:
            return
        first = int(rng.integers(2))
        kinds = ["fin", "fin"]
        kinds[first] = "inf"
        if rng.random() < 0.5:
            kinds[1 - first] = "inf"
        for c_ in (0, 1):
            child = (d + 1, 2 * i + c_)
            if kinds[c_] == "inf":
                edges.append(((d, i), child))
                grow_inf(*child)
            else:
                max_left = DEPTH - d - 2
                if max_left < 0 or rng.random() < 0.25:
                    continue
                edges.append(((d, i), child))
                grow_finite(child[0], child[1], int(rng.integers(0, max_left + 1)))

    grow_inf(0, 0)
    # the branch: always step into a child with an infinite subtree (leftmost here)
    branch = [(0, 0)]
    while branch[-1][0] < DEPTH:
        d, i = branch[-1]
        opts = [(d + 1, 2 * i + c_) for c_ in (0, 1) if kind.get((d + 1, 2 * i + c_)) == "inf"]
        branch.append(opts[int(rng.integers(len(opts)))])

    def pos(node):
        d, i = node
        return (i + 0.5) / 2 ** d, -d

    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    for a, b in edges:
        inf_edge = kind[a] == "inf" and kind[b] == "inf"
        (xa, ya), (xb, yb) = pos(a), pos(b)
        ax.plot([xa, xb], [ya, yb], color=C_BLUE if inf_edge else "#c3c9d3",
                lw=1.3 if inf_edge else 0.9, zorder=1)
    bx, by_ = zip(*[pos(n_) for n_ in branch])
    ax.plot(bx, by_, color=C_ORANGE, lw=3.2, zorder=2, solid_capstyle="round")
    for node, k in kind.items():
        xx, yy = pos(node)
        ax.plot(xx, yy, "o", color=C_BLUE if k == "inf" else "#c3c9d3",
                ms=5.5 if k == "inf" else 4, zorder=3)
    for node in [n_ for n_, k in kind.items() if k == "inf" and n_[0] == DEPTH]:
        xx, yy = pos(node)
        ax.text(xx, yy - 0.35, "⋮", ha="center", va="center", fontsize=11, color=C_BLUE)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-DEPTH - 0.8, 0.5)
    ax.set_yticks(range(0, -DEPTH - 1, -1))
    ax.set_yticklabels([f"level {d}" for d in range(DEPTH + 1)])
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    handles = [
        Line2D([], [], marker="o", color=C_BLUE, lw=1.3, ms=6, label="node whose subtree is infinite (there is always one per level)"),
        Line2D([], [], marker="o", color="#c3c9d3", lw=0.9, ms=5, label="finite subtree: it dies out after finitely many levels"),
        Line2D([], [], color=C_ORANGE, lw=3.2, label="the branch: at each node step into an infinite subtree"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), framealpha=0.95)
    ax.set_title("König's lemma: an infinite, finitely branching tree has an infinite branch", fontsize=12)
    fig.tight_layout()
    _save(fig, name)


# --------------------------------------------------------------------------- #
# 7. The diagonal argument                                                    #
# --------------------------------------------------------------------------- #
def plot_diagonal():
    rng = np.random.default_rng(5)
    M, rows = 30, 6
    keep = [np.arange(1, M + 1)]
    for j in range(1, rows):
        prev = keep[-1]
        mask = rng.random(len(prev)) < 0.74
        mask[0] = True
        keep.append(prev[mask])
    diag = [keep[j][j] for j in range(rows)]  # j-th term (0-based) of row j

    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    for j in range(rows):
        y = -j
        ax.plot(np.arange(1, M + 1), np.full(M, y), ".", color="#dfe3ea", ms=5, zorder=1)
        ax.plot(keep[j], np.full(len(keep[j]), y), "o", color=C_BLUE, ms=5.5, zorder=2)
        ax.plot([diag[j]], [y], "o", color=C_ORANGE, ms=11, mfc="none", mew=2.2, zorder=3)
    ax.plot(diag, [-j for j in range(rows)], ":", color=C_ORANGE, lw=1.6, zorder=2)
    ax.set_yticks([-j for j in range(rows)])
    ax.set_yticklabels(["row 0: all indices $n$"] +
                       [f"row {j}: coordinate {j} converges" for j in range(1, rows)])
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("index $n$ of the original sequence")
    ax.set_xlim(0, M + 10)
    ax.set_ylim(-rows + 0.4, 0.7)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(M + 9.3, -(rows - 1) / 2,
            "diagonal $n_j$ = $j$-th term of row $j$:\nfor $j\\geq i$ it lies in row $i$, so\n"
            "every coordinate converges along $(n_j)$",
            ha="right", va="center", fontsize=10.5, color=C_ORANGE,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_ORANGE, alpha=0.95))
    ax.set_title("The diagonal argument: refine the subsequence coordinate by coordinate, then take the diagonal",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "ffi_diagonal.png")


# --------------------------------------------------------------------------- #
# 8. Peano: Euler polygons converge, but to which solution?                   #
# --------------------------------------------------------------------------- #
def _euler_sqrt(h, x0, T=3.0):
    n = int(round(T / h))
    xs = np.empty(n + 1)
    xs[0] = x0
    for k in range(n):
        xs[k + 1] = xs[k] + h * np.sqrt(abs(xs[k]))
    return h * np.arange(n + 1), xs


def plot_peano():
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    t = np.linspace(0, 3, 600)
    for c in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        ax.plot(t, np.where(t > c, (t - c) ** 2 / 4, 0.0), color="#c3c9d3", lw=1.1, zorder=1)
    ax.text(1.0, 2.2, "grey: the exact solutions $x_c(t)=\\frac{(t-c)_+^2}{4}$,\none for every $c\\geq 0$",
            ha="left", va="top", fontsize=10, color=C_GRAY)
    for h, ls, lw in [(0.5, ":", 1.6), (0.15, "--", 1.6), (0.02, "-", 2.0)]:
        tt, xx = _euler_sqrt(h, h ** 2)
        ax.plot(tt, xx, ls=ls, lw=lw, color=C_ORANGE, label=f"Euler, $h={h}$, start $x_h(0)=h^2$", zorder=3)
    ax.plot(t, 0 * t, color=C_BLUE, lw=2.6, label="Euler, any $h$, start $x_h(0)=0$: stays at $0$", zorder=3)
    ax.text(2.42, 1.17, "limit $t^2/4$", color=C_ORANGE, fontsize=10.5, rotation=47)
    ax.text(1.6, 0.06, "limit $0$", color=C_BLUE, fontsize=10.5)
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.08, 2.3)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_title("$x' = \\sqrt{|x|},\\ x(0)=0$: two consistent approximation schemes, "
                 "two different limits", fontsize=12)
    ax.legend(loc="upper left", framealpha=0.95)
    _clean(ax)
    fig.tight_layout()
    _save(fig, "ffi_peano.png")


# --------------------------------------------------------------------------- #
# 9. Donsker                                                                  #
# --------------------------------------------------------------------------- #
def plot_donsker():
    rng = np.random.default_rng(7)
    S = np.concatenate([[0], np.cumsum(rng.choice([-1, 1], size=2000))])
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3), sharey=True)
    for ax, n in zip(axes, [20, 200, 2000]):
        t = np.arange(n + 1) / n
        ax.plot(t, S[:n + 1] / np.sqrt(n), color=C_BLUE, lw=1.9 if n == 20 else 1.1,
                marker="o" if n == 20 else None, ms=3.5)
        ax.axhline(0, color=C_GRAY, lw=0.7)
        ax.set_title(f"$n={n}$ steps on $[0,1]$")
        ax.set_xlabel("$t$")
        _clean(ax)
    axes[0].set_ylabel(r"$S_{\lfloor nt\rfloor}/\sqrt{n}$")
    fig.suptitle("Donsker: the same walk, rescaled. The CLT fixes the finite-dimensional "
                 "distributions; a uniform modulus of continuity gives tightness", fontsize=12)
    fig.tight_layout()
    _save(fig, "ffi_donsker.png")


# --------------------------------------------------------------------------- #
# 10. Metastability                                                           #
# --------------------------------------------------------------------------- #
def plot_metastability():
    eps = 0.05
    n = np.arange(1, 121)
    seqs = [
        (1 - 1 / np.sqrt(n + 1), C_BLUE, "$a_n = 1-(n+1)^{-1/2}$ (slow)"),
        (1 - np.exp(-n / 5), C_ORANGE, "$b_n = 1-e^{-n/5}$ (fast)"),
    ]

    def first_window(seq):
        for N in range(1, len(seq) // 2):
            if seq[2 * N - 1] - seq[N - 1] < eps:
                return N

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for seq, col, lab in seqs:
        N = first_window(seq)
        ax.plot(n, seq, ".", color=col, ms=4.5, label=lab)
        ax.axvspan(N, 2 * N, color=col, alpha=0.13, lw=0)
        ax.annotate(f"$[N, F(N)] = [{N}, {2 * N}]$:\nmoves less than $\\varepsilon$",
                    xy=((N + 2 * N) / 2, seq[N - 1] - 0.06 if col == C_BLUE else 0.66),
                    ha="center", va="top", fontsize=10,
                    color=col)
    ax.set_xlabel("$n$")
    ax.set_ylabel("value")
    ax.set_ylim(0.25, 1.05)
    ax.set_title("Metastability: a monotone sequence in $[0,1]$ is $\\varepsilon$-stable on some window "
                 "$[N, F(N)]$ with $N\\leq M(\\varepsilon, F)$\n"
                 "(here $\\varepsilon = 0.05$, $F(N) = 2N$; the bound $M$ does not depend on the sequence)",
                 fontsize=11)
    ax.legend(loc="lower right", framealpha=0.95)
    _clean(ax)
    fig.tight_layout()
    _save(fig, "ffi_metastability.png")


# --------------------------------------------------------------------------- #
# 11. Lax equivalence                                                         #
# --------------------------------------------------------------------------- #
def plot_lax():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 5.0))
    lam, T = 10.0, 2.0
    tt = np.linspace(0, T, 400)
    axL.plot(tt, np.exp(-lam * tt), color=C_GRAY, lw=2.4, label="exact $e^{-\\lambda t}$")
    for h, col, ls, lab in [
        (0.05, C_BLUE, "-", "$h=0.05$, $\\lambda h=0.5$: stable, converges"),
        (0.19, C_ORANGE, "--", "$h=0.19$, $\\lambda h=1.9$: stable, oscillates"),
        (0.21, C_RED, "-", "$h=0.21$, $\\lambda h=2.1$: unstable, blows up"),
    ]:
        k = np.arange(int(T / h) + 1)
        axL.plot(h * k, (1 - lam * h) ** k, ls=ls, marker="o", ms=3, lw=1.3, color=col, label=lab)
    axL.set_ylim(-2.5, 3.8)
    axL.set_xlabel("$t$")
    axL.set_ylabel("$u$")
    axL.set_title("Explicit Euler for $u'=-\\lambda u$: consistent for every $h$,\n"
                  "convergent only while stable ($|1-\\lambda h|\\leq 1$)", fontsize=11)
    axL.legend(loc="upper right", framealpha=0.95)
    _clean(axL)

    hs = np.logspace(-3.3, -1.3, 14)
    err = np.abs((1 - lam * hs) ** np.round(1.0 / hs) - np.exp(-lam))
    axR.loglog(hs, err, "o-", color=C_BLUE, ms=4.5, lw=1.4, label="global error at $t=1$")
    axR.loglog(hs, err[-1] * hs / hs[-1], "--", color=C_GRAY, lw=1.3, label="slope 1: $O(h)$")
    axR.set_xlabel("$h$")
    axR.set_ylabel("$|u_h(1) - e^{-\\lambda}|$")
    axR.set_title("Stable ⇒ global error $\\leq$ (stability constant) × $\\sum$ local errors $= O(h)$",
                  fontsize=11)
    axR.legend(loc="upper left", framealpha=0.95)
    _clean(axR)
    fig.tight_layout()
    _save(fig, "ffi_lax.png")


# --------------------------------------------------------------------------- #
# 12. Modica-Mortola / Gamma-convergence                                      #
# --------------------------------------------------------------------------- #
def plot_modica_mortola():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 5.0))
    x = np.linspace(-1, 1, 4001)
    sq2 = np.sqrt(2.0)
    for eps, col, lw in zip([0.4, 0.15, 0.05], BLUE_RAMP, [1.4, 1.7, 2.1]):
        axL.plot(x, np.tanh(x / (sq2 * eps)), color=col, lw=lw, label=f"$\\varepsilon={eps}$")
    axL.plot([-1, 0, 0, 1], [-1, -1, 1, 1], "--", color=C_ORANGE, lw=1.4, label="sharp interface (the limit)")
    axL.set_xlabel("$x$")
    axL.set_ylabel("$u_\\varepsilon$")
    axL.set_title("Optimal profiles $u_\\varepsilon(x)=\\tanh(x/\\sqrt{2}\\varepsilon)$: "
                  "the transition layer has width $\\sim\\varepsilon$", fontsize=11)
    axL.legend(loc="upper left", framealpha=0.95)
    _clean(axL)

    def F(u, eps):
        du = np.gradient(u, x)
        return np.trapezoid(eps / 2 * du ** 2 + (1 - u ** 2) ** 2 / (4 * eps), x)

    epss = np.logspace(-2, np.log10(0.5), 30)
    E_opt = [F(np.tanh(x / (sq2 * e)), e) for e in epss]
    E_fix = [F(np.tanh(x / (sq2 * 0.3)), e) for e in epss]
    cW = 2 * sq2 / 3
    axR.loglog(epss, E_opt, "o-", color=C_BLUE, ms=4, lw=1.4,
               label="recovery sequence (width $\\propto\\varepsilon$): $F_\\varepsilon\\to c_W$")
    axR.loglog(epss, E_fix, "s--", color=C_ORANGE, ms=4, lw=1.4,
               label="profile frozen at width $0.3$: $F_\\varepsilon\\sim 1/\\varepsilon$")
    axR.axhline(cW, color=C_GRAY, ls=":", lw=1.4)
    axR.text(0.011, cW * 1.12, "$c_W = \\int_{-1}^{1}\\sqrt{2W}\\,du = 2\\sqrt{2}/3$", fontsize=10, color=C_GRAY)
    axR.set_xlabel("$\\varepsilon$")
    axR.set_ylabel("$F_\\varepsilon(u)$")
    axR.set_title("$F_\\varepsilon(u)=\\int \\frac{\\varepsilon}{2}|u'|^2 + \\frac{W(u)}{\\varepsilon}$, "
                  "$W(u)=\\frac{(1-u^2)^2}{4}$:\nthe lower bound is stable, the recovery sequence is consistent",
                  fontsize=11)
    axR.legend(loc="upper right", framealpha=0.95)
    _clean(axR)
    fig.tight_layout()
    _save(fig, "ffi_modica_mortola.png")


# --------------------------------------------------------------------------- #
# 13. Three ways to lose compactness                                          #
# --------------------------------------------------------------------------- #
def plot_lost_compactness():
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3))
    x = np.linspace(-np.pi, np.pi, 3001)
    ax = axes[0]
    nn = 7
    ax.plot(x, np.sin(nn * x), color=C_BLUE, lw=0.8, label="$\\sin(nx)\\rightharpoonup 0$")
    ax.plot(x, np.sin(nn * x) ** 2, color=C_ORANGE, lw=0.8, label="$\\sin^2(nx)\\rightharpoonup \\frac{1}{2}$")
    ax.axhline(0.5, color=C_ORANGE, ls="--", lw=1.6)
    ax.axhline(0.0, color=C_BLUE, ls="--", lw=1.6)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r"$-\pi$", "$0$", r"$\pi$"])
    ax.set_title("oscillation ($n=7$): the weak limit of\n$\\sin^2$ is not the square of the weak limit", fontsize=10.5)
    ax.legend(loc="lower left", framealpha=0.95)
    _clean(ax)

    phi = lambda y: np.exp(-y ** 2)
    ax = axes[1]
    xx = np.linspace(-3, 3, 3001)
    for lam, col in zip([1, 3, 9], BLUE_RAMP):
        ax.plot(xx, np.sqrt(lam) * phi(lam * xx), color=col, lw=1.6, label=f"$\\lambda={lam}$")
    ax.set_title("concentration: $\\sqrt{\\lambda}\\,\\varphi(\\lambda x)$,\nsame $L^2$ norm, weak limit $0$", fontsize=10.5)
    ax.legend(loc="upper right", framealpha=0.95)
    _clean(ax)

    ax = axes[2]
    xx = np.linspace(-2.5, 9.5, 3001)
    for c, col in zip([0, 3, 6], BLUE_RAMP):
        ax.plot(xx, phi(xx - c), color=col, lw=1.6, label=f"$n={c:g}$")
    ax.annotate("", xy=(8.9, 0.55), xytext=(7.2, 0.55),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.3))
    ax.set_title("escape: $\\varphi(x-n)$, the mass\nruns off to infinity, weak limit $0$", fontsize=10.5)
    ax.legend(loc="upper left", framealpha=0.95)
    _clean(ax)
    fig.suptitle("Three ways a bounded sequence can fail to have a strongly convergent subsequence",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "ffi_lost_compactness.png")


if __name__ == "__main__":
    plot_skeleton()
    plot_escape_vs_tails()
    plot_blt()
    plot_dirichlet()
    plot_nested()
    plot_konig()
    plot_diagonal()
    plot_peano()
    plot_donsker()
    plot_metastability()
    plot_lax()
    plot_modica_mortola()
    plot_lost_compactness()
