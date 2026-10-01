"""
Figures for the "image made of particles" intuition in section 2.4 (The
Generator) of subpages/books/sde_hd/index.md.

Writes into assets/images/notes/sdes_diffusion_models/:

  image_particle_smoothing.png - a grey-scale image read as a particle density:
                                 particles jitter, the image blurs, (P_t u - u)/t
                                 ~ (1/2) Laplacian u; the neighbour-average
                                 stencil; Fourier modes dying like e^{-k^2 t/2};
                                 mass conserved, max down, min up
  noise_two_spaces.png         - heat flow on the image plane (blur) vs. noise on
                                 the pixel values (grain), and the heat equation
                                 acting on the density of all images

Blurs are exact Gaussian convolutions on the pixel grid (scipy.ndimage,
reflecting boundary, so total brightness is conserved).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm, to_rgb
from scipy.ndimage import gaussian_filter

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

NPX = 256


def tint(color, f):
    r, g, b = to_rgb(color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def _save(fig, name):
    out = os.path.join(OUT, name)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _grid(ax):
    ax.grid(True, lw=0.3, alpha=0.4)


def _panel(ax, letter, color=C_INK, bg=False):
    ax.text(0.015, 0.985, f"({letter})", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left", color=color,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2) if bg else None)


def make_image():
    """Synthetic image on [0,1]^2: a ramp, a sharp disk, a dark hole,
    a patch of fine stripes and a bright dot."""
    x = (np.arange(NPX) + 0.5) / NPX
    X, Y = np.meshgrid(x, x)                      # row index = y
    u = 0.15 + 0.35 * X                            # linear ramp: Laplacian 0
    u = u + 0.5 * (((X - 0.3) ** 2 + (Y - 0.65) ** 2) < 0.16**2)
    u = u - 0.12 * (((X - 0.3) ** 2 + (Y - 0.25) ** 2) < 0.08**2)
    patch = (X > 0.58) & (X < 0.9) & (Y > 0.55) & (Y < 0.87)
    u = u + 0.25 * patch * 0.5 * (1 + np.sin(2 * np.pi * 24 * X))
    u = u + 0.6 * np.exp(-((X - 0.75) ** 2 + (Y - 0.25) ** 2) / (2 * 0.012**2))
    return x, u


def blur(u, t):
    """Heat flow / Brownian blur for time t (unit square = 1)."""
    return gaussian_filter(u, sigma=np.sqrt(t) * NPX, mode="reflect")


def _show(ax, img, vmax=1.15, **kw):
    ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), cmap="gray", vmin=0, vmax=vmax, **kw)
    ax.set_xticks([])
    ax.set_yticks([])


# ----------------------------------------------------------------------
# Figure 1: the image as a particle density
# ----------------------------------------------------------------------
def fig_image_particle_smoothing():
    rng = np.random.default_rng(1)
    x, u = make_image()
    t_vis = (6 / NPX) ** 2          # blur radius 6 pixels
    t_small = (3 / NPX) ** 2

    fig = plt.figure(figsize=(13.6, 8.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.9], hspace=0.28)
    top = gs[0].subgridspec(1, 4, wspace=0.08)
    bot = gs[1].subgridspec(1, 3, wspace=0.27)

    # (a) the image
    ax = fig.add_subplot(top[0])
    _show(ax, u)
    win = (0.11, 0.49, 0.13, 0.62)                   # x0, x1, y0, y1 of the zoom
    ax.add_patch(Rectangle((win[0], win[2]), win[1] - win[0], win[3] - win[2],
                           fill=False, ec=C_ORANGE, lw=1.5, ls="--"))
    for txt, (px, py) in (("ramp", (0.62, 0.08)), ("hole", (0.3, 0.25)),
                          ("stripes", (0.74, 0.92)), ("dot", (0.84, 0.3))):
        ax.text(px, py, txt, color=C_ORANGE, fontsize=8.5, ha="center", va="center",
                fontweight="bold")
    ax.set_title("image $u$:\nbrightness = density of particles", fontsize=10.5)
    _panel(ax, "a", bg=True)

    # (b) the particles in the zoom window, before and after a short time
    ax = fig.add_subplot(top[1])
    n_p = 1500
    xs, ys = [], []
    umax = u.max()
    while len(xs) < n_p:
        cx = rng.uniform(win[0], win[1], 4000)
        cy = rng.uniform(win[2], win[3], 4000)
        ix = np.minimum((cx * NPX).astype(int), NPX - 1)
        iy = np.minimum((cy * NPX).astype(int), NPX - 1)
        keep = rng.uniform(0, umax, 4000) < u[iy, ix]
        xs.extend(cx[keep])
        ys.extend(cy[keep])
    xs, ys = np.array(xs[:n_p]), np.array(ys[:n_p])
    s = np.sqrt(t_vis)
    xe = xs + s * rng.standard_normal(n_p)
    ye = ys + s * rng.standard_normal(n_p)
    in_disk = lambda a, b: (a - 0.3) ** 2 + (b - 0.65) ** 2 < 0.16**2
    in_hole = lambda a, b: (a - 0.3) ** 2 + (b - 0.25) ** 2 < 0.08**2
    out_ = (in_disk(xs, ys) & ~in_disk(xe, ye)) | (in_hole(xs, ys) & ~in_hole(xe, ye))
    in_ = (~in_disk(xs, ys) & in_disk(xe, ye)) | (~in_hole(xs, ys) & in_hole(xe, ye))
    ax.set_facecolor("#f4f4f4")
    ax.scatter(xs, ys, s=3, color=tint(C_GREY, 0.45), lw=0, zorder=1)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.3 + 0.16 * np.cos(th), 0.65 + 0.16 * np.sin(th), color=C_INK, lw=1.2)
    ax.plot(0.3 + 0.08 * np.cos(th), 0.25 + 0.08 * np.sin(th), color=C_INK, lw=1.2)
    for mask, col in ((out_, C_RED), (in_, C_BLUE)):
        for a_, b_, c_, d_ in zip(xs[mask], ys[mask], xe[mask], ye[mask]):
            ax.annotate("", xy=(c_, d_), xytext=(a_, b_),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1,
                                        mutation_scale=7))
    nd_o = int(np.sum(in_disk(xs, ys) & ~in_disk(xe, ye)))
    nd_i = int(np.sum(~in_disk(xs, ys) & in_disk(xe, ye)))
    nh_o = int(np.sum(in_hole(xs, ys) & ~in_hole(xe, ye)))
    nh_i = int(np.sum(~in_hole(xs, ys) & in_hole(xe, ye)))
    box = dict(fc="white", ec="none", alpha=0.85, pad=1.5)
    ax.text(0.3, 0.535, f"bright disk: {nd_o} out, {nd_i} in", ha="center", fontsize=8.5,
            color=C_INK, bbox=box, zorder=5)
    ax.text(0.3, 0.145, f"dark hole: {nh_o} out, {nh_i} in", ha="center", fontsize=8.5,
            color=C_INK, bbox=box, zorder=5)
    ax.set_xlim(win[0], win[1])
    ax.set_ylim(win[2], win[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("zoom: particles (grey) jitter; boundary\ncrossings out (red) vs. in (blue)",
                 fontsize=10.5)
    _panel(ax, "b")

    # (c) the blurred image
    ax = fig.add_subplot(top[2])
    _show(ax, blur(u, t_vis))
    ax.set_title("after time $t$:\n$P_t u$ = Gaussian blur", fontsize=10.5)
    _panel(ax, "c", bg=True)

    # (d) (P_t u - u)/t  ~  (1/2) Laplacian u
    ax = fig.add_subplot(top[3])
    d = (blur(u, t_small) - u) / t_small
    lim = np.percentile(np.abs(d), 99.3)
    ax.imshow(np.clip(-d, -lim, lim), origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r",
              norm=TwoSlopeNorm(0, -lim, lim))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("$(P_t u - u)/t \\approx \\frac{1}{2}\\Delta u$ (small $t$)\n"
                 "red = loses particles, blue = gains", fontsize=10.5)
    ax.text(0.62, 0.08, "ramp: no change", color=C_INK, fontsize=8.5, ha="center",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
    _panel(ax, "d", bg=True)

    # (e) the Laplacian as "neighbour average minus self"
    ax = fig.add_subplot(bot[0])
    f = lambda z: (0.2 + 0.25 * z + 0.6 * np.exp(-((z - 0.3) / 0.06) ** 2)
                   - 0.3 * np.exp(-((z - 0.72) / 0.07) ** 2))
    z = np.linspace(0, 1, 800)
    ax.plot(z, f(z), color=C_INK, lw=1.8, label=r"$u$")
    # exact heat flow of the 1-d profile, for comparison
    dz = z[1] - z[0]
    ax.plot(z, gaussian_filter(f(z), sigma=0.035 / dz, mode="nearest"), color=C_GREY,
            lw=1.2, ls="--", label=r"$P_t u$ (a little later)")
    h = 0.07
    for x0, col, lab in ((0.3, C_RED, "peak"), (0.5, C_GREEN, "slope"), (0.72, C_BLUE, "valley")):
        ul, ur, uc = f(x0 - h), f(x0 + h), f(x0)
        avg = 0.5 * (ul + ur)
        ax.plot([x0 - h, x0 + h], [ul, ur], color=col, lw=1, ls=":")
        ax.plot([x0 - h, x0 + h], [ul, ur], "o", mfc="white", mec=col, ms=6)
        ax.plot(x0, uc, "o", color=col, ms=6)
        ax.plot(x0, avg, "D", color=col, ms=5)
        if abs(avg - uc) > 0.02:
            ax.annotate("", xy=(x0, avg), xytext=(x0, uc),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
        ax.text(x0, (min(uc, avg) - 0.09) if col != C_BLUE else (max(uc, avg) + 0.06),
                lab, color=col, fontsize=9, ha="center", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 0.95)
    ax.set_xlabel(r"$x$")
    ax.set_title("pixel (dot) vs. average of its neighbours (diamond)", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.text(0.03, 0.03, r"$\frac{u(x-h) + u(x+h)}{2} - u(x) \approx \frac{h^2}{2}\,u''(x)$",
            transform=ax.transAxes, fontsize=10, color=C_INK, va="bottom")
    _grid(ax)
    _panel(ax, "e")

    # (f) Fourier modes: fine detail dies first
    ax = fig.add_subplot(bot[1])
    tt = np.logspace(-6, -0.5, 300)
    for ell, col, nm in ((1 / 24, C_RED, "stripes, wavelength $1/24$"),
                         (1 / 6, C_ORANGE, "wavelength $1/6$"),
                         (1 / 2, C_BLUE, "wavelength $1/2$")):
        k = 2 * np.pi / ell
        ax.semilogx(tt, np.exp(-0.5 * k**2 * tt), color=col, lw=2, label=nm)
    ax.axvline(t_vis, color=C_GREY, ls="--", lw=1)
    ax.text(t_vis / 1.15, 0.4, "time of\npanel (c)", color=C_GREY, fontsize=8.5, ha="right")
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"surviving amplitude $e^{-|k|^2 t/2}$")
    ax.set_title("fine detail dies first: lifetime $\\propto$ (size)$^2$", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower left")
    _grid(ax)
    _panel(ax, "f")

    # (g) conserved mass, max principle
    ax = fig.add_subplot(bot[2])
    ts = np.concatenate([[0], np.logspace(-6, -1, 40)])
    stats = np.array([(b.mean(), b.max(), b.min()) for b in (blur(u, t) for t in ts)])
    ax.semilogx(ts[1:], stats[1:, 1], color=C_RED, lw=2, label="brightest pixel")
    ax.semilogx(ts[1:], stats[1:, 0], color=C_INK, lw=2, label="mean brightness (= total mass)")
    ax.semilogx(ts[1:], stats[1:, 2], color=C_BLUE, lw=2, label="darkest pixel")
    ax.set_xlabel(r"time $t$")
    ax.set_title("mass conserved; max only falls, min only rises", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_ylim(0, 1.25)
    _grid(ax)
    _panel(ax, "g")
    _save(fig, "image_particle_smoothing.png")


# ----------------------------------------------------------------------
# Figure 2: which space carries the particles?
# ----------------------------------------------------------------------
def fig_noise_two_spaces():
    rng = np.random.default_rng(4)
    x, u = make_image()
    fig = plt.figure(figsize=(13.6, 4.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.3)

    ax = fig.add_subplot(gs[0])
    _show(ax, blur(u, (6 / NPX) ** 2))
    ax.set_title("particles move in the image plane:\nthe image is blurred (heat flow)",
                 fontsize=10.5)
    _panel(ax, "a", bg=True)

    ax = fig.add_subplot(gs[1])
    _show(ax, u + 0.15 * rng.standard_normal(u.shape))
    ax.set_title("noise added to pixel values:\nthe image becomes grainy, not blurred",
                 fontsize=10.5)
    _panel(ax, "b", bg=True)

    # (c) the space of 2-pixel images
    ax = fig.add_subplot(gs[2])
    centres = np.array([(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)])
    w = np.array([0.35, 0.25, 0.25, 0.15])
    s0, t = 0.05, 0.03
    g = np.linspace(-0.45, 1.45, 400)
    G1, G2 = np.meshgrid(g, g)

    def dens(var):
        p = np.zeros_like(G1)
        for (c1, c2), wi in zip(centres, w):
            p += wi * np.exp(-((G1 - c1) ** 2 + (G2 - c2) ** 2) / (2 * var)) / (2 * np.pi * var)
        return p

    ax.imshow(dens(s0**2 + t), origin="lower", extent=(g[0], g[-1], g[0], g[-1]),
              cmap="Blues", alpha=0.9)
    ax.contour(G1, G2, dens(s0**2), levels=[2, 20], colors=[C_ORANGE], linewidths=1.1,
               linestyles="--")
    # one image performing Brownian motion in image space
    n = 400
    path = np.cumsum(np.vstack([centres[1], np.sqrt(t / n) * rng.standard_normal((n, 2))]),
                     axis=0)
    ax.plot(path[:, 0], path[:, 1], color=C_INK, lw=0.8)
    ax.plot(*path[0], "o", color=C_INK, ms=4)
    ax.plot(*path[-1], "o", color=C_RED, ms=5)
    # tiny 2-pixel icons next to the data clusters
    for (c1, c2) in centres:
        ox = c1 + (0.13 if c1 > 0.5 else -0.33)
        oy = c2 + 0.12
        for j, val in enumerate((c1, c2)):
            ax.add_patch(Rectangle((ox + 0.1 * j, oy), 0.1, 0.1, fc=str(val),
                                   ec=C_INK, lw=0.8))
    ax.set_xlim(g[0], g[-1])
    ax.set_ylim(g[0], g[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("value of pixel 1")
    ax.set_ylabel("value of pixel 2")
    ax.set_title("space of all 2-pixel images: each image is ONE point;\n"
                 r"heat flow $\partial_t p = \frac{1}{2}\Delta p$ blurs the data density $p$",
                 fontsize=10.5)
    ax.text(0.98, 0.02, "orange: data density at $t=0$\nblue: density after noising\n"
            "black: one image's Brownian path", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8.3, color=C_INK,
            bbox=dict(fc="white", ec="none", alpha=0.85))
    _panel(ax, "c", bg=True)
    _save(fig, "noise_two_spaces.png")


if __name__ == "__main__":
    fig_image_particle_smoothing()
    fig_noise_two_spaces()
