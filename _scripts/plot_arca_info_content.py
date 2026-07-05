"""
Plot for the Information Content Measures section of
subpages/books/algorithmic_randomness_computable_analysis/index.md.
Writes PNGs into assets/images/notes/books/algorithmic_randomness_computable_analysis/.

Figures:
  arca_epigraph_kraft.png -- left: the epigraph {(sigma,k) : F(sigma) <= k} of a
                             partial F as a set of lattice points enumerated over
                             time; right: the Kraft budget spent by the same F.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "assets",
        "images",
        "notes",
        "books",
        "algorithmic_randomness_computable_analysis",
    )
)
os.makedirs(OUT, exist_ok=True)

BLUE = "#2c3e94"
GREEN = "#1f9d55"
ORANGE = "#e67e22"
RED = "#c0392b"
GRAY = "#7f8c8d"
INK = "#222222"

plt.rcParams.update({
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "font.size": 11,
})


def _save(fig, name, tight=True):
    out = os.path.join(OUT, name)
    fig.savefig(out, dpi=160, bbox_inches="tight" if tight else None)
    plt.close(fig)
    print("wrote", out)


# The running example: a partial F with dom(F) = {0, 10, 001}.
# Deliberately NOT prefix-free (0 is a prefix of 001) -- the definition of an
# information content measure never asks for prefix-freeness.
WORDS = ["λ", "0", "1", "00", "01", "10", "11", "000", "001"]
F = {"0": 2, "10": 4, "001": 5}
COL = {"0": BLUE, "10": GREEN, "001": ORANGE}
K_MAX = 8


def fig_epigraph_kraft():
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.2, 5.0), gridspec_kw={"width_ratios": [1.3, 1.0]}
    )
    fig.suptitle(
        "Information content measure = enumerable upper bounds + finite Kraft budget",
        fontsize=12.5,
        y=0.985,
    )

    # ------------------------------------------------------------------ left
    axL.set_title(r"c.e. epigraph  $\{(\sigma,k) : F(\sigma) \leq k\}$", pad=10)

    for x in range(len(WORDS)):
        axL.axvline(x, color=GRAY, lw=0.5, ls=":", alpha=0.35, zorder=0)

    for x, w in enumerate(WORDS):
        if w not in F:
            continue
        c = COL[w]
        f = F[w]
        # the column of the epigraph above sigma: everything from F(sigma) upward
        axL.add_patch(
            Rectangle(
                (x - 0.30, f), 0.60, K_MAX + 0.75 - f,
                facecolor=c, alpha=0.12, edgecolor="none", zorder=1,
            )
        )
        ks = list(range(f, K_MAX + 1))
        axL.scatter([x] * len(ks), ks, s=42, color=c, zorder=3)
        # the least element of the column is the value F(sigma): graph of F
        axL.scatter([x], [f], s=110, facecolor=c, edgecolor=INK,
                    linewidth=1.4, zorder=4)
        axL.text(x, K_MAX + 0.42, "⋮", ha="center", va="center",
                 color=c, fontsize=13, zorder=3)

    # the c.e. story: pairs show up one at a time, better bounds later
    axL.annotate(
        "stage 1: (001, 7)", xy=(8, 7), xytext=(5.65, 7.55),
        fontsize=9, color=INK,
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0,
                        shrinkA=2, shrinkB=6),
    )
    axL.text(5, 3.15, "stage 2: (10, 4)", ha="center", fontsize=9, color=INK)
    axL.annotate(
        "stage 3: (001, 5)", xy=(8, 5), xytext=(5.55, 2.2),
        fontsize=9, color=INK,
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0,
                        shrinkA=2, shrinkB=6),
    )
    axL.add_patch(
        FancyArrowPatch(
            (8.38, 6.75), (8.38, 5.25),
            arrowstyle="->", mutation_scale=11,
            color=GRAY, lw=1.1, linestyle=(0, (4, 3)),
        )
    )

    axL.annotate(
        r"$F(\sigma)$ = least $k$ of the column",
        xy=(1.10, 2.06), xytext=(1.9, 0.75),
        fontsize=9, color=INK,
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0,
                        shrinkA=2, shrinkB=8),
    )
    axL.text(
        2.85, 6.55,
        "empty column:\n" + r"$\sigma \notin \mathrm{dom}(F)$, no pair ever appears",
        ha="center", fontsize=9, color=GRAY, linespacing=1.4,
    )

    axL.set_xlim(-0.6, 8.9)
    axL.set_ylim(0, K_MAX + 0.9)
    axL.set_xticks(range(len(WORDS)))
    axL.set_xticklabels(WORDS)
    axL.set_yticks(range(0, K_MAX + 1))
    axL.set_xlabel(r"$\sigma$ (length-lexicographic order)")
    axL.set_ylabel(r"$k$")
    axL.grid(axis="y", alpha=0.15)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)

    # ----------------------------------------------------------------- right
    axR.set_title(r"Kraft budget: masses $2^{-F(\sigma)}$ inside $[0,1]$", pad=10)
    axR.set_xlim(-0.04, 1.04)
    axR.set_ylim(0, 1)
    axR.axis("off")

    y0, h = 0.46, 0.16
    segs = [("0", 0.0, 2 ** -2), ("10", 0.25, 2 ** -4), ("001", 0.3125, 2 ** -5)]
    for w, start, width in segs:
        axR.add_patch(
            Rectangle((start, y0), width, h, facecolor=COL[w], alpha=0.85,
                      edgecolor="white", linewidth=2, zorder=3)
        )
    axR.add_patch(
        Rectangle((0.34375, y0), 1 - 0.34375, h, facecolor="#b9b7b1",
                  alpha=0.35, edgecolor="white", linewidth=2, zorder=2)
    )
    axR.text(0.67, y0 + h / 2, "unused budget", ha="center", va="center",
             fontsize=9.5, color=GRAY, zorder=4)

    # direct labels; leader lines for the two thin segments
    axR.text(0.125, y0 + h + 0.055, r"$\sigma = 0$:  $2^{-2}$",
             ha="center", fontsize=10, color=INK)
    axR.plot([0.28125, 0.24], [y0 + h + 0.005, y0 + h + 0.105],
             color=GREEN, lw=1.0, zorder=4)
    axR.text(0.24, y0 + h + 0.13, r"$10$: $2^{-4}$",
             ha="center", fontsize=10, color=INK)
    axR.plot([0.328125, 0.42], [y0 + h + 0.005, y0 + h + 0.185],
             color=ORANGE, lw=1.0, zorder=4)
    axR.text(0.42, y0 + h + 0.21, r"$001$: $2^{-5}$",
             ha="center", fontsize=10, color=INK)

    # scale under the bar
    axR.plot([0, 1], [y0 - 0.07, y0 - 0.07], color=INK, lw=1.0)
    for t, lab in [(0, "0"), (0.25, "1/4"), (0.5, "1/2"), (0.75, "3/4"), (1, "1")]:
        axR.plot([t, t], [y0 - 0.085, y0 - 0.055], color=INK, lw=1.0)
        axR.text(t, y0 - 0.135, lab, ha="center", fontsize=9, color=INK)

    axR.text(
        0.5, 0.20,
        r"$\sum_{\sigma \in \mathrm{dom}(F)} 2^{-F(\sigma)}"
        r" = \frac{1}{4} + \frac{1}{16} + \frac{1}{32}"
        r" = \frac{11}{32} \leq 1$",
        ha="center", fontsize=11, color=INK,
    )
    axR.text(
        0.5, 0.065,
        r"smaller $F(\sigma)$ $\Rightarrow$ larger mass:"
        "\nlow information content is expensive",
        ha="center", fontsize=9, color=GRAY, linespacing=1.4,
    )
    axR.text(
        0.5, 0.90,
        r"$\mathrm{dom}(F) = \{0, 10, 001\}$ is not prefix-free"
        "\n" + r"($0 \prec 001$) — and that is fine",
        ha="center", fontsize=9, color=GRAY, linespacing=1.4,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "arca_epigraph_kraft.png", tight=True)


if __name__ == "__main__":
    fig_epigraph_kraft()
