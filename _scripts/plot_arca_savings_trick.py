"""Plot a valid capital path and the smooth savings transform from ARCA 11.5."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = (
    Path(__file__).resolve().parents[1]
    / "assets/images/notes/books/algorithmic_randomness_computable_analysis"
)
BLUE, GREEN, RED, ORANGE = "#2c3e94", "#1f9d55", "#c0392b", "#e67e22"
INK, MUTED = "#222222", "#687383"


def plot_savings_trick():
    # Every next value is at most twice its predecessor. Thus this path can
    # occur on a nonnegative fair-coin martingale: assign the other child
    # capital 2*d[n] - d[n+1], and keep that off-path capital constant thereafter.
    capital = np.array(
        [1, 2, 4, 8, .5, 1, 2, 4, 8, 16, .5, 1, 2, 4, 8, 16, 32,
         .5, 1, 2, 4, 8, 16, 32, 64, .5], dtype=float
    )
    assert np.all(capital[1:] <= 2 * capital[:-1])
    steps = np.arange(len(capital))
    record = np.maximum.accumulate(capital)
    saved = np.log1p(record) - record / (1 + record)
    active = capital / (1 + record)
    transformed = saved + active
    assert np.all(np.diff(saved) >= 0)
    assert np.all((0 <= active) & (active < 1))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.fontsize": 10,
    })
    fig, (original, result) = plt.subplots(
        2, 1, figsize=(10.6, 7.8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.25], "hspace": .33}
    )
    fig.subplots_adjust(left=.10, right=.97, top=.89, bottom=.10)
    fig.suptitle("Remember the peaks; protect part of every gain", y=.975,
                 fontsize=16, weight="semibold")
    fig.text(.5, .937, "One capital path, before and after the smooth savings transform",
             ha="center", fontsize=11, color=MUTED)

    for ax in (original, result):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cbd2e0")
        ax.grid(axis="y", color="#e5e8ed", linewidth=.8)
        ax.set_axisbelow(True)
        ax.set_xlim(-.4, 25.6)
        for crash in (4, 10, 17, 25):
            ax.axvline(crash, color="#cbd2e0", linewidth=.9, linestyle=":")

    original.plot(steps, record, color=ORANGE, linestyle="--", linewidth=1.8,
                  label=r"Record $M_n=\max_{j\leq n}d_j$")
    original.plot(steps, capital, color=RED, marker="o", markersize=3.7,
                  linewidth=1.8, label=r"Original capital $d_n$")
    original.set_title("The original strategy reaches new highs, then crashes", loc="left", pad=12)
    original.set_ylabel("Original capital")
    original.set_ylim(-2, 70)
    original.set_yticks([0, 16, 32, 48, 64])
    original.legend(loc="upper left", frameon=False)
    original.annotate("64 falls back to 0.5", xy=(25, .5), xytext=(18.8, 38),
                      arrowprops={"arrowstyle": "->", "color": RED,
                                  "connectionstyle": "arc3,rad=-.16"},
                      color=RED, fontsize=10)

    result.fill_between(steps, saved, saved + 1, color=BLUE, alpha=.075)
    result.plot(steps, saved + 1, color=MUTED, linestyle=":", linewidth=1.2,
                label=r"Upper bound $s(M_n)+1$")
    result.plot(steps, transformed, color=BLUE, linewidth=2, marker="o",
                markersize=3.7, label=r"Transformed capital $\widetilde d_n$")
    result.plot(steps, saved, color=GREEN, linewidth=2.5,
                label=r"Protected savings $s(M_n)$")
    result.set_title("The transformed strategy stays within one unit of its savings", loc="left", pad=12)
    result.set_ylabel("Transformed capital / savings")
    result.set_xlabel("Number of revealed bits  n")
    result.set_ylim(0, 5.35)
    result.set_yticks([0, 1, 2, 3, 4, 5])
    result.set_xticks([0, 5, 10, 15, 20, 25])
    result.legend(loc="upper left", frameon=False)
    result.annotate("Same crash: savings stay at 3.19", xy=(25, saved[-1]),
                    xytext=(12.1, .43), color=GREEN, fontsize=10,
                    arrowprops={"arrowstyle": "->", "color": GREEN,
                                "connectionstyle": "arc3,rad=.13"})
    fig.text(.10, .028,
             "Same stages in both panels; different vertical scales. Lines connect discrete values.",
             fontsize=9.5, color=MUTED)

    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        destination = OUT / f"arca_smooth_savings.{extension}"
        fig.savefig(destination, dpi=160, facecolor="white")
        if extension == "svg":
            destination.write_text(
                "\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n"
            )
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    plot_savings_trick()
