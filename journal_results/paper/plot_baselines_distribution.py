"""Patient-level distributions of the baseline comparison on test -> PDF + LaTeX.

Four panels (CT Dice, HD95, |MTV change|, |TLG change|), one violin per method
(median marked) with one dot per patient (a patient's pairs averaged first). Initial (B1) is
left out: it is far off the scale of the others and its MTV / TLG change is 0
by construction (it is in the table).

    ../../.venv/bin/python plot_baselines_distribution.py
        ->  out/figures/baselines_distribution.{pdf,tex}  (vector PDF; LaTeX fonts)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
from common import BASELINES, UNRANKED  # noqa: E402

METHODS = [m for m in BASELINES if m not in UNRANKED]
# column -> (axis label, scale)
PANELS = {
    "dice_ct": (r"CT Dice (\%) $\uparrow$", 100.0),
    "hd95": (r"HD95 (mm) $\downarrow$", 1.0),
    "mtv_bias": (r"$|\Delta\mathrm{MTV}|$ (\%) $\downarrow$", 100.0),
    "tlg_bias": (r"$|\Delta\mathrm{TLG}|$ (\%) $\downarrow$", 100.0),
}
BASELINE_COLOR = "#8c8b86"  # neutral for the baselines
OURS_COLOR = "#2a78d6"  # one accent for our two rows
TEXT = "#0b0b0b"
MUTED = "#52514e"
SEED = 0


def main() -> None:
    patients = {
        m: common.patient_means(common.load_pairs(m), list(PANELS)) for m in METHODS
    }
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 8,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.0, 2.1))
    rng = np.random.default_rng(SEED)
    labels = [BASELINES[m][0] for m in METHODS]
    for ax, (column, (ylabel, scale)) in zip(axes, PANELS.items()):
        data = [patients[m][column].dropna().to_numpy() * scale for m in METHODS]
        colors = [OURS_COLOR if m in ("B5", "B6") else BASELINE_COLOR for m in METHODS]
        parts = ax.violinplot(data, widths=0.75, showextrema=False, showmedians=True)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.3)
        parts["cmedians"].set_color(TEXT)
        parts["cmedians"].set_linewidth(1.0)
        for i, (values, color) in enumerate(zip(data, colors), start=1):
            x = i + rng.uniform(-0.15, 0.15, len(values))
            ax.scatter(x, values, s=5, color=color, linewidths=0, zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, len(METHODS) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(w_pad=1.2)

    out_dir = common.OUT_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "baselines_distribution.pdf", bbox_inches="tight")
    n = len(patients["B6"])
    tex = rf"""\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/baselines_distribution.pdf}}
\caption{{Patient-level results on the test set ({n} patients; a patient's pairs are averaged). Violins: kernel density over patients, line: median; each dot is one patient. Initial alignment is omitted (CT Dice 9.5\,\%, HD95 44.4\,mm; MTV/TLG change 0 by construction, Table~\ref{{tab:baselines}}).}}
\label{{fig:baselines_distribution}}
\end{{figure}}
"""
    (out_dir / "baselines_distribution.tex").write_text(tex)
    print(tex)


if __name__ == "__main__":
    main()
