"""Per-lesion volume change by lesion size and location (experiment 1, todo
step 6) on test -> PDF + LaTeX.

Lesions: 26-connected components of the moving scan's ground-truth lesion mask
(evaluate.py *_lesions.csv); "tiny" ones (< 5 voxels, interpolation noise) are
left out. Strata: soft tissue / bone-associated (>= 50 % of the voxels on a
bone label) x small (< 0.5 ml) / medium (< 5 ml) / large.

Median per-lesion |MTV change| (upper tail in lesions_summary.txt).
Columns: network output (T1-T4) and network + matched IO (I1-I4); one marker
per loss set. Error bars: 95 % CI from a patient-cluster bootstrap (patients
resampled with all their lesions). Signed changes and the lesion counts are
printed and written to out/figures/lesions_summary.txt.

    ../../.venv/bin/python figure_lesions.py  ->  out/figures/lesions.{pdf,tex}
"""

from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import common  # noqa: E402

# loss set -> (network run, network + IO run, colour); colours as in figure_q2
CONDITIONS: Dict[str, Tuple[str, str, str]] = {
    "Neither": ("T1", "I1", "#8c8b86"),
    "PET": ("T2", "I2", "#2a78d6"),
    "Rigidity": ("T3", "I3", "#eb6834"),
    "PET + rigidity": ("T4", "I4", "#1baf7a"),
}
BLOCKS = ("Network", "Network + IO")
STRATA = [
    (False, "small"), (False, "medium"), (False, "large"),
    (True, "small"), (True, "medium"), (True, "large"),
]
STATS = {"median": 50}
N_BOOTSTRAP = 1000
SEED = 0
TEXT = "#0b0b0b"
MUTED = "#52514e"


def load_lesions(run_id: str) -> pd.DataFrame:
    df = pd.read_csv(common.EVAL_DIR / f"{run_id}_lesions.csv", dtype={"case_id": str})
    return df[df["size_bin"] != "tiny"]


def stat_ci(df: pd.DataFrame, q: float, rng) -> Tuple[float, float, float]:
    """Percentile q of |MTV change| (in %), with a patient-cluster bootstrap CI."""
    values = df["mtv_bias"].to_numpy() * 100
    groups = [g["mtv_bias"].to_numpy() * 100 for _, g in df.groupby("case_id")]
    samples = []
    for _ in range(N_BOOTSTRAP):
        picks = rng.integers(0, len(groups), len(groups))
        samples.append(np.percentile(np.concatenate([groups[i] for i in picks]), q))
    return np.percentile(values, q), np.percentile(samples, 2.5), np.percentile(samples, 97.5)


def main() -> None:
    lesions = {r: load_lesions(r) for runs in CONDITIONS.values() for r in runs[:2]}
    rng = np.random.default_rng(SEED)

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 8,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, axes = plt.subplots(len(STATS), len(BLOCKS), figsize=(7.0, 2.6), sharex=True, sharey="row", squeeze=False)
    offsets = np.linspace(-0.27, 0.27, len(CONDITIONS))
    x = np.arange(len(STRATA))
    for col, block in enumerate(BLOCKS):
        for row, (stat_name, q) in enumerate(STATS.items()):
            ax = axes[row, col]
            for offset, (name, runs) in zip(offsets, CONDITIONS.items()):
                df = lesions[runs[col]]
                est = [stat_ci(df[(df["bone_associated"] == bone) & (df["size_bin"] == size)], q, rng) for bone, size in STRATA]
                mid = np.array([e[0] for e in est])
                err = np.array([[m - e[1] for m, e in zip(mid, est)], [e[2] - m for m, e in zip(mid, est)]])
                ax.errorbar(x + offset, mid, yerr=err, fmt="o", color=runs[2], markersize=3.5, elinewidth=0.8, capsize=0)
            ax.axvline(2.5, color=MUTED, linewidth=0.5)
            ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(block, fontsize=9)
            if col == 0:
                ax.set_ylabel(rf"{stat_name} $|\Delta\mathrm{{MTV}}|$ (\%)")
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels([s for _, s in STRATA])
        for xpos, label in ((1, "soft tissue"), (4, "bone")):
            ax.text(xpos, -0.13, label, transform=ax.get_xaxis_transform(), ha="center", va="top")
    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=5, label=n) for n, (_, _, c) in CONDITIONS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()

    # numbers for the text: counts and median signed / absolute change
    lines = []
    for name, runs in CONDITIONS.items():
        for block, run in zip(BLOCKS, runs[:2]):
            df = lesions[run]
            for bone, size in STRATA:
                s = df[(df["bone_associated"] == bone) & (df["size_bin"] == size)]
                lines.append(
                    f"{run:3s} {name:15s} {'bone' if bone else 'soft':4s} {size:6s} n={len(s):4d} "
                    f"median signed MTV {100 * s['mtv_change'].median():+.1f} %  "
                    f"|MTV| {100 * s['mtv_bias'].median():.1f} / P90 {100 * s['mtv_bias'].quantile(0.9):.1f} %  "
                    f"|TLG| {100 * s['tlg_bias'].median():.1f} %"
                )

    out_dir = common.OUT_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "lesions.pdf", bbox_inches="tight", pad_inches=0)
    (out_dir / "lesions_summary.txt").write_text("\n".join(lines) + "\n")
    n_lesions = len(lesions["T1"])
    n_patients = lesions["T1"]["case_id"].nunique()
    tex = rf"""\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/lesions.pdf}}
\caption{{Per-lesion volume change on the test set ({n_lesions} lesions of {n_patients} patients), by location (soft tissue / bone-associated, $\geq$50\,\% of the lesion on bone) and size (small $<$0.5\,ml, medium $<$5\,ml, large). Median $|\Delta\mathrm{{MTV}}|$ per lesion; left: network output, right: with instance optimisation (IO) using the same losses. Error bars: 95\,\% confidence intervals from a patient-cluster bootstrap. Lesions below 5 voxels are excluded.}}
\label{{fig:lesions}}
\end{{figure}}
"""
    (out_dir / "lesions.tex").write_text(tex)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
