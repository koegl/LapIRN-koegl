"""Experiment 3: accuracy-preservation trade-off of the PET IO weights -> PDF + LaTeX.

T1 + PET IO (no rigidity) with all PET IO weights scaled together
(x 0 = W0: PET terms on with weight 0, same optimization as I1 but logging the
predicted-mask values; x 0.1 = I5 = tuned operating point; W* = the others).
Three panels against the scale (equally spaced = log, x 0 set apart):
- CT Dice on the predicted organ labels (hard_dice_ct of the IO history,
  same definition as the evaluation) and on the ground-truth labels (all pairs)
- |MTV change| and |TLG change| on the predicted lesion masks that IO
  optimizes (orange; hard_mtv / hard_tlg of the IO history at the returned
  step) and on the ground-truth masks of the evaluation (blue), on the pairs
  with lesions in both masks; solid = after IO, dashed horizontal lines = the network output (T1,
  before IO), the same for every scale
Points are patient means with 95 % patient-bootstrap CIs. Missing runs are left
out. Curve only, no tests (plan).

    ../../.venv/bin/python figure_tradeoff.py  ->  out/figures/tradeoff.{pdf,tex}
"""

from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402

# weight scale -> run
RUNS = {
    0.0: "W0", 0.0001: "W0.0001", 0.0003: "W0.0003", 0.001: "W0.001", 0.003: "W0.003",
    0.01: "W0.01", 0.03: "W0.03", 0.1: "I5", 0.3: "W0.3", 1.0: "W1",
}
OPERATING_POINT = 0.1
# IO with ground-truth lesion masks instead of predicted ones (scale -> run)
GT_MASK_RUNS = {0.1: "G1", 1.0: "G2"}
GT_MASK_COLOR = "#0b0b0b"
NETWORK_RUN = "T1"
HISTORY_DIR = common.EVAL_DIR / "io_history"
KEY = ["case_id", "tp_moving"]
N_BOOTSTRAP = 2000
SEED = 0
# categorical slots 1-2 of the reference palette: mask type; Dice neutral
COLORS = {"gt": "#2a78d6", "pred": "#eb6834"}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GAP = 1.5  # x distance between x 0 and the smallest non-zero scale


def mean_ci(values: pd.Series, rng) -> Tuple[float, float, float]:
    v = values.dropna().to_numpy()
    means = rng.choice(v, size=(N_BOOTSTRAP, len(v)), replace=True).mean(axis=1)
    return v.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def history(run: str) -> Optional[pd.DataFrame]:
    """Per pair: predicted-mask MTV / TLG change at step 0 and at the returned step."""
    files = sorted((HISTORY_DIR / run).glob("*.csv"))
    if not files:
        return None
    rows = []
    for f in files:
        h = pd.read_csv(f)
        best = h.loc[h["sel_score"].idxmin()]
        case_id, tp = f.stem.split("_")
        rows.append({
            "case_id": case_id, "tp_moving": int(tp),
            "pred_mtv_0": h["hard_mtv"].iloc[0], "pred_mtv_1": best["hard_mtv"],
            "pred_tlg_0": h["hard_tlg"].iloc[0], "pred_tlg_1": best["hard_tlg"],
            "pred_dice_0": h["hard_dice_ct"].iloc[0], "pred_dice_1": best["hard_dice_ct"],
        })
    return pd.DataFrame(rows)


def load() -> Tuple[Dict[float, pd.DataFrame], Dict[float, pd.DataFrame]]:
    """Per available scale: all pairs (Dice) and pairs with lesions in both masks."""
    network = common.load_run(NETWORK_RUN)[KEY + ["dice_ct", "mtv_bias", "tlg_bias"]].rename(
        columns={"dice_ct": "gt_dice_0", "mtv_bias": "gt_mtv_0", "tlg_bias": "gt_tlg_0"}
    )
    dice, masks = {}, {}
    for scale, run in RUNS.items():
        hist = history(run)
        if not (common.EVAL_DIR / f"{run}.csv").exists() or hist is None:
            continue
        gt = common.load_run(run)[KEY + ["dice_ct", "mtv_bias", "tlg_bias"]].rename(
            columns={"dice_ct": "gt_dice_1", "mtv_bias": "gt_mtv_1", "tlg_bias": "gt_tlg_1"}
        )
        df = network.merge(gt, on=KEY).merge(hist.astype({"case_id": str}), on=KEY)
        dice[scale] = df.dropna(subset=["gt_dice_0", "gt_dice_1", "pred_dice_0", "pred_dice_1"])
        masks[scale] = df.dropna(subset=[f"{k}_{m}_{t}" for k in ("gt", "pred") for m in ("mtv", "tlg") for t in (0, 1)])
    return dice, masks


def label(scale: float) -> str:
    return "0" if scale == 0 else f"{scale:g}"


def main() -> None:
    dice, masks = load()
    nonzero = [s for s in RUNS if s > 0]
    xpos = {s: i + GAP for i, s in enumerate(nonzero)}
    xpos[0.0] = 0.0
    rng = np.random.default_rng(SEED)

    def series(frames, column):
        scales = [s for s in RUNS if s in frames]
        stats = np.array([
            mean_ci(frames[s].groupby("case_id")[column].mean() * 100, rng) for s in scales
        ])
        return [xpos[s] for s in scales], stats

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
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.1))

    def curve(ax, x, stats, color):
        """After IO: solid line with CI band."""
        mean, lo, hi = stats.T
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, linewidth=1.5, zorder=3)
        ax.scatter(x, mean, s=12, color=color, edgecolors="white", linewidths=0.6, zorder=4)
        if xpos[OPERATING_POINT] in x:
            i = x.index(xpos[OPERATING_POINT])
            ax.scatter(x[i], mean[i], s=30, color=color, edgecolors=TEXT, linewidths=1.0, zorder=5)

    def network_line(ax, value, color):
        """Before IO (network output): dashed, the same for every scale."""
        ax.axhline(value, color=color, linewidth=1.0, linestyle=(0, (4, 2)), zorder=1)

    first_dice = next(iter(dice.values()))
    for kind in ("gt", "pred"):
        before = (first_dice.groupby("case_id")[f"{kind}_dice_0"].mean() * 100).mean()
        network_line(axes[0], before, COLORS[kind])
        x, stats = series(dice, f"{kind}_dice_1")
        curve(axes[0], x, stats, COLORS[kind])
    axes[0].set_title(r"CT Dice (\%) $\uparrow$", fontsize=8, color=TEXT)

    first = next(iter(masks.values()))
    for ax, metric, title in (
        (axes[1], "mtv", r"$|\Delta\mathrm{MTV}|$ (\%) $\downarrow$"),
        (axes[2], "tlg", r"$|\Delta\mathrm{TLG}|$ (\%) $\downarrow$"),
    ):
        for kind in ("gt", "pred"):
            before = (first.groupby("case_id")[f"{kind}_{metric}_0"].mean() * 100).mean()
            network_line(ax, before, COLORS[kind])
            x, stats = series(masks, f"{kind}_{metric}_1")
            curve(ax, x, stats, COLORS[kind])
        ax.set_title(title, fontsize=8, color=TEXT)
        ax.set_ylim(bottom=0)

    # IO with ground-truth lesion masks: evaluated on the ground-truth labels,
    # Dice over all pairs, MTV / TLG over the same pairs as the curves
    mask_pairs = first[KEY]
    for scale, run in GT_MASK_RUNS.items():
        df = common.load_run(run)
        for ax, column, pairs in (
            (axes[0], "dice_ct", None), (axes[1], "mtv_bias", mask_pairs), (axes[2], "tlg_bias", mask_pairs)
        ):
            sub = df if pairs is None else df.merge(pairs, on=KEY)
            mean, lo, hi = mean_ci(sub.groupby("case_id")[column].mean() * 100, rng)
            ax.errorbar(xpos[scale], mean, yerr=[[mean - lo], [hi - mean]], fmt="D", markersize=4,
                        color=GT_MASK_COLOR, elinewidth=0.8, capsize=0, zorder=6)

    ticks = [0.0] + nonzero
    for ax in axes:
        ax.axvline(xpos[OPERATING_POINT], color="#c9c8c3", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xticks([xpos[s] for s in ticks], [label(s) for s in ticks], rotation=60, fontsize=6.5)
        ax.set_xlim(-0.5, xpos[nonzero[-1]] + 0.5)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, names = [], []
    for kind, mask in (("gt", "ground-truth labels"), ("pred", "predicted labels (seen by IO)")):
        handles += [
            plt.Line2D([], [], color=COLORS[kind], linewidth=1.0, linestyle=(0, (4, 2))),
            plt.Line2D([], [], color=COLORS[kind], linewidth=1.5, marker="o", markersize=3),
        ]
        names += [f"{mask}, network output", f"{mask}, after IO"]
    handles.append(plt.Line2D([], [], color=GT_MASK_COLOR, marker="D", markersize=4, linestyle=""))
    names.append("IO with ground-truth\nlesion masks")
    fig.legend(handles, names, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.16), handlelength=2.5, fontsize=7)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.06, 1, 1))
    fig.supxlabel(r"PET IO weight scale", fontsize=8, color=TEXT, y=0.0)

    out_dir = common.OUT_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "tradeoff.pdf", bbox_inches="tight", pad_inches=0)
    n_dice = next(iter(dice.values()))["case_id"].nunique()
    n_masks = first["case_id"].nunique()
    tex = rf"""\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/tradeoff.pdf}}
\caption{{Effect of the PET loss weight in IO (test set): network trained without PET losses (T1) followed by IO with all PET weights scaled by the given factor (0: PET losses off; 1: the PET weights of network training). Each metric on the predicted labels that IO optimizes (orange; organs: TotalSegmentator, lesions: nnU-Net) and on the ground-truth labels used for evaluation (blue). Dice over all {n_dice} patients, lesion volume and uptake change over the {n_masks} patients with lesions in both label sets. Solid: after IO; dashed: network output before IO, the same for every factor. Outlined point and dotted line: operating point tuned on the validation set. Black diamonds: IO with the ground-truth instead of the predicted lesion masks (at $\times0.1$ and $\times1$), evaluated on the ground-truth labels. Points: patient means; bands: 95\,\% patient-bootstrap confidence intervals.}}
\label{{fig:tradeoff}}
\end{{figure}}
"""
    (out_dir / "tradeoff.tex").write_text(tex)
    for s in RUNS:
        if s in masks:
            p = masks[s].groupby("case_id").mean(numeric_only=True) * 100
            d = dice[s].groupby("case_id").mean(numeric_only=True) * 100
            print(f"x{label(s):7s} dice pred {d.pred_dice_1.mean():.1f} gt {d.gt_dice_1.mean():.1f}  mtv pred {p.pred_mtv_1.mean():.1f} gt {p.gt_mtv_1.mean():.1f}"
                  f"  tlg pred {p.pred_tlg_1.mean():.1f} gt {p.gt_tlg_1.mean():.1f}")


if __name__ == "__main__":
    main()
