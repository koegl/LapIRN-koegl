"""Baseline comparison table (plan, experiment 4) on test -> LaTeX.

Cells: mean +- SD over patients (a patient's pairs averaged first). Rank: the
challenge's significance ranking (ranking.py) over B2-B9, with the 95 % rank
interval over N_BOOTSTRAP patient bootstrap samples in brackets; B1 is a
reference row. Marks: B8 (Ours (full) + IO, our full contribution) vs. each other method, two-sided Wilcoxon signed-rank on the
patient values, Holm-adjusted over the comparisons of a column, p < 0.05:
dagger = B8 better, double dagger = B8 worse. %NDV is tested on the
threshold-adjusted values (as in the ranking); B1 is not tested on MTV / TLG
(0 by construction). "> thr." = pairs with %NDV above 0.005 %; runtime per
pair (B3 / B4: registration incl. their affine step; B2, B5-B9: our affine +
network (+ IO)), not tested.

    ../../.venv/bin/python table_baselines.py   ->  out/tables/baselines.tex
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import common
from common import BASELINES, OURS, UNRANKED, fmt, ranking, two_sided

ALPHA = 0.05
N_BOOTSTRAP = 1000
# first row of our methods: a rule above it separates them from the baselines
FIRST_OURS = "B5"
COLUMNS = common.METRIC_COLUMNS
NOT_TESTED_FOR_B1 = ("mtv_bias", "tlg_bias")


def marks(patients: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, str]]:
    """column -> method -> LaTeX mark."""
    out: Dict[str, Dict[str, str]] = {}
    for column, (_, higher, _, _, test_col) in COLUMNS.items():
        p_two, ours_better = {}, {}
        for m in patients:
            if m == OURS or (m in UNRANKED and column in NOT_TESTED_FOR_B1):
                continue
            p_two[m], ours_better[m] = two_sided(
                patients[OURS][test_col], patients[m][test_col], higher
            )
        adjusted = common.holm(p_two)
        out[column] = {
            m: (r"$^{\dagger}$" if ours_better[m] else r"$^{\ddagger}$") if p < ALPHA else ""
            for m, p in adjusted.items()
        }
    return out


def main() -> None:
    columns = list(COLUMNS) + ["ndv_adjusted"]
    pairs = {m: common.load_pairs(m) for m in BASELINES}
    patients = {m: common.patient_means(df, columns) for m, df in pairs.items()}
    candidates = {m: df for m, df in pairs.items() if m not in UNRANKED}
    ranked = ranking.rank({m: ranking.patient_level(df) for m, df in candidates.items()})
    intervals = ranking.bootstrap_ranks(candidates, n_samples=N_BOOTSTRAP)
    sig = marks(patients)

    # best displayed mean per column among the ranked methods -> bold
    best = {}
    for column, (_, higher, scale, decimals, _) in COLUMNS.items():
        shown = {
            m: fmt(patients[m][column].mean() * scale, decimals)
            for m in BASELINES
            if m not in UNRANKED
        }
        values = {m: float(s) for m, s in shown.items()}
        target = max(values.values()) if higher else min(values.values())
        best[column] = {m for m, v in values.items() if v == target}

    header = (
        ["Method", "Training", "IO", r"Rank $\downarrow$"]
        + [COLUMNS[c][0] for c in COLUMNS]
        + [r"$>$ thr. (\%)", "Runtime (s)"]
    )
    rows: List[str] = []
    for m, method in BASELINES.items():
        if m in UNRANKED:
            rank_cell = "--"
        else:
            lo, hi = (int(intervals.loc[m, k]) for k in ("rank_2.5", "rank_97.5"))
            rank_cell = f"{int(ranked.loc[m, 'rank'])} [{lo}--{hi}]"
        cells = [method.label, method.training, method.io, rank_cell]
        for column, (_, _, scale, decimals, _) in COLUMNS.items():
            values = patients[m][column].dropna() * scale
            mean = fmt(values.mean(), decimals)
            if m in best[column]:
                mean = rf"\textbf{{{mean}}}"
            cell = rf"{mean}$\pm${fmt(values.std(), decimals)}"
            cells.append(cell + sig[column].get(m, ""))
        above = (pairs[m]["ndv_percent"] > ranking.NDV_EQUIVALENCE_PERCENT).mean() * 100
        cells.append(f"{above:.0f}")
        rt = common.runtimes(m)
        cells.append("--" if rt is None else f"{rt.mean():.1f}$\\pm${rt.std():.1f}")
        if m == FIRST_OURS:
            rows.append(r"\midrule")
        rows.append(" & ".join(cells) + r" \\")

    n_patients = len(patients[OURS])
    n_pairs = len(pairs[OURS])
    n_pet = patients[OURS]["mtv_bias"].notna().sum()
    n_pet_pairs = pairs[OURS]["mtv_bias"].notna().sum()
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Comparison with baselines on the test set ({n_patients} patients, {n_pairs} follow-up$\rightarrow$baseline pairs; PET: {n_pet} patients, {n_pet_pairs} pairs). Values are mean$\pm$SD over patients (a patient's eligible pairs are averaged first). Rank: descriptive challenge significance ranking, with the 95\,\% interval of the rank over {N_BOOTSTRAP} patient bootstrap samples in brackets, over all methods except Initial, which is a reference (its MTV/TLG change is 0 by construction because nothing is resampled). Training / IO: PET and rigidity losses (PET + rig.) used in training and in instance optimisation (IO); IO ``--'': no IO, ``basic'': IO with the similarity and regularisation terms only. $^{{\dagger}}$/$^{{\ddagger}}$: Ours + IO with PET + rig. in training and IO (our full method) is significantly better/worse than this method (two-sided Wilcoxon signed-rank test on patient values, Holm-corrected over nine comparisons for CT/HD95/NDV and eight for PET within each column, $p<0.05$; NDV tested after the 0.005\,\% equivalence threshold). $>$ thr.: pairs with \%NDV above 0.005\,\%. Runtime per pair on one NVIDIA RTX A6000, including the affine step. Affine + IO: our instance optimisation directly after the affine pre-registration, without the network. Bold: best mean among ranked methods.}}
\label{{tab:baselines}}
\scriptsize
\setlength{{\tabcolsep}}{{2.5pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lll{"c" * (len(header) - 3)}}}
\toprule
{" & ".join(header)} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""
    out = common.OUT_DIR / "tables" / "baselines.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(tex)
    print(ranked[["accuracy", "biomarker", "regularity", "final", "rank"]].round(3).to_string())


if __name__ == "__main__":
    main()
