"""Baseline comparison table (plan, experiment 4) on test -> LaTeX.

Cells: mean +- SD over patients (a patient's pairs averaged first). Rank: the
challenge's significance ranking (ranking.py) over B2-B9, with the 95 % rank
interval over N_BOOTSTRAP patient bootstrap samples in brackets; B1 is a
reference row. Marks: "Ours" (B10 = I7, chosen on val) vs. each other method,
two-sided Wilcoxon signed-rank on the patient values, Holm-adjusted over the
comparisons of a column, p < 0.05: dagger = Ours better, double dagger = Ours
worse. %NDV is tested on the
threshold-adjusted values (as in the ranking); B1 is not tested on MTV / TLG
(0 by construction). NDV is summarized over patients; "> thr." = individual
pairs with %NDV above 0.005 %; runtime per
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
            m: (r"$^{\dagger}$" if ours_better[m] else r"$^{\ddagger}$")
            if p < ALPHA
            else ""
            for m, p in adjusted.items()
        }
    return out


def main() -> None:
    columns = list(COLUMNS) + ["ndv_adjusted"]
    pairs = {m: common.load_pairs(m) for m in BASELINES}
    patients = {m: common.patient_means(df, columns) for m, df in pairs.items()}
    candidates = {m: df for m, df in pairs.items() if m not in UNRANKED}
    ranked = ranking.rank(
        {m: ranking.patient_level(df) for m, df in candidates.items()}
    )
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
\caption{{Comparison with baselines on the test set.
CT-organ Dice, HD95, and NDV include {n_patients} patients and {n_pairs} follow-up$\rightarrow$baseline pairs.
The PET-preservation metrics include {n_pet} patients and {n_pet_pairs} pairs.
Values are mean$\pm$standard deviation (SD) over patients.
HD95 denotes the 95th-percentile Hausdorff distance, MTV metabolic tumor volume, TLG total lesion glycolysis, NDV non-diffeomorphic volume, and ppm parts per million.
Rank is the descriptive challenge composite rank with its 95\,\% patient-bootstrap interval.
Initial serves as a reference and is excluded from the ranking.
The Training and instance optimization (IO) columns show where the PET-preservation and bone-rigidity losses are applied.
``PET + rig.'' denotes both loss groups together.
In the IO column, ``--'' denotes no IO and ``basic'' denotes IO with only the similarity and deformation-regularization terms.
$^{{\dagger}}$ and $^{{\ddagger}}$ indicate that our selected method (last row) is significantly better or worse, respectively, after Holm correction at $p<0.05$ (Sec.~\ref{{sect:statistics}}).
NDV tests apply the 0.005\,\% equivalence threshold, and $>$ thr. gives the percentage of individual pairs above this threshold.
Runtime is reported per pair and includes affine registration.
Bold indicates the best mean among ranked methods.}}
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
    print(
        ranked[["accuracy", "biomarker", "regularity", "final", "rank"]]
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
