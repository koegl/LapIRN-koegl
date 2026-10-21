"""Leave-one-out of the PET terms in IO (experiment 1) on test -> LaTeX table.

T1 + PET IO with all four PET terms (I5) and with one term switched off each
(L1-L4). Columns: CT Dice, |MTV change|, |TLG change| (mean +- SD over
patients). Each L run is compared with I5: two-sided Wilcoxon signed-rank on
patient values, Holm-adjusted over the 4 comparisons of a column, p < 0.05;
dagger / double dagger = better / worse than with all terms (double dagger on
|MTV| / |TLG| = the term helps).

    ../../.venv/bin/python table_loo.py  ->  out/tables/loo.tex
"""

from typing import Dict

import common
from common import METRIC_COLUMNS, fmt, two_sided

ALPHA = 0.05
REFERENCE = ("I5", "All PET terms")
# run -> label (the term that is switched off)
RUNS: Dict[str, str] = {
    "L1": r"without $\mathcal{L}_{\mathrm{MTV}}$",
    "L2": r"without $\mathcal{L}_{\mathrm{MTV\text{-}avg}}$",
    "L3": r"without $\mathcal{L}_{\mathrm{TLG}}$",
    "L4": r"without $\mathcal{L}_{\mathrm{jac}}$",
}
METRICS = ("dice_ct", "mtv_bias", "tlg_bias")


def main() -> None:
    runs = [REFERENCE[0], *RUNS]
    patients = {r: common.patient_means(common.load_run(r), list(METRICS)) for r in runs}
    marks: Dict[str, Dict[str, str]] = {m: {} for m in METRICS}
    for metric in METRICS:
        higher = METRIC_COLUMNS[metric][1]
        p, better = {}, {}
        for run in RUNS:
            p[run], better[run] = two_sided(patients[run][metric], patients[REFERENCE[0]][metric], higher)
        for run, q in common.holm(p).items():
            if q < ALPHA:
                marks[metric][run] = r"$^{\dagger}$" if better[run] else r"$^{\ddagger}$"

    rows = []
    for run, label in [REFERENCE, *RUNS.items()]:
        cells = [label]
        for metric in METRICS:
            _, _, scale, decimals, _ = METRIC_COLUMNS[metric]
            values = patients[run][metric].dropna() * scale
            cells.append(
                rf"{fmt(values.mean(), decimals)}$\pm${fmt(values.std(), decimals)}"
                + marks[metric].get(run, "")
            )
        rows.append(" & ".join(cells) + r" \\")
        if run == REFERENCE[0]:
            rows.append(r"\midrule")

    n = len(patients[REFERENCE[0]])
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Contribution of each PET term in IO (test set, {n} patients): network trained without PET losses (T1), IO with all PET terms or with one term removed. Mean$\pm$SD over patients. $^{{\dagger}}$/$^{{\ddagger}}$: significantly better/worse than with all terms (two-sided Wilcoxon signed-rank test on patient values, Holm-corrected within each column, $p<0.05$).}}
\label{{tab:loo}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{l{"c" * len(METRICS)}}}
\toprule
IO PET terms & {" & ".join(METRIC_COLUMNS[m][0] for m in METRICS)} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    out = common.OUT_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "loo.tex").write_text(tex)
    print(tex)


if __name__ == "__main__":
    main()
