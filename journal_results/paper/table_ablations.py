"""Q1 of experiment 1: do the PET and rigidity losses help? -> LaTeX table.

The 2x2 training conditions (neither / PET / rigidity / both), each as network
output (T1-T4) and with matched instance optimisation (I1-I4), on test.
Columns per block: CT Dice, |MTV change|, |TLG change| (mean +- SD over
patients; a patient's pairs averaged first). Within each block, each loss set
is compared with "neither": two-sided Wilcoxon signed-rank on patient values,
Holm-adjusted over the 3 comparisons of a column, p < 0.05; dagger / double
dagger = better / worse. HD95 and NDV are left out (HD95 follows Dice, NDV is
under the 0.005 % threshold for every run): one sentence in the text.

    ../../.venv/bin/python table_ablations.py  ->  out/tables/ablations.tex
"""

from typing import Dict

import common
from common import METRIC_COLUMNS, fmt, two_sided

ALPHA = 0.05
# loss set -> (network run, network + IO run); the first row is the reference
CONDITIONS: Dict[str, tuple] = {
    "Neither": ("T1", "I1"),
    "PET": ("T2", "I2"),
    "Rigidity": ("T3", "I3"),
    "PET + rigidity": ("T4", "I4"),
}
METRICS = ("dice_ct", "mtv_bias", "tlg_bias")
BLOCKS = ("Network", "Network + IO")


def main() -> None:
    reference = next(iter(CONDITIONS))
    patients = {
        run: common.patient_means(common.load_run(run), list(METRICS))
        for runs in CONDITIONS.values()
        for run in runs
    }
    cells: Dict[str, list] = {name: [] for name in CONDITIONS}
    for block in range(len(BLOCKS)):
        ref_run = CONDITIONS[reference][block]
        for metric in METRICS:
            _, higher, scale, decimals, _ = METRIC_COLUMNS[metric]
            p, better = {}, {}
            for name, runs in CONDITIONS.items():
                if name != reference:
                    p[name], better[name] = two_sided(
                        patients[runs[block]][metric], patients[ref_run][metric], higher
                    )
            adjusted = common.holm(p)
            for name, runs in CONDITIONS.items():
                values = patients[runs[block]][metric].dropna() * scale
                mark = ""
                if adjusted.get(name, 1.0) < ALPHA:
                    mark = r"$^{\dagger}$" if better[name] else r"$^{\ddagger}$"
                cells[name].append(
                    rf"{fmt(values.mean(), decimals)}$\pm${fmt(values.std(), decimals)}{mark}"
                )

    headers = [METRIC_COLUMNS[m][0] for m in METRICS]
    n_cols = len(METRICS)
    rows = [f"{name} & " + " & ".join(c) + r" \\" for name, c in cells.items()]
    n = len(patients[CONDITIONS[reference][0]])
    n_pairs = len(common.load_run(CONDITIONS[reference][0]))
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Effect of the PET and rigidity losses on the test set ({n} patients, {n_pairs} pairs). Each network is trained with the given losses and evaluated as is (left) and with instance optimisation (IO) using the same losses (right). Mean$\pm$SD over patients. $^{{\dagger}}$/$^{{\ddagger}}$: significantly better/worse than ``Neither'' in the same block (two-sided Wilcoxon signed-rank test on patient values, Holm-corrected within each column, $p<0.05$).}}
\label{{tab:ablations}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{l{"c" * (2 * n_cols)}}}
\toprule
 & \multicolumn{{{n_cols}}}{{c}}{{{BLOCKS[0]}}} & \multicolumn{{{n_cols}}}{{c}}{{{BLOCKS[1]}}} \\
\cmidrule(lr){{2-{n_cols + 1}}} \cmidrule(lr){{{n_cols + 2}-{2 * n_cols + 1}}}
Losses & {" & ".join(headers)} & {" & ".join(headers)} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    out = common.OUT_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablations.tex").write_text(tex)
    print(tex)


if __name__ == "__main__":
    main()
