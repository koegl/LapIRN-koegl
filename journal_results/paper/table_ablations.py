"""Experiment 1 (Q1 + Q2): which losses, and where they act -> LaTeX table.

All eleven variants of figure_q2.py on test, one row per loss set (none / PET /
rigidity / both) and three blocks for where the losses act: only in training
(network output, no IO; T1-T4), in training and IO (I1-I4), only in IO on the
network trained without losses (I5-I7; for "None" this is I1, shown as --).
Columns per block: CT Dice, |MTV change|, |TLG change| (mean +- SD over
patients; a patient's pairs averaged first). Tests (two-sided Wilcoxon
signed-rank on patient values, p < 0.05):
    dagger / double dagger: better / worse than "None" in the same block, Holm
        over the 3 comparisons of a column
    *: "only in IO" differs from "training and IO" with the same losses, Holm
        over the 3 loss sets of a column (Q2)
HD95 and NDV are left out (HD95 follows Dice, NDV in the text).

    ../../.venv/bin/python table_ablations.py  ->  out/tables/ablations.tex
"""

from typing import Dict, Optional, Tuple

import common
from common import METRIC_COLUMNS, fmt, two_sided

ALPHA = 0.05
# loss set -> (only in training, training and IO, only in IO); first row = reference
CONDITIONS: Dict[str, Tuple[str, str, Optional[str]]] = {
    "None": ("T1", "I1", None),
    "PET": ("T2", "I2", "I5"),
    "Rigidity": ("T3", "I3", "I6"),
    "PET + rigidity": ("T4", "I4", "I7"),
}
DISPLAY_NAMES = {
    "None": "None",
    "PET": "PET preservation",
    "Rigidity": "Bone rigidity",
    "PET + rigidity": "PET preservation + bone rigidity",
}
METRICS = ("dice_ct", "mtv_bias", "tlg_bias")
BLOCKS = (
    "Applied only in training (no IO)",
    "Applied in training and IO",
    "Applied only in IO",
)


def main() -> None:
    reference = next(iter(CONDITIONS))
    runs = {r for rs in CONDITIONS.values() for r in rs if r is not None}
    patients = {
        r: common.patient_means(common.load_run(r), list(METRICS)) for r in runs
    }
    cells: Dict[str, list] = {name: [] for name in CONDITIONS}
    for block in range(len(BLOCKS)):
        # "only in IO" for None is the network without losses with IO (I1)
        ref_run = CONDITIONS[reference][block] or CONDITIONS[reference][1]
        for metric in METRICS:
            _, higher, scale, decimals, _ = METRIC_COLUMNS[metric]
            p, better = {}, {}
            for name, rs in CONDITIONS.items():
                if name != reference:
                    p[name], better[name] = two_sided(
                        patients[rs[block]][metric], patients[ref_run][metric], higher
                    )
            adjusted = common.holm(p)
            # Q2: only in IO vs. training and IO with the same losses
            q2 = {}
            if block == 2:
                q2 = common.holm(
                    {
                        name: two_sided(
                            patients[rs[2]][metric], patients[rs[1]][metric], higher
                        )[0]
                        for name, rs in CONDITIONS.items()
                        if rs[2] is not None
                    }
                )
            for name, rs in CONDITIONS.items():
                run = rs[block]
                if run is None:
                    cells[name].append("--")
                    continue
                values = patients[run][metric].dropna() * scale
                mark = ""
                if adjusted.get(name, 1.0) < ALPHA:
                    mark = r"^{\dagger}" if better[name] else r"^{\ddagger}"
                if q2.get(name, 1.0) < ALPHA:
                    mark += r"^{*}" if not mark else r"{}^{*}"
                mark = f"${mark}$" if mark else ""
                cells[name].append(
                    rf"{fmt(values.mean(), decimals)}$\pm${fmt(values.std(), decimals)}{mark}"
                )

    headers = [METRIC_COLUMNS[m][0] for m in METRICS]
    n_cols = len(METRICS)
    rows = [
        f"{DISPLAY_NAMES[name]} & " + " & ".join(c) + r" \\"
        for name, c in cells.items()
    ]
    first = CONDITIONS[reference][0]
    n = len(patients[first])
    n_pairs = len(common.load_run(first))
    n_pet = patients[first]["mtv_bias"].notna().sum()
    n_pet_pairs = common.load_run(first)["mtv_bias"].notna().sum()
    groups = " & ".join(
        rf"\multicolumn{{{n_cols}}}{{c}}{{{block}}}" for block in BLOCKS
    )
    cmid = " ".join(
        rf"\cmidrule(lr){{{2 + i * n_cols}-{1 + (i + 1) * n_cols}}}"
        for i in range(len(BLOCKS))
    )
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Effect of the PET-preservation and bone-rigidity losses and of where they act on the test set (CT Dice: {n} patients, {n_pairs} pairs; metabolic tumor volume [MTV] and total lesion glycolysis [TLG]: {n_pet} patients, {n_pet_pairs} pairs); numerical values from Fig.~\ref{{fig:q2}}.
Left: network trained with a given PET-preservation and bone-rigidity loss configuration, without instance optimization (IO); middle: the same network followed by IO with the same configuration; right: the network trained without either loss group followed by IO with the stated configuration.
Values are mean$\pm$SD over patients after averaging each patient's eligible pairs.
$^{{\dagger}}$/$^{{\ddagger}}$: significantly better/worse than without either loss group in the same block (for the right block: the network without either group followed by IO); $^{{*}}$: significantly different from the same configuration in training and IO (two-sided Wilcoxon signed-rank tests on patient values, Holm-corrected over three comparisons within each column, $p<0.05$).}}
\label{{tab:ablations}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l{"c" * (len(BLOCKS) * n_cols)}}}
\toprule
 & {groups} \\
{cmid}
Loss configuration & {" & ".join(headers * len(BLOCKS))} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""
    out = common.OUT_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablations.tex").write_text(tex)
    common.table_preview(out / "ablations.tex")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
