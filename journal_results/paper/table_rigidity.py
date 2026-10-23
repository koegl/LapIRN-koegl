"""Experiment 2: rigidity penalties compared -> LaTeX table.

Rows: R1 (no rigidity) and R2-R4 at the weights chosen on val (rigidity.py).
Columns: alignment (CT, bone, soft-tissue Dice), rigidity measured three ways
(residual of a per-bone rigid fit, the measure the rigid-fit penalty optimises;
deviation of det J from 1 and of J from orthonormality, the measures the
Staring penalty optimises; distance distortion, optimised by neither), NDV in
bone, and preservation. Mean +- SD over patients; marks vs. R1 (two-sided
Wilcoxon on patient values, Holm over the three penalties per column). The
pairwise tests between the penalties (Holm over the three pairs per column),
for the text, go to out/tables/rigidity_pairwise.txt.

    ../../.venv/bin/python table_rigidity.py  ->  out/tables/rigidity.tex
"""

import itertools

import common
import rigidity
from common import two_sided
from rigidity import MEASURES, RUNS

GROUPS = (
    ("Alignment", ("dice_ct", "bone_dice", "soft_dice")),
    (
        "Rigidity of bones",
        (
            "bre_mm",
            "staring_energy_bone",
            "dist_distortion_short",
            "dist_distortion_long",
            "ndv_bone_percent",
        ),
    ),
    ("Preservation", ("mtv_bias", "tlg_bias")),
)


def main() -> None:
    patients = rigidity.patients()
    columns = [c for _, cols in GROUPS for c in cols]
    marks = {}
    for column in columns:
        higher = MEASURES[column][1]
        p, better = {}, {}
        if patients["R1"][column].isna().all():
            continue
        for run in list(RUNS)[1:]:
            p[run], better[run] = two_sided(
                patients[run][column], patients["R1"][column], higher
            )
        adjusted = common.holm(p)
        for run in p:
            if adjusted[run] < 0.05:
                marks[run, column] = (
                    r"$^{\dagger}$" if better[run] else r"$^{\ddagger}$"
                )

    def cell(run, column):
        _, _, scale, decimals = MEASURES[column]
        v = patients[run][column] * scale
        if v.isna().all():
            return "--"
        return f"{v.mean():.{decimals}f}$\\pm${v.std():.{decimals}f}" + marks.get(
            (run, column), ""
        )

    header_groups = " & ".join(
        rf"\multicolumn{{{len(cols)}}}{{c}}{{{name}}}" for name, cols in GROUPS
    )
    cmid, start = "", 2
    for _, cols in GROUPS:
        cmid += rf"\cmidrule(lr){{{start}-{start + len(cols) - 1}}}"
        start += len(cols)
    header = " & ".join(MEASURES[c][0] for c in columns)
    body = "\n".join(
        name + "".join(f" & {cell(run, c)}" for c in columns) + r" \\"
        for run, name in RUNS.items()
    )
    n = len(patients["R1"])
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Rigidity penalties compared ({rigidity.split_label()}, {n} patients): network trained with the PET losses, IO with the PET terms and one rigidity penalty at the weight chosen on the validation set. Rigidity of bones is measured in three ways: the residual of a rigid fit to each bone (the quantity the rigid-fit penalty minimises), the Staring energy, i.e.\ the Staring penalty (properness, orthonormality, and affinity) on the whole bone (the quantity the Staring penalty minimises), and the distortion of short (5--15~mm) and long ($\geq$15~mm) within-bone distances (minimised by neither). Mean$\pm$SD over patients. $^{{\dagger}}$/$^{{\ddagger}}$: significantly better/worse than without rigidity (two-sided Wilcoxon signed-rank test on patient values, Holm-corrected over the three penalties per column, $p<0.05$).}}
\label{{tab:rigidity}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l{"c" * len(columns)}}}
\toprule
 & {header_groups} \\
{cmid}
Rigidity penalty & {header} \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""
    out_dir = common.OUT_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rigidity.tex").write_text(tex)
    common.table_preview(out_dir / "rigidity.tex")
    pairwise_text = pairwise(patients, columns)
    (out_dir / "rigidity_pairwise.txt").write_text(pairwise_text + "\n")
    print(body)
    print(pairwise_text)


def pairwise(patients, columns) -> str:
    """Pairwise tests between the three penalties, for the text: mean
    difference (a - b, display units) and two-sided Wilcoxon p, Holm over the
    three pairs per column."""
    pairs = list(itertools.combinations(list(RUNS)[1:], 2))
    lines = []
    for column in columns:
        _, higher, scale, decimals = MEASURES[column]
        if patients["R1"][column].isna().all():
            continue
        p, diff = {}, {}
        for a, b in pairs:
            key = f"{a} vs {b}"
            p[key], _ = two_sided(patients[a][column], patients[b][column], higher)
            diff[key] = ((patients[a][column] - patients[b][column]) * scale).mean()
        adjusted = common.holm(p)
        for key in p:
            star = " *" if adjusted[key] < 0.05 else ""
            lines.append(
                f"{column:22s} {key}: diff {diff[key]:+.{decimals + 1}f}"
                f"  p {p[key]:.3g}  p_holm {adjusted[key]:.3g}{star}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
