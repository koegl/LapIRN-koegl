"""Experiment 2: rigidity penalties compared -> LaTeX table.

Rows: R1 (no rigidity) and R2-R4 at the weights chosen on val (rigidity.py).
Columns: alignment (CT, bone, soft-tissue Dice), rigidity measured three ways
(residual of a per-bone rigid fit, the measure the rigid-fit penalty optimizes;
deviation of det J from 1 and of J from orthonormality, the measures the
Staring penalty optimizes; distance distortion, optimized by neither), NDV in
bone, and preservation. Mean +- SD over patients; marks vs. R1 (two-sided
Wilcoxon on patient values, Holm over the three penalties per column). The
pairwise tests between the penalties (Holm over the three pairs per column),
for the text, go to out/tables/rigidity_pairwise.txt.

A transposed table with all measures (appendix; rows = measures, columns =
runs, same marks) goes to out/tables/rigidity_full.tex.

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
        "Bone rigidity",
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


# appendix table: all measures, one row each
FULL_GROUPS = (
    (
        "Alignment",
        (
            "dice_ct",
            "hd95",
            "bone_dice",
            "bone_hd95",
            "soft_dice",
            "soft_hd95",
            "near_bone_soft_dice",
        ),
    ),
    (
        "Bone rigidity",
        (
            "bre_mm",
            "staring_energy_bone",
            "staring_properness_bone",
            "staring_orthonormality_bone",
            "staring_affinity_bone",
            "dist_distortion_short",
            "dist_distortion_long",
        ),
    ),
    ("Regularity", ("ndv_percent", "ndv_bone_percent", "ndv_band_percent")),
    ("Preservation", ("mtv_bias", "tlg_bias")),
)


def marks_vs_reference(patients, columns) -> dict:
    """(run, column) -> dagger / double dagger vs. R1 (Holm over the penalties)."""
    marks = {}
    for column in columns:
        higher = MEASURES[column][1]
        if patients["R1"][column].isna().all():
            continue
        p, better = {}, {}
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
    return marks


def full_table(patients) -> str:
    columns = [c for _, cols in FULL_GROUPS for c in cols]
    marks = marks_vs_reference(patients, columns)
    rows = []
    for name, cols in FULL_GROUPS:
        rows.append(rf"\multicolumn{{{1 + len(RUNS)}}}{{l}}{{\textit{{{name}}}}} \\")
        for column in cols:
            header, _, scale, decimals = MEASURES[column]
            cells = []
            for run in RUNS:
                v = patients[run][column] * scale
                cells.append(
                    "--"
                    if v.isna().all()
                    else f"{v.mean():.{decimals}f}$\\pm${v.std():.{decimals}f}"
                    + marks.get((run, column), "")
                )
            rows.append(f"{header} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    runs = " & ".join(RUNS.values())
    n = len(patients["R1"])
    return rf"""\begin{{table}}[t]
\centering
\caption{{All measures of the bone-rigidity comparison on the test set (alignment and bone rigidity: {n} patients, 38 pairs; PET preservation: 26 patients, 30 pairs), using the network trained with the PET-preservation losses followed by instance optimization (IO) with the PET-preservation terms and one bone-rigidity penalty at the weight chosen on the validation set.
Bone rigidity: residual of a rigid fit to each bone (minimized by the rigid-fit penalty), Staring energy (minimized by the Staring penalty), and distortion of short (5--15~mm) and long ($\geq15$~mm) within-bone distances (minimized by neither).
HD95: 95th-percentile Hausdorff distance over bones and soft-tissue organs; near-bone soft tissue: soft-tissue organs within three voxels of a bone; Staring energy: sum of its properness, orthonormality, and affinity terms on the whole bone; NDV: non-diffeomorphic volume; NDV in bone band: in a three-voxel band around the bones.
Mean$\pm$SD over patients; $^{{\dagger}}$/$^{{\ddagger}}$: significantly better/worse than without bone rigidity (two-sided Wilcoxon signed-rank test on patient values, Holm-corrected over the three penalties per row, $p<0.05$).}}
\label{{tab:rigidity_full}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l{"c" * len(RUNS)}}}
\toprule
Measure & {runs} \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


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
\caption{{Bone-rigidity penalties compared ({rigidity.split_label()}, {n} patients): network trained with the PET-preservation losses, IO with the PET-preservation terms and one bone-rigidity penalty at the weight chosen on the validation set. Bone rigidity is measured in three ways: the residual of a rigid fit to each bone (the quantity the rigid-fit penalty minimizes), the Staring energy, i.e.\ the Staring penalty (properness, orthonormality, and affinity) on the whole bone (the quantity the Staring penalty minimizes), and the distortion of short (5--15~mm) and long ($\geq$15~mm) within-bone distances (minimized by neither). Mean$\pm$SD over patients. $^{{\dagger}}$/$^{{\ddagger}}$: significantly better/worse than without bone rigidity (two-sided Wilcoxon signed-rank test on patient values, Holm-corrected over the three penalties per column, $p<0.05$).}}
\label{{tab:rigidity}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l{"c" * len(columns)}}}
\toprule
 & {header_groups} \\
{cmid}
Bone-rigidity penalty & {header} \\
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
    (out_dir / "rigidity_full.tex").write_text(full_table(patients))
    common.table_preview(out_dir / "rigidity_full.tex")


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
