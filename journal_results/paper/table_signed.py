"""Supplement: signed MTV / TLG change of all compared methods -> LaTeX table.

The baseline table reports |MTV change| and |TLG change|; this table gives the
signed changes (positive: the warped lesions are larger / take up more), so
that systematic growth or shrinkage is visible. Same methods and rows as
table_baselines.py (common.BASELINES; Initial is 0 by construction and left
out). Mean +- SD and median over patients with lesions (test set).

    ../../.venv/bin/python table_signed.py  ->  out/tables/signed.tex
"""

import common
from common import BASELINES, UNRANKED

COLUMNS = (
    ("mtv_change", r"$\Delta\mathrm{MTV}$ (\%)"),
    ("tlg_change", r"$\Delta\mathrm{TLG}$ (\%)"),
)


def main() -> None:
    lines = []
    n_patients = set()
    for method_id, method in BASELINES.items():
        if method_id in UNRANKED:
            continue
        if method_id == "B5":  # first of ours, as in the baseline table
            lines.append(r"\midrule")
        df = common.load_pairs(method_id)
        df = df[df["n_lesions"] > 0]
        patients = df.groupby("case_id")[[c for c, _ in COLUMNS]].mean() * 100
        n_patients |= set(patients.index)
        cells = "".join(
            f" & ${patients[c].mean():+.1f}\\pm{patients[c].std():.1f}$"
            f" & ${patients[c].median():+.1f}$"
            for c, _ in COLUMNS
        )
        lines.append(f"{method.label} & {method.training} & {method.io}{cells} \\\\")
    body = "\n".join(lines)
    header = " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}}}" for _, name in COLUMNS)
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Signed change in metabolic tumor volume (MTV) and total lesion glycolysis (TLG) caused by registration on the test set ({len(n_patients)} patients, 30 lesion-positive pairs): positive values mean that the warped lesions are larger or have greater total uptake.
Methods are as in Table~\ref{{tab:baselines}}; Training / IO indicates where losses are used, and PET + rig. denotes PET-preservation and bone-rigidity losses.
Values are mean$\pm$SD and median over patients after averaging each patient's eligible pairs.}}
\label{{tab:signed}}
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lllcccc}}
\toprule
 & & & {header} \\
\cmidrule(lr){{4-5}}\cmidrule(lr){{6-7}}
Method & Training & IO & Mean$\pm$SD & Median & Mean$\pm$SD & Median \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""
    out_dir = common.OUT_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "signed.tex").write_text(tex)
    common.table_preview(out_dir / "signed.tex")
    print(body)


if __name__ == "__main__":
    main()
