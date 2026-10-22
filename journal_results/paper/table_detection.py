"""Experiment 3: is PET IO limited by the predicted lesion masks? -> LaTeX table.

Each ground-truth lesion of the moving scan (26-connected component, as in the
per-lesion CSVs of evaluate.py) is grouped by the fraction of its voxels that
the nnU-Net mask (the lesion mask IO sees) covers: missed (0), partly covered
(< 50 %), covered (>= 50 %). Per group and run: median per-lesion |MTV change|
and the volume-weighted |MTV change| (sum of |voxel change| / sum of voxels,
which is what dominates the pair-level metric), lesions below
evaluate.LESION_MIN_VOXELS excluded as in the per-lesion figure.

    ../../.venv/bin/python table_detection.py  ->  out/tables/detection.tex
"""

import numpy as np
import pandas as pd
from scipy import ndimage

import common
import evaluate

# run -> row label (network; IO without / with PET losses)
RUNS = {
    "T1": "Network only",
    "I1": r"+ IO, PET $\times$0",
    "I5": r"+ IO, PET $\times$0.1 (tuned)",
    "W1": r"+ IO, PET $\times$1",
}
GROUPS = (("missed", "missed"), ("partial", r"partly ($<$50\,\%)"), ("covered", r"covered ($\geq$50\,\%)"))
CACHE = common.OUT_DIR.parent / "cache" / "lesion_coverage.csv"
KEY = ["case_id", "tp_moving", "lesion_id"]


def coverage() -> pd.DataFrame:
    """Per GT lesion of each moving test scan: fraction covered by the predicted mask."""
    if CACHE.exists():
        return pd.read_csv(CACHE, dtype={"case_id": str})
    pairs = common.load_run("T1")[["case_id", "tp_moving"]].drop_duplicates()
    rows = []
    for case_id, tp in pairs.itertuples(index=False):
        tp = f"{int(tp):02d}"
        name = f"PSMARegPSMA_{case_id}_0001_{tp}.nii.gz"
        gt = evaluate.my_data.nib.load(str(evaluate.GT_LABEL_DIR / name)).get_fdata() == 1
        pred = evaluate.my_data.nib.load(str(evaluate.PRED_LABEL_DIR / name)).get_fdata() > 0
        components, n = ndimage.label(gt, structure=evaluate.LESION_STRUCTURE)
        for lesion_id in range(1, n + 1):
            mask = components == lesion_id
            rows.append({
                "case_id": case_id, "tp_moving": int(tp), "lesion_id": lesion_id,
                "coverage": float(pred[mask].mean()),
            })
    df = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def group(c: float) -> str:
    return "missed" if c == 0 else "partial" if c < 0.5 else "covered"


def main() -> None:
    cov = coverage()
    cov["group"] = cov["coverage"].map(group)
    cells, counts = {}, {}
    for run in RUNS:
        les = pd.read_csv(common.EVAL_DIR / f"{run}_lesions.csv", dtype={"case_id": str})
        les = les[les["size_bin"] != "tiny"].merge(cov, on=KEY, how="inner")
        for g, _ in GROUPS:
            sub = les[les["group"] == g]
            counts[g] = (len(sub), sub["lesion_voxels"].sum())
            weighted = (sub["mtv_bias"] * sub["lesion_voxels"]).sum() / sub["lesion_voxels"].sum()
            cells[run, g] = (100 * sub["mtv_bias"].median(), 100 * weighted)
    total_lesions = sum(n for n, _ in counts.values())
    total_voxels = sum(v for _, v in counts.values())

    header = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{name}}}" for _, name in GROUPS
    )
    sub_header = " & ".join(
        rf"{counts[g][0]} lesions, {100 * counts[g][1] / total_voxels:.0f}\,\% of volume" for g, _ in GROUPS
    )
    lines = []
    for run, name in RUNS.items():
        lines.append(name + "".join(f" & {cells[run, g][0]:.1f} & {cells[run, g][1]:.1f}" for g, _ in GROUPS) + r" \\")
        if run == "T1":
            lines.append(r"\midrule")
    body = "\n".join(lines)
    cmid = "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(GROUPS)))
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Volume change of ground-truth lesions grouped by how much of each lesion the predicted lesion mask seen by IO covers (test set, {total_lesions} lesions; network trained without PET losses, IO with the PET weights scaled by the given factor). Median: median per-lesion $|\Delta\mathrm{{MTV}}|$; weighted: sum of absolute voxel changes over sum of voxels, both in \%.}}
\label{{tab:detection}}
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{l{"cc" * len(GROUPS)}}}
\toprule
Lesions & {header} \\
 & \multicolumn{{2}}{{c}}{{\tiny {sub_header.split(" & ")[0]}}} & \multicolumn{{2}}{{c}}{{\tiny {sub_header.split(" & ")[1]}}} & \multicolumn{{2}}{{c}}{{\tiny {sub_header.split(" & ")[2]}}} \\
{cmid}
 & {" & ".join(["median", "weighted"] * len(GROUPS))} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    out_dir = common.OUT_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "detection.tex").write_text(tex)
    print(f"coverage groups: {counts}")
    print(cov.groupby("group")["coverage"].describe())
    for run, name in RUNS.items():
        print(f"{run:3s}" + "".join(f"  {g}: med {cells[run, g][0]:5.1f} w {cells[run, g][1]:5.1f}" for g, _ in GROUPS))


if __name__ == "__main__":
    main()
