"""Experiment 3: is PET IO limited by missed lesions? Shared helpers of
table_detection_groups.py and table_detection_contributions.py.

Each ground-truth lesion of the moving scan (26-connected component, as in the
per-lesion CSVs of evaluate.py) is grouped by the fraction of its voxels that
the nnU-Net mask (the lesion mask IO sees) covers: missed (0), partly covered
(< 50 %), covered (>= 50 %). Per group, the evaluation metric restricted to the
group's lesions: per scan |sum of changes| / sum of values (voxels for MTV, PET
uptake for TLG), then mean over patients with lesions in the group. All lesions
are used (also the tiny ones), so the "all" column reproduces the evaluation
metric exactly (checked in table_detection_contributions.py).
"""

from typing import Dict

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
GROUPS = (
    ("missed", "Missed"),
    ("partial", r"Partly ($<$50\,\%)"),
    ("covered", r"Covered ($\geq$50\,\%)"),
    ("all", "All lesions"),
)
CACHE = common.OUT_DIR.parent / "cache" / "lesion_coverage.csv"
KEY = ["case_id", "tp_moving", "lesion_id"]
PAIR = ["case_id", "tp_moving"]


def coverage() -> pd.DataFrame:
    """Per GT lesion of each moving test scan: fraction covered by the predicted
    mask and PET uptake (sum of the PET image over the lesion)."""
    if CACHE.exists():
        return pd.read_csv(CACHE, dtype={"case_id": str})
    cfg = evaluate.TrainingConfig()
    pairs = common.load_run("T1")[PAIR].drop_duplicates()
    rows = []
    for case_id, tp in pairs.itertuples(index=False):
        tp = f"{int(tp):02d}"
        name = f"PSMARegPSMA_{case_id}_0001_{tp}.nii.gz"
        load = lambda path: evaluate.my_data.nib.load(str(path)).get_fdata()  # noqa: E731
        gt = load(evaluate.GT_LABEL_DIR / name) == 1
        pred = load(evaluate.PRED_LABEL_DIR / name) > 0
        pet = load(cfg.data_dir / "imagesTr" / name)
        components, n = ndimage.label(gt, structure=evaluate.LESION_STRUCTURE)
        for lesion_id in range(1, n + 1):
            mask = components == lesion_id
            rows.append({
                "case_id": case_id, "tp_moving": int(tp), "lesion_id": lesion_id,
                "coverage": float(pred[mask].mean()), "uptake": float(pet[mask].sum()),
            })
    df = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def group(c: float) -> str:
    return "missed" if c == 0 else "partial" if c < 0.5 else "covered"


def scan_metric(les: pd.DataFrame, change: str, weight: str, keep=None) -> float:
    """Evaluation metric per scan, |sum of changes| / sum of values, mean over
    patients, in %. keep: only these lesions change (the others count as
    unchanged but stay in the denominator)."""
    delta = les[change] * les[weight]
    if keep is not None:
        delta = delta * keep
    d = les.assign(delta=delta)
    per_scan = d.groupby(PAIR).apply(
        lambda g: abs(g["delta"].sum()) / g[weight].sum(), include_groups=False
    )
    return 100 * per_scan.groupby(level="case_id").mean().mean()


def lesions(run: str) -> pd.DataFrame:
    """Per-lesion CSV of a run with coverage group and uptake."""
    cov = coverage()
    cov["group"] = cov["coverage"].map(group)
    les = pd.read_csv(common.EVAL_DIR / f"{run}_lesions.csv", dtype={"case_id": str})
    return les.merge(cov, on=KEY, how="inner")


def shares() -> Dict[str, str]:
    """Header text per group: number of lesions and share of the lesion volume."""
    les = lesions("T1")
    total = les["lesion_voxels"].sum()
    text = {
        g: rf"{(les['group'] == g).sum()} lesions, "
        rf"{100 * les.loc[les['group'] == g, 'lesion_voxels'].sum() / total:.0f}\,\% of volume"
        for g, _ in GROUPS[:-1]
    }
    text["all"] = rf"{len(les)} lesions"
    return text


def table(cells, groups, caption: str, label: str) -> str:
    """LaTeX table: rows RUNS, per group an MTV and a TLG column."""
    text = shares()
    header = " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}}}" for _, name in groups)
    sub_header = " & ".join(rf"\multicolumn{{2}}{{c}}{{\tiny {text[g]}}}" for g, _ in groups)
    cmid = "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(groups)))
    metrics = " & ".join([r"$|\Delta\mathrm{MTV}|$", r"$|\Delta\mathrm{TLG}|$"] * len(groups))
    lines = []
    for run, name in RUNS.items():
        lines.append(name + "".join(f" & {cells[run, g][0]:.1f} & {cells[run, g][1]:.1f}" for g, _ in groups) + r" \\")
        if run == "T1":
            lines.append(r"\midrule")
    body = "\n".join(lines)
    return rf"""\begin{{table}}[t]
\centering
\caption{{{caption}}}
\label{{{label}}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{l{"cc" * len(groups)}}}
\toprule
 & {header} \\
 & {sub_header} \\
{cmid}
 & {metrics} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def write(tex: str, name: str) -> None:
    out_dir = common.OUT_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.tex").write_text(tex)
