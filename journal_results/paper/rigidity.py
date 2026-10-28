"""Experiment 2: shared inputs of table_rigidity.py and figure_rigidity_measures.py.

R1 (no rigidity) and R2-R4 (Staring, Staring with erosion, per-bone rigid fit at
the weights chosen on val; evaluation_official_scorer/rigidity_sweep/chosen.json) on the network trained
with the PET losses, IO with the PET terms. SOURCE selects the data:
    "test":    the test runs R1-R4 (common.EVAL_DIR, official scorer)
    "val_old": the val runs at the chosen weights from the first sweep, with
               distance distortion on eroded interiors only (drafting until
               the test runs with the whole-bone metric are in)
"""

from typing import Dict

import pandas as pd

import common

SOURCE = "test"
RUNS: Dict[str, str] = {
    "R1": "No bone rigidity",
    "R2": r"Staring ($\times$3)",
    "R3": r"Staring + erosion ($\times$100)",
    "R4": r"Per-bone rigid fit ($\times$100)",
}
# run -> color (as figure_rigidity.py; R1 neutral)
COLORS = {"R1": "#0b0b0b", "R2": "#eb6834", "R3": "#1baf7a", "R4": "#2a78d6"}
_OLD = common.REPO / "journal_results" / "rigidity_sweep_interior_metric"
_VAL_OLD = {
    "R1": common.REPO / "evaluation" / "val" / "I2_interior_metric.csv",
    "R2": _OLD / "staring" / "val" / "staring_w0.659.csv",
    "R3": _OLD / "staring_eroded" / "val" / "staring_eroded_w103.csv",
    "R4": _OLD / "kabsch" / "val" / "kabsch_w25.csv",
}
# column -> (header, higher is better, scale, decimals)
MEASURES: Dict[str, tuple] = {
    "dice_ct": (r"CT Dice (\%)", True, 100.0, 1),
    "bone_dice": (r"Bone Dice (\%)", True, 100.0, 1),
    "soft_dice": (r"Soft-tissue Dice (\%)", True, 100.0, 1),
    "bre_mm": (r"Rigid-fit residual (mm)", False, 1.0, 2),
    "staring_energy_bone": (r"Staring energy", False, 1.0, 3),
    "dist_distortion_short": (r"Distortion short (\%)", False, 100.0, 2),
    "dist_distortion_long": (r"Distortion long (\%)", False, 100.0, 2),
    "ndv_bone_percent": (r"NDV in bone (ppm)", False, 1e4, 1),
    "hd95": (r"CT HD95 (mm)", False, 1.0, 2),
    "bone_hd95": (r"Bone HD95 (mm)", False, 1.0, 2),
    "soft_hd95": (r"Soft-tissue HD95 (mm)", False, 1.0, 2),
    "near_bone_soft_dice": (r"Near-bone soft-tissue Dice (\%)", True, 100.0, 1),
    "staring_properness_bone": (r"Staring properness", False, 1.0, 3),
    "staring_orthonormality_bone": (r"Staring orthonormality", False, 1.0, 3),
    "staring_affinity_bone": (r"Staring affinity", False, 1.0, 3),
    "ndv_percent": (r"NDV (ppm)", False, 1e4, 1),
    "ndv_band_percent": (r"NDV in bone band (ppm)", False, 1e4, 1),
    "mtv_bias": (r"$|\Delta\mathrm{MTV}|$ (\%)", False, 100.0, 1),
    "tlg_bias": (r"$|\Delta\mathrm{TLG}|$ (\%)", False, 100.0, 1),
}


def load(run: str) -> pd.DataFrame:
    """Per-pair CSV of a rigidity run from SOURCE."""
    if SOURCE == "test":
        return common.load_run(run)
    return pd.read_csv(_VAL_OLD[run], dtype={"case_id": str})


def patients() -> Dict[str, pd.DataFrame]:
    """Run -> patient means of all MEASURES (NaN where a measure is missing,
    e.g. staring_energy_bone in the val_old draft)."""
    out = {}
    for run in RUNS:
        df = load(run).reindex(columns=["case_id", *MEASURES])
        out[run] = df.groupby("case_id")[list(MEASURES)].mean()
    return out


def split_label() -> str:
    return "test set" if SOURCE == "test" else "validation set, draft"
