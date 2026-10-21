"""Shared inputs of the paper scripts (tables / figures from the test CSVs).

Every script writes LaTeX (and its images) to OUT_DIR/<tables|figures>/, laid
out like the overleaf project, so the files can be copied over as they are.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Code"))
sys.path.insert(0, str(REPO / "journal_results"))

import ranking  # noqa: E402

EVAL_DIR = REPO / "evaluation" / "test"
BASELINE_DIR = REPO / "journal_results" / "baselines"
OUT_DIR = Path(__file__).resolve().parent / "out"

# baseline comparison (plan, experiment 4): ID -> (label, evaluation run).
# B5 / B6: the val-chosen method (select_full_method.py: I1, no PET / rigidity
# losses) as network and + IO; B7 / B8: the full method (PET + rigidity losses
# in training and IO, T4 / I4), fixed in the plan in advance. Runs of exp. 1.
BASELINES: Dict[str, Tuple[str, str]] = {
    "B1": ("Initial", "B1"),
    "B2": ("Affine", "B2"),
    "B3": ("NiftyReg", "B3"),
    "B4": ("ConvexAdam", "B4"),
    "B5": ("Ours", "T1"),
    "B6": ("Ours + IO", "I1"),
    "B7": ("Ours (full)", "T4"),
    "B8": ("Ours (full) + IO", "I4"),
}
# not ranked: its MTV / TLG change is 0 by construction (nothing resampled)
UNRANKED = ("B1",)
# reference of the significance marks
OURS = "B8"
# our rows (a rule above B5 in the table, accent colour in the plot)
OURS_IDS = ("B5", "B6", "B7", "B8")
# precomputed fields: their registration time is in the tuning timings, the
# runtime_s of the evaluation only covers loading the field
EXTERNAL_TIMINGS = {"B3": "niftyreg", "B4": "convexadam"}


def load_pairs(method_id: str) -> pd.DataFrame:
    return pd.read_csv(EVAL_DIR / f"{BASELINES[method_id][1]}.csv", dtype={"case_id": str})


def patient_means(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """One row per patient: mean over the patient's pairs (NaNs skipped)."""
    df = df.assign(
        ndv_adjusted=(df["ndv_percent"] - ranking.NDV_EQUIVALENCE_PERCENT).clip(lower=0.0)
    )
    return df.groupby("case_id")[columns].mean()


def runtimes(method_id: str) -> Optional[pd.Series]:
    """Per-pair registration time in s (None for the identity)."""
    if method_id in UNRANKED:
        return None
    if method_id in EXTERNAL_TIMINGS:
        method = EXTERNAL_TIMINGS[method_id]
        import json

        config = json.loads((BASELINE_DIR / method / "chosen.json").read_text())["config"]
        return pd.read_csv(BASELINE_DIR / method / config / "timings_test.csv")["seconds"]
    return load_pairs(method_id)["runtime_s"]


def holm(p: Dict[str, float]) -> Dict[str, float]:
    names = list(p)
    adjusted = ranking.holm(np.array([p[n] for n in names]))
    return dict(zip(names, adjusted))
