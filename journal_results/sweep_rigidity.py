"""Rigidity weight sweeps on val (plan, experiment 2; todo step 9).

IO on the PET-only network (T2) with the PET terms and one rigidity method:
R2 Staring, R3 Staring + erosion, R4 Kabsch. Each method's weights are its
calibrated reference weight x {0.1, 0.3, 1, 3, 10}
(rigidity_calibration/reference.json). Reference R1 = T2 + PET IO without
rigidity (= I2 on val, evaluation/val/I2.csv).

Chosen weight per method: the largest one whose mean patient-level CT Dice is
within DICE_TOLERANCE of R1. Written to rigidity_sweep/chosen.json together
with the curve (Dice, bone metrics, NDV per weight) for the figure.
Finished runs are reused, so a rerun resumes.

    ../.venv/bin/python sweep_rigidity.py  ->  rigidity_sweep/
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

import evaluate  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "rigidity_sweep"
CALIBRATION = HERE / "rigidity_calibration" / "reference.json"
NETWORK = "T2"
REFERENCE_RUN = "I2"  # R1 on val
METHODS = {"R2": "staring", "R3": "staring_eroded", "R4": "kabsch"}
DICE_TOLERANCE = 0.01  # 1 Dice point
CURVE_COLUMNS = [
    "dice_ct", "hd95", "bone_dice", "soft_dice", "dist_distortion_short",
    "dist_distortion_long", "ndv_bone_percent", "ndv_band_percent",
    "ndv_percent", "mtv_bias", "tlg_bias",
]


def patient_mean(df: pd.DataFrame) -> pd.Series:
    """Mean over patients (a patient's pairs averaged first)."""
    return df.groupby(df["case_id"].astype(str))[CURVE_COLUMNS].mean().mean()


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sweep = json.loads(CALIBRATION.read_text())["sweep"]
    reference = patient_mean(pd.read_csv(evaluate.OUT_DIR / "val" / f"{REFERENCE_RUN}.csv"))
    rows = [{"run": "R1", "method": "none", "weight": 0.0, **reference}]
    chosen = {}
    for run_id, method in METHODS.items():
        candidates = []
        for weight in sweep[method]:
            run = evaluate.EvalRun(
                NETWORK,
                evaluate.IOSpec(
                    pet=True,
                    rigidity=True,
                    cfg_overrides={"io_rigidity_method": method, "w_io_bone_rigidity": weight},
                ),
            )
            csv = evaluate.evaluate_run(
                f"{method}_w{weight:.3g}", "val", device, run=run, out_dir=OUT_DIR / method
            )
            means = patient_mean(pd.read_csv(csv))
            rows.append({"run": run_id, "method": method, "weight": weight, **means})
            candidates.append((weight, means["dice_ct"]))
        within = [w for w, d in candidates if d >= reference["dice_ct"] - DICE_TOLERANCE]
        # no weight within the tolerance: the smallest one
        chosen[run_id] = {"method": method, "weight": max(within) if within else min(sweep[method])}

    curve = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curve.to_csv(OUT_DIR / "curve.csv", index=False)
    (OUT_DIR / "chosen.json").write_text(
        json.dumps({"dice_tolerance": DICE_TOLERANCE, "reference_dice": reference["dice_ct"], "chosen": chosen}, indent=2)
    )
    print(curve.round(4).to_string(index=False))
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()
