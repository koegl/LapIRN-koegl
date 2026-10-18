"""Do the PET IO terms work at all? (IO tuning issue 2: more PET weight did not
lower |MTV| change on val.)

T4 + IO (PET + rigidity, optimiser of the first tuning round: lr 0.05, 20 it)
on all val pairs, PET weights scaled x0 / x1 / x4, with the lesion labels IO
sees taken from:

    pred    nnU-Net (labelsTr_pred)          -- as in the tuning
    recall  recall-biased mask (recall_masks.py)
    gt      ground truth (labelsTr)          -- upper bound; also what we score

pred x1 / x4 are the first tuning round's runs (reused). If gt x4 does not
clearly beat x0, the PET terms (or their weights) are the problem, not the
labels.

    ../.venv/bin/python io_pet_check.py
"""

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

import config  # noqa: E402
import evaluate  # noqa: E402
import ranking  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "io_pet_check"
TUNING_PET_DIR = Path(__file__).resolve().parent / "io_tuning/pet/after_lr0.05_it20/val"
NETWORK = "T4"
OPTIMISER = {"io_lr": 0.05, "io_it": 20}
PET_WEIGHTS = ("w_io_mtv", "w_io_mtv_avg", "w_io_tlg", "w_io_jacobian_tumor")
LABEL_DIRS = {"gt": evaluate.GT_LABEL_DIR, "recall": evaluate.RECALL_LABEL_DIR}
SCALES = (1.0, 4.0)
COLUMNS = ["dice_ct", "hd95", "ndv_percent", "mtv_change", "mtv_bias", "tlg_change", "tlg_bias"]


def pet_overrides(scale: float) -> dict:
    cfg = config.TrainingConfig()
    return {w: getattr(cfg, w) * scale for w in PET_WEIGHTS}


def run(name: str, scale: float, label_dir, device) -> pd.DataFrame:
    spec = evaluate.IOSpec(
        pet=True,
        rigidity=True,
        cfg_overrides={**OPTIMISER, **pet_overrides(scale)},
        pet_label_dir=label_dir,
    )
    csv = evaluate.evaluate_run(
        name, "val", device, run=evaluate.EvalRun(NETWORK, spec), out_dir=OUT_DIR
    )
    return pd.read_csv(csv)


def main() -> None:
    evaluate.IO_TUNED_PATH = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = {"x0": run("x0", 0.0, None, device)}
    for scale in SCALES:
        results[f"pred_x{scale:g}"] = pd.read_csv(TUNING_PET_DIR / f"pet_pet_x{scale:g}.csv")
    for source, label_dir in LABEL_DIRS.items():
        for scale in SCALES:
            name = f"{source}_x{scale:g}"
            results[name] = run(name, scale, label_dir, device)

    patients = {name: ranking.patient_level(df) for name, df in results.items()}
    summary = pd.DataFrame({name: df[COLUMNS].mean() for name, df in results.items()}).T
    # paired (patient-level) test against PET IO off, both directions
    for metric in ("mtv_bias", "tlg_bias"):
        summary[f"p_{metric}<x0"] = [
            ranking.one_sided_p(patients[n][metric], patients["x0"][metric], False)
            if n != "x0" else float("nan")
            for n in summary.index
        ]
    print(summary.round(4).to_string())
    summary.to_csv(OUT_DIR / "summary.csv")


if __name__ == "__main__":
    main()
