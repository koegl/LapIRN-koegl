"""Reference weights of the IO rigidity methods (plan, experiment 2; todo step 8).

The three terms have different units, so their weights are not comparable. On
N_PAIRS val pairs, each term is evaluated (unweighted) on the starting field of
IO in experiment 2 -- the T2 network output, composed with the affine prereg --
with the bones from the predicted fixed labels, exactly as IO computes it. The
reference weight of a method is the one at which its initial loss equals
Kabsch's at the IO setting:

    w_ref(m) = KABSCH_WEIGHT * median over pairs of L_kabsch / L_m

The val sweep (step 9) then uses w_ref(m) x {0.1, 0.3, 1, 3, 10}. Only a
forward pass per pair (no IO), so it runs next to an IO job.

    ../.venv/bin/python calibrate_rigidity.py  ->  rigidity_calibration/
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

import evaluate  # noqa: E402
import jacobian  # noqa: E402
import synthetic  # noqa: E402
import utils  # noqa: E402
from config import TrainingConfig  # noqa: E402

NETWORK = "T2"
N_PAIRS = 5
KABSCH_WEIGHT = 0.25  # IO setting (journal_results/tune_io.py, FIXED)
METHODS = ("kabsch", "staring", "staring_eroded")
SWEEP_FACTORS = (0.1, 0.3, 1.0, 3.0, 10.0)
OUT_DIR = Path(__file__).resolve().parent / "rigidity_calibration"


def rigidity_losses(total_unit_flow, y_lbl_ct, cfg, bone_values):
    """Unweighted rigidity term of each method, as in instance_opt.io_objective."""
    disp_voxel = jacobian.unit_flow_to_voxel_flow(total_unit_flow.permute(0, 2, 3, 4, 1))
    jac_det, jac = jacobian.jacobian_matrix(disp_voxel)
    out = {}
    out["kabsch"], _ = utils.per_label_rigid_loss(
        disp_voxel, y_lbl_ct, bone_values, min_voxels=cfg.rigidity_min_voxels
    )
    bones = utils.bone_label_map(y_lbl_ct, bone_values)
    for method in ("staring", "staring_eroded"):
        out[method] = utils.staring_rigidity_loss(
            disp_voxel,
            jac_det,
            jac,
            bones,
            erode=method == "staring_eroded",
            w_det=cfg.w_rig_det,
            w_ortho=cfg.w_rig_ortho,
            w_affine=cfg.w_rig_affine,
        )
    return {m: float(v) for m, v in out.items()}


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = TrainingConfig()
    cfg.model_save_dir = evaluate.MODEL_DIR
    run = evaluate.EvalRun(NETWORK)
    model = evaluate.create_model(cfg, evaluate.model_path_for(cfg, run), device)
    transform = evaluate.miccai2020_model_stage.SpatialTransform_unit().to(device)
    transform_nearest = evaluate.miccai2020_model_stage.SpatialTransformNearest_unit().to(device)
    grid_full = torch.from_numpy(evaluate.Functions.generate_grid_unit(cfg.img_shape)[None]).to(device).float()
    bone_values = torch.tensor(synthetic.BONE_LABEL_VALUES, dtype=torch.float32, device=device)

    pairs = evaluate.split_pairs(cfg, "val")
    pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, N_PAIRS).round().astype(int)]
    rows = []
    for case_id, tp_x, tp_y in pairs:
        total_unit_flow, _, _ = evaluate.register_pair(
            model, run, cfg, case_id, tp_x, tp_y, transform, transform_nearest,
            grid_full, device, history_csv=OUT_DIR / "unused.csv",
        )
        y_lbl_ct = evaluate.load_label(evaluate.PRED_LABEL_DIR, case_id, "0000", tp_y, device)
        losses = rigidity_losses(total_unit_flow, y_lbl_ct, cfg, bone_values)
        rows.append({"case_id": case_id, "tp_moving": tp_x, "tp_fixed": tp_y, **losses})
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    reference = {
        m: KABSCH_WEIGHT * float((df["kabsch"] / df[m]).median()) for m in METHODS
    }
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "per_pair.csv", index=False)
    result = {
        "network": NETWORK,
        "kabsch_weight": KABSCH_WEIGHT,
        "reference_weight": reference,
        "sweep": {m: [w * f for f in SWEEP_FACTORS] for m, w in reference.items()},
    }
    (OUT_DIR / "reference.json").write_text(json.dumps(result, indent=2))
    print(df.round(5).to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
