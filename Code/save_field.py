"""Compute and save the displacement field of one evaluation run on one pair.

Used for figures (journal_results/paper/figure_qualitative.py reads the saved
field from its cache), e.g. on the cluster, where IO runs much faster than on
a CPU. The field is the scorer-convention voxel displacement (1, 3, H, W, D),
computed exactly as in evaluate.evaluate_pair.

    python save_field.py RUN_ID CASE_ID TP_MOVING TP_FIXED OUT.pt
"""

import sys
from pathlib import Path

import torch

import evaluate
from config import TrainingConfig


def compute_field(run_id: str, case_id: str, tp_x: str, tp_y: str, device, history_csv: Path) -> torch.Tensor:
    cfg = TrainingConfig()
    cfg.model_save_dir = evaluate.MODEL_DIR
    run = evaluate.EVAL_RUNS[run_id]
    transform = evaluate.miccai2020_model_stage.SpatialTransform_unit().to(device)
    transform_nearest = evaluate.miccai2020_model_stage.SpatialTransformNearest_unit().to(device)
    grid_full = torch.from_numpy(evaluate.Functions.generate_grid_unit(cfg.img_shape)[None]).to(device).float()
    model = None
    if run.network is not None:
        model = evaluate.create_model(cfg, evaluate.model_path_for(cfg, run), device)
    unit, _, _ = evaluate.register_pair(
        model, run, cfg, case_id, tp_x, tp_y, transform, transform_nearest, grid_full, device,
        history_csv=history_csv,
    )
    voxel = evaluate.Functions.transform_unit_flow_to_flow_cuda(
        unit.permute(0, 2, 3, 4, 1).clone()
    ).permute(0, 4, 1, 2, 3)
    return voxel.flip(1).detach()  # as in evaluate.evaluate_pair


def main() -> None:
    run_id, case_id, tp_x, tp_y, out = sys.argv[1:6]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    field = compute_field(run_id, case_id, tp_x, tp_y, device, Path(out).with_suffix(".io_history.csv"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(field.cpu(), out)
    print(f"saved {out} {tuple(field.shape)}")


if __name__ == "__main__":
    main()
