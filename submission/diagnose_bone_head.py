"""Is the bone head's dice a registration signal, or its own self-consistency?

Both bone masks come from one feature map, computed from images the network has
ALREADY aligned (the lvl3 trunk sees `warpped_x`, not `x`), and channel 1 is
pulled into the moving frame with the inverse of a field the registration
already applied. So the two masks may agree with each other by construction. If
they do, a bone-dice term in IO would reward reproducing the field the network
already produced, not improving alignment -- and any weight tuned against it
would be meaningless.

This measures that directly, with no optimisation:

  A  head masks:  dice(warp(head moving bone, field), head fixed bone)
  B  real bone:   dice(warp(true moving bone, field), true fixed bone)

Same field, same warp, same dice -- only the masks differ.

  A ~ B   the head masks behave like real anatomy; the term carries signal.
  A >> B  the channels agree far more than the anatomy does; self-consistency.

`B` is worth having on its own: it is how well bone is aligned after
registration, i.e. the headroom any bone term could possibly recover.

    python submission/diagnose_bone_head.py

Settings are the constants below -- no argparse, matching Code/inference.py.
The checkpoint must have the bone head (`use_seg_bone_head=True` at training
time); the head flags and widths are read off the state dict, so only
START_CHANNEL has to match the run by hand.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("LAPIRN_CODE", str(REPO_ROOT / "Code"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, str(REPO_ROOT / "Code"))
sys.path.insert(0, str(REPO_ROOT / "submission"))

# Code/utils.py imports mlflow and wandb at module level, but nothing on the
# scoring path calls either. A broken experiment-tracking install (mlflow pulling
# in fastapi/anyio, say) would otherwise take down metric computation that has no
# business depending on it. The real package is used when it imports cleanly.
def _stub_unimportable(*names: str) -> None:
    import importlib
    import types

    for name in names:
        try:
            importlib.import_module(name)
        except Exception:
            sys.modules[name] = types.ModuleType(name)


_stub_unimportable("mlflow", "wandb")

import infer  # noqa: E402  (the container's own pipeline, reused verbatim)
import miccai2020_model_stage  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import synthetic  # noqa: E402
import torch  # noqa: E402
import utils  # noqa: E402

# --- variables (define here, no argparse) ---
CHECKPOINT = Path(
    "/home/iml/fryderyk.koegl/data/PSMAReg/models/"
    "PSMAReg_LapIRN_hilarious-moth-39720946_stagelvl3_best_combined.pth"
)
CASES = ["0001", "0003", "0005"]
IMAGE_DIR = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTs")
LABEL_DIR = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTs")
# not inferable from the state dict, so it has to match the training run
START_CHANNEL = 24
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CT_TEMPLATE = "PSMARegPSMA_{case_id}_0000_{tp}"


def inspect_checkpoint(state: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """Head presence and width, read off the weights rather than assumed."""
    info: Dict[str, object] = {
        "use_seg_pet_head": any(k.startswith("seg_pet_head.") for k in state),
        "use_seg_bone_head": any(k.startswith("seg_bone_head.") for k in state),
    }
    for name, key in (
        ("seg_pet_head_channels", "seg_pet_head"),
        ("seg_bone_head_channels", "seg_bone_head"),
    ):
        weights = [
            v for k, v in state.items() if k.startswith(key + ".") and v.ndim == 5
        ]
        info[name] = int(weights[0].shape[0]) if weights else 32
    return info


def load_bone_labels(label_dir: Path, case_id: str, tp: str, device) -> torch.Tensor:
    """Binary bone mask from the CT segmentation, using the same label set the
    bone head was trained against."""
    path = label_dir / f"{CT_TEMPLATE.format(case_id=case_id, tp=tp)}.nii.gz"
    if not path.exists():
        path = label_dir / f"{CT_TEMPLATE.format(case_id=case_id, tp=tp)}.nii"
    arr = nib.load(str(path)).get_fdata().astype(np.int16)
    lbl = torch.from_numpy(arr)[None, None].to(device).float()
    bone = torch.tensor(synthetic.BONE_LABEL_VALUES, device=device, dtype=lbl.dtype)
    return torch.isin(lbl, bone).float()


def hard_dice(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a > 0.5, b > 0.5
    total = a.sum() + b.sum()
    if total == 0:
        return float("nan")
    return float((2.0 * (a & b).sum() / total).item())


def diagnose_case(
    case_id: str,
    model: torch.nn.Module,
    image_dir: Path,
    label_dir: Path,
    cfg,
    transform,
    transform_nearest,
    grid,
    device,
) -> Dict[str, float]:
    fixed_ct = image_dir / f"{CT_TEMPLATE.format(case_id=case_id, tp='00')}.nii.gz"
    moving_ct = image_dir / f"{CT_TEMPLATE.format(case_id=case_id, tp='01')}.nii.gz"
    fixed_pet = image_dir / f"PSMARegPSMA_{case_id}_0001_00.nii.gz"
    moving_pet = image_dir / f"PSMARegPSMA_{case_id}_0001_01.nii.gz"

    X, Y = infer.load_pair(fixed_ct, fixed_pet, moving_ct, moving_pet)
    X, Y = X.to(device), Y.to(device)

    flow_affine = infer.affine_flow(fixed_ct, moving_ct, cfg, device)
    X_affine = transform(X, flow_affine, grid)

    with torch.no_grad():
        F_X_Y, _, _, _, _, _, _ = model(X_affine, Y)
    net_flow = F_X_Y.permute(0, 2, 3, 4, 1)

    if model.seg_bone_logits is None:
        raise SystemExit("checkpoint has no bone head")

    # --- head masks, exactly as utils.seg_head_terms scores them -------------
    # channel 0 is the fixed frame; channel 1 is pulled into the moving PREREG
    # frame, i.e. the frame of X_affine -- so the field that maps it onto the
    # fixed image is the network field, not the affine-composed total.
    head_fixed = (torch.sigmoid(model.seg_bone_logits[:, 0:1]) > 0.5).float()
    head_moving = (
        utils.seg_head_moving_probs(
            model.seg_bone_logits, model.lvl2_disp_up_inv, transform, grid
        )
        > 0.5
    ).float()

    # --- true bone, brought into the same prereg frame -----------------------
    true_fixed = load_bone_labels(label_dir, case_id, "00", device)
    true_moving = load_bone_labels(label_dir, case_id, "01", device)
    true_moving_prereg = transform_nearest(true_moving, flow_affine, grid)

    warp = lambda m: transform_nearest(m, net_flow, grid)  # noqa: E731

    return {
        # the two numbers the whole script exists for
        "A_head": hard_dice(warp(head_moving), head_fixed),
        "B_true": hard_dice(warp(true_moving_prereg), true_fixed),
        # context: how much the deformable field is even doing here
        "A_identity": hard_dice(head_moving, head_fixed),
        "B_identity": hard_dice(true_moving_prereg, true_fixed),
        # context: how good the head is in each frame, vs the training logs
        "head_fixed_vs_true": hard_dice(head_fixed, true_fixed),
        "head_moving_vs_true": hard_dice(head_moving, true_moving_prereg),
    }


def main() -> None:
    device = torch.device(DEVICE)

    state = torch.load(CHECKPOINT, map_location="cpu")
    info = inspect_checkpoint(state)
    if not info["use_seg_bone_head"]:
        raise SystemExit(
            f"{CHECKPOINT.name} has no seg_bone_head.* weights -- this needs a "
            "checkpoint trained with use_seg_bone_head=True"
        )
    print(f"checkpoint: {CHECKPOINT.name}")
    print(f"  heads: {info}\n")

    cfg = infer.build_config()
    for key, value in info.items():
        setattr(cfg, key, value)
    cfg.start_channel = START_CHANNEL

    model = infer.create_model(device, cfg, CHECKPOINT)
    transform = miccai2020_model_stage.SpatialTransform_unit().to(device)
    transform_nearest = miccai2020_model_stage.SpatialTransformNearest_unit().to(device)
    grid = infer.Functions.generate_grid_unit(cfg.img_shape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    cols = [
        "A_head",
        "B_true",
        "A_identity",
        "B_identity",
        "head_fixed_vs_true",
        "head_moving_vs_true",
    ]
    header = f"{'case':>6}  " + "  ".join(f"{c:>18}" for c in cols)
    print(header)
    print("-" * len(header))

    rows: List[Dict[str, float]] = []
    for case_id in CASES:
        res = diagnose_case(
            case_id,
            model,
            IMAGE_DIR,
            LABEL_DIR,
            cfg,
            transform,
            transform_nearest,
            grid,
            device,
        )
        rows.append(res)
        print(
            f"{case_id:>6}  " + "  ".join(f"{res[c]:>18.4f}" for c in cols), flush=True
        )

    print("-" * len(header))
    means = {c: float(np.nanmean([r[c] for r in rows])) for c in cols}
    print(f"{'mean':>6}  " + "  ".join(f"{means[c]:>18.4f}" for c in cols))

    gap = means["A_head"] - means["B_true"]
    headroom = 1.0 - means["B_true"]
    print(
        f"\nA - B = {gap:+.4f}\n"
        f"  A ~ B  -> head masks behave like real anatomy; a bone-dice IO term\n"
        f"           measures genuine misalignment.\n"
        f"  A >> B -> the two channels agree far more than the bone does; the term\n"
        f"           would score the head's self-consistency, not registration.\n"
        f"\nbone left unaligned after registration: {headroom:.4f} "
        f"(1 - B). That is the most any bone term could recover."
    )


if __name__ == "__main__":
    main()
