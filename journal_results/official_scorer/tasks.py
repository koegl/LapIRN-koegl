"""Re-evaluation with the official scorer conventions: the runs of each stage.

All results of the first evaluation (evaluation/, journal_results/io_tuning,
journal_results/rigidity_sweep*) stay untouched; everything here writes below
OUT_ROOT = evaluation_official_scorer (on janus: the repo; on the cluster:
next to the data on lustre) and saves every displacement field below
FIELD_ROOT (lustre).

A task is a spec string:
    eval/<split>/<RUN_ID>       evaluate.EVAL_RUNS[RUN_ID]
    tune/<stage>/<candidate>    IO tuning candidate on val (as tune_io.py)
    rig/<method>/<index>        rigidity weight sweep on val (as sweep_rigidity.py)
    rigtest/<RUN_ID>            R2-R4 on test at the weights chosen by the new sweep

Stages (each run forward on the cluster and backward on janus):
    1  IO tuning: learning rate x iterations
    2  IO tuning: PET weight scale (after stage 1)
    3  all test runs, I1-I8 on val (choice of "Ours", I2 = sweep reference),
       rigidity sweep (with the IO settings from stage 2)
    4  R2-R4 on test (after the sweep)
WORKSTATION_ONLY: short runs (network only, baselines whose fields live on
janus), done first by janus and never given to the cluster.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import config
import evaluate

NETWORK_TUNING = "T4"  # tune_io.NETWORK
NETWORK_RIGIDITY = "T2"  # sweep_rigidity.NETWORK
PET_WEIGHTS = ("w_io_mtv", "w_io_mtv_avg", "w_io_tlg", "w_io_jacobian_tumor")
_cfg = config.TrainingConfig()
TUNING: Dict[str, Dict[str, Dict]] = {
    "optimiser": {
        f"lr{lr:g}_it{it}": {"io_lr": lr, "io_it": it}
        for lr in (0.0125, 0.025, 0.05, 0.1)
        for it in (10, 20, 40)
    },
    "pet": {
        f"pet_x{scale:g}": {w: getattr(_cfg, w) * scale for w in PET_WEIGHTS}
        for scale in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    },
}
# not tuned (tune_io.FIXED); during tuning the config default applies
FIXED = {"w_io_bone_rigidity": 0.25}
RIGIDITY_METHODS = {"R2": "staring", "R3": "staring_eroded", "R4": "kabsch"}
# weights x0.1 ... x100 of the calibrated reference (rigidity_calibration)
N_SWEEP = {"staring": 7, "staring_eroded": 7, "kabsch": 7}

TEST_RUNS = (
    ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8"]
    + ["L1", "L2", "L3", "L4"]
    + ["W0", "W0.0001", "W0.0003", "W0.001", "W0.003", "W0.01", "W0.03", "W0.3", "W1"]
    + ["G1", "G2", "R1"]
)
WORKSTATION_ONLY = [
    f"eval/test/{r}" for r in ("T1", "T2", "T3", "T4", "T1-tum", "B1", "B2", "B3", "B4")
]
STAGES: Dict[int, List[str]] = {
    1: [f"tune/optimiser/{c}" for c in TUNING["optimiser"]],
    2: [f"tune/pet/{c}" for c in TUNING["pet"]],
    3: [f"eval/test/{r}" for r in TEST_RUNS]
    + [f"eval/val/I{i}" for i in range(1, 9)]
    + [f"rig/{m}/{k}" for m, n in N_SWEEP.items() for k in range(n)],
    4: [f"rigtest/{r}" for r in RIGIDITY_METHODS],
    5: ["eval/test/P1", "eval/test/P2"],  # per-lesion terms (outlook)
    # val sweep of the per-lesion weight (x1 = P1, x10, x100)
    6: ["eval/val/P1", "eval/val/P1x10", "eval/val/P1x100"],
}
STAGE_WORKSTATION_FIRST = {3: WORKSTATION_ONLY}


def rigidity_reference(code_root: Path) -> Dict:
    """Calibrated reference weights and sweep grid (rigidity_calibration)."""
    for path in (
        code_root / "config" / "rigidity_reference.json",  # cluster data folder
        code_root / "journal_results" / "rigidity_calibration" / "reference.json",
    ):
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError("rigidity_reference.json / reference.json")


def resolve(spec: str, out_root: Path, reference: Dict):
    """spec -> (run name, split, EvalRun, out_dir, tuned IO path or None)."""
    kind, *rest = spec.split("/")
    tuned = out_root / "io_tuning" / "chosen.json"
    if kind == "eval":
        split, run_id = rest
        return run_id, split, evaluate.EVAL_RUNS[run_id], out_root, tuned
    if kind == "tune":
        stage, candidate = rest
        overrides = dict(TUNING[stage][candidate])
        if stage == "pet":
            previous = json.loads(
                (out_root / "io_tuning" / "stage_optimiser.json").read_text()
            )
            overrides = {**TUNING["optimiser"][previous["chosen"]], **overrides}
        run = evaluate.EvalRun(
            NETWORK_TUNING,
            evaluate.IOSpec(pet=True, rigidity=True, cfg_overrides=overrides),
        )
        return f"{stage}_{candidate}", "val", run, out_root / "io_tuning" / stage, None
    if kind == "rig":
        method, index = rest
        weight = reference["sweep"][method][int(index)]
        run = evaluate.EvalRun(
            NETWORK_RIGIDITY,
            evaluate.IOSpec(
                pet=True,
                rigidity=True,
                cfg_overrides={
                    "io_rigidity_method": method,
                    "w_io_bone_rigidity": weight,
                },
            ),
        )
        out_dir = out_root / "rigidity_sweep" / method
        return f"{method}_w{weight:.3g}", "val", run, out_dir, tuned
    if kind == "rigtest":
        (run_id,) = rest
        chosen = json.loads((out_root / "rigidity_sweep" / "chosen.json").read_text())
        weight = chosen["chosen"][run_id]["weight"]
        run = evaluate.EvalRun(
            NETWORK_RIGIDITY,
            evaluate.IOSpec(
                pet=True,
                rigidity=True,
                cfg_overrides={
                    "io_rigidity_method": RIGIDITY_METHODS[run_id],
                    "w_io_bone_rigidity": weight,
                },
            ),
        )
        return run_id, "test", run, out_root, tuned
    raise ValueError(f"unknown task spec {spec}")


def result_csv(spec: str, out_root: Path, reference: Dict) -> Path:
    """Where a task's per-pair CSV goes. Needs no stage choices (JSON files
    written between stages), so it also works for paths on lustre seen from
    janus."""
    kind, *rest = spec.split("/")
    if kind == "eval":
        split, run_id = rest
        return out_root / split / f"{run_id}.csv"
    if kind == "tune":
        stage, candidate = rest
        return out_root / "io_tuning" / stage / "val" / f"{stage}_{candidate}.csv"
    if kind == "rig":
        method, index = rest
        weight = reference["sweep"][method][int(index)]
        return (
            out_root / "rigidity_sweep" / method / "val" / f"{method}_w{weight:.3g}.csv"
        )
    if kind == "rigtest":
        (run_id,) = rest
        return out_root / "test" / f"{run_id}.csv"
    raise ValueError(f"unknown task spec {spec}")


def run(
    spec: str, out_root: Path, field_root: Optional[Path], reference: Dict, device
) -> Path:
    """Evaluate one task; fields to field_root/<split>/<run name>/."""
    name, split, eval_run, out_dir, tuned = resolve(spec, out_root, reference)
    evaluate.IO_TUNED_PATH = tuned
    evaluate.FIELD_DIR = (
        None if field_root is None else field_root / out_dir.relative_to(out_root)
    )
    return evaluate.evaluate_run(name, split, device, run=eval_run, out_dir=out_dir)
