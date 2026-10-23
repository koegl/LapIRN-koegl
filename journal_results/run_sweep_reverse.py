"""Run the rigidity weight sweep on janus from the back while the cluster array
runs it from the front.

The cluster array ARRAY_JOB runs the tasks TASKS in order (rig/METHOD/INDEX, as
in Code/psmareg_eval_array.sbatch). Here they run in reverse order with the
same run definition as sweep_rigidity.py; a run whose CSV already exists
locally (finished here, or copied from the cluster by sync_rigidity_sweep.sh)
is skipped. After each run finished here, the matching cluster task is
cancelled if it is still pending, so the two meet in the middle.

    ../.venv/bin/python run_sweep_reverse.py
"""

import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Code"))

import evaluate  # noqa: E402

ARRAY_JOB = "41927826"
HOST = "fryderyk.koegl@hpc-submit02.scidom.de"
CALIBRATION = HERE / "rigidity_calibration" / "reference.json"
OUT_DIR = HERE / "rigidity_sweep"
NETWORK = "T2"  # sweep_rigidity.NETWORK
# array task index -> (method, weight index); index 0 is the val reference I2
TASKS = {
    1 + i: task
    for i, task in enumerate(
        [("staring", k) for k in range(5)]
        + [("staring_eroded", k) for k in range(7)]
        + [("kabsch", k) for k in range(7)]
    )
}


def cancel_if_pending(task: int) -> None:
    job = f"{ARRAY_JOB}_{task}"
    state = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", HOST, f"squeue -h -j {job} -o %T"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if state == "PENDING":
        subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, f"scancel {job}"])
        print(f"cancelled cluster task {job}", flush=True)
    else:
        print(f"cluster task {job}: {state or 'not queued'}", flush=True)


def main() -> None:
    device = torch.device("cuda:0")
    sweep = json.loads(CALIBRATION.read_text())["sweep"]
    for task in sorted(TASKS, reverse=True):
        method, index = TASKS[task]
        weight = sweep[method][index]
        name = f"{method}_w{weight:.3g}"
        csv = OUT_DIR / method / "val" / f"{name}.csv"
        if csv.exists():
            print(f"{name}: exists, stopping (met the cluster)", flush=True)
            break
        run = evaluate.EvalRun(
            NETWORK,
            evaluate.IOSpec(
                pet=True,
                rigidity=True,
                cfg_overrides={
                    "io_rigidity_method": method,
                    "w_io_bone_rigidity": weight,
                },
            ),
        )
        evaluate.evaluate_run(name, "val", device, run=run, out_dir=OUT_DIR / method)
        cancel_if_pending(task)


if __name__ == "__main__":
    main()
