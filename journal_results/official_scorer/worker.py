"""Run the tasks of one stage (tasks.py) on the cluster (forward) and on janus
(backward), coordinated by claims on lustre.

Before a task starts, its claim directory is created atomically on lustre
(mkdir); whoever cannot create it skips the task. A failed task releases its
claim, so the other side can pick it up. A task whose result CSV is on lustre
is done.

Cluster (one array task per stage task, in stage order):
    python worker.py cluster STAGE INDEX
    results -> $PSMAREG_DATA_PATH/evaluation_official_scorer/,
    fields  -> $PSMAREG_DATA_PATH/fields_official_scorer/
Janus (the short WORKSTATION_ONLY tasks first, then the stage backwards):
    ../../.venv/bin/python worker.py janus STAGE
    results -> repo evaluation_official_scorer/ (also copied to lustre),
    fields  -> written locally, copied to lustre, then deleted locally
"""

import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ON_CLUSTER = "PSMAREG_DATA_PATH" in os.environ
if ON_CLUSTER:
    # snapshot layout: <snapshot>/Code, <snapshot>/official_scorer
    sys.path.insert(0, str(HERE.parent / "Code"))
else:
    sys.path.insert(0, str(HERE.parents[1] / "Code"))

import config  # noqa: E402
import torch  # noqa: E402

LUSTRE = Path("/lustre/groups/iml/data/PET_CT_bone/test_set_for_eval")
HOST = "fryderyk.koegl@hpc-submit02.scidom.de"
REPO = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl")
LUSTRE_OUT = LUSTRE / "evaluation_official_scorer"
LUSTRE_FIELDS = LUSTRE / "fields_official_scorer"
CLAIMS = LUSTRE_OUT / "claims"
JANUS_OUT = REPO / "evaluation_official_scorer"
JANUS_FIELDS = Path("/home/iml/fryderyk.koegl/data/fields_official_scorer_tmp")

if ON_CLUSTER:
    data = Path(os.environ["PSMAREG_DATA_PATH"])
    config.TrainingConfig.split_path_journal = data / "config" / "split_journal.json"

import tasks  # noqa: E402


def remote(command: str) -> int:
    """Run a shell command on lustre (directly on the cluster, via ssh on janus)."""
    if ON_CLUSTER:
        return subprocess.run(command, shell=True).returncode
    return subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, command]).returncode


def claim_path(spec: str) -> Path:
    return CLAIMS / spec.replace("/", "__")


def claim(spec: str) -> bool:
    remote(f"mkdir -p {CLAIMS}")
    return remote(f"mkdir {claim_path(spec)} 2>/dev/null") == 0


def release(spec: str) -> None:
    remote(f"rmdir {claim_path(spec)} 2>/dev/null")


def done_on_lustre(spec: str, reference) -> bool:
    return remote(f"test -f {tasks.result_csv(spec, LUSTRE_OUT, reference)}") == 0


def push(spec: str, reference) -> None:
    """Janus: copy a finished task's results and fields to lustre, then delete
    the local fields."""
    csv = tasks.result_csv(spec, JANUS_OUT, reference)
    rel = csv.parent.relative_to(JANUS_OUT)
    name = csv.stem
    remote(f"mkdir -p {LUSTRE_OUT / rel / 'io_history'} {LUSTRE_FIELDS}")
    for path in (csv, csv.with_name(f"{name}_lesions.csv")):
        if path.exists():
            subprocess.run(
                ["rsync", "-a", str(path), f"{HOST}:{LUSTRE_OUT / rel}/"], check=True
            )
    history = csv.parent / "io_history" / name
    if history.exists():
        subprocess.run(
            ["rsync", "-a", str(history), f"{HOST}:{LUSTRE_OUT / rel / 'io_history'}/"],
            check=True,
        )
    fields = JANUS_FIELDS / rel / name
    if fields.exists():
        remote(f"mkdir -p {LUSTRE_FIELDS / rel}")
        subprocess.run(
            ["rsync", "-a", str(fields), f"{HOST}:{LUSTRE_FIELDS / rel}/"], check=True
        )
        shutil.rmtree(fields)


def execute(spec: str, out_root: Path, field_root: Path, reference, device) -> str:
    """Claim and run one task: "ran", "skipped" (done or claimed elsewhere), or
    "failed" (claim released)."""
    if done_on_lustre(spec, reference) or not claim(spec):
        print(f"skip {spec} (done or claimed)", flush=True)
        return "skipped"
    print(f"=== run {spec}", flush=True)
    try:
        tasks.run(spec, out_root, field_root, reference, device)
        if not ON_CLUSTER:
            push(spec, reference)
    except Exception:
        traceback.print_exc()
        # the result CSV of a failed run must not count as done
        csv = tasks.result_csv(spec, out_root, reference)
        if csv.exists() and not ON_CLUSTER:
            csv.rename(csv.with_suffix(".failed.csv"))
        release(spec)
        return "failed"
    return "ran"


def main() -> None:
    side, stage = sys.argv[1], int(sys.argv[2])
    device = torch.device("cuda:0")
    if side == "cluster":
        reference = tasks.rigidity_reference(LUSTRE)
        spec = tasks.STAGES[stage][int(sys.argv[3])]
        status = execute(spec, LUSTRE_OUT, LUSTRE_FIELDS, reference, device)
        sys.exit(1 if status == "failed" else 0)
    if side == "janus":
        reference = tasks.rigidity_reference(REPO)
        order = tasks.STAGE_WORKSTATION_FIRST.get(stage, []) + list(
            reversed(tasks.STAGES[stage])
        )
        for spec in order:
            execute(spec, JANUS_OUT, JANUS_FIELDS, reference, device)
        print(f"stage {stage}: janus finished its part", flush=True)
        return
    raise ValueError(f"side must be cluster or janus, got {side}")


if __name__ == "__main__":
    main()
