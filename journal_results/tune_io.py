"""Tune the shared IO settings on val (plan: "IO tuning"; todo step 7).

Network T4 (PET + rigidity) with every IO term on, all val pairs. One group of
settings per stage; the candidates of a stage are ranked with the challenge
ranking (ranking.py) and the winner is carried into the next stage:

    1. optimiser:  io_lr x io_it
    2. PET:        the four PET weights scaled together (ratios kept)

w_io_bone_rigidity is not tuned but fixed (FIXED): the ranking never rewards
rigidity (it costs Dice, NDV is always under the threshold), so a tuned weight
is 0 and the "+ rigidity" IO arms would be empty. Round 2 picked 0 (tied with
0.1); 0.25 costs 0.003 Dice. The stages above ran with the config default 1.0.

The result (io_tuning/chosen.json) is applied by evaluate.py to every IO run.
Finished candidates are reused, so a rerun resumes. A later stage's results
sit in a folder named after the earlier winners, so they are only reused when
those winners did not change (e.g. after a grid extension).

    ../.venv/bin/python tune_io.py
"""

import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

import config  # noqa: E402
import evaluate  # noqa: E402
import ranking  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "io_tuning"
NETWORK = "T4"
PET_WEIGHTS = ("w_io_mtv", "w_io_mtv_avg", "w_io_tlg", "w_io_jacobian_tumor")

_cfg = config.TrainingConfig()
STAGES: Dict[str, Dict[str, Dict]] = {
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
FIXED = {"w_io_bone_rigidity": 0.25}


def main() -> None:
    # tuning must not pick up a previous result
    evaluate.IO_TUNED_PATH = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    chosen: Dict = {"network": NETWORK, "overrides": {}, "stages": {}}
    winners = []
    for stage, candidates in STAGES.items():
        stage_dir = OUT_DIR / stage
        if winners:
            stage_dir = stage_dir / ("after_" + "__".join(winners))
        results = {}
        for name, overrides in candidates.items():
            run = evaluate.EvalRun(
                NETWORK,
                evaluate.IOSpec(
                    pet=True,
                    rigidity=True,
                    cfg_overrides={**chosen["overrides"], **overrides},
                ),
            )
            csv = evaluate.evaluate_run(
                f"{stage}_{name}", "val", device, run=run, out_dir=stage_dir
            )
            results[name] = pd.read_csv(csv)
        selection = ranking.select(results)
        chosen["overrides"].update(candidates[selection["chosen"]])
        chosen["stages"][stage] = selection
        winners.append(selection["chosen"])
        print(f"\n=== {stage}: chosen {selection['chosen']}")
        print(pd.DataFrame(selection["table"]).round(4).to_string(index=False), flush=True)

    chosen["overrides"].update(FIXED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "chosen.json").write_text(json.dumps(chosen, indent=2, default=str))
    print(f"\nfinal IO overrides: {chosen['overrides']}")


if __name__ == "__main__":
    main()
