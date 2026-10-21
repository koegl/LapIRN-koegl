"""Choose "Ours + IO" for the baseline comparison on val (not test).

Candidates: the IO runs I1-I7 (network + IO) and I8 (affine + IO, no
network), all with the frozen IO settings, ranked on val with the challenge
ranking (ranking.py, the tuning rule). The winner is used as B6 on test.

    ../.venv/bin/python select_full_method.py  ->  full_method/chosen.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

import evaluate  # noqa: E402
import ranking  # noqa: E402

CANDIDATES = ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8")
OUT_DIR = Path(__file__).resolve().parent / "full_method"


def main() -> None:
    results = {
        run_id: pd.read_csv(evaluate.OUT_DIR / "val" / f"{run_id}.csv", dtype={"case_id": str})
        for run_id in CANDIDATES
    }
    selection = ranking.select(results)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "chosen.json").write_text(json.dumps(selection, indent=2, default=str))
    print(pd.DataFrame(selection["table"]).round(4).to_string(index=False))
    print(f"chosen: {selection['chosen']}")


if __name__ == "__main__":
    main()
