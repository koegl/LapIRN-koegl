"""After a stage (tasks.py): pull its results from lustre, make the choice the
next stage needs (as tune_io.py / sweep_rigidity.py), and push it back.

    ../../.venv/bin/python select_stage.py pull          # results lustre -> janus
    ../../.venv/bin/python select_stage.py optimiser     # after stage 1
    ../../.venv/bin/python select_stage.py pet           # after stage 2 -> chosen.json
    ../../.venv/bin/python select_stage.py rigidity      # after stage 3 -> sweep chosen.json
    ../../.venv/bin/python select_stage.py ours          # after stage 3 -> choice of "Ours"
    ../../.venv/bin/python select_stage.py perlesion     # after stage 6 -> per-lesion weight

Each choice is written below evaluation_official_scorer/ on janus and copied to
the same place on lustre. "pull" copies CSVs and IO histories, not the claims.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "Code"))
sys.path.insert(0, str(HERE.parent))


import tasks  # noqa: E402
import worker  # noqa: E402

OUT = worker.JANUS_OUT
# selection rule (the same tolerance everywhere): IO tuning stages and the
# choice of "Ours": the candidate with the lowest |MTV change| + |TLG change|
# among those within DICE_TOLERANCE of the most accurate one; rigidity sweep:
# the largest weight within DICE_TOLERANCE of no rigidity (grid capped at x100)
DICE_TOLERANCE = 0.02
CURVE_COLUMNS = [  # sweep_rigidity.CURVE_COLUMNS
    "dice_ct",
    "hd95",
    "bone_dice",
    "soft_dice",
    "near_bone_soft_dice",
    "dist_distortion_short",
    "dist_distortion_long",
    "bre_mm",
    "staring_energy_bone",
    "ndv_percent",
    "ndv_bone_percent",
    "ndv_band_percent",
    "mtv_bias",
    "tlg_bias",
]


def pull() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude",
            "claims",
            f"{worker.HOST}:{worker.LUSTRE_OUT}/",
            f"{OUT}/",
        ],
        check=True,
    )


def push(path: Path) -> None:
    rel = path.relative_to(OUT)
    worker.remote(f"mkdir -p {worker.LUSTRE_OUT / rel.parent}")
    subprocess.run(
        ["rsync", "-a", str(path), f"{worker.HOST}:{worker.LUSTRE_OUT / rel}"],
        check=True,
    )


def select(results) -> dict:
    """The selection rule (DICE_TOLERANCE) on per-pair frames of candidates;
    returns the chosen candidate and the table of patient means."""
    table = pd.DataFrame(
        {
            name: df.groupby(df["case_id"].astype(str))[
                ["dice_ct", "hd95", "mtv_bias", "tlg_bias", "ndv_percent"]
            ]
            .mean()
            .mean()
            for name, df in results.items()
        }
    ).T
    table["preservation"] = table["mtv_bias"] + table["tlg_bias"]
    table["within_tolerance"] = (
        table["dice_ct"] >= table["dice_ct"].max() - DICE_TOLERANCE
    )
    chosen = table[table["within_tolerance"]]["preservation"].idxmin()
    table = table.sort_values("dice_ct", ascending=False)
    return {
        "chosen": chosen,
        "rule": f"lowest |MTV| + |TLG| within {100 * DICE_TOLERANCE:g} Dice points of the most accurate",
        "table": table.reset_index(names="candidate").to_dict("records"),
    }


def tuning(stage: str) -> None:
    stage_dir = OUT / "io_tuning" / stage / "val"
    results = {
        c: pd.read_csv(stage_dir / f"{stage}_{c}.csv", dtype={"case_id": str})
        for c in tasks.TUNING[stage]
    }
    selection = select(results)
    out = OUT / "io_tuning" / f"stage_{stage}.json"
    out.write_text(json.dumps(selection, indent=2, default=str))
    push(out)
    print(f"{stage}: chosen {selection['chosen']}")
    print(pd.DataFrame(selection["table"]).round(4).to_string(index=False))
    if stage == "pet":
        optimiser = json.loads((OUT / "io_tuning" / "stage_optimiser.json").read_text())
        overrides = {
            **tasks.TUNING["optimiser"][optimiser["chosen"]],
            **tasks.TUNING["pet"][selection["chosen"]],
            **tasks.FIXED,
        }
        chosen = {
            "network": tasks.NETWORK_TUNING,
            "overrides": overrides,
            "stages": {"optimiser": optimiser, "pet": selection},
        }
        path = OUT / "io_tuning" / "chosen.json"
        path.write_text(json.dumps(chosen, indent=2, default=str))
        push(path)
        print(f"final IO overrides: {overrides}")


def ours() -> None:
    """Choice of "Ours" among I1-I8 on val (select_full_method.py), with the
    selection rule instead of the challenge ranking."""
    results = {
        f"I{i}": pd.read_csv(OUT / "val" / f"I{i}.csv", dtype={"case_id": str})
        for i in range(1, 9)
    }
    selection = select(results)
    path = OUT / "full_method" / "chosen.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2, default=str))
    push(path)
    print(f'"Ours": chosen {selection["chosen"]}')
    print(pd.DataFrame(selection["table"]).round(4).to_string(index=False))


PER_LESION_CANDIDATES = {"x0": "I5", "x1": "P1", "x10": "P1x10", "x100": "P1x100"}


def per_lesion() -> None:
    """Scale of the per-lesion weights (stage 6) on val: as select(), but the
    preservation criterion is the target of these terms, the median per-lesion
    |MTV change| + |TLG change| (lesions >= 5 voxels, as in the paper)."""
    table = {}
    for scale, run in PER_LESION_CANDIDATES.items():
        pairs = pd.read_csv(OUT / "val" / f"{run}.csv", dtype={"case_id": str})
        lesions = pd.read_csv(
            OUT / "val" / f"{run}_lesions.csv", dtype={"case_id": str}
        )
        lesions = lesions[lesions["size_bin"] != "tiny"]
        means = (
            pairs.groupby("case_id")[["dice_ct", "mtv_bias", "tlg_bias"]].mean().mean()
        )
        table[scale] = {
            "run": run,
            **means.to_dict(),
            "lesion_mtv_median": lesions["mtv_bias"].median(),
            "lesion_tlg_median": lesions["tlg_bias"].median(),
        }
    table = pd.DataFrame(table).T
    for c in table.columns.drop("run"):
        table[c] = table[c].astype(float)
    table["lesion_preservation"] = (
        table["lesion_mtv_median"] + table["lesion_tlg_median"]
    )
    table["within_tolerance"] = (
        table["dice_ct"] >= table["dice_ct"].max() - DICE_TOLERANCE
    )
    chosen = table[table["within_tolerance"]]["lesion_preservation"].idxmin()
    path = OUT / "per_lesion" / "chosen.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "chosen": chosen,
                "run": table.loc[chosen, "run"],
                "rule": "lowest median per-lesion |MTV| + |TLG| within "
                f"{100 * DICE_TOLERANCE:g} Dice points of the most accurate",
                "table": table.reset_index(names="scale").to_dict("records"),
            },
            indent=2,
            default=str,
        )
    )
    push(path)
    print(f"per-lesion weight: chosen {chosen} ({table.loc[chosen, 'run']})")
    print(table.round(4).to_string())


def patient_mean(df: pd.DataFrame) -> pd.Series:
    df = df.reindex(columns=["case_id", *CURVE_COLUMNS])
    return df.groupby(df["case_id"].astype(str))[CURVE_COLUMNS].mean().mean()


def rigidity() -> None:
    reference_weights = tasks.rigidity_reference(worker.REPO)
    reference = patient_mean(pd.read_csv(OUT / "val" / "I2.csv"))
    rows = [{"run": "R1", "method": "none", "weight": 0.0, **reference}]
    chosen = {}
    for run_id, method in tasks.RIGIDITY_METHODS.items():
        candidates = []
        for index in range(tasks.N_SWEEP[method]):
            csv = tasks.result_csv(f"rig/{method}/{index}", OUT, reference_weights)
            weight = reference_weights["sweep"][method][index]
            means = patient_mean(pd.read_csv(csv))
            rows.append({"run": run_id, "method": method, "weight": weight, **means})
            candidates.append((weight, means["dice_ct"]))
        within = [
            w for w, d in candidates if d >= reference["dice_ct"] - DICE_TOLERANCE
        ]
        # largest weight within the tolerance (grid capped at x100); none: the smallest
        chosen[run_id] = {
            "method": method,
            "weight": max(within) if within else min(w for w, _ in candidates),
        }
    sweep_dir = OUT / "rigidity_sweep"
    curve = pd.DataFrame(rows)
    curve.to_csv(sweep_dir / "curve.csv", index=False)
    path = sweep_dir / "chosen.json"
    path.write_text(
        json.dumps(
            {
                "dice_tolerance": DICE_TOLERANCE,
                "reference_dice": reference["dice_ct"],
                "chosen": chosen,
            },
            indent=2,
        )
    )
    push(sweep_dir / "curve.csv")
    push(path)
    print(curve.round(4).to_string(index=False))
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "pull":
        pull()
    elif action in ("optimiser", "pet"):
        pull()
        tuning(action)
    elif action == "ours":
        pull()
        ours()
    elif action == "perlesion":
        pull()
        per_lesion()
    elif action == "rigidity":
        pull()
        rigidity()
    else:
        raise ValueError(action)
