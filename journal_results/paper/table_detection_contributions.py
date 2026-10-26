"""Experiment 3: contribution of each coverage group to the evaluation metric -> LaTeX table.

Ground-truth lesions grouped by how much of each the predicted lesion mask (the
one IO sees) covers (detection.py). Contribution of a group: the evaluation
metric when only the group's lesions change and all other lesions keep their
volume / uptake (they stay in the denominator). "All" is the evaluation metric
itself (checked against the evaluation CSVs). Because growth and shrinkage of
lesions in a scan partly cancel, the contributions add up to slightly more.

    ../../.venv/bin/python table_detection_contributions.py  ->  out/tables/detection_contributions.tex
"""

import common
import detection
from detection import GROUPS, RUNS


def main() -> None:
    cells = {}
    for run in RUNS:
        les = detection.lesions(run)
        for g, _ in GROUPS:
            keep = None if g == "all" else (les["group"] == g)
            cells[run, g] = (
                detection.scan_metric(les, "mtv_change", "lesion_voxels", keep),
                detection.scan_metric(les, "tlg_change", "uptake", keep),
            )
        pair = common.load_run(run)
        pair = pair[pair["n_lesions"] > 0].groupby("case_id")[["mtv_bias", "tlg_bias"]].mean().mean() * 100
        assert abs(cells[run, "all"][0] - pair["mtv_bias"]) < 0.01, run  # float32 sums in evaluate
        assert abs(cells[run, "all"][1] - pair["tlg_bias"]) < 0.01, run
        print(run, "  ".join(f"{g}: {cells[run, g][0]:.1f} / {cells[run, g][1]:.1f}" for g, _ in GROUPS))
    caption = (
        "Contribution of all 1435 provided lesion components in 26 test patients (30 pairs), grouped by "
        "how much of each component the predicted lesion mask used by instance optimization (IO) covers, "
        "to the evaluated metabolic tumor volume (MTV) and total lesion glycolysis (TLG) change; the network "
        "is trained without PET losses and IO uses the PET weights scaled by the stated factor. "
        "Contribution of a group: the change per scan if only the group's lesions changed "
        r"and all other lesions kept their volume and uptake, in \%; mean over patients. "
        "All: the evaluation metric. Since changes of different lesions in a scan partly cancel, "
        "the contributions add up to slightly more."
    )
    detection.write(detection.table(cells, GROUPS, caption, "tab:detection_contributions"), "detection_contributions")


if __name__ == "__main__":
    main()
