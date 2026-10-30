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
        "Contributions of lesion-coverage groups to the evaluated metabolic tumor volume (MTV) and "
        "total lesion glycolysis (TLG) change in 26 test patients (1435 reference lesion components, 30 pairs). "
        "Components are grouped by the fraction covered by the predicted lesion mask used during instance "
        "optimization (IO). Missed denotes 0\\,% coverage, partly covered less than 50\\,%, and covered at "
        "least 50\\,%. The network is trained without PET-preservation losses. "
        "PET $\\times0$ denotes IO without PET-preservation losses, PET $\\times0.1$ the tuned weights, "
        "and PET $\\times1$ the weights used in network training. A group's contribution is the percentage "
        "image-level change obtained when only lesions in that group change, averaged over patients after "
        "averaging eligible image pairs. "
        "All denotes the evaluation metric. The contributions can sum to more than the evaluation metric "
        "because changes in different groups can cancel."
    )
    detection.write(detection.table(cells, GROUPS, caption, "tab:detection_contributions"), "detection_contributions")


if __name__ == "__main__":
    main()
