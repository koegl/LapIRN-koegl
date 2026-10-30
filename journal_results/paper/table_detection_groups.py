"""Experiment 3: lesion volume / uptake change per coverage group -> LaTeX table.

Ground-truth lesions grouped by how much of each the predicted lesion mask (the
one IO sees) covers (detection.py). Per group, the evaluation metric computed
over the group's lesions only: how strongly the lesions of a group change.

    ../../.venv/bin/python table_detection_groups.py  ->  out/tables/detection_groups.tex
"""

import detection
from detection import GROUPS, RUNS


def main() -> None:
    groups = GROUPS[:-1]
    cells = {}
    for run in RUNS:
        les = detection.lesions(run)
        for g, _ in groups:
            sub = les[les["group"] == g]
            cells[run, g] = (
                detection.scan_metric(sub, "mtv_change", "lesion_voxels"),
                detection.scan_metric(sub, "tlg_change", "uptake"),
            )
        print(run, "  ".join(f"{g}: {cells[run, g][0]:.1f} / {cells[run, g][1]:.1f}" for g, _ in groups))
    caption = (
        "Metabolic tumor volume (MTV) and total lesion glycolysis (TLG) change within lesion-coverage groups "
        "in 26 test patients (1435 reference lesion components, 30 pairs). Components are grouped by the "
        "fraction covered by the predicted lesion mask used during instance optimization (IO). Missed denotes "
        "0\\,% coverage, partly covered less than 50\\,%, and covered at least 50\\,%. "
        "The network is trained without PET-preservation losses. PET $\\times0$ denotes IO without "
        "PET-preservation losses, PET $\\times0.1$ the tuned weights, and PET $\\times1$ the weights used "
        "in network training. Values are percentage changes in the summed volume or uptake of each group "
        "relative to the unwarped moving image, averaged over patients after averaging eligible image pairs."
    )
    detection.write(detection.table(cells, groups, caption, "tab:detection_groups"), "detection_groups")


if __name__ == "__main__":
    main()
