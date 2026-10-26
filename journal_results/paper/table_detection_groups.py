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
        "Metabolic tumor volume (MTV) and total lesion glycolysis (TLG) change of all 1435 provided "
        "lesion components in 26 test patients (30 pairs), grouped by how much of each component the "
        "predicted lesion mask used by instance optimization (IO) covers; the network is trained "
        "without PET losses and IO uses the PET weights scaled by the stated factor. Per image, the change of the "
        "summed volume or uptake of the group's lesions relative to the unwarped moving image, "
        r"in \%; mean over the patients with lesions in the group."
    )
    detection.write(detection.table(cells, groups, caption, "tab:detection_groups"), "detection_groups")


if __name__ == "__main__":
    main()
