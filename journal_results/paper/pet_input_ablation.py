"""PET-input ablation (2 x 2): the network trained without either loss group
(T1, used by Ours) and the network trained with both (T4), each with the real
PET input and with both PET channels set to zero (T1-ct, T4-ct; CT-only
inference, evaluation still on the real PET). Network output, no IO.

Patient-level statistics as in the manuscript: a patient's pairs are averaged
first; MTV / TLG only over lesion-positive pairs (26 patients, 30 pairs on
test). Two-sided Wilcoxon signed-rank tests on patient values:
    PET zeroed vs. real PET, per network, Holm over the 2 networks of a metric
    interaction: (T4-ct - T4) vs. (T1-ct - T1), uncorrected (single post-hoc)
Per-lesion medians (components >= 5 voxels) are descriptive.

    ../../.venv/bin/python pet_input_ablation.py [test|val]  ->  stdout
"""

import sys

import pandas as pd

import common
from common import two_sided

# network -> (real PET, PET zeroed)
NETWORKS = {"T1 (no losses)": ("T1", "T1-ct"), "T4 (PET + rig.)": ("T4", "T4-ct")}
# metric -> (higher is better, scale, unit)
METRICS = {
    "dice_ct": (True, 100.0, "%"),
    "hd95": (False, 1.0, "mm"),
    "ndv_percent": (False, 1.0, "%"),
    "bone_dice": (True, 100.0, "%"),
    "bre_mm": (False, 1.0, "mm"),
    "mtv_bias": (False, 100.0, "%"),
    "tlg_bias": (False, 100.0, "%"),
}
LESION_MIN_VOXELS = 5


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    eval_dir = common.EVAL_DIR.parent / split

    def load(run: str, suffix: str = "") -> pd.DataFrame:
        return pd.read_csv(eval_dir / f"{run}{suffix}.csv", dtype={"case_id": str})

    runs = [r for pair in NETWORKS.values() for r in pair]
    patients = {r: common.patient_means(load(r), list(METRICS)) for r in runs}
    pairs = load(runs[0])
    n_pat, n_pairs = pairs["case_id"].nunique(), len(pairs)
    lesion_pairs = pairs.dropna(subset=["mtv_bias"])
    print(
        f"{split}: {n_pat} patients / {n_pairs} pairs; MTV, TLG: "
        f"{lesion_pairs['case_id'].nunique()} patients / {len(lesion_pairs)} pairs"
    )

    for metric, (higher, scale, unit) in METRICS.items():
        print(f"\n{metric} [{unit}] (mean +- SD over patients)")
        p = {}
        for name, (real, zero) in NETWORKS.items():
            a, b = patients[zero][metric].dropna(), patients[real][metric].dropna()
            p[name] = two_sided(a, b, higher)[0]
        adjusted = common.holm(p)
        for name, (real, zero) in NETWORKS.items():
            a, b = patients[real][metric] * scale, patients[zero][metric] * scale
            diff = (b - a).dropna()
            print(
                f"  {name:16s} real {a.mean():8.4f} +- {a.std():.4f}  "
                f"zeroed {b.mean():8.4f} +- {b.std():.4f}  "
                f"diff {diff.mean():+.4f}  p_holm {adjusted[name]:.3g}  n {len(diff)}"
            )
        (r1, z1), (r4, z4) = NETWORKS.values()
        d1 = (patients[z1][metric] - patients[r1][metric]).dropna()
        d4 = (patients[z4][metric] - patients[r4][metric]).dropna()
        print(
            f"  interaction (T4 effect vs. T1 effect): p {two_sided(d4, d1, higher)[0]:.3g}"
        )

    print("\nper-lesion median |change| (>= 5 voxels)")
    for run in runs:
        lesions = load(run, "_lesions")
        lesions = lesions[lesions["lesion_voxels"] >= LESION_MIN_VOXELS]
        print(
            f"  {run:6s} MTV {lesions['mtv_bias'].median() * 100:.1f} %  "
            f"TLG {lesions['tlg_bias'].median() * 100:.1f} %  n {len(lesions)}"
        )


if __name__ == "__main__":
    main()
