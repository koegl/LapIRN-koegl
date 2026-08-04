"""Replicates the official Learn2Reg 2026 Task 1 (PSMAReg) test-phase scoring.

For each metric, every pair of methods is compared with a one-sided Wilcoxon
signed-rank test on the subject-level values. P-values are Holm-adjusted within
each metric, and a method's significance score is the fraction of other methods
it significantly outperforms. The component scores are combined as

    final = 100 * (0.50 * accuracy + 0.45 * pet_biomarker + 0.05 * regularity)

NOTE: the official accuracy score averages the significance scores of CT DSC,
PET DSC, CT HD95 and PET HD95. Only the CT metrics are evaluated locally, so
the accuracy score here is the mean over CT DSC and CT HD95 only.
"""

from itertools import permutations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

CSV_DIR = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs")

# metric -> (csv file, per-subject column prefix, higher_is_better)
METRICS: Dict[str, Tuple[str, str, bool]] = {
    "ct_dsc": ("results_official_val_dice.csv", "dice_", True),
    "ct_hd95": ("results_official_val_hd95.csv", "hd95_", False),
    "mtv": ("results_official_val_mtv.csv", "mtv_", False),
    "tlg": ("results_official_val_tlg.csv", "tlg_", False),
    "ndv": ("results_official_val_ndv.csv", "ndv_", False),
}

ACCURACY_METRICS = ["ct_dsc", "ct_hd95"]  # + pet_dsc, pet_hd95 officially
PET_BIOMARKER_METRICS = ["mtv", "tlg"]
REGULARITY_METRICS = ["ndv"]

ALPHA = 0.05
NDV_EQUIVALENCE_THRESHOLD = 0.005  # percent
# rank stability is a secondary analysis and does not change the leaderboard.
# the official protocol uses 1000 samples, which takes ~3 min here.
N_BOOTSTRAP = 0


def load_metric(csv_name: str, prefix: str) -> pd.DataFrame:
    """Subject-level values as a (method x subject) frame."""
    df = pd.read_csv(CSV_DIR / csv_name).set_index("model")
    cols = [c for c in df.columns if c.startswith(prefix)]
    df = df[cols]
    df.columns = [c[len(prefix) :] for c in cols]
    return df


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment. NaN p-values are treated as 1.0."""
    p = np.where(np.isnan(p_values), 1.0, p_values)
    m = len(p)
    order = np.argsort(p)
    # (m - j) * p_(j), then a running maximum to keep the sequence monotone
    adjusted = np.maximum.accumulate(np.arange(m, 0, -1) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def significance_scores(
    values: pd.DataFrame,
    higher_is_better: bool,
) -> Dict[str, float]:
    """Fraction of other methods each method significantly outperforms."""
    methods = list(values.index)
    pairs = list(permutations(methods, 2))
    alternative = "greater" if higher_is_better else "less"

    p_values = []
    for a, b in pairs:
        diff = values.loc[a].to_numpy() - values.loc[b].to_numpy()
        if not np.any(diff != 0):  # identical results, no evidence either way
            p_values.append(1.0)
            continue
        p_values.append(
            stats.wilcoxon(diff, alternative=alternative, zero_method="wilcox").pvalue
        )

    adjusted = holm(np.array(p_values))

    wins = {m: 0 for m in methods}
    for (a, _), p in zip(pairs, adjusted):
        if p < ALPHA:
            wins[a] += 1
    return {m: wins[m] / (len(methods) - 1) for m in methods}


def score_cohort(
    data: Dict[str, pd.DataFrame],
    subjects: List[str],
) -> pd.DataFrame:
    """Full scoring pipeline over the given subjects."""
    scores = {}
    for metric, (_, _, higher_is_better) in METRICS.items():
        scores[metric] = significance_scores(data[metric][subjects], higher_is_better)

    df = pd.DataFrame(scores)
    df["accuracy"] = df[ACCURACY_METRICS].mean(axis=1)
    df["pet_biomarker"] = df[PET_BIOMARKER_METRICS].mean(axis=1)
    df["regularity"] = df[REGULARITY_METRICS].mean(axis=1)
    df["final"] = 100.0 * (
        0.50 * df["accuracy"] + 0.45 * df["pet_biomarker"] + 0.05 * df["regularity"]
    )
    return df


def short_name(model: str) -> str:
    return model.replace("PSMAReg_LapIRN_", "").replace("_stagelvl3_best", "")


def main() -> None:
    data = {m: load_metric(f, p) for m, (f, p, _) in METRICS.items()}

    # only methods and subjects evaluated for every metric can be compared
    methods = sorted(set.intersection(*(set(d.index) for d in data.values())))
    subjects = sorted(set.intersection(*(set(d.columns) for d in data.values())))
    dropped = sorted(set().union(*(set(d.index) for d in data.values())) - set(methods))

    # absolute bias, and the %NDV practical-equivalence threshold
    data["mtv"] = data["mtv"].abs()
    data["tlg"] = data["tlg"].abs()
    data["ndv"] = (data["ndv"] - NDV_EQUIVALENCE_THRESHOLD).clip(lower=0.0)

    data = {m: d.loc[methods, subjects] for m, d in data.items()}

    print(f"{len(methods)} methods, {len(subjects)} subjects")
    if dropped:
        print(f"dropped (not in every metric): {', '.join(dropped)}")
    if not data["ndv"].to_numpy().any():
        print(
            f"all %NDV values are within the {NDV_EQUIVALENCE_THRESHOLD}% "
            "practical-equivalence threshold -> regularity score is 0 for everyone"
        )
    print("accuracy score covers CT DSC and CT HD95 only (no PET metrics locally)\n")

    df = score_cohort(data, subjects).sort_values("final", ascending=False)
    df["rank"] = df["final"].rank(ascending=False, method="min").astype(int)

    columns = list(METRICS) + ["accuracy", "pet_biomarker", "regularity", "final"]
    header = f"{'rank':<5}{'method':<28}" + "".join(f"{c:>15}" for c in columns)
    print(header)
    for model, row in df.iterrows():
        print(
            f"{int(row['rank']):<5}{short_name(model):<28}"
            + "".join(f"{row[c]:>15.3f}" for c in columns)
        )

    if N_BOOTSTRAP:
        rng = np.random.default_rng(0)
        ranks = {m: [] for m in methods}
        for _ in range(N_BOOTSTRAP):
            sample = list(rng.choice(subjects, size=len(subjects), replace=True))
            boot = score_cohort(data, sample)
            boot_rank = boot["final"].rank(ascending=False, method="min")
            for m in methods:
                ranks[m].append(boot_rank[m])

        print(f"\nbootstrap rank intervals ({N_BOOTSTRAP} samples)")
        for model in df.index:
            lo, hi = np.percentile(ranks[model], [2.5, 97.5])
            median = np.median(ranks[model])
            print(
                f"{short_name(model):<28}median {median:>5.1f}   "
                f"95% [{lo:.0f}, {hi:.0f}]"
            )


if __name__ == "__main__":
    main()
