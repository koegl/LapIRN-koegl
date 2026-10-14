"""The PSMAReg challenge's significance-based ranking, on patient-level values.

Used to pick the config of every tuned method (IO weights, NiftyReg,
ConvexAdam: the configs of one grid compete like methods) and for the final
test results (all methods, plus the bootstrap of the ranks).

Per metric, every ordered pair of candidates is compared with a one-sided
Wilcoxon signed-rank test on the patients both have values for; the p-values of
a metric are Holm-adjusted together; A beats B if its adjusted p < ALPHA.
significance score = (number of candidates beaten + 1) / K.

    accuracy   = mean(score CT Dice, score CT HD95)    (no PET Dice / HD95: no labels)
    biomarker  = mean(score |MTV|, score |TLG|)
    regularity = score of %NDV, after the 0.005 % equivalence threshold
    final      = 100 * accuracy^0.4 * biomarker^0.4 * regularity^0.2

Differences from the challenge, both because a patient may have several pairs:
the threshold is applied per pair, and each metric is then averaged over a
patient's pairs (NaNs, e.g. MTV / TLG of a lesion-free follow-up, skipped).
Equal final scores are broken by the higher mean CT Dice.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ALPHA = 0.05
NDV_EQUIVALENCE_PERCENT = 0.005
# metric column -> True if higher is better
METRICS: Dict[str, bool] = {
    "dice_ct": True,
    "hd95": False,
    "mtv_bias": False,
    "tlg_bias": False,
    "ndv_adjusted": False,
}
COMPONENTS = {
    "accuracy": ("dice_ct", "hd95"),
    "biomarker": ("mtv_bias", "tlg_bias"),
    "regularity": ("ndv_adjusted",),
}
WEIGHTS = {"accuracy": 0.4, "biomarker": 0.4, "regularity": 0.2}


def patient_level(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pair evaluation CSV (evaluate.py) -> one row per patient."""
    df = df.assign(
        ndv_adjusted=(df["ndv_percent"] - NDV_EQUIVALENCE_PERCENT).clip(lower=0.0)
    )
    return df.groupby(df["case_id"].astype(str))[list(METRICS)].mean()


def holm(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def one_sided_p(a: pd.Series, b: pd.Series, higher_is_better: bool) -> float:
    """p-value of "a is better than b" over the patients both have."""
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    diff = both["a"] - both["b"]
    if (diff == 0).all():  # also covers no common patients
        return 1.0
    return float(
        wilcoxon(
            both["a"], both["b"], alternative="greater" if higher_is_better else "less"
        ).pvalue
    )


def rank(patients: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Ranking table of the candidates, best first. patients: name ->
    patient-level frame (patient_level)."""
    names = list(patients)
    k = len(names)
    table = pd.DataFrame(index=names)
    for metric, higher in METRICS.items():
        ordered = [(a, b) for a in names for b in names if a != b]
        p = np.array(
            [one_sided_p(patients[a][metric], patients[b][metric], higher) for a, b in ordered]
        )
        beats = holm(p) < ALPHA
        wins = {name: 0 for name in names}
        for (a, _), won in zip(ordered, beats):
            wins[a] += int(won)
        table[f"score_{metric}"] = [(wins[n] + 1) / k for n in names]
        table[metric] = [patients[n][metric].mean() for n in names]
    for component, metrics in COMPONENTS.items():
        table[component] = table[[f"score_{m}" for m in metrics]].mean(axis=1)
    table["final"] = 100.0 * np.prod(
        [table[c] ** w for c, w in WEIGHTS.items()], axis=0
    )
    table = table.sort_values(["final", "dice_ct"], ascending=False)
    table["rank"] = table["final"].rank(ascending=False, method="min").astype(int)
    return table


def select(candidates: Dict[str, pd.DataFrame]) -> Dict:
    """Pick the best of several candidates from their per-pair CSV frames."""
    table = rank({name: patient_level(df) for name, df in candidates.items()})
    return {"chosen": table.index[0], "table": table.reset_index(names="candidate").to_dict("records")}


def bootstrap_ranks(
    candidates: Dict[str, pd.DataFrame], n_samples: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Rank distribution over patient bootstrap samples: the same resampled
    patients for every candidate in a sample (paired design). Returns per
    candidate the median rank and the 95 % rank interval."""
    patients = {name: patient_level(df) for name, df in candidates.items()}
    ids = sorted(set().union(*(p.index for p in patients.values())))
    rng = np.random.default_rng(seed)
    ranks: Dict[str, List[int]] = {name: [] for name in patients}
    for _ in range(n_samples):
        sample = rng.choice(ids, size=len(ids), replace=True)
        resampled = {
            name: p.reindex(sample).reset_index(drop=True) for name, p in patients.items()
        }
        table = rank(resampled)
        for name in patients:
            ranks[name].append(int(table.loc[name, "rank"]))
    return pd.DataFrame(
        {
            name: {
                "median_rank": float(np.median(r)),
                "rank_2.5": float(np.percentile(r, 2.5)),
                "rank_97.5": float(np.percentile(r, 97.5)),
            }
            for name, r in ranks.items()
        }
    ).T
