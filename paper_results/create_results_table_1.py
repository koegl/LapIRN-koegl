"""Build results table 1: LapIRN variants on the official validation set.

Reads the per-case metric CSVs from submission_results/csvs/chase_leaderboard,
reproduces the challenge ranking score, runs paired Wilcoxon signed-rank tests
of the best-scoring variant against every other variant, and writes the LaTeX
table plus a CSV of all p-values next to this file.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

HERE = Path(__file__).resolve().parent
CSV_DIR = HERE.parent / "submission_results" / "csvs" / "chase_leaderboard"
OUT_TEX = HERE / "results_table_1.tex"
OUT_PVALS = HERE / "results_table_1_pvalues.csv"

METRICS = ["dice", "hd95", "mtv", "tlg", "ndv"]
# True when a larger value is better.
HIGHER_IS_BETTER = {
    "dice": True,
    "hd95": False,
    "mtv": False,
    "tlg": False,
    "ndv": False,
}
# Multiplier applied for display only (the fractional errors are reported in %).
# DSC and the MTV/TLG errors are reported in %, NDV in ppm (its values span
# several orders of magnitude and are tiny, so % would be all leading zeros).
DISPLAY_SCALE = {"dice": 100.0, "hd95": 1.0, "mtv": 100.0, "tlg": 100.0, "ndv": 1e6}
# Fixed notation prints this many significant figures; the decimal places are
# derived from the cell's mean and reused for its std, so the two align.
SIGNIFICANT_FIGURES = 3
# Mantissa decimals used only in scientific notation.
SCIENTIFIC_DECIMALS = {"dice": 1, "hd95": 1, "mtv": 1, "tlg": 1, "ndv": 1}
# "fixed" prints plain decimals, "scientific" prints m.m x 10^e.
DISPLAY_NOTATION = {
    "dice": "fixed",
    "hd95": "fixed",
    "mtv": "fixed",
    "tlg": "fixed",
    "ndv": "fixed",
}
# Degrees to rotate the metric column headers (needs \usepackage{graphicx}).
# 0 keeps them upright, 45 is a good compromise, 90 makes the columns as narrow
# as their numbers.
HEADER_ROTATION = 45

HEADERS = {
    "dice": r"DSC (\%) $\uparrow$",
    "hd95": r"HD95 (mm) $\downarrow$",
    "mtv": r"MTV \%err $\downarrow$",
    "tlg": r"TLG \%err $\downarrow$",
    "ndv": r"NDV (ppm) $\downarrow$",
}

# NDV tolerance from the challenge scoring: max(%NDV - 0.005%, 0), i.e. 50 ppm.
# The CSVs store NDV as a fraction, so 0.005% == 5e-5.
NDV_TOLERANCE = 5e-5

# Challenge component weights.
WEIGHTS = {"accuracy": 0.4, "biomarker": 0.4, "regularity": 0.2}

ALPHA = 0.05

# Both off for the short method paper (width); turn back on for the journal
# extension. SHOW_STD adds "+- std" to the mean, SHOW_MEDIAN_IQR adds a second
# line per row with the median [IQR].
SHOW_STD = False
SHOW_MEDIAN_IQR = False

# Rows shown for context at the top of the table but excluded from the ranking
# and from the statistical tests (they are missing some of the five metrics).
REFERENCE_MODELS = ["before_registration", "affine"]
# Rows dropped entirely (reported in the baseline comparison table instead).
EXCLUDED_MODELS = ["convexadam", "niftyreg"]

# Reference rows keep fixed labels; every variant is auto-labelled V1, V2, ...
# in the order it appears in the CSVs, so adding a row needs no code change.
REFERENCE_LABELS = {
    "before_registration": "Initial",
    "affine": "Affine",
}
VARIANT_LABEL_PREFIX = "V"


def build_display_names(variants_in_csv_order):
    names = dict(REFERENCE_LABELS)
    for number, model in enumerate(variants_in_csv_order, start=1):
        names[model] = f"{VARIANT_LABEL_PREFIX}{number}"
    return names


def normalise_model_id(name):
    """Model ids differ slightly between CSVs (trailing spaces, submission ids)."""
    name = re.sub(r"\s*sub[ _]id:\s*\d+", "", name, flags=re.IGNORECASE)
    return name.strip().lower()


def load_metric(metric):
    """Return a (model x case) DataFrame of per-case values for one metric."""
    df = pd.read_csv(CSV_DIR / f"results_official_val_{metric}.csv")
    df["model"] = df["model"].map(normalise_model_id)
    case_columns = [c for c in df.columns if c.startswith(f"{metric}_")]
    cases = [c.split("_")[-1] for c in case_columns]
    out = df.set_index("model")[case_columns]
    out.columns = cases
    return out


def holm_correct(pvalues):
    """Holm-Bonferroni step-down correction; returns adjusted p-values."""
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    n = len(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for i, idx in enumerate(order):
        value = min(1.0, (n - i) * pvalues[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted


def challenge_scores(means, models):
    """Reproduce the challenge ranking: per-metric rank -> score -> weighted geometric mean."""
    metric_scores = {}
    k = len(models)
    for metric in METRICS:
        values = np.array([means[metric][m] for m in models], dtype=float)
        if metric == "ndv":
            values = np.maximum(values - NDV_TOLERANCE, 0.0)
        # rankdata with method="max" makes ties share the worst rank they span.
        ranks = rankdata(-values if HIGHER_IS_BETTER[metric] else values, method="max")
        metric_scores[metric] = (k - ranks + 1) / k

    accuracy = np.sqrt(metric_scores["dice"] * metric_scores["hd95"])
    biomarker = np.sqrt(metric_scores["mtv"] * metric_scores["tlg"])
    regularity = metric_scores["ndv"]
    total = 100.0 * (
        accuracy ** WEIGHTS["accuracy"]
        * biomarker ** WEIGHTS["biomarker"]
        * regularity ** WEIGHTS["regularity"]
    )
    return pd.DataFrame(
        {
            **metric_scores,
            "accuracy": accuracy,
            "biomarker": biomarker,
            "regularity": regularity,
            "score": total,
        },
        index=models,
    )


def rotate_header(text):
    """Rotated headers let a column be as narrow as its numbers."""
    if not HEADER_ROTATION:
        return text
    return r"\rotatebox[origin=l]{" + str(HEADER_ROTATION) + "}{" + text + "}"


def decimals_for(value, significant_figures=SIGNIFICANT_FIGURES):
    """Decimal places needed to show `value` to the given significant figures."""
    if value == 0 or not np.isfinite(value):
        return significant_figures - 1
    exponent = int(np.floor(np.log10(abs(value))))
    return max(significant_figures - 1 - exponent, 0)


def format_number(value, metric, decimals):
    """Format one number in the notation configured for its metric."""
    if DISPLAY_NOTATION[metric] == "fixed":
        return f"{value:.{decimals}f}"
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.{SCIENTIFIC_DECIMALS[metric]}e}".split("e")
    return f"{mantissa}\\!\\times\\!10^{{{int(exponent)}}}"


def format_cell(values, metric):
    """mean +- std on the first line, median [IQR] on the second."""
    if values is None:
        return "--", "--"
    scaled = np.asarray(values, dtype=float) * DISPLAY_SCALE[metric]
    mean, std = scaled.mean(), scaled.std(ddof=1)
    q1, median, q3 = np.percentile(scaled, [25, 50, 75])

    # The mean sets the precision; the std follows it so the pair lines up.
    mean_decimals = decimals_for(mean)
    median_decimals = decimals_for(median)

    if DISPLAY_NOTATION[metric] == "fixed":
        mean_line = format_number(mean, metric, mean_decimals)
        if SHOW_STD:
            mean_line += f" $\\pm$ {format_number(std, metric, mean_decimals)}"
        median_line = (
            f"{format_number(median, metric, median_decimals)} "
            f"[{format_number(q1, metric, median_decimals)}, "
            f"{format_number(q3, metric, median_decimals)}]"
        )
    else:
        mean_line = format_number(mean, metric, mean_decimals)
        if SHOW_STD:
            mean_line += f" \\pm {format_number(std, metric, mean_decimals)}"
        mean_line = f"${mean_line}$"
        median_line = (
            f"${format_number(median, metric, median_decimals)} "
            f"[{format_number(q1, metric, median_decimals)}, "
            f"{format_number(q3, metric, median_decimals)}]$"
        )
    return mean_line, median_line


def main():
    per_case = {metric: load_metric(metric) for metric in METRICS}

    # Variants are the models present in every metric file, minus the excluded
    # baselines and the reference rows.
    complete = set.intersection(*(set(df.index) for df in per_case.values()))
    variants = [
        m
        for m in per_case["dice"].index
        if m in complete and m not in REFERENCE_MODELS and m not in EXCLUDED_MODELS
    ]
    references = [m for m in REFERENCE_MODELS if m in per_case["dice"].index]

    display_names = build_display_names(variants)

    mean_description = r"mean $\pm$ std" if SHOW_STD else "mean"
    if SHOW_MEDIAN_IQR:
        cell_description = (
            f"Each cell reports the {mean_description} over cases (top) and the "
            r"median [IQR] (bottom). "
        )
    else:
        cell_description = f"Each cell reports the {mean_description} over cases. "

    n_cases = per_case["dice"].shape[1]
    means = {metric: per_case[metric].mean(axis=1) for metric in METRICS}

    # Best value per metric among the ranked variants; the reference rows do not
    # compete for the bolding.
    best_per_metric = {
        metric: (
            means[metric][variants].idxmax()
            if HIGHER_IS_BETTER[metric]
            else means[metric][variants].idxmin()
        )
        for metric in METRICS
    }

    scores = challenge_scores(means, variants)
    scores = scores.sort_values("score", ascending=False)
    ordered_variants = list(scores.index)
    best = ordered_variants[0]

    # Paired Wilcoxon signed-rank of the best variant against every other one,
    # Holm-corrected within each metric.
    pval_records = []
    significant = {}  # (model, metric) -> True when the best variant wins
    for metric in METRICS:
        others = ordered_variants[1:]
        raw = []
        for model in others:
            x = per_case[metric].loc[best].to_numpy(dtype=float)
            y = per_case[metric].loc[model].to_numpy(dtype=float)
            if np.allclose(x, y):
                raw.append(1.0)
            else:
                raw.append(wilcoxon(x, y, alternative="two-sided").pvalue)
        adjusted = holm_correct(raw)
        for model, p_raw, p_adj in zip(others, raw, adjusted):
            better = (
                means[metric][best] > means[metric][model]
                if HIGHER_IS_BETTER[metric]
                else means[metric][best] < means[metric][model]
            )
            significant[(model, metric)] = bool(p_adj < ALPHA and better)
            pval_records.append(
                {
                    "metric": metric,
                    "best_model": display_names.get(best, best),
                    "compared_model": display_names.get(model, model),
                    "p_raw": p_raw,
                    "p_holm": p_adj,
                    "best_is_better": better,
                    "significant": significant[(model, metric)],
                }
            )
    pd.DataFrame(pval_records).to_csv(OUT_PVALS, index=False)

    # --- LaTeX ------------------------------------------------------------
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(METRICS) + "c}",
        r"\toprule",
        "Model & "
        + " & ".join(rotate_header(HEADERS[m]) for m in METRICS)
        + " & "
        + rotate_header(r"Score $\uparrow$")
        + r" \\",
        r"\midrule",
    ]

    def emit(model, score_cell, bold=False):
        name = display_names.get(model, model)
        mean_cells, median_cells = [], []
        for metric in METRICS:
            table = per_case[metric]
            values = (
                table.loc[model].to_numpy(dtype=float) if model in table.index else None
            )
            mean_line, median_line = format_cell(values, metric)
            if best_per_metric[metric] == model:
                mean_line = r"\textbf{" + mean_line + "}"
            if significant.get((model, metric)):
                mean_line += r"$^{*}$"
            mean_cells.append(mean_line)
            median_cells.append(median_line)
        if bold:
            name = r"\textbf{" + name + "}"
            score_cell = r"\textbf{" + score_cell + "}"
        lines.append(
            f"{name} & " + " & ".join(mean_cells) + f" & {score_cell} " + r"\\"
        )
        if SHOW_MEDIAN_IQR:
            lines.append(
                r"\scriptsize\textcolor{gray}{median [IQR]} & "
                + " & ".join(
                    r"\scriptsize\textcolor{gray}{" + c + "}" for c in median_cells
                )
                + r" & \\"
            )
            lines.append(r"\addlinespace[2pt]")

    for model in references:
        emit(model, "--")
    if references:
        lines.append(r"\midrule")

    for model in ordered_variants:
        score = scores.loc[model, "score"]
        emit(model, f"{score:.{decimals_for(score)}f}", bold=(model == best))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{LapIRN variants on the official validation set ($n="
        + str(n_cases)
        + r"$ cases). "
        + cell_description
        + r"DSC and the MTV/TLG errors are given in \%, NDV in ppm. Score reproduces the challenge ranking: "
        r"each metric is ranked separately (ties take the worst rank spanned), turned into "
        r"$(K-\mathrm{rank}+1)/K$, and combined as "
        r"$100 \times \mathrm{accuracy}^{0.4}\,\mathrm{biomarker}^{0.4}\,\mathrm{regularity}^{0.2}$ "
        r"with accuracy $=$ geometric mean of DSC and HD95, biomarker $=$ geometric mean of "
        r"MTV and TLG \%err, and regularity $=$ NDV after the challenge tolerance "
        r"$\max(\mathrm{NDV}-50\,\mathrm{ppm},\,0)$. The best-scoring variant is shown in bold; "
        r"$^{*}$ marks metrics on which it is significantly better than the given variant "
        r"(paired Wilcoxon signed-rank over the "
        + str(n_cases)
        + r" cases, Holm-corrected "
        r"within each metric, $p<0.05$). The unregistered and affine-only rows are shown for "
        r"reference; they lack some metrics and are excluded from the ranking and the tests. "
        r"Because the winning variant is selected on the same cases, these tests are "
        r"exploratory rather than confirmatory.}",
        r"\label{tab:lapirn_variants}",
        r"\end{table}",
        "",
    ]

    OUT_TEX.write_text("\n".join(lines))
    print(f"wrote {OUT_TEX}")
    print(f"wrote {OUT_PVALS}")
    print(
        f"\nbest variant: {display_names.get(best, best)}  (score {scores.loc[best, 'score']:.2f})"
    )
    print("\nranking:")
    for position, model in enumerate(ordered_variants, start=1):
        print(
            f"  {position:2d}. {display_names.get(model, model)} {scores.loc[model, 'score']:6.2f}   {model}"
        )


if __name__ == "__main__":
    main()
