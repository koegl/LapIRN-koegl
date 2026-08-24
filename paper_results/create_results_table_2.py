"""Build results table 2: our three configurations against the baselines.

Shares the CSV loading, formatting and correction helpers with
create_results_table_1.py so that both tables stay typographically identical.

Three rows of ours are reported. The container row is the submitted method and
the one the baselines are tested against; the validation row is the
unconstrained configuration of Sec.~\\ref{sec:io}; the no-IO row is the backbone
alone, which isolates the contribution of test-time refinement.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from create_results_table_1 import (
    ALPHA,
    CAPTION_SKIP,
    HEADERS,
    HIGHER_IS_BETTER,
    METRICS,
    format_cell,
    holm_correct,
    load_metric,
    normalise_model_id,
    rotate_header,
)
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
OUT_TEX = HERE / "results_table_2.tex"
OUT_PVALS = HERE / "results_table_2_pvalues.csv"

# Metrics for the container run come from a single wide CSV written by the
# container's own evaluation, one row per case plus a trailing "mean" row.
CONTAINER_CSV = (
    HERE.parent / "submission" / "validation_predictions" / "metrics_io_it18_lr_10_dice.csv"
)
# Same evaluator, backbone only. Not used for a table row -- the no-IO row comes
# from the leaderboard CSVs like every other row -- but it evaluates the same
# model as NO_IO_MODEL below, so the two together measure how far the container's
# evaluator sits from the challenge one. Printed as a diagnostic on every run.
NO_IO_LOCAL_CSV = HERE.parent / "submission" / "validation_predictions" / "metrics_no_io.csv"

# Baselines in the order they should appear, above our rows.
BASELINE_LABELS = {
    "before_registration": "Initial",
    "affine": "Affine",
    "niftyreg": "NiftyReg",
    "convexadam": "ConvexAdam",
}

VALIDATION_MODEL = normalise_model_id(
    "polite-snake-38577202_io_lr1.0e-02_it90_wncc5.00_wdicect5.00_wjac10.00"
    "_wsmooth1.00_wbonerigid0.00_wmtv50.00_wmtvmean500.00_wjactum2.50_wtlg1.50"
)
NO_IO_MODEL = normalise_model_id("auspicious-sloth-39469081_combined")

# Our rows, in the order they should appear below the baselines. The source is
# either ("leaderboard", model id) for the per-metric challenge CSVs or
# ("local", path) for a wide one-row-per-case CSV.
OUR_ROWS = [
    ("no_io", "Ours (no IO)", ("leaderboard", NO_IO_MODEL)),
    ("container", "Ours (container)", ("local", CONTAINER_CSV)),
    ("validation", "Ours (validation)", ("leaderboard", VALIDATION_MODEL)),
]

# The row the baselines are tested against and whose name is bolded: the
# submitted container, since that is what gets ranked on the hidden test set.
PRIMARY_KEY = "container"

# Wall-clock seconds for one image pair on the hardware of Sec.~\ref{sec:infrastructure}.
# Measured, not per-case, so these take no mean, no std and no significance mark.
# None prints as a dash -- fill these in as the measurements come.
RUNTIME_S = {
    "before_registration": None,
    "affine": None,
    "niftyreg": None,
    "convexadam": None,
    "container": None,
    "validation": None,
    "no_io": None,
}
RUNTIME_HEADER = r"Time (s) $\downarrow$"
# Decimals used for the runtime cells.
RUNTIME_DECIMALS = 0


def load_local_metrics(path, cases):
    """Read a wide per-case CSV and return {metric: values in `cases` order}."""
    df = pd.read_csv(path, dtype={"case": str})
    df = df[df["case"] != "mean"].set_index("case")
    missing = [c for c in cases if c not in df.index]
    if missing:
        raise SystemExit(f"{path.name} is missing cases {missing}")
    absent = [m for m in METRICS if m not in df.columns]
    if absent:
        raise SystemExit(f"{path.name} is missing metrics {absent}")
    return {metric: df.loc[cases, metric].to_numpy(dtype=float) for metric in METRICS}


def format_runtime(seconds):
    if seconds is None:
        return "--"
    return f"{seconds:.{RUNTIME_DECIMALS}f}"


def report_evaluator_agreement(leaderboard, cases):
    """The container CSV and the challenge CSVs are produced by different code.

    Both evaluate NO_IO_MODEL, so their difference on that model bounds how much
    of any container-vs-baseline gap is the evaluator rather than the method.
    """
    if not NO_IO_LOCAL_CSV.exists() or NO_IO_MODEL not in leaderboard["dice"].index:
        return
    local = load_local_metrics(NO_IO_LOCAL_CSV, cases)
    print(f"\nevaluator agreement on {NO_IO_MODEL} (local CSV vs challenge CSVs):")
    for metric in METRICS:
        if NO_IO_MODEL not in leaderboard[metric].index:
            continue
        challenge = leaderboard[metric].loc[NO_IO_MODEL].to_numpy(dtype=float).mean()
        mine = local[metric].mean()
        delta = (mine - challenge) / abs(challenge) * 100 if challenge else 0.0
        print(f"  {metric:5s} challenge={challenge:.6g}  local={mine:.6g}  ({delta:+.1f}%)")
    print(
        "  the container row is read from the local CSV and the other rows from the\n"
        "  challenge CSVs, so a gap of this size is baked into every container cell."
    )


def main():
    leaderboard = {metric: load_metric(metric) for metric in METRICS}
    cases = list(leaderboard["dice"].columns)
    n_cases = len(cases)

    baselines = [m for m in BASELINE_LABELS if m in leaderboard["dice"].index]
    missing = [m for m in BASELINE_LABELS if m not in leaderboard["dice"].index]
    if missing:
        print(f"warning: baselines absent from the CSVs and skipped: {missing}")

    # Per-case values keyed by row key; None where a row has no such metric.
    values = {}
    display_names = {}

    for model in baselines:
        display_names[model] = BASELINE_LABELS[model]
        for metric in METRICS:
            table = leaderboard[metric]
            values[(model, metric)] = (
                table.loc[model].to_numpy(dtype=float) if model in table.index else None
            )

    our_keys = []
    for key, label, (kind, source) in OUR_ROWS:
        display_names[key] = label
        our_keys.append(key)
        if kind == "local":
            local = load_local_metrics(source, cases)
            for metric in METRICS:
                values[(key, metric)] = local[metric]
        else:
            if source not in leaderboard["dice"].index:
                raise SystemExit(f"model not found in the CSVs: {source}")
            for metric in METRICS:
                table = leaderboard[metric]
                values[(key, metric)] = (
                    table.loc[source].to_numpy(dtype=float)
                    if source in table.index
                    else None
                )

    rows = baselines + our_keys
    means = {
        metric: {
            key: values[(key, metric)].mean()
            for key in rows
            if values[(key, metric)] is not None
        }
        for metric in METRICS
    }

    # Best value per metric over the baselines and all three of our rows, so a
    # row like no-IO can hold the best NDV and still be marked.
    best_per_metric = {}
    for metric in METRICS:
        present = means[metric]
        best_per_metric[metric] = (
            max(present, key=present.get)
            if HIGHER_IS_BETTER[metric]
            else min(present, key=present.get)
        )

    # Paired Wilcoxon of the primary (container) row against each baseline,
    # Holm-corrected within each metric over the baselines that have that metric.
    # Our own rows are never tested against each other -- that is not what the
    # marks mean -- so the comparison count is unchanged.
    pval_records = []
    significant = {}
    for metric in METRICS:
        others = [m for m in baselines if values[(m, metric)] is not None]
        x = values[(PRIMARY_KEY, metric)]
        raw = []
        for model in others:
            y = values[(model, metric)]
            raw.append(
                1.0
                if np.allclose(x, y)
                else wilcoxon(x, y, alternative="two-sided").pvalue
            )
        adjusted = holm_correct(raw) if raw else []
        for model, p_raw, p_adj in zip(others, raw, adjusted):
            better = (
                means[metric][PRIMARY_KEY] > means[metric][model]
                if HIGHER_IS_BETTER[metric]
                else means[metric][PRIMARY_KEY] < means[metric][model]
            )
            significant[(model, metric)] = bool(p_adj < ALPHA and better)
            pval_records.append(
                {
                    "metric": metric,
                    "compared_baseline": display_names[model],
                    "p_raw": p_raw,
                    "p_holm": p_adj,
                    "ours_is_better": better,
                    "significant": significant[(model, metric)],
                }
            )
    pd.DataFrame(pval_records).to_csv(OUT_PVALS, index=False)

    # --- LaTeX ------------------------------------------------------------
    caption_lines = [
        r"\caption{Our method against the baselines on the official challenge "
        r"validation set ($n=" + str(n_cases) + r"$ cases), evaluated against the "
        r"surrogate labels of Sec.~\ref{sec:evaluation}. "
        r"\emph{Ours (container)} is the submitted method of Sec.~\ref{sec:container} and "
        r"the configuration the baselines are tested against; "
        r"\emph{Ours (validation)} is the unconstrained configuration, reported as the "
        r"upper bound the container's 90\,s per-pair budget gives up; "
        r"\emph{Ours (no IO)} is the backbone without instance optimisation. "
        r"Each cell reports the mean over cases. "
        r"DSC and the MTV/TLG errors are given in \%, NDV in ppm. "
        r"Time is the measured wall-clock runtime for one image pair on the hardware of "
        r"Sec.~\ref{sec:infrastructure}; it is a single measurement, not a per-case "
        r"distribution, and is not tested or marked. "
        r"Bold marks the best value in each column. "
        r"$^{*}$ marks metrics on which \emph{Ours (container)} is significantly better "
        r"than the baseline in the given row (paired Wilcoxon signed-rank, Holm-corrected "
        r"within each metric, $p<0.05$); the other two rows of ours are reported for "
        r"reference and carry no significance marks. "
        r"Dashes mark metrics that are undefined for a baseline: the unregistered and "
        r"affine cases have no deformation field, and the PET biomarkers are only "
        r"meaningful after resampling.}",
        r"\label{tab:baselines}",
    ]

    lines = [
        r"% Generated by paper_results/create_results_table_2.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\setlength{\belowcaptionskip}{" + CAPTION_SKIP + "}",
        *caption_lines,
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(METRICS) + "c}",
        r"\toprule",
        "Method & "
        + " & ".join(rotate_header(HEADERS[m]) for m in METRICS)
        + " & "
        + rotate_header(RUNTIME_HEADER)
        + r" \\",
        r"\midrule",
    ]

    def emit(key):
        name = display_names[key]
        cells = []
        for metric in METRICS:
            cell, _ = format_cell(values[(key, metric)], metric)
            if best_per_metric[metric] == key:
                cell = r"\textbf{" + cell + "}"
            if significant.get((key, metric)):
                cell += r"$^{*}$"
            cells.append(cell)
        cells.append(format_runtime(RUNTIME_S.get(key)))
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")

    for model in baselines:
        emit(model)
    lines.append(r"\midrule")
    for key in our_keys:
        emit(key)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    OUT_TEX.write_text("\n".join(lines))
    print(f"wrote {OUT_TEX}")
    print(f"wrote {OUT_PVALS}")

    unmeasured = [display_names[k] for k in rows if RUNTIME_S.get(k) is None]
    if unmeasured:
        print(f"\nruntime still unmeasured (prints as a dash): {unmeasured}")

    print("\nmeans:")
    for key in rows:
        summary = "  ".join(
            f"{metric}={means[metric][key]:.5g}" if key in means[metric] else f"{metric}=--"
            for metric in METRICS
        )
        print(f"  {display_names[key]:<18s} {summary}  time={format_runtime(RUNTIME_S.get(key))}")

    report_evaluator_agreement(leaderboard, cases)


if __name__ == "__main__":
    main()
