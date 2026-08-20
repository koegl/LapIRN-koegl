"""Build the checkpoint-selection figure: DSC plateau vs. tumour-metric optimum.

Reads the three W&B metric exports of a single level-3 run from this directory
and writes a LaTeX figure with two adjacent pgfplots axes: validation CT Dice on
the left, MTV and TLG bias on the right. The figure is drawn entirely by LaTeX
and needs no image file: the plotted series are written to a companion CSV next
to the .tex and read back by pgfplots at compile time, so both files have to be
copied into the document together.

The vertical red dashed line marking the selected checkpoint is set by
SELECTED_STEP below and is drawn at the same position in both panels.

Requires in the document preamble:
    \\usepackage{pgfplots}
    \\pgfplotsset{compat=1.18}
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# --- the checkpoint we selected -------------------------------------------
# Global step of the red dashed line, drawn in both panels. Set to None to
# omit the line entirely.
SELECTED_STEP = 43000

RUN = "rebellious_stork"
CSV_DSC = HERE / f"dsc_{RUN}.csv"
CSV_MTV = HERE / f"mtv_{RUN}.csv"
CSV_TLG = HERE / f"tlg_{RUN}.csv"
OUT_TEX = HERE.parents[1] / "overleaf" / "figures" / "dsc_plateau.tex"
# The plotted series are written next to the .tex and read back by pgfplots at
# compile time, so the figure source stays short and the numbers stay editable.
OUT_DATA = OUT_TEX.with_name("dsc_plateau_data.csv")
# How the data file is addressed from inside the document. \addplot resolves
# paths relative to the main .tex, not to the file doing the \input.
DATA_PATH_IN_TEX = f"figures/{OUT_DATA.name}"

# Validation is noisy from round to round. The raw curve is drawn faintly and a
# centred rolling mean over this many validation rounds on top of it; set to 1
# to disable the smoothing (only the raw curve is then drawn, at full opacity).
SMOOTH_WINDOW = 9

# x axis is drawn in thousands of steps to keep the tick labels short.
STEP_SCALE = 1e-3
# Fixed x range of both panels, in the units of the drawn axis (thousands of
# steps). Set to None to let pgfplots pick the range from the data.
XMIN, XMAX = 0.0, 120.0
# Spacing of the x ticks, in the same units.
XTICK_DISTANCE = 20
# Everything is logged as a fraction. Dice is shown as a fraction (0.80), the
# biomarker biases in per cent.
DSC_SCALE = 1.0
BIAS_SCALE = 100.0
# Decimals on the y tick labels of each panel; "zerofill" pads them so the
# labels keep a constant width.
DSC_PRECISION = 2
BIAS_PRECISION = 0

# Gap between an axis label and its tick labels. Negative values pull the label
# towards the axis; "ylabel near ticks" already anchors it to the tick labels
# rather than to the widest possible tick, so these are a fine adjustment.
YLABEL_SHIFT = "-4pt"
XLABEL_SHIFT = "-2pt"

# Panel geometry, in fractions of \linewidth of the enclosing figure. This is
# the width of the plotting area alone ("scale only axis"), so each picture is
# some 30pt wider once the y-label and tick labels are added; keep the pair
# comfortably below 0.5\linewidth each or the second panel wraps onto its own
# line.
AXIS_WIDTH = "0.39\\linewidth"
AXIS_HEIGHT = "0.26\\linewidth"

COLOR_DSC = "0.12,0.34,0.62"  # blue
COLOR_MTV = "0.85,0.37,0.01"  # orange
COLOR_TLG = "0.20,0.55,0.28"  # green

FIG_LABEL = "fig:selection"
CAPTION = (
    "Accuracy and biomarker preservation do not peak at the same checkpoint. "
    "Validation CT Dice (left) rises to a plateau, while the MTV and TLG bias "
    "(right) reach their optimum roughly halfway through level-3 training and "
    "degrade thereafter. The dashed line marks the checkpoint selected by "
    "Eq.~\\eqref{eq:score}. Faint lines are the per-round values, solid lines a "
    "centred rolling mean."
)


def read_metric(path: Path, scale: float) -> pd.DataFrame:
    """Return the (step, value) series of a W&B export.

    The exports carry a ``global_step`` column plus one column per run; the
    ``__MIN`` / ``__MAX`` companions are identical for a single run and the
    trailing ``_step`` columns are the wall-clock step, both of which we drop.
    """
    frame = pd.read_csv(path)
    value_columns = [
        column
        for column in frame.columns
        if column != "global_step"
        and not column.endswith(("__MIN", "__MAX"))
        and not column.endswith("_step")
    ]
    if len(value_columns) != 1:
        raise ValueError(f"{path.name}: expected one value column, got {value_columns}")

    out = frame[["global_step", value_columns[0]]].copy()
    out.columns = ["step", "value"]
    out = out.dropna().sort_values("step").reset_index(drop=True)
    out["step"] = out["step"].astype(float) * STEP_SCALE
    out["value"] = out["value"].astype(float) * scale
    return out


def smooth(values: pd.Series) -> pd.Series:
    if SMOOTH_WINDOW <= 1:
        return values
    return values.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()


def axis_limits(frames: list[pd.DataFrame], pad: float = 0.06) -> tuple[float, float]:
    """Common y range over several series, padded so the curves clear the frame."""
    lo = min(float(frame["value"].min()) for frame in frames)
    hi = max(float(frame["value"].max()) for frame in frames)
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


def build_data_frame(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One wide table of every plotted series, keyed by step.

    The exports of a single run share their step column, but they are merged
    rather than concatenated so that a metric logged on a different schedule
    would show up as missing values instead of silently misaligning.
    """
    merged: pd.DataFrame | None = None
    for name, frame in series.items():
        columns = frame[["step", "value", "smoothed"]].rename(
            columns={"value": name, "smoothed": f"{name}_smooth"}
        )
        merged = columns if merged is None else merged.merge(columns, on="step", how="outer")
    assert merged is not None
    return merged.sort_values("step").reset_index(drop=True)


def plot_lines(name: str, color: str, legend: str | None) -> list[str]:
    """A faint raw curve plus the smoothed curve on top of it.

    Both read their column out of the companion CSV. Only the smoothed curve
    carries the legend entry; the raw one is excluded from the legend so a
    single label covers the pair.
    """

    def table(column: str, style: str) -> str:
        return (
            f"    \\addplot [{style}] table "
            f"[x=step, y={column}, col sep=comma] {{{DATA_PATH_IN_TEX}}};"
        )

    lines = []
    if SMOOTH_WINDOW > 1:
        lines.append(
            table(
                name,
                f"draw={color}, line width=0.35pt, opacity=0.30, forget plot",
            )
        )
        column = f"{name}_smooth"
    else:
        column = name

    lines.append(table(column, f"draw={color}, line width=1.0pt"))
    if legend is not None:
        lines.append(f"    \\addlegendentry{{{legend}}}")
    return lines


def selection_line(ymin: float, ymax: float) -> list[str]:
    if SELECTED_STEP is None:
        return []
    x = SELECTED_STEP * STEP_SCALE
    return [
        "    \\addplot [selectionline, forget plot] coordinates "
        f"{{({x:.4f},{ymin:.6f}) ({x:.4f},{ymax:.6f})}};",
    ]


def axis(
    body: list[str],
    ylabel: str,
    ymin: float,
    ymax: float,
    precision: int,
    legend: bool = False,
) -> list[str]:
    options = [
        f"width={AXIS_WIDTH}",
        f"height={AXIS_HEIGHT}",
        "scale only axis",
        "xlabel={training steps ($\\times 10^{3}$)}",
        f"ylabel={{{ylabel}}}",
        f"ylabel near ticks",
        f"ylabel shift={YLABEL_SHIFT}",
        f"xlabel shift={XLABEL_SHIFT}",
        f"ymin={ymin:.6f}",
        f"ymax={ymax:.6f}",
        *(
            []
            if XMIN is None or XMAX is None
            else [f"xmin={XMIN:g}", f"xmax={XMAX:g}"]
        ),
        f"xtick distance={XTICK_DISTANCE}",
        "yticklabel style={/pgf/number format/fixed, "
        f"/pgf/number format/precision={precision}, "
        "/pgf/number format/zerofill}",
        "tick align=outside",
        "tick pos=left",
        "grid=major",
        "grid style={draw=black!12}",
        "axis line style={draw=black!45}",
    ]
    if legend:
        options += [
            "legend pos=north east",
            "legend cell align=left",
            "legend style={draw=black!25, fill=white, fill opacity=0.85, "
            "text opacity=1, font=\\scriptsize, inner sep=2pt}",
        ]
    lines = ["  \\begin{axis}["]
    lines += [f"    {option}," for option in options]
    lines.append("  ]")
    lines += body
    lines.append("  \\end{axis}")
    return lines


def main() -> None:
    dsc = read_metric(CSV_DSC, DSC_SCALE)
    mtv = read_metric(CSV_MTV, BIAS_SCALE)
    tlg = read_metric(CSV_TLG, BIAS_SCALE)
    for frame in (dsc, mtv, tlg):
        frame["smoothed"] = smooth(frame["value"])

    dsc_min, dsc_max = axis_limits([dsc])
    bio_min, bio_max = axis_limits([mtv, tlg])

    data = build_data_frame({"dsc": dsc, "mtv": mtv, "tlg": tlg})
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_DATA, index=False, float_format="%.6f")

    left = plot_lines("dsc", "colordsc", None) + selection_line(dsc_min, dsc_max)
    right = (
        plot_lines("mtv", "colormtv", "MTV bias")
        + plot_lines("tlg", "colortlg", "TLG bias")
        + selection_line(bio_min, bio_max)
    )

    lines = [
        "% Generated by paper_results/dsc_plateau/dsc_plateau.py -- do not edit.",
        f"% Run: {RUN}. Selected checkpoint: {SELECTED_STEP}.",
        f"% Curves are read from {DATA_PATH_IN_TEX} at compile time.",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\definecolor{{colordsc}}{{rgb}}{{{COLOR_DSC}}}",
        f"\\definecolor{{colormtv}}{{rgb}}{{{COLOR_MTV}}}",
        f"\\definecolor{{colortlg}}{{rgb}}{{{COLOR_TLG}}}",
        "\\tikzset{selectionline/.style={red, dashed, line width=0.8pt}}",
        "\\pgfplotsset{every axis/.append style={font=\\scriptsize, "
        "label style={font=\\scriptsize}, tick label style={font=\\tiny}}}",
        "\\begin{tikzpicture}",
    ]
    lines += axis(left, "validation DSC", dsc_min, dsc_max, DSC_PRECISION)
    # The trailing %% are load-bearing: a line break between the two pictures
    # would become an interword space, which is enough to push the pair over
    # \linewidth and wrap the second one onto its own line.
    lines.append("\\end{tikzpicture}%")
    lines.append("\\hfill%")
    lines.append("\\begin{tikzpicture}")
    lines += axis(
        right, "validation bias (\\%)", bio_min, bio_max, BIAS_PRECISION, legend=True
    )
    lines += [
        "\\end{tikzpicture}",
        f"\\caption{{{CAPTION}}}",
        f"\\label{{{FIG_LABEL}}}",
        "\\end{figure}",
        "",
    ]

    OUT_TEX.write_text("\n".join(lines))
    print(f"wrote {OUT_TEX}")
    print(f"wrote {OUT_DATA} ({len(data)} rows)")

    def summarise(
        name: str, frame: pd.DataFrame, higher_is_better: bool, unit: str = ""
    ) -> None:
        series = frame["smoothed"]
        best = series.idxmax() if higher_is_better else series.idxmin()
        step = frame["step"].iloc[best] / STEP_SCALE
        print(
            f"  {name:<4} best (smoothed) {series.iloc[best]:7.3f}{unit} "
            f"at step {step:>7.0f}"
        )
        if SELECTED_STEP is not None:
            at = int(np.argmin(np.abs(frame["step"] / STEP_SCALE - SELECTED_STEP)))
            print(
                f"       at the selected step {series.iloc[at]:7.3f}{unit} "
                f"(step {frame['step'].iloc[at] / STEP_SCALE:.0f})"
            )

    print(f"\n{len(dsc)} validation rounds, "
          f"steps {dsc['step'].min() / STEP_SCALE:.0f}"
          f"--{dsc['step'].max() / STEP_SCALE:.0f}")
    summarise("DSC", dsc, higher_is_better=True)
    summarise("MTV", mtv, higher_is_better=False, unit="%")
    summarise("TLG", tlg, higher_is_better=False, unit="%")


if __name__ == "__main__":
    main()
