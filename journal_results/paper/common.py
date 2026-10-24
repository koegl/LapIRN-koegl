"""Shared inputs of the paper scripts (tables / figures from the test CSVs).

Every script writes LaTeX (and its images) to OUT_DIR/<tables|figures>/, laid
out like the overleaf project, so the files can be copied over as they are.
"""

import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Code"))
sys.path.insert(0, str(REPO / "journal_results"))

import ranking  # noqa: E402

# results with the official scorer conventions (the first evaluation is in
# evaluation/; its paper outputs are backed up in out_old_scorer/)
EVAL_DIR = REPO / "evaluation_official_scorer" / "test"
# runtimes from the old-scorer runs: all on one RTX A6000 (janus) with the same
# IO (40 iterations, same weights; only the lr differed), whereas the official
# scorer runs were spread over janus and cluster GPUs
RUNTIME_DIR = REPO / "evaluation" / "test"
BASELINE_DIR = REPO / "journal_results" / "baselines"
OUT_DIR = Path(__file__).resolve().parent / "out"


# baseline comparison (plan, experiment 4). training / io: the PET and
# rigidity losses used in training and in IO ("--" in the IO column = no IO;
# "basic" = IO with similarity and regularisation terms only); short: x label
# of the plot (two lines: method, where the losses act). The ours rows are runs of experiment 1: T1 / I1 = without
# losses, T4 / I4 = both losses in training (and IO), I7 = both losses only in
# IO = "Ours", chosen on val with the selection rule (select_stage.py ours).
class Method(NamedTuple):
    label: str
    run: str
    training: str
    io: str
    short: str


PR = "PET + rig."
BASELINES: Dict[str, Method] = {
    "B1": Method("Initial", "B1", "--", "--", "Initial"),
    "B2": Method("Affine", "B2", "--", "--", "Affine"),
    "B3": Method("NiftyReg", "B3", "--", "--", "NiftyReg"),
    "B4": Method("ConvexAdam", "B4", "--", "--", "ConvexAdam"),
    # IO straight after the affine prereg, no network (what the network adds)
    "B9": Method("Affine + IO", "I8", "--", PR, "Affine + IO"),
    # network only
    "B5": Method("Ours", "T1", "--", "--", "Ours"),
    "B7": Method("Ours", "T4", PR, "--", "Ours\n(training: PET + rig.)"),
    # network + IO
    "B6": Method("Ours + IO", "I1", "--", "basic", "Ours + IO\n(IO: basic)"),
    "B8": Method("Ours + IO", "I4", PR, PR, "Ours + IO\n(both: PET + rig.)"),
    "B10": Method("Ours + IO", "I7", "--", PR, "Ours + IO\n(IO: PET + rig.)"),
}
# not ranked: its MTV / TLG change is 0 by construction (nothing resampled)
UNRANKED = ("B1",)
# reference of the significance marks
OURS = "B10"
# our rows (a rule above B5 in the table, accent colour in the plot)
OURS_IDS = ("B5", "B6", "B7", "B8", "B10")
# precomputed fields: their registration time is in the tuning timings, the
# runtime_s of the evaluation only covers loading the field
EXTERNAL_TIMINGS = {"B3": "niftyreg", "B4": "convexadam"}

# column -> (header, higher is better, scale, decimals, test column)
METRIC_COLUMNS: Dict[str, tuple] = {
    "dice_ct": (r"Dice (\%) $\uparrow$", True, 100.0, 1, "dice_ct"),
    "hd95": (r"HD95 (mm) $\downarrow$", False, 1.0, 2, "hd95"),
    "mtv_bias": (
        r"$|\Delta\mathrm{MTV}|$ (\%) $\downarrow$",
        False,
        100.0,
        1,
        "mtv_bias",
    ),
    "tlg_bias": (
        r"$|\Delta\mathrm{TLG}|$ (\%) $\downarrow$",
        False,
        100.0,
        1,
        "tlg_bias",
    ),
    # %NDV shown in ppm (1 % = 1e4 ppm)
    "ndv_percent": (r"NDV (ppm) $\downarrow$", False, 1e4, None, "ndv_adjusted"),
}


def load_run(run_id: str) -> pd.DataFrame:
    """Per-pair test CSV of an evaluation run (evaluate.py run ID)."""
    return pd.read_csv(EVAL_DIR / f"{run_id}.csv", dtype={"case_id": str})


def load_pairs(method_id: str) -> pd.DataFrame:
    return load_run(BASELINES[method_id].run)


def patient_means(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """One row per patient: mean over the patient's pairs (NaNs skipped)."""
    df = df.assign(
        ndv_adjusted=(df["ndv_percent"] - ranking.NDV_EQUIVALENCE_PERCENT).clip(
            lower=0.0
        )
    )
    return df.groupby("case_id")[columns].mean()


def runtimes(method_id: str) -> Optional[pd.Series]:
    """Per-pair registration time in s (None for the identity)."""
    if method_id in UNRANKED:
        return None
    if method_id in EXTERNAL_TIMINGS:
        method = EXTERNAL_TIMINGS[method_id]
        import json

        config = json.loads((BASELINE_DIR / method / "chosen.json").read_text())[
            "config"
        ]
        return pd.read_csv(BASELINE_DIR / method / config / "timings_test.csv")[
            "seconds"
        ]
    run = BASELINES[method_id].run
    return pd.read_csv(RUNTIME_DIR / f"{run}.csv")["runtime_s"]


def holm(p: Dict[str, float]) -> Dict[str, float]:
    names = list(p)
    adjusted = ranking.holm(np.array([p[n] for n in names]))
    return dict(zip(names, adjusted))


def fmt(value: float, decimals: Optional[int]) -> str:
    """Fixed decimals, or (None, for NDV in ppm) three significant digits
    without an exponent."""
    if decimals is not None:
        return f"{value:.{decimals}f}"
    if value == 0:
        return "0"
    return f"{float(f'{value:.3g}'):g}" if abs(value) >= 1 else f"{value:.2g}"


def two_sided(a: pd.Series, b: pd.Series, higher: bool) -> Tuple[float, bool]:
    """Two-sided paired Wilcoxon p of a vs. b (patient values), from the two
    one-sided tests, and whether a is the better one."""
    p_a = ranking.one_sided_p(a, b, higher)
    p_b = ranking.one_sided_p(b, a, higher)
    return min(1.0, 2 * min(p_a, p_b)), p_a < p_b


def table_preview(tex_path: Path) -> Path:
    """Compile a table's LaTeX into a standalone PDF next to it (for viewing in
    the repo; the manuscript inputs the .tex)."""
    import shutil
    import subprocess
    import tempfile

    tex_path = Path(tex_path)
    doc = (
        "\\documentclass[12pt]{article}\n"
        "\\usepackage{booktabs,graphicx,amsmath,amssymb}\n"
        "\\usepackage[margin=1in,letterpaper]{geometry}\n"
        "\\pagestyle{empty}\n\\begin{document}\n"
        f"\\input{{{tex_path.resolve()}}}\n\\end{{document}}\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "preview.tex").write_text(doc)
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "preview.tex"],
            cwd=tmp,
            stdout=subprocess.DEVNULL,
            check=False,
        )
        pdf = tex_path.with_suffix(".pdf")
        shutil.copy(Path(tmp) / "preview.pdf", pdf)
    return pdf
