"""Derive sel_scale_* constants for lvl3 checkpoint selection from wandb CSV exports.

Export one CSV per metric from the wandb UI (val_dice_ct, val_mtv_bias,
val_tlg_bias), then either fill in the three paths below and run with no
arguments, or pass them on the command line:

    python calibrate.py dice.csv mtv.csv tlg.csv

Prints the config lines to paste into TrainingConfig, plus a report of where
each selection criterion would have picked its checkpoint. If a CSV holds
several runs, set RUN to a substring of the run name to pick one.

Pure stdlib on purpose: no numpy/pandas needed, and the exported CSV is the
same data the UI shows -- unlike the wandb history API, which downsamples to
~500 points by default and refuses step scans on runs with no "_step" column.
"""

import csv
import statistics
import sys

# --- fill these in (or pass the three paths as arguments) -------------------
CSV_DICE = "/home/iml/fryderyk.koegl/code/LapIRN-koegl/Code/dsc.csv"
CSV_MTV = "/home/iml/fryderyk.koegl/code/LapIRN-koegl/Code/mtv.csv"
CSV_TLG = "/home/iml/fryderyk.koegl/code/LapIRN-koegl/Code/tlg.csv"
RUN = None  # substring of the run name, when a CSV holds more than one run
# ---------------------------------------------------------------------------

W = {"dice": 0.5, "mtv": 0.25, "tlg": 0.25}
STEP_NAMES = ("step", "_step", "global_step")
# the metric each CSV is expected to contain. Matching on these rather than on
# "the only non-step column" matters: an export usually carries several series
# per run (train_lvl3/dice_ct_epoch, a metric literally named _step, ...), and
# the train-side twin must not be picked up instead of the validation one.
METRICS = ("valid_lvl3/val_dice_ct", "valid_lvl3/val_mtv_bias", "valid_lvl3/val_tlg_bias")


def read_metric_csv(path, metric, run=None):
    """Return {step: value} for `metric` from a wandb CSV export."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path}: empty")

    cols = list(rows[0])
    # the step column is a bare "Step"; a column merely *ending* in _step is a
    # logged metric of that name belonging to some run, not the index
    step_col = next((c for c in cols if c.strip().lower() in STEP_NAMES), None)
    if step_col is None:
        raise SystemExit(f"{path}: no step column among {cols}")

    tail = metric.split("/")[-1]
    # wandb adds __MIN/__MAX band columns next to the real series; drop them
    value_cols = [
        c
        for c in cols
        if c != step_col
        and "__MIN" not in c
        and "__MAX" not in c
        and (metric in c or c.endswith(tail))
    ]
    if run is not None:
        value_cols = [c for c in value_cols if run in c]
    if not value_cols:
        raise SystemExit(
            f"{path}: no column for {metric!r} (RUN={run!r}).\ncolumns:\n  "
            + "\n  ".join(cols)
        )
    if len(value_cols) > 1:
        raise SystemExit(
            f"{path}: {len(value_cols)} columns match {metric!r} -- set RUN to "
            "a substring of the run you want:\n  " + "\n  ".join(value_cols)
        )

    col = value_cols[0]
    out = {}
    for r in rows:
        raw, step_raw = r.get(col), r.get(step_col)
        if raw in (None, "") or step_raw in (None, ""):
            continue  # wandb leaves blanks where a metric was not logged
        try:
            out[int(float(step_raw))] = float(raw)
        except ValueError:
            continue
    if not out:
        raise SystemExit(f"{path}: column {col!r} had no numeric rows")
    print(f"  {path}: column {col!r}, {len(out)} points")
    return out


def load(paths, run=None):
    dice, mtv, tlg = (
        read_metric_csv(p, metric, run) for p, metric in zip(paths, METRICS)
    )
    steps = sorted(set(dice) & set(mtv) & set(tlg))
    if not steps:
        raise SystemExit("the three CSVs share no common steps")
    for name, series in (("dice", dice), ("mtv", mtv), ("tlg", tlg)):
        if len(series) != len(steps):
            print(f"  note: {name} had {len(series)} points, {len(steps)} in common")
    return (
        steps,
        [dice[s] for s in steps],
        [mtv[s] for s in steps],
        [tlg[s] for s in steps],
    )


def main(argv):
    paths = [CSV_DICE, CSV_MTV, CSV_TLG]
    if len(paths) != 3 or not all(paths):
        raise SystemExit(__doc__)

    steps, d, m, t = load(paths, RUN)
    # drop the first quarter: the early transient inflates the spread
    cut = len(steps) // 4
    steps, d, m, t = steps[cut:], d[cut:], m[cut:], t[cut:]
    print(f"\nusing {len(steps)} evals after warmup (steps {steps[0]}-{steps[-1]})")

    s = {
        "dice": statistics.pstdev(d),
        "mtv": statistics.pstdev(m),
        "tlg": statistics.pstdev(t),
    }
    if min(s.values()) <= 0:
        raise SystemExit(f"a metric has zero spread, cannot normalise: {s}")

    print("\n# paste into TrainingConfig")
    print(f"sel_scale_dice_ct: float = {s['dice']:.4g}")
    print(f"sel_scale_mtv: float = {s['mtv']:.4g}")
    print(f"sel_scale_tlg: float = {s['tlg']:.4g}")

    tum = [W["mtv"] * mi / s["mtv"] + W["tlg"] * ti / s["tlg"] for mi, ti in zip(m, t)]
    comb = [W["dice"] * di / s["dice"] + tu for di, tu in zip(d, tum)]

    print("\n# where each criterion would select")
    for label, arr in (("dice", d), ("tumour", tum), ("combined", comb)):
        i = min(range(len(arr)), key=arr.__getitem__)
        print(
            f"  best {label:8s} @ step {steps[i]:>8d}"
            f"  dice_ct {d[i]:.4f}  mtv {m[i]:.4f}  tlg {t[i]:.4f}"
        )

    # --- weight sweep ------------------------------------------------------
    # score(a) = (1-a)*D + a*T with D = dice/s_d and T the tumour half. For a
    # fixed checkpoint that is a straight line in a, so the winner is the lower
    # envelope of len(steps) lines: only a handful of checkpoints can ever win,
    # and each wins on one contiguous interval of a. Evaluating just those
    # candidates is enough to choose a.
    D = [W["dice"] * di / s["dice"] for di in d]
    T = tum
    sweep, prev = [], None
    for k in range(1001):
        a = k / 1000
        i = min(range(len(steps)), key=lambda j: (1 - a) * D[j] + a * T[j])
        if i != prev:
            sweep.append([a, a, i])
            prev = i
        else:
            sweep[-1][1] = a

    print("\n# tumour share a -> selected checkpoint (only these can ever win)")
    for lo, hi, i in sweep:
        print(
            f"  a {lo:.3f}-{hi:.3f}  step {steps[i]:>8d}"
            f"  dice_ct {d[i]:.4f}  mtv {m[i]:.4f}  tlg {t[i]:.4f}"
        )
    print(
        "\n# a is the tumour half of the weights: sel_w_dice = 1-a,"
        " sel_w_mtv = sel_w_tlg = a/2"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
