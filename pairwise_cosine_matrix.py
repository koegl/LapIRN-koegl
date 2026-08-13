"""Pairwise gradient-cosine matrix for one wandb run.

Pulls the `grad_conflict_lvl3/cos_<x>_vs_<y>` history logged by
`utils.gradient_conflict_report`, averages each pair over a window of the run,
and renders the full symmetric term-vs-term matrix as a heatmap.

What to look for is block structure: terms that cluster into groups which are
internally positive and mutually negative. If those blocks match the
accuracy/tumour split the training code uses, a two-stage design has the right
shape; if the clustering cuts across it, the stage boundary is in the wrong
place.

Usage:
    python pairwise_cosine_matrix.py                       # defaults below
    python pairwise_cosine_matrix.py --run <name-or-id> --last 50
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PREFIX = "grad_conflict_lvl3"

# Objective terms first, then the diagnostic probes. The order is the plot
# order and carries no grouping claim: the point of the matrix is to let the
# block structure emerge from the data rather than assuming it. The only line
# drawn separates objective terms from probes, which really are different
# things (probes are not part of the optimised total).
TERMS = [
    "ncc",
    "dice",
    "jacob",
    "smooth",
    "rigidity",
    "tlg",
    "jactum",
    "dice_bone",
    "dice_soft",
]
PROBES = {"dice_bone", "dice_soft"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="wise-swan-39232655", help="run display name or id")
    p.add_argument("--project", default="PSMAReg_LapIRN")
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument(
        "--last",
        type=int,
        default=50,
        help="average over the last N logged points (0 = whole run)",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="use the per-step cosines instead of the windowed `_mean` keys",
    )
    p.add_argument("--out", default="pairwise_cosine_matrix.png")
    return p.parse_args()


def resolve_run(api, entity: str | None, project: str, run: str):
    """Accept either a wandb run id or a display name."""
    path = f"{entity}/{project}" if entity else project
    try:
        return api.run(f"{path}/{run}")
    except Exception:
        matches = list(api.runs(path, filters={"display_name": run}))
        if not matches:
            sys.exit(f"no run named or id'd {run!r} in {path}")
        if len(matches) > 1:
            ids = ", ".join(r.id for r in matches)
            sys.exit(f"{run!r} is ambiguous, pass an id instead: {ids}")
        return matches[0]


def pair_key(a: str, b: str, suffix: str) -> list[str]:
    """Both orderings -- which one exists depends on dict order at log time."""
    return [f"{PREFIX}/cos_{a}_vs_{b}{suffix}", f"{PREFIX}/cos_{b}_vs_{a}{suffix}"]


def build_matrix(hist: pd.DataFrame, suffix: str, last: int):
    n = len(TERMS)
    mat = np.full((n, n), np.nan)
    counts = np.zeros((n, n), dtype=int)
    missing: list[str] = []

    for i, a in enumerate(TERMS):
        for j, b in enumerate(TERMS[i + 1 :], start=i + 1):
            col = next((k for k in pair_key(a, b, suffix) if k in hist.columns), None)
            if col is None:
                missing.append(f"{a} vs {b}")
                continue
            series = hist[col].dropna()
            if last:
                series = series.tail(last)
            if series.empty:
                missing.append(f"{a} vs {b}")
                continue
            mat[i, j] = mat[j, i] = series.mean()
            counts[i, j] = counts[j, i] = len(series)

    np.fill_diagonal(mat, 1.0)
    return mat, counts, missing


def presence(hist: pd.DataFrame, last: int) -> dict[str, float]:
    """Fraction of steps each term was live -- the sample base behind its row."""
    out = {}
    for term in TERMS:
        col = f"{PREFIX}/present_{term}_mean"
        if col in hist.columns:
            series = hist[col].dropna()
            if last:
                series = series.tail(last)
            if not series.empty:
                out[term] = float(series.mean())
    return out


def plot(mat, counts, present, run_name, n_points, out_path):
    labels = [
        f"{t}\n({present[t]:.0%} live)" if t in present and present[t] < 0.99 else t
        for t in TERMS
    ]

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(TERMS)), labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(TERMS)), labels, fontsize=9)

    for i in range(len(TERMS)):
        for j in range(len(TERMS)):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "--", ha="center", va="center", color="0.5")
                continue
            shade = "white" if abs(mat[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color=shade, fontsize=8)

    # one separator: optimised objective terms | diagnostic probes
    split = len([t for t in TERMS if t not in PROBES]) - 0.5
    ax.axhline(split, color="black", lw=1.5)
    ax.axvline(split, color="black", lw=1.5)

    fig.colorbar(im, ax=ax, shrink=0.8, label="cosine of gradients")
    ax.set_title(
        f"Gradient-cosine matrix -- {run_name}\n"
        f"averaged over {n_points} logged points "
        f"(objective terms | probes)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")


def main() -> None:
    args = parse_args()
    import wandb

    api = wandb.Api()
    run = resolve_run(api, args.entity, args.project, args.run)
    suffix = "" if args.raw else "_mean"

    keys = [k for k in run.summary.keys() if k.startswith(f"{PREFIX}/")]
    if not keys:
        sys.exit(f"run {run.name} logged no {PREFIX}/ metrics")
    hist = run.history(keys=keys, pandas=True, samples=100_000)

    mat, counts, missing = build_matrix(hist, suffix, args.last)
    present = presence(hist, args.last)

    frame = pd.DataFrame(mat, index=TERMS, columns=TERMS)
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    frame.to_csv(csv_path)
    print(frame.round(2).to_string(), "\n")
    print(f"wrote {csv_path}")

    if missing:
        print(f"no data for: {', '.join(missing)}")
    thin = [
        f"{TERMS[i]} vs {TERMS[j]} ({counts[i, j]})"
        for i in range(len(TERMS))
        for j in range(i + 1, len(TERMS))
        if 0 < counts[i, j] < 10
    ]
    if thin:
        print(f"fewer than 10 points, treat as noise: {', '.join(thin)}")

    plot(mat, counts, present, run.name, args.last or len(hist), args.out)


if __name__ == "__main__":
    main()
