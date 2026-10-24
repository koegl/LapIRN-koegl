"""Appendix: IO tuning on the validation set -> LaTeX table (+ PDF preview).

The two tuning stages of journal_results/official_scorer (select_stage.py,
evaluation_official_scorer/io_tuning/chosen.json): learning rate x iterations,
then a common scale of the four PET weights, on the network trained with both
losses. Selection rule: the lowest |MTV change| + |TLG change| among the
candidates within 2 Dice points of the most accurate one. Patient means.

    ../../.venv/bin/python table_io_tuning.py  ->  out/tables/io_tuning.tex
"""

import json

import common

CHOSEN = common.REPO / "evaluation_official_scorer" / "io_tuning" / "chosen.json"
STAGES = (
    ("optimiser", "Stage 1: learning rate / iterations"),
    ("pet", "Stage 2: PET weight scale (learning rate 0.025, 40 iterations)"),
)
N_COLUMNS = 8


def candidate_label(stage: str, candidate: str) -> str:
    if stage == "optimiser":
        lr, it = candidate.split("_")
        return f"{lr[2:]} / {it[2:]}"
    return r"$\times$" + candidate.split("_x")[1]


def main() -> None:
    stages = json.loads(CHOSEN.read_text())["stages"]
    lines = []
    for stage, title in STAGES:
        table = stages[stage]["table"]
        chosen = stages[stage]["chosen"]
        lines.append(rf"\multicolumn{{{N_COLUMNS}}}{{l}}{{\textit{{{title}}}}} \\")
        for row in table:
            label = candidate_label(stage, row["candidate"])
            if row["candidate"] == chosen:
                label = rf"\textbf{{{label}}}"
            lines.append(
                f"{label} & {100 * row['dice_ct']:.1f} & {row['hd95']:.2f}"
                f" & {100 * row['mtv_bias']:.1f} & {100 * row['tlg_bias']:.1f}"
                f" & {100 * row['preservation']:.1f} & {1e4 * row['ndv_percent']:.0f}"
                f" & {'yes' if row['within_tolerance'] else 'no'} \\\\"
            )
        lines.append(r"\midrule")
    body = "\n".join(lines[:-1])
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{IO tuning on the validation set (network trained with the PET and rigidity losses). Stage 1: learning rate / iterations; stage 2: common scale of the four PET weights relative to the training-derived values. Within each stage, the chosen setting (bold) has the lowest $|\Delta\mathrm{{MTV}}|+|\Delta\mathrm{{TLG}}|$ among the candidates within two Dice points of the most accurate one (``Within''). Candidates sorted by Dice; patient means. The rigidity weight was fixed at 0.25 (Sec.~\ref{{sect:tuning}}).}}
\label{{tab:io_tuning}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l{"c" * (N_COLUMNS - 1)}}}
\toprule
Candidate & Dice (\%) & HD95 (mm) & $|\Delta\mathrm{{MTV}}|$ (\%) & $|\Delta\mathrm{{TLG}}|$ (\%) & Sum (\%) & NDV (ppm) & Within \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""
    out_dir = common.OUT_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "io_tuning.tex").write_text(tex)
    common.table_preview(out_dir / "io_tuning.tex")
    print(body)


if __name__ == "__main__":
    main()
