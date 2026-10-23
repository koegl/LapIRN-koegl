"""Supplement: IO tuning on the validation set -> LaTeX table (+ PDF preview).

The two tuning stages of journal_results/tune_io.py (io_tuning/chosen.json):
learning rate x iterations, then a common scale of the four PET weights, on
the network trained with both losses; candidates ranked with the challenge's
significance-based ranking (rank, final score, accuracy / preservation /
regularity scores) and their patient means.

    ../../.venv/bin/python table_io_tuning.py  ->  out/tables/io_tuning.tex
"""

import json

import common

CHOSEN = common.REPO / "journal_results" / "io_tuning" / "chosen.json"
STAGES = (
    ("optimiser", "Learning rate and iterations"),
    ("pet", "PET weight scale (at the chosen learning rate and iterations)"),
)


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
        lines.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{title}}}}} \\")
        for row in table:
            label = candidate_label(stage, row["candidate"])
            if row["candidate"] == chosen:
                label = rf"\textbf{{{label}}}"
            lines.append(
                f"{label} & {row['rank']} & {row['final']:.1f}"
                f" & {row['accuracy']:.2f} & {row['biomarker']:.2f} & {row['regularity']:.2f}"
                f" & {100 * row['dice_ct']:.1f} & {row['hd95']:.2f}"
                f" & {100 * row['mtv_bias']:.1f} & {100 * row['tlg_bias']:.1f} \\\\"
            )
        lines.append(r"\midrule")
    body = "\n".join(lines[:-1])
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{IO tuning on the validation set (network trained with the PET and rigidity losses). Stage 1: learning rate / iterations; stage 2: common scale of the four PET weights relative to the training-derived values. Candidates of a stage are ranked against each other with the significance-based ranking of the challenge (final score $100\cdot\mathrm{{acc}}^{{0.4}}\,\mathrm{{PET}}^{{0.4}}\,\mathrm{{reg}}^{{0.2}}$); chosen setting in bold. Metrics: patient means. The rigidity weight was fixed at 0.25 (Sec.~\ref{{sect:tuning}}).}}
\label{{tab:io_tuning}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccccccc}}
\toprule
Candidate & Rank & Score & Acc. & PET & Reg. & Dice (\%) & HD95 (mm) & $|\Delta\mathrm{{MTV}}|$ (\%) & $|\Delta\mathrm{{TLG}}|$ (\%) \\
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
