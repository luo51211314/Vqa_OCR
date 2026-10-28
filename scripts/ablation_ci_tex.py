"""CI companion tables mirror the corresponding main-table axes."""
import csv
import io
import os
from itertools import groupby


def write_ci_companion(out_dir, name, csv_text):
    reader = csv.DictReader(io.StringIO(csv_text))
    fields = reader.fieldnames or []
    variants = [f[:-6] for f in fields if f.endswith("_ci_lo")
                and f[:-6] + "_ci_hi" in fields]
    if not variants:
        return
    rows = list(reader)

    def esc(text):
        return str(text).replace("_", r"\_")

    def cell(row, variant):
        values = [row.get(variant + suffix, "") for suffix in
                  ("_delta", "_ci_lo", "_ci_hi")]
        if not all(values):
            return "TBD"
        delta, lo, hi = values
        return (r"\shortstack{" + delta + r"\\{\scriptsize [" +
                lo + ", " + hi + "]}}")

    def label(variant):
        if variant.startswith("tau"):
            return r"$\tau=" + variant[3:] + "$"
        if variant.startswith("b") and variant[1:].isdigit():
            return variant[1:]
        return {"p1": "P1", "p2": "P2 (fixed)", "p3": "P3",
                "ros": "ROS"}.get(variant, esc(variant))

    lines = [
        r"% Requires booktabs and multirow. Layout follows the companion main table.",
        r"% Variant cells: delta on first line, paired-bootstrap 95% CI below.",
        r"% Bounds are copied from CSV at four decimals; no new statistical test.",
        r"\begingroup", r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.2}",
    ]
    if name == "table_template_full":
        datasets = ("chartqa", "docvqa", "mydatavqa")
        models = ("llava", "qwen", "mplug")
        index = {(r["dataset"], r["model"]): r for r in rows}
        lines += [
            r"\begin{tabular}{l*{9}{c}}", r"\toprule",
            r"Setting & \multicolumn{3}{c}{ChartQA} & \multicolumn{3}{c}{DocVQA} & \multicolumn{3}{c}{MyDataVQA} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
            " & " + " & ".join(["LLaVA", "Qwen", "mPLUG"] * 3) + r" \\",
            r"\midrule",
            "w/o OCR & " + " & ".join(["---"] * 9) + r" \\",
        ]
        for variant in variants:
            lines.append(" & ".join(["OCR-" + label(variant)] +
                [cell(index.get((ds, model), {}), variant)
                 for ds in datasets for model in models]) + r" \\")
        note = "Baseline: w/o OCR. Baseline-row dashes indicate self-comparison."
    else:
        identifiers = ["dataset", "model"] if "dataset" in fields else ["model"]
        if name == "table_order":
            base_fields = [("native", "native")]
            note = "Baseline: native OCR-P2."
        elif name == "table_l2":
            base_fields = [("n", "$n$"), ("l1", "L1 (no filter)")]
            note = "Baseline: L1 OCR-P2."
        elif name.startswith("table_budget"):
            base_fields = [("n", "$n$")]
            if "off" in fields:
                base_fields.append(("off", "w/o"))
            base_fields.append(("inf", r"$\infty$"))
            note = "Baseline: unlimited-budget OCR-P2."
        else:
            base_fields = [("n", "$n$"), ("off", "w/o OCR")]
            note = "Baseline: w/o OCR."
        spec = "l" * len(identifiers) + "c" * (len(base_fields) + len(variants))
        lines += [r"\begin{tabular}{" + spec + "}", r"\toprule",
                  " & ".join([x.title() for x in identifiers] +
                    [title for _, title in base_fields] +
                    [label(v) for v in variants]) + r" \\", r"\midrule"]
        groups = [(ds, list(rs)) for ds, rs in groupby(rows, lambda r:r.get("dataset", ""))]
        for group_i, (dataset, group) in enumerate(groups):
            if group_i:
                lines.append(r"\midrule")
            for row_i, row in enumerate(group):
                cells = []
                if "dataset" in identifiers:
                    cells.append(r"\multirow{" + str(len(group)) + "}{*}{" +
                                 esc(dataset) + "}" if row_i == 0 else "")
                cells.append(esc(row["model"]))
                for key, _ in base_fields:
                    value = row.get(key, "") or "TBD"
                    cells.append(value)
                cells.extend(cell(row, v) for v in variants)
                lines.append(" & ".join(cells) + r" \\")
        note += " Baseline columns retain the main-table scores/sample counts."
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup",
              r"\par\smallskip{\footnotesize " + note +
              r" Variant cells: $\Delta$ [95\% CI]. TBD: pending result.}"]
    path = os.path.join(out_dir, name + "_ci.tex")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print("written:", path)
