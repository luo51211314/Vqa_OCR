#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""顺序消融表（全自动，读 metrics/detail，禁止手填）。

行：dataset/split × model（9 行）
列：native（主表 w/ OCR 格，order=native）/ random / ros / coordinate
每个非 native 顺序给出与 native 的配对 Δ 及 95% bootstrap CI（B=10000，
按 question_id 对齐），CI 跨 0 加 †（未观测到可测量差异）。

输出：<ablation_order 目录>/table_order/{table_order.csv,table_order.tex}
缺格一律 TBD 并在 stderr 报警。

用法:
  python scripts/make_order_ablation_table.py --run-dir results_repro/<dir>
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ci import load_scores  # noqa: E402
from ablation_ci_tex import write_ci_companion

ORDERS = ["random", "ros", "coordinate"]
ROWS = [
    ("chartqa", "test", "llava", "relaxed_accuracy"),
    ("chartqa", "test", "qwen", "relaxed_accuracy"),
    ("chartqa", "test", "mplug", "relaxed_accuracy"),
    ("docvqa", "val", "llava", "anls"),
    ("docvqa", "val", "qwen", "anls"),
    ("docvqa", "val", "mplug", "anls"),
    ("mydatavqa", "test", "llava", "token_f1"),
    ("mydatavqa", "test", "qwen", "token_f1"),
    ("mydatavqa", "test", "mplug", "token_f1"),
]

FN_RE = re.compile(
    r"^(?P<dataset>chartqa|docvqa|mydatavqa)_(?P<split>test|val)_(?P<model>llava|qwen|mplug)_"
    r"(?P<metric>relaxed_accuracy|anls|token_f1)_ocr_time(?:_(?P<order>random|ros|coordinate))?_metrics\.json$"
)


def paired_delta_ci(base_csv, other_csv, B=10000, seed=42):
    a = load_scores(base_csv)
    b = load_scores(other_csv)
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return None
    da = a.loc[common].to_numpy(float)
    db = b.loc[common].to_numpy(float)
    diff = db - da
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(B, n))
    deltas = diff[idx].mean(axis=1)
    return {
        "n": n,
        "delta": float(diff.mean()),
        "ci": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="如 results_repro/20261018_e67003c")
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args()

    main_dir = os.path.join(args.run_dir, "main_table")
    order_dir = os.path.join(args.run_dir, "ablation_order")
    out_dir = os.path.join(order_dir, "table_order")
    os.makedirs(out_dir, exist_ok=True)

    def find_metrics(directory):
        index = {}
        for p in glob.glob(os.path.join(directory, "*_metrics.json")):
            m = FN_RE.match(os.path.basename(p))
            if not m:
                continue
            key = (m["dataset"], m["split"], m["model"], m["metric"])
            index[(key, m["order"] or "native")] = p
        return index

    main_idx = find_metrics(main_dir)
    order_idx = find_metrics(order_dir)

    def score(path):
        d = json.load(open(path))
        return d.get(d.get("metric_type"))

    records = []
    warnings = []
    for row_i, row in enumerate(ROWS):
        key = row
        native_p = main_idx.get((key, "native"))
        native_csv = native_p.replace("_metrics.json", "_detail.csv") if native_p else None
        rec = {"dataset": key[0], "split": key[1], "model": key[2], "metric": key[3]}
        if native_p and os.path.exists(native_csv):
            rec["n"] = json.load(open(native_p)).get("total_samples")
            rec["native"] = score(native_p)
        else:
            warnings.append(f"缺少 native 基线: {key}")
            rec["n"] = None
            rec["native"] = None
        for od in ORDERS:
            p = order_idx.get((key, od))
            csv = p.replace("_metrics.json", "_detail.csv") if p else None
            if p and csv and os.path.exists(csv):
                rec[od] = score(p)
                if native_csv:
                    rec[od + "_stat"] = paired_delta_ci(
                        native_csv, csv, B=args.B, seed=1000 + row_i * 7 + ORDERS.index(od)
                    )
                else:
                    rec[od + "_stat"] = None
            else:
                warnings.append(f"缺少顺序格: {key} {od}")
                rec[od] = None
                rec[od + "_stat"] = None
        records.append(rec)

    # ---------- CSV ----------
    csv_lines = ["dataset,split,model,metric,n,native," +
                 ",".join(f"{o},{o}_delta,{o}_ci_lo,{o}_ci_hi" for o in ORDERS)]
    for r in records:
        cells = [r["dataset"], r["split"], r["model"], r["metric"],
                 str(r["n"] if r["n"] is not None else "TBD"),
                 f"{r['native']:.4f}" if r["native"] is not None else "TBD"]
        for o in ORDERS:
            if r[o] is None:
                cells += ["TBD", "", "", ""]
            else:
                st = r[o + "_stat"]
                if st:
                    cells += [f"{r[o]:.4f}", f"{st['delta']:+.4f}",
                              f"{st['ci'][0]:.4f}", f"{st['ci'][1]:.4f}"]
                else:
                    cells += [f"{r[o]:.4f}", "", "", ""]
        csv_lines.append(",".join(cells))
    open(os.path.join(out_dir, "table_order.csv"), "w").write("\n".join(csv_lines) + "\n")

    # ---------- TeX ----------
    L = []
    L.append(f"% generated by scripts/make_order_ablation_table.py; "
             f"delta=order-native, paired bootstrap B={args.B}; dagger=CI crosses 0")
    L.append("\\begin{tabular}{ll|c|ccc}")
    L.append("\\hline")
    L.append("Dataset & Model & native & random & ROS & coordinate \\\\")
    L.append("\\hline")
    last_ds = None
    for r in records:
        ds = r["dataset"] if r["dataset"] != last_ds else ""
        last_ds = r["dataset"]

        def cell(o):
            if r[o] is None:
                return "TBD"
            st = r[o + "_stat"]
            s = f"{r[o]:.3f}"
            if st:
                cross = st["ci"][0] <= 0 <= st["ci"][1]
                dagger = r"$\dagger$" if cross else ""
                s += f" ({st['delta']:+.3f}{dagger})"
            return s

        nat = f"{r['native']:.3f}" if r["native"] is not None else "TBD"
        L.append(f"{ds} & {r['model']} & {nat} & {cell('random')} & "
                 f"{cell('ros')} & {cell('coordinate')} \\\\")
    L.append("\\hline")
    L.append("\\end{tabular}")
    open(os.path.join(out_dir, "table_order.tex"), "w").write("\n".join(L) + "\n")

    write_ci_companion(out_dir, "table_order", "\n".join(csv_lines))
    print("written:", os.path.join(out_dir, "table_order.csv"))
    print("written:", os.path.join(out_dir, "table_order.tex"))
    for w in warnings:
        print("WARN:", w, file=sys.stderr)
    if warnings:
        sys.exit(2)


if __name__ == "__main__":
    main()
