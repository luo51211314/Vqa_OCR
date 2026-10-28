#!/usr/bin/env python3
"""Build the deterministic full main-table matrix used by queue simulation.

This is a CPU-only postprocess. It never launches inference and never samples:
each matrix row is one dataset/model pair, and off/P2 values come from exactly
the same full-evaluation artifacts used by Table 3.
"""
import argparse
import json
import os
import sys

import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import run_config as rc  # noqa: E402


def load_cell(job, out_dir):
    path = rc.expected_metrics_path(job, out_dir)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        metrics = json.load(f)
    detail_path = path.replace("_metrics.json", "_detail.csv")
    times = pd.Series(dtype=float)
    if os.path.exists(detail_path):
        detail = pd.read_csv(detail_path)
        for key in ("inference_time", "time", "latency"):
            if key in detail.columns:
                times = pd.to_numeric(detail[key], errors="coerce").dropna()
                break
    time_stats = metrics.get("time_stats") or {}
    avg = time_stats.get("avg_time")
    return {
        "score": metrics.get(job["metric"]),
        "n": (metrics.get("run_info") or {}).get(
            "n_samples_eval", metrics.get("total_samples")),
        "t_mean": float(times.mean()) if len(times) else avg,
        "t_p50": (float(times.quantile(0.50)) if len(times)
                  else time_stats.get("p50_time")),
        "t_p95": (float(times.quantile(0.95)) if len(times)
                  else time_stats.get("p95_time")),
        "metrics_path": os.path.relpath(path, os.path.dirname(out_dir)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(REPO, args.run_dir)
    main_dir = os.path.join(run_dir, "main_table")
    outdir = args.outdir or os.path.join(run_dir, "offline_matrix")
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(REPO, "experiments", "configs", "main_table.yaml"),
              encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cells = {}
    for job in rc.expand_jobs(cfg):
        cells[(job["dataset"], job["model_key"], job["variant"])] = load_cell(job, main_dir)

    rows = []
    for dataset in ("chartqa", "docvqa", "mydatavqa"):
        for model in ("llava", "qwen", "mplug"):
            off = cells.get((dataset, model, "off"))
            ocr = cells.get((dataset, model, "ocr"))
            row = {"dataset": dataset, "model": model}
            for prefix, cell in (("base", off), ("ocr", ocr)):
                for key in ("score", "n", "t_mean", "t_p50", "t_p95", "metrics_path"):
                    row[f"{prefix}_{key}"] = cell.get(key) if cell else None
            row["delta_score"] = (ocr["score"] - off["score"]
                                  if off and ocr and off["score"] is not None
                                  and ocr["score"] is not None else None)
            row["delta_t_mean"] = (ocr["t_mean"] - off["t_mean"]
                                   if off and ocr and off["t_mean"] is not None
                                   and ocr["t_mean"] is not None else None)
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "main_table_offline_matrix.csv")
    json_path = os.path.join(outdir, "main_table_offline_matrix.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"source": "full main-table artifacts", "sampled": False,
                   "rows": df.where(pd.notna(df), None).to_dict("records")},
                  f, ensure_ascii=False, indent=2)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
