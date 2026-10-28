#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""固定子集 × 多次运行的时延分布基准。

    python scripts/benchmark_latency.py --model_name qwen --model_type qwen \
        --dataset chartqa --split test --metric relaxed_accuracy --n 200 --repeats 3

每次运行都重新启动进程（冷启动计入），固定同一 200 个 question_id（seed=42），
off / ocr 各跑 --repeats 次；聚合各次 metrics.json 的 time_stats：
mean±std（跨运行）、p50/p95，写 latency_summary.json。
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def resolve_model_path(model_name):
    for cand in (f"/root/autodl-tmp/model/{model_name}_hug",
                 f"/root/autodl-tmp/model/{model_name}"):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"找不到模型 {model_name}")


def make_subset_ids(dataset, split, n, out_dir):
    from loaders import build_dataloader
    loader = build_dataloader(dataset, split, batch_size=1, num_workers=0)
    df = loader.dataset.df
    id_col = next(c for c in ("question_id", "questionId", "id", "sample_id")
                  if c in df.columns)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    ids = df.iloc[sorted(idx)][id_col].astype(str).tolist()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset}_{split}_lat{n}.ids")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")
    return path


def run_once(args, ids_file, variant, run_idx, out_dir):
    tag = f"{args.dataset}_{args.split}_{args.model_type}_{variant}_lat{args.n}_run{run_idx+1}"
    cmd = [
        sys.executable, "-m", "main_inference",
        "--dataset", args.dataset, "--split", args.split, "--bs", "1",
        "--model_path", resolve_model_path(args.model_name),
        "--model_type", args.model_type, "--metric_type", args.metric,
        "--time", "yes", "--use_feature_fusion", "off",
        "--output_dir", out_dir, "--seed", str(42 + run_idx),
        "--sample_ids_file", ids_file,
    ]
    if variant == "ocr":
        cmd += ["--use_experts", "manual", "--expert_names", "ocr",
                "--ocr_order", "native", "--prompt_template", "p2"]
    else:
        cmd += ["--use_experts", "off"]

    log_path = os.path.join(out_dir, tag + ".log")
    print("运行:", tag)
    with open(log_path, "w", encoding="utf-8") as logf:
        ret = subprocess.run(cmd, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        raise RuntimeError(f"{tag} 失败，见 {log_path}")
    # metrics.json 文件名按 main_inference 的 basename 规则定位
    suffix = "_ocr" if variant == "ocr" else ""
    mj = os.path.join(out_dir,
                      f"{args.dataset}_{args.split}_{args.model_type}_{args.metric}{suffix}_time_subset_metrics.json")
    with open(mj, encoding="utf-8") as f:
        return json.load(f)


def aggregate(stats_list):
    avgs = np.array([s["avg_time"] for s in stats_list])
    stds = np.array([s.get("std_time", 0.0) for s in stats_list])
    p50s = np.array([s.get("p50_time", np.nan) for s in stats_list])
    p95s = np.array([s.get("p95_time", np.nan) for s in stats_list])
    return {
        "n_runs": len(stats_list),
        "mean_over_runs": round(float(avgs.mean()), 4),
        "std_over_runs": round(float(avgs.std(ddof=0)), 4),
        "within_run_std_mean": round(float(stds.mean()), 4),
        "p50_mean": round(float(np.nanmean(p50s)), 4),
        "p95_mean": round(float(np.nanmean(p95s)), 4),
        "per_run_avg_time": [round(float(x), 4) for x in avgs],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--model_type", required=True)
    ap.add_argument("--dataset", default="chartqa")
    ap.add_argument("--split", default="test")
    ap.add_argument("--metric", default="relaxed_accuracy")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--variants", default="off,ocr")
    ap.add_argument("--outdir", default=os.path.join(REPO_ROOT, "results_repro", "latency"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ids_file = make_subset_ids(args.dataset, args.split, args.n,
                               os.path.join(REPO_ROOT, "experiments", "subsets"))

    summary = {"config": vars(args), "ids_file": ids_file, "variants": {}}
    for variant in args.variants.split(","):
        stats = []
        for r in range(args.repeats):
            mj = run_once(args, ids_file, variant, r, args.outdir)
            stats.append(mj["time_stats"])
        summary["variants"][variant] = aggregate(stats)

    out = os.path.join(args.outdir,
                       f"latency_summary_{args.dataset}_{args.split}_{args.model_type}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()
