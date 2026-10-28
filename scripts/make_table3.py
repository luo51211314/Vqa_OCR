#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 results_repro 自动生成论文 Table 3（禁止任何人工填数）。

流程：
  1. 递归扫描 <root> 下所有 *_metrics.json（取每格最新一个）；
  2. 只用 run_info 标记为 主表配置（ocr: order=native/template=p2/预算空；off）的格；
  3. 校验样本量（ChartQA=2500, DocVQA=5349）与 git commit 一致性，不一致报警；
  4. 读同目录 detail CSV 做配对 bootstrap（B=10000）+ McNemar；
  5. 输出 table3.csv / table3.tex；缺格（MyDataVQA 或未跑完）一律 TBD。

用法：
  python scripts/make_table3.py --root results_repro/20260601_abc123 \
      --outdir paper_tables
  python scripts/make_table3.py --root results_repro/20260601_abc123 --allow-missing
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.bootstrap_ci import (bootstrap_ci, mcnemar_exact_p,
                                  wilcoxon_signed_rank_p)  # noqa: E402

DATASETS = [
    ("chartqa", "test", "relaxed_accuracy", 2500),
    ("docvqa", "val", "anls", 5349),
    ("mydatavqa", "test", "token_f1", 400),
]
MODELS = ["llava", "qwen", "mplug"]
B = 10000


def find_cells(root):
    """返回 {(dataset, model, variant): metrics_dict路径信息}，同键取时间戳最新。"""
    cells = {}
    for path in glob.glob(os.path.join(root, "**", "*_metrics.json"), recursive=True):
        # 排除隔离/夜跑日志等下划线目录（_quarantine*、_night）与输出目录 table3
        rel_parts = os.path.relpath(path, root).split(os.sep)
        if any(part.startswith(("_", ".")) for part in rel_parts[:-1]) \
                or "table3" in rel_parts[:-1]:
            continue
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        ri = m.get("run_info")
        if ri is None:
            continue  # 旧产物无溯源信息，不允许进主表
        model = ri["model_type"]
        dataset = ri["dataset"]
        split = ri["split"]
        variant = "off" if m.get("use_experts") in ("off", "none", None) else "ocr"
        phi = ri.get("phi") or {}
        if phi.get("sample_ids_file"):
            continue  # no dev/subset rows in the full main table
        if variant == "ocr":
            if phi.get("order") != "native" or phi.get("template") != "p2" \
                    or phi.get("budget_tokens") or phi.get("conf_threshold") \
                    or phi.get("dedup"):
                continue  # exclude ablations
        key = (dataset, split, model, variant)
        ts = ri.get("timestamp", "")
        if key not in cells or ts > cells[key][0]:
            cells[key] = (ts, path, m)
    return cells


def detail_scores(metrics_path, basename):
    csv_path = os.path.join(os.path.dirname(metrics_path), basename + "_detail.csv")
    df = pd.read_csv(csv_path)
    df["question_id"] = df["question_id"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    return df[["question_id", "score"]].drop_duplicates("question_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(args.root, "table3")
    os.makedirs(outdir, exist_ok=True)

    cells = find_cells(args.root)
    rng = np.random.RandomState(42)
    warnings = []
    commits = set()
    rows = []

    # 审计：逐样本 OOM 跳过必须显式可见（跳过样本计 0 分，n 不变）
    for (_ts, path, m) in cells.values():
        ri = m.get("run_info") or {}
        n_skip = ri.get("n_oom_skipped") or 0
        if n_skip:
            warnings.append(f"OOM 跳过样本: {ri.get('dataset')}/{ri.get('model_type')} "
                            f"w/{'OCR' if m.get('use_experts') not in ('off','none',None) else 'off'} "
                            f"跳过 {n_skip} 个（计0分，n保持不变），见 {os.path.basename(path)}")

    for dataset, split, metric, expected_n in DATASETS:
        for model in MODELS:
            row = {"dataset": dataset, "model": model, "metric": metric}
            off_item = cells.get((dataset, split, model, "off"))
            ocr_item = cells.get((dataset, split, model, "ocr"))

            # 两侧都还没跑
            if not off_item and not ocr_item:
                row.update(off="TBD", ocr="TBD", delta="TBD", delta_ci="TBD",
                           p_mcnemar="TBD", lat_off="TBD", lat_ocr="TBD")
                warnings.append(f"缺格: {dataset}/{split}/{model} (off=无, ocr=无)")
                rows.append(row)
                continue

            # 配对未齐：已完成侧先给点估计+自身 CI，Δ/显著性 TBD
            if not off_item or not ocr_item:
                done_item = off_item or ocr_item
                _, mp, mm = done_item
                side = "off" if off_item else "ocr"
                point = mm.get(metric)
                base = os.path.basename(mp)[:-len("_metrics.json")]
                try:
                    df1 = detail_scores(mp, base)
                    side_ci = bootstrap_ci(df1["score"].values, B,
                                             np.random.RandomState(42))["ci95"]
                    ci_str = f"[{side_ci[0]:.4f},{side_ci[1]:.4f}]"
                except Exception:
                    ci_str = "TBD"
                ts = mm.get("time_stats", {})
                lat = round(ts["avg_time"], 3) if ts.get("avg_time") is not None else "TBD"
                row.update(
                    off=round(point, 4) if side == "off" and point is not None else "TBD",
                    ocr=round(point, 4) if side == "ocr" and point is not None else "TBD",
                    off_ci=ci_str if side == "off" else "",
                    ocr_ci=ci_str if side == "ocr" else "",
                    delta="TBD", delta_ci="TBD", p_mcnemar="TBD",
                    lat_off=lat if side == "off" else "TBD",
                    lat_ocr=lat if side == "ocr" else "TBD",
                )
                warnings.append(f"配对未齐: {dataset}/{split}/{model} "
                                f"(off={'有' if off_item else '无'}, ocr={'有' if ocr_item else '无'})")
                rows.append(row)
                continue

            _, off_path, off_m = off_item
            _, ocr_path, ocr_m = ocr_item
            for m in (off_m, ocr_m):
                if m.get("run_info", {}).get("git_commit"):
                    commits.add(m["run_info"]["git_commit"])
            n_eval = off_m["run_info"].get("n_samples_eval")
            if expected_n and n_eval != expected_n:
                warnings.append(f"{dataset}/{model} 样本量={n_eval}，期望 {expected_n}")

            off_point = off_m.get(metric)
            ocr_point = ocr_m.get(metric)
            if off_point is None or ocr_point is None:
                warnings.append(f"{dataset}/{model} metrics.json 中找不到指标键 {metric}")
                row.update(off="TBD", ocr="TBD", delta="TBD", ci="TBD", p_mcnemar="TBD")
                rows.append(row)
                continue

            off_base = os.path.basename(off_path)[:-len("_metrics.json")]
            ocr_base = os.path.basename(ocr_path)[:-len("_metrics.json")]
            off_df = detail_scores(off_path, off_base).set_index("question_id")["score"]
            ocr_df = detail_scores(ocr_path, ocr_base).set_index("question_id")["score"]
            common = off_df.index.intersection(ocr_df.index)
            a = off_df.loc[common].values
            b_ = ocr_df.loc[common].values

            # 每行独立确定性种子：缺格/调用顺序不影响本行 CI，保证可复现
            row_rng = np.random.RandomState(42)
            off_ci = bootstrap_ci(a, B, row_rng)["ci95"]
            ocr_ci = bootstrap_ci(b_, B, row_rng)["ci95"]
            n = len(common)
            idx = row_rng.randint(0, n, size=(B, n))
            d_means = (b_ - a)[idx].mean(axis=1)
            d_lo, d_hi = float(np.percentile(d_means, 2.5)), float(np.percentile(d_means, 97.5))
            if dataset == "chartqa":
                b01 = int(np.sum(~(a >= 1) & (b_ >= 1)))
                b10 = int(np.sum((a >= 1) & ~(b_ >= 1)))
                p_mc = p_paired = mcnemar_exact_p(b01, b10)
                paired_test = "McNemar exact"
            else:
                b01 = b10 = None
                p_mc = "N/A"
                p_paired = wilcoxon_signed_rank_p(a, b_)
                if p_paired is None:
                    p_paired = "N/A"
                paired_test = "Wilcoxon signed-rank"

            def _lat(mm):
                t = mm.get("time_stats", {})
                return round(t["avg_time"], 3) if t.get("avg_time") is not None else "TBD"
            row.update(
                n=n,
                off=round(off_point, 4),
                ocr=round(ocr_point, 4),
                off_ci=f"[{off_ci[0]:.4f},{off_ci[1]:.4f}]",
                ocr_ci=f"[{ocr_ci[0]:.4f},{ocr_ci[1]:.4f}]",
                delta=round(ocr_point - off_point, 4),
                delta_ci=f"[{d_lo:.4f},{d_hi:.4f}]",
                delta_crosses_zero=bool(d_lo <= 0.0 <= d_hi),
                p_mcnemar=p_mc,
                p_paired=p_paired,
                paired_test=paired_test,
                mcnemar_b_c=(b01, b10),
                lat_off=_lat(off_m),
                lat_ocr=_lat(ocr_m),
                off_source=os.path.relpath(off_path, args.root),
                ocr_source=os.path.relpath(ocr_path, args.root),
            )
            rows.append(row)

    # ---- 输出 ----
    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "table3.csv")
    df.to_csv(csv_path, index=False)

    def _cell(v, fmt="{:.4f}"):
        return "TBD" if v == "TBD" else (fmt.format(v) if isinstance(v, (int, float)) else str(v))

    def _p_cell(v):
        if v in ("TBD", "N/A") or v is None:
            return str(v)
        p = float(v)
        if p < 0.0001:
            return r"$<0.0001$"
        return f"{p:.4f}"

    tex_lines = [
        r"\begin{tabular}{ll|ccc|cc|cc}",
        r"\hline",
        r"Dataset & Model & w/o OCR & w/ OCR & $\Delta$ & 95\% CI of $\Delta$ & Paired-test $p$"
        r" & Lat$_{\rm off}$ (s) & Lat$_{\rm ocr}$ (s) \\",
        r"\hline",
    ]
    for _, r in df.iterrows():
        if r["off"] == "TBD" and r["ocr"] == "TBD":
            tex_lines.append(
                f"{r['dataset']} & {r['model']} & TBD & TBD & TBD & TBD & TBD & "
                f"{_cell(r.get('lat_off','TBD'),'{:.3f}')} & "
                f"{_cell(r.get('lat_ocr','TBD'),'{:.3f}')} \\\\")
        elif r["delta"] == "TBD":
            # 配对未齐：只显示已完成侧
            tex_lines.append(
                f"{r['dataset']} & {r['model']} & {_cell(r['off'])} & {_cell(r['ocr'])} & "
                f"TBD & TBD & TBD & "
                f"{_cell(r.get('lat_off','TBD'),'{:.3f}')} & "
                f"{_cell(r.get('lat_ocr','TBD'),'{:.3f}')} \\\\")
        else:
            d = r["delta"]
            d_str = f"{d:+.4f}"
            if r["delta_crosses_zero"]:
                d_str += r"$^\dagger$"
            tex_lines.append(
                f"{r['dataset']} & {r['model']} & {r['off']:.4f} & {r['ocr']:.4f} & "
                f"{d_str} & {r['delta_ci']} & {_p_cell(r.get('p_paired', r['p_mcnemar']))} & "
                f"{_cell(r.get('lat_off','TBD'),'{:.3f}')} & "
                f"{_cell(r.get('lat_ocr','TBD'),'{:.3f}')} \\\\")
    tex_lines += [
        r"\hline",
        r"\end{tabular}",
        r"",
        r"% $^\dagger$ CI crosses 0: no measurable gain observed.",
        r"% Paired-test p: exact McNemar for binary ChartQA relaxed accuracy;",
        r"% two-sided Wilcoxon signed-rank for DocVQA ANLS and MyDataVQA token-F1.",
        f"% generated by scripts/make_table3.py; commits: {sorted(commits)}",
    ]
    tex_path = os.path.join(outdir, "table3.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")

    # Reviewer-facing MyDataVQA metrics stay separate from the compact main table.
    my_rows = []
    metric_keys = ["token_precision", "token_recall", "token_f1",
                   "kc_recall", "rouge_l_f1", "avg_output_tokens"]
    for model in MODELS:
        for variant in ("off", "ocr"):
            item = cells.get(("mydatavqa", "test", model, variant))
            rec = {"model": model,
                   "setting": "w/o OCR" if variant == "off" else "OCR-P2"}
            if item:
                m = item[2]
                rec["n"] = (m.get("run_info") or {}).get(
                    "n_samples_eval", m.get("total_samples"))
                for key in metric_keys:
                    rec[key] = m.get(key, "TBD")
            else:
                rec.update({"n": "TBD",
                            **{key: "TBD" for key in metric_keys}})
            my_rows.append(rec)
    my_df = pd.DataFrame(my_rows)
    my_csv = os.path.join(outdir, "table_mydatavqa_metrics.csv")
    my_df.to_csv(my_csv, index=False)
    my_tex_lines = [
        r"\begin{tabular}{ll|c|ccccc|c}",
        r"\hline",
        r"Model & Setting & $n$ & Token-P & Token-R & Token-F1 & KC-R & ROUGE-L F1 & Avg. tokens \\",
        r"\hline",
    ]
    for _, r in my_df.iterrows():
        vals = [_cell(r[k]) for k in metric_keys[:-1]]
        avg_len = _cell(r["avg_output_tokens"], "{:.2f}")
        my_tex_lines.append(f"{r['model']} & {r['setting']} & {r['n']} & " +
                            " & ".join(vals) + f" & {avg_len}" + r" \\")
    my_tex_lines += [r"\hline", r"\end{tabular}",
                     r"% Token-F1 primary; KC-R and ROUGE-L F1 auxiliary."]
    my_tex = os.path.join(outdir, "table_mydatavqa_metrics.tex")
    with open(my_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(my_tex_lines) + "\n")

    print(df.to_string(index=False))
    print(f"\n-> {csv_path}\n-> {tex_path}\n-> {my_csv}\n-> {my_tex}")

    # 顺带生成论文延迟表（tab:latency_block 同构），两表同批刷新
    try:
        from scripts.make_latency_table import collect as _lat_collect  # noqa: F401
        import subprocess as _sp
        _sp.run([sys.executable, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "scripts", "make_latency_table.py"),
            "--root", args.root, "--outdir", outdir], check=False)
    except Exception as e:  # noqa: BLE001
        print(f"  !! 延迟表生成失败: {e}")
    if len(commits) > 1:
        warnings.append(f"结果来自多个 git commit: {sorted(commits)}")
    if warnings:
        print("\n[警告]")
        for w in warnings:
            print("  - " + w)
        if not args.allow_missing:
            sys.exit(1)


if __name__ == "__main__":
    main()
