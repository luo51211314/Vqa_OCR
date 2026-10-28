#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模板 / token预算 / L2 消融表（全自动，读 YAML 定义 + metrics/detail，禁止手填）。

格子不做文件名正则猜测，而是直接用 experiments/run_config.py 的
expand_jobs + expected_metrics_path 从 YAML 枚举，保证表格与实验定义严格一致；
产物缺失即 TBD 并报警。

统计口径：配对 bootstrap B=10000（按 question_id 对齐），95% percentile CI；
  * 模板表（dev300 预实验 / 全集 template_full）：各 P 相对 w/o OCR 的 Δ。
    P2 在评测前锁定，模板实验仅作敏感性检验，不做模板选择；
  * 预算表：各预算相对 ∞（主表 ocr 格）的 Δ；CSV 另附 mean OCR tokens / 丢块率；
  * L2 表：各规则相对 L1（主表 ocr 格，无过滤）的 Δ。
CI 跨 0 加 †（未观测到可测量差异）。

用法:
  python scripts/make_ablation_tables.py --run-dir results_repro/<dir> \
      --kind all|template|budget|l2
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "experiments"))
from bootstrap_ci import load_scores  # noqa: E402
from ablation_ci_tex import write_ci_companion
import run_config as rc  # noqa: E402

B_DEFAULT = 10000


# ----------------------------- 基础工具 -----------------------------
def load_cfg(stem):
    with open(os.path.join(REPO, "experiments", "configs", stem + ".yaml"),
              encoding="utf-8") as f:
        return yaml.safe_load(f)


def job_index(cfg, cfg_stem, run_dir):
    """返回 {(dataset,split,model_key,variant,phi_tuple): metrics_path}。"""
    out_dir = os.path.join(run_dir, cfg_stem)
    idx = {}
    for job in rc.expand_jobs(cfg):
        phi = job["phi"]
        phi_key = (
            phi.get("order", "native"),
            phi.get("template", "p2"),
            phi.get("budget_tokens"),
            phi.get("conf_threshold"),
            bool(phi.get("dedup")),
        )
        key = (job["dataset"], job["split"], job["model_key"], job["variant"], phi_key)
        idx[key] = rc.expected_metrics_path(job, out_dir)
    return idx


def main_index(run_dir):
    """主表（∞预算/L1/全集基线）索引。"""
    return job_index(load_cfg("main_table"), "main_table", run_dir)


def score_of(metrics_path):
    if not metrics_path or not os.path.exists(metrics_path):
        return None
    d = json.load(open(metrics_path))
    return d.get(d.get("metric_type"))


def meta_of(metrics_path):
    if not metrics_path or not os.path.exists(metrics_path):
        return None
    return json.load(open(metrics_path))


def detail_of(metrics_path):
    if not metrics_path:
        return None
    p = metrics_path.replace("_metrics.json", "_detail.csv")
    return p if os.path.exists(p) else None


def paired_delta(base_csv, other_csv, B, seed):
    """Δ = other - base，配对 bootstrap CI。"""
    if not base_csv or not other_csv:
        return None
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
    deltas = diff[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return {"n": n, "delta": float(diff.mean()),
            "ci": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]}


def tex_cell(value, stat):
    if value is None:
        return "TBD"
    s = f"{value:.3f}"
    if stat:
        dagger = r"$\dagger$" if stat["ci"][0] <= 0 <= stat["ci"][1] else ""
        s += f" ({stat['delta']:+.3f}{dagger})"
    return s


def write_outputs(out_dir, name, csv_text, tex_lines):
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, f"{name}.csv"), "w").write(csv_text if csv_text.endswith("\n")
                                                          else csv_text + "\n")
    open(os.path.join(out_dir, f"{name}.tex"), "w").write("\n".join(tex_lines) + "\n")
    write_ci_companion(out_dir, name, csv_text)
    print("written:", os.path.join(out_dir, name + ".csv"))
    print("written:", os.path.join(out_dir, name + ".tex"))


# ----------------------------- 模板消融 -----------------------------
def table_template(run_dir, B, warnings):
    cfg = load_cfg("ablation_template")
    idx = job_index(cfg, "ablation_template", run_dir)
    rows = [("chartqa", "test", "llava"), ("chartqa", "test", "qwen"),
            ("docvqa", "val", "llava"), ("docvqa", "val", "qwen")]
    templates = ["p1", "p2", "p3"]

    recs = []
    for r_i, (ds, sp, mk) in enumerate(rows):
        off = idx.get((ds, sp, mk, "off", ("native", "p2", None, None, False)))
        off_s, off_csv = score_of(off), detail_of(off)
        rec = {"row": (ds, mk), "n": None, "off": off_s, "tpl": {}}
        if off_csv:
            rec["n"] = meta_of(off).get("total_samples")
        for t_i, t in enumerate(templates):
            mp = idx.get((ds, sp, mk, "ocr", ("native", t, None, None, False)))
            st = paired_delta(off_csv, detail_of(mp), B, seed=2000 + r_i * 11 + t_i)
            rec["tpl"][t] = (score_of(mp), st)
            if mp is None or not os.path.exists(mp):
                warnings.append(f"[template] 缺格 {ds}/{mk}/{t}")
        if off is None or not os.path.exists(off):
            warnings.append(f"[template] 缺 w/o 基线 {ds}/{mk}")
        recs.append(rec)

    csv = ["dataset,model,n,off,p1,p1_delta,p1_ci_lo,p1_ci_hi,p2,p2_delta,p2_ci_lo,p2_ci_hi,p3,p3_delta,p3_ci_lo,p3_ci_hi"]
    for r in recs:
        cells = [r["row"][0], r["row"][1], str(r["n"] or "TBD"),
                 f"{r['off']:.4f}" if r["off"] is not None else "TBD"]
        for t in templates:
            v, st = r["tpl"][t]
            cells += [f"{v:.4f}" if v is not None else "TBD",
                      f"{st['delta']:+.4f}" if st else "",
                      f"{st['ci'][0]:.4f}" if st else "",
                      f"{st['ci'][1]:.4f}" if st else ""]
        csv.append(",".join(cells))

    tex = [f"% generated; dev300 development slice (pre-fixed P2; sensitivity pilot, not template selection); "
           f"delta = template OCR - w/o OCR; paired bootstrap B={B}; dagger=CI crosses 0",
           r"\begin{tabular}{ll|c|c|ccc}",
           r"\hline",
           r"Dataset & Model & $n$ & w/o OCR & P1 & P2 (used) & P3 \\",
           r"\hline"]
    last = None
    for r in recs:
        ds = r["row"][0] if r["row"][0] != last else ""
        last = r["row"][0]
        off_txt = f"{r['off']:.3f}" if r["off"] is not None else "TBD"
        n_txt = r["n"] if r["n"] is not None else "TBD"
        tex.append(f"{ds} & {r['row'][1]} & {n_txt} & {off_txt} & "
                   f"{tex_cell(*r['tpl']['p1'])} & {tex_cell(*r['tpl']['p2'])} & "
                   f"{tex_cell(*r['tpl']['p3'])} \\\\")
    tex += [r"\hline", r"\end{tabular}"]
    write_outputs(os.path.join(run_dir, "ablation_template", "table_template"),
                  "table_template", "\n".join(csv), tex)


# ------------------------- 模板消融（全集敏感性） -------------------------
def table_template_full(run_dir, B, warnings):
    """全集版模板敏感性检验。

    P1/P3 来自 ablation_template_full/（全集）；w/o 与 P2 复用主表全集
    detail CSV。定位为事后敏感性分析（P2 事前锁定），不是模板选择。
    """
    cfg = load_cfg("ablation_template_full")
    idx = job_index(cfg, "ablation_template_full", run_dir)
    midx = main_index(run_dir)
    phi_p2 = ("native", "p2", None, None, False)
    rows = [(ds, sp, mk)
            for ds, sp in (("chartqa", "test"), ("docvqa", "val"),
                           ("mydatavqa", "test"))
            for mk in ("llava", "qwen", "mplug")]
    templates = ["p1", "p2", "p3"]

    recs = []
    for r_i, (ds, sp, mk) in enumerate(rows):
        off = midx.get((ds, sp, mk, "off", phi_p2))
        off_csv = detail_of(off)
        rec = {"row": (ds, mk), "n": (meta_of(off) or {}).get("total_samples"),
               "off": score_of(off), "tpl": {}}
        for t_i, t in enumerate(templates):
            if t == "p2":
                mp = midx.get((ds, sp, mk, "ocr", phi_p2))  # 主表 P2 全集
            else:
                mp = idx.get((ds, sp, mk, "ocr", ("native", t, None, None, False)))
            st = paired_delta(off_csv, detail_of(mp), B, seed=5000 + r_i * 11 + t_i)
            rec["tpl"][t] = (score_of(mp), st)
            if mp is None or not os.path.exists(mp):
                warnings.append(f"[template_full] 缺格 {ds}/{mk}/{t}")
        if off is None or not os.path.exists(off):
            warnings.append(f"[template_full] 缺 w/o 主表基线 {ds}/{mk}")
        recs.append(rec)

    csv = ["dataset,model,n,off,p1,p1_delta,p1_ci_lo,p1_ci_hi,p2,p2_delta,p2_ci_lo,p2_ci_hi,p3,p3_delta,p3_ci_lo,p3_ci_hi"]
    for r in recs:
        cells = [r["row"][0], r["row"][1], str(r["n"] or "TBD"),
                 f"{r['off']:.4f}" if r["off"] is not None else "TBD"]
        for t in templates:
            v, st = r["tpl"][t]
            cells += [f"{v:.4f}" if v is not None else "TBD",
                      f"{st['delta']:+.4f}" if st else "",
                      f"{st['ci'][0]:.4f}" if st else "",
                      f"{st['ci'][1]:.4f}" if st else ""]
        csv.append(",".join(cells))

    tex = [f"% generated; FULL test/val set (post-hoc sensitivity; P2 fixed before evaluation); "
           f"delta = template OCR - w/o OCR; paired bootstrap B={B}; dagger=CI crosses 0",
           r"\begin{tabular}{ll|c|c|ccc}",
           r"\hline",
           r"Dataset & Model & $n$ & w/o OCR & P1 & P2 (used) & P3 \\",
           r"\hline"]
    last = None
    for r in recs:
        ds = r["row"][0] if r["row"][0] != last else ""
        last = r["row"][0]
        off_txt = f"{r['off']:.3f}" if r["off"] is not None else "TBD"
        n_txt = r["n"] if r["n"] is not None else "TBD"
        tex.append(f"{ds} & {r['row'][1]} & {n_txt} & {off_txt} & "
                   f"{tex_cell(*r['tpl']['p1'])} & {tex_cell(*r['tpl']['p2'])} & "
                   f"{tex_cell(*r['tpl']['p3'])} \\\\")
    tex += [r"\hline", r"\end{tabular}"]
    write_outputs(os.path.join(run_dir, "ablation_template_full", "table_template_full"),
                  "table_template_full", "\n".join(csv), tex)


# ----------------------------- 预算消融 -----------------------------
def table_budget(run_dir, B, warnings):
    cfg = load_cfg("ablation_budget")
    idx = job_index(cfg, "ablation_budget", run_dir)
    midx = main_index(run_dir)
    models = ["llava", "qwen", "mplug"]
    budgets = [256, 512, 1024, 2048]

    recs = []
    for r_i, mk in enumerate(models):
        off = midx.get(("docvqa", "val", mk, "off", ("native", "p2", None, None, False)))
        inf = midx.get(("docvqa", "val", mk, "ocr", ("native", "p2", None, None, False)))
        inf_csv = detail_of(inf)
        rec = {"model": mk, "n": (meta_of(inf) or {}).get("total_samples"),
               "off": score_of(off), "inf": score_of(inf), "cells": []}
        for b_i, bud in enumerate(budgets):
            mp = idx.get(("docvqa", "val", mk, "ocr", ("native", "p2", bud, None, False)))
            st = paired_delta(inf_csv, detail_of(mp), B, seed=3000 + r_i * 13 + b_i)
            phi = (meta_of(mp) or {}).get("phi_stats", {})
            rec["cells"].append({
                "b": bud, "score": score_of(mp), "stat": st,
                "ocr_tokens": phi.get("mean_ocr_tokens"),
                "drop": phi.get("block_drop_rate"),
            })
            if not mp or not os.path.exists(mp):
                warnings.append(f"[budget] 缺格 docvqa/{mk}/b{bud}")
        if not inf or not os.path.exists(inf):
            warnings.append(f"[budget] 缺 ∞ 主表基线 docvqa/{mk}")
        recs.append(rec)

    csv = ["model,n,off,inf," +
           ",".join(f"b{b},b{b}_delta,b{b}_ci_lo,b{b}_ci_hi,b{b}_ocr_tokens,b{b}_drop_rate"
                    for b in budgets)]
    for r in recs:
        cells = [r["model"], str(r["n"] or "TBD"),
                 f"{r['off']:.4f}" if r["off"] is not None else "TBD",
                 f"{r['inf']:.4f}" if r["inf"] is not None else "TBD"]
        for c in r["cells"]:
            cells += [f"{c['score']:.4f}" if c["score"] is not None else "TBD",
                      f"{c['stat']['delta']:+.4f}" if c["stat"] else "",
                      f"{c['stat']['ci'][0]:.4f}" if c["stat"] else "",
                      f"{c['stat']['ci'][1]:.4f}" if c["stat"] else "",
                      f"{c['ocr_tokens']:.1f}" if c["ocr_tokens"] is not None else "",
                      f"{c['drop']:.4f}" if c["drop"] is not None else ""]
        csv.append(",".join(cells))

    head = " & ".join(["Model", "$n$", "w/o", r"$\infty$"] +
                      [str(b) for b in budgets]) + r" \\"
    tex = [f"% generated; DocVQA val; delta = budget - infinity(main ocr); paired bootstrap B={B}; dagger=CI crosses 0",
           r"\begin{tabular}{l|c|cc|cccc}",
           r"\hline",
           r"& & \multicolumn{2}{c|}{main} & \multicolumn{4}{c}{OCR token budget} \\",
           head,
           r"\hline"]
    for r in recs:
        row = [r["model"], str(r["n"] or "TBD"),
               f"{r['off']:.3f}" if r["off"] is not None else "TBD",
               f"{r['inf']:.3f}" if r["inf"] is not None else "TBD"]
        for c in r["cells"]:
            row.append(tex_cell(c["score"], c["stat"]))
        tex.append(" & ".join(row) + r" \\")
    tex += [r"\hline", r"\end{tabular}",
            r"% mean OCR tokens / block-drop rate 见表 table_budget.csv"]
    write_outputs(os.path.join(run_dir, "ablation_budget", "table_budget"),
                  "table_budget", "\n".join(csv), tex)


# ----------------------- MyDataVQA budget -----------------------
def table_budget_mydatavqa(run_dir, B, warnings):
    cfg = load_cfg("ablation_budget")
    idx = job_index(cfg, "ablation_budget", run_dir)
    midx = main_index(run_dir)
    models = ["llava", "qwen", "mplug"]
    budgets = [128, 256, 512, 1024]
    phi_base = ("native", "p2", None, None, False)

    recs = []
    for r_i, mk in enumerate(models):
        base = midx.get(("mydatavqa", "test", mk, "ocr", phi_base))
        base_csv = detail_of(base)
        rec = {
            "model": mk,
            "n": (meta_of(base) or {}).get("total_samples"),
            "inf": score_of(base),
            "cells": [],
        }
        for b_i, budget in enumerate(budgets):
            mp = idx.get(("mydatavqa", "test", mk, "ocr",
                          ("native", "p2", budget, None, False)))
            stat = paired_delta(base_csv, detail_of(mp), B,
                                seed=7000 + r_i * 13 + b_i)
            phi = (meta_of(mp) or {}).get("phi_stats", {})
            rec["cells"].append({
                "budget": budget,
                "score": score_of(mp),
                "stat": stat,
                "ocr_tokens": phi.get("mean_ocr_tokens"),
                "drop": phi.get("block_drop_rate"),
            })
            if not mp or not os.path.exists(mp):
                warnings.append(
                    f"[budget_mydatavqa] missing {mk}/b{budget}")
        if not base or not os.path.exists(base):
            warnings.append(
                f"[budget_mydatavqa] missing infinity baseline {mk}")
        recs.append(rec)

    csv = ["model,n,inf," +
           ",".join(
               f"b{b},b{b}_delta,b{b}_ci_lo,b{b}_ci_hi,"
               f"b{b}_ocr_tokens,b{b}_drop_rate"
               for b in budgets)]
    for r in recs:
        cells = [
            r["model"], str(r["n"] or "TBD"),
            f"{r['inf']:.4f}" if r["inf"] is not None else "TBD",
        ]
        for c in r["cells"]:
            stat = c["stat"]
            cells += [
                f"{c['score']:.4f}" if c["score"] is not None else "TBD",
                f"{stat['delta']:+.4f}" if stat else "",
                f"{stat['ci'][0]:.4f}" if stat else "",
                f"{stat['ci'][1]:.4f}" if stat else "",
                f"{c['ocr_tokens']:.1f}" if c["ocr_tokens"] is not None else "",
                f"{c['drop']:.4f}" if c["drop"] is not None else "",
            ]
        csv.append(",".join(cells))

    tex = [
        f"% generated; MyDataVQA test; delta=budget-infinity(main OCR); paired bootstrap B={B}; dagger=CI crosses 0",
        r"\begin{tabular}{l|c|c|cccc}",
        r"\hline",
        r"Model & $n$ & $\infty$ & 128 & 256 & 512 & 1024 \\",
        r"\hline",
    ]
    for r in recs:
        row = [
            r["model"], str(r["n"] or "TBD"),
            f"{r['inf']:.3f}" if r["inf"] is not None else "TBD",
        ]
        for c in r["cells"]:
            row.append(tex_cell(c["score"], c["stat"]))
        tex.append(" & ".join(row) + r" \\")
    tex += [
        r"\hline", r"\end{tabular}",
        r"% mean OCR tokens / block-drop rate are in table_budget_mydatavqa.csv",
    ]
    write_outputs(
        os.path.join(run_dir, "ablation_budget",
                     "table_budget_mydatavqa"),
        "table_budget_mydatavqa", "\n".join(csv), tex)

    # Keep the standalone MyDataVQA artifact, but also append it to the
    # established budget TeX entry point.  The live watcher calls
    # table_budget() immediately before this function, so the DocVQA panel is
    # rebuilt first and this operation is idempotent for manual refreshes too.
    combined_path = os.path.join(
        run_dir, "ablation_budget", "table_budget", "table_budget.tex")
    if os.path.exists(combined_path):
        marker = "% MyDataVQA budget panel"
        with open(combined_path, encoding="utf-8") as f:
            doc_tex = f.read().split(marker, 1)[0].rstrip()
        combined = (
            doc_tex + "\n\n" + marker + "\n" + "\n".join(tex) + "\n"
        )
        tmp_path = combined_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(combined)
        os.replace(tmp_path, combined_path)
        print("written:", combined_path, "(DocVQA + MyDataVQA)")


# ----------------------------- L2 消融 -----------------------------
def table_l2(run_dir, B, warnings):
    cfg = load_cfg("ablation_l2")
    idx = job_index(cfg, "ablation_l2", run_dir)
    midx = main_index(run_dir)
    rows = [("docvqa", "val", mk) for mk in ("llava", "qwen", "mplug")]
    rows += [("chartqa", "test", mk) for mk in ("llava", "qwen")]
    rows += [("mydatavqa", "test", mk) for mk in ("llava", "qwen", "mplug")]
    taus = [0.5, 0.7, 0.9]

    recs = []
    for r_i, (ds, sp, mk) in enumerate(rows):
        base = midx.get((ds, sp, mk, "ocr",
                         ("native", "p2", None, None, False)))
        base_csv = detail_of(base)
        rec = {"dataset": ds, "model": mk,
               "n": (meta_of(base) or {}).get("total_samples"),
               "l1": score_of(base), "taus": [], "dedup": None}
        for t_i, tau in enumerate(taus):
            mp = idx.get((ds, sp, mk, "ocr",
                          ("native", "p2", None, tau, False)))
            rec["taus"].append(
                (tau, score_of(mp),
                 paired_delta(base_csv, detail_of(mp), B,
                              seed=4000 + r_i * 17 + t_i)))
            if not mp or not os.path.exists(mp):
                warnings.append(f"[l2] missing {ds}/{mk}/tau{tau}")
        mp = idx.get((ds, sp, mk, "ocr",
                      ("native", "p2", None, None, True)))
        rec["dedup"] = (
            score_of(mp),
            paired_delta(base_csv, detail_of(mp), B,
                         seed=4000 + r_i * 17 + 9))
        if not mp or not os.path.exists(mp):
            warnings.append(f"[l2] missing {ds}/{mk}/dedup")
        if not base or not os.path.exists(base):
            warnings.append(f"[l2] missing L1 main baseline {ds}/{mk}")
        recs.append(rec)

    csv = ["dataset,model,n,l1," +
           ",".join(f"tau{t},tau{t}_delta,tau{t}_ci_lo,tau{t}_ci_hi"
                    for t in taus) +
           ",dedup,dedup_delta,dedup_ci_lo,dedup_ci_hi"]
    for r in recs:
        cells = [r["dataset"], r["model"], str(r["n"] or "TBD"),
                 f"{r['l1']:.4f}" if r["l1"] is not None else "TBD"]
        for _, v, st in r["taus"]:
            cells += [f"{v:.4f}" if v is not None else "TBD",
                      f"{st['delta']:+.4f}" if st else "",
                      f"{st['ci'][0]:.4f}" if st else "",
                      f"{st['ci'][1]:.4f}" if st else ""]
        v, st = r["dedup"]
        cells += [f"{v:.4f}" if v is not None else "TBD",
                  f"{st['delta']:+.4f}" if st else "",
                  f"{st['ci'][0]:.4f}" if st else "",
                  f"{st['ci'][1]:.4f}" if st else ""]
        csv.append(",".join(cells))

    tex = [
        f"% generated; delta=L2-rule-L1(main OCR); paired bootstrap B={B}; dagger=CI crosses 0",
        r"\begin{tabular}{ll|c|c|ccc|c}",
        r"\hline",
        r"Dataset & Model & $n$ & L1 (no filter) & $\tau{=}0.5$ & $\tau{=}0.7$ & $\tau{=}0.9$ & dedup \\",
        r"\hline",
    ]
    last = None
    for r in recs:
        ds = r["dataset"] if r["dataset"] != last else ""
        last = r["dataset"]
        row = [ds, r["model"], str(r["n"] or "TBD"),
               f"{r['l1']:.3f}" if r["l1"] is not None else "TBD"]
        for _, v, st in r["taus"]:
            row.append(tex_cell(v, st))
        row.append(tex_cell(*r["dedup"]))
        tex.append(" & ".join(row) + r" \\")
    tex += [r"\hline", r"\end{tabular}"]
    write_outputs(os.path.join(run_dir, "ablation_l2", "table_l2"),
                  "table_l2", "\n".join(csv), tex)

    # A companion table makes L2 aggressiveness auditable instead of reporting
    # threshold labels alone. Cells are "retained OCR blocks / mean OCR tokens".
    retention_csv = [
        "dataset,model,l1_mean_ocr_tokens,"
        "tau0.5_block_retention,tau0.5_missing_confidence,tau0.5_mean_ocr_tokens,"
        "tau0.7_block_retention,tau0.7_missing_confidence,tau0.7_mean_ocr_tokens,"
        "tau0.9_block_retention,tau0.9_missing_confidence,tau0.9_mean_ocr_tokens,"
        "dedup_block_retention,dedup_mean_ocr_tokens"
    ]
    retention_tex = [
        r"% cell = retained OCR blocks / mean OCR tokens",
        r"\begin{tabular}{ll|c|ccc|c}",
        r"\hline",
        r"Dataset & Model & L1 tokens & $\tau{=}0.5$ & $\tau{=}0.7$ & $\tau{=}0.9$ & dedup \\",
        r"\hline",
    ]
    last_ds = None
    for ds, sp, mk in rows:
        base = midx.get((ds, sp, mk, "ocr",
                         ("native", "p2", None, None, False)))
        base_phi = (meta_of(base) or {}).get("phi_stats") or {}
        base_tokens = base_phi.get("mean_ocr_tokens")
        csv_cells = [ds, mk,
                     f"{base_tokens:.2f}" if base_tokens is not None else "TBD"]
        tex_cells = [ds if ds != last_ds else "", mk,
                     f"{base_tokens:.1f}" if base_tokens is not None else "TBD"]
        last_ds = ds
        for tau in taus:
            mp = idx.get((ds, sp, mk, "ocr",
                          ("native", "p2", None, tau, False)))
            phi = (meta_of(mp) or {}).get("phi_stats") or {}
            keep = phi.get("confidence_retention_rate")
            missing = phi.get("missing_confidence_rate")
            tokens = phi.get("mean_ocr_tokens")
            csv_cells += [
                f"{keep:.4f}" if keep is not None else "TBD",
                f"{missing:.4f}" if missing is not None else "TBD",
                f"{tokens:.2f}" if tokens is not None else "TBD",
            ]
            if keep is not None and tokens is not None:
                tex_cells.append(rf"{100 * keep:.1f}\% / {tokens:.1f}")
            else:
                tex_cells.append("TBD")
        mp = idx.get((ds, sp, mk, "ocr",
                      ("native", "p2", None, None, True)))
        phi = (meta_of(mp) or {}).get("phi_stats") or {}
        keep = phi.get("dedup_retention_rate")
        tokens = phi.get("mean_ocr_tokens")
        csv_cells += [
            f"{keep:.4f}" if keep is not None else "TBD",
            f"{tokens:.2f}" if tokens is not None else "TBD",
        ]
        if keep is not None and tokens is not None:
            tex_cells.append(rf"{100 * keep:.1f}\% / {tokens:.1f}")
        else:
            tex_cells.append("TBD")
        retention_csv.append(",".join(csv_cells))
        retention_tex.append(" & ".join(tex_cells) + r" \\")
    retention_tex += [r"\hline", r"\end{tabular}"]
    write_outputs(os.path.join(run_dir, "ablation_l2", "table_l2"),
                  "table_l2_retention", "\n".join(retention_csv),
                  retention_tex)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--kind", choices=["all", "template", "template_full", "budget", "l2"],
                    default="all")
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    args = ap.parse_args()

    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(REPO, args.run_dir)
    warnings = []
    if args.kind in ("all", "template"):
        table_template(run_dir, args.B, warnings)
    if args.kind in ("all", "template_full"):
        table_template_full(run_dir, args.B, warnings)
    if args.kind in ("all", "budget"):
        table_budget(run_dir, args.B, warnings)
        table_budget_mydatavqa(run_dir, args.B, warnings)
    if args.kind in ("all", "l2"):
        table_l2(run_dir, args.B, warnings)

    for w in warnings:
        print("WARN:", w, file=sys.stderr)
    if warnings:
        sys.exit(2)


if __name__ == "__main__":
    main()
