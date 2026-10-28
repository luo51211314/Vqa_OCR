#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配对 bootstrap 置信区间 + 配对显著性检验（纯离线，读 detail CSV）。

统计口径（写入论文/回复，保证可复算）：
  * B=10000 次配对 bootstrap（同一组 question_id 索引同时重抽样两条件）；
  * 95% CI 取 2.5/97.5 percentile；配对增益 Δ = mean(ocr) - mean(off) 同样 bootstrap；
  * 显著性：二值正确（score>=1）用 McNemar 精确二项检验，连续分数差用 Wilcoxon；
  * 若 Δ 的 CI 跨 0，结论必须表述为"未观测到可测量收益"，不得写"提升"。

用法：
  python scripts/bootstrap_ci.py --off off_detail.csv --ocr ocr_detail.csv \
      --out pair_ci.json [--B 10000]
  python scripts/bootstrap_ci.py --single some_detail.csv --out single_ci.json
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except Exception:  # scipy 缺失时降级为内置实现
    scipy_stats = None


def load_scores(path):
    df = pd.read_csv(path)
    if "question_id" not in df.columns or "score" not in df.columns:
        raise ValueError(f"{path} 缺少 question_id/score 列")
    df = df[["question_id", "score"]].copy()
    df["question_id"] = df["question_id"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    return df.drop_duplicates(subset="question_id").set_index("question_id")["score"]


def bootstrap_ci(values, B, rng):
    values = np.asarray(values, dtype=float)
    n = len(values)
    idx = rng.randint(0, n, size=(B, n))
    means = values[idx].mean(axis=1)
    return {
        "n": int(n),
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def mcnemar_exact_p(b, c):
    """McNemar 精确检验（二项分布双侧 p）；b/c 为不一致对数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X<=k | X~Bin(n,0.5)) * 2
    if scipy_stats is not None:
        p_one_side = scipy_stats.binom.cdf(k, n, 0.5)
    else:
        from math import comb
        p_one_side = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * p_one_side))


def wilcoxon_signed_rank_p(off_scores, ocr_scores):
    """Two-sided paired Wilcoxon p-value for continuous per-sample scores."""
    a = np.asarray(off_scores, dtype=float)
    b = np.asarray(ocr_scores, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Wilcoxon requires paired arrays with identical shape")
    if not np.any(b != a):
        return 1.0
    if scipy_stats is None:
        return None
    try:
        return float(scipy_stats.wilcoxon(
            b, a, zero_method="wilcox", alternative="two-sided").pvalue)
    except (ValueError, FloatingPointError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", dest="off_csv")
    ap.add_argument("--ocr", dest="ocr_csv")
    ap.add_argument("--single")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    result = {"B": args.B, "seed": args.seed}

    if args.single:
        s = load_scores(args.single)
        result["single"] = bootstrap_ci(s.values, args.B, rng)
        result["single"]["source"] = args.single
    else:
        if not (args.off_csv and args.ocr_csv):
            ap.error("配对模式需要同时提供 --off 和 --ocr")
        off = load_scores(args.off_csv)
        ocr = load_scores(args.ocr_csv)
        common = off.index.intersection(ocr.index)
        if len(common) == 0:
            raise ValueError("两个 CSV 没有共同 question_id，无法配对")

        a = off.loc[common].values
        b_ = ocr.loc[common].values
        result["off"] = bootstrap_ci(a, args.B, rng)
        result["ocr"] = bootstrap_ci(b_, args.B, rng)
        result["off"]["source"] = args.off_csv
        result["ocr"]["source"] = args.ocr_csv
        result["n_paired"] = int(len(common))
        if len(common) < len(off) or len(common) < len(ocr):
            print(f"警告: off={len(off)} ocr={len(ocr)} 配对后={len(common)}", file=sys.stderr)

        # 配对 Δ
        n = len(common)
        idx = rng.randint(0, n, size=(args.B, n))
        delta = (b_ - a)
        d_means = delta[idx].mean(axis=1)
        result["delta"] = {
            "mean": float(delta.mean()),
            "ci95": [float(np.percentile(d_means, 2.5)),
                     float(np.percentile(d_means, 97.5))],
            "crosses_zero": bool(np.percentile(d_means, 2.5) <= 0.0 <= np.percentile(d_means, 97.5)),
        }

        # McNemar（二值正确：score>=1）
        off_ok = a >= 1.0
        ocr_ok = b_ >= 1.0
        b01 = int(np.sum(~off_ok & ocr_ok))   # off 错 / ocr 对
        b10 = int(np.sum(off_ok & ~ocr_ok))   # off 对 / ocr 错
        result["mcnemar"] = {
            "off_wrong_ocr_right": b01,
            "off_right_ocr_wrong": b10,
            "p_value_exact": mcnemar_exact_p(b01, b10),
        }

        # Wilcoxon 符号秩（连续分数差，去掉零差）
        nz = delta[delta != 0.0]
        if len(nz) >= 1 and scipy_stats is not None:
            try:
                w_p = float(scipy_stats.wilcoxon(b_, a, zero_method="wilcox",
                                                 alternative="two-sided").pvalue)
            except Exception:
                w_p = None
            result["wilcoxon"] = {"p_value": w_p, "n_nonzero_diff": int(len(nz))}
        else:
            result["wilcoxon"] = {"p_value": None, "n_nonzero_diff": int(len(nz))}

        result["claim"] = (
            "no measurable gain (delta CI crosses 0)"
            if result["delta"]["crosses_zero"]
            else "significant-direction change (see delta CI)"
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
