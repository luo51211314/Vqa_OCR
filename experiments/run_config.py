#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YAML 实验驱动器：把声明式实验配置展开为 run.sh 命令并（可选）执行。

设计原则
--------
1. 所有实验参数只在 ``experiments/configs/*.yaml`` 中声明，禁止散落 shell 变量；
2. 默认 ``--dry-run`` 只打印命令，``--run`` 才真正执行；
3. 输出统一写入 ``results_repro/<日期>_<commit短哈希>/<配置名>/``，
   每个 job 一个 stdout 日志，metrics/detail CSV 由 main_inference 自动落盘；
4. 不做任何人工誊写：表格由 scripts/make_table3.py 从该目录扫描生成。

用法
----
    python experiments/run_config.py experiments/configs/main_table.yaml --dry-run
    python experiments/run_config.py experiments/configs/main_table.yaml --run
    python experiments/run_config.py experiments/configs/ablation_order.yaml \
        --only qwen --variants ocr,ros --run
"""
import argparse
import datetime
import os
import subprocess
import sys
import time

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "bs": 1,
    "time": "yes",
    "conda_env": "vqa_train",
    "seed": 42,
    "output_root": os.path.join(REPO_ROOT, "results_repro"),
}


def git_short_sha():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out
    except Exception:
        return "unknown"


def _norm_variant(v):
    """PyYAML 会把未加引号的 off/on 解析成布尔，这里统一纠正为字符串。"""
    if v is False:
        return "off"
    if v is True:
        return "on"
    return str(v)


def expand_jobs(cfg):
    """把 jobs × models × variants 展开为扁平 job dict 列表。"""
    defaults = {**DEFAULTS, **cfg.get("defaults", {})}
    models = cfg["models"]
    jobs = []
    for job in cfg["jobs"]:
        model_keys = job["models"]
        variants = [_norm_variant(v) for v in job.get("variants", ["off", "ocr"])]
        for mk in model_keys:
            for vk in variants:
                jobs.append({
                    "defaults": defaults,
                    "models": models,
                    "dataset": job["dataset"],
                    "split": job["split"],
                    "metric": job["metric"],
                    "model_key": mk,
                    "variant": vk,
                    "phi": job.get("phi", {}),
                    "sample_ids_file": job.get("sample_ids_file"),
                    "num_samples": job.get("num_samples"),
                    "bs": job.get("bs", defaults["bs"]),
                })
    return jobs


def build_command(job, output_dir):
    """构造 run.sh 的 17 个位置参数 + EXTRA_ARGS。"""
    d = job["defaults"]
    m = job["models"][job["model_key"]]
    # YAML 中 yes/no/on/off 可能被解析成布尔，统一成 run.sh 需要的字符串
    time_arg = "no" if d["time"] in (False, "no", "None", None) else "yes"
    use_experts = "off" if job["variant"] == "off" else "manual:ocr"

    positional = [
        job["dataset"],                    # 1 DATASET
        job["split"],                      # 2 SPLIT
        str(job["bs"]),                    # 3 BS
        str(job["num_samples"] or ""),     # 4 NUM_SAMPLES ("" = all)
        m["model_name"],                   # 5 MODEL_NAME
        m["model_type"],                   # 6 MODEL_TYPE
        job["metric"],                     # 7 METRIC_TYPE
        use_experts,                       # 8 USE_EXPERTS
        "direct",                          # 9 EXPERT_MODE
        "",                                # 10 FUSION_WEIGHT
        time_arg,                          # 11 TIME
        "no",                              # 12 USE_PIPELINE（主表/消融均走非 pipeline L1 路径）
        "full",                            # 13 PIPELINE_MODE
        output_dir,                        # 14 OUTPUT_DIR
        "no",                              # 15 DEBUG_PIPELINE
        d["conda_env"],                    # 16 CONDA_ENV
        "normal",                          # 17 OCR_MODE（兼容位，顺序由 EXTRA_ARGS 决定）
    ]

    extra = ["--seed", str(d["seed"])]
    if job["sample_ids_file"]:
        extra += ["--sample_ids_file", os.path.join(REPO_ROOT, job["sample_ids_file"])]
    if job["variant"] != "off":
        phi = job["phi"]
        extra += [
            "--ocr_order", str(phi.get("order", "native")),
            "--prompt_template", str(phi.get("template", "p2")),
        ]
        if phi.get("budget_tokens"):
            extra += ["--ocr_budget_tokens", str(phi["budget_tokens"])]
        if phi.get("conf_threshold"):
            extra += ["--ocr_conf_threshold", str(phi["conf_threshold"])]
        if phi.get("dedup"):
            extra += ["--ocr_dedup"]

    cmd = ["bash", os.path.join(REPO_ROOT, "run.sh")] + positional + extra
    return cmd


def expected_metrics_path(job, output_dir):
    """预测 main_inference 会写出的 metrics.json 路径（断点续跑判据）。"""
    d = job["defaults"]
    model_type = job["models"][job["model_key"]]["model_type"]
    suffix = "" if job["variant"] == "off" else "_ocr"
    time_suffix = "_time" if d["time"] not in (False, "no", "None", None) else ""
    phi = job["phi"]
    phi_suffix = ""
    if job["variant"] != "off":
        if phi.get("order", "native") != "native":
            phi_suffix += f"_{phi['order']}"
        if phi.get("template", "p2") != "p2":
            phi_suffix += f"_t{phi['template']}"
        if phi.get("budget_tokens"):
            phi_suffix += f"_b{phi['budget_tokens']}"
        if phi.get("conf_threshold"):
            phi_suffix += f"_tau{phi['conf_threshold']}"
        if phi.get("dedup"):
            phi_suffix += "_dedup"
    if job["sample_ids_file"]:
        phi_suffix += "_subset"
    basename = (f"{job['dataset']}_{job['split']}_{model_type}_"
                f"{job['metric']}{suffix}{time_suffix}{phi_suffix}")
    return os.path.join(output_dir, f"{basename}_metrics.json")


def job_tag(job):
    tag = f"{job['dataset']}_{job['split']}_{job['model_key']}_{job['variant']}"
    phi = job["phi"]
    if job["variant"] != "off":
        if phi.get("order", "native") != "native":
            tag += f"_{phi['order']}"
        if phi.get("template", "p2") != "p2":
            tag += f"_t{phi['template']}"
        if phi.get("budget_tokens"):
            tag += f"_b{phi['budget_tokens']}"
        if phi.get("conf_threshold"):
            tag += f"_tau{phi['conf_threshold']}"
        if phi.get("dedup"):
            tag += "_dedup"
    if job["sample_ids_file"]:
        tag += "_subset"
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="YAML 配置文件路径")
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="只打印命令（默认）")
    ap.add_argument("--run", action="store_true", help="真正顺序执行")
    ap.add_argument("--only", default=None, help="只跑指定 model key，逗号分隔")
    ap.add_argument("--variants", default=None, help="只跑指定 variant(off/ocr)，逗号分隔")
    ap.add_argument("--datasets", default=None, help="只跑指定 dataset，逗号分隔")
    ap.add_argument("--run-dir", default=None,
                    help="固定输出运行目录名（默认 <日期>_<commit>）；跨午夜续跑用")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                    help="已存在 metrics.json 的 job 直接跳过（默认开启）")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--retries", type=int, default=1,
                    help="失败 job 的额外重试次数（OOM/Killed 后自动等 30s 重试，默认1）")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    jobs = expand_jobs(cfg)
    if args.only:
        keep = set(args.only.split(","))
        jobs = [j for j in jobs if j["model_key"] in keep]
    if args.variants:
        keep = set(args.variants.split(","))
        jobs = [j for j in jobs if j["variant"] in keep]
    if args.datasets:
        keep = set(args.datasets.split(","))
        jobs = [j for j in jobs if j["dataset"] in keep]

    cfg_stem = os.path.splitext(os.path.basename(args.config))[0]
    run_name = args.run_dir or f"{datetime.datetime.now():%Y%m%d}_{git_short_sha()}"
    output_dir = os.path.join(
        cfg.get("defaults", {}).get("output_root", DEFAULTS["output_root"]),
        run_name,
        cfg_stem,
    )

    print(f"# 配置: {args.config}")
    print(f"# 共 {len(jobs)} 个 job，输出目录: {output_dir}")
    print(f"# 模式: {'执行' if args.run else 'dry-run（仅打印）'}")
    print()

    if args.run:
        os.makedirs(output_dir, exist_ok=True)

    failures = []
    skipped = []
    # 缓解长跑显存碎片导致的 CUDA OOM（不影响结果）
    run_env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    for i, job in enumerate(jobs, 1):
        cmd = build_command(job, output_dir)
        tag = job_tag(job)
        print(f"[{i}/{len(jobs)}] {tag}")
        print(" ".join(_quote(c) for c in cmd))
        if args.run:
            metrics_path = expected_metrics_path(job, output_dir)
            if args.skip_existing and os.path.exists(metrics_path):
                skipped.append(tag)
                print(f"  .. 跳过（已有产物 {os.path.basename(metrics_path)}）")
                print()
                continue
            log_path = os.path.join(output_dir, f"{tag}.log")
            rc = None
            for attempt in range(args.retries + 1):
                log_mode = "w" if attempt == 0 else "a"
                with open(log_path, log_mode, encoding="utf-8") as logf:
                    if attempt > 0:
                        logf.write(f"\n# ===== retry {attempt}/{args.retries} =====\n")
                        logf.flush()
                    ret = subprocess.run(cmd, cwd=REPO_ROOT, stdout=logf,
                                         stderr=subprocess.STDOUT, env=run_env)
                rc = ret.returncode
                if rc == 0 or os.path.exists(metrics_path):
                    rc = 0
                    break
                if attempt < args.retries:
                    print(f"  !! 第 {attempt+1} 次失败 rc={rc}，30s 后重试（OOM/被杀自动恢复）")
                    time.sleep(30)
            if rc != 0:
                failures.append((tag, rc))
                print(f"  !! 最终失败 rc={rc}，跳过不中断队列，见 {log_path}")
            else:
                print(f"  ok -> {log_path}")
        print()

    print(f"# 完成 {len(jobs)-len(skipped)-len(failures)} / "
          f"跳过 {len(skipped)} / 失败 {len(failures)} / 共 {len(jobs)}")

    # ---- 队列后处理：论文数字直接从产物生成，禁止人工誊写 ----
    if args.run and cfg.get("postprocess"):
        _run_postprocess(cfg["postprocess"], output_dir)

    if failures:
        print("失败 job:")
        for tag, rc in failures:
            print(f"  - {tag} (rc={rc})")
        sys.exit(1)


def _run_postprocess(spec, output_dir):
    """队列跑完后自动生成论文级产物（table3.csv / table3.tex）。

    即使有个别 job 失败也照常生成（缺格 TBD 并报警），保证论文表格
    始终由 metrics/detail 产物直接计算，不经过人手。
    """
    if spec == "table3":
        script = os.path.join(REPO_ROOT, "scripts", "make_table3.py")
        cmd = [sys.executable, script, "--root", output_dir, "--allow-missing"]
        print("# postprocess: auto Table 3 ->", " ".join(cmd))
        ret = subprocess.run(cmd, cwd=REPO_ROOT)
        if ret.returncode != 0:
            print("  !! Table 3 generation failed; check missing-cell warnings above")
    else:
        print(f"# unknown postprocess: {spec} (supported: table3)")


def _quote(s):
    return f'"{s}"' if (s == "" or " " in s) else str(s)


if __name__ == "__main__":
    main()
