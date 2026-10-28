# Reproduction and repository scope

## Published code

Inference: main_inference.py, run.sh, load_dataset.py, choose_expert.py.
Runtime modules: Algorithm/, expert/, loaders/, model_loader/.
Required LLaVA source: models/llava/llava/ (retain upstream LICENSE).
Optional fusion source: train_fusion/ is referenced by the fusion inference path.
Experiment definitions: experiments/configs/ and experiments/run_config.py.
Shell drivers: experiments/ablations/, main_table/, analysis/, and queue helpers.
Postprocessing and CI tables: scripts/.

Excluded: model weights, datasets, results, logs, caches, manuscript drafts,
editor state, upstream demo/documentation assets, and the independent finetune/
workflow. Existing Git history is preserved, not rewritten.

## Environment and paths

requirements_vqa_infer.txt is the existing dependency list, not a newly verified
clean-install lockfile. Experiment configs default to the vqa_train environment.
Several paths in run.sh, loaders/, model_loader/, and configs remain specific to
the original server. Adapt them before relocation.

Evaluated model families: LLaVA, Qwen2.5-VL, mPLUG-Owl3.
MyDataVQA test source:
 /root/autodl-tmp/dataset/MyDataVQA/MyDataVQA_test.parquet

Required fields: sample_id, image_id, ImageBytes, question, question_type,
answer_concise, answer_full. Primary metric: token_f1. Auxiliary metrics:
token precision/recall, concept recall, multi-reference ROUGE-L, output length.

## Reproduce

Inspect commands without GPU inference:

    python experiments/run_config.py experiments/configs/main_table.yaml --dry-run

Run all configs with the same explicit RUN_ID:

    python experiments/run_config.py experiments/configs/main_table.yaml --run --run-dir RUN_ID
    python experiments/run_config.py experiments/configs/ablation_template_full.yaml --run --run-dir RUN_ID
    python experiments/run_config.py experiments/configs/ablation_budget.yaml --run --run-dir RUN_ID
    python experiments/run_config.py experiments/configs/ablation_order.yaml --run --run-dir RUN_ID
    python experiments/run_config.py experiments/configs/ablation_l2.yaml --run --run-dir RUN_ID

Read driver help for model/dataset filters and existing-result handling.
YAML configs are the experiment specification. Historical night queue scripts
may contain machine-specific paths and run IDs. Dev300 is a separate sensitivity
artifact, not independent evidence of template selection before full evaluation.

## Generate outputs

    python scripts/make_table3.py --root results_repro/RUN_ID/main_table --allow-missing
    python scripts/make_ablation_tables.py --run-dir results_repro/RUN_ID --kind all
    python scripts/make_order_ablation_table.py --run-dir results_repro/RUN_ID
    python scripts/make_offline_matrix.py --run-dir results_repro/RUN_ID

Missing cells are reported; generators can write partial tables and return a
nonzero status. The watcher updates tables. CI companion tables follow main-table
axes. LaTeX may require booktabs, multirow, and xcolor with the table option.

## Interpretation limits

- Template delta baseline: off; order: native; budget: unlimited OCR-P2; L2: L1.
- CI is question-level paired bootstrap. Same-image clustering is not handled.
- Coordinate serialization changes representation, not only reading order.
- OOM incidents are recorded in run_info and must be disclosed.
- The offline matrix stores mean task quality and inference times, not queue
  waiting times. Different datasets use different quality metrics.
- Queue completion does not validate older manuscript or scheduler claims.

## Backups

Before cleanup, archive the full project, including .git and untracked results,
and the external dataset directory to D:\Vqa_ocr_backup\20261028.
Model weights at /root/autodl-tmp/model are not part of the dataset backup.
