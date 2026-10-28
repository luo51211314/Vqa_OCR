"""MyDataVQA loader and reproducible lexical metrics.

Primary: macro token F1 against the cleaned concise answer.
Diagnostics: strict concept recall and multi-reference ROUGE-L.
No new metric uses fuzzy or arbitrary substring matching.
"""

from collections import Counter
import io
import os
import re
import unicodedata

import editdistance
import numpy as np
import pandas as pd
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from loaders import VqaDataset


class Dataset(VqaDataset):
    name = "mydatavqa"
    METRIC_VERSION = "mydatavqa-token-f1-v1"
    QUESTION_TYPES = {
        "Comparison", "Comprehensive Understanding", "Location",
        "Numerical Calculation", "Trend",
    }

    def __init__(self, split="test", **_):
        super().__init__(split)
        self.data_dir = "/root/autodl-tmp/dataset/MyDataVQA"
        if split == "test":
            parquet_files = ["MyDataVQA_test.parquet"]
        elif split == "":
            parquet_files = sorted(
                f for f in os.listdir(self.data_dir) if f.endswith(".parquet")
            )
        else:
            parquet_files = sorted(
                f for f in os.listdir(self.data_dir)
                if f.startswith("batch_") and f.endswith(".parquet")
            )
        self.source_files = [os.path.join(self.data_dir, f) for f in parquet_files]
        missing_files = [p for p in self.source_files if not os.path.isfile(p)]
        if not self.source_files or missing_files:
            raise FileNotFoundError(
                f"MyDataVQA split={split!r} missing parquet files: {missing_files}"
            )
        self.df = pd.concat(
            [pd.read_parquet(p) for p in self.source_files], ignore_index=True
        )
        if split == "test":
            required = {
                "sample_id", "image_id", "ImageBytes", "question", "question_type",
                "answer_concise", "answer_full",
            }
            missing = sorted(required - set(self.df.columns))
            if missing:
                raise ValueError(f"MyDataVQA test missing columns: {missing}")
            if self.df["sample_id"].isna().any() or self.df["sample_id"].duplicated().any():
                raise ValueError("MyDataVQA sample_id must be non-null and unique")
            invalid = sorted(set(self.df["question_type"]) - self.QUESTION_TYPES)
            if invalid:
                raise ValueError(f"MyDataVQA invalid question_type values: {invalid}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        return self._load_image(sample["ImageBytes"]), sample["question"], [
            sample["answer_concise"]
        ], {
            "sample_id": sample["sample_id"],
            "image_id": sample["image_id"],
            "answer_full": list(sample["answer_full"]),
            "question_type": sample["question_type"],
        }

    @staticmethod
    def _load_image(image_field):
        if isinstance(image_field, bytes):
            return Image.open(io.BytesIO(image_field)).convert("RGB")
        if isinstance(image_field, str):
            return Image.open(image_field).convert("RGB")
        if isinstance(image_field, Image.Image):
            return image_field.convert("RGB")
        raise ValueError(f"Unsupported image field type: {type(image_field)}")

    @staticmethod
    def _normalize_tokens(text):
        """NFKC/casefold; strip ordinary punctuation/articles, preserve numbers."""
        text = unicodedata.normalize("NFKC", str(text)).casefold()
        text = re.sub(r"(?<=\d),(?=\d)", "", text)  # 1,200 -> 1200
        text = text.replace("\\_", "_")
        text = re.sub(r"[\x60$\\{}\[\]()]+", " ", text)
        tokens = re.findall(
            r"-?\d+(?:\.\d+)?%?|[^\W_]+(?:'[^\W_]+)?", text, flags=re.UNICODE
        )
        return [t for t in tokens if t not in {"a", "an", "the"}]

    @classmethod
    def _concepts(cls, concise):
        parts = [p.strip() for p in re.split(r"[,，]", str(concise)) if p.strip()]
        concepts = [cls._normalize_tokens(p) for p in parts]
        return [c for c in concepts if c]

    @staticmethod
    def _contains_span(sequence, span):
        if not span or len(span) > len(sequence):
            return False
        n = len(span)
        return any(
            sequence[i:i + n] == span for i in range(len(sequence) - n + 1)
        )

    @staticmethod
    def _prf(pred_tokens, gold_tokens):
        overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        precision = overlap / len(pred_tokens) if pred_tokens else 0.0
        recall = overlap / len(gold_tokens) if gold_tokens else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        return precision, recall, f1

    @staticmethod
    def _lcs_len(a, b):
        if len(a) < len(b):
            a, b = b, a
        prev = [0] * (len(b) + 1)
        for x in a:
            cur = [0]
            for j, y in enumerate(b, 1):
                cur.append(
                    prev[j - 1] + 1 if x == y else max(prev[j], cur[-1])
                )
            prev = cur
        return prev[-1]

    @classmethod
    def _rouge_l(cls, pred, references):
        pred_tokens = cls._normalize_tokens(pred)
        best = (0.0, 0.0, 0.0)
        for ref in references:
            ref_tokens = cls._normalize_tokens(ref)
            lcs = cls._lcs_len(pred_tokens, ref_tokens)
            precision = lcs / len(pred_tokens) if pred_tokens else 0.0
            recall = lcs / len(ref_tokens) if ref_tokens else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )
            if f1 > best[2]:
                best = (precision, recall, f1)
        return best

    @classmethod
    def _metrics_token_f1(cls, preds, refs, extra_info):
        if extra_info is None or len(extra_info) != len(preds):
            raise ValueError("token_f1 requires per-sample answer_full metadata")

        ps, rs, fs, kcs = [], [], [], []
        rouge_ps, rouge_rs, rouge_fs, lengths = [], [], [], []
        for pred, ref_list, meta in zip(preds, refs, extra_info):
            if isinstance(ref_list, str):
                ref_list = [ref_list]
            pred_tokens = cls._normalize_tokens(pred)
            lengths.append(len(pred_tokens))
            best_prf, best_kc = (0.0, 0.0, 0.0), 0.0
            for ref in ref_list:
                current = cls._prf(pred_tokens, cls._normalize_tokens(ref))
                if current[2] > best_prf[2]:
                    best_prf = current
                concepts = cls._concepts(ref)
                kc = (
                    sum(cls._contains_span(pred_tokens, c) for c in concepts)
                    / len(concepts) if concepts else 0.0
                )
                best_kc = max(best_kc, kc)
            ps.append(best_prf[0])
            rs.append(best_prf[1])
            fs.append(best_prf[2])
            kcs.append(best_kc)

            full_refs = meta.get("answer_full")
            if full_refs is None:
                full_refs = []
            if isinstance(full_refs, str):
                full_refs = [full_refs]
            rp, rr, rf = cls._rouge_l(pred, full_refs)
            rouge_ps.append(rp)
            rouge_rs.append(rr)
            rouge_fs.append(rf)

        mean = lambda values: float(np.mean(values)) if values else 0.0
        return {
            "token_precision": mean(ps),
            "token_recall": mean(rs),
            "token_f1": mean(fs),
            "kc_recall": mean(kcs),
            "rouge_l_precision": mean(rouge_ps),
            "rouge_l_recall": mean(rouge_rs),
            "rouge_l_f1": mean(rouge_fs),
            "avg_output_tokens": mean(lengths),
            "token_precision_scores": ps,
            "token_recall_scores": rs,
            "token_f1_scores": fs,
            "kc_recall_scores": kcs,
            "rouge_l_precision_scores": rouge_ps,
            "rouge_l_recall_scores": rouge_rs,
            "rouge_l_f1_scores": rouge_fs,
            "output_token_counts": lengths,
            "metric_version": cls.METRIC_VERSION,
            "total_samples": len(preds),
        }

    @staticmethod
    def _metrics_relaxed_accuracy(preds, refs):
        scores = []
        for pred, ref_list in zip(preds, refs):
            ref_list = [ref_list] if isinstance(ref_list, str) else ref_list
            p = str(pred).strip().lower()
            scores.append(float(any(
                str(ref).strip().lower() in p for ref in ref_list
                if str(ref).strip()
            )))
        return {
            "relaxed_accuracy": float(np.mean(scores)),
            "relaxed_accuracy_scores": scores,
            "total_samples": len(preds),
        }

    @staticmethod
    def _metrics_anls(preds, refs):
        scores = []
        for pred, ref_list in zip(preds, refs):
            ref_list = [ref_list] if isinstance(ref_list, str) else ref_list
            best = 0.0
            for ref in ref_list:
                p, r = str(pred).strip().lower(), str(ref).strip().lower()
                if not r:
                    continue
                max_len = max(len(p), len(r))
                norm = editdistance.eval(p, r) / max_len if max_len else 0.0
                best = max(best, 1 - norm)
            scores.append(best if best >= 0.5 else 0.0)
        return {"anls": float(np.mean(scores)), "anls_scores": scores,
                "total_samples": len(preds)}

    @staticmethod
    def _metrics_bleu(preds, refs):
        scores, smooth = [], SmoothingFunction().method1
        for pred, ref_list in zip(preds, refs):
            ref_list = [ref_list] if isinstance(ref_list, str) else ref_list
            references = [str(ref).lower().split() for ref in ref_list]
            scores.append(sentence_bleu(
                references, str(pred).lower().split(), smoothing_function=smooth
            ))
        return {"bleu": float(np.mean(scores)), "bleu_scores": scores,
                "total_samples": len(preds)}

    @staticmethod
    def _metrics_relaxed_accuracy_80(preds, refs):
        scores = []
        for pred, ref_list in zip(preds, refs):
            ref_list = [ref_list] if isinstance(ref_list, str) else ref_list
            pchars = set(str(pred).strip().lower())
            best = 0.0
            for ref in ref_list:
                rchars = set(str(ref).strip().lower())
                if rchars and len(pchars & rchars) / len(rchars) >= 0.8:
                    best = 1.0
                    break
            scores.append(best)
        return {
            "relaxed_accuracy_80": float(np.mean(scores)),
            "relaxed_accuracy_80_scores": scores,
            "total_samples": len(preds),
        }

    @classmethod
    def _metrics_relaxed_acc_keywords(cls, preds, refs):
        """Legacy key name; now strict token-span KC-Recall."""
        scores = []
        for pred, ref_list in zip(preds, refs):
            ref_list = [ref_list] if isinstance(ref_list, str) else ref_list
            pred_tokens = cls._normalize_tokens(pred)
            best = 0.0
            for ref in ref_list:
                concepts = cls._concepts(ref)
                value = (
                    sum(cls._contains_span(pred_tokens, c) for c in concepts)
                    / len(concepts) if concepts else 0.0
                )
                best = max(best, value)
            scores.append(best)
        return {
            "relaxed_acc_keywords": float(np.mean(scores)),
            "relaxed_acc_keywords_scores": scores,
            "total_samples": len(preds),
        }

    @classmethod
    def metrics(cls, preds, refs, extra_info=None, lambda1=0.6,
                metric_type="token_f1"):
        if metric_type == "token_f1":
            return cls._metrics_token_f1(preds, refs, extra_info)
        if metric_type == "anls":
            return cls._metrics_anls(preds, refs)
        if metric_type == "relaxed_accuracy":
            return cls._metrics_relaxed_accuracy(preds, refs)
        if metric_type == "relaxed_accuracy_80":
            return cls._metrics_relaxed_accuracy_80(preds, refs)
        if metric_type == "bleu":
            return cls._metrics_bleu(preds, refs)
        if metric_type == "relaxed_acc_keywords":
            return cls._metrics_relaxed_acc_keywords(preds, refs)
        if metric_type == "weighted":
            if extra_info is None:
                raise ValueError("weighted requires answer_full metadata")
            kc = cls._metrics_relaxed_acc_keywords(preds, refs)
            bleu = cls._metrics_bleu(preds, [m["answer_full"] for m in extra_info])
            scores = [
                lambda1 * x + (1 - lambda1) * y
                for x, y in zip(
                    kc["relaxed_acc_keywords_scores"], bleu["bleu_scores"]
                )
            ]
            return {
                **kc, **bleu,
                "weighted": float(np.mean(scores)),
                "weighted_scores": scores,
                "lambda1": lambda1,
                "lambda2": 1 - lambda1,
                "total_samples": len(preds),
            }
        raise ValueError(f"Unsupported metric_type: {metric_type}")
