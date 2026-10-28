"""L2 有损 OCR 预处理：置信度过滤 / 几何过滤 / 去重。

回应审稿意见："Token 过滤出现在多处，但没有置信阈值、几何规则、去重规则"。
本模块的每个过滤器都有**显式阈值参数**和返回统计，主表配置下全部关闭
（conf_threshold=0、不去重、不做几何过滤），仅在消融实验中按单因子开启。

与 ``expert/ocr_serializer.py``（L1 无损层）严格区分：
这里的任何操作都可能删除 block，必须在论文中按有损模块报告。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

_WS_RE = re.compile(r"\s+")


def confidence_filter(
    blocks: Sequence[Dict[str, Any]],
    threshold: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """保留 ``confidence >= threshold`` 的 block。

    confidence 为 None（引擎未返回）时视为缺失，**不参与阈值过滤、予以保留**
    （旧实现把缺失置信度伪造成 0.9，会导致静默偏差，已移除）。
    threshold=0 时等价于不过滤（主表配置）。
    """
    if not threshold or threshold <= 0:
        return list(blocks), {"kept": len(blocks), "dropped": 0, "missing_confidence": 0}

    kept, missing = [], 0
    for b in blocks:
        c = b.get("confidence")
        if c is None:
            missing += 1
            kept.append(b)
        elif float(c) >= threshold:
            kept.append(b)
    return kept, {"kept": len(kept), "dropped": len(blocks) - len(kept),
                  "missing_confidence": missing}


def normalize_text(text: Any) -> str:
    """去重使用的文本归一化：小写 + 空白折叠 + 去首尾标点空白。"""
    s = _WS_RE.sub(" ", str(text or "").lower()).strip()
    return s


def _center_of(b: Dict[str, Any]) -> Tuple[float, float]:
    """取归一化中心：优先 center_x/y，其次由 nmin..nmax 归一化 bbox 推导。"""
    try:
        return float(b["center_x"]), float(b["center_y"])
    except (KeyError, TypeError, ValueError):
        pass
    try:
        return ((float(b["nminx"]) + float(b["nmaxx"])) / 2.0,
                (float(b["nminy"]) + float(b["nmaxy"])) / 2.0)
    except (KeyError, TypeError, ValueError):
        return 0.5, 0.5


def _has_geometry(b: Dict[str, Any]) -> bool:
    return any(k in b for k in ("center_x", "nminx"))


def _center_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """归一化坐标系下的中心欧氏距离；两块都无几何字段时返回 1.0（视为不近）。"""
    if not (_has_geometry(a) and _has_geometry(b)):
        return 1.0
    ax, ay = _center_of(a)
    bx, by = _center_of(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def dedup_blocks(
    blocks: Sequence[Dict[str, Any]],
    center_dist_threshold: float = 0.02,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """精确去重 + 包含关系去重。

    规则（二者满足其一即视为重复，保留文本更长的一块）：
    1. 归一化文本完全相同，且中心点归一化距离 < center_dist_threshold；
    2. 一块文本是另一块的子串，且中心距离 < 阈值（OCR 重复行/包含框）。
    """
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for b in blocks:
        t = normalize_text(b.get("text", ""))
        is_dup = False
        for k in kept:
            kt = normalize_text(k.get("text", ""))
            if not t or not kt:
                continue
            near = _center_distance(b, k) < center_dist_threshold
            if not near:
                continue
            same_or_contained = (t == kt) or (t in kt) or (kt in t)
            if same_or_contained:
                is_dup = True
                # 保留更长的文本
                if len(t) > len(kt):
                    kept[kept.index(k)] = b
                break
        if is_dup:
            dropped += 1
        else:
            kept.append(b)
    return kept, {"kept": len(kept), "dropped": dropped}


def geometric_filter(
    blocks: Sequence[Dict[str, Any]],
    min_height_ratio: float = 0.002,
    min_width_ratio: float = 0.002,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """几何过滤：丢弃过小框/越界框（需要归一化 bbox 字段，否则跳过该块判断）。"""
    kept = []
    dropped = 0
    for b in blocks:
        if not all(k in b for k in ("nminx", "nminy", "nmaxx", "nmaxy")):
            kept.append(b)  # 无几何信息时不做判断
            continue
        w = b["nmaxx"] - b["nminx"]
        h = b["nmaxy"] - b["nminy"]
        out_of_bounds = (
            b["nmaxx"] < 0 or b["nmaxy"] < 0 or b["nminx"] > 1 or b["nminy"] > 1
        )
        if out_of_bounds or w < min_width_ratio or h < min_height_ratio:
            dropped += 1
            continue
        kept.append(b)
    return kept, {"kept": len(kept), "dropped": dropped}


def apply_l2_filters(
    blocks: Sequence[Dict[str, Any]],
    *,
    conf_threshold: float = 0.0,
    dedup: bool = False,
    geometric: bool = False,
    dedup_center_dist: float = 0.02,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """按固定顺序应用 L2 过滤器并汇总统计。主表全部参数为默认（无损等价）。"""
    stats: Dict[str, Any] = {"confidence_filter": None, "dedup": None, "geometric": None}
    current = list(blocks)

    current, stats["confidence_filter"] = confidence_filter(current, conf_threshold)
    if geometric:
        current, stats["geometric"] = geometric_filter(current)
    if dedup:
        current, stats["dedup"] = dedup_blocks(current, dedup_center_dist)
    return current, stats
