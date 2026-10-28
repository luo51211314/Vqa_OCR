"""Reading-Order Sort (ROS)：分辨率无关的 OCR block 阅读顺序排序。

替代旧实现中两套互相矛盾的排序：
- ``expert/ocr_expert.py`` 里仅按 ``(center_x, center_y)`` 的列优先排序
  （会破坏多行阅读顺序，且与论文图 3 的 TB-YX 标注不一致）；
- ``dbscan_clustering.py`` 里写死像素 ``epsilon=10`` 的 HDBSCAN 行聚类
  （分辨率变化即失效，噪声点被静默丢弃）。

算法（论文式(8) 的唯一权威实现，与图 3 同名 ROS）：

1. 每个 block 的几何量统一归一化到 ``[0, 1]``（除以图像宽/高），
   中心点 ``p_i = (cx, cy)``，行高 ``h_i = y2 - y1``；
2. 行分组：按 cy 升序扫描，以中位行高 ``h̃`` 为尺度，
   相邻中心距离 ``> rho * h̃`` 时新开一行；垂直重叠超过 50% 的碎片
   并入相邻行（处理跨行碎片）；
3. 行内按 cx 升序，行间按行首 cy 升序输出（top-to-bottom, left-to-right）。

本模块只做排序，不删除、不修改任何 block（无损）。
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Tuple

# 旧参数名 -> 规范名。保留一个版本周期的别名，避免旧脚本失效。
ORDER_ALIASES = {
    "normal": "native",
    "no_clustering": "native",
    "LR-TB": "ros",
    "lrtb": "ros",
    "TB-YX": "ros",
    "tbyx": "ros",
    "direct": "coordinate",
}

VALID_ORDERS = ("native", "random", "ros", "coordinate")


def normalize_order(order: str) -> str:
    """把历史命名（normal/LR-TB/direct/no_clustering）映射为规范命名。"""
    if order is None:
        return "native"
    key = str(order).strip()
    key = ORDER_ALIASES.get(key, key)
    if key not in VALID_ORDERS:
        raise ValueError(
            f"未知 OCR 顺序 '{order}'，可选: {VALID_ORDERS}（兼容旧名 normal/LR-TB/direct）"
        )
    return key


def _geom(block: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    """返回归一化几何量 (x1, y1, x2, y2, cx, cy)，全部在 [0,1] 量级。

    优先使用 ocr_expert 写入的归一化字段 nminx/nminy/nmaxx/nmaxy；
    否则用像素 bbox + image_width/image_height 现场归一化；
    再不行退回已归一化的 center_x/center_y（旧 schema）。
    """
    if all(k in block for k in ("nminx", "nminy", "nmaxx", "nmaxy")):
        x1, y1, x2, y2 = (block["nminx"], block["nminy"], block["nmaxx"], block["nmaxy"])
    else:
        w = block.get("image_width")
        h = block.get("image_height")
        if w and h:
            x1 = block.get("minx", 0.0) / w
            x2 = block.get("maxx", w) / w
            y1 = block.get("miny", 0.0) / h
            y2 = block.get("maxy", h) / h
        else:
            cx = block.get("center_x", 0.5)
            cy = block.get("center_y", 0.5)
            return cx, cy, cx, cy, cx, cy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return x1, y1, x2, y2, cx, cy


def _vertical_overlap_ratio(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """两个 [y1,y2] 区间的垂直重叠 / 较小区间高度。"""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    shorter = max(1e-9, min(a[1] - a[0], b[1] - b[0]))
    return inter / shorter


def reading_order_sort(blocks: Sequence[Dict[str, Any]], rho: float = 0.5) -> List[Dict[str, Any]]:
    """ROS：行分组 + 行内从左到右 + 行间从上到下。

    Args:
        blocks: OCR block 列表（dict，至少含归一化几何字段或像素 bbox）。
        rho: 行分组阈值相对中位行高的比例，默认 0.5。值越大越倾向合并为一行。
    """
    if not blocks:
        return []

    items = []
    for b in blocks:
        x1, y1, x2, y2, cx, cy = _geom(b)
        items.append({"block": b, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "cx": cx, "cy": cy, "h": max(1e-6, y2 - y1)})

    heights = sorted(it["h"] for it in items)
    median_h = heights[len(heights) // 2]
    gap_threshold = rho * median_h

    # 1) 按中心 y 初步排序
    items.sort(key=lambda it: it["cy"])

    # 2) 贪心行分组：cy 间距超阈值且垂直重叠不足 -> 新行
    rows: List[List[dict]] = []
    for it in items:
        placed = False
        for row in rows:
            row_cy = sum(r["cy"] for r in row) / len(row)
            close_by_center = abs(it["cy"] - row_cy) <= gap_threshold
            overlaps = any(
                _vertical_overlap_ratio((it["y1"], it["y2"]), (r["y1"], r["y2"])) > 0.5
                for r in row
            )
            if close_by_center or overlaps:
                row.append(it)
                placed = True
                break
        if not placed:
            rows.append([it])

    # 3) 行内从左到右；行间按行首（平均）cy 从上到下
    for row in rows:
        row.sort(key=lambda it: it["cx"])
    rows.sort(key=lambda row: sum(r["cy"] for r in row) / len(row))

    return [it["block"] for row in rows for it in row]


def order_blocks(
    blocks: Sequence[Dict[str, Any]],
    order: str = "native",
    seed: int = 42,
    rho: float = 0.5,
) -> List[Dict[str, Any]]:
    """按指定顺序返回 block 序列（不修改原列表内容）。

    - native:     OCR 引擎原始输出顺序（主表配置）
    - random:     固定种子随机打乱（顺序携带信息量的对照）
    - ros:        Reading-Order Sort（核心排序算法）
    - coordinate: 顺序不重排，仅提示调用方使用逐块坐标格式渲染
    """
    order = normalize_order(order)
    ordered = list(blocks)
    if order == "random":
        rng = random.Random(seed)
        rng.shuffle(ordered)
    elif order == "ros":
        ordered = reading_order_sort(ordered, rho=rho)
    return ordered
