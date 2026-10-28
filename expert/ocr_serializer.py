"""OCR prompt 序列化器 φ（论文式(10) 的唯一权威实现）。

回应审稿意见："缺少完整 prompt 模板、分隔符、转义、最大 OCR 长度、截断策略、
token 增量"。本模块给出：

1. 三套可逐字引用的模板（P1/P2/P3，供 prompt 措辞消融）和一套坐标模板；
2. 固定分隔符与文本清洗（sanitize）规则；
3. 以目标模型 tokenizer 的 **token 数**为预算，按 block 边界整块丢弃，
   绝不从 block 中间硬切（旧实现是 ``prompt[:3000] + "..."`` 的字符截断）；
4. 每次渲染返回 token 统计（token 增量 Δtokens、丢弃块数）。

设计约束：本模块是**无损层**——除空文本块外不删除任何 block、不改写识别
文本。置信度过滤/去重/纠错等有损操作属于 L2，见 ``Algorithm/ocr_filters.py``。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence

try:  # 以仓库根目录启动（python -m main_inference）
    from Algorithm.reading_order import normalize_order, order_blocks
except ImportError:  # 作为子包被其它路径引用时
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "Algorithm"))
    from reading_order import normalize_order, order_blocks

# ---------------------------------------------------------------------------
# 模板定义：{ocr} 为 OCR 段，{question} 为问题。模型侧的答案风格后缀
# （Answer briefly ...）不在此处，由各 model_loader 统一追加 common_suffix。
# ---------------------------------------------------------------------------
TEMPLATES: Dict[str, str] = {
    # P1：极简
    "p1": "OCR: {ocr}\nQ: {question}\nA:",
    # P2：陈述式（主表默认）
    "p2": (
        "OCR content (in reading order): {ocr}\n"
        "Question: {question}\n"
        "Answer the question based on the OCR content and the image."
    ),
    # P3：旧版指令式（历史对照）
    "p3": (
        "the ocr text: {ocr}\n\n"
        "Based on the above information, please answer the following question: {question}"
    ),
}

# 坐标模式模板（对应消融里的 coordinate / 旧 direct）
COORDINATE_TEMPLATE = (
    "OCR blocks:\n{blocks}\n"
    "Question: {question}\n"
    "Answer the question based on the OCR blocks and the image."
)

# 分隔符规范（论文中逐字给出，不得在别处另造）
TEXT_BLOCK_SEPARATOR = ", "
COORD_FIELD_SEPARATOR = " | "
COORD_BLOCK_PREFIX = "[{i}]"
COORD_BBOX_OPEN = "bbox=("
COORD_BBOX_CLOSE = ")"

_WHITESPACE_RE = re.compile(r"\s+")

EMPTY_OCR_PROMPT = "the ocr text: not found ocr text"


def sanitize_text(text: Any) -> str:
    """L1 清洗：换行/制表符转空格、连续空白合并、去首尾空白。

    不删除标点、不改写字符（保持识别文本不变）。坐标模式下 OCR 文本已被
    清洗掉换行，因此一行一块的解析不会被文本内换行破坏。
    """
    if text is None:
        return ""
    return _WHITESPACE_RE.sub(" ", str(text)).strip()


def _normalized_bbox(block: Dict[str, Any]) -> Optional[tuple]:
    """取归一化 bbox (x1,y1,x2,y2)；无法得到时返回 None。"""
    if all(k in block for k in ("nminx", "nminy", "nmaxx", "nmaxy")):
        return block["nminx"], block["nminy"], block["nmaxx"], block["nmaxy"]
    w, h = block.get("image_width"), block.get("image_height")
    if w and h and all(k in block for k in ("minx", "miny", "maxx", "maxy")):
        return block["minx"] / w, block["miny"] / h, block["maxx"] / w, block["maxy"] / h
    return None


def _render_text_segment(blocks: Sequence[Dict[str, Any]]) -> str:
    return TEXT_BLOCK_SEPARATOR.join(b["_text"] for b in blocks)


def _render_coordinate_segment(blocks: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for i, b in enumerate(blocks, 1):
        prefix = COORD_BLOCK_PREFIX.format(i=i)
        bbox = _normalized_bbox(b)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bbox_str = f"{COORD_BBOX_OPEN}{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}{COORD_BBOX_CLOSE}"
            lines.append(f"{prefix} {bbox_str}{COORD_FIELD_SEPARATOR}{b['_text']}")
        else:
            lines.append(f"{prefix}{COORD_FIELD_SEPARATOR}{b['_text']}")
    return "\n".join(lines)


def _count_tokens(text: str, token_counter: Optional[Callable[[str], int]]) -> Optional[int]:
    if token_counter is None:
        return None
    n = int(token_counter(text))
    return max(0, n)


def build_ocr_prompt(
    blocks: Sequence[Dict[str, Any]],
    question: str,
    *,
    order: str = "native",
    template: str = "p2",
    budget_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    random_seed: int = 42,
    ros_rho: float = 0.5,
) -> Dict[str, Any]:
    """把 OCR blocks 序列化为最终 prompt（φ）。

    Args:
        blocks: OCR block dict 列表（ocr_expert 的输出 schema）。
        question: 问题文本。
        order: native / random / ros / coordinate（兼容旧名 normal/LR-TB/direct）。
        template: p1 / p2 / p3（仅文本模式生效；coordinate 顺序强制坐标模板）。
        budget_tokens: OCR 段 token 预算；None 或 0 表示不限。超出预算的 block
            按阅读顺序从尾部**整块丢弃**。
        token_counter: 目标模型 tokenizer 计数函数 fn(str)->int。
            设置了预算则必须提供；未提供时忽略预算并在 stats 中标记。
        random_seed: random 顺序的固定种子。
        ros_rho: ROS 行分组阈值。

    Returns:
        {"prompt": str, "stats": {...}}，stats 含：
        order/template、n_blocks_in、n_blocks_dropped、budget_active、
        total_tokens、ocr_tokens、question_tokens、delta_tokens（无 tokenizer 时为 None）。
    """
    norm_order = normalize_order(order)
    question = "" if question is None else str(question)

    # 1) L1 清洗 + 丢弃空文本块（唯一允许的删除）
    cleaned: List[Dict[str, Any]] = []
    for b in blocks or []:
        t = sanitize_text(b.get("text", ""))
        if t:
            nb = dict(b)
            nb["_text"] = t
            cleaned.append(nb)

    stats: Dict[str, Any] = {
        "order": norm_order,
        "template": "coordinate" if norm_order == "coordinate" else template,
        "n_blocks_in": len(cleaned),
        "n_blocks_dropped": 0,
        "budget_active": False,
        "budget_tokens": budget_tokens or None,
        "total_tokens": None,
        "ocr_tokens": None,
        "question_tokens": None,
        "delta_tokens": None,
    }

    if not cleaned:
        stats["n_blocks_in"] = 0
        return {"prompt": EMPTY_OCR_PROMPT, "stats": stats}

    # 2) 排序（coordinate 保持原生顺序，仅改用坐标格式渲染）
    ordered = order_blocks(
        cleaned,
        order=norm_order,
        seed=random_seed,
        rho=ros_rho,
    )

    is_coordinate = norm_order == "coordinate"

    # 3) 预算：按 block 边界整块丢弃（只可能裁掉尾部 block）
    budget = budget_tokens or 0
    if budget and budget > 0 and token_counter is not None:
        stats["budget_active"] = True
        kept = _fit_budget(ordered, question, is_coordinate, template, token_counter, budget)
        dropped = len(ordered) - len(kept)
        ordered = kept
        stats["n_blocks_dropped"] = dropped
    elif budget and budget > 0 and token_counter is None:
        # 显式给了预算却没给 tokenizer：不能静默退化为不限
        raise ValueError(
            "设置了 budget_tokens 但未提供 token_counter，无法按 token 预算截断；"
            "请在主程序中注入目标模型的 tokenizer 计数函数。"
        )

    # 4) 渲染
    if is_coordinate:
        body = COORDINATE_TEMPLATE.format(
            blocks=_render_coordinate_segment(ordered), question=question
        )
    else:
        if template not in TEMPLATES:
            raise ValueError(f"未知 prompt 模板 '{template}'，可选: {sorted(TEMPLATES)}")
        body = TEMPLATES[template].format(
            ocr=_render_text_segment(ordered), question=question
        )

    # 5) token 统计
    if token_counter is not None:
        total = _count_tokens(body, token_counter)
        ocr_part = (
            _render_coordinate_segment(ordered) if is_coordinate else _render_text_segment(ordered)
        )
        ocr_tokens = _count_tokens(ocr_part, token_counter)
        question_tokens = _count_tokens(question, token_counter)
        stats["total_tokens"] = total
        stats["ocr_tokens"] = ocr_tokens
        stats["question_tokens"] = question_tokens
        stats["delta_tokens"] = (total - question_tokens) if question_tokens is not None else total

    return {"prompt": body, "stats": stats}


def _fit_budget(
    ordered: Sequence[Dict[str, Any]],
    question: str,
    is_coordinate: bool,
    template: str,
    token_counter: Callable[[str], int],
    budget: int,
) -> List[Dict[str, Any]]:
    """贪心装入：保证 OCR 段 token 数不超过 budget，按整块丢弃尾部 block。"""
    kept: List[Dict[str, Any]] = []
    for b in ordered:
        candidate = kept + [b]
        segment = (
            _render_coordinate_segment(candidate)
            if is_coordinate
            else _render_text_segment(candidate)
        )
        if _count_tokens(segment, token_counter) <= budget:
            kept = candidate
        else:
            # 后续 block 不保证更短，但按"保留阅读顺序前部"的策略直接停止
            break
    return kept
