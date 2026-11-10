"""Deterministic heuristics used both to sanity-check the LLM and as a fallback
when no LLM is configured. These are honest rule-based guesses — never presented
as if they came from the model.
"""
from __future__ import annotations

import re

from ..schemas.common import InferredType, ProblemType
from ..schemas.dataset import ColumnInfo, DatasetInfo

# Column-name tokens that very often denote a prediction target.
COMMON_TARGET_TOKENS = [
    "target", "label", "class", "outcome", "result", "response",
    "churn", "churned", "attrition", "default", "fraud", "fraudulent",
    "survived", "spam", "approved", "conversion", "converted", "clicked",
    "purchased", "subscribed", "diagnosis", "disease", "readmitted",
    "price", "cost", "revenue", "sales", "salary", "wage", "charges",
    "rating", "score", "quality", "demand", "amount", "value",
]


def _column_map(info: DatasetInfo) -> dict[str, ColumnInfo]:
    return {c.name: c for c in info.columns}


def _mentions(text: str, column: str) -> bool:
    """Whether a column name appears as a word/phrase in free text."""
    if len(column) < 3:
        return False
    variants = {column.lower(), column.lower().replace("_", " "), column.lower().replace(" ", "_")}
    for variant in variants:
        if re.search(rf"(?<![\w]){re.escape(variant)}(?![\w])", text):
            return True
    return False


def guess_target(info: DatasetInfo, user_query: str | None) -> tuple[str | None, float]:
    """Return (best_target_column, confidence in [0, 1])."""
    cols = info.column_names
    if not cols:
        return None, 0.0
    query = (user_query or "").lower()

    # 1) Column explicitly named in the request.
    mentioned = [c for c in cols if _mentions(query, c)]
    if len(mentioned) == 1:
        return mentioned[0], 0.95
    if len(mentioned) > 1:
        return mentioned[-1], 0.6

    # 2) A target-like token in the query maps to a column.
    for token in COMMON_TARGET_TOKENS:
        if re.search(rf"(?<![\w]){token}(?![\w])", query):
            for c in cols:
                if token in c.lower():
                    return c, 0.85

    # 3) A column literally named like a common target.
    for c in cols:
        if c.lower() in COMMON_TARGET_TOKENS:
            return c, 0.72
    for c in cols:
        if any(token in c.lower() for token in COMMON_TARGET_TOKENS):
            return c, 0.55

    # 4) Convention: the last column is often the target (low confidence).
    return cols[-1], 0.34


def guess_problem_type(info: DatasetInfo, target: str | None) -> ProblemType:
    """Infer classification vs regression from the target column."""
    if not target:
        return ProblemType.classification
    col = _column_map(info).get(target)
    if col is None:
        return ProblemType.classification
    if col.inferred_type in (InferredType.categorical, InferredType.text, InferredType.boolean):
        return ProblemType.classification
    if col.inferred_type == InferredType.numeric:
        # Few distinct numeric values -> encoded classes; many -> continuous.
        if col.unique <= 15:
            return ProblemType.classification
        return ProblemType.regression
    return ProblemType.classification


def default_metric(problem_type: ProblemType | str) -> str:
    pt = problem_type.value if isinstance(problem_type, ProblemType) else problem_type
    return "f1" if pt == "classification" else "r2"
