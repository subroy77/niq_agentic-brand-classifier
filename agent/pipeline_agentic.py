from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import dspy

from .subcategory_classifier import sub_category_module
from .validator_agent import validator_module
from .clarification_agent import clarification_module


LOW_CONF_THRESHOLD = 0.6
CLARIFICATION_ON = True


class BrandCategorySignature(dspy.Signature):
    """
    Extract primary brand and high-level category from a user query.

    No fixed taxonomy here. The model should:
    - extract a single primary brand, or "Unknown" if none.
    - infer a short category label like:
      "Electronics", "Home Appliances", "Streaming Service",
      "Retail", "Media & Entertainment", "General Inquiry", etc.
    """
    query = dspy.InputField(desc="User's query/prompt")
    brand = dspy.OutputField(
        desc="Primary brand name mentioned, or 'Unknown' if none."
    )
    category = dspy.OutputField(
        desc="High-level category label (short phrase)."
    )
    confidence = dspy.OutputField(
        desc="Model's confidence between 0 and 1."
    )


brand_category_module = dspy.Predict(BrandCategorySignature)


@dataclass
class ClassificationResult:
    query: str
    brand: str
    category: str
    sub_category: str
    category_confidence: float
    sub_category_confidence: float
    is_consistent: bool
    validation_reason: Optional[str]
    needs_clarification: bool
    clarification_question: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_conf(value: Any, default: float = 0.5) -> float:
    try:
        f = float(value)
    except Exception:
        return default
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def classify_single_query(query: str) -> ClassificationResult:
    """
    Full agentic pipeline for a single query:

    1. Brand + category classification.
    2. Sub-category classification.
    3. Validation (consistency check).
    4. Clarification trigger when needed (agentic behaviour).
    """

    bc = brand_category_module(
        query=(
            "You are an NIQ brand and category classifier.\n"
            "For the given query, extract:\n"
            "- primary brand name (or 'Unknown' if not clear),\n"
            "- a short high-level category label (e.g. 'Electronics', "
            "'Home Appliances', 'Streaming Service', 'Media', 'Retail', "
            "'General Inquiry').\n"
            "Return a confidence between 0 and 1.\n\n"
            f"User query: {query}"
        )
    )

    brand = (bc.brand or "Unknown").strip()
    category = (bc.category or "Unknown").strip()
    cat_conf = _safe_conf(getattr(bc, "confidence", 0.5))

    if category.lower() == "unknown":
        sub_category = "Unknown"
        sub_conf = 0.0
    else:
        sc = sub_category_module(
            query=(
                "You refine categories into more granular sub-categories.\n"
                "Given the user query and its high-level category, propose a "
                "short, business-meaningful sub-category under that category.\n"
                "If you are not confident or nothing fits, respond with "
                "'Other'. Also return a confidence between 0 and 1.\n\n"
                f"User query: {query}"
            ),
            category=category,
        )
        sub_category = (sc.sub_category or "Other").strip()
        sub_conf = _safe_conf(getattr(sc, "confidence", 0.5))

    validation = validator_module(
        query=query,
        brand=brand,
        category=category,
        sub_category=sub_category,
    )

    is_consistent = str(validation.is_consistent).strip().lower() == "yes"
    validation_reason = getattr(validation, "reason", None)

    reasons: List[str] = []

    if cat_conf < LOW_CONF_THRESHOLD:
        reasons.append(f"low category confidence ({cat_conf:.2f})")

    if sub_category.lower() != "other" and sub_conf < LOW_CONF_THRESHOLD:
        reasons.append(f"low sub-category confidence ({sub_conf:.2f})")

    if not is_consistent:
        reasons.append("validator marked labels as inconsistent")

    needs_clarification = bool(reasons) and CLARIFICATION_ON
    clarification_question: Optional[str] = None

    if needs_clarification:
        cl = clarification_module(
            query=query,
            issue="; ".join(reasons),
        )
        clarification_question = getattr(cl, "question", None)

    return ClassificationResult(
        query=query,
        brand=brand,
        category=category,
        sub_category=sub_category,
        category_confidence=cat_conf,
        sub_category_confidence=sub_conf,
        is_consistent=is_consistent,
        validation_reason=validation_reason,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
    )


def classify_queries(queries: List[str]) -> List[ClassificationResult]:
    return [classify_single_query(q) for q in queries]
