"""Safety validation for future live research queries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


@dataclass(frozen=True)
class QuerySafetyResult:
    passed: bool
    rejection_reasons: list[str]
    matched_terms: list[str]
    sanitized_query: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FORBIDDEN_QUERY_TERMS: tuple[tuple[str, str], ...] = (
    ("exact dataset answer", "exact_dataset_answer"),
    ("complete benchmark table", "complete_benchmark_table"),
    ("candidate-yield mapping", "candidate_yield_mapping"),
    ("candidate yield mapping", "candidate_yield_mapping"),
    ("candidate_yield_mapping", "candidate_yield_mapping"),
    ("yield by candidate", "candidate_yield_mapping"),
    ("best condition for this exact replay dataset", "exact_replay_best_condition"),
    ("target values for the current replay candidates", "current_replay_targets"),
    ("current replay candidates", "current_replay_targets"),
    ("leaderboard solution", "leaderboard_solution"),
    ("benchmark answer key", "benchmark_answer_key"),
    ("answer key", "benchmark_answer_key"),
    ("candidate ids", "candidate_ids"),
    ("candidate_id", "candidate_ids"),
    ("hidden outcome", "hidden_outcome"),
    ("hidden_y", "hidden_outcome"),
    ("hidden_yield", "hidden_outcome"),
    ("unobserved_y", "hidden_outcome"),
    ("oracle rank", "oracle_or_posthoc_rank"),
    ("oracle_rank", "oracle_or_posthoc_rank"),
    ("posthoc rank", "oracle_or_posthoc_rank"),
    ("posthoc_rank", "oracle_or_posthoc_rank"),
    ("global rank", "oracle_or_posthoc_rank"),
    ("global_rank", "oracle_or_posthoc_rank"),
    ("full-pool true rank", "oracle_or_posthoc_rank"),
    ("full_pool_true_rank", "oracle_or_posthoc_rank"),
    ("evaluator artifact", "evaluator_artifact"),
    ("evaluator internals", "evaluator_artifact"),
    ("candidate_scores.csv", "reference_bo_output"),
    ("reference bo output", "reference_bo_output"),
    ("bo reference output", "reference_bo_output"),
    ("reference acquisition", "reference_bo_output"),
    ("reference predictive", "reference_bo_output"),
    ("private map", "private_map"),
    ("private candidate map", "private_map"),
    ("display_to_internal_id", "private_map"),
)


GENERIC_ALLOWED_HINTS: tuple[str, ...] = (
    "bayesian optimization",
    "active learning",
    "high-throughput experimentation",
    "high throughput experimentation",
    "surrogate modeling",
    "small-sample chemistry optimization",
    "acquisition functions",
    "experimental design",
    "suzuki coupling optimization",
    "uncertainty-aware candidate selection",
    "reaction optimization",
)


def validate_live_research_query(
    query: str,
    dataset_profile: dict[str, Any] | None = None,
    forbidden_context: dict[str, Any] | None = None,
) -> QuerySafetyResult:
    """Reject exact-answer and artifact-seeking queries before any network call."""

    text = str(query or "").strip()
    lowered = text.lower()
    reasons: list[str] = []
    matched_terms: list[str] = []
    for term, reason in FORBIDDEN_QUERY_TERMS:
        if term in lowered:
            reasons.append(reason)
            matched_terms.append(term)

    if re.search(r"\bcand[_-]?\d{3,}\b", lowered) or re.search(r"\bcandidate\s*(id|ids)\s*[:=]?\s*\w+", lowered):
        reasons.append("candidate_ids")
        matched_terms.append("candidate_id_pattern")

    profile = dict(dataset_profile or {})
    dataset_name = str(profile.get("dataset_name") or profile.get("name") or "").strip().lower()
    if dataset_name and dataset_name in lowered and _asks_for_exact_answer(lowered):
        reasons.append("exact_dataset_answer")
        matched_terms.append(dataset_name)
    candidate_ids = [str(value).lower() for value in profile.get("candidate_ids", [])]
    for candidate_id in candidate_ids[:1000]:
        if candidate_id and candidate_id in lowered:
            reasons.append("candidate_ids")
            matched_terms.append(candidate_id)
            break

    context = dict(forbidden_context or {})
    for label, values in context.items():
        if isinstance(values, (list, tuple, set)):
            context_values = [str(value).lower() for value in values]
        else:
            context_values = [str(values).lower()]
        for value in context_values:
            if value and value in lowered:
                reasons.append(f"forbidden_context:{label}")
                matched_terms.append(value)
                break

    unique_reasons = sorted(set(reasons))
    unique_terms = sorted(set(matched_terms))
    return QuerySafetyResult(
        passed=not unique_reasons,
        rejection_reasons=unique_reasons,
        matched_terms=unique_terms,
        sanitized_query=_sanitize_query(text),
    )


def _asks_for_exact_answer(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "best condition",
            "best candidate",
            "answer",
            "answer key",
            "target value",
            "yield table",
            "complete table",
            "leaderboard",
        )
    )


def _sanitize_query(query: str) -> str:
    return re.sub(r"\bcand[_-]?\d{3,}\b", "[candidate_id]", query, flags=re.IGNORECASE)
