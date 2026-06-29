"""Safety constants and validation helpers for public candidate views."""

from __future__ import annotations

from collections.abc import Iterable


FORBIDDEN_IMPORT_MODULES: tuple[str, ...] = (
    "BOReferencePolicy",
    "evaluation.bo_reference",
    "csebo_harness",
)

FORBIDDEN_SECRET_TERMS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "authorization",
    "bearer ",
    "openai_api_key",
)

FORBIDDEN_CANDIDATE_FIELDS: tuple[str, ...] = (
    "candidate_scores.csv",
    "bo_rank",
    "bo_top_k",
    "bo_topk",
    "acquisition_score",
    "predictive_mean",
    "predictive_std",
    "surrogate_uncertainty",
    "oracle_rank",
    "hidden_outcome",
    "hidden_y",
    "unobserved_y",
    "outcome",
    "yield",
    "y",
    "turnover",
    "raw_row_index",
    "source_path",
)

FORBIDDEN_OBSERVED_FIELDS: tuple[str, ...] = (
    "candidate_scores.csv",
    "bo_rank",
    "bo_top_k",
    "bo_topk",
    "acquisition_score",
    "predictive_mean",
    "predictive_std",
    "surrogate_uncertainty",
    "oracle_rank",
    "hidden_outcome",
    "hidden_y",
    "unobserved_y",
    "turnover",
    "raw_row_index",
    "source_path",
)


def normalize_field_name(name: object) -> str:
    """Normalize a field name for conservative substring checks."""

    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def find_forbidden_fields(
    fields: Iterable[object],
    forbidden_terms: Iterable[str],
) -> list[str]:
    """Return fields whose normalized names contain forbidden contract terms."""

    normalized_terms = tuple(normalize_field_name(term) for term in forbidden_terms)
    matches: list[str] = []
    for field in fields:
        normalized = normalize_field_name(field)
        if any(
            normalized == term if len(term) <= 2 else term in normalized
            for term in normalized_terms
        ):
            matches.append(str(field))
    return matches


def assert_no_forbidden_fields(
    fields: Iterable[object],
    forbidden_terms: Iterable[str],
    *,
    context: str,
) -> None:
    """Raise if a view includes hidden, oracle, BO, provenance, or secret fields."""

    matches = find_forbidden_fields(fields, forbidden_terms)
    if matches:
        joined = ", ".join(matches)
        raise ValueError(f"{context} includes forbidden fields: {joined}")
