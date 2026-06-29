"""Public-safe method primitive extraction for tool portfolio mode."""

from __future__ import annotations

from typing import Any

from research_tool_agent_full_pool.artifact_logger import sanitize_text
from research_tool_agent_full_pool.decision_artifacts import assert_payload_public_safe


ALLOWED_PRIMITIVE_TYPES: tuple[str, ...] = (
    "feature_encoding",
    "surrogate_model",
    "uncertainty_estimation",
    "acquisition_rule",
    "diversity_or_density",
    "domain_prior",
    "ensemble_or_rank_aggregation",
    "diagnostic_guard",
)


_TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("feature_encoding", ("feature", "encoding", "descriptor", "categorical", "numeric", "scaling")),
    ("surrogate_model", ("surrogate", "regression", "model", "predict", "estimator", "fit")),
    ("uncertainty_estimation", ("uncertainty", "confidence", "variance", "exploration", "calibration")),
    ("acquisition_rule", ("acquisition", "ucb", "expected improvement", "exploit", "exploration-exploitation")),
    ("diversity_or_density", ("diversity", "density", "coverage", "distance", "neighborhood", "cluster")),
    ("domain_prior", ("domain", "reaction", "chem", "catalyst", "ligand", "solvent", "prior")),
    ("ensemble_or_rank_aggregation", ("ensemble", "aggregate", "aggregation", "rank", "blend", "voting")),
    ("diagnostic_guard", ("guard", "diagnostic", "robust", "fallback", "sanity", "stability")),
)


_CODE_COMPONENTS: dict[str, list[str]] = {
    "feature_encoding": ["public_feature_matrix", "categorical_encoding", "numeric_scaling"],
    "surrogate_model": ["observed_only_surrogate", "cross_validated_prediction"],
    "uncertainty_estimation": ["distance_uncertainty", "model_disagreement_proxy"],
    "acquisition_rule": ["exploration_exploitation_score", "best_region_bonus"],
    "diversity_or_density": ["coverage_bonus", "nearest_observed_distance"],
    "domain_prior": ["public_domain_feature_prior", "reaction_condition_prior"],
    "ensemble_or_rank_aggregation": ["component_rank_aggregation", "score_blending"],
    "diagnostic_guard": ["finite_score_guard", "degeneracy_guard", "tie_breaking_rule"],
}


def extract_method_primitives(
    *,
    accepted_cards: list[dict[str, Any]],
    dataset_profile: dict[str, Any],
    memory_text: str = "",
    tool_state: dict[str, Any] | None = None,
    max_primitives: int = 12,
) -> list[dict[str, Any]]:
    """Convert accepted research cards into public-safe optimizer primitives.

    The extractor intentionally uses only card summaries, method tags, dataset
    profile summaries, and public-safe state summaries. It never reads evaluator
    internals or unrevealed target values.
    """

    primitives: list[dict[str, Any]] = []
    for card in accepted_cards:
        card_id = str(card.get("card_id", "")).strip()
        if not card_id:
            continue
        explicit_primitives = card.get("method_primitives")
        if isinstance(explicit_primitives, list) and explicit_primitives:
            for item in explicit_primitives:
                if len(primitives) >= int(max_primitives):
                    break
                if not isinstance(item, dict):
                    continue
                primitive_type = str(item.get("primitive_type", "")).strip()
                if primitive_type not in ALLOWED_PRIMITIVE_TYPES:
                    continue
                primitive = {
                    "primitive_id": f"primitive_{len(primitives) + 1:03d}",
                    "type": primitive_type,
                    "idea": sanitize_text(str(item.get("implementation_hint", "")))[:600],
                    "source_card_ids": [card_id],
                    "possible_code_components": list(_CODE_COMPONENTS[primitive_type]),
                    "expected_failure_risk": sanitize_text(str(item.get("expected_failure_risk", "")))[:600],
                    "implementation_rule_for_our_hte_framework": sanitize_text(
                        str(card.get("implementation_rule_for_our_hte_framework", ""))
                    )[:600],
                    "non_transferable_parts": sanitize_text(str(card.get("non_transferable_parts", "")))[:600],
                    "why_not_just_static_ranker": sanitize_text(str(card.get("why_not_just_static_ranker", "")))[:600],
                    "safety_status": "public_safe",
                    "relevance_to_current_task": _relevance_statement(dataset_profile, primitive_type),
                }
                assert_payload_public_safe(primitive, label="method_primitive")
                primitives.append(primitive)
            if len(primitives) >= int(max_primitives):
                break
            continue
        card_text = _card_public_text(card)
        primitive_type = _infer_primitive_type(card_text)
        idea = _primitive_idea(card, primitive_type)
        primitive = {
            "primitive_id": f"primitive_{len(primitives) + 1:03d}",
            "type": primitive_type,
            "idea": idea,
            "source_card_ids": [card_id],
            "possible_code_components": list(_CODE_COMPONENTS[primitive_type]),
            "expected_failure_risk": sanitize_text(str(card.get("failure_risks", "")))[:600],
            "implementation_rule_for_our_hte_framework": sanitize_text(
                str(card.get("implementation_rule_for_our_hte_framework", ""))
            )[:600],
            "non_transferable_parts": sanitize_text(str(card.get("non_transferable_parts", "")))[:600],
            "why_not_just_static_ranker": sanitize_text(str(card.get("why_not_just_static_ranker", "")))[:600],
            "safety_status": "public_safe",
            "relevance_to_current_task": _relevance_statement(dataset_profile, primitive_type),
        }
        assert_payload_public_safe(primitive, label="method_primitive")
        primitives.append(primitive)
        if len(primitives) >= int(max_primitives):
            break

    if not primitives:
        fallback_types = ("feature_encoding", "surrogate_model", "diagnostic_guard")
        for primitive_type in fallback_types:
            primitive = {
                "primitive_id": f"primitive_{len(primitives) + 1:03d}",
                "type": primitive_type,
                "idea": _fallback_idea(dataset_profile, primitive_type),
                "source_card_ids": [],
                "possible_code_components": list(_CODE_COMPONENTS[primitive_type]),
                "safety_status": "public_safe_fallback_from_profile",
                "relevance_to_current_task": _relevance_statement(dataset_profile, primitive_type),
            }
            assert_payload_public_safe(primitive, label="method_primitive")
            primitives.append(primitive)

    state_primitive = _state_summary_primitive(
        dataset_profile=dataset_profile,
        memory_text=memory_text,
        tool_state=tool_state or {},
        existing_count=len(primitives),
    )
    if state_primitive is not None and len(primitives) < int(max_primitives):
        primitives.append(state_primitive)
    return primitives[: int(max_primitives)]


def primitive_ids(primitives: list[dict[str, Any]]) -> list[str]:
    """Return primitive IDs in prompt order."""

    return [str(item.get("primitive_id")) for item in primitives]


def _card_public_text(card: dict[str, Any]) -> str:
    summary = card.get("summary", {}) if isinstance(card.get("summary"), dict) else {}
    pieces: list[str] = [
        str(card.get("title", "")),
        " ".join(str(tag) for tag in card.get("method_tags", []) if tag is not None),
        str(summary.get("one_sentence_takeaway", "")),
        " ".join(str(item) for item in summary.get("method_guidance", [])[:4]),
        " ".join(str(item) for item in summary.get("safe_tool_design_implications", [])[:4]),
        " ".join(str(item) for item in summary.get("failure_modes_or_cautions", [])[:4]),
        str(card.get("problem_setting", "")),
        str(card.get("optimizer_or_planner_structure", "")),
        str(card.get("implementation_rule_for_our_hte_framework", "")),
        str(card.get("failure_risks", "")),
        str(card.get("why_not_just_static_ranker", "")),
    ]
    return sanitize_text(" ".join(pieces))[:1800]


def _infer_primitive_type(text: str) -> str:
    lowered = text.lower()
    scores: dict[str, int] = {}
    for primitive_type, keywords in _TYPE_KEYWORDS:
        scores[primitive_type] = sum(1 for keyword in keywords if keyword in lowered)
    best = max(scores.items(), key=lambda item: (item[1], -ALLOWED_PRIMITIVE_TYPES.index(item[0])))
    if best[1] <= 0:
        return "domain_prior"
    return best[0]


def _primitive_idea(card: dict[str, Any], primitive_type: str) -> str:
    summary = card.get("summary", {}) if isinstance(card.get("summary"), dict) else {}
    takeaway = str(summary.get("one_sentence_takeaway", "")).strip()
    if not takeaway:
        guidance = summary.get("method_guidance", [])
        takeaway = str(guidance[0]).strip() if guidance else ""
    if not takeaway:
        takeaway = f"Use public-safe {primitive_type.replace('_', ' ')} guidance from the accepted card."
    return sanitize_text(takeaway)[:600]


def _fallback_idea(dataset_profile: dict[str, Any], primitive_type: str) -> str:
    feature_count = dataset_profile.get("candidate_schema_summary", {}).get("feature_column_count", "available")
    if primitive_type == "feature_encoding":
        return f"Represent the {feature_count} public feature columns in a deterministic candidate matrix."
    if primitive_type == "surrogate_model":
        return "Use revealed observations only to estimate candidate promise under small-sample constraints."
    return "Guard the tool against nonfinite, constant, or unstable candidate scores."


def _relevance_statement(dataset_profile: dict[str, Any], primitive_type: str) -> str:
    observed_count = dataset_profile.get("observed_count", "unknown")
    remaining_count = dataset_profile.get("remaining_candidate_count", "unknown")
    task_summary = str(dataset_profile.get("task_summary", "finite-pool optimization task"))
    return (
        f"Relevant to {task_summary}; observed_count={observed_count}, "
        f"remaining_candidate_count={remaining_count}, primitive_type={primitive_type}."
    )


def _state_summary_primitive(
    *,
    dataset_profile: dict[str, Any],
    memory_text: str,
    tool_state: dict[str, Any],
    existing_count: int,
) -> dict[str, Any] | None:
    if not str(memory_text).strip() and not tool_state:
        return None
    primitive = {
        "primitive_id": f"primitive_{existing_count + 1:03d}",
        "type": "diagnostic_guard",
        "idea": "Use observed-safe run state only as a caution for robustness and small-sample overfit.",
        "source_card_ids": [],
        "possible_code_components": list(_CODE_COMPONENTS["diagnostic_guard"]),
        "safety_status": "public_safe_state_summary",
        "relevance_to_current_task": _relevance_statement(dataset_profile, "diagnostic_guard"),
    }
    assert_payload_public_safe(primitive, label="method_primitive")
    return primitive
