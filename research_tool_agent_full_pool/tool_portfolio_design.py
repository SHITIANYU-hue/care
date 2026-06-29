"""Portfolio design proposal for experimental ResearchToolAgent mode."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from research_tool_agent_full_pool.decision_artifacts import assert_payload_public_safe
from research_tool_agent_full_pool.diagnostics import sha256_text
from research_tool_agent_full_pool.tool_method_primitives import primitive_ids
from research_tool_agent_full_pool.tool_portfolio_artifacts import write_portfolio_design_artifacts


PORTFOLIO_DESIGN_SCHEMA_VERSION = "research_tool_portfolio_design_v0"
PORTFOLIO_POLICY_NAME = "ResearchToolAgentToolPortfolioDiagnostic"
DEFAULT_DESIGN_COUNT = 3
MANDATORY_TASK_STATEMENT = (
    "If my design only predicts yield without an explicit sequential selection/update policy, it is invalid."
)
REQUIRED_TASK_INTERPRETATION: dict[str, Any] = {
    "finite_pool": True,
    "observed_y_only": True,
    "unrevealed_candidate_y_hidden": True,
    "sequential_reveal": True,
    "goal_is_optimization_not_prediction": True,
    "decision_policy_required": True,
    "invalid_if_only_predicts_yield": True,
    "fixed_runner_owns": [
        "candidate pool construction",
        "target/yield column filtering",
        "observed/candidate split",
        "evaluator reveal",
        "campaign logging",
        "leakage audit",
        "budget control",
    ],
    "tool_may_control": [
        "public feature encoding",
        "surrogate or heuristic scoring",
        "uncertainty or novelty proxy",
        "selection or acquisition logic",
        "fallback behavior",
        "diagnostics",
        "private tool_state update suggestion",
    ],
    "mandatory_statement": MANDATORY_TASK_STATEMENT,
}
REQUIRED_STATIC_SELF_AUDIT_SCHEMA: dict[str, Any] = {
    "is_only_predicted_yield": False,
    "has_explicit_exploitation": True,
    "has_explicit_exploration_or_uncertainty_or_novelty": True,
    "has_finite_pool_selection_policy": True,
    "has_update_or_state_policy": True,
    "handles_small_n": True,
    "handles_mixed_numeric_categorical_features": True,
    "avoids_duplicate_or_near_duplicate_recommendations": True,
    "fake_uncertainty_risk": "low|medium|high",
    "hidden_y_leakage_self_check": "pass|fail",
    "uses_only_observed_y": True,
    "uses_only_public_candidate_features": True,
    "why_this_is_sequential_optimizer_not_static_ranker": "short string",
}


class PortfolioDesignError(ValueError):
    """Raised when portfolio design output is malformed or unsafe."""


def propose_portfolio_designs(
    *,
    client: Any,
    config: Any,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    dataset_profile: dict[str, Any],
    method_primitives: list[dict[str, Any]],
    research_source_metadata: dict[str, Any],
    output_dir: str,
    round_index: int = 1,
    design_count: int = DEFAULT_DESIGN_COUNT,
) -> dict[str, Any]:
    """Ask the LLM for structurally distinct optimizer designs and persist artifacts."""

    prompt = build_portfolio_design_prompt(
        config=config,
        observed_df=observed_df,
        candidate_df=candidate_df,
        dataset_profile=dataset_profile,
        method_primitives=method_primitives,
        research_source_metadata=research_source_metadata,
        round_index=round_index,
        design_count=design_count,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Return only one strict JSON object proposing safe offline optimizer tool designs. "
                "Do not return markdown."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    raw_text = client.create_tool(messages=messages)
    parsed = parse_portfolio_designs_json(
        raw_text,
        expected_run_id=str(config.run_id),
        expected_round_index=round_index,
        expected_design_count=design_count,
        method_primitive_ids=set(primitive_ids(method_primitives)),
    )
    request_metadata = {
        "mode": "portfolio_design",
        "run_id": str(config.run_id),
        "round_index": int(round_index),
        "design_count_requested": int(design_count),
        "candidate_rows_in_prompt": 0,
        "candidate_df_rows_available_to_tools": int(len(candidate_df)),
        "observed_rows_in_prompt": int(min(len(observed_df), 20)),
        "method_primitive_ids_included": primitive_ids(method_primitives),
        "research_source_metadata": dict(research_source_metadata),
        "task_interpretation": parsed["task_interpretation"],
        "static_self_audit": parsed["static_self_audit"],
    }
    write_portfolio_design_artifacts(
        output_dir=output_dir,
        prompt_text=prompt,
        request_metadata=request_metadata,
        raw_response=raw_text,
        parsed_designs=parsed["designs"],
        diagnostics=parsed["diagnostics"],
        task_interpretation=parsed["task_interpretation"],
        static_self_audit=parsed["static_self_audit"],
    )
    return {
        "prompt_text": prompt,
        "raw_text": raw_text,
        "task_interpretation": parsed["task_interpretation"],
        "designs": parsed["designs"],
        "diagnostics": parsed["diagnostics"],
        "static_self_audit": parsed["static_self_audit"],
        "request_metadata": {
            **request_metadata,
            "prompt_hash": sha256_text(prompt),
            "prompt_character_count": len(prompt),
        },
    }


def build_portfolio_design_prompt(
    *,
    config: Any,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    dataset_profile: dict[str, Any],
    method_primitives: list[dict[str, Any]],
    research_source_metadata: dict[str, Any],
    round_index: int,
    design_count: int = DEFAULT_DESIGN_COUNT,
) -> str:
    """Build the design prompt without candidate rows or prescribed algorithms."""

    payload = {
        "task": "research_tool_agent_tool_portfolio_design",
        "schema_version": PORTFOLIO_DESIGN_SCHEMA_VERSION,
        "policy_name": PORTFOLIO_POLICY_NAME,
        "run_id": str(config.run_id),
        "round_index": int(round_index),
        "requested_design_count": int(design_count),
        "objective": {
            "name": str(getattr(config, "target_column", "observed_y")),
            "direction": str(getattr(config, "objective_direction", "maximize")),
        },
        "dataset_profile_public_safe": _compact_profile(dataset_profile),
        "observed_history": {
            "row_count": int(len(observed_df)),
            "schema": _schema_summary(observed_df),
            "rows_in_prompt": _safe_records(observed_df.head(20)),
        },
        "candidate_pool": {
            "row_count": int(len(candidate_df)),
            "schema": _schema_summary(candidate_df),
            "full_candidate_rows_in_prompt": 0,
            "summary": _candidate_summary(candidate_df),
        },
        "research_source_metadata": dict(research_source_metadata),
        "method_primitives": method_primitives,
        "task_interpretation_required": REQUIRED_TASK_INTERPRETATION,
        "task_interpretation_instructions": [
            "This is a finite-pool sequential optimization problem.",
            "The tool may use observed y only from already revealed experiments.",
            "Remaining candidate y values are hidden and must not be inferred from artifacts.",
            "The evaluator reveals y only for selected candidates.",
            "The goal is to maximize future observed best yield under a finite experimental budget, not to produce a static yield predictor.",
            MANDATORY_TASK_STATEMENT,
        ],
        "design_rules": [
            (
                "Propose 3 structurally distinct finite-pool sequential optimizer designs. They should differ in "
                "mechanism, not just wording. You may include a Bayesian optimization-like design if you believe it "
                "is justified by the task constraints, but you are not required to. At most one design should be "
                "conventional Bayesian optimization-like. The other designs should use different public-safe "
                "optimization mechanisms such as bandit-style optimism, uncertainty-aware heuristics, cluster/local "
                "search, rank aggregation, diversity-aware exploration, domain-prior scoring, evolutionary/swarm-inspired "
                "search, or hybrid methods. These are examples, not required answers."
            ),
            "Each design must be implementable as rank_candidates(observed_df, candidate_df, memory=None, tool_state=None).",
            "Use only public candidate features and revealed observed_df observed_y.",
            "Each design must have an exploitation term.",
            "Each design must have an exploration, uncertainty, novelty, diversity, or anti-collapse term.",
            "Each design must include a small-n fallback.",
            "Each design must address mixed numeric/categorical variables.",
            "Each design must state what diagnostics it will emit.",
            "Each design must explain why it is not merely a predicted-yield ranker.",
            "Do not prescribe one required algorithm; choose designs based on the observed data and primitives.",
            "Public-safe heuristics, surrogate-style modeling, acquisition-style scoring, uncertainty or diversity bonuses, domain priors, rank aggregation across components, and bounded estimator classes permitted by the harness are allowed when justified.",
            "Do not mimic any named fixed baseline implementation.",
            "The three designs must differ in family and component structure.",
            "Each design must map every method primitive to an intended code component or mark it not_used with a rationale.",
            "No candidate design may request external files, network calls, subprocesses, dynamic imports, credentials, private evaluation state, non-public outcomes, retrospective ranking artifacts, comparator outputs, score-cache artifacts, non-public ID mappings, or answer keys.",
            "Candidate tools proposed later will be screened using observed-safe data only; do not rely on any non-public future feedback.",
        ],
        "static_self_audit_required": REQUIRED_STATIC_SELF_AUDIT_SCHEMA,
        "static_self_audit_rules": [
            "If the tool only predicts yield or produces a smooth score without explicit sequential selection logic, mark the design invalid.",
            "If uncertainty is just a constant, arbitrary noise, or a name without observed-data basis, mark fake_uncertainty_risk high.",
            "If categorical variables are ignored, mark mixed-variable handling as incomplete.",
            "If small observed n is not handled, mark handles_small_n false.",
            "If the design could repeatedly recommend near-duplicates without any diversity/anti-collapse logic, flag it.",
        ],
        "required_design_schema": {
            "design_id": "portfolio_design_001",
            "family": "short method family name",
            "mechanism_summary": "short mechanism description",
            "state_variables": ["public_state_name"],
            "selection_formula": "score = exploitation + exploration_or_anti_collapse + tie_break",
            "exploitation_term": "observed-only promise term",
            "exploration_or_anti_collapse_term": "uncertainty/novelty/diversity/anti-collapse term",
            "uncertainty_or_novelty_source": "observed-data/public-feature basis for uncertainty or novelty",
            "finite_pool_selection_rule": "score all remaining rows and rank/select rank 1",
            "small_n_fallback": "explicit fallback for sparse observed history",
            "mixed_variable_handling": "numeric and categorical handling plan",
            "diagnostics_to_emit": ["diagnostic name"],
            "state_patch_plan": "public-safe tool_state update suggestion",
            "components": ["optional component_name for compatibility"],
            "method_primitives_used": [
                {
                    "primitive_id": "primitive_001",
                    "intended_code_component": "component_name",
                    "rationale": "short public-safe reason",
                },
            ],
            "method_primitives_not_used": [
                {
                    "primitive_id": "primitive_002",
                    "reason": "short public-safe reason",
                },
            ],
            "failure_modes": ["short failure mode"],
            "why_this_is_not_a_static_ranker": "short string",
            "public_safe_boundary_statement": "uses only public runtime inputs and revealed observations",
            "static_self_audit": REQUIRED_STATIC_SELF_AUDIT_SCHEMA,
        },
        "required_json_output": {
            "schema_version": PORTFOLIO_DESIGN_SCHEMA_VERSION,
            "policy_name": PORTFOLIO_POLICY_NAME,
            "run_id": str(config.run_id),
            "round_index": int(round_index),
            "task_interpretation": REQUIRED_TASK_INTERPRETATION,
            "designs": ["exactly three design objects following required_design_schema"],
            "static_self_audit": REQUIRED_STATIC_SELF_AUDIT_SCHEMA,
            "self_reported_forbidden_info_used": False,
        },
        "strict_output_rules": [
            "Return exactly one JSON object and no markdown.",
            "No explanatory prose outside the JSON object.",
            "First character must be { and last character must be }.",
        ],
    }
    assert_payload_public_safe(payload, label="portfolio_design_prompt_payload")
    return json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=True)


def parse_portfolio_designs_json(
    raw_text: str,
    *,
    expected_run_id: str,
    expected_round_index: int,
    expected_design_count: int,
    method_primitive_ids: set[str],
) -> dict[str, Any]:
    """Parse and validate portfolio design JSON."""

    text = str(raw_text).strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise PortfolioDesignError("Portfolio design response must be exactly one JSON object.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PortfolioDesignError("Portfolio design response is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise PortfolioDesignError("Portfolio design response must decode to an object.")
    allowed_top = {
        "schema_version",
        "policy_name",
        "run_id",
        "round_index",
        "task_interpretation",
        "designs",
        "static_self_audit",
        "self_reported_forbidden_info_used",
    }
    extra = set(payload) - allowed_top
    if extra:
        raise PortfolioDesignError(f"Unsupported portfolio design fields: {sorted(extra)}")
    if payload.get("schema_version") != PORTFOLIO_DESIGN_SCHEMA_VERSION:
        raise PortfolioDesignError("Portfolio design schema_version mismatch.")
    if payload.get("policy_name") != PORTFOLIO_POLICY_NAME:
        raise PortfolioDesignError("Portfolio design policy_name mismatch.")
    if str(payload.get("run_id")) != str(expected_run_id):
        raise PortfolioDesignError("Portfolio design run_id mismatch.")
    if int(payload.get("round_index", -1)) != int(expected_round_index):
        raise PortfolioDesignError("Portfolio design round_index mismatch.")
    if payload.get("self_reported_forbidden_info_used") is not False:
        raise PortfolioDesignError("self_reported_forbidden_info_used must be false.")
    task_interpretation = _validate_task_interpretation(payload.get("task_interpretation"))
    static_self_audit = _validate_static_self_audit(payload.get("static_self_audit"), context="portfolio response")
    designs = payload.get("designs")
    if not isinstance(designs, list) or len(designs) != int(expected_design_count):
        raise PortfolioDesignError("Portfolio design response must contain exactly the requested number of designs.")
    parsed_designs = [
        _validate_design(
            design,
            index=index,
            method_primitive_ids=method_primitive_ids,
        )
        for index, design in enumerate(designs, start=1)
    ]
    _validate_bayesian_optimization_like_limit(parsed_designs)
    diagnostics = distinctness_diagnostics(parsed_designs)
    assert_payload_public_safe(
        {
            "task_interpretation": task_interpretation,
            "designs": parsed_designs,
            "diagnostics": diagnostics,
            "static_self_audit": static_self_audit,
        },
        label="portfolio_designs",
    )
    return {
        "task_interpretation": task_interpretation,
        "designs": parsed_designs,
        "diagnostics": diagnostics,
        "static_self_audit": static_self_audit,
    }


def distinctness_diagnostics(designs: list[dict[str, Any]]) -> dict[str, Any]:
    """Return warnings when design families or components are too similar."""

    warnings: list[str] = []
    normalized_families = [_normalize_family(str(design.get("family", ""))) for design in designs]
    duplicate_families = sorted({family for family in normalized_families if normalized_families.count(family) > 1})
    if duplicate_families:
        warnings.append("duplicate_or_too_similar_design_families")
    overlap_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(designs):
        left_components = {str(item).lower().strip() for item in left.get("components", [])}
        for right_index in range(left_index + 1, len(designs)):
            right = designs[right_index]
            right_components = {str(item).lower().strip() for item in right.get("components", [])}
            union = left_components | right_components
            overlap = len(left_components & right_components) / max(len(union), 1)
            if overlap >= 0.8:
                warnings.append("component_sets_too_similar")
                overlap_pairs.append(
                    {
                        "left_design_id": left.get("design_id"),
                        "right_design_id": right.get("design_id"),
                        "component_jaccard": round(overlap, 4),
                    }
                )
    return {
        "design_count": len(designs),
        "family_count": len(set(normalized_families)),
        "duplicate_families": duplicate_families,
        "component_overlap_pairs": overlap_pairs,
        "warnings": sorted(set(warnings)),
        "structurally_distinct": not warnings,
    }


def _validate_design(
    design: Any,
    *,
    index: int,
    method_primitive_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(design, dict):
        raise PortfolioDesignError("Each portfolio design must be an object.")
    required = {
        "design_id",
        "family",
        "mechanism_summary",
        "state_variables",
        "selection_formula",
        "exploitation_term",
        "exploration_or_anti_collapse_term",
        "uncertainty_or_novelty_source",
        "finite_pool_selection_rule",
        "small_n_fallback",
        "mixed_variable_handling",
        "diagnostics_to_emit",
        "state_patch_plan",
        "method_primitives_used",
        "method_primitives_not_used",
        "failure_modes",
        "why_this_is_not_a_static_ranker",
        "public_safe_boundary_statement",
        "static_self_audit",
    }
    missing = sorted(required - set(design))
    if missing:
        raise PortfolioDesignError(f"Portfolio design missing required fields: {missing}")
    design_id = str(design.get("design_id")).strip() or f"portfolio_design_{index:03d}"
    components = design.get("components")
    if isinstance(components, list) and any(str(item).strip() for item in components):
        parsed_components = [str(item).strip() for item in components if str(item).strip()]
    else:
        parsed_components = [
            _required_text(design, "exploitation_term")[:80],
            _required_text(design, "exploration_or_anti_collapse_term")[:80],
            _required_text(design, "finite_pool_selection_rule")[:80],
        ]
    primitive_usage = _validate_primitive_usage(
        used_value=design.get("method_primitives_used"),
        not_used_value=design.get("method_primitives_not_used"),
        method_primitive_ids=method_primitive_ids,
    )
    static_self_audit = _validate_static_self_audit(
        design.get("static_self_audit"),
        context=f"portfolio design {design_id}",
    )
    parsed = {
        "design_id": design_id,
        "family": str(design.get("family", "")).strip(),
        "mechanism_summary": _required_text(design, "mechanism_summary"),
        "state_variables": _required_list(design, "state_variables"),
        "selection_formula": _required_text(design, "selection_formula"),
        "exploitation_term": _required_text(design, "exploitation_term"),
        "exploration_or_anti_collapse_term": _required_text(design, "exploration_or_anti_collapse_term"),
        "uncertainty_or_novelty_source": _required_text(design, "uncertainty_or_novelty_source"),
        "finite_pool_selection_rule": _required_text(design, "finite_pool_selection_rule"),
        "small_n_fallback": _required_text(design, "small_n_fallback"),
        "mixed_variable_handling": _required_text(design, "mixed_variable_handling"),
        "diagnostics_to_emit": _required_list(design, "diagnostics_to_emit"),
        "state_patch_plan": _required_text(design, "state_patch_plan"),
        "components": parsed_components,
        "method_primitives_used": primitive_usage["used"],
        "method_primitives_not_used": primitive_usage["not_used"],
        "failure_modes": _required_list(design, "failure_modes"),
        "expected_failure_modes": _required_list(design, "failure_modes"),
        "why_this_is_not_a_static_ranker": _required_text(design, "why_this_is_not_a_static_ranker"),
        "public_safe_boundary_statement": _required_text(design, "public_safe_boundary_statement"),
        "static_self_audit": static_self_audit,
        "static_ranker_risk": bool(static_self_audit["is_only_predicted_yield"]),
    }
    if "why_suitable_for_current_observed_data" in design:
        parsed["why_suitable_for_current_observed_data"] = _required_text(
            design,
            "why_suitable_for_current_observed_data",
        )
    else:
        parsed["why_suitable_for_current_observed_data"] = parsed["mechanism_summary"]
    if "alternatives_considered" in design:
        parsed["alternatives_considered"] = _required_list(design, "alternatives_considered")
    else:
        parsed["alternatives_considered"] = []
    if not parsed["family"]:
        raise PortfolioDesignError("Portfolio design family must be non-empty.")
    return parsed


def _validate_task_interpretation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PortfolioDesignError("task_interpretation is required and must be an object.")
    parsed: dict[str, Any] = {}
    for key, required_value in REQUIRED_TASK_INTERPRETATION.items():
        if key not in value:
            raise PortfolioDesignError(f"task_interpretation missing required field: {key}")
        if isinstance(required_value, bool):
            if value.get(key) is not required_value:
                raise PortfolioDesignError(f"task_interpretation.{key} contradicts the required task.")
            parsed[key] = required_value
        elif isinstance(required_value, list):
            actual = value.get(key)
            if not isinstance(actual, list):
                raise PortfolioDesignError(f"task_interpretation.{key} must be a list.")
            missing = [item for item in required_value if item not in actual]
            if missing:
                raise PortfolioDesignError(f"task_interpretation.{key} missing required items: {missing}")
            parsed[key] = [str(item) for item in actual]
        else:
            actual = str(value.get(key, "")).strip()
            if actual != str(required_value):
                raise PortfolioDesignError(f"task_interpretation.{key} must match the required statement.")
            parsed[key] = actual
    return parsed


def _validate_static_self_audit(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PortfolioDesignError(f"{context} static_self_audit is required and must be an object.")
    required_fields = set(REQUIRED_STATIC_SELF_AUDIT_SCHEMA)
    missing = sorted(required_fields - set(value))
    if missing:
        raise PortfolioDesignError(f"{context} static_self_audit missing required fields: {missing}")
    fake_risk = str(value.get("fake_uncertainty_risk", "")).strip().lower()
    if fake_risk not in {"low", "medium", "high"}:
        raise PortfolioDesignError(f"{context} fake_uncertainty_risk must be low, medium, or high.")
    if str(value.get("hidden_y_leakage_self_check", "")).strip().lower() != "pass":
        raise PortfolioDesignError(f"{context} hidden_y_leakage_self_check must be pass.")
    if value.get("uses_only_observed_y") is not True:
        raise PortfolioDesignError(f"{context} uses_only_observed_y must be true.")
    if value.get("uses_only_public_candidate_features") is not True:
        raise PortfolioDesignError(f"{context} uses_only_public_candidate_features must be true.")
    parsed = {
        "is_only_predicted_yield": bool(value.get("is_only_predicted_yield")),
        "has_explicit_exploitation": bool(value.get("has_explicit_exploitation")),
        "has_explicit_exploration_or_uncertainty_or_novelty": bool(
            value.get("has_explicit_exploration_or_uncertainty_or_novelty")
        ),
        "has_finite_pool_selection_policy": bool(value.get("has_finite_pool_selection_policy")),
        "has_update_or_state_policy": bool(value.get("has_update_or_state_policy")),
        "handles_small_n": bool(value.get("handles_small_n")),
        "handles_mixed_numeric_categorical_features": bool(
            value.get("handles_mixed_numeric_categorical_features")
        ),
        "avoids_duplicate_or_near_duplicate_recommendations": bool(
            value.get("avoids_duplicate_or_near_duplicate_recommendations")
        ),
        "fake_uncertainty_risk": fake_risk,
        "hidden_y_leakage_self_check": "pass",
        "uses_only_observed_y": True,
        "uses_only_public_candidate_features": True,
        "why_this_is_sequential_optimizer_not_static_ranker": _required_text(
            value,
            "why_this_is_sequential_optimizer_not_static_ranker",
        ),
    }
    parsed["static_ranker_risk"] = bool(parsed["is_only_predicted_yield"])
    return parsed


def _validate_primitive_usage(
    *,
    used_value: Any,
    not_used_value: Any,
    method_primitive_ids: set[str],
) -> dict[str, list[dict[str, str]]]:
    if not isinstance(used_value, list):
        raise PortfolioDesignError("method_primitives_used must be a list.")
    if not isinstance(not_used_value, list):
        raise PortfolioDesignError("method_primitives_not_used must be a list.")
    seen: set[str] = set()
    used: list[dict[str, str]] = []
    not_used: list[dict[str, str]] = []
    for item in used_value:
        if not isinstance(item, dict):
            raise PortfolioDesignError("method_primitives_used entries must be objects.")
        primitive_id = str(item.get("primitive_id", "")).strip()
        if primitive_id not in method_primitive_ids:
            raise PortfolioDesignError("method_primitives_used contains an unknown primitive_id.")
        if primitive_id in seen:
            raise PortfolioDesignError("method_primitives_used contains duplicate primitive_id.")
        seen.add(primitive_id)
        component = str(item.get("intended_code_component", "")).strip()
        if not component:
            raise PortfolioDesignError("Used primitives require intended_code_component.")
        rationale = str(item.get("rationale", item.get("reason", ""))).strip()
        if not rationale:
            raise PortfolioDesignError("Used primitives require rationale.")
        used.append(
            {
                "primitive_id": primitive_id,
                "status": "used",
                "intended_code_component": component,
                "rationale": rationale,
            }
        )
    for item in not_used_value:
        if not isinstance(item, dict):
            raise PortfolioDesignError("method_primitives_not_used entries must be objects.")
        primitive_id = str(item.get("primitive_id", "")).strip()
        if primitive_id not in method_primitive_ids:
            raise PortfolioDesignError("method_primitives_not_used contains an unknown primitive_id.")
        if primitive_id in seen:
            raise PortfolioDesignError("method primitive appears in both used and not-used lists.")
        seen.add(primitive_id)
        reason = str(item.get("reason", item.get("rationale", ""))).strip()
        if not reason:
            raise PortfolioDesignError("Not-used primitives require reason.")
        not_used.append(
            {
                "primitive_id": primitive_id,
                "status": "not_used",
                "reason": reason,
                "rationale": reason,
            }
        )
    if seen != method_primitive_ids:
        raise PortfolioDesignError("Each design must map every method primitive exactly once.")
    return {"used": used, "not_used": not_used}


def _validate_bayesian_optimization_like_limit(designs: list[dict[str, Any]]) -> None:
    conventional_count = 0
    for design in designs:
        text = " ".join(
            str(design.get(key, ""))
            for key in ("family", "mechanism_summary", "selection_formula", "finite_pool_selection_rule")
        ).lower()
        if "bayesian optimization" in text or "bo-like" in text or "bo_like" in text:
            conventional_count += 1
    if conventional_count > 1:
        raise PortfolioDesignError("At most one design may be conventional Bayesian optimization-like.")


def _required_text(design: dict[str, Any], key: str) -> str:
    value = str(design.get(key, "")).strip()
    if not value:
        raise PortfolioDesignError(f"Portfolio design {key} must be non-empty.")
    return value


def _required_list(design: dict[str, Any], key: str) -> list[str]:
    value = design.get(key)
    if not isinstance(value, list):
        raise PortfolioDesignError(f"Portfolio design {key} must be a list.")
    parsed = [str(item).strip() for item in value if str(item).strip()]
    if not parsed:
        raise PortfolioDesignError(f"Portfolio design {key} must be a non-empty list.")
    return parsed


def _normalize_family(value: str) -> str:
    tokens = [token for token in value.lower().replace("-", " ").replace("_", " ").split() if token]
    return " ".join(tokens[:6])


def _compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "profile_version",
        "dataset_label",
        "task_summary",
        "objective",
        "observed_count",
        "remaining_candidate_count",
        "feature_columns",
        "candidate_schema_summary",
        "numeric_column_summaries",
        "categorical_column_summaries",
        "observed_y_summary",
        "non_public_information_policy",
    )
    compact = {key: profile.get(key) for key in keep if key in profile}
    if "feature_columns" in compact:
        compact["feature_columns"] = list(compact["feature_columns"] or [])[:40]
    if "numeric_column_summaries" in compact:
        compact["numeric_column_summaries"] = list(compact["numeric_column_summaries"] or [])[:20]
    if "categorical_column_summaries" in compact:
        compact["categorical_column_summaries"] = list(compact["categorical_column_summaries"] or [])[:20]
    return compact


def _schema_summary(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [{"column": str(column), "dtype": str(dtype)} for column, dtype in frame.dtypes.items()]


def _candidate_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
        "numeric_feature_count": int(
            sum(
                1
                for column in frame.columns
                if str(column) != "candidate_id" and pd.api.types.is_numeric_dtype(frame[column])
            )
        ),
        "non_numeric_feature_count": int(
            sum(
                1
                for column in frame.columns
                if str(column) != "candidate_id" and not pd.api.types.is_numeric_dtype(frame[column])
            )
        ),
    }


def _safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(row) for row in frame.to_dict(orient="records")]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value
