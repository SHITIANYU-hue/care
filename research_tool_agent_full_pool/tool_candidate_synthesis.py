"""Candidate optimizer tool synthesis for portfolio mode."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from research_tool_agent_full_pool.decision_artifacts import assert_payload_public_safe
from research_tool_agent_full_pool.diagnostics import sha256_text
from research_tool_agent_full_pool.tool_contract import ALLOWED_IMPORTS, REQUIRED_ENTRYPOINT
from research_tool_agent_full_pool.tool_portfolio_artifacts import (
    write_candidate_manifest,
    write_candidate_tool_artifacts,
    write_prompt_pipeline_audit,
)
from research_tool_agent_full_pool.tool_portfolio_design import REQUIRED_STATIC_SELF_AUDIT_SCHEMA


CANDIDATE_TOOL_SCHEMA_VERSION = "research_tool_portfolio_candidate_tool_v0"
CANDIDATE_TOOL_POLICY_NAME = "ResearchToolAgentToolPortfolioDiagnostic"


class CandidateSynthesisError(ValueError):
    """Raised when a candidate tool response is malformed or unsafe."""


def synthesize_candidate_tools(
    *,
    client: Any,
    config: Any,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    dataset_profile: dict[str, Any],
    method_primitives: list[dict[str, Any]],
    designs: list[dict[str, Any]],
    output_dir: str,
    round_index: int = 1,
) -> list[dict[str, Any]]:
    """Synthesize one candidate rank_candidates tool per portfolio design."""

    candidates: list[dict[str, Any]] = []
    for index, design in enumerate(designs, start=1):
        tool_id = f"candidate_tool_{index:03d}"
        prompt = build_candidate_tool_prompt(
            config=config,
            observed_df=observed_df,
            candidate_df=candidate_df,
            dataset_profile=dataset_profile,
            method_primitives=method_primitives,
            design=design,
            tool_id=tool_id,
            round_index=round_index,
        )
        raw_text = client.create_tool(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only one strict JSON object containing safe Python source for one "
                        "offline candidate-scoring tool. Do not return markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )
        payload = parse_candidate_tool_json(
            raw_text,
            expected_run_id=str(config.run_id),
            expected_round_index=round_index,
            expected_tool_id=tool_id,
            expected_design_id=str(design["design_id"]),
        )
        source = str(payload["generated_tool"]["code"])
        artifact = write_candidate_tool_artifacts(
            output_dir=output_dir,
            tool_id=tool_id,
            design_id=str(design["design_id"]),
            prompt_text=prompt,
            request_metadata={
                "mode": "portfolio_candidate_tool_synthesis",
                "run_id": str(config.run_id),
                "round_index": int(round_index),
                "tool_id": tool_id,
                "design_id": str(design["design_id"]),
                "design_family": str(design.get("family", "")),
                "candidate_rows_in_prompt": 0,
                "candidate_df_rows_available_to_tool": int(len(candidate_df)),
                "observed_rows_in_prompt": int(min(len(observed_df), 20)),
                "method_primitive_ids_included": [str(item.get("primitive_id")) for item in method_primitives],
                "design_state_variables": list(design.get("state_variables", [])),
                "design_exploitation_term": str(design.get("exploitation_term", "")),
                "design_exploration_or_anti_collapse_term": str(
                    design.get("exploration_or_anti_collapse_term", "")
                ),
                "design_small_n_fallback": str(design.get("small_n_fallback", "")),
                "design_mixed_variable_handling": str(design.get("mixed_variable_handling", "")),
                "why_this_is_not_a_static_ranker": str(design.get("why_this_is_not_a_static_ranker", "")),
            },
            raw_response=raw_text,
            source=source,
        )
        static_self_audit = dict(payload["static_self_audit"])
        candidate = {
            "tool_id": tool_id,
            "design_id": str(design["design_id"]),
            "tool_family": str(design.get("family", "")),
            "source": source,
            "source_path": artifact["source_path"],
            "prompt_path": artifact["prompt_path"],
            "raw_response_path": artifact["raw_response_path"],
            "prompt_hash": artifact["prompt_hash"],
            "source_hash": artifact["source_hash"],
            "allowed_imports": list(payload["generated_tool"].get("allowed_imports", [])),
            "declared_tool_name": str(payload["generated_tool"].get("tool_name", REQUIRED_ENTRYPOINT)),
            "design": design,
            "payload": payload,
            "static_self_audit": static_self_audit,
            "static_ranker_risk": bool(static_self_audit.get("is_only_predicted_yield")),
        }
        candidates.append(candidate)

    manifest = {
        "schema_version": "research_tool_portfolio_candidate_manifest_v0",
        "run_id": str(config.run_id),
        "round_index": int(round_index),
        "candidate_tool_count": len(candidates),
        "candidates": [
            {
                "tool_id": item["tool_id"],
                "design_id": item["design_id"],
                "tool_family": item["tool_family"],
                "source_file": item["source_path"],
                "prompt_file": item["prompt_path"],
                "raw_response_file": item["raw_response_path"],
                "prompt_hash": item["prompt_hash"],
                "source_hash": item["source_hash"],
                "allowed_imports": item["allowed_imports"],
                "state_variables": item["design"].get("state_variables", []),
                "exploitation_term": item["design"].get("exploitation_term"),
                "exploration_or_anti_collapse_term": item["design"].get("exploration_or_anti_collapse_term"),
                "small_n_fallback": item["design"].get("small_n_fallback"),
                "mixed_variable_handling": item["design"].get("mixed_variable_handling"),
                "static_self_audit": item.get("static_self_audit", {}),
                "why_this_is_not_a_static_ranker": item["design"].get("why_this_is_not_a_static_ranker"),
            }
            for item in candidates
        ],
    }
    write_candidate_manifest(output_dir, manifest)
    write_prompt_pipeline_audit(
        output_dir,
        {
            "candidate_tools": [
                {
                    "tool_id": item["tool_id"],
                    "design_id": item["design_id"],
                    "state_variables": item["design"].get("state_variables", []),
                    "exploitation_term": item["design"].get("exploitation_term"),
                    "exploration_or_anti_collapse_term": item["design"].get("exploration_or_anti_collapse_term"),
                    "small_n_fallback": item["design"].get("small_n_fallback"),
                    "mixed_variable_handling": item["design"].get("mixed_variable_handling"),
                    "static_self_audit": item.get("static_self_audit", {}),
                    "why_this_is_not_a_static_ranker": item["design"].get("why_this_is_not_a_static_ranker"),
                }
                for item in candidates
            ],
        },
    )
    return candidates


def build_candidate_tool_prompt(
    *,
    config: Any,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    dataset_profile: dict[str, Any],
    method_primitives: list[dict[str, Any]],
    design: dict[str, Any],
    tool_id: str,
    round_index: int,
) -> str:
    """Build one candidate-tool prompt from a preselected design."""

    payload = {
        "task": "research_tool_agent_candidate_tool_synthesis",
        "schema_version": CANDIDATE_TOOL_SCHEMA_VERSION,
        "policy_name": CANDIDATE_TOOL_POLICY_NAME,
        "run_id": str(config.run_id),
        "round_index": int(round_index),
        "tool_id": str(tool_id),
        "design": design,
        "objective": {
            "name": str(getattr(config, "target_column", "observed_y")),
            "direction": str(getattr(config, "objective_direction", "maximize")),
        },
        "dataset_profile_public_safe": _compact_profile(dataset_profile),
        "method_primitives": method_primitives,
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
        "implementation_contract": [
            "Implement exactly: def rank_candidates(observed_df, candidate_df, memory=None, tool_state=None):",
            "Return exactly one finite score for every row in candidate_df.",
            "This is finite-pool sequential optimization: the runner selects rank 1, reveals y only for that selected candidate, then calls future tools with updated observed_df.",
            "If the tool only predicts yield or produces a smooth score without explicit sequential selection logic, mark static_self_audit.is_only_predicted_yield true.",
            "Return a dict with ranked_candidates, tool_state, and tool_diagnostics.",
            "Each ranked candidate must contain candidate_id, rank, score, reason_code, and evidence_refs.",
            "evidence_refs must be [] unless referencing actual observation_id values from observed_df.",
            "Never put component names, method names, feature names, candidate IDs, labels, counts, scores, diagnostics, source names, or file names in evidence_refs.",
            "If unsure about evidence_refs, use evidence_refs: [].",
            "Use only observed_df, candidate_df, memory, and tool_state supplied at runtime.",
            "Use observed_df observed_y only for rows already revealed in observed_df.",
            "Candidate tools are offline screenable code; they must not request or perform an evaluator reveal.",
            "No external files, network calls, subprocesses, environment access, eval, exec, dynamic imports, credentials, private evaluation state, non-public target values, comparator outputs, cached score artifacts, non-public ID mappings, or answer keys.",
            "Allowed imports are limited to numpy, pandas, math, statistics, random, collections, itertools, and bounded sklearn modules already allowed by the harness.",
            "If using sklearn, import only symbols from the allowed modules and keep the implementation deterministic.",
            "Do not use unavailable builtins such as ord. For deterministic tie-breaking, sort by candidate_id string or stable numeric score columns.",
            "Do not rely on unlisted builtins. Keep deterministic helpers simple and sandbox-safe.",
            "Return exactly one row per candidate. Every row must include candidate_id, finite score, deterministic rank, reason_code, and evidence_refs.",
            "Scores must be finite numbers: no NaN or inf. Candidate coverage must be complete.",
            "Align observed and candidate public feature columns before vector operations.",
            "Handle missing columns and nonfinite values. Do not assume every optional public feature exists.",
            "Do not use DataFrame attrs.",
            "In tool_diagnostics, summarize the exploitation term, exploration/uncertainty/novelty term, small-n fallback, and expected failure modes without adding unsafe fields.",
            "Emit diagnostics that make exploitation, exploration or anti-collapse logic, small-n fallback, and mixed-variable handling auditable.",
            "Avoid matrix multiplication operators and fragile shape assumptions; use row-wise reductions and finite fallbacks.",
            "Do not claim to use a primitive unless a corresponding code component is present.",
        ],
        "static_self_audit_required": REQUIRED_STATIC_SELF_AUDIT_SCHEMA,
        "static_self_audit_rules": [
            "If the tool only predicts yield or produces a smooth score without explicit sequential selection logic, mark the design invalid.",
            "If uncertainty is just a constant, arbitrary noise, or a name without observed-data basis, mark fake_uncertainty_risk high.",
            "If categorical variables are ignored, mark mixed-variable handling as incomplete.",
            "If small observed n is not handled, mark handles_small_n false.",
            "If the design could repeatedly recommend near-duplicates without any diversity/anti-collapse logic, flag it.",
        ],
        "required_json_output": {
            "schema_version": CANDIDATE_TOOL_SCHEMA_VERSION,
            "policy_name": CANDIDATE_TOOL_POLICY_NAME,
            "run_id": str(config.run_id),
            "round_index": int(round_index),
            "tool_id": str(tool_id),
            "design_id": str(design["design_id"]),
            "generated_tool": {
                "tool_name": "safe_short_name",
                "entrypoint": REQUIRED_ENTRYPOINT,
                "allowed_imports": ["numpy", "pandas"],
                "code": "Python source code as a JSON string",
            },
            "static_self_audit": REQUIRED_STATIC_SELF_AUDIT_SCHEMA,
            "self_reported_forbidden_info_used": False,
        },
        "strict_output_rules": [
            "Return exactly one JSON object and no markdown.",
            "No explanatory prose outside the JSON object.",
            "First character must be { and last character must be }.",
            "The generated_tool.code string must decode to Python source with real newline characters.",
        ],
    }
    assert_payload_public_safe(payload, label="candidate_tool_prompt_payload")
    return json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=True)


def parse_candidate_tool_json(
    raw_text: str,
    *,
    expected_run_id: str,
    expected_round_index: int,
    expected_tool_id: str,
    expected_design_id: str,
) -> dict[str, Any]:
    """Parse one candidate-tool synthesis response."""

    text = str(raw_text).strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise CandidateSynthesisError("Candidate tool response must be exactly one JSON object.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CandidateSynthesisError("Candidate tool response is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise CandidateSynthesisError("Candidate tool response must decode to an object.")
    allowed_top = {
        "schema_version",
        "policy_name",
        "run_id",
        "round_index",
        "tool_id",
        "design_id",
        "generated_tool",
        "static_self_audit",
        "self_reported_forbidden_info_used",
    }
    extra = set(payload) - allowed_top
    if extra:
        raise CandidateSynthesisError(f"Unsupported candidate tool fields: {sorted(extra)}")
    if payload.get("schema_version") != CANDIDATE_TOOL_SCHEMA_VERSION:
        raise CandidateSynthesisError("Candidate tool schema_version mismatch.")
    if payload.get("policy_name") != CANDIDATE_TOOL_POLICY_NAME:
        raise CandidateSynthesisError("Candidate tool policy_name mismatch.")
    if str(payload.get("run_id")) != str(expected_run_id):
        raise CandidateSynthesisError("Candidate tool run_id mismatch.")
    if int(payload.get("round_index", -1)) != int(expected_round_index):
        raise CandidateSynthesisError("Candidate tool round_index mismatch.")
    if str(payload.get("tool_id")) != str(expected_tool_id):
        raise CandidateSynthesisError("Candidate tool_id mismatch.")
    if str(payload.get("design_id")) != str(expected_design_id):
        raise CandidateSynthesisError("Candidate design_id mismatch.")
    if payload.get("self_reported_forbidden_info_used") is not False:
        raise CandidateSynthesisError("self_reported_forbidden_info_used must be false.")
    payload["static_self_audit"] = _validate_static_self_audit(payload.get("static_self_audit"))
    tool = payload.get("generated_tool")
    if not isinstance(tool, dict):
        raise CandidateSynthesisError("generated_tool must be an object.")
    if set(tool) != {"tool_name", "entrypoint", "allowed_imports", "code"}:
        raise CandidateSynthesisError("generated_tool must contain exactly tool_name, entrypoint, allowed_imports, code.")
    if tool.get("entrypoint") != REQUIRED_ENTRYPOINT:
        raise CandidateSynthesisError("generated_tool.entrypoint must be rank_candidates.")
    source = tool.get("code")
    if not isinstance(source, str) or f"def {REQUIRED_ENTRYPOINT}" not in source:
        raise CandidateSynthesisError("generated_tool.code must define rank_candidates.")
    imports = tool.get("allowed_imports")
    if not isinstance(imports, list):
        raise CandidateSynthesisError("generated_tool.allowed_imports must be a list.")
    if not set(str(item) for item in imports).issubset(set(ALLOWED_IMPORTS)):
        raise CandidateSynthesisError("generated_tool.allowed_imports contains disallowed imports.")
    assert_payload_public_safe(payload, label="candidate_tool_payload")
    return payload


def _validate_static_self_audit(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CandidateSynthesisError("static_self_audit is required and must be an object.")
    missing = sorted(set(REQUIRED_STATIC_SELF_AUDIT_SCHEMA) - set(value))
    if missing:
        raise CandidateSynthesisError(f"static_self_audit missing required fields: {missing}")
    fake_risk = str(value.get("fake_uncertainty_risk", "")).strip().lower()
    if fake_risk not in {"low", "medium", "high"}:
        raise CandidateSynthesisError("fake_uncertainty_risk must be low, medium, or high.")
    if str(value.get("hidden_y_leakage_self_check", "")).strip().lower() != "pass":
        raise CandidateSynthesisError("hidden_y_leakage_self_check must be pass.")
    if value.get("uses_only_observed_y") is not True:
        raise CandidateSynthesisError("uses_only_observed_y must be true.")
    if value.get("uses_only_public_candidate_features") is not True:
        raise CandidateSynthesisError("uses_only_public_candidate_features must be true.")
    explanation = str(value.get("why_this_is_sequential_optimizer_not_static_ranker", "")).strip()
    if not explanation:
        raise CandidateSynthesisError("why_this_is_sequential_optimizer_not_static_ranker must be non-empty.")
    return {
        "is_only_predicted_yield": bool(value.get("is_only_predicted_yield")),
        "has_explicit_exploitation": bool(value.get("has_explicit_exploitation")),
        "has_explicit_exploration_or_uncertainty_or_novelty": bool(
            value.get("has_explicit_exploration_or_uncertainty_or_novelty")
        ),
        "has_finite_pool_selection_policy": bool(value.get("has_finite_pool_selection_policy")),
        "has_update_or_state_policy": bool(value.get("has_update_or_state_policy")),
        "handles_small_n": bool(value.get("handles_small_n")),
        "handles_mixed_numeric_categorical_features": bool(value.get("handles_mixed_numeric_categorical_features")),
        "avoids_duplicate_or_near_duplicate_recommendations": bool(
            value.get("avoids_duplicate_or_near_duplicate_recommendations")
        ),
        "fake_uncertainty_risk": fake_risk,
        "hidden_y_leakage_self_check": "pass",
        "uses_only_observed_y": True,
        "uses_only_public_candidate_features": True,
        "why_this_is_sequential_optimizer_not_static_ranker": explanation,
        "static_ranker_risk": bool(value.get("is_only_predicted_yield")),
    }


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
