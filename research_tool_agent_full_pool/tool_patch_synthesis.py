"""Feedback-driven generated-tool patch synthesis."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from research_tool_agent_full_pool.decision_artifacts import assert_payload_public_safe
from research_tool_agent_full_pool.diagnostics import sha256_text
from research_tool_agent_full_pool.tool_contract import (
    ALLOWED_IMPORTS,
    FORBIDDEN_TERMS,
    REQUIRED_ENTRYPOINT,
)
from research_tool_agent_full_pool.tool_replacement_verifier import verify_tool_replacement


TOOL_PATCH_SCHEMA_VERSION = "research_tool_patch_v1"
TOOL_PATCH_POLICY_NAME = "ResearchToolPatchLifecycle"


class ToolPatchParseError(ValueError):
    """Raised when a patch synthesis response is malformed or unsafe."""


@dataclass(frozen=True)
class ToolPatchSynthesisResult:
    raw_text: str
    payload: dict[str, Any]
    patched_tool_source: str
    parser_report: dict[str, Any]
    request_summary: dict[str, Any]
    prompt_text: str


def build_tool_patch_prompt(
    *,
    config: Any,
    old_tool_source: str,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    memory_text: str,
    strategy_state: dict[str, Any],
    tool_state: dict[str, Any],
    feedback_reports: list[Any],
    patch_decision: Any,
    research_context: dict[str, Any] | None = None,
    patch_research_context: dict[str, Any] | None = None,
    round_index: int,
    parser_error: str | None = None,
) -> str:
    """Build the patch prompt from observed/public-safe inputs only."""

    prompt_payload: dict[str, Any] = {
        "task": "feedback_driven_generated_tool_patch",
        "schema_version": TOOL_PATCH_SCHEMA_VERSION,
        "policy_name": TOOL_PATCH_POLICY_NAME,
        "run_id": str(getattr(config, "run_id", "")),
        "round_index": int(round_index),
        "dataset": str(getattr(config, "dataset_name", "public optimization dataset")),
        "objective": {
            "target_column": str(getattr(config, "target_column", "observed_y")),
            "direction": str(getattr(config, "objective_direction", "maximize")),
            "revealed_observations_only": True,
        },
        "old_generated_tool_source": str(old_tool_source),
        "tool_contract": {
            "entrypoint": REQUIRED_ENTRYPOINT,
            "required_signature": "def rank_candidates(observed_df, candidate_df, memory=None, tool_state=None):",
            "runtime_inputs": [
                "observed_df with legitimately revealed observed_y values",
                "complete remaining public candidate_df at runtime",
                "public-safe memory text",
                "public-safe tool_state mapping",
            ],
            "required_return": {
                "ranked_candidates": "one dict per remaining candidate with candidate_id, rank, score, reason_code, evidence_refs",
                "tool_state": "public-safe mapping",
                "tool_diagnostics": "public-safe mapping",
            },
        },
        "observed_df_schema": _schema_summary(observed_df),
        "observed_df_revealed_sample": observed_df.head(20).to_dict(orient="records"),
        "candidate_df_public_schema": _schema_summary(candidate_df),
        "candidate_df_public_summary": _candidate_summary(candidate_df),
        "memory": str(memory_text)[:3000],
        "strategy_state_summary": _compact_state(strategy_state),
        "tool_state_summary": _compact_state(tool_state),
        "recent_tool_feedback_reports": [_as_public_dict(report) for report in feedback_reports[-5:]],
        "patch_decision": _as_public_dict(patch_decision),
        "frozen_research_context": _compact_research_context(research_context or {}),
        "patch_time_frozen_research_context": _compact_research_context(patch_research_context or {}),
        "instructions": [
            "Inspect the old tool and observed-safe feedback, then revise the tool if the feedback supports it.",
            "Any public-safe optimization strategy is allowed, including hand-written heuristics, surrogate models, acquisition-style scoring, uncertainty or diversity bonuses, ensembles, domain priors, BO-like logic, and bounded sklearn estimators computed only from prompt/runtime public inputs.",
            "Bounded sklearn imports are allowed only for Ridge, RandomForestRegressor, KNeighborsRegressor, StandardScaler, SimpleImputer, and Pipeline from the allowlisted sklearn submodules.",
            "The revised tool must preserve the required rank_candidates function and harness interface.",
            "The patched_tool_source JSON string must decode to normal Python source with real newline characters; do not double-escape code newlines as literal backslash-n text.",
            "Do not use external files, network calls, environment access, subprocesses, dynamic imports, eval, exec, answer-key labels, non-public target values, retrospective rank artifacts, repository reference optimizer implementation/artifacts, evaluation internals, non-public ID mappings, score-cache artifacts, or secrets.",
            "Do not use dtype=object or the bare built-in name object; the sandbox does not expose that builtin.",
            "Do not use @, .dot(...), np.matmul, np.linalg, pinv, or matrix multiplication; use elementwise multiply plus sum(axis=1) for weighted public-safe estimates.",
            "When computing pairwise distances or weighted local means, keep candidate-by-observation arrays aligned and reduce with sum/mean over axis=1; never multiply two 2D matrices together.",
            "The patched source must be deterministic on identical inputs and stable to candidate_df row order.",
            "Diagnostics, reasons, evidence_refs, and tool_state must remain public-safe.",
        ],
        "response_schema": {
            "schema_version": TOOL_PATCH_SCHEMA_VERSION,
            "policy_name": TOOL_PATCH_POLICY_NAME,
            "run_id": str(getattr(config, "run_id", "")),
            "round_index": int(round_index),
            "patched_tool_source": "Python source string defining rank_candidates",
            "patch_rationale": "short public-safe rationale",
            "changed_assumptions": ["public-safe assumption change"],
            "expected_failure_modes": ["public-safe failure mode"],
            "design_rationale": {
                "chosen_strategy_family": "short public-safe method family",
                "alternatives_considered": ["short public-safe alternative"],
                "reason_for_choice": "short public-safe reason",
                "how_observed_y_is_used": "use only observed_df observed_y from revealed rows",
                "how_uncertainty_or_exploration_is_handled": "short string",
                "how_research_cards_are_used": "short string",
                "expected_failure_modes": ["short public-safe failure mode"],
                "public_safe_boundary_statement": "scores use only prompt/runtime public inputs",
            },
            "public_safe_boundary_statement": "short statement that all scores are computed from observed/public inputs and no reference optimizer or non-public artifacts are used",
            "required_imports": [
                "numpy",
                "pandas",
                "sklearn.linear_model",
                "sklearn.ensemble",
                "sklearn.neighbors",
                "sklearn.preprocessing",
                "sklearn.impute",
                "sklearn.pipeline",
            ],
            "tool_state_update_suggestion": {},
            "self_reported_forbidden_info_used": False,
        },
        "repair_context": {
            "parser_error": str(parser_error)[:600],
        }
        if parser_error
        else {},
    }
    assert_payload_public_safe(prompt_payload, label="tool_patch_prompt")
    return json.dumps(prompt_payload, indent=2, sort_keys=True, ensure_ascii=True)


def parse_tool_patch_json(
    raw_text: str,
    *,
    expected_run_id: str,
    expected_round_index: int,
) -> dict[str, Any]:
    """Parse and validate one strict patch JSON response."""

    text = str(raw_text).strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise ToolPatchParseError("Patch response must be exactly one JSON object.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolPatchParseError("Patch response is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ToolPatchParseError("Patch response must decode to an object.")
    allowed_top = {
        "schema_version",
        "policy_name",
        "run_id",
        "round_index",
        "patched_tool_source",
        "patch_rationale",
        "changed_assumptions",
        "expected_failure_modes",
        "public_safe_boundary_statement",
        "reference_bo_separation_statement",
        "why_this_is_not_direct_bo",
        "design_rationale",
        "required_imports",
        "tool_state_update_suggestion",
        "self_reported_forbidden_info_used",
        "generated_tool",
        "tool_design",
    }
    extra = set(payload) - allowed_top
    if extra:
        raise ToolPatchParseError(f"Patch response contains unsupported top-level fields: {sorted(extra)}")
    _reject_forbidden_payload_terms(payload)
    assert_payload_public_safe(payload, label="tool_patch_response")
    if payload.get("schema_version") != TOOL_PATCH_SCHEMA_VERSION:
        raise ToolPatchParseError("schema_version mismatch.")
    if payload.get("policy_name") != TOOL_PATCH_POLICY_NAME:
        raise ToolPatchParseError("policy_name mismatch.")
    if payload.get("run_id") != expected_run_id:
        raise ToolPatchParseError("run_id mismatch.")
    if int(payload.get("round_index", -999)) != int(expected_round_index):
        raise ToolPatchParseError("round_index mismatch.")
    if payload.get("self_reported_forbidden_info_used") is not False:
        raise ToolPatchParseError("self_reported_forbidden_info_used must be false.")

    patched_source = payload.get("patched_tool_source")
    if not isinstance(patched_source, str) or not patched_source.strip():
        generated_tool = payload.get("generated_tool")
        if isinstance(generated_tool, dict):
            patched_source = generated_tool.get("code")
            if isinstance(patched_source, str) and patched_source.strip():
                payload["patched_tool_source"] = patched_source
                payload.setdefault("required_imports", generated_tool.get("allowed_imports", []))
        if not isinstance(patched_source, str) or not patched_source.strip():
            raise ToolPatchParseError("patched_tool_source must be a non-empty string.")
    if f"def {REQUIRED_ENTRYPOINT}" not in patched_source:
        raise ToolPatchParseError("patched_tool_source must define rank_candidates.")

    if not isinstance(payload.get("patch_rationale"), str) or not str(payload.get("patch_rationale")).strip():
        raise ToolPatchParseError("patch_rationale must be a non-empty string.")
    if not isinstance(payload.get("changed_assumptions"), list):
        raise ToolPatchParseError("changed_assumptions must be a list.")
    if not isinstance(payload.get("expected_failure_modes"), list):
        raise ToolPatchParseError("expected_failure_modes must be a list.")
    boundary = (
        payload.get("public_safe_boundary_statement")
        or payload.get("reference_bo_separation_statement")
        or payload.get("why_this_is_not_direct_bo")
    )
    if not isinstance(boundary, str) or not boundary.strip():
        raise ToolPatchParseError("public_safe_boundary_statement is required.")
    payload["public_safe_boundary_statement"] = boundary
    required_imports = payload.get("required_imports", [])
    if not isinstance(required_imports, list):
        raise ToolPatchParseError("required_imports must be a list.")
    declared = {str(item) for item in required_imports}
    if declared and not declared.issubset(set(ALLOWED_IMPORTS)):
        raise ToolPatchParseError("required_imports contains disallowed imports.")
    suggestion = payload.get("tool_state_update_suggestion", {})
    if suggestion is None:
        payload["tool_state_update_suggestion"] = {}
    elif not isinstance(suggestion, dict):
        raise ToolPatchParseError("tool_state_update_suggestion must be an object when present.")
    return payload


def synthesize_tool_patch(
    *,
    client: Any,
    config: Any,
    old_tool_source: str,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    memory_text: str,
    strategy_state: dict[str, Any],
    tool_state: dict[str, Any],
    feedback_reports: list[Any],
    patch_decision: Any,
    research_context: dict[str, Any] | None = None,
    patch_research_context: dict[str, Any] | None = None,
    round_index: int,
    parser_error: str | None = None,
) -> ToolPatchSynthesisResult:
    """Call the configured LLM/client and parse a patched tool response."""

    if client is None or not hasattr(client, "create_tool"):
        raise ValueError("Patch synthesis requires a client with create_tool(messages=...).")
    prompt = build_tool_patch_prompt(
        config=config,
        old_tool_source=old_tool_source,
        observed_df=observed_df,
        candidate_df=candidate_df,
        memory_text=memory_text,
        strategy_state=strategy_state,
        tool_state=tool_state,
        feedback_reports=feedback_reports,
        patch_decision=patch_decision,
        research_context=research_context,
        patch_research_context=patch_research_context,
        round_index=round_index,
        parser_error=parser_error,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Return exactly one JSON object containing a public-safe patched optimizer tool. "
                "Do not return markdown."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    raw_text = client.create_tool(messages=messages)
    payload = parse_tool_patch_json(
        raw_text,
        expected_run_id=str(getattr(config, "run_id", "")),
        expected_round_index=round_index,
    )
    return ToolPatchSynthesisResult(
        raw_text=str(raw_text),
        payload=payload,
        patched_tool_source=str(payload["patched_tool_source"]),
        parser_report={
            "passed": True,
            "schema_version": payload["schema_version"],
            "policy_name": payload["policy_name"],
            "round_index": int(round_index),
        },
        request_summary={
            "run_id": str(getattr(config, "run_id", "")),
            "round_index": int(round_index),
            "prompt_hash": sha256_text(prompt),
            "prompt_character_count": len(prompt),
            "observed_rows_in_prompt": min(len(observed_df), 20),
            "candidate_rows_in_prompt": 0,
            "candidate_df_rows_available_to_tool": len(candidate_df),
            "feedback_report_count": len(feedback_reports),
            "patch_decision": str(_as_public_dict(patch_decision).get("decision", "")),
            "patch_research_context_included": bool(patch_research_context),
        },
        prompt_text=prompt,
    )


def patch_tool_after_reveal(
    *,
    state: Any,
    config: Any,
    client: Any,
    old_tool_source: str,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    memory_text: str,
    strategy_state: dict[str, Any],
    tool_state: dict[str, Any],
    feedback_report: Any,
    patch_decision: Any,
    research_context: dict[str, Any] | None = None,
    patch_research_context: dict[str, Any] | None = None,
    output_dir: str | Path,
    round_index: int,
) -> dict[str, Any]:
    """Synthesize, verify, and conditionally install a patched tool."""

    decision_text = str(_as_public_dict(patch_decision).get("decision", "reuse"))
    if str(getattr(config, "patch_mode", "decision_only")) != "enabled" or decision_text == "reuse":
        return {
            "attempted": False,
            "synthesis_called": False,
            "verifier_called": False,
            "replacement_performed": False,
            "reason": "patch_mode_not_enabled_or_reuse_decision",
        }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    current_version = int(getattr(state, "active_tool_version", 1) or 1)
    candidate_version = current_version + 1
    patch_dir = output_path / "tool_patches" / f"round_{int(round_index):03d}_v{candidate_version:03d}"
    patch_dir.mkdir(parents=True, exist_ok=True)

    old_path = output_path / f"old_tool_v{current_version:03d}.py"
    old_path.write_text(str(old_tool_source), encoding="utf-8")
    (patch_dir / old_path.name).write_text(str(old_tool_source), encoding="utf-8")

    acceptance_record: dict[str, Any] = {
        "schema_version": "batch3.patch_acceptance_record.v1",
        "round_index": int(round_index),
        "patch_decision": decision_text,
        "attempted": True,
        "synthesis_called": False,
        "verifier_called": False,
        "replacement_performed": False,
        "accepted": False,
        "old_tool_hash": sha256_text(old_tool_source or ""),
        "candidate_tool_hash": None,
        "active_tool_version_before": current_version,
        "active_tool_version_after": current_version,
        "reason": "",
    }
    try:
        synthesis = synthesize_tool_patch(
            client=client,
            config=config,
            old_tool_source=old_tool_source,
            observed_df=observed_df,
            candidate_df=candidate_df,
            memory_text=memory_text,
            strategy_state=strategy_state,
            tool_state=tool_state,
            feedback_reports=[feedback_report],
            patch_decision=patch_decision,
            research_context=research_context,
            patch_research_context=patch_research_context,
            round_index=round_index,
        )
        acceptance_record["synthesis_called"] = True
        prompt_payload = {
            "request_summary": synthesis.request_summary,
            "prompt_text": synthesis.prompt_text,
        }
        response_payload = {
            "parser_report": synthesis.parser_report,
            "response_payload": synthesis.payload,
        }
        _write_public_json(output_path / "patch_prompt.json", prompt_payload, "patch_prompt")
        _write_public_json(patch_dir / "patch_prompt.json", prompt_payload, "patch_prompt")
        _write_public_json(output_path / "patch_response.json", response_payload, "patch_response")
        _write_public_json(patch_dir / "patch_response.json", response_payload, "patch_response")

        candidate_source = synthesis.patched_tool_source
        candidate_hash = sha256_text(candidate_source)
        acceptance_record["candidate_tool_hash"] = candidate_hash
        candidate_path = output_path / f"patched_tool_candidate_v{candidate_version:03d}.py"
        candidate_path.write_text(candidate_source, encoding="utf-8")
        (patch_dir / candidate_path.name).write_text(candidate_source, encoding="utf-8")

        verifier_report = verify_tool_replacement(
            old_tool_source=old_tool_source,
            patched_tool_source=candidate_source,
            observed_df=observed_df,
            candidate_df=candidate_df,
            memory=memory_text,
            tool_state=tool_state,
            round_index=round_index,
        )
        acceptance_record["verifier_called"] = True
        _write_public_json(output_path / "patch_verifier_report.json", verifier_report, "patch_verifier_report")
        _write_public_json(patch_dir / "patch_verifier_report.json", verifier_report, "patch_verifier_report")

        if bool(verifier_report.get("deployable")):
            state.record_tool_replaced(
                source=candidate_source,
                tool_name=REQUIRED_ENTRYPOINT,
                round_index=round_index,
                patch_decision=decision_text,
                verification_result=verifier_report,
                provenance={
                    "patch_prompt_hash": synthesis.request_summary["prompt_hash"],
                    "patch_response_hash": sha256_text(json.dumps(synthesis.payload, sort_keys=True, default=str)),
                },
            )
            acceptance_record.update(
                {
                    "replacement_performed": True,
                    "accepted": True,
                    "active_tool_version_after": int(getattr(state, "active_tool_version", candidate_version)),
                    "reason": "verifier_passed",
                }
            )
        else:
            state.record_tool_patch_rejected(
                round_index=round_index,
                patch_decision=decision_text,
                verification_result=verifier_report,
                provenance={"candidate_tool_hash": candidate_hash},
            )
            acceptance_record["reason"] = str(verifier_report.get("reason", "verifier_failed"))
    except Exception as exc:
        acceptance_record["reason"] = f"patch_attempt_failed:{exc.__class__.__name__}"
        acceptance_record["error"] = str(exc)[:800]
        verifier_report = {
            "passed": False,
            "deployable": False,
            "failed_checks": ["patch_synthesis_or_verification_exception"],
            "warning_checks": [],
            "old_tool_hash": sha256_text(old_tool_source or ""),
            "patched_tool_hash": acceptance_record.get("candidate_tool_hash"),
            "reason": acceptance_record["reason"],
        }
        _write_public_json(output_path / "patch_verifier_report.json", verifier_report, "patch_verifier_report")
        _write_public_json(patch_dir / "patch_verifier_report.json", verifier_report, "patch_verifier_report")
        if hasattr(state, "record_tool_patch_rejected"):
            state.record_tool_patch_rejected(
                round_index=round_index,
                patch_decision=decision_text,
                verification_result=verifier_report,
                provenance={},
            )

    _write_public_json(output_path / "patch_acceptance_record.json", acceptance_record, "patch_acceptance_record")
    _write_public_json(patch_dir / "patch_acceptance_record.json", acceptance_record, "patch_acceptance_record")
    active_pointer = {
        "schema_version": "batch3.active_tool_pointer.v1",
        "round_index": int(round_index),
        "active_tool_version": int(getattr(state, "active_tool_version", current_version)),
        "active_tool_hash": getattr(state, "active_tool_hash", None),
        "tool_patch_count": int(getattr(state, "tool_patch_count", 0)),
        "last_patch_replacement_performed": bool(acceptance_record["replacement_performed"]),
    }
    _write_public_json(output_path / "active_tool_pointer.json", active_pointer, "active_tool_pointer")
    _write_public_json(patch_dir / "active_tool_pointer.json", active_pointer, "active_tool_pointer")
    return dict(acceptance_record)


def _write_public_json(path: Path, payload: Any, label: str) -> None:
    assert_payload_public_safe(payload, label=label)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _schema_summary(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [{"name": str(column), "dtype": str(dtype)} for column, dtype in frame.dtypes.items()]


def _candidate_summary(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "candidate_ids_listed": False,
        "numeric": {},
        "non_numeric": {},
    }
    for column in frame.columns:
        if str(column) == "candidate_id":
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            summary["numeric"][str(column)] = {
                "count": int(numeric.notna().sum()),
                "mean": _safe_float(numeric.mean()),
                "std": _safe_float(numeric.std()),
                "min": _safe_float(numeric.min()),
                "max": _safe_float(numeric.max()),
            }
        else:
            summary["non_numeric"][str(column)] = {
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": [str(value) for value in series.dropna().astype(str).unique()[:5]],
            }
    return summary


def _compact_state(state: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in dict(state or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[str(key)] = value
        elif isinstance(value, (list, tuple)):
            compact[str(key)] = list(value[:10])
        elif isinstance(value, dict):
            compact[str(key)] = {str(k): v for k, v in list(value.items())[:10]}
        else:
            compact[str(key)] = str(type(value).__name__)
    assert_payload_public_safe(compact, label="compact_patch_state")
    return compact


def _compact_research_context(context: dict[str, Any]) -> dict[str, Any]:
    if not context:
        return {}
    cards = context.get("cards", [])
    compact_cards = []
    if isinstance(cards, list):
        for card in cards[:8]:
            if isinstance(card, dict):
                summary = card.get("summary", {})
                compact_cards.append(
                    {
                        "card_id": card.get("card_id"),
                        "title": card.get("title"),
                        "method_tags": list(card.get("method_tags", []))[:8],
                        "one_sentence_takeaway": (
                            summary.get("one_sentence_takeaway") if isinstance(summary, dict) else card.get("one_sentence_takeaway")
                        ),
                        "method_guidance": (
                            list(summary.get("method_guidance", []))[:6]
                            if isinstance(summary, dict)
                            else list(card.get("method_guidance", []))[:6]
                        ),
                        "safety_status": (
                            card.get("safety", {}).get("status") if isinstance(card.get("safety", {}), dict) else card.get("safety_status")
                        ),
                    }
                )
    compact = {
        "context_version": context.get("context_version", "frozen_research_context"),
        "accepted_card_count": int(context.get("accepted_card_count", len(compact_cards)) or len(compact_cards)),
        "rejected_cards_included": False,
        "raw_sources_included": False,
        "cards": compact_cards,
    }
    assert_payload_public_safe(compact, label="compact_patch_research_context")
    return compact


def _as_public_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        payload = dict(value.to_dict())
    elif isinstance(value, dict):
        payload = dict(value)
    else:
        payload = {"value": str(value)}
    assert_payload_public_safe(payload, label="patch_public_input")
    return payload


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _reject_forbidden_payload_terms(payload: Any) -> None:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).lower()
    hits = [term for term in FORBIDDEN_TERMS if term.lower() in text]
    if hits:
        raise ToolPatchParseError("Patch response contains forbidden contract terms.")
