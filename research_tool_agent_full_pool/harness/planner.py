"""Task planning for self-evolving policy editing."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from research_tool_agent_full_pool.decision_artifacts import assert_payload_public_safe
from research_tool_agent_full_pool.harness.specs import DEFAULT_REQUIRED_CHECKS, TaskPlan, compact_json
from research_tool_agent_full_pool.research_cache import sha256_text


PLANNER_SCHEMA_VERSION = "self_evolving_task_plan_v1"


class PlannerParseError(ValueError):
    """Raised when an LLM planner response is malformed."""


def plan_next_task(
    *,
    client: Any | None,
    mode: str,
    run_id: str,
    round_index: int,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    memory_text: str,
    policy_state: dict[str, Any],
    last_gate_report: dict[str, Any] | None = None,
    parser_error: str | None = None,
) -> tuple[TaskPlan, dict[str, Any]]:
    """Return the next public-safe editing plan."""

    if str(mode).lower() == "fake" or client is None:
        action = "create_skill" if not policy_state.get("active_skill_id") else "patch_skill"
        if last_gate_report and not bool(last_gate_report.get("deployable", last_gate_report.get("passed", False))):
            action = "reuse_active_skill" if policy_state.get("active_skill_id") else "create_skill"
        plan = TaskPlan(
            action=action,  # type: ignore[arg-type]
            skill_family="ranker",
            objective="Improve finite-budget best observed yield with deterministic public-feature ranking.",
            target_skill_id=policy_state.get("active_skill_id"),
            risk_budget="low",
            required_checks=list(DEFAULT_REQUIRED_CHECKS),
            rationale="deterministic_fake_planner",
        )
        return plan, {"mode": "fake", "planner_schema_version": PLANNER_SCHEMA_VERSION}

    prompt = build_planner_prompt(
        run_id=run_id,
        round_index=round_index,
        observed_df=observed_df,
        candidate_df=candidate_df,
        memory_text=memory_text,
        policy_state=policy_state,
        last_gate_report=last_gate_report or {},
        parser_error=parser_error,
    )
    raw_text = client.create_tool(
        messages=[
            {
                "role": "developer",
                "content": (
                    "You are the public-safe planner for an offline finite-pool optimization harness. "
                    "Return only the JSON object required by the schema. Use only public observed rows, "
                    "public candidate features, memory, and gate/reward summaries supplied by the user."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        json_schema=planner_json_schema(run_id=run_id, round_index=round_index),
        schema_name="self_evolving_task_plan",
        schema_description="One public-safe policy-editing plan for the next optimization round.",
    )
    plan = parse_task_plan_json(raw_text, expected_run_id=run_id, expected_round_index=round_index)
    return plan, {
        "mode": "api",
        "raw_text": raw_text,
        "prompt_hash": sha256_text(prompt),
        "prompt_character_count": len(prompt),
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
    }


def build_planner_prompt(
    *,
    run_id: str,
    round_index: int,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    memory_text: str,
    policy_state: dict[str, Any],
    last_gate_report: dict[str, Any],
    parser_error: str | None = None,
) -> str:
    payload = {
        "task": "plan_next_public_safe_policy_edit",
        "schema_version": PLANNER_SCHEMA_VERSION,
        "run_id": run_id,
        "round_index": int(round_index),
        "observed_summary": _observed_summary(observed_df),
        "candidate_summary": _candidate_summary(candidate_df),
        "memory_summary": str(memory_text)[:1200],
        "policy_state": _compact_policy_state(policy_state),
        "reward_trend": _reward_trend(policy_state),
        "last_gate_report": _compact_gate(last_gate_report),
        "parser_error": str(parser_error or "")[:500],
        "allowed_actions": ["create_skill", "patch_skill", "reuse_active_skill"],
        "allowed_skill_families": ["ranker", "constraint", "exploration", "calibrator", "fallback"],
        "instructions": [
            "Plan only one next edit/action for a generated optimizer skill.",
            "Use only public views and already revealed objective values; do not seek evaluator-private labels, ranking answer keys, baseline artifacts, files, network, or credentials.",
            "Prefer low-risk deterministic code edits that compile to rank_candidates.",
            "The final deployed tool must score the full remaining candidate_df, not a menu.",
            "Reuse is allowed only when the active skill is both deployable and still improving or diversifying selected outcomes.",
            "Do not patch solely because one round failed to improve immediately after a recent positive best-observed improvement; reuse once to collect another reveal unless the last selected yield was clearly poor.",
            "Plan patch_skill mainly after two or more consecutive no-improvement rounds, or after a revealed selection far below the current best.",
            "If patching after stagnation, make a conservative edit that blends the active scoring rule with bounded diversity/novelty; avoid replacing a previously improving policy with an unrelated scorer.",
            "If the active skill repeatedly selects lower-yield near-duplicates after a first improvement, patch it to avoid over-exploiting that local region.",
            "Return JSON with task_plan and self_reported_forbidden_info_used=false.",
        ],
        "output_schema": {
            "schema_version": PLANNER_SCHEMA_VERSION,
            "run_id": run_id,
            "round_index": int(round_index),
            "task_plan": {
                "action": "create_skill|patch_skill|reuse_active_skill",
                "skill_family": "ranker|constraint|exploration|calibrator|fallback",
                "objective": "short public-safe objective",
                "target_skill_id": "active skill id or null",
                "risk_budget": "low|medium|high",
                "required_checks": list(DEFAULT_REQUIRED_CHECKS),
                "rationale": "short public-safe rationale",
            },
            "self_reported_forbidden_info_used": False,
        },
    }
    assert_payload_public_safe(payload, label="self_evolving_planner_prompt")
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def planner_json_schema(*, run_id: str, round_index: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "run_id",
            "round_index",
            "task_plan",
            "self_reported_forbidden_info_used",
        ],
        "properties": {
            "schema_version": {"type": "string", "enum": [PLANNER_SCHEMA_VERSION]},
            "run_id": {"type": "string", "enum": [str(run_id)]},
            "round_index": {"type": "integer", "enum": [int(round_index)]},
            "task_plan": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "action",
                    "skill_family",
                    "objective",
                    "target_skill_id",
                    "risk_budget",
                    "required_checks",
                    "rationale",
                ],
                "properties": {
                    "action": {"type": "string", "enum": ["create_skill", "patch_skill", "reuse_active_skill"]},
                    "skill_family": {
                        "type": "string",
                        "enum": ["ranker", "constraint", "exploration", "calibrator", "fallback"],
                    },
                    "objective": {"type": "string", "minLength": 1},
                    "target_skill_id": {"type": ["string", "null"]},
                    "risk_budget": {"type": "string", "enum": ["low", "medium", "high"]},
                    "required_checks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "rationale": {"type": "string"},
                },
            },
            "self_reported_forbidden_info_used": {"type": "boolean", "enum": [False]},
        },
    }


def parse_task_plan_json(raw_text: str, *, expected_run_id: str, expected_round_index: int) -> TaskPlan:
    text = _extract_json_object_text(str(raw_text).strip())
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlannerParseError("Planner output is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise PlannerParseError("Planner output must decode to an object.")
    assert_payload_public_safe(payload, label="self_evolving_planner_raw_response")
    payload = _normalize_planner_payload(payload)
    allowed = {"schema_version", "run_id", "round_index", "task_plan", "self_reported_forbidden_info_used"}
    extra = set(payload) - allowed
    if extra:
        raise PlannerParseError(f"Unsupported planner fields: {sorted(extra)}")
    assert_payload_public_safe(payload, label="self_evolving_planner_response")
    if payload.get("schema_version") != PLANNER_SCHEMA_VERSION:
        raise PlannerParseError("schema_version mismatch.")
    if payload.get("run_id") != expected_run_id:
        raise PlannerParseError("run_id mismatch.")
    if int(payload.get("round_index", -999)) != int(expected_round_index):
        raise PlannerParseError("round_index mismatch.")
    if payload.get("self_reported_forbidden_info_used") is not False:
        raise PlannerParseError("self_reported_forbidden_info_used must be false.")
    task_plan = payload.get("task_plan")
    if not isinstance(task_plan, dict):
        raise PlannerParseError("task_plan must be an object.")
    return TaskPlan.from_dict(task_plan)


def _extract_json_object_text(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    raise PlannerParseError("Planner output must contain one JSON object.")


def _normalize_planner_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize harmless check-name variants before leakage scanning."""

    normalized = dict(payload)
    task_plan = normalized.get("task_plan")
    if isinstance(task_plan, dict):
        plan = dict(task_plan)
        checks = plan.get("required_checks")
        if isinstance(checks, list):
            mapping = {
                "full_pool_rank": "full_pool",
                "full-pool rank": "full_pool",
                "full_pool_scoring": "full_pool",
                "full-pool scoring": "full_pool",
            }
            plan["required_checks"] = [
                mapping.get(str(item).strip().lower(), str(item).strip())
                for item in checks
                if str(item).strip()
            ]
        normalized["task_plan"] = plan
    return normalized


def _observed_summary(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"row_count": int(len(frame)), "columns": list(frame.columns)}
    if "observed_y" in frame.columns:
        y = pd.to_numeric(frame["observed_y"], errors="coerce").dropna()
        result["observed_y"] = {
            "count": int(len(y)),
            "min": _safe_float(y.min()) if len(y) else None,
            "max": _safe_float(y.max()) if len(y) else None,
            "mean": _safe_float(y.mean()) if len(y) else None,
        }
    return result


def _candidate_summary(frame: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = [
        str(column)
        for column in frame.columns
        if column != "candidate_id" and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return {
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "numeric_columns": numeric_columns,
        "schema_hash": sha256_text(compact_json([(str(col), str(dtype)) for col, dtype in frame.dtypes.items()])),
    }


def _compact_policy_state(policy_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "active_skill_id": policy_state.get("active_skill_id"),
        "active_version": policy_state.get("active_version"),
        "recent_rewards": list(policy_state.get("reward_history", []))[-5:],
        "recent_gates": list(policy_state.get("gate_history", []))[-3:],
        "fallback_count": policy_state.get("fallback_count", 0),
        "rollback_count": policy_state.get("rollback_count", 0),
    }


def _reward_trend(policy_state: dict[str, Any]) -> dict[str, Any]:
    rewards = [row for row in list(policy_state.get("reward_history", []))[-5:] if isinstance(row, dict)]
    deltas = [_safe_float(row.get("delta_best_y")) for row in rewards]
    deltas = [float(value) for value in deltas if value is not None]
    revealed = [_safe_float(row.get("revealed_y")) for row in rewards]
    revealed = [float(value) for value in revealed if value is not None]
    current_best = _safe_float(rewards[-1].get("current_best_y")) if rewards else None
    last_revealed = _safe_float(rewards[-1].get("revealed_y")) if rewards else None
    trailing_no_improvement = 0
    for value in reversed(deltas):
        if value > 1e-12:
            break
        trailing_no_improvement += 1
    last_selected_far_below_best = _is_far_below_best(last_revealed=last_revealed, current_best=current_best)
    active_exists = bool(policy_state.get("active_skill_id"))
    should_patch = active_exists and (trailing_no_improvement >= 2 or last_selected_far_below_best)
    recent_improvement = any(value > 1e-12 for value in deltas[-2:])
    return {
        "recent_reward_count": len(rewards),
        "recent_delta_best_y": deltas,
        "recent_revealed_y": revealed,
        "current_best_y": current_best,
        "last_revealed_y": last_revealed,
        "trailing_no_improvement_count": trailing_no_improvement,
        "last_selected_far_below_best": last_selected_far_below_best,
        "recent_improvement_within_two_rounds": recent_improvement,
        "should_patch_if_active_skill_exists": should_patch,
        "patch_guidance": (
            "Patch conservatively: either two or more consecutive no-improvement rounds have accumulated, or the last selection was far below the current best."
            if should_patch
            else "Prefer reuse: do not patch after a single no-improvement round when the active skill recently improved."
        ),
    }


def _is_far_below_best(*, last_revealed: float | None, current_best: float | None) -> bool:
    if last_revealed is None or current_best is None:
        return False
    if current_best <= 0:
        return last_revealed < current_best
    return (current_best - last_revealed) >= max(15.0, 0.30 * abs(current_best))


def _compact_gate(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": gate.get("passed"),
        "deployable": gate.get("deployable"),
        "failed_checks": gate.get("failed_checks", []),
        "warning_checks": gate.get("warning_checks", []),
        "reason": gate.get("reason", ""),
    }


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result
