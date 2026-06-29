"""Reveal-time reward aggregation for policy editing."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research_tool_agent_full_pool.harness.compiler import source_complexity_penalty
from research_tool_agent_full_pool.harness.specs import RewardRecord


def build_reward_record(
    *,
    round_index: int,
    selected_candidate_id: str,
    revealed_rows: pd.DataFrame,
    previous_best_y: float | None,
    current_best_y: float | None,
    objective_name: str = "yield",
    objective_direction: str = "maximize",
    gate_report: dict[str, Any] | None = None,
    fallback_used: bool = False,
    patch_deployed: bool = False,
    source: str = "",
    selection_attribution: dict[str, Any] | None = None,
) -> RewardRecord:
    """Compute a scalar reward only after evaluator reveal."""

    revealed_y = _revealed_y(revealed_rows, objective_name)
    delta = _delta_best(previous_best_y, current_best_y, objective_direction)
    checks = dict((gate_report or {}).get("checks", {}))
    verifier = checks.get("replacement_verifier", {}) if isinstance(checks.get("replacement_verifier"), dict) else {}
    verifier_checks = verifier.get("checks", {}) if isinstance(verifier.get("checks"), dict) else {}
    row_order = verifier_checks.get("row_order_perturbation", {})
    row_order_stable = bool(row_order.get("passed", True)) if isinstance(row_order, dict) else True
    tool_output_valid = bool((gate_report or {}).get("deployable", (gate_report or {}).get("passed", False)))
    patch_rejected = bool(gate_report) and not tool_output_valid
    complexity = source_complexity_penalty(source)
    reward = float(delta)
    shortfall_penalty = _shortfall_penalty(
        revealed_y=revealed_y,
        previous_best_y=previous_best_y,
        objective_direction=objective_direction,
    )
    reward += 0.10 if tool_output_valid else -0.20
    reward += 0.10 if row_order_stable else -0.30
    reward -= 0.20 if fallback_used else 0.0
    reward -= 0.20 if patch_rejected else 0.0
    reward += 0.05 if patch_deployed else 0.0
    attribution = dict(selection_attribution or {})
    llm_changed_final_selection = bool(attribution.get("llm_changed_final_selection", False))
    if llm_changed_final_selection and delta > 0.0:
        reward += 0.08
    reward -= shortfall_penalty
    reward -= float(complexity)
    return RewardRecord(
        round_index=int(round_index),
        selected_candidate_id=str(selected_candidate_id),
        revealed_y=revealed_y,
        previous_best_y=previous_best_y,
        current_best_y=current_best_y,
        delta_best_y=float(delta),
        tool_output_valid=tool_output_valid,
        row_order_stable=row_order_stable,
        fallback_used=bool(fallback_used),
        patch_deployed=bool(patch_deployed),
        patch_rejected=patch_rejected,
        complexity_penalty=float(complexity),
        total_reward=float(reward),
        components={
            "delta_best_y": float(delta),
            "tool_output_valid_bonus": 0.10 if tool_output_valid else -0.20,
            "row_order_bonus": 0.10 if row_order_stable else -0.30,
            "fallback_penalty": -0.20 if fallback_used else 0.0,
            "patch_rejected_penalty": -0.20 if patch_rejected else 0.0,
            "patch_deployed_bonus": 0.05 if patch_deployed else 0.0,
            "llm_changed_final_selection_bonus": 0.08 if llm_changed_final_selection and delta > 0.0 else 0.0,
            "shortfall_penalty": -float(shortfall_penalty),
            "complexity_penalty": -float(complexity),
            "selection_attribution": attribution,
        },
    )


def _revealed_y(revealed_rows: pd.DataFrame, objective_name: str) -> float | None:
    if objective_name not in revealed_rows.columns or revealed_rows.empty:
        return None
    try:
        value = float(revealed_rows[objective_name].iloc[0])
    except (TypeError, ValueError):
        return None
    if pd.isna(value):
        return None
    return value


def _delta_best(previous_best_y: float | None, current_best_y: float | None, objective_direction: str) -> float:
    if previous_best_y is None or current_best_y is None:
        return 0.0
    direction = str(objective_direction).lower()
    if direction == "minimize":
        return max(0.0, float(previous_best_y) - float(current_best_y))
    return max(0.0, float(current_best_y) - float(previous_best_y))


def _shortfall_penalty(
    *,
    revealed_y: float | None,
    previous_best_y: float | None,
    objective_direction: str,
) -> float:
    if revealed_y is None or previous_best_y is None:
        return 0.0
    direction = str(objective_direction).lower()
    shortfall = float(revealed_y) - float(previous_best_y) if direction == "minimize" else float(previous_best_y) - float(revealed_y)
    if shortfall <= 0:
        return 0.0
    scale = max(abs(float(previous_best_y)), 1.0)
    return min(0.75, max(0.0, 0.35 * (shortfall / scale)))
