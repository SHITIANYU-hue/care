"""Public-safe dataset profiles for controlled live research collection."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from research_tool_agent_full_pool.decision_artifacts import scan_payload_forbidden
from research_tool_agent_full_pool.views import (
    build_full_remaining_candidate_df,
    build_observed_df_from_revealed_state,
    validate_observed_df,
    validate_public_candidate_df,
)


PROFILE_VERSION = "batch2.public_dataset_profile.v1"
MAX_PROFILE_COLUMNS = 40
MAX_SAMPLE_VALUES = 8
MAX_PUBLIC_SUMMARY_CHARS = 1200


def build_public_dataset_profile(
    *,
    tables: Any | None = None,
    replay_state: Any | None = None,
    config: Any | None = None,
    observed_df: pd.DataFrame | None = None,
    candidate_df: pd.DataFrame | None = None,
    memory_text: str = "",
    tool_state: dict[str, Any] | None = None,
    strategy_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the only dataset summary allowed before live research retrieval.

    The profile is intentionally schema- and summary-level. It uses revealed
    observations only and never lists remaining target values, private ID maps,
    evaluator metadata, reference-BO outputs, or posthoc ranks.
    """

    if tables is not None and replay_state is not None:
        objective_name = str(getattr(config, "target_column", "") or tables.target_columns[0])
        if observed_df is None:
            observed_df = build_observed_df_from_revealed_state(
                tables,
                replay_state,
                objective_name=objective_name,
            )
        if candidate_df is None:
            candidate_df = build_full_remaining_candidate_df(tables, replay_state)
    if observed_df is None or candidate_df is None:
        raise ValueError("build_public_dataset_profile requires tables/state or observed_df/candidate_df.")

    observed = observed_df.copy()
    candidates = candidate_df.copy()
    validate_observed_df(observed)
    validate_public_candidate_df(candidates)

    dataset_label = _public_dataset_label(tables=tables, config=config)
    objective_direction = str(
        getattr(config, "objective_direction", "")
        or getattr(tables, "objective_direction", "")
        or "maximize"
    )
    target_label = str(
        getattr(config, "target_column", "")
        or (getattr(tables, "target_columns", ("observed_y",))[0] if tables is not None else "observed_y")
    )
    feature_columns = [
        str(column)
        for column in candidates.columns
        if str(column) not in {"candidate_id", "observation_id", "observed_y"}
    ]
    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(candidates[column])
    ]
    categorical_columns = [
        column
        for column in feature_columns
        if column not in numeric_columns
    ]

    profile: dict[str, Any] = {
        "profile_version": PROFILE_VERSION,
        "dataset_label": dataset_label,
        "dataset_identity_public": _public_dataset_identity(tables=tables),
        "task_summary": _task_summary(dataset_label, feature_columns),
        "objective": {
            "target_label": target_label,
            "direction": objective_direction,
            "revealed_observed_only": True,
        },
        "observed_count": int(len(observed)),
        "remaining_candidate_count": int(len(candidates)),
        "feature_columns": feature_columns[:MAX_PROFILE_COLUMNS],
        "candidate_schema_summary": {
            "row_count": int(len(candidates)),
            "column_count": int(len(candidates.columns)),
            "feature_column_count": len(feature_columns),
            "numeric_feature_count": len(numeric_columns),
            "categorical_feature_count": len(categorical_columns),
            "candidate_ids_listed": False,
            "private_candidate_maps_included": False,
        },
        "numeric_column_summaries": [
            _numeric_summary(candidates, column)
            for column in numeric_columns[:MAX_PROFILE_COLUMNS]
        ],
        "categorical_column_summaries": [
            _categorical_summary(candidates, column)
            for column in categorical_columns[:MAX_PROFILE_COLUMNS]
        ],
        "observed_y_summary": _observed_y_summary(observed),
        "state_public_summary": {
            "memory": _safe_public_state_summary(memory_text, label="memory"),
            "tool_state": _safe_public_state_summary(tool_state or {}, label="tool_state"),
            "strategy_state": _safe_public_state_summary(strategy_state or {}, label="strategy_state"),
        },
        "non_public_information_policy": {
            "remaining_targets_included": False,
            "unrevealed_secondary_metrics_included": False,
            "post_decision_metrics_included": False,
            "reference_optimizer_outputs_included": False,
            "score_cache_artifacts_included": False,
            "internal_id_maps_included": False,
            "offline_runner_details_included": False,
            "secret_values_included": False,
        },
    }
    violations = profile_public_safety_violations(profile)
    if violations:
        raise ValueError("Dataset profile contains forbidden public-safety terms.")
    return profile


def profile_public_safety_violations(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Return forbidden-term hits in a profile without exposing snippets."""

    return scan_payload_forbidden(profile, label="dataset_profile")


def _public_dataset_label(*, tables: Any | None, config: Any | None) -> str:
    configured = str(
        getattr(config, "dataset_name", "")
        or getattr(config, "dataset", "")
        or getattr(tables, "dataset_name", "")
        or "public optimization dataset"
    ).strip()
    return configured or "public optimization dataset"


def _public_dataset_identity(*, tables: Any | None) -> str:
    identity = str(getattr(tables, "dataset_identity", "") if tables is not None else "").strip()
    if not identity:
        return ""
    return identity


def _task_summary(dataset_label: str, feature_columns: list[str]) -> str:
    lowered = " ".join(feature_columns).lower()
    if "catalyst" in lowered or "temperature" in lowered or "ligand" in lowered:
        return "reaction optimization over public process and categorical design features"
    if feature_columns:
        return f"finite-pool optimization over {len(feature_columns)} public feature columns"
    return f"finite-pool optimization task for {dataset_label}"


def _numeric_summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    series = pd.to_numeric(frame[column], errors="coerce")
    return {
        "column": str(column),
        "missing_count": int(series.isna().sum()),
        "min": _safe_float(series.min()),
        "max": _safe_float(series.max()),
        "mean": _safe_float(series.mean()),
        "std": _safe_float(series.std()),
    }


def _categorical_summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    series = frame[column]
    values = [str(value) for value in series.dropna().astype(str).unique()[:MAX_SAMPLE_VALUES]]
    return {
        "column": str(column),
        "missing_count": int(series.isna().sum()),
        "unique_count": int(series.nunique(dropna=True)),
        "sample_values": values,
    }


def _observed_y_summary(observed_df: pd.DataFrame) -> dict[str, Any]:
    series = pd.to_numeric(observed_df["observed_y"], errors="coerce")
    return {
        "count": int(series.notna().sum()),
        "min": _safe_float(series.min()),
        "max": _safe_float(series.max()),
        "mean": _safe_float(series.mean()),
        "median": _safe_float(series.median()),
        "revealed_observations_only": True,
    }


def _safe_public_state_summary(value: Any, *, label: str) -> dict[str, Any]:
    payload = value
    hits = scan_payload_forbidden(payload, label=label)
    if hits:
        return {
            "included": False,
            "reason": "forbidden_public_safety_terms_detected",
            "forbidden_hit_count": len(hits),
        }
    if isinstance(value, str):
        text = value.strip()
        return {
            "included": bool(text),
            "kind": "text",
            "character_count": len(text),
            "summary": _truncate(text),
        }
    if isinstance(value, dict):
        text = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
        return {
            "included": bool(value),
            "kind": "mapping",
            "key_count": len(value),
            "keys": [str(key) for key in sorted(value.keys(), key=str)[:30]],
            "character_count": len(text),
            "summary": _truncate(text),
        }
    return {
        "included": value is not None,
        "kind": type(value).__name__,
        "summary": _truncate(str(value)),
    }


def _truncate(text: str, max_chars: int = MAX_PUBLIC_SUMMARY_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result
