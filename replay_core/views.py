from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from replay_core.state import ReplayState
from replay_core.schema import ReplayTables, is_forbidden_decision_column


TARGET_LIKE_COLUMN_NAMES = {
    "yield",
    "conversion",
    "outcome",
    "target",
    "turnover",
    "bad_outcome_flag",
    "acceptable_outcome_flag",
    "high_quality_flag",
    "failure_avoidance_score",
    "global_rank",
    "top_k_label",
    "oracle_score",
    "objective_missing",
}


@dataclass(frozen=True)
class DecisionView:
    candidate_view: pd.DataFrame
    observed_inputs_view: pd.DataFrame
    observed_objective_view: pd.DataFrame
    # Backward-compatible alias for modules that need revealed objective labels.
    observed_view: pd.DataFrame
    group_columns: tuple[str, ...]
    flag_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    cost_columns: tuple[str, ...]


def is_target_like_column(column: str) -> bool:
    normalized = str(column).lower()
    return (
        is_forbidden_decision_column(normalized)
        or normalized in TARGET_LIKE_COLUMN_NAMES
        or normalized.endswith("_target")
        or normalized.endswith("_outcome")
        or normalized.endswith("_outcome_flag")
        or normalized.endswith("_label")
        or normalized.endswith("_rank")
        or normalized.startswith("oracle_")
    )


def evaluator_only_columns(tables: ReplayTables) -> tuple[str, ...]:
    columns: set[str] = set(tables.evaluator_columns())
    columns.update(column for column in tables.outcome_table.columns if column != tables.id_column)
    for table in (tables.candidate_table, *tables.visible_side_tables()):
        if table is None:
            continue
        columns.update(column for column in table.columns if is_target_like_column(str(column)))
    return tuple(sorted(columns))


def merge_visible_side_tables(tables: ReplayTables, base: pd.DataFrame) -> pd.DataFrame:
    frame = base.copy()
    for label, table in (
        ("metadata_table", tables.metadata_table),
        ("group_table", tables.group_table),
        ("flag_table", tables.flag_table),
        ("cost_table", tables.cost_table),
    ):
        if table is None:
            continue
        overlap = [column for column in table.columns if column != tables.id_column and column in frame.columns]
        if overlap:
            raise ValueError(f"{label} overlaps candidate view columns: {sorted(overlap)}")
        frame = frame.merge(table, on=tables.id_column, how="left", validate="one_to_one", sort=False)
    return frame


def sanitize_decision_frame(frame: pd.DataFrame, tables: ReplayTables) -> pd.DataFrame:
    forbidden = set(evaluator_only_columns(tables))
    dropped = [column for column in frame.columns if column in forbidden or is_target_like_column(str(column))]
    sanitized = frame.drop(columns=dropped, errors="ignore").copy()
    validate_candidate_view(sanitized, tables)
    return sanitized


def validate_candidate_view(frame: pd.DataFrame, tables: ReplayTables) -> None:
    forbidden = set(evaluator_only_columns(tables))
    leaks = [column for column in frame.columns if column in forbidden or is_target_like_column(str(column))]
    if leaks:
        raise AssertionError(
            "candidate_view contains evaluator-only or target-derived columns: "
            + ", ".join(sorted(leaks))
        )


def build_candidate_view(tables: ReplayTables, remaining_candidates: pd.DataFrame) -> pd.DataFrame:
    base = remaining_candidates.copy()
    if tables.id_column not in base.columns:
        raise ValueError(f"remaining_candidates must contain {tables.id_column!r}.")
    merged = merge_visible_side_tables(tables, base)
    candidate_view = sanitize_decision_frame(merged, tables)
    validate_candidate_view(candidate_view, tables)
    return candidate_view.reset_index(drop=True)


def build_observed_inputs_view(tables: ReplayTables, state: ReplayState) -> pd.DataFrame:
    """Observed candidate inputs plus visible non-leaking metadata only."""

    observed = merge_visible_side_tables(tables, state.observed_candidates)
    return sanitize_decision_frame(observed, tables).reset_index(drop=True)


def build_observed_objective_view(tables: ReplayTables, state: ReplayState) -> pd.DataFrame:
    """Observed inputs plus revealed objective targets, excluding hidden outcomes."""

    observed_inputs = build_observed_inputs_view(tables, state)
    target_columns = [column for column in tables.target_columns if column in state.observed_outcomes.columns]
    objective_values = state.observed_outcomes.loc[:, [tables.id_column, *target_columns]].copy()
    observed = observed_inputs.merge(
        objective_values,
        on=tables.id_column,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    _validate_observed_objective_view(observed, tables)
    return observed.reset_index(drop=True)


def build_observed_evaluator_artifact_view(tables: ReplayTables, state: ReplayState) -> pd.DataFrame:
    """Reporting-only observed rows with all evaluator-revealed columns."""

    observed = state.observed_view()
    return merge_visible_side_tables(tables, observed).reset_index(drop=True)


def build_observed_view(tables: ReplayTables, state: ReplayState) -> pd.DataFrame:
    """Backward-compatible name for the guidance-safe objective-labeled view."""

    return build_observed_objective_view(tables, state)


def build_decision_view(tables: ReplayTables, state: ReplayState) -> DecisionView:
    candidate_view = build_candidate_view(tables, state.remaining_candidates)
    observed_inputs_view = build_observed_inputs_view(tables, state)
    observed_objective_view = build_observed_objective_view(tables, state)
    group_columns = _visible_non_id_columns(tables.group_table, tables)
    flag_columns = tuple(
        column
        for column in _visible_non_id_columns(tables.flag_table, tables)
        if column in candidate_view.columns
    )
    cost_columns = _visible_non_id_columns(tables.cost_table, tables)
    non_metadata = {
        tables.id_column,
        *tables.decision_columns,
        *group_columns,
        *flag_columns,
        *cost_columns,
    }
    metadata_columns = tuple(column for column in candidate_view.columns if column not in non_metadata)
    return DecisionView(
        candidate_view=candidate_view,
        observed_inputs_view=observed_inputs_view,
        observed_objective_view=observed_objective_view,
        observed_view=observed_objective_view,
        group_columns=tuple(column for column in group_columns if column in candidate_view.columns),
        flag_columns=flag_columns,
        metadata_columns=metadata_columns,
        cost_columns=tuple(column for column in cost_columns if column in candidate_view.columns),
    )


def _visible_non_id_columns(table: pd.DataFrame | None, tables: ReplayTables) -> tuple[str, ...]:
    if table is None:
        return ()
    forbidden = set(evaluator_only_columns(tables))
    return tuple(
        column
        for column in table.columns
        if column != tables.id_column and column not in forbidden and not is_target_like_column(str(column))
    )


def _validate_observed_objective_view(frame: pd.DataFrame, tables: ReplayTables) -> None:
    allowed_targets = set(tables.target_columns)
    forbidden = set(evaluator_only_columns(tables)) - allowed_targets
    leaks = [
        column
        for column in frame.columns
        if column in forbidden or (is_target_like_column(str(column)) and column not in allowed_targets)
    ]
    if leaks:
        raise AssertionError(
            "observed_objective_view contains hidden or target-derived columns: "
            + ", ".join(sorted(leaks))
        )
