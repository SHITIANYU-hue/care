"""Data-view builders for the full-pool ResearchToolAgent.

These skeletons define the data boundary for Step 1. They intentionally avoid
dataset loading in Step 0.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from research_tool_agent_full_pool.safety import (
    FORBIDDEN_CANDIDATE_FIELDS,
    FORBIDDEN_OBSERVED_FIELDS,
    assert_no_forbidden_fields,
    find_forbidden_fields,
)


def _columns_from_dataframe_like(df: Any) -> list[Any]:
    """Extract columns from a pandas-like object without importing pandas."""

    columns = getattr(df, "columns", None)
    if columns is None:
        return []
    return list(columns)


def _display_mapping_for_ids(internal_ids: list[Any]) -> dict[str, Any]:
    """Build stable opaque candidate display IDs for a run-local candidate set."""

    ordered = sorted(dict.fromkeys(internal_ids), key=lambda value: str(value))
    return {f"cand_{idx:06d}": internal_id for idx, internal_id in enumerate(ordered, start=1)}


def _reverse_mapping(mapping: dict[str, Any]) -> dict[Any, str]:
    return {internal_id: display_id for display_id, internal_id in mapping.items()}


def _full_internal_ids_from_state(state: Any, id_column: str) -> list[Any]:
    observed = (
        state.observed_candidates[id_column].tolist()
        if hasattr(state, "observed_candidates") and id_column in state.observed_candidates.columns
        else []
    )
    remaining = (
        state.remaining_candidates[id_column].tolist()
        if hasattr(state, "remaining_candidates") and id_column in state.remaining_candidates.columns
        else []
    )
    return [*observed, *remaining]


def _safe_public_columns(
    frame: pd.DataFrame,
    *,
    id_column: str,
    allowed_objective_columns: tuple[str, ...] = (),
    forbidden_terms: tuple[str, ...],
) -> list[str]:
    allowed = {id_column, *allowed_objective_columns}
    columns: list[str] = []
    for column in frame.columns:
        name = str(column)
        if name in allowed:
            columns.append(name)
            continue
        if find_forbidden_fields([name], forbidden_terms):
            continue
        columns.append(name)
    return columns


def build_observed_df_from_revealed_state(
    tables: Any,
    state: Any,
    *,
    objective_name: str | None = None,
) -> pd.DataFrame:
    """Build `observed_df` from revealed replay state only.

    `observed_df` must include only public features plus revealed `y` for
    already observed experiments. It must not include BO artifacts, oracle rank,
    unobserved outcomes, turnover, raw row indices, source paths, or secrets.
    """

    from replay_core.views import build_observed_objective_view

    id_column = tables.id_column
    objective = objective_name or tables.target_columns[0]
    display_to_internal = _display_mapping_for_ids(_full_internal_ids_from_state(state, id_column))
    internal_to_display = _reverse_mapping(display_to_internal)

    observed_view = build_observed_objective_view(tables, state)
    keep = _safe_public_columns(
        observed_view,
        id_column=id_column,
        allowed_objective_columns=(objective,),
        forbidden_terms=FORBIDDEN_OBSERVED_FIELDS,
    )
    observed = observed_view.loc[:, keep].copy().reset_index(drop=True)
    if objective not in observed.columns:
        raise ValueError(f"Observed view is missing revealed objective column {objective!r}.")

    internal_ids = observed[id_column].tolist()
    observed.insert(0, "observation_id", [f"obs_{idx:06d}" for idx in range(1, len(observed) + 1)])
    observed[id_column] = [internal_to_display[internal_id] for internal_id in internal_ids]
    observed = observed.rename(columns={id_column: "candidate_id", objective: "observed_y"})
    observed.attrs["display_to_internal_id"] = dict(display_to_internal)
    observed.attrs["observed_display_to_internal_id"] = {
        internal_to_display[internal_id]: internal_id for internal_id in internal_ids
    }
    observed.attrs["objective_name"] = objective
    observed.attrs["internal_id_column"] = id_column
    validate_observed_df(observed)
    return observed


def build_full_remaining_candidate_df(tables: Any, state: Any) -> pd.DataFrame:
    """Build the full remaining public-safe candidate pool.

    `candidate_df` must include every remaining eligible candidate with public
    features only. It must not include hidden `y`, turnover, oracle fields,
    BO ranks, acquisition scores, predictive statistics, raw row indices,
    source paths, candidate score artifact data, or secrets.
    """

    from replay_core.views import build_candidate_view

    id_column = tables.id_column
    display_to_internal = _display_mapping_for_ids(_full_internal_ids_from_state(state, id_column))
    internal_to_display = _reverse_mapping(display_to_internal)

    candidate_view = build_candidate_view(tables, state.remaining_candidates)
    keep = _safe_public_columns(
        candidate_view,
        id_column=id_column,
        forbidden_terms=FORBIDDEN_CANDIDATE_FIELDS,
    )
    candidates = candidate_view.loc[:, keep].copy().reset_index(drop=True)
    internal_ids = candidates[id_column].tolist()
    candidates[id_column] = [internal_to_display[internal_id] for internal_id in internal_ids]
    candidates = candidates.rename(columns={id_column: "candidate_id"})
    candidates.attrs["display_to_internal_id"] = {
        internal_to_display[internal_id]: internal_id for internal_id in internal_ids
    }
    candidates.attrs["full_display_to_internal_id"] = dict(display_to_internal)
    candidates.attrs["canonical_candidate_id_order"] = [str(value) for value in candidates["candidate_id"].tolist()]
    candidates.attrs["internal_id_column"] = id_column
    candidates.attrs["full_remaining_pool_size"] = len(state.remaining_candidates)
    validate_public_candidate_df(candidates)
    return candidates


def validate_public_candidate_df(candidate_df: Any) -> None:
    """Validate that a candidate view contains only public-safe columns."""

    columns = _columns_from_dataframe_like(candidate_df)
    if "candidate_id" not in columns:
        raise ValueError("candidate_df must contain opaque display column 'candidate_id'.")
    assert_no_forbidden_fields(
        columns,
        FORBIDDEN_CANDIDATE_FIELDS,
        context="candidate_df",
    )


def validate_observed_df(observed_df: Any) -> None:
    """Validate that an observed view contains no unrevealed/oracle/BO artifacts.

    Revealed outcome columns are allowed in `observed_df`; hidden or unobserved
    outcome columns remain forbidden.
    """

    columns = _columns_from_dataframe_like(observed_df)
    required = {"observation_id", "candidate_id", "observed_y"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"observed_df is missing required columns: {sorted(missing)}")
    assert_no_forbidden_fields(
        columns,
        FORBIDDEN_OBSERVED_FIELDS,
        context="observed_df",
    )


def map_display_candidate_to_internal_id(candidate_df: pd.DataFrame, display_candidate_id: str) -> Any:
    """Map a tool-facing display candidate ID back to the internal replay ID."""

    mapping = candidate_df.attrs.get("display_to_internal_id")
    if not isinstance(mapping, dict):
        raise ValueError("candidate_df is missing display_to_internal_id mapping.")
    if display_candidate_id not in mapping:
        raise KeyError(f"Unknown display candidate ID: {display_candidate_id!r}")
    return mapping[display_candidate_id]
