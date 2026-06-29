"""Shared initial-observation setup for full-pool replay policies."""

from __future__ import annotations

from typing import Any

from replay_core.evaluator import OfflineEvaluator
from replay_core.state import ReplayState, sample_initial_candidate_ids


def validate_initial_observed_count(
    initial_observed_count: Any,
    *,
    total_candidate_count: int | None = None,
) -> int:
    """Validate and return a replay initial-observation count."""

    if type(initial_observed_count) is not int:
        raise TypeError("initial_observed_count must be an int.")
    if initial_observed_count <= 0:
        raise ValueError("initial_observed_count must be positive.")
    if total_candidate_count is not None and initial_observed_count >= int(total_candidate_count):
        raise ValueError("initial_observed_count must be less than total candidate count.")
    return int(initial_observed_count)


def initial_candidate_ids_for_seed(
    tables: Any,
    *,
    seed: int,
    initial_observed_count: int,
) -> list[Any]:
    """Return the deterministic initial observations shared by all policies."""

    n0 = validate_initial_observed_count(
        initial_observed_count,
        total_candidate_count=len(tables.candidate_table),
    )
    return sample_initial_candidate_ids(
        tables.candidate_table,
        id_column=tables.id_column,
        count=n0,
        seed=int(seed),
    )


def initialize_full_pool_replay_state(
    *,
    tables: Any,
    evaluator: OfflineEvaluator,
    seed: int,
    initial_observed_count: int,
) -> ReplayState:
    """Initialize replay state with the shared full-pool n0 protocol."""

    initial_ids = initial_candidate_ids_for_seed(
        tables,
        seed=int(seed),
        initial_observed_count=initial_observed_count,
    )
    return ReplayState.initialize(
        tables,
        seed=int(seed),
        initial_observed_count=int(initial_observed_count),
        initial_candidate_ids=initial_ids,
        initial_revealed_outcomes=evaluator.reveal(initial_ids),
    )
