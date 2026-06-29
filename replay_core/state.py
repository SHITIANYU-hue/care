from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from replay_core.schema import ReplayTables


def sample_initial_candidate_ids(
    candidate_table: pd.DataFrame,
    *,
    id_column: str = "candidate_id",
    count: int,
    seed: int,
) -> list[object]:
    ordered = candidate_table.sort_values(id_column).reset_index(drop=True)
    if count <= 0:
        return []
    if count >= len(ordered):
        return ordered[id_column].tolist()
    rng = np.random.default_rng(seed)
    choices = rng.choice(ordered[id_column].to_numpy(), size=count, replace=False)
    return sorted(choices.tolist())


@dataclass
class ReplayState:
    """Mutable replay state with observed inputs separated from revealed outcomes."""

    id_column: str
    target_columns: tuple[str, ...]
    hidden_outcome_columns: tuple[str, ...]
    observed_candidates: pd.DataFrame
    observed_outcomes: pd.DataFrame
    remaining_candidates: pd.DataFrame
    selected_history: list[list[object]] = field(default_factory=list)
    round_index: int = 0

    @property
    def evaluator_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.target_columns, *self.hidden_outcome_columns)))

    @classmethod
    def initialize(
        cls,
        tables: ReplayTables,
        *,
        seed: int = 0,
        initial_observed_count: int = 0,
        initial_candidate_ids: Iterable[object] | None = None,
        initial_revealed_outcomes: pd.DataFrame | None = None,
    ) -> "ReplayState":
        ids = (
            list(initial_candidate_ids)
            if initial_candidate_ids is not None
            else sample_initial_candidate_ids(
                tables.candidate_table,
                id_column=tables.id_column,
                count=initial_observed_count,
                seed=seed,
            )
        )
        id_set = set(ids)
        missing_ids = id_set - set(tables.candidate_table[tables.id_column].tolist())
        if missing_ids:
            raise KeyError(f"Initial candidate IDs are not in candidate_table: {sorted(missing_ids)}")

        candidate_leaks = set(tables.evaluator_columns()).intersection(tables.candidate_table.columns)
        if candidate_leaks:
            raise ValueError(f"candidate_table contains evaluator columns: {sorted(candidate_leaks)}")

        observed_candidates = tables.candidate_table.loc[
            tables.candidate_table[tables.id_column].isin(id_set)
        ].copy()
        observed_candidates = _order_by_ids(observed_candidates, ids, tables.id_column)
        remaining_candidates = tables.candidate_table.loc[
            ~tables.candidate_table[tables.id_column].isin(id_set)
        ].copy().reset_index(drop=True)

        if ids:
            if initial_revealed_outcomes is None:
                raise ValueError("Initial observed candidates require evaluator-revealed outcome rows.")
            observed_outcomes = _validate_revealed_rows(
                initial_revealed_outcomes,
                id_column=tables.id_column,
                evaluator_columns=tables.evaluator_columns(),
                allowed_ids=set(ids),
            )
            observed_outcomes = _order_by_ids(observed_outcomes, ids, tables.id_column)
        else:
            observed_outcomes = pd.DataFrame(columns=[tables.id_column, *tables.evaluator_columns()])

        state = cls(
            id_column=tables.id_column,
            target_columns=tables.target_columns,
            hidden_outcome_columns=tables.hidden_outcome_columns,
            observed_candidates=observed_candidates.reset_index(drop=True),
            observed_outcomes=observed_outcomes.reset_index(drop=True),
            remaining_candidates=remaining_candidates.reset_index(drop=True),
            selected_history=[ids] if ids else [],
            round_index=0,
        )
        state._validate_no_remaining_leaks()
        return state

    def can_continue(self, max_rounds: int | None = None) -> bool:
        if max_rounds is not None and self.round_index >= max_rounds:
            return False
        return not self.remaining_candidates.empty

    def observed_view(self) -> pd.DataFrame:
        return self.observed_candidates.merge(
            self.observed_outcomes,
            on=self.id_column,
            how="left",
            validate="one_to_one",
            sort=False,
        )

    def best_observed(self, target_column: str, mode: str = "maximize") -> float:
        if target_column not in self.observed_outcomes.columns or self.observed_outcomes.empty:
            return float("nan")
        series = self.observed_outcomes[target_column].dropna()
        if series.empty:
            return float("nan")
        return float(series.max() if mode == "maximize" else series.min())

    def observe(self, revealed_rows: pd.DataFrame) -> None:
        revealed = _validate_revealed_rows(
            revealed_rows,
            id_column=self.id_column,
            evaluator_columns=self.evaluator_columns,
            allowed_ids=set(self.remaining_candidates[self.id_column].tolist()),
        )
        selected_ids = revealed[self.id_column].tolist()
        selected_candidates = self.remaining_candidates.loc[
            self.remaining_candidates[self.id_column].isin(selected_ids)
        ].copy()
        selected_candidates = _order_by_ids(selected_candidates, selected_ids, self.id_column)
        revealed = _order_by_ids(revealed, selected_ids, self.id_column)

        self.observed_candidates = pd.concat(
            [self.observed_candidates, selected_candidates],
            ignore_index=True,
        ).drop_duplicates(subset=[self.id_column], keep="first")
        self.observed_outcomes = pd.concat(
            [self.observed_outcomes, revealed],
            ignore_index=True,
        ).drop_duplicates(subset=[self.id_column], keep="first")
        self.remaining_candidates = self.remaining_candidates.loc[
            ~self.remaining_candidates[self.id_column].isin(selected_ids)
        ].copy().reset_index(drop=True)
        self.selected_history.append(list(selected_ids))
        self.round_index += 1
        self._validate_no_remaining_leaks()

    def _validate_no_remaining_leaks(self) -> None:
        leaks = set(self.evaluator_columns).intersection(self.remaining_candidates.columns)
        if leaks:
            raise AssertionError(
                "remaining_candidates contains evaluator-only columns: "
                + ", ".join(sorted(leaks))
            )


def _validate_revealed_rows(
    rows: pd.DataFrame,
    *,
    id_column: str,
    evaluator_columns: tuple[str, ...],
    allowed_ids: set[object],
) -> pd.DataFrame:
    if id_column not in rows.columns:
        raise ValueError(f"Revealed rows must contain {id_column!r}.")
    missing = [column for column in evaluator_columns if column not in rows.columns]
    if missing:
        raise ValueError(
            "Rows must be evaluator-revealed before observation; missing evaluator columns: "
            + ", ".join(missing)
        )
    unknown = set(rows[id_column].tolist()) - allowed_ids
    if unknown:
        raise ValueError(f"Rows contain unrevealed or unavailable candidate IDs: {sorted(unknown)}")
    if rows[id_column].duplicated().any():
        raise ValueError("Revealed rows contain duplicate candidate IDs.")
    return rows.loc[:, [id_column, *evaluator_columns]].copy().reset_index(drop=True)


def _order_by_ids(frame: pd.DataFrame, ids: Iterable[object], id_column: str) -> pd.DataFrame:
    id_list = list(ids)
    order = pd.DataFrame({id_column: id_list, "_order": range(len(id_list))})
    return (
        order.merge(frame, on=id_column, how="left", validate="one_to_one", sort=False)
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
