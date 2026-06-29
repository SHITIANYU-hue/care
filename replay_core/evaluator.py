from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from replay_core.schema import ReplayTables


class OfflineEvaluator:
    """Evaluator-side outcome revealer for finite offline replay.

    This object owns the hidden outcome table. Decision-facing modules receive
    only `reveal(candidate_ids)` outputs for already selected candidate IDs.
    """

    def __init__(
        self,
        outcome_table: pd.DataFrame,
        *,
        id_column: str = "candidate_id",
        evaluator_columns: Iterable[str] | None = None,
    ) -> None:
        if id_column not in outcome_table.columns:
            raise ValueError(f"outcome_table must contain {id_column!r}.")
        if outcome_table[id_column].duplicated().any():
            raise ValueError(f"outcome_table contains duplicate {id_column!r} values.")

        self.id_column = id_column
        self._evaluator_columns = tuple(
            evaluator_columns
            if evaluator_columns is not None
            else (column for column in outcome_table.columns if column != id_column)
        )
        missing = [column for column in self._evaluator_columns if column not in outcome_table.columns]
        if missing:
            raise ValueError(f"outcome_table is missing evaluator columns: {missing}")
        self._outcome_table = outcome_table.loc[:, [self.id_column, *self._evaluator_columns]].copy()

    @classmethod
    def from_tables(cls, tables: ReplayTables) -> "OfflineEvaluator":
        return cls(
            tables.outcome_table,
            id_column=tables.id_column,
            evaluator_columns=tables.evaluator_columns(),
        )

    @property
    def evaluator_columns(self) -> tuple[str, ...]:
        return self._evaluator_columns

    def reveal(self, candidate_ids: Iterable[object]) -> pd.DataFrame:
        selected_ids = list(candidate_ids)
        columns = [self.id_column, *self._evaluator_columns]
        if not selected_ids:
            return pd.DataFrame(columns=columns)

        requested = pd.DataFrame({self.id_column: selected_ids})
        revealed = requested.merge(
            self._outcome_table,
            on=self.id_column,
            how="left",
            validate="many_to_one",
            sort=False,
        )
        missing = revealed.loc[
            revealed.loc[:, list(self._evaluator_columns)].isna().all(axis=1),
            self.id_column,
        ].tolist()
        if missing:
            raise KeyError(f"No outcomes available for selected candidate IDs: {missing}")
        return revealed.loc[:, columns].copy()

    def reveal_for_frame(self, selected_frame: pd.DataFrame) -> pd.DataFrame:
        if self.id_column not in selected_frame.columns:
            raise ValueError(f"selected_frame must contain {self.id_column!r}.")
        visible = selected_frame.drop(columns=list(self._evaluator_columns), errors="ignore").copy()
        return visible.merge(
            self.reveal(visible[self.id_column].tolist()),
            on=self.id_column,
            how="left",
            validate="one_to_one",
            sort=False,
        )

