from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


ID_COLUMN = "candidate_id"

FORBIDDEN_DECISION_COLUMN_NAMES = {
    "yield",
    "conversion",
    "target",
    "outcome",
    "label",
    "turnover",
    "global_rank",
    "rank",
    "oracle_rank",
    "oracle_score",
    "top_k_label",
    "topk_label",
    "bad_outcome_flag",
    "failure_avoidance_score",
}


def is_forbidden_decision_column(
    column: object,
    *,
    target_columns: Iterable[object] = (),
    hidden_outcome_columns: Iterable[object] = (),
    evaluator_columns: Iterable[object] = (),
) -> bool:
    normalized = str(column).lower()
    configured_forbidden = {
        str(item).lower()
        for item in (*tuple(target_columns), *tuple(hidden_outcome_columns), *tuple(evaluator_columns))
    }
    return (
        normalized in FORBIDDEN_DECISION_COLUMN_NAMES
        or normalized in configured_forbidden
        or normalized.endswith("_target")
        or normalized.endswith("_outcome")
        or normalized.endswith("_outcome_flag")
        or normalized.endswith("_label")
        or normalized.endswith("_rank")
        or normalized.startswith("oracle_")
    )


def forbidden_decision_columns(
    columns: Iterable[object],
    *,
    target_columns: Iterable[object] = (),
    hidden_outcome_columns: Iterable[object] = (),
    evaluator_columns: Iterable[object] = (),
) -> tuple[str, ...]:
    return tuple(
        str(column)
        for column in columns
        if is_forbidden_decision_column(
            column,
            target_columns=target_columns,
            hidden_outcome_columns=hidden_outcome_columns,
            evaluator_columns=evaluator_columns,
        )
    )


def _copy_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None:
        return None
    return frame.copy().reset_index(drop=True)


def _validate_id_column(frame: pd.DataFrame | None, *, id_column: str, label: str) -> None:
    if frame is None:
        return
    if id_column not in frame.columns:
        raise ValueError(f"{label} must contain {id_column!r}.")
    if frame[id_column].duplicated().any():
        duplicates = frame.loc[frame[id_column].duplicated(), id_column].astype(str).tolist()
        raise ValueError(f"{label} contains duplicate {id_column!r} values: {duplicates[:5]}")


def _merge_one_to_one(
    left: pd.DataFrame,
    right: pd.DataFrame | None,
    *,
    id_column: str,
    label: str,
) -> pd.DataFrame:
    if right is None:
        return left
    overlap = [column for column in right.columns if column != id_column and column in left.columns]
    if overlap:
        raise ValueError(
            f"{label} overlaps existing columns during reporting merge: {sorted(overlap)}"
        )
    return left.merge(right, on=id_column, how="left", validate="one_to_one")


@dataclass
class ReplayTables:
    """Finite-pool replay tables with an explicit evaluator boundary.

    `candidate_table` is decision-facing and contains candidate IDs plus visible
    inputs. `outcome_table` is evaluator-only and contains objective targets plus
    hidden outcomes. Metadata/group/flag/cost tables are optional visible-side
    tables, but decision views still defensively remove target-like columns.
    """

    candidate_table: pd.DataFrame
    outcome_table: pd.DataFrame
    id_column: str = ID_COLUMN
    target_columns: tuple[str, ...] = ()
    hidden_outcome_columns: tuple[str, ...] = ()
    metadata_table: pd.DataFrame | None = None
    group_table: pd.DataFrame | None = None
    flag_table: pd.DataFrame | None = None
    cost_table: pd.DataFrame | None = None
    decision_columns: tuple[str, ...] | None = None
    dataset_name: str = ""
    dataset_identity: str = ""
    source_path: Path | str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.candidate_table = _copy_frame(self.candidate_table)  # type: ignore[assignment]
        self.outcome_table = _copy_frame(self.outcome_table)  # type: ignore[assignment]
        self.metadata_table = _copy_frame(self.metadata_table)
        self.group_table = _copy_frame(self.group_table)
        self.flag_table = _copy_frame(self.flag_table)
        self.cost_table = _copy_frame(self.cost_table)
        if self.source_path is not None:
            self.source_path = Path(self.source_path)

        for label, table in (
            ("candidate_table", self.candidate_table),
            ("outcome_table", self.outcome_table),
            ("metadata_table", self.metadata_table),
            ("group_table", self.group_table),
            ("flag_table", self.flag_table),
            ("cost_table", self.cost_table),
        ):
            _validate_id_column(table, id_column=self.id_column, label=label)

        if not self.target_columns:
            raise ValueError("ReplayTables requires at least one target column.")
        missing_targets = set(self.target_columns) - set(self.outcome_table.columns)
        if missing_targets:
            raise ValueError(f"outcome_table is missing target columns: {sorted(missing_targets)}")
        missing_hidden = set(self.hidden_outcome_columns) - set(self.outcome_table.columns)
        if missing_hidden:
            raise ValueError(f"outcome_table is missing hidden outcome columns: {sorted(missing_hidden)}")
        overlap = set(self.target_columns).intersection(self.hidden_outcome_columns)
        if overlap:
            raise ValueError(f"Columns cannot be both target and hidden outcome columns: {sorted(overlap)}")

        evaluator_columns = set(self.evaluator_columns())
        for label, table in (
            ("candidate_table", self.candidate_table),
            ("metadata_table", self.metadata_table),
            ("group_table", self.group_table),
            ("flag_table", self.flag_table),
            ("cost_table", self.cost_table),
        ):
            if table is None:
                continue
            leaks = forbidden_decision_columns(
                (column for column in table.columns if column != self.id_column),
                target_columns=self.target_columns,
                hidden_outcome_columns=self.hidden_outcome_columns,
                evaluator_columns=evaluator_columns,
            )
            if leaks:
                raise ValueError(
                    f"{label} must not contain forbidden decision-facing columns: "
                    + ", ".join(sorted(leaks))
                )

        if self.decision_columns is None:
            self.decision_columns = tuple(
                column for column in self.candidate_table.columns if column != self.id_column
            )
        missing_decision = set(self.decision_columns) - set(self.candidate_table.columns)
        if missing_decision:
            raise ValueError(f"candidate_table is missing decision columns: {sorted(missing_decision)}")
        if self.id_column in self.decision_columns:
            raise ValueError("decision_columns must not include the id column.")
        decision_leaks = forbidden_decision_columns(
            self.decision_columns,
            target_columns=self.target_columns,
            hidden_outcome_columns=self.hidden_outcome_columns,
            evaluator_columns=evaluator_columns,
        )
        if decision_leaks:
            raise ValueError(
                "decision_columns must not contain forbidden columns: "
                + ", ".join(sorted(decision_leaks))
            )

    def evaluator_columns(self) -> tuple[str, ...]:
        """Columns owned by the offline evaluator and revealed only after selection."""

        return tuple(dict.fromkeys((*self.target_columns, *self.hidden_outcome_columns)))

    def visible_side_tables(self) -> tuple[pd.DataFrame | None, ...]:
        return (self.metadata_table, self.group_table, self.flag_table, self.cost_table)

    def evaluator_joined_table(self) -> pd.DataFrame:
        """Reporting-only joined table.

        This helper intentionally carries an evaluator/reporting name because it
        includes outcome columns and must not be used by decision modules.
        """

        frame = self.candidate_table.copy()
        for label, table in (
            ("outcome_table", self.outcome_table),
            ("metadata_table", self.metadata_table),
            ("group_table", self.group_table),
            ("flag_table", self.flag_table),
            ("cost_table", self.cost_table),
        ):
            frame = _merge_one_to_one(frame, table, id_column=self.id_column, label=label)
        return frame

    def reporting_joined_table(self) -> pd.DataFrame:
        return self.evaluator_joined_table()

    def subset(self, candidate_ids: Iterable[object]) -> "ReplayTables":
        keep = set(candidate_ids)

        def subset_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
            if frame is None:
                return None
            return frame.loc[frame[self.id_column].isin(keep)].copy().reset_index(drop=True)

        return ReplayTables(
            candidate_table=subset_frame(self.candidate_table),  # type: ignore[arg-type]
            outcome_table=subset_frame(self.outcome_table),  # type: ignore[arg-type]
            id_column=self.id_column,
            target_columns=self.target_columns,
            hidden_outcome_columns=self.hidden_outcome_columns,
            metadata_table=subset_frame(self.metadata_table),
            group_table=subset_frame(self.group_table),
            flag_table=subset_frame(self.flag_table),
            cost_table=subset_frame(self.cost_table),
            decision_columns=self.decision_columns,
            dataset_name=self.dataset_name,
            dataset_identity=self.dataset_identity,
            source_path=self.source_path,
            notes=self.notes,
        )
