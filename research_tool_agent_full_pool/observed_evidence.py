"""Deterministic observed-only evidence markdown builder."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research_tool_agent_full_pool.safety import (
    FORBIDDEN_OBSERVED_FIELDS,
    find_forbidden_fields,
)


RESERVED_COLUMNS = {"observation_id", "candidate_id", "observed_y"}


def build_observed_evidence_markdown(
    observed_df: pd.DataFrame,
    round_index: int,
    objective_direction: str,
    max_examples: int = 12,
) -> str:
    """Render observed-only evidence as deterministic markdown."""

    _validate_observed_evidence_input(observed_df)
    direction = str(objective_direction).lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("objective_direction must be 'maximize' or 'minimize'.")

    observed = observed_df.copy(deep=True).reset_index(drop=True)
    observed["observed_y"] = pd.to_numeric(observed["observed_y"], errors="coerce")
    observed_count = int(len(observed))
    y = observed["observed_y"].dropna()
    best_row = _best_row(observed, direction)
    feature_columns = [
        str(column)
        for column in observed.columns
        if str(column) not in RESERVED_COLUMNS
        and not find_forbidden_fields([str(column)], FORBIDDEN_OBSERVED_FIELDS)
    ]

    lines: list[str] = [
        "# Observed Evidence",
        "",
        "Safe boundary: this file contains only revealed observations and summaries computed from revealed observations.",
        "",
        "## Run Metadata",
        "",
        f"- round_index: {int(round_index)}",
        f"- objective_direction: {direction}",
        f"- revealed_observation_count: {observed_count}",
        "",
        "## Summary Statistics",
        "",
    ]
    if y.empty:
        lines.extend(
            [
                "- observed_y_count: 0",
                "- observed_y_mean: NA",
                "- observed_y_std: NA",
                "- observed_y_min: NA",
                "- observed_y_median: NA",
                "- observed_y_max: NA",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- observed_y_count: {int(y.count())}",
                f"- observed_y_mean: {_format_number(y.mean())}",
                f"- observed_y_std: {_format_number(y.std())}",
                f"- observed_y_min: {_format_number(y.min())}",
                f"- observed_y_median: {_format_number(y.median())}",
                f"- observed_y_max: {_format_number(y.max())}",
                "",
            ]
        )

    lines.extend(["## Best Observed So Far", ""])
    if best_row is None:
        lines.append("- best observed so far: NA")
    else:
        lines.append(
            "- best observed so far: {candidate_id} ({observation_id}), observed_y={observed_y}".format(
                candidate_id=_cell(best_row["candidate_id"]),
                observation_id=_cell(best_row["observation_id"]),
                observed_y=_format_number(best_row["observed_y"]),
            )
        )
    lines.append("")

    lines.extend(["## Revealed Observations", ""])
    lines.extend(_table_for_rows(observed.head(max_examples), feature_columns))
    if observed_count > max_examples:
        lines.append(f"- additional_revealed_observations_not_shown: {observed_count - int(max_examples)}")
    lines.append("")

    lines.extend(["## Low-Y Observed Examples", ""])
    low_rows = observed.sort_values(["observed_y", "candidate_id"], ascending=[True, True]).head(max_examples)
    lines.extend(_compact_candidate_lines(low_rows))
    lines.append("")

    lines.extend(["## Recent Revealed Selections", ""])
    recent = observed.tail(max_examples)
    lines.extend(_compact_candidate_lines(recent))
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _validate_observed_evidence_input(observed_df: pd.DataFrame) -> None:
    required = RESERVED_COLUMNS
    missing = required - set(str(column) for column in observed_df.columns)
    if missing:
        raise ValueError(f"observed_df is missing required columns: {sorted(missing)}")
    forbidden = find_forbidden_fields(observed_df.columns, FORBIDDEN_OBSERVED_FIELDS)
    if forbidden:
        raise ValueError(f"observed_df contains forbidden columns: {sorted(forbidden)}")


def _best_row(observed: pd.DataFrame, direction: str) -> dict[str, Any] | None:
    valid = observed.dropna(subset=["observed_y"]).copy()
    if valid.empty:
        return None
    ascending = direction == "minimize"
    valid = valid.sort_values(["observed_y", "candidate_id"], ascending=[ascending, True])
    return dict(valid.iloc[0])


def _table_for_rows(rows: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    columns = ["observation_id", "candidate_id", "observed_y", *feature_columns[:8]]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in rows.iterrows():
        values = [_cell(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) == 0:
        lines.append("| NA | NA | NA |" + (" NA |" * len(feature_columns[:8])))
    return lines


def _compact_candidate_lines(rows: pd.DataFrame) -> list[str]:
    if rows.empty:
        return ["- NA"]
    lines: list[str] = []
    for _, row in rows.iterrows():
        lines.append(
            "- {observation_id}: candidate_id={candidate_id}, observed_y={observed_y}".format(
                observation_id=_cell(row.get("observation_id", "")),
                candidate_id=_cell(row.get("candidate_id", "")),
                observed_y=_format_number(row.get("observed_y")),
            )
        )
    return lines


def _cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    text = _format_number(value) if isinstance(value, (int, float)) else str(value)
    return text.replace("|", "/").replace("\n", " ").strip()


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return "NA"
    return f"{number:.6g}"
