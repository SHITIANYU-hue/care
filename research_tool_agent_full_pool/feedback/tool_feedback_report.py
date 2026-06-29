"""Observed-safe generated-tool feedback reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from research_tool_agent_full_pool.decision_artifacts import assert_payload_public_safe


HARNESS_VERSION = "batch1_safety_artifacts_v1"


@dataclass(frozen=True)
class ToolFeedbackReport:
    """Observed-only feedback available after the selected candidate is revealed."""

    round_id: int
    selected_candidate_id: str
    selected_candidate_public_features: dict[str, Any]
    tool_score: float | None
    tool_rank: int | None
    revealed_y: float | None
    best_y_before: float | None
    best_y_after: float | None
    improved_best: bool | None
    selected_y_vs_observed_median: float | None
    observed_count: int
    observed_y_min: float | None
    observed_y_median: float | None
    observed_y_max: float | None
    score_distribution_summary: dict[str, Any]
    rank1_margin: float | None
    fallback_used: bool
    parser_status: str
    static_check_status: str
    sandbox_status: str
    public_tool_diagnostics: dict[str, Any]
    created_by_harness_version: str = HARNESS_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_tool_feedback_report(
    *,
    round_id: int,
    selected_candidate_id: str,
    selected_candidate_public_features: dict[str, Any] | None,
    revealed_y: float | None,
    observed_df: pd.DataFrame,
    decision_metadata: dict[str, Any] | None = None,
    tool_diagnostics: dict[str, Any] | None = None,
    parser_status: str = "not_available",
    static_check_status: str = "not_available",
    sandbox_status: str = "not_available",
    objective_direction: str = "maximize",
) -> ToolFeedbackReport:
    """Build a report without hidden outcomes or posthoc/ranking artifacts."""

    metadata = dict(decision_metadata or {})
    observed_y = (
        pd.to_numeric(observed_df.get("observed_y", pd.Series(dtype=float)), errors="coerce").dropna()
        if isinstance(observed_df, pd.DataFrame)
        else pd.Series(dtype=float)
    )
    selected_y = _safe_float(revealed_y)
    prior_y = _observed_values_excluding_selected(
        observed_df=observed_df,
        selected_candidate_id=selected_candidate_id,
    )
    best_before = _best(prior_y, objective_direction)
    best_after = _best(observed_y, objective_direction)
    median_after = _safe_float(observed_y.median()) if len(observed_y) else None
    selected_delta = None
    if selected_y is not None and median_after is not None:
        selected_delta = selected_y - median_after
    report = ToolFeedbackReport(
        round_id=int(round_id),
        selected_candidate_id=str(selected_candidate_id),
        selected_candidate_public_features=dict(selected_candidate_public_features or {}),
        tool_score=_safe_float(metadata.get("selected_tool_score", metadata.get("tool_score"))),
        tool_rank=_safe_int(metadata.get("selected_tool_rank", metadata.get("tool_rank"))),
        revealed_y=selected_y,
        best_y_before=best_before,
        best_y_after=best_after,
        improved_best=_improved(best_before, best_after, objective_direction),
        selected_y_vs_observed_median=selected_delta,
        observed_count=int(len(observed_df)) if isinstance(observed_df, pd.DataFrame) else 0,
        observed_y_min=_safe_float(observed_y.min()) if len(observed_y) else None,
        observed_y_median=median_after,
        observed_y_max=_safe_float(observed_y.max()) if len(observed_y) else None,
        score_distribution_summary=dict(metadata.get("score_distribution_summary") or {}),
        rank1_margin=_safe_float(metadata.get("rank1_score_margin")),
        fallback_used=bool(metadata.get("fallback_used", False)),
        parser_status=str(parser_status),
        static_check_status=str(static_check_status),
        sandbox_status=str(sandbox_status),
        public_tool_diagnostics=dict(tool_diagnostics or {}),
    )
    assert_payload_public_safe(report.to_dict(), label="tool_feedback_report")
    return report


def selected_public_features_from_observed_df(
    observed_df: pd.DataFrame,
    selected_candidate_id: str,
) -> dict[str, Any]:
    """Extract selected candidate public features from rebuilt observed_df."""

    if not isinstance(observed_df, pd.DataFrame) or observed_df.empty:
        return {"candidate_id": str(selected_candidate_id)}
    rows = observed_df.loc[observed_df["candidate_id"].astype(str) == str(selected_candidate_id)]
    if rows.empty:
        return {"candidate_id": str(selected_candidate_id)}
    record = rows.iloc[0].to_dict()
    return {
        str(key): _json_safe(value)
        for key, value in record.items()
        if str(key) != "observed_y"
    }


def write_tool_feedback_report(output_dir: str | Path, report: ToolFeedbackReport) -> None:
    """Write latest and append-only feedback report artifacts."""

    payload = report.to_dict()
    assert_payload_public_safe(payload, label="tool_feedback_report")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / "tool_feedback_report.json", payload)
    _append_jsonl(output_path / "tool_feedback_reports.jsonl", payload)


def _observed_values_excluding_selected(
    *,
    observed_df: pd.DataFrame,
    selected_candidate_id: str,
) -> pd.Series:
    if not isinstance(observed_df, pd.DataFrame) or "observed_y" not in observed_df.columns:
        return pd.Series(dtype=float)
    if "candidate_id" not in observed_df.columns:
        return pd.to_numeric(observed_df["observed_y"], errors="coerce").dropna()
    prior = observed_df.loc[observed_df["candidate_id"].astype(str) != str(selected_candidate_id)]
    return pd.to_numeric(prior["observed_y"], errors="coerce").dropna()


def _best(values: pd.Series, objective_direction: str) -> float | None:
    if len(values) == 0:
        return None
    if str(objective_direction).lower() == "minimize":
        return _safe_float(values.min())
    return _safe_float(values.max())


def _improved(
    best_before: float | None,
    best_after: float | None,
    objective_direction: str,
) -> bool | None:
    if best_after is None:
        return None
    if best_before is None:
        return True
    if str(objective_direction).lower() == "minimize":
        return best_after < best_before
    return best_after > best_before


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

