"""Sandbox validation for generated optimizer tools."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research_tool_agent_full_pool.artifact_logger import sanitize_text
from research_tool_agent_full_pool.tool_output_parser import parse_ranked_candidates
from research_tool_agent_full_pool.tool_runner import execute_rank_candidates_tool


def validate_tool_in_sandbox(
    *,
    tool_source: str,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    memory: str | None = None,
    tool_state: dict[str, Any] | None = None,
    max_candidate_rows: int = 16,
) -> dict[str, Any]:
    """Run deterministic in-process sandbox validation before trusting a tool.

    This Step 2 sandbox uses restricted imports/builtins and public-safe data
    slices. It does not provide full process isolation; that remains a Step 3
    hardening risk.
    """

    report: dict[str, Any] = {
        "passed": False,
        "candidate_rows_validated": 0,
        "tool_output_valid": False,
        "runner_passed": False,
        "error_type": None,
        "error": None,
    }
    try:
        fixture_candidate_df = candidate_df.head(max(1, min(max_candidate_rows, len(candidate_df)))).copy()
        fixture_observed_df = observed_df.copy()
        report["candidate_rows_validated"] = len(fixture_candidate_df)
        raw_output, runner_report = execute_rank_candidates_tool(
            tool_source=tool_source,
            observed_df=fixture_observed_df,
            candidate_df=fixture_candidate_df,
            memory=memory,
            tool_state=tool_state or {},
        )
        report["runner_passed"] = bool(runner_report.get("passed"))
        parse_ranked_candidates(
            raw_output,
            candidate_df=fixture_candidate_df,
            observed_df=fixture_observed_df,
        )
        report["tool_output_valid"] = True
        report["passed"] = True
    except Exception as exc:
        report["error_type"] = exc.__class__.__name__
        report["error"] = sanitize_text(str(exc))[:800]
    return report
