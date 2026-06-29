"""Parser for generated optimizer tool outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd

from research_tool_agent_full_pool.decision_artifacts import assert_payload_public_safe
from research_tool_agent_full_pool.safety import FORBIDDEN_SECRET_TERMS
from research_tool_agent_full_pool.tool_contract import FORBIDDEN_TERMS


@dataclass(frozen=True)
class ParsedToolOutput:
    """Validated generated-tool ranking output."""

    ranked_candidates: list[dict[str, Any]]
    selected_display_candidate_id: str
    tool_state: dict[str, Any]
    tool_diagnostics: dict[str, Any]


def parse_ranked_candidates(
    output: Any,
    *,
    candidate_df: pd.DataFrame,
    observed_df: pd.DataFrame,
) -> ParsedToolOutput:
    """Parse and validate rank-candidate tool output.

    Expected output is one row per remaining candidate with candidate id, score,
    and rank. Step 1 will require a unique rank 1 and full-pool coverage.
    """

    if not isinstance(output, dict):
        raise ValueError("Tool output must be a dictionary.")
    rows = output.get("ranked_candidates")
    if not isinstance(rows, list):
        raise ValueError("Tool output must contain ranked_candidates list.")
    if not rows:
        raise ValueError("Tool output ranked_candidates must be non-empty.")

    valid_candidate_ids = {str(value) for value in candidate_df["candidate_id"].tolist()}
    observed_refs = {str(value) for value in observed_df.get("observation_id", pd.Series(dtype=str)).tolist()}
    seen_candidate_ids: set[str] = set()
    seen_ranks: set[int] = set()
    parsed_rows: list[dict[str, Any]] = []
    rank1_ids: list[str] = []

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each ranked candidate must be a dictionary.")
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id not in valid_candidate_ids:
            raise ValueError("Tool output includes candidate outside the full remaining pool.")
        if candidate_id in seen_candidate_ids:
            raise ValueError("Tool output includes duplicate candidate IDs.")
        seen_candidate_ids.add(candidate_id)

        rank = int(row.get("rank"))
        if rank <= 0:
            raise ValueError("Tool output ranks must be positive integers.")
        if rank in seen_ranks:
            raise ValueError("Tool output includes duplicate ranks.")
        seen_ranks.add(rank)
        if rank == 1:
            rank1_ids.append(candidate_id)

        score = float(row.get("score"))
        if not math.isfinite(score):
            raise ValueError("Tool output scores must be finite numeric values.")

        reason_code = str(row.get("reason_code", ""))
        if not reason_code:
            raise ValueError("Tool output reason_code must be non-empty.")
        if len(reason_code) > 120:
            raise ValueError("Tool output reason_code must be short.")
        _validate_text_is_public_safe(reason_code, label="reason_code")

        evidence_refs = row.get("evidence_refs", [])
        if evidence_refs is None:
            evidence_refs = []
        if not isinstance(evidence_refs, list):
            raise ValueError("Tool output evidence_refs must be a list.")
        refs = [str(ref) for ref in evidence_refs]
        outside_refs = [ref for ref in refs if ref not in observed_refs]
        if outside_refs:
            raise ValueError("Tool evidence_refs must refer only to observed aliases.")

        parsed_rows.append(
            {
                "candidate_id": candidate_id,
                "rank": rank,
                "score": score,
                "reason_code": reason_code,
                "evidence_refs": refs,
            }
        )

    missing = valid_candidate_ids - seen_candidate_ids
    if missing:
        raise ValueError("Tool output must score every row in the full remaining candidate_df.")
    expected_ranks = set(range(1, len(valid_candidate_ids) + 1))
    if seen_ranks != expected_ranks:
        raise ValueError("Tool output ranks must be exactly the contiguous range 1..N.")
    if len(rank1_ids) != 1:
        raise ValueError("Tool output must contain exactly one rank-1 candidate.")

    parsed_rows.sort(key=lambda item: (int(item["rank"]), str(item["candidate_id"])))
    tool_state = output.get("tool_state", {})
    diagnostics = output.get("tool_diagnostics", {})
    if not isinstance(tool_state, dict):
        raise ValueError("tool_state must be a dictionary when present.")
    if not isinstance(diagnostics, dict):
        raise ValueError("tool_diagnostics must be a dictionary when present.")
    assert_payload_public_safe(tool_state, label="tool_state")
    assert_payload_public_safe(diagnostics, label="tool_diagnostics")
    return ParsedToolOutput(
        ranked_candidates=parsed_rows,
        selected_display_candidate_id=rank1_ids[0],
        tool_state=dict(tool_state),
        tool_diagnostics=dict(diagnostics),
    )


def _validate_text_is_public_safe(text: str, *, label: str) -> None:
    lowered = text.lower()
    terms = tuple(term.lower() for term in (*FORBIDDEN_TERMS, *FORBIDDEN_SECRET_TERMS))
    if any(term in lowered for term in terms):
        raise ValueError(f"{label} contains forbidden contract terms.")
