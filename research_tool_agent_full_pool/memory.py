"""Memory lifecycle skeletons for the full-pool ResearchToolAgent."""

from __future__ import annotations

from typing import Any


def initialize_memory(
    *,
    task_name: str,
    objective_name: str,
    observed_count: int,
    current_best_y: float | None,
    tool_name: str | None = None,
) -> str:
    """Create initial `memory.md` text from public run context only."""

    best = "NA" if current_best_y is None else f"{float(current_best_y):.6g}"
    tool = tool_name or "pending"
    return "\n".join(
        [
            f"# {task_name}",
            "",
            f"- objective: {objective_name}",
            f"- observed_points: {int(observed_count)}",
            "- last_selected_candidate: NA",
            "- last_revealed_y: NA",
            f"- current_best_observed_y: {best}",
            f"- current_tool_name: {tool}",
            "- strategy_note: Deterministic fake full-pool scoring from public features and revealed observations.",
            "",
        ]
    )


def update_memory_after_reveal(
    *,
    task_name: str,
    objective_name: str,
    observed_count: int,
    last_selected_candidate: str,
    last_revealed_y: float | None,
    current_best_y: float | None,
    tool_name: str | None,
    round_index: int,
    previous_memory: str = "",
    **_: Any,
) -> str:
    """Update memory after the selected candidate outcome has been revealed."""

    last_y = "NA" if last_revealed_y is None else f"{float(last_revealed_y):.6g}"
    best = "NA" if current_best_y is None else f"{float(current_best_y):.6g}"
    tool = tool_name or "unknown"
    prior_lines = [
        line
        for line in previous_memory.splitlines()
        if line.startswith("- round_") and "selected=" in line
    ]
    round_line = (
        f"- round_{int(round_index):03d}: selected={last_selected_candidate}, "
        f"revealed_y={last_y}, best_y={best}"
    )
    return "\n".join(
        [
            f"# {task_name}",
            "",
            f"- objective: {objective_name}",
            f"- observed_points: {int(observed_count)}",
            f"- last_selected_candidate: {last_selected_candidate}",
            f"- last_revealed_y: {last_y}",
            f"- current_best_observed_y: {best}",
            f"- current_tool_name: {tool}",
            "- strategy_note: Reuse the persistent fake scoring tool and rescore the full remaining pool each round.",
            "",
            "## Round Log",
            *prior_lines,
            round_line,
            "",
        ]
    )
