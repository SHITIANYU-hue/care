"""Compile typed policy skills into the existing rank_candidates contract."""

from __future__ import annotations

from research_tool_agent_full_pool.harness.specs import SkillArtifact
from research_tool_agent_full_pool.tool_contract import REQUIRED_ENTRYPOINT
from research_tool_agent_full_pool.tool_static_check import static_check_generated_tool_source


def compile_skill_to_tool(skill: SkillArtifact) -> str:
    """Return executable source for the current MVP skill artifact.

    The MVP deliberately keeps one external interface: every skill source must
    define `rank_candidates`. Later versions can compose several typed skills
    into this same interface without changing replay/evaluator contracts.
    """

    source = str(skill.source)
    if f"def {REQUIRED_ENTRYPOINT}" not in source:
        raise ValueError("Compiled skill must define rank_candidates.")
    report = static_check_generated_tool_source(source)
    if not report["passed"]:
        raise ValueError("Compiled skill failed static check: " + ",".join(report["violations"]))
    return source


def source_complexity_penalty(source: str) -> float:
    """Small public-safe penalty for very large generated tools."""

    lines = [line for line in str(source).splitlines() if line.strip()]
    if len(lines) <= 120:
        return 0.0
    return min(0.5, (len(lines) - 120) / 1000.0)
