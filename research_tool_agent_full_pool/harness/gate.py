"""Conservative deployment gate for candidate policy skills."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research_tool_agent_full_pool.harness.compiler import compile_skill_to_tool
from research_tool_agent_full_pool.harness.specs import GateReport, SkillArtifact
from research_tool_agent_full_pool.tool_replacement_verifier import verify_tool_replacement
from research_tool_agent_full_pool.tool_sandbox import validate_tool_in_sandbox
from research_tool_agent_full_pool.tool_static_check import static_check_generated_tool_source


def run_conservative_gate(
    *,
    candidate_skill: SkillArtifact,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    old_tool_source: str = "",
    memory: str | None = None,
    tool_state: dict[str, Any] | None = None,
    round_index: int | None = None,
) -> GateReport:
    """Validate a candidate skill before deployment."""

    checks: dict[str, Any] = {}
    failed: list[str] = []
    warnings: list[str] = []
    try:
        source = compile_skill_to_tool(candidate_skill)
        checks["compiler"] = {"passed": True, "source_hash": candidate_skill.source_hash}
    except Exception as exc:
        checks["compiler"] = {"passed": False, "error_type": exc.__class__.__name__, "error": str(exc)[:500]}
        failed.append("compiler")
        source = str(candidate_skill.source)

    static_report = static_check_generated_tool_source(source)
    checks["static"] = static_report
    if not static_report.get("passed"):
        failed.append("static")

    if not failed:
        sandbox_report = validate_tool_in_sandbox(
            tool_source=source,
            observed_df=observed_df,
            candidate_df=candidate_df,
            memory=memory,
            tool_state=tool_state or {},
        )
    else:
        sandbox_report = {"passed": False, "skipped": True, "reason": "previous_gate_failure"}
    checks["sandbox"] = sandbox_report
    if not sandbox_report.get("passed"):
        failed.append("sandbox")

    if not failed:
        verifier_report = verify_tool_replacement(
            old_tool_source=old_tool_source,
            patched_tool_source=source,
            observed_df=observed_df,
            candidate_df=candidate_df,
            memory=memory,
            tool_state=tool_state or {},
            round_index=round_index,
        )
    else:
        verifier_report = {"passed": False, "deployable": False, "skipped": True, "reason": "previous_gate_failure"}
    checks["replacement_verifier"] = verifier_report
    if not verifier_report.get("deployable"):
        failed.append("replacement_verifier")
    warnings.extend([str(item) for item in verifier_report.get("warning_checks", [])])

    unique_failed = sorted(set(failed))
    deployable = not unique_failed
    return GateReport(
        passed=deployable,
        deployable=deployable,
        failed_checks=unique_failed,
        warning_checks=sorted(set(warnings)),
        checks=checks,
        reason="passed_all_required_checks" if deployable else "failed:" + ",".join(unique_failed),
    )
