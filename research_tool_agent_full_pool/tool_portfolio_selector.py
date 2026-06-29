"""Safety filtering, observed-only scoring, and deployment for portfolio tools."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd

from research_tool_agent_full_pool import artifact_logger
from research_tool_agent_full_pool.diagnostics import classify_generated_tool_family, sha256_text
from research_tool_agent_full_pool.observed_tool_quality_evaluator import evaluate_candidate_tools_observed_only
from research_tool_agent_full_pool.tool_mechanism_classifier import classify_tool_mechanism
from research_tool_agent_full_pool.tool_candidate_repair import (
    repair_failed_candidate_tool,
    write_repair_verifier_artifacts,
)
from research_tool_agent_full_pool.tool_contract import ALLOWED_IMPORTS
from research_tool_agent_full_pool.tool_portfolio_artifacts import (
    append_candidate_verifier_report,
    append_observed_quality_score,
    write_tool_mechanism_classifications,
    write_candidate_manifest,
    write_prompt_pipeline_audit,
    write_portfolio_summary_markdown,
    write_selected_portfolio_tool,
)
from research_tool_agent_full_pool.tool_replacement_verifier import verify_tool_replacement


class PortfolioSelectionError(ValueError):
    """Raised when no candidate tool can be safely deployed."""


def select_portfolio_tool(
    *,
    candidate_tools: list[dict[str, Any]],
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    method_primitives: list[dict[str, Any]],
    output_dir: str,
    objective_direction: str = "maximize",
    round_index: int = 1,
    old_tool_source: str = "",
    client: Any | None = None,
    portfolio_repair_enabled: bool = False,
    max_repair_attempts_per_candidate: int = 1,
    run_id: str = "",
) -> dict[str, Any]:
    """Filter candidate tools, score valid tools, and deploy exactly one."""

    verifier_results: list[dict[str, Any]] = []
    repair_records: list[dict[str, Any]] = []
    all_candidate_versions: list[dict[str, Any]] = [_with_candidate_defaults(candidate) for candidate in candidate_tools]
    valid_tools: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in all_candidate_versions[:]:
        report = _verify_candidate_tool(
            candidate,
            observed_df=observed_df,
            candidate_df=candidate_df,
            round_index=round_index,
            old_tool_source=old_tool_source,
        )
        verifier_results.append(report)
        report["mechanism_classification"] = _classify_candidate_tool(
            candidate,
            verifier_result=report,
        )
        append_candidate_verifier_report(output_dir, report)
        if report["passed"]:
            valid_tools.append(candidate)
        else:
            rejected.append(
                {
                    "tool_id": candidate.get("tool_id"),
                    "parent_tool_id": candidate.get("parent_tool_id"),
                    "candidate_version": candidate.get("candidate_version", "original"),
                    "design_id": candidate.get("design_id"),
                    "rejection_reasons": report.get("failed_checks", []),
                }
            )
            if (
                bool(portfolio_repair_enabled)
                and client is not None
                and int(max_repair_attempts_per_candidate) >= 1
                and str(candidate.get("candidate_version", "original")) == "original"
            ):
                repair_result = repair_failed_candidate_tool(
                    failed_tool_source=str(candidate.get("source", "")),
                    design_metadata=candidate.get("design", {}) if isinstance(candidate.get("design"), dict) else {},
                    verifier_result=report,
                    observed_schema_summary=_schema_summary(observed_df),
                    candidate_schema_summary=_schema_summary(candidate_df),
                    tool_contract=_tool_contract_summary(),
                    allowed_imports=list(ALLOWED_IMPORTS),
                    forbidden_boundary=None,
                    output_dir=output_dir,
                    client=client,
                    original_tool_id=str(candidate.get("tool_id")),
                    run_id=str(run_id),
                    round_index=round_index,
                    repair_attempt=1,
                )
                repair_record = repair_result.to_record()
                repaired_tool = repair_result.repaired_tool
                if repaired_tool is not None:
                    all_candidate_versions.append(repaired_tool)
                    repair_report = _verify_candidate_tool(
                        repaired_tool,
                        observed_df=observed_df,
                        candidate_df=candidate_df,
                        round_index=round_index,
                        old_tool_source=old_tool_source,
                    )
                    verifier_results.append(repair_report)
                    repair_report["mechanism_classification"] = _classify_candidate_tool(
                        repaired_tool,
                        verifier_result=repair_report,
                    )
                    append_candidate_verifier_report(output_dir, repair_report)
                    repair_record = write_repair_verifier_artifacts(
                        output_dir=output_dir,
                        repair_result=repair_result,
                        verifier_report=repair_report,
                        accepted_for_quality_evaluation=bool(repair_report.get("passed")),
                    )
                    if repair_report["passed"]:
                        valid_tools.append(repaired_tool)
                    else:
                        rejected.append(
                            {
                                "tool_id": repaired_tool.get("tool_id"),
                                "parent_tool_id": repaired_tool.get("parent_tool_id"),
                                "candidate_version": repaired_tool.get("candidate_version", "repair"),
                                "design_id": repaired_tool.get("design_id"),
                                "rejection_reasons": repair_report.get("failed_checks", []),
                            }
                        )
                repair_records.append(repair_record)

    if not valid_tools:
        mechanism_rows = _build_mechanism_classification_rows(
            candidate_tools=all_candidate_versions,
            verifier_results=verifier_results,
            quality_scores=[],
        )
        write_tool_mechanism_classifications(output_dir, mechanism_rows)
        failure = {
            "selected_tool_id": None,
            "reason": "no_verifier_passing_candidate_tools",
            "all_candidate_verifier_results": verifier_results,
            "repair_enabled": bool(portfolio_repair_enabled),
            "repair_attempt_count": len([row for row in repair_records if row.get("attempted")]),
            "repair_records": repair_records,
            "rejected_tools": rejected,
            "true_y_reveals": 0,
        }
        _write_augmented_candidate_manifest(
            output_dir=output_dir,
            candidate_tools=all_candidate_versions,
            verifier_results=verifier_results,
            quality_scores=[],
            repair_records=repair_records,
            selected_tool_id=None,
            round_index=round_index,
            run_id=run_id,
            mechanism_classifications=mechanism_rows,
        )
        write_prompt_pipeline_audit(
            output_dir,
            {
                "selection": {
                    "selected_tool_id": None,
                    "reason": failure["reason"],
                    "repair_enabled": bool(portfolio_repair_enabled),
                    "true_y_reveals": 0,
                }
            },
        )
        write_selected_portfolio_tool(output_dir, failure)
        write_portfolio_summary_markdown(output_dir, _portfolio_summary_markdown(failure, [], repair_records, mechanism_rows))
        raise PortfolioSelectionError("No portfolio candidate tool passed safety verification; no reveal should occur.")

    quality_scores = evaluate_candidate_tools_observed_only(
        candidate_tools=valid_tools,
        observed_df=observed_df,
        candidate_df=candidate_df,
        method_primitives=method_primitives,
        objective_direction=objective_direction,
    )
    tool_by_id = {str(tool.get("tool_id")): tool for tool in all_candidate_versions}
    verifier_by_tool = {str(row.get("tool_id")): row for row in verifier_results}
    for score in quality_scores:
        tool = tool_by_id.get(str(score.get("tool_id")), {})
        score["mechanism_classification"] = _classify_candidate_tool(
            tool,
            verifier_result=verifier_by_tool.get(str(score.get("tool_id")), {}),
            observed_quality=score,
            score_summary=_quality_score_summary(score),
        )
        append_observed_quality_score(output_dir, score)

    selected_score = quality_scores[0]
    selected_tool = next(tool for tool in valid_tools if str(tool.get("tool_id")) == str(selected_score["tool_id"]))
    selected_source = str(selected_tool["source"])
    deployed_path = artifact_logger.write_generated_tool_source(
        output_dir,
        round_index=round_index,
        tool_name=str(selected_tool["tool_id"]),
        source=selected_source,
    )
    mechanism_rows = _build_mechanism_classification_rows(
        candidate_tools=all_candidate_versions,
        verifier_results=verifier_results,
        quality_scores=quality_scores,
    )
    write_tool_mechanism_classifications(output_dir, mechanism_rows)
    mechanism_by_tool = {str(row.get("tool_id")): row for row in mechanism_rows}
    selected_mechanism = mechanism_by_tool.get(str(selected_tool["tool_id"]), {})
    selector_output = {
        "selected_tool_id": selected_tool["tool_id"],
        "selected_parent_tool_id": selected_tool.get("parent_tool_id"),
        "selected_candidate_version": selected_tool.get("candidate_version", "original"),
        "selected_repair_attempt": int(selected_tool.get("repair_attempt", 0) or 0),
        "selected_design_id": selected_tool["design_id"],
        "selected_tool_family": selected_tool.get("tool_family") or classify_generated_tool_family(selected_source),
        "selected_reason": (
            "highest observed-only quality score among verifier-passing portfolio candidates"
        ),
        "all_candidate_verifier_results": verifier_results,
        "all_observed_only_quality_scores": quality_scores,
        "repair_enabled": bool(portfolio_repair_enabled),
        "repair_attempt_count": len([row for row in repair_records if row.get("attempted")]),
        "repair_records": repair_records,
        "tools_entering_quality_evaluator": [
            {
                "tool_id": tool.get("tool_id"),
                "parent_tool_id": tool.get("parent_tool_id"),
                "candidate_version": tool.get("candidate_version", "original"),
                "design_id": tool.get("design_id"),
            }
            for tool in valid_tools
        ],
        "rejected_tools": rejected,
        "deployed_tool_path": str(deployed_path),
        "deployed_tool_source_hash": sha256_text(selected_source),
        "selected_quality_score": selected_score["quality_score"],
        "selected_tool_mechanism_classification": selected_mechanism,
        "selected_static_self_audit": selected_tool.get("static_self_audit", {}),
        "selected_design_state_variables": selected_tool.get("design", {}).get("state_variables", []),
        "selected_exploitation_term": selected_tool.get("design", {}).get("exploitation_term"),
        "selected_exploration_or_anti_collapse_term": selected_tool.get("design", {}).get(
            "exploration_or_anti_collapse_term"
        ),
        "selected_small_n_fallback": selected_tool.get("design", {}).get("small_n_fallback"),
        "selected_mixed_variable_handling": selected_tool.get("design", {}).get("mixed_variable_handling"),
        "selected_why_this_is_not_a_static_ranker": selected_tool.get("design", {}).get(
            "why_this_is_not_a_static_ranker"
        ),
        "true_y_reveals_during_selection": 0,
    }
    _write_augmented_candidate_manifest(
        output_dir=output_dir,
        candidate_tools=all_candidate_versions,
        verifier_results=verifier_results,
        quality_scores=quality_scores,
        repair_records=repair_records,
        selected_tool_id=str(selected_tool["tool_id"]),
        round_index=round_index,
        run_id=run_id,
        mechanism_classifications=mechanism_rows,
    )
    write_prompt_pipeline_audit(
        output_dir,
        {
            "selection": {
                "selected_tool_id": selector_output["selected_tool_id"],
                "selected_design_id": selector_output["selected_design_id"],
                "selected_static_self_audit": selector_output["selected_static_self_audit"],
                "selected_exploitation_term": selector_output["selected_exploitation_term"],
                "selected_exploration_or_anti_collapse_term": selector_output[
                    "selected_exploration_or_anti_collapse_term"
                ],
                "selected_small_n_fallback": selector_output["selected_small_n_fallback"],
                "selected_mixed_variable_handling": selector_output["selected_mixed_variable_handling"],
                "selected_why_this_is_not_a_static_ranker": selector_output[
                    "selected_why_this_is_not_a_static_ranker"
                ],
                "true_y_reveals": 0,
            }
        },
    )
    write_selected_portfolio_tool(output_dir, selector_output)
    write_portfolio_summary_markdown(output_dir, _portfolio_summary_markdown(selector_output, quality_scores, repair_records, mechanism_rows))
    returned = dict(selector_output)
    returned["selected_tool_source"] = selected_source
    return returned


def _verify_candidate_tool(
    candidate: dict[str, Any],
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    round_index: int,
    old_tool_source: str,
) -> dict[str, Any]:
    source = str(candidate.get("source", ""))
    extra_safety = _extra_portfolio_safety_scan(source)
    audit_safety = _static_self_audit_safety(candidate)
    if audit_safety["failed_checks"]:
        extra_safety["failed_checks"] = sorted(
            set(list(extra_safety.get("failed_checks", [])) + list(audit_safety["failed_checks"]))
        )
        extra_safety["passed"] = False
    extra_safety["static_self_audit"] = audit_safety
    if extra_safety["failed_checks"]:
        replacement_report = {
            "passed": False,
            "deployable": False,
            "failed_checks": ["portfolio_extra_safety_scan"],
            "warning_checks": [],
            "checks": {"portfolio_extra_safety_scan": extra_safety},
            "reason": "failed:portfolio_extra_safety_scan",
        }
    else:
        replacement_report = verify_tool_replacement(
            old_tool_source=old_tool_source,
            patched_tool_source=source,
            observed_df=observed_df,
            candidate_df=candidate_df,
            memory="",
            tool_state={},
            round_index=round_index,
        )
        replacement_report["checks"]["portfolio_extra_safety_scan"] = extra_safety
    failed_checks = list(replacement_report.get("failed_checks", []))
    if extra_safety["failed_checks"]:
        failed_checks.extend(extra_safety["failed_checks"])
    report = {
        "tool_id": candidate.get("tool_id"),
        "parent_tool_id": candidate.get("parent_tool_id"),
        "candidate_version": candidate.get("candidate_version", "original"),
        "repair_attempt": int(candidate.get("repair_attempt", 0) or 0),
        "design_id": candidate.get("design_id"),
        "tool_family": candidate.get("tool_family"),
        "source_hash": candidate.get("source_hash") or sha256_text(source),
        "passed": bool(replacement_report.get("passed")) and not extra_safety["failed_checks"],
        "deployable": bool(replacement_report.get("deployable")) and not extra_safety["failed_checks"],
        "failed_checks": sorted(set(failed_checks)),
        "warning_checks": sorted(set(replacement_report.get("warning_checks", []))),
        "checks": replacement_report.get("checks", {}),
        "reason": (
            "passed_all_required_checks"
            if bool(replacement_report.get("passed")) and not extra_safety["failed_checks"]
            else "failed:" + ",".join(sorted(set(failed_checks)))
        ),
        "candidate_tools_reveal_y": False,
    }
    return report


def _extra_portfolio_safety_scan(source: str) -> dict[str, Any]:
    failed: list[str] = []
    details: dict[str, Any] = {
        "reveal_call_count": 0,
        "evaluator_name_count": 0,
        "dynamic_import_call_count": 0,
    }
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"passed": False, "failed_checks": ["syntax_error"], **details}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name == "reveal":
                details["reveal_call_count"] += 1
            if name in {"__import__", "eval", "exec", "compile"}:
                details["dynamic_import_call_count"] += 1
        if isinstance(node, ast.Name) and node.id.lower() in {"evaluator", "offlineevaluator"}:
            details["evaluator_name_count"] += 1
        if isinstance(node, ast.Attribute) and node.attr == "reveal":
            details["reveal_call_count"] += 1
    if details["reveal_call_count"]:
        failed.append("candidate_tool_reveal_attempt")
    if details["evaluator_name_count"]:
        failed.append("candidate_tool_evaluator_access")
    if details["dynamic_import_call_count"]:
        failed.append("dynamic_code_or_import_attempt")
    return {"passed": not failed, "failed_checks": failed, **details}


def _static_self_audit_safety(candidate: dict[str, Any]) -> dict[str, Any]:
    audit = candidate.get("static_self_audit")
    if not isinstance(audit, dict):
        return {"passed": True, "failed_checks": [], "warning_checks": ["missing_static_self_audit_metadata"]}
    failed: list[str] = []
    warnings: list[str] = []
    if audit.get("is_only_predicted_yield") is True or candidate.get("static_ranker_risk") is True:
        failed.append("self_audit_static_ranker_risk")
    if str(audit.get("hidden_y_leakage_self_check", "")).lower() not in {"", "pass"}:
        failed.append("self_audit_nonpublic_target_leakage")
    if audit.get("uses_only_observed_y") is False:
        failed.append("self_audit_uses_unrevealed_targets")
    if audit.get("uses_only_public_candidate_features") is False:
        failed.append("self_audit_uses_nonpublic_features")
    if str(audit.get("fake_uncertainty_risk", "")).lower() == "high":
        warnings.append("self_audit_fake_uncertainty_high")
    return {
        "passed": not failed,
        "failed_checks": failed,
        "warning_checks": warnings,
        "fake_uncertainty_risk": audit.get("fake_uncertainty_risk"),
        "is_only_predicted_yield": bool(audit.get("is_only_predicted_yield")),
    }


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _with_candidate_defaults(candidate: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(candidate)
    prepared.setdefault("candidate_version", "original")
    prepared.setdefault("repair_attempt", 0)
    prepared.setdefault("parent_tool_id", None)
    if "source_file" not in prepared and "source_path" in prepared:
        prepared["source_file"] = prepared.get("source_path")
    return prepared


def _schema_summary(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [{"column": str(column), "dtype": str(dtype)} for column, dtype in frame.dtypes.items()]


def _tool_contract_summary() -> list[str]:
    return [
        "Implement exactly: def rank_candidates(observed_df, candidate_df, memory=None, tool_state=None).",
        "Return a dict with ranked_candidates, tool_state, and tool_diagnostics.",
        "Return exactly one finite scored row for every row in candidate_df.",
        "Every ranked row must include candidate_id, rank, score, reason_code, and evidence_refs.",
        "evidence_refs must be [] or actual observation_id aliases from observed_df.",
        "Do not use files, network, subprocesses, dynamic imports, credentials, private state, or evaluator reveal calls.",
    ]


def _classify_candidate_tool(
    candidate: dict[str, Any],
    *,
    verifier_result: dict[str, Any] | None = None,
    observed_quality: dict[str, Any] | None = None,
    score_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    classification = classify_tool_mechanism(
        tool_source=str(candidate.get("source", "")),
        tool_id=str(candidate.get("tool_id")) if candidate.get("tool_id") is not None else None,
        design_metadata=candidate.get("design", {}) if isinstance(candidate.get("design"), dict) else {},
        raw_response_metadata=candidate.get("payload", {}) if isinstance(candidate.get("payload"), dict) else {},
        verifier_result=verifier_result,
        observed_quality=observed_quality,
        score_summary=score_summary,
    ).to_dict()
    return classification


def _build_mechanism_classification_rows(
    *,
    candidate_tools: list[dict[str, Any]],
    verifier_results: list[dict[str, Any]],
    quality_scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    verifier_by_tool = {str(row.get("tool_id")): row for row in verifier_results}
    quality_by_tool = {str(row.get("tool_id")): row for row in quality_scores}
    rows: list[dict[str, Any]] = []
    for item in candidate_tools:
        tool_id = str(item.get("tool_id"))
        quality = quality_by_tool.get(tool_id)
        classification = _classify_candidate_tool(
            item,
            verifier_result=verifier_by_tool.get(tool_id, {}),
            observed_quality=quality,
            score_summary=_quality_score_summary(quality),
        )
        rows.append(
            {
                "schema_version": "research_tool_mechanism_classification_v1",
                "tool_id": tool_id,
                "parent_tool_id": item.get("parent_tool_id"),
                "candidate_version": item.get("candidate_version", "original"),
                "repair_attempt": int(item.get("repair_attempt", 0) or 0),
                "design_id": item.get("design_id"),
                "tool_family": item.get("tool_family"),
                "source_hash": item.get("source_hash") or sha256_text(str(item.get("source", ""))),
                **classification,
            }
        )
    return rows


def _quality_score_summary(score: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(score, dict):
        return {}
    diagnostics = score.get("diagnostics")
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("score_sanity"), dict):
        return dict(diagnostics["score_sanity"])
    direct = score.get("score_summary")
    return dict(direct) if isinstance(direct, dict) else {}


def _write_augmented_candidate_manifest(
    *,
    output_dir: str,
    candidate_tools: list[dict[str, Any]],
    verifier_results: list[dict[str, Any]],
    quality_scores: list[dict[str, Any]],
    repair_records: list[dict[str, Any]],
    selected_tool_id: str | None,
    round_index: int,
    run_id: str,
    mechanism_classifications: list[dict[str, Any]],
) -> None:
    verifier_by_tool = {str(row.get("tool_id")): row for row in verifier_results}
    quality_by_tool = {str(row.get("tool_id")): row for row in quality_scores}
    repair_by_repaired = {str(row.get("repaired_tool_id")): row for row in repair_records}
    mechanism_by_tool = {str(row.get("tool_id")): row for row in mechanism_classifications}
    manifest = {
        "schema_version": "research_tool_portfolio_candidate_manifest_v1",
        "run_id": str(run_id),
        "round_index": int(round_index),
        "original_candidate_tool_count": sum(1 for item in candidate_tools if item.get("candidate_version", "original") == "original"),
        "repaired_candidate_tool_count": sum(1 for item in candidate_tools if item.get("candidate_version") == "repair"),
        "candidate_tool_count": len(candidate_tools),
        "repair_attempt_count": len([row for row in repair_records if row.get("attempted")]),
        "selected_tool_id": selected_tool_id,
        "candidates": [],
    }
    for item in candidate_tools:
        tool_id = str(item.get("tool_id"))
        verifier = verifier_by_tool.get(tool_id, {})
        quality = quality_by_tool.get(tool_id, {})
        repair = repair_by_repaired.get(tool_id, {})
        mechanism = mechanism_by_tool.get(tool_id, {})
        manifest["candidates"].append(
            {
                "tool_id": tool_id,
                "parent_tool_id": item.get("parent_tool_id"),
                "candidate_version": item.get("candidate_version", "original"),
                "repair_attempt": int(item.get("repair_attempt", 0) or 0),
                "design_id": item.get("design_id"),
                "tool_family": item.get("tool_family"),
                "source_file": item.get("source_file") or item.get("source_path"),
                "prompt_file": item.get("prompt_path"),
                "raw_response_file": item.get("raw_response_path"),
                "repair_record_file": repair.get("repair_record_file"),
                "prompt_hash": item.get("prompt_hash"),
                "source_hash": item.get("source_hash") or sha256_text(str(item.get("source", ""))),
                "allowed_imports": item.get("allowed_imports", []),
                "state_variables": item.get("design", {}).get("state_variables", []),
                "exploitation_term": item.get("design", {}).get("exploitation_term"),
                "exploration_or_anti_collapse_term": item.get("design", {}).get(
                    "exploration_or_anti_collapse_term"
                ),
                "small_n_fallback": item.get("design", {}).get("small_n_fallback"),
                "mixed_variable_handling": item.get("design", {}).get("mixed_variable_handling"),
                "static_self_audit": item.get("static_self_audit", {}),
                "why_this_is_not_a_static_ranker": item.get("design", {}).get("why_this_is_not_a_static_ranker"),
                "verifier_status": verifier.get("reason"),
                "verifier_passed": bool(verifier.get("passed")),
                "entered_quality_evaluator": tool_id in quality_by_tool,
                "quality_score": quality.get("quality_score"),
                "selected": str(selected_tool_id) == tool_id if selected_tool_id is not None else False,
                "mechanism_classification": mechanism,
                "optimizer_class": mechanism.get("optimizer_class"),
                "is_static_ranker": mechanism.get("is_static_ranker"),
            }
        )
    write_candidate_manifest(output_dir, manifest)


def _portfolio_summary_markdown(
    selector_output: dict[str, Any],
    quality_scores: list[dict[str, Any]],
    repair_records: list[dict[str, Any]] | None = None,
    mechanism_classifications: list[dict[str, Any]] | None = None,
) -> str:
    repair_records = repair_records or []
    mechanism_classifications = mechanism_classifications or []
    lines = [
        "# Tool Portfolio Diagnostic Summary",
        "",
        f"- Selected tool: `{selector_output.get('selected_tool_id')}`",
        f"- Selected parent tool: `{selector_output.get('selected_parent_tool_id')}`",
        f"- Selected version: `{selector_output.get('selected_candidate_version')}`",
        f"- Selected design: `{selector_output.get('selected_design_id')}`",
        f"- Selected family: `{selector_output.get('selected_tool_family')}`",
        f"- Selection reason: {selector_output.get('selected_reason') or selector_output.get('reason')}",
        f"- Candidate tools evaluated with true y reveals: `{selector_output.get('true_y_reveals_during_selection', 0)}`",
        f"- Portfolio repair enabled: `{selector_output.get('repair_enabled', False)}`",
        f"- Repair attempts: `{selector_output.get('repair_attempt_count', 0)}`",
        f"- Selected exploitation term: {selector_output.get('selected_exploitation_term') or 'n/a'}",
        f"- Selected exploration/anti-collapse term: {selector_output.get('selected_exploration_or_anti_collapse_term') or 'n/a'}",
        f"- Selected small-n fallback: {selector_output.get('selected_small_n_fallback') or 'n/a'}",
        f"- Selected mixed-variable handling: {selector_output.get('selected_mixed_variable_handling') or 'n/a'}",
        f"- Selected static-ranker rationale: {selector_output.get('selected_why_this_is_not_a_static_ranker') or 'n/a'}",
        "",
        "## Prompt Pipeline Audit",
        "",
        "| selected_tool_id | design_id | exploitation | exploration_or_anti_collapse | small_n_fallback | mixed_variable_handling |",
        "|---|---|---|---|---|---|",
        (
            f"| `{selector_output.get('selected_tool_id')}` | `{selector_output.get('selected_design_id')}` | "
            f"{selector_output.get('selected_exploitation_term') or 'n/a'} | "
            f"{selector_output.get('selected_exploration_or_anti_collapse_term') or 'n/a'} | "
            f"{selector_output.get('selected_small_n_fallback') or 'n/a'} | "
            f"{selector_output.get('selected_mixed_variable_handling') or 'n/a'} |"
        ),
        "",
        "## Candidate Repair Summary",
        "",
        "| original_tool_id | repaired_tool_id | attempted | parse_status | verifier_status | entered_quality_evaluator |",
        "|---|---|---:|---|---|---:|",
    ]
    for row in repair_records:
        lines.append(
            f"| `{row.get('original_tool_id')}` | `{row.get('repaired_tool_id')}` | {bool(row.get('attempted'))} | "
            f"{row.get('parse_status')} | {row.get('verifier_status')} | {bool(row.get('accepted_for_quality_evaluation'))} |"
        )
    if not repair_records:
        lines.append("| none | none | false | not_attempted | not_run | false |")
    lines.extend(
        [
            "",
            "## Mechanism Classifications",
            "",
            "| tool_id | parent_tool_id | version | design_id | optimizer_class | static | surrogate | uncertainty | exploration | diversity | acquisition | sklearn import | sklearn class | handwritten estimator | fake uncertainty risk | over exploration risk | score scale risk | leakage risk | confidence |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|",
        ]
    )
    for row in mechanism_classifications:
        lines.append(
            f"| `{row.get('tool_id')}` | `{row.get('parent_tool_id')}` | {row.get('candidate_version', 'original')} | "
            f"`{row.get('design_id')}` | {row.get('optimizer_class')} | {bool(row.get('is_static_ranker'))} | "
            f"{bool(row.get('uses_surrogate'))} | {bool(row.get('uses_uncertainty'))} | {bool(row.get('uses_exploration'))} | "
            f"{bool(row.get('uses_diversity'))} | {bool(row.get('uses_acquisition_logic'))} | "
            f"{bool(row.get('uses_sklearn_import'))} | {bool(row.get('uses_sklearn_estimator_class'))} | "
            f"{bool(row.get('uses_handwritten_estimator_like_logic'))} | {row.get('fake_uncertainty_risk')} | "
            f"{row.get('over_exploration_risk')} | {row.get('score_scale_risk')} | {row.get('leakage_risk')} | "
            f"{row.get('classification_confidence')} |"
        )
    if not mechanism_classifications:
        lines.append("| none | none | none | none | unknown | false | false | false | false | false | false | false | false | false | unknown | unknown | unknown | unknown | low |")
    lines.extend(
        [
        "",
        "## Observed-only Scores",
        "",
        "| tool_id | parent_tool_id | version | design_id | quality_score | observed_ranking | stability | sanity |",
        "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in quality_scores:
        lines.append(
            f"| `{row.get('tool_id')}` | `{row.get('parent_tool_id')}` | {row.get('candidate_version', 'original')} | "
            f"`{row.get('design_id')}` | {row.get('quality_score')} | "
            f"{row.get('observed_ranking_score')} | {row.get('bootstrap_stability')} | {row.get('score_sanity')} |"
        )
    if not quality_scores:
        lines.append("| none | none | none | none | 0 | 0 | 0 | 0 |")
    return "\n".join(lines) + "\n"
