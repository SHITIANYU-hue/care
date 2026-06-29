"""Verifier for safe generated-tool patch replacement."""

from __future__ import annotations

import ast
import math
from typing import Any

import pandas as pd

from research_tool_agent_full_pool.artifact_logger import sanitize_text
from research_tool_agent_full_pool.diagnostics import sha256_text
from research_tool_agent_full_pool.tool_contract import REQUIRED_ENTRYPOINT
from research_tool_agent_full_pool.tool_output_parser import ParsedToolOutput, parse_ranked_candidates
from research_tool_agent_full_pool.tool_runner import execute_rank_candidates_tool
from research_tool_agent_full_pool.tool_sandbox import validate_tool_in_sandbox
from research_tool_agent_full_pool.tool_static_check import static_check_generated_tool_source


SCORE_TOLERANCE = 1e-9
DEGENERACY_TOLERANCE = 1e-12


def verify_tool_replacement(
    *,
    old_tool_source: str,
    patched_tool_source: str,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    memory: str | None = None,
    tool_state: dict[str, Any] | None = None,
    round_index: int | None = None,
) -> dict[str, Any]:
    """Return a structured replacement decision for a candidate patched tool.

    The verifier is intentionally stricter than first-pass tool validation
    because replacement would change the active run state. It never exposes
    hidden targets or private candidate maps to generated code.
    """

    result: dict[str, Any] = {
        "verifier_version": "batch3.tool_replacement_verifier.v1",
        "round_index": round_index,
        "passed": False,
        "deployable": False,
        "failed_checks": [],
        "warning_checks": [],
        "checks": {},
        "old_tool_hash": sha256_text(old_tool_source or ""),
        "patched_tool_hash": sha256_text(patched_tool_source or ""),
        "reason": "",
    }
    failed = result["failed_checks"]
    warnings = result["warning_checks"]
    checks = result["checks"]

    signature = _check_exact_signature(patched_tool_source)
    checks["exact_function_signature"] = signature
    if not signature["passed"]:
        failed.append("exact_function_signature")

    static_report = static_check_generated_tool_source(patched_tool_source)
    checks["static_check"] = static_report
    if not static_report["passed"]:
        failed.append("static_check")
    if "dataframe_attrs_access" in static_report.get("violations", []):
        failed.append("dataframe_attrs_leakage")
        checks["dataframe_attrs_leakage"] = {
            "passed": False,
            "mode": "static_block",
            "reason": "patched tool source accesses pandas DataFrame attrs",
        }
    else:
        checks["dataframe_attrs_leakage"] = {
            "passed": True,
            "mode": "runner_strips_attrs_and_static_blocks_attrs_access",
        }

    hardcoded = _scan_hardcoded_candidate_ids(patched_tool_source, candidate_df)
    checks["hardcoded_candidate_id_scan"] = hardcoded
    if hardcoded["severity"] == "fail":
        failed.append("hardcoded_candidate_id_scan")
    elif hardcoded["severity"] == "warning":
        warnings.append("hardcoded_candidate_id_scan")

    if not failed:
        sandbox_report = validate_tool_in_sandbox(
            tool_source=patched_tool_source,
            observed_df=_public_copy(observed_df),
            candidate_df=_public_copy(candidate_df),
            memory=memory,
            tool_state=dict(tool_state or {}),
            max_candidate_rows=max(1, min(16, len(candidate_df))),
        )
    else:
        sandbox_report = {
            "passed": False,
            "skipped": True,
            "reason": "signature_static_or_hardcoded_check_failed",
        }
    checks["sandbox_validation"] = sandbox_report
    if not sandbox_report.get("passed"):
        failed.append("sandbox_validation")

    first: ParsedToolOutput | None = None
    second: ParsedToolOutput | None = None
    shuffled: ParsedToolOutput | None = None
    if not failed:
        run1 = _run_full_pool(
            label="full_pool_preacceptance_first",
            tool_source=patched_tool_source,
            observed_df=observed_df,
            candidate_df=candidate_df,
            memory=memory,
            tool_state=tool_state,
        )
        checks["full_pool_preacceptance"] = run1["report"]
        first = run1["parsed"]
        if first is None:
            failed.append("full_pool_preacceptance")

    if first is not None and not failed:
        run2 = _run_full_pool(
            label="full_pool_preacceptance_second",
            tool_source=patched_tool_source,
            observed_df=observed_df,
            candidate_df=candidate_df,
            memory=memory,
            tool_state=tool_state,
        )
        second = run2["parsed"]
        checks["deterministic_repeated_run"] = _compare_repeated_runs(first, second, run2["report"])
        if not checks["deterministic_repeated_run"]["passed"]:
            failed.append("deterministic_repeated_run")

    if first is not None and not failed:
        shuffled_df = _public_copy(candidate_df).sample(frac=1.0, random_state=173).reset_index(drop=True)
        run3 = _run_full_pool(
            label="row_order_perturbation",
            tool_source=patched_tool_source,
            observed_df=observed_df,
            candidate_df=shuffled_df,
            memory=memory,
            tool_state=tool_state,
        )
        shuffled = run3["parsed"]
        checks["row_order_perturbation"] = _compare_row_order(first, shuffled, run3["report"])
        if not checks["row_order_perturbation"]["passed"]:
            failed.append("row_order_perturbation")

    if first is not None:
        degeneracy = _score_degeneracy_check(first)
        checks["score_degeneracy"] = degeneracy
        if degeneracy["severity"] == "warning":
            warnings.append("score_degeneracy")
        elif degeneracy["severity"] == "fail":
            failed.append("score_degeneracy")
    else:
        checks["score_degeneracy"] = {"passed": False, "skipped": True}

    result["failed_checks"] = sorted(set(failed))
    result["warning_checks"] = sorted(set(warnings))
    result["deployable"] = not result["failed_checks"]
    result["passed"] = bool(result["deployable"])
    result["reason"] = "passed_all_required_checks" if result["passed"] else "failed:" + ",".join(result["failed_checks"])
    return result


def _check_exact_signature(source: str) -> dict[str, Any]:
    report = {"passed": False, "reason": "", "expected": "rank_candidates(observed_df, candidate_df, memory=None, tool_state=None)"}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        report["reason"] = "syntax_error"
        return report
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == REQUIRED_ENTRYPOINT
    ]
    if len(functions) != 1:
        report["reason"] = "missing_or_duplicate_rank_candidates"
        return report
    func = functions[0]
    args = func.args
    arg_names = [arg.arg for arg in args.args]
    if arg_names != ["observed_df", "candidate_df", "memory", "tool_state"]:
        report["reason"] = "unexpected_positional_arguments"
        report["actual_args"] = arg_names
        return report
    if args.vararg is not None or args.kwarg is not None or args.kwonlyargs:
        report["reason"] = "varargs_or_keyword_only_arguments_not_allowed"
        return report
    if len(args.defaults) != 2 or not all(isinstance(default, ast.Constant) and default.value is None for default in args.defaults):
        report["reason"] = "memory_and_tool_state_defaults_must_be_none"
        return report
    report["passed"] = True
    report["reason"] = "signature_matches"
    return report


def _scan_hardcoded_candidate_ids(source: str, candidate_df: pd.DataFrame) -> dict[str, Any]:
    candidate_ids = {str(value) for value in candidate_df["candidate_id"].tolist()}
    report: dict[str, Any] = {
        "passed": True,
        "severity": "pass",
        "literal_candidate_id_count": 0,
        "direct_candidate_id_branch_count": 0,
    }
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return report

    literal_hits: set[str] = set()
    direct_branch_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in candidate_ids:
            literal_hits.add(node.value)
        if isinstance(node, ast.Compare):
            if _mentions_candidate_id(node.left) or any(_mentions_candidate_id(comp) for comp in node.comparators):
                if any(isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in node.ops):
                    compared_literals = set()
                    for comp in node.comparators:
                        compared_literals.update(_string_literals(comp))
                    compared_literals.update(_string_literals(node.left))
                    if candidate_ids & compared_literals:
                        direct_branch_count += 1
    if literal_hits or direct_branch_count:
        report.update(
            {
                "passed": False,
                "severity": "fail",
                "literal_candidate_id_count": len(literal_hits),
                "direct_candidate_id_branch_count": direct_branch_count,
                "reason": "candidate-specific string literals or equality logic detected",
            }
        )
    return report


def _mentions_candidate_id(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and node.value == "candidate_id":
        return True
    if isinstance(node, ast.Subscript):
        return _mentions_candidate_id(node.slice)
    if isinstance(node, ast.Call):
        return any(_mentions_candidate_id(arg) for arg in node.args)
    if isinstance(node, ast.Attribute):
        return node.attr == "candidate_id"
    return False


def _string_literals(node: ast.AST) -> set[str]:
    values: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            values.add(child.value)
    return values


def _run_full_pool(
    *,
    label: str,
    tool_source: str,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    memory: str | None,
    tool_state: dict[str, Any] | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "passed": False,
        "label": label,
        "candidate_count": int(len(candidate_df)),
        "parsed_candidate_count": 0,
        "top1_candidate_id": None,
        "score_min": None,
        "score_max": None,
        "error_type": None,
        "error": None,
    }
    try:
        raw_output, runner_report = execute_rank_candidates_tool(
            tool_source=tool_source,
            observed_df=_public_copy(observed_df),
            candidate_df=_public_copy(candidate_df),
            memory=memory,
            tool_state=dict(tool_state or {}),
        )
        parsed = parse_ranked_candidates(
            raw_output,
            candidate_df=_public_copy(candidate_df),
            observed_df=_public_copy(observed_df),
        )
        scores = [float(row["score"]) for row in parsed.ranked_candidates]
        report.update(
            {
                "passed": True,
                "runner_passed": bool(runner_report.get("passed")),
                "parsed_candidate_count": len(parsed.ranked_candidates),
                "top1_candidate_id": parsed.selected_display_candidate_id,
                "score_min": _safe_float(min(scores)),
                "score_max": _safe_float(max(scores)),
            }
        )
        return {"parsed": parsed, "report": report}
    except Exception as exc:
        report["error_type"] = exc.__class__.__name__
        report["error"] = sanitize_text(str(exc))[:800]
        return {"parsed": None, "report": report}


def _compare_repeated_runs(
    first: ParsedToolOutput,
    second: ParsedToolOutput | None,
    second_report: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "passed": False,
        "first_top1": first.selected_display_candidate_id,
        "second_top1": None,
        "max_abs_score_delta": None,
        "rank_mismatch_count": None,
        "second_run_report": second_report,
    }
    if second is None:
        return report
    first_rows = {str(row["candidate_id"]): row for row in first.ranked_candidates}
    second_rows = {str(row["candidate_id"]): row for row in second.ranked_candidates}
    deltas = []
    rank_mismatch = 0
    for candidate_id, first_row in first_rows.items():
        second_row = second_rows.get(candidate_id)
        if second_row is None:
            rank_mismatch += 1
            continue
        deltas.append(abs(float(first_row["score"]) - float(second_row["score"])))
        if int(first_row["rank"]) != int(second_row["rank"]):
            rank_mismatch += 1
    max_delta = max(deltas) if deltas else math.inf
    report.update(
        {
            "second_top1": second.selected_display_candidate_id,
            "max_abs_score_delta": _safe_float(max_delta),
            "rank_mismatch_count": rank_mismatch,
            "passed": (
                first.selected_display_candidate_id == second.selected_display_candidate_id
                and rank_mismatch == 0
                and max_delta <= SCORE_TOLERANCE
            ),
        }
    )
    return report


def _compare_row_order(
    first: ParsedToolOutput,
    shuffled: ParsedToolOutput | None,
    shuffled_report: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "passed": False,
        "first_top1": first.selected_display_candidate_id,
        "shuffled_top1": None,
        "max_abs_score_delta": None,
        "rank_mismatch_count": None,
        "missing_candidate_count": None,
        "shuffled_run_report": shuffled_report,
    }
    if shuffled is None:
        return report
    first_rows = {str(row["candidate_id"]): row for row in first.ranked_candidates}
    shuffled_rows = {str(row["candidate_id"]): row for row in shuffled.ranked_candidates}
    deltas = []
    rank_mismatch = 0
    missing = 0
    for candidate_id, first_row in first_rows.items():
        shuffled_row = shuffled_rows.get(candidate_id)
        if shuffled_row is None:
            missing += 1
            rank_mismatch += 1
            continue
        deltas.append(abs(float(first_row["score"]) - float(shuffled_row["score"])))
        if int(first_row["rank"]) != int(shuffled_row["rank"]):
            rank_mismatch += 1
    extra = len(set(shuffled_rows) - set(first_rows))
    missing += extra
    max_delta = max(deltas) if deltas else math.inf
    report["shuffled_top1"] = shuffled.selected_display_candidate_id
    report["max_abs_score_delta"] = _safe_float(max_delta)
    report["rank_mismatch_count"] = rank_mismatch
    report["missing_candidate_count"] = missing
    report["passed"] = (
        first.selected_display_candidate_id == shuffled.selected_display_candidate_id
        and rank_mismatch == 0
        and missing == 0
        and max_delta <= SCORE_TOLERANCE
    )
    return report


def _score_degeneracy_check(parsed: ParsedToolOutput) -> dict[str, Any]:
    scores = [float(row["score"]) for row in parsed.ranked_candidates]
    span = max(scores) - min(scores) if scores else 0.0
    severity = "warning" if len(scores) > 1 and span <= DEGENERACY_TOLERANCE else "pass"
    return {
        "passed": severity != "fail",
        "severity": severity,
        "score_span": _safe_float(span),
        "candidate_count": len(scores),
        "reason": "near_identical_scores" if severity == "warning" else "score_variation_present",
    }


def _public_copy(frame: pd.DataFrame) -> pd.DataFrame:
    copied = frame.copy()
    copied.attrs = {}
    return copied


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result
