"""One-shot public-safe repair for portfolio candidate tools."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from research_tool_agent_full_pool.artifact_logger import ensure_output_dir, sanitize_payload, sanitize_text
from research_tool_agent_full_pool.decision_artifacts import DecisionArtifactSafetyError, assert_payload_public_safe
from research_tool_agent_full_pool.diagnostics import sha256_text
from research_tool_agent_full_pool.tool_contract import ALLOWED_IMPORTS, REQUIRED_ENTRYPOINT
from research_tool_agent_full_pool.tool_portfolio_design import REQUIRED_STATIC_SELF_AUDIT_SCHEMA


CANDIDATE_REPAIR_SCHEMA_VERSION = "research_tool_portfolio_candidate_repair_v1"
CANDIDATE_REPAIR_POLICY_NAME = "ResearchToolAgentToolPortfolioRepair"


class CandidateRepairError(ValueError):
    """Raised when a candidate repair response is malformed or unsafe."""


@dataclass(frozen=True)
class RepairedCandidateToolResult:
    """Public-safe repair result plus artifact paths."""

    attempted: bool
    original_tool_id: str
    repaired_tool_id: str
    design_id: str
    repair_attempt: int
    parse_status: str
    failed_stage: str
    original_error: str
    source_hash_before: str
    source_hash_after: str | None
    leakage_audit_status: str
    repaired_tool: dict[str, Any] | None
    prompt_path: str | None
    request_path: str | None
    raw_response_path: str | None
    repaired_source_file: str | None
    parse_report_file: str | None
    repair_record_file: str | None
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "original_tool_id": self.original_tool_id,
            "repaired_tool_id": self.repaired_tool_id,
            "design_id": self.design_id,
            "failed_stage": self.failed_stage,
            "original_error": self.original_error,
            "repair_attempt": self.repair_attempt,
            "parse_status": self.parse_status,
            "verifier_status": "not_run",
            "accepted_for_quality_evaluation": False,
            "source_hash_before": self.source_hash_before,
            "source_hash_after": self.source_hash_after,
            "leakage_audit_status": self.leakage_audit_status,
            "prompt_file": self.prompt_path,
            "request_file": self.request_path,
            "raw_response_file": self.raw_response_path,
            "repaired_source_file": self.repaired_source_file,
            "parse_report_file": self.parse_report_file,
            "repair_record_file": self.repair_record_file,
            "error": self.error,
        }


def repair_failed_candidate_tool(
    *,
    failed_tool_source: str,
    design_metadata: dict[str, Any],
    verifier_result: dict[str, Any],
    observed_schema_summary: list[dict[str, str]],
    candidate_schema_summary: list[dict[str, str]],
    tool_contract: list[str] | dict[str, Any] | None,
    allowed_imports: list[str] | tuple[str, ...] | None,
    forbidden_boundary: list[str] | tuple[str, ...] | str | None,
    output_dir: str | Path,
    client: Any,
    original_tool_id: str | None = None,
    run_id: str = "",
    round_index: int = 1,
    repair_attempt: int = 1,
) -> RepairedCandidateToolResult:
    """Attempt exactly one narrow public-safe repair for one failed candidate.

    The repair context is preflight-scanned before any LLM call. If the failed
    source or verifier details contain non-public or forbidden material, repair
    is skipped rather than echoing unsafe content into a prompt.
    """

    original_id = str(original_tool_id or verifier_result.get("tool_id") or "candidate_tool")
    design_id = str(design_metadata.get("design_id") or verifier_result.get("design_id") or "")
    repaired_id = f"{original_id}_repair_{int(repair_attempt):03d}"
    failed_stage, original_error = failed_stage_and_error(verifier_result)
    before_hash = sha256_text(str(failed_tool_source))
    root = ensure_output_dir(output_dir) / "candidate_tool_repairs"
    root.mkdir(parents=True, exist_ok=True)

    base_paths = _repair_paths(root, original_id)
    try:
        prompt = build_candidate_repair_prompt(
            failed_tool_source=failed_tool_source,
            design_metadata=design_metadata,
            verifier_result=verifier_result,
            observed_schema_summary=observed_schema_summary,
            candidate_schema_summary=candidate_schema_summary,
            tool_contract=tool_contract,
            allowed_imports=allowed_imports,
            forbidden_boundary=forbidden_boundary,
            original_tool_id=original_id,
            repaired_tool_id=repaired_id,
            run_id=run_id,
            round_index=round_index,
            repair_attempt=repair_attempt,
        )
    except DecisionArtifactSafetyError:
        result = RepairedCandidateToolResult(
            attempted=False,
            original_tool_id=original_id,
            repaired_tool_id=repaired_id,
            design_id=design_id,
            repair_attempt=int(repair_attempt),
            parse_status="skipped_unsafe_repair_context",
            failed_stage=failed_stage,
            original_error=original_error,
            source_hash_before=before_hash,
            source_hash_after=None,
            leakage_audit_status="blocked_before_prompt",
            repaired_tool=None,
            prompt_path=None,
            request_path=None,
            raw_response_path=None,
            repaired_source_file=None,
            parse_report_file=str(base_paths["parse_report"]),
            repair_record_file=str(base_paths["record"]),
            error="repair context did not pass public-safe artifact scan",
        )
        _write_json(base_paths["parse_report"], {"parse_status": result.parse_status, "error": result.error})
        _write_json(base_paths["record"], result.to_record())
        return result

    request_metadata = {
        "mode": "portfolio_candidate_tool_repair",
        "schema_version": CANDIDATE_REPAIR_SCHEMA_VERSION,
        "policy_name": CANDIDATE_REPAIR_POLICY_NAME,
        "run_id": str(run_id),
        "round_index": int(round_index),
        "original_tool_id": original_id,
        "repaired_tool_id": repaired_id,
        "design_id": design_id,
        "repair_attempt": int(repair_attempt),
        "failed_stage": failed_stage,
        "original_error": original_error,
        "prompt_hash": sha256_text(prompt),
        "prompt_character_count": len(prompt),
        "source_hash_before": before_hash,
    }
    assert_payload_public_safe(request_metadata, label="candidate_repair_request")
    base_paths["prompt"].write_text(sanitize_text(prompt), encoding="utf-8")
    _write_json(base_paths["request"], request_metadata)

    raw_text = ""
    try:
        raw_text = client.create_tool(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only one strict JSON object containing the repaired Python source. "
                        "Do not return markdown or prose outside JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )
        _write_json(base_paths["raw"], {"raw_text": sanitize_text(raw_text)})
        payload = parse_candidate_repair_json(
            raw_text,
            expected_run_id=str(run_id),
            expected_round_index=int(round_index),
            expected_original_tool_id=original_id,
            expected_repaired_tool_id=repaired_id,
            expected_design_id=design_id,
            expected_repair_attempt=int(repair_attempt),
        )
        source = str(payload["repaired_tool_source"])
        after_hash = sha256_text(source)
        base_paths["source"].write_text(sanitize_text(source), encoding="utf-8")
        parse_report = {
            "parse_status": "parsed",
            "source_hash_before": before_hash,
            "source_hash_after": after_hash,
            "repair_summary": payload.get("repair_summary"),
            "changed_lines_or_components": payload.get("changed_lines_or_components"),
            "static_self_audit": payload.get("static_self_audit", {}),
        }
        _write_json(base_paths["parse_report"], parse_report)
        repaired_tool = {
            "tool_id": repaired_id,
            "parent_tool_id": original_id,
            "repair_attempt": int(repair_attempt),
            "candidate_version": "repair",
            "design_id": design_id,
            "tool_family": str(design_metadata.get("family", verifier_result.get("tool_family", ""))),
            "source": source,
            "source_file": str(base_paths["source"]),
            "source_hash": after_hash,
            "allowed_imports": list(payload.get("allowed_imports", [])),
            "design": dict(design_metadata),
            "repair_summary": str(payload.get("repair_summary", "")),
            "changed_lines_or_components": list(payload.get("changed_lines_or_components", [])),
            "public_safe_boundary_statement": str(payload.get("public_safe_boundary_statement", "")),
            "expected_remaining_risks": list(payload.get("expected_remaining_risks", [])),
            "static_self_audit": dict(payload.get("static_self_audit", {})),
            "static_ranker_risk": bool(payload.get("static_self_audit", {}).get("is_only_predicted_yield")),
        }
        result = RepairedCandidateToolResult(
            attempted=True,
            original_tool_id=original_id,
            repaired_tool_id=repaired_id,
            design_id=design_id,
            repair_attempt=int(repair_attempt),
            parse_status="parsed",
            failed_stage=failed_stage,
            original_error=original_error,
            source_hash_before=before_hash,
            source_hash_after=after_hash,
            leakage_audit_status="pass",
            repaired_tool=repaired_tool,
            prompt_path=str(base_paths["prompt"]),
            request_path=str(base_paths["request"]),
            raw_response_path=str(base_paths["raw"]),
            repaired_source_file=str(base_paths["source"]),
            parse_report_file=str(base_paths["parse_report"]),
            repair_record_file=str(base_paths["record"]),
        )
        _write_json(base_paths["record"], result.to_record())
        return result
    except Exception as exc:
        if raw_text and not base_paths["raw"].exists():
            _write_json(base_paths["raw"], {"raw_text": sanitize_text(raw_text)})
        parse_report = {
            "parse_status": "failed",
            "error_type": exc.__class__.__name__,
            "error": sanitize_text(str(exc))[:800],
        }
        _write_json(base_paths["parse_report"], parse_report)
        result = RepairedCandidateToolResult(
            attempted=True,
            original_tool_id=original_id,
            repaired_tool_id=repaired_id,
            design_id=design_id,
            repair_attempt=int(repair_attempt),
            parse_status="failed",
            failed_stage=failed_stage,
            original_error=original_error,
            source_hash_before=before_hash,
            source_hash_after=None,
            leakage_audit_status="pass",
            repaired_tool=None,
            prompt_path=str(base_paths["prompt"]),
            request_path=str(base_paths["request"]),
            raw_response_path=str(base_paths["raw"]) if base_paths["raw"].exists() else None,
            repaired_source_file=None,
            parse_report_file=str(base_paths["parse_report"]),
            repair_record_file=str(base_paths["record"]),
            error=f"{exc.__class__.__name__}: {str(exc)[:500]}",
        )
        _write_json(base_paths["record"], result.to_record())
        return result


def build_candidate_repair_prompt(
    *,
    failed_tool_source: str,
    design_metadata: dict[str, Any],
    verifier_result: dict[str, Any],
    observed_schema_summary: list[dict[str, str]],
    candidate_schema_summary: list[dict[str, str]],
    tool_contract: list[str] | dict[str, Any] | None,
    allowed_imports: list[str] | tuple[str, ...] | None,
    forbidden_boundary: list[str] | tuple[str, ...] | str | None,
    original_tool_id: str,
    repaired_tool_id: str,
    run_id: str,
    round_index: int,
    repair_attempt: int,
) -> str:
    """Build a narrow repair prompt from public-safe failure context."""

    failed_stage, original_error = failed_stage_and_error(verifier_result)
    payload = {
        "task": "research_tool_agent_candidate_tool_repair",
        "schema_version": CANDIDATE_REPAIR_SCHEMA_VERSION,
        "policy_name": CANDIDATE_REPAIR_POLICY_NAME,
        "run_id": str(run_id),
        "round_index": int(round_index),
        "original_tool_id": str(original_tool_id),
        "repaired_tool_id": str(repaired_tool_id),
        "design_id": str(design_metadata.get("design_id", verifier_result.get("design_id", ""))),
        "repair_attempt": int(repair_attempt),
        "repair_scope": [
            "Preserve the original design intent as much as possible.",
            "Fix only the verifier, sandbox, or output-schema failure.",
            "Do not rewrite the whole algorithm unless necessary.",
            "Keep the rank_candidates interface unchanged.",
            "Do not add forbidden imports, file access, network access, subprocesses, dynamic imports, credentials, or non-public data access.",
            "Use only observed_df, candidate_df, memory, and tool_state supplied at runtime.",
            "If the failure involves evidence_refs, set evidence_refs to [] or actual observation_id values from observed_df only.",
            "Never put component names, method names, feature names, candidate IDs, labels, counts, scores, diagnostics, source names, or file names in evidence_refs.",
            "If the failure involves an unavailable sandbox builtin, replace it with sandbox-safe deterministic logic.",
            "For deterministic tie-breaking, sort by candidate_id string or stable numeric score columns.",
            "Return exactly one row per candidate with candidate_id, finite score, deterministic rank, reason_code, and evidence_refs.",
            "Preserve or add explicit sequential selection logic; if the repaired tool only predicts yield or produces a smooth score without that policy, mark static_self_audit.is_only_predicted_yield true.",
            "Align observed and candidate public feature columns before vector operations; handle missing columns and nonfinite values.",
            "Do not use DataFrame attrs.",
            "Return the full repaired source.",
        ],
        "safe_boundary_summary": _safe_boundary_summary(forbidden_boundary),
        "failed_candidate_tool_source": str(failed_tool_source),
        "public_safe_design_metadata": _compact_design(design_metadata),
        "verifier_failure": {
            "failed_stage": failed_stage,
            "failed_checks": list(verifier_result.get("failed_checks", [])),
            "reason": str(verifier_result.get("reason", "")),
            "error": original_error,
        },
        "observed_df_schema_summary": observed_schema_summary,
        "candidate_df_schema_summary": candidate_schema_summary,
        "rank_candidates_contract": tool_contract
        or [
            "def rank_candidates(observed_df, candidate_df, memory=None, tool_state=None)",
            "return dict with ranked_candidates, tool_state, and tool_diagnostics",
            "ranked_candidates contains exactly one finite scored row for every candidate",
        ],
        "allowed_imports": list(allowed_imports or ALLOWED_IMPORTS),
        "static_self_audit_required": REQUIRED_STATIC_SELF_AUDIT_SCHEMA,
        "static_self_audit_rules": [
            "If the tool only predicts yield or produces a smooth score without explicit sequential selection logic, mark the design invalid.",
            "If uncertainty is just a constant, arbitrary noise, or a name without observed-data basis, mark fake_uncertainty_risk high.",
            "If categorical variables are ignored, mark mixed-variable handling as incomplete.",
            "If small observed n is not handled, mark handles_small_n false.",
            "If the design could repeatedly recommend near-duplicates without any diversity/anti-collapse logic, flag it.",
        ],
        "required_json_output": {
            "schema_version": CANDIDATE_REPAIR_SCHEMA_VERSION,
            "policy_name": CANDIDATE_REPAIR_POLICY_NAME,
            "run_id": str(run_id),
            "round_index": int(round_index),
            "original_tool_id": str(original_tool_id),
            "repaired_tool_id": str(repaired_tool_id),
            "design_id": str(design_metadata.get("design_id", verifier_result.get("design_id", ""))),
            "repair_attempt": int(repair_attempt),
            "allowed_imports": ["numpy", "pandas"],
            "repaired_tool_source": "full repaired Python source as a JSON string",
            "repair_summary": "short public-safe summary",
            "changed_lines_or_components": ["component name or line description"],
            "public_safe_boundary_statement": "short statement",
            "expected_remaining_risks": ["risk"],
            "static_self_audit": REQUIRED_STATIC_SELF_AUDIT_SCHEMA,
            "self_reported_forbidden_info_used": False,
        },
        "strict_output_rules": [
            "Return exactly one JSON object and no markdown.",
            "No explanatory prose outside the JSON object.",
            "First character must be { and last character must be }.",
            "The repaired_tool_source string must decode to Python source with real newline characters.",
        ],
    }
    assert_payload_public_safe(payload, label="candidate_repair_prompt_payload")
    return json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=True)


def parse_candidate_repair_json(
    raw_text: str,
    *,
    expected_run_id: str,
    expected_round_index: int,
    expected_original_tool_id: str,
    expected_repaired_tool_id: str,
    expected_design_id: str,
    expected_repair_attempt: int,
) -> dict[str, Any]:
    """Parse one candidate repair response."""

    text = str(raw_text).strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise CandidateRepairError("Candidate repair response must be exactly one JSON object.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CandidateRepairError("Candidate repair response is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise CandidateRepairError("Candidate repair response must decode to an object.")
    allowed_top = {
        "schema_version",
        "policy_name",
        "run_id",
        "round_index",
        "original_tool_id",
        "repaired_tool_id",
        "design_id",
        "repair_attempt",
        "allowed_imports",
        "repaired_tool_source",
        "repair_summary",
        "changed_lines_or_components",
        "public_safe_boundary_statement",
        "expected_remaining_risks",
        "static_self_audit",
        "self_reported_forbidden_info_used",
    }
    extra = set(payload) - allowed_top
    if extra:
        raise CandidateRepairError(f"Unsupported candidate repair fields: {sorted(extra)}")
    if payload.get("schema_version") != CANDIDATE_REPAIR_SCHEMA_VERSION:
        raise CandidateRepairError("Candidate repair schema_version mismatch.")
    if payload.get("policy_name") != CANDIDATE_REPAIR_POLICY_NAME:
        raise CandidateRepairError("Candidate repair policy_name mismatch.")
    if str(payload.get("run_id")) != str(expected_run_id):
        raise CandidateRepairError("Candidate repair run_id mismatch.")
    if int(payload.get("round_index", -1)) != int(expected_round_index):
        raise CandidateRepairError("Candidate repair round_index mismatch.")
    if str(payload.get("original_tool_id")) != str(expected_original_tool_id):
        raise CandidateRepairError("Candidate repair original_tool_id mismatch.")
    if str(payload.get("repaired_tool_id")) != str(expected_repaired_tool_id):
        raise CandidateRepairError("Candidate repair repaired_tool_id mismatch.")
    if str(payload.get("design_id")) != str(expected_design_id):
        raise CandidateRepairError("Candidate repair design_id mismatch.")
    if int(payload.get("repair_attempt", -1)) != int(expected_repair_attempt):
        raise CandidateRepairError("Candidate repair_attempt mismatch.")
    if payload.get("self_reported_forbidden_info_used") is not False:
        raise CandidateRepairError("self_reported_forbidden_info_used must be false.")
    payload["static_self_audit"] = _validate_static_self_audit(payload.get("static_self_audit"))
    imports = payload.get("allowed_imports")
    if not isinstance(imports, list):
        raise CandidateRepairError("allowed_imports must be a list.")
    if not set(str(item) for item in imports).issubset(set(ALLOWED_IMPORTS)):
        raise CandidateRepairError("allowed_imports contains disallowed imports.")
    source = payload.get("repaired_tool_source")
    if not isinstance(source, str) or f"def {REQUIRED_ENTRYPOINT}" not in source:
        raise CandidateRepairError("repaired_tool_source must define rank_candidates.")
    for field in ("repair_summary", "public_safe_boundary_statement"):
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
            raise CandidateRepairError(f"{field} must be a non-empty string.")
    for field in ("changed_lines_or_components", "expected_remaining_risks"):
        if not isinstance(payload.get(field), list):
            raise CandidateRepairError(f"{field} must be a list.")
    assert_payload_public_safe(payload, label="candidate_repair_payload")
    return payload


def _validate_static_self_audit(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CandidateRepairError("static_self_audit is required and must be an object.")
    missing = sorted(set(REQUIRED_STATIC_SELF_AUDIT_SCHEMA) - set(value))
    if missing:
        raise CandidateRepairError(f"static_self_audit missing required fields: {missing}")
    fake_risk = str(value.get("fake_uncertainty_risk", "")).strip().lower()
    if fake_risk not in {"low", "medium", "high"}:
        raise CandidateRepairError("fake_uncertainty_risk must be low, medium, or high.")
    if str(value.get("hidden_y_leakage_self_check", "")).strip().lower() != "pass":
        raise CandidateRepairError("hidden_y_leakage_self_check must be pass.")
    if value.get("uses_only_observed_y") is not True:
        raise CandidateRepairError("uses_only_observed_y must be true.")
    if value.get("uses_only_public_candidate_features") is not True:
        raise CandidateRepairError("uses_only_public_candidate_features must be true.")
    explanation = str(value.get("why_this_is_sequential_optimizer_not_static_ranker", "")).strip()
    if not explanation:
        raise CandidateRepairError("why_this_is_sequential_optimizer_not_static_ranker must be non-empty.")
    return {
        "is_only_predicted_yield": bool(value.get("is_only_predicted_yield")),
        "has_explicit_exploitation": bool(value.get("has_explicit_exploitation")),
        "has_explicit_exploration_or_uncertainty_or_novelty": bool(
            value.get("has_explicit_exploration_or_uncertainty_or_novelty")
        ),
        "has_finite_pool_selection_policy": bool(value.get("has_finite_pool_selection_policy")),
        "has_update_or_state_policy": bool(value.get("has_update_or_state_policy")),
        "handles_small_n": bool(value.get("handles_small_n")),
        "handles_mixed_numeric_categorical_features": bool(value.get("handles_mixed_numeric_categorical_features")),
        "avoids_duplicate_or_near_duplicate_recommendations": bool(
            value.get("avoids_duplicate_or_near_duplicate_recommendations")
        ),
        "fake_uncertainty_risk": fake_risk,
        "hidden_y_leakage_self_check": "pass",
        "uses_only_observed_y": True,
        "uses_only_public_candidate_features": True,
        "why_this_is_sequential_optimizer_not_static_ranker": explanation,
        "static_ranker_risk": bool(value.get("is_only_predicted_yield")),
    }


def failed_stage_and_error(verifier_result: dict[str, Any]) -> tuple[str, str]:
    """Extract a concise failed verifier stage and public-safe error message."""

    checks = verifier_result.get("checks", {}) if isinstance(verifier_result.get("checks"), dict) else {}
    failed_checks = [str(item) for item in verifier_result.get("failed_checks", [])]
    for stage in failed_checks:
        detail = checks.get(stage)
        if isinstance(detail, dict):
            error = detail.get("error") or detail.get("reason") or detail.get("error_type")
            if error:
                return stage, sanitize_text(str(error))[:800]
    for stage, detail in checks.items():
        if isinstance(detail, dict) and not detail.get("passed", True):
            error = detail.get("error") or detail.get("reason") or detail.get("error_type")
            return str(stage), sanitize_text(str(error or verifier_result.get("reason", "")))[:800]
    return (
        failed_checks[0] if failed_checks else "unknown",
        sanitize_text(str(verifier_result.get("reason", "")))[:800],
    )


def write_repair_verifier_artifacts(
    *,
    output_dir: str | Path,
    repair_result: RepairedCandidateToolResult,
    verifier_report: dict[str, Any],
    accepted_for_quality_evaluation: bool,
) -> dict[str, Any]:
    """Persist repair verifier report and final repair record."""

    root = ensure_output_dir(output_dir) / "candidate_tool_repairs"
    paths = _repair_paths(root, repair_result.original_tool_id)
    report = dict(verifier_report)
    report["original_tool_id"] = repair_result.original_tool_id
    report["repaired_tool_id"] = repair_result.repaired_tool_id
    report["repair_attempt"] = int(repair_result.repair_attempt)
    _write_json(paths["verifier"], report)
    record = repair_result.to_record()
    record.update(
        {
            "verifier_status": "passed" if verifier_report.get("passed") else "failed",
            "accepted_for_quality_evaluation": bool(accepted_for_quality_evaluation),
            "repair_verifier_report_file": str(paths["verifier"]),
            "verifier_reason": verifier_report.get("reason"),
            "verifier_failed_checks": list(verifier_report.get("failed_checks", [])),
        }
    )
    _write_json(paths["record"], record)
    return record


def _repair_paths(root: Path, original_tool_id: str) -> dict[str, Path]:
    safe = _safe_filename(original_tool_id)
    return {
        "prompt": root / f"{safe}_repair_prompt.md",
        "request": root / f"{safe}_repair_request.json",
        "raw": root / f"{safe}_repair_raw_response.json",
        "source": root / f"{safe}_repaired_source.py",
        "parse_report": root / f"{safe}_repair_parse_report.json",
        "verifier": root / f"{safe}_repair_verifier_report.json",
        "record": root / f"{safe}_repair_record.json",
    }


def _compact_design(design: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "design_id",
        "family",
        "components",
        "method_primitives_used",
        "method_primitives_not_used",
        "state_variables",
        "exploitation_term",
        "exploration_or_anti_collapse_term",
        "small_n_fallback",
        "mixed_variable_handling",
        "why_this_is_not_a_static_ranker",
        "static_self_audit",
        "why_suitable_for_current_observed_data",
        "expected_failure_modes",
        "failure_modes",
        "public_safe_boundary_statement",
    }
    return {key: design.get(key) for key in keep if key in design}


def _safe_boundary_summary(boundary: list[str] | tuple[str, ...] | str | None) -> list[str]:
    _ = boundary
    return [
        "Use only public candidate features and already revealed observations.",
        "Do not use non-public outcomes, retrospective rankings, comparator diagnostics, private ID mappings, score-cache artifacts, answer-key material, credentials, or internal evaluator state.",
        "Do not perform evaluator reveals during candidate repair or screening.",
    ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_payload_public_safe(payload, label=f"candidate_repair_artifact:{path.name}")
    path.write_text(
        json.dumps(sanitize_payload(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _safe_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return cleaned[:80] or "candidate_tool"
