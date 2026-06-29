"""Artifact logging skeleton for full-pool ResearchToolAgent runs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from research_tool_agent_full_pool.decision_artifacts import (
    ALLOWED_SCHEMA_TOKENS_WITH_FORBIDDEN_SUBSTRINGS,
    FORBIDDEN_DECISION_TERMS,
    build_artifact_manifest,
    scan_decision_artifacts,
)

DEFAULT_SCAN_TERMS: tuple[str, ...] = (
    "Authorization",
    "Bearer",
    "apikey",
    "COMMONSTACK_API_KEY",
    "api_key",
    "candidate_scores.csv",
    "oracle_rank",
    "BO rank",
    "BO top-K",
    "reference_acquisition_score",
    "reference_acquisition_value",
    "reference_predictive_mean",
    "reference_predictive_std",
    "bo_acquisition_score",
    "bo_predictive_mean",
    "bo_predictive_std",
    "hidden_yield",
    "unobserved_y",
    "turnover",
    "raw_row_index",
    "source_path",
    "csebo_harness",
    "full-pool rank",
)

REDACTION_TERMS: tuple[str, ...] = (
    *DEFAULT_SCAN_TERMS,
    "Authorization",
    "Bearer",
    "BOReferencePolicy",
    "evaluation.bo_reference",
    "candidate_scores.csv",
    "bo_rank",
    "bo_top_k",
    "reference_acquisition_score",
    "reference_acquisition_value",
    "reference_predictive_mean",
    "reference_predictive_std",
    "bo_acquisition_score",
    "bo_predictive_mean",
    "bo_predictive_std",
    "oracle_rank",
    "hidden_y",
    "unobserved_y",
    "raw_row_index",
    "source_path",
    "full-pool rank",
)

DECISION_ARTIFACT_REDACTION_TERMS: tuple[str, ...] = tuple(
    term for term, _category in FORBIDDEN_DECISION_TERMS
)


def ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def write_config(output_dir: str | Path, config: Any) -> None:
    """Write run config without API secrets."""

    output_path = ensure_output_dir(output_dir)
    payload = asdict(config) if is_dataclass(config) else dict(config)
    payload = _drop_secret_named_keys(payload)
    _write_json(output_path / "config.json", payload)


def initialize_artifact_files(output_dir: str | Path) -> None:
    """Create/truncate Step 1 JSONL artifacts for a fresh smoke run."""

    output_path = ensure_output_dir(output_dir)
    for name in (
        "round_summaries.jsonl",
        "full_pool_decisions.jsonl",
        "generated_tool_outputs.jsonl",
        "fallback_events.jsonl",
        "full_pool_candidate_view_audit.jsonl",
        "generated_tool_requests.jsonl",
        "parsed_tool_synthesis.jsonl",
        "static_check_reports.jsonl",
        "sandbox_reports.jsonl",
        "tool_state_by_round.jsonl",
        "tool_feedback_reports.jsonl",
        "patch_decisions.jsonl",
        "posthoc_selected_rank_metrics.jsonl",
    ):
        output_path.joinpath(name).write_text("", encoding="utf-8")
    for name in (
        "tool_feedback_report.json",
        "patch_decision.json",
        "patch_prompt.json",
        "patch_response.json",
        "patch_verifier_report.json",
        "patch_acceptance_record.json",
        "active_tool_pointer.json",
        "patch_research_manifest.json",
        "patch_research_context.json",
        "patch_research_query_log.jsonl",
        "patch_research_source_log.jsonl",
        "patch_research_card_safety_audit.json",
        "patch_research_freeze_summary.json",
        "patch_accepted_cards.jsonl",
        "patch_rejected_cards.jsonl",
    ):
        path = output_path.joinpath(name)
        if path.exists() and path.is_file():
            path.unlink()
    for path in output_path.glob("old_tool_v*.py"):
        if path.is_file():
            path.unlink()
    for path in output_path.glob("patched_tool_candidate_v*.py"):
        if path.is_file():
            path.unlink()
    for subdir_name in ("generated_tools", "raw_llm_outputs", "tool_patches", "patch_research"):
        subdir = output_path.joinpath(subdir_name)
        subdir.mkdir(parents=True, exist_ok=True)
        for path in subdir.rglob("*"):
            if path.is_file():
                path.unlink()


def initialize_research_enabled_artifact_files(output_dir: str | Path) -> None:
    """Create/truncate Step 8B research-enabled smoke artifacts."""

    output_path = ensure_output_dir(output_dir)
    for name in (
        "research_input_audit.jsonl",
        "round_summaries.jsonl",
        "full_pool_decisions.jsonl",
        "generated_tool_outputs.jsonl",
        "fallback_events.jsonl",
        "full_pool_candidate_view_audit.jsonl",
        "generated_tool_requests.jsonl",
        "parsed_tool_synthesis.jsonl",
        "static_check_reports.jsonl",
        "sandbox_reports.jsonl",
        "tool_state_by_round.jsonl",
        "state_updater_requests.jsonl",
        "state_updater_outputs.jsonl",
        "parsed_state_updates.jsonl",
        "tool_feedback_reports.jsonl",
        "patch_decisions.jsonl",
        "posthoc_selected_rank_metrics.jsonl",
    ):
        output_path.joinpath(name).write_text("", encoding="utf-8")
    for name in (
        "research_enabled_smoke_summary.json",
        "research_enabled_smoke_summary.md",
        "artifact_scan_summary.json",
        "tool_feedback_report.json",
        "patch_decision.json",
        "patch_prompt.json",
        "patch_response.json",
        "patch_verifier_report.json",
        "patch_acceptance_record.json",
        "active_tool_pointer.json",
        "patch_research_manifest.json",
        "patch_research_context.json",
        "patch_research_query_log.jsonl",
        "patch_research_source_log.jsonl",
        "patch_research_card_safety_audit.json",
        "patch_research_freeze_summary.json",
        "patch_accepted_cards.jsonl",
        "patch_rejected_cards.jsonl",
    ):
        path = output_path.joinpath(name)
        if path.exists() and path.is_file():
            path.unlink()
    for path in output_path.glob("old_tool_v*.py"):
        if path.is_file():
            path.unlink()
    for path in output_path.glob("patched_tool_candidate_v*.py"):
        if path.is_file():
            path.unlink()
    for subdir_name in ("generated_tools", "raw_llm_outputs", "tool_patches", "patch_research"):
        subdir = output_path.joinpath(subdir_name)
        subdir.mkdir(parents=True, exist_ok=True)
        for path in subdir.rglob("*"):
            if path.is_file():
                path.unlink()


def initialize_fixed_baseline_artifact_files(output_dir: str | Path) -> None:
    """Create/truncate fixed baseline JSONL artifacts for a fresh run."""

    output_path = ensure_output_dir(output_dir)
    for name in (
        "round_summaries.jsonl",
        "full_pool_decisions.jsonl",
        "fixed_tool_outputs.jsonl",
        "bo_reference_outputs.jsonl",
        "fallback_events.jsonl",
        "full_pool_candidate_view_audit.jsonl",
        "validator_reports.jsonl",
        "posthoc_selected_rank_metrics.jsonl",
    ):
        output_path.joinpath(name).write_text("", encoding="utf-8")


def write_memory(output_dir: str | Path, memory_text: str) -> None:
    """Write `memory.md` for a run without secrets or hidden candidate outcomes."""

    ensure_output_dir(output_dir).joinpath("memory.md").write_text(memory_text, encoding="utf-8")


def write_strategy_state(output_dir: str | Path, strategy_state: dict[str, Any]) -> None:
    """Write public-safe strategy state."""

    _write_json(ensure_output_dir(output_dir) / "strategy_state.json", strategy_state)


def write_tool_state(output_dir: str | Path, tool_state: dict[str, Any]) -> None:
    """Write public-safe generated-tool state."""

    _write_json(ensure_output_dir(output_dir) / "tool_state.json", tool_state)


def write_tool_state_by_round(output_dir: str | Path, payload: dict[str, Any]) -> None:
    """Append per-round public-safe tool state."""

    _append_jsonl(ensure_output_dir(output_dir) / "tool_state_by_round.jsonl", payload)


def write_observed_evidence(
    output_dir: str | Path,
    text: str,
    *,
    round_index: int | None = None,
) -> None:
    """Write latest observed evidence and optional deterministic round snapshot."""

    output_path = ensure_output_dir(output_dir)
    output_path.joinpath("observed_evidence.md").write_text(text, encoding="utf-8")
    if round_index is not None:
        snapshot_dir = output_path / "observed_evidence_by_round"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir.joinpath(f"round_{int(round_index):03d}.md").write_text(text, encoding="utf-8")


def write_agent_context(
    output_dir: str | Path,
    text: str,
    *,
    round_index: int | None = None,
) -> None:
    """Write latest assembled agent context and optional round snapshot."""

    output_path = ensure_output_dir(output_dir)
    output_path.joinpath("agent_context.md").write_text(text, encoding="utf-8")
    if round_index is not None:
        snapshot_dir = output_path / "agent_context_by_round"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir.joinpath(f"round_{int(round_index):03d}.md").write_text(text, encoding="utf-8")


def write_round_summary(output_dir: str | Path, summary: dict[str, Any]) -> None:
    """Write one public-safe round summary."""

    _append_jsonl(ensure_output_dir(output_dir) / "round_summaries.jsonl", summary)


def write_full_pool_decision(output_dir: str | Path, decision: dict[str, Any]) -> None:
    """Append one full-pool decision artifact."""

    _append_jsonl(ensure_output_dir(output_dir) / "full_pool_decisions.jsonl", decision)


def write_generated_tool_output(output_dir: str | Path, output: dict[str, Any]) -> None:
    """Append one generated-tool output artifact."""

    _append_jsonl(ensure_output_dir(output_dir) / "generated_tool_outputs.jsonl", output)


def write_fixed_tool_output(output_dir: str | Path, output: dict[str, Any]) -> None:
    """Append one fixed baseline tool output artifact."""

    _append_jsonl(ensure_output_dir(output_dir) / "fixed_tool_outputs.jsonl", sanitize_payload(output))


def write_validator_report(output_dir: str | Path, report: dict[str, Any]) -> None:
    """Append one fixed-tool validator report."""

    _append_jsonl(ensure_output_dir(output_dir) / "validator_reports.jsonl", sanitize_payload(report))


def write_bo_reference_output(output_dir: str | Path, output: dict[str, Any]) -> None:
    """Append one isolated reference-baseline selection summary."""

    _append_jsonl(ensure_output_dir(output_dir) / "bo_reference_outputs.jsonl", sanitize_payload(output))


def write_posthoc_rank_metric(output_dir: str | Path, output: dict[str, Any]) -> None:
    """Append one posthoc-only selected-rank metric."""

    _append_jsonl(ensure_output_dir(output_dir) / "posthoc_selected_rank_metrics.jsonl", sanitize_payload(output))


def write_generated_tool_request(output_dir: str | Path, request: dict[str, Any]) -> None:
    """Append a sanitized generated-tool request summary."""

    _append_jsonl(ensure_output_dir(output_dir) / "generated_tool_requests.jsonl", sanitize_payload(request))


def write_tool_synthesis_prompt_artifacts(
    output_dir: str | Path,
    *,
    round_index: int,
    attempt_index: int,
    prompt_text: str,
    request_summary: dict[str, Any],
) -> dict[str, str]:
    """Persist the full prompt text and request metadata for audit."""

    from research_tool_agent_full_pool.decision_artifacts import assert_payload_public_safe
    from research_tool_agent_full_pool.diagnostics import sha256_text

    output_path = ensure_output_dir(output_dir)
    prompt_hash = sha256_text(str(prompt_text))
    request_payload = {
        **dict(request_summary),
        "round_index": int(round_index),
        "attempt_index": int(attempt_index),
        "prompt_hash": str(request_summary.get("prompt_hash") or prompt_hash),
        "prompt_character_count": len(str(prompt_text)),
    }
    if request_payload["prompt_hash"] != prompt_hash:
        raise ValueError("Prompt artifact hash does not match prompt_text.")
    assert_payload_public_safe(
        {"prompt_text": str(prompt_text), "request_summary": request_payload},
        label="tool_synthesis_prompt_artifact",
    )
    prompt_name = f"tool_synthesis_prompt_round_{int(round_index):03d}_attempt_{int(attempt_index):02d}.md"
    request_name = f"tool_synthesis_request_round_{int(round_index):03d}_attempt_{int(attempt_index):02d}.json"
    prompt_path = output_path / prompt_name
    request_path = output_path / request_name
    prompt_path.write_text(str(prompt_text), encoding="utf-8")
    _write_json(request_path, sanitize_payload(request_payload))
    return {
        "prompt_path": str(prompt_path),
        "request_path": str(request_path),
        "prompt_hash": prompt_hash,
    }


def write_research_input_audit(output_dir: str | Path, audit: dict[str, Any]) -> None:
    """Append one manifest-only research input audit row."""

    _append_jsonl(ensure_output_dir(output_dir) / "research_input_audit.jsonl", sanitize_payload(audit))


def write_state_updater_request(output_dir: str | Path, request: dict[str, Any]) -> None:
    """Append a sanitized state-updater request summary."""

    _append_jsonl(ensure_output_dir(output_dir) / "state_updater_requests.jsonl", sanitize_payload(request))


def write_state_updater_output(
    output_dir: str | Path,
    *,
    round_index: int,
    raw_text: str,
    attempt: str = "initial",
) -> None:
    """Append sanitized raw state-updater output text."""

    _append_jsonl(
        ensure_output_dir(output_dir) / "state_updater_outputs.jsonl",
        {
            "round_index": int(round_index),
            "attempt": str(attempt),
            "raw_text": sanitize_text(
                raw_text,
                extra_secret_terms=DECISION_ARTIFACT_REDACTION_TERMS,
            ),
        },
    )


def write_parsed_state_update(output_dir: str | Path, payload: dict[str, Any]) -> None:
    """Append sanitized state-updater parser metadata and parsed update."""

    _append_jsonl(ensure_output_dir(output_dir) / "parsed_state_updates.jsonl", sanitize_payload(payload))


def write_raw_llm_output(
    output_dir: str | Path,
    *,
    round_index: int,
    attempt_index: int,
    raw_text: str,
) -> None:
    """Write sanitized raw LLM output for parser debugging."""

    output_path = ensure_output_dir(output_dir) / "raw_llm_outputs"
    output_path.mkdir(parents=True, exist_ok=True)
    safe = sanitize_text(raw_text, extra_secret_terms=DECISION_ARTIFACT_REDACTION_TERMS)
    output_path.joinpath(f"round_{round_index:03d}_attempt_{attempt_index:02d}.json").write_text(
        safe,
        encoding="utf-8",
    )
    _write_json(
        output_path / f"raw_llm_response_round_{round_index:03d}_attempt_{attempt_index:02d}.json",
        {
            "round_index": int(round_index),
            "attempt_index": int(attempt_index),
            "raw_text": safe,
        },
    )


def write_parsed_tool_synthesis(output_dir: str | Path, payload: dict[str, Any]) -> None:
    """Append sanitized parsed tool synthesis metadata."""

    _append_jsonl(ensure_output_dir(output_dir) / "parsed_tool_synthesis.jsonl", sanitize_payload(payload))


def write_static_check_report(output_dir: str | Path, report: dict[str, Any]) -> None:
    """Append a generated-tool static-check report."""

    _append_jsonl(ensure_output_dir(output_dir) / "static_check_reports.jsonl", sanitize_payload(report))


def write_sandbox_report(output_dir: str | Path, report: dict[str, Any]) -> None:
    """Append a generated-tool sandbox-validation report."""

    _append_jsonl(ensure_output_dir(output_dir) / "sandbox_reports.jsonl", sanitize_payload(report))


def write_generated_tool_source(
    output_dir: str | Path,
    *,
    round_index: int,
    tool_name: str,
    source: str,
) -> Path:
    """Write accepted generated tool source after validation."""

    output_path = ensure_output_dir(output_dir) / "generated_tools"
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / f"round_{round_index:03d}_{tool_name}.py"
    path.write_text(sanitize_text(source), encoding="utf-8")
    return path


def write_fallback_event(output_dir: str | Path, event: dict[str, Any]) -> None:
    """Append one fallback event artifact."""

    _append_jsonl(ensure_output_dir(output_dir) / "fallback_events.jsonl", event)


def write_candidate_view_audit(output_dir: str | Path, audit: dict[str, Any]) -> None:
    """Append one full-pool candidate view audit event."""

    _append_jsonl(ensure_output_dir(output_dir) / "full_pool_candidate_view_audit.jsonl", audit)


def write_candidate_id_map(output_dir: str | Path, candidate_df: pd.DataFrame) -> None:
    """Write private display-to-internal mapping for evaluator handoff."""

    mapping = candidate_df.attrs.get("display_to_internal_id", {})
    rows = [
        {
            "display_candidate_id": str(display_id),
            "internal_candidate_id": str(internal_id),
        }
        for display_id, internal_id in sorted(mapping.items())
    ]
    _write_json(ensure_output_dir(output_dir) / "candidate_id_map.private.json", {"rows": rows})


def scan_artifacts(
    output_dir: str | Path,
    *,
    terms: tuple[str, ...] = DEFAULT_SCAN_TERMS,
) -> dict[str, Any]:
    """Scan text artifacts and return forbidden-term counts without snippets."""

    output_path = ensure_output_dir(output_dir)
    term_counts = {term: 0 for term in terms}
    scanned_files = 0
    for path in output_path.rglob("*"):
        if not path.is_file() or path.name == "artifact_scan_summary.json":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned_files += 1
        lowered = text.lower()
        for term in terms:
            term_counts[term] += lowered.count(term.lower())
    total = int(sum(term_counts.values()))
    return {
        "scanned_file_count": scanned_files,
        "total_forbidden_term_matches": total,
        "term_counts": term_counts,
    }


def write_artifact_scan_summary(output_dir: str | Path, scan_result: dict[str, Any]) -> None:
    """Write safety scan summary for generated artifacts."""

    total = int(scan_result.get("total_forbidden_term_matches", 0))
    term_counts = dict(scan_result.get("term_counts", {}))
    matched_term_count = sum(1 for count in term_counts.values() if int(count) > 0)
    payload = {
        "status": "pass" if total == 0 else "fail",
        "scanned_file_count": int(scan_result.get("scanned_file_count", 0)),
        "total_forbidden_term_matches": total,
        "matched_term_count": matched_term_count,
    }
    _write_json(ensure_output_dir(output_dir) / "artifact_scan_summary.json", payload)


def write_artifact_manifest(output_dir: str | Path) -> dict[str, Any]:
    """Write the Step 9B artifact-classification manifest."""

    output_path = ensure_output_dir(output_dir)
    manifest = build_artifact_manifest(output_path)
    _write_json(output_path / "artifact_manifest.json", manifest)
    return manifest


def write_decision_artifact_scan_summary(output_dir: str | Path) -> dict[str, Any]:
    """Scan decision-facing/log artifacts without snippets or secret values."""

    output_path = ensure_output_dir(output_dir)
    scan_result = scan_decision_artifacts(output_path)
    payload = {
        "status": scan_result.get("status", "fail"),
        "scanned_file_count": int(scan_result.get("scanned_file_count", 0)),
        "total_forbidden_match_count": int(scan_result.get("total_forbidden_match_count", 0)),
        "files": scan_result.get("files", []),
    }
    _write_json(output_path / "decision_artifact_scan_summary.json", payload)
    return payload


def write_api_smoke_summary(output_dir: str | Path, summary: dict[str, Any]) -> None:
    """Write final API smoke summary."""

    _write_json(ensure_output_dir(output_dir) / "api_smoke_summary.json", sanitize_payload(summary))


def write_fixed_baseline_summary(output_dir: str | Path, summary: dict[str, Any]) -> None:
    """Write final fixed baseline summary JSON."""

    _write_json(ensure_output_dir(output_dir) / "fixed_tool_baseline_summary.json", sanitize_payload(summary))


def write_fixed_baseline_summary_markdown(output_dir: str | Path, text: str) -> None:
    """Write final fixed baseline summary Markdown without scan-sensitive text."""

    ensure_output_dir(output_dir).joinpath("fixed_tool_baseline_summary.md").write_text(
        sanitize_text(text),
        encoding="utf-8",
    )


def write_matched_eval_summary(output_dir: str | Path, summary: dict[str, Any]) -> None:
    """Write final matched-eval summary JSON."""

    _write_json(ensure_output_dir(output_dir) / "matched_eval_summary.json", sanitize_payload(summary))


def write_matched_eval_summary_markdown(output_dir: str | Path, text: str) -> None:
    """Write final matched-eval summary Markdown."""

    ensure_output_dir(output_dir).joinpath("matched_eval_summary.md").write_text(
        sanitize_text(text),
        encoding="utf-8",
    )


def write_research_tool_light_audit(output_dir: str | Path, audit: dict[str, Any]) -> None:
    """Write lightweight generated-tool audit JSON."""

    _write_json(ensure_output_dir(output_dir) / "research_tool_light_audit.json", sanitize_payload(audit))


def write_research_tool_light_audit_markdown(output_dir: str | Path, text: str) -> None:
    """Write lightweight generated-tool audit Markdown."""

    ensure_output_dir(output_dir).joinpath("research_tool_light_audit.md").write_text(
        sanitize_text(text),
        encoding="utf-8",
    )


def write_matched_artifact_scan_summary(output_dir: str | Path, scan_result: dict[str, Any]) -> None:
    """Write matched-eval scan counts without forbidden term names/snippets."""

    payload = {
        "status": scan_result.get("status", "fail"),
        "scanned_file_count": int(scan_result.get("scanned_file_count", 0)),
        "decision_facing_match_count": int(scan_result.get("decision_facing_match_count", 0)),
        "isolated_reference_match_count": int(scan_result.get("isolated_reference_match_count", 0)),
        "posthoc_only_match_count": int(scan_result.get("posthoc_only_match_count", 0)),
        "sensitive_match_count": int(scan_result.get("sensitive_match_count", 0)),
        "total_match_count": int(scan_result.get("total_match_count", 0)),
    }
    _write_json(ensure_output_dir(output_dir) / "artifact_scan_summary.json", payload)


def sanitize_text(text: Any, *, extra_secret_terms: tuple[str, ...] = ()) -> str:
    """Redact scan-sensitive terms and key-like text before artifact writes."""

    safe = str(text)
    protected: dict[str, str] = {}
    for index, token in enumerate(ALLOWED_SCHEMA_TOKENS_WITH_FORBIDDEN_SUBSTRINGS):
        placeholder = f"__CODEX_ALLOWED_SCHEMA_TOKEN_{index}__"
        if token in safe:
            safe = safe.replace(token, placeholder)
            protected[placeholder] = token
    safe = re.sub(
        r"(?i)authorization\s*:?\s*bearer\s+\S+",
        "[redacted]",
        safe,
    )
    safe = re.sub(r"(?i)bearer\s+\S+", "[redacted]", safe)
    safe = re.sub(r"(?i)api[_-]?key\s*[:=]?\s+\S+", "[redacted]", safe)
    for term in (*REDACTION_TERMS, *extra_secret_terms):
        if not term:
            continue
        safe = _replace_case_insensitive(safe, str(term), "[redacted]")
    for placeholder, token in protected.items():
        safe = safe.replace(placeholder, token)
    return safe


def sanitize_payload(payload: Any, *, extra_secret_terms: tuple[str, ...] = ()) -> Any:
    """Recursively sanitize strings and secret-named keys."""

    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            key_text = str(key)
            if _is_secret_named_key(key_text):
                continue
            safe_key = sanitize_text(key_text, extra_secret_terms=extra_secret_terms)
            sanitized[safe_key] = sanitize_payload(value, extra_secret_terms=extra_secret_terms)
        return sanitized
    if isinstance(payload, list):
        return [sanitize_payload(item, extra_secret_terms=extra_secret_terms) for item in payload]
    if isinstance(payload, tuple):
        return [sanitize_payload(item, extra_secret_terms=extra_secret_terms) for item in payload]
    if isinstance(payload, str):
        return sanitize_text(payload, extra_secret_terms=extra_secret_terms)
    return payload


def _drop_secret_named_keys(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if not _is_secret_named_key(str(key))
    }


def _is_secret_named_key(key: str) -> bool:
    lowered = key.lower()
    return any(term in lowered for term in ("api_key", "apikey", "authorization", "bearer"))


def _replace_case_insensitive(text: str, needle: str, replacement: str) -> str:
    lowered = text.lower()
    lower_needle = needle.lower()
    if not lower_needle:
        return text
    pieces: list[str] = []
    start = 0
    while True:
        index = lowered.find(lower_needle, start)
        if index < 0:
            pieces.append(text[start:])
            break
        pieces.append(text[start:index])
        pieces.append(replacement)
        start = index + len(needle)
    return "".join(pieces)
