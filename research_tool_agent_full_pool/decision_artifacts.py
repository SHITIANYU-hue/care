"""Artifact classification and decision-facing leakage scanning."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any


TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".py"}

ALLOWED_SCHEMA_TOKENS_WITH_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "hidden_y_leakage_self_check",
)

FORBIDDEN_DECISION_TERMS: tuple[tuple[str, str], ...] = (
    ("posthoc_selected_full_pool_rank", "posthoc_selected_full_pool_rank"),
    ("posthoc_selected_true_rank", "posthoc_selected_true_rank"),
    ("full_pool_true_rank", "full_pool_true_rank"),
    ("oracle_rank", "oracle_rank"),
    ("oracle", "oracle"),
    ("global_rank", "global_rank"),
    ("full_pool_rank", "full_pool_rank"),
    ("selected_full_pool_rank", "selected_full_pool_rank"),
    ("full-pool rank", "full_pool_rank"),
    ("full-pool true rank", "full_pool_true_rank"),
    ("true rank", "true_rank"),
    ("top percentile", "top_percentile"),
    ("regret", "regret"),
    ("hidden_y", "hidden_y"),
    ("hidden_yield", "hidden_yield"),
    ("unobserved_y", "unobserved_y"),
    ("turnover", "turnover"),
    ("candidate_scores.csv", "candidate_scores_csv"),
    ("all_remaining_y", "all_remaining_y"),
    ("all remaining y", "all_remaining_y"),
    ("all remaining target", "all_remaining_target"),
    ("hidden outcome table", "hidden_outcome_table"),
    ("candidate_yield_mapping", "candidate_yield_mapping"),
    ("candidate-yield mapping", "candidate_yield_mapping"),
    ("yield_by_candidate", "candidate_yield_mapping"),
    ("benchmark answer key", "benchmark_answer_key"),
    ("answer_key", "benchmark_answer_key"),
    ("evaluator_internals", "evaluator_internals"),
    ("evaluator internals", "evaluator_internals"),
    ("private candidate map", "private_candidate_map"),
    ("private candidate id map", "private_candidate_map"),
    ("candidate_id_map", "private_candidate_map"),
    ("reference_acquisition_score", "reference_acquisition_score"),
    ("reference_acquisition_value", "reference_acquisition_value"),
    ("reference_predictive_mean", "reference_predictive_mean"),
    ("reference_predictive_std", "reference_predictive_std"),
    ("bo_acquisition_score", "bo_acquisition_score"),
    ("bo_predictive_mean", "bo_predictive_mean"),
    ("bo_predictive_std", "bo_predictive_std"),
    ("reference_bo_predictions", "bo_reference_predictions"),
    ("reference_bo_acquisition", "bo_reference_acquisition"),
    ("reference_bo_ranks", "bo_reference_ranks"),
    ("bo_reference_predictions", "bo_reference_predictions"),
    ("bo_reference_acquisition", "bo_reference_acquisition"),
    ("bo_reference_ranks", "bo_reference_ranks"),
    ("bo rank", "bo_rank"),
    ("bo top-k", "bo_top_k"),
    ("BOReferencePolicy", "bo_reference_policy"),
    ("csebo_harness", "csebo_harness"),
    ("raw_row_index", "raw_row_index"),
    ("source_path", "source_path"),
    ("Authorization", "authorization"),
    ("Bearer", "bearer"),
    ("api_key", "api_key"),
    ("apikey", "apikey"),
    ("COMMONSTACK_API_KEY", "commonstack_api_key"),
)

DECISION_FACING_FILES = {
    "memory.md",
    "observed_evidence.md",
    "agent_context.md",
    "strategy_state.json",
    "tool_state.json",
}

DECISION_LOG_FILES = {
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
    "research_input_audit.jsonl",
    "validator_reports.jsonl",
    "fixed_tool_outputs.jsonl",
    "tool_feedback_report.json",
    "tool_feedback_reports.jsonl",
    "patch_decision.json",
    "patch_decisions.jsonl",
    "patch_prompt.json",
    "patch_response.json",
    "patch_verifier_report.json",
    "patch_acceptance_record.json",
    "active_tool_pointer.json",
    "patch_research_manifest.json",
    "patch_research_context.json",
    "patch_research_card_safety_audit.json",
    "patch_research_freeze_summary.json",
    "accepted_cards.jsonl",
    "research_manifest.json",
    "research_context.json",
    "live_research_freeze_summary.json",
}


class DecisionArtifactSafetyError(ValueError):
    """Raised when decision-facing state contains forbidden leakage."""


def classify_artifact_path(path: str | Path) -> str:
    """Classify one run artifact path relative to its output directory."""

    relative = Path(path)
    parts = tuple(part.lower() for part in relative.parts)
    name = relative.name
    lower_name = name.lower()
    if lower_name == "candidate_id_map.private.json":
        return "private_evaluator_only"
    if _is_research_artifact(parts, lower_name):
        return _classify_research_artifact(parts, lower_name)
    if lower_name.startswith("tool_synthesis_prompt_round_"):
        return "decision_log"
    if lower_name.startswith("tool_synthesis_request_round_"):
        return "decision_log"
    if "tool_patches" in parts or lower_name.startswith(("old_tool_v", "patched_tool_candidate_v")):
        return "decision_log"
    if lower_name in {"tool_feedback_report.json", "patch_decision.json"}:
        return "decision_log"
    if (
        "posthoc" in lower_name
        or "regret" in lower_name
        or lower_name == "bo_reference_outputs.jsonl"
        or "bo_reference_full_pool" in parts
    ):
        return "posthoc_only"
    if name in DECISION_FACING_FILES:
        return "decision_facing"
    if "candidate_tool_repairs" in parts:
        return "decision_log"
    if any(
        part in {"generated_tools", "observed_evidence_by_round", "agent_context_by_round"}
        for part in parts
    ):
        return "decision_facing"
    if "raw_llm_outputs" in parts:
        return "rejected_llm_output"
    if name in DECISION_LOG_FILES:
        return "decision_log"
    if lower_name in {"artifact_manifest.json", "artifact_scan_summary.json", "config.json"}:
        return "report_only"
    if "summary" in lower_name or "audit" in lower_name or lower_name.endswith(".csv"):
        return "report_only"
    return "report_only"


def build_artifact_manifest(output_dir: str | Path) -> dict[str, Any]:
    """Build a manifest classifying existing artifacts without reading them."""

    root = Path(output_dir)
    artifacts: list[dict[str, str]] = []
    class_counts: Counter[str] = Counter()
    if root.exists():
        for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            artifact_class = classify_artifact_path(relative)
            class_counts[artifact_class] += 1
            artifacts.append({"path": str(relative).replace("\\", "/"), "class": artifact_class})
    return {
        "manifest_version": "step9b.artifact_manifest.v1",
        "artifact_classes": {
            "decision_facing": "may be reused by the agent",
            "decision_log": "records public-safe decisions and diagnostics",
            "rejected_llm_output": "raw rejected model output kept only for debugging",
            "private_evaluator_only": "not provided to the agent",
            "posthoc_only": "created after decisions for evaluation",
            "research_audit": "query/source/rejected research audit logs not provided to prompts",
            "report_only": "human-readable or aggregate run reporting",
        },
        "class_counts": dict(sorted(class_counts.items())),
        "artifacts": artifacts,
    }


def scan_decision_artifacts(
    output_dir: str | Path,
    *,
    include_classes: tuple[str, ...] = ("decision_facing", "decision_log"),
) -> dict[str, Any]:
    """Scan accepted decision artifacts and separately report rejected outputs."""

    root = Path(output_dir)
    files: list[dict[str, Any]] = []
    rejected_output_files: list[dict[str, Any]] = []
    research_audit_files: list[dict[str, Any]] = []
    total = 0
    rejected_total = 0
    research_audit_total = 0
    scanned = 0
    rejected_scanned = 0
    research_audit_scanned = 0
    if root.exists():
        for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            relative = path.relative_to(root)
            artifact_class = classify_artifact_path(relative)
            is_rejected_output = artifact_class == "rejected_llm_output"
            is_research_audit = artifact_class == "research_audit"
            if artifact_class not in include_classes and not is_rejected_output and not is_research_audit:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            counts = forbidden_text_counts(text)
            count_sum = int(sum(counts.values()))
            if is_research_audit:
                research_audit_scanned += 1
                if count_sum:
                    research_audit_total += count_sum
                    research_audit_files.append(
                        {
                            "path": str(relative).replace("\\", "/"),
                            "class": artifact_class,
                            "forbidden_counts": dict(sorted(counts.items())),
                        }
                    )
                continue
            if is_rejected_output:
                rejected_scanned += 1
                if count_sum:
                    rejected_total += count_sum
                    rejected_output_files.append(
                        {
                            "path": str(relative).replace("\\", "/"),
                            "class": artifact_class,
                            "forbidden_counts": dict(sorted(counts.items())),
                        }
                    )
                continue
            scanned += 1
            if count_sum:
                total += count_sum
                files.append(
                    {
                        "path": str(relative).replace("\\", "/"),
                        "class": artifact_class,
                        "forbidden_counts": dict(sorted(counts.items())),
                    }
                )
    return {
        "status": "pass" if total == 0 else "fail",
        "scanned_file_count": scanned,
        "total_forbidden_match_count": total,
        "files": files,
        "rejected_output_scanned_file_count": rejected_scanned,
        "rejected_output_forbidden_match_count": rejected_total,
        "rejected_output_files": rejected_output_files,
        "research_audit_scanned_file_count": research_audit_scanned,
        "research_audit_forbidden_match_count": research_audit_total,
        "research_audit_files": research_audit_files,
    }


def forbidden_text_counts(text: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for term, category in FORBIDDEN_DECISION_TERMS:
        count = _forbidden_term_count(str(text), term)
        if count:
            counts[category] += int(count)
    return dict(counts)


def _forbidden_term_count(text: str, term: str) -> int:
    """Count forbidden terms, avoiding identifier-substring false positives."""

    for token in ALLOWED_SCHEMA_TOKENS_WITH_FORBIDDEN_SUBSTRINGS:
        text = re.sub(re.escape(token), "", text, flags=re.IGNORECASE)
    if term == "full_pool_rank":
        # Catch the exact forbidden field/phrase without matching legacy
        # selection-rule labels that merely combine full-pool and rank-1 words.
        return len(re.findall(r"(?<![A-Za-z0-9_])full_pool_rank(?![A-Za-z0-9_])", text, flags=re.IGNORECASE))
    if term == "full-pool rank":
        return len(re.findall(r"(?<![A-Za-z0-9_-])full-pool\s+rank(?![A-Za-z0-9_-])", text, flags=re.IGNORECASE))
    return str(text).lower().count(str(term).lower())


def scan_payload_forbidden(payload: Any, *, label: str = "payload") -> list[dict[str, Any]]:
    """Recursively scan dict/list/string payloads for forbidden terms."""

    hits: list[dict[str, Any]] = []
    _scan_payload(payload, path=label, hits=hits)
    return hits


def assert_payload_public_safe(payload: Any, *, label: str) -> None:
    hits = scan_payload_forbidden(payload, label=label)
    if hits:
        category_counts: Counter[str] = Counter()
        for hit in hits:
            category_counts[str(hit["category"])] += int(hit["count"])
        summary = ", ".join(f"{key}={value}" for key, value in sorted(category_counts.items()))
        raise DecisionArtifactSafetyError(f"{label} contains forbidden public-safety categories: {summary}")


def _scan_payload(payload: Any, *, path: str, hits: list[dict[str, Any]]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_counts = forbidden_text_counts(str(key))
            for category, count in key_counts.items():
                hits.append({"path": f"{path}.[key]", "category": category, "count": count})
            _scan_payload(value, path=f"{path}.{key}", hits=hits)
        return
    if isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            _scan_payload(item, path=f"{path}[{index}]", hits=hits)
        return
    if isinstance(payload, str):
        counts = forbidden_text_counts(payload)
        for category, count in counts.items():
            hits.append({"path": path, "category": category, "count": count})


def _is_research_artifact(parts: tuple[str, ...], lower_name: str) -> bool:
    if "live_research" in parts:
        return True
    if "patch_research" in parts:
        return True
    return lower_name in {
        "research_query_log.jsonl",
        "research_source_log.jsonl",
        "accepted_cards.jsonl",
        "rejected_cards.jsonl",
        "research_manifest.json",
        "research_context.json",
        "live_research_freeze_summary.json",
        "patch_research_query_log.jsonl",
        "patch_research_source_log.jsonl",
        "patch_accepted_cards.jsonl",
        "patch_rejected_cards.jsonl",
        "patch_research_manifest.json",
        "patch_research_context.json",
        "patch_research_freeze_summary.json",
    }


def _classify_research_artifact(parts: tuple[str, ...], lower_name: str) -> str:
    if (
        "rejected" in parts
        or lower_name in {
            "research_query_log.jsonl",
            "research_source_log.jsonl",
            "patch_research_query_log.jsonl",
            "patch_research_source_log.jsonl",
            "source_index.jsonl",
            "rejected_cards.jsonl",
            "patch_rejected_cards.jsonl",
            "research_card_safety_audit.json",
            "research_card_safety_audit.md",
            "patch_research_card_safety_audit.json",
        }
    ):
        return "research_audit"
    if (
        "cards" in parts
        or lower_name in {
            "accepted_cards.jsonl",
            "patch_accepted_cards.jsonl",
            "research_manifest.json",
            "research_context.json",
            "live_research_freeze_summary.json",
            "patch_research_manifest.json",
            "patch_research_context.json",
            "patch_research_freeze_summary.json",
        }
    ):
        return "decision_log"
    return "report_only"
