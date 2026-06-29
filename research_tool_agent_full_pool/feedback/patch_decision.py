"""Deterministic patch/reuse decision scaffold."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal

from research_tool_agent_full_pool.decision_artifacts import assert_payload_public_safe


PatchAction = Literal["reuse", "patch_without_search", "research_assisted_patch"]


@dataclass(frozen=True)
class PatchDecision:
    decision: PatchAction
    trigger_reason: str
    evidence_fields_used: list[str]
    requires_live_research: bool
    requires_tool_patch: bool
    safe_for_agent_prompt: bool
    notes: str
    created_by_harness_version: str = "batch1_safety_artifacts_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_patch_action(
    feedback_report: Any,
    tool_state: dict[str, Any] | None,
    strategy_state: dict[str, Any] | None,
    config: Any | None = None,
) -> PatchDecision:
    """Return a deterministic scaffold decision without replacing tool source."""

    report = _as_dict(feedback_report)
    tool_state = dict(tool_state or {})
    strategy_state = dict(strategy_state or {})
    patch_mode = str(getattr(config, "patch_mode", "decision_only"))
    if patch_mode == "disabled":
        decision = PatchDecision(
            decision="reuse",
            trigger_reason="patch_mode_disabled",
            evidence_fields_used=["config.patch_mode"],
            requires_live_research=False,
            requires_tool_patch=False,
            safe_for_agent_prompt=True,
            notes="Batch 1 records decisions only and keeps the active generated tool unchanged.",
        )
        assert_payload_public_safe(decision.to_dict(), label="patch_decision")
        return decision

    evidence: list[str] = []
    patch_needed = False
    reason = "valid_tool_output_reuse"
    if bool(report.get("fallback_used")):
        patch_needed = True
        reason = "fallback_used"
        evidence.append("feedback_report.fallback_used")
    if _bad_status(report.get("parser_status")):
        patch_needed = True
        reason = "parser_status_not_valid"
        evidence.append("feedback_report.parser_status")
    if _bad_status(report.get("static_check_status")):
        patch_needed = True
        reason = "static_check_status_not_valid"
        evidence.append("feedback_report.static_check_status")
    if _bad_status(report.get("sandbox_status")):
        patch_needed = True
        reason = "sandbox_status_not_valid"
        evidence.append("feedback_report.sandbox_status")
    if _diagnostics_are_severe(report.get("public_tool_diagnostics")):
        patch_needed = True
        reason = "severe_public_tool_diagnostics"
        evidence.append("feedback_report.public_tool_diagnostics")
    if _state_requests_patch(tool_state):
        patch_needed = True
        reason = "tool_state_consider_patch_later"
        evidence.append("tool_state")
    if _state_requests_patch(strategy_state):
        patch_needed = True
        reason = "strategy_state_consider_patch_later"
        evidence.append("strategy_state")

    if not patch_needed:
        decision = PatchDecision(
            decision="reuse",
            trigger_reason=reason,
            evidence_fields_used=evidence or ["feedback_report.fallback_used", "feedback_report.public_tool_diagnostics"],
            requires_live_research=False,
            requires_tool_patch=False,
            safe_for_agent_prompt=True,
            notes="Reuse the previously validated generated tool.",
        )
        assert_payload_public_safe(decision.to_dict(), label="patch_decision")
        return decision

    if _research_context_missing_or_stale(tool_state, strategy_state) or _repeated_patch_failures(strategy_state):
        decision = PatchDecision(
            decision="research_assisted_patch",
            trigger_reason="patch_needed_and_research_context_insufficient",
            evidence_fields_used=sorted(set([*evidence, "tool_state", "strategy_state"])),
            requires_live_research=True,
            requires_tool_patch=True,
            safe_for_agent_prompt=True,
            notes="Batch 1 records this decision only; live research and source replacement remain disabled.",
        )
    else:
        decision = PatchDecision(
            decision="patch_without_search",
            trigger_reason=reason,
            evidence_fields_used=sorted(set(evidence)),
            requires_live_research=False,
            requires_tool_patch=True,
            safe_for_agent_prompt=True,
            notes="Batch 1 records this decision only; active generated tool source is reused.",
        )
    assert_payload_public_safe(decision.to_dict(), label="patch_decision")
    return decision


def write_patch_decision(output_dir: str | Path, decision: PatchDecision) -> None:
    payload = decision.to_dict()
    assert_payload_public_safe(payload, label="patch_decision")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / "patch_decision.json", payload)
    _append_jsonl(output_path / "patch_decisions.jsonl", payload)


def _bad_status(status: Any) -> bool:
    text = str(status or "").lower()
    if text in {"", "not_available", "valid", "pass", "passed", "ok", "success", "reuse"}:
        return False
    return any(token in text for token in ("fail", "invalid", "error", "fallback", "reject"))


def _diagnostics_are_severe(diagnostics: Any) -> bool:
    text = json.dumps(diagnostics, sort_keys=True, default=str).lower()
    return any(
        token in text
        for token in (
            "severe",
            "invalid",
            "flat",
            "unstable",
            "nan",
            "nonfinite",
            "failed",
            "exception",
            "degenerate",
        )
    )


def _state_requests_patch(state: dict[str, Any]) -> bool:
    text = json.dumps(state, sort_keys=True, default=str).lower()
    return "consider_patch_later" in text or "patch_requested" in text


def _research_context_missing_or_stale(
    tool_state: dict[str, Any],
    strategy_state: dict[str, Any],
) -> bool:
    text = json.dumps({"tool_state": tool_state, "strategy_state": strategy_state}, sort_keys=True, default=str).lower()
    return any(token in text for token in ("research_context_missing", "research_context_stale", "insufficient_research"))


def _repeated_patch_failures(strategy_state: dict[str, Any]) -> bool:
    try:
        return int(strategy_state.get("patch_attempt_failure_count", 0)) >= 2
    except (TypeError, ValueError):
        return False


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if isinstance(value, dict):
        return dict(value)
    raise TypeError("feedback_report must be a ToolFeedbackReport or dictionary.")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
