"""Agent-context markdown assembly from decision-facing safe inputs."""

from __future__ import annotations

import json
from typing import Any

from research_tool_agent_full_pool.decision_artifacts import assert_payload_public_safe


def build_agent_context_markdown(
    observed_evidence_text: str,
    memory_text: str,
    strategy_state: dict[str, Any],
    tool_state: dict[str, Any],
    round_index: int,
) -> str:
    """Assemble the latest agent context from observed evidence and state."""

    assert_payload_public_safe(strategy_state, label="strategy_state")
    assert_payload_public_safe(tool_state, label="tool_state")
    strategy_summary = compact_state_summary(strategy_state)
    tool_summary = compact_state_summary(tool_state)
    lines = [
        "# Agent Context",
        "",
        f"- round_index: {int(round_index)}",
        "- assembly_rule: observed evidence, memory, and compact current state only",
        "",
        "## Observed Evidence",
        "",
        observed_evidence_text.strip(),
        "",
        "## Memory",
        "",
        str(memory_text).strip() or "NA",
        "",
        "## Strategy State Summary",
        "",
        "```json",
        json.dumps(strategy_summary, indent=2, sort_keys=True, ensure_ascii=True),
        "```",
        "",
        "## Tool State Summary",
        "",
        "```json",
        json.dumps(tool_summary, indent=2, sort_keys=True, ensure_ascii=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def compact_state_summary(state: dict[str, Any], *, max_items: int = 24, max_text: int = 240) -> dict[str, Any]:
    """Return a deterministic compact summary of a JSON-like state dict."""

    compact: dict[str, Any] = {}
    for key in sorted(state, key=str)[:max_items]:
        compact[str(key)] = _compact_value(state[key], max_text=max_text)
    if len(state) > max_items:
        compact["_additional_key_count"] = len(state) - max_items
    return compact


def _compact_value(value: Any, *, max_text: int) -> Any:
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: str(item[0]))[:10]
        result = {str(key): _compact_value(item, max_text=max_text) for key, item in items}
        if len(value) > 10:
            result["_additional_key_count"] = len(value) - 10
        return result
    if isinstance(value, (list, tuple)):
        result = [_compact_value(item, max_text=max_text) for item in list(value)[:10]]
        if len(value) > 10:
            result.append({"_additional_item_count": len(value) - 10})
        return result
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > max_text:
            return value[: max_text - 3].rstrip() + "..."
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(type(value).__name__)
