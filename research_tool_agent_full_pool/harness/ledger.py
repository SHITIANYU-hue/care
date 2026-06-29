"""JSONL ledger for self-evolving policy-editing runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_tool_agent_full_pool.artifact_logger import sanitize_payload


class HarnessLedger:
    """Append-only public-safe event ledger."""

    def __init__(self, output_dir: str | Path, *, filename: str = "self_evolving_ledger.jsonl") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / filename

    def reset(self) -> None:
        self.path.write_text("", encoding="utf-8")

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        event = sanitize_payload(
            {
                "event_type": str(event_type),
                **dict(payload),
            }
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True, default=str) + "\n")


def write_json(path: str | Path, payload: Any, *, sanitize: bool = True) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output = sanitize_payload(payload) if sanitize else payload
    target.write_text(
        json.dumps(output, sort_keys=True, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )
