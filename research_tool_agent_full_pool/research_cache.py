"""Filesystem helpers for frozen Step 8A research-card artifacts."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


RESEARCH_CARD_SUBDIRS: tuple[str, ...] = (
    "sources",
    "cards",
    "rejected",
    "audit",
)


def stable_json_dumps(payload: Any) -> str:
    """Return deterministic JSON text for hashing and cache writes."""

    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def compact_json_dumps(payload: Any) -> str:
    """Return deterministic single-line JSON text for JSONL artifacts."""

    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_payload(payload: Any) -> str:
    return sha256_text(stable_json_dumps(payload))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(stable_json_dumps(payload), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[Any]) -> None:
    text = "".join(compact_json_dumps(row) + "\n" for row in rows)
    Path(path).write_text(text, encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def ensure_research_card_dirs(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name in RESEARCH_CARD_SUBDIRS:
        root.joinpath(name).mkdir(parents=True, exist_ok=True)
    return root


def reset_research_card_output(output_dir: str | Path) -> Path:
    """Clear only known Step 8A cache files from the output directory."""

    root = ensure_research_card_dirs(output_dir)
    known_root_files = (
        "research_manifest.json",
        "research_manifest.md",
        "research_collection_summary.md",
        "research_card_safety_audit.json",
        "research_card_safety_audit.md",
        "rejected_cards.jsonl",
        "accepted_cards.jsonl",
        "source_index.jsonl",
    )
    for name in known_root_files:
        path = root / name
        if path.exists():
            path.unlink()
    for subdir in RESEARCH_CARD_SUBDIRS:
        directory = root / subdir
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".md", ".txt"}:
                path.unlink()
    return root
