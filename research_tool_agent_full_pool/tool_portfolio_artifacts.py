"""Artifact writers for ResearchToolAgent portfolio diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_tool_agent_full_pool.artifact_logger import ensure_output_dir, sanitize_payload, sanitize_text
from research_tool_agent_full_pool.decision_artifacts import assert_payload_public_safe
from research_tool_agent_full_pool.diagnostics import sha256_text


def initialize_portfolio_artifacts(output_dir: str | Path) -> None:
    """Create portfolio artifact folders and truncate JSONL diagnostics."""

    root = ensure_output_dir(output_dir)
    for subdir in (
        "candidate_tool_sources",
        "candidate_tool_prompts",
        "candidate_tool_raw_responses",
        "candidate_tool_repairs",
    ):
        path = root / subdir
        path.mkdir(parents=True, exist_ok=True)
        for child in path.glob("*"):
            if child.is_file():
                child.unlink()
    for name in ("candidate_verifier_reports.jsonl", "observed_quality_scores.jsonl", "tool_mechanism_classifications.jsonl"):
        (root / name).write_text("", encoding="utf-8")
    for name in (
        "method_primitives.json",
        "portfolio_designs.json",
        "portfolio_design_prompt.md",
        "portfolio_design_request.json",
        "portfolio_design_raw_response.json",
        "candidate_tools_manifest.json",
        "prompt_pipeline_audit.json",
        "selected_portfolio_tool.json",
        "portfolio_summary.md",
    ):
        path = root / name
        if path.exists() and path.is_file():
            path.unlink()


def write_portfolio_design_artifacts(
    *,
    output_dir: str | Path,
    prompt_text: str,
    request_metadata: dict[str, Any],
    raw_response: str,
    parsed_designs: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    task_interpretation: dict[str, Any] | None = None,
    static_self_audit: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Persist portfolio design prompt, request, raw response, and parsed JSON."""

    root = ensure_output_dir(output_dir)
    prompt_hash = sha256_text(prompt_text)
    request = {
        **dict(request_metadata),
        "prompt_hash": prompt_hash,
        "prompt_character_count": len(prompt_text),
    }
    assert_payload_public_safe({"request": request, "designs": parsed_designs, "diagnostics": diagnostics}, label="portfolio_design_artifacts")
    paths = {
        "prompt_path": str(root / "portfolio_design_prompt.md"),
        "request_path": str(root / "portfolio_design_request.json"),
        "raw_response_path": str(root / "portfolio_design_raw_response.json"),
        "designs_path": str(root / "portfolio_designs.json"),
    }
    Path(paths["prompt_path"]).write_text(sanitize_text(prompt_text), encoding="utf-8")
    _write_json(Path(paths["request_path"]), request)
    _write_json(Path(paths["raw_response_path"]), {"raw_text": sanitize_text(raw_response)})
    _write_json(
        Path(paths["designs_path"]),
        {
            "task_interpretation": task_interpretation or {},
            "designs": parsed_designs,
            "diagnostics": diagnostics,
            "static_self_audit": static_self_audit or {},
            "prompt_hash": prompt_hash,
            "method_primitive_ids_included": list(request_metadata.get("method_primitive_ids_included", [])),
            "research_source_metadata": request_metadata.get("research_source_metadata", {}),
        },
    )
    write_prompt_pipeline_audit(
        output_dir,
        {
            "task_interpretation": task_interpretation or {},
            "portfolio_static_self_audit": static_self_audit or {},
            "designs": [
                _design_audit_row(design)
                for design in parsed_designs
            ],
            "diagnostics": diagnostics,
            "prompt_hash": prompt_hash,
        },
    )
    return paths


def write_method_primitives(output_dir: str | Path, primitives: list[dict[str, Any]]) -> None:
    assert_payload_public_safe({"method_primitives": primitives}, label="method_primitives")
    _write_json(ensure_output_dir(output_dir) / "method_primitives.json", {"method_primitives": primitives})


def write_candidate_tool_artifacts(
    *,
    output_dir: str | Path,
    tool_id: str,
    design_id: str,
    prompt_text: str,
    request_metadata: dict[str, Any],
    raw_response: str,
    source: str,
) -> dict[str, Any]:
    """Persist one candidate tool prompt, raw response, and parsed source."""

    root = ensure_output_dir(output_dir)
    prompt_hash = sha256_text(prompt_text)
    source_hash = sha256_text(source)
    safe_tool_id = _safe_filename(tool_id)
    request = {
        **dict(request_metadata),
        "tool_id": tool_id,
        "design_id": design_id,
        "prompt_hash": prompt_hash,
        "prompt_character_count": len(prompt_text),
        "source_hash": source_hash,
    }
    assert_payload_public_safe(request, label="candidate_tool_request")
    prompt_path = root / "candidate_tool_prompts" / f"{safe_tool_id}.md"
    raw_path = root / "candidate_tool_raw_responses" / f"{safe_tool_id}.json"
    source_path = root / "candidate_tool_sources" / f"{safe_tool_id}.py"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(sanitize_text(prompt_text), encoding="utf-8")
    _write_json(raw_path, {"tool_id": tool_id, "design_id": design_id, "raw_text": sanitize_text(raw_response)})
    source_path.write_text(sanitize_text(source), encoding="utf-8")
    return {
        "tool_id": tool_id,
        "design_id": design_id,
        "prompt_path": str(prompt_path),
        "raw_response_path": str(raw_path),
        "source_path": str(source_path),
        "prompt_hash": prompt_hash,
        "source_hash": source_hash,
        "request_metadata": sanitize_payload(request),
    }


def write_candidate_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> None:
    assert_payload_public_safe(manifest, label="candidate_tools_manifest")
    _write_json(ensure_output_dir(output_dir) / "candidate_tools_manifest.json", manifest)


def write_prompt_pipeline_audit(output_dir: str | Path, update: dict[str, Any]) -> None:
    root = ensure_output_dir(output_dir)
    path = root / "prompt_pipeline_audit.json"
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    merged = {**existing, **sanitize_payload(update)}
    assert_payload_public_safe(merged, label="prompt_pipeline_audit")
    _write_json(path, merged)


def append_candidate_verifier_report(output_dir: str | Path, report: dict[str, Any]) -> None:
    _append_jsonl(ensure_output_dir(output_dir) / "candidate_verifier_reports.jsonl", report)


def append_observed_quality_score(output_dir: str | Path, score: dict[str, Any]) -> None:
    _append_jsonl(ensure_output_dir(output_dir) / "observed_quality_scores.jsonl", score)


def write_tool_mechanism_classifications(output_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    path = ensure_output_dir(output_dir) / "tool_mechanism_classifications.jsonl"
    path.write_text("", encoding="utf-8")
    for row in rows:
        _append_jsonl(path, row)


def write_selected_portfolio_tool(output_dir: str | Path, payload: dict[str, Any]) -> None:
    assert_payload_public_safe(payload, label="selected_portfolio_tool")
    _write_json(ensure_output_dir(output_dir) / "selected_portfolio_tool.json", payload)


def write_portfolio_summary_markdown(output_dir: str | Path, text: str) -> None:
    ensure_output_dir(output_dir).joinpath("portfolio_summary.md").write_text(sanitize_text(text), encoding="utf-8")


def _design_audit_row(design: dict[str, Any]) -> dict[str, Any]:
    return {
        "design_id": design.get("design_id"),
        "family": design.get("family"),
        "state_variables": design.get("state_variables", []),
        "exploitation_term": design.get("exploitation_term"),
        "exploration_or_anti_collapse_term": design.get("exploration_or_anti_collapse_term"),
        "small_n_fallback": design.get("small_n_fallback"),
        "mixed_variable_handling": design.get("mixed_variable_handling"),
        "method_primitives_used": design.get("method_primitives_used", []),
        "method_primitives_not_used": design.get("method_primitives_not_used", []),
        "static_self_audit": design.get("static_self_audit", {}),
        "why_this_is_not_a_static_ranker": design.get("why_this_is_not_a_static_ranker"),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_payload(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_payload(payload), sort_keys=True, default=str) + "\n")


def _safe_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return cleaned[:80] or "candidate_tool"
