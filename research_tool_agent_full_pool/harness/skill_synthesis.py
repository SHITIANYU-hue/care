"""Skill synthesis and patching for the self-evolving harness."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from research_tool_agent_full_pool.decision_artifacts import assert_payload_public_safe
from research_tool_agent_full_pool.fake_client import FAKE_FULL_POOL_TOOL_SOURCE
from research_tool_agent_full_pool.harness.specs import SkillArtifact, TaskPlan
from research_tool_agent_full_pool.research_cache import sha256_text


SKILL_SYNTHESIS_SCHEMA_VERSION = "self_evolving_skill_artifact_v1"


class SkillSynthesisParseError(ValueError):
    """Raised when a generated skill response is malformed."""


def synthesize_skill(
    *,
    client: Any | None,
    mode: str,
    run_id: str,
    round_index: int,
    task_plan: TaskPlan,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    memory_text: str,
    policy_state: dict[str, Any],
    active_skill: SkillArtifact | None = None,
    parser_error: str | None = None,
) -> tuple[SkillArtifact, dict[str, Any]]:
    """Create or patch one skill artifact."""

    skill_id = active_skill.skill_id if active_skill and task_plan.action == "patch_skill" else _skill_id(task_plan, round_index)
    version = int(active_skill.version) + 1 if active_skill and task_plan.action == "patch_skill" else 1

    if str(mode).lower() == "fake" or client is None:
        source = _fake_source_for_round(round_index=round_index, active_skill=active_skill)
        skill = SkillArtifact(
            skill_id=skill_id,
            version=version,
            family=task_plan.skill_family,
            source=source,
            parent_skill_id=active_skill.skill_id if active_skill else None,
            parent_version=active_skill.version if active_skill else None,
            created_round=int(round_index),
            objective=task_plan.objective,
            provenance={"mode": "fake", "task_plan": task_plan.to_dict()},
        )
        return skill, {"mode": "fake", "skill_schema_version": SKILL_SYNTHESIS_SCHEMA_VERSION}

    prompt = build_skill_synthesis_prompt(
        run_id=run_id,
        round_index=round_index,
        task_plan=task_plan,
        observed_df=observed_df,
        candidate_df=candidate_df,
        memory_text=memory_text,
        policy_state=policy_state,
        active_skill=active_skill,
        parser_error=parser_error,
    )
    raw_text = client.create_tool(
        messages=[
            {
                "role": "developer",
                "content": (
                    "You are the public-safe skill author for an offline finite-pool optimization harness. "
                    "Return only the JSON object required by the schema. The source must define exactly "
                    "rank_candidates(observed_df, candidate_df, memory=None, tool_state=None). Do not read "
                    "files, call networks, inspect DataFrame attrs, or use evaluator-private outcomes."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        json_schema=skill_synthesis_json_schema(run_id=run_id, round_index=round_index, skill_family=task_plan.skill_family),
        schema_name="self_evolving_skill_artifact",
        schema_description="One public-safe Python rank_candidates skill artifact.",
    )
    skill = parse_skill_synthesis_json(
        raw_text,
        expected_run_id=run_id,
        expected_round_index=round_index,
        task_plan=task_plan,
        default_skill_id=skill_id,
        default_version=version,
        active_skill=active_skill,
    )
    return skill, {
        "mode": "api",
        "raw_text": raw_text,
        "prompt_hash": sha256_text(prompt),
        "prompt_character_count": len(prompt),
        "skill_schema_version": SKILL_SYNTHESIS_SCHEMA_VERSION,
    }


def build_skill_synthesis_prompt(
    *,
    run_id: str,
    round_index: int,
    task_plan: TaskPlan,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    memory_text: str,
    policy_state: dict[str, Any],
    active_skill: SkillArtifact | None,
    parser_error: str | None = None,
) -> str:
    payload = {
        "task": "generate_or_patch_public_safe_rank_candidates_skill",
        "schema_version": SKILL_SYNTHESIS_SCHEMA_VERSION,
        "run_id": run_id,
        "round_index": int(round_index),
        "task_plan": task_plan.to_dict(),
        "observed_examples": _safe_records(_compact_prompt_frame(observed_df.tail(12), max_cell_chars=120)),
        "candidate_schema": [{"name": str(col), "dtype": str(dtype)} for col, dtype in candidate_df.dtypes.items()],
        "candidate_summary": {
            "row_count": int(len(candidate_df)),
            "columns": list(candidate_df.columns),
            "categorical_cardinality": _categorical_cardinality(candidate_df),
        },
        "memory_summary": str(memory_text)[:1200],
        "policy_state_summary": {
            "active_skill_id": policy_state.get("active_skill_id"),
            "active_version": policy_state.get("active_version"),
            "recent_rewards": list(policy_state.get("reward_history", []))[-5:],
            "recent_gates": list(policy_state.get("gate_history", []))[-3:],
        },
        "active_skill_source": str(active_skill.source)[:8000] if active_skill else "",
        "parser_error": str(parser_error or "")[:500],
        "instructions": [
            "Write normal Python source defining exactly rank_candidates(observed_df, candidate_df, memory=None, tool_state=None).",
            "The tool must return a dictionary, never a DataFrame. Required shape: {'ranked_candidates': list_of_dict_rows, 'tool_state': dict, 'tool_diagnostics': dict}.",
            "Each ranked row must include candidate_id, rank, score, reason_code, and evidence_refs. It must cover every candidate_df row with unique positive ranks and finite scores.",
            "evidence_refs must be [] unless you copy exact observation_id values from observed_df; never invent candidate:... or observed:... evidence strings.",
            "Use only public observed_y for observed rows and public candidate features.",
            "Do not read files, call networks, import disallowed modules, inspect DataFrame attrs, or reference evaluator-private labels, ranking answer keys, baseline artifacts, private provenance fields, or credentials.",
            "Do not use double-underscore names or strings anywhere in the source, including temporary columns, helper variables, imports, or escape hatches.",
            "Do not call getattr, setattr, hasattr, eval, exec, compile, globals, locals, vars, open, or __import__.",
            "Make ranking fully row-order invariant: do not use enumerate index, original row order, DataFrame index, or insertion order in scores or tie-breaks.",
            "Sort with public score first and candidate_id as the only deterministic tie-breaker; never tie-break by row position.",
            "Avoid creating helper columns such as _row_order, _index, __lig, or _position for ordering.",
            "Return JSON with skill_artifact.source and self_reported_forbidden_info_used=false.",
        ],
        "output_schema": {
            "schema_version": SKILL_SYNTHESIS_SCHEMA_VERSION,
            "run_id": run_id,
            "round_index": int(round_index),
            "skill_artifact": {
                "skill_id": "short id",
                "family": task_plan.skill_family,
                "source": "Python source string defining rank_candidates",
                "rationale": "short public-safe rationale",
            },
            "self_reported_forbidden_info_used": False,
        },
    }
    assert_payload_public_safe(payload, label="self_evolving_skill_prompt")
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def skill_synthesis_json_schema(*, run_id: str, round_index: int, skill_family: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "run_id",
            "round_index",
            "skill_artifact",
            "self_reported_forbidden_info_used",
        ],
        "properties": {
            "schema_version": {"type": "string", "enum": [SKILL_SYNTHESIS_SCHEMA_VERSION]},
            "run_id": {"type": "string", "enum": [str(run_id)]},
            "round_index": {"type": "integer", "enum": [int(round_index)]},
            "skill_artifact": {
                "type": "object",
                "additionalProperties": False,
                "required": ["skill_id", "family", "source", "rationale"],
                "properties": {
                    "skill_id": {"type": "string", "minLength": 1, "maxLength": 96},
                    "family": {"type": "string", "enum": [str(skill_family)]},
                    "source": {"type": "string", "minLength": 40},
                    "rationale": {"type": "string"},
                },
            },
            "self_reported_forbidden_info_used": {"type": "boolean", "enum": [False]},
        },
    }


def parse_skill_synthesis_json(
    raw_text: str,
    *,
    expected_run_id: str,
    expected_round_index: int,
    task_plan: TaskPlan,
    default_skill_id: str,
    default_version: int,
    active_skill: SkillArtifact | None = None,
) -> SkillArtifact:
    text = _extract_json_object_text(str(raw_text).strip())
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SkillSynthesisParseError("Skill synthesis output is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise SkillSynthesisParseError("Skill synthesis output must decode to an object.")
    allowed = {"schema_version", "run_id", "round_index", "skill_artifact", "self_reported_forbidden_info_used"}
    extra = set(payload) - allowed
    if extra:
        raise SkillSynthesisParseError(f"Unsupported skill synthesis fields: {sorted(extra)}")
    assert_payload_public_safe(payload, label="self_evolving_skill_response")
    if payload.get("schema_version") != SKILL_SYNTHESIS_SCHEMA_VERSION:
        raise SkillSynthesisParseError("schema_version mismatch.")
    if payload.get("run_id") != expected_run_id:
        raise SkillSynthesisParseError("run_id mismatch.")
    if int(payload.get("round_index", -999)) != int(expected_round_index):
        raise SkillSynthesisParseError("round_index mismatch.")
    if payload.get("self_reported_forbidden_info_used") is not False:
        raise SkillSynthesisParseError("self_reported_forbidden_info_used must be false.")
    artifact = payload.get("skill_artifact")
    if not isinstance(artifact, dict):
        raise SkillSynthesisParseError("skill_artifact must be an object.")
    source = artifact.get("source")
    if not isinstance(source, str) or "def rank_candidates" not in source:
        raise SkillSynthesisParseError("skill_artifact.source must define rank_candidates.")
    family = str(artifact.get("family", task_plan.skill_family))
    if family != task_plan.skill_family:
        family = task_plan.skill_family
    skill_id = str(artifact.get("skill_id") or default_skill_id)
    return SkillArtifact(
        skill_id=skill_id,
        version=int(default_version),
        family=family,  # type: ignore[arg-type]
        source=source,
        parent_skill_id=active_skill.skill_id if active_skill else None,
        parent_version=active_skill.version if active_skill else None,
        created_round=int(expected_round_index),
        objective=task_plan.objective,
        provenance={
            "mode": "api",
            "task_plan": task_plan.to_dict(),
            "rationale": str(artifact.get("rationale", ""))[:1000],
        },
    )


def _extract_json_object_text(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    raise SkillSynthesisParseError("Skill synthesis output must contain one JSON object.")


def _skill_id(task_plan: TaskPlan, round_index: int) -> str:
    return f"{task_plan.skill_family}_skill_round_{int(round_index):03d}"


def _fake_source_for_round(*, round_index: int, active_skill: SkillArtifact | None) -> str:
    if active_skill is None:
        return FAKE_FULL_POOL_TOOL_SOURCE
    return FAKE_FULL_POOL_TOOL_SOURCE.replace(
        '"fake_full_pool_score"',
        f'"fake_full_pool_score_patch_{int(round_index):03d}"',
    )


def _safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_safe(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _compact_prompt_frame(frame: pd.DataFrame, *, max_cell_chars: int) -> pd.DataFrame:
    compact = frame.copy()
    for column in compact.columns:
        if pd.api.types.is_numeric_dtype(compact[column]):
            continue
        compact[column] = compact[column].map(lambda value: _truncate_cell(value, max_cell_chars=max_cell_chars))
    return compact


def _truncate_cell(value: Any, *, max_cell_chars: int) -> Any:
    if pd.isna(value):
        return None
    text = str(value)
    if len(text) <= int(max_cell_chars):
        return text
    return text[: int(max_cell_chars)] + "...[truncated]"


def _categorical_cardinality(frame: pd.DataFrame) -> dict[str, int]:
    result: dict[str, int] = {}
    for column in frame.columns:
        if str(column) == "candidate_id":
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            result[str(column)] = int(frame[column].astype(str).nunique(dropna=False))
    return result
