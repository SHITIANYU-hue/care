"""Generated optimizer tool synthesis and parser."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import pandas as pd

from research_tool_agent_full_pool.diagnostics import sha256_text
from research_tool_agent_full_pool.fake_client import FakeResearchToolGenerator
from research_tool_agent_full_pool.tool_contract import (
    ALLOWED_IMPORTS,
    FORBIDDEN_TERMS,
    REQUIRED_ENTRYPOINT,
    TOOL_SYNTHESIS_POLICY_NAME,
    TOOL_SYNTHESIS_SCHEMA_VERSION,
)


@dataclass(frozen=True)
class ToolSynthesisResult:
    """Parsed and validated generated-tool synthesis payload."""

    raw_text: str
    payload: dict[str, Any]
    code: str
    tool_name: str
    parser_report: dict[str, Any]
    request_summary: dict[str, Any]
    repair_count: int = 0
    prompt_text: str = ""


class ToolSynthesisParseError(ValueError):
    """Raised when the LLM output does not satisfy the Step 2 schema."""


def synthesize_initial_tool(*args: Any, **kwargs: Any) -> str:
    """Create the initial optimizer-style scoring tool source.

    Fake mode returns deterministic local code. API mode is handled through
    `synthesize_initial_tool_with_reports` so callers can persist parser reports.
    """

    mode = str(kwargs.get("mode", "fake")).lower()
    if mode != "fake":
        result = synthesize_initial_tool_with_reports(*args, **kwargs)
        return result.code
    return FakeResearchToolGenerator().create_initial_tool(*args, **kwargs)


def synthesize_initial_tool_with_reports(
    *,
    mode: str,
    client: Any | None = None,
    config: Any | None = None,
    observed_df: pd.DataFrame | None = None,
    candidate_df: pd.DataFrame | None = None,
    memory_text: str = "",
    strategy_state: dict[str, Any] | None = None,
    tool_state: dict[str, Any] | None = None,
    round_index: int = 1,
    parser_error: str | None = None,
    agent_context_text: str = "",
) -> ToolSynthesisResult:
    """Create one generated tool and return parser metadata."""

    if str(mode).lower() == "fake":
        code = FakeResearchToolGenerator().create_initial_tool()
        return ToolSynthesisResult(
            raw_text="",
            payload={},
            code=code,
            tool_name=REQUIRED_ENTRYPOINT,
            parser_report={"passed": True, "mode": "fake"},
            request_summary={"mode": "fake"},
        )
    if client is None:
        raise ValueError("API mode requires a tool synthesis client.")
    if config is None or observed_df is None or candidate_df is None:
        raise ValueError("API mode requires config, observed_df, and candidate_df.")

    prompt = build_tool_synthesis_prompt(
        config=config,
        observed_df=observed_df,
        candidate_df=candidate_df,
        memory_text=memory_text,
        strategy_state=strategy_state or {},
        tool_state=tool_state or {},
        round_index=round_index,
        parser_error=parser_error,
        agent_context_text=agent_context_text,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You generate only one JSON object containing safe Python code for a "
                "full-pool offline optimizer tool. Do not return markdown."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    raw_text = client.create_tool(messages=messages)
    try:
        payload = parse_tool_synthesis_json(
            raw_text,
            expected_run_id=config.run_id,
            expected_round_index=round_index,
        )
    except ToolSynthesisParseError as exc:
        setattr(exc, "raw_text", raw_text)
        setattr(exc, "prompt_text", prompt)
        setattr(
            exc,
            "request_summary",
            {
                "mode": "api",
                "run_id": config.run_id,
                "round_index": round_index,
                "prompt_hash": sha256_text(prompt),
                "prompt_character_count": len(prompt),
                "observed_rows_in_prompt": min(len(observed_df), 20),
                "candidate_rows_in_prompt": 0,
                "candidate_df_rows_available_to_tool": len(candidate_df),
                "candidate_columns": list(candidate_df.columns),
                "observed_columns": list(observed_df.columns),
                "repair_prompt": parser_error is not None,
                "research_source": "none",
                "research_manifest_hash": None,
                "research_card_ids_included": [],
                "research_context_path": None,
            },
        )
        raise
    generated_tool = payload["generated_tool"]
    design = payload["tool_design"]
    return ToolSynthesisResult(
        raw_text=raw_text,
        payload=payload,
        code=generated_tool["code"],
        tool_name=str(design.get("tool_name") or REQUIRED_ENTRYPOINT),
        parser_report={
            "passed": True,
            "schema_version": payload["schema_version"],
            "run_id": payload["run_id"],
            "round_index": payload["round_index"],
            "tool_name": str(design.get("tool_name") or REQUIRED_ENTRYPOINT),
        },
        request_summary={
            "mode": "api",
            "run_id": config.run_id,
            "round_index": round_index,
            "prompt_hash": sha256_text(prompt),
            "prompt_character_count": len(prompt),
            "observed_rows_in_prompt": min(len(observed_df), 20),
            "candidate_rows_in_prompt": 0,
            "candidate_df_rows_available_to_tool": len(candidate_df),
            "candidate_columns": list(candidate_df.columns),
            "observed_columns": list(observed_df.columns),
            "repair_prompt": parser_error is not None,
            "research_source": "none",
            "research_manifest_hash": None,
            "research_card_ids_included": [],
            "research_context_path": None,
        },
        prompt_text=prompt,
    )


def parse_tool_synthesis_json(
    raw_text: str,
    *,
    expected_run_id: str,
    expected_round_index: int,
) -> dict[str, Any]:
    """Parse and validate the Step 2 generated-tool JSON object."""

    text = str(raw_text).strip()
    if not text.startswith("{") or not text.endswith("}"):
        if text.startswith("{") and not text.endswith("}"):
            raise ToolSynthesisParseError("LLM JSON object appears truncated before the closing brace.")
        raise ToolSynthesisParseError("LLM output must be exactly one JSON object.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolSynthesisParseError("LLM output is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ToolSynthesisParseError("LLM output must decode to a JSON object.")
    allowed_top = {
        "schema_version",
        "policy_name",
        "run_id",
        "round_index",
        "design_rationale",
        "tool_design",
        "generated_tool",
        "self_reported_forbidden_info_used",
    }
    extra = set(payload) - allowed_top
    if extra:
        raise ToolSynthesisParseError(f"LLM output contains unsupported top-level fields: {sorted(extra)}")
    _reject_forbidden_payload_terms(payload)
    if payload.get("schema_version") != TOOL_SYNTHESIS_SCHEMA_VERSION:
        raise ToolSynthesisParseError("schema_version mismatch.")
    if payload.get("policy_name") != TOOL_SYNTHESIS_POLICY_NAME:
        raise ToolSynthesisParseError("policy_name mismatch.")
    if payload.get("run_id") != expected_run_id:
        raise ToolSynthesisParseError("run_id mismatch.")
    if int(payload.get("round_index", -999)) != int(expected_round_index):
        raise ToolSynthesisParseError("round_index mismatch.")
    if payload.get("self_reported_forbidden_info_used") is not False:
        raise ToolSynthesisParseError("self_reported_forbidden_info_used must be false.")

    _validate_design_rationale(payload.get("design_rationale"), require_research_cards=False)

    design = payload.get("tool_design")
    if not isinstance(design, dict):
        raise ToolSynthesisParseError("tool_design must be an object.")
    required_design = {
        "tool_name",
        "strategy_summary",
        "expected_behavior",
    }
    separation_fields = {
        "reference_bo_separation_statement",
        "baseline_separation_statement",
        # Temporary backward compatibility for artifacts produced before the
        # policy-alignment wording change.
        "why_this_is_not_direct_bo",
    }
    if not required_design.issubset(design) or not (set(design) & separation_fields):
        raise ToolSynthesisParseError("tool_design is missing required fields.")
    if "reference_bo_separation_statement" not in design and "baseline_separation_statement" not in design:
        design["reference_bo_separation_statement"] = str(design.get("why_this_is_not_direct_bo", ""))

    generated_tool = payload.get("generated_tool")
    if not isinstance(generated_tool, dict):
        raise ToolSynthesisParseError("generated_tool must be an object.")
    if generated_tool.get("entrypoint") != REQUIRED_ENTRYPOINT:
        raise ToolSynthesisParseError("generated_tool.entrypoint must be rank_candidates.")
    code = generated_tool.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ToolSynthesisParseError("generated_tool.code must be a non-empty string.")
    allowed_imports = generated_tool.get("allowed_imports", [])
    if not isinstance(allowed_imports, list):
        raise ToolSynthesisParseError("generated_tool.allowed_imports must be a list.")
    declared = {str(item) for item in allowed_imports}
    if declared and not declared.issubset(set(ALLOWED_IMPORTS)):
        raise ToolSynthesisParseError("generated_tool.allowed_imports contains disallowed imports.")
    if f"def {REQUIRED_ENTRYPOINT}" not in code:
        raise ToolSynthesisParseError("generated_tool.code must define rank_candidates.")
    return payload


def build_tool_synthesis_prompt(
    *,
    config: Any,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    memory_text: str,
    strategy_state: dict[str, Any],
    tool_state: dict[str, Any],
    round_index: int,
    parser_error: str | None = None,
    agent_context_text: str = "",
) -> str:
    """Build an API prompt that never includes full candidate rows."""

    observed_sample = observed_df.head(20).to_dict(orient="records")
    candidate_summary = _candidate_summary(candidate_df)
    repair = (
        "\nRepair reason from previous attempt: "
        + str(parser_error)[:600]
        + "\nDo not add any new experimental information."
        if parser_error
        else ""
    )
    return f"""
Return exactly one compact JSON object. No markdown, no code fence outside JSON string fields.
First character must be {{ and last character must be }}.
The complete response must be valid JSON and must finish with the final closing brace.
The JSON generated_tool.code string must decode to normal Python source with real newline characters. Do not double-escape code newlines as literal backslash-n text.
Keep generated_tool.code compact: target under 220 lines and under 6500 characters.
Do not include explanatory text outside the JSON object.

Task: full-pool persistent ResearchToolAgent generated optimizer-tool synthesis.
Run id: {config.run_id}
Round index: {int(round_index)}
Dataset: {config.dataset_name}
Objective: {config.target_column}
Direction: {config.objective_direction}

observed_df schema:
{json.dumps(_schema_summary(observed_df), indent=2)}

candidate_df schema:
{json.dumps(_schema_summary(candidate_df), indent=2)}

candidate_df compact public-safe summary:
{json.dumps(candidate_summary, indent=2)}

Compact observed history with revealed objective only:
{json.dumps(observed_sample, indent=2)}

memory.md:
{memory_text[:3000]}

agent_context.md:
{agent_context_text[:5000]}

strategy_state summary:
{json.dumps(_compact_state(strategy_state), indent=2)}

tool_state summary:
{json.dumps(_compact_state(tool_state), indent=2)}

Tool requirements:
- Define exactly: def rank_candidates(observed_df, candidate_df, memory=None, tool_state=None):
- The generated function receives the full remaining candidate_df at runtime.
- Score every row in candidate_df and return one ranked row per candidate.
- Design the best public-safe full-pool candidate-scoring tool you consider appropriate under the isolation rules below.
- You may use any public-safe strategy if it fits the data and contract, including hand-written heuristics, surrogate models, acquisition-style scoring, uncertainty/diversity terms, domain priors, ensembles, Bayesian-optimization-like logic, and bounded sklearn estimators.
- Bounded sklearn imports are allowed only for: sklearn.linear_model.Ridge, sklearn.ensemble.RandomForestRegressor, sklearn.neighbors.KNeighborsRegressor, sklearn.preprocessing.StandardScaler, sklearn.impute.SimpleImputer, and sklearn.pipeline.Pipeline.
- Keep the implementation compact and deterministic.
- Do not use dtype=object or the bare built-in name object; the sandbox does not expose that builtin. Use Python lists, dtype=str, or no explicit dtype for string reason arrays.
- Do not use @, .dot(...), np.matmul, np.linalg, pinv, or matrix multiplication. Use explicit elementwise formulas such as (weights * values[None, :]).sum(axis=1) / (weights.sum(axis=1) + eps).
- When computing pairwise distances or weighted local means, keep candidate-by-observation arrays aligned and reduce with sum/mean over axis=1; never multiply two 2D matrices together.
- Return a Python dict, never a pandas DataFrame or list directly.
- The final return statement must have this shape:
  return {{
    "ranked_candidates": ranked_candidates,
    "tool_state": tool_state,
    "tool_diagnostics": tool_diagnostics
  }}
- Each ranked candidate dict must contain candidate_id, rank, score,
  reason_code, and evidence_refs.
- evidence_refs must be [] or a list of observation_id values from observed_df
  such as obs_000001. Do not put explanatory strings, candidate IDs, counts,
  scores, model names, or labels in evidence_refs.
- selected candidate will be rank 1; do not implement any final LLM override.
- Use only observed_df observed_y for revealed observed rows.
- Candidate features are public-safe only.
- You may import only allowed public-safe packages declared in generated_tool.allowed_imports.
- Avoid all file I/O, network calls, environment access, subprocesses, eval/exec,
  dynamic imports, and dunder escape patterns.
- Strict isolation boundary: do not import, call, read, copy, or condition on repository reference optimizer implementation code, local comparator modules, comparator outputs, cached score artifacts, reference predictive statistics, reference acquisition values, answer-key ranks, retrospective ranks, internal evaluator state, non-public candidate mappings, credentials, secrets, or benchmark answers.
- If using acquisition-style terminology in code, prefer public-safe names such as model_mean_public, uncertainty_public, and acquisition_like_score_public.
- Do not include private evaluator, cached optimizer, provenance, credential, or secret material in code, comments, strings, reason_code, diagnostics, or tool_state.

JSON schema:
{{
  "schema_version": "{TOOL_SYNTHESIS_SCHEMA_VERSION}",
  "policy_name": "{TOOL_SYNTHESIS_POLICY_NAME}",
  "run_id": "{config.run_id}",
  "round_index": {int(round_index)},
  "design_rationale": {{
    "chosen_strategy_family": "short public-safe method family",
    "alternatives_considered": ["short public-safe alternative"],
    "reason_for_choice": "short public-safe reason",
    "how_observed_y_is_used": "use only observed_df observed_y from revealed rows",
    "how_uncertainty_or_exploration_is_handled": "short string",
    "how_research_cards_are_used": "no research cards are provided in research_mode none",
    "expected_failure_modes": ["short public-safe failure mode"],
    "public_safe_boundary_statement": "scores use only prompt/runtime public inputs and no comparator, evaluator, private, credential, or answer-key artifacts"
  }},
  "tool_design": {{
    "tool_name": "safe_short_name",
    "strategy_summary": "short string",
    "reference_bo_separation_statement": "short string explaining that the tool computes all scores from public-safe inputs and does not use repository comparator code or baseline/evaluator artifacts",
    "expected_behavior": "short string"
  }},
  "generated_tool": {{
    "entrypoint": "{REQUIRED_ENTRYPOINT}",
    "allowed_imports": ["numpy", "pandas", "math", "statistics", "sklearn.linear_model", "sklearn.ensemble", "sklearn.neighbors", "sklearn.preprocessing", "sklearn.impute", "sklearn.pipeline"],
    "code": "Python source code as a string"
  }},
  "self_reported_forbidden_info_used": false
}}
{repair}
""".strip()


def build_tool_repair_prompt(*args: Any, **kwargs: Any) -> str:
    """Compatibility wrapper for repair prompt construction."""

    return build_tool_synthesis_prompt(*args, **kwargs)


def patch_tool_after_reveal(*args: Any, **kwargs: Any) -> Any:
    """Patch or replace generated tool source after a reveal, if enabled."""

    from research_tool_agent_full_pool.tool_patch_synthesis import patch_tool_after_reveal as _patch

    return _patch(*args, **kwargs)


def _validate_design_rationale(value: Any, *, require_research_cards: bool) -> None:
    if not isinstance(value, dict):
        raise ToolSynthesisParseError("design_rationale must be an object.")
    required = {
        "chosen_strategy_family",
        "alternatives_considered",
        "reason_for_choice",
        "how_observed_y_is_used",
        "how_uncertainty_or_exploration_is_handled",
        "how_research_cards_are_used",
        "expected_failure_modes",
        "public_safe_boundary_statement",
    }
    missing = [field for field in sorted(required) if field not in value]
    if missing:
        raise ToolSynthesisParseError(f"design_rationale is missing required fields: {missing}")
    for field in required - {"alternatives_considered", "expected_failure_modes"}:
        if not isinstance(value.get(field), str) or not str(value.get(field)).strip():
            raise ToolSynthesisParseError(f"design_rationale.{field} must be a non-empty string.")
    for field in ("alternatives_considered", "expected_failure_modes"):
        if not isinstance(value.get(field), list):
            raise ToolSynthesisParseError(f"design_rationale.{field} must be a list.")
    if require_research_cards and "research_card_usage" not in value:
        raise ToolSynthesisParseError("design_rationale.research_card_usage is required.")


def _schema_summary(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [{"name": str(column), "dtype": str(dtype)} for column, dtype in frame.dtypes.items()]


def _candidate_summary(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "row_count": len(frame),
        "columns": list(frame.columns),
        "numeric": {},
        "non_numeric": {},
    }
    for column in frame.columns:
        if column == "candidate_id":
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            summary["numeric"][str(column)] = {
                "count": int(numeric.notna().sum()),
                "mean": _safe_float(numeric.mean()),
                "std": _safe_float(numeric.std()),
                "min": _safe_float(numeric.min()),
                "max": _safe_float(numeric.max()),
            }
        else:
            summary["non_numeric"][str(column)] = {
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
            }
    return summary


def _compact_state(state: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[str(key)] = value
        elif isinstance(value, (list, tuple)):
            compact[str(key)] = list(value[:10])
        elif isinstance(value, dict):
            compact[str(key)] = {str(k): v for k, v in list(value.items())[:10]}
        else:
            compact[str(key)] = str(type(value).__name__)
    return compact


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _reject_forbidden_payload_terms(payload: Any) -> None:
    text = json.dumps(payload, sort_keys=True)
    lowered = text.lower()
    hits = [term for term in FORBIDDEN_TERMS if term.lower() in lowered]
    if hits:
        raise ToolSynthesisParseError("LLM output contains forbidden contract terms.")
