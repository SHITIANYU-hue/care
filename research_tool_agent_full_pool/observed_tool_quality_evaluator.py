"""Observed-only quality screening for portfolio candidate tools."""

from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np
import pandas as pd

from research_tool_agent_full_pool.artifact_logger import sanitize_text
from research_tool_agent_full_pool.tool_output_parser import ParsedToolOutput, parse_ranked_candidates
from research_tool_agent_full_pool.tool_runner import execute_rank_candidates_tool


QUALITY_WEIGHTS = {
    "observed_ranking_score": 0.35,
    "bootstrap_stability": 0.20,
    "score_sanity": 0.15,
    "research_mapping_score": 0.15,
    "feature_coverage_score": 0.10,
    "simplicity_score": 0.05,
}


def evaluate_candidate_tools_observed_only(
    *,
    candidate_tools: list[dict[str, Any]],
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    method_primitives: list[dict[str, Any]],
    objective_direction: str = "maximize",
    bootstrap_iterations: int = 12,
    top_k: int = 5,
    seed: int = 1,
) -> list[dict[str, Any]]:
    """Rank candidate tools using only observed-safe public data."""

    results = []
    for candidate in candidate_tools:
        result = evaluate_candidate_tool_observed_only(
            candidate_tool=candidate,
            observed_df=observed_df,
            candidate_df=candidate_df,
            method_primitives=method_primitives,
            objective_direction=objective_direction,
            bootstrap_iterations=bootstrap_iterations,
            top_k=top_k,
            seed=seed,
        )
        results.append(result)
    return sorted(results, key=lambda item: (-float(item.get("quality_score", 0.0)), str(item.get("tool_id", ""))))


def evaluate_candidate_tool_observed_only(
    *,
    candidate_tool: dict[str, Any],
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    method_primitives: list[dict[str, Any]],
    objective_direction: str = "maximize",
    bootstrap_iterations: int = 12,
    top_k: int = 5,
    seed: int = 1,
) -> dict[str, Any]:
    """Evaluate one tool without accessing unrevealed outcomes."""

    source = str(candidate_tool.get("source", ""))
    design = candidate_tool.get("design", {}) if isinstance(candidate_tool.get("design"), dict) else {}
    penalties: dict[str, float] = {}
    observed_ranking = _observed_leave_one_out_ranking_score(
        source,
        observed_df=observed_df,
        objective_direction=objective_direction,
    )
    bootstrap = _bootstrap_stability_score(
        source,
        observed_df=observed_df,
        candidate_df=candidate_df,
        iterations=bootstrap_iterations,
        top_k=top_k,
        seed=seed,
    )
    sanity = _score_sanity_score(source, observed_df=observed_df, candidate_df=candidate_df)
    mapping = _research_mapping_score(source, design=design, method_primitives=method_primitives)
    coverage = _feature_coverage_score(source, observed_df=observed_df, candidate_df=candidate_df)
    simplicity = _simplicity_score(source)

    if sanity["constant_score_collapse"]:
        penalties["constant_scores"] = 0.20
    if sanity["nonfinite_or_parse_failure"]:
        penalties["nan_inf_or_parse_failure"] = 0.35
    if sanity["score_explosion"]:
        penalties["score_explosion"] = 0.20
    if sanity["rank1_margin"] is not None and float(sanity["rank1_margin"]) <= 1e-10:
        penalties["near_zero_rank1_margin"] = 0.10
    if _row_order_instability(source, observed_df=observed_df, candidate_df=candidate_df):
        penalties["row_order_instability"] = 0.25
    if _hardcoded_candidate_id_count(source, candidate_df) > 0:
        penalties["hardcoded_candidate_id"] = 0.30
    if not coverage["uses_observed_y"]:
        penalties["ignores_observed_y"] = 0.15
    if coverage["feature_family_count"] <= 1:
        penalties["uses_only_one_narrow_feature_family"] = 0.08
    if simplicity["broad_exception_count"] >= 3:
        penalties["overly_broad_exception_swallowing"] = 0.08

    raw_quality = (
        QUALITY_WEIGHTS["observed_ranking_score"] * observed_ranking["score"]
        + QUALITY_WEIGHTS["bootstrap_stability"] * bootstrap["score"]
        + QUALITY_WEIGHTS["score_sanity"] * sanity["score"]
        + QUALITY_WEIGHTS["research_mapping_score"] * mapping["score"]
        + QUALITY_WEIGHTS["feature_coverage_score"] * coverage["score"]
        + QUALITY_WEIGHTS["simplicity_score"] * simplicity["score"]
    )
    quality = max(0.0, min(1.0, raw_quality - sum(penalties.values())))
    return {
        "tool_id": candidate_tool.get("tool_id"),
        "parent_tool_id": candidate_tool.get("parent_tool_id"),
        "repair_attempt": int(candidate_tool.get("repair_attempt", 0) or 0),
        "candidate_version": candidate_tool.get("candidate_version", "original"),
        "design_id": candidate_tool.get("design_id"),
        "tool_family": candidate_tool.get("tool_family"),
        "observed_ranking_score": _round(observed_ranking["score"]),
        "bootstrap_stability": _round(bootstrap["score"]),
        "score_sanity": _round(sanity["score"]),
        "research_mapping_score": _round(mapping["score"]),
        "feature_coverage_score": _round(coverage["score"]),
        "simplicity_score": _round(simplicity["score"]),
        "penalties": {key: _round(value) for key, value in sorted(penalties.items())},
        "quality_score": _round(quality),
        "diagnostics": {
            "observed_ranking": observed_ranking,
            "bootstrap_stability": bootstrap,
            "score_sanity": sanity,
            "research_mapping": mapping,
            "feature_coverage": coverage,
            "simplicity": simplicity,
            "observed_only": True,
            "low_confidence": bool(observed_ranking.get("low_confidence")),
        },
    }


def _observed_leave_one_out_ranking_score(
    source: str,
    *,
    observed_df: pd.DataFrame,
    objective_direction: str,
) -> dict[str, Any]:
    n = int(len(observed_df))
    if n < 3:
        return {
            "score": 0.5,
            "low_confidence": True,
            "reason": "observed_n_too_small_for_leave_one_out_ranking",
            "observed_n": n,
            "pair_count": 0,
        }
    scores: list[float] = []
    outcomes: list[float] = []
    for index in range(n):
        train = observed_df.drop(observed_df.index[index]).reset_index(drop=True)
        held = observed_df.iloc[[index]].copy()
        pseudo_candidate = held[[column for column in held.columns if column not in {"observed_y", "observation_id"}]].copy()
        if "candidate_id" not in pseudo_candidate.columns:
            pseudo_candidate["candidate_id"] = f"heldout_{index:03d}"
        parsed = _run_tool(source, observed_df=train, candidate_df=pseudo_candidate)
        if parsed is None or not parsed.ranked_candidates:
            return {
                "score": 0.0,
                "low_confidence": False,
                "reason": "tool_failed_on_leave_one_out_observed_context",
                "observed_n": n,
                "pair_count": 0,
            }
        scores.append(float(parsed.ranked_candidates[0]["score"]))
        outcomes.append(float(pd.to_numeric(held["observed_y"], errors="coerce").iloc[0]))
    if str(objective_direction).lower() == "minimize":
        outcomes = [-value for value in outcomes]
    wins = 0.0
    pairs = 0
    for left in range(n):
        for right in range(left + 1, n):
            y_delta = outcomes[left] - outcomes[right]
            if abs(y_delta) <= 1e-12:
                continue
            pairs += 1
            score_delta = scores[left] - scores[right]
            if y_delta * score_delta > 0:
                wins += 1.0
            elif abs(score_delta) <= 1e-12:
                wins += 0.5
    score = wins / pairs if pairs else 0.5
    return {
        "score": _round(score),
        "low_confidence": n < 5,
        "reason": "pairwise_concordance_on_observed_leave_one_out",
        "observed_n": n,
        "pair_count": pairs,
    }


def _bootstrap_stability_score(
    source: str,
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    iterations: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    if candidate_df.empty:
        return {"score": 0.0, "reason": "empty_candidate_df", "top1_mode_frequency": 0.0}
    rng = np.random.default_rng(int(seed))
    top1s: list[str] = []
    top_sets: list[set[str]] = []
    n = len(observed_df)
    for _ in range(max(int(iterations), 1)):
        if n:
            sample_indices = rng.integers(0, n, size=n)
            boot = observed_df.iloc[sample_indices].reset_index(drop=True)
        else:
            boot = observed_df.copy()
        parsed = _run_tool(source, observed_df=boot, candidate_df=candidate_df)
        if parsed is None:
            continue
        rows = sorted(parsed.ranked_candidates, key=lambda item: int(item["rank"]))
        top1s.append(str(rows[0]["candidate_id"]))
        top_sets.append({str(row["candidate_id"]) for row in rows[: max(1, int(top_k))]})
    if not top1s:
        return {"score": 0.0, "reason": "tool_failed_all_bootstrap_runs", "top1_mode_frequency": 0.0}
    top1_mode_frequency = max(top1s.count(item) for item in set(top1s)) / len(top1s)
    overlaps = []
    for index in range(1, len(top_sets)):
        union = top_sets[index - 1] | top_sets[index]
        overlaps.append(len(top_sets[index - 1] & top_sets[index]) / max(len(union), 1))
    topk_overlap = sum(overlaps) / len(overlaps) if overlaps else top1_mode_frequency
    score = 0.7 * top1_mode_frequency + 0.3 * topk_overlap
    return {
        "score": _round(score),
        "reason": "bootstrap_top1_and_topk_overlap",
        "runs_completed": len(top1s),
        "top1_mode_frequency": _round(top1_mode_frequency),
        "mean_adjacent_topk_overlap": _round(topk_overlap),
    }


def _score_sanity_score(source: str, *, observed_df: pd.DataFrame, candidate_df: pd.DataFrame) -> dict[str, Any]:
    parsed = _run_tool(source, observed_df=observed_df, candidate_df=candidate_df)
    if parsed is None:
        return {
            "score": 0.0,
            "nonfinite_or_parse_failure": True,
            "constant_score_collapse": False,
            "score_explosion": False,
            "rank1_margin": None,
            "near_tie_count": None,
        }
    scores = np.array([float(row["score"]) for row in parsed.ranked_candidates], dtype=float)
    finite = np.isfinite(scores)
    finite_coverage = float(finite.mean()) if len(scores) else 0.0
    finite_scores = scores[finite]
    if finite_scores.size == 0:
        return {
            "score": 0.0,
            "nonfinite_or_parse_failure": True,
            "constant_score_collapse": False,
            "score_explosion": False,
            "rank1_margin": None,
            "near_tie_count": None,
        }
    std = float(np.std(finite_scores))
    span = float(np.max(finite_scores) - np.min(finite_scores))
    ordered = sorted((float(row["score"]) for row in parsed.ranked_candidates), reverse=True)
    rank1_margin = ordered[0] - ordered[1] if len(ordered) > 1 else span
    median_abs = float(np.median(np.abs(finite_scores))) if finite_scores.size else 0.0
    max_abs = float(np.max(np.abs(finite_scores))) if finite_scores.size else 0.0
    ratio = max_abs / max(median_abs, 1e-12)
    score_explosion = max_abs > 1e12 or ratio > 1e9
    constant = finite_scores.size > 1 and span <= 1e-12
    near_tie_threshold = max(abs(ordered[0]) * 1e-8, 1e-10)
    near_tie_count = sum(1 for value in ordered[1:] if abs(ordered[0] - value) <= near_tie_threshold)
    variance_score = 0.0 if constant else min(1.0, math.log10(1.0 + max(std, 0.0)) + 0.25)
    margin_score = min(1.0, max(0.0, abs(rank1_margin)) / (abs(ordered[0]) + 1e-9) * 10.0)
    explosion_score = 0.0 if score_explosion else 1.0
    near_tie_score = max(0.0, 1.0 - near_tie_count / max(len(ordered) - 1, 1))
    score = 0.4 * finite_coverage + 0.2 * variance_score + 0.2 * margin_score + 0.2 * min(explosion_score, near_tie_score)
    return {
        "score": _round(score),
        "finite_score_coverage": _round(finite_coverage),
        "score_std": _safe_float(std),
        "score_span": _safe_float(span),
        "rank1_margin": _safe_float(rank1_margin),
        "near_tie_count": int(near_tie_count),
        "max_abs_score": _safe_float(max_abs),
        "max_abs_to_median_abs_ratio": _safe_float(ratio),
        "nonfinite_or_parse_failure": finite_coverage < 1.0,
        "constant_score_collapse": bool(constant),
        "score_explosion": bool(score_explosion),
    }


def _research_mapping_score(
    source: str,
    *,
    design: dict[str, Any],
    method_primitives: list[dict[str, Any]],
) -> dict[str, Any]:
    _ = method_primitives
    mapping = design.get("method_primitives_used", []) if isinstance(design, dict) else []
    used = [item for item in mapping if isinstance(item, dict) and item.get("status") == "used"]
    if not used:
        return {"score": 0.5, "used_primitive_count": 0, "matched_component_count": 0}
    lowered = source.lower()
    matched = 0
    unmatched: list[str] = []
    for item in used:
        component = str(item.get("intended_code_component", "")).strip().lower()
        tokens = [token for token in component.replace("-", "_").split("_") if len(token) >= 4]
        if component and (component in lowered or any(token in lowered for token in tokens)):
            matched += 1
        else:
            unmatched.append(str(item.get("primitive_id", "")))
    return {
        "score": _round(matched / len(used)),
        "used_primitive_count": len(used),
        "matched_component_count": matched,
        "unmatched_used_primitive_ids": unmatched,
    }


def _feature_coverage_score(
    source: str,
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> dict[str, Any]:
    lowered = source.lower()
    feature_columns = [str(column) for column in candidate_df.columns if str(column) != "candidate_id"]
    explicit_hits = [column for column in feature_columns if column.lower() in lowered]
    generic_feature_matrix = any(token in lowered for token in ("select_dtypes", "get_dummies", "drop(columns", "candidate_df["))
    uses_observed_y = "observed_y" in lowered
    numeric = any(pd.api.types.is_numeric_dtype(candidate_df[column]) for column in candidate_df.columns if str(column) != "candidate_id")
    non_numeric = any(
        not pd.api.types.is_numeric_dtype(candidate_df[column])
        for column in candidate_df.columns
        if str(column) != "candidate_id"
    )
    feature_family_count = 0
    if explicit_hits or generic_feature_matrix:
        feature_family_count += 1
    if numeric and any(token in lowered for token in ("select_dtypes", "to_numeric", "number", "numeric")):
        feature_family_count += 1
    if non_numeric and any(token in lowered for token in ("get_dummies", "astype(str)", "categorical", "category")):
        feature_family_count += 1
    if "candidate_df" in lowered and not explicit_hits and generic_feature_matrix:
        feature_hit_ratio = 0.75
    else:
        feature_hit_ratio = min(1.0, len(explicit_hits) / max(min(len(feature_columns), 4), 1))
    score = 0.55 * float(uses_observed_y) + 0.45 * feature_hit_ratio
    return {
        "score": _round(score),
        "uses_observed_y": bool(uses_observed_y),
        "explicit_feature_column_hits": explicit_hits,
        "generic_feature_matrix_detected": bool(generic_feature_matrix),
        "feature_family_count": int(feature_family_count),
        "observed_columns_available": [str(column) for column in observed_df.columns],
    }


def _simplicity_score(source: str) -> dict[str, Any]:
    line_count = len(str(source).splitlines())
    char_count = len(str(source))
    broad_exception_count = source.count("except Exception") + source.count("except:")
    try_count = source.count("try:")
    score = 1.0
    if line_count > 220 or char_count > 9000:
        score -= 0.30
    elif line_count > 160 or char_count > 7000:
        score -= 0.15
    if broad_exception_count >= 3:
        score -= 0.20
    elif broad_exception_count:
        score -= 0.05
    if try_count >= 5:
        score -= 0.10
    return {
        "score": _round(max(0.0, min(1.0, score))),
        "line_count": line_count,
        "character_count": char_count,
        "broad_exception_count": broad_exception_count,
        "try_count": try_count,
    }


def _row_order_instability(source: str, *, observed_df: pd.DataFrame, candidate_df: pd.DataFrame) -> bool:
    if len(candidate_df) <= 1:
        return False
    first = _run_tool(source, observed_df=observed_df, candidate_df=candidate_df)
    shuffled = _run_tool(
        source,
        observed_df=observed_df,
        candidate_df=candidate_df.sample(frac=1.0, random_state=913).reset_index(drop=True),
    )
    if first is None or shuffled is None:
        return True
    return str(first.selected_display_candidate_id) != str(shuffled.selected_display_candidate_id)


def _hardcoded_candidate_id_count(source: str, candidate_df: pd.DataFrame) -> int:
    candidate_ids = {str(value) for value in candidate_df.get("candidate_id", pd.Series(dtype=str)).tolist()}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    hits = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in candidate_ids:
            hits += 1
    return hits


def _run_tool(source: str, *, observed_df: pd.DataFrame, candidate_df: pd.DataFrame) -> ParsedToolOutput | None:
    try:
        output, _report = execute_rank_candidates_tool(
            tool_source=source,
            observed_df=_public_copy(observed_df),
            candidate_df=_public_copy(candidate_df),
            memory="",
            tool_state={},
        )
        return parse_ranked_candidates(
            output,
            candidate_df=_public_copy(candidate_df),
            observed_df=_public_copy(observed_df),
        )
    except Exception:
        return None


def _public_copy(frame: pd.DataFrame) -> pd.DataFrame:
    copied = frame.copy()
    copied.attrs = {}
    return copied


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _round(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return round(number, 6)
