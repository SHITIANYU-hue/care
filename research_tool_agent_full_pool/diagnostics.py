"""Lightweight deterministic diagnostics for full-pool agent runs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def hash_payload(payload: Any) -> str:
    text = json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=True)
    return sha256_text(text)


def score_distribution_summary(ranked_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["score"]) for row in ranked_candidates if _is_number(row.get("score"))]
    if not scores:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    series = pd.Series(scores, dtype=float)
    return {
        "count": int(series.count()),
        "min": _safe_float(series.min()),
        "max": _safe_float(series.max()),
        "mean": _safe_float(series.mean()),
        "std": _safe_float(series.std()),
    }


def rank1_score_margin(ranked_candidates: list[dict[str, Any]]) -> float | None:
    ordered = sorted(ranked_candidates, key=lambda row: int(row["rank"]))
    if len(ordered) < 2:
        return None
    return _safe_float(float(ordered[0]["score"]) - float(ordered[1]["score"]))


def classify_generated_tool_family(source: str) -> str:
    lowered = str(source).lower()
    if not lowered:
        return "unknown"
    if "ensemble" in lowered or ("ridge" in lowered and "neighbor" in lowered):
        return "ensemble"
    if "randomforest" in lowered or "random_forest" in lowered or "tree" in lowered:
        return "random_forest_or_tree"
    if "ridge" in lowered or "linear" in lowered or "coef" in lowered:
        return "linear_or_ridge"
    if "nearest" in lowered or "neighbor" in lowered or "distance" in lowered:
        return "nearest_neighbor"
    if "effect" in lowered or "scan" in lowered:
        return "feature_effect_scan"
    if "candidate_id" in lowered and "observed_y" not in lowered:
        return "trivial_or_id_sort"
    if any(token in lowered for token in ("score", "observed_y", "feature")):
        return "heuristic"
    return "unknown"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
