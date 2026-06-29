"""Fake generated-tool client for no-API smoke tests."""

from __future__ import annotations

from typing import Any


class FakeResearchToolGenerator:
    """Deterministic fake client used before live API integration exists."""

    def create_initial_tool(self, *args: Any, **kwargs: Any) -> str:
        """Return deterministic generated tool source without any API calls."""

        return FAKE_FULL_POOL_TOOL_SOURCE


FAKE_FULL_POOL_TOOL_SOURCE = '''
def rank_candidates(observed_df, candidate_df, memory=None, tool_state=None):
    def to_float(value, default=0.0):
        try:
            if value is None:
                return default
            if value != value:
                return default
            return float(value)
        except Exception:
            return default

    def safe_norm(value, low, high):
        span = max(float(high) - float(low), 1e-12)
        normalized = (float(value) - float(low)) / span
        return max(0.0, min(1.0, normalized))

    def median(values):
        ordered = sorted(values)
        if not ordered:
            return 0.0
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return 0.5 * (float(ordered[middle - 1]) + float(ordered[middle]))

    feature_columns = [
        column
        for column in list(candidate_df.columns)
        if column not in ("candidate_id", "observation_id", "observed_y")
    ]
    numeric_columns = []
    for column in feature_columns:
        values = [to_float(value, None) for value in candidate_df[column].tolist()]
        values = [value for value in values if value is not None]
        if values:
            numeric_columns.append(column)

    observed_rows = []
    for _, row in observed_df.iterrows():
        observed_rows.append(
            {
                "observation_id": str(row.get("observation_id", "")),
                "observed_y": to_float(row.get("observed_y"), 0.0),
                "features": {column: to_float(row.get(column), 0.0) for column in numeric_columns},
            }
        )

    stats = {}
    for column in numeric_columns:
        values = [to_float(value, 0.0) for value in candidate_df[column].tolist()]
        low = min(values) if values else 0.0
        high = max(values) if values else 1.0
        stats[column] = {"low": low, "high": high, "median": median(values)}

    y_values = [row["observed_y"] for row in observed_rows]
    y_mean = sum(y_values) / max(len(y_values), 1) if y_values else 0.0
    y_best = max(y_values) if y_values else y_mean
    best_row = None
    if observed_rows:
        best_row = sorted(observed_rows, key=lambda item: (item["observed_y"], item["observation_id"]), reverse=True)[0]
    evidence_refs = [best_row["observation_id"]] if best_row and best_row["observation_id"] else []

    slopes = {}
    for column in numeric_columns:
        xs = [row["features"].get(column, 0.0) for row in observed_rows]
        if len(xs) < 2:
            slopes[column] = 0.0
            continue
        x_mean = sum(xs) / len(xs)
        numerator = sum((x - x_mean) * (row["observed_y"] - y_mean) for x, row in zip(xs, observed_rows))
        denominator = sum((x - x_mean) * (x - x_mean) for x in xs)
        slopes[column] = numerator / denominator if denominator > 1e-12 else 0.0

    ligand_columns = [column for column in numeric_columns if column.startswith("L")]
    ligand_seen = {}
    ligand_best = {}
    for column in ligand_columns:
        seen = 0
        best = None
        for row in observed_rows:
            if row["features"].get(column, 0.0) >= 0.5:
                seen += 1
                best = row["observed_y"] if best is None else max(best, row["observed_y"])
        ligand_seen[column] = seen
        ligand_best[column] = y_mean if best is None else best

    has_minerva_columns = (
        "temperature" in numeric_columns
        and "catalyst_loading" in numeric_columns
        and "res_time" in numeric_columns
        and "L3" in numeric_columns
    )

    ranked = []
    for _, row in candidate_df.iterrows():
        predicted = y_mean
        best_distance = 0.0
        edge_bonus = 0.0
        present = 0
        for column in numeric_columns:
            value = to_float(row.get(column), 0.0)
            stat = stats[column]
            normalized = safe_norm(value, stat["low"], stat["high"])
            predicted += slopes.get(column, 0.0) * (value - stat["median"])
            edge_bonus += max(normalized, 1.0 - normalized)
            if best_row is not None:
                best_value = to_float(best_row["features"].get(column), stat["median"])
                best_distance += abs(normalized - safe_norm(best_value, stat["low"], stat["high"]))
            present += 1

        ligand_bonus = 0.0
        for column in ligand_columns:
            if to_float(row.get(column), 0.0) >= 0.5:
                # Prefer ligand families with good public observations, but
                # keep a smaller bonus for under-sampled families.
                ligand_bonus += 0.020 * (ligand_best.get(column, y_mean) - y_mean)
                ligand_bonus += 0.050 / (1.0 + float(ligand_seen.get(column, 0)))

        domain_prior = 0.0
        if has_minerva_columns:
            temp = safe_norm(to_float(row.get("temperature"), 0.0), stats["temperature"]["low"], stats["temperature"]["high"])
            loading = safe_norm(
                to_float(row.get("catalyst_loading"), 0.0),
                stats["catalyst_loading"]["low"],
                stats["catalyst_loading"]["high"],
            )
            time = safe_norm(to_float(row.get("res_time"), 0.0), stats["res_time"]["low"], stats["res_time"]["high"])
            l3 = to_float(row.get("L3"), 0.0)
            # Public reaction-design prior: explore higher temperature/loading
            # and the L3 family early, without reading any hidden labels.
            domain_prior = 0.80 * temp + 1.20 * loading + 0.35 * (1.0 - abs(time - 0.35)) + 1.00 * l3

        denom = max(present, 1)
        exploration = 0.04 * (edge_bonus / denom)
        local_penalty = -0.01 * (best_distance / denom) if best_row is not None else 0.0
        exploit_weight = 0.55 if len(observed_rows) < 12 else 0.70
        score = float(exploit_weight * predicted + (1.0 - exploit_weight) * y_best)
        score += float(domain_prior + ligand_bonus + exploration + local_penalty)
        ranked.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "score": score,
                "reason_code": "public_surrogate_domain_exploration_score",
                "evidence_refs": list(evidence_refs),
            }
        )

    ranked = sorted(ranked, key=lambda item: (-item["score"], item["candidate_id"]))
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    next_state = dict(tool_state or {})
    next_state["last_candidate_count"] = len(ranked)
    next_state["score_call_count"] = int(next_state.get("score_call_count", 0)) + 1
    return {
        "ranked_candidates": ranked,
        "tool_state": next_state,
        "tool_diagnostics": {
            "scored_candidate_count": len(ranked),
            "observed_count": len(observed_rows),
            "feature_count": len(numeric_columns),
            "score_mode": "public_surrogate_domain_exploration",
        },
    }
'''
