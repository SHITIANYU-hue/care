from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from replay_core.schema import ReplayTables


CHEMLEX_INPUT_COLUMNS = ("Acid", "Amine", "Reagents", "Solvent")


def load_chemlex_acidamine(
    path: str | Path,
    *,
    duplicate_policy: str = "raw",
    duplicate_conflict_threshold: float | None = None,
    duplicate_conflict_action: str = "keep",
    candidate_id_policy: str = "sequential",
    row_shuffle_seed: int | None = None,
) -> ReplayTables:
    """Load CHEMLEX Acid-Amine wetlab replay data.

    Product labels and split columns remain evaluator/reporting-side source
    columns and are not exposed through the decision-facing candidate table.
    """

    source_path = Path(path)
    frame = pd.read_csv(source_path).reset_index(drop=True)
    missing = [column for column in (*CHEMLEX_INPUT_COLUMNS, "Conversion") if column not in frame.columns]
    if missing:
        raise ValueError(f"CHEMLEX file is missing required columns: {missing}")
    duplicate_report = _duplicate_conflict_report(
        frame,
        threshold=duplicate_conflict_threshold,
        action=duplicate_conflict_action,
    )
    frame = _apply_duplicate_policy(
        frame,
        duplicate_policy=duplicate_policy,
        conflict_threshold=duplicate_conflict_threshold,
        conflict_action=duplicate_conflict_action,
    )
    if row_shuffle_seed is not None:
        frame = frame.sample(frac=1.0, random_state=int(row_shuffle_seed)).reset_index(drop=True)
    frame.insert(
        0,
        "candidate_id",
        _candidate_ids(
            frame,
            duplicate_policy=duplicate_policy,
            candidate_id_policy=candidate_id_policy,
        ),
    )

    candidate_table = frame.loc[:, ["candidate_id", *CHEMLEX_INPUT_COLUMNS]].copy()
    outcome_table = frame.loc[:, ["candidate_id", "Conversion"]].copy()
    group_table = pd.DataFrame(
        {
            "candidate_id": frame["candidate_id"],
            "acid_identity": frame["Acid"],
            "amine_identity": frame["Amine"],
        }
    )
    flag_table = pd.DataFrame(
        {
            "candidate_id": frame["candidate_id"],
            "objective_missing": outcome_table["Conversion"].isna(),
        }
    )
    return ReplayTables(
        candidate_table=candidate_table,
        outcome_table=outcome_table,
        metadata_table=None,
        group_table=group_table,
        flag_table=flag_table,
        decision_columns=CHEMLEX_INPUT_COLUMNS,
        target_columns=("Conversion",),
        hidden_outcome_columns=(),
        dataset_name="CHEMLEX Acid-Amine Wetlab",
        dataset_identity=_dataset_identity(
            duplicate_policy=duplicate_policy,
            duplicate_conflict_threshold=duplicate_conflict_threshold,
            duplicate_conflict_action=duplicate_conflict_action,
            candidate_id_policy=candidate_id_policy,
            row_shuffle_seed=row_shuffle_seed,
        ),
        source_path=source_path,
        notes=(
            "CHEMLEX objective target is Conversion.",
            "Split labels and product labels are omitted from decision-facing tables.",
            "The released CSV contains duplicate condition tuples; use with an explicit duplicate policy for paper claims.",
            (
                f"duplicate_policy={duplicate_policy}; "
                f"duplicate_conflict_threshold={duplicate_conflict_threshold}; "
                f"duplicate_conflict_action={duplicate_conflict_action}; "
                f"duplicate_conflict_groups={duplicate_report['conflict_group_count']}; "
                f"duplicate_conflict_raw_rows={duplicate_report['conflict_raw_row_count']}; "
                f"candidate_id_policy={candidate_id_policy}; row_shuffle_seed={row_shuffle_seed}"
            ),
        ),
    )


def _apply_duplicate_policy(
    frame: pd.DataFrame,
    *,
    duplicate_policy: str,
    conflict_threshold: float | None = None,
    conflict_action: str = "keep",
) -> pd.DataFrame:
    policy = str(duplicate_policy or "raw").strip().lower()
    if policy in {"raw", "none", "replicate"}:
        result = frame.copy().reset_index(drop=True)
        result["raw_row_index_min"] = range(len(result))
        result["raw_row_index_max"] = range(len(result))
        result["raw_replicate_count"] = 1
        return result
    if policy not in {"unique_mean", "mean", "unique_median", "median"}:
        raise ValueError(
            "Unsupported ChemLex duplicate_policy. Use raw, unique_mean, or unique_median."
        )
    reducer = "median" if "median" in policy else "mean"
    working = frame.copy().reset_index(drop=True)
    working["_raw_row_index"] = range(len(working))
    working = _drop_conflicting_duplicate_groups(
        working,
        threshold=conflict_threshold,
        action=conflict_action,
    )
    grouped = working.groupby(list(CHEMLEX_INPUT_COLUMNS), dropna=False, sort=False)
    aggregated = grouped.agg(
        Conversion=("Conversion", reducer),
        raw_replicate_count=("Conversion", "size"),
        raw_row_index_min=("_raw_row_index", "min"),
        raw_row_index_max=("_raw_row_index", "max"),
    ).reset_index()
    return aggregated.reset_index(drop=True)


def _duplicate_conflict_report(
    frame: pd.DataFrame,
    *,
    threshold: float | None,
    action: str,
) -> dict[str, int]:
    if threshold is None or str(action or "keep").strip().lower() in {"keep", "none", "ignore"}:
        return {"conflict_group_count": 0, "conflict_raw_row_count": 0}
    ranges = _duplicate_group_ranges(frame)
    conflicts = ranges.loc[(ranges["raw_replicate_count"] > 1) & (ranges["conversion_range"] >= float(threshold))]
    return {
        "conflict_group_count": int(len(conflicts)),
        "conflict_raw_row_count": int(conflicts["raw_replicate_count"].sum()) if len(conflicts) else 0,
    }


def _drop_conflicting_duplicate_groups(
    frame: pd.DataFrame,
    *,
    threshold: float | None,
    action: str,
) -> pd.DataFrame:
    normalized_action = str(action or "keep").strip().lower()
    if normalized_action in {"keep", "none", "ignore"} or threshold is None:
        return frame
    if normalized_action not in {"drop", "exclude"}:
        raise ValueError("Unsupported ChemLex duplicate_conflict_action. Use keep or drop.")
    ranges = _duplicate_group_ranges(frame)
    conflicts = ranges.loc[(ranges["raw_replicate_count"] > 1) & (ranges["conversion_range"] >= float(threshold))]
    if conflicts.empty:
        return frame
    marker = conflicts.loc[:, list(CHEMLEX_INPUT_COLUMNS)].assign(_drop_conflicting_duplicate_group=True)
    merged = frame.merge(marker, on=list(CHEMLEX_INPUT_COLUMNS), how="left")
    keep = merged["_drop_conflicting_duplicate_group"].isna()
    return frame.loc[keep.to_numpy()].reset_index(drop=True)


def _duplicate_group_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "Conversion" not in working.columns:
        raise ValueError("CHEMLEX conflict filtering requires a Conversion column.")
    working["_conversion_numeric"] = pd.to_numeric(working["Conversion"], errors="coerce")
    grouped = working.groupby(list(CHEMLEX_INPUT_COLUMNS), dropna=False, sort=False)
    ranges = grouped.agg(
        raw_replicate_count=("_conversion_numeric", "size"),
        conversion_min=("_conversion_numeric", "min"),
        conversion_max=("_conversion_numeric", "max"),
    ).reset_index()
    ranges["conversion_range"] = ranges["conversion_max"] - ranges["conversion_min"]
    return ranges


def _candidate_ids(
    frame: pd.DataFrame,
    *,
    duplicate_policy: str,
    candidate_id_policy: str,
) -> list[str]:
    policy = str(candidate_id_policy or "sequential").strip().lower()
    if policy in {"sequential", "raw_order", "row_order"}:
        return [f"chemlex_acidamine_{idx:05d}" for idx in range(len(frame))]
    if policy not in {"public_hash", "hash", "opaque_hash"}:
        raise ValueError("Unsupported ChemLex candidate_id_policy. Use sequential or public_hash.")
    ids: list[str] = []
    seen: dict[str, int] = {}
    include_raw = str(duplicate_policy or "raw").strip().lower() in {"raw", "none", "replicate"}
    for _, row in frame.iterrows():
        payload: dict[str, Any] = {column: _cell(row.get(column)) for column in CHEMLEX_INPUT_COLUMNS}
        if include_raw:
            payload["raw_row_index_min"] = int(row.get("raw_row_index_min", len(ids)))
        digest = hashlib.sha256(
            "|".join(f"{key}={payload[key]}" for key in sorted(payload)).encode("utf-8")
        ).hexdigest()[:16]
        base = f"chemlex_{digest}"
        suffix = seen.get(base, 0)
        seen[base] = suffix + 1
        ids.append(base if suffix == 0 else f"{base}_{suffix:02d}")
    return ids


def _cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _dataset_identity(
    *,
    duplicate_policy: str,
    duplicate_conflict_threshold: float | None,
    duplicate_conflict_action: str,
    candidate_id_policy: str,
    row_shuffle_seed: int | None,
) -> str:
    base = "chemlex_acidamine_wetlab"
    if str(duplicate_policy).lower() in {"raw", "none", "replicate"} and str(candidate_id_policy).lower() in {
        "sequential",
        "raw_order",
        "row_order",
    } and row_shuffle_seed is None:
        return base
    suffix = f"dup-{duplicate_policy}_ids-{candidate_id_policy}"
    if duplicate_conflict_threshold is not None and str(duplicate_conflict_action or "keep").strip().lower() not in {
        "keep",
        "none",
        "ignore",
    }:
        threshold = float(duplicate_conflict_threshold)
        suffix += f"_conflict-{str(duplicate_conflict_action).lower()}-{threshold:g}"
    if row_shuffle_seed is not None:
        suffix += f"_shuffle-{int(row_shuffle_seed)}"
    return f"{base}_{suffix}"
