from __future__ import annotations

from pathlib import Path

import pandas as pd

from replay_core.schema import ReplayTables


MINERVA_INPUT_COLUMNS = (
    "L0",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "res_time",
    "temperature",
    "catalyst_loading",
)


def _drop_unnamed_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    raw_index = frame.get("Unnamed: 0", pd.Series(range(len(frame)), name="raw_row_index"))
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed:")]
    return frame.drop(columns=unnamed), raw_index.reset_index(drop=True)


def load_minerva_olympus_suzuki(path: str | Path) -> ReplayTables:
    """Load MINERVA Suzuki replay data with turnover hidden evaluator-side."""

    source_path = Path(path)
    frame = pd.read_csv(source_path)
    frame, raw_row_index = _drop_unnamed_columns(frame)
    frame = frame.reset_index(drop=True)
    frame.insert(0, "candidate_id", [f"minerva_suzuki_i_{idx:05d}" for idx in range(len(frame))])

    missing = [column for column in (*MINERVA_INPUT_COLUMNS, "yield", "turnover") if column not in frame.columns]
    if missing:
        raise ValueError(f"MINERVA file is missing required columns: {missing}")

    ligand_identity = (
        frame.loc[:, [f"L{i}" for i in range(7)]]
        .idxmax(axis=1)
        .fillna("unknown_ligand")
        .rename("ligand_identity")
    )

    candidate_table = frame.loc[:, ["candidate_id", *MINERVA_INPUT_COLUMNS]].copy()
    outcome_table = frame.loc[:, ["candidate_id", "yield", "turnover"]].copy()
    metadata_table = pd.DataFrame(
        {
            "candidate_id": frame["candidate_id"],
            "raw_row_index": raw_row_index.to_numpy(),
            "source_path": str(source_path),
        }
    )
    group_table = pd.DataFrame(
        {
            "candidate_id": frame["candidate_id"],
            "ligand_identity": ligand_identity.to_numpy(),
        }
    )
    flag_table = pd.DataFrame(
        {
            "candidate_id": frame["candidate_id"],
            "objective_missing": outcome_table["yield"].isna(),
        }
    )

    return ReplayTables(
        candidate_table=candidate_table,
        outcome_table=outcome_table,
        metadata_table=metadata_table,
        group_table=group_table,
        flag_table=flag_table,
        decision_columns=MINERVA_INPUT_COLUMNS,
        target_columns=("yield",),
        hidden_outcome_columns=("turnover",),
        dataset_name="MINERVA Olympus Suzuki I",
        dataset_identity="minerva_olympus_suzuki_i",
        source_path=source_path,
        notes=(
            "MINERVA objective target is yield.",
            "turnover is hidden_outcome/evaluator-only and is revealed only for selected candidates.",
        ),
    )
