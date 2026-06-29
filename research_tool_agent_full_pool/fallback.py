"""Fallback selection helpers for the full-pool ResearchToolAgent."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


def full_pool_random_fallback(
    candidate_df: pd.DataFrame,
    *,
    seed: int,
    round_index: int,
    reason: str,
) -> dict[str, Any]:
    """Select randomly from the full remaining eligible candidate pool.

    This fallback must not call BO, must not use menu restriction, and must not
    read hidden outcomes or candidate score artifacts.
    """

    if "candidate_id" not in candidate_df.columns:
        raise ValueError("candidate_df must contain candidate_id for full-pool fallback.")
    candidate_ids = sorted(str(value) for value in candidate_df["candidate_id"].tolist())
    if not candidate_ids:
        raise ValueError("Cannot fallback-select from an empty candidate pool.")
    payload = f"full_pool_random|{seed}|{round_index}|{reason}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    index = int.from_bytes(digest, "big") % len(candidate_ids)
    selected = candidate_ids[index]
    return {
        "candidate_id": selected,
        "fallback_reason": str(reason),
        "fallback_pool_size": len(candidate_ids),
        "fallback_is_full_pool": True,
        "fallback_menu_restricted": False,
    }
