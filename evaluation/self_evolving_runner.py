"""Runner helpers for the self-evolving full-pool harness."""

from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import fields
from typing import Any

import numpy as np
import pandas as pd

from datasets.chemlex import load_chemlex_acidamine
from datasets.minerva import load_minerva_olympus_suzuki
from replay_core.evaluator import OfflineEvaluator
from replay_core.schema import ReplayTables
from research_tool_agent_full_pool.api_client import ApiKeyMissingError, CommonstackToolSynthesisClient
from research_tool_agent_full_pool.harness.ledger import write_json
from research_tool_agent_full_pool.harness.orchestrator import SelfEvolvingConfig, SelfEvolvingFullPoolAgent
from research_tool_agent_full_pool.initial_observed import initialize_full_pool_replay_state


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "minerva_public_replay.json"


def run_self_evolving_smoke(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = _resolve_output_dir(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = SelfEvolvingConfig(**{**config.__dict__, "output_dir": str(output_dir)})
    try:
        client = _client_for_config(config)
    except ApiKeyMissingError:
        if config.mode == "api":
            summary = {
                "status": "api_key_missing",
                "output_dir": str(output_dir),
                "message": "API key missing; API-backed replay was not run.",
            }
            write_json(output_dir / "self_evolving_summary.json", summary)
            print(json.dumps(summary, sort_keys=True, indent=2))
            return summary
        client = None
    tables, table_source = _load_tables(config)
    evaluator = OfflineEvaluator.from_tables(tables)
    replay_state = initialize_full_pool_replay_state(
        tables=tables,
        evaluator=evaluator,
        seed=config.seed,
        initial_observed_count=min(config.initial_observed_count, max(1, len(tables.candidate_table) - 1)),
    )
    agent = SelfEvolvingFullPoolAgent(config=config, client=client)
    try:
        agent.initialize_run(tables=tables, replay_state=replay_state)
        while replay_state.can_continue(config.max_rounds):
            decision = agent.decide(tables=tables, replay_state=replay_state)
            revealed = evaluator.reveal(decision.selected_candidate_ids)
            replay_state.observe(revealed)
            agent.update_after_reveal(
                tables=tables,
                replay_state=replay_state,
                decision=decision,
                revealed_rows=revealed,
            )
    except Exception as exc:
        summary = {
            "status": "failed",
            "error": f"{exc.__class__.__name__}: {str(exc)[:500]}",
            "output_dir": str(output_dir),
            "configured_dataset_name": config.dataset_name,
            "dataset_name": getattr(tables, "dataset_name", None),
            "table_source": table_source,
        }
        write_json(output_dir / "self_evolving_summary.json", summary)
        raise
    assert agent.state is not None
    summary = {
        "status": "ok",
        "error": None,
        "output_dir": str(output_dir),
        "configured_dataset_name": config.dataset_name,
        "dataset_name": tables.dataset_name,
        "table_source": table_source,
        "synthetic_fallback_used": table_source == "synthetic_fallback",
        "selected_count": int(agent.state.strategy_state.get("selected_count", 0)),
        "fallback_count": int(agent.state.strategy_state.get("fallback_count", 0)),
        "planner_call_count": int(agent.state.strategy_state.get("planner_call_count", 0)),
        "skill_synthesis_count": int(agent.state.strategy_state.get("skill_synthesis_count", 0)),
        "gate_pass_count": int(agent.state.strategy_state.get("gate_pass_count", 0)),
        "gate_reject_count": int(agent.state.strategy_state.get("gate_reject_count", 0)),
        "reward_count": int(agent.state.strategy_state.get("reward_count", 0)),
        "round_summaries": agent.state.round_summaries,
    }
    write_json(output_dir / "self_evolving_summary.json", summary)
    print(json.dumps(_compact_summary(summary), sort_keys=True, indent=2))
    return summary


def _load_config(path: str | Path) -> SelfEvolvingConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        return SelfEvolvingConfig()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    known_keys = {field.name for field in fields(SelfEvolvingConfig)} | {"dataset", "planner_mode"}
    unknown_keys = sorted(set(payload) - known_keys)
    if unknown_keys:
        raise ValueError(
            "Unsupported config fields in "
            f"{config_path}: {unknown_keys}. "
            "This runner intentionally rejects stale experiment switches so old configs do not silently "
            "run under the current v5 mainline."
        )
    return SelfEvolvingConfig(
        run_id=str(payload.get("run_id", SelfEvolvingConfig.run_id)),
        dataset_name=str(payload.get("dataset_name", payload.get("dataset", SelfEvolvingConfig.dataset_name))),
        decision_policy_name=str(payload.get("decision_policy_name", SelfEvolvingConfig.decision_policy_name)),
        output_dir=str(payload.get("output_dir", SelfEvolvingConfig.output_dir)),
        mode=str(payload.get("mode", payload.get("planner_mode", SelfEvolvingConfig.mode))),
        seed=int(payload.get("seed", SelfEvolvingConfig.seed)),
        max_rounds=int(payload.get("max_rounds", SelfEvolvingConfig.max_rounds)),
        initial_observed_count=int(payload.get("initial_observed_count", SelfEvolvingConfig.initial_observed_count)),
        target_column=str(payload.get("target_column", SelfEvolvingConfig.target_column)),
        objective_direction=str(payload.get("objective_direction", SelfEvolvingConfig.objective_direction)),
        repair_attempts=int(payload.get("repair_attempts", SelfEvolvingConfig.repair_attempts)),
        model=str(payload.get("model", SelfEvolvingConfig.model)),
        reasoning_effort=str(payload.get("reasoning_effort", SelfEvolvingConfig.reasoning_effort)),
        max_tokens=int(payload.get("max_tokens", SelfEvolvingConfig.max_tokens)),
        timeout=float(payload.get("timeout", SelfEvolvingConfig.timeout)),
        allow_synthetic_fallback=_as_bool(
            payload.get("allow_synthetic_fallback", SelfEvolvingConfig.allow_synthetic_fallback)
        ),
        resume_policy_state=_as_bool(payload.get("resume_policy_state", SelfEvolvingConfig.resume_policy_state)),
        api_mode=str(payload.get("api_mode", SelfEvolvingConfig.api_mode)),
        stream=_as_bool(payload.get("stream", SelfEvolvingConfig.stream)),
        response_verbosity=str(payload.get("response_verbosity", SelfEvolvingConfig.response_verbosity)),
        chemlex_duplicate_policy=str(
            payload.get("chemlex_duplicate_policy", SelfEvolvingConfig.chemlex_duplicate_policy)
        ),
        chemlex_duplicate_conflict_threshold=(
            None
            if payload.get(
                "chemlex_duplicate_conflict_threshold",
                SelfEvolvingConfig.chemlex_duplicate_conflict_threshold,
            )
            is None
            else float(payload.get("chemlex_duplicate_conflict_threshold"))
        ),
        chemlex_duplicate_conflict_action=str(
            payload.get(
                "chemlex_duplicate_conflict_action",
                SelfEvolvingConfig.chemlex_duplicate_conflict_action,
            )
        ),
        chemlex_candidate_id_policy=str(
            payload.get("chemlex_candidate_id_policy", SelfEvolvingConfig.chemlex_candidate_id_policy)
        ),
        chemlex_row_shuffle_seed=(
            None
            if payload.get("chemlex_row_shuffle_seed", SelfEvolvingConfig.chemlex_row_shuffle_seed) is None
            else int(payload.get("chemlex_row_shuffle_seed"))
        ),
        adaptive_categorical_experts=_as_bool(
            payload.get("adaptive_categorical_experts", SelfEvolvingConfig.adaptive_categorical_experts)
        ),
        parse_failure_reuse_active=_as_bool(
            payload.get("parse_failure_reuse_active", SelfEvolvingConfig.parse_failure_reuse_active)
        ),
        api_parse_retry_attempts=int(
            payload.get("api_parse_retry_attempts", SelfEvolvingConfig.api_parse_retry_attempts)
        ),
        llm_residual_scout_enabled=_as_bool(
            payload.get("llm_residual_scout_enabled", SelfEvolvingConfig.llm_residual_scout_enabled)
        ),
        llm_residual_scout_public_locked=_as_bool(
            payload.get("llm_residual_scout_public_locked", SelfEvolvingConfig.llm_residual_scout_public_locked)
        ),
        llm_residual_scout_budget=int(
            payload.get("llm_residual_scout_budget", SelfEvolvingConfig.llm_residual_scout_budget)
        ),
        llm_residual_scout_top_k=int(
            payload.get("llm_residual_scout_top_k", SelfEvolvingConfig.llm_residual_scout_top_k)
        ),
        llm_residual_scout_min_round=int(
            payload.get("llm_residual_scout_min_round", SelfEvolvingConfig.llm_residual_scout_min_round)
        ),
        llm_residual_scout_best_threshold=float(
            payload.get("llm_residual_scout_best_threshold", SelfEvolvingConfig.llm_residual_scout_best_threshold)
        ),
        llm_residual_scout_stagnation_best_threshold=float(
            payload.get(
                "llm_residual_scout_stagnation_best_threshold",
                SelfEvolvingConfig.llm_residual_scout_stagnation_best_threshold,
            )
        ),
        llm_residual_scout_min_public_support=int(
            payload.get(
                "llm_residual_scout_min_public_support",
                SelfEvolvingConfig.llm_residual_scout_min_public_support,
            )
        ),
        llm_residual_scout_public_rank_limit=int(
            payload.get(
                "llm_residual_scout_public_rank_limit",
                SelfEvolvingConfig.llm_residual_scout_public_rank_limit,
            )
        ),
        llm_residual_scout_min_certificate_score=float(
            payload.get(
                "llm_residual_scout_min_certificate_score",
                SelfEvolvingConfig.llm_residual_scout_min_certificate_score,
            )
        ),
        llm_residual_scout_chemlex_model_band_enabled=_as_bool(
            payload.get(
                "llm_residual_scout_chemlex_model_band_enabled",
                SelfEvolvingConfig.llm_residual_scout_chemlex_model_band_enabled,
            )
        ),
        llm_residual_scout_chemlex_model_band_min_round=int(
            payload.get(
                "llm_residual_scout_chemlex_model_band_min_round",
                SelfEvolvingConfig.llm_residual_scout_chemlex_model_band_min_round,
            )
        ),
        llm_residual_scout_chemlex_anchor_guard_threshold=float(
            payload.get(
                "llm_residual_scout_chemlex_anchor_guard_threshold",
                SelfEvolvingConfig.llm_residual_scout_chemlex_anchor_guard_threshold,
            )
        ),
        llm_macro_frontier_scout_enabled=_as_bool(
            payload.get(
                "llm_macro_frontier_scout_enabled",
                SelfEvolvingConfig.llm_macro_frontier_scout_enabled,
            )
        ),
        llm_macro_frontier_scout_budget=int(
            payload.get(
                "llm_macro_frontier_scout_budget",
                SelfEvolvingConfig.llm_macro_frontier_scout_budget,
            )
        ),
        llm_macro_frontier_scout_min_round=int(
            payload.get(
                "llm_macro_frontier_scout_min_round",
                SelfEvolvingConfig.llm_macro_frontier_scout_min_round,
            )
        ),
        llm_macro_frontier_scout_min_best_threshold=float(
            payload.get(
                "llm_macro_frontier_scout_min_best_threshold",
                SelfEvolvingConfig.llm_macro_frontier_scout_min_best_threshold,
            )
        ),
        llm_macro_frontier_scout_low_best_threshold=float(
            payload.get(
                "llm_macro_frontier_scout_low_best_threshold",
                SelfEvolvingConfig.llm_macro_frontier_scout_low_best_threshold,
            )
        ),
        llm_macro_frontier_scout_high_confidence_best_threshold=float(
            payload.get(
                "llm_macro_frontier_scout_high_confidence_best_threshold",
                SelfEvolvingConfig.llm_macro_frontier_scout_high_confidence_best_threshold,
            )
        ),
        care_enabled=_as_bool(payload.get("care_enabled", SelfEvolvingConfig.care_enabled)),
        care_certificate_mode=str(payload.get("care_certificate_mode", SelfEvolvingConfig.care_certificate_mode)),
        care_adaptive_planner_enabled=_as_bool(
            payload.get("care_adaptive_planner_enabled", SelfEvolvingConfig.care_adaptive_planner_enabled)
        ),
        care_certificate_margin=float(
            payload.get("care_certificate_margin", SelfEvolvingConfig.care_certificate_margin)
        ),
    )


def _client_for_config(config: SelfEvolvingConfig) -> CommonstackToolSynthesisClient | None:
    if config.mode != "api":
        return None
    endpoint = _normalize_endpoint(
        os.environ.get("COMMONSTACK_API_ENDPOINT", "https://api.commonstack.ai/v1"),
        api_mode=config.api_mode,
    )
    return CommonstackToolSynthesisClient(
        endpoint=endpoint,
        model=config.model,
        temperature=0.0,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        response_format_json_object=True,
        reasoning_effort=config.reasoning_effort,
        api_mode=config.api_mode,
        stream=config.stream,
        response_verbosity=config.response_verbosity,
    )


def _normalize_endpoint(endpoint: str, *, api_mode: str = "chat") -> str:
    text = str(endpoint).strip().rstrip("/")
    mode = str(api_mode).strip().lower()
    if mode == "responses":
        if text.endswith("/responses"):
            return text
        if text.endswith("/chat/completions"):
            return text[: -len("/chat/completions")] + "/responses"
        if text.endswith("/v1"):
            return text + "/responses"
        return text
    if text.endswith("/chat/completions"):
        return text
    if text.endswith("/v1"):
        return text + "/chat/completions"
    return text


def _load_tables(config: SelfEvolvingConfig) -> tuple[ReplayTables, str]:
    dataset_name = str(config.dataset_name).lower()
    if "chemlex" in dataset_name or "acid" in dataset_name:
        chemlex_path = _first_existing_path(
            REPO_ROOT / "data" / "chemlex" / "acid_amine_wetlab.csv",
            REPO_ROOT / "datasets" / "chemlex" / "acid_amine_wetlab.csv",
        )
        if chemlex_path is not None:
            return (
                load_chemlex_acidamine(
                    chemlex_path,
                    duplicate_policy=config.chemlex_duplicate_policy,
                    duplicate_conflict_threshold=config.chemlex_duplicate_conflict_threshold,
                    duplicate_conflict_action=config.chemlex_duplicate_conflict_action,
                    candidate_id_policy=config.chemlex_candidate_id_policy,
                    row_shuffle_seed=config.chemlex_row_shuffle_seed,
                ),
                "chemlex_csv",
            )
        if not config.allow_synthetic_fallback:
            raise FileNotFoundError(
                "CHEMLEX CSV not found at data/chemlex/acid_amine_wetlab.csv. "
                "Set allow_synthetic_fallback=true only for local synthetic checks."
            )

    minerva_path = _first_existing_path(
        REPO_ROOT / "data" / "minerva" / "suzuki_i.csv",
        REPO_ROOT / "datasets" / "minerva" / "suzuki_i.csv",
    )
    if minerva_path is not None:
        return load_minerva_olympus_suzuki(minerva_path), "minerva_csv"
    if config.allow_synthetic_fallback:
        return synthetic_replay_tables(), "synthetic_fallback"
    raise FileNotFoundError(
        "MINERVA CSV not found at "
        "data/minerva/suzuki_i.csv. Set allow_synthetic_fallback=true only for local synthetic checks."
    )


def _first_existing_path(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def synthetic_replay_tables(n: int = 48, *, seed: int = 0) -> ReplayTables:
    rng = np.random.default_rng(seed)
    feature_a = np.linspace(0.0, 1.0, n)
    feature_b = rng.uniform(0.0, 1.0, size=n)
    ligand = np.arange(n) % 4
    y = 0.35 + 0.45 * feature_a + 0.15 * np.sin(3.0 * feature_b) + 0.03 * ligand
    candidate_table = pd.DataFrame(
        {
            "candidate_id": [f"syn_{idx:04d}" for idx in range(n)],
            "feature_a": feature_a,
            "feature_b": feature_b,
            "ligand_code": ligand,
        }
    )
    outcome_table = pd.DataFrame(
        {
            "candidate_id": candidate_table["candidate_id"],
            "yield": y,
            "turnover": 10.0 + y,
        }
    )
    return ReplayTables(
        candidate_table=candidate_table,
        outcome_table=outcome_table,
        target_columns=("yield",),
        hidden_outcome_columns=("turnover",),
        decision_columns=("feature_a", "feature_b", "ligand_code"),
        dataset_name="SyntheticFinitePool",
        dataset_identity="synthetic_finite_pool_v0",
    )


def _resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": summary.get("status"),
        "dataset_name": summary.get("dataset_name"),
        "selected_count": summary.get("selected_count"),
        "fallback_count": summary.get("fallback_count"),
        "planner_call_count": summary.get("planner_call_count"),
        "skill_synthesis_count": summary.get("skill_synthesis_count"),
        "gate_pass_count": summary.get("gate_pass_count"),
        "gate_reject_count": summary.get("gate_reject_count"),
        "reward_count": summary.get("reward_count"),
        "output_dir": summary.get("output_dir"),
        "configured_dataset_name": summary.get("configured_dataset_name"),
        "table_source": summary.get("table_source"),
        "synthetic_fallback_used": summary.get("synthetic_fallback_used"),
        "error": summary.get("error"),
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    return bool(value)


if __name__ == "__main__":
    run_self_evolving_smoke()
