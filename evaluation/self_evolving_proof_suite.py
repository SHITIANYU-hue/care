"""Small proof suite for validating true self-evolving behavior before full runs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import math
import re
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_extraction import FeatureHasher

from replay_core.evaluator import OfflineEvaluator
from research_tool_agent_full_pool.fake_client import FAKE_FULL_POOL_TOOL_SOURCE
from research_tool_agent_full_pool.fallback import full_pool_random_fallback
from research_tool_agent_full_pool.harness.gate import run_conservative_gate
from research_tool_agent_full_pool.harness.ledger import write_json
from research_tool_agent_full_pool.harness.orchestrator import (
    SelfEvolvingConfig,
    SelfEvolvingFullPoolAgent,
    _build_public_expert_rankings,
    _choose_sparse_category_route,
    _expert_from_ranked_rows,
    _fixed_public_source,
    _public_categorical_shrinkage_rank,
    _public_gp_ei_rank,
    _public_rf_ucb_rank,
    _public_data_profile,
    _run_source_ranker_rows,
    _score_expert_candidates,
)
from research_tool_agent_full_pool.harness.skill_registry import SkillRegistry
from research_tool_agent_full_pool.harness.specs import SkillArtifact
from research_tool_agent_full_pool.initial_observed import initialize_full_pool_replay_state
from research_tool_agent_full_pool.tool_output_parser import parse_ranked_candidates
from research_tool_agent_full_pool.tool_runner import run_rank_candidates_tool
from research_tool_agent_full_pool.views import (
    build_full_remaining_candidate_df,
    build_observed_df_from_revealed_state,
    map_display_candidate_to_internal_id,
)

from evaluation.self_evolving_runner import REPO_ROOT, _client_for_config, _load_config, _load_tables


LMABO_STYLE_POLICY = "lmabo_style_nearest_neighbor_llm_bo"
LMABO_ACQUISITIONS = (
    "PI",
    "LogPI",
    "EI",
    "LogEI",
    "UCB",
    "PosMean",
    "PosSTD",
    "TS",
    "qKG",
    "qPES",
    "qMES",
    "qJES",
)
LMABO_DEFAULT_ACQUISITION = "UCB"
LMABO_OFFICIAL_REFERENCE = "giang-n-ngo/lmabo@b1671f5ad"


PROOF_POLICIES = (
    "random_full_pool",
    "stratified_random_public",
    "classical_bo_gp_ei",
    "classical_bo_gp_ucb",
    "bo_like_surrogate",
    "botorch_style_gp_logei",
    "smac_style_rf_ei",
    "tpe_style_bo",
    "chemistry_descriptor_bo",
    "edbo_style_descriptor_gp_ei",
    "gryffin_style_categorical_bo",
    "baybe_bofire_style_mixed_bo",
    "fixed_public_heuristic",
    "categorical_empirical_bayes_ucb",
    "llm_assisted_one_shot_api",
    "fixed_api_tool",
    "no_evolve_api_reuse",
    "true_self_evolving_api",
    "shared_initial_no_evolve_api",
)

ABLATION_POLICIES = (
    "public_expert_only_meta_controller",
    "llm_only_self_evolving",
)

CARE_POLICIES = (
    "true_self_evolving_api_care",
    "true_self_evolving_api_care_log_only",
    "true_self_evolving_api_care_no_adaptive_planner",
    "true_self_evolving_api_care_no_certificate",
    "true_self_evolving_api_care_no_residual_scout",
    "true_self_evolving_api_care_no_macro_scout",
)

PUBLIC_CAPACITY_ABLATION_POLICIES = (
    "public_expert_only_meta_controller_4experts",
    "public_expert_only_meta_controller_6experts",
)

SUPPORTED_PROOF_POLICIES = (
    PROOF_POLICIES + ABLATION_POLICIES + CARE_POLICIES + PUBLIC_CAPACITY_ABLATION_POLICIES + (LMABO_STYLE_POLICY,)
)

API_POLICIES = {
    "llm_assisted_one_shot_api",
    "fixed_api_tool",
    "no_evolve_api_reuse",
    "true_self_evolving_api",
    "llm_only_self_evolving",
    LMABO_STYLE_POLICY,
    *CARE_POLICIES,
}

EMNLP_NON_API_BASELINE_POLICIES = (
    "random_full_pool",
    "stratified_random_public",
    "fixed_public_heuristic",
    "classical_bo_gp_ei",
    "classical_bo_gp_ucb",
    "bo_like_surrogate",
    "botorch_style_gp_logei",
    "smac_style_rf_ei",
    "tpe_style_bo",
    "chemistry_descriptor_bo",
    "edbo_style_descriptor_gp_ei",
    "gryffin_style_categorical_bo",
    "baybe_bofire_style_mixed_bo",
    "categorical_empirical_bayes_ucb",
    "public_expert_only_meta_controller",
)

EMNLP_API_METHOD_POLICIES = ("true_self_evolving_api_care",)

EMNLP_API_BASELINE_POLICIES = (LMABO_STYLE_POLICY,)

EMNLP_API_ABLATION_POLICIES = (
    "true_self_evolving_api_care_no_adaptive_planner",
    "true_self_evolving_api_care_no_certificate",
    "true_self_evolving_api_care_no_residual_scout",
    "true_self_evolving_api_care_no_macro_scout",
    "llm_only_self_evolving",
    "no_evolve_api_reuse",
)

EMNLP_MAIN_POLICIES = (
    EMNLP_NON_API_BASELINE_POLICIES
    + EMNLP_API_BASELINE_POLICIES
    + EMNLP_API_METHOD_POLICIES
    + EMNLP_API_ABLATION_POLICIES
)


def run_proof_suite(
    config_path: str | Path,
    *,
    seeds: list[int],
    max_rounds: int,
    output_dir: str | Path,
    policies: list[str] | None = None,
    parallel_workers: int = 1,
    executor_type: str = "thread",
) -> dict[str, Any]:
    """Run a small, matched-seed proof suite across critical baselines."""

    base = _load_config(config_path)
    root = Path(output_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    selected_policies = _ordered_policies(policies or list(PROOF_POLICIES))
    unknown = sorted(set(selected_policies) - set(SUPPORTED_PROOF_POLICIES))
    if unknown:
        raise ValueError(f"Unsupported proof-suite policies: {unknown}")
    if "shared_initial_no_evolve_api" in selected_policies and "true_self_evolving_api" not in selected_policies:
        raise ValueError("shared_initial_no_evolve_api requires true_self_evolving_api in the same proof-suite run.")

    run_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    if int(parallel_workers) <= 1:
        run_rows = _run_policy_grid_sequential(
            base=base,
            root=root,
            seeds=seeds,
            max_rounds=max_rounds,
            selected_policies=selected_policies,
        )
    else:
        run_rows = _run_policy_grid_parallel(
            base=base,
            root=root,
            seeds=seeds,
            max_rounds=max_rounds,
            selected_policies=selected_policies,
            parallel_workers=int(parallel_workers),
            executor_type=executor_type,
        )
    run_rows = _augment_rows_with_oracle_metrics(run_rows, base=base)
    for row in run_rows:
        round_rows.extend({"policy_name": row["policy_name"], "seed": int(row["seed"]), **item} for item in row["round_summaries"])

    result = {
        "status": "ok",
        "config_path": str(config_path),
        "seeds": [int(seed) for seed in seeds],
        "max_rounds": int(max_rounds),
        "parallel_workers": int(parallel_workers),
        "executor_type": str(executor_type),
        "policies": selected_policies,
        "run_rows": run_rows,
        "round_rows": round_rows,
        "aggregate": _aggregate(run_rows),
        "pairwise_vs_true_self_evolving": _pairwise_vs(run_rows, reference_policy="true_self_evolving_api"),
        "pairwise_auc_vs_true_self_evolving": _pairwise_vs(
            run_rows,
            reference_policy="true_self_evolving_api",
            metric="auc_best_observed_yield",
        ),
        "pairwise_vs_care_log_only": _pairwise_vs(
            run_rows,
            reference_policy="true_self_evolving_api_care_log_only",
        )
        if any(row.get("policy_name") == "true_self_evolving_api_care_log_only" for row in run_rows)
        else [],
        "pairwise_auc_vs_care_log_only": _pairwise_vs(
            run_rows,
            reference_policy="true_self_evolving_api_care_log_only",
            metric="auc_best_observed_yield",
        )
        if any(row.get("policy_name") == "true_self_evolving_api_care_log_only" for row in run_rows)
        else [],
        "proof_checks": _proof_checks(run_rows, selected_policies=selected_policies, max_rounds=max_rounds),
        "care_checks": _care_checks(run_rows, selected_policies=selected_policies, max_rounds=max_rounds),
        "output_dir": str(root),
    }
    write_json(root / "proof_summary.json", result)
    write_json(root / "run_rows.json", run_rows)
    write_json(root / "round_rows.json", round_rows)
    print(json.dumps(_compact(result), indent=2, sort_keys=True))
    return result


def _run_policy_grid_sequential(
    *,
    base: SelfEvolvingConfig,
    root: Path,
    seeds: list[int],
    max_rounds: int,
    selected_policies: list[str],
) -> list[dict[str, Any]]:
    run_rows: list[dict[str, Any]] = []
    for seed in seeds:
        shared_initial_skill: SkillArtifact | None = None
        for policy in selected_policies:
            row = _run_one_policy_seed(
                base=base,
                root=root,
                seed=int(seed),
                max_rounds=int(max_rounds),
                policy=policy,
                shared_initial_skill=shared_initial_skill,
            )
            if policy == "true_self_evolving_api":
                shared_initial_skill = _first_activated_skill(root / f"true_self_evolving_api_seed_{int(seed)}")
            run_rows.append(row)
    return run_rows


def _run_policy_grid_parallel(
    *,
    base: SelfEvolvingConfig,
    root: Path,
    seeds: list[int],
    max_rounds: int,
    selected_policies: list[str],
    parallel_workers: int,
    executor_type: str = "thread",
) -> list[dict[str, Any]]:
    non_shared_policies = [policy for policy in selected_policies if policy != "shared_initial_no_evolve_api"]
    row_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    workers = max(1, int(parallel_workers))
    executor_name = str(executor_type).strip().lower()
    if executor_name not in {"thread", "process"}:
        raise ValueError(f"Unsupported executor_type: {executor_type!r}. Use 'thread' or 'process'.")
    executor_cls = ProcessPoolExecutor if executor_name == "process" else ThreadPoolExecutor
    with executor_cls(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_one_policy_seed,
                base=base,
                root=root,
                seed=int(seed),
                max_rounds=int(max_rounds),
                policy=policy,
            ): (int(seed), policy)
            for seed in seeds
            for policy in non_shared_policies
        }
        for future in as_completed(futures):
            seed, policy = futures[future]
            row_by_key[(seed, policy)] = future.result()

        if "shared_initial_no_evolve_api" in selected_policies:
            shared_futures = {}
            for seed in seeds:
                true_dir = root / f"true_self_evolving_api_seed_{int(seed)}"
                shared_initial_skill = _first_activated_skill(true_dir)
                future = executor.submit(
                    _run_one_policy_seed,
                    base=base,
                    root=root,
                    seed=int(seed),
                    max_rounds=int(max_rounds),
                    policy="shared_initial_no_evolve_api",
                    shared_initial_skill=shared_initial_skill,
                )
                shared_futures[future] = (int(seed), "shared_initial_no_evolve_api")
            for future in as_completed(shared_futures):
                seed, policy = shared_futures[future]
                row_by_key[(seed, policy)] = future.result()

    return [
        row_by_key[(int(seed), policy)]
        for seed in seeds
        for policy in selected_policies
        if (int(seed), policy) in row_by_key
    ]


def _run_one_policy_seed(
    *,
    base: SelfEvolvingConfig,
    root: Path,
    seed: int,
    max_rounds: int,
    policy: str,
    shared_initial_skill: SkillArtifact | None = None,
) -> dict[str, Any]:
    tables, table_source = _load_tables(base)
    evaluator = OfflineEvaluator.from_tables(tables)
    config = SelfEvolvingConfig(
        **{
            **base.__dict__,
            "run_id": f"{policy}_seed_{int(seed)}",
            "seed": int(seed),
            "max_rounds": int(max_rounds),
            "output_dir": str(root / f"{policy}_seed_{int(seed)}"),
            "mode": _policy_mode(base=base, policy=policy),
            "resume_policy_state": False,
        }
    )
    try:
        if policy == "true_self_evolving_api":
            row = _run_true_self_evolving(config=config, tables=copy.deepcopy(tables), evaluator=evaluator)
        elif policy in CARE_POLICIES:
            row = _run_true_self_evolving(
                config=_care_config_for_policy(config=config, policy=policy),
                tables=copy.deepcopy(tables),
                evaluator=evaluator,
            )
        elif policy == "llm_only_self_evolving":
            row = _run_llm_self_evolving_rank1_ablation(
                config=config,
                tables=copy.deepcopy(tables),
                evaluator=evaluator,
            )
        elif policy == "public_expert_only_meta_controller":
            row = _run_public_expert_only_meta_controller(
                config=config,
                tables=copy.deepcopy(tables),
                evaluator=evaluator,
            )
        elif policy == "public_expert_only_meta_controller_4experts":
            row = _run_public_expert_only_meta_controller_4experts(
                config=config,
                tables=copy.deepcopy(tables),
                evaluator=evaluator,
            )
        elif policy == "public_expert_only_meta_controller_6experts":
            row = _run_public_expert_only_meta_controller_6experts(
                config=config,
                tables=copy.deepcopy(tables),
                evaluator=evaluator,
            )
        elif policy == "shared_initial_no_evolve_api":
            if shared_initial_skill is None:
                true_dir = root / f"true_self_evolving_api_seed_{int(seed)}"
                shared_initial_skill = _first_activated_skill(true_dir)
            if shared_initial_skill is None:
                raise RuntimeError(
                    "shared_initial_no_evolve_api could not find the matched initial skill from "
                    "true_self_evolving_api."
                )
            row = _run_shared_initial_no_evolve(
                config=config,
                tables=copy.deepcopy(tables),
                evaluator=evaluator,
                initial_skill=shared_initial_skill,
            )
        elif policy in {"fixed_api_tool", "llm_assisted_one_shot_api"}:
            row = _run_fixed_api_tool(config=config, tables=copy.deepcopy(tables), evaluator=evaluator)
        elif policy == "no_evolve_api_reuse":
            row = _run_no_evolve_api_reuse(config=config, tables=copy.deepcopy(tables), evaluator=evaluator)
        elif policy == LMABO_STYLE_POLICY:
            row = _run_lmabo_style_nearest_neighbor_llm_bo(
                config=config,
                tables=copy.deepcopy(tables),
                evaluator=evaluator,
            )
        elif policy == "fixed_public_heuristic":
            row = _run_fixed_source_policy(
                config=config,
                tables=copy.deepcopy(tables),
                evaluator=evaluator,
                policy_name=policy,
                source=FAKE_FULL_POOL_TOOL_SOURCE,
            )
        elif policy == "categorical_empirical_bayes_ucb":
            row = _run_categorical_empirical_bayes_ucb(config=config, tables=copy.deepcopy(tables), evaluator=evaluator)
        elif policy == "stratified_random_public":
            row = _run_stratified_random_public(config=config, tables=copy.deepcopy(tables), evaluator=evaluator)
        elif policy == "bo_like_surrogate":
            row = _run_bo_like_surrogate(config=config, tables=copy.deepcopy(tables), evaluator=evaluator)
        elif policy == "botorch_style_gp_logei":
            row = _run_botorch_style_gp_logei(config=config, tables=copy.deepcopy(tables), evaluator=evaluator)
        elif policy == "smac_style_rf_ei":
            row = _run_smac_style_rf_ei(config=config, tables=copy.deepcopy(tables), evaluator=evaluator)
        elif policy == "classical_bo_gp_ucb":
            row = _run_classical_bo_gp_ucb(config=config, tables=copy.deepcopy(tables), evaluator=evaluator)
        elif policy == "tpe_style_bo":
            row = _run_tpe_style_bo(config=config, tables=copy.deepcopy(tables), evaluator=evaluator)
        elif policy == "chemistry_descriptor_bo":
            row = _run_chemistry_descriptor_bo(config=config, tables=copy.deepcopy(tables), evaluator=evaluator)
        elif policy == "edbo_style_descriptor_gp_ei":
            row = _run_edbo_style_descriptor_gp_ei(config=config, tables=copy.deepcopy(tables), evaluator=evaluator)
        elif policy == "gryffin_style_categorical_bo":
            row = _run_gryffin_style_categorical_bo(config=config, tables=copy.deepcopy(tables), evaluator=evaluator)
        elif policy == "baybe_bofire_style_mixed_bo":
            row = _run_baybe_bofire_style_mixed_bo(config=config, tables=copy.deepcopy(tables), evaluator=evaluator)
        elif policy == "classical_bo_gp_ei":
            row = _run_classical_bo_gp_ei(config=config, tables=copy.deepcopy(tables), evaluator=evaluator)
        else:
            row = _run_random(config=config, tables=copy.deepcopy(tables), evaluator=evaluator)
    except Exception as exc:
        row = _failed_run_summary(config=config, exc=exc, output_dir=config.output_dir)
    row.update({"policy_name": policy, "seed": int(seed), "table_source": table_source})
    return row


def _policy_mode(*, base: SelfEvolvingConfig, policy: str) -> str:
    if policy in API_POLICIES and str(base.mode).lower() == "api":
        return "api"
    return "fake"


def _care_config_for_policy(*, config: SelfEvolvingConfig, policy: str) -> SelfEvolvingConfig:
    overrides: dict[str, Any] = {
        **config.__dict__,
        "care_enabled": True,
        "decision_policy_name": f"{config.decision_policy_name}_{policy}",
    }
    if policy == "true_self_evolving_api_care":
        overrides.update(
            {
                "care_certificate_mode": "calibrated_scout",
                "care_adaptive_planner_enabled": True,
                "care_certificate_margin": 0.0,
            }
        )
    elif policy == "true_self_evolving_api_care_log_only":
        overrides.update(
            {
                "care_certificate_mode": "log_only",
                "care_adaptive_planner_enabled": False,
                "care_certificate_margin": 0.0,
            }
        )
    elif policy == "true_self_evolving_api_care_no_adaptive_planner":
        overrides.update(
            {
                "care_certificate_mode": "calibrated_scout",
                "care_adaptive_planner_enabled": False,
                "care_certificate_margin": 0.0,
            }
        )
    elif policy == "true_self_evolving_api_care_no_certificate":
        overrides.update(
            {
                "care_certificate_mode": "off",
                "care_adaptive_planner_enabled": True,
                "care_certificate_margin": 0.0,
            }
        )
    elif policy == "true_self_evolving_api_care_no_residual_scout":
        overrides.update(
            {
                "care_certificate_mode": "calibrated_scout",
                "care_adaptive_planner_enabled": True,
                "care_certificate_margin": 0.0,
                "llm_residual_scout_budget": 0,
            }
        )
    elif policy == "true_self_evolving_api_care_no_macro_scout":
        overrides.update(
            {
                "care_certificate_mode": "calibrated_scout",
                "care_adaptive_planner_enabled": True,
                "care_certificate_margin": 0.0,
                "llm_macro_frontier_scout_enabled": False,
            }
        )
    else:
        raise ValueError(f"Unsupported CARE policy: {policy}")
    return SelfEvolvingConfig(**overrides)


def _run_true_self_evolving(*, config: SelfEvolvingConfig, tables: Any, evaluator: OfflineEvaluator) -> dict[str, Any]:
    client = _client_for_config(config)
    replay_state = _initial_state(tables=tables, evaluator=evaluator, config=config)
    agent = SelfEvolvingFullPoolAgent(config=config, client=client)
    agent.initialize_run(tables=tables, replay_state=replay_state)
    initial_best = _best(replay_state, config)
    while replay_state.can_continue(config.max_rounds):
        decision = agent.decide(tables=tables, replay_state=replay_state)
        revealed = evaluator.reveal(decision.selected_candidate_ids)
        replay_state.observe(revealed)
        agent.update_after_reveal(tables=tables, replay_state=replay_state, decision=decision, revealed_rows=revealed)
    assert agent.state is not None
    return _summary_from_rounds(
        config=config,
        initial_best=initial_best,
        round_summaries=agent.state.round_summaries,
        strategy_state=agent.state.strategy_state,
        output_dir=config.output_dir,
        extra=_skill_hash_summary(Path(config.output_dir), active_only=True),
    )


class _Rank1SelfEvolvingAgent(SelfEvolvingFullPoolAgent):
    """Proof-suite ablation: keep LLM evolution, remove final public meta arbitration."""

    def _meta_controller_select_candidate(
        self,
        *,
        parsed: Any,
        observed_df: pd.DataFrame,
        candidate_df: pd.DataFrame,
        round_index: int,
        portfolio_selection: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ranked = list(getattr(parsed, "ranked_candidates", []))
        selected_id = str(parsed.selected_display_candidate_id)
        selected_row = next((row for row in ranked if str(row.get("candidate_id")) == selected_id), {})
        return {
            "selection_version": "proof_suite_llm_rank1_no_public_meta",
            "round_index": int(round_index),
            "selected_display_candidate_id": selected_id,
            "selected_by": "active_llm_skill_rank1",
            "selected_meta_score": _safe_float(selected_row.get("score")) or 0.0,
            "selected_rank": int(selected_row.get("rank", 1) or 1),
            "supporting_experts": ["active_llm_skill"],
            "expert_reports": {},
            "top_meta_candidates": [
                {
                    "candidate_id": str(row.get("candidate_id")),
                    "meta_score": _safe_float(row.get("score")) or 0.0,
                    "supporting_experts": ["active_llm_skill"],
                }
                for row in ranked[:12]
            ],
            "ablation_note": "LLM self-evolving skills choose their own rank-1 candidate; public expert meta-controller disabled.",
        }


def _run_llm_self_evolving_rank1_ablation(
    *,
    config: SelfEvolvingConfig,
    tables: Any,
    evaluator: OfflineEvaluator,
) -> dict[str, Any]:
    client = _client_for_config(config)
    replay_state = _initial_state(tables=tables, evaluator=evaluator, config=config)
    agent = _Rank1SelfEvolvingAgent(config=config, client=client)
    agent.initialize_run(tables=tables, replay_state=replay_state)
    initial_best = _best(replay_state, config)
    while replay_state.can_continue(config.max_rounds):
        decision = agent.decide(tables=tables, replay_state=replay_state)
        revealed = evaluator.reveal(decision.selected_candidate_ids)
        replay_state.observe(revealed)
        agent.update_after_reveal(tables=tables, replay_state=replay_state, decision=decision, revealed_rows=revealed)
    assert agent.state is not None
    row = _summary_from_rounds(
        config=config,
        initial_best=initial_best,
        round_summaries=agent.state.round_summaries,
        strategy_state=agent.state.strategy_state,
        output_dir=config.output_dir,
        extra=_skill_hash_summary(Path(config.output_dir), active_only=True),
    )
    row["ablation_note"] = (
        "LLM planner, skill synthesis, gate, reward, portfolio, and rollback are retained; "
        "the final public expert meta-controller is disabled and the active LLM skill rank-1 is selected."
    )
    row["ablation_family"] = "llm_only_self_evolving"
    return row


def _run_public_expert_only_meta_controller(
    *,
    config: SelfEvolvingConfig,
    tables: Any,
    evaluator: OfflineEvaluator,
) -> dict[str, Any]:
    replay_state = _initial_state(tables=tables, evaluator=evaluator, config=config)
    initial_best = _best(replay_state, config)
    round_summaries: list[dict[str, Any]] = []
    strategy_state: dict[str, Any] = {"selected_count": 0, "fallback_count": 0, "expert_meta_selection_count": 0}
    sparse_category_route = None
    while replay_state.can_continue(config.max_rounds):
        round_index = int(replay_state.round_index) + 1
        observed_df = build_observed_df_from_revealed_state(tables, replay_state, objective_name=config.target_column)
        candidate_df = build_full_remaining_candidate_df(tables, replay_state)
        data_profile = _public_data_profile(observed_df=observed_df, candidate_df=candidate_df)
        if bool(data_profile.get("high_cardinality_categorical_space", False)) and not sparse_category_route:
            sparse_category_route = _choose_sparse_category_route(data_profile)
        experts = _build_public_expert_rankings(
            parsed=None,
            observed_df=observed_df,
            candidate_df=candidate_df,
            seed=int(config.seed) + int(round_index),
            include_categorical=bool(config.adaptive_categorical_experts),
        )
        report = _score_expert_candidates(
            experts=experts,
            observed_df=observed_df,
            candidate_df=candidate_df,
            round_index=round_index,
            sparse_category_route=sparse_category_route,
        )
        display_id = str(report["selected_display_candidate_id"])
        internal_id = map_display_candidate_to_internal_id(candidate_df, display_id)
        revealed = evaluator.reveal([internal_id])
        replay_state.observe(revealed)
        strategy_state["expert_meta_selection_count"] = int(strategy_state.get("expert_meta_selection_count", 0)) + 1
        strategy_state["selected_count"] = int(strategy_state.get("selected_count", 0)) + 1
        round_summaries.append(
            {
                **_round_summary(
                    config=config,
                    replay_state=replay_state,
                    round_index=round_index,
                    display_id=display_id,
                    selected_y=_safe_float(revealed[config.target_column].iloc[0]),
                    candidate_df=candidate_df,
                    fallback_used=False,
                ),
                "expert_meta_selection": report,
            }
        )
    return _summary_from_rounds(
        config=config,
        initial_best=initial_best,
        round_summaries=round_summaries,
        strategy_state=strategy_state,
        output_dir=config.output_dir,
        extra={
            "unique_skill_hash_count": 0,
            "active_unique_skill_hash_count": 0,
            "ablation_note": (
                "Public expert meta-controller only: fixed public heuristic, GP/RF/categorical public experts, "
                "and v5 incumbent/challenger arbitration; no LLM planner or skill synthesis."
            ),
        },
    )


def _run_public_expert_only_meta_controller_6experts(
    *,
    config: SelfEvolvingConfig,
    tables: Any,
    evaluator: OfflineEvaluator,
) -> dict[str, Any]:
    replay_state = _initial_state(tables=tables, evaluator=evaluator, config=config)
    initial_best = _best(replay_state, config)
    round_summaries: list[dict[str, Any]] = []
    strategy_state: dict[str, Any] = {
        "selected_count": 0,
        "fallback_count": 0,
        "expert_meta_selection_count": 0,
    }
    sparse_category_route = None
    while replay_state.can_continue(config.max_rounds):
        round_index = int(replay_state.round_index) + 1
        observed_df = build_observed_df_from_revealed_state(tables, replay_state, objective_name=config.target_column)
        candidate_df = build_full_remaining_candidate_df(tables, replay_state)
        data_profile = _public_data_profile(observed_df=observed_df, candidate_df=candidate_df)
        if bool(data_profile.get("high_cardinality_categorical_space", False)) and not sparse_category_route:
            sparse_category_route = _choose_sparse_category_route(data_profile)
        experts = _capacity_public_expert_rankings(
            observed_df=observed_df,
            candidate_df=candidate_df,
            objective_direction=config.objective_direction,
            seed=int(config.seed) + int(round_index),
            top_k=40,
            include_extra_bo_experts=True,
        )
        report = _score_expert_candidates(
            experts=experts,
            observed_df=observed_df,
            candidate_df=candidate_df,
            round_index=round_index,
            sparse_category_route=sparse_category_route,
        )
        display_id = str(report["selected_display_candidate_id"])
        internal_id = map_display_candidate_to_internal_id(candidate_df, display_id)
        revealed = evaluator.reveal([internal_id])
        replay_state.observe(revealed)
        strategy_state["expert_meta_selection_count"] = int(strategy_state.get("expert_meta_selection_count", 0)) + 1
        strategy_state["selected_count"] = int(strategy_state.get("selected_count", 0)) + 1
        round_summaries.append(
            {
                **_round_summary(
                    config=config,
                    replay_state=replay_state,
                    round_index=round_index,
                    display_id=display_id,
                    selected_y=_safe_float(revealed[config.target_column].iloc[0]),
                    candidate_df=candidate_df,
                    fallback_used=False,
                ),
                "expert_meta_selection": report,
                "public_expert_count": int(len(report.get("expert_reports", {}))),
            }
        )
    return _summary_from_rounds(
        config=config,
        initial_best=initial_best,
        round_summaries=round_summaries,
        strategy_state=strategy_state,
        output_dir=config.output_dir,
        extra={
            "unique_skill_hash_count": 0,
            "active_unique_skill_hash_count": 0,
            "ablation_note": (
                "Expanded public-only capacity control: the public meta-controller receives the original public "
                "expert set plus EDBO-style descriptor GP-EI and BayBE/BoFire-style mixed-space experts. "
                "No LLM planner, skill synthesis, residual scout, or certificate is used."
            ),
            "public_expert_family": "expanded_public_only_meta_controller",
        },
    )


def _run_public_expert_only_meta_controller_4experts(
    *,
    config: SelfEvolvingConfig,
    tables: Any,
    evaluator: OfflineEvaluator,
) -> dict[str, Any]:
    replay_state = _initial_state(tables=tables, evaluator=evaluator, config=config)
    initial_best = _best(replay_state, config)
    round_summaries: list[dict[str, Any]] = []
    strategy_state: dict[str, Any] = {
        "selected_count": 0,
        "fallback_count": 0,
        "expert_meta_selection_count": 0,
    }
    sparse_category_route = None
    while replay_state.can_continue(config.max_rounds):
        round_index = int(replay_state.round_index) + 1
        observed_df = build_observed_df_from_revealed_state(tables, replay_state, objective_name=config.target_column)
        candidate_df = build_full_remaining_candidate_df(tables, replay_state)
        data_profile = _public_data_profile(observed_df=observed_df, candidate_df=candidate_df)
        if bool(data_profile.get("high_cardinality_categorical_space", False)) and not sparse_category_route:
            sparse_category_route = _choose_sparse_category_route(data_profile)
        experts = _capacity_public_expert_rankings(
            observed_df=observed_df,
            candidate_df=candidate_df,
            objective_direction=config.objective_direction,
            seed=int(config.seed) + int(round_index),
            top_k=40,
            include_extra_bo_experts=False,
        )
        report = _score_expert_candidates(
            experts=experts,
            observed_df=observed_df,
            candidate_df=candidate_df,
            round_index=round_index,
            sparse_category_route=sparse_category_route,
        )
        display_id = str(report["selected_display_candidate_id"])
        internal_id = map_display_candidate_to_internal_id(candidate_df, display_id)
        revealed = evaluator.reveal([internal_id])
        replay_state.observe(revealed)
        strategy_state["expert_meta_selection_count"] = int(strategy_state.get("expert_meta_selection_count", 0)) + 1
        strategy_state["selected_count"] = int(strategy_state.get("selected_count", 0)) + 1
        round_summaries.append(
            {
                **_round_summary(
                    config=config,
                    replay_state=replay_state,
                    round_index=round_index,
                    display_id=display_id,
                    selected_y=_safe_float(revealed[config.target_column].iloc[0]),
                    candidate_df=candidate_df,
                    fallback_used=False,
                ),
                "expert_meta_selection": report,
                "public_expert_count": int(len(report.get("expert_reports", {}))),
            }
        )
    return _summary_from_rounds(
        config=config,
        initial_best=initial_best,
        round_summaries=round_summaries,
        strategy_state=strategy_state,
        output_dir=config.output_dir,
        extra={
            "unique_skill_hash_count": 0,
            "active_unique_skill_hash_count": 0,
            "ablation_note": (
                "Public expert meta-controller with a fixed 4-expert pool: fixed heuristic, GP, RF, and "
                "categorical shrinkage. No LLM planner, skill synthesis, residual scout, or certificate is used."
            ),
            "public_expert_family": "four_expert_public_only_meta_controller",
        },
    )


def _capacity_public_expert_rankings(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    objective_direction: str,
    seed: int,
    top_k: int = 40,
    include_extra_bo_experts: bool,
) -> list[dict[str, Any]]:
    experts: list[dict[str, Any]] = []
    fixed_rows = _run_source_ranker_rows(
        source=_fixed_public_source(),
        observed_df=observed_df,
        candidate_df=candidate_df,
        name="fixed_public_heuristic",
    )
    if fixed_rows:
        experts.append(_expert_from_ranked_rows("fixed_public_heuristic", fixed_rows, max_rows=top_k))
    gp_rows = _public_gp_ei_rank(
        observed_df=observed_df,
        candidate_df=candidate_df,
        objective_direction="maximize",
        seed=seed,
        top_k=top_k,
    )
    if gp_rows:
        experts.append({"name": "classical_gp_ei", "ranked": gp_rows})
    rf_rows = _public_rf_ucb_rank(observed_df=observed_df, candidate_df=candidate_df, seed=seed, top_k=top_k)
    if rf_rows:
        experts.append({"name": "rf_ucb_surrogate", "ranked": rf_rows})
    categorical_rows = _public_categorical_shrinkage_rank(
        observed_df=observed_df,
        candidate_df=candidate_df,
        top_k=top_k,
    )
    if categorical_rows:
        experts.append({"name": "categorical_shrinkage", "ranked": categorical_rows})
    if include_extra_bo_experts:
        edbo_scores = _edbo_style_descriptor_gp_ei_scores(
            observed_df=observed_df,
            candidate_df=candidate_df,
            objective_direction=objective_direction,
            seed=seed + 101,
        )
        edbo_rows = _rank_rows_from_scores(candidate_df=candidate_df, scores=edbo_scores, top_k=top_k)
        if edbo_rows:
            experts.append({"name": "edbo_style_descriptor_gp_ei", "ranked": edbo_rows})
        baybe_scores = _baybe_bofire_style_mixed_scores(
            observed_df=observed_df,
            candidate_df=candidate_df,
            objective_direction=objective_direction,
            seed=seed + 211,
        )
        baybe_rows = _rank_rows_from_scores(candidate_df=candidate_df, scores=baybe_scores, top_k=top_k)
        if baybe_rows:
            experts.append({"name": "baybe_bofire_style_mixed_bo", "ranked": baybe_rows})
    return experts


def _expanded_public_experts_for_capacity_ablation(
    *,
    base_experts: list[dict[str, Any]],
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    objective_direction: str,
    seed: int,
    top_k: int = 40,
) -> list[dict[str, Any]]:
    """Add two strong public BO experts to test whether more public experts replace LLMs."""

    return list(base_experts)


def _run_fixed_api_tool(*, config: SelfEvolvingConfig, tables: Any, evaluator: OfflineEvaluator) -> dict[str, Any]:
    client = _client_for_config(config)
    replay_state = _initial_state(tables=tables, evaluator=evaluator, config=config)
    agent = SelfEvolvingFullPoolAgent(config=config, client=client)
    agent.initialize_run(tables=tables, replay_state=replay_state)
    decision = agent.decide(tables=tables, replay_state=replay_state)
    skill = agent.registry.active_skill()
    if skill is None:
        _write_failed_initial_api_summary(config=config, agent=agent, decision=decision)
        raise RuntimeError("fixed_api_tool could not synthesize a deployable initial skill.")
    actual_initial_counts = dict(agent.state.strategy_state if agent.state is not None else {})
    synthesis_mode = "api" if str(config.mode).lower() == "api" else "fake"
    return _run_fixed_source_policy(
        config=SelfEvolvingConfig(**{**config.__dict__, "mode": "fake"}),
        tables=tables,
        evaluator=evaluator,
        policy_name="fixed_api_tool",
        source=skill.source,
        initial_strategy_counts={
            "planner_call_count": int(actual_initial_counts.get("planner_call_count", 0) or 0),
            "skill_synthesis_count": int(actual_initial_counts.get("skill_synthesis_count", 0) or 0),
            "gate_pass_count": int(actual_initial_counts.get("gate_pass_count", 0) or 0),
            "gate_reject_count": int(actual_initial_counts.get("gate_reject_count", 0) or 0),
            "fallback_count": int(actual_initial_counts.get("fallback_count", 0) or 0),
        },
        output_dir=config.output_dir,
        extra={
            "synthesis_mode": synthesis_mode,
            "ablation_note": (
                "one tool synthesized once, then reused for all rounds; synthesis_mode records whether "
                "the initial tool came from the external API or the fake client"
            ),
        },
    )


def _run_no_evolve_api_reuse(*, config: SelfEvolvingConfig, tables: Any, evaluator: OfflineEvaluator) -> dict[str, Any]:
    # Same operational baseline as fixed_api_tool, with a name that states the
    # ablation: synthesize once, then reuse without patches or reward edits.
    row = _run_fixed_api_tool(config=config, tables=tables, evaluator=evaluator)
    row["ablation_note"] = "one synthesized tool reused for all rounds; no reward-conditioned patching"
    return row


def _run_shared_initial_no_evolve(
    *,
    config: SelfEvolvingConfig,
    tables: Any,
    evaluator: OfflineEvaluator,
    initial_skill: SkillArtifact,
) -> dict[str, Any]:
    """Replay a matched frozen-tool ablation using true_self_evolving's initial skill.

    This isolates the effect of reward-conditioned patching from stochastic
    variation in the first LLM-generated tool.
    """

    row = _run_fixed_source_policy(
        config=SelfEvolvingConfig(**{**config.__dict__, "mode": "fake"}),
        tables=tables,
        evaluator=evaluator,
        policy_name="shared_initial_no_evolve_api",
        source=initial_skill.source,
        output_dir=config.output_dir,
    )
    row.update(
        {
            "ablation_note": (
                "matched frozen ablation: reuse the initial active skill generated by "
                "true_self_evolving_api for the same seed; no planner, reward, or patch calls"
            ),
            "shared_initial_skill_id": initial_skill.skill_id,
            "shared_initial_skill_version": int(initial_skill.version),
            "shared_initial_skill_hash": initial_skill.source_hash,
        }
    )
    return row


def _run_fixed_source_policy(
    *,
    config: SelfEvolvingConfig,
    tables: Any,
    evaluator: OfflineEvaluator,
    policy_name: str,
    source: str,
    initial_strategy_counts: dict[str, int] | None = None,
    output_dir: str | None = None,
    run_gate: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    replay_state = _initial_state(tables=tables, evaluator=evaluator, config=config)
    initial_best = _best(replay_state, config)
    round_summaries: list[dict[str, Any]] = []
    tool_state: dict[str, Any] = {}
    gate_report = None
    while replay_state.can_continue(config.max_rounds):
        round_index = int(replay_state.round_index) + 1
        observed_df = build_observed_df_from_revealed_state(tables, replay_state, objective_name=config.target_column)
        candidate_df = build_full_remaining_candidate_df(tables, replay_state)
        if run_gate and gate_report is None:
            gate_report = run_conservative_gate(
                candidate_skill=SkillArtifact(
                    skill_id=policy_name,
                    version=1,
                    family="ranker",
                    source=source,
                    created_round=1,
                ),
                observed_df=observed_df,
                candidate_df=candidate_df,
                round_index=round_index,
            )
            if not gate_report.deployable:
                raise RuntimeError(f"{policy_name} source failed gate: {gate_report.reason}")
        raw_output = run_rank_candidates_tool(
            tool_source=source,
            observed_df=observed_df,
            candidate_df=candidate_df,
            memory="",
            tool_state=tool_state,
        )
        parsed = parse_ranked_candidates(raw_output, candidate_df=candidate_df, observed_df=observed_df)
        display_id = parsed.selected_display_candidate_id
        internal_id = map_display_candidate_to_internal_id(candidate_df, display_id)
        revealed = evaluator.reveal([internal_id])
        replay_state.observe(revealed)
        tool_state = dict(parsed.tool_state)
        round_summaries.append(
            _round_summary(
                config=config,
                replay_state=replay_state,
                round_index=round_index,
                display_id=display_id,
                selected_y=_safe_float(revealed[config.target_column].iloc[0]),
                candidate_df=candidate_df,
                fallback_used=False,
            )
        )
    counts = {
        "selected_count": len(round_summaries),
        "fallback_count": 0,
        "planner_call_count": 0,
        "skill_synthesis_count": 0,
        "gate_pass_count": 1 if round_summaries else 0,
        "gate_reject_count": 0,
        "reward_count": 0,
    }
    counts.update(initial_strategy_counts or {})
    summary_extra = {
        "unique_skill_hash_count": 1,
        "active_unique_skill_hash_count": 1,
        "skill_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }
    summary_extra.update(extra or {})
    return _summary_from_rounds(
        config=config,
        initial_best=initial_best,
        round_summaries=round_summaries,
        strategy_state=counts,
        output_dir=output_dir or config.output_dir,
        extra=summary_extra,
    )


def _run_bo_like_surrogate(*, config: SelfEvolvingConfig, tables: Any, evaluator: OfflineEvaluator) -> dict[str, Any]:
    replay_state = _initial_state(tables=tables, evaluator=evaluator, config=config)
    initial_best = _best(replay_state, config)
    round_summaries: list[dict[str, Any]] = []
    while replay_state.can_continue(config.max_rounds):
        round_index = int(replay_state.round_index) + 1
        observed_df = build_observed_df_from_revealed_state(tables, replay_state, objective_name=config.target_column)
        candidate_df = build_full_remaining_candidate_df(tables, replay_state)
        display_id = _bo_like_select(observed_df=observed_df, candidate_df=candidate_df, seed=config.seed + round_index)
        internal_id = map_display_candidate_to_internal_id(candidate_df, display_id)
        revealed = evaluator.reveal([internal_id])
        replay_state.observe(revealed)
        round_summaries.append(
            _round_summary(
                config=config,
                replay_state=replay_state,
                round_index=round_index,
                display_id=display_id,
                selected_y=_safe_float(revealed[config.target_column].iloc[0]),
                candidate_df=candidate_df,
                fallback_used=False,
            )
        )
    return _summary_from_rounds(
        config=config,
        initial_best=initial_best,
        round_summaries=round_summaries,
        strategy_state={"selected_count": len(round_summaries), "fallback_count": 0},
        output_dir=config.output_dir,
        extra={"unique_skill_hash_count": 0, "active_unique_skill_hash_count": 0},
    )


def _run_botorch_style_gp_logei(*, config: SelfEvolvingConfig, tables: Any, evaluator: OfflineEvaluator) -> dict[str, Any]:
    return _run_selector_policy(
        config=config,
        tables=tables,
        evaluator=evaluator,
        selector=lambda observed_df, candidate_df, seed: _gp_logei_select(
            observed_df=observed_df,
            candidate_df=candidate_df,
            objective_direction=config.objective_direction,
            seed=seed,
        ),
        baseline_note=(
            "BoTorch-style finite-pool GP LogEI baseline using public features only; "
            "kept dependency-light for proof-suite reproducibility."
        ),
    )


def _run_smac_style_rf_ei(*, config: SelfEvolvingConfig, tables: Any, evaluator: OfflineEvaluator) -> dict[str, Any]:
    return _run_selector_policy(
        config=config,
        tables=tables,
        evaluator=evaluator,
        selector=lambda observed_df, candidate_df, seed: _smac_style_rf_ei_select(
            observed_df=observed_df,
            candidate_df=candidate_df,
            objective_direction=config.objective_direction,
            seed=seed,
        ),
        baseline_note=(
            "SMAC3-style finite-pool RF/ExtraTrees empirical-EI baseline for mixed categorical spaces; "
            "public features and observed rewards only."
        ),
    )


def _run_classical_bo_gp_ucb(*, config: SelfEvolvingConfig, tables: Any, evaluator: OfflineEvaluator) -> dict[str, Any]:
    return _run_selector_policy(
        config=config,
        tables=tables,
        evaluator=evaluator,
        selector=lambda observed_df, candidate_df, seed: _gp_ucb_select(
            observed_df=observed_df,
            candidate_df=candidate_df,
            objective_direction=config.objective_direction,
            seed=seed,
        ),
        baseline_note="GaussianProcessRegressor + UCB acquisition over public finite-pool features.",
    )


def _run_tpe_style_bo(*, config: SelfEvolvingConfig, tables: Any, evaluator: OfflineEvaluator) -> dict[str, Any]:
    return _run_selector_policy(
        config=config,
        tables=tables,
        evaluator=evaluator,
        selector=lambda observed_df, candidate_df, seed: _tpe_style_select(
            observed_df=observed_df,
            candidate_df=candidate_df,
            objective_direction=config.objective_direction,
            seed=seed,
        ),
        baseline_note="TPE-style good-vs-bad density-ratio acquisition using public feature hashing only.",
    )


def _run_lmabo_style_nearest_neighbor_llm_bo(
    *,
    config: SelfEvolvingConfig,
    tables: Any,
    evaluator: OfflineEvaluator,
) -> dict[str, Any]:
    client = _client_for_config(config)
    if client is None:
        raise RuntimeError(f"{LMABO_STYLE_POLICY} requires config.mode='api'.")
    replay_state = _initial_state(tables=tables, evaluator=evaluator, config=config)
    initial_best = _best(replay_state, config)
    round_summaries: list[dict[str, Any]] = []
    acquisition_history: list[dict[str, Any]] = []
    acquisition_counts = {name: 0 for name in LMABO_ACQUISITIONS}
    strategy_state: dict[str, Any] = {
        "selected_count": 0,
        "fallback_count": 0,
        "planner_call_count": 0,
        "lmabo_llm_call_count": 0,
    }
    while replay_state.can_continue(config.max_rounds):
        round_index = int(replay_state.round_index) + 1
        before_best = _best(replay_state, config)
        observed_df = build_observed_df_from_revealed_state(tables, replay_state, objective_name=config.target_column)
        candidate_df = build_full_remaining_candidate_df(tables, replay_state)
        diagnostics = _lmabo_state_diagnostics(
            observed_df=observed_df,
            candidate_df=candidate_df,
            config=config,
            round_index=round_index,
        )
        prompt = _lmabo_follow_up_prompt(
            config=config,
            round_index=round_index,
            diagnostics=diagnostics,
            acquisition_history=acquisition_history,
        )
        raw_response = ""
        fallback_used = False
        fallback_reason = ""
        try:
            strategy_state["planner_call_count"] = int(strategy_state.get("planner_call_count", 0) or 0) + 1
            strategy_state["lmabo_llm_call_count"] = int(strategy_state.get("lmabo_llm_call_count", 0) or 0) + 1
            raw_response = client.create_tool(
                messages=[
                    {"role": "developer", "content": _lmabo_system_prompt(config)},
                    {"role": "user", "content": prompt},
                ],
                json_schema=_lmabo_choice_json_schema(),
                schema_name="lmabo_acquisition_choice",
                schema_description="Choose one LMABO acquisition function and justify it briefly.",
            )
            parsed = _parse_lmabo_acquisition_response(raw_response)
            acquisition = _normalize_lmabo_acquisition(parsed.get("acquisition"))
            justification = str(parsed.get("justification", "")).strip()[:800]
        except Exception as exc:
            fallback_used = True
            strategy_state["fallback_count"] = int(strategy_state.get("fallback_count", 0) or 0) + 1
            acquisition = LMABO_DEFAULT_ACQUISITION
            justification = "Defaulted to UCB after an invalid or unavailable LMABO acquisition response."
            fallback_reason = f"{exc.__class__.__name__}: {str(exc)[:500]}"
        acquisition_counts[acquisition] = int(acquisition_counts.get(acquisition, 0)) + 1
        try:
            scores = _lmabo_candidate_scores(
                acquisition=acquisition,
                observed_df=observed_df,
                candidate_df=candidate_df,
                objective_direction=config.objective_direction,
                seed=int(config.seed) + int(round_index),
            )
            display_id = _top_scoring_candidate(candidate_df, scores)
        except Exception as exc:
            fallback_used = True
            if not fallback_reason:
                strategy_state["fallback_count"] = int(strategy_state.get("fallback_count", 0) or 0) + 1
            fallback_reason = fallback_reason or f"{exc.__class__.__name__}: {str(exc)[:500]}"
            fallback = full_pool_random_fallback(
                candidate_df,
                seed=config.seed,
                round_index=round_index,
                reason=f"{LMABO_STYLE_POLICY}_selector_failure",
            )
            display_id = str(fallback["candidate_id"])
        internal_id = map_display_candidate_to_internal_id(candidate_df, display_id)
        revealed = evaluator.reveal([internal_id])
        selected_y = _safe_float(revealed[config.target_column].iloc[0])
        replay_state.observe(revealed)
        after_best = _best(replay_state, config)
        improved_best = _objective_improved(before_best, after_best, config.objective_direction)
        history_item = {
            "round_index": int(round_index),
            "acquisition": acquisition,
            "selected_display_candidate_id": str(display_id),
            "selected_yield": selected_y,
            "best_before": before_best,
            "best_after": after_best,
            "improved_best": bool(improved_best),
        }
        acquisition_history.append(history_item)
        strategy_state["selected_count"] = int(strategy_state.get("selected_count", 0) or 0) + 1
        round_summaries.append(
            {
                **_round_summary(
                    config=config,
                    replay_state=replay_state,
                    round_index=round_index,
                    display_id=display_id,
                    selected_y=selected_y,
                    candidate_df=candidate_df,
                    fallback_used=fallback_used,
                ),
                "lmabo_acquisition": acquisition,
                "lmabo_justification": justification,
                "lmabo_fallback_used": bool(fallback_used),
                "lmabo_fallback_reason": fallback_reason,
                "lmabo_state_diagnostics": diagnostics,
                "lmabo_raw_response": raw_response[:1200],
                "lmabo_reference": LMABO_OFFICIAL_REFERENCE,
                "lmabo_finite_pool_projection": (
                    "Official LMABO selects an acquisition function, then optimizes it in continuous space. "
                    "This replay adapts that step by scoring the unrevealed finite pool with public-feature "
                    "surrogate and nearest-neighbor terms, then selecting the top scored candidate."
                ),
            }
        )
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_json(
        output_path / "lmabo_trace.json",
        {
            "policy": LMABO_STYLE_POLICY,
            "official_reference": LMABO_OFFICIAL_REFERENCE,
            "acquisition_counts": acquisition_counts,
            "acquisition_history": acquisition_history,
            "round_summaries": round_summaries,
        },
    )
    return _summary_from_rounds(
        config=config,
        initial_best=initial_best,
        round_summaries=round_summaries,
        strategy_state=strategy_state,
        output_dir=config.output_dir,
        extra={
            "unique_skill_hash_count": 0,
            "active_unique_skill_hash_count": 0,
            "lmabo_official_reference": LMABO_OFFICIAL_REFERENCE,
            "lmabo_acquisition_counts": acquisition_counts,
            "lmabo_llm_call_count": int(strategy_state.get("lmabo_llm_call_count", 0) or 0),
            "baseline_note": (
                "LMABO-style nearest-neighbor LLM-BO baseline adapted from giang-n-ngo/lmabo: "
                "the LLM receives the structured BO state and selects one acquisition function from "
                "the LMABO portfolio; the selected acquisition is evaluated over the unrevealed finite "
                "candidate pool using only public features, revealed outcomes, and nearest-observed "
                "public-feature terms."
            ),
        },
    )


def _run_chemistry_descriptor_bo(*, config: SelfEvolvingConfig, tables: Any, evaluator: OfflineEvaluator) -> dict[str, Any]:
    return _run_selector_policy(
        config=config,
        tables=tables,
        evaluator=evaluator,
        selector=lambda observed_df, candidate_df, seed: _chemistry_descriptor_bo_select(
            observed_df=observed_df,
            candidate_df=candidate_df,
            objective_direction=config.objective_direction,
            seed=seed,
        ),
        baseline_note=(
            "Gryffin/EDBO-style descriptor BO baseline using public string descriptors, "
            "categorical shrinkage, and empirical ensemble EI."
        ),
    )


def _run_edbo_style_descriptor_gp_ei(*, config: SelfEvolvingConfig, tables: Any, evaluator: OfflineEvaluator) -> dict[str, Any]:
    return _run_selector_policy(
        config=config,
        tables=tables,
        evaluator=evaluator,
        selector=lambda observed_df, candidate_df, seed: _edbo_style_descriptor_gp_ei_select(
            observed_df=observed_df,
            candidate_df=candidate_df,
            objective_direction=config.objective_direction,
            seed=seed,
        ),
        baseline_note=(
            "EDBO-style descriptor GP-EI baseline: public descriptor hashing, standardized GP surrogate, "
            "and expected improvement over the finite candidate pool."
        ),
    )


def _run_gryffin_style_categorical_bo(*, config: SelfEvolvingConfig, tables: Any, evaluator: OfflineEvaluator) -> dict[str, Any]:
    return _run_selector_policy(
        config=config,
        tables=tables,
        evaluator=evaluator,
        selector=lambda observed_df, candidate_df, seed: _gryffin_style_categorical_select(
            observed_df=observed_df,
            candidate_df=candidate_df,
            objective_direction=config.objective_direction,
            seed=seed,
        ),
        baseline_note=(
            "Gryffin-style categorical BO baseline: public categorical kernel/shrinkage scores, "
            "descriptor diversity, and surrogate uncertainty."
        ),
    )


def _run_baybe_bofire_style_mixed_bo(*, config: SelfEvolvingConfig, tables: Any, evaluator: OfflineEvaluator) -> dict[str, Any]:
    return _run_selector_policy(
        config=config,
        tables=tables,
        evaluator=evaluator,
        selector=lambda observed_df, candidate_df, seed: _baybe_bofire_style_mixed_select(
            observed_df=observed_df,
            candidate_df=candidate_df,
            objective_direction=config.objective_direction,
            seed=seed,
        ),
        baseline_note=(
            "BayBE/BoFire-style mixed-space experimental-planning baseline with public feature hashing, "
            "ExtraTrees ensemble improvement, uncertainty, diversity, and constraint-free finite-pool selection."
        ),
    )


def _run_classical_bo_gp_ei(*, config: SelfEvolvingConfig, tables: Any, evaluator: OfflineEvaluator) -> dict[str, Any]:
    replay_state = _initial_state(tables=tables, evaluator=evaluator, config=config)
    initial_best = _best(replay_state, config)
    round_summaries: list[dict[str, Any]] = []
    while replay_state.can_continue(config.max_rounds):
        round_index = int(replay_state.round_index) + 1
        observed_df = build_observed_df_from_revealed_state(tables, replay_state, objective_name=config.target_column)
        candidate_df = build_full_remaining_candidate_df(tables, replay_state)
        display_id = _gp_ei_select(
            observed_df=observed_df,
            candidate_df=candidate_df,
            objective_direction=config.objective_direction,
            seed=config.seed + round_index,
        )
        internal_id = map_display_candidate_to_internal_id(candidate_df, display_id)
        revealed = evaluator.reveal([internal_id])
        replay_state.observe(revealed)
        round_summaries.append(
            _round_summary(
                config=config,
                replay_state=replay_state,
                round_index=round_index,
                display_id=display_id,
                selected_y=_safe_float(revealed[config.target_column].iloc[0]),
                candidate_df=candidate_df,
                fallback_used=False,
            )
        )
    return _summary_from_rounds(
        config=config,
        initial_best=initial_best,
        round_summaries=round_summaries,
        strategy_state={"selected_count": len(round_summaries), "fallback_count": 0},
        output_dir=config.output_dir,
        extra={
            "unique_skill_hash_count": 0,
            "active_unique_skill_hash_count": 0,
            "baseline_note": "GaussianProcessRegressor + expected improvement",
        },
    )


def _run_categorical_empirical_bayes_ucb(
    *,
    config: SelfEvolvingConfig,
    tables: Any,
    evaluator: OfflineEvaluator,
) -> dict[str, Any]:
    replay_state = _initial_state(tables=tables, evaluator=evaluator, config=config)
    initial_best = _best(replay_state, config)
    round_summaries: list[dict[str, Any]] = []
    while replay_state.can_continue(config.max_rounds):
        round_index = int(replay_state.round_index) + 1
        observed_df = build_observed_df_from_revealed_state(tables, replay_state, objective_name=config.target_column)
        candidate_df = build_full_remaining_candidate_df(tables, replay_state)
        display_id = _categorical_eb_select(observed_df=observed_df, candidate_df=candidate_df, seed=config.seed + round_index)
        internal_id = map_display_candidate_to_internal_id(candidate_df, display_id)
        revealed = evaluator.reveal([internal_id])
        replay_state.observe(revealed)
        round_summaries.append(
            _round_summary(
                config=config,
                replay_state=replay_state,
                round_index=round_index,
                display_id=display_id,
                selected_y=_safe_float(revealed[config.target_column].iloc[0]),
                candidate_df=candidate_df,
                fallback_used=False,
            )
        )
    return _summary_from_rounds(
        config=config,
        initial_best=initial_best,
        round_summaries=round_summaries,
        strategy_state={"selected_count": len(round_summaries), "fallback_count": 0},
        output_dir=config.output_dir,
        extra={
            "unique_skill_hash_count": 0,
            "active_unique_skill_hash_count": 0,
            "baseline_note": "public categorical empirical-Bayes shrinkage with small UCB/novelty bonus",
        },
    )


def _run_selector_policy(
    *,
    config: SelfEvolvingConfig,
    tables: Any,
    evaluator: OfflineEvaluator,
    selector: Any,
    baseline_note: str,
) -> dict[str, Any]:
    replay_state = _initial_state(tables=tables, evaluator=evaluator, config=config)
    initial_best = _best(replay_state, config)
    round_summaries: list[dict[str, Any]] = []
    while replay_state.can_continue(config.max_rounds):
        round_index = int(replay_state.round_index) + 1
        observed_df = build_observed_df_from_revealed_state(tables, replay_state, objective_name=config.target_column)
        candidate_df = build_full_remaining_candidate_df(tables, replay_state)
        display_id = selector(observed_df, candidate_df, config.seed + round_index)
        internal_id = map_display_candidate_to_internal_id(candidate_df, display_id)
        revealed = evaluator.reveal([internal_id])
        replay_state.observe(revealed)
        round_summaries.append(
            _round_summary(
                config=config,
                replay_state=replay_state,
                round_index=round_index,
                display_id=display_id,
                selected_y=_safe_float(revealed[config.target_column].iloc[0]),
                candidate_df=candidate_df,
                fallback_used=False,
            )
        )
    return _summary_from_rounds(
        config=config,
        initial_best=initial_best,
        round_summaries=round_summaries,
        strategy_state={"selected_count": len(round_summaries), "fallback_count": 0},
        output_dir=config.output_dir,
        extra={
            "unique_skill_hash_count": 0,
            "active_unique_skill_hash_count": 0,
            "baseline_note": baseline_note,
        },
    )


def _run_stratified_random_public(*, config: SelfEvolvingConfig, tables: Any, evaluator: OfflineEvaluator) -> dict[str, Any]:
    return _run_selector_policy(
        config=config,
        tables=tables,
        evaluator=evaluator,
        selector=lambda observed_df, candidate_df, seed: _stratified_random_public_select(
            observed_df=observed_df,
            candidate_df=candidate_df,
            seed=seed,
        ),
        baseline_note="Public-feature stratified random baseline using only candidate_df strata and observed coverage.",
    )


def _run_random(*, config: SelfEvolvingConfig, tables: Any, evaluator: OfflineEvaluator) -> dict[str, Any]:
    replay_state = _initial_state(tables=tables, evaluator=evaluator, config=config)
    initial_best = _best(replay_state, config)
    round_summaries: list[dict[str, Any]] = []
    while replay_state.can_continue(config.max_rounds):
        round_index = int(replay_state.round_index) + 1
        candidate_df = build_full_remaining_candidate_df(tables, replay_state)
        fallback = full_pool_random_fallback(candidate_df, seed=config.seed, round_index=round_index, reason="proof_suite")
        display_id = str(fallback["candidate_id"])
        internal_id = map_display_candidate_to_internal_id(candidate_df, display_id)
        revealed = evaluator.reveal([internal_id])
        replay_state.observe(revealed)
        round_summaries.append(
            _round_summary(
                config=config,
                replay_state=replay_state,
                round_index=round_index,
                display_id=display_id,
                selected_y=_safe_float(revealed[config.target_column].iloc[0]),
                candidate_df=candidate_df,
                fallback_used=False,
            )
        )
    return _summary_from_rounds(
        config=config,
        initial_best=initial_best,
        round_summaries=round_summaries,
        strategy_state={"selected_count": len(round_summaries), "fallback_count": 0},
        output_dir=config.output_dir,
        extra={"unique_skill_hash_count": 0, "active_unique_skill_hash_count": 0},
    )


def _lmabo_system_prompt(config: SelfEvolvingConfig) -> str:
    direction = "minimize" if str(config.objective_direction).lower() == "minimize" else "maximize"
    best_name = "lowest" if direction == "minimize" else "highest"
    return f"""
You are an expert in Bayesian Optimization, specifically tasked with recommending the most suitable acquisition function for the next iteration to {direction} an objective function.

For context, we use a Gaussian Process surrogate with a Matern 5/2 ARD-style public-feature representation. This implementation follows the LMABO interface from {LMABO_OFFICIAL_REFERENCE}: at each BO iteration you select only the acquisition function. The runner then optimizes that acquisition over a finite unrevealed candidate pool using only public candidate features, revealed outcomes, and nearest-neighbor public-feature distances.

I will provide a structured summary of the Bayesian Optimization process at each step:
- N: total number of points evaluated so far.
- Remaining iterations: number of reveal iterations left in the optimization process.
- D: dimensionality of the public search representation.
- f_range: range of objective values observed so far.
- f_best: current best ({best_name}) observed objective value.
- Shortest distance: shortest public-feature distance from the last revealed point to any previously revealed point.
- Model lengthscales: min, max, mean, and standard deviation of the ARD-style lengthscale diagnostics.
- Model outputscale: estimated objective amplitude of the surrogate.

Available acquisition functions:
1. PI (Probability of Improvement)
2. LogPI (Log Probability of Improvement)
3. EI (Expected Improvement)
4. LogEI (Log Expected Improvement)
5. UCB (Upper Confidence Bound)
6. PosMean (Posterior Mean)
7. PosSTD (Posterior Standard Deviation)
8. TS (Thompson Sampling)
9. qKG (Knowledge Gradient)
10. qPES (Predictive Entropy Search)
11. qMES (Max-value Entropy Search)
12. qJES (Joint Entropy Search)

Review the state, select the acquisition function that is most appropriate now, and avoid reusing acquisition functions that recently failed to improve the best observed objective value unless the state justifies retrying them. Respond only with JSON matching the provided schema.
""".strip()


def _lmabo_choice_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "acquisition": {"type": "string", "enum": list(LMABO_ACQUISITIONS)},
            "justification": {"type": "string", "maxLength": 800},
        },
        "required": ["acquisition", "justification"],
        "additionalProperties": False,
    }


def _lmabo_follow_up_prompt(
    *,
    config: SelfEvolvingConfig,
    round_index: int,
    diagnostics: dict[str, Any],
    acquisition_history: list[dict[str, Any]],
) -> str:
    recent_history = acquisition_history[-6:]
    if recent_history:
        history_lines = [
            (
                f"- round {item['round_index']}: {item['acquisition']}, "
                f"improved_best={item['improved_best']}, selected_y={_format_float(item.get('selected_yield'))}, "
                f"best_after={_format_float(item.get('best_after'))}"
            )
            for item in recent_history
        ]
    else:
        history_lines = ["- none yet"]
    failed_counts: dict[str, int] = {}
    for item in acquisition_history:
        if not bool(item.get("improved_best")):
            name = str(item.get("acquisition", ""))
            failed_counts[name] = int(failed_counts.get(name, 0)) + 1
    failed_summary = ", ".join(f"{name}:{count}" for name, count in sorted(failed_counts.items())) or "none"
    direction = "minimize" if str(config.objective_direction).lower() == "minimize" else "maximize"
    return f"""
Current optimization state for round {int(round_index)}:
- Objective direction: {direction}
- N: {diagnostics['observed_count']}
- Remaining iterations: {diagnostics['remaining_iterations']}
- Remaining finite-pool candidates: {diagnostics['candidate_count']}
- D: {diagnostics['feature_dimension']}
- f_range: [{_format_float(diagnostics.get('f_min'))}, {_format_float(diagnostics.get('f_max'))}]
- f_mean: {_format_float(diagnostics.get('f_mean'))}
- f_std: {_format_float(diagnostics.get('f_std'))}
- f_best: {_format_float(diagnostics.get('f_best'))}
- Shortest distance from last point: {_format_float(diagnostics.get('shortest_distance_from_last'))}
- Candidate nearest-observed distance range: [{_format_float(diagnostics.get('candidate_nearest_distance_min'))}, {_format_float(diagnostics.get('candidate_nearest_distance_max'))}]
- Model lengthscales min/max/mean/std: {_format_float(diagnostics.get('lengthscale_min'))} / {_format_float(diagnostics.get('lengthscale_max'))} / {_format_float(diagnostics.get('lengthscale_mean'))} / {_format_float(diagnostics.get('lengthscale_std'))}
- Model outputscale: {_format_float(diagnostics.get('outputscale'))}
- Acquisition choices that failed to improve best so far: {failed_summary}

Recent acquisition outcomes:
{chr(10).join(history_lines)}

Choose the next acquisition from {", ".join(LMABO_ACQUISITIONS)}. Return JSON with fields acquisition and justification.
""".strip()


def _lmabo_state_diagnostics(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    config: SelfEvolvingConfig,
    round_index: int,
) -> dict[str, Any]:
    x_obs, x_cand = _public_numeric_feature_matrices(
        observed_df=observed_df,
        candidate_df=candidate_df,
        hashed_features=128,
        max_features=160,
        include_string_descriptors=True,
    )
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce").dropna().to_numpy(dtype=float)
    maximize = str(config.objective_direction).lower() != "minimize"
    if len(y):
        f_best = float(np.max(y) if maximize else np.min(y))
        f_min = float(np.min(y))
        f_max = float(np.max(y))
        f_mean = float(np.mean(y))
        f_std = float(np.std(y))
        outputscale = float(np.var(y)) if len(y) > 1 else 0.0
    else:
        f_best = f_min = f_max = f_mean = f_std = outputscale = None
    shortest = None
    cand_nearest_min = None
    cand_nearest_max = None
    if len(x_obs) >= 2:
        x_obs_scaled, _x_empty = _standardize_train_test(x_obs, np.zeros((0, x_obs.shape[1]), dtype=float))
        last = x_obs_scaled[-1:]
        previous = x_obs_scaled[:-1]
        distances = _nearest_public_distance(last, previous)
        shortest = float(distances[0]) if len(distances) else None
    if len(x_obs) and len(x_cand):
        x_obs_scaled, x_cand_scaled = _standardize_train_test(x_obs, x_cand)
        cand_distances = _nearest_public_distance(x_cand_scaled, x_obs_scaled)
        if len(cand_distances):
            cand_nearest_min = float(np.nanmin(cand_distances))
            cand_nearest_max = float(np.nanmax(cand_distances))
    lengthscales = _lmabo_lengthscale_proxy(x_obs=x_obs, y=y, objective_direction=config.objective_direction)
    return {
        "observed_count": int(len(observed_df)),
        "candidate_count": int(len(candidate_df)),
        "remaining_iterations": int(max(int(config.max_rounds) - int(round_index) + 1, 0)),
        "feature_dimension": int(x_obs.shape[1] if x_obs.ndim == 2 else 0),
        "f_min": f_min,
        "f_max": f_max,
        "f_mean": f_mean,
        "f_std": f_std,
        "f_best": f_best,
        "shortest_distance_from_last": shortest,
        "candidate_nearest_distance_min": cand_nearest_min,
        "candidate_nearest_distance_max": cand_nearest_max,
        "lengthscale_min": float(np.min(lengthscales)) if len(lengthscales) else None,
        "lengthscale_max": float(np.max(lengthscales)) if len(lengthscales) else None,
        "lengthscale_mean": float(np.mean(lengthscales)) if len(lengthscales) else None,
        "lengthscale_std": float(np.std(lengthscales)) if len(lengthscales) else None,
        "outputscale": outputscale,
        "diagnostics_mode": "public_feature_correlation_ard_proxy",
    }


def _lmabo_lengthscale_proxy(*, x_obs: np.ndarray, y: np.ndarray, objective_direction: str) -> np.ndarray:
    if x_obs.ndim != 2 or x_obs.shape[1] == 0:
        return np.ones(1, dtype=float)
    if len(y) < 3 or len(x_obs) != len(y):
        return np.ones(x_obs.shape[1], dtype=float)
    target = -y if str(objective_direction).lower() == "minimize" else y
    x_scaled, _ = _standardize_train_test(x_obs, np.zeros((0, x_obs.shape[1]), dtype=float))
    target_std = float(np.nanstd(target))
    if target_std <= 1e-12:
        return np.ones(x_obs.shape[1], dtype=float)
    centered_y = target - float(np.nanmean(target))
    x_std = np.nanstd(x_scaled, axis=0)
    relevance = np.zeros(x_obs.shape[1], dtype=float)
    for idx in range(x_obs.shape[1]):
        if float(x_std[idx]) <= 1e-12:
            continue
        corr = float(np.nanmean((x_scaled[:, idx] - float(np.nanmean(x_scaled[:, idx]))) * centered_y) / target_std)
        relevance[idx] = abs(corr)
    lengthscales = 1.0 / np.maximum(relevance, 0.05)
    return np.clip(np.nan_to_num(lengthscales, nan=1.0, posinf=20.0, neginf=20.0), 0.05, 20.0)


def _parse_lmabo_acquisition_response(raw_response: str) -> dict[str, str]:
    text = str(raw_response or "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {
                "acquisition": str(parsed.get("acquisition", "")).strip(),
                "justification": str(parsed.get("justification", "")).strip(),
            }
    except json.JSONDecodeError:
        pass
    match = re.search(r"\b(PI|LogPI|EI|LogEI|UCB|PosMean|PosSTD|TS|qKG|qPES|qMES|qJES|KG|PES|MES|JES)\b", text)
    if match:
        acquisition = match.group(1)
        _, _, trailing = text.partition(":")
        return {"acquisition": acquisition, "justification": trailing.strip() or text[:800]}
    raise ValueError("LMABO response did not contain a valid acquisition choice.")


def _normalize_lmabo_acquisition(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {"KG": "qKG", "PES": "qPES", "MES": "qMES", "JES": "qJES"}
    text = aliases.get(text, text)
    if text not in LMABO_ACQUISITIONS:
        raise ValueError(f"Invalid LMABO acquisition: {text!r}")
    return text


def _objective_improved(before: Any, after: Any, objective_direction: str) -> bool:
    before_value = _safe_float(before)
    after_value = _safe_float(after)
    if before_value is None or after_value is None:
        return False
    if str(objective_direction).lower() == "minimize":
        return float(after_value) < float(before_value) - 1e-12
    return float(after_value) > float(before_value) + 1e-12


def _format_float(value: Any) -> str:
    safe = _safe_float(value)
    if safe is None:
        return "NA"
    return f"{safe:.4g}"


def _lmabo_candidate_scores(
    *,
    acquisition: str,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    objective_direction: str,
    seed: int,
) -> np.ndarray:
    acquisition = _normalize_lmabo_acquisition(acquisition)
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    if len(ids) == 0:
        return np.asarray([], dtype=float)
    components = _lmabo_surrogate_components(
        observed_df=observed_df,
        candidate_df=candidate_df,
        objective_direction=objective_direction,
        seed=seed,
    )
    mu = components["mu"]
    sigma = components["sigma"]
    ei = components["ei"]
    logei = components["logei"]
    pi = components["pi"]
    logpi = components["logpi"]
    diversity = components["diversity"]
    category = components["category"]
    nn_mu = components["nn_mu"]
    nn_distance = components["nn_distance"]
    rng = np.random.default_rng(int(seed))
    if acquisition == "PI":
        base = _rank_normalize(pi)
    elif acquisition == "LogPI":
        base = _rank_normalize(logpi)
    elif acquisition == "EI":
        base = _rank_normalize(ei)
    elif acquisition == "LogEI":
        base = _rank_normalize(logei)
    elif acquisition == "UCB":
        beta = 1.25 + 0.25 * math.log(max(int(components["observed_count"]) + 1, 2))
        base = _rank_normalize(mu + float(beta) * sigma)
    elif acquisition == "PosMean":
        base = 0.70 * _rank_normalize(mu) + 0.30 * _rank_normalize(nn_mu)
    elif acquisition == "PosSTD":
        base = 0.72 * _rank_normalize(sigma) + 0.28 * _rank_normalize(nn_distance)
    elif acquisition == "TS":
        sample = mu + rng.normal(0.0, 1.0, size=len(ids)) * sigma
        base = _rank_normalize(sample)
    elif acquisition == "qKG":
        value_of_information = sigma * (0.5 + _rank_normalize(np.abs(mu - float(components["best"]))))
        base = 0.50 * _rank_normalize(value_of_information) + 0.30 * _rank_normalize(ei) + 0.20 * _rank_normalize(diversity)
    elif acquisition == "qPES":
        base = 0.58 * _rank_normalize(sigma) + 0.30 * _rank_normalize(diversity) + 0.12 * _rank_normalize(category)
    elif acquisition == "qMES":
        base = 0.52 * _rank_normalize(sigma) + 0.28 * _rank_normalize(mu) + 0.20 * _rank_normalize(diversity)
    elif acquisition == "qJES":
        base = 0.38 * _rank_normalize(ei) + 0.34 * _rank_normalize(sigma) + 0.28 * _rank_normalize(diversity)
    else:
        base = _rank_normalize(mu + sigma)
    scores = (
        0.82 * _rank_normalize(base)
        + 0.10 * _rank_normalize(nn_mu)
        + 0.05 * _rank_normalize(diversity)
        + 0.03 * _rank_normalize(category)
    )
    return np.asarray(scores, dtype=float) + rng.uniform(0.0, 1e-10, size=len(ids))


def _lmabo_surrogate_components(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    objective_direction: str,
    seed: int,
) -> dict[str, Any]:
    x_obs, x_cand = _public_numeric_feature_matrices(
        observed_df=observed_df,
        candidate_df=candidate_df,
        hashed_features=128,
        max_features=160,
        include_string_descriptors=True,
    )
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    target = -y if str(objective_direction).lower() == "minimize" else y
    diversity = _public_diversity_scores(observed_df=observed_df, candidate_df=candidate_df)
    category = _categorical_descriptor_scores(observed_df=observed_df, candidate_df=candidate_df)
    if len(target) == 0 or len(ids) == 0:
        zero = np.zeros(len(ids), dtype=float)
        return {
            "observed_count": 0,
            "mu": zero,
            "sigma": np.maximum(_rank_normalize(diversity), 1e-3),
            "ei": _rank_normalize(diversity),
            "logei": np.log(np.maximum(_rank_normalize(diversity), 1e-300)),
            "pi": _rank_normalize(diversity),
            "logpi": np.log(np.maximum(_rank_normalize(diversity), 1e-300)),
            "diversity": diversity,
            "category": category,
            "nn_mu": zero,
            "nn_distance": diversity,
            "best": 0.0,
        }
    x_obs_scaled, x_cand_scaled = _standardize_train_test(x_obs, x_cand)
    nn_mu, nn_sigma, nn_distance = _nearest_neighbor_regression_components(
        x_obs=x_obs_scaled,
        x_cand=x_cand_scaled,
        target=target,
    )
    mu = np.asarray(nn_mu, dtype=float)
    sigma = np.maximum(np.asarray(nn_sigma, dtype=float), 1e-6)
    if len(target) >= 3:
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=1.0,
            length_scale_bounds="fixed",
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds="fixed")
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=int(seed),
            optimizer=None,
        )
        try:
            model.fit(x_obs_scaled, target)
            gp_mu, gp_sigma = model.predict(x_cand_scaled, return_std=True)
            mu = 0.70 * np.asarray(gp_mu, dtype=float) + 0.30 * np.asarray(nn_mu, dtype=float)
            sigma = np.maximum(0.70 * np.asarray(gp_sigma, dtype=float) + 0.30 * np.asarray(nn_sigma, dtype=float), 1e-6)
        except Exception:
            pass
    best = float(np.max(target))
    ei = _expected_improvement(mu=mu, sigma=sigma, best=best)
    logei = np.log(np.maximum(ei, 1e-300))
    z = (np.asarray(mu, dtype=float) - float(best)) / np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    pi = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    logpi = np.log(np.maximum(pi, 1e-300))
    return {
        "observed_count": int(len(target)),
        "mu": np.asarray(mu, dtype=float),
        "sigma": np.asarray(sigma, dtype=float),
        "ei": np.asarray(ei, dtype=float),
        "logei": np.asarray(logei, dtype=float),
        "pi": np.asarray(pi, dtype=float),
        "logpi": np.asarray(logpi, dtype=float),
        "diversity": np.asarray(diversity, dtype=float),
        "category": np.asarray(category, dtype=float),
        "nn_mu": np.asarray(nn_mu, dtype=float),
        "nn_distance": np.asarray(nn_distance, dtype=float),
        "best": best,
    }


def _nearest_neighbor_regression_components(
    *,
    x_obs: np.ndarray,
    x_cand: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(x_cand) == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty
    if len(x_obs) == 0 or len(target) == 0:
        zeros = np.zeros(len(x_cand), dtype=float)
        return zeros, np.ones(len(x_cand), dtype=float), np.ones(len(x_cand), dtype=float)
    distances = np.sqrt(((x_cand[:, None, :] - x_obs[None, :, :]) ** 2).mean(axis=2))
    k = min(5, len(target))
    nearest_order = np.argsort(distances, axis=1)[:, :k]
    nearest_distances = np.take_along_axis(distances, nearest_order, axis=1)
    nearest_targets = np.asarray(target, dtype=float)[nearest_order]
    weights = 1.0 / np.maximum(nearest_distances, 1e-6)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    mu = (weights * nearest_targets).sum(axis=1)
    local_var = (weights * (nearest_targets - mu[:, None]) ** 2).sum(axis=1)
    target_std = float(np.std(target)) if len(target) > 1 else 1.0
    sigma = np.sqrt(np.maximum(local_var, 0.0)) + 0.20 * float(max(target_std, 1e-6)) * _rank_normalize(
        nearest_distances[:, 0]
    )
    return mu, np.maximum(sigma, 1e-6), nearest_distances[:, 0]


def _bo_like_select(*, observed_df: pd.DataFrame, candidate_df: pd.DataFrame, seed: int) -> str:
    x_obs, feature_columns = _encode_features(observed_df, fit_columns=None)
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    x_cand, _ = _encode_features(candidate_df, fit_columns=feature_columns)
    if len(y) < 3:
        scores = x_cand.mean(axis=1)
    else:
        model = RandomForestRegressor(n_estimators=64, min_samples_leaf=1, random_state=int(seed), bootstrap=True)
        model.fit(x_obs, y)
        tree_preds = np.asarray([tree.predict(x_cand) for tree in model.estimators_], dtype=float)
        mu = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0)
        scores = mu + 0.75 * sigma
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    order = sorted(range(len(ids)), key=lambda idx: (-float(scores[idx]), ids[idx]))
    return str(ids[order[0]])


def _gp_logei_select(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    objective_direction: str,
    seed: int,
) -> str:
    high_categorical = _high_categorical_candidate_space(candidate_df)
    x_obs, x_cand = _public_numeric_feature_matrices(
        observed_df=observed_df,
        candidate_df=candidate_df,
        hashed_features=48 if high_categorical else 64,
        max_features=64 if high_categorical else 96,
    )
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    if len(y) < 3 or len(ids) == 0:
        scores = _public_diversity_scores(observed_df=observed_df, candidate_df=candidate_df)
    else:
        x_obs_scaled, x_cand_scaled = _standardize_train_test(x_obs, x_cand)
        target = -y if str(objective_direction).lower() == "minimize" else y
        length_scale = np.ones(x_obs_scaled.shape[1], dtype=float)
        kernel = ConstantKernel(1.0, (0.05, 20.0)) * Matern(
            length_scale=length_scale,
            length_scale_bounds=(0.05, 20.0),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1.0))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=int(seed),
            optimizer=None,
        )
        try:
            model.fit(x_obs_scaled, target)
            mu, sigma = model.predict(x_cand_scaled, return_std=True)
            scores = _log_expected_improvement(
                mu=np.asarray(mu),
                sigma=np.asarray(sigma),
                best=float(np.max(target)),
            )
        except Exception:
            scores = _public_diversity_scores(observed_df=observed_df, candidate_df=candidate_df)
    return _top_scoring_candidate(candidate_df, scores)


def _smac_style_rf_ei_select(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    objective_direction: str,
    seed: int,
) -> str:
    x_obs, x_cand = _public_numeric_feature_matrices(
        observed_df=observed_df,
        candidate_df=candidate_df,
        hashed_features=128,
        max_features=192,
    )
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    if len(y) < 3 or len(ids) == 0:
        scores = _public_diversity_scores(observed_df=observed_df, candidate_df=candidate_df)
    else:
        target = -y if str(objective_direction).lower() == "minimize" else y
        x_obs_scaled, x_cand_scaled = _standardize_train_test(x_obs, x_cand)
        model: Any
        if _high_categorical_candidate_space(candidate_df):
            model = ExtraTreesRegressor(
                n_estimators=192,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=int(seed),
                bootstrap=True,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=192,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=int(seed),
                bootstrap=True,
            )
        try:
            model.fit(x_obs_scaled, target)
            tree_preds = np.asarray([tree.predict(x_cand_scaled) for tree in model.estimators_], dtype=float)
            scores = _empirical_expected_improvement(tree_preds=tree_preds, best=float(np.max(target)))
            scores = scores + 0.05 * np.std(tree_preds, axis=0)
        except Exception:
            scores = _public_diversity_scores(observed_df=observed_df, candidate_df=candidate_df)
    return _top_scoring_candidate(candidate_df, scores)


def _chemistry_descriptor_bo_select(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    objective_direction: str,
    seed: int,
) -> str:
    x_obs, x_cand = _public_numeric_feature_matrices(
        observed_df=observed_df,
        candidate_df=candidate_df,
        hashed_features=256,
        max_features=320,
        include_string_descriptors=True,
    )
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    target = -y if str(objective_direction).lower() == "minimize" else y
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    category_scores = _categorical_descriptor_scores(observed_df=observed_df, candidate_df=candidate_df)
    diversity = _public_diversity_scores(observed_df=observed_df, candidate_df=candidate_df)
    if len(target) < 3 or len(ids) == 0:
        scores = 0.70 * category_scores + 0.30 * diversity
    else:
        x_obs_scaled, x_cand_scaled = _standardize_train_test(x_obs, x_cand)
        model = ExtraTreesRegressor(
            n_estimators=256,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=int(seed),
            bootstrap=True,
        )
        try:
            model.fit(x_obs_scaled, target)
            tree_preds = np.asarray([tree.predict(x_cand_scaled) for tree in model.estimators_], dtype=float)
            ei = _empirical_expected_improvement(tree_preds=tree_preds, best=float(np.max(target)))
            mu = tree_preds.mean(axis=0)
            sigma = tree_preds.std(axis=0)
            scores = 0.58 * _rank_normalize(ei) + 0.22 * _rank_normalize(mu) + 0.12 * _rank_normalize(sigma)
            scores = scores + 0.28 * _rank_normalize(category_scores) + 0.08 * _rank_normalize(diversity)
        except Exception:
            scores = 0.70 * category_scores + 0.30 * diversity
    return _top_scoring_candidate(candidate_df, scores)


def _gp_ucb_select(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    objective_direction: str,
    seed: int,
) -> str:
    high_categorical = _high_categorical_candidate_space(candidate_df)
    x_obs, x_cand = _public_numeric_feature_matrices(
        observed_df=observed_df,
        candidate_df=candidate_df,
        hashed_features=64 if high_categorical else 48,
        max_features=96 if high_categorical else 72,
    )
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    if len(y) < 3 or len(ids) == 0:
        scores = _public_diversity_scores(observed_df=observed_df, candidate_df=candidate_df)
    else:
        x_obs_scaled, x_cand_scaled = _standardize_train_test(x_obs, x_cand)
        target = -y if str(objective_direction).lower() == "minimize" else y
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=1.0,
            length_scale_bounds="fixed",
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds="fixed")
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=int(seed),
            optimizer=None,
        )
        try:
            model.fit(x_obs_scaled, target)
            mu, sigma = model.predict(x_cand_scaled, return_std=True)
            beta = 1.25 + 0.25 * math.log(max(len(y) + 1, 2))
            scores = np.asarray(mu, dtype=float) + float(beta) * np.asarray(sigma, dtype=float)
        except Exception:
            scores = _public_diversity_scores(observed_df=observed_df, candidate_df=candidate_df)
    return _top_scoring_candidate(candidate_df, scores)


def _tpe_style_select(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    objective_direction: str,
    seed: int,
) -> str:
    x_obs, x_cand = _public_numeric_feature_matrices(
        observed_df=observed_df,
        candidate_df=candidate_df,
        hashed_features=128,
        max_features=160,
        include_string_descriptors=True,
    )
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(ids) == 0:
        raise ValueError("candidate_df is empty.")
    if len(y) < 4:
        scores = _public_diversity_scores(observed_df=observed_df, candidate_df=candidate_df)
    else:
        target = -y if str(objective_direction).lower() == "minimize" else y
        cutoff = float(np.quantile(target, 0.65))
        good_mask = target >= cutoff
        if int(good_mask.sum()) == 0 or int((~good_mask).sum()) == 0:
            good_mask = target >= float(np.median(target))
        x_obs_scaled, x_cand_scaled = _standardize_train_test(x_obs, x_cand)
        good = x_obs_scaled[good_mask]
        bad = x_obs_scaled[~good_mask]
        if len(good) == 0 or len(bad) == 0:
            scores = _public_diversity_scores(observed_df=observed_df, candidate_df=candidate_df)
        else:
            good_dist = _nearest_public_distance(x_cand_scaled, good)
            bad_dist = _nearest_public_distance(x_cand_scaled, bad)
            density_ratio = _rank_normalize(bad_dist) - _rank_normalize(good_dist)
            diversity = _public_diversity_scores(observed_df=observed_df, candidate_df=candidate_df)
            scores = 0.82 * density_ratio + 0.18 * _rank_normalize(diversity)
            rng = np.random.default_rng(int(seed))
            scores = np.asarray(scores, dtype=float) + rng.uniform(0.0, 1e-9, size=len(ids))
    return _top_scoring_candidate(candidate_df, scores)


def _edbo_style_descriptor_gp_ei_select(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    objective_direction: str,
    seed: int,
) -> str:
    scores = _edbo_style_descriptor_gp_ei_scores(
        observed_df=observed_df,
        candidate_df=candidate_df,
        objective_direction=objective_direction,
        seed=seed,
    )
    return _top_scoring_candidate(candidate_df, scores)


def _edbo_style_descriptor_gp_ei_scores(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    objective_direction: str,
    seed: int,
) -> np.ndarray:
    x_obs, x_cand = _public_numeric_feature_matrices(
        observed_df=observed_df,
        candidate_df=candidate_df,
        hashed_features=256,
        max_features=256,
        include_string_descriptors=True,
    )
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    category_scores = _categorical_descriptor_scores(observed_df=observed_df, candidate_df=candidate_df)
    diversity = _public_diversity_scores(observed_df=observed_df, candidate_df=candidate_df)
    if len(ids) == 0:
        return np.asarray([], dtype=float)
    if len(y) < 3:
        scores = 0.55 * _rank_normalize(category_scores) + 0.45 * _rank_normalize(diversity)
    else:
        target = -y if str(objective_direction).lower() == "minimize" else y
        x_obs_scaled, x_cand_scaled = _standardize_train_test(x_obs, x_cand)
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=1.0,
            length_scale_bounds="fixed",
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-4, noise_level_bounds="fixed")
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=int(seed),
            optimizer=None,
        )
        try:
            model.fit(x_obs_scaled, target)
            mu, sigma = model.predict(x_cand_scaled, return_std=True)
            ei = _expected_improvement(mu=np.asarray(mu), sigma=np.asarray(sigma), best=float(np.max(target)))
            scores = (
                0.62 * _rank_normalize(ei)
                + 0.18 * _rank_normalize(mu)
                + 0.12 * _rank_normalize(category_scores)
                + 0.08 * _rank_normalize(diversity)
            )
        except Exception:
            scores = 0.55 * _rank_normalize(category_scores) + 0.45 * _rank_normalize(diversity)
    return np.asarray(scores, dtype=float)


def _gryffin_style_categorical_select(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    objective_direction: str,
    seed: int,
) -> str:
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    if len(ids) == 0:
        raise ValueError("candidate_df is empty.")
    category_scores = _categorical_descriptor_scores(observed_df=observed_df, candidate_df=candidate_df)
    diversity = _public_diversity_scores(observed_df=observed_df, candidate_df=candidate_df)
    x_obs, x_cand = _public_numeric_feature_matrices(
        observed_df=observed_df,
        candidate_df=candidate_df,
        hashed_features=192,
        max_features=224,
        include_string_descriptors=True,
    )
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(y) < 3:
        scores = 0.72 * _rank_normalize(category_scores) + 0.28 * _rank_normalize(diversity)
    else:
        target = -y if str(objective_direction).lower() == "minimize" else y
        x_obs_scaled, x_cand_scaled = _standardize_train_test(x_obs, x_cand)
        model = ExtraTreesRegressor(
            n_estimators=192,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            random_state=int(seed),
        )
        try:
            model.fit(x_obs_scaled, target)
            tree_preds = np.asarray([tree.predict(x_cand_scaled) for tree in model.estimators_], dtype=float)
            mu = tree_preds.mean(axis=0)
            sigma = tree_preds.std(axis=0)
            scores = (
                0.48 * _rank_normalize(category_scores)
                + 0.24 * _rank_normalize(mu)
                + 0.18 * _rank_normalize(sigma)
                + 0.10 * _rank_normalize(diversity)
            )
        except Exception:
            scores = 0.72 * _rank_normalize(category_scores) + 0.28 * _rank_normalize(diversity)
    return _top_scoring_candidate(candidate_df, scores)


def _baybe_bofire_style_mixed_select(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    objective_direction: str,
    seed: int,
) -> str:
    scores = _baybe_bofire_style_mixed_scores(
        observed_df=observed_df,
        candidate_df=candidate_df,
        objective_direction=objective_direction,
        seed=seed,
    )
    return _top_scoring_candidate(candidate_df, scores)


def _baybe_bofire_style_mixed_scores(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    objective_direction: str,
    seed: int,
) -> np.ndarray:
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    if len(ids) == 0:
        return np.asarray([], dtype=float)
    x_obs, x_cand = _public_numeric_feature_matrices(
        observed_df=observed_df,
        candidate_df=candidate_df,
        hashed_features=256,
        max_features=320,
        include_string_descriptors=True,
    )
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    diversity = _public_diversity_scores(observed_df=observed_df, candidate_df=candidate_df)
    category_scores = _categorical_descriptor_scores(observed_df=observed_df, candidate_df=candidate_df)
    if len(y) < 3:
        scores = 0.50 * _rank_normalize(diversity) + 0.50 * _rank_normalize(category_scores)
    else:
        target = -y if str(objective_direction).lower() == "minimize" else y
        x_obs_scaled, x_cand_scaled = _standardize_train_test(x_obs, x_cand)
        model = ExtraTreesRegressor(
            n_estimators=256,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            random_state=int(seed),
        )
        try:
            model.fit(x_obs_scaled, target)
            tree_preds = np.asarray([tree.predict(x_cand_scaled) for tree in model.estimators_], dtype=float)
            ei = _empirical_expected_improvement(tree_preds=tree_preds, best=float(np.max(target)))
            mu = tree_preds.mean(axis=0)
            sigma = tree_preds.std(axis=0)
            scores = (
                0.42 * _rank_normalize(ei)
                + 0.22 * _rank_normalize(mu)
                + 0.18 * _rank_normalize(sigma)
                + 0.10 * _rank_normalize(diversity)
                + 0.08 * _rank_normalize(category_scores)
            )
        except Exception:
            scores = 0.50 * _rank_normalize(diversity) + 0.50 * _rank_normalize(category_scores)
    return np.asarray(scores, dtype=float)


def _categorical_eb_select(*, observed_df: pd.DataFrame, candidate_df: pd.DataFrame, seed: int) -> str:
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    if len(ids) == 0:
        raise ValueError("candidate_df is empty.")
    feature_columns = [
        str(column)
        for column in candidate_df.columns
        if str(column) != "candidate_id" and not pd.api.types.is_numeric_dtype(candidate_df[column])
    ][:8]
    if not feature_columns or observed_df.empty:
        rng = np.random.default_rng(int(seed))
        return str(ids[int(rng.integers(0, len(ids)))])
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce")
    global_mean = float(y.dropna().mean()) if len(y.dropna()) else 0.0
    stats: dict[str, dict[str, list[float]]] = {column: {} for column in feature_columns}
    for _, row in observed_df.iterrows():
        target = _safe_float(row.get("observed_y"))
        if target is None:
            continue
        for column in feature_columns:
            if column in observed_df.columns:
                stats[column].setdefault(str(row.get(column, "")), []).append(float(target))
    scores: list[float] = []
    rng = np.random.default_rng(int(seed))
    for _, row in candidate_df.iterrows():
        score = global_mean
        for column in feature_columns:
            values = stats[column].get(str(row.get(column, "")), [])
            if values:
                count = len(values)
                mean = (sum(values) + 3.0 * global_mean) / (count + 3.0)
                score += 0.40 * mean + 0.20 / math.sqrt(float(count))
            else:
                score += 0.08
        score += float(rng.uniform(0.0, 1e-9))
        scores.append(float(score))
    order = sorted(range(len(ids)), key=lambda idx: (-float(scores[idx]), ids[idx]))
    return str(ids[order[0]])


def _gp_ei_select(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    objective_direction: str,
    seed: int,
) -> str:
    x_obs, feature_columns = _encode_features(observed_df, fit_columns=None)
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    x_cand, _ = _encode_features(candidate_df, fit_columns=feature_columns)
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    if len(y) < 3 or len(ids) == 0:
        scores = x_cand.mean(axis=1)
    else:
        x_obs_scaled, x_cand_scaled = _standardize_train_test(x_obs, x_cand)
        target = -y if str(objective_direction).lower() == "minimize" else y
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=1.0,
            length_scale_bounds="fixed",
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds="fixed")
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=int(seed),
            optimizer=None,
        )
        try:
            model.fit(x_obs_scaled, target)
            mu, sigma = model.predict(x_cand_scaled, return_std=True)
            scores = _expected_improvement(mu=np.asarray(mu), sigma=np.asarray(sigma), best=float(np.max(target)))
        except Exception:
            scores = x_cand.mean(axis=1)
    order = sorted(range(len(ids)), key=lambda idx: (-float(scores[idx]), ids[idx]))
    return str(ids[order[0]])


def _encode_features(frame: pd.DataFrame, fit_columns: list[str] | None) -> tuple[np.ndarray, list[str]]:
    public = frame.drop(columns=[col for col in ["candidate_id", "observation_id", "observed_y"] if col in frame.columns])
    encoded = pd.get_dummies(public, dummy_na=True)
    encoded = encoded.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if fit_columns is None:
        columns = list(encoded.columns)
    else:
        columns = fit_columns
        encoded = encoded.reindex(columns=columns, fill_value=0)
    return encoded.to_numpy(dtype=float), columns


def _public_numeric_feature_matrices(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    hashed_features: int,
    max_features: int,
    include_string_descriptors: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    obs_public = _public_feature_frame(observed_df)
    cand_public = _public_feature_frame(candidate_df)
    combined = pd.concat([obs_public, cand_public], ignore_index=True, sort=False)
    numeric_columns = [
        str(column)
        for column in combined.columns
        if pd.api.types.is_numeric_dtype(combined[column])
    ]
    if numeric_columns:
        numeric = combined.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        numeric = np.zeros((len(combined), 0), dtype=float)
    text_columns = [str(column) for column in combined.columns if column not in numeric_columns]
    if int(hashed_features) > 0 and text_columns:
        token_rows = [
            _hash_tokens_for_row(row, columns=text_columns, include_descriptors=include_string_descriptors)
            for _, row in combined.loc[:, text_columns].iterrows()
        ]
        hashed = FeatureHasher(
            n_features=int(hashed_features),
            input_type="string",
            alternate_sign=False,
        ).transform(token_rows).toarray()
    else:
        hashed = np.zeros((len(combined), 0), dtype=float)
    features = np.hstack([numeric, hashed]) if numeric.shape[1] or hashed.shape[1] else np.zeros((len(combined), 1))
    if features.shape[1] > int(max_features):
        variances = np.nanvar(features, axis=0)
        keep = np.argsort(-variances)[: int(max_features)]
        features = features[:, keep]
    features = np.nan_to_num(features.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    obs_count = len(obs_public)
    return features[:obs_count], features[obs_count:]


def _public_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    drop = [column for column in ["candidate_id", "observation_id", "observed_y"] if column in frame.columns]
    public = frame.drop(columns=drop).copy()
    return public.reset_index(drop=True)


def _stratified_random_public_select(*, observed_df: pd.DataFrame, candidate_df: pd.DataFrame, seed: int) -> str:
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    if len(ids) == 0:
        raise ValueError("candidate_df is empty.")
    candidate_strata = _public_strata(candidate_df, reference_df=candidate_df)
    observed_strata = _public_strata(observed_df, reference_df=candidate_df) if not observed_df.empty else []
    observed_counts = pd.Series(observed_strata, dtype=object).value_counts().to_dict() if observed_strata else {}
    weights = np.asarray([float(observed_counts.get(str(stratum), 0)) for stratum in candidate_strata], dtype=float)
    min_seen = float(np.min(weights)) if len(weights) else 0.0
    eligible = np.flatnonzero(weights <= min_seen)
    if len(eligible) == 0:
        eligible = np.arange(len(ids))
    rng = np.random.default_rng(int(seed))
    chosen = int(eligible[int(rng.integers(0, len(eligible)))])
    return str(ids[chosen])


def _public_strata(frame: pd.DataFrame, *, reference_df: pd.DataFrame) -> list[str]:
    public = _public_feature_frame(frame)
    reference = _public_feature_frame(reference_df)
    if public.empty or reference.empty:
        return ["<empty>"] * len(public)
    categorical_columns = [
        str(column)
        for column in reference.columns
        if column in public.columns and not pd.api.types.is_numeric_dtype(reference[column])
    ]
    numeric_columns = [
        str(column)
        for column in reference.columns
        if column in public.columns and pd.api.types.is_numeric_dtype(reference[column])
    ]
    selected_columns = categorical_columns[:3] + numeric_columns[:3]
    if not selected_columns:
        return ["<empty>"] * len(public)
    parts: list[list[str]] = []
    for column in selected_columns:
        if column in categorical_columns:
            values = public[column].astype(str).fillna("<NA>").tolist()
            parts.append([f"{column}={value}" for value in values])
            continue
        ref_values = pd.to_numeric(reference[column], errors="coerce").dropna()
        values = pd.to_numeric(public[column], errors="coerce")
        if ref_values.empty:
            labels = [f"{column}=nan"] * len(public)
        else:
            cuts = np.unique(np.quantile(ref_values.to_numpy(dtype=float), [0.33, 0.66]))
            labels = [
                f"{column}=bin{int(np.searchsorted(cuts, float(value), side='right'))}" if pd.notna(value) else f"{column}=nan"
                for value in values.tolist()
            ]
        parts.append(labels)
    return ["|".join(items) for items in zip(*parts)]


def _hash_tokens_for_row(row: pd.Series, *, columns: list[str], include_descriptors: bool) -> list[str]:
    tokens: list[str] = []
    for column in columns:
        value = row.get(column)
        if pd.isna(value):
            text = "<NA>"
        else:
            text = str(value)
        tokens.append(f"{column}={text}")
        if include_descriptors:
            tokens.extend(_string_descriptor_tokens(column=column, text=text))
    return tokens or ["<empty-public-row>"]


def _string_descriptor_tokens(*, column: str, text: str) -> list[str]:
    stripped = text.strip()
    tokens = [
        f"{column}:len_bin={min(len(stripped) // 12, 20)}",
        f"{column}:digit_count={min(sum(ch.isdigit() for ch in stripped), 20)}",
        f"{column}:upper_count={min(sum(ch.isupper() for ch in stripped), 20)}",
    ]
    for char in "CONFPSIBrcnl[]=#()+-@":
        count = stripped.count(char)
        if count:
            tokens.append(f"{column}:char:{char}:{min(count, 20)}")
    compact = "".join(ch for ch in stripped if not ch.isspace())
    tri_count = max(0, len(compact) - 2)
    if tri_count <= 32:
        tri_indices = range(tri_count)
    else:
        tri_indices = list(range(24)) + list(range(max(24, tri_count - 8), tri_count))
    for idx in tri_indices:
        tokens.append(f"{column}:tri={compact[idx:idx + 3]}")
    return tokens


def _public_diversity_scores(*, observed_df: pd.DataFrame, candidate_df: pd.DataFrame) -> np.ndarray:
    x_obs, x_cand = _public_numeric_feature_matrices(
        observed_df=observed_df,
        candidate_df=candidate_df,
        hashed_features=64,
        max_features=96,
    )
    if len(x_cand) == 0:
        return np.asarray([], dtype=float)
    if len(x_obs) == 0:
        norms = np.linalg.norm(x_cand - np.nanmean(x_cand, axis=0, keepdims=True), axis=1)
        return _rank_normalize(norms)
    x_obs_scaled, x_cand_scaled = _standardize_train_test(x_obs, x_cand)
    distances = np.sqrt(((x_cand_scaled[:, None, :] - x_obs_scaled[None, :, :]) ** 2).mean(axis=2))
    min_distance = distances.min(axis=1)
    return _rank_normalize(min_distance)


def _nearest_public_distance(x_cand: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    if len(x_cand) == 0:
        return np.asarray([], dtype=float)
    if len(x_ref) == 0:
        return np.zeros(len(x_cand), dtype=float)
    distances = np.sqrt(((x_cand[:, None, :] - x_ref[None, :, :]) ** 2).mean(axis=2))
    return distances.min(axis=1)


def _categorical_descriptor_scores(*, observed_df: pd.DataFrame, candidate_df: pd.DataFrame) -> np.ndarray:
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    if len(ids) == 0:
        return np.asarray([], dtype=float)
    public_candidate = _public_feature_frame(candidate_df)
    object_columns = [
        str(column)
        for column in public_candidate.columns
        if not pd.api.types.is_numeric_dtype(public_candidate[column])
    ][:12]
    if not object_columns or observed_df.empty:
        return np.zeros(len(ids), dtype=float)
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce").dropna()
    global_mean = float(y.mean()) if len(y) else 0.0
    stats: dict[str, dict[str, list[float]]] = {column: {} for column in object_columns}
    for _, row in observed_df.iterrows():
        target = _safe_float(row.get("observed_y"))
        if target is None:
            continue
        for column in object_columns:
            if column in observed_df.columns:
                stats[column].setdefault(str(row.get(column, "")), []).append(float(target))
    scores = np.zeros(len(ids), dtype=float)
    unseen = float(global_mean) + 0.15
    for column in object_columns:
        contribution_map: dict[str, float] = {}
        for value, values in stats[column].items():
            count = len(values)
            if count <= 0:
                continue
            mean = (sum(values) + 4.0 * global_mean) / (count + 4.0)
            contribution_map[str(value)] = float(mean + 0.10 / math.sqrt(float(count)))
        values = candidate_df[column].astype(str).fillna("") if column in candidate_df.columns else pd.Series([""] * len(ids))
        scores += values.map(contribution_map).fillna(unseen).to_numpy(dtype=float)
    return scores / float(max(len(object_columns), 1))


def _high_categorical_candidate_space(candidate_df: pd.DataFrame) -> bool:
    public = _public_feature_frame(candidate_df)
    object_columns = [
        column
        for column in public.columns
        if not pd.api.types.is_numeric_dtype(public[column])
    ]
    if not object_columns:
        return False
    max_unique = max(int(public[column].nunique(dropna=False)) for column in object_columns)
    return max_unique >= 20 or len(object_columns) >= 3


def _empirical_expected_improvement(*, tree_preds: np.ndarray, best: float) -> np.ndarray:
    improvements = np.maximum(np.asarray(tree_preds, dtype=float) - float(best), 0.0)
    return improvements.mean(axis=0)


def _log_expected_improvement(*, mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
    ei = _expected_improvement(mu=mu, sigma=sigma, best=best)
    return np.log(np.maximum(ei, 1e-300))


def _rank_normalize(values: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    finite = np.nan_to_num(arr, nan=float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0)
    order = np.argsort(np.argsort(finite, kind="mergesort"), kind="mergesort").astype(float)
    denom = max(len(finite) - 1, 1)
    return order / float(denom)


def _top_scoring_candidate(candidate_df: pd.DataFrame, scores: np.ndarray | list[float]) -> str:
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    if len(ids) == 0:
        raise ValueError("candidate_df is empty.")
    arr = np.nan_to_num(np.asarray(scores, dtype=float), nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))
    if len(arr) != len(ids):
        raise ValueError("scores length does not match candidate_df.")
    order = sorted(range(len(ids)), key=lambda idx: (-float(arr[idx]), ids[idx]))
    return str(ids[order[0]])


def _rank_rows_from_scores(
    *,
    candidate_df: pd.DataFrame,
    scores: np.ndarray | list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    if len(ids) == 0:
        return []
    arr = np.nan_to_num(np.asarray(scores, dtype=float), nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))
    if len(arr) != len(ids):
        raise ValueError("scores length does not match candidate_df.")
    order = sorted(range(len(ids)), key=lambda idx: (-float(arr[idx]), ids[idx]))
    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(order[: max(1, int(top_k))], start=1):
        score = float(arr[int(idx)])
        if not np.isfinite(score):
            score = 0.0
        rows.append({"candidate_id": str(ids[int(idx)]), "rank": int(rank), "score": score})
    return rows


def _standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmean(x_train, axis=0)
    scale = np.nanstd(x_train, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (x_train - center) / scale, (x_test - center) / scale


def _expected_improvement(*, mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    improvement = np.asarray(mu, dtype=float) - float(best)
    z = improvement / sigma
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return improvement * cdf + sigma * pdf


def _initial_state(*, tables: Any, evaluator: OfflineEvaluator, config: SelfEvolvingConfig) -> Any:
    return initialize_full_pool_replay_state(
        tables=tables,
        evaluator=evaluator,
        seed=config.seed,
        initial_observed_count=min(config.initial_observed_count, max(1, len(tables.candidate_table) - 1)),
    )


def _best(replay_state: Any, config: SelfEvolvingConfig) -> float | None:
    return _safe_float(replay_state.best_observed(config.target_column, config.objective_direction))


def _round_summary(
    *,
    config: SelfEvolvingConfig,
    replay_state: Any,
    round_index: int,
    display_id: str,
    selected_y: float | None,
    candidate_df: pd.DataFrame,
    fallback_used: bool,
) -> dict[str, Any]:
    return {
        "round_index": int(round_index),
        "selected_display_candidate_id": str(display_id),
        "selected_yield": selected_y,
        "best_observed_yield": _best(replay_state, config),
        "fallback_used": bool(fallback_used),
        "candidate_df_rows": int(len(candidate_df)),
        "full_remaining_pool_size": int(candidate_df.attrs.get("full_remaining_pool_size", len(candidate_df))),
        "selected_from_full_pool": str(display_id) in set(candidate_df["candidate_id"].astype(str).tolist()),
    }


def _summary_from_rounds(
    *,
    config: SelfEvolvingConfig,
    initial_best: float | None,
    round_summaries: list[dict[str, Any]],
    strategy_state: dict[str, Any],
    output_dir: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    best_curve = [_safe_float(row.get("best_observed_yield")) for row in round_summaries]
    best_curve = [value for value in best_curve if value is not None]
    selected_curve = [_safe_float(row.get("selected_yield")) for row in round_summaries]
    selected_curve = [value for value in selected_curve if value is not None]
    final_best = best_curve[-1] if best_curve else initial_best
    payload = {
        "status": "ok",
        "run_id": config.run_id,
        "mode": config.mode,
        "seed": int(config.seed),
        "selected_count": len(round_summaries),
        "initial_best_yield": initial_best,
        "final_best_yield": final_best,
        "delta_best_yield": None if initial_best is None or final_best is None else final_best - initial_best,
        "auc_best_observed_yield": sum(best_curve) / len(best_curve) if best_curve else final_best,
        "average_selected_yield": sum(selected_curve) / len(selected_curve) if selected_curve else None,
        "cumulative_selected_yield": sum(selected_curve) if selected_curve else None,
        "fallback_count": int(strategy_state.get("fallback_count", 0) or 0),
        "planner_call_count": int(strategy_state.get("planner_call_count", 0) or 0),
        "skill_synthesis_count": int(strategy_state.get("skill_synthesis_count", 0) or 0),
        "gate_pass_count": int(strategy_state.get("gate_pass_count", 0) or 0),
        "gate_reject_count": int(strategy_state.get("gate_reject_count", 0) or 0),
        "reward_count": int(strategy_state.get("reward_count", 0) or 0),
        "care_certificate_observed_count": int(strategy_state.get("care_certificate_observed_count", 0) or 0),
        "care_certificate_selected_count": int(strategy_state.get("care_certificate_selected_count", 0) or 0),
        "care_certificate_improvement_count": int(strategy_state.get("care_certificate_improvement_count", 0) or 0),
        "care_certificate_delta_best_sum": _safe_float(strategy_state.get("care_certificate_delta_best_sum")) or 0.0,
        "round_summaries": round_summaries,
        "output_dir": output_dir,
    }
    payload.update(extra or {})
    return payload


def _augment_rows_with_oracle_metrics(rows: list[dict[str, Any]], *, base: SelfEvolvingConfig) -> list[dict[str, Any]]:
    try:
        tables, _table_source = _load_tables(base)
    except Exception:
        return rows
    target = str(base.target_column)
    if target not in tables.outcome_table.columns or "candidate_id" not in tables.outcome_table.columns:
        return rows
    outcome = tables.outcome_table.loc[:, ["candidate_id", target]].copy()
    outcome[target] = pd.to_numeric(outcome[target], errors="coerce")
    outcome = outcome.dropna(subset=[target])
    if outcome.empty:
        return rows
    maximize = str(base.objective_direction).lower() != "minimize"
    values = outcome[target].to_numpy(dtype=float)
    oracle_best = float(np.max(values) if maximize else np.min(values))
    oracle_worst = float(np.min(values) if maximize else np.max(values))
    oracle_span = max(abs(oracle_best - oracle_worst), 1e-12)
    ascending = not maximize
    ranked = outcome.sort_values([target, "candidate_id"], ascending=[ascending, True]).reset_index(drop=True)
    internal_ids = sorted(dict.fromkeys(tables.candidate_table[tables.id_column].tolist()), key=lambda value: str(value))
    internal_to_display = {str(internal_id): f"cand_{idx:06d}" for idx, internal_id in enumerate(internal_ids, start=1)}
    ranked_internal_ids = ranked["candidate_id"].astype(str).tolist()
    rank_by_id: dict[str, int] = {}
    for idx, internal_id in enumerate(ranked_internal_ids, start=1):
        rank_by_id[str(internal_id)] = int(idx)
        display_id = internal_to_display.get(str(internal_id))
        if display_id is not None:
            rank_by_id[str(display_id)] = int(idx)
    top_sets = {
        k: set(
            str(item)
            for internal_id in ranked_internal_ids[:k]
            for item in (internal_id, internal_to_display.get(str(internal_id)))
            if item is not None
        )
        for k in (1, 5, 10)
    }
    thresholds = {
        80: oracle_worst + (0.80 * (oracle_best - oracle_worst)),
        90: oracle_worst + (0.90 * (oracle_best - oracle_worst)),
        95: oracle_worst + (0.95 * (oracle_best - oracle_worst)),
    }

    for row in rows:
        round_summaries = list(row.get("round_summaries", []))
        selected_ids = [str(item.get("selected_display_candidate_id", "")) for item in round_summaries]
        best_curve = [_safe_float(item.get("best_observed_yield")) for item in round_summaries]
        final_best = _safe_float(row.get("final_best_yield"))
        simple_regret = _simple_regret(final_best, oracle_best=oracle_best, maximize=maximize)
        row.update(
            {
                "oracle_best_yield": oracle_best,
                "oracle_worst_yield": oracle_worst,
                "oracle_range_yield": oracle_span,
                "simple_regret": simple_regret,
                "normalized_simple_regret": None if simple_regret is None else float(simple_regret) / oracle_span,
                "top1_hit": any(cid in top_sets[1] for cid in selected_ids),
                "top5_hit": any(cid in top_sets[5] for cid in selected_ids),
                "top10_hit": any(cid in top_sets[10] for cid in selected_ids),
            }
        )
        for pct, threshold in thresholds.items():
            row[f"time_to_{pct}pct_oracle"] = _time_to_threshold(
                best_curve,
                threshold=threshold,
                maximize=maximize,
            )
        for item in round_summaries:
            selected_id = str(item.get("selected_display_candidate_id", ""))
            round_best = _safe_float(item.get("best_observed_yield"))
            round_regret = _simple_regret(round_best, oracle_best=oracle_best, maximize=maximize)
            item["selected_global_rank"] = rank_by_id.get(selected_id)
            item["selected_is_top1"] = selected_id in top_sets[1]
            item["selected_is_top5"] = selected_id in top_sets[5]
            item["selected_is_top10"] = selected_id in top_sets[10]
            item["simple_regret"] = round_regret
            item["normalized_simple_regret"] = None if round_regret is None else float(round_regret) / oracle_span
        row["round_summaries"] = round_summaries
    return rows


def _simple_regret(value: Any, *, oracle_best: float, maximize: bool) -> float | None:
    current = _safe_float(value)
    if current is None:
        return None
    return float(oracle_best - current) if maximize else float(current - oracle_best)


def _time_to_threshold(values: list[float | None], *, threshold: float, maximize: bool) -> int | None:
    for index, value in enumerate(values, start=1):
        if value is None:
            continue
        if (maximize and float(value) >= float(threshold)) or ((not maximize) and float(value) <= float(threshold)):
            return int(index)
    return None


def _skill_hash_summary(output_dir: Path, *, active_only: bool = False) -> dict[str, Any]:
    skill_dir = output_dir / "self_evolving_skills"
    sources = [path.read_bytes() for path in skill_dir.glob("*.py")]
    all_hashes = [hashlib.sha256(source).hexdigest() for source in sources]
    state_payload = _read_json(output_dir / "self_evolving_policy_state.json")
    skill_history = list(state_payload.get("skill_history", [])) if isinstance(state_payload, dict) else []
    active_hashes = sorted(
        {
            str(row.get("source_hash"))
            for row in skill_history
            if row.get("activated") is True and row.get("source_hash")
        }
    )
    payload = {
        "skill_file_count": len(all_hashes),
        "unique_skill_hash_count": len(active_hashes if active_only else set(all_hashes)),
        "skill_source_hashes": active_hashes if active_only else sorted(set(all_hashes)),
        "active_skill_file_count": sum(1 for row in skill_history if row.get("activated") is True),
        "active_unique_skill_hash_count": len(active_hashes),
        "active_skill_source_hashes": active_hashes,
        "rejected_skill_file_count": max(0, len(all_hashes) - len(active_hashes)),
    }
    return payload


def _first_activated_skill(output_dir: Path) -> SkillArtifact | None:
    state_payload = _read_json(output_dir / "self_evolving_policy_state.json")
    skill_history = list(state_payload.get("skill_history", [])) if isinstance(state_payload, dict) else []
    activated = [row for row in skill_history if row.get("activated") is True]
    if not activated:
        return None
    first = sorted(
        activated,
        key=lambda row: (
            int(row.get("created_round", 10**9) or 10**9),
            int(row.get("version", 10**9) or 10**9),
            str(row.get("skill_id", "")),
        ),
    )[0]
    try:
        return SkillRegistry(output_dir).load_skill(str(first["skill_id"]), int(first["version"]))
    except (KeyError, FileNotFoundError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _failed_run_summary(*, config: SelfEvolvingConfig, exc: Exception, output_dir: str) -> dict[str, Any]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    row = _summary_from_rounds(
        config=config,
        initial_best=None,
        round_summaries=[],
        strategy_state={},
        output_dir=output_dir,
        extra={
            "status": "failed",
            "error_type": exc.__class__.__name__,
            "error": str(exc)[:800],
            "traceback_tail": traceback.format_exc(limit=6)[-2000:],
            **_skill_hash_summary(Path(output_dir), active_only=True),
        },
    )
    write_json(Path(output_dir) / "proof_policy_failure.json", row)
    return row


def _write_failed_initial_api_summary(
    *,
    config: SelfEvolvingConfig,
    agent: SelfEvolvingFullPoolAgent,
    decision: Any,
) -> None:
    state = agent.state
    write_json(
        Path(config.output_dir) / "fixed_api_initial_skill_failure.json",
        {
            "status": "failed",
            "reason": "no_deployable_initial_skill",
            "decision_metadata": getattr(decision, "decision_metadata", {}),
            "strategy_state": state.strategy_state if state is not None else {},
            "policy_state": agent.registry.policy_state.to_dict(),
            **_skill_hash_summary(Path(config.output_dir), active_only=True),
        },
    )


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for policy in sorted({str(row["policy_name"]) for row in rows}):
        subset = [row for row in rows if row["policy_name"] == policy]
        result.append(
            {
                "policy_name": policy,
                "run_success_count": sum(1 for row in subset if row.get("status") == "ok"),
                "mean_final_best_yield": _mean(_values(subset, "final_best_yield")),
                "mean_delta_best_yield": _mean(_values(subset, "delta_best_yield")),
                "mean_auc_best_observed_yield": _mean(_values(subset, "auc_best_observed_yield")),
                "mean_average_selected_yield": _mean(_values(subset, "average_selected_yield")),
                "mean_simple_regret": _mean(_values(subset, "simple_regret")),
                "mean_normalized_simple_regret": _mean(_values(subset, "normalized_simple_regret")),
                "top1_hit_rate": _mean(_values(subset, "top1_hit")),
                "top5_hit_rate": _mean(_values(subset, "top5_hit")),
                "top10_hit_rate": _mean(_values(subset, "top10_hit")),
                "mean_time_to_80pct_oracle": _mean(_values(subset, "time_to_80pct_oracle")),
                "mean_time_to_90pct_oracle": _mean(_values(subset, "time_to_90pct_oracle")),
                "mean_time_to_95pct_oracle": _mean(_values(subset, "time_to_95pct_oracle")),
                "total_planner_call_count": sum(int(row.get("planner_call_count", 0) or 0) for row in subset),
                "total_skill_synthesis_count": sum(int(row.get("skill_synthesis_count", 0) or 0) for row in subset),
                "total_gate_pass_count": sum(int(row.get("gate_pass_count", 0) or 0) for row in subset),
                "total_gate_reject_count": sum(int(row.get("gate_reject_count", 0) or 0) for row in subset),
                "total_fallback_count": sum(int(row.get("fallback_count", 0) or 0) for row in subset),
                "total_care_certificate_selected_count": sum(
                    int(row.get("care_certificate_selected_count", 0) or 0) for row in subset
                ),
                "total_care_certificate_improvement_count": sum(
                    int(row.get("care_certificate_improvement_count", 0) or 0) for row in subset
                ),
                "mean_unique_skill_hash_count": _mean(_values(subset, "unique_skill_hash_count")),
                "mean_active_unique_skill_hash_count": _mean(_values(subset, "active_unique_skill_hash_count")),
            }
        )
    return result


def _proof_checks(
    rows: list[dict[str, Any]],
    *,
    selected_policies: list[str],
    max_rounds: int,
) -> dict[str, Any]:
    policies = {str(row.get("policy_name")) for row in rows}
    true_rows = [row for row in rows if row.get("policy_name") == "true_self_evolving_api"]
    required_baseline_groups = {
        "random": bool({"random_full_pool"} & policies),
        "stratified_random": bool({"stratified_random_public"} & policies),
        "classical_bo": {"classical_bo_gp_ei", "bo_like_surrogate"}.issubset(policies),
        "expanded_classical_bo": {"classical_bo_gp_ei", "classical_bo_gp_ucb", "bo_like_surrogate"}.issubset(policies),
        "strong_bo_suite": {"botorch_style_gp_logei", "smac_style_rf_ei", "chemistry_descriptor_bo"}.issubset(policies),
        "expanded_strong_bo_suite": {
            "botorch_style_gp_logei",
            "smac_style_rf_ei",
            "tpe_style_bo",
            "chemistry_descriptor_bo",
        }.issubset(policies),
        "chemistry_bo_suite": {
            "edbo_style_descriptor_gp_ei",
            "gryffin_style_categorical_bo",
            "baybe_bofire_style_mixed_bo",
        }.issubset(policies),
        "fixed_no_llm": bool({"fixed_public_heuristic"} & policies),
        "llm_assisted_no_evolve": bool({"no_evolve_api_reuse", "shared_initial_no_evolve_api"} & policies),
        "self_evolving_ablation": {"llm_only_self_evolving", "public_expert_only_meta_controller"}.issubset(policies),
        "true_self_evolving": bool({"true_self_evolving_api"} & policies),
    }
    true_round_complete = [
        bool(row.get("selected_count") == int(max_rounds) and row.get("reward_count") == int(max_rounds))
        for row in true_rows
    ]
    planner_complete = [
        bool(row.get("planner_call_count") == int(max_rounds))
        for row in true_rows
    ]
    active_skill_hash_counts = [int(row.get("active_unique_skill_hash_count", 0) or 0) for row in true_rows]
    pairwise = _pairwise_vs(rows, reference_policy="true_self_evolving_api")
    pairwise_auc = _pairwise_vs(
        rows,
        reference_policy="true_self_evolving_api",
        metric="auc_best_observed_yield",
    )
    selected_seed_count = len({int(row.get("seed")) for row in rows})
    true_seed_count = len({int(row.get("seed")) for row in rows if row.get("policy_name") == "true_self_evolving_api"})
    success_by_policy = {
        policy: sum(1 for row in rows if row.get("policy_name") == policy and row.get("status") == "ok")
        for policy in sorted(policies)
    }
    full_success_by_policy = {
        policy: count == selected_seed_count and selected_seed_count > 0
        for policy, count in success_by_policy.items()
    }
    pairwise_full_seed_coverage = {
        item["comparison_policy"]: int(item.get("evaluated_seed_count", 0) or 0) == true_seed_count
        for item in pairwise
    }
    positive_pairwise = {
        item["comparison_policy"]: (
            item.get("mean_final_best_delta") is not None and float(item["mean_final_best_delta"]) > 0.0
        )
        for item in pairwise
    }
    positive_pairwise_auc = {
        item["comparison_policy"]: (
            item.get("mean_auc_best_observed_yield_delta") is not None
            and float(item["mean_auc_best_observed_yield_delta"]) > 0.0
        )
        for item in pairwise_auc
    }
    margin_threshold = 1.2
    pairwise_final_delta_by_policy = {
        item["comparison_policy"]: item.get("mean_final_best_delta")
        for item in pairwise
    }
    pairwise_auc_delta_by_policy = {
        item["comparison_policy"]: item.get("mean_auc_best_observed_yield_delta")
        for item in pairwise_auc
    }
    public_final_delta = pairwise_final_delta_by_policy.get("public_expert_only_meta_controller")
    public_auc_delta = pairwise_auc_delta_by_policy.get("public_expert_only_meta_controller")
    public_margin_met = bool(
        public_final_delta is not None
        and public_auc_delta is not None
        and float(public_final_delta) >= margin_threshold
        and float(public_auc_delta) >= margin_threshold
    )
    return {
        "selected_policies": list(selected_policies),
        "required_baseline_groups_present": required_baseline_groups,
        "all_required_baseline_groups_present": all(required_baseline_groups.values()),
        "expected_seed_count": selected_seed_count,
        "true_self_evolving_seed_count": true_seed_count,
        "success_count_by_policy": success_by_policy,
        "all_selected_policy_runs_successful": all(full_success_by_policy.values()) if full_success_by_policy else False,
        "full_success_by_policy": full_success_by_policy,
        "pairwise_full_seed_coverage": pairwise_full_seed_coverage,
        "all_pairwise_comparisons_have_full_seed_coverage": (
            all(pairwise_full_seed_coverage.values()) if pairwise_full_seed_coverage else False
        ),
        "uses_real_dataset_only": all(row.get("table_source") != "synthetic_fallback" for row in rows),
        "true_self_evolving_run_count": len(true_rows),
        "true_self_evolving_all_rounds_rewarded": all(true_round_complete) if true_rows else False,
        "true_self_evolving_all_rounds_planned": all(planner_complete) if true_rows else False,
        "true_self_evolving_any_code_evolution": any(count > 1 for count in active_skill_hash_counts),
        "true_self_evolving_mean_active_unique_skill_hash_count": _mean(
            [float(count) for count in active_skill_hash_counts]
        ),
        "true_self_evolving_pairwise_final_best_positive": positive_pairwise,
        "true_self_evolving_pairwise_auc_positive": positive_pairwise_auc,
        "true_self_evolving_pairwise_final_best_delta_by_policy": pairwise_final_delta_by_policy,
        "true_self_evolving_pairwise_auc_delta_by_policy": pairwise_auc_delta_by_policy,
        "true_self_evolving_vs_public_only_margin_threshold": float(margin_threshold),
        "true_self_evolving_vs_public_only_margin_met": public_margin_met,
        "reviewer_note": (
            "For a claim that self-evolution works, require true_self_evolving_api to be run on matched seeds, "
            "show reward/planner calls every round, show at least some accepted code evolution, and beat no-evolve "
            "LLM-assisted plus BO baselines on paired final-best/AUC metrics. For the stronger LLM marginal-benefit "
            "claim, require true_self_evolving_api to beat public_expert_only_meta_controller by at least 1.2 mean "
            "final-best and 1.2 mean AUC on matched seeds."
        ),
    }


def _care_checks(
    rows: list[dict[str, Any]],
    *,
    selected_policies: list[str],
    max_rounds: int,
) -> dict[str, Any]:
    care_policies = [policy for policy in selected_policies if policy in CARE_POLICIES]
    if not care_policies:
        return {"care_policies_selected": False}
    selected_seed_count = len({int(row.get("seed")) for row in rows})
    success_by_policy = {
        policy: sum(1 for row in rows if row.get("policy_name") == policy and row.get("status") == "ok")
        for policy in care_policies
    }
    full_success_by_policy = {
        policy: count == selected_seed_count and selected_seed_count > 0 for policy, count in success_by_policy.items()
    }
    round_complete_by_policy = {
        policy: all(
            int(row.get("selected_count", 0) or 0) == int(max_rounds)
            and int(row.get("reward_count", 0) or 0) == int(max_rounds)
            for row in rows
            if row.get("policy_name") == policy
        )
        for policy in care_policies
    }
    care_pairwise_final = (
        _pairwise_vs(rows, reference_policy="true_self_evolving_api_care_log_only")
        if "true_self_evolving_api_care_log_only" in care_policies
        else []
    )
    care_pairwise_auc = (
        _pairwise_vs(rows, reference_policy="true_self_evolving_api_care_log_only", metric="auc_best_observed_yield")
        if "true_self_evolving_api_care_log_only" in care_policies
        else []
    )
    return {
        "care_policies_selected": True,
        "selected_care_policies": care_policies,
        "expected_seed_count": selected_seed_count,
        "success_count_by_policy": success_by_policy,
        "full_success_by_policy": full_success_by_policy,
        "all_care_runs_successful": all(full_success_by_policy.values()) if full_success_by_policy else False,
        "all_care_runs_complete": all(round_complete_by_policy.values()) if round_complete_by_policy else False,
        "round_complete_by_policy": round_complete_by_policy,
        "has_full_care_mainline": "true_self_evolving_api_care" in care_policies,
        "has_log_only_reference": "true_self_evolving_api_care_log_only" in care_policies,
        "has_no_adaptive_planner_ablation": "true_self_evolving_api_care_no_adaptive_planner" in care_policies,
        "has_no_certificate_ablation": "true_self_evolving_api_care_no_certificate" in care_policies,
        "has_no_residual_scout_ablation": "true_self_evolving_api_care_no_residual_scout" in care_policies,
        "has_no_macro_scout_ablation": "true_self_evolving_api_care_no_macro_scout" in care_policies,
        "care_vs_log_only_final": care_pairwise_final,
        "care_vs_log_only_auc": care_pairwise_auc,
        "reviewer_note": (
            "CARE-only checks are scoped to targeted CARE validation. Full self-evolving claims still require the "
            "standard proof_checks with non-CARE true_self_evolving_api and matched baselines."
        ),
    }


def _pairwise_vs(
    rows: list[dict[str, Any]],
    *,
    reference_policy: str,
    metric: str = "final_best_yield",
) -> list[dict[str, Any]]:
    by_key = {(str(row["policy_name"]), int(row["seed"])): row for row in rows}
    seeds = sorted(seed for policy, seed in by_key if policy == reference_policy)
    result = []
    for policy in sorted({str(row["policy_name"]) for row in rows if row["policy_name"] != reference_policy}):
        deltas = []
        wins = losses = ties = 0
        for seed in seeds:
            ref = by_key.get((reference_policy, seed))
            other = by_key.get((policy, seed))
            if ref is None or other is None:
                continue
            delta = _delta(ref.get(metric), other.get(metric))
            if delta is None:
                continue
            deltas.append(delta)
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
            else:
                ties += 1
        ci_low, ci_high = _bootstrap_ci(deltas)
        result.append(
            _with_legacy_pairwise_alias(
                {
                    "reference_policy": reference_policy,
                    "comparison_policy": policy,
                    f"mean_{metric}_delta": _mean(deltas),
                    f"mean_{metric}_delta_ci95_low": ci_low,
                    f"mean_{metric}_delta_ci95_high": ci_high,
                    "win_count": wins,
                    "loss_count": losses,
                    "tie_count": ties,
                    "evaluated_seed_count": len(deltas),
                    "positive_delta_fraction": (wins / len(deltas)) if deltas else None,
                },
                metric=metric,
            )
        )
    return result


def _with_legacy_pairwise_alias(row: dict[str, Any], *, metric: str) -> dict[str, Any]:
    if metric == "final_best_yield":
        row["mean_final_best_delta"] = row.get("mean_final_best_yield_delta")
        row["mean_final_best_delta_ci95_low"] = row.get("mean_final_best_yield_delta_ci95_low")
        row["mean_final_best_delta_ci95_high"] = row.get("mean_final_best_yield_delta_ci95_high")
    elif metric == "auc_best_observed_yield":
        row["mean_auc_delta"] = row.get("mean_auc_best_observed_yield_delta")
        row["mean_auc_delta_ci95_low"] = row.get("mean_auc_best_observed_yield_delta_ci95_low")
        row["mean_auc_delta_ci95_high"] = row.get("mean_auc_best_observed_yield_delta_ci95_high")
    return row


def _bootstrap_ci(values: list[float], *, confidence: float = 0.95, repeats: int = 5000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        value = float(arr[0])
        return value, value
    rng = np.random.default_rng(9173 + len(arr))
    samples = rng.choice(arr, size=(int(repeats), len(arr)), replace=True).mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result["status"],
        "seeds": result["seeds"],
        "max_rounds": result["max_rounds"],
        "executor_type": result.get("executor_type", "thread"),
        "policies": result["policies"],
        "aggregate": result["aggregate"],
        "pairwise_vs_true_self_evolving": result["pairwise_vs_true_self_evolving"],
        "pairwise_auc_vs_true_self_evolving": result["pairwise_auc_vs_true_self_evolving"],
        "pairwise_vs_care_log_only": result.get("pairwise_vs_care_log_only", []),
        "pairwise_auc_vs_care_log_only": result.get("pairwise_auc_vs_care_log_only", []),
        "proof_checks": result["proof_checks"],
        "care_checks": result.get("care_checks", {}),
        "output_dir": result["output_dir"],
    }


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [value for value in (_safe_float(row.get(key)) for row in rows) if value is not None]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _delta(left: Any, right: Any) -> float | None:
    left_value = _safe_float(left)
    right_value = _safe_float(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _parse_seeds(text: str) -> list[int]:
    result: list[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            left, right = chunk.split("-", 1)
            result.extend(range(int(left), int(right) + 1))
        else:
            result.append(int(chunk))
    return sorted(dict.fromkeys(result))


def _ordered_policies(policies: list[str]) -> list[str]:
    ordered = list(dict.fromkeys(policies))
    dependent = "shared_initial_no_evolve_api"
    dependency = "true_self_evolving_api"
    if dependent in ordered and dependency in ordered and ordered.index(dependent) < ordered.index(dependency):
        ordered.remove(dependent)
        ordered.insert(ordered.index(dependency) + 1, dependent)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a finite-pool CARE replay proof suite.")
    parser.add_argument("config", nargs="?", default="configs/minerva_care_replay.json")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--output-dir", default="results/care_replay_proof_suite")
    parser.add_argument("--policies", default=",".join(PROOF_POLICIES))
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Run independent seed/policy jobs concurrently. shared_initial_no_evolve_api waits for matched true runs.",
    )
    parser.add_argument(
        "--executor",
        choices=("thread", "process"),
        default="thread",
        help="Parallel executor. Use process for CPU-bound non-API baselines; use thread for API policies.",
    )
    args = parser.parse_args()
    run_proof_suite(
        args.config,
        seeds=_parse_seeds(args.seeds),
        max_rounds=int(args.max_rounds),
        output_dir=args.output_dir,
        policies=[item.strip() for item in args.policies.split(",") if item.strip()],
        parallel_workers=int(args.parallel_workers),
        executor_type=str(args.executor),
    )


if __name__ == "__main__":
    main()
