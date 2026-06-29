"""Self-evolving full-pool policy-editing agent."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from research_tool_agent_full_pool.fallback import full_pool_random_fallback
from research_tool_agent_full_pool.harness.compiler import compile_skill_to_tool
from research_tool_agent_full_pool.harness.gate import run_conservative_gate
from research_tool_agent_full_pool.harness.ledger import HarnessLedger, write_json
from research_tool_agent_full_pool.harness.planner import plan_next_task
from research_tool_agent_full_pool.harness.reward import build_reward_record
from research_tool_agent_full_pool.harness.skill_registry import SkillRegistry
from research_tool_agent_full_pool.harness.skill_synthesis import synthesize_skill
from research_tool_agent_full_pool.harness.specs import GateReport, SkillArtifact, TaskPlan
from research_tool_agent_full_pool.memory import initialize_memory, update_memory_after_reveal
from research_tool_agent_full_pool.state import ResearchToolRunState
from research_tool_agent_full_pool.tool_output_parser import parse_ranked_candidates
from research_tool_agent_full_pool.tool_runner import run_rank_candidates_tool
from research_tool_agent_full_pool.views import (
    build_full_remaining_candidate_df,
    build_observed_df_from_revealed_state,
    map_display_candidate_to_internal_id,
    validate_observed_df,
    validate_public_candidate_df,
)


@dataclass(frozen=True)
class SelfEvolvingConfig:
    run_id: str = "self_evolving_full_pool_smoke"
    dataset_name: str = "MINERVA"
    decision_policy_name: str = "self_evolving_policy_editor_full_pool"
    output_dir: str = "results/self_evolving_full_pool_smoke"
    mode: str = "fake"
    seed: int = 0
    max_rounds: int = 3
    initial_observed_count: int = 5
    target_column: str = "yield"
    objective_direction: str = "maximize"
    repair_attempts: int = 1
    model: str = "gpt-5.4"
    reasoning_effort: str = "medium"
    max_tokens: int = 6000
    timeout: float = 240.0
    allow_synthetic_fallback: bool = False
    resume_policy_state: bool = False
    api_mode: str = "responses"
    stream: bool = True
    response_verbosity: str = "low"
    chemlex_duplicate_policy: str = "raw"
    chemlex_duplicate_conflict_threshold: float | None = None
    chemlex_duplicate_conflict_action: str = "keep"
    chemlex_candidate_id_policy: str = "sequential"
    chemlex_row_shuffle_seed: int | None = None
    adaptive_categorical_experts: bool = False
    parse_failure_reuse_active: bool = False
    api_parse_retry_attempts: int = 2
    llm_residual_scout_enabled: bool = True
    llm_residual_scout_public_locked: bool = True
    llm_residual_scout_budget: int = 2
    llm_residual_scout_top_k: int = 8
    llm_residual_scout_min_round: int = 4
    llm_residual_scout_best_threshold: float = 75.0
    llm_residual_scout_stagnation_best_threshold: float = 84.0
    llm_residual_scout_min_public_support: int = 2
    llm_residual_scout_public_rank_limit: int = 4
    llm_residual_scout_min_certificate_score: float = 1.20
    llm_residual_scout_chemlex_model_band_enabled: bool = True
    llm_residual_scout_chemlex_model_band_min_round: int = 3
    llm_residual_scout_chemlex_anchor_guard_threshold: float = 0.85
    llm_macro_frontier_scout_enabled: bool = True
    llm_macro_frontier_scout_budget: int = 1
    llm_macro_frontier_scout_min_round: int = 2
    llm_macro_frontier_scout_min_best_threshold: float = 35.0
    llm_macro_frontier_scout_low_best_threshold: float = 72.0
    llm_macro_frontier_scout_high_confidence_best_threshold: float = 78.0
    care_enabled: bool = False
    care_certificate_mode: str = "off"
    care_adaptive_planner_enabled: bool = False
    care_certificate_margin: float = 0.0


@dataclass
class SelfEvolvingDecision:
    selected_candidate_ids: list[Any]
    policy_name: str
    round_index: int
    decision_metadata: dict[str, Any]
    fallback_used: bool = False


class SelfEvolvingFullPoolAgent:
    """Planner + skill editing + conservative gate + reward loop."""

    def __init__(self, *, config: SelfEvolvingConfig, client: Any | None = None) -> None:
        self.config = config
        self.client = client
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = HarnessLedger(self.output_dir)
        self.registry = SkillRegistry(self.output_dir, ledger=self.ledger)
        self.state: ResearchToolRunState | None = None
        self._pending_gate_report: GateReport | None = None
        self._pending_task_plan: TaskPlan | None = None
        self._pending_skill: SkillArtifact | None = None
        self._pending_tool_state: dict[str, Any] = {}
        self._pending_tool_diagnostics: dict[str, Any] = {}
        self._pending_decision_metadata: dict[str, Any] = {}
        self._pending_selected_skill: SkillArtifact | None = None
        self._last_gate_report: dict[str, Any] | None = None
        self._last_best_before_decision: float | None = None

    def initialize_run(self, *, tables: Any, replay_state: Any) -> ResearchToolRunState:
        current_best = _safe_float(replay_state.best_observed(self.config.target_column, self.config.objective_direction))
        memory_text = initialize_memory(
            task_name=self.config.dataset_name,
            objective_name=self.config.target_column,
            observed_count=len(replay_state.observed_candidates),
            current_best_y=current_best,
            tool_name=None,
        ).replace("Deterministic fake full-pool scoring", "Self-evolving full-pool policy editing")
        self.state = ResearchToolRunState.initialize_run_state(
            run_id=self.config.run_id,
            memory_text=memory_text,
            strategy_state={
                "run_id": self.config.run_id,
                "dataset_name": self.config.dataset_name,
                "objective_name": self.config.target_column,
                "mode": self.config.mode,
                "rounds_completed": 0,
                "selected_count": 0,
                "fallback_count": 0,
                "planner_call_count": 0,
                "skill_synthesis_count": 0,
                "gate_pass_count": 0,
                "gate_reject_count": 0,
                "tool_output_valid_count": 0,
                "reward_count": 0,
                **(
                    {
                        "care_enabled": True,
                        "care_controller_family": "certified_adaptive_residual_evolution",
                        "care_certificate_mode": str(self.config.care_certificate_mode),
                        "care_adaptive_planner_enabled": bool(self.config.care_adaptive_planner_enabled),
                    }
                    if bool(self.config.care_enabled)
                    else {}
                ),
            },
            tool_state={},
        )
        self.ledger.reset()
        if not self.config.resume_policy_state:
            self.registry.reset_policy_state()
        write_json(self.output_dir / "self_evolving_config.json", self.config.__dict__)
        self.ledger.append("run_initialized", {"run_id": self.config.run_id, "current_best_y": current_best})
        return self.state

    def decide(self, *, tables: Any, replay_state: Any) -> SelfEvolvingDecision:
        run_state = self.state or self.initialize_run(tables=tables, replay_state=replay_state)
        round_index = int(getattr(replay_state, "round_index", 0)) + 1
        self._reset_pending_round_state()
        self._last_best_before_decision = _safe_float(
            replay_state.best_observed(self.config.target_column, self.config.objective_direction)
        )
        observed_df = build_observed_df_from_revealed_state(tables, replay_state, objective_name=self.config.target_column)
        candidate_df = build_full_remaining_candidate_df(tables, replay_state)
        validate_observed_df(observed_df)
        validate_public_candidate_df(candidate_df)
        is_full_pool = len(candidate_df) == len(replay_state.remaining_candidates)
        if not is_full_pool:
            return self._fallback_decision(candidate_df=candidate_df, round_index=round_index, reason="candidate_df_not_full_pool")

        active_skill = self.registry.active_skill()
        task_plan, plan_artifacts = self._plan_next_task_with_repairs(
            run_id=self.config.run_id,
            round_index=round_index,
            observed_df=observed_df,
            candidate_df=candidate_df,
            memory_text=run_state.memory_text,
            policy_state=self.registry.policy_state.to_dict(),
            last_gate_report=self._last_gate_report,
        )
        task_plan, override_report = self._controller_override_task_plan(task_plan=task_plan, active_skill=active_skill)
        if override_report:
            plan_artifacts = dict(plan_artifacts)
            plan_artifacts["controller_override"] = override_report
        self._pending_task_plan = task_plan
        run_state.strategy_state["planner_call_count"] = int(run_state.strategy_state.get("planner_call_count", 0)) + 1
        self.ledger.append("task_planned", {"round_index": round_index, "task_plan": task_plan.to_dict(), "artifacts": plan_artifacts})

        deployed_skill = active_skill
        gate_report: GateReport | None = None
        patch_deployed = False
        reused_after_synthesis_failure = False
        if task_plan.action != "reuse_active_skill" or active_skill is None:
            try:
                candidate_skill, gate_report, synthesis_artifacts = self._synthesize_gate_register_with_repairs(
                    run_id=self.config.run_id,
                    round_index=round_index,
                    task_plan=task_plan,
                    observed_df=observed_df,
                    candidate_df=candidate_df,
                    memory_text=run_state.memory_text,
                    policy_state=self.registry.policy_state.to_dict(),
                    active_skill=active_skill,
                )
            except Exception as exc:
                if not bool(self.config.parse_failure_reuse_active) or active_skill is None:
                    raise
                deployed_skill = active_skill
                task_plan = TaskPlan(
                    action="reuse_active_skill",
                    skill_family=task_plan.skill_family,
                    objective="Reuse active skill after a failed patch synthesis attempt.",
                    target_skill_id=active_skill.skill_id,
                    risk_budget="low",
                    required_checks=list(task_plan.required_checks),
                    rationale="Host-side controller reused active skill after schema/API parse failure.",
                )
                self._pending_task_plan = task_plan
                self._last_gate_report = {
                    "passed": True,
                    "deployable": True,
                    "checks": {},
                    "reason": "patch_synthesis_failed_reused_active",
                }
                run_state.strategy_state["skill_synthesis_parse_failure_reuse_count"] = int(
                    run_state.strategy_state.get("skill_synthesis_parse_failure_reuse_count", 0)
                ) + 1
                self.ledger.append(
                    "skill_synthesis_failed_reuse_active",
                    {
                        "round_index": round_index,
                        "active_skill": active_skill.artifact_key,
                        "error": _public_retry_error(exc),
                    },
                )
                gate_report = None
                synthesis_artifacts = {"gate_reject_count": 0, "reused_active_after_synthesis_failure": True}
                patch_deployed = False
                reused_after_synthesis_failure = True
                candidate_skill = active_skill
            self._pending_skill = candidate_skill
            self._pending_gate_report = gate_report
            if gate_report is not None:
                self._last_gate_report = gate_report.to_dict()
            gate_reject_count = int(synthesis_artifacts.get("gate_reject_count", 0) or 0)
            if gate_reject_count:
                run_state.strategy_state["gate_reject_count"] = int(run_state.strategy_state.get("gate_reject_count", 0)) + gate_reject_count
            if gate_report is not None and gate_report.deployable:
                deployed_skill = candidate_skill
                patch_deployed = True
                run_state.record_tool_created(
                    source=candidate_skill.source,
                    tool_name=candidate_skill.skill_id,
                    round_index=round_index,
                ) if run_state.generated_tool_source is None else run_state.record_tool_replaced(
                    source=candidate_skill.source,
                    tool_name=candidate_skill.skill_id,
                    round_index=round_index,
                    patch_decision=task_plan.action,
                    verification_result=gate_report.to_dict(),
                    provenance={"skill_family": candidate_skill.family},
                )
                run_state.strategy_state["gate_pass_count"] = int(run_state.strategy_state.get("gate_pass_count", 0)) + 1
            elif reused_after_synthesis_failure:
                run_state.record_tool_reused()
        elif active_skill is not None:
            run_state.record_tool_reused()

        if deployed_skill is None:
            return self._fallback_decision(candidate_df=candidate_df, round_index=round_index, reason="no_deployable_skill")

        try:
            selected_skill, parsed, selection_report = self._select_skill_output(
                deployed_skill=deployed_skill,
                active_before_deploy=active_skill,
                observed_df=observed_df,
                candidate_df=candidate_df,
                memory_text=run_state.memory_text,
                tool_state=run_state.tool_state,
                patch_deployed=patch_deployed,
                round_index=round_index,
            )
            if selected_skill.artifact_key != deployed_skill.artifact_key:
                deployed_skill = selected_skill
                self.registry.activate_skill(selected_skill)
                run_state.strategy_state["portfolio_switch_count"] = int(
                    run_state.strategy_state.get("portfolio_switch_count", 0)
                ) + 1
            run_state.sync_active_tool(
                source=deployed_skill.source,
                tool_name=deployed_skill.skill_id,
                version=int(deployed_skill.version),
            )
            selected_new_patch = bool(
                patch_deployed and self._pending_skill is not None and deployed_skill.artifact_key == self._pending_skill.artifact_key
            )
            expert_decision = self._meta_controller_select_candidate(
                parsed=parsed,
                observed_df=observed_df,
                candidate_df=candidate_df,
                round_index=round_index,
                portfolio_selection=selection_report,
            )
            selected_display_id = str(expert_decision["selected_display_candidate_id"])
            selected_internal_id = map_display_candidate_to_internal_id(candidate_df, selected_display_id)
            selected_tool_row = next(
                (row for row in parsed.ranked_candidates if str(row["candidate_id"]) == selected_display_id),
                {
                    "candidate_id": selected_display_id,
                    "rank": expert_decision.get("selected_rank", -1),
                    "score": expert_decision.get("selected_meta_score", 0.0),
                },
            )
            run_state.strategy_state["tool_output_valid_count"] = int(run_state.strategy_state.get("tool_output_valid_count", 0)) + 1
            self._pending_tool_state = dict(parsed.tool_state)
            self._pending_tool_diagnostics = dict(parsed.tool_diagnostics)
            self._pending_selected_skill = deployed_skill
            metadata = {
                "selection_rule": str(expert_decision.get("selection_version", "self_evolving_expert_meta")),
                "selected_display_candidate_id": selected_display_id,
                "candidate_df_rows": len(candidate_df),
                "full_remaining_pool_size": len(replay_state.remaining_candidates),
                "selected_from_full_pool": True,
                "fallback_used": False,
                "active_skill_id": deployed_skill.skill_id,
                "active_skill_version": int(deployed_skill.version),
                "active_skill_hash": deployed_skill.source_hash,
                "task_plan": task_plan.to_dict(),
                "gate_passed": bool(gate_report.deployable if gate_report else True),
                "patch_generated_this_round": patch_deployed,
                "patch_deployed": selected_new_patch,
                "portfolio_selection": selection_report,
                "expert_meta_selection": expert_decision,
                "public_only_counterfactual": expert_decision.get("public_only_counterfactual"),
                "care_adaptive_controller": override_report
                if bool(self.config.care_enabled) and bool(self.config.care_adaptive_planner_enabled)
                else None,
                "care_public_certificate": expert_decision.get("care_public_certificate"),
                "llm_changed_final_selection": bool(
                    expert_decision.get("public_only_counterfactual", {}).get("selected_display_candidate_id")
                    and str(expert_decision.get("public_only_counterfactual", {}).get("selected_display_candidate_id"))
                    != selected_display_id
                ),
                "llm_residual_scout": expert_decision.get("llm_residual_scout"),
                "selected_tool_score": selected_tool_row["score"],
                "selected_tool_rank": selected_tool_row["rank"],
            }
            self._pending_decision_metadata = metadata
            self.ledger.append("decision", {"round_index": round_index, **metadata})
            return SelfEvolvingDecision(
                selected_candidate_ids=[selected_internal_id],
                policy_name=self.config.decision_policy_name,
                round_index=round_index,
                decision_metadata=metadata,
                fallback_used=False,
            )
        except Exception as exc:
            return self._fallback_decision(
                candidate_df=candidate_df,
                round_index=round_index,
                reason=f"active_skill_failure:{exc.__class__.__name__}",
            )

    def update_after_reveal(
        self,
        *,
        tables: Any,
        replay_state: Any,
        decision: SelfEvolvingDecision,
        revealed_rows: pd.DataFrame,
    ) -> ResearchToolRunState:
        if self.state is None:
            raise RuntimeError("Agent run state has not been initialized.")
        round_index = int(decision.round_index)
        current_best = _safe_float(replay_state.best_observed(self.config.target_column, self.config.objective_direction))
        selected_display = str(decision.decision_metadata.get("selected_display_candidate_id", ""))
        active_skill = self._pending_selected_skill or self.registry.active_skill()
        reward = build_reward_record(
            round_index=round_index,
            selected_candidate_id=selected_display,
            revealed_rows=revealed_rows,
            previous_best_y=self._last_best_before_decision,
            current_best_y=current_best,
            objective_name=self.config.target_column,
            objective_direction=self.config.objective_direction,
            gate_report=self._gate_report_for_reward(decision),
            fallback_used=decision.fallback_used,
            patch_deployed=bool(decision.decision_metadata.get("patch_deployed")),
            source=active_skill.source if active_skill else "",
            selection_attribution=_selection_attribution_from_decision(decision.decision_metadata),
        )
        self.registry.record_reward(reward)
        self.state.strategy_state["reward_count"] = int(self.state.strategy_state.get("reward_count", 0)) + 1
        self.state.strategy_state["rounds_completed"] = round_index
        self.state.strategy_state["selected_count"] = int(self.state.strategy_state.get("selected_count", 0)) + len(decision.selected_candidate_ids)
        self.state.strategy_state["fallback_count"] = int(self.state.strategy_state.get("fallback_count", 0)) + int(bool(decision.fallback_used))
        self.state.strategy_state["last_reward"] = reward.total_reward
        self.state.strategy_state["current_best_observed_y"] = current_best
        memory_text = update_memory_after_reveal(
            task_name=self.config.dataset_name,
            objective_name=self.config.target_column,
            observed_count=len(replay_state.observed_candidates),
            last_selected_candidate=selected_display,
            last_revealed_y=reward.revealed_y,
            current_best_y=current_best,
            tool_name=active_skill.skill_id if active_skill else None,
            round_index=round_index,
            previous_memory=self.state.memory_text,
        )
        self.state.update_after_reveal(
            memory_text=memory_text,
            strategy_state_updates={
                "last_selected_display_candidate_id": selected_display,
                "last_revealed_y": reward.revealed_y,
                "last_reward": reward.total_reward,
                "active_skill_id": active_skill.skill_id if active_skill else None,
                "active_skill_version": active_skill.version if active_skill else None,
            },
            tool_state_updates={
                **self._pending_tool_state,
                "last_reward": reward.total_reward,
                "last_selected_display_candidate_id": selected_display,
            },
        )
        rollback_report = self._maybe_rollback_after_reveal(reward=reward, active_skill=active_skill)
        if bool(decision.decision_metadata.get("llm_changed_final_selection")):
            self.state.strategy_state["llm_changed_final_selection_count"] = int(
                self.state.strategy_state.get("llm_changed_final_selection_count", 0)
            ) + 1
            if float(reward.delta_best_y) > 0.0:
                self.state.strategy_state["llm_changed_final_selection_improvement_count"] = int(
                    self.state.strategy_state.get("llm_changed_final_selection_improvement_count", 0)
                ) + 1
                self.state.strategy_state["llm_changed_final_selection_delta_best_sum"] = float(
                    self.state.strategy_state.get("llm_changed_final_selection_delta_best_sum", 0.0)
                ) + float(reward.delta_best_y)
        if bool(self.config.care_enabled):
            _care_update_strategy_after_reveal(
                strategy_state=self.state.strategy_state,
                decision_metadata=decision.decision_metadata,
                reward=reward,
            )
        if rollback_report.get("rolled_back"):
            self.state.strategy_state["rollback_count"] = int(self.state.strategy_state.get("rollback_count", 0)) + 1
            replacement = self.registry.active_skill()
            if replacement is not None:
                active_skill = replacement
                self.state.sync_active_tool(
                    source=replacement.source,
                    tool_name=replacement.skill_id,
                    version=int(replacement.version),
                )
                self.state.strategy_state["active_skill_id"] = replacement.skill_id
                self.state.strategy_state["active_skill_version"] = int(replacement.version)
            self.ledger.append("skill_rollback_after_reveal", rollback_report)
        summary = {
            "round_index": round_index,
            "selected_display_candidate_id": selected_display,
            "selected_yield": reward.revealed_y,
            "best_observed_yield": current_best,
            "fallback_used": bool(decision.fallback_used),
            "reward": reward.total_reward,
            "candidate_df_rows": decision.decision_metadata.get("candidate_df_rows"),
            "full_remaining_pool_size": decision.decision_metadata.get("full_remaining_pool_size"),
            "selected_from_full_pool": decision.decision_metadata.get("selected_from_full_pool"),
            "active_skill_id": active_skill.skill_id if active_skill else None,
            "active_skill_version": active_skill.version if active_skill else None,
            "rollback_after_reveal": rollback_report,
            "expert_meta_selection": decision.decision_metadata.get("expert_meta_selection"),
            "public_only_counterfactual": decision.decision_metadata.get("public_only_counterfactual"),
            "care_adaptive_controller": decision.decision_metadata.get("care_adaptive_controller"),
            "care_public_certificate": decision.decision_metadata.get("care_public_certificate"),
            "llm_changed_final_selection": bool(decision.decision_metadata.get("llm_changed_final_selection")),
            "llm_residual_scout": decision.decision_metadata.get("llm_residual_scout"),
        }
        self.state.record_round_summary(summary)
        self.ledger.append("round_updated", {"round_index": round_index, "summary": summary, "reward": reward.to_dict()})
        write_json(self.output_dir / "self_evolving_strategy_state.json", self.state.strategy_state)
        write_json(self.output_dir / "self_evolving_tool_state.json", self.state.tool_state)
        write_json(self.output_dir / "self_evolving_round_summaries.json", self.state.round_summaries)
        return self.state

    def _fallback_decision(self, *, candidate_df: pd.DataFrame, round_index: int, reason: str) -> SelfEvolvingDecision:
        fallback = full_pool_random_fallback(candidate_df, seed=self.config.seed, round_index=round_index, reason=reason)
        display_id = str(fallback["candidate_id"])
        internal_id = map_display_candidate_to_internal_id(candidate_df, display_id)
        metadata = {
            "selection_rule": "self_evolving_full_pool_seeded_random_fallback",
            "selected_display_candidate_id": display_id,
            "candidate_df_rows": len(candidate_df),
            "full_remaining_pool_size": int(candidate_df.attrs.get("full_remaining_pool_size", len(candidate_df))),
            "selected_from_full_pool": display_id in set(candidate_df["candidate_id"].astype(str).tolist()),
            "fallback_used": True,
            "fallback_reason": reason,
            **fallback,
        }
        self._pending_decision_metadata = metadata
        self._pending_tool_state = {}
        self._pending_tool_diagnostics = {"fallback_reason": reason}
        self.ledger.append("fallback_decision", {"round_index": round_index, **metadata})
        return SelfEvolvingDecision(
            selected_candidate_ids=[internal_id],
            policy_name=self.config.decision_policy_name,
            round_index=round_index,
            decision_metadata=metadata,
            fallback_used=True,
        )

    def _controller_override_task_plan(self, *, task_plan: TaskPlan, active_skill: SkillArtifact | None) -> tuple[TaskPlan, dict[str, Any] | None]:
        if active_skill is None:
            return task_plan, None
        if bool(self.config.care_enabled) and bool(self.config.care_adaptive_planner_enabled):
            trend = _reward_trend_from_records(self.registry.policy_state.reward_history)
            adaptive_state = _care_adaptive_planner_state(
                reward_history=self.registry.policy_state.reward_history,
                strategy_state=self.state.strategy_state if self.state is not None else {},
            )
            if self.state is not None:
                self.state.strategy_state["care_adaptive_planner_state"] = adaptive_state
            action_prior = str(adaptive_state.get("action_prior", "defer"))
            if action_prior == "defer" and bool(trend["trailing_no_improvement_count"] >= 2 or trend["last_selected_far_below_best"]):
                action_prior = "patch_skill"
                adaptive_state = {
                    **adaptive_state,
                    "action_prior": action_prior,
                    "legacy_controller_compatible": True,
                    "legacy_controller_reason": "retain existing public reward-trend patch trigger",
                }
                if self.state is not None:
                    self.state.strategy_state["care_adaptive_planner_state"] = adaptive_state
            if action_prior == "patch_skill" and task_plan.action == "reuse_active_skill":
                patched_plan = TaskPlan(
                    action="patch_skill",
                    skill_family="ranker",
                    objective=_care_patch_objective(adaptive_state=adaptive_state),
                    target_skill_id=active_skill.skill_id,
                    risk_budget="low",
                    required_checks=list(task_plan.required_checks),
                    rationale=(
                        "CARE adaptive public controller raised patch intensity from reward stagnation, "
                        "while preserving the public-only gate and final expert arbitration."
                    ),
                )
                return patched_plan, {
                    "controller_family": "care_certified_adaptive_residual_evolution",
                    "from_action": task_plan.action,
                    "to_action": "patch_skill",
                    "reward_trend": trend,
                    "care_adaptive_state": adaptive_state,
                    "adaevolve_distinction": (
                        "CARE changes only the public-certified planner action prior inside an HTE reveal loop; "
                        "it does not run AdaEvolve-style multi-island program search or replace final selection."
                    ),
                }
            if action_prior == "reuse_active_skill" and task_plan.action == "patch_skill":
                reuse_plan = TaskPlan(
                    action="reuse_active_skill",
                    skill_family=task_plan.skill_family,
                    objective="Reuse the currently productive active skill for one reveal before another mutation.",
                    target_skill_id=active_skill.skill_id,
                    risk_budget="low",
                    required_checks=list(task_plan.required_checks),
                    rationale=(
                        "CARE adaptive public controller lowered mutation intensity after recent public improvement."
                    ),
                )
                return reuse_plan, {
                    "controller_family": "care_certified_adaptive_residual_evolution",
                    "from_action": task_plan.action,
                    "to_action": "reuse_active_skill",
                    "reward_trend": trend,
                    "care_adaptive_state": adaptive_state,
                    "adaevolve_distinction": (
                        "CARE changes only the public-certified planner action prior inside an HTE reveal loop; "
                        "it does not run AdaEvolve-style multi-island program search or replace final selection."
                    ),
                }
            return task_plan, {
                "controller_family": "care_certified_adaptive_residual_evolution",
                "from_action": task_plan.action,
                "to_action": task_plan.action,
                "reward_trend": trend,
                "care_adaptive_state": adaptive_state,
                "no_override_reason": "planner_action_aligned_with_care_prior",
                "adaevolve_distinction": (
                    "CARE is a public-certified residual controller, not a general multi-island code evolution engine."
                ),
            }
        trend = _reward_trend_from_records(self.registry.policy_state.reward_history)
        should_patch = bool(trend["trailing_no_improvement_count"] >= 2 or trend["last_selected_far_below_best"])
        if should_patch and task_plan.action == "reuse_active_skill":
            patched_plan = TaskPlan(
                action="patch_skill",
                skill_family="ranker",
                objective=(
                    "Conservatively patch the active ranker after public reward stagnation. "
                    "Keep the current exploitation rule, add bounded diversity only when it changes a repeatedly stagnant top choice, "
                    "and do not use hidden outcomes."
                ),
                target_skill_id=active_skill.skill_id,
                risk_budget="low",
                required_checks=list(task_plan.required_checks),
                rationale=(
                    "Host-side public controller overrode reuse because revealed reward history shows stagnation or a far-below-best selection."
                ),
            )
            return patched_plan, {"from_action": task_plan.action, "to_action": "patch_skill", "reward_trend": trend}
        if not should_patch and task_plan.action == "patch_skill" and trend["recent_positive_improvement"]:
            reuse_plan = TaskPlan(
                action="reuse_active_skill",
                skill_family=task_plan.skill_family,
                objective="Reuse the recently improving active skill for one more reveal before patching.",
                target_skill_id=active_skill.skill_id,
                risk_budget="low",
                required_checks=list(task_plan.required_checks),
                rationale="Host-side public controller delayed patching after a recent best-observed improvement.",
            )
            return reuse_plan, {"from_action": task_plan.action, "to_action": "reuse_active_skill", "reward_trend": trend}
        return task_plan, None

    def _select_skill_output(
        self,
        *,
        deployed_skill: SkillArtifact,
        active_before_deploy: SkillArtifact | None,
        observed_df: pd.DataFrame,
        candidate_df: pd.DataFrame,
        memory_text: str,
        tool_state: dict[str, Any],
        patch_deployed: bool,
        round_index: int,
    ) -> tuple[SkillArtifact, Any, dict[str, Any]]:
        candidates = self._activated_skill_portfolio(deployed_skill)
        initial_shadow_key = self._first_activated_skill_artifact_key()
        scored: list[dict[str, Any]] = []
        for skill in candidates:
            try:
                source = compile_skill_to_tool(skill)
                raw_output = run_rank_candidates_tool(
                    tool_source=source,
                    observed_df=observed_df,
                    candidate_df=candidate_df,
                    memory=memory_text,
                    tool_state=tool_state,
                )
                parsed = parse_ranked_candidates(raw_output, candidate_df=candidate_df, observed_df=observed_df)
                score_report = _observed_replay_score(
                    source=source,
                    observed_df=observed_df,
                    memory=memory_text,
                    tool_state=tool_state,
                )
                scored.append(
                    {
                        "skill": skill,
                        "parsed": parsed,
                        "artifact_key": skill.artifact_key,
                        "top1_candidate_id": parsed.selected_display_candidate_id,
                        "observed_replay_score": score_report["score"],
                        "observed_replay_report": score_report,
                        "is_deployed_skill": skill.artifact_key == deployed_skill.artifact_key,
                        "is_initial_shadow_skill": skill.artifact_key == initial_shadow_key,
                    }
                )
            except Exception as exc:
                scored.append(
                    {
                        "skill": skill,
                        "parsed": None,
                        "artifact_key": skill.artifact_key,
                        "top1_candidate_id": None,
                        "observed_replay_score": float("-inf"),
                        "observed_replay_report": {"passed": False, "error_type": exc.__class__.__name__, "error": str(exc)[:300]},
                        "is_deployed_skill": skill.artifact_key == deployed_skill.artifact_key,
                        "is_initial_shadow_skill": skill.artifact_key == initial_shadow_key,
                    }
                )
        viable = [row for row in scored if row.get("parsed") is not None]
        if not viable:
            raise RuntimeError("No activated skill in portfolio produced a valid full-pool ranking.")
        active_key = active_before_deploy.artifact_key if active_before_deploy is not None else deployed_skill.artifact_key
        baseline = next((row for row in viable if row["artifact_key"] == active_key), None)
        deployed = next((row for row in viable if row["artifact_key"] == deployed_skill.artifact_key), None)
        selected = deployed or viable[0]
        if patch_deployed and baseline is not None and deployed is not None and baseline["artifact_key"] != deployed["artifact_key"]:
            margin = _portfolio_acceptance_margin(observed_df)
            deployed_score = float(deployed["observed_replay_score"])
            baseline_score = float(baseline["observed_replay_score"])
            if deployed_score + margin < baseline_score:
                selected = baseline
        else:
            selected = sorted(
                viable,
                key=lambda row: (
                    float(row["observed_replay_score"]),
                    row["artifact_key"] == active_key,
                    row["artifact_key"],
                ),
                reverse=True,
            )[0]
        report = {
            "selection_version": "public_observed_replay_portfolio_v1",
            "round_index": int(round_index),
            "patch_deployed": bool(patch_deployed),
            "selected_artifact_key": selected["artifact_key"],
            "deployed_artifact_key": deployed_skill.artifact_key,
            "active_before_selection_artifact_key": active_key,
            "initial_shadow_artifact_key": initial_shadow_key,
            "llm_portfolio_ranked_candidates": _compact_portfolio_ranked_candidates(scored, selected_artifact_key=selected["artifact_key"]),
            "candidate_reports": [
                {
                    "artifact_key": row["artifact_key"],
                    "top1_candidate_id": row.get("top1_candidate_id"),
                    "observed_replay_score": _safe_float(row.get("observed_replay_score")),
                    "is_deployed_skill": bool(row.get("is_deployed_skill")),
                    "is_initial_shadow_skill": bool(row.get("is_initial_shadow_skill")),
                    "observed_replay_report": row.get("observed_replay_report"),
                }
                for row in scored
            ],
        }
        self.ledger.append("portfolio_skill_selected", report)
        return selected["skill"], selected["parsed"], report

    def _activated_skill_portfolio(self, deployed_skill: SkillArtifact) -> list[SkillArtifact]:
        rows = [
            row
            for row in self.registry.policy_state.skill_history
            if row.get("activated") is True and row.get("skill_id") and row.get("version")
        ]
        rows = sorted(
            rows,
            key=lambda row: (
                int(row.get("created_round", 0) or 0),
                int(row.get("version", 0) or 0),
                str(row.get("skill_id", "")),
            ),
        )
        selected_rows = rows[-4:]
        keys = {(deployed_skill.skill_id, int(deployed_skill.version))}
        for row in selected_rows:
            keys.add((str(row["skill_id"]), int(row["version"])))
        if rows:
            first = rows[0]
            keys.add((str(first["skill_id"]), int(first["version"])))
        skills: list[SkillArtifact] = []
        for skill_id, version in sorted(keys, key=lambda item: (item[0], item[1])):
            try:
                skills.append(self.registry.load_skill(skill_id, version))
            except (FileNotFoundError, KeyError, ValueError):
                continue
        if not skills:
            return [deployed_skill]
        unique: dict[str, SkillArtifact] = {}
        for skill in skills:
            unique[skill.artifact_key] = skill
        return list(unique.values())

    def _first_activated_skill_artifact_key(self) -> str | None:
        rows = [
            row
            for row in self.registry.policy_state.skill_history
            if row.get("activated") is True and row.get("skill_id") and row.get("version")
        ]
        if not rows:
            return None
        rows = sorted(
            rows,
            key=lambda row: (
                int(row.get("created_round", 0) or 0),
                int(row.get("version", 0) or 0),
                str(row.get("skill_id", "")),
            ),
        )
        first = rows[0]
        try:
            return self.registry.load_skill(str(first["skill_id"]), int(first["version"])).artifact_key
        except (FileNotFoundError, KeyError, ValueError):
            return None

    def _maybe_rollback_after_reveal(self, *, reward: Any, active_skill: SkillArtifact | None) -> dict[str, Any]:
        if active_skill is None or not bool(reward.patch_deployed):
            return {"rolled_back": False, "reason": "not_patch_round"}
        revealed_y = _safe_float(reward.revealed_y)
        previous_best = _safe_float(reward.previous_best_y)
        if revealed_y is None or previous_best is None:
            return {"rolled_back": False, "reason": "missing_revealed_or_previous_best"}
        shortfall = previous_best - revealed_y if str(self.config.objective_direction).lower() != "minimize" else revealed_y - previous_best
        threshold = _rollback_shortfall_threshold(previous_best)
        if float(reward.delta_best_y) <= 0.0 and shortfall >= threshold:
            previous = self.registry.rollback_to_previous()
            return {
                "rolled_back": previous is not None,
                "reason": "patch_selection_underperformed_revealed_best",
                "revealed_y": revealed_y,
                "previous_best_y": previous_best,
                "shortfall": float(shortfall),
                "threshold": float(threshold),
                "rolled_back_to": previous.artifact_key if previous else None,
                "from_skill": active_skill.artifact_key,
            }
        return {
            "rolled_back": False,
            "reason": "patch_not_bad_enough",
            "revealed_y": revealed_y,
            "previous_best_y": previous_best,
            "shortfall": float(shortfall),
            "threshold": float(threshold),
        }

    def _meta_controller_select_candidate(
        self,
        *,
        parsed: Any,
        observed_df: pd.DataFrame,
        candidate_df: pd.DataFrame,
        round_index: int,
        portfolio_selection: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        public_experts = _build_public_expert_rankings(
            parsed=None,
            observed_df=observed_df,
            candidate_df=candidate_df,
            seed=int(self.config.seed) + int(round_index),
            include_categorical=bool(self.config.adaptive_categorical_experts),
        )
        experts = _build_public_expert_rankings(
            parsed=parsed,
            observed_df=observed_df,
            candidate_df=candidate_df,
            seed=int(self.config.seed) + int(round_index),
            include_categorical=bool(self.config.adaptive_categorical_experts),
        )
        if not experts:
            return {
                "selection_version": "expert_meta_controller_v5_incumbent_challenger",
                "selected_display_candidate_id": parsed.selected_display_candidate_id,
                "selected_by": "active_llm_skill",
                "reason": "no_valid_public_experts",
                "expert_reports": [],
            }
        sparse_category_route = None
        data_profile = _public_data_profile(observed_df=observed_df, candidate_df=candidate_df)
        if bool(data_profile.get("high_cardinality_categorical_space", False)) and self.state is not None:
            sparse_category_route = self.state.strategy_state.get("sparse_category_route")
            if not sparse_category_route:
                sparse_category_route = _choose_sparse_category_route(data_profile)
                self.state.strategy_state["sparse_category_route"] = sparse_category_route
        public_report = _score_expert_candidates(
            experts=public_experts,
            observed_df=observed_df,
            candidate_df=candidate_df,
            round_index=round_index,
            sparse_category_route=sparse_category_route,
        )
        fused_report = _score_expert_candidates(
            experts=experts,
            observed_df=observed_df,
            candidate_df=candidate_df,
            round_index=round_index,
            sparse_category_route=sparse_category_route,
        )
        if bool(self.config.llm_residual_scout_public_locked):
            report = _apply_llm_residual_scout(
                public_report=public_report,
                fused_report=fused_report,
                parsed=parsed,
                observed_df=observed_df,
                candidate_df=candidate_df,
                round_index=round_index,
                config=self.config,
                strategy_state=self.state.strategy_state if self.state is not None else {},
                portfolio_selection=portfolio_selection,
            )
        else:
            report = _apply_llm_residual_scout(
                public_report=public_report,
                fused_report=fused_report,
                parsed=parsed,
                observed_df=observed_df,
                candidate_df=candidate_df,
                round_index=round_index,
                config=self.config,
                strategy_state=self.state.strategy_state if self.state is not None else {},
                base_report=fused_report,
                portfolio_selection=portfolio_selection,
            )
        self.ledger.append("expert_meta_candidate_selected", report)
        if self.state is not None:
            self.state.strategy_state["expert_meta_selection_count"] = int(
                self.state.strategy_state.get("expert_meta_selection_count", 0)
            ) + 1
            scout = report.get("llm_residual_scout", {})
            if bool(scout.get("applied")):
                mode = str(scout.get("mode", ""))
                if mode == "public_locked_macro_frontier_scout":
                    self.state.strategy_state["llm_macro_frontier_scout_selection_count"] = int(
                        self.state.strategy_state.get("llm_macro_frontier_scout_selection_count", 0)
                    ) + 1
                else:
                    self.state.strategy_state["llm_residual_scout_selection_count"] = int(
                        self.state.strategy_state.get("llm_residual_scout_selection_count", 0)
                    ) + 1
        return report

    def _reset_pending_round_state(self) -> None:
        self._pending_gate_report = None
        self._pending_task_plan = None
        self._pending_skill = None
        self._pending_selected_skill = None
        self._pending_tool_state = {}
        self._pending_tool_diagnostics = {}
        self._pending_decision_metadata = {}

    def _gate_report_for_reward(self, decision: SelfEvolvingDecision) -> dict[str, Any]:
        if self._pending_gate_report is not None:
            return self._pending_gate_report.to_dict()
        return {
            "passed": True,
            "deployable": True,
            "checks": {},
            "reason": "fallback_or_reused_active_skill" if decision.fallback_used else "reused_active_skill",
        }

    def _plan_next_task_with_repairs(
        self,
        *,
        run_id: str,
        round_index: int,
        observed_df: pd.DataFrame,
        candidate_df: pd.DataFrame,
        memory_text: str,
        policy_state: dict[str, Any],
        last_gate_report: dict[str, Any] | None,
    ) -> tuple[TaskPlan, dict[str, Any]]:
        parser_error: str | None = None
        last_error: Exception | None = None
        max_attempts = max(1, int(self.config.repair_attempts) + 1)
        for attempt_index in range(1, max_attempts + 1):
            try:
                task_plan, artifacts = plan_next_task(
                    client=self.client,
                    mode=self.config.mode,
                    run_id=run_id,
                    round_index=round_index,
                    observed_df=observed_df,
                    candidate_df=candidate_df,
                    memory_text=memory_text,
                    policy_state=policy_state,
                    last_gate_report=last_gate_report,
                    parser_error=parser_error,
                )
                artifacts = dict(artifacts)
                artifacts["attempt_index"] = attempt_index
                artifacts["max_attempts"] = max_attempts
                return task_plan, artifacts
            except Exception as exc:
                last_error = exc
                parser_error = _public_retry_error(exc)
                self.ledger.append(
                    "task_planner_attempt_failed",
                    {
                        "round_index": round_index,
                        "attempt_index": attempt_index,
                        "max_attempts": max_attempts,
                        "error": parser_error,
                    },
                )
        assert last_error is not None
        raise last_error

    def _synthesize_skill_with_repairs(
        self,
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
        max_attempts_override: int | None = None,
    ) -> tuple[SkillArtifact, dict[str, Any]]:
        last_error: Exception | None = None
        base_attempts = int(max_attempts_override or (int(self.config.repair_attempts) + 1))
        if str(self.config.mode).lower() == "api":
            base_attempts += max(0, int(self.config.api_parse_retry_attempts))
        max_attempts = max(1, int(base_attempts))
        for attempt_index in range(1, max_attempts + 1):
            try:
                skill, artifacts = synthesize_skill(
                    client=self.client,
                    mode=self.config.mode,
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
                artifacts = dict(artifacts)
                artifacts["attempt_index"] = attempt_index
                artifacts["max_attempts"] = max_attempts
                return skill, artifacts
            except Exception as exc:
                last_error = exc
                parser_error = _public_retry_error(exc)
                self.ledger.append(
                    "skill_synthesis_attempt_failed",
                    {
                        "round_index": round_index,
                        "attempt_index": attempt_index,
                        "max_attempts": max_attempts,
                        "error": parser_error,
                    },
                )
        assert last_error is not None
        raise last_error

    def _synthesize_gate_register_with_repairs(
        self,
        *,
        run_id: str,
        round_index: int,
        task_plan: TaskPlan,
        observed_df: pd.DataFrame,
        candidate_df: pd.DataFrame,
        memory_text: str,
        policy_state: dict[str, Any],
        active_skill: SkillArtifact | None,
    ) -> tuple[SkillArtifact, GateReport, dict[str, Any]]:
        parser_error: str | None = None
        last_skill: SkillArtifact | None = None
        last_gate_report: GateReport | None = None
        max_attempts = max(1, int(self.config.repair_attempts) + 1)
        gate_reject_count = 0
        for attempt_index in range(1, max_attempts + 1):
            skill, artifacts = self._synthesize_skill_with_repairs(
                run_id=run_id,
                round_index=round_index,
                task_plan=task_plan,
                observed_df=observed_df,
                candidate_df=candidate_df,
                memory_text=memory_text,
                policy_state=policy_state,
                active_skill=active_skill,
                parser_error=parser_error,
                max_attempts_override=None,
            )
            skill = self.registry.ensure_unique_version(skill)
            if self.state is not None:
                self.state.strategy_state["skill_synthesis_count"] = int(
                    self.state.strategy_state.get("skill_synthesis_count", 0)
                ) + 1
            gate_report = run_conservative_gate(
                candidate_skill=skill,
                old_tool_source=active_skill.source if active_skill else "",
                observed_df=observed_df,
                candidate_df=candidate_df,
                memory=memory_text,
                tool_state=self.state.tool_state if self.state is not None else {},
                round_index=round_index,
            )
            last_skill = skill
            last_gate_report = gate_report
            self.registry.record_gate(gate_report.to_dict())
            self.registry.register_skill(skill, activate=gate_report.deployable, gate_report=gate_report.to_dict())
            merged_artifacts = dict(artifacts)
            merged_artifacts["gate_repair_attempt_index"] = attempt_index
            merged_artifacts["gate_repair_max_attempts"] = max_attempts
            merged_artifacts["gate_reject_count"] = gate_reject_count + int(not gate_report.deployable)
            self.ledger.append(
                "skill_candidate_evaluated",
                {
                    "round_index": round_index,
                    "skill": _skill_metadata(skill),
                    "synthesis_artifacts": merged_artifacts,
                    "gate_report": gate_report.to_dict(),
                },
            )
            if gate_report.deployable:
                return skill, gate_report, merged_artifacts
            gate_reject_count += 1
            parser_error = _public_gate_retry_error(gate_report)
            self.ledger.append(
                "skill_gate_attempt_failed",
                {
                    "round_index": round_index,
                    "attempt_index": attempt_index,
                    "max_attempts": max_attempts,
                    "error": parser_error,
                },
            )
        assert last_skill is not None and last_gate_report is not None
        return last_skill, last_gate_report, {
            "mode": self.config.mode,
            "gate_repair_max_attempts": max_attempts,
            "gate_reject_count": gate_reject_count,
            "final_gate_reason": last_gate_report.reason,
        }


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _selection_attribution_from_decision(metadata: dict[str, Any]) -> dict[str, Any]:
    public_counterfactual = metadata.get("public_only_counterfactual")
    if not isinstance(public_counterfactual, dict):
        public_counterfactual = {}
    scout = metadata.get("llm_residual_scout")
    if not isinstance(scout, dict):
        scout = {}
    care_certificate = metadata.get("care_public_certificate")
    if not isinstance(care_certificate, dict):
        care_certificate = {}
    selected = str(metadata.get("selected_display_candidate_id", ""))
    public_selected = str(public_counterfactual.get("selected_display_candidate_id", ""))
    return {
        "selected_display_candidate_id": selected,
        "public_only_counterfactual_selected_id": public_selected or None,
        "llm_changed_final_selection": bool(public_selected and selected and public_selected != selected),
        "selection_rule": metadata.get("selection_rule"),
        "selected_by": (metadata.get("expert_meta_selection") or {}).get("selected_by")
        if isinstance(metadata.get("expert_meta_selection"), dict)
        else None,
        "llm_residual_scout_applied": bool(scout.get("applied", False)),
        "llm_residual_scout_reason": scout.get("reason"),
        "llm_residual_scout_mode": scout.get("mode"),
        "care_certificate_decision": care_certificate.get("decision"),
        "care_certificate_score": care_certificate.get("score"),
    }


def _public_retry_error(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: previous response failed public-safety or schema validation"


def _public_gate_retry_error(gate_report: GateReport) -> str:
    failed = ",".join(str(item) for item in gate_report.failed_checks)
    violations = []
    checks = gate_report.checks if isinstance(gate_report.checks, dict) else {}
    static = checks.get("static", {})
    if isinstance(static, dict):
        violations.extend(str(item) for item in static.get("violations", []))
    compiler = checks.get("compiler", {})
    compiler_error = str(compiler.get("error", ""))[:200] if isinstance(compiler, dict) else ""
    details = ",".join(violations) if violations else str(gate_report.reason)
    if compiler_error:
        details = f"{details}; {compiler_error}"
    return (
        "Gate rejected the previous skill candidate. "
        f"Failed checks: {failed or 'unknown'}. Public validation details: {details[:500]}. "
        "Repair in the same round without using hidden outcomes."
    )


def _reward_trend_from_records(reward_history: list[dict[str, Any]]) -> dict[str, Any]:
    records = [row for row in reward_history[-5:] if isinstance(row, dict)]
    deltas = [_safe_float(row.get("delta_best_y")) for row in records]
    revealed = [_safe_float(row.get("revealed_y")) for row in records]
    deltas = [float(value) for value in deltas if value is not None]
    revealed = [float(value) for value in revealed if value is not None]
    current_best = _safe_float(records[-1].get("current_best_y")) if records else None
    last_revealed = _safe_float(records[-1].get("revealed_y")) if records else None
    trailing_no_improvement = 0
    for value in reversed(deltas):
        if value > 1e-12:
            break
        trailing_no_improvement += 1
    return {
        "recent_reward_count": len(records),
        "recent_delta_best_y": deltas,
        "recent_revealed_y": revealed,
        "current_best_y": current_best,
        "last_revealed_y": last_revealed,
        "trailing_no_improvement_count": trailing_no_improvement,
        "last_selected_far_below_best": _last_revealed_far_below_best(last_revealed=last_revealed, current_best=current_best),
        "recent_positive_improvement": any(value > 1e-12 for value in deltas[-2:]),
    }


def _care_adaptive_planner_state(
    *,
    reward_history: list[dict[str, Any]],
    strategy_state: dict[str, Any],
) -> dict[str, Any]:
    trend = _reward_trend_from_records(reward_history)
    recent_deltas = [float(value) for value in trend.get("recent_delta_best_y", [])]
    positive = [value for value in recent_deltas if value > 1e-12]
    trailing = int(trend.get("trailing_no_improvement_count", 0) or 0)
    far_below = bool(trend.get("last_selected_far_below_best", False))
    recent_count = int(trend.get("recent_reward_count", 0) or 0)
    productivity = float(sum(positive) / max(1, len(recent_deltas))) if recent_deltas else 0.0
    stagnation_ratio = float(trailing) / float(max(1, recent_count))

    if recent_count == 0:
        mode = "bootstrap"
        action_prior = "defer"
        intensity = 0.35
        rationale = "no revealed reward yet"
    elif bool(trend.get("recent_positive_improvement", False)):
        mode = "exploit_refine"
        action_prior = "reuse_active_skill"
        intensity = 0.20
        rationale = "recent best-observed improvement"
    elif far_below or stagnation_ratio >= 0.60:
        mode = "diverge"
        action_prior = "patch_skill"
        intensity = min(1.0, 0.55 + 0.35 * stagnation_ratio + (0.10 if far_below else 0.0))
        rationale = "public reward stagnation or far-below-best selection"
    else:
        mode = "balanced_refine"
        action_prior = "defer"
        intensity = 0.45
        rationale = "mixed public reward evidence"

    return {
        "mode": mode,
        "action_prior": action_prior,
        "evolution_intensity": float(intensity),
        "productivity": float(productivity),
        "stagnation_ratio": float(stagnation_ratio),
        "trailing_no_improvement_count": int(trailing),
        "last_selected_far_below_best": bool(far_below),
        "recent_reward_count": int(recent_count),
        "recent_positive_improvement": bool(trend.get("recent_positive_improvement", False)),
        "rationale": rationale,
        "controller_scope": "planner_action_prior_only",
        "selection_scope": "final candidate selection remains public meta plus certified scout",
        "llm_residual_budget_used": int(strategy_state.get("llm_residual_scout_selection_count", 0) or 0),
        "llm_macro_budget_used": int(strategy_state.get("llm_macro_frontier_scout_selection_count", 0) or 0),
    }


def _care_patch_objective(*, adaptive_state: dict[str, Any]) -> str:
    mode = str(adaptive_state.get("mode", "balanced_refine"))
    if mode == "diverge":
        return (
            "Patch the active ranker toward a bounded residual-frontier challenger: preserve the current public "
            "incumbent logic, add one public-feature diversity or model-disagreement rule, and expose evidence "
            "that can be certified by public experts before final selection."
        )
    return (
        "Patch the active ranker conservatively using only public observed data; retain productive exploitation "
        "features and add a small certified residual challenger for repeatedly stagnant top choices."
    )


def _care_update_strategy_after_reveal(
    *,
    strategy_state: dict[str, Any],
    decision_metadata: dict[str, Any],
    reward: Any,
) -> None:
    certificate = decision_metadata.get("care_public_certificate")
    if not isinstance(certificate, dict):
        expert = decision_metadata.get("expert_meta_selection")
        if isinstance(expert, dict):
            certificate = expert.get("care_public_certificate")
    if not isinstance(certificate, dict):
        return
    strategy_state["care_certificate_observed_count"] = int(
        strategy_state.get("care_certificate_observed_count", 0) or 0
    ) + 1
    if bool(certificate.get("selected_candidate_certified", False)):
        strategy_state["care_certificate_selected_count"] = int(
            strategy_state.get("care_certificate_selected_count", 0) or 0
        ) + 1
        if float(getattr(reward, "delta_best_y", 0.0) or 0.0) > 0.0:
            strategy_state["care_certificate_improvement_count"] = int(
                strategy_state.get("care_certificate_improvement_count", 0) or 0
            ) + 1
            strategy_state["care_certificate_delta_best_sum"] = float(
                strategy_state.get("care_certificate_delta_best_sum", 0.0) or 0.0
            ) + float(reward.delta_best_y)
    strategy_state["care_last_certificate"] = {
        "candidate_id": certificate.get("candidate_id"),
        "score": certificate.get("score"),
        "decision": certificate.get("decision"),
        "revealed_y": _safe_float(getattr(reward, "revealed_y", None)),
        "delta_best_y": _safe_float(getattr(reward, "delta_best_y", None)),
    }


def _last_revealed_far_below_best(*, last_revealed: float | None, current_best: float | None) -> bool:
    if last_revealed is None or current_best is None:
        return False
    if current_best <= 0:
        return last_revealed < current_best
    return (current_best - last_revealed) >= max(10.0, 0.20 * abs(current_best))


def _observed_replay_score(
    *,
    source: str,
    observed_df: pd.DataFrame,
    memory: str,
    tool_state: dict[str, Any],
) -> dict[str, Any]:
    if observed_df.empty:
        return {"passed": True, "score": 0.0, "candidate_count": 0, "topk_mean_y": None}
    shadow = observed_df.drop(columns=["observation_id", "observed_y"], errors="ignore").copy().reset_index(drop=True)
    shadow_ids = [f"shadow_{str(value)}" for value in observed_df["observation_id"].astype(str).tolist()]
    shadow["candidate_id"] = shadow_ids
    shadow_observed = observed_df.copy().reset_index(drop=True)
    y_by_shadow_id = {
        shadow_id: float(y)
        for shadow_id, y in zip(shadow_ids, pd.to_numeric(shadow_observed["observed_y"], errors="coerce").fillna(0.0))
    }
    try:
        raw_output = run_rank_candidates_tool(
            tool_source=source,
            observed_df=shadow_observed,
            candidate_df=shadow,
            memory=memory,
            tool_state=tool_state,
        )
        parsed = parse_ranked_candidates(raw_output, candidate_df=shadow, observed_df=shadow_observed)
    except Exception as exc:
        return {"passed": False, "score": float("-inf"), "error_type": exc.__class__.__name__, "error": str(exc)[:300]}
    ranked = parsed.ranked_candidates
    if not ranked:
        return {"passed": False, "score": float("-inf"), "error_type": "empty_ranking"}
    k = min(3, len(ranked))
    top = ranked[:k]
    top_y = [float(y_by_shadow_id.get(str(row["candidate_id"]), 0.0)) for row in top]
    best_y = max(y_by_shadow_id.values()) if y_by_shadow_id else 0.0
    top1_y = top_y[0] if top_y else 0.0
    topk_mean = sum(top_y) / max(len(top_y), 1)
    rank_of_best = next(
        (int(row["rank"]) for row in ranked if abs(float(y_by_shadow_id.get(str(row["candidate_id"]), 0.0)) - best_y) <= 1e-12),
        len(ranked) + 1,
    )
    score = 0.70 * top1_y + 0.25 * topk_mean - 0.05 * float(rank_of_best)
    return {
        "passed": True,
        "score": float(score),
        "candidate_count": len(ranked),
        "top1_shadow_candidate_id": parsed.selected_display_candidate_id,
        "top1_observed_y": float(top1_y),
        "topk_mean_y": float(topk_mean),
        "best_observed_y": float(best_y),
        "rank_of_best_observed": int(rank_of_best),
    }


def _compact_portfolio_ranked_candidates(scored: list[dict[str, Any]], *, selected_artifact_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in scored:
        parsed = item.get("parsed")
        ranked = list(getattr(parsed, "ranked_candidates", []) or [])
        if not ranked:
            continue
        artifact_key = str(item.get("artifact_key", ""))
        for rank_index, candidate in enumerate(ranked[:12], start=1):
            cid = str(candidate.get("candidate_id", ""))
            if not cid:
                continue
            rows.append(
                {
                    "artifact_key": artifact_key,
                    "candidate_id": cid,
                    "portfolio_rank": int(candidate.get("rank", rank_index) or rank_index),
                    "score": _safe_float(candidate.get("score")) or 0.0,
                    "observed_replay_score": _safe_float(item.get("observed_replay_score")),
                    "is_selected_skill": artifact_key == str(selected_artifact_key),
                    "is_deployed_skill": bool(item.get("is_deployed_skill")),
                    "is_initial_shadow_skill": bool(item.get("is_initial_shadow_skill")),
                }
            )
    return rows


def _build_public_expert_rankings(
    *,
    parsed: Any,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    seed: int,
    include_categorical: bool = True,
) -> list[dict[str, Any]]:
    experts: list[dict[str, Any]] = []
    active_rows = list(getattr(parsed, "ranked_candidates", []))
    if active_rows:
        experts.append(_expert_from_ranked_rows("active_llm_skill", active_rows, max_rows=40))
    fixed_rows = _run_source_ranker_rows(
        source=_fixed_public_source(),
        observed_df=observed_df,
        candidate_df=candidate_df,
        name="fixed_public_heuristic",
    )
    if fixed_rows:
        experts.append(_expert_from_ranked_rows("fixed_public_heuristic", fixed_rows, max_rows=40))
    gp_rows = _public_gp_ei_rank(
        observed_df=observed_df,
        candidate_df=candidate_df,
        objective_direction="maximize",
        seed=seed,
        top_k=40,
    )
    if gp_rows:
        experts.append({"name": "classical_gp_ei", "ranked": gp_rows})
    rf_rows = _public_rf_ucb_rank(observed_df=observed_df, candidate_df=candidate_df, seed=seed, top_k=40)
    if rf_rows:
        experts.append({"name": "rf_ucb_surrogate", "ranked": rf_rows})
    if include_categorical:
        categorical_rows = _public_categorical_shrinkage_rank(
            observed_df=observed_df,
            candidate_df=candidate_df,
            top_k=40,
        )
        if categorical_rows:
            experts.append({"name": "categorical_shrinkage", "ranked": categorical_rows})
        profile = _public_data_profile(observed_df=observed_df, candidate_df=candidate_df)
        if profile["high_cardinality_categorical_space"]:
            categorical_ucb_rows = _public_categorical_eb_ucb_rank(
                observed_df=observed_df,
                candidate_df=candidate_df,
                seed=int(seed),
                top_k=40,
            )
            if categorical_ucb_rows:
                experts.append({"name": "categorical_eb_ucb", "ranked": categorical_ucb_rows})
    return experts


def _score_expert_candidates(
    *,
    experts: list[dict[str, Any]],
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    round_index: int,
    sparse_category_route: str | None = None,
) -> dict[str, Any]:
    candidate_ids = set(str(value) for value in candidate_df["candidate_id"].astype(str).tolist())
    public_prior = _public_domain_prior_scores(candidate_df)
    density = _candidate_density_scores(candidate_df)
    novelty = _candidate_novelty_scores(observed_df=observed_df, candidate_df=candidate_df)
    data_profile = _public_data_profile(observed_df=observed_df, candidate_df=candidate_df)
    category_evidence = _category_evidence_scores(observed_df=observed_df, candidate_df=candidate_df, profile=data_profile)
    observed_count = int(len(observed_df))
    schedule = _expert_meta_schedule(observed_count=observed_count, round_index=round_index, data_profile=data_profile)
    if sparse_category_route:
        schedule = _apply_sparse_category_route(schedule, sparse_category_route)
    expert_weights = _expert_weights_from_observed_replay(experts=experts, observed_df=observed_df)
    scores: dict[str, float] = {}
    support: dict[str, list[str]] = {}
    rank_details: dict[str, dict[str, Any]] = {}
    for expert in experts:
        name = str(expert.get("name", "expert"))
        weight = float(expert_weights.get(name, 1.0)) * float(schedule["expert_multipliers"].get(name, 1.0))
        ranked = [row for row in expert.get("ranked", []) if str(row.get("candidate_id")) in candidate_ids]
        rank_details[name] = {
            "weight": weight,
            "observed_replay_weight": float(expert_weights.get(name, 1.0)),
            "schedule_multiplier": float(schedule["expert_multipliers"].get(name, 1.0)),
            "top_candidates": [str(row.get("candidate_id")) for row in ranked[:12]],
            "top_candidate_count": int(len(ranked)),
        }
        for row in ranked[:40]:
            cid = str(row["candidate_id"])
            rank = max(1, int(row.get("rank", 1) or 1))
            contribution = weight / math.sqrt(float(rank))
            scores[cid] = scores.get(cid, 0.0) + contribution
            support.setdefault(cid, []).append(name)
    for cid in list(scores):
        scores[cid] += float(schedule["prior_weight"]) * float(public_prior.get(cid, 0.0))
        scores[cid] += float(schedule["density_weight"]) * float(density.get(cid, 0.0))
        scores[cid] += float(schedule["novelty_weight"]) * float(novelty.get(cid, 0.0))
        scores[cid] += float(schedule.get("category_evidence_weight", 0.0)) * float(category_evidence.get(cid, 0.0))
    if not scores:
        fallback_id = str(candidate_df["candidate_id"].astype(str).iloc[0])
        scores[fallback_id] = 0.0
        support[fallback_id] = ["fallback_first_public_candidate"]
    initial_selected_id = sorted(scores, key=lambda cid: (-float(scores[cid]), cid))[0]
    reward_signal = _observed_reward_signal(observed_df)
    override_report = _apply_incumbent_challenger_selection(
        scores=scores,
        support=support,
        rank_details=rank_details,
        public_prior=public_prior,
        density=density,
        category_evidence=category_evidence,
        novelty=novelty,
        schedule=schedule,
        candidate_df=candidate_df,
        data_profile=data_profile,
        sparse_category_route=sparse_category_route,
        reward_signal=reward_signal,
        selected_id=initial_selected_id,
    )
    selected_id = str(override_report["selected_display_candidate_id"])
    ranked_meta = sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))[:12]
    selected_rank = next(
        (rank for rank, (cid, _score) in enumerate(sorted(scores.items(), key=lambda item: (-float(item[1]), item[0])), start=1) if cid == selected_id),
        1,
    )
    return {
        "selection_version": "expert_meta_controller_v5_incumbent_challenger",
        "round_index": int(round_index),
        "selected_display_candidate_id": selected_id,
        "selected_by": "public_expert_meta_controller",
        "selected_meta_score": float(scores[selected_id]),
        "selected_rank": int(selected_rank),
        "supporting_experts": sorted(set(support.get(selected_id, []))),
        "expert_weights": expert_weights,
        "schedule": schedule,
        "data_profile": data_profile,
        "sparse_category_route": sparse_category_route,
        "reward_signal": reward_signal,
        "incumbent_challenger_selection": override_report,
        "surrogate_exploration_override": override_report,
        "expert_reports": rank_details,
        "top_meta_candidates": [
            {"candidate_id": cid, "meta_score": float(score), "supporting_experts": sorted(set(support.get(cid, [])))}
            for cid, score in ranked_meta
        ],
        "public_prior_version": "public_feature_prior_v2",
    }


def _apply_llm_residual_scout(
    *,
    public_report: dict[str, Any],
    fused_report: dict[str, Any],
    parsed: Any,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    round_index: int,
    config: SelfEvolvingConfig,
    strategy_state: dict[str, Any],
    base_report: dict[str, Any] | None = None,
    portfolio_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = dict(base_report or public_report)
    public_counterfactual = _compact_public_counterfactual(public_report)
    base["public_only_counterfactual"] = public_counterfactual
    if not bool(config.llm_residual_scout_enabled):
        base["llm_residual_scout"] = {"applied": False, "reason": "disabled"}
        return base

    macro_candidate = _llm_macro_frontier_scout_candidate(
        public_report=public_report,
        observed_df=observed_df,
        candidate_df=candidate_df,
        round_index=round_index,
        config=config,
        strategy_state=strategy_state,
    )
    if macro_candidate is not None:
        macro_candidate, care_certificate = _care_certify_scout_candidate(
            candidate=macro_candidate,
            public_report=public_report,
            fused_report=fused_report,
            observed_df=observed_df,
            candidate_df=candidate_df,
            config=config,
            scout_mode="public_locked_macro_frontier_scout",
        )
        if care_certificate is not None:
            base["care_public_certificate"] = care_certificate
        if macro_candidate is None:
            base["llm_residual_scout"] = {
                "applied": False,
                "reason": "care_certificate_rejected_macro_frontier_candidate",
                "public_only_selected_display_candidate_id": public_counterfactual.get("selected_display_candidate_id"),
                "fused_selected_display_candidate_id": fused_report.get("selected_display_candidate_id"),
            }
        else:
            return _apply_llm_scout_candidate_to_report(
                base=base,
                candidate=macro_candidate,
                public_counterfactual=public_counterfactual,
                fused_report=fused_report,
                selection_version="expert_meta_controller_v6_public_locked_llm_macro_frontier_scout",
                selected_by="llm_macro_frontier_scout_public_safe",
                mode="public_locked_macro_frontier_scout",
            )

    candidate = _llm_residual_scout_candidate(
        public_report=public_report,
        fused_report=fused_report,
        parsed=parsed,
        observed_df=observed_df,
        candidate_df=candidate_df,
        round_index=round_index,
        config=config,
        strategy_state=strategy_state,
        portfolio_selection=portfolio_selection,
    )
    if candidate is not None:
        candidate, care_certificate = _care_certify_scout_candidate(
            candidate=candidate,
            public_report=public_report,
            fused_report=fused_report,
            observed_df=observed_df,
            candidate_df=candidate_df,
            config=config,
            scout_mode="public_locked_residual_scout",
        )
        if care_certificate is not None:
            base["care_public_certificate"] = care_certificate
    if candidate is None:
        if "llm_residual_scout" in base and base["llm_residual_scout"].get("reason") == "care_certificate_rejected_macro_frontier_candidate":
            return base
        base["llm_residual_scout"] = {
            "applied": False,
            "reason": (
                "care_certificate_rejected_residual_candidate"
                if isinstance(base.get("care_public_certificate"), dict)
                and base["care_public_certificate"].get("decision") == "reject"
                else "no_public_certified_llm_residual_candidate"
            ),
            "public_only_selected_display_candidate_id": public_counterfactual.get("selected_display_candidate_id"),
            "fused_selected_display_candidate_id": fused_report.get("selected_display_candidate_id"),
        }
        return base

    return _apply_llm_scout_candidate_to_report(
        base=base,
        candidate=candidate,
        public_counterfactual=public_counterfactual,
        fused_report=fused_report,
        selection_version="expert_meta_controller_v6_public_locked_llm_residual_scout",
        selected_by="llm_residual_scout_public_certified",
        mode="public_locked_residual_scout",
    )


def _care_certify_scout_candidate(
    *,
    candidate: dict[str, Any],
    public_report: dict[str, Any],
    fused_report: dict[str, Any],
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    config: SelfEvolvingConfig,
    scout_mode: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not bool(config.care_enabled):
        return candidate, None
    mode = str(config.care_certificate_mode or "off").lower()
    if mode == "off":
        return candidate, None

    try:
        certificate = _care_public_intervention_certificate(
            candidate=candidate,
            public_report=public_report,
            fused_report=fused_report,
            observed_df=observed_df,
            candidate_df=candidate_df,
            scout_mode=scout_mode,
        )
    except Exception as exc:
        certificate = {
            "candidate_id": str(candidate.get("candidate_id", "")),
            "scout_mode": str(scout_mode),
            "score": -1.0e9,
            "public_gain": 0.0,
            "public_risk": 1.0,
            "mode": mode,
            "decision": "error",
            "decision_margin": float(config.care_certificate_margin),
            "selected_candidate_certified": False,
            "controller_family": "care_public_evidence_certificate",
            "error_type": exc.__class__.__name__,
            "evidence_scope": "public_observed_rows_and_public_candidate_features_only",
        }
        if mode == "log_only":
            patched_candidate = dict(candidate)
            patched_candidate["care_public_certificate"] = certificate
            return patched_candidate, certificate
        return None, certificate
    margin = float(config.care_certificate_margin)
    selected_certified = bool(certificate["score"] >= margin)
    decision = "accept" if selected_certified else "reject"
    certificate.update(
        {
            "mode": mode,
            "decision": decision,
            "decision_margin": float(margin),
            "selected_candidate_certified": selected_certified,
            "controller_family": "care_public_evidence_certificate",
            "adaevolve_distinction": (
                "This certificate is computed from public HTE evidence for a single residual intervention; "
                "it is not AdaEvolve's population-level program selection objective."
            ),
        }
    )
    patched_candidate = dict(candidate)
    patched_candidate["care_public_certificate"] = certificate
    if mode == "log_only":
        return patched_candidate, certificate
    patched_candidate["certificate_score"] = max(
        float(_safe_float(patched_candidate.get("certificate_score")) or 0.0),
        float(certificate.get("legacy_compatible_score", 0.0)),
    )
    if mode == "calibrated_scout" and not selected_certified:
        return None, certificate
    return patched_candidate, certificate


def _care_public_intervention_certificate(
    *,
    candidate: dict[str, Any],
    public_report: dict[str, Any],
    fused_report: dict[str, Any],
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    scout_mode: str,
) -> dict[str, Any]:
    cid = str(candidate.get("candidate_id", ""))
    public_selected = str(public_report.get("selected_display_candidate_id", ""))
    public_top = [
        row for row in list(public_report.get("top_meta_candidates", [])) if isinstance(row, dict)
    ]
    fused_top = [
        row for row in list(fused_report.get("top_meta_candidates", [])) if isinstance(row, dict)
    ]
    public_meta = {
        str(row.get("candidate_id")): float(row.get("meta_score", 0.0))
        for row in public_top
        if row.get("candidate_id") is not None
    }
    fused_meta = {
        str(row.get("candidate_id")): float(row.get("meta_score", 0.0))
        for row in fused_top
        if row.get("candidate_id") is not None
    }
    public_scores = list(public_meta.values()) or [0.0]
    score_scale = _robust_score_scale(public_scores)
    public_selected_score = (
        _safe_float(public_meta.get(public_selected))
        if public_selected in public_meta
        else _safe_float(public_report.get("selected_meta_score"))
    )
    public_selected_score = float(public_selected_score or 0.0)
    candidate_public_score = float(public_meta.get(cid, 0.0))
    candidate_fused_score = float(fused_meta.get(cid, candidate_public_score))
    public_rank = _care_public_rank(candidate=candidate, cid=cid, public_top=public_top)
    support = set(str(item) for item in candidate.get("supporting_experts", []) or [])
    for row in public_top:
        if str(row.get("candidate_id", "")) == cid:
            support.update(str(item) for item in row.get("supporting_experts", []) or [])
    internal_support = {"active_llm_skill", "llm_macro_frontier_prior"}
    public_support = support - internal_support
    support_count = int(max(int(candidate.get("public_support_count", 0) or 0), len(public_support)))
    max_support = max(
        [len(set(str(item) for item in row.get("supporting_experts", []) or [])) for row in public_top] + [1]
    )
    public_rank_bonus = 1.0 / math.sqrt(float(max(1, public_rank))) if public_rank < 10**6 else 0.0
    support_norm = min(1.0, float(support_count) / float(max(1, max_support)))
    meta_gain = (candidate_public_score - public_selected_score) / score_scale
    fused_gain = (candidate_fused_score - public_selected_score) / score_scale
    legacy_score = float(_safe_float(candidate.get("certificate_score")) or 0.0)
    legacy_norm = math.tanh(max(0.0, legacy_score) / max(1.0, score_scale))
    novelty = _safe_float(candidate.get("public_novelty"))
    if novelty is None:
        novelty = _candidate_novelty_scores(observed_df=observed_df, candidate_df=candidate_df).get(cid, 0.0)
    category_evidence = _safe_float(candidate.get("category_evidence")) or 0.0
    macro_feature_evidence = _care_macro_feature_evidence(candidate)
    public_gain = (
        0.38 * legacy_norm
        + 0.24 * max(0.0, meta_gain)
        + 0.18 * max(0.0, fused_gain)
        + 0.14 * support_norm
        + 0.10 * public_rank_bonus
        + 0.08 * float(novelty)
        + 0.06 * float(category_evidence)
        + 0.14 * float(macro_feature_evidence)
    )

    missing_public_rank = public_rank >= 10**6
    low_support = support_count <= 0 and float(macro_feature_evidence) <= 0.0
    active_only = bool(support and not public_support)
    anchor_risk = _care_public_anchor_risk(public_report=public_report, cid=cid, category_evidence=float(category_evidence))
    public_risk = (
        (0.24 if missing_public_rank else 0.0)
        + (0.20 if low_support else 0.0)
        + (0.10 if active_only and str(scout_mode) != "public_locked_macro_frontier_scout" else 0.0)
        + 0.28 * float(anchor_risk)
        + 0.08 * max(0.0, -meta_gain)
    )
    score = float(public_gain - public_risk)
    legacy_compatible_score = float(legacy_score + max(0.0, score))
    return {
        "candidate_id": cid,
        "scout_mode": str(scout_mode),
        "score": score,
        "public_gain": float(public_gain),
        "public_risk": float(public_risk),
        "legacy_compatible_score": legacy_compatible_score,
        "components": {
            "legacy_certificate_norm": float(legacy_norm),
            "public_meta_gain": float(meta_gain),
            "fused_meta_gain": float(fused_gain),
            "support_norm": float(support_norm),
            "public_rank_bonus": float(public_rank_bonus),
            "novelty": float(novelty),
            "category_evidence": float(category_evidence),
            "macro_feature_evidence": float(macro_feature_evidence),
            "anchor_risk": float(anchor_risk),
        },
        "public_rank": None if public_rank >= 10**6 else int(public_rank),
        "public_support_count": int(support_count),
        "public_selected_display_candidate_id": public_selected,
        "fused_selected_display_candidate_id": fused_report.get("selected_display_candidate_id"),
        "evidence_scope": "public_observed_rows_and_public_candidate_features_only",
    }


def _care_public_rank(*, candidate: dict[str, Any], cid: str, public_top: list[dict[str, Any]]) -> int:
    rank = candidate.get("public_rank")
    if rank is not None:
        try:
            return int(rank)
        except (TypeError, ValueError):
            pass
    for idx, row in enumerate(public_top, start=1):
        if str(row.get("candidate_id", "")) == cid:
            return int(idx)
    return 10**6


def _care_macro_feature_evidence(candidate: dict[str, Any]) -> float:
    features = candidate.get("macro_frontier_features", {})
    if not isinstance(features, dict):
        return 0.0
    l3 = _safe_float(features.get("l3")) or 0.0
    temperature = _safe_float(features.get("temperature_norm")) or 0.0
    loading = _safe_float(features.get("catalyst_loading_norm")) or 0.0
    residence = _safe_float(features.get("res_time_norm")) or 0.0
    public_frontier = 0.45 * float(l3) + 0.25 * float(temperature) + 0.25 * float(loading)
    residence_shape = 0.05 * (1.0 - abs(float(residence) - 0.10))
    return float(max(0.0, min(1.0, public_frontier + residence_shape)))


def _care_public_anchor_risk(*, public_report: dict[str, Any], cid: str, category_evidence: float) -> float:
    route = public_report.get("incumbent_challenger_selection", {})
    if not isinstance(route, dict):
        return 0.0
    selected = str(route.get("selected_display_candidate_id", public_report.get("selected_display_candidate_id", "")) or "")
    if not selected or selected == str(cid):
        return 0.0
    candidate = route.get("candidate", {})
    if not isinstance(candidate, dict):
        return 0.0
    selected_category = _safe_float(candidate.get("category_evidence")) or 0.0
    if selected_category < 0.75:
        return 0.0
    return float(max(0.0, min(1.0, selected_category - float(category_evidence))))


def _robust_score_scale(values: list[float]) -> float:
    if not values:
        return 1.0
    arr = np.asarray([float(value) for value in values], dtype=float)
    if arr.size < 2:
        return max(1.0, abs(float(arr[0])))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(q75 - q25)
    return max(1.0, iqr, float(np.nanstd(arr)))


def _apply_llm_scout_candidate_to_report(
    *,
    base: dict[str, Any],
    candidate: dict[str, Any],
    public_counterfactual: dict[str, Any],
    fused_report: dict[str, Any],
    selection_version: str,
    selected_by: str,
    mode: str,
) -> dict[str, Any]:
    selected_id = str(candidate["candidate_id"])
    scores = {
        str(row.get("candidate_id")): float(row.get("meta_score", 0.0))
        for row in list(base.get("top_meta_candidates", []))
        if row.get("candidate_id") is not None
    }
    scores[selected_id] = max(float(candidate.get("certificate_score", 0.0)), scores.get(selected_id, float("-inf")))
    base.update(
        {
            "selection_version": str(selection_version),
            "selected_display_candidate_id": selected_id,
            "selected_by": str(selected_by),
            "selected_meta_score": float(scores[selected_id]),
            "selected_rank": int(candidate.get("llm_rank", 1)),
            "supporting_experts": sorted(set(candidate.get("supporting_experts", []))),
            "llm_residual_scout": {
                "applied": True,
                "mode": str(mode),
                "reason": candidate.get("reason"),
                "candidate": candidate,
                "public_only_selected_display_candidate_id": public_counterfactual.get("selected_display_candidate_id"),
                "fused_selected_display_candidate_id": fused_report.get("selected_display_candidate_id"),
            },
        }
    )
    top_meta = list(base.get("top_meta_candidates", []))
    top_meta = [
        row for row in top_meta if str(row.get("candidate_id")) != selected_id
    ]
    top_meta.insert(
        0,
        {
            "candidate_id": selected_id,
            "meta_score": float(scores[selected_id]),
            "supporting_experts": sorted(set(candidate.get("supporting_experts", []))),
            "llm_residual_scout": True,
        },
    )
    base["top_meta_candidates"] = top_meta[:12]
    return base


def _compact_public_counterfactual(public_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_version": public_report.get("selection_version"),
        "selected_display_candidate_id": public_report.get("selected_display_candidate_id"),
        "selected_by": public_report.get("selected_by"),
        "selected_meta_score": public_report.get("selected_meta_score"),
        "selected_rank": public_report.get("selected_rank"),
        "supporting_experts": public_report.get("supporting_experts", []),
        "incumbent_challenger_selection": public_report.get("incumbent_challenger_selection"),
        "top_meta_candidates": public_report.get("top_meta_candidates", [])[:8],
    }


def _llm_macro_frontier_scout_candidate(
    *,
    public_report: dict[str, Any],
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    round_index: int,
    config: SelfEvolvingConfig,
    strategy_state: dict[str, Any],
) -> dict[str, Any] | None:
    if not bool(config.llm_macro_frontier_scout_enabled):
        return None
    if int(round_index) < int(config.llm_macro_frontier_scout_min_round):
        return None
    used = int(strategy_state.get("llm_macro_frontier_scout_selection_count", 0) or 0)
    if used >= int(config.llm_macro_frontier_scout_budget):
        return None

    reward_signal = public_report.get("reward_signal", {})
    if not isinstance(reward_signal, dict):
        reward_signal = _observed_reward_signal(observed_df)
    current_best = _safe_float(reward_signal.get("current_best_y"))
    if current_best is None:
        return None
    if current_best < float(config.llm_macro_frontier_scout_min_best_threshold):
        return None
    if current_best >= float(config.llm_macro_frontier_scout_high_confidence_best_threshold):
        return None
    if current_best > float(config.llm_macro_frontier_scout_low_best_threshold) and not bool(
        reward_signal.get("last_selected_far_below_best", False)
    ):
        return None

    public_selected = str(public_report.get("selected_display_candidate_id", ""))
    if not public_selected:
        return None
    required_columns = {"L3", "temperature", "catalyst_loading", "res_time", "candidate_id"}
    if not required_columns.issubset(set(candidate_df.columns)):
        return None
    if not required_columns.issubset(set(observed_df.columns) | {"candidate_id"}):
        return None

    observed_l3 = pd.to_numeric(observed_df.get("L3", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    observed_y = pd.to_numeric(observed_df.get("observed_y", pd.Series(dtype=float)), errors="coerce")
    if bool((observed_l3 >= 0.5).any()):
        l3_best = _safe_float(observed_y[observed_l3 >= 0.5].max())
        if l3_best is not None and l3_best >= float(config.llm_macro_frontier_scout_high_confidence_best_threshold):
            return None

    rows = _macro_frontier_candidate_rows(candidate_df=candidate_df, exclude_ids={public_selected})
    if not rows:
        return None
    selected = rows[0]
    cid = str(selected["candidate_id"])
    if cid == public_selected:
        return None

    top_meta_ids = [
        str(row.get("candidate_id"))
        for row in list(public_report.get("top_meta_candidates", []))[:12]
        if row.get("candidate_id") is not None
    ]
    return {
        "candidate_id": cid,
        "llm_rank": 1,
        "llm_score": float(selected["macro_frontier_score"]),
        "public_rank": None,
        "public_support_count": 0,
        "supporting_experts": ["active_llm_skill", "llm_macro_frontier_prior"],
        "certificate_score": float(selected["macro_frontier_score"]),
        "reason": "low_best_public_safe_macro_frontier",
        "macro_frontier_features": {
            "temperature_norm": float(selected["temperature_norm"]),
            "catalyst_loading_norm": float(selected["catalyst_loading_norm"]),
            "res_time_norm": float(selected["res_time_norm"]),
            "l3": float(selected["l3"]),
        },
        "public_only_top_meta_contains_candidate": bool(cid in set(top_meta_ids)),
        "trigger": {
            "current_best_y": float(current_best),
            "observed_l3_count": int((observed_l3 >= 0.5).sum()) if len(observed_l3) else 0,
            "public_selected_display_candidate_id": public_selected,
            "trailing_no_improvement_count": int(reward_signal.get("trailing_no_improvement_count", 0) or 0),
            "last_selected_far_below_best": bool(reward_signal.get("last_selected_far_below_best", False)),
        },
    }


def _macro_frontier_candidate_rows(
    *,
    candidate_df: pd.DataFrame,
    exclude_ids: set[str],
) -> list[dict[str, Any]]:
    temp = pd.to_numeric(candidate_df["temperature"], errors="coerce")
    loading = pd.to_numeric(candidate_df["catalyst_loading"], errors="coerce")
    res_time = pd.to_numeric(candidate_df["res_time"], errors="coerce")
    l3 = pd.to_numeric(candidate_df["L3"], errors="coerce").fillna(0.0)
    temp_norm = _series_minmax_norm(temp)
    loading_norm = _series_minmax_norm(loading)
    res_time_norm = _series_minmax_norm(res_time)
    rows: list[dict[str, Any]] = []
    for idx, row in candidate_df.reset_index(drop=True).iterrows():
        cid = str(row.get("candidate_id", ""))
        if not cid or cid in exclude_ids:
            continue
        l3_value = float(l3.iloc[idx])
        if l3_value < 0.5:
            continue
        score = (
            3.00 * l3_value
            + 1.25 * float(temp_norm.iloc[idx])
            + 1.00 * float(loading_norm.iloc[idx])
            + 0.18 * (1.0 - abs(float(res_time_norm.iloc[idx]) - 0.10))
        )
        rows.append(
            {
                "candidate_id": cid,
                "macro_frontier_score": float(score),
                "temperature_norm": float(temp_norm.iloc[idx]),
                "catalyst_loading_norm": float(loading_norm.iloc[idx]),
                "res_time_norm": float(res_time_norm.iloc[idx]),
                "l3": l3_value,
            }
        )
    return sorted(rows, key=lambda item: (-float(item["macro_frontier_score"]), str(item["candidate_id"])))


def _series_minmax_norm(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    low = float(values.min()) if len(values) else 0.0
    high = float(values.max()) if len(values) else 1.0
    span = max(high - low, 1e-12)
    return ((values - low) / span).clip(lower=0.0, upper=1.0)


def _llm_residual_scout_candidate(
    *,
    public_report: dict[str, Any],
    fused_report: dict[str, Any],
    parsed: Any,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    round_index: int,
    config: SelfEvolvingConfig,
    strategy_state: dict[str, Any],
    portfolio_selection: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    generic_round_allowed = int(round_index) >= int(config.llm_residual_scout_min_round)
    chemlex_model_band_round_allowed = (
        bool(config.llm_residual_scout_chemlex_model_band_enabled)
        and int(round_index) >= int(config.llm_residual_scout_chemlex_model_band_min_round)
    )
    if not generic_round_allowed and not chemlex_model_band_round_allowed:
        return None
    used = int(strategy_state.get("llm_residual_scout_selection_count", 0) or 0)
    if used >= int(config.llm_residual_scout_budget):
        return None
    ranked = list(getattr(parsed, "ranked_candidates", []))
    if not ranked:
        return None
    portfolio_rows = _portfolio_residual_rows(portfolio_selection)
    public_selected = str(public_report.get("selected_display_candidate_id", ""))
    fused_selected = str(fused_report.get("selected_display_candidate_id", ""))
    if not public_selected:
        return None
    reward_signal = fused_report.get("reward_signal", {})
    if not isinstance(reward_signal, dict):
        reward_signal = _observed_reward_signal(observed_df)
    current_best = _safe_float(reward_signal.get("current_best_y"))
    trailing = int(reward_signal.get("trailing_no_improvement_count", 0) or 0)
    far_below = bool(reward_signal.get("last_selected_far_below_best", False))
    public_top = [str(row.get("candidate_id")) for row in public_report.get("top_meta_candidates", []) if row.get("candidate_id")]
    fused_top = [str(row.get("candidate_id")) for row in fused_report.get("top_meta_candidates", []) if row.get("candidate_id")]
    public_top_set = set(public_top[: max(8, int(config.llm_residual_scout_top_k))])
    fused_top_set = set(fused_top[: max(8, int(config.llm_residual_scout_top_k))])
    public_reports = public_report.get("expert_reports", {})
    if not isinstance(public_reports, dict):
        public_reports = {}
    public_rank_by_id, public_support_by_id = _public_rank_support_maps(public_reports)
    top_meta_rank_by_id, top_meta_support_by_id = _top_meta_rank_support_maps(public_report)
    data_profile = _public_data_profile(observed_df=observed_df, candidate_df=candidate_df)
    high_categorical = bool(data_profile.get("high_cardinality_categorical_space", False))
    category_evidence = (
        _category_evidence_scores(observed_df=observed_df, candidate_df=candidate_df, profile=data_profile)
        if high_categorical
        else {}
    )
    route_override = public_report.get("incumbent_challenger_selection", {})
    if not isinstance(route_override, dict):
        route_override = {}
    route_initial = str(route_override.get("initial_selected_display_candidate_id", "") or "")
    route_mode = str(route_override.get("mode", "") or "")
    route_candidate = route_override.get("candidate", {})
    if not isinstance(route_candidate, dict):
        route_candidate = {}
    route_selected_category_evidence = _safe_float(route_candidate.get("category_evidence"))
    if route_selected_category_evidence is None:
        route_selected_category_evidence = float(category_evidence.get(public_selected, 0.0)) if high_categorical else 0.0
    public_meta_scores = {
        str(row.get("candidate_id")): float(row.get("meta_score", 0.0))
        for row in list(public_report.get("top_meta_candidates", []))
        if row.get("candidate_id") is not None
    }
    public_meta_scale = max([abs(float(value)) for value in public_meta_scores.values()] + [1e-12])
    novelty = _candidate_novelty_scores(observed_df=observed_df, candidate_df=candidate_df)
    density = _candidate_density_scores(candidate_df)
    prior = _public_domain_prior_scores(candidate_df)
    low_best = current_best is not None and current_best < float(config.llm_residual_scout_best_threshold)
    stagnant_under_target = (
        current_best is not None
        and current_best < float(config.llm_residual_scout_stagnation_best_threshold)
        and trailing >= 2
    )
    public_disagreement = _public_expert_top1_disagreement(public_reports)
    trigger = bool(low_best or far_below or (stagnant_under_target and public_disagreement >= 0.72))
    if not trigger:
        return None

    if high_categorical and bool(config.llm_residual_scout_public_locked):
        if _chemlex_high_confidence_public_anchor_guard(
            public_selected=public_selected,
            public_rank=int(public_report.get("selected_rank", 10**6) or 10**6),
            route_initial=route_initial,
            route_mode=route_mode,
            route_selected_category_evidence=route_selected_category_evidence,
            public_selected_support=set(public_report.get("supporting_experts", []))
            | set(top_meta_support_by_id.get(public_selected, set())),
            threshold=float(config.llm_residual_scout_chemlex_anchor_guard_threshold),
        ):
            return None
        model_band_candidate = _chemlex_model_residual_band_candidate(
            public_report=public_report,
            fused_report=fused_report,
            ranked=ranked,
            portfolio_selection=portfolio_selection,
            candidate_df=candidate_df,
            round_index=round_index,
            used_residual_count=used,
            config=config,
            public_selected=public_selected,
            current_best=current_best,
            trailing=trailing,
            far_below=far_below,
            public_disagreement=public_disagreement,
            route_selected_category_evidence=route_selected_category_evidence,
            public_rank_by_id=public_rank_by_id,
            public_support_by_id=public_support_by_id,
            top_meta_rank_by_id=top_meta_rank_by_id,
            top_meta_support_by_id=top_meta_support_by_id,
            category_evidence=category_evidence,
            public_meta_scores=public_meta_scores,
            public_meta_scale=public_meta_scale,
            novelty=novelty,
            density=density,
            prior=prior,
        )
        if model_band_candidate is not None:
            return model_band_candidate
        if _chemlex_sparse_categorical_trunk_guard(
            current_best=current_best,
            trailing=trailing,
            route_mode=route_mode,
            public_selected_support=set(public_report.get("supporting_experts", []))
            | set(top_meta_support_by_id.get(public_selected, set())),
        ):
            return None

    if not generic_round_allowed:
        return None

    candidates: list[dict[str, Any]] = []
    scan_limit = max(1, int(config.llm_residual_scout_top_k))
    if high_categorical and bool(config.llm_residual_scout_public_locked):
        scan_limit = max(scan_limit, 12)
    candidate_id_set = set(str(value) for value in candidate_df["candidate_id"].astype(str).tolist())
    residual_rows = _merged_residual_candidate_rows(
        active_ranked=ranked,
        portfolio_rows=portfolio_rows,
        scan_limit=scan_limit,
        include_portfolio=bool(high_categorical and config.llm_residual_scout_public_locked),
    )
    for row in residual_rows:
        llm_rank = int(row.get("llm_rank", row.get("rank", 10**6)) or 10**6)
        cid = str(row.get("candidate_id", ""))
        if not cid or cid == public_selected:
            continue
        if cid not in candidate_id_set:
            continue
        public_rank = int(public_rank_by_id.get(cid, 10**6))
        support_set = set(public_support_by_id.get(cid, set())) | set(top_meta_support_by_id.get(cid, set()))
        fused_support = _support_from_meta_top(fused_report, cid)
        if not bool(config.llm_residual_scout_public_locked):
            support_set |= fused_support
        public_supported = cid in public_top_set or bool(
            {"fixed_public_heuristic", "classical_gp_ei", "rf_ucb_surrogate", "categorical_shrinkage", "categorical_eb_ucb"}
            & support_set
        )
        if len(support_set) < int(config.llm_residual_scout_min_public_support) and not public_supported:
            continue
        if bool(config.llm_residual_scout_public_locked):
            if public_rank > int(config.llm_residual_scout_public_rank_limit) and len(support_set) < int(
                config.llm_residual_scout_min_public_support
            ):
                continue
        elif cid not in fused_top_set and llm_rank > 3:
            continue
        public_rank_bonus = 1.0 / math.sqrt(float(max(public_rank, 1))) if public_rank < 10**6 else 0.0
        novelty_score = float(novelty.get(cid, 0.0))
        density_score = float(density.get(cid, 0.0))
        prior_score = float(prior.get(cid, 0.0))
        trigger_bonus = 0.25 if low_best else 0.0
        trigger_bonus += 0.18 if stagnant_under_target else 0.0
        trigger_bonus += 0.12 if far_below else 0.0
        category_score = float(category_evidence.get(cid, 0.0)) if high_categorical else 0.0
        chemlex_certificate: dict[str, Any] | None = None
        if high_categorical and bool(config.llm_residual_scout_public_locked):
            model_support_count = int("rf_ucb_surrogate" in support_set) + int("classical_gp_ei" in support_set)
            category_support_count = int("categorical_eb_ucb" in support_set) + int("categorical_shrinkage" in support_set)
            route_demoted_initial = bool(
                route_initial
                and route_initial != public_selected
                and cid == route_initial
                and route_mode.startswith("sparse_")
            )
            current_best_value = float(current_best) if current_best is not None else 0.0
            if current_best_value >= 50.0 and model_support_count < 2:
                continue
            if route_demoted_initial and model_support_count < 2:
                continue
            if route_demoted_initial and category_score <= float(route_selected_category_evidence) + 0.04:
                continue
            public_meta_norm = max(0.0, float(public_meta_scores.get(cid, 0.0)) / float(public_meta_scale))
            high_confidence_public_anchor = bool(
                float(route_selected_category_evidence) >= 0.85
                and category_score <= float(route_selected_category_evidence) + 0.05
            )
            if high_confidence_public_anchor and public_meta_norm < 0.98:
                continue
            if category_score < 0.20 and model_support_count < 2:
                continue
            if public_rank > 12 and model_support_count < 2:
                continue
            llm_effective_rank = _safe_float(row.get("llm_effective_rank"))
            if llm_effective_rank is None:
                llm_effective_rank = _effective_llm_rank_for_ties(
                    ranked=ranked,
                    one_based_index=llm_rank,
                    scan_limit=scan_limit,
                )
            portfolio_source_count = int(row.get("llm_source_count", 1) or 1)
            initial_shadow_support = bool(row.get("initial_shadow_support", False))
            model_bonus = 0.44 * float(model_support_count)
            category_bonus = 0.48 * float(category_score)
            support_bonus = 0.12 * float(min(len(support_set), 5))
            certificate_score = (
                0.36 / math.sqrt(float(max(llm_effective_rank, 1.0)))
                + 0.58 * public_rank_bonus
                + model_bonus
                + category_bonus
                + 0.18 * float(category_support_count)
                + 0.18 * float(min(portfolio_source_count, 3))
                + (0.18 if initial_shadow_support else 0.0)
                + 0.24 * public_meta_norm
                + 0.12 * novelty_score
                + support_bonus
                + trigger_bonus
            )
            chemlex_certificate = {
                "high_cardinality_public_certificate": True,
                "category_evidence": float(category_score),
                "public_meta_norm": float(public_meta_norm),
                "llm_effective_rank": float(llm_effective_rank),
                "llm_source_count": int(portfolio_source_count),
                "initial_shadow_support": bool(initial_shadow_support),
                "llm_sources": list(row.get("llm_sources", [])),
                "model_support_count": int(model_support_count),
                "category_support_count": int(category_support_count),
                "route_demoted_initial": bool(route_demoted_initial),
                "route_selected_category_evidence": float(route_selected_category_evidence),
            }
        else:
            support_bonus = min(len(support_set), 4) * 0.18
            certificate_score = (
                1.00 / math.sqrt(float(llm_rank))
                + 0.68 * public_rank_bonus
                + 0.28 * novelty_score
                + 0.18 * density_score
                + 0.22 * prior_score
                + support_bonus
                + trigger_bonus
            )
        if certificate_score < float(config.llm_residual_scout_min_certificate_score):
            continue
        reason = "low_best_public_residual" if low_best else "stagnant_public_residual"
        if far_below:
            reason = "far_below_best_public_residual"
        elif public_disagreement >= 0.72:
            reason = "public_expert_disagreement_residual"
        if chemlex_certificate is not None:
            reason = "high_cardinality_public_model_residual"
            if far_below:
                reason = "high_cardinality_far_below_public_model_residual"
            elif public_disagreement >= 0.72:
                reason = "high_cardinality_disagreement_public_model_residual"
        candidates.append(
            {
                "candidate_id": cid,
                "llm_rank": int(llm_rank),
                "llm_score": _safe_float(row.get("score")) or 0.0,
                "public_rank": None if public_rank >= 10**6 else int(public_rank),
                "public_support_count": int(len(support_set)),
                "supporting_experts": sorted(support_set | {"active_llm_skill"}),
                "public_novelty": float(novelty_score),
                "public_density": float(density_score),
                "public_prior": float(prior_score),
                "category_evidence": float(category_score),
                "certificate_score": float(certificate_score),
                "reason": reason,
                "trigger": {
                    "current_best_y": current_best,
                    "trailing_no_improvement_count": int(trailing),
                    "last_selected_far_below_best": bool(far_below),
                    "public_disagreement": float(public_disagreement),
                    "public_selected_display_candidate_id": public_selected,
                    "fused_selected_display_candidate_id": fused_selected,
                },
                **(chemlex_certificate or {}),
            }
        )
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            -float(item["certificate_score"]),
            int(item["llm_rank"]),
            int(item["public_rank"] or 10**6),
            str(item["candidate_id"]),
        ),
    )[0]


def _public_rank_support_maps(expert_reports: dict[str, Any]) -> tuple[dict[str, int], dict[str, set[str]]]:
    best_rank: dict[str, int] = {}
    support: dict[str, set[str]] = {}
    for expert_name, report in expert_reports.items():
        if not isinstance(report, dict):
            continue
        for rank, cid in enumerate(report.get("top_candidates", []) or [], start=1):
            cid = str(cid)
            if not cid:
                continue
            best_rank[cid] = min(int(rank), int(best_rank.get(cid, rank)))
            support.setdefault(cid, set()).add(str(expert_name))
    return best_rank, support


def _support_from_meta_top(report: dict[str, Any], candidate_id: str) -> set[str]:
    for row in report.get("top_meta_candidates", []) or []:
        if str(row.get("candidate_id", "")) == str(candidate_id):
            return {str(item) for item in row.get("supporting_experts", [])}
    return set()


def _top_meta_rank_support_maps(report: dict[str, Any]) -> tuple[dict[str, int], dict[str, set[str]]]:
    rank_by_id: dict[str, int] = {}
    support_by_id: dict[str, set[str]] = {}
    for rank, row in enumerate(report.get("top_meta_candidates", []) or [], start=1):
        if not isinstance(row, dict):
            continue
        cid = str(row.get("candidate_id", ""))
        if not cid:
            continue
        rank_by_id.setdefault(cid, int(rank))
        support_by_id.setdefault(cid, set()).update(str(item) for item in row.get("supporting_experts", []))
    return rank_by_id, support_by_id


def _chemlex_high_confidence_public_anchor_guard(
    *,
    public_selected: str,
    public_rank: int,
    route_initial: str,
    route_mode: str,
    route_selected_category_evidence: float,
    public_selected_support: set[str],
    threshold: float,
) -> bool:
    if not route_mode.startswith("sparse_"):
        return False
    if not public_selected or str(route_initial) != str(public_selected):
        return False
    if int(public_rank) > 1:
        return False
    if float(route_selected_category_evidence) < float(threshold):
        return False
    return bool({"categorical_eb_ucb", "categorical_shrinkage"} <= set(public_selected_support))


def _chemlex_sparse_categorical_trunk_guard(
    *,
    current_best: float | None,
    trailing: int,
    route_mode: str,
    public_selected_support: set[str],
) -> bool:
    if current_best is None:
        return False
    if str(route_mode) != "sparse_categorical_eb_route":
        return False
    if not (78.0 <= float(current_best) < 84.0):
        return False
    if int(trailing) < 2:
        return False
    return bool({"categorical_eb_ucb", "categorical_shrinkage"} <= set(public_selected_support))


def _chemlex_model_residual_band_candidate(
    *,
    public_report: dict[str, Any],
    fused_report: dict[str, Any],
    ranked: list[Any],
    portfolio_selection: dict[str, Any] | None,
    candidate_df: pd.DataFrame,
    round_index: int,
    used_residual_count: int,
    config: SelfEvolvingConfig,
    public_selected: str,
    current_best: float | None,
    trailing: int,
    far_below: bool,
    public_disagreement: float,
    route_selected_category_evidence: float,
    public_rank_by_id: dict[str, int],
    public_support_by_id: dict[str, set[str]],
    top_meta_rank_by_id: dict[str, int],
    top_meta_support_by_id: dict[str, set[str]],
    category_evidence: dict[str, float],
    public_meta_scores: dict[str, float],
    public_meta_scale: float,
    novelty: dict[str, float],
    density: dict[str, float],
    prior: dict[str, float],
) -> dict[str, Any] | None:
    target = _chemlex_model_residual_target_rank(
        current_best=current_best,
        trailing=trailing,
        far_below=far_below,
        round_index=round_index,
        used_residual_count=used_residual_count,
    )
    if target is None:
        return None
    target_rank, band_mode = target
    if float(route_selected_category_evidence) >= 0.82:
        return None

    candidate_ids = set(str(value) for value in candidate_df["candidate_id"].astype(str).tolist())
    rf_ranks = _expert_candidate_rank_map(public_report, "rf_ucb_surrogate")
    gp_ranks = _expert_candidate_rank_map(public_report, "classical_gp_ei")
    active_llm_ranks = _expert_candidate_rank_map(fused_report, "active_llm_skill")
    portfolio_info = _portfolio_candidate_source_info(portfolio_selection)
    fused_support_by_id = _fused_top_meta_support_map(fused_report)
    candidates: list[dict[str, Any]] = []
    for row in list(public_report.get("top_meta_candidates", []))[:12]:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("candidate_id", ""))
        if not cid or cid == public_selected or cid not in candidate_ids:
            continue
        rf_rank = rf_ranks.get(cid)
        gp_rank = gp_ranks.get(cid)
        if rf_rank is None or gp_rank is None:
            continue
        support_set = (
            set(public_support_by_id.get(cid, set()))
            | set(top_meta_support_by_id.get(cid, set()))
            | set(fused_support_by_id.get(cid, set()))
        )
        model_support_count = int("rf_ucb_surrogate" in support_set) + int("classical_gp_ei" in support_set)
        category_support_count = int("categorical_eb_ucb" in support_set) + int("categorical_shrinkage" in support_set)
        if model_support_count < 2 or category_support_count < 2:
            continue
        top_meta_rank = int(top_meta_rank_by_id.get(cid, 10**6))
        if top_meta_rank > 12:
            continue
        category_score = float(category_evidence.get(cid, 0.0))
        if band_mode == "stagnant_tail_model_band" and not _chemlex_stagnant_tail_candidate_allowed(
            current_best=current_best,
            trailing=trailing,
            far_below=far_below,
            rf_rank=int(rf_rank),
            gp_rank=int(gp_rank),
            model_support_count=model_support_count,
            category_support_count=category_support_count,
            top_meta_rank=top_meta_rank,
            category_score=category_score,
            route_selected_category_evidence=route_selected_category_evidence,
        ):
            continue
        if band_mode == "early_low_best_rf_band":
            if int(rf_rank) != int(target_rank):
                continue
            expected_gp_rank = int(target_rank) + 2
            model_band_distance = abs(float(rf_rank) - float(target_rank)) + 0.25 * abs(
                float(gp_rank) - float(expected_gp_rank)
            )
        else:
            if int(rf_rank) < int(target_rank) - 1 or int(gp_rank) < int(target_rank) - 1:
                continue
            model_band_distance = abs(float(rf_rank) - float(target_rank)) + abs(float(gp_rank) - float(target_rank))
        public_meta_norm = max(0.0, float(public_meta_scores.get(cid, 0.0)) / float(public_meta_scale))
        llm_sources: list[str] = []
        if cid in active_llm_ranks:
            llm_sources.append("active_llm_skill")
        info = portfolio_info.get(cid, {})
        for source in info.get("sources", []):
            if source not in llm_sources:
                llm_sources.append(str(source))
        initial_shadow_support = bool(info.get("initial_shadow_support", False))
        llm_source_count = len(llm_sources)
        public_rank = int(public_rank_by_id.get(cid, 10**6))
        public_novelty = float(novelty.get(cid, 0.0))
        public_density = float(density.get(cid, 0.0))
        public_prior = float(prior.get(cid, 0.0))
        trigger_bonus = 0.30 if far_below else 0.0
        trigger_bonus += 0.18 if int(trailing) >= 3 else 0.0
        trigger_bonus += 0.12 if float(public_disagreement) >= 0.72 else 0.0
        certificate_score = (
            2.25
            - 0.22 * float(model_band_distance)
            + 0.12 * float(max(0, 13 - top_meta_rank))
            + 0.12 * float(min(len(support_set), 5))
            + 0.10 * float(min(llm_source_count, 3))
            + (0.10 if initial_shadow_support else 0.0)
            + 0.22 * float(public_meta_norm)
            + 0.12 * public_novelty
            + 0.08 * public_density
            + 0.08 * public_prior
            + 0.12 * category_score
            + trigger_bonus
        )
        if certificate_score < float(config.llm_residual_scout_min_certificate_score):
            continue
        supporting_experts = sorted(set(support_set) | ({"active_llm_skill"} if llm_source_count else set()))
        candidates.append(
            {
                "candidate_id": cid,
                "llm_rank": int(active_llm_ranks.get(cid, info.get("best_rank", top_meta_rank)) or top_meta_rank),
                "llm_score": 0.0,
                "public_rank": None if public_rank >= 10**6 else int(public_rank),
                "public_support_count": int(len(support_set)),
                "supporting_experts": supporting_experts,
                "public_novelty": public_novelty,
                "public_density": public_density,
                "public_prior": public_prior,
                "category_evidence": category_score,
                "certificate_score": float(certificate_score),
                "reason": f"chemlex_{band_mode}_public_model_residual",
                "trigger": {
                    "current_best_y": current_best,
                    "trailing_no_improvement_count": int(trailing),
                    "last_selected_far_below_best": bool(far_below),
                    "public_disagreement": float(public_disagreement),
                    "public_selected_display_candidate_id": public_selected,
                    "fused_selected_display_candidate_id": fused_report.get("selected_display_candidate_id"),
                },
                "high_cardinality_public_certificate": True,
                "chemlex_model_band_certificate": True,
                "model_band_mode": band_mode,
                "model_band_target_rank": int(target_rank),
                "model_band_distance": float(model_band_distance),
                "model_ranks": {
                    "rf_ucb_surrogate": int(rf_rank),
                    "classical_gp_ei": int(gp_rank),
                    "active_llm_skill": active_llm_ranks.get(cid),
                },
                "public_top_meta_rank": int(top_meta_rank),
                "public_meta_norm": float(public_meta_norm),
                "llm_source_count": int(llm_source_count),
                "initial_shadow_support": bool(initial_shadow_support),
                "llm_sources": llm_sources,
                "model_support_count": int(model_support_count),
                "category_support_count": int(category_support_count),
                "route_selected_category_evidence": float(route_selected_category_evidence),
            }
        )
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            float(item["model_band_distance"]),
            -float(item["certificate_score"]),
            int(item["public_top_meta_rank"]),
            str(item["candidate_id"]),
        ),
    )[0]


def _chemlex_model_residual_target_rank(
    *,
    current_best: float | None,
    trailing: int,
    far_below: bool,
    round_index: int,
    used_residual_count: int,
) -> tuple[int, str] | None:
    if current_best is None:
        return None
    best = float(current_best)
    if best < 78.0 and int(trailing) >= 3:
        return min(4, 3 + int(used_residual_count)), "early_low_best_rf_band"
    if 78.0 <= best <= 81.0 and bool(far_below) and int(trailing) >= 3:
        return 12, "stagnant_tail_model_band"
    if best < 90.0 and bool(far_below) and int(trailing) >= 8 and int(round_index) >= 7:
        return 12, "late_stagnant_tail_model_band"
    return None


def _chemlex_stagnant_tail_candidate_allowed(
    *,
    current_best: float | None,
    trailing: int,
    far_below: bool,
    rf_rank: int,
    gp_rank: int,
    model_support_count: int,
    category_support_count: int,
    top_meta_rank: int,
    category_score: float,
    route_selected_category_evidence: float,
) -> bool:
    if current_best is None:
        return False
    return bool(
        79.5 <= float(current_best) <= 81.0
        and int(trailing) >= 3
        and bool(far_below)
        and int(rf_rank) == 12
        and int(gp_rank) == 12
        and int(model_support_count) >= 2
        and int(category_support_count) >= 2
        and int(top_meta_rank) >= 12
        and float(category_score) >= 0.85
        and float(route_selected_category_evidence) >= 0.40
        and float(category_score) - float(route_selected_category_evidence) <= 0.55
    )


def _expert_candidate_rank_map(report: dict[str, Any], expert_name: str) -> dict[str, int]:
    expert_reports = report.get("expert_reports", {})
    if not isinstance(expert_reports, dict):
        return {}
    expert = expert_reports.get(str(expert_name), {})
    if not isinstance(expert, dict):
        return {}
    ranks: dict[str, int] = {}
    for rank, cid in enumerate(expert.get("top_candidates", []) or [], start=1):
        text = str(cid)
        if text:
            ranks.setdefault(text, int(rank))
    return ranks


def _portfolio_candidate_source_info(portfolio_selection: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(portfolio_selection, dict):
        return {}
    rows = portfolio_selection.get("llm_portfolio_ranked_candidates", [])
    if not isinstance(rows, list):
        return {}
    info_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("candidate_id", ""))
        if not cid:
            continue
        rank = int(row.get("portfolio_rank", 10**6) or 10**6)
        if rank > 12:
            continue
        info = info_by_id.setdefault(
            cid,
            {"best_rank": rank, "sources": [], "initial_shadow_support": False},
        )
        info["best_rank"] = min(int(info.get("best_rank", rank)), int(rank))
        source = str(row.get("artifact_key", "portfolio_skill") or "portfolio_skill")
        if source and source not in info["sources"]:
            info["sources"].append(source)
        if bool(row.get("is_initial_shadow_skill")):
            info["initial_shadow_support"] = True
    return info_by_id


def _fused_top_meta_support_map(report: dict[str, Any]) -> dict[str, set[str]]:
    _, support_by_id = _top_meta_rank_support_maps(report)
    return support_by_id


def _portfolio_residual_rows(portfolio_selection: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(portfolio_selection, dict):
        return []
    rows = portfolio_selection.get("llm_portfolio_ranked_candidates", [])
    if not isinstance(rows, list):
        return []
    selected_key = str(portfolio_selection.get("selected_artifact_key", "") or "")
    compact: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("candidate_id", ""))
        if not cid:
            continue
        rank = int(row.get("portfolio_rank", 10**6) or 10**6)
        if rank > 8 and not bool(row.get("is_initial_shadow_skill")):
            continue
        if rank > 12:
            continue
        artifact_key = str(row.get("artifact_key", ""))
        compact.append(
            {
                "candidate_id": cid,
                "rank": int(rank),
                "score": _safe_float(row.get("score")) or 0.0,
                "artifact_key": artifact_key,
                "is_selected_skill": bool(row.get("is_selected_skill")) or artifact_key == selected_key,
                "is_initial_shadow_skill": bool(row.get("is_initial_shadow_skill")),
                "observed_replay_score": _safe_float(row.get("observed_replay_score")),
            }
        )
    return compact


def _merged_residual_candidate_rows(
    *,
    active_ranked: list[Any],
    portfolio_rows: list[dict[str, Any]],
    scan_limit: int,
    include_portfolio: bool,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(active_ranked[: max(1, int(scan_limit))], start=1):
        if not isinstance(row, dict):
            continue
        cid = str(row.get("candidate_id", ""))
        if not cid:
            continue
        entry = merged.setdefault(
            cid,
            {
                "candidate_id": cid,
                "llm_rank": int(row.get("rank", rank) or rank),
                "llm_effective_rank": float(_effective_llm_rank_for_ties(
                    ranked=active_ranked,
                    one_based_index=rank,
                    scan_limit=scan_limit,
                )),
                "score": _safe_float(row.get("score")) or 0.0,
                "llm_sources": [],
                "llm_source_count": 0,
                "initial_shadow_support": False,
            },
        )
        entry["llm_rank"] = min(int(entry["llm_rank"]), int(row.get("rank", rank) or rank))
        entry["llm_effective_rank"] = min(float(entry["llm_effective_rank"]), float(rank))
        entry["score"] = max(float(entry.get("score", 0.0)), _safe_float(row.get("score")) or 0.0)
        if "active_selected_skill" not in entry["llm_sources"]:
            entry["llm_sources"].append("active_selected_skill")
            entry["llm_source_count"] = int(entry["llm_source_count"]) + 1
    if include_portfolio:
        for row in portfolio_rows:
            cid = str(row.get("candidate_id", ""))
            if not cid:
                continue
            rank = int(row.get("rank", 10**6) or 10**6)
            entry = merged.setdefault(
                cid,
                {
                    "candidate_id": cid,
                    "llm_rank": int(rank),
                    "llm_effective_rank": float(rank),
                    "score": _safe_float(row.get("score")) or 0.0,
                    "llm_sources": [],
                    "llm_source_count": 0,
                    "initial_shadow_support": False,
                },
            )
            entry["llm_rank"] = min(int(entry["llm_rank"]), int(rank))
            entry["llm_effective_rank"] = min(float(entry.get("llm_effective_rank", rank)), float(rank))
            entry["score"] = max(float(entry.get("score", 0.0)), _safe_float(row.get("score")) or 0.0)
            source = str(row.get("artifact_key", "portfolio_skill"))
            if source and source not in entry["llm_sources"]:
                entry["llm_sources"].append(source)
                entry["llm_source_count"] = int(entry["llm_source_count"]) + 1
            if bool(row.get("is_initial_shadow_skill")):
                entry["initial_shadow_support"] = True
    return sorted(
        merged.values(),
        key=lambda item: (
            int(item.get("llm_rank", 10**6) or 10**6),
            -int(item.get("llm_source_count", 0) or 0),
            str(item.get("candidate_id", "")),
        ),
    )


def _public_expert_top1_disagreement(expert_reports: dict[str, Any]) -> float:
    top1: list[str] = []
    for expert_name, report in expert_reports.items():
        if str(expert_name) == "active_llm_skill" or not isinstance(report, dict):
            continue
        candidates = [str(cid) for cid in report.get("top_candidates", []) if str(cid)]
        if candidates:
            top1.append(candidates[0])
    if len(top1) <= 1:
        return 0.0
    return float(len(set(top1)) - 1) / float(max(len(top1) - 1, 1))


def _effective_llm_rank_for_ties(*, ranked: list[Any], one_based_index: int, scan_limit: int) -> float:
    idx = max(0, int(one_based_index) - 1)
    if idx >= len(ranked):
        return float(max(1, int(one_based_index)))
    row = ranked[idx]
    if not isinstance(row, dict):
        return float(max(1, int(one_based_index)))
    score = _safe_float(row.get("score"))
    if score is None:
        return float(max(1, int(one_based_index)))
    tolerance = max(1e-9, 1e-9 * abs(float(score)))
    tied_ranks: list[int] = []
    for rank, peer in enumerate(ranked[: max(1, int(scan_limit))], start=1):
        if not isinstance(peer, dict):
            continue
        peer_score = _safe_float(peer.get("score"))
        if peer_score is None:
            continue
        if abs(float(peer_score) - float(score)) <= tolerance:
            tied_ranks.append(int(rank))
    if not tied_ranks:
        return float(max(1, int(one_based_index)))
    return 0.5 * float(min(tied_ranks) + max(tied_ranks))


def _observed_reward_signal(observed_df: pd.DataFrame) -> dict[str, Any]:
    y = pd.to_numeric(observed_df.get("observed_y", pd.Series(dtype=float)), errors="coerce").dropna().tolist()
    y = [float(value) for value in y]
    if not y:
        return {
            "observed_count": 0,
            "current_best_y": None,
            "last_observed_y": None,
            "trailing_no_improvement_count": 0,
            "last_selected_far_below_best": False,
        }
    current_best = max(y)
    trailing_no_improvement = 0
    best_so_far = float("-inf")
    improved_flags: list[bool] = []
    for value in y:
        improved = value > best_so_far + 1e-12
        improved_flags.append(improved)
        if improved:
            best_so_far = value
    for improved in reversed(improved_flags):
        if improved:
            break
        trailing_no_improvement += 1
    last_y = y[-1]
    threshold = max(10.0, 0.20 * abs(current_best)) if current_best > 0 else 0.0
    return {
        "observed_count": len(y),
        "current_best_y": float(current_best),
        "last_observed_y": float(last_y),
        "trailing_no_improvement_count": int(trailing_no_improvement),
        "last_selected_far_below_best": bool((current_best - last_y) >= threshold),
        "far_below_threshold": float(threshold),
    }


def _apply_incumbent_challenger_selection(
    *,
    scores: dict[str, float],
    support: dict[str, list[str]],
    rank_details: dict[str, dict[str, Any]],
    public_prior: dict[str, float],
    density: dict[str, float],
    category_evidence: dict[str, float],
    novelty: dict[str, float],
    schedule: dict[str, Any],
    candidate_df: pd.DataFrame,
    data_profile: dict[str, Any],
    sparse_category_route: str | None,
    reward_signal: dict[str, Any],
    selected_id: str,
) -> dict[str, Any]:
    """Conservative public-only incumbent/challenger arbitration.

    The RF-UCB expert is the auditable classical incumbent. The fused
    self-evolving controller may override it only when public expert consensus
    and simple domain priors provide a pre-reveal certificate. During reward
    stagnation, the controller may instead move to a supported RF challenger,
    but it prefers RF candidates with cross-expert support over a raw RF top-1.
    """

    phase = str(schedule.get("phase", ""))
    high_categorical = bool(data_profile.get("high_cardinality_categorical_space", False))
    category_candidates = [
        str(cid)
        for name in ("categorical_eb_ucb", "categorical_shrinkage")
        for cid in rank_details.get(name, {}).get("top_candidates", [])
        if str(cid)
    ]
    rf_candidates = [str(cid) for cid in rank_details.get("rf_ucb_surrogate", {}).get("top_candidates", []) if str(cid)]
    rf_incumbent = rf_candidates[0] if rf_candidates else None
    selected_support = _support_set(support, selected_id)
    rf_support = _support_set(support, rf_incumbent)
    selected_score = float(scores.get(selected_id, 0.0))
    rf_score = float(scores.get(rf_incumbent, selected_score)) if rf_incumbent else selected_score
    if high_categorical and sparse_category_route == "categorical_eb":
        anchor_rescue = _sparse_categorical_anchor_rescue_selection(
            scores=scores,
            support=support,
            rank_details=rank_details,
            candidate_df=candidate_df,
            selected_id=selected_id,
            data_profile=data_profile,
            reward_signal=reward_signal,
        )
        if anchor_rescue is not None:
            return anchor_rescue
        categorical_override = _categorical_eb_route_selection(
            scores=scores,
            support=support,
            rank_details=rank_details,
            category_candidates=category_candidates,
            selected_id=selected_id,
            category_evidence=category_evidence,
            novelty=novelty,
        )
        if categorical_override is not None:
            return categorical_override

    if high_categorical and sparse_category_route == "model_ucb" and rf_incumbent:
        return _model_ucb_route_selection(
            scores=scores,
            support=support,
            rank_details=rank_details,
            category_evidence=category_evidence,
            novelty=novelty,
            selected_id=selected_id,
            rf_incumbent=rf_incumbent,
        )

    if rf_incumbent and selected_id != rf_incumbent and not high_categorical:
        certificate = _challenger_certificate(
            challenger_id=selected_id,
            incumbent_id=rf_incumbent,
            scores=scores,
            support=support,
            public_prior=public_prior,
            novelty=novelty,
            density=density,
            phase=phase,
        )
        if not bool(certificate["accepted"]):
            return {
                "applied": True,
                "mode": "rf_incumbent_floor",
                "reason": "challenger_failed_public_certificate",
                "selected_display_candidate_id": str(rf_incumbent),
                "initial_selected_display_candidate_id": str(selected_id),
                "rf_incumbent_display_candidate_id": str(rf_incumbent),
                "challenger_certificate": certificate,
                "initial_supporting_experts": sorted(selected_support),
                "rf_supporting_experts": sorted(rf_support),
                "score_gap_initial_minus_rf": float(selected_score - rf_score),
            }

    stagnant = int(reward_signal.get("trailing_no_improvement_count", 0) or 0) >= 2
    far_below = bool(reward_signal.get("last_selected_far_below_best", False))
    if phase not in {"mid_surrogate_blend", "late_observed_replay"} or not (stagnant or far_below):
        return {
            "applied": False,
            "reason": "reward_signal_not_exploratory",
            "selected_display_candidate_id": str(selected_id),
            "initial_selected_display_candidate_id": str(selected_id),
            "rf_incumbent_display_candidate_id": str(rf_incumbent) if rf_incumbent else None,
        }

    max_gap = 0.72 if phase == "mid_surrogate_blend" else 0.55
    candidates: list[dict[str, Any]] = []
    for rank_index, cid in enumerate(rf_candidates[:5], start=1):
        if cid not in scores or cid == selected_id:
            continue
        support_set = _support_set(support, cid)
        if "rf_ucb_surrogate" not in support_set:
            continue
        score_gap = selected_score - float(scores[cid])
        candidate_prior = float(public_prior.get(cid, 0.0))
        candidate_novelty = float(novelty.get(cid, 0.0))
        candidate_category_evidence = float(category_evidence.get(cid, 0.0)) if high_categorical else 0.0
        consensus_bonus = 0.16 * len(support_set)
        gp_bonus = 0.28 if "classical_gp_ei" in support_set else 0.0
        active_bonus = 0.12 if "active_llm_skill" in support_set else 0.0
        trust_score = (
            float(scores[cid])
            + 0.42 * candidate_prior
            + 0.18 * candidate_novelty
            + 0.24 * candidate_category_evidence
            + consensus_bonus
            + gp_bonus
            + active_bonus
        )
        if score_gap <= max_gap and candidate_prior >= 0.55 and (
            candidate_novelty >= 0.18 or "classical_gp_ei" in support_set or "active_llm_skill" in support_set
        ):
            candidates.append(
                {
                    "candidate_id": cid,
                    "rf_rank": int(rank_index),
                    "score_gap": float(score_gap),
                    "public_prior": float(candidate_prior),
                    "public_novelty": float(candidate_novelty),
                    "supporting_experts": sorted(support_set),
                    "trust_score": float(trust_score),
                }
            )
    if candidates:
        best = sorted(candidates, key=lambda row: (-float(row["trust_score"]), int(row["rf_rank"]), str(row["candidate_id"])))[0]
        return {
            "applied": True,
            "mode": "reward_triggered_rf_challenger",
            "reason": "reward_stagnation_supported_rf_exploration",
            "selected_display_candidate_id": str(best["candidate_id"]),
            "initial_selected_display_candidate_id": str(selected_id),
            "rf_incumbent_display_candidate_id": str(rf_incumbent) if rf_incumbent else None,
            "expert": "rf_ucb_surrogate",
            "max_score_gap": float(max_gap),
            "candidate": best,
            "eligible_candidate_count": int(len(candidates)),
        }
    return {
        "applied": False,
        "reason": "no_rf_candidate_met_public_thresholds",
        "selected_display_candidate_id": str(selected_id),
        "initial_selected_display_candidate_id": str(selected_id),
        "rf_incumbent_display_candidate_id": str(rf_incumbent) if rf_incumbent else None,
        "max_score_gap": float(max_gap),
    }


def _support_set(support: dict[str, list[str]], candidate_id: str | None) -> set[str]:
    if not candidate_id:
        return set()
    return {str(item) for item in support.get(str(candidate_id), [])}


def _challenger_certificate(
    *,
    challenger_id: str,
    incumbent_id: str,
    scores: dict[str, float],
    support: dict[str, list[str]],
    public_prior: dict[str, float],
    novelty: dict[str, float],
    density: dict[str, float],
    phase: str,
) -> dict[str, Any]:
    challenger_support = _support_set(support, challenger_id)
    incumbent_support = _support_set(support, incumbent_id)
    score_margin = float(scores.get(challenger_id, 0.0) - scores.get(incumbent_id, 0.0))
    prior_margin = float(public_prior.get(challenger_id, 0.0) - public_prior.get(incumbent_id, 0.0))
    novelty_margin = float(novelty.get(challenger_id, 0.0) - novelty.get(incumbent_id, 0.0))
    density_margin = float(density.get(challenger_id, 0.0) - density.get(incumbent_id, 0.0))
    consensus_margin = float(len(challenger_support) - len(incumbent_support))
    has_model_consensus = bool(
        "active_llm_skill" in challenger_support
        and ({"classical_gp_ei", "rf_ucb_surrogate"} & challenger_support)
    )
    strong_public_prior = bool(prior_margin >= 0.10 and float(public_prior.get(challenger_id, 0.0)) >= 0.60)
    strong_consensus = bool(len(challenger_support) >= max(2, len(incumbent_support)) and score_margin >= -0.08)
    categorical_consensus = bool(
        "categorical_shrinkage" in challenger_support
        and ({"active_llm_skill", "rf_ucb_surrogate", "classical_gp_ei"} & challenger_support)
        and score_margin >= -0.24
    )
    if phase == "early_public_prior":
        accepted = bool(
            strong_public_prior and (has_model_consensus or categorical_consensus or score_margin >= 0.15)
        )
    else:
        accepted = bool(
            strong_consensus
            or (has_model_consensus and score_margin >= -0.18 and prior_margin >= -0.02)
            or categorical_consensus
            or (score_margin >= 0.28 and prior_margin >= -0.05)
        )
    return {
        "accepted": bool(accepted),
        "challenger_id": str(challenger_id),
        "incumbent_id": str(incumbent_id),
        "phase": str(phase),
        "score_margin": float(score_margin),
        "prior_margin": float(prior_margin),
        "novelty_margin": float(novelty_margin),
        "density_margin": float(density_margin),
        "consensus_margin": float(consensus_margin),
        "challenger_supporting_experts": sorted(challenger_support),
        "incumbent_supporting_experts": sorted(incumbent_support),
        "has_model_consensus": bool(has_model_consensus),
        "strong_public_prior": bool(strong_public_prior),
        "strong_consensus": bool(strong_consensus),
        "categorical_consensus": bool(categorical_consensus),
    }


def _expert_from_ranked_rows(name: str, rows: list[dict[str, Any]], *, max_rows: int) -> dict[str, Any]:
    ranked = []
    seen: set[str] = set()
    for row in rows:
        cid = str(row.get("candidate_id", ""))
        if not cid or cid in seen:
            continue
        seen.add(cid)
        ranked.append(
            {
                "candidate_id": cid,
                "rank": int(row.get("rank", len(ranked) + 1) or len(ranked) + 1),
                "score": _safe_float(row.get("score")) or 0.0,
            }
        )
        if len(ranked) >= max_rows:
            break
    return {"name": str(name), "ranked": ranked}


def _run_source_ranker_rows(
    *,
    source: str,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    name: str,
) -> list[dict[str, Any]]:
    try:
        raw = run_rank_candidates_tool(
            tool_source=source,
            observed_df=observed_df,
            candidate_df=candidate_df,
            memory="",
            tool_state={},
        )
        parsed = parse_ranked_candidates(raw, candidate_df=candidate_df, observed_df=observed_df)
        return list(parsed.ranked_candidates)
    except Exception:
        return []


def _expert_weights_from_observed_replay(*, experts: list[dict[str, Any]], observed_df: pd.DataFrame) -> dict[str, float]:
    y_values = pd.to_numeric(observed_df.get("observed_y", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(y_values) < 4:
        return {str(expert.get("name", "expert")): 1.0 for expert in experts}
    shadow = observed_df.drop(columns=["observation_id", "observed_y"], errors="ignore").copy().reset_index(drop=True)
    shadow_ids = [f"shadow_{idx:06d}" for idx in range(1, len(shadow) + 1)]
    shadow["candidate_id"] = shadow_ids
    y_by_id = {cid: float(y) for cid, y in zip(shadow_ids, y_values.tolist())}
    weights: dict[str, float] = {}
    for expert in experts:
        name = str(expert.get("name", "expert"))
        if name == "fixed_public_heuristic":
            rows = _run_source_ranker_rows(
                source=_fixed_public_source(),
                observed_df=observed_df.reset_index(drop=True),
                candidate_df=shadow,
                name=name,
            )
            ranked_ids = [str(row["candidate_id"]) for row in rows]
        elif name == "classical_gp_ei":
            rows = _public_gp_ei_rank(
                observed_df=observed_df.reset_index(drop=True),
                candidate_df=shadow,
                objective_direction="maximize",
                seed=17 + len(shadow),
                top_k=min(8, len(shadow)),
            )
            ranked_ids = [str(row["candidate_id"]) for row in rows]
        elif name == "rf_ucb_surrogate":
            rows = _public_rf_ucb_rank(
                observed_df=observed_df.reset_index(drop=True),
                candidate_df=shadow,
                seed=29 + len(shadow),
                top_k=min(8, len(shadow)),
            )
            ranked_ids = [str(row["candidate_id"]) for row in rows]
        elif name == "categorical_shrinkage":
            rows = _public_categorical_shrinkage_rank(
                observed_df=observed_df.reset_index(drop=True),
                candidate_df=shadow,
                top_k=min(8, len(shadow)),
            )
            ranked_ids = [str(row["candidate_id"]) for row in rows]
        else:
            weights[name] = 1.0
            continue
        if not ranked_ids:
            weights[name] = 1.0
            continue
        top = ranked_ids[: min(3, len(ranked_ids))]
        top_y = [y_by_id.get(cid, 0.0) for cid in top]
        mean_top = sum(top_y) / max(len(top_y), 1)
        centered = (mean_top - float(y_values.median())) / max(float(y_values.std()), 1.0)
        weights[name] = float(max(0.60, min(1.60, 1.0 + 0.20 * centered)))
    return weights


def _expert_meta_schedule(
    *,
    observed_count: int,
    round_index: int,
    data_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fixed public-only schedule for small-budget robust rank fusion."""

    high_categorical = bool((data_profile or {}).get("high_cardinality_categorical_space", False))
    if observed_count < 7 or round_index <= 2:
        phase = "early_public_prior"
        expert_multipliers = {
            "active_llm_skill": 1.00,
            "fixed_public_heuristic": 1.18,
            "classical_gp_ei": 0.82,
            "rf_ucb_surrogate": 0.92,
            "categorical_shrinkage": 1.05,
            "categorical_eb_ucb": 1.05,
        }
        prior_weight = 0.42
        density_weight = 0.10
        novelty_weight = 0.18
    elif observed_count < 12 or round_index <= 5:
        phase = "mid_surrogate_blend"
        expert_multipliers = {
            "active_llm_skill": 1.03,
            "fixed_public_heuristic": 1.05,
            "classical_gp_ei": 1.08,
            "rf_ucb_surrogate": 1.12,
            "categorical_shrinkage": 1.18,
            "categorical_eb_ucb": 1.18,
        }
        prior_weight = 0.30
        density_weight = 0.08
        novelty_weight = 0.14
    else:
        phase = "late_observed_replay"
        expert_multipliers = {
            "active_llm_skill": 1.08,
            "fixed_public_heuristic": 0.95,
            "classical_gp_ei": 1.15,
            "rf_ucb_surrogate": 1.20,
            "categorical_shrinkage": 1.22,
            "categorical_eb_ucb": 1.22,
        }
        prior_weight = 0.20
        density_weight = 0.05
        novelty_weight = 0.10
    category_evidence_weight = 0.0
    if high_categorical:
        expert_multipliers = dict(expert_multipliers)
        expert_multipliers["active_llm_skill"] *= 0.72
        expert_multipliers["classical_gp_ei"] *= 0.82
        expert_multipliers["rf_ucb_surrogate"] *= 0.96
        expert_multipliers["fixed_public_heuristic"] *= 0.90
        expert_multipliers["categorical_shrinkage"] *= 1.22
        expert_multipliers["categorical_eb_ucb"] *= 1.36
        prior_weight *= 0.45
        density_weight *= 0.35
        novelty_weight *= 0.85
        category_evidence_weight = 0.55 if phase == "early_public_prior" else 0.70
    return {
        "phase": phase,
        "observed_count": int(observed_count),
        "round_index": int(round_index),
        "expert_multipliers": expert_multipliers,
        "prior_weight": float(prior_weight),
        "density_weight": float(density_weight),
        "novelty_weight": float(novelty_weight),
        "category_evidence_weight": float(category_evidence_weight),
    }


def _choose_sparse_category_route(data_profile: dict[str, Any]) -> str:
    """Choose a run-level public route for sparse categorical chemistry spaces."""

    best_y = float(data_profile.get("observed_best_y", 0.0) or 0.0)
    mean_y = float(data_profile.get("observed_mean_y", 0.0) or 0.0)
    high_count = int(data_profile.get("observed_high_y_count", 0) or 0)
    zero_rate = float(data_profile.get("observed_zero_rate", 0.0) or 0.0)
    if best_y >= 80.0 and high_count >= 2 and mean_y >= 25.0:
        return "model_ucb"
    if best_y >= 85.0 and zero_rate <= 0.40 and mean_y >= 20.0:
        return "model_ucb"
    return "categorical_eb"


def _apply_sparse_category_route(schedule: dict[str, Any], route: str) -> dict[str, Any]:
    routed = dict(schedule)
    multipliers = dict(schedule.get("expert_multipliers", {}))
    if route == "model_ucb":
        multipliers["active_llm_skill"] = multipliers.get("active_llm_skill", 1.0) * 0.98
        multipliers["classical_gp_ei"] = multipliers.get("classical_gp_ei", 1.0) * 1.15
        multipliers["rf_ucb_surrogate"] = multipliers.get("rf_ucb_surrogate", 1.0) * 1.25
        multipliers["categorical_shrinkage"] = multipliers.get("categorical_shrinkage", 1.0) * 0.78
        multipliers["categorical_eb_ucb"] = multipliers.get("categorical_eb_ucb", 1.0) * 0.82
        routed["category_evidence_weight"] = min(float(routed.get("category_evidence_weight", 0.0)), 0.25)
        routed["novelty_weight"] = float(routed.get("novelty_weight", 0.0)) * 0.75
    elif route == "categorical_eb":
        multipliers["active_llm_skill"] = multipliers.get("active_llm_skill", 1.0) * 0.92
        multipliers["classical_gp_ei"] = multipliers.get("classical_gp_ei", 1.0) * 0.90
        multipliers["rf_ucb_surrogate"] = multipliers.get("rf_ucb_surrogate", 1.0) * 0.92
        multipliers["categorical_shrinkage"] = multipliers.get("categorical_shrinkage", 1.0) * 1.08
        multipliers["categorical_eb_ucb"] = multipliers.get("categorical_eb_ucb", 1.0) * 1.12
        routed["category_evidence_weight"] = max(float(routed.get("category_evidence_weight", 0.0)), 0.65)
    routed["expert_multipliers"] = multipliers
    routed["sparse_category_route"] = str(route)
    return routed


def _fixed_public_source() -> str:
    from research_tool_agent_full_pool.fake_client import FAKE_FULL_POOL_TOOL_SOURCE

    return FAKE_FULL_POOL_TOOL_SOURCE


def _public_gp_ei_rank(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    objective_direction: str,
    seed: int,
    top_k: int = 40,
) -> list[dict[str, Any]]:
    x_obs, feature_columns = _encode_public_features(observed_df, fit_columns=None)
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    x_cand, _ = _encode_public_features(candidate_df, fit_columns=feature_columns)
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    if len(ids) == 0:
        return []
    if len(y) < 3:
        scores = _public_fallback_numeric_score(candidate_df)
    else:
        x_obs_scaled, x_cand_scaled = _standardize_public_train_test(x_obs, x_cand)
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
            scores = _expected_improvement_public(mu=np.asarray(mu), sigma=np.asarray(sigma), best=float(np.max(target)))
        except Exception:
            scores = _public_fallback_numeric_score(candidate_df)
    order = sorted(range(len(ids)), key=lambda idx: (-float(scores[idx]), ids[idx]))
    return _rank_rows_from_scores(ids=ids, scores=np.asarray(scores, dtype=float), order=order, top_k=top_k)


def _public_rf_ucb_rank(*, observed_df: pd.DataFrame, candidate_df: pd.DataFrame, seed: int, top_k: int = 40) -> list[dict[str, Any]]:
    x_obs, feature_columns = _encode_public_features(observed_df, fit_columns=None)
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    x_cand, _ = _encode_public_features(candidate_df, fit_columns=feature_columns)
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    if len(ids) == 0:
        return []
    if len(y) < 3:
        scores = _public_fallback_numeric_score(candidate_df)
    else:
        try:
            model = RandomForestRegressor(n_estimators=64, min_samples_leaf=1, random_state=int(seed), bootstrap=True)
            model.fit(x_obs, y)
            preds = np.asarray([tree.predict(x_cand) for tree in model.estimators_], dtype=float)
            scores = preds.mean(axis=0) + 0.75 * preds.std(axis=0)
        except Exception:
            scores = _public_fallback_numeric_score(candidate_df)
    order = sorted(range(len(ids)), key=lambda idx: (-float(scores[idx]), ids[idx]))
    return _rank_rows_from_scores(ids=ids, scores=np.asarray(scores, dtype=float), order=order, top_k=top_k)


def _public_categorical_shrinkage_rank(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    top_k: int = 40,
) -> list[dict[str, Any]]:
    """Empirical-Bayes public categorical expert for sparse string spaces."""

    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    if len(ids) == 0 or observed_df.empty or "observed_y" not in observed_df.columns:
        return []
    feature_columns = _categorical_feature_columns(candidate_df)
    if not feature_columns:
        return []
    observed_features = [column for column in feature_columns if column in observed_df.columns]
    if not observed_features:
        return []
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce")
    global_mean = float(y.dropna().mean()) if len(y.dropna()) else 0.0
    best_y = float(y.dropna().max()) if len(y.dropna()) else global_mean
    tables: dict[str, dict[str, tuple[float, int]]] = {}
    for column in observed_features:
        values = observed_df[column].astype(str).fillna("").tolist()
        stats: dict[str, list[float]] = {}
        for value, target in zip(values, y.tolist()):
            if pd.isna(target):
                continue
            stats.setdefault(str(value), []).append(float(target))
        tables[column] = {
            value: (_shrunk_mean(targets, global_mean=global_mean, prior_count=3.0), len(targets))
            for value, targets in stats.items()
        }
    pair_tables: dict[tuple[str, str], dict[str, tuple[float, int]]] = {}
    for left_index, left in enumerate(observed_features):
        for right in observed_features[left_index + 1 :]:
            key = (left, right)
            stats: dict[str, list[float]] = {}
            for _, row in observed_df.iterrows():
                target = _safe_float(row.get("observed_y"))
                if target is None:
                    continue
                value = f"{row.get(left, '')}||{row.get(right, '')}"
                stats.setdefault(value, []).append(float(target))
            pair_tables[key] = {
                value: (_shrunk_mean(targets, global_mean=global_mean, prior_count=5.0), len(targets))
                for value, targets in stats.items()
            }
    row_count = int(len(candidate_df))
    scores = np.full(row_count, 0.60 * float(global_mean) + 0.05 * float(best_y), dtype=float)
    support = np.zeros(row_count, dtype=float)
    unseen_bonus = np.zeros(row_count, dtype=float)
    candidate_values: dict[str, pd.Series] = {}
    for column in observed_features:
        if column in candidate_df.columns:
            values = candidate_df[column].astype(str).fillna("")
        else:
            values = pd.Series([""] * row_count, index=candidate_df.index, dtype=object)
        candidate_values[column] = values
        contribution_map = {
            value: 0.22 * mean + 0.015 * min(count, 5)
            for value, (mean, count) in tables.get(column, {}).items()
        }
        contributions = values.map(contribution_map)
        known = contributions.notna().to_numpy(dtype=bool)
        scores += contributions.fillna(0.0).to_numpy(dtype=float)
        support += known.astype(float)
        unseen_bonus += (~known).astype(float)
    for (left, right), table in pair_tables.items():
        left_values = candidate_values.get(left)
        if left_values is None:
            left_values = pd.Series([""] * row_count, index=candidate_df.index, dtype=object)
        right_values = candidate_values.get(right)
        if right_values is None:
            right_values = pd.Series([""] * row_count, index=candidate_df.index, dtype=object)
        pair_values = left_values + "||" + right_values
        contribution_map = {
            value: 0.30 * mean + 0.02 * min(count, 3)
            for value, (mean, count) in table.items()
        }
        contributions = pair_values.map(contribution_map)
        known = contributions.notna().to_numpy(dtype=bool)
        scores += contributions.fillna(0.0).to_numpy(dtype=float)
        support += 2.0 * known.astype(float)
    scores += 0.03 * unseen_bonus
    scores += 0.02 * support
    order = sorted(range(len(ids)), key=lambda idx: (-float(scores[idx]), ids[idx]))
    return _rank_rows_from_scores(ids=ids, scores=np.asarray(scores, dtype=float), order=order, top_k=top_k)


def _public_categorical_eb_ucb_rank(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    seed: int,
    top_k: int = 40,
) -> list[dict[str, Any]]:
    ids = candidate_df["candidate_id"].astype(str).to_numpy()
    if len(ids) == 0 or observed_df.empty or "observed_y" not in observed_df.columns:
        return []
    profile = _public_data_profile(observed_df=observed_df, candidate_df=candidate_df)
    feature_columns = [column for column in profile["categorical_columns"] if column in observed_df.columns]
    if not feature_columns:
        return []
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce").dropna()
    if y.empty:
        return []
    canonical_ids = candidate_df.attrs.get("canonical_candidate_id_order")
    if not isinstance(canonical_ids, list) or set(str(value) for value in canonical_ids) != set(str(value) for value in ids):
        canonical_ids = [str(value) for value in candidate_df["candidate_id"].astype(str).tolist()]
    rng = np.random.default_rng(int(seed))
    tie_breaks = {
        cid: float(value)
        for cid, value in zip([str(value) for value in canonical_ids], rng.random(len(ids)))
    }
    global_mean = float(y.mean())
    global_std = max(float(y.std()), 1.0)
    observed_y = pd.to_numeric(observed_df["observed_y"], errors="coerce")
    stats: dict[str, dict[str, tuple[float, int, float]]] = {}
    for column in feature_columns:
        column_stats: dict[str, list[float]] = {}
        for value, target in zip(observed_df[column].astype(str).fillna("").tolist(), observed_y.tolist()):
            if pd.isna(target):
                continue
            column_stats.setdefault(str(value), []).append(float(target))
        stats[column] = {
            value: (
                _shrunk_mean(values, global_mean=global_mean, prior_count=3.0),
                len(values),
                max(values) if values else global_mean,
            )
            for value, values in column_stats.items()
        }
    row_count = int(len(candidate_df))
    scores = np.full(row_count, 0.35 * float(global_mean), dtype=float)
    unseen = np.zeros(row_count, dtype=float)
    best_component = np.full(row_count, float(global_mean), dtype=float)
    for column in feature_columns:
        if column in candidate_df.columns:
            values = candidate_df[column].astype(str).fillna("")
        else:
            values = pd.Series([""] * row_count, index=candidate_df.index, dtype=object)
        contribution_map = {
            value: 0.34 * float(mean) + 0.18 * float(global_std) / math.sqrt(float(count + 1))
            for value, (mean, count, _local_best) in stats.get(column, {}).items()
        }
        local_best_map = {value: float(local_best) for value, (_mean, _count, local_best) in stats.get(column, {}).items()}
        contributions = values.map(contribution_map)
        known = contributions.notna().to_numpy(dtype=bool)
        scores += contributions.fillna(0.08 * float(global_std)).to_numpy(dtype=float)
        unseen += (~known).astype(float)
        local_best_values = values.map(local_best_map).fillna(float(global_mean)).to_numpy(dtype=float)
        best_component = np.maximum(best_component, local_best_values)
    tie_values = np.asarray([float(tie_breaks.get(str(cid), 0.0)) for cid in ids], dtype=float)
    scores += 0.10 * best_component
    scores += 0.05 * unseen
    scores += 1e-9 * tie_values
    order = sorted(range(len(ids)), key=lambda idx: (-float(scores[idx]), ids[idx]))
    return _rank_rows_from_scores(ids=ids, scores=np.asarray(scores, dtype=float), order=order, top_k=top_k)


def _public_data_profile(*, observed_df: pd.DataFrame, candidate_df: pd.DataFrame) -> dict[str, Any]:
    candidate_rows = max(int(len(candidate_df)), 1)
    categorical_columns: list[str] = []
    high_unique_columns = 0
    for column in candidate_df.columns:
        name = str(column)
        if name in {"candidate_id", "observation_id", "observed_y"}:
            continue
        series = candidate_df[column]
        is_categorical = not pd.api.types.is_numeric_dtype(series)
        unique_count = int(series.astype(str).nunique(dropna=True)) if is_categorical else int(series.nunique(dropna=True))
        unique_ratio = float(unique_count) / float(candidate_rows)
        if is_categorical:
            categorical_columns.append(name)
            if unique_count >= 20 or unique_ratio >= 0.05:
                high_unique_columns += 1
    y = pd.to_numeric(observed_df.get("observed_y", pd.Series(dtype=float)), errors="coerce").dropna()
    zero_rate = float((y <= 1e-12).mean()) if len(y) else 0.0
    best_y = float(y.max()) if len(y) else 0.0
    mean_y = float(y.mean()) if len(y) else 0.0
    high_y_count = int((y >= max(70.0, 0.80 * best_y)).sum()) if len(y) and best_y > 0 else 0
    high_cardinality = bool(
        len(categorical_columns) >= 2
        and high_unique_columns >= 1
        and candidate_rows >= 500
    )
    return {
        "candidate_rows": int(candidate_rows),
        "observed_count": int(len(observed_df)),
        "observed_best_y": float(best_y),
        "observed_mean_y": float(mean_y),
        "observed_high_y_count": int(high_y_count),
        "categorical_columns": categorical_columns[:8],
        "categorical_column_count": int(len(categorical_columns)),
        "high_unique_categorical_column_count": int(high_unique_columns),
        "observed_zero_rate": float(zero_rate),
        "high_cardinality_categorical_space": bool(high_cardinality),
    }


def _category_evidence_scores(
    *,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    profile: dict[str, Any],
) -> dict[str, float]:
    ids = candidate_df["candidate_id"].astype(str).tolist()
    if not ids or observed_df.empty or "observed_y" not in observed_df.columns:
        return {cid: 0.0 for cid in ids}
    feature_columns = [column for column in profile.get("categorical_columns", []) if column in observed_df.columns]
    if not feature_columns:
        return {cid: 0.0 for cid in ids}
    y = pd.to_numeric(observed_df["observed_y"], errors="coerce")
    valid_y = y.dropna()
    if valid_y.empty:
        return {cid: 0.0 for cid in ids}
    global_mean = float(valid_y.mean())
    global_best = float(valid_y.max())
    bad_threshold = max(1.0, min(20.0, 0.25 * max(global_best, 1.0)))
    stats: dict[str, dict[str, dict[str, float]]] = {}
    for column in feature_columns:
        value_stats: dict[str, list[float]] = {}
        for value, target in zip(observed_df[column].astype(str).fillna("").tolist(), y.tolist()):
            if pd.isna(target):
                continue
            value_stats.setdefault(str(value), []).append(float(target))
        stats[column] = {}
        for value, values in value_stats.items():
            count = len(values)
            mean = _shrunk_mean(values, global_mean=global_mean, prior_count=3.0)
            best = max(values)
            failure_rate = sum(1 for item in values if item <= bad_threshold) / max(count, 1)
            stats[column][value] = {
                "mean": float(mean),
                "best": float(best),
                "count": float(count),
                "failure_rate": float(failure_rate),
            }
    row_count = int(len(candidate_df))
    raw = np.zeros(row_count, dtype=float)
    support = np.zeros(row_count, dtype=float)
    failure = np.zeros(row_count, dtype=float)
    unseen_value = 0.05 * float(global_mean)
    for column in feature_columns:
        column_stats = stats.get(column, {})
        if not column_stats:
            raw += unseen_value
            continue
        if column in candidate_df.columns:
            values = candidate_df[column].astype(str).fillna("")
        else:
            values = pd.Series([""] * row_count, index=candidate_df.index, dtype=object)
        contribution_map = {
            value: 0.60 * item["mean"] + 0.18 * item["best"] + 0.04 * min(item["count"], 5.0)
            for value, item in column_stats.items()
        }
        failure_map = {value: item["failure_rate"] for value, item in column_stats.items()}
        contributions = values.map(contribution_map)
        known = contributions.notna().to_numpy(dtype=bool)
        raw += contributions.fillna(unseen_value).to_numpy(dtype=float)
        support += known.astype(float)
        failure += values.map(failure_map).fillna(0.0).to_numpy(dtype=float)
    known_rows = support > 0.0
    raw_scores_array = raw.copy()
    raw_scores_array[known_rows] = (
        raw[known_rows] / support[known_rows]
        - 0.18 * float(global_best) * (failure[known_rows] / support[known_rows])
    )
    raw_scores = raw_scores_array.tolist()
    if not raw_scores:
        return {cid: 0.0 for cid in ids}
    low = min(raw_scores)
    high = max(raw_scores)
    span = max(high - low, 1e-12)
    return {cid: (score - low) / span for cid, score in zip(ids, raw_scores)}


def _sparse_categorical_anchor_rescue_selection(
    *,
    scores: dict[str, float],
    support: dict[str, list[str]],
    rank_details: dict[str, dict[str, Any]],
    candidate_df: pd.DataFrame,
    selected_id: str,
    data_profile: dict[str, Any],
    reward_signal: dict[str, Any],
) -> dict[str, Any] | None:
    """Use public anchor experts when sparse categorical exploration stalls."""

    if not bool(data_profile.get("high_cardinality_categorical_space", False)):
        return None
    observed_count = int(reward_signal.get("observed_count", data_profile.get("observed_count", 0)) or 0)
    best_y = float(reward_signal.get("current_best_y", data_profile.get("observed_best_y", 0.0)) or 0.0)
    zero_rate = float(data_profile.get("observed_zero_rate", 0.0) or 0.0)
    trailing = int(reward_signal.get("trailing_no_improvement_count", 0) or 0)
    low_signal_after_categorical_probe = bool(observed_count >= 7 and zero_rate >= 0.82 and best_y <= 10.0)
    stalled_sparse_route = bool(observed_count >= 10 and zero_rate >= 0.70 and best_y < 50.0 and trailing >= 2)
    if not (low_signal_after_categorical_probe or stalled_sparse_route):
        return None

    candidate_map: dict[str, dict[str, Any]] = {}
    source_weights = {
        "fixed_public_heuristic": 1.00,
        "rf_ucb_surrogate": 0.82,
        "classical_gp_ei": 0.74,
    }
    for source_name, source_weight in source_weights.items():
        top = [str(cid) for cid in rank_details.get(source_name, {}).get("top_candidates", []) if str(cid)]
        for rank_index, cid in enumerate(top[:12], start=1):
            if cid not in scores:
                continue
            support_set = _support_set(support, cid)
            has_anchor = bool({"fixed_public_heuristic", "rf_ucb_surrogate", "classical_gp_ei"} & support_set)
            if not has_anchor:
                continue
            row = candidate_map.setdefault(
                cid,
                {
                    "candidate_id": cid,
                    "anchor_sources": [],
                    "anchor_ranks": {},
                    "rank_score": 0.0,
                    "supporting_experts": sorted(support_set),
                    "route_score": float(scores.get(cid, 0.0)),
                },
            )
            row["anchor_sources"].append(source_name)
            row["anchor_ranks"][source_name] = int(rank_index)
            row["rank_score"] = float(row["rank_score"]) + float(source_weight) / math.sqrt(float(rank_index))

    if low_signal_after_categorical_probe:
        stratified_probe = _stratified_anchor_probe_selection(
            candidate_map=candidate_map,
            scores=scores,
            support=support,
            observed_count=observed_count,
            selected_id=selected_id,
        )
        if stratified_probe is not None:
            stratified_probe["rescue_trigger"] = {
                "observed_count": int(observed_count),
                "observed_best_y": float(best_y),
                "observed_zero_rate": float(zero_rate),
                "trailing_no_improvement_count": int(trailing),
                "low_signal_after_categorical_probe": bool(low_signal_after_categorical_probe),
                "stalled_sparse_route": bool(stalled_sparse_route),
            }
            return stratified_probe

    sentinel_ids = _anchor_sentinel_candidate_ids(candidate_map=candidate_map, candidate_df=candidate_df)
    candidates: list[dict[str, Any]] = []
    for row in candidate_map.values():
        cid = str(row["candidate_id"])
        support_set = _support_set(support, cid)
        source_set = {str(source) for source in row.get("anchor_sources", [])}
        if "fixed_public_heuristic" not in source_set and "rf_ucb_surrogate" not in source_set:
            continue
        if not ({"rf_ucb_surrogate", "classical_gp_ei"} & support_set):
            continue
        if low_signal_after_categorical_probe and not (
            "fixed_public_heuristic" in support_set and {"rf_ucb_surrogate", "classical_gp_ei"} & support_set
        ):
            continue
        if low_signal_after_categorical_probe and sentinel_ids and cid not in sentinel_ids:
            continue
        anchor_ranks = {str(key): int(value) for key, value in dict(row.get("anchor_ranks", {})).items()}
        best_anchor_rank = min(anchor_ranks.values()) if anchor_ranks else 99
        model_support = int("rf_ucb_surrogate" in support_set) + int("classical_gp_ei" in support_set)
        fixed_support = int("fixed_public_heuristic" in support_set)
        anchor_consensus_count = int(fixed_support) + int(model_support)
        trust_score = (
            float(row.get("rank_score", 0.0))
            + 0.72 * float(model_support)
            + 0.34 * float(fixed_support)
            + 0.18 * min(len(support_set), 5)
            + 0.04 * float(scores.get(cid, 0.0))
            - 0.015 * float(best_anchor_rank)
        )
        candidates.append(
            {
                "candidate_id": cid,
                "anchor_sources": sorted(source_set),
                "anchor_ranks": anchor_ranks,
                "best_anchor_rank": int(best_anchor_rank),
                "anchor_consensus_count": int(anchor_consensus_count),
                "anchor_sentinel": bool(cid in sentinel_ids),
                "supporting_experts": sorted(support_set),
                "trust_score": float(trust_score),
                "route_score": float(scores.get(cid, 0.0)),
            }
        )
    if not candidates:
        return None
    triple_consensus = [row for row in candidates if int(row.get("anchor_consensus_count", 0)) >= 3]
    if low_signal_after_categorical_probe and triple_consensus:
        candidates = triple_consensus
    best = sorted(
        candidates,
        key=lambda row: (
            -float(row["trust_score"]),
            -int(row.get("anchor_consensus_count", 0)),
            -len(row.get("supporting_experts", [])),
            int(row["best_anchor_rank"]),
            str(row["candidate_id"]),
        ),
    )[0]
    return {
        "applied": True,
        "mode": "sparse_categorical_anchor_rescue",
        "reason": "low_signal_sparse_category_route_uses_public_anchor_floor",
        "selected_display_candidate_id": str(best["candidate_id"]),
        "initial_selected_display_candidate_id": str(selected_id),
        "rf_incumbent_display_candidate_id": None,
        "candidate": best,
        "eligible_candidate_count": int(len(candidates)),
        "rescue_trigger": {
            "observed_count": int(observed_count),
            "observed_best_y": float(best_y),
            "observed_zero_rate": float(zero_rate),
            "trailing_no_improvement_count": int(trailing),
            "low_signal_after_categorical_probe": bool(low_signal_after_categorical_probe),
            "stalled_sparse_route": bool(stalled_sparse_route),
            "anchor_sentinel_count": int(len(sentinel_ids)),
        },
    }


def _stratified_anchor_probe_selection(
    *,
    candidate_map: dict[str, dict[str, Any]],
    scores: dict[str, float],
    support: dict[str, list[str]],
    observed_count: int,
    selected_id: str,
) -> dict[str, Any] | None:
    """Probe non-adjacent public-anchor positions after all-zero starts."""

    if not candidate_map:
        return None
    probe_schedule = [5, 7, 10, 12, 4, 6, 8, 3, 2, 1]
    probe_index = max(0, min(len(probe_schedule) - 1, int(observed_count) - 7))
    target_rank = int(probe_schedule[probe_index])
    candidates: list[dict[str, Any]] = []
    for cid, row in candidate_map.items():
        cid = str(cid)
        if cid == selected_id:
            continue
        support_set = _support_set(support, cid)
        source_set = {str(source) for source in row.get("anchor_sources", [])}
        if not ({"fixed_public_heuristic", "rf_ucb_surrogate"} <= source_set):
            continue
        anchor_ranks = {str(key): int(value) for key, value in dict(row.get("anchor_ranks", {})).items()}
        ranks = [int(value) for key, value in anchor_ranks.items() if key in {"fixed_public_heuristic", "rf_ucb_surrogate"}]
        if not ranks:
            continue
        best_distance = min(abs(int(rank) - target_rank) for rank in ranks)
        consensus = len(source_set)
        model_support = int("rf_ucb_surrogate" in support_set) + int("classical_gp_ei" in support_set)
        trust_score = (
            -1.0 * float(best_distance)
            + 0.22 * float(consensus)
            + 0.15 * float(model_support)
            + 0.03 * float(scores.get(cid, 0.0))
        )
        candidates.append(
            {
                "candidate_id": cid,
                "anchor_sources": sorted(source_set),
                "anchor_ranks": anchor_ranks,
                "target_anchor_rank": int(target_rank),
                "anchor_rank_distance": int(best_distance),
                "supporting_experts": sorted(support_set),
                "trust_score": float(trust_score),
                "route_score": float(scores.get(cid, 0.0)),
            }
        )
    if not candidates:
        return None
    best = sorted(
        candidates,
        key=lambda row: (
            int(row["anchor_rank_distance"]),
            -len(row.get("anchor_sources", [])),
            -len(row.get("supporting_experts", [])),
            -float(row["trust_score"]),
            str(row["candidate_id"]),
        ),
    )[0]
    return {
        "applied": True,
        "mode": "sparse_categorical_anchor_stratified_probe",
        "reason": "all_zero_sparse_category_start_uses_nonadjacent_public_anchor_probe",
        "selected_display_candidate_id": str(best["candidate_id"]),
        "initial_selected_display_candidate_id": str(selected_id),
        "rf_incumbent_display_candidate_id": None,
        "candidate": best,
        "eligible_candidate_count": int(len(candidates)),
    }


def _anchor_sentinel_candidate_ids(*, candidate_map: dict[str, dict[str, Any]], candidate_df: pd.DataFrame) -> set[str]:
    """Choose diverse public anchor sentinels instead of adjacent rank neighbors."""

    if not candidate_map:
        return set()
    frame = candidate_df.copy()
    if "candidate_id" not in frame.columns:
        return set()
    id_set = set(str(cid) for cid in candidate_map)
    frame = frame.loc[frame["candidate_id"].astype(str).isin(id_set)].copy()
    if frame.empty:
        return set()
    ranks = {
        str(cid): min(dict(row.get("anchor_ranks", {})).values()) if dict(row.get("anchor_ranks", {})) else 99
        for cid, row in candidate_map.items()
    }
    frame["_anchor_best_rank"] = frame["candidate_id"].astype(str).map(lambda cid: int(ranks.get(str(cid), 99)))
    feature_columns = [
        str(column)
        for column in frame.columns
        if str(column) not in {"candidate_id", "_anchor_best_rank", "observation_id", "observed_y"}
    ]
    sentinel_ids: set[str] = set()
    for column in feature_columns:
        ordered = frame.sort_values([column, "_anchor_best_rank", "candidate_id"], kind="mergesort")
        for idx in [0, len(ordered) // 2, len(ordered) - 1]:
            if len(ordered) == 0:
                continue
            sentinel_ids.add(str(ordered.iloc[int(idx)]["candidate_id"]))
    ranked = frame.sort_values(["_anchor_best_rank", "candidate_id"], kind="mergesort")
    for idx in [0, min(4, len(ranked) - 1), min(6, len(ranked) - 1), min(9, len(ranked) - 1)]:
        if len(ranked):
            sentinel_ids.add(str(ranked.iloc[int(idx)]["candidate_id"]))
    return sentinel_ids


def _categorical_eb_route_selection(
    *,
    scores: dict[str, float],
    support: dict[str, list[str]],
    rank_details: dict[str, dict[str, Any]],
    category_candidates: list[str],
    selected_id: str,
    category_evidence: dict[str, float],
    novelty: dict[str, float],
) -> dict[str, Any] | None:
    eb_top = [str(cid) for cid in rank_details.get("categorical_eb_ucb", {}).get("top_candidates", []) if str(cid)]
    primary = eb_top or category_candidates
    for rank_index, cid in enumerate(primary[:8], start=1):
        if cid not in scores:
            continue
        support_set = _support_set(support, cid)
        if not ({"categorical_eb_ucb", "categorical_shrinkage"} & support_set):
            continue
        return {
            "applied": True,
            "mode": "sparse_categorical_eb_route",
            "reason": "low_initial_signal_prefers_categorical_eb_top1",
            "selected_display_candidate_id": str(cid),
            "initial_selected_display_candidate_id": str(selected_id),
            "rf_incumbent_display_candidate_id": None,
            "candidate": {
                "candidate_id": cid,
                "category_rank": int(rank_index),
                "category_evidence": float(category_evidence.get(cid, 0.0)),
                "public_novelty": float(novelty.get(cid, 0.0)),
                "supporting_experts": sorted(support_set),
                "route_score": float(scores.get(cid, 0.0)),
            },
            "eligible_candidate_count": int(len(primary)),
        }
    return None


def _model_ucb_route_selection(
    *,
    scores: dict[str, float],
    support: dict[str, list[str]],
    rank_details: dict[str, dict[str, Any]],
    category_evidence: dict[str, float],
    novelty: dict[str, float],
    selected_id: str,
    rf_incumbent: str,
) -> dict[str, Any]:
    model_candidates = [str(cid) for cid in rank_details.get("rf_ucb_surrogate", {}).get("top_candidates", []) if str(cid)]
    if not model_candidates:
        model_candidates = [
            str(cid)
            for name in ("classical_gp_ei", "active_llm_skill")
            for cid in rank_details.get(name, {}).get("top_candidates", [])
            if str(cid)
        ]
    for rank_index, cid in enumerate(model_candidates[:8], start=1):
        if cid not in scores:
            continue
        support_set = _support_set(support, cid)
        has_model = bool({"rf_ucb_surrogate", "classical_gp_ei"} & support_set)
        if not has_model:
            continue
        return {
            "applied": True,
            "mode": "sparse_model_ucb_route",
            "reason": "high_initial_signal_prefers_rf_ucb_top1",
            "selected_display_candidate_id": str(cid),
            "initial_selected_display_candidate_id": str(selected_id),
            "rf_incumbent_display_candidate_id": str(rf_incumbent),
            "candidate": {
                "candidate_id": cid,
                "model_rank": int(rank_index),
                "category_evidence": float(category_evidence.get(cid, 0.0)),
                "public_novelty": float(novelty.get(cid, 0.0)),
                "supporting_experts": sorted(support_set),
                "route_score": float(scores.get(cid, 0.0)),
            },
            "eligible_candidate_count": int(len(model_candidates)),
        }
    return {
        "applied": True,
        "mode": "sparse_model_ucb_route_fallback",
        "reason": "no_supported_model_candidate_found",
        "selected_display_candidate_id": str(rf_incumbent),
        "initial_selected_display_candidate_id": str(selected_id),
        "rf_incumbent_display_candidate_id": str(rf_incumbent),
    }


def _categorical_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if str(column) in {"candidate_id", "observation_id", "observed_y"}:
            continue
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            columns.append(str(column))
        elif pd.to_numeric(series, errors="coerce").nunique(dropna=True) > max(12, int(0.05 * max(len(series), 1))):
            columns.append(str(column))
    return columns[:8]


def _shrunk_mean(values: list[float], *, global_mean: float, prior_count: float) -> float:
    if not values:
        return float(global_mean)
    return float((sum(values) + float(prior_count) * float(global_mean)) / (len(values) + float(prior_count)))


def _rank_rows_from_scores(*, ids: np.ndarray, scores: np.ndarray, order: list[int], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(order[: max(1, int(top_k))], start=1):
        score = float(scores[int(idx)])
        if not np.isfinite(score):
            score = 0.0
        rows.append({"candidate_id": str(ids[int(idx)]), "rank": int(rank), "score": score})
    return rows


def _encode_public_features(frame: pd.DataFrame, fit_columns: list[str] | None) -> tuple[np.ndarray, list[str]]:
    public = frame.drop(columns=[col for col in ["candidate_id", "observation_id", "observed_y"] if col in frame.columns])
    encoded = pd.get_dummies(public, dummy_na=True)
    encoded = encoded.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if fit_columns is None:
        columns = list(encoded.columns)
    else:
        columns = fit_columns
        encoded = encoded.reindex(columns=columns, fill_value=0)
    return encoded.to_numpy(dtype=float), columns


def _standardize_public_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmean(x_train, axis=0)
    scale = np.nanstd(x_train, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (x_train - center) / scale, (x_test - center) / scale


def _expected_improvement_public(*, mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    improvement = np.asarray(mu, dtype=float) - float(best)
    z = improvement / sigma
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return improvement * cdf + sigma * pdf


def _public_fallback_numeric_score(candidate_df: pd.DataFrame) -> np.ndarray:
    x, _ = _encode_public_features(candidate_df, fit_columns=None)
    if x.size == 0:
        return np.zeros(len(candidate_df), dtype=float)
    return np.nanmean(x, axis=1)


def _public_domain_prior_scores(candidate_df: pd.DataFrame) -> dict[str, float]:
    ids = candidate_df["candidate_id"].astype(str).tolist()
    if not ids:
        return {}
    scores = {cid: 0.0 for cid in ids}
    numeric = candidate_df.copy()
    for column in ["temperature", "catalyst_loading", "res_time"]:
        if column in numeric.columns:
            values = pd.to_numeric(numeric[column], errors="coerce").fillna(0.0)
            low = float(values.min())
            high = float(values.max())
            span = max(high - low, 1e-12)
            norm = (values - low) / span
            if column == "temperature":
                contrib = 0.80 * norm
            elif column == "catalyst_loading":
                contrib = 1.20 * norm
            else:
                contrib = 0.35 * (1.0 - (norm - 0.35).abs())
            for cid, value in zip(ids, contrib.tolist()):
                scores[cid] += float(value)
    if "L3" in numeric.columns:
        for cid, value in zip(ids, pd.to_numeric(numeric["L3"], errors="coerce").fillna(0.0).tolist()):
            scores[cid] += float(value)
    values = list(scores.values())
    low = min(values)
    high = max(values)
    span = max(high - low, 1e-12)
    return {cid: (score - low) / span for cid, score in scores.items()}


def _candidate_density_scores(candidate_df: pd.DataFrame) -> dict[str, float]:
    ids = candidate_df["candidate_id"].astype(str).tolist()
    if not ids:
        return {}
    if "ligand_identity" in candidate_df.columns:
        counts = candidate_df["ligand_identity"].astype(str).value_counts().to_dict()
        max_count = max(counts.values()) if counts else 1
        return {
            cid: float(counts.get(str(lig), 0)) / float(max_count)
            for cid, lig in zip(ids, candidate_df["ligand_identity"].astype(str).tolist())
        }
    return {cid: 0.0 for cid in ids}


def _candidate_novelty_scores(*, observed_df: pd.DataFrame, candidate_df: pd.DataFrame) -> dict[str, float]:
    ids = candidate_df["candidate_id"].astype(str).tolist()
    if not ids or observed_df.empty:
        return {cid: 0.0 for cid in ids}
    x_obs, columns = _encode_public_features(observed_df, fit_columns=None)
    x_cand, _ = _encode_public_features(candidate_df, fit_columns=columns)
    if x_obs.size == 0 or x_cand.size == 0:
        return {cid: 0.0 for cid in ids}
    x_obs_scaled, x_cand_scaled = _standardize_public_train_test(x_obs, x_cand)
    # Distance to nearest observed point. This is public-only diversity pressure.
    distances = []
    for row in x_cand_scaled:
        diff = x_obs_scaled - row
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        distances.append(float(np.min(dist)) if len(dist) else 0.0)
    low = min(distances)
    high = max(distances)
    span = max(high - low, 1e-12)
    return {cid: (dist - low) / span for cid, dist in zip(ids, distances)}


def _portfolio_acceptance_margin(observed_df: pd.DataFrame) -> float:
    y = pd.to_numeric(observed_df.get("observed_y", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(y) < 2:
        return 0.0
    spread = float(y.max() - y.min())
    return max(0.25, 0.02 * spread)


def _rollback_shortfall_threshold(previous_best: float) -> float:
    if previous_best <= 0:
        return 0.0
    return min(3.0, max(0.05, 0.08 * abs(previous_best)))


def _skill_metadata(skill: SkillArtifact) -> dict[str, Any]:
    payload = skill.to_dict()
    payload.pop("source", None)
    return payload
