"""Policy skeleton for the full-pool persistent ResearchToolAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from research_tool_agent_full_pool import artifact_logger
from research_tool_agent_full_pool.agent_context import build_agent_context_markdown
from research_tool_agent_full_pool.config import ResearchToolFullPoolConfig
from research_tool_agent_full_pool.diagnostics import (
    classify_generated_tool_family,
    hash_payload,
    rank1_score_margin,
    score_distribution_summary,
    sha256_text,
)
from research_tool_agent_full_pool.fallback import full_pool_random_fallback
from research_tool_agent_full_pool.feedback.patch_decision import decide_patch_action, write_patch_decision
from research_tool_agent_full_pool.feedback.tool_feedback_report import (
    build_tool_feedback_report,
    selected_public_features_from_observed_df,
    write_tool_feedback_report,
)
from research_tool_agent_full_pool.memory import initialize_memory, update_memory_after_reveal
from research_tool_agent_full_pool.observed_evidence import build_observed_evidence_markdown
from research_tool_agent_full_pool.state import ResearchToolRunState
from research_tool_agent_full_pool.tool_contract import REQUIRED_ENTRYPOINT
from research_tool_agent_full_pool.tool_output_parser import ParsedToolOutput, parse_ranked_candidates
from research_tool_agent_full_pool.tool_patch_synthesis import patch_tool_after_reveal
from research_tool_agent_full_pool.tool_runner import run_rank_candidates_tool
from research_tool_agent_full_pool.tool_sandbox import validate_tool_in_sandbox
from research_tool_agent_full_pool.tool_static_check import static_check_generated_tool_source
from research_tool_agent_full_pool.tool_synthesis import (
    ToolSynthesisParseError,
    synthesize_initial_tool,
    synthesize_initial_tool_with_reports,
)
from research_tool_agent_full_pool.views import (
    build_full_remaining_candidate_df,
    build_observed_df_from_revealed_state,
    map_display_candidate_to_internal_id,
    validate_observed_df,
    validate_public_candidate_df,
)


@dataclass
class FullPoolDecision:
    """Decision payload returned by the Step 1 full-pool policy."""

    selected_candidate_ids: list[Any]
    policy_name: str
    round_index: int
    decision_metadata: dict[str, Any] = field(default_factory=dict)
    fallback_used: bool = False


class ResearchToolFullPoolPolicy:
    """Full-pool persistent generated-tool optimizer policy skeleton.

    Contract:
    - initialize run-level memory and generated-tool state once per replay.
    - every round, build `observed_df` from revealed observations only.
    - every round, build the full remaining public-safe candidate pool.
    - score the entire remaining pool with the generated optimizer tool.
    - select rank 1 directly; no final LLM override in the first versions.
    - after reveal, update observed state, memory, strategy_state, and tool_state.

    The policy may use public-safe generated optimizer scoring, including
    surrogate-like and acquisition-like components computed from observed and
    public-safe candidate inputs. It must not call/import/read/copy or condition
    on BOReferencePolicy, repository BO modules, legacy csebo_harness, candidate
    score CSV artifacts, BO baseline ranks/scores, reference predictive
    statistics or acquisition values, oracle/posthoc ranks, hidden outcomes,
    evaluator internals, private ID maps, API keys, or Authorization/Bearer
    tokens.
    """

    def __init__(
        self,
        config: ResearchToolFullPoolConfig | None = None,
        *,
        tool_synthesis_client: Any | None = None,
    ) -> None:
        self.config = config or ResearchToolFullPoolConfig()
        self.config.validate_contract_flags()
        self.tool_synthesis_client = tool_synthesis_client
        self.state: ResearchToolRunState | None = None
        self.last_artifacts: list[dict[str, Any]] = []
        self._pending_tool_state: dict[str, Any] = {}
        self._pending_tool_diagnostics: dict[str, Any] = {}
        self._pending_decision_metadata: dict[str, Any] = {}
        self._latest_agent_context_text = ""
        self._latest_context_hashes: dict[str, str] = {}
        self._tool_synthesis_failed_permanently = False

    def initialize_run(self, *, tables: Any, replay_state: Any) -> ResearchToolRunState:
        """Initialize persistent run state before the first decision round."""

        current_best = _safe_float(replay_state.best_observed(self.config.target_column, self.config.objective_direction))
        memory_text = initialize_memory(
            task_name=self.config.dataset_name,
            objective_name=self.config.target_column,
            observed_count=len(replay_state.observed_candidates),
            current_best_y=current_best,
            tool_name=None,
        )
        self.state = ResearchToolRunState.initialize_run_state(
            run_id=self.config.run_id,
            memory_text=memory_text,
            strategy_state={
                "run_id": self.config.run_id,
                "dataset_name": self.config.dataset_name,
                "objective_name": self.config.target_column,
                "candidate_source": self.config.candidate_source,
                "mode": self.config.mode,
                "rounds_completed": 0,
                "selected_count": 0,
                "fallback_count": 0,
                "parser_valid_count": 0,
                "static_check_pass_count": 0,
                "sandbox_pass_count": 0,
                "tool_output_valid_count": 0,
                "validator_valid_count": None,
            },
            tool_state={},
        )
        artifact_logger.ensure_output_dir(self.config.output_dir)
        artifact_logger.initialize_artifact_files(self.config.output_dir)
        artifact_logger.write_config(self.config.output_dir, self.config)
        artifact_logger.write_memory(self.config.output_dir, self.state.memory_text)
        artifact_logger.write_strategy_state(self.config.output_dir, self.state.strategy_state)
        artifact_logger.write_tool_state(self.config.output_dir, self.state.tool_state)
        self._write_context_artifacts(
            tables=tables,
            replay_state=replay_state,
            round_index=0,
        )
        artifact_logger.write_artifact_manifest(self.config.output_dir)
        return self.state

    def decide(self, *, tables: Any, replay_state: Any) -> FullPoolDecision:
        """Score the full remaining public-safe pool and return rank-1 candidate.

        This method must not accept or construct a fixed-size candidate menu.
        Any fallback must sample from the full remaining eligible pool.
        """

        run_state = self.state or self.initialize_run(tables=tables, replay_state=replay_state)
        round_index = int(getattr(replay_state, "round_index", 0)) + 1
        observed_df = build_observed_df_from_revealed_state(
            tables,
            replay_state,
            objective_name=self.config.target_column,
        )
        candidate_df = build_full_remaining_candidate_df(tables, replay_state)
        validate_observed_df(observed_df)
        validate_public_candidate_df(candidate_df)
        full_remaining_pool_size = len(replay_state.remaining_candidates)
        candidate_df_rows = len(candidate_df)
        is_full_pool = candidate_df_rows == full_remaining_pool_size
        artifact_logger.write_candidate_view_audit(
            self.config.output_dir,
            {
                "round_index": round_index,
                "candidate_df_rows": candidate_df_rows,
                "full_remaining_pool_size": full_remaining_pool_size,
                "candidate_df_is_full_pool": is_full_pool,
                "candidate_columns": list(candidate_df.columns),
            },
        )
        artifact_logger.write_candidate_id_map(self.config.output_dir, candidate_df)
        context_hashes = self._write_context_artifacts(
            tables=tables,
            replay_state=replay_state,
            round_index=round_index,
        )
        if not is_full_pool:
            return self._fallback_decision(
                candidate_df=candidate_df,
                round_index=round_index,
                reason="candidate_df_not_full_pool",
            )

        if run_state.generated_tool_source is None:
            if self._tool_synthesis_failed_permanently:
                return self._fallback_decision(
                    candidate_df=candidate_df,
                    round_index=round_index,
                    reason="tool_synthesis_unavailable",
                )
            try:
                source, tool_name, repair_count = self._create_initial_tool(
                    observed_df=observed_df,
                    candidate_df=candidate_df,
                    round_index=round_index,
                )
                run_state.record_tool_created(
                    source=source,
                    tool_name=tool_name,
                    round_index=round_index,
                )
                for _ in range(repair_count):
                    run_state.record_tool_repaired()
            except Exception as exc:
                self._tool_synthesis_failed_permanently = True
                return self._fallback_decision(
                    candidate_df=candidate_df,
                    round_index=round_index,
                    reason=f"tool_synthesis_failure:{exc.__class__.__name__}",
                )
        else:
            run_state.record_tool_reused()

        try:
            parsed = self._score_full_pool_with_tool(
                observed_df=observed_df,
                candidate_df=candidate_df,
                round_index=round_index,
            )
            selected_display_id = self._select_rank1(parsed)
            selected_tool_row = next(
                row for row in parsed.ranked_candidates if str(row["candidate_id"]) == str(selected_display_id)
            )
            selected_internal_id = map_display_candidate_to_internal_id(candidate_df, selected_display_id)
            selected_from_full_pool = selected_display_id in set(candidate_df["candidate_id"].astype(str).tolist())
            if self.state is not None:
                self.state.strategy_state["tool_output_valid_count"] = int(
                    self.state.strategy_state.get("tool_output_valid_count", 0)
                ) + 1
            decision_metadata = {
                "selection_rule": f"{self.config.mode}_rank1_by_generated_tool_score",
                "selected_display_candidate_id": selected_display_id,
                "candidate_df_rows": candidate_df_rows,
                "full_remaining_pool_size": full_remaining_pool_size,
                "selected_from_full_pool": selected_from_full_pool,
                "fallback_used": False,
                "tool_ranked_candidate_count": len(parsed.ranked_candidates),
                "observed_evidence_hash": context_hashes["observed_evidence_hash"],
                "agent_context_hash": context_hashes["agent_context_hash"],
                "tool_state_hash_before_decision": hash_payload(run_state.tool_state),
                "generated_tool_family": classify_generated_tool_family(run_state.generated_tool_source or ""),
                "score_distribution_summary": score_distribution_summary(parsed.ranked_candidates),
                "rank1_score_margin": rank1_score_margin(parsed.ranked_candidates),
                "selected_tool_score": selected_tool_row["score"],
                "selected_tool_rank": selected_tool_row["rank"],
            }
            self._pending_tool_state = dict(parsed.tool_state)
            self._pending_tool_diagnostics = dict(parsed.tool_diagnostics)
            self._pending_decision_metadata = dict(decision_metadata)
            artifact_logger.write_generated_tool_output(
                self.config.output_dir,
                {
                    "round_index": round_index,
                    "ranked_candidates": parsed.ranked_candidates,
                    "tool_diagnostics": parsed.tool_diagnostics,
                },
            )
            artifact_logger.write_full_pool_decision(
                self.config.output_dir,
                {
                    "round_index": round_index,
                    "selected_display_candidate_id": selected_display_id,
                    "selected_internal_candidate_id": str(selected_internal_id),
                    **decision_metadata,
                },
            )
            artifact_logger.write_artifact_manifest(self.config.output_dir)
            return FullPoolDecision(
                selected_candidate_ids=[selected_internal_id],
                policy_name=self.config.decision_policy_name,
                round_index=round_index,
                decision_metadata=decision_metadata,
                fallback_used=False,
            )
        except Exception as exc:
            return self._fallback_decision(
                candidate_df=candidate_df,
                round_index=round_index,
                reason=f"tool_failure:{exc.__class__.__name__}",
            )

    def update_after_reveal(
        self,
        *,
        tables: Any,
        replay_state: Any,
        decision: FullPoolDecision,
        revealed_rows: pd.DataFrame,
    ) -> ResearchToolRunState:
        """Update memory/tool state after the evaluator reveals selected `y`."""

        if self.state is None:
            raise RuntimeError("Policy run state has not been initialized.")
        round_index = int(decision.round_index)
        selected_display = str(decision.decision_metadata.get("selected_display_candidate_id", ""))
        target = self.config.target_column
        last_y = None
        if target in revealed_rows.columns and not revealed_rows.empty:
            last_y = _safe_float(revealed_rows[target].iloc[0])
        best_y = _safe_float(replay_state.best_observed(target, self.config.objective_direction))
        observed_count = len(replay_state.observed_candidates)
        remaining_count = len(replay_state.remaining_candidates)
        observed_df = build_observed_df_from_revealed_state(
            tables,
            replay_state,
            objective_name=target,
        )
        feedback_report = build_tool_feedback_report(
            round_id=round_index,
            selected_candidate_id=selected_display,
            selected_candidate_public_features=selected_public_features_from_observed_df(
                observed_df,
                selected_display,
            ),
            revealed_y=last_y,
            observed_df=observed_df,
            decision_metadata=self._pending_decision_metadata,
            tool_diagnostics=self._pending_tool_diagnostics,
            parser_status="valid" if not decision.fallback_used else "not_available",
            static_check_status=_status_from_count(self.state.strategy_state, "static_check_pass_count"),
            sandbox_status=_status_from_count(self.state.strategy_state, "sandbox_pass_count"),
            objective_direction=self.config.objective_direction,
        )
        write_tool_feedback_report(self.config.output_dir, feedback_report)
        patch_decision = decide_patch_action(
            feedback_report,
            tool_state=self.state.tool_state,
            strategy_state=self.state.strategy_state,
            config=self.config,
        )
        write_patch_decision(self.config.output_dir, patch_decision)
        patch_outcome = {
            "attempted": False,
            "synthesis_called": False,
            "verifier_called": False,
            "replacement_performed": False,
            "reason": "patch_mode_not_enabled_or_no_research_flow",
        }
        if (
            str(getattr(self.config, "patch_mode", "decision_only")) == "enabled"
            and patch_decision.decision == "patch_without_search"
            and self.state.generated_tool_source
            and self.tool_synthesis_client is not None
        ):
            patch_candidate_df = build_full_remaining_candidate_df(tables, replay_state)
            validate_public_candidate_df(patch_candidate_df)
            patch_outcome = patch_tool_after_reveal(
                state=self.state,
                config=self.config,
                client=self.tool_synthesis_client,
                old_tool_source=self.state.generated_tool_source,
                observed_df=observed_df,
                candidate_df=patch_candidate_df,
                memory_text=self.state.memory_text,
                strategy_state=self.state.strategy_state,
                tool_state={**self.state.tool_state, **self._pending_tool_state},
                feedback_report=feedback_report,
                patch_decision=patch_decision,
                research_context={},
                patch_research_context={},
                output_dir=self.config.output_dir,
                round_index=round_index,
            )

        memory_text = update_memory_after_reveal(
            task_name=self.config.dataset_name,
            objective_name=target,
            observed_count=observed_count,
            last_selected_candidate=selected_display,
            last_revealed_y=last_y,
            current_best_y=best_y,
            tool_name=self.state.generated_tool_name,
            round_index=round_index,
            previous_memory=self.state.memory_text,
        )
        previous_selected = int(self.state.strategy_state.get("selected_count", 0))
        previous_fallback = int(self.state.strategy_state.get("fallback_count", 0))
        selected_count = previous_selected + len(decision.selected_candidate_ids)
        fallback_count = previous_fallback + (1 if decision.fallback_used else 0)
        strategy_updates = {
            "rounds_completed": round_index,
            "selected_count": selected_count,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_count / max(round_index, 1),
            "observed_count": observed_count,
            "remaining_count": remaining_count,
            "current_best_observed_y": best_y,
            "last_selected_display_candidate_id": selected_display,
            "last_revealed_y": last_y,
            "last_patch_decision": patch_decision.decision,
            "last_patch_decision_requires_tool_patch": patch_decision.requires_tool_patch,
            "patch_replacement_performed": bool(patch_outcome.get("replacement_performed")),
            "tool_patch_count": int(self.state.tool_patch_count),
            "last_patch_outcome": _compact_patch_outcome(patch_outcome),
        }
        tool_updates = {
            **self._pending_tool_state,
            "last_update_round": round_index,
            "last_selected_display_candidate_id": selected_display,
            "last_revealed_y": last_y,
            "last_patch_outcome": _compact_patch_outcome(patch_outcome),
        }
        self.state.update_after_reveal(
            memory_text=memory_text,
            strategy_state_updates=strategy_updates,
            tool_state_updates=tool_updates,
        )
        post_update_context_hashes = self._write_context_artifacts(
            tables=tables,
            replay_state=replay_state,
            round_index=round_index,
        )
        summary = {
            "round_index": round_index,
            "selected_display_candidate_id": selected_display,
            "selected_internal_candidate_id": str(decision.selected_candidate_ids[0]),
            "selected_y": last_y,
            "best_observed_y": best_y,
            "observed_count": observed_count,
            "remaining_count": remaining_count,
            "fallback_used": bool(decision.fallback_used),
            "candidate_df_rows": decision.decision_metadata.get("candidate_df_rows"),
            "full_remaining_pool_size": decision.decision_metadata.get("full_remaining_pool_size"),
            "selected_from_full_pool": decision.decision_metadata.get("selected_from_full_pool"),
            "observed_evidence_hash": post_update_context_hashes["observed_evidence_hash"],
            "agent_context_hash": post_update_context_hashes["agent_context_hash"],
            "tool_state_hash": hash_payload(self.state.tool_state),
            "patch_decision": patch_decision.decision,
            "tool_patch_replacement_performed": bool(patch_outcome.get("replacement_performed")),
            "patch_attempted": bool(patch_outcome.get("attempted")),
            "patch_verifier_called": bool(patch_outcome.get("verifier_called")),
        }
        self.state.record_round_summary(summary)
        artifact_logger.write_memory(self.config.output_dir, self.state.memory_text)
        artifact_logger.write_strategy_state(self.config.output_dir, self.state.strategy_state)
        artifact_logger.write_tool_state(self.config.output_dir, self.state.tool_state)
        artifact_logger.write_tool_state_by_round(
            self.config.output_dir,
            {
                "round_index": round_index,
                "tool_state": self.state.tool_state,
                "tool_state_hash": hash_payload(self.state.tool_state),
            },
        )
        artifact_logger.write_round_summary(self.config.output_dir, summary)
        artifact_logger.write_artifact_manifest(self.config.output_dir)
        return self.state

    def _score_full_pool_with_tool(
        self,
        *,
        observed_df: pd.DataFrame,
        candidate_df: pd.DataFrame,
        round_index: int,
    ) -> ParsedToolOutput:
        """Run the generated tool against all remaining candidates."""

        if self.state is None or self.state.generated_tool_source is None:
            raise RuntimeError("Generated tool source is not available.")
        raw_output = run_rank_candidates_tool(
            tool_source=self.state.generated_tool_source,
            observed_df=observed_df,
            candidate_df=candidate_df,
            memory=self.state.memory_text,
            tool_state=self.state.tool_state,
        )
        return parse_ranked_candidates(raw_output, candidate_df=candidate_df, observed_df=observed_df)

    def _select_rank1(self, parsed_output: ParsedToolOutput) -> str:
        """Select exactly the candidate ranked first by the tool output."""

        rank1 = [row for row in parsed_output.ranked_candidates if int(row["rank"]) == 1]
        if len(rank1) != 1:
            raise ValueError("Parsed tool output does not contain exactly one rank-1 row.")
        return str(rank1[0]["candidate_id"])

    def _fallback_full_pool_random(
        self,
        *,
        candidate_df: pd.DataFrame,
        round_index: int,
        reason: str,
    ) -> dict[str, Any]:
        """Fallback to full-pool Random over all remaining eligible candidates."""

        return full_pool_random_fallback(
            candidate_df,
            seed=self.config.seed,
            round_index=round_index,
            reason=reason,
        )

    def _fallback_decision(
        self,
        *,
        candidate_df: pd.DataFrame,
        round_index: int,
        reason: str,
    ) -> FullPoolDecision:
        fallback = self._fallback_full_pool_random(
            candidate_df=candidate_df,
            round_index=round_index,
            reason=reason,
        )
        display_id = str(fallback["candidate_id"])
        internal_id = map_display_candidate_to_internal_id(candidate_df, display_id)
        metadata = {
            "selection_rule": "full_pool_seeded_random_fallback",
            "selected_display_candidate_id": display_id,
            "candidate_df_rows": len(candidate_df),
            "full_remaining_pool_size": int(candidate_df.attrs.get("full_remaining_pool_size", len(candidate_df))),
            "selected_from_full_pool": display_id in set(candidate_df["candidate_id"].astype(str).tolist()),
            "fallback_used": True,
            **fallback,
            **self._latest_context_hashes,
            "tool_state_hash_before_decision": hash_payload(self.state.tool_state if self.state else {}),
        }
        self._pending_tool_state = {}
        self._pending_tool_diagnostics = {"fallback_reason": reason}
        self._pending_decision_metadata = dict(metadata)
        artifact_logger.write_fallback_event(
            self.config.output_dir,
            {
                "round_index": round_index,
                **metadata,
            },
        )
        artifact_logger.write_full_pool_decision(
            self.config.output_dir,
            {
                "round_index": round_index,
                "selected_display_candidate_id": display_id,
                "selected_internal_candidate_id": str(internal_id),
                **metadata,
            },
        )
        artifact_logger.write_artifact_manifest(self.config.output_dir)
        return FullPoolDecision(
            selected_candidate_ids=[internal_id],
            policy_name=self.config.decision_policy_name,
            round_index=round_index,
            decision_metadata=metadata,
            fallback_used=True,
        )

    def _create_initial_tool(
        self,
        *,
        observed_df: pd.DataFrame,
        candidate_df: pd.DataFrame,
        round_index: int,
    ) -> tuple[str, str, int]:
        """Create and validate the persistent generated tool."""

        if self.config.mode == "fake":
            source = synthesize_initial_tool(mode="fake")
            static_report = static_check_generated_tool_source(source)
            artifact_logger.write_static_check_report(
                self.config.output_dir,
                {"round_index": round_index, **static_report},
            )
            sandbox_report = validate_tool_in_sandbox(
                tool_source=source,
                observed_df=observed_df,
                candidate_df=candidate_df,
                memory=self.state.memory_text if self.state else "",
                tool_state=self.state.tool_state if self.state else {},
            )
            artifact_logger.write_sandbox_report(
                self.config.output_dir,
                {"round_index": round_index, **sandbox_report},
            )
            if not static_report["passed"] or not sandbox_report["passed"]:
                raise ValueError("Fake generated tool failed local validation.")
            artifact_logger.write_generated_tool_source(
                self.config.output_dir,
                round_index=round_index,
                tool_name=REQUIRED_ENTRYPOINT,
                source=source,
            )
            return source, REQUIRED_ENTRYPOINT, 0

        client = self.tool_synthesis_client
        if client is None:
            from research_tool_agent_full_pool.api_client import CommonstackToolSynthesisClient

            client = CommonstackToolSynthesisClient(
                endpoint=self.config.endpoint,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                response_format_json_object=self.config.response_format_json_object,
                api_key_path=self.config.api_key_path,
                backup_api_key_path=self.config.backup_api_key_path,
            )
            self.tool_synthesis_client = client

        last_error: str | None = None
        max_attempts = 1 + max(int(self.config.repair_attempts), 0)
        for attempt_index in range(max_attempts):
            try:
                result = synthesize_initial_tool_with_reports(
                    mode="api",
                    client=client,
                    config=self.config,
                    observed_df=observed_df,
                    candidate_df=candidate_df,
                    memory_text=self.state.memory_text if self.state else "",
                    strategy_state=self.state.strategy_state if self.state else {},
                    tool_state=self.state.tool_state if self.state else {},
                    round_index=round_index,
                    parser_error=last_error,
                    agent_context_text=self._latest_agent_context_text,
                )
                artifact_logger.write_tool_synthesis_prompt_artifacts(
                    self.config.output_dir,
                    round_index=round_index,
                    attempt_index=attempt_index,
                    prompt_text=result.prompt_text,
                    request_summary={
                        "round_index": round_index,
                        "attempt_index": attempt_index,
                        **result.request_summary,
                    },
                )
                artifact_logger.write_generated_tool_request(
                    self.config.output_dir,
                    {
                        "round_index": round_index,
                        "attempt_index": attempt_index,
                        **result.request_summary,
                    },
                )
                artifact_logger.write_raw_llm_output(
                    self.config.output_dir,
                    round_index=round_index,
                    attempt_index=attempt_index,
                    raw_text=result.raw_text,
                )
                artifact_logger.write_parsed_tool_synthesis(
                    self.config.output_dir,
                    {
                        "round_index": round_index,
                        "attempt_index": attempt_index,
                        **result.parser_report,
                    },
                )
                if self.state is not None:
                    self.state.strategy_state["parser_valid_count"] = int(
                        self.state.strategy_state.get("parser_valid_count", 0)
                    ) + 1

                static_report = static_check_generated_tool_source(result.code)
                artifact_logger.write_static_check_report(
                    self.config.output_dir,
                    {
                        "round_index": round_index,
                        "attempt_index": attempt_index,
                        **static_report,
                    },
                )
                if not static_report["passed"]:
                    last_error = "static_check_failed:" + ",".join(static_report["violations"])
                    continue
                if self.state is not None:
                    self.state.strategy_state["static_check_pass_count"] = int(
                        self.state.strategy_state.get("static_check_pass_count", 0)
                    ) + 1

                sandbox_report = validate_tool_in_sandbox(
                    tool_source=result.code,
                    observed_df=observed_df,
                    candidate_df=candidate_df,
                    memory=self.state.memory_text if self.state else "",
                    tool_state=self.state.tool_state if self.state else {},
                )
                artifact_logger.write_sandbox_report(
                    self.config.output_dir,
                    {
                        "round_index": round_index,
                        "attempt_index": attempt_index,
                        **sandbox_report,
                    },
                )
                if not sandbox_report["passed"]:
                    last_error = (
                        "sandbox_failed:"
                        + str(sandbox_report.get("error_type"))
                        + ":"
                        + str(sandbox_report.get("error", ""))[:300]
                    )
                    continue
                if self.state is not None:
                    self.state.strategy_state["sandbox_pass_count"] = int(
                        self.state.strategy_state.get("sandbox_pass_count", 0)
                    ) + 1
                artifact_logger.write_generated_tool_source(
                    self.config.output_dir,
                    round_index=round_index,
                    tool_name=result.tool_name,
                    source=result.code,
                )
                return result.code, result.tool_name, attempt_index
            except ToolSynthesisParseError as exc:
                last_error = "parser_failed:" + str(exc)[:300]
                request_summary = getattr(exc, "request_summary", None)
                if isinstance(request_summary, dict):
                    prompt_text = getattr(exc, "prompt_text", "")
                    if prompt_text:
                        artifact_logger.write_tool_synthesis_prompt_artifacts(
                            self.config.output_dir,
                            round_index=round_index,
                            attempt_index=attempt_index,
                            prompt_text=str(prompt_text),
                            request_summary={
                                "round_index": round_index,
                                "attempt_index": attempt_index,
                                **request_summary,
                            },
                        )
                    artifact_logger.write_generated_tool_request(
                        self.config.output_dir,
                        {
                            "round_index": round_index,
                            "attempt_index": attempt_index,
                            **request_summary,
                        },
                    )
                raw_text = getattr(exc, "raw_text", "")
                if raw_text:
                    artifact_logger.write_raw_llm_output(
                        self.config.output_dir,
                        round_index=round_index,
                        attempt_index=attempt_index,
                        raw_text=str(raw_text),
                    )
                artifact_logger.write_parsed_tool_synthesis(
                    self.config.output_dir,
                    {
                        "round_index": round_index,
                        "attempt_index": attempt_index,
                        "passed": False,
                        "error": last_error,
                    },
                )
            except Exception as exc:
                detail = str(exc)[:300]
                last_error = "synthesis_failed:" + exc.__class__.__name__
                if detail:
                    last_error = last_error + ":" + detail
                artifact_logger.write_parsed_tool_synthesis(
                    self.config.output_dir,
                    {
                        "round_index": round_index,
                        "attempt_index": attempt_index,
                        "passed": False,
                        "error": last_error,
                    },
                )
        raise ValueError(last_error or "generated_tool_validation_failed")

    def _write_context_artifacts(
        self,
        *,
        tables: Any,
        replay_state: Any,
        round_index: int,
    ) -> dict[str, str]:
        if self.state is None:
            raise RuntimeError("Policy run state has not been initialized.")
        observed_df = build_observed_df_from_revealed_state(
            tables,
            replay_state,
            objective_name=self.config.target_column,
        )
        observed_evidence_text = build_observed_evidence_markdown(
            observed_df,
            round_index=round_index,
            objective_direction=self.config.objective_direction,
        )
        agent_context_text = build_agent_context_markdown(
            observed_evidence_text=observed_evidence_text,
            memory_text=self.state.memory_text,
            strategy_state=self.state.strategy_state,
            tool_state=self.state.tool_state,
            round_index=round_index,
        )
        artifact_logger.write_observed_evidence(
            self.config.output_dir,
            observed_evidence_text,
            round_index=round_index,
        )
        artifact_logger.write_agent_context(
            self.config.output_dir,
            agent_context_text,
            round_index=round_index,
        )
        hashes = {
            "observed_evidence_hash": sha256_text(observed_evidence_text),
            "agent_context_hash": sha256_text(agent_context_text),
        }
        self._latest_agent_context_text = agent_context_text
        self._latest_context_hashes = hashes
        return hashes


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _compact_patch_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "attempted",
        "synthesis_called",
        "verifier_called",
        "replacement_performed",
        "accepted",
        "reason",
        "old_tool_hash",
        "candidate_tool_hash",
        "active_tool_version_before",
        "active_tool_version_after",
    )
    compact = {key: outcome.get(key) for key in keep if key in outcome}
    from research_tool_agent_full_pool.decision_artifacts import assert_payload_public_safe

    assert_payload_public_safe(compact, label="patch_outcome")
    return compact


def _status_from_count(strategy_state: dict[str, Any], key: str) -> str:
    try:
        count = int(strategy_state.get(key, 0))
    except (TypeError, ValueError):
        count = 0
    return "passed" if count > 0 else "not_available"
