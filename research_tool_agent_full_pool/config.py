"""Configuration contract for the full-pool ResearchToolAgent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal
import json

from research_tool_agent_full_pool.initial_observed import validate_initial_observed_count

ResearchMode = Literal["none", "frozen_cards", "live_collect_then_freeze"]
PatchMode = Literal["disabled", "decision_only", "enabled"]
ToolSynthesisMode = Literal["single", "portfolio"]


@dataclass(frozen=True)
class ResearchToolFullPoolConfig:
    """Run configuration for the full-pool persistent generated-tool policy.

    Defaults are deliberately conservative for Step 0/1: fake mode, full-pool
    candidate views, no menu restriction, no repository reference BO calls or
    baseline artifacts, no candidate score CSV, no hidden outcomes, and no
    final LLM override. Public-safe BO-inspired generated scoring is allowed
    when computed only from observed/public-safe inputs.
    """

    decision_policy_name: str = "llm_research_tool_agent_full_pool"
    mode: Literal["fake", "api"] = "fake"
    candidate_source: str = "full_remaining_public_candidate_view"
    require_full_pool: bool = True
    allow_menu_restricted_selection: bool = False
    # Deprecated compatibility name: this means reference BO implementation or
    # comparator artifact access, not public-safe BO-inspired generated scoring.
    allow_bo_calls: bool = False
    allow_reference_bo_artifact_access: bool = False
    allow_candidate_scores_csv: bool = False
    allow_hidden_outcomes: bool = False
    use_final_llm_override: bool = False
    research_mode: ResearchMode = "none"
    patch_mode: PatchMode = "decision_only"
    output_dir: str = "results/llm_research_tool_agent/full_pool/"
    run_id: str = "full_pool_scaffold"
    seed: int = 0
    max_rounds: int = 3
    dataset_name: str = "MINERVA"
    target_column: str = "yield"
    objective_direction: Literal["maximize", "minimize"] = "maximize"
    initial_observed_count: int = 5
    batch_size: int = 1
    api_provider: str = "commonstack"
    api_key_path: str | None = None
    backup_api_key_path: str | None = None
    endpoint: str = "https://api.commonstack.ai/v1/chat/completions"
    model: str = "openai/gpt-5.4-2026-03-05"
    temperature: float = 0.0
    max_tokens: int = 4000
    timeout: float = 90.0
    repair_attempts: int = 1
    response_format_json_object: bool = True
    require_real_agent: bool = False
    require_real_live_research: bool = False
    tool_synthesis_mode: ToolSynthesisMode = "single"
    portfolio_repair_enabled: bool = False
    max_repair_attempts_per_candidate: int = 1

    def validate_contract_flags(self) -> None:
        """Fail fast if a caller opts into a behavior forbidden by the contract."""

        if not self.require_full_pool:
            raise ValueError("ResearchToolFullPoolConfig requires full-pool candidate scoring.")
        if self.allow_menu_restricted_selection:
            raise ValueError("Menu-restricted selection is forbidden for full-pool policy.")
        if self.allow_bo_calls or self.allow_reference_bo_artifact_access:
            raise ValueError(
                "Calls/imports/reads of the repository BO reference implementation or BO baseline artifacts "
                "are forbidden for ResearchToolAgent; public-safe BO-inspired generated scoring is allowed."
            )
        if self.allow_candidate_scores_csv:
            raise ValueError("Candidate score CSV artifacts are forbidden for ResearchToolAgent.")
        if self.allow_hidden_outcomes:
            raise ValueError("Hidden outcomes are forbidden in candidate views.")
        if self.use_final_llm_override:
            raise ValueError("Final LLM override is disabled for the first full-pool versions.")
        if self.research_mode not in {"none", "frozen_cards", "live_collect_then_freeze"}:
            raise ValueError(f"Unsupported research_mode: {self.research_mode!r}.")
        if self.patch_mode not in {"disabled", "decision_only", "enabled"}:
            raise ValueError(f"Unsupported patch_mode: {self.patch_mode!r}.")
        if self.tool_synthesis_mode not in {"single", "portfolio"}:
            raise ValueError(f"Unsupported tool_synthesis_mode: {self.tool_synthesis_mode!r}.")
        if int(self.max_repair_attempts_per_candidate) < 0:
            raise ValueError("max_repair_attempts_per_candidate must be nonnegative.")
        if self.patch_mode == "enabled":
            _validate_patch_replacement_available()
        if self.mode not in {"fake", "api"}:
            raise ValueError(f"Unsupported ResearchToolAgent mode: {self.mode!r}.")
        if int(self.batch_size) != 1:
            raise ValueError("Full-pool ResearchToolAgent smoke supports batch_size=1 only.")
        if self.mode == "api" and str(self.api_provider).lower() != "commonstack":
            raise ValueError("Step 2 API smoke supports api_provider='commonstack' only.")
        validate_initial_observed_count(self.initial_observed_count)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ResearchToolFullPoolConfig":
        """Load the Step 1 config JSON without enabling API behavior."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        api_enabled = bool(payload.get("llm_api_enabled"))
        default_run_id = "minerva_full_pool_api_smoke" if api_enabled else "minerva_full_pool_fake_smoke"
        mapped: dict[str, Any] = {
            "dataset_name": payload.get("dataset", payload.get("dataset_name", cls.dataset_name)),
            "decision_policy_name": payload.get(
                "decision_policy",
                payload.get("decision_policy_name", cls.decision_policy_name),
            ),
            "mode": "api" if api_enabled else payload.get("mode", "fake"),
            "candidate_source": payload.get("candidate_source", cls.candidate_source),
            "research_mode": payload.get("research_mode", cls.research_mode),
            "patch_mode": payload.get("patch_mode", cls.patch_mode),
            "output_dir": payload.get("output_dir", cls.output_dir),
            "run_id": payload.get("run_id", default_run_id),
            "seed": int(payload.get("seed", cls.seed)),
            "max_rounds": int(payload.get("max_rounds", cls.max_rounds)),
            "target_column": payload.get("target_column", cls.target_column),
            "objective_direction": payload.get("objective_direction", cls.objective_direction),
            "initial_observed_count": payload.get("initial_observed_count", cls.initial_observed_count),
            "batch_size": int(payload.get("batch_size", cls.batch_size)),
            "api_provider": payload.get("api_provider", cls.api_provider),
            "api_key_path": payload.get("api_key_path", cls.api_key_path),
            "backup_api_key_path": payload.get("backup_api_key_path", cls.backup_api_key_path),
            "endpoint": payload.get("endpoint", cls.endpoint),
            "model": payload.get("model", cls.model),
            "temperature": float(payload.get("temperature", cls.temperature)),
            "max_tokens": int(payload.get("max_tokens", cls.max_tokens)),
            "timeout": float(payload.get("timeout", cls.timeout)),
            "repair_attempts": int(payload.get("repair_attempts", cls.repair_attempts)),
            "response_format_json_object": bool(
                payload.get("response_format_json_object", cls.response_format_json_object)
            ),
            "require_real_agent": bool(payload.get("require_real_agent", cls.require_real_agent)),
            "require_real_live_research": bool(
                payload.get("require_real_live_research", cls.require_real_live_research)
            ),
            "tool_synthesis_mode": payload.get("tool_synthesis_mode", cls.tool_synthesis_mode),
            "portfolio_repair_enabled": bool(payload.get("portfolio_repair_enabled", cls.portfolio_repair_enabled)),
            "max_repair_attempts_per_candidate": int(
                payload.get("max_repair_attempts_per_candidate", cls.max_repair_attempts_per_candidate)
            ),
        }
        config = cls(**mapped)
        config.validate_contract_flags()
        if payload.get("policy_restrict_to_menu"):
            raise ValueError("policy_restrict_to_menu must be false for full-pool ResearchToolAgent.")
        if payload.get("candidate_menu_size") is not None:
            raise ValueError("candidate_menu_size must be null for full-pool ResearchToolAgent.")
        if str(payload.get("candidate_menu_mode", "none")).lower() not in {"none", "full_pool"}:
            raise ValueError("candidate_menu_mode must not restrict full-pool ResearchToolAgent selection.")
        return config


def _validate_patch_replacement_available() -> None:
    """Validate that Batch 3 patch replacement modules are importable."""

    try:
        from research_tool_agent_full_pool.tool_patch_synthesis import patch_tool_after_reveal  # noqa: F401
        from research_tool_agent_full_pool.tool_replacement_verifier import verify_tool_replacement  # noqa: F401
    except Exception as exc:
        raise ValueError("patch_mode='enabled' requires the Batch 3 patch verifier/replacement path.") from exc
