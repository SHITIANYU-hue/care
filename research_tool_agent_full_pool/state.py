"""Persistent run state contract for the full-pool ResearchToolAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any


@dataclass
class ResearchToolRunState:
    """Persistent state carried across sequential decision rounds.

    `memory_text`, `strategy_state`, `tool_state`, and the generated tool source
    are run-level artifacts. They must be updated only after an evaluator reveal
    and must never include hidden outcomes for unrevealed candidates.
    """

    run_id: str = ""
    memory_text: str = ""
    strategy_state: dict[str, Any] = field(default_factory=dict)
    tool_state: dict[str, Any] = field(default_factory=dict)
    generated_tool_source: str | None = None
    generated_tool_name: str | None = None
    tool_created_round: int | None = None
    generated_tool_create_count: int = 0
    generated_tool_reuse_count: int = 0
    generated_tool_repair_count: int = 0
    tool_patch_count: int = 0
    active_tool_version: int = 0
    active_tool_hash: str | None = None
    previous_tool_hashes: list[str] = field(default_factory=list)
    last_patch_decision: str | None = None
    last_patch_verification_result: dict[str, Any] | None = None
    patch_history: list[dict[str, Any]] = field(default_factory=list)
    memory_update_count: int = 0
    tool_state_update_count: int = 0
    round_summaries: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def initialize_run_state(
        cls,
        *,
        run_id: str,
        memory_text: str = "",
        strategy_state: dict[str, Any] | None = None,
        tool_state: dict[str, Any] | None = None,
    ) -> "ResearchToolRunState":
        """Create persistent state at run start."""

        return cls(
            run_id=run_id,
            memory_text=memory_text,
            strategy_state=dict(strategy_state or {}),
            tool_state=dict(tool_state or {}),
        )

    def record_tool_created(
        self,
        *,
        source: str,
        tool_name: str,
        round_index: int,
    ) -> None:
        """Record the first fake/generated optimizer tool for the run."""

        self.generated_tool_source = source
        self.generated_tool_name = tool_name
        self.tool_created_round = int(round_index)
        self.generated_tool_create_count += 1
        self.active_tool_version = max(int(self.active_tool_version), 1)
        self.active_tool_hash = _sha256_text(source)

    def record_tool_reused(self) -> None:
        """Record reuse of the persistent optimizer tool in a later round."""

        self.generated_tool_reuse_count += 1

    def sync_active_tool(
        self,
        *,
        source: str,
        tool_name: str | None,
        version: int | None = None,
    ) -> None:
        """Align active tool lineage with the skill that actually made a decision."""

        self.generated_tool_source = source
        if tool_name:
            self.generated_tool_name = tool_name
        if version is not None:
            self.active_tool_version = int(version)
        self.active_tool_hash = _sha256_text(source)

    def record_tool_repaired(self) -> None:
        """Record one repair attempt that produced an accepted generated tool."""

        self.generated_tool_repair_count += 1

    def record_tool_replaced(
        self,
        *,
        source: str,
        tool_name: str | None,
        round_index: int,
        patch_decision: str,
        verification_result: dict[str, Any],
        provenance: dict[str, Any] | None = None,
    ) -> None:
        """Atomically replace the active generated tool after verifier approval."""

        old_hash = self.active_tool_hash or _sha256_text(self.generated_tool_source or "")
        if old_hash:
            self.previous_tool_hashes.append(old_hash)
        self.generated_tool_source = source
        if tool_name:
            self.generated_tool_name = tool_name
        self.active_tool_version = max(int(self.active_tool_version), 1) + 1
        self.active_tool_hash = _sha256_text(source)
        self.tool_patch_count += 1
        self.last_patch_decision = str(patch_decision)
        self.last_patch_verification_result = dict(verification_result)
        self.patch_history.append(
            {
                "round_index": int(round_index),
                "accepted": True,
                "patch_decision": str(patch_decision),
                "old_tool_hash": old_hash,
                "new_tool_hash": self.active_tool_hash,
                "active_tool_version": self.active_tool_version,
                **dict(provenance or {}),
            }
        )

    def record_tool_patch_rejected(
        self,
        *,
        round_index: int,
        patch_decision: str,
        verification_result: dict[str, Any],
        provenance: dict[str, Any] | None = None,
    ) -> None:
        """Record a rejected patch attempt without changing the active tool."""

        self.last_patch_decision = str(patch_decision)
        self.last_patch_verification_result = dict(verification_result)
        self.patch_history.append(
            {
                "round_index": int(round_index),
                "accepted": False,
                "patch_decision": str(patch_decision),
                "active_tool_hash": self.active_tool_hash,
                "active_tool_version": self.active_tool_version,
                **dict(provenance or {}),
            }
        )

    def record_round_summary(self, summary: dict[str, Any]) -> None:
        """Append a public-safe round summary.

        Step 1 will define the precise schema. Callers must keep this summary
        free of hidden candidate outcomes and BO artifacts.
        """

        self.round_summaries.append(dict(summary))

    def update_after_reveal(
        self,
        *,
        memory_text: str,
        strategy_state_updates: dict[str, Any] | None = None,
        tool_state_updates: dict[str, Any] | None = None,
    ) -> None:
        """Update persistent state after evaluator reveal."""

        self.memory_text = memory_text
        if strategy_state_updates:
            self.strategy_state.update(strategy_state_updates)
        if tool_state_updates:
            self.tool_state.update(tool_state_updates)
        self.memory_update_count += 1
        self.tool_state_update_count += 1


def initialize_run_state(
    *,
    run_id: str,
    memory_text: str = "",
    strategy_state: dict[str, Any] | None = None,
    tool_state: dict[str, Any] | None = None,
) -> ResearchToolRunState:
    """Functional helper for run-state initialization."""

    return ResearchToolRunState.initialize_run_state(
        run_id=run_id,
        memory_text=memory_text,
        strategy_state=strategy_state,
        tool_state=tool_state,
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()
