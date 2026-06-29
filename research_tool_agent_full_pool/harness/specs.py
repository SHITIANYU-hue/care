"""Typed artifacts for the self-evolving policy-editing harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Literal


TaskAction = Literal["create_skill", "patch_skill", "reuse_active_skill"]
SkillFamily = Literal["ranker", "constraint", "exploration", "calibrator", "fallback"]
RiskBudget = Literal["low", "medium", "high"]

ALLOWED_TASK_ACTIONS: tuple[str, ...] = ("create_skill", "patch_skill", "reuse_active_skill")
ALLOWED_SKILL_FAMILIES: tuple[str, ...] = (
    "ranker",
    "constraint",
    "exploration",
    "calibrator",
    "fallback",
)
ALLOWED_RISK_BUDGETS: tuple[str, ...] = ("low", "medium", "high")
DEFAULT_REQUIRED_CHECKS: tuple[str, ...] = (
    "static",
    "sandbox",
    "full_pool",
    "row_order",
)


@dataclass(frozen=True)
class TaskPlan:
    """Planner output that constrains how the LLM may edit policy code."""

    action: TaskAction
    skill_family: SkillFamily
    objective: str
    target_skill_id: str | None = None
    risk_budget: RiskBudget = "low"
    required_checks: list[str] = field(default_factory=lambda: list(DEFAULT_REQUIRED_CHECKS))
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.action not in ALLOWED_TASK_ACTIONS:
            raise ValueError(f"Unsupported task action: {self.action!r}.")
        if self.skill_family not in ALLOWED_SKILL_FAMILIES:
            raise ValueError(f"Unsupported skill family: {self.skill_family!r}.")
        if self.risk_budget not in ALLOWED_RISK_BUDGETS:
            raise ValueError(f"Unsupported risk budget: {self.risk_budget!r}.")
        if not isinstance(self.objective, str) or not self.objective.strip():
            raise ValueError("TaskPlan.objective must be a non-empty string.")
        if not isinstance(self.required_checks, list) or not all(
            isinstance(item, str) and item.strip() for item in self.required_checks
        ):
            raise ValueError("TaskPlan.required_checks must be a list of strings.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskPlan":
        allowed = {"action", "skill_family", "objective", "target_skill_id", "risk_budget", "required_checks", "rationale"}
        extra = set(payload) - allowed
        if extra:
            raise ValueError(f"Unsupported TaskPlan fields: {sorted(extra)}")
        return cls(
            action=str(payload.get("action", "reuse_active_skill")),  # type: ignore[arg-type]
            skill_family=str(payload.get("skill_family", "ranker")),  # type: ignore[arg-type]
            objective=str(payload.get("objective", "")).strip(),
            target_skill_id=(
                None if payload.get("target_skill_id") in (None, "") else str(payload.get("target_skill_id"))
            ),
            risk_budget=str(payload.get("risk_budget", "low")),  # type: ignore[arg-type]
            required_checks=[str(item) for item in payload.get("required_checks", DEFAULT_REQUIRED_CHECKS)],
            rationale=str(payload.get("rationale", ""))[:800],
        )


@dataclass(frozen=True)
class SkillArtifact:
    """Versioned editable policy artifact that compiles to rank_candidates."""

    skill_id: str
    version: int
    family: SkillFamily
    source: str
    parent_skill_id: str | None = None
    parent_version: int | None = None
    created_round: int = 0
    objective: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.skill_id:
            raise ValueError("SkillArtifact.skill_id must be non-empty.")
        if int(self.version) <= 0:
            raise ValueError("SkillArtifact.version must be positive.")
        if self.family not in ALLOWED_SKILL_FAMILIES:
            raise ValueError(f"Unsupported skill family: {self.family!r}.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("SkillArtifact.source must be non-empty.")
        if "def rank_candidates" not in self.source:
            raise ValueError("SkillArtifact.source must define rank_candidates.")

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()

    @property
    def artifact_key(self) -> str:
        return f"{self.skill_id}@v{int(self.version):03d}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_hash"] = self.source_hash
        payload["artifact_key"] = self.artifact_key
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillArtifact":
        return cls(
            skill_id=str(payload["skill_id"]),
            version=int(payload["version"]),
            family=str(payload["family"]),  # type: ignore[arg-type]
            source=str(payload["source"]),
            parent_skill_id=payload.get("parent_skill_id"),
            parent_version=payload.get("parent_version"),
            created_round=int(payload.get("created_round", 0)),
            objective=str(payload.get("objective", "")),
            provenance=dict(payload.get("provenance", {})),
        )


@dataclass(frozen=True)
class SkillPatch:
    """Public-safe record of one proposed skill edit."""

    patch_id: str
    task_plan: TaskPlan
    candidate_skill: SkillArtifact
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "task_plan": self.task_plan.to_dict(),
            "candidate_skill": self.candidate_skill.to_dict(),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class GateReport:
    """Deployment gate result for a candidate skill."""

    passed: bool
    deployable: bool
    failed_checks: list[str]
    warning_checks: list[str]
    checks: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RewardRecord:
    """Reveal-time reward computed only from selected/revealed public evidence."""

    round_index: int
    selected_candidate_id: str
    revealed_y: float | None
    previous_best_y: float | None
    current_best_y: float | None
    delta_best_y: float
    tool_output_valid: bool
    row_order_stable: bool
    fallback_used: bool
    patch_deployed: bool
    patch_rejected: bool
    complexity_penalty: float
    total_reward: float
    components: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyState:
    """Persistent public-safe state for policy editing."""

    active_skill_id: str | None = None
    active_version: int | None = None
    skill_history: list[dict[str, Any]] = field(default_factory=list)
    reward_history: list[dict[str, Any]] = field(default_factory=list)
    gate_history: list[dict[str, Any]] = field(default_factory=list)
    fallback_count: int = 0
    rollback_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PolicyState":
        return cls(
            active_skill_id=payload.get("active_skill_id"),
            active_version=payload.get("active_version"),
            skill_history=list(payload.get("skill_history", [])),
            reward_history=list(payload.get("reward_history", [])),
            gate_history=list(payload.get("gate_history", [])),
            fallback_count=int(payload.get("fallback_count", 0)),
            rollback_count=int(payload.get("rollback_count", 0)),
        )


def compact_json(payload: Any) -> str:
    """Stable compact JSON for prompt snippets and hashes."""

    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
