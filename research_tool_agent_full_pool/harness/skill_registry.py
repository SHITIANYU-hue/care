"""Versioned skill registry and rollback support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_tool_agent_full_pool.harness.ledger import HarnessLedger, write_json
from research_tool_agent_full_pool.harness.specs import PolicyState, RewardRecord, SkillArtifact


class SkillRegistry:
    """Persist skill artifacts and active-policy metadata."""

    def __init__(self, output_dir: str | Path, *, ledger: HarnessLedger | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.skills_dir = self.output_dir / "self_evolving_skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / "self_evolving_policy_state.json"
        self.ledger = ledger or HarnessLedger(self.output_dir)
        self.policy_state = PolicyState()
        if self.state_path.exists():
            import json

            self.policy_state = PolicyState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))

    def register_skill(self, skill: SkillArtifact, *, activate: bool = False, gate_report: dict[str, Any] | None = None) -> None:
        """Write a skill artifact and optionally make it active."""

        skill = self.ensure_unique_version(skill)
        skill_path = self.skill_path(skill.skill_id, skill.version)
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(skill.source, encoding="utf-8")
        metadata = skill.to_dict()
        metadata.pop("source", None)
        if gate_report is not None:
            metadata["gate_report"] = gate_report
        write_json(skill_path.with_suffix(".json"), metadata, sanitize=False)
        self.policy_state.skill_history.append(
            {
                "skill_id": skill.skill_id,
                "version": int(skill.version),
                "family": skill.family,
                "source_hash": skill.source_hash,
                "created_round": int(skill.created_round),
                "activated": bool(activate),
            }
        )
        if activate:
            self.activate_skill(skill)
        else:
            self._persist_state()
        self.ledger.append(
            "skill_registered",
            {
                "skill": metadata,
                "activated": bool(activate),
            },
        )

    def activate_skill(self, skill: SkillArtifact) -> None:
        for row in self.policy_state.skill_history:
            if (
                row.get("skill_id") == skill.skill_id
                and int(row.get("version", -1) or -1) == int(skill.version)
                and str(row.get("source_hash", "")) == skill.source_hash
            ):
                row["activated"] = True
        self.policy_state.active_skill_id = skill.skill_id
        self.policy_state.active_version = int(skill.version)
        self._persist_state()
        self.ledger.append(
            "skill_activated",
            {
                "skill_id": skill.skill_id,
                "version": int(skill.version),
                "source_hash": skill.source_hash,
            },
        )

    def active_skill(self) -> SkillArtifact | None:
        if not self.policy_state.active_skill_id or not self.policy_state.active_version:
            return None
        return self.load_skill(self.policy_state.active_skill_id, int(self.policy_state.active_version))

    def load_skill(self, skill_id: str, version: int) -> SkillArtifact:
        import json

        source_path = self.skill_path(skill_id, version)
        metadata_path = source_path.with_suffix(".json")
        if not source_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Missing skill artifact: {skill_id}@v{int(version):03d}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["source"] = source_path.read_text(encoding="utf-8")
        skill = SkillArtifact.from_dict(metadata)
        expected_hash = str(metadata.get("source_hash", "") or "")
        if expected_hash and expected_hash != skill.source_hash:
            raise ValueError(
                f"Skill artifact hash mismatch for {skill_id}@v{int(version):03d}."
            )
        return skill

    def ensure_unique_version(self, skill: SkillArtifact) -> SkillArtifact:
        """Return a skill with a version that will not overwrite an existing artifact."""

        path = self.skill_path(skill.skill_id, int(skill.version))
        metadata_path = path.with_suffix(".json")
        if not path.exists() and not metadata_path.exists():
            return skill
        existing_source = path.read_text(encoding="utf-8") if path.exists() else ""
        if existing_source == skill.source:
            return skill
        next_version = self.next_version(skill.skill_id)
        return SkillArtifact(
            skill_id=skill.skill_id,
            version=int(next_version),
            family=skill.family,
            source=skill.source,
            parent_skill_id=skill.parent_skill_id,
            parent_version=skill.parent_version,
            created_round=skill.created_round,
            objective=skill.objective,
            provenance={
                **dict(skill.provenance),
                "host_version_reassigned": {
                    "requested_version": int(skill.version),
                    "assigned_version": int(next_version),
                    "reason": "existing_artifact_path_with_different_source",
                },
            },
        )

    def next_version(self, skill_id: str) -> int:
        existing = sorted(self.skills_dir.glob(f"{_safe_name(skill_id)}_v*.py"))
        if not existing:
            return 1
        versions: list[int] = []
        for path in existing:
            stem = path.stem.rsplit("_v", 1)[-1]
            try:
                versions.append(int(stem))
            except ValueError:
                continue
        return max(versions, default=0) + 1

    def rollback_to_previous(self) -> SkillArtifact | None:
        active_id = self.policy_state.active_skill_id
        active_version = self.policy_state.active_version
        candidates = [
            row
            for row in self.policy_state.skill_history
            if row.get("activated") and (row.get("skill_id"), row.get("version")) != (active_id, active_version)
        ]
        if not candidates:
            return None
        previous = candidates[-1]
        skill = self.load_skill(str(previous["skill_id"]), int(previous["version"]))
        self.policy_state.rollback_count += 1
        self.activate_skill(skill)
        self.ledger.append(
            "skill_rollback",
            {"skill_id": skill.skill_id, "version": int(skill.version), "rollback_count": self.policy_state.rollback_count},
        )
        return skill

    def record_gate(self, gate_report: dict[str, Any]) -> None:
        self.policy_state.gate_history.append(dict(gate_report))
        self._persist_state()
        self.ledger.append("gate_report", dict(gate_report))

    def record_reward(self, reward: RewardRecord) -> None:
        self.policy_state.reward_history.append(reward.to_dict())
        self._persist_state()
        self.ledger.append("reward_record", reward.to_dict())

    def reset_policy_state(self) -> None:
        """Start a fresh run without deleting prior artifact files."""

        self.policy_state = PolicyState()
        self._persist_state()
        self.ledger.append("policy_state_reset", {})

    def skill_path(self, skill_id: str, version: int) -> Path:
        return self.skills_dir / f"{_safe_name(skill_id)}_v{int(version):03d}.py"

    def _persist_state(self) -> None:
        write_json(self.state_path, self.policy_state.to_dict(), sanitize=False)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))[:120]
