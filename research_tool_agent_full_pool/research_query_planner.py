"""Public-safe live research query planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

from research_tool_agent_full_pool.research_query_safety import validate_live_research_query


QUERY_PLANNER_VERSION = "batch2.live_query_planner.v1"
DEFAULT_MAX_QUERIES = 6


@dataclass(frozen=True)
class ResearchQueryPlan:
    planner_version: str
    queries: list[str]
    used_llm: bool
    fallback_used: bool
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def propose_live_research_queries(
    *,
    dataset_profile: dict[str, Any],
    task_domain_summary: str = "",
    prior_research_context: dict[str, Any] | None = None,
    memory_summary: str = "",
    client_or_provider: Any | None = None,
    max_queries: int = DEFAULT_MAX_QUERIES,
) -> ResearchQueryPlan:
    """Propose method/domain queries without asking for benchmark answers."""

    if client_or_provider is not None and hasattr(client_or_provider, "create_tool"):
        try:
            queries = _propose_with_llm(
                dataset_profile=dataset_profile,
                task_domain_summary=task_domain_summary,
                prior_research_context=prior_research_context or {},
                memory_summary=memory_summary,
                client=client_or_provider,
                max_queries=max_queries,
            )
            safe_queries = _dedupe_public_safe_queries(queries, dataset_profile, max_queries=max_queries)
            if safe_queries:
                return ResearchQueryPlan(
                    planner_version=QUERY_PLANNER_VERSION,
                    queries=safe_queries,
                    used_llm=True,
                    fallback_used=False,
                    source="configured_llm_client",
                )
        except Exception:
            pass

    return ResearchQueryPlan(
        planner_version=QUERY_PLANNER_VERSION,
        queries=_fallback_queries(dataset_profile, task_domain_summary, max_queries=max_queries),
        used_llm=False,
        fallback_used=True,
        source="deterministic_local_fallback",
    )


def _propose_with_llm(
    *,
    dataset_profile: dict[str, Any],
    task_domain_summary: str,
    prior_research_context: dict[str, Any],
    memory_summary: str,
    client: Any,
    max_queries: int,
) -> list[str]:
    prompt = {
        "task": "propose_public_safe_method_domain_research_queries",
        "planner_version": QUERY_PLANNER_VERSION,
        "max_queries": int(max_queries),
        "dataset_profile": _compact_profile(dataset_profile),
        "task_domain_summary": task_domain_summary,
        "prior_research_context_summary": prior_research_context,
        "memory_summary": memory_summary[:800],
        "instructions": [
            "Return method or domain research queries useful for designing a public-safe optimizer tool.",
            "Do not ask for a specific implementation from this repository.",
            "Do not seek exact dataset answers, benchmark tables, target values, candidate IDs, oracle ranks, BO baseline outputs, or answer keys.",
            "Prefer general scientific optimization, active learning, surrogate modeling, acquisition concepts, and domain priors.",
            "Return exactly JSON with a queries list of strings.",
        ],
        "output_schema": {"queries": ["short public-safe query"]},
    }
    raw = client.create_tool(
        messages=[
            {
                "role": "system",
                "content": "Return strict JSON only. Do not include secrets or benchmark answers.",
            },
            {"role": "user", "content": json.dumps(prompt, sort_keys=True, ensure_ascii=True)},
        ]
    )
    payload = json.loads(str(raw).strip())
    queries = payload.get("queries", [])
    if not isinstance(queries, list):
        return []
    return [str(query) for query in queries]


def _fallback_queries(
    dataset_profile: dict[str, Any],
    task_domain_summary: str,
    *,
    max_queries: int,
) -> list[str]:
    profile_text = json.dumps(dataset_profile, sort_keys=True, ensure_ascii=True).lower()
    task_text = f"{task_domain_summary} {profile_text}".lower()
    queries = [
        "Bayesian optimization for reaction optimization",
        "active learning for high-throughput experimentation",
        "surrogate modeling for small-sample chemistry optimization",
        "acquisition functions for experimental design",
        "uncertainty-aware candidate selection in finite-pool optimization",
        "regularized ensemble surrogate models for small tabular design spaces",
    ]
    if "suzuki" in task_text:
        queries.insert(4, "Suzuki coupling optimization priors")
    if "reaction" not in task_text and "chemistry" not in task_text:
        queries = [
            "Bayesian optimization for finite-pool experimental design",
            "active learning for small-sample tabular optimization",
            "surrogate modeling for sparse observed data",
            "acquisition functions for exploration exploitation balance",
            "uncertainty-aware candidate selection in finite-pool optimization",
            "regularized ensemble surrogate models for small design spaces",
        ]
    return _dedupe_public_safe_queries(queries, dataset_profile, max_queries=max_queries)


def _dedupe_public_safe_queries(
    queries: list[str],
    dataset_profile: dict[str, Any],
    *,
    max_queries: int,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for query in queries:
        text = " ".join(str(query).split())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        safety = validate_live_research_query(text, dataset_profile=dataset_profile)
        if not safety.passed:
            continue
        seen.add(key)
        result.append(safety.sanitized_query)
        if len(result) >= int(max_queries):
            break
    return result


def _compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    keep_keys = (
        "profile_version",
        "dataset_label",
        "task_summary",
        "objective",
        "observed_count",
        "remaining_candidate_count",
        "feature_columns",
        "candidate_schema_summary",
        "observed_y_summary",
    )
    return {key: profile[key] for key in keep_keys if key in profile}
