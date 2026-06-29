"""Run CARE public-evidence gate margin sensitivity experiments.

This runner keeps the EMNLP main replay protocol fixed and varies only
``care_certificate_margin`` for the CARE main policy.  It is intended for a
small appendix robustness table, not for re-running the full baseline suite.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


THREAD_CAP_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
)

for name in THREAD_CAP_ENV_VARS:
    os.environ.setdefault(name, "1")


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.self_evolving_proof_suite import (  # noqa: E402
    _aggregate,
    _augment_rows_with_oracle_metrics,
    _care_config_for_policy,
    _failed_run_summary,
    _parse_seeds,
    _run_true_self_evolving,
)
from evaluation.self_evolving_runner import _load_config, _load_tables  # noqa: E402
from replay_core.evaluator import OfflineEvaluator  # noqa: E402
from research_tool_agent_full_pool.harness.ledger import write_json  # noqa: E402
from research_tool_agent_full_pool.harness.orchestrator import SelfEvolvingConfig  # noqa: E402


DATASET_CONFIGS = {
    "minerva": "configs/minerva_care_replay.json",
    "chemlex": "configs/chemlex_care_replay.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CARE public-evidence gate margin sensitivity for EMNLP appendix robustness."
    )
    parser.add_argument("--datasets", default="minerva,chemlex", help="Comma-separated dataset keys: minerva,chemlex.")
    parser.add_argument("--seeds", default="0-9", help="Seed list/ranges, e.g. 0-9 or 0,1,2.")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument(
        "--margins",
        default="-0.10,0.10,0.20",
        help="Comma-separated margins to run. Default omits 0.00 and imports it from the main suite.",
    )
    parser.add_argument("--output-root", default="results/emnlp_gate_margin_sensitivity_10x10")
    parser.add_argument("--api-workers", type=int, default=8)
    parser.add_argument("--executor", choices=("thread", "process"), default="thread")
    parser.add_argument(
        "--default-care-root",
        default=str(REPO_ROOT / "results" / "care_main_replay_30x10"),
        help="Existing full-suite root used to import the default margin=0.00 CARE rows.",
    )
    parser.add_argument(
        "--no-include-default-margin",
        action="store_true",
        help="Do not import margin=0.00 rows from --default-care-root.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun even if matching run_row.json files exist.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-failed-runs",
        action="store_true",
        help="Write summaries even if some runs fail. By default the runner exits nonzero on failures.",
    )
    args = parser.parse_args()

    dataset_keys = _parse_csv(args.datasets)
    unknown = sorted(set(dataset_keys) - set(DATASET_CONFIGS))
    if unknown:
        raise SystemExit(f"Unknown datasets: {unknown}. Available: {sorted(DATASET_CONFIGS)}")

    seeds = _parse_seeds(args.seeds)
    margins = _parse_margins(args.margins)
    output_root = _resolve_path(args.output_root)
    default_root = _resolve_path(args.default_care_root)
    output_root.mkdir(parents=True, exist_ok=True)

    include_default_margin = not bool(args.no_include_default_margin)
    manifest: dict[str, Any] = {
        "status": "planned" if args.dry_run else "running",
        "created_at_unix": time.time(),
        "datasets": dataset_keys,
        "seeds": seeds,
        "max_rounds": int(args.max_rounds),
        "margins_to_run": margins,
        "include_default_margin": include_default_margin,
        "default_care_root": str(default_root),
        "output_root": str(output_root),
        "api_workers": int(args.api_workers),
        "executor": str(args.executor),
        "planned_runs": [],
    }

    for dataset in dataset_keys:
        for margin in margins:
            if include_default_margin and _is_zero_margin(margin):
                continue
            manifest["planned_runs"].append(
                {
                    "dataset": dataset,
                    "margin": float(margin),
                    "policy_name": _policy_name_for_margin(margin),
                    "run_count": len(seeds),
                }
            )
    write_json(output_root / "gate_margin_sensitivity_manifest.json", manifest)

    if args.dry_run:
        print(json.dumps(manifest, sort_keys=True, indent=2))
        return

    all_dataset_summaries: dict[str, Any] = {}
    for dataset in dataset_keys:
        config_path = DATASET_CONFIGS[dataset]
        dataset_root = output_root / dataset
        dataset_root.mkdir(parents=True, exist_ok=True)
        base = _load_config(config_path)

        rows: list[dict[str, Any]] = []
        imported_default_rows: list[dict[str, Any]] = []
        if include_default_margin:
            imported_default_rows = _load_default_margin_rows(
                default_root=default_root,
                dataset=dataset,
                seeds=seeds,
                max_rounds=int(args.max_rounds),
            )
            rows.extend(imported_default_rows)

        margins_to_run = [margin for margin in margins if not (include_default_margin and _is_zero_margin(margin))]
        rows.extend(
            _run_dataset_margins(
                base=base,
                dataset_root=dataset_root,
                seeds=seeds,
                max_rounds=int(args.max_rounds),
                margins=margins_to_run,
                workers=int(args.api_workers),
                executor_name=str(args.executor),
                force=bool(args.force),
            )
        )

        rows = _augment_rows_with_oracle_metrics(rows, base=base)
        round_rows = [
            {"policy_name": row["policy_name"], "seed": int(row["seed"]), **item}
            for row in rows
            for item in row.get("round_summaries", [])
        ]
        aggregate = _aggregate(rows)
        summary = {
            "status": "ok",
            "dataset": dataset,
            "config_path": config_path,
            "seeds": seeds,
            "max_rounds": int(args.max_rounds),
            "margins_requested": margins,
            "margins_run": margins_to_run,
            "imported_default_margin_count": len(imported_default_rows),
            "run_rows": rows,
            "round_rows": round_rows,
            "aggregate": aggregate,
            "appendix_table_rows": _appendix_table_rows(aggregate),
            "output_dir": str(dataset_root),
        }
        failed = [row for row in rows if row.get("status") != "ok"]
        if failed:
            summary["status"] = "failed"
            summary["failed_runs"] = [
                {
                    "policy_name": row.get("policy_name"),
                    "seed": row.get("seed"),
                    "error_type": row.get("error_type"),
                    "error": row.get("error"),
                }
                for row in failed
            ]
        write_json(dataset_root / "gate_margin_sensitivity_summary.json", summary)
        write_json(dataset_root / "run_rows.json", rows)
        write_json(dataset_root / "round_rows.json", round_rows)
        all_dataset_summaries[dataset] = {
            "status": summary["status"],
            "appendix_table_rows": summary["appendix_table_rows"],
            "failed_run_count": len(failed),
        }
        if failed and not args.allow_failed_runs:
            preview = [f"{row.get('policy_name')}_seed_{row.get('seed')}" for row in failed[:8]]
            raise SystemExit(f"{dataset} gate-margin sensitivity has failed runs: {preview}")

    manifest["status"] = "ok"
    manifest["finished_at_unix"] = time.time()
    manifest["dataset_summaries"] = all_dataset_summaries
    write_json(output_root / "gate_margin_sensitivity_manifest.json", manifest)
    print(json.dumps({"status": "ok", "output_root": str(output_root), "datasets": all_dataset_summaries}, indent=2))


def _run_dataset_margins(
    *,
    base: SelfEvolvingConfig,
    dataset_root: Path,
    seeds: list[int],
    max_rounds: int,
    margins: list[float],
    workers: int,
    executor_name: str,
    force: bool,
) -> list[dict[str, Any]]:
    if not margins:
        return []
    tasks = [(int(seed), float(margin)) for margin in margins for seed in seeds]
    if int(workers) <= 1:
        return [
            _run_one_margin_seed(
                base=base,
                dataset_root=dataset_root,
                seed=seed,
                max_rounds=max_rounds,
                margin=margin,
                force=force,
            )
            for seed, margin in tasks
        ]

    if executor_name not in {"thread", "process"}:
        raise ValueError(f"Unsupported executor: {executor_name!r}")
    executor_cls = ProcessPoolExecutor if executor_name == "process" else ThreadPoolExecutor
    rows_by_key: dict[tuple[float, int], dict[str, Any]] = {}
    with executor_cls(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(
                _run_one_margin_seed,
                base=base,
                dataset_root=dataset_root,
                seed=seed,
                max_rounds=max_rounds,
                margin=margin,
                force=force,
            ): (float(margin), int(seed))
            for seed, margin in tasks
        }
        for future in as_completed(futures):
            margin, seed = futures[future]
            rows_by_key[(margin, seed)] = future.result()
    return [rows_by_key[(float(margin), int(seed))] for margin in margins for seed in seeds]


def _run_one_margin_seed(
    *,
    base: SelfEvolvingConfig,
    dataset_root: Path,
    seed: int,
    max_rounds: int,
    margin: float,
    force: bool,
) -> dict[str, Any]:
    policy_name = _policy_name_for_margin(margin)
    run_dir = dataset_root / policy_name / f"{policy_name}_seed_{int(seed)}"
    existing_path = run_dir / "run_row.json"
    if existing_path.exists() and not force:
        existing = _read_json(existing_path)
        if _existing_row_matches(existing, seed=seed, max_rounds=max_rounds, margin=margin, policy_name=policy_name):
            return existing

    tables, table_source = _load_tables(base)
    evaluator = OfflineEvaluator.from_tables(tables)
    run_config = SelfEvolvingConfig(
        **{
            **base.__dict__,
            "run_id": f"{policy_name}_seed_{int(seed)}",
            "seed": int(seed),
            "max_rounds": int(max_rounds),
            "output_dir": str(run_dir),
            "resume_policy_state": False,
        }
    )
    care_config = _care_config_for_policy(config=run_config, policy="true_self_evolving_api_care")
    care_config = SelfEvolvingConfig(
        **{
            **care_config.__dict__,
            "care_certificate_margin": float(margin),
            "decision_policy_name": f"{base.decision_policy_name}_{policy_name}",
        }
    )
    try:
        row = _run_true_self_evolving(config=care_config, tables=tables, evaluator=evaluator)
    except Exception as exc:
        row = _failed_run_summary(config=care_config, exc=exc, output_dir=str(run_dir))
    row.update(
        {
            "policy_name": policy_name,
            "source_policy_name": "true_self_evolving_api_care",
            "seed": int(seed),
            "table_source": table_source,
            "sensitivity_axis": "care_certificate_margin",
            "care_certificate_margin": float(margin),
            "max_rounds": int(max_rounds),
        }
    )
    write_json(existing_path, row)
    write_json(run_dir / "round_summaries.json", row.get("round_summaries", []))
    return row


def _load_default_margin_rows(
    *,
    default_root: Path,
    dataset: str,
    seeds: list[int],
    max_rounds: int,
) -> list[dict[str, Any]]:
    path = default_root / dataset / "care_main" / "proof_summary.json"
    if not path.exists():
        print(f"[warn] default margin source not found: {path}", flush=True)
        return []
    payload = _read_json(path)
    wanted = set(int(seed) for seed in seeds)
    policy_name = _policy_name_for_margin(0.0)
    rows: list[dict[str, Any]] = []
    for row in payload.get("run_rows", []):
        if str(row.get("policy_name")) != "true_self_evolving_api_care":
            continue
        if int(row.get("seed", -1)) not in wanted:
            continue
        if int(row.get("selected_count", 0) or 0) != int(max_rounds):
            continue
        copied = dict(row)
        copied.update(
            {
                "policy_name": policy_name,
                "source_policy_name": "true_self_evolving_api_care",
                "sensitivity_axis": "care_certificate_margin",
                "care_certificate_margin": 0.0,
                "max_rounds": int(max_rounds),
                "imported_from_default_care_root": str(default_root),
            }
        )
        rows.append(copied)
    missing = sorted(wanted - {int(row.get("seed", -1)) for row in rows})
    if missing:
        print(f"[warn] default margin rows missing for {dataset} seeds: {missing}", flush=True)
    return sorted(rows, key=lambda row: int(row.get("seed", -1)))


def _appendix_table_rows(aggregate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in aggregate:
        policy_name = str(row.get("policy_name"))
        margin = _margin_from_policy_name(policy_name)
        rows.append(
            {
                "policy_name": policy_name,
                "margin": margin,
                "run_success_count": row.get("run_success_count"),
                "mean_final_best_yield": row.get("mean_final_best_yield"),
                "mean_auc_best_observed_yield": row.get("mean_auc_best_observed_yield"),
                "mean_simple_regret": row.get("mean_simple_regret"),
                "mean_normalized_simple_regret": row.get("mean_normalized_simple_regret"),
                "certified_interventions": row.get("total_care_certificate_selected_count"),
                "immediate_improvements": row.get("total_care_certificate_improvement_count"),
                "gate_pass_count": row.get("total_gate_pass_count"),
                "gate_reject_count": row.get("total_gate_reject_count"),
                "skill_synthesis_count": row.get("total_skill_synthesis_count"),
            }
        )
    return sorted(rows, key=lambda item: (float(item["margin"]) if item["margin"] is not None else 10**9))


def _existing_row_matches(
    row: dict[str, Any],
    *,
    seed: int,
    max_rounds: int,
    margin: float,
    policy_name: str,
) -> bool:
    if not row:
        return False
    if row.get("status") != "ok":
        return False
    if str(row.get("policy_name")) != str(policy_name):
        return False
    if int(row.get("seed", -1)) != int(seed):
        return False
    if int(row.get("selected_count", 0) or 0) != int(max_rounds):
        return False
    return abs(float(row.get("care_certificate_margin", 10**9)) - float(margin)) <= 1e-12


def _policy_name_for_margin(margin: float) -> str:
    sign = "m" if float(margin) < 0 else "p"
    value = f"{abs(float(margin)):.2f}".replace(".", "p")
    return f"care_margin_{sign}{value}"


def _margin_from_policy_name(policy_name: str) -> float | None:
    prefix = "care_margin_"
    if not policy_name.startswith(prefix):
        return None
    text = policy_name[len(prefix) :]
    if not text:
        return None
    sign = -1.0 if text[0] == "m" else 1.0
    value = text[1:].replace("p", ".")
    try:
        return sign * float(value)
    except ValueError:
        return None


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _parse_margins(text: str) -> list[float]:
    margins = [float(item) for item in _parse_csv(text)]
    return list(dict.fromkeys(margins))


def _is_zero_margin(margin: float) -> bool:
    return abs(float(margin)) <= 1e-12


def _resolve_path(text: str | Path) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    main()
