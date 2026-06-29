"""Run the EMNLP main-paper HTE experiment suite.

The default suite intentionally excludes internal historical variants such as
old true_self_evolving_api and CARE log-only. It runs paper-facing baselines,
CARE main, and CARE component ablations in separate scheduling groups so CPU
baselines and API-heavy policies can use different concurrency settings.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.self_evolving_proof_suite import (  # noqa: E402
    EMNLP_API_ABLATION_POLICIES,
    EMNLP_API_BASELINE_POLICIES,
    EMNLP_API_METHOD_POLICIES,
    EMNLP_NON_API_BASELINE_POLICIES,
    _aggregate,
    _augment_rows_with_oracle_metrics,
    _care_checks,
    _pairwise_vs,
    _parse_seeds,
    _proof_checks,
    run_proof_suite,
)
from evaluation.self_evolving_runner import _load_config  # noqa: E402
from research_tool_agent_full_pool.harness.ledger import write_json  # noqa: E402


DATASET_CONFIGS = {
    "minerva": "configs/minerva_care_replay.json",
    "chemlex": "configs/chemlex_care_replay.json",
}

EXCLUDED_PAPER_POLICIES = {"true_self_evolving_api", "true_self_evolving_api_care_log_only"}
PUBLIC_CAPACITY_ABLATION_POLICIES = (
    "public_expert_only_meta_controller_4experts",
    "public_expert_only_meta_controller_6experts",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EMNLP main-paper CARE HTE experiments.")
    parser.add_argument("--datasets", default="minerva,chemlex", help="Comma-separated dataset keys: minerva,chemlex.")
    parser.add_argument("--seeds", default="0-29", help="Seed list/ranges, e.g. 0-29 or 0,1,2.")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--output-root", default="results/emnlp_care_full_suite")
    parser.add_argument(
        "--groups",
        default="non_api_baselines,api_baselines,public_capacity_ablation,care_main,care_ablations",
        help=(
            "Comma-separated groups: non_api_baselines,api_baselines,public_capacity_ablation,"
            "care_main,care_ablations."
        ),
    )
    parser.add_argument("--baseline-workers", type=int, default=28)
    parser.add_argument("--api-workers", type=int, default=20)
    parser.add_argument("--baseline-executor", choices=("thread", "process"), default="process")
    parser.add_argument("--api-executor", choices=("thread", "process"), default="thread")
    parser.add_argument("--resume", action="store_true", help="Strictly validate and skip completed matching groups.")
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="With --resume, rerun only failed policy/seed pairs from an existing group summary before continuing.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-failed-runs", action="store_true", help="Deprecated: paper runner fails by default.")
    parser.add_argument("--allow-failed-runs", action="store_true", help="Write summaries even if some runs fail.")
    args = parser.parse_args()
    fail_on_failed_runs = not bool(args.allow_failed_runs)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds)
    dataset_keys = _parse_csv(args.datasets)
    group_keys = _parse_csv(args.groups)
    group_specs = _group_specs(
        baseline_workers=int(args.baseline_workers),
        api_workers=int(args.api_workers),
        baseline_executor=str(args.baseline_executor),
        api_executor=str(args.api_executor),
    )
    unknown_datasets = sorted(set(dataset_keys) - set(DATASET_CONFIGS))
    if unknown_datasets:
        raise SystemExit(f"Unknown datasets: {unknown_datasets}. Available: {sorted(DATASET_CONFIGS)}")
    unknown_groups = sorted(set(group_keys) - set(group_specs))
    if unknown_groups:
        raise SystemExit(f"Unknown groups: {unknown_groups}. Available: {sorted(group_specs)}")

    manifest = {
        "status": "planned" if args.dry_run else "running",
        "created_at_unix": time.time(),
        "datasets": dataset_keys,
        "seeds": seeds,
        "max_rounds": int(args.max_rounds),
        "groups": group_keys,
        "excluded_paper_policies": sorted(EXCLUDED_PAPER_POLICIES),
        "fail_on_failed_runs": fail_on_failed_runs,
        "output_root": str(output_root),
        "commands": [],
    }

    for dataset in dataset_keys:
        config_path = DATASET_CONFIGS[dataset]
        for group in group_keys:
            spec = group_specs[group]
            policies = list(spec["policies"])
            forbidden = sorted(set(policies) & EXCLUDED_PAPER_POLICIES)
            if forbidden:
                raise RuntimeError(f"Paper-facing group {group!r} contains excluded policies: {forbidden}")
            group_output = output_root / dataset / group
            command = _command_preview(
                config_path=config_path,
                seeds=args.seeds,
                max_rounds=int(args.max_rounds),
                output_dir=group_output,
                policies=policies,
                workers=int(spec["workers"]),
                executor=str(spec["executor"]),
            )
            manifest["commands"].append({"dataset": dataset, "group": group, "command": command})
            print(command, flush=True)
            if args.dry_run:
                continue
            if args.resume and (group_output / "proof_summary.json").exists():
                payload = _read_json(group_output / "proof_summary.json")
                problems = _validate_summary_payload(
                    payload,
                    config_path=config_path,
                    seeds=seeds,
                    max_rounds=int(args.max_rounds),
                    policies=policies,
                    executor=str(spec["executor"]),
                    require_success=False,
                )
                if problems:
                    raise SystemExit(
                        f"Cannot resume {dataset}/{group}; existing proof_summary.json is stale or invalid: {problems}"
                    )
                failed_keys = _failed_run_keys(payload)
                if failed_keys and args.rerun_failed:
                    payload = _rerun_failed_group(
                        config_path=config_path,
                        group_output=group_output,
                        existing_payload=payload,
                        seeds=seeds,
                        max_rounds=int(args.max_rounds),
                        policies=policies,
                        workers=int(spec["workers"]),
                        executor=str(spec["executor"]),
                    )
                    failed_keys = _failed_run_keys(payload)
                if failed_keys and fail_on_failed_runs:
                    failed_preview = [f"{policy}_seed_{seed}" for seed, policy in failed_keys[:8]]
                    raise SystemExit(
                        f"Cannot resume {dataset}/{group}; existing proof_summary.json has failed runs: {failed_preview}. "
                        "Use --rerun-failed to repair them, or --allow-failed-runs only for a temporary incomplete summary."
                    )
                print(f"[resume] skip existing {group_output}", flush=True)
                continue
            result = run_proof_suite(
                config_path,
                seeds=seeds,
                max_rounds=int(args.max_rounds),
                output_dir=group_output,
                policies=policies,
                parallel_workers=int(spec["workers"]),
                executor_type=str(spec["executor"]),
            )
            failed = [row for row in result.get("run_rows", []) if row.get("status") != "ok"]
            if failed and fail_on_failed_runs:
                raise SystemExit(f"{dataset}/{group} has failed runs: {[row.get('run_id') for row in failed[:8]]}")
        if not args.dry_run:
            _write_combined_dataset_summary(
                output_root / dataset,
                group_specs=group_specs,
                seeds=seeds,
                max_rounds=int(args.max_rounds),
                fail_on_failed_runs=fail_on_failed_runs,
            )

    manifest["status"] = "dry_run" if args.dry_run else "ok"
    manifest["finished_at_unix"] = time.time()
    write_json(output_root / "emnlp_full_experiment_manifest.json", manifest)


def _group_specs(*, baseline_workers: int, api_workers: int, baseline_executor: str, api_executor: str) -> dict[str, dict[str, Any]]:
    return {
        "non_api_baselines": {
            "policies": EMNLP_NON_API_BASELINE_POLICIES,
            "workers": int(baseline_workers),
            "executor": baseline_executor,
        },
        "public_capacity_ablation": {
            "policies": PUBLIC_CAPACITY_ABLATION_POLICIES,
            "workers": int(baseline_workers),
            "executor": baseline_executor,
        },
        "api_baselines": {
            "policies": EMNLP_API_BASELINE_POLICIES,
            "workers": int(api_workers),
            "executor": api_executor,
        },
        "care_main": {
            "policies": EMNLP_API_METHOD_POLICIES,
            "workers": int(api_workers),
            "executor": api_executor,
        },
        "care_ablations": {
            "policies": EMNLP_API_ABLATION_POLICIES,
            "workers": int(api_workers),
            "executor": api_executor,
        },
    }


def _write_combined_dataset_summary(
    dataset_root: Path,
    *,
    group_specs: dict[str, dict[str, Any]],
    seeds: list[int],
    max_rounds: int,
    fail_on_failed_runs: bool,
) -> None:
    run_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    group_summaries: dict[str, Any] = {}
    validation_problems: dict[str, list[str]] = {}
    for group, spec in group_specs.items():
        path = dataset_root / group / "proof_summary.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        problems = _validate_summary_payload(
            payload,
            config_path=None,
            seeds=seeds,
            max_rounds=int(max_rounds),
            policies=list(spec["policies"]),
            executor=str(spec["executor"]),
            require_success=fail_on_failed_runs,
        )
        if problems:
            validation_problems[group] = problems
            continue
        group_summaries[group] = {
            "output_dir": str(path.parent),
            "policies": payload.get("policies", []),
            "aggregate": payload.get("aggregate", []),
            "proof_checks": payload.get("proof_checks", {}),
            "care_checks": payload.get("care_checks", {}),
        }
        run_rows.extend(payload.get("run_rows", []))
        round_rows.extend(payload.get("round_rows", []))
    if validation_problems and fail_on_failed_runs:
        raise SystemExit(f"Combined summary validation failed under {dataset_root}: {validation_problems}")
    policies = sorted({str(row.get("policy_name")) for row in run_rows})
    failed = [row for row in run_rows if row.get("status") != "ok"]
    status = "incomplete" if validation_problems or failed else "ok"
    summary = {
        "status": status,
        "dataset_root": str(dataset_root),
        "groups": group_summaries,
        "validation_problems": validation_problems,
        "policies": policies,
        "aggregate": _aggregate(run_rows),
        "pairwise_vs_care_main_final": _pairwise_vs(run_rows, reference_policy="true_self_evolving_api_care"),
        "pairwise_vs_care_main_auc": _pairwise_vs(
            run_rows,
            reference_policy="true_self_evolving_api_care",
            metric="auc_best_observed_yield",
        ),
        "paper_checks": _paper_checks(run_rows),
    }
    write_json(dataset_root / "combined_run_rows.json", run_rows)
    write_json(dataset_root / "combined_round_rows.json", round_rows)
    write_json(dataset_root / "combined_summary.json", summary)


def _rerun_failed_group(
    *,
    config_path: str,
    group_output: Path,
    existing_payload: dict[str, Any],
    seeds: list[int],
    max_rounds: int,
    policies: list[str],
    workers: int,
    executor: str,
) -> dict[str, Any]:
    failed_keys = _failed_run_keys(existing_payload)
    if not failed_keys:
        return existing_payload
    failed_seed_set = {seed for seed, _policy in failed_keys}
    failed_policy_set = {policy for _seed, policy in failed_keys}
    rerun_seeds = [seed for seed in seeds if seed in failed_seed_set]
    rerun_policies = [policy for policy in policies if policy in failed_policy_set]
    rerun_root = group_output / "_rerun_failed" / time.strftime("%Y%m%d_%H%M%S")
    print(
        f"[resume] rerun {len(failed_keys)} failed run(s) under {group_output}: "
        f"seeds={rerun_seeds}, policies={rerun_policies}",
        flush=True,
    )
    rerun_payload = run_proof_suite(
        config_path,
        seeds=rerun_seeds,
        max_rounds=int(max_rounds),
        output_dir=rerun_root,
        policies=rerun_policies,
        parallel_workers=max(1, min(int(workers), max(1, len(rerun_seeds) * len(rerun_policies)))),
        executor_type=str(executor),
    )
    merged_rows = _merge_rerun_rows(
        existing_rows=list(existing_payload.get("run_rows", [])),
        rerun_rows=list(rerun_payload.get("run_rows", [])),
        failed_keys=failed_keys,
        seeds=seeds,
        policies=policies,
    )
    repaired = _build_group_summary_payload(
        config_path=config_path,
        output_dir=group_output,
        seeds=seeds,
        max_rounds=int(max_rounds),
        policies=policies,
        workers=int(workers),
        executor=str(executor),
        run_rows=merged_rows,
    )
    repaired_failed = _failed_run_keys(repaired)
    if repaired_failed:
        print(
            f"[resume] rerun completed but {len(repaired_failed)} run(s) still failed: "
            f"{[f'{policy}_seed_{seed}' for seed, policy in repaired_failed[:8]]}",
            flush=True,
        )
    else:
        print(f"[resume] repaired failed runs for {group_output}", flush=True)
    return repaired


def _failed_run_keys(payload: dict[str, Any]) -> list[tuple[int, str]]:
    keys: list[tuple[int, str]] = []
    for row in payload.get("run_rows", []):
        if row.get("status") == "ok":
            continue
        try:
            seed = int(row.get("seed"))
        except (TypeError, ValueError):
            continue
        policy = str(row.get("policy_name") or "")
        if policy:
            keys.append((seed, policy))
    return keys


def _merge_rerun_rows(
    *,
    existing_rows: list[dict[str, Any]],
    rerun_rows: list[dict[str, Any]],
    failed_keys: list[tuple[int, str]],
    seeds: list[int],
    policies: list[str],
) -> list[dict[str, Any]]:
    failed_key_set = {(int(seed), str(policy)) for seed, policy in failed_keys}
    existing_by_key = {
        (int(row.get("seed")), str(row.get("policy_name"))): row
        for row in existing_rows
        if row.get("seed") is not None and row.get("policy_name") is not None
    }
    rerun_by_key = {
        (int(row.get("seed")), str(row.get("policy_name"))): row
        for row in rerun_rows
        if row.get("seed") is not None and row.get("policy_name") is not None
    }
    merged: list[dict[str, Any]] = []
    missing: list[tuple[int, str]] = []
    for seed in seeds:
        for policy in policies:
            key = (int(seed), str(policy))
            if key in failed_key_set and key in rerun_by_key:
                merged.append(rerun_by_key[key])
            elif key in existing_by_key:
                merged.append(existing_by_key[key])
            else:
                missing.append(key)
    if missing:
        raise RuntimeError(f"Cannot merge rerun rows; missing policy/seed rows: {missing[:8]}")
    return merged


def _build_group_summary_payload(
    *,
    config_path: str,
    output_dir: Path,
    seeds: list[int],
    max_rounds: int,
    policies: list[str],
    workers: int,
    executor: str,
    run_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    base = _load_config(config_path)
    run_rows = _augment_rows_with_oracle_metrics(run_rows, base=base)
    round_rows: list[dict[str, Any]] = []
    for row in run_rows:
        try:
            seed = int(row.get("seed"))
        except (TypeError, ValueError):
            seed = -1
        round_rows.extend(
            {"policy_name": row.get("policy_name"), "seed": seed, **item}
            for item in row.get("round_summaries", [])
        )
    result = {
        "status": "ok",
        "config_path": str(config_path),
        "seeds": [int(seed) for seed in seeds],
        "max_rounds": int(max_rounds),
        "parallel_workers": int(workers),
        "executor_type": str(executor),
        "policies": list(policies),
        "run_rows": run_rows,
        "round_rows": round_rows,
        "aggregate": _aggregate(run_rows),
        "pairwise_vs_true_self_evolving": _pairwise_vs(run_rows, reference_policy="true_self_evolving_api"),
        "pairwise_auc_vs_true_self_evolving": _pairwise_vs(
            run_rows,
            reference_policy="true_self_evolving_api",
            metric="auc_best_observed_yield",
        ),
        "pairwise_vs_care_log_only": (
            _pairwise_vs(run_rows, reference_policy="true_self_evolving_api_care_log_only")
            if any(row.get("policy_name") == "true_self_evolving_api_care_log_only" for row in run_rows)
            else []
        ),
        "pairwise_auc_vs_care_log_only": (
            _pairwise_vs(
                run_rows,
                reference_policy="true_self_evolving_api_care_log_only",
                metric="auc_best_observed_yield",
            )
            if any(row.get("policy_name") == "true_self_evolving_api_care_log_only" for row in run_rows)
            else []
        ),
        "proof_checks": _proof_checks(run_rows, selected_policies=policies, max_rounds=max_rounds),
        "care_checks": _care_checks(run_rows, selected_policies=policies, max_rounds=max_rounds),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "proof_summary.json", result)
    write_json(output_dir / "run_rows.json", run_rows)
    write_json(output_dir / "round_rows.json", round_rows)
    return result


def _paper_checks(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    policies = {str(row.get("policy_name")) for row in run_rows}
    seeds = sorted({int(row.get("seed")) for row in run_rows if row.get("seed") is not None})
    expected_policies = (
        set(EMNLP_NON_API_BASELINE_POLICIES)
        | set(EMNLP_API_BASELINE_POLICIES)
        | set(EMNLP_API_METHOD_POLICIES)
        | set(EMNLP_API_ABLATION_POLICIES)
    )
    missing_expected = sorted(expected_policies - policies)
    success_by_policy = {
        policy: sum(1 for row in run_rows if row.get("policy_name") == policy and row.get("status") == "ok")
        for policy in sorted(policies)
    }
    required = {
        "care_main": "true_self_evolving_api_care" in policies,
        "public_only": "public_expert_only_meta_controller" in policies,
        "no_evolve_llm": "no_evolve_api_reuse" in policies,
        "random": "random_full_pool" in policies,
        "stratified_random": "stratified_random_public" in policies,
        "classical_bo": {"classical_bo_gp_ei", "classical_bo_gp_ucb", "tpe_style_bo"}.issubset(policies),
        "strong_public_bo": {"smac_style_rf_ei", "botorch_style_gp_logei", "chemistry_descriptor_bo"}.issubset(policies),
        "chemistry_bo": {
            "edbo_style_descriptor_gp_ei",
            "gryffin_style_categorical_bo",
            "baybe_bofire_style_mixed_bo",
        }.issubset(policies),
        "care_component_ablations": {
            "true_self_evolving_api_care_no_adaptive_planner",
            "true_self_evolving_api_care_no_certificate",
            "true_self_evolving_api_care_no_residual_scout",
            "true_self_evolving_api_care_no_macro_scout",
        }.issubset(policies),
        "llm_only_self_evolving": "llm_only_self_evolving" in policies,
        "lmabo_style_llm_bo": "lmabo_style_nearest_neighbor_llm_bo" in policies,
        "bo_like_surrogate": "bo_like_surrogate" in policies,
        "categorical_empirical_bayes_ucb": "categorical_empirical_bayes_ucb" in policies,
        "fixed_public_heuristic": "fixed_public_heuristic" in policies,
        "all_default_paper_policies": not missing_expected,
        "excluded_internal_versions_absent": not bool(policies & EXCLUDED_PAPER_POLICIES),
    }
    full_success_by_policy = {
        policy: count == len(seeds) and len(seeds) > 0 for policy, count in success_by_policy.items()
    }
    return {
        "seed_count": len(seeds),
        "success_count_by_policy": success_by_policy,
        "full_success_by_policy": full_success_by_policy,
        "expected_policies": sorted(expected_policies),
        "missing_expected_policies": missing_expected,
        "all_present": all(required.values()),
        "required_groups": required,
        "all_runs_successful": all(full_success_by_policy.values()) if full_success_by_policy else False,
    }


def _command_preview(
    *,
    config_path: str,
    seeds: str,
    max_rounds: int,
    output_dir: Path,
    policies: list[str],
    workers: int,
    executor: str,
) -> str:
    env = " ".join(f"{name}=1" for name in THREAD_CAP_ENV_VARS)
    argv = [
        sys.executable,
        "runners/run_self_evolving_proof_suite.py",
        config_path,
        "--seeds",
        str(seeds),
        "--max-rounds",
        str(int(max_rounds)),
        "--policies",
        ",".join(policies),
        "--parallel-workers",
        str(int(workers)),
        "--executor",
        str(executor),
        "--output-dir",
        str(output_dir),
    ]
    return f"cd {shlex.quote(str(REPO_ROOT))} && {env} " + " ".join(shlex.quote(item) for item in argv)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_summary_payload(
    payload: dict[str, Any],
    *,
    config_path: str | None,
    seeds: list[int],
    max_rounds: int,
    policies: list[str],
    executor: str,
    require_success: bool,
) -> list[str]:
    problems: list[str] = []
    payload_policies = [str(policy) for policy in payload.get("policies", [])]
    run_policies = {str(row.get("policy_name")) for row in payload.get("run_rows", [])}
    forbidden = sorted((set(payload_policies) | run_policies) & EXCLUDED_PAPER_POLICIES)
    if forbidden:
        problems.append(f"contains excluded paper policies: {forbidden}")
    if config_path is not None and str(payload.get("config_path")) != str(config_path):
        problems.append(f"config_path mismatch: {payload.get('config_path')!r} != {config_path!r}")
    if [int(seed) for seed in payload.get("seeds", [])] != [int(seed) for seed in seeds]:
        problems.append("seeds mismatch")
    if int(payload.get("max_rounds", -1) or -1) != int(max_rounds):
        problems.append("max_rounds mismatch")
    if payload_policies != [str(policy) for policy in policies]:
        problems.append("policy list mismatch")
    if str(payload.get("executor_type", "")) != str(executor):
        problems.append("executor mismatch")
    expected_run_count = len(seeds) * len(policies)
    run_rows = list(payload.get("run_rows", []))
    if len(run_rows) != expected_run_count:
        problems.append(f"run_rows count mismatch: {len(run_rows)} != {expected_run_count}")
    if require_success:
        failed = [str(row.get("run_id") or f"{row.get('policy_name')}_seed_{row.get('seed')}") for row in run_rows if row.get("status") != "ok"]
        if failed:
            problems.append(f"failed runs present: {failed[:8]}")
    return problems


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


if __name__ == "__main__":
    main()
