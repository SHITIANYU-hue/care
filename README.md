# CARE: Controlling LLM-Generated Policies through Auditable Review of Evidence in Scientific Experimentation

This repository contains the software supplement and datasets for the paper: **CARE: Controlling LLM-Generated Policies through Auditable Review of Evidence in Scientific Experimentation**.

It includes the finite-pool replay engine, public-information baselines, CARE controller, paper-facing run scripts, and aggregate paper result summaries.

## Overview

Granting Large Language Models (LLMs) direct control over costly, irreversible scientific experiments leads to unsafe exploration and unstable performance, but discarding LLM creativity entirely sacrifices significant optimization potential. We introduce **CARE** (Controlling LLM-Generated Policies through Auditable Review of Evidence in Scientific Experimentation), an auditable controller for high-throughput experimentation (HTE) optimization that keeps a non-LLM incumbent optimizer as the default action path while using LLMs to revise challenger ranking policies. 

Before each outcome is revealed, a public-evidence intervention gate compares the challenger with the incumbent. It authorizes the challenger's selection only when the evidence available before selection supports the change, with the decision recorded in the audit log. CARE outperforms all other evaluated methods on Minerva/Olympus and ChemLex benchmarks.

Read the full paper on arXiv: [arXiv:2606.14581](https://arxiv.org/abs/2606.14581)

## Repository Structure

- `configs/`: Paper-facing replay configurations for Minerva/Olympus Suzuki Coupling (i), ChemLex Acid-Amine Wetlab, and a no-API public replay check.
- `datasets/`: Dataset loaders and the data CSV files (`minerva/suzuki_i.csv` and `chemlex/acid_amine_wetlab.csv`).
- `replay_core/`: Finite-pool replay state and evaluator.
- `research_tool_agent_full_pool/`: Public-evidence-gated policy generation, validation, certificate, and audit-ledger code used by CARE.
- `evaluation/`: Matched replay policies, aggregate metrics, CARE ablations, and baseline adapters.
- `runners/`: Main paper and appendix run scripts.
- `paper_results_summary/`: Clean aggregate summaries corresponding to the tables in the paper.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Check

Dry-run the main replay manifest to verify the installation and data:

```bash
python runners/run_emnlp_full_experiments.py --dry-run --seeds 0 --max-rounds 1 --groups non_api_baselines
```

## Main Replay

The full paper suite uses API-backed language-generated challenger policies for CARE and LLM baselines. Set `COMMONSTACK_API_KEY` and, if needed, `COMMONSTACK_API_ENDPOINT`, then run:

```bash
bash runners/run_emnlp_full_suite_30x10.sh
```

The default output root is `results/care_main_replay_30x10`.

## Implementation Names Mapping

Some internal policy identifiers predate the paper terminology. The paper-facing mapping is:

- `true_self_evolving_api_care`: CARE.
- `true_self_evolving_api_care_no_adaptive_planner`: CARE without the scheduler.
- `true_self_evolving_api_care_no_certificate`: CARE without the public-evidence intervention gate.
- `true_self_evolving_api_care_no_residual_scout`: CARE without the trajectory-recovery challenger.
- `true_self_evolving_api_care_no_macro_scout`: CARE without schema-conditioned public-feature exploration.
- `public_expert_only_meta_controller`: public incumbent controller.
- `llm_only_self_evolving`: ungated LLM-evolving policy.
- `no_evolve_api_reuse`: non-evolving LLM policy.
- `lmabo_style_nearest_neighbor_llm_bo`: LMABO-style finite-pool LLM-BO baseline.

The implementation value `mode="fake"` means a deterministic no-API local controller used for public baselines and tests. It is not used to fabricate paper results.

## License and Intended Use

This software is provided for non-commercial research reproduction of the paper's finite-pool HTE replay experiments. The data files are governed by the upstream artifact licenses:
- Minerva/Olympus Suzuki Coupling (i): Creative Commons Attribution 4.0 International.
- ChemLex Acid-Amine Wetlab: Creative Commons Attribution Non Commercial 4.0 International.

## Citation

If you use this code or data in your research, please cite the paper:

```bibtex
@misc{liu2026care,
      title={CARE: Controlling LLM-Generated Policies through Auditable Review of Evidence in Scientific Experimentation}, 
      author={Guanyu Liu and Weiyi Kong and Zeyu Wang and Boer Zhang and Baiqing Li and Peiyu Zhang and Tianyu Shi},
      year={2026},
      eprint={2606.14581},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
