"""Conservative mechanism classifier for generated portfolio tools."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import io
import math
import re
import tokenize
from typing import Any


OPTIMIZER_CLASSES = {
    "static_predictor_or_ranker",
    "heuristic_optimizer",
    "uncertainty_aware_optimizer",
    "bo_like_optimizer",
    "hybrid_sequential_optimizer",
    "invalid_or_unsafe",
    "unknown",
}

RISK_LEVELS = {"low", "medium", "high", "unknown"}

SKLEARN_ESTIMATOR_CLASSES = {
    "Ridge",
    "RandomForestRegressor",
    "KNeighborsRegressor",
    "StandardScaler",
    "SimpleImputer",
    "Pipeline",
}

EVIDENCE_KEYS = (
    "surrogate_terms",
    "uncertainty_terms",
    "exploration_terms",
    "diversity_terms",
    "acquisition_terms",
    "state_update_terms",
    "fallback_terms",
    "mixed_variable_terms",
    "diagnostic_terms",
    "leakage_terms",
    "sklearn_terms",
    "handwritten_estimator_terms",
)


@dataclass
class ToolMechanismClassification:
    """JSON-serializable mechanism classification for one tool version."""

    tool_id: str | None
    optimizer_class: str
    is_static_ranker: bool
    uses_observed_y: bool
    uses_candidate_public_features: bool
    uses_surrogate: bool
    uses_uncertainty: bool
    uses_exploration: bool
    uses_diversity: bool
    uses_acquisition_logic: bool
    uses_state_update: bool
    uses_small_n_fallback: bool
    supports_mixed_vars: bool
    supports_batch: bool
    uses_diagnostics: bool
    uses_sklearn_import: bool
    uses_sklearn_estimator_class: bool
    uses_handwritten_estimator_like_logic: bool
    mentions_estimator_in_design_only: bool
    fake_uncertainty_risk: str
    over_exploration_risk: str
    score_scale_risk: str
    leakage_risk: str
    complexity_risk: str
    classification_confidence: str
    evidence: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict for JSONL/report writers."""

        return asdict(self)


@dataclass(frozen=True)
class _AstFacts:
    parse_ok: bool
    syntax_error: str | None
    names: set[str]
    attrs: set[str]
    calls: set[str]
    strings: set[str]
    imports: set[str]
    sklearn_imports: set[str]
    sklearn_estimator_classes: set[str]
    assigned_target_names: set[str]
    assignment_exprs: list[tuple[set[str], str]]
    score_exprs: list[str]
    function_names: set[str]
    compare_exprs: list[str]
    return_dicts: list[ast.Dict]
    tool_state_mutations: set[str]
    try_count: int
    broad_exception_count: int
    class_count: int


def classify_tool_mechanism(
    tool_source: str,
    tool_id: str | None = None,
    design_metadata: dict | None = None,
    raw_response_metadata: dict | None = None,
    verifier_result: dict | None = None,
    observed_quality: dict | None = None,
    score_summary: dict | None = None,
) -> ToolMechanismClassification:
    """Classify the mechanism used by one generated, repaired, or selected tool.

    The classifier intentionally avoids trusting family names or design labels.
    Source code is the primary signal; metadata and quality diagnostics only
    raise confidence, risks, or design-only mention flags.
    """

    source = str(tool_source or "")
    code_text = _code_without_comments(source)
    lowered_code = code_text.lower()
    comments = _comment_text(source).lower()
    facts = _ast_facts(source)
    evidence: dict[str, list[str]] = {key: [] for key in EVIDENCE_KEYS}
    warnings: list[str] = []

    leakage_terms = _leakage_categories(source, design_metadata, raw_response_metadata)
    evidence["leakage_terms"] = leakage_terms

    uses_observed_y = _contains_observed_y(facts, lowered_code)
    uses_candidate_public_features = _uses_candidate_features(facts, lowered_code)
    supports_batch = _supports_batch_ranking(facts, lowered_code)

    uses_sklearn_import = bool(facts.sklearn_imports)
    sklearn_terms = sorted(facts.sklearn_imports)
    uses_sklearn_estimator_class = bool(facts.sklearn_estimator_classes)
    sklearn_terms.extend(sorted(facts.sklearn_estimator_classes))
    evidence["sklearn_terms"] = _unique_limited(sklearn_terms)

    handwritten, handwritten_terms = _detect_handwritten_estimator_like_logic(facts, lowered_code)
    evidence["handwritten_estimator_terms"] = handwritten_terms
    uses_handwritten_estimator_like_logic = handwritten

    surrogate_terms = _matched_identifier_terms(
        facts,
        {
            "surrogate",
            "model",
            "prediction",
            "predicted",
            "predicted_y",
            "mean_prediction",
            "local_model",
            "fit",
            "predict",
            "regression",
            "regressor",
            "ridge",
            "neighbor",
            "nearest",
            "ensemble",
        },
    )
    if uses_sklearn_estimator_class or uses_handwritten_estimator_like_logic:
        surrogate_terms.append("implemented_estimator")
    uses_surrogate = bool(surrogate_terms) and (uses_observed_y or uses_sklearn_estimator_class or uses_handwritten_estimator_like_logic)
    evidence["surrogate_terms"] = _unique_limited(surrogate_terms)

    metadata_text = _metadata_text({"design": design_metadata or {}, "raw_response": raw_response_metadata or {}}).lower()
    metadata_estimator_mentions = _metadata_mentions_estimator(metadata_text)
    mentions_estimator_in_design_only = bool(metadata_estimator_mentions) and not (
        uses_sklearn_import or uses_sklearn_estimator_class or uses_handwritten_estimator_like_logic
    )

    uncertainty = _detect_uncertainty(facts, lowered_code)
    uses_uncertainty = uncertainty["uses_uncertainty"]
    evidence["uncertainty_terms"] = uncertainty["terms"]

    exploration_terms = _detect_score_linked_terms(
        facts,
        lowered_code,
        {
            "exploration",
            "explore",
            "novelty",
            "novel",
            "distance",
            "density",
            "sparsity",
            "coverage",
            "epsilon",
            "random",
            "unseen",
            "space_filling",
            "spacefill",
        },
    )
    uses_exploration = bool(exploration_terms) or bool(uncertainty["exploration_proxy_terms"])
    evidence["exploration_terms"] = _unique_limited(exploration_terms + uncertainty["exploration_proxy_terms"])

    diversity_terms = _detect_score_linked_terms(
        facts,
        lowered_code,
        {
            "diversity",
            "diverse",
            "novelty",
            "coverage",
            "cluster",
            "maxmin",
            "distance",
            "pairwise",
            "hamming",
            "cosine",
            "spread",
        },
    )
    uses_diversity = bool(diversity_terms)
    evidence["diversity_terms"] = _unique_limited(diversity_terms)

    acquisition = _detect_acquisition_logic(facts, lowered_code)
    uses_acquisition_logic = acquisition["uses_acquisition_logic"]
    evidence["acquisition_terms"] = acquisition["terms"]
    if uses_acquisition_logic:
        uses_uncertainty = uses_uncertainty or acquisition["implies_uncertainty"]
        uses_exploration = uses_exploration or acquisition["implies_exploration"]
        evidence["uncertainty_terms"] = _unique_limited(evidence["uncertainty_terms"] + acquisition["uncertainty_terms"])
        evidence["exploration_terms"] = _unique_limited(evidence["exploration_terms"] + acquisition["exploration_terms"])

    uses_state_update, state_terms = _detect_state_update(facts)
    evidence["state_update_terms"] = state_terms

    uses_small_n_fallback, fallback_terms = _detect_small_n_fallback(facts, lowered_code)
    evidence["fallback_terms"] = fallback_terms

    supports_mixed_vars, mixed_terms = _detect_mixed_variable_support(facts, lowered_code)
    evidence["mixed_variable_terms"] = mixed_terms

    uses_diagnostics, diagnostic_terms = _detect_diagnostics(facts, lowered_code)
    evidence["diagnostic_terms"] = diagnostic_terms

    fake_uncertainty_risk = _fake_uncertainty_risk(
        facts=facts,
        lowered_code=lowered_code,
        comments=comments,
        uses_uncertainty=uses_uncertainty,
        uncertainty_terms=evidence["uncertainty_terms"],
    )
    over_exploration_risk = _over_exploration_risk(
        uses_exploration=uses_exploration,
        uses_diversity=uses_diversity,
        uses_uncertainty=uses_uncertainty,
        uses_surrogate=uses_surrogate,
        uses_observed_y=uses_observed_y,
        evidence=evidence,
        observed_quality=observed_quality,
        score_summary=score_summary,
    )
    score_scale_risk = _score_scale_risk(observed_quality=observed_quality, score_summary=score_summary)
    leakage_risk = "high" if leakage_terms else "low"
    complexity_risk = _complexity_risk(source, facts)

    if not facts.parse_ok:
        warnings.append("Source could not be parsed as Python AST.")
    if leakage_terms:
        warnings.append("Non-public leakage-like source or metadata references were detected.")
    if fake_uncertainty_risk == "high":
        warnings.append("Uncertainty wording appears without an observed-data uncertainty basis.")
    if mentions_estimator_in_design_only:
        warnings.append("Estimator-like method is mentioned in metadata but not implemented in source.")

    verifier_failed = _verifier_failed(verifier_result)
    if verifier_failed:
        warnings.append("Verifier result marks this tool version as failed or undeployable.")

    non_static_mechanism = any(
        (
            uses_uncertainty,
            uses_exploration,
            uses_diversity,
            uses_acquisition_logic,
            uses_state_update,
            uses_small_n_fallback,
            uses_diagnostics,
        )
    )
    is_static_ranker = (
        bool(source.strip())
        and facts.parse_ok
        and not verifier_failed
        and leakage_risk != "high"
        and not non_static_mechanism
        and (uses_observed_y or uses_candidate_public_features or uses_surrogate)
    )

    optimizer_class = _optimizer_class(
        facts=facts,
        verifier_failed=verifier_failed,
        leakage_risk=leakage_risk,
        is_static_ranker=is_static_ranker,
        uses_surrogate=uses_surrogate,
        uses_uncertainty=uses_uncertainty,
        uses_exploration=uses_exploration,
        uses_diversity=uses_diversity,
        uses_acquisition_logic=uses_acquisition_logic,
        uses_small_n_fallback=uses_small_n_fallback,
        supports_mixed_vars=supports_mixed_vars,
        supports_batch=supports_batch,
        uses_diagnostics=uses_diagnostics,
        uses_state_update=uses_state_update,
    )
    confidence = _classification_confidence(
        facts=facts,
        optimizer_class=optimizer_class,
        evidence=evidence,
        verifier_failed=verifier_failed,
        leakage_risk=leakage_risk,
    )

    return ToolMechanismClassification(
        tool_id=str(tool_id) if tool_id is not None else None,
        optimizer_class=optimizer_class,
        is_static_ranker=bool(is_static_ranker),
        uses_observed_y=bool(uses_observed_y),
        uses_candidate_public_features=bool(uses_candidate_public_features),
        uses_surrogate=bool(uses_surrogate),
        uses_uncertainty=bool(uses_uncertainty),
        uses_exploration=bool(uses_exploration),
        uses_diversity=bool(uses_diversity),
        uses_acquisition_logic=bool(uses_acquisition_logic),
        uses_state_update=bool(uses_state_update),
        uses_small_n_fallback=bool(uses_small_n_fallback),
        supports_mixed_vars=bool(supports_mixed_vars),
        supports_batch=bool(supports_batch),
        uses_diagnostics=bool(uses_diagnostics),
        uses_sklearn_import=bool(uses_sklearn_import),
        uses_sklearn_estimator_class=bool(uses_sklearn_estimator_class),
        uses_handwritten_estimator_like_logic=bool(uses_handwritten_estimator_like_logic),
        mentions_estimator_in_design_only=bool(mentions_estimator_in_design_only),
        fake_uncertainty_risk=fake_uncertainty_risk,
        over_exploration_risk=over_exploration_risk,
        score_scale_risk=score_scale_risk,
        leakage_risk=leakage_risk,
        complexity_risk=complexity_risk,
        classification_confidence=confidence,
        evidence={key: _unique_limited(evidence.get(key, [])) for key in EVIDENCE_KEYS},
        warnings=_unique_limited(warnings, limit=12),
    )


def _ast_facts(source: str) -> _AstFacts:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return _AstFacts(
            parse_ok=False,
            syntax_error=str(exc),
            names=set(),
            attrs=set(),
            calls=set(),
            strings=set(),
            imports=set(),
            sklearn_imports=set(),
            sklearn_estimator_classes=set(),
            assigned_target_names=set(),
            assignment_exprs=[],
            score_exprs=[],
            function_names=set(),
            compare_exprs=[],
            return_dicts=[],
            tool_state_mutations=set(),
            try_count=0,
            broad_exception_count=0,
            class_count=0,
        )

    names: set[str] = set()
    attrs: set[str] = set()
    calls: set[str] = set()
    strings: set[str] = set()
    imports: set[str] = set()
    sklearn_imports: set[str] = set()
    sklearn_classes: set[str] = set()
    assigned: set[str] = set()
    assignment_exprs: list[tuple[set[str], str]] = []
    score_exprs: list[str] = []
    functions: set[str] = set()
    compare_exprs: list[str] = []
    return_dicts: list[ast.Dict] = []
    state_mutations: set[str] = set()
    broad_exception_count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
            if node.id in SKLEARN_ESTIMATOR_CLASSES:
                sklearn_classes.add(node.id)
        elif isinstance(node, ast.Attribute):
            attrs.add(node.attr)
            if node.attr in SKLEARN_ESTIMATOR_CLASSES:
                sklearn_classes.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)
        elif isinstance(node, ast.FunctionDef):
            functions.add(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            functions.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = str(alias.name).split(".")[0]
                imports.add(str(alias.name))
                if root == "sklearn":
                    sklearn_imports.add(str(alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = str(node.module or "")
            imports.add(module)
            if module == "sklearn" or module.startswith("sklearn."):
                sklearn_imports.add(module)
                for alias in node.names:
                    if alias.name in SKLEARN_ESTIMATOR_CLASSES:
                        sklearn_classes.add(alias.name)
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name:
                calls.add(name)
            if name in SKLEARN_ESTIMATOR_CLASSES:
                sklearn_classes.add(name)
            if isinstance(node.func, ast.Attribute):
                receiver = _name_text(node.func.value).lower()
                if receiver == "tool_state" and node.func.attr in {"update", "setdefault", "copy", "pop"}:
                    state_mutations.add(f"tool_state.{node.func.attr}")
        elif isinstance(node, ast.Assign):
            targets = _assigned_names(node.targets)
            assigned.update(targets)
            expr = _unparse_lower(node.value)
            assignment_exprs.append((targets, expr))
            if any(_is_score_name(name) for name in targets):
                score_exprs.append(expr)
            if any(name.lower() == "tool_state" or "state" in name.lower() for name in targets):
                if "tool_state" in expr or "{" in expr:
                    state_mutations.update(name for name in targets if "state" in name.lower())
        elif isinstance(node, ast.AnnAssign):
            targets = _assigned_names([node.target])
            assigned.update(targets)
            expr = _unparse_lower(node.value) if node.value is not None else ""
            assignment_exprs.append((targets, expr))
            if any(_is_score_name(name) for name in targets):
                score_exprs.append(expr)
        elif isinstance(node, ast.AugAssign):
            targets = _assigned_names([node.target])
            assigned.update(targets)
            expr = _unparse_lower(node.value)
            assignment_exprs.append((targets, expr))
            if any(_is_score_name(name) for name in targets):
                score_exprs.append(expr)
            if any(_name_text(node.target).lower().startswith("tool_state") for _ in [node.target]):
                state_mutations.add("tool_state_augassign")
        elif isinstance(node, ast.Compare):
            compare_exprs.append(_unparse_lower(node))
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            return_dicts.append(node.value)
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and key.value == "tool_state":
                    if isinstance(value, ast.Dict) and value.keys:
                        state_mutations.add("return_tool_state_dict")
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "score":
                    score_exprs.append(_unparse_lower(value))
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None:
                broad_exception_count += 1
            elif isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
                broad_exception_count += 1

    return _AstFacts(
        parse_ok=True,
        syntax_error=None,
        names=names,
        attrs=attrs,
        calls=calls,
        strings=strings,
        imports=imports,
        sklearn_imports=sklearn_imports,
        sklearn_estimator_classes=sklearn_classes,
        assigned_target_names=assigned,
        assignment_exprs=assignment_exprs,
        score_exprs=score_exprs,
        function_names=functions,
        compare_exprs=compare_exprs,
        return_dicts=return_dicts,
        tool_state_mutations=state_mutations,
        try_count=sum(1 for node in ast.walk(tree) if isinstance(node, ast.Try)),
        broad_exception_count=broad_exception_count,
        class_count=sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef)),
    )


def _contains_observed_y(facts: _AstFacts, lowered_code: str) -> bool:
    return "observed_y" in lowered_code or any(str(value).lower() == "observed_y" for value in facts.strings)


def _uses_candidate_features(facts: _AstFacts, lowered_code: str) -> bool:
    if "candidate_df" not in lowered_code and "candidate" not in facts.names:
        return False
    feature_signals = {
        "row.get",
        "candidate_df[",
        "select_dtypes",
        "get_dummies",
        "to_numeric",
        ".columns",
        "feature",
        "numeric",
        "categorical",
        "astype",
    }
    if any(signal in lowered_code for signal in feature_signals):
        return True
    return any(name.lower().startswith(("feature", "x_", "xobs", "xcand")) for name in facts.names)


def _supports_batch_ranking(facts: _AstFacts, lowered_code: str) -> bool:
    return (
        "ranked_candidates" in lowered_code
        and (
            "iterrows" in facts.calls
            or "itertuples" in facts.calls
            or "len(candidate_df)" in lowered_code
            or "for" in lowered_code and "candidate_df" in lowered_code
        )
    )


def _detect_handwritten_estimator_like_logic(facts: _AstFacts, lowered_code: str) -> tuple[bool, list[str]]:
    terms: list[str] = []
    call_or_attr = {item.lower() for item in facts.calls | facts.attrs | facts.names | facts.function_names}

    linalg_terms = {"solve", "lstsq", "pinv", "inv", "matmul", "dot"}
    if any(term in call_or_attr for term in linalg_terms) and _contains_observed_y(facts, lowered_code):
        terms.append("linear_algebra_fit")
    if any("fit" in name and "predict" in name for name in call_or_attr) and _contains_observed_y(facts, lowered_code):
        terms.append("custom_fit_predict_function")
    elif (
        any(name.startswith("fit") for name in call_or_attr)
        and any(name.startswith("predict") for name in call_or_attr)
        and _contains_observed_y(facts, lowered_code)
    ):
        terms.append("custom_fit_predict_pair")
    if any(term in call_or_attr for term in {"coef", "coefficients", "beta", "weights"}) and "observed_y" in lowered_code:
        if any(token in lowered_code for token in ("ridge", "regress", "surrogate", "model", "predict")):
            terms.append("custom_regression_coefficients")
    if any(term in lowered_code for term in ("local similarity", "similarity_weight", "neighbor_weight", "nearest")):
        if "observed_y" in lowered_code and any(term in lowered_code for term in ("weighted", "mean", "predict", "score")):
            terms.append("local_similarity_regressor")
    if any(term in lowered_code for term in ("leave_one_out", "loo", "heldout")) and "observed_y" in lowered_code:
        terms.append("leave_one_out_estimator")
    if any(term in lowered_code for term in ("bootstrap", "bagging", "ensemble")) and "observed_y" in lowered_code:
        terms.append("resampling_estimator")
    return bool(terms), _unique_limited(terms)


def _matched_identifier_terms(facts: _AstFacts, terms: set[str]) -> list[str]:
    identifiers = {item.lower() for item in facts.names | facts.attrs | facts.calls | facts.function_names}
    hits = []
    for term in terms:
        if any(term in identifier for identifier in identifiers):
            hits.append(term)
    return _unique_limited(hits)


def _detect_uncertainty(facts: _AstFacts, lowered_code: str) -> dict[str, Any]:
    terms: list[str] = []
    proxy_terms: list[str] = []
    identifiers = {item.lower() for item in facts.names | facts.attrs | facts.calls | facts.function_names}
    score_text = " ".join(facts.score_exprs)
    assignment_text = " ".join(expr for _targets, expr in facts.assignment_exprs)

    direct_basis = {
        "std",
        "nanstd",
        "stdev",
        "variance",
        "var",
        "bootstrap",
        "leave_one_out",
        "loo",
        "disagreement",
        "density",
        "sparsity",
        "residual",
        "quantile",
    }
    for term in direct_basis:
        if any(term in identifier for identifier in identifiers) or term in assignment_text:
            terms.append(term)

    if any(term in lowered_code for term in ("distance_to_observed", "nearest_distance", "neighbor_distance")):
        proxy_terms.append("distance_to_observed")
    elif "distance" in lowered_code and "observed_df" in lowered_code and "candidate_df" in lowered_code:
        proxy_terms.append("distance_proxy")
    if "density" in lowered_code or "sparsity" in lowered_code:
        proxy_terms.append("density_or_sparsity_proxy")

    uncertainty_var_names = {
        target.lower()
        for targets, _expr in facts.assignment_exprs
        for target in targets
        if any(term in target.lower() for term in ("uncert", "std", "variance", "var", "disagreement", "sparsity", "density"))
    }
    score_uses_uncertainty = bool(
        any(term in score_text for term in uncertainty_var_names)
        or any(term in score_text for term in ("std", "uncert", "variance", "var", "disagreement", "sparsity", "density"))
    )
    real_basis = bool(terms or proxy_terms)
    return {
        "uses_uncertainty": bool(real_basis and (score_uses_uncertainty or "uncertainty_bonus" in lowered_code or "exploration_bonus" in lowered_code)),
        "terms": _unique_limited(terms + proxy_terms),
        "exploration_proxy_terms": _unique_limited(proxy_terms),
    }


def _detect_score_linked_terms(facts: _AstFacts, lowered_code: str, terms: set[str]) -> list[str]:
    score_text = " ".join(facts.score_exprs)
    assignment_by_target = {target.lower(): expr for targets, expr in facts.assignment_exprs for target in targets}
    identifiers = {item.lower() for item in facts.names | facts.attrs | facts.calls | facts.function_names}
    hits: list[str] = []
    for term in terms:
        if term in score_text:
            hits.append(term)
            continue
        matching_vars = [name for name in identifiers if term in name]
        if any(name in score_text for name in matching_vars):
            hits.append(term)
            continue
        if any(term in target and target in score_text for target in assignment_by_target):
            hits.append(term)
    if not hits and "exploration_bonus" in lowered_code:
        hits.append("exploration_bonus")
    return _unique_limited(hits)


def _detect_acquisition_logic(facts: _AstFacts, lowered_code: str) -> dict[str, Any]:
    terms: list[str] = []
    score_text = " ".join(facts.score_exprs)
    for term in ("ucb", "lcb", "expected_improvement", "ei", "thompson", "acquisition", "upper_confidence", "lower_confidence"):
        if term in lowered_code:
            terms.append(term)

    mean_std_combo = bool(
        re.search(r"\b(mean|mu|pred|prediction|exploit)", score_text)
        and re.search(r"\b(std|sigma|uncert|variance|var)\b", score_text)
        and re.search(r"\+|\-", score_text)
    )
    if mean_std_combo:
        terms.append("surrogate_mean_plus_uncertainty")
    separate_components = "exploit" in lowered_code and any(term in lowered_code for term in ("explore", "uncertainty", "std", "novelty"))
    if separate_components:
        terms.append("separate_exploit_explore_components")
    return {
        "uses_acquisition_logic": bool(terms),
        "terms": _unique_limited(terms),
        "implies_uncertainty": bool(mean_std_combo or any(term in terms for term in ("ucb", "lcb", "expected_improvement", "ei"))),
        "implies_exploration": bool(terms),
        "uncertainty_terms": ["acquisition_uncertainty_component"] if terms else [],
        "exploration_terms": ["acquisition_exploration_component"] if terms else [],
    }


def _detect_state_update(facts: _AstFacts) -> tuple[bool, list[str]]:
    terms = sorted(facts.tool_state_mutations)
    for targets, expr in facts.assignment_exprs:
        lowered_targets = {target.lower() for target in targets}
        if any("state" in target for target in lowered_targets) and "tool_state" in expr and "copy" in expr:
            terms.append("state_copy_for_return")
    return bool(terms), _unique_limited(terms)


def _detect_small_n_fallback(facts: _AstFacts, lowered_code: str) -> tuple[bool, list[str]]:
    terms: list[str] = []
    if any(term in lowered_code for term in ("small_n", "cold_start", "fallback", "few_observed")):
        terms.append("fallback_term")
    for expr in facts.compare_exprs:
        if "len(observed_df)" in expr or "observed_n" in expr or "n_obs" in expr:
            if any(op in expr for op in ("<", "<=")):
                terms.append("observed_count_guard")
    return bool(terms), _unique_limited(terms)


def _detect_mixed_variable_support(facts: _AstFacts, lowered_code: str) -> tuple[bool, list[str]]:
    terms: list[str] = []
    for term in ("get_dummies", "onehot", "one_hot", "categorical", "category", "object", "astype(str)", "select_dtypes"):
        if term in lowered_code:
            terms.append(term)
    numeric_signal = any(term in lowered_code for term in ("numeric", "number", "to_numeric", "select_dtypes"))
    categorical_signal = any(term in lowered_code for term in ("categorical", "category", "object", "get_dummies", "astype(str)", "one_hot", "onehot"))
    return bool(numeric_signal and categorical_signal), _unique_limited(terms)


def _detect_diagnostics(facts: _AstFacts, lowered_code: str) -> tuple[bool, list[str]]:
    terms: list[str] = []
    if "tool_diagnostics" not in lowered_code:
        return False, []
    diagnostic_keywords = {
        "diagnostics",
        "score_summary",
        "score_component",
        "component_range",
        "exploit",
        "explore",
        "uncertainty",
        "fallback",
        "warning",
        "observed_n",
        "failure",
    }
    for term in diagnostic_keywords:
        if term in lowered_code:
            terms.append(term)
    for return_dict in facts.return_dicts:
        for key, value in zip(return_dict.keys, return_dict.values):
            if isinstance(key, ast.Constant) and key.value == "tool_diagnostics":
                if isinstance(value, ast.Dict) and value.keys:
                    terms.append("nonempty_tool_diagnostics")
    nontrivial = bool(set(terms) - {"diagnostics"}) or "nonempty_tool_diagnostics" in terms
    return nontrivial, _unique_limited(terms)


def _fake_uncertainty_risk(
    *,
    facts: _AstFacts,
    lowered_code: str,
    comments: str,
    uses_uncertainty: bool,
    uncertainty_terms: list[str],
) -> str:
    uncertainty_mentioned = (
        "uncert" in lowered_code
        or "uncert" in comments
        or "exploration" in lowered_code
        or "uncertainty" in comments
    )
    if not uncertainty_mentioned:
        return "low"
    fake_assignments = 0
    for targets, expr in facts.assignment_exprs:
        if not any(any(token in target.lower() for token in ("uncert", "exploration", "std", "variance")) for target in targets):
            continue
        if _expr_is_constant_or_random(expr):
            fake_assignments += 1
    if not uses_uncertainty and uncertainty_mentioned:
        return "high"
    if fake_assignments and not uncertainty_terms:
        return "high"
    if fake_assignments:
        return "medium"
    return "low"


def _over_exploration_risk(
    *,
    uses_exploration: bool,
    uses_diversity: bool,
    uses_uncertainty: bool,
    uses_surrogate: bool,
    uses_observed_y: bool,
    evidence: dict[str, list[str]],
    observed_quality: dict | None,
    score_summary: dict | None,
) -> str:
    if not (uses_exploration or uses_diversity):
        return "low"
    score_scale = _score_scale_risk(observed_quality=observed_quality, score_summary=score_summary)
    exploration_signal_count = len(evidence.get("exploration_terms", [])) + len(evidence.get("diversity_terms", []))
    exploit_signal = uses_surrogate or uses_observed_y or uses_uncertainty
    if exploration_signal_count >= 3 and not exploit_signal:
        return "high"
    if score_scale == "high" and exploration_signal_count >= 2:
        return "high"
    if not exploit_signal:
        return "medium"
    return "medium" if exploration_signal_count >= 3 else "low"


def _score_scale_risk(*, observed_quality: dict | None, score_summary: dict | None) -> str:
    summary = _merged_score_summary(observed_quality, score_summary)
    if not summary:
        return "unknown"
    finite_count = _safe_float(summary.get("finite_count"))
    count = _safe_float(summary.get("count"))
    nonfinite = bool(summary.get("nonfinite_or_parse_failure")) or (
        finite_count is not None and count is not None and finite_count < count
    )
    if nonfinite:
        return "high"
    max_abs = _safe_float(summary.get("max_abs_score"))
    if max_abs is None:
        values = [_safe_float(summary.get("min")), _safe_float(summary.get("max"))]
        finite_values = [abs(value) for value in values if value is not None]
        max_abs = max(finite_values) if finite_values else None
    ratio = _safe_float(summary.get("max_abs_to_median_abs_ratio"))
    span = _safe_float(summary.get("score_span"))
    if span is None:
        min_value = _safe_float(summary.get("min"))
        max_value = _safe_float(summary.get("max"))
        if min_value is not None and max_value is not None:
            span = max_value - min_value
    std = _safe_float(summary.get("score_std"))
    if std is None:
        std = _safe_float(summary.get("std"))
    margin = _safe_float(summary.get("rank1_margin"))
    if max_abs is not None and max_abs > 1e12:
        return "high"
    if ratio is not None and ratio > 1e9:
        return "high"
    if span is not None and abs(span) <= 1e-12:
        return "high"
    if std is not None and abs(std) <= 1e-12 and _safe_float(summary.get("count")) not in (None, 0.0, 1.0):
        return "high"
    if margin is not None and abs(margin) <= 1e-10:
        return "high"
    if max_abs is not None and max_abs > 1e6:
        return "medium"
    return "low"


def _merged_score_summary(observed_quality: dict | None, score_summary: dict | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(score_summary, dict):
        merged.update(score_summary)
    if isinstance(observed_quality, dict):
        diagnostics = observed_quality.get("diagnostics")
        if isinstance(diagnostics, dict):
            sanity = diagnostics.get("score_sanity")
            if isinstance(sanity, dict):
                merged.update(sanity)
        direct = observed_quality.get("score_summary")
        if isinstance(direct, dict):
            merged.update(direct)
    return merged


def _complexity_risk(source: str, facts: _AstFacts) -> str:
    if not facts.parse_ok:
        return "unknown"
    line_count = len(source.splitlines())
    char_count = len(source)
    if line_count > 260 or char_count > 14000 or facts.class_count >= 3:
        return "high"
    if facts.broad_exception_count >= 3 or facts.try_count >= 5:
        return "high"
    if line_count > 160 or char_count > 8000 or facts.broad_exception_count:
        return "medium"
    return "low"


def _optimizer_class(
    *,
    facts: _AstFacts,
    verifier_failed: bool,
    leakage_risk: str,
    is_static_ranker: bool,
    uses_surrogate: bool,
    uses_uncertainty: bool,
    uses_exploration: bool,
    uses_diversity: bool,
    uses_acquisition_logic: bool,
    uses_small_n_fallback: bool,
    supports_mixed_vars: bool,
    supports_batch: bool,
    uses_diagnostics: bool,
    uses_state_update: bool,
) -> str:
    if verifier_failed or leakage_risk == "high" or not facts.parse_ok:
        return "invalid_or_unsafe"
    hybrid_signals = sum(
        bool(value)
        for value in (
            uses_surrogate,
            uses_uncertainty or uses_exploration,
            uses_diversity,
            uses_acquisition_logic,
            uses_small_n_fallback,
            supports_mixed_vars,
            supports_batch,
            uses_diagnostics,
            uses_state_update,
        )
    )
    if hybrid_signals >= 6 and (uses_surrogate or uses_acquisition_logic) and (uses_uncertainty or uses_exploration or uses_diversity):
        return "hybrid_sequential_optimizer"
    if uses_acquisition_logic and (uses_uncertainty or uses_exploration) and (uses_surrogate or supports_batch):
        return "bo_like_optimizer"
    if uses_uncertainty:
        return "uncertainty_aware_optimizer"
    if uses_exploration or uses_diversity:
        return "heuristic_optimizer"
    if is_static_ranker:
        return "static_predictor_or_ranker"
    return "unknown"


def _classification_confidence(
    *,
    facts: _AstFacts,
    optimizer_class: str,
    evidence: dict[str, list[str]],
    verifier_failed: bool,
    leakage_risk: str,
) -> str:
    if not facts.parse_ok or optimizer_class == "unknown":
        return "low"
    if verifier_failed or leakage_risk == "high":
        return "high"
    evidence_count = sum(len(values) for values in evidence.values())
    if optimizer_class in {"static_predictor_or_ranker", "bo_like_optimizer", "hybrid_sequential_optimizer"}:
        return "high"
    if evidence_count >= 3:
        return "high"
    return "medium"


def _verifier_failed(verifier_result: dict | None) -> bool:
    if not isinstance(verifier_result, dict) or not verifier_result:
        return False
    if verifier_result.get("passed") is False or verifier_result.get("deployable") is False:
        return True
    failed_checks = verifier_result.get("failed_checks")
    return bool(failed_checks)


def _metadata_mentions_estimator(metadata_text: str) -> list[str]:
    return _unique_limited(
        [
            term
            for term in ("surrogate", "ridge", "estimator", "regression", "regressor", "ensemble")
            if term in metadata_text
        ]
    )


def _leakage_categories(*items: Any) -> list[str]:
    text = _metadata_text(items).lower()
    text = text.replace("hidden_y_leakage_self_check", "")
    patterns: tuple[tuple[str, str], ...] = (
        (r"hidden[_\s-]?y|hidden[_\s-]?yield|unobserved[_\s-]?y|all[_\s-]?remaining[_\s-]?y|answer[_\s-]?key", "nonpublic_outcome_reference"),
        (r"(?<![a-z0-9_])oracle(?![a-z0-9_])|(?<![a-z0-9_])posthoc(?![a-z0-9_])|(?<![a-z0-9_])true[_\s-]?rank(?![a-z0-9_])|(?<![a-z0-9_])full[_\s-]?pool[_\s-]?rank(?![a-z0-9_])|(?<![a-z0-9_])global[_\s-]?rank(?![a-z0-9_])", "retrospective_rank_reference"),
        (r"candidate_scores\.csv|score_cache|cached_score", "score_cache_artifact_reference"),
        (r"private[_\s-]?candidate|candidate_id_map|private[_\s-]?map", "private_mapping_reference"),
        (r"bo_reference|reference_acquisition|bo[_\s-]?acquisition|bo[_\s-]?predictive|boreferencepolicy|comparator", "reference_policy_artifact_reference"),
        (r"evaluator[_\s-]?internal|offlineevaluator|evaluator\.reveal|\.reveal\(", "evaluator_internal_reference"),
        (r"api[_\s-]?key|apikey|authorization|bearer|secret", "credential_reference"),
    )
    hits: list[str] = []
    for pattern, category in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(category)
    return _unique_limited(hits)


def _code_without_comments(source: str) -> str:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(str(source)).readline)
        kept = [tok.string for tok in tokens if tok.type != tokenize.COMMENT]
        return " ".join(kept)
    except tokenize.TokenError:
        return str(source)


def _comment_text(source: str) -> str:
    comments: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(str(source)).readline):
            if tok.type == tokenize.COMMENT:
                comments.append(tok.string)
    except tokenize.TokenError:
        return ""
    return "\n".join(comments)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _name_text(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_text(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Subscript):
        return _name_text(node.value)
    return ""


def _assigned_names(targets: list[ast.AST]) -> set[str]:
    names: set[str] = set()
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Tuple):
            names.update(_assigned_names(list(target.elts)))
        elif isinstance(target, ast.Attribute):
            text = _name_text(target)
            if text:
                names.add(text)
        elif isinstance(target, ast.Subscript):
            text = _name_text(target.value)
            if text:
                names.add(text)
    return names


def _is_score_name(name: str) -> bool:
    lowered = str(name).lower()
    return lowered == "score" or lowered.endswith("_score") or lowered in {"scores", "acquisition", "acquisition_score"}


def _unparse_lower(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node).lower()
    except Exception:
        return ""


def _expr_is_constant_or_random(expr: str) -> bool:
    stripped = str(expr).strip().lower()
    if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?", stripped):
        return True
    if "random" in stripped and not any(term in stripped for term in ("std", "var", "observed_y", "distance", "density")):
        return True
    if stripped in {"1.0", "0.0", "1", "0"}:
        return True
    return False


def _metadata_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(str(key) + " " + _metadata_text(item) for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_metadata_text(item) for item in value)
    return str(value)


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _unique_limited(values: list[str], *, limit: int = 10) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


__all__ = ["ToolMechanismClassification", "classify_tool_mechanism"]
