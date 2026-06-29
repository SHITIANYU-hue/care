"""Static checks for generated optimizer tools."""

from __future__ import annotations

import ast
from typing import Any

from research_tool_agent_full_pool.tool_contract import (
    ALLOWED_IMPORTS,
    ALLOWED_SKLEARN_SYMBOLS,
    FORBIDDEN_ATTRIBUTE_CALLS,
    FORBIDDEN_CALL_NAMES,
    FORBIDDEN_IMPORT_ROOTS,
    FORBIDDEN_TERMS,
    REQUIRED_ENTRYPOINT,
)


def check_generated_tool_source(source: str, *args: Any, **kwargs: Any) -> None:
    """Raise if generated code violates the full-pool safety contract."""

    report = static_check_generated_tool_source(source)
    if not report["passed"]:
        raise ValueError("Generated tool failed static check: " + "; ".join(report["violations"]))


def static_check_generated_tool_source(source: str) -> dict[str, Any]:
    """Return a structured static-check report for generated code."""

    report: dict[str, Any] = {
        "passed": False,
        "violations": [],
        "allowed_imports_seen": [],
        "forbidden_imports_seen": [],
        "forbidden_calls_seen": [],
        "forbidden_attribute_references_seen": [],
    }
    violations: list[str] = report["violations"]
    if not isinstance(source, str) or not source.strip():
        violations.append("empty_source")
        return report
    lowered = source.lower()
    forbidden_hits = [term for term in FORBIDDEN_TERMS if term.lower() in lowered]
    if forbidden_hits:
        violations.append("forbidden_terms")
    if "__" in source:
        violations.append("dunder_pattern")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        violations.append("syntax_error")
        return report

    entrypoints = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == REQUIRED_ENTRYPOINT
    ]
    if len(entrypoints) != 1:
        violations.append("missing_or_duplicate_rank_candidates")

    allowed_imports_seen: set[str] = set()
    forbidden_imports_seen: set[str] = set()
    allowed = set(ALLOWED_IMPORTS)
    forbidden_roots = set(FORBIDDEN_IMPORT_ROOTS)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_names = [alias.name for alias in node.names]
            for module_name in module_names:
                if module_name.startswith("sklearn"):
                    forbidden_imports_seen.add(module_name)
        elif isinstance(node, ast.ImportFrom):
            module_names = [node.module or ""]
            module_name = node.module or ""
            if module_name.startswith("sklearn"):
                allowed_symbols = set(ALLOWED_SKLEARN_SYMBOLS.get(module_name, ()))
                imported_names = {alias.name for alias in node.names}
                if not allowed_symbols or not imported_names.issubset(allowed_symbols):
                    forbidden_imports_seen.add(module_name)
        else:
            module_names = []
        for module_name in module_names:
            if _is_forbidden_import(module_name, forbidden_roots):
                forbidden_imports_seen.add(module_name)
            elif _is_allowed_import(module_name, allowed):
                allowed_imports_seen.add(module_name)
            else:
                forbidden_imports_seen.add(module_name)

        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            attr_name = _attribute_name(node.func)
            if call_name in FORBIDDEN_CALL_NAMES:
                report["forbidden_calls_seen"].append(call_name)
            if attr_name in FORBIDDEN_ATTRIBUTE_CALLS:
                report["forbidden_calls_seen"].append(attr_name)
        if isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRIBUTE_CALLS:
                report["forbidden_attribute_references_seen"].append(node.attr)
            if node.attr == "environ":
                violations.append("environment_access")
            if node.attr == "attrs":
                violations.append("dataframe_attrs_access")
        if isinstance(node, ast.Name) and node.id in {"__builtins__"}:
            violations.append("dunder_escape")

    report["allowed_imports_seen"] = sorted(allowed_imports_seen)
    report["forbidden_imports_seen"] = sorted(forbidden_imports_seen)
    if forbidden_imports_seen:
        violations.append("forbidden_imports")
    if report["forbidden_calls_seen"]:
        violations.append("forbidden_calls")
    if report["forbidden_attribute_references_seen"]:
        violations.append("forbidden_attribute_references")
    report["violations"] = sorted(set(violations))
    report["forbidden_calls_seen"] = sorted(set(report["forbidden_calls_seen"]))
    report["forbidden_attribute_references_seen"] = sorted(set(report["forbidden_attribute_references_seen"]))
    report["passed"] = not report["violations"]
    return report


def _is_allowed_import(module_name: str, allowed: set[str]) -> bool:
    return any(module_name == item or module_name.startswith(f"{item}.") for item in allowed)


def _is_forbidden_import(module_name: str, forbidden_roots: set[str]) -> bool:
    return any(module_name == item or module_name.startswith(f"{item}.") for item in forbidden_roots)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _attribute_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""
