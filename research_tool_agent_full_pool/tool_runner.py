"""Execution boundary for generated optimizer tools."""

from __future__ import annotations

import contextlib
import io
import importlib
from typing import Any

import pandas as pd

from research_tool_agent_full_pool.artifact_logger import sanitize_text
from research_tool_agent_full_pool.tool_contract import (
    ALLOWED_IMPORTS,
    ALLOWED_SKLEARN_SYMBOLS,
    REQUIRED_ENTRYPOINT,
)
from research_tool_agent_full_pool.tool_static_check import check_generated_tool_source


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "zip": zip,
}


def run_rank_candidates_tool(
    *,
    tool_source: str,
    observed_df: Any,
    candidate_df: Any,
    memory: str | None = None,
    tool_state: dict[str, Any] | None = None,
) -> Any:
    """Execute `rank_candidates` against observed_df and full candidate_df."""

    output, _report = execute_rank_candidates_tool(
        tool_source=tool_source,
        observed_df=observed_df,
        candidate_df=candidate_df,
        memory=memory,
        tool_state=tool_state,
    )
    return output


def execute_rank_candidates_tool(
    *,
    tool_source: str,
    observed_df: Any,
    candidate_df: Any,
    memory: str | None = None,
    tool_state: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Execute generated code with restricted builtins/imports and diagnostics."""

    check_generated_tool_source(tool_source)
    safe_builtins = dict(SAFE_BUILTINS)
    safe_builtins["__import__"] = _restricted_import
    namespace: dict[str, Any] = {"__builtins__": safe_builtins}
    stdout = io.StringIO()
    stderr = io.StringIO()
    report: dict[str, Any] = {
        "passed": False,
        "stdout": "",
        "stderr": "",
        "error_type": None,
        "error": None,
    }
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(tool_source, "<research_tool_agent_full_pool_generated_tool>", "exec"), namespace)
            entrypoint = namespace.get(REQUIRED_ENTRYPOINT)
            if not callable(entrypoint):
                raise ValueError(f"Generated tool entrypoint {REQUIRED_ENTRYPOINT!r} is not callable.")
            output = entrypoint(
                _public_dataframe_copy(observed_df),
                _public_dataframe_copy(candidate_df),
                memory=memory,
                tool_state=dict(tool_state or {}),
            )
        report["passed"] = True
        return output, _finalize_report(report, stdout, stderr)
    except Exception as exc:
        report["error_type"] = exc.__class__.__name__
        report["error"] = sanitize_text(str(exc))[:800]
        raise RuntimeError(report["error"] or report["error_type"]) from exc
    finally:
        report.update(_finalize_report(report, stdout, stderr))


def _restricted_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    if level != 0:
        raise ImportError("Relative imports are not allowed in generated tools.")
    if name.startswith("sklearn"):
        allowed_symbols = set(ALLOWED_SKLEARN_SYMBOLS.get(name, ()))
        requested = {str(item) for item in (fromlist or ()) if str(item) != "*"}
        if not allowed_symbols or not requested or not requested.issubset(allowed_symbols):
            raise ImportError("Only bounded sklearn estimator imports are allowed in generated tools.")
    if not _is_allowed_import(name):
        raise ImportError("Import is not allowed in generated tools.")
    return importlib.import_module(name)


def _is_allowed_import(module_name: str) -> bool:
    return any(module_name == item or module_name.startswith(f"{item}.") for item in ALLOWED_IMPORTS)


def _public_dataframe_copy(frame: Any) -> Any:
    """Copy DataFrames for generated tools without private pandas attrs."""

    if isinstance(frame, pd.DataFrame):
        copied = frame.copy()
        copied.attrs = {}
        return copied
    if hasattr(frame, "copy"):
        return frame.copy()
    return frame


def _finalize_report(report: dict[str, Any], stdout: io.StringIO, stderr: io.StringIO) -> dict[str, Any]:
    finalized = dict(report)
    finalized["stdout"] = sanitize_text(stdout.getvalue())[:1000]
    finalized["stderr"] = sanitize_text(stderr.getvalue())[:1000]
    return finalized
