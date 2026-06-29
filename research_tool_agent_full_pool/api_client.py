"""Commonstack API client for Step 2 generated-tool synthesis."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib import error, request

from research_tool_agent_full_pool.artifact_logger import sanitize_text


class ApiKeyMissingError(RuntimeError):
    """Raised when no Commonstack credential is available."""


class CommonstackAPIError(RuntimeError):
    """Raised for sanitized Commonstack API failures."""


@dataclass(frozen=True)
class ResolvedApiKey:
    value: str
    source_label: str

    @property
    def secret_scan_terms(self) -> tuple[str, ...]:
        return (self.value,) if self.value else ()


def resolve_commonstack_api_key(
    *,
    api_key_path: str | None = None,
    backup_api_key_path: str | None = None,
    cwd: str | Path = ".",
) -> ResolvedApiKey:
    """Resolve a Commonstack key without printing or logging its contents."""

    env_value = os.environ.get("COMMONSTACK_API_KEY", "").strip()
    if env_value:
        return ResolvedApiKey(value=env_value, source_label="COMMONSTACK_API_KEY environment variable")
    root = Path(cwd)
    for label, maybe_path in (("api_key_path", api_key_path), ("backup_api_key_path", backup_api_key_path)):
        if not maybe_path:
            continue
        path = root / maybe_path
        if not path.exists() or not path.is_file():
            continue
        value = _extract_api_key_from_text(path.read_text(encoding="utf-8"))
        if value:
            return ResolvedApiKey(value=value, source_label=label)
    raise ApiKeyMissingError(
        "Commonstack API key is missing. Set COMMONSTACK_API_KEY or provide a configured key file."
    )


def _extract_api_key_from_text(text: str) -> str:
    """Extract a bearer key from simple key files without treating URLs as keys."""

    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if lowered.startswith(("http://", "https://")):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key_lower = key.strip().lower()
            if key_lower in {"commonstack_api_key", "api_key", "key", "token", "bearer"}:
                candidate = value.strip().strip('"').strip("'")
            else:
                continue
        elif ":" in line and lowered.split(":", 1)[0] in {"commonstack_api_key", "api_key", "key", "token", "bearer"}:
            candidate = line.split(":", 1)[1].strip().strip('"').strip("'")
        else:
            candidate = line
        if candidate and not candidate.lower().startswith(("http://", "https://")):
            return candidate
    return ""


class CommonstackToolSynthesisClient:
    """Minimal OpenAI-compatible client for generated-tool synthesis."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        response_format_json_object: bool,
        api_key_path: str | None = None,
        backup_api_key_path: str | None = None,
        cwd: str | Path = ".",
        reasoning_effort: str | None = None,
        api_mode: str = "chat",
        stream: bool = False,
        response_verbosity: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.response_format_json_object = bool(response_format_json_object)
        self.reasoning_effort = str(reasoning_effort).strip() if reasoning_effort else None
        self.api_mode = _normalize_api_mode(api_mode, endpoint)
        self.stream = bool(stream)
        self.response_verbosity = str(response_verbosity).strip() if response_verbosity else None
        self._resolved_key = resolve_commonstack_api_key(
            api_key_path=api_key_path,
            backup_api_key_path=backup_api_key_path,
            cwd=cwd,
        )

    @property
    def secret_scan_terms(self) -> tuple[str, ...]:
        return self._resolved_key.secret_scan_terms

    def create_tool(
        self,
        *,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
        schema_name: str | None = None,
        schema_description: str | None = None,
    ) -> str:
        """Call Commonstack and return only the assistant text."""

        payload = self.build_payload(
            messages=messages,
            json_schema=json_schema,
            schema_name=schema_name,
            schema_description=schema_description,
        )
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0",
                "Origin": "https://www.commonstack.ai",
                "Referer": "https://www.commonstack.ai/",
                "Authorization": f"Bearer {self._resolved_key.value}",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                response_text = response.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise CommonstackAPIError(self._sanitize_api_error(f"HTTP {exc.code}: {detail}")) from exc
        except error.URLError as exc:
            raise CommonstackAPIError(self._sanitize_api_error(str(exc.reason))) from exc
        except TimeoutError as exc:
            raise CommonstackAPIError("Commonstack request timed out.") from exc

        try:
            if self.api_mode == "responses" and self.stream:
                content = _extract_responses_stream_text(response_text)
            else:
                parsed = json.loads(response_text)
                content = _extract_response_text(parsed, api_mode=self.api_mode)
        except Exception as exc:
            raise CommonstackAPIError(_sanitize_api_error("Commonstack response did not match expected schema.")) from exc
        if not isinstance(content, str) or not content.strip():
            raise CommonstackAPIError("Commonstack response content was empty.")
        return content

    def build_payload(
        self,
        *,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
        schema_name: str | None = None,
        schema_description: str | None = None,
    ) -> dict[str, Any]:
        """Build the JSON request payload without credentials for tests/audits."""

        if self.api_mode == "responses":
            return self.build_responses_payload(
                messages=messages,
                json_schema=json_schema,
                schema_name=schema_name,
                schema_description=schema_description,
            )
        return self.build_chat_payload(
            messages=messages,
            json_schema=json_schema,
            schema_name=schema_name,
            schema_description=schema_description,
        )

    def build_chat_payload(
        self,
        *,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
        schema_name: str | None = None,
        schema_description: str | None = None,
    ) -> dict[str, Any]:
        """Build a Chat Completions payload without credentials for tests/audits."""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_chat_message(message) for message in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _schema_name(schema_name),
                    "schema": json_schema,
                    "strict": True,
                },
            }
            if schema_description:
                payload["response_format"]["json_schema"]["description"] = str(schema_description)
        elif self.response_format_json_object:
            payload["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def build_responses_payload(
        self,
        *,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
        schema_name: str | None = None,
        schema_description: str | None = None,
    ) -> dict[str, Any]:
        """Build a Responses API payload without credentials for tests/audits."""

        text: dict[str, Any] = {}
        if json_schema is not None:
            text["format"] = {
                "type": "json_schema",
                "name": _schema_name(schema_name),
                "schema": json_schema,
                "strict": True,
            }
            if schema_description:
                text["format"]["description"] = str(schema_description)
        elif self.response_format_json_object:
            text["format"] = {"type": "json_object"}
        if self.response_verbosity:
            text["verbosity"] = self.response_verbosity
        payload: dict[str, Any] = {
            "model": self.model,
            "input": [_responses_message(message) for message in messages],
            "max_output_tokens": self.max_tokens,
        }
        if text:
            payload["text"] = text
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.stream:
            payload["stream"] = True
        return payload

    def _sanitize_api_error(self, text: str) -> str:
        return sanitize_text(text, extra_secret_terms=self.secret_scan_terms)[:800]


def _sanitize_api_error(text: str) -> str:
    return sanitize_text(text)[:800]


def _normalize_api_mode(api_mode: str, endpoint: str) -> str:
    mode = str(api_mode or "").strip().lower()
    if mode in {"responses", "response"}:
        return "responses"
    if mode in {"chat", "chat_completions", "chat.completions"}:
        return "chat"
    if str(endpoint).rstrip("/").endswith("/responses"):
        return "responses"
    return "chat"


def _schema_name(value: str | None) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "structured_output")).strip("_")
    return (name or "structured_output")[:64]


def _chat_message(message: dict[str, str]) -> dict[str, str]:
    role = str(message.get("role", "user"))
    if role == "developer":
        role = "system"
    return {"role": role, "content": str(message.get("content", ""))}


def _responses_message(message: dict[str, str]) -> dict[str, str]:
    role = str(message.get("role", "user"))
    if role == "system":
        role = "developer"
    return {"role": role, "content": str(message.get("content", ""))}


def _extract_response_text(response_payload: Any, *, api_mode: str) -> str:
    if api_mode == "responses":
        return _extract_responses_text(response_payload)
    return _extract_chat_text(response_payload)


def _extract_chat_text(response_payload: Any) -> str:
    choices = response_payload["choices"]
    choice = choices[0]
    message = choice.get("message", {})
    content = message.get("content", choice.get("text"))
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "".join(parts)
    raise KeyError("content")


def _extract_responses_text(response_payload: Any) -> str:
    if isinstance(response_payload.get("output_text"), str):
        return response_payload["output_text"]
    parts: list[str] = []
    for item in response_payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict):
                if isinstance(content.get("text"), str):
                    parts.append(content["text"])
                elif isinstance(content.get("content"), str):
                    parts.append(content["content"])
    if parts:
        return "".join(parts)
    raise KeyError("output_text")


def _extract_responses_stream_text(response_text: str) -> str:
    parts: list[str] = []
    final_payload: Any | None = None
    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        event_type = str(event.get("type", ""))
        if event_type == "response.output_text.delta" and isinstance(event.get("delta"), str):
            parts.append(event["delta"])
        elif event_type == "response.completed" and isinstance(event.get("response"), dict):
            final_payload = event["response"]
    if parts:
        if final_payload is not None:
            status = str(final_payload.get("status", "")).lower()
            incomplete = final_payload.get("incomplete_details")
            if status == "incomplete" or incomplete:
                raise CommonstackAPIError(
                    _sanitize_api_error(f"Responses stream ended incomplete: {json.dumps(incomplete, default=str)[:300]}")
                )
        return "".join(parts)
    if final_payload is not None:
        return _extract_responses_text(final_payload)
    raise KeyError("stream_output_text")
