from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Literal

import httpx


Role = Literal["weak", "teacher", "critic"]
Provider = Literal["openai", "anthropic"]


@dataclass(frozen=True)
class ModelSettings:
    role: Role
    provider: Provider
    base_url: str
    api_key: str
    model: str
    reasoning_effort: str | None = None
    timeout_seconds: float = 120.0
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0


class ChatClient:
    def __init__(self, settings: ModelSettings):
        self.settings = settings

    async def complete(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        if self.settings.provider == "anthropic":
            return await self._complete_anthropic(messages, temperature)
        return await self._complete_openai(messages, temperature)

    async def _complete_openai(self, messages: list[dict[str, str]], temperature: float) -> str:
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.settings.reasoning_effort:
            payload["reasoning_effort"] = self.settings.reasoning_effort
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        data = await self._post_json_with_retries(url, headers, payload)
        return data["choices"][0]["message"]["content"]

    async def _complete_anthropic(self, messages: list[dict[str, str]], temperature: float) -> str:
        url = self.settings.base_url.rstrip("/") + "/messages"
        system_parts = [msg["content"] for msg in messages if msg["role"] == "system"]
        conversation = [msg for msg in messages if msg["role"] != "system"]
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": 4096,
            "messages": conversation,
            "temperature": temperature,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        headers = {
            "x-api-key": self.settings.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        data = await self._post_json_with_retries(url, headers, payload)
        return _extract_anthropic_text(data)

    async def _post_json_with_retries(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        attempts = self.settings.max_retries + 1
        for attempt in range(attempts):
            try:
                return await self._post_json_once(url, headers, payload)
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError, json.JSONDecodeError) as exc:
                if attempt >= self.settings.max_retries or not _is_retryable_error(exc):
                    print(
                        "[model-retry] exhausted "
                        f"role={self.settings.role} model={self.settings.model} "
                        f"attempt={attempt + 1}/{attempts} error={_error_summary(exc)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise
                sleep_seconds = self.settings.retry_backoff_seconds * (2**attempt)
                print(
                    "[model-retry] retrying "
                    f"role={self.settings.role} model={self.settings.model} "
                    f"attempt={attempt + 1}/{attempts} wait_seconds={sleep_seconds:g} "
                    f"error={_error_summary(exc)}",
                    file=sys.stderr,
                    flush=True,
                )
                await asyncio.sleep(sleep_seconds)
        raise RuntimeError("unreachable retry state")

    async def _post_json_once(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()


def load_model_settings(role: Role, profile: str | None = None) -> ModelSettings:
    prefix = role.upper()
    fallback_prefix = "TEACHER" if role == "critic" else prefix
    suffix = f"_{profile.upper()}" if profile else ""
    default_timeout = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
    timeout = float(
        os.getenv(
            f"{prefix}_TIMEOUT_SECONDS{suffix}",
            os.getenv(f"{fallback_prefix}_TIMEOUT_SECONDS{suffix}", str(default_timeout)),
        )
    )
    default_max_retries = int(os.getenv("REQUEST_MAX_RETRIES", "2"))
    max_retries = int(
        os.getenv(
            f"{prefix}_MAX_RETRIES{suffix}",
            os.getenv(f"{fallback_prefix}_MAX_RETRIES{suffix}", str(default_max_retries)),
        )
    )
    default_backoff = float(os.getenv("REQUEST_RETRY_BACKOFF_SECONDS", "2"))
    retry_backoff = float(
        os.getenv(
            f"{prefix}_RETRY_BACKOFF_SECONDS{suffix}",
            os.getenv(f"{fallback_prefix}_RETRY_BACKOFF_SECONDS{suffix}", str(default_backoff)),
        )
    )
    reasoning_effort = _optional_env_with_fallback(
        f"{prefix}_REASONING_EFFORT{suffix}",
        f"{fallback_prefix}_REASONING_EFFORT{suffix}",
    )
    provider = os.getenv(f"{prefix}_PROVIDER{suffix}", os.getenv(f"{fallback_prefix}_PROVIDER{suffix}", "openai")).lower()
    if provider not in {"openai", "anthropic"}:
        raise RuntimeError(f"{prefix}_PROVIDER{suffix} must be openai or anthropic, got: {provider}")
    return ModelSettings(
        role=role,
        provider=provider,  # type: ignore[arg-type]
        base_url=_env_with_fallback(f"{prefix}_BASE_URL{suffix}", f"{fallback_prefix}_BASE_URL{suffix}"),
        api_key=_env_with_fallback(f"{prefix}_API_KEY{suffix}", f"{fallback_prefix}_API_KEY{suffix}"),
        model=_env_with_fallback(f"{prefix}_MODEL{suffix}", f"{fallback_prefix}_MODEL{suffix}"),
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff,
    )


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, json.JSONDecodeError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 409, 425, 429, 500, 502, 503, 504, 524}
    return False


def _error_summary(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"http_status_{exc.response.status_code}"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json_response"
    return exc.__class__.__name__


def _env_with_fallback(name: str, fallback_name: str) -> str:
    value = os.getenv(name) or os.getenv(fallback_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _optional_env_with_fallback(name: str, fallback_name: str) -> str | None:
    value = os.getenv(name) or os.getenv(fallback_name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _extract_anthropic_text(data: dict[str, Any]) -> str:
    parts = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)
