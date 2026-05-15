from __future__ import annotations

import os
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
    timeout_seconds: float = 120.0


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
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
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
        async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return _extract_anthropic_text(data)


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
    provider = os.getenv(f"{prefix}_PROVIDER{suffix}", os.getenv(f"{fallback_prefix}_PROVIDER{suffix}", "openai")).lower()
    if provider not in {"openai", "anthropic"}:
        raise RuntimeError(f"{prefix}_PROVIDER{suffix} must be openai or anthropic, got: {provider}")
    return ModelSettings(
        role=role,
        provider=provider,  # type: ignore[arg-type]
        base_url=_env_with_fallback(f"{prefix}_BASE_URL{suffix}", f"{fallback_prefix}_BASE_URL{suffix}"),
        api_key=_env_with_fallback(f"{prefix}_API_KEY{suffix}", f"{fallback_prefix}_API_KEY{suffix}"),
        model=_env_with_fallback(f"{prefix}_MODEL{suffix}", f"{fallback_prefix}_MODEL{suffix}"),
        timeout_seconds=timeout,
    )


def _env_with_fallback(name: str, fallback_name: str) -> str:
    value = os.getenv(name) or os.getenv(fallback_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


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
