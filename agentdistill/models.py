from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx


Role = Literal["weak", "teacher"]


@dataclass(frozen=True)
class ModelSettings:
    role: Role
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 120.0


class ChatClient:
    def __init__(self, settings: ModelSettings):
        self.settings = settings

    async def complete(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
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


def load_model_settings(role: Role) -> ModelSettings:
    prefix = role.upper()
    timeout = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
    return ModelSettings(
        role=role,
        base_url=_required_env(f"{prefix}_BASE_URL"),
        api_key=_required_env(f"{prefix}_API_KEY"),
        model=_required_env(f"{prefix}_MODEL"),
        timeout_seconds=timeout,
    )


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
