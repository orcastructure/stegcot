"""Anthropic Messages API interface configured via dotenv."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

from eval_constants import MAX_RETRIES, REQUEST_TIMEOUT_SECONDS, RETRY_BACKOFF_SECONDS


@dataclass
class AnthropicConfig:
    model: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS
    max_retries: int = MAX_RETRIES
    retry_backoff_seconds: float = RETRY_BACKOFF_SECONDS


class AnthropicClient:
    def __init__(self, config: AnthropicConfig) -> None:
        load_dotenv()
        self.config = config
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY in environment (.env).")
        self.base_url = "https://api.anthropic.com/v1/messages"

    @staticmethod
    def _normalize_model(model: str) -> str:
        if model.startswith("anthropic/"):
            model = model.split("/", 1)[1]
        # Accept OpenRouter-style dotted Claude 4.x aliases and map to Anthropic IDs.
        alias_map = {
            "claude-opus-4.6": "claude-opus-4-6",
            "claude-sonnet-4.6": "claude-sonnet-4-6",
        }
        return alias_map.get(model, model)

    def _to_anthropic_payload(self, messages: list[dict[str, str]], extra_body: dict[str, Any] | None) -> dict[str, Any]:
        if self.config.max_tokens is None:
            raise ValueError(
                "Anthropic requires max_tokens. Pass --max-tokens explicitly when using --provider anthropic."
            )

        system_parts: list[str] = []
        anth_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                system_parts.append(content)
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            anth_messages.append({"role": role, "content": content})

        payload: dict[str, Any] = {
            "model": self._normalize_model(self.config.model),
            "messages": anth_messages,
            "max_tokens": self.config.max_tokens,
        }
        if system_parts:
            payload["system"] = "\n\n".join(part for part in system_parts if part)
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p

        if extra_body:
            payload.update(extra_body)
        return payload

    def chat_completion_messages(
        self,
        messages: list[dict[str, str]],
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not messages:
            raise ValueError("messages must not be empty.")

        payload = self._to_anthropic_payload(messages, extra_body)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        attempt = 0
        while True:
            attempt += 1
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                if response.status_code >= 500:
                    raise requests.HTTPError(
                        f"Server error {response.status_code}: {response.text[:300]}",
                        response=response,
                    )
                response.raise_for_status()
                data = response.json()
                blocks = data.get("content", [])
                text_parts = [blk.get("text", "") for blk in blocks if isinstance(blk, dict) and blk.get("type") == "text"]
                text = "".join(text_parts).strip()
                # Normalize to OpenAI-style shape expected by existing experiment code.
                return {
                    "choices": [{"message": {"content": text}}],
                    "provider_raw_response": data,
                }
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                if attempt > self.config.max_retries:
                    detail = str(exc)
                    if isinstance(exc, requests.HTTPError) and exc.response is not None:
                        detail = f"HTTP {exc.response.status_code}: {exc.response.text[:800]}"
                    raise RuntimeError(
                        f"Anthropic request failed after {self.config.max_retries} retries: {detail}"
                    ) from exc
                time.sleep(self.config.retry_backoff_seconds * attempt)
