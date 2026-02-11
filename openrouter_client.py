"""OpenRouter API interface configured via dotenv."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

from eval_constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BACKOFF_SECONDS,
)


@dataclass
class OpenRouterConfig:
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS
    max_retries: int = MAX_RETRIES
    retry_backoff_seconds: float = RETRY_BACKOFF_SECONDS


class OpenRouterClient:
    def __init__(self, config: OpenRouterConfig | None = None) -> None:
        load_dotenv()
        self.config = config or OpenRouterConfig()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY in environment (.env).")

        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
        self.app_title = os.getenv("OPENROUTER_APP_TITLE")

    def chat_completion(self, user_prompt: str, system_prompt: str | None = None) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-Title"] = self.app_title

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
                return response.json()
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                if attempt > self.config.max_retries:
                    raise RuntimeError(
                        f"OpenRouter request failed after {self.config.max_retries} retries: {exc}"
                    ) from exc
                time.sleep(self.config.retry_backoff_seconds * attempt)
