"""Minimal client for any OpenAI-compatible chat endpoint (OpenAI, vLLM, Ollama, LM Studio)."""

from __future__ import annotations

import logging
import os
import time

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)


class ChatClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        base_url = os.environ.get("LLM_BASE_URL") or None
        api_key = os.environ.get("LLM_API_KEY") or ("not-needed" if base_url else None)
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, messages: list[dict[str, str]]) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content or ""
            except openai.APIError as exc:
                if attempt == self.max_retries:
                    raise
                delay = 2**attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %ds.",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable")
