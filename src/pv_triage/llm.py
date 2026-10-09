"""Pluggable LLM providers.

Every agent talks to an ``LLM`` with a single ``complete(system, user)`` method.
The default ``OfflineLLM`` is deterministic and needs no API key, so the whole
pipeline (and CI) runs anywhere. Set ``LLM_PROVIDER`` to ``openai`` or
``bedrock`` to use a hosted model instead.
"""

from __future__ import annotations

import os
from typing import Protocol


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class OfflineLLM:
    """Deterministic stand-in used for tests, demos and CI.

    It returns an empty string, which tells each agent to use its
    rule-based fallback path. This keeps behaviour identical with or
    without a real model and makes failures easy to reproduce.
    """

    name = "offline"

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        return ""


class OpenAILLM:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI  # optional dependency

        self._client = OpenAI()
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class BedrockLLM:
    name = "bedrock"

    def __init__(self, model_id: str | None = None) -> None:
        import boto3  # optional dependency

        self._client = boto3.client(
            "bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        self._model_id = model_id or os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
        )

    def complete(self, system: str, user: str) -> str:
        resp = self._client.converse(
            modelId=self._model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"temperature": 0, "maxTokens": 1024},
        )
        return resp["output"]["message"]["content"][0]["text"]


def get_llm(provider: str | None = None) -> LLM:
    provider = (provider or os.getenv("LLM_PROVIDER", "offline")).lower()
    if provider == "openai":
        return OpenAILLM()
    if provider == "bedrock":
        return BedrockLLM()
    return OfflineLLM()
