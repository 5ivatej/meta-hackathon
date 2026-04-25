"""OpenAI-compatible helpers shared across training scripts."""
from __future__ import annotations

import json
import re
import time
from typing import Any, List

from openai import OpenAI

from .config import EndpointConfig


def build_client(endpoint: EndpointConfig) -> OpenAI:
    return OpenAI(base_url=endpoint.api_base_url, api_key=endpoint.api_key)


def chat_text(
    client: OpenAI,
    model: str,
    messages: List[dict[str, str]],
    temperature: float,
    max_tokens: int,
    retries: int = 3,
) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as exc:  # pragma: no cover - runtime safety path
            last_error = exc
            time.sleep(min(2 ** attempt, 4))
    raise RuntimeError(f"LLM call failed after retries: {last_error}")


def chat_json(
    client: OpenAI,
    model: str,
    messages: List[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    text = chat_text(
        client=client,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _extract_json_dict(text)


def extract_response_text(text: str) -> tuple[str, str]:
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    response_match = re.search(r"<response>(.*?)</response>", text, flags=re.DOTALL | re.IGNORECASE)
    think = think_match.group(1).strip() if think_match else ""
    response = response_match.group(1).strip() if response_match else text.strip()
    return think, response


def _extract_json_dict(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse JSON object from response: {text[:200]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON was not an object.")
    return parsed

