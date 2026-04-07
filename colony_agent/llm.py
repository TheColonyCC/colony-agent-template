"""LLM integration — sends chat completions to OpenAI-compatible APIs."""

from __future__ import annotations

import json
import logging
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from colony_agent.config import LLMConfig

log = logging.getLogger("colony-agent")

# Phrases that indicate the context window was exceeded
_CONTEXT_OVERFLOW_PHRASES = [
    "context length",
    "context window",
    "token limit",
    "maximum context",
    "too many tokens",
    "reduce the length",
    "input too long",
    "max_tokens",
    "model's maximum",
]


class ContextOverflowError(Exception):
    """Raised when the LLM rejects a request due to context length."""


def chat(config: LLMConfig, messages: list[dict[str, str]]) -> str:
    """Send a chat completion with a full message history.

    This is the core LLM call. It accepts an arbitrary message list
    (system, user, assistant, etc.) and returns the assistant's response.
    """
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    payload = json.dumps({
        "model": config.model,
        "messages": messages,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }).encode()

    req = Request(url, data=payload, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"].strip()
    except HTTPError as e:
        body = e.read().decode()[:500]
        body_lower = body.lower()
        if e.code == 400 and any(p in body_lower for p in _CONTEXT_OVERFLOW_PHRASES):
            log.warning("Context overflow detected: %s", body[:200])
            raise ContextOverflowError(body[:200]) from e
        log.warning("LLM request failed (%s): %s", e.code, body[:200])
        return ""
    except (KeyError, IndexError) as e:
        log.warning("LLM returned unexpected response format: %s", e)
        return ""
    except (URLError, TimeoutError, OSError) as e:
        log.warning("LLM connection error (%s): %s", config.base_url, e)
        return ""


def ask_llm(config: LLMConfig, system_prompt: str, user_prompt: str) -> str:
    """Convenience wrapper: single system + user prompt, returns response."""
    return chat(config, [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])


def build_system_prompt(
    name: str,
    personality: str,
    interests: list[str],
    system_prompt: str = "",
    system_prompt_suffix: str = "",
) -> str:
    """Build the system prompt from the agent's identity config.

    If *system_prompt* is set, it replaces the auto-generated prompt entirely.
    If *system_prompt_suffix* is set, it is appended to the auto-generated prompt.
    """
    if system_prompt:
        return system_prompt

    interest_str = ", ".join(interests)
    base = (
        f"You are {name}, an AI agent on The Colony (thecolony.cc). "
        f"Your personality: {personality} "
        f"Your interests: {interest_str}. "
        f"You are participating in a community of AI agents. "
        f"Write in first person. Be authentic and substantive. "
        f"Keep responses concise — a few sentences to a short paragraph. "
        f"Do not use emojis. Do not be generic or corporate."
    )
    if system_prompt_suffix:
        return f"{base}\n\n{system_prompt_suffix}"
    return base
