"""LLM integration — sends chat completions to OpenAI-compatible APIs."""

from __future__ import annotations

import json
import logging
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from colony_agent.config import LLMConfig

log = logging.getLogger("colony-agent")


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
        log.warning("LLM request failed (%s): %s", e.code, e.read().decode()[:200])
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


def build_system_prompt(name: str, personality: str, interests: list[str]) -> str:
    """Build the system prompt from the agent's identity config."""
    interest_str = ", ".join(interests)
    return (
        f"You are {name}, an AI agent on The Colony (thecolony.cc). "
        f"Your personality: {personality} "
        f"Your interests: {interest_str}. "
        f"You are participating in a community of AI agents. "
        f"Write in first person. Be authentic and substantive. "
        f"Keep responses concise — a few sentences to a short paragraph. "
        f"Do not use emojis. Do not be generic or corporate."
    )
