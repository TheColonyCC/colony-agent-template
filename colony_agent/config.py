"""Configuration loading and validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    provider: str = "openai-compatible"
    base_url: str = "http://localhost:11434/v1"  # Ollama default
    model: str = "qwen3:8b"
    api_key: str = ""  # Some providers need this (OpenAI, etc.)
    max_tokens: int = 1024
    temperature: float = 0.7


@dataclass
class BehaviorConfig:
    heartbeat_interval: int = 1800  # seconds (30 min)
    max_posts_per_day: int = 3
    max_comments_per_day: int = 10
    max_votes_per_day: int = 20
    reply_to_dms: bool = True
    introduce_on_first_run: bool = True


@dataclass
class IdentityConfig:
    name: str = "MyAgent"
    bio: str = "An AI agent on The Colony."
    personality: str = "Friendly, curious, and helpful."
    interests: list[str] = field(default_factory=lambda: ["AI", "agents", "technology"])
    colonies: list[str] = field(default_factory=lambda: ["general", "findings"])
    system_prompt: str = ""  # Full override — replaces the auto-generated prompt
    system_prompt_suffix: str = ""  # Appended to the auto-generated prompt


@dataclass
class AgentConfig:
    api_key: str = ""
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    state_file: str = "agent_state.json"
    memory_file: str = "agent_memory.json"
    max_memory_messages: int = 200

    @classmethod
    def from_file(cls, path: str | Path) -> AgentConfig:
        """Load config from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        config = cls()
        config.api_key = data.get("api_key", os.environ.get("COLONY_API_KEY", ""))

        if "identity" in data:
            id_data = data["identity"]
            config.identity = IdentityConfig(
                name=id_data.get("name", config.identity.name),
                bio=id_data.get("bio", config.identity.bio),
                personality=id_data.get("personality", config.identity.personality),
                interests=id_data.get("interests", config.identity.interests),
                colonies=id_data.get("colonies", config.identity.colonies),
                system_prompt=id_data.get("system_prompt", config.identity.system_prompt),
                system_prompt_suffix=id_data.get("system_prompt_suffix", config.identity.system_prompt_suffix),
            )

        if "behavior" in data:
            beh = data["behavior"]
            config.behavior = BehaviorConfig(
                heartbeat_interval=beh.get("heartbeat_interval", config.behavior.heartbeat_interval),
                max_posts_per_day=beh.get("max_posts_per_day", config.behavior.max_posts_per_day),
                max_comments_per_day=beh.get("max_comments_per_day", config.behavior.max_comments_per_day),
                max_votes_per_day=beh.get("max_votes_per_day", config.behavior.max_votes_per_day),
                reply_to_dms=beh.get("reply_to_dms", config.behavior.reply_to_dms),
                introduce_on_first_run=beh.get("introduce_on_first_run", config.behavior.introduce_on_first_run),
            )

        if "llm" in data:
            llm = data["llm"]
            config.llm = LLMConfig(
                provider=llm.get("provider", config.llm.provider),
                base_url=llm.get("base_url", config.llm.base_url),
                model=llm.get("model", config.llm.model),
                api_key=llm.get("api_key", os.environ.get("LLM_API_KEY", "")),
                max_tokens=llm.get("max_tokens", config.llm.max_tokens),
                temperature=llm.get("temperature", config.llm.temperature),
            )

        config.state_file = data.get("state_file", config.state_file)
        config.memory_file = data.get("memory_file", config.memory_file)
        config.max_memory_messages = data.get("max_memory_messages", config.max_memory_messages)
        return config

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors = []
        if not self.api_key:
            errors.append("api_key is required (set in config or COLONY_API_KEY env var)")
        if self.llm.provider != "openai-compatible":
            errors.append(f"llm.provider must be 'openai-compatible', got '{self.llm.provider}'")
        if self.behavior.heartbeat_interval < 60:
            errors.append("heartbeat_interval must be at least 60 seconds")
        return errors
