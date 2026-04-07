"""Tests for colony_agent.config."""

import json

import pytest

from colony_agent.config import AgentConfig, BehaviorConfig, IdentityConfig, LLMConfig


class TestLLMConfig:
    def test_defaults(self):
        c = LLMConfig()
        assert c.provider == "openai-compatible"
        assert c.model == "qwen3:8b"
        assert c.max_tokens == 1024
        assert c.temperature == 0.7

    def test_custom_values(self):
        c = LLMConfig(provider="openai-compatible", model="gpt-4", api_key="sk-test")
        assert c.provider == "openai-compatible"
        assert c.model == "gpt-4"
        assert c.api_key == "sk-test"


class TestBehaviorConfig:
    def test_defaults(self):
        c = BehaviorConfig()
        assert c.heartbeat_interval == 1800
        assert c.max_posts_per_day == 3
        assert c.max_comments_per_day == 10
        assert c.max_votes_per_day == 20
        assert c.reply_to_dms is True
        assert c.introduce_on_first_run is True


class TestIdentityConfig:
    def test_defaults(self):
        c = IdentityConfig()
        assert c.name == "MyAgent"
        assert "AI" in c.interests
        assert "general" in c.colonies

    def test_custom_interests(self):
        c = IdentityConfig(interests=["music", "art"])
        assert c.interests == ["music", "art"]


class TestAgentConfig:
    def test_defaults(self):
        c = AgentConfig()
        assert c.api_key == ""
        assert c.state_file == "agent_state.json"
        assert isinstance(c.identity, IdentityConfig)
        assert isinstance(c.behavior, BehaviorConfig)
        assert isinstance(c.llm, LLMConfig)

    def test_from_file_minimal(self, tmp_path):
        cfg = {"api_key": "col_test123"}
        path = tmp_path / "agent.json"
        path.write_text(json.dumps(cfg))

        config = AgentConfig.from_file(path)
        assert config.api_key == "col_test123"
        # Defaults should be preserved
        assert config.identity.name == "MyAgent"
        assert config.behavior.heartbeat_interval == 1800
        assert config.llm.provider == "openai-compatible"

    def test_from_file_full(self, tmp_path):
        cfg = {
            "api_key": "col_full",
            "identity": {
                "name": "TestBot",
                "bio": "A test agent",
                "personality": "Serious",
                "interests": ["testing", "QA"],
                "colonies": ["findings"],
            },
            "behavior": {
                "heartbeat_interval": 600,
                "max_posts_per_day": 1,
                "max_comments_per_day": 5,
                "max_votes_per_day": 10,
                "reply_to_dms": False,
                "introduce_on_first_run": False,
            },
            "llm": {
                "provider": "openai-compatible",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4",
                "api_key": "sk-test",
                "max_tokens": 512,
                "temperature": 0.3,
            },
            "state_file": "custom_state.json",
        }
        path = tmp_path / "agent.json"
        path.write_text(json.dumps(cfg))

        config = AgentConfig.from_file(path)
        assert config.api_key == "col_full"
        assert config.identity.name == "TestBot"
        assert config.identity.interests == ["testing", "QA"]
        assert config.behavior.heartbeat_interval == 600
        assert config.behavior.reply_to_dms is False
        assert config.llm.provider == "openai-compatible"
        assert config.llm.model == "gpt-4"
        assert config.state_file == "custom_state.json"

    def test_from_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AgentConfig.from_file(tmp_path / "nonexistent.json")

    def test_from_file_env_vars(self, tmp_path, monkeypatch):
        cfg = {}
        path = tmp_path / "agent.json"
        path.write_text(json.dumps(cfg))

        monkeypatch.setenv("COLONY_API_KEY", "col_from_env")
        config = AgentConfig.from_file(path)
        assert config.api_key == "col_from_env"

    def test_from_file_llm_api_key_env(self, tmp_path, monkeypatch):
        cfg = {"api_key": "col_x", "llm": {"provider": "openai-compatible"}}
        path = tmp_path / "agent.json"
        path.write_text(json.dumps(cfg))

        monkeypatch.setenv("LLM_API_KEY", "sk-from-env")
        config = AgentConfig.from_file(path)
        assert config.llm.api_key == "sk-from-env"

    def test_validate_missing_api_key(self):
        config = AgentConfig()
        errors = config.validate()
        assert any("api_key" in e for e in errors)

    def test_validate_invalid_provider(self):
        config = AgentConfig(api_key="col_x", llm=LLMConfig(provider="bad"))
        errors = config.validate()
        assert any("provider" in e for e in errors)

    def test_validate_none_provider_rejected(self):
        config = AgentConfig(api_key="col_x", llm=LLMConfig(provider="none"))
        errors = config.validate()
        assert any("provider" in e for e in errors)

    def test_validate_low_heartbeat(self):
        config = AgentConfig(
            api_key="col_x", behavior=BehaviorConfig(heartbeat_interval=10)
        )
        errors = config.validate()
        assert any("heartbeat_interval" in e for e in errors)

    def test_validate_valid(self):
        config = AgentConfig(api_key="col_test")
        errors = config.validate()
        assert errors == []
