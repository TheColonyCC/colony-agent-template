"""Tests for colony_agent.cli."""

import json
from unittest.mock import MagicMock, patch

import pytest
from colony_sdk.client import ColonyAPIError

from colony_agent.cli import cmd_init, cmd_status, cmd_test_llm


def make_status_args(config_path: str):
    """Create a minimal args namespace for cmd_status."""
    args = MagicMock()
    args.config = config_path
    return args


def write_config(tmp_path, **overrides):
    """Write a minimal config file and return its path."""
    config = {
        "api_key": "col_test",
        "identity": {"name": "TestBot"},
        "llm": {"provider": "openai-compatible"},
        "state_file": str(tmp_path / "state.json"),
        "memory_file": str(tmp_path / "memory.json"),
        **overrides,
    }
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(config))
    return str(path)


class TestCmdStatus:
    @patch("colony_agent.cli.ColonyClient")
    def test_shows_basic_info(self, mock_client_cls, tmp_path, capsys):
        mock_client = MagicMock()
        mock_client.get_me.return_value = {"username": "testbot", "karma": 42}
        mock_client.get_unread_count.return_value = {"unread_count": 3}
        mock_client_cls.return_value = mock_client

        config_path = write_config(tmp_path)
        cmd_status(make_status_args(config_path))

        output = capsys.readouterr().out
        assert "TestBot" in output
        assert "testbot" in output
        assert "42" in output
        assert "3" in output

    @patch("colony_agent.cli.ColonyClient")
    def test_shows_memory_stats(self, mock_client_cls, tmp_path, capsys):
        mock_client = MagicMock()
        mock_client.get_me.return_value = {"username": "testbot", "karma": 0}
        mock_client.get_unread_count.return_value = {"unread_count": 0}
        mock_client_cls.return_value = mock_client

        # Create a memory file with some messages
        memory_path = tmp_path / "memory.json"
        memory_path.write_text(json.dumps([
            {"role": "user", "content": "post by alice: Hello"},
            {"role": "assistant", "content": "Interesting post."},
            {"role": "user", "content": "DM from bob: Hey there"},
            {"role": "assistant", "content": "Hi bob!"},
            {"role": "user", "content": "post by alice: Follow-up"},
            {"role": "assistant", "content": "Good point alice."},
        ]))

        config_path = write_config(tmp_path)
        cmd_status(make_status_args(config_path))

        output = capsys.readouterr().out
        assert "6 messages" in output
        assert "alice" in output
        assert "bob" in output

    @patch("colony_agent.cli.ColonyClient")
    def test_shows_trimmed_indicator(self, mock_client_cls, tmp_path, capsys):
        mock_client = MagicMock()
        mock_client.get_me.return_value = {"username": "testbot", "karma": 0}
        mock_client.get_unread_count.return_value = {"unread_count": 0}
        mock_client_cls.return_value = mock_client

        memory_path = tmp_path / "memory.json"
        memory_path.write_text(json.dumps([
            {"role": "assistant", "content": "[Memory summary of earlier interactions]\nTalked to alice about CRDTs."},
            {"role": "user", "content": "New message"},
        ]))

        config_path = write_config(tmp_path)
        cmd_status(make_status_args(config_path))

        output = capsys.readouterr().out
        assert "trimmed" in output.lower()

    @patch("colony_agent.cli.ColonyClient")
    def test_empty_memory(self, mock_client_cls, tmp_path, capsys):
        mock_client = MagicMock()
        mock_client.get_me.return_value = {"username": "testbot", "karma": 0}
        mock_client.get_unread_count.return_value = {"unread_count": 0}
        mock_client_cls.return_value = mock_client

        config_path = write_config(tmp_path)
        cmd_status(make_status_args(config_path))

        output = capsys.readouterr().out
        assert "0 messages" in output
        assert "Agents interacted" not in output

    @patch("colony_agent.cli.ColonyClient")
    def test_api_failure_graceful(self, mock_client_cls, tmp_path, capsys):
        mock_client = MagicMock()
        mock_client.get_me.side_effect = ColonyAPIError("fail", status=500)
        mock_client.get_unread_count.side_effect = ColonyAPIError("fail", status=500)
        mock_client_cls.return_value = mock_client

        config_path = write_config(tmp_path)
        cmd_status(make_status_args(config_path))

        output = capsys.readouterr().out
        assert "TestBot" in output
        assert "?" in output


def make_init_args(tmp_path, **overrides):
    """Create args namespace for cmd_init."""
    defaults = dict(
        name="test-agent",
        display_name=None,
        bio="A test agent.",
        personality=None,
        interests=None,
        config=str(tmp_path / "agent.json"),
    )
    defaults.update(overrides)
    args = MagicMock()
    for k, v in defaults.items():
        setattr(args, k, v)
    return args


class TestCmdInit:
    @patch("colony_agent.cli.ColonyClient")
    def test_creates_config_file(self, mock_client_cls, tmp_path):
        mock_client_cls.register.return_value = {"api_key": "col_test_key_123"}
        config_path = tmp_path / "agent.json"
        cmd_init(make_init_args(tmp_path))

        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert config["api_key"] == "col_test_key_123"
        assert config["identity"]["name"] == "test-agent"
        assert config["identity"]["bio"] == "A test agent."

    @patch("colony_agent.cli.ColonyClient")
    def test_uses_display_name(self, mock_client_cls, tmp_path):
        mock_client_cls.register.return_value = {"api_key": "col_x"}
        cmd_init(make_init_args(tmp_path, display_name="Test Agent"))

        config = json.loads((tmp_path / "agent.json").read_text())
        assert config["identity"]["name"] == "Test Agent"

    @patch("colony_agent.cli.ColonyClient")
    def test_custom_personality_and_interests(self, mock_client_cls, tmp_path):
        mock_client_cls.register.return_value = {"api_key": "col_x"}
        cmd_init(make_init_args(
            tmp_path,
            personality="Very serious and technical.",
            interests="robotics, CRDTs, consensus",
        ))

        config = json.loads((tmp_path / "agent.json").read_text())
        assert config["identity"]["personality"] == "Very serious and technical."
        assert config["identity"]["interests"] == ["robotics", "CRDTs", "consensus"]

    @patch("colony_agent.cli.ColonyClient")
    def test_username_taken_error(self, mock_client_cls, tmp_path, capsys):
        mock_client_cls.register.side_effect = ColonyAPIError(
            "Username already taken", status=409,
        )
        with pytest.raises(SystemExit):
            cmd_init(make_init_args(tmp_path))

        output = capsys.readouterr().out
        assert "already taken" in output.lower()

    @patch("colony_agent.cli.ColonyClient")
    def test_other_registration_error(self, mock_client_cls, tmp_path, capsys):
        mock_client_cls.register.side_effect = ColonyAPIError(
            "Internal server error", status=500,
        )
        with pytest.raises(SystemExit):
            cmd_init(make_init_args(tmp_path))

        output = capsys.readouterr().out
        assert "Registration failed" in output

    @patch("colony_agent.cli.ColonyClient")
    def test_existing_config_blocked(self, mock_client_cls, tmp_path, capsys):
        config_path = tmp_path / "agent.json"
        config_path.write_text("{}")

        with pytest.raises(SystemExit):
            cmd_init(make_init_args(tmp_path))

        output = capsys.readouterr().out
        assert "already exists" in output.lower()

    @patch("colony_agent.cli.ColonyClient")
    def test_interactive_prompts(self, mock_client_cls, tmp_path, monkeypatch):
        mock_client_cls.register.return_value = {"api_key": "col_interactive"}

        inputs = iter(["my-bot", "My Bot", "I help with things", "Cheerful and curious", "music, art, design"])
        monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

        cmd_init(make_init_args(tmp_path, name=None, bio=None))

        config = json.loads((tmp_path / "agent.json").read_text())
        assert config["identity"]["name"] == "My Bot"
        assert config["identity"]["bio"] == "I help with things"
        assert config["identity"]["personality"] == "Cheerful and curious"
        assert config["identity"]["interests"] == ["music", "art", "design"]

    @patch("colony_agent.cli.ColonyClient")
    def test_interactive_defaults(self, mock_client_cls, tmp_path, monkeypatch):
        mock_client_cls.register.return_value = {"api_key": "col_defaults"}

        # User presses enter for all defaults except username (required)
        inputs = iter(["my-bot", "", "", "", ""])
        monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

        cmd_init(make_init_args(tmp_path, name=None, bio=None))

        config = json.loads((tmp_path / "agent.json").read_text())
        assert config["identity"]["name"] == "my-bot"  # default = username
        assert config["identity"]["bio"] == "An AI agent on The Colony."
        assert config["identity"]["personality"] == "Friendly, curious, and helpful."
        assert config["identity"]["interests"] == ["AI", "agents", "technology"]


def make_test_llm_args(config_path: str, prompt: str | None = None):
    """Create args for cmd_test_llm."""
    args = MagicMock()
    args.config = config_path
    args.prompt = prompt
    return args


class TestCmdTestLLM:
    @patch("colony_agent.llm.chat", return_value="Hello! I am TestBot, nice to meet you.")
    def test_successful_connection(self, mock_chat, tmp_path, capsys):
        config_path = write_config(tmp_path)
        cmd_test_llm(make_test_llm_args(config_path))

        output = capsys.readouterr().out
        assert "Hello! I am TestBot" in output
        assert "working" in output.lower()

    @patch("colony_agent.llm.chat", return_value="")
    def test_no_response_shows_troubleshooting(self, mock_chat, tmp_path, capsys):
        config_path = write_config(tmp_path)
        with pytest.raises(SystemExit):
            cmd_test_llm(make_test_llm_args(config_path))

        output = capsys.readouterr().out
        assert "No response" in output
        assert "localhost:11434" in output

    @patch("colony_agent.llm.chat", return_value="Custom response.")
    def test_custom_prompt(self, mock_chat, tmp_path, capsys):
        config_path = write_config(tmp_path)
        cmd_test_llm(make_test_llm_args(config_path, prompt="What is 2+2?"))

        output = capsys.readouterr().out
        assert "Custom response" in output
        # Verify the custom prompt was sent
        call_messages = mock_chat.call_args[0][1]
        assert call_messages[-1]["content"] == "What is 2+2?"

    @patch("colony_agent.llm.chat", return_value="Works!")
    def test_shows_llm_config(self, mock_chat, tmp_path, capsys):
        config_path = write_config(tmp_path)
        cmd_test_llm(make_test_llm_args(config_path))

        output = capsys.readouterr().out
        assert "openai-compatible" in output
        assert "qwen3:8b" in output or "localhost" in output

    @patch("colony_agent.llm.chat", return_value="Response!")
    def test_shows_response_time(self, mock_chat, tmp_path, capsys):
        config_path = write_config(tmp_path)
        cmd_test_llm(make_test_llm_args(config_path))

        output = capsys.readouterr().out
        assert "s)" in output  # e.g. "(0.1s)"

    @patch("colony_agent.llm.chat", return_value="")
    def test_warns_about_missing_api_key(self, mock_chat, tmp_path, capsys):
        config_path = write_config(tmp_path)
        with pytest.raises(SystemExit):
            cmd_test_llm(make_test_llm_args(config_path))

        output = capsys.readouterr().out
        assert "API key" in output or "api_key" in output
