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


def begun(api_key="col_test_key_123", claim_token="claim-tok-1"):
    """A `register_begin` response."""
    return {
        "status": "pending",
        "api_key": api_key,
        "claim_token": claim_token,
        "expires_at": "2026-08-11T12:00:00Z",
    }


# NOTE: autospec=True is load-bearing on every patch below, not stylistic.
# These tests previously used a bare MagicMock and set `.register.return_value`.
# A bare MagicMock invents any attribute you touch, so they kept passing after
# colony-sdk removed `ColonyClient.register` — `colony-agent init` raised
# AttributeError on a real machine while the suite stayed green. autospec binds
# the mock to the real class, so a call to a method the SDK no longer has fails
# here instead of in a new user's terminal, and the call *signatures* are
# checked too.
class TestCmdInit:
    @patch("colony_agent.cli.ColonyClient", autospec=True)
    def test_creates_config_file(self, mock_client_cls, tmp_path):
        mock_client_cls.register_begin.return_value = begun()
        config_path = tmp_path / "agent.json"
        cmd_init(make_init_args(tmp_path))

        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert config["api_key"] == "col_test_key_123"
        assert config["identity"]["name"] == "test-agent"
        assert config["identity"]["bio"] == "A test agent."

    @patch("colony_agent.cli.ColonyClient", autospec=True)
    def test_uses_display_name(self, mock_client_cls, tmp_path):
        mock_client_cls.register_begin.return_value = begun("col_x")
        cmd_init(make_init_args(tmp_path, display_name="Test Agent"))

        config = json.loads((tmp_path / "agent.json").read_text())
        assert config["identity"]["name"] == "Test Agent"

    @patch("colony_agent.cli.ColonyClient", autospec=True)
    def test_custom_personality_and_interests(self, mock_client_cls, tmp_path):
        mock_client_cls.register_begin.return_value = begun("col_x")
        cmd_init(make_init_args(
            tmp_path,
            personality="Very serious and technical.",
            interests="robotics, CRDTs, consensus",
        ))

        config = json.loads((tmp_path / "agent.json").read_text())
        assert config["identity"]["personality"] == "Very serious and technical."
        assert config["identity"]["interests"] == ["robotics", "CRDTs", "consensus"]

    @patch("colony_agent.cli.ColonyClient", autospec=True)
    def test_username_taken_error(self, mock_client_cls, tmp_path, capsys):
        mock_client_cls.register_begin.side_effect = ColonyAPIError(
            "Username already taken", status=409,
        )
        with pytest.raises(SystemExit):
            cmd_init(make_init_args(tmp_path))

        output = capsys.readouterr().out
        assert "already taken" in output.lower()

    @patch("colony_agent.cli.ColonyClient", autospec=True)
    def test_other_registration_error(self, mock_client_cls, tmp_path, capsys):
        mock_client_cls.register_begin.side_effect = ColonyAPIError(
            "Internal server error", status=500,
        )
        with pytest.raises(SystemExit):
            cmd_init(make_init_args(tmp_path))

        output = capsys.readouterr().out
        assert "Registration failed" in output

    @patch("colony_agent.cli.ColonyClient", autospec=True)
    def test_existing_config_blocked(self, mock_client_cls, tmp_path, capsys):
        config_path = tmp_path / "agent.json"
        config_path.write_text("{}")

        with pytest.raises(SystemExit):
            cmd_init(make_init_args(tmp_path))

        output = capsys.readouterr().out
        assert "already exists" in output.lower()

    @patch("colony_agent.cli.ColonyClient", autospec=True)
    def test_interactive_prompts(self, mock_client_cls, tmp_path, monkeypatch):
        mock_client_cls.register_begin.return_value = begun("col_interactive")

        inputs = iter(["my-bot", "My Bot", "I help with things", "Cheerful and curious", "music, art, design"])
        monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

        cmd_init(make_init_args(tmp_path, name=None, bio=None))

        config = json.loads((tmp_path / "agent.json").read_text())
        assert config["identity"]["name"] == "My Bot"
        assert config["identity"]["bio"] == "I help with things"
        assert config["identity"]["personality"] == "Cheerful and curious"
        assert config["identity"]["interests"] == ["music", "art", "design"]

    @patch("colony_agent.cli.ColonyClient", autospec=True)
    def test_interactive_defaults(self, mock_client_cls, tmp_path, monkeypatch):
        mock_client_cls.register_begin.return_value = begun("col_defaults")

        # User presses enter for all defaults except username (required)
        inputs = iter(["my-bot", "", "", "", ""])
        monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

        cmd_init(make_init_args(tmp_path, name=None, bio=None))

        config = json.loads((tmp_path / "agent.json").read_text())
        assert config["identity"]["name"] == "my-bot"  # default = username
        assert config["identity"]["bio"] == "An AI agent on The Colony."
        assert config["identity"]["personality"] == "Friendly, curious, and helpful."
        assert config["identity"]["interests"] == ["AI", "agents", "technology"]


class TestCmdInitTwoStepRegistration:
    """The begin → persist → read back → confirm ordering.

    `register_begin` leaves the account pending and unusable; only
    `register_confirm` activates it, and only by proving the key was kept. The
    assertions here are about *order*, because an implementation that confirms
    before saving would pass every test in TestCmdInit above while defeating the
    entire point of the two-step flow.
    """

    @patch("colony_agent.cli.ColonyClient", autospec=True)
    def test_confirms_with_last_six_chars_of_the_key(self, mock_client_cls, tmp_path):
        mock_client_cls.register_begin.return_value = begun("col_abcdef_XYZ789")
        cmd_init(make_init_args(tmp_path))

        mock_client_cls.register_confirm.assert_called_once_with("claim-tok-1", "XYZ789")

    @patch("colony_agent.cli.ColonyClient", autospec=True)
    def test_key_is_on_disk_before_confirm_runs(self, mock_client_cls, tmp_path):
        """The whole gate: confirm must not run until the key is saved."""
        config_path = tmp_path / "agent.json"
        seen = {}

        def record(claim_token, fingerprint):
            seen["existed"] = config_path.exists()
            seen["key"] = (
                json.loads(config_path.read_text()).get("api_key")
                if config_path.exists() else None
            )

        mock_client_cls.register_begin.return_value = begun()
        mock_client_cls.register_confirm.side_effect = record
        cmd_init(make_init_args(tmp_path))

        assert seen["existed"] is True
        assert seen["key"] == "col_test_key_123"

    @patch("colony_agent.cli.ColonyClient", autospec=True)
    def test_missing_claim_token_aborts_before_confirm(self, mock_client_cls, tmp_path, capsys):
        """A response without a claim_token can never be activated.

        Carrying on would leave the user with a config file and a pending
        account that silently cannot post.
        """
        mock_client_cls.register_begin.return_value = {"api_key": "col_only"}
        with pytest.raises(SystemExit):
            cmd_init(make_init_args(tmp_path))

        mock_client_cls.register_confirm.assert_not_called()
        assert "claim_token" in capsys.readouterr().out

    @patch("colony_agent.cli.ColonyClient", autospec=True)
    def test_already_active_is_tolerated(self, mock_client_cls, tmp_path):
        """REGISTER_ALREADY_ACTIVE is the documented idempotent guard — the
        account works, so init should finish normally rather than error."""
        mock_client_cls.register_begin.return_value = begun()
        mock_client_cls.register_confirm.side_effect = ColonyAPIError(
            "already active", status=409, code="REGISTER_ALREADY_ACTIVE",
        )
        cmd_init(make_init_args(tmp_path))  # must not raise

        assert (tmp_path / "agent.json").exists()

    @patch("colony_agent.cli.ColonyClient", autospec=True)
    def test_other_confirm_failure_exits(self, mock_client_cls, tmp_path, capsys):
        """Control for the test above.

        Without this, an `except ColonyAPIError: pass` would satisfy the
        already-active case and quietly swallow a genuine activation failure,
        telling the user everything worked.
        """
        mock_client_cls.register_begin.return_value = begun()
        mock_client_cls.register_confirm.side_effect = ColonyAPIError(
            "claim expired", status=410, code="REGISTER_CLAIM_EXPIRED",
        )
        with pytest.raises(SystemExit):
            cmd_init(make_init_args(tmp_path))

        assert "INACTIVE" in capsys.readouterr().out

    @patch("colony_agent.cli.ColonyClient", autospec=True)
    def test_removed_one_step_register_is_not_used(self, mock_client_cls, tmp_path):
        """Regression guard for the break this file exists to prevent.

        `ColonyClient.register` no longer exists in colony-sdk, so under
        autospec the attribute is absent — asserting that directly documents
        why the flow above cannot go back to being one call.
        """
        mock_client_cls.register_begin.return_value = begun()
        cmd_init(make_init_args(tmp_path))

        assert not hasattr(mock_client_cls, "register")
        mock_client_cls.register_begin.assert_called_once()


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
