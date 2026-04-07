"""Tests for colony_agent.cli — status command."""

import json
from unittest.mock import MagicMock, patch

from colony_agent.cli import cmd_status


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
        from colony_sdk.client import ColonyAPIError

        mock_client = MagicMock()
        mock_client.get_me.side_effect = ColonyAPIError("fail", status=500)
        mock_client.get_unread_count.side_effect = ColonyAPIError("fail", status=500)
        mock_client_cls.return_value = mock_client

        config_path = write_config(tmp_path)
        cmd_status(make_status_args(config_path))

        output = capsys.readouterr().out
        assert "TestBot" in output
        assert "?" in output
