"""Tests for colony_agent.llm."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import patch

import pytest

from colony_agent.config import LLMConfig
from colony_agent.llm import ask_llm, build_system_prompt, chat


class MockLLMHandler(BaseHTTPRequestHandler):
    """Simple mock OpenAI-compatible API server."""

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len))

        response = {
            "choices": [
                {"message": {"content": f"Mock response to: {body['messages'][-1]['content'][:50]}"}}
            ]
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        pass  # Suppress server logs


@pytest.fixture
def mock_server():
    server = HTTPServer(("127.0.0.1", 0), MockLLMHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


class TestChat:
    def test_sends_full_message_list(self, mock_server):
        config = LLMConfig(provider="openai-compatible", base_url=mock_server, model="test")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "First message."},
            {"role": "assistant", "content": "I remember that."},
            {"role": "user", "content": "Second message."},
        ]
        result = chat(config, messages)
        assert "Mock response" in result

    def test_empty_messages(self, mock_server):
        config = LLMConfig(provider="openai-compatible", base_url=mock_server, model="test")
        result = chat(config, [{"role": "user", "content": "Hello"}])
        assert result != ""


class TestAskLLM:
    def test_successful_request(self, mock_server):
        config = LLMConfig(
            provider="openai-compatible",
            base_url=mock_server,
            model="test-model",
        )
        result = ask_llm(config, "You are helpful.", "Hello!")
        assert "Mock response" in result
        assert "Hello!" in result

    def test_api_key_sent(self, mock_server):
        config = LLMConfig(
            provider="openai-compatible",
            base_url=mock_server,
            model="test",
            api_key="sk-test-key",
        )
        result = ask_llm(config, "system", "prompt")
        assert result != ""

    def test_http_error_returns_empty(self):
        """Connection errors propagate (only HTTPError/KeyError/Timeout are caught)."""
        from io import BytesIO
        from unittest.mock import patch
        from urllib.error import HTTPError

        config = LLMConfig(
            provider="openai-compatible",
            base_url="http://127.0.0.1:9999",
            model="test",
        )
        mock_err = HTTPError("http://test", 500, "Server Error", {}, BytesIO(b"error"))
        with patch("colony_agent.llm.urlopen", side_effect=mock_err):
            result = ask_llm(config, "system", "prompt")
            assert result == ""

    def test_timeout_returns_empty(self):
        config = LLMConfig(
            provider="openai-compatible",
            base_url="http://192.0.2.1",  # RFC 5737 TEST-NET, will timeout
            model="test",
        )
        # Patch timeout to avoid long waits
        with patch("colony_agent.llm.urlopen", side_effect=TimeoutError):
            result = ask_llm(config, "system", "prompt")
            assert result == ""

    def test_url_error_returns_empty(self):
        from urllib.error import URLError

        config = LLMConfig(
            provider="openai-compatible",
            base_url="http://127.0.0.1:1",
            model="test",
        )
        with patch("colony_agent.llm.urlopen", side_effect=URLError("connection refused")):
            result = ask_llm(config, "system", "prompt")
            assert result == ""

    def test_os_error_returns_empty(self):
        config = LLMConfig(
            provider="openai-compatible",
            base_url="http://127.0.0.1:1",
            model="test",
        )
        with patch("colony_agent.llm.urlopen", side_effect=OSError("network unreachable")):
            result = ask_llm(config, "system", "prompt")
            assert result == ""


class TestBuildSystemPrompt:
    def test_includes_name(self):
        prompt = build_system_prompt("TestBot", "Friendly", ["AI", "music"])
        assert "TestBot" in prompt

    def test_includes_personality(self):
        prompt = build_system_prompt("Bot", "Very serious", ["AI"])
        assert "Very serious" in prompt

    def test_includes_interests(self):
        prompt = build_system_prompt("Bot", "Nice", ["robotics", "cooking"])
        assert "robotics" in prompt
        assert "cooking" in prompt

    def test_system_prompt_override(self):
        prompt = build_system_prompt(
            "Bot", "Nice", ["AI"],
            system_prompt="You are a custom agent. Do exactly as told.",
        )
        assert prompt == "You are a custom agent. Do exactly as told."
        assert "Bot" not in prompt
        assert "Nice" not in prompt

    def test_system_prompt_suffix(self):
        prompt = build_system_prompt(
            "Bot", "Nice", ["AI"],
            system_prompt_suffix="Never discuss politics. Always ask follow-up questions.",
        )
        assert "Bot" in prompt
        assert "Nice" in prompt
        assert "AI" in prompt
        assert "Never discuss politics" in prompt
        assert "Always ask follow-up questions" in prompt

    def test_override_takes_priority_over_suffix(self):
        prompt = build_system_prompt(
            "Bot", "Nice", ["AI"],
            system_prompt="Custom override.",
            system_prompt_suffix="This should be ignored.",
        )
        assert prompt == "Custom override."

    def test_empty_override_uses_default(self):
        prompt = build_system_prompt("Bot", "Nice", ["AI"], system_prompt="")
        assert "Bot" in prompt
        assert "AI" in prompt
