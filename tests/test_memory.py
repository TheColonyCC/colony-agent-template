"""Tests for colony_agent.memory."""

import json

import pytest

from colony_agent.memory import AgentMemory


@pytest.fixture
def memory(tmp_path):
    return AgentMemory(str(tmp_path / "memory.json"), max_messages=10)


class TestAgentMemory:
    def test_fresh_memory_is_empty(self, memory):
        assert len(memory) == 0
        assert memory.messages == []

    def test_add_message(self, memory):
        memory.add("user", "Hello")
        memory.add("assistant", "Hi there!")
        assert len(memory) == 2
        assert memory.messages[0] == {"role": "user", "content": "Hello"}
        assert memory.messages[1] == {"role": "assistant", "content": "Hi there!"}

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "memory.json")
        m1 = AgentMemory(path)
        m1.add("user", "Remember this")
        m1.add("assistant", "I will remember.")
        m1.save()

        m2 = AgentMemory(path)
        assert len(m2) == 2
        assert m2.messages[0]["content"] == "Remember this"
        assert m2.messages[1]["content"] == "I will remember."

    def test_get_messages_for_llm(self, memory):
        memory.add("user", "What do you think?")
        memory.add("assistant", "I think it's great.")
        messages = memory.get_messages_for_llm("You are a test agent.")
        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "You are a test agent."}
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_system_prompt_always_fresh(self, memory):
        memory.add("user", "Hello")
        msgs1 = memory.get_messages_for_llm("Prompt version 1")
        msgs2 = memory.get_messages_for_llm("Prompt version 2")
        assert msgs1[0]["content"] == "Prompt version 1"
        assert msgs2[0]["content"] == "Prompt version 2"

    def test_needs_trim(self, memory):
        assert not memory.needs_trim()
        for i in range(11):
            memory.add("user", f"Message {i}")
        assert memory.needs_trim()

    def test_trim(self, memory):
        for i in range(11):
            memory.add("user", f"Message {i}")
        assert memory.needs_trim()

        memory.trim("Summary of old messages.")
        assert not memory.needs_trim()
        # Should have summary + recent half
        assert memory.messages[0]["content"].startswith("[Memory summary")
        assert "Summary of old messages." in memory.messages[0]["content"]

    def test_trim_keeps_recent(self, memory):
        for i in range(12):
            memory.add("user", f"Message {i}")
        memory.trim("Old stuff happened.")
        # max_messages=10, keep half=5, plus summary=6 total
        assert len(memory) == 6
        # Last message should be the most recent
        assert memory.messages[-1]["content"] == "Message 11"

    def test_clear(self, memory):
        memory.add("user", "Hello")
        memory.add("assistant", "Hi")
        memory.clear()
        assert len(memory) == 0

    def test_atomic_save(self, tmp_path):
        path = tmp_path / "memory.json"
        m = AgentMemory(str(path))
        m.add("user", "test")
        m.save()
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "memory.json"
        path.write_text("not valid json")
        m = AgentMemory(str(path))
        assert len(m) == 0  # graceful fallback

    def test_load_wrong_type(self, tmp_path):
        path = tmp_path / "memory.json"
        path.write_text(json.dumps({"not": "a list"}))
        m = AgentMemory(str(path))
        assert len(m) == 0  # only loads lists
