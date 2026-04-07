"""Tests for colony_agent.agent."""

from unittest.mock import MagicMock, patch

import pytest

from colony_agent.agent import ColonyAgent
from colony_agent.config import AgentConfig, BehaviorConfig, IdentityConfig, LLMConfig


def make_config(tmp_path, **overrides):
    """Create a minimal AgentConfig for testing."""
    defaults = dict(
        api_key="col_test_key",
        identity=IdentityConfig(
            name="TestBot",
            bio="A test agent.",
            interests=["AI", "testing"],
            colonies=["general"],
        ),
        behavior=BehaviorConfig(
            heartbeat_interval=60,
            introduce_on_first_run=False,
            reply_to_dms=False,
        ),
        llm=LLMConfig(provider="openai-compatible"),
        state_file=str(tmp_path / "state.json"),
        memory_file=str(tmp_path / "memory.json"),
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


def make_agent(config):
    """Create a ColonyAgent with mocked ColonyClient."""
    with patch("colony_agent.agent.ColonyClient"):
        agent = ColonyAgent(config)
        agent.client = MagicMock()
        return agent


@pytest.fixture
def agent(tmp_path):
    return make_agent(make_config(tmp_path))


class TestColonyAgentInit:
    def test_creates_agent(self, tmp_path):
        config = make_config(tmp_path)
        with patch("colony_agent.agent.ColonyClient"):
            agent = ColonyAgent(config)
            assert agent.config == config
            assert agent.dry_run is False
            assert len(agent.memory) == 0

    def test_dry_run_flag(self, tmp_path):
        config = make_config(tmp_path)
        with patch("colony_agent.agent.ColonyClient"):
            agent = ColonyAgent(config, dry_run=True)
            assert agent.dry_run is True


class TestConversationalMemory:
    @patch("colony_agent.agent.chat", return_value="I remember alice.")
    def test_converse_adds_to_memory(self, mock_chat, agent):
        result = agent._converse("Tell me about alice.")
        assert result == "I remember alice."
        assert len(agent.memory) == 2  # user + assistant
        assert agent.memory.messages[0]["role"] == "user"
        assert agent.memory.messages[1]["role"] == "assistant"

    @patch("colony_agent.agent.chat", return_value="First response.")
    def test_memory_persists_across_calls(self, mock_chat, agent):
        agent._converse("First question")
        agent._converse("Second question")
        assert len(agent.memory) == 4  # 2 user + 2 assistant
        # The second LLM call should include history from the first exchange
        second_call_messages = mock_chat.call_args_list[1][0][1]
        # system + user1 + assistant1 + user2 (assistant2 added after call returns)
        assert len(second_call_messages) == 4

    @patch("colony_agent.agent.chat", return_value="")
    def test_empty_response_not_added_to_memory(self, mock_chat, agent):
        result = agent._converse("Hello?")
        assert result == ""
        assert len(agent.memory) == 1  # only the user message

    @patch("colony_agent.agent.chat", return_value="Response")
    def test_memory_saved_after_heartbeat(self, mock_chat, agent):
        agent.client.get_posts.return_value = {"posts": []}
        agent.run_once()
        assert agent.memory.path.exists()


class TestIntroduce:
    def test_skips_when_disabled(self, agent):
        agent.config.behavior.introduce_on_first_run = False
        agent.client.get_posts.return_value = {"posts": []}
        agent.heartbeat()
        agent.client.create_post.assert_not_called()

    @patch("colony_agent.agent.chat", return_value="Hello, I am TestBot.")
    def test_introduces_on_first_run(self, mock_chat, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(introduce_on_first_run=True, reply_to_dms=False),
        )
        agent = make_agent(config)
        agent.client.get_posts.return_value = {"posts": []}
        agent.client.get_me.return_value = {"username": "testbot"}

        agent.heartbeat()
        agent.client.create_post.assert_called_once()
        assert "introductions" in str(agent.client.create_post.call_args)

    @patch("colony_agent.agent.chat", return_value="Hello, I am TestBot.")
    def test_does_not_reintroduce(self, mock_chat, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(introduce_on_first_run=True, reply_to_dms=False),
        )
        agent = make_agent(config)
        agent.client.get_posts.return_value = {"posts": []}
        agent.client.get_me.return_value = {"username": "testbot"}

        agent.heartbeat()
        agent.heartbeat()
        assert agent.client.create_post.call_count == 1

    @patch("colony_agent.agent.chat", return_value="Hello!")
    def test_dry_run_does_not_post(self, mock_chat, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(introduce_on_first_run=True, reply_to_dms=False),
        )
        agent = make_agent(config)
        agent.dry_run = True
        agent.client.get_posts.return_value = {"posts": []}

        agent.heartbeat()
        agent.client.create_post.assert_not_called()

    @patch("colony_agent.agent.chat", return_value="")
    def test_skips_intro_when_llm_fails(self, mock_chat, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(introduce_on_first_run=True, reply_to_dms=False),
        )
        agent = make_agent(config)
        agent.client.get_posts.return_value = {"posts": []}

        agent.heartbeat()
        agent.client.create_post.assert_not_called()

    def test_skips_when_post_limit_reached(self, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(
                introduce_on_first_run=True, reply_to_dms=False, max_posts_per_day=0,
            ),
        )
        agent = make_agent(config)
        agent.client.get_posts.return_value = {"posts": []}

        agent.heartbeat()
        agent.client.create_post.assert_not_called()


class TestBrowseAndEngage:
    @patch("colony_agent.agent.chat", return_value="SKIP")
    def test_skips_own_posts(self, mock_chat, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "My post", "body": "Content.",
                    "author": {"username": "testbot"},
                }
            ]
        }
        agent.heartbeat()
        agent.client.vote_post.assert_not_called()

    @patch("colony_agent.agent.chat", return_value="UPVOTE")
    def test_skips_seen_posts(self, mock_chat, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.state.mark_seen("p1")
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "AI stuff", "body": "Content.",
                    "author": {"username": "other"},
                }
            ]
        }
        agent.heartbeat()
        agent.client.vote_post.assert_not_called()

    @patch("colony_agent.agent.chat", return_value="VOTE: UPVOTE\nCOMMENT: SKIP")
    def test_upvotes_when_llm_says_upvote(self, mock_chat, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "Great post",
                    "body": "Thoughtful.", "author": {"username": "other"},
                }
            ]
        }
        agent.heartbeat()
        agent.client.vote_post.assert_called_once_with("p1", 1)

    @patch("colony_agent.agent.chat", return_value="VOTE: DOWNVOTE\nCOMMENT: SKIP")
    def test_downvotes_when_llm_says_downvote(self, mock_chat, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "Buy tokens now",
                    "body": "Spam.", "author": {"username": "other"},
                }
            ]
        }
        agent.heartbeat()
        agent.client.vote_post.assert_called_once_with("p1", -1)

    @patch("colony_agent.agent.chat", return_value="SKIP")
    def test_skips_when_llm_says_skip(self, mock_chat, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "Meh",
                    "body": "Whatever.", "author": {"username": "other"},
                }
            ]
        }
        agent.heartbeat()
        agent.client.vote_post.assert_not_called()
        agent.client.create_comment.assert_not_called()

    @patch("colony_agent.agent.chat", return_value="")
    def test_no_action_when_llm_fails(self, mock_chat, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "Post", "body": "Content.",
                    "author": {"username": "other"},
                }
            ]
        }
        agent.heartbeat()
        agent.client.vote_post.assert_not_called()
        agent.client.create_comment.assert_not_called()

    @patch(
        "colony_agent.agent.chat",
        return_value="VOTE: UPVOTE\nCOMMENT: This is a great insight on testing.",
    )
    def test_votes_and_comments_in_single_response(self, mock_chat, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "AI testing",
                    "body": "New approach.", "author": {"username": "other"},
                }
            ]
        }
        agent.heartbeat()
        agent.client.vote_post.assert_called_once_with("p1", 1)
        agent.client.create_comment.assert_called_once()

    @patch("colony_agent.agent.chat", return_value="VOTE: UPVOTE\nCOMMENT: SKIP")
    def test_respects_vote_limit(self, mock_chat, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(
                max_votes_per_day=1, introduce_on_first_run=False, reply_to_dms=False,
            ),
        )
        agent = make_agent(config)
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {"id": "p1", "title": "Post 1", "body": "x", "author": {"username": "a"}},
                {"id": "p2", "title": "Post 2", "body": "y", "author": {"username": "b"}},
            ]
        }
        agent.heartbeat()
        assert agent.client.vote_post.call_count == 1

    @patch("colony_agent.agent.chat", return_value="COMMENT: Thoughtful comment.")
    def test_respects_comment_limit(self, mock_chat, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(
                max_comments_per_day=1, introduce_on_first_run=False, reply_to_dms=False,
            ),
        )
        agent = make_agent(config)
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {"id": "p1", "title": "Post 1", "body": "x", "author": {"username": "a"}},
                {"id": "p2", "title": "Post 2", "body": "y", "author": {"username": "b"}},
            ]
        }
        agent.heartbeat()
        assert agent.client.create_comment.call_count == 1

    @patch("colony_agent.agent.chat", return_value="UPVOTE")
    def test_dry_run_no_api_calls(self, mock_chat, tmp_path):
        config = make_config(tmp_path)
        agent = make_agent(config)
        agent.dry_run = True
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "AI testing",
                    "body": "Content.", "author": {"username": "other"},
                }
            ]
        }
        agent.heartbeat()
        agent.client.vote_post.assert_not_called()
        agent.client.create_comment.assert_not_called()


class TestCheckDMs:
    @patch("colony_agent.agent.chat", return_value="SKIP")
    def test_skips_when_no_unread(self, mock_chat, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(reply_to_dms=True, introduce_on_first_run=False),
        )
        agent = make_agent(config)
        agent.client.get_unread_count.return_value = {"unread_count": 0}
        agent.client.get_posts.return_value = {"posts": []}

        agent.heartbeat()
        agent.client.send_message.assert_not_called()

    @patch("colony_agent.agent.chat", return_value="Hey, thanks for reaching out!")
    def test_replies_to_unread_dm(self, mock_chat, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(reply_to_dms=True, introduce_on_first_run=False),
        )
        agent = make_agent(config)
        agent.client.get_unread_count.return_value = {"unread_count": 1}
        agent.client._raw_request.return_value = [
            {"other_user": {"username": "alice"}}
        ]
        agent.client.get_conversation.return_value = {
            "messages": [
                {"sender": {"username": "alice"}, "body": "Hello!", "is_read": False}
            ]
        }
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {"posts": []}

        agent.heartbeat()
        agent.client.send_message.assert_called_once()
        assert agent.client.send_message.call_args[0][0] == "alice"

    @patch("colony_agent.agent.chat", return_value="SKIP")
    def test_skips_own_last_message(self, mock_chat, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(reply_to_dms=True, introduce_on_first_run=False),
        )
        agent = make_agent(config)
        agent.client.get_unread_count.return_value = {"unread_count": 1}
        agent.client._raw_request.return_value = [
            {"other_user": {"username": "alice"}}
        ]
        agent.client.get_conversation.return_value = {
            "messages": [
                {"sender": {"username": "alice"}, "body": "Hi!", "is_read": False},
                {"sender": {"username": "testbot"}, "body": "Hello!", "is_read": True},
            ]
        }
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {"posts": []}

        agent.heartbeat()
        agent.client.send_message.assert_not_called()

    @patch("colony_agent.agent.chat", return_value="")
    def test_no_reply_when_llm_fails(self, mock_chat, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(reply_to_dms=True, introduce_on_first_run=False),
        )
        agent = make_agent(config)
        agent.client.get_unread_count.return_value = {"unread_count": 1}
        agent.client._raw_request.return_value = [
            {"other_user": {"username": "alice"}}
        ]
        agent.client.get_conversation.return_value = {
            "messages": [
                {"sender": {"username": "alice"}, "body": "Hello!", "is_read": False}
            ]
        }
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {"posts": []}

        agent.heartbeat()
        agent.client.send_message.assert_not_called()

    @patch("colony_agent.agent.chat", return_value="Hey alice, great to hear from you!")
    def test_dm_reply_in_memory(self, mock_chat, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(reply_to_dms=True, introduce_on_first_run=False),
        )
        agent = make_agent(config)
        agent.client.get_unread_count.return_value = {"unread_count": 1}
        agent.client._raw_request.return_value = [
            {"other_user": {"username": "alice"}}
        ]
        agent.client.get_conversation.return_value = {
            "messages": [
                {"sender": {"username": "alice"}, "body": "What do you think about CRDTs?", "is_read": False}
            ]
        }
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {"posts": []}

        agent.heartbeat()
        # Memory should contain the DM exchange
        assert any("alice" in m["content"] for m in agent.memory.messages)
        assert any("CRDTs" in m["content"] for m in agent.memory.messages)


class TestMemoryTrimming:
    @patch("colony_agent.agent.chat")
    def test_trim_triggered_when_needed(self, mock_chat, tmp_path):
        config = make_config(tmp_path, max_memory_messages=10)
        agent = make_agent(config)
        agent.client.get_posts.return_value = {"posts": []}

        # Fill memory beyond limit
        for i in range(12):
            agent.memory.add("user", f"Message {i}")

        mock_chat.return_value = "Summary: I discussed AI with various agents."
        agent.heartbeat()
        # Memory should have been trimmed
        assert len(agent.memory) <= 10


class TestRunOnce:
    def test_saves_state_and_memory(self, agent):
        agent.client.get_posts.return_value = {"posts": []}
        agent.run_once()
        assert agent.state.last_heartbeat > 0
        assert agent.memory.path.exists()


class TestMyUsername:
    def test_caches_username(self, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        assert agent._my_username() == "testbot"
        assert agent._my_username() == "testbot"
        agent.client.get_me.assert_called_once()


class TestExtractComment:
    def test_explicit_prefix(self, agent):
        assert agent._extract_comment("COMMENT: Great post about AI.") == "Great post about AI."

    def test_with_vote_prefix(self, agent):
        result = agent._extract_comment("VOTE: UPVOTE\nCOMMENT: Really insightful work.")
        assert result == "Really insightful work."

    def test_skip_comment(self, agent):
        assert agent._extract_comment("COMMENT: SKIP") == ""

    def test_just_skip(self, agent):
        assert agent._extract_comment("SKIP") == ""

    def test_plain_comment_text(self, agent):
        assert agent._extract_comment("This is a thoughtful observation about distributed systems.") != ""

    def test_short_text_ignored(self, agent):
        assert agent._extract_comment("ok") == ""
