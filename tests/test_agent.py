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
        llm=LLMConfig(provider="none"),
        state_file=str(tmp_path / "state.json"),
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


@pytest.fixture
def agent(tmp_path):
    config = make_config(tmp_path)
    with patch("colony_agent.agent.ColonyClient"):
        a = ColonyAgent(config)
        a.client = MagicMock()
        return a


class TestColonyAgentInit:
    def test_creates_agent(self, tmp_path):
        config = make_config(tmp_path)
        with patch("colony_agent.agent.ColonyClient"):
            agent = ColonyAgent(config)
            assert agent.config == config
            assert agent.dry_run is False

    def test_dry_run_flag(self, tmp_path):
        config = make_config(tmp_path)
        with patch("colony_agent.agent.ColonyClient"):
            agent = ColonyAgent(config, dry_run=True)
            assert agent.dry_run is True


class TestIntroduce:
    def test_skips_when_disabled(self, agent):
        agent.config.behavior.introduce_on_first_run = False
        agent.heartbeat()
        agent.client.create_post.assert_not_called()

    def test_introduces_on_first_run(self, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(
                introduce_on_first_run=True,
                reply_to_dms=False,
            ),
        )
        with patch("colony_agent.agent.ColonyClient"):
            agent = ColonyAgent(config)
            agent.client = MagicMock()
            agent.client.get_posts.return_value = {"posts": []}
            agent.client.get_me.return_value = {"username": "testbot"}

            agent.heartbeat()
            agent.client.create_post.assert_called_once()
            call_kwargs = agent.client.create_post.call_args
            assert "introductions" in str(call_kwargs)

    def test_does_not_reintroduce(self, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(
                introduce_on_first_run=True,
                reply_to_dms=False,
            ),
        )
        with patch("colony_agent.agent.ColonyClient"):
            agent = ColonyAgent(config)
            agent.client = MagicMock()
            agent.client.get_posts.return_value = {"posts": []}
            agent.client.get_me.return_value = {"username": "testbot"}

            agent.heartbeat()
            agent.heartbeat()
            # Should only introduce once
            assert agent.client.create_post.call_count == 1

    def test_dry_run_does_not_post(self, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(
                introduce_on_first_run=True,
                reply_to_dms=False,
            ),
        )
        with patch("colony_agent.agent.ColonyClient"):
            agent = ColonyAgent(config, dry_run=True)
            agent.client = MagicMock()
            agent.client.get_posts.return_value = {"posts": []}

            agent.heartbeat()
            agent.client.create_post.assert_not_called()

    def test_skips_when_post_limit_reached(self, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(
                introduce_on_first_run=True,
                reply_to_dms=False,
                max_posts_per_day=0,
            ),
        )
        with patch("colony_agent.agent.ColonyClient"):
            agent = ColonyAgent(config)
            agent.client = MagicMock()
            agent.client.get_posts.return_value = {"posts": []}

            agent.heartbeat()
            agent.client.create_post.assert_not_called()


class TestBrowseAndEngage:
    def test_skips_own_posts(self, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1",
                    "title": "AI and testing combined",
                    "body": "My post about AI testing.",
                    "author": {"username": "testbot"},
                }
            ]
        }
        agent.heartbeat()
        agent.client.vote_post.assert_not_called()
        agent.client.create_comment.assert_not_called()

    def test_skips_seen_posts(self, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.state.mark_seen("p1")
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1",
                    "title": "AI stuff",
                    "body": "Content.",
                    "author": {"username": "other"},
                }
            ]
        }
        agent.heartbeat()
        agent.client.vote_post.assert_not_called()

    def test_votes_on_matching_post(self, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1",
                    "title": "New AI research",
                    "body": "Groundbreaking stuff.",
                    "author": {"username": "other"},
                }
            ]
        }
        agent.heartbeat()
        agent.client.vote_post.assert_called_once_with("p1")

    def test_comments_on_highly_relevant_post(self, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1",
                    "title": "AI testing framework",
                    "body": "A new approach to AI testing.",
                    "author": {"username": "other"},
                    "post_type": "discussion",
                }
            ]
        }
        agent.heartbeat()
        agent.client.create_comment.assert_called_once()

    def test_respects_vote_limit(self, agent):
        agent.config.behavior.max_votes_per_day = 1
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {"id": "p1", "title": "AI news", "body": "x", "author": {"username": "a"}},
                {"id": "p2", "title": "AI update", "body": "y", "author": {"username": "b"}},
            ]
        }
        agent.heartbeat()
        assert agent.client.vote_post.call_count == 1

    def test_respects_comment_limit(self, agent):
        agent.config.behavior.max_comments_per_day = 1
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "AI testing", "body": "both",
                    "author": {"username": "a"}, "post_type": "discussion",
                },
                {
                    "id": "p2", "title": "AI testing", "body": "both",
                    "author": {"username": "b"}, "post_type": "discussion",
                },
            ]
        }
        agent.heartbeat()
        assert agent.client.create_comment.call_count == 1

    def test_dry_run_no_api_calls(self, tmp_path):
        config = make_config(tmp_path)
        with patch("colony_agent.agent.ColonyClient"):
            agent = ColonyAgent(config, dry_run=True)
            agent.client = MagicMock()
            agent.client.get_me.return_value = {"username": "testbot"}
            agent.client.get_posts.return_value = {
                "posts": [
                    {
                        "id": "p1",
                        "title": "AI testing framework",
                        "body": "New AI testing approach.",
                        "author": {"username": "other"},
                    }
                ]
            }
            agent.heartbeat()
            agent.client.vote_post.assert_not_called()
            agent.client.create_comment.assert_not_called()


class TestCheckDMs:
    def test_skips_when_no_unread(self, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(reply_to_dms=True, introduce_on_first_run=False),
        )
        with patch("colony_agent.agent.ColonyClient"):
            agent = ColonyAgent(config)
            agent.client = MagicMock()
            agent.client.get_unread_count.return_value = {"unread_count": 0}
            agent.client.get_posts.return_value = {"posts": []}

            agent.heartbeat()
            agent.client.send_message.assert_not_called()

    def test_replies_to_unread_dm(self, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(reply_to_dms=True, introduce_on_first_run=False),
        )
        with patch("colony_agent.agent.ColonyClient"):
            agent = ColonyAgent(config)
            agent.client = MagicMock()
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
            call_args = agent.client.send_message.call_args[0]
            assert call_args[0] == "alice"

    def test_skips_own_last_message(self, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(reply_to_dms=True, introduce_on_first_run=False),
        )
        with patch("colony_agent.agent.ColonyClient"):
            agent = ColonyAgent(config)
            agent.client = MagicMock()
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


class TestRunOnce:
    def test_saves_state(self, agent):
        agent.client.get_posts.return_value = {"posts": []}
        agent.run_once()
        assert agent.state.last_heartbeat > 0


class TestMyUsername:
    def test_caches_username(self, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        assert agent._my_username() == "testbot"
        assert agent._my_username() == "testbot"
        agent.client.get_me.assert_called_once()
