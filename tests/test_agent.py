"""Tests for colony_agent.agent."""

from unittest.mock import MagicMock, patch

import pytest

from colony_agent.agent import ColonyAgent
from colony_agent.config import AgentConfig, BehaviorConfig, IdentityConfig, LLMConfig
from colony_agent.llm import ContextOverflowError


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

    @patch("colony_agent.agent.chat", return_value="I think this post is really interesting and thought-provoking.")
    def test_warns_on_missing_vote_keyword(self, mock_chat, agent, caplog):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "Some post", "body": "Content.",
                    "author": {"username": "other"},
                }
            ]
        }
        import logging
        with caplog.at_level(logging.WARNING, logger="colony-agent"):
            agent.heartbeat()
        assert any("missing vote keyword" in r.message for r in caplog.records)
        agent.client.vote_post.assert_not_called()

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

    @patch("colony_agent.agent.chat", return_value="VOTE: UPVOTE\nCOMMENT: Great insight here.")
    def test_comments_include_existing_comments_context(self, mock_chat, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_comments.return_value = {
            "comments": [
                {"author": {"username": "alice"}, "body": "I agree with this approach."},
                {"author": {"username": "bob"}, "body": "Have you considered CRDTs?"},
            ]
        }
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "Distributed systems",
                    "body": "New approach.", "author": {"username": "other"},
                }
            ]
        }
        agent.heartbeat()
        # The LLM should have been called with existing comments in the prompt
        call_messages = mock_chat.call_args_list[-1][0][1]
        last_user_msg = [m for m in call_messages if m["role"] == "user"][-1]
        assert "alice" in last_user_msg["content"]
        assert "bob" in last_user_msg["content"]
        assert "CRDTs" in last_user_msg["content"]

    @patch("colony_agent.agent.chat", return_value="COMMENT: Good post.")
    def test_comments_work_when_no_existing_comments(self, mock_chat, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_comments.return_value = {"comments": []}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "New topic",
                    "body": "First post.", "author": {"username": "other"},
                }
            ]
        }
        agent.heartbeat()
        agent.client.create_comment.assert_called_once()
        # Prompt should not contain "Existing comments" section
        call_messages = mock_chat.call_args_list[-1][0][1]
        last_user_msg = [m for m in call_messages if m["role"] == "user"][-1]
        assert "Existing comments" not in last_user_msg["content"]

    @patch("colony_agent.agent.chat", return_value="COMMENT: Good post.")
    def test_comments_context_fetch_failure_graceful(self, mock_chat, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_comments.return_value = None  # simulate retry_api_call failure
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "Some post",
                    "body": "Content.", "author": {"username": "other"},
                }
            ]
        }
        agent.heartbeat()
        # Should still comment even if comments couldn't be fetched
        agent.client.create_comment.assert_called_once()

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


class TestBudgetDistribution:
    @patch("colony_agent.agent.chat", return_value="VOTE: UPVOTE\nCOMMENT: SKIP")
    def test_distributes_votes_across_colonies(self, mock_chat, tmp_path):
        config = make_config(
            tmp_path,
            identity=IdentityConfig(
                name="TestBot", interests=["AI"], colonies=["general", "findings", "questions"],
            ),
            behavior=BehaviorConfig(
                max_votes_per_day=6, max_comments_per_day=0,
                introduce_on_first_run=False, reply_to_dms=False,
            ),
        )
        agent = make_agent(config)
        agent.client.get_me.return_value = {"username": "testbot"}

        # Each colony returns 4 posts — without budgeting, first colony
        # would use all 6 votes and the others get nothing
        def fake_get_posts(colony=None, limit=10):
            return {"posts": [
                {"id": f"{colony}-{i}", "title": f"Post {i}", "body": "x", "author": {"username": "other"}}
                for i in range(4)
            ]}
        agent.client.get_posts.side_effect = fake_get_posts

        agent.heartbeat()
        # Should have voted on posts from multiple colonies, not just the first
        voted_ids = [call[0][0] for call in agent.client.vote_post.call_args_list]
        colonies_voted = {vid.split("-")[0] for vid in voted_ids}
        assert len(colonies_voted) >= 2, f"Votes only went to: {colonies_voted}"

    @patch("colony_agent.agent.chat", return_value="COMMENT: Great insight here.")
    def test_distributes_comments_across_colonies(self, mock_chat, tmp_path):
        config = make_config(
            tmp_path,
            identity=IdentityConfig(
                name="TestBot", interests=["AI"], colonies=["general", "findings"],
            ),
            behavior=BehaviorConfig(
                max_votes_per_day=0, max_comments_per_day=4,
                introduce_on_first_run=False, reply_to_dms=False,
            ),
        )
        agent = make_agent(config)
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_comments.return_value = {"comments": []}

        def fake_get_posts(colony=None, limit=10):
            return {"posts": [
                {"id": f"{colony}-{i}", "title": f"Post {i}", "body": "x", "author": {"username": "other"}}
                for i in range(4)
            ]}
        agent.client.get_posts.side_effect = fake_get_posts

        agent.heartbeat()
        commented_ids = [call[0][0] for call in agent.client.create_comment.call_args_list]
        colonies_commented = {cid.split("-")[0] for cid in commented_ids}
        assert len(colonies_commented) == 2, f"Comments only went to: {colonies_commented}"

    @patch("colony_agent.agent.chat", return_value="VOTE: UPVOTE\nCOMMENT: SKIP")
    def test_single_colony_gets_full_budget(self, mock_chat, tmp_path):
        config = make_config(
            tmp_path,
            identity=IdentityConfig(
                name="TestBot", interests=["AI"], colonies=["general"],
            ),
            behavior=BehaviorConfig(
                max_votes_per_day=5, introduce_on_first_run=False, reply_to_dms=False,
            ),
        )
        agent = make_agent(config)
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {"posts": [
            {"id": f"p{i}", "title": f"Post {i}", "body": "x", "author": {"username": "other"}}
            for i in range(10)
        ]}

        agent.heartbeat()
        assert agent.client.vote_post.call_count == 5


class TestDryRunSummary:
    @patch("colony_agent.agent.chat", return_value="VOTE: UPVOTE\nCOMMENT: Great insight.")
    def test_prints_summary(self, mock_chat, tmp_path, capsys):
        config = make_config(tmp_path)
        agent = make_agent(config)
        agent.dry_run = True
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "AI research update",
                    "body": "New findings.", "author": {"username": "alice"},
                },
                {
                    "id": "p2", "title": "Spam post",
                    "body": "Buy now.", "author": {"username": "bob"},
                },
            ]
        }
        agent.heartbeat()
        output = capsys.readouterr().out
        assert "DRY RUN SUMMARY" in output
        assert "upvote" in output.lower()
        assert "comment" in output.lower()

    @patch("colony_agent.agent.chat", return_value="SKIP")
    def test_no_summary_when_no_actions(self, mock_chat, tmp_path, capsys):
        config = make_config(tmp_path)
        agent = make_agent(config)
        agent.dry_run = True
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "Meh",
                    "body": "Nothing.", "author": {"username": "alice"},
                }
            ]
        }
        agent.heartbeat()
        output = capsys.readouterr().out
        assert "DRY RUN SUMMARY" not in output

    @patch("colony_agent.agent.chat", return_value="VOTE: UPVOTE\nCOMMENT: Interesting work.")
    def test_summary_shows_content(self, mock_chat, tmp_path, capsys):
        config = make_config(tmp_path)
        agent = make_agent(config)
        agent.dry_run = True
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "Distributed systems paper",
                    "body": "Analysis.", "author": {"username": "alice"},
                }
            ]
        }
        agent.heartbeat()
        output = capsys.readouterr().out
        assert "Interesting work" in output
        assert "Distributed systems" in output

    @patch("colony_agent.agent.chat", return_value="Hello, I am TestBot!")
    def test_summary_includes_introduction(self, mock_chat, tmp_path, capsys):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(introduce_on_first_run=True, reply_to_dms=False),
        )
        agent = make_agent(config)
        agent.dry_run = True
        agent.client.get_posts.return_value = {"posts": []}
        agent.heartbeat()
        output = capsys.readouterr().out
        assert "INTRODUCE" in output

    @patch("colony_agent.agent.chat", return_value="Hey alice, good question!")
    def test_summary_includes_dm_reply(self, mock_chat, tmp_path, capsys):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(reply_to_dms=True, introduce_on_first_run=False),
        )
        agent = make_agent(config)
        agent.dry_run = True
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
        output = capsys.readouterr().out
        assert "DM_REPLY" in output
        assert "alice" in output


class TestRepliestoOwnPosts:
    @patch("colony_agent.agent.chat", return_value="Thanks for the feedback, alice!")
    def test_replies_to_comment_on_own_post(self, mock_chat, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "My thoughts on AI",
                    "body": "Here is what I think.", "author": {"username": "testbot"},
                }
            ]
        }
        agent.client.get_comments.return_value = {
            "comments": [
                {
                    "id": "c1", "body": "Great post!",
                    "author": {"username": "alice"},
                }
            ]
        }
        agent.heartbeat()
        agent.client.create_comment.assert_called_once()

    @patch("colony_agent.agent.chat", return_value="Thanks!")
    def test_skips_own_comments(self, mock_chat, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "My post",
                    "body": "Content.", "author": {"username": "testbot"},
                }
            ]
        }
        agent.client.get_comments.return_value = {
            "comments": [
                {
                    "id": "c1", "body": "My own follow-up",
                    "author": {"username": "testbot"},
                }
            ]
        }
        agent.heartbeat()
        agent.client.create_comment.assert_not_called()

    @patch("colony_agent.agent.chat", return_value="Thanks!")
    def test_does_not_reply_twice(self, mock_chat, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.state.mark_replied_to_comment("c1")
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "My post",
                    "body": "Content.", "author": {"username": "testbot"},
                }
            ]
        }
        agent.client.get_comments.return_value = {
            "comments": [
                {
                    "id": "c1", "body": "Already replied to this",
                    "author": {"username": "alice"},
                }
            ]
        }
        agent.heartbeat()
        agent.client.create_comment.assert_not_called()

    @patch("colony_agent.agent.chat", return_value="SKIP")
    def test_skip_reply_still_marks_as_handled(self, mock_chat, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "My post",
                    "body": "Content.", "author": {"username": "testbot"},
                }
            ]
        }
        agent.client.get_comments.return_value = {
            "comments": [
                {
                    "id": "c1", "body": "Meh",
                    "author": {"username": "alice"},
                }
            ]
        }
        agent.heartbeat()
        agent.client.create_comment.assert_not_called()
        assert agent.state.has_replied_to_comment("c1")

    @patch("colony_agent.agent.chat", return_value="Thanks for the thoughtful feedback, really appreciate it!")
    def test_respects_comment_limit(self, mock_chat, tmp_path):
        config = make_config(
            tmp_path,
            behavior=BehaviorConfig(
                max_comments_per_day=1, introduce_on_first_run=False,
                reply_to_dms=False,
            ),
        )
        agent = make_agent(config)
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "My post",
                    "body": "Content.", "author": {"username": "testbot"},
                }
            ]
        }
        agent.client.get_comments.return_value = {
            "comments": [
                {"id": "c1", "body": "Comment 1", "author": {"username": "alice"}},
                {"id": "c2", "body": "Comment 2", "author": {"username": "bob"}},
            ]
        }
        agent.heartbeat()
        assert agent.client.create_comment.call_count == 1

    @patch("colony_agent.agent.chat", return_value="Good point alice!")
    def test_reply_includes_thread_context(self, mock_chat, agent):
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {
            "posts": [
                {
                    "id": "p1", "title": "My AI post",
                    "body": "Thoughts.", "author": {"username": "testbot"},
                }
            ]
        }
        agent.client.get_comments.return_value = {
            "comments": [
                {"id": "c1", "body": "I disagree", "author": {"username": "alice"}},
                {"id": "c2", "body": "I agree with alice", "author": {"username": "bob"}},
            ]
        }
        agent.heartbeat()
        # The prompt for the first comment should include thread context
        call_messages = mock_chat.call_args_list[0][0][1]
        last_user_msg = [m for m in call_messages if m["role"] == "user"][-1]
        assert "alice" in last_user_msg["content"]
        assert "bob" in last_user_msg["content"]


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


    @patch("colony_agent.agent.chat", return_value="Yes, CRDTs are great for that use case!")
    def test_dm_includes_full_thread_context(self, mock_chat, tmp_path):
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
                {
                    "sender": {"username": "alice"},
                    "body": "Hey, what do you think about CRDTs?",
                    "is_read": True,
                },
                {
                    "sender": {"username": "testbot"},
                    "body": "They're interesting for distributed state.",
                    "is_read": True,
                },
                {
                    "sender": {"username": "alice"},
                    "body": "Would they work for our voting system?",
                    "is_read": False,
                },
            ]
        }
        agent.client.get_me.return_value = {"username": "testbot"}
        agent.client.get_posts.return_value = {"posts": []}

        agent.heartbeat()
        agent.client.send_message.assert_called_once()
        # The prompt should include the full thread, not just the last message
        call_messages = mock_chat.call_args_list[-1][0][1]
        last_user_msg = [m for m in call_messages if m["role"] == "user"][-1]
        assert "CRDTs" in last_user_msg["content"]
        assert "distributed state" in last_user_msg["content"]
        assert "voting system" in last_user_msg["content"]


class TestFormatDMThread:
    def test_formats_basic_thread(self, agent):
        messages = [
            {"sender": {"username": "alice"}, "body": "Hello!"},
            {"sender": {"username": "testbot"}, "body": "Hi alice!"},
            {"sender": {"username": "alice"}, "body": "How are you?"},
        ]
        result = agent._format_dm_thread(messages, "testbot")
        assert "alice: Hello!" in result
        assert "You: Hi alice!" in result
        assert "alice: How are you?" in result

    def test_truncates_long_threads(self, agent):
        messages = [
            {"sender": {"username": "alice"}, "body": f"Message {i}"}
            for i in range(20)
        ]
        result = agent._format_dm_thread(messages, "testbot", max_messages=5)
        lines = result.strip().split("\n")
        assert len(lines) == 5
        # Should include the most recent messages
        assert "Message 19" in result
        assert "Message 10" not in result

    def test_empty_thread(self, agent):
        result = agent._format_dm_thread([], "testbot")
        assert result == ""


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


class TestContextOverflowRecovery:
    @patch("colony_agent.agent.chat")
    def test_converse_trims_and_retries_on_overflow(self, mock_chat, agent):
        # Fill memory so there's something to trim
        for i in range(20):
            agent.memory.add("user", f"Old message {i}")

        # First call overflows, second (after trim) succeeds
        mock_chat.side_effect = [
            ContextOverflowError("context length exceeded"),
            "Summary of old interactions.",  # trim summary call
            "Response after trim.",  # retry of the original call
        ]
        result = agent._converse("New question")
        assert result == "Response after trim."
        assert mock_chat.call_count == 3

    @patch("colony_agent.agent.chat")
    def test_converse_reduces_memory_on_overflow(self, mock_chat, tmp_path):
        config = make_config(tmp_path, max_memory_messages=10)
        agent = make_agent(config)
        for i in range(20):
            agent.memory.add("user", f"Message {i}")

        mock_chat.side_effect = [
            ContextOverflowError("too many tokens"),
            "Summary.",
            "OK after trim.",
        ]
        agent._converse("Question")
        # Old 20 messages should have been compacted (summary + recent half)
        # plus the new question + response = well under 20
        old_messages = [m for m in agent.memory.messages if m["content"].startswith("Message ")]
        assert len(old_messages) <= 5  # kept at most half of max_messages

    @patch("colony_agent.agent.chat")
    def test_trim_handles_overflow_during_summary(self, mock_chat, agent):
        for i in range(20):
            agent.memory.add("user", f"Message {i}")

        # Both the original call and the summary call overflow
        mock_chat.side_effect = [
            ContextOverflowError("context length exceeded"),
            ContextOverflowError("still too long"),  # summary also overflows
            "OK after fallback trim.",  # retry succeeds after hard trim
        ]
        result = agent._converse("Question")
        assert result == "OK after fallback trim."
        # Memory should have been hard-trimmed (kept recent half, no summary)
        assert len(agent.memory) <= agent.memory.max_messages


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

    def test_vote_only_response_no_comment(self, agent):
        assert agent._extract_comment("VOTE: UPVOTE") == ""

    def test_skip_in_middle_of_text(self, agent):
        assert agent._extract_comment("I'll SKIP this one, nothing to add.") == ""

    def test_comment_skip_explicit(self, agent):
        assert agent._extract_comment("VOTE: UPVOTE\nCOMMENT: SKIP") == ""
