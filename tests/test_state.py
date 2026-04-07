"""Tests for colony_agent.state."""

import json
import time

import pytest

from colony_agent.state import AgentState


@pytest.fixture
def state(tmp_path):
    return AgentState(str(tmp_path / "state.json"))


class TestAgentState:
    def test_fresh_state(self, state):
        assert state.introduced is False
        assert state.posts_today == 0
        assert state.comments_today == 0
        assert state.votes_today == 0
        assert state.last_heartbeat == 0

    def test_mark_seen(self, state):
        assert not state.has_seen("post-1")
        state.mark_seen("post-1")
        assert state.has_seen("post-1")
        assert not state.has_seen("post-2")

    def test_mark_commented(self, state):
        assert not state.has_commented_on("post-1")
        state.mark_commented("post-1")
        assert state.has_commented_on("post-1")
        assert state.comments_today == 1

    def test_mark_voted(self, state):
        assert not state.has_voted_on("post-1")
        state.mark_voted("post-1")
        assert state.has_voted_on("post-1")
        assert state.votes_today == 1

    def test_mark_posted(self, state):
        assert state.posts_today == 0
        state.mark_posted()
        assert state.posts_today == 1
        state.mark_posted()
        assert state.posts_today == 2

    def test_mark_introduced(self, state):
        assert state.introduced is False
        state.mark_introduced()
        assert state.introduced is True

    def test_mark_heartbeat(self, state):
        before = time.time()
        state.mark_heartbeat()
        assert state.last_heartbeat >= before

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "state.json")
        s1 = AgentState(path)
        s1.mark_seen("post-a")
        s1.mark_commented("post-b")
        s1.mark_voted("post-c")
        s1.mark_posted()
        s1.mark_introduced()
        s1.save()

        s2 = AgentState(path)
        assert s2.has_seen("post-a")
        assert s2.has_commented_on("post-b")
        assert s2.has_voted_on("post-c")
        assert s2.posts_today == 1
        assert s2.introduced is True

    def test_daily_counter_reset(self, tmp_path):
        path = str(tmp_path / "state.json")
        s = AgentState(path)
        s.mark_posted()
        s.mark_commented("x")
        s.mark_voted("y")
        assert s.posts_today == 1
        assert s.comments_today == 1
        assert s.votes_today == 1

        # Simulate next day
        s._data["last_reset_date"] = "2020-01-01"
        assert s.posts_today == 0
        assert s.comments_today == 0
        assert s.votes_today == 0

    def test_prune(self, state):
        old_time = time.time() - (31 * 86400)  # 31 days ago
        state._data["seen_posts"]["old-post"] = old_time
        state._data["seen_posts"]["new-post"] = time.time()
        state._data["commented_on"]["old-comment"] = old_time
        state._data["voted_on"]["old-vote"] = old_time

        removed = state.prune(max_age_days=30)
        assert removed == 3
        assert not state.has_seen("old-post")
        assert state.has_seen("new-post")
        assert not state.has_commented_on("old-comment")
        assert not state.has_voted_on("old-vote")

    def test_prune_nothing_to_remove(self, state):
        state.mark_seen("recent")
        removed = state.prune(max_age_days=30)
        assert removed == 0
        assert state.has_seen("recent")

    def test_atomic_save(self, tmp_path):
        path = tmp_path / "state.json"
        s = AgentState(str(path))
        s.mark_seen("test")
        s.save()

        assert path.exists()
        assert not path.with_suffix(".tmp").exists()
        data = json.loads(path.read_text())
        assert "test" in data["seen_posts"]

    def test_total_tracked(self, state):
        assert state.total_tracked == 0
        state.mark_seen("p1")
        state.mark_commented("p2")
        state.mark_voted("p3")
        assert state.total_tracked == 3

    def test_prune_on_load_after_long_offline(self, tmp_path):
        path = str(tmp_path / "state.json")
        s = AgentState(path)
        # Add old entries and set last heartbeat to 14 days ago
        old_time = time.time() - (20 * 86400)
        s._data["seen_posts"]["old1"] = old_time
        s._data["seen_posts"]["old2"] = old_time
        s._data["commented_on"]["old3"] = old_time
        s._data["seen_posts"]["recent"] = time.time()
        s._data["last_heartbeat"] = time.time() - (14 * 86400)
        s.save()

        # Reload — should auto-prune stale entries
        s2 = AgentState(path)
        assert not s2.has_seen("old1")
        assert not s2.has_seen("old2")
        assert not s2.has_commented_on("old3")
        assert s2.has_seen("recent")

    def test_no_prune_when_recently_active(self, tmp_path):
        path = str(tmp_path / "state.json")
        s = AgentState(path)
        old_time = time.time() - (20 * 86400)
        s._data["seen_posts"]["old"] = old_time
        s._data["last_heartbeat"] = time.time() - (2 * 86400)  # 2 days ago
        s.save()

        s2 = AgentState(path)
        # Should NOT auto-prune — agent was recently active
        assert s2.has_seen("old")

    def test_no_prune_on_fresh_state(self, tmp_path):
        path = str(tmp_path / "state.json")
        s = AgentState(path)
        assert s.total_tracked == 0  # no crash on empty state
