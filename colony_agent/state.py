"""Persistent state tracking via JSON file."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


class AgentState:
    """Tracks what the agent has seen and done across sessions."""

    def __init__(self, path: str = "agent_state.json"):
        self.path = Path(path)
        self._data: dict = {
            "seen_posts": {},       # post_id -> timestamp
            "commented_on": {},     # post_id -> timestamp
            "voted_on": {},         # post_id -> timestamp
            "posts_today": 0,
            "comments_today": 0,
            "votes_today": 0,
            "last_reset_date": "",
            "introduced": False,
            "last_heartbeat": 0,
        }
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                saved = json.load(f)
                self._data.update(saved)
        self._reset_daily_counters_if_needed()

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self.path)

    def _reset_daily_counters_if_needed(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if self._data["last_reset_date"] != today:
            self._data["posts_today"] = 0
            self._data["comments_today"] = 0
            self._data["votes_today"] = 0
            self._data["last_reset_date"] = today

    # ── Queries ──────────────────────────────────────────────────────

    def has_seen(self, post_id: str) -> bool:
        return post_id in self._data["seen_posts"]

    def has_commented_on(self, post_id: str) -> bool:
        return post_id in self._data["commented_on"]

    def has_voted_on(self, post_id: str) -> bool:
        return post_id in self._data["voted_on"]

    @property
    def introduced(self) -> bool:
        return self._data["introduced"]

    @property
    def posts_today(self) -> int:
        self._reset_daily_counters_if_needed()
        return self._data["posts_today"]

    @property
    def comments_today(self) -> int:
        self._reset_daily_counters_if_needed()
        return self._data["comments_today"]

    @property
    def votes_today(self) -> int:
        self._reset_daily_counters_if_needed()
        return self._data["votes_today"]

    @property
    def last_heartbeat(self) -> float:
        return self._data["last_heartbeat"]

    # ── Mutations ────────────────────────────────────────────────────

    def mark_seen(self, post_id: str) -> None:
        self._data["seen_posts"][post_id] = time.time()

    def mark_commented(self, post_id: str) -> None:
        self._data["commented_on"][post_id] = time.time()
        self._data["comments_today"] += 1

    def mark_voted(self, post_id: str) -> None:
        self._data["voted_on"][post_id] = time.time()
        self._data["votes_today"] += 1

    def mark_posted(self) -> None:
        self._data["posts_today"] += 1

    def mark_introduced(self) -> None:
        self._data["introduced"] = True

    def mark_heartbeat(self) -> None:
        self._data["last_heartbeat"] = time.time()

    # ── Maintenance ──────────────────────────────────────────────────

    def prune(self, max_age_days: int = 30) -> int:
        """Remove entries older than max_age_days. Returns count removed."""
        cutoff = time.time() - (max_age_days * 86400)
        removed = 0
        for key in ("seen_posts", "commented_on", "voted_on"):
            before = len(self._data[key])
            self._data[key] = {
                k: v for k, v in self._data[key].items() if v > cutoff
            }
            removed += before - len(self._data[key])
        return removed
