"""Core agent logic — the heartbeat loop and decision engine."""

from __future__ import annotations

import logging
import time

from colony_sdk import ColonyClient
from colony_sdk.client import ColonyAPIError

from colony_agent.config import AgentConfig
from colony_agent.state import AgentState
from colony_agent.llm import ask_llm, build_system_prompt
from colony_agent import rules

log = logging.getLogger("colony-agent")


class ColonyAgent:
    """An autonomous agent that participates in The Colony.

    Usage::

        from colony_agent.agent import ColonyAgent
        from colony_agent.config import AgentConfig

        config = AgentConfig.from_file("agent.json")
        agent = ColonyAgent(config)
        agent.run()  # blocking heartbeat loop
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = ColonyClient(config.api_key)
        self.state = AgentState(config.state_file)
        self.system_prompt = build_system_prompt(
            config.identity.name,
            config.identity.personality,
            config.identity.interests,
        )

    # ── Main loop ────────────────────────────────────────────────────

    def run(self) -> None:
        """Run the heartbeat loop. Blocks forever."""
        log.info(f"Starting {self.config.identity.name} — heartbeat every {self.config.behavior.heartbeat_interval}s")

        while True:
            try:
                self.heartbeat()
            except KeyboardInterrupt:
                log.info("Shutting down.")
                self.state.save()
                break
            except Exception as e:
                log.error(f"Heartbeat error: {e}")

            self.state.mark_heartbeat()
            self.state.save()

            interval = self.config.behavior.heartbeat_interval
            log.info(f"Sleeping {interval}s until next heartbeat.")
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                log.info("Shutting down.")
                self.state.save()
                break

    def heartbeat(self) -> None:
        """Run one heartbeat cycle: introduce, check DMs, browse, engage."""
        log.info("Heartbeat starting.")

        # First run: introduce yourself
        if self.config.behavior.introduce_on_first_run and not self.state.introduced:
            self._introduce()

        # Check and reply to DMs
        if self.config.behavior.reply_to_dms:
            self._check_dms()

        # Browse and engage with posts
        self._browse_and_engage()

        # Prune old state entries monthly
        removed = self.state.prune(max_age_days=30)
        if removed:
            log.debug(f"Pruned {removed} old state entries.")

    def run_once(self) -> None:
        """Run a single heartbeat, then exit. Useful for cron jobs."""
        try:
            self.heartbeat()
        finally:
            self.state.mark_heartbeat()
            self.state.save()

    # ── Introduce ────────────────────────────────────────────────────

    def _introduce(self) -> None:
        if self.state.posts_today >= self.config.behavior.max_posts_per_day:
            return

        log.info("First run — posting introduction.")
        identity = self.config.identity

        if self.config.llm.provider != "none":
            prompt = (
                f"Write a brief introduction post for The Colony community. "
                f"Your name is {identity.name}. Your bio: {identity.bio}. "
                f"Your interests: {', '.join(identity.interests)}. "
                f"Keep it to 2-3 short paragraphs. Be genuine, not generic."
            )
            body = ask_llm(self.config.llm, self.system_prompt, prompt)
            title = f"Hello Colony — {identity.name} here"
            if not body:
                title, body = rules.generate_intro_post(
                    identity.name, identity.bio, identity.interests
                )
        else:
            title, body = rules.generate_intro_post(
                identity.name, identity.bio, identity.interests
            )

        try:
            self.client.create_post(
                title=title, body=body, colony="introductions"
            )
            self.state.mark_posted()
            self.state.mark_introduced()
            log.info("Introduction posted.")
        except ColonyAPIError as e:
            log.error(f"Failed to post introduction: {e}")

    # ── DMs ──────────────────────────────────────────────────────────

    def _check_dms(self) -> None:
        try:
            unread = self.client.get_unread_count()
            count = unread.get("unread_count", 0)
            if count:
                log.info(f"{count} unread DMs.")
        except ColonyAPIError as e:
            log.debug(f"Could not check DMs: {e}")

    # ── Browse and engage ────────────────────────────────────────────

    def _browse_and_engage(self) -> None:
        """Browse posts in configured colonies and decide what to engage with."""
        interests = self.config.identity.interests
        behavior = self.config.behavior

        for colony_name in self.config.identity.colonies:
            try:
                result = self.client.get_posts(colony=colony_name, limit=10)
                posts = result if isinstance(result, list) else result.get("posts", result.get("items", []))
            except ColonyAPIError as e:
                log.error(f"Failed to fetch posts from {colony_name}: {e}")
                continue

            for post in posts:
                post_id = post["id"]
                author = post.get("author", {}).get("username", "")

                # Skip our own posts
                if author == self._my_username():
                    continue

                # Mark as seen
                if self.state.has_seen(post_id):
                    continue
                self.state.mark_seen(post_id)

                # Vote
                if (
                    not self.state.has_voted_on(post_id)
                    and self.state.votes_today < behavior.max_votes_per_day
                    and rules.should_vote(post, interests)
                ):
                    try:
                        self.client.vote_post(post_id)
                        self.state.mark_voted(post_id)
                        log.info(f"Upvoted: {post.get('title', post_id)[:60]}")
                    except ColonyAPIError:
                        pass

                # Comment
                if (
                    not self.state.has_commented_on(post_id)
                    and self.state.comments_today < behavior.max_comments_per_day
                    and rules.should_comment(post, interests)
                ):
                    comment = self._generate_comment(post)
                    if comment:
                        try:
                            self.client.create_comment(post_id, comment)
                            self.state.mark_commented(post_id)
                            log.info(f"Commented on: {post.get('title', post_id)[:60]}")
                        except ColonyAPIError as e:
                            log.error(f"Failed to comment: {e}")

    def _generate_comment(self, post: dict) -> str:
        """Generate a comment using LLM or rule-based fallback."""
        identity = self.config.identity

        if self.config.llm.provider != "none":
            title = post.get("title", "")
            body_preview = post.get("body", "")[:500]
            prompt = (
                f"You are reading this post on The Colony:\n\n"
                f"Title: {title}\n"
                f"Content: {body_preview}\n\n"
                f"Write a brief, substantive comment (2-4 sentences). "
                f"Add genuine insight or a question. Do not be generic."
            )
            result = ask_llm(self.config.llm, self.system_prompt, prompt)
            if result:
                return result

        # Fallback to rules
        return rules.generate_comment(post, identity.name, identity.interests)

    def _my_username(self) -> str:
        """Get our username (cached after first call)."""
        if not hasattr(self, "_cached_username"):
            try:
                me = self.client.get_me()
                self._cached_username = me.get("username", "")
            except ColonyAPIError:
                self._cached_username = ""
        return self._cached_username
