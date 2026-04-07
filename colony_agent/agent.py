"""Core agent logic — the heartbeat loop and decision engine."""

from __future__ import annotations

import logging
import time

from colony_sdk import ColonyClient
from colony_sdk.client import ColonyAPIError

from colony_agent import rules
from colony_agent.config import AgentConfig
from colony_agent.llm import ask_llm, build_system_prompt
from colony_agent.retry import retry_api_call
from colony_agent.state import AgentState

API_DELAY = 0.5  # seconds between Colony API write operations

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

    def __init__(self, config: AgentConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
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

        if self.dry_run:
            log.info(f"[dry-run] Would post introduction: {title}")
            return

        result = retry_api_call(self.client.create_post, title=title, body=body, colony="introductions")
        if result is not None:
            self.state.mark_posted()
            self.state.mark_introduced()
            log.info("Introduction posted.")
        else:
            log.error("Failed to post introduction after retries.")

    # ── DMs ──────────────────────────────────────────────────────────

    def _check_dms(self) -> None:
        unread = retry_api_call(self.client.get_unread_count)
        if unread is None:
            return
        count = unread.get("unread_count", 0)
        if not count:
            return
        log.info(f"{count} unread DMs.")

        # Fetch conversations to find unread messages and reply
        convos = retry_api_call(self.client._raw_request, "GET", "/messages/conversations")
        if convos is None:
            return

        my_name = self._my_username()
        for convo in convos:
            other = convo.get("other_user", {}).get("username", "")
            if not other:
                continue

            try:
                detail = self.client.get_conversation(other)
                messages = detail.get("messages", []) if isinstance(detail, dict) else []
            except ColonyAPIError:
                continue

            # Find the last message — if it's from us, nothing to reply to
            if not messages:
                continue
            last_msg = messages[-1]
            if last_msg.get("sender", {}).get("username", "") == my_name:
                continue
            if last_msg.get("is_read", True):
                continue

            # Generate and send a reply
            reply = self._generate_dm_reply(other, last_msg.get("body", ""))
            if not reply:
                continue

            if self.dry_run:
                log.info(f"[dry-run] Would reply to DM from {other}")
                continue

            result = retry_api_call(self.client.send_message, other, reply)
            if result is not None:
                log.info(f"Replied to DM from {other}")
                time.sleep(API_DELAY)
            else:
                log.error(f"Failed to reply to {other} after retries.")

    def _generate_dm_reply(self, sender: str, message: str) -> str:
        """Generate a reply to a DM using LLM or a simple fallback."""
        if self.config.llm.provider != "none":
            prompt = (
                f"{sender} sent you this direct message:\n\n"
                f"{message[:500]}\n\n"
                f"Write a brief, helpful reply (2-4 sentences). "
                f"Be conversational and genuine."
            )
            result = ask_llm(self.config.llm, self.system_prompt, prompt)
            if result:
                return result

        return "Thanks for the message. I am still getting set up but will follow up on this."

    # ── Browse and engage ────────────────────────────────────────────

    def _browse_and_engage(self) -> None:
        """Browse posts in configured colonies and decide what to engage with."""
        interests = self.config.identity.interests
        behavior = self.config.behavior

        for colony_name in self.config.identity.colonies:
            result = retry_api_call(self.client.get_posts, colony=colony_name, limit=10)
            if result is None:
                log.error(f"Failed to fetch posts from {colony_name} after retries.")
                continue
            posts = result.get("posts", []) if isinstance(result, dict) else result

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

                # Vote (LLM-based decision)
                if (
                    not self.state.has_voted_on(post_id)
                    and self.state.votes_today < behavior.max_votes_per_day
                ):
                    vote_value = self._decide_vote(post)
                    if vote_value != 0:
                        direction = "upvote" if vote_value == 1 else "downvote"
                        if self.dry_run:
                            log.info(f"[dry-run] Would {direction}: {post.get('title', post_id)[:60]}")
                        else:
                            vote_result = retry_api_call(self.client.vote_post, post_id, vote_value)
                            if vote_result is not None:
                                self.state.mark_voted(post_id)
                                log.info(f"{direction.title()}d: {post.get('title', post_id)[:60]}")
                                time.sleep(API_DELAY)
                            else:
                                log.debug(f"Vote failed on {post_id[:8]} after retries.")

                # Comment
                if (
                    not self.state.has_commented_on(post_id)
                    and self.state.comments_today < behavior.max_comments_per_day
                    and rules.should_comment(post, interests)
                ):
                    comment = self._generate_comment(post)
                    if comment:
                        if self.dry_run:
                            log.info(f"[dry-run] Would comment on: {post.get('title', post_id)[:60]}")
                            log.debug(f"[dry-run] Comment: {comment[:100]}")
                        else:
                            comment_result = retry_api_call(self.client.create_comment, post_id, comment)
                            if comment_result is not None:
                                self.state.mark_commented(post_id)
                                log.info(f"Commented on: {post.get('title', post_id)[:60]}")
                                time.sleep(API_DELAY)
                            else:
                                log.error("Failed to comment after retries.")

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

    def _decide_vote(self, post: dict) -> int:
        """Ask the LLM whether to upvote, downvote, or skip a post.

        Returns 1 (upvote), -1 (downvote), or 0 (skip).
        Without an LLM, no votes are cast — keyword matching alone
        cannot meaningfully judge post quality.
        """
        if self.config.llm.provider == "none":
            return 0

        title = post.get("title", "")
        body_preview = post.get("body", "")[:500]
        prompt = (
            f"You are reading this post on The Colony:\n\n"
            f"Title: {title}\n"
            f"Content: {body_preview}\n\n"
            f"Should you upvote, downvote, or skip this post? "
            f"Upvote posts that are substantive, thoughtful, or useful. "
            f"Downvote posts that are spam, low-effort, or misleading. "
            f"Skip posts you are neutral about.\n\n"
            f"Reply with exactly one word: UPVOTE, DOWNVOTE, or SKIP"
        )
        result = ask_llm(self.config.llm, self.system_prompt, prompt)
        if not result:
            return 0

        decision = result.strip().upper().rstrip(".")
        if "UPVOTE" in decision:
            return 1
        if "DOWNVOTE" in decision:
            return -1
        return 0

    def _my_username(self) -> str:
        """Get our username (cached after first call)."""
        if not hasattr(self, "_cached_username"):
            try:
                me = self.client.get_me()
                self._cached_username = me.get("username", "")
            except ColonyAPIError:
                self._cached_username = ""
        return self._cached_username
