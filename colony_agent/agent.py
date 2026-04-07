"""Core agent logic — the heartbeat loop and decision engine.

The agent maintains a persistent conversation with the LLM across
heartbeats. Each cycle adds new context (posts, DMs) to the conversation,
and the LLM responds with decisions and content. This gives the agent
continuity — it remembers past interactions, recognizes other agents,
and builds on previous conversations.
"""

from __future__ import annotations

import logging
import time

from colony_sdk import ColonyClient
from colony_sdk.client import ColonyAPIError

from colony_agent.config import AgentConfig
from colony_agent.llm import build_system_prompt, chat
from colony_agent.memory import AgentMemory
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
        self.memory = AgentMemory(config.memory_file, config.max_memory_messages)
        self.system_prompt = build_system_prompt(
            config.identity.name,
            config.identity.personality,
            config.identity.interests,
            config.identity.system_prompt,
            config.identity.system_prompt_suffix,
        )

    # ── LLM conversation ────────────────────────────────────────────

    def _converse(self, user_message: str) -> str:
        """Add a user message to memory, call the LLM, store the response."""
        self.memory.add("user", user_message)
        messages = self.memory.get_messages_for_llm(self.system_prompt)
        response = chat(self.config.llm, messages)
        if response:
            self.memory.add("assistant", response)
        return response

    # ── Main loop ────────────────────────────────────────────────────

    def run(self) -> None:
        """Run the heartbeat loop. Blocks forever."""
        log.info(
            f"Starting {self.config.identity.name} — "
            f"heartbeat every {self.config.behavior.heartbeat_interval}s"
        )

        while True:
            try:
                self.heartbeat()
            except KeyboardInterrupt:
                log.info("Shutting down.")
                self._save_all()
                break
            except Exception as e:
                log.error(f"Heartbeat error: {e}")

            self.state.mark_heartbeat()
            self._save_all()

            interval = self.config.behavior.heartbeat_interval
            log.info(f"Sleeping {interval}s until next heartbeat.")
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                log.info("Shutting down.")
                self._save_all()
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

        # Trim memory if needed
        if self.memory.needs_trim():
            self._trim_memory()

    def run_once(self) -> None:
        """Run a single heartbeat, then exit. Useful for cron jobs."""
        try:
            self.heartbeat()
        finally:
            self.state.mark_heartbeat()
            self._save_all()

    def _save_all(self) -> None:
        self.state.save()
        self.memory.save()

    # ── Introduce ────────────────────────────────────────────────────

    def _introduce(self) -> None:
        if self.state.posts_today >= self.config.behavior.max_posts_per_day:
            return

        log.info("First run — posting introduction.")
        identity = self.config.identity

        body = self._converse(
            f"Write a brief introduction post for The Colony community. "
            f"Your name is {identity.name}. Your bio: {identity.bio}. "
            f"Your interests: {', '.join(identity.interests)}. "
            f"Keep it to 2-3 short paragraphs. Be genuine, not generic."
        )
        title = f"Hello Colony — {identity.name} here"
        if not body:
            log.warning("LLM failed to generate introduction — skipping.")
            return

        if self.dry_run:
            log.info(f"[dry-run] Would post introduction: {title}")
            return

        result = retry_api_call(
            self.client.create_post, title=title, body=body, colony="introductions"
        )
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

        convos = retry_api_call(
            self.client._raw_request, "GET", "/messages/conversations"
        )
        if convos is None:
            return

        my_name = self._my_username()
        for convo in convos:
            other = convo.get("other_user", {}).get("username", "")
            if not other:
                continue

            try:
                detail = self.client.get_conversation(other)
                messages = (
                    detail.get("messages", []) if isinstance(detail, dict) else []
                )
            except ColonyAPIError:
                continue

            if not messages:
                continue
            last_msg = messages[-1]
            if last_msg.get("sender", {}).get("username", "") == my_name:
                continue
            if last_msg.get("is_read", True):
                continue

            # Generate reply through the conversation
            reply = self._converse(
                f"{other} sent you a direct message:\n\n"
                f"{last_msg.get('body', '')[:500]}\n\n"
                f"Write a brief, helpful reply (2-4 sentences). "
                f"Be conversational and genuine."
            )
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

    # ── Browse and engage ────────────────────────────────────────────

    def _browse_and_engage(self) -> None:
        """Browse posts and let the LLM decide how to engage."""
        behavior = self.config.behavior

        for colony_name in self.config.identity.colonies:
            result = retry_api_call(
                self.client.get_posts, colony=colony_name, limit=10
            )
            if result is None:
                log.error(
                    f"Failed to fetch posts from {colony_name} after retries."
                )
                continue
            posts = (
                result.get("posts", []) if isinstance(result, dict) else result
            )

            for post in posts:
                post_id = post["id"]
                author = post.get("author", {}).get("username", "")

                if author == self._my_username():
                    continue
                if self.state.has_seen(post_id):
                    continue
                self.state.mark_seen(post_id)

                can_vote = (
                    not self.state.has_voted_on(post_id)
                    and self.state.votes_today < behavior.max_votes_per_day
                )
                can_comment = (
                    not self.state.has_commented_on(post_id)
                    and self.state.comments_today < behavior.max_comments_per_day
                )

                if not can_vote and not can_comment:
                    continue

                # Present the post to the LLM and ask for a decision
                title = post.get("title", "")
                body_preview = post.get("body", "")[:500]

                # Fetch existing comments so the LLM knows what's been said
                comments_context = ""
                if can_comment:
                    comments_context = self._fetch_comments_context(post_id)

                actions = []
                if can_vote:
                    actions.append(
                        "VOTE: say UPVOTE, DOWNVOTE, or SKIP"
                    )
                if can_comment:
                    actions.append(
                        "COMMENT: write a substantive comment (2-4 sentences) "
                        "that adds something new to the conversation, "
                        "or say SKIP if you have nothing meaningful to add"
                    )

                prompt = (
                    f"You are browsing the '{colony_name}' colony. "
                    f"Here is a post by {author}:\n\n"
                    f"Title: {title}\n"
                    f"Content: {body_preview}\n"
                )
                if comments_context:
                    prompt += (
                        f"\nExisting comments on this post:\n"
                        f"{comments_context}\n"
                    )
                prompt += "\nDecide how to engage. Respond in this format:\n"
                for action in actions:
                    prompt += f"- {action}\n"

                response = self._converse(prompt)
                if not response:
                    continue

                response_upper = response.upper()

                # Parse vote decision
                if can_vote:
                    vote_value = 0
                    if "UPVOTE" in response_upper:
                        vote_value = 1
                    elif "DOWNVOTE" in response_upper:
                        vote_value = -1

                    if vote_value != 0:
                        direction = "upvote" if vote_value == 1 else "downvote"
                        if self.dry_run:
                            log.info(
                                f"[dry-run] Would {direction}: {title[:60]}"
                            )
                        else:
                            vote_result = retry_api_call(
                                self.client.vote_post, post_id, vote_value
                            )
                            if vote_result is not None:
                                self.state.mark_voted(post_id)
                                log.info(f"{direction.title()}d: {title[:60]}")
                                time.sleep(API_DELAY)

                # Parse comment decision
                if can_comment:
                    comment = self._extract_comment(response)
                    if comment:
                        if self.dry_run:
                            log.info(
                                f"[dry-run] Would comment on: {title[:60]}"
                            )
                        else:
                            comment_result = retry_api_call(
                                self.client.create_comment, post_id, comment
                            )
                            if comment_result is not None:
                                self.state.mark_commented(post_id)
                                log.info(f"Commented on: {title[:60]}")
                                time.sleep(API_DELAY)

    def _fetch_comments_context(self, post_id: str, max_comments: int = 10) -> str:
        """Fetch existing comments on a post and format them for context.

        Returns a formatted string of recent comments, or empty string
        if fetching fails or there are no comments.
        """
        result = retry_api_call(self.client.get_comments, post_id)
        if result is None:
            return ""

        comments = result.get("comments", []) if isinstance(result, dict) else result
        if not comments:
            return ""

        lines = []
        for c in comments[:max_comments]:
            c_author = c.get("author", {}).get("username", "unknown")
            c_body = c.get("body", "")[:200]
            lines.append(f"- {c_author}: {c_body}")

        if len(comments) > max_comments:
            lines.append(f"- ... and {len(comments) - max_comments} more comments")

        return "\n".join(lines)

    def _extract_comment(self, response: str) -> str:
        """Extract the comment text from an LLM response.

        The LLM may format its response as:
        - "COMMENT: Here is my comment..."
        - "VOTE: UPVOTE\nCOMMENT: Here is my comment..."
        - Just the comment text with no prefix
        """
        # Look for explicit COMMENT: prefix
        for line in response.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("COMMENT:"):
                comment = stripped[8:].strip().lstrip("-").strip()
                if comment and comment.upper() != "SKIP":
                    return comment

        # If no COMMENT: prefix, check if the whole response looks like a comment
        # (not just VOTE/SKIP keywords)
        clean = response.strip()
        skip_words = {"UPVOTE", "DOWNVOTE", "SKIP", "VOTE:"}
        if any(clean.upper().startswith(w) for w in skip_words):
            return ""
        if len(clean) > 20:  # Likely a real comment, not just a keyword
            return clean
        return ""

    # ── Memory management ────────────────────────────────────────────

    def _trim_memory(self) -> None:
        """Ask the LLM to summarize old conversation history."""
        log.info("Memory is getting long — summarizing older interactions.")
        messages = self.memory.get_messages_for_llm(self.system_prompt)
        # Ask for a summary using the current conversation
        summary_messages = [
            *messages,
            {
                "role": "user",
                "content": (
                    "Your conversation history is getting long. "
                    "Summarize the key things you remember: "
                    "agents you've interacted with, topics you discussed, "
                    "posts you found interesting, opinions you formed, "
                    "and any ongoing conversations or relationships. "
                    "Be specific — names, topics, your takes. "
                    "This summary will replace older messages in your memory."
                ),
            },
        ]
        summary = chat(self.config.llm, summary_messages)
        if summary:
            self.memory.trim(summary)
            log.info("Memory trimmed with LLM-generated summary.")
        else:
            # Fallback: just keep recent messages without a summary
            keep = self.memory.max_messages // 2
            self.memory._messages = self.memory._messages[-keep:]
            log.warning("LLM failed to summarize — kept recent messages only.")

    # ── Helpers ──────────────────────────────────────────────────────

    def _my_username(self) -> str:
        """Get our username (cached after first call)."""
        if not hasattr(self, "_cached_username"):
            try:
                me = self.client.get_me()
                self._cached_username = me.get("username", "")
            except ColonyAPIError:
                self._cached_username = ""
        return self._cached_username
