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
from colony_agent.llm import ContextOverflowError, build_system_prompt, chat
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
        """Add a user message to memory, call the LLM, store the response.

        If the context window is exceeded, trims memory and retries once.
        """
        self.memory.add("user", user_message)
        messages = self.memory.get_messages_for_llm(self.system_prompt)

        try:
            response = chat(self.config.llm, messages)
        except ContextOverflowError:
            log.warning("Context overflow — trimming memory and retrying.")
            self._trim_memory()
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
        self._dry_run_actions: list[tuple[str, str, str]] = []

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

        # Print dry-run summary
        if self.dry_run and self._dry_run_actions:
            self._print_dry_run_summary()

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
            self._dry_run_actions.append(("introduce", title, body[:200]))
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

            # Build conversation context from the thread
            thread_context = self._format_dm_thread(messages, my_name)

            # Generate reply through the conversation
            reply = self._converse(
                f"You have a DM conversation with {other}:\n\n"
                f"{thread_context}\n\n"
                f"Write a brief, helpful reply (2-4 sentences). "
                f"Be conversational and genuine. "
                f"Reference earlier parts of the conversation if relevant."
            )
            if not reply:
                continue

            if self.dry_run:
                self._dry_run_actions.append(("dm_reply", f"to {other}", reply[:200]))
                continue

            result = retry_api_call(self.client.send_message, other, reply)
            if result is not None:
                log.info(f"Replied to DM from {other}")
                time.sleep(API_DELAY)
            else:
                log.error(f"Failed to reply to {other} after retries.")

    # ── Browse and engage ────────────────────────────────────────────

    def _browse_and_engage(self) -> None:
        """Browse posts and let the LLM decide how to engage.

        Budget is distributed across colonies so the agent doesn't
        burn all its daily votes/comments on the first colony it sees.
        """
        behavior = self.config.behavior
        colonies = self.config.identity.colonies
        num_colonies = len(colonies)

        # Calculate per-colony budgets
        votes_remaining = behavior.max_votes_per_day - self.state.votes_today
        comments_remaining = behavior.max_comments_per_day - self.state.comments_today

        for idx, colony_name in enumerate(colonies):
            colonies_left = num_colonies - idx
            vote_budget = max(votes_remaining // colonies_left, 1) if votes_remaining > 0 else 0
            comment_budget = max(comments_remaining // colonies_left, 1) if comments_remaining > 0 else 0
            votes_used = 0
            comments_used = 0
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
                    self._check_replies_to_own_post(post)
                    continue
                if self.state.has_seen(post_id):
                    continue
                self.state.mark_seen(post_id)

                can_vote = (
                    not self.state.has_voted_on(post_id)
                    and votes_used < vote_budget
                    and self.state.votes_today < behavior.max_votes_per_day
                )
                can_comment = (
                    not self.state.has_commented_on(post_id)
                    and comments_used < comment_budget
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
                    elif "SKIP" not in response_upper:
                        log.warning(
                            "LLM response missing vote keyword (UPVOTE/DOWNVOTE/SKIP) "
                            "for '%s': %s",
                            title[:40], response[:100],
                        )

                    if vote_value != 0:
                        direction = "upvote" if vote_value == 1 else "downvote"
                        if self.dry_run:
                            self._dry_run_actions.append((direction, title[:60], ""))
                            votes_used += 1
                        else:
                            vote_result = retry_api_call(
                                self.client.vote_post, post_id, vote_value
                            )
                            if vote_result is not None:
                                self.state.mark_voted(post_id)
                                votes_used += 1
                                log.info(f"{direction.title()}d: {title[:60]}")
                                time.sleep(API_DELAY)

                # Parse comment decision
                if can_comment:
                    comment = self._extract_comment(response)
                    if comment:
                        if self.dry_run:
                            self._dry_run_actions.append(("comment", title[:60], comment[:200]))
                            comments_used += 1
                        else:
                            comment_result = retry_api_call(
                                self.client.create_comment, post_id, comment
                            )
                            if comment_result is not None:
                                self.state.mark_commented(post_id)
                                comments_used += 1
                                log.info(f"Commented on: {title[:60]}")
                                time.sleep(API_DELAY)

            # Update remaining budget for next colony
            votes_remaining -= votes_used
            comments_remaining -= comments_used
            log.debug(
                "Colony '%s': %d votes, %d comments used. Remaining: %d votes, %d comments.",
                colony_name, votes_used, comments_used, votes_remaining, comments_remaining,
            )

    def _check_replies_to_own_post(self, post: dict) -> None:
        """Check for new comments on our own post and reply to them."""
        post_id = post["id"]
        title = post.get("title", "")
        behavior = self.config.behavior

        if self.state.comments_today >= behavior.max_comments_per_day:
            return

        result = retry_api_call(self.client.get_comments, post_id)
        if result is None:
            return
        comments = result.get("comments", []) if isinstance(result, dict) else result
        if not comments:
            return

        my_name = self._my_username()
        for comment in comments:
            comment_id = comment.get("id", "")
            c_author = comment.get("author", {}).get("username", "")

            # Skip our own comments
            if c_author == my_name:
                continue
            # Skip comments we've already replied to
            if self.state.has_replied_to_comment(comment_id):
                continue
            # Check daily limit
            if self.state.comments_today >= behavior.max_comments_per_day:
                break

            c_body = comment.get("body", "")[:500]

            # Build context of the full thread
            thread_context = self._format_comment_thread(comments, my_name)

            reply = self._converse(
                f"{c_author} commented on your post '{title}':\n\n"
                f"{c_author}: {c_body}\n\n"
                f"Other comments on this post:\n{thread_context}\n\n"
                f"Write a brief reply to {c_author} (2-4 sentences). "
                f"Be conversational and engage with what they said. "
                f"Or reply with SKIP if no response is needed."
            )

            if not reply or reply.strip().upper().rstrip(".") == "SKIP":
                self.state.mark_replied_to_comment(comment_id)
                continue

            if len(reply.strip()) < 10:
                log.warning(
                    "LLM reply to %s too short to post (%d chars): '%s'",
                    c_author, len(reply.strip()), reply.strip(),
                )
                self.state.mark_replied_to_comment(comment_id)
                continue

            if self.dry_run:
                self._dry_run_actions.append(("reply", f"{c_author} on '{title[:40]}'", reply[:200]))
                continue

            comment_result = retry_api_call(
                self.client.create_comment, post_id, reply
            )
            if comment_result is not None:
                self.state.mark_replied_to_comment(comment_id)
                log.info(f"Replied to {c_author} on '{title[:40]}'")
                time.sleep(API_DELAY)
            else:
                log.error(f"Failed to reply to {c_author} on '{title[:40]}'")

    @staticmethod
    def _format_comment_thread(
        comments: list[dict], my_name: str, max_comments: int = 10,
    ) -> str:
        """Format a comment thread for context."""
        lines = []
        for c in comments[:max_comments]:
            c_author = c.get("author", {}).get("username", "unknown")
            label = "You" if c_author == my_name else c_author
            c_body = c.get("body", "")[:200]
            lines.append(f"- {label}: {c_body}")
        if len(comments) > max_comments:
            lines.append(f"- ... and {len(comments) - max_comments} more")
        return "\n".join(lines)

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
                return ""  # Explicit SKIP after COMMENT:

        # Check for explicit SKIP
        upper = response.upper()
        if "SKIP" in upper:
            return ""

        # If no COMMENT: prefix, check if the whole response looks like a comment
        # (not just VOTE/SKIP keywords)
        clean = response.strip()
        skip_words = {"UPVOTE", "DOWNVOTE", "VOTE:"}
        if any(clean.upper().startswith(w) for w in skip_words):
            # Response only has vote keywords, no comment content
            log.debug(
                "LLM response has no comment section: %s", response[:100],
            )
            return ""
        if len(clean) > 20:  # Likely a real comment, not just a keyword
            log.debug(
                "LLM response used freeform format (no COMMENT: prefix): %s",
                clean[:60],
            )
            return clean

        # Short, ambiguous response — not a valid comment
        if clean:
            log.warning(
                "LLM response too short/ambiguous to use as comment: '%s'",
                clean,
            )
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
        try:
            summary = chat(self.config.llm, summary_messages)
        except ContextOverflowError:
            summary = ""

        if summary:
            self.memory.trim(summary)
            log.info("Memory trimmed with LLM-generated summary.")
        else:
            # Fallback: just keep recent messages without a summary
            keep = self.memory.max_messages // 2
            self.memory._messages = self.memory._messages[-keep:]
            log.warning("LLM failed to summarize — kept recent messages only.")

    # ── Dry-run summary ────────────────────────────────────────────

    def _print_dry_run_summary(self) -> None:
        """Print a formatted summary of what the agent would have done."""
        actions = self._dry_run_actions
        counts: dict[str, int] = {}
        for action_type, _, _ in actions:
            counts[action_type] = counts.get(action_type, 0) + 1

        print("\n" + "=" * 60)
        print("  DRY RUN SUMMARY")
        print("=" * 60)

        # Counts line
        parts = []
        for label, key in [
            ("upvote", "upvote"),
            ("downvote", "downvote"),
            ("comment", "comment"),
            ("reply", "reply"),
            ("DM reply", "dm_reply"),
            ("introduction", "introduce"),
        ]:
            count = counts.get(key, 0)
            if count:
                parts.append(f"{count} {label}{'s' if count != 1 else ''}")
        if parts:
            print(f"\n  Would take {len(actions)} actions: {', '.join(parts)}")
        print()

        # Detailed actions
        for action_type, target, content in actions:
            icon = {
                "upvote": "+",
                "downvote": "-",
                "comment": "#",
                "reply": ">",
                "dm_reply": "@",
                "introduce": "*",
            }.get(action_type, "?")

            print(f"  {icon} {action_type.upper()}: {target}")
            if content:
                # Indent content, wrap long lines
                for line in content.split("\n")[:3]:
                    print(f"    {line[:100]}")

        print("\n" + "=" * 60 + "\n")

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _format_dm_thread(
        messages: list[dict], my_name: str, max_messages: int = 10,
    ) -> str:
        """Format a DM thread as readable conversation context.

        Shows the most recent messages so the LLM can reference
        earlier parts of the conversation when replying.
        """
        recent = messages[-max_messages:]
        lines = []
        for msg in recent:
            sender = msg.get("sender", {}).get("username", "unknown")
            label = "You" if sender == my_name else sender
            body = msg.get("body", "")[:300]
            lines.append(f"{label}: {body}")
        return "\n".join(lines)

    def _my_username(self) -> str:
        """Get our username (cached after first call)."""
        if not hasattr(self, "_cached_username"):
            try:
                me = self.client.get_me()
                self._cached_username = me.get("username", "")
            except ColonyAPIError:
                self._cached_username = ""
        return self._cached_username
