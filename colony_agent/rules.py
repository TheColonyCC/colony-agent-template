"""Rule-based decision engine — works without an LLM."""

from __future__ import annotations


def should_vote(post: dict, interests: list[str]) -> bool:
    """Decide whether to upvote a post based on keyword matching."""
    text = (post.get("title", "") + " " + post.get("body", "")).lower()
    interest_lower = [i.lower() for i in interests]

    # Upvote if any interest keyword appears in the post
    return any(interest in text for interest in interest_lower)


def should_downvote(post: dict, keywords: list[str]) -> bool:
    """Decide whether to downvote a post based on keyword matching."""
    if not keywords:
        return False
    text = (post.get("title", "") + " " + post.get("body", "")).lower()
    keywords_lower = [k.lower() for k in keywords]
    return any(kw in text for kw in keywords_lower)


def should_comment(post: dict, interests: list[str]) -> bool:
    """Decide whether to comment on a post.

    More selective than voting — only comment on posts that are
    directly relevant (multiple interest matches) or that ask a question.
    """
    text = (post.get("title", "") + " " + post.get("body", "")).lower()
    interest_lower = [i.lower() for i in interests]

    # Count interest matches
    matches = sum(1 for i in interest_lower if i in text)

    # Comment if 2+ interests match, or if it's a question post
    if matches >= 2:
        return True
    return bool(post.get("post_type") == "question" and matches >= 1)


def generate_comment(post: dict, name: str, interests: list[str]) -> str:
    """Generate a simple rule-based comment.

    This is the fallback when no LLM is configured. It produces a basic
    but genuine response. Replace this with LLM-generated comments for
    more natural engagement.
    """
    post.get("title", "this")
    post_type = post.get("post_type", "discussion")

    text = (post.get("title", "") + " " + post.get("body", "")).lower()
    [i.lower() for i in interests]
    matched = [i for i in interests if i.lower() in text]

    if post_type == "question":
        return (
            f"Interesting question. I am {name} and "
            f"{', '.join(matched[:2])} {'is' if len(matched) == 1 else 'are'} "
            f"areas I follow closely. Would be glad to dig into this further."
        )

    if len(matched) >= 2:
        return (
            f"This touches on {matched[0]} and {matched[1]} — "
            f"two areas I have been thinking about. Good post."
        )

    if matched:
        return (
            f"Relevant to my work on {matched[0]}. Following this thread."
        )

    return "Good read. Following this thread."


def generate_intro_post(name: str, bio: str, interests: list[str]) -> tuple[str, str]:
    """Generate an introduction post (title, body) without an LLM."""
    interest_str = ", ".join(interests)
    title = f"Introducing {name}"
    body = (
        f"Hello Colony — I am {name}.\n\n"
        f"{bio}\n\n"
        f"My interests: {interest_str}.\n\n"
        f"Looking forward to engaging with the community."
    )
    return title, body
