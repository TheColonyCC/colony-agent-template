"""Tests for colony_agent.rules."""

from colony_agent.rules import (
    generate_comment,
    generate_intro_post,
    should_comment,
    should_downvote,
    should_vote,
)


class TestShouldVote:
    def test_match_in_title(self):
        post = {"title": "New AI breakthrough", "body": "Some details."}
        assert should_vote(post, ["AI", "robotics"]) is True

    def test_match_in_body(self):
        post = {"title": "Update", "body": "Working on robotics project."}
        assert should_vote(post, ["robotics"]) is True

    def test_no_match(self):
        post = {"title": "Cooking tips", "body": "How to make pasta."}
        assert should_vote(post, ["AI", "robotics"]) is False

    def test_case_insensitive(self):
        post = {"title": "MACHINE LEARNING", "body": ""}
        assert should_vote(post, ["machine learning"]) is True

    def test_empty_interests(self):
        post = {"title": "AI news", "body": "Breaking developments."}
        assert should_vote(post, []) is False

    def test_empty_post(self):
        post = {}
        assert should_vote(post, ["AI"]) is False


class TestShouldDownvote:
    def test_match_in_title(self):
        post = {"title": "Buy cheap tokens now", "body": "Details."}
        assert should_downvote(post, ["cheap tokens", "scam"]) is True

    def test_match_in_body(self):
        post = {"title": "Opportunity", "body": "This is a total scam."}
        assert should_downvote(post, ["scam"]) is True

    def test_no_match(self):
        post = {"title": "AI research", "body": "Interesting findings."}
        assert should_downvote(post, ["spam", "scam"]) is False

    def test_case_insensitive(self):
        post = {"title": "FREE GIVEAWAY", "body": ""}
        assert should_downvote(post, ["free giveaway"]) is True

    def test_empty_keywords(self):
        post = {"title": "Anything", "body": "spam scam garbage"}
        assert should_downvote(post, []) is False

    def test_empty_post(self):
        post = {}
        assert should_downvote(post, ["spam"]) is False


class TestShouldComment:
    def test_two_interest_matches(self):
        post = {"title": "AI agents for robotics", "body": "Combining both fields."}
        assert should_comment(post, ["AI", "robotics"]) is True

    def test_single_match_no_question(self):
        post = {"title": "AI news", "body": "Some update.", "post_type": "discussion"}
        assert should_comment(post, ["AI", "robotics"]) is False

    def test_question_with_one_match(self):
        post = {"title": "How does AI work?", "body": "Curious.", "post_type": "question"}
        assert should_comment(post, ["AI", "robotics"]) is True

    def test_question_with_no_match(self):
        post = {"title": "Cooking tips?", "body": "Help.", "post_type": "question"}
        assert should_comment(post, ["AI"]) is False

    def test_no_matches(self):
        post = {"title": "Random", "body": "Stuff."}
        assert should_comment(post, ["AI"]) is False


class TestGenerateComment:
    def test_question_type(self):
        post = {"title": "How does AI work?", "body": "explain", "post_type": "question"}
        result = generate_comment(post, "TestBot", ["AI"])
        assert "TestBot" in result
        assert "AI" in result

    def test_two_matches(self):
        post = {"title": "AI and robotics", "body": "combined", "post_type": "discussion"}
        result = generate_comment(post, "Bot", ["AI", "robotics"])
        assert "AI" in result
        assert "robotics" in result

    def test_single_match(self):
        post = {"title": "AI stuff", "body": "", "post_type": "discussion"}
        result = generate_comment(post, "Bot", ["AI"])
        assert "AI" in result

    def test_no_matches_fallback(self):
        post = {"title": "Random", "body": "things"}
        result = generate_comment(post, "Bot", ["crypto"])
        assert "Good read" in result


class TestGenerateIntroPost:
    def test_generates_title_and_body(self):
        title, body = generate_intro_post("TestBot", "I test things.", ["testing", "QA"])
        assert "TestBot" in title
        assert "TestBot" in body
        assert "I test things." in body
        assert "testing" in body
        assert "QA" in body

    def test_returns_tuple(self):
        result = generate_intro_post("Bot", "Bio.", ["AI"])
        assert isinstance(result, tuple)
        assert len(result) == 2
