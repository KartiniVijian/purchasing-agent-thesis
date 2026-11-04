"""
test_d3_instruction_hierarchy.py

Tests D3's message construction and scoring logic using a mocked LLM.
No Ollama installation needed.
"""

from unittest.mock import MagicMock

from langchain_core.messages import SystemMessage, HumanMessage
from decision_function import Goal
from defenses.d3_instruction_hierarchy import (
    build_hierarchical_messages,
    get_soft_score_with_d3,
    parse_score_from_structured_text,
)


def make_fake_llm(fake_score_text: str):
    fake_llm = MagicMock()
    fake_response = MagicMock()
    fake_response.content = fake_score_text
    fake_llm.invoke.return_value = fake_response
    return fake_llm


def test_build_hierarchical_messages_returns_system_then_human():
    goal = Goal(max_price=100, preference_description="lightweight for travel")
    product = {"title": "Travel Headphones", "description": "Lightweight and foldable."}
    messages = build_hierarchical_messages(goal, product)
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)


def test_system_message_contains_goal_and_title_only():
    goal = Goal(max_price=100, preference_description="lightweight for travel")
    product = {"title": "Travel Headphones", "description": "UNIQUE_DESCRIPTION_MARKER"}
    messages = build_hierarchical_messages(goal, product)
    system_content = messages[0].content
    assert "lightweight for travel" in system_content
    assert "Travel Headphones" in system_content
    assert "UNIQUE_DESCRIPTION_MARKER" not in system_content


def test_human_message_contains_only_the_description():
    goal = Goal(max_price=100, preference_description="lightweight for travel")
    product = {"title": "Travel Headphones", "description": "UNIQUE_DESCRIPTION_MARKER"}
    messages = build_hierarchical_messages(goal, product)
    human_content = messages[1].content
    assert "UNIQUE_DESCRIPTION_MARKER" in human_content
    assert "lightweight for travel" not in human_content


def test_human_message_wraps_description_in_tags():
    goal = Goal(max_price=100)
    product = {"title": "X", "description": "some text"}
    messages = build_hierarchical_messages(goal, product)
    assert "<seller_provided_description>" in messages[1].content
    assert "</seller_provided_description>" in messages[1].content


def test_system_message_includes_fabricated_specificity_guidance():
    """
    Regression guard for the Week 5 v2 revision: D3's system message must
    warn against specific-but-unverifiable claims (named awards, exact
    buyer counts, quoted reviews, named certifications), since the v1
    wording (commands/instructions only) was shown empirically to let C2
    and similar attacks through unchallenged.
    """
    from defenses.d3_instruction_hierarchy import D3_SYSTEM_MESSAGE_TEMPLATE
    assert "UNVERIFIABLE" in D3_SYSTEM_MESSAGE_TEMPLATE.upper()
    assert "certification" in D3_SYSTEM_MESSAGE_TEMPLATE.lower()


def test_system_message_scopes_skepticism_away_from_factual_specs():
    """
    Regression guard for the Week 5 v3 revision: v2's skepticism guidance
    was shown empirically to over-generalise, penalising a genuinely
    honest listing (battery life stated, no attack present) for not
    repeating the buyer's exact preference wording. v3 must explicitly
    scope the warning to third-party trust claims and permit reasonable
    inference from concrete factual specs.
    """
    from defenses.d3_instruction_hierarchy import D3_SYSTEM_MESSAGE_TEMPLATE
    assert "battery life" in D3_SYSTEM_MESSAGE_TEMPLATE.lower()
    assert "reasonable" in D3_SYSTEM_MESSAGE_TEMPLATE.lower()


def test_get_soft_score_with_d3_parses_bare_number_fallback():
    """Backward compatibility: a bare-number response (no FINAL SCORE marker) still parses."""
    llm = make_fake_llm("77")
    goal = Goal(max_price=100, preference_description="travel")
    product = {"title": "X", "description": "Y"}
    score = get_soft_score_with_d3(llm, goal, product)
    assert score == 77.0


def test_get_soft_score_with_d3_sends_a_message_list_not_a_string():
    llm = make_fake_llm("50")
    goal = Goal(max_price=100)
    product = {"title": "X", "description": "Y"}
    get_soft_score_with_d3(llm, goal, product)
    sent_prompt = llm.invoke.call_args[0][0]
    assert isinstance(sent_prompt, list)
    assert len(sent_prompt) == 2


def test_get_soft_score_with_d3_fails_safe_on_unparseable_text():
    llm = make_fake_llm("not a number at all")
    goal = Goal(max_price=100)
    product = {"title": "X", "description": "Y"}
    score = get_soft_score_with_d3(llm, goal, product)
    assert score == 0.0


def test_parses_final_score_marker_with_reasoning_before_it():
    raw = (
        "The description mentions wireless connectivity but does not address "
        "reliability directly.\nFINAL SCORE: 45"
    )
    assert parse_score_from_structured_text(raw) == 45.0


def test_final_score_marker_ignores_stray_numbers_in_reasoning():
    """
    Critical regression test for the digit-concatenation bug: reasoning
    text containing unrelated numbers (a battery spec, a buyer count)
    must NOT corrupt the parsed score. Without the FINAL SCORE marker,
    naive digit-scraping would read this as "30100" or similar garbage.
    """
    raw = (
        "This claims a 30-hour battery life and cites 50,000 buyers, which "
        "sounds like fabricated specificity and should not inflate the score.\n"
        "FINAL SCORE: 20"
    )
    assert parse_score_from_structured_text(raw) == 20.0


def test_final_score_marker_clamps_to_valid_range():
    raw = "Reasoning here.\nFINAL SCORE: 150"
    assert parse_score_from_structured_text(raw) == 100.0
