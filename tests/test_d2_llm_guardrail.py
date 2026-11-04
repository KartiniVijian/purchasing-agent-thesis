from unittest.mock import MagicMock
from defenses.d2_llm_guardrail import guardrail_check


def make_fake_llm(fake_response_text: str):
    fake_llm = MagicMock()
    fake_response = MagicMock()
    fake_response.content = fake_response_text
    fake_llm.invoke.return_value = fake_response
    return fake_llm


def test_parses_clean_yes():
    llm = make_fake_llm("YES")
    is_manipulative, raw, parse_error = guardrail_check(llm, "some description")
    assert is_manipulative is True
    assert parse_error is False


def test_parses_clean_no():
    llm = make_fake_llm("NO")
    is_manipulative, raw, parse_error = guardrail_check(llm, "some description")
    assert is_manipulative is False
    assert parse_error is False


def test_parses_lowercase_yes():
    llm = make_fake_llm("yes")
    is_manipulative, raw, parse_error = guardrail_check(llm, "some description")
    assert is_manipulative is True
    assert parse_error is False


def test_parses_yes_with_trailing_punctuation():
    llm = make_fake_llm("YES.")
    is_manipulative, raw, parse_error = guardrail_check(llm, "some description")
    assert is_manipulative is True
    assert parse_error is False


def test_unparseable_response_fails_open_but_flags_parse_error():
    llm = make_fake_llm("I'm not sure, this could go either way.")
    is_manipulative, raw, parse_error = guardrail_check(llm, "some description")
    assert is_manipulative is False
    assert parse_error is True


def test_guardrail_receives_the_description_in_the_prompt():
    llm = make_fake_llm("NO")
    guardrail_check(llm, "UNIQUE_MARKER_TEXT_12345")
    sent_prompt = llm.invoke.call_args[0][0]
    assert "UNIQUE_MARKER_TEXT_12345" in sent_prompt


def test_prompt_includes_fabricated_specificity_guidance():
    from defenses.d2_llm_guardrail import GUARDRAIL_PROMPT_TEMPLATE
    assert "UNVERIFIABLE" in GUARDRAIL_PROMPT_TEMPLATE.upper()
    assert "certification" in GUARDRAIL_PROMPT_TEMPLATE.lower()


def test_parses_final_answer_marker_yes_with_reasoning_before_it():
    llm = make_fake_llm(
        "The description cites an exact buyer count and a named award, "
        "which are unverifiable claims typical of fabricated social proof.\n"
        "FINAL ANSWER: YES"
    )
    is_manipulative, raw, parse_error = guardrail_check(llm, "some description")
    assert is_manipulative is True
    assert parse_error is False


def test_parses_final_answer_marker_no_with_reasoning_before_it():
    llm = make_fake_llm(
        "The description only mentions battery life and comfort, which "
        "are ordinary factual product attributes.\n"
        "FINAL ANSWER: NO"
    )
    is_manipulative, raw, parse_error = guardrail_check(llm, "some description")
    assert is_manipulative is False
    assert parse_error is False


def test_final_answer_marker_takes_priority_over_leading_words():
    llm = make_fake_llm(
        "No obvious red flags jump out immediately, but on reflection the "
        "certification claim cannot be verified.\n"
        "FINAL ANSWER: YES"
    )
    is_manipulative, raw, parse_error = guardrail_check(llm, "some description")
    assert is_manipulative is True
    assert parse_error is False


def test_old_bare_yes_no_format_still_parses_correctly():
    llm_yes = make_fake_llm("YES")
    is_manipulative, _, parse_error = guardrail_check(llm_yes, "some description")
    assert is_manipulative is True
    assert parse_error is False

    llm_no = make_fake_llm("NO")
    is_manipulative, _, parse_error = guardrail_check(llm_no, "some description")
    assert is_manipulative is False
    assert parse_error is False
