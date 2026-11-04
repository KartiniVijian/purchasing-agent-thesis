from unittest.mock import MagicMock
from decision_function import Goal
from defense_pipeline import evaluate_with_defenses


def make_fake_llm(guardrail_response: str, soft_score_response: str):
    llm = MagicMock()

    def fake_invoke(prompt):
        response = MagicMock()
        if isinstance(prompt, str) and "MANIPULATIVE" in prompt:
            response.content = guardrail_response
        else:
            response.content = soft_score_response
        return response

    llm.invoke.side_effect = fake_invoke
    return llm


CLEAN_LISTING = {
    "title": "Wireless Headphones", "price": 89.99, "rating": 4.5,
    "features": ["wireless"],
    "description": "Comfortable over-ear headphones with good battery life.",
}


def test_clean_listing_passes_through_both_defenses_and_is_scored():
    llm = make_fake_llm(guardrail_response="NO", soft_score_response="80")
    goal = Goal(max_price=150, min_rating=0, preference_description="travel headphones")
    decision = evaluate_with_defenses(llm, goal, CLEAN_LISTING)
    assert decision.accepted is True
    assert decision.soft_score == 80.0
    assert decision.d2_flagged is False
    assert decision.d1_flags == []


def test_d2_rejects_manipulative_listing_before_soft_scoring():
    llm = make_fake_llm(guardrail_response="YES", soft_score_response="100")
    goal = Goal(max_price=150, min_rating=0, preference_description="travel headphones")
    decision = evaluate_with_defenses(llm, goal, CLEAN_LISTING)
    assert decision.accepted is False
    assert decision.d2_flagged is True
    assert decision.soft_score is None
    assert "D2 guardrail" in decision.reason


def test_hard_constraint_failure_short_circuits_before_d2_is_ever_called():
    llm = make_fake_llm(guardrail_response="YES", soft_score_response="100")
    goal = Goal(max_price=10, min_rating=0)
    decision = evaluate_with_defenses(llm, goal, CLEAN_LISTING)
    assert decision.accepted is False
    assert decision.soft_score is None
    llm.invoke.assert_not_called()


def test_use_d1_false_skips_d1_and_records_no_flags():
    llm = make_fake_llm(guardrail_response="NO", soft_score_response="60")
    goal = Goal(max_price=150, min_rating=0)
    decision = evaluate_with_defenses(llm, goal, CLEAN_LISTING, use_d1=False)
    assert decision.d1_flags == []
    assert decision.accepted is True


def test_use_d2_false_skips_guardrail_and_goes_straight_to_scoring():
    llm = make_fake_llm(guardrail_response="YES", soft_score_response="90")
    goal = Goal(max_price=150, min_rating=0)
    decision = evaluate_with_defenses(llm, goal, CLEAN_LISTING, use_d2=False)
    assert decision.d2_flagged is False
    assert decision.accepted is True
    assert decision.soft_score == 90.0


def test_undefended_mode_matches_plain_evaluate_product_behaviour():
    llm = make_fake_llm(guardrail_response="YES", soft_score_response="70")
    goal = Goal(max_price=150, min_rating=0)
    decision = evaluate_with_defenses(llm, goal, CLEAN_LISTING, use_d1=False, use_d2=False)
    assert decision.accepted is True
    assert decision.soft_score == 70.0
    assert decision.d1_flags == []
    assert decision.d2_flagged is False


def test_d3_defaults_to_off_so_existing_call_sites_are_unaffected():
    """
    Regression guard: scripts/demos/demo_d1_sanitization.py and scripts/demos/demo_d2_guardrail.py call
    evaluate_with_defenses() without ever mentioning use_d3. If D3 ever
    defaulted to True, those existing "Undefended"/"D1-only"/"D1+D2"
    condition labels would silently stop meaning what they say.
    """
    llm = make_fake_llm(guardrail_response="NO", soft_score_response="65")
    goal = Goal(max_price=150, min_rating=0)
    decision = evaluate_with_defenses(llm, goal, CLEAN_LISTING)  # use_d3 not passed
    assert decision.d3_applied is False
    # with D3 off, scoring must go through the flat-string path, so the
    # guardrail-detection substring check in make_fake_llm still works -
    # if this assertion fails, it means a message-list prompt was sent
    # instead of a string, which only happens when D3 is (wrongly) on.
    assert decision.soft_score == 65.0


def test_d3_true_uses_hierarchical_scoring_and_records_it():
    llm = make_fake_llm(guardrail_response="NO", soft_score_response="72")
    goal = Goal(max_price=150, min_rating=0, preference_description="travel headphones")
    decision = evaluate_with_defenses(llm, goal, CLEAN_LISTING, use_d3=True)
    assert decision.d3_applied is True
    assert decision.soft_score == 72.0
    assert decision.accepted is True


def test_d3_alone_isolates_its_own_effect():
    """D3 can be tested independently of D1/D2, isolating its own contribution."""
    llm = make_fake_llm(guardrail_response="YES", soft_score_response="55")
    goal = Goal(max_price=150, min_rating=0)
    decision = evaluate_with_defenses(
        llm, goal, CLEAN_LISTING, use_d1=False, use_d2=False, use_d3=True
    )
    assert decision.d3_applied is True
    assert decision.d2_flagged is False  # D2 was off, so never even asked
    assert decision.soft_score == 55.0
