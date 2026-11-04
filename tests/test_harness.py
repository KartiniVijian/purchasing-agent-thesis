"""
test_harness.py

Tests the evaluation harness. run_single_trial/run_harness are tested
with a mocked LLM (same pattern as every other week). The aggregation
functions (compute_success_rates, compute_defence_effectiveness,
compute_risk_ranking) are pure data processing with no LLM involved at
all, so they're tested directly against hand-built TrialResult lists -
this is the most important part to get right, since it's what actually
produces Chapter 6's numbers from raw trial data.
"""

from unittest.mock import MagicMock

from decision_function import Goal
from harness import (
    CONDITIONS,
    TrialResult,
    run_single_trial,
    run_harness,
    compute_success_rates,
    compute_defence_effectiveness,
    compute_risk_ranking,
)


CLEAN_LISTING = {
    "title": "Wireless Headphones", "price": 89.99, "rating": 4.5,
    "features": ["wireless"], "description": "Comfortable, good battery life.",
    "_source_file": "test_listing.json",
}


def make_fake_llm(guardrail_response: str, soft_score_response: str):
    llm = MagicMock()

    def fake_invoke(prompt):
        response = MagicMock()
        if isinstance(prompt, str) and "MANIPULATIVE" in prompt:
            response.content = guardrail_response
        elif isinstance(prompt, list):
            response.content = f"Reasoning.\nFINAL SCORE: {soft_score_response}"
        else:
            response.content = soft_score_response
        return response

    llm.invoke.side_effect = fake_invoke
    return llm


# --- run_single_trial / run_harness ---

def test_run_single_trial_returns_populated_result():
    llm = make_fake_llm(guardrail_response="NO", soft_score_response="70")
    goal = Goal(max_price=150, min_rating=0, preference_description="travel")
    result = run_single_trial(llm, goal, CLEAN_LISTING, "A1_direct_instruction", "undefended", trial=1)
    assert result.attack_code == "A1_direct_instruction"
    assert result.condition == "undefended"
    assert result.trial == 1
    assert result.accepted is True
    assert result.soft_score == 70.0


def test_run_harness_covers_every_combination():
    llm = make_fake_llm(guardrail_response="NO", soft_score_response="60")
    goal = Goal(max_price=150, min_rating=0)
    results = run_harness(
        llm, goal, [CLEAN_LISTING],
        attack_codes=["A1_direct_instruction", "B1_scarcity_urgency"],
        conditions=["undefended", "D1_only"],
        n_trials=2,
        progress=False,
    )
    # 1 listing x 2 attacks x 2 conditions x 2 trials = 8 rows
    assert len(results) == 8


def test_all_seven_conditions_are_defined():
    assert len(CONDITIONS) == 7
    assert CONDITIONS["undefended"] == (False, False, False)
    assert CONDITIONS["full_stack"] == (True, True, True)
    assert CONDITIONS["D1_D3_no_D2"] == (True, False, True)  # isolates D1+D3 with D2 off


# --- compute_success_rates ---

def make_result(attack_code, condition, accepted, trial=1):
    return TrialResult(
        listing_id="x", attack_code=attack_code, attack_family="prompt_injection",
        condition=condition, trial=trial, accepted=accepted, soft_score=50.0,
        d2_flagged=False, hard_constraint_failed=False,
    )


def test_compute_success_rates_all_succeed():
    results = [make_result("A1_direct_instruction", "undefended", True) for _ in range(3)]
    rates = compute_success_rates(results)
    assert rates["A1_direct_instruction"]["undefended"] == 1.0


def test_compute_success_rates_partial():
    results = [
        make_result("A1_direct_instruction", "undefended", True),
        make_result("A1_direct_instruction", "undefended", True),
        make_result("A1_direct_instruction", "undefended", False),
        make_result("A1_direct_instruction", "undefended", False),
    ]
    rates = compute_success_rates(results)
    assert rates["A1_direct_instruction"]["undefended"] == 0.5


def test_compute_success_rates_separates_by_condition():
    results = [
        make_result("A1_direct_instruction", "undefended", True),
        make_result("A1_direct_instruction", "full_stack", False),
    ]
    rates = compute_success_rates(results)
    assert rates["A1_direct_instruction"]["undefended"] == 1.0
    assert rates["A1_direct_instruction"]["full_stack"] == 0.0


# --- compute_defence_effectiveness ---

def test_compute_defence_effectiveness_zero_successes_is_fully_effective():
    results = [make_result("A1_direct_instruction", "full_stack", False) for _ in range(5)]
    effectiveness = compute_defence_effectiveness(results)
    assert effectiveness["full_stack"] == 1.0


def test_compute_defence_effectiveness_all_successes_is_zero():
    results = [make_result("A1_direct_instruction", "undefended", True) for _ in range(5)]
    effectiveness = compute_defence_effectiveness(results)
    assert effectiveness["undefended"] == 0.0


def test_compute_defence_effectiveness_averages_across_attacks():
    results = [
        make_result("A1_direct_instruction", "D1_only", True),   # attack succeeds
        make_result("B2_social_proof_authority", "D1_only", False),  # attack fails
    ]
    effectiveness = compute_defence_effectiveness(results)
    assert effectiveness["D1_only"] == 0.5


# --- compute_risk_ranking ---

def test_compute_risk_ranking_sorts_highest_risk_first():
    results = [
        # C2 (severity 4.67): high undefended success rate -> should rank highest
        make_result("C2_fake_certification", "undefended", True),
        make_result("C2_fake_certification", "undefended", True),
        # B4 (severity 1.67): also high success rate, but much lower severity
        make_result("B4_emotional_narrative", "undefended", True),
        make_result("B4_emotional_narrative", "undefended", True),
    ]
    ranking = compute_risk_ranking(results)
    assert ranking[0]["attack_code"] == "C2_fake_certification"
    assert ranking[0]["risk_score"] > ranking[1]["risk_score"]


def test_compute_risk_ranking_zero_success_rate_gives_zero_risk():
    results = [make_result("C2_fake_certification", "undefended", False)]
    ranking = compute_risk_ranking(results)
    assert ranking[0]["risk_score"] == 0.0
