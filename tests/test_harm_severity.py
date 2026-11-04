"""test_harm_severity.py - tests the harm severity rubric's structure and math."""

from harm_severity import HARM_SCORES, get_severity
from attacks.attack_functions import list_attack_codes


def test_every_attack_code_has_a_harm_score():
    """Every attack subtype in the taxonomy must have a severity entry, or
    the risk ranking (Table 6.4) would silently drop attacks."""
    for code in list_attack_codes():
        assert code in HARM_SCORES, f"{code} is missing a harm severity score"


def test_severity_is_average_of_three_axes():
    score = HARM_SCORES["C1_fake_review_injection"]
    expected = (score.deception_directness + score.financial_exposure + score.discoverability) / 3.0
    assert get_severity("C1_fake_review_injection") == expected


def test_severity_stays_within_valid_range():
    for code in list_attack_codes():
        severity = get_severity(code)
        assert 1.0 <= severity <= 5.0


def test_trust_mechanism_attacks_score_highest_on_average():
    """
    Sanity check on the rubric's design intent: C-family attacks (fabricated
    reviews/certifications) fabricate specific checkable facts and should
    score at least as high, on average, as B-family (persuasive framing,
    no checkable fact).
    """
    c_family = ["C1_fake_review_injection", "C2_fake_certification", "C3_evaluator_impersonation"]
    b_family = ["B1_scarcity_urgency", "B2_social_proof_authority", "B3_anchoring_fake_discount", "B4_emotional_narrative"]

    c_avg = sum(get_severity(c) for c in c_family) / len(c_family)
    b_avg = sum(get_severity(b) for b in b_family) / len(b_family)

    assert c_avg >= b_avg
