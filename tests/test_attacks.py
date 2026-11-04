import pytest
from attacks.attack_functions import load_payloads, list_attack_codes, apply_attack, apply_all_attacks

CLEAN_LISTING = {
    "title": "Wireless Headphones", "price": 89.99, "rating": 4.5,
    "features": ["wireless", "bluetooth"],
    "description": "Comfortable over-ear headphones with good battery life.",
}


def test_payload_file_loads_and_has_ten_attacks():
    assert len(load_payloads()) == 10


def test_all_three_families_are_represented():
    payloads = load_payloads()
    families = {info["family"] for info in payloads.values()}
    assert families == {"prompt_injection", "cognitive_bias_exploitation", "trust_mechanism_abuse"}


def test_list_attack_codes_returns_all_ten():
    codes = list_attack_codes()
    assert len(codes) == 10
    assert "A1_direct_instruction" in codes
    assert "C3_evaluator_impersonation" in codes


def test_apply_attack_modifies_description_only():
    attacked = apply_attack("A1_direct_instruction", CLEAN_LISTING)
    assert len(attacked["description"]) > len(CLEAN_LISTING["description"])
    assert CLEAN_LISTING["description"] in attacked["description"]


def test_apply_attack_does_not_touch_hard_constraint_fields():
    attacked = apply_attack("B2_social_proof_authority", CLEAN_LISTING)
    assert attacked["price"] == CLEAN_LISTING["price"]
    assert attacked["rating"] == CLEAN_LISTING["rating"]
    assert attacked["features"] == CLEAN_LISTING["features"]


def test_apply_attack_does_not_mutate_original_listing():
    original_description = CLEAN_LISTING["description"]
    apply_attack("A2_delimiter_escape", CLEAN_LISTING)
    assert CLEAN_LISTING["description"] == original_description


def test_apply_attack_tags_the_listing():
    attacked = apply_attack("C1_fake_review_injection", CLEAN_LISTING)
    assert attacked["_attack_code"] == "C1_fake_review_injection"
    assert attacked["_attack_family"] == "trust_mechanism_abuse"
    assert "_attack_name" in attacked


def test_apply_attack_rejects_unknown_code():
    with pytest.raises(ValueError):
        apply_attack("Z9_not_a_real_attack", CLEAN_LISTING)


def test_apply_all_attacks_returns_ten_variants():
    all_attacked = apply_all_attacks(CLEAN_LISTING)
    assert len(all_attacked) == 10
    for code, listing in all_attacked.items():
        assert listing["_attack_code"] == code


def test_apply_all_attacks_leaves_original_untouched():
    original_description = CLEAN_LISTING["description"]
    apply_all_attacks(CLEAN_LISTING)
    assert CLEAN_LISTING["description"] == original_description
