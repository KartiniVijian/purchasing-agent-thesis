from unittest.mock import MagicMock
from decision_function import Goal, check_hard_constraints, get_soft_score, evaluate_product


def make_fake_llm(fake_score_text: str):
    fake_llm = MagicMock()
    fake_response = MagicMock()
    fake_response.content = fake_score_text
    fake_llm.invoke.return_value = fake_response
    return fake_llm


def test_hard_constraints_pass():
    goal = Goal(max_price=100, min_rating=4.0, required_features=["wireless"])
    product = {"price": 89.99, "rating": 4.5, "features": ["wireless", "bluetooth"]}
    assert check_hard_constraints(goal, product) == []


def test_hard_constraints_fail_on_price():
    goal = Goal(max_price=50, min_rating=4.0)
    product = {"price": 89.99, "rating": 4.5, "features": []}
    failures = check_hard_constraints(goal, product)
    assert any("Price" in f for f in failures)


def test_hard_constraints_fail_on_missing_feature():
    goal = Goal(max_price=100, min_rating=0, required_features=["waterproof"])
    product = {"price": 20, "rating": 4.0, "features": ["wireless"]}
    failures = check_hard_constraints(goal, product)
    assert any("waterproof" in f for f in failures)


def test_get_soft_score_parses_number():
    fake_llm = make_fake_llm("85")
    goal = Goal(max_price=100, preference_description="lightweight for travel")
    product = {"title": "Travel Headphones", "description": "Lightweight and foldable."}
    assert get_soft_score(fake_llm, goal, product) == 85.0


def test_get_soft_score_clamps_out_of_range():
    fake_llm = make_fake_llm("999")
    goal = Goal(max_price=100)
    product = {"title": "X", "description": "Y"}
    assert get_soft_score(fake_llm, goal, product) == 100.0


def test_get_soft_score_handles_garbage_response():
    fake_llm = make_fake_llm("I cannot determine a score.")
    goal = Goal(max_price=100)
    product = {"title": "X", "description": "Y"}
    assert get_soft_score(fake_llm, goal, product) == 0.0


def test_evaluate_product_rejects_on_hard_constraint_without_calling_llm():
    fake_llm = make_fake_llm("100")
    goal = Goal(max_price=10)
    product = {"price": 500, "rating": 5.0, "features": []}
    decision = evaluate_product(fake_llm, goal, product)
    assert decision.accepted is False
    assert decision.soft_score is None
    fake_llm.invoke.assert_not_called()


def test_evaluate_product_accepts_above_threshold():
    fake_llm = make_fake_llm("75")
    goal = Goal(max_price=100, min_rating=4.0, preference_description="travel")
    product = {"price": 89.99, "rating": 4.5, "features": [], "title": "X", "description": "Y"}
    decision = evaluate_product(fake_llm, goal, product, soft_score_threshold=50)
    assert decision.accepted is True
    assert decision.soft_score == 75.0


def test_evaluate_product_rejects_below_threshold():
    fake_llm = make_fake_llm("20")
    goal = Goal(max_price=100, min_rating=4.0, preference_description="travel")
    product = {"price": 89.99, "rating": 4.5, "features": [], "title": "X", "description": "Y"}
    decision = evaluate_product(fake_llm, goal, product, soft_score_threshold=50)
    assert decision.accepted is False
