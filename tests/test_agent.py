from unittest.mock import MagicMock
from decision_function import Goal
from agent import PurchasingAgent


def make_fake_llm(fake_score_text: str):
    fake_llm = MagicMock()
    fake_response = MagicMock()
    fake_response.content = fake_score_text
    fake_llm.invoke.return_value = fake_response
    return fake_llm


def test_agent_evaluates_single_listing():
    fake_llm = make_fake_llm("80")
    agent = PurchasingAgent(llm=fake_llm)
    goal = Goal(max_price=100, min_rating=4.0, preference_description="travel")
    listing = {"title": "Travel Headphones", "price": 89.99, "rating": 4.5, "features": [],
               "description": "Lightweight, foldable.", "_source_file": "test.json"}
    decision = agent.evaluate_listing(goal, listing)
    assert decision.accepted is True
    assert len(agent.trace) == 1
    assert agent.trace[0].listing_title == "Travel Headphones"


def test_agent_evaluates_batch_of_listings():
    fake_llm = make_fake_llm("60")
    agent = PurchasingAgent(llm=fake_llm)
    goal = Goal(max_price=100, min_rating=0)
    listings = [
        {"title": "A", "price": 10, "rating": 4.0, "features": [], "description": "d"},
        {"title": "B", "price": 500, "rating": 4.0, "features": [], "description": "d"},
    ]
    decisions = agent.evaluate_all(goal, listings)
    assert len(decisions) == 2
    assert decisions[0].accepted is True
    assert decisions[1].accepted is False


def test_agent_never_needs_real_ollama_in_tests():
    fake_llm = make_fake_llm("50")
    agent = PurchasingAgent(llm=fake_llm)
    assert agent.llm is fake_llm
