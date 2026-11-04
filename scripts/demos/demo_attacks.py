"""scripts/demos/demo_attacks.py - Week 2 attack comparison. Needs real Ollama running."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # data/ and results/ paths are relative to the project root
from decision_function import Goal
from listing_loader import load_listing
from agent import PurchasingAgent
from attacks.attack_functions import apply_all_attacks


def main():
    goal = Goal(max_price=150, min_rating=0,
                preference_description="I want reliable, good-value wireless headphones.")
    base_listing = load_listing("data/product_listings/listing_004_borderline.json")
    agent = PurchasingAgent(model_name="llama3.1:8b")

    print("=" * 70)
    print("BASELINE (no attack)")
    print("=" * 70)
    baseline_decision = agent.evaluate_listing(goal, base_listing)
    print(f"Soft score: {baseline_decision.soft_score}")
    print(f"Reason: {baseline_decision.reason}\n")

    attacked_listings = apply_all_attacks(base_listing)
    print("=" * 70)
    print("ATTACKED VARIANTS")
    print("=" * 70)
    for code, attacked_listing in attacked_listings.items():
        decision = agent.evaluate_listing(goal, attacked_listing)
        shift = None
        if decision.soft_score is not None and baseline_decision.soft_score is not None:
            shift = decision.soft_score - baseline_decision.soft_score
        shift_str = f"{shift:+.1f}" if shift is not None else "n/a"
        print(f"{code:35s}  soft_score={decision.soft_score!s:>6}  shift={shift_str}")


if __name__ == "__main__":
    main()
