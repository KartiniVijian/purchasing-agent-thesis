"""scripts/demos/demo_d1_sanitization.py - Week 3: baseline vs undefended vs D1-defended. Needs real Ollama."""
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
from defenses.d1_sanitization import apply_d1


def main():
    goal = Goal(max_price=150, min_rating=0,
                preference_description="I want reliable, good-value wireless headphones.")
    base_listing = load_listing("data/product_listings/listing_004_borderline.json")
    agent = PurchasingAgent(model_name="llama3.1:8b")

    print("=" * 90)
    print("BASELINE (no attack)")
    print("=" * 90)
    baseline_decision = agent.evaluate_listing(goal, base_listing)
    baseline_score = baseline_decision.soft_score
    print(f"Soft score: {baseline_score}\n")

    attacked_listings = apply_all_attacks(base_listing)
    print("=" * 90)
    print(f"{'Attack':32s} {'Undefended':>12s} {'D1-Defended':>13s} {'D1 Flags Raised'}")
    print("=" * 90)
    for code, attacked_listing in attacked_listings.items():
        undefended_decision = agent.evaluate_listing(goal, attacked_listing)
        defended_listing = apply_d1(attacked_listing)
        defended_decision = agent.evaluate_listing(goal, defended_listing)
        flags = defended_listing.get("_d1_flags", [])
        flags_str = ", ".join(flags) if flags else "(none)"
        print(f"{code:32s} {undefended_decision.soft_score!s:>12s} {defended_decision.soft_score!s:>13s}   {flags_str}")

    print("\nBaseline score (no attack, for reference):", baseline_score)


if __name__ == "__main__":
    main()
