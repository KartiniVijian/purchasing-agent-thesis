"""scripts/demos/demo_d2_guardrail.py - Week 4: undefended vs D1-only vs D1+D2. Needs real Ollama."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # data/ and results/ paths are relative to the project root
from langchain_ollama import ChatOllama
from decision_function import Goal
from listing_loader import load_listing
from attacks.attack_functions import apply_all_attacks
from defense_pipeline import evaluate_with_defenses


def format_result(decision) -> str:
    if decision.d2_flagged:
        return "REJECTED (by D2 guardrail)"
    if decision.soft_score is None:
        return "REJECTED (hard constraint)"
    status = "ACCEPTED" if decision.accepted else "REJECTED"
    return f"{status} (score={decision.soft_score:.0f})"


def main():
    goal = Goal(max_price=150, min_rating=0,
                preference_description="I want reliable, good-value wireless headphones.")
    base_listing = load_listing("data/product_listings/listing_004_borderline.json")
    llm = ChatOllama(model="llama3.1:8b", temperature=0.0)

    print("=" * 100)
    print("BASELINE (no attack, no defenses)")
    print("=" * 100)
    baseline = evaluate_with_defenses(llm, goal, base_listing, use_d1=False, use_d2=False)
    print(format_result(baseline), "\n")

    attacked_listings = apply_all_attacks(base_listing)
    print("=" * 100)
    print(f"{'Attack':32s} {'Undefended':>28s} {'D1-only':>28s} {'D1+D2':>28s}")
    print("=" * 100)
    for code, listing in attacked_listings.items():
        undefended = evaluate_with_defenses(llm, goal, listing, use_d1=False, use_d2=False)
        d1_only = evaluate_with_defenses(llm, goal, listing, use_d1=True, use_d2=False)
        d1_plus_d2 = evaluate_with_defenses(llm, goal, listing, use_d1=True, use_d2=True)
        print(f"{code:32s} {format_result(undefended):>28s} {format_result(d1_only):>28s} {format_result(d1_plus_d2):>28s}")

    print("\nBaseline (no attack):", format_result(baseline))


if __name__ == "__main__":
    main()
