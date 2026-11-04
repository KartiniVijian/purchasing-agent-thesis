"""
scripts/demos/demo_d3_instruction_hierarchy.py

Four-way comparison, using the real Ollama model:
    1. UNDEFENDED     - no defenses at all
    2. D3-ONLY        - just the structural instruction-hierarchy separation,
                        isolating what D3 contributes on its own
    3. D1+D2          - the Week 4 stack, for direct comparison
    4. FULL STACK     - D1 + D2 + D3 all together

D3-only is the important new data point here: everything else in this
project so far tells us what D1 and D2 each contribute. This run tells us
whether structural message separation, entirely on its own (no text
editing, no classification gate), changes anything - particularly against
A2 (delimiter mimicry), which is exactly the kind of attack D3's design
rationale predicts it should help with.

Needs real Ollama running.

Run it with:
    python scripts/demos/demo_d3_instruction_hierarchy.py
"""

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
        return "REJECTED (D2 guardrail)"
    if decision.soft_score is None:
        return "REJECTED (hard constraint)"
    status = "ACCEPTED" if decision.accepted else "REJECTED"
    return f"{status} ({decision.soft_score:.0f})"


def main():
    goal = Goal(
        max_price=150,
        min_rating=0,
        preference_description="I want reliable, good-value wireless headphones.",
    )

    base_listing = load_listing("data/product_listings/listing_004_borderline.json")
    llm = ChatOllama(model="llama3.1:8b", temperature=0.0)

    print("=" * 110)
    print("BASELINE (no attack, no defenses)")
    print("=" * 110)
    baseline = evaluate_with_defenses(llm, goal, base_listing, use_d1=False, use_d2=False, use_d3=False)
    print(format_result(baseline), "\n")

    attacked_listings = apply_all_attacks(base_listing)

    print("=" * 110)
    print(f"{'Attack':30s} {'Undefended':>18s} {'D3-only':>18s} {'D1+D2':>22s} {'Full stack (D1+D2+D3)':>24s}")
    print("=" * 110)

    for code, listing in attacked_listings.items():
        undefended = evaluate_with_defenses(llm, goal, listing, use_d1=False, use_d2=False, use_d3=False)
        d3_only = evaluate_with_defenses(llm, goal, listing, use_d1=False, use_d2=False, use_d3=True)
        d1_d2 = evaluate_with_defenses(llm, goal, listing, use_d1=True, use_d2=True, use_d3=False)
        full_stack = evaluate_with_defenses(llm, goal, listing, use_d1=True, use_d2=True, use_d3=True)

        print(
            f"{code:30s} {format_result(undefended):>18s} {format_result(d3_only):>18s} "
            f"{format_result(d1_d2):>22s} {format_result(full_stack):>24s}"
        )

    print("\nBaseline (no attack):", format_result(baseline))
    print("\nD3-only isolates the effect of structural message separation alone,")
    print("with no text editing (D1) or classification gate (D2) involved.")


if __name__ == "__main__":
    main()
