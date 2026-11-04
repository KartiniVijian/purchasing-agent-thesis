"""
scripts/diagnostics/d3_clean_score_shift.py

Checks whether D3's score for CLEAN (non-attacked) listings differs from
the plain flat-string score for the same listings, using the same goal
used throughout scripts/demos/demo_d3_instruction_hierarchy.py.

Why this matters: scripts/demos/demo_d3_instruction_hierarchy.py showed B1, B3, and C3 all shifting by
exactly +20 under D3-only, without flipping the accept/reject decision.
That uniform shift could mean either:
    (a) D3 makes the model generally more lenient/generous across the
        board (a format effect, unrelated to attack content), or
    (b) D3 is specifically responding to something in those particular
        attack payloads.
This script isolates (a) by measuring D3's shift on listings with NO
attack applied at all. If clean listings shift by a similar amount, the
B1/B3/C3 pattern is explained by general format inflation, not by D3
partially falling for those specific attacks.

Run it with:
    python scripts/diagnostics/d3_clean_score_shift.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # data/ and results/ paths are relative to the project root

from langchain_ollama import ChatOllama
from decision_function import Goal
from listing_loader import load_all_listings
from defense_pipeline import evaluate_with_defenses


def main():
    goal = Goal(
        max_price=150,
        min_rating=0,
        preference_description="I want reliable, good-value wireless headphones.",
    )
    listings = load_all_listings("data/product_listings")
    llm = ChatOllama(model="llama3.1:8b", temperature=0.0)

    print(f"{'Listing':30s} {'Flat score':>12s} {'D3 score':>12s} {'Shift':>8s}")
    print("-" * 70)

    for listing in listings:
        flat_decision = evaluate_with_defenses(
            llm, goal, listing, use_d1=False, use_d2=False, use_d3=False
        )
        d3_decision = evaluate_with_defenses(
            llm, goal, listing, use_d1=False, use_d2=False, use_d3=True
        )
        shift = None
        if flat_decision.soft_score is not None and d3_decision.soft_score is not None:
            shift = d3_decision.soft_score - flat_decision.soft_score
        shift_str = f"{shift:+.0f}" if shift is not None else "n/a"

        print(
            f"{listing['_source_file']:30s} {flat_decision.soft_score!s:>12s} "
            f"{d3_decision.soft_score!s:>12s} {shift_str:>8s}"
        )


if __name__ == "__main__":
    main()
