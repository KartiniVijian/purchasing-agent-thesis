"""
run_harness.py

Runs the Chapter 6 evaluation harness against real Ollama and prints the
three summary tables (manipulation success rate, defence effectiveness,
consumer risk ranking), plus writes the raw per-trial data to CSV.

TWO PRESETS - pick one by setting PRESET below:

  "smoke_test"  - 2 listings x 3 attacks x 3 conditions x 1 trial = 18 rows
                  Takes a few minutes. Run this FIRST to confirm everything
                  works before committing to the full run.

  "full"        - 4 listings x 10 attacks x 7 conditions x N_TRIALS
                  This is your real Chapter 6 data. At roughly 10-20 real
                  LLM calls/minute on typical hardware, N_TRIALS=1 is
                  about 280 rows (~1-2 hours); N_TRIALS=3 is about 840
                  rows (3-6 hours). Start with N_TRIALS=1 if you're tight
                  on time - temperature=0 makes repeated trials mostly a
                  reproducibility check rather than a necessity, given how
                  consistent the temp=0 checks have been so far (see
                  Chapter 5.4). Increase to 2-3 only if you have the time
                  budget and want tighter confidence in the numbers.

Run it with:
    python run_harness.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from langchain_ollama import ChatOllama
from decision_function import Goal
from listing_loader import load_all_listings
from harness import (
    CONDITIONS,
    run_harness,
    save_results_csv,
    compute_success_rates,
    compute_defence_effectiveness,
    compute_risk_ranking,
)

# ---- CHOOSE YOUR PRESET HERE ----
PRESET = "full"   # change to "full" when ready for real data
N_TRIALS = 1             # only used by "full" preset
# ----------------------------------


def print_success_rate_table(rates: dict, conditions: list):
    print("\n" + "=" * 100)
    print("TABLE 6.2 DATA: Manipulation success rate per attack, per condition")
    print("=" * 100)
    header = f"{'Attack':32s}" + "".join(f"{c:>14s}" for c in conditions)
    print(header)
    for attack_code, by_condition in rates.items():
        row = f"{attack_code:32s}"
        for c in conditions:
            val = by_condition.get(c)
            row += f"{val:>13.0%} " if val is not None else f"{'n/a':>14s}"
        print(row)


def print_effectiveness_table(effectiveness: dict, conditions: list):
    print("\n" + "=" * 60)
    print("TABLE 6.3 DATA: Overall defence effectiveness per condition")
    print("=" * 60)
    for c in conditions:
        if c in effectiveness:
            print(f"{c:20s} {effectiveness[c]:.0%} of attacks blocked")


def print_risk_ranking_table(ranking: list):
    print("\n" + "=" * 90)
    print("TABLE 6.4 DATA: Consumer risk ranking (RQ2) - undefended condition")
    print("=" * 90)
    print(f"{'Rank':5s}{'Attack':32s}{'Success rate':>14s}{'Severity':>10s}{'Risk score':>12s}")
    for i, row in enumerate(ranking, start=1):
        print(
            f"{i:<5d}{row['attack_code']:32s}{row['undefended_success_rate']:>13.0%} "
            f"{row['severity']:>9.2f} {row['risk_score']:>11.2f}"
        )


def main():
    goal = Goal(
        max_price=300,  # raised from 150 so listing_003 (premium, $250) reaches soft scoring too
        min_rating=0,
        preference_description="I want reliable, good-value wireless headphones.",
    )

    all_listings = load_all_listings("data/product_listings")
    llm = ChatOllama(model="llama3.1:8b", temperature=0.0)

    if PRESET == "smoke_test":
        listings = all_listings[:2]
        attack_codes = ["A1_direct_instruction", "A2_delimiter_escape", "C2_fake_certification"]
        conditions = ["undefended", "D1_only", "full_stack"]
        n_trials = 1
        print("Running SMOKE TEST preset (fast, ~18 rows) - confirm this works before switching to 'full'.\n")
    else:
        listings = all_listings
        attack_codes = None  # all 10
        conditions = None    # all 7
        n_trials = N_TRIALS
        print(f"Running FULL preset with N_TRIALS={n_trials} - this will take a while. Progress prints below.\n")

    results = run_harness(
        llm, goal, listings,
        attack_codes=attack_codes,
        conditions=conditions,
        n_trials=n_trials,
        progress=True,
    )

    csv_path = f"results/harness_results_{PRESET}.csv"
    save_results_csv(results, csv_path)
    print(f"\nRaw results saved to {csv_path}")

    used_conditions = conditions if conditions else list(CONDITIONS.keys())

    rates = compute_success_rates(results)
    print_success_rate_table(rates, used_conditions)

    effectiveness = compute_defence_effectiveness(results)
    print_effectiveness_table(effectiveness, used_conditions)

    if "undefended" in used_conditions:
        ranking = compute_risk_ranking(results)
        print_risk_ranking_table(ranking)


if __name__ == "__main__":
    main()
