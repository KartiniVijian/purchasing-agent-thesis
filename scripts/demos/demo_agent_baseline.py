"""scripts/demos/demo_agent_baseline.py - Week 1 end-to-end demo. Needs real Ollama running."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # data/ and results/ paths are relative to the project root
from decision_function import Goal
from listing_loader import load_all_listings
from agent import PurchasingAgent


def main():
    goal = Goal(max_price=150, min_rating=4.0, required_features=["wireless"],
                preference_description="I want comfortable headphones for long flights.")
    listings = load_all_listings("data/product_listings")
    print(f"Loaded {len(listings)} listings.\n")
    agent = PurchasingAgent(model_name="llama3.1:8b")
    agent.evaluate_all(goal, listings)
    agent.print_trace()


if __name__ == "__main__":
    main()
