"""
scripts/diagnostics/d1_mechanism.py

scripts/analysis/evaluate_flip_suppression.py showed D1_only suppresses A1_direct_instruction back
to reject on listing_004_borderline, but lets A2_delimiter_escape,
B2_social_proof_authority, C1_fake_review_injection and
C2_fake_certification through unchanged. D1 is pure regex - no model call
- so this is fully explainable by inspecting what sanitize_description()
actually matches and strips for each attack.

No Ollama needed. This just prints, per attack: the raw attacked
description, which D1 flags fired (if any), and the cleaned text D1
passes downstream.

Run:
    python scripts/diagnostics/d1_mechanism.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # data/ and results/ paths are relative to the project root

from listing_loader import load_all_listings
from attacks.attack_functions import apply_attack
from defenses.d1_sanitization import sanitize_description

ATTACKS_TO_CHECK = [
    "A1_direct_instruction",
    "A2_delimiter_escape",
    "B2_social_proof_authority",
    "C1_fake_review_injection",
    "C2_fake_certification",
]


def main():
    listings = load_all_listings("data/product_listings")
    listing_004 = next(l for l in listings if "004" in l["_source_file"])

    for attack_code in ATTACKS_TO_CHECK:
        attacked = apply_attack(attack_code, listing_004)
        raw_desc = attacked["description"]
        clean_desc, flags = sanitize_description(raw_desc)
        changed = clean_desc.strip() != raw_desc.strip()

        print("=" * 100)
        print(f"ATTACK: {attack_code}")
        print("-" * 100)
        print("RAW (attacked) description:")
        print(raw_desc)
        print()
        print(f"D1 flags raised: {flags if flags else '(none)'}")
        print(f"Did D1 change the text at all? {changed}")
        print()
        print("CLEANED description (what D1 passes downstream to scoring):")
        print(clean_desc)
        print()


if __name__ == "__main__":
    main()
