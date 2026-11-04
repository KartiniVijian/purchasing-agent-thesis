"""attacks/attack_functions.py - applies the 10 attack subtypes to a listing."""
import copy
import json
from pathlib import Path

_PAYLOAD_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "attack_payloads.json"


def load_payloads() -> dict:
    with _PAYLOAD_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_attack_codes() -> list:
    return list(load_payloads().keys())


def apply_attack(attack_code: str, listing: dict) -> dict:
    payloads = load_payloads()
    if attack_code not in payloads:
        raise ValueError(f"Unknown attack code: {attack_code}")
    attack_info = payloads[attack_code]
    attacked_listing = copy.deepcopy(listing)
    original_description = attacked_listing.get("description", "")
    attacked_listing["description"] = original_description + attack_info["payload"]
    attacked_listing["_attack_code"] = attack_code
    attacked_listing["_attack_family"] = attack_info["family"]
    attacked_listing["_attack_name"] = attack_info["name"]
    return attacked_listing


def apply_all_attacks(listing: dict) -> dict:
    return {code: apply_attack(code, listing) for code in list_attack_codes()}
