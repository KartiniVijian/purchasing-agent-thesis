"""listing_loader.py - loads product listings from data/product_listings/."""
import json
from pathlib import Path


def load_listing(filepath) -> dict:
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_all_listings(directory) -> list:
    dir_path = Path(directory)
    listings = []
    for json_file in sorted(dir_path.glob("*.json")):
        listing = load_listing(json_file)
        listing["_source_file"] = json_file.name
        listings.append(listing)
    return listings
