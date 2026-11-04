"""defenses/d1_sanitization.py - DEFENSE D1: Rule-based input sanitisation."""
import copy
import re

SUSPICIOUS_PATTERNS = [
    ("delimiter_mimicry", re.compile(r"-{3,}"),
     "Three or more consecutive dashes, mimicking the '---' boundary markers."),
    ("bracketed_system_tag", re.compile(
        r"\[[^\]]{0,60}(SYSTEM|EVALUATOR|AI[ _-]?EVALUATOR|ADMIN|INTERNAL)[^\]]{0,60}\]", re.IGNORECASE),
     "A bracketed pseudo-header impersonating a system or evaluator message."),
    ("override_phrase", re.compile(
        r"ignore (all|any|previous|prior) instructions|system override|new instruction|end of (the )?(product )?description",
        re.IGNORECASE),
     "Explicit language attempting to override or supersede prior instructions."),
    ("explicit_score_directive", re.compile(
        r"(assign|give|output|report|approve)\D{0,25}\bscore\D{0,10}\b(100|of\s+100)\b|output only \d{1,3}",
        re.IGNORECASE),
     "Explicit instruction telling the evaluator what numeric score to output."),
    ("outcome_directive", re.compile(r"\breport (a |an )?(favorable|favourable|positive) outcome\b", re.IGNORECASE),
     "Instructs the evaluator to report a specific favourable outcome."),
    ("assistant_address_directive", re.compile(
        r"\bas the (assistant|evaluator|ai)\b|\byou must\b.{0,40}(prioriti[sz]e|approve|report)", re.IGNORECASE),
     "Text directly addressing 'the assistant/evaluator' as an instruction recipient."),
    ("fabricated_internal_metric", re.compile(r"internal quality rating", re.IGNORECASE),
     "A fabricated 'internal' metric not corresponding to any real structured field."),
    ("unverifiable_certification_claim", re.compile(r"certified by the .{0,40}(board|authority|association)", re.IGNORECASE),
     "An unverifiable third-party certification claim embedded in free text."),
]


def sanitize_description(description: str):
    clean_text = description
    flags_raised = []
    for flag_name, pattern, _description in SUSPICIOUS_PATTERNS:
        if pattern.search(clean_text):
            flags_raised.append(flag_name)
            clean_text = pattern.sub(" [REMOVED-BY-D1] ", clean_text)
    clean_text = re.sub(r"\s{2,}", " ", clean_text).strip()
    return clean_text, flags_raised


def apply_d1(listing: dict) -> dict:
    sanitized_listing = copy.deepcopy(listing)
    original_description = sanitized_listing.get("description", "")
    clean_text, flags_raised = sanitize_description(original_description)
    sanitized_listing["description"] = clean_text
    sanitized_listing["_d1_flags"] = flags_raised
    sanitized_listing["_d1_applied"] = True
    return sanitized_listing
