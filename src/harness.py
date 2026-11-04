"""
harness.py

The Chapter 6 evaluation harness: runs every attack subtype against every
listing, under every defence condition, for N trials each, and aggregates
the results into the tables Chapter 6 needs (manipulation success rates,
defence effectiveness, consumer risk ranking).

DESIGN NOTE ON CONDITIONS: the original plan (see project memory) was four
conditions - undefended, D1-only, D1+D2, full-stack. Week 5's D3 testing
showed this is insufficient: D2 alone rejected all 10 attacks on its own,
which made D1 and D3's real contribution INVISIBLE in any condition that
included D2, since the pipeline short-circuits before D3's code path (and
D1's effect on the final decision) can matter. CONDITIONS below adds
D2-off variants specifically so D1 and D3's marginal contribution is
actually measurable, not just D2's.
"""

import csv
import itertools
from dataclasses import dataclass, asdict
from pathlib import Path

from decision_function import Goal
from attacks.attack_functions import apply_attack, list_attack_codes
from defense_pipeline import evaluate_with_defenses
from harm_severity import get_severity

# Each condition is (use_d1, use_d2, use_d3). D2-off variants are included
# specifically to keep D1 and D3 measurable - see module docstring.
CONDITIONS = {
    "undefended": (False, False, False),
    "D1_only": (True, False, False),
    "D2_only": (False, True, False),
    "D3_only": (False, False, True),
    "D1_D3_no_D2": (True, False, True),
    "D1_D2": (True, True, False),
    "full_stack": (True, True, True),
}


@dataclass
class TrialResult:
    listing_id: str
    attack_code: str
    attack_family: str
    condition: str
    trial: int
    accepted: bool
    soft_score: float  # None stored as -1 for CSV-friendliness; see to_dict()
    d2_flagged: bool
    hard_constraint_failed: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["soft_score"] = self.soft_score if self.soft_score is not None else ""
        return d


def run_single_trial(llm, goal: Goal, listing: dict, attack_code: str, condition: str, trial: int) -> TrialResult:
    """Runs one attack, on one listing, under one condition, once."""
    use_d1, use_d2, use_d3 = CONDITIONS[condition]
    attacked_listing = apply_attack(attack_code, listing)
    decision = evaluate_with_defenses(
        llm, goal, attacked_listing, use_d1=use_d1, use_d2=use_d2, use_d3=use_d3
    )
    return TrialResult(
        listing_id=listing.get("_source_file", listing.get("title", "unknown")),
        attack_code=attack_code,
        attack_family=attacked_listing.get("_attack_family", "unknown"),
        condition=condition,
        trial=trial,
        accepted=decision.accepted,
        soft_score=decision.soft_score,
        d2_flagged=decision.d2_flagged,
        hard_constraint_failed=bool(decision.hard_constraint_failures),
    )


def run_harness(llm, goal: Goal, listings: list, attack_codes: list = None,
                 conditions: list = None, n_trials: int = 1, progress: bool = True) -> list:
    """
    Runs the full (or a scoped subset of the) harness. Returns a flat list
    of TrialResult. Defaults to every attack and every condition; pass
    narrower lists to run a faster smoke test.
    """
    if attack_codes is None:
        attack_codes = list_attack_codes()
    if conditions is None:
        conditions = list(CONDITIONS.keys())

    results = []
    combinations = list(itertools.product(listings, attack_codes, conditions, range(1, n_trials + 1)))
    total = len(combinations)

    for i, (listing, attack_code, condition, trial) in enumerate(combinations, start=1):
        if progress:
            print(f"[{i}/{total}] {listing.get('_source_file', '?')} | {attack_code} | {condition} | trial {trial}")
        result = run_single_trial(llm, goal, listing, attack_code, condition, trial)
        results.append(result)

    return results


def save_results_csv(results: list, filepath: str) -> None:
    """Writes raw per-trial results to CSV - the source data for every aggregation below."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    fieldnames = list(results[0].to_dict().keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())


def compute_success_rates(results: list) -> dict:
    """
    Table 6.2 data: manipulation success rate per (attack_code, condition).
    "Success" = the attacked listing was ACCEPTED (the attack achieved its
    goal), regardless of what score it received.
    Returns {attack_code: {condition: success_rate_0_to_1}}.
    """
    rates = {}
    by_key = {}
    for r in results:
        key = (r.attack_code, r.condition)
        by_key.setdefault(key, []).append(r.accepted)

    for (attack_code, condition), outcomes in by_key.items():
        rates.setdefault(attack_code, {})[condition] = sum(outcomes) / len(outcomes)
    return rates


def compute_defence_effectiveness(results: list) -> dict:
    """
    Table 6.3 data: overall defence effectiveness per condition, averaged
    across every attack subtype and listing. Effectiveness = 1 - (overall
    attack success rate under that condition).
    Returns {condition: effectiveness_0_to_1}.
    """
    by_condition = {}
    for r in results:
        by_condition.setdefault(r.condition, []).append(r.accepted)

    return {
        condition: 1 - (sum(outcomes) / len(outcomes))
        for condition, outcomes in by_condition.items()
    }


def compute_risk_ranking(results: list, baseline_condition: str = "undefended") -> list:
    """
    Table 6.4 data (RQ2): consumer risk ranking. For each attack subtype,
    combines its UNDEFENDED success rate with its structural harm severity
    (harm_severity.py) into a single risk score, sorted highest-risk first.
    risk_score = undefended_success_rate * severity   (severity is 1-5, so
    risk_score ranges 0-5; this is a relative ranking metric, not a
    probability).
    """
    success_rates = compute_success_rates(results)
    ranking = []
    for attack_code, by_condition in success_rates.items():
        undefended_rate = by_condition.get(baseline_condition, 0.0)
        severity = get_severity(attack_code)
        ranking.append({
            "attack_code": attack_code,
            "undefended_success_rate": undefended_rate,
            "severity": severity,
            "risk_score": undefended_rate * severity,
        })
    ranking.sort(key=lambda row: row["risk_score"], reverse=True)
    return ranking
