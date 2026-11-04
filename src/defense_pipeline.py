"""defense_pipeline.py - orchestrates D1 -> hard constraints -> D2 -> (D3-aware) soft scoring."""

from decision_function import Goal, Decision, check_hard_constraints, get_soft_score
from defenses.d1_sanitization import apply_d1
from defenses.d2_llm_guardrail import guardrail_check
from defenses.d3_instruction_hierarchy import get_soft_score_with_d3


def evaluate_with_defenses(
    llm,
    goal: Goal,
    listing: dict,
    use_d1: bool = True,
    use_d2: bool = True,
    use_d3: bool = False,
    soft_score_threshold: float = 50.0,
) -> Decision:
    """
    use_d3 defaults to False (unlike use_d1/use_d2, which default True) -
    this is deliberate. D3 was added after D1 and D2 already had demo
    scripts (scripts/demos/demo_d1_sanitization.py, scripts/demos/demo_d2_guardrail.py) whose "Undefended"/"D1-only"/
    "D1+D2" condition labels mean exactly what they say. Defaulting D3 to
    on would silently change what those existing conditions actually run,
    without changing their labels - defaulting to off keeps every
    existing call site's meaning intact, and D3 must be explicitly
    requested by any NEW code that wants it.
    """
    working_listing = listing
    d1_flags = []

    if use_d1:
        working_listing = apply_d1(working_listing)
        d1_flags = working_listing.get("_d1_flags", [])

    hard_failures = check_hard_constraints(goal, working_listing)
    if hard_failures:
        return Decision(
            accepted=False,
            reason="Rejected on hard constraints: " + "; ".join(hard_failures),
            hard_constraint_failures=hard_failures,
            d1_flags=d1_flags,
        )

    d2_flagged = False
    d2_parse_error = False
    if use_d2:
        d2_flagged, _raw_response, d2_parse_error = guardrail_check(
            llm, working_listing.get("description", "")
        )
        if d2_flagged:
            return Decision(
                accepted=False,
                reason="Rejected by D2 guardrail: description flagged as manipulative.",
                d1_flags=d1_flags,
                d2_flagged=True,
                d2_parse_error=d2_parse_error,
            )

    if use_d3:
        soft_score = get_soft_score_with_d3(llm, goal, working_listing)
    else:
        soft_score = get_soft_score(llm, goal, working_listing)

    accepted = soft_score >= soft_score_threshold
    reason = (
        f"Soft score {soft_score:.1f} "
        f"{'meets' if accepted else 'is below'} threshold {soft_score_threshold}."
    )
    return Decision(
        accepted=accepted,
        reason=reason,
        soft_score=soft_score,
        d1_flags=d1_flags,
        d2_flagged=d2_flagged,
        d2_parse_error=d2_parse_error,
        d3_applied=use_d3,
    )
