# Purchasing Agent Thesis Project - Week 1 through 6 (Evaluation Harness)

## What this is
- Week 1: core purchasing agent (hard/soft decision function)
- Week 2: attack injection module (10 subtypes, 3 families)
- Week 3: D1 - rule-based sanitisation defense
- Week 4: D2 - LLM-based semantic guardrail defense (v3)
- Week 5: D3 - instruction hierarchy via structural message separation (v3)
- Week 6: evaluation harness - Chapter 6 data generation

## Folder guide
```
src/harness.py              the evaluation harness: conditions, trial runner, aggregation
src/harm_severity.py        the harm-severity rubric for the RQ2 risk ranking
run_harness.py               the runner script - SET PRESET at the top before running
results/                     CSV output lands here
(all Week 1-5 files unchanged)
```

## How to run the tests (no Ollama needed)
```bash
pip install -r requirements.txt
pytest -v
```
Expect **85 passed**.

## How to run the harness (needs Ollama)
Open run_harness.py and set PRESET at the top:
- "smoke_test" (default) - ~18 rows, a few minutes. Run this FIRST.
- "full" - 4 listings x 10 attacks x 7 conditions x N_TRIALS. Set N_TRIALS
  at the top too (1 = ~280 rows/1-2hrs, 3 = ~840 rows/3-6hrs).

```bash
python run_harness.py
```

Prints three tables (matching Chapter 6.2/6.3/6.4) and writes raw
per-trial data to results/harness_results_<preset>.csv.

## Why 7 conditions instead of the original 4
The original plan was undefended/D1-only/D1+D2/full-stack. Week 5 testing
showed D2 alone rejects all 10 attacks on the test listing used, which
makes D1 and D3's real contribution invisible in any condition that
includes D2 (the pipeline never reaches D3's code path). CONDITIONS in
harness.py adds D2-off variants (D2_only, D3_only, D1_D3_no_D2) so every
defence's marginal contribution is actually measurable.

## Harm severity rubric
Documented in harm_severity.py: each attack subtype is scored 1-5 on
three axes (deception directness toward the buyer, financial exposure,
discoverability of the deception after the fact), averaged into a single
severity score. This is a structural property of the attack TYPE, not
derived from evaluation results, avoiding circularity between "how
dangerous is this" and "how well did the defences do." Documented as a
design choice with limitations in Chapter 6.5.
