# AAOIFI Judgmental Dataset Generation Report

Generated: 2025-08-25 14:33:08
Target Size: 3000
Actual Size: 3

## Distribution Analysis
- Arabic Examples: 3 (100.0%)
- English Examples: 0 (0.0%)
- True Verdicts: 2 (66.7%)
- False Verdicts: 1 (33.3%)

## Quality Metrics
- Unknown References: 3 (100.0%)
- Suspected Fabrications: 0

## Target Achievement
- Size Target: ❌ (3/3000)
- Arabic Balance: ✅ (100.0% >= 40%)
- Verdict Balance: ❌ (66.7% in 47-53% range)
- Reference Quality: ❌ (100.0% <= 10%)

## Files Generated
- data/judgmental_final.jsonl: Main dataset
- data/train.jsonl: Training split (80%)
- data/val.jsonl: Validation split (10%)  
- data/test.jsonl: Test split (10%)
- review/human_sample.jsonl: Human review sample

## Generation Log
logs/dataset_generation_20250825_114131.log

Dataset generation completed with issues.
