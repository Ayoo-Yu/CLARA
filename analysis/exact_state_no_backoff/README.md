# Exact-state ablation of support-based back-off

The intervention in PROTOCOL.md uses exact historical groups whenever nonempty,
even below normal observation/zone support thresholds. Zero-history states use
Static. All other fitted choices, shrinkage recursion, screening and tie rules
are unchanged. The source snapshot contains the five evaluated Python scripts
without numerical changes; SOURCE_HASHES.json identifies them.

## Portable result verification

Run from the repository root:

    python analysis/exact_state_no_backoff/verify_results.py

This recomputes paired summaries and bootstrap intervals from the included
condition-level Parquet tables using the evaluated aggregation function. It
checks same-event denominators and the CSV results; it does not retrain a model
or replay event-level candidates. Requires numpy, pandas and pyarrow.

## Full policy-build/replay dependency boundary

The source snapshot is archival code, not a self-contained runner. Its expected
root is the evaluated analysis tree containing complete_ablation_20260909,
reviewer_closure_20260905/policy/build_policy_variants.py,
guardrail_redesign_gefcom_20260908, and ablation_integration_v47_20260911.
It additionally requires all 30 source-statistics caches, the frozen six-action
candidate/endpoint archive, 150 source-fitted policy receipts and their hashes.
Some runtime modules are included under src/evaluated; the large caches and
receipts are not supplied by the small result tables.

To rebuild, restore the matched evaluated tree and caches, place the snapshot
as an analysis directory two levels below that tree root, and execute in order:
audit_support, build_policies, qa_policies, evaluate_exact, analyze_exact.
The original freeze checks require the original PLAN.md with its accepted hash;
PROTOCOL.md here is an editorial description, not a replacement hash receipt.
Do not bypass cache/hash assertions or treat the portable summary check as
validation of a new policy fit. A new source/plan configuration requires fresh
fit and evaluation receipts.
