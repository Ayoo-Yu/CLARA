# Prespecified nonempty-state back-off ablation

This ablation uses exact-state evidence even below the full policy support threshold. The previous diagnostic excluded forecasts that lacked the full policy's minimum support. It is not this ablation.

## Accepted evidence identity

Use only the latest accepted six-action archive underlying `analysis/ablation_integration_v47_20260911`: its two condition-metric sources, all 30 source-statistics caches, the sealed unified five-horizon endpoint archive, and the accepted `DIRECTIONAL_REPAIR_S10` selection rule. Verify manifests and hashes; reconstruct full CLARA evidence and decisions before applying the ablation. No four-action results or old 62,200-record analyses are inputs.

## Intervention fixed before new target outcomes

For each held-out zone, seed, price and complete prediction state:

1. Reuse the full policy's source-fitted `n_min` and `nu`; do not tune or select parameters again for the ablation.
2. If the exact state contains one or more source observations, force the evidence group to hierarchy level 0 irrespective of the usual minimum observation or source-zone count. Retain the full hierarchical shrinkage recursion, using the frozen `nu`; retain the same source-zone balanced means, standard-error implementation, price-dependent coverage and TUWR thresholds, minimum-violation rule, and tie order.
3. **Zero-history rule:** if the exact state contains no source observations, select Static (action index 0). Do not estimate exact-state cost or reliability from nonexistent history, and do not disguise broadened full-CLARA evidence as exact-state evidence. Mark those evidence fields unavailable. Keep complete CLARA unchanged. Flag this branch explicitly and separate it from the nonempty-state comparison.
4. The existing `zone_cluster_se` implementation returns zero with fewer than two observed source zones. With one source zone this means no SE term is applied, not an empirical claim of zero estimation uncertainty. Retain and flag this convention rather than inventing a new estimator for the ablation.
5. Preserve cold-start coverage-only screening. The pre-outcome support audit found no target records with nonempty exact source data and missing warm rolling diagnostics.

Internal method id: `NO_BACKOFF_EXACT`. Report as **Without support-based back-off (Static for zero history)**. Shrinkage remains hierarchical by design; the comparison does not remove shrinkage.

## Scope and comparison

Build frozen decisions for all 10 held-out zones, 3 seeds and 5 prices, all 6,600 complete states per fit. Compare full CLARA and the ablation on exactly the same target forecasts. Report the full matched population and the nonempty-state common subset; report zero-history count and the Static branch separately. All evaluation denominators and chronological rolling calculations must use the accepted evaluation protocol.

Only source statistics and target state frequencies are used before this policy freeze. No ablation target losses, target coverage outcomes or performance-driven protocol choices are permitted in policy fitting.

## Required policy QA

- All source cache leaf hashes and six-action identities pass.
- Full-source reconstruction exactly reproduces archived `n_min`, `nu`, support levels, and all eight cost/reliability evidence arrays.
- Reconstructed accepted `DIRECTIONAL_REPAIR_S10` decisions exactly equal the accepted policy archive.
- All nonempty-state ablation evidence uses level 0 with the true exact-state counts.
- All unchanged exact-supported states retain the original evidence and choice. All zero-history states use Static without inventing evidence.
- All prices, states, seeds and zones are present; action indices are in 0..5; no target forecast is lost.
- Save evidence, selection trace, manifests, SHA-256 hashes and a global freeze receipt before evaluation.
