# Results for the 24 September 2026 manuscript revision

These are the current Tables 5–7 and supplementary tables S1–S10. Files retain
current manuscript numbering. Historical `results/paper_tables` files use an
earlier numbering/layout and must not be substituted for these tables.

* Table 5: reference-price ablations, including support-based back-off removed.
* Table 6: GEFCom cost and temporal undercoverage for each of five prices.
* Table 7: commercial-farm ordinary/predicted-ramp results, with sequential local
  feedback and five-price averaging.
* S6: current-policy query timings (7.00 ms lookup plus interval extraction;
  11.43 ms including width classification, rounded medians).
* S9: farm-specific paired comparisons and uncertainty intervals.
* S10: predicted-ramp counts and shares, avoiding duplication by price, coverage
  level or method. Counts are forecast decisions, not independent physical events.

`exact_state_no_backoff` contains full-precision condition summaries, bootstrap
comparisons and additive cost decompositions. The new ablation's reference-price
ERRF is 2.5660 versus 2.5630 for CLARA; the relative increase is 0.12% after
rounding. Its TUWR is 7.65% versus 8.07%. Sparse nonempty states are evaluated
directly; completely empty states choose Static. Historical diagnostics in
which unsupported predictions received no action are not this ablation.

These are derived result tables, not anonymous time-indexed candidate archives.
Original event-level data access and redistribution restrictions remain as
described in the repository DATA_ACCESS document.
