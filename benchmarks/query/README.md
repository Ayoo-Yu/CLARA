# Chronological fitted-policy query timing

These fixtures use the v1.1.0 source-prefix fit (zone1, seed0, reference price). Two complete runs retain all 1,920 measured calls. Median single-query time is 6.14 ms for lookup and interval retrieval, or 9.01 ms including raw-width classification. The observed system CPU-load range is 17.0–35.5%; timings describe that measured background load. No observations are discarded.

Validate all 6,600 mappings, 220 choices and 440 interval endpoints without collecting new measurements:

```bash
python benchmarks/query/benchmark_query.py --repository-root .
```

To collect new timings, add `--measure --output outputs/query_new_run`. Results depend on hardware, software and background load. Timings include resident lookup and interval retrieval; they exclude forecasting, candidate construction, full history maintenance, sequential updating, policy fitting and file reads. Batch times measure 220 queries together, not 220 independent request latencies. `metadata.json` records input hashes and each complete measurement run.
