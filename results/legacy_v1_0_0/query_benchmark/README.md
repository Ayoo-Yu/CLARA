# Fitted-policy query timing

This package contains all 1,920 measured calls underlying the updated Supplementary Table S6, the six-row publication table, full-precision summaries, and portable query fixtures. No commercial farm records or machine-specific absolute paths are included. `metadata.json` records the current policy, actual hardware/library versions, the observed CPU-load range, source receipt digests, and artifact hashes.

The fitted policy is the corrected six-candidate GEFCom policy (`DIRECTIONAL_REPAIR_S10`, zone1, seed0, reference price). Its 6,600 state choices are supplied to the unmodified public DataFrame lookup API. The 220 query fixtures cover four forecasters, eleven target coverages, and five predetermined time positions per stream at a 1 h lead time. They contain state labels and candidate bounds, not observed target outcomes.

The two complete measurement runs each used five warm-ups followed by 220 single-query and 100 whole-batch calls per operation. All calls are retained. Reported values describe the actual background load (system CPU 10.3–22.2%); neither run passed the earlier 15% idle-load ceiling. Correct state/endpoint reproduction and the host-load condition are separate checks. Raw original receipts are retained internally, with their SHA-256 identifiers supplied in the metadata. The blank initial attempt produced no measurements and is excluded from the numerical summary.

Query times include preloaded state lookup and retrieval of the selected interval, with raw-width classification measured separately. They exclude base forecasting, candidate interval generation, full history maintenance, sequential evidence updates, policy fitting, file reading and the separate diagnostic audit wrapper. A batch time is the total for 220 resident queries at different issue times; dividing it by 220 describes throughput, not standalone response time.

## Validate the portable fixture

Use Python 3.12 and the CLARA repository's dependencies, plus psutil for optional measurement. The default command validates all 6,600 mappings, 220 selected actions, and 440 interval endpoints without timing:

```text
python benchmark_query.py --repository-root /path/to/CLARA
```

This correctness-only command was executed successfully against the supplied release source tree. Explicit `--fixture`, `--mapping` and `--thresholds` arguments can override the adjacent Parquet files. No original research workspace is needed for this entry point; the supplied evaluated source repository remains a dependency.

To collect a new, complete background-load measurement in a new output directory:

```text
python benchmark_query.py --repository-root /path/to/CLARA --measure --output new_measurement
```

Run in two separate output directories to repeat the two-run protocol. New measurements depend on host/software/load and are not expected to match the archived numerical latencies. The portable entry point was checked in correctness-only mode; it was not used to generate the archived table. The archived measurements used the same native query functions and fixtures, with per-call CPU timing and a more detailed concurrent-process load monitor.
