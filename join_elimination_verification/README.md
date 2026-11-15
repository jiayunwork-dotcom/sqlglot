# LEFT JOIN elimination verification

Reproducible checks for the `eliminate_joins` safety fix. Run with the
repository's Python (DuckDB is only used here as an external test oracle, never
imported by sqlglot itself; SQLite is the standard library):

```bash
PYTHONPATH=/workspace python3 join_elimination_verification/pre_fix_repro.py
PYTHONPATH=/workspace python3 join_elimination_verification/engine_multiset_checks.py
PYTHONPATH=/workspace python3 join_elimination_verification/final_collation_matrix.py
PYTHONPATH=/workspace python3 join_elimination_verification/perf_benchmark.py
```

## Scripts

- `pre_fix_repro.py` — the original two bug reports, executed against real
  DuckDB 2.0.0 and stdlib `sqlite3`. It compares column names and full row
  multisets (including multiplicities) of the original SQL and the optimized
  SQL. Against the fixed code the `optimize`/rule-with-schema paths match the
  engine for every case. The remaining `[FAIL]` lines are the *bare*
  `eliminate_joins(parse_one(sql))` calls without dialect/schema: those
  intentionally keep the historical structural behavior, because no type or
  collation evidence is available to reject the dangerous comparison.
- `engine_multiset_checks.py` — 22 query shapes × 3 entry points (bare rule,
  qualified+annotated rule, full pipeline) covering integers, finite floats,
  exact decimals, text, NULLs, empty tables, left duplicates, no-match rows,
  composite keys, `AND`/`OR` predicates, constants, chained joins and all
  SQLite collations. Every expected answer comes from executing the original
  SQL in the engine, never from the optimizer.
- `final_collation_matrix.py` — full 3 (BINARY/NOCASE/RTRIM uniqueness
  collations) × 4 (join collations incl. default) matrix, each cell validated
  against SQLite's actual row multiset.
- `perf_benchmark.py` — wall time, `tracemalloc` peak and the number of unique
  scope analyses for 10/50/100 forwarding layers and 1/10/50 shared-CTE
  references. Shared CTEs are analyzed once; work is linear in forwarding
  depth and reference count, never exponential.
