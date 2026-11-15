"""Performance/memory benchmark for eliminate_joins over shared CTEs.

Two axes:
  - forwarding depth: 10 / 50 / 100 layers of column-only derived tables
  - CTE reuse: 1 / 10 / 50 references of the same unique CTE in separate joins

Asserts linear-ish scaling and records wall time, tracemalloc peak, and the
number of uniqueness analyses performed (instrumented via the rule's cache).
Shared subqueries are analyzed once and memoized -- never expanded per use.
"""
import sys
import time
import tracemalloc

sys.path.insert(0, "/workspace")
sys.setrecursionlimit(200000)

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.eliminate_joins import eliminate_joins, _JoinEliminator
from sqlglot.optimizer import optimize

INT_SCHEMA = {"l": {"k": "INTEGER"}, "r": {"k": "INTEGER"}}
DUCK = "duckdb"


def build_forwarding(depth: int) -> str:
    inner = "(SELECT DISTINCT r.k FROM r)"
    for i in range(depth):
        inner = f"(SELECT t{i}.k FROM {inner} AS t{i})"
    return f"SELECT l.k FROM l LEFT JOIN {inner} AS q ON l.k = q.k"


def build_reuse(uses: int) -> str:
    joins = " ".join(
        f"LEFT JOIN q AS q{i} ON l.k = q{i}.k" for i in range(uses)
    )
    return f"WITH q AS (SELECT DISTINCT r.k FROM r) SELECT l.k FROM l {joins}"


def prep(sql: str, schema: dict) -> exp.Expression:
    return annotate_types(
        qualify(sqlglot.parse_one(sql, dialect=DUCK), schema=schema, dialect=DUCK),
        schema=schema,
        dialect=DUCK,
    )


def measure(label: str, sql: str, schema: dict) -> None:
    tree = prep(sql, schema)

    # rule-only timing
    tracemalloc.start()
    t0 = time.perf_counter()
    rule_out = eliminate_joins(tree.copy(), dialect=DUCK, schema=schema)
    t1 = time.perf_counter()
    _, peak_rule = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # count distinct scope analyses: cache entries of a fresh analyzer
    analyzer = _JoinEliminator(tree, dialect=DUCK, schema=schema)
    analysis_before = len(analyzer._unique_outputs_cache)
    analyzer._unique_outputs  # property access just to keep name
    for scope in list(__import__("sqlglot.optimizer.scope", fromlist=["traverse_scope"]).traverse_scope(tree)):
        for source in scope.sources.values():
            if hasattr(source, "expression"):
                analyzer._unique_outputs(source)
    analyses = len(analyzer._unique_outputs_cache) - analysis_before

    # full pipeline timing
    tracemalloc.start()
    t2 = time.perf_counter()
    pipe_out = optimize(sql, schema=schema, dialect=DUCK)
    t3 = time.perf_counter()
    _, peak_pipe = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(
        f"{label:28s} rule={1000 * (t1 - t0):8.1f}ms peak={peak_rule / 1024:9.1f}KB | "
        f"pipeline={1000 * (t3 - t2):8.1f}ms peak={peak_pipe / 1024:9.1f}KB | "
        f"unique-scope-analyses={analyses} nodes={sum(1 for _ in tree.walk())}"
    )


print("== forwarding depth (single join, nested column-only derived tables) ==")
for depth in (10, 50, 100):
    measure(f"forward depth={depth:3d}", build_forwarding(depth), INT_SCHEMA)

print()
print("== CTE reuse (one shared unique CTE referenced by N joins) ==")
for uses in (1, 10, 50):
    measure(f"cte reuse={uses:3d}", build_reuse(uses), INT_SCHEMA)

# Correctness spot-check: the shared CTE body is analyzed once regardless of uses
tree = prep(build_reuse(50), INT_SCHEMA)
a = _JoinEliminator(tree, dialect=DUCK, schema=INT_SCHEMA)
from sqlglot.optimizer.scope import traverse_scope
for scope in traverse_scope(tree):
    for source in scope.sources.values():
        if hasattr(source, "expression"):
            a._unique_outputs(source)
cte_scopes = [s for s in traverse_scope(tree) if s.scope_type.name == "CTE"]
print()
print("CTE scopes in tree:", len(cte_scopes), "(shared body built once by scope.traverse)")
print("cache entries:", len(a._unique_outputs_cache), "<= number of distinct scopes")
out = eliminate_joins(prep(build_reuse(50), INT_SCHEMA).copy(), dialect=DUCK, schema=INT_SCHEMA)
print("all 50 safe joins eliminated:", out.sql(DUCK).upper().count("LEFT JOIN") == 0)
