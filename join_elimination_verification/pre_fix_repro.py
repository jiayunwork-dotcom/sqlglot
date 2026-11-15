"""Pre-fix reproduction of the unsafe LEFT JOIN elimination (run BEFORE the fix).

Uses real engines (DuckDB 2.0.0 and stdlib sqlite3) to compare the full row
multiset + column names of the original SQL against the eliminate_joins() /
optimize() output. Expected answers are produced by the engines themselves,
never by the optimizer under test.
"""
import collections
import sqlite3
import sys

import duckdb

sys.path.insert(0, "/workspace")
import sqlglot
from sqlglot.optimizer import optimize
from sqlglot.optimizer.eliminate_joins import eliminate_joins

FAILURES = []


def multiset(rows, cols):
    return cols, collections.Counter(tuple(r) for r in rows)


def run_duckdb(sql, schema_sql):
    con = duckdb.connect()
    for stmt in schema_sql:
        con.execute(stmt)
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return multiset(cur.fetchall(), cols)


def run_sqlite(sql, schema_sql):
    con = sqlite3.connect(":memory:")
    con.executescript(";\n".join(schema_sql))
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return multiset(cur.fetchall(), cols)


def check(name, engine, original_sql, optimized_sql, schema_sql, dialect):
    runner = run_duckdb if engine == "duckdb" else run_sqlite
    expected = runner(original_sql, schema_sql)
    try:
        actual = runner(optimized_sql, schema_sql)
    except Exception as exc:  # optimized SQL must still be executable
        FAILURES.append((name, f"optimized SQL failed to execute: {exc!r}"))
        print(f"[FAIL] {name}: optimized SQL error {exc!r}")
        return
    if actual != expected:
        FAILURES.append((name, (expected, actual)))
        print(f"[FAIL] {name}\n  expected={expected}\n  actual  ={actual}")
    else:
        print(f"[OK] {name} (rows={sum(expected[1].values())})")


def rule_output(sql, dialect):
    return eliminate_joins(sqlglot.parse_one(sql, dialect=dialect).copy()).sql(dialect=dialect)


def pipeline_output(sql, dialect, schema):
    return optimize(sql, schema=schema, dialect=dialect).sql(dialect=dialect)


# ---------------------------------------------------------------- DuckDB case
ddb_schema = [
    "CREATE TABLE l(k INTEGER); INSERT INTO l VALUES (1),(2)",
    "CREATE TABLE r(k VARCHAR); INSERT INTO r VALUES ('1'),('01'),('2')",
]
ddb_sql = "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k = q.k"
ddb_opt_rule = rule_output(ddb_sql, "duckdb")
ddb_opt_pipe = pipeline_output(
    ddb_sql, "duckdb", {"l": {"k": "INTEGER"}, "r": {"k": "VARCHAR"}}
)
print("duckdb rule output    :", ddb_opt_rule)
print("duckdb pipeline output:", ddb_opt_pipe)
check("duckdb/rule/int-vs-varchar", "duckdb", ddb_sql, ddb_opt_rule, ddb_schema, "duckdb")
check("duckdb/pipeline/int-vs-varchar", "duckdb", ddb_sql, ddb_opt_pipe, ddb_schema, "duckdb")

# ---------------------------------------------------------------- SQLite cases
sqlite_schema = [
    "CREATE TABLE l(k TEXT)", "INSERT INTO l VALUES ('a')",
    "CREATE TABLE r(k TEXT)", "INSERT INTO r VALUES ('a'),('A')",
]
sqlite_cases = {
    "distinct": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k COLLATE NOCASE = q.k",
    "groupby": "SELECT l.k FROM l LEFT JOIN (SELECT r.k FROM r GROUP BY r.k) q ON l.k COLLATE NOCASE = q.k",
    "cte": "WITH q AS (SELECT DISTINCT r.k FROM r) SELECT l.k FROM l LEFT JOIN q ON l.k COLLATE NOCASE = q.k",
}
for label, sql in sqlite_cases.items():
    opt = rule_output(sql, "sqlite")
    print(f"sqlite {label} output:", opt)
    check(f"sqlite/rule/nocase-{label}", "sqlite", sql, opt, sqlite_schema, "sqlite")

print()
if FAILURES:
    print(f"TOTAL FAILURES: {len(FAILURES)}")
    sys.exit(1)
print("all multisets matched")
