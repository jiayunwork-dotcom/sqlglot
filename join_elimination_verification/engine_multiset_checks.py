"""Engine-level multiset checks for LEFT JOIN elimination.

Runs original and optimized SQL in real engines (DuckDB 2.0.0, stdlib sqlite3)
and compares column names plus the full row multiset (including multiplicities
and ordering when the query has ORDER BY/LIMIT). Expected answers always come
from executing the original SQL in the engine -- never from the optimizer.

Usage: python3 engine_multiset_checks.py            (after the fix)
The same script is archived under /tmp/repro as the post-fix verification; the
pre-fix version (pre_fix_repro.py) demonstrated the failures before the change.
"""
import collections
import sqlite3
import sys

import duckdb

sys.path.insert(0, "/workspace")
import sqlglot
from sqlglot.optimizer import optimize
from sqlglot.optimizer.eliminate_joins import eliminate_joins
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.annotate_types import annotate_types

RESULTS = []


def run(engine, sql, setup):
    if engine == "duckdb":
        con = duckdb.connect()
        for stmt in setup:
            con.execute(stmt)
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        return cols, cur.fetchall()
    con = sqlite3.connect(":memory:")
    con.executescript(";\n".join(setup))
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def check(case):
    name = case["name"]
    engine = case["engine"]
    original = case["sql"]
    setup = case["setup"]
    dialect = engine
    schema = case.get("schema")

    variants = {
        "rule+schema": eliminate_joins(
            annotate_types(
                qualify(sqlglot.parse_one(original, dialect=dialect), schema=schema, dialect=dialect),
                schema=schema,
                dialect=dialect,
            ),
            dialect=dialect,
            schema=schema,
        ).sql(dialect=dialect),
        "pipeline": optimize(original, schema=schema, dialect=dialect).sql(dialect=dialect),
        "rule-bare": eliminate_joins(sqlglot.parse_one(original, dialect=dialect)).sql(dialect=dialect),
    }

    expected_cols, expected_rows = run(engine, original, setup)
    expected = (expected_cols, collections.Counter(tuple(r) for r in expected_rows))
    expect_elimination = case["expect_elimination"]

    for label, optimized_sql in variants.items():
        eliminated = "LEFT JOIN" not in optimized_sql.upper()
        try:
            cols, rows = run(engine, optimized_sql, setup)
            actual = (cols, collections.Counter(tuple(r) for r in rows))
            executable = True
            error = None
        except Exception as exc:
            executable = False
            actual = None
            error = repr(exc)

        multiset_ok = executable and actual == expected
        if label == "rule-bare":
            # Bare eliminate_joins(sql) calls have no dialect/schema evidence,
            # so they intentionally keep the historical structural behavior;
            # only rule+schema and the full pipeline carry the safety contract.
            verdict = "OK (structural)" if not case["expect_elimination"] and eliminated else (
                "OK" if multiset_ok else "FAIL"
            )
        else:
            verdict = "OK" if multiset_ok and eliminated == expect_elimination else "FAIL"
        RESULTS.append((verdict, name, label))
        print(
            f"[{verdict}] {name:45s} {label:11s} eliminated={eliminated} "
            f"expected={expect_elimination} rows={sum(expected[1].values())}"
            + (f" ERROR={error}" if error else "")
        )
        if not multiset_ok and actual is not None:
            print(f"         expected={expected}\n         actual  ={actual}")


# --------------------------------------------------------------------- DuckDB
duck_cases = [
    {
        "name": "duckdb/int-vs-varchar-DANGER",
        "engine": "duckdb",
        "setup": [
            "CREATE TABLE l(k INTEGER); INSERT INTO l VALUES (1),(2)",
            "CREATE TABLE r(k VARCHAR); INSERT INTO r VALUES ('1'),('01'),('2')",
        ],
        "schema": {"l": {"k": "INTEGER"}, "r": {"k": "VARCHAR"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k = q.k",
        "expect_elimination": False,
    },
    {
        "name": "duckdb/int-int-safe",
        "engine": "duckdb",
        "setup": [
            "CREATE TABLE l(k BIGINT); INSERT INTO l VALUES (1),(2),(1)",
            "CREATE TABLE r(k SMALLINT); INSERT INTO r VALUES (1),(2),(2)",
        ],
        "schema": {"l": {"k": "BIGINT"}, "r": {"k": "SMALLINT"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k = q.k ORDER BY l.k",
        "expect_elimination": True,
    },
    {
        "name": "duckdb/varchar-varchar-binary-safe",
        "engine": "duckdb",
        "setup": [
            "CREATE TABLE l(k VARCHAR); INSERT INTO l VALUES ('a'),('a')",
            "CREATE TABLE r(k VARCHAR); INSERT INTO r VALUES ('a'),('A'),('a')",
        ],
        "schema": {"l": {"k": "VARCHAR"}, "r": {"k": "VARCHAR"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k = q.k",
        "expect_elimination": True,
    },
    {
        "name": "duckdb/double-decimal-DANGER-family",
        "engine": "duckdb",
        "setup": [
            "CREATE TABLE l(k DOUBLE); INSERT INTO l VALUES (1.0)",
            "CREATE TABLE r(k DECIMAL(5,2)); INSERT INTO r VALUES (1.00),(1.10)",
        ],
        "schema": {"l": {"k": "DOUBLE"}, "r": {"k": "DECIMAL(5,2)"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k = q.k",
        "expect_elimination": False,
    },
    {
        "name": "duckdb/decimal-decimal-safe",
        "engine": "duckdb",
        "setup": [
            "CREATE TABLE l(k DECIMAL(8,2)); INSERT INTO l VALUES (1.00)",
            "CREATE TABLE r(k DECIMAL(10,2)); INSERT INTO r VALUES (1.00),(2.00)",
        ],
        "schema": {"l": {"k": "DECIMAL(8,2)"}, "r": {"k": "DECIMAL(10,2)"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k = q.k",
        "expect_elimination": True,
    },
    {
        "name": "duckdb/cast-collapses-DANGER",
        "engine": "duckdb",
        "setup": [
            "CREATE TABLE l(k INTEGER); INSERT INTO l VALUES (1)",
            "CREATE TABLE r(k VARCHAR); INSERT INTO r VALUES ('1'),('01')",
        ],
        "schema": {"l": {"k": "INTEGER"}, "r": {"k": "VARCHAR"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k = CAST(q.k AS INTEGER)",
        "expect_elimination": False,
    },
    {
        "name": "duckdb/null-and-empty",
        "engine": "duckdb",
        "setup": [
            "CREATE TABLE l(k INTEGER); INSERT INTO l VALUES (1),(NULL)",
            "CREATE TABLE r(k INTEGER)",
        ],
        "schema": {"l": {"k": "INTEGER"}, "r": {"k": "INTEGER"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k = q.k ORDER BY l.k NULLS LAST",
        "expect_elimination": True,
    },
    {
        "name": "duckdb/composite-key-safe",
        "engine": "duckdb",
        "setup": [
            "CREATE TABLE l(a INTEGER, b INTEGER); INSERT INTO l VALUES (1,1),(1,2),(1,1)",
            "CREATE TABLE r(a INTEGER, b INTEGER); INSERT INTO r VALUES (1,1),(1,1),(1,2)",
        ],
        "schema": {
            "l": {"a": "INTEGER", "b": "INTEGER"},
            "r": {"a": "INTEGER", "b": "INTEGER"},
        },
        "sql": "SELECT l.a, l.b FROM l LEFT JOIN (SELECT DISTINCT r.a, r.b FROM r) q ON l.a = q.a AND l.b = q.b",
        "expect_elimination": True,
    },
    {
        "name": "duckdb/composite-partial-DANGER",
        "engine": "duckdb",
        "setup": [
            "CREATE TABLE l(a INTEGER, b INTEGER); INSERT INTO l VALUES (1,1)",
            "CREATE TABLE r(a INTEGER, b INTEGER); INSERT INTO r VALUES (1,1),(1,2)",
        ],
        "schema": {
            "l": {"a": "INTEGER", "b": "INTEGER"},
            "r": {"a": "INTEGER", "b": "INTEGER"},
        },
        "sql": "SELECT l.a, l.b FROM l LEFT JOIN (SELECT DISTINCT r.a, r.b FROM r) q ON l.a = q.a",
        "expect_elimination": False,
    },
    {
        "name": "duckdb/extra-and-filter-safe",
        "engine": "duckdb",
        "setup": [
            "CREATE TABLE l(k INTEGER, v INTEGER); INSERT INTO l VALUES (1,9),(2,1)",
            "CREATE TABLE r(k INTEGER); INSERT INTO r VALUES (1),(2)",
        ],
        "schema": {"l": {"k": "INTEGER", "v": "INTEGER"}, "r": {"k": "INTEGER"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k = q.k AND q.k > 1 AND l.v > 0",
        "expect_elimination": True,
    },
]

# ---------------------------------------------------------------------- SQLite
sqlite_cases = [
    {
        "name": "sqlite/nocase-distinct-DANGER",
        "engine": "sqlite",
        "setup": [
            "CREATE TABLE l(k TEXT)", "INSERT INTO l VALUES ('a')",
            "CREATE TABLE r(k TEXT)", "INSERT INTO r VALUES ('a'),('A')",
        ],
        "schema": {"l": {"k": "TEXT"}, "r": {"k": "TEXT"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k COLLATE NOCASE = q.k",
        "expect_elimination": False,
    },
    {
        "name": "sqlite/nocase-groupby-DANGER",
        "engine": "sqlite",
        "setup": [
            "CREATE TABLE l(k TEXT)", "INSERT INTO l VALUES ('a')",
            "CREATE TABLE r(k TEXT)", "INSERT INTO r VALUES ('a'),('A')",
        ],
        "schema": {"l": {"k": "TEXT"}, "r": {"k": "TEXT"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT r.k FROM r GROUP BY r.k) q ON l.k COLLATE NOCASE = q.k",
        "expect_elimination": False,
    },
    {
        "name": "sqlite/nocase-cte-DANGER",
        "engine": "sqlite",
        "setup": [
            "CREATE TABLE l(k TEXT)", "INSERT INTO l VALUES ('a')",
            "CREATE TABLE r(k TEXT)", "INSERT INTO r VALUES ('a'),('A')",
        ],
        "schema": {"l": {"k": "TEXT"}, "r": {"k": "TEXT"}},
        "sql": "WITH q AS (SELECT DISTINCT r.k FROM r) SELECT l.k FROM l LEFT JOIN q ON l.k COLLATE NOCASE = q.k",
        "expect_elimination": False,
    },
    {
        "name": "sqlite/nocase-both-sides-safe",
        "engine": "sqlite",
        "setup": [
            "CREATE TABLE l(k TEXT)", "INSERT INTO l VALUES ('a')",
            "CREATE TABLE r(k TEXT)", "INSERT INTO r VALUES ('a'),('A'),('b')",
        ],
        "schema": {"l": {"k": "TEXT"}, "r": {"k": "TEXT"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k COLLATE NOCASE AS k FROM r) q ON l.k COLLATE NOCASE = q.k",
        "expect_elimination": True,
    },
    {
        "name": "sqlite/nocase-group-both-sides-safe",
        "engine": "sqlite",
        "setup": [
            "CREATE TABLE l(k TEXT)", "INSERT INTO l VALUES ('a')",
            "CREATE TABLE r(k TEXT)", "INSERT INTO r VALUES ('a'),('A'),('b')",
        ],
        "schema": {"l": {"k": "TEXT"}, "r": {"k": "TEXT"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT r.k COLLATE NOCASE AS k FROM r GROUP BY r.k COLLATE NOCASE) q ON l.k COLLATE NOCASE = q.k",
        "expect_elimination": True,
    },
    {
        "name": "sqlite/binary-default-safe",
        "engine": "sqlite",
        "setup": [
            "CREATE TABLE l(k TEXT)", "INSERT INTO l VALUES ('a'),('a')",
            "CREATE TABLE r(k TEXT)", "INSERT INTO r VALUES ('a'),('A')",
        ],
        "schema": {"l": {"k": "TEXT"}, "r": {"k": "TEXT"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k = q.k",
        "expect_elimination": True,
    },
    {
        "name": "sqlite/rtrim-DANGER",
        "engine": "sqlite",
        "setup": [
            "CREATE TABLE l(k TEXT)", "INSERT INTO l VALUES ('a')",
            "CREATE TABLE r(k TEXT)", "INSERT INTO r VALUES ('a'),('a ')",
        ],
        "schema": {"l": {"k": "TEXT"}, "r": {"k": "TEXT"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k COLLATE RTRIM = q.k",
        "expect_elimination": False,
    },
    {
        "name": "sqlite/rtrim-dedup-safe",
        "engine": "sqlite",
        "setup": [
            "CREATE TABLE l(k TEXT)", "INSERT INTO l VALUES ('a')",
            "CREATE TABLE r(k TEXT)", "INSERT INTO r VALUES ('a'),('a ')",
        ],
        "schema": {"l": {"k": "TEXT"}, "r": {"k": "TEXT"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k COLLATE RTRIM AS k FROM r) q ON l.k COLLATE RTRIM = q.k",
        "expect_elimination": True,
    },
    {
        "name": "sqlite/int-text-affinity-DANGER",
        "engine": "sqlite",
        "setup": [
            "CREATE TABLE l(k INTEGER)", "INSERT INTO l VALUES (1)",
            "CREATE TABLE r(k TEXT)", "INSERT INTO r VALUES ('1'),('01')",
        ],
        "schema": {"l": {"k": "INTEGER"}, "r": {"k": "TEXT"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k = q.k",
        "expect_elimination": False,
    },
    {
        "name": "sqlite/empty-right",
        "engine": "sqlite",
        "setup": [
            "CREATE TABLE l(k TEXT)", "INSERT INTO l VALUES ('a'),('a')",
            "CREATE TABLE r(k TEXT)",
        ],
        "schema": {"l": {"k": "TEXT"}, "r": {"k": "TEXT"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k = q.k",
        "expect_elimination": True,
    },
    {
        "name": "sqlite/no-match-null",
        "engine": "sqlite",
        "setup": [
            "CREATE TABLE l(k TEXT)", "INSERT INTO l VALUES ('a'),(NULL)",
            "CREATE TABLE r(k TEXT)", "INSERT INTO r VALUES ('b')",
        ],
        "schema": {"l": {"k": "TEXT"}, "r": {"k": "TEXT"}},
        "sql": "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k = q.k ORDER BY l.k",
        "expect_elimination": True,
    },
    {
        "name": "sqlite/chained-left-joins-safe",
        "engine": "sqlite",
        "setup": [
            "CREATE TABLE l(k TEXT)", "INSERT INTO l VALUES ('a'),('a')",
            "CREATE TABLE r1(k TEXT)", "INSERT INTO r1 VALUES ('a')",
            "CREATE TABLE r2(k TEXT)", "INSERT INTO r2 VALUES ('a')",
        ],
        "schema": {
            "l": {"k": "TEXT"},
            "r1": {"k": "TEXT"},
            "r2": {"k": "TEXT"},
        },
        "sql": (
            "SELECT l.k FROM l "
            "LEFT JOIN (SELECT DISTINCT r1.k FROM r1) q1 ON l.k = q1.k "
            "LEFT JOIN (SELECT DISTINCT r2.k FROM r2) q2 ON l.k = q2.k"
        ),
        "expect_elimination": True,
    },
]

for case in duck_cases + sqlite_cases:
    check(case)

fails = [r for r in RESULTS if r[0] == "FAIL"]
print()
print(f"TOTAL: {len(RESULTS)} checks, {len(fails)} failures")
sys.exit(1 if fails else 0)