# LEFT JOIN Elimination: Safety Basis and Conservative Boundaries

This document explains when `eliminate_joins` may remove a `LEFT JOIN` whose
right-hand outputs are never referenced outside its own `ON` clause, and why
the extra checks were added.

## The failure being fixed

Removing such a join is only correct when every left row matches **at most
one** right row. The old rule proved this purely structurally: if the right
query was `DISTINCT` (or grouped, or returned at most one row) and the join
predicate mentioned every unique output, the join was dropped. That ignores
the *comparison semantics* of the `ON` predicate, which can identify distinct
right values:

```sql
-- DuckDB: r.k is VARCHAR, l.k is INTEGER. '1' and '01' are distinct text
-- values but both compare equal to integer 1 after implicit casting.
CREATE TABLE l(k INTEGER);   -- 1, 2
CREATE TABLE r(k VARCHAR);   -- '1', '01', '2'
SELECT l.k FROM l
LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k = q.k;
-- l.k = 1 appears twice; dropping the join loses a row.
```

```sql
-- SQLite: BINARY-deduped text ('a' and 'A' both survive) compared with NOCASE.
SELECT l.k FROM l
LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k COLLATE NOCASE = q.k;
-- the two distinct right values compare equal to 'a' under NOCASE.
```

The same shape appears with `GROUP BY` instead of `DISTINCT`, and when the
right side is a non-recursive CTE.

## What is required to prove one-to-one

For every column on the right side that the right query guarantees unique
(its **unique outputs**), the join must contain an equality that:

1. **Reaches a unique output.** The right operand is a bare output column of
   the joined source (an optional `COLLATE` wrapper is allowed). `CAST`,
   arithmetic or function calls on the right key are rejected because they can
   collapse distinct values (e.g. `CAST('01' AS INT) = CAST('1' AS INT)`).
2. **Uses a bare left column.** The left operand must be a plain column
   (optionally `COLLATE`-wrapped). Conversions on the left key can collapse
   distinct *left* values onto the same right value and fan the join out.
   Literal/`NULL` equalities instead *pin* the unique output to one value, so
   they count as one-to-one coverage for that output.
3. **Compares injectively given the available type evidence.**
4. **Uses a compatible collation** when both sides are text.

All unique outputs of the right side must be covered; an equality that only
mentions part of a composite unique key is rejected. Extra `AND` filters
(including equalities on non-column right expressions) can only narrow matches
and never invalidate the proof. An `OR` predicate is accepted only when the
same equalities appear in **every** disjunct; different keys per disjunct are
rejected.

## Type evidence

Types come from two sources, in order:

- Types already present on the AST (a pre-annotated expression).
- Otherwise, when a `schema` is supplied, a **private deep copy** is type
  annotated and node types are mapped back structurally. The caller's AST, its
  types and its metadata are never mutated, which preserves the
  `optimize(copy=True)` contract. The mapping is structural rather than by
  node identity because `join_condition` analyzes a copy of the `ON` clause.

Two concrete types allow elimination when they belong to the same comparison
family, where equality cannot identify distinct values:

| Family                | Members                                              |
| --------------------- | ---------------------------------------------------- |
| Integer               | `TINYINT` … `BIGINT`, unsigned integers, `BIT`       |
| Floating point        | `FLOAT`, `DOUBLE`                                    |
| Exact decimal         | `DECIMAL`, `BIGDECIMAL`, `DECIMAL32/64/128/256`, money types |
| Text                  | `CHAR`, `VARCHAR`, `TEXT`, `NVARCHAR`, …            |
| Binary                | `BLOB`, `BINARY`, `VARBINARY`, …                     |
| Temporal              | `DATE`, `DATETIME`, `TIMESTAMP`, `TIME`, …           |
| Boolean               | `BOOLEAN`                                            |

`NULL`-typed operands never match and therefore never duplicate rows. Any
other combination (integer vs. text, text vs. decimal, boolean vs. integer,
…) depends on engine-specific implicit coercion and is rejected. When type
evidence is missing on either side, the rule falls back to its historical
structural behavior instead of guessing, so `eliminate_joins(expression)`
without schema keeps working.

## Collation evidence (SQLite built-ins)

The right-side `DISTINCT`/`GROUP BY` guarantees uniqueness under a specific
collation. A join is injective when its effective collation distinguishes at
least as finely as that uniqueness collation:

| Right uniqueness collation | Join collations that stay one-to-one | Rejected join collations |
| -------------------------- | ------------------------------------- | ------------------------ |
| `BINARY`                   | `BINARY`                              | `NOCASE`, `RTRIM`, unknown |
| `NOCASE`                   | `BINARY`, `NOCASE`                    | `RTRIM`, unknown         |
| `RTRIM`                    | `BINARY`, `RTRIM`                     | `NOCASE`, unknown         |

`BINARY` is the codepoint-identical collation and is finer than every other
built-in collation, so a `BINARY` join can never identify rows that survived a
`NOCASE`/`RTRIM` dedup. Two explicitly different collations on the two
operands, or an unrecognized collation, are handled conservatively. On
non-SQLite engines only the default `BINARY` comparison is modeled; any
explicit collation makes the rule leave the join in place unless it can be
proved equal on both sides.

## Where the analysis lives and what is reused

The check stays inside the existing `eliminate_joins` rule, which runs after
`qualify` and name resolution in the optimizer pipeline. The pipeline now
passes its `dialect` and `schema` to the rule; the rule accepts them as
optional keyword arguments so existing single-argument calls are unchanged.

A per-invocation `_JoinEliminator` holds:

- A memo of each scope's unique outputs, keyed by scope identity.
- The type lookup over the private annotated copy (when needed).

Both caches die with the rule call, so conclusions never leak between calls or
between joins. Uniqueness is traced through arbitrary layers of
**column-forwarding** derived tables and CTEs (`SELECT t.a FROM (…) t`,
including renamed and shadowed aliases); each output is traced by name, so
inner/outer alias names cannot be confused. A forwarding layer may carry
additional non-column projections (constants, aggregates) without breaking
the trace of the key columns. Set operations, layers with their own joins, and
projections that recompute columns end the trace conservatively.

Because scope traversal builds a shared CTE body exactly once and the memo is
keyed by scope identity, a CTE referenced by N joins is analyzed once rather
than N times, and shared subqueries are never copied or expanded. Work grows
linearly with forwarding depth and with the number of distinct scopes, not
with the number of references. Comparison verdicts are intentionally *not*
cached across joins: the same CTE may feed one safe and one dangerous join,
and each must be judged under its own `ON` semantics.

## Conservative boundaries (join is kept)

- No usable type information on both sides of a cross-family-looking
  comparison when the caller asked for schema-based safety (the bare
  no-schema call intentionally keeps the old structural behavior).
- Mixed type families, casts or other value-mapping expressions on either key.
- `DISTINCT ON`, set-operation right sides, unknown collations, collations
  coarser than the right-side uniqueness collation.
- Unique outputs not fully covered by the equalities (including partially
  covered composite keys), or `OR` predicates whose disjuncts use different
  keys.
- Any non-`LEFT` join, semi/anti joins, nearest joins, unqualified columns, or
  right columns referenced outside the `ON` clause.

Cross joins and `LEFT JOIN`s without an `ON` clause keep their existing
treatment: they are removed only when the right side is provably a single
output row.
