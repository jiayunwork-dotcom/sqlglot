# Recursive CTE analysis: scope binding, lineage and projection pruning

## Scope of support

The recursive analysis applies to a `WITH RECURSIVE` block where every recursive CTE:

- declares **explicit, non-duplicated column names** (an explicit CTE column list, or
  unique projection aliases);
- is exactly one non-recursive `SELECT` followed by one recursive `SELECT`, combined with
  `UNION ALL`;
- references itself **once** in the recursive term, in `FROM` or an `INNER JOIN`;
- has no aggregate, `DISTINCT`, window function, `ORDER BY` / `LIMIT`, or correlated
  subquery inside the recursive term;
- uses only integer, text and `NULL` data expressible in SQLite.

Ordinary (non-recursive) CTEs, derived tables, base-table joins, qualified and
unqualified column names, projection stars (expanded by `qualify`), and `WHERE` / `ON`
conditions are all supported inside and around the recursive CTEs. Mutual recursion is not
supported; a relation outside the supported shape is marked `supported=False` on its
binding and every consumer falls back to the previous conservative behavior.

## The shared fact: `RecursiveRelation` and `RecursiveFlow`

`build_scope` gives each supported recursive CTE a `RecursiveRelation` binding
(`sqlglot.optimizer.scope`), reachable in three equivalent ways:

- `scope.recursive_relation` on the CTE's `UNION ALL` scope and on the recursive arm;
- `scope.recursive_relations` on the scope that owns the `WITH` clause (inner same-named
  CTEs shadow outer ones instead of overwriting them);
- `scope.recursive_binding(name)` for a hierarchy-aware lookup.

The binding exposes distinct column identities for the relation itself
(`relation.columns`, positional) separate from the anchor arm's projections. The
recursive arm's self-reference in `scope.sources` points at the full `UNION ALL` scope,
not at the anchor scope, so qualification, lineage and pruning all resolve the recursion
through one object.

`build_recursive_flow` (`sqlglot.optimizer.recursive`) turns the binding into one finite
`RecursiveFlow` summary, cached on the binding:

- `anchor_inputs[i]` / `recursive_external[i]`: external (base/CTE/derived) columns read
  by output `i` in each arm;
- `recursive_inputs[i]`: relation column indexes the recursive term reads from the
  previous iteration to produce output `i`;
- `predicate_recursive` / `anchor_conditions` / `recursive_conditions`: relation columns
  and external columns used only by `WHERE` / `ON`, kept per arm.

Bindings built by different scope traversals of the same AST share their cached flow
through a weak-registry keyed by the CTE node, so the cache neither keeps an AST alive
nor survives an expression copy. The registry also holds a weak reference to the CTE and
sweeps entries whose CTE has been collected, so it is bounded by the number of *live*
recursive relations rather than by historical analyses. For C column identities and E
direct dependencies the resident state is O(C + E). The fixed point is a work-list
closure over the stored edges — SQL runtime iterations, recursive paths and data rows
are never enumerated or used as an analysis budget.

## Source identity for external references

A local table alias is meaningful only inside the query block that introduces it, so it
is **never** part of an external source's identity. The flow classifies each reference
once, per arm, into a `SourceKey`:

- query source (CTE, derived table, any scope source): `("query", id(query_expr), output_name)`;
- physical table: `("table", catalog, db, name, column_name)`;
- unresolved / correlated reference: `("unresolved", alias, column_name)`.

The alias is used only to resolve the reference within its own arm via `Scope.sources`,
and an alias column list (`x AS s(p, q)`, `r AS rr(a, b)`) is mapped positionally to the
source's real output name before the key is formed. A recursive self-reference is matched
exactly as the binder matches it: an unqualified table whose name equals the CTE name,
possibly written under a different alias (`FROM r AS rr`).

Because the key is source-based, two arms may reuse one alias for different sources, two
same-named CTEs in different nested scopes stay distinct, an alias can equal the CTE name
while pointing at a real table, and an aliased self-reference is not mistaken for an
external one. Per-arm reference-site maps let every consumer look up a column by the
`exp.Column` node identity instead of re-interpreting the alias, so lineage and projection
pruning consume the same decision the flow made; neither re-guesses it.

Value use and predicate use are classified separately: a column seen only in `WHERE` /
`ON` is never a lineage value source, but it still keeps the column that produces it
alive for row semantics.

## Design alternatives considered

**Option A — unroll the recursion to a fixed depth in the lineage graph.** Rejected:
the depth would be an arbitrary analysis budget, the graph size would grow with the
runtime number of rounds, and a source introduced only after the cutoff would still be
missed. The chosen representation stores each column-swap edge exactly once and renders
the fixed-point cycle explicitly, so the answer is depth-independent and the graph stays
finite by construction.

**Option B — keep the flow summary global keyed by SQL text, instead of on the
binding.** Rejected: text-keyed state would survive unrelated queries, could not tell a
same-named inner CTE from an outer one, and would go stale after an in-place rewrite.
The chosen cache is attached to the CTE node through a weak registry: multiple
references to one relation share the same summary, shadowing keeps separate bindings,
and pruning invalidates only the relation it rewrote.

**Option C — re-resolve the alias at each consumer.** Rejected: lineage and projection
pruning could then disagree about which table a same-named alias means, which is exactly
the bug this design removes. Resolving once per arm and storing the classification keeps
the two consumers consistent at the cost of a small per-reference map (linear in the
number of column references in the two arms).

## Lineage

`lineage(column, ...)` and `lineage(None, ...)` include every base-table value source
reachable through repeated recursive column swaps, including sources introduced only by
the recursive term. Predicate-only columns never appear as value sources. The graph keeps
one node per relation column identity with explicit back edges for the swaps, so
`Node.walk()` and `Node.to_html()` terminate without unrolling rounds. A pre-built
`scope=` produces the same sources and relation as building from SQL.

## Projection pruning

`pushdown_projections` keeps a recursive column when it is a retained output, feeds a
retained output through a recursive swap, or appears in any `WHERE` / `ON` condition
(recursion stop or join match). A column that only loops to itself, with no retained
output or condition depending on it, is removed from the CTE column list and both arm
projections together. The external columns demanded from CTEs and derived tables are the
same source keys the flow resolved, so a column kept only because an alias matched in the
*other* arm is no longer retained. Unsupported shapes (e.g. `UNION DISTINCT`, aggregates)
keep the historical conservative behavior unchanged.

After pruning, call `qualify` / `build_scope` / `lineage` again normally: cached flows are
invalidated by the rule, and the new analysis binds the remaining columns at their real
positions.
