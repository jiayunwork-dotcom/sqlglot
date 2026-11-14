from __future__ import annotations

import json
import logging
import typing as t
from dataclasses import dataclass, field

from sqlglot import Schema, exp, maybe_parse
from sqlglot.errors import OptimizeError, SqlglotError
from sqlglot.optimizer import Scope, build_scope, find_all_in_scope, normalize_identifiers, qualify
from sqlglot.optimizer.scope import ScopeType, walk_in_scope
from sqlglot.schema import ensure_schema

if t.TYPE_CHECKING:
    from sqlglot.dialects.dialect import DialectType
    from collections.abc import Iterator, Mapping, Sequence
    from sqlglot._typing import GraphHTMLArgs
    from typing_extensions import Unpack

logger = logging.getLogger("sqlglot")

# A dependency on a base table. The last component is the column name, or "*" for the
# table's row set itself (rather than shorthand for every column).
ColumnDependency = t.Tuple[str, str, str, str]


@dataclass(frozen=True)
class Node:
    name: str
    expression: exp.Expr
    source: exp.Expr
    downstream: list[Node] = field(default_factory=list)
    source_name: str = ""
    reference_node_name: str = ""

    # Caller-injected per-node data, populated via the `on_node` hook on lineage()
    payload: dict[str, t.Any] = field(default_factory=dict)

    def walk(self) -> Iterator[Node]:
        visited: set[int] = set()
        queue = [self]
        while queue:
            node = queue.pop()
            node_id = id(node)
            if node_id in visited:
                continue
            visited.add(node_id)
            yield node
            queue.extend(reversed(node.downstream))

    def to_html(self, dialect: DialectType = None, **opts: Unpack[GraphHTMLArgs]) -> GraphHTML:
        nodes = {}
        edges = []

        for node in self.walk():
            if isinstance(node.expression, exp.Table):
                label = f"FROM {node.expression.this}"
                title = f"<pre>SELECT {node.name} FROM {node.expression.this}</pre>"
                group = 1
            else:
                label = node.expression.sql(pretty=True, dialect=dialect)
                source = node.source.transform(
                    lambda n: (
                        exp.Tag(this=n, prefix="<b>", postfix="</b>") if n is node.expression else n
                    ),
                    copy=False,
                ).sql(pretty=True, dialect=dialect)
                title = f"<pre>{source}</pre>"
                group = 0

            node_id = id(node)

            nodes[node_id] = {
                "id": node_id,
                "label": label,
                "title": title,
                "group": group,
            }

            for d in node.downstream:
                edges.append({"from": node_id, "to": id(d)})
        return GraphHTML(nodes, edges, **opts)


@t.overload
def lineage(
    column: str | exp.Column, sql: str | exp.Expr, schema: dict | Schema, **kwargs: t.Any
) -> Node: ...


@t.overload
def lineage(
    column: None, sql: str | exp.Expr, schema: dict | Schema, **kwargs: t.Any
) -> dict[str, Node]: ...


def lineage(
    column: str | exp.Column | None,
    sql: str | exp.Expr,
    schema: dict | Schema,
    sources: Mapping[str, str | exp.Query] | None = None,
    dialect: DialectType = None,
    scope: Scope | None = None,
    trim_selects: bool = True,
    copy: bool = True,
    on_node: t.Callable[[Node], None] | None = None,
    **kwargs,
) -> Node | dict[str, Node]:
    """Build the lineage graph for a SQL query.

    If `column` is given, returns the lineage Node for that single output column.
    If `column` is None, returns a dict mapping every top-level output column name
    to its lineage Node (with a shared cache so cross-column work is deduplicated).

    Args:
        column: The column to build the lineage for. Pass None to get all output columns.
        sql: The SQL string or expression.
        schema: The schema of tables.
        sources: A mapping of queries which will be used to continue building lineage.
        dialect: The dialect of input SQL.
        scope: A pre-created scope to use instead.
        trim_selects: Whether to clean up selects by trimming to only relevant columns.
        copy: Whether to copy the Expr arguments.
        on_node: Optional callback invoked for every Node created during the walk,
            after the Node's downstream is populated. Useful for injecting
            caller-managed data into Node.payload during the walk.
        **kwargs: Qualification optimizer kwargs.

    Returns:
        A Node when `column` is provided, or a dict[str, Node] when `column` is None.
    """
    expression = maybe_parse(sql, copy=copy, dialect=dialect)

    if sources:
        expression = exp.expand(
            expression,
            {
                k: t.cast(exp.Query, maybe_parse(v, copy=copy, dialect=dialect))
                for k, v in sources.items()
            },
            dialect=dialect,
            copy=copy,
        )

    schema = ensure_schema(schema, dialect=dialect)

    if not scope:
        expression = qualify.qualify(
            expression,
            dialect=dialect,
            schema=schema,
            **{"validate_qualify_columns": False, "identify": False, **kwargs},  # type: ignore
        )
        scope = build_scope(expression)

    if not scope:
        raise SqlglotError("Cannot build lineage, sql must be SELECT")

    selectable = scope.expression
    if not isinstance(selectable, exp.Selectable):
        raise SqlglotError("Cannot build lineage, sql must be a query")

    cache: dict[tuple, Node] = {}
    scope_meta: dict[int, tuple[bool, dict[str, exp.Expr]]] = {}

    if column is not None:
        column_name = normalize_identifiers.normalize_identifiers(column, dialect=dialect).name
        if not any(select.alias_or_name == column_name for select in selectable.selects):
            raise SqlglotError(f"Cannot find column '{column_name}' in query.")

        return to_node(
            column_name,
            scope,
            dialect,
            trim_selects=trim_selects,
            schema=schema,
            _cache=cache,
            _scope_meta=scope_meta,
            on_node=on_node,
        )

    result: dict[str, Node] = {}
    for sel in selectable.selects:
        name = sel.alias_or_name
        if not name:
            raise SqlglotError(
                f"Cannot fetch lineage for unnamed projection: {sel.sql(dialect=dialect)}."
            )

        result[name] = to_node(
            name,
            scope,
            dialect,
            trim_selects=trim_selects,
            schema=schema,
            _cache=cache,
            _scope_meta=scope_meta,
            on_node=on_node,
        )

    return result


def to_node(
    column: str | int,
    scope: Scope,
    dialect: DialectType,
    schema: Schema,
    scope_name: str | None = None,
    upstream: Node | None = None,
    source_name: str | None = None,
    reference_node_name: str | None = None,
    trim_selects: bool = True,
    _cache: dict[tuple, Node] | None = None,
    _scope_meta: dict[int, tuple[bool, dict[str, exp.Expr]]] | None = None,
    on_node: t.Callable[[Node], None] | None = None,
) -> Node:
    cache_key = (column, id(scope), scope_name, source_name, reference_node_name)

    if _cache is not None and cache_key in _cache:
        cached_node = _cache[cache_key]
        if upstream:
            upstream.downstream.append(cached_node)
        return cached_node

    # Find the specific select clause that is the source of the column we want.
    # This can either be a specific, named select or a generic `*` clause.
    selectable = t.cast(exp.Selectable, scope.expression)
    if isinstance(column, int):
        if column >= len(selectable.selects):
            raise SqlglotError(
                f"Cannot find column's source with index {column} in query: {selectable.sql(dialect=dialect)}"
            )
        select = selectable.selects[column]
    else:
        # Resolving a column to its select scans selectable.selects on every call;
        # memoize a per-scope {name: select} map and is_star bit instead.
        if _scope_meta is None:
            select = next(
                (s for s in selectable.selects if s.alias_or_name == column),
                exp.Star() if selectable.is_star else scope.expression,
            )
        else:
            scope_id = id(scope)
            meta = _scope_meta.get(scope_id)
            if meta is None:
                select_by_name: dict[str, exp.Expr] = {}
                for sel in selectable.selects:
                    select_by_name.setdefault(sel.alias_or_name, sel)
                meta = (selectable.is_star, select_by_name)
                _scope_meta[scope_id] = meta
            is_star, select_by_name = meta
            select = select_by_name.get(column, exp.Star() if is_star else scope.expression)

    if isinstance(scope.expression, exp.Subquery):
        for inner_scope in scope.subquery_scopes:
            result = to_node(
                column,
                scope=inner_scope,
                dialect=dialect,
                upstream=upstream,
                source_name=source_name,
                reference_node_name=reference_node_name,
                trim_selects=trim_selects,
                schema=schema,
                _cache=_cache,
                _scope_meta=_scope_meta,
                on_node=on_node,
            )
            # Skip caching a passed-in upstream returned by an inner SetOp:
            # a sibling call at the same key with that node as its upstream
            # would otherwise self-loop on the cache hit.
            if _cache is not None and result is not upstream:
                _cache[cache_key] = result
            return result
    if isinstance(scope.expression, exp.SetOperation):
        name = type(scope.expression).__name__.upper()
        created_setop = upstream is None
        upstream = upstream or Node(name=name, source=scope.expression, expression=select)

        index = (
            column
            if isinstance(column, int)
            else next(
                (
                    i
                    for i, select in enumerate(selectable.selects)
                    if select.alias_or_name == column or select.is_star
                ),
                -1,  # mypy will not allow a None here, but a negative index should never be returned
            )
        )

        if index == -1:
            raise ValueError(f"Could not find {column} in {scope.expression}")

        for s in scope.set_operation_scopes:
            to_node(
                index,
                scope=s,
                dialect=dialect,
                upstream=upstream,
                source_name=source_name,
                reference_node_name=reference_node_name,
                trim_selects=trim_selects,
                schema=schema,
                _cache=_cache,
                _scope_meta=_scope_meta,
                on_node=on_node,
            )

        if _cache is not None and created_setop:
            _cache[cache_key] = upstream
        if created_setop and on_node:
            on_node(upstream)
        return upstream

    if trim_selects and isinstance(scope.expression, exp.Select):
        # For better ergonomics in our node labels, replace the full select with
        # a version that has only the column we care about.
        #   "x", SELECT x, y FROM foo
        #     => "x", SELECT x FROM foo
        source: exp.Expr = scope.expression.select(select, append=False)
    else:
        source = scope.expression

    # Create the node for this step in the lineage chain, and attach it to the previous one.
    node = Node(
        name=f"{scope_name}.{column}" if scope_name else str(column),
        source=source,
        expression=select,
        source_name=source_name or "",
        reference_node_name=reference_node_name or "",
    )

    if upstream:
        upstream.downstream.append(node)

    subquery_scopes = {
        id(subquery_scope.expression): subquery_scope for subquery_scope in scope.subquery_scopes
    }

    for subquery in find_all_in_scope(select, *exp.UNWRAPPED_QUERIES):
        subquery_scope: Scope | None = subquery_scopes.get(id(subquery))
        if not subquery_scope:
            logger.warning(f"Unknown subquery scope: {subquery.sql(dialect=dialect)}")
            continue

        for name in subquery.named_selects:
            to_node(
                name,
                scope=subquery_scope,
                dialect=dialect,
                upstream=node,
                trim_selects=trim_selects,
                schema=schema,
                _cache=_cache,
                _scope_meta=_scope_meta,
                on_node=on_node,
            )

    # if the select is a star add all scope sources as downstreams
    if isinstance(select, exp.Star):
        for src in scope.sources.values():
            src_expr = src.expression if isinstance(src, Scope) else src
            star_node = Node(name=select.sql(comments=False), source=src_expr, expression=src_expr)
            node.downstream.append(star_node)
            if on_node:
                on_node(star_node)

    # Find all columns that went into creating this one to list their lineage nodes.
    source_columns = set(find_all_in_scope(select, exp.Column))

    # If the source is a UDTF find columns used in the UDTF to generate the table
    if isinstance(source, exp.UDTF):
        source_columns |= set(source.find_all(exp.Column))
        derived_tables: Sequence[exp.Expr] = [
            src.expression.parent
            for src in scope.sources.values()
            if isinstance(src, Scope) and src.is_derived_table and src.expression.parent
        ]
    else:
        derived_tables = scope.derived_tables

    source_names = {
        dt.alias: dt.comments[0].split()[1]
        for dt in derived_tables
        if dt.comments and dt.comments[0].startswith("source: ")
    }

    pivots = scope.pivots
    if pivots and pivots[0].parent is not pivots[-1].parent:
        # The scope's pivots only form a chain when they all hang off of the same source;
        # otherwise they can't be folded, so their columns degrade to unresolved leaves
        pivots = []

    pivot_renames: dict[str, str] = {}
    pivot_column_mapping: dict[str, list[exp.Column]] = {}

    if pivots:
        pivot_renames, pivot_column_mapping = _pivot_chain_mapping(pivots, scope, schema)

    for c in source_columns:
        table = c.table
        col_source: exp.Table | Scope | None = scope.sources.get(table)

        # Lineage-specific single-source inference: when the optimizer could not qualify a
        # column (e.g. no schema was provided), attribute it to the sole named source so
        # the graph remains accurate rather than degrading to a Placeholder leaf.
        if col_source is None and not table:
            named_sources = {k: v for k, v in scope.sources.items() if k}
            if len(named_sources) == 1:
                table, col_source = next(iter(named_sources.items()))
                c.set("table", exp.to_identifier(table))

        if isinstance(col_source, Scope):
            reference_node_name = None
            if col_source.scope_type == ScopeType.DERIVED_TABLE and table not in source_names:
                reference_node_name = table
            elif col_source.scope_type == ScopeType.CTE:
                selected_node, _ = scope.selected_sources.get(table, (None, None))
                reference_node_name = selected_node.name if selected_node else None

            # The table itself came from a more specific scope. Recurse into that one using the unaliased column name.
            to_node(
                c.name,
                scope=col_source,
                dialect=dialect,
                scope_name=table,
                upstream=node,
                source_name=source_names.get(table) or source_name,
                reference_node_name=reference_node_name,
                trim_selects=trim_selects,
                schema=schema,
                _cache=_cache,
                _scope_meta=_scope_meta,
                on_node=on_node,
            )
        elif pivots and pivots[-1].alias_or_name == c.table:
            # Only the last operator in a chain names the resulting source
            pivot_parent = pivots[-1].parent
            downstream_columns = []

            column_name = c.name
            if column_name in pivot_column_mapping:
                downstream_columns.extend(pivot_column_mapping[column_name])
            else:
                # The column is not in the pivot, so it must be an implicit column of the
                # pivoted source -- adapt column to be from the implicit pivoted source.
                downstream_columns.append(
                    exp.column(
                        pivot_renames.get(c.name, c.this),
                        table=pivot_parent.alias_or_name if pivot_parent else None,
                    )
                )

            for downstream_column in downstream_columns:
                if not downstream_column.table:
                    # Some dialects (e.g. bigquery) don't qualify the IN-list columns,
                    # but they can only come from the pivoted source
                    downstream_column = exp.column(
                        downstream_column.this,
                        table=pivot_parent.alias_or_name if pivot_parent else None,
                    )

                table = downstream_column.table
                col_source = scope.sources.get(table)
                if isinstance(col_source, exp.Table) and not col_source.db:
                    # A pivoted CTE reference maps to the raw table in `scope.sources`,
                    # so recover the CTE's scope to keep tracing through it
                    col_source = scope.cte_sources.get(col_source.name, col_source)
                if isinstance(col_source, Scope):
                    to_node(
                        downstream_column.name,
                        scope=col_source,
                        scope_name=table,
                        dialect=dialect,
                        upstream=node,
                        source_name=source_names.get(table) or source_name,
                        reference_node_name=reference_node_name,
                        trim_selects=trim_selects,
                        schema=schema,
                        _cache=_cache,
                        _scope_meta=_scope_meta,
                        on_node=on_node,
                    )
                else:
                    col_expr = col_source or exp.Placeholder()
                    pivot_leaf = Node(
                        name=downstream_column.sql(comments=False),
                        source=col_expr,
                        expression=col_expr,
                    )
                    node.downstream.append(pivot_leaf)
                    if on_node:
                        on_node(pivot_leaf)
        else:
            # The source is not a scope and the column is not in any pivot - we've reached the end
            # of the line. At this point, if a source is not found it means this column's lineage
            # is unknown. This can happen if the definition of a source used in a query is not
            # passed into the `sources` map.
            col_expr = col_source or exp.Placeholder()
            leaf = Node(name=c.sql(comments=False), source=col_expr, expression=col_expr)
            node.downstream.append(leaf)
            if on_node:
                on_node(leaf)

    if _cache is not None:
        _cache[cache_key] = node

    if on_node:
        on_node(node)

    return node


def _pre_pivot_columns(pivot: exp.Pivot, scope: Scope, schema: Schema) -> list[str]:
    """
    The columns the first operator of a chain sees, taken from the projections of a
    derived table or CTE source, or from the schema for a physical table. Returns an
    empty list when they can't be determined (e.g. an unexpanded star), since anything
    positional over them would silently shift.
    """
    parent = pivot.parent
    columns: list[str] = []
    if isinstance(parent, exp.DerivedTable) and isinstance(parent.this, exp.Query):
        columns = parent.this.named_selects
    elif isinstance(parent, exp.Table):
        cte_source = scope.cte_sources.get(parent.name) if not parent.db else None
        if isinstance(cte_source, Scope) and isinstance(cte_source.expression, exp.Query):
            columns = cte_source.expression.named_selects
        elif schema is not None:
            columns = list(schema.column_names(parent, only_visible=True))

    return [] if "*" in columns else columns


def _pivot_chain_mapping(
    pivots: list[exp.Pivot], scope: Scope, schema: Schema
) -> tuple[dict[str, str], dict[str, list[exp.Column]]]:
    """
    Fold a chain of (UN)PIVOT operators into a single view of its output, since each one
    consumes the previous one's columns rather than the pivoted source's.

    Returns the composed output-name -> pre-chain-name renames (from alias column lists),
    and the composed output-name -> source columns it derives from.
    """
    available = _pre_pivot_columns(pivots[0], scope, schema)
    renames: dict[str, str] = {}
    mapping: dict[str, list[exp.Column]] = {}

    for pivot in pivots:
        # Renames are positional over the operator's full output, so they can only be
        # applied when the columns going into it are known
        step_renames = (
            pivot.output_columns(available) if pivot.alias_column_names and available else {}
        )
        step_mapping = _pivot_column_mapping(pivot)
        if step_renames:
            step_mapping = {
                post: step_mapping[pre] for post, pre in step_renames.items() if pre in step_mapping
            }

        # Columns this operator consumed may have been produced by an earlier one, or be
        # alias-list renames of passthroughs; resolve through what we've folded so far
        def resolve(col: exp.Column) -> list[exp.Column]:
            if col.name in mapping:
                return mapping[col.name]
            if col.name in renames:
                return [exp.column(renames[col.name])]
            return [col]

        composed = {
            out: [resolved for col in cols for resolved in resolve(col)]
            for out, cols in step_mapping.items()
        }

        # Whatever an earlier operator produced and this one didn't consume passes through,
        # under whatever name this operator's alias column list gives it
        consumed = {col.name for cols in step_mapping.values() for col in cols}
        pre_to_post = {pre: post for post, pre in step_renames.items()}
        for out, cols in mapping.items():
            if out not in consumed:
                composed.setdefault(pre_to_post.get(out, out), cols)

        renames = (
            {post: renames.get(pre, pre) for post, pre in step_renames.items()}
            if step_renames
            else renames
        )
        mapping = composed
        available = list(pivot.output_columns(available)) if available else []

    return renames, mapping


def _pivot_column_mapping(pivot: exp.Pivot) -> dict[str, list[exp.Column]]:
    """Map each (UN)PIVOT output column name to the source columns it's derived from."""
    mapping: dict[str, list[exp.Column]] = {}

    if pivot.unpivot:
        # UNPIVOT((v1, v2) FOR name IN ((a1, a2), (b1, b2))): each value column is derived
        # positionally from the IN-list entries, and the name column from all of them
        value_columns = [
            identifier for e in pivot.expressions for identifier in e.find_all(exp.Identifier)
        ]
        for value_column in value_columns:
            mapping[value_column.name] = []

        for field in pivot.fields:
            if not isinstance(field, exp.In):
                continue

            name_columns = mapping.setdefault(field.this.name, [])
            for entry in field.expressions:
                entry_columns = list(entry.find_all(exp.Column))
                name_columns.extend(entry_columns)

                if len(entry_columns) == len(value_columns):
                    for value_column, column in zip(value_columns, entry_columns):
                        mapping[value_column.name].append(column)
                else:
                    for value_column in value_columns:
                        mapping[value_column.name].extend(entry_columns)

        return mapping

    # For each aggregation function, the pivot creates a new column for each field in category
    # combined with the aggfunc. So the columns parsed have this order: cat_a_value_sum, cat_a,
    # b_value_sum, b. Because of this step wise manner the aggfunc 'sum(value) as value_sum'
    # belongs to the column indices 0, 2, and the aggfunc 'max(price)' without an alias belongs
    # to the column indices 1, 3. Here, only the columns used in the aggregations are of interest
    # in the lineage, so lookup the pivot column name by index and map that with the columns used
    # in the aggregation.
    #
    # Example: PIVOT (SUM(value) AS value_sum, MAX(price)) FOR category IN ('a' AS cat_a, 'b')
    pivot_columns = pivot.args["columns"]
    pivot_aggs_count = len(pivot.expressions)

    mapping = {}
    for i, agg in enumerate(pivot.expressions):
        agg_cols = list(agg.find_all(exp.Column))
        for col_index in range(i, len(pivot_columns), pivot_aggs_count):
            mapping[pivot_columns[col_index].name] = agg_cols
    return mapping


class GraphHTML:
    """Node to HTML generator using vis.js.

    https://visjs.github.io/vis-network/docs/network/
    """

    def __init__(
        self,
        nodes: dict,
        edges: list,
        imports: bool = True,
        options: Mapping[str, object] | None = None,
    ):
        self.imports = imports

        self.options = {
            "height": "500px",
            "width": "100%",
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "nodeSpacing": 200,
                    "sortMethod": "directed",
                },
            },
            "interaction": {
                "dragNodes": False,
                "selectable": False,
            },
            "physics": {
                "enabled": False,
            },
            "edges": {
                "arrows": "to",
            },
            "nodes": {
                "font": "20px monaco",
                "shape": "box",
                "widthConstraint": {
                    "maximum": 300,
                },
            },
            **(options or {}),
        }

        self.nodes = nodes
        self.edges = edges

    def __str__(self):
        nodes = json.dumps(list(self.nodes.values()))
        edges = json.dumps(self.edges)
        options = json.dumps(self.options)
        imports = (
            """<script type="text/javascript" src="https://unpkg.com/vis-data@latest/peer/umd/vis-data.min.js"></script>
  <script type="text/javascript" src="https://unpkg.com/vis-network@latest/peer/umd/vis-network.min.js"></script>
  <link rel="stylesheet" type="text/css" href="https://unpkg.com/vis-network/styles/vis-network.min.css" />"""
            if self.imports
            else ""
        )

        return f"""<div>
  <div id="sqlglot-lineage"></div>
  {imports}
  <script type="text/javascript">
    var nodes = new vis.DataSet({nodes})
    nodes.forEach(row => row["title"] = new DOMParser().parseFromString(row["title"], "text/html").body.childNodes[0])

    new vis.Network(
        document.getElementById("sqlglot-lineage"),
        {{
            nodes: nodes,
            edges: new vis.DataSet({edges})
        }},
        {options},
    )
  </script>
</div>"""

    def _repr_html_(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class LineageRoles:
    """The two static lineage role sets of a single query output column.

    Attributes:
        value: Base table columns the output expression explicitly derives its value from
            (function arguments, CASE conditions and branches, ...). A plain constant has
            an empty `value`; `COUNT(*)` instead carries the row set markers of the sources
            being counted.
        rows: The row set markers of every source in the enclosing query's FROM clause, plus
            every base table column referenced by its WHERE, JOIN ON/USING, GROUP BY and
            HAVING clauses (including columns referenced inside EXISTS subqueries).
    """

    value: frozenset[ColumnDependency]
    rows: frozenset[ColumnDependency]


def lineage_roles(
    sql: str | exp.Expr,
    schema: dict | Schema,
    *,
    dialect: DialectType = None,
    sources: Mapping[str, str | exp.Query] | None = None,
) -> dict[str, LineageRoles]:
    """Build value-vs-rows static column lineage for a query.

    Unlike :func:`lineage`, which only traces where output values come from, this also
    records which base tables and columns influence *which rows* reach the output through
    filters, joins and grouping.

    The two roles are kept separate:

    - ``value`` answers which base table columns an output expression explicitly uses.
      Function arguments, CASE conditions and branches all count. A plain constant has an
      empty ``value``; ``COUNT(*)`` instead carries the row set markers of the sources
      whose rows are counted.
    - ``rows`` contains the row set markers of every effective FROM source of the
      enclosing query, plus every base table column referenced by its WHERE, JOIN
      ON/USING, GROUP BY and HAVING clauses and by any EXISTS/NOT EXISTS subqueries. A
      grouping key that is only used for grouping still enters ``rows``. The same column
      may appear in both roles.

    Dependencies are recorded structurally: predicates are not eliminated by constant
    folding, so every explicitly referenced column is retained. A top-level ORDER BY
    without LIMIT contributes to neither role.

    A dependency is a ``(catalog, db, table, column)`` tuple of strings, with missing
    catalog/db components as empty strings. ``column == "*"`` is a row set marker for the
    table, not shorthand for all of its columns. Tracing continues through CTEs, derived
    tables and named ``sources`` down to physical base tables. When an intermediate column
    is used to compute an output it feeds ``value``; when it is used for filtering,
    joining or grouping it feeds ``rows``. UNION ALL branches are matched positionally and
    merged per output position, with the output names taken from the first branch.

    Supported subset: plain SELECT with aliases, stars, scalar operators and CASE;
    INNER/LEFT/CROSS joins (ON or USING); non-recursive CTEs and derived tables; WHERE,
    EXISTS/NOT EXISTS, GROUP BY/HAVING; COUNT/SUM/MIN/MAX/AVG aggregates; DISTINCT and
    UNION ALL. Anything outside this subset (recursive CTEs, other set operations,
    windows, QUALIFY, LIMIT/OFFSET, LATERAL, table functions, PIVOT, scalar or other
    subqueries, other join types, ORDER BY outside the outermost query) raises
    `OptimizeError` rather than returning a partial result.

    Args:
        sql: The SQL string or query AST. It is never modified.
        schema: The schema of the base tables, as accepted by :func:`qualify`.
        dialect: The dialect used to parse and interpret the input. Identifier
            normalization and quoting rules of that dialect apply.
        sources: A mapping of named queries (SQL strings or ASTs) used to continue tracing
            table references, with the same recursive expansion semantics as
            :func:`lineage`. Cyclic references raise `OptimizeError`.

    Returns:
        An ordered dict keyed by the qualified output column names, in the order of the
        outermost projection. Each value is a :class:`LineageRoles` whose `value` and
        `rows` are frozensets of ``(catalog, db, table, column)`` tuples. Missing catalog
        or database components are empty strings; ``column == "*"`` denotes the table's
        row set, never all of its columns.

    Raises:
        ParseError: If `sql` (or one of the sources) cannot be parsed.
        OptimizeError: If the input is not a query, references a table or column that
            cannot be resolved, produces unnamed or duplicate output columns, or uses a
            construct outside the supported subset.
    """
    expression = maybe_parse(sql, copy=True, dialect=dialect)

    if not isinstance(expression, exp.Query):
        raise OptimizeError("lineage_roles only supports queries")

    _validate_projection_names(expression)
    _validate_supported_constructs(expression)

    if sources:
        parsed_sources = {
            k: t.cast(exp.Query, maybe_parse(v, copy=True, dialect=dialect))
            for k, v in sources.items()
        }
        for name, source in parsed_sources.items():
            if not isinstance(source, exp.Query):
                raise OptimizeError(
                    f"Source '{name}' must be a query, got {type(source).__name__}"
                )
        expression = _expand_sources(expression, parsed_sources, dialect=dialect)

    schema = ensure_schema(schema, dialect=dialect)

    expression = qualify.qualify(
        expression,
        dialect=dialect,
        schema=schema,
        validate_qualify_columns=True,
        identify=False,
    )

    root_scope = build_scope(expression)
    if not root_scope:
        raise OptimizeError("Cannot build lineage_roles, sql must be a query")

    analyzer = _RoleAnalyzer()
    analyzer.validate_scope_tree(root_scope)

    output_names = list(t.cast(exp.Selectable, root_scope.expression).named_selects)

    results: dict[str, LineageRoles] = {}
    for index, name in enumerate(output_names):
        value, rows = analyzer.analyze_output(root_scope, index)
        results[name] = LineageRoles(value=frozenset(value), rows=frozenset(rows))

    return results


def _validate_projection_names(expression: exp.Expr) -> None:
    """Reject unnamed complex projections and duplicate output names up front.

    `qualify` invents names like `_col_0` for unnamed expressions, so the check has to
    run on the original AST. Set operation branches are positional and named after the
    first branch, so only the first SELECT of each set operation needs named outputs;
    subsequent branches are matched by position. Each SELECT has its own output
    namespace, so the duplicate check is scoped per SELECT.
    """
    for select in expression.find_all(exp.Select):
        # Projections inside an EXISTS subquery never produce output columns.
        if isinstance(select.parent, exp.Exists):
            continue
        # A right operand of a set operation is matched positionally and never names
        # outputs. The leftmost branch is not a right operand of any node, so it still
        # goes through the naming check even in a chained UNION ALL.
        if isinstance(select.parent, exp.SetOperation) and select is not select.parent.this:
            continue
        seen: set[str] = set()
        for projection in select.expressions:
            explicit_alias = isinstance(projection, exp.Alias)
            inner = projection.this if explicit_alias else projection
            name = projection.alias_or_name

            if not name or not (explicit_alias or isinstance(inner, (exp.Column, exp.Star))):
                raise OptimizeError(
                    f"Output expressions must have an explicit alias: {projection.sql()}"
                )
            if name in seen:
                raise OptimizeError(f"Duplicate output column name: {name}")
            seen.add(name)


def _validate_supported_constructs(expression: exp.Expr) -> None:
    """Reject unsupported syntax before `qualify` can rewrite or erase it.

    Some constructs (e.g. NATURAL JOIN) are normalized away by qualification, so the
    scope-level validation alone would miss them. This walks the raw AST and only checks
    properties that survive parsing but may not survive qualification.
    """
    for node in expression.walk():
        if isinstance(node, exp.Join) and node.args.get("method") is not None:
            raise OptimizeError(f"Unsupported join type: {node.sql()}")
        if isinstance(node, exp.SetOperation) and (
            node.args.get("distinct") or not isinstance(node, exp.Union)
        ):
            raise OptimizeError(f"Only UNION ALL set operations are supported: {node.sql()}")
        if isinstance(node, exp.With) and node.recursive:
            raise OptimizeError(f"Recursive CTEs are not supported: {node.sql()}")


def _expand_sources(
    expression: exp.Expr,
    sources: dict[str, exp.Query],
    dialect: DialectType,
) -> exp.Expr:
    """Expand named source references with cycle detection.

    `exp.expand` inlines sources recursively but does not detect cycles, which would
    otherwise blow the recursion limit. The expansion otherwise mirrors `lineage()`.
    """
    normalized_sources = {
        exp.normalize_table_name(k, dialect=dialect): v for k, v in sources.items()
    }

    def expand(node: exp.Expr, active: frozenset[str]) -> exp.Expr:
        if isinstance(node, exp.Table):
            name = exp.normalize_table_name(node, dialect=dialect)
            source = normalized_sources.get(name)
            if source is not None:
                if name in active:
                    raise OptimizeError(f"Cyclic source reference: {name}")
                subquery = source.subquery(node.alias_or_name or name)
                subquery.comments = [f"source: {name}"]
                return subquery.transform(
                    lambda child: expand(child, active | {name}), copy=False
                )
        return node

    return expression.transform(lambda node: expand(node, frozenset()), copy=True)


# Join shapes accepted by the first version. A bare join (comma-separated sources) parses
# without kind/side and behaves like a cross join.
_SUPPORTED_JOIN_SIDES = (None, "LEFT")
_SUPPORTED_JOIN_KINDS = (None, "INNER", "CROSS")


def _is_relation_subquery(node: exp.Subquery) -> bool:
    """Whether a Subquery node is a FROM/JOIN relation rather than a value subquery."""
    parent = node.parent
    while isinstance(parent, exp.Subquery):
        parent = parent.parent
    return isinstance(parent, (exp.From, exp.Join))


class _RoleAnalyzer:
    """Computes base-table column dependencies in the `value` and `rows` roles.

    A single instance spans one query analysis; all memoization is per-instance, so
    analyses never share state.
    """

    def __init__(self) -> None:
        # id(scope) -> per-output value sets
        self._value_cache: dict[int, list[frozenset[ColumnDependency]]] = {}
        # id(scope) -> scope-level rows set
        self._rows_cache: dict[int, frozenset[ColumnDependency]] = {}
        # Query/SetOperation expression id -> the scope scoping it
        self._subquery_scopes: dict[int, Scope] = {}

    # ------------------------------------------------------------------ validation

    def validate_scope_tree(self, root_scope: Scope) -> None:
        for scope in root_scope.traverse():
            self._index_subqueries(scope)
            self._validate_scope(scope)

    def _index_subqueries(self, scope: Scope) -> None:
        for subquery_scope in scope.subquery_scopes:
            self._subquery_scopes[id(subquery_scope.expression)] = subquery_scope

    def _validate_scope(self, scope: Scope) -> None:
        expression = scope.expression

        if isinstance(expression, exp.SetOperation):
            if expression.args.get("distinct") or not isinstance(expression, exp.Union):
                raise OptimizeError(
                    f"Only UNION ALL set operations are supported: {expression.sql()}"
                )
            if (
                expression.args.get("limit") is not None
                or expression.args.get("offset") is not None
            ):
                raise OptimizeError(
                    f"LIMIT/OFFSET is not supported: {expression.sql()}"
                )
            order = expression.args.get("order")
            if order is not None and not scope.is_root:
                raise OptimizeError(
                    "ORDER BY is only supported on the outermost query without "
                    f"LIMIT: {expression.sql()}"
                )
            arities = {
                len(branch.expression.named_selects)
                for branch in scope.set_operation_scopes
                if isinstance(branch.expression, exp.Selectable)
            }
            if len(arities) > 1:
                raise OptimizeError(
                    f"Set operation branches must have the same number of columns: {expression.sql()}"
                )
            return

        if not isinstance(expression, exp.Select):
            raise OptimizeError(f"Unsupported query type: {type(expression).__name__}")

        with_ = expression.args.get("with_")
        if with_ and with_.recursive:
            raise OptimizeError(f"Recursive CTEs are not supported: {expression.sql()}")

        if expression.args.get("limit") is not None or expression.args.get("offset") is not None:
            raise OptimizeError(f"LIMIT/OFFSET is not supported: {expression.sql()}")

        order = expression.args.get("order")
        if order is not None and not scope.is_root:
            raise OptimizeError(
                f"ORDER BY is only supported on the outermost query without LIMIT: {expression.sql()}"
            )

        if expression.args.get("qualify") is not None:
            raise OptimizeError(f"QUALIFY is not supported: {expression.sql()}")

        if scope.udtfs or expression.find(exp.UDTF):
            raise OptimizeError(f"Table functions and LATERAL are not supported: {expression.sql()}")

        if scope.pivots or expression.find(exp.Pivot):
            raise OptimizeError(f"PIVOT is not supported: {expression.sql()}")

        for join in expression.args.get("joins") or []:
            if (
                join.args.get("side") not in _SUPPORTED_JOIN_SIDES
                or join.args.get("kind") not in _SUPPORTED_JOIN_KINDS
                or join.args.get("method") is not None
            ):
                raise OptimizeError(f"Unsupported join type: {join.sql()}")

        for node in walk_in_scope(expression):
            if isinstance(node, exp.Window):
                raise OptimizeError(f"Window functions are not supported: {expression.sql()}")
            if isinstance(node, exp.SubqueryPredicate) and not isinstance(node, exp.Exists):
                raise OptimizeError(
                    f"Only EXISTS/NOT EXISTS subqueries are supported: {expression.sql()}"
                )
            if (
                isinstance(node, exp.Subquery)
                and not isinstance(node.parent, exp.Exists)
                and not _is_relation_subquery(node)
            ):
                raise OptimizeError(
                    f"Only EXISTS/NOT EXISTS subqueries are supported: {expression.sql()}"
                )

    # ------------------------------------------------------------------ entry point

    def analyze_output(
        self, scope: Scope, index: int
    ) -> tuple[set[ColumnDependency], set[ColumnDependency]]:
        return set(self._output_values(scope, index)), set(self._scope_rows(scope))

    # ------------------------------------------------------------------ value role

    def _output_values(self, scope: Scope, index: int) -> frozenset[ColumnDependency]:
        expression = scope.expression
        if isinstance(expression, exp.SetOperation):
            merged: set[ColumnDependency] = set()
            for branch in scope.set_operation_scopes:
                merged |= self._output_values(branch, index)
            return frozenset(merged)

        cached = self._value_cache.get(id(scope))
        if cached is None:
            cached = [
                self._projection_values(scope, projection)
                for projection in t.cast(exp.Select, expression).expressions
            ]
            self._value_cache[id(scope)] = cached
        return cached[index]

    def _projection_values(self, scope: Scope, projection: exp.Expr) -> frozenset[ColumnDependency]:
        if isinstance(projection, exp.Alias):
            projection = projection.this

        if isinstance(projection, exp.Star):
            # A surviving star means qualification could not expand it (typically
            # because the table's columns are absent from the schema).
            raise OptimizeError(
                f"Cannot resolve star projection without schema: {projection.sql()}"
            )

        dependencies: set[ColumnDependency] = set()
        for node in projection.walk():
            if isinstance(node, exp.Column):
                dependencies |= self._resolve_column_value(node, scope)
            elif isinstance(node, exp.Count) and isinstance(node.this, exp.Star):
                dependencies |= self._scope_rows(scope)
        return frozenset(dependencies)

    def _resolve_column_value(
        self, column: exp.Column, scope: Scope
    ) -> frozenset[ColumnDependency]:
        """Resolve a column reference used in a value context to base table columns."""
        source = self._lookup_source(column, scope)

        if isinstance(source, Scope):
            index = self._source_output_index(source, column.name)
            return self._output_values(source, index)

        return frozenset((self._table_marker(source, column.name),))

    def _lookup_source(self, column: exp.Column, scope: Scope) -> exp.Table | Scope:
        """Find the source a column references, walking up enclosing scopes.

        Local aliases shadow outer names because the nearest scope defining the table
        alias wins; qualify has already disambiguated unqualified names this way.
        """
        current: Scope | None = scope
        while current is not None:
            source = current.sources.get(column.table)
            if source is not None:
                return source
            current = current.parent
        raise OptimizeError(f"Cannot resolve column: {column.sql()}")

    def _source_output_index(self, source: Scope, name: str) -> int:
        selectable = t.cast(exp.Selectable, source.expression)
        for index, output_name in enumerate(selectable.named_selects):
            if output_name == name:
                return index
        raise OptimizeError(f"Cannot resolve column through subquery: {name}")

    # ------------------------------------------------------------------ rows role

    def _scope_rows(self, scope: Scope) -> frozenset[ColumnDependency]:
        cached = self._rows_cache.get(id(scope))
        if cached is not None:
            return cached

        expression = scope.expression
        if isinstance(expression, exp.SetOperation):
            merged: set[ColumnDependency] = set()
            for branch in scope.set_operation_scopes:
                merged |= self._scope_rows(branch)
            result = frozenset(merged)
        else:
            result = frozenset(self._select_rows(t.cast(exp.Select, expression), scope))

        self._rows_cache[id(scope)] = result
        return result

    def _select_rows(self, select: exp.Select, scope: Scope) -> set[ColumnDependency]:
        dependencies: set[ColumnDependency] = set()

        # Every effective FROM source contributes its row set marker. Only sources the
        # query actually selects from count: unused CTEs are defined in the scope but not
        # referenced, so they must not leak their tables into the result. A scope source
        # expands to the row sets of its own query; a table source is a base row marker.
        for source in scope.selected_sources.values():
            node = source[1]
            if isinstance(node, Scope):
                dependencies |= self._scope_rows(node)
            else:
                dependencies.add(self._table_marker(node, "*"))

        predicate_roots: list[exp.Expr] = []
        where = select.args.get("where")
        if where is not None:
            predicate_roots.append(where.this)

        for join in select.args.get("joins") or []:
            on = join.args.get("on")
            if on is not None:
                predicate_roots.append(on)
            # qualify rewrites USING(col) into an ON with both sides, but handle a
            # surviving USING clause explicitly so both join columns are captured.
            for column in join.find_all(exp.Column) if join.args.get("using") else ():
                predicate_roots.append(column)

        group = select.args.get("group")
        if group is not None:
            predicate_roots.extend(group.expressions)

        having = select.args.get("having")
        if having is not None:
            predicate_roots.append(having.this)

        for root in predicate_roots:
            self._collect_rows(root, scope, dependencies)

        return dependencies

    def _collect_rows(
        self, node: exp.Expr, scope: Scope, dependencies: set[ColumnDependency]
    ) -> None:
        """Collect the rows-role dependencies of a predicate or grouping expression.

        EXISTS subqueries are delegated to `_exists_rows` and their SELECT projection is
        never descended into: columns selected but unused by the existence predicate are
        not dependencies.
        """
        stack: list[exp.Expr] = [node]
        while stack:
            child = stack.pop()
            if isinstance(child, exp.Exists):
                self._exists_rows(child, dependencies)
                continue
            if isinstance(child, exp.Column):
                self._resolve_column_rows(child, scope, dependencies)
            for value in child.args.values():
                if isinstance(value, list):
                    stack.extend(v for v in reversed(value) if isinstance(v, exp.Expr))
                elif isinstance(value, exp.Expr):
                    stack.append(value)

    def _resolve_column_rows(
        self,
        column: exp.Column,
        scope: Scope,
        dependencies: set[ColumnDependency],
    ) -> None:
        """Resolve a column used in a rows context (filter, join, grouping, ...)."""
        source = self._lookup_source(column, scope)

        if isinstance(source, Scope):
            index = self._source_output_index(source, column.name)
            # The intermediate column is now used to decide which rows survive, so its
            # own value lineage degrades to rows dependencies.
            dependencies |= self._output_values(source, index)
            dependencies |= self._scope_rows(source)
        else:
            dependencies.add(self._table_marker(source, column.name))

    def _exists_rows(
        self, exists: exp.Exists, dependencies: set[ColumnDependency]
    ) -> None:
        subquery_scope = self._subquery_scopes.get(id(exists.this))
        if subquery_scope is None:
            raise OptimizeError(f"Cannot build scope for subquery: {exists.sql()}")

        # The subquery's FROM sources and internal predicates (including nested EXISTS)
        # always participate in the existence test. Its SELECT projection does not.
        dependencies |= self._scope_rows(subquery_scope)

        # Correlated columns are not part of the subquery's own sources, so
        # `_scope_rows` above cannot resolve them. They reference any enclosing scope,
        # with nearest-visible-scope shadowing handled by `_lookup_source`.
        for column in subquery_scope.external_columns:
            self._resolve_column_rows(column, subquery_scope, dependencies)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _table_marker(table: exp.Table, column: str) -> ColumnDependency:
        return (table.catalog, table.db, table.name, column)
