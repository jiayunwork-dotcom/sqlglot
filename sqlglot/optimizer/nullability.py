from __future__ import annotations

"""
Nullability inference.

This module infers whether each output column of a query can ever produce a
``NULL`` value. Its entry point is :func:`infer_nullability`.

The central contract is **no false negatives**:

    If a column is inferred as *not nullable* (``nullable=False``), then every
    row of the query result is guaranteed to be non-NULL, provided that the
    data in the base tables satisfies the supplied ``schema`` and ``not_null``
    constraints.

When the nullability of an expression cannot be established, it is reported
as nullable (``nullable=True``) instead of raising. Unknown functions, table
functions and other unsupported constructs are therefore treated
conservatively rather than causing an error. Being overly conservative is
always safe; claiming an expression is non-NULL when it might be is not.

Base-table nullability is governed exclusively by ``not_null``: only the base
table columns listed there are considered to carry a ``NOT NULL`` constraint.
Every other base-table column is treated as potentially nullable, regardless
of its declared data type.

Args:
    expression: A SQL string or a parsed query (``SELECT`` / set operations
        such as ``UNION`` / ``INTERSECT`` / ``EXCEPT``, including ``WITH``
        queries). Parsed expressions are never modified: a private copy is
        qualified and analyzed.
    schema: The database schema, in the same shape accepted by
        :func:`sqlglot.optimizer.qualify.qualify` (a nested mapping or a
        :class:`~sqlglot.schema.Schema` instance). ``SELECT *`` is expanded
        against it.
    not_null: Mapping of table name to the set of columns that have a
        ``NOT NULL`` constraint. Table names use the same key form as the
        schema, e.g. ``"db.orders"`` or ``"orders"``. Columns not listed are
        treated as potentially nullable.
    dialect: The dialect used to parse the SQL and to interpret function
        semantics. At least ``"duckdb"`` matches DuckDB's execution semantics.

Returns:
    A list of ``(column_name, nullable)`` tuples, ordered like the outermost
    query's output columns. ``nullable`` is ``True`` when the column may
    produce a ``NULL`` and ``False`` when it is guaranteed not to. Column
    names match the output column names produced after
    :func:`~sqlglot.optimizer.qualify.qualify` (stars are expanded).

Raises:
    sqlglot.errors.OptimizeError: If ``expression`` is not a query (e.g. an
        ``INSERT`` or ``CREATE`` statement), or if it references a table or
        column missing from ``schema``.
    sqlglot.errors.SchemaError: If ``not_null`` names a table or column that
        does not exist in ``schema``. The offending name is included in the
        error message.
"""

import typing as t

from sqlglot import exp, parse_one
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.errors import OptimizeError, SchemaError
from sqlglot.helper import seq_get
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import MappingSchema, Schema, ensure_schema


# Aggregates that never produce NULL, even as scalar (no GROUP BY) aggregates
# over an empty table.
_NON_NULL_AGG_FUNCS: tuple[type[exp.Func], ...] = (exp.Count, exp.Grouping, exp.GroupingId)

# Aggregates whose result is non-NULL when their arguments are non-NULL, but
# only in the presence of a GROUP BY: without one, an empty table yields NULL.
_GROUPED_NON_NULL_AGG_FUNCS: tuple[type[exp.Func], ...] = (
    exp.Sum,
    exp.Min,
    exp.Max,
    exp.Avg,
)

# Window functions that are guaranteed to produce a non-NULL value per row.
_NON_NULL_WINDOW_FUNCS: tuple[type[exp.Func], ...] = (
    exp.RowNumber,
    exp.Rank,
    exp.DenseRank,
)

# Binary arithmetic operators whose result is non-NULL iff both operands are.
_NON_NULL_ARITHMETIC: tuple[type[exp.Binary], ...] = (exp.Add, exp.Sub, exp.Mul)

# Comparison predicates that are NULL-rejecting on each of their operands:
# a NULL on either side makes the predicate NULL rather than TRUE.
_NULL_REJECTING_COMPARISONS: tuple[type[exp.Predicate], ...] = (
    exp.EQ,
    exp.NEQ,
    exp.GT,
    exp.GTE,
    exp.LT,
    exp.LTE,
)


def infer_nullability(
    expression: str | exp.Expr,
    schema: dict[str, object] | Schema | None = None,
    not_null: dict[str, t.Collection[str]] | None = None,
    dialect: DialectType = None,
) -> list[tuple[str, bool]]:
    """
    Infer the nullability of every output column of a query.

    See the module docstring for the full contract, argument shapes and the
    raised exceptions.

    Example:
        >>> from sqlglot.optimizer.nullability import infer_nullability
        >>> schema = {"orders": {"id": "INT", "amount": "INT"}}
        >>> not_null = {"orders": {"id"}}
        >>> infer_nullability(
        ...     "SELECT id, amount FROM orders",
        ...     schema=schema,
        ...     not_null=not_null,
        ...     dialect="duckdb",
        ... )
        [('id', False), ('amount', True)]
    """
    dialect_cls = Dialect.get_or_raise(dialect)
    resolved_schema = ensure_schema(schema, dialect=dialect)

    if isinstance(expression, str):
        parsed = parse_one(expression, dialect=dialect)
    else:
        # Never mutate a caller-owned AST: qualify rewrites in place, so work
        # on a deep copy.
        parsed = expression.copy()

    if not isinstance(parsed, (exp.Query, exp.Values)):
        raise OptimizeError(
            f"Nullability can only be inferred for queries, got {type(parsed).__name__}: "
            f"{parsed.sql(dialect=dialect_cls)}"
        )

    not_null_columns = _validate_not_null(not_null or {}, resolved_schema, dialect_cls)

    # qualify expands stars, qualifies every column/table reference and raises
    # OptimizeError on unknown tables/columns, matching its own validation.
    qualified = qualify(parsed, schema=resolved_schema, dialect=dialect)

    analyzer = _NullabilityAnalyzer(resolved_schema, not_null_columns, dialect_cls)
    return analyzer.analyze(qualified)


def _validate_not_null(
    not_null: dict[str, t.Collection[str]],
    schema: Schema,
    dialect: Dialect,
) -> dict[tuple[str, ...], frozenset[str]]:
    """Validate ``not_null`` against ``schema`` and normalize its keys."""
    validated: dict[tuple[str, ...], frozenset[str]] = {}

    for table_name, columns in not_null.items():
        table = parse_one(table_name, into=exp.Table, dialect=dialect)

        table_columns = schema.column_names(table)
        if not table_columns:
            raise SchemaError(f"not_null references unknown table: {table_name}")

        normalized_columns = _normalize_names(columns, schema, dialect)
        table_column_set = set(table_columns)
        for column in normalized_columns:
            if column not in table_column_set:
                raise SchemaError(
                    f"not_null references unknown column: {column} in table {table_name}"
                )

        validated[_table_parts(table, schema, dialect)] = frozenset(normalized_columns)

    return validated


def _normalize_names(
    names: t.Collection[str], schema: Schema, dialect: Dialect
) -> list[str]:
    if isinstance(schema, MappingSchema):
        return [schema._normalize_name(name, dialect=dialect) for name in names]
    return list(names)


def _table_parts(table: exp.Table, schema: Schema, dialect: Dialect) -> tuple[str, ...]:
    if isinstance(schema, MappingSchema):
        normalized = schema._normalize_table(table, dialect=dialect)
        return tuple(part.name for part in normalized.parts)
    return tuple(part.name for part in table.parts)


class _NullabilityAnalyzer:
    """Computes column nullability across the scope tree of a qualified query."""

    def __init__(
        self,
        schema: Schema,
        not_null_columns: dict[tuple[str, ...], frozenset[str]],
        dialect: Dialect,
    ) -> None:
        self.schema = schema
        self.not_null_columns = not_null_columns
        self.dialect = dialect

        # Nullability of each scope's output columns, keyed by scope id. Each
        # value is a list aligned with the scope's output columns.
        self._scope_outputs: dict[int, list[bool]] = {}

        # Per-scope memo of expression id -> nullable. None values are not
        # stored, so None doubles as the "not yet computed" sentinel.
        self._memo: dict[int, bool] = {}
        self._guaranteed_columns: set[tuple[str, str]] = set()
        self._guaranteed_node_ids: set[int] = set()
        # Column nodes NULL-synthesized by ROLLUP/CUBE/GROUPING SETS subtotals.
        self._nullified_grouping_columns: set[tuple[str, str]] = set()

    def analyze(self, expression: exp.Expr) -> list[tuple[str, bool]]:
        scopes = traverse_scope(expression)
        if not scopes:
            # A top-level VALUES query does not produce traversable scopes.
            if isinstance(expression, exp.Values):
                nullable = self._values_nullability(expression)
                return list(zip(_values_output_names(expression), nullable))
            raise OptimizeError(
                f"Could not build scopes for expression: {expression.sql(dialect=self.dialect)}"
            )

        for scope in scopes:
            self._analyze_scope(scope)

        root = scopes[-1]
        root_expression = root.expression
        if isinstance(root_expression, exp.SetOperation):
            output_names = list(dict.fromkeys(root_expression.named_selects))
        else:
            output_names = [
                select.alias_or_name
                for select in root_expression.selects
                if not select.is_star
            ]
        nullability = self._scope_outputs[id(root)]
        return list(zip(output_names, nullability))

    # ------------------------------------------------------------------
    # Scope analysis
    # ------------------------------------------------------------------

    def _analyze_scope(self, scope: Scope) -> None:
        expression = scope.expression

        if isinstance(expression, exp.Select):
            self._analyze_select_scope(scope, expression)
        elif isinstance(expression, exp.SetOperation):
            self._analyze_set_operation_scope(scope, expression)
        elif isinstance(expression, exp.Values):
            self._scope_outputs[id(scope)] = self._values_nullability(expression)
        elif isinstance(expression, exp.Lateral):
            inner = seq_get(scope.subquery_scopes, 0)
            if inner is not None:
                outputs = self._scope_outputs.get(id(inner))
                self._scope_outputs[id(scope)] = (
                    outputs if outputs is not None else self._fallback_scope_outputs(inner)
                )
            else:
                self._scope_outputs[id(scope)] = self._fallback_scope_outputs(scope)
        else:
            # UDTFs and any other scope we cannot reason about: every output
            # column is conservatively nullable.
            self._scope_outputs[id(scope)] = self._fallback_scope_outputs(scope)

    def _fallback_scope_outputs(self, scope: Scope) -> list[bool]:
        expression = scope.expression
        selects = getattr(expression, "selects", None)
        if selects:
            return [True for select in selects if not select.is_star]
        # Fall back to the alias column list (e.g. UNNEST ... AS t(c1, c2)).
        return [True for _ in expression.alias_column_names]

    def _analyze_select_scope(self, scope: Scope, select: exp.Select) -> None:
        self._memo = {}
        self._guaranteed_columns = set()
        self._guaranteed_node_ids = set()
        self._nullified_grouping_columns: set[tuple[str, str]] = set()

        # 1. Columns guaranteed non-NULL by WHERE/HAVING predicates. These
        #    additionally recover NOT NULL columns nullified by outer joins
        #    when the predicate rejects the NULL-padded rows.
        where = select.args.get("where")
        if where:
            self._collect_guarantees(where.this, scope)

        having = select.args.get("having")
        if having:
            self._collect_guarantees(having.this, scope)

        # 2. Nullability of each source column as exposed by the FROM/JOIN
        #    tree, using the guarantees collected above.
        source_nullability = self._build_source_nullability(scope, select)

        # 3. ROLLUP/CUBE/GROUPING SETS synthesize overall (sub)total rows on
        #    which grouping columns are NULL, even if they are NOT NULL in
        #    the base table.
        self._nullified_grouping_columns = self._collect_nullified_grouping_columns(select)

        # 4. Nullability of every output projection.
        self._scope_outputs[id(scope)] = [
            self._is_nullable(projection, scope, source_nullability)
            for projection in select.selects
            if not projection.is_star
        ]

    def _collect_nullified_grouping_columns(
        self, select: exp.Select
    ) -> set[tuple[str, str]]:
        group = select.args.get("group")
        if not group:
            return set()

        nullified: set[tuple[str, str]] = set()
        for node in group.walk():
            if isinstance(node, (exp.Rollup, exp.Cube)):
                # Every grouping key can be replaced by a synthetic NULL on
                # some subtotal row.
                for column in node.find_all(exp.Column):
                    nullified.add((column.table, column.name))
            elif isinstance(node, exp.GroupingSets):
                # A grouping key is synthesized as NULL on every grouping set
                # that does not include it. Only keys present in *every*
                # non-empty grouping set keep their intrinsic nullability.
                sets: list[set[tuple[str, str]]] = []
                for element in node.expressions:
                    inner = element.this if isinstance(element, exp.Paren) else element
                    elements = (
                        inner.expressions if isinstance(inner, exp.Tuple) else [inner]
                    )
                    sets.append(
                        {
                            (column.table, column.name)
                            for column in elements
                            if isinstance(column, exp.Column)
                        }
                    )
                for column in node.find_all(exp.Column):
                    key = (column.table, column.name)
                    if not sets or any(key not in grouping_set for grouping_set in sets):
                        nullified.add(key)
        return nullified

    def _join_on_partners(self, select: exp.Select) -> dict[str, set[str]]:
        """
        For each joined source, collect the other sources referenced by an
        *equality* in that join's ON clause. When the joined source's row is
        present (it matched its ON predicate), each equality operand on another
        source is non-NULL, so that partner row is present too.

        Only equality predicates are considered: non-equality conditions
        (``>``, ``IS NOT NULL``, ...) can hold on a NULL-padded row, so they
        cannot prove that a partner row exists.
        """
        partners: dict[str, set[str]] = {}
        for join in select.args.get("joins") or []:
            source_name = join.alias_or_name
            on = join.args.get("on")
            if not source_name or on is None:
                continue

            referenced: set[str] = set()
            for equality in on.find_all(exp.EQ):
                for operand in (equality.left, equality.right):
                    column = operand if isinstance(operand, exp.Column) else None
                    if column and column.table and column.table != source_name:
                        referenced.add(column.table)
            partners.setdefault(source_name, set()).update(referenced)
        return partners

    def _build_source_nullability(
        self, scope: Scope, select: exp.Select
    ) -> dict[str, dict[str, bool]]:
        """Map source name -> (column name -> nullable) after FROM/JOIN logic."""
        result: dict[str, dict[str, bool]] = {}

        # Start with the FROM source (the first reference in source order).
        references = [name for name, _ in scope.references if name in scope.sources]
        if not references:
            return result

        first_name = references[0]
        result[first_name] = dict(self._intrinsic_source_nullability(scope, first_name))

        # Sources nullified by an outer join side; their NOT NULL columns can
        # only be recovered through WHERE/HAVING guarantees.
        nullified_sources: set[str] = set()

        for join in select.args.get("joins") or []:
            source_name = join.alias_or_name
            if not source_name or source_name not in scope.sources:
                # Parenthesized join groups (a JOIN (b JOIN c ...)) expose no
                # single source here; their member sources were already
                # registered via the FROM traversal. Mark them according to
                # the enclosing join's side below.
                continue

            intrinsic = self._intrinsic_source_nullability(scope, source_name)

            side = join.side
            kind = join.kind
            if kind in ("SEMI", "ANTI"):
                # SEMI/ANTI joins never add NULL-padded rows: keep intrinsic.
                result[source_name] = dict(intrinsic)
            elif side == "LEFT":
                result[source_name] = {column: True for column in intrinsic}
                nullified_sources.add(source_name)
            elif side == "RIGHT":
                result[source_name] = dict(intrinsic)
                nullified_sources.update(result.keys() - {source_name})
                for name in result.keys() - {source_name}:
                    result[name] = {column: True for column in result[name]}
            elif side == "FULL":
                result[source_name] = {column: True for column in intrinsic}
                nullified_sources.add(source_name)
                for name in list(result.keys() - {source_name}):
                    result[name] = {column: True for column in result[name]}
                    nullified_sources.add(name)
            else:
                # INNER / CROSS joins preserve the intrinsic nullability.
                result[source_name] = dict(intrinsic)

        # Recover NOT NULL columns of outer-join-nullified sources when a
        # WHERE/HAVING predicate guarantees that *some* column of the source
        # is non-NULL, which excludes every NULL-padded row of that source.
        # Recovery also propagates inward along LEFT-join chains: if the
        # right side of a LEFT JOIN is present, its ON equality matched, so
        # every equality operand on another source is non-NULL and that
        # partner row is present too. FULL/RIGHT joins never participate:
        # knowing one side is present does not prove the other side is.
        guaranteed_sources = {
            table
            for table, _ in self._guaranteed_columns
            if table in nullified_sources
        }
        on_partners = self._join_on_partners(select)
        left_join_names = {
            join.alias_or_name
            for join in select.args.get("joins") or []
            if join.side == "LEFT" and join.alias_or_name
        }

        # ``depends_on[x]`` is the set of *nullified* sources that LEFT-joined
        # source x references in its ON equality: x can only be present when
        # those partners are present too. Partners on the preserved (left)
        # side need no recovery, so they are excluded here. Starting from the
        # directly guaranteed sources, repeatedly recover every nullified
        # source whose nullified dependencies are all satisfied.
        depends_on: dict[str, set[str]] = {
            source_name: {
                partner
                for partner in on_partners.get(source_name, set())
                if partner in nullified_sources
            }
            for source_name in left_join_names & nullified_sources
        }

        worklist = list(guaranteed_sources)
        while worklist:
            current = worklist.pop()
            for source_name, required in depends_on.items():
                if source_name in guaranteed_sources:
                    continue
                if required <= guaranteed_sources:
                    guaranteed_sources.add(source_name)
                    worklist.append(source_name)

        for source_name in nullified_sources:
            source_columns = result.get(source_name, {})
            if source_name in guaranteed_sources:
                # The predicate (or a transitive ON equality) proves the row
                # is not NULL-padded, so intrinsic NOT NULL columns apply.
                intrinsic = self._intrinsic_source_nullability(scope, source_name)
                recovered = dict(source_columns)
                for column in intrinsic:
                    if not intrinsic[column]:
                        recovered[column] = False
                result[source_name] = recovered

        return result

    def _intrinsic_source_nullability(
        self, scope: Scope, source_name: str
    ) -> dict[str, bool]:
        """Nullability of a source's columns ignoring outer-join padding."""
        source = scope.sources.get(source_name)

        if isinstance(source, Scope):
            return self._scope_source_nullability(source)

        if isinstance(source, exp.Table):
            return self._table_source_nullability(source)

        # Unknown source kind (e.g. UDTF): everything is nullable.
        return {}

    def _scope_source_nullability(self, source: Scope) -> dict[str, bool]:
        expression = source.expression

        if isinstance(expression, exp.Lateral):
            # A LATERAL subquery's columns come from its inner query scope.
            inner = seq_get(source.subquery_scopes, 0)
            if inner is not None:
                return self._scope_source_nullability(inner)
            return {name: True for name in expression.alias_column_names}

        if isinstance(expression, exp.UDTF):
            return {name: True for name in expression.alias_column_names}

        outputs = self._scope_outputs.get(id(source))
        if outputs is None:
            # Recursive CTE references can point at a scope that has not been
            # analyzed yet. Be conservative.
            return {}

        names = [select.alias_or_name for select in expression.selects if not select.is_star]
        if len(names) != len(outputs):
            # Defensive fallback for unusual source shapes.
            return {name: True for name in names}
        return dict(zip(names, outputs))

    def _table_source_nullability(self, table: exp.Table) -> dict[str, bool]:
        columns = self.schema.column_names(table)
        parts = tuple(part.name for part in table.parts)

        not_null = self._lookup_not_null(parts)
        return {column: column not in not_null for column in columns}

    def _lookup_not_null(self, table_parts: tuple[str, ...]) -> frozenset[str]:
        # Match by suffix so both "orders" and "db.orders" keys resolve a
        # physical "db.orders" table, mirroring the schema's own trie lookup.
        for key, columns in self.not_null_columns.items():
            if key and table_parts[-len(key) :] == key:
                return columns
        return frozenset()

    def _analyze_set_operation_scope(
        self, scope: Scope, set_operation: exp.SetOperation
    ) -> None:
        arm_maps: list[dict[str, bool]] = []
        for arm_scope in scope.set_operation_scopes:
            arm_outputs = self._scope_outputs.get(id(arm_scope))
            if arm_outputs is None:
                arm_outputs = self._fallback_scope_outputs(arm_scope)
            arm_expression = arm_scope.expression
            arm_names = [
                select.alias_or_name
                for select in arm_expression.selects
                if not select.is_star
            ]
            if len(arm_names) != len(arm_outputs):
                arm_maps.append({})
            else:
                arm_maps.append(dict(zip(arm_names, arm_outputs)))

        if set_operation.args.get("by_name"):
            # Columns are aligned by name; a column missing from one arm is
            # NULL-filled, so only columns present and non-NULL in *every*
            # arm are non-NULL overall.
            output_names = list(dict.fromkeys(set_operation.named_selects))
            result = [
                any(name not in arm or arm[name] for arm in arm_maps)
                for name in output_names
            ]
        else:
            # Positional alignment: every arm must declare the column
            # non-NULL at the same position.
            if len(arm_maps) == 2 and arm_maps[0] and arm_maps[1]:
                width = min(len(arm_maps[0]), len(arm_maps[1]))
                result = [
                    arm_maps[0][name0] or arm_maps[1][name1]
                    for (name0, _), (name1, _) in zip(
                        arm_maps[0].items(), arm_maps[1].items()
                    )
                ][:width]
            else:
                width = len(arm_maps[0]) if arm_maps else 0
                result = [True] * width

        self._scope_outputs[id(scope)] = result

    # ------------------------------------------------------------------
    # WHERE / HAVING guarantees
    # ------------------------------------------------------------------

    def _collect_guarantees(self, predicate: exp.Expr, scope: Scope) -> None:
        """
        Collect expressions that are guaranteed non-NULL when ``predicate``
        evaluates to TRUE. The result is stored as two sets:

        - ``_guaranteed_columns``: qualified ``(source, column)`` pairs
        - ``_guaranteed_node_ids``: ids of arbitrary non-column expressions

        Parenthesized wrappers are handled later by the expression-level
        nullability check itself.
        """
        guaranteed = self._null_rejecting_expressions(predicate, scope)
        for expression in guaranteed:
            self._guaranteed_node_ids.add(id(expression))
            if isinstance(expression, exp.Column) and expression.table:
                self._guaranteed_columns.add((expression.table, expression.name))

    def _null_rejecting_expressions(
        self, predicate: exp.Expr, scope: Scope
    ) -> list[exp.Expr]:
        """
        Return the expressions that are guaranteed non-NULL on every row for
        which ``predicate`` is TRUE.
        """
        if isinstance(predicate, exp.Paren):
            return self._null_rejecting_expressions(predicate.this, scope)

        if isinstance(predicate, exp.And):
            return self._null_rejecting_expressions(
                predicate.left, scope
            ) + self._null_rejecting_expressions(predicate.right, scope)

        if isinstance(predicate, exp.Or):
            left = self._null_rejecting_expressions(predicate.left, scope)
            right = self._null_rejecting_expressions(predicate.right, scope)
            # An expression is guaranteed only if BOTH branches reject NULLs
            # on it. Match structurally by identity first, then by SQL text.
            right_ids = {id(expression) for expression in right}
            right_sql = {expression.sql(self.dialect) for expression in right}
            return [
                expression
                for expression in left
                if id(expression) in right_ids
                or expression.sql(self.dialect) in right_sql
            ]

        if isinstance(predicate, exp.Not):
            inner = predicate.this
            if isinstance(inner, exp.Paren):
                inner = inner.this
            if isinstance(inner, exp.Not):
                # NOT NOT p is equivalent to p. Strip both negations together
                # and recurse on the underlying predicate, so e.g.
                # NOT (NOT (x IS NOT NULL)) still guarantees x while
                # NOT (NOT (x IS NULL)) guarantees nothing.
                inner_inner = inner.this
                if isinstance(inner_inner, exp.Paren):
                    inner_inner = inner_inner.this
                return self._null_rejecting_expressions(inner_inner, scope)
            if isinstance(inner, exp.Is) and isinstance(inner.expression, exp.Null):
                # NOT(x IS NULL) is the canonical "x IS NOT NULL" form.
                return [inner.this]
            # Any other negated predicate (NOT IN, NOT BETWEEN, ...) is not
            # reliably NULL-rejecting, so nothing is inferred.
            return []

        if isinstance(predicate, exp.Is):
            # x IS NOT NULL has already been normalized to NOT(x IS NULL).
            # A plain x IS NULL does not reject NULLs.
            return []

        if isinstance(predicate, _NULL_REJECTING_COMPARISONS):
            return [operand for operand in (predicate.left, predicate.right) if operand]

        if isinstance(predicate, exp.In):
            # x IN (...) rejects NULL x, including subquery and list forms.
            return [predicate.this]

        if isinstance(predicate, exp.Between):
            # x BETWEEN low AND high rejects NULL x.
            return [predicate.this]

        if isinstance(predicate, exp.Column):
            # A bare boolean column in WHERE/HAVING must be TRUE, hence non-NULL.
            return [predicate]

        return []

    # ------------------------------------------------------------------
    # Expression-level nullability
    # ------------------------------------------------------------------

    def _is_nullable(
        self,
        expression: exp.Expr,
        scope: Scope,
        source_nullability: dict[str, dict[str, bool]],
    ) -> bool:
        # Use None as the "not cached" sentinel so that a cached False result
        # is not mistaken for a cache miss.
        cached = self._memo.get(id(expression))
        if cached is not None:
            return cached

        nullable = self._compute_nullable(expression, scope, source_nullability)
        self._memo[id(expression)] = nullable
        return nullable

    def _compute_nullable(
        self,
        expression: exp.Expr,
        scope: Scope,
        source_nullability: dict[str, dict[str, bool]],
    ) -> bool:
        # Predicates guaranteed by WHERE/HAVING are non-NULL on retained rows.
        # Unwrap parentheses: WHERE predicates are commonly wrapped in them.
        guaranteed_node = expression
        while isinstance(guaranteed_node, exp.Paren):
            guaranteed_node = guaranteed_node.this
        if id(guaranteed_node) in self._guaranteed_node_ids:
            return False

        if isinstance(expression, exp.Alias):
            return self._is_nullable(expression.this, scope, source_nullability)

        if isinstance(expression, exp.Paren):
            return self._is_nullable(expression.this, scope, source_nullability)

        if isinstance(expression, (exp.Literal, exp.Boolean)):
            return False

        if isinstance(expression, exp.Null):
            return True

        if isinstance(expression, exp.Column):
            return self._column_nullable(expression, scope, source_nullability)

        if isinstance(expression, exp.Coalesce):
            # COALESCE is non-NULL if at least one argument is non-NULL.
            # A guaranteed argument (e.g. a column proven non-NULL by WHERE)
            # counts even when it is nested inside an arithmetic expression.
            arguments = [expression.this, *expression.expressions]
            return all(
                self._is_nullable(argument, scope, source_nullability)
                for argument in arguments
                if argument
            )

        if isinstance(expression, exp.Case):
            return self._case_nullable(expression, scope, source_nullability)

        if isinstance(expression, exp.Cast):
            # A regular CAST never introduces NULLs (unlike TRY_CAST), so it
            # inherits the nullability of its input.
            if isinstance(expression, exp.TryCast):
                return True
            return self._is_nullable(expression.this, scope, source_nullability)

        if isinstance(expression, exp.Window):
            return self._window_nullable(expression, scope, source_nullability)

        if isinstance(expression, exp.Filter):
            # agg(...) FILTER (WHERE ...): SUM/MIN/MAX/AVG return NULL on a
            # group where no row passes the filter, so only COUNT-family
            # aggregates keep their non-NULL guarantee.
            if isinstance(expression.this, _NON_NULL_AGG_FUNCS):
                return False
            return True

        if isinstance(expression, exp.AggFunc):
            return self._aggregate_nullable(expression, scope, source_nullability)

        if isinstance(expression, exp.Distinct):
            # DISTINCT never changes the nullability of its input.
            return self._all_arguments_nullable(
                expression.expressions, scope, source_nullability
            )

        if isinstance(expression, exp.Exists):
            # EXISTS always yields a boolean, never NULL.
            return False

        if isinstance(expression, exp.Subquery):
            # Scalar subqueries can yield zero rows and thus NULL.
            return True

        if isinstance(expression, _NON_NULL_ARITHMETIC):
            return self._binary_operand_nullable(
                expression, scope, source_nullability
            )

        if isinstance(expression, exp.Predicate):
            return self._predicate_nullable(expression, scope, source_nullability)

        if isinstance(expression, exp.Neg):
            return self._is_nullable(expression.this, scope, source_nullability)

        # Named functions with known nullability behavior.
        if isinstance(expression, exp.If):
            return self._if_nullable(expression, scope, source_nullability)

        if isinstance(expression, exp.Extract):
            return self._is_nullable(expression.this, scope, source_nullability)

        # Unknown functions and any other expression: conservatively nullable.
        return True

    def _column_nullable(
        self,
        column: exp.Column,
        scope: Scope,
        source_nullability: dict[str, dict[str, bool]],
    ) -> bool:
        if (column.table, column.name) in self._guaranteed_columns:
            return False

        if (column.table, column.name) in self._nullified_grouping_columns:
            # ROLLUP/CUBE/GROUPING SETS synthesize NULL for this grouping key.
            return True

        columns = source_nullability.get(column.table)
        if columns is not None and column.name in columns:
            return columns[column.name]

        # Correlated columns reference outer scopes: resolve them there.
        outer = self._resolve_outer_column(column, scope)
        if outer is not None:
            return outer

        return True

    def _resolve_outer_column(self, column: exp.Column, scope: Scope) -> bool | None:
        parent = scope.parent
        while parent is not None:
            source = parent.sources.get(column.table)
            if isinstance(source, Scope):
                outputs = self._scope_outputs.get(id(source))
                if outputs is not None:
                    names = [
                        select.alias_or_name
                        for select in source.expression.selects
                        if not select.is_star
                    ]
                    if column.name in names:
                        return outputs[names.index(column.name)]
                return True
            if isinstance(source, exp.Table):
                return self._table_source_nullability(source).get(column.name, True)
            parent = parent.parent
        return None

    def _case_nullable(
        self,
        case: exp.Case,
        scope: Scope,
        source_nullability: dict[str, dict[str, bool]],
    ) -> bool:
        # Every THEN branch and the ELSE (or implicit NULL when absent) must be
        # non-NULL for the CASE result to be non-NULL.
        branches = [branch.args.get("true") for branch in case.args.get("ifs") or []]
        default = case.args.get("default")
        if default is None:
            # CASE without ELSE yields NULL when no WHEN matches.
            return True
        branches.append(default)

        for branch in branches:
            if branch is None or self._is_nullable(branch, scope, source_nullability):
                return True
        return False

    def _if_nullable(
        self,
        if_expression: exp.If,
        scope: Scope,
        source_nullability: dict[str, dict[str, bool]],
    ) -> bool:
        true_value = if_expression.args.get("true")
        false_value = if_expression.args.get("false")
        # IF without an explicit false branch yields NULL when the condition
        # is not met.
        if true_value is None or false_value is None:
            return True
        return self._is_nullable(
            true_value, scope, source_nullability
        ) or self._is_nullable(false_value, scope, source_nullability)

    def _window_nullable(
        self,
        window: exp.Window,
        scope: Scope,
        source_nullability: dict[str, dict[str, bool]],
    ) -> bool:
        func = window.this
        if isinstance(func, exp.Filter):
            # agg(...) FILTER (WHERE ...) OVER (...) keeps the window's shape.
            func = func.this
        if isinstance(func, _NON_NULL_WINDOW_FUNCS):
            return False
        if isinstance(func, exp.Count):
            # COUNT as a window function still emits a row count, never NULL.
            return False
        # Any other window function (SUM/MIN/.../unknown) is nullable.
        return True

    def _aggregate_nullable(
        self,
        aggregate: exp.AggFunc,
        scope: Scope,
        source_nullability: dict[str, dict[str, bool]],
    ) -> bool:
        if isinstance(aggregate, _NON_NULL_AGG_FUNCS):
            return False

        grouped = self._is_grouped_aggregate(scope)

        if isinstance(aggregate, _GROUPED_NON_NULL_AGG_FUNCS):
            if not grouped:
                # A scalar aggregate over an empty table yields NULL.
                return True
            return self._aggregate_arguments_nullable(aggregate, scope, source_nullability)

        # Unknown aggregate: conservatively nullable.
        return True

    def _is_grouped_aggregate(self, scope: Scope) -> bool:
        expression = scope.expression
        if not isinstance(expression, exp.Select):
            return False
        group = expression.args.get("group")
        if not group:
            return False
        # GROUPING SETS / CUBE / ROLLUP may emit an overall row whose grouping
        # columns are NULL, but aggregate values themselves remain non-NULL on
        # emitted groups; the aggregate rule still applies.
        return True

    def _all_arguments_nullable(
        self,
        arguments: t.Iterable[exp.Expr],
        scope: Scope,
        source_nullability: dict[str, dict[str, bool]],
    ) -> bool:
        for argument in arguments:
            if self._is_nullable(argument, scope, source_nullability):
                return True
        return False

    def _aggregate_arguments_nullable(
        self,
        aggregate: exp.AggFunc,
        scope: Scope,
        source_nullability: dict[str, dict[str, bool]],
    ) -> bool:
        arguments = list(aggregate.iter_expressions())
        if not arguments:
            return False
        for argument in arguments:
            # FILTER / DISTINCT / ORDER wrappers are not value arguments.
            if isinstance(argument, (exp.Filter, exp.Distinct, exp.Ordered)):
                continue
            if self._is_nullable(argument, scope, source_nullability):
                return True
        return False

    def _binary_operand_nullable(
        self,
        binary: exp.Binary,
        scope: Scope,
        source_nullability: dict[str, dict[str, bool]],
    ) -> bool:
        for operand in (binary.left, binary.right):
            if operand is not None and self._is_nullable(
                operand, scope, source_nullability
            ):
                return True
        return False

    def _predicate_nullable(
        self,
        predicate: exp.Predicate,
        scope: Scope,
        source_nullability: dict[str, dict[str, bool]],
    ) -> bool:
        if isinstance(predicate, (exp.Is, exp.NullSafeEQ, exp.NullSafeNEQ)):
            # These predicates always yield TRUE/FALSE, never NULL.
            return False

        if isinstance(predicate, exp.Binary):
            return self._binary_operand_nullable(
                predicate, scope, source_nullability
            )

        if isinstance(predicate, exp.Not):
            return self._is_nullable(predicate.this, scope, source_nullability)

        # Unknown predicate shape: conservatively nullable.
        return True

    # ------------------------------------------------------------------
    # VALUES
    # ------------------------------------------------------------------

    def _values_nullability(self, values: exp.Values) -> list[bool]:
        rows = values.expressions
        if not rows:
            return []

        width = max(
            (len(row.expressions) for row in rows if isinstance(row, exp.Tuple)),
            default=0,
        )
        result: list[bool] = []
        for index in range(width):
            nullable = False
            for row in rows:
                if not isinstance(row, exp.Tuple) or index >= len(row.expressions):
                    nullable = True
                    break
                cell = row.expressions[index]
                if isinstance(cell, (exp.Literal, exp.Boolean)):
                    continue
                if isinstance(cell, exp.Null):
                    nullable = True
                    break
                # Complex expressions in VALUES: evaluate on a throwaway
                # memo without any sources.
                if self._standalone_expression_nullable(cell):
                    nullable = True
                    break
            result.append(nullable)
        return result

    def _standalone_expression_nullable(self, expression: exp.Expr) -> bool:
        if isinstance(expression, (exp.Literal, exp.Boolean)):
            return False
        if isinstance(expression, exp.Null):
            return True
        if isinstance(expression, exp.Paren):
            return self._standalone_expression_nullable(expression.this)
        if isinstance(expression, exp.Neg):
            return self._standalone_expression_nullable(expression.this)
        if isinstance(expression, exp.Cast) and not isinstance(expression, exp.TryCast):
            return self._standalone_expression_nullable(expression.this)
        if isinstance(expression, _NON_NULL_ARITHMETIC) and isinstance(
            expression, exp.Binary
        ):
            return self._standalone_expression_nullable(
                expression.left
            ) or self._standalone_expression_nullable(expression.right)
        # Anything else (column references, functions, ...) is unknown here.
        return True


def _values_output_names(values: exp.Values) -> list[str]:
    alias_columns = values.alias_column_names
    if alias_columns:
        return list(alias_columns)
    width = max(
        (len(row.expressions) for row in values.expressions if isinstance(row, exp.Tuple)),
        default=0,
    )
    return [f"col{index}" for index in range(width)]
