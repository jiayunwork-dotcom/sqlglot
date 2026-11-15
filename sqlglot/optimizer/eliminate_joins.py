from __future__ import annotations

import typing as t

from sqlglot import expressions as exp
from sqlglot.dialects.dialect import DialectType
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import Scope, find_in_scope, traverse_scope

if t.TYPE_CHECKING:
    from sqlglot._typing import E
    from sqlglot.schema import Schema

# Output columns of a scope that are guaranteed unique per row, mapped to the
# expression that produces them. An empty mapping means "no uniqueness guarantee".
UniqueOutputs = dict[str, exp.Expr]

# Collation used by uncollated text comparisons. The modeled engines all use
# binary/codepoint ordering by default; SQLite's NOCASE/RTRIM only apply when
# the query says so explicitly.
DEFAULT_COLLATION = "BINARY"
_KNOWN_COLLATIONS = frozenset({"BINARY", "NOCASE", "RTRIM"})


def eliminate_joins(
    expression: E,
    dialect: DialectType = None,
    schema: dict[str, object] | Schema | None = None,
) -> E:
    """
    Remove unused joins from an expression.

    A `LEFT JOIN` whose right-hand columns are never referenced outside of its
    own `ON` clause is only removed when the join is *provably* one-to-one for
    every left row, i.e. at most one right row can match each left row. That
    requires both:

      1. a uniqueness guarantee on the joined right-side output columns (the
         right query is `DISTINCT` / grouped on its full key / returns at most
         one row), visible through any number of column-forwarding derived
         tables and CTEs; and
      2. an equality predicate between each unique output and a left-side
         expression whose comparison semantics cannot identify distinct right
         values. Comparisons across non-coercible type families (e.g. integer
         vs. text) and incompatible collations (e.g. `BINARY`-deduped text
         compared `COLLATE NOCASE`) are rejected unless the type information is
         missing, in which case the structural check alone is used just like
         the historical behavior.

    Cross joins on at-most-one-row subqueries are also removed.

    Example:
        >>> import sqlglot
        >>> sql = "SELECT x.a FROM x LEFT JOIN (SELECT DISTINCT y.b FROM y) AS y ON x.b = y.b"
        >>> expression = sqlglot.parse_one(sql)
        >>> eliminate_joins(expression).sql()
        'SELECT x.a FROM x'

    Args:
        expression: expression to optimize
        dialect: dialect deciding comparison semantics (coercion, collations).
            When `schema` is given without `dialect`, types are inferred with
            the generic dialect. Type evidence is only used to *reject* unsafe
            eliminations; queries without type information keep the structural
            behavior.
        schema: optional schema used to infer comparison types on a private copy
            when the incoming AST has no type annotations. The caller's AST and
            this schema are never mutated.

    Returns:
        The optimized expression
    """
    analyzer = _JoinEliminator(expression, dialect=dialect, schema=schema)
    for scope in traverse_scope(expression):
        joins: list[exp.Join] = scope.expression.args.get("joins", [])
        if not joins:
            continue

        # If any columns in this scope aren't qualified, it's hard to determine if a join isn't used.
        # It's probably possible to infer this from the outputs of derived tables.
        # But for now, let's just skip this rule.
        if scope.unqualified_columns:
            continue

        # Reverse the joins so we can remove chains of unused joins
        for join in reversed(joins):
            if join.is_semi_or_anti_join or join.args.get("nearest"):
                continue

            alias = join.alias_or_name
            if _should_eliminate_join(scope, join, alias, analyzer):
                join.pop()
                scope.remove_source(alias)

    return expression


def _should_eliminate_join(
    scope: Scope, join: exp.Join, alias: str, analyzer: _JoinEliminator
) -> bool:
    inner_source = scope.sources.get(alias)
    if not isinstance(inner_source, Scope) or _join_is_used(scope, join, alias):
        return False

    if not join.args.get("on"):
        return _has_single_output_row(inner_source)

    if join.side != "LEFT":
        return False

    return analyzer.is_one_to_one(join, inner_source)


def _join_is_used(scope: Scope, join: exp.Join, alias: str) -> bool:
    # We need to find all columns that reference this join.
    # But columns in the ON clause shouldn't count.
    on: exp.Expr | None = join.args.get("on")
    if on is not None:
        on_clause_columns: set[int] = {id(column) for column in on.find_all(exp.Column)}
    else:
        on_clause_columns = set()
    return any(
        column for column in scope.source_columns(alias) if id(column) not in on_clause_columns
    )


class _JoinEliminator:
    """
    Collects the evidence needed to decide join eliminations for one invocation.

    All caches are keyed by node identity and live for a single
    `eliminate_joins` call, so conclusions never leak across calls or between
    joins sharing a CTE. Structural facts about a right-side scope (its unique
    outputs) are cached, while the comparison-safety verdict always stays local
    to the join being analyzed.
    """

    def __init__(
        self,
        expression: exp.Expr,
        dialect: DialectType = None,
        schema: dict[str, object] | Schema | None = None,
    ) -> None:
        self.default_collation = DEFAULT_COLLATION
        self._unique_outputs_cache: dict[int, UniqueOutputs] = {}
        self._type_lookup: t.Callable[[exp.Expr], exp.DataType | None] = self._build_type_lookup(
            expression, dialect, schema
        )

    # ------------------------------------------------------------------ types
    def _build_type_lookup(
        self,
        expression: exp.Expr,
        dialect: DialectType,
        schema: dict[str, object] | Schema | None,
    ) -> t.Callable[[exp.Expr], exp.DataType | None]:
        def effective_type(node: exp.Expr) -> exp.DataType | None:
            # `COLLATE` does not change a value's type, but the annotator does
            # not assign Collate nodes a type, so read through the wrapper.
            dtype = node.type
            if dtype is None and isinstance(node, exp.Collate):
                dtype = node.this.type
            return dtype

        # Reuse types already carried by the (pre-annotated) AST only when it
        # contains *usable* types: earlier pipeline rules may leave a few
        # columns partially annotated as UNKNOWN, which must not prevent the
        # fully-inferred private annotation below from running.
        if any(
            node.type is not None and not node.type.is_type(exp.DType.UNKNOWN)
            for node in expression.find_all(exp.Column)
        ):
            return effective_type

        # Without a schema there is nothing meaningful to infer; stay structural.
        if schema is None:
            return effective_type

        # Annotate a private deep copy so the caller's AST, its types and its
        # metadata are untouched. The copy has identical structure, so nodes are
        # matched structurally (by their position in a BFS walk) rather than by
        # identity: `join_condition` analyzes a copy of the ON clause, whose
        # nodes never appear in the original tree.
        from sqlglot.optimizer.annotate_types import annotate_types

        annotated = annotate_types(expression.copy(), schema=schema, dialect=dialect)
        type_by_position: list[exp.DataType | None] = [node.type for node in annotated.walk()]
        original_nodes = list(expression.walk())
        type_by_index: dict[int, exp.DataType | None] = {
            id(node): dtype for node, dtype in zip(original_nodes, type_by_position)
        }
        # Memo for detached copies (e.g. the ON-clause copy in
        # join_condition): map their identity to the structurally identical
        # original node once instead of rescanning the tree.
        detached_originals: dict[int, exp.Expr] = {}

        def lookup(node: exp.Expr) -> exp.DataType | None:
            dtype = effective_type(node)
            if dtype is not None:
                return dtype
            dtype = type_by_index.get(id(node))
            if dtype is not None:
                return dtype
            original = detached_originals.get(id(node))
            if original is None:
                original = next(
                    (candidate for candidate in original_nodes if candidate == node), None
                )
                if original is None:
                    return None
                detached_originals[id(node)] = original
            return effective_type(original) or type_by_index.get(id(original))

        return lookup

    def is_one_to_one(self, join: exp.Join, inner_scope: Scope) -> bool:
        unique_outputs = self._unique_outputs(inner_scope)
        if not unique_outputs:
            return False

        source_keys, join_keys, _ = join_condition(join)
        if not join_keys:
            return False

        # OR conditions (DNF) emit one key per disjunct: a provably
        # one-to-one OR repeats the same equalities in every disjunct, so
        # deduplicate on the structural hash.
        pairs: dict[int, tuple[exp.Expr, exp.Expr]] = {}
        for join_key, source_key in zip(join_keys, source_keys):
            pairs.setdefault(hash(join_key), (join_key, source_key))

        join_alias = join.alias_or_name
        matched_outputs: set[str] = set()
        for join_key, source_key in pairs.values():
            output_name = self._output_column_name(join_key, join_alias)
            if output_name is None:
                # An equality on a non-column right expression (e.g. a CAST or
                # an arithmetic projection) is just an additional filter that
                # can only narrow the match and never fans the join out.
                continue
            if output_name in matched_outputs:
                return False
            if output_name not in unique_outputs:
                return False
            # The left operand must be a bare column. A conversion (CAST,
            # TRY_CAST, arithmetic, ...) can map several distinct left values
            # onto the same right value and fan the join out. A literal/NULL
            # pins the unique output to one value, which is itself a
            # one-to-one guarantee for that output.
            if isinstance(source_key, exp.Literal) or _is_null(source_key):
                matched_outputs.add(output_name)
                continue
            if not self._is_plain_key(source_key):
                return False
            if not self._comparison_is_injective(join_key, source_key, unique_outputs[output_name]):
                return False
            matched_outputs.add(output_name)

        # Every unique right output must be covered by an injective column
        # equality or pinned by a constant equality.
        return bool(matched_outputs) and matched_outputs == set(unique_outputs)

    def _is_plain_key(self, expression: exp.Expr) -> bool:
        return isinstance(expression, exp.Column) or (
            isinstance(expression, exp.Collate) and isinstance(expression.this, exp.Column)
        )

    def _output_column_name(self, join_key: exp.Expr, join_alias: str) -> str | None:
        # The join key must be a bare output column of the joined source (an
        # optional COLLATE wrapper is fine). Anything else, e.g. CAST(y.b AS
        # INT), can map distinct right values onto the same key and is unsafe.
        expression = join_key.this if isinstance(join_key, exp.Collate) else join_key
        if not isinstance(expression, exp.Column) or expression.table != join_alias:
            return None
        return expression.name

    def _comparison_is_injective(
        self, join_key: exp.Expr, source_key: exp.Expr, unique_expression: exp.Expr
    ) -> bool:
        # Types are only consulted when evidence exists on both sides. With
        # unknown types we fall back to the purely structural decision, which
        # matches the historical rule and keeps SQL-only calls working.
        join_type = self._known_type(join_key) or self._known_type(unique_expression)
        source_type = self._known_type(source_key)
        if (
            join_type
            and source_type
            and not self._types_compare_injectively(join_type, source_type)
        ):
            return False

        # Text comparisons additionally require matching collations: a join
        # under a coarser collation than the right-side DISTINCT/GROUP BY
        # identifies distinct right rows.
        collation_safe = not (
            join_type and source_type and join_type.is_type(*exp.DataType.TEXT_TYPES)
        ) or self._collations_compatible(join_key, source_key, unique_expression)

        return collation_safe

    def _known_type(self, node: exp.Expr) -> exp.DataType | None:
        dtype = self._type_lookup(node)
        if dtype is None or dtype.is_type(exp.DType.UNKNOWN):
            # A derived-table output produced by an unannotated wrapper
            # expression (e.g. `col COLLATE NOCASE`, whose operand is still
            # typed) keeps the operand's type.
            expression = node.this if isinstance(node, exp.Collate) else None
            if isinstance(expression, exp.Column):
                operand_type = self._type_lookup(expression)
                if operand_type and not operand_type.is_type(exp.DType.UNKNOWN):
                    return operand_type
            return None
        return dtype

    def _types_compare_injectively(
        self, join_type: exp.DataType, source_type: exp.DataType
    ) -> bool:
        # Same type family compares without cross-type coercion: distinct
        # values of the same family stay distinct under equality.
        for family in (
            exp.DataType.INTEGER_TYPES,
            exp.DataType.FLOAT_TYPES,
            exp.DataType.TEXT_TYPES,
            exp.DataType.BINARY_TYPES,
            exp.DataType.TEMPORAL_TYPES,
            _BOOLEAN_TYPES,
        ):
            if join_type.this in family and source_type.this in family:
                return True

        # Fixed-precision decimals only collapse values when precisions differ;
        # within the exact-decimal family equality is identity-preserving.
        if join_type.this in _DECIMAL_TYPES and source_type.this in _DECIMAL_TYPES:
            return True

        # NULL-typed operands never match, so they can never duplicate a row.
        # Any other combination (e.g. INTEGER vs. TEXT, TEXT vs. DECIMAL)
        # relies on engine-specific implicit coercion that can merge distinct
        # values.
        return join_type.this == exp.DType.NULL or source_type.this == exp.DType.NULL

    def _collations_compatible(
        self, join_key: exp.Expr, source_key: exp.Expr, unique_expression: exp.Expr
    ) -> bool:
        join_collation = _collation_name(join_key)
        source_collation = _collation_name(source_key)
        if join_collation is not None and source_collation is not None:
            if join_collation != source_collation:
                # Explicitly different collations make the equality semantics
                # engine-defined and not provably injective.
                return False
            effective_join_collation = join_collation
        else:
            effective_join_collation = join_collation or source_collation or self.default_collation

        # The right-side DISTINCT/GROUP BY guarantees uniqueness under its own
        # collation. The join is injective when it distinguishes at least as
        # finely as that uniqueness collation: either the same collation, or
        # BINARY (the codepoint-identical collation, finer than every built-in
        # SQLite collation). A coarser join collation (e.g. NOCASE against
        # BINARY-deduped text) identifies distinct right rows; an unrecognized
        # collation cannot be ordered against BINARY and is left alone.
        unique_collation = _collation_name(unique_expression) or self.default_collation
        if effective_join_collation == unique_collation:
            return True
        return effective_join_collation == "BINARY" and unique_collation in _KNOWN_COLLATIONS

    # ------------------------------------------------------------- uniqueness
    def _unique_outputs(self, scope: Scope) -> UniqueOutputs:
        cache_key = id(scope)
        cached = self._unique_outputs_cache.get(cache_key)
        if cached is not None:
            return cached

        unique_outputs = self._compute_unique_outputs(scope)
        self._unique_outputs_cache[cache_key] = unique_outputs
        return unique_outputs

    def _compute_unique_outputs(self, scope: Scope) -> UniqueOutputs:
        expression = scope.expression
        if not isinstance(expression, exp.Select):
            # Set operations and other query forms are handled conservatively.
            return {}

        direct = _direct_unique_outputs(scope)
        if direct:
            return direct

        # No uniqueness at this level: look through pure column-forwarding
        # derived tables/CTEs (SELECT t.a, ... FROM (unique_query) t) until a
        # real uniqueness guarantee is found. Each output is traced by name
        # through the projections, so aliases and shadowing stay correct.
        return self._forwarded_unique_outputs(scope)

    def _forwarded_unique_outputs(self, scope: Scope) -> UniqueOutputs:
        if not isinstance(scope.expression, exp.Select):
            return {}

        # A forwarding scope has exactly one non-lateral source and no joins of
        # its own. Each output traced from an inner unique output must be a bare
        # column of that source; extra projections (constants, aggregates over a
        # single-row source, ...) simply don't participate in the trace. CTE
        # sources are skipped so the trace continues from the CTE's own scope.
        sources = {
            name: source
            for name, source in scope.sources.items()
            if name not in scope.lateral_sources and name not in scope.cte_sources
        }
        if len(sources) != 1 or scope.expression.args.get("joins"):
            return {}
        source_name, inner_scope = next(iter(sources.items()))
        if not isinstance(inner_scope, Scope):
            return {}

        inner_outputs = self._unique_outputs(inner_scope)
        if not inner_outputs:
            return {}

        forwarded: UniqueOutputs = {}
        for select in scope.expression.selects:
            output = select.unalias()
            if not isinstance(output, exp.Column) or output.table != source_name:
                continue
            inner_expression = inner_outputs.get(output.name)
            if inner_expression is None:
                # A bare source column whose provenance is not unique cannot
                # itself be unique, so no output of this scope is trustworthy.
                return {}
            forwarded[select.alias_or_name] = inner_expression

        return forwarded


_BOOLEAN_TYPES = {exp.DType.BOOLEAN}
_DECIMAL_TYPES = {
    exp.DType.DECIMAL,
    exp.DType.BIGDECIMAL,
    exp.DType.DECIMAL32,
    exp.DType.DECIMAL64,
    exp.DType.DECIMAL128,
    exp.DType.DECIMAL256,
    exp.DType.UDECIMAL,
    exp.DType.MONEY,
    exp.DType.SMALLMONEY,
}


def _collation_name(expression: exp.Expr) -> str | None:
    if isinstance(expression, exp.Collate):
        return expression.expression.name.upper()
    return None


def _is_null(expression: exp.Expr) -> bool:
    return isinstance(expression, exp.Null)


def _direct_unique_outputs(scope: Scope) -> UniqueOutputs:
    """Unique outputs produced by DISTINCT / GROUP BY / at-most-one-row queries."""
    expr = t.cast(exp.Query, scope.expression)
    if (distinct := expr.args.get("distinct")) is not None and not (
        isinstance(distinct, exp.Distinct) and distinct.args.get("on")
    ):
        return {select.alias_or_name: select.unalias() for select in expr.selects}

    group: exp.Group | None = expr.args.get("group")
    if group is not None:
        grouped_expressions: set[exp.Expr] = set(group.expressions)
        grouped_outputs: set[exp.Expr] = set()

        unique_outputs: UniqueOutputs = {}
        for select in expr.selects:
            output = select.unalias()
            if output in grouped_expressions:
                grouped_outputs.add(output)
                unique_outputs[select.alias_or_name] = output

        # All the grouped expressions must be in the output
        if not grouped_expressions.difference(grouped_outputs):
            return unique_outputs
        return {}

    if _is_limit_1(scope) or _has_single_output_row(scope):
        return {select.alias_or_name: select.unalias() for select in expr.selects}

    return {}


def _has_single_output_row(scope: Scope) -> bool:
    expression = scope.expression
    if not isinstance(expression, exp.Select):
        return False

    args = expression.args
    if any(args.get(key) for key in ("group", "having", "qualify", "limit", "offset")):
        return False
    if args.get("from_") or args.get("where"):
        return all(isinstance(select.unalias(), exp.AggFunc) for select in expression.selects)
    return not any(
        find_in_scope(select, *exp.SET_RETURNING_FUNCTIONS) for select in expression.selects
    )


def _is_limit_1(scope: Scope) -> bool:
    limit = scope.expression.args.get("limit")
    return isinstance(limit, exp.Limit) and limit.expression.this == "1"


def join_condition(join: exp.Join) -> tuple[list[exp.Expr], list[exp.Expr], exp.Expr]:
    """
    Extract the join condition from a join expression.

    Args:
        join (exp.Join)
    Returns:
        tuple[list[str], list[str], exp.Expr]:
            Tuple of (source key, join key, remaining predicate)
    """
    name = join.alias_or_name
    on: exp.Expr = (join.args.get("on") or exp.true()).copy()
    source_key: list[exp.Expr] = []
    join_key: list[exp.Expr] = []

    def extract_condition(condition: exp.Expr) -> None:
        left, right = condition.unnest_operands()
        left_tables = exp.column_table_names(left)
        right_tables = exp.column_table_names(right)

        if name in left_tables and name not in right_tables:
            join_key.append(left)
            source_key.append(right)
            condition.replace(exp.true())
        elif name in right_tables and name not in left_tables:
            join_key.append(right)
            source_key.append(left)
            condition.replace(exp.true())

    # find the join keys
    # SELECT
    # FROM x
    # JOIN y
    #   ON x.a = y.b AND y.b > 1
    #
    # should pull y.b as the join key and x.a as the source key
    if normalized(on):
        on = on if isinstance(on, exp.And) else exp.and_(on, exp.true(), copy=False)

        for condition in on.flatten():
            if isinstance(condition, exp.EQ):
                extract_condition(condition)
    elif normalized(on, dnf=True):
        conditions: list[exp.EQ] = []

        for condition in on.flatten():
            parts = [part for part in condition.flatten() if isinstance(part, exp.EQ)]
            if not conditions:
                conditions = parts
            else:
                temp: list[exp.EQ] = []
                for p in parts:
                    cs = [c for c in conditions if p == c]

                    if cs:
                        temp.append(p)
                        temp.extend(cs)
                conditions = temp

        for condition in conditions:
            extract_condition(condition)

    return source_key, join_key, on
