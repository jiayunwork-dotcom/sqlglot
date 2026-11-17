from __future__ import annotations

import itertools
import logging
import typing as t
from builtins import type as Type
from collections import defaultdict
from enum import Enum, auto
from weakref import ref

from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import find_new_name, mypyc_attr, seq_get

logger = logging.getLogger("sqlglot")

if t.TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlglot._typing import E
    from sqlglot.optimizer.recursive import RecursiveFlow

TRAVERSABLES = (exp.Query, exp.DDL, exp.DML)


# CTE node id -> weak references of every RecursiveRelation binding built for it. The
# registry never owns either the AST or a binding, so scope rebuilds and expression
# copies leave nothing behind; stale weak references are pruned on access.
_recursive_relation_registry: dict[int, list[ref[RecursiveRelation]]] = {}

ROW_LEVEL_AGG_FUNCS = (exp.Count,)

# The node types `Scope._collect` classifies, roughly ordered by frequency. `exp.Query`
# covers subqueries, and `exp.UDTF` covers laterals.
COLLECTIBLE_TYPES = (
    exp.Column,
    exp.Dot,
    exp.Table,
    exp.Query,
    exp.UDTF,
    exp.CTE,
    exp.Star,
    exp.TableColumn,
    exp.JoinHint,
    exp.Comprehension,
)


class ScopeType(Enum):
    ROOT = auto()
    SUBQUERY = auto()
    DERIVED_TABLE = auto()
    CTE = auto()
    SET_OPERATION = auto()
    UDTF = auto()


@mypyc_attr(allow_interpreted_subclasses=True)
class RecursiveRelation:
    """The fixed-point binding of a supported ``WITH RECURSIVE`` CTE.

    A recursive CTE has the shape::

        WITH RECURSIVE r(c1, ..., cC) AS (
            <anchor>      -- non-recursive term
            UNION ALL
            <recursive>   -- references r itself once, in FROM or an INNER JOIN
        )

    The recursive term reads the relation ``r`` while its columns are also being
    defined, so its columns need an identity distinct from both the anchor's output
    columns and any base-table column. This object owns that identity.

    The binding is a finite summary of the recursion, not an unrolling: it records,
    once per recursive output column, which *recursive input* columns (values of the
    same relation at the previous iteration) and which *non-recursive* columns (base
    tables, ordinary CTEs and derived tables visible to the recursive term) feed it.
    Its size is therefore O(C + E) for C column identities and E direct dependencies;
    it never grows with the number of iterations, paths or data rows.

    Attributes:
        name: The CTE / relation name.
        scope: The scope rooted at the CTE's ``UNION ALL`` expression. ``scope.sources``
            maps this name back to the same scope, so column qualification, lineage and
            projection pruning all resolve self-references through this single binding.
        anchor_scope: Scope of the non-recursive (first) ``UNION ALL`` arm.
        recursive_scope: Scope of the recursive (second) ``UNION ALL`` arm.
        columns: Output column names of the relation, in positional order. They are the
            names used by every self-reference inside the recursive term.
        cte: The ``exp.CTE`` node declaring the relation.
        alias_columns: Explicit column list on the CTE, when one was given.
        supported: Whether the relation matches the supported shape. Unsupported
            relations keep ``supported=False``; consumers must use conservative
            behavior and never treat the binding as a complete dependency summary.
    """

    __slots__ = (
        "__weakref__",
        "_flow",
        "alias_columns",
        "anchor_scope",
        "columns",
        "cte",
        "name",
        "recursive_scope",
        "scope",
        "self_reference",
        "supported",
    )

    def __init__(
        self,
        name: str,
        scope: Scope,
        anchor_scope: Scope,
        recursive_scope: Scope,
        columns: list[str],
        cte: exp.CTE,
        alias_columns: list[str],
        supported: bool,
        self_reference: exp.Table | None = None,
    ) -> None:
        self.name = name
        self.scope = scope
        self.anchor_scope = anchor_scope
        self.recursive_scope = recursive_scope
        self.columns = columns
        self.cte = cte
        self.alias_columns = alias_columns
        self.supported = supported
        self.self_reference = self_reference
        # The flow cache is a private slot on this binding; bindings built by different
        # scope traversals share it via the weak registry keyed by the CTE node, so the
        # cache survives a scope rebuild over the same AST without keeping either the AST
        # or the binding alive.
        self._flow: RecursiveFlow | None = None
        relations = _recursive_relation_registry.setdefault(id(cte), [])
        relations.append(ref(self))
        for other in relations:
            cached = other()
            if cached is not None and cached is not self and cached._flow is not None:
                self._flow = cached._flow
                break

    @property
    def flow(self) -> RecursiveFlow:
        """The cached RecursiveFlow summary, computed on first access."""
        if self._flow is None:
            from sqlglot.optimizer.recursive import build_recursive_flow

            self._flow = build_recursive_flow(self)
        return self._flow

    def invalidate_flow(self) -> None:
        """Drop this relation's cached flow, including on earlier bindings for the same CTE."""
        for binding_ref in _recursive_relation_registry.get(id(self.cte), []):
            binding = binding_ref()
            if binding is not None:
                binding._flow = None

    @property
    def width(self) -> int:
        """Number of column identities owned by the relation."""
        return len(self.columns)

    def column_index(self, name: str) -> int:
        """Position of a recursive input/output column name in the relation."""
        return self.columns.index(name)

    def recursive_columns(self, table_name: str | None = None) -> list[exp.Column]:
        """Columns the recursive term reads from its own relation.

        ``table_name`` filters on the qualifier used at the reference site. Every
        returned column belongs to :attr:`recursive_scope`; predicates (WHERE / ON)
        and value projections are not distinguished here, callers classify by the
        column's ancestor.
        """
        result: list[exp.Column] = []
        for column in find_all_in_scope(self.recursive_scope.expression, exp.Column):
            if column.table == self.name and (table_name is None or column.table == table_name):
                result.append(column)
        return result

    def __repr__(self) -> str:
        return f"RecursiveRelation<{self.name}({', '.join(self.columns)})>"


@mypyc_attr(native_class=True)
class Scope:
    """
    Selection scope.

    Attributes:
        expression: Root expression of this scope
        sources: Mapping of source name to either
            a Table expression or another Scope instance. For example:
                SELECT * FROM x                     {"x": Table(this="x")}
                SELECT * FROM x AS y                {"y": Table(this="x")}
                SELECT * FROM (SELECT ...) AS y     {"y": Scope(...)}
        lateral_sources: Sources from laterals
            For example:
                SELECT c FROM x LATERAL VIEW EXPLODE (a) AS c;
            The LATERAL VIEW EXPLODE gets x as a source.
        cte_sources: Sources from CTES
        outer_columns: If this is a derived table or CTE, and the outer query
            defines a column list for the alias of this scope, this is that list of columns.
            For example:
                SELECT * FROM (SELECT ...) AS y(col1, col2)
            The inner query would have `["col1", "col2"]` for its `outer_columns`
        parent: Parent scope
        scope_type: Type of this scope, relative to it's parent
        subquery_scopes: List of all child scopes for subqueries
        cte_scopes: List of all child scopes for CTEs
        derived_table_scopes: List of all child scopes for derived_tables
        udtf_scopes: List of all child scopes for user defined tabular functions
        table_scopes: derived_table_scopes + udtf_scopes, in the order that they're defined
        set_operation_scopes: If this Scope is for a SetOperation expression, this will be
            a list of the left and right child scopes.
        recursive_relation: When this scope is the UNION ALL body of a supported recursive
            CTE, the RecursiveRelation binding that gives the relation's columns their own
            identity; None otherwise. The same binding is reachable from the root scope's
            recursive_relations registry.
    """

    _collected: bool
    _scans_all_subscope_columns: bool
    _raw_columns: list[exp.Column]
    _table_columns: list[exp.TableColumn]
    _stars: list[exp.Column | exp.Dot]
    _derived_tables: list[exp.Subquery]
    _udtfs: list[exp.UDTF]
    _tables: list[exp.Table]
    _ordered_sources: list[exp.Expr]
    _ctes: list[exp.CTE]
    _subqueries: list[exp.Select | exp.SetOperation]
    _join_hints: list[exp.JoinHint]
    _comprehensions: list[exp.Comprehension]
    _semi_anti_join_tables: set[str]
    _column_index: set[int]
    _selected_sources: dict[str, tuple[exp.Selectable, exp.Table | Scope]] | None
    _columns: list[exp.Column] | None
    _external_columns: list[exp.Column] | None
    _local_columns: list[exp.Column] | None
    _pivots: list[exp.Pivot] | None
    _references: list[tuple[str, exp.Selectable]] | None

    def __init__(
        self,
        expression: exp.Expr,
        sources: dict[str, exp.Table | Scope] | None = None,
        outer_columns: list[str] | None = None,
        parent: Scope | None = None,
        scope_type: ScopeType = ScopeType.ROOT,
        lateral_sources: dict[str, exp.Table | Scope] | None = None,
        cte_sources: dict[str, exp.Table | Scope] | None = None,
        can_be_correlated: bool | None = None,
        recursive_relation: RecursiveRelation | None = None,
    ) -> None:
        self.expression = expression
        self.sources = sources or {}
        self.lateral_sources = lateral_sources or {}
        self.cte_sources = cte_sources or {}
        self.sources.update(self.lateral_sources)
        self.sources.update(self.cte_sources)
        self.outer_columns = outer_columns or []
        self.parent = parent
        self.scope_type = scope_type
        self.subquery_scopes: list[Scope] = []
        self.derived_table_scopes: list[Scope] = []
        self.table_scopes: list[Scope] = []
        self.cte_scopes: list[Scope] = []
        self.set_operation_scopes: list[Scope] = []
        self.udtf_scopes: list[Scope] = []
        self.can_be_correlated = can_be_correlated
        self.recursive_relation = recursive_relation
        # RecursiveRelation bindings whose WITH clause is owned by this scope, keyed by
        # relation name. Inner scopes keep their own map, so a same-named inner CTE
        # shadows an outer one instead of overwriting it. The root additionally indexes
        # every binding by `id(relation scope)` for whole-tree lookups.
        self.recursive_relations: dict[str | int, RecursiveRelation] = {}
        self.clear_cache()

    def clear_column_cache(self) -> None:
        """Invalidate the column-classification caches after columns are qualified in place."""
        self._columns = None
        self._external_columns = None
        self._local_columns = None

    def clear_cache(self) -> None:
        self._collected = False
        self._scans_all_subscope_columns = False
        self._raw_columns = []
        self._table_columns = []
        self._stars = []
        self._derived_tables = []
        self._udtfs = []
        self._tables = []
        self._ordered_sources = []
        self._ctes = []
        self._subqueries = []
        self._join_hints = []
        self._comprehensions = []
        self._semi_anti_join_tables = set()
        self._column_index = set()
        self._selected_sources = None
        self._columns = None
        self._external_columns = None
        self._local_columns = None
        self._pivots = None
        self._references = None

    def branch(
        self,
        expression: exp.Expr,
        scope_type: ScopeType,
        sources: dict[str, exp.Table | Scope] | None = None,
        cte_sources: dict[str, exp.Table | Scope] | None = None,
        lateral_sources: dict[str, exp.Table | Scope] | None = None,
        outer_columns: list[str] | None = None,
    ) -> Scope:
        """Branch from the current scope to a new, inner scope"""
        return Scope(
            expression=expression.unnest(),
            sources=sources.copy() if sources else None,
            parent=self,
            scope_type=scope_type,
            cte_sources={**self.cte_sources, **(cte_sources or {})},
            lateral_sources=lateral_sources.copy() if lateral_sources else None,
            can_be_correlated=self.can_be_correlated
            or scope_type in (ScopeType.SUBQUERY, ScopeType.UDTF),
            outer_columns=outer_columns,
        )

    def _collect(self) -> None:
        self._tables = []
        self._ordered_sources = []
        self._ctes = []
        self._subqueries = []
        self._derived_tables = []
        self._udtfs = []
        self._raw_columns = []
        self._table_columns = []
        self._stars = []
        self._join_hints = []
        self._comprehensions = []
        self._semi_anti_join_tables = set()
        self._column_index = set()

        # The inner query of a Subquery-rooted scope is scoped as a derived table by
        # `_traverse_tables`, so it must not also be collected as a subquery
        inner_query = (
            self.expression.unnest() if isinstance(self.expression, exp.Subquery) else None
        )

        for node in self.walk():
            # Most nodes (identifiers, literals, operators etc.) aren't collectible, so a
            # single isinstance gate lets them skip the classification chain below.
            if node is self.expression or not isinstance(node, COLLECTIBLE_TYPES):
                continue

            if isinstance(node, exp.Dot) and node.is_star:
                self._stars.append(node)
            elif type(node) is exp.Column:
                self._column_index.add(id(node))

                if isinstance(node.this, exp.Star):
                    self._stars.append(node)
                else:
                    self._raw_columns.append(node)
            elif isinstance(node, exp.Table) and not isinstance(node.parent, exp.JoinHint):
                if _is_semi_or_anti_join_rhs(node):
                    self._semi_anti_join_tables.add(node.alias_or_name)

                self._tables.append(node)
                self._ordered_sources.append(node)
            elif isinstance(node, exp.JoinHint):
                self._join_hints.append(node)
            elif type(node) is exp.Lateral or (
                isinstance(node, exp.UDTF) and isinstance(node.parent, (exp.From, exp.Join))
            ):
                if _is_semi_or_anti_join_rhs(node):
                    self._semi_anti_join_tables.add(node.alias_or_name)

                self._udtfs.append(node)
                self._ordered_sources.append(node)
            elif isinstance(node, exp.CTE):
                self._ctes.append(node)
            elif _is_derived_table(node) and _is_from_or_join(node):
                if _is_semi_or_anti_join_rhs(node):
                    self._semi_anti_join_tables.add(node.alias_or_name)

                self._derived_tables.append(t.cast(exp.Subquery, node))
                self._ordered_sources.append(node)
            elif (
                isinstance(node, exp.UNWRAPPED_QUERIES)
                and not _is_from_or_join(node)
                and node is not inner_query
            ):
                self._subqueries.append(node)
            elif isinstance(node, exp.TableColumn):
                self._table_columns.append(node)
            elif isinstance(node, exp.Comprehension):
                self._comprehensions.append(node)
            elif isinstance(node, exp.Star) and (
                node.args.get("except_") or not isinstance(node.parent, ROW_LEVEL_AGG_FUNCS)
            ):
                self._scans_all_subscope_columns = True

        self._collected = True

    def _ensure_collected(self) -> None:
        if not self._collected:
            self._collect()

    def walk(self, prune: t.Callable[[exp.Expr], bool] | None = None) -> Iterator[exp.Expr]:
        return walk_in_scope(self.expression, prune=prune)

    def find(self, *expression_types: Type[E]) -> E | None:
        return find_in_scope(self.expression, *expression_types)

    def find_all(self, *expression_types: Type[E]) -> Iterator[E]:
        return find_all_in_scope(self.expression, *expression_types)

    def replace(self, old: exp.Expr, new: exp.Expr) -> None:
        """
        Replace `old` with `new`.

        This can be used instead of `exp.Expr.replace` to ensure the `Scope` is kept up-to-date.

        Args:
            old (exp.Expr): old node
            new (exp.Expr): new node
        """
        old.replace(new)
        self.clear_cache()

    @property
    def tables(self) -> list[exp.Table]:
        """
        List of tables in this scope.

        Returns:
            list[exp.Table]: tables
        """
        self._ensure_collected()
        return self._tables

    @property
    def comprehensions(self) -> list[exp.Comprehension]:
        """List of comprehensions in this scope."""
        self._ensure_collected()
        return self._comprehensions

    @property
    def ctes(self) -> list[exp.CTE]:
        """
        List of CTEs in this scope.

        Returns:
            list[exp.CTE]: ctes
        """
        self._ensure_collected()
        return self._ctes

    @property
    def derived_tables(self) -> list[exp.Subquery]:
        """
        List of derived tables in this scope.

        For example:
            SELECT * FROM (SELECT ...) <- that's a derived table

        Returns:
            list[exp.Subquery]: derived tables
        """
        self._ensure_collected()
        return self._derived_tables

    @property
    def udtfs(self) -> list[exp.UDTF]:
        """
        List of "User Defined Tabular Functions" in this scope.

        Returns:
            list[exp.UDTF]: UDTFs
        """
        self._ensure_collected()
        return self._udtfs

    @property
    def subqueries(self) -> list[exp.Select | exp.SetOperation]:
        """
        List of subqueries in this scope.

        For example:
            SELECT * FROM x WHERE a IN (SELECT ...) <- that's a subquery

        Returns:
            list[exp.Select | exp.SetOperation]: subqueries
        """
        self._ensure_collected()
        return self._subqueries

    @property
    def scans_all_subscope_columns(self) -> bool:
        self._ensure_collected()
        return self._scans_all_subscope_columns

    @property
    def stars(self) -> list[exp.Column | exp.Dot]:
        """
        List of star expressions (columns or dots) in this scope.
        """
        self._ensure_collected()
        return self._stars

    @property
    def column_index(self) -> set[int]:
        """
        Set of column object IDs that belong to this scope's expression.
        """
        self._ensure_collected()
        return self._column_index

    @property
    def columns(self) -> list[exp.Column]:
        """
        List of columns in this scope.

        Returns:
            list[exp.Column]: Column instances in this scope, plus any
                Columns that reference this scope from correlated subqueries.
        """
        if self._columns is None:
            self._ensure_collected()
            columns = self._raw_columns

            external_columns = [
                column
                for scope in itertools.chain(
                    self.subquery_scopes,
                    self.udtf_scopes,
                    (dts for dts in self.derived_table_scopes if dts.can_be_correlated),
                )
                for column in scope.external_columns
            ]

            expr = self.expression
            named_selects = set(expr.named_selects) if isinstance(expr, exp.Query) else set()

            self._columns = []
            for column in columns + external_columns:
                ancestor = column.find_ancestor(
                    exp.Select,
                    exp.Qualify,
                    exp.Order,
                    exp.Cluster,
                    exp.Having,
                    exp.Hint,
                    exp.Table,
                    exp.Star,
                    exp.Distinct,
                )
                if (
                    not ancestor
                    or column.text("table")
                    or isinstance(ancestor, exp.Select)
                    or (isinstance(ancestor, exp.Table) and not isinstance(ancestor.this, exp.Func))
                    or (
                        isinstance(ancestor, (exp.Order, exp.Cluster, exp.Distinct))
                        and (
                            isinstance(ancestor.parent, (exp.Window, exp.WithinGroup))
                            or not isinstance(ancestor.parent, exp.Select)
                            or column.name not in named_selects
                        )
                    )
                    or (isinstance(ancestor, exp.Star) and not column.arg_key == "except_")
                ):
                    self._columns.append(column)

        return self._columns

    @property
    def table_columns(self) -> list[exp.TableColumn]:
        self._ensure_collected()
        return self._table_columns

    @property
    def selected_sources(self) -> dict[str, tuple[exp.Selectable, exp.Table | Scope]]:
        """
        Mapping of nodes and sources that are actually selected from in this scope.

        That is, all tables in a schema are selectable at any point. But a
        table only becomes a selected source if it's included in a FROM or JOIN clause.

        Returns:
            dict[str, (exp.Table|exp.Select, exp.Table|Scope)]: selected sources and nodes
        """
        if self._selected_sources is None:
            result: dict[str, tuple[exp.Selectable, exp.Table | Scope]] = {}

            for name, node in self.references:
                if name in self._semi_anti_join_tables:
                    # The RHS table of SEMI/ANTI joins shouldn't be collected as a
                    # selected source
                    continue

                if name in result:
                    raise OptimizeError(f"Alias already used: {name}")
                if name in self.sources:
                    result[name] = (node, self.sources[name])

            self._selected_sources = result
        return self._selected_sources

    @property
    def references(self) -> list[tuple[str, exp.Selectable]]:
        if self._references is None:
            self._ensure_collected()
            self._references = []

            # Sources are kept in walk order, which star expansion relies on
            for expression in self._ordered_sources:
                if isinstance(expression, exp.Table):
                    self._references.append((expression.alias_or_name, expression))
                else:
                    self._references.append(
                        (
                            _get_source_alias(expression),
                            (
                                expression if expression.args.get("pivots") else expression.unnest()
                            ).assert_is(exp.Selectable),
                        )
                    )

        return self._references

    @property
    def external_columns(self) -> list[exp.Column]:
        """
        Columns that appear to reference sources in outer scopes.

        Returns:
            list[exp.Column]: Column instances that don't reference sources in the current scope.
        """
        if self._external_columns is None:
            if isinstance(self.expression, exp.SetOperation):
                left, right = self.set_operation_scopes
                self._external_columns = left.external_columns + right.external_columns
            else:
                local_source_names = {name for name, _ in self.references}
                self._external_columns = [
                    c
                    for c in self.columns
                    if c.text("table") not in local_source_names
                    and c.text("table") not in self.semi_or_anti_join_tables
                ]

        return self._external_columns

    @property
    def local_columns(self) -> list[exp.Column]:
        """
        Columns in this scope that are not external.

        Returns:
            list[exp.Column]: Column instances that reference sources in the current scope.
        """
        if self._local_columns is None:
            # Compare nodes by identity: structural equality would conflate distinct
            # column nodes that happen to look the same, and is much more expensive.
            external_column_ids = {id(c) for c in self.external_columns}
            self._local_columns = [c for c in self.columns if id(c) not in external_column_ids]

        return self._local_columns

    @property
    def unqualified_columns(self) -> list[exp.Column]:
        """
        Unqualified columns in the current scope.

        Returns:
             list[exp.Column]: Unqualified columns
        """
        return [c for c in self.columns if not c.text("table")]

    @property
    def join_hints(self) -> list[exp.JoinHint]:
        """
        Hints that exist in the scope that reference tables

        Returns:
            list[exp.JoinHint]: Join hints that are referenced within the scope
        """
        self._ensure_collected()
        return self._join_hints

    @property
    def pivots(self) -> list[exp.Pivot]:
        if self._pivots is None:
            self._pivots = [
                pivot for _, node in self.references for pivot in node.args.get("pivots") or []
            ]

        return self._pivots

    @property
    def semi_or_anti_join_tables(self) -> set[str]:
        self._ensure_collected()
        return self._semi_anti_join_tables

    def source_columns(self, source_name: str) -> list[exp.Column]:
        """
        Get all columns in the current scope for a particular source.

        Args:
            source_name (str): Name of the source
        Returns:
            list[exp.Column]: Column instances that reference `source_name`
        """
        return [column for column in self.columns if column.text("table") == source_name]

    @property
    def is_subquery(self) -> bool:
        """Determine if this scope is a subquery"""
        return self.scope_type == ScopeType.SUBQUERY

    @property
    def is_derived_table(self) -> bool:
        """Determine if this scope is a derived table"""
        return self.scope_type == ScopeType.DERIVED_TABLE

    @property
    def is_set_operation(self) -> bool:
        """Determine if this scope is a set operation"""
        return self.scope_type == ScopeType.SET_OPERATION

    @property
    def is_cte(self) -> bool:
        """Determine if this scope is a common table expression"""
        return self.scope_type == ScopeType.CTE

    @property
    def is_root(self) -> bool:
        """Determine if this is the root scope"""
        return self.scope_type == ScopeType.ROOT

    @property
    def is_udtf(self) -> bool:
        """Determine if this scope is a UDTF (User Defined Table Function)"""
        return self.scope_type == ScopeType.UDTF

    @property
    def is_correlated_subquery(self) -> bool:
        """Determine if this scope is a correlated subquery"""
        return bool(self.can_be_correlated and self.external_columns)

    def rename_source(self, old_name: str | None, new_name: str) -> None:
        """Rename a source in this scope"""
        old_name = old_name or ""
        if old_name in self.sources:
            self.sources[new_name] = self.sources.pop(old_name)

    def add_source(self, name: str, source: exp.Table | Scope) -> None:
        """Add a source to this scope"""
        self.sources[name] = source
        self.clear_cache()

    def remove_source(self, name: str) -> None:
        """Remove a source from this scope"""
        self.sources.pop(name, None)
        self.clear_cache()

    def __repr__(self) -> str:
        return f"Scope<{self.expression.sql()}>"

    def traverse(self) -> Iterator[Scope]:
        """
        Traverse the scope tree from this node.

        Yields:
            Scope: scope instances in depth-first-search post-order
        """
        stack: list[Scope] = [self]
        result: list[Scope] = []
        while stack:
            scope = stack.pop()
            result.append(scope)
            stack.extend(
                itertools.chain(
                    scope.cte_scopes,
                    scope.set_operation_scopes,
                    scope.table_scopes,
                    scope.subquery_scopes,
                )
            )

        yield from reversed(result)

    def ref_count(self) -> dict[int, int]:
        """
        Count the number of times each scope in this tree is referenced.

        Returns:
            dict[int, int]: Mapping of Scope instance ID to reference count
        """
        scope_ref_count: dict[int, int] = defaultdict(int)

        for scope in self.traverse():
            for _, source in scope.selected_sources.values():
                scope_ref_count[id(source)] += 1

            for name in scope._semi_anti_join_tables:
                # semi/anti join sources are not actually selected but we still need to
                # increment their ref count to avoid them being optimized away
                if name in scope.sources:
                    scope_ref_count[id(scope.sources[name])] += 1

        return scope_ref_count

    @property
    def root_scope(self) -> Scope:
        """The topmost scope of this scope tree."""
        scope = self
        while scope.parent is not None:
            scope = scope.parent
        return scope

    def recursive_binding(self, name: str) -> RecursiveRelation | None:
        """Look up a recursive relation binding visible from this scope, by CTE name.

        Inner scopes shadow outer CTEs of the same name: the lookup walks the scope
        hierarchy innermost-first and returns the first binding declared by a scope
        whose relation source is actually visible here.
        """
        scope: Scope | None = self
        while scope is not None:
            relation = scope.recursive_relations.get(name)
            if relation is not None:
                return relation
            scope = scope.parent
        return None


def traverse_scope(expression: exp.Expr) -> list[Scope]:
    """
    Traverse an expression by its "scopes".

    "Scope" represents the current context of a Select statement.

    This is helpful for optimizing queries, where we need more information than
    the expression tree itself. For example, we might care about the source
    names within a subquery. Returns a list because a generator could result in
    incomplete properties which is confusing.

    Scopes are returned in depth-first post-order. For each scope, its descendants
    occupy consecutive entries in the returned list immediately before it. A scope
    is a descendant of another scope if following its `parent` links eventually
    reaches that other scope.

    Examples:
        >>> import sqlglot
        >>> expression = sqlglot.parse_one("SELECT a FROM (SELECT a FROM x) AS y")
        >>> scopes = traverse_scope(expression)
        >>> scopes[0].expression.sql(), list(scopes[0].sources)
        ('SELECT a FROM x', ['x'])
        >>> scopes[1].expression.sql(), list(scopes[1].sources)
        ('SELECT a FROM (SELECT a FROM x) AS y', ['y'])

    Args:
        expression: Expr to traverse

    Returns:
        A list of the created scope instances
    """
    if isinstance(expression, TRAVERSABLES):
        return list(_traverse_scope(Scope(expression)))
    return []


def build_scope(expression: exp.Expr) -> Scope | None:
    """
    Build a scope tree.

    Args:
        expression: Expr to build the scope tree for.

    Returns:
        The root scope
    """
    return seq_get(traverse_scope(expression), -1)


def _traverse_scope(scope: Scope) -> Iterator[Scope]:
    expression = scope.expression

    if isinstance(expression, exp.Select):
        yield from _traverse_select(scope)
    elif isinstance(expression, exp.SetOperation):
        yield from _traverse_ctes(scope)
        yield from _traverse_set_operation(scope)
        return
    elif isinstance(expression, exp.Subquery):
        if scope.is_root:
            yield from _traverse_select(scope)
        else:
            yield from _traverse_subqueries(scope)
    elif isinstance(expression, exp.Table):
        yield from _traverse_tables(scope)
    elif isinstance(expression, exp.UDTF):
        yield from _traverse_udtfs(scope)
    elif isinstance(expression, exp.DDL):
        ddl_expression = expression.expression
        if isinstance(ddl_expression, exp.Query):
            yield from _traverse_ctes(scope)
            yield from _traverse_scope(Scope(ddl_expression, cte_sources=scope.cte_sources))
        return
    elif isinstance(expression, exp.DML):
        yield from _traverse_ctes(scope)

        # Bare tables in relation position (e.g. UPDATE ... FROM t, DELETE / MERGE ... USING t)
        # aren't part of any query, so they're scoped as standalone tables; `_traverse_tables`
        # also picks up any joins hanging off of them
        relations: list[exp.Expr] = []
        from_ = expression.args.get("from_")

        if isinstance(from_, exp.From):
            relations.append(from_.this)

        using = expression.args.get("using")
        if isinstance(using, list):
            relations.extend(using)
        elif isinstance(using, exp.Expr):
            relations.append(using)

        for relation in relations:
            if isinstance(relation, exp.Table):
                yield from _traverse_scope(Scope(relation, cte_sources=scope.cte_sources))

        for query in find_all_in_scope(expression, exp.Query):
            # This check ensures we don't yield the CTE/nested queries twice
            if isinstance(query.parent, (exp.CTE, exp.Subquery)):
                continue

            if _is_from_or_join(query):
                parent = query.parent
                if isinstance(parent, exp.Join) and isinstance(
                    parent.parent, (exp.Subquery, exp.Table)
                ):
                    # Scoped by the FROM-position relation (wrapper or table) it's joined to
                    continue

                # A query in FROM/JOIN position (e.g. UPDATE ... FROM (SELECT ...) AS s) acts
                # like a derived table, so its scope stays rooted at the Subquery wrapper to
                # pick up the wrapper's alias, column list and joins
                yield from _traverse_scope(Scope(query, cte_sources=scope.cte_sources))
            else:
                # Queries in value position (SET, WHERE, USING, ...) are scoped as subqueries,
                # e.g. so their columns can be correlated to the DML's target table
                yield from _traverse_scope(scope.branch(query, scope_type=ScopeType.SUBQUERY))
        return
    else:
        logger.warning("Cannot traverse scope %s with type '%s'", expression, type(expression))
        return

    yield scope


def _traverse_select(scope: Scope) -> Iterator[Scope]:
    yield from _traverse_ctes(scope)
    yield from _traverse_tables(scope)
    yield from _traverse_subqueries(scope)


def _traverse_set_operation(scope: Scope) -> Iterator[Scope]:
    prev_scope: Scope | None = None
    set_op_scope_stack: list[Scope] = [scope]

    set_op = scope.expression
    assert isinstance(set_op, exp.SetOperation)
    # Access the args directly instead of the left/right properties, because set operation
    # operands aren't guaranteed to be Query nodes, e.g. in VALUES (1) UNION ALL SELECT 1
    expression_stack: list[exp.Expr] = [set_op.expression, set_op.this]
    by_name_col_map: dict[str, str] | None = None

    while expression_stack:
        expression = expression_stack.pop()
        set_op_scope = set_op_scope_stack[-1]

        outer = set_op_scope.outer_columns
        set_op_expr = set_op_scope.expression
        # all arms need their outer aliases reordered for BY NAME
        if (
            outer
            and prev_scope is not None
            and isinstance(set_op_expr, exp.SetOperation)
            and set_op_expr.args.get("by_name")
        ):
            if by_name_col_map is None:
                names = set_op.named_selects
                by_name_col_map = {} if "*" in names else dict(zip(names, scope.outer_columns))

            if by_name_col_map:
                arm_node = expression.unnest()
                outer = [
                    by_name_col_map.get(name, "") for name in getattr(arm_node, "named_selects", [])
                ]

        new_scope = set_op_scope.branch(
            expression,
            outer_columns=outer,
            scope_type=ScopeType.SET_OPERATION,
        )

        # Use the scope's unwrapped expression rather than the operand itself so parenthesized
        # left operands are descended into iteratively. Only descend while prev_scope is None,
        # i.e. until the chain's leftmost query has been scoped; later scopes are paired as
        # right operands.
        inner = new_scope.expression
        if prev_scope is None and isinstance(inner, exp.SetOperation):
            yield from _traverse_ctes(new_scope)

            set_op_scope_stack.append(new_scope)
            expression_stack.extend([inner.expression, inner.this])
            continue

        branch_scope: Scope | None = None
        for branch_scope in _traverse_scope(new_scope):
            yield branch_scope

        if branch_scope is None:
            raise OptimizeError(f"Cannot build a scope for set operation operand: {expression}")

        if prev_scope:
            set_op_scope_stack.pop()
            set_op_scope.set_operation_scopes = [prev_scope, branch_scope]
            prev_scope = set_op_scope

            yield set_op_scope
        else:
            prev_scope = branch_scope


def _traverse_ctes(scope: Scope) -> Iterator[Scope]:
    sources: dict[str, exp.Table | Scope] = {}
    recursive_bindings: list[RecursiveRelation] = []

    for cte in scope.ctes:
        cte_name = cte.alias

        # if the scope is a recursive cte, it must be in the form of base_case UNION recursive.
        # thus the recursive scope is the first section of the union.
        with_ = scope.expression.args.get("with_")
        union = cte.this if with_ and with_.recursive else None

        if isinstance(union, exp.SetOperation):
            sources[cte_name] = scope.branch(union.this, scope_type=ScopeType.CTE)

        child_scope: Scope | None = None

        for child_scope in _traverse_scope(
            scope.branch(
                cte.this,
                cte_sources=sources,
                outer_columns=cte.alias_column_names,
                scope_type=ScopeType.CTE,
            )
        ):
            yield child_scope

        # append the final child_scope yielded
        if child_scope:
            sources[cte_name] = child_scope
            scope.cte_scopes.append(child_scope)

            if isinstance(union, exp.SetOperation):
                binding = _bind_recursive_cte(cte, union, child_scope)
                if binding is not None:
                    recursive_bindings.append(binding)

    scope.sources.update(sources)
    scope.cte_sources.update(sources)

    # Bindings declared by this WITH clause are kept on the scope that owns it, so an
    # inner CTE of the same name shadows the outer binding instead of overwriting it.
    scope.recursive_relations.update({binding.name: binding for binding in recursive_bindings})

    root = scope.root_scope
    for binding in recursive_bindings:
        # The root registry still indexes every binding by expression id for callers
        # that start from the root; same-named declarations remain distinguishable via
        # their own scope's recursive_relations map and recursive_binding().
        root.recursive_relations.setdefault(id(binding.scope), binding)


def _bind_recursive_cte(
    cte: exp.CTE, union: exp.SetOperation, union_scope: Scope
) -> RecursiveRelation | None:
    """Give a recursive CTE's self-reference a fixed-point column identity.

    The traversal seeds the relation name with the anchor scope so the recursive term
    can be collected, then replaces the seed everywhere within the CTE body with the
    finished UNION scope. Without the replacement, self-references would resolve to the
    anchor term only, silently dropping sources the recursive term introduces later.
    """
    name = cte.alias
    arm_scopes = union_scope.set_operation_scopes
    if len(arm_scopes) != 2:
        return None

    anchor_scope, recursive_scope = arm_scopes
    self_references = [
        table
        for table in find_all_in_scope(recursive_scope.expression, exp.Table)
        if not table.db and table.name == name
    ]

    # Rewire every self-reference inside the recursive term and the union's own source
    # map to the relation's scope instead of the anchor seed scope.
    rewired = False
    for arm_scope in (anchor_scope, recursive_scope, union_scope):
        for source_name, source in list(arm_scope.sources.items()):
            if source_name == name and source is not union_scope:
                arm_scope.sources[source_name] = union_scope
                rewired = True
        for source_name, source in list(arm_scope.cte_sources.items()):
            if source_name == name and source is not union_scope:
                arm_scope.cte_sources[source_name] = union_scope
                rewired = True

    if not rewired and not self_references:
        return None

    alias_columns = cte.alias_column_names
    columns = alias_columns or [select.alias_or_name for select in union.this.selects]
    supported = _is_supported_recursive_relation(union, self_references)

    binding = RecursiveRelation(
        name=name,
        scope=union_scope,
        anchor_scope=anchor_scope,
        recursive_scope=recursive_scope,
        columns=list(columns),
        cte=cte,
        alias_columns=alias_columns,
        supported=supported and bool(self_references),
        self_reference=self_references[0] if self_references else None,
    )
    union_scope.recursive_relation = binding
    recursive_scope.recursive_relation = binding
    return binding


def _is_supported_recursive_relation(
    union: exp.SetOperation, self_references: list[exp.Table]
) -> bool:
    """Whether the recursive CTE matches the shape the fixed-point analysis supports."""
    if not isinstance(union, exp.Union) or union.args.get("distinct"):
        return False

    anchor = union.this.unnest()
    recursive = union.expression.unnest()
    if not isinstance(anchor, exp.Select) or not isinstance(recursive, exp.Select):
        return False

    if len(self_references) != 1:
        return False

    reference = self_references[0]
    parent = reference.parent
    if not isinstance(parent, (exp.From, exp.Join)) or (
        isinstance(parent, exp.Join) and parent.method not in (None, "INNER")
    ):
        return False

    # The recursive term must not aggregate, deduplicate, window, paginate or nest a
    # correlated query while reading its own relation. Table-isolation rewrites ordinary
    # base tables into FROM/JOIN-position derived tables, which stay within the supported
    # shape; only value/predicate-position subqueries disqualify the relation.
    if recursive.args.get("distinct") or recursive.args.get("group") or recursive.args.get("order"):
        return False
    if recursive.args.get("limit") or recursive.args.get("offset") or recursive.args.get("having"):
        return False
    for node in recursive.walk():
        if node is recursive:
            continue
        if isinstance(node, (exp.AggFunc, exp.Window)):
            return False
        if isinstance(node, exp.Subquery) and not _is_from_or_join(node):
            return False

    return True


def _is_derived_table(expression: exp.Expr) -> bool:
    """
    We represent (tbl1 JOIN tbl2) as a Subquery, but it's not really a "derived table",
    as it doesn't introduce a new scope. An alias shadows the names under the Subquery,
    and a pivot changes its output columns, so both require a separate scope.
    """
    return isinstance(expression, exp.Subquery) and bool(
        expression.alias
        or expression.args.get("pivots")
        or isinstance(expression.this, exp.UNWRAPPED_QUERIES)
    )


def _is_from_or_join(expression: exp.Expr) -> bool:
    """
    Determine if `expression` is the FROM or JOIN clause of a SELECT statement.
    """
    parent = expression.parent

    # Subqueries can be arbitrarily nested
    while type(parent) is exp.Subquery:
        parent = parent.parent

    return type(parent) in (exp.From, exp.Join)


def _is_semi_or_anti_join_rhs(node: exp.Expr) -> bool:
    """
    Determine if `node` is the right-hand side source of a SEMI/ANTI join, whose columns
    are only visible within the join's own ON clause, not the rest of the enclosing scope.
    """
    parent = node.parent
    return isinstance(parent, exp.Join) and parent.is_semi_or_anti_join


def _traverse_tables(scope: Scope) -> Iterator[Scope]:
    sources: dict[str, exp.Table | Scope] = {}

    # Traverse FROMs, JOINs, and LATERALs in the order they are defined
    expressions: list[exp.Expr] = []
    from_ = scope.expression.args.get("from_")
    if from_:
        expressions.append(from_.this)

    for join in scope.expression.args.get("joins") or []:
        expressions.append(join.this)

    if isinstance(scope.expression, (exp.Table, exp.Subquery)):
        # A Subquery-rooted scope, e.g., the FROM clause of a DML statement, a DDL source or
        # a parenthesized query like (SELECT ...) LIMIT 1, scopes its own inner query as a
        # derived table
        expressions.append(scope.expression)

    expressions.extend(scope.expression.args.get("laterals") or [])

    for expression in expressions:
        if isinstance(expression, exp.Final):
            expression = expression.this
        if isinstance(expression, exp.Table):
            table_name = expression.name
            source_name = expression.alias_or_name

            if table_name in scope.sources and not expression.db:
                # This is a reference to a parent source (e.g. a CTE), not an actual table. If
                # it is pivoted, we also register the pivot's alias as a new table source.
                sources[source_name] = scope.sources[table_name]
                pivots = expression.args.get("pivots")
                if pivots:
                    sources[pivots[-1].alias] = expression
            elif source_name in sources:
                sources[find_new_name(sources, table_name)] = expression
            else:
                sources[source_name] = expression

            # Make sure to not include the joins twice
            if expression is not scope.expression:
                expressions.extend(join.this for join in expression.args.get("joins") or [])

            continue

        if not isinstance(expression, exp.DerivedTable):
            continue

        # TODO (mypyc): rebind to exp.Expr to avoid DerivedTable/UDTF trait vtable dispatch
        node: exp.Expr = expression

        if isinstance(expression, exp.UDTF):
            lateral_sources = sources
            scope_type = ScopeType.UDTF
            scopes = scope.udtf_scopes
        elif _is_derived_table(expression):
            lateral_sources = None
            scope_type = ScopeType.DERIVED_TABLE
            scopes = scope.derived_table_scopes
            if node is not scope.expression:
                # The scope expression's own joins were already added above
                expressions.extend(join.this for join in node.args.get("joins") or [])
        else:
            # Makes sure we check for possible sources in nested table constructs
            expressions.append(node.this)
            if node is not scope.expression:
                expressions.extend(join.this for join in node.args.get("joins") or [])
            continue

        child_scope: Scope | None = None

        for child_scope in _traverse_scope(
            scope.branch(
                node,
                lateral_sources=lateral_sources,
                outer_columns=node.alias_column_names,
                scope_type=scope_type,
            )
        ):
            yield child_scope

            # Tables without aliases will be set as ""
            # This shouldn't be a problem once qualify_columns runs, as it adds aliases on everything.
            # Until then, this means that only a single, unaliased derived table is allowed (rather,
            # the latest one wins.
            sources[_get_source_alias(node)] = child_scope

        # append the final child_scope yielded
        if child_scope:
            scopes.append(child_scope)
            scope.table_scopes.append(child_scope)

    scope.sources.update(sources)


def _traverse_subqueries(scope: Scope) -> Iterator[Scope]:
    for subquery in scope.subqueries:
        top: Scope | None = None
        for child_scope in _traverse_scope(scope.branch(subquery, scope_type=ScopeType.SUBQUERY)):
            yield child_scope
            top = child_scope
        if top is not None:
            scope.subquery_scopes.append(top)


def _traverse_udtfs(scope: Scope) -> Iterator[Scope]:
    if isinstance(scope.expression, exp.Unnest):
        udtf_expressions = scope.expression.expressions
    elif isinstance(scope.expression, exp.Lateral):
        udtf_expressions = [scope.expression.this]
    else:
        udtf_expressions = []

    sources: dict[str, exp.Table | Scope] = {}
    for expression in udtf_expressions:
        if isinstance(expression, exp.Subquery):
            top: Scope | None = None
            for child_scope in _traverse_scope(
                scope.branch(
                    expression,
                    scope_type=ScopeType.SUBQUERY,
                    outer_columns=expression.alias_column_names,
                )
            ):
                yield child_scope
                top = child_scope
                sources[_get_source_alias(expression)] = child_scope

            if top is not None:
                scope.subquery_scopes.append(top)

    scope.sources.update(sources)


def walk_in_scope(
    expression: exp.Expr,
    prune: t.Callable[[exp.Expr], bool] | None = None,
) -> Iterator[exp.Expr]:
    """
    Returns a generator object which visits all nodes in the syntrax tree, stopping at
    nodes that start child scopes. This does a custom DFS traversal rather than using
    expression.walk() because the nested generators aren't optimized in mypyc.

    Args:
        expression:
        prune: callable that returns True if
            the generator should stop traversing this branch of the tree.

    Yields:
        exp.Expr: each node in scope
    """
    stack: list[exp.Expr] = [expression]

    while stack:
        node = stack.pop()

        yield node

        # Only CTEs and Queries can start child scopes; checking that first lets all
        # other nodes (the vast majority) skip the rest of the boundary checks.
        if (
            node is not expression
            and isinstance(node, (exp.CTE, exp.Query))
            and (
                isinstance(node, exp.CTE)
                or (isinstance(node.parent, (exp.From, exp.Join)) and _is_derived_table(node))
                or isinstance(node.parent, exp.UDTF)
                or isinstance(node, exp.UNWRAPPED_QUERIES)
            )
        ):
            if isinstance(node, (exp.Subquery, exp.UDTF)):
                for key in ("joins", "laterals", "pivots"):
                    for arg in node.args.get(key) or []:
                        yield from walk_in_scope(arg)
            continue

        if prune and prune(node):
            continue

        for vs in reversed(node.args.values()):
            if isinstance(vs, list):
                for v in reversed(vs):
                    if isinstance(v, exp.Expr):
                        stack.append(v)
            elif isinstance(vs, exp.Expr):
                stack.append(vs)


def find_all_in_scope(
    expression: exp.Expr,
    *expression_types: Type[E],
) -> Iterator[E]:
    """
    Returns a generator object which visits all nodes in this scope and only yields those that
    match at least one of the specified expression types.

    This does NOT traverse into subscopes.

    Args:
        expression: the expression to search.
        expression_types: the expression type(s) to match.

    Yields:
        The matching nodes.
    """
    for node in walk_in_scope(expression):
        if isinstance(node, expression_types):
            yield node


def find_in_scope(
    expression: exp.Expr,
    *expression_types: Type[E],
) -> E | None:
    """
    Returns the first node in this scope which matches at least one of the specified types.

    This does NOT traverse into subscopes.

    Args:
        expression: the expression to search.
        expression_types: the expression type(s) to match.

    Returns:
        The node which matches the criteria or None if no node matching
        the criteria was found.
    """
    return next(find_all_in_scope(expression, *expression_types), None)


def _get_source_alias(expression: exp.Expr) -> str:
    alias_arg = expression.args.get("alias")
    alias_name = expression.alias

    if not alias_name and isinstance(alias_arg, exp.TableAlias) and len(alias_arg.columns) == 1:
        alias_name = alias_arg.columns[0].name

    return alias_name
