"""Fixed-point column dependency analysis for supported recursive CTEs.

The scope tree gives every recursive CTE a :class:`~sqlglot.optimizer.scope.RecursiveRelation`
binding whose columns have an identity independent of the anchor term. This module turns
that binding into one finite column-level dependency summary shared by:

* :mod:`sqlglot.lineage` — the complete set of base-table *value* sources reachable from
  each output, including sources the recursive term introduces in later iterations;
* :mod:`sqlglot.optimizer.pushdown_projections` — which recursive columns must be kept to
  preserve the output values and the row conditions.

The summary is a graph over the relation's own column identities plus the external
(anchor/base/derived/CTE) columns referenced by the two arms. For a model with C column
identities and E direct dependencies the resident state is O(C + E): edges are stored
once each and the fixed point is reached by a work-list fixpoint, never by unrolling
iterations or enumerating recursive paths, so the cost is independent of runtime
recursion depth, path count and data volume.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

from sqlglot import exp
from sqlglot.optimizer.scope import RecursiveRelation, find_all_in_scope

if t.TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class ExternalColumn:
    """A column the recursive body reads from a source other than the relation itself.

    Attributes:
        table: Qualifier of the column at the recursive reference site (source alias).
        name: Column name under that qualifier.
        expression: The actual qualified ``exp.Column`` node.
        predicate_only: True when the column appears only inside WHERE / ON conditions,
            never in a value projection of either arm. It still controls which rows the
            recursion produces, but is not a value source for any output.
    """

    table: str
    name: str
    expression: exp.Column
    predicate_only: bool

    @property
    def key(self) -> tuple[str, str]:
        return (self.table, self.name)


@dataclass
class RecursiveFlow:
    """Finite column-level dependency summary of one :class:`RecursiveRelation`.

    All column references are positional indexes into :attr:`RecursiveRelation.columns`
    for the relation's own columns, and keys into :attr:`external` for outside sources.

    Attributes:
        relation: The binding this flow summarizes.
        anchor_inputs: For each output position, external (table, column) keys the anchor
            term reads for that projection.
        recursive_inputs: For each output position, relation column indexes the recursive
            term reads from its own previous iteration for that projection.
        recursive_external: For each output position, external keys the recursive term
            introduces in that projection.
        external: Every distinct external column referenced anywhere in the two arms,
            keyed by (qualifier, name), with predicate usage classified.
        predicate_recursive: Relation column indexes used by WHERE / ON in either arm.
        condition_external: External keys used by WHERE / ON in either arm.
    """

    relation: RecursiveRelation
    anchor_inputs: list[list[tuple[str, str]]]
    recursive_inputs: list[list[int]]
    recursive_external: list[list[tuple[str, str]]]
    external: dict[tuple[str, str], ExternalColumn]
    predicate_recursive: frozenset[int]
    condition_external: frozenset[tuple[str, str]]

    def value_closure(
        self, outputs: t.Iterable[int]
    ) -> tuple[frozenset[int], frozenset[tuple[str, str]]]:
        """All relation columns and external value sources needed for ``outputs``.

        The closure follows recursive value edges (output i reads previous-iteration
        column j) until it stops changing, then unions the anchor and recursive-term
        external sources of every reached output. Predicate-only edges are deliberately
        excluded: a column used only in WHERE/ON is not a value source.
        """
        pending = list(dict.fromkeys(outputs))
        reached: set[int] = set()
        while pending:
            i = pending.pop()
            if i in reached:
                continue
            reached.add(i)
            for j in self.recursive_inputs[i]:
                if j not in reached:
                    pending.append(j)

        external: set[tuple[str, str]] = set()
        for i in reached:
            external.update(self.anchor_inputs[i])
            external.update(self.recursive_external[i])

        return frozenset(reached), frozenset(
            key for key in external if not self.external[key].predicate_only
        )

    def required_columns(self, live_outputs: t.Iterable[int]) -> frozenset[int]:
        """Relation columns that must survive projection pruning.

        A column is required when it is:

        * a live output of the relation, or feeds one through a recursive value swap;
        * read by a WHERE / ON row condition (recursion stop or join match);
        * exchanged into a required column even if it is never output directly.
        """
        reached, _ = self.value_closure(live_outputs)
        required = set(reached)
        required.update(self.predicate_recursive)

        # A predicate column can itself be fed by other relation columns, which then
        # have to be kept as well, so close over value edges once more.
        pending = list(required)
        while pending:
            i = pending.pop()
            for j in self.recursive_inputs[i]:
                if j not in required:
                    required.add(j)
                    pending.append(j)

        return frozenset(required)

    def walk_value_dependencies(self, output: int) -> Iterator[tuple[int, int]]:
        """Yield each distinct recursive value edge (consumer, producer) reachable once.

        Used by lineage to render the recursion as a finite graph with an identifiable
        cycle rather than expanding it for some fixed number of rounds.
        """
        seen: set[tuple[int, int]] = set()
        pending = [output]
        reached: set[int] = set()
        while pending:
            i = pending.pop()
            if i in reached:
                continue
            reached.add(i)
            for j in self.recursive_inputs[i]:
                edge = (i, j)
                if edge not in seen:
                    seen.add(edge)
                    yield edge
                if j not in reached:
                    pending.append(j)


def build_recursive_flow(relation: RecursiveRelation) -> RecursiveFlow:
    """Compute the single shared dependency summary for a recursive relation binding.

    Repeated calls within one analysis (lineage of several outputs, reachability) are
    served from the binding's cache; call :meth:`RecursiveRelation.invalidate_flow`
    after rewriting the AST in place.

    Raises:
        OptimizeError: when an arm references a relation column name that does not exist
            or the two arms do not have the same projection width as the binding.
    """
    if relation._flow is not None:
        return relation._flow

    from sqlglot.errors import OptimizeError

    width = relation.width
    anchor_expression = relation.anchor_scope.expression
    recursive_expression = relation.recursive_scope.expression
    if not isinstance(anchor_expression, exp.Select) or not isinstance(
        recursive_expression, exp.Select
    ):
        raise OptimizeError(f"Recursive CTE '{relation.name}' arms must be SELECT statements.")
    anchor_selects = anchor_expression.selects
    recursive_selects = recursive_expression.selects

    if len(anchor_selects) != width or len(recursive_selects) != width:
        raise OptimizeError(
            "Recursive CTE arms must have the same number of columns as the relation "
            f"'{relation.name}': {relation.columns}."
        )

    name_to_index = {name: i for i, name in enumerate(relation.columns)}
    external: dict[tuple[str, str], ExternalColumn] = {}
    value_external: set[tuple[str, str]] = set()

    def classify_external(scope_expression: exp.Expr, projections: list[exp.Expr]) -> None:
        predicate_tables = _predicate_tables(scope_expression)
        for column in find_all_in_scope(scope_expression, exp.Column):
            if column.table == relation.name:
                continue
            key = (column.table, column.name)
            in_predicate = column.table in predicate_tables and _is_inside_predicate(column)
            entry = external.get(key)
            if entry is None:
                external[key] = ExternalColumn(
                    table=column.table,
                    name=column.name,
                    expression=column,
                    predicate_only=in_predicate,
                )
            elif not in_predicate:
                external[key] = ExternalColumn(
                    table=column.table,
                    name=column.name,
                    expression=column,
                    predicate_only=False,
                )
        for projection in projections:
            for column in find_all_in_scope(projection, exp.Column):
                if column.table != relation.name:
                    value_external.add((column.table, column.name))

    classify_external(relation.anchor_scope.expression, anchor_selects)
    classify_external(relation.recursive_scope.expression, recursive_selects)

    # Anything seen in a value projection is not predicate-only.
    for key in value_external:
        entry = external.get(key)
        if entry is not None and entry.predicate_only:
            external[key] = ExternalColumn(
                table=entry.table,
                name=entry.name,
                expression=entry.expression,
                predicate_only=False,
            )

    anchor_inputs: list[list[tuple[str, str]]] = []
    recursive_inputs: list[list[int]] = []
    recursive_external: list[list[tuple[str, str]]] = []

    for i, (anchor_select, recursive_select) in enumerate(zip(anchor_selects, recursive_selects)):
        anchor_keys = [
            (c.table, c.name)
            for c in find_all_in_scope(anchor_select, exp.Column)
            if c.table != relation.name
        ]
        anchor_inputs.append(anchor_keys)

        own_indexes: list[int] = []
        outside_keys: list[tuple[str, str]] = []
        for column in find_all_in_scope(recursive_select, exp.Column):
            if column.table == relation.name:
                if column.name not in name_to_index:
                    raise OptimizeError(
                        f"Unknown recursive column '{column.name}' for relation '{relation.name}'."
                    )
                own_indexes.append(name_to_index[column.name])
            else:
                outside_keys.append((column.table, column.name))
        recursive_inputs.append(list(dict.fromkeys(own_indexes)))
        recursive_external.append(outside_keys)

    predicate_recursive, condition_external = _predicate_dependencies(relation, name_to_index)

    flow = RecursiveFlow(
        relation=relation,
        anchor_inputs=anchor_inputs,
        recursive_inputs=recursive_inputs,
        recursive_external=recursive_external,
        external=external,
        predicate_recursive=frozenset(predicate_recursive),
        condition_external=frozenset(condition_external),
    )
    relation._flow = flow
    return flow


def _predicate_tables(scope_expression: exp.Expr) -> set[str]:
    """Source aliases that carry a WHERE clause or an INNER JOIN ON clause."""
    tables: set[str] = set()
    where = scope_expression.args.get("where")
    if where is not None:
        for column in find_all_in_scope(where, exp.Column):
            tables.add(column.table)
    for join in scope_expression.args.get("joins") or []:
        on = join.args.get("on")
        if on is not None:
            for column in find_all_in_scope(on, exp.Column):
                tables.add(column.table)
    return tables


def _is_inside_predicate(column: exp.Column) -> bool:
    return column.find_ancestor(exp.Where, exp.Join) is not None


def _predicate_dependencies(
    relation: RecursiveRelation, name_to_index: dict[str, int]
) -> tuple[set[int], set[tuple[str, str]]]:
    recursive_indexes: set[int] = set()
    external_keys: set[tuple[str, str]] = set()

    for scope_expression in (relation.anchor_scope.expression, relation.recursive_scope.expression):
        clauses: list[exp.Expr] = []
        where = scope_expression.args.get("where")
        if where is not None:
            clauses.append(where)
        for join in scope_expression.args.get("joins") or []:
            on = join.args.get("on")
            if on is not None:
                clauses.append(on)

        for clause in clauses:
            for column in find_all_in_scope(clause, exp.Column):
                if column.table == relation.name:
                    if column.name in name_to_index:
                        recursive_indexes.add(name_to_index[column.name])
                else:
                    external_keys.add((column.table, column.name))

    return recursive_indexes, external_keys
