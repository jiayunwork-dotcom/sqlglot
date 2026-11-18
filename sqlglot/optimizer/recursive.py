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

External sources are identified by the *resolved source*, never by the local alias used
at a reference site. A scope source is keyed by the identity of its query expression; a
base table is keyed by its catalog/database/name. The alias is only used to resolve the
reference inside its own query block. Consequently:

* the two arms of a recursive CTE may reuse one alias (``s``) for different sources
  without merging them, and a self-reference may be aliased (``FROM r AS rr``) without
  being mistaken for an external table;
* two aliases of the same CTE are one source, while two same-named CTEs in different
  nested scopes are different sources;
* an alias column list (``x AS s(p, q)``, ``r AS rr(a, b)``) is mapped positionally to
  the source's output columns before the source key is formed.

The flow is cached once per :class:`RecursiveRelation` binding (O(C + E) resident state),
so every reference to the same CTE, both arms, lineage of several outputs and repeated
projection reachability share one analysis. It is deliberately *not* shared between scope
trees built over different or rewritten ASTs; :meth:`RecursiveRelation.invalidate_flow`
drops it after in-place projection pruning.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.scope import RecursiveRelation, Scope, find_all_in_scope

if t.TYPE_CHECKING:
    from collections.abc import Iterator

SourceKey = tuple[str, ...]


@dataclass(frozen=True)
class ExternalColumn:
    """A column the recursive body reads from a source other than the relation itself.

    Attributes:
        source_key: Identity of the resolved source *and* the source's output column name.
            Query sources are keyed by their query expression identity, base tables by
            their catalog/database/name, so two local aliases with the same spelling never
            collide while different columns of one source stay distinct.
        kind: ``"query"`` (CTE / derived table / other scope), ``"table"`` (physical
            table) or ``"unresolved"`` (a correlated or unqualified reference whose source
            cannot be resolved from the arm's own scope).
        expression: The ``exp.Table`` node of a base table, the query expression of a
            scope source, or the unresolved ``exp.Column``.
        table: Reference-site alias, used for lineage node labels.
        name: Column name at the reference site.
        output_name: Column name exposed by the resolved source (after alias-column
            remapping).
        predicate_only: True when the column appears only inside WHERE / ON conditions,
            never in a value projection of either arm. It still controls which rows the
            recursion produces, but is not a value source for any output.
    """

    source_key: SourceKey
    kind: str
    expression: exp.Expr
    table: str
    name: str
    output_name: str
    predicate_only: bool

    @property
    def key(self) -> SourceKey:
        return self.source_key


@dataclass
class RecursiveFlow:
    """Finite column-level dependency summary of one :class:`RecursiveRelation`.

    All column references are positional indexes into :attr:`RecursiveRelation.columns`
    for the relation's own columns, and keys into :attr:`external` for outside sources.

    Attributes:
        relation: The binding this flow summarizes.
        anchor_inputs: For each output position, external source keys the anchor term
            reads for that projection.
        recursive_inputs: For each output position, relation column indexes the recursive
            term reads from its own previous iteration for that projection.
        recursive_external: For each output position, external keys the recursive term
            introduces in that projection.
        external: Every distinct external column referenced anywhere in the two arms,
            keyed by resolved-source identity, with predicate usage classified.
        predicate_recursive: Relation column indexes used by WHERE / ON in either arm.
        anchor_conditions: External keys used by WHERE / ON of the anchor arm.
        recursive_conditions: External keys used by WHERE / ON of the recursive arm.
        condition_external: Union of both condition sets.
    """

    relation: RecursiveRelation
    anchor_inputs: list[list[SourceKey]]
    recursive_inputs: list[list[int]]
    recursive_external: list[list[SourceKey]]
    external: dict[SourceKey, ExternalColumn]
    predicate_recursive: frozenset[int]
    anchor_conditions: frozenset[SourceKey]
    recursive_conditions: frozenset[SourceKey]
    # Reference-site ``exp.Column`` id -> classification. Consumers (projection
    # pruning, lineage) look columns up here instead of re-resolving the local alias,
    # so both consumers agree with the arm analysis that built this flow.
    anchor_column_keys: dict[int, SourceKey]
    recursive_column_keys: dict[int, SourceKey]
    anchor_self_columns: dict[int, int]
    recursive_self_columns: dict[int, int]

    @property
    def condition_external(self) -> frozenset[SourceKey]:
        """External keys used by WHERE / ON in either arm."""
        return self.anchor_conditions | self.recursive_conditions

    def column_key(self, column: exp.Column) -> SourceKey | None:
        """External source key for a column reference, or None if it is not external.

        Lookup is by the ``exp.Column`` node's identity, which is stable across the scope
        rebuilds performed over one AST, so a flow shared between analyses stays valid
        even though their ``Scope`` instances differ. A column node belongs to exactly
        one arm, so the two maps cannot disagree.
        """
        key = self.anchor_column_keys.get(id(column))
        if key is None:
            key = self.recursive_column_keys.get(id(column))
        return key

    def self_column_index(self, column: exp.Column) -> int | None:
        """Positional relation column index for a self-reference, or None."""
        index = self.anchor_self_columns.get(id(column))
        if index is None:
            index = self.recursive_self_columns.get(id(column))
        return index

    def scope_for_key(self, key: SourceKey, *scope_pools: Scope) -> Scope | None:
        """Find the Scope object whose expression is the resolved source of ``key``.

        The flow stores source identity by expression identity, which survives the
        scope rebuilds that projection pruning and lineage perform over the same AST.
        """
        if key[0] != "query":
            return None

        expression_id = int(key[1])
        for root in scope_pools:
            for scope in root.traverse():
                if id(scope.expression) == expression_id:
                    return scope
        return None

    def value_closure(
        self, outputs: t.Iterable[int]
    ) -> tuple[frozenset[int], frozenset[SourceKey]]:
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

        external: set[SourceKey] = set()
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
    external: dict[SourceKey, ExternalColumn] = {}
    value_keys: set[SourceKey] = set()
    anchor_conditions: set[SourceKey] = set()
    recursive_conditions: set[SourceKey] = set()
    predicate_recursive: set[int] = set()
    anchor_inputs: list[list[SourceKey]] = []
    recursive_inputs: list[list[int]] = []
    recursive_external: list[list[SourceKey]] = []
    anchor_column_keys: dict[int, SourceKey] = {}
    recursive_column_keys: dict[int, SourceKey] = {}
    anchor_self_columns: dict[int, int] = {}
    recursive_self_columns: dict[int, int] = {}

    def register(key: SourceKey, entry: ExternalColumn) -> None:
        existing = external.get(key)
        if existing is None:
            external[key] = entry
        elif existing.kind == "unresolved" and entry.kind != "unresolved":
            # Prefer a resolved source over a conservative placeholder.
            external[key] = entry

    def analyze_arm(arm_scope: Scope, is_recursive: bool) -> None:
        reference_nodes: dict[str, exp.Expr] = {}
        for reference_name, reference_node in arm_scope.references:
            reference_nodes.setdefault(reference_name, reference_node)

        def resolve(column: exp.Column) -> tuple[int | None, SourceKey | None]:
            alias = column.table
            source = arm_scope.sources.get(alias) if alias else None
            reference = reference_nodes.get(alias)
            alias_columns = (
                list(reference.alias_column_names) if isinstance(reference, exp.Table) else []
            )

            # Self-reference: an unqualified table whose name is the relation's own name.
            # That is exactly how the binder recognises the self-reference (a db-qualified
            # table of the same name is a real table). The alias spelling and an optional
            # alias-column list (r AS rr(a, b)) are local to this arm.
            if (
                isinstance(source, Scope)
                and isinstance(reference, exp.Table)
                and not reference.db
                and reference.name == relation.name
            ):
                index = _self_index(column, alias_columns, name_to_index, relation)
                if is_recursive:
                    recursive_self_columns[id(column)] = index
                else:
                    anchor_self_columns[id(column)] = index
                return index, None

            if isinstance(source, Scope):
                output_name = _map_output_name(column.name, alias_columns, source)
                key: SourceKey = ("query", str(id(source.expression)), output_name)
                entry = ExternalColumn(
                    source_key=key,
                    kind="query",
                    expression=source.expression,
                    table=alias,
                    name=column.name,
                    output_name=output_name,
                    predicate_only=False,
                )
            elif isinstance(source, exp.Table):
                key = (
                    "table",
                    source.catalog or "",
                    source.db or "",
                    source.name,
                    column.name,
                )
                entry = ExternalColumn(
                    source_key=key,
                    kind="table",
                    expression=source,
                    table=alias,
                    name=column.name,
                    output_name=column.name,
                    predicate_only=False,
                )
            else:
                key = ("unresolved", alias or "", column.name)
                entry = ExternalColumn(
                    source_key=key,
                    kind="unresolved",
                    expression=column,
                    table=alias,
                    name=column.name,
                    output_name=column.name,
                    predicate_only=False,
                )

            if is_recursive:
                recursive_column_keys[id(column)] = key
            else:
                anchor_column_keys[id(column)] = key

            register(key, entry)
            return None, key

        selects = anchor_selects if not is_recursive else recursive_selects
        for i, projection in enumerate(selects):
            own_indexes: list[int] = []
            outside_keys: list[SourceKey] = []
            for column in find_all_in_scope(projection, exp.Column):
                own_index, key = resolve(column)
                if own_index is not None:
                    if is_recursive:
                        own_indexes.append(own_index)
                    # A supported relation never reads itself from the anchor term; if
                    # that shape is encountered it is left unsupported by the binder, so
                    # the anchor contribution is simply not a recursive value edge.
                elif key is not None:
                    outside_keys.append(key)
                    value_keys.add(key)

            if is_recursive:
                recursive_inputs.append(list(dict.fromkeys(own_indexes)))
                recursive_external.append(list(dict.fromkeys(outside_keys)))
            else:
                anchor_inputs.append(list(dict.fromkeys(outside_keys)))

        condition_keys = anchor_conditions if not is_recursive else recursive_conditions
        clauses: list[exp.Expr] = []
        where = arm_scope.expression.args.get("where")
        if where is not None:
            clauses.append(where)
        for join in arm_scope.expression.args.get("joins") or []:
            on = join.args.get("on")
            if on is not None:
                clauses.append(on)

        for clause in clauses:
            for column in find_all_in_scope(clause, exp.Column):
                own_index, key = resolve(column)
                if own_index is not None:
                    predicate_recursive.add(own_index)
                elif key is not None:
                    condition_keys.add(key)

    analyze_arm(relation.anchor_scope, is_recursive=False)
    analyze_arm(relation.recursive_scope, is_recursive=True)

    # A column used only in WHERE / ON controls rows but is not a value source. A column
    # that also appears in a value projection remains a value source.
    conditions = anchor_conditions | recursive_conditions
    for key in conditions:
        if key not in value_keys:
            entry = external[key]
            external[key] = ExternalColumn(
                source_key=entry.source_key,
                kind=entry.kind,
                expression=entry.expression,
                table=entry.table,
                name=entry.name,
                output_name=entry.output_name,
                predicate_only=True,
            )

    flow = RecursiveFlow(
        relation=relation,
        anchor_inputs=anchor_inputs,
        recursive_inputs=recursive_inputs,
        recursive_external=recursive_external,
        external=external,
        predicate_recursive=frozenset(predicate_recursive),
        anchor_conditions=frozenset(anchor_conditions),
        recursive_conditions=frozenset(recursive_conditions),
        anchor_column_keys=anchor_column_keys,
        recursive_column_keys=recursive_column_keys,
        anchor_self_columns=anchor_self_columns,
        recursive_self_columns=recursive_self_columns,
    )
    relation._flow = flow
    return flow


def _self_index(
    column: exp.Column,
    alias_columns: list[str],
    name_to_index: dict[str, int],
    relation: RecursiveRelation,
) -> int:
    """Positional index of a recursive input column, honouring the reference alias list."""
    width = relation.width
    if alias_columns:
        try:
            position = alias_columns.index(column.name)
        except ValueError:
            position = -1
        if 0 <= position < width:
            return position
    index = name_to_index.get(column.name)
    if index is None:
        raise OptimizeError(
            f"Unknown recursive column '{column.name}' for relation '{relation.name}'."
        )
    return index


def _map_output_name(name: str, alias_columns: list[str], source: Scope) -> str:
    """Map a reference-site column name to the resolved source's output column.

    A same-named source alias may carry a positional alias list, e.g.
    ``JOIN (SELECT ...) AS s(p, q)``; the source's real output for reference ``s.p`` is
    its first projection, not ``p`` itself. Consumers need that real name to select the
    right output of the source scope.
    """
    outputs = list(getattr(source.expression, "named_selects", []) or [])
    if alias_columns and name in alias_columns:
        index = alias_columns.index(name)
        if 0 <= index < len(outputs):
            return outputs[index]

    return name
