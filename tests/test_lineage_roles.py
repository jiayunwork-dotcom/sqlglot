from __future__ import annotations

import time
import unittest

import sqlglot
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.lineage import LineageRoles, lineage_roles
from sqlglot.schema import MappingSchema

SCHEMA = {
    "orders": {
        "id": "INT",
        "amount": "DECIMAL",
        "region": "VARCHAR",
        "cust_id": "INT",
    },
    "payments": {"order_id": "INT", "ok": "INT", "secret": "VARCHAR"},
    "customers": {"id": "INT", "name": "VARCHAR", "city": "VARCHAR"},
    "t": {"a": "INT", "b": "INT", "c": "INT", "region": "VARCHAR"},
    "u": {"a": "INT", "d": "INT"},
}


def dep(catalog: str, db: str, table: str, column: str) -> tuple[str, str, str, str]:
    return catalog, db, table, column


def sets(roles: LineageRoles) -> tuple[set, set]:
    return set(roles.value), set(roles.rows)


class TestLineageRoles(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        sqlglot.schema = MappingSchema()

    def assertRoles(self, result, name, value, rows):
        self.assertIn(name, result)
        actual_value, actual_rows = sets(result[name])
        self.assertEqual(actual_value, set(value))
        self.assertEqual(actual_rows, set(rows))

    # ---------------------------------------------------------------- basics

    def test_simple_filter(self):
        result = lineage_roles("SELECT amount FROM orders WHERE region=1", SCHEMA)
        self.assertEqual(list(result), ["amount"])
        self.assertRoles(
            result,
            "amount",
            value=[dep("", "", "orders", "amount")],
            rows=[dep("", "", "orders", "*"), dep("", "", "orders", "region")],
        )

    def test_value_and_rows_are_frozensets_of_strings(self):
        result = lineage_roles("SELECT amount AS amount FROM orders", SCHEMA)
        roles = result["amount"]
        self.assertIsInstance(roles, LineageRoles)
        self.assertIsInstance(roles.value, frozenset)
        self.assertIsInstance(roles.rows, frozenset)
        marker = next(iter(roles.rows))
        self.assertEqual(len(marker), 4)
        self.assertTrue(all(isinstance(part, str) for part in marker))

    def test_output_order_matches_projection(self):
        result = lineage_roles("SELECT c AS z, a AS x, b AS y FROM t", SCHEMA)
        self.assertEqual(list(result), ["z", "x", "y"])

    def test_constant_has_empty_value(self):
        result = lineage_roles(
            "SELECT 1 AS one, 'x' AS s, NULL AS n, a AS a FROM t", SCHEMA
        )
        self.assertEqual(result["one"].value, frozenset())
        self.assertEqual(result["s"].value, frozenset())
        self.assertEqual(result["n"].value, frozenset())
        # constants still belong to a query whose rows come from the FROM source
        self.assertEqual(result["one"].rows, frozenset([dep("", "", "t", "*")]))
        self.assertEqual(
            result["a"].value, frozenset([dep("", "", "t", "a")])
        )

    def test_no_from_select(self):
        result = lineage_roles("SELECT 1 AS one", {})
        self.assertEqual(result["one"].value, frozenset())
        self.assertEqual(result["one"].rows, frozenset())

    def test_scalar_operators_and_case_include_all_arguments(self):
        result = lineage_roles(
            "SELECT CASE WHEN a IS NULL THEN b WHEN a = 0 THEN c ELSE a END AS x, "
            "COALESCE(a, b) AS y FROM t",
            SCHEMA,
        )
        self.assertEqual(
            result["x"].value,
            frozenset(
                dep("", "", "t", c) for c in ("a", "b", "c")
            ),
        )
        self.assertEqual(
            result["y"].value,
            frozenset([dep("", "", "t", "a"), dep("", "", "t", "b")]),
        )

    def test_column_can_be_in_both_roles(self):
        result = lineage_roles("SELECT a AS a FROM t WHERE a > 1", SCHEMA)
        marker = dep("", "", "t", "a")
        self.assertIn(marker, result["a"].value)
        self.assertIn(marker, result["a"].rows)
        self.assertIn(dep("", "", "t", "*"), result["a"].rows)

    def test_structural_dependencies_ignore_constant_folding(self):
        result = lineage_roles(
            "SELECT a AS a FROM t WHERE a = a AND 1 = 1", SCHEMA
        )
        self.assertIn(dep("", "", "t", "a"), result["a"].rows)

    # ---------------------------------------------------------------- joins

    def test_inner_join_on(self):
        result = lineage_roles(
            "SELECT o.amount AS amount FROM orders o "
            "JOIN payments p ON o.id = p.order_id WHERE o.region = 'x'",
            SCHEMA,
        )
        self.assertRoles(
            result,
            "amount",
            value=[dep("", "", "orders", "amount")],
            rows=[
                dep("", "", "orders", "*"),
                dep("", "", "payments", "*"),
                dep("", "", "orders", "id"),
                dep("", "", "payments", "order_id"),
                dep("", "", "orders", "region"),
            ],
        )

    def test_using_includes_both_sides(self):
        result = lineage_roles(
            "SELECT o.id AS id, c.name AS name FROM orders o "
            "JOIN customers c USING (id)",
            SCHEMA,
        )
        for name in ("id", "name"):
            rows = result[name].rows
            self.assertIn(dep("", "", "orders", "id"), rows)
            self.assertIn(dep("", "", "customers", "id"), rows)
            self.assertIn(dep("", "", "orders", "*"), rows)
            self.assertIn(dep("", "", "customers", "*"), rows)

    def test_left_and_cross_join(self):
        left = lineage_roles(
            "SELECT t.a AS a FROM t LEFT JOIN u ON t.a = u.a", SCHEMA
        )
        self.assertEqual(
            left["a"].rows,
            frozenset(
                [
                    dep("", "", "t", "*"),
                    dep("", "", "u", "*"),
                    dep("", "", "t", "a"),
                    dep("", "", "u", "a"),
                ]
            ),
        )

        cross = lineage_roles(
            "SELECT t.a AS a, u.d AS d FROM t CROSS JOIN u", SCHEMA
        )
        self.assertEqual(
            cross["a"].rows,
            frozenset([dep("", "", "t", "*"), dep("", "", "u", "*")]),
        )

        comma = lineage_roles("SELECT t.a AS a FROM t, u", SCHEMA)
        self.assertEqual(
            comma["a"].rows,
            frozenset([dep("", "", "t", "*"), dep("", "", "u", "*")]),
        )

    def test_self_join_distinguishes_aliases(self):
        result = lineage_roles(
            "SELECT t1.a AS a1, t2.b AS a2 FROM t t1 JOIN t t2 ON t1.a = t2.b",
            SCHEMA,
        )
        self.assertEqual(
            result["a1"].value, frozenset([dep("", "", "t", "a")])
        )
        self.assertEqual(
            result["a2"].value, frozenset([dep("", "", "t", "b")])
        )
        rows = result["a1"].rows
        self.assertIn(dep("", "", "t", "a"), rows)
        self.assertIn(dep("", "", "t", "b"), rows)

    # ---------------------------------------------------------------- grouping

    def test_group_by_and_having(self):
        result = lineage_roles(
            "SELECT region AS region, SUM(amount) AS total FROM orders "
            "GROUP BY region HAVING SUM(amount) > 10",
            SCHEMA,
        )
        self.assertEqual(
            result["region"].value,
            frozenset([dep("", "", "orders", "region")]),
        )
        self.assertEqual(
            result["total"].value,
            frozenset([dep("", "", "orders", "amount")]),
        )
        for roles in result.values():
            self.assertEqual(
                roles.rows,
                frozenset(
                    [
                        dep("", "", "orders", "*"),
                        dep("", "", "orders", "region"),
                        dep("", "", "orders", "amount"),
                    ]
                ),
            )

    def test_group_only_key_used_for_grouping_enters_rows(self):
        result = lineage_roles(
            "SELECT SUM(a) AS s FROM t GROUP BY region", SCHEMA
        )
        self.assertEqual(
            result["s"].value, frozenset([dep("", "", "t", "a")])
        )
        self.assertEqual(
            result["s"].rows,
            frozenset([dep("", "", "t", "*"), dep("", "", "t", "region")]),
        )

    def test_having_only_column_enters_rows(self):
        result = lineage_roles(
            "SELECT a AS a, SUM(b) AS s FROM t GROUP BY a, region "
            "HAVING region > 'x'",
            SCHEMA,
        )
        for roles in result.values():
            self.assertIn(dep("", "", "t", "region"), roles.rows)

    def test_group_by_expression_and_ordinal(self):
        expr_result = lineage_roles(
            "SELECT SUM(c) AS s FROM t GROUP BY a + b", SCHEMA
        )
        self.assertEqual(
            expr_result["s"].value, frozenset([dep("", "", "t", "c")])
        )
        self.assertEqual(
            expr_result["s"].rows,
            frozenset(
                [dep("", "", "t", "*"), dep("", "", "t", "a"), dep("", "", "t", "b")]
            ),
        )

        ordinal_result = lineage_roles(
            "SELECT b AS b, SUM(a) AS s FROM t GROUP BY 1", SCHEMA
        )
        self.assertEqual(
            ordinal_result["s"].rows,
            frozenset([dep("", "", "t", "*"), dep("", "", "t", "b")]),
        )

    def test_aggregates(self):
        result = lineage_roles(
            "SELECT COUNT(a) AS c, SUM(a) AS s, MIN(b) AS mn, MAX(c) AS mx, "
            "AVG(a) AS av, COUNT(DISTINCT a) AS cd FROM t",
            SCHEMA,
        )
        self.assertEqual(result["c"].value, frozenset([dep("", "", "t", "a")]))
        self.assertEqual(result["s"].value, frozenset([dep("", "", "t", "a")]))
        self.assertEqual(result["mn"].value, frozenset([dep("", "", "t", "b")]))
        self.assertEqual(result["mx"].value, frozenset([dep("", "", "t", "c")]))
        self.assertEqual(result["av"].value, frozenset([dep("", "", "t", "a")]))
        self.assertEqual(result["cd"].value, frozenset([dep("", "", "t", "a")]))

    def test_count_star_uses_row_set_markers_as_value(self):
        result = lineage_roles("SELECT COUNT(*) AS cnt FROM t", SCHEMA)
        self.assertEqual(
            result["cnt"].value, frozenset([dep("", "", "t", "*")])
        )
        self.assertEqual(
            result["cnt"].rows, frozenset([dep("", "", "t", "*")])
        )

    def test_count_star_with_join_marks_every_source(self):
        result = lineage_roles(
            "SELECT COUNT(*) AS cnt FROM t JOIN u ON t.a = u.a", SCHEMA
        )
        self.assertEqual(
            result["cnt"].value,
            frozenset(
                [
                    dep("", "", "t", "*"),
                    dep("", "", "u", "*"),
                    dep("", "", "t", "a"),
                    dep("", "", "u", "a"),
                ]
            ),
        )

    def test_count_literal_has_empty_value(self):
        result = lineage_roles("SELECT COUNT(1) AS c FROM t", SCHEMA)
        self.assertEqual(result["c"].value, frozenset())
        self.assertEqual(result["c"].rows, frozenset([dep("", "", "t", "*")]))

    # ---------------------------------------------------------------- exists

    def test_exists_spec_example(self):
        result = lineage_roles(
            "SELECT o.amount AS amount FROM orders o WHERE EXISTS "
            "(SELECT p.secret FROM payments p WHERE p.order_id=o.id AND p.ok=1)",
            SCHEMA,
        )
        self.assertRoles(
            result,
            "amount",
            value=[dep("", "", "orders", "amount")],
            rows=[
                dep("", "", "orders", "*"),
                dep("", "", "payments", "*"),
                dep("", "", "orders", "id"),
                dep("", "", "payments", "order_id"),
                dep("", "", "payments", "ok"),
            ],
        )
        # the projected-but-unused payments.secret must not appear anywhere
        all_deps = result["amount"].value | result["amount"].rows
        self.assertNotIn(dep("", "", "payments", "secret"), all_deps)

    def test_not_exists(self):
        result = lineage_roles(
            "SELECT amount AS amount FROM orders WHERE NOT EXISTS "
            "(SELECT 1 FROM payments WHERE payments.order_id = orders.id)",
            SCHEMA,
        )
        self.assertEqual(
            result["amount"].value,
            frozenset([dep("", "", "orders", "amount")]),
        )
        self.assertEqual(
            result["amount"].rows,
            frozenset(
                [
                    dep("", "", "orders", "*"),
                    dep("", "", "payments", "*"),
                    dep("", "", "orders", "id"),
                    dep("", "", "payments", "order_id"),
                ]
            ),
        )

    def test_nested_exists_with_correlation_and_shadowing(self):
        result = lineage_roles(
            "SELECT o.amount AS amount FROM orders o WHERE EXISTS "
            "(SELECT 1 FROM payments p WHERE p.order_id=o.id AND EXISTS "
            "(SELECT 1 FROM customers c WHERE c.id = o.cust_id AND c.city = p.secret))",
            SCHEMA,
        )
        self.assertEqual(
            result["amount"].value,
            frozenset([dep("", "", "orders", "amount")]),
        )
        self.assertEqual(
            result["amount"].rows,
            frozenset(
                [
                    dep("", "", "orders", "*"),
                    dep("", "", "payments", "*"),
                    dep("", "", "customers", "*"),
                    dep("", "", "orders", "id"),
                    dep("", "", "payments", "order_id"),
                    dep("", "", "orders", "cust_id"),
                    dep("", "", "customers", "id"),
                    dep("", "", "customers", "city"),
                    dep("", "", "payments", "secret"),
                ]
            ),
        )

    def test_exists_in_join_on(self):
        result = lineage_roles(
            "SELECT t.a AS a FROM t JOIN u ON t.a = u.a AND EXISTS "
            "(SELECT 1 FROM orders WHERE orders.id = u.d)",
            SCHEMA,
        )
        rows = result["a"].rows
        self.assertIn(dep("", "", "orders", "*"), rows)
        self.assertIn(dep("", "", "orders", "id"), rows)
        self.assertIn(dep("", "", "u", "d"), rows)

    def test_unused_cte_inside_exists_excluded(self):
        result = lineage_roles(
            "SELECT t.a AS a FROM t WHERE EXISTS "
            "(WITH g AS (SELECT a FROM u) SELECT 1 FROM t t2 WHERE t2.a = t.a)",
            SCHEMA,
        )
        rows = result["a"].rows
        self.assertNotIn(dep("", "", "u", "*"), rows)
        self.assertIn(dep("", "", "t", "*"), rows)
        self.assertIn(dep("", "", "t", "a"), rows)

    def test_exists_inside_having(self):
        result = lineage_roles(
            "SELECT b AS b, SUM(a) AS s FROM t GROUP BY b "
            "HAVING EXISTS (SELECT 1 FROM u WHERE u.a = t.b)",
            SCHEMA,
        )
        for roles in result.values():
            self.assertIn(dep("", "", "u", "*"), roles.rows)
            self.assertIn(dep("", "", "u", "a"), roles.rows)

    # ---------------------------------------------------------------- cte / derived

    def test_derived_table_filter_propagates_as_rows(self):
        result = lineage_roles(
            "SELECT t.amount AS amount FROM "
            "(SELECT amount FROM orders WHERE region = 1) AS t",
            SCHEMA,
        )
        self.assertEqual(
            result["amount"].value,
            frozenset([dep("", "", "orders", "amount")]),
        )
        self.assertEqual(
            result["amount"].rows,
            frozenset(
                [dep("", "", "orders", "*"), dep("", "", "orders", "region")]
            ),
        )

    def test_intermediate_column_switches_role(self):
        # inside the CTE region filters rows; outside it filters on the intermediate
        # region column, so the physical region column stays a rows dependency while
        # amount remains a pure value dependency
        result = lineage_roles(
            "WITH x AS (SELECT amount, region FROM orders) "
            "SELECT x.amount AS amount FROM x "
            "JOIN customers c ON c.id = x.region WHERE x.region = 'z'",
            SCHEMA,
        )
        roles = result["amount"]
        self.assertEqual(
            roles.value, frozenset([dep("", "", "orders", "amount")])
        )
        self.assertIn(dep("", "", "orders", "region"), roles.rows)
        self.assertIn(dep("", "", "orders", "*"), roles.rows)
        self.assertIn(dep("", "", "customers", "*"), roles.rows)
        self.assertIn(dep("", "", "customers", "id"), roles.rows)

    def test_unused_derived_columns_are_excluded(self):
        result = lineage_roles(
            "SELECT q.a AS a FROM (SELECT a, b, c FROM t) AS q", SCHEMA
        )
        self.assertEqual(
            result["a"].value, frozenset([dep("", "", "t", "a")])
        )
        self.assertNotIn(dep("", "", "t", "b"), result["a"].value)
        self.assertNotIn(dep("", "", "t", "c"), result["a"].value)

    def test_unused_cte_does_not_leak_sources(self):
        result = lineage_roles(
            "WITH unused AS (SELECT a FROM u) SELECT b AS b FROM t", SCHEMA
        )
        self.assertEqual(result["b"].rows, frozenset([dep("", "", "t", "*")]))

    def test_alias_shadowing_across_levels(self):
        result = lineage_roles(
            "SELECT t.a AS a FROM (SELECT a FROM t WHERE b = 1) AS t "
            "JOIN u ON t.a = u.a",
            SCHEMA,
        )
        rows = result["a"].rows
        self.assertIn(dep("", "", "t", "b"), rows)
        self.assertIn(dep("", "", "u", "a"), rows)

    def test_cte_reused_twice(self):
        result = lineage_roles(
            "WITH s AS (SELECT a, b FROM t WHERE region = 1) "
            "SELECT s1.a AS a1, s2.b AS b2 FROM s s1 JOIN s s2 ON s1.a = s2.a",
            SCHEMA,
        )
        for roles in result.values():
            self.assertIn(dep("", "", "t", "region"), roles.rows)
            # a set contains the base row marker exactly once despite the double use
            self.assertEqual(
                roles.rows,
                frozenset(
                    [
                        dep("", "", "t", "*"),
                        dep("", "", "t", "a"),
                        dep("", "", "t", "region"),
                    ]
                ),
            )

    def test_multi_level_cte_filters_accumulate(self):
        result = lineage_roles(
            "WITH c1 AS (SELECT a, b FROM t WHERE region = 1), "
            "c2 AS (SELECT a FROM c1 WHERE b > 2), "
            "c3 AS (SELECT a FROM c2 WHERE a < 9) "
            "SELECT c3.a AS a FROM c3",
            SCHEMA,
        )
        self.assertEqual(
            result["a"].value, frozenset([dep("", "", "t", "a")])
        )
        self.assertEqual(
            result["a"].rows,
            frozenset(
                [
                    dep("", "", "t", "*"),
                    dep("", "", "t", "region"),
                    dep("", "", "t", "b"),
                    dep("", "", "t", "a"),
                ]
            ),
        )

    def test_derived_column_list_alias(self):
        result = lineage_roles(
            "SELECT q.p AS p FROM (SELECT a, b FROM t) AS q(p, qq) WHERE q.qq = 1",
            SCHEMA,
        )
        self.assertEqual(
            result["p"].value, frozenset([dep("", "", "t", "a")])
        )
        self.assertEqual(
            result["p"].rows,
            frozenset([dep("", "", "t", "*"), dep("", "", "t", "b")]),
        )

    # ---------------------------------------------------------------- set ops

    def test_union_all_merges_by_position(self):
        result = lineage_roles(
            "SELECT a AS x, b AS y FROM t UNION ALL SELECT a, d FROM u", SCHEMA
        )
        self.assertEqual(list(result), ["x", "y"])
        self.assertEqual(
            result["x"].value,
            frozenset([dep("", "", "t", "a"), dep("", "", "u", "a")]),
        )
        self.assertEqual(
            result["y"].value,
            frozenset([dep("", "", "t", "b"), dep("", "", "u", "d")]),
        )
        self.assertEqual(
            result["x"].rows,
            frozenset([dep("", "", "t", "*"), dep("", "", "u", "*")]),
        )

    def test_union_all_branch_filters_not_crossed(self):
        result = lineage_roles(
            "SELECT a AS x FROM t WHERE b > 1 "
            "UNION ALL SELECT a FROM u WHERE d < 2",
            SCHEMA,
        )
        self.assertEqual(
            result["x"].rows,
            frozenset(
                [
                    dep("", "", "t", "*"),
                    dep("", "", "t", "b"),
                    dep("", "", "u", "*"),
                    dep("", "", "u", "d"),
                ]
            ),
        )

    def test_union_output_name_from_first_branch(self):
        result = lineage_roles(
            "SELECT a AS first_name FROM t UNION ALL SELECT a FROM u", SCHEMA
        )
        self.assertEqual(list(result), ["first_name"])

    def test_union_branch_arity_mismatch_rejected(self):
        with self.assertRaises(OptimizeError):
            lineage_roles(
                "SELECT a, b FROM t UNION ALL SELECT a FROM u", SCHEMA
            )

    def test_count_star_inside_union(self):
        result = lineage_roles(
            "SELECT COUNT(*) AS c FROM t UNION ALL SELECT COUNT(*) FROM u",
            SCHEMA,
        )
        self.assertEqual(
            result["c"].value,
            frozenset([dep("", "", "t", "*"), dep("", "", "u", "*")]),
        )

    def test_set_operation_limit_offset_rejected(self):
        for sql in (
            "SELECT a AS x FROM t UNION ALL SELECT a FROM u LIMIT 5",
            "SELECT a AS x FROM t UNION ALL SELECT a FROM u OFFSET 2",
        ):
            with self.subTest(sql=sql):
                with self.assertRaises(OptimizeError):
                    lineage_roles(sql, SCHEMA)

    def test_set_operation_top_order_by_without_limit_allowed(self):
        result = lineage_roles(
            "SELECT a AS x FROM t UNION ALL SELECT a FROM u ORDER BY x", SCHEMA
        )
        self.assertEqual(list(result), ["x"])

    # ---------------------------------------------------------------- star / distinct / order

    def test_star_expands_per_schema(self):
        result = lineage_roles("SELECT * FROM t", SCHEMA)
        self.assertEqual(set(result), {"a", "b", "c", "region"})
        for name, column in zip(result, ("a", "b", "c", "region")):
            self.assertEqual(
                result[name].value,
                frozenset([dep("", "", "t", column)]),
            )

    def test_star_without_schema_rejected(self):
        with self.assertRaises(OptimizeError):
            lineage_roles("SELECT * FROM t", {})

    def test_distinct_does_not_add_dependencies(self):
        result = lineage_roles(
            "SELECT DISTINCT a AS a, b AS b FROM t", SCHEMA
        )
        self.assertEqual(
            result["a"].value, frozenset([dep("", "", "t", "a")])
        )
        self.assertEqual(result["a"].rows, frozenset([dep("", "", "t", "*")]))

    def test_top_level_order_by_without_limit_ignored(self):
        result = lineage_roles(
            "SELECT a AS a FROM t ORDER BY b, c", SCHEMA
        )
        self.assertEqual(result["a"].rows, frozenset([dep("", "", "t", "*")]))

    # ---------------------------------------------------------------- sources

    def test_named_sources_trace_to_base_tables(self):
        result = lineage_roles(
            "SELECT a AS a FROM z",
            schema={"x": {"a": "INT", "b": "INT"}},
            sources={
                "y": "SELECT * FROM x",
                "z": "SELECT a FROM y",
            },
        )
        self.assertEqual(
            result["a"].value, frozenset([dep("", "", "x", "a")])
        )

    def test_source_internal_filter_is_rows(self):
        result = lineage_roles(
            "SELECT a AS a FROM z",
            schema={"x": {"a": "INT", "b": "INT"}},
            sources={"z": "SELECT a FROM x WHERE b > 1"},
        )
        self.assertEqual(
            result["a"].rows,
            frozenset([dep("", "", "x", "*"), dep("", "", "x", "b")]),
        )

    def test_cyclic_sources_rejected(self):
        with self.assertRaises(OptimizeError):
            lineage_roles(
                "SELECT a AS a FROM x",
                schema={"x": {"a": "INT"}},
                sources={
                    "x": "SELECT a FROM y",
                    "y": "SELECT a FROM x",
                },
            )

    def test_three_node_cycle_rejected(self):
        with self.assertRaises(OptimizeError):
            lineage_roles(
                "SELECT a AS a FROM a1",
                schema={"x": {"a": "INT"}},
                sources={
                    "a1": "SELECT a FROM a2",
                    "a2": "SELECT a FROM a3",
                    "a3": "SELECT a FROM a1",
                },
            )

    def test_non_query_source_rejected(self):
        with self.assertRaises(OptimizeError):
            lineage_roles(
                "SELECT a AS a FROM z",
                schema={"t": {"a": "INT"}},
                sources={"z": "INSERT INTO t VALUES (1)"},
            )

    # ---------------------------------------------------------------- dialects

    def test_postgres_quoted_identifier_case_preserved(self):
        schema = MappingSchema(
            {"MixedTable": {"MixedCol": "INT", "other": "INT"}},
            dialect="postgres",
            normalize=False,
        )
        result = lineage_roles(
            'SELECT "MixedCol" AS v FROM "MixedTable" WHERE "other" = 1',
            schema,
            dialect="postgres",
        )
        self.assertEqual(
            result["v"].value,
            frozenset([dep("", "", "MixedTable", "MixedCol")]),
        )
        self.assertIn(dep("", "", "MixedTable", "other"), result["v"].rows)

    def test_duckdb_unquoted_identifiers_fold_lowercase(self):
        result = lineage_roles(
            "SELECT MixedCol AS v FROM MixedTable",
            {"mixedtable": {"mixedcol": "INT"}},
            dialect="duckdb",
        )
        self.assertEqual(
            result["v"].value,
            frozenset([dep("", "", "mixedtable", "mixedcol")]),
        )

    def test_postgres_full_query_end_to_end(self):
        schema = {
            "orders": {"id": "INT", "amount": "DECIMAL", "region": "VARCHAR"},
            "payments": {"order_id": "INT", "ok": "INT"},
        }
        result = lineage_roles(
            "SELECT o.region AS region, SUM(o.amount) AS total "
            "FROM orders o JOIN payments p ON o.id = p.order_id "
            "WHERE p.ok = 1 GROUP BY o.region HAVING SUM(o.amount) > 10",
            schema,
            dialect="postgres",
        )
        self.assertEqual(
            result["region"].value,
            frozenset([dep("", "", "orders", "region")]),
        )
        total_rows = result["total"].rows
        self.assertIn(dep("", "", "orders", "id"), total_rows)
        self.assertIn(dep("", "", "payments", "order_id"), total_rows)
        self.assertIn(dep("", "", "payments", "ok"), total_rows)
        self.assertIn(dep("", "", "orders", "amount"), total_rows)

    def test_postgres_unquoted_identifiers_fold_lowercase(self):
        result = lineage_roles(
            "SELECT MixedCol AS v FROM MixedTable",
            {"mixedtable": {"mixedcol": "INT"}},
            dialect="postgres",
        )
        self.assertEqual(
            result["v"].value,
            frozenset([dep("", "", "mixedtable", "mixedcol")]),
        )

    def test_catalog_db_qualified_names(self):
        result = lineage_roles(
            "SELECT x AS v FROM cat.db.tbl WHERE y = 1",
            {"cat": {"db": {"tbl": {"x": "INT", "y": "INT"}}}},
            dialect="duckdb",
        )
        self.assertEqual(
            result["v"].value, frozenset([dep("cat", "db", "tbl", "x")])
        )
        self.assertIn(dep("cat", "db", "tbl", "*"), result["v"].rows)
        self.assertIn(dep("cat", "db", "tbl", "y"), result["v"].rows)

    # ---------------------------------------------------------------- immutability

    def test_input_ast_not_modified(self):
        sql = "SELECT amount AS amount FROM orders WHERE region = 1"
        ast = sqlglot.parse_one(sql)
        lineage_roles(ast, SCHEMA)
        self.assertEqual(ast.sql(), sql)

    def test_dict_schema_not_modified(self):
        schema = {"orders": {"amount": "INT", "region": "VARCHAR"}}
        import copy

        before = copy.deepcopy(schema)
        lineage_roles(
            "SELECT amount AS amount FROM orders WHERE region = 1", schema
        )
        self.assertEqual(schema, before)

    def test_mapping_schema_not_modified(self):
        schema = MappingSchema(
            {"orders": {"amount": "INT", "region": "VARCHAR"}}
        )
        before = dict(schema.mapping)
        lineage_roles(
            "SELECT amount AS amount FROM orders WHERE region = 1", schema
        )
        self.assertEqual(schema.mapping, before)

    def test_sources_not_modified(self):
        source_sql = "SELECT amount FROM orders WHERE region = 1"
        source = sqlglot.parse_one(source_sql)
        lineage_roles(
            "SELECT amount AS amount FROM z",
            SCHEMA,
            sources={"z": source},
        )
        self.assertEqual(source.sql(), source_sql)

    def test_repeatable_and_interleaved_calls(self):
        sql = "SELECT amount AS amount FROM orders WHERE region = 1"
        first = lineage_roles(sql, SCHEMA)
        other = lineage_roles("SELECT id AS id FROM orders", SCHEMA)
        again = lineage_roles(sql, SCHEMA)
        self.assertEqual(
            {k: (v.value, v.rows) for k, v in first.items()},
            {k: (v.value, v.rows) for k, v in again.items()},
        )
        self.assertNotEqual(set(first), set(other))

    def test_failure_leaves_no_state(self):
        sql = "SELECT amount AS amount FROM orders WHERE region = 1"
        with self.assertRaises(OptimizeError):
            lineage_roles("SELECT nope FROM orders", SCHEMA)
        after = lineage_roles(sql, SCHEMA)
        self.assertEqual(
            after["amount"].value,
            frozenset([dep("", "", "orders", "amount")]),
        )

    def test_alias_rewrite_keeps_physical_sets(self):
        first = lineage_roles(
            "SELECT a AS x, b AS y FROM t WHERE c = 1", SCHEMA
        )
        second = lineage_roles(
            "SELECT a AS p, b AS q FROM t WHERE c = 1", SCHEMA
        )
        self.assertEqual(
            [tuple(sets(r)) for r in first.values()],
            [tuple(sets(r)) for r in second.values()],
        )

    def test_adding_unrelated_projection_keeps_sets(self):
        base = lineage_roles(
            "SELECT a AS x, b AS y FROM t WHERE c = 1", SCHEMA
        )
        extended = lineage_roles(
            "SELECT a AS x, b AS y, c AS z FROM t WHERE c = 1", SCHEMA
        )
        for name in ("x", "y"):
            self.assertEqual(base[name].value, extended[name].value)
            self.assertEqual(base[name].rows, extended[name].rows)

    def test_adding_unused_cte_keeps_sets(self):
        base = lineage_roles(
            "SELECT a AS x, b AS y FROM t WHERE c = 1", SCHEMA
        )
        extended = lineage_roles(
            "WITH ghost AS (SELECT region FROM t) "
            "SELECT a AS x, b AS y FROM t WHERE c = 1",
            SCHEMA,
        )
        for name in ("x", "y"):
            self.assertEqual(base[name].value, extended[name].value)
            self.assertEqual(base[name].rows, extended[name].rows)

    # ---------------------------------------------------------------- errors

    def _assert_optimize_error(self, sql, **kw):
        with self.assertRaises(OptimizeError):
            lineage_roles(sql, SCHEMA, **kw)

    def test_unnamed_complex_expression_rejected(self):
        self._assert_optimize_error("SELECT a + 1 FROM t")

    def test_duplicate_output_names_rejected(self):
        self._assert_optimize_error("SELECT a AS x, b AS x FROM t")

    def test_unresolved_table_and_column_rejected(self):
        self._assert_optimize_error("SELECT a AS a FROM nope")
        self._assert_optimize_error("SELECT nope AS n FROM t")

    def test_parse_error(self):
        with self.assertRaises(ParseError):
            lineage_roles("SELECT FROM FROM", SCHEMA)

    def test_non_query_rejected(self):
        self._assert_optimize_error("INSERT INTO t VALUES (1)")
        self._assert_optimize_error("CREATE TABLE x (a INT)")
        self._assert_optimize_error("UPDATE t SET a = 1")
        self._assert_optimize_error("VALUES (1), (2)")

    def test_unsupported_constructs_rejected(self):
        cases = [
            "WITH RECURSIVE r AS (SELECT a FROM t UNION ALL SELECT a FROM r) SELECT a AS a FROM r",
            "SELECT a AS a FROM t UNION SELECT a FROM u",
            "SELECT a AS a FROM t INTERSECT SELECT a FROM u",
            "SELECT a AS a FROM t EXCEPT SELECT a FROM u",
            "SELECT ROW_NUMBER() OVER (PARTITION BY a) AS rn FROM t",
            "SELECT a AS a FROM t QUALIFY a > 1",
            "SELECT a AS a FROM t LIMIT 3",
            "SELECT a AS a FROM t OFFSET 3",
            "SELECT a AS a FROM t ORDER BY a LIMIT 3",
            "SELECT * FROM t PIVOT (SUM(a) FOR b IN (1))",
            "SELECT (SELECT MAX(a) FROM t) AS m FROM u",
            "SELECT a AS a FROM t WHERE a IN (SELECT a FROM u)",
            "SELECT a AS a FROM t WHERE a > ALL (SELECT a FROM u)",
            "SELECT t.a AS a FROM t FULL JOIN u ON t.a = u.a",
            "SELECT t.a AS a FROM t RIGHT JOIN u ON t.a = u.a",
            "SELECT t.a AS a FROM t NATURAL JOIN u",
            "SELECT a AS a FROM (SELECT a FROM t ORDER BY b) AS x",
        ]
        for sql in cases:
            with self.subTest(sql=sql):
                self._assert_optimize_error(sql)

    def test_unsupported_constructs_nested_rejected(self):
        cases = [
            "SELECT a AS a FROM t WHERE EXISTS (SELECT 1 FROM u LIMIT 1)",
            "SELECT a AS a FROM t WHERE EXISTS (SELECT 1 FROM u ORDER BY d)",
            "WITH x AS (SELECT a FROM t LIMIT 3) SELECT a AS a FROM x",
            "WITH x AS (SELECT a FROM t UNION SELECT a FROM u) SELECT a AS a FROM x",
            "SELECT q.a AS a FROM (SELECT a FROM t QUALIFY a > 1) q",
        ]
        for sql in cases:
            with self.subTest(sql=sql):
                self._assert_optimize_error(sql)

    def test_lateral_and_table_functions_rejected(self):
        self._assert_optimize_error(
            "SELECT x FROM t, LATERAL UNNEST(t.b) AS x",
        )
        self._assert_optimize_error("SELECT * FROM UNNEST([1,2,3])")

    def test_semi_anti_join_rejected(self):
        for sql in (
            "SELECT t.a AS a FROM t LEFT SEMI JOIN u ON t.a = u.a",
            "SELECT t.a AS a FROM t LEFT ANTI JOIN u ON t.a = u.a",
        ):
            with self.assertRaises(OptimizeError):
                lineage_roles(sql, SCHEMA, dialect="spark")

    # ---------------------------------------------------------------- scale

    def test_100_layer_cte_chain_performance(self):
        parts = [
            "WITH c0 AS (SELECT a, b, region FROM t WHERE region = 1)"
        ]
        for i in range(1, 100):
            parts.append(
                f", c{i} AS (SELECT a, b FROM c{i - 1} WHERE b > {i})"
            )
        parts.append("SELECT c99.a AS a FROM c99 WHERE c99.a > 0")
        sql = " ".join(parts)

        start = time.perf_counter()
        result = lineage_roles(sql, SCHEMA)
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 3.0)
        self.assertEqual(
            result["a"].value, frozenset([dep("", "", "t", "a")])
        )
        self.assertIn(dep("", "", "t", "region"), result["a"].rows)

    def test_50_union_branches_sharing_cte_performance(self):
        branches = [
            f"SELECT a, b FROM shared WHERE a = {i}" for i in range(50)
        ]
        sql = (
            "WITH shared AS (SELECT a, b FROM t WHERE region = 'x') "
            + " UNION ALL ".join(branches)
        )

        start = time.perf_counter()
        result = lineage_roles(sql, SCHEMA)
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 3.0)
        self.assertEqual(len(result), 2)
        self.assertIn(dep("", "", "t", "region"), result["a"].rows)


if __name__ == "__main__":
    unittest.main()
